"""Column and table statistics for cardinality estimation.

Provides reusable statistical primitives that can be consumed by the query
planner, query plan analyzer, or any other component that needs selectivity
estimates.  Extracted from ``query_planner.py`` to reduce module coupling
and enable independent testing.

Key classes:

- :class:`ColumnStatistics` — per-column NDV, null fraction, histograms.
- :class:`TableStatistics` — lazily computed column statistics for a table.
- :class:`CardinalityFeedbackStore` — accumulates actual-vs-estimated
  ratios for self-correcting estimates.

Usage::

    stats = TableStatistics(df)
    col = stats.column_stats("age")
    if col is not None:
        sel = col.equality_selectivity()  # 1/NDV adjusted for nulls
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from shared.logger import LOGGER

__all__ = [
    "CardinalityFeedbackStore",
    "ColumnStatistics",
    "TableStatistics",
]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

#: Maximum number of rows to sample when computing column statistics.
STATS_SAMPLE_SIZE: int = 10_000

#: Number of equi-width bins for histogram-based range selectivity.
HISTOGRAM_BINS: int = 64

#: Minimum non-null rows required to build a histogram.
HISTOGRAM_MIN_ROWS: int = 10

#: Seed for DuckDB's ``USING SAMPLE ... REPEATABLE (n)``.  Mirrors the
#: pandas path's ``random_state=42`` so DuckDB-computed statistics are
#: reproducible run to run, exactly as the pandas ones are.  (The two paths
#: still pick *different* rows — same table, same size, different sample —
#: so estimates agree in distribution, not value for value.)
STATS_SAMPLE_SEED: int = 42

#: DuckDB type names treated as numeric for min/max and histogram purposes.
#: Matched by prefix so parameterised types (``DECIMAL(18,3)``) are covered.
_DUCKDB_NUMERIC_PREFIXES: tuple[str, ...] = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "DECIMAL",
    "NUMERIC",
)


#: Default selectivity factor for WHERE predicates when no statistics are
#: available.  Assumes an equality filter keeps ~33% of rows.
DEFAULT_FILTER_SELECTIVITY: float = 0.33

#: Average bytes per cell for memory estimation when the actual DataFrame
#: is not available.  Conservative estimate for mixed-type columns.
AVG_BYTES_PER_CELL: int = 64

#: Maximum rolling window per entity type in the feedback store.
_MAX_HISTORY: int = 32


def _quote_ident(name: str) -> str:
    """Return *name* as a double-quoted SQL identifier."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnStatistics:
    """Statistics for a single column, used for selectivity estimation.

    Attributes:
        ndv: Number of distinct values (excluding nulls).
        null_fraction: Fraction of rows that are null (0.0-1.0).
        min_value: Minimum non-null value (numeric columns only).
        max_value: Maximum non-null value (numeric columns only).
        row_count: Total rows in the table at time of collection.
        histogram_edges: Bin edges for equi-width histogram (numeric only).
            Length is ``num_bins + 1``.
        histogram_counts: Row counts per histogram bin (numeric only).
            Length is ``num_bins``.

    """

    ndv: int
    null_fraction: float
    min_value: float | None = None
    max_value: float | None = None
    row_count: int = 0
    histogram_edges: tuple[float, ...] | None = None
    histogram_counts: tuple[int, ...] | None = None

    def equality_selectivity(self) -> float:
        """Selectivity for ``col = value``: 1/NDV, adjusted for nulls."""
        if self.ndv <= 0:
            return DEFAULT_FILTER_SELECTIVITY
        return (1.0 - self.null_fraction) / self.ndv

    def range_selectivity(
        self,
        low: float | None = None,
        high: float | None = None,
    ) -> float:
        """Selectivity for range predicates (``col > low``, ``col < high``).

        When a histogram is available, estimates selectivity by summing the
        fraction of rows in bins that overlap the query range.  Falls back
        to a uniform distribution assumption when no histogram is present.
        """
        if self.min_value is None or self.max_value is None:
            return DEFAULT_FILTER_SELECTIVITY
        span = self.max_value - self.min_value
        if span <= 0:
            return DEFAULT_FILTER_SELECTIVITY

        lo = low if low is not None else self.min_value
        hi = high if high is not None else self.max_value
        lo = max(lo, self.min_value)
        hi = min(hi, self.max_value)

        if lo >= hi:
            return 1.0 / max(self.row_count, 1)

        # Use histogram when available for more accurate estimation.
        if (
            self.histogram_edges is not None
            and self.histogram_counts is not None
            and len(self.histogram_counts) > 0
        ):
            sel = self._histogram_range_selectivity(lo, hi)
        else:
            # Uniform distribution fallback.
            sel = (hi - lo) / span

        sel *= 1.0 - self.null_fraction
        return max(sel, 1.0 / max(self.row_count, 1))

    def _histogram_range_selectivity(
        self,
        lo: float,
        hi: float,
    ) -> float:
        """Estimate range selectivity from histogram bins.

        For each bin that overlaps [lo, hi], count the proportional fraction
        of its rows that fall within the range.
        """
        assert self.histogram_edges is not None
        assert self.histogram_counts is not None

        edges = self.histogram_edges
        counts = self.histogram_counts
        total_rows = sum(counts)
        if total_rows == 0:
            return DEFAULT_FILTER_SELECTIVITY

        matching_rows = 0.0
        for i, count in enumerate(counts):
            bin_lo = edges[i]
            bin_hi = edges[i + 1]
            bin_width = bin_hi - bin_lo
            if bin_width <= 0 or count == 0:
                continue

            # Compute overlap between [lo, hi] and [bin_lo, bin_hi].
            overlap_lo = max(lo, bin_lo)
            overlap_hi = min(hi, bin_hi)
            if overlap_lo >= overlap_hi:
                continue

            # Fraction of this bin covered by the query range.
            fraction = (overlap_hi - overlap_lo) / bin_width
            matching_rows += count * fraction

        return matching_rows / total_rows


# ---------------------------------------------------------------------------
# Table statistics
# ---------------------------------------------------------------------------


class TableStatistics:
    """Collects and caches column-level statistics for an entity or
    relationship table.

    Statistics are computed lazily on first access and cached.  For large
    tables, a random sample of ``STATS_SAMPLE_SIZE`` rows is used.
    """

    def __init__(
        self,
        source_obj: pd.DataFrame | Any,
        *,
        relation: Any = None,
    ) -> None:
        """Create a statistics collector for one table.

        Args:
            source_obj: The in-memory source (pandas or Arrow).  May be
                ``None`` when *relation* is supplied.
            relation: A DuckDB relation over the same rows.  When given,
                statistics are computed in SQL over the whole (or sampled)
                relation rather than by materialising *source_obj* into
                pandas — see :meth:`_compute_column_stats_sql`.  The pandas
                path remains the fallback for anything SQL cannot answer.

        """
        self._source = source_obj
        self._relation = relation
        self._columns: dict[str, ColumnStatistics] = {}
        self._row_count: int | None = None

    @property
    def row_count(self) -> int:
        """Return the number of rows in the source table."""
        if self._row_count is None:
            if hasattr(self._source, "__len__"):
                self._row_count = len(self._source)
            elif self._relation is not None:
                self._row_count = int(
                    self._relation.aggregate("count(*)").fetchone()[0],
                )
            else:
                self._row_count = 0
        return self._row_count

    def column_stats(self, column: str) -> ColumnStatistics | None:
        """Return cached statistics for *column*, computing on first call."""
        if column in self._columns:
            return self._columns[column]
        stats = self._compute_column_stats(column)
        if stats is not None:
            self._columns[column] = stats
        return stats

    def _compute_column_stats(self, column: str) -> ColumnStatistics | None:
        """Compute statistics for a single column from the source data.

        Prefers the SQL path when a DuckDB relation is available, falling
        back to pandas on any failure.  Statistics are an optimisation input,
        never a correctness input, so a failure here degrades plan quality
        rather than breaking the query.
        """
        if self._relation is not None:
            try:
                stats = self._compute_column_stats_sql(column)
            except Exception:  # noqa: BLE001 — stats are best-effort; fall back to pandas
                LOGGER.debug(
                    "DuckDB statistics failed for column %r; "
                    "falling back to the pandas path",
                    column,
                    exc_info=True,
                )
            else:
                if stats is not None:
                    return stats
            if self._source is None:
                return None
        return self._compute_column_stats_pandas(column)

    def _compute_column_stats_sql(
        self, column: str
    ) -> ColumnStatistics | None:
        """Compute statistics for *column* inside DuckDB.

        One aggregate query answers row count, null count, NDV, and (for
        numeric columns) min/max; a second builds the histogram once the
        range is known.  Both read the same ``REPEATABLE`` sample, so the two
        queries see identical rows.

        Sampling matches the pandas path's rule — only when the table exceeds
        :data:`STATS_SAMPLE_SIZE` rows — and its intent, a bounded-cost
        estimate rather than an exact answer.

        Returns:
            The statistics, or ``None`` when *column* is not in the relation.

        """
        relation = self._relation
        if column not in relation.columns:
            return None

        col = _quote_ident(column)
        type_name = str(
            relation.types[relation.columns.index(column)],
        ).upper()
        is_numeric = type_name.startswith(_DUCKDB_NUMERIC_PREFIXES)

        if self.row_count > STATS_SAMPLE_SIZE:
            sample = (
                f"SELECT {col} AS v FROM src USING SAMPLE reservoir("
                f"{STATS_SAMPLE_SIZE} ROWS) REPEATABLE ({STATS_SAMPLE_SEED})"
            )
        else:
            sample = f"SELECT {col} AS v FROM src"

        extrema = ", min(v), max(v)" if is_numeric else ""
        row = relation.query(
            "src",
            f"WITH s AS ({sample}) "  # nosec B608 — identifier quoted by _quote_ident; constants are module-level ints
            f"SELECT count(*), count(v), count(DISTINCT v){extrema} FROM s",
        ).fetchone()
        if row is None:
            return None

        sampled, non_null, ndv = int(row[0]), int(row[1]), int(row[2])
        null_fraction = (sampled - non_null) / max(sampled, 1)
        min_val = float(row[3]) if is_numeric and row[3] is not None else None
        max_val = float(row[4]) if is_numeric and row[4] is not None else None

        hist_edges: tuple[float, ...] | None = None
        hist_counts: tuple[int, ...] | None = None
        if (
            min_val is not None
            and max_val is not None
            and min_val < max_val
            and non_null >= HISTOGRAM_MIN_ROWS
        ):
            hist_edges, hist_counts = self._histogram_sql(
                sample, min_val, max_val, min(HISTOGRAM_BINS, ndv)
            )

        return ColumnStatistics(
            ndv=ndv,
            null_fraction=null_fraction,
            min_value=min_val,
            max_value=max_val,
            row_count=self.row_count,
            histogram_edges=hist_edges,
            histogram_counts=hist_counts,
        )

    def _histogram_sql(
        self,
        sample: str,
        min_val: float,
        max_val: float,
        num_bins: int,
    ) -> tuple[tuple[float, ...] | None, tuple[int, ...] | None]:
        """Build an equi-width histogram over *sample* inside DuckDB.

        Equi-width edges (rather than ``approx_quantile``'s equi-depth ones)
        are used deliberately so
        :meth:`ColumnStatistics._histogram_range_selectivity` needs no
        changes.

        DuckDB's ``histogram(col, edges)`` returns a map keyed by each bin's
        **upper** bound, with bins half-open on the left — ``(e[i-1], e[i]]``
        — where numpy's are half-open on the right. The two therefore assign
        values landing exactly on an interior edge to different bins. That is
        immaterial here: the consumer treats bins as ranges to overlap
        proportionally, not as exact memberships. The first key counts values
        ``<= min`` (i.e. exactly the minimum), which belong in bin 0, so it is
        folded into the second to yield ``num_bins`` counts for
        ``num_bins + 1`` edges — the shape ``ColumnStatistics`` documents.

        Returns:
            ``(edges, counts)``, or ``(None, None)`` if the shape came back
            unexpected (the caller then falls back to uniform selectivity).

        """
        if num_bins < 1:
            return None, None
        step = (max_val - min_val) / num_bins
        edges = tuple(min_val + step * i for i in range(num_bins + 1))
        literal = "[" + ", ".join(repr(float(e)) for e in edges) + "]"

        row = self._relation.query(
            "src",
            f"WITH s AS ({sample}) SELECT histogram(v, {literal}) FROM s",  # nosec B608 — literal is a list of computed floats
        ).fetchone()
        if row is None or not row[0]:
            return None, None

        buckets = row[0]
        ordered = [int(buckets[k]) for k in sorted(buckets)]
        if len(ordered) != num_bins + 1:
            LOGGER.debug(
                "Unexpected histogram width %d for %d bins; "
                "falling back to uniform selectivity",
                len(ordered),
                num_bins,
            )
            return None, None
        counts = (ordered[0] + ordered[1], *ordered[2:])
        return edges, counts

    def _compute_column_stats_pandas(
        self, column: str
    ) -> ColumnStatistics | None:
        """Compute statistics for a single column from the source data."""
        try:
            if isinstance(self._source, pd.DataFrame):
                df = self._source
            elif hasattr(self._source, "to_pandas"):
                df = self._source.to_pandas()
            else:
                return None

            if column not in df.columns:
                return None

            # Sample for large tables
            n = len(df)
            if n > STATS_SAMPLE_SIZE:
                sample = df[column].sample(
                    n=STATS_SAMPLE_SIZE,
                    random_state=42,
                )
            else:
                sample = df[column]

            null_count = int(sample.isna().sum())
            null_fraction = null_count / max(len(sample), 1)
            non_null = sample.dropna()
            ndv = int(non_null.nunique())

            min_val: float | None = None
            max_val: float | None = None
            hist_edges: tuple[float, ...] | None = None
            hist_counts: tuple[int, ...] | None = None
            if len(non_null) > 0 and pd.api.types.is_numeric_dtype(non_null):
                min_val = float(non_null.min())
                max_val = float(non_null.max())
                # Build equi-width histogram for range selectivity.
                if min_val < max_val and len(non_null) >= HISTOGRAM_MIN_ROWS:
                    num_bins = min(HISTOGRAM_BINS, ndv)
                    try:
                        counts, edges = np.histogram(
                            non_null.values,
                            bins=num_bins,
                        )
                        hist_edges = tuple(float(e) for e in edges)
                        hist_counts = tuple(int(c) for c in counts)
                    except ValueError, TypeError:
                        pass  # Non-histogrammable data; use uniform fallback.

            return ColumnStatistics(
                ndv=ndv,
                null_fraction=null_fraction,
                min_value=min_val,
                max_value=max_val,
                row_count=n,
                histogram_edges=hist_edges,
                histogram_counts=hist_counts,
            )
        except (TypeError, ValueError, ArithmeticError) as _stats_exc:
            LOGGER.debug(
                "Failed to compute statistics for column %r: %s",
                column,
                _stats_exc,
                exc_info=True,
            )
            return None


# ---------------------------------------------------------------------------
# CardinalityFeedbackStore — learns from execution history
# ---------------------------------------------------------------------------


class CardinalityFeedbackStore:
    """Accumulates actual vs estimated cardinality ratios per entity type.

    After each query execution, call :meth:`record` with the entity types
    involved and the (estimated, actual) row counts.  Before a future
    estimate, call :meth:`correction_factor` to get a multiplicative
    adjustment derived from historical accuracy.

    Thread-safe via a simple lock.  History is bounded to the most recent
    ``_MAX_HISTORY`` observations per entity type.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # entity_type -> deque of (estimated, actual) tuples
        self._history: dict[str, deque[tuple[int, int]]] = {}

    def record(
        self,
        entity_type: str,
        estimated: int,
        actual: int,
    ) -> None:
        """Record an (estimated, actual) observation for *entity_type*."""
        if estimated <= 0 and actual <= 0:
            return
        with self._lock:
            if entity_type not in self._history:
                self._history[entity_type] = deque(maxlen=_MAX_HISTORY)
            self._history[entity_type].append((estimated, actual))

    def correction_factor(self, entity_type: str) -> float:
        """Return a multiplicative correction for *entity_type*.

        If the estimator consistently overestimates by 2x, this returns
        ~0.5 so the caller can multiply the heuristic estimate by it.
        Returns 1.0 when no history is available.
        """
        with self._lock:
            history = self._history.get(entity_type)
            if not history:
                return 1.0

        # Compute mean(actual / estimated) with clamp.
        ratios = [act / max(est, 1) for est, act in history]
        avg_ratio = sum(ratios) / len(ratios)
        # Clamp to [0.01, 100] to prevent runaway corrections.
        return max(0.01, min(100.0, avg_ratio))

    @property
    def entity_types_tracked(self) -> list[str]:
        """Return entity types with recorded history."""
        with self._lock:
            return list(self._history.keys())

    def clear(self) -> None:
        """Drop all recorded history."""
        with self._lock:
            self._history.clear()
