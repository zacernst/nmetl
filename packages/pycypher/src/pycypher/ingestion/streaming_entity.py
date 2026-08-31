"""Streaming entity registration — load a file source without pandas.

Open item 2 from ``docs/duckdb_eager_path_report.md``, and the piece Phase 8
of ``docs/duckdb_eager_path_design.md`` identified as missing.

``ContextBuilder.add_entity`` normally calls ``DataSource.read()``, which
materialises the whole source into an Arrow table before any query runs.
Every later phase is then working downstream of a full in-memory copy, which
is why peak RSS never fell.  This module provides the alternative: scan the
file with ``DataSource.read_relation`` and ``CREATE TABLE AS SELECT`` it
straight into DuckDB, so the rows never pass through pandas or Arrow.

Measured on a 400k x 23 parquet source, `MATCH ... WHERE ... RETURN
count` (`tests/benchmarks/bench_duckdb_streaming_source.py`)::

    eager (pandas backend)   peak_mb=793
    eager (duckdb backend)   peak_mb=918
    streaming                peak_mb=235

The normalisation `normalize_entity_table` performs in Arrow is reproduced
in SQL here, because the point is that no Arrow table is ever built:

* a named ``id_col`` is renamed to ``__ID__`` and moved to the front;
* rows are de-duplicated on ``__ID__``, keeping the first in scan order,
  matching ``_dedup_on_id``;
* with no ``id_col``, sequential ids are generated in scan order.

Entities only.  Relationship normalisation has more moving parts
(``__SOURCE__``/``__TARGET__``, parallel-edge collapsing) and is left eager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.logger import LOGGER

from pycypher.constants import ID_COLUMN

if TYPE_CHECKING:
    import pandas as pd

#: Scan-order column used to make de-duplication and sequential id
#: generation deterministic.  Dropped before the table is created.
_ROW_ORDER_COLUMN = "_pyc_scan_order"


def _quote(name: str) -> str:
    """Return *name* as a double-quoted SQL identifier."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


class LazyEntitySource:
    """Stands in for an entity's in-memory source table.

    ``EntityTable.source_obj`` is normally an Arrow table, and the pandas
    execution path reads it whenever a DuckDB route declines.  A streaming
    entity has no such table — so this proxy materialises one **from the
    registered DuckDB table** on demand, rather than leaving the fallback
    with nothing to read.

    That keeps the streaming source safe rather than merely fast: queries
    the DuckDB path handles never touch it, and queries that fall back still
    get correct answers, at the cost of the memory the streaming load was
    avoiding.  The materialisation is logged at WARNING because it silently
    undoes the benefit and the user should be able to see which query did it.

    Row count and column names are answered from the relation, so the common
    introspection calls stay free.
    """

    __slots__ = ("_registry", "_label", "_materialised")

    def __init__(self, registry: Any, label: str) -> None:
        """Bind the proxy to *label*'s table in *registry*."""
        self._registry = registry
        self._label = label
        self._materialised: pd.DataFrame | None = None

    def _relation(self) -> Any:
        relation = self._registry.relation(self._label)
        if relation is None:
            msg = (
                f"Streaming source for {self._label!r} is no longer "
                "registered; its backend may have been closed."
            )
            raise RuntimeError(msg)
        return relation

    @property
    def columns(self) -> list[str]:
        """Column names, from the relation schema."""
        return list(self._relation().columns)

    @property
    def column_names(self) -> list[str]:
        """Alias of :attr:`columns`, matching ``pa.Table``'s spelling."""
        return self.columns

    def __len__(self) -> int:
        if self._materialised is not None:
            return len(self._materialised)
        row = self._relation().aggregate("count(*)").fetchone()
        return int(row[0]) if row else 0

    def to_pandas(self) -> pd.DataFrame:
        """Materialise the whole table, undoing the streaming benefit."""
        if self._materialised is None:
            LOGGER.warning(
                "Streaming entity %r is being materialised into pandas: a "
                "query fell back off the DuckDB path and needs the whole "
                "table in memory. Results stay correct, but the bounded "
                "memory of the streaming source is lost for this run.",
                self._label,
            )
            self._materialised = self._relation().fetchdf()
        return self._materialised

    def __repr__(self) -> str:
        return f"LazyEntitySource(label={self._label!r})"


def is_streaming_source(source_obj: Any) -> bool:
    """Return ``True`` if *source_obj* is a streaming entity's proxy.

    Call sites use this to avoid a code path that would materialise the
    source — the whole point of registering it as streaming.
    """
    return isinstance(source_obj, LazyEntitySource)


def context_has_streaming_source(context: Any, entity_type: str) -> bool:
    """Return ``True`` if *entity_type* is backed by a streaming source."""
    table = context.entity_mapping.mapping.get(entity_type)
    return table is not None and is_streaming_source(
        getattr(table, "source_obj", None),
    )


def _projection_sql(columns: list[str], id_col: str | None) -> str:
    """Return a **window-free** projection into normalised entity shape.

    Windowing is avoided deliberately.  An earlier version generated ids and
    de-duplicated with ``row_number() OVER ()``, which forces DuckDB to
    buffer the entire input — under a 256 MB budget the load did not merely
    slow down, it failed with ``OutOfMemoryException``, defeating the exact
    thing streaming registration exists to do.  So the scan stays a straight
    projection, and the two cases that need more are handled afterwards
    against the materialised (on-disk, spillable) table.
    """
    if id_col is None:
        rest = ", ".join(_quote(c) for c in columns)
        return f"SELECT {rest} FROM src"  # nosec B608 — identifiers quoted

    rest = ", ".join(_quote(c) for c in columns if c != id_col)
    projection = f"{_quote(id_col)} AS {_quote(ID_COLUMN)}"
    if rest:
        projection = f"{projection}, {rest}"
    return f"SELECT {projection} FROM src"  # nosec B608 — identifiers quoted


def _add_sequential_ids(connection: Any, table_name: str) -> None:
    """Prepend ``__ID__`` as 0-based ids in scan order.

    Uses the materialised table's ``rowid`` rather than a window function
    over the scan, so the pass streams and can spill.  Matches the Arrow
    path's ``range(len(table))``.
    """
    quoted = _quote(table_name)
    staging = _quote(f"{table_name}_seq")
    connection.execute(f"DROP TABLE IF EXISTS {staging}")  # nosec B608 — name derived from a validated table name
    connection.execute(
        f"CREATE TABLE {staging} AS "  # nosec B608 — identifiers quoted
        f"SELECT rowid AS {_quote(ID_COLUMN)}, * FROM {quoted}",
    )
    connection.execute(f"DROP TABLE {quoted}")  # nosec B608 — validated name
    connection.execute(f"ALTER TABLE {staging} RENAME TO {quoted}")  # nosec B608 — validated names


def _dedup_on_id(connection: Any, table_name: str, label: str) -> None:
    """Keep the first row per ``__ID__``, matching the Arrow path.

    Only runs when duplicates actually exist: the check is two aggregates,
    and skipping the rewrite is what keeps the common case window-free.
    """
    quoted = _quote(table_name)
    counts = connection.execute(
        f"SELECT count(*), count(DISTINCT {_quote(ID_COLUMN)}) FROM {quoted}",  # nosec B608 — identifiers quoted
    ).fetchone()
    if counts is None or counts[0] == counts[1]:
        return

    before, after = int(counts[0]), int(counts[1])
    staging = _quote(f"{table_name}_dedup")
    connection.execute(f"DROP TABLE IF EXISTS {staging}")  # nosec B608 — derived name
    connection.execute(
        f"CREATE TABLE {staging} AS "  # nosec B608 — identifiers quoted
        f"SELECT * EXCLUDE ({_quote(_ROW_ORDER_COLUMN)}) FROM ("
        f"  SELECT *, rowid AS {_quote(_ROW_ORDER_COLUMN)} FROM {quoted}"
        f") QUALIFY row_number() OVER ("
        f"  PARTITION BY {_quote(ID_COLUMN)} "
        f"  ORDER BY {_quote(_ROW_ORDER_COLUMN)}"
        ") = 1",
    )
    connection.execute(f"DROP TABLE {quoted}")  # nosec B608 — validated name
    connection.execute(f"ALTER TABLE {staging} RENAME TO {quoted}")  # nosec B608 — validated names
    LOGGER.warning(
        "streaming entity %r: dropped %d duplicate __ID__ rows (%d → %d). "
        "An entity's __ID__ must be unique; the first occurrence is kept. "
        "If you loaded a fact table at the wrong grain, project to the "
        "entity grain via the source's `query` field.",
        label,
        before - after,
        before,
        after,
    )


def register_streaming_entity(
    backend: Any,
    label: str,
    data_source: Any,
    *,
    id_col: str | None = None,
) -> Any:
    """Scan *data_source* straight into a DuckDB table for *label*.

    Args:
        backend: A ``DuckDBBackend`` (or instrumented wrapper) whose
            ``tables`` registry will own the resulting table.
        label: The entity label.
        data_source: A ``DataSource`` supporting ``read_relation``.
        id_col: Column to use as ``__ID__``.  ``None`` generates sequential
            ids in scan order.

    Returns:
        The :class:`~pycypher.backends.table_registry.RegisteredTable`.

    Raises:
        ValueError: If *id_col* is not a column of the source, matching
            ``normalize_entity_table``'s behaviour.

    """
    from pycypher.backends.table_registry import ENTITY_KIND

    connection = backend.connection
    lazy = data_source.read_relation(connection)
    columns = list(lazy.columns)

    if id_col is not None and id_col not in columns:
        msg = f"id_col {id_col!r} not found in table columns: {columns}"
        raise ValueError(msg)

    relation = lazy.relation.query("src", _projection_sql(columns, id_col))
    entry = backend.tables.register_relation(
        label,
        relation,
        kind=ENTITY_KIND,
        id_col=ID_COLUMN,
    )

    # Post-passes run against the materialised table, which lives on disk
    # and can therefore spill, unlike a window over the file scan.
    if id_col is None:
        if ID_COLUMN not in columns:
            _add_sequential_ids(connection, entry.table_name)
    else:
        _dedup_on_id(connection, entry.table_name, label)

    # The post-passes rebuild the table, so the registry's cached schema
    # (and its relation handle) must be re-read.
    entry = backend.tables.refresh(label, kind=ENTITY_KIND)

    LOGGER.debug(
        "Streaming entity %r registered from %s (%d columns, id_col=%r)",
        label,
        getattr(data_source, "uri", "<source>"),
        len(entry.columns),
        id_col,
    )
    return entry
