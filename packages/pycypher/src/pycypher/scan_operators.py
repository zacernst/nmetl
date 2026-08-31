"""Scan and filter operators for the BindingFrame execution path.

Extracted from :mod:`pycypher.binding_frame` to separate scan-level
concerns (entity/relationship table scanning, predicate pushdown,
dtype coercion) from the core BindingFrame data container.

Classes:
    EntityScan — produces a BindingFrame of entity IDs for a given type.
    RelationshipScan — produces a BindingFrame of relationship IDs.
    BindingFilter — filters a BindingFrame by a boolean AST predicate.

Helpers:
    _coerce_pushdown_ids — coerce pushdown IDs to match target column dtype.
    _coerce_pushdown_series — coerce a pushdown Series to match target dtype.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import pandas as pd
from shared.helpers import suggest_close_match
from shared.logger import LOGGER

from pycypher.constants import (
    ID_COLUMN,
    RELATIONSHIP_SOURCE_COLUMN,
    RELATIONSHIP_TARGET_COLUMN,
)
from pycypher.cypher_types import FrameSeries
from pycypher.dataframe_utils import source_to_pandas as _source_to_pandas

if TYPE_CHECKING:
    from pycypher.ast_models import Expression
    from pycypher.binding_frame import BindingFrame
    from pycypher.evaluator_protocol import ExpressionEvaluatorFactory

# ---------------------------------------------------------------------------
# Performance: module-level debug check avoids per-call overhead
# ---------------------------------------------------------------------------
_DEBUG_ENABLED: bool = LOGGER.isEnabledFor(logging.DEBUG)


# ---------------------------------------------------------------------------
# Dtype coercion helpers for predicate pushdown
# ---------------------------------------------------------------------------


def _coerce_pushdown_ids(
    ids: FrameSeries,
    target_col: pd.Series,
) -> pd.Index:
    """Build a ``pd.Index`` of unique pushdown IDs, coercing dtype to match *target_col*.

    When the DuckDB backend materialises join results via ``fetchdf()``,
    originally-integer columns may come back as ``StringDtype``.  A naive
    ``isin()`` then fails because ``'2' != 2``.  This helper converts the
    pushdown IDs to the target column's dtype so the comparison succeeds.
    """
    unique_ids = ids.dropna().unique()
    pushdown_idx = pd.Index(unique_ids)
    target_dtype = target_col.dtype

    # Fast path: dtypes already compatible.
    if pushdown_idx.dtype == target_dtype:
        return pushdown_idx

    # String-like pushdown IDs vs numeric target — try numeric conversion.
    if pd.api.types.is_string_dtype(
        pushdown_idx
    ) and pd.api.types.is_numeric_dtype(target_dtype):
        try:
            return pd.Index(pd.to_numeric(pushdown_idx))
        except ValueError, TypeError:
            return pushdown_idx

    # Numeric pushdown IDs vs object/string target — cast to object.
    if pd.api.types.is_numeric_dtype(pushdown_idx) and (
        target_dtype == object or pd.api.types.is_string_dtype(target_dtype)
    ):
        return pushdown_idx.astype(object)

    return pushdown_idx


def _coerce_pushdown_series(
    ids: FrameSeries,
    target_col: pd.Series,
) -> pd.Series:
    """Coerce a pushdown ID Series to match *target_col*'s dtype.

    Used in :meth:`RelationshipScan.scan` to normalise pushdown IDs
    before both the adjacency-index path and the table-scan path so that
    dict lookups and ``isin()`` comparisons succeed across dtype boundaries.
    """
    target_dtype = target_col.dtype

    # Fast path: already compatible.
    if ids.dtype == target_dtype:
        return ids

    # String-like IDs vs numeric target — try numeric conversion.
    if pd.api.types.is_string_dtype(
        ids.dtype
    ) and pd.api.types.is_numeric_dtype(target_dtype):
        try:
            return pd.to_numeric(ids)
        except ValueError, TypeError:
            return ids

    # Numeric IDs vs object/string target — cast to object.
    if pd.api.types.is_numeric_dtype(ids.dtype) and (
        target_dtype == object or pd.api.types.is_string_dtype(target_dtype)
    ):
        return ids.astype(object)

    return ids


# ---------------------------------------------------------------------------
# DuckDB scan helpers (Phase 3, docs/duckdb_eager_path_design.md)
# ---------------------------------------------------------------------------


def _sql_literal(value: Any) -> str | None:
    """Render *value* as a SQL literal, or ``None`` if it cannot be pushed.

    Deliberately narrow: only the scalar types a Cypher inline-property
    predicate can carry.  Anything else — including ``None`` — returns
    ``None`` so the caller falls back to the pandas path rather than
    guessing at semantics.  ``None`` is excluded on purpose: the pandas
    pushdown goes through ``PropertyValueIndex``, whose build skips nulls
    (``graph_index.py``), so ``{prop: null}`` matches nothing there and
    ``prop = NULL`` would be a silent behaviour change.
    """
    from pycypher.ingestion.security import escape_sql_string_literal

    # bool first — it is a subclass of int.
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        return escape_sql_string_literal(value)
    return None


def _quote(name: str) -> str:
    """Return *name* as a double-quoted SQL identifier."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


#: Column name used for the pushdown-id relation in a semi-join.  Prefixed
#: so it cannot collide with a real relationship-table column.
_PUSHDOWN_ID_COLUMN = "_pycypher_pushdown_id"


class LazyIds(NamedTuple):
    """Endpoint-pushdown ids that are still a DuckDB relation.

    Passing a ``pd.Series`` for pushdown means the driving frame had to be
    materialised to produce it.  Passing this instead keeps the ids in
    DuckDB, so a scan chained onto a lazy frame never leaves the engine.

    Attributes:
        frame: The :class:`~pycypher.backends.duckdb_backend.DuckDBLazyFrame`
            holding the ids.
        column: Which of its columns holds them.

    """

    frame: Any
    column: str

    def to_series(self) -> FrameSeries:
        """Materialise the ids, for the pandas fallback path."""
        return self.frame.to_pandas()[self.column]


#: DuckDB type-name prefixes treated as integral for pushdown coercion.
_INTEGRAL_PREFIXES = (
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
)


def _column_duckdb_type(relation: Any, column: str) -> str | None:
    """Return the declared DuckDB type of *column*, or ``None`` if absent."""
    try:
        return str(relation.types[relation.columns.index(column)]).upper()
    except ValueError, IndexError, AttributeError:
        return None


def _pushdown_frame(ids: FrameSeries, target_type: str | None) -> Any:
    """Build a one-column frame of unique *ids* typed to match *target_type*.

    Returns ``None`` when the ids cannot be coerced cleanly — the caller then
    falls back to the pandas scan rather than risk a semi-join that silently
    matches nothing because one side is text and the other integral.  This is
    the SQL-side counterpart of :func:`_coerce_pushdown_series`.
    """
    unique = pd.Series(pd.Series(ids).dropna().unique())
    if target_type is not None and target_type.startswith(_INTEGRAL_PREFIXES):
        converted = pd.to_numeric(unique, errors="coerce")
        if len(converted) and converted.isna().any():
            return None
        unique = converted.astype("int64") if len(converted) else converted
    elif len(unique):
        unique = unique.astype(str)
    return pd.DataFrame({_PUSHDOWN_ID_COLUMN: unique})


def _registered_table(context: Any, label: str, kind: str) -> Any:
    """Return the registered DuckDB table for *label*, or ``None``.

    ``None`` whenever the DuckDB scan path does not apply: a backend with no
    registry (pandas, polars, spark), or a label that was never materialised.
    """
    registry = getattr(getattr(context, "backend", None), "tables", None)
    if registry is None:
        return None
    try:
        return registry.get(label, kind)
    except Exception:  # noqa: BLE001 — scan path is optional; fall back to pandas
        LOGGER.debug(
            "Table registry lookup failed for %s %r; using the pandas scan",
            kind,
            label,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Scan operators
# ---------------------------------------------------------------------------


@dataclass
class EntityScan:
    """Produces a BindingFrame of all entity IDs for a given entity type.

    The resulting frame has a single column named *var_name* containing every
    ``__ID__`` value from the entity table.  Attributes are **not** included —
    they are fetched on demand via :meth:`BindingFrame.get_property`.

    Attributes:
        entity_type: The entity label (e.g. ``"Person"``).
        var_name: The Cypher variable name to bind the IDs to (e.g. ``"p"``).

    """

    entity_type: str
    var_name: str

    def _scan_duckdb(
        self,
        context: Any,
        property_filters: dict[str, Any] | None,
    ) -> BindingFrame | None:
        """Return a lazy relation-backed frame, or ``None`` to use pandas.

        Produces ``SELECT "__ID__" AS <var> FROM <table>`` with any pushable
        inline-property predicates compiled into a ``WHERE``.  That WHERE
        supersedes ``PropertyValueIndex`` entirely for this path — no index
        to build, no frozenset intersection.

        Returns ``None`` — and the caller falls back to the unchanged pandas
        scan — when any of these does not hold:

        * the backend has a table registry and this label is in it;
        * the entity type has no pending mutation overlay in
          ``context._shadow`` (a lazy scan must never read past uncommitted
          writes; this is the contract with the mutation work in
          ``duckdb_full_parity_design.md``);
        * the table exposes ``__ID__``;
        * every requested property maps to a real column and every filter
          value renders as a SQL literal.
        """
        from pycypher.backends.duckdb_backend import DuckDBLazyFrame
        from pycypher.backends.table_registry import ENTITY_KIND
        from pycypher.binding_frame import BindingFrame

        if self.entity_type in getattr(context, "_shadow", {}):
            return None
        entry = _registered_table(context, self.entity_type, ENTITY_KIND)
        if entry is None or ID_COLUMN not in entry.columns:
            return None

        predicates: list[str] = []
        for prop_name, value in (property_filters or {}).items():
            column = entry.attr_map.get(prop_name, prop_name)
            if column not in entry.columns:
                return None
            literal = _sql_literal(value)
            if literal is None:
                return None
            predicates.append(f"{_quote(column)} = {literal}")

        try:
            relation = entry.relation
            if predicates:
                relation = relation.filter(" AND ".join(predicates))
            relation = relation.project(
                f"{_quote(ID_COLUMN)} AS {_quote(self.var_name)}",
            )
        except Exception:  # noqa: BLE001 — any relation error: use the pandas scan
            LOGGER.debug(
                "DuckDB entity scan failed for %s; falling back to pandas",
                self.entity_type,
                exc_info=True,
            )
            return None

        return BindingFrame(
            relation=DuckDBLazyFrame(relation, context.backend.connection),
            type_registry={self.var_name: self.entity_type},
            context=context,
        )

    def scan(
        self,
        context: Any,
        property_filters: dict[str, Any] | None = None,
    ) -> BindingFrame:
        """Return a :class:`BindingFrame` containing all IDs for this entity type.

        Args:
            context: The query :class:`~pycypher.relational_models.Context`.
            property_filters: Optional dict of ``{prop_name: value}`` equality
                predicates.  When provided and a :class:`PropertyValueIndex`
                is available, the scan returns only IDs matching **all**
                predicates instead of the full entity table — O(1) per
                predicate instead of O(N) post-scan filtering.

        Returns:
            A :class:`BindingFrame` with one column (*var_name*) of entity IDs.

        """
        from pycypher.binding_frame import BindingFrame

        if _DEBUG_ENABLED:
            _t0 = time.perf_counter()

        lazy = self._scan_duckdb(context, property_filters)
        if lazy is not None:
            if _DEBUG_ENABLED:
                LOGGER.debug(
                    "EntityScan.scan  DUCKDB  entity_type=%s  var=%s  "
                    "filters=%s  elapsed=%.4fs",
                    self.entity_type,
                    self.var_name,
                    property_filters,
                    time.perf_counter() - _t0,
                )
            return lazy

        try:
            entity_table = context.entity_mapping[self.entity_type]
        except KeyError:
            from pycypher.exceptions import GraphTypeNotFoundError

            available = list(context.entity_mapping.mapping.keys())
            hint = suggest_close_match(self.entity_type, available)
            raise GraphTypeNotFoundError(
                self.entity_type,
                f"Entity type {self.entity_type!r} is not registered in the context. "
                f"Available entity types: {available or []}"
                f"{hint}",
            ) from None

        # --- Predicate pushdown via property index ---
        _pushed_down = False
        if property_filters:
            shadow: dict = getattr(context, "_shadow", {})
            if self.entity_type not in shadow:
                index_mgr = getattr(context, "index_manager", None)
                if index_mgr is not None:
                    try:
                        candidate_ids: frozenset | None = None
                        for prop_name, value in property_filters.items():
                            matching = index_mgr.indexed_property_lookup(
                                self.entity_type,
                                prop_name,
                                value,
                            )
                            if matching is None:
                                # No index for this property — skip pushdown
                                candidate_ids = None
                                break
                            if candidate_ids is None:
                                candidate_ids = matching
                            else:
                                candidate_ids = candidate_ids & matching
                        if candidate_ids is not None:
                            ids = pd.Series(
                                list(candidate_ids),
                                name=ID_COLUMN,
                            )
                            _pushed_down = True
                            if _DEBUG_ENABLED:
                                LOGGER.debug(
                                    "EntityScan.scan  PUSHDOWN  entity_type=%s  var=%s  "
                                    "filters=%s  matched=%d  elapsed=%.4fs",
                                    self.entity_type,
                                    self.var_name,
                                    property_filters,
                                    len(ids),
                                    time.perf_counter() - _t0,
                                )
                    except (
                        KeyError,
                        ValueError,
                        TypeError,
                        IndexError,
                        AttributeError,
                    ):
                        LOGGER.debug(
                            "EntityScan: predicate pushdown failed for %s, "
                            "falling back to full scan",
                            self.entity_type,
                            exc_info=True,
                        )

        if not _pushed_down:
            # --- Standard full scan ---
            cache: dict = getattr(context, "_property_lookup_cache", {})
            if self.entity_type not in cache:
                raw_df: pd.DataFrame = _source_to_pandas(
                    entity_table.source_obj
                )
                cache[self.entity_type] = raw_df.set_index(ID_COLUMN)
            indexed_df = cache[self.entity_type]
            ids = pd.Series(
                indexed_df.index.to_numpy(dtype=object),
                name=ID_COLUMN,
            )
            if _DEBUG_ENABLED:
                LOGGER.debug(
                    "EntityScan.scan  entity_type=%s  var=%s  rows=%d  elapsed=%.4fs",
                    self.entity_type,
                    self.var_name,
                    len(ids),
                    time.perf_counter() - _t0,
                )

        return BindingFrame(
            bindings=pd.DataFrame({self.var_name: ids}),
            type_registry={self.var_name: self.entity_type},
            context=context,
        )


@dataclass
class RelationshipScan:
    """Produces a BindingFrame of all relationship IDs for a given type.

    The resulting frame has **three** columns:

    * ``rel_var`` — the relationship's own ``__ID__``.
    * ``_src_{rel_var}`` — the source-node ``__SOURCE__`` ID.
    * ``_tgt_{rel_var}`` — the target-node ``__TARGET__`` ID.

    The source and target columns are structural join keys consumed by the
    pattern translator (Phase 5); they are **not** user-visible Cypher
    variables and are therefore absent from the ``type_registry``.

    Attributes:
        rel_type: The relationship type label (e.g. ``"KNOWS"``).
        rel_var: The Cypher variable name for the relationship (e.g. ``"r"``).
            Use a synthetic name such as ``"_anon_0"`` for anonymous
            relationships.

    """

    rel_type: str
    rel_var: str
    #: Cached column name for source-node IDs.
    src_col: str = ""
    #: Cached column name for target-node IDs.
    tgt_col: str = ""

    def __post_init__(self) -> None:
        """Cache derived column names to avoid repeated f-string creation."""
        self.src_col = f"_src_{self.rel_var}"
        self.tgt_col = f"_tgt_{self.rel_var}"

    def _scan_duckdb(
        self,
        context: Any,
        source_ids: FrameSeries | None,
        target_ids: FrameSeries | None,
    ) -> BindingFrame | None:
        """Return a lazy relation-backed frame, or ``None`` to use pandas.

        The relationship table stays inside DuckDB: only the (small) pushdown
        id set crosses the pandas boundary, as a one-column relation
        semi-joined against the endpoint column.  That replaces the pandas
        path's whole-table ``set_index`` cache plus ``isin()`` — the single
        largest table in a graph workload no longer has to be resident.

        Returns ``None`` — caller falls back to the unchanged pandas scan —
        when there is no registered table, the type has a pending mutation
        overlay in ``context._shadow_rels``, the structural columns are
        missing, or the pushdown ids cannot be coerced cleanly to the
        endpoint column's type.
        """
        from pycypher.backends.duckdb_backend import DuckDBLazyFrame
        from pycypher.backends.table_registry import RELATIONSHIP_KIND
        from pycypher.binding_frame import BindingFrame

        if self.rel_type in getattr(context, "_shadow_rels", {}):
            return None
        entry = _registered_table(context, self.rel_type, RELATIONSHIP_KIND)
        if entry is None:
            return None
        required = (
            ID_COLUMN,
            RELATIONSHIP_SOURCE_COLUMN,
            RELATIONSHIP_TARGET_COLUMN,
        )
        if any(col not in entry.columns for col in required):
            return None

        con = context.backend.connection
        try:
            relation = entry.relation
            for index, (ids, column) in enumerate(
                (
                    (source_ids, RELATIONSHIP_SOURCE_COLUMN),
                    (target_ids, RELATIONSHIP_TARGET_COLUMN),
                ),
            ):
                if ids is None:
                    continue
                if isinstance(ids, LazyIds):
                    # Ids that never left DuckDB: semi-join straight against
                    # their relation, no pandas round trip at all.
                    side = ids.frame.relation.project(
                        f"{_quote(ids.column)} AS "
                        f"{_quote(_PUSHDOWN_ID_COLUMN)}",
                    ).set_alias(f"_pyc_push{index}")
                    relation = relation.set_alias(f"_pyc_rel{index}").join(
                        side,
                        f"{_quote(column)} = {_quote(_PUSHDOWN_ID_COLUMN)}",
                        how="semi",
                    )
                    continue
                frame = _pushdown_frame(
                    ids,
                    _column_duckdb_type(entry.relation, column),
                )
                if frame is None:
                    return None
                relation = relation.join(
                    con.from_df(frame),
                    f"{_quote(column)} = {_quote(_PUSHDOWN_ID_COLUMN)}",
                    how="semi",
                )
            relation = relation.project(
                f"{_quote(ID_COLUMN)} AS {_quote(self.rel_var)}, "
                f"{_quote(RELATIONSHIP_SOURCE_COLUMN)} AS {_quote(self.src_col)}, "
                f"{_quote(RELATIONSHIP_TARGET_COLUMN)} AS {_quote(self.tgt_col)}",
            )
        except Exception:  # noqa: BLE001 — any relation error: use the pandas scan
            LOGGER.debug(
                "DuckDB relationship scan failed for %s; falling back to pandas",
                self.rel_type,
                exc_info=True,
            )
            return None

        return BindingFrame(
            relation=DuckDBLazyFrame(relation, con),
            type_registry={self.rel_var: self.rel_type},
            context=context,
        )

    def scan(
        self,
        context: Any,
        *,
        source_ids: FrameSeries | None = None,
        target_ids: FrameSeries | None = None,
    ) -> BindingFrame:
        """Return a :class:`BindingFrame` containing relationship IDs for this type.

        Supports **predicate pushdown**: when *source_ids* or *target_ids*
        are provided, only relationships whose ``__SOURCE__`` / ``__TARGET__``
        column values appear in the given ID set are materialised.  This
        avoids loading the full relationship table when the query pattern
        already constrains one endpoint.

        Args:
            context: The query :class:`~pycypher.relational_models.Context`.
            source_ids: If provided, only materialise relationships whose
                ``__SOURCE__`` is in this set.
            target_ids: If provided, only materialise relationships whose
                ``__TARGET__`` is in this set.

        Returns:
            A :class:`BindingFrame` with columns *rel_var*, ``_src_{rel_var}``,
            and ``_tgt_{rel_var}``.

        """
        from pycypher.binding_frame import BindingFrame

        if _DEBUG_ENABLED:
            _t0 = time.perf_counter()

        lazy = self._scan_duckdb(context, source_ids, target_ids)
        if lazy is None:
            # The pandas path below indexes these as Series; ids that were
            # kept lazy for the DuckDB route have to come back down.
            if isinstance(source_ids, LazyIds):
                source_ids = source_ids.to_series()
            if isinstance(target_ids, LazyIds):
                target_ids = target_ids.to_series()
        if lazy is not None:
            if _DEBUG_ENABLED:
                LOGGER.debug(
                    "RelationshipScan.scan  DUCKDB  rel_type=%s  var=%s  "
                    "pushdown=%s  elapsed=%.4fs",
                    self.rel_type,
                    self.rel_var,
                    source_ids is not None or target_ids is not None,
                    time.perf_counter() - _t0,
                )
            return lazy

        try:
            rel_table = context.relationship_mapping[self.rel_type]
        except KeyError:
            from pycypher.exceptions import GraphTypeNotFoundError

            available = list(context.relationship_mapping.mapping.keys())
            hint = suggest_close_match(self.rel_type, available)
            raise GraphTypeNotFoundError(
                self.rel_type,
                f"Relationship type {self.rel_type!r} is not registered in the context. "
                f"Available relationship types: {available or []}"
                f"{hint}",
            ) from None

        # --- Coerce pushdown IDs to match relationship table dtypes ---
        if source_ids is not None or target_ids is not None:
            raw_df_for_dtype: pd.DataFrame = _source_to_pandas(
                rel_table.source_obj
            )
            if source_ids is not None:
                source_ids = _coerce_pushdown_series(
                    source_ids,
                    raw_df_for_dtype[RELATIONSHIP_SOURCE_COLUMN],
                )
            if target_ids is not None:
                target_ids = _coerce_pushdown_series(
                    target_ids,
                    raw_df_for_dtype[RELATIONSHIP_TARGET_COLUMN],
                )

        # --- Fast path: adjacency index for O(degree) pushdown ---
        _shadow_rels: dict = getattr(context, "_shadow_rels", {})
        if (
            source_ids is not None or target_ids is not None
        ) and self.rel_type not in _shadow_rels:
            try:
                index_mgr = getattr(context, "index_manager", None)
                if index_mgr is not None:
                    idx_result = index_mgr.indexed_relationship_scan(
                        self.rel_type,
                        source_ids=source_ids,
                        target_ids=target_ids,
                    )
                    if idx_result is not None:
                        bindings = pd.DataFrame(
                            {
                                self.rel_var: idx_result[ID_COLUMN].values,
                                self.src_col: idx_result[
                                    RELATIONSHIP_SOURCE_COLUMN
                                ].values,
                                self.tgt_col: idx_result[
                                    RELATIONSHIP_TARGET_COLUMN
                                ].values,
                            },
                        )
                        if _DEBUG_ENABLED:
                            LOGGER.debug(
                                "RelationshipScan.scan  rel_type=%s  var=%s  rows=%d  pushdown=index  elapsed=%.4fs",
                                self.rel_type,
                                self.rel_var,
                                len(bindings),
                                time.perf_counter() - _t0,
                            )
                        return BindingFrame(
                            bindings=bindings,
                            type_registry={self.rel_var: self.rel_type},
                            context=context,
                        )
            except (
                KeyError,
                ValueError,
                TypeError,
                IndexError,
                AttributeError,
            ):
                # Fall through to table-scan path on any index error
                LOGGER.debug(
                    "RelationshipScan: index scan failed for %s, falling back to table scan",
                    self.rel_type,
                    exc_info=True,
                )

        # --- Fallback: table scan with isin() pushdown ---
        cache: dict = getattr(context, "_property_lookup_cache", {})
        cache_key = f"__rel__{self.rel_type}"
        if cache_key not in cache:
            raw_df: pd.DataFrame = _source_to_pandas(rel_table.source_obj)
            cache[cache_key] = raw_df.set_index(ID_COLUMN)
        indexed_df = cache[cache_key]

        # --- Predicate pushdown: filter at scan level ---
        mask: pd.Series | None = None
        if source_ids is not None:
            source_set = _coerce_pushdown_ids(
                source_ids, indexed_df[RELATIONSHIP_SOURCE_COLUMN]
            )
            src_mask = indexed_df[RELATIONSHIP_SOURCE_COLUMN].isin(source_set)
            mask = src_mask if mask is None else mask & src_mask
        if target_ids is not None:
            target_set = _coerce_pushdown_ids(
                target_ids, indexed_df[RELATIONSHIP_TARGET_COLUMN]
            )
            tgt_mask = indexed_df[RELATIONSHIP_TARGET_COLUMN].isin(target_set)
            mask = tgt_mask if mask is None else mask & tgt_mask

        if mask is not None:
            filtered = indexed_df[mask]
        else:
            filtered = indexed_df

        # Recover the three columns from the (possibly filtered) DataFrame.
        ids = pd.Series(
            filtered.index.to_numpy(dtype=object),
            name=ID_COLUMN,
        )
        bindings = pd.DataFrame(
            {
                self.rel_var: ids,
                self.src_col: filtered[RELATIONSHIP_SOURCE_COLUMN].to_numpy(
                    dtype=object,
                ),
                self.tgt_col: filtered[RELATIONSHIP_TARGET_COLUMN].to_numpy(
                    dtype=object,
                ),
            },
        )
        if _DEBUG_ENABLED:
            _pushdown = source_ids is not None or target_ids is not None
            LOGGER.debug(
                "RelationshipScan.scan  rel_type=%s  var=%s  rows=%d  pushdown=%s  elapsed=%.4fs",
                self.rel_type,
                self.rel_var,
                len(ids),
                _pushdown,
                time.perf_counter() - _t0,
            )
        return BindingFrame(
            bindings=bindings,
            type_registry={self.rel_var: self.rel_type},
            context=context,
        )


# ---------------------------------------------------------------------------
# Predicate pushdown helpers (Phase 5, docs/duckdb_eager_path_design.md)
# ---------------------------------------------------------------------------


def _walk_ast(node: Any) -> Any:
    """Yield *node* and every nested AST node beneath it.

    Generic over the pydantic AST models rather than enumerating node types,
    so a predicate shape added later is still traversed.
    """
    yield node
    fields = getattr(type(node), "model_fields", None)
    if not fields:
        return
    for name in fields:
        value = getattr(node, name, None)
        if isinstance(value, (list, tuple)):
            for item in value:
                if getattr(type(item), "model_fields", None):
                    yield from _walk_ast(item)
        elif getattr(type(value), "model_fields", None):
            yield from _walk_ast(value)


def _property_refs(expr: Any) -> set[tuple[str, str]] | None:
    """Return the ``(variable, property)`` pairs *expr* reads.

    ``None`` when the expression contains a property lookup on something
    other than a plain variable (e.g. ``f(x).prop``), which this pushdown
    cannot resolve to a column.
    """
    from pycypher.ast_models import PropertyLookup
    from pycypher.ast_models import Variable as _Variable

    refs: set[tuple[str, str]] = set()
    for node in _walk_ast(expr):
        if isinstance(node, PropertyLookup):
            if not isinstance(node.expression, _Variable):
                return None
            refs.add((node.expression.name, node.property))
    return refs


def _split_conjuncts(expr: Any) -> list[Any]:
    """Flatten a top-level ``AND`` chain into its conjuncts.

    ``And`` carries either an ``operands`` list or ``left``/``right``, so
    both spellings are handled.  Splitting is safe for a ``WHERE``: keeping
    rows where every conjunct is TRUE is the same as keeping rows where
    their conjunction is TRUE, under Kleene logic as under pandas'
    ``fillna(False)``.
    """
    from pycypher.ast_models import And

    if not isinstance(expr, And):
        return [expr]
    parts = list(expr.operands) if expr.operands else [expr.left, expr.right]
    out: list[Any] = []
    for part in parts:
        if part is not None:
            out.extend(_split_conjuncts(part))
    return out


def _entity_relation_entry(context: Any, entity_type: str) -> Any:
    """Return the registered table for *entity_type*, entity or relationship."""
    from pycypher.backends.table_registry import (
        ENTITY_KIND,
        RELATIONSHIP_KIND,
    )

    return _registered_table(
        context, entity_type, ENTITY_KIND
    ) or _registered_table(context, entity_type, RELATIONSHIP_KIND)


# ---------------------------------------------------------------------------
# Filter operator
# ---------------------------------------------------------------------------


@dataclass
class BindingFilter:
    """Filters a BindingFrame by evaluating a boolean AST expression.

    This is the Phase-3 analogue of the legacy ``FilterRows`` operator.
    Unlike ``FilterRows``, it never touches prefixed column names — it
    delegates directly to :class:`~pycypher.binding_evaluator.BindingExpressionEvaluator`.

    Attributes:
        predicate: The Cypher AST boolean expression to evaluate (e.g. a
            ``Comparison``, ``And``, ``NullCheck``, etc.).

    """

    predicate: Expression
    evaluator_factory: ExpressionEvaluatorFactory

    def _push_to_sql(
        self,
        frame: BindingFrame,
    ) -> tuple[BindingFrame, list[Any]]:
        """Push what compiles into the relation; return the rest.

        Splits the predicate on ``AND`` and compiles each conjunct through
        :func:`~pycypher.relation_sql.compile_expression`.  A conjunct that
        compiles becomes a SQL ``WHERE``; anything else — an unbridged
        function, an ``EXISTS`` subquery, a pattern comprehension, a bare
        variable reference — is returned for the unchanged pandas evaluator.
        A partially pushed filter is still a large win, so this is
        deliberately not all-or-nothing.

        Properties are resolved by LEFT-joining the referenced entity table
        on ``<var> = __ID__`` and reading the aliased column.  The join is
        LEFT so an id with no matching row yields NULL, which is what
        ``get_property`` produces on the pandas side.  Each side relation is
        projected down to just its id and the wanted property *before*
        joining, so an entity column can never collide with a binding
        variable's name.  The result is projected back to exactly the
        frame's original columns, so the frame's variable list is unchanged
        — property columns must never leak into ``var_names``, which the
        engine reads as the list of bound Cypher variables.

        Returns:
            ``(frame, residual_conjuncts)``.  The frame is unchanged and the
            residual is the full conjunct list when nothing could be pushed.

        """
        from pycypher.backends.duckdb_backend import DuckDBLazyFrame
        from pycypher.binding_frame import BindingFrame as _BF
        from pycypher.relation_sql import compile_expression

        context = frame.context
        conjuncts = _split_conjuncts(self.predicate)
        shadow = getattr(context, "_shadow", {})
        shadow_rels = getattr(context, "_shadow_rels", {})
        udfs = frozenset(getattr(context, "_relation_udfs", set()) or set())

        aliases: dict[tuple[str, str], str] = {}
        sources: dict[tuple[str, str], tuple[str, str]] = {}

        def resolve(var: str, prop: str) -> str | None:
            key = (var, prop)
            if key in aliases:
                return aliases[key]
            entity_type = frame.type_registry.get(var)
            if entity_type is None or entity_type == "__MULTI__":
                return None
            if entity_type in shadow or entity_type in shadow_rels:
                return None
            entry = _entity_relation_entry(context, entity_type)
            if entry is None or ID_COLUMN not in entry.columns:
                return None
            column = entry.attr_map.get(prop, prop)
            if column not in entry.columns:
                # Cypher: reading a property that does not exist yields null.
                aliases[key] = "NULL"
                return "NULL"
            alias = f"_pyc_p{len(sources)}"
            sources[key] = (entity_type, column)
            aliases[key] = _quote(alias)
            return aliases[key]

        pushable: list[tuple[Any, str]] = []
        residual: list[Any] = []
        for conjunct in conjuncts:
            refs = _property_refs(conjunct)
            if refs is None:
                residual.append(conjunct)
                continue
            sql = compile_expression(conjunct, resolve, udfs)
            if sql is None:
                residual.append(conjunct)
            else:
                pushable.append((conjunct, sql))

        if not pushable:
            return frame, conjuncts

        original_columns = list(frame.var_names)
        try:
            # Every relation here descends from the same registered tables,
            # so DuckDB rejects a join between two of them unless their
            # aliases are made distinct first.
            relation = frame.relation.relation.set_alias("_pyc_lhs")
            for index, (key, (entity_type, column)) in enumerate(
                sources.items()
            ):
                var, _prop = key
                entry = _entity_relation_entry(context, entity_type)
                id_alias = f"_pyc_id{index}"
                side = entry.relation.project(
                    f"{_quote(ID_COLUMN)} AS {_quote(id_alias)}, "
                    f"{_quote(column)} AS {aliases[key]}",
                ).set_alias(f"_pyc_side{index}")
                relation = relation.join(
                    side,
                    f"{_quote(var)} = {_quote(id_alias)}",
                    how="left",
                )
            for _conjunct, sql in pushable:
                relation = relation.filter(sql)
            relation = relation.project(
                ", ".join(_quote(c) for c in original_columns),
            )
        except Exception:  # noqa: BLE001 — any relation error: evaluate it all in pandas
            LOGGER.debug(
                "Predicate pushdown failed; evaluating in pandas",
                exc_info=True,
            )
            return frame, conjuncts

        pushed = _BF(
            relation=DuckDBLazyFrame(relation, context.backend.connection),
            type_registry=frame.type_registry,
            context=context,
        )
        return pushed, residual

    def apply(self, frame: BindingFrame) -> BindingFrame:
        """Return a new BindingFrame containing only rows where *predicate* is True.

        Args:
            frame: The input :class:`BindingFrame`.

        Returns:
            A filtered :class:`BindingFrame`.

        """
        if _DEBUG_ENABLED:
            _t0 = time.perf_counter()
            _rows_before = len(frame)

        remaining: list[Any] = [self.predicate]
        if frame.is_lazy:
            frame, remaining = self._push_to_sql(frame)

        # Applying conjuncts in sequence is equivalent to applying their
        # conjunction: a WHERE keeps only rows where each is TRUE, under
        # Kleene logic exactly as under fillna(False).
        result = frame
        for predicate in remaining:
            evaluator = self.evaluator_factory(result)
            mask: FrameSeries = (
                evaluator.evaluate(predicate).fillna(False).astype(bool)
            )
            result = result.filter(mask)

        if _DEBUG_ENABLED:
            LOGGER.debug(
                "BindingFilter.apply  predicate=%s  pushed=%d  rows_before=%d  "
                "rows_after=%d  elapsed=%.4fs",
                type(self.predicate).__name__,
                len(_split_conjuncts(self.predicate)) - len(remaining),
                _rows_before,
                len(result),
                time.perf_counter() - _t0,
            )
        return result
