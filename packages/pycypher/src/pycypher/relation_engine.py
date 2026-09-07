"""Out-of-core relation execution path (Approach A, Phase 4+).

A separate, **opt-in** execution path that runs an *eligible* subset of Cypher
queries directly as DuckDB relations instead of the pandas ``BindingFrame``
engine.  Anything not eligible falls back to the existing engine, so coverage
grows one query-feature at a time while the suite stays green.

Disabled by default — enabled per-:class:`Context` via a truthy
``_relation_engine_enabled`` attribute or the
``PYCYPHER_DUCKDB_RELATION_ENGINE`` environment variable.  When disabled the
dispatch never fires, guaranteeing zero behaviour change.

Which queries are eligible is decided by translation to a logical plan —
``pycypher.plan.translate`` applies one rule per clause type and raises
``Unsupported`` naming any construct it has no rule for; see
``docs/cypher_relational_algebra_generalization_plan.md`` for the operator
set and the coverage matrix. Nothing in this module enumerates query shapes.

Source modes: with :func:`register_streaming_source` the base relation is a
lazy ``read_relation`` view over a file (genuinely out-of-core); otherwise it
falls back to the entity's in-memory ``source_obj``.  Combined with
``materialize=False`` + ``write_relation_to_uri`` (COPY), an eligible query
streams file → relation → sink without a pandas frame, and ``nmetl run`` uses
this automatically when enabled (see ``cli/pipeline.py`` ``_try_streaming_run``).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from shared.logger import LOGGER

if TYPE_CHECKING:
    import pandas as pd

    from pycypher.relational_models import Context

_ENABLE_ENV_VAR = "PYCYPHER_DUCKDB_RELATION_ENGINE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Reserved relationship columns carrying the entity label each edge's
#: endpoints were *declared* to have (``source_entity_type`` /
#: ``target_entity_type`` on the relationship source). NULL when the source
#: declared none. A pattern join filters on them when present, so an edge
#: registered as ``HousingSurvey5yr -> PUMA`` can never be traversed from a
#: ``HousingSurvey1yr`` node that happens to share the same id value. Not
#: exposed as Cypher properties (see :func:`register_streaming_relationship`).
SOURCE_LABEL_COLUMN = "__SOURCE_LABEL__"
TARGET_LABEL_COLUMN = "__TARGET_LABEL__"
_RESERVED_REL_COLUMNS = frozenset(
    {
        "__ID__",
        "__SOURCE__",
        "__TARGET__",
        SOURCE_LABEL_COLUMN,
        TARGET_LABEL_COLUMN,
    }
)


class RelationBindings:
    """Relation-backed bindings — the out-of-core counterpart to the pandas
    ``BindingFrame``.

    Wraps a ``DuckDBLazyFrame`` and exposes :meth:`to_pandas` as the single
    materialisation boundary.  Kept deliberately thin at this phase; later
    phases grow it into the full relation IR.
    """

    __slots__ = ("_lazy",)

    def __init__(self, lazy: Any) -> None:
        self._lazy = lazy

    @property
    def lazy(self) -> Any:
        """The underlying ``DuckDBLazyFrame`` (for streaming to a sink)."""
        return self._lazy

    @property
    def columns(self) -> list[str]:
        """Column names, from the relation schema (no materialisation)."""
        return list(self._lazy.columns)

    def to_pandas(self) -> pd.DataFrame:
        """Materialise the relation to a pandas DataFrame."""
        return self._lazy.to_pandas()


def relation_engine_enabled(context: Context) -> bool:
    """Return True if the opt-in relation engine is enabled for *context*."""
    if getattr(context, "_relation_engine_enabled", False):
        return True
    return os.environ.get(_ENABLE_ENV_VAR, "").strip().lower() in _TRUTHY


def register_streaming_source(
    context: Context,
    label: str,
    data_source: Any,
    *,
    id_col: str | None = None,
) -> None:
    """Register a file-backed entity as a streaming DuckDB relation.

    Reads *data_source* via :meth:`DataSource.read_relation` on the context's
    shared DuckDB connection, then hands it to the backend's
    :class:`~pycypher.backends.table_registry.TableRegistry`, which
    materialises it once into a real DuckDB table (a streaming
    ``CREATE TABLE AS SELECT`` that never loads the file into pandas) instead
    of leaving it as a lazy view the relation engine would otherwise re-scan
    from disk on every query that reads this source (see
    docs/duckdb_full_parity_design.md, Phase 1).  The registry derives the
    property→column map and records the ID column's declared type; the result
    is mirrored onto ``context._streaming_sources`` as a read-through alias
    for the existing relation-engine call sites.  Requires a DuckDB-backed
    context.

    Args:
        context: A DuckDB-backed :class:`Context`.
        label: The entity label to register the source under.
        data_source: A :class:`DataSource` (typically from
            :func:`data_source_from_uri`).
        id_col: The column designated as the entity ID.  Excluded from the
            property map so semantics match the in-memory path (where the ID
            column is consumed into ``__ID__`` and is not a property).

    """
    from pycypher.backends.table_registry import ENTITY_KIND

    con = context.backend.connection
    lazy = data_source.read_relation(con)
    # Every non-ID column is exposed as a property named after the column
    # (identity map, derived by the registry), mirroring the ContextBuilder
    # convention for file sources.
    entry = context.backend.tables.register_relation(
        label,
        lazy,
        kind=ENTITY_KIND,
        id_col=id_col,
    )
    # ``_streaming_sources`` is kept as a read-through alias onto the
    # registry so the existing relation-engine call sites (``_base_relation``,
    # ``_streaming_id_col``, the mutation slices) keep working unchanged.
    context._streaming_sources[label] = (
        entry.frame,
        entry.attr_map,
        entry.id_col,
    )


def _quote_source_identifier(name: str) -> str:
    """Validate *name* as a safe raw-column identifier and return it quoted.

    Uses :func:`~pycypher.ingestion.security.sanitize_sql_identifier` — the
    same validation :mod:`pycypher.ingestion.data_sources` applies to
    ``schema_hints`` column names — since these identifiers come from the
    same trusted-but-config-supplied source (a pipeline YAML's
    ``source_col``/``target_col``/``id_col``/raw column names).
    """
    from pycypher.ingestion.security import sanitize_sql_identifier

    safe = sanitize_sql_identifier(name)
    return f'"{safe}"'


def _relationship_streaming_sql(
    columns: list[str],
    *,
    source_col: str,
    target_col: str,
    id_col: str | None,
    allow_multi_edges: bool,
    view_name: str,
    source_label: str | None = None,
    target_label: str | None = None,
) -> str:
    """Build the SQL that reshapes a raw relationship scan into
    ``__ID__``/``__SOURCE__``/``__TARGET__`` form, plus the reserved
    ``__SOURCE_LABEL__``/``__TARGET_LABEL__`` columns (the declared endpoint
    entity labels as string literals, or NULL when undeclared).

    Mirrors :func:`~pycypher.ingestion.arrow_utils.normalize_relationship_table`
    (the eager path's equivalent) as SQL window functions instead of an Arrow
    pass, so the source ``DuckDBPyRelation`` never gets materialised into
    pandas/Arrow here — the caller (:func:`register_streaming_relationship`)
    hands the result straight to the table registry's ``CREATE TABLE AS
    SELECT``, which is the only place this data actually gets written down.

    Semantics matched exactly (verified against ``normalize_relationship_table``
    on representative fixtures, including NULL endpoints and duplicate ids):
    *source_col*/*target_col* renamed to ``__SOURCE__``/``__TARGET__``; if
    *id_col* is given, it's renamed to ``__ID__`` and duplicate ``__ID__``
    rows are collapsed to the first occurrence; unless *allow_multi_edges*,
    duplicate ``(__SOURCE__, __TARGET__)`` pairs are then collapsed to the
    first occurrence (NULL endpoints group together, matching pandas'
    ``duplicated()``); if no *id_col* was given, a sequential ``__ID__`` is
    assigned afterward so ids stay contiguous.

    "First occurrence" is tie-broken by ``ROW_NUMBER() OVER ()`` taken over
    the raw scan before any reshaping, and the final result is explicitly
    ``ORDER BY``'d on that ordinal so output order matches input order the
    same way the eager Arrow path's original-row-order guarantee does.
    DuckDB does not document a global ordering guarantee for an unordered
    window function over a parallel scan, so on a source that actually has
    duplicates to collapse, *which* physical row survives could in principle
    differ from the eager path in rare cases; a source with no duplicates
    (the common case for this pipeline's relationship sources) is unaffected
    either way, since there is nothing to choose between.

    Args:
        columns: Column names present in the raw (post-``query``,
            post-``schema_hints``) scan.
        source_col: Column to rename to ``__SOURCE__``.
        target_col: Column to rename to ``__TARGET__``.
        id_col: Column to rename to ``__ID__``, or ``None`` to
            auto-generate.
        allow_multi_edges: When ``False``, duplicate ``(__SOURCE__,
            __TARGET__)`` pairs are collapsed to one edge.
        view_name: The virtual table name the caller registered this SQL's
            ``FROM`` clause against (see ``DuckDBPyRelation.query()``).
        source_label: Entity label every edge's source node is declared to
            have, or ``None``. Emitted as a constant string column.
        target_label: Same for the target node.

    Returns:
        A complete SQL ``SELECT`` statement.

    Raises:
        ValueError: If *source_col* or *target_col* is not present in
            *columns*.
        SecurityError: If any identifier is not a safe SQL column name.

    """
    if source_col not in columns:
        msg = f"source_col {source_col!r} not found in columns: {columns}"
        raise ValueError(msg)
    if target_col not in columns:
        msg = f"target_col {target_col!r} not found in columns: {columns}"
        raise ValueError(msg)

    from pycypher.ingestion.security import escape_sql_string_literal

    other_cols = [
        c
        for c in columns
        if c not in (source_col, target_col, id_col)
        and c not in (SOURCE_LABEL_COLUMN, TARGET_LABEL_COLUMN)
    ]

    def label_item(label: str | None, column: str) -> str:
        value = (
            escape_sql_string_literal(label)
            if label is not None
            else "CAST(NULL AS VARCHAR)"
        )
        return f"{value} AS {column}"

    select_items = ["ROW_NUMBER() OVER () AS __row_ord__"]
    if id_col is not None:
        select_items.append(
            f"{_quote_source_identifier(id_col)} AS __ID__",
        )
    select_items.append(
        f"{_quote_source_identifier(source_col)} AS __SOURCE__"
    )
    select_items.append(
        f"{_quote_source_identifier(target_col)} AS __TARGET__"
    )
    select_items.append(label_item(source_label, SOURCE_LABEL_COLUMN))
    select_items.append(label_item(target_label, TARGET_LABEL_COLUMN))
    select_items.extend(_quote_source_identifier(c) for c in other_cols)

    # nosec B608 — every identifier above is validated by
    # _quote_source_identifier (sanitize_sql_identifier); view_name is a
    # literal chosen by this module, not caller-supplied.
    sql = f"WITH base AS (SELECT {', '.join(select_items)} FROM {view_name})"
    last_cte = "base"

    if id_col is not None:
        sql += (
            f", dedup_id AS (SELECT * FROM {last_cte} "
            "QUALIFY ROW_NUMBER() OVER "
            "(PARTITION BY __ID__ ORDER BY __row_ord__) = 1)"
        )
        last_cte = "dedup_id"

    if not allow_multi_edges:
        sql += (
            f", dedup_endpoints AS (SELECT * FROM {last_cte} "
            "QUALIFY ROW_NUMBER() OVER "
            "(PARTITION BY __SOURCE__, __TARGET__ ORDER BY __row_ord__) = 1)"
        )
        last_cte = "dedup_endpoints"

    final_id = (
        "__ID__"
        if id_col is not None
        else "ROW_NUMBER() OVER (ORDER BY __row_ord__) - 1 AS __ID__"
    )
    final_cols = [
        "__SOURCE__",
        "__TARGET__",
        SOURCE_LABEL_COLUMN,
        TARGET_LABEL_COLUMN,
    ] + [_quote_source_identifier(c) for c in other_cols]
    sql += (
        f" SELECT {final_id}, {', '.join(final_cols)} "
        f"FROM {last_cte} ORDER BY __row_ord__"
    )
    return sql


def register_streaming_relationship(
    context: Context,
    label: str,
    data_source: Any,
    *,
    source_col: str,
    target_col: str,
    id_col: str | None = None,
    allow_multi_edges: bool = False,
    source_entity_type: str | None = None,
    target_entity_type: str | None = None,
) -> None:
    """Register a file-backed relationship as a streaming DuckDB relation.

    The relationship counterpart to :func:`register_streaming_source` — see
    the FastOpenData streaming-qualification plan (private repository), "Phase 1". Reads
    *data_source* via :meth:`DataSource.read_relation` on the context's
    shared DuckDB connection (lazy — never materialises pandas/Arrow), then
    reshapes it to ``__ID__``/``__SOURCE__``/``__TARGET__`` form with the
    same semantics as :func:`~pycypher.ingestion.arrow_utils.
    normalize_relationship_table` (see :func:`_relationship_streaming_sql`),
    and hands the still-lazy result to the table registry, which
    materialises it once via ``CREATE TABLE AS SELECT`` — the only place
    this data is actually written down, and DuckDB's own storage rather
    than a Python-process pandas/Arrow object.

    ``_rel_base_relation``/``_rel_attr_map`` already check the table
    registry before falling back to ``context.relationship_mapping``
    (mirroring how ``_base_relation``/``_entity_attr_map`` check
    ``context._streaming_sources`` before ``context.entity_mapping``), so
    no separate ``context._streaming_relationships`` alias is needed here —
    registering into the table registry is sufficient for the relation
    engine to find this relationship.

    Args:
        context: A DuckDB-backed :class:`Context`.
        label: The relationship type to register the source under.
        data_source: A :class:`DataSource` (typically from
            :func:`data_source_from_uri`).
        source_col: Column holding source-node IDs (renamed to
            ``__SOURCE__``).
        target_col: Column holding target-node IDs (renamed to
            ``__TARGET__``).
        id_col: Column holding the relationship's own identity, if any.
            Auto-generated when ``None``.
        allow_multi_edges: When ``False`` (default), rows sharing a
            ``(source_col, target_col)`` pair are collapsed to one edge —
            matching :func:`~pycypher.ingestion.arrow_utils.
            normalize_relationship_table`'s default.
        source_entity_type: Entity label every edge in this source starts
            from, if known. Recorded per edge in ``__SOURCE_LABEL__`` and
            enforced by pattern joins, so two entity types that share id
            values (the fastopendata 1-year and 5-year survey files) each
            traverse only their own edges. ``None`` (default) records NULL,
            which matches any label — the pre-existing behaviour.
        target_entity_type: Same for the edge's target node.

    """
    from pycypher.backends.table_registry import RELATIONSHIP_KIND

    con = context.backend.connection
    lazy = data_source.read_relation(con)
    raw = lazy.relation
    view_name = "__rel_raw_source__"
    sql = _relationship_streaming_sql(
        raw.columns,
        source_col=source_col,
        target_col=target_col,
        id_col=id_col,
        allow_multi_edges=allow_multi_edges,
        view_name=view_name,
        source_label=source_entity_type,
        target_label=target_entity_type,
    )
    transformed = raw.query(view_name, sql)
    # __SOURCE__/__TARGET__ (and the endpoint-label columns) are reserved
    # structural columns, not properties — register_relation's default
    # attr_map (every non-id_col column) would otherwise expose them as
    # Cypher properties, unlike the eager path's infer_attribute_map(),
    # which excludes them explicitly.
    attr_map = {
        c: c for c in transformed.columns if c not in _RESERVED_REL_COLUMNS
    }

    context.backend.tables.register_relation(
        label,
        transformed,
        kind=RELATIONSHIP_KIND,
        id_col="__ID__",
        attr_map=attr_map,
    )


def register_relation_udf(
    context: Context,
    name: str,
    fn: Any,
    *,
    param_types: list[str],
    return_type: str,
) -> None:
    """Register a scalar Python function as a DuckDB UDF for the relation engine.

    Makes ``fn`` callable from eligible out-of-core queries as ``name(args)``.
    Types must be given explicitly (DuckDB type strings, e.g. ``"DOUBLE"``,
    ``"VARCHAR"``, ``"BIGINT"``) — the query engine can't infer them.  DuckDB's
    default null handling returns NULL for NULL input without invoking ``fn``.

    Args:
        context: A DuckDB-backed :class:`Context`.
        name: Cypher function name used in queries (case-insensitive).
        fn: A plain scalar Python callable (one value per argument → one value).
        param_types: DuckDB type string per positional argument.
        return_type: DuckDB return type string.

    """
    lname = name.lower()
    if (
        lname in context._relation_udfs
    ):  # idempotent — bridging may re-run over the same context
        return
    context.backend.connection.create_function(
        lname, fn, param_types, return_type
    )
    context._relation_udfs.add(lname)


def _udf_names(context: Context) -> frozenset[str]:
    """Return the set of registered relation-engine UDF names (lowercase)."""
    return frozenset(context._relation_udfs)


#: Python annotation type → DuckDB type string for bridging user functions.
_PY_TO_DUCKDB: dict[type, str] = {
    int: "BIGINT",
    float: "DOUBLE",
    str: "VARCHAR",
    bool: "BOOLEAN",
    bytes: "BLOB",
}


def _duckdb_types_from_annotations(func: Any) -> tuple[list[str], str] | None:
    """Derive ``(param_types, return_type)`` from *func*'s type annotations.

    Returns ``None`` if the signature has non-positional params, or any
    parameter / the return lacks a mappable annotation — such functions can't be
    bridged and stay on the pandas engine.
    """
    import inspect
    import typing

    try:
        hints = typing.get_type_hints(func)
        sig = inspect.signature(func)
    except TypeError, ValueError, NameError:
        return None

    param_types: list[str] = []
    for param in sig.parameters.values():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return None
        duckdb_type = _PY_TO_DUCKDB.get(hints.get(param.name))
        if duckdb_type is None:
            return None
        param_types.append(duckdb_type)

    return_type = _PY_TO_DUCKDB.get(hints.get("return"))
    if return_type is None or not param_types:
        return None
    return param_types, return_type


def bridge_user_functions(context: Context) -> None:
    """Bridge annotated user scalar functions to DuckDB UDFs for out-of-core use.

    Iterates the :class:`ScalarFunctionRegistry`; for each function registered
    from a plain scalar callable (recoverable via ``__wrapped__``) whose type
    annotations map cleanly to DuckDB types, registers it on the context's
    connection via :func:`register_relation_udf`.  Functions without a
    recoverable original or mappable annotations (e.g. built-ins, unannotated
    user functions) are skipped and continue to fall back to the pandas engine.
    """
    if getattr(context, "backend_name", None) != "duckdb":
        return
    from pycypher.scalar_functions import ScalarFunctionRegistry

    registry = ScalarFunctionRegistry.get_instance()
    for name, meta in registry._functions.items():  # noqa: SLF001 — read-only bridge
        original = getattr(meta.callable, "__wrapped__", None)
        if original is None:
            continue
        types = _duckdb_types_from_annotations(original)
        if types is None:
            continue
        param_types, return_type = types
        try:
            register_relation_udf(
                context,
                name,
                original,
                param_types=param_types,
                return_type=return_type,
            )
        except Exception:  # noqa: BLE001 — best-effort; skip anything DuckDB rejects
            from shared.logger import LOGGER

            LOGGER.debug(
                "could not bridge user function %r", name, exc_info=True
            )


def _entity_attr_map(context: Context, label: str) -> dict[str, str] | None:
    """Property→column map for *label*.

    Prefers a streaming source, then the table registry's record (the live
    dict :func:`_ensure_column` extends when a ``SET`` creates a column —
    an in-memory label materialised into the registry must resolve that
    column on the next query), then the ``EntityTable`` mapping.
    """
    from pycypher.backends.table_registry import ENTITY_KIND

    streaming = context._streaming_sources
    if label in streaming:
        return streaming[label][1]
    registry = getattr(context.backend, "tables", None)
    if registry is not None:
        entry = registry.get(label, ENTITY_KIND)
        if entry is not None:
            return entry.attr_map
    entity = context.entity_mapping.mapping.get(label)
    return entity.attribute_map if entity is not None else None


def _registered_relation(context: Context, label: str, kind: str) -> Any:
    """Return *label*'s relation from the backend table registry, or ``None``.

    Tolerates a backend without a registry (pandas, polars, spark) and a
    registry that has never been populated, so callers can try this first
    and fall through to their existing conversion path.
    """
    registry = getattr(context.backend, "tables", None)
    if registry is None:
        return None
    return registry.relation(label, kind)


def _base_relation(context: Context, label: str, con: Any) -> Any:
    """Return a lazy DuckDB relation for *label*'s source rows.

    Prefers a registered streaming relation (file-backed, out-of-core), then
    any table the registry has materialised for this label, and only then
    falls back to converting the entity's in-memory ``source_obj`` — which
    is a fresh conversion on *every* query, so it is the last resort.
    """
    from pycypher.backends.table_registry import ENTITY_KIND

    streaming = context._streaming_sources
    if label in streaming:
        return streaming[label][0].relation
    registered = _registered_relation(context, label, ENTITY_KIND)
    if registered is not None:
        return registered
    entity = context.entity_mapping.mapping[label]
    src = entity.source_obj
    import pandas as pd

    if isinstance(src, pd.DataFrame):
        return con.from_df(src)
    try:
        import pyarrow as pa

        if isinstance(src, pa.Table):
            return con.from_arrow(src)
    except ImportError:
        pass
    from pycypher.backends._helpers import _to_pandas

    return con.from_df(_to_pandas(src))


def _rel_attr_map(context: Context, label: str) -> dict[str, str] | None:
    """Property→column map for a relationship *label*.

    Prefers the table registry (populated by
    :func:`register_streaming_relationship` or, for an in-memory source,
    :func:`~pycypher.backends.table_registry.register_context_tables`) —
    mirroring how :func:`_entity_attr_map` prefers
    ``context._streaming_sources`` — before falling back to
    ``context.relationship_mapping`` for a relationship that was never
    registered into the registry at all (e.g. a non-DuckDB backend).
    """
    from pycypher.backends.table_registry import RELATIONSHIP_KIND

    registry = getattr(context.backend, "tables", None)
    if registry is not None:
        entry = registry.get(label, RELATIONSHIP_KIND)
        if entry is not None:
            return entry.attr_map
    rel = context.relationship_mapping.mapping.get(label)
    return rel.attribute_map if rel is not None else None


def _rel_base_relation(context: Context, label: str, con: Any) -> Any:
    """Return a DuckDB relation over a relationship's source rows.

    Prefers a table the registry has materialised; otherwise converts the
    in-memory ``source_obj`` afresh, as before.
    """
    from pycypher.backends.table_registry import RELATIONSHIP_KIND

    registered = _registered_relation(context, label, RELATIONSHIP_KIND)
    if registered is not None:
        return registered
    rel = context.relationship_mapping.mapping[label]
    src = rel.source_obj
    import pandas as pd

    if isinstance(src, pd.DataFrame):
        return con.from_df(src)
    try:
        import pyarrow as pa

        if isinstance(src, pa.Table):
            return con.from_arrow(src)
    except ImportError:
        pass
    from pycypher.backends._helpers import _to_pandas

    return con.from_df(_to_pandas(src))


def _edge_endpoint_predicates(
    edge_rel: Any,
    edge_alias: str,
    source_label: str | None,
    target_label: str | None,
) -> list[str]:
    """SQL predicates restricting *edge_rel* to edges whose *declared*
    endpoint labels are compatible with the pattern's node labels.

    An edge declared (via ``source_entity_type``/``target_entity_type`` at
    registration) for a different label is excluded; an edge with no
    declaration (NULL) matches any label. Returns ``[]`` when the relation
    has no label columns at all (an in-memory ``source_obj`` fallback, or a
    table registered by another path), so those keep their old behaviour.
    Every predicate references only *edge_alias*, so it is safe to fold
    into the edge join's ON condition — which is where it must go for a
    LEFT join (OPTIONAL MATCH) to keep its unmatched rows.
    """
    from pycypher.ingestion.security import escape_sql_string_literal

    columns = set(edge_rel.columns)
    preds: list[str] = []
    for label, column in (
        (source_label, SOURCE_LABEL_COLUMN),
        (target_label, TARGET_LABEL_COLUMN),
    ):
        if label is None or column not in columns:
            continue
        col = f'{edge_alias}."{column}"'
        preds.append(
            f"({col} IS NULL OR {col} = {escape_sql_string_literal(label)})"
        )
    return preds


def _new_column_name_allowed(prop: str) -> bool:
    """True if *prop* is safe to create as a brand-new column via
    :func:`_ensure_column` (Phase 3a,
    the FastOpenData streaming-qualification plan (private repository)).

    Used only to relax a ``SET`` *target* property's eligibility check
    after the normal ``resolve`` lookup against the existing attr_map has
    already failed — read-side property references must always already
    resolve; this is never a fallback for those.
    """
    from pycypher.ingestion.security import (
        SecurityError,
        sanitize_sql_identifier,
    )

    try:
        sanitize_sql_identifier(prop)
    except SecurityError:
        return False
    return True


def _ensure_column(
    context: Context,
    label: str,
    property_name: str,
    type_probe_relation: Any,
) -> None:
    """Ensure *label*'s registered table has a column for *property_name*,
    creating it if needed (Phase 3a).

    A no-op when the property already exists (the common case). Otherwise
    runs ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` — idempotent, and
    existing rows get ``NULL`` for the new column, matching the semantics
    every ``SET``-family executor's "unmatched rows left untouched"
    behaviour already assumes. The column's type is read off
    *type_probe_relation*'s own ``.types`` (a static DuckDB relation
    property, computed without executing or fetching any rows) rather than
    guessed, so it always matches what the real ``UPDATE`` would have
    produced anyway. The property name is used as the column name
    directly, matching ``TableRegistry.register_relation``'s identity-map
    convention for every entity in this pipeline.

    Mutates the registry entry's ``attr_map`` dict in place — safe despite
    ``RegisteredTable`` being a frozen dataclass (frozen only blocks
    reassigning the *field*, not mutating the dict it already points to).
    Every caller in this module resolves a label's attr_map via
    :func:`_entity_attr_map`, which always returns that same dict object
    (from ``context._streaming_sources`` or the table registry, never a
    copy), so this mutation is immediately visible to a `resolve`/
    dict-lookup closure built before this call, with no need to rebuild it.
    """
    from pycypher.backends.table_registry import (
        ENTITY_KIND,
        physical_table_name,
    )

    attr = _entity_attr_map(context, label)
    if attr is not None and property_name in attr:
        return

    con = context.backend.connection
    table = physical_table_name(label)
    col_type = str(type_probe_relation.types[0])
    con.execute(
        f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{property_name}" {col_type}'
    )  # nosec B608 — table/column validated identifiers (property_name already passed sanitize_sql_identifier at the eligibility check); type read from DuckDB's own static schema inference, not user input

    registry = getattr(context.backend, "tables", None)
    entry = registry.get(label, ENTITY_KIND) if registry is not None else None
    if entry is not None:
        entry.attr_map[property_name] = property_name
    streaming = context._streaming_sources.get(label)
    if streaming is not None:
        streaming[1][property_name] = property_name


#: DuckDB integer type names for which sequence-based ``MAX(id)+1`` ID
#: generation is well-defined.
_INTEGER_DUCKDB_TYPES = frozenset(
    {
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
    }
)


def _streaming_id_col(context: Context, label: str) -> str | None:
    """Return the registered ID column for *label*'s streaming source, or
    ``None`` if there is no streaming source or it has no ``id_col``.
    """
    entry = context._streaming_sources.get(label)
    return entry[2] if entry is not None else None


def _node_id_column(context: Context, label: str) -> str:
    """Return the physical column name that identifies a node of *label*.

    The eager Arrow path (``normalize_entity_table``) always renames the
    configured ``id_col`` to a literal ``__ID__`` column before an entity
    ever reaches a table. A streaming entity (``register_streaming_source``)
    does not — it registers the raw file's columns unchanged and records
    which one is the id as metadata instead of renaming it, so join SQL
    that assumes a literal ``"__ID__"`` column breaks for any streaming
    entity whose configured ``id_col`` isn't literally named ``__ID__``
    (true for almost every real source — e.g. this pipeline's ``PUMA``
    entity keys on ``PUMA_FIPS``, not ``__ID__``). This resolves the real
    column name so join-condition builders work for both entity kinds.
    """
    streaming_id = _streaming_id_col(context, label)
    return streaming_id if streaming_id is not None else "__ID__"


def _streaming_id_is_integer(context: Context, label: str) -> bool:
    """True if *label*'s streaming ``id_col`` (if any) has an integer type.

    ``True`` when there is no ``id_col`` to validate (nothing to check).
    Sequence-based ``nextval()`` ID generation only produces integers, so a
    non-integer ID column makes ``CREATE`` ineligible for this slice.
    """
    id_col = _streaming_id_col(context, label)
    if id_col is None:
        return True
    materialized = context._streaming_sources[label][0]
    relation = materialized.relation
    try:
        idx = relation.columns.index(id_col)
    except ValueError:
        return False
    return str(relation.types[idx]).upper() in _INTEGER_DUCKDB_TYPES


# ---------------------------------------------------------------------------
# Eligibility and execution via the logical plan (pycypher.plan)
# ---------------------------------------------------------------------------


def _translate(query: Any, context: Context) -> Any | None:
    """Translate *query* to a plan, or ``None`` if a construct is unsupported."""
    from pycypher.plan import Unsupported, translate

    try:
        return translate(query, context)
    except Unsupported as exc:
        LOGGER.debug("[duckdb-plan] ineligible: %s", exc)
        return None


def _ends_with_return(query: Any) -> bool:
    from pycypher.ast_models import Query, Return

    return (
        isinstance(query, Query)
        and bool(query.clauses)
        and isinstance(query.clauses[-1], Return)
    )


def is_relation_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a read (``RETURN``-terminated) query the
    relation engine can run natively.

    Eligibility is successful translation by :func:`pycypher.plan.translate`
    — there is no separate whitelist of query shapes. A query is ineligible
    exactly when it uses a construct with no translation rule (see
    :class:`pycypher.plan.Unsupported`), and the reason is logged at DEBUG.
    """
    if not _ends_with_return(query):
        return False
    return _translate(query, context) is not None


def execute_relation_query(
    query: Any,
    context: Context,
    *,
    materialize: bool = True,
) -> pd.DataFrame | RelationBindings:
    """Execute an eligible read query as DuckDB SQL.

    Precondition: :func:`is_relation_eligible` returned ``True``. Any ``SET``
    stage inside the query runs as native DML before the stages after it
    (see :class:`pycypher.plan.emit_duckdb.Emitter`).

    Args:
        query: The parsed, eligible query AST.
        context: The DuckDB-backed context.
        materialize: When ``True`` (default) return a pandas DataFrame; when
            ``False`` return a :class:`RelationBindings` for streaming to a sink.

    """
    from pycypher.backends.duckdb_backend import DuckDBLazyFrame
    from pycypher.plan import translate
    from pycypher.plan.emit_duckdb import Emitter

    plan = translate(query, context)
    con = context.backend.connection
    rel = Emitter(context).run(plan)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("[duckdb-relation] %s", rel.sql_query())
    bindings = RelationBindings(DuckDBLazyFrame(rel, con))
    return bindings.to_pandas() if materialize else bindings


def _classify_mutation(query: Any) -> str:
    """Name a mutation-only query's shape for progress output and the
    per-kind compatibility wrappers below (``set``/``scalar_set``/
    ``group_set``/``copy_set``/``create``/``delete``). Purely descriptive:
    every kind translates and executes through the same plan rules.
    """
    from pycypher.ast_models import Create, Delete, Match, Set, With
    from pycypher.relation_sql import is_aggregate

    clauses = list(query.clauses)
    last = clauses[-1]
    if isinstance(last, Create):
        return "create"
    if isinstance(last, Delete):
        return "delete"
    if not isinstance(last, Set):
        return "mutation"
    withs = [c for c in clauses if isinstance(c, With)]
    if any(is_aggregate(it.expression) for w in withs for it in w.items):
        return "group_set"
    match = clauses[0] if isinstance(clauses[0], Match) else None
    multi_node = (
        match is not None
        and match.pattern is not None
        and len(match.pattern.paths) == 1
        and len(match.pattern.paths[0].elements) > 1
    )
    if multi_node:
        return "copy_set"
    return "scalar_set" if withs else "set"


def is_relation_mutation_eligible(query: Any, context: Context) -> str | None:
    """Return the mutation kind name if *query* is a mutation-only query
    (no ``RETURN``) the relation engine can execute natively, else ``None``.
    """
    from pycypher.plan.nodes import has_side_effects

    if _ends_with_return(query):
        return None
    plan = _translate(query, context)
    if plan is None or not has_side_effects(plan):
        return None
    return _classify_mutation(query)


def execute_relation_mutation(
    query: Any, context: Context, kind: str | None = None
) -> None:  # noqa: ARG001 — kind kept for call-site compatibility
    """Execute an eligible mutation-only query as native DML.

    Precondition: :func:`is_relation_mutation_eligible` returned a kind.
    """
    from pycypher.plan import translate
    from pycypher.plan.emit_duckdb import Emitter

    Emitter(context).run(translate(query, context))


# Per-kind names kept for existing call sites and tests; all route through
# the same plan-based eligibility and execution.


def is_relation_set_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"set"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "set"


def is_relation_scalar_set_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"scalar_set"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "scalar_set"


def is_relation_group_set_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"group_set"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "group_set"


def is_relation_copy_set_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"copy_set"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "copy_set"


def is_relation_create_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"create"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "create"


def is_relation_delete_eligible(query: Any, context: Context) -> bool:
    """True if the mutation kind is ``"delete"`` (see :func:`is_relation_mutation_eligible`)."""
    return is_relation_mutation_eligible(query, context) == "delete"


execute_relation_set = execute_relation_mutation
execute_relation_scalar_set = execute_relation_mutation
execute_relation_group_set = execute_relation_mutation
execute_relation_copy_set = execute_relation_mutation
execute_relation_create = execute_relation_mutation
execute_relation_delete = execute_relation_mutation
