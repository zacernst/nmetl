"""Out-of-core relation execution path (Approach A, Phase 4+).

A separate, **opt-in** execution path that runs an *eligible* subset of Cypher
queries directly as DuckDB relations instead of the pandas ``BindingFrame``
engine.  Anything not eligible falls back to the existing engine, so coverage
grows one query-feature at a time while the suite stays green.

Disabled by default — enabled per-:class:`Context` via a truthy
``_relation_engine_enabled`` attribute or the
``PYCYPHER_DUCKDB_RELATION_ENGINE`` environment variable.  When disabled the
dispatch never fires, guaranteeing zero behaviour change.

Eligible subset so far: a required leading ``MATCH`` (single node or a
fixed-length directed path of one or more hops) with optional inline node
properties; zero or more ``OPTIONAL MATCH`` LEFT-join extensions from a bound
node; an optional ``WHERE`` (compiled to a SQL predicate via
:mod:`pycypher.relation_sql`); zero or more ``WITH`` stages; and a ``RETURN`` of
compilable expressions (property lookups, ``id()``/``elementId()``,
arithmetic, literals, registered scalar UDFs, and
``count/sum/avg/min/max`` aggregates with implicit GROUP BY),
plus DISTINCT / ORDER BY / SKIP+LIMIT.  Also: a leading ``UNWIND`` of a list, a
leading ``WITH`` of constants, and ``UNWIND`` of a scalar list column in a
``WITH`` stage.  Duplicate output column names are rejected.  Not yet:
undirected / variable-length paths, a second required MATCH, OPTIONAL MATCH
combined with aggregation, ``UNWIND`` in pattern scope (right after MATCH),
``collect()``, and unregistered functions.  See
``docs/duckdb_full_parity_design.md``.

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
) -> str:
    """Build the SQL that reshapes a raw relationship scan into
    ``__ID__``/``__SOURCE__``/``__TARGET__`` form.

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

    other_cols = [
        c for c in columns if c not in (source_col, target_col, id_col)
    ]

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
    final_cols = ["__SOURCE__", "__TARGET__"] + [
        _quote_source_identifier(c) for c in other_cols
    ]
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
) -> None:
    """Register a file-backed relationship as a streaming DuckDB relation.

    The relationship counterpart to :func:`register_streaming_source` — see
    ``docs/fastopendata_streaming_qualification_plan.md``, "Phase 1". Reads
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
    )
    transformed = raw.query(view_name, sql)
    # __SOURCE__/__TARGET__ are reserved structural columns, not properties —
    # register_relation's default attr_map (every non-id_col column) would
    # otherwise expose them as Cypher properties, unlike the eager path's
    # infer_attribute_map(), which excludes them explicitly.
    attr_map = {
        c: c
        for c in transformed.columns
        if c not in ("__ID__", "__SOURCE__", "__TARGET__")
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
    """Property→column map for *label* from a streaming source or EntityTable."""
    streaming = context._streaming_sources
    if label in streaming:
        return streaming[label][1]
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


def _make_resolve(variables: dict[str, tuple[str, dict[str, str]]]) -> Any:
    """Build a ``resolve(var, prop)`` closure over the pattern's variables.

    *variables* maps each bound variable to ``(sql_alias, attr_map)``.  An empty
    alias references the column unqualified (single-relation case); otherwise it
    is qualified as ``alias."col"`` (joins).
    """
    from pycypher.ingestion.security import sanitize_sql_identifier

    def resolve(var: str, prop: str) -> str | None:
        entry = variables.get(var)
        if entry is None:
            return None
        alias, attr = entry
        col = attr.get(prop)
        if col is None:
            return None
        quoted = f'"{sanitize_sql_identifier(col)}"'
        return f"{alias}.{quoted}" if alias else quoted

    return resolve


def _valid_node(node: Any) -> bool:
    """True if *node* is a single-label variable node.

    Inline properties (``{prop: value}``) are allowed — they are desugared into
    equality predicates during pattern analysis.
    """
    from pycypher.ast_models import NodePattern

    return (
        isinstance(node, NodePattern)
        and node.variable is not None
        and len(node.labels) == 1
    )


def _inline_predicates(nodes: list[Any]) -> list[tuple[str, str, Any]]:
    """Collect ``(var, property, value_ast)`` triples from nodes' inline props."""
    preds: list[tuple[str, str, Any]] = []
    for node in nodes:
        for prop, value in (node.properties or {}).items():
            preds.append((node.variable.name, prop, value))
    return preds


def _compile_inline_predicates(
    inline_preds: list[tuple[str, str, Any]],
    resolve: Any,
    udfs: frozenset[str],
) -> list[str] | None:
    """Compile inline-property predicates to ``col = value`` SQL, or ``None``."""
    from pycypher.relation_sql import compile_expression

    out: list[str] = []
    for var, prop, value_ast in inline_preds:
        col = resolve(var, prop)
        if col is None:
            return None
        val_sql = compile_expression(value_ast, resolve, udfs)
        if val_sql is None:
            return None
        out.append(f"{col} = {val_sql}")
    return out


class _Plan:
    """A resolved, eligible relation-query plan: a base + pipeline stages."""

    __slots__ = (
        "alias_gen",
        "build",
        "initial_scope",
        "inline_preds",
        "match_where",
        "stages",
        "unwind_expr",
    )

    def __init__(
        self,
        initial_scope: _Scope,
        build: Any,
        match_where: Any,
        stages: list[Any],
        inline_preds: list[tuple[str, str, Any]],
        unwind_expr: Any = None,
        alias_gen: Any = None,
    ) -> None:
        self.initial_scope = initial_scope  # scope over the base relation
        self.build = build  # build(con) -> DuckDBPyRelation
        self.match_where = match_where  # WHERE from the MATCH clause | None
        self.stages = stages  # [With | Unwind, …, Return]
        self.inline_preds = (
            inline_preds  # (var, prop, value_ast) equality preds
        )
        self.unwind_expr = unwind_expr  # list expr of a leading UNWIND | None
        # Shared alias counter, reused for any embedded second MATCH so its
        # aliases can never collide with the leading pattern's (DuckDB raises
        # "Ambiguous reference to table" if two crossed relations reuse an alias).
        self.alias_gen = alias_gen


class _Scope:
    """Name resolution for one pipeline stage."""

    __slots__ = (
        "names",
        "node_vars",
        "pattern_vars",
        "qualified",
        "resolve",
        "resolve_var",
        "var_labels",
    )

    def __init__(
        self,
        resolve: Any,
        resolve_var: Any,
        *,
        qualified: bool,
        node_vars: frozenset[str],
        names: frozenset[str] | None = None,
        pattern_vars: dict[str, tuple[str, Any]] | None = None,
        var_labels: dict[str, str] | None = None,
    ) -> None:
        self.resolve = resolve  # (var, prop) -> col | None
        self.resolve_var = resolve_var  # (name) -> col | None (bare scalars)
        self.qualified = qualified
        self.node_vars = node_vars
        self.names = names  # scalar column names (for UNWIND '*'), None in pattern scope
        # (alias, attr_map) per still-fully-in-scope node var -- the same
        # shape _make_resolve() consumes, exposed directly (not just via the
        # resolve() closure) so a mid-pipeline SET stage / a WITH item that
        # passes a node through *alongside* new named expressions can build
        # its own projection SQL (e.g. "alias.* EXCLUDE (...)") without
        # re-deriving what resolve() already knows. Empty for a purely
        # scalar (post-aggregating-WITH) scope.
        self.pattern_vars = pattern_vars or {}
        # var -> entity label, for the node vars in pattern_vars that came
        # from a real MATCH (not a relationship var) -- needed by a
        # mid-pipeline SET stage to find the variable's physical table
        # (_node_id_column/physical_table_name/_streaming_sources all key
        # off the label, which resolve()'s closure doesn't expose).
        self.var_labels = var_labels or {}


def _no_prop(_var: str, _prop: str) -> None:
    return None


def _no_var(_name: str) -> None:
    return None


def _scalar_scope(output_names: list[str]) -> _Scope:
    """A scope where bare variables resolve to a prior stage's output columns."""
    names = frozenset(output_names)

    def resolve_var(name: str) -> str | None:
        return _quote_output_alias(name) if name in names else None

    return _Scope(
        _no_prop,
        resolve_var,
        qualified=False,
        node_vars=frozenset(),
        names=names,
    )


def _analyze_leading_pattern(
    match: Any,
    context: Context,
    alias_gen: Any,
) -> (
    tuple[
        dict[str, tuple[str, dict[str, str]]],
        Any,
        list[tuple[str, str, Any]],
        dict[str, str],
    ]
    | None
):
    """Analyse the required leading MATCH pattern.

    Returns ``(variables, build, inline_preds, var_labels)`` — variables maps
    each bound Cypher variable to ``(sql_alias, attr_map)``, build(con)
    returns the pattern's relation, inline_preds are the nodes'
    inline-property equalities, and var_labels maps each *node* variable
    (not relationship variables) to its entity label — needed by a
    mid-pipeline ``SET`` stage or a mixed node-passthrough ``WITH`` item to
    find the variable's physical table (see ``_Scope.var_labels``).
    Aliases come from *alias_gen* so they are unique across the whole query.
    """
    from collections import ChainMap

    from pycypher.ast_models import RelationshipDirection, RelationshipPattern
    from pycypher.relation_sql import ID_SENTINEL

    if match.optional:
        return None
    paths = match.pattern.paths
    if len(paths) != 1:
        return None
    path = paths[0]
    if path.variable is not None:
        return None
    if getattr(path, "shortest_path_mode", "none") not in ("none", None):
        return None
    elements = path.elements

    # --- Single node ---
    if len(elements) == 1:
        node = elements[0]
        if not _valid_node(node):
            return None
        attr = _entity_attr_map(context, node.labels[0])
        if attr is None:
            return None
        alias = alias_gen()
        label = node.labels[0]
        # ChainMap, not a copy: attr must stay the *same* live dict object
        # _ensure_column() mutates in place (see its docstring) — a copy here
        # would silently desync from a later ALTER-TABLE-added column.
        id_attr = ChainMap(
            {ID_SENTINEL: _node_id_column(context, label)}, attr
        )
        variables = {node.variable.name: (alias, id_attr)}
        var_labels = {node.variable.name: label}

        def build(con: Any, label: str = label, alias: str = alias) -> Any:
            return _base_relation(context, label, con).set_alias(alias)

        return variables, build, _inline_predicates([node]), var_labels

    # --- Fixed-length directed path (one or more hops) ---
    if len(elements) >= 3 and len(elements) % 2 == 1:
        nodes = elements[0::2]
        rels = elements[1::2]
        if not all(_valid_node(nd) for nd in nodes):
            return None
        for rp in rels:
            if not isinstance(rp, RelationshipPattern):
                return None
            if rp.length is not None or getattr(rp, "properties", None):
                return None
            if len(rp.labels) != 1:
                return None
            if rp.direction not in (
                RelationshipDirection.RIGHT,
                RelationshipDirection.LEFT,
            ):
                return None

        node_attrs = [_entity_attr_map(context, nd.labels[0]) for nd in nodes]
        rel_attrs = [_rel_attr_map(context, rp.labels[0]) for rp in rels]
        if any(a is None for a in node_attrs) or any(
            a is None for a in rel_attrs
        ):
            return None

        node_aliases = [alias_gen() for _ in nodes]
        rel_aliases = [alias_gen() for _ in rels]
        variables = {}
        for i, nd in enumerate(nodes):
            # ChainMap, not a copy — see the single-node branch above.
            id_attr = ChainMap(
                {ID_SENTINEL: _node_id_column(context, nd.labels[0])},
                node_attrs[i],
            )
            variables[nd.variable.name] = (node_aliases[i], id_attr)
        for j, rp in enumerate(rels):
            if rp.variable is not None:
                variables[rp.variable.name] = (rel_aliases[j], rel_attrs[j])
        n_named = len(nodes) + sum(1 for rp in rels if rp.variable is not None)
        if len(variables) != n_named:
            return None

        node_labels = [nd.labels[0] for nd in nodes]
        rel_labels = [rp.labels[0] for rp in rels]
        rights = [rp.direction == RelationshipDirection.RIGHT for rp in rels]
        var_labels = {nd.variable.name: nd.labels[0] for nd in nodes}

        def build(
            con: Any,
            node_labels: list[str] = node_labels,
            rel_labels: list[str] = rel_labels,
            rights: list[bool] = rights,
            node_aliases: list[str] = node_aliases,
            rel_aliases: list[str] = rel_aliases,
        ) -> Any:
            node_rels = [
                _base_relation(context, lbl, con).set_alias(al)
                for lbl, al in zip(node_labels, node_aliases, strict=True)
            ]
            rel_rels = [
                _rel_base_relation(context, lbl, con).set_alias(al)
                for lbl, al in zip(rel_labels, rel_aliases, strict=True)
            ]
            acc = node_rels[0]
            for j, right in enumerate(rights):
                na, nb, ea = (
                    node_aliases[j],
                    node_aliases[j + 1],
                    rel_aliases[j],
                )
                na_id = _node_id_column(context, node_labels[j])
                nb_id = _node_id_column(context, node_labels[j + 1])
                if right:
                    c1 = f'{na}."{na_id}" = {ea}."__SOURCE__"'
                    c2 = f'{ea}."__TARGET__" = {nb}."{nb_id}"'
                else:
                    c1 = f'{na}."{na_id}" = {ea}."__TARGET__"'
                    c2 = f'{ea}."__SOURCE__" = {nb}."{nb_id}"'
                acc = acc.join(rel_rels[j], c1).join(node_rels[j + 1], c2)
            return acc

        return variables, build, _inline_predicates(nodes), var_labels

    return None


def _analyze_optional_pattern(
    bound: dict[str, tuple[str, dict[str, str]]],
    context: Context,
    opt_match: Any,
    alias_gen: Any,
) -> tuple[dict[str, tuple[str, dict[str, str]]], Any, dict[str, str]] | None:
    """Analyse one OPTIONAL MATCH as a LEFT-join extension.

    Supports a single directed relationship ``(x)-[e]->(y)`` / ``(x)<-[e]-(y)``
    where the left node *x* is already bound and the right node *y* (and an
    optional relationship variable) is new.  Returns ``(new_variables,
    extend, var_labels)`` where ``extend(con, base_rel)`` LEFT-joins the hop
    onto *base_rel*, and ``var_labels`` maps *y* (only — *x*'s label is
    already known to the caller from when it was bound; the relationship
    variable, if any, has no node label) to its entity label.
    """
    from collections import ChainMap

    from pycypher.ast_models import (
        NodePattern,
        RelationshipDirection,
        RelationshipPattern,
    )
    from pycypher.relation_sql import ID_SENTINEL

    if opt_match.where is not None:
        return None  # WHERE on an optional pattern would need join-condition placement
    paths = opt_match.pattern.paths
    if len(paths) != 1:
        return None
    path = paths[0]
    if path.variable is not None:
        return None
    if getattr(path, "shortest_path_mode", "none") not in ("none", None):
        return None
    elements = path.elements
    if len(elements) != 3:
        return None
    n_left, rp, n_right = elements
    # The left node is already bound: referenced by variable, its label is
    # optional (and ignored).  The right node is new and needs a single label.
    if not (isinstance(n_left, NodePattern) and n_left.variable is not None):
        return None
    if not _valid_node(n_right):
        return None
    if getattr(n_left, "properties", None) or getattr(
        n_right, "properties", None
    ):
        return None  # inline props on an optional pattern not supported
    if not isinstance(rp, RelationshipPattern):
        return None
    if rp.length is not None or getattr(rp, "properties", None):
        return None
    if len(rp.labels) != 1:
        return None
    if rp.direction not in (
        RelationshipDirection.RIGHT,
        RelationshipDirection.LEFT,
    ):
        return None

    x_var, y_var = n_left.variable.name, n_right.variable.name
    if x_var not in bound or y_var in bound:
        return None  # left must be bound, right must be new
    rel_attr = _rel_attr_map(context, rp.labels[0])
    y_attr = _entity_attr_map(context, n_right.labels[0])
    if rel_attr is None or y_attr is None:
        return None

    x_alias = bound[x_var][0]
    y_alias, e_alias = alias_gen(), alias_gen()
    # ChainMap, not a copy — see _analyze_leading_pattern's single-node branch.
    y_attr = ChainMap(
        {ID_SENTINEL: _node_id_column(context, n_right.labels[0])}, y_attr
    )
    new_vars: dict[str, tuple[str, dict[str, str]]] = {
        y_var: (y_alias, y_attr)
    }
    if rp.variable is not None:
        rv = rp.variable.name
        if rv in bound or rv == y_var:
            return None
        new_vars[rv] = (e_alias, rel_attr)

    right = rp.direction == RelationshipDirection.RIGHT
    y_label, e_label = n_right.labels[0], rp.labels[0]

    def extend(
        con: Any,
        base_rel: Any,
        x_alias: str = x_alias,
        y_alias: str = y_alias,
        e_alias: str = e_alias,
        y_label: str = y_label,
        e_label: str = e_label,
        right: bool = right,  # noqa: FBT001
    ) -> Any:
        # NOTE: the left node's id column is assumed to be literally
        # "__ID__" here, unlike the fixed-length-path builder's use of
        # _node_id_column() — x's entity label is "optional and ignored"
        # for an already-bound OPTIONAL MATCH left node (see the docstring
        # above), so there is no label to resolve a streaming entity's real
        # id column from. An OPTIONAL MATCH extending from a streaming
        # entity whose id_col isn't literally "__ID__" will therefore still
        # raise a BinderException here — a known, narrower-scoped gap left
        # for a follow-up (see docs/fastopendata_streaming_qualification_plan.md,
        # Phase 1) rather than this pass.
        e_rel = _rel_base_relation(context, e_label, con).set_alias(e_alias)
        y_rel = _base_relation(context, y_label, con).set_alias(y_alias)
        y_id = _node_id_column(context, y_label)
        if right:
            c1 = f'{x_alias}."__ID__" = {e_alias}."__SOURCE__"'
            c2 = f'{e_alias}."__TARGET__" = {y_alias}."{y_id}"'
        else:
            c1 = f'{x_alias}."__ID__" = {e_alias}."__TARGET__"'
            c2 = f'{e_alias}."__SOURCE__" = {y_alias}."{y_id}"'
        return base_rel.join(e_rel, c1, how="left").join(y_rel, c2, how="left")

    return new_vars, extend, {y_var: y_label}


def _compose_build(base_build: Any, extend: Any) -> Any:
    """Return a build that applies *extend* to *base_build*'s relation."""

    def composed(
        con: Any, base_build: Any = base_build, extend: Any = extend
    ) -> Any:
        return extend(con, base_build(con))

    return composed


def _leading_unwind_build(var: str, list_expr: Any) -> Any:
    """Build for a leading ``UNWIND <list> AS var`` (base = the unnested list)."""

    def build(con: Any, var: str = var, list_expr: Any = list_expr) -> Any:
        from pycypher.relation_sql import compile_expression

        list_sql = compile_expression(list_expr, _no_prop, resolve_var=_no_var)
        return con.sql(
            f"SELECT UNNEST({list_sql}) AS {_quote_output_alias(var)}"
        )  # nosec B608 — list_sql from internal AST compiler, var safely double-quoted via _quote_output_alias

    return build


def _analyze_query(query: Any, context: Context) -> _Plan | None:
    """Return a pipeline plan for *query* if eligible, else ``None``.

    Shape: either a required leading ``MATCH`` (then zero or more ``OPTIONAL
    MATCH`` LEFT-join extensions), or a leading ``UNWIND`` of a list; followed by
    zero or more ``WITH``/``UNWIND``/``SET`` stages, optionally including exactly
    one additional required ``MATCH`` immediately after a ``WITH`` (a
    cross-joined multi-pattern query); ending in ``RETURN``.  Any other
    ``MATCH`` after the pattern phase is not supported. A ``SET`` stage
    (Phase 2b category (E) — docs/fastopendata_streaming_qualification_plan.md)
    executes as a native ``UPDATE`` side effect against the target's
    physical table when reached, then folds its computed values into the
    in-flight relation so later stages read them like any other property —
    see :func:`_plan_stage`'s ``Set``-stage handling and
    :func:`_compile_set_stage`.
    """
    from pycypher.ast_models import Match, Query, Return, Set, Unwind, With

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) < 2 or not isinstance(clauses[-1], Return):
        return None

    counter = [0]

    def alias_gen() -> str:
        alias = f"v{counter[0]}"
        counter[0] += 1
        return alias

    def _valid_stages(stages: list[Any]) -> bool:
        # Middle stages (all but the final RETURN) must be WITH, UNWIND, or
        # SET, except for at most one embedded MATCH: non-optional, not
        # first, and immediately preceded by a WITH (a cross-joined second
        # pattern). Unlike the embedded MATCH, a SET stage has no
        # "preceded by WITH" requirement — it can follow the leading MATCH
        # directly (e.g. decode_tags-shaped: MATCH ... SET ... RETURN ...)
        # or any WITH; its own eligibility (single target var, compilable
        # values) is checked later in is_relation_eligible's stage loop.
        non_terminal = stages[:-1]
        match_idxs = [
            i for i, c in enumerate(non_terminal) if isinstance(c, Match)
        ]
        if len(match_idxs) > 1:
            return False
        if match_idxs:
            i = match_idxs[0]
            if (
                i == 0
                or not isinstance(non_terminal[i - 1], With)
                or non_terminal[i].optional
            ):
                return False
        skip = match_idxs[0] if match_idxs else -1
        return all(
            isinstance(c, (With, Unwind, Set))
            for j, c in enumerate(non_terminal)
            if j != skip
        )

    # --- Leading UNWIND of a list ---
    if isinstance(clauses[0], Unwind):
        uw = clauses[0]
        if uw.alias is None:
            return None
        stages = list(clauses[1:])
        if not _valid_stages(stages):
            return None
        return _Plan(
            _scalar_scope([uw.alias]),
            _leading_unwind_build(uw.alias, uw.expression),
            None,
            stages,
            [],
            unwind_expr=uw.expression,
            alias_gen=alias_gen,
        )

    # --- Leading WITH of constants (no source) → single-row base ---
    if isinstance(clauses[0], With):
        stages = list(clauses)
        if not _valid_stages(stages):
            return None

        def build(con: Any) -> Any:
            return con.sql("SELECT 1 AS __unit")

        return _Plan(
            _scalar_scope([]), build, None, stages, [], alias_gen=alias_gen
        )

    # --- Leading MATCH pattern (+ optional matches) ---
    if not isinstance(clauses[0], Match) or clauses[0].optional:
        return None

    idx = 1
    opt_matches: list[Any] = []
    while idx < len(clauses) - 1 and isinstance(clauses[idx], Match):
        if not clauses[idx].optional:
            return None  # a second required MATCH is not supported
        opt_matches.append(clauses[idx])
        idx += 1
    stages = list(clauses[idx:])
    if not _valid_stages(stages):
        return None

    # Aggregating over an OPTIONAL pattern is not supported: count(<optional
    # node>) must count non-null matches, but the engine's count(node) →
    # COUNT(*) shortcut would over-count.  Fall back for correctness.
    if opt_matches:
        from pycypher.relation_sql import is_aggregate

        if any(
            is_aggregate(it.expression)
            for stage in stages
            for it in getattr(stage, "items", [])
        ):
            return None

    lead = _analyze_leading_pattern(clauses[0], context, alias_gen)
    if lead is None:
        return None
    variables, build, inline_preds, var_labels = lead

    for opt_match in opt_matches:
        ext = _analyze_optional_pattern(
            variables, context, opt_match, alias_gen
        )
        if ext is None:
            return None
        new_vars, extend, new_var_labels = ext
        variables = {**variables, **new_vars}
        var_labels = {**var_labels, **new_var_labels}
        build = _compose_build(build, extend)

    initial_scope = _Scope(
        _make_resolve(variables),
        _no_var,
        qualified=len(variables) > 1,
        node_vars=frozenset(variables),
        pattern_vars=variables,
        var_labels=var_labels,
    )
    return _Plan(
        initial_scope,
        build,
        clauses[0].where,
        stages,
        inline_preds,
        alias_gen=alias_gen,
    )


def _analyze_second_match(
    prior_scope: _Scope,
    match: Any,
    context: Context,
    alias_gen: Any,
) -> tuple[_Scope, Any, Any, list[tuple[str, str, Any]], str] | None:
    """Analyse a ``MATCH`` embedded after a ``WITH`` (a cross-joined pattern).

    Returns ``(new_scope, build, where, inline_preds, acc_alias)``. *build*
    is the second pattern's own relation builder (reused verbatim from
    :func:`_analyze_leading_pattern`). *new_scope* resolves both the new
    pattern's variables and the prior stage's scalar outputs, the latter
    qualified against *acc_alias* — the alias the caller must
    ``set_alias()`` on the accumulated relation before crossing, so that
    column names shared between the two sides of the cross join never
    collide. *alias_gen* must be the plan's shared counter (not a fresh
    one), or the second pattern's aliases could collide with the leading
    pattern's, which DuckDB rejects as an ambiguous table reference.
    """
    if match.optional:
        return None
    lead = _analyze_leading_pattern(match, context, alias_gen)
    if lead is None:
        return None
    new_vars, build, inline_preds, var_labels = lead
    prior_names = prior_scope.names or frozenset()
    if prior_names & new_vars.keys():
        return None  # name collision between WITH output and new pattern var
    acc_alias = alias_gen()
    resolve = _make_resolve(new_vars)

    def resolve_var(name: str) -> str | None:
        return (
            f"{acc_alias}.{_quote_output_alias(name)}"
            if name in prior_names
            else None
        )

    new_scope = _Scope(
        resolve,
        resolve_var,
        qualified=True,
        node_vars=frozenset(new_vars),
        pattern_vars=new_vars,
        var_labels=var_labels,
    )
    return new_scope, build, match.where, inline_preds, acc_alias


def _quote_output_alias(name: str) -> str:
    """Quote *name* as a DuckDB output identifier (safe for dots etc.).

    Output aliases can legitimately contain dots (e.g. the pandas engine names a
    bare join return ``a.name``), so we escape for a quoted identifier rather
    than using the stricter :func:`sanitize_sql_identifier` (source columns).
    """
    return '"' + name.replace('"', '""') + '"'


def _output_column(item: Any, *, qualified: bool) -> str:
    """Output column name for a return/with *item*, matching the pandas engine.

    An explicit ``AS alias`` wins.  A bare variable (post-``WITH`` scalar) is
    named after the variable.  A bare property lookup is named after the
    property (single-variable patterns) or as ``var.property`` (multi-variable /
    join patterns).  Deterministic regardless of cached-AST alias mutation.
    """
    from pycypher.ast_models import Variable

    if item.alias is not None:
        return str(item.alias)
    expr = item.expression
    if isinstance(expr, Variable):
        return str(expr.name)
    if qualified:
        return f"{expr.expression.name}.{expr.property}"
    return str(expr.property)


def _build_order_clause(
    order_by: Any, items: Any, *, qualified: bool
) -> str | None:
    """Build a DuckDB ORDER BY clause referencing a stage's output columns.

    Supports ordering by an output alias (``ORDER BY name``) or by a returned
    property lookup (``ORDER BY n.age`` when ``n.age`` is in the output).  Other
    order keys, or an explicit NULLS placement, return ``None`` (fall back).
    Emits ``NULLS LAST`` to match the pandas engine's null ordering.
    """
    from pycypher.ast_models import PropertyLookup, Variable

    output_names: set[str] = set()
    prop_to_output: dict[tuple[str, str], str] = {}
    for item in items:
        name = _output_column(item, qualified=qualified)
        output_names.add(name)
        expr = item.expression
        if isinstance(expr, PropertyLookup) and isinstance(
            expr.expression, Variable
        ):
            prop_to_output[(expr.expression.name, expr.property)] = name

    parts: list[str] = []
    for ob in order_by:
        if getattr(ob, "nulls_placement", None) is not None:
            return None  # explicit NULLS FIRST/LAST not mapped yet
        expr = ob.expression
        col: str | None = None
        if isinstance(expr, Variable) and expr.name in output_names:
            col = expr.name
        elif isinstance(expr, PropertyLookup) and isinstance(
            expr.expression, Variable
        ):
            col = prop_to_output.get((expr.expression.name, expr.property))
        if col is None:
            return None
        direction = "ASC" if ob.ascending else "DESC"
        parts.append(f"{_quote_output_alias(col)} {direction} NULLS LAST")

    return ", ".join(parts)


class _StageSQL:
    """Compiled SQL pieces for one pipeline stage."""

    __slots__ = (
        "aggregating",
        "distinct",
        "group_parts",
        "having_sql",
        "limit",
        "new_scope",
        "order_clause",
        "passthrough",
        "select_parts",
        "skip",
        "unwind",
        "unwind_select",
        "where_sql",
    )

    def __init__(self, **kw: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))


def _stage_is_passthrough(stage: Any, scope: _Scope) -> bool:
    """True if *stage* just passes the in-scope pattern variables through."""
    from pycypher.ast_models import Variable

    if (
        stage.distinct
        or stage.order_by
        or stage.skip is not None
        or stage.limit is not None
    ):
        return False
    if not stage.items:
        return False
    return all(
        it.alias is None
        and isinstance(it.expression, Variable)
        and it.expression.name in scope.node_vars
        for it in stage.items
    )


def _plan_stage(
    stage: Any,
    scope: _Scope,
    udfs: frozenset[str],
    *,
    is_return: bool,
) -> _StageSQL | None:
    """Compile one WITH/RETURN stage over *scope*, or return ``None``.

    Returns SQL pieces plus the resulting scope for the next stage.
    """
    from pycypher.ast_models import PropertyLookup, Unwind, Variable
    from pycypher.relation_sql import (
        compile_aggregate,
        compile_expression,
        is_aggregate,
    )

    # --- UNWIND stage: expand a list column, keeping current columns ---
    if isinstance(stage, Unwind):
        if scope.names is None or stage.alias is None:
            return None  # only supported in scalar scope (post-WITH / leading UNWIND)
        if stage.alias in scope.names:
            return None  # would shadow an existing column
        expr_sql = compile_expression(
            stage.expression,
            scope.resolve,
            udfs,
            scope.resolve_var,
        )
        if expr_sql is None:
            return None
        select = f"*, UNNEST({expr_sql}) AS {_quote_output_alias(stage.alias)}"
        new_names = [*sorted(scope.names), stage.alias]
        return _StageSQL(
            unwind=True,
            unwind_select=select,
            new_scope=_scalar_scope(new_names),
        )

    # --- Node pass-through WITH (filter only, scope unchanged) ---
    if not is_return and _stage_is_passthrough(stage, scope):
        stage_where = getattr(stage, "where", None)
        where_sql = None
        if stage_where is not None:
            where_sql = compile_expression(
                stage_where,
                scope.resolve,
                udfs,
                scope.resolve_var,
            )
            if where_sql is None:
                return None
        return _StageSQL(
            passthrough=True, where_sql=where_sql, new_scope=scope
        )

    # --- Mixed WITH: one or more bound nodes passed through *alongside*
    # new named items, e.g. "WITH o.longitude AS longitude, id(o) AS
    # identifier, o" -- the passthrough branch above only fires when
    # *every* item is a bare node passthrough; this handles the case where
    # at least one, but not all, items are. Single-component scopes only
    # (_single_component_scope) -- see _set_stage_eligible's docstring for
    # why. Falls through to the generic branch below (which will reject a
    # bare node-variable item outright) when this doesn't apply, so declining
    # here is always safe, never silently wrong.
    if not is_return and _single_component_scope(scope):
        passthrough_names = [
            it.expression.name
            for it in stage.items
            if it.alias is None
            and isinstance(it.expression, Variable)
            and it.expression.name in scope.node_vars
        ]
        if passthrough_names and not (
            stage.distinct
            or stage.order_by
            or stage.skip is not None
            or stage.limit is not None
        ):
            return _plan_mixed_stage(stage, scope, udfs, passthrough_names)

    if not stage.items:
        return None
    if stage.skip is not None and stage.limit is None:
        return None

    aggregating = any(is_aggregate(it.expression) for it in stage.items)
    select_parts: list[str] = []
    group_parts: list[str] = []
    output_names: list[str] = []
    for it in stage.items:
        if is_aggregate(it.expression):
            sql = compile_aggregate(
                it.expression,
                scope.resolve,
                udfs,
                scope.resolve_var,
            )
            if sql is None or it.alias is None:
                return None
        else:
            sql = compile_expression(
                it.expression, scope.resolve, udfs, scope.resolve_var
            )
            if sql is None:
                return None
            if (
                not isinstance(it.expression, (PropertyLookup, Variable))
                and it.alias is None
            ):
                return None
            group_parts.append(sql)
        name = _output_column(it, qualified=scope.qualified)
        output_names.append(name)
        select_parts.append(f"{sql} AS {_quote_output_alias(name)}")

    if len(set(output_names)) != len(output_names):
        return None

    new_scope = _scalar_scope(output_names)

    stage_where = getattr(stage, "where", None)
    having_sql = None
    if stage_where is not None:
        having_sql = compile_expression(
            stage_where,
            new_scope.resolve,
            udfs,
            new_scope.resolve_var,
        )
        if having_sql is None:
            return None

    order_clause = None
    if stage.order_by:
        order_clause = _build_order_clause(
            stage.order_by,
            stage.items,
            qualified=scope.qualified,
        )
        if order_clause is None:
            return None

    return _StageSQL(
        passthrough=False,
        aggregating=aggregating,
        select_parts=select_parts,
        group_parts=group_parts,
        having_sql=having_sql,
        distinct=bool(stage.distinct),
        order_clause=order_clause,
        skip=stage.skip,
        limit=stage.limit,
        new_scope=new_scope,
    )


def _plan_mixed_stage(
    stage: Any,
    scope: _Scope,
    udfs: frozenset[str],
    passthrough_names: list[str],
) -> _StageSQL | None:
    """Compile a WITH stage that passes one or more bound nodes through
    *alongside* new named expressions, or return ``None``.

    Projects each passed-through node's full raw column set (``alias.*``,
    ``EXCLUDE``-ing any raw column a new item's own output name would
    otherwise collide with) plus the other items' compiled SQL, so the
    resulting relation still carries every property a later stage — or a
    mid-pipeline ``SET`` (:func:`_execute_set_stage`) — might read off the
    passed-through node(s). Declines (returns ``None``, so the caller falls
    back conservatively) rather than risk silently wrong SQL when: any
    "other" item aggregates (cardinality-incompatible with a raw
    passthrough); or two passed-through nodes' raw column sets collide
    (ambiguous — not attempted). Caller (:func:`_plan_stage`) has already
    verified :func:`_single_component_scope`.
    """
    from pycypher.ast_models import PropertyLookup, Variable
    from pycypher.relation_sql import (
        compile_expression,
        is_aggregate,
    )

    other_items = [
        it
        for it in stage.items
        if not (
            it.alias is None
            and isinstance(it.expression, Variable)
            and it.expression.name in passthrough_names
        )
    ]
    if not other_items or any(
        is_aggregate(it.expression) for it in other_items
    ):
        return None

    other_output_names: list[str] = []
    other_select_parts: list[str] = []
    for it in other_items:
        sql = compile_expression(
            it.expression, scope.resolve, udfs, scope.resolve_var
        )
        if sql is None:
            return None
        if (
            not isinstance(it.expression, (PropertyLookup, Variable))
            and it.alias is None
        ):
            return None
        name = _output_column(it, qualified=scope.qualified)
        other_output_names.append(name)
        other_select_parts.append(f"{sql} AS {_quote_output_alias(name)}")
    if len(set(other_output_names)) != len(other_output_names):
        return None

    passthrough_select_parts: list[str] = []
    new_pattern_vars: dict[str, tuple[str, Any]] = {}
    new_var_labels: dict[str, str] = {}
    seen_cols: set[str] = set()
    for var in passthrough_names:
        alias, attr = scope.pattern_vars[var]
        remaining = set(attr.values()) - set(other_output_names)
        if seen_cols & remaining:
            return None  # two passed-through nodes share a raw column name
        seen_cols |= remaining
        shadowed = sorted(c for c in attr.values() if c in other_output_names)
        excl = (
            " EXCLUDE (" + ", ".join(f'"{c}"' for c in shadowed) + ")"
            if shadowed
            else ""
        )
        passthrough_select_parts.append(f"{alias}.*{excl}")
        new_pattern_vars[var] = (alias, attr)
        if var in scope.var_labels:
            new_var_labels[var] = scope.var_labels[var]

    select_parts = passthrough_select_parts + other_select_parts

    new_resolve = _make_resolve(new_pattern_vars)
    other_names_frozen = frozenset(other_output_names)

    def new_resolve_var(name: str) -> str | None:
        return (
            _quote_output_alias(name) if name in other_names_frozen else None
        )

    new_scope = _Scope(
        new_resolve,
        new_resolve_var,
        qualified=scope.qualified,
        node_vars=frozenset(passthrough_names),
        pattern_vars=new_pattern_vars,
        var_labels=new_var_labels,
    )

    having_sql = None
    stage_where = getattr(stage, "where", None)
    if stage_where is not None:
        having_sql = compile_expression(
            stage_where, new_scope.resolve, udfs, new_scope.resolve_var
        )
        if having_sql is None:
            return None

    return _StageSQL(
        passthrough=False,
        aggregating=False,
        select_parts=select_parts,
        group_parts=[],
        having_sql=having_sql,
        distinct=False,
        order_clause=None,
        skip=None,
        limit=None,
        new_scope=new_scope,
    )


def _single_component_scope(scope: _Scope) -> bool:
    """True if *scope*'s relation is still a single physical table scan
    under a single alias — no join has happened yet (no fixed-length-path
    hop, no ``OPTIONAL MATCH`` extension, no embedded second ``MATCH``).

    Verified empirically in a DuckDB sandbox: a single-table relation's
    alias survives being referenced (``alias."col"``) across arbitrarily
    many chained ``.project()`` calls, but a *joined* relation's component
    aliases (``v0``/``v1``) reliably stop resolving the moment a
    ``.project()`` is chained directly onto the join — even though
    chaining a ``.filter()`` first, then one ``.project()``, still works,
    which is exactly the fragility that makes it unsafe to build a
    multi-stage feature on. ``len(scope.pattern_vars) == 1`` is an exact
    proxy for this: every join this module performs (fixed-length path,
    ``OPTIONAL MATCH``, a second embedded ``MATCH``) adds at least one more
    entry to ``pattern_vars``, so more than one entry means a join already
    happened somewhere upstream of this scope.
    """
    return len(scope.pattern_vars) == 1


def _set_stage_target(stage: Any) -> str | None:
    """Return the single variable every item of *stage* (a ``Set`` clause)
    targets, or ``None`` if the items are empty, target more than one
    variable, or aren't plain property assignments (label-set/whole-map
    forms are out of scope — same restriction every other mutation kind in
    this module applies).
    """
    if not stage.items:
        return None
    var_names = {
        it.variable.name for it in stage.items if it.variable is not None
    }
    if len(var_names) != 1:
        return None
    if any(
        it.variable is None
        or it.property is None
        or it.property in ("*", "*+")
        or it.expression is None
        or it.labels
        for it in stage.items
    ):
        return None
    return next(iter(var_names))


def _set_stage_eligible(stage: Any, scope: _Scope, context: Context) -> bool:
    """Return True if a mid-pipeline ``SET`` stage (``Set`` not the
    terminal clause — Phase 2b category (E),
    docs/fastopendata_streaming_qualification_plan.md) can compile.

    Eligible shape: every item targets the same variable, which must
    currently be a full node in *scope* (``scope.node_vars`` — a scalar
    post-``WITH`` alias has no physical table to write to) with a known
    label and a real registered streaming source (a writable table); each
    item's value must compile via :func:`~pycypher.relation_sql.
    compile_expression` over the *current* scope (so it can reference the
    node's own properties, any ``WITH``-stage alias already in scope, and
    registered UDFs — the same general compiler the scalar_set slice uses);
    each target property must already resolve or be a safe new column name
    (deferred to :func:`_ensure_column` at execution time, matching every
    other mutation kind — this check never writes). Also requires
    *scope* to be single-component (:func:`_single_component_scope`) —
    verified empirically (not assumed) that a joined ``DuckDBPyRelation``'s
    component aliases stop reliably resolving after being chained through
    a ``.project()`` (the physical ``UPDATE``'s own SQL text is unaffected,
    but folding the computed value back into the in-flight relation, and
    every later stage after that, needs its alias to keep working).
    """
    from pycypher.relation_sql import compile_expression

    if not _single_component_scope(scope):
        return False
    var_name = _set_stage_target(stage)
    if var_name is None:
        return False
    if var_name not in scope.node_vars or var_name not in scope.var_labels:
        return False
    label = scope.var_labels[var_name]
    if label not in context._streaming_sources:
        return False
    udfs = _udf_names(context)
    for item in stage.items:
        if (
            compile_expression(
                item.expression, scope.resolve, udfs, scope.resolve_var
            )
            is None
        ):
            return False
        if scope.resolve(
            var_name, item.property
        ) is None and not _new_column_name_allowed(item.property):
            return False
    return True


def _scope_after_set_stage(stage: Any, scope: _Scope) -> _Scope:
    """Return the scope :func:`is_relation_eligible`'s *own* continued walk
    should use after an eligible mid-pipeline ``SET`` stage.

    A brand-new target property is deliberately not created here — a query
    that fails eligibility later must leave no side effect, matching every
    other mutation kind's eligibility check in this module (real creation
    is :func:`_ensure_column`'s job, run for real by
    :func:`_execute_set_stage` at *execution* time, never at eligibility-
    check time). But a *later stage of the same query* needs to see that
    property as resolvable to itself pass eligibility (e.g.
    ``osm_longitude``'s trailing ``WITH ... o.foo AS foo ...`` reading the
    property its own earlier ``SET`` stage just created) — the same class
    of gap Phase 3b closed across queries (see the module docstring's
    finding (3)), here within one query's own pipeline. Fixed the same way
    in spirit: a local, non-mutating ``ChainMap`` overlay (never touching
    the real, shared attr_map) makes the property resolvable for this
    eligibility walk only; whether it actually gets created is still
    decided for real, later, by ``_ensure_column``.
    """
    from collections import ChainMap

    var_name = _set_stage_target(stage)
    alias, attr = scope.pattern_vars[var_name]  # type: ignore[index]
    new_props = {
        it.property: it.property
        for it in stage.items
        if it.property not in attr
    }
    if not new_props:
        return scope
    new_attr = ChainMap(new_props, attr)
    new_pattern_vars = {**scope.pattern_vars, var_name: (alias, new_attr)}
    return _Scope(
        _make_resolve(new_pattern_vars),
        scope.resolve_var,
        qualified=scope.qualified,
        node_vars=scope.node_vars,
        pattern_vars=new_pattern_vars,
        var_labels=scope.var_labels,
    )


def _execute_set_stage(
    stage: Any,
    scope: _Scope,
    context: Context,
    con: Any,
    rel: Any,
) -> Any:
    """Execute a mid-pipeline ``SET`` stage and return the updated relation.

    Precondition: :func:`_set_stage_eligible` returned ``True`` for *stage*
    in *scope*. Two effects, matching every other mutation kind in this
    module:

    1. **Durable**: a native ``UPDATE`` against the target's physical table,
       sourced from *rel* as of this point in the pipeline (one row per
       target id — a ``QUALIFY ROW_NUMBER() ... = 1`` dedup guards against a
       pattern that fanned out via a join before reaching this stage, same
       deterministic-pick rationale as :func:`execute_relation_copy_set`;
       none of this plan's actual target queries fan out, so this is
       defence-in-depth, not exercised by them).
    2. **In-flight**: the computed values are folded into *rel* itself (an
       ``EXCLUDE`` + re-``project()`` for a property that already had a
       column, a plain added column for a brand-new one) so later pipeline
       stages read the mutated properties exactly like any other. Crucially,
       *scope* itself never needs to change: ``_ensure_column`` mutates the
       registered table's attr_map dict **in place** (see its docstring),
       and every resolve() closure in this module was built over that same
       live dict, so a property that was just created resolves correctly
       for later stages with no scope rebuild — the same invariant the
       ``id()`` sentinel's ``ChainMap`` (not a copy) relies on.
    """
    import uuid

    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import ID_SENTINEL, compile_expression

    var_name = _set_stage_target(stage)
    label = scope.var_labels[var_name]  # type: ignore[index]
    _alias, attr = scope.pattern_vars[var_name]  # type: ignore[index]
    udfs = _udf_names(context)
    id_col = _node_id_column(context, label)
    id_ref = scope.resolve(var_name, ID_SENTINEL)  # type: ignore[arg-type]

    value_sqls: list[str] = []
    phys_cols: list[str] = []
    select_parts = [f'{id_ref} AS "__id__"']
    for i, item in enumerate(stage.items):
        value_sql = compile_expression(
            item.expression, scope.resolve, udfs, scope.resolve_var
        )
        value_sqls.append(value_sql)  # type: ignore[arg-type]
        if item.property not in attr:
            probe = rel.project(f"{value_sql} AS __probe__").limit(0)
            _ensure_column(context, label, item.property, probe)
        # attr is the same live dict _ensure_column just mutated in place
        # (see its docstring) -- a brand-new column's physical name is
        # visible here immediately, no scope rebuild needed.
        phys_cols.append(attr[item.property])
        val_col = f"__setval_{i}__"
        select_parts.append(f'{value_sql} AS "{val_col}"')

    projected = rel.project(", ".join(select_parts))

    src_view = f"__pycypher_set_stage_src_{uuid.uuid4().hex}__"
    qualify_sql = (
        f'SELECT * FROM "{src_view}" '
        'QUALIFY ROW_NUMBER() OVER (PARTITION BY "__id__" ORDER BY "__id__") = 1'
    )
    deduped = projected.query(src_view, qualify_sql)

    view_name = f"__pycypher_set_stage_{uuid.uuid4().hex}__"
    deduped.to_view(view_name, replace=True)

    table = physical_table_name(label)
    set_sql_parts = [
        f'"{col}" = sub."__setval_{i}__"' for i, col in enumerate(phys_cols)
    ]
    sql = (
        f'UPDATE "{table}" SET {", ".join(set_sql_parts)} '  # nosec B608 — table/columns validated identifiers; values from relation_sql's whitelisted compiler; view_name is an internally generated uuid4, not user input
        f'FROM "{view_name}" AS sub '
        f'WHERE "{table}"."{id_col}" = sub."__id__"'
    )
    LOGGER.debug("[duckdb-relation] %s", sql)
    try:
        con.execute(sql)
    finally:
        con.execute(f'DROP VIEW IF EXISTS "{view_name}"')

    # Fold the computed values into the in-flight relation. EXCLUDE any
    # column a value is overwriting (already part of rel's columns) to
    # avoid a duplicate-column error; a brand-new column has no such
    # collision.
    existing_cols = set(rel.columns)
    overwritten = [c for c in phys_cols if c in existing_cols]
    fold_parts = (
        ["* EXCLUDE (" + ", ".join(f'"{c}"' for c in overwritten) + ")"]
        if overwritten
        else ["*"]
    )
    fold_parts.extend(
        f'{value_sqls[i]} AS "{phys_cols[i]}"' for i in range(len(phys_cols))
    )
    return rel.project(", ".join(fold_parts))


def is_relation_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is in the subset the relation engine can execute.

    Conservative by design: anything not explicitly handled makes the query
    ineligible so the caller falls back to the pandas engine.  Eligible: a
    single-node or fixed-length directed-path ``MATCH`` (one or more hops);
    zero or more ``OPTIONAL MATCH`` LEFT-join extensions from a bound node;
    an optional compilable ``WHERE``; zero or more ``WITH`` stages (projection
    / aggregation / filter / DISTINCT / ORDER BY / SKIP+LIMIT, or a node
    pass-through); a cross-joined second required ``MATCH`` immediately after
    a ``WITH``; a mid-pipeline ``SET`` targeting a single currently-bound
    node variable with compilable values (see :func:`_set_stage_eligible`);
    and a ``RETURN`` of compilable expressions.  Ineligible: undirected /
    variable-length paths, more than one embedded second ``MATCH`` or one
    not preceded by a ``WITH``, ``OPTIONAL MATCH`` combined with
    aggregation, ``UNWIND`` in pattern scope, unsupported
    functions/operators, and ``collect()``.
    """
    from pycypher.ast_models import Match, Set
    from pycypher.relation_sql import compile_expression

    plan = _analyze_query(query, context)
    if plan is None:
        return False

    udfs = _udf_names(context)
    resolve = plan.initial_scope.resolve
    if plan.match_where is not None and (
        compile_expression(plan.match_where, resolve, udfs) is None
    ):
        return False
    if plan.unwind_expr is not None and (
        compile_expression(plan.unwind_expr, _no_prop, resolve_var=_no_var)
        is None
    ):
        return False
    if _compile_inline_predicates(plan.inline_preds, resolve, udfs) is None:
        return False

    scope = plan.initial_scope
    last = len(plan.stages) - 1
    for i, stage in enumerate(plan.stages):
        if isinstance(stage, Match):
            ext = _analyze_second_match(scope, stage, context, plan.alias_gen)
            if ext is None:
                return False
            new_scope, _build, where, inline_preds, _acc_alias = ext
            if where is not None and (
                compile_expression(
                    where, new_scope.resolve, udfs, new_scope.resolve_var
                )
                is None
            ):
                return False
            if (
                _compile_inline_predicates(
                    inline_preds, new_scope.resolve, udfs
                )
                is None
            ):
                return False
            scope = new_scope
            continue
        if isinstance(stage, Set):
            if not _set_stage_eligible(stage, scope, context):
                return False
            scope = _scope_after_set_stage(stage, scope)
            continue
        sp = _plan_stage(stage, scope, udfs, is_return=(i == last))
        if sp is None:
            return False
        scope = sp.new_scope
    return True


def execute_relation_query(
    query: Any,
    context: Context,
    *,
    materialize: bool = True,
) -> pd.DataFrame | RelationBindings:
    """Execute an eligible query via a pipeline of lazy DuckDB relations.

    Builds the base relation from the pattern, applies the MATCH ``WHERE``, then
    runs each ``WITH``/``RETURN`` stage (projection / aggregation / filter /
    DISTINCT / ORDER BY / SKIP+LIMIT) as lazy ``DuckDBPyRelation`` ops.

    Precondition: :func:`is_relation_eligible` returned ``True`` for *query*.

    Args:
        query: The parsed, eligible query AST.
        context: The DuckDB-backed context.
        materialize: When ``True`` (default) return a pandas DataFrame; when
            ``False`` return a :class:`RelationBindings` for streaming to a sink.

    """
    from pycypher.ast_models import Match, Set
    from pycypher.backends.duckdb_backend import DuckDBLazyFrame
    from pycypher.relation_sql import compile_expression

    plan = _analyze_query(query, context)
    con = context.backend.connection
    udfs = _udf_names(context)

    resolve = plan.initial_scope.resolve
    rel = plan.build(con)
    if plan.match_where is not None:
        rel = rel.filter(compile_expression(plan.match_where, resolve, udfs))
    for pred in (
        _compile_inline_predicates(plan.inline_preds, resolve, udfs) or []
    ):
        rel = rel.filter(pred)

    scope = plan.initial_scope
    last = len(plan.stages) - 1
    for i, stage in enumerate(plan.stages):
        if isinstance(stage, Match):
            new_scope, build, where, inline_preds, acc_alias = (
                _analyze_second_match(
                    scope,
                    stage,
                    context,
                    plan.alias_gen,
                )
            )
            # `.join(other, "true")` rather than `.cross()`: DuckDB's relation
            # API drops component-alias info after a `.filter()`/`.project()`
            # chained onto a `.cross()` result (verified — raises "Referenced
            # table ... not found"), but preserves it across `.join()`.
            rel = rel.set_alias(acc_alias).join(build(con), "true")
            if where is not None:
                rel = rel.filter(
                    compile_expression(
                        where, new_scope.resolve, udfs, new_scope.resolve_var
                    ),
                )
            for pred in (
                _compile_inline_predicates(
                    inline_preds, new_scope.resolve, udfs
                )
                or []
            ):
                rel = rel.filter(pred)
            scope = new_scope
            continue
        if isinstance(stage, Set):
            rel = _execute_set_stage(stage, scope, context, con, rel)
            continue
        sp = _plan_stage(stage, scope, udfs, is_return=(i == last))
        if sp.unwind:
            rel = rel.project(sp.unwind_select)
            scope = sp.new_scope
            continue
        if sp.passthrough:
            if sp.where_sql is not None:
                rel = rel.filter(sp.where_sql)
            continue
        if sp.aggregating:
            rel = rel.aggregate(
                ", ".join(sp.select_parts), ", ".join(sp.group_parts)
            )
        else:
            rel = rel.project(", ".join(sp.select_parts))
        if sp.having_sql is not None:
            rel = rel.filter(sp.having_sql)
        if sp.distinct:
            rel = rel.distinct()
        if sp.order_clause is not None:
            rel = rel.order(sp.order_clause)
        if sp.limit is not None:
            rel = rel.limit(sp.limit, offset=sp.skip or 0)
        scope = sp.new_scope

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("[duckdb-relation] %s", rel.sql_query())

    bindings = RelationBindings(DuckDBLazyFrame(rel, con))
    return bindings.to_pandas() if materialize else bindings


# ---------------------------------------------------------------------------
# Mutations (docs/duckdb_full_parity_design.md, Phase 2: SET / DELETE / CREATE)
# ---------------------------------------------------------------------------


def _analyze_single_node_match(
    query: Any,
    context: Context,
    clause_type: type,
) -> tuple[Any, Any, str, str, Any] | None:
    """Return ``(match, next_clause, label, var_name, resolve)`` for a
    required, non-optional, single-node ``MATCH [WHERE ...]`` immediately
    followed by exactly one clause of *clause_type*, on a label with a
    registered streaming source — the shared eligibility shape for ``SET``
    and ``DELETE``.  Callers must still validate the following clause's own
    item/expression shape (this only checks its type and position).

    *Label* must have a registered streaming source (:func:`register_streaming_source`)
    — the real, writable DuckDB table the mutation compiles a native
    statement against; entities only available via the in-memory
    ``entity_mapping`` fallback are ineligible since there is no table to
    write to.
    """
    from pycypher.ast_models import Match, Query

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) != 2:
        return None
    match, next_clause = clauses
    if not isinstance(match, Match) or match.optional:
        return None
    if not isinstance(next_clause, clause_type):
        return None

    paths = match.pattern.paths
    if len(paths) != 1 or paths[0].variable is not None:
        return None
    elements = paths[0].elements
    if len(elements) != 1:
        return None
    node = elements[0]
    if not _valid_node(node):
        return None

    label = node.labels[0]
    if label not in context._streaming_sources:
        return None
    attr = _entity_attr_map(context, label)
    if attr is None:
        return None

    var_name = node.variable.name
    resolve = _make_resolve({var_name: ("", attr)})

    return match, next_clause, label, var_name, resolve


def _new_column_name_allowed(prop: str) -> bool:
    """True if *prop* is safe to create as a brand-new column via
    :func:`_ensure_column` (Phase 3a,
    docs/fastopendata_streaming_qualification_plan.md).

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


def _analyze_set_query(
    query: Any,
    context: Context,
) -> tuple[Any, Any, str, str, Any] | None:
    """Return ``(match, set_clause, label, var_name, resolve)`` if *query* is
    a single-table ``SET``-eligible mutation, else ``None``.

    Eligible shape: exactly ``MATCH (var:Label) [WHERE ...] SET ...`` — see
    :func:`_analyze_single_node_match` for the shared MATCH-shape checks
    (no relationship pattern, no ``OPTIONAL MATCH``, no ``WITH`` stages,
    label has a registered streaming source), plus a ``SET`` whose items are
    all plain ``var.prop = expr`` assignments targeting *var*.  The AST
    converter represents every ``SET`` item form (property assignment,
    ``SET n:Label``, ``SET n = {..}``, ``SET n += {..}``) as the same
    :class:`SetItem` class, distinguished only by which fields are
    populated: a plain property assignment has a non-sentinel ``property``
    (not ``None``, ``"*"``, or ``"*+"``), a populated ``expression``, and no
    ``labels`` — anything else (label sets, whole-map sets/merges) is
    rejected here.
    """
    from pycypher.ast_models import Set

    info = _analyze_single_node_match(query, context, Set)
    if info is None:
        return None
    match, set_clause, label, var_name, resolve = info
    if not set_clause.items:
        return None

    if not all(
        item.variable is not None
        and item.variable.name == var_name
        and item.property is not None
        and item.property not in ("*", "*+")
        and item.expression is not None
        and not item.labels
        for item in set_clause.items
    ):
        return None

    return match, set_clause, label, var_name, resolve


def is_relation_set_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a single-table ``SET`` the relation engine
    can execute as a native ``UPDATE`` (see :func:`_analyze_set_query`).
    """
    from pycypher.relation_sql import compile_expression

    info = _analyze_set_query(query, context)
    if info is None:
        return False
    match, set_clause, _label, var_name, resolve = info

    udfs = _udf_names(context)
    if (
        match.where is not None
        and compile_expression(match.where, resolve, udfs) is None
    ):
        return False
    node = match.pattern.paths[0].elements[0]
    if (
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        is None
    ):
        return False
    for item in set_clause.items:
        if resolve(
            var_name, item.property
        ) is None and not _new_column_name_allowed(item.property):
            return False
        if compile_expression(item.expression, resolve, udfs) is None:
            return False
    return True


def execute_relation_set(query: Any, context: Context) -> None:
    """Execute an eligible single-table ``SET`` as a native ``UPDATE``.

    Precondition: :func:`is_relation_set_eligible` returned ``True`` for
    *query*. No return value — mirrors the pandas path's convention of an
    empty result for a non-``RETURN``-terminal mutation query. A target
    property not yet in the entity's schema is created via
    :func:`_ensure_column` (Phase 3a) before being assigned.
    """
    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_expression

    match, set_clause, label, var_name, resolve = _analyze_set_query(
        query, context
    )  # type: ignore[misc]
    udfs = _udf_names(context)
    con = context.backend.connection
    table = physical_table_name(label)

    where_parts: list[str] = []
    if match.where is not None:
        where_parts.append(compile_expression(match.where, resolve, udfs))  # type: ignore[arg-type]
    node = match.pattern.paths[0].elements[0]
    where_parts.extend(
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        or [],
    )

    set_parts = []
    for item in set_clause.items:
        value_sql = compile_expression(item.expression, resolve, udfs)
        target_col = resolve(var_name, item.property)
        if target_col is None:
            probe = (
                con.table(table).project(f"{value_sql} AS __probe__").limit(0)
            )
            _ensure_column(context, label, item.property, probe)
            target_col = resolve(var_name, item.property)
        set_parts.append(f"{target_col} = {value_sql}")

    sql = f'UPDATE "{table}" SET {", ".join(set_parts)}'  # nosec B608 — table validated by validate_identifier; set/where parts built exclusively from relation_sql.compile_expression's whitelisted compiler
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    LOGGER.debug("[duckdb-relation] %s", sql)
    con.execute(sql)


def _analyze_scalar_set_query(
    query: Any,
    context: Context,
) -> tuple[Any, list[Any], Any, str, str, Any] | None:
    """Return a plan for a single-node scalar ``WITH``-then-``SET`` mutation,
    or ``None`` (docs/fastopendata_streaming_qualification_plan.md, Phase 2b-i).

    Eligible shape: ``MATCH (var:Label) [WHERE ...] WITH var, expr1 AS
    a1[, ...] SET var.prop1 = value1[, ...]`` — the same single-node,
    no-relationship ``MATCH`` shape :func:`_analyze_single_node_match`
    validates for the plain ``"set"`` kind (duplicated here rather than
    reused, since that helper's shape is fixed at exactly two clauses), plus
    a ``WITH`` in between that does nothing but name non-aggregate scalar
    expressions over the same bound variable (no relationship, no
    ``GROUP BY`` — that shape is :func:`_analyze_group_set_query`'s job)
    before a ``SET`` whose values may be *any* expression compilable via
    :func:`~pycypher.relation_sql.compile_expression` over the bound var's
    own properties and the ``WITH`` stage's aliases — a bare alias, a UDF
    call over one (e.g. the real pipeline's ``decode_tags`` query), a
    negation, arithmetic, etc. — not just a bare-alias pass-through (that
    restriction was Phase 2b-i's original narrower slice; relaxed once (D)'s
    long tail needed the same general compiler anyway). This closes the
    single largest slice of the Phase 2 long tail: `_analyze_set_query`'s
    ``len(clauses) != 2`` check rejects *any* ``WITH`` outright, even one
    that only exists to name an expression before assigning it — purely a
    syntactic gap, not a correctness one.

    Returns ``(match, scalar_items, set_clause, label, var_name, resolve)``:
    *scalar_items* are the ``WITH`` clause's non-passthrough items
    (structurally validated, not yet compiled to SQL); the rest mirrors
    :func:`_analyze_set_query`'s return convention. Callers must still
    compile ``match.where``, the inline predicates, and each scalar item to
    SQL (see :func:`is_relation_scalar_set_eligible` /
    :func:`execute_relation_scalar_set`).
    """
    from pycypher.ast_models import Match, Query, Set, Variable, With
    from pycypher.relation_sql import is_aggregate

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) != 3:
        return None
    match, with_clause, set_clause = clauses
    if not isinstance(match, Match) or match.optional:
        return None
    if not isinstance(with_clause, With):
        return None
    if not isinstance(set_clause, Set):
        return None
    if (
        with_clause.distinct
        or with_clause.order_by
        or with_clause.skip is not None
        or with_clause.limit is not None
        or getattr(with_clause, "where", None) is not None
        or not with_clause.items
    ):
        return None
    if any(is_aggregate(it.expression) for it in with_clause.items):
        return None  # aggregates are _analyze_group_set_query's job

    paths = match.pattern.paths
    if len(paths) != 1 or paths[0].variable is not None:
        return None
    elements = paths[0].elements
    if len(elements) != 1:
        return None
    node = elements[0]
    if not _valid_node(node):
        return None

    label = node.labels[0]
    if label not in context._streaming_sources:
        return None
    attr = _entity_attr_map(context, label)
    if attr is None:
        return None

    var_name = node.variable.name
    resolve = _make_resolve({var_name: ("", attr)})

    def _is_passthrough(it: Any) -> bool:
        # A bare, originally-unaliased ``var`` item is what this checks for
        # -- but the pandas engine's projection planner mutates a cached
        # AST's None alias to the variable's own name as a side effect of
        # executing the *same* query text once via that path (ASTConverter
        # caches by text via lru_cache, so this mutation is visible here
        # too). Treating alias in (None, var_name) as equivalent avoids
        # this check flipping based on execution history.
        if (
            not isinstance(it.expression, Variable)
            or it.expression.name != var_name
        ):
            return False
        return it.alias is None or it.alias == var_name

    passthrough_items = [it for it in with_clause.items if _is_passthrough(it)]
    if len(passthrough_items) != 1:
        return None
    scalar_items = [it for it in with_clause.items if not _is_passthrough(it)]
    if not scalar_items or any(it.alias is None for it in scalar_items):
        return None

    if not set_clause.items:
        return None
    # A SET item's *value* may be any compilable expression (a bare WITH
    # alias, a UDF call over one, arithmetic, ...) -- actual compilability
    # is compile_expression's job (see is_relation_scalar_set_eligible /
    # execute_relation_scalar_set, which resolve WITH aliases via
    # resolve_var), not pre-validated here. Only the item's *shape*
    # (targets var_name, a real property, no label-set) is checked.
    if not all(
        item.variable is not None
        and item.variable.name == var_name
        and item.property is not None
        and item.property not in ("*", "*+")
        and item.expression is not None
        and not item.labels
        for item in set_clause.items
    ):
        return None

    return match, scalar_items, set_clause, label, var_name, resolve


def is_relation_scalar_set_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a single-node scalar ``WITH``-then-``SET``
    mutation the relation engine can execute as a native ``UPDATE`` (see
    :func:`_analyze_scalar_set_query`).
    """
    from pycypher.relation_sql import compile_expression

    info = _analyze_scalar_set_query(query, context)
    if info is None:
        return False
    match, scalar_items, set_clause, _label, var_name, resolve = info

    udfs = _udf_names(context)
    if (
        match.where is not None
        and compile_expression(match.where, resolve, udfs) is None
    ):
        return False
    node = match.pattern.paths[0].elements[0]
    if (
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        is None
    ):
        return False
    alias_sql: dict[str, str] = {}
    for it in scalar_items:
        sql = compile_expression(it.expression, resolve, udfs)
        if sql is None:
            return False
        alias_sql[it.alias] = sql

    def resolve_var(name: str) -> str | None:
        return alias_sql.get(name)

    for item in set_clause.items:
        if (
            compile_expression(item.expression, resolve, udfs, resolve_var)
            is None
        ):
            return False
        if resolve(
            var_name, item.property
        ) is None and not _new_column_name_allowed(item.property):
            return False
    return True


def execute_relation_scalar_set(query: Any, context: Context) -> None:
    """Execute an eligible single-node scalar ``WITH``-then-``SET`` mutation
    as a native ``UPDATE``.

    Precondition: :func:`is_relation_scalar_set_eligible` returned ``True``
    for *query*. No return value, matching the other mutation executors.
    Each ``WITH`` scalar item is compiled once and substituted directly into
    the ``SET`` values it feeds — no join, no view, no ``GROUP BY``; this is
    plain single-table ``UPDATE`` compilation identical to
    :func:`execute_relation_set`, with the ``WITH`` stage inlined away. A
    target property not yet in the entity's schema is created via
    :func:`_ensure_column` (Phase 3a) before being assigned.
    """
    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_expression

    match, scalar_items, set_clause, label, var_name, resolve = (
        _analyze_scalar_set_query(query, context)  # type: ignore[misc]
    )
    udfs = _udf_names(context)
    con = context.backend.connection
    table = physical_table_name(label)

    where_parts: list[str] = []
    if match.where is not None:
        where_parts.append(compile_expression(match.where, resolve, udfs))  # type: ignore[arg-type]
    node = match.pattern.paths[0].elements[0]
    where_parts.extend(
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        or [],
    )

    alias_sql = {
        it.alias: compile_expression(it.expression, resolve, udfs)
        for it in scalar_items
    }

    def resolve_var(name: str) -> str | None:
        return alias_sql.get(name)

    set_parts = []
    for item in set_clause.items:
        value_sql = compile_expression(
            item.expression, resolve, udfs, resolve_var
        )
        target_col = resolve(var_name, item.property)
        if target_col is None:
            probe = (
                con.table(table).project(f"{value_sql} AS __probe__").limit(0)
            )
            _ensure_column(context, label, item.property, probe)
            target_col = resolve(var_name, item.property)
        set_parts.append(f"{target_col} = {value_sql}")

    sql = f'UPDATE "{table}" SET {", ".join(set_parts)}'  # nosec B608 — table validated by validate_identifier; set/where parts built exclusively from relation_sql.compile_expression's whitelisted compiler
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    LOGGER.debug("[duckdb-relation] %s", sql)
    con.execute(sql)


def _analyze_group_set_query(
    query: Any,
    context: Context,
) -> (
    tuple[
        Any,
        list[Any],
        Any,
        str,
        str,
        dict[str, tuple[str, dict[str, str]]],
        Any,
        list[tuple[str, str, Any]],
    ]
    | None
):
    """Return a plan for an eligible aggregate-then-``SET`` mutation, or
    ``None`` (docs/fastopendata_streaming_qualification_plan.md, Phase 2
    first slice).

    Eligible shape: exactly ``MATCH <pattern> [WHERE ...] WITH <target>,
    AGG(...) AS a1[, ...] SET <target>.prop = a1[, ...]`` — a required,
    non-optional leading ``MATCH`` (single node or fixed-length directed
    path, see :func:`_analyze_leading_pattern`); a ``WITH`` with no
    ``DISTINCT``/``ORDER BY``/``SKIP``/``LIMIT``/``WHERE`` whose items are
    exactly one bare, unaliased pass-through of a bound node variable (the
    grouping key) plus one or more ``count/sum/avg/min/max`` aggregates each
    with an alias; and a ``SET`` whose items all target that same node and
    each assign a bare aggregate alias — no further expression — to a
    property.  A second required ``MATCH`` after the ``WITH``, a group
    variable renamed through the ``WITH`` (``WITH target AS t``), a
    grouping key that is not a bound node (e.g. a property), and a ``SET``
    expression combining aliases (``prop = a1 + a2``) are all out of scope
    for this slice and deferred to Phase 2b, per the plan.  The grouping
    node's label must have a registered streaming source (there must be a
    real table to write to).

    Returns ``(match, agg_items, set_clause, label, group_var, variables,
    build, inline_preds)``: *agg_items* are the ``WITH`` clause's aggregate
    items (structurally validated, not yet compiled to SQL); the rest mirror
    :func:`_analyze_leading_pattern`'s return plus the ``SET``-eligible
    helpers' ``(match, ..., label, var_name, ...)`` convention. Callers must
    still compile ``match.where``, the inline predicates, and each aggregate
    to SQL (see :func:`is_relation_group_set_eligible` /
    :func:`execute_relation_group_set`) — this function only checks AST
    shape, mirroring :func:`_analyze_set_query`'s division of labour.
    """
    from pycypher.ast_models import Match, Query, Set, Variable, With
    from pycypher.relation_sql import is_aggregate

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) != 3:
        return None
    match, with_clause, set_clause = clauses
    if not isinstance(match, Match) or match.optional:
        return None
    if not isinstance(with_clause, With):
        return None
    if not isinstance(set_clause, Set):
        return None
    if (
        with_clause.order_by
        or with_clause.skip is not None
        or with_clause.limit is not None
        or getattr(with_clause, "where", None) is not None
        or not with_clause.items
    ):
        return None
    if with_clause.distinct and not any(
        is_aggregate(it.expression) for it in with_clause.items
    ):
        # DISTINCT is only a no-op (safe to ignore) when grouping by an
        # aggregate already collapses to one row per group — see Phase 2b
        # category (C). A DISTINCT with no aggregate at all is a different,
        # unsupported shape.
        return None

    counter = [0]

    def alias_gen() -> str:
        alias = f"v{counter[0]}"
        counter[0] += 1
        return alias

    lead = _analyze_leading_pattern(match, context, alias_gen)
    if lead is None:
        return None
    variables, build, inline_preds, _var_labels = lead
    if len(variables) < 2:
        return None  # need the grouping node plus at least one other var

    nodes = match.pattern.paths[0].elements[0::2]
    var_labels = {nd.variable.name: nd.labels[0] for nd in nodes}

    group_items = [
        it for it in with_clause.items if not is_aggregate(it.expression)
    ]
    agg_items = [it for it in with_clause.items if is_aggregate(it.expression)]
    if len(group_items) != 1 or not agg_items:
        return None
    group_item = group_items[0]
    if not isinstance(group_item.expression, Variable):
        return None
    group_var = group_item.expression.name
    # A bare, originally-unaliased grouping item is what this shape needs --
    # but the pandas engine's projection planner mutates a cached AST's None
    # alias to the variable's own name as a side effect of executing the
    # *same* query text once via that path (ASTConverter caches by text via
    # lru_cache, so this mutation is visible here too). Treating alias in
    # (None, group_var) as equivalent avoids this check flipping based on
    # execution history; an alias renaming to anything else is still
    # rejected (out of scope for this slice, per the plan).
    if group_item.alias is not None and group_item.alias != group_var:
        return None  # first slice: no renaming the grouping var through WITH
    if group_var not in variables:
        return None
    if any(it.alias is None for it in agg_items):
        return None

    label = var_labels.get(group_var)
    if label is None or label not in context._streaming_sources:
        return None

    if not set_clause.items:
        return None
    agg_aliases = {it.alias for it in agg_items}
    if not all(
        item.variable is not None
        and item.variable.name == group_var
        and item.property is not None
        and item.property not in ("*", "*+")
        and isinstance(item.expression, Variable)
        and item.expression.name in agg_aliases
        and not item.labels
        for item in set_clause.items
    ):
        return None

    return (
        match,
        agg_items,
        set_clause,
        label,
        group_var,
        variables,
        build,
        inline_preds,
    )


def is_relation_group_set_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is an aggregate-then-``SET`` mutation the
    relation engine can execute as a native ``UPDATE ... FROM`` (see
    :func:`_analyze_group_set_query`).
    """
    from pycypher.relation_sql import compile_aggregate, compile_expression

    info = _analyze_group_set_query(query, context)
    if info is None:
        return False
    (
        match,
        agg_items,
        set_clause,
        _label,
        group_var,
        variables,
        _build,
        inline_preds,
    ) = info

    udfs = _udf_names(context)
    resolve = _make_resolve(variables)
    if (
        match.where is not None
        and compile_expression(match.where, resolve, udfs) is None
    ):
        return False
    if _compile_inline_predicates(inline_preds, resolve, udfs) is None:
        return False
    for it in agg_items:
        if compile_aggregate(it.expression, resolve, udfs, _no_var) is None:
            return False
    for item in set_clause.items:
        if resolve(
            group_var, item.property
        ) is None and not _new_column_name_allowed(item.property):
            return False
    return True


def execute_relation_group_set(query: Any, context: Context) -> None:
    """Execute an eligible aggregate-then-``SET`` mutation as a native
    ``UPDATE ... FROM`` against a grouped aggregate view.

    Precondition: :func:`is_relation_group_set_eligible` returned ``True``
    for *query*. No return value, matching the other mutation executors.
    Groups the pattern's joined relation (an INNER join, from
    :func:`_analyze_leading_pattern`) by the ``SET`` target's id column into
    a temporary view, then updates from it — a target row with zero matches
    is simply absent from the aggregate view and left untouched by the
    ``UPDATE``, matching Cypher's own zero-match-rows semantics (empirically
    verified against the pandas engine: no ``COALESCE``/zero-fill). A
    target property not yet in the entity's schema is created via
    :func:`_ensure_column` (Phase 3a) before being assigned.
    """
    import uuid

    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_aggregate, compile_expression

    (
        match,
        agg_items,
        set_clause,
        label,
        group_var,
        variables,
        build,
        inline_preds,
    ) = _analyze_group_set_query(query, context)  # type: ignore[misc]
    udfs = _udf_names(context)
    resolve = _make_resolve(variables)
    con = context.backend.connection

    rel = build(con)
    if match.where is not None:
        rel = rel.filter(compile_expression(match.where, resolve, udfs))  # type: ignore[arg-type]
    for pred in _compile_inline_predicates(inline_preds, resolve, udfs) or []:
        rel = rel.filter(pred)

    group_table_alias, group_attr = variables[group_var]
    id_col = _node_id_column(context, label)
    group_col_sql = f'{group_table_alias}."{id_col}"'

    select_parts = [f'{group_col_sql} AS "__grp_id__"']
    for it in agg_items:
        agg_sql = compile_aggregate(it.expression, resolve, udfs, _no_var)
        select_parts.append(f"{agg_sql} AS {_quote_output_alias(it.alias)}")

    agg_rel = rel.aggregate(", ".join(select_parts), group_col_sql)
    view_name = f"__pycypher_group_set_{uuid.uuid4().hex}__"
    agg_rel.to_view(view_name, replace=True)

    table = physical_table_name(label)
    for item in set_clause.items:
        if item.property not in group_attr:
            probe = agg_rel.project(
                _quote_output_alias(item.expression.name)
            ).limit(0)
            _ensure_column(context, label, item.property, probe)

    set_sql_parts = [
        f'"{group_attr[item.property]}" = sub.{_quote_output_alias(item.expression.name)}'
        for item in set_clause.items
    ]
    sql = (
        f'UPDATE "{table}" SET {", ".join(set_sql_parts)} '  # nosec B608 — table/columns validated identifiers; aggregate SQL from relation_sql's whitelisted compiler; view_name is an internally generated uuid4, not user input
        f'FROM "{view_name}" AS sub '
        f'WHERE "{table}"."{id_col}" = sub."__grp_id__"'
    )

    LOGGER.debug("[duckdb-relation] %s", sql)
    try:
        con.execute(sql)
    finally:
        con.execute(f'DROP VIEW IF EXISTS "{view_name}"')


def _analyze_copy_set_query(
    query: Any,
    context: Context,
) -> (
    tuple[
        Any,
        str,
        str,
        dict[str, tuple[str, dict[str, str]]],
        Any,
        list[tuple[str, str, Any]],
        dict[str, str],
        list[tuple[str, Any]],
    ]
    | None
):
    """Return a plan for a relationship property-copy ``SET`` mutation
    (no aggregation), or ``None`` (docs/fastopendata_streaming_qualification_plan.md,
    Phase 2b category (B)).

    Eligible shape: a required, non-optional leading ``MATCH`` (single hop
    or fixed-length path, see :func:`_analyze_leading_pattern`) followed by
    either:

    - no further clauses but a ``SET`` whose items all target the *same*
      bound variable, each assigning any compilable expression over the
      pattern's bound variables (the ``tract_rucc_code`` shape,
      ``MATCH (t)-[:REL]->(c) SET t.x = c.y``); or
    - one ``WITH`` naming a bound-variable expression before assigning it —
      exactly one bare, unaliased pass-through of the target variable plus
      one or more non-aggregate scalar items each with an alias — followed
      by a ``SET`` whose items target that same variable and each assign a
      bare alias from the ``WITH`` (the more common shape,
      ``MATCH (e)-[:REL]->(c) WITH c, e.prop AS a SET c.x = a``).

    Aggregating ``WITH`` items are out of scope here (that's
    :func:`_analyze_group_set_query`'s job); a single-node pattern with no
    relationship is also out of scope (:func:`_analyze_scalar_set_query`'s
    job) since :func:`_analyze_leading_pattern` requires at least two bound
    variables to have anything to copy from.

    Returns ``(match, target_var, label, variables, build, inline_preds,
    var_labels, set_pairs)`` — *set_pairs* is ``[(property_name, expr_ast),
    ...]``, with each *expr_ast* already resolved to the expression that
    should be compiled against the pattern-scope resolve (the ``WITH``
    alias indirection, if any, is substituted away here so both shapes
    reduce to the same representation); *var_labels* maps every bound node
    variable to its entity label (needed by the executor to resolve every
    matched variable's id column, not just the target's). Callers must
    still compile ``match.where``, the inline predicates, and each
    *set_pairs* expression to SQL (see
    :func:`is_relation_copy_set_eligible` /
    :func:`execute_relation_copy_set`).
    """
    from pycypher.ast_models import Match, Query, Set, Variable, With
    from pycypher.relation_sql import is_aggregate

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) not in (2, 3):
        return None
    match = clauses[0]
    if not isinstance(match, Match) or match.optional:
        return None

    counter = [0]

    def alias_gen() -> str:
        alias = f"v{counter[0]}"
        counter[0] += 1
        return alias

    lead = _analyze_leading_pattern(match, context, alias_gen)
    if lead is None:
        return None
    variables, build, inline_preds, _var_labels = lead
    if len(variables) < 2:
        return None  # a relationship is required here; single-node is _analyze_scalar_set_query's job

    nodes = match.pattern.paths[0].elements[0::2]
    var_labels = {nd.variable.name: nd.labels[0] for nd in nodes}

    if len(clauses) == 2:
        set_clause = clauses[1]
        if not isinstance(set_clause, Set) or not set_clause.items:
            return None
        target_vars = {
            item.variable.name
            for item in set_clause.items
            if item.variable is not None
        }
        if len(target_vars) != 1:
            return None
        target_var = next(iter(target_vars))
        if not all(
            item.variable is not None
            and item.property is not None
            and item.property not in ("*", "*+")
            and item.expression is not None
            and not item.labels
            for item in set_clause.items
        ):
            return None
        set_pairs = [
            (item.property, item.expression) for item in set_clause.items
        ]
    else:
        with_clause, set_clause = clauses[1], clauses[2]
        if not isinstance(with_clause, With) or not isinstance(
            set_clause, Set
        ):
            return None
        if (
            with_clause.distinct
            or with_clause.order_by
            or with_clause.skip is not None
            or with_clause.limit is not None
            or getattr(with_clause, "where", None) is not None
            or not with_clause.items
        ):
            return None
        if any(is_aggregate(it.expression) for it in with_clause.items):
            return None  # aggregates are _analyze_group_set_query's job

        def _is_bound_passthrough(it: Any) -> bool:
            # A bare, originally-unaliased pass-through is what this checks
            # for -- but the pandas engine's projection planner mutates a
            # cached AST's None alias to the variable's own name as a side
            # effect of executing the *same* query text once via that path
            # (ASTConverter caches by text via lru_cache, so this mutation
            # is visible here too). Treating alias in (None, var name) as
            # equivalent avoids this check flipping based on execution
            # history.
            if not isinstance(it.expression, Variable):
                return False
            name = it.expression.name
            if it.alias is not None and it.alias != name:
                return False
            return name in variables

        passthrough_items = [
            it for it in with_clause.items if _is_bound_passthrough(it)
        ]
        if len(passthrough_items) != 1:
            return None
        target_var = passthrough_items[0].expression.name
        scalar_items = [
            it for it in with_clause.items if not _is_bound_passthrough(it)
        ]
        if not scalar_items or any(it.alias is None for it in scalar_items):
            return None

        if not set_clause.items:
            return None
        scalar_expr_by_alias = {it.alias: it.expression for it in scalar_items}
        if not all(
            item.variable is not None
            and item.variable.name == target_var
            and item.property is not None
            and item.property not in ("*", "*+")
            and isinstance(item.expression, Variable)
            and item.expression.name in scalar_expr_by_alias
            and not item.labels
            for item in set_clause.items
        ):
            return None
        set_pairs = [
            (item.property, scalar_expr_by_alias[item.expression.name])
            for item in set_clause.items
        ]

    if target_var not in variables:
        return None
    label = var_labels.get(target_var)
    if label is None or label not in context._streaming_sources:
        return None

    return (
        match,
        target_var,
        label,
        variables,
        build,
        inline_preds,
        var_labels,
        set_pairs,
    )


def is_relation_copy_set_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a relationship property-copy ``SET`` (no
    aggregation) the relation engine can execute as a native
    ``UPDATE ... FROM`` (see :func:`_analyze_copy_set_query`).
    """
    from pycypher.relation_sql import compile_expression

    info = _analyze_copy_set_query(query, context)
    if info is None:
        return False
    (
        match,
        target_var,
        _label,
        variables,
        _build,
        inline_preds,
        _var_labels,
        set_pairs,
    ) = info

    udfs = _udf_names(context)
    resolve = _make_resolve(variables)
    if (
        match.where is not None
        and compile_expression(match.where, resolve, udfs) is None
    ):
        return False
    if _compile_inline_predicates(inline_preds, resolve, udfs) is None:
        return False
    for prop, expr in set_pairs:
        if resolve(target_var, prop) is None and not _new_column_name_allowed(
            prop
        ):
            return False
        if compile_expression(expr, resolve, udfs, _no_var) is None:
            return False
    return True


def execute_relation_copy_set(query: Any, context: Context) -> None:
    """Execute an eligible relationship property-copy ``SET`` mutation as a
    native ``UPDATE ... FROM`` against a one-row-per-target view.

    Precondition: :func:`is_relation_copy_set_eligible` returned ``True``
    for *query*. No return value, matching the other mutation executors.
    Joins the pattern (an INNER join, from :func:`_analyze_leading_pattern`),
    then collapses to at most one row per target via
    ``QUALIFY ROW_NUMBER() OVER (PARTITION BY <target id> ORDER BY <the
    other matched vars' id columns>) = 1`` — a deterministic pick, not an
    attempt to replicate the pandas engine's undocumented, source-table-row-
    order-dependent tie-break for the (never-happens-in-this-pipeline's-real-
    data) multi-match case; see the "(B)" correctness write-up in
    docs/fastopendata_streaming_qualification_plan.md. A target property
    not yet in the entity's schema is created via :func:`_ensure_column`
    (Phase 3a) before being assigned.
    """
    import uuid

    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_expression

    (
        match,
        target_var,
        label,
        variables,
        build,
        inline_preds,
        var_labels,
        set_pairs,
    ) = _analyze_copy_set_query(query, context)  # type: ignore[misc]
    udfs = _udf_names(context)
    resolve = _make_resolve(variables)
    con = context.backend.connection

    rel = build(con)
    if match.where is not None:
        rel = rel.filter(compile_expression(match.where, resolve, udfs))  # type: ignore[arg-type]
    for pred in _compile_inline_predicates(inline_preds, resolve, udfs) or []:
        rel = rel.filter(pred)

    target_alias, target_attr = variables[target_var]
    target_id_col = _node_id_column(context, label)
    target_id_sql = f'{target_alias}."{target_id_col}"'

    other_vars = [v for v in var_labels if v != target_var]

    select_parts = [f'{target_id_sql} AS "__tgt_id__"']
    tie_cols: list[str] = []
    for i, v in enumerate(other_vars):
        v_alias = variables[v][0]
        v_id_col = _node_id_column(context, var_labels[v])
        tie_col = f"__tie_{i}__"
        tie_cols.append(tie_col)
        select_parts.append(f'{v_alias}."{v_id_col}" AS "{tie_col}"')

    val_cols: list[
        tuple[str, str]
    ] = []  # (physical target column, projected value column)
    for i, (prop, expr) in enumerate(set_pairs):
        expr_sql = compile_expression(expr, resolve, udfs, _no_var)
        val_col = f"__val_{i}__"
        if prop not in target_attr:
            probe = rel.project(f"{expr_sql} AS __probe__").limit(0)
            _ensure_column(context, label, prop, probe)
        val_cols.append((target_attr[prop], val_col))
        select_parts.append(f'{expr_sql} AS "{val_col}"')

    projected = rel.project(", ".join(select_parts))

    src_view = f"__pycypher_copy_set_src_{uuid.uuid4().hex}__"
    order_clause = ""
    if tie_cols:
        order_clause = " ORDER BY " + ", ".join(f'"{c}"' for c in tie_cols)
    qualify_sql = (
        f'SELECT * FROM "{src_view}" '
        f'QUALIFY ROW_NUMBER() OVER (PARTITION BY "__tgt_id__"{order_clause}) = 1'
    )
    deduped = projected.query(src_view, qualify_sql)

    final_view = f"__pycypher_copy_set_{uuid.uuid4().hex}__"
    deduped.to_view(final_view, replace=True)

    table = physical_table_name(label)
    set_sql_parts = [f'"{col}" = sub."{val_col}"' for col, val_col in val_cols]
    sql = (
        f'UPDATE "{table}" SET {", ".join(set_sql_parts)} '  # nosec B608 — table/columns validated identifiers; values from relation_sql's whitelisted compiler; view names are internally generated uuid4s, not user input
        f'FROM "{final_view}" AS sub '
        f'WHERE "{table}"."{target_id_col}" = sub."__tgt_id__"'
    )

    LOGGER.debug("[duckdb-relation] %s", sql)
    try:
        con.execute(sql)
    finally:
        con.execute(f'DROP VIEW IF EXISTS "{final_view}"')


def _analyze_delete_query(
    query: Any,
    context: Context,
) -> tuple[Any, Any, str, str, Any] | None:
    """Return ``(match, delete_clause, label, var_name, resolve)`` if *query*
    is a single-table ``DELETE``-eligible mutation, else ``None``.

    Eligible shape: exactly ``MATCH (var:Label) [WHERE ...] DELETE var`` —
    see :func:`_analyze_single_node_match` for the shared MATCH-shape checks,
    plus a non-``DETACH`` ``DELETE`` whose sole expression is a ``Variable``
    matching *var* (mirrors the pandas ``process_delete``'s Variable-only
    handling, ``mutation_engine.py:647-666``).  ``DETACH DELETE`` and
    deleting anything other than the bound node (e.g. a relationship
    variable, or multiple expressions) are out of scope for this slice.
    """
    from pycypher.ast_models import Delete, Variable

    info = _analyze_single_node_match(query, context, Delete)
    if info is None:
        return None
    match, delete_clause, label, var_name, resolve = info
    if delete_clause.detach:
        return None
    if len(delete_clause.expressions) != 1:
        return None
    expr = delete_clause.expressions[0]
    if not (isinstance(expr, Variable) and expr.name == var_name):
        return None

    return match, delete_clause, label, var_name, resolve


def is_relation_delete_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a single-table ``DELETE`` the relation
    engine can execute as a native ``DELETE FROM`` (see
    :func:`_analyze_delete_query`).
    """
    from pycypher.relation_sql import compile_expression

    info = _analyze_delete_query(query, context)
    if info is None:
        return False
    match, _delete_clause, _label, _var_name, resolve = info

    udfs = _udf_names(context)
    if (
        match.where is not None
        and compile_expression(match.where, resolve, udfs) is None
    ):
        return False
    node = match.pattern.paths[0].elements[0]
    return (
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        is not None
    )


def execute_relation_delete(query: Any, context: Context) -> None:
    """Execute an eligible single-table ``DELETE`` as a native ``DELETE FROM``.

    Precondition: :func:`is_relation_delete_eligible` returned ``True`` for
    *query*. No return value — mirrors the pandas path's convention of an
    empty result for a non-``RETURN``-terminal mutation query.  Compiles the
    ``MATCH``'s ``WHERE`` + inline predicates straight into the ``DELETE``'s
    ``WHERE`` (no ``id IN (...)`` subquery) since the eligible shape only
    ever has the one bound variable being deleted.
    """
    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_expression

    match, _delete_clause, label, _var_name, resolve = _analyze_delete_query(  # type: ignore[misc]
        query,
        context,
    )
    udfs = _udf_names(context)

    where_parts: list[str] = []
    if match.where is not None:
        where_parts.append(compile_expression(match.where, resolve, udfs))  # type: ignore[arg-type]
    node = match.pattern.paths[0].elements[0]
    where_parts.extend(
        _compile_inline_predicates(_inline_predicates([node]), resolve, udfs)
        or [],
    )

    table = physical_table_name(label)
    sql = f'DELETE FROM "{table}"'  # nosec B608 — table validated by validate_identifier; where parts built exclusively from relation_sql.compile_expression's whitelisted compiler
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    LOGGER.debug("[duckdb-relation] %s", sql)
    context.backend.connection.execute(sql)


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


def _analyze_create_query(
    query: Any,
    context: Context,
) -> tuple[str, dict[str, Any]] | None:
    """Return ``(label, properties)`` if *query* is a standalone single-node
    ``CREATE``-eligible mutation, else ``None``.

    Eligible shape: exactly ``CREATE (var:Label {...})`` — a single clause
    (no preceding ``MATCH``), a single path, a single node with exactly one
    label and no relationship.  Row-per-matched-row ``CREATE`` (following a
    ``MATCH``) and relationship ``CREATE``, both supported by the pandas
    ``process_create``, are out of scope for this slice.  *Label* must have
    a registered streaming source, and every property key must already
    resolve to an existing column via :func:`_entity_attr_map` — creating a
    brand-new property column would require ``ALTER TABLE``, which this
    slice does not attempt (an unknown property key falls back to pandas,
    which grows the shadow schema dynamically).
    """
    from pycypher.ast_models import Create, NodePattern, Query

    if getattr(context, "backend_name", None) != "duckdb":
        return None
    if not hasattr(getattr(context, "backend", None), "connection"):
        return None
    if not isinstance(query, Query):
        return None
    clauses = query.clauses
    if len(clauses) != 1:
        return None
    create = clauses[0]
    if not isinstance(create, Create) or create.pattern is None:
        return None
    paths = create.pattern.paths
    if len(paths) != 1:
        return None
    path = paths[0]
    if path.variable is not None:
        return None
    elements = path.elements
    if len(elements) != 1:
        return None
    node = elements[0]
    if not (isinstance(node, NodePattern) and len(node.labels) == 1):
        return None

    label = node.labels[0]
    if label not in context._streaming_sources:
        return None
    attr = _entity_attr_map(context, label)
    if attr is None:
        return None
    if any(key not in attr for key in node.properties):
        return None

    return label, node.properties


def is_relation_create_eligible(query: Any, context: Context) -> bool:
    """Return True if *query* is a standalone single-node ``CREATE`` the
    relation engine can execute as a native ``INSERT`` (see
    :func:`_analyze_create_query`).
    """
    from pycypher.relation_sql import compile_expression

    info = _analyze_create_query(query, context)
    if info is None:
        return False
    label, properties = info
    if not _streaming_id_is_integer(context, label):
        return False

    udfs = _udf_names(context)
    return all(
        compile_expression(expr, _no_prop, udfs, resolve_var=_no_var)
        is not None
        for expr in properties.values()
    )


def execute_relation_create(query: Any, context: Context) -> None:
    """Execute an eligible standalone single-node ``CREATE`` as a native
    ``INSERT``.

    Precondition: :func:`is_relation_create_eligible` returned ``True`` for
    *query*. No return value — mirrors the pandas path's convention of an
    empty result for a non-``RETURN``-terminal mutation query.  When *label*
    has a registered ``id_col``, generates the new ID via a DuckDB
    ``SEQUENCE`` (created lazily, idempotently — ``START`` only applies the
    first time — seeded above the current max), replacing
    ``MutationEngine._next_ids``'s pandas max-scan for this path only; the
    pandas path itself is untouched.
    """
    from pycypher.backends._helpers import validate_identifier
    from pycypher.backends.table_registry import physical_table_name
    from pycypher.relation_sql import compile_expression

    label, properties = _analyze_create_query(query, context)  # type: ignore[misc]
    attr = _entity_attr_map(context, label)  # type: ignore[assignment]
    udfs = _udf_names(context)
    con = context.backend.connection
    table = physical_table_name(label)

    cols: list[str] = []
    values: list[str] = []

    id_col = _streaming_id_col(context, label)
    if id_col is not None:
        quoted_id_col = validate_identifier(id_col)
        seq = f"_streaming_seq_{validate_identifier(label)}"
        max_id = con.execute(
            f'SELECT COALESCE(MAX("{quoted_id_col}"), 0) FROM "{table}"',  # nosec B608 — table/id_col validated identifiers
        ).fetchone()[0]
        con.execute(
            f'CREATE SEQUENCE IF NOT EXISTS "{seq}" START {int(max_id) + 1}',  # nosec B608 — seq name validated; start value is an int, not user SQL
        )
        cols.append(quoted_id_col)
        values.append(f"nextval('{seq}')")

    for key, expr in properties.items():
        cols.append(validate_identifier(attr[key]))
        values.append(
            compile_expression(expr, _no_prop, udfs, resolve_var=_no_var)
        )  # type: ignore[arg-type]

    col_list = ", ".join(f'"{c}"' for c in cols)
    val_list = ", ".join(values)
    sql = f'INSERT INTO "{table}" ({col_list}) SELECT {val_list}'  # nosec B608 — table/columns validated identifiers; values from relation_sql.compile_expression's whitelisted compiler or nextval() sequence call
    LOGGER.debug("[duckdb-relation] %s", sql)
    con.execute(sql)


def is_relation_mutation_eligible(query: Any, context: Context) -> str | None:
    """Return ``"set"``/``"scalar_set"``/``"group_set"``/``"copy_set"``/
    ``"create"``/``"delete"`` if *query* is an eligible single-table
    mutation the relation engine can execute natively, else ``None``.
    Checked in that order; a single query shape can only ever match one
    kind.
    """
    if is_relation_set_eligible(query, context):
        return "set"
    if is_relation_scalar_set_eligible(query, context):
        return "scalar_set"
    if is_relation_group_set_eligible(query, context):
        return "group_set"
    if is_relation_copy_set_eligible(query, context):
        return "copy_set"
    if is_relation_create_eligible(query, context):
        return "create"
    if is_relation_delete_eligible(query, context):
        return "delete"
    return None


def execute_relation_mutation(query: Any, context: Context, kind: str) -> None:
    """Dispatch to the native execute function for *kind* (see
    :func:`is_relation_mutation_eligible`).
    """
    if kind == "set":
        execute_relation_set(query, context)
    elif kind == "scalar_set":
        execute_relation_scalar_set(query, context)
    elif kind == "group_set":
        execute_relation_group_set(query, context)
    elif kind == "copy_set":
        execute_relation_copy_set(query, context)
    elif kind == "create":
        execute_relation_create(query, context)
    elif kind == "delete":
        execute_relation_delete(query, context)
    else:
        msg = f"Unknown relation mutation kind: {kind!r}"
        raise ValueError(msg)
