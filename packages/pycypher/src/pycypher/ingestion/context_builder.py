"""Fluent builder that assembles a ``Context`` from Arrow-loaded data.

Usage example::

    from pycypher.ingestion import ContextBuilder

    context = (
        ContextBuilder()
        .add_entity("Person", "people.csv", id_col="person_id")
        .add_relationship(
            "KNOWS",
            "knows.csv",
            source_col="from_id",
            target_col="to_id",
        )
        .build()
    )
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
from shared.logger import LOGGER

from pycypher.ingestion.arrow_utils import (
    infer_attribute_map,
    normalize_entity_table,
    normalize_relationship_table,
)
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relational_models import (
    RELATIONSHIP_SOURCE_COLUMN,
    RELATIONSHIP_TARGET_COLUMN,
    Context,
    EntityMapping,
    EntityTable,
    RelationshipMapping,
    RelationshipTable,
)


class ContextBuilder:
    """Fluent builder that assembles a ``Context`` from heterogeneous sources.

    Methods can be chained::

        ctx = ContextBuilder().add_entity(...).add_relationship(...).build()
    """

    def __init__(self) -> None:
        #: Each entry is ``(table, had_explicit_id_col)`` — the flag is
        #: needed at merge time (Phase 3c): two sources sharing a label can
        #: only be merged on identity if *both* declared a real id_col,
        #: since an auto-generated sequential ``__ID__`` isn't a shared
        #: identity across sources. See :func:`_merge_entity_tables`.
        self._entity_tables: list[tuple[EntityTable, bool]] = []
        self._relationship_tables: list[RelationshipTable] = []
        #: Entities deferred until ``build()`` because they were registered
        #: with ``streaming=True``: they need the DuckDB connection, which
        #: does not exist until the backend is created.  Each item is
        #: ``(entity_type, data_source, id_col)``.
        self._streaming_entities: list[tuple[str, Any, str | None]] = []

    def add_entity(
        self,
        entity_type: str,
        source: str | pd.DataFrame | pa.Table,
        *,
        id_col: str | None = None,
        query: str | None = None,
        schema_hints: dict[str, str] | None = None,
        streaming: bool = False,
    ) -> ContextBuilder:
        """Register an entity type loaded from *source*.

        Args:
            entity_type: The entity label (e.g. ``"Person"``).
            source: Data source — a file path, pandas DataFrame, or Arrow table.
            id_col: Column to use as ``__ID__``.  Defaults to auto-generated
                sequential integers.
            query: Optional SQL query applied after loading (file paths only).
            schema_hints: Optional mapping of column name → DuckDB type
                string, applied as a ``CAST`` before *query* runs.
            streaming: When ``True``, do not read the source now.  It is
                scanned straight into a DuckDB table at :meth:`build` time
                (see :mod:`pycypher.ingestion.streaming_entity`), so the rows
                never pass through pandas or Arrow — the difference between a
                bounded and an unbounded memory profile on a large source.
                Requires a file/URI *source* and ``backend="duckdb"``; both
                are checked at :meth:`build` time and fall back to an eager
                read rather than failing.

        Returns:
            ``self`` for chaining.

        """
        if streaming:
            self._streaming_entities.append(
                (
                    entity_type,
                    data_source_from_uri(
                        source, query=query, schema_hints=schema_hints
                    ),
                    id_col,
                ),
            )
            return self
        raw = data_source_from_uri(
            source, query=query, schema_hints=schema_hints
        ).read()
        table = normalize_entity_table(raw, id_col=id_col)
        entity_table = EntityTable.from_arrow(entity_type, table)
        self._entity_tables.append((entity_table, id_col is not None))
        return self

    def add_relationship(
        self,
        relationship_type: str,
        source: str | pd.DataFrame | pa.Table,
        *,
        source_col: str,
        target_col: str,
        id_col: str | None = None,
        query: str | None = None,
        allow_multi_edges: bool = False,
        schema_hints: dict[str, str] | None = None,
    ) -> ContextBuilder:
        """Register a relationship type loaded from *source*.

        Args:
            relationship_type: The relationship label (e.g. ``"KNOWS"``).
            source: Data source — a file path, pandas DataFrame, or Arrow table.
            source_col: Column that holds the source node ID.
            target_col: Column that holds the target node ID.
            id_col: Column to use as ``__ID__``.  Defaults to auto-generated
                sequential integers.
            query: Optional SQL query applied after loading (file paths only).
            allow_multi_edges: When ``False`` (default), rows with the same
                ``(source, target)`` pair are collapsed into a single edge.
                Set to ``True`` to preserve parallel edges (e.g. one row per
                transaction).
            schema_hints: Optional mapping of column name → DuckDB type
                string, applied as a ``CAST`` before *query* runs.

        Returns:
            ``self`` for chaining.

        """
        raw = data_source_from_uri(
            source, query=query, schema_hints=schema_hints
        ).read()
        table = normalize_relationship_table(
            raw,
            source_col=source_col,
            target_col=target_col,
            id_col=id_col,
            allow_multi_edges=allow_multi_edges,
        )
        rel_table = RelationshipTable.from_arrow(relationship_type, table)
        self._relationship_tables.append(rel_table)
        return self

    @classmethod
    def from_dict(
        cls,
        entity_frames: dict[str, pd.DataFrame],
        *,
        id_column: str | None = None,
    ) -> Context:
        """Build a :class:`~pycypher.relational_models.Context` from a dict of DataFrames.

        Each key is a node or relationship label; the corresponding value is a
        pandas DataFrame.  DataFrames are automatically classified:

        * If the DataFrame contains both ``__SOURCE__`` and ``__TARGET__``
          columns it is registered as a **relationship table**.
        * Otherwise it is registered as an **entity table**.

        This allows a single ``from_dict()`` call to supply both nodes and
        edges without needing the verbose ``add_entity`` / ``add_relationship``
        builder chain::

            ctx = ContextBuilder.from_dict({
                "Person": persons_df,        # entity — no __SOURCE__/__TARGET__
                "KNOWS":  knows_df,          # relationship — has both columns
            })

        Args:
            entity_frames: Mapping of label to DataFrame (entity or relationship).
            id_column: Column to use as the ``__ID__`` identity key.  Defaults
                to the standard ``ID_COLUMN`` (``"__ID__"``).  Pass the name of
                an existing column to use it as the identifier.

        Returns:
            A fully populated :class:`~pycypher.relational_models.Context`.

        Raises:
            TypeError: If any value in *entity_frames* is not a
                :class:`pandas.DataFrame`.

        """
        builder = cls()
        for label, df in entity_frames.items():
            if not isinstance(df, pd.DataFrame):
                msg = (
                    f"Expected a pandas DataFrame for label '{label}', "
                    f"got {type(df).__name__}"
                )
                from pycypher.exceptions import WrongCypherTypeError

                raise WrongCypherTypeError(
                    msg,
                )
            cols = set(df.columns)
            if (
                RELATIONSHIP_SOURCE_COLUMN in cols
                and RELATIONSHIP_TARGET_COLUMN in cols
            ):
                # DataFrame has both __SOURCE__ and __TARGET__ — treat as a
                # relationship table using the standard column names.
                builder.add_relationship(
                    label,
                    df,
                    source_col=RELATIONSHIP_SOURCE_COLUMN,
                    target_col=RELATIONSHIP_TARGET_COLUMN,
                    id_col=id_column,
                )
            else:
                builder.add_entity(label, df, id_col=id_column)
        return builder.build()

    def build(
        self,
        backend: str = "auto",
        *,
        instrument: bool = False,
        register_tables: bool | None = None,
        scratch_database: bool | None = None,
    ) -> Context:
        """Assemble and return the :class:`~pycypher.relational_models.Context`.

        Args:
            backend: Backend engine hint — ``"auto"`` (default), ``"pandas"``,
                ``"duckdb"``, or ``"polars"`` — or a ready-made
                ``BackendEngine`` instance.
            instrument: When ``True``, wrap the backend so every operation
                logs its backend name and timing at DEBUG level.
            register_tables: Whether to materialise every entity and
                relationship source into a real DuckDB table (see
                :mod:`pycypher.backends.table_registry`).  ``None`` (default)
                means "yes when the resolved backend is DuckDB, no
                otherwise".  Pass ``False`` to skip it — the cost is one
                ``CREATE TABLE AS SELECT`` per source at build time, paid
                back by every query that would otherwise re-convert the
                source from pandas/Arrow.
            scratch_database: Whether a DuckDB backend built here should be
                file-backed rather than ``:memory:``.  The registered tables
                then live on disk, and the file is deleted when the backend
                closes.  ``None`` (default) means "yes when a memory budget
                is configured", i.e. when ``PYCYPHER_DUCKDB_MEMORY_LIMIT`` is
                set — because a file-backed database only *saves* memory when
                DuckDB has a budget telling it to evict buffer pages; on its
                own it measurably costs a little instead (see
                :func:`~pycypher.backends.duckdb_backend.memory_limit_configured`).
                Pass ``True``/``False`` to decide explicitly.  Ignored when
                *backend* is already a
                :class:`~pycypher.backend_engine.BackendEngine` instance,
                since that caller has chosen its own storage.

        Returns:
            A fully populated :class:`~pycypher.relational_models.Context`.

        """
        entity_mapping = EntityMapping(
            mapping=_merge_entity_groups(self._entity_tables),
        )
        relationship_mapping = RelationshipMapping(
            mapping=_merge_relationship_groups(self._relationship_tables),
        )
        if backend == "duckdb":
            from pycypher.backends.duckdb_backend import (
                DuckDBBackend,
                create_scratch_database_path,
                memory_limit_configured,
            )

            if scratch_database is None:
                # Streaming entities exist to keep a large source off the
                # heap; putting the table it lands in back on the heap would
                # defeat that, so they imply a file-backed database.
                scratch_database = bool(self._streaming_entities) or (
                    memory_limit_configured()
                )
            if scratch_database:
                # Built here rather than left to select_backend() so the
                # scratch path and its ownership can be set; the Context
                # accepts a ready backend instance just as it accepts a hint
                # string.
                backend = DuckDBBackend(
                    database_path=create_scratch_database_path(),
                    own_database_file=True,
                )

        context = Context(
            entity_mapping=entity_mapping,
            relationship_mapping=relationship_mapping,
            backend=backend,
            instrument=instrument,
        )
        if register_tables is None:
            register_tables = context.backend_name == "duckdb"
        if register_tables:
            from pycypher.backends.table_registry import (
                register_context_tables,
            )

            register_context_tables(context)
        self._attach_streaming_entities(context)
        return context

    def _attach_streaming_entities(self, context: Context) -> None:
        """Scan deferred streaming sources into *context*'s DuckDB tables.

        Falls back to an ordinary eager read when the resolved backend is not
        DuckDB — ``streaming=True`` is a memory optimisation, not a change of
        semantics, so asking for it on a pandas context loads the source
        rather than failing.
        """
        if not self._streaming_entities:
            return

        from pycypher.ingestion.streaming_entity import (
            LazyEntitySource,
            register_streaming_entity,
        )

        registry = getattr(context.backend, "tables", None)
        for entity_type, data_source, id_col in self._streaming_entities:
            if registry is None:
                LOGGER.warning(
                    "Entity %r requested streaming=True but the backend has "
                    "no DuckDB table registry; falling back to an eager, "
                    "fully in-memory read of this source. This may exceed "
                    "available memory on large datasets. (See "
                    "the FastOpenData streaming-qualification plan (private repository), "
                    "'Phase 0'.)",
                    entity_type,
                )
                table = normalize_entity_table(
                    data_source.read(), id_col=id_col
                )
                context.entity_mapping.mapping[entity_type] = (
                    EntityTable.from_arrow(entity_type, table)
                )
                continue

            entry = register_streaming_entity(
                context.backend,
                entity_type,
                data_source,
                id_col=id_col,
            )
            source = LazyEntitySource(registry, entity_type)
            context.entity_mapping.mapping[entity_type] = EntityTable(
                entity_type=entity_type,
                source_obj=source,
                column_names=list(entry.columns),
                attribute_map=dict(entry.attr_map),
                source_obj_attribute_map=dict(entry.attr_map),
            )


# ---------------------------------------------------------------------------
# Multi-source merge (Phase 3c — docs/fastopendata_streaming_qualification_
# plan.md): a label produced by more than one source is merged rather than
# only the last-registered source surviving. Mirrors
# TableRegistry.register_relation's two merge strategies at the Arrow level,
# since this path has no DuckDB connection to lean on by default.
# ---------------------------------------------------------------------------

#: Temporary suffix used to disambiguate a stale (pre-merge) column during
#: an entity merge join, before it's dropped in favour of the newer source's
#: column of the same name. Not expected to collide with a real column name.
_STALE_SUFFIX = "__pycypher_stale__"


def _merge_entity_groups(
    entries: list[tuple[EntityTable, bool]],
) -> dict[str, EntityTable]:
    """Group *entries* by entity type and merge each group into one table."""
    groups: dict[str, list[tuple[EntityTable, bool]]] = {}
    for table, had_id_col in entries:
        groups.setdefault(table.entity_type, []).append((table, had_id_col))
    return {
        entity_type: _merge_entity_tables(group)
        for entity_type, group in groups.items()
    }


def _merge_entity_tables(group: list[tuple[EntityTable, bool]]) -> EntityTable:
    """Merge every ``EntityTable`` sharing one label into a single table.

    A column-wise union keyed by identity: a ``FULL OUTER JOIN`` on
    ``__ID__`` across every source in *group*, in registration order. A row
    exists in the result if any source has it — unmatched sources' columns
    are ``NULL`` for that row, not dropped or zero-filled (matching the SET
    semantics already established elsewhere in this project). A same-named
    column from two sources collides — the later-registered source's column
    wins, logged as a warning since it's more likely a config mistake than
    an intentional overwrite.

    Falls back to keeping only the last source (the pre-Phase-3c behaviour)
    when any source in *group* has no explicit id_col: an auto-generated
    sequential ``__ID__`` isn't a shared identity across sources, so joining
    on it would silently produce a meaningless result rather than an
    error — worse than the plain replace this falls back to.
    """
    tables = [table for table, _ in group]
    if len(tables) == 1:
        return tables[0]

    entity_type = tables[0].entity_type
    if not all(had_id_col for _, had_id_col in group):
        LOGGER.warning(
            "ContextBuilder: entity %r has %d sources sharing this label, "
            "but at least one has no explicit id_col, so there is no "
            "shared identity to merge them on. Keeping only the "
            "last-registered source's data for this label (pre-Phase-3c "
            "behaviour) -- give every source sharing a label an explicit "
            "id_col to merge them instead.",
            entity_type,
            len(tables),
        )
        return tables[-1]

    merged = tables[0].source_obj
    for other in tables[1:]:
        other_table = other.source_obj
        collisions = (
            set(merged.schema.names) & set(other_table.schema.names)
        ) - {"__ID__"}
        if collisions:
            LOGGER.warning(
                "ContextBuilder: entity %r has a column-name collision "
                "across multiple sources sharing this label: %s -- the "
                "later-registered source's column wins.",
                entity_type,
                sorted(collisions),
            )
        merged = merged.join(
            other_table,
            keys="__ID__",
            join_type="full outer",
            coalesce_keys=True,
            left_suffix=_STALE_SUFFIX,
        )
        stale = [f"{c}{_STALE_SUFFIX}" for c in collisions]
        if stale:
            merged = merged.drop_columns(stale)
    merged = merged.sort_by("__ID__")

    attribute_map = infer_attribute_map(merged)
    return EntityTable(
        entity_type=entity_type,
        source_obj=merged,
        column_names=merged.column_names,
        attribute_map=attribute_map,
        source_obj_attribute_map=attribute_map,
    )


def _merge_relationship_groups(
    tables: list[RelationshipTable],
) -> dict[str, RelationshipTable]:
    """Group *tables* by relationship type and merge each group into one."""
    groups: dict[str, list[RelationshipTable]] = {}
    for table in tables:
        groups.setdefault(table.relationship_type, []).append(table)
    return {
        relationship_type: _merge_relationship_tables(group)
        for relationship_type, group in groups.items()
    }


def _merge_relationship_tables(
    tables: list[RelationshipTable],
) -> RelationshipTable:
    """Merge every ``RelationshipTable`` sharing one label into one table.

    A row-wise union (concatenation), not a join: each source contributes
    distinct edges, not more properties for existing ones. Schemas may
    differ across sources (e.g. only one has a property column) --
    ``promote_options="default"`` fills a missing column with ``NULL``
    rather than raising. ``__ID__`` is re-sequenced fresh across the
    concatenated result regardless of what either source's own id was,
    since two sources' own ids would otherwise collide once combined.
    """
    if len(tables) == 1:
        return tables[0]

    relationship_type = tables[0].relationship_type
    combined = pa.concat_tables(
        [table.source_obj for table in tables], promote_options="default"
    )
    id_index = combined.schema.get_field_index("__ID__")
    ids = pa.array(range(combined.num_rows), type=pa.int64())
    combined = combined.set_column(
        id_index, pa.field("__ID__", pa.int64()), ids
    )

    attribute_map = infer_attribute_map(combined)
    return RelationshipTable(
        relationship_type=relationship_type,
        source_obj=combined,
        column_names=combined.column_names,
        attribute_map=attribute_map,
        source_obj_attribute_map=attribute_map,
    )
