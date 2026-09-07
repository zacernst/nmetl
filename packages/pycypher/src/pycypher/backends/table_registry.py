"""Registry of DuckDB tables materialised from a ``Context``'s sources.

Phase 1a of ``docs/duckdb_eager_path_design.md``.

Before this module, the only way a source became a real DuckDB table was
:func:`pycypher.relation_engine.register_streaming_source`, which covered
**entities only**, was called from exactly one place
(``cli/pipeline.py``'s ``_try_streaming_run``), and recomputed the physical
table name at four separate call sites.  Everything else — every entity in
a normally-built ``Context``, and every relationship without exception —
was re-converted from pandas/Arrow into a throwaway DuckDB relation on
*every query* (``relation_engine._base_relation``).

:class:`TableRegistry` is the single owner of that materialisation: one
table per (kind, label), created once, named in one place, with the ID
column and its declared DuckDB type recorded so downstream SQL doesn't have
to re-derive them.

The registry deliberately does **not** import
:mod:`pycypher.relational_models`.  It takes relations and metadata, not
``EntityTable`` objects, so it can be owned by ``DuckDBBackend`` without a
circular import.  :func:`register_context_tables` is the adapter that knows
about the Context model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shared.logger import LOGGER

from pycypher.backends._helpers import validate_identifier
from pycypher.backends.duckdb_backend import DuckDBLazyFrame

if TYPE_CHECKING:
    from pycypher.relational_models import Context

#: Table kinds.  Entity and relationship labels live in separate namespaces
#: in Cypher, so the registry keys on ``(kind, label)`` rather than label
#: alone — a graph may legitimately have a node label and a relationship
#: type spelled identically.
ENTITY_KIND = "entity"
RELATIONSHIP_KIND = "relationship"

#: Physical table-name prefix per kind.  The entity prefix is unchanged from
#: ``register_streaming_source``'s original literal so that any DuckDB
#: database or log from before this module still reads the same.
_TABLE_PREFIX: dict[str, str] = {
    ENTITY_KIND: "_streaming_source_",
    RELATIONSHIP_KIND: "_rel_source_",
}


def physical_table_name(label: str, kind: str = ENTITY_KIND) -> str:
    """Return the DuckDB table name for *label* of *kind*.

    Args:
        label: The entity or relationship label.
        kind: :data:`ENTITY_KIND` or :data:`RELATIONSHIP_KIND`.

    Returns:
        The physical table name.

    Raises:
        ValueError: If *label* is not a safe SQL identifier, or *kind* is
            not a recognised kind.

    """
    try:
        prefix = _TABLE_PREFIX[kind]
    except KeyError:
        msg = (
            f"Unknown table kind {kind!r}; "
            f"expected one of {sorted(_TABLE_PREFIX)}."
        )
        raise ValueError(msg) from None
    return f"{prefix}{validate_identifier(label)}"


@dataclass(frozen=True, slots=True)
class RegisteredTable:
    """A source materialised as a real, mutable DuckDB table.

    Attributes:
        label: The entity or relationship label.
        kind: :data:`ENTITY_KIND` or :data:`RELATIONSHIP_KIND`.
        table_name: The physical DuckDB table name.
        attr_map: Cypher property name → column name in the table.
        id_col: The column holding the entity/relationship identity, or
            ``None`` when the source has no designated ID column.  For
            sources derived from a ``Context``'s mappings this is
            ``"__ID__"``; for file-backed streaming sources it is whatever
            raw column the caller nominated.
        id_type: The declared DuckDB type of *id_col* (e.g. ``"BIGINT"``),
            or ``None`` when there is no ID column.  Recorded so downstream
            SQL can emit consistent casts instead of rediscovering the type
            after a pandas round-trip has already mangled it.
        frame: A :class:`~pycypher.backends.duckdb_backend.DuckDBLazyFrame`
            over the table.

    .. warning::
       :attr:`relation` re-executes against the table and therefore always
       sees current contents, including rows changed by a later
       ``UPDATE``/``INSERT``/``DELETE``.  :attr:`frame` does **not**:
       ``DuckDBLazyFrame`` caches its first ``to_pandas()`` result for the
       lifetime of the object, so a frame materialised before a mutation
       keeps serving the pre-mutation rows.  Read through :attr:`relation`
       (or :meth:`TableRegistry.relation`) whenever a mutation may have
       intervened; :attr:`frame` exists for the
       ``context._streaming_sources`` alias and for callers that materialise
       exactly once.

    """

    label: str
    kind: str
    table_name: str
    attr_map: dict[str, str]
    id_col: str | None
    id_type: str | None
    frame: DuckDBLazyFrame

    @property
    def relation(self) -> Any:
        """The underlying DuckDB relation over this table."""
        return self.frame.relation

    @property
    def columns(self) -> list[str]:
        """Column names of the materialised table."""
        return list(self.frame.columns)


class TableRegistry:
    """Owns the DuckDB tables materialised for one connection.

    Held by :class:`~pycypher.backends.duckdb_backend.DuckDBBackend` and
    reached as ``backend.tables``.  Re-registering a label **merges** into
    the existing table rather than replacing it (Phase 3c —
    the FastOpenData streaming-qualification plan (private repository)), so a label produced
    by more than one source (e.g. an entity enriched by several files
    sharing one identity, or a relationship type whose edges come from
    several files) ends up with all of their data rather than only the
    last-registered source's. See :meth:`register_relation` for exactly
    what "merge" means per kind, and the one case where it still replaces
    (no ``id_col`` on one side to merge entities on).
    """

    __slots__ = ("_con", "_tables")

    def __init__(self, connection: Any) -> None:
        """Create a registry bound to *connection*.

        Args:
            connection: An open ``duckdb.DuckDBPyConnection``.  The registry
                does not own its lifecycle.

        """
        self._con = connection
        self._tables: dict[tuple[str, str], RegisteredTable] = {}

    # -- Registration ---------------------------------------------------

    def register_relation(
        self,
        label: str,
        relation: Any,
        *,
        kind: str = ENTITY_KIND,
        id_col: str | None = None,
        attr_map: dict[str, str] | None = None,
    ) -> RegisteredTable:
        """Materialise *relation* as a real table and record it under *label*.

        The materialisation is a ``CREATE TABLE AS SELECT`` driven by the
        relation, so a lazy file scan (``DataSource.read_relation``) streams
        into the table without ever building a full pandas/Arrow frame.

        When *label* (of this *kind*) is already registered, *relation* is
        **merged** into the existing table instead of replacing it (Phase
        3c — the FastOpenData streaming-qualification plan (private repository)):

        An entity registration with an *id_col* keeps only the **first row
        per id** (file order), dropping later duplicates with a warning —
        the same contract as the eager path's ``normalize_entity_table``
        and ``streaming_entity._dedup_on_id``. This matters for sources
        loaded at a finer grain than the entity (the fastopendata
        state/county/tract/PUMA crosswalk has one row per *tract* yet
        defines ``State``), and it must happen *before* a merge: joining
        two undeduplicated sides fans out to the product of their
        duplicate counts. See :meth:`_dedup_entity_table`.

        * :data:`ENTITY_KIND` — a ``FULL OUTER JOIN`` keyed by identity,
          coalescing the id column and unioning every other column. A row
          exists in the result if *either* source has it (unmatched
          columns are ``NULL``, not dropped or zero-filled). A same-named
          column on both sides is a collision — the newly registered
          source's value wins, and it's logged, since two genuinely
          different sources declaring the same property name is more
          likely a config mistake than an intentional overwrite. The
          merged table's id column keeps the *existing* registration's
          name, not the new one's, so a label's id column name never
          changes across repeated registrations. **Exception**: if either
          the existing registration or this call has no ``id_col``, there
          is no join key to merge on, so this falls back to the pre-3c
          replace behaviour (logged).
        * :data:`RELATIONSHIP_KIND` — a row-wise union (concatenation) of
          every source's edges, not a join: each source contributes
          distinct edges, not more properties for existing ones. ``__ID__``
          is re-sequenced fresh across the concatenated result regardless
          of what either side's own id was, since two sources' own ids
          would otherwise collide once combined.

        Args:
            label: The entity or relationship label.
            relation: A ``DuckDBPyRelation`` or
                :class:`~pycypher.backends.duckdb_backend.DuckDBLazyFrame`.
            kind: :data:`ENTITY_KIND` or :data:`RELATIONSHIP_KIND`.
            id_col: Column holding the identity, if any.  Excluded from a
                derived *attr_map* so semantics match the in-memory path,
                where the ID column is consumed into ``__ID__`` and is not
                itself a property.
            attr_map: Explicit property → column map.  When ``None``, an
                identity map over every non-ID column is derived from the
                materialised schema.

        Returns:
            The :class:`RegisteredTable` record.

        """
        rel = (
            relation.relation
            if isinstance(relation, DuckDBLazyFrame)
            else relation
        )
        table_name = physical_table_name(label, kind)
        existing = self._tables.get((kind, label))

        if attr_map is None:
            attr_map = {c: c for c in rel.columns if c != id_col}

        if existing is None:
            # Belt and braces against a partially-created table from a
            # failed prior run on a file-backed (scratch) database.
            self._con.execute(f'DROP TABLE IF EXISTS "{table_name}"')  # nosec B608 — name built by physical_table_name/validate_identifier
            rel.create(table_name)
            if (
                kind == ENTITY_KIND
                and id_col is not None
                and id_col in rel.columns
            ):
                self._dedup_entity_table(table_name, id_col, label)
            final_id_col = id_col
            final_attr_map = attr_map
        else:
            merge = (
                self._merge_entity_table
                if kind == ENTITY_KIND
                else self._merge_relationship_table
            )
            final_id_col = merge(existing, rel, id_col, table_name)
            final_attr_map = {**existing.attr_map, **attr_map}

        materialised = DuckDBLazyFrame(self._con.table(table_name), self._con)
        columns = list(materialised.columns)
        # Drop any attr_map entry whose column didn't survive the merge
        # (defensive — mirrors refresh()'s same cleanup).
        final_attr_map = {
            prop: column
            for prop, column in final_attr_map.items()
            if column in columns
        }

        entry = RegisteredTable(
            label=label,
            kind=kind,
            table_name=table_name,
            attr_map=final_attr_map,
            id_col=final_id_col,
            id_type=_column_type(materialised, final_id_col),
            frame=materialised,
        )
        self._tables[kind, label] = entry
        LOGGER.debug(
            "TableRegistry: materialised %s %r as %r (%d columns, id_col=%r)",
            kind,
            label,
            table_name,
            len(columns),
            final_id_col,
        )
        return entry

    def _merge_entity_table(
        self,
        existing: RegisteredTable,
        new_rel: Any,
        new_id_col: str | None,
        table_name: str,
    ) -> str | None:
        """Fold *new_rel* into *existing*'s physical table (Phase 3c).

        Returns the merged table's id column name — always *existing*'s,
        never *new_rel*'s (see :meth:`register_relation`).
        """
        old_id = existing.id_col
        if old_id is None or new_id_col is None:
            LOGGER.warning(
                "TableRegistry: entity %r has a source with no id_col — "
                "there is no join key to merge on, so this registration "
                "replaces the existing table instead of merging (pre-"
                "Phase-3c behaviour). Give every source sharing this "
                "label an explicit id_col to merge them instead.",
                existing.label,
            )
            self._con.execute(f'DROP TABLE IF EXISTS "{table_name}"')  # nosec B608 — name built by physical_table_name/validate_identifier
            new_rel.create(table_name)
            return new_id_col

        old_id = validate_identifier(old_id)
        new_id = validate_identifier(new_id_col)
        old_cols = [
            validate_identifier(c)
            for c in existing.columns
            if c != existing.id_col
        ]
        new_cols = [
            validate_identifier(c) for c in new_rel.columns if c != new_id_col
        ]
        collisions = set(old_cols) & set(new_cols)
        if collisions:
            LOGGER.warning(
                "TableRegistry: entity %r has a column-name collision "
                "across multiple sources sharing this label: %s — the "
                "newly registered source's column wins.",
                existing.label,
                sorted(collisions),
            )
        old_keep = [c for c in old_cols if c not in collisions]

        select_parts = [
            f'COALESCE(old."{old_id}", new."{new_id}") AS "{old_id}"'
        ]
        select_parts += [f'old."{c}" AS "{c}"' for c in old_keep]
        select_parts += [f'new."{c}" AS "{c}"' for c in new_cols]

        # Materialise the new side first so it can be deduplicated by id
        # (rowid order, same as a fresh registration) before the join; the
        # existing side was deduplicated when it was registered. Joining
        # the lazy scan directly would fan out on duplicate ids.
        new_tmp = f"{table_name}__new_tmp__"
        self._con.execute(f'DROP TABLE IF EXISTS "{new_tmp}"')  # nosec B608 — name derived from physical_table_name/validate_identifier plus a fixed literal suffix
        new_rel.create(new_tmp)
        try:
            self._dedup_entity_table(new_tmp, new_id, existing.label)
            sql = (
                f"SELECT {', '.join(select_parts)} "
                f'FROM "{table_name}" AS old '
                f'FULL OUTER JOIN "{new_tmp}" AS new '
                f'ON old."{old_id}" = new."{new_id}"'
            )  # nosec B608 — table_name/new_tmp from physical_table_name/validate_identifier; old_id/new_id/old_cols/new_cols all pass validate_identifier above
            self._swap_in_merged_table(self._con.sql(sql), table_name)
        finally:
            self._con.execute(f'DROP TABLE IF EXISTS "{new_tmp}"')  # nosec B608 — see above
        return old_id

    def _dedup_entity_table(
        self, table_name: str, id_col: str, label: str
    ) -> None:
        """Keep the first row per *id_col* in *table_name*; drop the rest.

        Mirrors the eager path (``arrow_utils._dedup_on_id`` and
        ``streaming_entity._dedup_on_id``): "first" is file order, taken
        from the materialised table's ``rowid`` so it is deterministic (a
        window over a parallel file scan would not be), and the rewrite
        happens on disk so it can spill. Only rewrites when duplicates
        actually exist — the check is two aggregates — and warns when it
        does, since a duplicate entity id usually means a fact table was
        loaded at the wrong grain.
        """
        quoted_id = f'"{validate_identifier(id_col)}"'
        counts = self._con.execute(
            f'SELECT count(*), count(DISTINCT {quoted_id}) FROM "{table_name}"',  # nosec B608 — identifiers validated/quoted
        ).fetchone()
        if counts is None or counts[0] == counts[1]:
            return
        before, after = int(counts[0]), int(counts[1])
        staging = f"{table_name}__dedup_tmp__"
        self._con.execute(f'DROP TABLE IF EXISTS "{staging}"')  # nosec B608 — derived name
        self._con.execute(
            f'CREATE TABLE "{staging}" AS '  # nosec B608 — identifiers validated/quoted
            'SELECT * EXCLUDE ("__row_ord__") FROM ('
            f'  SELECT *, rowid AS "__row_ord__" FROM "{table_name}"'
            ") QUALIFY row_number() OVER ("
            f'  PARTITION BY {quoted_id} ORDER BY "__row_ord__"'
            ') = 1 ORDER BY "__row_ord__"',
        )
        self._con.execute(f'DROP TABLE "{table_name}"')  # nosec B608 — validated name
        self._con.execute(f'ALTER TABLE "{staging}" RENAME TO "{table_name}"')  # nosec B608 — validated names
        LOGGER.warning(
            "TableRegistry: entity %r dropped %d duplicate %r rows (%d → %d). "
            "An entity's id must be unique; the first occurrence is kept. "
            "If you loaded a fact table at the wrong grain, project to the "
            "entity grain via the source's `query` field.",
            label,
            before - after,
            id_col,
            before,
            after,
        )

    def _merge_relationship_table(
        self,
        existing: RegisteredTable,  # noqa: ARG002 — kept for a uniform merge-function signature with _merge_entity_table
        new_rel: Any,
        _new_id_col: str | None,
        table_name: str,
    ) -> str:
        """Fold *new_rel*'s rows into *existing*'s physical table (Phase 3c).

        A row-wise union, not a join — see :meth:`register_relation`.
        Returns ``"__ID__"``, the only id column a relationship ever has.
        """
        sql = (
            "SELECT (ROW_NUMBER() OVER (ORDER BY __branch__, __ord__) - 1) "
            'AS "__ID__", * EXCLUDE ("__ID__", __branch__, __ord__) FROM ('
            "SELECT 0 AS __branch__, ROW_NUMBER() OVER () AS __ord__, * "
            f'FROM "{table_name}" '
            "UNION ALL BY NAME "
            "SELECT 1 AS __branch__, ROW_NUMBER() OVER () AS __ord__, * "
            "FROM new)"
        )  # nosec B608 — table_name from physical_table_name/validate_identifier; no other identifiers interpolated
        merged_rel = new_rel.query("new", sql)
        self._swap_in_merged_table(merged_rel, table_name)
        return "__ID__"

    def _swap_in_merged_table(self, merged_rel: Any, table_name: str) -> None:
        """Materialise *merged_rel* and swap it in for *table_name*."""
        tmp_name = f"{table_name}__merge_tmp__"
        self._con.execute(f'DROP TABLE IF EXISTS "{tmp_name}"')  # nosec B608 — name derived from physical_table_name/validate_identifier plus a fixed literal suffix
        merged_rel.create(tmp_name)
        self._con.execute(f'DROP TABLE "{table_name}"')  # nosec B608 — name built by physical_table_name/validate_identifier
        self._con.execute(f'ALTER TABLE "{tmp_name}" RENAME TO "{table_name}"')  # nosec B608 — both names derived from physical_table_name/validate_identifier plus a fixed literal suffix

    def register_source_object(
        self,
        label: str,
        source_obj: Any,
        *,
        kind: str = ENTITY_KIND,
        id_col: str | None = None,
        attr_map: dict[str, str] | None = None,
    ) -> RegisteredTable:
        """Materialise an in-memory ``source_obj`` as a table.

        Accepts a pandas DataFrame or an Arrow table — the two shapes a
        ``Context``'s ``EntityTable.source_obj`` / ``RelationshipTable
        .source_obj`` actually take.  File-backed sources should go through
        :meth:`register_relation` with ``DataSource.read_relation`` instead,
        which streams rather than materialising the source first.

        Args:
            label: The entity or relationship label.
            source_obj: A ``pd.DataFrame`` or ``pa.Table``.
            kind: :data:`ENTITY_KIND` or :data:`RELATIONSHIP_KIND`.
            id_col: Column holding the identity, if any.
            attr_map: Explicit property → column map (see
                :meth:`register_relation`).

        Returns:
            The :class:`RegisteredTable` record.

        """
        return self.register_relation(
            label,
            _relation_from_source(self._con, source_obj),
            kind=kind,
            id_col=id_col,
            attr_map=attr_map,
        )

    def refresh(self, label: str, kind: str = ENTITY_KIND) -> RegisteredTable:
        """Re-read *label*'s physical table and update its record.

        Needed when something rewrites the table behind the registry — the
        streaming-entity post-passes rebuild it to add ids or de-duplicate —
        because the cached :class:`RegisteredTable` holds a relation handle
        and a column list captured at registration time.

        Args:
            label: The registered label.
            kind: :data:`ENTITY_KIND` or :data:`RELATIONSHIP_KIND`.

        Returns:
            The refreshed record.

        Raises:
            KeyError: If *label* is not registered.

        """
        existing = self._tables[kind, label]
        materialised = DuckDBLazyFrame(
            self._con.table(existing.table_name), self._con
        )
        columns = list(materialised.columns)
        attr_map = {
            prop: column
            for prop, column in existing.attr_map.items()
            if column in columns
        }
        entry = RegisteredTable(
            label=label,
            kind=kind,
            table_name=existing.table_name,
            attr_map=attr_map,
            id_col=existing.id_col,
            id_type=_column_type(materialised, existing.id_col),
            frame=materialised,
        )
        self._tables[kind, label] = entry
        return entry

    # -- Lookup ---------------------------------------------------------

    def get(
        self, label: str, kind: str = ENTITY_KIND
    ) -> RegisteredTable | None:
        """Return the record for *label*, or ``None`` if not registered."""
        return self._tables.get((kind, label))

    def has(self, label: str, kind: str = ENTITY_KIND) -> bool:
        """Return ``True`` if *label* of *kind* is registered."""
        return (kind, label) in self._tables

    def relation(self, label: str, kind: str = ENTITY_KIND) -> Any | None:
        """Return a DuckDB relation over *label*'s table, or ``None``."""
        entry = self.get(label, kind)
        return None if entry is None else entry.relation

    def labels(self, kind: str | None = None) -> list[str]:
        """Return registered labels, optionally filtered to one *kind*."""
        return [
            label for (k, label) in self._tables if kind is None or k == kind
        ]

    def drop(self, label: str, kind: str = ENTITY_KIND) -> bool:
        """Drop *label*'s table and forget it.

        Returns:
            ``True`` if a registration was removed, ``False`` if there was
            nothing registered under *label*.

        """
        entry = self._tables.pop((kind, label), None)
        if entry is None:
            return False
        self._con.execute(f'DROP TABLE IF EXISTS "{entry.table_name}"')  # nosec B608 — name built by physical_table_name/validate_identifier
        return True

    def clear(self) -> None:
        """Drop every registered table."""
        for kind, label in list(self._tables):
            self.drop(label, kind)

    def __len__(self) -> int:
        return len(self._tables)

    def __repr__(self) -> str:
        entities = len(self.labels(ENTITY_KIND))
        rels = len(self.labels(RELATIONSHIP_KIND))
        return f"TableRegistry(entities={entities}, relationships={rels})"


# ---------------------------------------------------------------------------
# Context adapter
# ---------------------------------------------------------------------------


def register_context_tables(context: Context) -> int:
    """Materialise every entity and relationship table in *context*.

    The adapter between the Context model and the model-agnostic
    :class:`TableRegistry`.  Skips labels that are already registered — a
    file-backed streaming source registered earlier by
    :func:`~pycypher.relation_engine.register_streaming_source` is a
    *better* registration than the in-memory one this would produce, so it
    wins.

    A label whose source cannot be materialised is logged and skipped rather
    than raising: the registry is an optimisation substrate, and every
    consumer falls back to the pandas path when a label is absent.

    Args:
        context: A DuckDB-backed :class:`~pycypher.relational_models.Context`.

    Returns:
        The number of tables newly registered.

    """
    from pycypher.constants import ID_COLUMN

    registry = getattr(context.backend, "tables", None)
    if registry is None:
        return 0

    registered = 0
    sources = (
        (ENTITY_KIND, context.entity_mapping.mapping),
        (RELATIONSHIP_KIND, context.relationship_mapping.mapping),
    )
    for kind, mapping in sources:
        for label, table in mapping.items():
            if registry.has(label, kind):
                continue
            source_obj = getattr(table, "source_obj", None)
            if source_obj is None:
                continue
            try:
                registry.register_source_object(
                    label,
                    source_obj,
                    kind=kind,
                    id_col=ID_COLUMN,
                    attr_map=dict(getattr(table, "attribute_map", {}) or {}),
                )
            except Exception:  # noqa: BLE001 — substrate is best-effort; consumers fall back to pandas
                LOGGER.warning(
                    "TableRegistry: could not materialise %s %r; "
                    "queries on it will use the pandas path",
                    kind,
                    label,
                    exc_info=True,
                )
                continue
            registered += 1
    return registered


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _relation_from_source(con: Any, source_obj: Any) -> Any:
    """Return a DuckDB relation over an in-memory *source_obj*.

    Mirrors ``relation_engine._base_relation``'s conversion ladder so both
    accept exactly the same source shapes.
    """
    import pandas as pd

    if isinstance(source_obj, pd.DataFrame):
        return con.from_df(source_obj)
    try:
        import pyarrow as pa

        if isinstance(source_obj, pa.Table):
            return con.from_arrow(source_obj)
    except ImportError:
        pass
    from pycypher.backends._helpers import _to_pandas

    return con.from_df(_to_pandas(source_obj))


def _column_type(frame: DuckDBLazyFrame, column: str | None) -> str | None:
    """Return the declared DuckDB type of *column*, or ``None``.

    ``None`` both when there is no column to look up and when the named
    column is absent from the materialised schema — the caller treats an
    unknown type the same as an unspecified one.
    """
    if column is None:
        return None
    relation = frame.relation
    try:
        idx = relation.columns.index(column)
    except ValueError:
        return None
    return str(relation.types[idx]).upper()
