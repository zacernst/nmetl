"""What the translator and emitter know about registered tables.

A thin, cached view over the table registry (and the in-memory
``source_obj`` fallbacks) so the rest of the package never touches
``Context`` internals directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EntityRef:
    """How an entity label is addressed in emitted SQL."""

    label: str
    sql_ref: str  # quoted name usable in FROM
    #: SQL for the id column *as referenced through an alias* (quoted
    #: column, or the ``rowid`` pseudo-column for a physical table whose
    #: source declared no id column).
    id_col: str
    attr: Any  # live property -> column mapping (mutated in place by _ensure_column)
    writable: bool  # has a real registry table a mutation can target
    id_is_integer: bool


@dataclass
class RelRef:
    """How a relationship type is addressed in emitted SQL."""

    rel_type: str
    sql_ref: str
    attr: Any
    has_label_cols: bool


class Catalog:
    """Resolve entity labels and relationship types to SQL references."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._entities: dict[str, EntityRef | None] = {}
        self._rels: dict[str, RelRef | None] = {}

    @property
    def connection(self) -> Any:
        return self.context.backend.connection

    def udf_names(self) -> frozenset[str]:
        from pycypher.relation_engine import _udf_names

        return _udf_names(self.context)

    def entity(self, label: str) -> EntityRef | None:
        if label not in self._entities:
            self._entities[label] = self._load_entity(label)
        return self._entities[label]

    def relationship(self, rel_type: str) -> RelRef | None:
        if rel_type not in self._rels:
            self._rels[rel_type] = self._load_rel(rel_type)
        return self._rels[rel_type]

    # -- loading -----------------------------------------------------------

    def _registry_entry(self, label: str, kind: str) -> Any | None:
        """The registry record for *label*, materialising an in-memory
        source into a real table first if the registry lacks it.

        A DuckDB context built by hand (``Context(entity_mapping=…)``)
        rather than by ``ContextBuilder`` has its sources only in the
        mapping. With the relation engine on there is no in-memory engine
        to fall back to, so the same adapter ``ContextBuilder.build`` uses
        registers them here, on first use.
        """
        registry = getattr(self.context.backend, "tables", None)
        if registry is None:
            return None
        entry = registry.get(label, kind)
        if entry is None:
            from pycypher.backends.table_registry import (
                register_context_tables,
            )

            register_context_tables(self.context)
            entry = registry.get(label, kind)
        return entry

    def _load_entity(self, label: str) -> EntityRef | None:
        from pycypher.backends.table_registry import (
            ENTITY_KIND,
            physical_table_name,
        )
        from pycypher.relation_engine import (
            _base_relation,
            _entity_attr_map,
            _node_id_column,
            _streaming_id_is_integer,
        )

        attr = _entity_attr_map(self.context, label)
        if attr is None:
            return None
        entry = self._registry_entry(label, ENTITY_KIND)
        id_col = _node_id_column(self.context, label)
        if entry is not None:
            sql_ref = f'"{physical_table_name(label)}"'
            if id_col not in entry.columns:
                # No declared id column: a physical table's rowid is a
                # stable identity for the duration of one query.
                id_col = "rowid"
        else:
            # In-memory source_obj fallback: expose it under a view so the
            # emitted SQL can name it. Same materialisation the old
            # relation builder did.
            rel = _base_relation(self.context, label, self.connection)
            if id_col not in rel.columns:
                return None
            view = f"__pycypher_plan_entity_{label}__"
            rel.create_view(view, replace=True)
            sql_ref = f'"{view}"'
        # Any registry table is a real DuckDB table a mutation can target;
        # entities reachable only through the in-memory mapping are not.
        writable = entry is not None
        return EntityRef(
            label=label,
            sql_ref=sql_ref,
            id_col=id_col,
            attr=attr,
            writable=writable,
            id_is_integer=_streaming_id_is_integer(self.context, label),
        )

    def _load_rel(self, rel_type: str) -> RelRef | None:
        from pycypher.backends.table_registry import (
            RELATIONSHIP_KIND,
            physical_table_name,
        )
        from pycypher.relation_engine import (
            SOURCE_LABEL_COLUMN,
            _rel_attr_map,
            _rel_base_relation,
        )

        attr = _rel_attr_map(self.context, rel_type)
        if attr is None:
            return None
        entry = self._registry_entry(rel_type, RELATIONSHIP_KIND)
        if entry is not None:
            sql_ref = f'"{physical_table_name(rel_type, RELATIONSHIP_KIND)}"'
            columns = list(entry.columns)
        else:
            rel = _rel_base_relation(self.context, rel_type, self.connection)
            view = f"__pycypher_plan_rel_{rel_type}__"
            rel.create_view(view, replace=True)
            sql_ref = f'"{view}"'
            columns = list(rel.columns)
        return RelRef(
            rel_type=rel_type,
            sql_ref=sql_ref,
            attr=attr,
            has_label_cols=SOURCE_LABEL_COLUMN in columns,
        )
