"""Phase 1a (DuckDB eager path) — the DuckDB table registry.

Covers :mod:`pycypher.backends.table_registry`: materialising entity and
relationship sources into real DuckDB tables, the metadata recorded about
each, and the wiring that makes ``ContextBuilder.build()`` and
``register_streaming_source`` share one registry.

See docs/duckdb_eager_path_design.md, Phase 1a.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.backends.duckdb_backend import DuckDBBackend
from pycypher.backends.table_registry import (
    ENTITY_KIND,
    RELATIONSHIP_KIND,
    TableRegistry,
    physical_table_name,
    register_context_tables,
)
from pycypher.ingestion.context_builder import ContextBuilder
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)

ID_COLUMN = "__ID__"


@pytest.fixture
def people() -> pd.DataFrame:
    return pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})


@pytest.fixture
def knows() -> pd.DataFrame:
    return pd.DataFrame({"__SOURCE__": [0], "__TARGET__": [1]})


@pytest.fixture
def duckdb_context(people, knows) -> Context:
    return (
        ContextBuilder()
        .add_entity("Person", people)
        .add_relationship(
            "KNOWS", knows, source_col="__SOURCE__", target_col="__TARGET__"
        )
        .build(backend="duckdb")
    )


class TestPhysicalTableName:
    def test_entity_prefix_is_unchanged_from_before_the_registry(self):
        # The literal register_streaming_source used to build inline, kept so
        # a pre-existing scratch database still reads the same.
        assert physical_table_name("Person") == "_streaming_source_Person"

    def test_relationship_uses_a_distinct_prefix(self):
        assert (
            physical_table_name("KNOWS", RELATIONSHIP_KIND)
            == "_rel_source_KNOWS"
        )

    def test_rejects_unsafe_label(self):
        with pytest.raises(ValueError, match="Invalid SQL identifier"):
            physical_table_name('Person"; DROP TABLE x; --')

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown table kind"):
            physical_table_name("Person", "nonsense")


class TestContextBuilderRegistration:
    def test_registers_entities_and_relationships(self, duckdb_context):
        registry = duckdb_context.backend.tables
        assert registry.labels(ENTITY_KIND) == ["Person"]
        assert registry.labels(RELATIONSHIP_KIND) == ["KNOWS"]
        assert len(registry) == 2

    def test_records_id_column_and_its_declared_type(self, duckdb_context):
        entry = duckdb_context.backend.tables.get("Person")
        assert entry.id_col == ID_COLUMN
        # Recorded from the materialised schema, not guessed — this is what
        # lets downstream SQL cast consistently instead of rediscovering the
        # type after a pandas round-trip has mangled it.
        assert entry.id_type == "BIGINT"

    def test_attr_map_comes_from_the_entity_table_and_excludes_the_id(
        self, duckdb_context
    ):
        entry = duckdb_context.backend.tables.get("Person")
        assert entry.attr_map == {"name": "name", "age": "age"}
        assert ID_COLUMN in entry.columns

    def test_table_holds_the_source_rows(self, duckdb_context):
        entry = duckdb_context.backend.tables.get("Person")
        assert sorted(entry.relation.fetchdf()["name"]) == ["Alice", "Bob"]

    def test_relationship_table_keeps_endpoint_columns(self, duckdb_context):
        entry = duckdb_context.backend.tables.get("KNOWS", RELATIONSHIP_KIND)
        assert set(entry.columns) >= {ID_COLUMN, "__SOURCE__", "__TARGET__"}

    def test_pandas_backend_gets_no_registry(self, people):
        context = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="pandas")
        )
        assert getattr(context.backend, "tables", None) is None

    def test_register_tables_false_skips_materialisation(self, people):
        context = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="duckdb", register_tables=False)
        )
        assert len(context.backend.tables) == 0

    def test_register_context_tables_is_a_noop_without_a_registry(
        self, people
    ):
        context = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="pandas")
        )
        assert register_context_tables(context) == 0


class TestKindNamespacing:
    def test_same_label_as_entity_and_relationship_does_not_collide(self):
        # Cypher lets a node label and a relationship type share a spelling,
        # so the registry keys on (kind, label), not label alone.
        backend = DuckDBBackend()
        try:
            registry = backend.tables
            registry.register_source_object(
                "Link", pd.DataFrame({"a": [1]}), kind=ENTITY_KIND
            )
            registry.register_source_object(
                "Link", pd.DataFrame({"b": [2, 3]}), kind=RELATIONSHIP_KIND
            )
            assert len(registry) == 2
            assert registry.get("Link", ENTITY_KIND).columns == ["a"]
            assert registry.get("Link", RELATIONSHIP_KIND).columns == ["b"]
        finally:
            backend.close()


class TestRegistryOperations:
    @pytest.fixture
    def registry(self) -> TableRegistry:
        backend = DuckDBBackend()
        try:
            yield backend.tables
        finally:
            backend.close()

    def test_registration_is_idempotent_and_replaces(self, registry):
        registry.register_source_object("P", pd.DataFrame({"a": [1]}))
        registry.register_source_object("P", pd.DataFrame({"a": [1, 2, 3]}))
        assert len(registry) == 1
        assert len(registry.get("P").relation.fetchdf()) == 3

    def test_derived_attr_map_is_identity_over_non_id_columns(self, registry):
        entry = registry.register_source_object(
            "P", pd.DataFrame({"pid": [1], "x": [2]}), id_col="pid"
        )
        assert entry.attr_map == {"x": "x"}

    def test_missing_label_lookups_return_none(self, registry):
        assert registry.get("Nope") is None
        assert registry.relation("Nope") is None
        assert registry.has("Nope") is False

    def test_id_type_is_none_when_id_col_absent_from_schema(self, registry):
        entry = registry.register_source_object(
            "P", pd.DataFrame({"a": [1]}), id_col="not_a_column"
        )
        assert entry.id_type is None

    def test_drop_removes_registration_and_table(self, registry):
        registry.register_source_object("P", pd.DataFrame({"a": [1]}))
        assert registry.drop("P") is True
        assert registry.has("P") is False
        assert registry.drop("P") is False

    def test_clear_drops_everything(self, registry):
        registry.register_source_object("P", pd.DataFrame({"a": [1]}))
        registry.register_source_object(
            "R", pd.DataFrame({"a": [1]}), kind=RELATIONSHIP_KIND
        )
        registry.clear()
        assert len(registry) == 0

    def test_relation_reflects_mutations_but_a_materialised_frame_does_not(
        self, registry
    ):
        # Documents a real sharp edge: DuckDBLazyFrame caches its first
        # to_pandas() forever, so post-mutation reads must go through
        # .relation.  See RegisteredTable's docstring warning.
        entry = registry.register_source_object("P", pd.DataFrame({"a": [1]}))
        assert entry.frame.to_pandas()["a"].tolist() == [1]
        registry._con.execute(
            f'UPDATE "{entry.table_name}" SET a = 99'  # noqa: S608 — table name is registry-generated, not user input
        )
        assert entry.relation.fetchdf()["a"].tolist() == [99]
        assert entry.frame.to_pandas()["a"].tolist() == [1]


class TestStreamingSourceSharesTheRegistry:
    @pytest.fixture
    def people_parquet(self, tmp_path):
        path = tmp_path / "people.parquet"
        pd.DataFrame({"name": ["Alice"], "age": [30]}).to_parquet(path)
        return path

    def _ctx(self) -> Context:
        context = Context(
            entity_mapping=EntityMapping(mapping={}),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="duckdb",
        )
        context._relation_engine_enabled = True
        return context

    def test_streaming_registration_lands_in_the_registry(
        self, people_parquet
    ):
        context = self._ctx()
        register_streaming_source(
            context, "Person", data_source_from_uri(str(people_parquet))
        )
        assert context.backend.tables.has("Person")

    def test_streaming_sources_alias_still_has_its_tuple_shape(
        self, people_parquet
    ):
        # The relation engine's mutation slices read this tuple directly;
        # the registry is layered underneath without changing its shape.
        context = self._ctx()
        register_streaming_source(
            context,
            "Person",
            data_source_from_uri(str(people_parquet)),
            id_col="name",
        )
        frame, attr_map, id_col = context._streaming_sources["Person"]
        assert attr_map == {"age": "age"}
        assert id_col == "name"
        assert "name" in frame.columns

    def test_register_context_tables_does_not_clobber_a_streaming_source(
        self, people_parquet, people
    ):
        # A file-backed streaming registration is strictly better than the
        # in-memory one, so it must win.
        context = self._ctx()
        register_streaming_source(
            context, "Person", data_source_from_uri(str(people_parquet))
        )
        before = context.backend.tables.get("Person")
        context.entity_mapping.mapping["Person"] = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="pandas")
            .entity_mapping.mapping["Person"]
        )
        assert register_context_tables(context) == 0
        assert context.backend.tables.get("Person") is before


class TestInstrumentedBackendForwarding:
    def test_instrumented_backend_exposes_the_registry(self, people):
        # nmetl run -v wraps the backend; without forwarding, every registry
        # consumer would silently disable itself under verbose mode.
        context = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="duckdb", instrument=True)
        )
        assert context.backend.tables.has("Person")


class TestBaseRelationPrefersTheRegistry:
    def test_base_relation_returns_the_registered_table(self, duckdb_context):
        from pycypher.relation_engine import _base_relation

        relation = _base_relation(
            duckdb_context, "Person", duckdb_context.backend.connection
        )
        assert sorted(relation.fetchdf()["name"]) == ["Alice", "Bob"]

    def test_rel_base_relation_returns_the_registered_table(
        self, duckdb_context
    ):
        from pycypher.relation_engine import _rel_base_relation

        relation = _rel_base_relation(
            duckdb_context, "KNOWS", duckdb_context.backend.connection
        )
        assert set(relation.columns) >= {"__SOURCE__", "__TARGET__"}

    def test_falls_back_when_nothing_is_registered(self, people):
        from pycypher.relation_engine import _base_relation

        context = (
            ContextBuilder()
            .add_entity("Person", people)
            .build(backend="duckdb", register_tables=False)
        )
        relation = _base_relation(
            context, "Person", context.backend.connection
        )
        assert sorted(relation.fetchdf()["name"]) == ["Alice", "Bob"]


class TestMultiSourceMerge:
    """Phase 3c (the FastOpenData streaming-qualification plan (private repository)) -- a
    label registered more than once (as happens for the real fastopendata
    config's Tract/LOCATED_IN/etc.) merges instead of only the last
    registration surviving.
    """

    def _ctx(self) -> Context:
        context = Context(
            entity_mapping=EntityMapping(mapping={}),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="duckdb",
        )
        context._relation_engine_enabled = True
        return context

    def test_entity_merge_unions_columns_and_ids(self, tmp_path):
        context = self._ctx()
        df1 = pd.DataFrame({"tract": ["t1", "t2"], "statefp": ["06", "06"]})
        df2 = pd.DataFrame({"tract": ["t1", "t3"], "longitude": [1.1, 2.2]})
        register_streaming_source(
            context, "Tract", data_source_from_uri(df1), id_col="tract"
        )
        entry = register_streaming_source(
            context, "Tract", data_source_from_uri(df2), id_col="tract"
        )
        got = context.backend.tables.get("Tract").relation.fetchdf()
        rows = {
            tract: (row.statefp, row.longitude)
            for tract, row in got.set_index("tract").iterrows()
        }
        assert rows["t1"] == ("06", 1.1)
        assert rows["t2"][0] == "06"
        assert pd.isna(rows["t2"][1])
        assert pd.isna(rows["t3"][0])
        assert rows["t3"][1] == 2.2
        entry = context.backend.tables.get("Tract")
        assert entry.attr_map == {
            "statefp": "statefp",
            "longitude": "longitude",
        }
        assert entry.id_col == "tract"

    def test_entity_merge_column_collision_newer_source_wins(self):
        context = self._ctx()
        register_streaming_source(
            context,
            "Tract",
            data_source_from_uri(pd.DataFrame({"id": ["t1"], "x": ["old"]})),
            id_col="id",
        )
        register_streaming_source(
            context,
            "Tract",
            data_source_from_uri(pd.DataFrame({"id": ["t1"], "x": ["new"]})),
            id_col="id",
        )
        got = context.backend.tables.get("Tract").relation.fetchdf()
        assert got["x"].tolist() == ["new"]

    def test_entity_merge_id_col_name_stays_the_first_sources(self):
        context = self._ctx()
        register_streaming_source(
            context,
            "Tract",
            data_source_from_uri(pd.DataFrame({"TRACT_FIPS": ["t1"]})),
            id_col="TRACT_FIPS",
        )
        register_streaming_source(
            context,
            "Tract",
            data_source_from_uri(pd.DataFrame({"GEOID": ["t1"], "y": [1]})),
            id_col="GEOID",
        )
        entry = context.backend.tables.get("Tract")
        assert entry.id_col == "TRACT_FIPS"
        assert "TRACT_FIPS" in entry.columns
        assert "GEOID" not in entry.columns

    def test_entity_merge_falls_back_to_replace_without_an_id_col(self):
        context = self._ctx()
        register_streaming_source(
            context,
            "P",
            data_source_from_uri(pd.DataFrame({"a": [1]})),
        )
        register_streaming_source(
            context,
            "P",
            data_source_from_uri(pd.DataFrame({"a": [1, 2, 3]})),
        )
        got = context.backend.tables.get("P").relation.fetchdf()
        assert len(got) == 3

    def test_relationship_merge_unions_rows_and_renumbers_ids(self):
        context = self._ctx()
        register_streaming_relationship(
            context,
            "LOCATED_IN",
            data_source_from_uri(
                pd.DataFrame({"tract": ["t1", "t2"], "county": ["c1", "c1"]})
            ),
            source_col="tract",
            target_col="county",
        )
        register_streaming_relationship(
            context,
            "LOCATED_IN",
            data_source_from_uri(
                pd.DataFrame({"tract": ["t1", "t2"], "puma": ["p1", "p2"]})
            ),
            source_col="tract",
            target_col="puma",
        )
        got = context.backend.tables.get(
            "LOCATED_IN", RELATIONSHIP_KIND
        ).relation.fetchdf()
        edges = set(zip(got["__SOURCE__"], got["__TARGET__"]))
        assert edges == {
            ("t1", "c1"),
            ("t2", "c1"),
            ("t1", "p1"),
            ("t2", "p2"),
        }
        assert sorted(got["__ID__"]) == [0, 1, 2, 3]
        assert got["__ID__"].is_unique

    def test_relationship_merge_keeps_per_source_endpoint_labels(self):
        # A declared and an undeclared source merged under one type: each
        # edge keeps the labels *its* source declared (NULL for none).
        context = self._ctx()
        register_streaming_relationship(
            context,
            "LOCATED_IN",
            data_source_from_uri(
                pd.DataFrame({"unit": ["u1"], "puma": ["p1"]})
            ),
            source_col="unit",
            target_col="puma",
            source_entity_type="Unit1yr",
            target_entity_type="PUMA",
        )
        register_streaming_relationship(
            context,
            "LOCATED_IN",
            data_source_from_uri(
                pd.DataFrame({"tract": ["t1"], "county": ["c1"]})
            ),
            source_col="tract",
            target_col="county",
        )
        got = context.backend.tables.get(
            "LOCATED_IN", RELATIONSHIP_KIND
        ).relation.fetchdf()
        rows = {
            r["__SOURCE__"]: (r["__SOURCE_LABEL__"], r["__TARGET_LABEL__"])
            for _, r in got.iterrows()
        }
        assert rows["u1"] == ("Unit1yr", "PUMA")
        assert pd.isna(rows["t1"][0])
        assert pd.isna(rows["t1"][1])
        entry = context.backend.tables.get("LOCATED_IN", RELATIONSHIP_KIND)
        assert "__SOURCE_LABEL__" not in entry.attr_map

    def test_context_builder_merges_entities_across_sources(self):
        ctx = (
            ContextBuilder()
            .add_entity(
                "Tract",
                pd.DataFrame({"tract": ["t1", "t2"], "statefp": ["06", "06"]}),
                id_col="tract",
            )
            .add_entity(
                "Tract",
                pd.DataFrame({"tract": ["t1", "t3"], "longitude": [1.1, 2.2]}),
                id_col="tract",
            )
            .build(backend="duckdb")
        )
        tract = ctx.entity_mapping.mapping["Tract"]
        assert set(tract.attribute_map) == {"statefp", "longitude"}
        assert tract.source_obj.num_rows == 3

    def test_context_builder_merges_relationships_across_sources(self):
        ctx = (
            ContextBuilder()
            .add_relationship(
                "LOCATED_IN",
                pd.DataFrame({"tract": ["t1", "t2"], "county": ["c1", "c1"]}),
                source_col="tract",
                target_col="county",
            )
            .add_relationship(
                "LOCATED_IN",
                pd.DataFrame({"tract": ["t1", "t2"], "puma": ["p1", "p2"]}),
                source_col="tract",
                target_col="puma",
            )
            .build(backend="duckdb")
        )
        rel = ctx.relationship_mapping.mapping["LOCATED_IN"]
        assert rel.source_obj.num_rows == 4


class TestEntityDedup:
    """An entity source at a finer grain than the entity (one row per tract
    defining State, as in the real crosswalk) keeps only the first row per
    id, matching the eager path -- and does so *before* the Phase 3c merge,
    which would otherwise fan out to the product of both sides' duplicates.
    Found on the first real Phase 4 run (2026-09-05): `MATCH (s:State)`
    returned one row per tract.
    """

    def _ctx(self) -> Context:
        context = Context(
            entity_mapping=EntityMapping(mapping={}),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="duckdb",
        )
        context._relation_engine_enabled = True
        return context

    def test_fresh_registration_keeps_first_row_per_id_and_warns(self):
        from unittest.mock import patch

        from pycypher.backends import table_registry

        context = self._ctx()
        df = pd.DataFrame(
            {
                "state": ["06", "06", "13", "06"],
                "tract": ["t1", "t2", "t3", "t4"],
            }
        )
        with patch.object(table_registry.LOGGER, "warning") as warn:
            register_streaming_source(
                context, "State", data_source_from_uri(df), id_col="state"
            )
        got = context.backend.tables.get("State").relation.fetchdf()
        assert list(got["state"]) == ["06", "13"]
        assert list(got["tract"]) == ["t1", "t3"]
        assert warn.call_count == 1
        assert warn.call_args.args[1:] == ("State", 2, "state", 4, 2)

    def test_unique_ids_are_left_alone_without_a_warning(self):
        from unittest.mock import patch

        from pycypher.backends import table_registry

        context = self._ctx()
        df = pd.DataFrame({"state": ["06", "13"], "name": ["CA", "GA"]})
        with patch.object(table_registry.LOGGER, "warning") as warn:
            register_streaming_source(
                context, "State", data_source_from_uri(df), id_col="state"
            )
        got = context.backend.tables.get("State").relation.fetchdf()
        assert list(got["state"]) == ["06", "13"]
        assert warn.call_count == 0

    def test_merge_does_not_fan_out_duplicate_ids(self):
        context = self._ctx()
        df1 = pd.DataFrame({"tract": ["t1", "t1", "t2"], "a": [1, 2, 3]})
        df2 = pd.DataFrame({"tract": ["t1", "t1", "t3"], "b": [10, 20, 30]})
        register_streaming_source(
            context, "Tract", data_source_from_uri(df1), id_col="tract"
        )
        register_streaming_source(
            context, "Tract", data_source_from_uri(df2), id_col="tract"
        )
        got = (
            context.backend.tables.get("Tract")
            .relation.fetchdf()
            .sort_values("tract")
            .reset_index(drop=True)
        )
        assert list(got["tract"]) == ["t1", "t2", "t3"]
        rows = {r.tract: (r.a, r.b) for r in got.itertuples()}
        assert rows["t1"] == (1, 10)  # first occurrence on both sides
        assert rows["t2"][0] == 3
        assert pd.isna(rows["t2"][1])
        assert pd.isna(rows["t3"][0])
        assert rows["t3"][1] == 30
