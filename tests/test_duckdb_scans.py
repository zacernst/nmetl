"""Phase 3 (DuckDB eager path) — entity and relationship scans in SQL.

``EntityScan`` and ``RelationshipScan`` return relation-backed
``BindingFrame``s on the DuckDB backend instead of materialising the source
table into ``context._property_lookup_cache``.  Inline-property pushdown
becomes a ``WHERE``; endpoint pushdown becomes a semi-join.

See docs/duckdb_eager_path_design.md, Phase 3.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import assert_same_across_backends, build_context
from pycypher.backends.duckdb_backend import count_materialisations
from pycypher.scan_operators import EntityScan, RelationshipScan

PEOPLE = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Carol", "Dan"],
        "age": [30, 25, 35, 40],
        "city": ["NY", "LA", "NY", "SF"],
        "active": [True, False, True, True],
    },
)
KNOWS = pd.DataFrame(
    {"__SOURCE__": [0, 1, 0, 2], "__TARGET__": [1, 2, 2, 3]},
)

ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}


@pytest.fixture
def duck():
    return build_context(ENTITIES, RELATIONSHIPS, backend="duckdb")


@pytest.fixture
def pandas_ctx():
    return build_context(ENTITIES, RELATIONSHIPS, backend="pandas")


class TestEntityScan:
    def test_scan_is_lazy_on_duckdb(self, duck):
        frame = EntityScan("Person", "p").scan(duck)
        assert frame.is_lazy is True
        assert frame.var_names == ["p"]

    def test_scan_stays_lazy_through_schema_and_count(self, duck):
        with count_materialisations() as log:
            frame = EntityScan("Person", "p").scan(duck)
            assert frame.var_names == ["p"]
            assert len(frame) == 4
        assert log.count == 0

    def test_scan_is_not_lazy_on_pandas(self, pandas_ctx):
        assert EntityScan("Person", "p").scan(pandas_ctx).is_lazy is False

    def test_ids_match_the_pandas_path(self, duck, pandas_ctx):
        duck_ids = EntityScan("Person", "p").scan(duck).bindings["p"].tolist()
        pandas_ids = (
            EntityScan("Person", "p").scan(pandas_ctx).bindings["p"].tolist()
        )
        assert duck_ids == pandas_ids

    def test_unknown_entity_type_still_raises(self, duck):
        from pycypher.exceptions import GraphTypeNotFoundError

        with pytest.raises(GraphTypeNotFoundError):
            EntityScan("Ghost", "g").scan(duck)


class TestEntityScanPushdown:
    def test_string_equality_pushes_into_sql(self, duck):
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"city": "NY"}
        )
        assert frame.is_lazy is True
        assert len(frame) == 2

    def test_integer_equality_pushes_into_sql(self, duck):
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"age": 30}
        )
        assert frame.is_lazy is True
        assert len(frame) == 1

    def test_boolean_equality_pushes_into_sql(self, duck):
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"active": True}
        )
        assert frame.is_lazy is True
        assert len(frame) == 3

    def test_multiple_predicates_are_conjunctive(self, duck):
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"city": "NY", "age": 35}
        )
        assert len(frame) == 1

    def test_pushdown_agrees_with_the_pandas_path(self, duck, pandas_ctx):
        for filters in (
            {"city": "NY"},
            {"age": 25},
            {"city": "SF", "age": 40},
            {"city": "nowhere"},
        ):
            duck_rows = sorted(
                EntityScan("Person", "p")
                .scan(duck, property_filters=filters)
                .bindings["p"]
                .tolist()
            )
            pandas_rows = sorted(
                EntityScan("Person", "p")
                .scan(pandas_ctx, property_filters=filters)
                .bindings["p"]
                .tolist()
            )
            assert duck_rows == pandas_rows, filters

    def test_unknown_property_falls_back_to_pandas(self, duck):
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"nope": 1}
        )
        assert frame.is_lazy is False

    def test_null_filter_falls_back_rather_than_emitting_is_null(self, duck):
        # The pandas pushdown goes through PropertyValueIndex, which skips
        # nulls at build time, so `prop = NULL` would be a silent semantic
        # change.  Falling back keeps the two paths identical.
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"city": None}
        )
        assert frame.is_lazy is False

    def test_string_literals_are_escaped(self, duck):
        # A quote in the value must not terminate the literal.
        frame = EntityScan("Person", "p").scan(
            duck, property_filters={"city": "N'; DROP TABLE x; --"}
        )
        assert len(frame) == 0


class TestRelationshipScan:
    def test_scan_is_lazy_and_free_to_inspect(self, duck):
        scan = RelationshipScan("KNOWS", "r")
        with count_materialisations() as log:
            frame = scan.scan(duck)
            assert frame.var_names == ["r", "_src_r", "_tgt_r"]
            assert len(frame) == 4
        assert log.count == 0
        assert frame.is_lazy is True

    def test_source_pushdown_semi_joins(self, duck):
        frame = RelationshipScan("KNOWS", "r").scan(
            duck, source_ids=pd.Series([0])
        )
        assert frame.is_lazy is True
        assert sorted(frame.bindings["_tgt_r"].tolist()) == [1, 2]

    def test_target_pushdown_semi_joins(self, duck):
        frame = RelationshipScan("KNOWS", "r").scan(
            duck, target_ids=pd.Series([2])
        )
        assert sorted(frame.bindings["_src_r"].tolist()) == [0, 1]

    def test_both_endpoints_push_down_together(self, duck):
        frame = RelationshipScan("KNOWS", "r").scan(
            duck, source_ids=pd.Series([0]), target_ids=pd.Series([2])
        )
        assert len(frame) == 1

    def test_object_dtype_pushdown_ids_are_coerced(self, duck):
        # A pandas-scanned frame yields object-dtype IDs; the semi-join must
        # still match against an integral endpoint column.
        frame = RelationshipScan("KNOWS", "r").scan(
            duck, source_ids=pd.Series([0, 2], dtype=object)
        )
        assert len(frame) == 3

    def test_uncoercible_ids_fall_back_instead_of_matching_nothing(self, duck):
        # The dangerous failure is a semi-join that silently returns zero
        # rows because one side is text.  Falling back keeps it correct.
        frame = RelationshipScan("KNOWS", "r").scan(
            duck, source_ids=pd.Series(["not-an-id"])
        )
        assert frame.is_lazy is False
        assert len(frame) == 0

    def test_pushdown_agrees_with_the_pandas_path(self, duck, pandas_ctx):
        for ids in ([0], [1, 2], [], [99]):
            duck_rows = len(
                RelationshipScan("KNOWS", "r").scan(
                    duck, source_ids=pd.Series(ids, dtype="int64")
                )
            )
            pandas_rows = len(
                RelationshipScan("KNOWS", "r").scan(
                    pandas_ctx, source_ids=pd.Series(ids, dtype="int64")
                )
            )
            assert duck_rows == pandas_rows, ids


class TestShadowGuards:
    """A lazy scan must never read past an uncommitted mutation overlay."""

    def test_entity_shadow_forces_the_pandas_path(self, duck):
        duck._shadow["Person"] = PEOPLE.assign(__ID__=range(len(PEOPLE)))
        assert EntityScan("Person", "p").scan(duck).is_lazy is False

    def test_relationship_shadow_forces_the_pandas_path(self, duck):
        duck._shadow_rels["KNOWS"] = KNOWS.assign(__ID__=range(len(KNOWS)))
        assert RelationshipScan("KNOWS", "r").scan(duck).is_lazy is False


class TestDifferentialQueries:
    """End-to-end: the two backends must return the same rows."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person) RETURN p.name ORDER BY p.name",
            "MATCH (p:Person) WHERE p.age > 28 RETURN p.name ORDER BY p.name",
            "MATCH (p:Person {city: 'NY'}) RETURN p.name ORDER BY p.name",
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE q.age > 30 "
                "RETURN p.name, q.age ORDER BY p.name"
            ),
            (
                "MATCH (p:Person {name: 'Alice'})-[:KNOWS]->(q) "
                "RETURN q.name ORDER BY q.name"
            ),
            "MATCH (p:Person) RETURN count(p)",
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city",
            "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC LIMIT 2",
        ],
    )
    def test_query_matches_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher, RELATIONSHIPS)
