"""Lazy joins — the last materialisation boundary in the eager path.

Open item 3 from ``docs/duckdb_eager_path_report.md``.  ``BindingFrame``'s
inner, left, and cross joins now compose into DuckDB relations instead of
executing, so a multi-pattern query can reach ``RETURN`` without ever
building a pandas frame over the joined rows.

This changes *when* the join runs, not what it produces: ``DuckDBBackend
.join`` was already SQL on the eager path, so column selection and row order
are the same either way.

See docs/duckdb_eager_path_design.md, Phase 6 ("Known boundary").
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import assert_same_across_backends, build_context
from pycypher.backends.duckdb_backend import count_materialisations
from pycypher.scan_operators import EntityScan, RelationshipScan

PEOPLE = pd.DataFrame(
    {
        "name": ["a", "b", "c", "d"],
        "age": [10, 20, 30, 40],
        "city": ["NY", "LA", "NY", "SF"],
    },
)
KNOWS = pd.DataFrame(
    {"__SOURCE__": [0, 1, 2, 0], "__TARGET__": [1, 2, 3, 2]},
)
ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}


@pytest.fixture
def duck():
    return build_context(ENTITIES, RELATIONSHIPS, backend="duckdb")


@pytest.fixture
def pandas_ctx():
    return build_context(ENTITIES, RELATIONSHIPS, backend="pandas")


def _people(context, var="p"):
    return EntityScan("Person", var).scan(context)


def _knows(context, var="r"):
    return RelationshipScan("KNOWS", var).scan(context)


class TestJoinsStayLazy:
    def test_inner_join(self, duck):
        with count_materialisations() as log:
            joined = _people(duck).join(_knows(duck), "p", "_src_r")
            assert len(joined) == 4
        assert joined.is_lazy is True
        assert log.count == 0

    def test_left_join(self, duck):
        with count_materialisations() as log:
            joined = _people(duck).left_join(_knows(duck), "p", "_src_r")
            assert len(joined) == 5
        assert joined.is_lazy is True
        assert log.count == 0

    def test_cross_join(self, duck):
        with count_materialisations() as log:
            joined = _people(duck).cross_join(_people(duck, "q"))
            assert len(joined) == 16
        assert joined.is_lazy is True
        assert log.count == 0

    def test_chained_joins_stay_lazy(self, duck):
        with count_materialisations() as log:
            joined = (
                _people(duck)
                .join(_knows(duck), "p", "_src_r")
                .join(_knows(duck, "r2"), "_tgt_r", "_src_r2")
            )
            assert len(joined) >= 1
        assert joined.is_lazy is True
        assert log.count == 0


class TestJoinShape:
    """Column selection must match the eager SQL exactly."""

    def test_redundant_right_key_is_dropped(self, duck):
        joined = _people(duck).join(_knows(duck), "p", "_src_r")
        assert joined.var_names == ["p", "r", "_tgt_r"]
        assert "_src_r" not in joined.var_names

    def test_columns_match_the_pandas_backend(self, duck, pandas_ctx):
        lazy = _people(duck).join(_knows(duck), "p", "_src_r")
        eager = _people(pandas_ctx).join(_knows(pandas_ctx), "p", "_src_r")
        assert lazy.var_names == eager.var_names

    def test_cross_join_columns_match(self, duck, pandas_ctx):
        lazy = _people(duck).cross_join(_people(duck, "q"))
        eager = _people(pandas_ctx).cross_join(_people(pandas_ctx, "q"))
        assert lazy.var_names == eager.var_names

    def test_type_registry_merges_both_sides(self, duck):
        joined = _people(duck).join(_knows(duck), "p", "_src_r")
        assert joined.type_registry == {"p": "Person", "r": "KNOWS"}

    def test_rows_match_the_pandas_backend(self, duck, pandas_ctx):
        lazy = _people(duck).join(_knows(duck), "p", "_src_r")
        eager = _people(pandas_ctx).join(_knows(pandas_ctx), "p", "_src_r")
        assert sorted(lazy.bindings["p"]) == sorted(eager.bindings["p"])

    def test_missing_join_column_still_raises(self, duck):
        from pycypher.exceptions import VariableNotFoundError

        with pytest.raises(VariableNotFoundError):
            _people(duck).join(_knows(duck), "nope", "_src_r")


class TestSafetyLimitsStillApply:
    def test_cross_join_ceiling_is_enforced_before_execution(
        self, duck, monkeypatch
    ):
        # The ceiling exists to stop a catastrophic materialisation, so it
        # must be checked from the free row counts, not after the join runs.
        from pycypher import binding_frame
        from pycypher.exceptions import QueryMemoryBudgetError

        monkeypatch.setattr(binding_frame, "MAX_CROSS_JOIN_ROWS", 4)
        with pytest.raises(QueryMemoryBudgetError):
            _people(duck).cross_join(_people(duck, "q"))


class TestDifferentialQueries:
    @pytest.mark.parametrize(
        "cypher",
        [
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person)-[:KNOWS]->(r:Person) "
                "RETURN p.name, r.name ORDER BY p.name, r.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "WHERE p.age > 15 AND q.age < 40 "
                "RETURN p.name, q.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person), (q:Person) WHERE p.age < q.age "
                "RETURN count(*)"
            ),
            (
                "MATCH (p:Person)-[:KNOWS*1..2]->(q:Person) "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
        ],
    )
    def test_query_matches_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher, RELATIONSHIPS)

    @pytest.mark.parametrize(
        "cypher",
        [
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "RETURN p.city, count(q) ORDER BY p.city"
            ),
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city",
        ],
    )
    def test_grouped_aggregation_matches(self, cypher):
        # These were compared as multisets while `ORDER BY` was being
        # dropped for grouped-aggregation output, which left group order
        # falling out of row arrival order — something lazy joins
        # legitimately change. That bug is fixed, so the ordering is now
        # real and worth asserting.
        assert_same_across_backends(ENTITIES, cypher, RELATIONSHIPS)
