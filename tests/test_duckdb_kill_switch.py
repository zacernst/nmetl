"""Phase 8 (DuckDB eager path) — the fallback is a genuine kill switch.

Every DuckDB route added by Phases 3-7 declines when the table registry is
absent, and the pandas path then runs exactly as it did before. That is the
property that makes the whole feature safe to ship: if any of it misbehaves,
``register_tables=False`` reverts execution without changing results.

The bounded-RSS acceptance test that Phase 8 originally called for is *not*
here, and deliberately so — see
``tests/benchmarks/bench_duckdb_eager_path_memory.py`` for the measurements
showing why that goal is not reachable through the eager path, and
``docs/duckdb_eager_path_design.md`` Phase 8 for the write-up.

See docs/duckdb_eager_path_design.md, Phase 8.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import build_context, run_query
from pycypher.ast_models import RelationshipDirection as Direction
from pycypher.path_expander import PathExpander
from pycypher.scan_operators import EntityScan, RelationshipScan

PEOPLE = pd.DataFrame(
    {
        "name": ["a", "b", "c", "d"],
        "age": [10, 20, 30, 40],
        "city": ["NY", "LA", "NY", "SF"],
    },
)
KNOWS = pd.DataFrame({"__SOURCE__": [0, 1, 2], "__TARGET__": [1, 2, 3]})
ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}

QUERIES = [
    "MATCH (p:Person) RETURN p.name ORDER BY p.name",
    "MATCH (p:Person) WHERE p.age > 15 RETURN p.name ORDER BY p.name",
    "MATCH (p:Person {city: 'NY'}) RETURN p.name ORDER BY p.name",
    (
        "MATCH (p:Person)-[:KNOWS]->(q:Person) "
        "RETURN p.name, q.name ORDER BY p.name"
    ),
    (
        "MATCH (p:Person)-[:KNOWS*1..2]->(q:Person) "
        "RETURN p.name, q.name ORDER BY p.name, q.name"
    ),
]


def _disabled():
    """A DuckDB context with the table registry never populated."""
    from pycypher.ingestion.context_builder import ContextBuilder

    builder = ContextBuilder().add_entity("Person", PEOPLE)
    builder = builder.add_relationship(
        "KNOWS", KNOWS, source_col="__SOURCE__", target_col="__TARGET__"
    )
    return builder.build(backend="duckdb", register_tables=False)


class TestEveryRouteDeclines:
    def test_entity_scan_declines(self):
        assert EntityScan("Person", "p").scan(_disabled()).is_lazy is False

    def test_entity_pushdown_declines(self):
        frame = EntityScan("Person", "p").scan(
            _disabled(), property_filters={"city": "NY"}
        )
        assert frame.is_lazy is False
        assert len(frame) == 2

    def test_relationship_scan_declines(self):
        assert (
            RelationshipScan("KNOWS", "r").scan(_disabled()).is_lazy is False
        )

    def test_path_expansion_declines(self):
        context = _disabled()
        seed = EntityScan("Person", "p").scan(context)
        out = PathExpander(context).expand_variable_length_path(
            seed,
            "p",
            "KNOWS",
            Direction.RIGHT,
            "q",
            "Person",
            1,
            2,
            [0],
        )
        assert out.is_lazy is False


class TestResultsAreIdentical:
    @pytest.mark.parametrize("cypher", QUERIES)
    def test_disabled_registry_matches_pandas(self, cypher):
        assert run_query(_disabled(), cypher) == run_query(
            build_context(ENTITIES, RELATIONSHIPS, backend="pandas"), cypher
        )

    @pytest.mark.parametrize("cypher", QUERIES)
    def test_enabled_registry_matches_pandas(self, cypher):
        assert run_query(
            build_context(ENTITIES, RELATIONSHIPS, backend="duckdb"), cypher
        ) == run_query(
            build_context(ENTITIES, RELATIONSHIPS, backend="pandas"), cypher
        )
