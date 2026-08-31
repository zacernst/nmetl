"""Phase 6 (DuckDB eager path) — graph indexes stop being built.

The four structures in ``graph_index.py`` exist to accelerate the pandas
path.  On DuckDB each is superseded: ``PropertyValueIndex`` by a ``WHERE``,
``AdjacencyIndex`` by a semi-join, and ``VectorizedPropertyStore`` by
resolving properties in the same join that materialises the frame.

The design doc proposed making ``GraphIndexManager`` return ``None`` for all
four.  That would be a regression: the indexes still serve the *pandas
fallback*, which is taken whenever the DuckDB route declines, and there is
no DuckDB replacement in that case.  The right end state is the one measured
here — the indexes are simply never reached on the DuckDB path.

See docs/duckdb_eager_path_design.md, Phase 6.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import BACKENDS, build_context, run_query

PEOPLE = pd.DataFrame(
    {
        "name": [f"p{i}" for i in range(30)],
        "age": list(range(30)),
        "city": ["NY", "LA"] * 15,
    },
)
KNOWS = pd.DataFrame(
    {"__SOURCE__": list(range(29)), "__TARGET__": list(range(1, 30))},
)
ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}


def _index_counts(backend: str, cypher: str) -> dict[str, list[str]]:
    """Run *cypher* and report which index structures got built."""
    context = build_context(ENTITIES, RELATIONSHIPS, backend=backend)
    run_query(context, cypher)
    manager = context._index_manager
    if manager is None:
        return {"adjacency": [], "property": [], "vectorized": []}
    stats = manager.stats()
    return {
        "adjacency": list(stats["adjacency_indexes"]),
        "property": list(stats["property_indexes"]),
        "vectorized": list(stats["vectorized_stores"]),
    }


class TestIndexesAreNotBuilt:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person) WHERE p.age > 25 RETURN p.name ORDER BY p.name",
            "MATCH (p:Person {city: 'NY'}) RETURN count(p)",
            (
                "MATCH (p:Person) WHERE p.age < 5 "
                "RETURN p.name, p.age, p.city ORDER BY p.name"
            ),
        ],
    )
    def test_single_table_queries_build_nothing_on_duckdb(self, cypher):
        counts = _index_counts("duckdb", cypher)
        assert counts == {"adjacency": [], "property": [], "vectorized": []}

    def test_property_index_is_superseded_by_a_where(self):
        cypher = "MATCH (p:Person {city: 'NY'}) RETURN count(p)"
        assert _index_counts("pandas", cypher)["property"] == ["Person.city"]
        assert _index_counts("duckdb", cypher)["property"] == []

    def test_adjacency_index_is_superseded_by_a_semi_join(self):
        cypher = (
            "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 25 "
            "RETURN p.name, q.name ORDER BY p.name"
        )
        assert _index_counts("pandas", cypher)["adjacency"] == ["KNOWS"]
        assert _index_counts("duckdb", cypher)["adjacency"] == []

    def test_vectorized_store_is_superseded_for_single_table_projection(self):
        cypher = "MATCH (p:Person) RETURN p.name, p.age ORDER BY p.name"
        # pandas prebuilds a store for every registered type, touched or
        # not — that eager prebuild is what Phase 6-partial removed here.
        assert "Person" in _index_counts("pandas", cypher)["vectorized"]
        assert _index_counts("duckdb", cypher)["vectorized"] == []


class TestKnownBoundary:
    """What is left after joins were made lazy (report open item 3).

    The boundary used to be "any join materialises". It is now narrower:
    resolving a property materialises the frame (it has to — the caller
    wants values), so a *second* variable projected afterwards finds a
    materialised frame and falls back to the property store. Closing that
    needs the projection planner to batch property resolution across
    variables, which is beyond making joins lazy.
    """

    @pytest.mark.parametrize(
        "cypher",
        [
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "RETURN p.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "RETURN p.name, p.age ORDER BY p.name"
            ),
            "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN count(*)",
        ],
    )
    def test_joins_alone_no_longer_build_a_store(self, cypher):
        assert _index_counts("duckdb", cypher)["vectorized"] == []

    def test_a_second_projected_variable_still_builds_one(self):
        cypher = (
            "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 25 "
            "RETURN p.name, q.name ORDER BY p.name"
        )
        assert _index_counts("duckdb", cypher)["vectorized"] == ["Person"]


class TestResultsUnchanged:
    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person) WHERE p.age > 25 RETURN p.name ORDER BY p.name",
            "MATCH (p:Person {city: 'NY'}) RETURN count(p)",
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 25 "
                "RETURN p.name, q.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person) WHERE p.age < 5 "
                "RETURN p.name, p.age, p.city ORDER BY p.name"
            ),
        ],
    )
    def test_backends_agree(self, cypher):
        results = {
            backend: run_query(
                build_context(ENTITIES, RELATIONSHIPS, backend=backend),
                cypher,
            )
            for backend in BACKENDS
        }
        assert results["duckdb"] == results["pandas"]
