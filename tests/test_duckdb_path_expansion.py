"""Phase 7 (DuckDB eager path) — variable-length paths as a recursive CTE.

``PathExpander`` replaces its per-hop ``frontier.merge()`` loop with a
``WITH RECURSIVE`` walk over ``(start_id, tip, hop)``, re-joining the seed
at the end so the recursion moves three columns instead of the seed's full
width.

Most of what matters here is the *declines*: several shapes are deliberately
left to the pandas BFS because the SQL version would not be an exact
substitute.

See docs/duckdb_eager_path_design.md, Phase 7.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import assert_same_across_backends, build_context
from pycypher.ast_models import RelationshipDirection as Direction
from pycypher.backends.duckdb_backend import count_materialisations
from pycypher.binding_frame import BindingFrame
from pycypher.path_expander import PathExpander
from pycypher.scan_operators import EntityScan

PEOPLE = pd.DataFrame({"name": ["a", "b", "c", "d", "e"]})
# A simple chain a->b->c->d->e plus a shortcut a->c.
KNOWS = pd.DataFrame(
    {"__SOURCE__": [0, 1, 2, 3, 0], "__TARGET__": [1, 2, 3, 4, 2]},
)
ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}


@pytest.fixture
def duck():
    return build_context(ENTITIES, RELATIONSHIPS, backend="duckdb")


def _expand(context, **kwargs):
    seed = EntityScan("Person", "p").scan(context)
    defaults = {
        "start_frame": seed,
        "start_var": "p",
        "rel_type": "KNOWS",
        "direction": Direction.RIGHT,
        "end_var": "q",
        "end_type": "Person",
        "min_hops": 1,
        "max_hops": 3,
        "anon_counter": [0],
    }
    defaults.update(kwargs)
    return PathExpander(context).expand_variable_length_path(**defaults)


class TestRecursiveCTE:
    def test_expansion_stays_lazy(self, duck):
        with count_materialisations() as log:
            out = _expand(duck)
            rows = len(out)
        assert out.is_lazy is True
        assert log.count == 0
        assert rows > 0

    def test_output_columns(self, duck):
        assert _expand(duck).var_names == ["p", "q"]

    def test_path_length_column_is_added(self, duck):
        out = _expand(duck, path_length_col="_len")
        assert out.var_names == ["p", "q", "_len"]
        assert out.is_lazy is True

    def test_hop_bounds_are_respected(self, duck):
        out = _expand(duck, min_hops=2, max_hops=2, path_length_col="_len")
        assert set(out.bindings["_len"]) == {2}

    def test_reverse_direction(self, duck):
        out = _expand(duck, direction=Direction.LEFT)
        assert out.is_lazy is True
        assert len(out) > 0

    def test_type_registry_records_the_endpoint(self, duck):
        assert _expand(duck).type_registry["q"] == "Person"


class TestDeclines:
    """Shapes deliberately left to the pandas BFS."""

    def test_row_limit_declines(self, duck):
        # pandas fills hop 1, then hop 2, trimming at the boundary, so which
        # rows survive depends on hop order and on frontier order within a
        # hop.  LIMIT over the CTE would keep a different subset.
        out = _expand(duck, row_limit=2)
        assert out.is_lazy is False
        assert len(out) == 2

    def test_duplicate_start_values_decline(self, duck):
        # The pandas frontier dedupes on (start_var, tip) while carrying the
        # seed's other columns, so a repeated start silently drops seed rows.
        # Joining the narrow walk back to the seed would keep them.
        seed = BindingFrame(
            bindings=pd.DataFrame({"p": [0, 0, 1], "tag": ["x", "y", "z"]}),
            type_registry={"p": "Person"},
            context=duck,
        )
        out = _expand(duck, start_frame=seed)
        assert out.is_lazy is False

    def test_undirected_declines(self, duck):
        # Not supported by the pandas path either, so there is no reference
        # behaviour to match.
        out = _expand(duck, direction=Direction.UNDIRECTED)
        assert out.is_lazy is False

    def test_shadow_overlay_declines(self, duck):
        duck._shadow_rels["KNOWS"] = KNOWS.assign(__ID__=range(len(KNOWS)))
        assert _expand(duck).is_lazy is False

    def test_unregistered_relationship_declines(self):
        context = build_context(ENTITIES, RELATIONSHIPS, backend="pandas")
        assert _expand(context).is_lazy is False


class TestDifferentialQueries:
    @pytest.mark.parametrize(
        "cypher",
        [
            (
                "MATCH (p:Person {name: 'a'})-[:KNOWS*1..3]->(q:Person) "
                "RETURN q.name ORDER BY q.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS*1..2]->(q:Person) "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person {name: 'a'})-[:KNOWS*2..2]->(q:Person) "
                "RETURN q.name ORDER BY q.name"
            ),
            "MATCH (p:Person)-[:KNOWS*1..4]->(q:Person) RETURN count(q)",
            (
                "MATCH (p:Person {name: 'e'})<-[:KNOWS*1..3]-(q:Person) "
                "RETURN q.name ORDER BY q.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS*1..3]->(q:Person) "
                "RETURN p.name, count(q) ORDER BY p.name"
            ),
        ],
    )
    def test_query_matches_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher, RELATIONSHIPS)


class TestCycles:
    """A cyclic graph is where per-hop dedup earns its keep."""

    def test_cycle_terminates_and_matches_pandas(self):
        entities = {"N": pd.DataFrame({"name": ["x", "y", "z"]})}
        # A 3-cycle: 0 -> 1 -> 2 -> 0.
        edges = pd.DataFrame(
            {"__SOURCE__": [0, 1, 2], "__TARGET__": [1, 2, 0]},
        )
        relationships = {"E": (edges, "__SOURCE__", "__TARGET__")}
        assert_same_across_backends(
            entities,
            "MATCH (a:N)-[:E*1..4]->(b:N) RETURN a.name, b.name "
            "ORDER BY a.name, b.name",
            relationships,
        )
