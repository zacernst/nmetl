"""Golden plans and rule-level tests for pycypher.plan.translate.

A golden plan pins the *shape* of the translation — which operators, in
which order, binding which variables — so a change to a rule shows up as
a readable diff rather than only as a changed result. Expected renderings
are hand-written.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.plan import Unsupported, translate
from pycypher.plan.nodes import (
    Binding,
    Expand,
    Filter,
    Join,
    Mutate,
    Project,
    Scan,
    Scope,
    Unnest,
    describe,
)
from pycypher.relation_engine import (
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)


@pytest.fixture
def ctx(tmp_path) -> Context:
    pd.DataFrame({"pid": [1], "name": ["A"], "age": [1]}).to_parquet(
        tmp_path / "p.parquet"
    )
    pd.DataFrame({"cid": [1], "name": ["C"]}).to_parquet(
        tmp_path / "c.parquet"
    )
    pd.DataFrame({"src": [1], "tgt": [1], "since": [1]}).to_parquet(
        tmp_path / "k.parquet"
    )
    context = Context(
        entity_mapping=EntityMapping(mapping={}),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend="duckdb",
    )
    context._relation_engine_enabled = True
    register_streaming_source(
        context,
        "Person",
        data_source_from_uri(str(tmp_path / "p.parquet")),
        id_col="pid",
    )
    register_streaming_source(
        context,
        "City",
        data_source_from_uri(str(tmp_path / "c.parquet")),
        id_col="cid",
    )
    register_streaming_relationship(
        context,
        "KNOWS",
        data_source_from_uri(str(tmp_path / "k.parquet")),
        source_col="src",
        target_col="tgt",
    )
    register_streaming_relationship(
        context,
        "LIVES_IN",
        data_source_from_uri(str(tmp_path / "k.parquet")),
        source_col="src",
        target_col="tgt",
    )
    return context


def plan(ctx, cypher):
    return translate(ASTConverter.from_cypher(cypher), ctx)


GOLDEN = {
    "MATCH (p:Person) RETURN p.name AS name": """\
Project(items=(name=PropertyLookup)) -> [name]
  Scan(var='p', label='Person') -> [p]""",
    "MATCH (p:Person)-[k:KNOWS]->(q:Person) WHERE p.age > 1 RETURN q.name AS n": """\
Project(items=(n=PropertyLookup)) -> [n]
  Filter(predicates=(Comparison)) -> [p, k, q]
    Expand(from_var='p', rel_type='KNOWS', right=True, to_var='q', to_label='Person', rel_var='k') -> [p, k, q]
      Scan(var='p', label='Person') -> [p]""",
    "MATCH (c:City) OPTIONAL MATCH (c)<-[:LIVES_IN]-(p:Person) WITH c, COUNT(p) AS n RETURN c.name AS city, n": """\
Project(items=(city=PropertyLookup, n=Variable)) -> [city, n]
  Project(items=(c, n=FunctionInvocation)) -> [c, n]
    Expand(from_var='c', rel_type='LIVES_IN', to_var='p', to_label='Person', optional=True) -> [c, p]
      Scan(var='c', label='City') -> [c]""",
    "MATCH (p:Person) WITH p.name AS name UNWIND [1, 2] AS k RETURN name, k ORDER BY k LIMIT 1": """\
Project(items=(name=Variable, k=Variable), order_by=(('k', True)), limit=1) -> [name, k]
  Unnest(expr=ListLiteral, var='k') -> [name, k]
    Project(items=(name=PropertyLookup)) -> [name]
      Scan(var='p', label='Person') -> [p]""",
    "MATCH (p:Person) WITH p MATCH (p)-[:KNOWS]->(q:Person) SET p.x = q.age": """\
Mutate(target='p', label='Person', assignments=(('x', PropertyLookup))) -> [p, q]
  Join(on=('p')) -> [p, q]
    Project(items=(p)) -> [p]
      Scan(var='p', label='Person') -> [p]
    Expand(from_var='p', rel_type='KNOWS', right=True, to_var='q', to_label='Person') -> [p, q]
      Scan(var='p', label='Person') -> [p]""",
}


@pytest.mark.parametrize(
    ("cypher", "expected"), list(GOLDEN.items()), ids=list(range(len(GOLDEN)))
)
def test_golden_plans(ctx, cypher, expected):
    assert describe(plan(ctx, cypher)) == expected


class TestRules:
    def test_scan_binds_a_node(self, ctx):
        p = plan(ctx, "MATCH (p:Person) RETURN p.name AS n")
        assert isinstance(p.child, Scan)
        assert p.child.scope == Scope(
            (Binding("p", "node", "Person"),), qualified=False
        )

    def test_expand_marks_scope_qualified(self, ctx):
        p = plan(
            ctx, "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p.name, q.name"
        )
        assert isinstance(p.child, Expand)
        assert p.child.scope.qualified is True
        assert [i.name for i in p.items] == ["p.name", "q.name"]

    def test_inline_properties_become_filters(self, ctx):
        p = plan(ctx, "MATCH (p:Person {name: 'A'}) RETURN p.age AS a")
        assert isinstance(p.child, Filter)
        assert p.child.predicates[0].var == "p"
        assert p.child.predicates[0].prop == "name"

    def test_projection_scope_drops_unmentioned_bindings(self, ctx):
        p = plan(
            ctx,
            "MATCH (p:Person)-[k:KNOWS]->(q:Person) WITH p RETURN p.name AS n",
        )
        assert p.child.scope.names == ("p",)
        assert isinstance(p.child, Project)

    def test_set_records_new_property_for_later_stages(self, ctx):
        p = plan(
            ctx,
            "MATCH (p:Person) SET p.brand_new = 1 WITH p.brand_new AS b RETURN b",
        )
        assert isinstance(p.child.child, Mutate)
        assert ("p", "brand_new") in p.child.child.scope.new_props

    def test_second_match_without_shared_vars_is_a_cross_join(self, ctx):
        p = plan(
            ctx,
            "MATCH (c:City) WITH c.name AS city MATCH (p:Person) RETURN city, p.name AS n",
        )
        assert isinstance(p.child, Join)
        assert p.child.on == ()

    def test_unwind_shadowing_is_unsupported(self, ctx):
        with pytest.raises(Unsupported, match="already bound"):
            plan(
                ctx,
                "MATCH (p:Person) WITH p.name AS x UNWIND [1] AS x RETURN x",
            )

    def test_duplicate_output_names_are_unsupported(self, ctx):
        with pytest.raises(Unsupported, match="duplicate"):
            plan(ctx, "MATCH (p:Person) RETURN p.name AS a, p.age AS a")

    def test_set_on_unregistered_label_is_unsupported(self, ctx):
        # Entities only reachable through the in-memory mapping have no
        # table to write to.
        context = Context(
            entity_mapping=EntityMapping(mapping={}),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="duckdb",
        )
        with pytest.raises(Unsupported, match="label"):
            translate(
                ASTConverter.from_cypher("MATCH (p:Ghost) SET p.x = 1"),
                context,
            )

    def test_unnest_keeps_prior_bindings(self, ctx):
        p = plan(
            ctx, "MATCH (p:Person) UNWIND [1, 2] AS k RETURN p.name AS n, k"
        )
        assert isinstance(p.child, Unnest)
        assert p.child.scope.names == ("p", "k")
