"""Emitter tests: the SQL shape of each operator, and executed results for
the constructs whose SQL is easy to get subtly wrong (LEFT joins, GROUP BY
with passthrough nodes, NULLS LAST ordering, post-projection WHERE)."""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.plan import translate
from pycypher.plan.emit_duckdb import Emitter
from pycypher.plan.nodes import Mutate, Project, Scan, Scope, Unit
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
    pd.DataFrame(
        {"pid": [1, 2, 3], "name": ["A", "B", None], "age": [3, None, 1]}
    ).to_parquet(tmp_path / "p.parquet")
    pd.DataFrame({"src": [1], "tgt": [2]}).to_parquet(tmp_path / "k.parquet")
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
    register_streaming_relationship(
        context,
        "KNOWS",
        data_source_from_uri(str(tmp_path / "k.parquet")),
        source_col="src",
        target_col="tgt",
    )
    return context


def sql(ctx, cypher) -> str:
    return Emitter(ctx).emit(translate(ASTConverter.from_cypher(cypher), ctx))


def run(ctx, cypher) -> list[tuple]:
    return (
        Emitter(ctx)
        .run(translate(ASTConverter.from_cypher(cypher), ctx))
        .fetchall()
    )


class TestSqlShape:
    def test_scan_and_projection_fuse_without_a_join(self, ctx):
        # The single-table case must stay a single scan: no self-join on
        # the id, or a memory-limited scan could not stream.
        s = sql(ctx, "MATCH (p:Person) RETURN p.name AS n")
        assert "JOIN" not in s
        assert s.count('"_streaming_source_Person"') == 1

    def test_property_read_after_projection_is_a_left_join_on_the_id(
        self, ctx
    ):
        s = sql(
            ctx, "MATCH (p:Person) WITH p, 1 AS one RETURN p.name AS n, one"
        )
        assert 'LEFT JOIN "_streaming_source_Person"' in s
        assert '"pid" = t."p"' in s

    def test_optional_match_uses_left_joins_for_edge_and_node(self, ctx):
        s = sql(
            ctx,
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) RETURN q.name AS n",
        )
        # One LEFT JOIN against the (edge JOIN node) pair -- not two
        # successive LEFT JOINs, which would keep an edge whose far end is
        # not a node of the pattern's label as a spurious NULL row.
        assert s.count("LEFT JOIN") == 1
        assert "LEFT JOIN (" in s
        assert s.count("JOIN") == 2

    def test_aggregation_groups_by_passthrough_id(self, ctx):
        s = sql(
            ctx,
            "MATCH (p:Person)-[:KNOWS]->(q:Person) WITH p, COUNT(q) AS n RETURN n",
        )
        assert 'GROUP BY n1."pid"' in s
        assert 'COUNT(n3."pid")' in s

    def test_order_by_emits_nulls_last(self, ctx):
        assert "NULLS LAST" in sql(
            ctx, "MATCH (p:Person) RETURN p.name AS n ORDER BY n"
        )

    def test_emit_refuses_side_effects(self, ctx):
        from pycypher.plan import Unsupported

        with pytest.raises(Unsupported, match="side effects"):
            sql(ctx, "MATCH (p:Person) SET p.x = 1")


class TestExecutedSemantics:
    def test_nulls_sort_last_both_directions(self, ctx):
        assert run(ctx, "MATCH (p:Person) RETURN p.name AS n ORDER BY n") == [
            ("A",),
            ("B",),
            (None,),
        ]
        assert run(
            ctx, "MATCH (p:Person) RETURN p.name AS n ORDER BY n DESC"
        ) == [("B",), ("A",), (None,)]

    def test_post_projection_where_filters_output_scope(self, ctx):
        assert run(
            ctx, "MATCH (p:Person) WITH p.age AS a WHERE a > 1 RETURN a"
        ) == [(3,)]

    def test_count_of_optional_node_ignores_unmatched(self, ctx):
        got = run(
            ctx,
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) WITH p, COUNT(q) AS n RETURN id(p) AS pid, n ORDER BY pid",
        )
        assert got == [(1, 1), (2, 0), (3, 0)]

    def test_run_returns_none_for_mutation_only_plans(self, ctx):
        assert (
            Emitter(ctx).run(
                translate(
                    ASTConverter.from_cypher("MATCH (p:Person) SET p.x = 1"),
                    ctx,
                )
            )
            is None
        )
        assert run(ctx, "MATCH (p:Person) RETURN p.x AS x ORDER BY x") == [
            (1,),
            (1,),
            (1,),
        ]

    def test_mutate_dedups_fanned_out_rows_per_target(self, ctx):
        # Two edges into the same target would produce two candidate
        # values; exactly one row per target id reaches the UPDATE.
        Emitter(ctx).run(
            translate(
                ASTConverter.from_cypher(
                    "MATCH (p:Person)-[:KNOWS]->(q:Person) SET q.friend = p.name"
                ),
                ctx,
            )
        )
        assert run(
            ctx, "MATCH (q:Person) RETURN id(q) AS i, q.friend AS f ORDER BY i"
        ) == [(1, None), (2, "A"), (3, None)]

    def test_unit_plan_projects_a_single_row(self, ctx):
        p = Project(Scope(), child=Unit(Scope()), items=())
        # An empty projection is rejected by the translator; the emitter's
        # Unit alone yields one row.
        assert Emitter(ctx).con.sql(
            Emitter(ctx).sql(Unit(Scope()))
        ).fetchall() == [(1,)]
        assert isinstance(p.child, Unit)

    def test_scan_node_alone(self, ctx):
        from pycypher.plan.nodes import Binding

        s = Scan(
            Scope((Binding("p", "node", "Person"),)), var="p", label="Person"
        )
        got = Emitter(ctx).con.sql(Emitter(ctx).sql(s)).fetchall()
        assert sorted(r[0] for r in got) == [1, 2, 3]
        assert not isinstance(s, Mutate)
