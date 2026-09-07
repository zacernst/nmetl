"""Phase 0/2 of docs/cypher_relational_algebra_generalization_plan.md — the
golden corpus.

Every query below runs through the plan-based relation engine against a
small fixed graph, and every expected result was computed *by hand* from
that graph — not by another engine. The pandas engine is not an oracle
(see the qualification plan, Phase 4: it mis-joins on labels, misaligns
mid-pipeline SETs, and drops copy-SET values).

Graph::

    Person(pid, name, age)      1 Alice 30 | 2 Bob 25 | 3 Carol 35 | 4 Dave 40
    City(cid, name, state)      10 Springfield IL | 20 Shelbyville IL | 30 Ogdenville NT
    KNOWS(src, tgt, since)      1->2 2020 | 2->3 2021 | 1->3 2019
    LIVES_IN(src, tgt)          1->10 | 2->10 | 3->20        (Dave lives nowhere)
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.plan import Unsupported, translate
from pycypher.plan.emit_duckdb import Emitter
from pycypher.plan.report import classify, summarize
from pycypher.relation_engine import (
    execute_relation_mutation,
    execute_relation_query,
    is_relation_eligible,
    is_relation_mutation_eligible,
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)


def _ast(q: str):
    return ASTConverter.from_cypher(q)


@pytest.fixture
def graph(tmp_path) -> Context:
    pd.DataFrame(
        {
            "pid": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "Dave"],
            "age": [30, 25, 35, 40],
        }
    ).to_parquet(tmp_path / "person.parquet")
    pd.DataFrame(
        {
            "cid": [10, 20, 30],
            "name": ["Springfield", "Shelbyville", "Ogdenville"],
            "state": ["IL", "IL", "NT"],
        }
    ).to_parquet(tmp_path / "city.parquet")
    pd.DataFrame(
        {"src": [1, 2, 1], "tgt": [2, 3, 3], "since": [2020, 2021, 2019]}
    ).to_parquet(tmp_path / "knows.parquet")
    pd.DataFrame({"src": [1, 2, 3], "tgt": [10, 10, 20]}).to_parquet(
        tmp_path / "lives.parquet"
    )
    ctx = Context(
        entity_mapping=EntityMapping(mapping={}),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend="duckdb",
    )
    ctx._relation_engine_enabled = True
    register_streaming_source(
        ctx,
        "Person",
        data_source_from_uri(str(tmp_path / "person.parquet")),
        id_col="pid",
    )
    register_streaming_source(
        ctx,
        "City",
        data_source_from_uri(str(tmp_path / "city.parquet")),
        id_col="cid",
    )
    register_streaming_relationship(
        ctx,
        "KNOWS",
        data_source_from_uri(str(tmp_path / "knows.parquet")),
        source_col="src",
        target_col="tgt",
    )
    register_streaming_relationship(
        ctx,
        "LIVES_IN",
        data_source_from_uri(str(tmp_path / "lives.parquet")),
        source_col="src",
        target_col="tgt",
    )
    return ctx


def rows(ctx: Context, cypher: str) -> list[list]:
    ast = _ast(cypher)
    assert is_relation_eligible(ast, ctx), cypher
    df = execute_relation_query(ast, ctx)
    return [
        [None if pd.isna(v) else v for v in r] for r in df.to_numpy().tolist()
    ]


def mutate(ctx: Context, cypher: str) -> None:
    ast = _ast(cypher)
    kind = is_relation_mutation_eligible(ast, ctx)
    assert kind is not None, cypher
    execute_relation_mutation(ast, ctx, kind)


READ_CORPUS: list[tuple[str, str, list[list]]] = [
    (
        "scan_props",
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name",
        [["Alice"], ["Bob"], ["Carol"], ["Dave"]],
    ),
    (
        "where",
        "MATCH (p:Person) WHERE p.age > 28 RETURN p.name AS name ORDER BY name",
        [["Alice"], ["Carol"], ["Dave"]],
    ),
    (
        "inline_prop",
        "MATCH (p:Person {name: 'Bob'}) RETURN p.age AS age",
        [[25]],
    ),
    (
        "expand_right",
        "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p.name AS a, q.name AS b ORDER BY a, b",
        [["Alice", "Bob"], ["Alice", "Carol"], ["Bob", "Carol"]],
    ),
    (
        "expand_left",
        "MATCH (p:Person)<-[:KNOWS]-(q:Person) RETURN p.name AS a, q.name AS b ORDER BY a, b",
        [["Bob", "Alice"], ["Carol", "Alice"], ["Carol", "Bob"]],
    ),
    (
        "rel_property",
        "MATCH (p:Person)-[k:KNOWS]->(q:Person) RETURN p.name AS a, k.since AS since ORDER BY since",
        [["Alice", 2019], ["Alice", 2020], ["Bob", 2021]],
    ),
    (
        "two_hops",
        "MATCH (p:Person)-[:KNOWS]->(q:Person)-[:LIVES_IN]->(c:City) RETURN p.name AS a, c.name AS city ORDER BY a, city",
        [
            ["Alice", "Shelbyville"],
            ["Alice", "Springfield"],
            ["Bob", "Shelbyville"],
        ],
    ),
    (
        "optional_match",
        "MATCH (c:City) OPTIONAL MATCH (c)<-[:LIVES_IN]-(p:Person) RETURN c.name AS city, p.name AS person ORDER BY city, person",
        [
            ["Ogdenville", None],
            ["Shelbyville", "Carol"],
            ["Springfield", "Alice"],
            ["Springfield", "Bob"],
        ],
    ),
    (
        "optional_shared_edge_type_no_fanout",
        # KNOWS edges never reach a City. Cypher yields exactly one row per
        # person (city NULL); a naive edge-then-node LEFT JOIN would emit
        # one NULL row per edge instead.
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(c:City) RETURN p.name AS name, c.name AS city ORDER BY name",
        [["Alice", None], ["Bob", None], ["Carol", None], ["Dave", None]],
    ),
    (
        "optional_chain_from_optional",
        (
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c:City) OPTIONAL MATCH (c)<-[:LIVES_IN]-(q:Person) "
            "WITH p.name AS name, COUNT(q) AS neighbours RETURN name, neighbours ORDER BY name"
        ),
        [["Alice", 2], ["Bob", 2], ["Carol", 1], ["Dave", 0]],
    ),
    (
        "optional_then_count",
        "MATCH (c:City) OPTIONAL MATCH (c)<-[:LIVES_IN]-(p:Person) WITH c.name AS city, COUNT(p) AS n RETURN city, n ORDER BY city",
        [["Ogdenville", 0], ["Shelbyville", 1], ["Springfield", 2]],
    ),
    (
        "group_by_node",
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH c, COUNT(p) AS n RETURN c.name AS city, n ORDER BY city",
        [["Shelbyville", 1], ["Springfield", 2]],
    ),
    (
        "distinct",
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN DISTINCT c.state AS state",
        [["IL"]],
    ),
    (
        "order_limit",
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age DESC LIMIT 2",
        [["Dave", 40], ["Carol", 35]],
    ),
    (
        "skip_limit",
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age ORDER BY age ASC SKIP 1 LIMIT 2",
        [["Alice", 30], ["Carol", 35]],
    ),
    ("leading_unwind", "UNWIND [1, 2, 3] AS x RETURN x", [[1], [2], [3]]),
    (
        "mid_unwind",
        "MATCH (p:Person) WITH p.name AS name UNWIND [1, 2] AS k RETURN name, k ORDER BY name, k",
        [
            ["Alice", 1],
            ["Alice", 2],
            ["Bob", 1],
            ["Bob", 2],
            ["Carol", 1],
            ["Carol", 2],
            ["Dave", 1],
            ["Dave", 2],
        ],
    ),
    (
        "with_where",
        "MATCH (p:Person) WITH p.age AS age WHERE age > 30 RETURN age ORDER BY age",
        [[35], [40]],
    ),
    (
        "having",
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH c.name AS city, COUNT(p) AS n WHERE n > 1 RETURN city",
        [["Springfield"]],
    ),
    (
        "second_match_shared_var",
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH p MATCH (p)-[:KNOWS]->(q:Person) RETURN p.name AS a, q.name AS b ORDER BY a, b",
        [["Alice", "Bob"], ["Alice", "Carol"], ["Bob", "Carol"]],
    ),
    (
        "id_function",
        "MATCH (p:Person) RETURN id(p) AS pid ORDER BY pid",
        [[1], [2], [3], [4]],
    ),
    (
        "case_in_group",
        "MATCH (p:Person) RETURN CASE WHEN p.age >= 35 THEN 'senior' ELSE 'junior' END AS band, COUNT(p) AS n ORDER BY band",
        [["junior", 2], ["senior", 2]],
    ),
    (
        "cast_arithmetic",
        "MATCH (p:Person) RETURN toFloat(p.age) / 2 AS half ORDER BY half",
        [[12.5], [15.0], [17.5], [20.0]],
    ),
    ("leading_with", "WITH 1 AS x RETURN x + 1 AS y", [[2]]),
    (
        "qualified_bare_props",
        "MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p.name, q.name ORDER BY p.name, q.name",
        [["Alice", "Bob"], ["Alice", "Carol"], ["Bob", "Carol"]],
    ),
]


@pytest.mark.parametrize(
    ("qid", "cypher", "expected"), READ_CORPUS, ids=[q[0] for q in READ_CORPUS]
)
def test_read_corpus(graph, qid, cypher, expected):
    assert rows(graph, cypher) == expected, qid


def test_cross_join_after_with(graph):
    got = rows(
        graph,
        "MATCH (c:City) WITH c.name AS city MATCH (p:Person) RETURN city, p.name AS name ORDER BY city, name",
    )
    assert len(got) == 12
    assert got[0] == ["Ogdenville", "Alice"]


def test_qualified_output_names(graph):
    df = execute_relation_query(
        _ast("MATCH (p:Person)-[:KNOWS]->(q:Person) RETURN p.name, q.name"),
        graph,
    )
    assert list(df.columns) == ["p.name", "q.name"]
    df = execute_relation_query(_ast("MATCH (p:Person) RETURN p.name"), graph)
    assert list(df.columns) == ["name"]


class TestMutationCorpus:
    def test_set_with_where_creates_typed_column(self, graph):
        mutate(graph, "MATCH (p:Person) WHERE p.age > 30 SET p.senior = true")
        assert rows(
            graph,
            "MATCH (p:Person) RETURN p.name AS name, p.senior AS senior ORDER BY name",
        ) == [
            ["Alice", None],
            ["Bob", None],
            ["Carol", True],
            ["Dave", True],
        ]

    def test_group_set(self, graph):
        mutate(
            graph,
            "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH c, COUNT(p) AS n SET c.residents = n",
        )
        assert rows(
            graph,
            "MATCH (c:City) RETURN c.name AS city, c.residents AS r ORDER BY city",
        ) == [
            ["Ogdenville", None],
            ["Shelbyville", 1],
            ["Springfield", 2],
        ]

    def test_copy_set(self, graph):
        mutate(
            graph,
            "MATCH (p:Person)-[:LIVES_IN]->(c:City) SET p.city_name = c.name",
        )
        assert rows(
            graph,
            "MATCH (p:Person) RETURN p.name AS name, p.city_name AS city ORDER BY name",
        ) == [
            ["Alice", "Springfield"],
            ["Bob", "Springfield"],
            ["Carol", "Shelbyville"],
            ["Dave", None],
        ]

    def test_mid_pipeline_set_then_read(self, graph):
        got = rows(
            graph,
            "MATCH (p:Person) WITH id(p) AS pid, p SET p.tag = pid * 10 WITH pid, p.tag AS tag RETURN pid, tag ORDER BY pid",
        )
        assert got == [[1, 10], [2, 20], [3, 30], [4, 40]]
        # Durable, not just in-flight.
        assert rows(
            graph, "MATCH (p:Person) RETURN p.tag AS tag ORDER BY tag"
        ) == [[10], [20], [30], [40]]

    def test_set_after_join_and_aggregate_in_one_query(self, graph):
        mutate(
            graph,
            "MATCH (p:Person)-[:KNOWS]->(q:Person) WITH p, COUNT(q) AS friends SET p.friends = friends",
        )
        assert rows(
            graph,
            "MATCH (p:Person) RETURN p.name AS name, p.friends AS f ORDER BY name",
        ) == [
            ["Alice", 2],
            ["Bob", 1],
            ["Carol", None],
            ["Dave", None],
        ]

    def test_delete(self, graph):
        mutate(graph, "MATCH (p:Person) WHERE p.name = 'Dave' DELETE p")
        assert rows(graph, "MATCH (p:Person) RETURN COUNT(p) AS n") == [[3]]

    def test_create(self, graph):
        mutate(graph, "CREATE (p:Person {name: 'Eve', age: 22})")
        assert rows(
            graph,
            "MATCH (p:Person) WHERE p.name = 'Eve' RETURN p.age AS age, id(p) AS pid",
        ) == [[22, 5]]

    def test_mutation_kind_names(self, graph):
        kinds = {
            "MATCH (p:Person) SET p.x = 1": "set",
            "MATCH (p:Person) WITH p, p.age AS a SET p.x = a": "scalar_set",
            "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH c, COUNT(p) AS n SET c.x = n": "group_set",
            "MATCH (p:Person)-[:LIVES_IN]->(c:City) SET p.x = c.name": "copy_set",
            "MATCH (p:Person) DELETE p": "delete",
            "CREATE (p:Person {name: 'Z'})": "create",
        }
        for q, kind in kinds.items():
            assert is_relation_mutation_eligible(_ast(q), graph) == kind, q


UNSUPPORTED: list[tuple[str, str]] = [
    (
        "MATCH (p:Person)-[:KNOWS]-(q:Person) RETURN p.name AS a",
        "undirected relationship",
    ),
    (
        "MATCH (p:Person)-[:KNOWS*1..2]->(q:Person) RETURN p.name AS a",
        "variable-length path",
    ),
    ("MATCH (p:Person) RETURN p", "RETURN"),
    ("MATCH (p:Person) RETURN collect(p.name) AS names", "expression"),
    ("MATCH (p:Person) RETURN p.name AS n ORDER BY p.age", "ORDER BY"),
    ("MATCH (p:Person) DETACH DELETE p", "DETACH DELETE"),
    ("MATCH (p:Person) CREATE (q:City {name: 'X'})", "CREATE"),
    ("MATCH (p:Person) RETURN p.nope AS x", "expression"),
    ("MATCH (p:Nope) RETURN p.name AS x", "label"),
]


@pytest.mark.parametrize(
    ("cypher", "construct"), UNSUPPORTED, ids=[u[1] for u in UNSUPPORTED]
)
def test_unsupported_names_the_construct(graph, cypher, construct):
    with pytest.raises(Unsupported) as exc:
        translate(_ast(cypher), graph)
    assert exc.value.construct == construct
    assert classify(_ast(cypher), graph) == ("unsupported", construct)


def test_summarize_groups_by_construct(graph):
    queries = [(f"q{i}", _ast(q)) for i, (q, _) in enumerate(UNSUPPORTED[:2])]
    queries += [
        ("r", _ast("MATCH (p:Person) RETURN p.name AS n")),
        ("m", _ast("MATCH (p:Person) SET p.x = 1")),
    ]
    got = summarize(queries, graph)
    assert got["read"] == ["r"]
    assert got["mutation"] == ["m"]
    assert got["unsupported:undirected relationship"] == ["q0"]
    assert got["unsupported:variable-length path"] == ["q1"]


def test_never_materialises_to_pandas(graph, monkeypatch):
    from pycypher.backends import _helpers

    calls: list[type] = []
    original = _helpers._to_pandas

    def counted(obj):
        calls.append(type(obj))
        return original(obj)

    monkeypatch.setattr(_helpers, "_to_pandas", counted)
    for _, cypher, _ in READ_CORPUS:
        plan = translate(_ast(cypher), graph)
        Emitter(graph).run(plan).fetchall()
    assert calls == []


class TestNoFallbackThroughStar:
    """Phase 5: with the relation engine enabled on a DuckDB backend,
    Star.execute_query raises Unsupported instead of switching engines;
    with it disabled the in-memory engine still answers."""

    def test_enabled_raises_naming_the_construct(self, graph):
        from pycypher.star import Star

        with pytest.raises(Unsupported, match="undirected relationship"):
            Star(context=graph).execute_query(
                "MATCH (p:Person)-[:KNOWS]-(q:Person) RETURN p.name AS a"
            )

    def test_disabled_falls_back_to_the_in_memory_engine(self):
        from helpers_differential import build_context, run_query

        ctx = build_context(
            {"Person": pd.DataFrame({"__ID__": [1, 2], "name": ["A", "B"]})},
            {
                "KNOWS": (
                    pd.DataFrame({"__SOURCE__": [1], "__TARGET__": [2]}),
                    "__SOURCE__",
                    "__TARGET__",
                )
            },
            backend="duckdb",
        )
        ctx._relation_engine_enabled = False
        got = run_query(
            ctx, "MATCH (p:Person)-[:KNOWS]-(q:Person) RETURN p.name AS a"
        )
        assert sorted(r[0] for r in got) == ["A", "B"]

    def test_stream_query_to_uri_raises_when_unsupported(
        self, graph, tmp_path
    ):
        from pycypher.star import Star

        with pytest.raises(Unsupported, match="undirected relationship"):
            Star(context=graph).stream_query_to_uri(
                "MATCH (p:Person)-[:KNOWS]-(q:Person) RETURN p.name AS a",
                str(tmp_path / "out.parquet"),
            )


class TestTerminalMutationsRunDirect:
    """Phase 6: a mutation at the top of the plan is one DML statement over
    the fused SELECT, with no temporary table."""

    def test_group_set_is_a_single_update(self, graph):
        plan = translate(
            _ast(
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) WITH c, COUNT(p) AS n SET c.residents = n"
            ),
            graph,
        )
        stmts = Emitter(graph).dml_statements(plan)
        assert len(stmts) == 1
        assert stmts[0].startswith('UPDATE "_streaming_source_City"')
        assert "TEMP" not in stmts[0]
        assert "GROUP BY" in stmts[0]

    def test_copy_set_reads_properties_in_the_same_pass(self, graph):
        plan = translate(
            _ast(
                "MATCH (p:Person)-[:LIVES_IN]->(c:City) SET p.city_name = c.name"
            ),
            graph,
        )
        (sql,) = Emitter(graph).dml_statements(plan)
        # One join to City for the pattern, none added to re-read c.name.
        assert sql.count('"_streaming_source_City"') == 1

    def test_delete_is_a_single_statement(self, graph):
        plan = translate(
            _ast("MATCH (p:Person) WHERE p.age > 30 DELETE p"), graph
        )
        (sql,) = Emitter(graph).dml_statements(plan)
        assert sql.startswith('DELETE FROM "_streaming_source_Person"')

    def test_set_then_read_still_materialises_for_the_later_stage(
        self, graph, monkeypatch
    ):
        sources: list[str] = []
        original = Emitter._run_mutate

        def spy(self, plan, source):
            sources.append(source)
            return original(self, plan, source)

        monkeypatch.setattr(Emitter, "_run_mutate", spy)
        rows(
            graph,
            "MATCH (p:Person) WITH p SET p.tag = 1 WITH p.tag AS tag RETURN tag",
        )
        assert len(sources) == 1
        assert sources[0].startswith(
            '"__pycypher_plan_'
        )  # a temp table, not a subquery
