"""Streaming entity registration — load a file source without pandas.

Open item 2 from ``docs/duckdb_eager_path_report.md``, and the piece Phase 8
identified as missing.  ``add_entity(..., streaming=True)`` defers the read
and scans the file straight into a DuckDB table at ``build()`` time, so the
rows never pass through pandas or Arrow.

Two properties matter and are tested separately:

* **Equivalence** — a streaming entity must answer exactly as an eagerly
  loaded one, including id assignment and ``__ID__`` de-duplication.
* **Non-materialisation** — the whole point is that the source is never
  pulled into pandas, and correctness alone cannot detect a regression there
  because the fallback proxy would silently make it correct-but-unbounded.

See docs/duckdb_eager_path_design.md, Phase 8.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import run_query
from pycypher.ingestion.context_builder import ContextBuilder
from pycypher.ingestion.streaming_entity import (
    LazyEntitySource,
    is_streaming_source,
)

QUERIES = [
    "MATCH (p:Person) RETURN count(p)",
    "MATCH (p:Person) WHERE p.age > 30 RETURN p.name ORDER BY p.name",
    "MATCH (p:Person) WHERE p.city = 'NY' RETURN p.name ORDER BY p.name",
    "MATCH (p:Person {city: 'LA'}) RETURN p.name ORDER BY p.name",
    (
        "MATCH (p:Person) WHERE p.age > 25 "
        "RETURN p.name, p.age, p.city ORDER BY p.name"
    ),
]


@pytest.fixture
def people_parquet(tmp_path):
    path = tmp_path / "people.parquet"
    pd.DataFrame(
        {
            "pid": [10, 20, 30, 40],
            "name": ["Alice", "Bob", "Carol", "Dan"],
            "age": [30, 25, 35, 40],
            "city": ["NY", "LA", "NY", "SF"],
        },
    ).to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def duplicated_parquet(tmp_path):
    path = tmp_path / "dupes.parquet"
    pd.DataFrame(
        {
            "pid": [1, 1, 2],
            "name": ["first", "second", "other"],
        },
    ).to_parquet(path, index=False)
    return str(path)


def _stream(path, **kwargs):
    return (
        ContextBuilder()
        .add_entity("Person", path, streaming=True, **kwargs)
        .build(backend="duckdb")
    )


def _eager(path, backend="pandas", **kwargs):
    return (
        ContextBuilder()
        .add_entity("Person", path, **kwargs)
        .build(backend=backend)
    )


class TestEquivalence:
    @pytest.mark.parametrize("cypher", QUERIES)
    @pytest.mark.parametrize("id_col", ["pid", None])
    def test_matches_the_eager_path(self, people_parquet, cypher, id_col):
        streamed = run_query(_stream(people_parquet, id_col=id_col), cypher)
        eager = run_query(_eager(people_parquet, id_col=id_col), cypher)
        assert streamed == eager

    def test_named_id_column_becomes_the_entity_id(self, people_parquet):
        context = _stream(people_parquet, id_col="pid")
        relation = context.backend.tables.relation("Person")
        assert sorted(relation.project('"__ID__"').fetchdf()["__ID__"]) == [
            10,
            20,
            30,
            40,
        ]

    def test_absent_id_column_generates_scan_order_ids(self, people_parquet):
        # Must match the Arrow path's range(len(table)) — 0-based, in file
        # order — or ids would silently disagree between the two paths.
        context = _stream(people_parquet)
        relation = context.backend.tables.relation("Person")
        frame = relation.project('"__ID__", "name"').fetchdf()
        assert frame["__ID__"].tolist() == [0, 1, 2, 3]
        assert frame["name"].tolist() == ["Alice", "Bob", "Carol", "Dan"]

    def test_id_column_is_not_a_property(self, people_parquet):
        context = _stream(people_parquet, id_col="pid")
        assert "pid" not in context.entity_mapping["Person"].attribute_map

    def test_duplicate_ids_keep_the_first_row(self, duplicated_parquet):
        streamed = _stream(duplicated_parquet, id_col="pid")
        eager = _eager(duplicated_parquet, id_col="pid")
        cypher = "MATCH (p:Person) RETURN p.name ORDER BY p.name"
        assert run_query(streamed, cypher) == run_query(eager, cypher)
        assert run_query(streamed, cypher) == [["first"], ["other"]]

    def test_missing_id_column_raises_like_the_eager_path(
        self, people_parquet
    ):
        with pytest.raises(ValueError, match="not found in table columns"):
            _stream(people_parquet, id_col="nope")


class TestSourceIsNotMaterialised:
    """Correctness cannot detect a regression here — the proxy hides it."""

    def _materialisations(self, context, cypher, monkeypatch):
        calls: list[str] = []
        original = LazyEntitySource.to_pandas

        def counted(inner_self):
            calls.append(inner_self._label)
            return original(inner_self)

        monkeypatch.setattr(LazyEntitySource, "to_pandas", counted)
        run_query(context, cypher)
        return calls

    @pytest.mark.parametrize("cypher", QUERIES)
    def test_queries_never_load_the_source(
        self, people_parquet, cypher, monkeypatch
    ):
        context = _stream(people_parquet, id_col="pid")
        assert self._materialisations(context, cypher, monkeypatch) == []

    def test_source_object_is_the_proxy(self, people_parquet):
        context = _stream(people_parquet, id_col="pid")
        source = context.entity_mapping["Person"].source_obj
        assert is_streaming_source(source)
        assert len(source) == 4
        assert "name" in source.columns


class TestFallbackSafety:
    def test_the_proxy_can_still_materialise(self, people_parquet):
        # The safety net: a query that leaves the DuckDB path must still get
        # correct answers, even though it costs the memory streaming saved.
        context = _stream(people_parquet, id_col="pid")
        frame = context.entity_mapping["Person"].source_obj.to_pandas()
        assert sorted(frame["name"]) == ["Alice", "Bob", "Carol", "Dan"]

    def test_pandas_backend_falls_back_to_an_eager_read(self, people_parquet):
        # streaming=True is a memory optimisation, not a semantic change, so
        # asking for it without DuckDB loads the source rather than failing.
        context = (
            ContextBuilder()
            .add_entity("Person", people_parquet, id_col="pid", streaming=True)
            .build(backend="pandas")
        )
        assert not is_streaming_source(
            context.entity_mapping["Person"].source_obj
        )
        assert run_query(context, "MATCH (p:Person) RETURN count(p)") == [[4]]

    def test_streaming_implies_a_file_backed_database(self, people_parquet):
        # Landing a streamed source in an in-memory database would put it
        # straight back on the heap.
        context = _stream(people_parquet, id_col="pid")
        assert context.backend.database_path is not None


class TestMixedSources:
    def test_streaming_and_eager_entities_coexist(self, people_parquet):
        widgets = pd.DataFrame({"sku": ["a", "b"]})
        context = (
            ContextBuilder()
            .add_entity("Person", people_parquet, id_col="pid", streaming=True)
            .add_entity("Widget", widgets)
            .build(backend="duckdb")
        )
        assert run_query(context, "MATCH (p:Person) RETURN count(p)") == [[4]]
        assert run_query(context, "MATCH (w:Widget) RETURN count(w)") == [[2]]
        assert is_streaming_source(
            context.entity_mapping["Person"].source_obj
        )
        assert not is_streaming_source(
            context.entity_mapping["Widget"].source_obj
        )

    def test_relationships_still_work_against_a_streamed_entity(
        self, people_parquet
    ):
        edges = pd.DataFrame({"__SOURCE__": [10, 20], "__TARGET__": [20, 30]})
        context = (
            ContextBuilder()
            .add_entity("Person", people_parquet, id_col="pid", streaming=True)
            .add_relationship(
                "KNOWS", edges, source_col="__SOURCE__", target_col="__TARGET__"
            )
            .build(backend="duckdb")
        )
        assert run_query(
            context,
            "MATCH (p:Person)-[:KNOWS]->(q:Person) "
            "RETURN p.name, q.name ORDER BY p.name",
        ) == [["Alice", "Bob"], ["Bob", "Carol"]]
