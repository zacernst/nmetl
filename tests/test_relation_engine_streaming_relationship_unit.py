"""Streaming relationship registration — the relationship counterpart to
``test_relation_streaming_e2e.py``.

Closes Gap 1 of ``docs/fastopendata_streaming_qualification_plan.md``
("relationship sources are never registered for streaming"): before
``register_streaming_relationship``, no relationship source could ever
qualify for ``_try_streaming_run()`` — a pattern traversing any relationship
type fell back to the eager pandas path unconditionally, regardless of the
query's own shape.

Two properties matter, mirroring ``test_streaming_entity.py``'s split:

* **Equivalence** — a streaming relationship must answer exactly as an
  eagerly loaded one, including id assignment, ``(source, target)``
  de-duplication, and ``allow_multi_edges``.
* **Non-materialisation** — the whole point is that the source is never
  pulled into pandas/Arrow; correctness alone cannot detect a regression
  here, since a silent eager fallback would still be correct, just
  unbounded.

The priority stress case named in the plan — ``osm_node_in_tract``'s shape
(a large file, no ``id_col``, no ``query:``) — is covered by
``TestLargeUncappedSource``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import run_query
from pycypher.backends.table_registry import RELATIONSHIP_KIND
from pycypher.ingestion.context_builder import ContextBuilder
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    is_relation_eligible,
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)
from pycypher.star import Star


def _streaming_ctx() -> Context:
    ctx = Context(
        entity_mapping=EntityMapping(mapping={}),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend="duckdb",
    )
    ctx._relation_engine_enabled = True
    return ctx


@pytest.fixture
def people_parquet(tmp_path):
    path = tmp_path / "people.parquet"
    pd.DataFrame(
        {"pid": [10, 20, 30], "name": ["Alice", "Bob", "Carol"]},
    ).to_parquet(path)
    return path


@pytest.fixture
def knows_parquet(tmp_path):
    path = tmp_path / "knows.parquet"
    pd.DataFrame(
        {"src": [10, 20], "tgt": [20, 30], "since": [2020, 2021]},
    ).to_parquet(path)
    return path


@pytest.fixture
def knows_dupes_parquet(tmp_path):
    # Two rows share the (src, tgt) pair (10 -> 20) — the second should be
    # collapsed away by default, or both kept with allow_multi_edges=True.
    path = tmp_path / "knows_dupes.parquet"
    pd.DataFrame(
        {
            "src": [10, 10, 20],
            "tgt": [20, 20, 30],
            "since": [2020, 1999, 2021],
        },
    ).to_parquet(path)
    return path


def _streamed_context(people_path, knows_path, **rel_kwargs) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx, "Person", data_source_from_uri(str(people_path)), id_col="pid"
    )
    register_streaming_relationship(
        ctx,
        "KNOWS",
        data_source_from_uri(str(knows_path)),
        source_col="src",
        target_col="tgt",
        **rel_kwargs,
    )
    return ctx


def _eager_context(people_path, knows_path, **rel_kwargs) -> Context:
    return (
        ContextBuilder()
        .add_entity("Person", str(people_path), id_col="pid")
        .add_relationship(
            "KNOWS",
            str(knows_path),
            source_col="src",
            target_col="tgt",
            **rel_kwargs,
        )
        .build(backend="pandas")
    )


CYPHER = (
    "MATCH (p:Person)-[k:KNOWS]->(q:Person) "
    "RETURN p.name, q.name, k.since ORDER BY p.name"
)


class TestEquivalence:
    def test_matches_the_eager_path(self, people_parquet, knows_parquet):
        streamed = _streamed_context(people_parquet, knows_parquet)
        eager = _eager_context(people_parquet, knows_parquet)
        assert run_query(streamed, CYPHER) == run_query(eager, CYPHER)
        assert run_query(streamed, CYPHER) == [
            ["Alice", "Bob", 2020],
            ["Bob", "Carol", 2021],
        ]

    def test_matches_the_eager_path_with_explicit_id_col(
        self, tmp_path, people_parquet
    ):
        path = tmp_path / "knows_id.parquet"
        pd.DataFrame(
            {
                "eid": ["e1", "e2"],
                "src": [10, 20],
                "tgt": [20, 30],
                "since": [2020, 2021],
            },
        ).to_parquet(path)
        streamed = _streamed_context(people_parquet, path, id_col="eid")
        eager = _eager_context(people_parquet, path, id_col="eid")
        assert run_query(streamed, CYPHER) == run_query(eager, CYPHER)

    def test_duplicate_endpoints_collapse_by_default(
        self, people_parquet, knows_dupes_parquet
    ):
        streamed = _streamed_context(people_parquet, knows_dupes_parquet)
        eager = _eager_context(people_parquet, knows_dupes_parquet)
        assert run_query(streamed, CYPHER) == run_query(eager, CYPHER)
        # First occurrence (since=2020) survives, not the second (since=1999).
        assert run_query(streamed, CYPHER) == [
            ["Alice", "Bob", 2020],
            ["Bob", "Carol", 2021],
        ]

    def test_allow_multi_edges_preserves_parallel_edges(
        self, people_parquet, knows_dupes_parquet
    ):
        streamed = _streamed_context(
            people_parquet, knows_dupes_parquet, allow_multi_edges=True
        )
        eager = _eager_context(
            people_parquet, knows_dupes_parquet, allow_multi_edges=True
        )
        assert sorted(run_query(streamed, CYPHER)) == sorted(
            run_query(eager, CYPHER)
        )
        assert len(run_query(streamed, CYPHER)) == 3

    def test_missing_source_col_raises_like_the_eager_path(
        self, knows_parquet
    ):
        with pytest.raises(ValueError, match="source_col"):
            register_streaming_relationship(
                _streaming_ctx(),
                "KNOWS",
                data_source_from_uri(str(knows_parquet)),
                source_col="nope",
                target_col="tgt",
            )


class TestRegistryIntegration:
    def test_relationship_is_eligible_for_the_relation_engine(
        self, people_parquet, knows_parquet
    ):
        ctx = _streamed_context(people_parquet, knows_parquet)
        from pycypher.ast_converter import ASTConverter

        assert is_relation_eligible(
            ASTConverter.from_cypher(CYPHER),
            ctx,
        )

    def test_registered_under_relationship_kind_not_entity_kind(
        self, people_parquet, knows_parquet
    ):
        ctx = _streamed_context(people_parquet, knows_parquet)
        assert ctx.backend.tables.has("KNOWS", RELATIONSHIP_KIND)
        assert not ctx.backend.tables.has("KNOWS")  # default kind is entity

    def test_source_and_target_are_not_exposed_as_properties(
        self, people_parquet, knows_parquet
    ):
        ctx = _streamed_context(people_parquet, knows_parquet)
        entry = ctx.backend.tables.get("KNOWS", RELATIONSHIP_KIND)
        assert "__SOURCE__" not in entry.attr_map
        assert "__TARGET__" not in entry.attr_map
        assert entry.attr_map == {"since": "since"}


class TestSourceIsNotMaterialised:
    """Correctness cannot detect a regression here — the proxy hides it."""

    def test_registration_never_touches_pandas(
        self, people_parquet, knows_parquet, monkeypatch
    ):
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)
        _streamed_context(people_parquet, knows_parquet)
        assert calls == []

    def test_query_never_touches_pandas(
        self, people_parquet, knows_parquet, monkeypatch
    ):
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)
        ctx = _streamed_context(people_parquet, knows_parquet)
        Star(context=ctx).execute_query(CYPHER)
        assert calls == []


class TestLargeUncappedSource:
    """Priority stress case from the plan: an ``osm_node_in_tract``-shaped
    source — large, no ``id_col``, no ``query:`` (plain ``SELECT *``)."""

    @pytest.fixture
    def large_nodes(self, tmp_path):
        n = 50_000
        path = tmp_path / "nodes.parquet"
        pd.DataFrame(
            {"pid": range(n), "name": [f"n{i}" for i in range(n)]}
        ).to_parquet(path)
        return path

    @pytest.fixture
    def large_crosswalk(self, tmp_path):
        # No id_col, no query — mirrors osm_node_in_tract exactly: bare
        # source_col/target_col over an otherwise-unprojected file.
        n = 50_000
        path = tmp_path / "crosswalk.parquet"
        pd.DataFrame(
            {"id": range(n), "GEOID": [str(i % 500) for i in range(n)]},
        ).to_parquet(path)
        return path

    @pytest.fixture
    def geo_areas(self, tmp_path):
        # The target entity — mirrors Tract, keyed on a non-"__ID__"-named
        # column (geo_id, string-typed to match the crosswalk's GEOID
        # target column), which is the case the join-column fix exists
        # for. region_name stays a plain (queryable) property, distinct
        # from the id column — register_streaming_source has a separate,
        # pre-existing limitation with id_col=None (no __ID__ is
        # auto-generated the way the eager path's normalize_entity_table
        # does), so this deliberately avoids that case rather than also
        # exercising it here.
        path = tmp_path / "geo_areas.parquet"
        pd.DataFrame(
            {
                "geo_id": [str(i) for i in range(500)],
                "region_name": [f"region-{i}" for i in range(500)],
            },
        ).to_parquet(path)
        return path

    def test_registers_and_answers_correctly(
        self, large_nodes, large_crosswalk, geo_areas
    ):
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx, "Node", data_source_from_uri(str(large_nodes)), id_col="pid"
        )
        register_streaming_source(
            ctx,
            "GeoArea",
            data_source_from_uri(str(geo_areas)),
            id_col="geo_id",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(large_crosswalk)),
            source_col="id",
            target_col="GEOID",
        )
        out = run_query(
            ctx,
            "MATCH (n:Node)-[:LOCATED_IN]->(g:GeoArea) "
            "WHERE n.name = 'n42' RETURN g.region_name",
        )
        assert out == [["region-42"]]

    def test_never_materialises_the_50k_row_source(
        self, large_nodes, large_crosswalk, geo_areas, monkeypatch
    ):
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx, "Node", data_source_from_uri(str(large_nodes)), id_col="pid"
        )
        register_streaming_source(
            ctx,
            "GeoArea",
            data_source_from_uri(str(geo_areas)),
            id_col="geo_id",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(large_crosswalk)),
            source_col="id",
            target_col="GEOID",
        )
        # WITH must project the grouping key as a property (not a bare node
        # pass-through) to stay relation-eligible today — grouping by a raw
        # node variable is a separate, pre-existing relation-engine gap
        # unrelated to streaming relationships (also relevant to Phase 2 of
        # docs/fastopendata_streaming_qualification_plan.md, since the real
        # pipeline's aggregate-then-SET queries group by a bare node too).
        run_query(
            ctx,
            "MATCH (n:Node)-[:LOCATED_IN]->(g:GeoArea) "
            "WITH g.region_name AS region, COUNT(n) AS cnt "
            "RETURN region, cnt ORDER BY region LIMIT 3",
        )
        assert calls == []
