"""Streaming relationship registration — the relationship counterpart to
``test_relation_streaming_e2e.py``.

Closes Gap 1 of the FastOpenData streaming-qualification plan (private repository)
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
        # the FastOpenData streaming-qualification plan (private repository), since the real
        # pipeline's aggregate-then-SET queries group by a bare node too).
        run_query(
            ctx,
            "MATCH (n:Node)-[:LOCATED_IN]->(g:GeoArea) "
            "WITH g.region_name AS region, COUNT(n) AS cnt "
            "RETURN region, cnt ORDER BY region LIMIT 3",
        )
        assert calls == []


# ---------------------------------------------------------------------------
# Declared endpoint labels (source_entity_type / target_entity_type)
# ---------------------------------------------------------------------------


@pytest.fixture
def overlapping_survey_files(tmp_path):
    """Two entity types sharing id values, each with its own edge file --
    the fastopendata 1-year/5-year survey shape. Every 1-year unit id is
    also a 5-year unit id, and both edge files point at the same PUMAs.
    """
    pd.DataFrame({"pfips": ["p1", "p2", "p9"]}).to_parquet(
        tmp_path / "puma.parquet"
    )
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3"], "income": [10, 20, 30]}
    ).to_parquet(tmp_path / "units_1yr.parquet")
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3", "u4"], "income": [11, 21, 31, 41]}
    ).to_parquet(tmp_path / "units_5yr.parquet")
    # 1yr: u1,u2 -> p1; u3 -> p2.   5yr: same plus u4 -> p2.
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3"], "pfips": ["p1", "p1", "p2"]}
    ).to_parquet(tmp_path / "edges_1yr.parquet")
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3", "u4"], "pfips": ["p1", "p1", "p2", "p2"]}
    ).to_parquet(tmp_path / "edges_5yr.parquet")
    return tmp_path


def _survey_context(d, *, declare_1yr=True, declare_5yr=True) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx,
        "PUMA",
        data_source_from_uri(str(d / "puma.parquet")),
        id_col="pfips",
    )
    register_streaming_source(
        ctx,
        "Unit1yr",
        data_source_from_uri(str(d / "units_1yr.parquet")),
        id_col="serial",
    )
    register_streaming_source(
        ctx,
        "Unit5yr",
        data_source_from_uri(str(d / "units_5yr.parquet")),
        id_col="serial",
    )
    for name, declare in (("1yr", declare_1yr), ("5yr", declare_5yr)):
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(d / f"edges_{name}.parquet")),
            source_col="serial",
            target_col="pfips",
            source_entity_type=f"Unit{name}" if declare else None,
            target_entity_type="PUMA" if declare else None,
        )
    return ctx


def _counts(ctx, cypher) -> dict:
    # Guard against silently testing the pandas fallback, which does not
    # honour endpoint labels.
    from pycypher.ast_converter import ASTConverter

    assert is_relation_eligible(ASTConverter.from_cypher(cypher), ctx)
    return {row[0]: row[1] for row in run_query(ctx, cypher)}


def _eligible_rows(ctx, cypher) -> list:
    from pycypher.ast_converter import ASTConverter

    assert is_relation_eligible(ASTConverter.from_cypher(cypher), ctx)
    return run_query(ctx, cypher)


class TestDeclaredEndpointLabels:
    """A relationship source may declare which entity labels its endpoints
    belong to. Node identity in this engine is the id *value*, so without
    the declaration an edge registered for one entity type is traversed
    from any other type sharing that id -- which doubled every 1-year
    survey aggregate in the real pipeline (2026-09-05).
    """

    COUNT_1YR = (
        "MATCH (pu:PUMA)<-[:LOCATED_IN]-(h:Unit1yr) "
        "WITH id(pu) AS puma, COUNT(h) AS n RETURN puma, n ORDER BY puma"
    )
    COUNT_5YR = (
        "MATCH (h:Unit5yr)-[:LOCATED_IN]->(pu:PUMA) "
        "WITH id(pu) AS puma, COUNT(h) AS n RETURN puma, n ORDER BY puma"
    )
    SUM_1YR = (
        "MATCH (pu:PUMA)<-[:LOCATED_IN]-(h:Unit1yr) "
        "WITH id(pu) AS puma, SUM(h.income) AS s RETURN puma, s ORDER BY puma"
    )

    def test_declared_labels_are_reserved_columns_not_properties(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        entry = ctx.backend.tables.get("LOCATED_IN", RELATIONSHIP_KIND)
        assert {"__SOURCE_LABEL__", "__TARGET_LABEL__"} <= set(entry.columns)
        assert "__SOURCE_LABEL__" not in entry.attr_map
        assert "__TARGET_LABEL__" not in entry.attr_map
        labels = set(
            zip(
                entry.relation.fetchdf()["__SOURCE_LABEL__"],
                entry.relation.fetchdf()["__TARGET_LABEL__"],
            )
        )
        assert labels == {("Unit1yr", "PUMA"), ("Unit5yr", "PUMA")}

    def test_undeclared_source_records_null_labels(
        self, people_parquet, knows_parquet
    ):
        ctx = _streamed_context(people_parquet, knows_parquet)
        got = ctx.backend.tables.get(
            "KNOWS", RELATIONSHIP_KIND
        ).relation.fetchdf()
        assert got["__SOURCE_LABEL__"].isna().all()
        assert got["__TARGET_LABEL__"].isna().all()

    def test_overlapping_ids_no_longer_double_count(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        assert _counts(ctx, self.COUNT_1YR) == {"p1": 2, "p2": 1}
        assert _counts(ctx, self.COUNT_5YR) == {"p1": 2, "p2": 2}
        assert _counts(ctx, self.SUM_1YR) == {"p1": 30, "p2": 30}

    def test_without_declarations_the_old_doubling_is_reproduced(
        self, overlapping_survey_files
    ):
        # Documents the semantics this feature exists to fix: with no
        # declared endpoint labels, u1/u2/u3 traverse both edge sets.
        ctx = _survey_context(
            overlapping_survey_files, declare_1yr=False, declare_5yr=False
        )
        assert _counts(ctx, self.COUNT_1YR) == {"p1": 4, "p2": 2}

    def test_undeclared_source_still_matches_any_label(
        self, overlapping_survey_files
    ):
        # Only the 5yr edges are declared: 1yr units see their own
        # (undeclared, wildcard) edges but not the 5yr ones; 5yr units see
        # both, since the undeclared edges match any label.
        ctx = _survey_context(overlapping_survey_files, declare_1yr=False)
        assert _counts(ctx, self.COUNT_1YR) == {"p1": 2, "p2": 1}
        assert _counts(ctx, self.COUNT_5YR) == {"p1": 4, "p2": 3}

    def test_relationship_variable_and_multi_hop_paths(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        rows = _eligible_rows(
            ctx,
            "MATCH (h:Unit1yr)-[e:LOCATED_IN]->(pu:PUMA) "
            "RETURN id(h) AS unit, id(pu) AS puma ORDER BY unit",
        )
        assert rows == [["u1", "p1"], ["u2", "p1"], ["u3", "p2"]]
        # Two hops through the same PUMA: 1yr unit -> PUMA <- 5yr unit.
        rows = _eligible_rows(
            ctx,
            "MATCH (a:Unit1yr)-[:LOCATED_IN]->(pu:PUMA)<-[:LOCATED_IN]-(b:Unit5yr) "
            "WITH id(pu) AS puma, COUNT(*) AS pairs RETURN puma, pairs ORDER BY puma",
        )
        assert rows == [["p1", 4], ["p2", 2]]

    def test_optional_match_filters_in_the_join_condition(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        rows = _eligible_rows(
            ctx,
            "MATCH (pu:PUMA) OPTIONAL MATCH (pu)<-[:LOCATED_IN]-(h:Unit1yr) "
            "RETURN id(pu) AS puma, id(h) AS unit ORDER BY puma, unit",
        )
        # p9 has no units and must survive the LEFT join as a NULL row;
        # p1/p2 must not pick up the 5yr edges.
        rows = [[None if pd.isna(v) else v for v in row] for row in rows]
        assert rows == [
            ["p1", "u1"],
            ["p1", "u2"],
            ["p2", "u3"],
            ["p9", None],
        ]

    def test_second_match_after_with_respects_labels(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        rows = _eligible_rows(
            ctx,
            "MATCH (pu:PUMA) WITH id(pu) AS puma "
            "MATCH (h:Unit1yr)-[:LOCATED_IN]->(q:PUMA) "
            "WHERE id(q) = puma "
            "WITH puma, COUNT(h) AS n RETURN puma, n ORDER BY puma",
        )
        assert rows == [["p1", 2], ["p2", 1]]

    def test_label_mismatch_yields_no_rows_not_an_error(
        self, overlapping_survey_files
    ):
        ctx = _survey_context(overlapping_survey_files)
        # Units are never LOCATED_IN other units; both endpoints are declared
        # as something else, so the join is empty rather than wrong.
        rows = _eligible_rows(
            ctx,
            "MATCH (a:Unit1yr)-[:LOCATED_IN]->(b:Unit5yr) RETURN count(*) AS n",
        )
        assert rows == [[0]]

    def test_label_literals_are_escaped(self):
        from pycypher.relation_engine import _relationship_streaming_sql

        sql = _relationship_streaming_sql(
            ["s", "t"],
            source_col="s",
            target_col="t",
            id_col=None,
            allow_multi_edges=False,
            view_name="v",
            source_label="Bad'Label",
            target_label=None,
        )
        assert "'Bad''Label' AS __SOURCE_LABEL__" in sql
        assert "CAST(NULL AS VARCHAR) AS __TARGET_LABEL__" in sql

    def test_never_touches_pandas(self, overlapping_survey_files, monkeypatch):
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)
        ctx = _survey_context(overlapping_survey_files)
        from pycypher.ast_converter import ASTConverter
        from pycypher.relation_engine import execute_relation_query

        ast = ASTConverter.from_cypher(self.COUNT_1YR)
        assert is_relation_eligible(ast, ctx)
        execute_relation_query(ast, ctx, materialize=False)
        assert calls == []
