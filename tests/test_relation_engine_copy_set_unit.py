"""Phase 2b-ii (docs/fastopendata_streaming_qualification_plan.md) —
relationship property-copy ``SET`` mutation eligibility (Phase 2 long-tail
category (B)).

Closes 9 of the pipeline's 34 long-tail queries: a single-hop
``MATCH (source)-[:REL]->(target) [WITH target, source.prop AS a] SET
target.x = a`` (or the no-``WITH`` variant, ``SET target.x = source.prop``
directly) previously fell back to pandas because neither the plain
``"set"`` kind (single-node only) nor ``"group_set"`` (requires an
aggregate) covers it.

Two properties matter, mirroring the rest of Phase 2:

* **Correctness for the well-formed (1:1) case** — every real query of this
  shape in the pipeline has exactly one matching source row per target, and
  for that case there is only one possible answer regardless of tie-break
  rule.
* **Documented (not silent) divergence in the multi-match case** — verified
  empirically that the pandas engine's own tie-break is an undocumented
  implementation detail (last row by the source entity table's own row
  order), not a deliberate spec; this engine instead picks one row
  deterministically via ``QUALIFY ROW_NUMBER() ... = 1``, tie-broken by the
  matched variables' id columns. ``TestMultiMatchDivergence`` documents this
  explicitly rather than asserting parity in the case that never occurs in
  real pipeline data.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.backends.table_registry import physical_table_name
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    execute_relation_copy_set,
    is_relation_copy_set_eligible,
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    EntityTable,
    RelationshipMapping,
    RelationshipTable,
)
from pycypher.star import Star

ID_COLUMN = "__ID__"


def _ast(query: str):
    return ASTConverter.from_cypher(query)


@pytest.fixture
def result_parquet(tmp_path):
    path = tmp_path / "results.parquet"
    pd.DataFrame(
        {"eid": ["e1", "e2"], "rep_votes": [100, 200]},
    ).to_parquet(path)
    return path


@pytest.fixture
def county_parquet(tmp_path):
    path = tmp_path / "counties.parquet"
    pd.DataFrame(
        {
            "GEOID": ["c1", "c2"],
            "pres_rep_votes_2024": [None, None],
            "rucc_2023": [None, None],
        },
    ).to_parquet(path)
    return path


@pytest.fixture
def located_in_parquet(tmp_path):
    path = tmp_path / "located_in.parquet"
    pd.DataFrame({"src": ["e1", "e2"], "tgt": ["c1", "c2"]}).to_parquet(path)
    return path


def _streaming_ctx() -> Context:
    ctx = Context(
        entity_mapping=EntityMapping(mapping={}),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend="duckdb",
    )
    ctx._relation_engine_enabled = True
    return ctx


def _streamed_context(
    result_parquet, county_parquet, located_in_parquet
) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx,
        "CountyElectionResult",
        data_source_from_uri(str(result_parquet)),
        id_col="eid",
    )
    register_streaming_source(
        ctx,
        "County",
        data_source_from_uri(str(county_parquet)),
        id_col="GEOID",
    )
    register_streaming_relationship(
        ctx,
        "LOCATED_IN",
        data_source_from_uri(str(located_in_parquet)),
        source_col="src",
        target_col="tgt",
    )
    return ctx


def _pandas_ctx() -> Context:
    results = pd.DataFrame({ID_COLUMN: ["e1", "e2"], "rep_votes": [100, 200]})
    counties = pd.DataFrame(
        {
            ID_COLUMN: ["c1", "c2"],
            "pres_rep_votes_2024": [None, None],
            "rucc_2023": [None, None],
        },
    )
    located_in = pd.DataFrame(
        {
            "__ID__": [1, 2],
            "__SOURCE__": ["e1", "e2"],
            "__TARGET__": ["c1", "c2"],
        },
    )
    ctx = Context(
        entity_mapping=EntityMapping(
            mapping={
                "CountyElectionResult": EntityTable.from_dataframe(
                    "CountyElectionResult", results
                ),
                "County": EntityTable.from_dataframe("County", counties),
            },
        ),
        relationship_mapping=RelationshipMapping(
            mapping={
                "LOCATED_IN": RelationshipTable.from_dataframe(
                    "LOCATED_IN", located_in
                ),
            },
        ),
        backend="pandas",
    )
    ctx._relation_engine_enabled = True
    return ctx


_WITH_QUERY = (
    "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
    "WITH c, e.rep_votes AS rep_votes "
    "SET c.pres_rep_votes_2024 = rep_votes"
)

_NO_WITH_QUERY = (
    "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
    "SET c.pres_rep_votes_2024 = e.rep_votes"
)


class TestEligibility:
    def test_eligible_with_intervening_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert is_relation_copy_set_eligible(_ast(_WITH_QUERY), ctx)

    def test_eligible_no_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert is_relation_copy_set_eligible(_ast(_NO_WITH_QUERY), ctx)

    def test_eligible_multiple_set_items_no_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert is_relation_copy_set_eligible(
            _ast(
                "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "SET c.pres_rep_votes_2024 = e.rep_votes, c.rucc_2023 = e.rep_votes",
            ),
            ctx,
        )

    def test_ineligible_single_node_no_relationship(
        self, county_parquet, result_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert not is_relation_copy_set_eligible(
            _ast("MATCH (c:County) SET c.pres_rep_votes_2024 = 1"),
            ctx,
        )

    def test_ineligible_aggregate_in_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        # An aggregate here is _analyze_group_set_query's shape, not this one.
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert not is_relation_copy_set_eligible(
            _ast(
                "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "WITH c, COUNT(e) AS cnt "
                "SET c.pres_rep_votes_2024 = cnt",
            ),
            ctx,
        )

    def test_ineligible_optional_match(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert not is_relation_copy_set_eligible(
            _ast(
                "OPTIONAL MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "SET c.pres_rep_votes_2024 = e.rep_votes",
            ),
            ctx,
        )

    def test_ineligible_set_targets_two_different_variables(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert not is_relation_copy_set_eligible(
            _ast(
                "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "SET c.pres_rep_votes_2024 = e.rep_votes, e.rep_votes = c.pres_rep_votes_2024",
            ),
            ctx,
        )

    def test_ineligible_set_expression_not_bare_alias_with_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert not is_relation_copy_set_eligible(
            _ast(
                "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "WITH c, e.rep_votes AS rep_votes "
                "SET c.pres_rep_votes_2024 = rep_votes * 2",
            ),
            ctx,
        )

    def test_ineligible_no_streaming_source(
        self, result_parquet, located_in_parquet
    ) -> None:
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "CountyElectionResult",
            data_source_from_uri(str(result_parquet)),
            id_col="eid",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(located_in_parquet)),
            source_col="src",
            target_col="tgt",
        )
        counties = pd.DataFrame(
            {ID_COLUMN: ["c1", "c2"], "pres_rep_votes_2024": [None, None]}
        )
        ctx.entity_mapping.mapping["County"] = EntityTable.from_dataframe(
            "County", counties
        )
        assert not is_relation_copy_set_eligible(_ast(_NO_WITH_QUERY), ctx)

    def test_ineligible_pandas_backend(self) -> None:
        ctx = _pandas_ctx()
        assert not is_relation_copy_set_eligible(_ast(_NO_WITH_QUERY), ctx)


class TestExecution:
    def test_execute_with_intervening_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        query = _ast(_WITH_QUERY)
        assert is_relation_copy_set_eligible(query, ctx)
        execute_relation_copy_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", pres_rep_votes_2024 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["GEOID"], got["pres_rep_votes_2024"]))
        assert rows == {"c1": 100, "c2": 200}

    def test_execute_no_with(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        query = _ast(_NO_WITH_QUERY)
        assert is_relation_copy_set_eligible(query, ctx)
        execute_relation_copy_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", pres_rep_votes_2024 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["GEOID"], got["pres_rep_votes_2024"]))
        assert rows == {"c1": 100, "c2": 200}

    def test_execute_query_dispatches_through_pipeline(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        out = Star(context=ctx).execute_query(_WITH_QUERY)
        assert out.empty

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", pres_rep_votes_2024 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["GEOID"], got["pres_rep_votes_2024"]))
        assert rows == {"c1": 100, "c2": 200}

    def test_never_materialises_to_pandas(
        self, result_parquet, county_parquet, located_in_parquet, monkeypatch
    ) -> None:
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)

        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        query = _ast(_WITH_QUERY)
        assert is_relation_copy_set_eligible(query, ctx)
        execute_relation_copy_set(query, ctx)
        assert calls == []


class TestNewColumn:
    """Phase 3a (docs/fastopendata_streaming_qualification_plan.md) -- a
    SET target property that doesn't exist yet in the raw file is created
    via ALTER TABLE rather than making the whole query ineligible. Also
    closes a pre-existing bug found while building this: is_relation_
    copy_set_eligible never checked the target property resolved at all,
    so a query targeting an unseen property was wrongly reported eligible
    and would have raised a KeyError in execute_relation_copy_set.
    """

    def test_eligible_new_target_property(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        assert is_relation_copy_set_eligible(
            _ast(
                "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
                "SET c.pres_dem_votes_2024 = e.rep_votes",
            ),
            ctx,
        )

    def test_execute_creates_column_and_leaves_unmatched_rows_null(
        self, tmp_path
    ) -> None:
        results_path = tmp_path / "results.parquet"
        pd.DataFrame({"eid": ["e1"], "rep_votes": [100]}).to_parquet(
            results_path
        )
        county_path = tmp_path / "counties.parquet"
        # c2 has no matching CountyElectionResult -- the no-zero-fill case.
        pd.DataFrame({"GEOID": ["c1", "c2"]}).to_parquet(county_path)
        located_in_path = tmp_path / "located_in.parquet"
        pd.DataFrame({"src": ["e1"], "tgt": ["c1"]}).to_parquet(
            located_in_path
        )

        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "CountyElectionResult",
            data_source_from_uri(str(results_path)),
            id_col="eid",
        )
        register_streaming_source(
            ctx,
            "County",
            data_source_from_uri(str(county_path)),
            id_col="GEOID",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(located_in_path)),
            source_col="src",
            target_col="tgt",
        )
        query = _ast(
            "MATCH (e:CountyElectionResult)-[:LOCATED_IN]->(c:County) "
            "SET c.pres_dem_votes_2024 = e.rep_votes",
        )
        assert is_relation_copy_set_eligible(query, ctx)
        execute_relation_copy_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", pres_dem_votes_2024 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["GEOID"], got["pres_dem_votes_2024"]))
        assert rows["c1"] == 100
        assert pd.isna(rows["c2"])


class TestParityWithPandasEngine:
    def test_matches_pandas_engine_result_for_the_well_formed_1to1_case(
        self, result_parquet, county_parquet, located_in_parquet
    ) -> None:
        streamed = _streamed_context(
            result_parquet, county_parquet, located_in_parquet
        )
        Star(context=streamed).execute_query(_WITH_QUERY)
        table = physical_table_name("County")
        streamed_rows = streamed.backend.connection.execute(
            f'SELECT "GEOID", pres_rep_votes_2024 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()

        pandas_ctx = _pandas_ctx()
        Star(context=pandas_ctx).execute_query(_WITH_QUERY)
        pandas_out = Star(context=pandas_ctx).execute_query(
            "MATCH (c:County) RETURN id(c) AS geoid, "
            "c.pres_rep_votes_2024 AS pres_rep_votes_2024",
        )

        streamed_by_geoid = dict(
            zip(streamed_rows["GEOID"], streamed_rows["pres_rep_votes_2024"])
        )
        pandas_by_geoid = dict(
            zip(pandas_out["geoid"], pandas_out["pres_rep_votes_2024"])
        )
        assert streamed_by_geoid == pandas_by_geoid


class TestMultiMatchDivergence:
    """When a target matches more than one source row, the pandas engine
    picks a row via an undocumented, source-table-row-order-dependent
    tie-break (verified empirically — see the plan doc). This engine
    intentionally does not replicate that: it picks deterministically via
    ``QUALIFY ROW_NUMBER() ... = 1``, tie-broken by the matched variables'
    id columns. This test documents the divergence rather than asserting
    (false) parity — no real query in this pipeline has multi-match data.
    """

    def test_picks_one_row_deterministically_not_matching_pandas_tiebreak(
        self, tmp_path
    ) -> None:
        results_path = tmp_path / "results.parquet"
        pd.DataFrame(
            {"eid": ["e1", "e2"], "rep_votes": [100, 200]},
        ).to_parquet(results_path)
        county_path = tmp_path / "counties.parquet"
        pd.DataFrame(
            {"GEOID": ["c1"], "pres_rep_votes_2024": [None]},
        ).to_parquet(county_path)
        located_in_path = tmp_path / "located_in.parquet"
        # Both e1 and e2 match c1 -- the ambiguous case.
        pd.DataFrame({"src": ["e1", "e2"], "tgt": ["c1", "c1"]}).to_parquet(
            located_in_path
        )

        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "CountyElectionResult",
            data_source_from_uri(str(results_path)),
            id_col="eid",
        )
        register_streaming_source(
            ctx,
            "County",
            data_source_from_uri(str(county_path)),
            id_col="GEOID",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(located_in_path)),
            source_col="src",
            target_col="tgt",
        )
        query = _ast(_WITH_QUERY)
        assert is_relation_copy_set_eligible(query, ctx)
        execute_relation_copy_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT pres_rep_votes_2024 FROM "{table}"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        # e1 sorts before e2 by id, so the deterministic tie-break picks e1
        # (100) -- documented as the engine's own rule, not a pandas match.
        assert got["pres_rep_votes_2024"].iloc[0] == 100

        # Confirm this really does diverge from the pandas engine's own
        # (undocumented) tie-break for the same ambiguous data, rather than
        # coincidentally agreeing.
        pandas_results = pd.DataFrame(
            {ID_COLUMN: ["e1", "e2"], "rep_votes": [100, 200]}
        )
        pandas_counties = pd.DataFrame(
            {ID_COLUMN: ["c1"], "pres_rep_votes_2024": [None]}
        )
        pandas_located_in = pd.DataFrame(
            {
                "__ID__": [1, 2],
                "__SOURCE__": ["e1", "e2"],
                "__TARGET__": ["c1", "c1"],
            },
        )
        pandas_ctx = Context(
            entity_mapping=EntityMapping(
                mapping={
                    "CountyElectionResult": EntityTable.from_dataframe(
                        "CountyElectionResult", pandas_results
                    ),
                    "County": EntityTable.from_dataframe(
                        "County", pandas_counties
                    ),
                },
            ),
            relationship_mapping=RelationshipMapping(
                mapping={
                    "LOCATED_IN": RelationshipTable.from_dataframe(
                        "LOCATED_IN", pandas_located_in
                    ),
                },
            ),
            backend="pandas",
        )
        Star(context=pandas_ctx).execute_query(_WITH_QUERY)
        pandas_out = Star(context=pandas_ctx).execute_query(
            "MATCH (c:County) RETURN c.pres_rep_votes_2024 AS rep_votes",
        )
        assert pandas_out["rep_votes"].iloc[0] == 200
