"""Phase 2 first slice (the FastOpenData streaming-qualification plan (private repository)) —
aggregate-then-``SET`` mutation eligibility.

Closes Gap 2 of the plan: a single-hop ``MATCH (source)-[:REL]->(target)
WITH target, COUNT(source) AS cnt[, ...] SET target.prop = cnt[, ...]``
previously fell back to the pandas engine unconditionally (a bare-node
``WITH`` item mixed with an aggregate is not handled by the general
``_plan_stage`` read path), regardless of the streaming eligibility of every
other query in the pipeline — for the fastopendata pipeline this shape is
the single largest source of pandas materialisation on the biggest tables.

Two properties matter, mirroring the SET/relationship streaming test split:

* **Correctness** — matches the pandas engine's own zero-match-rows
  semantics: a target with no matching source rows is left untouched, not
  zero-filled (verified empirically against the real pandas engine before
  Phase 2 was implemented; see the plan doc's "Open questions" section).
* **Non-materialisation** — the aggregate compiles to a native
  ``UPDATE ... FROM`` over a DuckDB view; the source is never pulled into
  pandas.

The 8-query long tail excluded from this slice (multi-hop paths combined
with a second required MATCH, ``WITH DISTINCT``) was deferred to Phase 2b
and simply falls back to the pandas engine, covered here by the
``TestEligibility`` negative cases. Phase 2b category (D) -- a ``SET`` value
that is a scalar expression over several aggregate aliases
(``toFloat(a1) / a2``), with ``CASE`` inside an aggregate argument -- was
the last piece of that long tail and is covered by ``TestSetExpression``
(closed 2026-09-05).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.backends.table_registry import physical_table_name
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    execute_relation_group_set,
    is_relation_group_set_eligible,
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
def household_parquet(tmp_path):
    path = tmp_path / "households.parquet"
    pd.DataFrame(
        {"hh_id": ["hh1", "hh2", "hh3"], "income": [50000, 70000, 90000]},
    ).to_parquet(path)
    return path


@pytest.fixture
def puma_parquet(tmp_path):
    path = tmp_path / "pumas.parquet"
    pd.DataFrame(
        {
            "PUMA_FIPS": ["P1", "P2", "P3"],
            "region": ["P1", "P2", "P3"],
            "household_count": [None, None, 999],
            "total_income": [None, None, 999.0],
        },
    ).to_parquet(path)
    return path


@pytest.fixture
def located_in_parquet(tmp_path):
    # hh1 and hh2 -> P1, hh3 -> P2, P3 has zero matches (the no-zero-fill case).
    path = tmp_path / "located_in.parquet"
    pd.DataFrame(
        {"src": ["hh1", "hh2", "hh3"], "tgt": ["P1", "P1", "P2"]}
    ).to_parquet(path)
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
    household_parquet, puma_parquet, located_in_parquet
) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx,
        "Household",
        data_source_from_uri(str(household_parquet)),
        id_col="hh_id",
    )
    register_streaming_source(
        ctx,
        "PUMA",
        data_source_from_uri(str(puma_parquet)),
        id_col="PUMA_FIPS",
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
    households = pd.DataFrame(
        {ID_COLUMN: ["hh1", "hh2", "hh3"], "income": [50000, 70000, 90000]},
    )
    pumas = pd.DataFrame(
        {
            ID_COLUMN: ["P1", "P2", "P3"],
            "region": ["P1", "P2", "P3"],
            "household_count": [None, None, 999],
            "total_income": [None, None, 999.0],
        },
    )
    located_in = pd.DataFrame(
        {
            "__ID__": [1, 2, 3],
            "__SOURCE__": ["hh1", "hh2", "hh3"],
            "__TARGET__": ["P1", "P1", "P2"],
        },
    )
    ctx = Context(
        entity_mapping=EntityMapping(
            mapping={
                "Household": EntityTable.from_dataframe(
                    "Household", households
                ),
                "PUMA": EntityTable.from_dataframe("PUMA", pumas),
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


_GROUP_SET_QUERY = (
    "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
    "WITH p, COUNT(h) AS cnt, SUM(h.income) AS total "
    "SET p.household_count = cnt, p.total_income = total"
)


class TestEligibility:
    def test_eligible_group_set(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(_ast(_GROUP_SET_QUERY), ctx)

    def test_eligible_single_aggregate(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET p.household_count = cnt",
            ),
            ctx,
        )

    def test_ineligible_optional_match(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert not is_relation_group_set_eligible(
            _ast(
                "OPTIONAL MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET p.household_count = cnt",
            ),
            ctx,
        )

    def test_eligible_distinct_with_aggregate_is_a_redundant_noop(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        # Phase 2b category (C): DISTINCT after a GROUP BY-shaped aggregate
        # is a semantic no-op (grouping already yields one row per group).
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH DISTINCT p, COUNT(h) AS cnt "
                "SET p.household_count = cnt",
            ),
            ctx,
        )

    def test_eligible_set_expression_combines_aliases(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        # Phase 2b category (D): was the first-slice negative case; now
        # eligible. See TestSetExpression for execution semantics.
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt, SUM(h.income) AS total "
                "SET p.household_count = cnt + total",
            ),
            ctx,
        )

    def test_ineligible_set_expression_references_non_alias(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        # A SET value may only reach the aggregate row: a pattern variable
        # (h) or an undefined name is not an alias and must not compile.
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert not is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET p.household_count = cnt + h.income",
            ),
            ctx,
        )
        assert not is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET p.household_count = cnt + nope",
            ),
            ctx,
        )

    def test_ineligible_set_targets_other_variable(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert not is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET h.household_count = cnt",
            ),
            ctx,
        )

    def test_eligible_aggregate_over_single_node(self, puma_parquet) -> None:
        # Plan translator: grouping by a bare node is a Project with a passthrough item; no relationship needed.
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "PUMA",
            data_source_from_uri(str(puma_parquet)),
            id_col="PUMA_FIPS",
        )
        assert is_relation_group_set_eligible(
            _ast(
                "MATCH (p:PUMA) WITH p, COUNT(p) AS cnt SET p.household_count = cnt"
            ),
            ctx,
        )

    def test_eligible_group_var_in_memory_source_is_materialised(
        self, household_parquet, located_in_parquet
    ) -> None:
        # Plan catalog: an in-memory-only label is materialised into a registry table on first use, so it is writable.
        # PUMA only exists via the in-memory entity_mapping fallback, not a
        # registered streaming source — no writable table to UPDATE.
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "Household",
            data_source_from_uri(str(household_parquet)),
            id_col="hh_id",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(located_in_parquet)),
            source_col="src",
            target_col="tgt",
        )
        pumas = pd.DataFrame(
            {ID_COLUMN: ["P1", "P2"], "household_count": [None, None]}
        )
        ctx.entity_mapping.mapping["PUMA"] = EntityTable.from_dataframe(
            "PUMA", pumas
        )
        assert is_relation_group_set_eligible(_ast(_GROUP_SET_QUERY), ctx)

    def test_ineligible_pandas_backend(self) -> None:
        ctx = _pandas_ctx()
        assert not is_relation_group_set_eligible(_ast(_GROUP_SET_QUERY), ctx)


class TestExecution:
    def test_execute_updates_matched_pumas_and_leaves_unmatched_untouched(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        query = _ast(_GROUP_SET_QUERY)
        assert is_relation_group_set_eligible(query, ctx)
        execute_relation_group_set(query, ctx)

        table = physical_table_name("PUMA")
        got = ctx.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_count, total_income FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            r["PUMA_FIPS"]: (r["household_count"], r["total_income"])
            for _, r in got.iterrows()
        }
        assert rows["P1"] == (2, 120000)
        assert rows["P2"] == (1, 90000)
        # P3 had zero matching households: left exactly as the sentinel
        # values it started with, not zero-filled.
        assert rows["P3"] == (999, 999.0)

    def test_execute_distinct_with_aggregate_matches_non_distinct(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        query = _ast(
            "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
            "WITH DISTINCT p, COUNT(h) AS cnt, SUM(h.income) AS total "
            "SET p.household_count = cnt, p.total_income = total",
        )
        assert is_relation_group_set_eligible(query, ctx)
        execute_relation_group_set(query, ctx)

        table = physical_table_name("PUMA")
        got = ctx.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_count, total_income FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            r["PUMA_FIPS"]: (r["household_count"], r["total_income"])
            for _, r in got.iterrows()
        }
        assert rows["P1"] == (2, 120000)
        assert rows["P2"] == (1, 90000)
        assert rows["P3"] == (999, 999.0)

    def test_execute_query_dispatches_through_pipeline(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        out = Star(context=ctx).execute_query(_GROUP_SET_QUERY)
        assert out.empty

        table = physical_table_name("PUMA")
        got = ctx.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_count FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["PUMA_FIPS"], got["household_count"]))
        assert rows == {"P1": 2, "P2": 1, "P3": 999}

    def test_never_materialises_to_pandas(
        self, household_parquet, puma_parquet, located_in_parquet, monkeypatch
    ) -> None:
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)

        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        query = _ast(_GROUP_SET_QUERY)
        assert is_relation_group_set_eligible(query, ctx)
        execute_relation_group_set(query, ctx)
        assert calls == []


class TestNewColumn:
    """Phase 3a (the FastOpenData streaming-qualification plan (private repository)) -- a
    SET target property that doesn't exist yet in the raw file is created
    via ALTER TABLE rather than making the whole query ineligible. This is
    the shape that blocked almost the entire real fastopendata pipeline
    from streaming (e.g. PUMA's pop_estimate_1yr) until this fix.
    """

    def test_eligible_new_target_property(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "WITH p, COUNT(h) AS cnt "
                "SET p.household_pop_estimate = cnt",
            ),
            ctx,
        )

    def test_execute_creates_column_and_leaves_unmatched_rows_null(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        query = _ast(
            "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
            "WITH p, COUNT(h) AS cnt "
            "SET p.household_pop_estimate = cnt",
        )
        assert is_relation_group_set_eligible(query, ctx)
        execute_relation_group_set(query, ctx)

        table = physical_table_name("PUMA")
        got = ctx.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_pop_estimate FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["PUMA_FIPS"], got["household_pop_estimate"]))
        assert rows["P1"] == 2
        assert rows["P2"] == 1
        # P3 has zero matching households -- left NULL, not zero-filled,
        # same as the existing (pre-existing-column) group_set semantics.
        assert pd.isna(rows["P3"])


class TestParityWithPandasEngine:
    def test_matches_pandas_engine_result(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        streamed = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        Star(context=streamed).execute_query(_GROUP_SET_QUERY)
        table = physical_table_name("PUMA")
        streamed_rows = streamed.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_count, total_income FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()

        pandas_ctx = _pandas_ctx()
        Star(context=pandas_ctx).execute_query(_GROUP_SET_QUERY)
        pandas_out = Star(context=pandas_ctx).execute_query(
            "MATCH (p:PUMA) RETURN p.region AS region, "
            "p.household_count AS household_count, p.total_income AS total_income",
        )

        streamed_by_region = {
            r["PUMA_FIPS"]: (r["household_count"], r["total_income"])
            for _, r in streamed_rows.iterrows()
        }
        pandas_by_region = {
            r["region"]: (r["household_count"], r["total_income"])
            for _, r in pandas_out.iterrows()
        }
        assert streamed_by_region == pandas_by_region


_CATEGORY_D_QUERY = (
    # The real config's puma_rucc_stats/state_rucc_stats shape, on the
    # fixture: CASE inside an aggregate argument, toFloat inside AVG, and
    # SET values that are arithmetic over two WITH aliases.
    "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
    "WHERE h.income IS NOT NULL "
    "WITH p, AVG(toFloat(h.income)) AS avg_income, "
    "SUM(CASE WHEN h.income >= 60000 THEN 1 ELSE 0 END) AS high_income, "
    "COUNT(h) AS total "
    "SET p.avg_income = avg_income, "
    "p.pct_high_income = toFloat(high_income) / total, "
    "p.high_income_count = high_income"
)


class TestSetExpression:
    """Phase 2b category (D) (the FastOpenData streaming-qualification plan (private repository)):
    a SET value that is a scalar expression over the WITH's aggregate
    aliases, plus CASE and toFloat inside aggregate arguments. These two
    shapes were the last four queries keeping the real fastopendata
    pipeline on the eager pandas path.
    """

    def test_eligible_category_d_shape(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        assert is_relation_group_set_eligible(_ast(_CATEGORY_D_QUERY), ctx)

    def test_execute_creates_typed_columns_and_matches_pandas_engine(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
        ctx = _streamed_context(
            household_parquet, puma_parquet, located_in_parquet
        )
        query = _ast(_CATEGORY_D_QUERY)
        execute_relation_group_set(query, ctx)

        table = physical_table_name("PUMA")
        con = ctx.backend.connection
        got = con.execute(
            f'SELECT "PUMA_FIPS", avg_income, pct_high_income, high_income_count FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        streamed = {
            r["PUMA_FIPS"]: (
                r["avg_income"],
                r["pct_high_income"],
                r["high_income_count"],
            )
            for _, r in got.iterrows()
        }
        # P1: hh1 (50000) + hh2 (70000); P2: hh3 (90000); P3: no matches.
        assert streamed["P1"] == (60000.0, 0.5, 1)
        assert streamed["P2"] == (90000.0, 1.0, 1)
        assert all(pd.isna(v) for v in streamed["P3"])

        # The new column's type comes from the SET *expression*, not from
        # one alias it mentions: toFloat(int) / int is DOUBLE.
        described = con.execute(f'DESCRIBE "{table}"').fetchall()
        col_types = {row[0]: str(row[1]) for row in described}
        assert col_types["pct_high_income"] == "DOUBLE"
        assert col_types["avg_income"] == "DOUBLE"

        pandas_ctx = _pandas_ctx()
        Star(context=pandas_ctx).execute_query(_CATEGORY_D_QUERY)
        pandas_out = Star(context=pandas_ctx).execute_query(
            "MATCH (p:PUMA) RETURN p.region AS region, p.avg_income AS a, "
            "p.pct_high_income AS b, p.high_income_count AS c",
        )
        pandas_rows = {
            r["region"]: tuple(
                None if pd.isna(v) else float(v)
                for v in (r["a"], r["b"], r["c"])
            )
            for _, r in pandas_out.iterrows()
        }
        streamed_rows = {
            k: tuple(None if pd.isna(v) else float(v) for v in vals)
            for k, vals in streamed.items()
        }
        assert streamed_rows == pandas_rows
