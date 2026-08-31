"""Phase 2 (DuckDB eager path) — column statistics computed in SQL.

``TableStatistics`` computes NDV, null fraction, min/max, and an equi-width
histogram inside DuckDB when a relation is available, instead of
materialising the source into pandas and running ``nunique`` / ``np.histogram``
over it.

See docs/duckdb_eager_path_design.md, Phase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pycypher.backends.duckdb_backend import count_materialisations
from pycypher.cardinality_estimator import (
    HISTOGRAM_BINS,
    STATS_SAMPLE_SIZE,
    TableStatistics,
)
from pycypher.ingestion.context_builder import ContextBuilder


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, 5000),
            "name": [f"n{i % 300}" for i in range(5000)],
            "score": rng.normal(50, 10, 5000),
        },
    )


@pytest.fixture
def context(frame):
    # The context must stay alive for the whole test: DuckDBBackend.__del__
    # closes the connection, and a relation outliving its backend raises
    # "Connection has already been closed".
    return ContextBuilder().add_entity("Person", frame).build(backend="duckdb")


@pytest.fixture
def relation(context):
    return context.backend.tables.relation("Person")


@pytest.fixture
def sql_stats(relation) -> TableStatistics:
    return TableStatistics(None, relation=relation)


@pytest.fixture
def pandas_stats(frame) -> TableStatistics:
    return TableStatistics(frame)


class TestAgreesWithPandas:
    """The SQL path must be a drop-in for the pandas one."""

    @pytest.mark.parametrize("column", ["age", "name", "score"])
    def test_ndv_and_null_fraction_match(
        self, sql_stats, pandas_stats, column
    ):
        sql = sql_stats.column_stats(column)
        pandas_ = pandas_stats.column_stats(column)
        assert sql.ndv == pandas_.ndv
        assert sql.null_fraction == pytest.approx(pandas_.null_fraction)

    @pytest.mark.parametrize("column", ["age", "score"])
    def test_extrema_match_for_numeric_columns(
        self, sql_stats, pandas_stats, column
    ):
        sql = sql_stats.column_stats(column)
        pandas_ = pandas_stats.column_stats(column)
        assert sql.min_value == pytest.approx(pandas_.min_value)
        assert sql.max_value == pytest.approx(pandas_.max_value)

    def test_equality_selectivity_matches(self, sql_stats, pandas_stats):
        assert sql_stats.column_stats("age").equality_selectivity() == (
            pytest.approx(
                pandas_stats.column_stats("age").equality_selectivity()
            )
        )

    def test_range_selectivity_is_close(self, sql_stats, pandas_stats):
        # Bin membership differs at interior edges (see below), so this is
        # "close", not identical.
        sql = sql_stats.column_stats("age").range_selectivity(low=50)
        pandas_ = pandas_stats.column_stats("age").range_selectivity(low=50)
        assert sql == pytest.approx(pandas_, abs=0.05)


class TestNonNumeric:
    def test_string_columns_get_no_extrema_or_histogram(self, sql_stats):
        stats = sql_stats.column_stats("name")
        assert stats.min_value is None
        assert stats.max_value is None
        assert stats.histogram_edges is None
        assert stats.histogram_counts is None

    def test_string_columns_still_get_ndv(self, sql_stats):
        assert sql_stats.column_stats("name").ndv == 300


class TestHistogramShape:
    """The consumer contract: N counts for N+1 edges, covering every row."""

    def test_counts_are_one_shorter_than_edges(self, sql_stats):
        stats = sql_stats.column_stats("age")
        assert len(stats.histogram_counts) == len(stats.histogram_edges) - 1

    def test_counts_cover_every_non_null_row(self, sql_stats):
        # The fold of DuckDB's first bucket (values <= min) into bin 0 must
        # not drop or double-count anything.
        stats = sql_stats.column_stats("age")
        assert sum(stats.histogram_counts) == 5000

    def test_bin_count_is_capped_by_ndv_and_the_bin_limit(self, sql_stats):
        stats = sql_stats.column_stats("age")
        assert len(stats.histogram_counts) == min(HISTOGRAM_BINS, stats.ndv)

    def test_edges_span_the_observed_range(self, sql_stats):
        stats = sql_stats.column_stats("age")
        assert stats.histogram_edges[0] == pytest.approx(stats.min_value)
        assert stats.histogram_edges[-1] == pytest.approx(stats.max_value)

    def test_constant_column_gets_no_histogram(self):
        ctx = (
            ContextBuilder()
            .add_entity("C", pd.DataFrame({"k": [7] * 100}))
            .build(backend="duckdb")
        )
        stats = TableStatistics(
            None, relation=ctx.backend.tables.relation("C")
        ).column_stats("k")
        # min == max, so there is no range to bin.
        assert stats.histogram_counts is None
        assert stats.min_value == 7.0

    def test_short_column_gets_no_histogram(self):
        ctx = (
            ContextBuilder()
            .add_entity("C", pd.DataFrame({"k": [1, 2, 3]}))
            .build(backend="duckdb")
        )
        stats = TableStatistics(
            None, relation=ctx.backend.tables.relation("C")
        ).column_stats("k")
        assert stats.histogram_counts is None


class TestNulls:
    def test_null_fraction_is_measured(self):
        ctx = (
            ContextBuilder()
            .add_entity(
                "N", pd.DataFrame({"v": [1.0, None, 3.0, None, 5.0] * 20})
            )
            .build(backend="duckdb")
        )
        stats = TableStatistics(
            None, relation=ctx.backend.tables.relation("N")
        ).column_stats("v")
        assert stats.null_fraction == pytest.approx(0.4)
        assert stats.ndv == 3


class TestSampling:
    def test_results_are_reproducible(self, relation):
        # REPEATABLE(seed) keeps DuckDB's sampling deterministic, matching
        # the pandas path's random_state=42 guarantee.
        first = TableStatistics(None, relation=relation).column_stats("score")
        second = TableStatistics(None, relation=relation).column_stats("score")
        assert first == second

    def test_large_tables_are_sampled(self):
        rows = STATS_SAMPLE_SIZE * 3
        ctx = (
            ContextBuilder()
            .add_entity("Big", pd.DataFrame({"v": range(rows)}))
            .build(backend="duckdb")
        )
        stats = TableStatistics(
            None, relation=ctx.backend.tables.relation("Big")
        )
        column = stats.column_stats("v")
        # row_count is the whole table; NDV comes from the bounded sample.
        assert stats.row_count == rows
        assert column.row_count == rows
        assert column.ndv <= STATS_SAMPLE_SIZE

    def test_small_tables_are_not_sampled(self, sql_stats):
        # 5000 rows < STATS_SAMPLE_SIZE, so NDV is exact.
        assert sql_stats.column_stats("name").ndv == 300


class TestNoMaterialisation:
    def test_statistics_never_materialise_to_pandas(self, relation):
        with count_materialisations() as log:
            stats = TableStatistics(None, relation=relation)
            stats.column_stats("age")
            stats.column_stats("name")
            _ = stats.row_count
        assert log.count == 0


class TestFallback:
    def test_unknown_column_returns_none(self, sql_stats):
        assert sql_stats.column_stats("nope") is None

    def test_column_absent_from_relation_falls_back_to_source(
        self, relation, frame
    ):
        # Relation lacks the column but the pandas source has it — the
        # pandas path answers rather than the caller getting nothing.
        extended = frame.assign(extra=1)
        stats = TableStatistics(extended, relation=relation)
        assert stats.column_stats("extra") is not None

    def test_sql_failure_falls_back_to_pandas(self, frame, relation):
        class Exploding:
            columns = relation.columns
            types = relation.types

            def query(self, *_args, **_kwargs):
                msg = "boom"
                raise RuntimeError(msg)

        stats = TableStatistics(frame, relation=Exploding())
        result = stats.column_stats("age")
        assert result is not None
        assert result.ndv == TableStatistics(frame).column_stats("age").ndv

    def test_sql_failure_without_a_source_returns_none(self, relation):
        class Exploding:
            columns = relation.columns
            types = relation.types

            def query(self, *_args, **_kwargs):
                msg = "boom"
                raise RuntimeError(msg)

        assert (
            TableStatistics(None, relation=Exploding()).column_stats("age")
            is None
        )

    def test_caching_survives_a_none_result(self, sql_stats):
        assert sql_stats.column_stats("nope") is None
        assert sql_stats.column_stats("nope") is None


class TestRowCount:
    def test_row_count_from_relation_when_source_is_absent(self, relation):
        assert TableStatistics(None, relation=relation).row_count == 5000

    def test_row_count_prefers_the_in_memory_source(self, frame, relation):
        assert TableStatistics(frame, relation=relation).row_count == len(
            frame
        )


class TestPlannerWiring:
    def test_planner_hands_relations_to_table_statistics(self, context):
        from pycypher.query_planner import QueryPlanAnalyzer

        estimator = QueryPlanAnalyzer(query=None, context=context)
        assert estimator._table_stats["Person"]._relation is not None

    def test_pandas_backend_gets_no_relation(self, frame):
        from pycypher.query_planner import QueryPlanAnalyzer

        context = (
            ContextBuilder()
            .add_entity("Person", frame)
            .build(backend="pandas")
        )
        estimator = QueryPlanAnalyzer(query=None, context=context)
        assert estimator._table_stats["Person"]._relation is None
