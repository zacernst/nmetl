"""Phase 2b-i (docs/fastopendata_streaming_qualification_plan.md) —
single-node scalar ``WITH``-then-``SET`` mutation eligibility.

Closes category (A) of the Phase 2 long tail: a single-node ``MATCH (var)
WITH var, expr AS alias SET var.prop = alias`` previously fell back to the
pandas engine unconditionally, purely because ``_analyze_set_query`` rejects
*any* ``WITH`` stage outright — 21 of the pipeline's 34 long-tail queries
(62%) are exactly this shape: a ``WITH`` that exists only to name an
expression before assigning it, with no relationship and no aggregation.

No join, no ``GROUP BY`` — this compiles to the same plain single-table
``UPDATE`` :func:`execute_relation_set` already produces, with the ``WITH``
stage's aliases inlined directly into the ``SET`` values.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.backends.table_registry import physical_table_name
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    bridge_user_functions,
    execute_relation_scalar_set,
    is_relation_scalar_set_eligible,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    EntityTable,
    RelationshipMapping,
)
from pycypher.scalar_functions import ScalarFunctionRegistry
from pycypher.scalar_functions.user_functions import register_user_function
from pycypher.star import Star

ID_COLUMN = "__ID__"


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Isolate the singleton UDF registry per test (see test_relation_udf_bridge.py)."""
    reg = ScalarFunctionRegistry.get_instance()
    saved = dict(reg._functions)
    yield
    reg._functions.clear()
    reg._functions.update(saved)


def _double(x: float) -> float:
    return x * 2


def _ast(query: str):
    return ASTConverter.from_cypher(query)


def _na_to_none(value):
    return None if pd.isna(value) else value


@pytest.fixture
def counties_parquet(tmp_path):
    path = tmp_path / "counties.parquet"
    pd.DataFrame(
        {
            "GEOID": ["001", "002", "003"],
            "rucc_2023": [2, 8, None],
            "is_metro": pd.array([None, None, None], dtype="boolean"),
            "rucc_class": pd.array(["", "", ""], dtype="string"),
        },
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


def _streamed_context(counties_parquet) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx,
        "County",
        data_source_from_uri(str(counties_parquet)),
        id_col="GEOID",
    )
    return ctx


def _entity_ctx(backend: str = "duckdb") -> Context:
    counties = pd.DataFrame(
        {
            ID_COLUMN: ["001", "002", "003"],
            "rucc_2023": [2, 8, None],
            "is_metro": pd.array([None, None, None], dtype="boolean"),
        },
    )
    ctx = Context(
        entity_mapping=EntityMapping(
            mapping={"County": EntityTable.from_dataframe("County", counties)},
        ),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend=backend,
    )
    ctx._relation_engine_enabled = True
    return ctx


_SCALAR_SET_QUERY = (
    "MATCH (c:County) "
    "WITH c, c.rucc_2023 <= 3 AS is_metro "
    "SET c.is_metro = is_metro"
)


class TestEligibility:
    def test_eligible_scalar_set(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(_ast(_SCALAR_SET_QUERY), ctx)

    def test_eligible_multiple_scalar_items(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) "
                "WITH c, c.rucc_2023 <= 3 AS is_metro, c.rucc_2023 * 10 AS scaled "
                "SET c.is_metro = is_metro, c.rucc_2023 = scaled",
            ),
            ctx,
        )

    def test_eligible_with_match_where(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WHERE c.rucc_2023 IS NOT NULL "
                "WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = is_metro",
            ),
            ctx,
        )

    def test_eligible_case_expression(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) "
                "WITH c, CASE WHEN c.rucc_2023 <= 3 THEN 'metro' "
                "WHEN c.rucc_2023 <= 7 THEN 'nonmetro' ELSE 'rural' END AS rucc_class "
                "SET c.is_metro = rucc_class",
            ),
            ctx,
        )

    def test_ineligible_optional_match(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "OPTIONAL MATCH (c:County) "
                "WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = is_metro",
            ),
            ctx,
        )

    def test_ineligible_relationship_in_pattern(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County)-[:LOCATED_IN]->(s:State) "
                "WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = is_metro",
            ),
            ctx,
        )

    def test_ineligible_aggregate_in_with(self, counties_parquet) -> None:
        # An aggregate here means this is _analyze_group_set_query's shape,
        # not this one (and there's no relationship to aggregate over here
        # anyway, so it's ineligible for both).
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c, COUNT(c) AS cnt SET c.is_metro = cnt",
            ),
            ctx,
        )

    def test_ineligible_missing_passthrough_item(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = is_metro",
            ),
            ctx,
        )

    def test_eligible_set_expression_general_expression(
        self, counties_parquet
    ) -> None:
        # Was rejected before the SET-value restriction was relaxed to a
        # general compile_expression() compile (any expression over the
        # bound node's properties and the WITH stage's own aliases, not
        # just a bare alias pass-through) -- see docs/
        # fastopendata_streaming_qualification_plan.md, Phase 2b-i.
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = NOT is_metro",
            ),
            ctx,
        )

    def test_eligible_set_expression_calls_a_registered_udf(
        self, counties_parquet
    ) -> None:
        # The exact shape that blocked the real pipeline's decode_tags query
        # (SET o.decoded_tags = decode_serialized(encoded_tags)): a UDF call
        # over a WITH alias, not the alias itself.
        ctx = _streamed_context(counties_parquet)
        register_user_function(_double, name="doublit")
        bridge_user_functions(ctx)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c, c.rucc_2023 AS r "
                "SET c.rucc_2023 = doublit(r)",
            ),
            ctx,
        )

    def test_ineligible_set_value_references_unbound_name(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET c.is_metro = not_a_real_alias",
            ),
            ctx,
        )

    def test_ineligible_set_targets_unbound_variable(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        assert not is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) WITH c, c.rucc_2023 <= 3 AS is_metro "
                "SET d.is_metro = is_metro",
            ),
            ctx,
        )

    def test_ineligible_no_streaming_source(self) -> None:
        ctx = _entity_ctx()
        assert not is_relation_scalar_set_eligible(
            _ast(_SCALAR_SET_QUERY), ctx
        )

    def test_ineligible_pandas_backend(self) -> None:
        ctx = _entity_ctx(backend="pandas")
        assert not is_relation_scalar_set_eligible(
            _ast(_SCALAR_SET_QUERY), ctx
        )


class TestExecution:
    def test_execute_updates_via_native_update(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        query = _ast(_SCALAR_SET_QUERY)
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", is_metro FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            geoid: _na_to_none(v)
            for geoid, v in zip(got["GEOID"], got["is_metro"])
        }
        assert rows == {"001": True, "002": False, "003": None}

    def test_execute_query_dispatches_through_pipeline(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        out = Star(context=ctx).execute_query(_SCALAR_SET_QUERY)
        assert out.empty

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", is_metro FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            geoid: _na_to_none(v)
            for geoid, v in zip(got["GEOID"], got["is_metro"])
        }
        assert rows == {"001": True, "002": False, "003": None}

    def test_execute_case_expression(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        query = _ast(
            "MATCH (c:County) "
            "WITH c, CASE WHEN c.rucc_2023 <= 3 THEN 'metro' "
            "WHEN c.rucc_2023 <= 7 THEN 'nonmetro' ELSE 'rural' END AS rucc_class "
            "SET c.rucc_class = rucc_class",
        )
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", rucc_class FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["GEOID"], got["rucc_class"]))
        assert rows == {"001": "metro", "002": "rural", "003": "rural"}

    def test_execute_general_set_expression(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        query = _ast(
            "MATCH (c:County) WITH c, c.rucc_2023 <= 3 AS is_metro "
            "SET c.is_metro = NOT is_metro",
        )
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", is_metro FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            geoid: _na_to_none(v)
            for geoid, v in zip(got["GEOID"], got["is_metro"])
        }
        # rucc_2023 is [2, 8, None] -- <= 3 is [True, False, None], negated
        # is [False, True, None] (NOT NULL stays NULL, not True).
        assert rows == {"001": False, "002": True, "003": None}

    def test_execute_set_expression_calls_a_registered_udf(
        self, counties_parquet
    ) -> None:
        # Exercises the real decode_tags shape end to end: SET assigns a UDF
        # call over a WITH alias, not the alias itself.
        ctx = _streamed_context(counties_parquet)
        register_user_function(_double, name="doublit")
        bridge_user_functions(ctx)
        query = _ast(
            "MATCH (c:County) WITH c, c.rucc_2023 AS r "
            "SET c.rucc_2023 = doublit(r)",
        )
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", rucc_2023 FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            geoid: _na_to_none(v)
            for geoid, v in zip(got["GEOID"], got["rucc_2023"])
        }
        assert rows == {"001": 4, "002": 16, "003": None}

    def test_never_materialises_to_pandas(
        self, counties_parquet, monkeypatch
    ) -> None:
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)

        ctx = _streamed_context(counties_parquet)
        query = _ast(_SCALAR_SET_QUERY)
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)
        assert calls == []


class TestNewColumn:
    """Phase 3a (docs/fastopendata_streaming_qualification_plan.md) -- a
    SET target property that doesn't exist yet in the raw file is created
    via ALTER TABLE rather than making the whole query ineligible.
    """

    def test_eligible_new_target_property(self, counties_parquet) -> None:
        ctx = _streamed_context(counties_parquet)
        assert is_relation_scalar_set_eligible(
            _ast(
                "MATCH (c:County) "
                "WITH c, c.rucc_2023 * 10 AS scaled "
                "SET c.rucc_scaled = scaled",
            ),
            ctx,
        )

    def test_execute_creates_column_and_leaves_unmatched_rows_null(
        self, counties_parquet
    ) -> None:
        ctx = _streamed_context(counties_parquet)
        query = _ast(
            "MATCH (c:County) "
            "WITH c, c.rucc_2023 * 10 AS scaled "
            "SET c.rucc_scaled = scaled",
        )
        assert is_relation_scalar_set_eligible(query, ctx)
        execute_relation_scalar_set(query, ctx)

        table = physical_table_name("County")
        got = ctx.backend.connection.execute(
            f'SELECT "GEOID", rucc_scaled FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {
            geoid: _na_to_none(v)
            for geoid, v in zip(got["GEOID"], got["rucc_scaled"])
        }
        # rucc_2023 is [2, 8, None] for 001/002/003 -- 003 has no rucc_2023,
        # so its computed value is NULL too (not zero-filled).
        assert rows == {"001": 20, "002": 80, "003": None}


class TestParityWithPandasEngine:
    def test_matches_pandas_engine_result(self, counties_parquet) -> None:
        streamed = _streamed_context(counties_parquet)
        Star(context=streamed).execute_query(_SCALAR_SET_QUERY)
        table = physical_table_name("County")
        streamed_rows = streamed.backend.connection.execute(
            f'SELECT "GEOID", is_metro FROM "{table}" ORDER BY "GEOID"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()

        pandas_ctx = _entity_ctx()
        Star(context=pandas_ctx).execute_query(_SCALAR_SET_QUERY)
        # id(c) returns the underlying __ID__ value directly, so it lines up
        # with the streamed side's GEOID column with no type-conversion
        # ambiguity (the id itself isn't a queryable Cypher property).
        pandas_out = Star(context=pandas_ctx).execute_query(
            "MATCH (c:County) RETURN id(c) AS geoid, c.is_metro AS is_metro",
        )

        streamed_by_geoid = {
            geoid: _na_to_none(v)
            for geoid, v in zip(
                streamed_rows["GEOID"], streamed_rows["is_metro"]
            )
        }
        pandas_by_geoid = {
            geoid: _na_to_none(v)
            for geoid, v in zip(pandas_out["geoid"], pandas_out["is_metro"])
        }
        assert streamed_by_geoid == pandas_by_geoid
