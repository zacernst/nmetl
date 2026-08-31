"""Phase 2b category (E) (docs/fastopendata_streaming_qualification_plan.md)
— a ``SET`` clause as an ordinary *stage* inside the read-eligible pipeline,
not just the terminal clause of a dedicated mutation kind.

Closes the real pipeline's ``osm_longitude`` shape: ``MATCH (o:OSMNode)
WITH o.longitude AS longitude, id(o) AS identifier, o SET o.foo = longitude
+ 1000 WITH identifier, o.foo AS foo, o.decoded_tags AS decoded_tags RETURN
identifier, foo, decoded_tags`` — a mutation embedded mid-pipeline with a
trailing ``WITH``/``RETURN``, previously a "one-off, not worth a dedicated
eligibility path" gap. Two capabilities, both general:

* **A mid-pipeline ``SET`` stage** (:func:`~pycypher.relation_engine.
  _execute_set_stage` / :func:`~pycypher.relation_engine._set_stage_eligible`)
  — executes as a native ``UPDATE`` the moment it's reached, then folds the
  computed values into the in-flight relation so later stages read them like
  any other property.
* **A mixed ``WITH`` item list** — a bound node passed through bare
  (``o``) *alongside* new named expressions (:func:`~pycypher.
  relation_engine._plan_mixed_stage`) — the pre-existing
  ``_stage_is_passthrough`` check only handled *every* item being a bare
  passthrough.

Both are scoped to single-component patterns only
(:func:`~pycypher.relation_engine._single_component_scope`) — verified
empirically that a joined relation's component aliases stop reliably
resolving once chained through a ``.project()``, while a single-table
relation's alias survives arbitrarily many chained ``.project()`` calls.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.backends.table_registry import physical_table_name
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    bridge_user_functions,
    execute_relation_query,
    is_relation_eligible,
    register_streaming_relationship,
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


def _shout(x: str) -> str:
    return x.upper()


def _ast(query: str):
    return ASTConverter.from_cypher(query)


def _na_to_none(value):
    return None if pd.isna(value) else value


@pytest.fixture
def osm_parquet(tmp_path):
    path = tmp_path / "osm.parquet"
    pd.DataFrame(
        {
            "id": ["n1", "n2", "n3"],
            "longitude": [10.0, 20.0, 30.0],
            "encoded_tags": ["abc", "def", "ghi"],
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


def _osm_ctx(osm_parquet) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx, "OSMNode", data_source_from_uri(str(osm_parquet)), id_col="id"
    )
    return ctx


_OSM_LONGITUDE_QUERY = (
    "MATCH (o:OSMNode) WITH o.longitude AS longitude, id(o) AS identifier, o "
    "SET o.foo = longitude + 1000 "
    "WITH identifier, o.foo AS foo, o.encoded_tags AS et "
    "RETURN identifier, foo, et"
)


class TestEligibility:
    def test_full_osm_longitude_shape_eligible(self, osm_parquet) -> None:
        ctx = _osm_ctx(osm_parquet)
        assert is_relation_eligible(_ast(_OSM_LONGITUDE_QUERY), ctx)

    def test_set_immediately_after_match_no_leading_with(
        self, osm_parquet
    ) -> None:
        ctx = _osm_ctx(osm_parquet)
        assert is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) SET o.foo = o.longitude + 1000 "
                "RETURN o.foo AS foo",
            ),
            ctx,
        )

    def test_set_stage_property_visible_to_later_stage(
        self, osm_parquet
    ) -> None:
        # Pins the within-query analog of Phase 3b's cross-query-sequencing
        # fix: a later stage of the *same* query reading a property this
        # query's own earlier SET stage creates must see it as resolvable
        # at eligibility-check time too, even though nothing is actually
        # created until execution (_set_stage_eligible never writes).
        ctx = _osm_ctx(osm_parquet)
        assert is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) SET o.brand_new = o.longitude "
                "WITH o.brand_new AS bn RETURN bn",
            ),
            ctx,
        )

    def test_multiple_set_items_in_one_stage(self, osm_parquet) -> None:
        ctx = _osm_ctx(osm_parquet)
        assert is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) "
                "SET o.a = o.longitude + 1, o.b = o.longitude + 2 "
                "RETURN o.a AS a, o.b AS b",
            ),
            ctx,
        )

    def test_ineligible_set_items_target_different_variables(
        self, osm_parquet
    ) -> None:
        ctx = _streaming_ctx()
        register_streaming_source(
            ctx, "OSMNode", data_source_from_uri(str(osm_parquet)), id_col="id"
        )
        register_streaming_source(
            ctx, "Other", data_source_from_uri(str(osm_parquet)), id_col="id"
        )
        assert not is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) WITH o "
                "MATCH (p:Other) "
                "SET o.foo = 1, p.foo = 1 "
                "RETURN o.foo AS foo",
            ),
            ctx,
        )

    def test_ineligible_set_on_a_scalar_alias_not_a_node(
        self, osm_parquet
    ) -> None:
        ctx = _osm_ctx(osm_parquet)
        assert not is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) WITH o.longitude AS lon "
                "SET lon.foo = 1 "
                "RETURN lon",
            ),
            ctx,
        )

    def test_ineligible_set_after_a_join_pattern(self, tmp_path) -> None:
        # Single-component restriction: a SET stage after a fixed-length
        # (joined) pattern is out of scope -- see _single_component_scope.
        hh_path = tmp_path / "households.parquet"
        pd.DataFrame({"hh_id": ["h1", "h2"], "income": [1, 2]}).to_parquet(
            hh_path
        )
        puma_path = tmp_path / "pumas.parquet"
        pd.DataFrame({"PUMA_FIPS": ["P1", "P2"]}).to_parquet(puma_path)
        rel_path = tmp_path / "located_in.parquet"
        pd.DataFrame({"src": ["h1", "h2"], "tgt": ["P1", "P1"]}).to_parquet(
            rel_path
        )

        ctx = _streaming_ctx()
        register_streaming_source(
            ctx,
            "Household",
            data_source_from_uri(str(hh_path)),
            id_col="hh_id",
        )
        register_streaming_source(
            ctx,
            "PUMA",
            data_source_from_uri(str(puma_path)),
            id_col="PUMA_FIPS",
        )
        register_streaming_relationship(
            ctx,
            "LOCATED_IN",
            data_source_from_uri(str(rel_path)),
            source_col="src",
            target_col="tgt",
        )
        assert not is_relation_eligible(
            _ast(
                "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
                "SET p.foo = h.income "
                "RETURN p.foo AS foo",
            ),
            ctx,
        )

    def test_ineligible_mixed_with_combines_passthrough_and_aggregate(
        self, osm_parquet
    ) -> None:
        # Aggregation changes cardinality -- incompatible with carrying a
        # raw node's full row through unchanged.
        ctx = _osm_ctx(osm_parquet)
        assert not is_relation_eligible(
            _ast(
                "MATCH (o:OSMNode) WITH o, COUNT(o) AS cnt RETURN cnt",
            ),
            ctx,
        )

    def test_ineligible_pandas_backend(self) -> None:
        osm = pd.DataFrame(
            {ID_COLUMN: ["n1"], "longitude": [10.0], "encoded_tags": ["abc"]}
        )
        ctx = Context(
            entity_mapping=EntityMapping(
                mapping={
                    "OSMNode": EntityTable.from_dataframe("OSMNode", osm)
                },
            ),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="pandas",
        )
        ctx._relation_engine_enabled = True
        assert not is_relation_eligible(_ast(_OSM_LONGITUDE_QUERY), ctx)


class TestExecution:
    def test_execute_full_osm_longitude_shape(self, osm_parquet) -> None:
        ctx = _osm_ctx(osm_parquet)
        query = _ast(_OSM_LONGITUDE_QUERY)
        assert is_relation_eligible(query, ctx)
        out = execute_relation_query(query, ctx)
        rows = {
            r["identifier"]: (r["foo"], r["et"]) for _, r in out.iterrows()
        }
        assert rows == {
            "n1": (1010.0, "abc"),
            "n2": (1020.0, "def"),
            "n3": (1030.0, "ghi"),
        }

    def test_execute_persists_durably_to_the_physical_table(
        self, osm_parquet
    ) -> None:
        # The mid-pipeline SET's UPDATE is a real, durable side effect on
        # the registered table -- not just visible through the in-flight
        # relation this one query happens to be building.
        ctx = _osm_ctx(osm_parquet)
        query = _ast(_OSM_LONGITUDE_QUERY)
        assert is_relation_eligible(query, ctx)
        execute_relation_query(query, ctx)

        table = physical_table_name("OSMNode")
        got = ctx.backend.connection.execute(
            f'SELECT "id", foo FROM "{table}" ORDER BY "id"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = {r["id"]: _na_to_none(r["foo"]) for _, r in got.iterrows()}
        assert rows == {"n1": 1010.0, "n2": 1020.0, "n3": 1030.0}

    def test_execute_set_immediately_after_match(self, osm_parquet) -> None:
        ctx = _osm_ctx(osm_parquet)
        query = _ast(
            "MATCH (o:OSMNode) SET o.foo = o.longitude + 1000 "
            "RETURN id(o) AS oid, o.foo AS foo",
        )
        assert is_relation_eligible(query, ctx)
        out = execute_relation_query(query, ctx)
        rows = dict(zip(out["oid"], out["foo"]))
        assert rows == {"n1": 1010.0, "n2": 1020.0, "n3": 1030.0}

    def test_execute_query_calls_a_udf_and_uses_a_new_column_later(
        self, osm_parquet
    ) -> None:
        ctx = _osm_ctx(osm_parquet)
        register_user_function(_shout, name="shout")
        bridge_user_functions(ctx)
        query = _ast(
            "MATCH (o:OSMNode) SET o.tag_shouted = shout(o.encoded_tags) "
            "WITH o.tag_shouted AS ts RETURN ts",
        )
        assert is_relation_eligible(query, ctx)
        out = execute_relation_query(query, ctx)
        assert sorted(out["ts"]) == ["ABC", "DEF", "GHI"]

    def test_stream_query_to_uri_dispatches_through_pipeline(
        self, osm_parquet, tmp_path
    ) -> None:
        # Star.execute_query() only ever dispatches *mutations* to the
        # relation engine (see is_relation_mutation_eligible there); a read
        # like this one only reaches it via stream_query_to_uri (what
        # cli/pipeline.py's _try_streaming_run uses for every output-sink
        # read) or a direct execute_relation_query() call.
        ctx = _osm_ctx(osm_parquet)
        out_path = tmp_path / "out.csv"
        streamed = Star(context=ctx).stream_query_to_uri(
            _OSM_LONGITUDE_QUERY, f"file://{out_path}"
        )
        assert streamed is True
        got = pd.read_csv(out_path)
        assert sorted(got["identifier"]) == ["n1", "n2", "n3"]

    def test_never_materialises_to_pandas(
        self, osm_parquet, monkeypatch
    ) -> None:
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)

        ctx = _osm_ctx(osm_parquet)
        query = _ast(_OSM_LONGITUDE_QUERY)
        assert is_relation_eligible(query, ctx)
        execute_relation_query(query, ctx)
        assert calls == []


class TestParityWithPandasEngine:
    def test_matches_pandas_engine_result(self, osm_parquet) -> None:
        streamed = _osm_ctx(osm_parquet)
        query = _ast(_OSM_LONGITUDE_QUERY)
        assert is_relation_eligible(query, streamed)
        streamed_out = execute_relation_query(query, streamed)

        osm = pd.DataFrame(
            {
                ID_COLUMN: ["n1", "n2", "n3"],
                "longitude": [10.0, 20.0, 30.0],
                "encoded_tags": ["abc", "def", "ghi"],
            },
        )
        pandas_ctx = Context(
            entity_mapping=EntityMapping(
                mapping={
                    "OSMNode": EntityTable.from_dataframe("OSMNode", osm)
                },
            ),
            relationship_mapping=RelationshipMapping(mapping={}),
            backend="pandas",
        )
        pandas_ctx._relation_engine_enabled = True
        pandas_out = Star(context=pandas_ctx).execute_query(
            _OSM_LONGITUDE_QUERY
        )

        streamed_by_id = {
            r["identifier"]: (r["foo"], r["et"])
            for _, r in streamed_out.iterrows()
        }
        pandas_by_id = {
            r["identifier"]: (r["foo"], r["et"])
            for _, r in pandas_out.iterrows()
        }
        assert streamed_by_id == pandas_by_id
