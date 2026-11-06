"""``cycle_streaming_connection`` and the two things it re-establishes.

``DuckDBBackend.cycle_connection`` + ``TableRegistry.rebind``
(``tests/test_duckdb_connection_cycle.py``) cover the backend-level
primitive. A streaming run binds two more things to the connection that the
backend does not know about:

* ``context._streaming_sources`` — the read-through alias
  ``register_streaming_source`` writes, which ``_base_relation`` prefers
  over the table registry (see ``register_streaming_source``'s docstring).
* bridged user functions (``bridge_user_functions`` / ``_relation_udfs``).

This covers ``resync_streaming_sources``, ``rebridge_user_functions``, and
the ``cycle_streaming_connection`` wrapper that runs all three steps
together. Board 86bcqr38m / 86bcqm58n.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    create_scratch_database_path,
)
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    bridge_user_functions,
    cycle_streaming_connection,
    rebridge_user_functions,
    register_streaming_relationship,
    register_streaming_source,
    resync_streaming_sources,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)
from pycypher.scalar_functions import ScalarFunctionRegistry
from pycypher.scalar_functions.user_functions import register_user_function
from pycypher.star import Star


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Isolate the singleton ScalarFunctionRegistry per test (same as
    packages/nmetl/tests/test_relation_udf_bridge.py)."""
    reg = ScalarFunctionRegistry.get_instance()
    saved = dict(reg._functions)
    yield
    reg._functions.clear()
    reg._functions.update(saved)


def _double_it(x: int) -> int:
    return x * 2


def _file_backed_ctx() -> Context:
    path = create_scratch_database_path()
    backend = DuckDBBackend(database_path=path, own_database_file=True)
    ctx = Context(
        entity_mapping=EntityMapping(mapping={}),
        relationship_mapping=RelationshipMapping(mapping={}),
        backend=backend,
    )
    ctx._relation_engine_enabled = True
    return ctx


@pytest.fixture
def people_parquet(tmp_path):
    path = tmp_path / "people.parquet"
    pd.DataFrame(
        {"name": ["Alice", "Bob", "Carol"], "age": [30, 25, 35]},
    ).to_parquet(path)
    return path


@pytest.fixture
def knows_parquet(tmp_path):
    path = tmp_path / "knows.parquet"
    pd.DataFrame({"__SOURCE__": [0, 1], "__TARGET__": [1, 2]}).to_parquet(path)
    return path


class TestResyncStreamingSources:
    def test_alias_points_at_the_new_connection_after_a_cycle(
        self, people_parquet
    ):
        ctx = _file_backed_ctx()
        register_streaming_source(
            ctx, "Person", data_source_from_uri(str(people_parquet))
        )
        ctx.backend.cycle_connection()
        # Before resync, the alias still holds the pre-cycle frame bound to
        # the now-closed connection.
        with pytest.raises(Exception):
            ctx._streaming_sources["Person"][0].relation.fetchdf()

        resync_streaming_sources(ctx)
        out = ctx._streaming_sources["Person"][0].relation.fetchdf()
        assert sorted(out["name"].tolist()) == ["Alice", "Bob", "Carol"]

    def test_query_still_works_after_resync(self, people_parquet):
        ctx = _file_backed_ctx()
        register_streaming_source(
            ctx, "Person", data_source_from_uri(str(people_parquet))
        )
        ctx.backend.cycle_connection()
        resync_streaming_sources(ctx)
        out = Star(context=ctx).execute_query(
            "MATCH (n:Person) RETURN n.name AS name"
        )
        assert sorted(out["name"].tolist()) == ["Alice", "Bob", "Carol"]


class TestRebridgeUserFunctions:
    def test_udf_still_callable_after_a_cycle(self, people_parquet):
        register_user_function(_double_it, name="double_it")
        ctx = _file_backed_ctx()
        register_streaming_source(
            ctx, "Person", data_source_from_uri(str(people_parquet))
        )
        bridge_user_functions(ctx)
        assert "double_it" in ctx._relation_udfs

        ctx.backend.cycle_connection()
        resync_streaming_sources(ctx)
        rebridge_user_functions(ctx)

        out = Star(context=ctx).execute_query(
            "MATCH (n:Person) RETURN double_it(n.age) AS doubled"
        )
        assert sorted(out["doubled"].tolist()) == [50, 60, 70]


class TestCycleStreamingConnection:
    def test_wraps_backend_cycle_and_both_resyncs(
        self, people_parquet, knows_parquet
    ):
        register_user_function(_double_it, name="double_it")
        ctx = _file_backed_ctx()
        register_streaming_source(
            ctx, "Person", data_source_from_uri(str(people_parquet))
        )
        register_streaming_relationship(
            ctx,
            "KNOWS",
            data_source_from_uri(str(knows_parquet)),
            source_col="__SOURCE__",
            target_col="__TARGET__",
        )
        bridge_user_functions(ctx)

        cycle_streaming_connection(ctx)

        people = Star(context=ctx).execute_query(
            "MATCH (n:Person) RETURN double_it(n.age) AS doubled"
        )
        assert sorted(people["doubled"].tolist()) == [50, 60, 70]

        rel_count = ctx.backend.tables.get(
            "KNOWS", kind="relationship"
        ).relation.fetchdf()
        assert len(rel_count) == 2

    def test_cycling_with_no_sources_registered_is_a_noop(self):
        ctx = _file_backed_ctx()
        cycle_streaming_connection(ctx)
        assert ctx._streaming_sources == {}
