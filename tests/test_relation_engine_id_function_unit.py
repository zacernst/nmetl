"""id()/elementId() support in the out-of-core relation engine.

Closes the read-side gap the streaming-eligibility audit found (see
docs/fastopendata_streaming_qualification_plan.md, Phase 3 finding (4)):
``relation_sql.compile_expression`` previously had no case for ``id()``/
``elementId()`` at all, forcing any query that referenced a node's internal
ID (e.g. ``RETURN id(c) AS county_fips``) back to the eager pandas engine.

``id(var)``/``elementId(var)`` compiles to *var*'s physical ID column by
reusing the existing ``resolve(var, prop)`` closure via a reserved sentinel
property key (``relation_sql.ID_SENTINEL``) rather than threading a second
callable through every call site — so it works everywhere ``resolve``
already does (``WHERE``, inline predicates, ``WITH``/``RETURN`` items).

The sentinel is injected via ``collections.ChainMap``, not a dict copy: a
copy would desync from ``_ensure_column``'s in-place ``attr_map`` mutation
(Phase 3a), which every mutation executor's "new SET target column" support
depends on holding the *same* dict object end to end. ``TestNewColumnStillWorks``
below pins that interaction directly, since ``id()``-free tests elsewhere
wouldn't catch a regression that reintroduces a copy in the sentinel-injection
path specifically.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ast_converter import ASTConverter
from pycypher.backends.table_registry import physical_table_name
from pycypher.ingestion.data_sources import data_source_from_uri
from pycypher.relation_engine import (
    execute_relation_group_set,
    execute_relation_query,
    is_relation_eligible,
    is_relation_group_set_eligible,
    register_streaming_relationship,
    register_streaming_source,
)
from pycypher.relational_models import (
    Context,
    EntityMapping,
    RelationshipMapping,
)
from pycypher.star import Star


def _ast(query: str):
    return ASTConverter.from_cypher(query)


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
        {"person_id": ["p1", "p2", "p3"], "name": ["Alice", "Bob", "Carol"]},
    ).to_parquet(path)
    return path


@pytest.fixture
def household_parquet(tmp_path):
    path = tmp_path / "households.parquet"
    pd.DataFrame({"hh_id": ["h1", "h2"], "income": [50000, 90000]}).to_parquet(
        path
    )
    return path


@pytest.fixture
def puma_parquet(tmp_path):
    path = tmp_path / "pumas.parquet"
    pd.DataFrame({"PUMA_FIPS": ["P1", "P2"]}).to_parquet(path)
    return path


@pytest.fixture
def located_in_parquet(tmp_path):
    path = tmp_path / "located_in.parquet"
    pd.DataFrame({"src": ["h1", "h2"], "tgt": ["P1", "P1"]}).to_parquet(path)
    return path


def _people_ctx(people_parquet) -> Context:
    ctx = _streaming_ctx()
    register_streaming_source(
        ctx,
        "Person",
        data_source_from_uri(str(people_parquet)),
        id_col="person_id",
    )
    return ctx


class TestEligibility:
    def test_id_of_bound_node_is_eligible(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        assert is_relation_eligible(
            _ast("MATCH (p:Person) RETURN id(p) AS pid"), ctx
        )

    def test_elementid_alias_is_eligible(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        assert is_relation_eligible(
            _ast("MATCH (p:Person) RETURN elementId(p) AS pid"), ctx
        )

    def test_id_mixed_with_ordinary_properties(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        assert is_relation_eligible(
            _ast("MATCH (p:Person) RETURN id(p) AS pid, p.name AS name"), ctx
        )

    def test_id_in_where_clause(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        assert is_relation_eligible(
            _ast("MATCH (p:Person) WHERE id(p) = 'p1' RETURN p.name AS name"),
            ctx,
        )

    def test_id_of_unbound_variable_is_ineligible(
        self, people_parquet
    ) -> None:
        ctx = _people_ctx(people_parquet)
        assert not is_relation_eligible(
            _ast("MATCH (p:Person) RETURN id(q) AS pid"), ctx
        )

    def test_id_of_property_expression_is_ineligible(
        self, people_parquet
    ) -> None:
        # id() only accepts a bare node/relationship variable, never an
        # arbitrary expression -- id(p.name) isn't meaningful Cypher either.
        ctx = _people_ctx(people_parquet)
        assert not is_relation_eligible(
            _ast("MATCH (p:Person) RETURN id(p.name) AS pid"), ctx
        )

    def test_id_with_wrong_arg_count_is_ineligible(
        self, people_parquet
    ) -> None:
        ctx = _people_ctx(people_parquet)
        assert not is_relation_eligible(
            _ast("MATCH (p:Person) RETURN id(p, p) AS pid"), ctx
        )


class TestExecution:
    def test_returns_the_physical_id_column(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        query = _ast("MATCH (p:Person) RETURN id(p) AS pid, p.name AS name")
        assert is_relation_eligible(query, ctx)
        out = execute_relation_query(query, ctx)
        rows = dict(zip(out["pid"], out["name"]))
        assert rows == {"p1": "Alice", "p2": "Bob", "p3": "Carol"}

    def test_id_on_the_far_node_of_a_fixed_length_path(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
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
        query = _ast(
            "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
            "RETURN id(h) AS hh_id, id(p) AS puma_fips"
        )
        assert is_relation_eligible(query, ctx)
        out = execute_relation_query(query, ctx)
        rows = dict(zip(out["hh_id"], out["puma_fips"]))
        assert rows == {"h1": "P1", "h2": "P1"}

    def test_query_dispatches_through_pipeline(self, people_parquet) -> None:
        ctx = _people_ctx(people_parquet)
        out = Star(context=ctx).execute_query(
            "MATCH (p:Person) RETURN id(p) AS pid"
        )
        assert sorted(out["pid"]) == ["p1", "p2", "p3"]

    def test_never_materialises_to_pandas(
        self, people_parquet, monkeypatch
    ) -> None:
        from pycypher.backends import _helpers

        calls: list[type] = []
        original = _helpers._to_pandas

        def counted(obj):
            calls.append(type(obj))
            return original(obj)

        monkeypatch.setattr(_helpers, "_to_pandas", counted)

        ctx = _people_ctx(people_parquet)
        query = _ast("MATCH (p:Person) RETURN id(p) AS pid")
        assert is_relation_eligible(query, ctx)
        execute_relation_query(query, ctx)
        assert calls == []


class TestNewColumnStillWorks:
    """Regression test for the attr_map-identity bug the first version of
    this feature introduced: injecting the id() sentinel via a dict copy
    (``{**attr, ID_SENTINEL: ...}``) desynced ``variables``' attr_map from
    the table registry's live one, so a later ``_ensure_column`` call
    (creating a SET target column that doesn't exist yet) became invisible
    to any resolve() closure built before it ran -- ``KeyError`` on the new
    column name. Fixed via ``collections.ChainMap`` (a live overlay, not a
    copy). This test exercises both concerns together: an ``id()`` read
    alongside a property a ``group_set`` mutation creates fresh.
    """

    def test_id_query_alongside_a_freshly_created_column(
        self, household_parquet, puma_parquet, located_in_parquet
    ) -> None:
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

        # 1. A leading-pattern analysis that also builds an id() sentinel
        #    (the ChainMap overlay) for PUMA, via a read-eligibility check --
        #    exercised before the column below is created, same ordering
        #    as the real pipeline's per-query classification loop.
        read_query = _ast("MATCH (p:PUMA) RETURN id(p) AS puma_fips")
        assert is_relation_eligible(read_query, ctx)

        # 2. A group_set mutation creates a brand-new PUMA column.
        set_query = _ast(
            "MATCH (h:Household)-[:LOCATED_IN]->(p:PUMA) "
            "WITH p, COUNT(h) AS cnt "
            "SET p.household_count = cnt"
        )
        assert is_relation_group_set_eligible(set_query, ctx)
        execute_relation_group_set(set_query, ctx)

        table = physical_table_name("PUMA")
        got = ctx.backend.connection.execute(
            f'SELECT "PUMA_FIPS", household_count FROM "{table}" ORDER BY "PUMA_FIPS"',  # noqa: S608 -- table name from validated physical_table_name(), test-only
        ).fetchdf()
        rows = dict(zip(got["PUMA_FIPS"], got["household_count"]))
        assert rows["P1"] == 2

        # 3. A second id()-bearing query, reading the newly created column
        #    together with id() in the same RETURN, must see it -- no
        #    KeyError, no stale attr_map.
        combined_query = _ast(
            "MATCH (p:PUMA) RETURN id(p) AS puma_fips, "
            "p.household_count AS household_count"
        )
        assert is_relation_eligible(combined_query, ctx)
        out = execute_relation_query(combined_query, ctx)
        combined_rows = dict(zip(out["puma_fips"], out["household_count"]))
        assert combined_rows["P1"] == 2
