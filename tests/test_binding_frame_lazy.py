"""Phase 1b (DuckDB eager path) — BindingFrame's lazy relation carrier.

A ``BindingFrame`` may be built over an unmaterialised DuckDB relation
instead of a pandas DataFrame.  ``.bindings`` then materialises on first
access, so all existing callers work unchanged; operators that know how to
stay in SQL check ``.is_lazy`` and consume ``.relation`` first.

Schema and row-count access must stay free — that is what makes a later
"this query materialises zero times" assertion meaningful rather than
vacuous.

See docs/duckdb_eager_path_design.md, Phase 1b.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    DuckDBLazyFrame,
    count_materialisations,
)
from pycypher.binding_frame import BindingFrame
from pycypher.ingestion.context_builder import ContextBuilder


@pytest.fixture
def people() -> pd.DataFrame:
    return pd.DataFrame(
        {"name": ["Alice", "Bob", "Carol"], "age": [30, 25, 35]}
    )


@pytest.fixture
def context(people):
    return (
        ContextBuilder().add_entity("Person", people).build(backend="duckdb")
    )


@pytest.fixture
def backend():
    be = DuckDBBackend()
    try:
        yield be
    finally:
        be.close()


def _lazy(backend, sql: str) -> DuckDBLazyFrame:
    return DuckDBLazyFrame(backend.connection.sql(sql), backend.connection)


@pytest.fixture
def lazy_frame(context):
    """A lazy BindingFrame of Person IDs bound to variable ``p``."""
    entry = context.backend.tables.get("Person")
    relation = entry.relation.project('"__ID__" AS "p"')
    return BindingFrame(
        relation=DuckDBLazyFrame(relation, context.backend.connection),
        type_registry={"p": "Person"},
        context=context,
    )


class TestConstruction:
    def test_pandas_construction_is_unchanged(self, context):
        frame = BindingFrame(
            bindings=pd.DataFrame({"p": [1, 2]}),
            type_registry={"p": "Person"},
            context=context,
        )
        assert frame.is_lazy is False
        assert frame.relation is None
        assert frame.var_names == ["p"]

    def test_relation_construction_starts_lazy(self, lazy_frame):
        assert lazy_frame.is_lazy is True
        assert lazy_frame.relation is not None

    def test_requires_exactly_one_source(self, context):
        with pytest.raises(ValueError, match="neither"):
            BindingFrame(type_registry={}, context=context)
        with pytest.raises(ValueError, match="both"):
            BindingFrame(
                bindings=pd.DataFrame({"p": [1]}),
                relation=object(),
                type_registry={},
                context=context,
            )

    def test_type_registry_and_context_are_carried(self, lazy_frame, context):
        assert lazy_frame.type_registry == {"p": "Person"}
        assert lazy_frame.context is context


class TestFreeOperations:
    """Schema and cardinality must not force materialisation."""

    def test_var_names_is_free(self, lazy_frame):
        with count_materialisations() as log:
            assert lazy_frame.var_names == ["p"]
        assert log.count == 0
        assert lazy_frame.is_lazy is True

    def test_len_is_free(self, lazy_frame):
        with count_materialisations() as log:
            assert len(lazy_frame) == 3
        assert log.count == 0
        assert lazy_frame.is_lazy is True

    def test_repr_is_free(self, lazy_frame):
        with count_materialisations() as log:
            text = repr(lazy_frame)
        assert log.count == 0
        assert "lazy" in text
        assert "'p'" in text

    def test_repr_of_a_materialised_frame_reports_rows(self, context):
        frame = BindingFrame(
            bindings=pd.DataFrame({"p": [1, 2]}),
            type_registry={"p": "Person"},
            context=context,
        )
        assert "rows=2" in repr(frame)


class TestMaterialisation:
    def test_bindings_access_materialises_once(self, lazy_frame):
        with count_materialisations() as log:
            first = lazy_frame.bindings
            second = lazy_frame.bindings
        assert log.count == 1
        assert first is second
        assert list(first.columns) == ["p"]
        assert len(first) == 3

    def test_materialisation_is_one_way(self, lazy_frame):
        assert lazy_frame.is_lazy is True
        _ = lazy_frame.bindings
        assert lazy_frame.is_lazy is False
        # The relation is released rather than served alongside a DataFrame
        # a later pandas-side edit could make stale.
        assert lazy_frame.relation is None

    def test_len_after_materialisation_uses_pandas(self, lazy_frame):
        _ = lazy_frame.bindings
        with count_materialisations() as log:
            assert len(lazy_frame) == 3
        assert log.count == 0

    def test_row_order_survives_materialisation(self, context):
        # Row order is one of the three invariants Phase 1b has to preserve
        # (docs/duckdb_eager_path_design.md, Cross-cutting concerns): DuckDB
        # keeps insertion order for a plain table scan while
        # preserve_insertion_order is left at its default.
        entry = context.backend.tables.get("Person")
        frame = BindingFrame(
            relation=DuckDBLazyFrame(
                entry.relation.project('"name"'),
                context.backend.connection,
            ),
            type_registry={},
            context=context,
        )
        assert frame.bindings["name"].tolist() == ["Alice", "Bob", "Carol"]

    def test_column_order_survives_materialisation(self, backend, context):
        backend.connection.execute(
            "CREATE TABLE t AS SELECT 1 AS z, 2 AS a, 3 AS m"
        )
        frame = BindingFrame(
            relation=_lazy(backend, "SELECT z, a, m FROM t"),
            type_registry={},
            context=context,
        )
        assert frame.var_names == ["z", "a", "m"]
        assert list(frame.bindings.columns) == ["z", "a", "m"]


class TestSetter:
    def test_assigning_bindings_clears_the_relation(self, lazy_frame):
        lazy_frame.bindings = pd.DataFrame({"p": [9]})
        assert lazy_frame.is_lazy is False
        assert lazy_frame.relation is None
        assert lazy_frame.bindings["p"].tolist() == [9]

    def test_assignment_does_not_materialise_the_discarded_relation(
        self, lazy_frame
    ):
        with count_materialisations() as log:
            lazy_frame.bindings = pd.DataFrame({"p": [9]})
        assert log.count == 0


class TestExistingOperationsStillWork:
    """A lazy frame must be indistinguishable to callers that don't opt in."""

    def test_get_property_materialises_and_resolves(self, lazy_frame):
        ages = lazy_frame.get_property("p", "age")
        assert sorted(ages.tolist()) == [25, 30, 35]

    def test_filter_works_through_the_carrier(self, lazy_frame):
        ages = lazy_frame.get_property("p", "age")
        filtered = lazy_frame.filter(ages > 26)
        assert len(filtered) == 2

    def test_entity_type_lookup(self, lazy_frame):
        assert lazy_frame.entity_type("p") == "Person"
