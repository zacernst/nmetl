"""Phase 1c (DuckDB eager path) — the materialisation tripwire.

Every point where a lazy DuckDB relation is forced into pandas is a point
where the out-of-core guarantee is lost.  ``count_materialisations`` makes
that countable, so later phases can assert "this query materialises zero
times" instead of eyeballing a profile.

See docs/duckdb_eager_path_design.md, Phase 1c.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    DuckDBLazyFrame,
    count_materialisations,
)


@pytest.fixture
def backend():
    be = DuckDBBackend()
    try:
        yield be
    finally:
        be.close()


@pytest.fixture
def lazy(backend) -> DuckDBLazyFrame:
    df = pd.DataFrame({"a": [1, 2, 3]})
    backend.connection.register("_t", df)
    return DuckDBLazyFrame(
        backend.connection.sql("SELECT * FROM _t"), backend.connection
    )


class TestCounting:
    def test_no_materialisation_is_counted_as_zero(self, lazy):
        with count_materialisations() as log:
            # Schema access must stay free — it is answered from the
            # relation, never by executing it.
            assert lazy.columns == ["a"]
        assert log.count == 0

    def test_row_count_does_not_materialise(self, lazy):
        with count_materialisations() as log:
            assert len(lazy) == 3
        assert log.count == 0

    def test_to_pandas_is_counted(self, lazy):
        with count_materialisations() as log:
            lazy.to_pandas()
        assert log.count == 1
        assert log.rows == 3
        assert log.events[0].columns == ("a",)

    def test_cached_second_materialisation_is_not_double_counted(self, lazy):
        with count_materialisations() as log:
            lazy.to_pandas()
            lazy.to_pandas()
        assert log.count == 1

    def test_attribute_delegation_materialises(self, lazy):
        with count_materialisations() as log:
            _ = lazy.shape
        assert log.count == 1

    def test_counts_are_scoped(self, lazy):
        with count_materialisations() as first:
            pass
        with count_materialisations() as second:
            lazy.to_pandas()
        assert first.count == 0
        assert second.count == 1

    def test_outside_any_scope_is_harmless(self, lazy):
        assert lazy.to_pandas()["a"].tolist() == [1, 2, 3]


class TestNesting:
    def test_inner_scope_does_not_hide_events_from_the_outer(self, backend):
        df = pd.DataFrame({"a": [1]})
        backend.connection.register("_t", df)

        def make() -> DuckDBLazyFrame:
            return DuckDBLazyFrame(
                backend.connection.sql("SELECT * FROM _t"), backend.connection
            )

        with count_materialisations() as outer:
            make().to_pandas()
            with count_materialisations() as inner:
                make().to_pandas()
            assert inner.count == 1
        # A helper that opens its own scope must not shadow work from an
        # enclosing assertion.
        assert outer.count == 2


class TestOrigin:
    def test_origin_is_none_by_default(self, lazy):
        with count_materialisations() as log:
            lazy.to_pandas()
        assert log.events[0].origin is None

    def test_capture_stack_records_the_forcing_caller(self, lazy):
        with count_materialisations(capture_stack=True) as log:
            lazy.to_pandas()
        origin = log.events[0].origin
        assert origin is not None
        # The interesting frame is this test, never the lazy-frame plumbing.
        assert "test_materialisation_tripwire.py" in origin

    def test_capture_stack_is_inherited_by_nested_scopes(self, backend):
        backend.connection.register("_t", pd.DataFrame({"a": [1]}))
        with (
            count_materialisations(capture_stack=True) as outer,
            count_materialisations() as inner,
        ):
            DuckDBLazyFrame(
                backend.connection.sql("SELECT * FROM _t"),
                backend.connection,
            ).to_pandas()
        assert inner.events[0].origin is not None
        assert outer.events[0].origin is not None


class TestLogObject:
    def test_empty_log_is_truthy(self, lazy):
        # `if log:` must not silently mean "if anything materialised".
        with count_materialisations() as log:
            pass
        assert bool(log) is True
        assert len(log) == 0

    def test_repr_reports_count_and_rows(self, lazy):
        with count_materialisations() as log:
            lazy.to_pandas()
        assert "count=1" in repr(log)
        assert "rows=3" in repr(log)
