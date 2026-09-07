"""Phase 4 (DuckDB eager path) — renames, concats, and distinct in SQL.

``rename`` becomes a projection, ``concat`` a ``UNION ALL``, and ``distinct``
a lazy ``DISTINCT``, so a chain of scans can be reshaped without leaving
DuckDB.

The interesting content here is the *guards*: DuckDB's ``relation.union()``
diverges from ``pd.concat`` in two ways that produce wrong answers rather
than errors, so the lazy path is taken only when the inputs are exactly
compatible.

See docs/duckdb_eager_path_design.md, Phase 4.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import assert_same_across_backends, build_context
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    DuckDBLazyFrame,
    count_materialisations,
)
from pycypher.binding_frame import concat_binding_frames
from pycypher.scan_operators import EntityScan

PEOPLE = pd.DataFrame({"name": ["A", "B"], "age": [1, 2]})
WIDGETS = pd.DataFrame({"sku": ["x", "y", "z"]})
ENTITIES = {"Person": PEOPLE, "Widget": WIDGETS}


@pytest.fixture
def duck():
    return build_context(ENTITIES, backend="duckdb")


@pytest.fixture
def backend():
    be = DuckDBBackend()
    try:
        yield be
    finally:
        be.close()


def _lazy(backend, frame: pd.DataFrame, name: str) -> DuckDBLazyFrame:
    backend.connection.register(name, frame)
    return DuckDBLazyFrame(
        backend.connection.sql(f"SELECT * FROM {name}"),  # noqa: S608 — name is a test-local literal
        backend.connection,
    )


class TestRename:
    def test_rename_stays_lazy(self, duck):
        frame = EntityScan("Person", "p").scan(duck)
        with count_materialisations() as log:
            renamed = frame.rename("p", "q", new_type="Person")
            assert renamed.var_names == ["q"]
            assert len(renamed) == 2
        assert log.count == 0
        assert renamed.is_lazy is True

    def test_rename_preserves_column_order(self, backend):
        lazy = _lazy(backend, pd.DataFrame({"z": [1], "a": [2], "m": [3]}), "t")
        out = backend.rename(lazy, {"a": "b"})
        assert list(out.columns) == ["z", "b", "m"]

    def test_rename_to_an_existing_name_falls_back_to_pandas(self, backend):
        # pandas tolerates duplicate labels; SQL cannot express them.
        lazy = _lazy(backend, pd.DataFrame({"a": [1], "b": [2]}), "t2")
        out = backend.rename(lazy, {"a": "b"})
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["b", "b"]

    def test_rename_of_a_pandas_frame_is_unchanged(self, backend):
        out = backend.rename(pd.DataFrame({"a": [1]}), {"a": "b"})
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["b"]

    def test_renaming_an_absent_column_still_raises(self, duck):
        from pycypher.exceptions import VariableNotFoundError

        frame = EntityScan("Person", "p").scan(duck)
        with pytest.raises(VariableNotFoundError):
            frame.rename("nope", "q")


class TestConcat:
    def test_matching_frames_union_lazily(self, duck):
        first = EntityScan("Person", "n").scan(duck)
        second = EntityScan("Widget", "n").scan(duck)
        with count_materialisations() as log:
            combined = concat_binding_frames(
                duck, [first, second], type_registry={"n": "__MULTI__"}
            )
            assert len(combined) == 5
        assert log.count == 0
        assert combined.is_lazy is True

    def test_distinct_stays_lazy(self, duck):
        first = EntityScan("Person", "n").scan(duck)
        second = EntityScan("Widget", "n").scan(duck)
        combined = concat_binding_frames(
            duck,
            [first, second],
            type_registry={"n": "__MULTI__"},
            distinct=True,
        )
        assert combined.is_lazy is True
        # Person ids {0,1} and Widget ids {0,1,2} overlap.
        assert len(combined) == 3

    def test_concat_of_pandas_frames_is_unchanged(self, backend):
        out = backend.concat(
            [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        )
        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1, 2]


class TestConcatGuards:
    """The two ways relation.union() silently diverges from pd.concat."""

    def test_differing_columns_fall_back_instead_of_unioning_positionally(
        self, duck
    ):
        # union() would line these up by position and keep the left names,
        # yielding one column of mislabelled data. pandas gives two columns
        # with nulls, which is the established semantics.
        first = EntityScan("Person", "n").scan(duck)
        second = EntityScan("Widget", "m").scan(duck)
        combined = concat_binding_frames(duck, [first, second], type_registry={})
        assert combined.is_lazy is False
        assert sorted(combined.bindings.columns) == ["m", "n"]
        assert len(combined) == 5

    def test_differing_types_fall_back_instead_of_coercing(self, backend):
        # union() turns [1, 2] + ["x"] into ["1", "2", "x"]; pandas keeps
        # [1, 2, "x"] as object dtype.
        ints = _lazy(backend, pd.DataFrame({"v": [1, 2]}), "ints")
        texts = _lazy(backend, pd.DataFrame({"v": ["x"]}), "texts")
        out = backend.concat([ints, texts])
        assert isinstance(out, pd.DataFrame)
        assert out["v"].tolist() == [1, 2, "x"]

    def test_column_order_differences_fall_back(self, backend):
        left = _lazy(backend, pd.DataFrame({"a": [1], "b": [2]}), "l")
        right = _lazy(backend, pd.DataFrame({"b": [3], "a": [4]}), "r")
        out = backend.concat([left, right])
        assert isinstance(out, pd.DataFrame)
        # pandas aligns by name, so column 'a' holds [1, 4].
        assert out["a"].tolist() == [1, 4]

    def test_mixed_lazy_and_pandas_inputs_fall_back(self, backend):
        lazy = _lazy(backend, pd.DataFrame({"a": [1]}), "mixed")
        out = backend.concat([lazy, pd.DataFrame({"a": [2]})])
        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1, 2]

    def test_ignore_index_false_falls_back(self, backend):
        left = _lazy(backend, pd.DataFrame({"a": [1]}), "i1")
        right = _lazy(backend, pd.DataFrame({"a": [2]}), "i2")
        out = backend.concat([left, right], ignore_index=False)
        assert isinstance(out, pd.DataFrame)

    def test_empty_input_keeps_the_pandas_error(self, backend):
        # pd.concat([]) has always raised here; the lazy path must not
        # quietly start returning an empty frame instead.
        with pytest.raises(ValueError, match="No objects to concatenate"):
            backend.concat([])


class TestDifferentialQueries:
    @pytest.mark.parametrize(
        "cypher",
        [
            # Unlabeled node: fans out across both entity types and concats.
            "MATCH (n) RETURN count(n)",
            "MATCH (p:Person) RETURN p.name ORDER BY p.name",
            "MATCH (w:Widget) RETURN w.sku ORDER BY w.sku",
        ],
    )
    def test_query_matches_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher)
