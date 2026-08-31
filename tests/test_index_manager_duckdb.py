"""Phase 6-partial (DuckDB eager path) — no eager vectorized-store prebuild.

``Context.index_manager`` used to prebuild a ``VectorizedPropertyStore`` for
every entity *and* relationship type on first access, copying each table into
object-dtype numpy arrays whether or not the query touched it.  On DuckDB
that prebuild is skipped; stores are still built on demand, so results are
unchanged and only untouched tables are spared.

See docs/duckdb_eager_path_design.md, Phase 6.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.ingestion.context_builder import ContextBuilder


@pytest.fixture
def frames():
    return {
        "Person": pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]}),
        "Widget": pd.DataFrame({"sku": ["a", "b", "c"]}),
    }


def _context(frames, backend: str):
    builder = ContextBuilder()
    for label, df in frames.items():
        builder = builder.add_entity(label, df)
    return builder.build(backend=backend)


class TestPrebuildPolicy:
    def test_duckdb_does_not_prebuild_any_store(self, frames):
        context = _context(frames, "duckdb")
        assert context.index_manager.stats()["vectorized_stores"] == {}

    def test_pandas_still_prebuilds_every_store(self, frames):
        context = _context(frames, "pandas")
        built = context.index_manager.stats()["vectorized_stores"]
        assert set(built) == {"Person", "Widget"}


class TestOnDemandBuildStillWorks:
    def test_store_is_built_lazily_on_duckdb(self, frames):
        context = _context(frames, "duckdb")
        manager = context.index_manager
        assert manager.stats()["vectorized_stores"] == {}

        store = manager.get_vectorized_store("Person")
        assert store is not None
        assert store.size == 2
        # Only the type actually asked for — Widget is still untouched.
        assert set(manager.stats()["vectorized_stores"]) == {"Person"}

    def test_property_resolution_is_unchanged(self, frames):
        # The prebuild is a performance policy, never a correctness one:
        # get_property must return the same values either way.
        results = {}
        for backend in ("pandas", "duckdb"):
            context = _context(frames, backend)
            from pycypher.binding_frame import BindingFrame

            ids = context.entity_mapping["Person"].source_obj.to_pandas()[
                "__ID__"
            ]
            frame = BindingFrame(
                bindings=pd.DataFrame({"p": list(ids)}),
                type_registry={"p": "Person"},
                context=context,
            )
            results[backend] = frame.get_property("p", "age").tolist()
        assert results["duckdb"] == results["pandas"]
