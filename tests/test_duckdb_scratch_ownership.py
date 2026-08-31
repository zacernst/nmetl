"""Scratch-database *ownership* and the ContextBuilder auto policy.

Complements ``test_duckdb_scratch_database.py``, which covers the file-backed
``database_path`` primitive and the orphan sweep.  This file covers what was
added on top: a backend that owns its file and deletes it on ``close()``, and
``ContextBuilder.build`` deciding when to create one.


Open item 1 from ``docs/duckdb_eager_path_report.md``.

The *when* matters and is measured, not assumed: a file-backed database on
its own costs a little memory rather than saving it, because DuckDB will not
evict buffer-pool pages until it is given a budget.  It pays off only
alongside ``PYCYPHER_DUCKDB_MEMORY_LIMIT``, so that is exactly when it is
switched on.  See ``tests/benchmarks/bench_duckdb_eager_path_memory.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from helpers_differential import build_context, run_query
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    create_scratch_database_path,
    memory_limit_configured,
)
from pycypher.ingestion.context_builder import ContextBuilder

MEMORY_LIMIT_ENV = "PYCYPHER_DUCKDB_MEMORY_LIMIT"
PEOPLE = pd.DataFrame({"name": ["a", "b", "c"], "age": [1, 2, 3]})
ENTITIES = {"Person": PEOPLE}
QUERY = "MATCH (p:Person) WHERE p.age > 1 RETURN p.name ORDER BY p.name"


@pytest.fixture(autouse=True)
def _clear_memory_limit(monkeypatch):
    """Each test states its own budget; never inherit the caller's."""
    monkeypatch.delenv(MEMORY_LIMIT_ENV, raising=False)


def _build(**kwargs):
    return ContextBuilder().add_entity("Person", PEOPLE).build(**kwargs)


class TestFileOwnership:
    def test_owned_file_is_deleted_on_close(self):
        path = create_scratch_database_path()
        backend = DuckDBBackend(database_path=path, own_database_file=True)
        backend.connection.execute("CREATE TABLE t AS SELECT 1 AS a")
        assert Path(path).exists()
        backend.close()
        assert not Path(path).exists()

    def test_write_ahead_log_is_deleted_too(self):
        # A leftover .wal would keep the orphan sweep finding a sibling of a
        # database that no longer exists.
        path = create_scratch_database_path()
        backend = DuckDBBackend(database_path=path, own_database_file=True)
        backend.connection.execute("CREATE TABLE t AS SELECT 1 AS a")
        backend.close()
        assert not Path(f"{path}.wal").exists()

    def test_unowned_file_survives_close(self, tmp_path):
        path = str(tmp_path / "keep.duckdb")
        backend = DuckDBBackend(database_path=path)
        backend.connection.execute("CREATE TABLE t AS SELECT 1 AS a")
        backend.close()
        assert Path(path).exists()

    def test_ownership_requires_a_path(self):
        # own_database_file is meaningless for :memory: and must not blow up.
        backend = DuckDBBackend(own_database_file=True)
        assert backend.database_path is None
        backend.close()

    def test_close_is_idempotent(self):
        path = create_scratch_database_path()
        backend = DuckDBBackend(database_path=path, own_database_file=True)
        backend.close()
        backend.close()
        assert not Path(path).exists()

    def test_context_manager_deletes_on_exit(self):
        path = create_scratch_database_path()
        with DuckDBBackend(database_path=path, own_database_file=True):
            assert Path(path).exists()
        assert not Path(path).exists()


class TestAutoPolicy:
    def test_no_memory_limit_means_in_memory(self):
        context = _build(backend="duckdb")
        assert context.backend.database_path is None

    def test_memory_limit_means_file_backed(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "256MB")
        assert memory_limit_configured() is True
        context = _build(backend="duckdb")
        path = context.backend.database_path
        assert path is not None
        assert Path(path).exists()
        context.backend.close()
        assert not Path(path).exists()

    def test_explicit_true_overrides_the_absent_limit(self):
        context = _build(backend="duckdb", scratch_database=True)
        assert context.backend.database_path is not None
        context.backend.close()

    def test_explicit_false_overrides_the_present_limit(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "256MB")
        context = _build(backend="duckdb", scratch_database=False)
        assert context.backend.database_path is None

    def test_pandas_backend_is_unaffected(self, monkeypatch):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "256MB")
        context = _build(backend="pandas")
        assert getattr(context.backend, "database_path", None) is None

    def test_a_supplied_backend_instance_is_left_alone(self):
        backend = DuckDBBackend()
        context = _build(backend=backend, scratch_database=True)
        assert context.backend.database_path is None
        backend.close()

    def test_blank_memory_limit_does_not_count_as_configured(
        self, monkeypatch
    ):
        monkeypatch.setenv(MEMORY_LIMIT_ENV, "   ")
        assert memory_limit_configured() is False


class TestBehaviourUnchanged:
    def test_registry_works_on_a_file_backed_database(self):
        context = _build(backend="duckdb", scratch_database=True)
        try:
            assert context.backend.tables.has("Person")
            assert run_query(context, QUERY) == [["b"], ["c"]]
        finally:
            context.backend.close()

    @pytest.mark.parametrize("scratch", [True, False])
    def test_results_match_pandas_either_way(self, scratch):
        context = _build(backend="duckdb", scratch_database=scratch)
        try:
            assert run_query(context, QUERY) == run_query(
                build_context(ENTITIES, backend="pandas"), QUERY
            )
        finally:
            context.backend.close()
