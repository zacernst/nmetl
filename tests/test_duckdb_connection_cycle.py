"""``DuckDBBackend.cycle_connection`` and ``TableRegistry.rebind``.

Ticket 86bcqr38m / 86bcqm58n (FastOpenData board): a long streaming
pipeline run keeps every entity/relationship table resident on one
DuckDB connection for the whole run, which OOMs once enough tables have
accumulated. ``DROP TABLE`` plus ``CHECKPOINT`` inside a live connection
gives nothing back to the OS; only closing the connection releases the
buffer-pool RSS. These tests cover the primitive that makes closing and
reopening the connection mid-run safe: the tables themselves live in the
file-backed database, not the connection, so they must survive a cycle
unchanged and remain queryable once ``rebind`` re-reads them.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pycypher.backend_engine import InstrumentedBackend
from pycypher.backends.duckdb_backend import (
    DuckDBBackend,
    create_scratch_database_path,
)
from pycypher.backends.table_registry import RELATIONSHIP_KIND


class TestCycleConnectionRequiresAFile:
    def test_in_memory_backend_raises(self):
        backend = DuckDBBackend()
        with pytest.raises(ValueError, match="file-backed"):
            backend.cycle_connection()
        backend.close()


class TestCycleConnectionOnAFileBackedBackend:
    @pytest.fixture
    def backend(self):
        path = create_scratch_database_path()
        backend = DuckDBBackend(database_path=path, own_database_file=True)
        try:
            yield backend
        finally:
            backend.close()

    def test_connection_object_is_replaced(self, backend):
        old_conn = backend.connection
        backend.cycle_connection()
        assert backend.connection is not old_conn

    def test_old_connection_is_closed(self, backend):
        old_conn = backend.connection
        backend.cycle_connection()
        with pytest.raises(Exception):
            old_conn.execute("SELECT 1")

    def test_registered_entity_table_survives_with_its_data(self, backend):
        backend.tables.register_source_object(
            "Person",
            pd.DataFrame({"pid": [1, 2, 3], "name": ["a", "b", "c"]}),
            id_col="pid",
        )
        backend.cycle_connection()
        entry = backend.tables.get("Person")
        assert entry.relation.fetchdf()["name"].tolist() == ["a", "b", "c"]

    def test_registered_relationship_table_survives_with_its_data(
        self, backend
    ):
        backend.tables.register_source_object(
            "KNOWS",
            pd.DataFrame({"__SOURCE__": [1], "__TARGET__": [2]}),
            kind=RELATIONSHIP_KIND,
        )
        backend.cycle_connection()
        entry = backend.tables.get("KNOWS", kind=RELATIONSHIP_KIND)
        assert entry.relation.fetchdf()["__TARGET__"].tolist() == [2]

    def test_multiple_registered_labels_all_survive_one_cycle(self, backend):
        backend.tables.register_source_object(
            "Person", pd.DataFrame({"pid": [1], "name": ["a"]}), id_col="pid"
        )
        backend.tables.register_source_object(
            "Place", pd.DataFrame({"pid": [1], "city": ["x"]}), id_col="pid"
        )
        backend.cycle_connection()
        assert backend.tables.get("Person").relation.fetchdf()[
            "name"
        ].tolist() == ["a"]
        assert backend.tables.get("Place").relation.fetchdf()[
            "city"
        ].tolist() == ["x"]

    def test_registering_a_new_label_after_a_cycle_still_works(self, backend):
        # The connection swap must leave the backend usable for ordinary
        # registration afterwards, not just for re-reading what came before.
        backend.cycle_connection()
        entry = backend.tables.register_source_object(
            "Person", pd.DataFrame({"pid": [1], "name": ["a"]}), id_col="pid"
        )
        assert entry.relation.fetchdf()["name"].tolist() == ["a"]

    def test_cycling_twice_in_a_row_still_preserves_data(self, backend):
        backend.tables.register_source_object(
            "Person", pd.DataFrame({"pid": [1], "name": ["a"]}), id_col="pid"
        )
        backend.cycle_connection()
        backend.cycle_connection()
        assert backend.tables.get("Person").relation.fetchdf()[
            "name"
        ].tolist() == ["a"]


class TestCycleConnectionThroughInstrumentedBackend:
    """``nmetl run -v`` wraps every backend in ``InstrumentedBackend``
    (``select_backend(..., instrument=True)``); cycling must survive that
    wrapper too, not just the raw ``DuckDBBackend``. Found running the real
    national pipeline with both ``-v`` and ``PYCYPHER_DUCKDB_CYCLE_EVERY``
    set: ``InstrumentedBackend`` forwarded ``connection``/``tables`` but not
    ``cycle_connection``, so the first cycle raised ``AttributeError``.
    """

    @pytest.fixture
    def backend(self):
        path = create_scratch_database_path()
        inner = DuckDBBackend(database_path=path, own_database_file=True)
        try:
            yield InstrumentedBackend(inner)
        finally:
            inner.close()

    def test_cycle_connection_is_available(self, backend):
        backend.cycle_connection()

    def test_registered_entity_table_survives_with_its_data(self, backend):
        backend.tables.register_source_object(
            "Person",
            pd.DataFrame({"pid": [1, 2, 3], "name": ["a", "b", "c"]}),
            id_col="pid",
        )
        backend.cycle_connection()
        entry = backend.tables.get("Person")
        assert entry.relation.fetchdf()["name"].tolist() == ["a", "b", "c"]


class TestRebindDirectly:
    """``TableRegistry.rebind`` in isolation, without going through the backend."""

    def test_rebind_points_every_label_at_the_new_connection(self):
        path = create_scratch_database_path()
        backend = DuckDBBackend(database_path=path, own_database_file=True)
        try:
            backend.tables.register_source_object(
                "Person", pd.DataFrame({"a": [1]})
            )
            backend.tables.register_source_object(
                "Rel", pd.DataFrame({"a": [1]}), kind=RELATIONSHIP_KIND
            )
            backend.connection.close()
            import duckdb

            new_conn = duckdb.connect(path)
            backend._conn = new_conn
            backend.tables.rebind(new_conn)

            assert backend.tables.get("Person").relation.fetchdf()[
                "a"
            ].tolist() == [1]
            assert backend.tables.get(
                "Rel", kind=RELATIONSHIP_KIND
            ).relation.fetchdf()["a"].tolist() == [1]
        finally:
            backend.close()
