"""Tests for pycypher.ingestion package __init__.py — re-exports and __all__.

Validates that:
- All symbols in __all__ are importable from pycypher.ingestion
- Key classes are the correct types
- Names that moved to the nmetl package fail with a pointer to their new home
- pycypher.ingestion never imports nmetl (the dependency is one-way)
"""

from __future__ import annotations

import pycypher.ingestion as ingestion_pkg
import pytest


class TestIngestionAllExports:
    """Every name in __all__ should be importable and non-None."""

    def test_all_is_defined(self):
        assert hasattr(ingestion_pkg, "__all__")
        assert len(ingestion_pkg.__all__) > 0

    @pytest.mark.parametrize("name", ingestion_pkg.__all__)
    def test_export_importable(self, name: str):
        obj = getattr(ingestion_pkg, name)
        assert obj is not None, f"{name} resolved to None"


class TestIngestionKeyClasses:
    """Spot-check that key re-exported classes are the expected types."""

    @pytest.mark.parametrize(
        "name",
        [
            "ContextBuilder",
            "CsvFormat",
            "ParquetFormat",
            "JsonFormat",
            "FileDataSource",
            "DataFrameDataSource",
            "ArrowDataSource",
            "SqlDataSource",
            "DuckDBReader",
            "OutputFormat",
        ],
    )
    def test_is_class(self, name: str):
        assert isinstance(getattr(ingestion_pkg, name), type)


class TestIngestionFunctions:
    """Re-exported functions should be callable."""

    def test_data_source_from_uri_is_callable(self):
        assert callable(ingestion_pkg.data_source_from_uri)

    def test_write_dataframe_to_uri_is_callable(self):
        assert callable(ingestion_pkg.write_dataframe_to_uri)


class TestIngestionSubmoduleConsistency:
    """Verify that re-exports point to the same objects as direct submodule imports."""

    def test_context_builder_same_object(self):
        from pycypher.ingestion.context_builder import ContextBuilder

        assert ingestion_pkg.ContextBuilder is ContextBuilder

    def test_csv_format_same_object(self):
        from pycypher.ingestion.data_sources import CsvFormat

        assert ingestion_pkg.CsvFormat is CsvFormat

    def test_duckdb_reader_same_object(self):
        from pycypher.ingestion.duckdb_reader import DuckDBReader

        assert ingestion_pkg.DuckDBReader is DuckDBReader

    def test_output_format_same_object(self):
        from pycypher.ingestion.output_writer import OutputFormat

        assert ingestion_pkg.OutputFormat is OutputFormat


class TestMovedToNmetl:
    """Pipeline-config names now live in nmetl; stale imports must say so."""

    @pytest.mark.parametrize(
        ("name", "new_home"),
        [
            ("PipelineConfig", "nmetl.config"),
            ("load_pipeline_config", "nmetl.config"),
            ("validate_config", "nmetl.validation"),
            ("PipelineBuilder", "nmetl.pipeline_builder"),
            ("DataSourceIntrospector", "nmetl.introspector"),
            ("DataSampler", "nmetl.data_preview"),
        ],
    )
    def test_stale_import_points_to_new_home(self, name: str, new_home: str):
        with pytest.raises(AttributeError, match=new_home):
            getattr(ingestion_pkg, name)
        assert name not in ingestion_pkg.__all__

    def test_unknown_name_is_a_plain_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute"):
            ingestion_pkg.definitely_not_a_thing  # noqa: B018

    def test_pycypher_does_not_import_nmetl(self):
        import subprocess
        import sys

        code = (
            "import sys, pycypher, pycypher.ingestion, pycypher.star, "
            "pycypher.relation_engine; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'nmetl'))"
        )
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == "[]"
