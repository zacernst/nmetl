"""Ingestion layer for loading external data into pycypher.

Provides Arrow (via PyArrow) as the canonical in-memory tabular format and
DuckDB as the universal ingestion adapter.  Everything here is engine-level:
it is what :class:`~pycypher.star.Star` and the relation engine need to read
tabular data into a :class:`~pycypher.relational_models.Context` and to write
results back out.  The YAML pipeline configuration, config validation,
pipeline builder, and data-preview helpers live in the ``nmetl`` package,
which depends on this one.

Data Sources
------------

Load data from files, DataFrames, Arrow tables, or SQL databases::

    from pycypher.ingestion import (
        FileDataSource, CsvFormat, ParquetFormat,
        DataFrameDataSource, SqlDataSource, data_source_from_uri,
    )

    # From a CSV file
    source = FileDataSource("people.csv", format=CsvFormat())

    # From a Parquet file
    source = FileDataSource("people.parquet", format=ParquetFormat())

    # Auto-detect format from URI
    source = data_source_from_uri("data/people.csv")

    # From an existing pandas DataFrame
    source = DataFrameDataSource(df)

    # From a SQL database
    source = SqlDataSource("sqlite:///mydb.db", query="SELECT * FROM people")

Context Building
----------------

Use :class:`ContextBuilder` to assemble a query context from data sources::

    from pycypher.ingestion import ContextBuilder
    from pycypher import Star

    context = (
        ContextBuilder()
        .add_entity("Person", FileDataSource("people.csv", format=CsvFormat()))
        .add_relationship("KNOWS", knows_df,
                          source_col="__SOURCE__", target_col="__TARGET__")
        .build()
    )
    star = Star(context=context)

Writing Results
---------------

:func:`write_dataframe_to_uri` writes a result DataFrame to a local path or
``file://`` URI, inferring the format from the extension unless an explicit
:class:`OutputFormat` is given::

    from pycypher.ingestion import write_dataframe_to_uri, OutputFormat

    write_dataframe_to_uri(result, "out/people.parquet")
    write_dataframe_to_uri(result, "out/people.dat", fmt=OutputFormat.CSV)

Submodules
----------

* :mod:`~pycypher.ingestion.data_sources` -- ``DataSource`` implementations
  and URI dispatch.
* :mod:`~pycypher.ingestion.context_builder` -- ``ContextBuilder``.
* :mod:`~pycypher.ingestion.streaming_entity` -- registry-backed entities for
  the DuckDB out-of-core path.
* :mod:`~pycypher.ingestion.duckdb_reader` -- DuckDB-backed file reader.
* :mod:`~pycypher.ingestion.arrow_utils` -- Arrow schema helpers.
* :mod:`~pycypher.ingestion.security` -- URI, path, and SQL identifier
  sanitisation shared by every reader and writer.
* :mod:`~pycypher.ingestion.output_writer` -- result writers.
"""

from __future__ import annotations

from pycypher.ingestion.context_builder import ContextBuilder
from pycypher.ingestion.data_sources import (
    ArrowDataSource,
    CsvFormat,
    DataFrameDataSource,
    DataSource,
    FileDataSource,
    Format,
    JsonFormat,
    ParquetFormat,
    SqlDataSource,
    data_source_from_uri,
)
from pycypher.ingestion.duckdb_reader import DuckDBReader
from pycypher.ingestion.output_writer import (
    OutputFormat,
    write_dataframe_to_uri,
)

__all__ = [
    "ArrowDataSource",
    "ContextBuilder",
    "CsvFormat",
    "DataFrameDataSource",
    "DataSource",
    "DuckDBReader",
    "FileDataSource",
    "Format",
    "JsonFormat",
    "OutputFormat",
    "ParquetFormat",
    "SqlDataSource",
    "data_source_from_uri",
    "write_dataframe_to_uri",
]

# Names that used to be re-exported here and now live in the nmetl package.
# Kept only so that a stale import fails with a pointer instead of a bare
# AttributeError; pycypher itself must never import nmetl.
_MOVED_TO_NMETL: dict[str, str] = {
    "PipelineConfig": "nmetl.config",
    "load_pipeline_config": "nmetl.config",
    "ValidationResult": "nmetl.validation",
    "validate_config": "nmetl.validation",
    "validate_config_dict": "nmetl.validation",
    "PipelineBuilder": "nmetl.pipeline_builder",
    "PipelineOperation": "nmetl.pipeline_builder",
    "PipelineSnapshot": "nmetl.pipeline_builder",
    "DataSourceIntrospector": "nmetl.introspector",
    "ColumnStats": "nmetl.data_preview",
    "DataSampler": "nmetl.data_preview",
    "PreviewCache": "nmetl.data_preview",
    "QueryResult": "nmetl.data_preview",
    "QueryTester": "nmetl.data_preview",
    "SamplingStrategy": "nmetl.data_preview",
    "SchemaInfo": "nmetl.data_preview",
}


def __getattr__(name: str) -> object:
    """Point stale imports of pipeline-config names at their new home."""
    if name in _MOVED_TO_NMETL:
        msg = (
            f"{name!r} moved out of pycypher.ingestion; "
            f"import it from {_MOVED_TO_NMETL[name]} (package 'nmetl')."
        )
        raise AttributeError(msg)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
