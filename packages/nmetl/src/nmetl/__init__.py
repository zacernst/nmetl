"""nmetl: ETL pipelines and a command-line tool built on the pycypher engine.

``pycypher`` parses and executes Cypher against tabular data.  ``nmetl`` is
the layer above it that turns a YAML file into a runnable pipeline: it owns
the configuration models, config validation, the ``nmetl`` CLI, the
interactive REPL, the health server, and the Neo4j sink.  The dependency runs
one way only: ``nmetl`` imports ``pycypher``; ``pycypher`` never imports
``nmetl``.

Running a pipeline from Python::

    from nmetl import load_pipeline_config, validate_config

    config = load_pipeline_config("pipeline.yaml")
    result = validate_config(config)
    if not result.is_valid:
        for error in result.errors:
            print(error)

or from the shell::

    nmetl validate pipeline.yaml
    nmetl run pipeline.yaml

Package layout
--------------

* :mod:`nmetl.config` -- Pydantic models for the pipeline YAML
  (:class:`PipelineConfig`, data sources, queries, outputs) and
  :func:`load_pipeline_config`.
* :mod:`nmetl.validation` -- :func:`validate_config` and
  :class:`ValidationResult` (structured, categorised errors).
* :mod:`nmetl.pipeline_builder` -- :class:`PipelineBuilder`, an undoable
  in-memory editor over a :class:`PipelineConfig`.
* :mod:`nmetl.introspector` / :mod:`nmetl.data_preview` -- schema
  discovery, sampling, and query previews over a data source.
* :mod:`nmetl.nmetl_cli` and :mod:`nmetl.cli` -- the ``nmetl`` command and
  its sub-commands (``run``, ``validate``, ``query``, ``repl``, ``health``,
  ``security-check``, ...).
* :mod:`nmetl.repl` -- interactive Cypher shell over a
  :class:`~pycypher.star.Star`.
* :mod:`nmetl.health_server` -- minimal HTTP health/metrics endpoint.
* :mod:`nmetl.sinks.neo4j` -- write query results into Neo4j.
"""

from __future__ import annotations

from nmetl.config import PipelineConfig, load_pipeline_config
from nmetl.data_preview import (
    ColumnStats,
    DataSampler,
    PreviewCache,
    QueryResult,
    QueryTester,
    SamplingStrategy,
    SchemaInfo,
)
from nmetl.introspector import DataSourceIntrospector
from nmetl.pipeline_builder import (
    PipelineBuilder,
    PipelineOperation,
    PipelineSnapshot,
)
from nmetl.validation import (
    ValidationResult,
    validate_config,
    validate_config_dict,
)

__version__ = "0.0.1"

__all__ = [
    "ColumnStats",
    "DataSampler",
    "DataSourceIntrospector",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelineOperation",
    "PipelineSnapshot",
    "PreviewCache",
    "QueryResult",
    "QueryTester",
    "SamplingStrategy",
    "SchemaInfo",
    "ValidationResult",
    "load_pipeline_config",
    "validate_config",
    "validate_config_dict",
]
