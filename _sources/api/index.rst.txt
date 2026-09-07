API Reference
=============

Complete API reference for all PyCypher packages.  Each module's public
classes, functions, and constants are documented with type signatures,
parameter descriptions, return values, and usage examples extracted from
source docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Package APIs:

   pycypher
   nmetl
   shared

Package Overview
----------------

PyCypher
~~~~~~~~

The engine — Cypher query parsing, AST processing, BindingFrame and
DuckDB relation execution, and the ingestion layer that loads tabular data
into a query context.

**Entry points:**

* :class:`~pycypher.star.Star` — execute Cypher queries against DataFrames
* :class:`~pycypher.ingestion.context_builder.ContextBuilder` — build graph contexts from tabular data
* :func:`~pycypher.semantic_validator.validate_query` — pre-execution query validation

**Execution pipeline:**

* ``grammar_parser`` / ``grammar_transformers`` — Cypher → Lark → AST
* ``ast_models`` / ``ast_converter`` / ``ast_rewriter`` — immutable Pydantic AST, conversion, and rewriting
* ``pattern_matcher`` / ``path_expander`` — MATCH clause evaluation
* ``binding_frame`` / ``binding_evaluator`` — vectorised expression execution
* ``pipeline`` — stage-based parse → validate → plan → execute pipeline
* ``backend_engine`` — pluggable backends (Pandas, DuckDB, Polars)
* ``query_planner`` / ``query_optimizer`` / ``query_profiler`` — planning, optimization, and profiling

**Evaluators:**

* ``arithmetic_evaluator`` / ``boolean_evaluator`` / ``comparison_evaluator`` — operator evaluation
* ``collection_evaluator`` / ``string_predicate_evaluator`` / ``exists_evaluator`` — complex expressions
* ``aggregation_evaluator`` / ``aggregation_planner`` — grouped and full-table aggregation
* ``scalar_function_evaluator`` / ``scalar_functions`` — 131 built-in scalar functions

**Multi-query composition:**

* ``multi_query_analyzer`` / ``multi_query_executor`` — dependency-aware ETL pipelines

**Write path:**

* ``mutation_engine`` — CREATE, SET, DELETE, MERGE, FOREACH
* ``ingestion.output_writer`` — write result frames to CSV, Parquet, or JSON

nmetl
~~~~~

The ETL layer on top of pycypher: YAML pipeline configuration, validation,
the ``nmetl`` command-line tool, the REPL, the health server, and the
Neo4j sink.  Depends on pycypher; pycypher never imports it.

**Entry points:**

* ``nmetl run pipeline.yaml`` — the CLI (:mod:`nmetl.nmetl_cli`, sub-commands in :mod:`nmetl.cli`)
* :func:`~nmetl.config.load_pipeline_config` / :class:`~nmetl.config.PipelineConfig` — the pipeline YAML model
* :func:`~nmetl.validation.validate_config` — structured config validation

**Modules:**

* ``config`` / ``validation`` / ``pipeline_builder`` — configuration model, checks, and an undoable editor
* ``introspector`` / ``data_preview`` — schema discovery and sampling for data sources
* ``repl`` / ``health_server`` — interactive shell and HTTP health endpoint
* ``sinks.neo4j`` — write results to Neo4j

Shared
~~~~~~

Common utilities: structured logging, query metrics, serialisation helpers,
observability, and compatibility checking.

* ``logger`` — ``LOGGER`` singleton for structured output
* ``metrics`` — ``QUERY_METRICS`` collector and ``MetricsSnapshot``
* ``helpers`` — serialisation and utility functions
* ``telemetry`` — optional Pyroscope profiling integration
* ``otel`` — OpenTelemetry tracing (no-op fallback when not installed)
* ``exporters`` — Prometheus, StatsD, and JSON metrics export
* ``compat`` — API surface snapshot and backward-compatibility diffing
* ``deprecation`` — structured deprecation warnings
