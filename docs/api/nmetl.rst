nmetl API
=========

``nmetl`` is the ETL layer on top of the pycypher engine.  It turns a YAML
pipeline file into a runnable job: data sources are registered into a
:class:`~pycypher.relational_models.Context`, each Cypher query is executed
(streamed through DuckDB where the relation engine can express it), and
results are written to the declared outputs.  The package also provides the
``nmetl`` command-line tool, the interactive REPL, the health server, and the
Neo4j sink.

The dependency is one-way: ``nmetl`` imports ``pycypher``; ``pycypher`` never
imports ``nmetl``.

Quick Start
-----------

.. code-block:: bash

   nmetl validate pipeline.yaml
   nmetl run pipeline.yaml
   nmetl repl --entity Person=people.csv

.. code-block:: python

   from nmetl import load_pipeline_config, validate_config

   config = load_pipeline_config("pipeline.yaml")
   result = validate_config(config)
   if not result.is_valid:
       for error in result.errors:
           print(error)

Package
-------

.. automodule:: nmetl
   :no-members:

Pipeline Configuration
----------------------

Config Models
~~~~~~~~~~~~~

Pydantic models for the pipeline YAML and :func:`~nmetl.config.load_pipeline_config`.

.. automodule:: nmetl.config
   :members:
   :undoc-members:
   :show-inheritance:

Validation
~~~~~~~~~~

Structured, categorised validation of a loaded config (missing files,
dangling query references, unsafe URIs, ...).

.. automodule:: nmetl.validation
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Builder
~~~~~~~~~~~~~~~~

Undoable in-memory editor over a :class:`~nmetl.config.PipelineConfig`.

.. automodule:: nmetl.pipeline_builder
   :members:
   :undoc-members:
   :show-inheritance:

Data Discovery
--------------

Introspector
~~~~~~~~~~~~

.. automodule:: nmetl.introspector
   :members:
   :undoc-members:
   :show-inheritance:

Data Preview
~~~~~~~~~~~~

Sampling strategies, column statistics, and ad-hoc query testing over a
data source before it is wired into a pipeline.

.. automodule:: nmetl.data_preview
   :members:
   :undoc-members:
   :show-inheritance:

Command-Line Interface
----------------------

The ``nmetl`` command.  :mod:`nmetl.nmetl_cli` builds the Click group and
error translation; the sub-commands live in :mod:`nmetl.cli`.

.. automodule:: nmetl.nmetl_cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: nmetl.cli.pipeline
   :members:
   :undoc-members:

.. automodule:: nmetl.cli.query
   :members:
   :undoc-members:

.. automodule:: nmetl.cli.schema
   :members:
   :undoc-members:

.. automodule:: nmetl.cli.security
   :members:
   :undoc-members:

.. automodule:: nmetl.cli.system
   :members:
   :undoc-members:

.. automodule:: nmetl.cli.common
   :members:
   :undoc-members:

REPL
~~~~

.. automodule:: nmetl.repl
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Health Server
~~~~~~~~~~~~~

Minimal HTTP endpoint backing ``nmetl health-server``; the Docker image's
``HEALTHCHECK`` uses ``nmetl health``.

.. automodule:: nmetl.health_server
   :members:
   :undoc-members:
   :show-inheritance:

Sinks
-----

Neo4j Sink
~~~~~~~~~~

Write query results to a Neo4j graph database using idempotent ``MERGE``
semantics.  Requires the ``neo4j`` driver (``pip install "nmetl[neo4j]"``).

.. automodule:: nmetl.sinks.neo4j
   :members:
   :undoc-members:
   :show-inheritance:
