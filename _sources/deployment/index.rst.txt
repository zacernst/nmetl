Deployment & Scaling
====================

This guide covers production deployment, container orchestration, scaling
strategies, monitoring, and operational best practices for PyCypher.

.. toctree::
   :maxdepth: 2
   :caption: Topics:

   security
   docker
   environment
   scaling
   monitoring
   troubleshooting


Architecture Overview
---------------------

PyCypher deployments consist of these components:

* **PyCypher core** -- the Cypher query engine (always required)
* **Spark cluster** -- optional distributed compute for large datasets

Query results can also be written to an external Neo4j instance via
``nmetl.sinks.neo4j.Neo4jSink`` (see :doc:`security`), but Neo4j itself is
not part of this repo's Docker Compose stack -- point the sink at your own
Neo4j deployment.

All components are containerised and orchestrated via Docker Compose.  The
Makefile provides convenience targets for every operational task.

Deployment Modes
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Mode
     - Command
     - What it starts
   * - Development
     - ``make dev-up``
     - ``pycypher-dev`` only (``--no-deps`` -- boots standalone)
   * - Infrastructure only
     - ``make infra-up``
     - Alias for ``spark-up`` -- Spark master + worker
   * - Spark cluster
     - ``make spark-up``
     - Spark master + worker(s)

Prerequisites
~~~~~~~~~~~~~

* **Docker** (via Docker Desktop or Rancher Desktop)
* **Python 3.14+** (for local development outside containers)
* **uv** package manager (``curl -LsSf https://astral.sh/uv/install.sh | sh``)
* **make** (standard on macOS/Linux)
