Troubleshooting
===============

Common issues and their solutions when deploying and operating PyCypher.

Docker Issues
-------------

``docker compose up`` fails with "variable not set"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Required environment variables are missing.

**Fix**: Copy and populate the ``.env`` file:

.. code-block:: bash

   cp .env.example .env
   # Set SPARK_RPC_SECRET

Container starts but ``uv sync`` fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: uv cache is stale or corrupted.

**Fix**:

.. code-block:: bash

   # Clear the uv cache volume
   docker compose down
   docker volume rm pycypher-nmetl_uv-cache
   make dev-up

Port conflicts
~~~~~~~~~~~~~~

**Cause**: Another service is using the same port.

**Fix**: Check which ports are in use:

.. code-block:: bash

   # macOS / Linux
   lsof -i :7077   # Spark
   lsof -i :8090   # Spark UI

Stop the conflicting service or change the port mapping in
``docker-compose.yml``.

Python / Dependency Issues
--------------------------

``ModuleNotFoundError: No module named 'pycypher'``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Workspace not synced after changes.

**Fix**:

.. code-block:: bash

   uv sync
   uv run python -c "from pycypher import Star; print('OK')"

``Python >= 3.14 required``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: System Python is too old.

**Fix**: Install Python 3.14+ via ``uv``:

.. code-block:: bash

   uv python install 3.14

Backend failures
~~~~~~~~~~~~~~~~

**Cause**: Optional backend dependency not installed.

**Fix**:

.. code-block:: bash

   # Install DuckDB backend
   uv pip install duckdb

   # Install Polars backend
   uv pip install polars

   # Install Spark backend (also requires a JVM -- verified with OpenJDK 21)
   uv pip install pyspark

   # Verify
   uv run python -c "from pycypher.backend_engine import select_backend; print(select_backend(hint='pandas').name)"

Spark Issues
------------

Spark worker cannot connect to master
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: RPC secret mismatch or network issue.

**Fix**: Verify the secret is consistent:

.. code-block:: bash

   # Both master and worker must use the same SPARK_RPC_SECRET
   docker compose logs spark-master | grep -i "auth"
   docker compose logs spark-worker | grep -i "auth"

Spark job fails with OOM
~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Worker memory is insufficient.

**Fix**: Increase worker memory in ``docker-compose.yml``:

.. code-block:: yaml

   spark-worker:
     environment:
       - SPARK_WORKER_MEMORY=4G

Or scale out with more workers:

.. code-block:: bash

   make spark-scale WORKERS=4

Neo4j Sink Issues
-----------------

Neo4j is not part of this repo's Docker Compose stack -- there is no
``neo4j`` service, and no ``make neo4j-*`` targets.  ``Neo4jSink``
(``packages/nmetl/src/nmetl/sinks/neo4j.py``) writes to a Neo4j
instance you provide and manage yourself.

``Neo4jSink`` connection or authentication failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: The URI, credentials, or TLS setting passed to ``Neo4jSink`` don't
match your external Neo4j instance.

**Fix**: Verify connectivity independently of PyCypher first:

.. code-block:: bash

   # From the same network PyCypher runs in
   python -c "
   from neo4j import GraphDatabase
   GraphDatabase.driver('bolt://your-neo4j-host:7687', auth=('neo4j', 'password')).verify_connectivity()
   "

Running the Neo4j sink test suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: ``packages/nmetl/tests/test_neo4j_sink_integration.py`` and similar are marked
``neo4j`` and need a live instance to run against.

**Fix**: Point the tests at your own Neo4j instance and select the marker:

.. code-block:: bash

   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_PASSWORD=secret
   uv run pytest -m neo4j


Query Execution Issues
----------------------

Query hangs or runs too long
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fix**: Set a timeout:

.. code-block:: python

   result = star.execute_query("...", timeout_seconds=30)

Cross-product produces too many rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fix**: The cross-join limit prevents runaway cartesian products:

.. code-block:: python

   # Default limit applies; increase if needed
   result = star.execute_query("...", max_cross_join_rows=500_000)

``GraphTypeNotFoundError``
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Query references an entity or relationship type not in the context.

**Fix**: Check available types:

.. code-block:: python

   print(repr(context))
   # Context(backend='pandas', entities={'Person': 4}, relationships={'KNOWS': 3})

Getting Help
------------

* **Issue tracker**: Report bugs at the GitHub repository
* **Test suite**: Run ``uv run pytest -x`` to identify failures
* **Logs**: Check ``make dev-logs`` for container output
* **Metrics**: Use ``QUERY_METRICS.snapshot().diagnostic_report()`` for execution stats
