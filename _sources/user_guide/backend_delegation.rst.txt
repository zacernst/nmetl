Backend Delegation Guide
========================

PyCypher supports pluggable backend engines that execute relational operations.
This guide covers backend selection, performance characteristics, and when to
use each option.

.. contents:: On this page
   :local:
   :depth: 2

Available Backends
------------------

PyCypher ships with four backends:

Pandas (Default)
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pycypher.backend_engine import PandasBackend

   backend = PandasBackend()

- Zero overhead — thin wrapper around pandas operations
- Best for: small to medium datasets, prototyping, compatibility
- Memory model: eager evaluation, full DataFrame materialization

DuckDB
~~~~~~

.. code-block:: python

   from pycypher.backends.duckdb_backend import DuckDBBackend

   backend = DuckDBBackend()

- Lazy evaluation via ``DuckDBLazyFrame`` wrapping ``duckdb.Relation``
- SQL composition enables operation fusion (e.g. ORDER BY + LIMIT)
- Best for: analytical workloads, large datasets, columnar operations
- Memory model: lazy — materializes only when results are needed

Polars
~~~~~~

- Arrow-native lazy evaluation
- Best for: high-throughput columnar processing
- Experimental: passes the equivalence suite but round-trips pandas<->polars
  on every operation; not yet in the e2e acceptance matrix

Spark
~~~~~

.. code-block:: python

   from pycypher.backends.spark_backend import SparkBackend

   backend = SparkBackend()

- Distributed execution via PySpark; every op returns pandas (like DuckDB),
  materializing with ``.toPandas()`` after the set op runs on Spark
- Best for: distributed joins/aggregations across a Spark cluster
- Explicit-only: excluded from the ``auto`` fallback chain (JVM startup
  cost) -- must be requested with ``backend="spark"``. Requires a JVM.

BackendEngine Protocol
----------------------

All backends implement the ``BackendEngine`` protocol, which defines 16
primitive operations in five categories:

**Scan**

- ``scan_entity(source_obj, entity_type)`` — Load entity data from source

**Transform**

- ``filter(frame, mask)`` — Apply boolean mask filter
- ``join(frame_left, frame_right, on, how)`` — Join two frames
- ``rename(frame, columns)`` — Rename columns
- ``concat(frames, ignore_index)`` — Concatenate frames vertically
- ``distinct(frame)`` — Remove duplicate rows
- ``assign_column(frame, name, values)`` — Add or replace a column
- ``drop_columns(frame, columns)`` — Remove columns

**Aggregate**

- ``aggregate(frame, group_by, agg_funcs)`` — Group and aggregate

**Order**

- ``sort(frame, by, ascending)`` — Sort by columns
- ``limit(frame, n)`` — Take first n rows
- ``skip(frame, n)`` — Skip first n rows

**Materialize**

- ``to_pandas(frame)`` — Convert to pandas DataFrame
- ``row_count(frame)`` — Get row count without full materialization
- ``is_empty(frame)`` — Check if empty without materialization
- ``memory_estimate_bytes(frame)`` — Estimate memory usage

Choosing a Backend
------------------

======== ================ ================ ================ =================
Criteria Pandas           DuckDB           Polars           Spark
======== ================ ================ ================ =================
Dataset  < 100K rows      Any size         Any size         Very large / distributed
Eval     Eager            Lazy             Lazy             Distributed (set ops)
Memory   Full copies      Lazy references  Arrow buffers    Cluster-distributed
Joins    Hash join        Optimized SQL    Native merge     Distributed shuffle
Install  Included         Requires duckdb  Requires polars  Requires pyspark + JVM
======== ================ ================ ================ =================

**Use Pandas** when datasets fit in memory and you need maximum compatibility
with existing pandas-based code.

**Use DuckDB** when queries involve aggregations, sorts, or joins on larger
datasets — DuckDB's lazy evaluation avoids unnecessary intermediate
materialization. Pandas and DuckDB are the two production-ready backends,
covered by the equivalence and e2e acceptance suites and validated through
``nmetl run``.

**Use Polars** for Arrow-native processing pipelines where zero-copy
interoperability matters — but treat it as experimental until it lands in
the e2e acceptance matrix.

**Use Spark** only when the dataset genuinely needs distributed compute and
you're willing to pay JVM startup cost; it's never selected by ``auto``, so
request it explicitly.
