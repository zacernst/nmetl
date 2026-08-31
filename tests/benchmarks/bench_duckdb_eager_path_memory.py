"""Peak-RSS comparison for the DuckDB eager path (Phase 8).

Run directly::

    uv run python tests/benchmarks/bench_duckdb_eager_path_memory.py

Each configuration runs in a **subprocess**, because ``ru_maxrss`` is a
high-water mark for the whole process and cannot be reset: measuring two
backends in one process would report the max of both.

Why this is a benchmark and not a passing test
----------------------------------------------

``docs/duckdb_eager_path_design.md`` Phase 8 asks for an acceptance test
proving bounded RSS on a dataset larger than a constrained memory limit.
**That goal is not reachable through the eager path, and this script is the
evidence.**

``ContextBuilder.add_entity`` calls ``DataSource.read()``, which materialises
the entire source into an Arrow table before any query runs.
``register_context_tables`` then copies it again into DuckDB.  With an
``:memory:`` database — which ``ContextBuilder.build`` creates — that second
copy is also RAM, so peak RSS goes *up*, not down.  Phases 3-7 make execution
columnar; they cannot undo an eager source load that happens before they are
reached.

Reaching the Phase 8 goal needs the source to never enter pandas at all:
``add_entity`` would have to register file-backed sources through
``DataSource.read_relation`` (a streaming scan) instead of ``read()``, and
the backend would need a file-backed scratch database.  Both already exist
for the relation engine — see ``register_streaming_source`` and
``create_scratch_database_path`` — but are not wired into ``ContextBuilder``.

Representative numbers (200k rows x 22 columns, one machine)::

    pandas               query_delta_mb= 40   peak_mb=535
    duckdb-noreg         query_delta_mb= 40   peak_mb=536
    duckdb-mem           query_delta_mb=115   peak_mb=610
    duckdb-mem-limited   query_delta_mb=109   peak_mb=603
    duckdb-file          query_delta_mb=125   peak_mb=622
    duckdb-file-limited  query_delta_mb= 94   peak_mb=590

Three things worth reading off that table:

* ``duckdb-noreg`` (registry disabled via ``register_tables=False``) measures
  identical to ``pandas``: the fallback is unchanged and the whole feature has
  a working kill switch.
* **A file-backed database alone is worse than ``:memory:``** (125 vs 115).
  It only pays off with a memory budget (94), because DuckDB will not evict
  buffer-pool pages until told a limit, and can only evict somewhere durable.
  Neither half works alone — which is why ``ContextBuilder.build`` creates the
  scratch file exactly when ``PYCYPHER_DUCKDB_MEMORY_LIMIT`` is set.
* Even the best configuration is above ``pandas``, because the pandas/Arrow
  source load dominates everything the query does.

"""

from __future__ import annotations

import resource
import subprocess
import sys
from pathlib import Path

ROWS = 200_000
EXTRA_COLUMNS = 20
QUERY = "MATCH (p:Person) WHERE p.age > 40 RETURN p.city, count(p)"
MODES = (
    "pandas",
    "duckdb-noreg",
    "duckdb-mem",
    "duckdb-mem-limited",
    "duckdb-file",
    "duckdb-file-limited",
)


def _build_frame(rows: int, extra_columns: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data = {
        "age": rng.integers(18, 90, rows),
        "city": rng.choice(["NY", "LA", "SF", "CHI"], rows),
    }
    for index in range(extra_columns):
        data[f"attr{index}"] = [f"v{i % 997}" for i in range(rows)]
    return pd.DataFrame(data)


def _build_context(mode: str, frame):
    from pycypher.backends.duckdb_backend import (
        DuckDBBackend,
        create_scratch_database_path,
    )
    from pycypher.ingestion.context_builder import ContextBuilder

    builder = ContextBuilder().add_entity("Person", frame)
    if mode == "pandas":
        return builder.build(backend="pandas")
    if mode == "duckdb-noreg":
        return builder.build(
            backend="duckdb", register_tables=False, scratch_database=False
        )
    if mode == "duckdb-mem":
        return builder.build(backend="duckdb", scratch_database=False)
    if mode == "duckdb-mem-limited":
        return builder.build(
            backend=DuckDBBackend(memory_limit="256MB"),
        )
    if mode == "duckdb-file":
        # The current default for backend="duckdb".
        return builder.build(backend="duckdb")
    if mode == "duckdb-file-limited":
        return builder.build(
            backend=DuckDBBackend(
                database_path=create_scratch_database_path(),
                own_database_file=True,
                memory_limit="256MB",
            ),
        )
    msg = f"Unknown mode {mode!r}"
    raise ValueError(msg)


def run_one(mode: str, rows: int, extra_columns: int) -> None:
    """Measure one configuration and print a one-line result."""
    from pycypher.star import Star

    def rss_mb() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    before = rss_mb()
    frame = _build_frame(rows, extra_columns)
    context = _build_context(mode, frame)
    after_build = rss_mb()
    Star(context=context).execute_query(QUERY)
    peak = rss_mb()
    print(
        f"{mode:14s} build_delta_mb={after_build - before:4d} "
        f"query_delta_mb={peak - after_build:4d} peak_mb={peak}",
    )


def main() -> None:
    """Run every mode in its own subprocess and print a comparison table."""
    if len(sys.argv) > 1:
        run_one(sys.argv[1], ROWS, EXTRA_COLUMNS)
        return
    print(f"rows={ROWS} extra_columns={EXTRA_COLUMNS}")
    print(f"query: {QUERY}\n")
    for mode in MODES:
        subprocess.run(  # noqa: S603 — fixed argv, no shell
            [sys.executable, str(Path(__file__).resolve()), mode],
            check=True,
        )


if __name__ == "__main__":
    main()
