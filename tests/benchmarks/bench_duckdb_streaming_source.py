"""Peak-RSS comparison for streaming entity registration.

Run directly::

    uv run python tests/benchmarks/bench_duckdb_streaming_source.py

Each configuration runs in a subprocess, because ``ru_maxrss`` is a
process-wide high-water mark that cannot be reset.

This is the measurement that settles Phase 8 of
``docs/duckdb_eager_path_design.md``.  Its sibling,
``bench_duckdb_eager_path_memory.py``, shows that no amount of columnar
execution helps while ``add_entity`` reads the whole source into Arrow
first.  ``add_entity(..., streaming=True)`` removes that read, and the
profile changes completely.

Representative numbers (400k rows x 23 columns from parquet, one machine)::

    mode                  build     query     peak
    eager-pandas          156 MB    513 MB    794 MB
    eager-duckdb          245 MB    549 MB    920 MB
    streaming             178 MB     78 MB    382 MB
    streaming-limited     124 MB     69 MB    319 MB

(``streaming-limited`` sets ``PYCYPHER_DUCKDB_MEMORY_LIMIT=128MB``.)

Two things worth noting:

* The query itself drops from ~510 MB to well under 100 MB: with the source
  in DuckDB, ``MATCH ... WHERE ... RETURN count`` never builds a frame over
  the full table.
* A budget helps here, unlike in the eager case, because the table has a
  durable home to be evicted to. The load also *fails* under a tight budget
  if the normalisation SQL uses a window function — see
  ``streaming_entity._projection_sql``, which is deliberately window-free for
  that reason.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

ROWS = 400_000
EXTRA_COLUMNS = 20
QUERY = "MATCH (p:Person) WHERE p.age > 40 RETURN p.city, count(p)"
MODES = (
    "eager-pandas",
    "eager-duckdb",
    "streaming",
    "streaming-limited",
)
SOURCE = Path(tempfile.gettempdir()) / "pycypher-bench-streaming.parquet"


def _write_source() -> None:
    import numpy as np
    import pandas as pd

    if SOURCE.exists():
        return
    rng = np.random.default_rng(0)
    data = {
        "pid": np.arange(ROWS),
        "age": rng.integers(18, 90, ROWS),
        "city": rng.choice(["NY", "LA", "SF", "CHI"], ROWS),
    }
    for index in range(EXTRA_COLUMNS):
        data[f"attr{index}"] = [f"v{i % 997}" for i in range(ROWS)]
    pd.DataFrame(data).to_parquet(SOURCE, index=False)


def run_one(mode: str) -> None:
    """Measure one configuration and print a one-line result."""
    from pycypher.ingestion.context_builder import ContextBuilder
    from pycypher.star import Star

    def rss_mb() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    if mode == "streaming-limited":
        os.environ["PYCYPHER_DUCKDB_MEMORY_LIMIT"] = "128MB"

    before = rss_mb()
    builder = ContextBuilder()
    streaming = mode.startswith("streaming")
    builder = builder.add_entity(
        "Person", str(SOURCE), id_col="pid", streaming=streaming
    )
    backend = "pandas" if mode == "eager-pandas" else "duckdb"
    context = builder.build(backend=backend)
    after_build = rss_mb()
    Star(context=context).execute_query(QUERY)
    peak = rss_mb()
    print(
        f"{mode:20s} build_delta_mb={after_build - before:4d} "
        f"query_delta_mb={peak - after_build:4d} peak_mb={peak}",
    )


def _spawn(argument: str) -> None:
    subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(Path(__file__).resolve()), argument],
        check=True,
    )


def main() -> None:
    """Run every mode in its own subprocess and print a comparison table.

    The source file is written in a subprocess too.  ``ru_maxrss`` is a
    high-water mark that a forked child *inherits* from its parent, so a
    parent that had just built a 400k-row DataFrame would hand every child a
    ~700 MB floor and flatten the whole comparison.
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "--write-source":
            _write_source()
        else:
            run_one(sys.argv[1])
        return
    _spawn("--write-source")
    print(f"rows={ROWS} extra_columns={EXTRA_COLUMNS} source={SOURCE}")
    print(f"query: {QUERY}\n", flush=True)
    for mode in MODES:
        _spawn(mode)


if __name__ == "__main__":
    main()
