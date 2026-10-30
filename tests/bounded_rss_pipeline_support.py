"""Support code for ``test_bounded_rss_streaming_acceptance.py``.

Not a test module itself (no ``test_*`` functions, so pytest does not collect
it) — a ``__main__`` dispatcher run as a subprocess, for the same reason
``tests/benchmarks/bench_duckdb_streaming_source.py`` runs each configuration
in its own process: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a
process-wide high-water mark that cannot be reset within one process, so
each pipeline run needs a fresh process to measure cleanly.

Writes two parquet sources (a ``Person`` entity and a ``KNOWS`` relationship)
sized so that loading ``Person`` eagerly costs real memory: each row carries
several low-cardinality text columns, which is exactly the shape
``bench_duckdb_streaming_source.py`` used to show the eager/streaming gap.
"""

from __future__ import annotations

import resource
import sys
from pathlib import Path

ROWS = 200_000
EXTRA_COLUMNS = 12
EDGES = ROWS * 2

PEOPLE_FILENAME = "people.parquet"
KNOWS_FILENAME = "knows.parquet"

CONFIG_TEMPLATE = """\
version: "1.0"

sources:
  entities:
    - id: people
      uri: file://{people_uri}
      entity_type: Person
      id_col: pid
      schema_hints:
        pid: BIGINT
        age: BIGINT
        city: VARCHAR
{extra_schema_hints}
  relationships:
    - id: knows
      uri: file://{knows_uri}
      relationship_type: KNOWS
      source_col: src
      target_col: dst
      source_entity_type: Person
      target_entity_type: Person
      schema_hints:
        src: BIGINT
        dst: BIGINT

queries:
  - id: count_known
    description: "Mutation + aggregation: how many people does each person know"
    inline: |
      MATCH (p:Person)-[:KNOWS]->(q:Person)
      WITH p, COUNT(q) AS num_known
      SET p.num_known = num_known
  - id: city_counts
    description: "Read-only aggregation with a sink"
    inline: |
      MATCH (p:Person)
      WITH p.city AS city, COUNT(p) AS n
      RETURN city, n
  - id: popularity
    description: "Read of the mutated attribute, with a sink"
    inline: |
      MATCH (p:Person)
      RETURN id(p) AS person_id, p.num_known AS num_known

output:
  - query_id: city_counts
    uri: file://{city_counts_uri}
  - query_id: popularity
    uri: file://{popularity_uri}

backend_engine: {backend_engine}
relation_engine: {relation_engine}
"""


def write_people(path: Path) -> None:
    """Write the synthetic ``Person`` source — the memory-heavy side."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "pid": np.arange(ROWS),
        "age": rng.integers(18, 90, ROWS),
        "city": rng.choice(["NY", "LA", "SF", "CHI", "ATL", "SEA"], ROWS),
    }
    for index in range(EXTRA_COLUMNS):
        data[f"extra_{index}"] = [f"v{i % 997}_{index}" for i in range(ROWS)]
    pd.DataFrame(data).to_parquet(path, index=False)


def write_knows(path: Path) -> None:
    """Write random directed ``KNOWS`` edges between people."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    src = rng.integers(0, ROWS, EDGES)
    dst = rng.integers(0, ROWS, EDGES)
    pd.DataFrame({"src": src, "dst": dst}).to_parquet(path, index=False)


def write_config(
    config_path: Path,
    *,
    dataset: tuple[Path, Path],
    output_dir: Path,
    backend_engine: str,
    relation_engine: bool,
) -> None:
    people_path, knows_path = dataset
    extra_schema_hints = "\n".join(
        f"        extra_{index}: VARCHAR" for index in range(EXTRA_COLUMNS)
    )
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            people_uri=people_path.resolve(),
            knows_uri=knows_path.resolve(),
            city_counts_uri=(output_dir / "city_counts.csv").resolve(),
            popularity_uri=(output_dir / "popularity.parquet").resolve(),
            backend_engine=backend_engine,
            relation_engine=str(relation_engine).lower(),
            extra_schema_hints=extra_schema_hints,
        ),
    )


def run_and_report_peak_rss_mb(config_path: Path) -> int:
    """Run the pipeline in *this* process and return peak RSS in MB."""
    from nmetl.cli.pipeline import run_impl

    run_impl(config_path, False, (), None, "fail", False)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def main() -> None:
    action = sys.argv[1]
    if action == "write-people":
        write_people(Path(sys.argv[2]))
    elif action == "write-knows":
        write_knows(Path(sys.argv[2]))
    elif action == "run":
        peak_mb = run_and_report_peak_rss_mb(Path(sys.argv[2]))
        print(f"PEAK_RSS_MB={peak_mb}")
    else:
        msg = f"unknown action: {action!r}"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
