"""Phase 5 (``docs/duckdb_full_parity_design.md:217-224``) — bounded RSS
over a full pipeline run.

No test exercised bounded memory on a full pipeline run before this one —
only benchmarks (``tests/benchmarks/bench_duckdb_streaming_source.py``,
measuring a single read query) and ``test_duckdb_kill_switch.py`` (whose
cases are scan/pushdown-decline assertions with no RSS assertion at all).

This test runs one pipeline — a ``Person``/``KNOWS`` dataset, a
``MATCH``+aggregation+``SET`` mutation, a read with an aggregation sink, and
a read of the mutated attribute with its own sink — twice:

* once through the out-of-core streaming path (``backend_engine: duckdb``,
  ``relation_engine: true``) under a tight ``PYCYPHER_DUCKDB_MEMORY_LIMIT``,
  asserting peak RSS stays under ``BOUND_MB``;
* once through the eager in-memory path (``backend_engine: pandas``, no
  memory limit in effect), asserting peak RSS clears ``BOUND_MB`` — so the
  first assertion is a real bound and not a number nothing could exceed
  anyway. Each run happens in its own subprocess: ``ru_maxrss`` is a
  process-wide high-water mark that cannot be reset within one process (see
  ``bounded_rss_pipeline_support.py`` and
  ``bench_duckdb_streaming_source.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from bounded_rss_pipeline_support import (
    KNOWS_FILENAME,
    PEOPLE_FILENAME,
    write_config,
)

SUPPORT_MODULE = Path(__file__).parent / "bounded_rss_pipeline_support.py"

#: The streaming run, under a tight PYCYPHER_DUCKDB_MEMORY_LIMIT, must stay
#: under this; the eager run, with no limit, must clear it. Chosen from
#: measured peaks at this dataset size (streaming ~374 MB with a 128MB
#: DuckDB budget, eager ~673 MB) with headroom on both sides rather than
#: pinned to either run's exact number.
BOUND_MB = 500

pytestmark = pytest.mark.slow


def _spawn(*args: str) -> str:
    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(SUPPORT_MODULE), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _peak_rss_mb(stdout: str) -> int:
    for line in stdout.splitlines():
        if line.startswith("PEAK_RSS_MB="):
            return int(line.removeprefix("PEAK_RSS_MB="))
    msg = f"no PEAK_RSS_MB line in subprocess output: {stdout!r}"
    raise AssertionError(msg)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Write the Person/KNOWS sources once, each in its own subprocess.

    A subprocess that had just built these DataFrames would hand every
    later child a memory floor via fork inheritance — exactly the reason
    ``bench_duckdb_streaming_source.py`` writes its source the same way.
    """
    data_dir = tmp_path_factory.mktemp("bounded-rss-data")
    people_path = data_dir / PEOPLE_FILENAME
    knows_path = data_dir / KNOWS_FILENAME
    _spawn("write-people", str(people_path))
    _spawn("write-knows", str(knows_path))
    return people_path, knows_path


def _run_pipeline(
    tmp_path: Path,
    dataset: tuple[Path, Path],
    *,
    backend_engine: str,
    relation_engine: bool,
    memory_limit: str | None,
) -> int:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config_path = tmp_path / "pipeline.yaml"
    write_config(
        config_path,
        dataset=dataset,
        output_dir=output_dir,
        backend_engine=backend_engine,
        relation_engine=relation_engine,
    )

    result = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, str(SUPPORT_MODULE), "run", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
        env=_env_with_memory_limit(memory_limit),
    )
    peak_mb = _peak_rss_mb(result.stdout)

    assert (output_dir / "city_counts.csv").exists()
    assert (output_dir / "popularity.parquet").exists()
    return peak_mb


def _env_with_memory_limit(memory_limit: str | None) -> dict[str, str]:
    import os

    env = dict(os.environ)
    if memory_limit is None:
        env.pop("PYCYPHER_DUCKDB_MEMORY_LIMIT", None)
    else:
        env["PYCYPHER_DUCKDB_MEMORY_LIMIT"] = memory_limit
    return env


class TestBoundedRssOverAFullPipelineRun:
    def test_streaming_run_stays_under_the_bound(
        self, tmp_path: Path, dataset: tuple[Path, Path]
    ) -> None:
        peak_mb = _run_pipeline(
            tmp_path,
            dataset,
            backend_engine="duckdb",
            relation_engine=True,
            memory_limit="128MB",
        )

        assert peak_mb < BOUND_MB, (
            f"streaming run peaked at {peak_mb} MB, expected under "
            f"{BOUND_MB} MB with PYCYPHER_DUCKDB_MEMORY_LIMIT=128MB"
        )

    def test_the_bound_is_not_vacuous_without_streaming(
        self, tmp_path: Path, dataset: tuple[Path, Path]
    ) -> None:
        """Same dataset and queries, eager path, no memory limit in effect.

        If this also stayed under ``BOUND_MB`` the bound above would be
        proving nothing — this is the test failing (confirming the gate
        matters) when the limit/engine that makes it hold is removed.
        """
        peak_mb = _run_pipeline(
            tmp_path,
            dataset,
            backend_engine="pandas",
            relation_engine=False,
            memory_limit=None,
        )

        assert peak_mb >= BOUND_MB, (
            f"eager run peaked at {peak_mb} MB, expected at or above "
            f"{BOUND_MB} MB — if it stayed under, the streaming bound "
            "above is not actually testing anything"
        )
