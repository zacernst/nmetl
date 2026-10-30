"""Phase 5 (``docs/duckdb_full_parity_design.md:217-224``) — bounded RSS
over a full pipeline run.

No test exercised bounded memory on a full pipeline run before this one —
only benchmarks (``tests/benchmarks/bench_duckdb_streaming_source.py``,
measuring a single read query) and ``test_duckdb_kill_switch.py`` (whose
cases are scan/pushdown-decline assertions with no RSS assertion at all).

This test runs one pipeline — a ``Person``/``KNOWS`` dataset, a
``MATCH``+aggregation+``SET`` mutation, a read with an aggregation sink, and
a read of the mutated attribute with its own sink — twice: once through the
out-of-core streaming path (``backend_engine: duckdb``, ``relation_engine:
true``) under a tight ``PYCYPHER_DUCKDB_MEMORY_LIMIT``, once through the
eager in-memory path (``backend_engine: pandas``, no memory limit in
effect). Each run happens in its own subprocess: ``ru_maxrss`` is a
process-wide high-water mark that cannot be reset within one process (see
``bounded_rss_pipeline_support.py`` and
``bench_duckdb_streaming_source.py``).

The bound is comparative, not a fixed number of MB: a first version pinned
an absolute ceiling (500 MB) and it was flaky across machines — the same
config peaked at 374 MB on a dev box and 532 MB on a CI runner, because a
process's baseline RSS (interpreter, venv, imports) varies by machine. That
baseline shifts both the streaming and eager runs by about the same amount,
so comparing the two peaks *within the same run* cancels it out.
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

#: Loose backstop against a genuine leak/explosion. Not tuned to any
#: particular machine's baseline RSS — this dataset needs nowhere near it
#: even loaded eagerly.
ABSOLUTE_CEILING_MB = 1500

#: How much smaller the streaming run's peak must be than the eager run's,
#: in MB. Chosen with headroom under the measured savings (~300 MB on a dev
#: box, ~300+ MB on CI) rather than pinned to either machine's number.
MIN_SAVINGS_MB = 150

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


@pytest.fixture(scope="module")
def peaks(
    tmp_path_factory: pytest.TempPathFactory, dataset: tuple[Path, Path]
) -> tuple[int, int]:
    """Peak RSS in MB for the streaming and eager runs of the same pipeline."""
    base = tmp_path_factory.mktemp("bounded-rss-runs")
    streaming_dir = base / "streaming"
    streaming_dir.mkdir()
    eager_dir = base / "eager"
    eager_dir.mkdir()

    streaming_peak = _run_pipeline(
        streaming_dir,
        dataset,
        backend_engine="duckdb",
        relation_engine=True,
        memory_limit="128MB",
    )
    eager_peak = _run_pipeline(
        eager_dir,
        dataset,
        backend_engine="pandas",
        relation_engine=False,
        memory_limit=None,
    )
    return streaming_peak, eager_peak


class TestBoundedRssOverAFullPipelineRun:
    def test_streaming_run_stays_under_a_loose_absolute_ceiling(
        self, peaks: tuple[int, int]
    ) -> None:
        streaming_peak, _eager_peak = peaks

        assert streaming_peak < ABSOLUTE_CEILING_MB, (
            f"streaming run peaked at {streaming_peak} MB, expected under "
            f"{ABSOLUTE_CEILING_MB} MB — this dataset is not large enough "
            "to need anywhere close to that much even loaded eagerly"
        )

    def test_streaming_uses_meaningfully_less_memory_than_eager(
        self, peaks: tuple[int, int]
    ) -> None:
        """The real bound — comparative, not a fixed number.

        Confirms the gate matters: if the memory limit/streaming engine
        were removed (making this run identical to the eager one), the
        savings would collapse to ~0 and this would fail.
        """
        streaming_peak, eager_peak = peaks

        assert eager_peak - streaming_peak >= MIN_SAVINGS_MB, (
            f"streaming peaked at {streaming_peak} MB and eager at "
            f"{eager_peak} MB — expected streaming to save at least "
            f"{MIN_SAVINGS_MB} MB over eager; if it didn't, the memory "
            "limit/streaming engine isn't doing anything"
        )
