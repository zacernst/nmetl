"""Test-session bootstrap for the nmetl package.

The nmetl tests exercise the CLI, pipeline config, sinks, and REPL on top of
the pycypher engine, so they reuse the engine test suite's session fixtures
(Spark/Neo4j connections, timing thresholds) and its isolation guards
instead of duplicating them.  Importing the fixture functions here registers
them for every test under ``packages/nmetl/tests``.
"""

from __future__ import annotations

import pytest

from tests.conftest import (  # noqa: F401
    _clear_pending_sigalrm,
    _restore_shared_logger_level,
    neo4j_driver,
    neo4j_session,
    perf_threshold,
    spark_session,
)


@pytest.fixture
def isolated_health_inputs(monkeypatch: pytest.MonkeyPatch):
    """Make health checks depend only on the test, not on the worker.

    Health status is computed from two process-global inputs:

    * ``shared.metrics.QUERY_METRICS`` -- every other test on the same
      xdist worker records queries, errors, and slow queries into it, so a
      fresh "no queries -> healthy" expectation is false unless the
      collector is reset first.
    * ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` -- the CLI flags RSS
      above 2 GB as degraded, and a pytest worker running the whole suite
      under coverage crosses that on CI runners.

    Both are neutralised here; the collector is reset again on teardown so
    the test's own activity does not leak to its neighbours either.
    """
    import resource
    from types import SimpleNamespace

    from shared.metrics import QUERY_METRICS

    real_getrusage = resource.getrusage

    def small_rss(who: int):
        usage = real_getrusage(who)
        # Linux reports KB; 100 MB keeps it far below the 2 GB threshold.
        return SimpleNamespace(ru_maxrss=100 * 1024, _real=usage)

    monkeypatch.setattr(resource, "getrusage", small_rss)
    QUERY_METRICS.reset()
    yield
    QUERY_METRICS.reset()
