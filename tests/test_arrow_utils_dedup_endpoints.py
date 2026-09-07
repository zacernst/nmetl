"""Endpoint dedup stays exactly as it was, minus pandas.

Phase 1 of the FastOpenData pandas-free design (private repository).  ``_dedup_endpoints``
was the one place a fully-migrated fastopendata would still reach pandas,
via ``ContextBuilder.add_relationship``: it converted both key columns with
``to_pandas()`` purely to call ``DataFrame.duplicated``.

Replacing that with an Arrow hash-group is only safe if the replacement is
*indistinguishable*, so the central test here is differential — it asserts
the new implementation agrees with ``duplicated(keep="first")`` on generated
inputs rather than on a handful of hand-picked ones.  Three specific
behaviours get their own tests because getting any of them wrong would be
silent:

* nulls group together, matching pandas' treatment of NaN as equal to NaN;
* surviving rows keep input order, because the caller assigns sequential
  ``__ID__``s immediately afterwards and a permutation renumbers every edge;
* the warning text is unchanged, since it is the only signal a user gets
  that their edge table shrank.

pandas is imported here deliberately.  It is the oracle, not a dependency of
the code under test — which is the point of the last test in the file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pycypher.ingestion.arrow_utils import _dedup_endpoints

SETTINGS = settings(max_examples=200, deadline=None)

#: Endpoints drawn from a tiny alphabet so duplicates actually occur; a
#: strategy over arbitrary text would almost never generate a collision and
#: the differential test would silently cover nothing.
_endpoint = st.one_of(st.none(), st.sampled_from(["a", "b", "c"]))
_pairs = st.lists(st.tuples(_endpoint, _endpoint), max_size=40)


def _pandas_expectation(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Return the surviving pairs according to ``DataFrame.duplicated``."""
    frame = pd.DataFrame(
        {
            "__SOURCE__": [source for source, _ in pairs],
            "__TARGET__": [target for _, target in pairs],
        },
        dtype=object,
    )
    kept = frame[~frame.duplicated(subset=["__SOURCE__", "__TARGET__"], keep="first")]
    return list(zip(kept["__SOURCE__"], kept["__TARGET__"], strict=True))


def _actual(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Return the surviving pairs according to the Arrow implementation."""
    table = pa.table(
        {
            "__SOURCE__": pa.array([source for source, _ in pairs], type=pa.string()),
            "__TARGET__": pa.array([target for _, target in pairs], type=pa.string()),
        },
    )
    result = _dedup_endpoints(table)
    return list(
        zip(
            result.column("__SOURCE__").to_pylist(),
            result.column("__TARGET__").to_pylist(),
            strict=True,
        ),
    )


@SETTINGS
@given(pairs=_pairs)
def test_agrees_with_pandas_duplicated(pairs: list[tuple[Any, Any]]) -> None:
    """The Arrow implementation matches pandas on generated inputs.

    Compared as ordered lists, not sets: input order is part of the
    contract, so a result that contained the right pairs in the wrong order
    must fail here.
    """
    assert _actual(pairs) == _pandas_expectation(pairs)


def test_null_endpoints_collapse_together() -> None:
    """Rows with null endpoints are duplicates of each other, not distinct.

    Arrow's hash grouping and pandas' ``duplicated`` agree on this, but they
    agree by coincidence of design rather than by specification — SQL would
    say ``NULL != NULL`` and keep both rows. Pinned so a future move to a
    SQL-based dedup has to confront the difference.
    """
    table = pa.table(
        {
            "__SOURCE__": ["a", None, "a", None],
            "__TARGET__": ["x", None, "x", None],
            "payload": [1, 2, 3, 4],
        },
    )

    result = _dedup_endpoints(table)

    assert result.column("__SOURCE__").to_pylist() == ["a", None]
    assert result.column("__TARGET__").to_pylist() == ["x", None]
    assert result.column("payload").to_pylist() == [1, 2]


def test_keeps_first_occurrence_in_input_order() -> None:
    """Survivors appear in input order and carry the first row's attributes.

    The first occurrences here are at indices 0, 1, 4 — deliberately not the
    order a hash grouping would emit them in, so a missing sort shows up.
    """
    table = pa.table(
        {
            "__SOURCE__": ["c", "a", "c", "a", "b", "b"],
            "__TARGET__": ["z", "x", "z", "x", "y", "y"],
            "weight": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        },
    )

    result = _dedup_endpoints(table)

    assert result.column("__SOURCE__").to_pylist() == ["c", "a", "b"]
    assert result.column("weight").to_pylist() == [0.1, 0.2, 0.5]


def test_integer_endpoints_with_nulls() -> None:
    """Integer keys dedup without the float coercion pandas would apply.

    pandas turns an int column containing nulls into float64, so the keys it
    hashed were ``1.0``/``2.0``. Arrow keeps them as int64 with a null. The
    surviving *rows* are the same either way, which is what this asserts —
    but the endpoint values are now the integers the caller supplied rather
    than floats.
    """
    table = pa.table(
        {
            "__SOURCE__": pa.array([1, None, 1, 2, None], type=pa.int64()),
            "__TARGET__": pa.array([10, None, 10, 20, None], type=pa.int64()),
        },
    )

    result = _dedup_endpoints(table)

    assert result.column("__SOURCE__").to_pylist() == [1, None, 2]
    assert result.column("__TARGET__").to_pylist() == [10, None, 20]
    assert result.column("__SOURCE__").type == pa.int64()


def test_no_duplicates_returns_input_unchanged() -> None:
    """The early exit still fires, and no warning is emitted."""
    table = pa.table({"__SOURCE__": ["a", "b"], "__TARGET__": ["x", "y"]})

    assert _dedup_endpoints(table) is table


def test_warning_text_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warning template and its arguments are byte-for-byte the same.

    Asserted against the format string rather than the rendered record, so
    the test fails on a reworded message even if the numbers still come out
    right. This is a user-facing string that names a config key; changing it
    silently would strand anyone searching for it.
    """
    from pycypher.ingestion import arrow_utils

    captured: list[tuple[Any, ...]] = []

    class _Recorder:
        def warning(self, *args: Any) -> None:
            captured.append(args)

    monkeypatch.setattr(arrow_utils, "LOGGER", _Recorder())

    table = pa.table(
        {
            "__SOURCE__": ["a", "a", "b"],
            "__TARGET__": ["x", "x", "y"],
        },
    )
    arrow_utils._dedup_endpoints(table)

    assert captured == [
        (
            (
                "normalize_relationship_table: collapsed %d duplicate "
                "(__SOURCE__, __TARGET__) edges (%d → %d). "
                "If parallel edges are intentional, set `allow_multi_edges: true` "
                "on the relationship source."
            ),
            1,
            3,
            2,
        ),
    ]


#: Runs in a subprocess so the block cannot disturb this session's
#: ``sys.modules``. Blocks pandas at ``sys.meta_path``, then loads
#: ``arrow_utils`` **from its file** and exercises the full
#: relationship-normalisation path including endpoint dedup.
#:
#: Loading by path rather than ``import pycypher.ingestion.arrow_utils``
#: is not a dodge — it is the only way to test this module. Importing it
#: normally runs ``pycypher.ingestion.__init__``, which reaches
#: ``data_sources.py`` and its module-scope ``import pandas``. That is a
#: real blocker but a different one: 49 pycypher modules import pandas at
#: module scope, and porting them is not in scope for the fastopendata
#: plan. See "Reachability beyond fastopendata" in the design doc.
_NO_PANDAS_SCRIPT = """
import importlib.util
import pathlib
import sys

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "pandas":
            raise ImportError("pandas is blocked")

for name in [n for n in sys.modules if n.split(".", 1)[0] == "pandas"]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

import pyarrow as pa

source = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("arrow_utils_isolated", source)
arrow_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arrow_utils)

table = pa.table({"src": ["a", "a", "b"], "tgt": ["x", "x", "y"]})
result = arrow_utils.normalize_relationship_table(
    table, source_col="src", target_col="tgt"
)
assert result.num_rows == 2, result.num_rows
assert result.column("__ID__").to_pylist() == [0, 1]
assert result.column("__SOURCE__").to_pylist() == ["a", "b"]
assert "pandas" not in sys.modules
print("ok")
"""

ARROW_UTILS_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "packages"
    / "pycypher"
    / "src"
    / "pycypher"
    / "ingestion"
    / "arrow_utils.py"
)


@pytest.mark.timeout(120)
def test_normalization_needs_no_pandas() -> None:
    """``arrow_utils`` normalises relationships with pandas blocked.

    This is what Phase 1 exists to achieve, and it is the assertion that
    would have failed before the change: ``to_pandas()`` triggers a lazy
    ``import pandas`` inside pyarrow, so there was no module-level import to
    delete — the dependency was invisible in the import list and only showed
    up when a relationship table actually contained a duplicate.
    """
    completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", _NO_PANDAS_SCRIPT, str(ARROW_UTILS_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"normalisation still reaches pandas:\n{completed.stderr}"
    )
    assert completed.stdout.strip().endswith("ok")
