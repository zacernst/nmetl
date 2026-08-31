"""Shared helpers for pandas-vs-DuckDB differential testing.

The DuckDB eager path is only correct if it is *indistinguishable* from the
pandas path for every query it handles.  These helpers build the same graph
on both backends and compare results, so each phase of
``docs/duckdb_eager_path_design.md`` can assert equivalence rather than just
"it ran".

Not named ``test_*`` so pytest does not collect it as a test module.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BACKENDS = ("pandas", "duckdb")


def build_context(
    entities: dict[str, pd.DataFrame],
    relationships: dict[str, tuple[pd.DataFrame, str, str]] | None = None,
    *,
    backend: str = "pandas",
) -> Any:
    """Build a Context over *entities* and *relationships* for *backend*.

    Args:
        entities: label → DataFrame of node rows.
        relationships: label → ``(DataFrame, source_col, target_col)``.
        backend: Backend hint passed to ``ContextBuilder.build``.

    Returns:
        The assembled ``Context``.

    """
    from pycypher.ingestion.context_builder import ContextBuilder

    builder = ContextBuilder()
    for label, frame in entities.items():
        builder = builder.add_entity(label, frame)
    for label, (frame, src, tgt) in (relationships or {}).items():
        builder = builder.add_relationship(
            label, frame, source_col=src, target_col=tgt
        )
    return builder.build(backend=backend)


def run_query(context: Any, cypher: str) -> list[list[Any]]:
    """Execute *cypher* against *context* and return plain row lists."""
    from pycypher.star import Star

    result = Star(context=context).execute_query(cypher)
    if not isinstance(result, pd.DataFrame):
        result = result.to_pandas()
    return result.to_numpy().tolist()


def assert_same_across_backends(
    entities: dict[str, pd.DataFrame],
    cypher: str,
    relationships: dict[str, tuple[pd.DataFrame, str, str]] | None = None,
    *,
    ordered: bool = True,
) -> list[list[Any]]:
    """Assert *cypher* returns identical rows on pandas and DuckDB.

    Args:
        entities: Node tables (see :func:`build_context`).
        cypher: The query to run on both backends.
        relationships: Edge tables (see :func:`build_context`).
        ordered: When ``False``, compare as multisets — for queries whose
            Cypher does not pin an order, where row order is not part of the
            contract.

    Returns:
        The DuckDB result rows, so callers can additionally assert content.

    """
    results = {
        backend: run_query(
            build_context(entities, relationships, backend=backend), cypher
        )
        for backend in BACKENDS
    }
    left, right = results["pandas"], results["duckdb"]
    if not ordered:
        left = sorted(left, key=repr)
        right = sorted(right, key=repr)
    assert right == left, (
        f"backends disagree for {cypher!r}\n"
        f"  pandas: {left}\n"
        f"  duckdb: {right}"
    )
    return results["duckdb"]
