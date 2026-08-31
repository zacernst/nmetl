"""ORDER BY must survive a projection that changes the row count.

``RETURN p.city, count(p) ORDER BY p.city`` returned its groups unsorted, on
both backends.  ``apply_projection_modifiers`` evaluated the sort key against
the *pre-projection* frame and then did ``sort_series.values[:len(df)]`` —
truncating six pre-aggregation values to four post-aggregation rows and
pairing them arbitrarily.  The result looked plausible and was wrong.

In Cypher an ``ORDER BY`` key that also appears in the projection refers to
the *projected* value, so the fix resolves it to the output column. These
tests cover the shapes that were broken and the ones that already worked, so
a future change cannot quietly reintroduce either failure.

Backend-independent: the bug affected pandas and DuckDB alike.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import BACKENDS, build_context, run_query

PEOPLE = pd.DataFrame(
    {
        "city": ["SF", "NY", "LA", "NY", "SF", "CHI"],
        "name": ["f", "b", "c", "d", "a", "e"],
        "age": [1, 2, 3, 4, 5, 6],
    },
)
ENTITIES = {"Person": PEOPLE}


def _run(cypher: str, backend: str = "pandas"):
    return run_query(build_context(ENTITIES, backend=backend), cypher)


@pytest.mark.parametrize("backend", BACKENDS)
class TestRowCountChangingProjections:
    """The shapes that were broken: aggregation and DISTINCT."""

    def test_group_key_ascending(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city",
            backend,
        ) == [["CHI", 1], ["LA", 1], ["NY", 2], ["SF", 2]]

    def test_group_key_descending(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city DESC",
            backend,
        ) == [["SF", 2], ["NY", 2], ["LA", 1], ["CHI", 1]]

    def test_distinct(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN DISTINCT p.city ORDER BY p.city",
            backend,
        ) == [["CHI"], ["LA"], ["NY"], ["SF"]]

    def test_group_key_with_limit(self, backend):
        # LIMIT must apply *after* the sort, so this is the two smallest
        # cities, not the first two of an arbitrary order.
        assert _run(
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city LIMIT 2",
            backend,
        ) == [["CHI", 1], ["LA", 1]]

    def test_group_key_with_skip(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.city SKIP 2",
            backend,
        ) == [["NY", 2], ["SF", 2]]

    def test_multiple_keys(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.city AS c, count(p) AS n "
            "ORDER BY n DESC, c",
            backend,
        ) == [["NY", 2], ["SF", 2], ["CHI", 1], ["LA", 1]]


@pytest.mark.parametrize("backend", BACKENDS)
class TestShapesThatAlreadyWorked:
    """Regression cover for the paths the fix must not disturb."""

    def test_plain_order_by(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.name ORDER BY p.name", backend
        ) == [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]

    def test_alias_of_a_grouping_key(self, backend):
        assert _run(
            "MATCH (p:Person) RETURN p.city AS c, count(p) AS n ORDER BY c",
            backend,
        ) == [["CHI", 1], ["LA", 1], ["NY", 2], ["SF", 2]]

    def test_alias_of_an_aggregate(self, backend):
        result = _run(
            "MATCH (p:Person) RETURN p.city AS c, count(p) AS n ORDER BY n",
            backend,
        )
        assert [row[1] for row in result] == [1, 1, 2, 2]

    def test_with_then_order_by(self, backend):
        assert _run(
            "MATCH (p:Person) WITH p.city AS c, count(p) AS n "
            "RETURN c, n ORDER BY c",
            backend,
        ) == [["CHI", 1], ["LA", 1], ["NY", 2], ["SF", 2]]

    def test_sort_key_outside_the_projection(self, backend):
        # `age` is not returned, and the projection does not change the row
        # count, so evaluating the key against the pre-projection frame is
        # still legitimate and must keep working.
        assert _run(
            "MATCH (p:Person) RETURN p.name ORDER BY p.age DESC", backend
        ) == [["e"], ["a"], ["d"], ["c"], ["b"], ["f"]]

    def test_two_variables_are_not_conflated(self, backend):
        # The renderer emits a PropertyLookup as its bare property name, so
        # `p.name` and `q.name` render identically; matching sort keys to
        # projection items by rendered text silently sorted both by `p.name`.
        entities = {"Person": pd.DataFrame({"name": ["a", "b", "c", "d"]})}
        relationships = {
            "KNOWS": (
                pd.DataFrame(
                    {"__SOURCE__": [0, 1, 2, 0], "__TARGET__": [1, 2, 3, 2]},
                ),
                "__SOURCE__",
                "__TARGET__",
            ),
        }
        context = build_context(entities, relationships, backend=backend)
        assert run_query(
            context,
            "MATCH (p:Person)-[:KNOWS]->(q:Person) "
            "RETURN p.name, q.name ORDER BY q.name, p.name",
        ) == [["a", "b"], ["a", "c"], ["b", "c"], ["c", "d"]]


@pytest.mark.parametrize("backend", BACKENDS)
class TestUnalignableSortKey:
    """A sort key may not reach past a DISTINCT or an aggregation.

    Cypher rejects these outright.  This engine used to accept them and zip
    the six pre-projection sort values against four result rows, producing a
    plausible-looking but arbitrary order.  Failing loudly is the honest
    behaviour; the message names the fix.
    """

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person) RETURN DISTINCT p.city ORDER BY p.name",
            "MATCH (p:Person) RETURN p.city, count(p) ORDER BY p.name",
        ],
    )
    def test_raises_rather_than_zipping_mismatched_lengths(
        self, backend, cypher
    ):
        with pytest.raises(ValueError, match="ORDER BY key"):
            _run(cypher, backend)

    def test_variable_dropped_by_a_with_is_a_scoping_error(self, backend):
        # Distinct from the above: `p` does not survive the WITH at all, so
        # this is caught earlier, by name resolution, and never reaches the
        # length check.
        from pycypher.exceptions import VariableNotFoundError

        with pytest.raises(VariableNotFoundError):
            _run(
                "MATCH (p:Person) WITH p.city AS c, count(p) AS n "
                "RETURN c ORDER BY p.age",
                backend,
            )
