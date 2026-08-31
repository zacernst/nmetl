"""Phase 5 (DuckDB eager path) — WHERE predicates compiled to SQL.

``BindingFilter`` splits its predicate on ``AND`` and compiles each conjunct
through ``relation_sql.compile_expression``.  Compiled conjuncts become a SQL
``WHERE`` on the lazy relation; anything else falls through to the unchanged
pandas evaluator, so a partially pushable filter still gets most of the
benefit.

Properties are resolved by LEFT-joining the entity table rather than by
projecting property columns into the frame — ``var_names`` is the engine's
list of bound Cypher variables, and property columns must never leak into it.

See docs/duckdb_eager_path_design.md, Phase 5.
"""

from __future__ import annotations

import pandas as pd
import pytest
from helpers_differential import assert_same_across_backends, build_context
from pycypher.ast_converter import ASTConverter
from pycypher.backends.duckdb_backend import count_materialisations
from pycypher.binding_evaluator import BindingExpressionEvaluator
from pycypher.scan_operators import BindingFilter, EntityScan

PEOPLE = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Carol", "Dan", None],
        "age": [30, 25, 35, 40, None],
        "city": ["NY", "LA", "NY", "SF", None],
    },
)
KNOWS = pd.DataFrame({"__SOURCE__": [0, 1, 0], "__TARGET__": [1, 2, 2]})
ENTITIES = {"Person": PEOPLE}
RELATIONSHIPS = {"KNOWS": (KNOWS, "__SOURCE__", "__TARGET__")}


@pytest.fixture
def duck():
    return build_context(ENTITIES, RELATIONSHIPS, backend="duckdb")


def _where(cypher: str):
    return ASTConverter.from_cypher(cypher).clauses[0].where


def _filter(duck, cypher: str):
    frame = EntityScan("Person", "p").scan(duck)
    return BindingFilter(
        predicate=_where(cypher),
        evaluator_factory=BindingExpressionEvaluator,
    ).apply(frame)


class TestPushdownHappens:
    """These assert `is_lazy`, not just correctness.

    A predicate that fails to compile still returns the right rows via the
    pandas fallback, so correctness alone cannot tell you whether pushdown
    is working.  During development an unrelated DuckDB error made every
    push fall back silently while the whole suite stayed green; only an
    `is_lazy` assertion catches that.
    """

    def test_comparison_pushes(self, duck):
        out = _filter(duck, "MATCH (p:Person) WHERE p.age > 26 RETURN p")
        assert out.is_lazy is True
        # ages 30, 35, 40 qualify; 25 does not and the null row is dropped.
        assert len(out) == 3

    def test_conjunction_pushes(self, duck):
        out = _filter(
            duck,
            "MATCH (p:Person) WHERE p.age > 26 AND p.city = 'NY' RETURN p",
        )
        assert out.is_lazy is True
        assert len(out) == 2

    def test_disjunction_pushes(self, duck):
        out = _filter(
            duck,
            "MATCH (p:Person) WHERE p.age > 34 OR p.city = 'LA' RETURN p",
        )
        assert out.is_lazy is True
        assert len(out) == 3

    def test_negation_pushes(self, duck):
        out = _filter(duck, "MATCH (p:Person) WHERE NOT (p.age = 30) RETURN p")
        assert out.is_lazy is True

    def test_pushdown_does_not_materialise(self, duck):
        with count_materialisations() as log:
            out = _filter(duck, "MATCH (p:Person) WHERE p.age > 26 RETURN p")
            assert len(out) == 3
        assert log.count == 0

    def test_variable_list_is_unchanged_by_the_property_join(self, duck):
        # The LEFT join adds id/property columns; they must be projected
        # away, because var_names is read as the bound-variable list.
        out = _filter(duck, "MATCH (p:Person) WHERE p.age > 26 RETURN p")
        assert out.var_names == ["p"]
        assert list(out.bindings.columns) == ["p"]

    def test_multiple_properties_of_one_variable(self, duck):
        # Two joins against the same table; their aliases must stay distinct.
        out = _filter(
            duck,
            "MATCH (p:Person) WHERE p.age > 26 AND p.name = 'Alice' RETURN p",
        )
        assert out.is_lazy is True
        assert len(out) == 1


class TestFallback:
    def test_uncompilable_conjunct_falls_back(self, duck):
        out = _filter(
            duck,
            "MATCH (p:Person) WHERE toUpper(p.name) = 'ALICE' RETURN p",
        )
        assert out.is_lazy is False
        assert len(out) == 1

    def test_partial_push_keeps_the_residual_in_pandas(self, duck):
        out = _filter(
            duck,
            (
                "MATCH (p:Person) WHERE p.age > 26 "
                "AND toUpper(p.name) = 'ALICE' RETURN p"
            ),
        )
        assert len(out) == 1

    def test_shadow_overlay_blocks_pushdown(self, duck):
        duck._shadow["Person"] = PEOPLE.assign(__ID__=range(len(PEOPLE)))
        frame = EntityScan("Person", "p").scan(duck)
        assert frame.is_lazy is False

    def test_pandas_backend_is_untouched(self):
        context = build_context(ENTITIES, backend="pandas")
        frame = EntityScan("Person", "p").scan(context)
        out = BindingFilter(
            predicate=_where("MATCH (p:Person) WHERE p.age > 26 RETURN p"),
            evaluator_factory=BindingExpressionEvaluator,
        ).apply(frame)
        assert out.is_lazy is False
        assert len(out) == 3


class TestThreeValuedLogic:
    """Null handling must match pandas exactly.

    ``boolean_evaluator`` implements Kleene logic (``NOT null -> null``), the
    same as SQL, so pushing ``OR``/``NOT`` is safe — a concern the design doc
    raised and these tests settle.  The row with all-null properties is the
    one that discriminates.
    """

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (p:Person) WHERE p.age > 26 RETURN p.name ORDER BY p.name",
            (
                "MATCH (p:Person) WHERE NOT (p.age = 30) "
                "RETURN p.name ORDER BY p.name"
            ),
            "MATCH (p:Person) WHERE p.age IS NULL RETURN p.name",
            (
                "MATCH (p:Person) WHERE p.age IS NOT NULL "
                "RETURN p.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person) WHERE p.age > 26 OR p.city = 'LA' "
                "RETURN p.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person) WHERE NOT (p.age > 26 AND p.city = 'NY') "
                "RETURN p.name ORDER BY p.name"
            ),
            "MATCH (p:Person) WHERE p.missing = 1 RETURN p.name",
            (
                "MATCH (p:Person) WHERE p.missing IS NULL "
                "RETURN p.name ORDER BY p.name"
            ),
        ],
    )
    def test_null_semantics_match_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher)


class TestDifferentialQueries:
    @pytest.mark.parametrize(
        "cypher",
        [
            (
                "MATCH (p:Person) WHERE p.age >= 30 AND p.age <= 35 "
                "RETURN p.name ORDER BY p.name"
            ),
            "MATCH (p:Person) WHERE p.name STARTS WITH 'A' RETURN p.name",
            (
                "MATCH (p:Person) WHERE p.name CONTAINS 'o' "
                "RETURN p.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person) WHERE p.city IN ['NY', 'SF'] "
                "RETURN p.name ORDER BY p.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 26 "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person)-[:KNOWS]->(q:Person) "
                "WHERE p.age > 26 AND q.age < 40 "
                "RETURN p.name, q.name ORDER BY p.name, q.name"
            ),
            (
                "MATCH (p:Person) WHERE p.age > 26 "
                "RETURN p.city, count(p) ORDER BY p.city"
            ),
        ],
    )
    def test_query_matches_pandas(self, cypher):
        assert_same_across_backends(ENTITIES, cypher, RELATIONSHIPS)
