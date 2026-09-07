"""Compile expression ASTs over a :class:`~pycypher.plan.nodes.Scope`.

Wraps :func:`pycypher.relation_sql.compile_expression` (the recursive
expression compiler, unchanged) with resolvers that know how a scope's
bindings are laid out in the emitted SQL.

Two layouts exist. Inside a *pattern block* (a chain of Scan/Expand/Filter
fused into one FROM clause) every node and relationship variable has its
own table alias, and a property is read straight off it — no extra join,
so a single-table scan never becomes a self-join. After a projection,
bindings are columns of the stage's input table (aliased *t*), a node or
relationship is carried as its id, and a property is read by LEFT
JOINing that entity's table on the id; the joins a stage needs are
recorded on the compiler so the emitter can append them.

The translator uses the same class (id layout) as the emitter, so what it
accepts is exactly what the emitter can emit.
"""

from __future__ import annotations

from typing import Any

from pycypher.plan.errors import Unsupported
from pycypher.plan.nodes import NODE, REL, SCALAR, PropEq, Scope
from pycypher.relation_sql import (
    ID_SENTINEL,
    compile_aggregate,
    compile_expression,
    is_aggregate,
)


def quote(name: str) -> str:
    """Quote an output identifier (safe for dots etc.)."""
    return '"' + name.replace('"', '""') + '"'


def idref(id_col: str) -> str:
    """Reference an entity's id column: quoted, or the bare ``rowid``."""
    return "rowid" if id_col == "rowid" else quote(id_col)


class ScopeCompiler:
    """Expression compiler bound to one scope and one SQL layout.

    *direct* maps variables whose own table is present in the FROM clause
    to that table's alias (the pattern-block layout); every other node or
    relationship binding is read through *table_alias* by id.
    """

    def __init__(
        self,
        scope: Scope,
        catalog: Any,
        table_alias: str = "t",
        direct: dict[str, str] | None = None,
        alias_gen: Any = None,
    ) -> None:
        self.scope = scope
        self.catalog = catalog
        self.t = table_alias
        self.direct = direct or {}
        self.udfs = catalog.udf_names()
        self._alias_gen = alias_gen
        #: var -> property-join alias, for every node/rel read by id.
        self.joins: dict[str, str] = {}

    # -- resolvers ---------------------------------------------------------

    def _join_alias(self, var: str) -> str:
        alias = self.joins.get(var)
        if alias is None:
            alias = (
                self._alias_gen("p")
                if self._alias_gen is not None
                else f"p{len(self.joins)}_{var}"
            )
            self.joins[var] = alias
        return alias

    def _ref(self, b: Any) -> Any:
        return (
            self.catalog.entity(b.label)
            if b.kind == NODE
            else self.catalog.relationship(b.label)
        )

    def id_sql(self, var: str) -> str:
        """SQL for a node/relationship binding's id."""
        b = self.scope.get(var)
        alias = self.direct.get(var)
        if alias is None:
            return f"{self.t}.{quote(var)}"
        if b.kind == NODE:
            return f"{alias}.{idref(self.catalog.entity(b.label).id_col)}"
        return f'{alias}."__ID__"'

    def col_sql(self, name: str) -> str:
        """SQL carrying binding *name* through unchanged (id or value)."""
        b = self.scope.get(name)
        if b.kind == SCALAR:
            return f"{self.t}.{quote(name)}"
        return self.id_sql(name)

    def resolve(self, var: str, prop: str) -> str | None:
        b = self.scope.get(var)
        if b is None or b.kind == SCALAR:
            return None
        if prop == ID_SENTINEL:
            return self.id_sql(var)
        ref = self._ref(b)
        if ref is None:
            return None
        col = ref.attr.get(prop)
        if col is None:
            if (var, prop) in self.scope.new_props:
                col = prop  # created by an earlier SET in this query
            else:
                return None
        alias = self.direct.get(var)
        if alias is None:
            alias = quote(self._join_alias(var))
        return f"{alias}.{quote(col)}"

    def resolve_var(self, name: str) -> str | None:
        b = self.scope.get(name)
        if b is None or b.kind != SCALAR:
            return None
        return f"{self.t}.{quote(name)}"

    # -- compilation -------------------------------------------------------

    def expr(self, node: Any, construct: str = "expression") -> str:
        sql = compile_expression(
            node, self.resolve, self.udfs, self.resolve_var
        )
        if sql is None:
            raise Unsupported(construct, _describe(node))
        return sql

    def aggregate(self, node: Any) -> str:
        """Compile an aggregate; ``count(<node var>)`` counts the id so an
        OPTIONAL MATCH's unmatched rows are not counted.
        """
        from pycypher.ast_models import FunctionInvocation, Variable

        if (
            isinstance(node, FunctionInvocation)
            and node.name.lower() == "count"
        ):
            args = (
                node.arguments.get("arguments", [])
                if isinstance(node.arguments, dict)
                else []
            )
            if len(args) == 1 and isinstance(args[0], Variable):
                b = self.scope.get(args[0].name)
                if b is not None and b.kind in (NODE, REL):
                    distinct = (
                        "DISTINCT " if getattr(node, "distinct", False) else ""
                    )
                    return f"COUNT({distinct}{self.id_sql(b.name)})"
        sql = compile_aggregate(
            node, self.resolve, self.udfs, self.resolve_var
        )
        if sql is None:
            raise Unsupported("aggregate", _describe(node))
        return sql

    def predicate(self, pred: Any) -> str:
        if isinstance(pred, PropEq):
            left = self.resolve(pred.var, pred.prop)
            if left is None:
                raise Unsupported("property", f"{pred.var}.{pred.prop}")
            right = compile_expression(
                pred.value, self.resolve, self.udfs, self.resolve_var
            )
            if right is None:
                raise Unsupported("inline property value", pred.prop)
            return f"({left} = {right})"
        return self.expr(pred, "predicate")

    def join_clauses(self) -> str:
        """LEFT JOINs for every entity read by id."""
        parts = []
        for var, alias in self.joins.items():
            b = self.scope.get(var)
            ref = self._ref(b)
            key = idref(ref.id_col) if b.kind == NODE else '"__ID__"'
            parts.append(
                f"LEFT JOIN {ref.sql_ref} AS {quote(alias)} "
                f"ON {quote(alias)}.{key} = {self.t}.{quote(var)}"
            )
        return " ".join(parts)


def _describe(node: Any) -> str:
    name = type(node).__name__
    fname = getattr(node, "name", None)
    return f"{name}({fname})" if isinstance(fname, str) else name


__all__ = ["ScopeCompiler", "idref", "is_aggregate", "quote"]
