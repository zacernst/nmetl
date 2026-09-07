"""Logical plan → DuckDB SQL, and execution of side-effecting plans.

``emit(plan)`` renders a pure read plan as SQL. ``run(plan)`` walks a
plan that may contain ``Mutate``/``Delete``/``Create`` nodes: each side
effect materialises its input rows into a temporary table, runs native
DML against the registry's table, and the rest of the plan continues from
that temporary table — so a stage after a ``SET`` reads the updated
values like any other property.

A *pattern block* — a chain of Scan/Expand/Filter — is rendered as one
FROM clause with a table alias per variable, and the Project or Unnest
that consumes it is fused into the same SELECT, reading properties off
those aliases directly. That keeps a single-table scan a single scan
(no self-join), which is what keeps the path out-of-core. After a
projection only ids are carried and properties are joined back on
demand (see :mod:`pycypher.plan.expressions`).

The only place in the package that knows SQL syntax.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Any

from shared.logger import LOGGER

from pycypher.plan.catalog import Catalog
from pycypher.plan.errors import Unsupported
from pycypher.plan.expressions import ScopeCompiler, idref, quote
from pycypher.plan.nodes import (
    NODE,
    REL,
    Create,
    Delete,
    Expand,
    Filter,
    Join,
    Mutate,
    Plan,
    Project,
    Scan,
    Scope,
    Unit,
    Unnest,
    has_side_effects,
)


@dataclass(frozen=True)
class _TableSource(Plan):
    """A materialised temporary table standing in for a subtree."""

    name: str = ""


@dataclass
class _Block:
    """A fused pattern phase: FROM fragments, predicates, var → alias."""

    froms: list[str]
    preds: list[str]
    direct: dict[str, str]
    scope: Scope

    def from_sql(self) -> str:
        return " ".join(self.froms)

    def where_sql(self) -> str:
        return f" WHERE {' AND '.join(self.preds)}" if self.preds else ""


class Emitter:
    """Render and run plans against one DuckDB-backed context."""

    def __init__(self, context: Any, catalog: Catalog | None = None) -> None:
        self.context = context
        self.catalog = catalog or Catalog(context)
        self.con = context.backend.connection
        self._n = 0

    def _alias(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # -- public --------------------------------------------------------------

    def emit(self, plan: Plan) -> str:
        """SQL for a plan with no side effects."""
        if has_side_effects(plan):
            raise Unsupported("emit", "plan has side effects; use run()")
        return self.sql(plan)

    def run(self, plan: Plan) -> Any | None:
        """Execute side effects and return the final relation (or ``None``
        when the plan ends in a mutation).
        """
        if isinstance(plan, (Mutate, Delete, Create)):
            self._run_terminal(plan)
            return None
        src = self._materialise(plan)
        LOGGER.debug("[duckdb-plan] %s", src)
        return self.con.sql(src)

    # -- pattern blocks --------------------------------------------------------

    def _block(self, plan: Plan) -> _Block | None:
        """Fuse a Scan/Expand/Filter chain, or ``None`` for anything else."""
        if isinstance(plan, Scan):
            ref = self.catalog.entity(plan.label)
            a = self._alias("n")
            return _Block(
                [f"{ref.sql_ref} AS {a}"], [], {plan.var: a}, plan.scope
            )
        if isinstance(plan, Expand):
            blk = self._block(plan.child)
            if blk is None:
                return None
            from pycypher.relation_engine import _edge_endpoint_predicates

            rel = self.catalog.relationship(plan.rel_type)
            to = self.catalog.entity(plan.to_label)
            from_b = plan.child.scope.get(plan.from_var)
            from_id = ScopeCompiler(
                blk.scope, self.catalog, direct=blk.direct
            ).id_sql(plan.from_var)
            e, n = self._alias("e"), self._alias("n")
            near, far = (
                ("__SOURCE__", "__TARGET__")
                if plan.right
                else ("__TARGET__", "__SOURCE__")
            )
            src_label, tgt_label = (
                (from_b.label, plan.to_label)
                if plan.right
                else (plan.to_label, from_b.label)
            )
            preds = (
                _edge_endpoint_predicates(_Cols(rel), e, src_label, tgt_label)
                if rel.has_label_cols
                else []
            )
            on_edge = " AND ".join([f"{from_id} = {e}.{quote(near)}", *preds])
            on_node = f"{e}.{quote(far)} = {n}.{idref(to.id_col)}"
            if plan.optional:
                # One LEFT JOIN against the (edge JOIN far-node) pair: an
                # edge whose far end is not a node of the pattern's label
                # then contributes nothing, exactly as in Cypher. Two
                # successive LEFT JOINs would keep such an edge as a row
                # with a NULL node -- one spurious row per such edge.
                blk.froms.append(
                    f"LEFT JOIN ({rel.sql_ref} AS {e} JOIN {to.sql_ref} AS {n} "
                    f"ON {on_node}) ON {on_edge}"
                )
            else:
                blk.froms.append(f"JOIN {rel.sql_ref} AS {e} ON {on_edge}")
                blk.froms.append(f"JOIN {to.sql_ref} AS {n} ON {on_node}")
            blk.direct[plan.to_var] = n
            if plan.rel_var is not None:
                blk.direct[plan.rel_var] = e
            blk.scope = plan.scope
            return blk
        if isinstance(plan, Filter):
            blk = self._block(plan.child)
            if blk is None:
                return None
            sc = ScopeCompiler(
                blk.scope,
                self.catalog,
                direct=blk.direct,
                alias_gen=self._alias,
            )
            blk.preds.extend(sc.predicate(p) for p in plan.predicates)
            return blk
        return None

    def _block_ids(self, blk: _Block) -> str:
        sc = ScopeCompiler(
            blk.scope, self.catalog, direct=blk.direct, alias_gen=self._alias
        )
        return ", ".join(
            f"{sc.id_sql(n)} AS {quote(n)}" for n in blk.scope.names
        )

    def _source(self, plan: Plan) -> tuple[str, str, ScopeCompiler]:
        """``(from_sql, where_sql, compiler)`` for *plan* as a stage input.

        A pattern block is inlined with direct aliases (its predicates are
        the WHERE); anything else becomes a subquery aliased ``t`` read by
        id, with an empty WHERE. Kept as two strings so a consuming stage
        can put its property joins between them.
        """
        blk = self._block(plan)
        if blk is not None:
            sc = ScopeCompiler(
                blk.scope,
                self.catalog,
                direct=blk.direct,
                alias_gen=self._alias,
            )
            return f"FROM {blk.from_sql()}", blk.where_sql(), sc
        sc = ScopeCompiler(plan.scope, self.catalog, alias_gen=self._alias)
        return f"FROM ({self.sql(plan)}) AS t", "", sc

    # -- read-only rendering -------------------------------------------------

    def sql(self, plan: Plan) -> str:
        blk = self._block(plan)
        if blk is not None:
            return f"SELECT {self._block_ids(blk)} FROM {blk.from_sql()}{blk.where_sql()}"
        method = getattr(self, f"_sql_{type(plan).__name__}", None)
        if method is None:
            raise Unsupported("emit", type(plan).__name__)
        return method(plan)

    @staticmethod
    def _cols(names: tuple[str, ...], alias: str) -> str:
        return (
            ", ".join(f"{alias}.{quote(n)}" for n in names)
            if names
            else '1 AS "__unit__"'
        )

    def _sql_Unit(self, plan: Unit) -> str:  # noqa: N802
        return 'SELECT 1 AS "__unit__"'

    def _sql__TableSource(self, plan: Any) -> str:  # noqa: N802
        return f'SELECT {self._cols(plan.scope.names, "t")} FROM "{plan.name}" AS t'

    def _sql_Filter(self, plan: Filter) -> str:  # noqa: N802
        # Only reached for a Filter whose child is not a pattern block
        # (e.g. a WHERE after a second MATCH's join); blocks fuse filters.
        from_sql, where, sc = self._source(plan.child)
        preds = [sc.predicate(p) for p in plan.predicates]
        cols = ", ".join(
            f"{sc.col_sql(n)} AS {quote(n)}" for n in plan.scope.names
        )
        where = f"{where} AND " if where else " WHERE "
        return f"SELECT {cols} {from_sql} {sc.join_clauses()}{where}{' AND '.join(preds)}"

    def _sql_Unnest(self, plan: Unnest) -> str:  # noqa: N802
        from_sql, where, sc = self._source(plan.child)
        expr = sc.expr(plan.expr)
        cols = ", ".join(
            f"{sc.col_sql(n)} AS {quote(n)}" for n in plan.child.scope.names
        )
        head = f"{cols}, " if plan.child.scope.names else ""
        return f"SELECT {head}UNNEST({expr}) AS {quote(plan.var)} {from_sql} {sc.join_clauses()}{where}"

    def _sql_Join(self, plan: Join) -> str:  # noqa: N802
        left = self.sql(plan.child)
        right = self.sql(plan.other)
        right_new = tuple(
            n for n in plan.other.scope.names if n not in plan.on
        )
        select = [self._cols(plan.child.scope.names, "l")]
        if right_new:
            select.append(self._cols(right_new, "r"))
        if plan.on:
            cond = " AND ".join(
                f"l.{quote(n)} = r.{quote(n)}" for n in plan.on
            )
            return f"SELECT {', '.join(select)} FROM ({left}) AS l JOIN ({right}) AS r ON {cond}"
        return f"SELECT {', '.join(select)} FROM ({left}) AS l CROSS JOIN ({right}) AS r"

    def _sql_Project(self, plan: Project) -> str:  # noqa: N802
        from_sql, where, sc = self._source(plan.child)
        select: list[str] = []
        group: list[str] = []
        for it in plan.items:
            if it.passthrough is not None:
                sql = sc.id_sql(it.passthrough)
                group.append(sql)
            elif it.aggregate:
                sql = sc.aggregate(it.expr)
            else:
                sql = sc.expr(it.expr)
                group.append(sql)
            select.append(f"{sql} AS {quote(it.name)}")
        distinct = "DISTINCT " if plan.distinct else ""
        inner = f"SELECT {distinct}{', '.join(select)} {from_sql} {sc.join_clauses()}{where}"
        if plan.aggregating and group:
            inner += f" GROUP BY {', '.join(group)}"
        if plan.where is None and not plan.order_by and plan.limit is None:
            return inner
        osc = ScopeCompiler(
            plan.scope, self.catalog, table_alias="q", alias_gen=self._alias
        )
        post_where = (
            f" WHERE {osc.expr(plan.where)}" if plan.where is not None else ""
        )
        joins = osc.join_clauses()
        order = ""
        if plan.order_by:
            order = " ORDER BY " + ", ".join(
                f"q.{quote(n)} {'ASC' if asc else 'DESC'} NULLS LAST"
                for n, asc in plan.order_by
            )
        limit = ""
        if plan.limit is not None:
            limit = f" LIMIT {int(plan.limit)} OFFSET {int(plan.skip or 0)}"
        return f"SELECT {self._cols(plan.scope.names, 'q')} FROM ({inner}) AS q {joins}{post_where}{order}{limit}"

    # -- side effects ----------------------------------------------------------

    def _mutate_select(self, plan: Mutate) -> str:
        """One fused SELECT of *plan*'s input rows plus the assignment values
        (``__setval_i__``), computed in the same pass as the pattern.
        """
        from_sql, where, sc = self._source(plan.child)
        cols = [
            f"{sc.col_sql(n)} AS {quote(n)}" for n in plan.child.scope.names
        ]
        vals = [
            f'{sc.expr(expr)} AS "__setval_{i}__"'
            for i, (_, expr) in enumerate(plan.assignments)
        ]
        return f"SELECT {', '.join(cols + vals)} {from_sql} {sc.join_clauses()}{where}"

    def dml_statements(self, plan: Plan) -> list[str]:
        """The DML a terminal ``Mutate``/``Delete``/``Create`` would run
        (without running it, and without the new-column probes). For tests
        and debugging.
        """
        if isinstance(plan, Mutate):
            return [self._update_sql(plan, f"({self._mutate_select(plan)})")]
        if isinstance(plan, Delete):
            return [self._delete_sql(plan, f"({self.sql(plan.child)})")]
        if isinstance(plan, Create):
            return [self._insert_sql(plan)]
        raise Unsupported("dml", type(plan).__name__)

    def _materialise(self, plan: Plan) -> str:
        """SQL selecting *plan*'s rows, having executed any side effects below it.

        A side effect that is the *top* of the plan (the common pipeline
        case: a mutation-only query) runs as a single DML statement over the
        fused SELECT, with no temporary table. One that later stages read
        from materialises its rows first, so those stages see the updated
        values.
        """
        if isinstance(plan, Create):
            self._run_create(plan)
            return self._sql_Unit(Unit(plan.scope))
        if isinstance(plan, Mutate):
            if has_side_effects(plan.child):
                child_src = self._materialise(plan.child)
                view = f"__pycypher_plan_view_{uuid.uuid4().hex}__"
                self.con.execute(f'CREATE TEMP VIEW "{view}" AS {child_src}')  # nosec B608 — generated name; SQL from this emitter only
                plan = replace(
                    plan, child=_TableSource(plan.child.scope, name=view)
                )
            tmp = f"__pycypher_plan_{uuid.uuid4().hex}__"
            self.con.execute(
                f'CREATE TEMP TABLE "{tmp}" AS {self._mutate_select(plan)}'  # nosec B608 — see above
            )
            self._run_mutate(plan, f'"{tmp}"')
            return (
                f'SELECT {self._cols(plan.scope.names, "t")} FROM "{tmp}" AS t'
            )
        if isinstance(plan, Delete):
            child_src = self._materialise(plan.child)
            tmp = f"__pycypher_plan_{uuid.uuid4().hex}__"
            self.con.execute(f'CREATE TEMP TABLE "{tmp}" AS {child_src}')  # nosec B608 — see above
            self._run_delete(plan, f'"{tmp}"')
            return (
                f'SELECT {self._cols(plan.scope.names, "t")} FROM "{tmp}" AS t'
            )
        if not has_side_effects(plan):
            return self.sql(plan)
        child_src = self._materialise(plan.child)
        view = f"__pycypher_plan_view_{uuid.uuid4().hex}__"
        self.con.execute(f'CREATE TEMP VIEW "{view}" AS {child_src}')  # nosec B608 — see above
        return self.sql(
            replace(plan, child=_TableSource(plan.child.scope, name=view))
        )

    def _run_terminal(self, plan: Plan) -> None:
        """Run a side effect at the top of the plan without materialising."""
        if isinstance(plan, Create):
            self._run_create(plan)
        elif isinstance(plan, Mutate):
            if has_side_effects(plan.child):
                self._materialise(plan)  # needs the child's effects first
                return
            self._run_mutate(plan, f"({self._mutate_select(plan)})")
        elif isinstance(plan, Delete):
            if has_side_effects(plan.child):
                self._materialise(plan)
                return
            self._run_delete(plan, f"({self.sql(plan.child)})")

    def _update_sql(self, plan: Mutate, source: str) -> str:
        from pycypher.backends.table_registry import physical_table_name

        ref = self.catalog.entity(plan.label)
        others = [n for n in plan.child.scope.names if n != plan.target]
        order = (
            ", ".join(f"t.{quote(n)}" for n in others)
            or f"t.{quote(plan.target)}"
        )
        select = [f't.{quote(plan.target)} AS "__id__"'] + [
            f't."__setval_{i}__"' for i in range(len(plan.assignments))
        ]
        sub = (
            f"SELECT {', '.join(select)} FROM {source} AS t "
            f"QUALIFY ROW_NUMBER() OVER (PARTITION BY t.{quote(plan.target)} ORDER BY {order}) = 1"
        )
        table = physical_table_name(plan.label)
        sets = ", ".join(
            f'{quote(ref.attr.get(prop, prop))} = sub."__setval_{i}__"'
            for i, (prop, _) in enumerate(plan.assignments)
        )
        return (
            f'UPDATE "{table}" SET {sets} FROM ({sub}) AS sub '  # nosec B608 — identifiers validated by the registry; values from the whitelisted expression compiler
            f'WHERE "{table}".{idref(ref.id_col)} = sub."__id__"'
        )

    def _run_mutate(self, plan: Mutate, source: str) -> None:
        """Run *plan*'s UPDATE over *source* (a quoted temp-table name or a
        parenthesised SELECT carrying ``__setval_i__`` columns).
        """
        from pycypher.relation_engine import _ensure_column

        ref = self.catalog.entity(plan.label)
        for i, (prop, _) in enumerate(plan.assignments):
            if prop not in ref.attr:
                probe = self.con.sql(
                    f'SELECT t."__setval_{i}__" AS "__probe__" FROM {source} AS t LIMIT 0'  # nosec B608 — see above
                )
                _ensure_column(self.context, plan.label, prop, probe)
        sql = self._update_sql(plan, source)
        LOGGER.debug("[duckdb-plan] %s", sql)
        self.con.execute(sql)

    def _delete_sql(self, plan: Delete, source: str) -> str:
        from pycypher.backends.table_registry import physical_table_name

        ref = self.catalog.entity(plan.label)
        table = physical_table_name(plan.label)
        return (
            f'DELETE FROM "{table}" WHERE {idref(ref.id_col)} IN '  # nosec B608 — see above
            f"(SELECT {quote(plan.var)} FROM {source} AS t)"
        )

    def _run_delete(self, plan: Delete, source: str) -> None:
        sql = self._delete_sql(plan, source)
        LOGGER.debug("[duckdb-plan] %s", sql)
        self.con.execute(sql)

    def _run_create(self, plan: Create) -> None:
        sql = self._insert_sql(plan)
        LOGGER.debug("[duckdb-plan] %s", sql)
        self.con.execute(sql)

    def _insert_sql(self, plan: Create) -> str:
        from pycypher.backends._helpers import validate_identifier
        from pycypher.backends.table_registry import physical_table_name
        from pycypher.relation_engine import _streaming_id_col

        ref = self.catalog.entity(plan.label)
        table = physical_table_name(plan.label)
        cols: list[str] = []
        values: list[str] = []
        id_col = _streaming_id_col(self.context, plan.label)
        if id_col is not None:
            quoted_id_col = validate_identifier(id_col)
            seq = f"_streaming_seq_{validate_identifier(plan.label)}"
            max_id = self.con.execute(
                f'SELECT COALESCE(MAX("{quoted_id_col}"), 0) FROM "{table}"'  # nosec B608 — validated identifiers
            ).fetchone()[0]
            self.con.execute(
                f'CREATE SEQUENCE IF NOT EXISTS "{seq}" START {int(max_id) + 1}'  # nosec B608 — validated name, int value
            )
            cols.append(quoted_id_col)
            values.append(f"nextval('{seq}')")
        sc = ScopeCompiler(Scope(), self.catalog)
        for key, expr in plan.properties:
            cols.append(validate_identifier(ref.attr[key]))
            values.append(sc.expr(expr))
        return (
            f'INSERT INTO "{table}" ({", ".join(quote(c) for c in cols)}) '  # nosec B608 — validated identifiers; values from the whitelisted compiler or nextval()
            f"SELECT {', '.join(values)}"
        )


class _Cols:
    """Adapter giving :func:`_edge_endpoint_predicates` a ``.columns``."""

    def __init__(self, rel: Any) -> None:
        from pycypher.relation_engine import (
            SOURCE_LABEL_COLUMN,
            TARGET_LABEL_COLUMN,
        )

        self.columns = (
            [SOURCE_LABEL_COLUMN, TARGET_LABEL_COLUMN]
            if rel.has_label_cols
            else []
        )


__all__ = ["Emitter"]
