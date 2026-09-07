"""Cypher AST → logical plan, as a fold of one rule per clause type.

``translate(query, context)`` seeds a :class:`~pycypher.plan.nodes.Unit`
plan and applies ``RULES[type(clause)]`` to each clause in turn. Each
rule maps *its* clause onto the IR operators and returns the new plan;
everything else — name resolution, what an expression may reference,
whether a property exists — is decided by the same
:class:`~pycypher.plan.expressions.ScopeCompiler` the emitter uses, so
successful translation *is* eligibility. Anything without a rule raises
:class:`~pycypher.plan.errors.Unsupported` naming the construct.
"""

from __future__ import annotations

from typing import Any, Callable

from pycypher.plan.catalog import Catalog
from pycypher.plan.errors import Unsupported
from pycypher.plan.expressions import ScopeCompiler, is_aggregate
from pycypher.plan.nodes import (
    NODE,
    REL,
    SCALAR,
    Binding,
    Create,
    Delete,
    Expand,
    Filter,
    Join,
    Mutate,
    Plan,
    Project,
    ProjectItem,
    PropEq,
    Scan,
    Scope,
    Unit,
    Unnest,
)

Rule = Callable[[Any, Plan, Catalog, bool], Plan]


def translate(query: Any, context: Any) -> Plan:
    """Translate *query* for *context*, or raise :class:`Unsupported`."""
    from pycypher.ast_models import Query, Return

    if getattr(context, "backend_name", None) != "duckdb":
        raise Unsupported("backend", "relation plans need a DuckDB backend")
    if not hasattr(getattr(context, "backend", None), "connection"):
        raise Unsupported("backend", "no DuckDB connection")
    if not isinstance(query, Query):
        raise Unsupported("query", type(query).__name__)
    clauses = list(query.clauses)
    if not clauses:
        raise Unsupported("query", "no clauses")
    catalog = Catalog(context)
    plan: Plan = Unit(Scope())
    last = len(clauses) - 1
    for i, clause in enumerate(clauses):
        if isinstance(clause, Return) and i != last:
            raise Unsupported("RETURN", "must be the final clause")
        rule = RULES.get(type(clause))
        if rule is None:
            raise Unsupported(type(clause).__name__)
        plan = rule(clause, plan, catalog, i == last)
    return plan


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def _node_ok(node: Any) -> bool:
    from pycypher.ast_models import NodePattern

    return (
        isinstance(node, NodePattern)
        and node.variable is not None
        and len(node.labels) == 1
    )


def _inline_preds(nodes: list[Any]) -> list[PropEq]:
    preds = []
    for nd in nodes:
        for prop, value in (getattr(nd, "properties", None) or {}).items():
            preds.append(PropEq(nd.variable.name, prop, value))
    return preds


def _translate_path(
    path: Any, catalog: Catalog, scope: Scope
) -> tuple[Plan, list[PropEq]]:
    """A single required path as Scan + Expand hops from a fresh Unit.

    A node written without a label is accepted when *scope* already binds
    its variable to a node (a later ``MATCH (p)-[:R]->(q)`` continuing
    from a bound ``p``); its label is the binding's.
    """
    from pycypher.ast_models import (
        NodePattern,
        RelationshipDirection,
        RelationshipPattern,
    )

    if path.variable is not None:
        raise Unsupported("path variable")
    if getattr(path, "shortest_path_mode", "none") not in ("none", None):
        raise Unsupported("shortestPath")
    elements = path.elements
    if not elements or len(elements) % 2 == 0:
        raise Unsupported("pattern", "malformed path")
    nodes = elements[0::2]
    rels = elements[1::2]
    labels: dict[str, str] = {}
    for nd in nodes:
        if not isinstance(nd, NodePattern) or nd.variable is None:
            raise Unsupported("node pattern", "needs a variable")
        bound = scope.get(nd.variable.name) if not nd.labels else None
        if len(nd.labels) == 1:
            labels[nd.variable.name] = nd.labels[0]
        elif bound is not None and bound.kind == NODE:
            labels[nd.variable.name] = bound.label
        else:
            raise Unsupported(
                "node pattern",
                "needs exactly one label (or an already-bound variable)",
            )
        if catalog.entity(labels[nd.variable.name]) is None:
            raise Unsupported("label", labels[nd.variable.name])
    seen: set[str] = set()
    for nd in nodes:
        if nd.variable.name in seen:
            raise Unsupported(
                "pattern", f"variable {nd.variable.name!r} bound twice"
            )
        seen.add(nd.variable.name)

    first = nodes[0]
    plan: Plan = Scan(
        Scope(
            (Binding(first.variable.name, NODE, labels[first.variable.name]),),
            qualified=False,
            new_props=scope.new_props,
        ),
        var=first.variable.name,
        label=labels[first.variable.name],
    )
    for j, rp in enumerate(rels):
        if not isinstance(rp, RelationshipPattern):
            raise Unsupported("pattern", "expected a relationship")
        if rp.length is not None:
            raise Unsupported("variable-length path")
        if getattr(rp, "properties", None):
            raise Unsupported("relationship properties in pattern")
        if len(rp.labels) != 1:
            raise Unsupported("relationship type", "exactly one type required")
        if rp.direction not in (
            RelationshipDirection.RIGHT,
            RelationshipDirection.LEFT,
        ):
            raise Unsupported("undirected relationship")
        if catalog.relationship(rp.labels[0]) is None:
            raise Unsupported("relationship type", rp.labels[0])
        to = nodes[j + 1]
        rel_var = rp.variable.name if rp.variable is not None else None
        if rel_var is not None:
            if rel_var in seen:
                raise Unsupported(
                    "pattern", f"variable {rel_var!r} bound twice"
                )
            seen.add(rel_var)
        bindings = list(plan.scope.bindings)
        if rel_var is not None:
            bindings.append(Binding(rel_var, REL, rp.labels[0]))
        bindings.append(
            Binding(to.variable.name, NODE, labels[to.variable.name])
        )
        plan = Expand(
            Scope(
                tuple(bindings),
                qualified=len(bindings) > 1,
                new_props=scope.new_props,
            ),
            child=plan,
            from_var=nodes[j].variable.name,
            rel_type=rp.labels[0],
            right=rp.direction == RelationshipDirection.RIGHT,
            to_var=to.variable.name,
            to_label=labels[to.variable.name],
            rel_var=rel_var,
        )
    return plan, _inline_preds(nodes)


def _filter(plan: Plan, preds: list[Any], catalog: Catalog) -> Plan:
    if not preds:
        return plan
    sc = ScopeCompiler(plan.scope, catalog)
    for p in preds:
        sc.predicate(p)  # validates
    return Filter(plan.scope, child=plan, predicates=tuple(preds))


def rule_match(clause: Any, plan: Plan, catalog: Catalog, _last: bool) -> Plan:
    paths = clause.pattern.paths if clause.pattern is not None else []
    if len(paths) != 1:
        raise Unsupported("pattern", "exactly one path per MATCH")
    path = paths[0]

    if clause.optional:
        return _optional_match(clause, path, plan, catalog)

    sub, preds = _translate_path(path, catalog, plan.scope)
    if isinstance(plan, Unit):
        out: Plan = sub
    else:
        shared = tuple(
            n for n in sub.scope.names if plan.scope.get(n) is not None
        )
        for n in shared:
            a, b = plan.scope.get(n), sub.scope.get(n)
            if a.kind != b.kind or a.label != b.label:
                raise Unsupported(
                    "pattern", f"variable {n!r} rebound with a different kind"
                )
        new = tuple(b for b in sub.scope.bindings if b.name not in shared)
        out = Join(
            Scope(
                plan.scope.bindings + new,
                qualified=True,
                new_props=plan.scope.new_props,
            ),
            child=plan,
            other=sub,
            on=shared,
        )
    if clause.where is not None:
        preds = [*preds, clause.where]
    return _filter(out, preds, catalog)


def _optional_match(
    clause: Any, path: Any, plan: Plan, catalog: Catalog
) -> Plan:
    from pycypher.ast_models import (
        NodePattern,
        RelationshipDirection,
        RelationshipPattern,
    )

    if isinstance(plan, Unit):
        raise Unsupported("OPTIONAL MATCH", "must follow a bound pattern")
    if clause.where is not None:
        raise Unsupported("OPTIONAL MATCH WHERE")
    if path.variable is not None:
        raise Unsupported("path variable")
    if getattr(path, "shortest_path_mode", "none") not in ("none", None):
        raise Unsupported("shortestPath")
    elements = path.elements
    if len(elements) != 3:
        raise Unsupported(
            "OPTIONAL MATCH", "exactly one hop from a bound node"
        )
    n_left, rp, n_right = elements
    if not (isinstance(n_left, NodePattern) and n_left.variable is not None):
        raise Unsupported(
            "OPTIONAL MATCH", "left node must be a bound variable"
        )
    if not _node_ok(n_right):
        raise Unsupported(
            "node pattern", "needs a variable and exactly one label"
        )
    if getattr(n_left, "properties", None) or getattr(
        n_right, "properties", None
    ):
        raise Unsupported("OPTIONAL MATCH", "inline properties")
    if not isinstance(rp, RelationshipPattern):
        raise Unsupported("pattern")
    if rp.length is not None or getattr(rp, "properties", None):
        raise Unsupported("variable-length path")
    if len(rp.labels) != 1:
        raise Unsupported("relationship type", "exactly one type required")
    if rp.direction not in (
        RelationshipDirection.RIGHT,
        RelationshipDirection.LEFT,
    ):
        raise Unsupported("undirected relationship")
    x, y = n_left.variable.name, n_right.variable.name
    xb = plan.scope.get(x)
    if xb is None or xb.kind != NODE:
        raise Unsupported("OPTIONAL MATCH", f"{x!r} is not a bound node")
    if plan.scope.get(y) is not None:
        raise Unsupported("OPTIONAL MATCH", f"{y!r} is already bound")
    if catalog.relationship(rp.labels[0]) is None:
        raise Unsupported("relationship type", rp.labels[0])
    if catalog.entity(n_right.labels[0]) is None:
        raise Unsupported("label", n_right.labels[0])
    rel_var = rp.variable.name if rp.variable is not None else None
    if rel_var is not None and (
        plan.scope.get(rel_var) is not None or rel_var == y
    ):
        raise Unsupported("OPTIONAL MATCH", f"{rel_var!r} is already bound")
    bindings = list(plan.scope.bindings)
    if rel_var is not None:
        bindings.append(Binding(rel_var, REL, rp.labels[0]))
    bindings.append(Binding(y, NODE, n_right.labels[0]))
    return Expand(
        Scope(tuple(bindings), qualified=True, new_props=plan.scope.new_props),
        child=plan,
        from_var=x,
        rel_type=rp.labels[0],
        right=rp.direction == RelationshipDirection.RIGHT,
        to_var=y,
        to_label=n_right.labels[0],
        rel_var=rel_var,
        optional=True,
    )


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def _output_name(item: Any, qualified: bool) -> str:
    from pycypher.ast_models import PropertyLookup, Variable

    if item.alias is not None:
        return str(item.alias)
    expr = item.expression
    if isinstance(expr, Variable):
        return str(expr.name)
    if isinstance(expr, PropertyLookup) and isinstance(
        expr.expression, Variable
    ):
        return (
            f"{expr.expression.name}.{expr.property}"
            if qualified
            else str(expr.property)
        )
    raise Unsupported("projection", "an expression item needs an alias")


def _int_or_none(value: Any, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise Unsupported(what, "must be an integer literal")
    return value


def _projection(
    clause: Any, plan: Plan, catalog: Catalog, *, is_return: bool
) -> Plan:
    from pycypher.ast_models import PropertyLookup, Variable

    if not clause.items:
        raise Unsupported("projection", "no items")
    sc = ScopeCompiler(plan.scope, catalog)
    items: list[ProjectItem] = []
    names: list[str] = []
    bindings: list[Binding] = []
    any_pass = False
    for it in clause.items:
        expr = it.expression
        bound = (
            plan.scope.get(expr.name) if isinstance(expr, Variable) else None
        )
        if bound is not None and bound.kind in (NODE, REL):
            if it.alias is not None and it.alias != expr.name:
                raise Unsupported(
                    "projection", "renaming a node/relationship variable"
                )
            if is_return:
                raise Unsupported(
                    "RETURN", "a bare node/relationship variable"
                )
            name = expr.name
            items.append(ProjectItem(name, passthrough=name))
            bindings.append(bound)
            any_pass = True
        else:
            name = _output_name(it, plan.scope.qualified)
            if is_aggregate(expr):
                if it.alias is None:
                    raise Unsupported("aggregate", "needs an alias")
                sc.aggregate(expr)
                items.append(ProjectItem(name, expr=expr, aggregate=True))
            else:
                sc.expr(expr)
                if (
                    not isinstance(expr, (PropertyLookup, Variable))
                    and it.alias is None
                ):
                    raise Unsupported(
                        "projection", "an expression item needs an alias"
                    )
                items.append(ProjectItem(name, expr=expr))
            bindings.append(Binding(name, SCALAR))
        if name in names:
            raise Unsupported(
                "projection", f"duplicate output column {name!r}"
            )
        names.append(name)

    out_scope = Scope(
        tuple(bindings),
        qualified=plan.scope.qualified if any_pass else False,
        new_props=plan.scope.new_props,
    )
    where = getattr(clause, "where", None)
    if where is not None:
        ScopeCompiler(out_scope, catalog).expr(where, "WITH WHERE")

    order: list[tuple[str, bool]] = []
    prop_to_name = {
        (it.expression.expression.name, it.expression.property): n
        for it, n in zip(clause.items, names, strict=True)
        if isinstance(it.expression, PropertyLookup)
        and isinstance(it.expression.expression, Variable)
    }
    for ob in clause.order_by or []:
        if getattr(ob, "nulls_placement", None) is not None:
            raise Unsupported("ORDER BY NULLS FIRST/LAST")
        e = ob.expression
        col = None
        if isinstance(e, Variable) and e.name in names:
            col = e.name
        elif isinstance(e, PropertyLookup) and isinstance(
            e.expression, Variable
        ):
            col = prop_to_name.get((e.expression.name, e.property))
        if col is None:
            raise Unsupported("ORDER BY", "key must be an output column")
        order.append((col, bool(ob.ascending)))

    skip = _int_or_none(clause.skip, "SKIP")
    limit = _int_or_none(clause.limit, "LIMIT")
    if skip is not None and limit is None:
        raise Unsupported("SKIP without LIMIT")

    return Project(
        out_scope,
        child=plan,
        items=tuple(items),
        distinct=bool(clause.distinct),
        where=where,
        order_by=tuple(order),
        skip=skip,
        limit=limit,
    )


def rule_with(clause: Any, plan: Plan, catalog: Catalog, _last: bool) -> Plan:
    return _projection(clause, plan, catalog, is_return=False)


def rule_return(
    clause: Any, plan: Plan, catalog: Catalog, _last: bool
) -> Plan:
    return _projection(clause, plan, catalog, is_return=True)


def rule_unwind(
    clause: Any, plan: Plan, catalog: Catalog, _last: bool
) -> Plan:
    if clause.alias is None:
        raise Unsupported("UNWIND", "needs an alias")
    if plan.scope.get(clause.alias) is not None:
        raise Unsupported("UNWIND", f"{clause.alias!r} is already bound")
    ScopeCompiler(plan.scope, catalog).expr(clause.expression, "UNWIND")
    return Unnest(
        plan.scope.with_bindings(
            plan.scope.bindings + (Binding(clause.alias, SCALAR),)
        ),
        child=plan,
        expr=clause.expression,
        var=clause.alias,
    )


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


def rule_set(clause: Any, plan: Plan, catalog: Catalog, _last: bool) -> Plan:
    from pycypher.relation_engine import _new_column_name_allowed

    if not clause.items:
        raise Unsupported("SET", "no items")
    targets = {
        it.variable.name if it.variable is not None else None
        for it in clause.items
    }
    if len(targets) != 1 or None in targets:
        raise Unsupported(
            "SET", "all items must target one bound node variable"
        )
    target = next(iter(targets))
    b = plan.scope.get(target)
    if b is None or b.kind != NODE:
        raise Unsupported("SET", f"{target!r} is not a bound node")
    ref = catalog.entity(b.label)
    if ref is None or not ref.writable:
        raise Unsupported(
            "SET", f"{b.label!r} has no registered table to write to"
        )
    sc = ScopeCompiler(plan.scope, catalog)
    assignments = []
    new_props = set()
    for it in clause.items:
        if it.labels:
            raise Unsupported("SET labels")
        if (
            it.property is None
            or it.property in ("*", "*+")
            or it.expression is None
        ):
            raise Unsupported(
                "SET", "only `var.property = expression` is supported"
            )
        sc.expr(it.expression, "SET value")
        if sc.resolve(target, it.property) is None:
            if not _new_column_name_allowed(it.property):
                raise Unsupported(
                    "SET", f"unsafe property name {it.property!r}"
                )
            new_props.add((target, it.property))
        assignments.append((it.property, it.expression))
    return Mutate(
        plan.scope.with_new_props(frozenset(new_props)),
        child=plan,
        target=target,
        label=b.label,
        assignments=tuple(assignments),
    )


def rule_delete(
    clause: Any, plan: Plan, catalog: Catalog, _last: bool
) -> Plan:
    from pycypher.ast_models import Variable

    if clause.detach:
        raise Unsupported("DETACH DELETE")
    if len(clause.expressions) != 1 or not isinstance(
        clause.expressions[0], Variable
    ):
        raise Unsupported("DELETE", "exactly one bound node variable")
    var = clause.expressions[0].name
    b = plan.scope.get(var)
    if b is None or b.kind != NODE:
        raise Unsupported("DELETE", f"{var!r} is not a bound node")
    ref = catalog.entity(b.label)
    if ref is None or not ref.writable:
        raise Unsupported(
            "DELETE", f"{b.label!r} has no registered table to write to"
        )
    remaining = tuple(x for x in plan.scope.bindings if x.name != var)
    return Delete(
        plan.scope.with_bindings(remaining), child=plan, var=var, label=b.label
    )


def rule_create(clause: Any, plan: Plan, catalog: Catalog, last: bool) -> Plan:
    from pycypher.ast_models import NodePattern

    if not isinstance(plan, Unit) or not last:
        raise Unsupported(
            "CREATE", "only a standalone single-node CREATE is supported"
        )
    paths = clause.pattern.paths if clause.pattern is not None else []
    if (
        len(paths) != 1
        or paths[0].variable is not None
        or len(paths[0].elements) != 1
    ):
        raise Unsupported(
            "CREATE", "only a standalone single-node CREATE is supported"
        )
    node = paths[0].elements[0]
    if not (isinstance(node, NodePattern) and len(node.labels) == 1):
        raise Unsupported("CREATE", "node needs exactly one label")
    ref = catalog.entity(node.labels[0])
    if ref is None or not ref.writable:
        raise Unsupported(
            "CREATE", f"{node.labels[0]!r} has no registered table to write to"
        )
    if not ref.id_is_integer:
        raise Unsupported(
            "CREATE", "id column must be integer for sequence-based ids"
        )
    props = node.properties or {}
    if any(key not in ref.attr for key in props):
        raise Unsupported("CREATE", "unknown property")
    sc = ScopeCompiler(Scope(), catalog)
    for key, expr in props.items():
        sc.expr(expr, f"CREATE property {key}")
    return Create(
        Scope(), label=node.labels[0], properties=tuple(props.items())
    )


def _register_rules() -> dict[type, Rule]:
    from pycypher.ast_models import Create as CreateClause
    from pycypher.ast_models import Delete as DeleteClause
    from pycypher.ast_models import Match, Return, Set, Unwind, With

    return {
        Match: rule_match,
        With: rule_with,
        Return: rule_return,
        Unwind: rule_unwind,
        Set: rule_set,
        DeleteClause: rule_delete,
        CreateClause: rule_create,
    }


RULES: dict[type, Rule] = _register_rules()

__all__ = ["RULES", "translate"]
