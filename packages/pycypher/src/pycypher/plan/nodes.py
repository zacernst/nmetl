"""The relational-algebra IR.

Every node is an immutable dataclass carrying the :class:`Scope` of Cypher
variables bound *after* it. Columns of the table a node denotes are named
exactly after those variables: a ``node``/``rel`` binding's column holds
the entity's id, a ``scalar`` binding's column holds its value. Nothing
else is carried between stages; the emitter joins an entity's table back
in whenever a stage reads one of its properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NODE = "node"
REL = "rel"
SCALAR = "scalar"


@dataclass(frozen=True)
class Binding:
    """One Cypher variable in scope."""

    name: str
    kind: str  # NODE | REL | SCALAR
    label: str | None = None  # entity label / relationship type


@dataclass(frozen=True)
class Scope:
    """Variables bound after a plan node, in order.

    *qualified* reproduces the pandas engine's output-naming rule: a bare
    ``n.prop`` return item is named ``n.prop`` when more than one pattern
    variable is in scope, else ``prop``. *new_props* records ``(var, prop)``
    pairs an earlier ``SET`` in the same query created, so a later stage
    can resolve them before the column physically exists.
    """

    bindings: tuple[Binding, ...] = ()
    qualified: bool = False
    new_props: frozenset[tuple[str, str]] = frozenset()

    def get(self, name: str) -> Binding | None:
        for b in self.bindings:
            if b.name == name:
                return b
        return None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bindings)

    def with_bindings(
        self, bindings: tuple[Binding, ...], *, qualified: bool | None = None
    ) -> Scope:
        return Scope(
            bindings,
            self.qualified if qualified is None else qualified,
            self.new_props,
        )

    def with_new_props(self, pairs: frozenset[tuple[str, str]]) -> Scope:
        return Scope(self.bindings, self.qualified, self.new_props | pairs)


@dataclass(frozen=True)
class PropEq:
    """An inline pattern predicate ``{prop: value}`` on *var*."""

    var: str
    prop: str
    value: Any  # expression AST


@dataclass(frozen=True)
class ProjectItem:
    """One ``WITH``/``RETURN`` item.

    Exactly one of *expr* (an expression AST evaluated over the input
    scope) or *passthrough* (a bound node/relationship variable carried
    through by id) is set. *aggregate* marks an aggregating *expr*.
    """

    name: str
    expr: Any = None
    passthrough: str | None = None
    aggregate: bool = False


@dataclass(frozen=True)
class Plan:
    """Base of every IR node; *scope* is what is bound after it."""

    scope: Scope

    @property
    def children(self) -> tuple[Plan, ...]:
        return ()


@dataclass(frozen=True)
class Unit(Plan):
    """One row, no columns — the seed for clause-first queries."""


@dataclass(frozen=True)
class Scan(Plan):
    """All ids of an entity label, bound to *var*."""

    var: str = ""
    label: str = ""


@dataclass(frozen=True)
class Expand(Plan):
    """Join one relationship hop onto *child*.

    From the bound node *from_var*, traverse edges of *rel_type* in the
    given direction to a new node *to_var* of *to_label*, optionally
    binding the edge as *rel_var*. *optional* makes both joins LEFT.
    """

    child: Plan = None  # type: ignore[assignment]
    from_var: str = ""
    rel_type: str = ""
    right: bool = True  # (from)-[]->(to) when True, (from)<-[]-(to) when False
    to_var: str = ""
    to_label: str = ""
    rel_var: str | None = None
    optional: bool = False

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Filter(Plan):
    """Keep rows of *child* satisfying every predicate."""

    child: Plan = None  # type: ignore[assignment]
    predicates: tuple[Any, ...] = ()  # expression ASTs or PropEq

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Project(Plan):
    """A ``WITH``/``RETURN`` stage.

    When any item aggregates, the non-aggregating items form the GROUP
    BY. *where* is evaluated over the *output* scope (Cypher's
    ``WITH … WHERE``), *order_by* names output columns.
    """

    child: Plan = None  # type: ignore[assignment]
    items: tuple[ProjectItem, ...] = ()
    distinct: bool = False
    where: Any = None
    order_by: tuple[tuple[str, bool], ...] = ()  # (output name, ascending)
    skip: int | None = None
    limit: int | None = None

    @property
    def aggregating(self) -> bool:
        return any(it.aggregate for it in self.items)

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Unnest(Plan):
    """``UNWIND`` a list-valued expression into *var*, keeping *child*'s columns."""

    child: Plan = None  # type: ignore[assignment]
    expr: Any = None
    var: str = ""

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Join(Plan):
    """Join *child* with an independent pattern *right* on shared variables
    (a cross join when none are shared).
    """

    child: Plan = None  # type: ignore[assignment]
    other: Plan = None  # type: ignore[assignment]
    on: tuple[str, ...] = ()

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child, self.other)


@dataclass(frozen=True)
class Mutate(Plan):
    """``SET`` properties on the node bound to *target* for every row of
    *child*, as a native ``UPDATE … FROM``. Passes *child*'s rows through.
    """

    child: Plan = None  # type: ignore[assignment]
    target: str = ""
    label: str = ""
    assignments: tuple[tuple[str, Any], ...] = ()  # (property, expression AST)

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Delete(Plan):
    """``DELETE`` the nodes bound to *var* in *child*."""

    child: Plan = None  # type: ignore[assignment]
    var: str = ""
    label: str = ""

    @property
    def children(self) -> tuple[Plan, ...]:
        return (self.child,)


@dataclass(frozen=True)
class Create(Plan):
    """Standalone ``CREATE (v:Label {props})`` as a native ``INSERT``."""

    label: str = ""
    properties: tuple[tuple[str, Any], ...] = ()


def has_side_effects(plan: Plan) -> bool:
    """True if *plan* contains a Mutate, Delete, or Create node."""
    if isinstance(plan, (Mutate, Delete, Create)):
        return True
    return any(has_side_effects(c) for c in plan.children)


def describe(plan: Plan, indent: int = 0) -> str:
    """A stable, human-readable rendering (used by golden-plan tests)."""
    pad = "  " * indent
    name = type(plan).__name__
    attrs = []
    for f in plan.__dataclass_fields__:  # noqa: SLF001
        if f in ("scope", "child", "other"):
            continue
        v = getattr(plan, f)
        if v in (None, (), "", False):
            continue
        attrs.append(f"{f}={_short(v)}")
    line = (
        f"{pad}{name}({', '.join(attrs)}) -> [{', '.join(plan.scope.names)}]"
    )
    return "\n".join([line, *(describe(c, indent + 1) for c in plan.children)])


def _short(v: Any) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(_short(x) for x in v) + ")"
    if isinstance(v, ProjectItem):
        return v.name if v.passthrough else f"{v.name}={_short(v.expr)}"
    if isinstance(v, PropEq):
        return f"{v.var}.{v.prop}=…"
    if hasattr(v, "model_dump"):
        return type(v).__name__
    return repr(v) if isinstance(v, str) else str(v)


__all__ = [
    "NODE",
    "REL",
    "SCALAR",
    "Binding",
    "Create",
    "Delete",
    "Expand",
    "Filter",
    "Join",
    "Mutate",
    "Plan",
    "Project",
    "ProjectItem",
    "PropEq",
    "Scan",
    "Scope",
    "Unit",
    "Unnest",
    "describe",
    "has_side_effects",
]
