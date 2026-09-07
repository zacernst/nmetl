"""Coverage reporting: which queries translate, and what blocks the rest.

``classify`` names the construct that stops a query from translating, so a
pipeline's ineligible queries can be grouped by *what is missing* rather
than listed one by one. ``summarize`` turns a batch into a construct →
query-ids table, which is what the developer-guide coverage matrix is
generated from.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from pycypher.plan.errors import Unsupported
from pycypher.plan.nodes import has_side_effects
from pycypher.plan.translate import translate

READ = "read"
MUTATION = "mutation"
UNSUPPORTED = "unsupported"


def classify(query: Any, context: Any) -> tuple[str, str | None]:
    """Return ``(status, construct)`` for one parsed *query*.

    *status* is ``"read"`` (translates and ends in RETURN), ``"mutation"``
    (translates, side effects, no RETURN), or ``"unsupported"`` with the
    blocking *construct* named.
    """
    from pycypher.ast_models import Return

    try:
        plan = translate(query, context)
    except Unsupported as exc:
        return UNSUPPORTED, exc.construct
    ends_in_return = bool(query.clauses) and isinstance(
        query.clauses[-1], Return
    )
    if ends_in_return:
        return READ, None
    if has_side_effects(plan):
        return MUTATION, None
    return UNSUPPORTED, "query without RETURN or side effects"


def summarize(
    queries: Iterable[tuple[str, Any]], context: Any
) -> dict[str, list[str]]:
    """Group query ids by outcome: ``{"read": [...], "mutation": [...],
    "unsupported:<construct>": [...]}``.
    """
    out: dict[str, list[str]] = defaultdict(list)
    for qid, query in queries:
        status, construct = classify(query, context)
        key = f"{status}:{construct}" if construct else status
        out[key].append(qid)
    return dict(out)


__all__ = ["MUTATION", "READ", "UNSUPPORTED", "classify", "summarize"]
