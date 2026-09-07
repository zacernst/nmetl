"""Errors raised by the plan translator and emitter."""

from __future__ import annotations


class Unsupported(ValueError):  # noqa: N818 — named in the design doc; a category, not an error suffix
    """A Cypher construct the plan translator has no rule for.

    Raised from exactly one kind of place — a missing rule, or an
    expression the expression compiler declines — and named by
    *construct* so a caller (or a coverage report) can say *what* is
    missing rather than merely that a query is ineligible.

    A ``ValueError`` so that, when the relation engine is explicitly
    enabled and no fallback exists, it surfaces through the CLI's
    per-query error policy like any other query failure.
    """

    def __init__(self, construct: str, reason: str = "") -> None:
        self.construct = construct
        self.reason = reason
        detail = f"{construct}: {reason}" if reason else construct
        super().__init__(
            f"unsupported by the out-of-core relation engine: {detail}"
        )
