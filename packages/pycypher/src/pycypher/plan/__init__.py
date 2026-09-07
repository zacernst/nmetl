"""Logical-plan translation of Cypher for the out-of-core relation engine.

See ``docs/cypher_relational_algebra_generalization_plan.md``. The package
replaces the shape-by-shape analysers that used to live in
``relation_engine.py`` with three small pieces:

* :mod:`pycypher.plan.nodes` — an immutable relational-algebra IR whose
  every node carries the :class:`~pycypher.plan.nodes.Scope` (the Cypher
  variables bound after it, in the BindingFrame sense of ADR-002: a node
  or relationship variable is carried as its id, a scalar as its value;
  properties are looked up lazily by the emitter).
* :mod:`pycypher.plan.translate` — a fold of one rule per clause type over
  the clause list. A construct with no rule raises
  :class:`~pycypher.plan.errors.Unsupported`; eligibility *is* successful
  translation.
* :mod:`pycypher.plan.emit_duckdb` — a post-order walk that turns a plan
  into one DuckDB ``SELECT`` per read stage, and runs ``Mutate``/
  ``Delete``/``Create`` nodes as native DML against the registry's tables.
"""

from pycypher.plan.errors import Unsupported
from pycypher.plan.translate import translate

__all__ = ["Unsupported", "translate"]
