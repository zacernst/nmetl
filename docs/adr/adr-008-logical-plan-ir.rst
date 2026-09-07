ADR-008: A logical-plan IR for the out-of-core relation engine
==============================================================

:Status: Accepted
:Date: 2026-09-05
:Affects: ``packages/pycypher/src/pycypher/plan/``,
          ``packages/pycypher/src/pycypher/relation_engine.py``

Context
-------

The relation engine — the path that compiles Cypher to DuckDB SQL so a
pipeline can run out-of-core — grew shape by shape. Each new query
structure in the fastopendata pipeline needed its own analyser
(``_analyze_group_set_query``, ``_analyze_copy_set_query``, a
"mixed WITH" planner, a "single-component scope" rule, …), and
eligibility was a whitelist of clause sequences with 167 separate
rejection sites. A novel query meant new code, and the eager pandas
engine it fell back to turned out to have correctness bugs of its own.
``docs/cypher_relational_algebra_generalization_plan.md`` records the
assessment.

Decision
--------

Translate Cypher into a small relational-algebra IR by a fold of **one
rule per clause type**, and emit SQL from the IR by **one method per
operator**. Nothing enumerates query shapes.

* **IR** (``plan/nodes.py``): ``Unit``, ``Scan``, ``Expand``, ``Filter``,
  ``Project``, ``Unnest``, ``Join``, ``Mutate``, ``Delete``, ``Create``.
  Every node carries a ``Scope``: the Cypher variables bound after it.
* **Binding model**: the BindingFrame model of ADR-002, made lazy. A
  node or relationship variable is carried as its id; a scalar as its
  value; properties are resolved by the emitter, never carried between
  stages.
* **Translator** (``plan/translate.py``): ``translate(query, context)``
  seeds ``Unit`` and applies ``RULES[type(clause)]`` per clause. A
  construct with no rule raises ``Unsupported(construct)``. Eligibility
  *is* successful translation; there is no second walk.
* **Emitter** (``plan/emit_duckdb.py``): renders read plans as SQL and
  runs ``Mutate``/``Delete``/``Create`` as native DML, materialising the
  rows feeding a side effect into a temporary table so later stages
  read the updated values.
* **Pattern-block fusion**: a Scan/Expand/Filter chain and the Project
  that consumes it become one SELECT with a table alias per variable,
  so a single-table scan never becomes a self-join. After a projection,
  properties are LEFT JOINed back on the id.

Consequences
------------

* A query built from any combination of supported constructs
  translates without new code. Multiple ``MATCH`` clauses, ``SET`` after
  a join or after aggregation, ``UNWIND`` in any scope, and
  ``OPTIONAL MATCH`` followed by aggregation all work now and needed no
  rule of their own.
* ``relation_engine.py`` shrank from ~3 600 lines to ~980; the six
  mutation "kinds" are descriptive names over one ``Mutate`` rule.
* Rejections name a construct. The coverage matrix
  (``docs/developer_guide/relation_plan.md``) lists what is left.
* Correctness is pinned by a hand-verified golden corpus
  (``tests/test_plan_corpus.py``) and golden plans
  (``tests/test_plan_translate.py``), not by parity with the pandas
  engine.
* The eager BindingFrame path remains the fallback for unsupported
  constructs and for ``backend_engine: pandas``. Retiring it is Phase 5
  of the generalisation plan.
