# Generalising Cypher → relational algebra translation

Status: assessment written 2026-09-05; **Phases 0–3 implemented the same
day** (see "Progress" at the end), with several Phase 4 items landing for
free. Decision record: `docs/adr/adr-008-logical-plan-ir.rst`; developer
guide with the operator table and coverage matrix:
`docs/developer_guide/relation_plan.md`. Sibling docs: `duckdb_full_parity_design.md` (how the
current relation engine was grown), `duckdb_eager_path_design.md` (the
BindingFrame path on DuckDB), the FastOpenData streaming-qualification plan (private repository)
(the per-query eligibility campaign that motivated this assessment).

## The question

> Is the code structured to support arbitrarily-structured Cypher queries
> through a small set of recursively applied translation rules, or are we
> locked into writing new code for each novel query structure?

## Short answer

**Locked in, on the path that matters.** There are two execution paths and
they answer the question oppositely:

| | BindingFrame path (`clause_executor.py`, `pattern_matcher.py`, …) | Relation engine (`relation_engine.py`, `relation_sql.py`) |
|---|---|---|
| Structure | One dispatch per **clause type**, folded left-to-right over a bindings table. General: any clause sequence the parser accepts. | One hand-written analyser per **query shape**. A whitelist of clause sequences; everything else is "ineligible". |
| Execution | Eager. pandas frames (or DuckDB-backed frames on `backend_engine: duckdb`, but still materialised at most operator boundaries). | Lazy DuckDB relations compiled to one SQL query per Cypher query. Genuinely out-of-core. |
| Coverage | Full grammar: `MATCH`, `OPTIONAL MATCH`, `WITH`, `RETURN`, `UNWIND`, `SET`, `REMOVE`, `DELETE`, `CREATE`, `MERGE`, `FOREACH`, `CALL`, `UNION`, variable-length and undirected paths, `collect()`, `EXISTS`, comprehensions. | A leading `MATCH` (single node or fixed-length *directed* path) plus `OPTIONAL MATCH`es, `WHERE`, `WITH`/`UNWIND`/`SET` stages, at most one embedded second `MATCH`, `RETURN`; and six separately coded mutation shapes. |
| Correctness on the real pipeline | Wrong in several ways found on 2026-09-05 (see the qualification plan, Phase 4): label-blind edge matching, mid-pipeline `SET` writing to the wrong rows, dropped copy-`SET` values. | Correct on every case checked against source files. |

So the general engine is the one we are trying to retire for memory reasons,
and the out-of-core engine is the one built shape-by-shape. The eligibility
campaign in the FastOpenData streaming-qualification plan (private repository) is the direct
cost of that: Phases 2, 2b-i, 2b-ii, 3a, 3b, 3c, and category (D) each added
a new *shape* — "aggregate then bare-alias SET", "aggregate then DISTINCT",
"SET of a new column", "SET expression over aliases", … — rather than a new
*rule*. The next novel query in a config will need another slice.

## Evidence

Measured against the working tree on 2026-09-05.

- `relation_engine.py` is 3 582 lines with **167 `return None` sites**,
  each one a point where a query is declared ineligible. They are spread
  over shape checks (`len(clauses) != 3`, "exactly one bare grouping
  variable plus aggregates", "every item targets the same variable", "no
  DISTINCT unless an aggregate is present", "single-component scope only",
  "a second MATCH only if immediately preceded by a WITH", …), not over
  unsupported *constructs*.
- Mutations are six parallel triples with no shared plan: `_analyze_set_query`
  / `is_relation_set_eligible` / `execute_relation_set`, and the same for
  `scalar_set`, `group_set`, `copy_set`, `create`, `delete`, tried in a fixed
  order by `is_relation_mutation_eligible`. A `SET` whose value mixes an
  aggregate alias with a property of a *second* node, or a `SET` after two
  `WITH`s, or a `DELETE` after a join, matches none of them and falls back.
  Each triple re-implements pattern analysis, WHERE compilation, the id
  column lookup, `_ensure_column`, and the `UPDATE … FROM` emission.
- The read path has a mini-IR (`_Plan`: base relation builder + a list of
  raw AST stages; `_Scope`: two name-resolution closures plus flags) but the
  stages are still the AST clauses, interpreted twice — once in
  `is_relation_eligible` and again in `execute_relation_query`, in lockstep,
  with `_plan_stage`/`_plan_mixed_stage`/`_stage_is_passthrough`
  distinguishing "WITH of bare nodes", "WITH of bare nodes *and* new
  expressions" and "WITH of expressions" as three cases with different code.
- The AST already declares the intent that was never built:
  `ast_models/core.py` defines `Algebraizable` ("AST node that can be
  translated into a relational algebra operator … implement a
  `to_relation()` method") and `PatternPath`, `NodePattern`,
  `RelationshipPattern`, `PatternIntersection` inherit it. **No
  `to_relation` method exists anywhere in the package.**
- What *is* general and reusable: `relation_sql.compile_expression` is a
  proper recursive expression compiler (literals, property lookups,
  arithmetic, comparisons, boolean logic, `IS NULL`, `CASE`, casts, UDFs,
  `id()`), parameterised by a resolver. It is the one piece of the relation
  engine that already works the way the whole thing should.
- `TableRegistry` + `DuckDBLazyFrame` + the file-backed scratch database
  give a real, mutable, out-of-core table per label. That storage layer is
  the right substrate and does not need to change.

## Why it grew this way

`duckdb_full_parity_design.md` chose "Approach A": an opt-in path that
covers an *eligible subset*, growing one feature at a time so the suite
stays green and every ineligible query falls back to the known-good engine.
That was a sound way to get from zero to a working out-of-core path, and it
is why the fastopendata pipeline streams today. Its failure mode is exactly
the one raised in the question: eligibility was defined by enumerating
shapes, so the code accreted shape-specific branches, and the fallback
engine turned out to be a poor oracle (its own bugs went unnoticed until the
streaming path produced *different* numbers and source-level ground-truth
was checked).

## Target design

One translation, structured as a fold over the clause list, producing a
small logical-plan IR that a single emitter turns into DuckDB SQL. Cypher's
own semantics already say what the rule is: **every clause consumes a table
of variable bindings and produces a table of variable bindings.** That is
precisely the BindingFrame model of ADR-002, made lazy. The design below
keeps that model and replaces both the eager executor and the shape
whitelist with one recursive translator.

### The IR

A `Plan` node is an immutable dataclass with typed children; every node
exposes `schema()` → ordered `Binding`s. A `Binding` is a Cypher variable
name plus its kind (`node(label)`, `rel(type)`, `scalar`, `list`, `path`)
and the physical columns that carry it (a node is its id column plus the
label; properties are resolved lazily by the emitter as `alias."col"`
against the label's registered table, exactly as `_make_resolve` does now).

Operators — deliberately few:

| Node | Meaning | Cypher source |
|---|---|---|
| `Unit` | one row, no columns | clause-first queries, leading `WITH` of constants |
| `Scan(label, var)` | all rows of a registered entity table | `(n:Label)` |
| `Expand(child, from_var, rel_type, dir, to_var, rel_var, hops)` | join `child` to a relationship table and the far node's table; `hops` is `1` or a range (recursive CTE) | `-[r:T]->`, `<-[:T]-`, `-[:T]-`, `-[:T*1..3]->` |
| `Join(left, right, on=shared vars)` / `Cross` | inner join on shared variables, else cross | a further `MATCH` |
| `LeftJoin(left, right, on)` | left outer join | `OPTIONAL MATCH` |
| `Filter(child, expr)` | keep rows | `WHERE`, inline `{prop: v}` |
| `Project(child, items, distinct)` | compute columns; when any item aggregates, group by the others | `WITH`, `RETURN` |
| `Unnest(child, list_expr, var)` | lateral unnest | `UNWIND` |
| `Sort(child, keys)` / `Limit(child, skip, n)` | | `ORDER BY`, `SKIP`, `LIMIT` |
| `Union(left, right, all)` | | `UNION [ALL]` |
| `SemiJoin` / `AntiJoin(child, pattern_plan)` | pattern predicate | `WHERE (n)-[:T]->()`, `EXISTS { … }`, `NOT …` |
| `Mutate(child, kind, target_var, assignments)` | side effect against the target's physical table, keyed by the target's id column; passes `child` through, with the assigned values folded in | `SET`, `REMOVE`, `DELETE`, `DETACH DELETE`, `CREATE`, `MERGE` |

A `Mutate` node is the single replacement for all six mutation kinds and the
mid-pipeline `SET` stage: `UPDATE t SET … FROM (<child SQL>) sub WHERE
t.id = sub.<target id>`. "Aggregate then SET" is `Mutate(Project(Expand(Scan)))`;
"copy SET" is `Mutate(Expand(Scan))`; "scalar SET" is `Mutate(Scan)`. The
`_ensure_column` type probe becomes "project the assignment expression over
the child plan with `LIMIT 0` and read `.types`", which is what the group
slice already does.

### The translator

```
translate(query: Query) -> Plan
    plan, scope = Unit, {}
    for clause in query.clauses:
        plan, scope = RULES[type(clause)](clause, plan, scope, ctx)
    return plan
```

`RULES` has one entry per clause type — roughly a dozen — and each rule is
a few lines because it only has to say how *its* clause maps onto the
operators above. Patterns are translated by a second small recursion over
`PatternPath` elements (`Scan` for the first node, `Expand` per hop, `Join`
on already-bound variables — this is the `to_relation()` the AST promised).
Expressions are compiled by `compile_expression` as now, with the resolver
built from `scope`.

A clause the translator cannot handle raises `Unsupported(node, reason)`
from exactly one place — the missing `RULES` entry or an expression
construct the compiler lacks — and the caller decides whether to fall back.
"Eligibility" stops being a separate pass that mirrors execution; it *is*
plan construction. There is no second walk.

### The emitter

`emit(plan) -> str` (or a `DuckDBPyRelation`) is a post-order walk with one
method per node type. All alias generation, `QUALIFY`/`GROUP BY`/`HAVING`
placement, recursive CTEs for variable-length paths, and the `UPDATE … FROM`
form live here, once. The emitter is the only file that knows SQL syntax.

### What this buys

- A novel query is handled by composition. `MATCH … WITH … MATCH … WITH …
  SET … RETURN` needs no new code because each clause already has a rule.
- Rejections are per-construct, so the coverage matrix is a list of
  constructs (`collect()`, `MERGE`, `shortestPath`, …), each closable by one
  rule or one emitter method, instead of a growing list of shapes.
- Ground truth for tests can be the IR itself: golden plans per query, and
  algebraic property tests (`Filter` commutes with `Project` when it doesn't
  reference projected-away columns, `Join` associativity, etc.).
- The BindingFrame engine becomes unnecessary for `backend_engine: duckdb`;
  it can be kept for the pandas backend or retired.

## Phased plan

Every phase ends green on the whole suite, lands with tests and docs, and
is independently useful. Estimates assume one engineer familiar with the
code.

### Phase 0 — Characterise and freeze behaviour (1 week)

Goal: a test oracle that is not the pandas engine.

- **Golden corpus.** Collect every Cypher query the repo already exercises:
  the 272 in the FastOpenData pipeline config (private repository), every query string in
  `tests/test_relation*.py` and `tests/test_golden_ir.py`, plus a curated
  set covering each grammar production. Store as `tests/corpus/*.cypher`
  with a small fixed dataset (extend `test_golden_ir.py`'s fixtures) and
  **hand-verified** expected outputs, checked in. Where the pandas engine
  disagrees with the hand-verified answer, record it as a known pandas bug
  (three are already known: label-blind edge matching, misaligned
  mid-pipeline `SET`, dropped copy-`SET` values).
- **Differential harness.** `tests/test_engine_parity.py` runs the corpus
  through whatever engines exist and compares to the golden outputs with
  numeric tolerance and normalised booleans, reporting per-query
  `eligible / correct / fallback`. This replaces the pandas-as-oracle
  comparisons scattered across the relation tests.
- **Coverage report.** A script that runs `is_relation_eligible` over the
  corpus and prints the rejection reason per query, so Phase 2's progress is
  a number.
- Docs: `docs/testing/cypher_corpus.md` explaining how to add a query and
  its expected output.

### Phase 1 — IR, emitter, and expression compiler as a package (2 weeks)

Goal: the new machinery exists and is unit-tested, but nothing routes
through it yet.

- New package `pycypher/plan/`: `nodes.py` (the dataclasses above, with
  `schema()`), `emit_duckdb.py` (post-order SQL emitter), `scope.py`
  (bindings and resolvers), `errors.py` (`Unsupported`).
- Move `relation_sql.compile_expression`/`compile_aggregate` under
  `pycypher/plan/expressions.py` unchanged in behaviour; the resolver
  interface becomes the `Scope`. `relation_sql` stays as a thin re-export
  for one release.
- Tests: one test module per node type asserting the emitted SQL and the
  executed result on a tiny DuckDB fixture; property-based tests (Hypothesis
  is already a dev dependency) generating random plans over a fixed schema
  and checking `emit` produces valid SQL and `schema()` matches the result's
  columns.
- Docs: ADR-008 "Logical plan IR for the relation engine" (`docs/adr/`),
  following ADR-002's format; a developer-guide page with the operator
  table above and one worked example per operator.

### Phase 2 — Read path through the translator (2 weeks)

Goal: `translate()` handles everything `is_relation_eligible` handles today,
and the old read analysers are deleted.

- Implement `RULES` for `Match` (required and optional), `Unwind`, `With`,
  `Return`; pattern translation for single nodes and fixed-length directed
  paths; the `Set` rule as a `Mutate` node (this also subsumes the
  mid-pipeline `SET` stage).
- `is_relation_eligible` becomes `try: translate(query) except Unsupported:
  return False`. `execute_relation_query` becomes `emit(translate(query))`.
  Keep both names so `star.py` and `cli/pipeline.py` don't change.
- Delete `_analyze_query`, `_analyze_leading_pattern`,
  `_analyze_optional_pattern`, `_analyze_second_match`, `_plan_stage`,
  `_plan_mixed_stage`, `_stage_is_passthrough`, `_set_stage_eligible`,
  `_scope_after_set_stage`, `_execute_set_stage`, `_Plan`, `_Scope`.
- Exit criteria: the Phase 0 coverage report shows every previously
  eligible corpus query still eligible; golden outputs unchanged; the
  fastopendata sample run still streams all 272 queries with identical
  outputs (the Phase 4 comparison in the qualification plan is the
  procedure).
- Tests: the existing `test_relation_engine*.py` read tests are kept and
  must pass unchanged — they are the regression suite for this phase. Add
  golden-plan tests (`translate(q)` → expected IR, as a repr) for the
  corpus, so a future change to translation is a visible diff.
- Docs: update the module docstring of `relation_engine.py` and the
  "Eligible subset" prose in `duckdb_full_parity_design.md` to point at the
  coverage matrix instead of listing shapes.

### Phase 3 — Mutations through `Mutate` (1–2 weeks)

Goal: the six mutation triples are gone.

- `Mutate` handles `SET` (all value shapes), `REMOVE`, `DELETE`, `DETACH
  DELETE`, `CREATE` of nodes and relationships, keyed on the target's id
  column, with the same "unmatched rows untouched, new columns via
  `ALTER TABLE ADD COLUMN`" semantics the slices established.
- Delete `_analyze_single_node_match`, `_analyze_set_query`,
  `_analyze_scalar_set_query`, `_analyze_group_set_query`,
  `_analyze_copy_set_query`, `_analyze_delete_query`,
  `_analyze_create_query`, their `is_relation_*_eligible` and
  `execute_relation_*` partners, and `_agg_alias_resolver`.
  `is_relation_mutation_eligible` returns `"mutation"` for any query whose
  plan contains a `Mutate` and no `Return`; `execute_relation_mutation`
  emits it.
- Tests: the mutation unit tests (`test_relation_engine_{set,scalar_set,
  group_set,copy_set,create,delete}_unit.py`) are kept and must pass; add
  cases the slices could never express — `SET` after two `WITH`s, `SET` on
  two variables in one clause, `DELETE` after a join, `SET` value mixing an
  alias and a second node's property.
- Docs: retire the "Phase 2 / 2b" shape taxonomy in the qualification plan
  with a pointer here.

### Phase 4 — Coverage by rule (3–4 weeks, parallelisable per item)

Each item is one rule or one emitter method plus tests and a row in the
coverage matrix. Order by pipeline leverage; none block the others.

1. Multiple required `MATCH` clauses anywhere → `Join` on shared variables,
   `Cross` otherwise (today: one, only after a `WITH`).
2. Undirected relationships → `Expand` with a `UNION ALL` of both
   orientations in the emitter.
3. Variable-length paths `[*m..n]` → recursive CTE with a hop counter and
   a visited-list guard; `shortestPath` as the minimum-hop filter on top.
4. Pattern predicates and `EXISTS { }` → `SemiJoin`/`AntiJoin`.
5. `collect()` → DuckDB `LIST()`; list comprehensions and `size()`,
   `head()`, `[..]` slicing → `list_transform`/`list_filter`/`list_slice`.
6. `OPTIONAL MATCH` followed by aggregation → `LeftJoin` then `Project`
   with `COUNT(<far node id>)` (the current engine refuses this only
   because its `COUNT(node) → COUNT(*)` shortcut over-counts; with an id
   column per binding the shortcut is unnecessary).
7. `UNION` / `UNION ALL` → `Union`.
8. `MERGE` → `Mutate(kind="merge")`: anti-join to find missing rows, insert,
   then behave as `MATCH`.
9. `FOREACH` → unnest then `Mutate`.
10. `CALL … YIELD` → a procedure registry mapping to table functions;
    out of scope for the pipeline, so last.
11. Relationship endpoint labels — **already done in the current engine
    (2026-09-05)**: `register_streaming_relationship` records
    `source_entity_type`/`target_entity_type` per edge in reserved
    `__SOURCE_LABEL__`/`__TARGET_LABEL__` columns and every pattern join
    filters on them. `Expand` must carry that predicate over unchanged.

Tests per item: unit tests for the rule, corpus queries exercising it with
hand-verified outputs, and a property test where the construct has an
algebraic identity (e.g. an undirected expand equals the union of the two
directed ones; `[*1..1]` equals a single hop).

### Phase 5 — One engine per backend (1 week)

- For `backend_engine: duckdb`, `Star.execute_query` always calls
  `emit(translate(query))`; `Unsupported` is a hard error with the
  construct named, not a silent fallback. `_try_streaming_run`'s
  pre-check and `_warn_streaming_fallback` become dead and are removed.
- The BindingFrame path stays only for `backend_engine: pandas`. Decide
  then whether to keep it (small in-memory graphs, no DuckDB dependency) or
  route pandas through DuckDB's `from_df`.
- Docs: rewrite `docs/user_guide` execution section; mark
  `duckdb_full_parity_design.md` and the eager-path docs as historical.

### Phase 6 — Plan-level optimisation (optional, 1 week)

DuckDB's optimiser does projection pruning, predicate pushdown and join
ordering on the emitted SQL, so this phase is small: dead-binding
elimination between `Mutate` nodes (so a 40-stage pipeline query doesn't
carry every property through every stage), and de-duplication of identical
`Scan`s within one plan. Measure first; skip if DuckDB already handles it.

## Testing strategy, summarised

| Layer | What | Where |
|---|---|---|
| Rule unit tests | one clause → expected IR fragment | `tests/plan/test_rules_*.py` |
| Emitter unit tests | one IR node → expected SQL and result | `tests/plan/test_emit_*.py` |
| Golden plans | corpus query → IR repr, checked in | `tests/plan/golden/` |
| Golden outputs | corpus query → hand-verified rows | `tests/corpus/` |
| Differential | every engine vs golden outputs, per-query status | `tests/test_engine_parity.py` |
| Property-based | random plans: valid SQL, schema agreement, algebraic identities | `tests/plan/test_properties.py` |
| Real pipeline | `make nmetl-go-sample` streams all queries; outputs byte-compared to the previous run | qualification plan Phase 4 procedure |
| Existing suites | all `test_relation_engine*` tests kept through Phases 2–3 as regression guards | unchanged |

## Documentation deliverables

- `docs/adr/adr-008-logical-plan-ir.rst` — decision and consequences.
- `docs/developer_guide/relation_plan.md` — the operator table, the fold,
  one worked example (a real pipeline query shown as Cypher → IR → SQL).
- `docs/developer_guide/coverage_matrix.md` — generated from the `RULES`
  table and the emitter's method registry by a small script run in CI, so
  it cannot go stale.
- Module docstrings for `pycypher/plan/*`.
- Retirement notes in the three sibling design docs.

## Risks

- **Semantics drift while rewriting.** Mitigated by Phase 0 coming first
  and by keeping every existing relation-engine test through Phases 2–3.
- **DuckDB relation-API quirks** (component aliases lost after `.project()`
  on a joined relation, which forced today's "single-component scope"
  restriction). The emitter should produce SQL text with explicit aliases
  and subqueries rather than chaining `DuckDBPyRelation` methods; this
  sidesteps the alias problem entirely.
- **Variable-length paths on large graphs.** Recursive CTEs are correct but
  can be slow without a depth cap; keep the current 1M-frontier style cap
  as a `Limit` inside the CTE and document it.
- **The pandas engine's bugs.** They are now known to exist; until Phase 5
  any query that still falls back may produce wrong numbers silently. The
  Phase 0 differential report should be run in CI so a fallback is visible.

## Success criteria

- A query built from any combination of the constructs in the coverage
  matrix translates without code changes.
- `relation_engine.py` shrinks from ~3 600 lines to a translator, a rule
  table and an emitter that together fit in well under a third of that.
- Zero `return None` shape checks; rejections name a construct.
- The fastopendata pipeline runs entirely through the plan path with
  outputs equal to the ground-truthed 2026-09-05 sample results.

## Progress (2026-09-05)

- **Phase 0** — done as `tests/test_plan_corpus.py`: a hand-verified
  corpus (23 read queries, 8 mutation cases, 9 named unsupported
  constructs) over a fixed graph, plus `pycypher.plan.report` for the
  per-construct coverage summary. The pandas engine is deliberately not
  used as an oracle.
- **Phase 1** — done: `pycypher/plan/` (`nodes`, `catalog`, `expressions`,
  `translate`, `emit_duckdb`, `errors`, `report`). The expression compiler
  stayed in `relation_sql.py` and is wrapped, not moved.
- **Phase 2** — done: `is_relation_eligible`/`execute_relation_query` are
  `translate`/`emit`. The old read analysers are deleted.
- **Phase 3** — done: one `Mutate` rule replaces the six mutation triples;
  `is_relation_mutation_eligible` still returns the old kind names, now
  purely descriptive, and the per-kind functions are wrappers.
  `relation_engine.py` is ~980 lines (was 3 582).
- **Phase 4** landed early for items 1 (any number of `MATCH` clauses,
  joined on shared variables), 6 (`OPTIONAL MATCH` then aggregation,
  via `count(<node>)` counting the id) and 11 (endpoint labels), plus
  `UNWIND` in pattern scope and `SET` after a join. Nine old tests that
  asserted these shapes were *ineligible* were flipped to positive parity
  tests. Remaining Phase 4 items are listed in the developer guide's
  coverage matrix.
- **Verification**: 12 631 tests pass; the fastopendata sample pipeline
  streams all 272 queries with outputs byte-identical to the shape-based
  engine's.
- **One design correction versus the plan above**: carrying only ids and
  joining properties back on demand made a single-table scan a self-join,
  which failed the memory-limited streaming test. The emitter therefore
  fuses a Scan/Expand/Filter chain with the projection that consumes it
  (one SELECT, properties read off the pattern's own aliases) and only
  falls back to id-joins after a projection. The IR is unchanged; this is
  an emitter concern, as the plan intended.
- **Phase 5 — done (2026-09-06)**, scoped to "relation engine enabled":
  with `relation_engine: true` (or the env var) on a DuckDB backend,
  `Star.execute_query`, `Star.stream_query_to_uri` and `nmetl run` never
  fall back. `Unsupported` is a `ValueError` naming the construct; the CLI
  reports it per query under `--on-error`, executes sink-less reads for
  their side effects, and treats a source that fails to register as a hard
  error. `_warn_streaming_fallback` and the whole-run fallback are gone.
  In-memory-only labels are materialised into registry tables on first
  use so they are writable. With the engine *off*, `backend_engine:
  duckdb` still runs the BindingFrame engine — that path keeps the
  constructs the plan does not yet cover (see the coverage matrix)
  reachable, so retiring it entirely waits on Phase 4's remaining items.
- **Phase 6 — done (2026-09-06)**, narrowly: a `Mutate`/`Delete` at the
  top of the plan runs as one DML statement over the fused SELECT (no
  temporary table, values computed in the same pass as the pattern), which
  removes the regression the temp tables had introduced. Dead-binding
  elimination and scan de-duplication were not needed: DuckDB's optimiser
  handles the emitted SQL.
