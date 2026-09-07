# The relation plan: Cypher → IR → DuckDB SQL

Package: `packages/pycypher/src/pycypher/plan/`. Decision record:
[ADR-008](../adr/adr-008-logical-plan-ir.rst). Background and phased plan:
[`cypher_relational_algebra_generalization_plan.md`](../cypher_relational_algebra_generalization_plan.md).

## The one rule

Every Cypher clause consumes a table of variable bindings and produces a
table of variable bindings. `translate()` folds one rule per clause type
over the clause list:

```
plan = Unit
for clause in query.clauses:
    plan = RULES[type(clause)](clause, plan, catalog, is_last)
```

A node or relationship variable is bound as its **id**; a scalar as its
value. Properties are never carried between stages — the emitter reads
them from the entity's table when an expression asks.

## Operators

| Node | Cypher source | Emitted as |
|---|---|---|
| `Unit` | clause-first queries | `SELECT 1` |
| `Scan(var, label)` | `(n:Label)` | the label's table, aliased |
| `Expand(child, from, type, dir, to, rel_var, optional)` | `-[r:T]->`, `<-[:T]-`, `OPTIONAL MATCH` | `JOIN` / `LEFT JOIN` edge table then node table; declared endpoint labels filter in the ON clause |
| `Filter(child, preds)` | `WHERE`, inline `{p: v}` | `WHERE` |
| `Project(child, items, distinct, where, order_by, skip, limit)` | `WITH`, `RETURN` | `SELECT [DISTINCT] … [GROUP BY …]`, wrapped for post-projection `WHERE`/`ORDER BY`/`LIMIT` |
| `Unnest(child, expr, var)` | `UNWIND` | `UNNEST(expr)` |
| `Join(child, other, on)` | a further `MATCH` | `JOIN … ON` shared variables, or `CROSS JOIN` |
| `Mutate(child, target, assignments)` | `SET` | temp table of rows → `UPDATE … FROM` keyed on the target id |
| `Delete(child, var)` | `DELETE` | `DELETE FROM … WHERE id IN (…)` |
| `Create(label, props)` | standalone `CREATE` | `INSERT` with a sequence id |

`Project` groups by every non-aggregating item when any item aggregates; a
passed-through node groups by its id. `count(<node>)` counts the id
column, so unmatched `OPTIONAL MATCH` rows count as zero.

## Pattern-block fusion

A chain of `Scan`/`Expand`/`Filter` is rendered as one FROM clause with a
table alias per variable, and the `Project` (or `Unnest`) that consumes it
is fused into the same SELECT so properties are read directly off those
aliases. This keeps a single-table scan a single scan, which is what
keeps the path out-of-core. After a projection only ids are carried, and a
later property read is a `LEFT JOIN` on the id.

## Side effects

A `Mutate`/`Delete` at the top of the plan — the common pipeline case, a
mutation-only query — runs as **one DML statement** over the fused SELECT:
`UPDATE … FROM (SELECT ids, values … QUALIFY …)` or `DELETE … WHERE id IN
(SELECT …)`, with no temporary table. Only a `Mutate` that later stages
read from materialises its rows into a temporary table first, so those
stages see the new values. A property that does not exist yet is
added with `ALTER TABLE ADD COLUMN`, typed from the assignment expression.
One row per target id reaches the `UPDATE` (a fanned-out pattern is
de-duplicated deterministically). Rows with no match are left untouched,
never zero-filled.

## Worked example

```cypher
MATCH (pu:PUMA)<-[:LOCATED_IN]-(h:HousingSurvey1yr)
WITH pu, COUNT(h) AS n
SET pu.housing_unit_count_1yr = n
```

```
Mutate(target='pu', assignments=(('housing_unit_count_1yr', Variable))) -> [pu, n]
  Project(items=(pu, n=FunctionInvocation)) -> [pu, n]
    Expand(from_var='pu', rel_type='LOCATED_IN', right=False, to_var='h', to_label='HousingSurvey1yr') -> [pu, h]
      Scan(var='pu', label='PUMA') -> [pu]
```

```sql
CREATE TEMP TABLE "__pycypher_plan_…" AS
  SELECT n1."PUMA_FIPS" AS "pu", COUNT(n3."SERIALNO") AS "n"
  FROM "_streaming_source_PUMA" AS n1
  JOIN "_rel_source_LOCATED_IN" AS e2
    ON n1."PUMA_FIPS" = e2."__TARGET__"
   AND (e2."__SOURCE_LABEL__" IS NULL OR e2."__SOURCE_LABEL__" = 'HousingSurvey1yr')
   AND (e2."__TARGET_LABEL__" IS NULL OR e2."__TARGET_LABEL__" = 'PUMA')
  JOIN "_streaming_source_HousingSurvey1yr" AS n3 ON e2."__SOURCE__" = n3."SERIALNO"
  GROUP BY n1."PUMA_FIPS";
UPDATE "_streaming_source_PUMA" SET "housing_unit_count_1yr" = sub."__setval_0__"
  FROM (SELECT t."pu" AS "__id__", t."n" AS "__setval_0__" FROM "__pycypher_plan_…" AS t
        QUALIFY ROW_NUMBER() OVER (PARTITION BY t."pu" ORDER BY t."n") = 1) AS sub
  WHERE "_streaming_source_PUMA"."PUMA_FIPS" = sub."__id__";
```

## Coverage matrix

Supported: `MATCH` (single node, fixed-length directed paths, any number
of `MATCH` clauses joined on shared variables), `OPTIONAL MATCH` (one hop
from a bound node), `WHERE`, inline property maps, `WITH`/`RETURN` with
aggregates, `DISTINCT`, `ORDER BY` on output columns, `SKIP`/`LIMIT`,
`UNWIND` in any scope, `SET` on any bound node (any compilable value,
new properties created), `DELETE` of a bound node, standalone
single-node `CREATE`; expressions per `relation_sql.compile_expression`
(literals, properties, `id()`, arithmetic, comparison, boolean, `IS
NULL`, `CASE`, `toFloat`/`toInteger`, registered UDFs) and
`count/sum/avg/min/max`.

Unsupported, each a named `Unsupported(construct)` and one rule or emitter
method away:

| Construct | Notes |
|---|---|
| undirected relationship | `Expand` with a `UNION ALL` of both orientations |
| variable-length path, `shortestPath` | recursive CTE with a depth cap |
| `OPTIONAL MATCH` with `WHERE`, inline properties, or more than one hop | predicate placement in the LEFT JOIN's ON clause |
| pattern predicates, `EXISTS { }` | `SemiJoin`/`AntiJoin` |
| `collect()`, list comprehensions, slicing | `LIST()` / `list_transform` / `list_filter` |
| `ORDER BY` on a non-output expression, `NULLS FIRST` | sort keys compiled over the input scope |
| `RETURN` of a bare node/relationship | needs a row-to-map representation |
| `UNION` | `Union` node |
| `MERGE`, `FOREACH`, `REMOVE`, `DETACH DELETE`, `CREATE` after `MATCH`, relationship `CREATE`, `CALL` | further `Mutate` kinds / a procedure registry |
| `SET` labels, `SET n = {…}` | — |
| entities without a registered table (in-memory only) for writes | reads work through a view |

Run `pycypher.plan.report.summarize` over a config's queries to see which
of these block a given pipeline.

## No fallback

With the relation engine enabled on a DuckDB backend (`relation_engine:
true` in a pipeline config, or `PYCYPHER_DUCKDB_RELATION_ENGINE`), every
query runs through the plan. A construct without a rule raises
`pycypher.plan.Unsupported`, a `ValueError` naming the construct;
`nmetl run` reports it per query under `--on-error` and never switches the
run to the in-memory engine. A read query with no output sink is executed
for its side effects and its result discarded. Entities that exist only in
a context's in-memory mapping are materialised into registry tables on
first use, so mutations on them are native DML too. With the engine off,
the BindingFrame engine runs as before.

## Tests

| What | Where |
|---|---|
| Hand-verified golden corpus, mutation semantics, unsupported constructs by name | `tests/test_plan_corpus.py` |
| Golden plans and rule-level checks | `tests/test_plan_translate.py` |
| SQL shape per operator, executed semantics of the risky bits | `tests/test_plan_emit.py` |
| Regression guards kept from the shape-based engine | `tests/test_relation_*.py` |
| The real fastopendata config: every query eligible | `test_streaming_eligibility_audit.py` (FastOpenData repository) |
