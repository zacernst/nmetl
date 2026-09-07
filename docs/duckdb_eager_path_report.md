# DuckDB eager path — implementation report

Covers the work carried out against `duckdb_eager_path_design.md` on
2026-08-23/24. That document is the plan and now also records, per phase,
what landed. This one is the summary: what changed, what the plan got wrong,
and what is still open.

## Outcome

All eight phases landed. Phase 8 was first reported as *not achieved* — correctly, at the time — and then closed by building the piece that
analysis identified as missing.

| Phase | Scope | Outcome |
| --- | --- | --- |
| 1a | Table registry | Landed |
| 1b | Lazy `BindingFrame` carrier | Landed |
| 1c | Materialisation tripwire | Landed |
| 2 | Statistics in SQL | Landed |
| 3 | Entity + relationship scans | Landed |
| 4 | Rename / concat / distinct | Landed |
| 5 | WHERE predicates | Landed |
| 6 | Index construction | Landed, but *not* as prescribed |
| 7 | Variable-length paths | Landed, with two deliberate declines |
| 8 | Bounded-RSS | Achieved, after adding streaming registration |

Four follow-up items from the list below were then closed as well: the
file-backed scratch database, streaming source registration, lazy
`BindingFrame.join`, and the `ORDER BY` correctness bug the lazy joins
exposed.

No regressions at any step. Final whole-repo state: **12439 passing**, 43
skipped, 7 xfailed, plus 3 failures in `fastopendata` and `pycypher-tui`
that predate this work and reproduce identically with these changes
reverted.

## What the eager path does now

On `backend_engine: duckdb`, for a query that is *not* relation-engine
eligible:

- Entity and relationship scans return lazy DuckDB relations. Inline
  `{prop: value}` predicates compile to a `WHERE`; endpoint pushdown is a
  semi-join.
- `WHERE` conjuncts compile to SQL where possible, with properties resolved
  by LEFT-joining the entity table. Uncompilable conjuncts fall back to the
  pandas evaluator individually, so a partly-pushable filter still benefits.
- Renames are projections, concats are `UNION ALL`, distinct is lazy.
- Variable-length paths run as a `WITH RECURSIVE` walk instead of a per-hop
  pandas `merge` loop.
- Column statistics are computed by SQL aggregates over a `REPEATABLE`
  sample.
- Property resolution goes through the relation while a frame is still lazy,
  so `VectorizedPropertyStore` is not built at all for single-table queries.

Every one of those routes declines to a `None` return and lets the unchanged
pandas code run when its preconditions do not hold.

## Where the plan was wrong

The plan was written from a read of the code; five of its judgements did not
survive contact with the implementation. These are the substantive results.

**Phase 8's goal was unreachable *as scoped*.** `ContextBuilder.add_entity`
calls `DataSource.read()`, materialising the whole source into Arrow before
any query runs; `register_context_tables` then copies it into DuckDB. Peak
RSS therefore went *up* for in-memory sources — 610 MB vs 535 MB on a 200k×22
workload. That analysis was right, and it named the missing piece exactly, so
the piece was built: `add_entity(..., streaming=True)` scans the file
straight into DuckDB. Peak then falls from 794 MB (eager pandas) / 920 MB
(eager DuckDB) to **382 MB**, or **319 MB** under a 128 MB budget, with
query-time RSS dropping from ~510 MB to under 80 MB. Phases 3–7 are what make
that possible; the eager read was masking them. Phases 3–7 make execution columnar; they
cannot undo an eager load that happens first. Measurements in
`tests/benchmarks/bench_duckdb_eager_path_memory.py`.

**Phase 6's prescription was a regression.** The plan said to make
`GraphIndexManager` return `None` for all four index types on DuckDB. Those
indexes still serve the *pandas fallback*, which is taken whenever a DuckDB
route declines; disabling them would slow the fallback with nothing to
replace it. The correct end state — indexes never *reached* on the DuckDB
path — was already achieved by Phases 3 and 5. Measured: pandas builds
adjacency, property and per-type vectorized stores; DuckDB builds none for
single-table queries.

**Phase 5's prerequisite would have broken the engine.** The plan called for
a whole-query pass so scans could project needed property columns.
`BindingFrame.var_names` is read as *the list of bound Cypher variables*, not
as a column list — property columns leaking into it would corrupt variable
resolution everywhere. Resolving properties by a LEFT join at filter time
avoids the pass entirely and leaves the frame's column set untouched.

**The three-valued-logic risk was not real.** The plan suspected the pandas
path might be quietly wrong about `NOT (x = 5)` when `x` is null.
`boolean_evaluator` implements proper Kleene logic (`NOT null → null`),
matching SQL, so `OR` and `NOT` are safe to push. Pinned by null-bearing
differential tests.

**DuckDB's sampling is deterministic.** The plan expected to lose
reproducibility and to have to loosen tests. `REPEATABLE (seed)` exists; no
test needed changing.

## What probing the API caught that reading it would not

Four defects were found by exercising DuckDB rather than trusting its
documentation or its error behaviour. Each returns a wrong answer or does
nothing, silently.

- **`relation.union()` unions positionally.** With differing column sets it
  does not error — it lines columns up by position and keeps the left
  relation's names, mislabelling data, where `pd.concat` produces the union
  with nulls. It also silently coerces `[1, 2] ∪ ["x"]` to `["1", "2", "x"]`
  where pandas keeps `[1, 2, "x"]`. The lazy concat path now requires
  identical names, order, and types; both divergences have a test pinning the
  pandas behaviour.
- **DuckDB rejects a join between two relations descending from the same
  table** unless aliases are made distinct. The broad `except` around
  relation building swallowed it, so *every* predicate push fell back to
  pandas: results correct, whole suite green, feature completely inert. This
  is why `TestPushdownHappens` asserts `is_lazy is True` and not only row
  counts — a fallback designed to be invisible needs something checking it
  was not taken.
- **`hasattr(frame, "bindings")` materialises the frame.** `hasattr` calls
  the getter, and `bindings` is a materialise-on-access property, so an
  existence check was executing the relation. A hazard the Phase 1b design
  created; worth watching wherever a property does real work.
- **`histogram(col, edges)` bins right-closed**, `(e[i-1], e[i]]`, where
  numpy's are left-closed. Matching `np.histogram` exactly is the wrong
  target — both are correct over the same edges. The tests assert the
  consumer contract instead.

## Pre-existing bugs surfaced, not fixed

- ~~**`ORDER BY` is ignored for grouped aggregation output.**~~ **Fixed** —
  see open item 4 below. It predated the work and became visible because the
  two backends' incidental group order diverges at scale once scans are lazy.
  Anything relying on that ordering was relying on an accident.
- **`DuckDBLazyFrame` caches its first `to_pandas()` forever**, so a frame
  materialised before a mutation keeps serving pre-mutation rows. Harmless
  today because `_base_relation` reads `.relation`, and now documented on
  `RegisteredTable` with a test.

## Deliberate declines

Where matching pandas exactly was impossible, the DuckDB route declines
rather than returning a different answer:

- **`row_limit` on a variable-length path.** pandas fills hop 1, then hop 2,
  trimming at the boundary, so which rows survive depends on hop order and on
  frontier order within a hop; `LIMIT` over the CTE keeps a different subset.
- **A path seed with duplicate start values.** The pandas frontier dedupes on
  `(start_var, tip)` while carrying the seed's other columns, silently
  dropping seed rows. The narrow walk re-joined to the seed keeps them —
  arguably more correct, but different.
- **Undirected traversal**, which the recursive form would give cheaply, was
  *not* added: the pandas path does not support it either, so adding it would
  make the backends differ rather than agree.
- **Null inline-property filters**, uncoercible pushdown ids, shadow overlays,
  and unregistered labels all decline for the same reason.

## Verification

- **Full suite run at every commit**, with no test passing before a change
  and failing after it. Per-phase counts quoted in
  `duckdb_eager_path_design.md` were measured over the `tests/` tree; the
  whole-repo figure above is the final state and is not comparable to them.
- **Differential testing** (`tests/helpers_differential.py`) builds the same
  graph on both backends and asserts identical rows; ~40 query shapes across
  scans, filters, transforms, paths, and null semantics.
- **Materialisation tripwire** (`count_materialisations`) lets tests assert a
  query materialises zero times, which is what caught schema and row-count
  access silently forcing execution.
- **Kill switch**: `register_tables=False` makes every DuckDB route decline;
  results and memory both measure identical to pandas.

## Open items

1. ~~**Point `ContextBuilder.build` at a file-backed scratch database.**~~
   **Done**, with a correction. `DuckDBBackend` gained `own_database_file`
   (deleting the database and its `.wal` on `close()`), and
   `ContextBuilder.build` gained `scratch_database`. But isolating the
   variables overturned the reason for doing it: a file-backed database *on
   its own costs* memory — 125 MB of query-time RSS against 115 MB for
   `:memory:` — and only helps alongside a budget (94 MB with both). DuckDB
   will not evict buffer-pool pages until told a limit, and can only evict
   somewhere durable; neither half works alone. So the file is created
   exactly when `PYCYPHER_DUCKDB_MEMORY_LIMIT` is set. The measured win is
   small regardless, because the source load dominates — this is groundwork
   for item 2, not a result on its own.
   Tests: `tests/test_duckdb_scratch_ownership.py`.
2. ~~**Register file sources via `read_relation` in `add_entity`.**~~
   **Done** — `add_entity(..., streaming=True)`, opt-in and entity-only.
   Two findings worth carrying forward:
   *Normalisation must not use window functions.* The first version
   generated ids and de-duplicated with `row_number() OVER ()`, which forces
   DuckDB to buffer the whole input; under a 256 MB budget the load did not
   merely slow down, it raised `OutOfMemoryException` — defeating the exact
   thing streaming exists to do. The scan is now a straight projection, with
   id assignment and de-duplication done afterwards against the materialised
   (spillable) table, and de-duplication skipped entirely unless duplicates
   are actually present.
   *A streaming source must never be materialised for property lookups.*
   `VectorizedPropertyStore` and `_get_indexed_dataframe` both read the
   entity's source table, so the first `RETURN p.city` pulled the whole file
   back in. Property resolution now pushes the frame's ids into DuckDB and
   joins, bounding cost by the frame rather than the source.
   Tests: `tests/test_streaming_entity.py`. Relationships remain eager.
   Benchmark: `tests/benchmarks/bench_duckdb_streaming_source.py`.
3. ~~**Make `BindingFrame.join` lazy.**~~ **Done.** Inner, left, and cross
   joins compose into DuckDB relations. This changes *when* the join runs,
   not what it produces — `DuckDBBackend.join` was already SQL on the eager
   path, so column selection and row order are unchanged, which is what made
   it safe. Endpoint pushdown in `pattern_matcher` also stopped materialising
   the driving frame: it now passes a `LazyIds` relation reference and
   semi-joins in SQL, where before it read `frame.bindings[prev_var]`.
   The boundary narrowed rather than vanished — see Phase 6 in the design
   doc. Tests: `tests/test_duckdb_joins.py`.
   One consequence worth knowing: grouped-aggregation row order changed on
   DuckDB, because it fell out of row arrival order. That was item 4 below —
   `ORDER BY` was being dropped for aggregated output on *both* backends, so
   the order was never meaningful. Now that item 4 is fixed the ordering is
   real, and those join tests assert it strictly rather than as multisets.
4. ~~**Fix `ORDER BY` after grouped aggregation.**~~ **Done.** The root cause
   was in `apply_projection_modifiers`, not in aggregation: when a sort key
   could not be evaluated against the post-projection frame it was
   re-evaluated against the *pre-projection* frame and then aligned with
   `sort_series.values[:len(temp_df)]`. After a `GROUP BY` that zips six
   pre-aggregation values onto four grouped rows — a plausible-looking,
   arbitrary order rather than an error.

   Two changes. `_projected_sort_column` resolves a sort key to the output
   column that already computes it, which is what Cypher means anyway (in
   `RETURN p.city, count(p) ORDER BY p.city` the key *is* the grouping key),
   and is the only source with the right number of rows. What remains
   genuinely unalignable now raises instead of truncating: the length check
   is `!=`, not `<`, because the damaging direction is a key with *more*
   values than result rows — a `<` guard would have let every real instance
   of this bug through untouched.

   **This is a behaviour change, not only a fix.** `RETURN DISTINCT p.city
   ORDER BY p.name` and `RETURN p.city, count(p) ORDER BY p.name` used to
   return rows in some order; they now raise `ValueError`. Neo4j rejects
   both — a sort key may not reach past a `DISTINCT` or an aggregation — so
   the previous answers were not a weaker guarantee, they were meaningless.
   Any caller relying on them was relying on nothing.

   Worth recording as a caution: my first attempt matched sort keys to
   projection items by rendered text. The renderer emits a `PropertyLookup`
   as its bare property name, so `p.name` and `q.name` render identically
   and `ORDER BY q.name, p.name` silently sorted by `p.name` twice — the
   same class of silent-wrong-answer bug, reintroduced by the fix for it.
   Structural AST equality is the correct comparison. Three differential
   tests caught it. Tests: `tests/test_order_by_after_projection.py`.
5. **`shortest_path_to_binding_frame`** is untouched; DuckDB 1.5's
   `WITH RECURSIVE … USING KEY` is the candidate.
