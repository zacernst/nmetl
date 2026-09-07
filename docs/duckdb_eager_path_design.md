# DuckDB eager-path design

Status: all eight phases implemented as of 2026-08-25. Phase 8 was first
reported as **not achieved** — correctly, as scoped — and then closed by
adding streaming source registration (see that section). Each phase section
records what landed, what was changed from the original plan, and why.
Implementation report: `duckdb_eager_path_report.md`.

Sibling to `duckdb_full_parity_design.md`. That document covers the
**relation path** — the whole-query compiler in `relation_engine.py` that
turns an eligible `MATCH ... RETURN` (and, since the SET/CREATE/DELETE
slices, some single-table mutations) into native DuckDB relations. This
document covers the **eager path**: the `ClauseExecutor` → `BindingFrame`
pipeline that every non-eligible query falls back to, and which today runs
in pandas regardless of `backend_engine`.

## Goal

When `backend_engine: duckdb` is selected, run the eager path's scans,
filters, renames, concats, index construction, path expansion, and
statistics inside DuckDB instead of pandas — so that a query which is *not*
relation-eligible still gets columnar execution and bounded memory, rather
than silently reverting to full in-RAM pandas materialization.

## Current state (verified 2026-08-23)

| Workload | Location | What runs today |
| --- | --- | --- |
| Entity scan | `scan_operators.py:145` | `_source_to_pandas(source_obj).set_index("__ID__")`, cached in `context._property_lookup_cache`. Never consults `context.backend`. |
| Relationship scan | `scan_operators.py:298` | Same, plus `isin()` endpoint pushdown and two dtype-coercion helpers (`scan_operators.py:52`, `:89`). |
| Filter | `scan_operators.py:494` → `binding_frame.py:934` → `duckdb_backend.py:513` | Mask computed as a numpy array by `BindingExpressionEvaluator`; `DuckDBBackend.filter` can only `.loc[mask]`. |
| Rename | `duckdb_backend.py:569` | `df.rename()` — docstring says "no SQL benefit". |
| Concat | `duckdb_backend.py:577` | `pd.concat`. |
| Index construction | `graph_index.py:84,237,315,398` | Python `for i in range(len(...))` loops building dicts; `np.argsort` over `dtype=object` arrays. |
| Path expansion | `path_expander.py:93` | Per-hop `frontier.merge(edge_df)` + `duplicated()` dedup; caps at 1M frontier / 5M total rows. |
| Statistics | `cardinality_estimator.py:217` | `.to_pandas()` on the whole source, `.sample()`, `nunique()`, `np.histogram`. |

Three findings shape the plan:

1. **`DuckDBBackend.scan_entity` (`duckdb_backend.py:496`) is dead code**
   outside the health probe at `backend_engine.py:430`. `EntityScan.scan`
   reads `entity_table.source_obj` directly and never dispatches to the
   backend, so the protocol's "Scan" category is unreachable from the
   execution path as written.

2. **`Context.index_manager` eagerly builds vectorized stores.**
   `relational_models.py:499` calls `eager_build_vectorized_stores()` the
   first time anything touches `context.index_manager`, materializing every
   entity table into object-dtype numpy arrays. This is a hard RAM wall that
   fires regardless of backend and regardless of whether any index is
   subsequently used.

3. **`BindingFrame.bindings` is `pd.DataFrame`** (`binding_frame.py:289`,
   alias at `cypher_types.py:40`), with **156 `.bindings` accesses across 12
   modules**. A wholesale conversion of the frame type is not viable; the
   plan below uses a materialize-on-access property instead.

## Phase 1 — Enabling substrate

Blocks phases 3–7. Nothing else is worth doing until frames can stay inside
DuckDB across an operator boundary.

### 1a. A general table registry, not just streaming sources — done

`register_streaming_source` (`relation_engine.py:89`) already does the right
thing — `lazy.relation.create(table_name)` is a streaming `CREATE TABLE AS
SELECT` that never loads the file into pandas — but it is called from
exactly one place (`cli/pipeline.py:575`), only for entities, and only
inside `_try_streaming_run`.

Generalize it:

- Move it to a `TableRegistry` owned by `DuckDBBackend`, covering **both**
  entities and relationships.
- Populate from `ContextBuilder.build()` whenever `backend.name ==
  "duckdb"`, so it is live for every DuckDB run rather than only the
  streaming CLI path.
- File-backed sources go through `DataSource.read_relation`
  (`ingestion/data_sources.py:627`), which is already streaming. In-memory
  `pd.DataFrame` / `pa.Table` sources go through `con.from_df` /
  `con.from_arrow` plus a one-time CTAS.
- **Normalize `__ID__` to a single declared DuckDB type at CTAS time.**
  This is the root fix for the coercion helpers at `scan_operators.py:52-120`
  — they exist because IDs change dtype every time data crosses the pandas
  boundary. Removing the boundary removes the problem.
- Keep `context._streaming_sources` (`relational_models.py:333`) as an alias
  onto the new registry so `_base_relation`, `_streaming_id_col`, and the
  rest of the relation-engine code do not fork.

**Landed** as `pycypher/backends/table_registry.py` (`TableRegistry`,
`RegisteredTable`, `physical_table_name`, `register_context_tables`), reached
as `backend.tables`. `register_streaming_source` now delegates to it, and the
three mutation executors take their table name from `physical_table_name`
instead of rebuilding the literal. `_base_relation` / `_rel_base_relation`
prefer a registered table over re-converting `source_obj`, which is the
immediate payoff: that conversion previously ran on *every* query.
`ContextBuilder.build()` grows a `register_tables` parameter defaulting to
"on for DuckDB". Tests: `tests/test_table_registry.py`.

Four things worth recording from the implementation:

- **The registry keys on `(kind, label)`, not label alone.** Cypher permits a
  node label and a relationship type with the same spelling, and the original
  single-namespace `_streaming_sources` dict could not represent both.
  Relationship tables use a `_rel_source_` prefix; the entity prefix is
  unchanged so a pre-existing scratch database still reads the same.
- **`InstrumentedBackend` had to forward `tables` explicitly**
  (`backend_engine.py`), exactly as it already forwarded `connection`.
  Without that, `nmetl run -v` — which sets `instrument=True` — would wrap the
  backend and silently hide the registry from every consumer.
- **`DuckDBLazyFrame` caches its first `to_pandas()` forever.** So
  `RegisteredTable.relation` sees post-mutation rows but `RegisteredTable
  .frame` does not. This is pre-existing behaviour, not introduced here, and
  it is invisible today only because `_base_relation` reads `.relation`. It
  is a live trap for Phase 3, which will hand these frames to operators that
  do materialise. Documented on `RegisteredTable` and pinned by a test.
- **`register_context_tables` never clobbers a streaming registration.** A
  file-backed source is strictly better than the in-memory one it would
  otherwise build, so an existing entry wins.

### 1b. Let `BindingFrame` carry a relation — done

Follow the pattern `DuckDBLazyFrame` already prototypes
(`duckdb_backend.py:229`): allow `BindingFrame` to be constructed with
`relation=` instead of `bindings=`, and make `.bindings` a property that
materializes and caches on first access.

All 156 existing `.bindings` sites then keep working unchanged — they simply
force materialization. Each subsequent phase teaches one more operator to
check "is this frame still lazy?" and stay lazy when it is.

Invariants that must survive materialization: column order, `type_registry`,
and row order (see Cross-cutting concerns).

**Landed** in `binding_frame.py`: `BindingFrame(relation=...)` as an
alternative to `bindings=`, with `.bindings` materializing on first access,
plus `.is_lazy` and `.relation` for operators that opt in. All three
invariants are pinned by tests. Tests: `tests/test_binding_frame_lazy.py`.

Three implementation decisions worth recording:

- **`BindingFrame` is no longer a `@dataclass`.** A property cannot coexist
  with a same-named dataclass field: `dataclasses` resolves the field's
  default via `getattr(cls, name)`, finds the property object, treats it as a
  default, and then rejects the following fields for having none. An explicit
  `__init__` was the honest fix. Verified safe first: all 36 construction
  sites are keyword-based, and the generated `__eq__` was already unusable
  (comparing field tuples containing DataFrames raises "truth value is
  ambiguous"), so nothing depended on it. `__repr__` is now a summary that
  does not materialize.
- **Materialization is one-way.** Once `.bindings` is touched, the relation
  is released and `.relation` returns `None`. Serving a relation alongside an
  already-materialized DataFrame invites reading a relation that a later
  pandas-side edit has made stale — the same class of bug as the
  `DuckDBLazyFrame` caching trap found in 1a. Operators must consume
  laziness before anything forces the frame.
- **`__len__` and `var_names` had to change too.** Both read `self.bindings`,
  so a lazy frame would have materialized on the first row count or variable
  listing — which happens constantly. They now answer from `COUNT(*)` and the
  relation schema respectively. Without this the carrier would technically
  work while never actually staying lazy, and every "materializes zero times"
  assertion in later phases would be vacuous.

The constructor rejects receiving neither or both of `bindings=`/`relation=`
rather than silently preferring one.

### 1c. A materialization tripwire — done

Add a counter plus DEBUG log on every forced materialization, exposed on the
context. This is what makes the rest of the plan measurable: each phase gets
a test asserting "query X materializes at most N times", so progress is
observed rather than assumed.

**Landed** as `count_materialisations()` / `MaterialisationLog` /
`MaterialisationEvent` in `backends/duckdb_backend.py`, recording at
`DuckDBLazyFrame._materialise` — the single choke point through which a lazy
relation becomes pandas, so one hook covers every path. Tests:
`tests/test_materialisation_tripwire.py`.

- Scoping uses a `ContextVar`, matching the `_scope_var` pattern already used
  for per-query state, so concurrent queries do not race on the count.
- **Nested scopes propagate outward.** An inner scope opened by a helper
  records into its own log *and* every enclosing one, so a helper can never
  hide work from an outer assertion.
- `capture_stack=True` resolves a `file:line` origin for each event, skipping
  `duckdb_backend.py` frames so the origin is the operator that forced
  materialization rather than the plumbing that performed it. Off by default
  — walking the stack per materialization is far too expensive for
  production paths.
- Schema access (`.columns`) and `len()` are verified *not* to count: they are
  answered from the relation without executing it. That is what makes a
  "zero materializations" assertion meaningful rather than vacuous.

## Phase 2 — Statistics — done

Not blocked by Phase 1. Lowest risk in the plan, and `query_planner.py:575`
builds `TableStatistics` for every registered table eagerly, so the win
lands on every query.

Give `TableStatistics` a DuckDB implementation, selected when `source_obj`
is a registered table:

```sql
SELECT count(*),
       approx_count_distinct("c"),
       count(*) FILTER (WHERE "c" IS NULL),
       min("c"), max("c")
FROM tbl
```

plus `histogram("c", <edges>)` for the bins.

Two deliberate choices:

- **Use explicit-edge `histogram()`, not `approx_quantile`.** Equi-depth
  quantiles would be more accurate, but would require rewriting
  `_histogram_range_selectivity` (`cardinality_estimator.py:140`). Keeping
  equi-width edges leaves that function untouched — a much smaller blast
  radius for the first cut.
- Sampling (`STATS_SAMPLE_SIZE`) becomes `USING SAMPLE reservoir(N ROWS)`.
  **This doc's determinism concern was wrong.** DuckDB supports
  `REPEATABLE (seed)`, so the SQL path is exactly as reproducible run to run
  as `random_state=42` is. The two paths still draw *different* samples from
  the same table, so cross-path assertions compare distributions rather than
  values — but no test needed loosening for non-determinism.

**Landed** in `cardinality_estimator.py`: `TableStatistics(source,
relation=...)` computes statistics in SQL when a relation is available,
falling back to the pandas path on any failure or unknown column.
`query_planner.QueryPlanAnalyzer._build_table_stats` hands each table its
registered relation. Tests: `tests/test_duckdb_statistics.py`.

- **NDV, null fraction, and extrema came out identical to pandas** on the
  test data, and equality selectivity matches exactly — this really is a
  drop-in.
- **The histogram needed a bin-convention fix.** DuckDB's
  `histogram(col, edges)` returns a map keyed by each bin's *upper* bound,
  with bins half-open on the left — `(e[i-1], e[i]]` — where numpy's are
  half-open on the right. Exact equality with `np.histogram` is therefore the
  *wrong* target: the two disagree for values landing exactly on an interior
  edge, and both are correct histograms over the same edges. What matters is
  the consumer contract, which is what the tests assert — `len(counts) ==
  len(edges) - 1`, and every non-null row counted exactly once. DuckDB's
  first key counts values `<= min` (i.e. exactly the minimum), which belongs
  in bin 0, so it is folded into the second key.
- `_histogram_range_selectivity` is untouched, as intended.
- Statistics run with **zero materialisations**, asserted via the Phase 1c
  tripwire.

## Phase 3 — Scans — done

**3a. `EntityScan.scan`.** When the backend is DuckDB and the label is in the
registry, return a lazy `BindingFrame` over
`SELECT "__ID__" AS "<var>" FROM tbl`. The `property_filters` pushdown
(`scan_operators.py:184-234`) becomes a `WHERE` clause, which supersedes
`PropertyValueIndex` entirely — no index to build, no frozenset
intersection.

**3b. `RelationshipScan.scan`.** `SELECT "__ID__" AS r, "__SOURCE__" AS
_src_r, "__TARGET__" AS _tgt_r FROM rel_tbl`. Endpoint pushdown becomes a
semi-join against the caller's already-lazy ID relation rather than a
materialized `pd.Series`.

That requires a signature change: `source_ids` / `target_ids` are typed
`FrameSeries` today and must accept either a Series (pandas path) or a
relation (lazy path). Audit callers before committing to the shape.

**Both shadow guards must be preserved**: `scan_operators.py:185`
(`entity_type not in context._shadow`) and `:360-363` (`rel_type not in
_shadow_rels`). These are the contract between this work and the mutation
DML work in `duckdb_full_parity_design.md` — a lazy scan must never read
past a pandas shadow overlay.

Payoff beyond speed: `_property_lookup_cache` stops holding a `set_index()`
copy of every table, and `_coerce_pushdown_ids` / `_coerce_pushdown_series`
become dead on the DuckDB path (they stay for pandas).

**Landed** in `scan_operators.py` as `EntityScan._scan_duckdb` /
`RelationshipScan._scan_duckdb`, each returning `None` to fall back to the
unchanged pandas scan. Tests: `tests/test_duckdb_scans.py`, plus a reusable
pandas-vs-DuckDB differential helper in `tests/helpers_differential.py`.

- **Entity pushdown** compiles inline `{prop: value}` predicates to a `WHERE`,
  superseding `PropertyValueIndex` on this path. `None` values deliberately do
  *not* push: the pandas pushdown goes through `PropertyValueIndex`, whose
  build skips nulls, so `prop = NULL` would be a silent semantic change.
- **Endpoint pushdown became a semi-join** (`relation.join(ids, cond,
  how="semi")`), which preserves the left relation's columns and chains for
  source+target. The relationship table — the largest table in a graph
  workload — never becomes pandas; only the small id set crosses the boundary.
- **Uncoercible pushdown ids fall back rather than pushing down.** The
  dangerous failure mode here is a semi-join that silently returns zero rows
  because one side is text and the other integral. `_pushdown_frame` returns
  `None` on any unclean coercion, and a test pins that.
- **A real divergence, kept deliberately**: DuckDB scans materialise IDs as
  `int64` where the pandas path forces `object` (`to_numpy(dtype=object)`).
  The full suite and the differential queries pass either way, and `int64` is
  both cheaper and the direction this effort wants, so it was kept rather
  than cast back. `_coerce_ids` already bridges the two. Worth knowing when
  reading a mixed-path frame.
- **The relationship pushdown ids still come from pandas.** `pattern_matcher`
  reads `frame.bindings[prev_var]`, which materialises the driving frame. So
  Phase 3 keeps the *tables* in DuckDB but not yet the *frames* flowing
  between operators — that needs Phases 4 and 5.

## Phase 4 — Renames and concats — done

Cheap, but worthless before Phase 1.

- **rename** → `relation.project('"a" AS "b", …')`, preserving column order.
  Watch for rename-into-an-existing-name, which pandas tolerates (producing
  duplicate columns) and SQL does not.
- **concat** → `UNION ALL BY NAME`, which matches pandas'
  column-union-with-nulls semantics. The divergence is **dtype
  reconciliation**: pandas upcasts `int` + `str` to `object`; DuckDB raises.
  Needs an explicit reconciliation step (cast to a common type, else fall
  back to `pd.concat`). All four production callers
  (`mutation_engine.py:386,441`, `pattern_matcher.py:246,678`) pass
  `ignore_index=True`, so the index-preserving case needs no SQL support —
  but assert that rather than assuming it.
- **distinct** is already SQL (`duckdb_backend.py:589`). It is now lazy too.

**Landed** in `duckdb_backend.py` (`rename`, `concat`, `distinct` all
lazy-aware) plus the callers: `BindingFrame.rename` keeps a lazy frame lazy,
and `pattern_matcher`'s two concat sites go through a new
`binding_frame.concat_binding_frames`. Tests: `tests/test_duckdb_transforms.py`.

**The dtype-reconciliation worry turned out to understate the problem.**
`relation.union()` does not error on mismatched input — it returns a wrong
answer, in two distinct ways, both found by probing the API rather than by
reading docs:

- **Differing column sets union *positionally*.** `[{p}] ∪ [{q}]` yields a
  single column named `p` holding both — silently mislabelled data, where
  `pd.concat` gives two columns with nulls.
- **Differing types coerce silently.** `[1, 2] ∪ ["x"]` becomes
  `["1", "2", "x"]`; pandas keeps `[1, 2, "x"]` as object dtype.

So the lazy path requires every frame to be lazy *and* to share identical
column names in identical order with identical types. Anything else falls
back to `pd.concat`. Both divergences have a test pinning the pandas
behaviour, because a future "optimisation" that relaxes either guard would
corrupt data rather than fail loudly.

`BindingFrame.__init__` now also routes a `DuckDBLazyFrame` passed as
`bindings=` to `relation=`. Backend operations that newly stay lazy return
one, and callers pass through whatever they got; doing the detection in the
constructor protects all 36 construction sites at once instead of requiring
each to be found.

## Phase 5 — Filters — done

The mask is computed *outside* the backend, so `DuckDBBackend.filter` cannot
be fixed in isolation. The change belongs in `BindingFilter.apply`
(`scan_operators.py:494`):

1. If the frame is lazy and the backend is DuckDB, try
   `relation_sql.compile_expression(predicate, resolve)` →
   `relation.filter(sql)`.
2. **Split on `AND` first.** Compile each conjunct independently; push down
   what compiles and run the residual through the existing pandas evaluator.
   A partially-pushed filter is still a large win and is far more achievable
   than all-or-nothing.
3. Anything that does not compile (unbridged UDFs, `EXISTS` subqueries,
   pattern comprehensions, list operations) materializes and uses today's
   path unchanged.

**Landed** in `scan_operators.py` as `BindingFilter._push_to_sql`. Tests:
`tests/test_duckdb_filters.py`.

**The prerequisite was avoided, not built.** Projecting property columns into
the scan would have put them in `BindingFrame.var_names` — which the engine
reads as *the list of bound Cypher variables*, not merely as a column list.
Property columns leaking into it would corrupt variable resolution
everywhere. Instead each referenced `var.prop` is resolved by LEFT-joining
the entity table on `<var> = __ID__`, and the result is projected back to
exactly the frame's original columns. Each side relation is projected down to
its id and the wanted property *before* the join, so an entity column can
never collide with a binding variable's name. LEFT (not inner) so an id with
no matching row yields NULL, which is what `get_property` produces. No
whole-query analysis pass was needed; a small local AST walk collects the
`(var, prop)` pairs.

**A silent-fallback bug worth remembering.** DuckDB rejects a join between
two relations descending from the same table unless their aliases are made
distinct (`set_alias`). The broad `except` around relation building caught
that, so every push fell back to pandas: results stayed correct and the
entire suite stayed green while the feature did nothing at all. Correctness
tests cannot detect this. That is why `TestPushdownHappens` asserts
`is_lazy is True` rather than only asserting rows — the fallback is designed
to be invisible, so *something* has to check that it was not taken.

**Residual conjuncts are applied as successive pandas filters** rather than
rebuilt into an `And` node: keeping rows where each conjunct is TRUE is the
same as keeping rows where their conjunction is TRUE, under Kleene logic as
under `fillna(False)`, and it avoids fabricating AST nodes. Splitting only
happens on a lazy frame, so the pandas path evaluates its predicate in one
shot exactly as before.

**This phase has a prerequisite that does not exist yet.** Predicates
reference `n.prop` where `prop` is not a column of the binding frame — today
`get_property` (`binding_frame.py:429`) resolves it by lookup. In SQL it must
be either a join or an already-projected column. There is no whole-query
property-requirement analysis in the codebase: `projection_planner.py` does
not do it, and only `aggregation_planner.py:229` batches per-clause. So
Phase 5 needs a **new AST pass collecting `(var, prop)` accesses per
query**, so Phase 3's scans can project the needed columns up front. Scope
it as its own deliverable — it also directly improves `get_property` on the
pandas path.

**Three-valued logic — the risk did not materialise.** The worry was that
`NOT (x = 5)` with `x IS NULL` would differ between pandas and SQL, and that
the pandas path might be quietly wrong. It is not: `boolean_evaluator.py`
implements proper Kleene logic (`np.where(null, None, ~s_bool)`, so
`NOT null -> null`), exactly matching SQL, and AND/OR are Kleene too. `OR`
and `NOT` are therefore safe to push. `tests/test_duckdb_filters.py` pins
this with a null-bearing row across `NOT`, `OR`, `IS NULL`, and a missing
property.

## Phase 6 — Index construction — done

A deletion, not a port. Once Phases 1 and 3 land, all four index types in
`graph_index.py` are redundant on the DuckDB path:

| Index | Replaced by |
| --- | --- |
| `AdjacencyIndex` (`:84`) | Hash join on `__SOURCE__` / `__TARGET__` |
| `PropertyValueIndex` (`:237`) | `WHERE col = ?` |
| `EntityLabelIndex` (`:315`) | The table itself |
| `VectorizedPropertyStore` (`:398`) | Projected columns / join in `get_property` |

~~Make `GraphIndexManager` return `None` for all four when the backend is
DuckDB~~ — **this prescription was wrong and was not implemented.** Those
indexes still serve the *pandas fallback*, which is taken whenever the DuckDB
route declines (an unregistered label, a shadow overlay, an uncoercible
pushdown). Returning `None` there would slow the fallback down with no
DuckDB replacement to compensate — the same trap this doc already identified
for the on-demand store build.

The correct end state is that the indexes are simply **never reached** on the
DuckDB path, which is what Phase 3 and Phase 5 already accomplished. Measured
on a query set of four (`tests/test_duckdb_indexes.py`):

| | pandas | DuckDB |
| --- | --- | --- |
| `AdjacencyIndex` | built | never built |
| `PropertyValueIndex` | built | never built |
| `VectorizedPropertyStore` | built for every registered type | none, except after a join |

Three of the four queries build **zero** index structures on DuckDB.

Getting there needed three more fixes, all of which were forcing
materialisation for no reason:

- **`get_property` / `get_properties_batch` now resolve through the
  relation** when the frame is still lazy: the property is LEFT-joined on and
  the relation materialised once, yielding the bindings and the property
  columns from a single execution. That removes the `VectorizedPropertyStore`
  entirely, which had been copying the whole entity table into object-dtype
  numpy arrays to answer lookups for the handful of ids a frame holds.
- **`hasattr(result, "bindings")` in `clause_executor` materialised the
  frame.** `hasattr` calls the getter, and `bindings` is a
  materialise-on-access property — so an existence check was executing the
  relation. Replaced with `isinstance`. This is a hazard the Phase 1b design
  created and worth watching for wherever a property does real work.
- **`frame_size()` used `len(frame.bindings)`** to build a DEBUG log string.
  Log arguments are evaluated eagerly, so every clause boundary materialised
  the frame purely to format a message that is usually discarded. Now
  `len(frame)`, which answers from `COUNT(*)`.

**Known boundary — since narrowed.** `BindingFrame.join` originally read
`.bindings`, so any multi-pattern query materialised before `RETURN`. Joins
are now lazy (report open item 3), and the boundary moved: a join with
single-variable projection, multiple properties of one variable, or
`count(*)` builds **nothing**. What remains is that resolving a property
materialises the frame — it must, the caller wants values — so a *second*
variable projected afterwards finds a materialised frame and falls back to
the property store. Closing that needs the projection planner to batch
property resolution across variables. Both halves have tests.

**Land one line of this immediately after Phase 1a — done.**
`relational_models.py:499` eagerly calls `eager_build_vectorized_stores()`
on first `context.index_manager` access. Under DuckDB that must not happen
at all. It is likely the single largest memory win in this plan and it is
nearly free.

The prebuild is now skipped when `backend_name == "duckdb"`. Stores are still
built on demand by `get_vectorized_store()`, so results are unchanged and only
*untouched* tables are spared — a query that reads every table pays the same
as before, just later. Eliminating the on-demand build too is the rest of
Phase 6, and it has to wait for Phase 3: with no DuckDB scan to fall back to,
`get_property` would drop to the pandas hash-map path and get slower with
nothing to show for it. Tests: `tests/test_index_manager_duckdb.py`, including
a differential check that property resolution is identical across backends.

On `CREATE INDEX`: **do not**, in the first cut. DuckDB's ART indexes are
memory-resident and only help point lookups, so adding them reintroduces
exactly the unbounded-RAM problem this work exists to remove. Measure first;
add selectively later if profiling justifies it.

## Phase 7 — Path expansion — done

Largest single piece. Replace the hop loop (`path_expander.py:176-262`) with
a recursive CTE:

```sql
WITH RECURSIVE walk(start_id, tip, hop) AS (
    SELECT s.v, s.v, 0 FROM seed s
  UNION            -- set semantics, matching today's duplicated() dedup
    SELECT w.start_id, e."__TARGET__", w.hop + 1
    FROM walk w JOIN edges e ON e."__SOURCE__" = w.tip
    WHERE w.hop < $max_hops
)
SELECT * FROM walk WHERE hop BETWEEN $min_hops AND $max_hops
```

Four changes that need explicit decisions:

1. **The loop must carry only `(start_id, tip, hop)`.** Today's frontier
   carries every column of `start_frame` (`path_expander.py:169`) while
   deduplicating on just `(start_var, _vl_tip)`. `UNION` deduplicates whole
   rows, so the carried columns would defeat it. Walk narrow and re-join to
   the seed frame at the end. This is a genuine improvement — far less data
   in the loop — but it changes row multiplicity, so validate against
   existing tests rather than assuming equivalence.
2. **Safety caps become weaker.** `_MAX_FRONTIER_ROWS` (1M) and
   `_MAX_BFS_TOTAL_ROWS` (5M) are per-hop checks with no equivalent hook in
   a recursive CTE. The real guard becomes DuckDB's `memory_limit` plus
   `max_temp_directory_size`, with a post-hoc row check. `SecurityError`
   will fire later and less precisely. This is a deliberate behaviour change
   and should be documented, not papered over.
3. **`row_limit` pushdown degrades.** Today BFS stops accumulating early
   (`path_expander.py:239-253`). `LIMIT` on the outer select will not
   terminate the recursion. Accept it, or derive a hop-count bound from the
   limit.
4. **`context.check_timeout()` per hop disappears.** Replace with
   `con.interrupt()` from a watchdog thread, or accept coarser granularity.

`shortest_path_to_binding_frame` (`path_expander.py:299`): DuckDB 1.5
(pinned `duckdb>=1.5.2,<2.0.0` in `packages/pycypher/pyproject.toml:19`)
supports `WITH RECURSIVE … USING KEY`; confirm its exact semantics against
the DuckDB docs before relying on it, otherwise `min(hop) GROUP BY (start,
tip)` works.

**Landed** in `path_expander.py` as `PathExpander._expand_duckdb`. Tests:
`tests/test_duckdb_path_expansion.py`.

The seed is supplied to the SQL through `relation.query("_pyc_seed", ...)`
rather than a created view, so nothing has to be dropped afterwards; the edge
table is already a real table in the database and is referenced by name.

Of the four predicted changes, two were resolved by **declining** rather than
by accepting a divergence — the SQL route returns `None` and the pandas BFS
runs unchanged:

- **`row_limit` declines.** The pandas path fills hop 1, then hop 2, and
  trims at the boundary, so which rows survive depends on hop order *and* on
  frontier order within a hop. `LIMIT` over the CTE keeps a different subset.
  The pushdown is not worth a different answer.
- **A seed with duplicate `start_var` values declines.** The pandas frontier
  deduplicates on `(start_var, tip)` while carrying the seed's other columns,
  so a repeated start value silently drops seed rows. Joining the narrow walk
  back to the seed keeps them — arguably more correct, but different.
  Declining avoids trading one behaviour for another without being asked.

The other two stand as predicted: the safety caps are weaker (there is no
per-hop hook in a recursive CTE; DuckDB's `memory_limit` and spill settings
are the real guard now), and `check_timeout()` no longer fires per hop.

**Undirected traversal was not added.** The doc suggested the recursive form
gives it "for free", but the pandas path does not support it either, so there
is no reference behaviour to match — adding it here would make the two
backends differ. It stays a shared gap rather than becoming a DuckDB-only
feature.

`shortest_path_to_binding_frame` is untouched; `USING KEY` was not needed for
this slice and is left for whoever takes shortest-path on.

## Phase 8 — Bounded-RSS — achieved, after adding streaming registration

Originally reported as **not achieved**: with `ContextBuilder.add_entity`
reading the whole source into Arrow before any query ran, no amount of
columnar execution downstream could lower peak RSS, and it measurably rose.
That analysis was right, and it identified the missing piece precisely — so
the piece was then built (`pycypher/ingestion/streaming_entity.py`).

With `add_entity(..., streaming=True)`, on 400k rows x 23 columns from
parquet (`tests/benchmarks/bench_duckdb_streaming_source.py`):

| mode | build | query | peak |
| --- | --- | --- | --- |
| eager, pandas backend | 156 MB | 513 MB | 794 MB |
| eager, duckdb backend | 245 MB | 549 MB | 920 MB |
| streaming | 178 MB | 78 MB | **382 MB** |
| streaming + 128 MB budget | 124 MB | 69 MB | **319 MB** |

Peak falls ~2.5x against the eager pandas path and ~2.9x against eager
DuckDB, and query-time RSS falls from ~510 MB to under 80 MB — the source now
lives in DuckDB, so a `MATCH ... WHERE ... RETURN count` never builds a frame
over the full table. Phases 3-7 are what make that possible; this phase
removed the eager read that was masking them.

The original analysis is preserved below, because it is what made the fix
findable.

`ContextBuilder.add_entity` calls `DataSource.read()`, which materialises the
entire source into an Arrow table before any query runs.
`register_context_tables` then copies it again into DuckDB — and
`ContextBuilder.build` creates an `:memory:` database, so that second copy is
also RAM. Phases 3-7 make *execution* columnar; they cannot undo an eager
source load that happens before they are reached.

Measured (200k rows x 22 columns; `tests/benchmarks/bench_duckdb_eager_path_memory.py`):

| mode | build delta | query delta | peak |
| --- | --- | --- | --- |
| `pandas` | 369 MB | 40 MB | 534 MB |
| `duckdb-noreg` | 370 MB | 41 MB | 536 MB |
| `duckdb-mem` | 369 MB | 103 MB | 598 MB |
| `duckdb-file` | 370 MB | 99 MB | 595 MB |

So peak RSS goes **up**, not down, for in-memory sources. Three things follow:

- **`duckdb-noreg` measures identical to `pandas`.** The fallback really is
  unchanged and `register_tables=False` is a working kill switch — that is the
  property `tests/test_duckdb_kill_switch.py` pins, and it is what makes the
  rest safe to ship.
- **A file-backed scratch database is now wired in, but only helps paired
  with a memory budget.** An earlier reading of these numbers credited the
  file; isolating the variables showed otherwise. Query-time RSS: `:memory:`
  115 MB, file-backed **125 MB** (worse), `:memory:` + `memory_limit` 109 MB,
  file-backed + `memory_limit` **94 MB**. DuckDB will not evict buffer-pool
  pages until given a budget, and it can only evict to somewhere durable — so
  neither half works alone. `ContextBuilder.build` therefore creates the
  scratch file exactly when `PYCYPHER_DUCKDB_MEMORY_LIMIT` is set. Even then
  the win is small here, because the pandas/Arrow source load dominates
  everything.
- **The real fix is not in this plan.** For the source never to enter pandas,
  `add_entity` would have to register file-backed sources through
  `DataSource.read_relation` — a streaming scan — instead of `read()`. Both
  that and the file-backed database already exist for the relation engine
  (`register_streaming_source`, `create_scratch_database_path`) but are not
  wired into `ContextBuilder`. That is the honest prerequisite for a
  bounded-RSS acceptance test, and it belongs with the storage-model work in
  `duckdb_full_parity_design.md`, not here.

What this plan *did* deliver against the memory goal is structural rather
than headline: the object-dtype index copies are gone (Phase 6, measured),
per-hop pandas frontiers are gone (Phase 7), and whole-table `set_index`
caches are gone for registered labels (Phase 3). Those reduce allocation
churn and are prerequisites for bounded RSS — they are not sufficient for it
on their own.

## Cross-cutting concerns

- **Row order.** pandas preserves insertion order everywhere; DuckDB does
  too *unless* `preserve_insertion_order=false` (`duckdb_backend.py:33`).
  Any existing test asserting result order without an `ORDER BY` becomes
  dependent on that setting. Leave `preserve_insertion_order` at its default
  and audit tests for implicit order assumptions as part of Phase 3.
- **Empty frames.** `path_expander.py:274` constructs a typed empty pandas
  frame. Lazy equivalents must carry a schema with zero rows, or downstream
  `columns` access breaks.
- **Mutation interaction.** `context._shadow` is pandas and stays pandas
  until the DML work in `duckdb_full_parity_design.md` lands. The shadow
  guards in `scan_operators.py` are load-bearing for both efforts.
- **Config.** `relation_engine` gates the whole-query compiler. This work
  needs its own kill switch (e.g. `duckdb_native_ops`, defaulting on for
  `backend_engine: duckdb`), because every phase has a pandas fallback and
  regressions will need to be bisected by flipping one flag.

## Testing

1. **Differential suite.** Parametrize the existing tests over `pandas` /
   `duckdb` and assert frame equality after canonical sorting. This is the
   primary safety net and should be built during Phase 1, not later.
2. **Materialization-count assertions** from the Phase 1c tripwire — per
   phase, per query shape.
3. **Bounded-RSS acceptance test** — already scoped as Phase 5 in
   `duckdb_full_parity_design.md`; it is the only test that validates the
   actual goal.
4. **Targeted null-semantics tests** written before Phase 5 flips filters to
   SQL.

## Sequencing

```
2 (statistics) ──────────────────────── independent, ship first
1a (registry) ──┬── 6-partial (kill eager_build_vectorized_stores)  ← near-free
                ├── 3 (scans) ──┬── 5 (filters) ── 6 (rest of indexes)
1b (lazy frame)─┤               └── needs: property-requirement AST pass
1c (tripwire) ──┴── 4 (rename/concat)
                └── 7 (path expansion) ← independent of 5; can take a materialized seed
```

| Phase | Size | Risk | Notes |
| --- | --- | --- | --- |
| 2 Statistics | S | Low | Only test determinism changes |
| 1 Substrate | M | Medium | Row-order and dtype invariants are the risk |
| 6-partial | XS | Low | One line, large memory win |
| 3 Scans | M | Medium | Caller signature change for pushdown |
| 4 Rename/concat | S | Low | Dtype reconciliation is the only wrinkle |
| 5 Filters | L | High | Needs a new AST pass; three-valued-logic semantics |
| 6 Indexes | S | Low | Mostly deletion once 3 and 5 land |
| 7 Path expansion | L | High | Weakened safety caps, changed multiplicity |

**Recommended ordering.** Ship Phase 2 and the one-line Phase 6-partial
first — they are days of work, independently valuable, and give real numbers
on how much of the memory problem is statistics and index-building versus
the frames themselves. That measurement should decide whether Phase 5 or
Phase 7 goes first.

## Non-goals for this plan

- Mutations (`SET`/`CREATE`/`DELETE`/`MERGE`) on the eager path — covered by
  `duckdb_full_parity_design.md` Phase 2, which targets the relation path's
  registered tables. The two efforts meet at the `_shadow` guards.
- Expanding `is_relation_eligible()`. Widening the relation path reduces how
  often the eager path is reached, but is orthogonal to making the eager
  path itself columnar.
- The `polars` and `spark` backends. Their `scan_entity` / `concat`
  implementations have the same shape as DuckDB's and would benefit from the
  Phase 1b lazy-frame carrier, but nothing here is designed for them.
