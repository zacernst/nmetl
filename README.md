# PyCypher and nmetl

A Cypher query engine for tabular data, and an ETL tool built on it.

- **pycypher** parses [openCypher](https://opencypher.org/) queries and executes them against pandas DataFrames, DuckDB tables, or Polars frames. No graph database required.
- **nmetl** turns a YAML pipeline file into a runnable ETL job: it loads sources into a query context, runs a sequence of Cypher queries (streamed through DuckDB out-of-core where possible), and writes the results to CSV, Parquet, JSON, or Neo4j.
- **shared** holds the logging, metrics, and telemetry plumbing both use.

Everything here is MIT licensed.

## Install

```bash
pip install nmetl          # pulls in pycypher and shared
pip install pycypher       # the engine alone
```

Or work from the repository:

```bash
git clone https://github.com/zacernst/nmetl.git
cd nmetl
make setup                 # uv sync (core dev deps) + pre-commit hooks
make test-fast
```

Python 3.14 is required.

## Query a DataFrame in 60 seconds

```python
import pandas as pd
from pycypher import ContextBuilder, Star

people = pd.DataFrame({
    "__ID__": [1, 2, 3],
    "name": ["Alice", "Bob", "Carol"],
    "age": [30, 25, 35],
})
knows = pd.DataFrame({"__SOURCE__": [1, 2], "__TARGET__": [2, 3]})

context = (
    ContextBuilder()
    .add_entity("Person", people)
    .add_relationship("KNOWS", knows, source_col="__SOURCE__", target_col="__TARGET__")
    .build()
)
star = Star(context=context)

print(star.execute_query(
    "MATCH (p:Person)-[:KNOWS]->(q:Person) WHERE p.age > 28 "
    "RETURN p.name AS person, q.name AS knows"
))
#   person knows
# 0  Alice   Bob
```

See [docs/hello_world.rst](docs/hello_world.rst) for a progressive tutorial and [docs/getting_started.rst](docs/getting_started.rst) for the full walkthrough.

## Run a pipeline

```yaml
# pipeline.yaml
version: "1.0"
sources:
  entities:
    - id: people
      uri: file:///data/people.csv
      entity_type: Person
      id_col: person_id
  relationships:
    - id: knows
      uri: file:///data/knows.csv
      relationship_type: KNOWS
      source_col: from_id
      target_col: to_id
      source_entity_type: Person
      target_entity_type: Person
queries:
  - id: friend_counts
    cypher: |
      MATCH (p:Person)-[:KNOWS]->(q:Person)
      RETURN p.name AS name, count(q) AS friends
outputs:
  - query_id: friend_counts
    uri: file:///out/friend_counts.parquet
```

```bash
nmetl validate pipeline.yaml
nmetl run pipeline.yaml
nmetl repl --entity Person=people.csv      # interactive Cypher shell
```

See [packages/nmetl/README.md](packages/nmetl/README.md) and the [ETL tutorial](docs/tutorials/data_etl_pipeline.rst).

## What is in the box

| Area | Highlights |
|---|---|
| Cypher support | MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, UNWIND, CREATE, MERGE, SET, DELETE, FOREACH, UNION, variable-length paths, shortestPath, list/pattern comprehensions, CASE, 130+ scalar functions |
| Execution | Lark Earley parser, Pydantic AST, vectorised BindingFrame evaluation, pluggable pandas / DuckDB / Polars backends, graph-native indexes, cardinality-driven optimisation |
| Out-of-core | A logical-plan translator compiles Cypher to DuckDB SQL so pipelines can exceed RAM; unsupported constructs are reported by name, never silently downgraded |
| Safety | Pre-execution semantic validation, query timeouts, complexity limits, result caching, audit logging, rate limiting |
| Tooling | `nmetl` CLI (run, validate, query, repl, health, metrics, security-check), health server for container probes, Cypher language server (`python -m pycypher.cypher_lsp`) |

## Repository layout

```
nmetl/
├── packages/
│   ├── pycypher/    # parser, AST, evaluators, Star, backends, plan translator, ingestion layer
│   ├── nmetl/       # pipeline config, validation, CLI, REPL, health server, Neo4j sink
│   └── shared/      # logger, metrics, telemetry, exporters
├── tests/           # engine test suite (nmetl's own tests live in packages/nmetl/tests)
├── docs/            # Sphinx documentation, ADRs, design notes
├── examples/        # runnable example scripts
└── scripts/         # quality gates used by CI (lint, import cycles, API surface)
```

Dependency order is `shared` → `pycypher` → `nmetl`; the engine never imports the ETL layer.

## Development

```bash
make check          # lock-check, format, lint, typecheck, import-cycle ratchet, fast tests
make test           # full suite, parallel
make docs           # Sphinx HTML into docs/_build/html
make help           # every target
```

[DEVELOPMENT.md](DEVELOPMENT.md) covers dependency groups, test markers, debugging, Docker, and benchmarks. [CONTRIBUTING.md](CONTRIBUTING.md) has the contribution checklist; architecture decisions are recorded under [docs/adr](docs/adr) and [docs/developer_guide/adr](docs/developer_guide/adr).

## Status

Alpha (`0.0.x`). The public API of `Star`, `ContextBuilder`, `Context`, the exception hierarchy, and `validate_query` is stable; see the [API stability table](packages/pycypher/README.md#api-stability) and [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE.txt](LICENSE.txt).
