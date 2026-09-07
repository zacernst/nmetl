# nmetl

ETL pipelines and a command-line tool built on the [pycypher](../pycypher/README.md)
query engine.

`pycypher` parses and executes Cypher against tabular data. `nmetl` is the
layer above it that turns a YAML file into a runnable pipeline: it owns the
pipeline configuration models, config validation, the `nmetl` CLI, the
interactive REPL, the health server, and the Neo4j sink. The dependency runs
one way only: `nmetl` imports `pycypher`; `pycypher` never imports `nmetl`.

## Installation

Inside the monorepo the package is a uv workspace member and is installed by
`uv sync`. Standalone:

```bash
pip install nmetl            # pulls in pycypher
pip install "nmetl[neo4j]"   # adds the Neo4j driver for the Neo4j sink
```

## Quick start

A pipeline is a YAML file that declares data sources, the entities and
relationships they populate, a list of Cypher queries, and where each
query's result goes:

```yaml
version: "1.0"
project:
  name: people_demo

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
nmetl validate pipeline.yaml   # structural + semantic checks, no data touched
nmetl run pipeline.yaml        # execute every query and write every output
nmetl run pipeline.yaml --dry-run
nmetl list-queries pipeline.yaml
```

Queries can also be kept in separate files and referenced with `source:`
(relative to the config file). Values may contain `${ENV_VAR}` placeholders.

### Other commands

```bash
nmetl query pipeline.yaml "MATCH (p:Person) RETURN count(p)"   # ad-hoc query over a config
nmetl repl --entity Person=people.csv                          # interactive Cypher shell
nmetl schema pipeline.yaml                                     # entity / relationship summary
nmetl functions                                                # list scalar functions
nmetl security-check pipeline.yaml                             # URI / credential hygiene
nmetl health                                                   # exit 0/1/2 for probes
nmetl health-server --bind 0.0.0.0                             # HTTP health + metrics
nmetl metrics --diagnostic
nmetl config --show-effective                                  # resolved PYCYPHER_* settings
```

Run `nmetl --help` or `nmetl <command> --help` for the full option list.

## Out-of-core execution

With the DuckDB backend selected and the relation engine enabled (the
default for `nmetl run` on DuckDB), every eligible query is translated to
SQL and streamed through DuckDB without materialising pandas frames, so a
pipeline can exceed available RAM. Set `PYCYPHER_DUCKDB_MEMORY_LIMIT` to
bound DuckDB's working set. Queries the translator cannot express are
reported per query under the config's `on_error` policy; there is no silent
fallback to the in-memory path.

## Python API

```python
from nmetl import load_pipeline_config, validate_config

config = load_pipeline_config("pipeline.yaml")
result = validate_config(config)
if not result.is_valid:
    for error in result.errors:
        print(error.category, error.message)
```

| Module | What it holds |
|---|---|
| `nmetl.config` | Pydantic models for the pipeline YAML (`PipelineConfig`, source/query/output records) and `load_pipeline_config` |
| `nmetl.validation` | `validate_config` / `validate_config_dict` returning a categorised `ValidationResult` |
| `nmetl.pipeline_builder` | `PipelineBuilder`, an undoable in-memory editor over a config |
| `nmetl.introspector`, `nmetl.data_preview` | schema discovery, sampling, and query previews over a data source |
| `nmetl.nmetl_cli`, `nmetl.cli` | the `nmetl` command and its sub-commands |
| `nmetl.repl` | interactive Cypher shell over a `pycypher.Star` |
| `nmetl.health_server` | minimal HTTP health / metrics endpoint |
| `nmetl.sinks.neo4j` | write query results into Neo4j with idempotent `MERGE` |

`OutputFormat` is defined in `pycypher.ingestion.output_writer` (the engine
writes results too) and re-exported from `nmetl.config`.

## Tests

```bash
uv run pytest packages/nmetl/tests
```

The suite reuses the engine's session fixtures (Spark, Neo4j, timing
thresholds) from `tests/conftest.py`; static pipeline configs and sample
data live in `packages/nmetl/tests/fixtures/`.

## Documentation

- [Data ETL pipeline tutorial](../../docs/tutorials/data_etl_pipeline.rst)
- [API reference](../../docs/api/nmetl.rst)
- [Deployment guide](../../docs/deployment/index.rst) (Docker image runs `nmetl health-server`)

## License

MIT License - see [LICENSE.txt](LICENSE.txt)
