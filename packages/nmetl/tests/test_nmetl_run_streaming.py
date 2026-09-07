"""Phase 5b (out-of-core DuckDB) — nmetl run streaming integration.

Verifies the run_impl streaming fast path: with the relation engine enabled and
a duckdb backend, an all-eligible pipeline streams each query file->sink via
DuckDB (out-of-core); ineligible pipelines and the disabled default fall back to
the normal in-memory path unchanged.

See docs/duckdb_full_parity_design.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from click.testing import CliRunner
from nmetl.nmetl_cli import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_people(tmp_path: Path) -> Path:
    src = tmp_path / "people.parquet"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "age": [30, 25, 35],
        },
    ).to_parquet(src)
    return src


def _config(
    tmp_path: Path,
    out: Path,
    query: str,
    *,
    relation_engine: bool = False,
) -> Path:
    src = _write_people(tmp_path)
    cfg = tmp_path / "pipeline.yaml"
    relation_engine_line = (
        f"relation_engine: {str(relation_engine).lower()}\n"
        if relation_engine
        else ""
    )
    cfg.write_text(
        f"""\
version: "1.0"
backend_engine: duckdb
{relation_engine_line}sources:
  entities:
    - id: people_src
      uri: "{src}"
      entity_type: Person
      id_col: id
queries:
  - id: q1
    inline: "{query}"
output:
  - query_id: q1
    uri: "{out}"
    format: parquet
""",
    )
    return cfg


class TestStreamingRun:
    def test_eligible_pipeline_streams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYCYPHER_DUCKDB_RELATION_ENGINE", "1")
        out = tmp_path / "out.parquet"
        cfg = _config(
            tmp_path,
            out,
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
        )

        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output  # took the streaming path
        got = pd.read_parquet(out).sort_values("name").reset_index(drop=True)
        assert got["name"].tolist() == ["Alice", "Bob", "Carol"]
        assert set(got.columns) == {"name", "age"}

    def test_relation_engine_enabled_via_config(self, tmp_path: Path) -> None:
        # No env var — `relation_engine: true` in the YAML alone must enable
        # the streaming path.
        out = tmp_path / "out.parquet"
        cfg = _config(
            tmp_path,
            out,
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
            relation_engine=True,
        )

        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output
        got = pd.read_parquet(out).sort_values("name").reset_index(drop=True)
        assert got["name"].tolist() == ["Alice", "Bob", "Carol"]


def _config_multi(
    tmp_path: Path,
    out: Path,
    queries: list[tuple[str, str]],
) -> Path:
    """Build a pipeline config from *queries* (a list of ``(id, inline)``
    pairs); only the last query gets an output sink, to *out*.
    """
    src = _write_people(tmp_path)
    cfg = tmp_path / "pipeline.yaml"
    queries_yaml = "\n".join(
        f'  - id: {qid}\n    inline: "{text}"' for qid, text in queries
    )
    last_id = queries[-1][0]
    cfg.write_text(
        f"""\
version: "1.0"
backend_engine: duckdb
relation_engine: true
sources:
  entities:
    - id: people_src
      uri: "{src}"
      entity_type: Person
      id_col: id
queries:
{queries_yaml}
output:
  - query_id: {last_id}
    uri: "{out}"
    format: parquet
""",
    )
    return cfg


class TestMutationInterleavedWithRead:
    """A no-sink SET/CREATE/DELETE mutation ahead of a read+sink query still
    takes the streaming path, and the sink reflects the mutation.
    """

    def test_set_then_read_streams(self, tmp_path: Path) -> None:
        out = tmp_path / "out.parquet"
        cfg = _config_multi(
            tmp_path,
            out,
            [
                ("q1", "MATCH (n:Person) WHERE n.age > 28 SET n.age = 99"),
                ("q2", "MATCH (n:Person) RETURN n.name AS name, n.age AS age"),
            ],
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output
        got = pd.read_parquet(out).sort_values("name").reset_index(drop=True)
        expected = {"Alice": 99, "Bob": 25, "Carol": 99}
        assert dict(zip(got["name"], got["age"])) == expected

    def test_create_then_delete_then_read_streams(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.parquet"
        cfg = _config_multi(
            tmp_path,
            out,
            [
                ("q1", "CREATE (n:Person {name: 'Dave', age: 40})"),
                ("q2", "MATCH (n:Person) WHERE n.name = 'Bob' DELETE n"),
                ("q3", "MATCH (n:Person) RETURN n.name AS name, n.age AS age"),
            ],
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output
        got = pd.read_parquet(out).sort_values("name").reset_index(drop=True)
        assert got["name"].tolist() == ["Alice", "Carol", "Dave"]
        assert dict(zip(got["name"], got["age"]))["Dave"] == 40


class TestCrossQuerySequencing:
    """Phase 3b (the FastOpenData streaming-qualification plan (private repository)) -- a
    query reading a property that an *earlier* query's SET creates fresh
    (not present in the raw source at all) still streams, because the
    earlier mutation executes before the later query's own eligibility is
    checked, not only at final execution time.
    """

    def test_read_of_earlier_set_created_column_streams(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.parquet"
        cfg = _config_multi(
            tmp_path,
            out,
            [
                (
                    "q1",
                    "MATCH (n:Person) WHERE n.age > 28 SET n.tier = 'senior'",
                ),
                (
                    "q2",
                    (
                        "MATCH (n:Person) WHERE n.tier = 'senior' "
                        "RETURN n.name AS name"
                    ),
                ),
            ],
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output
        got = pd.read_parquet(out)
        assert sorted(got["name"].tolist()) == ["Alice", "Carol"]


class TestNoFallback:
    """Generalisation plan, Phase 5: with the relation engine enabled there
    is no fallback to the in-memory engine. An unsupported construct is a
    per-query error naming the construct, under the --on-error policy."""

    def test_unsupported_query_fails_the_run_naming_the_construct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYCYPHER_DUCKDB_RELATION_ENGINE", "1")
        out = tmp_path / "out.parquet"
        # SKIP without LIMIT has no translation rule.
        cfg = _config(
            tmp_path,
            out,
            "MATCH (n:Person) RETURN n.name AS name ORDER BY name SKIP 2",
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code != 0
        assert "SKIP without LIMIT" in result.output
        assert "q1" in result.output
        assert "Falling back" not in result.output
        assert not out.exists()

    def test_on_error_warn_continues_past_an_unsupported_query(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYCYPHER_DUCKDB_RELATION_ENGINE", "1")
        out = tmp_path / "out.parquet"
        cfg = _config(
            tmp_path,
            out,
            "MATCH (n:Person) RETURN n.name AS name ORDER BY name SKIP 2",
        )
        result = CliRunner().invoke(
            cli, ["run", str(cfg), "--on-error", "warn"]
        )
        assert result.exit_code == 0, result.output
        assert "SKIP without LIMIT" in result.output
        assert "out-of-core" in result.output

    def test_disabled_by_default_uses_normal_path(
        self, tmp_path: Path
    ) -> None:
        # No env flag => streaming never attempted => normal path, correct output.
        out = tmp_path / "out.parquet"
        cfg = _config(
            tmp_path,
            out,
            "MATCH (n:Person) RETURN n.name AS name, n.age AS age",
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "query/queries (out-of-core)" not in result.output
        # backend_engine: duckdb is set (by _config) but relation_engine is
        # not, so this is a real (if expected) fallback and should warn —
        # not the silent "streaming never requested" case.
        assert "Running on the in-memory engine" in result.output
        assert "relation engine is not enabled" in result.output
        got = pd.read_parquet(out)
        assert sorted(got["name"].tolist()) == ["Alice", "Bob", "Carol"]


def _config_overlapping_surveys(tmp_path: Path, out: Path) -> Path:
    """Two entity types sharing id values, each with its own edge file into
    the same PUMAs -- the fastopendata 1-year/5-year survey shape. With the
    endpoint types declared, a 1-year aggregate must not count 5-year edges.
    """
    pd.DataFrame({"pfips": ["p1", "p2"]}).to_parquet(tmp_path / "puma.parquet")
    pd.DataFrame({"serial": ["u1", "u2", "u3"]}).to_parquet(
        tmp_path / "hus_1yr.parquet"
    )
    pd.DataFrame({"serial": ["u1", "u2", "u3", "u4"]}).to_parquet(
        tmp_path / "hus_5yr.parquet"
    )
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3"], "pfips": ["p1", "p1", "p2"]}
    ).to_parquet(tmp_path / "edges_1yr.parquet")
    pd.DataFrame(
        {"serial": ["u1", "u2", "u3", "u4"], "pfips": ["p1", "p1", "p2", "p2"]}
    ).to_parquet(tmp_path / "edges_5yr.parquet")
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        f"""\
version: "1.0"
backend_engine: duckdb
relation_engine: true
sources:
  entities:
    - id: puma
      uri: "{tmp_path / "puma.parquet"}"
      entity_type: PUMA
      id_col: pfips
    - id: hus_1yr
      uri: "{tmp_path / "hus_1yr.parquet"}"
      entity_type: HousingSurvey1yr
      id_col: serial
    - id: hus_5yr
      uri: "{tmp_path / "hus_5yr.parquet"}"
      entity_type: HousingSurvey5yr
      id_col: serial
  relationships:
    - id: edges_1yr
      uri: "{tmp_path / "edges_1yr.parquet"}"
      relationship_type: LOCATED_IN
      source_col: serial
      target_col: pfips
      source_entity_type: HousingSurvey1yr
      target_entity_type: PUMA
    - id: edges_5yr
      uri: "{tmp_path / "edges_5yr.parquet"}"
      relationship_type: LOCATED_IN
      source_col: serial
      target_col: pfips
      source_entity_type: HousingSurvey5yr
      target_entity_type: PUMA
queries:
  - id: count_1yr
    inline: "MATCH (pu:PUMA)<-[:LOCATED_IN]-(h:HousingSurvey1yr) WITH pu, COUNT(h) AS n SET pu.units_1yr = n"
  - id: read
    inline: "MATCH (pu:PUMA) RETURN id(pu) AS puma, pu.units_1yr AS units_1yr"
output:
  - query_id: read
    uri: "{out}"
    format: parquet
""",
    )
    return cfg


class TestReadWithoutSink:
    def test_read_without_sink_runs_its_side_effects(
        self, tmp_path: Path
    ) -> None:
        # A RETURN query with no output sink used to make the whole run fall
        # back; now it runs (its mid-pipeline SET is durable) and its result
        # is discarded.
        out = tmp_path / "out.parquet"
        cfg = _config_multi(
            tmp_path,
            out,
            [
                (
                    "q1",
                    (
                        "MATCH (n:Person) WITH n SET n.tag = n.age * 2 "
                        "WITH n.tag AS tag RETURN tag"
                    ),
                ),
                ("q2", "MATCH (n:Person) RETURN n.name AS name, n.tag AS tag"),
            ],
        )
        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "no output sink" in result.output
        got = pd.read_parquet(out).sort_values("name").reset_index(drop=True)
        assert got["tag"].tolist() == [60, 50, 70]


class TestDeclaredEndpointTypesThroughTheCli:
    def test_declared_endpoint_types_reach_the_streaming_join(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.parquet"
        cfg = _config_overlapping_surveys(tmp_path, out)

        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output
        assert "Falling back" not in result.output
        got = pd.read_parquet(out).sort_values("puma").reset_index(drop=True)
        # p1 has u1,u2 in the 1-year file; p2 has u3. Without the endpoint
        # types the 5-year edges (same serials) would make this 4 and 2.
        assert got["puma"].tolist() == ["p1", "p2"]
        assert got["units_1yr"].tolist() == [2, 1]
