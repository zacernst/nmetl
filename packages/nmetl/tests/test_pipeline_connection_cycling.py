"""``PYCYPHER_DUCKDB_CYCLE_EVERY`` wired into ``_try_streaming_run``.

Board 86bcqr38m / 86bcqm58n: a streaming run with enough sources keeps every
one's table resident on a single DuckDB connection for the whole run, which
OOMs. ``cycle_streaming_connection`` (``pycypher.relation_engine``) is the
primitive; this covers the pipeline-loop wiring that calls it every N
registered sources, end to end through ``nmetl run`` — proving entities and
relationships registered both before *and* after a mid-run cycle still
produce correct query results, and that the opt-in env var is the only
behaviour change (unset means no cycling, same as before this feature).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from click.testing import CliRunner
from nmetl.nmetl_cli import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _write_pipeline(tmp_path: Path, out_a: Path, out_rel: Path) -> Path:
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    c = tmp_path / "c.parquet"
    knows = tmp_path / "knows.parquet"
    pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_parquet(a)
    pd.DataFrame({"id": [1, 2], "name": ["Carol", "Dan"]}).to_parquet(b)
    pd.DataFrame({"id": [1, 2], "name": ["Eve", "Frank"]}).to_parquet(c)
    pd.DataFrame({"__SOURCE__": [1, 2], "__TARGET__": [1, 2]}).to_parquet(
        knows
    )

    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(
        f"""\
version: "1.0"
backend_engine: duckdb
relation_engine: true
sources:
  entities:
    - id: a_src
      uri: "{a}"
      entity_type: A
      id_col: id
    - id: b_src
      uri: "{b}"
      entity_type: B
      id_col: id
    - id: c_src
      uri: "{c}"
      entity_type: C
      id_col: id
  relationships:
    - id: knows_src
      uri: "{knows}"
      relationship_type: KNOWS
      source_col: "__SOURCE__"
      target_col: "__TARGET__"
      source_entity_type: A
      target_entity_type: B
queries:
  - id: q_c
    inline: "MATCH (c:C) RETURN c.name AS name"
  - id: q_rel
    inline: "MATCH (a:A)-[:KNOWS]->(b:B) RETURN a.name AS aname, b.name AS bname"
output:
  - query_id: q_c
    uri: "{out_a}"
    format: parquet
  - query_id: q_rel
    uri: "{out_rel}"
    format: parquet
""",
    )
    return cfg


class TestCycleEveryWiring:
    def test_cycling_every_source_preserves_entities_and_relationships(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYCYPHER_DUCKDB_CYCLE_EVERY", "1")

        import pycypher.relation_engine as relation_engine_module

        calls = []
        original = relation_engine_module.cycle_streaming_connection

        def _counting_cycle(context):
            calls.append(context)
            original(context)

        monkeypatch.setattr(
            relation_engine_module,
            "cycle_streaming_connection",
            _counting_cycle,
        )

        out_c = tmp_path / "out_c.parquet"
        out_rel = tmp_path / "out_rel.parquet"
        cfg = _write_pipeline(tmp_path, out_c, out_rel)

        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "out-of-core" in result.output

        # 3 entities + 1 relationship, cycling after every single one.
        assert len(calls) == 4

        got_c = pd.read_parquet(out_c)
        assert sorted(got_c["name"].tolist()) == ["Eve", "Frank"]

        got_rel = (
            pd.read_parquet(out_rel)
            .sort_values("aname")
            .reset_index(drop=True)
        )
        assert got_rel["aname"].tolist() == ["Alice", "Bob"]
        assert got_rel["bname"].tolist() == ["Carol", "Dan"]

    def test_unset_env_var_never_cycles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PYCYPHER_DUCKDB_CYCLE_EVERY", raising=False)

        import pycypher.relation_engine as relation_engine_module

        calls = []
        monkeypatch.setattr(
            relation_engine_module,
            "cycle_streaming_connection",
            lambda context: calls.append(context),
        )

        out_c = tmp_path / "out_c.parquet"
        out_rel = tmp_path / "out_rel.parquet"
        cfg = _write_pipeline(tmp_path, out_c, out_rel)

        result = CliRunner().invoke(cli, ["run", str(cfg)])
        assert result.exit_code == 0, result.output
        assert calls == []
