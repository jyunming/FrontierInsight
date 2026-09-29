"""A quest keeps the data its experiment produced in ``data/``, and a two-script quest keeps both scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _classify, _fake_response_for

_WRITES_A_TABLE = """\
import csv, json
with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["x", "y"])
    w.writeheader()
    w.writerows([{"x": i, "y": i * i} for i in range(5)])
print("RESULT_JSON: " + json.dumps({"mean_y": 6.0}))
"""


def _config(tmp_path: Path) -> Config:
    return Config(
        topic="data kept", title="data kept", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_a_table_the_experiment_wrote_is_kept_in_data(tmp_path: Path, monkeypatch) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Implementation":
            return json.dumps({"code": _WRITES_A_TABLE, "deps": []})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_config(tmp_path))
    await engine.run()
    kept = engine.quest_root / "data" / "results" / "results.csv"
    assert kept.is_file() and kept.read_text(encoding="utf-8").startswith("x,y")


def test_a_two_script_quest_stays_two_scripts_when_a_redesign_reads_less_stochastic(tmp_path: Path) -> None:
    """``split_analysis: auto`` decides from the design; a later design that no longer names a random process must
    not turn a quest that already has ``simulate.py`` into a one-script quest (which deletes it)."""
    cfg = _config(tmp_path)
    assert cfg.execution.split_analysis == "auto"
    engine = Engine(cfg)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    plain = {"design": {"hypothesis": "the value rises", "method": "closed form", "protocol": {}}}
    assert engine._split_on(plain) is False
    (engine.quest_root / "code" / "simulate.py").write_text("print('sim')\n", encoding="utf-8")
    assert engine._split_on(plain) is True


def test_save_run_data_replaces_raw_skips_secrets_and_big_files(tmp_path: Path, monkeypatch) -> None:
    import time

    cfg = _config(tmp_path)
    cfg.execution.split_analysis = True
    engine = Engine(cfg)
    root = engine.quest_root
    root.mkdir(parents=True, exist_ok=True)
    since = time.time()
    (root / "results.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (root / "token.json").write_text("{}", encoding="utf-8")
    (root / "tokens.csv").write_text("t\n1\n", encoding="utf-8")
    (root / "big.npy").write_bytes(b"0" * 2048)
    monkeypatch.setattr(Engine, "_RUN_DATA_MAX_BYTES", 1024)
    raw = root / "raw"
    (raw / "seed0").mkdir(parents=True)
    (raw / "seed0" / "a.csv").write_text("a\n", encoding="utf-8")
    stale = root / "data" / "results" / "raw" / "old_seed"
    stale.mkdir(parents=True)
    (stale / "z.csv").write_text("z\n", encoding="utf-8")
    saved = engine._save_run_data(since, True)
    out = root / "data" / "results"
    assert (out / "results.csv").is_file() and (out / "raw" / "seed0" / "a.csv").is_file()
    assert not (out / "requirements.txt").exists() and not (out / "token.json").exists()
    assert (out / "tokens.csv").is_file(), "a result table whose name holds the word token is kept"
    assert not (out / "big.npy").exists()
    assert not (out / "raw" / "old_seed").exists(), "the raw copy is a snapshot of the last run"
    assert "raw/" in saved
    (raw / "seed1").mkdir()
    (raw / "seed1" / "b.csv").write_text("b\n", encoding="utf-8")
    engine._save_run_data(since, True, root_files=False)
    assert (out / "raw" / "seed1" / "b.csv").is_file()


def test_data_results_is_not_part_of_the_data_a_quest_was_given(tmp_path: Path) -> None:
    from core.attempt_records import _folder_manifest

    data = tmp_path / "data"
    (data / "results").mkdir(parents=True)
    (data / "given.csv").write_text("a\n", encoding="utf-8")
    before = _folder_manifest(data)
    (data / "results" / "out.csv").write_text("b\n", encoding="utf-8")
    assert _folder_manifest(data) == before


@pytest.mark.asyncio
async def test_a_one_script_reply_removes_the_stale_simulation(tmp_path: Path, monkeypatch) -> None:
    """A quest that has ``simulate.py`` asks for two scripts; when the reply holds one (``split_failure`` is not
    ``block``) the old simulation must not stay to run beside the new experiment."""
    async def fake_chat(prompt, **kw):  # noqa: ANN001
        return "```python\nprint('RESULT_JSON: {}')\n```"

    cfg = _config(tmp_path)
    cfg.execution.split_failure = "warn"
    engine = Engine(cfg)
    monkeypatch.setattr(engine, "_chat", fake_chat)
    code_dir = engine.quest_root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "simulate.py").write_text("print('old sim')\n", encoding="utf-8")
    state = {"topic": "t", "design": {"hypothesis": "h", "method": "m", "protocol": {}}, "iteration": 1}
    await engine._node_implement(state)
    assert not (code_dir / "simulate.py").exists()
    assert (code_dir / "experiment.py").is_file()
    assert engine._split_on(state) is False


def test_a_table_from_an_earlier_iteration_is_not_kept_beside_this_runs(tmp_path: Path) -> None:
    import os
    import time

    engine = Engine(_config(tmp_path))
    root = engine.quest_root
    out = root / "data" / "results"
    out.mkdir(parents=True)
    old = out / "table_a.csv"
    old.write_text("a\n", encoding="utf-8")
    os.utime(old, (time.time() - 600, time.time() - 600))
    since = time.time()
    (out / "written_by_the_script.csv").write_text("s\n", encoding="utf-8")
    (root / "table_b.csv").write_text("b\n", encoding="utf-8")
    engine._save_run_data(since, False)
    assert not old.exists()
    assert (out / "table_b.csv").is_file() and (out / "written_by_the_script.csv").is_file()


def test_raw_files_past_the_cap_leave_no_older_partial_copy(tmp_path: Path, monkeypatch) -> None:
    import time

    cfg = _config(tmp_path)
    cfg.execution.split_analysis = True
    engine = Engine(cfg)
    root = engine.quest_root
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "a.csv").write_text("a\n", encoding="utf-8")
    engine._save_run_data(time.time(), True)
    assert (root / "data" / "results" / "raw" / "a.csv").is_file()
    monkeypatch.setattr(Engine, "_RUN_DATA_MAX_BYTES", 4)
    (root / "raw" / "b.csv").write_text("b" * 100, encoding="utf-8")
    engine._save_run_data(time.time(), True)
    assert not (root / "data" / "results" / "raw").exists()


def test_data_results_is_not_offered_to_web_plots_as_user_data(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    data = engine.quest_root / "data"
    (data / "results").mkdir(parents=True)
    (data / "results" / "out.csv").write_text("marker_from_run,1\n", encoding="utf-8")
    (data / "given.csv").write_text("marker_given,2\n", encoding="utf-8")
    text = engine._gather_collected_text({})
    assert "marker_given" in text and "marker_from_run" not in text


@pytest.mark.asyncio
async def test_the_replicate_seeds_raw_files_reach_data_results_and_a_crash_saves_nothing(tmp_path: Path) -> None:
    from tests.test_split_analysis import _engine, _write

    eng = _engine(tmp_path, replicates=2)
    _write(eng)
    out = await eng._node_execute({"deps": []})
    assert out["exec_result"]["returncode"] == 0
    kept = eng.quest_root / "data" / "results" / "raw"
    assert (kept / "seed0" / "values.json").is_file() and (kept / "seed1" / "values.json").is_file()

    crashed = _engine(tmp_path / "crash", replicates=1)
    _write(crashed)
    (crashed.quest_root / "analysis_should_fail").write_text("", encoding="utf-8")
    bad = await crashed._node_execute({"deps": []})
    assert bad["exec_result"]["returncode"] != 0
    assert not (crashed.quest_root / "data" / "results").exists()
