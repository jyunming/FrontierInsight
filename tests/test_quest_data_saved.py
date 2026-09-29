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
