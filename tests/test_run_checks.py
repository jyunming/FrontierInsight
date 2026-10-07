"""A result typed into the code is sent back to be computed and, when it stays, left out of the paper
(core/run_checks.py). Fake model, toy simulations (a cooling cup), real subprocesses."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from core import trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor

# ---- a result typed into the code ---------------------------------------------------------------------------------------

TYPED_SIM = '''\
import random

def run_trial(cell, trial, seed):
    rng = random.Random(seed)
    temp = 90.0
    for _ in range(int(cell["n"])):
        temp += (20.0 - temp) * 0.01 + rng.gauss(0, 0.1)
    return {"final_temp": temp, "lid_on": 1.0}
'''
COMPUTED_SIM = TYPED_SIM.replace('"lid_on": 1.0', '"lid_on": float(temp < 80.0)')


class _Client:
    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _typed_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic about a cooling cup", title="typed", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(TYPED_SIM, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_typed_in_result_is_sent_back_and_a_repair_that_computes_it_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": COMPUTED_SIM, "patch_summary": "computed"})])
    assert await eng._results_from_computation({}) is True and len(asked) == 1
    assert "simulate.py line 8: `lid_on` is returned as 1.0, a number typed into the code" in asked[0]
    assert "compute it from the simulation" in asked[0]
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == COMPUTED_SIM
    assert eng._left_out_quantities() == set() and eng._run_checks_note() == ""
    assert not (eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").exists()


@pytest.mark.asyncio
async def test_a_typed_in_result_that_stays_is_left_out_with_a_plain_note_and_is_not_asked_about_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": TYPED_SIM, "patch_summary": "no change"})])
    assert await eng._results_from_computation({}) is False
    assert len(asked) == 2, "asked twice, the simulation kept as written"
    assert eng._left_out_quantities() == {"lid_on"}
    record = json.loads((eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_quantities"] == ["lid_on"] and "typed into the code" in record["note"]
    note = eng._run_checks_note()
    assert "`lid_on` was left out" in note and "limitations" in note and "not computed by the simulation" in note
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "The paper does not use it and says so in its limitations" in log
    # The quantity is not in the result the analysis hands on.
    assert eng._without_typed_results({"final_temp": 1.0, "lid_on": 1.0, "rows": [{"lid_on": 1}]}) == {
        "final_temp": 1.0, "rows": [{}]}
    # A resume (the same simulation) asks no more, and still leaves it out.
    assert await eng._results_from_computation({}) is False and len(asked) == 2
    assert eng._left_out_quantities() == {"lid_on"}
    # Once the simulation computes it, the record goes.
    (eng.quest_root / "code" / "simulate.py").write_text(COMPUTED_SIM, encoding="utf-8")
    await eng._results_from_computation({})
    assert eng._left_out_quantities() == set() and eng._run_checks_note() == ""


@pytest.mark.asyncio
async def test_a_repair_that_is_not_the_same_simulation_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for reply in ("{}", json.dumps({"code": "print('hello')\n"}), json.dumps({"code": "def broken(:\n"}),
                  json.dumps({"code": TYPED_SIM})):
        eng, asked = _typed_engine(tmp_path / re.sub(r"\W", "", reply)[:8], monkeypatch, [reply])
        assert await eng._results_from_computation({}) is False
        assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == TYPED_SIM, reply


@pytest.mark.asyncio
async def test_a_simulation_with_nothing_typed_in_is_not_asked_about(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _typed_engine(tmp_path, monkeypatch, ["{}"])
    (eng.quest_root / "code" / "simulate.py").write_text(COMPUTED_SIM, encoding="utf-8")
    assert await eng._results_from_computation({}) is False and asked == []


ANALYSIS = '''\
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
names = sorted({m for c in data["cells"] for m in c["metrics"]})
print("RESULT_JSON: " + json.dumps({"seen": names}))
'''


def test_the_analysis_is_not_given_a_left_out_quantity_but_the_ledger_keeps_every_value(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(TYPED_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    cmd = [sys.executable, str(root / "code" / "experiment.py")]

    def run(leave_out: Any):  # noqa: ANN202
        runner = trial_runner.TrialsRunner(
            SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
            protocol={"grid": {"n": [3]}, "runs_per_setting": 2}, deterministic=False,
            simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py", leave_out=leave_out)
        return asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={}))

    assert '"lid_on"' in run(None).stdout
    result = run(lambda: {"lid_on"})
    assert result.returncode == 0 and '"seen": ["final_temp"]' in result.stdout
    ledger = (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8")
    assert "values_sha256" in ledger
    summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    assert "lid_on" in summary["cells"][0]["metrics"], "FI's own summary is not changed"
