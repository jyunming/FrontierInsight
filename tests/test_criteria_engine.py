"""The engine and the criteria (core/criteria.py): the plan step's search-then-ask when the draft names none, the stop or
the warning when there is still none, and the row FI writes after every run."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from core import criteria as cr, frozen_protocol as fp, plan
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from core.knowledge import RetrievedDoc
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_oracle_gate import ORACLE as SCRIPT_ORACLE, _PASSING, _cfg as _gate_cfg, _fake as _gate_fake

CASE_ORACLE = {"name": "rk4 error", "kind": "special_case", "check": "error at t=1 of y'=-y against exp(-1)",
               "expected": 0.0, "tolerance": 1e-5, "case": {"dt": 0.1}, "measure": "error",
               "reference": "derivation: y' = -y, y(0) = 1 gives y(1) = exp(-1) = 0.3679"}
PROTOCOL = {"grid": {"dt": [0.1, 0.05]}, "oracles": [CASE_ORACLE]}
CRITERION = {"name": "rk4 error small", "oracle": "rk4 error", "direction": "lower", "target": 1e-5, "tolerance": 1e-7}

SIMULATE = '''
import math

def run_cell(cell):
    y, dt = 1.0, cell["dt"]
    for _ in range(int(round(1.0 / dt))):
        k1 = -y; k2 = -(y + dt * k1 / 2); k3 = -(y + dt * k2 / 2); k4 = -(y + dt * k3)
        y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return {"error": abs(y - math.exp(-1.0))}
'''


def _config(tmp_path: Path, plan_pause: str = "off") -> Config:
    return Config(
        topic="RK4 on y' = -y", title="criteria", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan=plan_pause, papers=False),
    )


def _design_reply(protocol: dict[str, Any]) -> str:
    body = json.loads(_FAKE_RESPONSES["design"])
    body["protocol"] = protocol
    body["plan"] = {"in_short": "RK4 on a linear ODE.", "literature": [], "gap": "none", "success_criteria": ["x"],
                    "risks": ["y"], "out_of_scope": ["z"]}
    return json.dumps(body)


class _Model:
    """A fake model: the design with ``protocol``, and ``criteria`` for the second question about criteria."""

    def __init__(self, protocol: dict[str, Any], criteria: list[dict[str, Any]] | None) -> None:
        self.protocol, self.criteria, self.asked = protocol, criteria, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith("# Correctness Criteria"):
            self.asked.append(prompt)
            return json.dumps({"criteria": self.criteria or []})
        if _classify(prompt) == "Experiment Design":
            return _design_reply(self.protocol)
        return _fake_response_for(prompt)

    async def aclose(self) -> None:
        return None


class _Search:
    """``Knowledge.asearch`` replaced: records the criteria search (the literature step searches too) and returns one hit."""

    def __init__(self) -> None:
        self.queries: list[str] = []
        search = self

        async def asearch(knowledge, query, **kw):  # noqa: ANN001
            if "correctness" in query:
                search.queries.append(query)
            return [RetrievedDoc(content="Verification uses the observed order of convergence against the exact solution.",
                                 metadata={"title": "Code verification by the method of manufactured solutions",
                                           "doi": "10.1/x"})]

        self.fn = asearch


# --- the plan step ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_draft_with_no_criterion_searches_once_asks_again_and_the_plan_carries_what_came_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = _Search()
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", search.fn)
    engine = Engine(_config(tmp_path))
    model = _Model(PROTOCOL, [CRITERION, {"name": "headline", "trials": "error", "direction": "lower", "tolerance": 1,
                                          "result": "score"}])
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(search.queries) == 1 and "correctness" in search.queries[0]
    assert len(model.asked) == 1 and "manufactured solutions" in model.asked[0] and '"rk4 error"' in model.asked[0]
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert plan.parse(text).design["protocol"]["criteria"] == [{**CRITERION, "use": "error"}]
    section = text.split(f"## {plan.CRITERIA_HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "**rk4 error small**" in section
    # The one the model got wrong is named in the plan, not dropped in silence.
    assert "'headline' takes its number from the script's own results" in text


@pytest.mark.asyncio
async def test_still_none_and_no_plan_pause_warns_plainly_and_records_that_there_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", _Search().fn)
    engine = Engine(_config(tmp_path))
    engine._client = _Model(PROTOCOL, [])
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "[criteria] the plan has no check of correctness FI can compute" in log and "going on without one" in log
    assert "no way to judge whether the code got better" in capsys.readouterr().out
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert "criteria" not in plan.parse(text).design["protocol"], "no criterion is faked"
    section = text.split(f"## {plan.CRITERIA_HEADING}", 1)[1].split("\n## ", 1)[0]
    assert "(none" in section
    assert "has no criterion for judging whether the code got better" in text  # Checks already made
    assert not (engine.fi_dir / "pause.json").exists()


@pytest.mark.asyncio
async def test_a_draft_that_names_criteria_does_not_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    search = _Search()
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", search.fn)
    engine = Engine(_config(tmp_path))
    model = _Model({**PROTOCOL, "criteria": [CRITERION]}, None)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert search.queries == [] and model.asked == []


@pytest.mark.asyncio
async def test_still_none_with_the_plan_pause_on_stops_and_asks_the_person_to_write_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = _Search()
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", search.fn)
    model = _Model(PROTOCOL, [])
    monkeypatch.setattr("core.engine.LLMClient.chat", lambda self, messages, **kw: model.chat(messages, **kw))
    engine = Engine(_config(tmp_path, plan_pause="ask"))
    await engine.run()
    assert len(search.queries) == 1
    descriptor = json.loads((engine.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "plan"
    nxt = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "no way yet to judge whether the code got better" in nxt.splitlines()[0]
    assert "read and edit the plan" in nxt.splitlines()[0]
    assert "`criteria`" in nxt and "Or resume without one" in nxt
    assert not (engine.quest_root / "code" / "experiment.py").exists(), "nothing ran before the person read it"


# --- after every run -------------------------------------------------------------------------------------------------


def _trial_record(root: Path, values: list[float]) -> None:
    """FI's run record and ledger for one setting whose trials returned ``values`` (hash-matched, as FI writes them)."""
    rows, ledger = [], []
    for i, v in enumerate(values):
        vals = {"outbreak": v}
        digest = hashlib.sha256(json.dumps(vals, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
        rows.append({"trial": i, "status": "ok", "values": vals})
        ledger.append({"event": "trial", "status": "ok", "cell": "R0=2", "trial": i, "values_sha256": digest})
    (root / ".fi" / "trials").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "trials" / "run.json").write_text(json.dumps({"cells": [{"key": "R0=2", "rows": rows}]}),
                                                       encoding="utf-8")
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in ledger) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_after_a_run_each_criterion_is_computed_from_what_fi_measured_and_recorded_with_the_code_commit(
    tmp_path: Path,
) -> None:
    import random

    engine = Engine(_config(tmp_path))
    root = engine.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    protocol = plan.normalize_protocol({**PROTOCOL, "criteria": [
        CRITERION,
        {"name": "finer step", "case": {"dt": 0.05}, "measure": "error", "direction": "lower", "target": 1e-6,
         "tolerance": 1e-9},
        {"name": "error bars shrink", "trials": "outbreak", "direction": "target", "target": 0.5, "tolerance": 0.2},
    ]})[0]
    fp.freeze(root, protocol, approved_by="human: test", source="plan.md")
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "ok", "attempts": [{"judged": [
        {"name": "rk4 error", "value": 3.3e-7, "expected": 0.0, "measured_by": "engine", "passed_by_engine": True}]}]}),
        encoding="utf-8")
    rng = random.Random(7)
    _trial_record(root, [float(rng.random() < 0.5) for _ in range(256)])
    has_git = shutil.which("git") is not None
    if has_git:
        from core import code_project

        assert code_project.record_change(root, "code written")
        (root / "code" / "simulate.py").write_text(SIMULATE + "\n# repaired\n", encoding="utf-8")
    engine.executor = SharedInterpreterExecutor(python_version="3.11")
    engine._trial_mode = True
    await engine._record_criteria({}, sys.executable, None, root / "code" / "simulate.py")

    (row,) = cr.history(root)
    by = {c["name"]: c for c in row["criteria"]}
    assert by["rk4 error small"]["value"] == pytest.approx(3.3e-7) and by["rk4 error small"]["met"] is True
    assert by["finer step"]["counts"] is True and 0 < by["finer step"]["value"] < 1e-6, by["finer step"]
    assert by["error bars shrink"]["value"] == pytest.approx(0.5, abs=0.2) and by["error bars shrink"]["counts"] is True
    assert row["run"] == "run_1"
    if has_git:
        # The repair made after the code was recorded is recorded too, so the row names the code that ran.
        from core import code_project

        assert row["code_commit"] == code_project.head(root)[0] and row["code_changed_since_commit"] is False
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "[criteria] run 1: 3 of 3 checks of correctness that FI measured itself are met" in log


@pytest.mark.asyncio
async def test_without_a_trial_contract_a_case_and_a_trial_record_are_named_as_not_measured(tmp_path: Path) -> None:
    engine = Engine(_config(tmp_path))
    root = engine.quest_root
    protocol = plan.normalize_protocol({**PROTOCOL, "criteria": [
        {"name": "finer step", "case": {"dt": 0.05}, "measure": "error", "direction": "lower", "tolerance": 1e-9},
        {"name": "error bars shrink", "trials": "outbreak", "direction": "target", "target": 0.5, "tolerance": 0.2},
    ]})[0]
    fp.freeze(root, protocol, approved_by="human: test", source="plan.md")
    await engine._record_criteria({}, sys.executable, None, root / "code" / "experiment.py")
    (row,) = cr.history(root)
    assert all(c["value"] is None and c["why"] for c in row["criteria"])
    assert "run_trial or run_cell" in row["criteria"][0]["why"] and "runs them itself" in row["criteria"][1]["why"]


@pytest.mark.asyncio
async def test_a_quest_run_through_the_graph_records_its_criteria_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole path, with a fake model: the plan's criterion rests on a check the script answers itself, so the row
    shows its number and says it does not count."""
    calls: list[str] = []
    criterion = {"name": "final size right", "oracle": SCRIPT_ORACLE["name"], "direction": "lower", "target": 0.05,
                 "tolerance": 0.001}
    protocol = {"runs_per_setting": 300, "oracles": [SCRIPT_ORACLE], "criteria": [criterion]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _gate_fake(calls, implement=_PASSING, protocol=protocol))
    engine = Engine(_gate_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert fp.protocol_of(engine.quest_root)["criteria"][0]["name"] == "final size right"
    rows = cr.history(engine.quest_root)
    assert rows, "a row after the run"
    result = rows[-1]["criteria"][0]
    assert result["value"] is not None and result["counts"] is False and result["measured_by"] == "the script"
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "[criteria] run 1: 0 of 0 checks of correctness that FI measured itself are met" in log
    assert "measured by the script itself so not counted" in log
