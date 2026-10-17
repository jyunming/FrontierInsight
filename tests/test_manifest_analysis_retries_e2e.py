"""Through the real graph: an analysis that leaves out what the run record needs is sent back alone up to three times (the
simulation stands, nothing is run again), and when none of the three passes the same check the quest ends in the honest
stop. Fake model, real venv."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_run_manifest import (
    _MEAN_PROTOCOL, _cfg, _reply, _stuck, ANALYSIS, ANALYSIS_WITH_VALUES, SIM_OK,
)


def _model(monkeypatch: pytest.MonkeyPatch, repairs: list[str], calls: list[str], prompts: list[str]) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        prompts.append(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = _MEAN_PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(SIM_OK, ANALYSIS)
        if kind == "ExecuteReflect":
            code = repairs[min(calls.count("ExecuteReflect") - 1, len(repairs) - 1)]
            return json.dumps({"code": code, "deps": [], "patch_summary": "tries again"})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)


@pytest.mark.asyncio
async def test_an_analysis_wrong_twice_is_repaired_on_the_third_try(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    prompts: list[str] = []
    _model(monkeypatch, [ANALYSIS, ANALYSIS, ANALYSIS_WITH_VALUES], calls, prompts)
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("ExecuteReflect") == 3
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "sending experiment.py back (1 of 3)" in log and "(3 of 3)" in log
    assert all("Rewrite experiment.py (not simulate.py)" in p for p in prompts if _classify(p) == "ExecuteReflect")
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8").strip() == SIM_OK.strip()


@pytest.mark.asyncio
async def test_an_analysis_that_stays_wrong_after_three_tries_ends_in_the_honest_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    prompts: list[str] = []
    _model(monkeypatch, [ANALYSIS], calls, prompts)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    assert calls.count("ExecuteReflect") == 3, "three tries, then the check stands"
    assert "back to be repaired 3 time(s)" in _stuck(engine)
    assert not (engine.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_the_analysis_repairs_have_a_budget_of_their_own_apart_from_crash_repairs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one crash repair allowed, three repairs of the analysis still run, and spend none of it."""
    calls: list[str] = []
    prompts: list[str] = []
    _model(monkeypatch, [ANALYSIS, ANALYSIS, ANALYSIS_WITH_VALUES], calls, prompts)
    engine = Engine(_cfg(tmp_path, engine={"exec_reflect_max_iterations": 1}))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("ExecuteReflect") == 3


@pytest.mark.asyncio
async def test_a_crash_between_analysis_repairs_does_not_earn_them_a_fresh_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An analysis repair that crashes is fixed by a crash repair; the analysis repairs already spent stay spent, so the
    analysis still gets three repairs in all (four requests with the crash fix), not three more after the crash."""
    calls: list[str] = []
    prompts: list[str] = []
    crashing = 'raise RuntimeError("the analysis broke")\n'
    _model(monkeypatch, [crashing, ANALYSIS], calls, prompts)
    engine = Engine(_cfg(tmp_path, engine={"exec_reflect_max_iterations": 3}))
    await engine.run()
    assert calls.count("ExecuteReflect") == 4, calls.count("ExecuteReflect")
    assert not (engine.quest_root / "paper" / "paper.md").exists()
