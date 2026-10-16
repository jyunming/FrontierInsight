"""Through the real graph: under ``engine.run_manifest_check: warn`` (the default) a difference in what was counted is
recorded and the quest goes on, but numbers FI's trials never produced are not a count: an analysis that prints one is
sent back, and when it stays the quest stops. Fake model, real venv."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import run_manifest as rm
from core import trial_runner as tr
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_run_manifest import _MEAN_PROTOCOL, _cfg, _record, _reply, ANALYSIS_TRIAL, SIM_TRIAL

# One value of one setting replaced by a number no trial returned (the review's err 0.33 case).
ANALYSIS_MADE_UP = ANALYSIS_TRIAL.replace(
    '    out[c["key"]] = {"final_size_values": m["values"], "final_size_count": m["count"]}',
    '    vals = m["values"][:-1] + [0.33] if c["key"] == "R0=0.9" else m["values"]\n'
    '    out[c["key"]] = {"final_size_values": vals, "final_size_count": m["count"]}',
)


def _model(monkeypatch: pytest.MonkeyPatch, first: str, repairs: list[str], calls: list[str]) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = _MEAN_PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(SIM_TRIAL, first)
        if kind == "ExecuteReflect":
            code = repairs[min(calls.count("ExecuteReflect") - 1, len(repairs) - 1)]
            return json.dumps({"code": code, "deps": [], "patch_summary": "tries again"})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)


def test_the_kind_travels_with_the_finding() -> None:
    made_up = tr.reported_values_not_run({"err": __import__("collections").Counter({"0.1": 5, "0.2": 5})},
                                         {"s": {"err_values": [0.1] * 5 + [0.2] * 4 + [0.33]}})
    assert made_up and all(rm.kind_of(f) == rm.FABRICATED for f in made_up)
    assert "never produced (e.g. 0.33)" in made_up[0]
    assert rm.kind_of("the protocol fixes 3 runs per setting; 1 setting(s) ran another number") == rm.UNACCOUNTED
    assert json.dumps(made_up) == json.dumps([str(f) for f in made_up]), "a finding is still its sentence"


@pytest.mark.asyncio
async def test_a_value_no_trial_produced_is_sent_back_and_stops_under_warn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_MADE_UP, [ANALYSIS_MADE_UP], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert calls.count("ExecuteReflect") >= 1, "the analysis is sent back"
    assert artifacts.paper_md is None and not (engine.quest_root / "paper" / "paper.md").exists()
    record = _record(engine)
    assert record["status"] == "stopped" and any("never produced" in p for p in record["problems"])


@pytest.mark.asyncio
async def test_a_value_no_trial_produced_that_the_repair_removes_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_MADE_UP, [ANALYSIS_TRIAL], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert calls.count("ExecuteReflect") == 1 and artifacts.paper_md is not None
    assert _record(engine)["status"] == "ok"
