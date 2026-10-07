"""An experiment whose result holds no number is not findings: it goes back to the design once, and when it still
measures nothing the quest stops with no paper (needs/STUCK.json), in plain words."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _classify, _fake_response_for

EMPTY_CODE = _FAKE_EXPERIMENT_CODE.replace('{"score": 0.987}', '{"status": "done", "ok": true}')


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="a neutral topic about a cooling curve", title="no-numbers", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False, auto_accept_on_pass=True, oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _chat(calls: list[str], code: str):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Implementation":
            return json.dumps({"code": code, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    return fake_chat


def test_the_fixture_prints_a_result_with_no_number() -> None:
    assert '"status"' in EMPTY_CODE and "0.987" not in EMPTY_CODE


@pytest.mark.asyncio
async def test_a_result_with_no_number_goes_back_to_design_once_then_stops_with_no_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _chat(calls, EMPTY_CODE))
    eng = Engine(_cfg(tmp_path))
    art = await eng.run()
    assert art.paper_md is None
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert record["kind"] == "no_findings" and "no number in it" in record["problem"]
    assert calls.count("Experiment Design") == 2, "the design was asked for once more"
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "sending the quest back to design once" in log and "stopping without a paper" in log
    assert "Writing" not in calls
    out = capsys.readouterr().out
    assert "has no number in it" in out and "No paper was written" in out


@pytest.mark.asyncio
async def test_a_result_with_a_number_is_written_up_as_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _chat(calls, _FAKE_EXPERIMENT_CODE))
    eng = Engine(_cfg(tmp_path))
    art = await eng.run()
    assert art.paper_md is not None and not (eng.quest_root / "needs" / "STUCK.json").is_file()
    assert calls.count("Experiment Design") == 1


def test_only_a_real_number_counts_as_a_result(tmp_path: Path) -> None:
    eng = Engine(_cfg(tmp_path))
    base = {"exec_result": {"returncode": 0}, "iteration": 2}  # no iteration left: the verdict is final
    for empty in ({"status": "done", "ok": True, "note": "n/a"}, {"a": [], "b": {"c": None}}):
        ruled = eng._no_results_verdict({**base, "result_json": empty})
        assert ruled is not None and ruled["stuck"] is True and ruled["stuck_reason"] == "no_numbers"
    assert eng._no_results_verdict({**base, "result_json": {"x": {"y": [1.5]}}}) is None
