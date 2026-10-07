"""The nodes that read a model's design or analysis: LaTeX in the reply is read, and a reply that cannot be read twice
stops the quest instead of running an empty design (tests use a stubbed client; none calls a model)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import plan as _plan
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

BS = "\\"

DESIGN = {
    "hypothesis": "SNR rises as SIGMA falls",
    "variables": {"independent": ["sigma"], "dependent": ["snr"], "controls": ["seed"]},
    "method": "sweep SIGMA from 0.3 to 0.9 and measure TEXT",
    "expected_outcome": "SNR falls with sigma",
    "figures_planned": ["c.png"],
    "dependencies": ["numpy"],
    "protocol": {"runs_per_setting": 300},
}
SIGMA = "$" + BS + "sigma=0.6$"
TEXT = "$" + BS + "text{SNR}$"


def _latex_json(obj: dict) -> str:
    """JSON as a model writes it when it puts LaTeX in a string: single backslashes (not valid JSON)."""
    return json.dumps(obj).replace("SIGMA", SIGMA).replace("TEXT", TEXT)


def _engine(tmp_path: Path, replies: list) -> Engine:
    cfg = Config(
        topic="SNR under noise", title="snr", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=replies)})()
    return eng


def _stops(eng: Engine) -> list[dict]:
    stops: list[dict] = []

    def pause(**kwargs):  # noqa: ANN003
        stops.append(kwargs)
        raise RuntimeError("stopped")

    eng._pause_for_human = pause  # type: ignore[method-assign]
    return stops


AUDIT = {"objections_addressed": [], "amended_design": DESIGN}


@pytest.mark.asyncio
async def test_the_plan_node_with_a_latex_reply_makes_a_real_design(tmp_path: Path) -> None:
    reply = _latex_json(DESIGN)
    with pytest.raises(json.JSONDecodeError):
        json.loads(reply)
    eng = _engine(tmp_path, [reply, _latex_json(AUDIT)])
    await eng._node_plan({"topic": "SNR", "iteration": 0})
    design, why = _plan.load_design(eng.quest_root)
    assert design is not None, why
    assert design["hypothesis"] == "SNR rises as " + SIGMA + " falls"
    assert TEXT in design["method"]
    assert "(parse failed)" not in _plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "asking once more" not in log  # the first answer was read: the model was not asked again


@pytest.mark.asyncio
async def test_the_design_node_with_a_latex_reply_makes_a_real_design(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [_latex_json(DESIGN), _latex_json(AUDIT)])
    out = await eng._node_design({"topic": "SNR", "iteration": 0})
    assert out["design"]["hypothesis"] == "SNR rises as " + SIGMA + " falls"
    assert TEXT in out["design"]["method"]


@pytest.mark.asyncio
async def test_a_design_that_cannot_be_read_is_asked_for_once_more_then_the_quest_stops_and_nothing_runs(tmp_path: Path) -> None:
    eng = _engine(tmp_path, ["I would design an experiment about {SNR", "sorry, here is a description instead"])
    stops = _stops(eng)
    with pytest.raises(RuntimeError, match="stopped"):
        await eng._node_plan({"topic": "SNR", "iteration": 0})
    assert eng._client.chat.await_count == 2  # asked once, then once more: no third call, no audit of an empty design
    (stop,) = stops
    assert stop["kind"] == "model_unreadable" and stop["interaction"] == "supply"
    text = " ".join([stop["headline"], *stop["steps"]])
    assert "could not be read" in text and "asked twice" in text and "Nothing was run" in text
    assert "node_models" in text
    for word in ("debug", "design block", "write a design", "YAML"):
        assert word not in text, word
    assert not _plan.plan_path(eng.quest_root).exists()


@pytest.mark.asyncio
async def test_the_design_node_stops_the_same_way_instead_of_going_on_with_an_empty_design(tmp_path: Path) -> None:
    eng = _engine(tmp_path, ["no json here", "still none"])
    stops = _stops(eng)
    with pytest.raises(RuntimeError, match="stopped"):
        await eng._node_design({"topic": "SNR", "iteration": 0})
    assert [s["kind"] for s in stops] == ["model_unreadable"]
    assert eng._client.chat.await_count == 2


@pytest.mark.asyncio
async def test_a_second_answer_that_can_be_read_is_used(tmp_path: Path) -> None:
    eng = _engine(tmp_path, ["no json here", json.dumps(DESIGN), json.dumps(AUDIT)])
    out = await eng._node_design({"topic": "SNR", "iteration": 0})
    assert out["design"]["hypothesis"] == DESIGN["hypothesis"]


_ANALYZE_STATE = {
    "result_json": {"contrast": 0.45, "SNR": 1.42}, "exec_result": {"returncode": 0}, "figures": [], "design": {},
}


def _analysis_engine(tmp_path: Path, replies: list) -> Engine:
    eng = _engine(tmp_path, replies)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    return eng


@pytest.mark.asyncio
async def test_an_analysis_with_latex_is_read(tmp_path: Path) -> None:
    reply = ('{"summary": "SNR of ' + TEXT + ' is 1.42 at ' + SIGMA + '", "key_findings": ["a"], "next_step": "publish"}')
    eng = _analysis_engine(tmp_path, [reply])
    patch = await eng._node_analyze(dict(_ANALYZE_STATE))  # type: ignore[arg-type]
    assert patch["analysis"]["key_findings"] == ["a"] and "unreadable" not in patch["analysis"]
    assert SIGMA in patch["analysis"]["summary"] and TEXT in patch["analysis"]["summary"]


@pytest.mark.asyncio
async def test_an_analysis_unreadable_twice_is_not_findings_and_the_evidence_gate_fails_closed(tmp_path: Path) -> None:
    eng = _analysis_engine(tmp_path, ["not json", "still not json"])
    patch = await eng._node_analyze(dict(_ANALYZE_STATE))  # type: ignore[arg-type]
    assert eng._client.chat.await_count == 2
    assert patch["analysis"]["unreadable"] is True and patch["analysis"]["key_findings"] == []

    gate = _analysis_engine(tmp_path / "g", [RuntimeError("the gate must not ask a model")])
    out = await gate._node_evidence_gate({"topic": "SNR", **_ANALYZE_STATE, "analysis": patch["analysis"]})
    assessment = out["evidence_assessment"]
    assert assessment["verdict"] == "insufficient" and "could not be read" in assessment["rationale"]
    assert assessment["decided_by"] == "rule" and not out.get("redesign")
    assert gate._client.chat.await_count == 0


def test_a_plan_rewrite_with_latex_in_a_double_quoted_value_keeps_it() -> None:
    block = ('hypothesis: "SNR of $' + BS + 'text{SNR}$ rises"\n'
             'variables:\n  independent: ["' + BS + 'sigma"]\n  dependent: [snr]\n  controls: []\n'
             'method: "sweep ' + BS + 'sigma"\nexpected_outcome: x\nfigures_planned: [c.png]\ndependencies: [numpy]\n')
    text = "# Plan\n\n## " + _plan.DESIGN_HEADING + "\n\n```yaml\n" + block + "```\n"
    # as written, `\t` of `\text` is read as a tab, and `\s` is no escape at all
    assert _plan.parse(text).design is None
    kept = _plan.keep_latex_in_block(text)
    design = _plan.parse(kept).design
    assert design["hypothesis"] == "SNR of $" + BS + "text{SNR}$ rises"
    assert design["variables"]["independent"] == [BS + "sigma"]
    assert _plan.keep_latex_in_block("no block here " + BS + "text") == "no block here " + BS + "text"


@pytest.mark.asyncio
async def test_an_analysis_unreadable_twice_stops_the_whole_quest_with_no_paper_and_a_stuck_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_engine_smoke import _classify, _fake_response_for

    analysis_calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Analysis":
            analysis_calls.append(prompt)
            return "I looked at the results but cannot put them in the shape asked for"
        return _fake_response_for(prompt)

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = Config(
        topic="a topic for the unreadable analysis test", title="unreadable-analysis", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    await eng.run()

    assert len(analysis_calls) == 2  # asked, then asked once more: nothing else
    root = eng.quest_root
    assert not (root / "paper.md").exists() and not (root / "paper" / "paper.md").exists()
    record = json.loads((root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert "could not be read, even when asked twice" in record["problem"] and "no findings to write up" in record["problem"]
    assert len(record["tried"]) == 3
    assert "debug" not in json.dumps(record).lower()
    log = (root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[stuck] the experiment ran, but the model's reading of its results could not be read" in log
    assert (root / "code").is_dir() and any((root / "code").iterdir())  # the run's files are kept
    from core import todo
    assert any(i.kind == "stuck" for i in todo.waiting(root))
