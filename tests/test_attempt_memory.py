"""Shadow recommendations from past attempts (core/attempt_memory.py): recorded at decisions, never acted on.

BLOCK only for the same complete conditions that failed the same way more than once, and only before an experiment
runs; another model or FI version is at most VERIFY; a record that is incomplete or of an older schema gives INFO at
most. A quest runs exactly the same with the recording on or off, and a failure in it never touches the quest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import attempt_memory as mem
from core import attempt_records as ar
from core import audit_log


def _ctx(**over) -> dict:
    ctx = {
        "context_kind": "after_run", "provider": "openai", "model": "m1", "settings": {"node_models": {}},
        "prompts_sha256": "p1", "fi": {"source_sha256": "fi1"}, "skills": [], "environment_sha256": "env1",
        "dependency_lock_sha256": None, "inputs": {"sha256": "in1"}, "data": None, "code": {"simulate.py": "c1"},
        "question": {"topic_sha256": "t1", "design_sha256": "d1"}, "protocol_sha256": "pr1",
        "metric_specs_sha256": "ms1", "budget": {"max_iterations": 3}, "missing": [], "complete": True,
    }
    ctx.update(over)
    return ctx


def _failed(record_id: str, *, outcome: str = "process_error", schema: int = ar.SCHEMA, **ctx) -> mem.Attempt:
    record = {"kind": "run", "outcome": outcome, "record_id": record_id, "quest_id": "q0", "schema": schema,
              "context": _ctx(**ctx)}
    attempt = mem._as_attempt(record)
    assert attempt is not None
    return attempt


def test_block_needs_the_same_complete_conditions_to_have_failed_the_same_way_twice() -> None:
    once = [_failed("a")]
    assert mem.recommend(_ctx(), once, decision="execute").action == "VERIFY", "one failure is a check, not a block"
    twice = [_failed("a"), _failed("b")]
    rec = mem.recommend(_ctx(), twice, decision="execute")
    assert rec.action == "BLOCK" and set(rec.matched_attempt_ids) == {"a", "b"} and rec.failure_class == "process_error"
    assert mem.recommend(_ctx(), [_failed("a"), _failed("b", outcome="oracle_failure")],
                         decision="execute").action == "VERIFY", "two different failures are not one reproduced hazard"


def test_a_plan_or_a_repair_is_at_most_a_check() -> None:
    twice = [_failed("a"), _failed("b")]
    assert mem.recommend(_ctx(), twice, decision="plan").action == "VERIFY"
    assert mem.recommend(_ctx(), twice, decision="repair").action == "VERIFY"


def test_another_model_or_fi_version_is_at_most_a_check() -> None:
    twice = [_failed("a", model="m2"), _failed("b", model="m2")]
    rec = mem.recommend(_ctx(), twice, decision="execute")
    assert rec.action == "VERIFY" and "model" in rec.fields_differed
    assert mem.recommend(_ctx(), [_failed("a", fi={"source_sha256": "fi2"}), _failed("b", fi={"source_sha256": "fi2"})],
                         decision="execute").action == "VERIFY"


def test_an_incomplete_or_older_record_gives_a_note_at_most() -> None:
    incomplete = [_failed("a", complete=False, missing=["no environment record"]),
                  _failed("b", complete=False, missing=["no environment record"])]
    assert mem.recommend(_ctx(), incomplete, decision="execute").action == "INFO"
    older = [_failed("a", schema=ar.SCHEMA - 1), _failed("b", schema=ar.SCHEMA - 1)]
    assert mem.recommend(_ctx(), older, decision="execute").action == "INFO"
    assert mem.recommend(_ctx(complete=False), [_failed("a"), _failed("b")], decision="execute").action == "INFO", \
        "the decision's own conditions not fully known"


def test_a_similar_failure_is_a_note_and_nothing_comparable_is_ignored() -> None:
    similar = [_failed("a", code={"simulate.py": "other"}, environment_sha256="env2")]
    assert mem.recommend(_ctx(), similar, decision="execute").action == "INFO", "same design and protocol"
    unrelated = [_failed("a", question={"topic_sha256": "t9", "design_sha256": "d9"}, protocol_sha256="pr9",
                         code={"x.py": "z"})]
    assert mem.recommend(_ctx(), unrelated, decision="execute").action == "IGNORE"
    assert mem.recommend(_ctx(), [], decision="execute").action == "IGNORE"


def test_only_failures_are_indexed(tmp_path: Path) -> None:
    fi = tmp_path / "q1" / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "inconclusive", "quest_id": "q1", "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "oracle_failure", "quest_id": "q1", "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "stop", "pause": "manifest", "outcome": "protocol_mismatch", "quest_id": "q1",
                                "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "quest", "execution_status": "crashed", "error": "KeyError", "quest_id": "q1",
                                "context": _ctx()})
    found = mem.Index(tmp_path).attempts()
    assert sorted(a.failure for a in found) == ["crashed:KeyError", "oracle_failure", "protocol_mismatch:manifest"]


def test_the_report_scores_recommendations_against_what_followed(tmp_path: Path) -> None:
    fi = tmp_path / "q1" / ".fi"

    def rec(action: str, at: float, decision: str = "execute") -> None:
        ar.append(fi, mem.SHADOW, {"kind": "shadow", "quest_id": "q1", "decision": decision, "action": action,
                                   "model_family": "openai/m1", "at": at})

    def run(outcome: str, at: float) -> None:
        ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": outcome, "quest_id": "q1", "at": at})

    rec("BLOCK", 1.0)
    run("process_error", 2.0)   # the warning was followed by a failure: right
    rec("VERIFY", 3.0)
    run("inconclusive", 4.0)    # followed by no failure: wrong
    rec("IGNORE", 5.0)
    run("oracle_failure", 6.0)  # nothing warned, a failure followed: wrong
    rec("INFO", 7.0)            # nothing followed yet: open
    result = mem.score(tmp_path)
    assert result.counts["BLOCK"] == {"right": 1, "wrong": 0, "open": 0}
    assert result.counts["VERIFY"] == {"right": 0, "wrong": 1, "open": 0}
    assert result.counts["IGNORE"] == {"right": 0, "wrong": 1, "open": 0}
    assert result.counts["INFO"] == {"right": 0, "wrong": 0, "open": 1}
    lines = mem.report_lines(result)
    assert any("BLOCK: 1" in line for line in lines) and any("openai/m1" in line for line in lines)
    assert "changed nothing" in lines[0] or "none changed" in lines[0]


def _bare_engine(tmp_path: Path):
    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.quest_root = tmp_path / "outputs" / "q1"
    eng.fi_dir = eng.quest_root / ".fi"
    eng.fi_dir.mkdir(parents=True)
    eng.quest_id = "q1"
    eng.config = SimpleNamespace(provider=SimpleNamespace(name="openai", model="m1"), engine=None, execution=None,
                                 topic="t")
    eng._prompts = {}
    eng._log = SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None)
    return eng


@pytest.mark.asyncio
async def test_the_hook_records_and_leaves_the_state_alone(tmp_path: Path) -> None:
    eng = _bare_engine(tmp_path)
    state = {"topic": "t", "design": {"hypothesis": "h"}, "iteration": 0}
    before = copy.deepcopy(state)
    await eng._shadow("implement", state, taken="implemented the design")
    assert state == before
    (row,) = ar.read(eng.fi_dir, mem.SHADOW)
    assert row["decision"] == "implement" and row["action"] in mem.ACTIONS and row["taken"] == "implemented the design"
    assert not (eng.fi_dir / "audit.jsonl").exists(), "nothing written to the trace"


@pytest.mark.asyncio
async def test_a_failure_in_the_shadow_path_never_touches_the_quest(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _bare_engine(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("index broken")

    monkeypatch.setattr(mem, "recommend", boom)
    await eng._shadow("execute", {"topic": "t"}, taken="ran the experiment")
    assert ar.lost(eng.fi_dir, mem.SHADOW_LOST) == 1
    assert ar.read(eng.fi_dir, mem.SHADOW) == []


def test_the_tool_is_listed() -> None:
    import launch

    assert "shadow-report" in launch._tools_help()
    assert launch._expand_tools_argv(["tools", "shadow-report"]) == ["--shadow-report"]


# --- a real quest: the same with the recording on or off --------------------------------------------------------------

def _routes(fi_dir: Path) -> list[tuple]:
    return [(e.get("kind"), e.get("node"), e.get("chosen")) for e in audit_log.read(fi_dir / "audit.jsonl")
            if e.get("kind") in ("node_completed", "route_decision")]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_quest_runs_the_same_with_the_recording_on_or_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    def cfg(folder: str) -> Config:
        return Config(topic="smoke-test topic for the engine", title="engine-smoke", provider=ProviderConfig(name="openai"),
                      engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True),
                      execution=ExecutionConfig(sandbox="venv", timeout_s=120),
                      knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / folder))

    runs = {}
    for label, on in (("on", True), ("off", False)):
        engine = Engine(cfg(label))
        engine._shadow_enabled = on
        artifacts = await engine.run()
        runs[label] = (engine, artifacts)
    (on_eng, on_art), (off_eng, off_art) = runs["on"], runs["off"]
    assert _routes(on_eng.fi_dir) == _routes(off_eng.fi_dir), "the same steps and the same routes"
    volatile = {"quest_id", "quest_root", "started_at", "paper_md", "figures_dir", "bundle_manifest"}
    on_state = {k: v for k, v in (on_art.raw_state or {}).items() if k not in volatile}
    off_state = {k: v for k, v in (off_art.raw_state or {}).items() if k not in volatile}
    assert sorted(on_state) == sorted(off_state)
    for key in ("design", "review", "result_json", "iteration", "exec_reflect_iter"):
        assert json.dumps(on_state.get(key), sort_keys=True, default=str) == \
            json.dumps(off_state.get(key), sort_keys=True, default=str), key
    shadow = ar.read(on_eng.fi_dir, mem.SHADOW)
    assert {r["decision"] for r in shadow} >= {"implement", "execute"}
    assert not (off_eng.fi_dir / mem.SHADOW).exists()
