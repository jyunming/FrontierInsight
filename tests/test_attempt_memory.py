"""Shadow recommendations from past attempts (core/attempt_memory.py): recorded at decisions, never acted on.

Each decision is compared only on what it depends on; BLOCK only before an experiment runs, for the same key that failed
with the same non-transient signature at least twice; another model is at most VERIFY; an unknown part or an older
record gives INFO at most; a repair is never compared with the run that triggered it. A quest runs exactly the same
with the recording on or off, a late or failing recommendation is never written, and outcomes are linked by lineage.
"""

from __future__ import annotations

import asyncio
import copy
import json
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import attempt_memory as mem
from core import attempt_records as ar
from core import audit_log


def _ctx(**over) -> dict:
    ctx = {
        "provider": "openai", "model": "m1",
        "prompt_shas": {"design": "pd", "implement": "pi", "implement_body": "pb"},
        "skills": [], "environment_sha256": "env1", "dependency_lock_sha256": None,
        "inputs": None, "data": None, "code": {"simulate.py": "c1"}, "deps_sha256": "deps1", "harness_sha256": "h1",
        "question": {"topic_sha256": "t1", "design_sha256": "d1"},
        "policy": {"rigor_profile": "research", "review_panel": ["methodologist"], "result_use": "research"},
        "protocol_sha256": "pr1", "metric_specs_sha256": "ms1",
        "design_features": {"metrics": ["m"], "grid_axes": ["R0"]},
        "budget": {"runs_per_setting": 10, "timeout_s": 60, "execute_replicates": 3, "replicate_seed_stride": 1000},
        "models_used": {"plan": {"provider": "openai", "model": "m1", "reported": True},
                        "implement": {"provider": "openai", "model": "m1", "reported": True}},
        "lineage": {"design_revision": 0}, "missing": [], "complete": True,
    }
    ctx.update(over)
    return ctx


SIG = {"returncode": 1, "exception": "NameError", "pause": None, "transient": False}


def _failed(record_id: str, *, quest_id: str = "q0", schema: int = ar.SCHEMA, sig: dict | None = None, **ctx) -> mem.Attempt:
    record = {"kind": "run", "outcome": "process_error", "record_id": record_id, "quest_id": quest_id,
              "schema": schema, "failure_signature": sig or SIG, "context": _ctx(**ctx)}
    attempt = mem._as_attempt(record)
    assert attempt is not None
    return attempt


def _cand(decision: str, sig: dict | None = None, **ctx) -> dict:
    return mem.parts(_ctx(**ctx), decision, signature=sig)


# --- signatures ---------------------------------------------------------------------------------------------------

def test_the_failure_signature_names_the_exception_and_what_is_transient() -> None:
    tb = "Traceback (most recent call last):\n  File \"x.py\", line 1\nNameError: name 'x' is not defined\n"
    assert mem.failure_signature(returncode=1, stderr=tb) == {"returncode": 1, "exception": "NameError", "pause": None,
                                                              "transient": False}
    assert mem.failure_signature(returncode=1, timed_out=True)["transient"]
    assert mem.failure_signature(returncode=-9)["transient"], "killed (out of memory)"
    assert mem.failure_signature(returncode=1, stderr="MemoryError\n")["transient"]
    assert mem.failure_signature(error="ReadTimeout")["transient"]
    assert mem.failure_signature(error="RemoteProtocolError")["transient"]
    assert mem.failure_signature(pause="oracle")["pause"] == "oracle"


# --- matching ------------------------------------------------------------------------------------------------------

def test_block_needs_the_same_code_to_have_failed_the_same_way_twice() -> None:
    assert mem.recommend(_cand("execute"), [_failed("a")], decision="execute").action == "VERIFY"
    rec = mem.recommend(_cand("execute"), [_failed("a"), _failed("b")], decision="execute")
    assert rec.action == "BLOCK" and set(rec.matched_attempt_ids) == {"a", "b"}
    other_sig = {**SIG, "exception": "KeyError"}
    assert mem.recommend(_cand("execute"), [_failed("a"), _failed("b", sig=other_sig)],
                         decision="execute").action == "VERIFY", "two different failures are not one hazard"


def test_a_timeout_or_a_connection_failure_never_blocks() -> None:
    timeout = {**SIG, "exception": None, "transient": True}
    rec = mem.recommend(_cand("execute"), [_failed("a", sig=timeout), _failed("b", sig=timeout)], decision="execute")
    assert rec.action == "VERIFY" and "not taken as reproducible" in rec.reason


def test_the_same_code_is_the_same_test_whoever_wrote_it() -> None:
    other_writer = dict(model="m9", prompt_shas={"design": "x", "implement": "y"},
                        models_used={"implement": {"provider": "other", "model": "m9", "reported": True}})
    rec = mem.recommend(_cand("execute"), [_failed("a", **other_writer), _failed("b", **other_writer)],
                        decision="execute")
    assert rec.action == "BLOCK"


def test_an_environment_only_one_side_knows_is_not_compared() -> None:
    rec = mem.recommend(_cand("execute", environment_sha256=None), [_failed("a"), _failed("b")], decision="execute")
    assert rec.action == "BLOCK"
    assert mem.recommend(_cand("execute", environment_sha256="env2"), [_failed("a"), _failed("b")],
                         decision="execute").action == "INFO", "both known and different: similar only"


def test_plan_implement_and_repair_are_at_most_a_check() -> None:
    twice = [_failed("a"), _failed("b")]
    assert mem.recommend(_cand("plan"), twice, decision="plan").action == "VERIFY"
    assert mem.recommend(_cand("implement"), twice, decision="implement").action == "VERIFY"
    assert mem.recommend(_cand("repair", sig=SIG), [_failed("a", quest_id="qX"), _failed("b", quest_id="qY")],
                         decision="repair").action == "VERIFY"


def test_another_model_is_at_most_a_check_and_other_design_features_a_note() -> None:
    other = dict(models_used={"implement": {"provider": "openai", "model": "m2", "reported": True},
                              "plan": {"provider": "openai", "model": "m2", "reported": True}})
    rec = mem.recommend(_cand("implement"), [_failed("a", **other)], decision="implement")
    assert rec.action == "VERIFY" and "another model" in rec.reason
    feat = dict(design_features={"metrics": ["other"], "grid_axes": ["N"]})
    assert mem.recommend(_cand("plan"), [_failed("a", **feat)], decision="plan").action == "INFO"


def test_an_unknown_part_or_an_older_record_is_a_note_at_most() -> None:
    assert mem.recommend(_cand("execute"), [_failed("a", harness_sha256=None), _failed("b", harness_sha256=None)],
                         decision="execute").action == "INFO"
    assert mem.recommend(_cand("execute"), [_failed("a", schema=ar.SCHEMA - 1), _failed("b", schema=ar.SCHEMA - 1)],
                         decision="execute").action == "INFO"
    assert mem.recommend(_cand("execute", code={}), [_failed("a"), _failed("b")],
                         decision="execute").action == "INFO", "no code yet: not the same test"
    assert mem.recommend(_cand("execute"), [], decision="execute").action == "IGNORE"


def test_a_repair_is_never_compared_with_the_run_that_triggered_it() -> None:
    own = [_failed("trigger", quest_id="q1"), _failed("earlier", quest_id="q1")]
    rec = mem.recommend(_cand("repair", sig=SIG), own, decision="repair", exclude_ids={"trigger"},
                        exclude_lineage=("q1", 0))
    assert rec.action == "IGNORE" and rec.matched_attempt_ids == []
    other = [_failed("x", quest_id="q2")]
    assert mem.recommend(_cand("repair", sig=SIG), other, decision="repair",
                         exclude_lineage=("q1", 0)).action == "VERIFY"
    assert mem.recommend(_cand("repair", sig={**SIG, "exception": "KeyError"}), other, decision="repair",
                         exclude_lineage=("q1", 0)).action == "INFO", "another failure: similar code only"


def test_only_failures_are_indexed_and_the_index_is_shared(tmp_path: Path) -> None:
    fi = tmp_path / "q1" / ".fi"
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "inconclusive", "quest_id": "q1", "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "run", "outcome": "oracle_failure", "quest_id": "q1", "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "stop", "pause": "manifest", "outcome": "protocol_mismatch", "quest_id": "q1",
                                "context": _ctx()})
    ar.append(fi, ar.ATTEMPTS, {"kind": "quest", "execution_status": "crashed", "error": "KeyError", "quest_id": "q1",
                                "context": _ctx()})
    index = mem.index_for(tmp_path)
    assert index is mem.index_for(tmp_path), "one per output root in a process"
    sigs = sorted(json.dumps(a.signature, sort_keys=True) for a in index.attempts())
    assert len(sigs) == 3 and any('"pause": "manifest"' in s for s in sigs) and any("KeyError" in s for s in sigs)


# --- lineage and scoring ---------------------------------------------------------------------------------------------

def test_recommendations_are_linked_to_the_records_they_are_about(tmp_path: Path) -> None:
    fi = tmp_path
    mem.stamp(fi, "plan", "p1")
    mem.stamp(fi, "implement", "i1")
    mem.stamp(fi, "execute", "e1")
    assert sorted(mem.take(fi, "run")) == ["e1", "i1", "p1"]
    mem.stamp(fi, "repair", "r1")
    assert sorted(mem.take(fi, "run")) == ["p1", "r1"], "the plan's stays until the quest ends"
    assert mem.take(fi, "quest") == ["p1"]
    assert mem.take(fi, "run") == []


def test_the_report_scores_by_decision_with_a_base_rate(tmp_path: Path) -> None:
    fi = tmp_path / "q1" / ".fi"

    def rec(rid: str, action: str, decision: str) -> None:
        ar.append(fi, mem.SHADOW, {"kind": "shadow", "record_id": rid, "decision": decision, "action": action,
                                   "model_family": "openai/m1"})

    def attempt(kind: str, parents: list[str], **fields) -> None:
        ar.append(fi, ar.ATTEMPTS, {"kind": kind, "quest_id": "q1", "parent_shadow_ids": parents, **fields})

    rec("b1", "BLOCK", "execute")
    attempt("run", ["b1"], outcome="process_error")          # warned, failed
    rec("i1", "IGNORE", "execute")
    attempt("run", ["i1"], outcome="inconclusive")           # quiet, no failure
    rec("i2", "IGNORE", "execute")
    attempt("run", ["i2"], outcome="oracle_failure")          # quiet, failed: the base rate
    rec("p1", "VERIFY", "plan")
    attempt("quest", ["p1"], execution_status="no_result", review_status="none")  # plan failed: no usable result
    rec("x1", "INFO", "repair")                              # nothing followed yet
    result = mem.score(tmp_path)
    assert result.counts["execute"]["BLOCK"] == {"failed": 1, "ok": 0, "open": 0}
    assert result.counts["execute"]["IGNORE"] == {"failed": 1, "ok": 1, "open": 0}
    assert result.counts["plan"]["VERIFY"] == {"failed": 1, "ok": 0, "open": 0}
    assert result.counts["repair"]["INFO"] == {"failed": 0, "ok": 0, "open": 1}
    lines = mem.report_lines(result)
    assert lines[0].endswith("no route or state was changed.")
    assert any("base rate" in line and "1 of 2" in line for line in lines)


# --- the engine hook --------------------------------------------------------------------------------------------------

def _bare_engine(tmp_path: Path, mode: str = "shadow"):
    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.quest_root = tmp_path / "outputs" / "q1"
    eng.fi_dir = eng.quest_root / ".fi"
    eng.fi_dir.mkdir(parents=True)
    eng.quest_id = "q1"
    eng.config = SimpleNamespace(provider=SimpleNamespace(name="openai", model="m1", node_models={}),
                                 engine=SimpleNamespace(attempt_memory=mode), execution=None, topic="t")
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
    assert set(row["parts"]) >= set(mem.KEYS["implement"]) and "context_missing" in row
    assert not (eng.fi_dir / "audit.jsonl").exists(), "nothing written to the trace"
    assert mem.take(eng.fi_dir, "run") == [row["record_id"]], "stamped for the run it is about"
    eng._shadow_close()


@pytest.mark.asyncio
async def test_off_reads_and_writes_nothing(tmp_path: Path) -> None:
    eng = _bare_engine(tmp_path, mode="off")
    await eng._shadow("execute", {"topic": "t"}, taken="ran the experiment")
    assert not (eng.fi_dir / mem.SHADOW).exists() and not (eng.fi_dir / mem.SHADOW_PENDING).exists()


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
    eng._shadow_close()


@pytest.mark.asyncio
async def test_a_late_recommendation_is_dropped_never_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _bare_engine(tmp_path)
    real = mem.recommend

    def slow(*a, **k):
        time.sleep(0.6)
        return real(*a, **k)

    monkeypatch.setattr(mem, "DEADLINE_S", 0.1)
    monkeypatch.setattr(mem, "recommend", slow)
    started = time.monotonic()
    await eng._shadow("execute", {"topic": "t"}, taken="ran the experiment")
    assert time.monotonic() - started < 0.5, "the quest is not held past the deadline"
    await asyncio.sleep(1.0)  # the worker finishes later...
    assert ar.read(eng.fi_dir, mem.SHADOW) == [], "...and writes nothing"
    assert ar.lost(eng.fi_dir, mem.SHADOW_LOST) == 1
    eng._shadow_close()


def test_the_tool_is_listed() -> None:
    import launch

    assert "shadow-report" in launch._tools_help()
    assert launch._expand_tools_argv(["tools", "shadow-report"]) == ["--shadow-report"]


# --- real quests (slow tier: CI runs @slow tests on every code PR) ----------------------------------------------------

_BAD = textwrap.dedent("""\
    import sys
    print(undefined_name)
""")


def _cfg(root: Path, *, repairs: int = 0, mode: str = "shadow"):
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    return Config(topic="smoke-test topic for the engine", title="engine-smoke", provider=ProviderConfig(name="openai"),
                  engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True,
                                      exec_reflect_max_iterations=repairs, attempt_memory=mode),
                  execution=ExecutionConfig(sandbox="venv", timeout_s=120),
                  knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=root))


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_script_that_fails_the_same_way_twice_is_a_block_at_the_third_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
        if "Implementation" in head or "Execute-Reflect" in head:
            return json.dumps({"code": _BAD, "deps": [], "patch_summary": "same", "give_up_reason": ""})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path / "out", repairs=2))
    await engine.run()
    runs = [r for r in ar.read(engine.fi_dir, ar.ATTEMPTS) if r.get("kind") == "run"]
    assert len(runs) >= 3 and all(r["failure_signature"]["exception"] == "NameError" for r in runs), runs
    execute = [r for r in ar.read(engine.fi_dir, mem.SHADOW) if r["decision"] == "execute"]
    assert [r["action"] for r in execute][:3] == ["IGNORE", "VERIFY", "BLOCK"], [r["action"] for r in execute]
    repair = [r for r in ar.read(engine.fi_dir, mem.SHADOW) if r["decision"] == "repair"]
    revision = {r["record_id"]: r["context"]["lineage"]["design_revision"] for r in runs}
    assert repair
    for r in repair:
        (trigger,) = r["excluded_attempt_ids"]
        assert trigger in revision and trigger not in r["matched_attempt_ids"], "never the run that triggered it"
        assert all(revision.get(m) != revision[trigger] for m in r["matched_attempt_ids"]),             "never a run of the same design revision (its own repair lineage)"
    # Each execute recommendation is linked to the run it led to.
    for rec in execute[:3]:
        assert any(rec["record_id"] in (r.get("parent_shadow_ids") or []) for r in runs)


#: The only fields two runs of the same quest may differ in: when something was recorded and how long it took.
_TIMING = ("recorded_at", "duration_s")


def _without_timing(value):
    if isinstance(value, dict):
        return {k: _without_timing(v) for k, v in value.items() if k not in _TIMING}
    if isinstance(value, list):
        return [_without_timing(v) for v in value]
    return value


def _normalized_state(artifacts, quest_root: Path) -> dict:
    """The final state with what names the quest folder replaced and the timing fields (:data:`_TIMING`, at any depth)
    left out; nothing else is left out."""
    text = json.dumps(artifacts.raw_state or {}, sort_keys=True, default=str)
    for old in (str(quest_root).replace("\\", "\\\\"), quest_root.as_posix(), str(quest_root), quest_root.name):
        text = text.replace(old, "<quest>")
    return _without_timing(json.loads(text))


@pytest.mark.slow
@pytest.mark.asyncio
async def test_a_quest_runs_the_same_with_the_recording_on_or_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_engine_smoke import _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    runs = {}
    for mode in ("shadow", "off"):
        engine = Engine(_cfg(tmp_path / mode, mode=mode))
        artifacts = await engine.run()
        runs[mode] = (engine, artifacts)

    def routes(fi_dir: Path) -> list[tuple]:
        return [(e.get("kind"), e.get("node"), e.get("chosen")) for e in audit_log.read(fi_dir / "audit.jsonl")
                if e.get("kind") in ("node_completed", "route_decision")]

    (on_eng, on_art), (off_eng, off_art) = runs["shadow"], runs["off"]
    assert routes(on_eng.fi_dir) == routes(off_eng.fi_dir), "the same steps and the same routes"
    on_state = _normalized_state(on_art, on_eng.quest_root)
    off_state = _normalized_state(off_art, off_eng.quest_root)
    differ = sorted(k for k in set(on_state) | set(off_state) if on_state.get(k) != off_state.get(k))
    assert differ == [], differ
    assert ar.read(on_eng.fi_dir, mem.SHADOW)
    assert not ar.read(off_eng.fi_dir, mem.SHADOW), "off: nothing recorded (the seal may leave an empty file)"
