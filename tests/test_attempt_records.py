"""What a quest tried and under which conditions (core/attempt_records.py): the explore/confirm work's first phase.

Records only: the quest writes them as it runs and nothing that decides a route reads them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import attempt_records as ar
from core.config import Config
from core.engine import Engine
from tests.test_engine_smoke import _fake_response_for, smoke_config  # noqa: F401 -- the fixture


def test_a_run_s_outcome() -> None:
    assert ar.run_outcome(returncode=1, has_result=False) == "process_error"
    assert ar.run_outcome(returncode=0, has_result=False) == "process_error"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="repairing") == "protocol_mismatch"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="warned") == "protocol_mismatch"
    # The oracle gate writes ok / warned / stopped; a failed check that only warned is still a failed check.
    assert ar.run_outcome(returncode=0, has_result=True, oracle_status="warned") == "oracle_failure"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="ok") == "inconclusive"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="pending") is None, "a job still running"


def test_a_quest_s_outcome_is_accepted_only_when_a_reviewer_accepted_it() -> None:
    ran = {"result_json": {"x": 1}}
    executed = {"status": "executed"}
    assert ar.quest_outcome({}, {"status": "not_executed"}, reviewer_accepted=True) == "process_error"
    assert ar.quest_outcome(ran, executed, reviewer_accepted=True) == "accepted"
    assert ar.quest_outcome(ran, executed, reviewer_accepted=False) == "inconclusive"
    # No direction is guessed from the analysis's prose: the real analysis has no field that states it.
    real_analysis = {"summary": "The hypothesis is not supported; no significant difference.", "key_findings": [],
                     "claims_supported": [], "claims_unsupported": ["h"], "next_step": "write"}
    assert ar.quest_outcome({**ran, "analysis": real_analysis}, executed, reviewer_accepted=True) == "accepted"
    assert set(ar.STOP_OUTCOMES.values()) <= set(ar.OUTCOMES)


def test_the_context_names_what_a_failure_depends_on(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "ENVIRONMENT.json").write_text('{"packages": ["a==1"]}', encoding="utf-8")
    state = {"design": {"protocol": {"runs_per_setting": 30, "metrics": [{"id": "x"}]}}, "iteration": 1}
    ctx = ar.context_fingerprint(cfg, tmp_path, state, prompts={"design": "prompt text"})
    assert ctx["model"] == "m1" and ctx["provider"] == "openai"
    assert ctx["skills"] == [] and ctx["fi"] is None, "no fi_repo given: unknown, not an empty string"
    assert ctx["environment_sha256"] and ctx["metric_specs_sha256"] and ctx["budget"]["runs_per_setting"] == 30
    assert ctx["lineage"]["iteration"] == 1
    other = ar.context_fingerprint(cfg.model_copy(update={"provider": cfg.provider.model_copy(update={"model": "m2"})}),
                                   tmp_path, state, prompts={"design": "prompt text"})
    assert other["model"] != ctx["model"], "a different model is a different context"
    assert ar.context_fingerprint(cfg, tmp_path, state, prompts={"design": "changed"})["prompts_sha256"] != ctx["prompts_sha256"]


def test_two_quests_with_the_same_packages_and_protocol_match(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    protocol = {"runs_per_setting": 30, "metrics": [{"id": "x"}]}
    ctxs = []
    for name, venv in (("a", "/q/a/.venv/python"), ("b", "/q/b/.venv/python")):
        root = tmp_path / name
        (root / "needs").mkdir(parents=True)
        (root / "needs" / "ENVIRONMENT.json").write_text(json.dumps(
            {"python": "3.11", "executable": venv, "recorded": name, "packages": ["a==1"]}), encoding="utf-8")
        (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(
            {"protocol": protocol, "approved_at": name, "run_id": name}), encoding="utf-8")
        ctxs.append(ar.context_fingerprint(cfg, root, {}, prompts={}))
    assert ctxs[0]["environment_sha256"] == ctxs[1]["environment_sha256"]
    assert ctxs[0]["protocol_sha256"] == ctxs[1]["protocol_sha256"]
    # Before the freeze the same protocol gives the same hash.
    before = ar.context_fingerprint(cfg, tmp_path / "none", {"design": {"protocol": protocol}}, prompts={})
    assert before["protocol_sha256"] == ctxs[0]["protocol_sha256"]


def test_a_large_input_is_not_read_whole(tmp_path: Path) -> None:
    big = tmp_path / "big.bin"
    with big.open("wb") as fh:
        fh.seek(ar._WHOLE_FILE_LIMIT + 10)
        fh.write(b"x")
    first = ar._file_sha(big)
    assert first and first == ar._file_sha(big)
    small = tmp_path / "small.txt"
    small.write_text("abc", encoding="utf-8")
    import hashlib
    assert ar._file_sha(small) == hashlib.sha256(b"abc").hexdigest()


def test_records_append_and_a_torn_line_is_skipped(tmp_path: Path) -> None:
    assert ar.append(tmp_path, ar.ATTEMPTS, {"kind": "run", "outcome": "inconclusive"})
    with (tmp_path / ar.ATTEMPTS).open("a", encoding="utf-8") as fh:
        fh.write('{"torn": ')
    assert [r["kind"] for r in ar.read(tmp_path, ar.ATTEMPTS)] == ["run"]


@pytest.mark.asyncio
async def test_a_quest_records_its_ideas_design_runs_and_end(smoke_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(smoke_config)
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    ledger = ar.read(engine.fi_dir, ar.LEDGER)
    kinds = [r["kind"] for r in ledger]
    assert "ideas" in kinds and "design" in kinds, kinds
    ideas = next(r for r in ledger if r["kind"] == "ideas")
    assert ideas["candidates"] and ideas["chosen"] and ideas["rule"] in ("ensemble", "tournament", "reflection", "model")
    design = next(r for r in ledger if r["kind"] == "design")
    assert design["revision"] == 0 and design["parent"] is None, "numbered as DESIGN_HISTORY.json numbers it"
    attempts = ar.read(engine.fi_dir, ar.ATTEMPTS)
    runs = [r for r in attempts if r["kind"] == "run"]
    (end,) = [r for r in attempts if r["kind"] == "quest"]
    assert runs and all(r["outcome"] in ar.OUTCOMES for r in runs)
    assert end["outcome"] in ar.OUTCOMES and end["context"]["provider"] == "openai"
    assert "answered_by" in end and "person" in end


def test_a_record_that_cannot_be_built_never_touches_the_quest(tmp_path: Path) -> None:
    engine = Engine(Config.model_validate({
        "topic": "t", "provider": {"name": "openai", "model": "m"}, "knowledge": {"enabled": False},
        "output": {"output_dir": str(tmp_path / "out")},
    }))
    engine._record(ar.LEDGER, lambda: {"x": 1 / 0})
    engine._record(ar.LEDGER, lambda: None)
    assert ar.read(engine.fi_dir, ar.LEDGER) == []


def test_a_stop_for_a_failed_check_is_recorded_and_other_stops_are_not(tmp_path: Path) -> None:
    engine = Engine(Config.model_validate({
        "topic": "t", "provider": {"name": "openai", "model": "m"}, "knowledge": {"enabled": False},
        "output": {"output_dir": str(tmp_path / "out")},
    }))
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    (engine.fi_dir / "pause.json").write_text(json.dumps({"kind": "plan"}), encoding="utf-8")
    engine._record_stop({}, {})
    assert ar.read(engine.fi_dir, ar.ATTEMPTS) == []
    (engine.fi_dir / "pause.json").write_text(json.dumps({"kind": "oracle"}), encoding="utf-8")
    engine._record_stop({}, {})
    (stop,) = ar.read(engine.fi_dir, ar.ATTEMPTS)
    assert stop["kind"] == "stop" and stop["outcome"] == "oracle_failure"
