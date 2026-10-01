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


def test_a_quest_s_status_keeps_execution_review_evidence_and_claim_apart() -> None:
    ran = {"result_json": {"x": 1}, "review": {"verdict": "accept", "status": "ok"}}
    ready = {"status": "publication_ready"}
    s = ar.quest_status(ran, ready, reviewer_accepted=True, no_experiment=False)
    assert s == {"execution_status": "completed", "review_status": "accepted", "evidence_status": "publication_ready",
                 "claim_outcome": None}
    # A survey runs no experiment by design: accepted, not a process error.
    survey = ar.quest_status({"review": {"verdict": "accept"}}, {"status": "executed"}, reviewer_accepted=True,
                             no_experiment=True)
    assert survey["execution_status"] == "no_experiment_by_design" and survey["review_status"] == "accepted"
    # An accept with evidence gaps is an accepted review of a result that is not publication-ready.
    gaps = ar.quest_status(ran, {"status": "protocol_runtime_matched"}, reviewer_accepted=True, no_experiment=False)
    assert gaps["review_status"] == "accepted" and gaps["evidence_status"] == "protocol_runtime_matched"
    assert ar.quest_status({}, None, reviewer_accepted=False, no_experiment=False)["execution_status"] == "no_result"
    unreal = ar.quest_status({**ran, "review": {"verdict": "accept", "status": "error"}}, ready,
                             reviewer_accepted=False, no_experiment=False)
    assert unreal["review_status"] == "unavailable"
    assert ar.quest_status({**ran, "review": {"verdict": "revise"}}, ready, reviewer_accepted=False,
                           no_experiment=False)["review_status"] == "revise"
    # No direction is guessed from the analysis's prose: the real analysis has no field that states it.
    assert set(ar.STOP_OUTCOMES.values()) <= set(ar.OUTCOMES)


def test_the_context_names_what_a_failure_depends_on(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "ENVIRONMENT.json").write_text('{"packages": ["a==1"]}', encoding="utf-8")
    state = {"design": {"protocol": {"runs_per_setting": 30, "metrics": [{"id": "x"}]}}, "iteration": 1}
    ctx = ar.context_fingerprint(cfg, tmp_path, state, kind="in_progress", prompts={"design": "prompt text"})
    assert ctx["model"] == "m1" and ctx["provider"] == "openai"
    assert ctx["skills"] == [] and ctx["fi"] is None, "no fi_repo given: unknown, not an empty string"
    assert ctx["environment_sha256"] and ctx["metric_specs_sha256"] and ctx["budget"]["runs_per_setting"] == 30
    assert ctx["lineage"]["iteration"] == 1
    assert ctx["policy"]["result_use"] == "explore" and ctx["question"]["topic_sha256"]
    assert ctx["complete"] is False, "no FI version known: never treated as the same conditions"
    other = ar.context_fingerprint(cfg.model_copy(update={"provider": cfg.provider.model_copy(update={"model": "m2"})}),
                                   tmp_path, state, kind="in_progress", prompts={"design": "prompt text"})
    assert other["model"] != ctx["model"], "a different model is a different context"
    assert ar.context_fingerprint(cfg, tmp_path, state, kind="in_progress", prompts={"design": "changed"})["prompts_sha256"] != ctx["prompts_sha256"]


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
        ctxs.append(ar.context_fingerprint(cfg, root, {}, kind="in_progress", prompts={}))
    assert ctxs[0]["environment_sha256"] == ctxs[1]["environment_sha256"]
    assert ctxs[0]["protocol_sha256"] == ctxs[1]["protocol_sha256"]
    # Before the freeze the same protocol gives the same hash.
    before = ar.context_fingerprint(cfg, tmp_path / "none", {"design": {"protocol": protocol}}, kind="in_progress", prompts={})
    assert before["protocol_sha256"] == ctxs[0]["protocol_sha256"]


def test_a_changed_program_or_a_late_input_file_changes_the_context(tmp_path: Path) -> None:
    """The delta re-audit's counterexamples: an edited experiment.py kept the context, and so did input file 201."""
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text("print(1)", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for i in range(201):
        (inputs / f"f{i:03d}.txt").write_text(str(i), encoding="utf-8")
    before = ar.context_fingerprint(cfg, tmp_path, {}, kind="in_progress", prompts={})
    (tmp_path / "code" / "experiment.py").write_text("print(2)", encoding="utf-8")
    after_code = ar.context_fingerprint(cfg, tmp_path, {}, kind="in_progress", prompts={})
    assert after_code["code"] != before["code"]
    (inputs / "f200.txt").write_text("changed", encoding="utf-8")
    after_input = ar.context_fingerprint(cfg, tmp_path, {}, kind="in_progress", prompts={})
    assert after_input["inputs"]["sha256"] != after_code["inputs"]["sha256"]
    assert after_input["inputs"]["files"] == 201 and after_input["inputs"]["complete"] is True
    # A different question or policy is a different context.
    assert ar.context_fingerprint(cfg.model_copy(update={"topic": "other"}), tmp_path, {}, kind="in_progress", prompts={})["question"] != \
        before["question"]
    assert ar.context_fingerprint(cfg.model_copy(update={"result_use": "research"}), tmp_path, {}, kind="in_progress", prompts={})[
        "policy"] != before["policy"]


def test_too_many_inputs_are_marked_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar, "_INPUT_FILES_LIMIT", 3)
    for i in range(5):
        (tmp_path / f"{i}.txt").write_text(str(i), encoding="utf-8")
    m = ar._folder_manifest(tmp_path, limit=3)
    assert m["files"] == 5 and m["complete"] is False


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
    rid = ar.append(tmp_path, ar.ATTEMPTS, {"kind": "run", "outcome": "inconclusive"})
    assert rid and ar.read(tmp_path, ar.ATTEMPTS)[0]["record_id"] == rid
    assert ar.read(tmp_path, ar.ATTEMPTS)[0]["schema"] == ar.SCHEMA
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
    assert end["execution_status"] in ar.EXECUTION and end["review_status"] in ar.REVIEW
    assert end["context"]["provider"] == "openai" and end["quest_id"] == engine.quest_id
    assert end["records_not_written"] == 0 and end["context"]["models_used"], "which model answered each step"
    assert all(r["scripts"] for r in runs), "each run names the scripts it ran by hash"
    assert all("parent_id" in r and "run_id" in r for r in runs)
    design_ids = {r["record_id"] for r in ledger if r["kind"] in ("design", "repair")}
    assert runs[0]["parent_id"] in design_ids, "the run names the design revision it ran"
    from core import audit_log
    last = audit_log.read(engine.audit.path)[-1]
    assert last["kind"] == "quest_finalized", "the seal is the trace's last event"
    assert last["files"][".fi/attempts.jsonl"] == ar._sha((engine.fi_dir / ar.ATTEMPTS).read_bytes()), "anchored"
    assert end["context"]["context_kind"] == "quest_end"
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


@pytest.mark.asyncio
async def test_a_failed_quest_is_recorded_once_and_a_finished_one_is_not_recorded_again(
        smoke_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    # A failure after the quest's line was written (in the clean-up) does not add a second one.
    def boom(self, *a, **k):  # noqa: ANN001
        raise RuntimeError("clean-up failed")

    monkeypatch.setattr(Engine, "_write_cost_summary", boom)
    engine = Engine(smoke_config)
    with pytest.raises(RuntimeError):
        await engine.run()
    ends = [r for r in ar.read(engine.fi_dir, ar.ATTEMPTS) if r["kind"] == "quest"]
    assert len(ends) == 1 and ends[0]["execution_status"] != "crashed"


@pytest.mark.asyncio
async def test_a_quest_that_fails_is_recorded_as_a_process_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        raise RuntimeError("the provider is gone")

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(Config.model_validate({
        "topic": "t", "provider": {"name": "openai", "model": "m"}, "knowledge": {"enabled": False},
        "engine": {"clarify_mode": "off"}, "output": {"output_dir": str(tmp_path / "out")},
    }))
    with pytest.raises(Exception):
        await engine.run()
    ends = [r for r in ar.read(engine.fi_dir, ar.ATTEMPTS) if r["kind"] == "quest"]
    assert len(ends) == 1 and ends[0]["execution_status"] == "crashed" and ends[0]["error"]
    assert ends[0]["context"] and ends[0]["context"]["question"]["topic_sha256"], "a context even for an early failure"


def test_a_finished_quest_without_a_result_is_not_called_crashed() -> None:
    s = ar.quest_status({"review": {}}, None, reviewer_accepted=False, no_experiment=False)
    assert s["execution_status"] == "no_result"
    assert ar.quest_status({}, None, reviewer_accepted=False, no_experiment=False,
                           data_analysis=True)["execution_status"] == "data_analysis"


def test_the_code_folder_is_hashed_whole_and_fi_is_known_by_its_source(tmp_path: Path) -> None:
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "run.sh").write_text("python experiment.py", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "helper.py").write_text("X = 1", encoding="utf-8")
    assert set(ar.script_hashes(tmp_path)) == {"run.sh", "pkg/helper.py"}
    import shutil
    repo = Path(__file__).resolve().parent.parent
    copy = tmp_path / "installed"
    shutil.copytree(repo / "core", copy / "core", ignore=shutil.ignore_patterns("__pycache__"))
    fi = ar._fi_version(copy)
    assert fi and fi["source_sha256"] and fi["commit"] is None, "an installed copy (no git) is known by its source"


def test_what_keeps_a_context_from_being_complete_is_named(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    repo = Path(__file__).resolve().parent.parent
    ctx = ar.context_fingerprint(cfg, tmp_path, {}, prompts={}, fi_repo=repo, kind="after_run")
    assert not ctx["complete"]
    assert "no model call recorded yet" in ctx["missing"] and "no script in code/" in ctx["missing"]
    assert ctx["policy"]["config_sha256"]
    # A run resumed only to take a decision makes no call itself; a model answered the quest's earlier runs.
    (tmp_path / ".fi").mkdir(exist_ok=True)
    calls = tmp_path / ".fi" / ar.MODEL_CALLS
    counted = {"model_call_counts": {"write": 2}}
    calls.write_text(json.dumps({"node": "write", "outcome": "error"}) + "\n", encoding="utf-8")
    failed = ar.context_fingerprint(cfg, tmp_path, counted, prompts={}, fi_repo=repo, kind="after_run")
    assert "no model call recorded yet" in failed["missing"], "calls that all failed: no model has answered"
    calls.write_text(json.dumps({"node": "write", "outcome": "ok"}) + "\n", encoding="utf-8")
    resumed = ar.context_fingerprint(cfg, tmp_path, counted, prompts={}, fi_repo=repo, kind="after_run")
    assert "no model call recorded yet" not in resumed["missing"]
    part = ar.context_fingerprint(cfg, tmp_path, {}, kind="in_progress", prompts={}, fi_repo=repo, partial=True,
                                  models_used={"design": {"provider": "p", "model": "m"}})
    assert any("without the quest's state" in m for m in part["missing"])


def test_lost_records_are_counted_across_runs(tmp_path: Path) -> None:
    ar.count_lost(tmp_path)
    ar.count_lost(tmp_path)
    assert ar.lost(tmp_path) == 2


@pytest.mark.asyncio
async def test_a_late_crash_keeps_the_context_last_worked_out(
        smoke_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if "Writing" in prompt or "write the paper" in prompt.lower():
            raise RuntimeError("the provider went away while writing")
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(smoke_config)
    with pytest.raises(Exception):
        await engine.run()
    (end,) = [r for r in ar.read(engine.fi_dir, ar.ATTEMPTS) if r["kind"] == "quest"]
    assert end["execution_status"] == "crashed"
    assert not any("without the quest's state" in m for m in end["context"]["missing"]), "not a fresh, empty context"
    assert end["context"]["lineage"]["iteration"] >= 0 and end["context"]["question"]["design_sha256"]
