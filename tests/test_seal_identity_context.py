"""The 2026-09-27b re-audit's three trust leaks, each a counterexample that used to pass.

- A reviewer whose connection did not say which model answered is unproven, not independent: an evidence gap.
- The seal is the trace's last event and names the evidence and attempt records; whoever reads the evidence checks it.
- A finished quest's context needs its code, environment and protocol; code is hashed whole; a cluster job's code is
  compared between submission and collection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import attempt_records as ar
from core import audit_log, evidence, trial_runner
from core.config import Config
from core.engine import Engine
from core.provider import _extract_claude_usage
from tests.test_evidence import ON, _quest, _state

RESEARCH = {**ON, "rigor_profile": "research"}


# --- the reviewer's identity -----------------------------------------------------------------------------------------

def _engine(panel_models: dict[str, str], *, one_model_review: bool = False) -> Engine:
    eng = object.__new__(Engine)
    eng.config = SimpleNamespace(
        rigor_profile="research",
        engine=SimpleNamespace(one_model_review=one_model_review, review_panel=list(panel_models)),
    )
    return eng


def _panel(**reported: bool) -> list[dict]:
    return [{"persona": p, "status": "ok", "requested_model": f"m-{i}", "actual_model_reported": r}
            for i, (p, r) in enumerate(reported.items())]


def test_a_reviewer_whose_model_was_not_reported_is_listed() -> None:
    eng = _engine({"methodologist": "a", "statistician": "b"})
    state = {"review_panel": _panel(methodologist=True, statistician=False)}
    assert eng._review_identity_unverified(state) == ["statistician"]
    assert eng._review_identity_unverified({"review_panel": _panel(methodologist=True, statistician=True)}) == []


def test_a_panel_knowingly_on_one_model_is_not_listed_twice() -> None:
    eng = _engine({"methodologist": "a", "statistician": "a"}, one_model_review=True)
    eng._reviewer_model = lambda p: "a"  # every persona on one model
    assert eng._review_identity_unverified({"review_panel": _panel(methodologist=False, statistician=False)}) == []


def test_an_unverified_reviewer_is_a_gap_under_research_only(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    gap = "the review's independence is unverified"
    research = evidence.assess(root, _state(), settings={**RESEARCH, "review_identity_unverified": ["statistician"]})
    assert research["status"] != "publication_ready"
    assert any(gap in g and "statistician" in g for g in research["all_gaps"]["publication_ready"])
    default = evidence.assess(root, _state(), settings={**ON, "review_identity_unverified": ["statistician"]})
    assert not any(gap in g for g in default.get("all_gaps", {}).get("publication_ready", []))


def test_the_claude_cli_says_which_model_answered() -> None:
    envelope = json.dumps({"type": "result", "usage": {"input_tokens": 1, "output_tokens": 5},
                           "modelUsage": {"claude-haiku-4-5": {"outputTokens": 2},
                                          "claude-opus-4-7": {"outputTokens": 400}}})
    assert _extract_claude_usage(envelope)["served_model"] == "claude-opus-4-7"
    no_models = json.dumps({"type": "result", "usage": {"input_tokens": 1, "output_tokens": 5}})
    assert "served_model" not in _extract_claude_usage(no_models)


# --- the seal ---------------------------------------------------------------------------------------------------------

def _finished(tmp_path: Path, **seal_over) -> tuple[Path, dict]:
    """A research quest that wrote its evidence before sealing (as a finishing quest does), then sealed."""
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("quest_started")
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    record = evidence.assess(root, _state(), settings={**RESEARCH, "sealing": True})
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(record), encoding="utf-8")
    files = {rel: evidence._file_sha256(root / rel) for rel in evidence.SEALED_FILES}
    log.append("quest_finalized", events_before=len(audit_log.read(trace)), write_errors=0, records_not_written=0,
               nodes_completed=["review", "write"], files=files, paper_path="paper/paper.md",
               paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"), **seal_over)
    return root, record


@pytest.mark.parametrize("flag", ["verified", None])
def test_the_record_cannot_vouch_for_its_own_seal(tmp_path: Path, flag) -> None:
    root, record = _finished(tmp_path)
    forged = {k: v for k, v in record.items() if k != "trace_seal"}
    if flag:
        forged["trace_seal"] = flag
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(forged), encoding="utf-8")
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified"
    assert any("EVIDENCE.json changed after the quest was sealed" in g for g in read["all_gaps"]["publication_ready"])
    assert read["levels"]["publication_ready"] is False


def test_a_record_that_says_it_was_not_research_is_still_checked(tmp_path: Path) -> None:
    root, record = _finished(tmp_path, rigor_profile="research")
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps({**record, "rigor_profile": "default"}), encoding="utf-8")
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified" and read["levels"]["publication_ready"] is False


def test_a_paper_changed_after_the_seal_is_a_gap(tmp_path: Path) -> None:
    root, _record = _finished(tmp_path)
    (root / "paper" / "paper.md").write_text("# edited", encoding="utf-8")
    assert any("paper (paper/paper.md) changed" in g for g in evidence.read(root)["all_gaps"]["publication_ready"])


def test_the_record_written_before_the_seal_says_so(tmp_path: Path) -> None:
    root, record = _finished(tmp_path)
    assert record["trace_seal"] == "pending"
    assert not any("seal" in g for g in record.get("all_gaps", {}).get("publication_ready", []))
    read = evidence.read(root)
    assert read["trace_seal"] == "verified" and read["status"] == record["status"]


def test_an_event_after_the_seal_is_not_covered_by_it(tmp_path: Path) -> None:
    root, record = _finished(tmp_path)
    audit_log.AuditLog(root / ".fi" / "audit.jsonl", root.name).append("check_result", check="late", status="ok")
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified"
    assert any("after the decision trace's final seal" in g for g in read["all_gaps"]["publication_ready"])
    assert read["levels"]["publication_ready"] is False


def test_evidence_changed_after_the_seal_is_not_what_it_sealed(tmp_path: Path) -> None:
    root, record = _finished(tmp_path)
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps({**record, "edited": True}), encoding="utf-8")
    read = evidence.read(root)
    assert any("EVIDENCE.json changed after the quest was sealed" in g for g in read["all_gaps"]["publication_ready"])


def test_a_ready_record_whose_seal_fails_is_read_one_level_down() -> None:
    levels = dict.fromkeys(evidence.LEVELS, True)
    record = {"status": "publication_ready", "levels": levels, "next_level": None, "gaps": [], "all_gaps": {},
              "ladder": [{"level": lv, "reached": True, "gaps": []} for lv in evidence.LEVELS],
              "trace_seal": "pending", "rigor_profile": "research"}
    read = evidence.verify_seal(Path("does-not-exist"), record)
    assert read["status"] == evidence.LEVELS[-2] and read["next_level"] == "publication_ready"
    assert read["gaps"] and read["ladder"][-1]["reached"] is False
    assert record["status"] == "publication_ready", "the record passed in is not changed"


def test_lost_events_before_the_seal_are_a_gap_when_sealing(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    trace = root / ".fi" / "audit.jsonl"
    trace.unlink(missing_ok=True)
    log = audit_log.AuditLog(trace, root.name)
    log.append("node_completed", node="write")
    log.append("node_completed", node="review")
    audit_log._lost_path(trace).write_text("2", encoding="utf-8")
    record = evidence.assess(root, _state(), settings={**RESEARCH, "sealing": True})
    assert any("2 event(s)" in g for g in record["all_gaps"]["publication_ready"])


def test_a_record_not_written_before_the_seal_is_a_gap(tmp_path: Path) -> None:
    root, _record = _finished(tmp_path)
    trace = root / ".fi" / "audit.jsonl"
    lines = trace.read_text(encoding="utf-8").splitlines()
    trace.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    audit_log.AuditLog(trace, root.name).append(
        "quest_finalized", events_before=len(lines) - 1, write_errors=0, records_not_written=1,
        nodes_completed=["review", "write"], paper_path="paper/paper.md",
        paper_sha256=evidence._file_sha256(root / "paper" / "paper.md"),
        files={rel: evidence._file_sha256(root / rel) for rel in evidence.SEALED_FILES})
    assert any("attempt record(s) could not be written" in g
               for g in evidence.read(root)["all_gaps"]["publication_ready"])


# --- the context of an attempt ----------------------------------------------------------------------------------------

def _cfg() -> Config:
    return Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                  "knowledge": {"enabled": False}})


def test_a_finished_quest_needs_its_code_environment_and_protocol(tmp_path: Path) -> None:
    ctx = ar.context_fingerprint(_cfg(), tmp_path, {}, kind="quest_end", prompts={})
    assert ctx["context_kind"] == "quest_end" and not ctx["complete"]
    assert {"no script in code/", "no environment record", "no protocol"} <= set(ctx["missing"])
    survey = ar.context_fingerprint(_cfg(), tmp_path, {"survey_mode_resolved": True}, kind="quest_end", prompts={})
    assert not {"no script in code/", "no environment record", "no protocol"} & set(survey["missing"])
    with pytest.raises(ValueError):
        ar.context_fingerprint(_cfg(), tmp_path, {}, kind="post_run", prompts={})


def test_two_programs_that_differ_only_in_the_middle_hash_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar, "_WHOLE_FILE_LIMIT", 16)
    monkeypatch.setattr(ar, "_CHUNK", 4)
    script = tmp_path / "code" / "simulate.py"
    script.parent.mkdir()
    script.write_bytes(b"head" + b"A" * 40 + b"tail")
    before = ar.script_hashes(tmp_path)
    script.write_bytes(b"head" + b"B" * 40 + b"tail")
    assert ar.script_hashes(tmp_path) != before, "code is read whole, not by its size and ends"
    data = tmp_path / "inputs" / "big.bin"
    data.parent.mkdir()
    data.write_bytes(b"x" * 40)
    assert ar._folder_manifest(tmp_path / "inputs")["complete"] is False, "a partial data hash is said to be partial"


def test_code_changed_while_the_cluster_job_was_queued_is_named(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("from helper import f\n", encoding="utf-8")
    (code / "helper.py").write_text("def f(): return 1\n", encoding="utf-8")
    (code / "experiment.py").write_text("print(1)\n", encoding="utf-8")
    record = trial_runner.prepare_cluster(tmp_path, "code/simulate.py", {"x": [1]}, runs_per_setting=1, base_seed=0,
                                          deterministic=True, key="k" * 12)
    assert record["code_at_submit"]["helper.py"]
    (code / "experiment.py").write_text("print(2)\n", encoding="utf-8")  # runs here, after collection: not the tasks'
    assert trial_runner.code_changed_while_queued(tmp_path, record, local=("experiment.py",)) == []
    (code / "helper.py").write_text("def f(): return 2\n", encoding="utf-8")
    assert trial_runner.code_changed_while_queued(tmp_path, record, local=("experiment.py",)) == ["helper.py"]
    ctx = ar.context_fingerprint(_cfg(), tmp_path, {}, kind="after_run", prompts={})
    assert any("changed while the cluster job was queued: helper.py" in m for m in ctx["missing"])
    assert any("cluster" in m for m in ar.context_fingerprint(_cfg(), tmp_path, {}, kind="in_progress",
                                                               prompts={})["missing"]), "a stop after it says so too"


@pytest.mark.asyncio
async def test_a_cli_call_never_takes_the_model_another_call_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from core.config import ProviderConfig
    from core.provider import LAST_CALL, LLMClient, resolve_endpoint

    async def fake_run_cli(spec, prompt, *, usage_out=None, **kw):  # noqa: ANN001
        if "slow" in prompt:
            await asyncio.sleep(0.05)  # finishes last and reports nothing
        elif usage_out is not None:
            usage_out.update({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                              "served_model": "claude-other"})
        return "ok"

    monkeypatch.setattr("core.provider._run_cli", fake_run_cli)
    client = LLMClient(resolve_endpoint(ProviderConfig(name="claude_cli")))

    async def ask(text: str) -> dict:
        await client.chat([{"role": "user", "content": text}])
        return dict(LAST_CALL.get() or {})

    slow, fast = await asyncio.gather(ask("slow reviewer"), ask("fast reviewer"))
    assert fast["reported"] is True and fast["model"] == "claude-other"
    assert slow["reported"] is False, "the slow call reported nothing; it must not borrow the other call's model"


def test_a_stale_cluster_note_goes_when_the_trials_run_here(tmp_path: Path) -> None:
    record_path = tmp_path / ".fi" / "trials" / "cluster.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps({"key": "k", "code_changed_while_queued": ["helper.py"]}), encoding="utf-8")
    trial_runner._forget_queued_changes(tmp_path)
    assert "code_changed_while_queued" not in json.loads(record_path.read_text(encoding="utf-8"))
