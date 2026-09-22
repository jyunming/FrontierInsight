"""The audit trace (core/audit_log.py): the file, its hash chain, redaction, and what the engine writes into it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import audit_log as al
from core.engine import Engine
from tests.test_engine_smoke import _classify
from tests.test_frozen_protocol import _cfg, _fake


# --- the file and its chain ---------------------------------------------------------------------------------------------


def _log(tmp_path: Path) -> al.AuditLog:
    return al.AuditLog(tmp_path / ".fi" / "audit.jsonl", "q1")


def test_events_are_numbered_chained_and_verify(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("node_started", node="plan")
    log.append("node_completed", node="plan", duration_s=1.5, wrote=["plan_md"])
    events = al.read(log.path)
    assert [e["seq"] for e in events] == [1, 2] and events[0]["prev"] == al.GENESIS and events[1]["prev"] == events[0]["hash"]
    assert all(e["schema"] == al.SCHEMA and e["quest_id"] == "q1" and e["provenance"] == al.DETERMINISTIC for e in events)
    v = al.verify(log.path)
    assert v.ok and v.events == 2 and "intact" in v.line()


def test_an_empty_or_missing_trace_is_intact(tmp_path: Path) -> None:
    assert al.verify(tmp_path / "nothing.jsonl").ok


@pytest.mark.parametrize("tamper", ["edit", "delete", "swap", "insert"])
def test_a_changed_trace_fails_the_chain_and_says_where(tmp_path: Path, tamper: str) -> None:
    log = _log(tmp_path)
    for i in range(4):
        log.append("check_result", check=f"c{i}", status="ok")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    if tamper == "edit":
        lines[1] = lines[1].replace('"status":"ok"', '"status":"fail"')
    elif tamper == "delete":
        del lines[1]
    elif tamper == "swap":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines.insert(2, lines[1])
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v = al.verify(log.path)
    assert not v.ok and v.bad_seq is not None and v.bad_seq <= 3 and "BROKEN" in v.line()


def test_a_resumed_quest_continues_the_same_chain(tmp_path: Path) -> None:
    first = _log(tmp_path)
    first.append("node_started", node="design")
    first.append("node_paused", node="design", pause="plan")
    second = _log(tmp_path)        # a new process, the same file
    assert second.last_kind is None and second.event_count() == 2
    assert (second.last_kind, second.last_node, second.paused_node) == ("node_paused", "design", "design")
    second.append("node_started", node="design", resumed=True)
    assert second.paused_node is None
    assert [e["seq"] for e in al.read(second.path)] == [1, 2, 3] and al.verify(second.path).ok


def test_a_torn_last_line_is_dropped_and_said_so(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("node_started", node="plan")
    with log.path.open("ab") as fh:
        fh.write(b'{"schema":"fi.audit/v1","seq":2,"ki')       # a crash in the middle of a write
    again = _log(tmp_path)
    again.append("node_completed", node="plan")
    events = al.read(again.path)
    assert [e["kind"] for e in events] == ["node_started", "audit_repair", "node_completed"]
    assert events[1]["dropped_bytes"] > 0 and al.verify(again.path).ok


def test_a_write_that_fails_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / ".fi"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    log = _log(tmp_path)
    assert log.append("node_started", node="plan") is None and log.write_errors == 1


def test_a_disabled_trace_writes_nothing(tmp_path: Path) -> None:
    log = al.AuditLog(tmp_path / ".fi" / "audit.jsonl", "q1", enabled=False)
    assert log.append("node_started", node="plan") is None and not log.path.exists()


# --- redaction ----------------------------------------------------------------------------------------------------------


def test_credentials_and_the_home_directory_never_reach_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-secret-value-123456")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok_abcdefgh12345678")
    log = _log(tmp_path)
    home = str(Path.home())
    log.append(
        "node_failed", node="plan",
        error=f"401 for key moon-secret-value-123456; header Bearer abcdefghijklmnopqrst; sk-abcdefghijklmnopqrstuv; "
              f"api_key=hunter2hunter2 at {home}\\work\\x.py and tok_abcdefgh12345678",
        nested={"list": ["moon-secret-value-123456"], "n": 3},
    )
    text = log.path.read_text(encoding="utf-8")
    for leaked in ("moon-secret-value-123456", "tok_abcdefgh12345678", "abcdefghijklmnopqrst", "hunter2hunter2", home.replace("\\", "\\\\")):
        assert leaked not in text, leaked
    event = al.read(log.path)[0]
    assert event["nested"]["n"] == 3 and "~" in event["error"] and "[redacted]" in event["error"]


def test_long_text_and_long_lists_are_cut_and_say_so(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("check_result", check="x", status="warned", summary="x" * 5000, problems=[str(i) for i in range(100)])
    event = al.read(log.path)[0]
    assert "more characters not kept" in event["summary"] and len(event["problems"]) == 61 and "more not kept" in event["problems"][-1]


def test_a_field_cannot_overwrite_the_chain_fields(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("node_started", node="plan", seq=99, hash="forged", prev="forged")
    assert al.verify(log.path).ok and al.read(log.path)[0]["seq"] == 1


# --- presenting ---------------------------------------------------------------------------------------------------------


def test_a_trace_file_written_before_thinking_fragment_was_retired_still_renders_safely() -> None:
    # `kind`/`provenance` are not validated against KINDS/PROVENANCES at write time (they are the
    # vocabulary real callers use, not an enforced schema), so a trace file written before
    # thinking_fragment/provider_commentary were retired still has them on disk; nothing rewrites
    # old files, so describe()/_tag() must degrade to a plain fallback for those, not crash.
    legacy_event = {
        "seq": 1, "kind": "thinking_fragment", "node": "design",
        "provenance": "provider_commentary", "text": "hmm",
    }
    assert al.describe(legacy_event) == "design: thinking_fragment"  # kind fallback: f"{where}{kind}"
    assert al._tag(legacy_event) == ""  # provenance fallback: unrecognized tag is silently empty, not KeyError


def test_detail_levels_and_the_node_filter(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("node_started", node="design")
    log.append("model_claim", node="design", provenance=al.MODEL_CLAIM, topic="assumption", claim="trials are independent")
    log.append("check_result", node="design", check="protocol", status="ok")
    log.append("node_completed", node="design", duration_s=2.0, wrote=["design"])
    log.append("route_decision", node="review", chosen="done", facts={"verdict": "accept"})
    events = al.read(log.path)
    kinds = lambda **kw: [e["kind"] for e in al.select(events, **kw)]  # noqa: E731
    assert kinds(detail="summary") == ["node_completed", "route_decision"]
    assert kinds(detail="checks") == ["model_claim", "check_result", "node_completed", "route_decision"]
    assert len(kinds(detail="debug")) == 5 and kinds(detail="debug", node="review") == ["route_decision"]
    assert "node_started" not in kinds(detail="checks") and "node_started" in kinds(detail="debug")
    text = "\n".join(al.render(events))
    assert "[model claim]" in text and "next is done because verdict=accept" in text


# --- what the engine writes ---------------------------------------------------------------------------------------------


def _events(engine: Engine) -> list[dict[str, Any]]:
    return al.read(engine.quest_root / ".fi" / "audit.jsonl")


def _with_rationale(inner):
    async def chat(self, messages, **kw):  # noqa: ANN001
        reply = await inner(self, messages, **kw)
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design" and "Apply this checklist" not in prompt:
            body = json.loads(reply)
            body["rationale"] = {
                "assumptions": ["trials are independent"],
                "alternatives_considered": [{"option": "Wilson pooled", "decision": "rejected", "reason": "clustered trajectories"}],
            }
            return json.dumps(body)
        return reply

    return chat


@pytest.mark.asyncio
async def test_a_quest_leaves_a_trace_of_its_nodes_checks_routes_and_the_reasons_the_model_gave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _with_rationale(_fake(prompts, second_design={})))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    events = _events(engine)
    assert al.verify(engine.audit.path).ok, al.verify(engine.audit.path).line()
    assert events[0]["kind"] == "quest_started" and events[0]["resumed"] is False

    # Every node that started finished, in order, and the review asked for one revision.
    started = [e["node"] for e in events if e["kind"] == "node_started"]
    done = [e["node"] for e in events if e["kind"] == "node_completed"]
    assert started == done and started[:3] == ["clarify", "ideate", "literature"] and started.count("design") == 2
    assert all(e["duration_s"] >= 0 for e in events if e["kind"] == "node_completed")
    routes = [(e["node"], e["chosen"]) for e in events if e["kind"] == "route_decision"]
    assert ("review", "revise") in routes and routes[-1] == ("review", "done")
    first_review = next(e for e in events if e["kind"] == "route_decision" and e["node"] == "review")
    assert first_review["facts"]["verdict"] == "revise"

    # Each check's verdict is written with the record that holds it, and only deterministic events carry checks.
    checks = {e["check"]: e for e in events if e["kind"] == "check_result"}
    assert {"protocol", "oracle", "evidence", "design_self_critique"} <= set(checks)
    assert checks["oracle"]["status"] == "ok" and checks["oracle"]["record"] == "needs/ORACLE_CHECK.json"
    assert checks["oracle"]["sha256"] == al.file_sha256(engine.quest_root / "needs" / "ORACLE_CHECK.json")
    assert all(e["provenance"] == al.DETERMINISTIC for e in events if e["kind"] in ("check_result", "node_started", "route_decision"))

    # What the model said about why is its own kind of event, labelled, and kept apart from the checks.
    claims = [e for e in events if e["kind"] == "model_claim"]
    assert claims and all(e["provenance"] == al.MODEL_CLAIM and "check" not in e for e in claims)
    topics = {(e["topic"], e["claim"]) for e in claims}
    assert ("assumption", "trials are independent") in topics and ("alternative", "Wilson pooled") in topics
    assert any(t == "review_weakness" and c == "the grid is too thin" for t, c in topics)
    assert any(t.startswith("design_self_critique") and "precision target" in c for t, c in topics)
    # The methodology audit does not see the rationale, so an amendment that does not repeat it is still an amendment.
    assert checks["design_self_critique"]["status"] == "ok", checks["design_self_critique"]
    alt = next(e for e in claims if e["topic"] == "alternative")
    assert (alt["decision"], alt["reason"]) == ("rejected", "clustered trajectories")
    # The rationale is a note for a reader: later prompts do not carry it.
    assert "rationale" not in (artifacts.raw_state["design"] or {})

    # The files a quest wrote are named with their hash, once per change.
    written = [(e["path"], e["sha256"]) for e in events if e["kind"] == "artifact_created"]
    assert len(written) == len(set(written)) and {p for p, _ in written} >= {"plan.md", "code/experiment.py", "paper/paper.md"}
    last_plan = [sha for path, sha in written if path == "plan.md"][-1]
    assert last_plan == al.file_sha256(engine.quest_root / "plan.md"), "the hash in the trace is the hash of the file"


@pytest.mark.asyncio
async def test_a_pause_is_a_pause_not_a_failure_and_the_resumed_run_continues_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={}))
    cfg = _cfg(tmp_path)
    cfg.pauses.plan = "ask"
    first = Engine(cfg)
    await first.run()
    events = _events(first)
    assert not [e for e in events if e["kind"] == "node_failed"]
    paused = [e for e in events if e["kind"] == "node_paused"]
    assert paused and paused[-1]["pause"] == "plan"
    requested = [e for e in events if e["kind"] == "pause_requested"]
    assert requested and requested[-1]["pause"] == "plan"
    count = len(events)

    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run()
    events = _events(second)
    assert al.verify(second.audit.path).ok and len(events) > count
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    restarted = next(e for e in events[count:] if e["kind"] == "node_started")
    assert restarted["node"] == paused[-1]["node"] and restarted["resumed"] is True
    assert events[count]["kind"] == "quest_started" and events[count]["resumed"] is True


@pytest.mark.asyncio
async def test_a_node_that_raises_is_recorded_and_still_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(self, state):  # noqa: ANN001
        raise RuntimeError("the literature search fell over: key moon-secret-value-123456")

    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-secret-value-123456")
    monkeypatch.setattr("core.engine.Engine._node_literature", boom)
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], second_design={}))
    engine = Engine(_cfg(tmp_path))
    with pytest.raises(RuntimeError):
        await engine.run()
    failed = [e for e in _events(engine) if e["kind"] == "node_failed"]
    assert len(failed) == 1 and failed[0]["node"] == "literature" and "fell over" in failed[0]["error"]
    assert "moon-secret-value" not in (engine.quest_root / ".fi" / "audit.jsonl").read_text(encoding="utf-8")
    assert al.verify(engine.audit.path).ok


@pytest.mark.asyncio
async def test_the_trace_can_be_turned_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], second_design={}))
    engine = Engine(_cfg(tmp_path, audit_trace=False))
    await engine.run()
    assert not (engine.quest_root / ".fi" / "audit.jsonl").exists()


# --- the command line and the web page ----------------------------------------------------------------------------------


def _quest_with_trace(root: Path, quest_id: str = "q-trace") -> Path:
    quest = root / quest_id
    log = al.AuditLog(quest / ".fi" / "audit.jsonl", quest_id)
    log.append("quest_started", resumed=False)
    log.append("node_started", node="design", iteration=0)
    log.append("model_claim", node="design", provenance=al.MODEL_CLAIM, topic="assumption", claim="trials are independent")
    log.append("check_result", node="design", check="protocol", status="stopped", summary="the grid differs", problems=["grid: 3 vs 5"])
    log.append("node_completed", node="design", duration_s=1.2, wrote=["design"])
    log.append("route_decision", node="review", chosen="done", facts={"verdict": "accept"})
    return quest


def test_the_command_line_prints_the_timeline_and_verifies_the_chain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _show_trace

    _quest_with_trace(tmp_path)
    assert _show_trace("q-trace", "", "checks", tmp_path) == 0
    out = capsys.readouterr().out
    assert "check protocol: stopped - the grid differs" in out and "[model claim]" in out and "next is done because verdict=accept" in out
    assert "chain intact (6 events)" in out and "started" not in out.replace("quest started", ""), "node starts are debug detail"
    assert _show_trace("q-trace", "review", "summary", tmp_path) == 0
    only = capsys.readouterr().out
    assert "next is done" in only and "check protocol" not in only


def test_the_command_line_fails_on_a_tampered_trace_and_says_when_there_is_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _show_trace

    quest = _quest_with_trace(tmp_path)
    path = quest / ".fi" / "audit.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"status":"stopped"', '"status":"ok"'), encoding="utf-8")
    assert _show_trace("q-trace", "", "checks", tmp_path) == 1
    assert "CHAIN BROKEN at event 4" in capsys.readouterr().out
    (tmp_path / "q-old" / ".fi").mkdir(parents=True)
    assert _show_trace("q-old", "", "checks", tmp_path) == 1 and "no audit trace" in capsys.readouterr().out
    assert _show_trace("q-missing", "", "checks", tmp_path) == 1 and "no quest" in capsys.readouterr().out


def test_the_web_api_serves_the_trace_with_the_same_descriptions_and_refuses_a_bad_detail(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    root = tmp_path / "outputs"
    _quest_with_trace(root)
    client = TestClient(make_app(root))
    body = client.get("/api/quests/q-trace/trace").json()
    assert body["total"] == 6 and body["chain"]["ok"] is True and body["nodes"] == ["design", "review"]
    kinds = [e["kind"] for e in body["events"]]
    assert kinds == ["quest_started", "model_claim", "check_result", "node_completed", "route_decision"], "checks detail hides node starts"
    claim = next(e for e in body["events"] if e["kind"] == "model_claim")
    assert claim["provenance"] == "model_claim" and claim["description"] == "design: assumption: trials are independent"
    assert [e["kind"] for e in client.get("/api/quests/q-trace/trace?node=review").json()["events"]] == ["route_decision"]
    assert len(client.get("/api/quests/q-trace/trace?detail=debug").json()["events"]) == 6
    assert client.get("/api/quests/q-trace/trace?detail=everything").status_code == 400
    assert client.get("/api/quests/bad id!/trace").status_code == 400

    path = root / "q-trace" / ".fi" / "audit.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("trials are independent", "trials are dependent"), encoding="utf-8")
    broken = client.get("/api/quests/q-trace/trace").json()
    assert broken["chain"]["ok"] is False and broken["chain"]["bad_seq"] == 3

    (root / "q-none" / ".fi").mkdir(parents=True)
    empty = client.get("/api/quests/q-none/trace").json()
    assert empty["total"] == 0 and empty["events"] == [] and empty["chain"]["ok"] is True


def test_the_quest_page_has_the_trace_panel() -> None:
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert 'id="trace-section"' in page and "/trace?detail=" in page and "model claim" in page
