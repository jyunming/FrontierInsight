"""Changing a quest's title after it has run (core/quest_title.py), from the CLI and the web page."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import TypedDict

import pytest
import yaml

from core import audit_log
from core import quest_title

_QID = "1790000000-rename-me"
_PAPER = "# A Poor Model Title\n\n## Abstract\n\nWe did a thing.\n\n## Results\n\n# Not the title\n"


def _quest(root: Path, *, config: str | None = None, paper: str | None = _PAPER, summary: bool = True,
           fresh_log: bool = False) -> Path:
    q = root / _QID
    (q / ".fi").mkdir(parents=True)
    (q / "paper").mkdir()
    (q / "code").mkdir()
    (q / "code" / "run.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    (q / "config.yaml").write_text(
        config if config is not None else "# my quest\ntopic: How fast is it\ntitle: rename_me\nprovider:\n  name: ollama\n",
        encoding="utf-8")
    if paper is not None:
        (q / "paper" / "paper.md").write_text(paper, encoding="utf-8")
    if summary:
        (q / "frontier_insight_summary.json").write_text(json.dumps({"quest_id": _QID, "provider": "ollama"}),
                                                         encoding="utf-8")
    log = q / ".fi" / "run.log"
    log.write_text("[write] done\n", encoding="utf-8")
    if not fresh_log:
        old = time.time() - 3600
        os.utime(log, (old, old))
    audit_log.AuditLog(q / ".fi" / "audit.jsonl", _QID).append("quest_started", resumed=False)
    return q


def test_rename_updates_every_place_the_title_lives(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    before_code = (q / "code" / "run.py").read_text(encoding="utf-8")
    result = quest_title.rename(q, "  Step Size and Energy Drift in Symplectic Integrators ")
    new = "Step Size and Energy Drift in Symplectic Integrators"
    assert result.old == "A Poor Model Title"
    assert result.new == new
    paper = (q / "paper" / "paper.md").read_text(encoding="utf-8")
    assert paper == _PAPER.replace("# A Poor Model Title", f"# {new}", 1), "only the title line changes"
    assert "# Not the title" in paper, "only the first heading is the title"
    cfg_text = (q / "config.yaml").read_text(encoding="utf-8")
    assert cfg_text.startswith("# my quest\n"), "the comment in the YAML is kept"
    data = yaml.safe_load(cfg_text)
    assert data["title"] == new and data["topic"] == "How fast is it" and data["provider"] == {"name": "ollama"}
    from core.config import Config
    assert Config.from_yaml(q / "config.yaml").title == new
    assert json.loads((q / "frontier_insight_summary.json").read_text(encoding="utf-8"))["title"] == new
    assert (q / "code" / "run.py").read_text(encoding="utf-8") == before_code
    assert quest_title.current_title(q) == new


def test_rename_is_recorded_in_the_audit_trace(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    quest_title.rename(q, "Better Title")
    path = q / ".fi" / "audit.jsonl"
    events = audit_log.read(path)
    last = events[-1]
    assert last["kind"] == "title_changed"
    assert last["old"] == "A Poor Model Title" and last["new"] == "Better Title"
    assert audit_log.verify(path).ok
    assert "Better Title" in audit_log.describe(last)
    assert last in audit_log.select(events, detail="summary")


def test_rename_adds_a_title_line_to_a_yaml_without_one(tmp_path: Path) -> None:
    q = _quest(tmp_path, config="topic: x\n")
    quest_title.rename(q, 'A "quoted": title')
    assert yaml.safe_load((q / "config.yaml").read_text(encoding="utf-8")) == {"topic": "x", "title": 'A "quoted": title'}


def test_rename_edits_a_front_matter_title(tmp_path: Path) -> None:
    q = _quest(tmp_path, paper='---\ntitle: "Old"\nauthor: me\n---\n\n## Abstract\n\nx\n')
    quest_title.rename(q, "New")
    paper = (q / "paper" / "paper.md").read_text(encoding="utf-8")
    assert paper.startswith('---\ntitle: "New"\nauthor: me\n---\n')


@pytest.mark.parametrize("bad", ["", "   ", "two\nlines", "x" * 201])
def test_rename_refuses_a_bad_title(tmp_path: Path, bad: str) -> None:
    q = _quest(tmp_path)
    with pytest.raises(quest_title.RenameRefused):
        quest_title.rename(q, bad)
    assert (q / "paper" / "paper.md").read_text(encoding="utf-8") == _PAPER


def test_rename_refuses_while_the_quest_is_running(tmp_path: Path) -> None:
    q = _quest(tmp_path, summary=False, fresh_log=True)
    assert quest_title.looks_running(q)
    with pytest.raises(quest_title.RenameRefused, match="still running"):
        quest_title.rename(q, "New")
    assert (q / "paper" / "paper.md").read_text(encoding="utf-8") == _PAPER


def test_a_paused_quest_can_be_renamed(tmp_path: Path) -> None:
    q = _quest(tmp_path, summary=False, fresh_log=True)
    (q / ".fi" / "pause.json").write_text('{"kind": "review"}', encoding="utf-8")
    assert not quest_title.looks_running(q)
    assert quest_title.rename(q, "New").new == "New"


def test_rename_names_the_outputs_that_keep_the_old_title(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    (q / "paper.pdf").write_bytes(b"%PDF-1.5")
    (q / "slides.html").write_text("x", encoding="utf-8")
    result = quest_title.rename(q, "New")
    assert result.outputs_to_redo == ["paper_pdf", "slides"]
    text = "\n".join(result.lines())
    assert "--emit paper_pdf" in text and "--emit slides" in text


class _S(TypedDict, total=False):
    topic: str
    title: str
    title_confirmed: bool


def test_rename_updates_the_saved_state_a_later_rerun_reads(tmp_path: Path) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.graph import END, StateGraph

    q = _quest(tmp_path)

    def _graph():
        g = StateGraph(_S)
        g.add_node("write", lambda s: {"title": "A Poor Model Title"})
        g.set_entry_point("write")
        g.add_edge("write", END)
        return g

    cfg = {"configurable": {"thread_id": _QID}}

    async def run() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(q / ".fi" / "state.sqlite")) as saver:
            await _graph().compile(checkpointer=saver).ainvoke({"topic": "t"}, cfg)

    async def read() -> dict:
        async with AsyncSqliteSaver.from_conn_string(str(q / ".fi" / "state.sqlite")) as saver:
            return dict((await _graph().compile(checkpointer=saver).aget_state(cfg)).values)

    asyncio.run(run())
    result = quest_title.rename(q, "Better Title")
    assert result.saved_state
    values = asyncio.run(read())
    assert values["title"] == "Better Title" and values["title_confirmed"] is True
    assert values["topic"] == "t"


def test_a_quoted_title_key_is_not_written_twice(tmp_path: Path) -> None:
    q = _quest(tmp_path, config='"title": Old\ntopic: x\n')
    quest_title.rename(q, "New")
    text = (q / "config.yaml").read_text(encoding="utf-8")
    assert text.count("title") == 1 and yaml.safe_load(text) == {"title": "New", "topic": "x"}


def test_an_unreadable_config_is_refused_before_anything_changes(tmp_path: Path) -> None:
    q = _quest(tmp_path, config="topic: [unclosed\n")
    with pytest.raises(quest_title.RenameRefused, match="config.yaml cannot be read"):
        quest_title.rename(q, "New")
    assert (q / "paper" / "paper.md").read_text(encoding="utf-8") == _PAPER


def test_a_failed_write_puts_back_what_was_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q = _quest(tmp_path)
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(quest_title.os, "replace", flaky)
    with pytest.raises(quest_title.RenameRefused, match="Nothing was changed"):
        quest_title.rename(q, "New")
    assert (q / "paper" / "paper.md").read_text(encoding="utf-8") == _PAPER
    assert "rename_me" in (q / "config.yaml").read_text(encoding="utf-8")
    assert not list(q.rglob("*.renaming"))
    assert audit_log.read(q / ".fi" / "audit.jsonl")[-1]["kind"] != "title_changed"


def test_titles_with_invisible_characters_or_not_text_are_refused() -> None:
    for bad in ("A\x7fB", "A‮B", ["a"]):
        with pytest.raises(quest_title.RenameRefused):
            quest_title.clean(bad)


def test_rename_from_inside_the_quest_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    q = _quest(tmp_path)
    monkeypatch.chdir(q)
    result = quest_title.rename(Path("."), "New")
    assert result.quest_id == _QID
    assert audit_log.read(q / ".fi" / "audit.jsonl")[-1]["quest_id"] == _QID


def test_front_matter_without_a_title_keeps_the_heading_as_the_title(tmp_path: Path) -> None:
    q = _quest(tmp_path, paper="---\nauthor: me\n---\n\n# Old\n\ntext\n")
    assert quest_title.current_title(q) == "Old"
    quest_title.rename(q, "New")
    assert (q / "paper" / "paper.md").read_text(encoding="utf-8") == "---\nauthor: me\n---\n\n# New\n\ntext\n"


def _seal(q: Path) -> None:
    """A finished research quest's seal, as core/engine.py writes it, naming the paper's hash."""
    from core import evidence

    fi = q / ".fi"
    for rel in evidence.SEALED_FILES:
        (q / rel).parent.mkdir(parents=True, exist_ok=True)
        (q / rel).touch()
    log = audit_log.AuditLog(fi / "audit.jsonl", _QID)
    for node in ("write", "review"):
        log.append("node_completed", node=node)
    log.append(
        "quest_finalized", events_before=len(audit_log.read(fi / "audit.jsonl")), write_errors=0, records_not_written=0,
        model_calls_not_written=0, model_calls={"gaps": []}, rigor_profile="research",
        nodes_completed=list(evidence._SEALED_STEPS), paper_path="paper/paper.md",
        paper_sha256=quest_title._sha256(q / "paper" / "paper.md"),
        files={rel: quest_title._sha256(q / rel) for rel in evidence.SEALED_FILES},
        sealed_records=evidence.SEAL_RECORDS,
    )


def test_renaming_a_sealed_quest_keeps_its_seal(tmp_path: Path) -> None:
    from core import evidence

    q = _quest(tmp_path)
    _seal(q)
    trace = q / ".fi" / "audit.jsonl"
    assert evidence._trace_completeness_gaps(trace, audit_log) == []
    quest_title.rename(q, "Better Title")
    quest_title.rename(q, "Best Title")
    assert evidence._trace_completeness_gaps(trace, audit_log) == []
    # Any other edit of the paper is still one.
    paper = q / "paper" / "paper.md"
    paper.write_text(paper.read_text(encoding="utf-8") + "\nmore\n", encoding="utf-8")
    assert any("changed after the quest was sealed" in g for g in evidence._trace_completeness_gaps(trace, audit_log))


def test_following_a_renamed_paused_quest_says_it_stopped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _follow_trace

    q = _quest(tmp_path)
    audit_log.AuditLog(q / ".fi" / "audit.jsonl", _QID).append("node_paused", node="review", pause="review")
    (q / ".fi" / "pause.json").write_text('{"kind": "review"}', encoding="utf-8")
    quest_title.rename(q, "New")
    assert _follow_trace(_QID, "", "summary", tmp_path, poll_s=0.01, max_hours=0.001) == 0
    assert "stopped for you" in capsys.readouterr().out


# --- CLI -----------------------------------------------------------------------------------------------------------


def test_cli_title_flag_takes_a_title_that_starts_with_a_dash(tmp_path: Path) -> None:
    from launch import _rename_quest, parse_args

    q = _quest(tmp_path)
    args = parse_args(["--rename", _QID, "--title=-40 C and --help", "--output-root", str(tmp_path)])
    assert args.rename == [_QID] and args.title == "-40 C and --help"
    assert _rename_quest(args.rename, args.output_root, title=args.title) == 0
    assert quest_title.current_title(q) == "-40 C and --help"


def test_cli_tools_rename_parses_to_the_rename_flag() -> None:
    from launch import parse_args

    args = parse_args(["tools", "rename", _QID, "Better", "Title"])
    assert args.rename == [_QID, "Better", "Title"]


def test_cli_rename_changes_the_title(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _rename_quest

    q = _quest(tmp_path)
    assert _rename_quest([_QID, "Better", "Title"], tmp_path) == 0
    assert quest_title.current_title(q) == "Better Title"
    assert "Better Title" in capsys.readouterr().out


def test_cli_rename_refusal_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _rename_quest

    _quest(tmp_path, summary=False, fresh_log=True)
    assert _rename_quest([_QID, "New"], tmp_path) == 1
    assert "still running" in capsys.readouterr().out
    assert _rename_quest([_QID], tmp_path) == 2
    assert _rename_quest(["no-such-quest", "x"], tmp_path) == 1


# --- web -----------------------------------------------------------------------------------------------------------


def test_web_rename_endpoint(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    q = _quest(tmp_path)
    client = TestClient(make_app(output_root=tmp_path))
    assert client.get(f"/api/quests/{_QID}").json()["title"] == "A Poor Model Title"
    r = client.post(f"/api/quests/{_QID}/title", json={"title": "Better Title"})
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Better Title"
    assert quest_title.current_title(q) == "Better Title"
    assert client.get(f"/api/quests/{_QID}").json()["title"] == "Better Title"
    assert client.post(f"/api/quests/{_QID}/title", json={"title": "a\nb"}).status_code == 400
    assert client.post("/api/quests/nope/title", json={"title": "x"}).status_code == 404


def test_web_rename_refuses_a_running_quest(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    _quest(tmp_path, summary=False, fresh_log=True)
    client = TestClient(make_app(output_root=tmp_path))
    r = client.post(f"/api/quests/{_QID}/title", json={"title": "New"})
    assert r.status_code == 409
    assert "still running" in r.json()["detail"]
