"""The to-do card (core/todo.py): one place for everything a quest is waiting for."""

from __future__ import annotations

import json
from pathlib import Path

from core import todo


def test_every_pause_kind_the_engine_raises_has_a_decision_and_a_recommendation() -> None:
    engine = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    import re

    kinds = set(re.findall(r'kind="([a-z_]+)",\s*\n?\s*interaction=', engine))
    kinds |= {"split", "manifest", "design_audit_unknown"}  # passed in as a variable at their call sites
    assert kinds, "no pause kinds found"
    for kind in kinds:
        decide, recommended, _ = todo.advice(kind)
        assert decide and recommended, kind


def test_the_card_puts_the_pause_first_then_everything_else_waiting(tmp_path: Path) -> None:
    root, fi = tmp_path / "q", tmp_path / "q" / ".fi"
    (root / "needs").mkdir(parents=True)
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "warned"}), encoding="utf-8")
    (root / "needs" / "WANTED_PAPERS.md").write_text("- a paper\n", encoding="utf-8")
    item = todo.pause_item("numeric", "the run's numerics warned", ["Fix experiment.py, then resume."])
    items = todo.write(root, fi, "q1", item)
    card = (root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert card.startswith("# Action needed — the run's numerics warned")
    for part in ("## What to decide", "## Recommended", "## Or", "## What to do", "## Also waiting for you",
                 "the oracle checks", "WANTED_PAPERS", "## Then go on", "--resume q1"):
        assert part in card, part
    assert [i.kind for i in items] == ["numeric", "warned", "papers"]
    saved = json.loads((fi / todo.TODO_NAME).read_text(encoding="utf-8"))
    assert saved["items"][0]["blocking"] and saved["items"][0]["recommended"].startswith("Fix experiment.py")
    text = todo.text(fi)
    assert text.splitlines()[0] == "[FI] Waiting for you: the run's numerics warned" and "Recommended:" in text


def test_a_call_site_can_give_its_own_recommendation(tmp_path: Path) -> None:
    item = todo.pause_item("oracle", "h", [], recommended="Accept the proposed tolerance.", alternatives=["Fix it."])
    assert item.recommended == "Accept the proposed tolerance." and item.alternatives == ["Fix it."]
    assert todo.pause_item("design_audit_unknown", "h", []).alternatives[0].startswith("Go on as it is")


def test_a_finished_quest_writes_no_next_step_and_only_what_is_worth_a_look(tmp_path: Path) -> None:
    root, fi = tmp_path / "q", tmp_path / "q" / ".fi"
    (root / "needs").mkdir(parents=True)
    todo.write(root, fi, "q1", None)
    assert not (root / "NEXT_STEP.md").exists() and not (fi / todo.TODO_NAME).exists()
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(
        {"status": "executed", "next_level": "internally_reconciled", "gaps": ["the audits did not run", "b"]}),
        encoding="utf-8")
    todo.write(root, fi, "q1", None)
    assert not (root / "NEXT_STEP.md").exists()
    (item,) = todo.read(fi)
    assert not item["blocking"] and "Next: the audits did not run (+1 more)" in item["why"]
    assert "internally_reconciled" not in item["why"], "a level is named in words, never by its identifier"
    assert todo.text(fi).startswith("[FI] Also:")
