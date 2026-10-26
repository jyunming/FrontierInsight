"""What study was picked, or that nobody picked one, is said in QUALITY.md and in the writer's note about limitations.
The comparison of the plan with the chosen card is a reading by names: it is told, never a stop. Fake model only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import frozen_protocol as frozen
from core import quality_report as qr
from core import scope_cards
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

CARDS = [
    {"question": "How fast does a cup of tea cool in air?", "study_type": "measure",
     "main_results": ["cooling_rate"], "compared_settings": ["cup_material"], "literature": [], "run_time": ""},
    {"question": "Which cup shape keeps tea hot longest?", "study_type": "find_best_design",
     "main_results": ["time_to_cool"], "compared_settings": [], "literature": [], "run_time": ""},
    {"question": "Does stirring change the cooling curve?", "study_type": "measure",
     "main_results": ["curve_shift"], "compared_settings": ["stirring"], "literature": [], "run_time": ""},
]


def _record(root: Path, chosen: dict | None, removed: list[str] | None = None) -> None:
    scope_cards.write_record(root, CARDS, chosen, removed or [], widest=True)


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="a neutral topic about a cooling cup of tea", title="scope-disclosure", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(_cfg(tmp_path))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    return eng


def _quality(root: Path) -> tuple[dict[str, Any], str]:
    report = qr.build(root, running=False)
    return report, qr.render(report)


def test_quality_lists_the_cards_when_nobody_chose(tmp_path: Path) -> None:
    _record(tmp_path, None)
    report, text = _quality(tmp_path)
    assert scope_cards.nobody_line() in text
    for card in CARDS:
        assert card["question"] in text
    assert report["scope"]["chosen_by"] == "nobody"


def test_quality_names_the_chosen_study_and_the_removed_lines(tmp_path: Path) -> None:
    removed = ["A literature line on the study \"X\" was removed because its source (none given) is not one FI read: \"y\""]
    _record(tmp_path, CARDS[1], removed)
    report, text = _quality(tmp_path)
    assert "Which cup shape keeps tea hot longest?" in text
    assert scope_cards.nobody_line() not in text
    assert removed[0] in text
    # FI's own correction of a card, said in the study section; not a problem of the quest, so not counted as one
    assert f"- Removed: {removed[0]}" in text
    assert all(p["status"] != "removed" for s in report["stages"] for p in s["problems"]["items"])


def _freeze(root: Path) -> None:
    frozen.freeze(root, {"metrics": [{"id": "cooling_rate", "label": "Cooling rate"}], "grid": {"cup_material": [1, 2]}},
                  approved_by="test", source="test")


def test_a_plan_gap_against_the_chosen_card_is_in_the_note_and_does_not_stop(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _freeze(eng.quest_root)
    card = {**CARDS[0], "main_results": ["cooling_rate", "peak_temperature"]}
    _record(eng.quest_root, card)
    note = eng._run_checks_note()
    assert "peak_temperature" in note and "Say so in the limitations" in note
    assert "cooling_rate" not in note
    assert eng._cannot_repair_stop() is None


def test_no_scope_record_changes_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _freeze(eng.quest_root)
    assert eng._run_checks_note() == ""
    before = qr.render(qr.build(eng.quest_root, running=False))
    assert "Nobody picked" not in before and "study" not in before.lower().split("## outcome")[0]
    report = qr.build(eng.quest_root, running=False)
    assert not report.get("scope")
    assert all(p["status"] != "removed" for s in report["stages"] for p in s["problems"]["items"])
    assert json.dumps(report["problems"])  # the totals are unchanged by a missing record
