"""The gaps an external review found in the extend / redraw routes: what is excused after an extension, a request the
run could not carry out is told to the paper once, a bare marker is removed, the extension is not repaired."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core import protocol_check
from core.engine import Engine, _take_refine_points
from tests.test_refine_extend import STATE, _engine, _replot_quest, _run_replot


def test_a_bare_marker_line_is_removed_from_the_paper() -> None:
    paper, points = _take_refine_points("# P\n\nBody.\n\nNEEDS_DATA:\n**NEEDS_LAYOUT:**\n")
    assert "NEEDS_" not in paper and "Body." in paper
    assert points == {"experiment": [], "data": [], "layout": []}


def test_only_what_the_person_added_is_excused_and_other_drift_is_still_reported(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    protocol = {"grid": {"R0": [0.9, 1.5, 3.0]}}
    extended = "R0_LIST = [0.9, 1.5, 3.0, 6.0]\n"
    (code_dir / "experiment.py").write_text(extended, encoding="utf-8")
    added = [m.message() for m in protocol_check.check(protocol, {"experiment.py": extended})]
    assert added
    eng._protocol_record({"status": "extended_by_person", "differences": added})
    assert eng._protocol_drift_not_asked_for(protocol) == []
    # A repair after the extension changes a constant the person did not ask about: that is still reported.
    (code_dir / "experiment.py").write_text("R0_LIST = [0.9, 1.5, 3.0, 6.0, 9.0]\n", encoding="utf-8")
    assert eng._protocol_drift_not_asked_for(protocol)
    # Without an extension nothing is excused.
    eng._protocol_record({"status": "ok"})
    (code_dir / "experiment.py").write_text(extended, encoding="utf-8")
    assert eng._protocol_drift_not_asked_for(protocol) == added


def test_a_note_the_redraw_cannot_do_is_kept_and_sends_the_paper_back_to_the_writer(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    script = "print('NOT_DRAWN: Figure 2 should come before Figure 1 in the paper')"
    out, _ran = _run_replot(eng, script)
    assert out["layout_missed"] == ["Figure 2 should come before Figure 1 in the paper"]
    assert eng._route_after_replot({**STATE, **out}) == "write"  # type: ignore[arg-type]


def test_a_redraw_with_nothing_to_draw_from_or_that_fails_is_reported(tmp_path: Path) -> None:
    eng = _replot_quest(tmp_path)
    out, _ = _run_replot(eng, "raise SystemExit(3)")
    assert out["layout_missed"] == ["Figure 2 above Figure 1"]
    empty = _engine(tmp_path / "other")
    (empty.quest_root / "code").mkdir(parents=True)
    gone = asyncio.run(empty._node_replot_layout({**STATE, "refine_layout": ["make it bigger"]}))  # type: ignore[arg-type]
    assert gone["layout_missed"] == ["make it bigger"]
    fine, _ = _run_replot(_replot_quest(tmp_path / "third"), "print('REPLOTTED: fig2.png')")
    assert fine["layout_missed"] == [] and eng._route_after_replot({**STATE, **fine}) == "check"  # type: ignore[arg-type]


def test_a_collection_that_is_switched_off_says_what_was_not_added(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.engine.auto_collect_data = False  # type: ignore[assignment]
    out = asyncio.run(eng._node_auto_collect_data(  # type: ignore[arg-type]
        {**STATE, "no_simulation_resolved": True, "refine_extend": ["GDP of Peru in 2019"]}))
    assert out["extend_missed"] == ["GDP of Peru in 2019"] and out["refine_extend"] == []


def test_the_paper_is_told_once_what_could_not_be_done(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "paper").mkdir(parents=True)
    prompts: list[str] = []

    async def chat(prompt: str, **kw: Any) -> str:
        prompts.append(prompt)
        return "# P\n\nBody.\n"

    eng._chat = chat  # type: ignore[method-assign]
    state = {**STATE, "refine_written_for": 1, "extend_missed": ["the runtime for n=64"],
             "layout_missed": ["Figure 2 above Figure 1"]}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    assert "the runtime for n=64" in prompts[-1] and "Figure 2 above Figure 1" in prompts[-1]
    assert out["extend_missed"] == [] and out["layout_missed"] == []
    asyncio.run(eng._node_write({**state, **out}))  # type: ignore[arg-type]
    assert "the runtime for n=64" in prompts[-1], "the person's own notes are still shown to the writer"
    assert "could not be added" not in prompts[-1] and "not done" not in prompts[-1].split("limitations")[-1][:0]


def test_the_write_after_an_extended_run_still_shows_the_person_the_notes(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "paper").mkdir(parents=True)
    prompts: list[str] = []

    async def chat(prompt: str, **kw: Any) -> str:
        prompts.append(prompt)
        return "# P\n\nBody.\n"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_write({**STATE, "refine_written_for": 1,  # type: ignore[arg-type]
                                 "result_json": {"runtime_n64": 12.5}, "analysis": {"runtime_n64": 12.5}}))
    assert "I do not see the runtime for n=64" in prompts[-1] and "12.5" in prompts[-1]


def test_an_extension_does_not_run_the_seed_repair(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    code_dir = eng.quest_root / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "experiment.py").write_text("for n in (8, 16):\n    pass\nprint('RESULT_JSON: {}')\n", encoding="utf-8")
    called: list[bool] = []

    async def repair(*a: Any, **kw: Any) -> Any:
        called.append(True)
        return a[2], a[3]

    async def chat(prompt: str, *, node: str = "") -> str:
        return "```python\nfor n in (8, 16, 32):\n    pass\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    eng._repair_ignored_replicate_seed = repair  # type: ignore[method-assign]
    asyncio.run(eng._node_implement({**STATE, "design": {}, "refine_extend": ["n=32"]}))  # type: ignore[arg-type]
    assert called == []
    assert isinstance(eng, Engine)
