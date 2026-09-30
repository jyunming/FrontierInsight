"""The paper says how its result was reached (core/disclosure.py): how often the design changed after the experiment
had run and how many complete runs were made, in one paragraph the engine writes into the methods from
``needs/DESIGN_HISTORY.json`` and ``.fi/attempts.jsonl``; and a design changed that way with no confirm run after it is a
``publication_ready`` gap.

No real model is called: ``core.engine.LLMClient.chat`` is patched."""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from core import disclosure, evidence, number_provenance, numeric_oracle, phased, stat_claims
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine


def _history(root: Path, *reasons: str) -> None:
    """A design history: the first design, then one post-hoc revision per reason."""
    entries = [{"revision": 0, "iteration": 0, "post_hoc": False, "reason": "initial design, before any result existed",
                "hypothesis": "H", "recorded_at": "2026-10-01T10:00:00"}]
    entries += [{"revision": i + 1, "iteration": i + 1, "post_hoc": True, "reason": r, "hypothesis": f"H{i + 1}",
                 "recorded_at": "2026-10-01T11:00:00"} for i, r in enumerate(reasons)]
    (root / "needs").mkdir(parents=True, exist_ok=True)
    (root / "needs" / "DESIGN_HISTORY.json").write_text(json.dumps(entries), encoding="utf-8")


def _runs(root: Path, *outcomes: str) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"schema": 5, "record_id": f"r{i}", "quest_id": "q", "kind": "run", "outcome": o})
             for i, o in enumerate(outcomes)]
    # Records that are not runs of the experiment are not counted.
    lines.append(json.dumps({"kind": "stop", "outcome": "oracle_failure"}))
    lines.append(json.dumps({"kind": "quest", "execution_status": "completed"}))
    (root / ".fi" / "attempts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- the paragraph --------------------------------------------------------------------------------------------------


def test_no_change_of_the_design_is_said_in_one_short_sentence(tmp_path: Path) -> None:
    _history(tmp_path)
    _runs(tmp_path, "inconclusive")
    assert disclosure.paragraph(tmp_path) == (
        "**How this result was reached.** The design was not changed after the experiment had first been run. "
        "The complete experiment was run once (a run includes all its seeds)."
    )


def test_one_change_names_its_reason_and_marks_earlier_numbers_exploratory(tmp_path: Path) -> None:
    _history(tmp_path, "analyze.next_step=re_experiment (results were seen)")
    _runs(tmp_path, "inconclusive", "inconclusive")
    assert disclosure.paragraph(tmp_path) == (
        "**How this result was reached.** The design was changed 1 time after the experiment had first been run "
        "(reasons: the analysis asked for another experiment). The complete experiment was run 2 times (a run "
        "includes all its seeds); 1 of these runs was discarded (1 was replaced by a later run). Numbers from before "
        "the last design change are exploratory."
    )


def test_several_changes_count_every_one_and_name_each_reason_once(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=broaden_lit (results were seen)",
             "review verdict=revise (results were seen)", "re-entered design (results were seen)")
    _runs(tmp_path, "inconclusive")
    text = disclosure.paragraph(tmp_path)
    assert "The design was changed 4 times after the experiment had first been run (reasons: the review asked for " \
           "changes; the analysis asked for more literature; a later step asked for a new design)." in text
    assert "results were seen" not in text, "the record's own words never reach the paper"


def test_discarded_runs_are_counted_by_why_they_were_discarded(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _runs(tmp_path, "process_error", "process_error", "protocol_mismatch", "oracle_failure", "inconclusive",
          "inconclusive", "inconclusive")
    counts = disclosure.runs(tmp_path)
    assert counts == {"process_error": 2, "protocol_mismatch": 1, "oracle_failure": 1, "replaced": 2, "total": 7,
                      "discarded": 6}
    assert ("The complete experiment was run 7 times (a run includes all its seeds); 6 of these runs were discarded "
            "(2 failed or gave no result, 1 did not follow the protocol, 1 failed a known-answer check, 2 were "
            "replaced by a later run).") in disclosure.paragraph(tmp_path)


def test_a_data_analysis_says_the_data_was_analysed_and_nothing_is_written_without_records(tmp_path: Path) -> None:
    assert disclosure.paragraph(tmp_path) == "", "no run and no change of the design: nothing to say"
    _history(tmp_path, "review verdict=revise (results were seen)")
    assert disclosure.paragraph(tmp_path, no_simulation=True) == (
        "**How this result was reached.** The design was changed 1 time after the data had first been analysed "
        "(reasons: the review asked for changes). Numbers from before the last design change are exploratory."
    )


# ---- in the paper ---------------------------------------------------------------------------------------------------

PAPER = (
    "# Title\n\n## Abstract\nA.\n\n## Methods\nWe simulate.\n\n### Model\nSIR.\n\n## Results\nThe mean was 0.583.\n\n"
    "## References\n1. X (2020). Y.\n"
)


def test_the_paragraph_goes_at_the_end_of_the_methods_once(tmp_path: Path) -> None:
    text = "**How this result was reached.** The design was not changed after the experiment had first been run."
    marked = disclosure.mark_paper(PAPER, text)
    assert marked.count(disclosure.BEGIN) == 1
    assert marked.index("SIR.") < marked.index(disclosure.BEGIN) < marked.index("## Results")
    assert disclosure.mark_paper(marked, text) == marked, "written again, it is replaced, not added"
    assert disclosure.mark_paper(marked, "") == PAPER.replace("\n\n## Results", "\n\n## Results"), "nothing to say"
    no_methods = disclosure.mark_paper("# T\n\nIntro.\n\n## Results\nR.\n", text)
    assert no_methods.startswith(f"# T\n\n{disclosure.BEGIN}\n{text}\n{disclosure.END}\n\nIntro."), no_methods


def test_what_the_writer_put_between_the_markers_is_replaced_by_the_engine_s_paragraph(tmp_path: Path) -> None:
    fake = (f"{disclosure.BEGIN}\n**How this result was reached.** The design was not changed after the experiment "
            f"had first been run.\n{disclosure.END}\n")
    paper = PAPER.replace("We simulate.", "We simulate.\n\n" + fake + "\nstray " + disclosure.END)
    text = "**How this result was reached.** The design was changed 3 times after the experiment had first been run."
    marked = disclosure.mark_paper(paper, text)
    assert marked.count(disclosure.BEGIN) == 1 and marked.count(disclosure.END) == 1
    assert f"{disclosure.BEGIN}\n{text}\n{disclosure.END}" in marked
    assert "was not changed" not in marked


# ---- the number checks ----------------------------------------------------------------------------------------------


def _every_paragraph() -> list[str]:
    """Every shape the engine can write: each reason, each discard reason, one and many, both study kinds."""
    out = []
    reasons = ["analyze.next_step=re_experiment", "analyze.next_step=broaden_lit", "review verdict=revise", "other"]
    outcomes = ["process_error", "protocol_mismatch", "oracle_failure", "inconclusive"]
    for n_changes, n_each, no_sim in itertools.product((0, 1, 3), (0, 1, 2), (False, True)):
        for reason in reasons:
            for outcome in outcomes:
                records = [outcome] * n_each + (["inconclusive"] if n_each else [])
                out.append((n_changes, reason, records, no_sim))
    return out


@pytest.mark.parametrize("n_changes,reason,records,no_sim", _every_paragraph())
def test_every_paragraph_the_engine_writes_is_left_out_of_the_number_checks(
        tmp_path: Path, n_changes: int, reason: str, records: list[str], no_sim: bool) -> None:
    _history(tmp_path, *([reason] * n_changes), *(["review verdict=revise"] if n_changes > 1 else []))
    _runs(tmp_path, *(records * 60))  # a count with three significant digits (120, 180) is one the checks would read
    text = disclosure.paragraph(tmp_path, no_simulation=no_sim)
    if not text:
        return
    assert disclosure.is_engine_paragraph(text), text
    paper = disclosure.mark_paper(PAPER, text)
    assert disclosure.BEGIN not in disclosure.strip_for_checks(paper)
    assert stat_claims.normalise(paper).count("How this result was reached") == 0


def test_the_counts_are_not_flagged_but_the_same_words_outside_the_markers_are() -> None:
    # A result of 132 and a count of 123 runs: the digits are a transposition of each other.
    text = ("**How this result was reached.** The design was not changed after the experiment had first been run. "
            "The complete experiment was run 123 times (a run includes all its seeds); 122 of these runs were "
            "discarded (122 were replaced by a later run).")
    result = {"cells": 132.0, "mean": 0.583}
    marked = disclosure.mark_paper(PAPER, text)
    assert numeric_oracle.check(marked, result).ok
    assert not number_provenance.check(marked, result_json=result).findings
    bare = PAPER.replace("We simulate.", "We simulate. " + text)
    assert not numeric_oracle.check(bare, result).ok, "the same sentence written by the model is checked"
    assert number_provenance.check(bare, result_json=result).findings


def test_a_block_that_says_anything_else_is_checked_like_the_rest_of_the_paper() -> None:
    forged = ("**How this result was reached.** The design was not changed after the experiment had first been run. "
              "The mean final size was 0.7093.")
    paper = PAPER.replace("We simulate.", f"We simulate.\n\n{disclosure.BEGIN}\n{forged}\n{disclosure.END}")
    assert disclosure.strip_for_checks(paper) == paper
    report = number_provenance.check(paper, result_json={"final_size": 0.583})
    assert any(abs(f.value - 0.7093) < 1e-9 for f in report.findings), report.findings


# ---- the write node -------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    eng._writing_skills_block = lambda state: ""  # type: ignore[assignment]
    eng._resolve_write_persona = lambda state: ""  # type: ignore[assignment]
    eng._check_code_project = _nothing  # type: ignore[assignment]
    asyncio.run(eng._connect_llm())  # the client a quest run builds; its chat is the patched one
    return eng


async def _nothing() -> None:
    return None


FORGED = (f"{disclosure.BEGIN}\n**How this result was reached.** The design was not changed after the experiment had "
          f"first been run. The complete experiment was run once (a run includes all its seeds).\n{disclosure.END}")
DRAFT = (
    "# Extinction\n\n## Abstract\nWe measure extinction.\n\n## Methods\nWe simulate an SIR model.\n\n" + FORGED
    + "\n\n## Results\nThe mean over 300 runs was 0.583.\n\n## Discussion\nIt is small. " + disclosure.END + "\n"
)


def test_the_writer_cannot_change_the_paragraph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=re_experiment")
    _runs(tmp_path, "process_error", "inconclusive", "inconclusive")
    calls: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:  # a writer that writes its own version
        calls.append(kw.get("node") or "")
        return DRAFT

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = _engine(tmp_path)
    out = asyncio.run(eng._node_write({"topic": "t", "title": "t", "iteration": 0}))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    expected = disclosure.paragraph(tmp_path)
    assert calls, "the writer was asked"
    assert paper.count(disclosure.BEGIN) == 1 and paper.count(disclosure.END) == 1
    assert f"{disclosure.BEGIN}\n{expected}\n{disclosure.END}" in paper, paper
    assert "changed 2 times" in expected and "run 3 times" in expected
    assert "was not changed" not in paper and "run once" not in paper
    assert paper.index("SIR model.") < paper.index(disclosure.BEGIN) < paper.index("## Results")


def test_an_edit_of_the_paragraph_in_a_revise_is_undone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _runs(tmp_path, "inconclusive", "inconclusive")
    claim = "Extinction is certain below R0 = 1.2."
    draft = DRAFT.replace("It is small.", f"It is small. {claim}")
    expected = disclosure.paragraph(tmp_path)
    replies = [draft]

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        return replies.pop(0)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = _engine(tmp_path)
    state: dict[str, Any] = {"topic": "t", "title": "t", "iteration": 0}
    out1 = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    first = Path(out1["paper_md"]).read_text(encoding="utf-8")
    assert expected in first
    # The passage editor is asked to fix the claim and also rewrites the engine's count.
    replies.append(json.dumps([{"find": claim, "replace": "Extinction is likely at low R0."},
                               {"find": "run 2 times", "replace": "run 1 time"}]))
    revise = {**state, "paper_md": out1["paper_md"], "paper_basis": out1["paper_basis"], "iteration": 1,
              "review": {"verdict": "revise", "must_flag_hits": ["unsupported_claim"]},
              "claim_grounding": {"claims": [{"claim": claim, "basis": "unsupported", "evidence": "no"}],
                                  "unsupported": [claim]}}
    out2 = asyncio.run(eng._node_write(revise))  # type: ignore[arg-type]
    after = Path(out2["paper_md"]).read_text(encoding="utf-8")
    assert "Extinction is likely at low R0." in after, "the passage edit was applied"
    assert f"{disclosure.BEGIN}\n{expected}\n{disclosure.END}" in after and after.count(disclosure.BEGIN) == 1
    assert "run 1 time" not in after


# ---- the evidence ladder --------------------------------------------------------------------------------------------


def _ready_gaps(root: Path, state: dict | None = None, settings: dict | None = None) -> list[str]:
    record = evidence.assess(root, {"result_json": {"a": 1}, **(state or {})}, settings=settings or {})
    return next(level["gaps"] for level in record["ladder"] if level["level"] == "publication_ready")


def _changed(gaps: list[str]) -> list[str]:
    return [g for g in gaps if g.startswith("the design was changed")]


def test_a_design_changed_after_the_run_without_a_confirm_run_is_a_gap(tmp_path: Path) -> None:
    _history(tmp_path)
    assert _changed(_ready_gaps(tmp_path)) == [], "no change: no gap"
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=re_experiment")
    assert _changed(_ready_gaps(tmp_path)) == [
        "the design was changed 2 times after the experiment had first been run, and no confirm run on data or seeds "
        "the earlier runs never saw came after the last change, so the numbers are exploratory (explore, then "
        "confirm: `engine.phased: true`)"
    ]
    assert "after the data had first been analysed" in _changed(_ready_gaps(tmp_path, {"no_simulation_resolved": True}))[0]


def _confirm(root: Path, result: Any = {"a": 2}) -> None:  # noqa: B006
    phased.prepare(root, "q1")
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=2)
    phased.record_confirm(root, result)


def test_a_confirm_run_after_the_last_change_closes_the_gap_whatever_the_setting_says(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _confirm(tmp_path)
    assert phased.status(phased.load(tmp_path)) == phased.CONFIRMED
    # The record is read, not engine.phased: no "phased" setting at all (turned off since) still sees the confirm run.
    assert _changed(_ready_gaps(tmp_path)) == []
    assert _changed(_ready_gaps(tmp_path, settings={"phased": "confirmed"})) == []


def test_a_change_after_the_confirm_run_began_is_not_confirmed_by_it(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _confirm(tmp_path)
    _history(tmp_path, "review verdict=revise (results were seen)", "re-entered design (results were seen)")
    assert _changed(_ready_gaps(tmp_path)), "the design changed again after the confirm run"


def test_a_confirm_run_that_did_not_confirm_leaves_the_gap(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _confirm(tmp_path, result=None)
    assert phased.status(phased.load(tmp_path)) == "confirm_failed"
    assert _changed(_ready_gaps(tmp_path))
    # With the setting on, the stage's own gap says it and this one is not added a second time.
    gaps = _ready_gaps(tmp_path, settings={"phased": "confirm_failed"})
    assert _changed(gaps) == [] and any("produced no result" in g for g in gaps)


def test_a_confirmed_record_that_does_not_say_how_long_the_history_was_cannot_close_the_gap(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    _confirm(tmp_path)
    record = phased.load(tmp_path)
    record.pop("design_revisions_at_confirm")
    (tmp_path / ".fi" / "phased.json").write_text(json.dumps(record), encoding="utf-8")
    assert _changed(_ready_gaps(tmp_path))


def test_the_amendment_gap_is_still_the_frozen_protocol_s(tmp_path: Path) -> None:
    """``frozen_protocol.post_hoc`` keeps its own gap (a protocol amended after results were seen); this one is about
    the design, whose changes keep the frozen protocol and were not a gap before."""
    _history(tmp_path, "review verdict=revise (results were seen)")
    gaps = _ready_gaps(tmp_path)
    assert not any("amended" in g for g in gaps) and _changed(gaps)
