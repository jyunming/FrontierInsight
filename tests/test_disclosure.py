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
        "**How this result was reached.** The design was not revised after the experiment had first been run. "
        "The complete experiment was run once (a run includes all its seeds)."
    )


def test_one_change_names_its_reason_and_marks_earlier_numbers_exploratory(tmp_path: Path) -> None:
    _history(tmp_path, "analyze.next_step=re_experiment (results were seen)")
    _runs(tmp_path, "inconclusive", "inconclusive")
    assert disclosure.paragraph(tmp_path) == (
        "**How this result was reached.** The design was revised 1 time after the experiment had first been run "
        "(reasons: the analysis asked for another experiment). The complete experiment was run 2 times (a run "
        "includes all its seeds); 1 of these runs was discarded (1 was replaced by a later run). Numbers from before "
        "the last design revision are exploratory."
    )


def test_several_changes_count_every_one_and_name_each_reason_once(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=broaden_lit (results were seen)",
             "review verdict=revise (results were seen)", "re-entered design (results were seen)")
    _runs(tmp_path, "inconclusive")
    text = disclosure.paragraph(tmp_path)
    assert "The design was revised 4 times after the experiment had first been run (reasons: the review asked for " \
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


def test_the_last_run_is_the_one_kept_even_when_a_check_only_warned_about_it(tmp_path: Path) -> None:
    """With ``oracle_check: warn`` a run that failed a known-answer check still goes on to the paper: it is kept."""
    _history(tmp_path)
    _runs(tmp_path, "oracle_failure")
    assert disclosure.runs(tmp_path)["discarded"] == 0
    assert disclosure.paragraph(tmp_path).endswith("The complete experiment was run once (a run includes all its seeds).")
    _runs(tmp_path, "process_error", "protocol_mismatch")
    assert "1 of these runs was discarded (1 failed or gave no result)." in disclosure.paragraph(tmp_path)


def test_a_design_done_again_before_anything_ran_is_not_a_revision_after_results(tmp_path: Path) -> None:
    import logging

    from core.engine import _append_design_revision

    log = logging.getLogger("test")
    h = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, log)
    h = _append_design_revision({"design_history": h}, {"hypothesis": "H2"}, tmp_path, log)  # a rerun from the plan
    assert h[1]["post_hoc"] is True and h[1]["after_results"] is False
    assert disclosure.revisions(tmp_path) == [] and disclosure.paragraph(tmp_path) == ""
    h = _append_design_revision({"design_history": h, "result_json": {"a": 1}, "review": {"verdict": "revise"}},
                                {"hypothesis": "H3"}, tmp_path, log)
    assert h[2]["after_results"] is True and len(disclosure.revisions(tmp_path)) == 1
    _runs(tmp_path, "inconclusive")
    h = _append_design_revision({"design_history": h}, {"hypothesis": "H4"}, tmp_path, log)
    assert h[3]["after_results"] is True, "a run in the attempt record counts as well"


def test_a_survey_says_the_literature_was_analysed_and_calls_nothing_exploratory(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    assert disclosure.paragraph(tmp_path, no_simulation=True, survey=True) == (
        "**How this result was reached.** The design was revised 1 time after the literature had first been analysed "
        "(reasons: the review asked for changes)."
    )
    assert disclosure.is_engine_paragraph(disclosure.paragraph(tmp_path, no_simulation=True, survey=True))


def test_a_data_analysis_says_the_data_was_analysed_and_nothing_is_written_without_records(tmp_path: Path) -> None:
    assert disclosure.paragraph(tmp_path) == "", "no run and no change of the design: nothing to say"
    _history(tmp_path, "review verdict=revise (results were seen)")
    assert disclosure.paragraph(tmp_path, no_simulation=True) == (
        "**How this result was reached.** The design was revised 1 time after the data had first been analysed "
        "(reasons: the review asked for changes). Numbers from before the last design revision are exploratory."
    )


# ---- in the paper ---------------------------------------------------------------------------------------------------

PAPER = (
    "# Title\n\n## Abstract\nA.\n\n## Methods\nWe simulate.\n\n### Model\nSIR.\n\n## Results\nThe mean was 0.583.\n\n"
    "## References\n1. X (2020). Y.\n"
)


def test_the_paragraph_goes_at_the_end_of_the_methods_once(tmp_path: Path) -> None:
    text = "**How this result was reached.** The design was not revised after the experiment had first been run."
    marked = disclosure.mark_paper(PAPER, text)
    assert marked.count(disclosure.BEGIN) == 1
    assert marked.index("SIR.") < marked.index(disclosure.BEGIN) < marked.index("## Results")
    assert disclosure.mark_paper(marked, text) == marked, "written again, it is replaced, not added"
    assert disclosure.mark_paper(marked, "") == PAPER.replace("\n\n## Results", "\n\n## Results"), "nothing to say"
    block = f"{disclosure.BEGIN}\n{text}\n{disclosure.END}"
    no_methods = disclosure.mark_paper("# T\n\n## Abstract\nA.\n\n## Introduction\nI.\n\n## Results\nR.\n", text)
    assert f"I.\n\n{block}\n\n## Results" in no_methods, "no methods: before the results"
    only_abstract = disclosure.mark_paper("# T\n\n## Abstract\nA.\n\n## Discussion\nD.\n", text)
    assert f"A.\n\n{block}\n\n## Discussion" in only_abstract, "no results either: after the abstract"
    assert disclosure.mark_paper("# T\n\nBody.\n", text).startswith(f"# T\n\n{block}\n\nBody."), "only a title"


@pytest.mark.parametrize("heading", [
    "## Methods", "## 2. Methods", "## II. Methods", "## Model and Methods", "## Numerical Methods",
    "## Data and Methods", "## Methodology", "## Simulation Setup", "## Materials and Methods", "## Approach",
])
def test_the_methods_section_is_found_under_the_names_papers_give_it(heading: str) -> None:
    paper = f"# T\n\n## Abstract\nA.\n\n## Introduction\nI.\n\n{heading}\nM.\n\n### Sub\nS.\n\n## Results\nR.\n"
    marked = disclosure.mark_paper(paper, "**How this result was reached.** x")
    assert marked.index("S.") < marked.index(disclosure.BEGIN) < marked.index("## Results"), marked


@pytest.mark.parametrize("paper", [
    "# A Comparison of Numerical Methods for SIR Models\n\n## Introduction\nI.\n\n## Methods\nM.\n\n## Results\nR.\n",
    "# T\n\n## Prior methods for extinction\nI.\n\n## Methods\nM.\n\n## Results\nR.\n",
    "# T\n\n## Methods\nM.\n\n## Results and comparison of methods\nR.\n\n## Discussion\nD.\n",
])
def test_a_title_or_another_section_that_names_methods_is_not_the_methods(paper: str) -> None:
    marked = disclosure.mark_paper(paper, "**How this result was reached.** x")
    assert marked.index("M.") < marked.index(disclosure.BEGIN) < marked.index("\n## Results"), marked


def test_a_data_analysis_rerun_from_the_design_after_its_analysis_is_a_revision_after_results(tmp_path: Path) -> None:
    import logging

    from core.engine import _append_design_revision

    log = logging.getLogger("test")
    h = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, log)
    (tmp_path / ".fi" / "previous" / "20261001-1200" / "data" / "auto_collected").mkdir(parents=True)
    h = _append_design_revision({"design_history": h}, {"hypothesis": "H2"}, tmp_path, log)
    assert h[1]["after_results"] is False, "an empty folder moved aside is not data that was analysed"
    (tmp_path / ".fi" / "previous" / "20261001-1200" / "data" / "auto_collected" / "gdp.csv").write_text("x,y\n")
    h = _append_design_revision({"design_history": h[:1]}, {"hypothesis": "H2"}, tmp_path, log)
    assert h[1]["after_results"] is True, "the analysed data a rerun moved aside shows the data had been analysed"


def test_a_rerun_from_the_plan_of_a_quest_that_never_ran_is_not_marked_as_after_results(tmp_path: Path) -> None:
    import logging

    from core import rerun_from
    from core.engine import _append_design_revision

    for name in ("paper", "code", "figures", "raw/oracle_check"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "oracle_check" / "case.json").write_text("{}")
    rerun_from.back_up(tmp_path, "plan")
    log = logging.getLogger("test")
    h = _append_design_revision({}, {"hypothesis": "H"}, tmp_path, log)
    h = _append_design_revision({"design_history": h}, {"hypothesis": "H2"}, tmp_path, log)
    assert h[1]["after_results"] is False


def test_the_shallowest_methods_heading_wins_and_a_rule_is_not_front_matter() -> None:
    paper = "---\n\n# T\n\n## Introduction\n### Approach\nI.\n\n## Methods\nM.\n\n## Results\nR.\n"
    marked = disclosure.mark_paper(paper, "**How this result was reached.** x")
    assert marked.index("M.") < marked.index(disclosure.BEGIN) < marked.index("## Results"), marked
    yaml = "---\ntitle: T\n---\n# T\n\nBody.\n"
    assert disclosure.mark_paper(yaml, "L").startswith("---\ntitle: T\n---\n# T\n\n" + disclosure.BEGIN)


def test_markers_rebuilt_from_pieces_do_not_survive(tmp_path: Path) -> None:
    """Taking one marker out must not join what is around it into a new one."""
    b, e = disclosure.BEGIN, disclosure.END
    forged = (f"## Introduction\n{b[:10]}{e}{b[10:]}\n**Fine.** The design was not revised after the experiment had "
              f"first been run.\n{e[:10]}{e}{e[10:]}\n")
    text = "**How this result was reached.** The design was revised 2 times after the experiment had first been run."
    marked = disclosure.mark_paper(PAPER.replace("## Results", forged + "\n## Results"), text)
    assert marked.count(b) == 1 and marked.count(e) == 1, marked
    for piece in (f"{b[:6]}{e}{b[6:]} x", f"{e[:6]}{b}{e[6:]} x", f"{b[:4]}{b[:9]}{e}{b[9:]}{b[4:]}"):
        left = disclosure.without_block(piece)
        assert b not in left and e not in left, (piece, left)


def test_a_passage_edit_that_drops_one_marker_leaves_one_paragraph_not_two() -> None:
    old = "**How this result was reached.** The design was revised 1 time after the experiment had first been run."
    edited = PAPER.replace("SIR.", f"SIR.\n\n{old.replace('1 time', '5 times')}\n{disclosure.END}")
    marked = disclosure.mark_paper(edited, old)
    assert marked.count("How this result was reached") == 1 and "5 times" not in marked


def test_crlf_text_is_stripped_too() -> None:
    text = "**How this result was reached.** The design was not revised after the experiment had first been run."
    paper = disclosure.mark_paper(PAPER, text).replace("\n", "\r\n")
    assert disclosure.BEGIN not in disclosure.strip_for_checks(paper)


def test_the_page_limit_trim_never_offers_the_engine_s_paragraph() -> None:
    from core import paper_trim

    text = ("**How this result was reached.** The design was revised 1 time after the experiment had first been run "
            "(reasons: the review asked for changes). Numbers from before the last design revision are exploratory.")

    def offered(body: str) -> list[str]:
        paper = ("# Title\n\n## Background\n\nIntro opens here.\n\n" + body + "\n\n## Discussion\nThe first sentence "
                 "opens it. Diffusion models are widely used across many fields of physics today.\n")
        return [s.text for s in paper_trim.candidates(paper, len(paper))]

    assert any("exploratory" in s for s in offered(text)), "unmarked, the same sentence would be offered"
    marked = offered(f"{disclosure.BEGIN}\n{text}\n{disclosure.END}")
    assert marked and not any("exploratory" in s or "How this result" in s for s in marked), marked


def test_what_the_writer_put_between_the_markers_is_replaced_by_the_engine_s_paragraph(tmp_path: Path) -> None:
    fake = (f"{disclosure.BEGIN}\n**How this result was reached.** The design was not revised after the experiment "
            f"had first been run.\n{disclosure.END}\n")
    paper = PAPER.replace("We simulate.", "We simulate.\n\n" + fake + "\nstray " + disclosure.END)
    text = "**How this result was reached.** The design was revised 3 times after the experiment had first been run."
    marked = disclosure.mark_paper(paper, text)
    assert marked.count(disclosure.BEGIN) == 1 and marked.count(disclosure.END) == 1
    assert f"{disclosure.BEGIN}\n{text}\n{disclosure.END}" in marked
    assert "was not revised" not in marked


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
    text = ("**How this result was reached.** The design was not revised after the experiment had first been run. "
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
    forged = ("**How this result was reached.** The design was not revised after the experiment had first been run. "
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


FORGED = (f"{disclosure.BEGIN}\n**How this result was reached.** The design was not revised after the experiment had "
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
    assert "revised 2 times" in expected and "run 3 times" in expected
    assert "was not revised" not in paper and "run once" not in paper
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
    return [g for g in gaps if g.startswith("the design was revised")]


def test_a_design_changed_after_the_run_without_a_confirm_run_is_a_gap(tmp_path: Path) -> None:
    _history(tmp_path)
    assert _changed(_ready_gaps(tmp_path)) == [], "no change: no gap"
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=re_experiment")
    assert _changed(_ready_gaps(tmp_path)) == [
        "the design was revised 2 times after the experiment had first been run, and no confirm run on data or seeds "
        "the earlier runs never saw is shown to have come after the last revision, so the numbers are exploratory "
        "(explore, then "
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


def test_a_revision_an_approved_amendment_made_is_that_amendment_s_gap_only(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    (tmp_path / "needs" / "PROTOCOL_AMENDMENT_1.json").write_text(json.dumps(
        {"n": 1, "changes": ["grid"], "prespecified": False, "results_seen_before_change": True}), encoding="utf-8")
    gaps = _ready_gaps(tmp_path)
    assert any("amended after results were seen" in g for g in gaps) and _changed(gaps) == []
    _history(tmp_path, "review verdict=revise (results were seen)", "analyze.next_step=re_experiment")
    assert _changed(_ready_gaps(tmp_path))[0].startswith("the design was revised 1 time"), "the one beyond it"


def test_a_survey_has_no_gap_and_a_data_analysis_is_not_told_to_turn_on_a_confirm_run(tmp_path: Path) -> None:
    _history(tmp_path, "review verdict=revise (results were seen)")
    assert _changed(_ready_gaps(tmp_path, {"no_simulation_resolved": True, "survey_mode_resolved": True})) == []
    (gap,) = _changed(_ready_gaps(tmp_path, {"no_simulation_resolved": True}))
    assert "engine.phased" not in gap and "has no confirm run" in gap


def test_the_amendment_gap_is_still_the_frozen_protocol_s(tmp_path: Path) -> None:
    """``frozen_protocol.post_hoc`` keeps its own gap (a protocol amended after results were seen); this one is about
    the design, whose changes keep the frozen protocol and were not a gap before."""
    _history(tmp_path, "review verdict=revise (results were seen)")
    gaps = _ready_gaps(tmp_path)
    assert not any("amended" in g for g in gaps) and _changed(gaps)
