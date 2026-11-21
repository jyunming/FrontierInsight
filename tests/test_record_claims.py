"""FI states the search outcome and the known-answer status itself; the paper's own words about them are only listed
(core/record_claims.py): never a rewrite, a take-out or a stop. Neutral toy topics only (a heat sink, a cooling cup)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import best_design_report as bdr
from core import paper_facts
from core import record_claims as rc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

NOT_BETTER = {"best": {"design": {"fin": 2.0}, "objective": 41.0}, "improvement": {"value": -1.5, "better": False},
              "check": {"verdict": "improvement_not_shown"}}
BETTER = {"best": {"design": {"fin": 3.0}, "objective": 30.0}, "improvement": {"value": 11.0, "better": True},
          "check": {"verdict": "verified"}}
NO_FEASIBLE = {"best": None, "improvement": None}
CHECK_NOT_SHOWN = {"verdict": "improvement_not_shown", "says": "the improvement disappears at finer settings"}
CHECK_OK = {"verdict": "verified", "passed": True}
OBJECTIVE = {"objective": {"quantity": "peak temperature", "unit": "K"}}

OVERCLAIM = ("# A heat sink study\n\n## Results\n\nThe optimised design lowers the peak temperature of the heat sink.\n\n"
             "Figure 1 shows the optimised design of the heat sink.\n\n"
             "![**Figure 1.** The optimised design of the heat sink.](figures/optimised.png)\n")
HONEST = ("# A heat sink study\n\n## Results\n\nThe search found no design better than the baseline, so there is no "
          "optimised design to report. The best design found was the baseline itself.\n\n"
          "## Limitations\n\nThe known-answer checks did not pass, so the result is not validated against known answers.\n\n"
          "![Peak temperature against fin count](figures/peak.png)\n")


def _kinds(paper: str, **records: Any) -> list[str]:
    return [e["kind"] for e in rc.unchecked(paper, **records)]


def test_a_paper_that_calls_a_design_optimised_against_a_record_that_says_it_was_not_is_listed() -> None:
    found = rc.unchecked(OVERCLAIM, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN)
    assert [e["kind"] for e in found] == [rc.SEARCH, rc.FIGURE]  # a sentence is listed once
    assert "The optimised design lowers the peak temperature" in found[0]["sentence"]
    assert "improvement not shown" in found[0]["says"] and "no optimised design was computed" in found[1]["says"]
    # The image's own caption is FI's, written from the figure's record: it is not read as the writer's claim.
    assert not any("**Figure 1.**" in e["sentence"] for e in found), found
    line = rc.disclosure(found[0])
    assert line.startswith("the paper's own words about whether the search beat the baseline")
    assert "not checked by FI" in line


def test_the_same_paper_is_fine_when_the_record_says_the_design_is_better() -> None:
    assert rc.unchecked(OVERCLAIM, best_design=BETTER, optimum_check=CHECK_OK) == []


def test_an_honest_paper_lists_nothing_whatever_the_record_says() -> None:
    for best, check in ((NOT_BETTER, CHECK_NOT_SHOWN), (NO_FEASIBLE, None), (BETTER, CHECK_OK), (None, None)):
        found = rc.unchecked(HONEST, best_design=best, optimum_check=check, oracle={"status": "went_on_failing"})
        assert found == [], (best, check)


@pytest.mark.parametrize("sentence", [
    "We did not find an optimised design.",
    "Whether the optimised design is better could not be shown.",
    "If the optimised design were better, the heat sink would run cooler.",
    "The best design found is the baseline itself.",
    "No improved design was found.",
    "The optimised design may be better than the baseline, but this was not shown.",
])
def test_a_sentence_that_denies_or_doubts_it_is_not_listed(sentence: str) -> None:
    assert rc.unchecked(f"# T\n\n{sentence}\n", best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN) == []


def test_a_hedge_exempts_a_claim_only_in_its_own_clause() -> None:
    def listed(sentence: str) -> list[dict[str, str]]:
        return rc.unchecked(f"# T\n\n{sentence}\n", best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN)

    assert listed("The optimised design may be better than the baseline, but this was not shown.") == []
    assert listed("If the optimised design were best, the heat sink would run cooler.") == []
    assert listed("It is unclear whether the optimised design is best.") == []
    assert listed("If the plots are read carefully, the optimised design is best.")
    assert listed("The optimised design is best; any reader can see it.")
    assert listed("The optimised design is best, although we did not test whether it is robust.")


def test_no_feasible_design_means_there_is_no_best_design_to_call_best() -> None:
    found = rc.unchecked("# T\n\nThe best design reduces the mass.\n", best_design=NO_FEASIBLE)
    assert found and "no design that meets every limit" in found[0]["says"]


def test_validated_is_listed_only_against_unconfirmed_or_failed_checks() -> None:
    paper = "# T\n\nThe simulation was validated against known analytic solutions.\n\nKnown-answer checks passed.\n"
    for status in ("went_on_failing", "stopped"):
        assert _kinds(paper, oracle={"status": status}) == [rc.KNOWN_ANSWER] * 2
    for oracle in ({"status": "ok"}, {"status": "warned"}, None, {}):
        assert rc.unchecked(paper, oracle=oracle) == []
    hedged = "# T\n\nThe simulation could not be validated against known analytic solutions.\n"
    assert rc.unchecked(hedged, oracle={"status": "went_on_failing"}) == []


def test_headings_tables_and_code_are_not_prose() -> None:
    paper = "# The optimised design\n\n| a | b |\n|---|---|\n| the optimised design | 1 |\n\n```\nthe optimised design\n```\n"
    assert rc.unchecked(paper, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN) == []


# --- the engine: nothing is asked for, taken out or stopped ------------------------------------------------------------


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **engine: Any) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    return Engine(Config(
        topic="a neutral topic about a heat sink", title="claims", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def _write_records(root: Path, best: dict[str, Any], check: dict[str, Any] | None, oracle: dict[str, Any] | None) -> None:
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "needs").mkdir(parents=True, exist_ok=True)
    (root / "results" / "best_design.json").write_text(json.dumps(best), encoding="utf-8")
    if check is not None:
        (root / "needs" / "OPTIMUM_CHECK.json").write_text(json.dumps(check), encoding="utf-8")
    if oracle is not None:
        (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(oracle), encoding="utf-8")


def test_the_paper_is_never_sent_back_or_cut_for_these_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On the old code a hit named ``record_contradiction`` made the router write the paper again twice and then cut the
    sentence; now no review hit of that name exists and the router keeps its usual route."""
    eng = _engine(tmp_path, monkeypatch)
    for gone in ("_record_contradiction_hits", "_take_out_contradicted", "_record_claims_left"):
        assert not hasattr(eng, gone), gone
    assert not hasattr(rc, "take_out") and not hasattr(rc, "found") and not hasattr(rc, "contradictions")
    review = {"verdict": "accept", "must_flag_hits": ["record_contradiction: the paper says validated"]}
    assert eng._route_after_review({"review": review, "iteration": 1, "record_rewrites": 1}) != "rewrite"


def _paper_on_disk(root: Path, text: str) -> Path:
    path = root / "paper" / "paper.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_quality_lists_the_words_and_the_paper_keeps_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import quality_report

    eng = _engine(tmp_path, monkeypatch)
    _write_records(eng.quest_root, NOT_BETTER, CHECK_NOT_SHOWN, {"status": "went_on_failing"})
    paper = _paper_on_disk(eng.quest_root,
                           OVERCLAIM + "\nThe simulation was validated against known analytic solutions.\n")
    before = paper.read_text(encoding="utf-8")
    text = quality_report.render(quality_report.build(eng.quest_root, running=False))
    for about in ("whether the search beat the baseline", "which design a figure shows",
                  "whether the results were checked against a known answer"):
        assert f"the paper's own words about {about} were not checked by FI" in text, about
    assert paper.read_text(encoding="utf-8") == before, "the sentences stay in the paper"
    assert "### The paper's own words FI did not check" in text
    # Still listed in full when the stage table shows only the first three of many paper problems.
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / ".fi" / "citations_removed.json").write_text(
        json.dumps({"removed": ["a1", "b2", "c3", "d4", "e5"]}), encoding="utf-8")
    crowded = quality_report.render(quality_report.build(eng.quest_root, running=False))
    assert crowded.count("were not checked by FI") >= 3 and "The optimised design lowers the peak" in crowded
    # A quest whose records say nothing of the kind lists nothing.
    other = _engine(tmp_path / "other", monkeypatch)
    _paper_on_disk(other.quest_root, OVERCLAIM)
    assert "were not checked by FI" not in quality_report.render(quality_report.build(other.quest_root, running=False))


# --- FI states the two facts itself --------------------------------------------------------------------------------------


def test_the_best_design_section_opens_with_what_the_search_found(tmp_path: Path) -> None:
    _write_records(tmp_path, {**NOT_BETTER, **OBJECTIVE}, CHECK_NOT_SHOWN, None)
    section = bdr.section(tmp_path)
    assert "**What the search found.** The search did not find a design shown to be better than the baseline" in section
    assert "improvement is not shown at finer numerical settings" in section
    (tmp_path / "needs" / "OPTIMUM_CHECK.json").unlink()
    flat = {**NOT_BETTER, **OBJECTIVE, "check": {}}
    _write_records(tmp_path, flat, None, None)
    assert ("**What the search found.** The search did not find a design better than the baseline; the best design found "
            "is no better than it.") in bdr.section(tmp_path)
    _write_records(tmp_path, {**BETTER, **OBJECTIVE}, CHECK_OK, None)
    assert "**What the search found.** The best design the search found is better than the baseline" in bdr.section(tmp_path)


def test_the_writer_is_told_fi_states_the_search_outcome(tmp_path: Path) -> None:
    _write_records(tmp_path, NOT_BETTER, CHECK_NOT_SHOWN, None)
    note = bdr.write_note(tmp_path)
    assert "do not state that yourself" in note and "do not call any design optimal" in note


def test_the_writer_is_told_fi_states_the_known_answer_status(tmp_path: Path) -> None:
    from core import accepted_checks

    entry = {"name": "cooling_curve", "expected": 1.0, "tolerance": 0.05, "measured": 0.5, "by": accepted_checks.AUTOMATIC}
    record = {"status": accepted_checks.WENT_ON, "went_on": [entry]}
    note = accepted_checks.failing_disclosure(record)
    assert "FI states each of these checks itself in the limitations" in note and "Do not write" in note
    _write_records(tmp_path, NOT_BETTER, None, record)
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps({"protocol": {"oracles": [
        {"name": "cooling_curve", "check": "The cup cools by the known law. Sized by the plan."}]}}), encoding="utf-8")
    items = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.KNOWN_ANSWER]
    assert items and "did not pass" in items[0].sentence and "not validated against it" in items[0].sentence
    section = paper_facts.writer_section(tmp_path)
    assert "Known answers: FI states in that list which checks" in section
    assert "Do not state in your own sentence" in section


def test_a_search_that_did_not_beat_the_baseline_gets_fi_s_sentence_in_the_paper(tmp_path: Path) -> None:
    """The known positive: the record says the search did not beat the baseline, and the paper carries FI's sentence once."""
    _write_records(tmp_path, {**NOT_BETTER, **OBJECTIVE}, CHECK_NOT_SHOWN, None)
    paper = bdr.mark_paper("# A heat sink study\n\n## Results\n\nThe writer's text.\n\n## Discussion\n\nMore.\n",
                           bdr.section(tmp_path))
    assert paper.count("**What the search found.**") == 1 and "did not find a design shown to be better" in paper


def test_the_outcome_line_follows_the_reported_design_and_the_check_s_own_words() -> None:
    qualified = {"verdict": "verified", "improvement": {"value": 2.0, "beyond_numerical_error": False}}
    line = bdr.search_outcome(BETTER, qualified)
    assert "above the plan's threshold" in line and "not shown to be more than their numerical error" in line
    assert "holds" not in line
    # The check replaced the search's best design with one that is no better: the line is about that design.
    replaced = {"verdict": "verified", "best": {"design": {"fin": 2.5}}, "improvement": {"value": -0.5}}
    assert "no better than it" in bdr.search_outcome(BETTER, replaced)
    assert bdr.search_outcome(None, None) == "" and "no design that meets every limit" in bdr.search_outcome(NO_FEASIBLE, None)


def test_every_sentence_of_each_kind_is_listed_and_the_paper_is_unchanged() -> None:
    paper = ("# A heat sink study\n\n## Results\n\nThe optimised design lowers the peak temperature of the heat sink.\n\n"
             "The optimised design also lowers the mass of the heat sink.\n\n"
             "![The optimised design of the heat sink.](figures/a.png)\n\n"
             "Figure 2 shows the optimised design of the cooling cup.\n\n"
             "The results are validated against known answers.\n\n"
             "The code is verified against known answers.\n")
    before = paper
    found = rc.unchecked(paper, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN,
                         oracle={"status": "went_on_failing"})
    kinds = [e["kind"] for e in found]
    assert kinds.count(rc.SEARCH) == 2 and kinds.count(rc.FIGURE) == 1 and kinds.count(rc.KNOWN_ANSWER) == 2, found
    sentences = " ".join(e["sentence"] for e in found)
    for part in ("lowers the peak temperature", "also lowers the mass", "validated against known answers",
                 "verified against known answers"):
        assert part in sentences, part
    assert paper == before


def test_the_outcome_line_opens_the_section_once_when_no_design_meets_every_limit(tmp_path: Path) -> None:
    _write_records(tmp_path, {**NO_FEASIBLE, **OBJECTIVE}, None, None)
    section = bdr.section(tmp_path)
    line = "**What the search found.**"
    assert section.count(line) == 1, section
    head = [x for x in section.splitlines() if x.strip()]
    assert head[2].startswith(line), head[:4]
    assert "no design that meets every limit" in head[2]
    assert section.lower().count("meets every limit") == 1, section  # the outcome is stated once
