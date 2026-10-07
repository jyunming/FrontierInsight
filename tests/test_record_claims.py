"""The paper may not claim what FI's own records contradict (core/record_claims.py): the hits, the rewrite loop and the
honest stop. Neutral toy topics only (a heat sink, a cooling cup)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import record_claims as rc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _hit_name, _hits_need_only_a_rewrite

NOT_BETTER = {"best": {"design": {"fin": 2.0}, "objective": 41.0}, "improvement": {"value": -1.5, "better": False},
              "check": {"verdict": "improvement_not_shown"}}
BETTER = {"best": {"design": {"fin": 3.0}, "objective": 30.0}, "improvement": {"value": 11.0, "better": True},
          "check": {"verdict": "verified"}}
NO_FEASIBLE = {"best": None, "improvement": None}
CHECK_NOT_SHOWN = {"verdict": "improvement_not_shown", "says": "the improvement disappears at finer settings"}
CHECK_OK = {"verdict": "verified", "passed": True}

OVERCLAIM = ("# A heat sink study\n\n## Results\n\nThe optimised design lowers the peak temperature of the heat sink.\n\n"
             "![The optimised design of the heat sink](figures/optimised.png)\n")
HONEST = ("# A heat sink study\n\n## Results\n\nThe search found no design better than the baseline, so there is no "
          "optimised design to report. The best design found was the baseline itself.\n\n"
          "## Limitations\n\nThe known-answer checks did not pass, so the result is not validated against known answers.\n\n"
          "![Peak temperature against fin count](figures/peak.png)\n")


def test_a_paper_that_calls_a_design_optimised_against_a_record_that_says_it_was_not_is_flagged() -> None:
    hits = rc.contradictions(OVERCLAIM, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN, figures_on_disk={"optimised.png"})
    assert hits and all(h.startswith("record_contradiction:") for h in hits)
    assert any("The optimised design lowers the peak temperature" in h and "improvement not shown" in h for h in hits)
    assert any("captioned" in h and "no optimised design was computed" in h for h in hits), hits


def test_the_same_paper_is_fine_when_the_record_says_the_design_is_better() -> None:
    assert rc.contradictions(OVERCLAIM, best_design=BETTER, optimum_check=CHECK_OK, figures_on_disk={"optimised.png"}) == []


def test_an_honest_paper_gets_no_hit_whatever_the_record_says() -> None:
    for best, check in ((NOT_BETTER, CHECK_NOT_SHOWN), (NO_FEASIBLE, None), (BETTER, CHECK_OK), (None, None)):
        assert rc.contradictions(HONEST, best_design=best, optimum_check=check, oracle={"status": "went_on_failing"},
                                 figures_on_disk={"peak.png"}) == [], (best, check)


@pytest.mark.parametrize("sentence", [
    "We did not find an optimised design.",
    "Whether the optimised design is better could not be shown.",
    "If the optimised design were better, the heat sink would run cooler.",
    "The best design found is the baseline itself.",
    "No improved design was found.",
    "The optimised design may be better than the baseline, but this was not shown.",
])
def test_a_sentence_that_denies_or_doubts_it_is_not_flagged(sentence: str) -> None:
    assert rc.contradictions(f"# T\n\n{sentence}\n", best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN) == []


def test_no_feasible_design_means_there_is_no_best_design_to_call_best() -> None:
    hits = rc.contradictions("# T\n\nThe best design reduces the mass.\n", best_design=NO_FEASIBLE)
    assert hits and "no design that meets every limit" in hits[0]


def test_a_figure_the_run_did_not_draw_is_flagged_and_one_it_drew_is_not() -> None:
    paper = "# T\n\nSee the plot.\n\n![Cooling curve](figures/cooling.png)\n\n![Other](figures/other.png)\n"
    hits = rc.contradictions(paper, figures_on_disk={"cooling.png"})
    assert len(hits) == 1 and "figures/other.png" in hits[0] and "drew no figure of that name" in hits[0]
    assert rc.contradictions(paper, figures_on_disk={"cooling.png", "other.png"}) == []
    assert rc.contradictions(paper, figures_on_disk=None) == [], "figures not known: nothing judged"


def test_validated_is_flagged_only_against_unconfirmed_or_failed_checks() -> None:
    paper = "# T\n\nThe simulation was validated against known analytic solutions.\n\nKnown-answer checks passed.\n"
    for status in ("went_on_failing", "stopped"):
        hits = rc.contradictions(paper, oracle={"status": status})
        assert hits and "unconfirmed or failed" in hits[0]
    for oracle in ({"status": "ok"}, {"status": "warned"}, None, {}):
        assert rc.contradictions(paper, oracle=oracle) == []
    hedged = "# T\n\nThe simulation could not be validated against known analytic solutions.\n"
    assert rc.contradictions(hedged, oracle={"status": "went_on_failing"}) == []


def test_headings_tables_and_code_are_not_prose() -> None:
    paper = "# The optimised design\n\n| a | b |\n|---|---|\n| the optimised design | 1 |\n\n```\nthe optimised design\n```\n"
    assert rc.contradictions(paper, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN) == []


# --- the engine: records on disk, the rewrite loop, the stop ----------------------------------------------------------


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **engine: Any) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    return Engine(Config(
        topic="a neutral topic about a heat sink", title="claims", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def _write_records(eng: Engine, best: dict[str, Any], check: dict[str, Any] | None, oracle: dict[str, Any] | None) -> None:
    root = eng.quest_root
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "needs").mkdir(parents=True, exist_ok=True)
    (root / "results" / "best_design.json").write_text(json.dumps(best), encoding="utf-8")
    if check is not None:
        (root / "needs" / "OPTIMUM_CHECK.json").write_text(json.dumps(check), encoding="utf-8")
    if oracle is not None:
        (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(oracle), encoding="utf-8")


def test_the_engine_reads_the_records_and_the_figures_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch)
    _write_records(eng, NOT_BETTER, CHECK_NOT_SHOWN, {"status": "went_on_failing"})
    (eng.quest_root / "figures").mkdir()
    (eng.quest_root / "figures" / "peak.png").write_bytes(b"x")
    hits = eng._record_contradiction_hits(OVERCLAIM, {})
    assert len(hits) >= 3 and any("optimised.png" in h for h in hits)
    assert eng._record_contradiction_hits(HONEST, {}) == []
    # A quest with no search and no checks recorded flags nothing in a paper that says the same words.
    other = _engine(tmp_path / "other", monkeypatch)
    assert other._record_contradiction_hits(OVERCLAIM.replace("![", "[").replace("](", " ("), {}) == []


def test_such_a_hit_is_a_text_problem_named_as_one() -> None:
    hit = rc.contradictions(OVERCLAIM, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN)[0]
    assert _hit_name(hit) == rc.HIT and _hits_need_only_a_rewrite([hit])


def test_the_paper_is_written_again_twice_and_then_the_quest_stops_with_no_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch)
    hit = rc.contradictions(OVERCLAIM, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN)[0]
    review = {"verdict": "accept", "must_flag_hits": [hit]}
    # Even a quest whose iteration budget is spent (max_iterations is 1) gets its rewrites.
    for seen, route in ((1, "rewrite"), (2, "rewrite"), (3, "stuck")):
        assert eng._route_after_review({"review": review, "iteration": 1, "record_rewrites": seen}) == route
    # Without the hit the route is the usual one.
    assert eng._route_after_review({"review": {"verdict": "accept", "must_flag_hits": []}, "iteration": 1}) not in ("rewrite", "stuck")


@pytest.mark.asyncio
async def test_the_stop_says_what_the_paper_claimed_and_keeps_no_paper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    eng = _engine(tmp_path, monkeypatch)
    (eng.quest_root / "paper").mkdir(parents=True)
    (eng.quest_root / "paper" / "paper.md").write_text(OVERCLAIM, encoding="utf-8")
    hit = rc.contradictions(OVERCLAIM, best_design=NOT_BETTER, optimum_check=CHECK_NOT_SHOWN)[0]
    out = await eng._node_stuck_no_findings({"review": {"verdict": "accept", "must_flag_hits": [hit]}})
    record = out["stuck"]
    assert "still claims what FI's own records contradict" in record["problem"] and "not hand it over" in record["problem"]
    assert any("The optimised design lowers the peak temperature" in t for t in record["tried"])
    assert (eng.quest_root / "needs" / "STUCK.json").is_file()
    assert not (eng.quest_root / "paper" / "paper.md").exists(), "the paper is set aside, not delivered"
    assert "No paper was written" in capsys.readouterr().out


def test_a_run_that_drew_no_figure_does_not_judge_the_papers_figures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch)
    (eng.quest_root / "figures").mkdir(parents=True)
    paper = "# T\n\nSee the plot.\n\n![Cooling curve](figures/cooling.png)\n"
    assert eng._record_contradiction_hits(paper, {}) == []
    (eng.quest_root / "figures" / "other.png").write_bytes(b"x")
    assert len(eng._record_contradiction_hits(paper, {})) == 1
