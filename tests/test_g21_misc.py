"""Group G21 (a model-read objection never costs a rewrite twice) and its neighbours: the reviewer's repeated reason, the
figure-caption instruction to the reviewer, and the power limit of the standard-error check.  No model, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import disclosure, paper_facts, trial_checks
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    PausesConfig, ProviderConfig,
)
from core.engine import Engine, _format_figure_check


def _engine(tmp_path: Path, *, max_iterations: int = 3) -> Engine:
    eng = Engine(Config(
        topic="Settling time of a damped oscillator", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=max_iterations, review_loop=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        pauses=PausesConfig(review="off"),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    return eng


# --- L170: a repeated reason of a revise verdict costs no second rewrite -----------------------------------------------

def test_the_same_reason_again_does_not_cost_another_rewrite(tmp_path: Path) -> None:
    """Old code: a revise verdict with no must-flag was a rewrite every round up to engine.max_iterations, however often the
    reviewer gave the very same reason (only objection NAMES were de-duplicated, free-text reasons were not)."""
    eng = _engine(tmp_path)
    first = {"verdict": "revise", "must_flag_hits": [], "blocking": "The gain is only shown for one damping value."}
    eng._note_design_objections({"iteration": 0}, first)  # type: ignore[arg-type]
    assert "reason_repeated" not in first
    assert eng._route_after_review({"review": first, "iteration": 1}) == "rewrite"  # type: ignore[arg-type]
    # the next round: the same reason, wording and case aside
    again = {"verdict": "revise", "must_flag_hits": [], "blocking": "  the gain is only shown for one damping value "}
    eng._note_design_objections({"iteration": 1}, again)  # type: ignore[arg-type]
    assert again["reason_repeated"] is True
    assert eng._route_after_review({"review": again, "iteration": 2}) == "done"  # type: ignore[arg-type]


def test_a_repeated_reason_keeps_revise_and_is_stated_as_unresolved(tmp_path: Path) -> None:
    """Review finding: the same reason coming back means the rewrite did not answer it. The verdict stays revise (never
    accept), no further rewrite is spent, and the reason is in the quality report and the paper's Limitations as unresolved
    (old code recorded it only as an objection that was 'kept', with no sign it was still standing)."""
    from core import frozen_protocol

    frozen_protocol.freeze(tmp_path, {"grid": {"n": [5]}, "runs_per_setting": 2}, approved_by="test", source="test")
    eng = _engine(tmp_path)
    reason = "The gain is only shown for one damping value."
    eng._note_design_objections({"iteration": 0}, {"verdict": "revise", "must_flag_hits": [], "blocking": reason})  # type: ignore[arg-type]
    assert "still asked for a revision" not in disclosure.run_record(tmp_path)
    again = {"verdict": "revise", "must_flag_hits": [], "blocking": reason}
    eng._note_design_objections({"iteration": 1}, again)  # type: ignore[arg-type]
    assert eng._route_after_review({"review": again, "iteration": 2}) == "done"  # type: ignore[arg-type]
    assert again["verdict"] == "revise"
    assert "still asked for a revision" in disclosure.run_record(tmp_path) and "one damping value" in disclosure.run_record(tmp_path)
    [item] = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.UNRESOLVED_REVIEW]
    assert "one damping value" in item.sentence and "not resolved" in item.sentence


def test_the_delivered_paper_carries_the_unresolved_reason(tmp_path: Path) -> None:
    """Review finding: the item was built but nothing placed it after the last review, so paper.md lacked the sentence.
    Old code: the saved paper after the repeated-reason review had no such sentence."""
    from core import frozen_protocol

    frozen_protocol.freeze(tmp_path, {"grid": {"n": [5]}, "runs_per_setting": 2}, approved_by="test", source="test")
    eng = _engine(tmp_path)
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text("\n".join(["# Title", "", "## Results", "", "Text.", "", "## Limitations", "", "One limit.", ""]),
                     encoding="utf-8")
    state: Any = {"paper_md": str(paper), "iteration": 1}
    reason = "The gain is only shown for one damping value."
    eng._note_design_objections(state, {"verdict": "revise", "must_flag_hits": [], "blocking": reason})
    assert "damping value" not in paper.read_text(encoding="utf-8")
    eng._note_design_objections(state, {"verdict": "revise", "must_flag_hits": [], "blocking": reason})
    text = paper.read_text(encoding="utf-8")
    assert "one damping value" in text and "not resolved" in text and "{{" not in text


def test_a_new_reason_or_a_measured_hit_still_costs_a_rewrite(tmp_path: Path) -> None:
    """Known positive: a different reason is a new objection, and a must-flag hit is never skipped on a repeated reason."""
    eng = _engine(tmp_path)
    eng._note_design_objections({"iteration": 0}, {"verdict": "revise", "must_flag_hits": [], "blocking": "Reason one."})  # type: ignore[arg-type]
    other = {"verdict": "revise", "must_flag_hits": [], "blocking": "A different reason."}
    eng._note_design_objections({"iteration": 1}, other)  # type: ignore[arg-type]
    assert "reason_repeated" not in other
    assert eng._route_after_review({"review": other, "iteration": 2}) == "rewrite"  # type: ignore[arg-type]
    with_hit = {"verdict": "revise", "must_flag_hits": ["unsupported_claim"], "blocking": "Reason one."}
    eng._note_design_objections({"iteration": 1}, with_hit)  # type: ignore[arg-type]
    assert "reason_repeated" not in with_hit
    assert eng._route_after_review({"review": with_hit, "iteration": 2}) == "rewrite"  # type: ignore[arg-type]


def test_reason_key_reads_two_wordings_of_one_sentence_alike() -> None:
    assert disclosure.reason_key(" Too FEW  trials. ") == disclosure.reason_key("too few trials")
    assert disclosure.reason_key("too few trials") != disclosure.reason_key("too many trials")
    assert disclosure.reason_key(None) == ""


# --- L46: a caption that only names a series flat in some panels is not a must-fix -------------------------------------

_SOME_PANELS = {
    "file": "sizes.png",
    "axes": [
        {"title": "a=1", "ylabel": "settle_time", "ylim": [0, 1],
         "series": [{"label": "reference", "min": 0.1, "max": 0.7, "shows": "yes"}]},
        {"title": "a=3", "ylabel": "", "ylim": [0, 1],
         "series": [{"label": "reference", "min": 0.9, "max": 0.9, "shows": "flat"}]},
    ],
}


def test_a_series_flat_in_some_panels_is_not_made_a_must_fix_for_the_reviewer() -> None:
    """Old code told the reviewer 'You MUST add a figure_caption entry ... and set verdict=revise' for a caption that
    names a series drawn flat in one panel and varying in the other (the caption said nothing false); the model obeyed and
    the review's must-fix was false."""
    block = _format_figure_check("![Settling time against the reference.](figures/sizes.png)",
                                 {"figure_records": {"sizes.png": _SOME_PANELS}})
    assert "in only some panels" in block
    assert "You MUST" not in block and "Do NOT add them to must_flag_hits" in block


def test_a_caption_that_names_a_series_not_drawn_at_all_is_still_a_must_fix() -> None:
    """Known positive: a series the figure does not show at all is something the caption says falsely."""
    record = {"file": "g.png", "axes": [{"title": "", "ylabel": "gain", "ylim": [0, 1],
                                         "series": [{"label": "ghost", "min": 0.0, "max": 0.0, "shows": "no"}]}]}
    block = _format_figure_check("![The ghost curve.](figures/g.png)", {"figure_records": {"g.png": record}})
    assert block.startswith("CAPTIONS THAT DESCRIBE WHAT THEIR FIGURE DOES NOT SHOW (1):")
    assert "You MUST add a `figure_caption` entry" in block


# --- L293: the power limit of the standard-error check is said in the paper --------------------------------------------

def test_the_detection_floor_shrinks_with_more_trials() -> None:
    assert trial_checks.ratio_detection_floor(19) > trial_checks.ratio_detection_floor(19 * 9) > trial_checks.RATIO_HIGH
    # 20 trials a setting over 9 settings, pooled: a standard error 1.4 times too small is below what could be flagged
    assert 1.4 < trial_checks.ratio_detection_floor(19 * 9) < 1.8


def _record(root: Path, quantity: dict[str, Any]) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    trial_checks.write_record(root, {"settings": 9, "calibration": {"quantities": [quantity], "checked": True,
                                                                     "not_checked": []}})


def _root(tmp_path: Path) -> Path:
    from core import frozen_protocol

    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    frozen_protocol.freeze(root, {"grid": {"n": [5, 10]}, "runs_per_setting": 20}, approved_by="test", source="test")
    return root


def _ratio_quantity(**more: Any) -> dict[str, Any]:
    return {"kind": "se_ratio", "label": "gain", "names": ["gain_hat", "gain_se"], "judged": 9, "below": 0, "gross": 0,
            "fault": False, "detects_from": 1.668, "settings": [{"key": f"n={i}", "value": 1.0} for i in range(9)], **more}


def test_a_passed_standard_error_check_says_how_small_a_miss_it_could_not_see(tmp_path: Path) -> None:
    """Old code: a standard error 1.4 times too small passed the check at 20 trials a setting and the paper said nothing
    of the check's power."""
    root = _root(tmp_path)
    _record(root, _ratio_quantity())
    [item] = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CALIBRATION]
    assert "1.7" in item.sentence and "not ruled out" in item.sentence
    assert "gain" in item.sentence


def test_no_power_item_for_a_flagged_error_or_a_record_without_the_limit(tmp_path: Path) -> None:
    """Known positive/negative: a standard error already flagged has its own item (not this one), and an older record that
    holds no limit gives none."""
    root = _root(tmp_path)
    _record(root, _ratio_quantity(below=2))
    sentences = [i.sentence for i in paper_facts.required_items(root) if i.kind == paper_facts.CALIBRATION]
    assert len(sentences) == 1 and "differs from the spread" in sentences[0]
    quantity = _ratio_quantity()
    del quantity["detects_from"]
    _record(root, quantity)
    assert [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CALIBRATION] == []


def test_the_summary_keeps_the_limit_of_a_judged_ratio() -> None:
    """The record written from a real judgement carries the limit (nine settings, twenty trials each)."""
    import random

    rng = random.Random(3)
    cells = []
    for i in range(9):
        trials = {str(t): {"est": rng.gauss(0.0, 1.0), "se": 1.0} for t in range(20)}
        cells.append({"key": f"n={i}", "trials": trials})
    q = trial_checks.Quantity(kind="se_ratio", label="est", names=["est", "se"])
    trial_checks._judge_ratio(q, [(c["key"], [t["est"] for t in c["trials"].values()],
                                   [t["se"] for t in c["trials"].values()]) for c in cells])
    assert q.detects_from is not None and 1.4 < q.detects_from < 1.8
    assert json.dumps(q.detects_from)
