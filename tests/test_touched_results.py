"""-381: an open code finding that changes a result the paper prints is marked in the paper by a sentence FI writes from its
records. Which results a finding changes comes from the call graph of the study's code (the function the finding records),
never from the words of the finding."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review, frozen_protocol, paper_facts, quality_report, shown_reach

PROTOCOL = {
    "form": "contract", "grid": {"noise": [0.1, 0.2]}, "runs_per_setting": 20, "constants": {"truth_a": 2.0, "truth_b": 5.0},
    "metrics": [
        {"id": "est_a", "kind": "mean", "estimand": "the first quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_a", "estimate": "est_a", "standard_error": "se_a", "interval": ["lo_a", "hi_a"], "level": 0.95}},
        {"id": "est_b", "kind": "mean", "estimand": "the second quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_b", "estimate": "est_b", "standard_error": "se_b", "interval": ["lo_b", "hi_b"], "level": 0.95}},
        {"id": "cover", "kind": "proportion", "estimand": "coverage", "unit": "one setting", "role": "physical_bound", "returned": "afterwards"},
        {"id": "corr_ab", "kind": "mean", "estimand": "correlation", "unit": "one setting", "role": "result", "returned": "afterwards"}]}
FILES = {
    "model.py": ("def helper_for_cover(x):\n    return x\n\n\ndef helper_for_trial(x):\n    return x\n\n\n"
                 "def helper_unused(x):\n    return x\n"),
    "derive.py": ("import model\n\n\ndef derive_cover(table):\n    return model.helper_for_cover(1)\n\n\n"
                  "def derive_corr_ab(table):\n    return 0.5\n"),
    "simulate.py": ("import model\n\n\ndef trial(cell, ctx):\n    return {'est_a': model.helper_for_trial(1), 'est_b': 2.0, 'se_a': 1.0, "
                    "'lo_a': 0.0, 'hi_a': 3.0}\n"),
}


def _quest(root: Path, issues: dict[str, Any]) -> Path:
    for rel, text in FILES.items():
        (root / "code" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / rel).write_text(text, encoding="utf-8")
    (root / "needs").mkdir(parents=True, exist_ok=True)
    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": PROTOCOL, "sha256": frozen_protocol.sha256(PROTOCOL)}
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")
    (root / "raw").mkdir(exist_ok=True)
    metrics = {k: {"values": [1.0], "trials": [0]} for k in ("est_a", "est_b", "se_a", "lo_a", "hi_a")}
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [{"cell": {"noise": 0.1}, "metrics": metrics}]}), encoding="utf-8")
    derived = {"schema": "fi.derived/v1", "complete": True, "not_computed": [], "bodies": {},
               "derived": {"corr_ab": {"status": "ok", "by": "derive_corr_ab", "inputs": ["est_a"], "by_setting": {"noise=0.1": 0.4}},
                           "cover": {"status": "ok", "by": "derive_cover", "inputs": [], "by_setting": {"noise=0.1": 0.94}}},
               "figures": []}
    (root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (root / "data" / "results" / "derived.json").write_text(json.dumps(derived), encoding="utf-8")
    (root / ".fi").mkdir(exist_ok=True)
    (root / ".fi" / "code_review.json").write_text(json.dumps({"issues": issues}), encoding="utf-8")
    return root


def _v(fid: str, text: str, file: str = "", function: str = "", **extra: Any) -> dict[str, Any]:
    return {"fid": fid, "text": text, "severity": "high", "kind": "logic", "file": file, "function": function, **extra}


def _found(*flagged: dict[str, Any], **other: list[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"missing": [], "different": [], "conflicts": [], "unshown": [], "never_run": [],
                           "not_rechecked": [], "flagged": list(flagged)}
    out.update(other)
    return out


def _touched(root: Path, monkeypatch: pytest.MonkeyPatch, *flagged: dict[str, Any], **other: list[Any]) -> dict[str, list[str]]:
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: _found(*flagged, **other))
    reach = shown_reach.Reach(root, PROTOCOL, [])
    return code_review.touched_results(root, root / "code", reach.touched_by)


def test_a_finding_marks_the_results_its_function_reaches_not_the_words_it_uses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, {})
    got = _touched(
        root, monkeypatch,
        _v("d", "the trial failed because n is wrong for est_b", "derive.py", "derive_cover"),
        _v("h", "a helper is off", "model.py", "helper_for_cover"),
        _v("t", "the helper of the trial is off", "model.py", "helper_for_trial"),
        _v("u", "est_a is computed wrongly here", "model.py", "helper_unused"),
        _v("w", "est_a and est_b are wrong"),
        _v("x", "est_a is wrong", "model.py", "no_such_function"))
    assert got["d"] == ["cover"], "a derive_<id> marks exactly that id; the words of the finding mark nothing"
    assert got["h"] == ["cover"]
    assert set(got["t"]) >= {"est_a", "est_b"} and "cover" not in got["t"]
    assert "u" not in got, "a helper nothing reaches marks none, whatever its text says"
    assert "w" not in got and "x" not in got, "no file and function FI finds in the code: nothing is marked"


def test_touched_results_leaves_out_what_cannot_change_a_printed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, {"no": {"changes_result": False}, "yes": {"changes_result": True},
                             "gate": {"changes_result": True, "at_gate": {"outcome": "left_out"}, "outcome": "open",
                                      "readings": [{"status": "open"}]}})
    got = _touched(
        root, monkeypatch,
        _v("no", "x", "derive.py", "derive_cover", changes_result=False),
        _v("yes", "x", "derive.py", "derive_cover"),
        _v("low", "x", "derive.py", "derive_cover", severity="low"),
        _v("gate", "x", "derive.py", "derive_cover"),
        never_run=[_v("nr", "x", "derive.py", "derive_cover")], conflicts=[_v("pc", "x", "derive.py", "derive_cover")])
    assert got == {"yes": ["cover"]}


def test_a_shown_fault_whose_test_errored_is_placed_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, {"g": {"changes_result": True, "file": "derive.py", "function": "derive_corr_ab", "text": "x",
                                   "gate_error": {"detail": "boom"}, "demonstrated": True, "failure_kind": "assertion",
                                   "outcome": "open", "readings": [{"status": "open"}]}})
    assert _touched(root, monkeypatch) == {"g": ["corr_ab"]}


def _printed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = _quest(tmp_path, {"f1": {"changes_result": True, "text": "cover is scaled twice"}})
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: _found(
        _v("f1", "cover is scaled twice", "derive.py", "derive_cover", changes_result=True),
        _v("f2", "an unrelated problem in the code", "model.py", "helper_unused", changes_result=True)))
    return root


PAPER = "# P\n\n## Methods\n{{run_facts}}\n\n## Results\nThe value is {{v:cover:mean_ci @ n=all}}.\n"


def test_fi_writes_a_sentence_for_a_touched_result_the_paper_prints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed(tmp_path, monkeypatch)
    text = paper_facts.run_facts_text(root, paper_md=PAPER)
    assert "comes from code a review flagged" in text and "cover is scaled twice" in text
    assert "comes from code" not in paper_facts.run_facts_text(root), "without the paper's text nothing is marked"
    assert "comes from code" not in paper_facts.run_facts_text(root, paper_md="# P\n\n{{v:est_a @ n=all}}\n")


def test_the_sentence_is_placed_even_when_it_is_the_only_run_fact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed(tmp_path, monkeypatch)
    draft = "# P\n\n## Methods\nWe did it.\n\n## Results\nThe value is {{v:cover:mean_ci @ n=all}}.\n"
    placed = paper_facts.place_blocks(draft, root)
    assert "{{run_facts}}" in placed
    assert "scaled twice" in paper_facts.render(placed, root).text


def test_the_sentence_is_in_the_rendered_paper_and_the_finding_is_met(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed(tmp_path, monkeypatch)
    out = paper_facts.render(PAPER, root).text
    paper_facts.keep_check(root, [], out)
    kept = paper_facts.kept_check(root)
    assert [u["id"] for u in kept["unmet_items"] if u["kind"] == paper_facts.CODE_FINDING] == ["finding-f2"]
    md = chr(10).join(quality_report._paper_record_lines({"paper_check": {"unmet_items": kept["unmet_items"]}}))
    lines = [ln for ln in md.splitlines() if ln.startswith("- Not in the paper's Limitations")]
    assert len(lines) == 1 and "`finding-f2`" in lines[0]


def test_the_sentence_counts_only_inside_fi_s_run_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _printed(tmp_path, monkeypatch)
    sentence = paper_facts.touched_sentences(root, PAPER)["f1"]
    stray = "# P\n\n## Methods\nWe did it.\n\n> " + sentence + "\n\n## Results\nThe value is {{v:cover:mean_ci @ n=all}}.\n"
    paper_facts.keep_check(root, [], stray)
    assert "finding-f1" in {u["id"] for u in paper_facts.kept_check(root)["unmet_items"]}
