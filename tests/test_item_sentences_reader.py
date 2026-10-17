"""The sentences FI puts in a paper's Limitations from its own records read as a paper's limitations, not as FI talking
about itself (round s6 graders: "FI-internal text in the paper"). The strings below are the real ones of the s5/s6 records.
No model is called."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import paper_facts, run_manifest

UNPRINTED = ("the metric `fdr_unc` is a mean over the trials that measured it, in 45000 trial(s) of FI's record: 45000 with a "
             "value and 0 with none (the measurement does not exist there), but the analysis's RESULT_JSON has no "
             "`fdr_unc_values` list at all — the analysis must print the per-trial values it averaged, so the claimed "
             "trial count can be checked against them")
OLD_CODE_FILES = ("The topic asks for the code in one file; FI keeps more: the simulation (simulate.py), the analysis "
                  "(experiment.py), the model's functions in a package and their unit tests, because it runs the simulation "
                  "and the analysis apart and checks the code function by function. The paper says the code is not in one file.")
FORBIDDEN = ("FI", "RESULT_JSON", "_values", "simulate.py", "experiment.py", "must print")


def _root(tmp_path: Path) -> Path:
    (tmp_path / ".fi").mkdir()
    (tmp_path / "needs").mkdir()
    return tmp_path


def _write(root: Path, rel: str, data: object) -> None:
    (root / rel).write_text(json.dumps(data), encoding="utf-8")


def _of(root: Path, kind: str) -> list[paper_facts.Item]:
    return [i for i in paper_facts.required_items(root) if i.kind == kind]


def _clean(text: str) -> None:
    import re

    for word in FORBIDDEN:
        assert not re.search(rf"\b{re.escape(word)}\b" if word == "FI" else re.escape(word), text), (word, text)


def test_the_page_overrun_reads_as_a_paper_sentence(tmp_path: Path) -> None:
    root = _root(tmp_path)
    paper_facts.write_page_record(root, 6, 4, final=True)
    [item] = _of(root, paper_facts.PAGE_LIMIT)
    assert item.sentence == "This paper is 6 pages, over the 4-page limit set for it."


def test_the_code_files_sentence_is_made_from_the_layout_not_the_stored_prose(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root, ".fi/topic_covered.json", {"not_done": [{"name": "code in the files the topic asked for",
                                                          "sentence": OLD_CODE_FILES, "kind": "code_files"}]})
    _write(root, ".fi/code_layout.json", {"shape": "package", "topic_code_files": 1})
    [item] = _of(root, paper_facts.NOT_DONE)
    assert item.sentence == ("The code is kept in separate parts (the simulation, the analysis, the model's functions and "
                             "their tests), not in the single file the topic asked for.")
    _clean(item.sentence)
    # no layout on record: still no FI in the sentence
    (root / ".fi" / "code_layout.json").unlink()
    [item] = _of(root, paper_facts.NOT_DONE)
    _clean(item.sentence)


def test_an_analysis_that_prints_no_list_is_no_difference_of_the_record(tmp_path: Path) -> None:
    """The s6 false-discovery record: six findings, all of this one kind. Not a limitation of the results."""
    root = _root(tmp_path)
    _write(root, "needs/RUN_MANIFEST_CHECK.json", {"status": "warned", "problems": [UNPRINTED], "attempts": 0})
    assert run_manifest.is_unprinted(UNPRINTED)
    assert not _of(root, paper_facts.RUN_DIFFERENCE)


def test_a_real_disagreement_is_still_an_item_in_reader_words(tmp_path: Path) -> None:
    root = _root(tmp_path)
    real = ("the metric `power` is measured in 400 trial(s) of FI's record: 300 with a value and 100 with none; the analysis "
            "accounts for 200 value(s) + 0 undefined = 200")
    _write(root, "needs/RUN_MANIFEST_CHECK.json", {"status": "warned", "problems": [UNPRINTED, real]})
    [item] = _of(root, paper_facts.RUN_DIFFERENCE)
    assert item.sentence.startswith("The numbers the analysis reports do not all match the record of the trials that were run")
    assert "1 place" in item.sentence and "fdr_unc" not in item.sentence
    _clean(item.sentence)
    # a plain sentence FI counted keeps its meaning, without code quotes
    _write(root, "needs/RUN_MANIFEST_CHECK.json", {"status": "warned", "problems": [real],
                                                  "headline": "The analysis leaves out trials of one setting (for `final_size`)."})
    [item] = _of(root, paper_facts.RUN_DIFFERENCE)
    assert "leaves out trials of one setting (for final_size)" in item.sentence and "`" not in item.sentence
    _clean(item.sentence)


def test_the_unprinted_kind_survives_the_engine_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_manifest_problems`` turned findings into plain strings, so the engine never saw one as unprinted."""
    from tests.test_engine_callable_simulation import _one_trial_per_setting
    from core.execution import ExecutionResult

    engine = _one_trial_per_setting(tmp_path, monkeypatch)
    found = [run_manifest.Finding(UNPRINTED, run_manifest.UNPRINTED), run_manifest.Finding("another thing differs")]
    monkeypatch.setattr("core.engine._run_manifest.problems", lambda *a, **k: found)
    ok = ExecutionResult(0, 'RESULT_JSON: {"x": 1}', "", 0.1, False)
    status, problems = engine._run_manifest_problems({}, True, ok, ("", "", ""))
    assert status == "differs" and UNPRINTED in problems
    assert engine._manifest_unprinted == [UNPRINTED]


def test_calibration_not_checked_is_said_for_a_reader(tmp_path: Path) -> None:
    root = _root(tmp_path)
    monkeypatch_runs = paper_facts._planned_runs
    paper_facts._planned_runs = lambda r: 52  # type: ignore[assignment]
    try:
        _write(root, ".fi/trial_checks.json", {"calibration": {"quantities": [], "checked": False, "not_checked": [
            "`Ea_hat`: its true value `Ea_true_kJ_per_mol` is neither a threshold nor a grid parameter of the plan"]}})
        [item] = _of(root, paper_facts.CALIBRATION)
    finally:
        paper_facts._planned_runs = monkeypatch_runs  # type: ignore[assignment]
    assert item.sentence == ("The coverage of the reported intervals, and the accuracy of their standard errors, were not "
                             "checked against known true values.")


def test_other_templates_have_no_fi_in_them(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _write(root, ".fi/code_review.json", {"issues": {"f1": {"at_gate": {"outcome": "left_out", "quantity": "p_hat"},
                                                            "text": "x"}}})
    _write(root, "needs/RESULT_CHECK.json", {"checks": [{"quantity": "k_hat", "note": "off by a factor"}]})
    for item in paper_facts.required_items(root):
        _clean(item.sentence)
    assert paper_facts.fewer_runs_text({"from": {"runs_per_setting": 100}, "to": {"runs_per_setting": 10}}).count("FI") == 0
