"""The paper's one closing sentence (core/quality_pointer.py): it names only the kinds of case that occur, stands once at the
end of the body before the reference list, and is gone when the last case is gone. Neutral made-up draft."""
from __future__ import annotations

import json
import re
from pathlib import Path

from core import paper_facts as pf
from core import quality_pointer as qp
from core import result_checks as rc

BOTH = "Values that fell outside the range the study expected, and statements that the study's results and the cited sources do not support, are listed in the accompanying quality report."
RANGE = "Values that fell outside the range the study expected are listed in the accompanying quality report."
CLAIMS = "Statements that the study's results and the cited sources do not support are listed in the accompanying quality report."
FORBIDDEN = ["FI", "check", "record", "recorded", "unconfirmed", "QUALITY.md", "quality.md", "ledger"]


def test_the_sentence_names_only_the_kinds_that_occur() -> None:
    assert qp.sentence(True, True) == BOTH
    assert qp.sentence(True, False) == RANGE
    assert qp.sentence(False, True) == CLAIMS
    assert qp.sentence(False, False) == ""
    for said in (BOTH, RANGE, CLAIMS):
        assert not re.search(r"\d", said)  # no count, no identifier
        for word in FORBIDDEN:
            assert not re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", said, 0 if word == "FI" else re.I), (word, said)


def _root(tmp_path: Path, *, outside: bool = False, unsupported: bool = False) -> Path:
    (tmp_path / ".fi").mkdir(parents=True, exist_ok=True)
    if outside:
        (tmp_path / "needs").mkdir(exist_ok=True)
        (tmp_path / "needs" / rc.RECORD).write_text(json.dumps({rc.KEPT_BREAKS: [{"quantity": "q", "note": "n"}]}), encoding="utf-8")
    if unsupported:
        pf.write_unsupported_record(tmp_path, [{"claim": "A claim.", "evidence": "none"}])
    return tmp_path


def test_the_text_follows_the_records(tmp_path: Path) -> None:
    assert qp.text(_root(tmp_path / "a")) == ""
    assert qp.text(_root(tmp_path / "b", outside=True)) == RANGE
    assert qp.text(_root(tmp_path / "c", unsupported=True)) == CLAIMS
    assert qp.text(_root(tmp_path / "d", outside=True, unsupported=True)) == BOTH
    # an emptied record of claims names none
    pf.write_unsupported_record(tmp_path / "d", [])
    assert qp.text(tmp_path / "d") == RANGE


def test_the_placeholder_goes_before_the_reference_list_or_at_the_end_of_the_body() -> None:
    plain = "# T\n\n## Results\n\nR.\n\n## Limitations\n\nL.\n\n## References\n\n1. A.\n"
    out = qp.with_block(plain)
    assert out.index("{{quality_pointer}}") > out.index("L.") and out.index("{{quality_pointer}}") < out.index("## References")
    ruled = "# T\n\n## Discussion\n\nD.\n\n---\n\n## References\n\n1. A.\n"
    out = qp.with_block(ruled)
    assert out.index("D.") < out.index("{{quality_pointer}}") < out.index("---")
    assert qp.with_block("# T\n\n## Results\n\nR.\n").rstrip().endswith("{{quality_pointer}}")  # no reference list: the end
    assert qp.with_block(out) == out  # already there: not put in twice


EQUATION = "results that rest on an equation not confirmed before the run"
FEWER = "results from fewer runs than planned"
RANGE_PART = "values that fell outside the range the study expected"
CLAIMS_PART = "statements that the study's results and the cited sources do not support"


def _equation(root: Path) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "equation_tests.json").write_text(json.dumps({
        "unconfirmed": {"E1": {"function": "f", "marked": True, "results": ["q_mean"]}}}), encoding="utf-8")


def _fewer(root: Path) -> None:
    from core import frozen_protocol, run_estimate

    frozen_protocol.freeze(root, {"metrics": [{"id": "y", "kind": "mean"}], "runs_per_setting": 12}, approved_by="auto: test",
                           source="test")
    run_estimate.write_record(root, {"reduced": {"note": "n", "to": {"runs_per_setting": 12}, "marked_unconfirmed": True}})


def test_each_new_kind_is_named_alone(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _root(a)
    _equation(a)
    assert qp.text(a) == "Results that rest on an equation not confirmed before the run are listed in the accompanying quality report."
    _root(b)
    _fewer(b)
    assert qp.text(b) == "Results from fewer runs than planned are listed in the accompanying quality report."


def test_two_three_and_four_kinds_read_as_one_short_list() -> None:
    assert qp.sentence(False, False, True, True) == (
        "Results that rest on an equation not confirmed before the run, and results from fewer runs than planned, "
        "are listed in the accompanying quality report.")
    three = qp.sentence(True, False, True, True)
    assert three == (f"Values that fell outside the range the study expected, {EQUATION}, and {FEWER}, "
                     "are listed in the accompanying quality report.")
    four = qp.sentence(True, True, True, True)
    assert four == (f"Values that fell outside the range the study expected, {EQUATION}, {FEWER}, and {CLAIMS_PART}, "
                    "are listed in the accompanying quality report.")
    for said in (three, four):
        assert said.count(".") == 1 and not re.search(r"\d", said)
        for word in FORBIDDEN:
            assert not re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", said, 0 if word == "FI" else re.I), (word, said)


def test_all_four_kinds_from_the_records_stand_in_the_paper_once_and_go_with_their_cases(tmp_path: Path) -> None:
    root = _root(tmp_path, outside=True, unsupported=True)
    _equation(root)
    _fewer(root)
    full = qp.text(root)
    assert full == qp.sentence(True, True, True, True)
    draft = "# T\n\n## Results\n\nR.\n\n## References\n\n1. A.\n"
    marked = "<!-- fi:fact-quality_pointer -->" + full + "<!-- /fi:fact-quality_pointer -->"
    assert qp.in_paper(draft.replace("R.", marked)) == full
    # one case after another goes: the sentence follows the records
    pf.write_unsupported_record(root, [])
    (root / ".fi" / "equation_tests.json").unlink()
    assert qp.text(root) == qp.sentence(True, False, False, True)
    (root / "needs" / rc.RECORD).unlink()
    assert qp.text(root) == FEWER[0].upper() + FEWER[1:] + " are listed in the accompanying quality report."
    (root / ".fi" / "run_estimate.json").unlink()
    assert qp.text(root) == ""


# --- FI places the two blocks itself: a copy the draft carries anywhere else is taken out ------------------------------------

DRAFT_CASES = {
    "table cell": "# T\n\n## Results\n\n| a | b |\n|---|---|\n| x | {{quality_pointer}} |\n\nR.\n\n## Discussion\n\nD.\n\n## References\n\n1. A.\n",
    "reference list": "# T\n\n## Results\n\nR.\n\n## Discussion\n\nD.\n\n## References\n\n1. A. {{quality_pointer}}\n",
    "twice": "# T\n\n## Results\n\nR.\n\n{{quality_pointer}}\n\n## Discussion\n\nD.\n\n{{quality_pointer}}\n\n## References\n\n1. A.\n",
}


def test_a_placeholder_the_draft_carries_anywhere_is_replaced_by_one_at_the_closing_position(tmp_path: Path) -> None:
    root = _root(tmp_path, outside=True)
    (root / qp.FILE).write_text(qp.text(root) + "\n", encoding="utf-8")
    for case, draft in DRAFT_CASES.items():
        placed = pf.place_blocks(draft, root)
        assert placed.count("{{quality_pointer}}") == 1, case
        assert placed.index("D.") < placed.index("{{quality_pointer}}") < placed.index("## References"), case
        assert "| x | |" in placed or "| x |  |" in placed or case != "table cell", placed
        assert "A. {{" not in placed
        # filled in as well: the sentence stands once, at that place, and a second pass changes nothing
        again = pf.place_blocks(placed.replace("{{quality_pointer}}", "<!-- fi:fact-quality_pointer -->S<!-- /fi:fact-quality_pointer -->")
                                + "\n{{quality_pointer}}\n", root)
        assert again.count("{{quality_pointer}}") == 1 and "<!-- fi:fact-quality_pointer" not in again, case


def test_with_no_case_left_every_copy_goes_and_none_is_put_in(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for draft in DRAFT_CASES.values():
        assert "quality_pointer" not in pf.place_blocks(draft, root)


FAILED = "results whose check against a known answer did not pass"


def _failed_check(root: Path, status: str = "went_on_failing", entries: list | None = None) -> None:
    (root / "needs").mkdir(parents=True, exist_ok=True)
    rows = [{"name": "decay_at_zero", "measured": 0.5, "expected": 1.0}] if entries is None else entries
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": status, "went_on": rows}), encoding="utf-8")


def test_a_known_answer_check_that_did_not_pass_and_was_gone_on_with_is_named_alone_and_in_order(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _failed_check(root)
    assert qp.text(root) == "Results whose check against a known answer did not pass are listed in the accompanying quality report."
    assert qp.sentence(True, True, True, True, True) == (
        f"Values that fell outside the range the study expected, {FAILED}, {EQUATION}, {FEWER}, and {CLAIMS_PART}, "
        "are listed in the accompanying quality report.")
    assert qp.sentence(True, False, failed_check=True) == (
        f"Values that fell outside the range the study expected, and {FAILED}, are listed in the accompanying quality report.")


def test_a_check_that_passed_or_was_not_gone_on_with_names_nothing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _failed_check(root, status="passed")
    assert qp.text(root) == ""
    _failed_check(root, entries=[])
    assert qp.text(root) == ""
    _failed_check(root, entries=[{"unmeasured": ["a check no trial measured"]}])
    assert qp.text(root) == "Results whose check against a known answer did not pass are listed in the accompanying quality report."
    (root / "needs" / "ORACLE_CHECK.json").unlink()
    assert qp.text(root) == ""


# --- a curve or map FI left out of a figure the paper places ----------------------------------------------------------------

LEFT_OUT = "Curves left out of a figure because their values did not match the study's results are listed in the accompanying quality report."


def _figure(root: Path, draw, *, placed: bool = True) -> Path:
    """A quest whose figure ``g.png`` is drawn by ``draw(host)`` (FI's frame on a made-up table) and placed in the paper."""
    from tests.test_frame_figure_labels_plan import _Host

    h = _Host()
    draw(h)
    _root(root)
    (root / "paper").mkdir(exist_ok=True)
    h.write(root / ".fi" / "figure_records" / "g.png")
    import matplotlib.pyplot as plt

    plt.close(h.fig)
    (root / "paper" / "paper.md").write_text("# T\n\n" + ("![A figure.](figures/g.png)\n" if placed else "No figure.\n"),
                                             encoding="utf-8")
    return root


def test_a_curve_left_out_of_a_placed_figure_is_named_once(tmp_path: Path) -> None:
    def wrong(h):  # FI's record of growth is 3, 5, 7: neither curve is it
        h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth")
        h.line(0, "uptake", [1, 2, 3], [9.0, 9.5, 9.9], "uptake")
    root = _figure(tmp_path / "a", wrong)
    assert qp.has_left_out_figure(root) == "curves"
    said = qp.text(root)
    assert said == LEFT_OUT and said.count("left out") == 1
    for word in FORBIDDEN:
        assert not re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", said, 0 if word == "FI" else re.I), (word, said)
    assert qp.sentence(True, False, left_out="curves") == (
        "Values that fell outside the range the study expected, and curves left out of a figure because their values did not "
        "match the study's results, are listed in the accompanying quality report.")


def test_no_curve_left_out_or_none_in_a_placed_figure_names_nothing(tmp_path: Path) -> None:
    root = _figure(tmp_path / "a", lambda h: h.line(0, "growth", [1, 2, 3], [3.0, 5.0, 7.0], "growth"))  # FI's record
    assert qp.has_left_out_figure(root) == "" and qp.text(root) == ""
    hidden = _figure(tmp_path / "b", lambda h: h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth"), placed=False)
    assert qp.has_left_out_figure(hidden) == "" and qp.text(hidden) == ""


def test_the_draft_being_written_decides_which_figures_count(tmp_path: Path) -> None:
    # A write round places the sentence before the draft is saved: the draft's figures count, not the saved paper's.
    import logging
    from types import SimpleNamespace

    from core import engine as E

    root = _figure(tmp_path, lambda h: h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth"), placed=False)
    draft = "# T\n\n![A figure.](figures/g.png)\n"
    assert qp.text(root, draft) == LEFT_OUT and qp.text(root, "# T\n\nNo figure.\n") == ""
    me = SimpleNamespace(quest_root=root, _log=logging.getLogger("t"))
    E.Engine._write_quality_pointer(me, draft)
    assert (root / qp.FILE).read_text(encoding="utf-8").strip() == LEFT_OUT
    E.Engine._write_quality_pointer(me, "# T\n\nNo figure.\n")
    assert not (root / qp.FILE).exists()


def test_a_map_left_out_is_named_as_a_map(tmp_path: Path) -> None:
    root = _figure(tmp_path, lambda h: h.image(0, [[1.0, 2.0], [3.0, 4.0]], ("dose", [1, 2]), ("feed", [1, 2]), "uptake"))
    assert qp.has_left_out_figure(root) == "maps"
    assert qp.text(root) == ("Maps left out of a figure because their values could not be confirmed as the study's results are "
                             "listed in the accompanying quality report.")


def test_a_curve_and_a_map_of_one_result_and_two_results_of_one_label_each_get_a_row(tmp_path: Path) -> None:
    # One panel loses a curve of growth, a map of growth and a curve of uptake, which the plan labels like growth: three
    # omissions, three rows, and the closing sentence names both curves and maps.
    from core import quality_report

    def draw(h):
        h.labels["uptake"] = h.labels["growth"]
        h.line(0, "growth", [1, 2, 3], [9.0, 9.5, 9.9], "growth")  # FI's record of growth is 3, 5, 7
        h.image(0, [[1.0, 2.0], [3.0, 4.0]], [1, 2], [1, 2], "growth")  # plain coordinates: FI cannot confirm the map
        h.line(0, "uptake", [1, 2, 3], [9.0, 9.5, 9.9], "uptake")  # FI's record of uptake is 1, 2, 3
    root = _figure(tmp_path, draw)
    rows = [p["text"] for p in quality_report._paper_problems(quality_report._Quest(root)) if "declared as" in p["text"]]
    assert rows == ["Figure 1: a series declared as Growth (1/h) does not match FI's record of that result; it is left out of the figure.",
                    "Figure 1: a map declared as Growth (1/h) could not be confirmed as FI's record of that result; it is left out "
                    "of the figure.",
                    "Figure 1: a series declared as Growth (1/h) does not match FI's record of that result; it is left out of the figure."]
    assert qp.has_left_out_figure(root) == "curves and maps"
