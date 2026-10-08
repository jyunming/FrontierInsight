"""Findings of the code reader by stable id (core/code_findings.py): the id does not depend on the reader's words, the same
issue told twice is one, the history of an issue is kept across readings, and what can change a reported result ranks
first. Neutral code (a mass on a damped spring); no model is called."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import code_findings as cf
from core import code_review as cr
from core import review_pairs as pairs

CODE = '''import math


def damping_ratio(m, k, c):
    return c / (2.0 * math.sqrt(m * k))


def summarize(rows):
    total = 0.0
    for r in rows:
        total += r["value"]
    return {"quality_factor": total / max(1, len(rows)), "settling_time": 1.0}
'''
SIM = "def run_cell(cell):\n    return {'quality': 1.0}\n"


def _code(tmp_path: Path, experiment: str = CODE) -> Path:
    code = tmp_path / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "experiment.py").write_text(experiment, encoding="utf-8")
    (code / "simulate.py").write_text(SIM, encoding="utf-8")
    return code


def _logic(where: str, what: str, quantity: str = "", **extra: Any) -> dict[str, Any]:
    reply = {"findings": [{"where": where, "what": what, "why": "w", "fix": "f", "severity": "medium",
                           "quantity": quantity, **extra}]}
    return cr.parse_logic(reply, [])[0]


def _req(kind: str, text: str, name: str, status: str = "different", file: str = "experiment.py",
         line: int | None = 8) -> dict[str, Any]:
    return {"id": "R1", "kind": kind, "text": text, "status": status, "file": file, "line": line, "note": "n",
            "headline": False, "name": name}


def _ids(problems: list[dict[str, Any]], code: Path) -> list[str]:
    cf.index_problems(problems, code)
    return [p["fid"] for p in problems]


def test_a_reworded_finding_keeps_its_id_and_the_same_issue_told_twice_is_one(tmp_path: Path) -> None:
    code = _code(tmp_path)
    first = _logic("experiment.py:9", "the sum is divided by the row count", "quality_factor")
    reworded = _logic("experiment.py:10", "an average that is not the average the plan defines", "Quality_Factor")
    requirement = _req("dependent variable", "the dependent variable quality_factor is computed", "quality_factor", line=11)
    a, b, c = _ids([first, reworded, requirement], code)
    assert a == b == c, "the words differ, the place and the quantity do not"
    other = _logic("experiment.py:5", "the ratio ignores the stiffness", "damping_ratio")
    assert _ids([other], code)[0] != a
    elsewhere = _logic("simulate.py:2", "the same quantity in another file", "quality_factor")
    assert _ids([elsewhere], code)[0] != a


def test_the_id_follows_the_function_not_the_line_number(tmp_path: Path) -> None:
    before = _logic("experiment.py:9", "w", "quality_factor")
    code = _code(tmp_path)
    (fid,) = _ids([before], code)
    moved = _code(tmp_path / "moved", "# a header\n# two lines\n\n" + CODE)
    again = _logic("experiment.py:12", "w", "quality_factor")
    assert _ids([again], moved) == [fid]
    code2 = _code(tmp_path / "renamed", CODE.replace("def summarize", "def summarise"))
    assert _ids([_logic("experiment.py:9", "w", "quality_factor")], code2) != [fid], "another function is another issue"


def test_a_finding_without_a_quantity_is_told_apart_by_what_the_code_assigns_there(tmp_path: Path) -> None:
    code = _code(tmp_path)
    a = _logic("experiment.py:9", "one fault with no quantity named")
    b = _logic("experiment.py:10", "another fault with no quantity named")
    c = _logic("experiment.py:9", "the first, in other words")
    ids = _ids([a, b, c], code)
    assert ids[0] == ids[2] and ids[0] != ids[1]


def test_a_finding_that_names_several_quantities_joins_the_requirement_that_one_of_them_has(tmp_path: Path) -> None:
    code = _code(tmp_path)
    requirement = _req("metric", "the result reports settling_time", "settling_time", line=11)
    logic = _logic("experiment.py:11", "both are computed wrongly", "quality_factor, settling_time")
    r, l = _ids([requirement, logic], code)
    assert r == l
    alone = _logic("experiment.py:11", "both are computed wrongly", "quality_factor, settling_time")
    assert _ids([alone], code)[0] != r, "with no requirement to join it takes the first quantity"


def test_a_figure_a_swept_setting_and_a_check_are_issues_of_their_own(tmp_path: Path) -> None:
    code = _code(tmp_path)
    one = _req("figure", "the script draws the planned figure decay.png", "", status="missing", file="", line=None)
    two = _req("figure", "the script draws the planned figure other.png", "", status="missing", file="", line=None)
    sweep = _req("sweep", "the parameter c is varied over all 3 values of the plan's grid", "", file="simulate.py", line=2)
    assert len(set(_ids([one, two, sweep], code))) == 3


def test_the_reader_is_asked_for_every_finding_and_none_is_cut() -> None:
    many = {"findings": [{"where": f"experiment.py:{i}", "what": f"fault {i}", "quantity": f"q{i}"} for i in range(1, 41)]}
    got = cr.parse_logic(many, [])
    assert len(got) == 40, "the old limit of twelve is gone"
    assert [g["text"] for g in got][:3] == ["fault 1", "fault 2", "fault 3"]


def test_the_reader_says_whether_a_finding_can_change_a_result_and_whether_the_plan_contradicts_itself() -> None:
    got = cr.parse_logic({"findings": [
        {"where": "a.py:1", "what": "x", "changes_result": True}, {"where": "a.py:2", "what": "y", "changes_result": "no"},
        {"where": "a.py:3", "what": "z"}, {"where": "a.py:4", "what": "p", "plan_conflict": "yes"}]}, [])
    assert [g["changes_result"] for g in got] == [True, False, None, None]
    assert [g["plan_conflict"] for g in got] == [False, False, False, True]
    # A reply that does not say: a high-severity finding or a requirement about a quantity counts as one that does.
    assert cf.changes_result({"kind": "logic", "severity": "high", "changes_result": None})
    assert not cf.changes_result({"kind": "logic", "severity": "medium", "changes_result": None})
    assert cf.changes_result({"kind": "metric"}) and not cf.changes_result({"kind": "figure"})
    assert not cf.changes_result({"kind": "logic", "severity": "high", "changes_result": False})


def _reading(record: dict[str, Any], problems: list[dict[str, Any]], code: Path, version: str = "v") -> None:
    cr.note_reading(record, version, problems, code)


def test_the_history_of_an_issue_is_kept_across_readings(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record: dict[str, Any] = {}
    _reading(record, [_logic("experiment.py:9", "the sum is divided by the row count", "quality_factor", changes_result=True),
                      _logic("experiment.py:5", "the ratio ignores the stiffness", "damping_ratio", changes_result=False)], code)
    (fid,) = [i["id"] for i in cf.issues_of(record).values() if i["quantity"] == "qualityfactor"]
    issue = cf.issues_of(record)[fid]
    assert issue["first_seen"] == 1 and issue["changes_result"] is True and issue["readings"][-1]["status"] == "open"
    # Reading two: the first comes back in other words, the second is not flagged, a new one appears.
    _reading(record, [_logic("experiment.py:10", "an average that is not what the plan defines", "quality_factor",
                             changes_result=True),
                      _logic("simulate.py:2", "a new thing", "quality", changes_result=True)], code)
    issues = cf.issues_of(record)
    assert len(issues) == 3
    assert issues[fid]["first_seen"] == 1 and issues[fid]["last_seen"] == 2 and len(issues[fid]["wordings"]) == 2
    assert [r["status"] for r in issues[fid]["readings"]] == ["open", "open"]
    gone = next(i for i in issues.values() if i["quantity"] == "dampingratio")
    assert [r["status"] for r in gone["readings"]] == ["open", "not flagged"] and gone["outcome"] == cf.GONE
    new = next(i for i in issues.values() if i["quantity"] == "quality")
    assert new["first_seen"] == 2
    assert [r["open"] for r in record["readings"]] == [2, 2] and record["readings"][1]["ids"] == [fid, new["id"]]
    # A fixed issue the reader flags again is reopened, once more tried.
    issues[fid]["outcome"] = cf.FIXED
    _reading(record, [_logic("experiment.py:9", "still the sum", "quality_factor", changes_result=True)], code)
    again = cf.issues_of(record)[fid]
    assert again["outcome"] == cf.REOPENED and again["rounds"] == 1 and again["demonstrated"] is None
    assert [r["n"] for r in again["readings"]] == [1, 2, 3]


def test_what_can_change_a_result_comes_first_then_severity_then_the_headline(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record: dict[str, Any] = {}
    problems = [_logic("experiment.py:5", "minor", "damping_ratio", changes_result=False, severity="high"),
                _logic("experiment.py:9", "medium but changes", "quality_factor", changes_result=True, severity="medium"),
                _logic("experiment.py:11", "high and changes", "settling_time", changes_result=True, severity="high"),
                _logic("simulate.py:2", "low but changes", "quality", changes_result=True, severity="low")]
    _reading(record, problems, code)
    assert [i["text"] for i in cf.ranked(record)] == ["high and changes", "medium but changes", "low but changes", "minor"]
    assert [i["text"] for i in cf.changing_open(record)] == ["high and changes", "medium but changes", "low but changes"]
    summary = cf.summary(record)
    assert summary["issues"] == 4 and summary["result_changing"] == 3 and summary["open"] == 4


def test_the_convergence_statement_counts_issues_a_test_showed_not_findings_per_reading(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record: dict[str, Any] = {}
    _reading(record, [_logic("experiment.py:9", "a", "quality_factor", changes_result=True),
                      _logic("experiment.py:5", "b", "damping_ratio", changes_result=True),
                      _logic("experiment.py:11", "c", "settling_time", changes_result=True)], code)
    issues = list(cf.issues_of(record).values())
    issues[0].update(demonstrated=True, outcome=cf.FIXED, fixed_by="model-a")
    issues[1].update(demonstrated=False, outcome=cf.NOT_DEMONSTRATED)
    issues[2].update(demonstrated=True)
    line = cf.convergence_line(record)
    assert "3 distinct issue(s)" in line and "2 of those were shown by a failing test" in line
    assert "1 issue(s) were repaired" in line and "1 shown result-changing issue(s) are still open" in line
    assert cf.summary(record)["fixed_by"] == {"model-a": 1}
    assert cr.metrics_line(record).endswith(line)


def test_a_contradiction_of_the_plan_is_not_among_the_issues_the_repairs_are_for(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record: dict[str, Any] = {}
    _reading(record, [_logic("experiment.py:9", "the plan defines it two ways", "quality_factor", plan_conflict=True,
                             changes_result=True)], code)
    (issue,) = cf.issues_of(record).values()
    assert issue["plan_conflict"] is True and cf.changing_open(record) == []


def test_the_request_to_the_plan_s_model_holds_the_contradiction_and_no_value() -> None:
    text = cf.plan_conflict_request("a spring", [("the spread is a standard deviation in one part and an absolute error "
                                                   "in another", "the metric says one, the method the other")], [0.25])
    assert "contradicts itself" in text and "standard deviation" in text and "a spring" in text
    assert "protocol.metrics" in text and "Do not change a check, a threshold" in text
    leaky = cf.plan_conflict_request("t", [("the value should be 0.25 here", "")], [0.25])
    assert "0.25" not in leaky and pairs.PLACEHOLDER in leaky
    assert set(cf.PLAN_PARTS) == {"metrics", "ci_method", "model"}, "no check, threshold or tolerance can be rewritten"


@pytest.mark.parametrize("text,expected", [("p_missing", "pmissing"), ("Quality_Factor", "qualityfactor")])
def test_names_are_compared_without_case_or_punctuation(text: str, expected: str) -> None:
    assert cf.norm(text) == expected


def test_the_quality_report_lists_each_finding_by_id_with_what_a_test_showed_and_who_repaired_it(tmp_path: Path) -> None:
    import json

    from core import quality_report as qr

    code = _code(tmp_path / "quest")
    record: dict[str, Any] = {"read_version": "v", "final_read": True, "repair_calls": 3, "send_backs": 1}
    _reading(record, [_logic("experiment.py:9", "the average is not the plan's", "quality_factor", changes_result=True,
                             severity="high"),
                      _logic("experiment.py:5", "the ratio ignores the stiffness", "damping_ratio", changes_result=True),
                      _logic("experiment.py:11", "a note is missing", "settling_time", changes_result=False),
                      _logic("simulate.py:2", "the plan defines it two ways", "quality", plan_conflict=True, changes_result=True)],
             code)
    by_quantity = {i["quantity"]: i for i in cf.issues_of(record).values()}
    by_quantity["qualityfactor"].update(demonstrated=True, outcome=cf.FIXED, fixed_by="model-a")
    by_quantity["dampingratio"].update(demonstrated=True, outcome=cf.GAVE_UP, reason="no repair passed its test")
    by_quantity["settlingtime"].update(demonstrated=False, outcome=cf.NOT_DEMONSTRATED)
    by_quantity["quality"].update(outcome=cf.PLAN_PROBLEM)
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "code_review.json").write_text(json.dumps(record), encoding="utf-8")
    report = qr.build(root)
    rows = {r["id"]: r for r in report["code"]["issues"]}
    fixed = rows[by_quantity["qualityfactor"]["id"]]
    assert fixed["demonstrated"] is True and fixed["repaired"] is True and fixed["repaired_by"] == "model-a" and fixed["open"] is False
    stuck = rows[by_quantity["dampingratio"]["id"]]
    assert stuck["demonstrated"] is True and stuck["repaired"] is False and stuck["open"] is True and stuck["changes_result"] is True
    assert rows[by_quantity["settlingtime"]["id"]]["demonstrated"] is False
    assert rows[by_quantity["quality"]["id"]]["plan_problem"] is True
    assert report["code"]["issue_summary"]["fixed_by"] == {"model-a": 1}
    text = qr.render(report)
    assert "### Findings of the code reader, by id" in text and "Repaired by: model-a (1)." in text
    assert "yes, by model-a" in text and "yes, and it changes a result" in text
    assert "no: the test passed on the code, so it was not repaired" in text and "plan problem: the plan contradicts itself" in text
    assert "shown by a failing test" in text and "distinct issue(s)" in text
