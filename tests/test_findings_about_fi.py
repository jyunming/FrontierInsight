"""A finding of the code reader whose place lies in lines FI wrote (its frame, a file it writes whole) is about FI, not the
study's code: it is set aside by position, never repaired, never stops the quest and takes no result out. A finding in the
study's own function is handled as before. Real small files; no model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core import analysis_frame as af
from core import code_findings as cf
from core import code_review
from core import skeleton as sk

DERIVE = (
    '"""The study\'s own part."""\n\n' + af.BEGIN + "\nRESULTS = ('a',)\nREFERENCES = {}\n" + af.END + "\n\n\n"
    "def derive_a(table):\n    return table.mean('a')\n")
SIMULATE = ("import numpy as np\n" + sk.TOP_BEGIN + "\nSETTINGS = {}\n" + sk.TOP_END + "\n\n\ndef trial(cell, ctx):\n    return 1.0\n\n"
            + sk.BOTTOM_BEGIN + "\ndef run_trial(c):\n    return trial(c, None)\n" + sk.TOP_END + "\n")
EXPERIMENT = af.EXPERIMENT_HEAD + "\nprint('x')\n"


def _code(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    code.mkdir()
    (code / "derive.py").write_text(DERIVE, encoding="utf-8")
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (code / "experiment.py").write_text(EXPERIMENT, encoding="utf-8")
    (code / "helper.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return code


def _line(text: str, needle: str) -> int:
    return next(n for n, row in enumerate(text.splitlines(), 1) if needle in row)


def test_a_place_in_a_frame_or_a_file_fi_writes_is_fis(tmp_path: Path) -> None:
    code = _code(tmp_path)
    assert cf.in_fi_lines(code, "derive.py", _line(DERIVE, "REFERENCES"))
    assert cf.in_fi_lines(code, "derive.py", _line(DERIVE, "RESULTS"))
    assert cf.in_fi_lines(code, "simulate.py", _line(SIMULATE, "SETTINGS"))
    assert cf.in_fi_lines(code, "simulate.py", _line(SIMULATE, "def run_trial"))
    assert cf.in_fi_lines(code, "experiment.py", 2) and cf.in_fi_lines(code, "experiment.py", None)


def test_a_place_in_the_studys_own_function_or_one_that_cannot_be_resolved_is_not(tmp_path: Path) -> None:
    code = _code(tmp_path)
    assert not cf.in_fi_lines(code, "derive.py", _line(DERIVE, "return table.mean"))
    assert not cf.in_fi_lines(code, "simulate.py", _line(SIMULATE, "return 1.0"))
    assert not cf.in_fi_lines(code, "helper.py", 2)
    assert not cf.in_fi_lines(code, "derive.py", None)  # no line: handled as before
    assert not cf.in_fi_lines(code, "derive.py", 9999)
    assert not cf.in_fi_lines(code, "missing.py", 3)
    assert not cf.in_fi_lines(code, "", 3)
    (code / "experiment.py").write_text("print('the study wrote this')\n", encoding="utf-8")  # not FI's text: the study's
    assert not cf.in_fi_lines(code, "experiment.py", 1)


def _problem(file: str, line: int, text: str) -> dict[str, Any]:
    return {"id": "L1", "kind": "logic", "text": text, "status": "different", "file": file, "line": line, "note": "",
            "fix": "", "severity": "high", "name": "", "where": f"{file}:{line}", "changes_result": True,
            "plan_conflict": False, "headline": False}


def _read(code: Path, problems: list[dict[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {"problems": problems, "read_problems": list(problems)}
    code_review.note_reading(record, "v1", problems, code)
    return record


def test_a_finding_about_fis_frame_is_set_aside_and_one_about_the_studys_body_is_not(tmp_path: Path) -> None:
    code = _code(tmp_path)
    frame = _problem("derive.py", _line(DERIVE, "REFERENCES"), "REFERENCES is empty")
    body = _problem("derive.py", _line(DERIVE, "return table.mean"), "the mean is not the plan's estimand")
    record = _read(code, [frame, body])
    issues = cf.issues_of(record)
    own = [i for i in issues.values() if i.get("fi_own")]
    assert [i["text"] for i in own] == ["REFERENCES is empty"] and own[0]["outcome"] == cf.ABOUT_FI
    assert [i["text"] for i in cf.open_issues(record)] == ["the mean is not the plan's estimand"]
    assert [v["text"] for v in record["problems"]] == [body["text"]] and [v["text"] for v in record["read_problems"]] == [body["text"]]
    assert record["about_fi"][0]["id"] == own[0]["id"] and "derive.py" in record["about_fi"][0]["where"]
    assert record["readings"][0]["open"] == 1


def test_a_shown_fault_about_fis_frame_is_never_a_candidate_for_the_gates_stop(tmp_path: Path) -> None:
    code = _code(tmp_path)
    frame_line = _line(DERIVE, "REFERENCES")
    body_line = _line(DERIVE, "return table.mean")
    for line, stops in ((frame_line, False), (body_line, True)):
        record = _read(code, [_problem("derive.py", line, "a fault")])
        for issue in cf.issues_of(record).values():
            issue.update(demonstrated=True, failure_kind=cf.ASSERTION, outcome=cf.GAVE_UP)
        # an old record (before the mark existed) is set aside from its place alone
        for issue in cf.issues_of(record).values():
            issue.pop("fi_own", None)
        cf.set_aside_own(record, code)
        for issue in cf.issues_of(record).values():
            if issue.get("fi_own"):
                issue["outcome"] = cf.ABOUT_FI
        assert bool(cf.shown_open(record)) is stops


def test_setting_aside_again_changes_nothing_and_the_quality_row_says_it(tmp_path: Path) -> None:
    from core import quality_report as qr

    code = _code(tmp_path)
    record = _read(code, [_problem("derive.py", _line(DERIVE, "REFERENCES"), "REFERENCES is empty")])
    assert cf.set_aside_own(record, code) == [] and len(record["about_fi"]) == 1
    row = qr._issue_rows(record)[0]
    assert row["about_fi"] is True and row["open"] is False
