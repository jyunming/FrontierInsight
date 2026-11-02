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
    for line, stops in ((_line(DERIVE, "REFERENCES"), False), (_line(DERIVE, "return table.mean"), True)):
        record = _read(code, [_problem("derive.py", line, "a fault")])
        for issue in cf.issues_of(record).values():
            issue.update(demonstrated=True, failure_kind=cf.ASSERTION)
        assert bool(cf.shown_open(record)) is stops


def test_a_position_is_never_judged_against_code_it_was_not_read_from(tmp_path: Path) -> None:
    code = _code(tmp_path)
    body = "def trial(cell, ctx):\n" + "".join(f"    x{n} = {n}\n" for n in range(30)) + "    return 1.0\n"
    text = ("import numpy as np\n" + sk.TOP_BEGIN + "\nSETTINGS = {}\n" + sk.TOP_END + "\n\n\n" + body + "\n"
            + sk.BOTTOM_BEGIN + "\ndef run_trial(c):\n    return trial(c, None)\n" + sk.TOP_END + "\n")
    (code / "simulate.py").write_text(text, encoding="utf-8")
    short = text.replace(body, "def trial(cell, ctx):\n    return 1.0\n")
    line = _line(short, "def run_trial")  # a body line before the repair, a line of the bottom frame after it
    assert not cf.in_fi_lines(code, "simulate.py", line)
    version = code_review.version(code)
    record: dict[str, Any] = {}
    problems = [_problem("simulate.py", line, "the model body is wrong")]
    record["problems"] = problems
    code_review.note_reading(record, version, problems, code)
    issue = next(iter(cf.issues_of(record).values()))
    assert not issue.get("fi_own")
    issue.update(demonstrated=True, failure_kind=cf.ASSERTION, outcome=cf.GAVE_UP)
    # a later repair shortens the model body: the bottom frame now starts above the stored line
    (code / "simulate.py").write_text(short, encoding="utf-8")
    assert cf.in_fi_lines(code, "simulate.py", line)
    assert cf.set_aside_own(record, code, code_review.version(code)) == []
    issue["outcome"] = cf.OPEN
    assert cf.set_aside_own(record, code, code_review.version(code)) == []  # still another version than the one read
    assert not issue.get("fi_own") and cf.open_issues(record)


def test_an_issue_with_a_final_outcome_is_never_changed(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record = _read(code, [_problem("derive.py", _line(DERIVE, "REFERENCES"), "x")])
    issue = next(iter(cf.issues_of(record).values()))
    issue.pop("fi_own"), issue.update(outcome=cf.GAVE_UP)
    assert cf.set_aside_own(record, code, "v1") == [] and issue["outcome"] == cf.GAVE_UP


def test_a_range_or_several_places_are_fis_only_when_every_one_is(tmp_path: Path) -> None:
    code = _code(tmp_path)
    frame, body = _line(DERIVE, "RESULTS"), _line(DERIVE, "return table.mean")
    assert cf.place_in_fi_lines(code, f"derive.py:{frame}-{frame + 1}")
    assert not cf.place_in_fi_lines(code, f"derive.py:{frame}-{body}")
    assert not cf.place_in_fi_lines(code, f"derive.py:{frame} and simulate.py:{_line(SIMULATE, 'return 1.0')}")
    assert cf.place_in_fi_lines(code, f"derive.py:{frame} and simulate.py:{_line(SIMULATE, 'SETTINGS')}")


def test_a_trial_harness_module_in_code_is_judged_by_its_content(tmp_path: Path) -> None:
    code = _code(tmp_path)
    (code / "trial_harness.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert not cf.in_fi_lines(code, "trial_harness.py", 2)
    (code / "trial_harness.py").unlink()
    assert cf.in_fi_lines(code, "trial_harness.py", 2)  # absent from code/: FI's own, in .fi/


def test_the_convergence_summary_does_not_count_a_finding_about_fi(tmp_path: Path) -> None:
    code = _code(tmp_path)
    record = _read(code, [_problem("derive.py", _line(DERIVE, "REFERENCES"), "x")])
    for issue in cf.issues_of(record).values():
        issue.update(demonstrated=True)
    s = cf.summary(record)
    assert s["issues"] == 0 and s["result_changing"] == 0 and s["demonstrated"] == 0 and s["about_fi"] == 1


def test_setting_aside_again_changes_nothing_and_the_quality_row_says_it(tmp_path: Path) -> None:
    from core import quality_report as qr

    code = _code(tmp_path)
    record = _read(code, [_problem("derive.py", _line(DERIVE, "REFERENCES"), "REFERENCES is empty")])
    assert cf.set_aside_own(record, code, "v1") == [] and len(record["about_fi"]) == 1
    row = qr._issue_rows(record)[0]
    assert row["about_fi"] is True and row["open"] is False


# ---- the real gate ---------------------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig  # noqa: E402
from core.engine import Engine  # noqa: E402

FAILING_TEST = "import helper\n\n\ndef test_f_is_two():\n    assert helper.f() == 2, 'f is one'\n"
STATE: dict[str, Any] = {"exec_result": {"returncode": 0}, "result_json": {"dose": 60.0}}


async def _gate(tmp_path: Path, line_of: str, outcome: str) -> dict[str, Any] | None:
    cfg = Config(topic="a dose to size", title="dose", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    for name, text in (("derive.py", DERIVE), ("helper.py", "def f():\n    return 1\n")):
        (code / name).write_text(text, encoding="utf-8")
    (engine.quest_root / ".fi" / "finding_tests").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / ".fi" / "finding_tests" / "F-1_1.py").write_text(FAILING_TEST, encoding="utf-8")
    where = f"derive.py:{_line(DERIVE, line_of)}"
    issue = {"id": "F-1", "changes_result": True, "demonstrated": True, "failure_kind": cf.ASSERTION, "outcome": outcome,
             "plan_conflict": False, "readings": [{"n": 1, "status": "open", "version": code_review.version(code)}],
             "severity": "high", "first_seen": 1, "file": "derive.py", "function": "", "where": where,
             "text": "a fault shown by a test", "test": ".fi/finding_tests/F-1_1.py"}
    code_review.save(engine.quest_root, {"issues": {"F-1": issue}})
    return await engine._shown_fault_verdict(STATE)


@pytest.mark.usefixtures("this_python")
@pytest.mark.asyncio
async def test_the_gate_does_not_stop_for_a_shown_fault_in_fis_frame(tmp_path: Path) -> None:
    assert await _gate(tmp_path, "REFERENCES", cf.OPEN) is None
    record = code_review.load(next((tmp_path / "outputs").iterdir()))
    assert record["issues"]["F-1"]["fi_own"] is True and record["about_fi"][0]["id"] == "F-1"


@pytest.mark.usefixtures("this_python")
@pytest.mark.asyncio
async def test_the_gate_still_stops_for_a_shown_fault_in_the_studys_function(tmp_path: Path) -> None:
    verdict = await _gate(tmp_path, "return table.mean", cf.OPEN)
    assert verdict and verdict["stuck"] is True and verdict["stuck_reason"] == "shown_fault_open"


def _requirement(file: str, line: int) -> dict[str, Any]:
    """A problem of the requirement kind: it has a file and a line but no ``where``."""
    return {"id": "R1", "kind": "metric", "text": "a plan quantity is not computed", "status": "different", "file": file,
            "line": line, "note": "", "fix": "", "severity": "high", "name": "q", "headline": False}


def test_a_requirement_problem_at_a_frame_line_is_set_aside_and_one_at_a_body_line_is_not(tmp_path: Path) -> None:
    code = _code(tmp_path)
    frame = _read(code, [_requirement("derive.py", _line(DERIVE, "REFERENCES"))])
    assert [i["outcome"] for i in cf.issues_of(frame).values()] == [cf.ABOUT_FI] and not cf.open_issues(frame)
    assert frame["problems"] == [] and len(frame["about_fi"]) == 1
    body = _read(code, [_requirement("derive.py", _line(DERIVE, "return table.mean"))])
    assert not any(i.get("fi_own") for i in cf.issues_of(body).values()) and len(cf.open_issues(body)) == 1
