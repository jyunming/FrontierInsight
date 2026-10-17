"""Only a test's own assertion about values shows a fault (core/finding_repair.py, core/code_findings.py), replayed on the
finding tests and code of three real quests (tests/fixtures/replay/finding_v2/: each quest's simulation, its package and
its analysis as written and as they ran, the kept finding tests, the code-review record and the result with every list
cut to two items).

* E2 stopped without a paper on F-b4bb7f48, whose test first "failed" only because it turned a crash inside the code
  (``model.py:141``) into ``assert False``. Shown again here, that is a crash for the crash repair, not a shown fault; its
  other shown fault (F-331a25f1) fails by its own assertion when first shown and on the code that ran, so its quantity is
  left out and the quest does not stop.
* r14b stopped on F-27f456d7, a figure finding whose test failed by assertion when first shown but raised AttributeError in
  the test itself on the rewritten code: an error shows nothing, so the quest does not stop.
* r12 (truly wrong code) is not caught here: both of its key finding tests only ever errored, one inside the code (the
  solver's optimality failure, for the crash repair) and one at its own call. r12 is caught by FI's checks of the trials
  (tests/test_trial_checks.py, tests/test_trial_checks_run.py).

The headline names are left empty (the plan's design is not in the fixtures), as the study-5 gate replay did."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import shutil
import string
from pathlib import Path
from typing import Any

import pytest

from core import code_findings, code_review, finding_repair

FIX = Path(__file__).parent / "fixtures" / "replay" / "finding_v2"
AGENTS = Path(__file__).parent.parent / "agents"


def _gz(name: str) -> Any:
    with gzip.open(FIX / name, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _env(root: Path, record: dict[str, Any], replies: dict[str, str]) -> finding_repair.Env:
    """The repair loop's environment on a copy of the code as written: the model's reply is the test the quest kept, and
    the tests run through FI's own harness in their own process."""
    import os
    import subprocess
    import sys

    async def ask(prompt: str, who: str, kind: str) -> str:
        return "```python\n" + replies["current"] + "\n```"

    async def run_tests(paths: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for path in paths:
            done = subprocess.run([sys.executable, "-B", "-c", finding_repair.harness_source("code", [path])], cwd=root,
                                  capture_output=True, text=True, timeout=300,
                                  env={**os.environ, **finding_repair.run_environment(root)})
            out.update(finding_repair.parse_runs(done.stdout, [path]))
        return out

    async def snapshot() -> dict[str, str]:
        return {}

    prompts = {"code_finding_test": string.Template((AGENTS / "code_finding_test.md").read_text(encoding="utf-8"))}
    return finding_repair.Env(root=root, code=root / "code", record=record, save=lambda: None, ask=ask,
                              run_tests=run_tests, snapshot=snapshot, static=dict, keeps=lambda *a: "", prompts=prompts,
                              topic="t", equations="[]", scrub=lambda t: t, log=logging.getLogger("test"))


def _first_show(tmp_path: Path, case: str, record: dict[str, Any], fid: str) -> dict[str, Any]:
    """The issue shown again, as FI now does it, on the code as written, with the test the quest kept for it."""
    root = tmp_path / case
    if not root.exists():
        shutil.copytree(FIX / case, root)
    issue = record["issues"][fid]
    kept = Path(issue["test"]).name
    number = int(kept.rsplit("_", 1)[1].split(".")[0])
    replies = {"current": (root / ".fi" / "finding_tests" / kept).read_text(encoding="utf-8")}
    for k in ("demonstrated", "failure", "failure_kind", "test", "calls", "crash", "errors"):
        issue.pop(k, None)
    issue["tests"] = number - 1
    record["finding_calls"] = 0  # the quest spent its requests; this replay asks for none
    env = _env(root, record, replies)
    rel = finding_repair.target_file(issue, root / "code")
    finding_repair.TEST_CALLS, calls = 1, finding_repair.TEST_CALLS  # one test, the one the quest kept
    try:
        asyncio.run(finding_repair.demonstrate(env, issue, rel))
    finally:
        finding_repair.TEST_CALLS = calls
    return issue


@pytest.mark.usefixtures("this_python")
def test_replay_e2_the_crash_is_not_a_shown_fault_and_the_quest_does_not_stop(tmp_path: Path) -> None:
    record = _gz("e2_record.json.gz")
    b4 = _first_show(tmp_path, "e2_written", record, "F-b4bb7f48")
    assert b4["demonstrated"] is False and b4["outcome"] == code_findings.NOT_DEMONSTRATED
    assert b4["crash"]["at"] == "code/kinetics_parameter_recovery/model.py:141"
    assert "turned it into an assertion" in b4["crash"]["detail"]
    f33 = _first_show(tmp_path, "e2_written", record, "F-331a25f1")
    assert f33["demonstrated"] is True and f33["failure_kind"] == code_findings.ASSERTION
    assert f33["assertion"].startswith("assert hasattr(") and f33["test"].endswith("F-331a25f1_2.py")
    assert [i["id"] for i in code_findings.shown_open(record)] == ["F-331a25f1"]

    # the gate, on the code that ran
    from tests.test_disclosure import _engine

    root = tmp_path / "gate"
    shutil.copytree(FIX / "e2_final", root)
    code_review.save(root, record)
    eng = _engine(root)
    verdict = asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": _gz("e2_result.json.gz")}))
    assert verdict is None
    gate = code_review.load(root)["issues"]["F-331a25f1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "convergence_check" in gate["keys"]


@pytest.mark.usefixtures("this_python")
def test_replay_r14b_a_test_that_errors_on_the_code_that_ran_does_not_stop(tmp_path: Path) -> None:
    record = _gz("r14b_record.json.gz")
    issue = _first_show(tmp_path, "r14b_written", record, "F-27f456d7")
    assert issue["demonstrated"] is True and issue["failure_kind"] == code_findings.ASSERTION
    from tests.test_disclosure import _engine

    root = tmp_path / "gate"
    shutil.copytree(FIX / "r14b_final", root)
    for other in record["issues"].values():  # only this issue is replayed; the others are as the quest left them
        if other["id"] != "F-27f456d7":
            other.pop("failure_kind", None)
    code_review.save(root, record)
    assert [i["id"] for i in code_findings.shown_open(record)] == ["F-27f456d7"]
    eng = _engine(root)
    verdict = asyncio.run(eng._shown_fault_verdict(  # type: ignore[arg-type]
        {"exec_result": {"returncode": 0}, "result_json": _gz("r14b_result.json.gz")}))
    assert verdict is None
    error = code_review.load(root)["issues"]["F-27f456d7"]["gate_error"]
    assert error["place"] == "test" and "has no attribute 'plt'" in error["detail"]


def test_replay_r12_its_finding_tests_only_error_one_inside_the_code(tmp_path: Path) -> None:
    root = tmp_path / "r12"
    shutil.copytree(FIX / "r12_written", root)
    record = {"issues": {fid: {"id": fid, "file": "kinetics_parameter_recovery/model.py", "function": "fit_rate",
                               "text": "t", "changes_result": True, "tests": 0}
                         for fid in ("F-aece8568", "F-b166c00c")}}
    for fid in record["issues"]:
        replies = {"current": (root / ".fi" / "finding_tests" / f"{fid}_1.py").read_text(encoding="utf-8")}
        env = _env(root, record, replies)
        finding_repair.TEST_CALLS, calls = 1, finding_repair.TEST_CALLS
        try:
            asyncio.run(finding_repair.demonstrate(env, record["issues"][fid], "kinetics_parameter_recovery/model.py"))
        finally:
            finding_repair.TEST_CALLS = calls
    aece, b166 = record["issues"]["F-aece8568"], record["issues"]["F-b166c00c"]
    assert aece["demonstrated"] is False and aece["crash"]["at"] == "code/kinetics_parameter_recovery/model.py:91"
    assert b166["demonstrated"] is False and "crash" not in b166
    assert b166["errors"][-1]["place"] == "test" and "unexpected keyword argument" in b166["errors"][-1]["detail"]
    assert not code_findings.shown_open(record)


@pytest.mark.usefixtures("this_python")
def test_replay_r12_the_crash_goes_to_the_crash_repair_once(tmp_path: Path) -> None:
    from tests.test_disclosure import _engine

    root = tmp_path / "r12"
    shutil.copytree(FIX / "r12_written", root)
    record = {"issues": {"F-aece8568": {"id": "F-aece8568", "file": "kinetics_parameter_recovery/model.py",
                                        "function": "fit_rate", "text": "t", "changes_result": True, "tests": 0}}}
    replies = {"current": (root / ".fi" / "finding_tests" / "F-aece8568_1.py").read_text(encoding="utf-8")}
    finding_repair.TEST_CALLS, calls = 1, finding_repair.TEST_CALLS
    try:
        asyncio.run(finding_repair.demonstrate(_env(root, record, replies), record["issues"]["F-aece8568"],
                                               "kinetics_parameter_recovery/model.py"))
    finally:
        finding_repair.TEST_CALLS = calls
    code_review.save(root, record)
    eng = _engine(root)
    state = {"exec_reflect_iter": 0, "design": {"split": True}}
    sent = asyncio.run(eng._review_crash_run(state))  # type: ignore[arg-type]
    assert sent is not None and sent["exec_result"]["returncode"] == 1
    assert "code/kinetics_parameter_recovery/model.py:91" in sent["exec_result"]["stderr_tail"]
    assert "Rate fit did not satisfy" in sent["exec_result"]["stderr_tail"]
    assert asyncio.run(eng._review_crash_run(state)) is None  # sent once
    assert code_review.load(root)["crashes_sent"] == 1
