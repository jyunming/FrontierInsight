"""A finding counts as shown only when its failing assertion comes from a value the study returned
(core/finding_repair.py ``failure_from_study``). Two facts FI records itself, no model: the test failed before any call into
the study's code returned and reads nothing of it, or it looks for an output name that no record the study returned holds.
The capture it reads (``CAPTURE``) also reads a dict subclass (FI's own trial record), takes a NumPy number for a bare number,
and keeps its record complete past 200 returns of names already seen.

Neutral made-up study; the tests are run by the real harness in this interpreter. No model is called."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

from core import finding_repair as fr
from tests.test_finding_repair import Run, _block

STUDY = '''
import numpy as np


class Rec(dict):
    """A trial record: a dict filled by assignment, as FI's own."""


def trial(cell, ctx=None):
    out = Rec()
    out["est_a"] = np.float64(2.0 * cell.get("x", 1))
    out["est_b"] = 5.0
    return out


def many(n, late=False):
    return [Rec(est_a=1.0, est_b=2.0, **({"new_key": 1.0} if late and i == n - 1 else {})) for i in range(n)]


def fill(out):
    out["filled_value"] = 1.0


def rate(x):
    return 3.0 * x
'''


def _held(tmp_path: Path, test: str, study: str = STUDY, name: str = "t.py") -> tuple[Any, str, Path]:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "study.py").write_text(study, encoding="utf-8")
    path = tmp_path / name
    path.write_text(test, encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(code), [str(path)], capture=True)],
                          capture_output=True, text=True, cwd=tmp_path, timeout=120)
    return fr.parse_runs(done.stdout, [str(path)])[str(path)], test, code


def _verdict(tmp_path: Path, test: str, study: str = STUDY) -> fr.StudyFailure:
    run, src, code = _held(tmp_path, test, study)
    assert run.status == "fail" and run.kind == "assertion", run.detail
    return fr.failure_from_study(src, run.held, fr.study_modules(code))


# ---- what the capture records ----------------------------------------------------------------------------------------------
def test_the_capture_reads_a_dict_subclass_and_a_numpy_number(tmp_path: Path) -> None:
    run, _src, _code = _held(tmp_path, "import study\n\ndef test_x():\n    r = study.trial({})\n    assert r['est_a'] == 1\n")
    held = run.held
    assert held["returned"][0]["record"] is True and held["returned"][0]["names"] == ["est_a", "est_b"]
    assert held["returned_complete"] is True and held["calls"] == 1 and held["inits"] == 0


def test_a_return_past_the_200th_of_names_already_seen_keeps_the_record_complete(tmp_path: Path) -> None:
    loop, _s, _c = _held(tmp_path, "import study\n\ndef test_x():\n    for _ in range(300):\n        study.trial({})\n    assert False\n",
                         name="loop.py")
    assert loop.held["returned_complete"] is True and len(loop.held["returned"]) == 200 and loop.held["calls"] == 300


def test_a_late_return_with_a_new_name_makes_the_record_incomplete(tmp_path: Path) -> None:
    study = STUDY + "\n\ndef late(i):\n    return {'est_a': 1.0} if i < 250 else {'new_key': 1.0}\n"
    run, _s, _c = _held(tmp_path, "import study\n\ndef test_x():\n    for i in range(300):\n        study.late(i)\n    assert False\n",
                        study)
    assert run.held["returned_complete"] is False
    assert fr.failure_from_study("assert False", run.held, {"study"}).kind == "", "incomplete: judged as before"


def test_a_local_the_capture_cannot_list_does_not_make_the_returned_values_incomplete(tmp_path: Path) -> None:
    run, _s, _c = _held(tmp_path, "import study\n\ndef test_x():\n    gen = (i for i in range(3))\n    r = study.trial({})\n"
                        "    assert 'zz_missing' in r\n")
    assert run.held["complete"] is False and run.held["returned_complete"] is True


# ---- the rule --------------------------------------------------------------------------------------------------------------
def test_an_assertion_that_fails_on_the_tests_own_input_before_any_call_is_no_study_failure(tmp_path: Path) -> None:
    test = ("import study\n\ndef test_x():\n    cell = {'x': 2}\n    assert cell['x'] == 3, 'the input is wrong'\n"
            "    study.trial(cell)\n")
    v = _verdict(tmp_path, test)
    assert v.kind == fr.NO_STUDY_CALL and v.names == [] and "own input" in v.why


def test_an_assertion_on_a_value_the_study_returned_is_a_value_fault(tmp_path: Path) -> None:
    for assertion in ("r['est_a'] == 3.0", "r['est_b'] == 4.0", "study.rate(2) == 7.0"):
        test = f"import study\n\ndef test_x():\n    r = study.trial({{'x': 1}})\n    assert {assertion}, 'wrong'\n"
        assert not _verdict(tmp_path, test), assertion


def test_a_value_the_study_reads_without_a_call_is_a_study_value(tmp_path: Path) -> None:
    study = STUDY + "\nLIMIT = 4\n"
    test = "import study\n\ndef test_x():\n    assert study.LIMIT == 5, 'wrong limit'\n"
    assert _verdict(tmp_path, test, study).kind == "", "no call returned, but the assertion reads a name of the study"


def test_a_name_no_returned_record_holds_is_an_output_the_study_does_not_give(tmp_path: Path) -> None:
    test = ("import study\n\ndef test_x():\n    r = study.trial({'x': 1})\n"
            "    found = [k for k in r if 'zz_spread' in k]\n    assert found, 'no spread: %r' % sorted(r)\n")
    v = _verdict(tmp_path, test)
    assert v.kind == fr.NAME_NOT_GIVEN and v.names == ["zz_spread"]
    for lookup in ("r.get('zz_spread') is not None", "'zz_spread' in r", "r['zz_spread'] == 1"):
        run, src, code = _held(tmp_path, f"import study\n\ndef test_x():\n    r = study.trial({{'x': 1}})\n    assert {lookup}\n")
        if run.status == "fail":  # the subscript raises, it is an error: only the two lookups are assertions
            assert fr.failure_from_study(src, run.held, fr.study_modules(code)).kind == fr.NAME_NOT_GIVEN, lookup


def test_a_name_the_study_returns_is_never_called_absent_and_a_local_name_never_hides_one(tmp_path: Path) -> None:
    given = "import study\n\ndef test_x():\n    r = study.trial({'x': 1})\n    assert 'est_b' in r and 'est_a' not in r\n"
    assert _verdict(tmp_path, given).kind == "", "est_a is returned: a value fault, not an absent name"
    # a name the test holds itself (its input) is not an output the study gave
    own = ("import study\n\ndef test_x():\n    label = 'zz_guess'\n    cell = {label: 1}\n    r = study.trial(cell)\n"
           "    assert label in r, 'missing'\n")
    v = _verdict(tmp_path, own)
    assert v.kind == fr.NAME_NOT_GIVEN and v.names == ["zz_guess"]


def test_an_answer_through_an_argument_keeps_the_rule_off(tmp_path: Path) -> None:
    # the study fills the dict the test passes in and returns nothing: a call that returned no record of names, so FI cannot
    # say the name was never given (a real fault of this shape is not moved out of "shown")
    test = ("import study\n\ndef test_x():\n    out = {}\n    study.fill(out)\n    r = study.trial({})\n"
            "    assert 'filled_value' in r, 'missing'\n")
    assert _verdict(tmp_path, test).kind == ""


def test_a_constructor_call_or_a_bare_number_return_keeps_the_rule_off(tmp_path: Path) -> None:
    study = STUDY + "\n\nclass Model:\n    def __init__(self):\n        self.k = 1\n"
    test = ("import study\n\ndef test_x():\n    m = study.Model()\n    r = study.trial({})\n"
            "    assert 'zz_spread' in r\n")
    assert _verdict(tmp_path, test, study).kind == "", "a constructor is a call whose value FI does not list"
    bare = "import study\n\ndef test_x():\n    v = study.rate(2)\n    r = study.trial({})\n    assert 'zz_spread' in r\n"
    assert _verdict(tmp_path, bare).kind == "", "a bare number returned is not a record of names"


def test_what_cannot_be_read_is_judged_as_before() -> None:
    held = {"returned": [], "returned_complete": True, "calls": 0, "inits": 0, "call_lines": [], "failures": [{"line": 3, "empty": []}]}
    assert not fr.failure_from_study("this is not python (", held, {"study"})
    assert not fr.failure_from_study("assert 1 == 2\n", None, {"study"})
    assert not fr.failure_from_study("assert 1 == 2\n", {"calls": 0}, {"study"}), "no failing line recorded"
    assert not fr.failure_from_study("x = 1\n", held, {"study"}), "the failing line is not an assertion"
    old = {"returned": [], "names": [], "complete": True, "failures": [{"line": 1, "empty": []}]}
    assert not fr.failure_from_study("assert 1 == 2\n", old, {"study"}), "a record made before FI counted calls"


# ---- where it is used ------------------------------------------------------------------------------------------------------
class CaptureRun(Run):
    """The loop's run of a test, with the capture of the study's returns for the second run (as the engine gives)."""

    def __init__(self, tmp_path: Path, issues: list[dict[str, Any]], replies: dict[str, list[str]]) -> None:
        super().__init__(tmp_path, issues, replies)
        (self.code / "study.py").write_text(STUDY, encoding="utf-8")
        self.captured = 0

        async def capture(paths: list[str]) -> dict[str, fr.TestRun]:
            self.captured += 1
            out = {}
            for p in paths:
                done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(self.code), [str(self.root / p)], True)],
                                      capture_output=True, text=True, cwd=self.root, timeout=60)
                out[p] = fr.parse_runs(done.stdout, [str(self.root / p)])[str(self.root / p)]
            return out

        self.env.capture_tests = capture


OWN_INPUT = "import study\n\ndef test_x():\n    cell = {'x': 2}\n    assert cell['x'] == 3, 'wrong input'\n    study.trial(cell)\n"
VALUE = "import study\n\ndef test_x():\n    r = study.trial({'x': 1})\n    assert r['est_a'] == 9.0, 'est_a is wrong'\n"
ABSENT = ("import study\n\ndef test_x():\n    r = study.trial({'x': 1})\n"
          "    assert [k for k in r if 'zz_spread' in k], 'no spread'\n")


def test_a_test_that_fails_on_its_own_input_is_asked_for_again_once(tmp_path: Path) -> None:
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}],
                     {"code:test": [_block(OWN_INPUT), _block(VALUE)]})
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True
    issue = run.record["issues"]["F-1"]
    assert issue["no_study_call"] is True and issue["tests"] == 2 and "absent_output" not in issue
    assert "own input" in run.prompts[1], "the second request says why the first test was not used"
    assert (run.root / issue["test"]).read_text(encoding="utf-8") == VALUE


def test_two_tests_that_fail_on_their_own_input_end_the_tries_not_demonstrated(tmp_path: Path) -> None:
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}],
                     {"code:test": [_block(OWN_INPUT), _block(OWN_INPUT), _block(VALUE)]})
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is False
    issue = run.record["issues"]["F-1"]
    assert issue["demonstrated"] is False and issue["outcome"] == "not demonstrated" and issue["tests"] == 2
    assert "own input" in issue["reason"] and issue["errored_only"] is False


def test_a_test_that_looks_for_an_output_the_study_does_not_give_stays_shown_and_says_so(tmp_path: Path) -> None:
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}], {"code:test": [_block(ABSENT)]})
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True
    issue = run.record["issues"]["F-1"]
    assert issue["demonstrated"] is True and issue["absent_output"] == ["zz_spread"]


def test_without_the_capture_the_loop_judges_the_test_by_its_failure_alone(tmp_path: Path) -> None:
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}], {"code:test": [_block(OWN_INPUT)]})
    run.env.capture_tests = None
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True


def test_a_recording_run_that_gives_no_answer_is_not_an_unstable_test(tmp_path: Path) -> None:
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}], {"code:test": [_block(VALUE)]})

    async def timed_out(paths: list[str]) -> dict[str, fr.TestRun]:
        return {p: fr.TestRun("error", "the test could not be run here") for p in paths}

    run.env.capture_tests = timed_out
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True


# ---- the gate ------------------------------------------------------------------------------------------------------------------
def _gate_run(tmp_path: Path, test: str) -> tuple[Any, dict[str, Any], Any]:
    from tests.test_absent_output_part_not_done import _real_test_runs
    from tests.test_shown_reach import _engine, _run

    engine = _engine(tmp_path)
    _real_test_runs(engine)
    files = {"model.py": ("class Rec(dict):\n    pass\n\n\ndef helper_record(x):\n    r = Rec()\n    r['est_a'] = 2.0 * x\n"
                          "    r['est_b'] = 5.0\n    return r\n"),
             "simulate.py": ("import model\n\n\ndef trial(cell, ctx):\n    r = model.helper_record(1)\n"
                             "    return {**r, 'se_a': 1.0, 'lo_a': 0.0, 'hi_a': 3.0, 'extra_t': 7.0}\n")}
    out, record = _run(engine, test, file="model.py", function="helper_record", quantity="spread of the estimate", files=files)
    return out, record, engine


def test_the_gate_clears_a_test_that_failed_on_its_own_input_before_any_call(tmp_path: Path) -> None:
    test = ("import model\n\ndef test_x():\n    cell = {'x': 2}\n    assert cell['x'] == 3, 'the input is wrong'\n"
            "    model.helper_record(1)\n")
    out, record, _engine = _gate_run(tmp_path, test)
    assert out is None, "main stops here: 'its test still fails on it'"
    issue = record["issues"]["F-1"]
    assert issue["demonstrated"] is False and issue["outcome"] == "not demonstrated" and "own input" in issue["reason"]


def test_the_gate_still_stops_for_a_wrong_value_the_study_returned(tmp_path: Path) -> None:
    test = "import model\n\ndef test_x():\n    r = model.helper_record(1)\n    assert r['est_a'] == 9.0, 'wrong'\n"
    out, record, _engine = _gate_run(tmp_path, test)
    assert out is not None and out.get("stuck") is True
    assert record["issues"]["F-1"].get("demonstrated") is True


def test_the_gate_places_a_name_the_study_does_not_give_as_a_part_not_done(tmp_path: Path) -> None:
    test = ("import model\n\ndef test_x():\n    r = model.helper_record(1)\n"
            "    found = [k for k, v in r.items() if 'zz_spread' in str(k) and v == 0.5]\n"
            "    assert found, 'the spread is never stored, got %r' % sorted(r)\n")
    out, record, _engine = _gate_run(tmp_path, test)
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


def test_the_harness_source_stays_valid_python() -> None:
    compile(fr.harness_source("code", ["a.py"], capture=True), "harness", "exec")
