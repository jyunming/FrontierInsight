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

import pytest

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
def _gate_run(tmp_path: Path, test: str, wrapper: str = "") -> tuple[Any, dict[str, Any], Any]:
    from tests.test_absent_output_part_not_done import _real_test_runs
    from tests.test_shown_reach import _engine, _run

    engine = _engine(tmp_path)
    _real_test_runs(engine)
    if wrapper:  # a module outside code/ that calls the study
        engine.quest_root.mkdir(parents=True, exist_ok=True)
        (engine.quest_root / "installed_wrapper.py").write_text(wrapper, encoding="utf-8")
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


# ---- review round 1: B1 is a fact about what ran, and a recording never decides a verdict ---------------------------------
SELF_HASH = ("import hashlib, pathlib\n"
             "FILE = pathlib.Path.cwd() / 'code' / 'study.py'\n"
             "NOTE = pathlib.Path(__file__).with_suffix('.orig')\n\n\n"
             "def test_the_file_was_changed():\n"
             "    now = hashlib.sha256(FILE.read_bytes()).hexdigest()\n"
             "    if not NOTE.exists():\n        NOTE.write_text(now)\n"
             "    assert now != NOTE.read_text(), 'study.py is as it was'\n")
#: the same test with a read of a study name that looks like a dependence on the study but runs none of its code
SELF_HASH_SPOOFED = ("import study as _STUDY\n" + SELF_HASH.replace("assert now != NOTE.read_text()",
                                                                    "assert now != NOTE.read_text() and _STUDY.__name__"))
WRAPPER = "import study\n\n\ndef value():\n    return study.trial({'x': 1})\n"
VIA_WRAPPER = ("from installed_wrapper import value\n\n\ndef test_estimate():\n    assert value()['est_a'] == 9.0, 'wrong'\n")


def _wrapped(tmp_path: Path, test: str, study: str = STUDY) -> tuple[Any, str, Path]:
    (tmp_path / "installed_wrapper.py").write_text(WRAPPER, encoding="utf-8")  # outside code/
    return _held(tmp_path, test, study)


def test_a_test_that_reads_a_study_name_but_ran_no_study_code_shows_nothing(tmp_path: Path) -> None:
    for text in (SELF_HASH, SELF_HASH_SPOOFED):
        run, src, code = _held(tmp_path, text, name=f"t{len(text)}.py")
        assert run.status == "fail" and run.held["study_ran"] is False
        assert fr.failure_from_study(src, run.held, fr.study_modules(code)).kind == fr.NO_STUDY_CALL


def test_a_study_module_that_runs_code_when_it_loads_counts_as_study_code_that_ran(tmp_path: Path) -> None:
    for extra in ("\nLIMIT = 4\n", "\nTABLE = dict(a=1)\n", "\n_x = rate(2.0)\n"):
        run, src, code = _held(tmp_path, "import study\n\ndef test_x():\n    assert 1 == 2, 'own'\n", STUDY + extra,
                               name=f"m{len(extra)}.py")
        assert run.held["study_ran"] is True, extra
        assert not fr.failure_from_study(src, run.held, fr.study_modules(code))


def test_a_wrong_value_returned_through_a_wrapper_outside_code_stays_shown(tmp_path: Path) -> None:
    run, src, code = _wrapped(tmp_path, VIA_WRAPPER)
    assert run.status == "fail" and run.kind == "assertion", run.detail
    assert run.held["study_ran"] is True and run.held["calls"] == 0 and run.held["indirect"] >= 1
    assert not fr.failure_from_study(src, run.held, fr.study_modules(code))


def test_a_name_guess_after_a_wrapper_called_the_study_is_not_called_absent(tmp_path: Path) -> None:
    test = ("from installed_wrapper import value\nimport study\n\n\ndef test_x():\n    value()\n    r = study.trial({})\n"
            "    assert 'zz_spread' in r, 'none'\n")
    run, src, code = _wrapped(tmp_path, test)
    assert run.held["indirect"] >= 1 and not fr.failure_from_study(src, run.held, fr.study_modules(code))


def test_study_code_that_runs_in_a_thread_counts(tmp_path: Path) -> None:
    test = ("import threading\nimport study\n\n\ndef test_x():\n    t = threading.Thread(target=study.rate, args=(1.0,))\n"
            "    t.start()\n    t.join()\n    assert 1 == 2, 'own'\n")
    run, _src, _code = _held(tmp_path, test)
    assert run.held["study_ran"] is True


def test_the_creation_keeps_the_wrapper_test_and_refuses_the_self_hash_one(tmp_path: Path) -> None:
    (tmp_path / "installed_wrapper.py").write_text(WRAPPER, encoding="utf-8")
    run = CaptureRun(tmp_path, [{"id": "F-1", "file": "study.py", "function": "trial"}], {"code:test": [_block(VIA_WRAPPER)]})
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True
    assert "no_study_call" not in run.record["issues"]["F-1"]
    run = CaptureRun(tmp_path / "b", [{"id": "F-1", "file": "study.py", "function": "trial"}],
                     {"code:test": [_block(SELF_HASH_SPOOFED), _block(SELF_HASH)]})
    assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is False
    assert run.record["issues"]["F-1"]["outcome"] == "not demonstrated" and run.record["issues"]["F-1"]["no_study_call"] is True


def test_the_gate_clears_the_self_hash_test_and_keeps_the_wrapper_test(tmp_path: Path) -> None:
    gate_test = ("import hashlib, pathlib\nFILE = pathlib.Path(__file__).resolve().parents[2] / 'code' / 'model.py'\n"
                 "NOTE = pathlib.Path(__file__).with_suffix('.orig')\n\n\ndef test_x():\n    now = hashlib.sha256(FILE.read_bytes()).hexdigest()\n"
                 "    if not NOTE.exists():\n        NOTE.write_text(now)\n    assert now != NOTE.read_text(), 'as it was'\n")
    out, record, _engine = _gate_run(tmp_path / "a", gate_test)
    assert out is None and record["issues"]["F-1"]["demonstrated"] is False
    via = ("from installed_wrapper import value\n\n\ndef test_x():\n    assert value()['est_a'] == 9.0, 'wrong'\n")
    out, record, engine = _gate_run(tmp_path / "b", via, wrapper="import model\n\n\ndef value():\n    return model.helper_record(1)\n")
    assert out is not None and out.get("stuck") is True and record["issues"]["F-1"].get("demonstrated") is True


class _Doubles:
    """A harness double: the plain run says ``plain`` and the recording run says whatever it is told, in order."""

    def __init__(self, plain: str, recorded: list[Any]) -> None:
        self.plain, self.recorded, self.calls = plain, list(recorded), []

    async def __call__(self, rel: str, capture: bool = False) -> Any:
        self.calls.append(capture)
        if not capture:
            return types_ns(self.plain)
        return self.recorded.pop(0) if self.recorded else None


def types_ns(status: str) -> Any:
    import types

    return types.SimpleNamespace(status=status, crash_in_program=False, kind="assertion", detail="", place="", at="", held=None)


@pytest.mark.parametrize("recorded", [None, "pass", "error", "timeout"])
def test_the_recording_run_never_changes_the_verdict_of_the_plain_runs(tmp_path: Path, recorded: str | None) -> None:
    from tests.test_shown_reach import _engine, _run

    engine = _engine(tmp_path)
    shown = {"pass": types_ns("pass"), "error": types_ns("error"), "timeout": None, None: None}[recorded]
    double = _Doubles("fail", [shown])
    engine._rerun_finding_test = double  # type: ignore[method-assign,assignment]
    test = "import model\n\ndef test_x():\n    r = model.helper_record(1)\n    assert r['est_a'] == 9.0, 'wrong'\n"
    out, record = _run(engine, test, file="model.py", function="helper_record")
    assert out is not None and out.get("stuck") is True, "a recording that passes, errors or gives no answer clears nothing"
    assert double.calls == [False, False, True], "two plain runs decide, then one recording run, used only to classify"
    assert record["issues"]["F-1"].get("capture_unavailable") is True and record["issues"]["F-1"]["demonstrated"] is True


def test_the_recording_run_is_not_made_when_a_plain_run_passes(tmp_path: Path) -> None:
    from tests.test_shown_reach import _engine, _run

    engine = _engine(tmp_path)
    double = _Doubles("pass", [])
    engine._rerun_finding_test = double  # type: ignore[method-assign,assignment]
    out, _record = _run(engine, "import model\n\ndef test_x():\n    assert True\n", file="model.py", function="helper_record")
    assert out is None and double.calls == [False]


def test_the_creation_keeps_the_plain_verdict_when_the_recording_gives_a_pass_or_fails_to_run(tmp_path: Path) -> None:
    for answer in ("pass", "error", "raise"):
        run = CaptureRun(tmp_path / answer, [{"id": "F-1", "file": "study.py", "function": "trial"}], {"code:test": [_block(OWN_INPUT)]})

        async def recorded(paths: list[str], answer: str = answer) -> dict[str, fr.TestRun]:
            if answer == "raise":
                raise RuntimeError("the recording ran past its time")
            return {p: fr.TestRun(answer) for p in paths}

        run.env.capture_tests = recorded
        assert asyncio.run(fr.demonstrate(run.env, run.record["issues"]["F-1"], "study.py")) is True, answer
        assert run.record["issues"]["F-1"]["capture_unavailable"] is True and run.record["issues"]["F-1"]["demonstrated"] is True


# ---- a study run the recording cannot watch keeps the finding shown --------------------------------------------------------
def _unwatched(tmp_path: Path, test: str, study: str = STUDY, name: str = "u.py") -> bool:
    run, _src, _code = _held(tmp_path, test, study, name=name)
    assert run.status == "fail" and run.kind == "assertion", run.detail
    return run.held["study_ran"]


def test_a_test_that_runs_the_study_in_a_child_process_keeps_its_finding_shown(tmp_path: Path) -> None:
    test = ("import subprocess, sys\nimport study\n\n\ndef test_x():\n"
            "    out = subprocess.run([sys.executable, '-c', 'print(1)'], capture_output=True, text=True).stdout\n"
            "    assert out.strip() == '2', 'wrong'\n")
    assert _unwatched(tmp_path, test) is True


def test_a_test_that_reads_a_result_file_the_study_wrote_keeps_its_finding_shown(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text('{"est_a": 1.0}', encoding="utf-8")
    test = ("import json\nimport study\n\n\ndef test_x():\n    with open('results.json') as fh:\n        r = json.load(fh)\n"
            "    assert r['est_a'] == 2.0, 'wrong'\n")
    assert _unwatched(tmp_path, test) is True


def test_a_study_module_that_only_re_exports_a_compiled_function_counts_as_study_code(tmp_path: Path) -> None:
    test = "import study\n\n\ndef test_x():\n    assert study.sqrt(4.0) == 3.0, 'wrong'\n"
    assert _unwatched(tmp_path, test, "from math import sqrt\n") is True


def test_a_test_that_turns_the_profiler_off_keeps_its_finding_shown(tmp_path: Path) -> None:
    test = "import sys\nimport study\n\n\ndef test_x():\n    sys.setprofile(None)\n    assert study.__name__ == 'x', 'wrong'\n"
    assert _unwatched(tmp_path, test) is True


def test_study_code_run_by_a_thread_started_while_the_test_file_loads_counts(tmp_path: Path) -> None:
    test = ("import threading\nimport study\n\nt = threading.Thread(target=study.rate, args=(1.0,))\nt.start()\nt.join()\n\n\n"
            "def test_x():\n    assert 1 == 2, 'own'\n")
    assert _unwatched(tmp_path, test) is True


def test_a_test_that_runs_no_study_code_and_reads_no_result_file_is_still_not_shown(tmp_path: Path) -> None:
    test = "import study\n\n\ndef test_x():\n    assert study.__name__ == 'x', 'wrong'\n"
    assert _unwatched(tmp_path, test) is False


def test_opening_the_null_device_does_not_make_the_recording_unclear(tmp_path: Path) -> None:
    test = "import os\nimport study\n\n\ndef test_x():\n    open(os.devnull).close()\n    assert study.__name__ == 'x', 'wrong'\n"
    assert _unwatched(tmp_path, test) is False
