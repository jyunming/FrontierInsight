"""A code finding whose failing test looks for a name the study's code never gave it, and that FI's own records hold no value
for, is a part not done (delivered and stated), not a stop. The fact comes from FI, not from the finding's words: when FI runs
the test again it records what the study's code returned to the test and the records the test held at its failed assertion
(core/finding_repair.py ``CAPTURE``), and compares that with the names the test reads and with FI's records of the run
(core/shown_reach.py ``Reach.absent_names``). A finding about a value that differs still stops or leaves out as before.

Neutral made-up study (the one of tests/test_shown_reach.py) with a helper of the trial that returns a record; the finding's
test is run by the real harness in this interpreter. No model is called."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from core import finding_repair
from core.rerun_check import Verdict
from tests.test_shown_reach import FILES, _engine, _run, _stops

#: The trial's helper returns a record of two estimates; the trial adds the rest of what it returns.
STUDY = {
    "model.py": FILES["model.py"] + "\n\ndef helper_record(x):\n    return {'est_a': 2.0 * x, 'est_b': 5.0}\n",
    "simulate.py": ("import model\n\n\ndef trial(cell, ctx):\n    r = model.helper_record(1)\n"
                    "    return {**r, 'se_a': 1.0, 'lo_a': 0.0, 'hi_a': 3.0, 'extra_t': 7.0}\n"),
}

#: The finding: the helper never stores the spread. Its test looks for the name among what the helper returned.
ABSENT = ("from model import helper_record\n\n\ndef test_x():\n    r = helper_record(1)\n"
          "    found = [k for k, v in r.items() if 'zz_spread' in str(k) and v == 0.5]\n"
          "    assert found, 'the spread is never stored, got %r' % sorted(r)\n")


def _source(paths: list[str], capture: bool) -> str:
    try:
        return finding_repair.harness_source("code", paths, capture=capture)
    except TypeError:  # a harness that records nothing (FI before this record existed): the test runs as it did then
        return finding_repair.harness_source("code", paths)


def _real_test_runs(engine: Any) -> None:
    """The gate runs the finding's test with the real harness (the quest's environment is this interpreter)."""

    async def run(rel: str, capture: bool = False) -> Any:
        src = _source([rel], capture)
        done = subprocess.run([sys.executable, "-B", "-c", src], cwd=engine.quest_root, capture_output=True, text=True,
                              timeout=120)
        return finding_repair.parse_runs(done.stdout, [rel]).get(rel)

    engine._rerun_finding_test = run  # type: ignore[method-assign,assignment]


def _gate(tmp_path: Path, test: str, files: dict[str, str] | None = None, rerun: Verdict | None = None,
          plant: Any = None) -> tuple[dict[str, Any] | None, dict[str, Any], Any]:
    engine = _engine(tmp_path)
    _real_test_runs(engine)
    if rerun is not None:
        async def answers(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
            return {str(i["id"]): rerun for i, _h in verdicts}

        engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    if plant is not None:
        original = engine._shown_fault_reach

        def reach_after_plant(headline: Any, state: Any) -> Any:
            plant(engine.quest_root)
            return original(headline, state)

        engine._shown_fault_reach = reach_after_plant  # type: ignore[method-assign]
    out, record = _run(engine, test, file="model.py", function="helper_record", quantity="spread of the estimate",
                       files={**STUDY, **(files or {})})
    return out, record, engine


def _not_done(engine: Any) -> list[str]:
    path = engine.quest_root / "needs" / "RESULT_CHECK.json"
    if not path.is_file():
        return []
    return [c["quantity"] for c in json.loads(path.read_text(encoding="utf-8")).get("checks", []) if c.get("kind") == "not_computed"]


# ---- the harness records what the code gave the test ---------------------------------------------------------------------------
def _harness(tmp_path: Path, code: dict[str, str], test: str, capture: bool) -> Any:
    for rel, text in code.items():
        (tmp_path / "code" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "code" / rel).write_text(text, encoding="utf-8")
    (tmp_path / "t").mkdir(exist_ok=True)
    (tmp_path / "t" / "test_f.py").write_text(test, encoding="utf-8")
    src = _source(["t/test_f.py"], capture)
    done = subprocess.run([sys.executable, "-B", "-c", src], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    return finding_repair.parse_runs(done.stdout, ["t/test_f.py"])["t/test_f.py"]


def test_the_harness_records_the_names_of_what_the_code_returned_to_a_failing_test(tmp_path: Path) -> None:
    code = {"m.py": "def rec(x):\n    return {'a': x, 'b': {'c': 2}}\n\n\ndef num(x):\n    return 3.0\n"}
    run = _harness(tmp_path, code, "import m\n\n\ndef test_x():\n    r = m.rec(1)\n    assert 'z' in r\n", capture=True)
    assert run.status == "fail" and run.held["complete"] is True
    assert run.held["returned"] == [{"function": "rec", "record": True, "names": ["a", "b", "c"]}]
    run = _harness(tmp_path, code, "import m\n\n\ndef test_x():\n    assert m.num(1) == 2\n", capture=True)
    assert run.held["returned"][0]["record"] is False  # a bare number is no record of names


def test_every_failing_test_function_of_the_file_is_recorded_together(tmp_path: Path) -> None:
    code = {"m.py": "def rec(x):\n    return {'a': x}\n\n\ndef num(x):\n    return 3.0\n"}
    test = ("import m\n\n\ndef test_x():\n    assert 'z' in m.rec(1)\n\n\ndef test_y():\n    assert m.num(1) == 2\n")
    run = _harness(tmp_path, code, test, capture=True)
    assert [r["record"] for r in run.held["returned"]] == [True, False]


def test_without_capture_the_harness_records_nothing_more(tmp_path: Path) -> None:
    code = {"m.py": "def rec(x):\n    return {'a': x}\n"}
    run = _harness(tmp_path, code, "import m\n\n\ndef test_x():\n    assert 'z' in m.rec(1)\n", capture=False)
    assert run.status == "fail" and run.held is None


# ---- the gate --------------------------------------------------------------------------------------------------------------------
def test_a_name_the_code_never_gave_the_test_and_no_record_holds_is_a_part_not_done_not_a_stop(tmp_path: Path) -> None:
    # on the old gate: stopped without a paper ("FI cannot find which of the results it changes")
    out, record, engine = _gate(tmp_path, ABSENT)
    assert out is None
    issue = record["issues"]["F-1"]
    assert issue["at_gate"]["outcome"] == "not_computed" and issue["at_gate"]["names"] == ["zz_spread"]
    assert issue["held_at_failure"]["returned"][0]["names"] == ["est_a", "est_b"]
    assert _not_done(engine) == ["zz_spread"]  # the paper states it as not computed


def test_the_same_failure_with_the_name_stored_under_a_longer_name_still_stops(tmp_path: Path) -> None:
    files = {"model.py": STUDY["model.py"].replace("'est_b': 5.0}", "'est_b': 5.0, 'zz_spread_raw': 0.1}")}
    out, _record, _engine_ = _gate(tmp_path, ABSENT, files)
    assert _stops(out)  # the value may be wrong: present under a name that contains the one looked for


def test_a_name_stored_in_a_nested_record_still_stops(tmp_path: Path) -> None:
    files = {"model.py": STUDY["model.py"].replace("'est_b': 5.0}", "'est_b': 5.0, 'more': {'zz_spread': 0.1}}")}
    out, _record, _engine_ = _gate(tmp_path, ABSENT, files)
    assert _stops(out)


def test_a_bare_number_the_code_returned_to_the_test_keeps_the_stop(tmp_path: Path) -> None:
    test = ABSENT.replace("    r = helper_record(1)\n",
                          "    r = helper_record(1)\n    from model import helper_for_trial\n    v = helper_for_trial(1)\n") \
        .replace("    assert found,", "    assert found and v == 2,")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_a_name_read_with_a_default_keeps_the_stop(tmp_path: Path) -> None:
    test = ("from model import helper_record\n\n\ndef test_x():\n    r = helper_record(1)\n"
            "    assert r.get('zz_spread', 0.0) == 0.5\n")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_a_name_a_trial_returned_keeps_the_stop(tmp_path: Path) -> None:
    test = ABSENT.replace("zz_spread", "extra")  # the trial returned `extra_t`
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_a_name_a_derived_result_with_a_value_is_worked_out_from_keeps_the_stop(tmp_path: Path) -> None:
    def plant(root: Path) -> None:
        path = root / "data" / "results" / "derived.json"
        derived = json.loads(path.read_text(encoding="utf-8"))
        derived["derived"]["corr_ab"]["inputs"] = ["est_a", "zz_spread"]
        path.write_text(json.dumps(derived), encoding="utf-8")

    out, _record, _engine_ = _gate(tmp_path, ABSENT, plant=plant)
    assert _stops(out)


def test_a_rerun_that_moved_numbers_is_judged_as_before(tmp_path: Path) -> None:
    moved = Verdict("changed", metrics=["est_b"], keys=["trials/noise=0.1/est_b"])
    out, record, _engine_ = _gate(tmp_path, ABSENT, rerun=moved)
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"  # what moved is left out, not a part not done


def test_a_test_reading_a_key_the_run_holds_is_never_a_part_not_done(tmp_path: Path) -> None:
    test = ABSENT.replace("    found =", "    assert r['est_b'] == 4.0, 'wrong'\n    found =")
    out, record, _engine_ = _gate(tmp_path, test)
    assert (record["issues"]["F-1"].get("at_gate") or {}).get("outcome") != "not_computed"


def test_without_a_record_of_what_the_test_was_given_the_gate_stops_as_before(tmp_path: Path) -> None:
    engine = _engine(tmp_path)  # the re-run is replaced by "still fails", with nothing recorded
    out, _record = _run(engine, ABSENT, file="model.py", function="helper_record", quantity="spread of the estimate",
                        files=STUDY)
    assert _stops(out)


def test_a_finding_the_reader_called_missing_is_a_part_not_done_by_the_same_record(tmp_path: Path) -> None:
    # the reader's word for the quantity is no declared name, so FI's record of the analysis cannot be read by it (on the old
    # gate: a stop); what FI recorded as the test ran can
    import asyncio

    from tests.test_shown_reach import RESULT, _quest

    engine = _engine(tmp_path)
    _real_test_runs(engine)
    root = _quest(engine, STUDY)
    rel = ".fi/finding_tests/F-1_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(ABSENT, encoding="utf-8")
    issue = {"id": "F-1", "file": "", "function": "", "quantity": "relative_spread", "kind": "quantity", "changes_result": True,
             "demonstrated": True, "test": rel, "text": "the relative spread is not computed", "outcome": "not repaired",
             "failure_kind": "assertion", "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1}
    record = {"issues": {"F-1": issue}, "read_problems": [{"id": "R1", "kind": "metric", "status": "missing",
                                                          "name": "relative_spread", "fid": "F-1", "text": "x"}]}
    state = {"design": None, "result_json": RESULT, "exec_result": {"returncode": 0}}
    out = asyncio.run(engine._shown_fault_check([issue], record, RESULT, ["est_a"], state))
    assert out is None
    assert record["issues"]["F-1"]["at_gate"] == {**record["issues"]["F-1"]["at_gate"], "outcome": "not_computed",
                                                  "names": ["zz_spread"]}


# ---- the assertion that failed must only look for the names ------------------------------------------------------------------
def test_a_failed_assertion_that_compares_the_whole_returned_record_keeps_the_stop(tmp_path: Path) -> None:
    # the test still reads the absent name elsewhere, but what failed compares values the code returned
    test = ABSENT.replace("    assert found,", "    assert r == {'est_a': 9.0},")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_a_value_compared_before_the_assertion_keeps_the_stop(tmp_path: Path) -> None:
    test = ABSENT.replace("    assert found,", "    same = r == {'est_a': 9.0}\n    assert found and same,")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_the_absence_asserted_as_a_count_or_with_any_is_a_part_not_done(tmp_path: Path) -> None:
    # known positives: other ways of writing the same absence
    for i, form in enumerate(("    assert len(found) > 0,", "    assert any('zz_spread' in str(k) for k in r),")):
        test = ABSENT.replace("    assert found,", form)
        out, record, _engine_ = _gate(tmp_path / str(i), test)
        assert out is None, form
        assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed", form


def test_the_harness_records_the_failed_line_and_the_names_that_held_nothing(tmp_path: Path) -> None:
    code = {"m.py": "def rec(x):\n    return {'a': x}\n"}
    run = _harness(tmp_path, code, "import m\n\n\ndef test_x():\n    r = m.rec(1)\n    found = []\n    assert found\n", capture=True)
    assert run.held["failures"] == [{"line": 7, "empty": ["found"]}]


def test_an_empty_list_kept_by_value_alone_keeps_the_stop_even_when_the_name_is_looked_for_elsewhere(tmp_path: Path) -> None:
    # the name is read once on the side, but what failed is a list kept by a value: it is empty when the value differs
    test = ("from model import helper_record\n\n\ndef test_x():\n    r = helper_record(1)\n"
            "    'zz_spread' in r\n"
            "    found = [k for k, v in r.items() if v == 0.5]\n"
            "    assert found, 'no value of one half'\n")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


def test_a_value_picked_by_its_value_and_asserted_present_keeps_the_stop(tmp_path: Path) -> None:
    test = ("from model import helper_record\n\n\ndef test_x():\n    r = helper_record(1)\n"
            "    'zz_spread' in r\n"
            "    got = next((v for v in r.values() if v == 0.5), None)\n"
            "    assert got is not None, 'no value of one half'\n")
    out, _record, _engine_ = _gate(tmp_path, test)
    assert _stops(out)


# ---- the recording only looks: it never runs the study's own code ----------------------------------------------------------
def test_a_returned_object_whose_items_changes_a_value_does_not_turn_a_failing_test_into_a_pass(tmp_path: Path) -> None:
    # an outside review's input: the recording called the returned value's own items(), which set the value the test then
    # checked, so the second run passed and the gate cleared a wrong value as a flaky test
    code = {"m.py": ("class Result(dict):\n    def items(self):\n        self['mean'] = 1.0\n        return super().items()\n\n\n"
                     "def run():\n    return Result(mean=0.0)\n")}
    test = "import m\n\n\ndef test_mean():\n    assert m.run()['mean'] == 1.0\n"
    plain = _harness(tmp_path / "a", code, test, capture=False)
    recorded = _harness(tmp_path / "b", code, test, capture=True)
    assert plain.status == "fail" and recorded.status == "fail"
    assert recorded.held["returned"][0]["record"] is False  # an object of the study's own class is not a record of names


def test_the_recording_never_calls_a_method_of_a_value_the_code_returned_or_the_test_held(tmp_path: Path) -> None:
    # every method a recording could reach writes to a file when called; only the test's own lines run them
    hooks = ("keys", "items", "values", "__getitem__", "__iter__", "__len__", "__getattr__", "__repr__", "__str__",
             "__eq__", "__hash__", "__bool__", "__contains__")
    body = "".join(f"    def {h}(self, *a, **k):\n        _called({h!r})\n        return 0\n" for h in hooks)
    code = {"m.py": ("def _called(name):\n    with open('calls.txt', 'a') as fh:\n        fh.write(name + '\\n')\n\n\n"
                     "class Loud(dict):\n" + body + "\n\nclass Thing:\n" + body
                     + "    @property\n    def dtype(self):\n        _called('dtype')\n        return None\n\n\n"
                     "def run():\n    return Loud(a=1), Thing(), [Thing()], {'k': Thing()}\n")}
    test = "import m\n\n\ndef test_x():\n    got = m.run()\n    held = m.Thing()\n    assert not got\n"
    run = _harness(tmp_path, code, test, capture=True)
    assert run.status == "fail" and run.held is not None
    assert not (tmp_path / "calls.txt").is_file(), (tmp_path / "calls.txt").read_text()
    assert [r["record"] for r in run.held["returned"]] == [False] and run.held["complete"] is False


def test_a_wrong_value_in_a_record_whose_items_changes_it_still_stops_the_study(tmp_path: Path) -> None:
    # the outside review's input at the gate: on the old recording the second run passed and the finding was cleared
    files = {"model.py": FILES["model.py"] + ("\n\nclass Result(dict):\n    def items(self):\n        self['est_a'] = 9.0\n"
                                              "        return super().items()\n\n\n"
                                              "def helper_record(x):\n    return Result(est_a=2.0 * x, est_b=5.0)\n")}
    test = ("from model import helper_record\n\n\ndef test_x():\n    r = helper_record(1)\n"
            "    assert r['est_a'] == 9.0, 'the estimate is wrong'\n")
    out, _record, _engine_ = _gate(tmp_path, test, files=files)
    assert _stops(out)
