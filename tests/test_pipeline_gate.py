"""The analysis gate (core/pipeline_gate.py): a change is refused when the analysis, run from start to end on a small copy
of the quest's own raw, has a fault the code before the change did not have. Neutral made-up study: a "tank" whose
fill time is measured at a few pump rates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core import pipeline_gate as gate

OK_SCRIPT = '''
import json, os

def summarise(cells):
    return {c["key"]: sum(c["metrics"]["fill_s"]["values"]) for c in cells}

def main():
    data = json.load(open(os.environ["FI_TRIALS"]))
    print("RESULT_JSON: " + json.dumps(summarise(data["cells"])))

if __name__ == "__main__":
    main()
'''


def make_quest(root: Path, trials: int = 5, settings: int = 3) -> Path:
    (root / "raw").mkdir(parents=True)
    cells = []
    for i in range(settings):
        vals = [10.0 + i + t * 0.1 for t in range(trials)]
        cells.append({"cell": {"rate": i + 1}, "key": f"rate={i + 1}", "planned": trials, "ok": trials, "failed": 0,
                      "metrics": {"fill_s": {"values": vals, "count": trials, "total": sum(vals), "non_finite": 0}}})
    (root / "raw" / "trials.json").write_text(json.dumps({"schema": "x", "cells": cells}), encoding="utf-8")
    return root


def make_code(root: Path, script: str) -> Path:
    code = root / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text(script, encoding="utf-8")
    return code


async def run_gate(tmp_path: Path, script: str, name: str = "q", **kw) -> gate.GateResult:
    quest = make_quest(tmp_path / name)
    code = make_code(tmp_path / f"{name}_code", script)
    return await gate.analysis_gate(quest, code, None, timeout_s=60, **kw)


async def test_a_script_that_runs_to_the_end_passes(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT)
    assert got.ok and not got.kind


async def test_a_definition_after_the_main_guard_is_a_fault(tmp_path):
    script = OK_SCRIPT.replace("def summarise", "def unused").replace('summarise(data["cells"])', "later(data)")
    script += "\n\ndef later(data):\n    return {}\n"
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.FAULT and got.stage == "static"
    assert got.late == ["experiment.py:later"] and "line" in got.where
    assert "main guard" not in got.detail and "starts the script" in got.detail


def test_a_definition_after_the_guard_that_nothing_reaches_is_not_a_fault():
    source = OK_SCRIPT + "\n\ndef spare():\n    return 1\n"
    assert gate.late_definitions(source) == {}


def test_a_definition_after_the_guard_reached_through_a_function_is_found():
    source = OK_SCRIPT.replace("return {c", "return helper() or {c") + "\n\ndef helper():\n    return None\n"
    assert list(gate.late_definitions(source)) == ["helper"]


async def test_an_undefined_name_is_a_fault(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT.replace("sum(", "total_of("))
    assert not got.ok and got.kind == gate.FAULT and got.stage == "derive"
    assert got.exception == "NameError" and "experiment.py" in got.where


async def test_an_error_from_too_little_data_is_small_data_and_is_not_refused(tmp_path):
    script = OK_SCRIPT.replace('print("RESULT_JSON', 'if len(data["cells"][0]["metrics"]["fill_s"]["values"]) < 4:\n'
                               '        raise ValueError("need at least 4 trials for the fit")\n    print("RESULT_JSON')
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.SMALL_DATA and got.exception == "ValueError"
    verdict = gate.gate_verdict(gate.GateResult(True), got)
    assert not verdict.reject and verdict.disclose


async def test_a_run_past_its_time_is_time_and_is_not_refused(tmp_path):
    quest = make_quest(tmp_path / "q")
    code = make_code(tmp_path / "c", "import time\ntime.sleep(30)\n")
    got = await gate.analysis_gate(quest, code, None, timeout_s=2)
    assert got.kind == gate.TIME
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


async def test_the_not_checked_record_of_a_search_reaches_the_analysis(tmp_path):
    script = ('import json, os\ncheck = json.load(open(os.environ["FI_OPTIMUM_CHECK"]))\n'
              'says = (check or {}).get("best", {}).get("says")\nprint("RESULT_JSON: {}")\n')
    got = await run_gate(tmp_path, script, search=True)
    assert not got.ok and got.kind == gate.FAULT and got.exception == "AttributeError"


def test_the_small_copy_keeps_three_trials_of_every_setting(tmp_path):
    quest = make_quest(tmp_path / "q", trials=8, settings=4)
    assert gate.small_raw(quest, tmp_path / "scratch")
    cells = json.loads((tmp_path / "scratch" / "raw" / "trials.json").read_text(encoding="utf-8"))["cells"]
    assert [c["key"] for c in cells] == [f"rate={i}" for i in (1, 2, 3, 4)]
    assert all(len(c["metrics"]["fill_s"]["values"]) == 3 and c["metrics"]["fill_s"]["count"] == 3 for c in cells)
    assert all(c["ok"] == 3 for c in cells)


def test_a_change_is_refused_when_it_passed_before_and_has_a_new_fault():
    after = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 9", "NameError: x", "NameError")
    assert gate.gate_verdict(gate.GateResult(True), after).reject


def test_a_change_is_not_refused_when_the_code_already_failed_before():
    before = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 5", "TypeError: y", "TypeError")
    after = gate.GateResult(False, "derive", gate.FAULT, "code/experiment.py line 9", "NameError: x", "NameError")
    assert not gate.gate_verdict(before, after).reject
    assert not gate.gate_verdict(None, after).reject


def test_a_definition_after_the_guard_that_the_code_before_did_not_have_is_refused_even_when_it_stopped_later():
    before = gate.GateResult(False, "derive", gate.FAULT, "", "AttributeError", "AttributeError")
    after = gate.GateResult(False, "static", gate.FAULT, "code/experiment.py line 9", "`f` is defined after", "NameError",
                            late=["experiment.py:f"])
    assert gate.gate_verdict(before, after).reject
    same = gate.GateResult(False, "static", gate.FAULT, "", "x", "NameError", late=["experiment.py:f"])
    assert not gate.gate_verdict(same, after).reject


def _raise_if_short(exc: str, below: int) -> str:
    return OK_SCRIPT.replace('print("RESULT_JSON', 'if len(data["cells"][0]["metrics"]["fill_s"]["values"]) < '
                             f'{below}:\n        {exc}\n    print("RESULT_JSON')


async def test_a_type_error_that_only_3_trials_cause_is_small_data_and_is_not_refused(tmp_path):
    got = await run_gate(tmp_path, _raise_if_short("raise TypeError('need more points than parameters')", 4))
    assert not got.ok and got.kind == gate.SMALL_DATA and got.exception == "TypeError"
    assert "confirm" in got.seconds and "too few trials" in got.detail
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


async def test_an_attribute_error_that_shows_on_more_trials_at_the_same_place_is_a_fault(tmp_path):
    got = await run_gate(tmp_path, _raise_if_short("None.fill_rate", 99))
    assert got.kind == gate.FAULT and got.exception == "AttributeError" and "confirm" in got.seconds
    assert gate.gate_verdict(gate.GateResult(True), got).reject


async def test_a_name_error_is_a_fault_at_once_without_a_second_run(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT.replace("sum(", "total_of("))
    assert got.kind == gate.FAULT and "confirm" not in got.seconds


async def test_a_type_error_with_no_more_trials_on_record_is_the_same_run_and_stays_a_fault(tmp_path):
    quest = make_quest(tmp_path / "q", trials=2)
    code = make_code(tmp_path / "c", _raise_if_short("raise TypeError('x')", 99))
    got = await gate.analysis_gate(quest, code, None, timeout_s=60)
    assert got.kind == gate.FAULT and "confirm" not in got.seconds


async def test_a_run_that_ends_without_its_result_line_is_a_fault(tmp_path):
    got = await run_gate(tmp_path, "print('done')\n")
    assert not got.ok and got.kind == gate.FAULT and "RESULT_JSON" in got.detail


TB = 'Traceback (most recent call last):\n  File "a.py", line 3, in f\n    x.y\n'


def test_last_exception_alone():
    assert gate.last_exception(TB + "AttributeError: 'NoneType' object has no attribute 'y'\n") == "AttributeError"
    assert gate.last_exception("") == "" and gate.last_exception("[run] gave up\n") == ""
    chained = (TB + "KeyError: 'a'\n\nDuring handling of the above exception, another exception occurred:\n\n"
               + TB + "scipy.linalg.LinAlgError: singular\n")
    assert gate.last_exception(chained) == "LinAlgError"
    assert gate.last_exception('  File "a.py", line 1\n    x = (\n        ^\n') == ""
    assert gate.last_exception(TB + "TypeError: bad\n    a note line\n") == "TypeError"


def test_a_traceback_the_script_handled_is_not_what_stopped_it():
    assert gate.last_exception(TB + "AttributeError: z\n[run] gave up: too few points\n") == ""


async def test_a_handled_traceback_followed_by_a_message_is_small_data(tmp_path):
    script = ("import sys, traceback\ntry:\n    None.z\nexcept AttributeError:\n    traceback.print_exc()\n"
              "sys.exit('[run] gave up: too few points')\n")
    got = await run_gate(tmp_path, script)
    assert got.kind == gate.SMALL_DATA and got.exception == ""


def test_a_loop_or_comprehension_variable_after_the_guard_is_not_a_late_definition():
    source = (OK_SCRIPT.replace("    main()", "    out = [i for i in range(3)]\n    main()")
              + "\nfor i in range(2):\n    pass\n")
    assert gate.late_definitions(source) == {}


def test_a_helper_modules_own_guard_is_not_looked_at(tmp_path):
    code = make_code(tmp_path / "c", OK_SCRIPT)
    guarded = 'if __name__ == "__main__":\n    selftest()\n\ndef selftest():\n    pass\n'
    (code / "helpers.py").write_text(guarded, encoding="utf-8")
    assert gate.late_definitions_in(code) == []
    (code / "simulate.py").write_text(guarded, encoding="utf-8")
    assert [n for _f, n, _l in gate.late_definitions_in(code)] == ["selftest"]


async def test_the_gate_runs_with_a_plain_executor_that_commits_nothing(tmp_path, monkeypatch):
    from core.execution import VenvExecutor

    seen = []
    real = VenvExecutor.execute

    async def spy(self, *a, **kw):
        seen.append(self.before_run)
        return await real(self, *a, **kw)

    monkeypatch.setattr(VenvExecutor, "execute", spy)
    got = await run_gate(tmp_path, OK_SCRIPT)
    assert got.ok and seen == [None]


async def test_under_docker_the_gate_says_it_did_not_check(tmp_path):
    got = await run_gate(tmp_path, OK_SCRIPT, sandbox="docker")
    assert got.kind == gate.NOT_CHECKED and "Docker" in got.detail and not got.ok
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


def test_the_quests_own_venv_python_is_used_when_it_exists(tmp_path):
    assert gate._quest_python(tmp_path) == sys.executable
    exe = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    assert gate._quest_python(tmp_path) == str(exe)


def test_a_setting_with_no_good_trial_stays_with_none_and_failed_rows_agree(tmp_path):
    quest = make_quest(tmp_path / "q", trials=5, settings=2)
    path = quest / "raw" / "trials.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cells"][0].update(ok=0, failed=5)
    data["cells"][0]["metrics"]["fill_s"]["values"] = []
    data["cells"][1].update(ok=4, failed=1)
    path.write_text(json.dumps(data), encoding="utf-8")
    rows = [{"event": "trial", "cell": "rate=2", "trial": t, "status": "failed" if t in (1, 4) else "ok"}
            for t in range(5)]
    (quest / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert gate.small_raw(quest, tmp_path / "s")
    got = json.loads((tmp_path / "s" / "raw" / "trials.json").read_text(encoding="utf-8"))["cells"]
    assert got[0]["ok"] == 0
    assert (got[1]["ok"], got[1]["failed"], got[1]["planned"]) == (2, 1, 3)
    kept = [json.loads(ln) for ln in (tmp_path / "s" / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in kept if r["status"] == "failed") == got[1]["failed"]


def test_raw_subfolders_are_copied_and_a_huge_folder_is_left_out(tmp_path, monkeypatch):
    quest = make_quest(tmp_path / "q")
    (quest / "raw" / "checks").mkdir()
    (quest / "raw" / "checks" / "a.json").write_text("{}", encoding="utf-8")
    (quest / "results").mkdir()
    (quest / "results" / "big.bin").write_bytes(b"0" * 4096)
    (quest / "needs").mkdir()
    (quest / "needs" / "n.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gate, "COPY_CAP", 1024)
    monkeypatch.setattr(gate, "DATA_CAP", 1024)
    scratch = tmp_path / "scratch"
    (scratch / "raw").mkdir(parents=True)
    gate._copy_quest_folders(quest, scratch)
    assert (scratch / "raw" / "checks" / "a.json").is_file() and (scratch / "needs" / "n.json").is_file()
    assert not (scratch / "results").exists()


def test_what_a_person_reads_has_no_internal_ids():
    r = gate.GateResult(False, "derive", gate.SMALL_DATA, "code/experiment.py line 9", "ValueError: x", "ValueError")
    assert "small_data" not in r.says() and "too few trials" in r.says() and "derive" not in r.says()


async def test_a_folder_too_large_to_copy_makes_an_attribute_error_small_data_not_a_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "COPY_CAP", 1024)
    monkeypatch.setattr(gate, "DATA_CAP", 1024)
    quest = make_quest(tmp_path / "q")
    (quest / "inputs").mkdir()
    (quest / "inputs" / "big.bin").write_bytes(b"0" * 4096)
    script = ("import os\n\ndef load():\n    return open('inputs/big.bin', 'rb').read() if os.path.exists("
              "'inputs/big.bin') else None\n\nb = load()\nprint(b.hex())\n")
    code = make_code(tmp_path / "c", script)
    got = await gate.analysis_gate(quest, code, None, timeout_s=60)
    assert got.kind == gate.SMALL_DATA and got.exception == "AttributeError" and got.left_out == ["inputs"]
    assert "too large to copy" in got.detail
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


def test_failed_trials_among_the_first_three_keep_ok_and_count_consistent(tmp_path):
    quest = make_quest(tmp_path / "q", trials=6, settings=1)
    path = quest / "raw" / "trials.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cells"][0].update(ok=4, failed=2)
    path.write_text(json.dumps(data), encoding="utf-8")
    rows = [{"event": "trial", "cell": "rate=1", "trial": t, "status": "failed" if t == 1 else "ok"} for t in range(6)]
    (quest / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert gate.small_raw(quest, tmp_path / "s")
    cell = json.loads((tmp_path / "s" / "raw" / "trials.json").read_text(encoding="utf-8"))["cells"][0]
    assert (cell["ok"], cell["failed"], cell["planned"]) == (2, 1, 3)
    assert cell["metrics"]["fill_s"]["count"] == cell["ok"] == len(cell["metrics"]["fill_s"]["values"])


async def test_a_read_the_records_shape_refuses_is_a_fault_not_too_few_trials(tmp_path):
    # FI's frame raises UnknownResult for an id the plan does not declare or a result read in a form the record does not
    # hold; the small copy keeps the record's shape, so more trials never change it.
    script = ("class UnknownResult(KeyError):\n    pass\n\n"
              + OK_SCRIPT.replace('print("RESULT_JSON', 'raise UnknownResult("`x` is recorded for each value of `m`")\n    print("RESULT_JSON'))
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.stage == "derive" and got.kind == gate.FAULT and got.exception == "UnknownResult"
    assert gate.kind_of("UnknownResult") == gate.FAULT
    assert gate.gate_verdict(gate.GateResult(True), got).reject


async def test_a_division_by_zero_from_too_few_trials_stays_small_data(tmp_path):
    script = OK_SCRIPT.replace('print("RESULT_JSON', 'print(1 / (len(data["cells"][0]["metrics"]["fill_s"]["values"]) - 3))\n    print("RESULT_JSON')
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.SMALL_DATA and got.exception == "ZeroDivisionError"
    assert not gate.gate_verdict(gate.GateResult(True), got).reject


async def test_a_name_refused_by_fis_figure_calls_is_a_fault_on_any_copy(tmp_path):
    # The frame's figure calls raise UnknownResult (a ValueError) for a result the plan does not declare. The refusal does not
    # depend on the data, so the gate sends it to the repair as a fault instead of blaming a small copy.
    from core import frame_figure
    script = ("class UnknownResult(ValueError):\n    __module__ = 'fi_figure_calls'\n\n"
              + OK_SCRIPT.replace('print("RESULT_JSON', 'raise UnknownResult("`what=\'x 0.3\'` is not a result of the plan")\n    print("RESULT_JSON'))
    assert issubclass(frame_figure.UnknownResult, ValueError) and issubclass(frame_figure.BadArgument, ValueError)
    got = await run_gate(tmp_path, script)
    assert not got.ok and got.kind == gate.FAULT and got.exception == "UnknownResult"
    assert gate.gate_verdict(gate.GateResult(True), got).reject
    plain = OK_SCRIPT.replace('print("RESULT_JSON', 'raise ValueError("what is not a result of the plan")\n    print("RESULT_JSON')
    other = await run_gate(tmp_path, plain, name="p")  # the same words in a plain ValueError keep the small-copy treatment
    assert other.kind == gate.SMALL_DATA and other.exception == "ValueError"
    assert gate.kind_of("BadArgument") == gate.FAULT and gate.kind_of("ValueError") == gate.SMALL_DATA
