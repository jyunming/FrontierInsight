"""What each run compares inside it (``protocol.in_each_run``, core/in_run.py): every run goes through all its values
together, on the same random numbers; FI records a number per run once per run and a number per value once per value.

The r13 suite round stopped twice for want of this: the five temperatures one Arrhenius fit spans were made a setting
(125 settings that were 25, five times the time, so the study did not fit), and the three procedures applied to one
simulated family were made a setting (the paired difference recorded three times, so the analysis's honest count of it
looked like two thirds of the trials gone missing)."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

from core import code_project, in_run, plan, protocol_check, run_estimate, run_manifest, topic_requirements, trial_runner
from core.execution import SharedInterpreterExecutor

PAIRED = '''
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    assert cell["procedure"] == ["bonferroni", "bh"], cell     # FI gives the whole list
    found = rng.random() * cell["m"]
    power = {"bonferroni": found / 10.0, "bh": found / 8.0}
    return {"power": power, "power_diff": power["bh"] - power["bonferroni"], "seed_seen": float(seed % 1000)}
'''

INSIDE = {"procedure": ["bonferroni", "bh"]}


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}")


def _run(tmp_path: Path, source: str = PAIRED, *, runs: int = 3) -> tuple[Path, trial_runner.TrialRun]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", {"m": [10, 20]}, runs_per_setting=runs, base_seed=0,
        deterministic=False, timeout_s=60, run_id="r1", in_each_run=INSIDE))
    trial_runner._save_run(root, "k", run)
    return root, run


# --- reading what a run returned -----------------------------------------------------------------------------------------

LEVELS = in_run.level_cells({"T": [300.0, 310.0, 320.0]})


def test_a_number_per_value_is_read_from_a_mapping_or_a_list_in_the_order_given() -> None:
    by_name, why = in_run.read_returned({"a": 5.0, "k": {"300.0": 1.0, 310: 2.0, "T=320.0": 3.0}}, LEVELS, "run_trial")
    assert why == "" and by_name == {"a": 5.0, "k": {"T=300.0": 1.0, "T=310.0": 2.0, "T=320.0": 3.0}}
    as_list, why = in_run.read_returned({"k": [1.0, 2.0, 3.0]}, LEVELS, "run_trial")
    assert why == "" and as_list == {"k": {"T=300.0": 1.0, "T=310.0": 2.0, "T=320.0": 3.0}}


def test_a_number_per_value_must_give_every_value_and_nothing_else() -> None:
    assert "without T=320.0" in in_run.read_returned({"k": {"300": 1.0, "310": 2.0}}, LEVELS, "run_trial")[1]
    assert "not one of the values" in in_run.read_returned({"k": {"300": 1.0, "310": 2.0, "999": 3.0}}, LEVELS,
                                                           "run_trial")[1]
    assert "list of 2" in in_run.read_returned({"k": [1.0, 2.0]}, LEVELS, "run_trial")[1]
    # With nothing compared inside the run, a list is what it always was: not a number.
    assert in_run.read_returned({"k": [1.0, 2.0]}, [], "run_trial") == (None, "run_trial() returned non-numbers for ['k']")


def test_a_value_written_as_a_text_with_a_comma_is_read_back_under_fis_own_key() -> None:
    """A real study compared text values inside each run, one of them with a comma in it; the run returned them under
    the very keys FI gives, and every one was refused, because the key was split back into pairs at its commas."""
    levels = in_run.level_cells({"family": ["plain", "shaped (symmetric, tuned for the target)"], "mode": ["a", "b"]})
    keys = [in_run.level_key(level) for level in levels]
    assert any("," in k.split("family=")[-1] for k in keys)
    got, why = in_run.read_returned({"r": {k: float(i) for i, k in enumerate(keys)}}, levels, "run_cell")
    assert why == "" and got == {"r": {k: float(i) for i, k in enumerate(keys)}}
    # a key that is none of the values is still refused
    assert "not one of the values" in in_run.read_returned({"r": {"family=other,mode=a": 1.0}}, levels, "run_cell")[1]


def test_a_check_reads_a_number_per_value_at_the_value_its_case_names_or_as_the_list() -> None:
    got = {"a": 5.0, "k": {"T=300.0": 1.0, "T=310.0": 2.0, "T=320.0": 3.0}}
    assert in_run.for_case(got, LEVELS, {"T": 310}) == {"a": 5.0, "k": 2.0}
    assert in_run.for_case(got, LEVELS, {}) == {"a": 5.0, "k": [1.0, 2.0, 3.0]}


# --- the plan -------------------------------------------------------------------------------------------------------------

def test_the_plan_states_what_each_run_compares_apart_from_its_settings() -> None:
    ok, why = plan.normalize_protocol({"grid": {"m": [20, 200]}, "in_each_run": {"procedure": ["bonferroni", "bh"]}})
    assert why is None and ok["in_each_run"] == {"procedure": ["bonferroni", "bh"]}
    assert "one or the other" in plan.normalize_protocol({"grid": {"m": [1]}, "in_each_run": {"m": [1, 2]}})[1]
    assert "single value" in plan.normalize_protocol({"in_each_run": {"T": [300]}})[1]


def test_the_topic_check_finds_a_factor_each_run_compares_inside_it() -> None:
    seen = topic_requirements.view({}, {"grid": {"noise": [0.01]}, "in_each_run": {"temperature_K": [290.0, 300.0]}})
    assert seen["axes"]["temperature_K"] == [290.0, 300.0]


def test_a_script_that_names_other_values_for_what_each_run_compares_is_caught() -> None:
    protocol = {"grid": {"m": [20]}, "in_each_run": {"temperature_K": [290.0, 300.0, 310.0]}}
    scripts = {"simulate.py": "TEMPERATURE_K = [290.0, 300.0]\n\ndef run_trial(cell, trial_id, seed):\n    return {}\n"}
    found = protocol_check.check(protocol, scripts)
    assert [m.name for m in found] == ["temperature_K"]


# --- the run and its record -----------------------------------------------------------------------------------------------

def test_each_run_is_one_trial_on_one_seed_and_each_value_has_a_record_of_its_own(tmp_path: Path) -> None:
    root, run = _run(tmp_path)
    ledger = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [r for r in ledger if r["event"] == "trial"]
    assert len(trials) == 6 and {r["cell"] for r in trials} == {"m=10", "m=20"}  # 2 settings x 3 runs, run once each
    summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    keys = [c["key"] for c in summary["cells"]]
    assert keys == ["m=10", "m=10,procedure=bonferroni", "m=10,procedure=bh",
                    "m=20", "m=20,procedure=bonferroni", "m=20,procedure=bh"]
    by_key = {c["key"]: c for c in summary["cells"]}
    assert set(by_key["m=10"]["metrics"]) == {"power_diff", "seed_seen"}  # a number per run, once per run
    assert set(by_key["m=10,procedure=bh"]["metrics"]) == {"power"}        # a number per value, in the value's record
    assert by_key["m=10,procedure=bh"]["cell"] == {"m": 10, "procedure": "bh"}
    assert by_key["m=10,procedure=bh"]["level"] == {"procedure": "bh"}
    # The values of one run come from the same random numbers: the paired difference is exactly theirs.
    bh, bonf = by_key["m=20,procedure=bh"]["metrics"]["power"], by_key["m=20,procedure=bonferroni"]["metrics"]["power"]
    diff = by_key["m=20"]["metrics"]["power_diff"]
    assert bh["trials"] == bonf["trials"] == diff["trials"] == [0, 1, 2]
    assert all(math.isclose(a - b, d) for a, b, d in zip(bh["values"], bonf["values"], diff["values"]))
    # FI's own reading of the record: a number per value under the value's key, a number per run under the run's.
    per_cell, problems = trial_runner.recorded_values_by_cell(root)
    assert problems == []
    assert sum(per_cell["m=10"]["power_diff"].values()) == 3 and "power" not in per_cell["m=10"]
    assert sum(per_cell["m=10,procedure=bh"]["power"].values()) == 3
    assert [c.key for c in run.record_cells()] == keys


def test_a_run_that_returns_a_value_it_was_not_given_fails_with_why(tmp_path: Path) -> None:
    wrong = PAIRED.replace('"bh": found / 8.0}', '"holm": found / 8.0}').replace('power["bh"]', 'power["holm"]')
    _root, run = _run(tmp_path, wrong, runs=1)
    assert run.ok_trials == 0
    assert "not one of the values" in run.cells[0].rows[0]["reason"]


def test_the_count_check_holds_a_number_per_run_to_the_runs_and_a_number_per_value_to_the_values(tmp_path: Path) -> None:
    root, _run_ = _run(tmp_path)
    per_cell, _ = trial_runner.recorded_values_by_cell(root)
    recorded = trial_runner.pooled(per_cell)
    protocol = {"grid": {"m": [10, 20]}, "in_each_run": INSIDE, "runs_per_setting": 3,
                "metrics": [{"id": "power_diff", "kind": "mean"}, {"id": "power", "kind": "mean"}]}
    summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    by_key = {c["key"]: c for c in summary["cells"]}
    honest = {key: {name: {f"{name}_values": m["values"]} for name, m in c["metrics"].items()} for key, c in by_key.items()}
    manifest = {"successful_per_cell": {"m=10": 3, "m=20": 3}}

    def findings(result: dict) -> list[str]:
        return [s for s, _ in run_manifest._value_count_findings(protocol, manifest, result, recorded=recorded,
                                                                 recorded_by_cell=per_cell)]

    assert findings(honest) == []
    # A number per run counted once per value (each run's difference listed for both procedures of its setting) is more
    # than the runs that measured it: caught, the setting named.
    doubled = {**honest, "m=10": {**honest["m=10"], "power_diff": {
        "power_diff_values": by_key["m=10"]["metrics"]["power_diff"]["values"] * 2}}}
    assert any("power_diff" in s and "m=10" in s for s in findings(doubled))


def test_a_check_on_a_case_reads_the_value_its_case_names(tmp_path: Path) -> None:
    root, _ = _run(tmp_path, runs=1)
    at, why = asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py",
                                                cell={"m": 10, "procedure": "bh"}, timeout_s=60, in_each_run=INSIDE))
    assert why == "" and isinstance(at["power"], float)
    every, _ = asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py",
                                                 cell={"m": 10}, timeout_s=60, in_each_run=INSIDE))
    assert isinstance(every["power"], list) and len(every["power"]) == 2
    assert math.isclose(every["power"][1], at["power"])  # the same run's seed as the study's for m=10
    bad, why = asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py",
                                                 cell={"m": 10, "procedure": "holm"}, timeout_s=60, in_each_run=INSIDE))
    assert bad is None and "each run compares procedure" in why


def test_code_run_py_records_the_study_as_fi_does(tmp_path: Path) -> None:
    root, _ = _run(tmp_path)
    fi_summary = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    code = root / "code"
    (code / "run.py").write_text(code_project.RUN_SOURCE, encoding="utf-8")
    (code / "study.json").write_text(json.dumps(code_project.study_of(code, {
        "grid": {"m": [10, 20]}, "in_each_run": INSIDE, "runs_per_setting": 3})), encoding="utf-8")
    (code / "experiment.py").write_text("print('analysis')\n", encoding="utf-8")
    import subprocess

    subprocess.run([sys.executable, str(code / "run.py")], cwd=code, check=True, capture_output=True, timeout=120)
    mine = json.loads((root / "run_output" / "raw" / "trials.json").read_text(encoding="utf-8"))
    assert [c["key"] for c in mine["cells"]] == [c["key"] for c in fi_summary["cells"]]
    assert [c["metrics"] for c in mine["cells"]] == [c["metrics"] for c in fi_summary["cells"]]


def test_the_time_estimate_counts_one_run_per_setting_not_one_per_value() -> None:
    # r13 kinetics: about 25 ms a run (all five temperatures in it), 400 runs per setting. Five temperatures made settings
    # gave 125 x 400 runs, about 21 minutes against 15; the 25 real settings are about 4 minutes.
    grid = {"noise": [0.0025, 0.005, 0.01, 0.02, 0.04], "n_time": [5, 7, 9, 13, 17]}
    probes = [run_estimate.Probe(axis, index, cell, 0.0247 * 400) for axis, index, cell in run_estimate.probe_cells(grid)]
    est = run_estimate.estimate(grid, 400, False, probes)
    assert est is not None and est.cells == 25 and est.trials == 10000
    assert est.seconds < 900


def test_pooled_counts_are_unchanged_for_a_study_that_compares_nothing_inside_its_runs() -> None:
    rows = [{"trial": 0, "status": "ok", "values": {"x": 1.0}}, {"trial": 1, "status": "failed", "reason": "r"}]
    cells = in_run.summary_cells({"a": 1}, "a=1", 2, rows, [])
    assert cells == [{"cell": {"a": 1}, "key": "a=1", "planned": 2, "ok": 1, "failed": 1,
                      "metrics": {"x": {"values": [1.0], "trials": [0], "count": 1, "total": 1.0, "non_finite": 0}}}]
    assert in_run.flat("a=1", {"x": 1.0}) == [("a=1", "x", 1.0)]


def test_a_formula_reads_a_number_per_value_only_inside_min_max_or_sum() -> None:
    from core import oracle_forms
    values = {"k_hat": [1.0, 2.0, 10.0], "c": 2.0}
    assert oracle_forms.evaluate("max(k_hat) / min(k_hat)", values).value == 10.0
    assert oracle_forms.evaluate("sum(k_hat) * c", values).value == 26.0
    elsewhere = oracle_forms.evaluate("k_hat * c", values)
    assert elsewhere.value is None and "one number per value" in elsewhere.problem


def test_the_paper_says_what_each_run_compared_on_the_same_random_numbers(tmp_path: Path) -> None:
    from core import disclosure
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "study.json").write_text(json.dumps(
        {"base_seed": 7, "paired": False, "in_each_run": {"procedure": ["none", "bh"]}}), encoding="utf-8")
    said = disclosure._seed_sentence(tmp_path, None)
    assert "base seed 7" in said and "`procedure` (none, bh)" in said and "same simulated data" in said
    (tmp_path / "code" / "study.json").write_text(json.dumps({"base_seed": 7, "paired": False}), encoding="utf-8")
    assert "Within each run" not in disclosure._seed_sentence(tmp_path, None)


def test_a_collapsed_axis_is_found_in_a_table_of_the_records_per_value() -> None:
    from core import reported_settings
    cells = []
    for m in (20, 200):
        cells.append({"cell": {"m": m}, "metrics": {"fdr_gap": [0.1 * m]}})
        for t in (300, 350, 400):
            cells.append({"cell": {"m": m, "T": t}, "metrics": {"k": [m + t / 7]}})
    rows = [{"m": 20, "T": 300, "z": float(i)} for i in range(6)]
    found = reported_settings.problems(cells, [("", {"table": rows})])
    assert any(f.axis == "T" and f.collapsed for f in found)


def test_the_code_check_lets_the_code_read_what_each_run_compares_and_flags_a_name_it_does_not_have() -> None:
    from core import grid_keys
    protocol = {"grid": {"m": [10, 20]}, "in_each_run": INSIDE}
    ok = {"simulate.py": "def run_trial(cell, trial_id, seed):\n    procs = cell['procedure']\n    return {'x': cell['m']}\n",
          "experiment.py": "def f(entry):\n    return entry['cell']['procedure'], entry['level']\n"}
    assert grid_keys.problems(ok, protocol) == []
    wrong = {"simulate.py": "def run_trial(cell, trial_id, seed):\n    return {'x': cell['procedures']}\n"}
    assert grid_keys.problems(wrong, protocol)
    only_inside = {"grid": {}, "in_each_run": INSIDE}
    assert grid_keys.problems(wrong, only_inside)


def test_a_saved_run_keeps_the_values_each_run_compared(tmp_path: Path) -> None:
    root, run = _run(tmp_path)
    again = trial_runner._load_run(root, "k")
    assert again is not None
    assert [c.key for c in again.record_cells()] == [c.key for c in run.record_cells()]


def test_the_topic_check_writes_a_factor_compared_inside_each_run_back_to_its_list_never_to_the_grid() -> None:
    old = {"protocol": {"grid": {"m": [10, 20]}, "in_each_run": {"T": [300, 350]}, "metrics": []}}
    asked = [{"axis": "T"}]
    in_grid = topic_requirements.merge(old, {"protocol": {"grid": {"m": [10, 20], "T": [300, 350, 400]}}}, asked)
    assert in_grid is not None
    assert in_grid["protocol"]["in_each_run"] == {"T": [300, 350, 400]} and "T" not in in_grid["protocol"]["grid"]
    in_list = topic_requirements.merge(old, {"protocol": {"in_each_run": {"T": [300, 400]}}}, asked)
    assert in_list is not None and in_list["protocol"]["in_each_run"] == {"T": [300, 400]}
    plain = {"protocol": {"grid": {"m": [10]}, "metrics": []}}
    assert "in_each_run" not in (topic_requirements.merge(plain, {"protocol": {"grid": {"m": [10], "n": [1, 2]}}}, []) or {})["protocol"]


def test_a_value_compared_inside_each_run_gets_no_record_when_the_run_returned_no_number_per_value() -> None:
    """s3 kinetics: 25 settings and 1000 empty records (5 temperatures x 4 interval types x 2 centrings), each claiming
    400 ok trials with no metrics, because every number the run returned was one per run."""
    levels = in_run.level_cells({"t": [1.0, 2.0], "kind": ["a", "b"]})
    rows = [{"trial": i, "status": "ok", "values": {"x": float(i)}} for i in range(3)]
    cells = in_run.summary_cells({"n": 5}, "n=5", 3, rows, levels)
    assert [c["key"] for c in cells] == ["n=5"] and cells[0]["metrics"]["x"]["count"] == 3
    # a run that does return a number per value still gets one record per value
    per_value = [{"trial": i, "status": "ok", "values": {"x": float(i), "k": {in_run.level_key(lv): 1.0 for lv in levels}}}
                 for i in range(3)]
    kept = in_run.summary_cells({"n": 5}, "n=5", 3, per_value, levels)
    assert len(kept) == 1 + len(levels) and all(c["metrics"]["k"]["count"] == 3 for c in kept[1:])


def test_none_is_read_as_no_value_inside_a_run_too() -> None:
    got, why = in_run.read_returned({"a": None, "k": {"300.0": 1.0, "310.0": None, "320.0": 3.0}}, LEVELS, "run_trial")
    assert why == "" and math.isnan(got["a"]) and math.isnan(got["k"]["T=310.0"])
    as_list, why = in_run.read_returned({"k": [1.0, None, 3.0]}, LEVELS, "run_trial")
    assert why == "" and math.isnan(as_list["k"]["T=310.0"])
    assert "non-numbers" in in_run.read_returned({"k": "none"}, [], "run_trial")[1]


NUMERIC_INSIDE = {"temperature": [300.0, 320.0]}
NUMERIC = '''
def run_trial(cell, trial_id, seed):
    return {"rate": [cell["scale"] * t / 100.0 for t in cell["temperature"]]}
'''


def test_a_check_on_a_number_each_run_does_not_compare_is_run_with_that_number_added(tmp_path: Path) -> None:
    """A real plan compared two values of a number inside each run and checked a limiting case at a third value; FI
    refused to run the case and sent the code back for a repair no code can make."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(NUMERIC, encoding="utf-8")
    at, why = asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py",
                                                cell={"scale": 2.0, "temperature": 310.0}, timeout_s=60,
                                                in_each_run=NUMERIC_INSIDE))
    assert why == "" and math.isclose(at["rate"], 6.2), (at, why)
    rows, why, _later, started = asyncio.run(trial_runner.run_case_trials(
        _executor(), sys.executable, root, "code/simulate.py", cell={"scale": 2.0, "temperature": 310.0}, trials=[0, 1],
        timeout_s=60, in_each_run=NUMERIC_INSIDE))
    assert why == "" and started == 2 and len(rows) == 2, (rows, why)
    assert trial_runner.case_inside(NUMERIC_INSIDE, {"temperature": 320.0}) == NUMERIC_INSIDE, "a value it has: as it is"
    assert trial_runner.case_inside(INSIDE, {"procedure": "other"}) == INSIDE, "a choice it does not have: not added"


SEVEN_LISTS = {"pitch": list(range(61)), "distance": list(range(83)), "family": list(range(5)), "kind": list(range(3)),
               "rep": list(range(4)), "state": [0, 1], "method": [0, 1]}


def test_a_run_given_long_lists_that_returns_one_number_per_run_stays_small(tmp_path: Path) -> None:
    """A live plan put seven lists in each run (1,215,120 combinations); its runs returned plain numbers, and FI wrote every
    combination into each setting's spec (251 MB each) and held them (a 30 GB process)."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(
        "def run_cell(cell):\n    return {'y': float(len(cell['pitch']) + cell['m'])}\n", encoding="utf-8")
    sizes: list[int] = []
    run = asyncio.run(trial_runner.run_trials(
        _executor(), sys.executable, root, "code/simulate.py", {"m": [1, 2]}, runs_per_setting=1, base_seed=0,
        deterministic=True, timeout_s=120, run_id="r1", in_each_run=SEVEN_LISTS,
        on_plan=lambda plan: sizes.extend((root / t["spec"]).stat().st_size for t in plan)))
    assert len(sizes) == 2 and max(sizes) < 20_000, sizes
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert [c["key"] for c in summary["cells"]] == ["m=1", "m=2"]
    assert all("compared_in_each_run" not in c for c in summary["cells"])
    assert [c["metrics"]["y"]["values"] for c in summary["cells"]] == [[62.0], [63.0]]
    trial_runner._save_run(root, "k", run)
    again = trial_runner._load_run(root, "k")
    assert again is not None and len(again.cells[0].levels) == 1_215_120


def test_the_combinations_are_read_one_at_a_time_in_the_order_they_were_listed() -> None:
    levels = in_run.level_cells({"a": [1, 2], "b": ["x", "y", "z"]})
    assert list(levels) == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 1, "b": "z"},
                            {"a": 2, "b": "x"}, {"a": 2, "b": "y"}, {"a": 2, "b": "z"}]
    assert len(levels) == 6 and levels[4] == {"a": 2, "b": "y"} and levels[-1] == {"a": 2, "b": "z"}
    assert levels[:2] == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}]
    assert in_run.levels_of(json.loads(json.dumps(in_run.levels_json(levels)))) == levels
    assert in_run.level_cells({}) == [] and in_run.levels_of([{"a": 1}]) == [{"a": 1}]


def test_a_pilot_run_reads_a_number_per_value_from_the_lists_it_was_given(tmp_path: Path) -> None:
    """The time estimate's and the values pilot's specs wrote the combinations with ``default=str``: the run then read a
    string as its levels and a number per value broke it."""
    from core import run_estimate

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(
        "def run_cell(cell):\n    return {'y': {str(t): t * 2.0 for t in cell['T']}, 'n': 1.0}\n", encoding="utf-8")
    got = asyncio.run(run_estimate.run_values(
        _executor(), sys.executable, root, "code/simulate.py", {"m": 1}, deterministic=True, timeout_s=60, env=None,
        thresholds=None, in_each_run={"T": [1.0, 2.0]}))
    assert got is not None and got["n"] == 1.0 and got["y"] == {"T=1.0": 2.0, "T=2.0": 4.0}, got


TWO_INSIDE = {"method": ["fast", "exact"], "focus": [0.0, 100.0]}
TWO = '''
def run_trial(cell, trial_id, seed):
    base = {"fast": 1.0, "exact": 2.0}
    return {"signal": {f"method={m},focus={f}": base[m] + f / 100.0
                       for m in cell["method"] for f in cell["focus"]}}
'''


def test_a_case_that_names_only_some_of_the_values_compared_in_each_run_is_run_at_every_value_it_leaves_out(
        tmp_path: Path) -> None:
    """A real plan compared two things inside each run and wrote its known-answer cases with one of them (a focus); FI
    refused every such case ("a case names one of the values each run is given for every one of them") and the checks
    stayed unjudged, the code sent to a repair no code can make. The case now reads the number at each value of what
    it leaves out; a value the run is not given is still refused."""
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(TWO, encoding="utf-8")

    def case(cell):
        return asyncio.run(trial_runner.run_case(_executor(), sys.executable, root, "code/simulate.py", cell=cell,
                                                 timeout_s=60, in_each_run=TWO_INSIDE))

    some, why = case({"focus": 100.0})
    assert why == "" and some["signal"] == [2.0, 3.0], (some, why)  # fast, exact at focus 100
    one, why = case({"focus": 100.0, "method": "exact"})
    assert why == "" and one["signal"] == 3.0
    none, why = case({})
    assert why == "" and sorted(none["signal"]) == [1.0, 2.0, 2.0, 3.0]
    bad, why = case({"focus": 100.0, "method": "other"})
    assert bad is None and "no value each run is given has that" in why  # the known positive: a wrong value is still refused
    assert in_run.levels_named({"focus": 100.0}, in_run.level_cells(TWO_INSIDE)) == [
        {"method": "fast", "focus": 100.0}, {"method": "exact", "focus": 100.0}]
