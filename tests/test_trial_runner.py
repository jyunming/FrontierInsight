"""FI runs the trials and writes their record (core/trial_runner.py): the experiment's code only says what a trial does."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

from core import run_manifest, trial_runner
from core.execution import SharedInterpreterExecutor

SIM = '''
import random, warnings, sys

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    print("noise the simulation prints")               # must never pass for a result
    if cell["rate_a"] == 2.0 and trial_id == 1:
        raise ValueError("diverged")
    if trial_id == 2:
        warnings.warn("overflow encountered in exp", RuntimeWarning)
    return {"tally": cell["rate_a"] * 100 + rng.random(), "peak_step": 10.0 + trial_id}
'''

CRASH = '''
import os

def run_trial(cell, trial_id, seed):
    if trial_id == 1:
        os._exit(7)                                       # the process dies part-way through the cell
    return {"x": 1.0}
'''

DET = '''
def run_cell(cell):
    return {"energy_drift": cell["dt"] * 2}
'''


def _run(tmp_path: Path, source: str, grid: dict, *, runs: int = 3, deterministic: bool = False, timeout: int = 60):
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    return root, asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
        sys.executable, root, "code/simulate.py", grid, runs_per_setting=runs, base_seed=0,
        deterministic=deterministic, timeout_s=timeout, run_id="r1",
    ))


def test_cells_and_seeds_are_fixed_by_the_protocol() -> None:
    grid = {"rate_a": [1.5, 2.0], "N": [100]}
    assert trial_runner.cells(grid) == [{"rate_a": 1.5, "N": 100}, {"rate_a": 2.0, "N": 100}]
    assert trial_runner.cell_key({"rate_a": 1.5, "N": 100}) == "rate_a=1.5,N=100"
    a = trial_runner.trial_seed(0, "rate_a=1.5,N=100", 0)
    assert a == trial_runner.trial_seed(0, "rate_a=1.5,N=100", 0)
    assert len({trial_runner.trial_seed(0, k, t) for k in ("rate_a=1.5,N=100", "rate_a=2.0,N=100") for t in range(50)}) == 100


def test_fi_writes_the_ledger_and_the_per_cell_summary(tmp_path: Path) -> None:
    root, run = _run(tmp_path, SIM, {"rate_a": [1.5, 2.0]})
    assert run.ok_trials == 5 and run.failed_trials == 1
    rows = trial_runner.read_ledger(root)
    assert len(rows) == 6 and {r["status"] for r in rows} == {"ok", "failed"}
    failed = next(r for r in rows if r["status"] == "failed")
    assert failed["cell"] == "rate_a=2.0" and "diverged" in failed["reason"]
    # The existing checker reads FI's rows, and finds the grid and counts complete.
    manifest, row_problems = run_manifest.manifest_from_ledger({"grid": {"rate_a": [1.5, 2.0]}, "runs_per_setting": 3}, rows, {})
    assert not row_problems and len(manifest["failed_trials"]) == 1
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    first = summary["cells"][0]
    assert first["key"] == "rate_a=1.5" and first["ok"] == 3 and first["metrics"]["tally"]["count"] == 3
    assert 150 < first["metrics"]["tally"]["total"] / 3 < 151
    assert "noise the simulation prints" in run.stderr() and "overflow encountered" not in json.dumps(summary)
    from core import numeric_warnings
    assert any("overflow encountered in exp" in str(w) for w in numeric_warnings.scan(run.stderr(), {})),         "a warning inside a trial reaches the numeric check"
    events = [json.loads(line) for line in run.ledger_path.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events][:1] == ["planned"] and events[-1]["warnings"] in (0, 1)


def test_the_spec_file_is_gone_before_the_simulation_loads(tmp_path: Path) -> None:
    """The spec names the nonce; the harness deletes it before loading the simulation. On Windows a file still open
    cannot be deleted, so this also checks the harness closed it first."""
    peek = '''
import glob
SEEN = len(glob.glob("**/cell*.json", recursive=True))

def run_trial(cell, trial_id, seed):
    return {"spec_files_seen": float(SEEN)}
'''
    _root, run = _run(tmp_path, peek, {"a": [1]}, runs=1)
    assert run.ok_trials == 1 and run.cells[0].rows[0]["values"]["spec_files_seen"] == 0.0


def test_trials_a_crashed_process_never_reported_are_failed_with_why(tmp_path: Path) -> None:
    root, run = _run(tmp_path, CRASH, {"a": [1]})
    statuses = [r.get("status") for r in run.cells[0].rows]
    assert statuses == ["ok", "failed", "failed"]
    assert "exit code 7" in run.cells[0].rows[1]["reason"]


def test_a_simulation_without_the_function_fails_every_trial_and_says_so(tmp_path: Path) -> None:
    root, run = _run(tmp_path, "x = 1\n", {"a": [1]}, runs=2)
    assert run.failed_trials == 2 and "has no function run_trial()" in run.cells[0].rows[0]["reason"]


def test_a_deterministic_study_runs_each_cell_once(tmp_path: Path) -> None:
    root, run = _run(tmp_path, DET, {"dt": [0.1, 0.01]}, deterministic=True)
    assert [len(c.rows) for c in run.cells] == [1, 1] and run.ok_trials == 2
    assert run.cells[1].rows[0]["values"]["energy_drift"] == 0.02


def test_the_harness_is_written_fresh_so_an_edited_one_never_runs(tmp_path: Path) -> None:
    root, _ = _run(tmp_path, SIM, {"rate_a": [1.5]})
    harness = root / trial_runner.HARNESS_PATH
    harness.write_text("print('tampered')", encoding="utf-8")
    asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", {"rate_a": [1.5]},
        runs_per_setting=1, base_seed=0, deterministic=False, timeout_s=60,
    ))
    assert "tampered" not in harness.read_text(encoding="utf-8")


ANALYSIS = '''
import json, os
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
cells = {c["key"]: c["metrics"]["tally"]["count"] for c in data["cells"]}
print("RESULT_JSON: " + json.dumps({"counts": cells}))
'''

ORACLE = '''
def run_trial(cell, trial_id, seed):
    return {"x": 1.0}

def oracle():
    return {"closed_form": 0.5}
'''


def test_the_runner_runs_every_trial_then_the_analysis_on_fis_summary(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    assert trial_runner.entries(root / "code" / "simulate.py") == {"run_trial"}
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
        protocol={"grid": {"rate_a": [1.5, 2.0]}, "runs_per_setting": 3}, deterministic=False,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
    )
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0 and '"rate_a=2.0": 2' in result.stdout
    assert "noise the simulation prints" in result.stderr, "the simulation's stderr reaches the numeric check"
    from core import numeric_warnings
    assert any("overflow encountered in exp" in str(w) for w in numeric_warnings.scan(result.stderr, {}))
    assert runner.last is not None and runner.last.failed_trials == 1 and runner.failed_script is None


def test_the_oracle_is_its_own_function(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(ORACLE, encoding="utf-8")
    values, why = asyncio.run(trial_runner.run_oracle(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", timeout_s=60))
    assert values == {"closed_form": 0.5} and not why
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    values, why = asyncio.run(trial_runner.run_oracle(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", timeout_s=60))
    assert values is None and "has no function oracle()" in why


def test_unchanged_trials_are_not_run_again_for_a_new_analysis(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    calls: list[list[str]] = []

    class Counting(SharedInterpreterExecutor):
        async def execute(self, cmd, **kw):  # noqa: ANN001, ANN003
            calls.append(cmd)
            return await super().execute(cmd, **kw)

    def runner():  # noqa: ANN202
        return trial_runner.TrialsRunner(
            Counting(python_version="3.11"), quest_root=root, protocol=lambda: {"grid": {"rate_a": [1.5]}, "runs_per_setting": 2},
            deterministic=False, simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
        )

    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 2  # one cell, then the analysis
    again = runner()
    asyncio.run(again.execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 3 and again.last is not None and again.last.ok_trials == 2, "only the analysis ran again"
    (root / "code" / "simulate.py").write_text(SIM + "\n# changed\n", encoding="utf-8")
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert len(calls) == 5, "a changed simulation runs its trials again"


FORGER = '''
import json, sys, glob

def run_trial(cell, trial_id, seed):
    # Looks for the results file the way a forger would: argv, then the spec files FI wrote.
    for arg in sys.argv[1:]:
        open(arg, "a").write(json.dumps({"trial": trial_id, "seed": seed, "status": "ok", "values": {"x": 999.0}}) + "\\n")
    for path in glob.glob(".fi/trials/*.out.jsonl"):
        open(path, "a").write(json.dumps({"trial": trial_id, "seed": seed, "status": "ok", "values": {"x": 999.0}}) + "\\n")
    return {"x": 1.0}
'''

SLOW = '''
import time

def run_trial(cell, trial_id, seed):
    time.sleep(3)
    return {"x": 1.0}
'''


def test_rows_the_simulation_writes_itself_are_not_taken_and_a_trial_reported_twice_fails(tmp_path: Path) -> None:
    root, run = _run(tmp_path, FORGER, {"a": [1]}, runs=2)
    values = [r.get("values") for r in run.cells[0].rows]
    assert all(v != {"x": 999.0} for v in values), "a row without the cell's nonce is not the harness's"
    rows = run.cells[0].rows
    assert all(r["status"] == "ok" or "more than once" in r.get("reason", "") for r in rows)


def test_timeout_s_bounds_the_whole_study_not_each_setting(tmp_path: Path) -> None:
    # Three settings of 3 s each under a 4 s budget: bounded per setting this took 9 s or more; bounded for the study it
    # stops near 4 s. Whether the first setting finishes depends on how fast a process starts on the machine, so only
    # the bound and the reason are asserted.
    import time as _time

    started = _time.monotonic()
    root, run = _run(tmp_path, SLOW, {"a": [1, 2, 3]}, runs=1, timeout=4)
    elapsed = _time.monotonic() - started
    reasons = [r.get("reason", "") for c in run.cells for r in c.rows]
    assert elapsed < 8.5, elapsed
    assert "the study's time (execution.timeout_s) ran out before this setting's trial" in reasons[-1], reasons


def test_a_ledger_edited_by_hand_makes_the_trials_run_again(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    protocol = {"grid": {"rate_a": [1.5]}, "runs_per_setting": 2}

    def runner():  # noqa: ANN202
        return trial_runner.TrialsRunner(
            SharedInterpreterExecutor(python_version="3.11"), quest_root=root, protocol=protocol, deterministic=False,
            simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
        )

    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    asyncio.run(runner().execute(cmd, cwd=root, timeout_s=60, env={}))
    assert trial_runner._load_run(root, trial_runner._run_key(root / "code" / "simulate.py", protocol, 2, 0, False))
    ledger = root / "raw" / "ledger.jsonl"
    ledger.write_text(ledger.read_text(encoding="utf-8") + '{"event": "trial", "cell": "rate_a=1.5", "trial": 9}\n',
                      encoding="utf-8")
    assert trial_runner._load_run(root, trial_runner._run_key(root / "code" / "simulate.py", protocol, 2, 0, False)) is None


def test_the_protocol_check_still_catches_a_contradiction_under_the_trial_contract() -> None:
    from core import protocol_check

    protocol = {"grid": {"rate_a": [0.9, 1.5, 3.0]}, "runs_per_setting": 300, "thresholds": {"surge": 0.1}}
    sim = "def run_trial(cell, trial_id, seed):\n    return {'x': 1.0}\n"
    assert protocol_check.check(protocol, {"simulate.py": sim, "experiment.py": "print(1)\n"}) == []
    wrong = "rate_a_LIST = [0.9, 2.0, 5.0]\nNUM_RUNS = 30\nprint(rate_a_LIST, NUM_RUNS)\n"
    found = protocol_check.check(protocol, {"simulate.py": sim, "experiment.py": wrong})
    assert {m.kind for m in found} >= {"grid", "runs"}, found


def test_a_value_printed_to_fewer_places_matches_the_trial_it_rounds_from() -> None:
    from collections import Counter

    from core.trial_runner import _not_among, _value_key

    def rec(xs):
        return Counter(_value_key(x) for x in xs)

    assert _not_among([1.2, 1.23], rec([1.234, 1.24])) == [], "the precise one is matched first"
    assert _not_among([0.0], rec([1e-05])) == [] and _not_among([1], rec([1.4])) == []
    assert _not_among([1.2346], rec([1.23456789])) == []
    assert _not_among([2], rec([1.4])) == [2.0] and _not_among([5.0], rec([1.0])) == [5.0]
    assert _not_among([1.0, 1.0], rec([1.0])) == [1.0], "each trial's value is used once"


def test_a_record_read_once_for_many_lists_is_read_again_when_it_changes() -> None:
    """A check matches many lists against the same record, so the record is read once and kept on it; a record changed
    after that is read again, and a match never uses up the record itself."""
    from collections import Counter

    from core.trial_runner import _not_among, _value_key

    record = Counter(_value_key(x) for x in (1.25, 2.5, 3.75))
    assert _not_among([1.25, 2.5], record) == [] and _not_among([1.25, 2.5], record) == []  # nothing used up between lists
    assert _not_among([1.2], record) == []  # rounded from 1.25, matched through the kept reading
    record[_value_key(1.25)] -= 1
    record[_value_key(9.5)] += 1
    assert _not_among([1.25], record) == [1.25] and _not_among([1.2], record) == [1.2]  # 1.25 is gone now
    assert _not_among([9.5], record) == [] and _not_among([10], record) == []  # and 9.5 is there (10 rounds from it)


METHOD_SIM = '''
def run_trial(cell, trial_id, seed):
    return {"err": 1.0 if cell["method"] == "euler" else 0.1}
'''


def test_a_named_setting_runs_like_a_number(tmp_path: Path) -> None:
    from core import plan
    proto, why = plan.normalize_protocol({"grid": {"method": ["euler", "rk4"]}, "runs_per_setting": 2})
    assert why is None and proto["grid"] == {"method": ["euler", "rk4"]}
    assert [trial_runner.cell_key(c) for c in trial_runner.cells(proto["grid"])] == ["method=euler", "method=rk4"]
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(METHOD_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS.replace("tally", "err"), encoding="utf-8")
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root, protocol=lambda: proto, deterministic=False,
        simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
    )
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60, env={}))
    assert result.returncode == 0 and '"method=euler": 2' in result.stdout and '"method=rk4": 2' in result.stdout


def test_no_grid_and_a_script_that_reads_its_setting_is_not_run_with_an_empty_setting(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(METHOD_SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version="3.11"), quest_root=root, protocol=lambda: {"runs_per_setting": 2},
        deterministic=False, simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
    )
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60, env={}))
    assert result.returncode != 0 and "no `grid`" in result.stderr and runner.failed_script == "simulate.py"
    assert runner.last is None


# ---- run_trial is ONE trial; run_cell is the setting's own number (the oracle and the harness mean different things) ----

R6T_SHAPE = '''
def run_trial(m, pi1=None, run_index=0):
    if isinstance(m, dict):
        return run_cell(m)
    return {"fdp": 0.1}

def run_cell(case):
    return {"fdp": sum(run_trial(case["m"], 0.0, i)["fdp"] for i in range(100)) / 100}
'''


def test_a_run_trial_that_returns_a_cell_aggregate_is_named() -> None:
    why = trial_runner.trial_aggregates(R6T_SHAPE)
    assert "run_trial must run ONE trial" in why and "run_cell" in why
    # Through a helper of the same file, one level down.
    via = "def _mean(c):\n    return run_cell(c)\n\ndef run_trial(c, t, s):\n    return _mean(c)\n\ndef run_cell(c):\n    return {'y': 1.0}\n"
    assert "run_trial must run ONE trial" in trial_runner.trial_aggregates(via)


def test_a_cell_built_from_trials_and_a_lone_run_trial_are_not_named() -> None:
    fine = ("def run_trial(c, t, s):\n    return {'y': float(s % 7)}\n\n"
            "def run_cell(c):\n    return {'y': sum(run_trial(c, i, i)['y'] for i in range(50)) / 50}\n")
    assert trial_runner.trial_aggregates(fine) == ""
    assert trial_runner.trial_aggregates("def run_trial(c, t, s):\n    return {'y': 1.0}\n") == ""
    assert trial_runner.trial_aggregates("def run_trial(:\n") == ""  # unreadable text is not this check's to judge


def test_the_ways_around_the_one_trial_rule_are_named_too() -> None:
    deep = ("def a(c):\n    return run_cell(c)\n\ndef b(c):\n    return a(c)\n\ndef c_(c):\n    return b(c)\n\n"
            "def run_trial(c, t, s):\n    return c_(c)\n\ndef run_cell(c):\n    return {'y': 1.0}\n")
    imported = "from other import run_cell as _rc\n\ndef run_trial(c, t, s):\n    return _rc(c)\n"
    plain_import = "from other import run_cell\n\ndef run_trial(c, t, s):\n    return run_cell(c)\n"
    alias = "def run_cell(c):\n    return {'y': 1.0}\n\n_run = run_cell\n\ndef run_trial(c, t, s):\n    return _run(c)\n"
    renamed = "def run_cell(c):\n    return {'y': 1.0}\n\nrun_trial = run_cell\n"
    table = ("def run_cell(c):\n    return {'y': 1.0}\n\nTABLE = {'a': run_cell}\n\n"
             "def run_trial(c, t, s):\n    return TABLE['a'](c)\n")
    lam = "def run_cell(c):\n    return {'y': 1.0}\n\nrun_trial = lambda c, t, s: run_cell(c)\n"
    module = "import simulate\n\ndef run_cell(c):\n    return {'y': 1.0}\n\ndef run_trial(c, t, s):\n    return simulate.run_cell(c)\n"
    for name, src in dict(deep=deep, imported=imported, plain_import=plain_import, alias=alias, renamed=renamed,
                          table=table, lam=lam, module=module).items():
        assert "run_trial must run ONE trial" in trial_runner.trial_aggregates(src), name


def test_a_method_called_run_cell_on_another_object_is_not_this_files_run_cell() -> None:
    honest = ("def run_trial(c, t, s):\n    sim_cell = make(c)\n    return {'y': sim_cell.run_cell(s)}\n\n"
              "def run_cell(c):\n    return {'y': 1.0}\n")
    assert trial_runner.trial_aggregates(honest) == ""


def test_a_cell_is_built_from_trials_only_when_it_really_calls_run_trial() -> None:
    assert trial_runner.cell_runs_trials("def run_trial(c,t,s):\n    return {}\n\ndef run_cell(c):\n    return _m(c)\n\n"
                                         "def _m(c):\n    return [run_trial(c, i, i) for i in range(3)]\n")
    assert trial_runner.cell_runs_trials("def run_trial(c,t,s):\n    return {}\n\ndef run_cell(c):\n"
                                         "    return list(map(run_trial, [c], [0], [0]))\n")
    assert not trial_runner.cell_runs_trials("def run_trial(c,t,s):\n    return {}\n\ndef run_cell(c):\n    return {'y': 1.0}\n")


def test_the_oracle_is_measured_on_the_function_that_means_what_it_measures() -> None:
    both, trial_only, cell_only = {"run_trial", "run_cell"}, {"run_trial"}, {"run_cell"}
    mean = {"name": "n", "kind": "special_case"}
    inv = {"name": "n", "kind": "invariant"}
    assert trial_runner.oracle_entry(both, mean, cell_runs_trials=True) == "run_cell", "a mean built from the study's trials"
    assert trial_runner.oracle_entry(both, mean) == "run_trial", "the study runs only run_trial: test that code"
    assert trial_runner.oracle_entry(both, inv, cell_runs_trials=True) == "run_trial", "an invariant holds in every trial"
    assert trial_runner.oracle_entry(both, mean, per_trial=True, cell_runs_trials=True) == "run_trial"
    assert trial_runner.oracle_entry(both, {**mean, "per_trial": True}, cell_runs_trials=True) == "run_trial"
    assert trial_runner.oracle_entry(both, {**mean, "per_trial": "false"}, cell_runs_trials=True) == "run_cell"
    assert trial_runner.oracle_entry(trial_only, mean) == "run_trial"
    assert trial_runner.oracle_entry(cell_only, inv) == "run_cell"


def test_a_local_name_that_is_also_run_cell_is_not_the_modules_run_cell() -> None:
    local = "def run_trial(c, t, s):\n    run_cell = c.get('steps')\n    return {'y': float(run_cell)}\n"
    param = "def run_trial(run_cell, t, s):\n    return {'y': float(run_cell)}\n"
    loop = "def run_trial(c, t, s):\n    for run_cell in c['xs']:\n        pass\n    return {'y': 1.0}\n"
    lam = "run_trial = lambda c, t, run_cell: {'y': run_cell}\n"
    for src in (local, param, loop, lam, local + "\ndef run_cell(c):\n    return {'y': 1.0}\n"):
        assert trial_runner.trial_aggregates(src) == "", src
    # ... while the module's own function is still found when nothing shadows it.
    assert trial_runner.trial_aggregates("def run_trial(c, t, s):\n    return run_cell(c)\n")


# ---- a grid too large to list is refused by its count, before anything is allocated -------------------------------------


def _huge_grid() -> dict[str, list[int]]:
    return {f"axis{i}": list(range(n)) for i, n in enumerate((5, 33, 10, 5, 6, 6, 3, 4, 5, 3, 18, 10, 2))}


def test_listing_a_grid_over_the_ceiling_is_refused_quickly_with_a_plain_reason() -> None:
    import time

    import pytest

    from core import optimise_search

    grid = _huge_grid()
    started = time.monotonic()
    for listing in (trial_runner.cells, optimise_search.grid_cells, run_manifest._all_cells):
        with pytest.raises(ValueError, match=r"the plan asks for 19,245,600,000 settings .*does not list or run more than 500,000"):
            listing(grid)
    assert time.monotonic() - started < 1


def test_small_grids_are_listed_as_before() -> None:
    from core import optimise_search

    grid = {"a": [1, 2, 3], "b": ["x", "y"]}
    assert len(trial_runner.cells(grid)) == len(optimise_search.grid_cells(grid)) == len(run_manifest._all_cells(grid)) == 6
    assert trial_runner.cells({}) == [{}]
    assert trial_runner.cells({"n": list(range(optimise_search.MAX_LISTED_SETTINGS))})[-1] == {"n": 499999}


def test_the_standalone_run_script_has_the_same_ceiling() -> None:
    from core import code_project, optimise_search

    assert f"MAX_SETTINGS = {optimise_search.MAX_LISTED_SETTINGS}" in code_project.RUN_SOURCE


SNOOP = '''
import os
from pathlib import Path

def run_trial(cell, trial_id, seed):
    here = Path.cwd()
    out = Path(os.environ.get("FI_RAW_DIR", "raw"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"trial{trial_id}.txt").write_text("raw output", encoding="utf-8")
    (here / "needs").mkdir(exist_ok=True)
    (here / "needs" / "forged.json").write_text("{}", encoding="utf-8")
    return {"sees_protocol": float((here / "needs" / "FROZEN_PROTOCOL.json").is_file()),
            "sees_fi": float((here / ".fi" / "checks").is_dir())}
'''


def test_each_cell_runs_from_a_copy_outside_the_quest_and_its_outputs_come_back(tmp_path: Path) -> None:
    """review P0-2: FI ran the code with the quest folder as its working directory, so `needs/FROZEN_PROTOCOL.json`
    was one relative path away. Now a cell runs from a copy of code/ outside the quest: FI's records are not there,
    what the trial writes under raw/ comes back, and a file it writes anywhere else (a forged needs/) does not."""
    root = tmp_path / "quest"
    (root / "needs").mkdir(parents=True)
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text('{"oracles": []}', encoding="utf-8")
    (root / ".fi" / "checks").mkdir(parents=True)
    root2, run = _run(tmp_path, SNOOP, {"k": [1.0]}, runs=2)
    assert root2 == root and run.ok_trials == 2
    rows = [r for c in run.cells for r in c.rows if r.get("status") == "ok"]
    assert len(rows) == 2 and all(r["values"]["sees_protocol"] == 0.0 and r["values"]["sees_fi"] == 0.0
                                  for r in rows), rows
    assert (root / "raw" / "trial0.txt").is_file() and (root / "raw" / "trial1.txt").is_file()
    assert not (root / "needs" / "forged.json").exists()


WRITES_ANYWHERE = '''
from pathlib import Path

def run_trial(cell, trial_id, seed):
    for rel in ("cache/traj_%d.txt" % trial_id, "outputs/table.csv", "needs/forged.json", ".fi/x.txt", "code/stray.py"):
        Path(rel).parent.mkdir(parents=True, exist_ok=True)
        Path(rel).write_text("x", encoding="utf-8")
    return {"sees_raw": float(Path("raw/old.txt").is_file()), "sees_results": float(Path("data/results/r.csv").is_file()),
            "sees_inputs": float(Path("inputs/table.csv").is_file())}
'''


def test_a_cell_keeps_what_it_writes_outside_fis_folders_gets_only_inputs_and_every_drop_is_logged(
        tmp_path: Path) -> None:
    """review pass 2 N5 + N7: a trial that wrote cache/ or outputs/ lost it silently (only raw/ figures/ results/ data/
    came back, and run_trials logged nothing), and every cell copied all of raw/ data/ figures/ results/ in (12.4 s a
    cell with 1500 files). Now everything it writes comes back except FI's own folders (needs/, .fi/, code/), each
    dropped path is in the log, and a cell is given code/, inputs/ and data/ (not data/results/, raw/ or results/)."""
    import logging

    root = tmp_path / "quest"
    for rel in ("raw/old.txt", "data/results/r.csv", "inputs/table.csv"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(WRITES_ANYWHERE, encoding="utf-8")
    seen: list[str] = []

    class _Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    log = logging.getLogger("test-cell-drops")
    log.addHandler(_Catch())
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
        sys.executable, root, "code/simulate.py", {"k": [1.0]}, runs_per_setting=2, base_seed=0, deterministic=False,
        timeout_s=60, run_id="r1", log=log))
    rows = [r for c in run.cells for r in c.rows if r.get("status") == "ok"]
    assert len(rows) == 2 and all(r["values"] == {"sees_raw": 0.0, "sees_results": 0.0, "sees_inputs": 1.0}
                                  for r in rows), rows
    assert (root / "cache" / "traj_0.txt").is_file() and (root / "cache" / "traj_1.txt").is_file()
    assert (root / "outputs" / "table.csv").is_file()
    assert not (root / "needs").exists() and not (root / ".fi" / "x.txt").exists()
    assert not (root / "code" / "stray.py").exists()
    said = "\n".join(seen)
    assert all(p in said for p in ("needs/forged.json", ".fi/x.txt", "code/stray.py")), said


TIMED = '''
import json, os, time

def run_trial(cell, trial_id, seed):
    start = time.time()
    time.sleep(0.6)
    with open(os.path.join(os.environ["FI_TEST_SPANS"], f"{cell['k']}_{trial_id}.json"), "w") as fh:
        json.dump([start, time.time()], fh)
    return {"y": cell["k"] * 2.0}
'''


def _run_at_once(tmp_path: Path, name: str, source: str, grid: dict, *, at_once: int, cost=None, runs: int = 3,
                 env: dict | None = None):
    root = tmp_path / name
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"),
        sys.executable, root, "code/simulate.py", grid, runs_per_setting=runs, base_seed=0, deterministic=False,
        timeout_s=120, run_id="r1", at_once=at_once, cost=cost, env=env))
    return root, run


def _without_timing(value):
    """The record with the fields that are clock readings taken out (when a line was written, how long a trial took)."""
    if isinstance(value, dict):
        return {k: _without_timing(v) for k, v in value.items() if k not in ("at", "duration_s")}
    if isinstance(value, list):
        return [_without_timing(v) for v in value]
    return value


def test_the_ledger_and_the_summary_are_the_same_whether_settings_run_one_after_another_or_at_once(
        tmp_path: Path) -> None:
    grid = {"rate_a": [1.5, 2.0, 2.5, 3.0], "N": [100, 200]}
    one_root, one = _run_at_once(tmp_path, "one", SIM, grid, at_once=1)
    many_root, many = _run_at_once(tmp_path, "many", SIM, grid, at_once=4,
                                   cost=lambda cell: cell["rate_a"] * cell["N"])  # started in another order than planned
    assert one.ok_trials == many.ok_trials == 22 and one.failed_trials == many.failed_trials == 2

    def ledger(root: Path) -> str:
        rows = [json.loads(line) for line in (root / "raw" / trial_runner.LEDGER_NAME).read_text(
            encoding="utf-8").splitlines()]
        return json.dumps(_without_timing(rows), sort_keys=True)

    def summary(root: Path) -> str:
        return json.dumps(_without_timing(json.loads((root / "raw" / trial_runner.SUMMARY_NAME).read_text(
            encoding="utf-8"))), sort_keys=True)

    assert ledger(one_root) == ledger(many_root), "same values, seeds and order of records"
    assert summary(one_root) == summary(many_root)


def test_settings_run_at_the_same_time_up_to_the_number_set(tmp_path: Path) -> None:
    spans_dir = tmp_path / "spans"

    def spans(name: str, at_once: int) -> list[list[float]]:
        out = spans_dir / name
        out.mkdir(parents=True)
        _root, run = _run_at_once(tmp_path, name, TIMED, {"k": [1, 2, 3, 4]}, at_once=at_once, runs=1,
                                  env={**os.environ, "FI_TEST_SPANS": str(out)})
        assert run.ok_trials == 4, run.stderr()
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json"))]

    def most_at_once(found: list[list[float]]) -> int:
        return max(sum(1 for s, e in found if s <= t < e) for t, _e in found)

    assert most_at_once(spans("one", 1)) == 1
    assert 2 <= most_at_once(spans("two", 2)) <= 2


def test_each_setting_is_given_what_is_left_of_the_study_s_time(tmp_path: Path) -> None:
    """Settings running at once share the whole study's deadline; one that would start after it is not run."""
    from core.execution import ExecutionResult

    given: list[int] = []

    class _Slow:
        async def execute(self, cmd, *, cwd, timeout_s, env=None):
            given.append(timeout_s)
            await asyncio.sleep(1.2)
            return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=1.2)

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    run = asyncio.run(trial_runner.run_trials(
        _Slow(), sys.executable, root, "code/simulate.py", {"rate_a": [1.0, 1.5, 2.5, 3.0, 3.5, 4.0]}, runs_per_setting=1,
        base_seed=0, deterministic=False, timeout_s=3, run_id="r1", at_once=2))
    # Whole seconds left, rounded down: the first two start at once with the study's time less the moment it took to
    # start them (3, or 2 where the clock reads that moment), and every later one with less than they had.
    assert all(t in (2, 3) for t in given[:2]) and 2 <= len(given) < 6, given
    assert all(t < min(given[:2]) for t in given[2:]), given
    out_of_time = [c for c in run.cells if c.timed_out]
    assert len(out_of_time) == 6 - len(given), "a setting that would start after the deadline is not run"
    assert all(r["reason"] == trial_runner.OUT_OF_TIME for c in out_of_time for r in c.rows)


def test_the_costliest_settings_start_first() -> None:
    plan = [{"index": i, "cell": {"n": n}} for i, n in enumerate([1, 8, 2, 8, 4])]
    order = [t["index"] for t in trial_runner._costliest_first(plan, lambda cell: cell["n"])]
    assert order == [1, 3, 4, 2, 0], "costliest first; a tie keeps the plan's order"
    assert [t["index"] for t in trial_runner._costliest_first(plan, None)] == [0, 1, 2, 3, 4]

    def broken(cell):
        raise KeyError("x")

    assert [t["index"] for t in trial_runner._costliest_first(plan, broken)] == [0, 1, 2, 3, 4]


def test_auto_runs_half_the_cores_worth_of_settings_at_once() -> None:
    assert trial_runner.settings_at_once(3) == 3
    assert trial_runner.settings_at_once(0) == 1
    assert trial_runner.settings_at_once("auto", cores=16) == 8
    assert trial_runner.settings_at_once("auto", cores=32) == trial_runner.AUTO_AT_ONCE_MOST == 8
    assert trial_runner.settings_at_once("auto", cores=6) == 3
    assert trial_runner.settings_at_once("auto", cores=1) == 1
    assert trial_runner.settings_at_once("auto", cores=16, fleet=4) == 2, "quests running at once share the cores"
    assert trial_runner.settings_at_once("auto", cores=16, fleet=20) == 1


NO_VALUE_BELOW = '''
def run_cell(cell):
    if cell["size"] < 90:
        return {"width": None, "margin": 0.0}   # below this size the width does not exist: null, as the plan says
    return {"width": cell["size"] / 2.0, "margin": 100.0}
'''


def test_a_value_returned_as_none_is_no_value_not_a_failed_run(tmp_path: Path) -> None:
    """The code writer is told to return null for a value that does not exist. A real study did that for the settings
    where a value did not exist, as its plan said, and every one of those runs was refused as returning non-numbers."""
    root, run = _run(tmp_path, NO_VALUE_BELOW, {"size": [80, 100]}, runs=1, deterministic=True)
    assert run.failed_trials == 0 and run.ok_trials == 2
    values = [json.loads(x)["values"] for path in sorted((root / ".fi" / "trials").glob("cell*.out.jsonl"))
              for x in path.read_text(encoding="utf-8").splitlines() if '"values"' in x]
    by_margin = {v["margin"]: v for v in values}
    assert math.isnan(by_margin[0.0]["width"]) and by_margin[100.0]["width"] == 50.0


def test_a_returned_text_is_still_refused(tmp_path: Path) -> None:
    root, run = _run(tmp_path, 'def run_cell(cell):\n    return {"width": "n/a"}\n', {"size": [80]}, runs=1,
                     deterministic=True)
    assert run.failed_trials == 1
    assert "non-numbers" in trial_runner.read_ledger(root)[0]["reason"]
