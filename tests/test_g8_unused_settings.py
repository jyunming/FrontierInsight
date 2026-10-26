"""A setting the study compares that changes nothing is known from FI's own record of the run (core/flat_output.py).

Neutral toy topics only (a cooling cup, two estimators of a rate).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core import flat_output as flat
from core import trial_runner
from core.execution import SharedInterpreterExecutor
from tests.test_flat_output import SWEEP_ANALYSIS, _quest


def _cell(cell: dict[str, Any], *values: dict[str, Any]) -> Any:
    return SimpleNamespace(cell=cell, rows=[{"status": "ok", "values": v} for v in values])


def _runner(root: Path, grid: dict[str, list[Any]], **kw: Any) -> trial_runner.TrialsRunner:
    return trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=root,
        protocol={"grid": grid}, deterministic=True, simulate=root / "code" / "simulate.py",
        analysis=root / "code" / "experiment.py", **kw)


# --- only the plan's declared results are compared ---------------------------------------------------------------------

def _timed_cells() -> list[Any]:
    # `lid` reaches no result; `wall_s` (a counter the simulation also returns) moves with every cell.
    return [_cell({"flow": f, "lid": lid}, {"t_out": 20.0 - f, "wall_s": 0.1 * i + 0.01 * j})
            for i, f in enumerate((1.0, 2.0)) for j, lid in enumerate(("open", "shut"))]


def test_a_number_the_plan_did_not_declare_cannot_hide_a_setting_that_changed_every_declared_result_not_at_all() -> None:
    """Old code compared every returned number, so a counter that differs per cell (not a timing FI can re-run and find
    unsteady) made `lid` look like it changed something and nothing was named."""
    cells = _timed_cells()
    assert flat.unchanged_axes(cells) == [], "compared with the counter, the lid looks like it moved something"
    assert flat.unchanged_axes(cells, declared=["t_out"]) == [("lid", 2)]


def test_a_declared_result_that_moves_with_the_setting_is_still_a_change() -> None:
    cells = [_cell({"flow": f, "lid": lid}, {"t_out": 20.0 - f - (0.5 if lid == "shut" else 0.0), "wall_s": 1.0 * f})
             for f in (1.0, 2.0) for lid in ("open", "shut")]
    assert flat.unchanged_axes(cells, declared=["t_out"]) == []


def test_when_the_run_returned_none_of_the_declared_names_every_number_is_compared() -> None:
    """FI cannot tell which numbers are the results then; judging nothing would hide the setting."""
    cells = [_cell({"flow": f, "lid": lid}, {"t_out": 20.0 - f}) for f in (1.0, 2.0) for lid in ("open", "shut")]
    assert flat.unchanged_axes(cells, declared=["other_name"]) == [("lid", 2)]
    assert flat.declared_only(["other_name"], ["t_out"]) == set()
    assert flat.declared_only(["t_out"], ["t_out", "wall_s"]) == {"wall_s"}


# --- two results that agree trial by trial -----------------------------------------------------------------------------

def test_two_estimators_that_agree_in_every_trial_are_named_as_identical() -> None:
    cells = [_cell({"n": n}, {"est_plain": 0.5 * n, "est_weighted": 0.5 * n, "other": 3.0 * n * n}) for n in (1.0, 2.0, 3.0, 4.0)]
    got = flat.identical_results(cells)
    assert got == [("est_plain", "est_weighted", 4)]
    note = flat.identical_results_note(got)
    assert "`est_plain` and `est_weighted`" in note and "by design" in note and "``" not in note


def test_results_that_differ_in_any_trial_or_never_vary_are_not_paired() -> None:
    differ = [_cell({"n": n}, {"a": 0.5 * n, "b": 0.5 * n + (0.1 if n == 3.0 else 0.0)}) for n in (1.0, 2.0, 3.0, 4.0)]
    assert flat.identical_results(differ) == []
    constant = [_cell({"n": n}, {"a": 2.0, "b": 2.0}) for n in (1.0, 2.0, 3.0, 4.0)]
    assert flat.identical_results(constant) == [], "a constant is the flat-sweep check's, not a pair"
    few = [_cell({"n": n}, {"a": n, "b": n}) for n in (1.0, 2.0)]
    assert flat.identical_results(few) == [], "two agreeing trials prove nothing"
    cells = [_cell({"n": n}, {"a": n, "b": n, "wall_s": n}) for n in (1.0, 2.0, 3.0)]
    assert flat.identical_results(cells, declared=["a", "b"]) == [("a", "b", 3)]
    assert flat.identical_results(cells, declared=["a", "wall_s"], ignore={"wall_s"}) == []


# --- the run -----------------------------------------------------------------------------------------------------------

def test_a_study_whose_compared_setting_reaches_no_result_is_sent_back_once_with_the_setting_named(tmp_path: Path) -> None:
    """Old code only wrote a note when one setting changed nothing while others did; the simulation went on to the paper."""
    root = _quest(tmp_path, 'def run_cell(cell):\n    return {"t_out": 20.0 - cell["flow"]}\n')
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    grid = {"flow": [1.0, 2.0, 3.0], "lid": ["open", "shut"]}
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    first = _runner(root, grid)
    sent = asyncio.run(first.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert sent.returncode == 1 and first.failed_script == "simulate.py"
    assert "changing lid changed none of the results" in sent.stderr and "keep every setting" in sent.stderr
    assert (root / trial_runner.UNCHANGED_SENT).is_file()
    second = _runner(root, grid)
    done = asyncio.run(second.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert done.returncode == 0 and second.no_difference and "`lid`" in second.no_difference


def test_a_study_whose_every_setting_matters_is_not_sent_back_and_identical_estimators_are_recorded(tmp_path: Path) -> None:
    root = _quest(tmp_path, 'def run_cell(cell):\n    return {"est_plain": cell["n"] * 0.5, "est_weighted": cell["n"] * 0.5}\n')
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    runner = _runner(root, {"n": [1.0, 2.0, 3.0, 4.0]})
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    result = asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0
    assert runner.no_difference and "`est_plain` and `est_weighted`" in runner.no_difference
    assert not (root / trial_runner.UNCHANGED_SENT).exists()


def test_a_setting_the_code_reads_that_truly_changes_nothing_is_an_observation_not_a_repair_request(tmp_path: Path) -> None:
    """The simulation reads `lid` but the effect is nil in this range. Old code (and the first version of the send-back)
    asked for a repair, pressing the model to invent an effect; now the run goes on with the code unchanged and the fact
    is recorded as an observed absence of difference."""
    sim = ('def run_cell(cell):\n    gain = 0.0 if cell["lid"] == "shut" else 0.0\n'
           '    return {"t_out": 20.0 - cell["flow"] + gain}\n')
    root = _quest(tmp_path, sim)
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    runner = _runner(root, {"flow": [1.0, 2.0, 3.0], "lid": ["open", "shut"]})
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    result = asyncio.run(runner.execute(cmd, cwd=root, timeout_s=60, env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0 and runner.flat is None and runner.failed_script is None
    assert runner.no_difference and "`lid`" in runner.no_difference
    assert not (root / trial_runner.UNCHANGED_SENT).exists()
