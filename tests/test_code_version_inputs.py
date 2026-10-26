"""The keys that decide whether cached trials or a search can be reused cover every file the simulation may read.

Before this change, :func:`core.outside_run.code_version` and the two ``_run_key`` functions hashed ``simulate.py``, the
model package and the files below ``code/`` that the search hashed; a helper module beside ``simulate.py`` (outside the
package), a data or parameter file in ``code/`` read by the trial runner, or a file in the quest's ``inputs/`` changed
nothing. A stale cached trial set was then reused and its results reached the paper.
"""

from __future__ import annotations

from pathlib import Path

from core import code_layout, optimise, outside_run
from core import trial_runner as tr

SIMULATE = "import helper_gain\n\ndef run_cell(cell):\n    return {'settle_time': helper_gain.scale(cell['gain'])}\n"
HELPER = "def scale(value):\n    return 2 * value\n"
ANALYSIS = "print('analysis')\n"


def _quest(base: Path) -> Path:
    root = base / "quest"
    (root / "code" / "tests").mkdir(parents=True)
    (root / "inputs").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (root / "code" / "helper_gain.py").write_text(HELPER, encoding="utf-8")
    (root / "code" / "params.json").write_text('{"gain": 3}\n', encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    (root / "code" / "tests" / "test_gain.py").write_text("def test_gain():\n    pass\n", encoding="utf-8")
    (root / "inputs" / "table.csv").write_text("gain,hit_rate\n1,0.5\n", encoding="utf-8")
    return root


def _run_key(root: Path) -> str:
    return tr._run_key(root / "code" / "simulate.py", {"grid": {"gain": [1, 2]}}, 2, 0, False)


def _block_key(root: Path) -> str:
    block = {"objective": "hit_rate", "parameters": {"gain": [1, 2]}}
    return optimise._run_key(root, root / "code" / "simulate.py", block, 0, "run_cell", None)


def test_a_helper_module_beside_simulate_changes_the_code_version(tmp_path: Path) -> None:
    """Old code: a top-level module beside simulate.py is not in the package, so changing it left the version
    unchanged and a stale cache was reused. Known positive: the same change now moves the version."""
    root = _quest(tmp_path)
    before = outside_run.code_version(root)
    (root / "code" / "helper_gain.py").write_text("def scale(value):\n    return 5 * value\n", encoding="utf-8")
    assert outside_run.code_version(root) != before


def test_a_parameter_file_in_code_changes_the_code_version_and_both_keys(tmp_path: Path) -> None:
    """Old code: a .json parameter file below code/ was read by the simulation but not hashed by code_version or by
    trial_runner._run_key, so changing it left every key the same."""
    root = _quest(tmp_path)
    version, trial_key, block_key = outside_run.code_version(root), _run_key(root), _block_key(root)
    (root / "code" / "params.json").write_text('{"gain": 4}\n', encoding="utf-8")
    assert outside_run.code_version(root) != version
    assert _run_key(root) != trial_key
    assert _block_key(root) != block_key


def test_a_file_in_inputs_changes_the_code_version_and_both_keys(tmp_path: Path) -> None:
    """Old code: nothing under the quest's inputs/ was hashed, so a changed table left the keys the same."""
    root = _quest(tmp_path)
    version, trial_key, block_key = outside_run.code_version(root), _run_key(root), _block_key(root)
    (root / "inputs" / "table.csv").write_text("gain,hit_rate\n1,0.9\n", encoding="utf-8")
    assert outside_run.code_version(root) != version
    assert _run_key(root) != trial_key
    assert _block_key(root) != block_key


def test_the_analysis_script_does_not_change_the_code_version(tmp_path: Path) -> None:
    """Known positive of the existing exclusion: the analysis script reads what the simulation wrote and changes nothing
    the simulation computes, so repairing it keeps the cached values (see the docstring of code_version)."""
    root = _quest(tmp_path)
    version, trial_key = outside_run.code_version(root), _run_key(root)
    (root / "code" / "experiment.py").write_text("print('a different analysis')\n", encoding="utf-8")
    assert outside_run.code_version(root) == version
    assert _run_key(root) == trial_key


def test_a_test_file_does_not_change_the_code_version(tmp_path: Path) -> None:
    """The tests below code/tests/ are not part of the simulation."""
    root = _quest(tmp_path)
    version = outside_run.code_version(root)
    (root / "code" / "tests" / "test_gain.py").write_text("def test_gain():\n    assert True\n", encoding="utf-8")
    assert outside_run.code_version(root) == version


def test_what_a_run_writes_under_code_does_not_change_the_key(tmp_path: Path) -> None:
    """A run writes its caches and results into folders of its own; a key that covered them would change with every run
    and never be reused. Known positive for the exclusion of generated folders."""
    root = _quest(tmp_path)
    version, trial_key = outside_run.code_version(root), _run_key(root)
    (root / "code" / "results").mkdir()
    (root / "code" / "results" / "values.json").write_text('{"settle_time": 6}\n', encoding="utf-8")
    assert outside_run.code_version(root) == version
    assert _run_key(root) == trial_key


def test_simulation_inputs_lists_the_files_in_sorted_order(tmp_path: Path) -> None:
    """The helper's own listing: the simulation, its helper, its data file and the person's input file are listed; the
    analysis script, the tests and the notes are not; the order is stable (by path)."""
    root = _quest(tmp_path)
    (root / "code" / "notes.md").write_text("notes\n", encoding="utf-8")
    names = [p.relative_to(root).as_posix() for p in code_layout.simulation_inputs(root)]
    assert names == sorted(names)
    assert names == ["code/helper_gain.py", "code/params.json", "code/simulate.py", "inputs/table.csv"]


def test_a_file_of_any_suffix_under_code_changes_the_code_version(tmp_path: Path) -> None:
    """Old code (the suffix whitelist of the first version of this change): a pickled lookup and a file with no suffix,
    both read by the simulation, were not listed, so changing them left the version unchanged and a stale cache was
    reused. Known positive: the listing takes every file under code/ except the excluded ones, whatever its suffix."""
    root = _quest(tmp_path)
    (root / "code" / "lookup.pkl").write_bytes(b"\x80\x04old")
    (root / "code" / "constants").write_text("gain 3\n", encoding="utf-8")
    version = outside_run.code_version(root)
    (root / "code" / "lookup.pkl").write_bytes(b"\x80\x04new")
    assert outside_run.code_version(root) != version
    version = outside_run.code_version(root)
    (root / "code" / "constants").write_text("gain 4\n", encoding="utf-8")
    assert outside_run.code_version(root) != version
