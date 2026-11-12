"""What core/code_reach.py makes of FI's record of calls (core/call_record.py): a function the record shows called is reached;
one that nothing names is unreached; one that only a setting the small run skipped would reach is "cannot tell". The record
says only "this function ran", never "this function did not".
"""

from __future__ import annotations

from pathlib import Path

from core import call_record, code_reach
from tests.test_call_record_of_runs import _check_run, _quest


# ---- what code_reach makes of it -----------------------------------------------------------------------------------------
def _verdict(root: Path, function: str, file: str = "simulate.py", record="load") -> str:
    scripts, harness = code_reach.run_entries(root)
    seen = call_record.load(root) if record == "load" else record
    return code_reach.verdict(root / "code", file, function, scripts=scripts, harness=harness, record=seen)


def test_a_nested_inner_function_that_ran_is_reached_and_one_that_never_runs_is_not(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _check_run(root)
    assert _verdict(root, "tally.square") == code_reach.RAN
    assert _verdict(root, "tally.<locals>.square") == code_reach.RAN
    assert _verdict(root, "square") == code_reach.RAN
    # named nowhere, and the function it sits in runs: it is a dead function
    assert _verdict(root, "tally.never_used") == code_reach.UNREACHED
    never_run = code_reach.never_run_in(root)
    assert never_run("simulate.py", "tally.never_used") and not never_run("simulate.py", "tally.square")


def test_a_function_only_used_where_the_small_run_did_not_go_is_cannot_tell(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _check_run(root, size=3)  # the branch for a large size is not taken
    assert ("simulate.py", "big_only") not in call_record.load(root)
    assert _verdict(root, "big_only") == code_reach.CANNOT_TELL
    assert not code_reach.never_run_in(root)("simulate.py", "big_only")


def test_a_function_called_by_a_name_chosen_at_run_time_is_reached_by_the_record_not_by_reading(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert _verdict(root, "lookup_step", record=None) == code_reach.UNREACHED  # nothing writes its name as a call
    _check_run(root)
    assert _verdict(root, "lookup_step") == code_reach.RAN
    assert not code_reach.never_run_in(root)("simulate.py", "lookup_step")


def test_without_a_record_nothing_is_called_ran_and_a_function_never_named_is_still_unreached(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert call_record.load(root) is None
    assert _verdict(root, "Box.grow") == code_reach.UNREACHED
    assert _verdict(root, "tally.square") == code_reach.CANNOT_TELL  # named in the body of a function that runs
    assert _verdict(root, "no_such.thing") == code_reach.CANNOT_TELL


def test_a_parameter_called_table_does_not_make_the_user_s_code_unreached(tmp_path: Path) -> None:
    """A computed ``getattr`` on a parameter of the study's own code is a name FI cannot read, whatever the parameter is called."""
    files = {"simulate.py": "from helpers import pick\n\ndef run_trial(cell, trial_id, seed):\n    return {'a': pick(cell, 'size')}\n",
             "helpers.py": "def pick(table, name):\n    return getattr(table, name)\n\ndef elsewhere():\n    return 2\n"}
    root = _quest(tmp_path, files)
    assert code_reach.reached(root / "code", scripts=["experiment.py"], harness="simulate.py") is None
    assert not code_reach.never_run_in(root)("helpers.py", "elsewhere")
