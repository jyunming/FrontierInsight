"""A setting of the grid the simulation never reads is found before the run (core/unread_settings.py::unread_axes).

The real case (rounds s1 and r13, first-order kinetics): five temperatures sat in `protocol.grid` beside the noise level and
the number of time points, but run_trial builds its own five temperatures and never reads `temperature_K`: every (noise,
points) setting ran five times over, 125 settings instead of 25, and the paper said there was no replication.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import unread_settings
from core import plan as _plan
from core import run_estimate

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"


def _simulation(quest: str) -> dict[str, str]:
    code = FIX / quest / "code"
    return {"simulate.py": (code / "simulate.py").read_text(encoding="utf-8"),
            "kinetics_parameter_recovery/model.py": (code / "kinetics_parameter_recovery" / "model.py").read_text(encoding="utf-8")}


def _protocol(quest: str) -> dict:
    design = _plan.parse((FIX / quest / "plan.md").read_text(encoding="utf-8")).design
    return design["protocol"]


@pytest.mark.parametrize("quest, read", [("s1_kinetics", {"sigma", "n_points"}),
                                         ("r13_kinetics", {"noise_sigma_over_c0", "n_time"})])
def test_the_kinetics_simulations_never_read_temperature(quest, read):
    keys, why = unread_settings.setting_reads(_simulation(quest))
    assert why == "" and read <= keys and "temperature_K" not in keys
    protocol = _protocol(quest)
    assert run_estimate.cell_count(protocol["grid"]) == 125
    unread, why = unread_settings.unread_axes(_simulation(quest), protocol)
    assert unread == ["temperature_K"], why  # t_max_over_k (s1) has one value: it repeats nothing
    kept = {k: v for k, v in protocol["grid"].items() if k not in unread}
    assert run_estimate.cell_count(kept) == 25


def _quest(tmp_path: Path, quest: str):
    import shutil

    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / quest / "plan.md", _plan.plan_path(eng.quest_root))
    shutil.copytree(FIX / quest / "code", eng.quest_root / "code")
    return eng


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_replay_the_unread_setting_leaves_the_grid_before_the_run(tmp_path: Path, quest: str) -> None:
    eng = _quest(tmp_path, quest)
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0]}
    assert run_estimate.cell_count(eng._draft_protocol(state)["grid"]) == 125
    eng._drop_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    grid = eng._draft_protocol(state)["grid"]
    assert "temperature_K" not in grid and run_estimate.cell_count(grid) == 25
    assert "temperature_K" not in state["design"]["protocol"]["grid"]
    text = _plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "the simulation never reads it" in text and "runs 25 setting(s), not 125" in text
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "never reads it" in log and "`temperature_K` (5 values" in log
    note = eng._run_checks_note()
    assert "runs 25 setting(s), not 125" in note and "Do not describe `temperature_K` as a setting" in note
    # Done once: a second pass finds nothing more to take out and writes nothing.
    versions = len(_plan.history(eng.quest_root))
    eng._drop_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    assert len(_plan.history(eng.quest_root)) == versions


def test_a_frozen_protocol_or_a_redesign_is_left_alone(tmp_path: Path) -> None:
    eng = _quest(tmp_path, "s1_kinetics")
    state = {"topic": "t", "iteration": 1, "design": _plan.load_design(eng.quest_root)[0]}
    eng._drop_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    assert "temperature_K" in _plan.load_design(eng.quest_root)[0]["protocol"]["grid"]


def _one(source: str) -> tuple[set[str] | None, str]:
    return unread_settings.setting_reads({"simulate.py": source})


def test_reads_followed_through_copies_and_calls_the_code_defines():
    keys, why = _one(
        "def helper(s, k):\n    return s['b'] + s.get('c', 1)\n"
        "class M:\n    def go(self, setting):\n        return setting['d']\n"
        "def run_trial(cell, trial_id, seed):\n"
        "    c = dict(cell)\n    d = c.copy()\n    if 'e' in d:\n        pass\n"
        "    return {'x': cell['a'] + helper(d, 2) + M().go(cell)}\n")
    assert why == "" and keys == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize("body", [
    "return {'x': sum(cell.values())}",              # every value at once
    "return dict(**cell)",                           # spread
    "k = 'a'\n    return {'x': cell[k]}",            # a name FI cannot see
    "import json\n    return {'x': len(json.dumps(cell))}",  # handed to code FI cannot read
    "for k in cell:\n        pass\n    return {}",    # a loop over it
    "return {'x': other(cell)}",                     # a function these files do not define
])
def test_any_use_that_could_read_any_name_means_nothing_is_known(body):
    keys, why = _one(f"def run_trial(cell, trial_id, seed):\n    {body}\n")
    assert keys is None and why
    unread, why2 = unread_settings.unread_axes({"simulate.py": f"def run_trial(cell, trial_id, seed):\n    {body}\n"},
                                         {"grid": {"a": [1, 2], "b": [3, 4]}})
    assert unread == [] and why2


@pytest.mark.parametrize("body", [
    "from scipy.integrate import solve_ivp\n    return {'x': solve_ivp(lambda t, y: -cell['sigma'] * y, (0, 1), [1.0])}",
    "f = lambda: cell['sigma']\n    return {'x': f()}",
    "def rhs(t, y):\n        return -cell['sigma'] * y\n    return {'x': rhs(0, 1)}",
    "def inner():\n        def deeper():\n            return cell.get('sigma')\n        return deeper()\n    return {'x': inner()}",
    "def inner(c):\n        return c['sigma']\n    return {'x': inner(cell)}",
])
def test_a_setting_read_inside_a_lambda_or_a_nested_function_is_read(body):
    source = f"def run_trial(cell, trial_id, seed):\n    a = cell['a']\n    {body}\n"
    keys, why = _one(source)
    assert why == "" and {"a", "sigma"} <= keys
    assert unread_settings.unread_axes({"simulate.py": source}, {"grid": {"a": [1, 2], "sigma": [0.1, 0.2]}}) == ([], "")


@pytest.mark.parametrize("source", [
    # A library function with the same name as one of these files: the call reaches the library, not the file.
    "from somelib import helper\ndef helper(s):\n    return s['a']\ndef run_trial(cell, t, s):\n    return helper(cell)\n",
    # A method on an object FI cannot place.
    "def go(s):\n    return s['a']\ndef run_trial(cell, t, s):\n    obj = make()\n    return obj.go(cell)\n",
    # A name the function binds to some function itself.
    "def helper(s):\n    return s['a']\ndef run_trial(cell, t, s):\n    helper = pick()\n    return helper(cell)\n",
    # Kept for other code through a global name.
    "HELD = None\ndef run_trial(cell, t, s):\n    global HELD\n    HELD = cell\n    return {'x': cell['a']}\n",
    # Stored on an object or used as a default value.
    "def run_trial(cell, t, s):\n    t.cell = cell\n    return {'x': cell['a']}\n",
    "def run_trial(cell, t, s):\n    def f(c=cell):\n        return c\n    return {'x': cell['a'], 'y': f()}\n",
])
def test_a_call_or_a_use_FI_cannot_place_means_nothing_is_known(source):
    keys, why = _one(source)
    assert keys is None and why


def test_a_function_of_the_package_called_through_its_module_is_followed():
    files = {"simulate.py": "from pkg import model\ndef run_trial(cell, t, s):\n    return model.solve(cell, 3)\n",
             "pkg/__init__.py": "",
             "pkg/model.py": "def solve(setting, n):\n    return setting['k'] * n\n"}
    assert unread_settings.setting_reads(files) == ({"k"}, "")
    assert unread_settings.unread_axes(files, {"grid": {"k": [1, 2], "T": [3, 4]}}) == (["T"], "")


def test_a_grid_whose_swept_settings_are_all_unread_is_left_alone():
    source = "def run_trial(cell, trial_id, seed):\n    return {'x': cell['fixed']}\n"
    unread, why = unread_settings.unread_axes({"simulate.py": source}, {"grid": {"a": [1, 2], "b": [3, 4]}})
    assert unread == [] and why


def test_no_entry_point_and_one_value_settings_are_left_alone():
    assert unread_settings.unread_axes({"simulate.py": "def f(cell):\n    return cell['a']\n"}, {"grid": {"b": [1, 2]}})[0] == []
    source = "def run_trial(cell, trial_id, seed):\n    return {'x': cell['a']}\n"
    assert unread_settings.unread_axes({"simulate.py": source}, {"grid": {"a": [1, 2], "b": [7]}}) == ([], "")
    assert unread_settings.unread_axes({"simulate.py": source}, {"grid": {"a": [1, 2], "b": [7, 8]}}) == (["b"], "")
