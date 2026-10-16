"""A setting of the grid FI's reading of the code finds no use of (core/unread_settings.py::unread_axes) is said, never
taken out: FI keeps every setting and runs it.

The real case (rounds s1 and r13, first-order kinetics): five temperatures sat in `protocol.grid` beside the noise level and
the number of time points, but run_trial builds its own five temperatures and never reads `temperature_K`: every (noise,
points) setting ran five times over, 125 settings instead of 25. An earlier version took such a setting out of the grid;
a second review found ordinary code (an import under another name, a wrapped helper, ...) that the reading misses, so a
setting the code does read was taken out. A reading of code can be wrong, and a setting taken out is a study not run: the
grid is now always kept, and only a setting the study compares sends the code back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import unread_settings
from core import plan as _plan
from core import disclosure, paper_facts, run_estimate

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


def _quest(tmp_path: Path, quest: str, sim_patch=None):
    import shutil

    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / quest / "plan.md", _plan.plan_path(eng.quest_root))
    shutil.copytree(FIX / quest / "code", eng.quest_root / "code")
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / quest / "topic_requirements.json", eng.fi_dir / "topic_requirements.json")
    if sim_patch:
        path = eng.quest_root / "code" / "simulate.py"
        path.write_text(sim_patch(path.read_text(encoding="utf-8")), encoding="utf-8")
    return eng


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_replay_the_unread_setting_is_said_and_every_setting_is_kept(tmp_path: Path, quest: str) -> None:
    eng = _quest(tmp_path, quest)
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0]}
    design_before = json.dumps(state["design"], sort_keys=True)
    assert eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py") is None
    grid = eng._draft_protocol(state)["grid"]
    assert "temperature_K" in grid and run_estimate.cell_count(grid) == 125
    assert json.dumps(state["design"], sort_keys=True) == design_before
    assert run_estimate.cell_count(_plan.load_design(eng.quest_root)[0]["protocol"]["grid"]) == 125
    line = "FI's reading of the code finds no use of `temperature_K`; every setting is kept and run."
    assert line in _plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert line in (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    # Nothing the paper must state, no writer's note, no run-record line, no record file.
    assert not [i for i in paper_facts.required_items(eng.quest_root) if i.kind == paper_facts.NOT_DONE]
    assert "temperature_K" not in eng._run_checks_note()
    assert "not 125" not in disclosure.run_record(eng.quest_root)
    assert not (eng.fi_dir / "unread_settings.json").exists()
    # Said once: a second pass writes nothing more.
    versions = len(_plan.history(eng.quest_root))
    eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    assert len(_plan.history(eng.quest_root)) == versions
    assert _plan.plan_path(eng.quest_root).read_text(encoding="utf-8").count(line) == 1


def test_a_frozen_protocol_or_a_redesign_is_left_alone(tmp_path: Path) -> None:
    eng = _quest(tmp_path, "s1_kinetics")
    before = _plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    state = {"topic": "t", "iteration": 1, "design": _plan.load_design(eng.quest_root)[0]}
    eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    assert _plan.plan_path(eng.quest_root).read_text(encoding="utf-8") == before


def test_the_reviews_alias_case_keeps_the_setting_the_code_reads(tmp_path: Path) -> None:
    """The second review's case: simulate.py imports `scale_full as scale`, `scale_full` reads `temperature_K`, and the
    package also defines a `scale` that does not. The reading follows the wrong `scale`; before, FI took `temperature_K`
    out of the grid and the run failed on it. Now the grid is unchanged."""
    def patch(s: str) -> str:
        s = s.replace("from kinetics_parameter_recovery import model",
                      "from kinetics_parameter_recovery import model\n"
                      "from kinetics_parameter_recovery.model import scale_full as scale")
        return s.replace('sigma = float(cell["sigma"])', 'sigma = float(cell["sigma"]) * scale(cell)')

    eng = _quest(tmp_path, "s1_kinetics", patch)
    model = eng.quest_root / "code" / "kinetics_parameter_recovery" / "model.py"
    model.write_text(model.read_text(encoding="utf-8") + "\n\ndef scale(c):\n    return 1.0\n\n\n"
                     "def scale_full(c):\n    return float(c['temperature_K']) / 300.0\n", encoding="utf-8")
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0]}
    assert eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py") is None
    for grid in (eng._draft_protocol(state)["grid"], _plan.load_design(eng.quest_root)[0]["protocol"]["grid"],
                 state["design"]["protocol"]["grid"]):
        assert "temperature_K" in grid and run_estimate.cell_count(grid) == 125
    assert "nothing in the code reads it" not in _plan.plan_path(eng.quest_root).read_text(encoding="utf-8")


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


# ---- the review's probes: a setting the code should read is the code's fault, never taken out -------------------------

def test_a_misspelt_read_of_a_swept_setting_is_left_to_the_code_check():
    """grid {rate, n}; the code reads `rate_hz` with a fallback: the code check sends it back. Nothing is named unread,
    so `rate` is never taken out (before: `rate` was removed and the misspelling was no longer caught)."""
    from core import grid_keys

    protocol = {"grid": {"rate": [10.0, 20.0, 40.0], "n": [100, 1000]}, "runs_per_setting": 20}
    for sim in ("def run_trial(cell, trial_id, seed):\n    return {'x': cell['n'] * cell.get('rate_hz', 80.0)}\n",
                "def run_trial(cell, trial_id, seed):\n    return {'x': cell['n'] * cell['rate_hz']}\n"):
        files = {"simulate.py": sim}
        assert grid_keys.problems(files, protocol)
        unread, why = unread_settings.unread_axes(files, protocol)
        assert unread == [] and "rate_hz" in why


def test_s1_with_a_misspelt_noise_level_keeps_every_setting(tmp_path: Path) -> None:
    eng = _quest(tmp_path, "s1_kinetics",
                 lambda s: s.replace('sigma = float(cell["sigma"])', 'sigma = float(cell.get("sigma_noise", 0.02))'))
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0]}
    assert eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py") is None
    grid = eng._draft_protocol(state)["grid"]
    assert "sigma" in grid and "temperature_K" in grid and run_estimate.cell_count(grid) == 125
    assert "the code reads a name the plan does not give" in (eng.fi_dir / "run.log").read_text(encoding="utf-8")


def test_s1_with_the_noise_level_ignored_sends_the_code_back(tmp_path: Path) -> None:
    """The simulation uses a constant noise: `sigma`, which the topic compares, is unread. The code goes back to be
    repaired, naming it; the grid stays as the plan wrote it."""
    eng = _quest(tmp_path, "s1_kinetics", lambda s: s.replace('sigma = float(cell["sigma"])', 'sigma = 0.02'))
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0], "exec_reflect_iter": 0}
    back = eng._check_unread_settings(state, eng.quest_root / "code" / "simulate.py")
    assert back is not None and back["exec_result"]["returncode"] == 1
    assert "`sigma`" in back["exec_result"]["stderr_tail"] and "the topic compares noise level" in back["exec_result"]["stderr_tail"]
    grid = eng._draft_protocol(state)["grid"]
    assert "sigma" in grid and "temperature_K" in grid
    # The repairs used up: the grid is kept.
    spent = {**state, "exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations}
    assert eng._check_unread_settings(spent, eng.quest_root / "code" / "simulate.py") is None
    assert "sigma" in eng._draft_protocol(spent)["grid"]


def test_an_ignored_method_the_topic_compares_is_not_taken_out(tmp_path: Path) -> None:
    """A setting the plan's checks run at, or the topic compares by name, is part of the study: never taken out."""
    eng = _quest(tmp_path, "s1_kinetics")
    state = {"topic": "t", "iteration": 0, "design": _plan.load_design(eng.quest_root)[0]}
    protocol = eng._draft_protocol(state)
    req = {"factors": [{"name": "temperature", "compared": True, "levels": [], "range": []}], "comparisons": []}
    assert eng._settings_the_study_compares({**state, "design": state["design"]}, protocol, ["temperature_K"]) == {}
    import json

    (eng.fi_dir / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    found = eng._settings_the_study_compares(state, protocol, ["temperature_K", "sigma"])
    assert found["temperature_K"] == "the topic compares temperature"
    assert "check" in found["sigma"]  # s1's checks run at values of sigma


def test_a_not_done_item_the_plan_no_longer_supports_is_taken_away_before_the_run(tmp_path: Path) -> None:
    """A plan changed after the plan step (an approved change, a person's edit) that now has a usable `in_each_run`: the
    entry recorded when it could not be used is taken away, so the paper is not made to state it."""
    eng = _quest(tmp_path, "s1_kinetics")
    eng._note_not_done([{"name": "comparison inside each run", "sentence": "x has one value", "kind": "in_each_run"},
                        {"name": "old", "sentence": "an entry of a kind no longer written", "kind": "unread_setting"}])
    design = _plan.load_design(eng.quest_root)[0]
    state = {"topic": "t", "iteration": 0, "design": design}
    eng._refresh_not_done(state)  # the plan still lacks it: kept; the unknown kind goes
    assert [e["kind"] for e in eng._topic_covered_read()["not_done"]] == ["in_each_run"]
    # A person's edit of plan.md (the protocol's source before the freeze) gives a usable in_each_run.
    from core import in_run

    path = _plan.plan_path(eng.quest_root)
    path.write_text(_plan.edit_design_block(path.read_text(encoding="utf-8"), lambda d: {
        **d, "protocol": {**d["protocol"], "in_each_run": {"wall": ["a", "b"]}}}), encoding="utf-8")
    assert in_run.of(eng._draft_protocol(state))
    eng._refresh_not_done(state)
    assert eng._topic_covered_read()["not_done"] == []
    assert not [i for i in paper_facts.required_items(eng.quest_root) if i.kind == paper_facts.NOT_DONE]


def test_the_check_runs_after_the_checks_of_the_code_and_before_the_timing():
    source = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    gate = source.index("stop, oracle_code, gate_repaired = await self._gate_pass(state, py, exec_env, seed_path)")
    drop = source.index("self._check_unread_settings(state, simulate_path)")
    size = source.index("too_long = await self._size_the_run(state, runner, py, exec_env)")
    assert gate < drop < size
