"""The frame of simulate.py (core/skeleton.py) fits every form of study, so turning ``execution.code_skeleton`` on breaks none of
them: a search for the best design, sub-studies, values compared inside each run, checks against a known answer, a trial
answered for all its seeds at once, a cluster job, a standalone run.py, and a resume. Made-up topics (a colony, a bridge)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import code_layout, code_project, function_steps, optimise_search, skeleton, small_run, trial_runner
from core.execution import SharedInterpreterExecutor

COLONY: dict[str, Any] = {"protocol": {
    "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
    "metrics": [{"id": "growth", "kind": "mean"}]}}

SEARCH: dict[str, Any] = {"protocol": {"optimisation": {
    "objective": {"quantity": "drag", "direction": "minimise"},
    "constraints": [{"quantity": "mass", "limit": "<= 5"}],
    "design_variables": [{"name": "width", "low": 1, "high": 4}],
    "fixed": {"speed": 3.0}, "numerical_settings": {"mesh": {"search": 8}}}}}


def _executor() -> SharedInterpreterExecutor:
    return SharedInterpreterExecutor(python_version="3.11")


def _quest(tmp_path: Path, source: str, name: str = "quest") -> Path:
    root = tmp_path / name
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (root / "code" / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME, encoding="utf-8")
    return root


def _rows(root: Path, cell: dict[str, Any], trials: list[dict[str, Any]], *, entry: str = "run_trial", kind: str = "scan") -> list[dict[str, Any]]:
    rows, _result = asyncio.run(trial_runner._run_batch(_executor(), sys.executable, root, "code/simulate.py", entry=entry, cell=cell,
                                                        trials=trials, timeout_s=120, env=None, thresholds=None, kind=kind))
    return rows


def _framed(design: dict[str, Any], body: str, *, stochastic: bool = True) -> str:
    text, why = skeleton.fill(skeleton.frame_source(design, stochastic=stochastic), body)
    assert text is not None, why
    return text


# ---- search for the best design ------------------------------------------------------------------------------------------

def test_a_search_gets_a_frame_whose_results_are_the_objective_and_the_constrained_quantities() -> None:
    assert skeleton.applies(SEARCH) == ""
    assert skeleton.declared_ids(SEARCH["protocol"]) == ["drag", "mass"]
    assert skeleton.settings_of(SEARCH["protocol"]) == ["speed", "mesh", "width"]
    text = skeleton.frame_source(SEARCH, stochastic=False)
    assert "SETTINGS = ('speed', 'mesh', 'width')" in text and "'drag'" in text and "'mass'" in text


def test_a_framed_search_evaluation_is_told_what_it_is_asked_for_and_returns_the_objective(tmp_path: Path) -> None:
    body = ("def trial(cell, ctx):\n    out = TrialResult()\n    out['drag'] = cell['width'] * cell['speed'] + (1000 if ctx.kind == 'search' else 0)\n"
            "    out['mass'] = cell['width'] + cell['mesh']\n    return out\n")
    root = _quest(tmp_path, _framed(SEARCH, body, stochastic=False))
    got, why = asyncio.run(trial_runner._run_one(_executor(), sys.executable, root, "code/simulate.py", entry="run_cell",
                                                 cell={"speed": 3.0, "mesh": 8, "width": 2.0}, seed=None, timeout_s=120, env=None,
                                                 thresholds=None, label="run_cell()", kind="search"))
    assert why == "" and got == {"drag": 1006.0, "mass": 10.0}
    scan, _ = asyncio.run(trial_runner._run_one(_executor(), sys.executable, root, "code/simulate.py", entry="run_cell",
                                                cell={"speed": 3.0, "mesh": 8, "width": 2.0}, seed=None, timeout_s=120, env=None,
                                                thresholds=None, label="run_cell()", kind="scan"))
    assert scan["drag"] == 6.0, "a row of the coarse scan is not a design the search tries"


def test_a_search_result_under_another_name_is_seen_at_the_first_evaluation(tmp_path: Path) -> None:
    text, _ = skeleton.fill(skeleton.frame_source(SEARCH, stochastic=False, strict=True),
                            "def trial(cell, ctx):\n    out = TrialResult()\n    out['drag_amount'] = 1.0\n    return out\n")
    rows = _rows(_quest(tmp_path, text or ""), {"width": 2.0}, [{"trial": 0, "seed": None}], entry="run_cell", kind="search")
    assert rows[0]["status"] == "failed" and "`drag_amount` is not a result the plan declares" in rows[0]["reason"] and "drag, drag_count" in rows[0]["reason"] and "mass_values" in rows[0]["reason"]


# ---- sub-studies -----------------------------------------------------------------------------------------------------------

SUBS = {"protocol": {"grid": {"a": [1, 2, 3], "b": [10, 20]}, "runs_per_setting": 2, "metrics": [{"id": "y"}],
                     "sub_studies": [{"name": "first", "vary": ["a"], "hold": {"b": 10}},
                                     {"name": "second", "vary": ["b"], "hold": {"a": 2}}]}}


def test_the_cell_of_a_part_holds_every_setting_and_reading_one_it_does_not_vary_is_not_an_unknown_read(tmp_path: Path) -> None:
    protocol = SUBS["protocol"]
    cells = trial_runner.cells(optimise_search.study_grid(protocol))
    assert len(cells) == 4 and all(set(c) == {"a", "b"} for c in cells), "a part's cell has the settings it does not vary at the value it holds"
    assert skeleton.settings_of(protocol) == ["a", "b"]
    root = _quest(tmp_path, _framed(SUBS, "def trial(cell, ctx):\n    out = TrialResult()\n    out['y'] = cell['a'] + cell['b']\n    return out\n"))
    seen = []
    for cell in cells:
        (row,) = _rows(root, cell, [{"trial": 0, "seed": 5}])
        assert row["status"] == "ok" and row["unknown_reads"] == [] and row["read"] == ["a", "b"]
        seen.append(row["values"]["y"])
    assert seen == [11.0, 12.0, 13.0, 22.0]


# ---- values compared inside each run ------------------------------------------------------------------------------------------

INSIDE = {"protocol": {"grid": {"rate": [0.1, 0.2]}, "in_each_run": {"method": ["fast", "slow"]}, "runs_per_setting": 2,
                       "metrics": [{"id": "growth"}, {"id": "speed"}]}}


def test_the_levels_compared_inside_each_run_are_in_the_cell_and_each_gives_a_result_per_level(tmp_path: Path) -> None:
    assert skeleton.settings_of(INSIDE["protocol"]) == ["rate", "method"]
    body = ("def trial(cell, ctx):\n    out = TrialResult()\n    levels = cell['method']\n"
            "    out.each('growth', {m: cell['rate'] * (1 + i) for i, m in enumerate(levels)})\n"
            "    out.each('speed', [cell['rate'] * (3 + i) for i, m in enumerate(levels)])\n    return out\n")
    root = _quest(tmp_path, _framed(INSIDE, body))
    got = asyncio.run(small_run.run(_executor(), sys.executable, root, "code/simulate.py", {"rate": [0.1, 0.2]}, deterministic=False,
                                    timeout_s=120, in_each_run={"method": ["fast", "slow"]}))
    assert not got.crashed, got.says()
    first = got.settings[0].values
    assert first["growth"][0] == {"method=fast": 0.1, "method=slow": 0.2} and first["speed"][0]["method=slow"] == pytest.approx(0.4)


def test_each_refuses_a_result_the_plan_does_not_declare(tmp_path: Path) -> None:
    text, _ = skeleton.fill(skeleton.frame_source(INSIDE, stochastic=True, strict=True),
                            "def trial(cell, ctx):\n    out = TrialResult()\n    out.each('mine', [1, 2])\n    return out\n")
    (row,) = _rows(_quest(tmp_path, text or ""), {"rate": 0.1, "method": ["fast", "slow"]}, [{"trial": 0, "seed": 1}])
    assert row["status"] == "failed" and "`mine` is not a result the plan declares" in row["reason"]


# ---- checks against a known answer --------------------------------------------------------------------------------------------

ORACLES = {"protocol": {**COLONY["protocol"], "oracles": [{"name": "zero_dose"}]}}


def test_a_check_with_no_case_is_answered_by_an_oracle_function_beside_the_trial(tmp_path: Path) -> None:
    assert skeleton.applies(ORACLES) == "" and skeleton.case_less_checks(ORACLES["protocol"]) == ["zero_dose"]
    frame = skeleton.frame_source(ORACLES, stochastic=True)
    body = ("def trial(cell, ctx):\n    out = TrialResult()\n    out['growth'] = cell['rate']\n    return out\n\n"
            "def oracle():\n    return {'zero_dose': 0.0}\n")
    text, note = skeleton.compose(frame, body)
    assert skeleton.frame_intact(frame, text) == "" and "def oracle():" in text and note == ""
    root = _quest(tmp_path, text)
    values, why = asyncio.run(trial_runner.run_oracle(_executor(), sys.executable, root, "code/simulate.py", timeout_s=120, env=None,
                                                      thresholds=None))
    assert why == "" and values == {"zero_dose": 0.0}


# ---- a trial answered for all its seeds at once -----------------------------------------------------------------------------------

BATCH = ("CALLS = []\n\ndef trial_batch(cell, ctx):\n    CALLS.append(len(ctx.seeds))\n    out = ctx.results()\n"
         "    for record, rng in zip(out, ctx.rngs()):\n        record['growth'] = cell['rate'] + rng.random() * 0 + len(CALLS)\n"
         "        record['growth_seeds'] = len(ctx.seeds)\n    return out\n")
TRIALS = [{"trial": 0, "seed": 11}, {"trial": 1, "seed": 22}, {"trial": 2, "seed": 33}]


def test_a_trial_batch_is_called_once_for_the_setting_with_every_seed_and_each_trial_gets_its_own_result(tmp_path: Path) -> None:
    text = _framed(COLONY, BATCH)
    assert skeleton.has_frame(text) and "from fi_frame import batch_of_trials" in text and "def trial_batch(cell, ctx)" in text
    rows = _rows(_quest(tmp_path, text), {"rate": 0.1, "dose": 1}, TRIALS)
    assert [r["status"] for r in rows] == ["ok"] * 3, rows
    assert [r["values"]["growth"] for r in rows] == [1.1, 1.1, 1.1], "the batch ran once for the three trials"
    assert [r["trial"] for r in rows] == [0, 1, 2] and all(r["values"]["growth_seeds"] == 3 for r in rows)


@pytest.mark.parametrize("reply, said", [
    ("def trial_batch(cell, ctx):\n    return [{'growth': 1.0}]\n", "one result for each of the 3 seed(s)"),
    ("def trial_batch(cell, ctx):\n    out = ctx.results()\n    for r in out:\n        r['growth'] = 1.0\n    return out[::-1]\n",
     "keep the results in the order of ctx.seeds"),
    ("def trial_batch(cell, ctx):\n    return 5\n", "returned int"),
])
def test_a_batch_that_does_not_give_one_result_per_seed_in_order_fails_the_setting_with_a_plain_reason(tmp_path: Path, reply: str, said: str) -> None:
    rows = _rows(_quest(tmp_path, _framed(COLONY, reply)), {"rate": 0.1, "dose": 1}, TRIALS)
    assert [r["status"] for r in rows] == ["failed"] * 3 and said in rows[0]["reason"], rows[0]["reason"]


def test_a_batch_function_with_other_parameters_is_refused_before_it_runs() -> None:
    text, why = skeleton.fill(skeleton.frame_source(COLONY, stochastic=True), "def trial_batch(cell):\n    return []\n")
    assert text is None and "`trial_batch` must take exactly (cell, ctx)" in why


def test_a_cells_batch_is_given_the_one_setting_and_must_give_one_result_for_it(tmp_path: Path) -> None:
    good = ("def cells_batch(cells, ctx):\n    out = []\n    for cell in cells:\n        r = TrialResult()\n        r['growth'] = cell['rate'] * 2\n"
            "        out.append(r)\n    return out\n")
    (row,) = _rows(_quest(tmp_path, _framed(COLONY, good), "a"), {"rate": 0.1, "dose": 1}, [{"trial": 0, "seed": 1}])
    assert row["status"] == "ok" and row["values"]["growth"] == 0.2
    (bad,) = _rows(_quest(tmp_path, _framed(COLONY, "def cells_batch(cells, ctx):\n    return []\n"), "b"), {"rate": 0.1, "dose": 1},
                   [{"trial": 0, "seed": 1}])
    assert bad["status"] == "failed" and "one result for each cell it is given (1)" in bad["reason"]


def test_a_reply_with_its_own_trial_and_a_batch_function_uses_its_trial() -> None:
    text = _framed(COLONY, "def trial_batch(cell, ctx):\n    return []\n\ndef trial(cell, ctx):\n    return TrialResult()\n")
    assert "FI: this study gives" not in text


# ---- a cluster job, the run key, run.py ---------------------------------------------------------------------------------------------

def _engine(tmp_path: Path, **execution: Any) -> Any:
    from tests.test_disclosure import _engine as make

    eng = make(tmp_path)
    eng.config.execution.code_skeleton = True
    for key, value in execution.items():
        setattr(eng.config.execution, key, value)
    return eng


def test_a_cluster_job_gets_the_frame_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path, background_jobs=True)
    state = {"design": COLONY}
    assert "ONLY the function `trial(cell, ctx)`" in eng._frame_note(state) and skeleton.has_frame(eng._framed(state, "def trial(cell, ctx):\n    return {}\n"))


def test_the_frame_note_says_what_each_form_needs(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    note = eng._frame_note({"design": SEARCH})
    assert "searches for the best design" in note and "`drag`" in note and "`mass`" in note and "ctx.kind" in note
    inside = eng._frame_note({"design": INSIDE})
    assert "LIST of their values" in inside and "out.each" in inside
    assert "`oracle()`" in eng._frame_note({"design": ORACLES}) and "`zero_dose`" in eng._frame_note({"design": ORACLES})
    assert "trial_batch(cell, ctx)" in eng._frame_note({"design": COLONY})
    assert "sub-studies" in eng._frame_note({"design": SUBS}) and "sub-studies" not in eng._frame_note({"design": COLONY})


def test_the_frame_runtime_is_part_of_the_simulation_the_run_key_covers(tmp_path: Path) -> None:
    root = _quest(tmp_path, "def run_trial(cell, trial_id, seed):\n    return {}\n")
    sim = root / "code" / "simulate.py"
    assert [p.name for p in code_layout.simulation_inputs(root, root / "code")] == ["fi_frame.py", "simulate.py"]
    one = trial_runner._run_key(sim, {"a": [1]}, 1, 0, False)
    (root / "code" / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME + "\n# changed\n", encoding="utf-8")
    assert trial_runner._run_key(sim, {"a": [1]}, 1, 0, False) != one


def test_run_py_of_a_framed_simulation_carries_the_frame_and_runs_standalone(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    (root / "code").mkdir(parents=True)
    code = root / "code"
    (code / "simulate.py").write_text(_framed(COLONY, BATCH), encoding="utf-8")
    plain = code_project.run_source(tmp_path / "nowhere")
    assert plain == code_project.RUN_SOURCE, "a simulation without the frame gets run.py exactly as before"
    (code / "run.py").write_text(code_project.run_source(code), encoding="utf-8")
    assert not (code / skeleton.RUNTIME_FILE).exists(), "a copy of code/ that lost fi_frame.py"
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"entry": "run_trial", "grid": {"rate": [0.1], "dose": [1]}, "runs_per_setting": 3, "base_seed": 0}),
                     encoding="utf-8")
    out = tmp_path / "rows.json"
    done = subprocess.run([sys.executable, str(code / "run.py"), "--cell", "0", str(out), str(study)], capture_output=True, text=True, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-600:]
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [r["ok"] for r in rows] == [True] * 3 and {r["values"]["growth_seeds"] for r in rows} == {3.0}, rows
    assert (code / skeleton.RUNTIME_FILE).read_text(encoding="utf-8") == skeleton.RUNTIME


# ---- a resume keeps the frame, a change of the plan writes it again -----------------------------------------------------------------

def _older(text: str) -> str:
    return text.replace(f"version {skeleton.VERSION}", "version 1").replace(f"FRAME_VERSION = {skeleton.VERSION}", "FRAME_VERSION = 1")


def test_a_frame_an_older_version_wrote_for_the_same_plan_is_kept_and_a_changed_plan_gets_a_new_one() -> None:
    old = _older(skeleton.fill(skeleton.frame_source(COLONY, stochastic=True), BATCH)[0] or "")
    assert skeleton.frame_version(old) == 1
    kept = skeleton.frame_for(COLONY, stochastic=True, existing=old)
    assert skeleton.frame_version(kept) == 1 and skeleton.frame_intact(kept, old) == "", "the frame in the file, nothing regenerated"
    fresh = skeleton.frame_for(COLONY, stochastic=True, existing="")
    assert skeleton.frame_version(fresh) == skeleton.VERSION
    changed = {"protocol": {**COLONY["protocol"], "metrics": [{"id": "growth"}, {"id": "decay"}]}}
    new = skeleton.frame_for(changed, stochastic=True, existing=old)
    assert skeleton.frame_version(new) == skeleton.VERSION and "'decay'" in new, "a result was added: a new frame"
    same = skeleton.frame_source(COLONY, stochastic=True)
    assert skeleton.frame_for(COLONY, stochastic=True, existing=same) == same


def test_an_old_frame_still_checks_its_repairs_and_keeps_its_runtime(tmp_path: Path) -> None:
    old = _older(skeleton.fill(skeleton.frame_source(COLONY, stochastic=True), BATCH)[0] or "")
    assert skeleton.has_frame(old)
    assert skeleton.repair_problem(old, old.replace("RESULTS = (", "RESULTS = ('mine', "), "simulate.py").startswith("the frame FI wrote was changed")
    code = tmp_path / "code"
    code.mkdir()
    (code / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME.replace(f"FRAME_RUNTIME_VERSION = {skeleton.VERSION}", "FRAME_RUNTIME_VERSION = 1"),
                                              encoding="utf-8")
    assert skeleton.ensure_runtime(code, old) is False, "the runtime an older frame was written with stays"
    assert skeleton.ensure_runtime(code, skeleton.frame_source(COLONY, stochastic=True)) is True
    assert (code / skeleton.RUNTIME_FILE).read_text(encoding="utf-8") == skeleton.RUNTIME
    assert skeleton.ensure_runtime(code, "") is False, "already FI's runtime"


def test_a_reply_that_keeps_the_old_frame_after_the_plan_changed_goes_into_the_new_frame() -> None:
    old = skeleton.fill(skeleton.frame_source(COLONY, stochastic=True), BATCH)[0] or ""
    changed = {"protocol": {**COLONY["protocol"], "metrics": [{"id": "growth"}, {"id": "decay"}]}}
    new_frame = skeleton.frame_source(changed, stochastic=True)
    text, note = skeleton.compose(new_frame, old, previous=old)
    assert skeleton.frame_intact(new_frame, text) == "" and "'decay'" in text and "def trial_batch" in text and "plan changed" in note
    plain, problem = skeleton.compose(new_frame, old)
    assert plain == old and problem.startswith("the frame FI wrote was changed"), "without the previous frame it is still a changed frame"
    tampered = old.replace("seed=seed", "seed=1")
    again, bad = skeleton.compose(new_frame, tampered, previous=old)
    assert again == tampered and bad.startswith("the frame FI wrote was changed"), "a change of the model's is refused whatever the plan did"


def test_the_code_step_puts_a_reply_into_the_frame_already_in_the_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    state = {"design": COLONY}
    body = "def trial(cell, ctx):\n    out = TrialResult()\n    out['growth'] = 1.0\n    return out\n"
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    older = _older(skeleton.fill(skeleton.frame_source(COLONY, stochastic=True), body)[0] or "")
    (code / "simulate.py").write_text(older, encoding="utf-8")
    (code / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME.replace(f"FRAME_RUNTIME_VERSION = {skeleton.VERSION}", "FRAME_RUNTIME_VERSION = 1"),
                                              encoding="utf-8")
    text = eng._framed(state, body)
    assert skeleton.frame_version(text) == 1, "the quest's frame, written by the older version, is the one the reply goes into"
    assert "FRAME_RUNTIME_VERSION = 1" in (code / skeleton.RUNTIME_FILE).read_text(encoding="utf-8")


# ---- the number the package reads: the plan's, written by FI --------------------------------------------------------------------------

PACKAGE = ('"""A beam."""\nimport math\n\n\ndef deflection(load):\n    return load * LENGTH / YOUNG\n\n\n'
           'def tip(load):\n    return deflection(load) * math.sin(angle_rad)\n\n\ndef margin(x):\n    return x - SAFETY\n')
OUTLINE = {"constants": [{"name": "LENGTH", "value": "2.5", "source": "design"}, {"name": "YOUNG", "value": "1e3", "source": "design"},
                         {"name": "GRID", "value": "1,2,...,9", "source": "design"}]}
BEAM_PROTOCOL = {"constants": {"angle_rad": 0.5, "SAFETY": 0.25, "unused": 7}}


def test_the_numbers_the_plan_fixes_that_the_package_reads_are_written_into_it_by_fi() -> None:
    values = function_steps.plan_constants(OUTLINE, BEAM_PROTOCOL)
    assert values == {"LENGTH": 2.5, "YOUNG": 1000.0, "angle_rad": 0.5, "SAFETY": 0.25, "unused": 7}, "a range is not a constant"
    new, added = function_steps.with_constants(PACKAGE, values)
    assert sorted(added) == sorted(["LENGTH", "YOUNG", "angle_rad", "SAFETY"])
    ns: dict[str, Any] = {}
    exec(compile(new, "model.py", "exec"), ns)
    assert ns["deflection"](10.0) == pytest.approx(0.025) and ns["margin"](1.0) == 0.75
    assert "UNUSED" not in new and "unused" not in new, "only what the package reads"
    again, names = function_steps.with_constants(new, values)
    assert again == new and sorted(names) == sorted(added), "the block is made again from the plan: the same plan, the same text"
    broken: dict[str, Any] = {}
    exec(compile(PACKAGE, "model.py", "exec"), broken)
    with pytest.raises(NameError):
        broken["deflection"](10.0)


def test_a_name_the_plan_does_not_fix_stays_missing_and_a_name_the_module_binds_is_not_touched() -> None:
    module = "import math\nLENGTH = 9.0\n\n\ndef f(x):\n    return x * LENGTH * OTHER\n"
    new, added = function_steps.with_constants(module, {"LENGTH": 2.5})
    assert added == [] and new == module, "LENGTH is the module's own; OTHER is not the plan's, the code check names it"
    spelt = function_steps.with_constants("def f():\n    return G\n", {"g": 9.81})
    assert spelt == ("def f():\n    return G\n", []), "a plan's `g` is not guessed to be the module's `G`"


def test_the_filler_writes_the_plan_numbers_before_it_checks_a_function_and_needs_no_repair(tmp_path: Path) -> None:
    from tests.test_function_steps import OUTLINE as OUT, PROTOCOL as PROTO, _Model, _filler, _specs

    reply = "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m) * UNIT\n"
    model = _Model({"frequency": [reply], "quality": ["# E2\ndef quality(m, w, c):\n    return m * w / c\n"],
                    "decay": ["import math\n\n\n# E3\ndef decay(x0, g, t):\n    return x0 * math.exp(-g * t)\n"]})
    filler = _filler(tmp_path, model, values={"UNIT": 1.0})
    outcome = asyncio.run(filler.run_all(_specs()))
    assert outcome.status == "done" and model.count["frequency"] == 1, "no repair: UNIT was there when the function was checked"
    assert "UNIT = 1.0" in (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert OUT and PROTO
    without = _filler(tmp_path / "w", _Model({"frequency": [reply], "quality": ["# E2\ndef quality(m, w, c):\n    return m * w / c\n"],
                                              "decay": ["import math\n\n\n# E3\ndef decay(x0, g, t):\n    return x0 * math.exp(-g * t)\n"]}))
    assert asyncio.run(without.run_all(_specs())).status != "done", "without the numbers the same function fails its check"


@pytest.mark.asyncio
async def test_the_engine_writes_the_outlines_numbers_into_the_package_and_into_a_package_kept_from_before(tmp_path: Path) -> None:
    from tests.test_equation_gate import GOOD, OUTLINE as GATE_OUTLINE, STATE, _engine as gate_engine, _plan_md, _Writer

    engine = gate_engine(tmp_path)
    _plan_md(engine)
    outline = {**GATE_OUTLINE, "constants": [{"name": "UNIT", "value": "1.0", "source": "design"}]}
    scaled = "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m) * UNIT\n"
    writer = _Writer({"frequency": [scaled], "quality": [GOOD["quality"]]})
    engine._client = writer
    assert await engine._fill_model_functions({**STATE, "implement_outline": outline}, outline) is True
    assert writer.seen == {"frequency": 1, "quality": 1}, "UNIT was in the module when frequency was checked: no repair"
    model_path = engine.quest_root / "code" / "spring" / "model.py"
    assert "UNIT = 1.0" in model_path.read_text(encoding="utf-8")
    # A package kept from an earlier pass that lacks the line gets it.
    model_path.write_text(model_path.read_text(encoding="utf-8").replace("UNIT = 1.0\n", ""), encoding="utf-8")
    assert "UNIT = 1.0" not in model_path.read_text(encoding="utf-8")
    engine._write_plan_constants({**STATE, "implement_outline": outline}, "spring")
    assert "UNIT = 1.0" in model_path.read_text(encoding="utf-8")


# ---- a number the plan changes, a number it drops, every module of the package ----------------------------------------------------------

def test_a_number_the_plan_changed_is_changed_in_fis_block_and_one_the_plan_dropped_leaves_it() -> None:
    module = "def f(x):\n    return x * LENGTH + SAFETY\n"
    first, names = function_steps.with_constants(module, {"LENGTH": 6.0, "SAFETY": 1})
    assert "LENGTH = 6.0" in first and sorted(names) == ["LENGTH", "SAFETY"]
    second, _ = function_steps.with_constants(first, {"LENGTH": 7.0, "SAFETY": 1})
    assert "LENGTH = 7.0" in second and "6.0" not in second and second.count("LENGTH =") == 1
    third, names3 = function_steps.with_constants(second, {"LENGTH": 7.0})
    assert "SAFETY =" not in third and names3 == ["LENGTH"], "a number the plan no longer fixes is not kept"
    fourth, none = function_steps.with_constants(third, {})
    assert none == [] and "FI: the numbers" not in fourth and "LENGTH =" not in fourth


def test_a_name_the_model_binds_itself_after_fi_wrote_it_is_the_models_and_leaves_fis_block() -> None:
    first, _ = function_steps.with_constants("def f():\n    return LENGTH\n", {"LENGTH": 6.0})
    now = first + "\nLENGTH = 3.0\n"
    new, names = function_steps.with_constants(now, {"LENGTH": 7.0})
    assert names == [] and "LENGTH = 3.0" in new and "7.0" not in new and "6.0" not in new


def test_the_plan_numbers_reach_every_module_of_the_package_and_a_rewrite_changes_a_changed_number(tmp_path: Path) -> None:
    package = tmp_path / "beam"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model.py").write_text("def f():\n    return LENGTH\n", encoding="utf-8")
    (package / "loads.py").write_text("def g(x):\n    return x * YOUNG\n", encoding="utf-8")
    written = function_steps.write_package_constants(package, {"LENGTH": 2.5, "YOUNG": 1000.0})
    assert sorted(written) == ["LENGTH", "YOUNG"]
    assert "LENGTH = 2.5" in (package / "model.py").read_text(encoding="utf-8")
    assert "YOUNG = 1000.0" in (package / "loads.py").read_text(encoding="utf-8")
    assert function_steps.write_package_constants(package, {"LENGTH": 2.5, "YOUNG": 1000.0}) == [], "nothing to change"
    function_steps.write_package_constants(package, {"LENGTH": 3.0, "YOUNG": 1000.0})
    assert "LENGTH = 3.0" in (package / "model.py").read_text(encoding="utf-8")


# ---- with the frame off nothing moves ----------------------------------------------------------------------------------------------

def test_the_trial_harness_and_the_unframed_run_py_are_what_they_were_before_the_frame() -> None:
    assert "FI_TRIAL_SEEDS" not in trial_runner.HARNESS_SOURCE and "fi_frame" not in trial_runner.HARNESS_SOURCE
    assert code_project.run_source(None) == code_project.RUN_SOURCE
    assert "fi_frame" not in code_project.RUN_SOURCE


def test_an_unframed_simulation_gets_the_plain_run_py(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("def run_cell(cell):\n    return {'a': 1}\n", encoding="utf-8")
    assert code_project.run_source(code) == code_project.RUN_SOURCE
    (code / "simulate.py").write_text(_framed(COLONY, "def trial(cell, ctx):\n    return TrialResult()\n"), encoding="utf-8")
    assert code_project.run_source(code) != code_project.RUN_SOURCE


def test_a_framed_run_py_whose_edit_no_longer_applies_is_a_fault_not_a_silent_plain_run_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text(_framed(COLONY, "def trial(cell, ctx):\n    return TrialResult()\n"), encoding="utf-8")
    monkeypatch.setattr(code_project, "_FRAMED_EDITS", (("text no run.py has", "other"),))
    with pytest.raises(RuntimeError, match="no longer applies"):
        code_project.run_source(code)


_VALUES = "def run_trial(cell, trial_id, seed):\n    return {'growth': cell['rate'] * cell['dose'] + seed * 0.001}\n"
_FRAMED_VALUES = ("def trial(cell, ctx):\n    out = TrialResult()\n    out['growth'] = cell['rate'] * cell['dose'] + ctx.seed * 0.001\n"
                  "    return out\n")


def test_a_plain_grid_gives_the_same_rows_through_the_framed_simulation_as_through_the_plain_one(tmp_path: Path) -> None:
    cell, trials = {"rate": 0.2, "dose": 2}, [{"trial": t, "seed": 100 + t} for t in range(3)]
    plain = _rows(_quest(tmp_path, _VALUES, "plain"), cell, trials)
    framed = _rows(_quest(tmp_path, _framed(COLONY, _FRAMED_VALUES), "framed"), cell, trials)
    drop = {"duration_s", "at_s", "nonce", "warnings"}

    def core(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k not in drop} for r in rows]

    assert [r["status"] for r in plain] == ["ok"] * 3
    assert core(framed) == core(plain), "the frame changes how the study is written, not a trial's row"


# ---- a batch: its cost per trial, and its order by seed ---------------------------------------------------------------------------

def test_the_estimate_of_a_trial_batch_study_divides_the_batch_time_over_its_seeds(tmp_path: Path) -> None:
    from core import run_estimate

    body = ("import time\n\ndef trial_batch(cell, ctx):\n    out = ctx.results()\n    for r in out:\n        time.sleep(0.05)\n"
            "        r['growth'] = 1.0\n    return out\n")
    root = _quest(tmp_path, _framed(COLONY, body))
    probe, why = asyncio.run(run_estimate._time_one(_executor(), sys.executable, root, root / "code" / "simulate.py",
                                                     {"rate": 0.1, "dose": 1}, runs=8, deterministic=False, timeout_s=120,
                                                     env=None, thresholds=None))
    assert why == "" and probe is not None
    assert probe.per_trial_s is not None and probe.per_trial_s >= 0.04, "the batch's time is shared by its 8 seeds"
    assert probe.at(800, 8).seconds >= 800 * 0.04, "a full run of 800 trials is not estimated as the 8 timed"


def test_a_batch_of_results_the_model_built_itself_is_refused_because_their_order_cannot_be_told(tmp_path: Path) -> None:
    body = "def trial_batch(cell, ctx):\n    return [TrialResult() for _ in ctx.seeds]\n"
    rows = _rows(_quest(tmp_path, _framed(COLONY, body)), {"rate": 0.1, "dose": 1}, TRIALS)
    assert [r["status"] for r in rows] == ["failed"] * 3 and "ctx.results()" in rows[0]["reason"], rows[0]["reason"]
