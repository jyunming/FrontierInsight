"""FI writes the frame of simulate.py from the plan (core/skeleton.py): the entry point the trial harness calls, the seed, the
thresholds and the list of declared results; the model writes only ``trial(cell, ctx)``. Made-up topic: a colony's growth."""

from __future__ import annotations

import asyncio
import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from core import function_steps, skeleton, small_run
from core.execution import SharedInterpreterExecutor

DESIGN: dict[str, Any] = {"protocol": {
    "grid": {"rate": [0.1, 0.2, 0.3], "dose": [1, 2, 3]}, "runs_per_setting": 5,
    "metrics": [{"id": "growth", "kind": "mean"}, {"id": "survival", "kind": "proportion"}]}}

BODY = '''
import math

def _noise(ctx):
    return ctx.rng.random() * 0.01

def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = cell["rate"] * cell["dose"] + _noise(ctx)
    out["survival_count"] = 1
    out["survival_total"] = 1
    out["extra_curve"] = math.exp(-cell["rate"])
    return out
'''


def _frame(**kw: Any) -> str:
    return skeleton.frame_source(DESIGN, stochastic=True, **kw)


def test_the_frame_names_the_declared_results_and_the_entry_point() -> None:
    text = _frame()
    ast.parse(text)
    assert "def run_trial(cell, trial_id, seed)" in text and "def run_cell" not in text
    for name in ("growth", "growth_count", "survival", "survival_total", "survival_values"):
        assert f"'{name}'" in text
    assert "def run_cell(cell)" in skeleton.frame_source(DESIGN, stochastic=False)
    assert skeleton.declared_ids(DESIGN["protocol"]) == ["growth", "survival"]


def test_the_models_trial_is_put_into_the_frame_with_its_helpers() -> None:
    frame = _frame()
    text, why = skeleton.fill(frame, BODY)
    assert why == "" and text is not None
    assert "def _noise(ctx)" in text and "import math" in text and "raise NotImplementedError" not in text
    assert skeleton.frame_intact(frame, text) == ""
    assert not function_steps.is_stub(text, "trial")


def test_a_trial_with_other_parameters_is_refused() -> None:
    text, why = skeleton.fill(_frame(), "def trial(cell, seed):\n    return {}\n")
    assert text is None and "(cell, ctx)" in why


def test_a_repair_cannot_change_the_frame() -> None:
    frame = _frame()
    ok, _ = skeleton.fill(frame, BODY)
    assert ok is not None
    assert skeleton.frame_intact(frame, ok.replace("\n\n", "\n\n\n")) == "", "spacing is not a change"
    assert "changed" in skeleton.frame_intact(frame, ok.replace("RESULTS = (", "RESULTS = ('mine', "))
    assert "changed" in skeleton.frame_intact(frame, ok.replace("seed=seed", "seed=1"))
    assert "missing" in skeleton.frame_intact(frame, "def trial(cell, ctx):\n    return {}\n")
    assert "parameters" in skeleton.frame_intact(frame, ok.replace("def trial(cell, ctx)", "def trial(cell, ctx, extra=1)"))


def test_a_reply_that_is_a_whole_simulation_stays_as_written() -> None:
    whole = "def run_trial(cell, trial_id, seed):\n    return {'growth': 1.0}\n"
    text, note = skeleton.compose(_frame(), whole)
    assert text == whole and "without the frame" in note
    assert skeleton.compose(_frame(), "def helper():\n    pass\n")[1] == "the reply holds no function `trial`"
    framed, _ = skeleton.fill(_frame(), BODY)
    assert skeleton.compose(_frame(), framed or "") == (framed, "")


@pytest.mark.parametrize("change, why", [
    ({}, ""),
    ({"protocol": {"grid": {"a": [1, 2]}}}, "the plan declares no result"),
    ({"protocol": {**DESIGN["protocol"], "optimisation": {"design_variables": []}}}, "the plan searches for a best design"),
    ({"protocol": {**DESIGN["protocol"], "in_each_run": {"t": [1, 2]}}}, ""),
    ({"protocol": {**DESIGN["protocol"], "oracles": [{"name": "known"}]}}, "the check `known` names no case"),
])
def test_where_the_frame_applies(change: dict[str, Any], why: str) -> None:
    assert skeleton.applies({**DESIGN, **change}) == why


def test_what_the_small_run_shows_of_the_declared_results() -> None:
    protocol = DESIGN["protocol"]
    assert skeleton.empty_slots(protocol, ["growth", "survival_count", "other"]) == []
    assert skeleton.empty_slots(protocol, ["growth"]) == ["survival"]
    assert skeleton.unknown_names(protocol, ["growth", "extra_curve", "survival_total"]) == ["extra_curve"]


# ---- the frame in the harness -----------------------------------------------------------------------------------------

def _quest(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (root / "code" / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME, encoding="utf-8")
    return root


def _small(root: Path) -> small_run.SmallRun:
    return asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py",
                                     DESIGN["protocol"]["grid"], deterministic=False, timeout_s=120))


def test_the_framed_simulation_runs_in_fis_harness_and_records_what_it_read(tmp_path: Path) -> None:
    text, why = skeleton.fill(_frame(), BODY)
    assert text is not None, why
    got = _small(_quest(tmp_path, text))
    assert not got.crashed, got.says()
    assert set(got.record()["returned"]) == {"growth", "survival_count", "survival_total", "extra_curve"}
    assert got.reads["measurable"] and got.reads["never"] == []
    assert skeleton.unknown_names(DESIGN["protocol"], got.record()["returned"]) == ["extra_curve"]
    assert skeleton.empty_slots(DESIGN["protocol"], got.record()["returned"]) == []


def test_a_strict_frame_refuses_a_name_the_plan_does_not_declare_at_the_first_trial(tmp_path: Path) -> None:
    text, _ = skeleton.fill(_frame(strict=True), BODY)
    assert text is not None
    got = _small(_quest(tmp_path, text))
    assert got.crashed
    assert "`extra_curve` is not a result the plan declares" in got.directive() and "growth" in got.directive()


def test_the_seed_comes_from_fi_and_the_trial_is_repeatable(tmp_path: Path) -> None:
    text, _ = skeleton.fill(_frame(), BODY)
    root = _quest(tmp_path, text or "")
    one, two = _small(root), _small(root)
    assert one.settings[0].values["growth"] == two.settings[0].values["growth"]
    assert len(set(one.settings[0].values["growth"])) == 3, "three trials, three seeds"


# ---- the code-writing step ---------------------------------------------------------------------------------------------

def _engine_with(tmp_path: Path, on: bool) -> Any:
    from tests.test_disclosure import _engine

    eng = _engine(tmp_path)
    eng.config.execution.code_skeleton = on
    return eng


def test_the_frame_is_off_unless_asked_for(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path, False)
    state = {"design": DESIGN}
    assert eng._frame_note(state) == "" and eng._framed(state, BODY) == BODY
    assert not (tmp_path / "code" / skeleton.RUNTIME_FILE).exists()


def test_the_code_step_puts_the_models_trial_into_the_frame_and_writes_the_runtime(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path, True)
    state = {"design": DESIGN}
    note = eng._frame_note(state)
    assert "ONLY the function `trial(cell, ctx)`" in note and "`growth`" in note and "`survival`" in note
    text = eng._framed(state, BODY)
    assert skeleton.has_frame(text) and "def run_trial(cell, trial_id, seed)" in text and "def _noise(ctx)" in text
    assert (tmp_path / "code" / skeleton.RUNTIME_FILE).read_text(encoding="utf-8") == skeleton.RUNTIME


def test_a_plan_the_frame_does_not_fit_is_written_as_before(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path, True)
    search = {"protocol": {**DESIGN["protocol"], "optimisation": {"design_variables": []}}}
    assert eng._frame_note({"design": search}) == "" and eng._framed({"design": search}, BODY) == BODY
    whole = "def run_trial(cell, trial_id, seed):\n    return {'growth': 1.0}\n"
    assert eng._framed({"design": DESIGN}, whole) == whole, "a reply that is a whole simulation is kept"


def test_values_compared_inside_each_run_pass_through_the_frame(tmp_path: Path) -> None:
    body = "def trial(cell, ctx):\n    out = TrialResult()\n    out['growth'] = [cell['rate'] * t for t in cell['t']]\n    return out\n"
    text, why = skeleton.fill(_frame(), body)
    assert text is not None, why
    root = _quest(tmp_path, text)
    got = asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py",
                                    {"rate": [0.1, 0.2]}, deterministic=False, timeout_s=120, in_each_run={"t": [1, 2]}))
    assert not got.crashed, got.says()
    assert any(name.startswith("growth") for name in got.record()["returned"])


# ---- what the model's reply brings besides its function -----------------------------------------------------------------

CARRY = '''
import math
from dataclasses import dataclass

try:
    import numpy as np
except ImportError:
    np = None

TWO_PI = 2 * math.pi
SCALE = math.sqrt(4.0)

@dataclass
class Tally:
    n: int = 0

    def add(self) -> "Tally":
        return Tally(self.n + 1)

def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = cell["rate"] * TWO_PI * SCALE + Tally().add().n
    out["survival_count"] = 1
    out["survival_total"] = 1
    return out
'''


def test_a_class_a_computed_constant_and_an_import_block_come_with_the_function(tmp_path: Path) -> None:
    frame = _frame()
    text, why = skeleton.fill(frame, CARRY)
    assert text is not None, why
    for kept in ("class Tally", "@dataclass", "TWO_PI = 2 * math.pi", "SCALE = math.sqrt(4.0)", "except ImportError"):
        assert kept in text, kept
    got = _small(_quest(tmp_path, text))
    assert not got.crashed, got.directive()
    assert any(abs(got.settings[0].values["growth"][0] - (r * 4 * 3.141592653589793 + 1)) < 1e-9 for r in (0.1, 0.2, 0.3))


def test_what_the_file_cannot_carry_is_said_plainly() -> None:
    reply = BODY + "\nfor i in range(3):\n    print(i)\n\ndef trial_helper():\n    return 1\n\nTrialResult2 = 1\n"
    frame = _frame()
    text, note = skeleton.compose(frame, reply)
    assert skeleton.has_frame(text) and "left out" in note and "for i in range(3)" in note
    clash = BODY.replace("def _noise(ctx)", "def trial_x(ctx)") + "\ndef run_trial(cell, trial_id, seed):\n    return {}\n"
    kept, note = skeleton.compose(frame, clash)
    assert skeleton.frame_intact(frame, kept) == "" and "return {}" not in kept, "the reply's own entry point never lands"
    assert "`run_trial`" in note and "already has that name" in note
    steps_reply = "def f(x):\n    return x\n\nclass K:\n    pass\n\nGRID = list(range(3))\n\ndef trial(cell, ctx):\n    return {}\n"
    source = "def f(x):\n    return x + 1\n\nclass K:\n    v = 1\n\ndef trial(cell, ctx):\n    raise NotImplementedError\n"
    out, why, left = function_steps.replace_function_ex(source, "trial", steps_reply, signature="def trial(cell, ctx)", equation="t", wide=True)
    assert out is not None, why
    assert "GRID = list(range(3))" in out and out.count("class K") == 1
    assert any("`f`" in x for x in left) and any("`K`" in x for x in left)


# ---- the frame is the frame FI wrote ----------------------------------------------------------------------------------

@pytest.mark.parametrize("extra, said", [
    ("\ndef run_trial(cell, trial_id, seed):\n    return {'growth': 1.0}\n", "run_trial"),
    ("\ndef run_cell(cell):\n    return {'growth': 1.0}\n", "run_cell"),
    ("\nTrialResult.strict = False\n", "TrialResult.strict"),
    ("\nTrialResult.declare(('anything',))\n", "TrialResult.declare"),
    ("\nclass TrialResult(dict):\n    pass\n", "TrialResult"),
    ("\nTrialResult = dict\n", "TrialResult"),
    ("\nclass Ctx:\n    pass\n", "Ctx"),
    ("\nsetattr(TrialResult, 'strict', False)\n", "TrialResult"),
    ("\nfrom fi_frame import TrialResult as T\nT.strict = False\n", "fi_frame"),
])
def test_a_reply_that_defines_the_entry_or_rebinds_the_frame_is_refused(extra: str, said: str) -> None:
    frame = _frame(strict=True)
    ok, _ = skeleton.fill(frame, BODY)
    assert ok is not None and skeleton.frame_intact(frame, ok) == ""
    bad = ok + extra
    problem = skeleton.frame_intact(frame, bad)
    assert problem.startswith("the frame FI wrote was changed") and said in problem, problem
    filled, _why = skeleton.fill(frame, BODY + extra)
    assert filled is None or skeleton.frame_intact(frame, filled) == "", "refused, or left out of the file; never carried in"
    inside = (ok or "").replace("    out = TrialResult()" + chr(10), "    TrialResult.strict = False" + chr(10) + "    out = TrialResult()" + chr(10))
    assert "TrialResult.strict" in skeleton.frame_intact(frame, inside), "also inside the trial"


def test_a_changed_frame_says_what_changed() -> None:
    frame = _frame()
    ok, _ = skeleton.fill(frame, BODY)
    problem = skeleton.frame_intact(frame, (ok or "").replace("RESULTS = (", "RESULTS = ('mine', "))
    assert problem.startswith("the frame FI wrote was changed") and "RESULTS" in problem and "mine" in problem


def test_a_damaged_frame_is_met_as_a_changed_frame_not_as_a_missing_module(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path, True)
    state = {"design": DESIGN}
    good = eng._framed(state, BODY)
    damaged = good.replace("seed=seed", "seed=1")
    text = eng._framed(state, damaged)
    assert (tmp_path / "code" / skeleton.RUNTIME_FILE).read_text(encoding="utf-8") == skeleton.RUNTIME, "written whenever the frame is used"
    got = _small(_quest(tmp_path / "again", text))
    assert got.crashed
    said = got.directive()
    assert "the frame FI wrote was changed" in said and "seed" in said and "ModuleNotFoundError" not in said
    assert "ModuleNotFoundError" not in got.says()


def test_the_runtime_is_written_even_when_the_models_reply_carries_its_own_frame(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path, True)
    state = {"design": DESIGN}
    whole = skeleton.fill(_frame(), BODY)[0] or ""
    (tmp_path / "code").mkdir(exist_ok=True)
    (tmp_path / "code" / skeleton.RUNTIME_FILE).unlink(missing_ok=True)
    eng._framed(state, whole + "\nTrialResult.strict = False\n")
    assert (tmp_path / "code" / skeleton.RUNTIME_FILE).is_file()


# ---- a small run with the frame, through the gate ---------------------------------------------------------------------

def _gate_engine(tmp_path: Path, design: dict[str, Any]) -> Any:
    eng = _engine_with(tmp_path, True)
    eng._draft_protocol = lambda state: design["protocol"]  # type: ignore[assignment]
    eng._protocol_block = lambda state: design["protocol"]  # type: ignore[assignment]
    eng._trial_mode, eng._trial_entries = True, {"run_trial"}
    eng.executor = SharedInterpreterExecutor(python_version="3.11")
    (tmp_path / "plan.md").write_text("# Plan\n\n## Method\nx\n", encoding="utf-8")
    return eng


def _framed_trial(tmp_path: Path, design: dict[str, Any], body: str) -> Any:
    eng = _gate_engine(tmp_path, design)
    state = {"design": design, "iteration": 0, "exec_reflect_iter": 0}
    text = eng._framed(state, body)
    assert skeleton.has_frame(text)
    (tmp_path / "code" / "simulate.py").write_text(text, encoding="utf-8")
    return eng, state


def test_the_small_run_of_a_framed_simulation_sends_back_a_compared_setting_it_never_read(tmp_path: Path) -> None:
    design = {"comparisons": ["dose"], "protocol": DESIGN["protocol"]}
    ignoring = BODY.replace('cell["dose"]', "2")
    eng, state = _framed_trial(tmp_path, design, ignoring)
    sent = asyncio.run(eng._small_run_gate(state, sys.executable, {}))
    assert sent is not None and "`dose`" in sent["exec_result"]["stderr_tail"]
    honest = tmp_path / "honest"
    honest.mkdir()
    eng2, state2 = _framed_trial(honest, design, BODY)
    assert asyncio.run(eng2._small_run_gate(state2, sys.executable, {})) is None
    assert eng2._small_run_note and "never asked for" not in eng2._small_run_note



# ---- the function-step path is not changed; the wider carry is the frame's alone --------------------------------------------

_STEP_SOURCE = "import numpy as np\nfrom foo import Bar\n\nLIMIT = 3\n\ndef f(x):\n    return x\n\ndef g(x):\n    raise NotImplementedError\n"
_STEP_REPLY = ("import math\nnp = None\nclass Bar:\n    pass\nKEY = open('x').read()\nSCALE = 2\nTAU = math.pi * 2\n"
               "def helper(y):\n    return y\ndef g(x):\n    return helper(x) * SCALE\n")
#: What replace_function gave for these two before the frame existed (a912fa03), byte for byte.
_STEP_OLD = ("import numpy as np\nfrom foo import Bar\nimport math\n\n\nLIMIT = 3\n\ndef f(x):\n    return x\n\n# E1\ndef g(x):\n"
             "    return helper(x) * SCALE\n\n\nnp = None\n\n\nSCALE = 2\n\n\ndef helper(y):\n    return y\n")


def test_a_function_step_replacement_is_byte_identical_to_what_it_always_gave() -> None:
    out, why = function_steps.replace_function(_STEP_SOURCE, "g", _STEP_REPLY, signature="def g(x)", equation="E1")
    assert why == "" and out == _STEP_OLD


def test_the_wide_carry_keeps_the_files_own_imports_and_says_what_it_left_out() -> None:
    out, why, left = function_steps.replace_function_ex(_STEP_SOURCE, "g", _STEP_REPLY, signature="def g(x)", equation="E1", wide=True)
    assert out is not None, why
    assert "np = None" not in out and out.count("class Bar") == 0 and "TAU = math.pi * 2" in out
    assert any(x.startswith("a constant `np`") for x in left) and any(x.startswith("a class `Bar`") for x in left)


def test_an_import_of_a_name_the_frame_uses_is_not_a_rebinding() -> None:
    frame = _frame()
    ok, why = skeleton.fill(frame, "from helpers import run_trial\n" + BODY)
    assert ok is not None, why
    assert skeleton.frame_intact(frame, ok) == ""


def test_a_reply_the_frame_refuses_stops_the_run_with_the_reason() -> None:
    bad_params, note = skeleton.compose(_frame(), "def trial(cell, seed):\n    return {}\n")
    assert "(cell, ctx)" in note
    with pytest.raises(RuntimeError, match=r"\(cell, ctx\)"):
        exec(compile(bad_params, "simulate.py", "exec"), {})
    clash = "import fi_frame" + chr(10) + BODY
    refused, note2 = skeleton.compose(_frame(), clash)
    assert skeleton.has_frame(refused) and "fi_frame" in note2  # FI's frame, not the reply: a later repair is checked against it
    with pytest.raises(RuntimeError, match="fi_frame"):
        exec(compile(refused, "simulate.py", "exec"), {})
