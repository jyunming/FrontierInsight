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
