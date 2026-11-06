"""The range check never guesses a cap from the script's code (-380).

An earlier version reported a result as a "clamped number" when it sat on a constant the script's code caps something at
(``min(x, c)``, ``np.clip``, ``max(R, 1)`` under a denominator, ``out["failed"] = 1.0``). Over every quest it fired on it
was wrong: the values were 0/1 indicators and the "cap" a denominator floor, an exit code or a failure flag, and each
flag cost a repair round. Which result a piece of free code feeds cannot be told from the code, so the check no longer
reads it for caps. What stays: a value outside its range, a bounded quantity that disappeared, and a quantity on one bound
in several settings, which is disclosed in the paper and never sent back for repair.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import plausibility
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine, _assertion_violations, _assertion_warnings

RMSE_MAX = {"result_assertions": [
    {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
]}

# The script a past quest wrote after a bound rejected a diverged run: the divergence became an in-range number.
CAPPING_SCRIPT = '''
import json
import numpy as np
rmse = float(np.sqrt(np.mean(err ** 2)))
if not np.isfinite(rmse) or rmse > 1e6:
    rmse = 10.0  # Cap at assertion max for divergence
print("RESULT_JSON: " + json.dumps({"rmse": rmse}))
'''


def _kinds(found: list) -> list[str]:
    return [v.kind for v in found]


# --- (a) the -380 shape: a failure flag in the trial code is not a cap of the coverage ------------------------------

COVERAGE = {"result_assertions": [{"path": "coverage", "min": 0.0, "max": 1.0, "unit": "fraction"}]}
FAILURE_FLAG_TRIAL = '''
def run_trial(params, seed):
    out = {"coverage": float(covered)}
    if failed:
        out["failed"] = 1.0
    return out
'''
SEVEN_SETTINGS = {"by_setting": {f"s{i}": {"coverage": 1.0} for i in range(7)}}


def test_a_failure_flag_in_the_trial_code_is_not_a_cap_of_the_coverage() -> None:
    found = plausibility.check_design(SEVEN_SETTINGS, COVERAGE, code=FAILURE_FLAG_TRIAL)
    assert "clamped" not in _kinds(found) and "clamped_unproven" not in _kinds(found)
    assert _kinds(found) == ["at_bound"]  # said, in the paper: not a repair


def test_that_shape_sends_nothing_back_and_is_disclosed() -> None:
    state = {"design": COVERAGE, "result_json": SEVEN_SETTINGS, "code": FAILURE_FLAG_TRIAL}
    assert _assertion_violations(state) == []
    (said,) = _assertion_warnings(state)
    assert said.kind == "at_bound" and "coverage equals its bound 1 in 7 settings" in said.disclosure()


def test_a_plan_that_says_the_bound_is_honest_is_not_even_disclosed() -> None:
    design = {"result_assertions": [{**COVERAGE["result_assertions"][0], "may_equal_bound": True}]}
    assert plausibility.check_design(SEVEN_SETTINGS, design, code=FAILURE_FLAG_TRIAL) == []


# --- (b) a floor under a denominator is not a cap ----------------------------------------------------------------------

PROPORTION = {"result_assertions": [{"path": "fdr", "min": 0.0, "max": 1.0, "unit": "dimensionless"}]}


@pytest.mark.parametrize("code", [
    "FDR = V / max(R, 1)",
    "FDR = V / np.maximum(R, 1)",
    "FDR = V / np.clip(R, 1, None)",
    'out = {"fdr": float(np.mean(V / max(R, 1)))}',
    'out = {"fdr": min(V / max(R, 1), 1.0)}',
    "hit = 1 if v >= 1 else 0",
])
def test_a_denominator_floor_with_an_fdr_of_one_is_no_violation(code: str) -> None:
    assert plausibility.check_design({"fdr": 1.0}, PROPORTION, code=code) == []


# --- (c) known positives: a real cap is still seen where the results show it ----------------------------------------------

CAPPED_PROPORTION = '''
raw = hits / total
out = {"share": min(raw, 1.0)}
'''
SHARE = {"result_assertions": [{"path": "share", "min": 0.0, "max": 1.0, "unit": "dimensionless"}]}


def test_a_real_cap_with_the_value_on_the_bound_in_three_settings_is_still_disclosed() -> None:
    result = {"by_setting": {f"s{i}": {"share": 1.0} for i in range(3)}}
    (found,) = plausibility.check_design(result, SHARE, code=CAPPED_PROPORTION)
    assert found.kind == "at_bound" and len(found.paths) == 3
    assert "share equals its bound 1 in 3 settings" in found.disclosure()


def test_a_value_out_of_range_is_still_a_violation() -> None:
    found = plausibility.check_design({"rmse": 11.0}, RMSE_MAX, code=CAPPING_SCRIPT)
    assert _kinds(found) == ["out_of_range"]
    assert _kinds(plausibility.check_design({"rmse": 11.0}, RMSE_MAX)) == ["out_of_range"]


# --- what a capping script's result looks like to the check now --------------------------------------------------------

@pytest.mark.parametrize("code", [
    CAPPING_SCRIPT,
    "rmse = min(rmse, 10.0)",
    "rmse = np.clip(rmse, 0, 10)",
    "rmse = torch.clamp(rmse, max=10.0)",
    "rmse = 10.0 if diverged else rmse",
    "if rmse > 1e6:\n    rmse = 10.0",
])
def test_one_value_on_its_bound_is_left_alone_whatever_the_code_does(code: str) -> None:
    assert plausibility.check_design({"rmse": 10.0}, RMSE_MAX, code=code) == []


def test_a_floor_value_inside_the_range_is_left_alone() -> None:
    design = {"result_assertions": [{"path": "k1", "min": 0.0, "max": 0.5}]}
    assert plausibility.check_design({"k1": 0.25}, design, code="k1 = max(k1, 0.25)") == []


def test_zero_bounds_and_an_honest_null_pass() -> None:
    design = {"result_assertions": [{"path": "var", "min": 0.0, "max": 5.0}]}
    assert plausibility.check_design({"var": 0.0}, design, code="var = max(0.0, var)") == []
    assert plausibility.check_design({"rmse": None, "diverged": True}, RMSE_MAX, code=CAPPING_SCRIPT) == []


def test_the_removed_code_readers_are_gone() -> None:
    for name in ("clamp_constants", "clamp_directions", "direct_caps", "caps_of_other_values"):
        assert not hasattr(plausibility, name), name


# --- wiring: the engine's gate, router and repair prompt ----------------------------------------------------------------

def _engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="clamp", title="clamp",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.executor.install = AsyncMock(  # type: ignore[method-assign]
        return_value=type("IR", (), {"returncode": 0, "stderr": ""})())
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    return eng


def _state(result: dict, code: str = CAPPING_SCRIPT) -> dict:
    return {
        "exec_result": {"returncode": 0, "stdout_tail": "", "stderr_tail": "",
                        "duration_s": 1.0},
        "result_json": result, "design": RMSE_MAX, "code": code,
        "exec_reflect_iter": 0,
    }


def test_a_result_on_a_bound_the_script_caps_is_not_sent_back(tmp_path: Path) -> None:
    state = _state({"rmse": 10.0})
    assert _assertion_violations(state) == []
    assert _engine(tmp_path)._route_after_execute_reflect(state) != "retry"


def test_engine_gate_leaves_a_real_zero_alone() -> None:
    state = {"design": RMSE_MAX, "result_json": {"rmse": 0.0}, "code": "rmse = max(0.0, rmse)"}
    assert _assertion_violations(state) == []


async def _reflect_prompt(eng: Engine, state: dict) -> tuple[str, dict]:
    seen: dict = {}

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        seen["prompt"] = prompt
        return json.dumps({
            "code": 'print("RESULT_JSON: {\\"rmse\\": null, \\"diverged\\": true}")',
            "patch_summary": "report the divergence instead of capping it",
        })

    eng._chat = fake_chat  # type: ignore[assignment]
    patch = await eng._node_execute_reflect(state)  # type: ignore[arg-type]
    return seen.get("prompt", ""), patch


@pytest.mark.asyncio
async def test_out_of_range_prompt_names_the_bound_and_forbids_the_cap(tmp_path: Path) -> None:
    """The model used to see rc=0 and nothing else, so it invented a bug."""
    prompt, patch = await _reflect_prompt(_engine(tmp_path), _state({"rmse": 1e6}, code="rmse = 1e6"))
    assert "IMPLAUSIBLE RESULT" in prompt
    assert "rmse = 1e+06 violates rmse must be [0, 10]" in prompt
    assert "NEVER clamp" in prompt
    assert "remove that cap" not in prompt and "capped by the script" not in prompt
    assert patch["exec_reflect_iter"] == 1


@pytest.mark.asyncio
async def test_a_quantity_on_its_bound_in_several_settings_is_disclosed_and_not_repaired(
    tmp_path: Path,
) -> None:
    """An undeclared bound hit is a guess about the computation, so it is said in the paper's record and the code is not
    sent back."""
    state = _state({"by_h": {"0.1": {"rmse": 0.0}, "0.5": {"rmse": 0.0}, "1.0": {"rmse": 2.5}}}, code="rmse = compute(h)")
    assert _assertion_violations(state) == []
    (said,) = _assertion_warnings(state)
    assert said.kind == "at_bound" and "rmse equals its bound 0 in 2 settings" in said.disclosure()
    prompt, _ = await _reflect_prompt(_engine(tmp_path), state)
    assert "sits exactly on a bound in several settings" not in prompt
