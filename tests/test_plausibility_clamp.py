"""A value capped at its bound is a lie the range check could not see.

A real quest: an integrator diverged, the plausibility gate rejected the huge
error, and the regenerated script read ``rmse = 10.0  # Cap at assertion max
for divergence``. Divergence became an in-range number. These pin the check
that now rejects that -- and, just as much, the legitimate code and results
it has to leave alone.
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
from core.engine import Engine, _assertion_violations

RMSE_MAX = {"result_assertions": [
    {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
]}

OBSERVED = '''
import json
import numpy as np
rmse = float(np.sqrt(np.mean(err ** 2)))
if not np.isfinite(rmse) or rmse > 1e6:
    rmse = 10.0  # Cap at assertion max for divergence
print("RESULT_JSON: " + json.dumps({"rmse": rmse}))
'''


# --- recognising a cap -------------------------------------------------------

@pytest.mark.parametrize("code", [
    "rmse = min(rmse, 10.0)",
    "rmse = np.clip(rmse, 0, 10)",
    "rmse = np.minimum(rmse, 10.0)",
    "rmse = torch.clamp(rmse, max=10.0)",
    "rmse = 10.0 if diverged else rmse",
    "CAP = 10.0\nrmse = min(rmse, CAP)",
    "if rmse > 1e6:\n    rmse = 10.0",
    "rmse = np.where(rmse > 10, 10.0, rmse)",
    "rmse = np.nan_to_num(rmse, posinf=10.0)",
])
def test_cap_spellings_are_recognised(code: str) -> None:
    assert 10.0 in plausibility.clamp_constants(code)


@pytest.mark.parametrize("code", [
    "fig, ax = plt.subplots(figsize=(10, 6))",
    "for i in range(10):\n    pass",
    "n_bins = 10",
    "x = y * 10.0",
])
def test_ordinary_uses_of_the_number_are_not_caps(code: str) -> None:
    assert 10.0 not in plausibility.clamp_constants(code)


def test_unparseable_code_yields_nothing() -> None:
    assert plausibility.clamp_constants("def broken(:\n") == set()


# --- the check: both halves are required -------------------------------------

def test_the_observed_script_is_rejected() -> None:
    v = plausibility.check_design({"rmse": 10.0}, RMSE_MAX, code=OBSERVED)
    assert [x.kind for x in v] == ["clamped"]
    assert "rmse = 10 sits exactly on a bound" in v[0].describe()


def test_a_value_on_its_bound_without_a_cap_is_a_real_result() -> None:
    """A perfect score is allowed to be perfect."""
    design = {"result_assertions": [{"path": "accuracy", "min": 0, "max": 1}]}
    assert plausibility.check_design(
        {"accuracy": 1.0}, design, code="accuracy = correct / total",
    ) == []


def test_a_cap_that_did_not_fire_rejects_nothing() -> None:
    assert plausibility.check_design({"rmse": 3.2}, RMSE_MAX, code=OBSERVED) == []


def test_zero_bounds_are_exempt() -> None:
    """Real zeros and ``max(0, x)`` are both everywhere."""
    design = {"result_assertions": [{"path": "var", "min": 0.0, "max": 5.0}]}
    assert plausibility.check_design(
        {"var": 0.0}, design, code="var = max(0.0, var)",
    ) == []


def test_a_floor_cap_on_a_nonzero_min_is_rejected_too() -> None:
    design = {"result_assertions": [{"path": "k1", "min": 0.25, "max": 0.5}]}
    v = plausibility.check_design({"k1": 0.25}, design, code="k1 = max(k1, 0.25)")
    assert [x.kind for x in v] == ["clamped"]


def test_without_code_the_check_is_unchanged() -> None:
    assert plausibility.check_design({"rmse": 10.0}, RMSE_MAX) == []
    assert [x.kind for x in plausibility.check_design({"rmse": 11.0}, RMSE_MAX)] == [
        "out_of_range"
    ]


def test_an_honest_null_passes() -> None:
    """What the repair is told to emit instead of a cap."""
    assert plausibility.check_design(
        {"rmse": None, "diverged": True}, RMSE_MAX, code=OBSERVED,
    ) == []


# --- wiring: the engine's gate, router and repair prompt ----------------------

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


def _capped_state() -> dict:
    return {
        "exec_result": {"returncode": 0, "stdout_tail": "", "stderr_tail": "",
                        "duration_s": 1.0},
        "result_json": {"rmse": 10.0}, "design": RMSE_MAX, "code": OBSERVED,
        "exec_reflect_iter": 0,
    }


def test_engine_gate_reads_the_script_off_the_state() -> None:
    assert [v.kind for v in _assertion_violations(_capped_state())] == ["clamped"]


def test_engine_gate_leaves_a_real_zero_alone() -> None:
    state = {"design": RMSE_MAX, "result_json": {"rmse": 0.0},
             "code": "rmse = max(0.0, rmse)"}
    assert _assertion_violations(state) == []


def test_a_capped_result_routes_back_to_repair(tmp_path: Path) -> None:
    assert _engine(tmp_path)._route_after_execute_reflect(_capped_state()) == "retry"


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
async def test_repair_prompt_names_the_bound_and_forbids_the_cap(tmp_path: Path) -> None:
    """The model used to see rc=0 and nothing else, so it invented a bug."""
    prompt, patch = await _reflect_prompt(_engine(tmp_path), _capped_state())
    assert "IMPLAUSIBLE RESULT" in prompt
    assert "rmse = 10 sits exactly on a bound" in prompt
    assert "NEVER clamp" in prompt and "remove that cap" in prompt
    assert patch["exec_reflect_iter"] == 1


@pytest.mark.asyncio
async def test_out_of_range_prompt_names_the_bound_without_the_cap_note(
    tmp_path: Path,
) -> None:
    state = {**_capped_state(), "result_json": {"rmse": 1e6}, "code": "rmse = 1e6"}
    prompt, _ = await _reflect_prompt(_engine(tmp_path), state)
    assert "rmse = 1e+06 violates rmse must be [0, 10]" in prompt
    assert "remove that cap" not in prompt
    assert "sits exactly on a bound in several settings" not in prompt


@pytest.mark.asyncio
async def test_a_quantity_on_its_bound_in_several_settings_gets_the_trivial_answer_note(
    tmp_path: Path,
) -> None:
    """A computation that returned the trivial answer is in range, so the repair
    has to be told what to look for, and that a real bound value stays."""
    state = {
        **_capped_state(),
        # One non-zero value, or the all-zero guard answers first.
        "result_json": {"by_h": {"0.1": {"rmse": 0.0}, "0.5": {"rmse": 0.0}, "1.0": {"rmse": 2.5}}},
        "code": "rmse = compute(h)",
    }
    prompt, patch = await _reflect_prompt(_engine(tmp_path), state)
    assert "rmse equals its bound 0 in 2 settings" in prompt
    assert "sits exactly on a bound in several settings" in prompt
    assert "leave the code as it is" in prompt
    assert "remove that cap" not in prompt
    assert patch["exec_reflect_iter"] == 1
