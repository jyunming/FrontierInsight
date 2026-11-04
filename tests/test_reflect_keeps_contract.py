"""A repaired script must keep what FI calls it by: a stand-in word as the whole of simulate.py is refused.

A live repair of a crashed simulation answered ``{"code": "PLACEHOLDER", "patch_summary": "PLACEHOLDER"}``; the word
parses as Python, so it was written over the whole of simulate.py and every later test crashed with a NameError.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import trial_runner
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

OLD = "import math\n\n\ndef run_cell(cell):\n    return {'x': float(cell['a'])}\n\n\ndef run_trial(cell, trial_id, seed):\n    return run_cell(cell)\n"
GOOD = OLD.replace("float(cell['a'])", "float(sum(cell['a']))")


def _eng(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="contract tests", title="contract", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(OLD, encoding="utf-8")
    return eng


def _state() -> dict:
    return {"exec_result": {"returncode": 1, "stdout_tail": "", "failed_script": "simulate.py",
                            "stderr_tail": "TypeError: float() argument must be a string or a real number"},
            "result_json": None, "code": "", "design": {"hypothesis": "h"}}


async def _reflect(tmp_path: Path, reply: dict) -> tuple[dict, str]:
    eng = _eng(tmp_path)
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps(reply))})()
    patch = await eng._node_execute_reflect(_state())
    return patch, (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_stand_in_word_is_refused_and_the_script_kept(tmp_path: Path) -> None:
    patch, on_disk = await _reflect(tmp_path, {"code": "PLACEHOLDER", "patch_summary": "PLACEHOLDER", "deps": []})
    assert on_disk == OLD
    assert not patch.get("exec_patch_pending")
    assert patch["exec_reflect_iter"] == 1
    assert "was not used" in patch["exec_reflect_history"][0]["patch_summary"]


@pytest.mark.asyncio
async def test_a_script_that_loses_its_entry_points_is_refused(tmp_path: Path) -> None:
    patch, on_disk = await _reflect(tmp_path, {"code": "def helper(x):\n    return x\n", "patch_summary": "fix"})
    assert on_disk == OLD and not patch.get("exec_patch_pending")
    assert "run_cell" in patch["exec_reflect_history"][0]["patch_summary"]


@pytest.mark.asyncio
async def test_a_real_repair_is_still_written(tmp_path: Path) -> None:
    patch, on_disk = await _reflect(tmp_path, {"code": GOOD, "patch_summary": "sum the list"})
    assert on_disk == GOOD and patch["exec_patch_pending"] is True


def test_contract_helper() -> None:
    f = trial_runner.repair_breaks_contract
    assert f(OLD, "PLACEHOLDER", "simulate.py")
    assert f(OLD, "def (", "simulate.py")
    assert f(OLD, "def run_cell(c):\n    return {}\n", "simulate.py") == "it does not define run_trial, which FI calls and the script has now"
    assert f(OLD, GOOD, "simulate.py") == ""
    assert f("print('RESULT_JSON: {}')", "print('hi')", "experiment.py")
    assert f("print('RESULT_JSON: {}')", "print('RESULT_JSON: {}')", "experiment.py") == ""


def test_an_entry_point_imported_from_a_helper_counts() -> None:
    """A repair that moves run_trial / run_cell into a helper module and imports them still gives FI what it calls."""
    f = trial_runner.repair_breaks_contract
    assert f(OLD, "from pkg.fixed import run_cell, run_trial\n", "simulate.py") == ""
    assert f(OLD, "from pkg.fixed import run_cell\n", "simulate.py") == (
        "it does not define run_trial, which FI calls and the script has now")
    assert f(OLD, "import pkg as run_cell\nfrom pkg.fixed import run_trial\n", "simulate.py")
    assert f(OLD, "run_cell: object\nfrom pkg.fixed import run_trial\n", "simulate.py")
