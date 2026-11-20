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


def test_an_oracle_is_what_the_module_binds_not_a_word_in_its_text() -> None:
    """A trial-mode repair for a check with no case needs ``oracle`` bound at the module's top level, read as Python binds
    it: FI takes ``oracle`` from the module when it calls it, so one imported from a helper is one; a comment, a function
    whose name only starts with the word, or a ``def oracle`` nested in another function is none."""
    f = trial_runner.repair_breaks_contract
    checks = [{"name": "drift"}]  # no case: the script's own oracle() answers it
    imported = GOOD + "from checks import oracle\n"
    assert f(OLD, imported, "simulate.py", oracles=checks, trial=True) == ""
    assert f(OLD, GOOD + "\n\ndef oracle():\n    return {'drift': 0.0}\n", "simulate.py", oracles=checks, trial=True) == ""
    # not imported: read from the code, and the reason says so
    refused = "it defines no oracle() and some check has no case FI can run (read from the code: it was not imported)"
    assert f(OLD, GOOD + "# def oracle() is not needed here\n", "simulate.py", oracles=checks, trial=True) == refused
    assert f(OLD, GOOD + "\n\ndef oracle_helper():\n    return {'drift': 0.0}\n", "simulate.py", oracles=checks,
             trial=True) == refused
    nested = GOOD + "\n\ndef outer():\n    def oracle():\n        return {'drift': 0.0}\n    return oracle\n"
    assert f(OLD, nested, "simulate.py", oracles=checks, trial=True) == refused
    assert f(OLD, GOOD, "simulate.py", oracles=[{"name": "drift", "case": {"a": 1}, "measure": "x"}], trial=True) == ""


# A module binds oracle() when FI, importing it as it does to call it, finds a callable one: asked of the module, not read.
_IN_A_BRANCH = GOOD + "\n\nif True:\n    def oracle():\n        return {'drift': 0.0}\n"
_STAR = GOOD + "\nfrom checks import *\n"
_CHECKS = "__all__ = ['oracle']\n\n\ndef oracle():\n    return {'drift': 0.0}\n\n\ndef _hidden():\n    return 1\n"


def _bound(tmp_path: Path, text: str, helper: str = "") -> tuple:
    import asyncio
    import sys

    from core.execution import SharedInterpreterExecutor

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(text, encoding="utf-8")
    if helper:
        (root / "code" / "checks.py").write_text(helper, encoding="utf-8")
    return asyncio.run(trial_runner.binds_callable(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), sys.executable, root,
        "code/simulate.py", "oracle", timeout_s=60))


def test_whether_a_module_binds_oracle_is_asked_of_the_module_and_nothing_is_called(tmp_path: Path) -> None:
    assert _bound(tmp_path / "a", _IN_A_BRANCH) == (True, "")
    assert _bound(tmp_path / "b", _STAR, _CHECKS) == (True, "")
    assert _bound(tmp_path / "c", GOOD + "\n\ndef oracle_helper():\n    return {}\n") == (False, "")
    assert _bound(tmp_path / "d", GOOD + "\noracle = 3\n") == (False, ""), "a name that is not callable is no oracle()"
    # nothing of the module is called: an oracle() that would leave a file leaves none
    marks = GOOD + "\n\ndef oracle():\n    open('called.txt', 'w').write('x')\n    return {}\n"
    assert _bound(tmp_path / "e", marks) == (True, "")
    assert not list((tmp_path / "e").rglob("called.txt"))
    # the import itself failed: not an answer, and why is kept
    got, why = _bound(tmp_path / "f", GOOD + "\nraise RuntimeError('needs a data file')\n")
    assert got is None and "needs a data file" in why


def test_a_repair_s_oracle_is_what_the_imported_module_binds(tmp_path: Path) -> None:
    """FI's answer from the import decides; the reading of the code only when the import failed, and then it says so."""
    f = trial_runner.repair_breaks_contract
    checks = [{"name": "drift"}]
    refused = "it defines no oracle() and some check has no case FI can run"
    # read from the code, both of these bind no top-level oracle; the module itself binds one
    assert f(OLD, _IN_A_BRANCH, "simulate.py", oracles=checks, trial=True).startswith(refused)
    assert f(OLD, _STAR, "simulate.py", oracles=checks, trial=True).startswith(refused)
    for text in (_IN_A_BRANCH, _STAR):
        assert f(OLD, text, "simulate.py", oracles=checks, trial=True, oracle_bound=True) == ""
    # a def oracle the module rebinds to something else is no oracle(), whatever the text shows
    assert f(OLD, GOOD + "\n\ndef oracle():\n    return {}\noracle = None\n", "simulate.py", oracles=checks, trial=True,
             oracle_bound=False) == refused
    failed = f(OLD, _STAR, "simulate.py", oracles=checks, trial=True, import_why="ImportError: no module checks")
    assert failed == refused + " (read from the code: FI's import of it failed, ImportError: no module checks)"
    assert trial_runner.needs_oracle(checks, True) and not trial_runner.needs_oracle(checks, False)
    assert not trial_runner.needs_oracle([{"name": "drift", "case": {"a": 1}, "measure": "x"}], True)
