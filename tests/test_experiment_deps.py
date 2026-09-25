"""What a quest environment is given: packages, the selected skills' packages, and library skills on the path.

The live failure this pins (2026-09-25, a user's quest on another machine): the code-writing step asked pip for
``lieflat_charts`` (a Node.js tool skill, not a Python package) together with numpy; pip installed nothing, and the
oracle run died on ``import numpy``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import experiment_deps as deps_mod
from core.execution import ExecutionResult
from tests.test_engine_execute_retry import _mk_engine, _mock_execute_router


def _skill(tmp_path: Path, name: str, kind: str, pip_requires: list[str] | None = None) -> SimpleNamespace:
    folder = tmp_path / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    prov = {"pip_requires": pip_requires or []}
    return SimpleNamespace(
        name=name, path=folder, kind=SimpleNamespace(value=kind),
        provenance=lambda prov=prov: prov,
    )


def test_a_tool_skill_a_path_imported_library_or_a_local_file_is_never_asked_of_pip(tmp_path: Path) -> None:
    tool = _skill(tmp_path, "lieflat-charts", "tool")
    lib = _skill(tmp_path, "sir-kernels", "library")
    (lib.path / "sir_kernels.py").write_text("x = 1", encoding="utf-8")   # imported from the skill's folder
    named_after_package = _skill(tmp_path, "scipy", "library")             # a skill that teaches a PyPI package
    install, dropped = deps_mod.split_deps(
        ["numpy>=1.26", "lieflat_charts", "Sir.Kernels", "helpers", "numpy", "scipy>=1.11", "matplotlib"],
        skills=[tool, lib, named_after_package], local_modules=["experiment", "helpers"],
    )
    assert install == ["numpy>=1.26", "scipy>=1.11", "matplotlib"], "scipy is a real package, not the skill's module"
    reasons = dict(dropped)
    assert "a tool, not a Python package" in reasons["lieflat_charts"]
    assert "puts on the experiment's path" in reasons["Sir.Kernels"]
    assert "code/helpers.py" in reasons["helpers"]


def test_the_note_names_a_selected_skill_that_cannot_be_used() -> None:
    note = deps_mod.repair_note([], [], ["broken-skill"])
    assert note.startswith("FI NOTE (skills)") and "broken-skill" in note and "do not import" in note


def test_skill_requirements_and_library_paths(tmp_path: Path) -> None:
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy>=1.11", "numpy"])
    (lib.path / "scripts").mkdir()
    tool = _skill(tmp_path, "lieflat-charts", "tool", ["numpy"])
    assert deps_mod.skill_requirements([lib, tool]) == ["scipy>=1.11", "numpy"]
    assert deps_mod.library_paths([lib, tool]) == [lib.path, lib.path / "scripts"]


def test_failure_reasons_are_plain() -> None:
    assert "no such package on PyPI" in deps_mod.failure_reason(
        "ERROR: Could not find a version that satisfies the requirement lieflat_charts\n"
        "ERROR: No matching distribution found for lieflat_charts")
    assert deps_mod.repair_note([], []) == ""
    note = deps_mod.repair_note([("lieflat_charts", "is the skill lieflat-charts, a tool")], [("nopkg", "no such package on PyPI")])
    assert note.startswith("FI NOTE (packages)") and "lieflat_charts" in note and "remove the import" in note


@pytest.mark.asyncio
async def test_one_missing_package_no_longer_leaves_numpy_uninstalled(tmp_path: Path) -> None:
    """The batch line fails on one name; FI installs the rest one at a time and tells the repair which failed."""
    eng = _mk_engine(tmp_path)
    ok = ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    missing = ExecutionResult(returncode=1, stdout="", stderr="ERROR: No matching distribution found for nopkg", duration_s=0.1)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return missing if "nopkg" in pkgs else ok

    eng.executor.install = install  # type: ignore[method-assign]
    crash = ExecutionResult(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'nopkg'", duration_s=1.0)
    eng.executor.execute = _mock_execute_router(crash)  # type: ignore[method-assign]

    update = await eng._node_execute({"deps": ["numpy", "nopkg", "matplotlib"]})

    assert installed[0] == ["numpy", "nopkg", "matplotlib"]   # one line first
    assert ["numpy"] in installed and ["matplotlib"] in installed  # then each on its own
    stderr = update["exec_result"]["stderr_tail"]
    assert stderr.startswith("FI NOTE (packages)") and "nopkg: no such package on PyPI" in stderr


@pytest.mark.asyncio
async def test_selected_skills_packages_are_installed_and_libraries_are_on_the_path(tmp_path: Path, monkeypatch) -> None:
    eng = _mk_engine(tmp_path)
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy"])
    tool = _skill(tmp_path, "lieflat-charts", "tool")
    monkeypatch.setattr(eng, "_experiment_skill_states", lambda _state: [SimpleNamespace(skill=lib), SimpleNamespace(skill=tool)])
    # A quest's own clean environment (the research profile's): the skills' packages go into it.
    monkeypatch.setattr(eng.config.execution, "shared_interpreter", False)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    envs: list[dict] = []
    happy = ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0)

    async def execute(cmd, *_, env=None, **__):  # noqa: ANN001
        envs.append(env or {})
        return happy

    eng.executor.execute = execute  # type: ignore[method-assign]
    await eng._node_execute({"deps": ["numpy", "lieflat_charts"]})

    assert installed == [["numpy", "scipy"]]      # the tool's name left out, the library's own package added
    run_env = envs[-1]
    assert str(lib.path) in run_env.get("PYTHONPATH", "").split(__import__("os").pathsep)
    assert str(tool.path) not in run_env.get("PYTHONPATH", "")


@pytest.mark.asyncio
async def test_on_the_shared_interpreter_a_quest_does_not_install_the_skills_packages(tmp_path: Path, monkeypatch) -> None:
    """They are in FI's own interpreter from the skill's approval; a quest does not change it."""
    eng = _mk_engine(tmp_path)
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy"])
    monkeypatch.setattr(eng, "_experiment_skill_states", lambda _state: [SimpleNamespace(skill=lib)])
    monkeypatch.setattr(eng.config.execution, "shared_interpreter", True)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    eng.executor.execute = _mock_execute_router(  # type: ignore[method-assign]
        ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0))
    await eng._node_execute({"deps": ["numpy"]})
    assert installed == [["numpy"]]
