"""VenvExecutor tests. Slower than unit tests because they create a venv."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.execution import VenvExecutor


@pytest.fixture(scope="module")
def venv_quest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("venv-quest")


@pytest.mark.asyncio
async def test_setup_creates_venv(venv_quest: Path) -> None:
    exe = VenvExecutor()
    await exe.setup(venv_quest)
    py = exe.python_path(venv_quest)
    assert py.exists(), f"venv python missing at {py}"


@pytest.mark.asyncio
async def test_execute_simple_script(venv_quest: Path) -> None:
    exe = VenvExecutor()
    await exe.setup(venv_quest)
    py = exe.python_path(venv_quest)
    result = await exe.execute(
        [str(py), "-c", "print('hello-fi')"],
        cwd=venv_quest,
        timeout_s=30,
    )
    assert result.returncode == 0
    assert "hello-fi" in result.stdout
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_execute_timeout(venv_quest: Path) -> None:
    exe = VenvExecutor()
    await exe.setup(venv_quest)
    py = exe.python_path(venv_quest)
    result = await exe.execute(
        [str(py), "-c", "import time; time.sleep(5)"],
        cwd=venv_quest,
        timeout_s=2,
    )
    assert result.timed_out is True
    assert result.returncode != 0


@pytest.mark.asyncio
async def test_cleanup_after_success_freezes_and_removes_venv(
    tmp_path: Path,
) -> None:
    """On a successful quest finish, the executor freezes the venv to
    ``.fi/requirements.lock.txt`` and removes ``.venv/`` to reclaim
    disk space. The lock file is what a future reader uses to
    reproduce the environment without keeping the heavy ``.venv/``
    around."""
    quest_root = tmp_path / "quest-clean"
    quest_root.mkdir()
    exe = VenvExecutor()
    await exe.setup(quest_root)
    venv_dir = quest_root / ".venv"
    assert (venv_dir / "pyvenv.cfg").exists()

    lock_path = await exe.cleanup_after_success(quest_root)

    # The lock file is the *load-bearing* artifact — reproduce depends
    # on it. We assert it firmly.
    assert lock_path is not None
    assert lock_path == quest_root / ".fi" / "requirements.lock.txt"
    assert lock_path.is_file()
    body = lock_path.read_text(encoding="utf-8")
    # The header carries reproduction commands for both platforms.
    assert "Reproduce" in body
    assert ".venv/bin/pip install -r .fi/requirements.lock.txt" in body
    assert ".venv\\Scripts\\pip install -r .fi\\requirements.lock.txt" in body
    # The .venv directory deletion is *best-effort*. On POSIX it
    # reliably goes away; on Windows ``shutil.rmtree(ignore_errors=
    # True)`` can leave residue when an AV scanner holds a DLL open.
    # The contract is "freeze succeeded → lock_path returned"; the
    # delete is a disk-reclaim convenience, not a correctness
    # requirement. Assert removal only where the FS makes that
    # reliable.
    if sys.platform != "win32":
        assert not venv_dir.exists(), "venv dir should be removed after cleanup"


@pytest.mark.asyncio
async def test_cleanup_after_success_is_noop_when_no_venv(tmp_path: Path) -> None:
    """When the quest never created a venv (no_simulation mode, or
    cleanup already ran on a prior resume), cleanup_after_success is a
    quiet no-op returning None — never raises."""
    quest_root = tmp_path / "quest-no-venv"
    quest_root.mkdir()
    exe = VenvExecutor()
    result = await exe.cleanup_after_success(quest_root)
    assert result is None
    # No .fi dir should be created speculatively when there's nothing
    # to freeze.
    assert not (quest_root / ".fi" / "requirements.lock.txt").exists()
