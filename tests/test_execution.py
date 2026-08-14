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


# --- A2: setup() must not reuse a partially-built (interrupted) venv ----------


def _fake_cfg_and_interp(exe: VenvExecutor, quest_root: Path, *, interp: bool):
    """Create .venv/pyvenv.cfg and optionally the interpreter file."""
    venv = quest_root / ".venv"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    if interp:
        py = exe.python_path(quest_root)
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("", encoding="utf-8")


@pytest.mark.asyncio
async def test_setup_rebuilds_when_interpreter_missing(tmp_path: Path, monkeypatch):
    """pyvenv.cfg present but NO interpreter (killed mid-build) → rebuild, not
    skip. Reusing it would spawn a missing python on every --resume."""
    exe = VenvExecutor()
    _fake_cfg_and_interp(exe, tmp_path, interp=False)
    built: list[dict] = []
    monkeypatch.setattr("core.execution._build_venv", lambda vd, **kw: built.append(kw))
    await exe.setup(tmp_path)
    assert built, "a venv whose interpreter is missing must be rebuilt"
    assert built[0].get("clear") is True


@pytest.mark.asyncio
async def test_setup_reuses_fully_built_venv(tmp_path: Path, monkeypatch):
    """pyvenv.cfg + interpreter + passing pip probe → reuse, no rebuild."""
    from core.execution import ExecutionResult
    exe = VenvExecutor()
    _fake_cfg_and_interp(exe, tmp_path, interp=True)
    built: list[dict] = []
    monkeypatch.setattr("core.execution._build_venv", lambda vd, **kw: built.append(kw))

    async def ok_exec(cmd, **kw):  # noqa: ANN001
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr(exe, "execute", ok_exec)
    await exe.setup(tmp_path)
    assert not built, "a fully-built venv must be reused, not rebuilt"


@pytest.mark.asyncio
async def test_setup_rebuilds_when_pip_probe_fails(tmp_path: Path, monkeypatch):
    """Interpreter present but pip import fails (partial ensurepip) → rebuild."""
    from core.execution import ExecutionResult
    exe = VenvExecutor()
    _fake_cfg_and_interp(exe, tmp_path, interp=True)
    built: list[dict] = []
    monkeypatch.setattr("core.execution._build_venv", lambda vd, **kw: built.append(kw))

    async def bad_exec(cmd, **kw):  # noqa: ANN001
        return ExecutionResult(returncode=1, stdout="", stderr="No module named pip", duration_s=0.0)

    monkeypatch.setattr(exe, "execute", bad_exec)
    await exe.setup(tmp_path)
    assert built, "a venv failing the pip probe must be rebuilt"
    assert built[0].get("clear") is True


# --- A4: Windows MAX_PATH venv diagnostics ------------------------------------


def test_looks_like_dll_load_failure_signature():
    from core.execution import _looks_like_dll_load_failure
    assert _looks_like_dll_load_failure(
        'ImportError: DLL load failed while importing _imaging: '
        'The filename or extension is too long.'
    )
    assert _looks_like_dll_load_failure("The filename or extension is too long")
    # Ordinary experiment errors must NOT trip it.
    assert not _looks_like_dll_load_failure("ValueError: shapes not aligned")
    assert not _looks_like_dll_load_failure("")


@pytest.mark.skipif(
    not sys.platform.startswith("win"),
    reason="the MAX_PATH DLL hint is Windows-specific and gated to win32",
)
@pytest.mark.asyncio
async def test_execute_logs_dll_hint_on_native_load_failure(tmp_path: Path, caplog):
    """A failed run whose stderr shows a DLL-load failure logs the actionable
    MAX_PATH hint (so even the silent web_plots path surfaces the cause)."""
    import logging
    exe = VenvExecutor()
    py = sys.executable
    code = (
        "import sys; sys.stderr.write('ImportError: DLL load failed while "
        "importing _imaging: The filename or extension is too long.'); "
        "sys.exit(1)"
    )
    with caplog.at_level(logging.WARNING, logger="frontier_insight.execution"):
        res = await exe.execute([py, "-c", code], cwd=tmp_path, timeout_s=30)
    assert res.returncode == 1
    assert any("MAX_PATH" in r.message or "long-path" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_execute_no_hint_on_ordinary_failure(tmp_path: Path, caplog):
    import logging
    exe = VenvExecutor()
    py = sys.executable
    with caplog.at_level(logging.WARNING, logger="frontier_insight.execution"):
        res = await exe.execute(
            [py, "-c", "import sys; sys.stderr.write('ValueError: nope'); sys.exit(1)"],
            cwd=tmp_path, timeout_s=30,
        )
    assert res.returncode == 1
    assert not any("MAX_PATH" in r.message for r in caplog.records)
