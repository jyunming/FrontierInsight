"""VenvExecutor tests. Slower than unit tests because they create a venv."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from core.execution import ExecutionResult, VenvExecutor


@pytest.fixture(scope="module")
def venv_quest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("venv-quest")


@pytest.mark.asyncio
async def test_setup_creates_venv(venv_quest: Path) -> None:
    exe = VenvExecutor()
    await exe.setup(venv_quest)
    py = exe.python_path(venv_quest)
    assert py.exists(), f"venv python missing at {py}"
    # Default is now system_site_packages=True (was the venv.EnvBuilder
    # default of False) — each quest's venv should see what FI's own
    # interpreter already has installed instead of reinstalling it.
    cfg = (venv_quest / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "include-system-site-packages = true" in cfg.lower()


@pytest.mark.asyncio
async def test_setup_honors_system_site_packages_false(tmp_path: Path) -> None:
    """The isolation escape hatch: system_site_packages=False must still
    build a fully isolated venv for anyone who wants it back."""
    exe = VenvExecutor(system_site_packages=False)
    quest_root = tmp_path / "isolated-quest"
    await exe.setup(quest_root)
    cfg = (quest_root / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8")
    assert "include-system-site-packages = false" in cfg.lower()


def test_resolve_python_for_version_prefers_sys_executable_when_it_matches():
    """No subprocess search needed when the running interpreter already IS
    the requested version — the common case on a single-Python machine."""
    from core.execution import _resolve_python_for_version
    want = f"{sys.version_info[0]}.{sys.version_info[1]}"
    assert _resolve_python_for_version(want) == sys.executable


def test_resolve_python_for_version_returns_none_for_an_unavailable_version():
    """A version nobody has installed must not crash — _build_venv falls
    back to sys.executable with a logged warning, not an exception."""
    from core.execution import _resolve_python_for_version
    assert _resolve_python_for_version("2.4") is None


@pytest.mark.asyncio
async def test_build_venv_falls_back_to_sys_executable_and_warns(
    tmp_path: Path, caplog,
) -> None:
    """core.execution.python_version was previously declared but never
    read — venv.EnvBuilder().create() always used sys.executable
    regardless. This pins the NEW behavior's honest fallback: a version
    nobody has installed still produces a working venv (from
    sys.executable), but now says so instead of silently ignoring the
    setting."""
    import logging
    from core.execution import _build_venv
    venv_dir = tmp_path / "fallback-venv"
    with caplog.at_level(logging.WARNING, logger="frontier_insight.execution"):
        await asyncio.to_thread(
            _build_venv, venv_dir, with_pip=True, clear=True,
            python_version="2.4",
        )
    assert (venv_dir / ("Scripts" if sys.platform == "win32" else "bin")).is_dir()
    assert any("no Python 2.4 interpreter found" in r.message for r in caplog.records)


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
async def test_lock_pins_a_requested_package_the_venv_inherited(
    tmp_path: Path,
) -> None:
    """With system_site_packages a satisfied request installs nothing into
    the venv, so ``pip freeze --local`` alone would drop it — and the venv is
    deleted right after, taking the only record of the version the experiment
    ran against. pytest is satisfied purely by inheritance here (no network)."""
    quest_root = tmp_path / "quest-inherit"
    quest_root.mkdir()
    exe = VenvExecutor()
    await exe.setup(quest_root)
    result = await exe.install(["pytest>=1"], quest_root=quest_root)
    assert result.returncode == 0, result.stderr

    lock_path = await exe.cleanup_after_success(quest_root)

    assert lock_path is not None
    body = lock_path.read_text(encoding="utf-8")
    assert f"pytest=={pytest.__version__}" in body
    assert "provided by FI's own interpreter" in body


def test_make_executor_defaults_to_the_shared_interpreter() -> None:
    from core.execution import SharedInterpreterExecutor, make_executor
    exe = make_executor("venv", python_version="3.11", docker_image="x")
    assert isinstance(exe, SharedInterpreterExecutor)
    isolated = make_executor(
        "venv", python_version="3.11", docker_image="x", shared_interpreter=False,
    )
    assert type(isolated) is VenvExecutor


@pytest.mark.asyncio
async def test_shared_interpreter_builds_no_venv_and_runs_on_fi_python(
    tmp_path: Path,
) -> None:
    """The point of 'one Python': no venv is built under the quest's path, and
    quest code runs on the very interpreter that runs FI."""
    from core.execution import SharedInterpreterExecutor
    exe = SharedInterpreterExecutor()
    quest_root = tmp_path / "shared-quest"
    await exe.setup(quest_root)
    assert not (quest_root / ".venv").exists()
    py = exe.python_path(quest_root)
    assert py == Path(sys.executable)
    res = await exe.execute(
        [str(py), "-c", "import sys; print(sys.executable)"],
        cwd=quest_root, timeout_s=60,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == sys.executable


@pytest.mark.asyncio
async def test_shared_interpreter_records_what_the_quest_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.execution import SharedInterpreterExecutor
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    exe = SharedInterpreterExecutor()
    quest_root = tmp_path / "shared-quest"
    await exe.setup(quest_root)
    # pytest is already installed in this interpreter: nothing to download.
    res = await exe.install(["pytest>=1"], quest_root=quest_root)
    assert res.returncode == 0, res.stderr

    lock_path = await exe.cleanup_after_success(quest_root)

    assert lock_path == quest_root / ".fi" / "requirements.lock.txt"
    body = lock_path.read_text(encoding="utf-8")
    assert f"pytest=={pytest.__version__}" in body
    assert not (quest_root / ".venv").exists()


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


# --- install(): one retry for a failed pip run --------------------------------


def _scripted_execute(exe: VenvExecutor, results: list[ExecutionResult]) -> list[list[str]]:
    """Replace ``exe.execute`` with one that returns ``results`` in order and
    records each command. A call past the end of ``results`` raises, so an
    unexpected extra attempt fails the test."""
    calls: list[list[str]] = []

    async def fake_execute(cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001
        calls.append(list(cmd))
        return results[len(calls) - 1]

    exe.execute = fake_execute  # type: ignore[method-assign]
    return calls


def _pip_result(rc: int, *, timed_out: bool = False) -> ExecutionResult:
    return ExecutionResult(
        returncode=rc,
        stdout="",
        stderr="" if rc == 0 else "ERROR: Could not build wheels for matplotlib",
        duration_s=1.0,
        timed_out=timed_out,
    )


@pytest.mark.asyncio
async def test_install_retries_a_failed_pip_run_once(tmp_path: Path) -> None:
    exe = VenvExecutor()
    calls = _scripted_execute(exe, [_pip_result(1), _pip_result(0)])

    result = await exe.install(["matplotlib"], quest_root=tmp_path)

    assert result.returncode == 0
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][-3:] == ["install", "--quiet", "matplotlib"]


@pytest.mark.asyncio
async def test_install_reports_the_second_failure(tmp_path: Path) -> None:
    exe = VenvExecutor()
    calls = _scripted_execute(exe, [_pip_result(1), _pip_result(2)])

    result = await exe.install(["matplotlib"], quest_root=tmp_path)

    assert result.returncode == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_install_does_not_retry_a_success_or_a_timeout(tmp_path: Path) -> None:
    exe = VenvExecutor()
    calls = _scripted_execute(exe, [_pip_result(0)])
    assert (await exe.install(["numpy"], quest_root=tmp_path)).returncode == 0
    assert len(calls) == 1

    exe = VenvExecutor()
    calls = _scripted_execute(exe, [_pip_result(-1, timed_out=True)])
    result = await exe.install(["numpy"], quest_root=tmp_path)
    assert result.timed_out is True
    assert len(calls) == 1


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
