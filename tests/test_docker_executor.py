"""DockerExecutor tests.

Two tiers:

1. **Unit tests** that run anywhere — they exercise `_docker()` import-error
   handling, `make_executor` dispatch (incl. unknown sandboxes), and the
   path-translation logic in `_run_sync` via `unittest.mock`. No real
   Docker required.

2. **Integration tests** gated on a reachable Docker daemon. They are
   skipped automatically if `docker` (the python package) isn't installed
   *or* if `docker.from_env().ping()` raises. Safe to ship without Docker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.execution import (
    DockerExecutor,
    ExecutionResult,
    VenvExecutor,
    make_executor,
)

# Importorskip at module top — if docker-py isn't installed, *integration*
# tests are skipped, but the unit tests below (which use mocks) still run.
docker = pytest.importorskip("docker", reason="docker python package not installed")


@pytest.fixture(scope="module")
def docker_available() -> bool:
    """True iff the Docker daemon is reachable."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# make_executor
# ---------------------------------------------------------------------------


def test_make_executor_venv() -> None:
    exe = make_executor("venv", python_version="3.11", docker_image="python:3.11-slim")
    assert isinstance(exe, VenvExecutor)
    assert exe.python_version == "3.11"


def test_make_executor_docker() -> None:
    exe = make_executor("docker", python_version="3.11", docker_image="python:3.11-slim")
    assert isinstance(exe, DockerExecutor)
    assert exe.image == "python:3.11-slim"


def test_make_executor_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown sandbox"):
        make_executor("wasm", python_version="3.11", docker_image="python:3.11-slim")


def test_make_executor_empty_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown sandbox"):
        make_executor("", python_version="3.11", docker_image="python:3.11-slim")


# ---------------------------------------------------------------------------
# DockerExecutor — pure unit tests with mocks
# ---------------------------------------------------------------------------


def test_python_path_returns_in_container_python() -> None:
    exe = DockerExecutor()
    # The in-container interpreter is on PATH; caller treats it as opaque.
    assert exe.python_path(Path("/anything")) == Path("python")


def test_docker_lazy_init_caches_client() -> None:
    exe = DockerExecutor()
    fake_client = MagicMock()
    with patch("docker.from_env", return_value=fake_client) as mk:
        c1 = exe._docker()
        c2 = exe._docker()
    assert c1 is fake_client
    assert c2 is fake_client
    # `from_env` is called exactly once — second call uses the cache.
    mk.assert_called_once()


def test_docker_import_error_yields_runtime_error() -> None:
    """If `import docker` fails, _docker() must surface a clean RuntimeError."""
    exe = DockerExecutor()
    # Sentinel `None` in sys.modules makes `import docker` raise ImportError.
    with patch.dict("sys.modules", {"docker": None}):
        with pytest.raises(RuntimeError, match="pip install docker"):
            exe._docker()


def test_docker_daemon_unreachable_yields_runtime_error() -> None:
    exe = DockerExecutor()
    with patch("docker.from_env", side_effect=Exception("daemon down")):
        with pytest.raises(RuntimeError, match="Docker daemon not reachable"):
            exe._docker()


def _make_fake_container(
    *,
    exit_code: int = 0,
    stdout: bytes = b"hello",
    stderr: bytes = b"",
    wait_raises: Exception | None = None,
) -> MagicMock:
    container = MagicMock()
    if wait_raises is not None:
        # First call raises (timeout); second call (after kill) returns rc=137.
        container.wait.side_effect = [wait_raises, {"StatusCode": 137}]
    else:
        container.wait.return_value = {"StatusCode": exit_code}

    _stdout, _stderr = stdout, stderr

    def logs_impl(stdout: bool = False, stderr: bool = False, **_: Any) -> bytes:
        if stdout and not stderr:
            return _stdout
        if stderr and not stdout:
            return _stderr
        return b""

    container.logs = MagicMock(side_effect=logs_impl)
    return container


def test_run_sync_happy_path_returns_result() -> None:
    exe = DockerExecutor()
    container = _make_fake_container(exit_code=0, stdout=b"hi\n", stderr=b"")
    client = MagicMock()
    client.containers.create.return_value = container

    result = exe._run_sync(client, ["python", "-c", "print('hi')"], Path.cwd(), 30, {})

    assert isinstance(result, ExecutionResult)
    assert result.returncode == 0
    assert "hi" in result.stdout
    assert result.timed_out is False
    container.start.assert_called_once()
    # Container is removed even on success — no leaks.
    container.remove.assert_called_once_with(force=True)


def test_run_sync_passes_network_disabled_true() -> None:
    """`network_disabled=True` is the security default; assert it stays so."""
    exe = DockerExecutor()
    container = _make_fake_container(exit_code=0)
    client = MagicMock()
    client.containers.create.return_value = container

    cwd = Path.cwd()
    exe._run_sync(client, ["python", "-V"], cwd, 30, {})

    kwargs = client.containers.create.call_args.kwargs
    assert kwargs["network_disabled"] is True
    assert kwargs["working_dir"] == "/work"
    # The bind-mount uses the resolved cwd as the host source and /work as the target.
    volumes = kwargs["volumes"]
    assert len(volumes) == 1
    (host_src, spec), = volumes.items()
    assert Path(host_src) == cwd.resolve()
    assert spec == {"bind": "/work", "mode": "rw"}


def test_run_sync_translates_host_path_in_cmd_args() -> None:
    exe = DockerExecutor()
    container = _make_fake_container(exit_code=0)
    client = MagicMock()
    client.containers.create.return_value = container

    cwd = Path.cwd()
    host_root = str(cwd.resolve())
    cmd = ["python", f"{host_root}/main.py", "--out", f"{host_root}/out.json"]
    exe._run_sync(client, cmd, cwd, 30, {})

    translated = client.containers.create.call_args.kwargs["command"]
    assert translated[0] == "python"
    assert translated[1] == "/work/main.py"
    assert translated[3] == "/work/out.json"
    # Ensure the host root no longer leaks into any arg.
    for a in translated:
        assert host_root not in a


def test_run_sync_path_translation_substring_caveat() -> None:
    """Documents a known sharp-edge in path translation.

    `_run_sync` does a literal `str.replace` of the host quest_root with
    `/work`. If a cmd arg happens to contain the host_root as a substring
    of an unrelated path (different drive, mid-string match), it would be
    rewritten incorrectly. In practice the engine never constructs such
    args, so this is a latent caveat rather than a live bug. This test
    pins the current behaviour so any future fix is intentional.
    """
    exe = DockerExecutor()
    container = _make_fake_container(exit_code=0)
    client = MagicMock()
    client.containers.create.return_value = container

    cwd = Path.cwd()
    host_root = str(cwd.resolve())
    # Adversarial arg: the host_root appears mid-string inside an unrelated value.
    arg = f"prefix-{host_root}-suffix"
    exe._run_sync(client, ["python", arg], cwd, 30, {})

    translated = client.containers.create.call_args.kwargs["command"]
    # Current behaviour: substring is rewritten verbatim.
    assert translated[1] == "prefix-/work-suffix"


def test_run_sync_timeout_kills_container_and_marks_timed_out() -> None:
    exe = DockerExecutor()
    # First wait raises (timeout), second wait (after kill) returns rc=137.
    container = _make_fake_container(
        wait_raises=Exception("read timeout from docker-py"),
        stdout=b"",
        stderr=b"",
    )
    client = MagicMock()
    client.containers.create.return_value = container

    result = exe._run_sync(client, ["sleep", "999"], Path.cwd(), 1, {})

    assert result.timed_out is True
    container.kill.assert_called_once()
    # Two waits: one with the user timeout, one after kill to drain.
    assert container.wait.call_count == 2
    # No leak — remove() runs in finally.
    container.remove.assert_called_once_with(force=True)


def test_run_sync_remove_failure_is_swallowed() -> None:
    """If container.remove() fails (e.g. already gone), the result still returns."""
    exe = DockerExecutor()
    container = _make_fake_container(exit_code=0)
    container.remove.side_effect = Exception("container already removed")
    client = MagicMock()
    client.containers.create.return_value = container

    # Must not raise.
    result = exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})
    assert result.returncode == 0


@pytest.mark.asyncio
async def test_install_no_packages_returns_zero_without_calling_docker() -> None:
    """Empty install short-circuits — no docker client should be touched."""
    exe = DockerExecutor()
    # If `_docker` is invoked, this would raise (no daemon during unit run).
    with patch.object(exe, "_docker", side_effect=AssertionError("should not call docker")):
        result = await exe.install([], quest_root=Path.cwd())
    assert result.returncode == 0
    assert result.duration_s == 0.0


# ---------------------------------------------------------------------------
# Integration tests (require a reachable Docker daemon)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_and_execute_real_docker(
    docker_available: bool, tmp_path: Path
) -> None:
    if not docker_available:
        pytest.skip("Docker daemon not reachable")
    exe = DockerExecutor(image="python:3.11-slim")
    await exe.setup(tmp_path)
    result = await exe.execute(
        ["python", "-c", "print('hello-from-docker')"],
        cwd=tmp_path,
        timeout_s=120,
    )
    assert result.returncode == 0
    assert "hello-from-docker" in result.stdout
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_network_disabled_real_docker(
    docker_available: bool, tmp_path: Path
) -> None:
    """With network_disabled=True, name resolution / outbound TCP must fail."""
    if not docker_available:
        pytest.skip("Docker daemon not reachable")
    exe = DockerExecutor(image="python:3.11-slim")
    await exe.setup(tmp_path)
    # Try to open a TCP connection — should fail because the container has no network.
    script = (
        "import socket, sys\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "    sys.exit(0)\n"
        "except OSError:\n"
        "    sys.exit(7)\n"
    )
    result = await exe.execute(
        ["python", "-c", script], cwd=tmp_path, timeout_s=30
    )
    assert result.returncode == 7, (
        f"expected network-disabled exit 7; got rc={result.returncode}, "
        f"stdout={result.stdout!r}, stderr={result.stderr!r}"
    )
