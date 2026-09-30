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

# Soft import: keep all *unit* tests in this file runnable without docker-py.
# Integration tests use the `docker_module` fixture below which skips
# explicitly when the package or the daemon is unavailable.
try:
    import docker as _docker_module  # type: ignore[import-not-found]
except ImportError:
    _docker_module = None


@pytest.fixture(scope="module")
def docker_module():
    """Yield the docker-py module; skip if it's not installed."""
    if _docker_module is None:
        pytest.skip("docker python package not installed")
    return _docker_module


@pytest.fixture(scope="module")
def docker_available(docker_module) -> bool:
    """True iff the Docker daemon is reachable AND can serve the Linux
    container image these tests need. Skips when docker-py is missing,
    when the daemon is not reachable, or when the daemon is in
    Windows-containers mode (GitHub-hosted ``windows-latest`` runners
    have a Docker daemon, but it defaults to Windows containers and
    ``python:3.11-slim`` is a Linux image — pulling it 404s with
    "No such image"). Pinging is necessary but not sufficient: the
    Windows-CI failure mode was "daemon up + image pull fails", so
    we additionally require the daemon to report Linux-server OS."""
    try:
        client = docker_module.from_env()
        client.ping()
    except Exception:
        return False
    try:
        info = client.info()
    except Exception:
        return False
    # Daemon's reported "OSType" — "linux" or "windows". Linux images
    # are only servable by a Linux-mode daemon. (On Linux hosts this
    # is always "linux"; on Docker Desktop / Windows it switches
    # depending on the user's container-mode toggle.)
    return str(info.get("OSType", "")).lower() == "linux"


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
    # The folder name does not end at "-suffix", so this is some other path
    # (e.g. a sibling folder) and is left alone.
    assert translated[1] == arg


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


# ---------------------------------------------------------------------------
# Resource limits, dropped privileges, non-root user
# ---------------------------------------------------------------------------


def _client_with(container: MagicMock) -> MagicMock:
    client = MagicMock()
    client.containers.create.return_value = container
    return client


def test_run_sync_passes_default_limits_and_drops_privileges() -> None:
    """Every container is capped (memory, CPU, processes), has no Linux
    capabilities, cannot gain privileges, and runs as a named user."""
    exe = DockerExecutor()
    exe._user = "1000:1000"  # as setup() would have resolved it
    client = _client_with(_make_fake_container(exit_code=0))

    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})

    kw = client.containers.create.call_args.kwargs
    assert kw["mem_limit"] == "4g"
    # Swap equal to memory: going over the limit stops the run instead of
    # swapping it slowly to a halt.
    assert kw["memswap_limit"] == "4g"
    assert kw["nano_cpus"] == 2_000_000_000
    assert kw["pids_limit"] == 1024
    assert kw["cap_drop"] == ["ALL"]
    assert kw["security_opt"] == ["no-new-privileges:true"]
    assert kw["user"] == "1000:1000"
    assert kw["network_disabled"] is True
    assert "oom_kill_disable" not in kw


def test_run_sync_non_root_gets_a_writable_home() -> None:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))

    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {"HOME": "C:\\Users\\me", "X": "1"})

    env = client.containers.create.call_args.kwargs["environment"]
    assert env["HOME"] == "/tmp"
    assert env["X"] == "1"


def test_limits_flow_from_config_through_make_executor() -> None:
    from core.config import ExecutionConfig
    from core.execution import DockerLimits

    cfg = ExecutionConfig(
        sandbox="docker", docker_memory_gb=16, docker_cpus=0.5, docker_max_processes=64,
    )
    exe = make_executor(
        "docker", python_version="3.11", docker_image="python:3.11-slim",
        docker_limits=DockerLimits.from_config(cfg),
    )
    assert isinstance(exe, DockerExecutor)
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})

    kw = client.containers.create.call_args.kwargs
    assert kw["mem_limit"] == "16g"
    assert kw["memswap_limit"] == "16g"
    assert kw["nano_cpus"] == 500_000_000
    assert kw["pids_limit"] == 64


def test_fractional_memory_limit_is_given_in_megabytes() -> None:
    from core.execution import DockerLimits

    exe = DockerExecutor(limits=DockerLimits(memory_gb=1.5))
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})
    assert client.containers.create.call_args.kwargs["mem_limit"] == "1536m"


def test_cpu_limit_is_capped_at_what_docker_has() -> None:
    """Docker refuses a container asked for more CPUs than it has; a
    one-CPU Docker VM must not make every experiment fail."""
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    client.info.return_value = {"NCPU": 1, "OperatingSystem": "Docker Desktop", "SecurityOptions": []}
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})
    assert client.containers.create.call_args.kwargs["nano_cpus"] == 1_000_000_000


def test_config_rejects_non_positive_limits() -> None:
    from pydantic import ValidationError

    from core.config import ExecutionConfig

    for bad in ({"docker_memory_gb": 0}, {"docker_cpus": -1}, {"docker_max_processes": 0}):
        with pytest.raises(ValidationError):
            ExecutionConfig(**bad)


def test_config_defaults() -> None:
    from core.config import ExecutionConfig

    cfg = ExecutionConfig()
    assert cfg.docker_memory_gb == 4
    assert cfg.docker_cpus == 2
    assert cfg.docker_max_processes == 1024


def _oom_container(*, oom: bool, exit_code: int = 137, stderr: bytes = b"") -> MagicMock:
    container = _make_fake_container(exit_code=exit_code, stderr=stderr)
    container.attrs = {"State": {"OOMKilled": oom, "ExitCode": exit_code}}
    return container


def test_oom_killed_run_says_so_in_plain_words() -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    exe._user = "1000:1000"
    container = _oom_container(oom=True)
    client = _client_with(container)

    result = exe._run_sync(client, ["python", "big.py"], Path.cwd(), 30, {})

    assert result.returncode == 137
    last = result.stderr.strip().splitlines()[-1]
    assert last.startswith("[FI] ")
    assert "more than the 4 GB memory limit" in last
    assert "execution.docker_memory_gb" in last
    container.reload.assert_called()
    # The same sentence goes to run.log (the quest's logger).
    logged = " ".join(str(a) for c in logger.warning.call_args_list for a in c.args)
    assert "execution.docker_memory_gb" in logged
    container.remove.assert_called_once_with(force=True)


def test_exit_137_without_the_oom_flag_names_the_memory_limit_as_likely() -> None:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_oom_container(oom=False, exit_code=137))

    result = exe._run_sync(client, ["python", "big.py"], Path.cwd(), 30, {})

    last = result.stderr.strip().splitlines()[-1]
    assert last.startswith("[FI] ")
    assert "4 GB memory limit" in last
    assert "execution.docker_memory_gb" in last


def test_plain_mock_attrs_are_not_read_as_an_oom() -> None:
    """A MagicMock's attrs is truthy everywhere; only a real True counts."""
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=1, stderr=b"Traceback\nValueError: x\n"))

    result = exe._run_sync(client, ["python", "x.py"], Path.cwd(), 30, {})

    assert "[FI]" not in result.stderr
    assert result.stderr.endswith("ValueError: x\n")


def test_timeout_is_not_reported_as_a_memory_limit() -> None:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    container = _make_fake_container(wait_raises=Exception("read timeout"), stdout=b"", stderr=b"")
    container.attrs = {"State": {"OOMKilled": False, "ExitCode": 137}}
    client = _client_with(container)

    result = exe._run_sync(client, ["sleep", "999"], Path.cwd(), 1, {})

    assert result.timed_out is True
    assert "memory limit" not in result.stderr


def test_process_limit_hit_says_so_in_plain_words() -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    exe._user = "1000:1000"
    stderr = b"Traceback (most recent call last):\nRuntimeError: can't start new thread\n"
    client = _client_with(_oom_container(oom=False, exit_code=1, stderr=stderr))

    result = exe._run_sync(client, ["python", "pool.py"], Path.cwd(), 30, {})

    last = result.stderr.strip().splitlines()[-1]
    assert last.startswith("[FI] ")
    assert "1024" in last
    assert "execution.docker_max_processes" in last
    logger.warning.assert_called()


def test_user_candidates_on_windows_host_fixed_non_root_then_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker Desktop on Windows/macOS: the bind mount maps ownership, so any
    fixed non-root id can write the quest folder."""
    monkeypatch.setattr("core.execution.sys.platform", "win32")
    exe = DockerExecutor()
    client = MagicMock()
    client.info.return_value = {"OperatingSystem": "Docker Desktop", "SecurityOptions": []}
    assert exe._user_candidates(client, Path.cwd()) == ["1000:1000", ""]


def test_user_candidates_on_linux_quest_folder_owner_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("core.execution.sys.platform", "linux")
    monkeypatch.setattr("core.execution._owner_of", lambda p: (1234, 5678))
    exe = DockerExecutor()
    client = MagicMock()
    client.info.return_value = {"OperatingSystem": "Ubuntu 24.04", "SecurityOptions": ["name=seccomp"]}
    assert exe._user_candidates(client, tmp_path) == ["1234:5678", ""]


def test_user_candidates_rootless_docker_only_container_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Rootless Docker: the container's root IS the unprivileged host user;
    any other id maps to a sub-id that cannot write the quest folder."""
    monkeypatch.setattr("core.execution.sys.platform", "linux")
    monkeypatch.setattr("core.execution._owner_of", lambda p: (1000, 1000))
    exe = DockerExecutor()
    client = MagicMock()
    client.info.return_value = {
        "OperatingSystem": "Ubuntu 24.04",
        "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless", "name=cgroupns"],
    }
    assert exe._user_candidates(client, tmp_path) == [""]


def test_user_candidates_docker_desktop_on_linux_tries_the_owner_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Docker Desktop reports the same name under WSL integration (real
    ownership kept, the owner can write) and on Linux (owner mapped to root):
    the owner goes first and the write check decides."""
    monkeypatch.setattr("core.execution.sys.platform", "linux")
    monkeypatch.setattr("core.execution._owner_of", lambda p: (1000, 1000))
    exe = DockerExecutor()
    client = MagicMock()
    client.info.return_value = {"OperatingSystem": "Docker Desktop", "SecurityOptions": []}
    assert exe._user_candidates(client, tmp_path) == ["1000:1000", ""]


def test_user_candidates_root_owned_folder_on_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """FI itself running as root: only root can write the folder, so the
    container runs as root, with every privilege still dropped."""
    monkeypatch.setattr("core.execution.sys.platform", "linux")
    monkeypatch.setattr("core.execution._owner_of", lambda p: (0, 0))
    exe = DockerExecutor()
    client = MagicMock()
    client.info.return_value = {"OperatingSystem": "Ubuntu 24.04", "SecurityOptions": []}
    assert exe._user_candidates(client, tmp_path) == [""]


def test_root_user_does_not_override_home() -> None:
    exe = DockerExecutor()
    exe._user = ""  # container root (rootless Docker)
    client = _client_with(_make_fake_container(exit_code=0))
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {"A": "b"})
    kw = client.containers.create.call_args.kwargs
    assert kw["environment"]["A"] == "b"
    assert "HOME" not in kw["environment"]
    # docker-py treats user=None/"" as the image default; cap_drop still applies.
    assert not kw.get("user")
    assert kw["cap_drop"] == ["ALL"]


@pytest.mark.asyncio
async def test_setup_probes_the_user_and_falls_back_to_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """setup() checks the chosen user can write the quest folder; when it
    cannot, experiments run as the container's root and run.log says why."""
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)  # CI may run as root
    probe = _make_fake_container(exit_code=1, stderr=b"PermissionError: [Errno 13]")
    client = _client_with(probe)
    exe._client = client

    await exe.setup(tmp_path)

    assert exe._user == ""
    first, second = client.containers.create.call_args_list
    assert first.kwargs["user"] == "1000:1000"
    assert first.kwargs["network_disabled"] is True
    assert first.kwargs["cap_drop"] == ["ALL"]
    assert first.kwargs["mem_limit"] == "4g"
    assert first.kwargs["volumes"] == {str(tmp_path.resolve()): {"bind": "/work", "mode": "rw"}}
    # The root fallback is checked too, still with every privilege dropped.
    assert second.kwargs["user"] is None
    assert second.kwargs["cap_drop"] == ["ALL"]
    assert probe.remove.call_count == 2
    # Neither can write: run.log says the results cannot be saved.
    warned = " ".join(str(a) for c in logger.warning.call_args_list for a in c.args)
    assert "no user can write the quest folder" in warned


@pytest.mark.asyncio
async def test_setup_root_fallback_that_works_warns_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)  # CI may run as root
    fails, works = _make_fake_container(exit_code=1), _make_fake_container(exit_code=0)
    client = MagicMock()
    client.containers.create.side_effect = [fails, works]
    exe._client = client

    await exe.setup(tmp_path)

    assert exe._user == ""
    assert logger.warning.call_count == 1


@pytest.mark.asyncio
async def test_setup_probe_success_keeps_non_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)  # CI may run as root
    client = _client_with(_make_fake_container(exit_code=0))
    exe._client = client

    await exe.setup(tmp_path)
    await exe.setup(tmp_path)  # resolved once, not probed again

    assert exe._user == "1000:1000"
    assert client.containers.create.call_count == 1
    logger.warning.assert_not_called()


def test_numerical_libraries_are_told_the_cpu_limit() -> None:
    """A CPU cap does not change the core count libraries see; each one is
    told the cap, unless the environment already says."""
    from core.execution import DockerLimits

    exe = DockerExecutor(limits=DockerLimits(cpus=2.5))
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {"MKL_NUM_THREADS": "1"})
    env = client.containers.create.call_args.kwargs["environment"]
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["NUMEXPR_NUM_THREADS"] == "3"
    assert env["PYTHON_CPU_COUNT"] == "3"
    assert env["MKL_NUM_THREADS"] == "1"  # the caller's own choice stays


def test_limits_docker_cannot_apply_are_left_out() -> None:
    """Without the kernel's CPU quota Docker refuses nano_cpus outright; the
    container must still start, with the other limits."""
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    client.info.return_value = {"CpuCfsQuota": False, "SwapLimit": False, "SecurityOptions": []}
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {})
    kw = client.containers.create.call_args.kwargs
    assert "nano_cpus" not in kw
    assert "memswap_limit" not in kw
    assert kw["mem_limit"] == "4g"
    assert kw["pids_limit"] == 1024


def test_oom_of_one_process_in_a_run_that_still_exited_zero() -> None:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_oom_container(oom=True, exit_code=0))
    result = exe._run_sync(client, ["python", "pool.py"], Path.cwd(), 30, {})
    last = result.stderr.strip().splitlines()[-1]
    assert last.startswith("[FI] ")
    assert "one of its processes" in last
    assert "execution.docker_memory_gb" in last


def test_process_limit_words_far_above_the_error_are_not_blamed() -> None:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    stderr = ("OpenBLAS warning: pthread_create failed, retrying\n" + "x\n" * 30 + "ValueError: bad shape\n").encode()
    client = _client_with(_make_fake_container(exit_code=1, stderr=stderr))
    result = exe._run_sync(client, ["python", "x.py"], Path.cwd(), 30, {})
    assert "[FI]" not in result.stderr


def test_config_rejects_limits_docker_refuses() -> None:
    from pydantic import ValidationError

    from core.config import ExecutionConfig

    with pytest.raises(ValidationError):
        ExecutionConfig(docker_memory_gb=0.001)
    with pytest.raises(ValidationError):
        ExecutionConfig(docker_cpus=0.001)
    assert ExecutionConfig(docker_cpus=0.5).docker_cpus == 0.5


@pytest.mark.asyncio
async def test_setup_write_check_error_keeps_the_first_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)  # CI may run as root
    client = MagicMock()
    client.containers.create.side_effect = RuntimeError("image has no python")
    exe._client = client

    await exe.setup(tmp_path)

    assert exe._user == "1000:1000"
    assert client.containers.create.call_count == 1
    warned = " ".join(str(a) for c in logger.warning.call_args_list for a in c.args)
    assert "image has no python" in warned


@pytest.mark.asyncio
async def test_setup_says_when_the_cpu_limit_is_lowered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from core.execution import DockerLimits

    logger = MagicMock()
    exe = DockerExecutor(limits=DockerLimits(cpus=8), log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)  # CI may run as root
    client = _client_with(_make_fake_container(exit_code=0))
    client.info.return_value = {"NCPU": 4, "SecurityOptions": []}
    exe._client = client

    await exe.setup(tmp_path)

    warned = " ".join(str(a) for c in logger.warning.call_args_list for a in c.args)
    assert "execution.docker_cpus" in warned
    assert client.containers.create.call_args.kwargs["nano_cpus"] == 4_000_000_000


def test_root_owned_folders_left_by_an_older_run_are_named(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "raw" / "seed0").mkdir(parents=True)
    (tmp_path / "code").mkdir()
    real_scandir = __import__("os").scandir

    class _Entry:
        def __init__(self, e: Any) -> None:
            self._e, self.path, self.name = e, e.path, e.name

        def is_dir(self, follow_symlinks: bool = True) -> bool:
            return self._e.is_dir(follow_symlinks=follow_symlinks)

        def stat(self, follow_symlinks: bool = True) -> Any:
            return MagicMock(st_uid=0 if self.name == "seed0" else 1000)

    monkeypatch.setattr("core.execution.os.scandir", lambda p: [_Entry(e) for e in real_scandir(p)])
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    exe._warn_root_owned(tmp_path, "1000:1000")
    warned = " ".join(str(a) for c in logger.warning.call_args_list for a in c.args)
    assert "seed0" in warned
    assert "chown -R 1000:1000" in " ".join(
        str(c.args[0]) % c.args[1:] for c in logger.warning.call_args_list
    )


@pytest.mark.asyncio
async def test_setup_check_error_after_a_failed_check_does_not_pick_the_failed_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    exe = DockerExecutor(log=MagicMock())
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)
    client = MagicMock()
    client.containers.create.side_effect = [_make_fake_container(exit_code=1), RuntimeError("daemon hiccup")]
    exe._client = client

    await exe.setup(tmp_path)

    assert exe._user == ""


@pytest.mark.asyncio
async def test_setup_names_each_limit_docker_cannot_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    logger = MagicMock()
    exe = DockerExecutor(log=logger)
    monkeypatch.setattr(exe, "_user_candidates", lambda client, root: ["1000:1000", ""])
    monkeypatch.setattr(exe, "_warn_root_owned", lambda *a: None)
    client = _client_with(_make_fake_container(exit_code=0))
    client.info.return_value = {"MemoryLimit": False, "PidsLimit": False, "CpuCfsPeriod": False, "SecurityOptions": []}
    exe._client = client

    await exe.setup(tmp_path)

    def text(calls: Any) -> str:
        return " ".join(str(c.args[0]) % c.args[1:] for c in calls)

    warned = text(logger.warning.call_args_list)
    assert "cannot limit memory" in warned
    assert "cannot limit the number of processes" in warned
    assert "cannot limit CPU use" in warned
    assert "nano_cpus" not in client.containers.create.call_args.kwargs
    assert "no resource limits" in text(logger.info.call_args_list)


def test_host_thread_settings_above_the_cpu_limit_are_lowered() -> None:
    """An HPC shell's OMP_NUM_THREADS=64 would crowd a 2-CPU container."""
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = _client_with(_make_fake_container(exit_code=0))
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, {"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "x"})
    env = client.containers.create.call_args.kwargs["environment"]
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "2"


def test_host_environment_is_made_to_fit_the_container(tmp_path: Path) -> None:
    """The engine passes the host's whole environment: its PATH would hide the
    image's python, and paths in the quest folder must read as /work."""
    import os as _os

    exe = DockerExecutor()
    exe._user = ""
    client = _client_with(_make_fake_container(exit_code=0))
    root = tmp_path.resolve()
    boot = str(root / ".fi" / "boot")
    env = {
        "PATH": r"C:\Windows\system32" if _os.sep == "\\" else "/home/me/bin",
        "HOME": "/home/me",
        "PYTHONPATH": _os.pathsep.join([boot, "/fi-skills/lib", "C:\\host\\lib"]),
        "FI_INPUT_DIR": str(root / "inputs" / "examples"),
        "OTHER": "kept",
    }
    exe._run_sync(client, ["python", str(root / "code" / "experiment.py")], root, 30, env)
    kw = client.containers.create.call_args.kwargs
    out = kw["environment"]
    assert "PATH" not in out and "HOME" not in out
    assert out["PYTHONPATH"] == "/work/.fi/boot:/fi-skills/lib"
    assert out["FI_INPUT_DIR"] == "/work/inputs/examples"
    assert out["OTHER"] == "kept"
    assert kw["command"] == ["python", "/work/code/experiment.py"]


def test_windows_path_under_the_quest_folder_uses_forward_slashes() -> None:
    from core.execution import _to_container

    root = r"C:\q\abc"
    assert _to_container(r"C:\q\abc\code\experiment.py", root, windows=True) == "/work/code/experiment.py"
    assert _to_container(r"C:\q\abc\my code\e x.py", root, windows=True) == "/work/my code/e x.py"
    assert _to_container(root, root, windows=True) == "/work"
    assert _to_container(r"x=C:\q\abc", root, windows=True) == "x=/work"
    assert _to_container(r"--out=C:\q\abc\figs\a.png", root, windows=True) == "--out=/work/figs/a.png"
    assert _to_container(r"c:/Q/ABC/code/e.py", root, windows=True) == "/work/code/e.py"
    assert _to_container(r"a;C:\q\abc\x;b", root, windows=True) == "a;/work/x;b"
    # A sibling folder whose name only starts the same is not the quest.
    assert _to_container(r"C:\q\abcd\x", root, windows=True) == r"C:\q\abcd\x"
    assert _to_container(r"C:\q\abc-old", root, windows=True) == r"C:\q\abc-old"


def test_linux_path_translation_stops_at_the_folder_name() -> None:
    from core.execution import _to_container

    root = "/q/abc"
    assert _to_container("/q/abc/code/e.py", root, windows=False) == "/work/code/e.py"
    assert _to_container("/q/abc", root, windows=False) == "/work"
    assert _to_container("/q/abcd/x", root, windows=False) == "/q/abcd/x"
    assert _to_container("/q/abc-old", root, windows=False) == "/q/abc-old"
    assert _to_container("--out=/q/abc/f.png", root, windows=False) == "--out=/work/f.png"
    # Backslashes are ordinary file-name characters on Linux: left alone.
    assert _to_container("/q/abc/a\\b", root, windows=False) == "/work/a\\b"


def test_host_python_and_system_paths_are_not_passed_in() -> None:
    exe = DockerExecutor()
    exe._user = ""
    client = _client_with(_make_fake_container(exit_code=0))
    env = {
        "PYTHONHOME": "C:\\Python311", "Path": "C:\\Windows", "LD_PRELOAD": "/x.so",
        "PYTHONPYCACHEPREFIX": "C:\\cache", "TEMP": "C:\\Temp", "FI_REPLICATE_SEED": "3",
        "PYTHONPATH": "/home/me/site" + __import__("os").pathsep + "/usr/lib/python3/dist-packages",
    }
    exe._run_sync(client, ["python", "-V"], Path.cwd(), 30, env)
    out = client.containers.create.call_args.kwargs["environment"]
    for gone in ("PYTHONHOME", "Path", "LD_PRELOAD", "PYTHONPYCACHEPREFIX", "TEMP", "PYTHONPATH"):
        assert gone not in out, gone
    assert out["FI_REPLICATE_SEED"] == "3"


def test_engine_passes_the_configured_limits_and_its_log(tmp_path: Path) -> None:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
    )
    from core.engine import Engine

    cfg = Config(
        topic="docker limits", title="docker-limits",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(sandbox="docker", docker_memory_gb=8, docker_cpus=1, docker_max_processes=99),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    try:
        assert isinstance(engine.executor, DockerExecutor)
        assert (engine.executor.limits.memory_gb, engine.executor.limits.cpus,
                engine.executor.limits.max_processes) == (8, 1, 99)
        assert engine.executor._qlog is engine._log
    finally:
        from core.engine import _close_quest_logger
        _close_quest_logger(engine.quest_id)


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
