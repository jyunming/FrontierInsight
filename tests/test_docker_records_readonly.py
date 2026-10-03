"""FI's own records (.fi/, needs/) are read-only inside the Docker container.

Mocked docker-py for the mounts and the run.log line; one real-daemon test
(skipped when no Docker daemon is reachable) that a script really cannot write
into the records while it can write its results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.execution import DockerExecutor

try:
    import docker as _docker_module  # type: ignore[import-not-found]
except ImportError:
    _docker_module = None


def _container(exit_code: int = 0, stderr: bytes = b"") -> MagicMock:
    c = MagicMock()
    c.wait.return_value = {"StatusCode": exit_code}

    def logs(stdout: bool = False, stderr: bool = False, **_: Any) -> bytes:
        return b"" if stdout and not stderr else (_err if stderr and not stdout else b"")

    _err = stderr
    c.logs = MagicMock(side_effect=logs)
    return c


def _binds(tmp_path: Path, stderr: bytes = b"", exit_code: int = 0) -> dict[str, str]:
    exe = DockerExecutor()
    exe._user = "1000:1000"
    client = MagicMock()
    client.containers.create.return_value = _container(exit_code, stderr)
    exe._run_sync(client, ["python", "-V"], tmp_path, 30, {})
    vols = client.containers.create.call_args.kwargs["volumes"]
    return {spec["bind"]: spec["mode"] for spec in vols.values()}


def test_records_are_read_only_and_the_scripts_own_places_stay_writable(tmp_path: Path) -> None:
    (tmp_path / ".fi" / "trials").mkdir(parents=True)
    (tmp_path / ".fi" / "trials" / "run.json").write_text("{}", encoding="utf-8")
    binds = _binds(tmp_path)
    assert binds["/work"] == "rw"
    assert binds["/work/.fi"] == "ro"
    assert binds["/work/needs"] == "ro"
    # where FI's harness has the script write its rows stays writable ...
    assert binds["/work/.fi/trials"] == "rw"
    assert binds["/work/.fi/optimisation"] == "rw"
    # ... but a record FI keeps there is read-only again; one not written yet is not mounted
    assert binds["/work/.fi/trials/run.json"] == "ro"
    assert "/work/.fi/trials/cluster.json" not in binds
    assert (tmp_path / "needs").is_dir()


def test_a_refused_write_into_the_records_is_named_in_run_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        _binds(tmp_path, b"OSError: [Errno 30] Read-only file system: '/work/.fi/audit.jsonl'", 1)
    assert any("own records" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _binds(tmp_path, b"OSError: [Errno 30] Read-only file system: '/proc/x'", 1)
    assert not any("own records" in r.getMessage() for r in caplog.records)


@pytest.fixture
def docker_up() -> bool:
    if _docker_module is None:
        return False
    try:
        _docker_module.from_env().ping()
        return True
    except Exception:
        return False


@pytest.mark.asyncio
async def test_records_cannot_be_written_from_a_real_container(docker_up: bool, tmp_path: Path) -> None:
    if not docker_up:
        pytest.skip("Docker daemon not reachable")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "audit.jsonl").write_text("{}\n", encoding="utf-8")
    exe = DockerExecutor(image="python:3.11-slim")
    await exe.setup(tmp_path)
    script = (
        "def w(p):\n"
        "    try:\n"
        "        open(p, 'a').write('x'); return 'wrote'\n"
        "    except OSError:\n"
        "        return 'refused'\n"
        "print(w('/work/.fi/audit.jsonl'), w('/work/.fi/new.json'), w('/work/needs/N.json'),\n"
        "      w('/work/.fi/trials/cell0.out.jsonl'), w('/work/results.json'))\n"
    )
    result = await exe.execute(["python", "-c", script], cwd=tmp_path, timeout_s=120)
    assert result.stdout.split() == ["refused", "refused", "refused", "wrote", "wrote"], (result.stdout, result.stderr)
    assert (tmp_path / ".fi" / "audit.jsonl").read_text(encoding="utf-8") == "{}\n"
