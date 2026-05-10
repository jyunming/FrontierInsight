"""Per-quest code execution.

`VenvExecutor` is the default — each quest gets `<quest_root>/.venv/` and
agent-generated code runs as a child process of the FI engine. Cross-
platform: the venv's Python lives at `Scripts/python.exe` on Windows and
`bin/python` on POSIX.

`DockerExecutor` is the opt-in (Phase D) that builds an ephemeral image,
mounts the quest_root, and runs commands inside the container.

Both expose the same async `execute(cmd, cwd, timeout_s) -> ExecutionResult`.
"""

from __future__ import annotations

import asyncio
import sys
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


class Executor(Protocol):
    async def setup(self, quest_root: Path) -> None: ...
    async def execute(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult: ...
    async def install(self, packages: Iterable[str], *, quest_root: Path) -> ExecutionResult: ...
    def python_path(self, quest_root: Path) -> Path: ...


class VenvExecutor:
    """Runs commands in a per-quest venv; no sandbox.

    Suitable for personal / trusted-topic use. For untrusted code, prefer
    `DockerExecutor` (Phase D).
    """

    def __init__(self, *, python_version: str = "3.11") -> None:
        self.python_version = python_version

    async def setup(self, quest_root: Path) -> None:
        quest_root.mkdir(parents=True, exist_ok=True)
        venv_dir = quest_root / ".venv"
        if (venv_dir / "pyvenv.cfg").exists():
            return
        # `venv.EnvBuilder` is sync; offload so we don't block the loop.
        await asyncio.to_thread(
            _build_venv, venv_dir, with_pip=True
        )

    def python_path(self, quest_root: Path) -> Path:
        venv_dir = quest_root / ".venv"
        if sys.platform.startswith("win"):
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    async def install(
        self, packages: Iterable[str], *, quest_root: Path
    ) -> ExecutionResult:
        pkgs = list(packages)
        if not pkgs:
            return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.0)
        py = self.python_path(quest_root)
        cmd = [str(py), "-m", "pip", "install", "--quiet", *pkgs]
        return await self.execute(cmd, cwd=quest_root, timeout_s=600)

    async def execute(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
            timed_out = False
        except asyncio.TimeoutError:
            proc.kill()
            stdout, stderr = await proc.communicate()
            timed_out = True
        return ExecutionResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
        )


def _build_venv(venv_dir: Path, *, with_pip: bool) -> None:
    builder = venv.EnvBuilder(with_pip=with_pip, clear=False, upgrade_deps=False)
    builder.create(str(venv_dir))


class DockerExecutor:
    """Sandboxed executor — runs every command inside an ephemeral Docker
    container with the quest_root bind-mounted at /work.

    The same Executor protocol as VenvExecutor: `setup` ensures the image
    is available, `install` runs `pip install ...` inside the container,
    `execute` runs the given command. `python_path` returns the
    in-container interpreter path (caller treats it as opaque).
    """

    def __init__(self, *, image: str = "python:3.11-slim") -> None:
        self.image = image
        self._client: object | None = None  # lazy docker client

    def _docker(self) -> object:
        if self._client is not None:
            return self._client
        try:
            import docker  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "execution.sandbox=docker requires `pip install docker`"
            ) from e
        try:
            self._client = docker.from_env()
        except Exception as e:
            raise RuntimeError(
                "Docker daemon not reachable. Install Docker Desktop "
                "(Windows/macOS) or run `dockerd` (Linux), then retry."
            ) from e
        return self._client

    async def setup(self, quest_root: Path) -> None:
        quest_root.mkdir(parents=True, exist_ok=True)
        client = self._docker()
        # Pull the image if not present. docker-py is sync; offload.

        def _ensure() -> None:
            try:
                client.images.get(self.image)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
            client.images.pull(self.image)  # type: ignore[attr-defined]

        await asyncio.to_thread(_ensure)

    def python_path(self, quest_root: Path) -> Path:
        # Inside the container the Python interpreter is on PATH as `python`.
        return Path("python")

    async def install(
        self, packages: Iterable[str], *, quest_root: Path
    ) -> ExecutionResult:
        pkgs = list(packages)
        if not pkgs:
            return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.0)
        return await self.execute(
            ["python", "-m", "pip", "install", "--quiet", *pkgs],
            cwd=quest_root,
            timeout_s=600,
        )

    async def execute(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        timeout_s: int,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        client = self._docker()
        return await asyncio.to_thread(
            self._run_sync, client, cmd, cwd, timeout_s, env or {}
        )

    def _run_sync(
        self,
        client: object,
        cmd: list[str],
        cwd: Path,
        timeout_s: int,
        env: dict[str, str],
    ) -> ExecutionResult:
        start = time.monotonic()
        # Translate the quest-root path inside the container to /work, and
        # rewrite any cmd args that contain the host quest_root prefix.
        cwd_abs = cwd.resolve()
        host_root = cwd_abs
        translated: list[str] = []
        for a in cmd:
            translated.append(a.replace(str(host_root), "/work"))

        container = client.containers.create(  # type: ignore[attr-defined]
            self.image,
            command=translated,
            working_dir="/work",
            volumes={str(host_root): {"bind": "/work", "mode": "rw"}},
            environment=env,
            network_disabled=True,  # no network from the experiment by default
            detach=True,
        )
        try:
            container.start()
            try:
                exit_status = container.wait(timeout=timeout_s)
                timed_out = False
                rc = int(exit_status.get("StatusCode", -1))
            except Exception:
                container.kill()
                exit_status = container.wait(timeout=10)
                timed_out = True
                rc = int(exit_status.get("StatusCode", -1)) if isinstance(exit_status, dict) else -1
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            return ExecutionResult(
                returncode=rc,
                stdout=stdout,
                stderr=stderr,
                duration_s=time.monotonic() - start,
                timed_out=timed_out,
            )
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass


def make_executor(sandbox: str, *, python_version: str, docker_image: str) -> Executor:
    if sandbox == "venv":
        return VenvExecutor(python_version=python_version)
    if sandbox == "docker":
        return DockerExecutor(image=docker_image)
    raise ValueError(f"unknown sandbox: {sandbox!r}")
