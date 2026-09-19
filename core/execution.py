"""Per-quest code execution.

`VenvExecutor` is the default — each quest gets `<quest_root>/.venv/` and
agent-generated code runs as a child process of the FI engine. Cross-
platform: the venv's Python lives at `Scripts/python.exe` on Windows and
`bin/python` on POSIX.

`DockerExecutor` is the opt-in that builds an ephemeral image,
mounts the quest_root, and runs commands inside the container.

Both expose the same async `execute(cmd, cwd, timeout_s) -> ExecutionResult`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import sys
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

_log = logging.getLogger("frontier_insight.execution")


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
    async def cleanup_after_success(self, quest_root: Path) -> Path | None:
        """Optional. Called when the quest reaches a terminal success
        state. Implementations should free any heavyweight resources
        (e.g. the per-quest venv) while preserving a reproducibility
        artifact (e.g. ``requirements.lock.txt``). A no-op by default;
        ``DockerExecutor`` has nothing to clean. Returns the path to
        the produced reproducibility artifact, or ``None`` if there
        was nothing to clean up or freeze."""
        return None


class VenvExecutor:
    """Runs commands in a per-quest venv; no sandbox.

    Suitable for personal / trusted-topic use. For untrusted code, prefer
    `DockerExecutor`.
    """

    def __init__(
        self, *, python_version: str = "3.11", system_site_packages: bool = True,
    ) -> None:
        self.python_version = python_version
        self.system_site_packages = system_site_packages

    async def setup(self, quest_root: Path) -> None:
        quest_root.mkdir(parents=True, exist_ok=True)
        venv_dir = quest_root / ".venv"
        # Proactive Windows MAX_PATH warning: once native extensions
        # (PIL/matplotlib .pyd files, ~55 chars under site-packages) are
        # installed, a venv path already near 260 chars overflows and their DLL
        # load fails — silently killing figure generation. Warn early so the
        # user can enable LongPathsEnabled or shorten output.output_dir.
        if sys.platform.startswith("win") and len(str(venv_dir)) > 200:
            _log.warning(
                "[setup] venv path is %d chars (%s) — on Windows this risks "
                "exceeding the 260-char MAX_PATH once native extensions are "
                "installed, which fails figure generation. Enable Windows "
                "long-path support (LongPathsEnabled) or set a shorter "
                "output.output_dir.", len(str(venv_dir)), venv_dir,
            )
        py = self.python_path(quest_root)
        # A venv is safe to REUSE only if it was fully built. CPython writes
        # pyvenv.cfg BEFORE copying the interpreter and running ensurepip, so a
        # run killed mid-build leaves pyvenv.cfg with no interpreter — reusing
        # it makes every later execute()/install() spawn a missing python
        # (FileNotFoundError) on every --resume, an unrecoverable loop until the
        # user manually deletes .venv/. Gate the skip on the interpreter
        # actually existing (mirrors cleanup_after_success) AND a cheap pip
        # probe (catches a partial ensurepip); rebuild a broken dir from clean.
        if (venv_dir / "pyvenv.cfg").exists() and py.is_file():
            probe = await self.execute(
                [str(py), "-c", "import pip"], cwd=quest_root, timeout_s=30,
            )
            if probe.returncode == 0:
                return
            _log.warning(
                "setup: reusable venv at %s failed the pip probe (rc=%s) — "
                "rebuilding from clean", venv_dir, probe.returncode,
            )
        # `venv.EnvBuilder`/subprocess venv creation is sync; offload so we
        # don't block the loop. clear=True wipes any partial/broken dir
        # before recreating.
        await asyncio.to_thread(
            _build_venv, venv_dir, with_pip=True, clear=True,
            python_version=self.python_version,
            system_site_packages=self.system_site_packages,
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
        result = await self.execute(cmd, cwd=quest_root, timeout_s=600)
        # Retry a failed install once. Two quests installing matplotlib at the
        # same moment in one process have twice left one of them with a pip
        # that fell back to building it from source and failed ("Could not
        # build wheels for matplotlib", Windows CI), while the other quest's
        # identical install succeeded. The cause did not reproduce locally, so
        # this is a second attempt, not a fix of a known race. A timeout is not
        # retried: another 600-second wait is not a transient.
        if result.returncode != 0 and not result.timed_out:
            _log.warning(
                "[install] pip install rc=%d; retrying once. %s",
                result.returncode, pip_failure_summary(result.stderr),
            )
            result = await self.execute(cmd, cwd=quest_root, timeout_s=600)
        if result.returncode == 0:
            self._record_requested(pkgs, quest_root)
        return result

    @staticmethod
    def _record_requested(pkgs: list[str], quest_root: Path) -> None:
        """Remember what the quest asked pip for. With system_site_packages a
        satisfied request installs nothing into the venv, so ``pip freeze
        --local`` cannot see it; this file is how cleanup finds those."""
        try:
            fi_dir = quest_root / ".fi"
            fi_dir.mkdir(parents=True, exist_ok=True)
            with (fi_dir / "pip_requested.txt").open("a", encoding="utf-8") as fh:
                fh.write("\n".join(pkgs) + "\n")
        except OSError as exc:
            _log.debug("[install] could not record requested packages: %s", exc)

    async def _inherited_pins(
        self, py: Path, quest_root: Path, local_freeze: str
    ) -> str:
        """``name==version`` lines for packages the quest requested that the
        venv got from FI's own interpreter rather than installing itself."""
        try:
            requested = (quest_root / ".fi" / "pip_requested.txt").read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            return ""
        local = {
            _norm_dist(line.split("==")[0])
            for line in local_freeze.splitlines() if "==" in line
        }
        wanted: list[str] = []
        for spec in requested:
            m = _REQ_NAME.match(spec.strip())
            if m and _norm_dist(m.group(0)) not in local and m.group(0) not in wanted:
                wanted.append(m.group(0))
        if not wanted:
            return ""
        probe = await self.execute(
            [str(py), "-c", _PIN_SCRIPT, *wanted], cwd=quest_root, timeout_s=60,
        )
        return probe.stdout if probe.returncode == 0 else ""

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
        result = ExecutionResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
        )
        # Reactive diagnostic: a native-extension DLL load failure on Windows is
        # almost always the venv path crossing MAX_PATH. Surface an actionable
        # hint even on the otherwise-silent web_plots path (rc!=0 -> {} there).
        # Gated to Windows — the MAX_PATH hint is Windows-specific advice.
        if (
            sys.platform.startswith("win")
            and result.returncode not in (0, None)
            and _looks_like_dll_load_failure(result.stderr)
        ):
            _log.warning("[execute] %s", _DLL_LOAD_HINT)
        return result

    async def cleanup_after_success(self, quest_root: Path) -> Path | None:
        """Freeze the venv's pip state to ``.fi/requirements.lock.txt``
        then delete ``<quest_root>/.venv/`` to reclaim disk space.

        A typical quest venv lands at 150–250 MB (matplotlib, pandas,
        scipy, etc.); over a multi-quest session this dominates the
        ``outputs/`` footprint. The frozen lock file is everything a
        future reader needs to reproduce the environment:
        ``python -m venv .venv && .venv/bin/pip install -r .fi/requirements.lock.txt``.

        Idempotent and best-effort: if the venv is missing (Docker run,
        ``no_simulation=true``, already cleaned) or freeze/delete fails
        (permission denied on a stuck handle, AV scanner holding files
        open on Windows, …), we log and return None — never raise.
        A failed cleanup must not mask a successful quest.
        """
        venv_dir = quest_root / ".venv"
        if not (venv_dir / "pyvenv.cfg").exists():
            return None

        # The whole body must honor "log and return None — never raise"
        # so a venv-cleanup hiccup can't mask a successful quest. Every
        # filesystem op below (mkdir, write_text, rmtree) can raise on
        # permission / disk-full / AV-lock failures; the outer
        # try/except converts all of them to a logged warning.
        try:
            fi_dir = quest_root / ".fi"
            fi_dir.mkdir(parents=True, exist_ok=True)
            lock_path = fi_dir / "requirements.lock.txt"
            py = self.python_path(quest_root)
            if not py.is_file():
                _log.warning(
                    "cleanup_after_success: venv python missing at %s; skipping freeze + delete",
                    py,
                )
                return None
            # ``pip freeze`` produces a deterministic, pip-installable list.
            # ``--local`` excludes globally-installed packages when the venv
            # has global access (``system_site_packages=True``) — without
            # it, the lock file would list everything FI's own interpreter
            # happens to have installed alongside what this quest actually
            # asked for, which is not what "what did this quest need"
            # should mean. Harmless no-op when system_site_packages=False.
            freeze = await self.execute(
                [str(py), "-m", "pip", "freeze", "--local"],
                cwd=quest_root,
                timeout_s=60,
            )
            if freeze.returncode != 0:
                _log.warning(
                    "cleanup_after_success: pip freeze rc=%d stderr=%s; "
                    "keeping .venv/ for debugging",
                    freeze.returncode,
                    freeze.stderr[-300:],
                )
                return None
            # Stamp the lock file with a short header so a future reader
            # knows what produced it and how to reuse it. POSIX and
            # Windows reproduction lines are both spelled out — the
            # ``pip install -r ...`` form is the load-bearing part and
            # was previously truncated on the Windows hint.
            header = (
                "# Frozen by FrontierInsight on quest success.\n"
                "# Reproduce (POSIX):\n"
                "#   python -m venv .venv && "
                ".venv/bin/pip install -r .fi/requirements.lock.txt\n"
                "# Reproduce (Windows):\n"
                "#   python -m venv .venv && "
                ".venv\\Scripts\\pip install -r .fi\\requirements.lock.txt\n"
            )
            inherited = await self._inherited_pins(py, quest_root, freeze.stdout)
            if inherited.strip():
                inherited = (
                    "\n# Requested by the quest but provided by FI's own "
                    "interpreter, so not installed into the venv.\n"
                    "# Pinned to the versions this quest ran against; their "
                    "dependencies resolve fresh on reinstall.\n" + inherited
                )
            lock_path.write_text(header + freeze.stdout + inherited, encoding="utf-8")
            # Delete the venv. ``ignore_errors`` rather than ``onerror=``
            # so a stuck file handle on Windows doesn't propagate — the
            # lock file is the durable artifact; a stray .venv/ is
            # harmless leftover that the user can ``rm -rf`` themselves.
            await asyncio.to_thread(shutil.rmtree, venv_dir, True)
            # Verify the delete actually succeeded before claiming it.
            # On Windows ``ignore_errors=True`` can leave the directory
            # partially intact when a DLL handle is held open by an AV
            # scanner — the lock file is still valid (reproducibility
            # is preserved) but the disk-reclaim claim would be a lie.
            if venv_dir.exists():
                _log.warning(
                    "cleanup_after_success: froze %d packages to %s, but "
                    ".venv/ at %s was only partially removed (likely a "
                    "Windows file lock); you can delete it manually.",
                    freeze.stdout.count("\n"),
                    lock_path,
                    venv_dir,
                )
            else:
                _log.info(
                    "cleanup_after_success: froze %d packages to %s and removed %s",
                    freeze.stdout.count("\n"),
                    lock_path,
                    venv_dir,
                )
            return lock_path
        except OSError as e:
            _log.warning(
                "cleanup_after_success: filesystem error during cleanup "
                "(quest still succeeded): %r",
                e,
            )
            return None


_DLL_LOAD_HINT = (
    "a native extension failed to load (DLL load failed). On Windows this is "
    "usually the per-quest venv path exceeding the 260-char MAX_PATH limit, "
    "which breaks PIL/matplotlib and silently kills figure generation. Enable "
    "Windows long-path support (LongPathsEnabled) or set a shorter "
    "output.output_dir (e.g. C:\\fi) and re-run."
)


_LONG_PATH_HINT = (
    "cause: a file in this install has a path over Windows' 260-character "
    "limit, so pip installed nothing. Fix: enable LongPathsEnabled (needs "
    "admin), or run FI on a Python installed at a short path (e.g. "
    "C:\\Python311); with execution.shared_interpreter: false, set "
    "output.output_dir to a short path such as C:\\fi."
)


def pip_failure_summary(stderr: str) -> str:
    """What to log when ``pip install`` fails. pip prints the real cause on its
    ``ERROR:`` lines and follows them with generic hints, so the last few
    hundred characters (what used to be logged) were only the hint — never the
    package or the path that failed."""
    errors = [
        ln.strip() for ln in stderr.splitlines() if ln.strip().startswith("ERROR:")
    ]
    summary = " | ".join(errors) if errors else stderr.strip()[-400:]
    summary = summary[:800]
    low = stderr.lower()
    if (
        "enable-long-paths" in low or "winerror 206" in low
        or "filename or extension is too long" in low
    ):
        summary += " || " + _LONG_PATH_HINT
    return summary


def _looks_like_dll_load_failure(stderr: str) -> bool:
    """True when stderr carries the native-extension load-failure signature.
    Specific enough not to false-positive on ordinary experiment errors."""
    low = stderr.lower()
    return (
        "dll load failed" in low
        or "the filename or extension is too long" in low
    )


_REQ_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# One line on purpose: passed as a `-c` argument, and a multi-line argument is
# fragile across Windows argv quoting.
_PIN_SCRIPT = (
    "import re,sys,importlib.metadata as m;"
    "n=lambda s:re.sub('[-_.]+','-',s).lower();"
    "w={n(x) for x in sys.argv[1:]};"
    "[print(x) for x in sorted({d.metadata['Name']+'=='+d.version "
    "for d in m.distributions() "
    "if d.metadata['Name'] and n(d.metadata['Name']) in w})]"
)


def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _resolve_python_for_version(python_version: str) -> str | None:
    """Find an interpreter matching ``python_version`` (e.g. ``"3.11"``),
    or ``None`` when none is found — callers fall back to ``sys.executable``.

    ``sys.executable`` is checked FIRST and preferred when it already
    matches: no subprocess needed, and it's guaranteed to have ``venv`` +
    ``pip`` working (it's the interpreter running FI itself). Only searches
    elsewhere when it does not match, so a single-Python-install machine
    (the common case) never pays for the search.
    """
    want = tuple(int(p) for p in python_version.split(".")[:2])
    have = sys.version_info[:2]
    if have == want:
        return sys.executable

    candidates: list[str] = []
    if sys.platform == "win32":
        # The `py` launcher (ships with every python.org Windows install)
        # picks a specific installed version regardless of what's on PATH
        # or which interpreter is currently running FI — the one reliable
        # way to target a version other than sys.executable on Windows.
        py_launcher = shutil.which("py")
        if py_launcher:
            probe = subprocess.run(
                [py_launcher, f"-{python_version}", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=15,
            )
            if probe.returncode == 0:
                candidates.append(probe.stdout.strip())
    else:
        found = shutil.which(f"python{python_version}")
        if found:
            candidates.append(found)

    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    return None


def _build_venv(
    venv_dir: Path, *, with_pip: bool, clear: bool = False,
    python_version: str = "3.11", system_site_packages: bool = True,
) -> None:
    """Build the quest venv from the interpreter matching ``python_version``,
    not whichever interpreter happens to be running FI.

    Before this, ``venv.EnvBuilder().create()`` unconditionally used
    ``sys.executable`` — the CURRENTLY RUNNING interpreter — and silently
    ignored ``execution.python_version`` entirely (it was stored on
    ``VenvExecutor`` but never read). On a machine with more than one
    Python install, whichever one happened to launch FI that particular
    time decided every quest's venv, with no consistency guarantee quest
    to quest — the same declared ``python_version`` could silently mean a
    different interpreter, and therefore different wheels / ABI, run to
    run. ``clear=True`` wipes a partial/broken venv dir (e.g. left by a run
    killed mid-build) before recreating it, so a rebuild starts clean.

    ``system_site_packages=True`` (default, was the ``venv.EnvBuilder``
    default of ``False``) lets each quest's venv see whatever FI's own
    interpreter already has installed — matplotlib, numpy, pandas are
    near-universal across quests, so on a machine with a cold pip cache
    every quest previously re-downloaded and re-built them from scratch.
    A quest's own ``pip install`` still installs INTO the venv as normal
    and takes precedence there; inheriting only fills in what a quest
    doesn't ask for itself.
    """
    resolved = _resolve_python_for_version(python_version)
    if resolved is None:
        _log.warning(
            "[setup] no Python %s interpreter found (checked the `py` "
            "launcher on Windows, `python%s` on PATH elsewhere) — "
            "building the venv from the currently-running interpreter "
            "(%s) instead. Install python.org's Python %s or add it to "
            "PATH for a consistent venv across runs.",
            python_version, python_version, sys.executable, python_version,
        )
        resolved = sys.executable

    if resolved == sys.executable:
        # Fast, in-process path — no subprocess needed.
        builder = venv.EnvBuilder(
            with_pip=with_pip, clear=clear, upgrade_deps=False,
            system_site_packages=system_site_packages,
        )
        builder.create(str(venv_dir))
        return

    # A different interpreter than the one running FI: venv.EnvBuilder has
    # no way to target one, since it always builds from sys.executable.
    # Spawn that interpreter's own `-m venv` instead — this function
    # already runs off the event loop (asyncio.to_thread), so a blocking
    # subprocess call here is consistent with the rest of this module.
    if clear and venv_dir.exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
    cmd = [resolved, "-m", "venv"]
    if system_site_packages:
        cmd.append("--system-site-packages")
    if not with_pip:
        cmd.append("--without-pip")
    cmd.append(str(venv_dir))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"venv creation failed via {resolved} (python_version="
            f"{python_version}): rc={result.returncode}\n{result.stderr[-2000:]}"
        )


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

    async def cleanup_after_success(self, quest_root: Path) -> Path | None:
        """Docker has no per-quest venv on disk — the interpreter and
        every installed package live inside an ephemeral container
        that's already torn down per execute() call. Nothing to free."""
        return None

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


class SharedInterpreterExecutor(VenvExecutor):
    """Runs quest code with the interpreter that runs FI itself — no venv.

    One Python for everything. What ``pip install -e .`` (or an earlier quest)
    put there is simply there; nothing is built under the quest's own output
    path, which is where Windows' 260-character limit stopped pip installing
    ``torch`` into a per-quest venv. No isolation: what a quest installs stays
    in FI's environment.
    """

    _pip_lock_timeout_s: float = 1800

    async def setup(self, quest_root: Path) -> None:
        quest_root.mkdir(parents=True, exist_ok=True)
        want =tuple(int(p) for p in self.python_version.split(".")[:2] if p.isdigit())
        if want and want != tuple(sys.version_info[:2]):
            _log.info(
                "[setup] execution.python_version=%s is not used: quests run on "
                "FI's own Python %d.%d (%s).",
                self.python_version, sys.version_info[0], sys.version_info[1],
                sys.executable,
            )

    def python_path(self, quest_root: Path) -> Path:
        return Path(sys.executable)

    async def install(
        self, packages: Iterable[str], *, quest_root: Path
    ) -> ExecutionResult:
        pkgs = list(packages)
        if not pkgs:
            return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.0)
        # Every quest now installs into the same site-packages, and two pip
        # processes writing it at once corrupt it — the fleet runner runs
        # quests concurrently, so serialise across processes.
        from filelock import FileLock
        lock_dir = Path.home() / ".frontier-insight"
        lock_dir.mkdir(parents=True, exist_ok=True)
        # thread_local=False is load-bearing: acquire runs in a worker thread
        # and release on the event-loop thread, and with the default a release
        # from a different thread is a silent no-op — the lock stays held and
        # the NEXT install in this process blocks for the whole timeout.
        lock = FileLock(
            str(lock_dir / "pip-install.lock"),
            timeout=self._pip_lock_timeout_s, thread_local=False,
        )
        _log.info("[install] waiting for the shared pip lock (%s)", lock.lock_file)
        await asyncio.to_thread(lock.acquire)
        try:
            return await super().install(pkgs, quest_root=quest_root)
        finally:
            lock.release()

    async def cleanup_after_success(self, quest_root: Path) -> Path | None:
        """Nothing to delete. Record what the quest asked for, pinned to the
        versions it ran against, as ``.fi/requirements.lock.txt``."""
        try:
            pins = await self._inherited_pins(Path(sys.executable), quest_root, "")
            if not pins.strip():
                return None
            fi_dir = quest_root / ".fi"
            fi_dir.mkdir(parents=True, exist_ok=True)
            lock_path = fi_dir / "requirements.lock.txt"
            lock_path.write_text(
                "# Frozen by FrontierInsight on quest success.\n"
                f"# This quest ran in FI's own Python {sys.version.split()[0]} "
                f"({sys.executable}), not a per-quest venv.\n"
                "# These are the packages it asked for, pinned to the versions "
                "it ran against; their dependencies resolve fresh.\n"
                "# Reproduce: pip install -r .fi/requirements.lock.txt\n" + pins,
                encoding="utf-8",
            )
            return lock_path
        except OSError as exc:
            _log.warning("cleanup_after_success: could not write lock file: %s", exc)
            return None


def make_executor(
    sandbox: str, *, python_version: str, docker_image: str,
    system_site_packages: bool = True, shared_interpreter: bool = True,
) -> Executor:
    if sandbox == "venv" and shared_interpreter:
        return SharedInterpreterExecutor(python_version=python_version)
    if sandbox == "venv":
        return VenvExecutor(
            python_version=python_version,
            system_site_packages=system_site_packages,
        )
    if sandbox == "docker":
        return DockerExecutor(image=docker_image)
    raise ValueError(f"unknown sandbox: {sandbox!r}")
