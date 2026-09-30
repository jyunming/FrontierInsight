"""Per-quest code execution.

`VenvExecutor` is the default — each quest gets `<quest_root>/.venv/` and
agent-generated code runs as a child process of the FI engine. Cross-
platform: the venv's Python lives at `Scripts/python.exe` on Windows and
`bin/python` on POSIX.

`DockerExecutor` is the opt-in that runs each command in an ephemeral
container from a stock image with the quest_root mounted: no network,
capped memory / CPU / process count (`DockerLimits`), every Linux
capability dropped, no privilege gain, and a non-root user where the
host allows one.

Both expose the same async `execute(cmd, cwd, timeout_s) -> ExecutionResult`.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol

if TYPE_CHECKING:
    from core.skills.mounts import SkillMount

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


@dataclass(frozen=True)
class DockerLimits:
    """What one experiment container may use: ``execution.docker_memory_gb``,
    ``execution.docker_cpus`` and ``execution.docker_max_processes``."""

    memory_gb: float = 4.0
    cpus: float = 2.0
    max_processes: int = 1024

    @classmethod
    def from_config(cls, execution: object) -> "DockerLimits":
        return cls(
            memory_gb=float(getattr(execution, "docker_memory_gb", cls.memory_gb)),
            cpus=float(getattr(execution, "docker_cpus", cls.cpus)),
            max_processes=int(getattr(execution, "docker_max_processes", cls.max_processes)),
        )

    @property
    def mem_limit(self) -> str:
        """docker-py's spelling: whole gigabytes as ``4g``, anything else in megabytes."""
        mb = max(1, int(round(self.memory_gb * 1024)))
        return f"{mb // 1024}g" if mb % 1024 == 0 else f"{mb}m"

    @property
    def memory_label(self) -> str:
        # "GB" as a person says it; Docker is given GiB (x 1024), a little more.
        return f"{self.memory_gb:g} GB"


# The id experiments run as under Docker Desktop on Windows / macOS (see
# DockerExecutor._user_candidates). Any fixed non-root id works there.
_DESKTOP_USER = "1000:1000"
# What a script prints when it cannot start one more process or thread: the
# process-count limit (pids cgroup) makes fork/clone/pthread_create fail with
# EAGAIN, which each runtime words its own way.
_PROCESS_LIMIT_RE = re.compile(
    r"can't start new thread|pthread_create|fork: (?:retry: )?Resource temporarily unavailable"
    r"|BlockingIOError: \[Errno 11\] Resource temporarily unavailable"
)
# How many threads numerical libraries start (OpenMP, OpenBLAS, MKL, numexpr),
# and os.cpu_count() / multiprocessing's default pool size on Python 3.13+.
_THREAD_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTHON_CPU_COUNT",
)
# Creates and removes one file in the quest folder: can this user write there?
_WRITE_CHECK = (
    "import os, tempfile; fd, p = tempfile.mkstemp(dir='/work', prefix='.fi-write-check-'); "
    "os.close(fd); os.remove(p)"
)


def _owner_of(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_uid, st.st_gid


# Host variables that mean nothing in the container, or break its Python
# (PYTHONHOME stops it starting; a Windows PYTHONPYCACHEPREFIX becomes a
# folder in the quest; LD_PRELOAD prints an error on every run).
_HOST_ONLY_VARS = frozenset({
    "PATH", "HOME", "PWD", "OLDPWD", "TMPDIR", "TEMP", "TMP",
    "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONPYCACHEPREFIX", "PYTHONEXECUTABLE",
    "LD_PRELOAD", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
})
# Where a path in a value ends: the separators of a list or an option.
_PATH_END = r"""(?=$|[;,=\s"'])"""


def _to_container(value: str, host_root: str, *, windows: bool | None = None) -> str:
    """``value`` with the host quest folder as /work, only where the folder
    name ends (``/q/abc`` is not in ``/q/abcd``). On a Windows host the folder
    matches in either slash and any letter case, and the rest of the path is
    written with slashes: a Linux container reads a backslash as part of a
    file name."""
    if windows is None:
        windows = os.sep == "\\"
    if not windows:
        root = host_root.rstrip("/") or "/"
        return re.sub(re.escape(root) + r"(?=$|/|[:;,=\s\"'])", "/work", value)
    parts = [p for p in re.split(r"[\\/]+", host_root) if p]
    root_rx = r"[\\/]+".join(re.escape(p) for p in parts)
    # The whole value is one path under the folder (a command argument, which
    # may hold spaces): all of the rest is the path.
    whole = re.match(root_rx + r"(?P<rest>[\\/].*)?$", value, re.IGNORECASE | re.DOTALL)
    if whole:
        return "/work" + (whole.group("rest") or "").replace("\\", "/")
    return re.sub(
        root_rx + r"""(?P<rest>[\\/][^;,=\s"']*)?""" + _PATH_END,
        lambda m: "/work" + (m.group("rest") or "").replace("\\", "/"),
        value, flags=re.IGNORECASE,
    )


def _who(user: str) -> str:
    return f"user {user} (not root)" if user else "the container's root user"


class DockerExecutor:
    """Sandboxed executor — runs every command inside an ephemeral Docker
    container with the quest_root bind-mounted at /work.

    The same Executor protocol as VenvExecutor: `setup` ensures the image
    is available, `install` runs `pip install ...` inside the container,
    `execute` runs the given command. `python_path` returns the
    in-container interpreter path (caller treats it as opaque).

    Every container has no network, the memory / CPU / process caps of
    ``limits``, no Linux capabilities, cannot gain privileges, and runs as a
    non-root user where the host allows one (``setup`` decides which, once).
    A run stopped by a cap ends with a plain ``[FI]`` line on its stderr, and
    the same line goes to ``log`` (the quest's run.log).

    The approved external skills a quest selected are the one other thing
    mounted, read-only, each at ``/fi-skills/<name>`` (``set_skill_mounts``).
    """

    def __init__(
        self, *, image: str = "python:3.11-slim", limits: DockerLimits | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.image = image
        self.limits = limits or DockerLimits()
        self._qlog = log  # the quest's logger (run.log); None -> this module's
        self._client: object | None = None  # lazy docker client
        self._skill_mounts: tuple["SkillMount", ...] = ()
        # The container user: None = not decided yet, "" = the image's default
        # (root inside the container), else "uid:gid".
        self._user: str | None = None
        self._info: dict | None = None  # the daemon's `docker info`, read once

    def _say(self, level: int, msg: str, *args: object) -> None:
        logger = self._qlog or _log
        (logger.warning if level >= logging.WARNING else logger.info)(msg, *args)

    def _daemon_info(self, client: object) -> dict:
        if self._info is None:
            try:
                info = client.info()  # type: ignore[attr-defined]
            except Exception:
                info = None
            self._info = info if isinstance(info, dict) else {}
        return self._info

    def _cpus(self, client: object) -> float:
        """The CPU cap, never more than the daemon has: Docker refuses to
        create a container asked for more CPUs than it has."""
        ncpu = self._daemon_info(client).get("NCPU")
        if isinstance(ncpu, int) and ncpu > 0:
            return min(self.limits.cpus, float(ncpu))
        return self.limits.cpus

    def _user_candidates(self, client: object, host_root: Path) -> list[str]:
        """The users experiments may run as, best first; ``setup`` takes the
        first that can write the quest folder ("" = the container's root).

        Who can write the mounted quest folder depends on the host:

        * Windows / macOS (Docker Desktop and the like): the folder reaches the
          Linux VM through a file-sharing layer that writes as the host user
          whatever id the container uses, so a fixed non-root id works.
        * Rootless Docker: the container's root is mapped to the unprivileged
          host user, and any other container id to a sub-id that cannot write
          the host user's files. Root inside is already not root on the host.
        * Linux otherwise (native Docker, Docker Desktop's WSL integration,
          Docker Desktop for Linux): the owner of the quest folder (normally the
          person running FI), so what the experiment writes stays theirs; where
          the host maps that owner to the container's root instead (Docker
          Desktop for Linux), the write check fails and root is next. A folder
          owned by root (FI itself run as root) leaves only root.

        The container's root is always the last resort. Whoever runs, every
        capability is dropped and privileges cannot be gained.
        """
        if not sys.platform.startswith("linux"):
            return [_DESKTOP_USER, ""]
        info = self._daemon_info(client)
        security = " ".join(str(s) for s in (info.get("SecurityOptions") or []))
        if "name=rootless" in security:
            return [""]
        try:
            uid, gid = _owner_of(host_root)
        except OSError:
            return [""]
        return [""] if uid == 0 else [f"{uid}:{gid}", ""]

    def _create_kwargs(self, client: object, user: str) -> dict[str, object]:
        """The isolation every container gets, experiment or write check."""
        info = self._daemon_info(client)
        kw: dict[str, object] = {
            "network_disabled": True,  # no network from the experiment by default
            "mem_limit": self.limits.mem_limit,
            "pids_limit": int(self.limits.max_processes),
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "user": user or None,  # None = the image's default
        }
        # Swap equal to memory: over the cap the run is stopped (and said so)
        # instead of swapping itself slowly to a halt. Left out where the
        # kernel cannot limit swap (Docker would only warn and drop it).
        if info.get("SwapLimit") is not False:
            kw["memswap_limit"] = self.limits.mem_limit
        # A CPU cap needs the kernel's CFS quota; without it (e.g. rootless
        # Docker whose cpu controller is not delegated) Docker refuses the
        # container outright, so the cap is left out and setup() says so.
        if info.get("CpuCfsQuota") is not False and info.get("CpuCfsPeriod") is not False:
            kw["nano_cpus"] = int(round(self._cpus(client) * 1e9))
        return kw

    def _write_check(self, client: object, host_root: Path, user: str) -> bool:
        container = client.containers.create(  # type: ignore[attr-defined]
            self.image,
            command=["python", "-c", _WRITE_CHECK],
            working_dir="/work",
            volumes={str(host_root): {"bind": "/work", "mode": "rw"}},
            environment=self._env_for(client, user, {}),
            detach=True,
            **self._create_kwargs(client, user),
        )
        try:
            container.start()
            status = container.wait(timeout=120)
            return isinstance(status, dict) and status.get("StatusCode") == 0
        finally:
            try:
                container.remove(force=True)
            except Exception:
                pass

    def _resolve_user(self, client: object, host_root: Path) -> None:
        """Decide the container user once: the first candidate that can write a
        file in the quest folder."""
        candidates = self._user_candidates(client, host_root)
        chosen, ok = candidates[-1], False
        for user in candidates:
            try:
                ok = self._write_check(client, host_root, user)
            except Exception as exc:  # noqa: BLE001 -- the run itself will say what is wrong
                # Unknown for this one, known to fail for any before it: go
                # with this one.
                chosen, ok = user, True
                self._say(logging.WARNING, "[docker] could not check who can write the quest folder (%s); "
                          "experiments run as %s", exc, _who(chosen))
                break
            if ok:
                chosen = user
                break
        self._user = chosen
        if not ok:
            self._say(logging.WARNING, "[docker] no user can write the quest folder %s from the container (or the "
                      "image %s cannot run python), so experiments will not be able to save their results. Check "
                      "that Docker is allowed to share that folder, and execution.docker_image.",
                      host_root, self.image)
        elif chosen != candidates[0]:
            self._say(logging.WARNING, "[docker] experiments cannot write the quest folder as user %s on this Docker "
                      "setup, so they run as the container's root user, still with no network, no added privileges "
                      "and the same limits", candidates[0])
        if chosen and sys.platform.startswith("linux"):
            self._warn_root_owned(host_root, chosen)
        self._say_limits(client, chosen)

    def _say_limits(self, client: object, user: str) -> None:
        """One run.log line with the limits actually applied, and one warning
        for each this Docker cannot apply (it would drop it with only a warning
        of its own, which nobody sees)."""
        info = self._daemon_info(client)
        cpus = self._cpus(client)
        cpu_capped = "nano_cpus" in self._create_kwargs(client, user)
        applied = [f"at most {self.limits.memory_label} of memory"] if info.get("MemoryLimit") is not False else []
        if cpu_capped:
            applied.append(f"{cpus:g} CPUs")
        if info.get("PidsLimit") is not False:
            applied.append(f"{self.limits.max_processes} processes and threads")
        self._say(logging.INFO, "[docker] experiments run as %s, with %s (execution.docker_memory_gb / docker_cpus "
                  "/ docker_max_processes)", _who(user), ", ".join(applied) or "no resource limits")
        if cpus < self.limits.cpus:
            self._say(logging.WARNING, "[docker] execution.docker_cpus is %g but Docker has only %g CPUs; using %g",
                      self.limits.cpus, cpus, cpus)
        if not cpu_capped:
            self._say(logging.WARNING, "[docker] this Docker cannot limit CPU use, so execution.docker_cpus only sets "
                      "how many threads numerical libraries start")
        if info.get("MemoryLimit") is False:
            self._say(logging.WARNING, "[docker] this Docker cannot limit memory, so execution.docker_memory_gb is "
                      "not applied")
        elif info.get("SwapLimit") is False:
            self._say(logging.WARNING, "[docker] this Docker cannot limit swap, so an experiment over the %s memory "
                      "limit may slow down instead of being stopped", self.limits.memory_label)
        if info.get("PidsLimit") is False:
            self._say(logging.WARNING, "[docker] this Docker cannot limit the number of processes, so "
                      "execution.docker_max_processes is not applied")

    def _warn_root_owned(self, host_root: Path, user: str) -> None:
        """A quest first run before experiments stopped running as root can hold
        files and folders root owns, which the quest's own user cannot change:
        say which, and how to take them back. Looks two levels down, at most
        2000 entries."""
        found: list[str] = []
        seen = 0
        stack: list[tuple[Path, int]] = [(host_root, 0)]
        while stack and seen < 2000 and len(found) < 3:
            folder, depth = stack.pop()
            try:
                it = os.scandir(folder)
            except OSError:
                continue
            try:
                for e in it:  # lazily: a folder of 100k files stops at the cap
                    seen += 1
                    if seen > 2000:
                        break
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if st.st_uid == 0:
                        found.append(e.path)
                        if len(found) >= 3:
                            break
                    elif depth < 1 and e.is_dir(follow_symlinks=False):
                        stack.append((Path(e.path), depth + 1))
            finally:
                close = getattr(it, "close", None)
                if close is not None:
                    close()
        if found:
            more = " (and maybe more)" if len(found) >= 3 or seen > 2000 else ""
            self._say(logging.WARNING, "[docker] %s%s in the quest folder belong to root (written when experiments "
                      "still ran as root), and experiments now run as user %s, which cannot change them. Run "
                      "`sudo chown -R %s %s` once to give them back.", ", ".join(found), more, user, user, host_root)

    def _env_for(
        self, client: object, user: str, env: dict[str, str], host_root: Path | None = None,
    ) -> dict[str, str]:
        """The container's environment, from the host-side ``env``.

        The engine hands over the host's whole environment. The host's system
        and Python paths (``_HOST_ONLY_VARS``) mean nothing in the container or
        break its Python (a Windows PATH even hides the image's python), so they
        go; a value naming a path in the quest folder is translated to /work,
        and PYTHONPATH keeps only what FI puts there for the container (paths
        under /work and /fi-skills), joined the Linux way.

        Numerical libraries start one thread per core they see, and a CPU cap
        does not change what they see (every core of the host), so on a large
        machine they would crowd into the cap and hit the process limit: each
        gets the cap, unless it already asks for no more than that. A non-root
        id has no home directory in a stock image: it gets /tmp, so libraries
        that keep a cache or config there (matplotlib) work."""
        out = {k: v for k, v in env.items() if k.upper() not in _HOST_ONLY_VARS and isinstance(v, str)}
        root = str(host_root) if host_root is not None else None
        for k, v in list(out.items()):
            if k == "PYTHONPATH":
                parts = [_to_container(p, root) if root else p for p in v.split(os.pathsep) if p]
                kept = [p for p in parts if p == "/work" or p.startswith(("/work/", "/fi-skills/"))]
                if kept:
                    out[k] = ":".join(kept)
                else:
                    out.pop(k)
            elif root:
                out[k] = _to_container(v, root)
        cap = max(1, math.ceil(self._cpus(client)))
        for k in _THREAD_VARS:
            v = str(out.get(k, "")).strip()
            if not (v.isdigit() and 1 <= int(v) <= cap):
                out[k] = str(cap)
        if user:
            out["HOME"] = "/tmp"
            if not any(out.get(k) for k in ("LOGNAME", "USER", "LNAME", "USERNAME")):
                out["USER"] = "fi"
        return out

    @property
    def skill_mounts(self) -> tuple["SkillMount", ...]:
        """The approved external skill folders every container from now on
        gets, read-only."""
        return self._skill_mounts

    def set_skill_mounts(self, mounts: Iterable["SkillMount"]) -> None:
        """Replace the skill folders mounted read-only next to the quest. The
        engine sets these from the skills a quest selected and a person
        approved (``core.skills.mounts.plan_mounts``); nothing else is ever
        mounted, and an empty list (the default) mounts nothing."""
        self._skill_mounts = tuple(mounts)

    def _volumes(self, host_root: Path) -> dict[str, dict[str, str]]:
        """The quest read-write at /work, plus each planned skill folder
        read-only. A skill folder that is no longer what was planned (replaced
        by a link since) is left out and said so: the container never gets a
        path the plan did not check."""
        volumes: dict[str, dict[str, str]] = {
            str(host_root): {"bind": "/work", "mode": "rw"},
        }
        for m in self._skill_mounts:
            try:
                host = m.bind_host if m.still_safe() else None
            except ValueError:
                host = None
            if host is None:
                _log.warning(
                    "[docker] not mounting skill %r: %s is no longer the folder that "
                    "was approved and planned", m.name, m.declared,
                )
                continue
            volumes[host] = m.volume
        return volumes

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
        if self._user is None:
            await asyncio.to_thread(self._resolve_user, client, quest_root.resolve())

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
        translated = [_to_container(a, str(host_root)) for a in cmd]

        # setup() decides the user; without it, the best candidate, unchecked
        # (and not kept, so a later setup() still checks).
        user = self._user if self._user is not None else self._user_candidates(client, host_root)[0]
        container = client.containers.create(  # type: ignore[attr-defined]
            self.image,
            command=translated,
            working_dir="/work",
            volumes=self._volumes(host_root),
            environment=self._env_for(client, user, env, host_root),
            detach=True,
            **self._create_kwargs(client, user),
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
            note = "" if timed_out else self._limit_note(rc, stderr, _oom_killed(container))
            if note:
                # The repair step and the error a person sees read the end of
                # stderr; run.log gets the same sentence.
                self._say(logging.WARNING, "[docker] %s", note)
                sep = "" if not stderr or stderr.endswith("\n") else "\n"
                stderr = stderr + sep + "[FI] " + note + "\n"
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

    def _limit_note(self, rc: int, stderr: str, oom: bool) -> str:
        """One plain sentence when a run was stopped by one of the container's
        caps, naming the setting that raises it; "" otherwise."""
        mem = self.limits.memory_label
        if oom and rc == 0:
            return (f"part of the experiment (one of its processes) used more than the {mem} memory limit and "
                    "was stopped, so its results may be incomplete; raise execution.docker_memory_gb "
                    f"(now {self.limits.memory_gb:g}) to give it more")
        if oom:
            return (f"the experiment used more than the {mem} memory limit and was stopped; raise "
                    f"execution.docker_memory_gb (now {self.limits.memory_gb:g}) to give it more")
        if rc == 137:
            # Killed (SIGKILL) without Docker recording an out-of-memory stop:
            # some kernels do not report it. The memory cap is the usual cause.
            return (f"the experiment was stopped by a kill signal (exit code 137), most often because it used "
                    f"more than the {mem} memory limit; if it needs more, raise execution.docker_memory_gb "
                    f"(now {self.limits.memory_gb:g})")
        # Only near the end, where the error that stopped the run is: a thread
        # warning earlier on, followed by an unrelated error, is not this.
        if rc != 0 and _PROCESS_LIMIT_RE.search("\n".join((stderr or "").splitlines()[-20:])):
            n = self.limits.max_processes
            return (f"the experiment could not start another process or thread, most likely because it reached "
                    f"the limit of {n} processes and threads at once; raise execution.docker_max_processes "
                    f"(now {n}) or use fewer worker processes")
        return ""


def _oom_killed(container: object) -> bool:
    """Whether Docker recorded that the container went over its memory cap.
    Strictly ``True``: anything else (a missing field, a test double) is no."""
    try:
        container.reload()  # type: ignore[attr-defined]
        attrs = container.attrs  # type: ignore[attr-defined]
    except Exception:
        return False
    if not isinstance(attrs, dict):
        return False
    state = attrs.get("State")
    return isinstance(state, dict) and state.get("OOMKilled") is True


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
    docker_limits: DockerLimits | None = None, log: logging.Logger | None = None,
) -> Executor:
    """``docker_limits`` and ``log`` (the quest's logger, for run.log) are
    used by the Docker sandbox only."""
    if sandbox == "venv" and shared_interpreter:
        return SharedInterpreterExecutor(python_version=python_version)
    if sandbox == "venv":
        return VenvExecutor(
            python_version=python_version,
            system_site_packages=system_site_packages,
        )
    if sandbox == "docker":
        return DockerExecutor(image=docker_image, limits=docker_limits, log=log)
    raise ValueError(f"unknown sandbox: {sandbox!r}")
