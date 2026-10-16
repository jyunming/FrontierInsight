"""EXPERIMENTAL, not wired into the engine yet: a coding agent (Claude CLI) writes and fixes the quest's code in ``code/``.

The agent works in the quest's ``code/`` folder the way a person would: it writes files, runs Python, reads the errors and
tries again. Its own report counts for nothing: FI runs its own checks on the files the agent leaves, and when one fails
the failure text goes back to the same agent session (:meth:`AgentCoder.continue_`). This module holds the parts that do
not depend on how the engine uses them:

* :class:`ClaudeCliRunner`, the agent: ``claude -p``. The normal path (``mode="host"``) runs the CLI directly on this
  computer, as the person, with their own login. **It is not isolated**: there is no sandbox, and Python run by the
  agent can read and write anything the person can. It works in a scratch COPY of ``code/`` outside the quest, in a
  folder named by a random id (:func:`scratch_id`), with only file tools and Python allowed. Docker is an optional
  extra used only when its daemon is running (``mode="docker"``): a throwaway container that has only that copy (and
  a per-quest folder for the agent's session files) mounted. :func:`make_runner` chooses, :func:`docker_status` says
  whether Docker is running.
* :class:`AgentCoder`: the rounds (a write, then a bounded number of fixes), recorded in ``.fi/agent_coder.json`` before
  each starts. The copy is made fresh from ``code/`` before each round (no ``.git`` history, no ``.venv``, no value file
  an older FI left there); after it FI compares the copy with ``code/``, keeps its own files as they were (a change to
  one is recorded) and copies back only the agent's own files, by FI's list (Python, and plain text or data files).
  Without a container a TRIPWIRE watches for writes: any write into the quest folder (``.venv/`` included), beside the
  copy, or into the folders Python starts from (a ``.pth``, ``sitecustomize.py``) is a cheat that is put back, and the
  agent's code is read for reaches toward FI's check values (:func:`reaches_outside`). A round with a cheat changes
  nothing in ``code/``. The tripwire is not the defence: it never sees a read, and a determined program can get round
  it. The defence is that nothing the agent can see holds a value FI checks with: the values live in the quest's
  ``.fi/checks/`` (:mod:`core.equation_tests`), FI compares every number in its own process
  (:mod:`core.equation_measure`), and FI runs the agent's code from a copy outside the quest
  (:mod:`core.outside_run`). The quest's evidence record says the code writer ran without isolation
  (:func:`evidence_settings`).
* :class:`TreeSnapshot` / :func:`guarded_round`: one fix round that is undone whole when it fails a condition.

**The login.** Without Docker the CLI uses the person's normal login on this computer: nothing is copied and nothing is
printed. In a container nothing secret is copied into the image, the container's folders or a log: the container is
given one environment variable by NAME (``-e CLAUDE_CODE_OAUTH_TOKEN``, no value on the command line), whose value Docker
copies from the environment FI starts the ``docker`` command with; the token is a long-lived one made once with ``claude
setup-token`` (a Claude subscription), set in the person's environment or FI's ``.env``. No API key is passed either way
(``ANTHROPIC_API_KEY`` would take precedence over the subscription). FI's own code runs never see the token
(``core.execution.scrub_secrets`` drops every ``*_TOKEN``).

The agent itself is one swappable piece (:class:`AgentRunner`); a test's stand-in replaces the ``claude`` program
(``launcher``) or the ``docker`` program (``docker``).

Pure of process globals: every :class:`AgentCoder` belongs to one quest.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .execution import scrub_secrets
from .proc_tree import AsyncProcessTree

RECORD = Path(".fi") / "agent_coder.json"
#: The random id that names this quest's copy of ``code/`` outside the quest (kept so a resume finds the same copy).
SCRATCH_ID = Path(".fi") / "agent_coder_copy.json"
_TAIL = 2000


def scratch_id(quest_root: Path | str) -> str:
    """This quest's random copy id (:data:`SCRATCH_ID`), made the first time it is asked for."""
    path = Path(quest_root) / SCRATCH_ID
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("id")
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value):
            return value
    except (OSError, ValueError, AttributeError):
        pass
    value = uuid.uuid4().hex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": value}) + "\n", encoding="utf-8")
    return value

#: Files of ``code/`` FI writes itself (so the agent must not change them) when they are there, and folders whose whole
#: content is FI's. Relative to ``code/``.
FI_OWNED_FILES = ("run.py", "study.json", "fi_search.py", "README.md", "requirements.txt", "CHANGELOG.md", "METHODS.md")
FI_OWNED_DIRS = ("tests",)
#: Outside ``code/`` (host mode only) FI watches the rest of the quest folder: a CLI's own sandbox does not keep its
#: file-writing tool inside the working folder on Windows (checked on the Codex CLI: it wrote `../escape.txt` and a file
#: in `.fi/`). The quest's ``.venv/`` is watched too (by size and time). Not watched: history and caches, and the files
#: FI itself writes while a round runs. This is a tripwire, not a sandbox: it sees writes, never reads.
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache"}
_SKIP_FILES = {".fi/run.log", ".fi/state.sqlite", ".fi/state.sqlite-wal", ".fi/state.sqlite-shm", ".fi/state.sqlite-journal"}
_KEEP_BYTES = 2_000_000          # a file this small is kept in memory to put back; a larger one is only recorded
_KEEP_TOTAL = 64_000_000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _no_window() -> dict[str, Any]:
    # CREATE_NO_WINDOW, never DETACHED_PROCESS (a detached process opens a console window of its own).
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


@dataclass
class RunnerOutput:
    """What one run of the agent program left."""
    exit_code: int = 0
    timed_out: bool = False
    message: str = ""          # the agent's last message
    session_id: str = ""       # so the next round can continue the same conversation ("" when the program never ran)
    duration_s: float = 0.0
    stderr_tail: str = ""


class AgentRunner(Protocol):
    """The swappable piece: run the agent once, in ``cwd``, on ``prompt``; continue ``session_id`` when given."""

    async def run(self, prompt: str, *, cwd: Path, session_id: str | None, timeout_s: float) -> RunnerOutput: ...


@dataclass
class AgentResult:
    ok: bool                       # the agent program exited 0 in time
    round: int = 0                 # 1 for the first write, then 2, 3 ... for each continue
    exit_code: int = 0
    timed_out: bool = False
    exhausted: bool = False        # a continue was refused: the rounds allowed are used up
    message: str = ""
    duration_s: float = 0.0
    tampered: list[str] = field(default_factory=list)   # FI-owned files the agent changed and FI put back
    error: str = ""
    replayed: bool = False         # the write was already done (a resume): nothing was spawned
    cheats: list[str] = field(default_factory=list)     # writes outside its folder, reaches for FI's values (no Docker)
    copied: list[str] = field(default_factory=list)     # no Docker: the files copied back into code/


# ---- Docker on this machine -------------------------------------------------------------------------------------------

DOCKER_OK, DOCKER_MISSING, DOCKER_STOPPED = "ok", "not_installed", "not_running"


async def docker_status(docker: list[str] | None = None, *, timeout_s: float = 20.0) -> tuple[str, str]:
    """``(state, plain sentence)``: :data:`DOCKER_OK` when the ``docker`` command is there and its daemon answers,
    :data:`DOCKER_MISSING` when there is no ``docker`` command, :data:`DOCKER_STOPPED` when the daemon does not answer
    (Docker Desktop is installed but not running)."""
    binary = list(docker) if docker else ([shutil.which("docker")] if shutil.which("docker") else [])
    if not binary:
        return DOCKER_MISSING, "Docker is not installed on this computer (there is no `docker` command)."
    try:
        proc = await asyncio.create_subprocess_exec(
            *binary, "version", "--format", "{{.Server.Version}}", stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **_no_window())
        try:
            out, _err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return DOCKER_STOPPED, "Docker did not answer within %g s; start Docker Desktop and try again." % timeout_s
    except OSError as e:
        return DOCKER_MISSING, f"the `docker` command could not be started ({type(e).__name__})."
    if proc.returncode == 0 and out.decode("utf-8", "replace").strip():
        return DOCKER_OK, f"Docker {out.decode('utf-8', 'replace').strip()} is running."
    return DOCKER_STOPPED, "Docker is installed but not running: start Docker Desktop and try again."


# ---- the Claude CLI -----------------------------------------------------------------------------------------------------

CLAUDE_VERSION = "2.1.299"                      # the version the image pins (the one checked on the host)
IMAGE = f"fi-agent-coder:claude-{CLAUDE_VERSION}"
CONTAINER_HOME = "/home/agent"
CONTAINER_CODE = "/work/code"
#: The one secret the container is given, by name only (see the module docstring).
TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
#: The agent's tools: read, write and edit files, search them, and run commands (in the container: Python, pip).
AGENT_TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
#: What a host-mode agent may run without asking (nobody is there to answer: anything else is refused).
HOST_COMMANDS = ("Bash(python:*)", "Bash(python3:*)", "Bash(py:*)")
#: No self-update, no telemetry or error reports: the container's only traffic is the model API.
QUIET_ENV = {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
             "DISABLE_ERROR_REPORTING": "1"}
#: Runs in the container: install the quest's requirements if it has some (the network is on for the model API anyway),
#: then the CLI. The arguments are passed as "$@", never put in the script.
_ENTRY = ('if [ -f requirements.txt ]; then pip install -q --user -r requirements.txt >&2 || true; fi; '
          'exec claude "$@"')


def parse_stream(stdout: str) -> dict[str, Any]:
    """What ``claude -p --output-format stream-json --verbose`` printed: ``{"session_id", "message", "is_error",
    "events"}``. The session id is on the ``system``/``init`` event and on the closing ``result`` event; the last message
    is the ``result`` event's ``result`` (or, when there is none, the last text an ``assistant`` event carried)."""
    session, message, last_text, is_error, events = "", "", "", False, 0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        events += 1
        if isinstance(event.get("session_id"), str) and event["session_id"]:
            session = event["session_id"]
        kind = event.get("type")
        if kind == "result":
            if isinstance(event.get("result"), str):
                message = event["result"]
            is_error = bool(event.get("is_error")) or str(event.get("subtype") or "").startswith("error")
        elif kind == "assistant":
            content = (event.get("message") or {}).get("content") if isinstance(event.get("message"), dict) else None
            for part in content if isinstance(content, list) else []:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    last_text = part["text"]
    return {"session_id": session, "message": message or last_text, "is_error": is_error, "events": events}


class ClaudeCliRunner:
    """``claude -p`` as the coding agent.

    ``mode="host"`` (the normal path): the CLI on this computer with the person's own login, in the scratch copy of
    ``code/`` outside the quest (:meth:`workdir`; :class:`AgentCoder` makes and compares the copy), ``--restricted``
    (the CLI's own file tools kept to the working folder, command tools only as named, no ``--add-dir``), ``--tools``
    without any web tool, ``--permission-mode acceptEdits``, Python allowed (:data:`HOST_COMMANDS`) and every other
    request refused (``--permission-prompts none``). NOT isolated: ``python`` (``python -c`` included) can read and
    write anything the person can. FI's snapshot of the quest folder around the round and its reading of the agent's
    code (:func:`reaches_outside`) are a tripwire for writes, not a sandbox.

    ``mode="docker"`` (optional, only when the daemon runs): a throwaway container (image :data:`IMAGE`,
    ``docker/agent_coder/Dockerfile``) whose only mounts are the copy of ``code/`` (``/work/code``, read-write; no
    ``.git``) and ``home_dir`` (the CLI's own folder, which holds the session files ``--resume`` needs and nothing
    secret: the login comes in by environment variable, see the module docstring). It has no capabilities, cannot gain
    privileges, runs as a user that is not root, has memory, CPU and process limits, and is stopped (``docker kill``) at
    the wall-clock cap. Its network is on, because the CLI must reach the model's API. Inside, the CLI runs with
    ``--permission-mode bypassPermissions``, ``--safe-mode --strict-mcp-config`` (no person's settings, plugins or MCP
    servers).

    A round starts a new session with an id FI chooses (``--session-id``) and continues one with ``--resume``. The
    prompt goes on stdin, never on the command line. ``launcher`` replaces the ``claude`` program, ``docker`` the
    ``docker`` program (a test's stand-in).
    """

    def __init__(self, *, mode: str = "host", home_dir: Path | None = None, model: str = "", effort: str = "",
                 image: str = IMAGE, memory_gb: float = 4.0, cpus: float = 2.0, max_processes: int = 1024,
                 docker: list[str] | None = None, launcher: list[str] | None = None, token_env: str = TOKEN_ENV,
                 environ: dict[str, str] | None = None, uid_gid: str | None = None,
                 work_root: Path | None = None) -> None:
        if mode not in ("docker", "host"):
            raise ValueError(f"mode must be 'docker' or 'host', not {mode!r}")
        if mode == "docker" and home_dir is None:
            raise ValueError("a Docker runner needs a home_dir for the agent's session files")
        self.mode = mode
        self.home_dir = Path(home_dir) if home_dir is not None else None
        self.model, self.effort, self.image = model, effort, image
        self.memory_gb, self.cpus, self.max_processes = memory_gb, cpus, max_processes
        self.docker, self.launcher, self.token_env = docker, launcher, token_env
        self._environ = environ          # the environment to read the token from (default: FI's own)
        self.uid_gid = uid_gid or (f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else "1000:1000")
        self.work_root = Path(work_root) if work_root is not None else Path(tempfile.gettempdir()) / "fi_agent_work"
        self._built = False

    @property
    def confined(self) -> bool:
        """The agent can reach only what is mounted: :class:`AgentCoder` need not watch the rest of the quest folder."""
        return self.mode == "docker"

    @property
    def works_on_copy(self) -> bool:
        """The agent works on a scratch copy of ``code/`` outside the quest (in a container, that copy is what is
        mounted): :class:`AgentCoder` makes and compares it, and only its own list of files comes back."""
        return True

    def workdir(self, quest_root: Path) -> Path:
        """The copy's folder for this quest: outside the quest, the same every round and after a resume (the CLI files a
        session under its working folder, so ``--resume`` needs the same one), named by a random id made once for the
        quest and kept in its ``.fi/`` (:data:`SCRATCH_ID`), never by the quest's name or path."""
        return self.work_root / scratch_id(quest_root) / "code"

    # -- the command --

    def _claude(self) -> list[str]:
        if self.launcher:
            return list(self.launcher)
        if self.mode == "docker":
            return []   # the image's entry runs `claude`
        # ``create_subprocess_exec`` does not honour PATHEXT on Windows, so ``claude`` is resolved here.
        found = shutil.which("claude")
        if found is None:
            raise RuntimeError("the `claude` command was not found on PATH: install Claude Code and log in "
                               "(the coding agent uses it)")
        return [found]

    def _docker(self) -> list[str]:
        if self.docker:
            return list(self.docker)
        found = shutil.which("docker")
        if found is None:
            raise RuntimeError("the `docker` command was not found on PATH: install Docker Desktop (the coding agent "
                               "runs in a container)")
        return [found]

    def claude_args(self, *, session_id: str, resume: bool) -> list[str]:
        """The CLI's arguments (after the program name)."""
        args = ["-p", "--output-format", "stream-json", "--verbose", "--safe-mode", "--strict-mcp-config",
                "--tools", AGENT_TOOLS]
        if self.mode == "docker":
            args += ["--permission-mode", "bypassPermissions"]
        else:
            args += ["--restricted", "--permission-mode", "acceptEdits", "--permission-prompts", "none",
                     "--allowedTools", *HOST_COMMANDS]
        if self.model:
            args += ["--model", self.model]
        if self.effort:
            args += ["--effort", self.effort]
        args += ["--resume", session_id] if resume else ["--session-id", session_id]
        return args

    def argv(self, *, code_dir: Path, name: str, session_id: str, resume: bool) -> list[str]:
        """The whole command: the CLI itself (host), or ``docker run ... <image> sh -c <entry> sh <args>``."""
        args = self.claude_args(session_id=session_id, resume=resume)
        if self.mode == "host":
            return [*self._claude(), *args]
        assert self.home_dir is not None
        env = ["-e", f"HOME={CONTAINER_HOME}", "-e", f"CLAUDE_CONFIG_DIR={CONTAINER_HOME}/.claude", "-e", self.token_env]
        for key, value in QUIET_ENV.items():
            env += ["-e", f"{key}={value}"]
        entry = ["--entrypoint", "sh", self.image, "-c", _ENTRY, "sh"]
        if self.launcher:   # a stand-in program in the image (tests)
            entry = ["--entrypoint", self.launcher[0], self.image, *self.launcher[1:]]
        return [
            *self._docker(), "run", "--rm", "-i", "--name", name,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", f"{self.memory_gb:g}g", "--cpus", f"{self.cpus:g}", "--pids-limit", str(self.max_processes),
            "--user", self.uid_gid, *env,
            "--mount", f"type=bind,source={code_dir},target={CONTAINER_CODE}",
            "--mount", f"type=bind,source={self.home_dir},target={CONTAINER_HOME}/.claude",
            "-w", CONTAINER_CODE, *entry, *args,
        ]

    def process_env(self) -> dict[str, str]:
        """The environment of the program FI starts (``docker`` or ``claude``): FI's own without any secret, plus the
        login token in Docker mode (Docker passes it on by name; it never appears on a command line)."""
        env = scrub_secrets()
        env.update(QUIET_ENV)
        if self.mode == "docker":
            source = self._environ if self._environ is not None else os.environ
            token = str(source.get(self.token_env) or "").strip()
            if not token:
                raise RuntimeError(
                    f"the coding agent's container needs a Claude login token: run `claude setup-token` once and set "
                    f"{self.token_env} (in your environment or FI's .env)")
            env[self.token_env] = token
        return env

    # -- the image --

    async def _docker_run(self, *args: str, timeout_s: float = 60.0) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *self._docker(), *args, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=scrub_secrets(), **_no_window())
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return -1, "", "(timed out)"
        return int(proc.returncode or 0), out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def ensure_image(self) -> None:
        """Build the image once if it is not there (``docker/agent_coder/Dockerfile``)."""
        if self._built or self.mode != "docker":
            return
        rc, _out, _err = await self._docker_run("image", "inspect", self.image)
        if rc != 0:
            context = Path(__file__).resolve().parent.parent / "docker" / "agent_coder"
            rc, _out, err = await self._docker_run("build", "-t", self.image, "--build-arg",
                                                   f"CLAUDE_VERSION={CLAUDE_VERSION}", str(context), timeout_s=1800.0)
            if rc != 0:
                raise RuntimeError(f"could not build the coding agent's Docker image {self.image}: {err.strip()[-400:]}")
        self._built = True

    def close(self, *, final: bool = False) -> None:
        """The run is over. The session files stay (a resumed quest continues its session) unless ``final`` (the quest
        finished or stopped), which deletes the whole per-quest folder. A login file the CLI may have left there is
        removed either way."""
        if self.home_dir is None:
            return
        if final:
            shutil.rmtree(self.home_dir, ignore_errors=True)
            return
        for name in (".credentials.json",):
            with contextlib.suppress(OSError):
                (self.home_dir / name).unlink()

    # -- one run --

    async def run(self, prompt: str, *, cwd: Path, session_id: str | None, timeout_s: float) -> RunnerOutput:
        cwd.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        resume = bool(session_id)
        session = session_id or str(uuid.uuid4())
        name = ""
        env = self.process_env()   # first: a missing login is said before any image is built
        if self.mode == "docker":
            await self.ensure_image()
            assert self.home_dir is not None
            self.home_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(self.home_dir, 0o700)
            name = f"fi-agent-{self.home_dir.name[:30]}-{os.getpid()}-{int(time.time())}".replace("_", "-")
        argv = self.argv(code_dir=cwd.resolve(), name=name, session_id=session, resume=resume)
        tree = await AsyncProcessTree.start(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd) if self.mode == "host" else None, env=env, **_no_window())
        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(tree.proc.communicate(prompt.encode("utf-8")),
                                                            timeout=timeout_s)
            except asyncio.TimeoutError:
                if name:
                    await self._docker_run("kill", name, timeout_s=30.0)   # the container itself, not just the client
                await tree.kill()
                return RunnerOutput(exit_code=-1, timed_out=True, duration_s=time.monotonic() - started,
                                    stderr_tail=f"(stopped after {timeout_s:g} s)")
            parsed = parse_stream(stdout_b.decode("utf-8", "replace"))
            code = int(tree.proc.returncode or 0)
            if code == 0 and parsed["is_error"]:
                code = 1
            return RunnerOutput(exit_code=code, message=str(parsed["message"]).strip(),
                                session_id=str(parsed["session_id"] or (session if parsed["events"] else "")),
                                duration_s=time.monotonic() - started,
                                stderr_tail=stderr_b.decode("utf-8", "replace")[-_TAIL:])
        finally:
            await tree.aclose(aborted=False)


#: The one line said when the agent runs without Docker (information, not a warning).
NO_DOCKER_LINE = ("the coding agent runs on this computer without isolation (no sandbox): it works on a copy of code/ "
                  "outside the quest, FI watches for writes outside that copy (a tripwire, not a sandbox), and the "
                  "quest's record says the code writer ran without isolation")


async def make_runner(quest_root: Path | str, *, model: str = "", effort: str = "", use_docker: bool = True,
                      docker: list[str] | None = None, log: logging.Logger | None = None,
                      **kwargs: Any) -> ClaudeCliRunner:
    """The runner for this machine. The normal path needs no Docker: the CLI on this computer, on a scratch copy of
    ``code/`` (said in one plain line, :data:`NO_DOCKER_LINE`). Only when ``use_docker`` and the Docker daemon is actually
    running: the container (its session files in a per-quest folder next to the quest folders, never inside one).
    Docker that is missing or stopped never blocks anything."""
    quest_root = Path(quest_root)
    log = log or logging.getLogger(__name__)
    if use_docker:
        state, _why = await docker_status(docker)
        if state == DOCKER_OK:
            home = quest_root.resolve().parent / ".agent_claude_home" / quest_root.name
            log.info("[implement] the coding agent runs in a Docker container that sees only code/")
            return ClaudeCliRunner(mode="docker", home_dir=home, model=model, effort=effort, docker=docker, **kwargs)
    log.info("[implement] %s", NO_DOCKER_LINE)
    return ClaudeCliRunner(mode="host", model=model, effort=effort, **kwargs)


# ---- what the agent's code reaches for --------------------------------------------------------------------------------

#: Names of FI's own check files and folders: code that names one is reaching for the values FI checks it against.
_FI_CHECK_NAMES = ("fi_checks", "oracle_values", "equation_values", "equation_numbers", "FI_CHECKS_DIR",
                   "test_oracles", "test_equations", "frozen_protocol", "FROZEN_PROTOCOL", "agent_coder.json")
_PARENT_PART = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
_FI_FOLDER = re.compile(r"(^|[\\/])\.fi([\\/]|$)")
#: A path that starts at a drive, a share, a home or the root (``/c`` alone is cmd's switch, not a path: not here).
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|root|tmp|work|mnt|etc|var)(?:/|$)|~[\\/])")


def reaches_outside(source: str, rel: str = "") -> list[str]:
    """Plain sentences, one per place where ``source`` reaches for FI's check values or outside its own folder: a string
    that names one of FI's check files (``.fi``, ``oracle_values`` ...), a path that climbs out (``..``) or starts at a
    drive, a home or the root, two or more ``.parent`` steps, nested ``os.path.dirname``, ``Path.home()`` /
    ``expanduser``. Read from the syntax only; empty when nothing is found (or the file is not Python). This is the
    no-Docker guard against reads, which a snapshot of the folders cannot see."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    where = f"{rel}:" if rel else "line "
    found: list[str] = []
    # A string that is a statement of its own (a docstring) is text for a reader, never a path the code opens.
    prose = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
             and isinstance(n.value.value, str)}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            text = node.value
            if len(text) > 300:   # a docstring or a long message, not a path
                continue
            named = next((n for n in _FI_CHECK_NAMES if n in text), "") or (".fi" if _FI_FOLDER.search(text) else "")
            if named:
                found.append(f"{where}{line} names `{named}`, one of FI's own check files")
            elif _PARENT_PART.search(text):
                found.append(f"{where}{line} uses the path `{text[:60]}`, which climbs out of its folder")
            elif _ABSOLUTE.match(text):
                found.append(f"{where}{line} uses the absolute path `{text[:60]}`")
        elif isinstance(node, ast.Attribute) and node.attr == "parent" and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "parent":
            found.append(f"{where}{line} climbs two folders up (`.parent.parent`)")
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            found.append(f"{where}{line} climbs folders up (`.parents[...]`)")
        elif isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name in ("home", "expanduser", "getcwdb"):
                found.append(f"{where}{line} reaches the home folder (`{name}`)")
            elif name == "dirname" and any(isinstance(a, ast.Call) and getattr(a.func, "attr", getattr(a.func, "id", ""))
                                           == "dirname" for a in node.args):
                found.append(f"{where}{line} climbs two folders up (`dirname(dirname(...))`)")
    return sorted(set(found), key=found.index)


def startup_dirs() -> list[Path]:
    """The folders where a file makes every later Python run do something on start (a ``.pth``, ``sitecustomize.py``,
    ``usercustomize.py``): the person's user site and this interpreter's site folders. FI's own engine starts from
    them, so a file left there would run inside FI."""
    import site

    found: list[Path] = []
    with contextlib.suppress(Exception):
        found.append(Path(site.getusersitepackages()))
    with contextlib.suppress(Exception):
        found += [Path(p) for p in site.getsitepackages()]
    return [p for p in dict.fromkeys(found)]


def _startup_files(dirs: Iterable[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for folder in dirs:
        try:
            names = [p for p in Path(folder).iterdir() if p.is_file()]
        except OSError:
            continue
        for path in names:
            if path.suffix.lower() == ".pth" or path.name.lower() in _STARTUP_NAMES:
                with contextlib.suppress(OSError):
                    st = path.stat()
                    out[str(path)] = (st.st_size, st.st_mtime_ns)
    return out


#: What the evidence record says about how the code writer ran (core/evidence.py), the way core/phased_isolation.py
#: marks held-back rows: without a container nothing kept the agent from the rest of the computer.
ISOLATION_UNVERIFIED = "isolation_unverified"
NO_ISOLATION_WHY = ("the coding agent that wrote the code ran on this computer as the person, without isolation: FI "
                    "gave it a copy of code/ outside the quest folder and watched what changed, which is a tripwire, "
                    "not a sandbox (it sees writes, never reads)")


def evidence_settings(quest_root: Path | str) -> dict[str, str]:
    """What the evidence ladder is told about the code writer (``settings["code_writer_isolation"]``): ``{}`` when no
    coding agent wrote this quest's code, ``isolation_unverified`` with the reason when any round ran without a
    container, ``docker`` when every round ran in one."""
    try:
        record = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    rounds = [r for r in (record.get("rounds") or []) if isinstance(r, dict)] if isinstance(record, dict) else []
    if not rounds:
        return {}
    if all(r.get("isolation") == "docker" for r in rounds):
        return {"code_writer_isolation": "docker"}
    return {"code_writer_isolation": ISOLATION_UNVERIFIED, "code_writer_isolation_why": NO_ISOLATION_WHY}


# ---- the coder -----------------------------------------------------------------------------------------------------


class AgentCoder:
    """One quest's coding agent: :meth:`write` once, then :meth:`continue_` with what FI's checks found, a bounded number
    of times. Rounds are recorded in ``.fi/agent_coder.json`` before they start, so a crash cannot make a resume forget one.

    ``owned`` names the files FI owns under ``code/`` beyond the fixed ones (:data:`FI_OWNED_FILES`,
    :data:`FI_OWNED_DIRS`); ``max_fixes`` is the number of :meth:`continue_` rounds allowed after a write and ``timeout_s``
    the wall-clock cap of each round.
    """

    def __init__(self, quest_root: Path | str, runner: AgentRunner, *, max_fixes: int = 3, timeout_s: float = 1800.0,
                 owned: Callable[[], Iterable[str]] | None = None, log: logging.Logger | None = None,
                 startup: Iterable[Path] | None = None) -> None:
        self._startup = [Path(p) for p in startup] if startup is not None else None
        self.quest_root = Path(quest_root)
        self.code_dir = self.quest_root / "code"
        self.runner = runner
        self.max_fixes = max(0, int(max_fixes))
        self.timeout_s = float(timeout_s)
        self._owned = owned
        self._log = log or logging.getLogger(__name__)

    def close(self, *, final: bool = False) -> None:
        """The run is over: the runner tidies its own folder. ``final`` (the quest finished or stopped, it will not be
        resumed) also removes the session files a resume would have continued, and the scratch copy of ``code/``."""
        if final and getattr(self.runner, "works_on_copy", False):
            shutil.rmtree(self.runner.workdir(self.quest_root).parent, ignore_errors=True)
        closer = getattr(self.runner, "close", None)
        if closer is None:
            return
        try:
            closer(final=final)
        except TypeError:   # a runner whose close() takes no argument
            closer()

    # -- the record --

    @property
    def total_rounds(self) -> int:
        return 1 + self.max_fixes

    def load(self) -> dict[str, Any]:
        try:
            data = json.loads((self.quest_root / RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save(self, record: dict[str, Any]) -> None:
        path = self.quest_root / RECORD
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @property
    def last_message(self) -> str:
        """The agent's last message of the last round."""
        rounds = self.load().get("rounds") or []
        return str(rounds[-1].get("message") or "") if rounds else ""

    def fixes_left(self) -> int:
        return max(0, self.max_fixes - int(self.load().get("fixes") or 0))

    def has_written(self) -> bool:
        return bool(self.load().get("write_done"))

    # -- the tamper guard --

    def _guarded(self) -> dict[str, bytes]:
        """``{relative path: bytes}`` of every file FI owns that is there now."""
        out: dict[str, bytes] = {}
        names = list(FI_OWNED_FILES) + [str(n) for n in (self._owned() if self._owned else [])]
        for name in names:
            path = self.code_dir / name
            if path.is_file():
                out[Path(name).as_posix()] = path.read_bytes()
        for folder in FI_OWNED_DIRS:
            root = self.code_dir / folder
            if root.is_dir():
                for path in root.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts:
                        out[path.relative_to(self.code_dir).as_posix()] = path.read_bytes()
        return out

    def _restore(self, before: dict[str, bytes]) -> list[str]:
        """Put FI's copies back; return what the agent touched (``"<file> (changed|removed|added)"``)."""
        touched: list[str] = []
        for rel, data in before.items():
            path = self.code_dir / rel
            try:
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    touched.append(f"code/{rel} (removed)")
                elif path.read_bytes() != data:
                    path.write_bytes(data)
                    touched.append(f"code/{rel} (changed)")
            except OSError as e:  # noqa: BLE001
                touched.append(f"code/{rel} (could not be put back: {e!r})")
        for folder in FI_OWNED_DIRS:
            root = self.code_dir / folder
            if root.is_dir():
                for path in list(root.rglob("*")):
                    rel = path.relative_to(self.code_dir).as_posix()
                    if path.is_file() and rel not in before and "__pycache__" not in path.parts:
                        with contextlib.suppress(OSError):
                            path.unlink()
                            touched.append(f"code/{rel} (added)")
        return touched

    # -- the rest of the quest folder --

    def _outside(self, *, with_code: bool = False) -> tuple[dict[str, tuple[Any, bytes | None]], set[str]]:
        """``({relative path: (signature, bytes or None)}, directories)`` of the quest folder outside ``code/`` (the whole
        quest folder, ``code/`` included, with ``with_code``: the agent then works on a copy and may touch none of it)."""
        files: dict[str, tuple[Any, bytes | None]] = {}
        dirs: set[str] = set()
        kept = 0
        root = self.quest_root
        for here, names, leaves in os.walk(root):
            at_root = Path(here) == root
            names[:] = [n for n in names if n not in _SKIP_DIRS and not (at_root and n == "code" and not with_code)]
            if not at_root:
                dirs.add(Path(here).relative_to(root).as_posix())
            for leaf in leaves:
                path = Path(here) / leaf
                rel = path.relative_to(root).as_posix()
                if rel in _SKIP_FILES or rel.endswith(".pyc"):
                    continue
                try:
                    st = path.stat()
                    if rel.startswith(".venv/"):
                        # The quest's Python environment, which FI later runs its own checks with: watched by size and
                        # time (a change is a cheat), never kept in memory.
                        files[rel] = ((st.st_size, st.st_mtime_ns), None)
                    elif st.st_size <= _KEEP_BYTES and kept + st.st_size <= _KEEP_TOTAL:
                        data = path.read_bytes()
                        kept += st.st_size
                        files[rel] = (_sha(data), data)
                    else:
                        files[rel] = ((st.st_size, st.st_mtime_ns), None)
                except OSError:
                    continue
        return files, dirs

    def _restore_outside(self, before: tuple[dict[str, tuple[Any, bytes | None]], set[str]], *,
                         with_code: bool = False) -> list[str]:
        """Delete what the agent created outside ``code/``, put back what it changed or removed (when FI kept the bytes).
        ``with_code``: the whole quest folder (the agent worked on a copy), said as "outside its copy"."""
        old_files, old_dirs = before
        now_files, now_dirs = self._outside(with_code=with_code)
        place = "outside its copy of code/" if with_code else "outside code/"
        touched: list[str] = []
        for rel in sorted(set(now_files) - set(old_files)):
            with contextlib.suppress(OSError):
                (self.quest_root / rel).unlink()
                touched.append(f"{rel} (created {place})")
        for rel, (sig, data) in old_files.items():
            cur = now_files.get(rel)
            if cur is not None and cur[0] == sig:
                continue
            what = "removed" if cur is None else "changed"
            if data is None:
                touched.append(f"{rel} ({what} {place}; too large for FI to put back)")
                continue
            try:
                (self.quest_root / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.quest_root / rel).write_bytes(data)
                touched.append(f"{rel} ({what} {place})")
            except OSError as e:  # noqa: BLE001
                touched.append(f"{rel} ({what} {place}; could not be put back: {e!r})")
        for rel in sorted(now_dirs - old_dirs, key=len, reverse=True):
            with contextlib.suppress(OSError):
                (self.quest_root / rel).rmdir()
        return touched

    def _startup_dirs(self) -> list[Path]:
        return list(self._startup) if self._startup is not None else startup_dirs()

    def _restore_startup(self, before: dict[str, Any]) -> list[str]:
        """A file Python runs on start that the round added (removed again) or changed (recorded; FI cannot put it
        back): each is a cheat."""
        now = _startup_files(self._startup_dirs())
        found: list[str] = []
        for path in sorted(set(now) - set(before)):
            with contextlib.suppress(OSError):
                Path(path).unlink()
            found.append(f"{path} (a file Python runs on start, created outside the copy; removed)")
        for path in sorted(p for p in before if p in now and now[p] != before[p]):
            found.append(f"{path} (a file Python runs on start, changed outside the copy; FI cannot put it back)")
        for path in sorted(set(before) - set(now)):
            found.append(f"{path} (a file Python runs on start, removed outside the copy)")
        return found

    def _code_changed_since(self, snap: "TreeSnapshot | None") -> bool:
        return snap is not None and bool(snap.changed(self.code_dir))

    async def _run_once(self, prompt: str, session_id: str | None, cwd: Path | None = None) -> RunnerOutput:
        try:
            return await self.runner.run(prompt, cwd=cwd or self.code_dir, session_id=session_id,
                                         timeout_s=self.timeout_s)
        except Exception as e:  # noqa: BLE001 -- a runner that could not start is a failed round, said as such
            return RunnerOutput(exit_code=-1, stderr_tail=f"{type(e).__name__}: {e}"[-_TAIL:])

    # -- rounds --

    async def write(self, task: str) -> AgentResult:
        """Round 1: the agent writes the code from ``task``. A write already done for this very task (a resume) is not
        run again."""
        task_sha = _sha(task.encode("utf-8"))
        record = self.load()
        if record.get("write_done") and record.get("task_sha") == task_sha:
            last = (record.get("rounds") or [{}])[-1]
            self._log.info("[implement] the coding agent already wrote this code (round 1 of %d, from the record)",
                           self.total_rounds)
            return AgentResult(ok=True, round=1, message=str(last.get("message") or ""), replayed=True)
        record = {"task": task, "task_sha": task_sha, "session_id": "", "fixes": 0, "write_done": False, "rounds": []}
        return await self._round(record, "write", task, session_id=None)

    async def continue_(self, feedback: str) -> AgentResult:
        """A further round: ``feedback`` (what FI's checks found, in plain words) goes to the same agent session. Refused
        (``exhausted``) once ``max_fixes`` rounds have been used."""
        record = self.load()
        if not record.get("task"):
            return AgentResult(ok=False, error="the coding agent has not written anything yet", exhausted=True)
        if int(record.get("fixes") or 0) >= self.max_fixes:
            return AgentResult(ok=False, exhausted=True, round=1 + int(record.get("fixes") or 0),
                               error=f"the {self.max_fixes} fix round(s) allowed are used")
        record["fixes"] = int(record.get("fixes") or 0) + 1
        session = str(record.get("session_id") or "") or None
        prompt = ("FI ran its own checks on the code you left in code/ and they found this. FI's checks are the judge; "
                  "your own runs do not replace them.\n\n" + feedback.strip() + "\n\n"
                  "Fix the cause in code/ (change as little as you can, keep what works), run what you changed yourself, "
                  "and stop. The same rules as before apply: write only under code/, do not edit the files FI owns, and "
                  "never write an expected value of a check into the code.\n")
        # The whole task again, with what was found: used when there is no session to continue (an earlier round did not
        # report one) and when continuing it fails (a resume of a quest whose session files are gone).
        whole = str(record["task"]) + "\n\n# What FI's checks found in the code now in code/\n\n" + feedback.strip() + "\n"
        if session is None:
            prompt = whole
        return await self._round(record, "fix", prompt, session_id=session, feedback=feedback, fallback_prompt=whole)

    async def _round(self, record: dict[str, Any], kind: str, prompt: str, *, session_id: str | None,
                     feedback: str = "", fallback_prompt: str = "") -> AgentResult:
        n = 1 + int(record.get("fixes") or 0)
        entry: dict[str, Any] = {"n": n, "kind": kind, "status": "started", "feedback_head": feedback.strip()[:400]}
        record.setdefault("rounds", []).append(entry)
        self._save(record)   # before it starts: a crash in the round still counts as a round
        self._log.info("[implement] the coding agent is %s in code/ (round %d of %d)",
                       "writing the code" if kind == "write" else "fixing what FI's checks found", n, self.total_rounds)
        if getattr(self.runner, "works_on_copy", False):
            return await self._copy_round(record, entry, kind, prompt, session_id=session_id,
                                          fallback_prompt=fallback_prompt)
        before = await asyncio.to_thread(self._guarded)
        # A runner that can reach only what it is given (a container with code/ mounted) cannot touch the rest of the quest
        # folder, so it is not walked; the walk of a big folder is off the event loop otherwise.
        outside = None if getattr(self.runner, "confined", False) else await asyncio.to_thread(self._outside)
        resumed_from = await asyncio.to_thread(TreeSnapshot.take, self.code_dir) if session_id is not None else None
        out = await self._run_once(prompt, session_id)
        if (session_id is not None and fallback_prompt and out.exit_code != 0 and not out.timed_out
                and not out.session_id and not await asyncio.to_thread(self._code_changed_since, resumed_from)):
            # The session could not be continued (its files are gone after a resume, or the CLI refused it): it reported no
            # session and left no change, so nothing was done. The same round is made again in a new session from the
            # whole task and what FI found. One round, not two.
            self._log.warning("[implement] the coding agent's session could not be continued (exit code %s); this round "
                              "starts a new session from the whole task", out.exit_code)
            record["session_id"] = ""
            entry["resume_failed"] = True
            out = await self._run_once(fallback_prompt, None)
        tampered = [*await asyncio.to_thread(self._restore, before),
                    *(await asyncio.to_thread(self._restore_outside, outside) if outside is not None else [])]
        if tampered:
            self._log.warning("[implement] the coding agent changed files FI owns; FI put its own copies back: %s",
                              "; ".join(tampered))
        ok = out.exit_code == 0 and not out.timed_out
        if out.session_id:
            record["session_id"] = out.session_id
        if kind == "write" and ok:
            record["write_done"] = True
        entry.update({"status": "done" if ok else ("timed_out" if out.timed_out else "failed"),
                      "exit_code": out.exit_code, "timed_out": out.timed_out, "duration_s": round(out.duration_s, 1),
                      "tampered": tampered, "message": out.message[-_TAIL:], "stderr_tail": out.stderr_tail[-500:]})
        self._save(record)
        if out.timed_out:
            self._log.warning("[implement] the coding agent was stopped after %g s (the wall-clock cap)", self.timeout_s)
        elif not ok:
            self._log.warning("[implement] the coding agent ended with exit code %s: %s", out.exit_code,
                              out.stderr_tail.strip()[-300:] or "(no message)")
        return AgentResult(ok=ok, round=n, exit_code=out.exit_code, timed_out=out.timed_out, message=out.message,
                           duration_s=out.duration_s, tampered=tampered, error="" if ok else out.stderr_tail[-500:])

    # -- the no-Docker round: the agent works on a scratch copy of code/ --

    def _owned_rel(self, rel: str) -> bool:
        names = {Path(n).as_posix() for n in (*FI_OWNED_FILES, *(self._owned() if self._owned else []))}
        return rel in names or rel.split("/", 1)[0] in FI_OWNED_DIRS

    def _fresh_copy(self, work: Path) -> None:
        """The scratch folder holds exactly a copy of ``code/``: no ``.git`` history, no environment, no caches, and none
        of the files an older FI left in ``code/tests/`` with expected values in them (removed from ``code/`` first,
        :func:`core.equation_tests.remove_stale`; FI writes its tests again in the current form)."""
        from . import equation_tests as _eqt
        from . import outside_run as _outside

        removed = _eqt.remove_stale(self.code_dir)
        if removed:
            self._log.info("[implement] removed files an older FI left in code/ with expected values in them: %s",
                           ", ".join(removed))
        scratch = work.parent
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        _outside.copy_code(self.code_dir, work)

    @staticmethod
    def _hashes(folder: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        if not folder.is_dir():
            return out
        for here, names, leaves in os.walk(folder):
            names[:] = [n for n in names if n not in _TREE_SKIP]
            for leaf in leaves:
                if leaf.endswith(".pyc"):
                    continue
                path = Path(here) / leaf
                with contextlib.suppress(OSError):
                    out[path.relative_to(folder).as_posix()] = _sha(path.read_bytes())
        return out

    def _copy_diff(self, work: Path) -> list[str]:
        """The files the copy has that differ from ``code/``: created, removed or changed."""
        mine, theirs = self._hashes(self.code_dir), self._hashes(work)
        return sorted(rel for rel in {*mine, *theirs} if mine.get(rel) != theirs.get(rel))

    def _beside(self, work: Path) -> list[str]:
        """What the agent wrote beside its copy (in the scratch folder, outside ``code/``); removed."""
        found: list[str] = []
        scratch = work.parent
        for path in sorted(scratch.iterdir()) if scratch.is_dir() else []:
            if path.name == work.name:
                continue
            found.append(f"{path.name} (written beside its copy of code/, outside it)")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    path.unlink()
        return found

    def _take_back(self, work: Path, changed: list[str]) -> tuple[list[str], list[str], list[str]]:
        """``(copied, tampered, cheats)``: FI's own files are kept as they are in ``code/`` (a change recorded as
        tampered), the agent's run outputs (``figures/`` ...) are left in the copy, and code that reaches for FI's check
        values or outside its folder is a cheat; with none, the rest is copied into ``code/`` (a file removed in the copy
        is removed)."""
        tampered = [f"code/{rel} ({'removed' if not (work / rel).is_file() else 'added' if not (self.code_dir / rel).is_file() else 'changed'})"
                    for rel in changed if self._owned_rel(rel)]
        take = [rel for rel in changed if not self._owned_rel(rel) and not is_scratch(rel)]
        cheats: list[str] = []
        for rel in take:
            path = work / rel
            if not path.is_file():
                continue
            name = rel.rsplit("/", 1)[-1].lower()
            if name.endswith(".pth") or name in _STARTUP_NAMES:
                cheats.append(f"code/{rel} is a file Python runs when it starts, which the study's code never needs")
            elif rel.endswith(".py"):
                cheats += reaches_outside(path.read_text(encoding="utf-8", errors="replace"), f"code/{rel}")
            elif not name.endswith(TEXT_SUFFIXES):
                cheats.append(f"code/{rel} is not Python or a plain text or data file FI can read "
                              f"({', '.join(TEXT_SUFFIXES)})")
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                named = next((n for n in _FI_CHECK_NAMES if n in text), "")
                if named:
                    cheats.append(f"code/{rel} names `{named}`, one of FI's own check files")
        if cheats:
            return [], tampered, cheats
        for rel in take:
            src, dst = work / rel, self.code_dir / rel
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            else:
                with contextlib.suppress(OSError):
                    dst.unlink()
        return take, tampered, []

    async def _copy_round(self, record: dict[str, Any], entry: dict[str, Any], kind: str, prompt: str, *,
                          session_id: str | None, fallback_prompt: str) -> AgentResult:
        n = int(entry["n"])
        work = self.runner.workdir(self.quest_root)   # type: ignore[attr-defined]
        confined = bool(getattr(self.runner, "confined", False))
        entry["isolation"] = "docker" if confined else "none"
        await asyncio.to_thread(self._fresh_copy, work)
        # Without a container the agent can reach the whole computer: the quest folder (its .venv/ included) and the
        # folders Python starts from are watched for writes. A tripwire, not a sandbox: a read is never seen.
        quest_before = None if confined else await asyncio.to_thread(self._outside, with_code=True)
        startup_before = None if confined else await asyncio.to_thread(_startup_files, self._startup_dirs())
        out = await self._run_once(prompt, session_id, cwd=work)
        if (session_id is not None and fallback_prompt and out.exit_code != 0 and not out.timed_out
                and not out.session_id and not await asyncio.to_thread(self._copy_diff, work)):
            self._log.warning("[implement] the coding agent's session could not be continued (exit code %s); this round "
                              "starts a new session from the whole task", out.exit_code)
            record["session_id"] = ""
            entry["resume_failed"] = True
            await asyncio.to_thread(self._fresh_copy, work)
            out = await self._run_once(fallback_prompt, None, cwd=work)
        # Every write outside the copy is a cheat, put back: in the quest folder (code/ included) and beside the copy.
        cheats = [*(await asyncio.to_thread(self._restore_outside, quest_before, with_code=True)
                    if quest_before is not None else []),
                  *(await asyncio.to_thread(self._restore_startup, startup_before) if startup_before is not None else []),
                  *await asyncio.to_thread(self._beside, work)]
        changed = await asyncio.to_thread(self._copy_diff, work)
        if cheats:
            copied, tampered = [], [f"code/{rel}" for rel in changed if self._owned_rel(rel)]
        else:
            copied, tampered, cheats = await asyncio.to_thread(self._take_back, work, changed)
        ran = out.exit_code == 0 and not out.timed_out
        ok = ran and not cheats
        if tampered:
            self._log.warning("[implement] the coding agent changed files FI owns in its copy; FI kept its own: %s",
                              "; ".join(tampered))
        if cheats:
            self._log.warning("[implement] the coding agent reached outside its copy of code/, so nothing of this round "
                              "is kept: %s", "; ".join(cheats[:6]))
        if out.session_id:
            record["session_id"] = out.session_id
        if kind == "write" and ok:
            record["write_done"] = True
        entry.update({"status": "done" if ok else ("cheat" if ran else "timed_out" if out.timed_out else "failed"),
                      "mode": "copy", "exit_code": out.exit_code, "timed_out": out.timed_out,
                      "duration_s": round(out.duration_s, 1), "tampered": tampered, "cheats": cheats, "copied": copied,
                      "message": out.message[-_TAIL:], "stderr_tail": out.stderr_tail[-500:]})
        self._save(record)
        if out.timed_out:
            self._log.warning("[implement] the coding agent was stopped after %g s (the wall-clock cap)", self.timeout_s)
        elif not ran:
            self._log.warning("[implement] the coding agent ended with exit code %s: %s", out.exit_code,
                              out.stderr_tail.strip()[-300:] or "(no message)")
        error = ("the coding agent reached outside its copy of code/: " + "; ".join(cheats[:6])) if cheats else (
            "" if ran else out.stderr_tail[-500:])
        return AgentResult(ok=ok, round=n, exit_code=out.exit_code, timed_out=out.timed_out, message=out.message,
                           duration_s=out.duration_s, tampered=tampered, error=error, cheats=cheats, copied=copied)


# ---- one round of repair, undone whole when it fails a condition ----------------------------------------------------------

#: Folders the agent's own runs leave in ``code/`` (the scripts write ``figures/``, ``data/``, ``raw/`` next to themselves):
#: not a change to the code, kept on an accepted round and put back with everything else on a rejected one.
SCRATCH_DIRS = ("figures", "data", "raw", "results")
_TREE_SKIP = {"__pycache__", ".pytest_cache", ".git", ".venv", "node_modules"}
#: The only files besides ``.py`` that come back from the agent's copy into ``code/``: plain text and data FI can read.
TEXT_SUFFIXES = (".json", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml", ".toml")
#: Files Python runs by itself when it starts (with ``*.pth``): never written by the study's code.
_STARTUP_NAMES = {"sitecustomize.py", "usercustomize.py"}


@dataclass
class TreeSnapshot:
    """Every file under ``code/`` (bytes, up to the size caps; larger files by size and time) and every folder, taken before
    a round so a rejected round can be put back whole."""
    files: dict[str, tuple[Any, bytes | None]]
    dirs: set[str]

    @classmethod
    def take(cls, code_dir: Path) -> "TreeSnapshot":
        files: dict[str, tuple[Any, bytes | None]] = {}
        dirs: set[str] = set()
        kept = 0
        for here, names, leaves in os.walk(code_dir):
            names[:] = [n for n in names if n not in _TREE_SKIP]
            rel_here = Path(here).relative_to(code_dir).as_posix()
            if rel_here != ".":
                dirs.add(rel_here)
            for leaf in leaves:
                if leaf.endswith(".pyc"):
                    continue
                path = Path(here) / leaf
                rel = path.relative_to(code_dir).as_posix()
                try:
                    st = path.stat()
                    if st.st_size <= _KEEP_BYTES and kept + st.st_size <= _KEEP_TOTAL:
                        data = path.read_bytes()
                        kept += st.st_size
                        files[rel] = (_sha(data), data)
                    else:
                        files[rel] = ((st.st_size, st.st_mtime_ns), None)
                except OSError:
                    continue
        return cls(files, dirs)

    def changed(self, code_dir: Path) -> list[str]:
        """The files now different from the snapshot: created, removed or changed."""
        now = TreeSnapshot.take(code_dir)
        return sorted(rel for rel in {*self.files, *now.files}
                      if (self.files.get(rel) or (None,))[0] != (now.files.get(rel) or (None,))[0])

    def restore(self, code_dir: Path, only: Callable[[str], bool] | None = None) -> list[str]:
        """Put the snapshot back: delete what was created, rewrite what was changed or removed (``only``: just the paths it
        accepts). Returns what could not be put back (a file too large for the snapshot to hold)."""
        now = TreeSnapshot.take(code_dir)
        lost: list[str] = []
        pick = only or (lambda rel: True)
        for rel in sorted(set(now.files) - set(self.files)):
            if pick(rel):
                with contextlib.suppress(OSError):
                    (code_dir / rel).unlink()
        for rel, (sig, data) in self.files.items():
            if not pick(rel):
                continue
            cur = now.files.get(rel)
            if cur is not None and cur[0] == sig:
                continue
            if data is None:
                lost.append(rel)
                continue
            try:
                (code_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                (code_dir / rel).write_bytes(data)
            except OSError:
                lost.append(rel)
        for rel in sorted(now.dirs - self.dirs, key=len, reverse=True):
            if pick(rel + "/x") or pick(rel):
                with contextlib.suppress(OSError):
                    (code_dir / rel).rmdir()
        return lost


def is_scratch(rel: str) -> bool:
    return rel.split("/", 1)[0] in SCRATCH_DIRS


def parse_failures(code_dir: Path, changed: Iterable[str]) -> list[str]:
    """The changed ``.py`` files that are there and are not Python."""
    bad: list[str] = []
    for rel in changed:
        path = code_dir / rel
        if rel.endswith(".py") and path.is_file():
            try:
                ast.parse(path.read_bytes())
            except (SyntaxError, ValueError):
                bad.append(rel)
    return bad


@dataclass
class GuardedRound:
    """What one guarded round came to. ``ok`` only when the round passed every condition; otherwise the whole ``code/``
    folder is as it was before the round and ``reason`` says why, in a sentence. ``kind`` says which condition refused it
    (``failed``: the round did not finish, ``nothing``: no change, ``outside``: a file the caller does not allow,
    ``syntax``: a changed file is not Python, ``refused``: the caller's own check). ``undo()`` puts the whole folder back
    after an accepted round too (a check the caller makes later)."""
    ok: bool
    reason: str = ""
    changed: list[str] = field(default_factory=list)   # the files the round changed (the agent's scratch folders left out)
    message: str = ""
    result: AgentResult | None = None
    kind: str = ""
    snapshot: TreeSnapshot | None = None
    code_dir: Path | None = None

    def undo(self) -> list[str]:
        if self.snapshot is None or self.code_dir is None:
            return []
        return self.snapshot.restore(self.code_dir)


async def guarded_round(coder: AgentCoder, feedback: str, *, allowed: Callable[[str], bool] | None = None,
                        required: bool = True, on_message: Callable[[str], None] | None = None,
                        accept: Callable[[list[str], str], Awaitable[str]] | None = None) -> GuardedRound:
    """ONE fix round of the coding agent, undone whole when it fails a condition.

    ``code/`` is snapshotted whole (bytes) first; ``feedback`` goes to the agent (:meth:`AgentCoder.continue_`); the
    round is rejected and the WHOLE snapshot put back when: the round did not finish (a cap, an error, the rounds used),
    nothing changed while a change was ``required`` (a ``.py`` file), a file ``allowed`` does not accept changed, a changed
    ``.py`` file is not Python, or the caller's own check (``accept(changed files, the agent's last message)``: ``""`` to
    keep, else the reason) refuses it. ``on_message(message)`` is called with the agent's last message as soon as the round
    has finished, before any of that, so a caller that reads something the agent states never loses it to a rejection."""
    code_dir = coder.code_dir
    snap = await asyncio.to_thread(TreeSnapshot.take, code_dir)

    async def rejected(kind: str, reason: str, changed: list[str], res: AgentResult | None) -> GuardedRound:
        lost = await asyncio.to_thread(snap.restore, code_dir)
        if lost:
            reason += f" (could not put back: {', '.join(lost[:3])})"
        return GuardedRound(False, reason, changed, res.message if res else "", res, kind, snap, code_dir)

    res = await coder.continue_(feedback)
    if res.exhausted or not res.ok:
        reason = ("the fix rounds allowed are used" if res.exhausted else
                  "the coding agent was stopped on time" if res.timed_out else "the coding agent ended with an error")
        return await rejected("failed", reason, [], res)
    if on_message is not None:
        on_message(res.message)
    changed = [c for c in await asyncio.to_thread(snap.changed, code_dir) if not is_scratch(c)]
    if required and not any(c.endswith(".py") for c in changed):
        return await rejected("nothing", "the coding agent changed no file", changed, res)
    if allowed is not None:
        outside = [c for c in changed if not allowed(c)]
        if outside:
            return await rejected("outside", "it changed " + ", ".join(outside[:4]) + ", which this repair is not about",
                                  changed, res)
    bad = await asyncio.to_thread(parse_failures, code_dir, changed)
    if bad:
        return await rejected("syntax", ", ".join(bad[:3]) + " is not valid Python", changed, res)
    if accept is not None:
        why = await accept(changed, res.message)
        if why:
            return await rejected("refused", why, changed, res)
    # Only the allowed code carries an accepted round's changes: what the agent's own runs left under figures/, data/, raw/,
    # results/ (new or changed) is put back to what it was.
    await asyncio.to_thread(snap.restore, code_dir, is_scratch)
    return GuardedRound(True, "", changed, res.message, res, "", snap, code_dir)
