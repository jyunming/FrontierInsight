"""Coding agents other than the Claude CLI, for ``engine.code_writer: agent``.

:class:`core.agent_coder.AgentCoder` drives any runner with one method, ``run(prompt, *, cwd, session_id, timeout_s) ->
RunnerOutput``: start a round (``session_id`` ``None``) or continue the session a round left (its ``session_id``), in
``cwd``, the scratch copy of ``code/``. This module gives that method to two more command-line tools the person may
already use, each on this computer with the person's own login, never in a container:

- :class:`CodexAgentRunner`: ``codex exec`` with ``-s workspace-write`` (Codex's own sandbox: file writes only inside
  the working folder; on Windows the sandbox mode is set explicitly, since without the person's config the folder is
  silently read-only), web search off, the tool surfaces FI does not need off, and ``codex exec resume <thread id>`` to
  continue. The session id is the thread id Codex reports (FI cannot choose it).
- :class:`AgyAgentRunner`: Google Antigravity's ``agy --print`` with edits accepted, its terminal sandbox on, web pages,
  MCP servers and files outside the workspace refused through session settings, and ``--conversation <id>`` to continue.

As with the Claude CLI on this computer, neither is an isolation: Python the agent runs can read and write what the
person can. :class:`AgentCoder` watches the quest folder around each round (a tripwire) and reads the agent's code for
reaches outside it, whatever the runner. The prompt goes on stdin, never on the command line.

Measured on 2026-11-27 (``C:/fi_runs/replays/2026-11-27_cli_agents/RESEARCH.md``): each wrote a file, ran Python and
continued its session in a second round.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .agent_coder import RunnerOutput, _no_window, default_workdir
from .execution import scrub_secrets
from .proc_tree import AsyncProcessTree

_TAIL = 2000
_WINDOWS = os.name == "nt"   # read once; a test sets this, never os.name (pathlib reads that)


class CliAgentRunner:
    """What every runner here shares: start the CLI in ``cwd`` with the prompt on stdin, stop the whole process tree at
    the time limit, and turn its output into a :class:`RunnerOutput`. A subclass gives :meth:`argv`, :meth:`stdin` and
    :meth:`parse`. ``launcher`` replaces the CLI program (a test's stand-in)."""

    #: The program, found on PATH (``create_subprocess_exec`` does not honour PATHEXT on Windows).
    program = ""
    #: What the person installs when the program is missing.
    install = ""
    #: Environment variables the CLI must not see (one would outrank the person's own login).
    env_unset: tuple[str, ...] = ()
    #: FI's provider name for the same CLI (its reasoning-effort argument comes from there).
    provider = ""

    def __init__(self, *, model: str = "", effort: str = "", launcher: list[str] | None = None,
                 work_root: Path | None = None) -> None:
        self.model, self.effort, self.launcher = model, effort, launcher
        self.work_root = Path(work_root) if work_root is not None else None

    confined = False

    def workdir(self, quest_root: Path) -> Path:
        return default_workdir(quest_root, self.work_root)

    def close(self, *, final: bool = False) -> None:
        """Nothing of the run is kept by FI: the CLI keeps its own session files where it always does."""

    def _program(self) -> list[str]:
        if self.launcher:
            return list(self.launcher)
        found = shutil.which(self.program)
        if found is None:
            raise RuntimeError(f"the `{self.program}` command was not found on PATH: {self.install}")
        return [found]

    def process_env(self, cwd: Path) -> dict[str, str]:
        env = scrub_secrets()
        for name in self.env_unset:
            env.pop(name, None)
        return env

    def effort_args(self) -> list[str]:
        """The CLI's argument for ``effort`` (FI's own mapping for the same CLI: a level it does not take is left out
        with a warning)."""
        if not self.effort:
            return []
        from .provider import _CLI_SPECS, _cli_effort_args

        return _cli_effort_args(_CLI_SPECS[self.provider], self.effort)

    def argv(self, *, cwd: Path, session_id: str, resume: bool, timeout_s: float) -> list[str]:
        raise NotImplementedError

    def stdin(self, prompt: str) -> str:
        return prompt

    def parse(self, stdout: str) -> dict[str, Any]:
        """``{"session_id", "message", "is_error", "finished", "events"}`` from what the CLI printed (``finished``: its
        own end-of-turn event was seen)."""
        raise NotImplementedError

    async def run(self, prompt: str, *, cwd: Path, session_id: str | None, timeout_s: float) -> RunnerOutput:
        cwd.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        argv = [*self._program(), *self.argv(cwd=cwd.resolve(), session_id=session_id or "", resume=bool(session_id),
                                             timeout_s=timeout_s)]
        tree = await AsyncProcessTree.start(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd), env=self.process_env(cwd), **_no_window())
        done = False
        try:
            try:
                stdout_b, stderr_b = await asyncio.wait_for(tree.proc.communicate(self.stdin(prompt).encode("utf-8")),
                                                            timeout=timeout_s)
            except asyncio.TimeoutError:
                await tree.kill()
                done = True
                return RunnerOutput(exit_code=-1, timed_out=True, duration_s=time.monotonic() - started,
                                    stderr_tail=f"(stopped after {timeout_s:g} s)")
            done = True
            stdout = stdout_b.decode("utf-8", "replace")
            parsed = self.parse(stdout)
            code = int(tree.proc.returncode or 0)
            message = str(parsed["message"]).strip()
            if code == 0 and parsed["is_error"]:
                code = 1
            elif code == 0 and not parsed["finished"]:
                # Exit 0 without the CLI's own end-of-turn event (no output, cut-off output, a diagnostic only): the
                # round is not known to be done, so it is not counted as done.
                code = 1
                printed = " ".join(ln.strip() for ln in stdout.splitlines() if ln.strip() and not ln.lstrip().startswith("{"))
                message = message or (f"the `{self.program}` command ended without reporting the end of its turn"
                                      + (f"; it printed: {printed[-500:]}" if printed else ""))
            return RunnerOutput(exit_code=code, message=message,
                                # Only the id the CLI reported: none means the session did not start or could not be
                                # continued, and AgentCoder then starts a new one from the whole task.
                                session_id=str(parsed["session_id"] or ""),
                                duration_s=time.monotonic() - started,
                                stderr_tail=stderr_b.decode("utf-8", "replace")[-_TAIL:])
        finally:
            # Cancelled or failed mid-call: stop the tree even when the CLI itself has exited, since a helper it started
            # can still be running and holding the pipe the call waited on.
            await tree.aclose(aborted=not done)


def _events(stdout: str) -> list[dict[str, Any]]:
    out = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            out.append(event)
    return out


def _is_app_shortcut_dir(part: str) -> bool:
    """A folder of Windows' app execution shortcuts (``%LOCALAPPDATA%/Microsoft/WindowsApps`` and the per-app folders
    under it)."""
    parts = [x.lower() for x in part.replace("/", "\\").split("\\") if x]
    return any(a == "microsoft" and b == "windowsapps" for a, b in zip(parts, parts[1:]))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


# ---- Codex -------------------------------------------------------------------------------------------------------------

#: Codex's tool surfaces the agent does not need (it keeps its shell and its file patches): the same list as FI's
#: answer-only Codex calls, less ``shell_tool``, ``unified_exec`` and ``code_mode_host`` (with the last one off, Codex
#: 2026-11 runs no command at all: "code-mode host is disabled", and the agent only says it could not write).
CODEX_DISABLED = ("skill_search", "apps", "plugins", "browser_use", "computer_use", "memories", "multi_agent",
                  "image_generation", "view_image", "tool_suggest")


class CodexAgentRunner(CliAgentRunner):
    program = "codex"
    provider = "codex_cli"
    install = "install the Codex CLI and log in (`codex login`)"
    env_unset = ("OPENAI_API_KEY",)   # the person's own login, never a key FI was given for its own requests

    def _common(self) -> list[str]:
        args = ["--json", "--ignore-user-config", "--skip-git-repo-check", "-c", "web_search=disabled"]
        if _WINDOWS:
            # Without the person's config the Windows sandbox is not set up and the folder is read-only, silently
            # (the agent only says it cannot write): the mode that needs no administrator setup is named here.
            args += ["-c", 'windows.sandbox="unelevated"']
        for feature in CODEX_DISABLED:
            args += ["--disable", feature]
        if self.model:
            args += ["-m", self.model]
        return args + self.effort_args()

    def process_env(self, cwd: Path) -> dict[str, str]:
        env = super().process_env(cwd)
        if _WINDOWS:
            # Windows' app shortcuts (``...\Microsoft\WindowsApps``: the Store's pwsh.exe, python.exe) cannot start
            # inside Codex's sandbox ("CreateProcessAsUserW failed: 5 (Access is denied)"): with them on the PATH, every
            # command the agent ran failed and it wrote code it could not run. Left off, the real programs are found.
            for key in [k for k in env if k.upper() == "PATH"]:
                env[key] = ";".join(part for part in env[key].split(";")
                                           if not _is_app_shortcut_dir(part))
        return env

    def argv(self, *, cwd: Path, session_id: str, resume: bool, timeout_s: float) -> list[str]:
        if resume:
            # `exec resume` has no -s and no -C: the sandbox is set as a config value, the folder is the process's own.
            return ["exec", "resume", *self._common(), "-c", 'sandbox_mode="workspace-write"', session_id, "-"]
        return ["exec", *self._common(), "-s", "workspace-write", "-C", str(cwd), "-"]

    def parse(self, stdout: str) -> dict[str, Any]:
        session, message, is_error, finished, events = "", "", False, False, 0
        for event in _events(stdout):
            events += 1
            kind = event.get("type")
            if kind in ("turn.completed", "turn.failed"):
                finished = True
            if kind == "thread.started" and isinstance(event.get("thread_id"), str):
                session = event["thread_id"]
            elif kind == "item.completed":
                item = _dict(event.get("item"))
                if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    message = item["text"]
            elif kind in ("turn.failed", "error"):
                is_error = True
                err = _dict(event.get("error"))
                message = message or str(err.get("message") or event.get("message") or "")
        return {"session_id": session, "message": message, "is_error": is_error, "finished": finished, "events": events}


# ---- Antigravity (agy) ---------------------------------------------------------------------------------------------------

#: The agent's agy settings, in a home of the quest's own (as FI's answer-only agy calls have, so the person's own
#: ``settings.json``, its MCP servers and grants are never used or changed; their chosen model is kept): edits and
#: commands go ahead without asking, nothing outside the workspace, no web pages and no MCP servers. Web search cannot
#: be turned off this way (no setting reaches it).
AGY_SETTINGS: dict[str, Any] = {
    "toolPermission": "always-proceed",
    "allowNonWorkspaceAccess": False,
    "permissions": {"deny": ["read_url(*)", "mcp(*)"]},
}


class AgyAgentRunner(CliAgentRunner):
    program = "agy"
    provider = "antigravity_cli"
    install = "install Google Antigravity's CLI (`agy`) and log in"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._homes: set[Path] = set()

    @staticmethod
    def home(cwd: Path) -> Path:
        """agy's home for this quest: next to the quest's scratch folder (outside the quest, and outside the folder
        whose every new file FI counts as the agent's), the same every round, since agy keeps the conversation a later
        round continues there."""
        return cwd.parent.parent / f"{cwd.parent.name}_agy_home"

    def workdir(self, quest_root: Path) -> Path:
        work = super().workdir(quest_root)
        self._homes.add(self.home(work))   # known after a resume too, so the quest's end removes it
        return work

    def process_env(self, cwd: Path) -> dict[str, str]:
        from .provider import _write_cli_home

        env = super().process_env(cwd)
        home = self.home(cwd)
        if not (home / ".gemini" / "antigravity-cli" / "settings.json").is_file():
            shutil.rmtree(home, ignore_errors=True)
            _write_cli_home(home, AGY_SETTINGS)
        self._homes.add(home)
        env.update({"HOME": str(home), "USERPROFILE": str(home)})
        # A grant inherited from an agy or Antigravity terminal FI was started from would reach the agent.
        env.pop("ANTIGRAVITY_PERM_GRANTS", None)
        return env

    async def run(self, prompt: str, *, cwd: Path, session_id: str | None, timeout_s: float) -> RunnerOutput:
        if session_id and not (self.home(cwd) / ".gemini" / "antigravity-cli" / "conversations"
                               / f"{session_id}.db").is_file():
            # agy does not fail on a conversation it does not have: it says `conversation "<id>" not found` on stderr
            # and starts a new one from this round's words alone (seen when a quest's earlier rounds were another
            # agent's). Said as a session that could not be continued, so AgentCoder starts again from the whole task.
            return RunnerOutput(exit_code=1, stderr_tail=f"agy has no conversation {session_id} in this quest's home")
        return await super().run(prompt, cwd=cwd, session_id=session_id, timeout_s=timeout_s)

    def close(self, *, final: bool = False) -> None:
        if final:
            for home in self._homes:
                shutil.rmtree(home, ignore_errors=True)
            self._homes.clear()

    def argv(self, *, cwd: Path, session_id: str, resume: bool, timeout_s: float) -> list[str]:
        args = ["--print", "", "--input-format", "stream-json", "--output-format", "stream-json",
                "--mode", "accept-edits", "--sandbox", "--dangerously-skip-permissions",
                # agy's own print-mode limit (5 minutes by default) ends a round with no word; FI's limit decides.
                "--print-timeout", f"{int(timeout_s) + 60}s"]
        if self.model:
            args += ["--model", self.model]
        args += self.effort_args()
        if resume:
            args += ["--conversation", session_id]
        return args

    def stdin(self, prompt: str) -> str:
        # agy takes a prompt on stdin only as one stream-json turn (``-p ""`` alone is refused as an empty prompt).
        return json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}, ensure_ascii=False) + "\n"

    def parse(self, stdout: str) -> dict[str, Any]:
        session, message, is_error, finished, events = "", "", False, False, 0
        texts: dict[str, str] = {}   # the agent's words, by step (the last step's are its last message)
        for event in _events(stdout):
            events += 1
            kind = str(event.get("event") or "")
            body = _dict(event.get(kind))
            if kind == "init" and isinstance(event.get("conversation_id"), str):
                session = event["conversation_id"]
            if isinstance(body.get("conversation_id"), str) and body["conversation_id"]:
                session = body["conversation_id"]
            if kind == "result":
                finished = True
                message = str(body.get("response") or "")
                status = str(body.get("status") or "")
                is_error = status != "SUCCESS"   # a result event without its status is not a finished turn
                if is_error and not message:
                    message = str(body.get("error") or status)
            elif kind == "step_update" and body.get("step_type") == "agent_response" and body.get("text_delta"):
                step = str(body.get("step_index", ""))
                texts[step] = texts.pop(step, "") + str(body["text_delta"])
        last_text = list(texts.values())[-1] if texts else ""
        return {"session_id": session, "message": message or last_text, "is_error": is_error, "finished": finished,
                "events": events}


#: The provider a quest's code-writing step uses -> the runner class of the same CLI. A provider not named here keeps
#: the Claude CLI (the agent mode's first runner).
RUNNERS: dict[str, type[CliAgentRunner]] = {"codex_cli": CodexAgentRunner, "antigravity_cli": AgyAgentRunner}

#: The providers whose command-line tool can be the coding agent, and that tool's program.
AGENT_PROGRAMS: dict[str, str] = {"claude_cli": "claude", "codex_cli": "codex", "antigravity_cli": "agy"}

#: Where a quest keeps who writes its code, so a resumed quest keeps the writer it started with.
CODE_WRITER_RECORD = Path(".fi") / "code_writer.json"


def step_provider(config: Any) -> tuple[str, str]:
    """The provider of the code-writing step (its own ``provider.node_providers.implement`` entry, else the main
    provider) and its reasoning level."""
    from .provider import node_provider_for

    provider = config.provider
    step = node_provider_for(getattr(provider, "node_providers", None), "implement")
    entry = step[1] if step is not None else provider
    return str(getattr(entry, "name", "") or ""), str(getattr(entry, "reasoning_effort", "") or "")


def _program_on_path(program: str) -> str | None:
    return shutil.which(program)


class CodeWriterRecordError(RuntimeError):
    """The quest's record of who writes its code cannot be read or saved: deciding again on a later start could switch
    the writer of a quest half-way, so the run stops and says what to set."""


def resolve_code_writer(config: Any, quest_root: Path, *, keep: bool = False) -> tuple[str, str]:
    """``("agent" | "chat", the one plain line to say or "")``. ``engine.code_writer: chat`` or ``agent`` is the person's
    choice and is kept. ``auto`` (the default): the coding agent when the code-writing step's provider is a command-line
    tool that can be one (Claude, Codex, Antigravity) and that tool is installed, else the chat model, function by
    function. A quest that already ran before this record existed keeps the chat model it was written with.

    ``keep`` (a quest that runs, :meth:`core.engine.Engine.run`): the choice is recorded in the quest the first time, so
    a resume keeps it; a record that exists but cannot be read, or a choice that cannot be saved, raises
    :class:`CodeWriterRecordError` before anything runs. Without ``keep`` (an engine built only to look at a quest)
    nothing is written and nothing raises."""
    asked = str(getattr(config.engine, "code_writer", "chat") or "chat")
    if asked in ("chat", "agent"):
        return asked, ""
    record = Path(quest_root) / CODE_WRITER_RECORD
    if record.exists():
        try:
            kept = json.loads(record.read_text(encoding="utf-8")).get("writer")
        except (OSError, ValueError, AttributeError):
            kept = None
        if kept in ("chat", "agent"):
            return kept, ""
        if not keep:
            return "chat", ""
        raise CodeWriterRecordError(
            f"this quest's record of who writes its code ({record}) cannot be read; set engine.code_writer to chat "
            "or agent in the config to say which, then run it again")
    if (Path(quest_root) / ".fi" / "state.sqlite").exists():
        writer, line = "chat", ""   # started before code_writer had an automatic choice: chat was the default then
    else:
        name, _effort = step_provider(config)
        program = AGENT_PROGRAMS.get(name)
        if program and _program_on_path(program):
            writer = "agent"
            line = (f"the code is written by the `{program}` command-line tool as a coding agent, which writes, runs and "
                    "fixes the whole program while FI's checks judge it (engine.code_writer: chat to have the chat "
                    "model write it function by function instead)")
        else:
            writer = "chat"
            why = (f"the `{program}` command was not found" if program
                   else f"the {name or 'main'} provider has no command-line coding agent")
            line = (f"the code is written by the chat model, function by function ({why}; a coding agent needs the "
                    "Claude, Codex or Antigravity command-line tool as the code-writing step's provider)")
    if not keep:
        return writer, line
    tmp = record.with_name(record.name + ".tmp")
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"writer": writer}) + "\n", encoding="utf-8")
        os.replace(tmp, record)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise CodeWriterRecordError(
            f"who writes this quest's code could not be saved in {record} ({exc}); a resumed quest must keep the writer "
            "it started with: make the quest folder writable, or set engine.code_writer to chat or agent") from exc
    return writer, line
