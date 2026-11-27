"""The Codex and Antigravity coding agents (core/agent_runners.py), with stand-ins for the ``codex`` and ``agy`` programs.

No model is called: the stand-ins read the prompt on stdin, do what the plan file says, record their arguments and
environment, and print the event lines the real CLIs printed in the probes of 2026-11-27 (only the fields FI reads).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import agent_coder as ac
from core import agent_runners as ar
from core.agent_build import AgentBuildMixin

GOOD_SIM = "def run_trial(cell, trial_id, seed):\n    return {'x': float(seed % 7)}\n"

STANDIN = r'''
import json, os, sys, time
from pathlib import Path
kind = os.environ["FI_STANDIN_KIND"]
plan = json.loads(Path(os.environ["FI_STANDIN_PLAN"]).read_text(encoding="utf-8"))
log = Path(os.environ["FI_STANDIN_LOG"])
raw = sys.stdin.read()
args = sys.argv[1:]
if kind == "agy":
    prompt = json.loads(raw.splitlines()[0])["message"]["content"]
    session = args[args.index("--conversation") + 1] if "--conversation" in args else "conv-new"
else:
    prompt = raw
    session = args[-2] if "resume" in args else "thread-new"   # `exec resume ... <thread id> -`
with log.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"argv": args, "cwd": os.getcwd(), "prompt": prompt, "raw": raw[:200],
                        "openai_key": "OPENAI_API_KEY" in os.environ}) + "\n")
for step in plan:
    if "sleep" in step:
        time.sleep(step["sleep"])
    if "write" in step:
        p = Path(step["write"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(step["text"], encoding="utf-8")
    if "fail" in step:
        if kind == "agy":
            print(json.dumps({"event": "result", "result": {"status": "ERROR", "error": step["fail"]}}))
        else:
            print(json.dumps({"type": "turn.failed", "error": {"message": step["fail"]}}))
        sys.exit(0)
    if "exit" in step:
        sys.exit(step["exit"])
    if "exit_on_resume" in step and ("resume" in args or "--conversation" in args):
        sys.exit(step["exit_on_resume"])
    if "say" in step:   # print this and stop, exit 0
        print(step["say"])
        sys.exit(0)
if kind == "agy":
    conv = Path(os.environ["HOME"]) / ".gemini" / "antigravity-cli" / "conversations"
    conv.mkdir(parents=True, exist_ok=True)
    (conv / f"{session}.db").write_text("", encoding="utf-8")
    print(json.dumps({"event": "init", "conversation_id": session, "init": {"cwd": os.getcwd()}}))
    for i, part in enumerate(["Wrote it ", "and ran it."]):
        print(json.dumps({"event": "step_update", "step_update": {"conversation_id": session, "step_index": 6,
                          "step_type": "agent_response", "text_delta": part}}))
    print(json.dumps({"event": "result", "result": {"conversation_id": session, "status": "SUCCESS",
                      "response": "Wrote it and ran it."}}))
else:
    print(json.dumps({"type": "thread.started", "thread_id": session}))
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "I will write it."}}))
    print(json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "python x.py"}}))
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "Wrote it and ran it."}}))
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}))
'''


@pytest.fixture
def quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    code = root / "code"
    code.mkdir(parents=True)
    (code / "simulate.py").write_text("def run_trial(cell, trial_id, seed):\n    return {}\n", encoding="utf-8")
    (root / ".fi").mkdir()
    return root


def _standin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    folder = tmp_path / f"standin_{kind}"
    folder.mkdir()
    script = folder / "cli.py"
    script.write_text(STANDIN, encoding="utf-8")
    plan, log = folder / "plan.json", folder / "log.jsonl"
    monkeypatch.setenv("FI_STANDIN_KIND", kind)
    monkeypatch.setenv("FI_STANDIN_PLAN", str(plan))
    monkeypatch.setenv("FI_STANDIN_LOG", str(log))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-the-agent")

    class S:
        launcher = [sys.executable, "-B", str(script)]

        @staticmethod
        def plan(steps: list[dict]) -> None:
            plan.write_text(json.dumps(steps), encoding="utf-8")

        @staticmethod
        def calls() -> list[dict]:
            return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.is_file() else []
    S.plan([])
    return S


# --- the commands -------------------------------------------------------------------------------------------------------


def test_codex_command_asks_for_its_folder_sandbox_and_no_web_and_continues_by_thread(tmp_path: Path) -> None:
    runner = ar.CodexAgentRunner(model="gpt-x", effort="high")
    first = runner.argv(cwd=tmp_path, session_id="", resume=False, timeout_s=60)
    assert first[:1] == ["exec"] and first[-1] == "-", "the prompt goes on stdin"
    assert first[first.index("-s") + 1] == "workspace-write" and first[first.index("-C") + 1] == str(tmp_path)
    assert "--ignore-user-config" in first and "web_search=disabled" in first and "--ephemeral" not in first
    disabled = {first[i + 1] for i, a in enumerate(first) if a == "--disable"}
    assert {"browser_use", "apps", "plugins", "multi_agent"} <= disabled
    assert not disabled & {"shell_tool", "unified_exec", "code_mode_host"}, "the agent keeps its shell"
    assert first[first.index("-m") + 1] == "gpt-x" and 'model_reasoning_effort="high"' in first
    assert ('windows.sandbox="unelevated"' in first) == ar._WINDOWS
    again = runner.argv(cwd=tmp_path, session_id="thread-1", resume=True, timeout_s=60)
    assert again[:2] == ["exec", "resume"] and again[-2:] == ["thread-1", "-"]
    assert 'sandbox_mode="workspace-write"' in again and "-s" not in again and "-C" not in again


def test_agy_command_continues_by_conversation_and_its_own_limit_is_past_fis(tmp_path: Path, caplog) -> None:
    runner = ar.AgyAgentRunner(effort="xhigh")
    with caplog.at_level(logging.WARNING):
        first = runner.argv(cwd=tmp_path, session_id="", resume=False, timeout_s=600)
    assert first[first.index("--print-timeout") + 1] == "660s"
    assert first[first.index("--input-format") + 1] == "stream-json" and "--conversation" not in first
    assert "--effort" not in first, "agy takes low/medium/high only: FI's own mapping leaves xhigh out"
    assert "--sandbox" in first and first[first.index("--mode") + 1] == "accept-edits"
    again = runner.argv(cwd=tmp_path, session_id="conv-1", resume=True, timeout_s=600)
    assert again[again.index("--conversation") + 1] == "conv-1"
    envelope = json.loads(runner.stdin("do it").splitlines()[0])
    assert envelope == {"event": "user", "message": {"role": "user", "content": "do it"}}


def test_codex_output_gives_the_thread_and_the_last_message_and_a_failed_turn_is_an_error() -> None:
    lines = "\n".join(json.dumps(e) for e in [
        {"type": "thread.started", "thread_id": "t-9"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "last"}},
        {"type": "turn.completed", "usage": {}}])
    assert ar.CodexAgentRunner().parse("noise\n" + lines) == {"session_id": "t-9", "message": "last",
                                                               "is_error": False, "finished": True, "events": 4}
    failed = ar.CodexAgentRunner().parse(json.dumps({"type": "turn.failed", "error": {"message": "usage limit"}}))
    assert failed["is_error"] and failed["message"] == "usage limit" and failed["session_id"] == ""


def test_agy_output_gives_the_conversation_and_falls_back_to_the_last_steps_words() -> None:
    steps = [{"event": "init", "conversation_id": "c-1", "init": {}},
             {"event": "step_update", "step_update": {"step_index": 3, "step_type": "agent_response",
                                                      "text_delta": "early words"}},
             {"event": "step_update", "step_update": {"step_index": 6, "step_type": "agent_response",
                                                      "text_delta": "late "}},
             {"event": "step_update", "step_update": {"step_index": 6, "step_type": "agent_response",
                                                      "text_delta": "words"}}]
    text = "\n".join(json.dumps(e) for e in steps)
    parsed = ar.AgyAgentRunner().parse(text)
    assert parsed["session_id"] == "c-1" and parsed["message"] == "late words" and not parsed["is_error"]
    assert not parsed["finished"], "no result event: the turn is not known to be over"
    done = ar.AgyAgentRunner().parse(text + "\n" + json.dumps(
        {"event": "result", "result": {"conversation_id": "c-1", "status": "SUCCESS", "response": "all done"}}))
    assert done["message"] == "all done"
    bad = ar.AgyAgentRunner().parse(json.dumps({"event": "result", "result": {"status": "ERROR"}}))
    assert bad["is_error"] and bad["message"] == "ERROR"


# --- choosing the runner -------------------------------------------------------------------------------------------------


async def test_the_code_writing_steps_provider_chooses_the_agent(tmp_path: Path) -> None:
    """Old code: every provider got the Claude CLI, so a person without it could not use agent mode."""
    codex = await ac.make_runner(tmp_path / "q", provider="codex_cli", model="m", effort="low")
    assert isinstance(codex, ar.CodexAgentRunner) and codex.model == "m" and codex.effort == "low"
    assert isinstance(await ac.make_runner(tmp_path / "q", provider="antigravity_cli"), ar.AgyAgentRunner)
    claude = await ac.make_runner(tmp_path / "q", provider="claude_cli", use_docker=False)
    assert isinstance(claude, ac.ClaudeCliRunner) and claude.mode == "host"
    assert isinstance(await ac.make_runner(tmp_path / "q", use_docker=False), ac.ClaudeCliRunner)


def test_the_implement_steps_own_provider_wins_over_the_main_one() -> None:
    main = SimpleNamespace(name="claude_cli", reasoning_effort="max", node_providers={})
    eng = SimpleNamespace(config=SimpleNamespace(provider=main))
    assert AgentBuildMixin._code_agent_provider(eng) == ("claude_cli", "max")
    main.node_providers = {"implement": SimpleNamespace(name="codex_cli", reasoning_effort="medium")}
    assert AgentBuildMixin._code_agent_provider(eng) == ("codex_cli", "medium")
    main.node_providers = {"review": SimpleNamespace(name="antigravity_cli", reasoning_effort=None)}
    assert AgentBuildMixin._code_agent_provider(eng) == ("claude_cli", "max")


# --- a whole round through FI's agent loop -------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,cls", [("codex", ar.CodexAgentRunner), ("agy", ar.AgyAgentRunner)])
async def test_a_round_and_its_continuation_run_on_the_copy(quest: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, kind: str, cls) -> None:
    s = _standin(tmp_path, monkeypatch, kind)
    s.plan([{"write": "simulate.py", "text": GOOD_SIM}])
    runner = cls(launcher=s.launcher, work_root=tmp_path / "work")
    coder = ac.AgentCoder(quest, runner, max_fixes=2, timeout_s=60)
    res = await coder.write("write the code")
    assert res.ok and res.copied == ["simulate.py"], res
    assert (quest / "code" / "simulate.py").read_text(encoding="utf-8") == GOOD_SIM
    first = s.calls()[0]
    assert first["prompt"] == "write the code" and "write the code" not in " ".join(first["argv"])
    assert Path(first["cwd"]).resolve() == runner.workdir(quest).resolve()
    if kind == "codex":
        assert not first["openai_key"], "the agent runs on the person's own login, never FI's key"
    session = coder.load()["session_id"]
    assert session == ("thread-new" if kind == "codex" else "conv-new")
    s.plan([{"write": "simulate.py", "text": GOOD_SIM + "# fixed\n"}])
    again = await coder.continue_("the check failed")
    assert again.ok and again.copied == ["simulate.py"]
    second = s.calls()[1]
    assert (session in second["argv"]) and second["cwd"] == first["cwd"]
    assert ("resume" in second["argv"]) if kind == "codex" else ("--conversation" in second["argv"])


async def test_a_failed_turn_is_a_failed_round_and_a_lost_session_starts_again(quest: Path, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    s = _standin(tmp_path, monkeypatch, "codex")
    runner = ar.CodexAgentRunner(launcher=s.launcher, work_root=tmp_path / "work")
    s.plan([{"fail": "usage limit reached"}])
    out = await runner.run("task", cwd=tmp_path / "w", session_id=None, timeout_s=60)
    assert out.exit_code == 1 and out.message == "usage limit reached" and out.session_id == ""
    coder = ac.AgentCoder(quest, runner, timeout_s=60)
    s.plan([{"write": "simulate.py", "text": GOOD_SIM}])
    assert (await coder.write("the whole task")).ok
    # The CLI cannot continue the thread (exit 3, nothing reported); a new thread from the whole task works.
    s.plan([{"exit_on_resume": 3}, {"write": "simulate.py", "text": GOOD_SIM + "# again\n"}])
    again = await coder.continue_("the check failed")
    calls = s.calls()
    assert "resume" in calls[-2]["argv"]
    assert "resume" not in calls[-1]["argv"] and calls[-1]["prompt"].startswith("the whole task")
    assert again.ok and again.copied == ["simulate.py"]
    assert (quest / "code" / "simulate.py").read_text(encoding="utf-8").endswith("# again\n")
    record = coder.load()
    assert record["rounds"][-1]["resume_failed"] is True and record["session_id"] == "thread-new"


async def test_a_round_past_its_limit_is_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _standin(tmp_path, monkeypatch, "agy")
    s.plan([{"sleep": 30}])
    out = await ar.AgyAgentRunner(launcher=s.launcher).run("task", cwd=tmp_path / "w", session_id=None, timeout_s=3)
    assert out.timed_out and out.exit_code == -1


def test_a_missing_program_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ar.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="codex login"):
        ar.CodexAgentRunner()._program()


def test_codex_gets_no_windows_app_shortcut_folder_on_its_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: the Store's pwsh.exe came first on the PATH, cannot start in Codex's sandbox, and every command the agent
    ran failed ("CreateProcessAsUserW failed: 5")."""
    shortcut = r"C:\Users\someone\AppData\Local\Microsoft\WindowsApps"
    real = r"C:\Program Files\PowerShell\7"
    monkeypatch.setattr(ar, "_WINDOWS", True)
    monkeypatch.setattr(ar, "scrub_secrets", lambda: {"PATH": ";".join(
        [shortcut + r"\Microsoft.PowerShell_8wekyb3d8bbwe", real, shortcut])})
    assert ar.CodexAgentRunner().process_env(Path("w"))["PATH"] == real
    assert ar._is_app_shortcut_dir("c:/users/x/appdata/local/microsoft/windowsapps/")
    assert not ar._is_app_shortcut_dir(r"C:\Program Files\WindowsApps Tools")


def test_agy_runs_in_a_home_of_the_quests_own_whose_settings_refuse_the_web_and_mcp(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: agy followed the person's own settings.json (their MCP servers, their grants)."""
    monkeypatch.setenv("ANTIGRAVITY_PERM_GRANTS", "command(*)")
    runner = ar.AgyAgentRunner()
    cwd = tmp_path / "id" / "code"
    env = runner.process_env(cwd)
    home = Path(env["HOME"])
    assert home == tmp_path / "id_agy_home" and env["USERPROFILE"] == str(home)
    assert "ANTIGRAVITY_PERM_GRANTS" not in env
    settings = json.loads((home / ".gemini" / "antigravity-cli" / "settings.json").read_text(encoding="utf-8"))
    assert settings["allowNonWorkspaceAccess"] is False
    assert {"read_url(*)", "mcp(*)"} <= set(settings["permissions"]["deny"])
    (home / "conversation.db").write_text("kept", encoding="utf-8")
    runner.process_env(cwd)
    assert (home / "conversation.db").is_file(), "a later round continues the conversation agy keeps there"
    runner.close()
    assert home.is_dir()
    runner.close(final=True)
    assert not home.exists()


async def test_agy_without_the_conversation_starts_again_from_the_whole_task(quest: Path, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: a quest whose earlier rounds were another agent's sent agy that agent's session id; agy said
    `conversation not found`, exited 0 and worked from the fix request alone, without the task."""
    s = _standin(tmp_path, monkeypatch, "agy")
    runner = ar.AgyAgentRunner(launcher=s.launcher, work_root=tmp_path / "work")
    coder = ac.AgentCoder(quest, runner, timeout_s=60)
    s.plan([{"write": "simulate.py", "text": GOOD_SIM}])
    assert (await coder.write("the whole task")).ok
    record = coder.load()
    record["session_id"] = "another-agents-session"
    (quest / ".fi" / "agent_coder.json").write_text(json.dumps(record), encoding="utf-8")
    s.plan([{"write": "simulate.py", "text": GOOD_SIM + "# fixed\n"}])
    again = await coder.continue_("the check failed")
    calls = s.calls()
    assert len(calls) == 2, "agy is not started on a conversation it does not have"
    assert "--conversation" not in calls[-1]["argv"] and calls[-1]["prompt"].startswith("the whole task")
    assert again.ok and coder.load()["rounds"][-1]["resume_failed"] is True


@pytest.mark.parametrize("kind,cls", [("codex", ar.CodexAgentRunner), ("agy", ar.AgyAgentRunner)])
@pytest.mark.parametrize("said", ["", "a diagnostic line, not an event", '{"type": "thread.started", "thread_id": "t"}'])
async def test_exit_0_without_the_end_of_turn_is_not_a_done_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                  kind: str, cls, said: str) -> None:
    """Old code: a CLI that printed nothing (or a cut-off stream) and exited 0 was a round that worked."""
    s = _standin(tmp_path, monkeypatch, kind)
    s.plan([{"say": said}])
    out = await cls(launcher=s.launcher).run("task", cwd=tmp_path / "w", session_id=None, timeout_s=60)
    assert out.exit_code == 1 and "end of its turn" in out.message, out
    if said.startswith("a diagnostic"):
        assert said in out.message, "what the CLI printed is kept for the person"


def test_an_agy_result_without_its_status_is_not_a_success() -> None:
    parsed = ar.AgyAgentRunner().parse(json.dumps({"event": "result", "result": {"response": "partial"}}))
    assert parsed["finished"] and parsed["is_error"]


@pytest.mark.parametrize("make", [lambda launcher: ar.CodexAgentRunner(launcher=launcher),
                                  lambda launcher: ac.ClaudeCliRunner(mode="host", launcher=launcher)])
async def test_a_cancelled_round_stops_the_whole_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make) -> None:
    """Old code: the tree was closed as not aborted, so on POSIX a helper the CLI left holding the pipe kept running."""
    import asyncio

    from core import proc_tree

    seen: list[bool] = []
    real = proc_tree.AsyncProcessTree.aclose

    async def spy(self, *, aborted: bool = False) -> None:
        seen.append(aborted)
        await real(self, aborted=aborted)

    monkeypatch.setattr(proc_tree.AsyncProcessTree, "aclose", spy)
    s = _standin(tmp_path, monkeypatch, "codex")
    s.plan([{"sleep": 30}])
    task = asyncio.ensure_future(make(s.launcher).run("task", cwd=tmp_path / "w", session_id=None, timeout_s=60))
    await asyncio.sleep(2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [True]
    s.plan([])
    await ar.CodexAgentRunner(launcher=s.launcher).run("task", cwd=tmp_path / "w", session_id=None, timeout_s=60)
    assert seen == [True, False], "a round that ended by itself is closed normally"


def test_the_end_of_a_quest_removes_the_agents_scratch_and_a_stop_keeps_it(tmp_path: Path) -> None:
    """Old code: nothing called AgentCoder.close, so every quest left its scratch copy (and agy's home) behind."""
    quest = tmp_path / "quest"
    (quest / "code").mkdir(parents=True)
    runner = ar.AgyAgentRunner(work_root=tmp_path / "work")
    coder = ac.AgentCoder(quest, runner, timeout_s=60)
    work = runner.workdir(quest)
    work.mkdir(parents=True)
    runner.process_env(work)
    home = runner.home(work)
    eng = SimpleNamespace(_agent_coder_obj=coder, _quest_recorded=False, _log=logging.getLogger("t"))
    AgentBuildMixin._close_code_agent(eng)
    assert work.is_dir() and home.is_dir(), "a run that did not reach its end keeps what a resume needs"
    eng._quest_recorded = True
    AgentBuildMixin._close_code_agent(eng)
    assert not work.parent.exists() and not home.exists()
    AgentBuildMixin._close_code_agent(SimpleNamespace(_log=logging.getLogger("t")))   # no agent: nothing to do
    # After a resume the runner is a new one that never started agy: the quest's end still removes the home.
    work.mkdir(parents=True)
    home.mkdir(parents=True)
    resumed = ac.AgentCoder(quest, ar.AgyAgentRunner(work_root=tmp_path / "work"), timeout_s=60)
    AgentBuildMixin._close_code_agent(SimpleNamespace(_agent_coder_obj=resumed, _quest_recorded=True,
                                                      _log=logging.getLogger("t")))
    assert not work.parent.exists() and not home.exists()


def test_the_engine_closes_the_agent_on_every_way_a_run_ends() -> None:
    import inspect

    from core.engine import Engine

    source = inspect.getsource(Engine.run)
    assert source.index("self._close_code_agent()") > source.rindex("finally:")
