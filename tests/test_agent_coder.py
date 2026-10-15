"""The coding agent's runner and guard (core/agent_coder.py), with stand-ins for the ``claude`` and ``docker`` programs.

No model is called and no container is started here: the stand-ins act as the agent would (write files, climb out of
their folder, edit FI's tests) and print the CLI's stream-json events. The one test that needs a running Docker daemon
skips when it is down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest

from core import agent_coder as ac
from core.execution import scrub_secrets

SESSION = "bb9bf6bf-972b-47f7-91c8-543edb8794e5"

# The shape `claude -p --output-format stream-json --verbose` printed in the one smoke call (2.1.299, host mode), with
# only the fields FI reads kept.
STREAM = "\n".join(json.dumps(e) for e in [
    {"type": "system", "subtype": "ui_invalidate", "session_id": SESSION},
    {"type": "system", "subtype": "init", "session_id": SESSION, "permissionMode": "acceptEdits",
     "tools": ["Bash", "Edit", "Glob", "Grep", "Read", "Write"]},
    {"type": "assistant", "session_id": SESSION, "message": {"content": [{"type": "text", "text": "Running it."}]}},
    {"type": "user", "session_id": SESSION, "message": {"content": [{"type": "tool_result", "content": "4"}]}},
    {"type": "result", "subtype": "success", "is_error": False, "session_id": SESSION, "result": "4", "num_turns": 2},
])

# A stand-in for `claude`: reads the prompt on stdin, does what the plan file says (paths relative to its working
# folder, or absolute), records its arguments, prints stream-json events.
STANDIN = r'''
import json, os, sys, time
from pathlib import Path
plan = json.loads(Path(os.environ["FI_STANDIN_PLAN"]).read_text(encoding="utf-8"))
log = Path(os.environ["FI_STANDIN_LOG"])
prompt = sys.stdin.read()
with log.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "prompt": prompt[:200],
                        "token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", ""),
                        "api_key": "ANTHROPIC_API_KEY" in os.environ}) + "\n")
args = sys.argv[1:]
session = args[args.index("--resume") + 1] if "--resume" in args else args[args.index("--session-id") + 1]
for step in plan:
    if "sleep" in step:
        time.sleep(step["sleep"])
    if "write" in step:
        p = Path(step["write"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(step["text"], encoding="utf-8")
    if "delete" in step:
        Path(step["delete"]).unlink()
    if "exit" in step:
        sys.exit(step["exit"])
print(json.dumps({"type": "system", "subtype": "init", "session_id": session}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                  "result": "done"}))
'''

GOOD_SIM = "def run_trial(cell, trial_id, seed):\n    return {'x': float(seed % 7)}\n"


@pytest.fixture
def quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    code = root / "code"
    (code / "tests").mkdir(parents=True)
    (code / "simulate.py").write_text("def run_trial(cell, trial_id, seed):\n    return {}\n", encoding="utf-8")
    (code / "run.py").write_text("# FI's run.py\n", encoding="utf-8")
    (code / "tests" / "test_oracles.py").write_text("# FI's loader\n", encoding="utf-8")
    (code / ".git").mkdir()
    (code / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".fi" / "checks").mkdir(parents=True)
    (root / ".fi" / "checks" / "oracle_values.json").write_text('{"c": {"expected": 0.7385113}}', encoding="utf-8")
    (root / "plan.md").write_text("# plan\n", encoding="utf-8")
    return root


@pytest.fixture
def standin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / "standin" / "claude_standin.py"
    script.parent.mkdir()
    script.write_text(STANDIN, encoding="utf-8")
    plan, log = tmp_path / "standin" / "plan.json", tmp_path / "standin" / "log.jsonl"
    monkeypatch.setenv("FI_STANDIN_PLAN", str(plan))
    monkeypatch.setenv("FI_STANDIN_LOG", str(log))

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


def _host(tmp_path: Path, standin) -> ac.ClaudeCliRunner:
    return ac.ClaudeCliRunner(mode="host", launcher=standin.launcher, work_root=tmp_path / "work")


# --- the CLI's output and command ------------------------------------------------------------------------------------------


def test_the_stream_gives_the_session_and_the_last_message() -> None:
    parsed = ac.parse_stream(STREAM)
    assert parsed == {"session_id": SESSION, "message": "4", "is_error": False, "events": 5}
    error = ac.parse_stream(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True,
                                        "session_id": SESSION}))
    assert error["is_error"] is True
    # No result event: the last text an assistant event carried.
    assert ac.parse_stream(STREAM.rsplit("\n", 1)[0])["message"] == "Running it."
    assert ac.parse_stream("not json\n")["events"] == 0


def test_the_existing_stream_fixture_parses() -> None:
    text = (Path(__file__).parent / "fixtures" / "claude_stream_invented_tool_calls.jsonl").read_text(encoding="utf-8")
    assert ac.parse_stream(text)["events"] > 0


def test_without_docker_the_cli_is_kept_to_files_and_python_and_never_bypasses_permissions(tmp_path: Path) -> None:
    runner = ac.ClaudeCliRunner(mode="host", launcher=["claude"], model="sonnet", effort="high")
    argv = runner.argv(code_dir=tmp_path, name="", session_id=SESSION, resume=False)
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == "Read,Edit,Write,Glob,Grep,Bash"
    assert "WebFetch" not in " ".join(argv) and "WebSearch" not in " ".join(argv)
    assert "--restricted" in argv and argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert "bypassPermissions" not in argv and "--add-dir" not in argv and "--bare" not in argv
    assert "Bash(python:*)" in argv
    assert argv[argv.index("--session-id") + 1] == SESSION and "--resume" not in argv
    assert argv[argv.index("--model") + 1] == "sonnet" and argv[argv.index("--effort") + 1] == "high"
    again = runner.argv(code_dir=tmp_path, name="", session_id=SESSION, resume=True)
    assert again[again.index("--resume") + 1] == SESSION and "--session-id" not in again
    assert runner.works_on_copy and not runner.confined


def test_the_copy_lives_outside_the_quest_and_is_the_same_every_round(tmp_path: Path) -> None:
    runner = ac.ClaudeCliRunner(mode="host", work_root=tmp_path / "work")
    quest = tmp_path / "out" / "q1"
    work = runner.workdir(quest)
    assert work == runner.workdir(quest) and work.name == "code"
    assert quest.resolve() not in work.resolve().parents
    assert runner.workdir(tmp_path / "other" / "q1") != work, "two quests of the same name never share a copy"


def test_in_docker_the_token_goes_by_name_only(tmp_path: Path) -> None:
    runner = ac.ClaudeCliRunner(mode="docker", home_dir=tmp_path / "home", docker=["docker"],
                                environ={ac.TOKEN_ENV: "sk-ant-oat01-SECRET"})
    argv = runner.argv(code_dir=tmp_path / "code", name="fi-agent-x", session_id=SESSION, resume=False)
    assert "sk-ant-oat01-SECRET" not in " ".join(argv)
    assert argv[argv.index(ac.TOKEN_ENV) - 1] == "-e"
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert len(mounts) == 2 and all(m.startswith("type=bind,source=") for m in mounts)
    assert f"target={ac.CONTAINER_CODE}" in mounts[0] and str(tmp_path / "home") in mounts[1]
    for flag in ("--cap-drop", "--security-opt", "--memory", "--pids-limit", "--user"):
        assert flag in argv
    assert "bypassPermissions" in argv and "ANTHROPIC_API_KEY" not in " ".join(argv)
    assert runner.process_env()[ac.TOKEN_ENV] == "sk-ant-oat01-SECRET"
    with pytest.raises(RuntimeError) as missing:
        ac.ClaudeCliRunner(mode="docker", home_dir=tmp_path, environ={}).process_env()
    assert "claude setup-token" in str(missing.value)


def test_no_secret_reaches_the_host_cli_or_the_quests_own_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ac.TOKEN_ENV, "tok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    env = ac.ClaudeCliRunner(mode="host").process_env()
    assert ac.TOKEN_ENV not in env and "ANTHROPIC_API_KEY" not in env, "the host CLI uses the person's own login"
    assert ac.TOKEN_ENV not in scrub_secrets(), "FI's code runs never see the agent's token"


# --- Docker on this machine ---------------------------------------------------------------------------------------------------


def _fake(tmp_path: Path, name: str, body: str) -> list[str]:
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return [sys.executable, "-B", str(path)]


async def test_docker_status_says_running_stopped_or_missing(tmp_path: Path) -> None:
    ok = _fake(tmp_path, "ok", "print('27.1.1')\n")
    stopped = _fake(tmp_path, "stopped", "import sys\nsys.stderr.write('cannot connect')\nsys.exit(1)\n")
    assert (await ac.docker_status(ok))[0] == ac.DOCKER_OK
    state, why = await ac.docker_status(stopped)
    assert state == ac.DOCKER_STOPPED and "not running" in why
    assert (await ac.docker_status([str(tmp_path / "no-such-docker.exe")]))[0] == ac.DOCKER_MISSING


async def test_without_a_running_docker_the_copy_is_the_normal_path_and_says_so(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    stopped = _fake(tmp_path, "stopped", "import sys\nsys.exit(1)\n")
    with caplog.at_level(logging.INFO):
        runner = await ac.make_runner(tmp_path / "q", docker=stopped)
    assert runner.mode == "host" and ac.NO_DOCKER_LINE in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], "information, not a warning"
    ok = _fake(tmp_path, "ok", "print('27.1.1')\n")
    docker = await ac.make_runner(tmp_path / "q", docker=ok)
    assert docker.mode == "docker" and (tmp_path / "q").resolve() not in docker.home_dir.parents
    assert (await ac.make_runner(tmp_path / "q", docker=ok, use_docker=False)).mode == "host"


# --- the no-Docker round: a copy of code/, compared and copied back --------------------------------------------------------


async def test_a_round_works_on_a_copy_and_only_the_agents_files_come_back(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM}, {"write": "pkg/model.py", "text": "def f(x):\n    return x\n"},
                  {"write": "figures/f.png", "text": "png"}])
    coder = ac.AgentCoder(quest, _host(tmp_path, standin), max_fixes=2, timeout_s=60)
    res = await coder.write("write the code")
    assert res.ok and not res.cheats and not res.tampered
    assert sorted(res.copied) == ["pkg/model.py", "simulate.py"]
    code = quest / "code"
    assert (code / "simulate.py").read_text(encoding="utf-8") == GOOD_SIM
    assert not (code / "figures").exists(), "the agent's own run outputs stay in the copy"
    call = standin.calls()[0]
    work = _host(tmp_path, standin).workdir(quest)
    assert Path(call["cwd"]).resolve() == work.resolve() and quest.resolve() not in work.resolve().parents
    assert not (work / ".git").exists(), "the copy has no history"
    assert not any("0.7385113" in p.read_text(encoding="utf-8", errors="replace")
                   for p in work.rglob("*") if p.is_file()), "the copy never holds FI's check values"
    # The next round continues the same session, in the same folder, on a fresh copy of code/.
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM + "# fixed\n"}])
    again = await coder.continue_("the check failed")
    assert again.ok and again.copied == ["simulate.py"]
    second = standin.calls()[1]
    assert second["argv"][second["argv"].index("--resume") + 1] == coder.load()["session_id"]
    assert second["cwd"] == call["cwd"]
    record = coder.load()
    assert [r["mode"] for r in record["rounds"]] == ["copy", "copy"]
    assert record["rounds"][0]["copied"] == ["pkg/model.py", "simulate.py"]


async def test_an_edit_of_fis_tests_is_kept_out_and_recorded(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"write": "tests/test_oracles.py", "text": "def test_x():\n    pass\n"},
                  {"write": "tests/test_extra.py", "text": "x = 1\n"}, {"write": "simulate.py", "text": GOOD_SIM}])
    res = await ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60).write("task")
    code = quest / "code"
    assert (code / "tests" / "test_oracles.py").read_text(encoding="utf-8") == "# FI's loader\n"
    assert not (code / "tests" / "test_extra.py").exists()
    assert sorted(res.tampered) == ["code/tests/test_extra.py (added)", "code/tests/test_oracles.py (changed)"]
    assert res.copied == ["simulate.py"] and res.ok


async def test_a_write_beside_the_copy_is_a_cheat_and_nothing_is_kept(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM}, {"write": "../x", "text": "escaped"}])
    res = await ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60).write("task")
    assert not res.ok and res.cheats == ["x (written beside its copy of code/, outside it)"]
    assert res.copied == [] and "def run_trial(cell, trial_id, seed):\n    return {}" in (
        quest / "code" / "simulate.py").read_text(encoding="utf-8")
    assert not (_host(tmp_path, standin).workdir(quest).parent / "x").exists()


async def test_a_write_into_the_quest_folder_is_a_cheat_and_is_put_back(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"write": str(quest / "plan.md"), "text": "# my plan\n"},
                  {"write": str(quest / "code" / "simulate.py"), "text": GOOD_SIM},
                  {"write": str(quest / ".fi" / "checks" / "new.json"), "text": "{}"}])
    res = await ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60).write("task")
    assert not res.ok and len(res.cheats) == 3
    assert (quest / "plan.md").read_text(encoding="utf-8") == "# plan\n"
    assert "return {}" in (quest / "code" / "simulate.py").read_text(encoding="utf-8")
    assert not (quest / ".fi" / "checks" / "new.json").exists()
    assert coder_status(quest) == "cheat"


def coder_status(quest: Path) -> str:
    return json.loads((quest / ac.RECORD).read_text(encoding="utf-8"))["rounds"][-1]["status"]


async def test_code_that_reads_fis_check_values_is_a_cheat(quest: Path, tmp_path: Path, standin) -> None:
    reader = ("import json\nfrom pathlib import Path\n\n"
              "WANT = json.loads((Path(__file__).resolve().parent.parent / '.fi' / 'checks' / 'oracle_values.json')"
              ".read_text())\n\n"
              "def run_trial(cell, trial_id, seed):\n    return {'c': WANT['c']['expected']}\n")
    standin.plan([{"write": "simulate.py", "text": reader}])
    res = await ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60).write("task")
    assert not res.ok and res.copied == []
    assert any("oracle_values" in c for c in res.cheats) and any(".parent.parent" in c for c in res.cheats)


def test_reaches_outside_finds_the_ways_out_and_spares_ordinary_code() -> None:
    assert ac.reaches_outside("open('../x', 'w')")
    assert ac.reaches_outside("p = 'C:\\\\Users\\\\me\\\\q\\\\.fi\\\\checks'")
    assert ac.reaches_outside("import os\nos.path.dirname(os.path.dirname(__file__))")
    assert ac.reaches_outside("from pathlib import Path\nPath.home()")
    assert ac.reaches_outside("import os\nos.environ['FI_CHECKS_DIR']")
    assert ac.reaches_outside("from pathlib import Path\nPath(__file__).parents[2]")
    ordinary = ("import os, json\nfrom pathlib import Path\nHERE = Path(__file__).resolve().parent\n"
                "out = Path(os.environ.get('FI_RAW_DIR', 'raw'))\nlabel = 'k / s'\nfits = 'model.fits'\n"
                "t = json.loads(os.environ.get('FI_THRESHOLDS', '{}'))\n")
    assert ac.reaches_outside(ordinary) == []


async def test_a_round_that_overruns_its_cap_is_stopped(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"sleep": 30}])
    res = await ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=3).write("task")
    assert res.timed_out and not res.ok


async def test_a_lost_session_starts_again_from_the_whole_task(quest: Path, tmp_path: Path, standin) -> None:
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM}])
    coder = ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60)
    assert (await coder.write("the whole task")).ok
    standin.plan([{"exit": 3}])   # the CLI refuses to continue and reports nothing
    lost = await coder.continue_("the check failed")
    calls = standin.calls()
    assert len(calls) == 3 and "--session-id" in calls[2]["argv"] and calls[2]["prompt"].startswith("the whole task")
    assert coder.load()["rounds"][-1]["resume_failed"] is True and not lost.ok


async def test_the_scratch_copy_is_removed_when_the_quest_is_over(quest: Path, tmp_path: Path, standin) -> None:
    coder = ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60)
    await coder.write("task")
    work = _host(tmp_path, standin).workdir(quest)
    coder.close()
    assert work.is_dir(), "kept for a resume"
    coder.close(final=True)
    assert not work.parent.exists()


# --- guarded rounds ---------------------------------------------------------------------------------------------------------


async def test_a_refused_round_is_undone_whole(quest: Path, tmp_path: Path, standin) -> None:
    coder = ac.AgentCoder(quest, _host(tmp_path, standin), timeout_s=60)
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM}])
    await coder.write("task")
    before = (quest / "code" / "simulate.py").read_text(encoding="utf-8")
    standin.plan([{"write": "simulate.py", "text": GOOD_SIM + "x = (\n"}])
    syntax = await ac.guarded_round(coder, "fix it")
    assert not syntax.ok and syntax.kind == "syntax"
    assert (quest / "code" / "simulate.py").read_text(encoding="utf-8") == before

    async def refuse(changed: list[str], message: str) -> str:
        return "the equation test still fails"

    standin.plan([{"write": "simulate.py", "text": GOOD_SIM + "# other\n"}, {"write": "new.py", "text": "y = 2\n"}])
    refused = await ac.guarded_round(coder, "fix it", accept=refuse)
    assert not refused.ok and refused.kind == "refused" and not (quest / "code" / "new.py").exists()
    assert (quest / "code" / "simulate.py").read_text(encoding="utf-8") == before


# --- Docker, with a stand-in `docker` (the real daemon is not needed) --------------------------------------------------------

FAKE_DOCKER = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["FI_STANDIN_LOG"])
if args[:2] == ["image", "inspect"] or args[:1] == ["kill"]:
    sys.exit(0)
assert args[0] == "run"
code = next(a.split("source=", 1)[1].split(",target=")[0] for a in args if a.startswith("type=bind") and "/work/code" in a)
with log.open("a", encoding="utf-8") as f:
    f.write(json.dumps({"argv": args, "token": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")}) + "\n")
sys.stdin.read()
Path(code, "simulate.py").write_text("def run_trial(cell, trial_id, seed):\n    return {'x': 1.0}\n", encoding="utf-8")
Path(code, "run.py").write_text("# changed by the agent\n", encoding="utf-8")
session = args[args.index("--session-id") + 1] if "--session-id" in args else args[args.index("--resume") + 1]
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session, "result": "ok"}))
'''


async def test_in_a_container_the_agent_works_in_code_and_fis_files_are_put_back(
        quest: Path, tmp_path: Path, standin) -> None:
    docker = _fake(tmp_path, "docker_standin", FAKE_DOCKER)
    runner = ac.ClaudeCliRunner(mode="docker", home_dir=tmp_path / "home", docker=docker,
                                environ={ac.TOKEN_ENV: "tok-123"})
    res = await ac.AgentCoder(quest, runner, timeout_s=60).write("task")
    assert res.ok and res.tampered == ["code/run.py (changed)"]
    assert (quest / "code" / "run.py").read_text(encoding="utf-8") == "# FI's run.py\n"
    assert "{'x': 1.0}" in (quest / "code" / "simulate.py").read_text(encoding="utf-8")
    call = standin.calls()[0]
    assert call["token"] == "tok-123" and "tok-123" not in " ".join(call["argv"])


# --- the real daemon ----------------------------------------------------------------------------------------------------------


def _daemon_running() -> bool:
    return asyncio.run(ac.docker_status())[0] == ac.DOCKER_OK


@pytest.mark.slow
@pytest.mark.skipif(not _daemon_running(), reason="the Docker daemon is not running on this machine")
def test_a_container_started_with_the_runners_flags_sees_only_code(tmp_path: Path) -> None:
    """With a running daemon: the runner's own `docker run` flags on a plain Python image, with `sh` in place of the
    CLI. It can write in code/ and sees nothing of the quest folder above it."""
    import subprocess

    code = tmp_path / "quest" / "code"
    code.mkdir(parents=True)
    (tmp_path / "quest" / "secret.txt").write_text("0.7385113", encoding="utf-8")
    runner = ac.ClaudeCliRunner(mode="docker", home_dir=tmp_path / "home", image="python:3.11-slim-bookworm",
                                launcher=["sh", "-c", "touch /work/code/ok; ls -a /work; cat /work/secret.txt || true",
                                          "sh"], environ={ac.TOKEN_ENV: "unused"})
    (tmp_path / "home").mkdir()
    argv = runner.argv(code_dir=code.resolve(), name="fi-agent-test", session_id=SESSION, resume=False)
    argv = argv[: argv.index("-p")]   # the stand-in takes no CLI arguments
    done = subprocess.run(argv, capture_output=True, text=True, timeout=600, env=runner.process_env(),
                          **ac._no_window())
    assert done.returncode == 0, done.stderr
    assert (code / "ok").is_file() and "0.7385113" not in done.stdout and "secret.txt" not in done.stdout
