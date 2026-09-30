"""The compiled VSCode extension loads, activates and routes chat commands.

Runs the shipped bundle (``out/extension.js``, the thing the .vsix contains) in real
Node against a stand-in ``vscode`` module, with a real workspace folder. It checks
that activation registers the ``@fi`` participant, and that commands resolve the FI
folder and the user's work folder separately (a project workspace, FI elsewhere)
without spawning Python. Skips without Node or a current compile, like the other
extension tests; CI's own job compiles it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
BUNDLE = EXT / "out" / "extension.js"
SMOKE = Path(__file__).resolve().parent / "fixtures" / "vscode_activate_smoke.js"


def _run(tmp_path: Path, **arg) -> dict:  # noqa: ANN003
    node = shutil.which("node")
    if node is None or not BUNDLE.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/extension.js")
    newest_src = max(p.stat().st_mtime for p in (EXT / "src").glob("*.ts"))
    if BUNDLE.stat().st_mtime < newest_src:
        pytest.skip("out/ is older than the sources; run npm run compile")
    env = {
        **os.environ,
        # The extension opens a per-user pipe/socket; keep the test off the real one
        # a running VSCode may be holding.
        "USERNAME": f"fi_smoke_{os.getpid()}",
        "XDG_RUNTIME_DIR": str(tmp_path),
        # A picked FI folder is remembered under the home folder: never the real one.
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "FI_HOME": str(tmp_path / "home" / ".frontier-insight"),
    }
    # Unless a test says otherwise, the Python knows nothing of FI (the machine's own may).
    arg["settings"] = {"pythonPath": str(tmp_path / "no-such-python"), **arg.get("settings", {})}
    done = subprocess.run(
        [node, str(SMOKE), str(BUNDLE), json.dumps(arg)],
        capture_output=True, text=True, encoding="utf-8", timeout=90, env=env,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture()
def folders(tmp_path: Path) -> dict[str, Path]:
    fi = tmp_path / "FrontierInsight"
    project = tmp_path / "my_project"
    (fi / "core").mkdir(parents=True)
    project.mkdir()
    (fi / "launch.py").write_text("", encoding="utf-8")
    (fi / "core" / "engine.py").write_text("", encoding="utf-8")
    return {"fi": fi, "project": project}


def test_the_bundle_activates_and_registers_the_fi_participant(tmp_path: Path) -> None:
    got = _run(tmp_path, commands=[], workspace=None, settings={})
    assert got["participant"] == "frontier-insight.fi"
    assert got["subscriptions"] >= 2, "the participant and the output channel are registered"


def test_declared_chat_commands_are_all_routed(tmp_path: Path) -> None:
    """Every command in package.json reaches a handler (none falls through to help)."""
    pkg = json.loads((EXT / "package.json").read_text(encoding="utf-8"))
    declared = [c["name"] for c in pkg["contributes"]["chatParticipants"][0]["commands"]]
    assert "watch" in declared and "resume" in declared
    ext_src = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    missing = [c for c in declared if f'cmd === "{c}"' not in ext_src]
    assert not missing, f"declared in package.json but never handled: {missing}"


def test_without_a_folder_open_the_commands_say_what_is_missing(tmp_path: Path, folders) -> None:
    """With no folder open there is nowhere to put a study: never FI's own checkout, even
    when FI is found (here, a folder picked before)."""
    saved = tmp_path / "home" / ".frontier-insight" / "fi_location.json"
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"path": str(folders["fi"])}), encoding="utf-8")
    got = _run(tmp_path, commands=["watch", "resume", "plan", "skills"], workspace=None, settings={})
    for cmd, reply in got["replies"].items():
        assert "No folder open" in reply, (cmd, reply)
    assert got["pickerCalls"] == 0


def test_when_fi_cannot_be_found_the_commands_say_how_to_fix_it(tmp_path: Path, folders) -> None:
    got = _run(tmp_path, commands=["watch", "skills"], workspace=str(folders["project"]), settings={})
    for cmd, reply in got["replies"].items():
        assert "FrontierInsight was not found" in reply and "pip install -e" in reply, (cmd, reply)


def test_two_commands_at_once_ask_for_fi_only_once(tmp_path: Path, folders) -> None:
    got = _run(
        tmp_path, commands=["resume", "watch", "plan"], workspace=str(folders["project"]), settings={},
        picked=str(folders["fi"]), concurrent=True,
    )
    assert got["pickerCalls"] == 1, "two commands started together both opened the picker"
    for cmd in ("resume", "watch", "plan"):
        assert str(folders["project"] / "outputs") in got["replies"][cmd], got["replies"][cmd]


def test_a_project_workspace_uses_its_own_outputs_not_fis(tmp_path: Path, folders) -> None:
    """FI lives elsewhere (repoPath); the quests, and where /resume and /watch
    look for them, are the workspace's."""
    got = _run(
        tmp_path, commands=["resume", "watch"], workspace=str(folders["project"]),
        settings={"repoPath": str(folders["fi"])},
    )
    want = str(folders["project"] / "outputs")
    for cmd in ("resume", "watch"):
        assert want in got["replies"][cmd], got["replies"][cmd]
        assert str(folders["fi"] / "outputs") not in got["replies"][cmd]


def test_watch_lists_only_quests_that_are_waiting_on_a_job(tmp_path: Path, folders) -> None:
    quest = folders["project"] / "outputs" / "1700000000-x-abcdef"
    (quest / ".fi").mkdir(parents=True)
    (quest / ".fi" / "state.sqlite").write_bytes(b"")
    got = _run(
        tmp_path, commands=["watch"], workspace=str(folders["project"]),
        settings={"repoPath": str(folders["fi"])},
    )
    assert "waiting on a background job" in got["replies"]["watch"]


def test_a_project_folder_finds_fi_with_no_setting_and_asks_at_most_once(
    tmp_path: Path, folders,
) -> None:
    """Nothing set: the one folder picker is the last resort, its answer is remembered in
    ~/.frontier-insight/fi_location.json (not a VS Code setting), and the quests are the
    open folder's."""
    got = _run(
        tmp_path, commands=["resume", "watch"], workspace=str(folders["project"]), settings={},
        picked=str(folders["fi"]),
    )
    assert got["pickerCalls"] == 1, "asked once for the whole session, not once per command"
    for cmd in ("resume", "watch"):
        assert str(folders["project"] / "outputs") in got["replies"][cmd], got["replies"][cmd]
    saved = tmp_path / "home" / ".frontier-insight" / "fi_location.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == {"path": str(folders["fi"])}
    # A new window (a second study folder) finds it without asking.
    other = tmp_path / "study_b"
    other.mkdir()
    got = _run(tmp_path, commands=["resume"], workspace=str(other), settings={})
    assert got["pickerCalls"] == 0 and str(other / "outputs") in got["replies"]["resume"]


def test_a_cancelled_picker_is_told_how_to_fix_it(tmp_path: Path, folders) -> None:
    got = _run(tmp_path, commands=["resume"], workspace=str(folders["project"]), settings={})
    assert "FrontierInsight was not found" in got["replies"]["resume"]
    assert got["pickerCalls"] == 1


def test_plan_lists_only_quests_that_have_written_a_plan(tmp_path: Path, folders) -> None:
    quest = folders["project"] / "outputs" / "1700000000-x-abcdef"
    (quest / ".fi").mkdir(parents=True)
    (quest / ".fi" / "state.sqlite").write_bytes(b"")
    got = _run(
        tmp_path, commands=["plan"], workspace=str(folders["project"]),
        settings={"repoPath": str(folders["fi"])},
    )
    assert "has a plan yet" in got["replies"]["plan"]


def test_plan_with_a_quest_id_opens_plan_md_and_says_how_to_change_it_and_run_it(
    tmp_path: Path, folders,
) -> None:
    quest_id = "1700000000-x-abcdef"
    quest = folders["project"] / "outputs" / quest_id
    (quest / ".fi").mkdir(parents=True)
    (quest / ".fi" / "state.sqlite").write_bytes(b"")
    (quest / "config.yaml").write_text("topic: x\n", encoding="utf-8")
    (quest / "plan.md").write_text("# Plan\n", encoding="utf-8")
    got = _run(
        tmp_path, commands=["plan"], workspace=str(folders["project"]),
        settings={"repoPath": str(folders["fi"])}, prompts={"plan": quest_id},
    )
    reply = got["replies"]["plan"]
    assert f"@fi /plan {quest_id} <what to change>" in reply
    assert f"@fi /resume {quest_id}" in reply
    assert got["opened"] == [str(quest / "plan.md")], "the file is opened for editing"
