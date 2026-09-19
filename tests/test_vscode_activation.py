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
    }
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
    fi.mkdir()
    project.mkdir()
    (fi / "launch.py").write_text("", encoding="utf-8")
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


def test_without_a_workspace_the_commands_say_what_is_missing(tmp_path: Path) -> None:
    got = _run(tmp_path, commands=["watch", "resume", "skills"], workspace=None, settings={})
    for cmd, reply in got["replies"].items():
        assert "No workspace open" in reply, (cmd, reply)


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


def test_a_workspace_that_is_not_fi_and_names_no_fi_folder_is_told_how_to_fix_it(
    tmp_path: Path, folders,
) -> None:
    got = _run(tmp_path, commands=["resume"], workspace=str(folders["project"]), settings={})
    assert "no `launch.py`" in got["replies"]["resume"]
    assert "frontierInsight.repoPath" in got["replies"]["resume"]
