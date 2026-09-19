"""The VSCode extension separates where FI is installed from where the user works.

Quests run in the workspace folder (the user's project), so a relative `outputs/`,
the YAML and the example files mean that project; FI's checkout only supplies
launch.py. Runs the compiled resolver in real Node against real folders, and pins
that no command still runs from FI's folder by accident."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
ROOTS_JS = EXT / "out" / "roots.js"
ROOTS_TS = EXT / "src" / "roots.ts"


def _resolve(**inp) -> dict:  # noqa: ANN003
    node = shutil.which("node")
    if node is None or not ROOTS_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/roots.js")
    if ROOTS_JS.stat().st_mtime < ROOTS_TS.stat().st_mtime:
        pytest.skip("out/roots.js is older than the source; run npm run compile")
    script = (
        "const r = require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(r.resolveRoots(JSON.parse(process.argv[2]))));"
    )
    done = subprocess.run(
        [node, "-e", script, str(ROOTS_JS), json.dumps(inp)],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture()
def folders(tmp_path: Path) -> dict[str, str]:
    fi = tmp_path / "FrontierInsight"
    project = tmp_path / "my_project"
    fi.mkdir()
    project.mkdir()
    (fi / "launch.py").write_text("", encoding="utf-8")
    return {"fi": str(fi), "project": str(project)}


def _in(**kw) -> dict:  # noqa: ANN003
    return {"repoPathSetting": "", "workingDirSetting": "", **kw}


def test_when_the_workspace_is_the_fi_checkout_nothing_changes(folders) -> None:
    got = _resolve(**_in(workspaceRoot=folders["fi"]))
    assert got == {"repoPath": folders["fi"], "workDir": folders["fi"]}


def test_a_project_workspace_runs_quests_in_the_project_not_in_fi(folders) -> None:
    got = _resolve(**_in(repoPathSetting=folders["fi"], workspaceRoot=folders["project"]))
    assert got == {"repoPath": folders["fi"], "workDir": folders["project"]}


def test_a_working_folder_setting_wins_and_a_relative_one_means_the_workspace(folders) -> None:
    got = _resolve(**_in(
        repoPathSetting=folders["fi"], workspaceRoot=folders["project"], workingDirSetting="runs/a",
    ))
    assert Path(got["workDir"]) == Path(folders["project"]) / "runs" / "a"
    other = str(Path(folders["project"]).parent / "elsewhere")
    got = _resolve(**_in(
        repoPathSetting=folders["fi"], workspaceRoot=folders["project"], workingDirSetting=other,
    ))
    assert got["workDir"] == other


def test_the_errors_say_what_to_do(folders) -> None:
    assert "No workspace open" in _resolve(**_in())["error"]
    no_fi = _resolve(**_in(workspaceRoot=folders["project"]))["error"]
    assert "no `launch.py`" in no_fi and "frontierInsight.repoPath" in no_fi
    wrong = _resolve(**_in(repoPathSetting=folders["project"], workspaceRoot=folders["project"]))["error"]
    assert "has no `launch.py`" in wrong


def test_no_command_runs_from_fis_own_folder_by_accident() -> None:
    """The extension used one folder for FI and for the user's work, so a
    project workspace could not run at all. Every command now resolves both."""
    for name in ("extension.ts", "skills.ts"):
        src = (EXT / "src" / name).read_text(encoding="utf-8")
        assert 'cfg.get<string>("repoPath")' not in src or name == "roots-config.ts", name
        assert "cwd: repoPath" not in src and "cwd: repo," not in src, name
    ext = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    assert len(re.findall(r"rootsFromConfig\(cfg\)", ext)) >= 12
    assert 'path.join(repoPath, "launch.py")' in ext, "launch.py is still found in FI's folder"
    assert '"frontierInsight.workingDir"' in (EXT / "package.json").read_text(encoding="utf-8")
