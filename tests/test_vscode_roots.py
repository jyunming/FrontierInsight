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
    (fi / "core").mkdir(parents=True)
    project.mkdir()
    (fi / "launch.py").write_text("", encoding="utf-8")
    (fi / "core" / "engine.py").write_text("", encoding="utf-8")
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
    for name in ("extension.ts", "skills.ts", "trace.ts", "quest-map.ts"):
        src = (EXT / "src" / name).read_text(encoding="utf-8")
        assert 'cfg.get<string>("repoPath")' not in src, name
        assert "cwd: repoPath" not in src and "cwd: repo," not in src, name
        assert "rootsFromConfig" not in src, f"{name} still reads the settings without finding FI"
    ext = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    assert len(re.findall(r"await rootsForCommand\(", ext)) >= 12
    assert 'path.join(repoPath, "launch.py")' in ext, "launch.py is still found in FI's folder"
    assert "workspaceFolders?.[0]" not in ext, "a command still ignores which folder the study is in"
    assert '"frontierInsight.workingDir"' in (EXT / "package.json").read_text(encoding="utf-8")


# --- Finding FI with no setting -------------------------------------------------------------

def _node_eval(script: str, *args: str, env: dict | None = None) -> dict:  # noqa: ANN001
    node = shutil.which("node")
    if node is None or not ROOTS_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/roots.js")
    if ROOTS_JS.stat().st_mtime < ROOTS_TS.stat().st_mtime:
        pytest.skip("out/roots.js is older than the source; run npm run compile")
    done = subprocess.run(
        [node, "-e", script, str(ROOTS_JS), *args],
        capture_output=True, text=True, encoding="utf-8", timeout=60, env=env,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


_LOCATE = """
const r = require(process.argv[1]);
const inp = JSON.parse(process.argv[2]);
const calls = [];
r.locateFi({
  setting: inp.setting || "",
  workspaceFolders: inp.folders || [],
  savedFile: inp.savedFile,
  askPython: async () => { calls.push("python"); return inp.python || undefined; },
  askUser: async () => { calls.push("picker"); return inp.picked || undefined; },
}).then((got) => process.stdout.write(JSON.stringify({ got, calls })));
"""


def _locate(**inp) -> dict:  # noqa: ANN003
    return _node_eval(_LOCATE, json.dumps(inp))


def _fi_stub(folder: Path) -> str:
    """A folder that looks like FI: launch.py beside core/engine.py. Its launch.py leaves a
    mark if anything ever imports (runs) it."""
    (folder / "core").mkdir(parents=True)
    (folder / "core" / "engine.py").write_text("", encoding="utf-8")
    (folder / "launch.py").write_text(
        "import pathlib\npathlib.Path(__file__).with_name('RAN').write_text('ran')\n", encoding="utf-8",
    )
    return str(folder)


def test_fi_is_found_in_order_and_the_picker_is_the_last_resort(folders, tmp_path) -> None:
    saved = str(tmp_path / "home" / ".frontier-insight" / "fi_location.json")
    # The open folder is FI: nothing is asked, not even the Python.
    got = _locate(folders=[folders["project"], folders["fi"]], savedFile=saved)
    assert got == {"got": {"repoPath": folders["fi"], "how": "workspace"}, "calls": []}
    # A project folder: the Python says where FI is installed; no picker.
    got = _locate(folders=[folders["project"]], savedFile=saved, python=folders["fi"])
    assert got == {"got": {"repoPath": folders["fi"], "how": "python"}, "calls": ["python"]}
    # Nothing found: one folder picker, and the answer is remembered in ~/.frontier-insight.
    got = _locate(folders=[folders["project"]], savedFile=saved, picked=folders["fi"])
    assert got == {
        "got": {"repoPath": folders["fi"], "how": "picked", "remembered": True}, "calls": ["python", "picker"],
    }
    assert json.loads(Path(saved).read_text(encoding="utf-8")) == {"path": folders["fi"]}
    # The next window (or session) finds the remembered folder without asking.
    got = _locate(folders=[folders["project"]], savedFile=saved)
    assert got == {"got": {"repoPath": folders["fi"], "how": "saved"}, "calls": ["python"]}


def test_a_studys_own_launch_py_is_never_taken_for_fi(folders, tmp_path) -> None:
    """`launch.py` is a common name for a study's start script: an open folder, a remembered
    folder or a picked one is FI only with FI's `core/engine.py` beside it."""
    (Path(folders["project"]) / "launch.py").write_text("", encoding="utf-8")
    saved = tmp_path / "fi_location.json"
    saved.write_text(json.dumps({"path": folders["project"]}), encoding="utf-8")
    got = _locate(folders=[folders["project"]], savedFile=str(saved), python=folders["fi"])
    assert got == {"got": {"repoPath": folders["fi"], "how": "python"}, "calls": ["python"]}
    got = _locate(folders=[folders["project"]], savedFile=str(saved), picked=folders["project"])
    assert "is not the FrontierInsight folder" in got["got"]["error"]


def test_a_wrong_answer_is_refused_in_plain_words(folders, tmp_path) -> None:
    saved = str(tmp_path / "fi_location.json")
    got = _locate(folders=[folders["project"]], savedFile=saved, picked=folders["project"])["got"]
    assert "is not the FrontierInsight folder" in got["error"] and not Path(saved).exists()
    got = _locate(folders=[folders["project"]], savedFile=saved)["got"]
    assert "was not found" in got["error"] and "pip install -e" in got["error"]
    # A picked folder that cannot be saved still works now, and says it was not remembered.
    blocker = tmp_path / "a_file"
    blocker.write_text("", encoding="utf-8")
    got = _locate(folders=[folders["project"]], savedFile=str(blocker / "fi_location.json"), picked=folders["fi"])
    assert got["got"] == {"repoPath": folders["fi"], "how": "picked", "remembered": False}
    # A setting someone typed is used as is, and a wrong one is reported, not skipped over.
    got = _locate(setting=folders["project"], folders=[folders["fi"]], savedFile=saved)
    assert "frontierInsight.repoPath" in got["got"]["error"] and got["calls"] == []
    got = _locate(setting=folders["fi"], folders=[folders["project"]], savedFile=saved)
    assert got == {"got": {"repoPath": folders["fi"], "how": "setting"}, "calls": []}


def test_the_python_is_asked_without_running_launch_py(tmp_path) -> None:
    """After `pip install -e <FI>` the configured Python knows where FI is. The question
    only looks the module up: FI's launch.py (and a stray one in the current folder)
    never runs."""
    import os
    import sys

    fi = _fi_stub(tmp_path / "FI")
    decoy = tmp_path / "decoy"  # a launch.py with no FI beside it (another project's module)
    decoy.mkdir()
    (decoy / "launch.py").write_text("", encoding="utf-8")
    script = """
const r = require(process.argv[1]);
r.askPythonWhereFiIs(process.argv[2]).then((got) => process.stdout.write(JSON.stringify({ got: got || null })));
"""
    env = {**os.environ, "PYTHONPATH": fi}
    assert _node_eval(script, sys.executable, env=env) == {"got": fi}
    assert not (Path(fi) / "RAN").exists(), "asking the Python ran FI's launch.py"
    env = {**os.environ, "PYTHONPATH": str(decoy)}
    assert _node_eval(script, sys.executable, env=env) == {"got": None}
    # The current folder is never searched: run the question itself inside a folder holding a launch.py.
    code = _node_eval("process.stdout.write(JSON.stringify(require(process.argv[1]).PYTHON_WHERE_IS_FI))")
    here = _fi_stub(tmp_path / "cwd_fi")
    clean = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-c", code], cwd=here, capture_output=True, text=True, env=clean, timeout=60)
    assert here not in out.stdout and not (Path(here) / "RAN").exists()
    # A Python that is not there answers nothing (and does not hang).
    assert _node_eval(script, str(tmp_path / "no-such-python"), env=env) == {"got": None}


_CHOOSE = """
const r = require(process.argv[1]);
process.stdout.write(JSON.stringify({ got: r.chooseWorkFolder(JSON.parse(process.argv[2])) || null }));
"""


def test_with_several_folders_open_the_study_is_where_its_yaml_is(tmp_path) -> None:
    a, b = tmp_path / "study_a", tmp_path / "study_b"
    for d in (a, b):
        d.mkdir()
    (b / "b.yaml").write_text("", encoding="utf-8")

    def choose(**inp) -> str | None:  # noqa: ANN003
        return _node_eval(_CHOOSE, json.dumps(inp))["got"]

    folders = [str(a), str(b)]
    assert choose(folders=[str(a)], namedPath="whatever.yaml") == str(a)
    assert choose(folders=folders, namedPath="b.yaml") == str(b)
    assert choose(folders=folders, namedPath=str(b / "b.yaml")) == str(b)
    assert choose(folders=folders, activeFile=str(b / "notes.md")) == str(b)
    # Nothing to go on: the first folder; never a question.
    assert choose(folders=folders) == str(a)
    assert choose(folders=[]) is None
    # ... but not FI's own checkout when a study folder is open beside it.
    fi = tmp_path / "FI"
    (fi / "core").mkdir(parents=True)
    (fi / "launch.py").write_text("", encoding="utf-8")
    (fi / "core" / "engine.py").write_text("", encoding="utf-8")
    assert choose(folders=[str(fi), str(a)]) == str(a)
    # A folder whose name merely starts with ".." is still inside.
    (a / "..data").mkdir()
    assert choose(folders=[str(b), str(a)], activeFile=str(a / "..data" / "x.csv")) == str(a)


def test_a_second_window_gets_a_bridge_address_of_its_own(tmp_path) -> None:
    """Two VS Code windows (two study folders) open at once: the second can't take the
    per-user bridge address, so it binds one of its own and hands that to the Python it
    starts, instead of routing through the first window."""
    import os

    node = shutil.which("node")
    bridge_js = EXT / "out" / "persistent-bridge.js"
    if node is None or not bridge_js.is_file():
        pytest.skip("needs node and a compiled persistent-bridge.js")
    if bridge_js.stat().st_mtime < (EXT / "src" / "persistent-bridge.ts").stat().st_mtime:
        pytest.skip("out/persistent-bridge.js is older than the source; run npm run compile")
    # The first window is a plain listener on the per-user address; the second is the real
    # PersistentBridge. It must take an address of its own, and closing it must leave the
    # first window's listener alone.
    script = r"""
const Module = require("module");
const realLoad = Module._load;
Module._load = function (req, ...rest) { return req === "vscode" ? {} : realLoad.call(this, req, ...rest); };
const net = require("net");
const outDir = require("path").dirname(process.argv[1]);
const { persistentBridgePath } = require(outDir + "/bridge-path.js");
const { PersistentBridge } = require(process.argv[1]);
const shared = persistentBridgePath();
const first = net.createServer((s) => s.end());
first.listen(shared, async () => {
  const lines = [];
  const b = new PersistentBridge({ appendLine: (l) => lines.push(l) });
  const got = await b.listen();
  const bound = b.boundPath;
  await b.close();
  const alive = await new Promise((res) => {
    const c = net.createConnection(shared);
    c.once("connect", () => { c.destroy(); res(true); });
    c.once("error", () => res(false));
  });
  first.close();
  process.stdout.write(JSON.stringify({ shared, got, bound, alive, pid: process.pid }));
  process.exit(0);
});
"""
    import tempfile

    # A short folder: a unix socket path is limited to ~104 bytes.
    runtime = tempfile.mkdtemp(prefix="fiw")
    env = {**os.environ, "USERNAME": f"fi_two_windows_{os.getpid()}", "XDG_RUNTIME_DIR": runtime}
    try:
        done = subprocess.run(
            [node, "-e", script, str(bridge_js)],
            capture_output=True, text=True, encoding="utf-8", timeout=60, env=env,
        )
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got["got"] != got["shared"], got
    assert f"-{got['pid']}" in got["got"], got
    assert got["bound"] == got["got"]
    assert got["alive"], "closing the second window's bridge broke the first window's"
    ext = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    # The one line meant for a terminal (the `--update` line /update shows for answering the questions one by one)
    # carries this window's address; no chat command runs in a terminal any more.
    assert ext.count("bridgeSocket: thisWindowsBridge(),") == 1
    bridge = (EXT / "src" / "persistent-bridge.ts").read_text(encoding="utf-8")
    assert "persistentBridgePath(String(process.pid))" in bridge
