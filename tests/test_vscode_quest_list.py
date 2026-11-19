"""The VS Code extension offers the one list of quests the web page shows, with FI's own running status, and offers no
second run of a running quest. Runs the compiled reader (out/quest-list.js) in real Node on the JSON the real
``fi tools quests --json`` prints, in a temporary FI_HOME (tests/conftest.py)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import launch
from core import quest_index

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
JS = EXT / "out" / "quest-list.js"
TS = EXT / "src" / "quest-list.ts"


def _node(script: str, *args: str) -> object:
    node = shutil.which("node")
    if node is None or not JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/quest-list.js")
    if JS.stat().st_mtime < TS.stat().st_mtime:
        pytest.skip("out/quest-list.js is older than the source; run npm run compile")
    done = subprocess.run([node, "-e", script, str(JS), *args], capture_output=True, text=True, encoding="utf-8",
                          timeout=60, env=os.environ.copy())
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _quest(folder: Path, qid: str, title: str) -> Path:
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    quest_index.register(root, working_folder=folder, title=title)
    return root


def test_the_extension_lists_every_folders_quests_and_refuses_a_running_one(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    psutil = pytest.importorskip("psutil")
    a = _quest(tmp_path / "study_a", "1790000001-first-aaaaaa", "First")
    b = _quest(tmp_path / "study_b", "1790000002-second-bbbbbb", "Second")
    _quest(tmp_path / "study_c", "1790000003-third-cccccc", "Third")
    me = {"pid": os.getpid(), "started": psutil.Process(os.getpid()).create_time(), "seen_at": time.time(), "ended": None}
    (a / ".fi" / "process.json").write_text(json.dumps(me), encoding="utf-8")
    # the folder the extension is open on holds one quest; the record holds three
    assert launch._list_quests(as_json=True, output_root=b.parent) == 0
    printed = capsys.readouterr().out
    got = _node(
        "const q = require(process.argv[1]); const list = q.parseQuestList(process.argv[2]);"
        "process.stdout.write(JSON.stringify({ids: list.map(x => x.questId), status: list.map(x => x.status),"
        " first: q.isRunning(list, process.argv[3]), second: q.isRunning(list, process.argv[4]),"
        " mark: list.map(q.runningMark)}));",
        printed, str(a.resolve()), str(b.resolve()),
    )
    assert got["ids"][0] == "1790000002-second-bbbbbb"  # this folder's quest first
    assert sorted(got["ids"]) == ["1790000001-first-aaaaaa", "1790000002-second-bbbbbb", "1790000003-third-cccccc"]
    assert got["status"][got["ids"].index("1790000001-first-aaaaaa")] == "running"
    assert got["first"] is True and got["second"] is False  # the running quest is refused, a stopped one is not
    assert got["mark"][got["ids"].index("1790000001-first-aaaaaa")] == "running now"


def test_a_quest_whose_status_is_unknown_is_never_refused(tmp_path: Path) -> None:
    """Python that cannot be run falls back to the index file with no status: nothing is called running, so a resume the
    person asked for is not blocked by a broken Python."""
    root = _quest(tmp_path / "study_a", "1790000001-first-aaaaaa", "First")
    got = _node(
        "const q = require(process.argv[1]);"
        "q.loadQuestList(async () => ({code: 1, stdout: ''}), process.argv[2]).then(list =>"
        " process.stdout.write(JSON.stringify({ids: list.map(x => x.questId), status: list.map(x => x.status),"
        " running: q.isRunning(list, process.argv[3])})));",
        str(root.parent), str(root.resolve()),
    )
    assert got == {"ids": ["1790000001-first-aaaaaa"], "status": [""], "running": False}
