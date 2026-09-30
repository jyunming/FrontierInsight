"""The VS Code extension reads the quest index Python writes (core/quest_index.py) and matches ids the same way.

Runs the compiled reader in real Node against an index written by the real Python writer, in a temporary FI_HOME."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from core import quest_index

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
JS = EXT / "out" / "quest-index.js"
TS = EXT / "src" / "quest-index.ts"


def _node(script: str, *args: str) -> object:
    node = shutil.which("node")
    if node is None or not JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/quest-index.js")
    if JS.stat().st_mtime < TS.stat().st_mtime:
        pytest.skip("out/quest-index.js is older than the source; run npm run compile")
    done = subprocess.run([node, "-e", script, str(JS), *args], capture_output=True, text=True, encoding="utf-8",
                          timeout=60, env=os.environ.copy())
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _quest(folder: Path, qid: str, title: str) -> Path:
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    quest_index.register(root, working_folder=folder, title=title)
    return root


def test_the_extension_reads_the_index_python_wrote_and_skips_gone_quests(tmp_path: Path) -> None:
    a = _quest(tmp_path / "a", "1790000001-first-study-479b06", "First study")
    gone = _quest(tmp_path / "b", "1790000002-gone-study-111111", "Gone")
    shutil.rmtree(gone)
    got = _node("const q = require(process.argv[1]);"
                "process.stdout.write(JSON.stringify(q.loadIndex()));")
    assert [e["questId"] for e in got] == ["1790000001-first-study-479b06"]
    e = got[0]
    assert e["questRoot"] == str(a.resolve()) and e["title"] == "First study"
    assert e["workingFolder"] == str((tmp_path / "a").resolve())


def test_the_extension_matches_ids_as_python_does(tmp_path: Path) -> None:
    ids = ["1790000001-first-c0ffee", "1790000002-second-c0ffee", "1790000003-third-479b06"]
    for text in ("479b06", "c0ffee", "1790000003", "b06", "1790000001-first-c0ffee", "nothing"):
        js = _node("const q = require(process.argv[1]);"
                   "process.stdout.write(JSON.stringify(q.matchIds(process.argv[2], JSON.parse(process.argv[3]))));",
                   text, json.dumps(ids))
        assert js == quest_index.matching(text, ids), text
    assert _node("const q = require(process.argv[1]);"
                 "process.stdout.write(JSON.stringify(q.shortId('1790000003-third-479b06')));") == "479b06"


def test_the_extension_uses_the_same_per_person_folder(tmp_path: Path) -> None:
    got = _node("const q = require(process.argv[1]); process.stdout.write(JSON.stringify(q.indexPath()));")
    assert Path(got) == quest_index.index_path()
