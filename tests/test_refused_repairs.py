"""A refused repair is kept on a side reference of code/'s git history, never on the branch."""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess

import pytest

from core import code_project
from core.engine import Engine

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(code, *args):
    return subprocess.run(["git", *args], cwd=code, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def _quest(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("x = 1\n", encoding="utf-8")
    assert code_project.record_change(tmp_path, "the code as it ran")
    return code


def _engine(tmp_path):
    eng = object.__new__(Engine)
    eng.quest_root = tmp_path
    eng._log = logging.getLogger("test_refused")
    return eng


def test_a_refused_repair_is_kept_beside_the_branch_with_its_reason_and_result(tmp_path):
    code = _quest(tmp_path)
    head, tree = _git(code, "rev-parse", "HEAD"), _git(code, "status", "--porcelain")
    asyncio.run(_engine(tmp_path)._keep_refused_repair(
        "execute_reflect", "simulate.py", "x = 2\n", "it lost run_cell", {"returncode": 1, "stderr_tail": "boom"}))
    assert _git(code, "for-each-ref", "--format=%(refname)", "refs/fi/refused/") == "refs/fi/refused/001"
    ref = "refs/fi/refused/001"
    assert _git(code, "show", f"{ref}:simulate.py") == "x = 2"
    assert "it lost run_cell" in _git(code, "show", f"{ref}:.fi_refused/reason.txt")
    assert '"stderr_tail": "boom"' in _git(code, "show", f"{ref}:.fi_refused/result.json")
    assert _git(code, "rev-parse", "HEAD") == head and _git(code, "status", "--porcelain") == tree == ""
    assert (code / "simulate.py").read_text(encoding="utf-8") == "x = 1\n"
    asyncio.run(_engine(tmp_path)._keep_refused_repair("oracle", "simulate.py", "x = 3\n", "again"))
    assert _git(code, "show", "refs/fi/refused/002:simulate.py") == "x = 3"


def test_an_accepted_repair_makes_no_refused_reference(tmp_path):
    code = _quest(tmp_path)
    (code / "simulate.py").write_text("x = 2\n", encoding="utf-8")
    assert code_project.record_change(tmp_path, "finding repair accepted: F1", category="fixed")
    assert _git(code, "for-each-ref", "refs/fi/refused/") == ""


def test_an_empty_reply_keeps_nothing_and_missing_git_does_not_stop_the_repair(tmp_path, monkeypatch):
    code = _quest(tmp_path)
    asyncio.run(_engine(tmp_path)._keep_refused_repair("seed", "simulate.py", "", "no script"))
    assert _git(code, "for-each-ref", "refs/fi/refused/") == ""

    def no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(code_project, "_git", no_git)
    asyncio.run(_engine(tmp_path)._keep_refused_repair("seed", "simulate.py", "x = 9\n", "no git"))
    assert code_project.keep_refused(tmp_path, None, {"a.py": "x"}, "m") is False


def test_two_refusals_at_the_same_time_each_keep_only_their_own_files(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    code = _quest(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        done = list(pool.map(lambda i: code_project.keep_refused(
            tmp_path, None, {f"s{i}.py": f"x = {i}\n"}, f"try {i}", extras={"reason.txt": "overwritten?"}), range(2)))
    assert done == [True, True]
    refs = _git(code, "for-each-ref", "--format=%(refname)", "refs/fi/refused/").split()
    assert len(refs) == 2
    for ref in refs:
        names = _git(code, "ls-tree", "-r", "--name-only", ref).split()
        mine = [n for n in names if n[:1] == "s" and n[1:2].isdigit()]
        assert len(mine) == 1
        assert "try" in _git(code, "show", f"{ref}:.fi_refused/reason.txt")
    assert _git(code, "status", "--porcelain") == ""


def test_only_the_newest_numbered_tries_stay(tmp_path, monkeypatch):
    code = _quest(tmp_path)
    monkeypatch.setattr(code_project, "REFUSED_KEPT", 3)
    for i in range(5):
        assert code_project.keep_refused(tmp_path, None, {"a.py": f"x = {i}\n"}, "m")
    refs = _git(code, "for-each-ref", "--format=%(refname)", "refs/fi/refused/").split()
    assert sorted(refs) == [f"refs/fi/refused/{n:03d}" for n in (3, 4, 5)]
