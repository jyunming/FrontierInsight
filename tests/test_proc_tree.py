"""``core/proc_tree.py``: a program that runs too long is stopped together with everything it started.

``taskkill /T`` walks parent links at the moment it runs, so on a loaded Windows machine a helper created while it
walked, or one whose parent had already exited, kept running (and kept writing) after the "kill". These start real
process trees and check that nothing of them is left. Each tree's deepest process (the writer) adds a byte to its
``beat`` file every 0.1 s: the tree is stopped only once that file grows, and a file still growing afterwards means
a process survived. Nothing depends on how fast the machine starts processes.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from core.proc_tree import ProcessTree

# argv: depth, mode, folder. Every level starts the next one; depth 0 is the writer.
# mode "deep": every level then sleeps. mode "orphan": the level just above the writer exits at once, so the
# writer's parent is gone before anything is stopped.
_CHAIN = textwrap.dedent(
    """
    import pathlib, subprocess, sys, time
    depth, mode, folder = int(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    if depth == 0:
        end = time.monotonic() + 120  # a writer that survives a failed test does not run for ever
        with open(folder / "beat", "ab", buffering=0) as beat:
            while time.monotonic() < end:
                beat.write(b".")
                time.sleep(0.1)
        sys.exit(0)
    subprocess.Popen([sys.executable, __file__, str(depth - 1), mode, str(folder)])
    if mode == "orphan" and depth == 1:
        (folder / "parent-gone").write_text("gone")
        sys.exit(0)
    time.sleep(120)
    """
)


def _chain(tmp_path: Path, name: str) -> tuple[Path, Path]:
    folder = tmp_path / name
    folder.mkdir()
    script = folder / "chain.py"
    script.write_text(_CHAIN, encoding="utf-8")
    return script, folder


def _wait_for(path: Path, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


def _beats(folder: Path) -> int:
    beat = folder / "beat"
    if not beat.exists():
        return 0
    with open(beat, "rb") as fh:  # the open file's own length, not a directory entry that may lag
        return fh.seek(0, os.SEEK_END)


def _wait_until_beating(folder: Path, timeout: float = 90.0) -> None:
    """The writer is running: its file has grown past what it had a moment ago."""
    _wait_for(folder / "beat", timeout)
    first = _beats(folder)
    deadline = time.monotonic() + timeout
    while _beats(folder) <= first:
        assert time.monotonic() < deadline, "the writer stopped writing on its own"
        time.sleep(0.05)


def _start(stack: contextlib.ExitStack, script: Path, folder: Path, depth: int, mode: str) -> ProcessTree:
    tree = stack.enter_context(ProcessTree([sys.executable, str(script), str(depth), mode, str(folder)]))
    if os.name == "nt":
        assert tree._job is not None  # noqa: SLF001 - the job, not the taskkill fallback, is what is under test
    return tree


def _assert_nothing_left(tree: ProcessTree, folder: Path) -> None:
    assert tree.proc.returncode is not None
    before = _beats(folder)
    time.sleep(1.5)  # fifteen beats of a writer that survived
    assert _beats(folder) == before, "a process of the stopped tree is still running"


def test_every_level_of_a_deep_tree_is_stopped(tmp_path: Path) -> None:
    script, folder = _chain(tmp_path, "deep")
    with contextlib.ExitStack() as stack:
        tree = _start(stack, script, folder, 3, "deep")  # program -> child -> grandchild -> writer
        _wait_until_beating(folder)
        tree.kill()
        _assert_nothing_left(tree, folder)


def test_a_helper_whose_parent_already_exited_is_stopped(tmp_path: Path) -> None:
    script, folder = _chain(tmp_path, "orphan")
    with contextlib.ExitStack() as stack:
        tree = _start(stack, script, folder, 2, "orphan")  # program -> middle (exits) -> writer
        _wait_for(folder / "parent-gone")
        _wait_until_beating(folder)
        time.sleep(0.5)  # the middle process has written its note; let it finish exiting
        tree.kill()
        _assert_nothing_left(tree, folder)


def test_two_trees_stopped_at_the_same_time_are_both_stopped_and_only_their_own(tmp_path: Path) -> None:
    with contextlib.ExitStack() as stack:
        trees: list[tuple[ProcessTree, Path]] = []
        for name in ("a", "b"):
            script, folder = _chain(tmp_path, name)
            trees.append((_start(stack, script, folder, 2, "deep"), folder))
        bystander_script, bystander = _chain(tmp_path, "bystander")
        other = _start(stack, bystander_script, bystander, 1, "deep")
        for _, folder in trees:
            _wait_until_beating(folder)
        _wait_until_beating(bystander)
        errors: list[BaseException] = []

        def time_out(tree: ProcessTree) -> None:
            try:
                with pytest.raises(subprocess.TimeoutExpired):
                    tree.proc.wait(timeout=1)
                tree.kill()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
                errors.append(exc)

        threads = [threading.Thread(target=time_out, args=(tree,)) for tree, _ in trees]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert not errors and not any(t.is_alive() for t in threads)
        for tree, folder in trees:
            _assert_nothing_left(tree, folder)
        # Stopping the two trees left an unrelated tree running.
        assert other.proc.poll() is None
        _wait_until_beating(bystander)


def test_leaving_the_block_early_stops_a_tree_that_is_still_running(tmp_path: Path) -> None:
    script, folder = _chain(tmp_path, "early")
    with pytest.raises(RuntimeError), contextlib.ExitStack() as stack:
        tree = _start(stack, script, folder, 2, "deep")
        _wait_until_beating(folder)
        raise RuntimeError("the caller gave up (an error, or Ctrl+C)")
    _assert_nothing_left(tree, folder)


def test_a_program_that_finishes_on_its_own_keeps_its_exit_code_and_output(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    with out.open("wb") as fh, ProcessTree(
        [sys.executable, "-c", "print('hello'); raise SystemExit(3)"], stdout=fh, stdin=subprocess.DEVNULL,
    ) as tree:
        assert tree.proc.wait(timeout=60) == 3
    assert out.read_text().strip() == "hello"


@pytest.mark.skipif(os.name != "nt", reason="the Job Object structures exist on Windows only")
def test_the_job_object_structures_have_the_sizes_windows_expects() -> None:
    import ctypes

    from core import proc_tree

    expected = (144, 48) if ctypes.sizeof(ctypes.c_void_p) == 8 else (112, 48)
    assert (
        ctypes.sizeof(proc_tree._JOBOBJECT_EXTENDED_LIMIT_INFORMATION),  # noqa: SLF001
        ctypes.sizeof(proc_tree._JOBOBJECT_BASIC_ACCOUNTING_INFORMATION),  # noqa: SLF001
    ) == expected
