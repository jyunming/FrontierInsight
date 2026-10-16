"""A LaTeX run that hangs ends at its timeout, with everything it started.

``subprocess.run`` killed pandoc alone on a timeout and then, on Windows, waited with no
limit for output a child still held (pandoc's xelatex, that one's dvipdfmx, a headless
browser's helpers), so a hung engine hung the quest. Here a stand-in "pandoc" starts a
child that keeps the output handles and sleeps; the run must still end at the timeout,
and the child must be gone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.proc_tree import run_tree

HOLDER = (
    "import subprocess, sys, time\n"
    "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
    " stdout=sys.stdout, stderr=sys.stderr)\n"
    "open(sys.argv[1], 'w').write(str(child.pid))\n"
    "print('started', flush=True)\n"
    "time.sleep(120)\n"
)


def _alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _holder(tmp_path: Path) -> tuple[list[str], Path]:
    script = tmp_path / "holder.py"
    script.write_text(HOLDER, encoding="utf-8")
    pid_file = tmp_path / "child.pid"
    return [sys.executable, str(script), str(pid_file)], pid_file


def test_a_run_whose_child_holds_the_output_ends_at_the_timeout_and_stops_the_child(tmp_path: Path) -> None:
    cmd, pid_file = _holder(tmp_path)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_tree(cmd, timeout=3, cwd=str(tmp_path))
    assert time.monotonic() - start < 30
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child), "the child the program started is still running"


def test_a_run_that_finishes_returns_its_output_as_text(tmp_path: Path) -> None:
    done = run_tree([sys.executable, "-c", "import sys; print('é ok'); sys.stderr.write('warn'); sys.exit(3)"],
                    timeout=30, cwd=str(tmp_path))
    assert done.returncode == 3 and "ok" in done.stdout and done.stderr == "warn"


def test_the_paper_pandoc_run_stops_a_hung_engine_at_its_timeout(tmp_path: Path) -> None:
    from generation.paper import PaperGenerator

    cmd, pid_file = _holder(tmp_path)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        PaperGenerator._run_pandoc(cmd, tmp_path, 3)
    assert time.monotonic() - start < 30
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child)


def test_the_browser_fallback_runs_stop_a_hung_browser_at_their_timeout(tmp_path: Path) -> None:
    import generation._html_pdf as html_pdf

    cmd, pid_file = _holder(tmp_path)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        html_pdf._run(cmd, timeout=3, cwd=str(tmp_path))
    assert time.monotonic() - start < 30
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child)
