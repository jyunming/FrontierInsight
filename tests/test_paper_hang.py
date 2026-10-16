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


def test_the_poster_compile_stops_a_hung_engine_at_its_timeout(tmp_path: Path, monkeypatch) -> None:
    """The poster runs its LaTeX engine as ``engine -interaction=nonstopmode -halt-on-error poster.tex``;
    a stand-in engine (a script the engine path names) starts a child that keeps the output and sleeps.
    The compile must end at its time limit with the poster's timeout reason, and the child must be gone."""
    import generation.poster as poster

    script, pid_file = _holder(tmp_path)
    if os.name == "nt":
        engine = tmp_path / "engine.cmd"
        engine.write_text(f'@"{script[0]}" "{script[1]}" "{pid_file}" %*\r\n', encoding="utf-8")
    else:
        engine = tmp_path / "engine.sh"
        engine.write_text(f'#!/bin/sh\nexec "{script[0]}" "{script[1]}" "{pid_file}" "$@"\n', encoding="utf-8")
        engine.chmod(0o755)
    tex = tmp_path / "poster.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")
    monkeypatch.setattr(poster, "_COMPILE_TIMEOUT_S", 3)
    start = time.monotonic()
    failure = poster.PosterGenerator._compile(
        None, ("pdflatex", str(engine)), tex, tmp_path, tmp_path / "poster.pdf",  # type: ignore[arg-type]
    )
    assert time.monotonic() - start < 30
    assert failure is not None and failure[0] == "pdflatex_timeout"
    assert "(>3s)" in failure[1]
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _alive(child) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _alive(child), "the child the engine started is still running"


# ---------------------------------------------------------------------------
# After a timeout, or an exit 0 with no PDF, the paper is made by the browser, marked


def _generator_with_a_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_run):  # noqa: ANN001, ANN202
    import generation._html_pdf as html_mod
    import generation.paper as paper_mod
    from core.config import Config
    from generation.paper import PaperGenerator
    from tests.paper_fakes import fake_pandoc

    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))
    monkeypatch.setattr(PaperGenerator, "_find_unicode_engine", lambda self, engine: None)
    monkeypatch.setattr(html_mod, "find_html_browser", lambda: ("edge", "/fake/edge"))
    banners: list[str] = []

    def fake_html_render(paper_md, out_pdf, **kw):  # noqa: ANN001, ANN003
        banners.append(kw.get("not_typeset") or "")
        out_pdf.write_bytes(b"%PDF-1.4 fallback")
        return out_pdf, ""

    monkeypatch.setattr(html_mod, "render_paper_html_pdf", fake_html_render)
    fake_pandoc(monkeypatch, fake_run)
    cfg = Config.model_validate({"topic": "t", "title": "t", "output": {"output_dir": str(tmp_path / "o")}})
    md = tmp_path / "paper.md"
    md.write_text("# A Paper\n\n## Abstract\n\nShort.\n\n## Results\n\nOne result.\n", encoding="utf-8")
    return PaperGenerator(cfg), md, banners


def test_a_latex_timeout_leaves_a_browser_made_paper_marked_not_typeset(tmp_path: Path, monkeypatch) -> None:
    def hung(cmd, **_kw):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

    gen, md, banners = _generator_with_a_browser(tmp_path, monkeypatch, hung)
    out = tmp_path / "out"
    out.mkdir()
    pdf, skip = gen._compile_pdf(md, out)
    assert skip is None and pdf is not None and pdf.read_bytes().startswith(b"%PDF")
    record = gen.last_render
    assert record["typeset"] is False and "did not finish within 300 s" in record["reason"]
    assert banners and "did not finish" in banners[0]


def test_latex_exiting_0_without_a_pdf_leaves_a_browser_made_paper_marked_not_typeset(
    tmp_path: Path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    gen, md, _banners = _generator_with_a_browser(
        tmp_path, monkeypatch, lambda cmd, **_kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    out = tmp_path / "out"
    out.mkdir()
    pdf, skip = gen._compile_pdf(md, out)
    assert skip is None and pdf is not None
    assert gen.last_render["typeset"] is False and "wrote no PDF" in gen.last_render["reason"]


def test_a_latex_timeout_still_gives_the_page_check_a_count(tmp_path: Path, monkeypatch) -> None:
    def hung(cmd, **_kw):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

    gen, md, _banners = _generator_with_a_browser(tmp_path, monkeypatch, hung)
    scratch = tmp_path / "count"
    scratch.mkdir()
    pdf, why, record = gen.compile_for_page_count(md, scratch)
    assert pdf is not None and why == "" and record["typeset"] is False
