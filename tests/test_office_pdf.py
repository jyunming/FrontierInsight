"""LibreOffice discovery and the pptx export the visual check uses."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from generation import _office_pdf as op


def test_libreoffice_on_path_is_found_first(monkeypatch) -> None:
    monkeypatch.setattr(op.shutil, "which", lambda name: "/usr/bin/soffice" if name == "soffice" else None)
    assert op.find_libreoffice() == "/usr/bin/soffice"


def test_the_windows_default_install_is_found_off_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(op.shutil, "which", lambda name: None)
    monkeypatch.setattr(op.sys, "platform", "win32")
    exe = tmp_path / "LibreOffice" / "program" / "soffice.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    assert op.find_libreoffice() == str(exe)


def test_no_libreoffice_means_no_pdf_and_says_why(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(op.shutil, "which", lambda name: None)
    monkeypatch.setattr(op.sys, "platform", "linux")
    assert op.pptx_to_pdf(tmp_path / "slides.pptx", tmp_path / "out") == (None, "LibreOffice was not found")


class _Proc:
    """A stand-in for ``soffice --convert-to pdf``."""

    def __init__(self, argv, *, write: bool = True, hang: bool = False, returncode: int = 0, **kwargs):  # noqa: ANN001
        self.argv, self.kwargs = argv, kwargs
        self.pid = 4242
        self.returncode = returncode
        self.hang = hang
        self.killed = False
        if write:
            out_dir = Path(argv[argv.index("--outdir") + 1])
            (out_dir / f"{Path(argv[-1]).stem}.pdf").write_bytes(b"%PDF-1.4 exported")

    def communicate(self, timeout=None):  # noqa: ANN001
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return b"convert slides.pptx -> slides.pdf using filter : impress_pdf_Export", None

    def kill(self) -> None:
        self.killed = True


def _fake_soffice(monkeypatch, **behaviour) -> list[_Proc]:
    started: list[_Proc] = []

    def popen(argv, **kwargs):  # noqa: ANN001
        started.append(_Proc(argv, **behaviour, **kwargs))
        return started[-1]

    monkeypatch.setattr(op, "find_libreoffice", lambda: "/opt/soffice")
    monkeypatch.setattr(op.subprocess, "Popen", popen)
    return started


def test_the_export_runs_headless_with_its_own_profile_and_names_the_pdf_after_the_pptx(
    tmp_path: Path, monkeypatch,
) -> None:
    started = _fake_soffice(monkeypatch)
    pptx = tmp_path / "slides.pptx"
    pptx.write_bytes(b"pptx")
    out_dir = tmp_path / ".fi" / "visual_check" / "slides_pptx"
    pdf, reason = op.pptx_to_pdf(pptx, out_dir)
    assert (pdf, reason) == (out_dir / "slides.pptx.pdf", "")
    assert pdf.read_bytes() == b"%PDF-1.4 exported"
    # The export's own "slides.pdf" is renamed, so nothing called slides.pdf
    # sits beside the screenshots to be mistaken for the Marp deck.
    assert not (out_dir / "slides.pdf").exists()
    argv = started[0].argv
    assert argv[0] == "/opt/soffice" and argv[-1] == str(pptx)
    assert {"--headless", "--norestore", "--convert-to", "pdf"} <= set(argv)
    profile = next(a for a in argv if a.startswith("-env:UserInstallation="))
    assert profile.startswith("-env:UserInstallation=file:") and "fi-libreoffice-" in profile
    assert started[0].kwargs["stdin"] is subprocess.DEVNULL


def test_an_export_that_writes_nothing_reports_the_exit_code_and_output(tmp_path: Path, monkeypatch) -> None:
    _fake_soffice(monkeypatch, write=False, returncode=1)
    pdf, reason = op.pptx_to_pdf(tmp_path / "slides.pptx", tmp_path / "out")
    assert pdf is None
    assert reason.startswith("LibreOffice exported no PDF (exit 1): convert slides.pptx")


def test_a_hung_export_is_killed_with_its_children(tmp_path: Path, monkeypatch) -> None:
    started = _fake_soffice(monkeypatch, write=False, hang=True)
    killed: list[list[str]] = []
    monkeypatch.setattr(op.sys, "platform", "win32")
    monkeypatch.setattr(op.subprocess, "run", lambda argv, **kw: killed.append(argv))
    pdf, reason = op.pptx_to_pdf(tmp_path / "slides.pptx", tmp_path / "out", timeout_s=5)
    assert pdf is None and reason == "LibreOffice took longer than 5 s to export slides.pptx"
    # soffice.exe only launches soffice.bin: the whole tree goes.
    assert killed == [["taskkill", "/PID", "4242", "/T", "/F"]]
    assert started[0].killed


@pytest.mark.slow
@pytest.mark.skipif(op.find_libreoffice() is None, reason="LibreOffice is not installed")
def test_a_real_deck_exports_one_pdf_page_per_slide(tmp_path: Path) -> None:
    from generation._pdf_measure import measure_pdf
    from generation._pptx_slides import render_marp_to_pptx

    deck = tmp_path / "slides.md"
    deck.write_text(
        "---\nmarp: true\n---\n\n# A title slide\n\n---\n\n## Findings\n\n- First point\n- Second point\n",
        encoding="utf-8",
    )
    pptx = tmp_path / "slides.pptx"
    assert render_marp_to_pptx(deck, pptx)
    pdf, reason = op.pptx_to_pdf(pptx, tmp_path / "export")
    assert reason == "" and pdf is not None
    doc = measure_pdf(pdf)
    assert len(doc.pages) == 2
    assert "Second point" in " ".join(line.text for line in doc.pages[1].lines)
