"""PowerPoint deck to PDF through LibreOffice, so the visual check can see
``slides.pptx``.

FI draws ``slides.pptx`` with its own renderer, not with Marp, so the Marp
PDF says nothing about how the pptx looks. LibreOffice is optional: without
it the pptx is not checked, and the visual check report says so.

Every export starts LibreOffice with a fresh profile in a temporary folder.
Without one, an export while LibreOffice is already open is handed to that
running copy, which may be the user's own window with their documents in it.
Measured on Windows, a fresh profile costs about 11 s per deck.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_log = logging.getLogger("frontier_insight.office_pdf")

_PATH_NAMES = ("soffice", "libreoffice")
_TIMEOUT_S = 240.0
NOT_FOUND = "LibreOffice was not found"


def find_libreoffice() -> str | None:
    """Locate LibreOffice's ``soffice``: PATH first, then the default install
    locations, since the Windows and macOS installers do not add it to PATH."""
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    candidates: list[Path] = []
    if sys.platform == "win32":
        for var, default in (("ProgramFiles", r"C:\Program Files"), ("ProgramFiles(x86)", r"C:\Program Files (x86)")):
            candidates.append(Path(os.environ.get(var, default)) / "LibreOffice" / "program" / "soffice.exe")
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return None


def pptx_to_pdf(pptx: Path, out_dir: Path, *, timeout_s: float = _TIMEOUT_S) -> tuple[Path | None, str]:
    """Export ``pptx`` to ``out_dir/<name>.pdf`` (``slides.pptx.pdf``).

    Returns ``(pdf, "")``, or ``(None, reason)`` when there is no PDF.
    Never raises."""
    soffice = find_libreoffice()
    if soffice is None:
        return None, NOT_FOUND
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        exported = out_dir / f"{pptx.stem}.pdf"
        target = out_dir / f"{pptx.name}.pdf"
        exported.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="fi-libreoffice-", ignore_cleanup_errors=True) as profile:
            argv = [
                soffice, f"-env:UserInstallation={Path(profile).as_uri()}",
                "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx),
            ]
            code, output = _run(argv, timeout_s)
        if code is None:
            return None, f"LibreOffice took longer than {timeout_s:.0f} s to export {pptx.name}"
        if not exported.is_file():
            tail = output.strip()[-300:]
            return None, f"LibreOffice exported no PDF (exit {code})" + (f": {tail}" if tail else "")
        exported.replace(target)
        return target, ""
    except OSError as exc:
        _log.info("LibreOffice export of %s failed: %r", pptx.name, exc)
        return None, f"LibreOffice could not export {pptx.name}: {exc}"


def _run(argv: list[str], timeout_s: float) -> tuple[int | None, str]:
    """Run ``argv``; on timeout kill it with its children and return
    ``(None, "")``. On Windows ``soffice.exe`` only launches ``soffice.bin``,
    so killing the launcher alone would leave LibreOffice running."""
    windows = sys.platform == "win32"
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=not windows,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if windows:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        proc.kill()
        proc.communicate()
        return None, ""
    return proc.returncode, (out or b"").decode("utf-8", "replace")
