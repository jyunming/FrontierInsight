"""The VS Code extension's .vsix is byte-reproducible: the same sources and lockfile give the same file.

vsce stamps each zip entry with the time its file was compiled or packed and lists entries in filesystem order, so two
builds of one commit used to differ in bytes while every file inside was identical. ``npm run package`` now ends with
``scripts/normalize-vsix.js``, which sorts the entries, gives them one fixed timestamp and deflates them again at a
fixed level, and writes ``<vsix>.manifest.json`` (the sha256 of every packed file).

Skipped without ``node`` (the normalizer), and the full package test without ``node_modules``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

EXT_DIR = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
NORMALIZE = EXT_DIR / "scripts" / "normalize-vsix.js"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(path: Path) -> None:
    proc = subprocess.run([shutil.which("node") or "node", str(NORMALIZE), str(path)],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


def _zip(path: Path, files: dict[str, bytes], *, when: tuple, order: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=when)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, files[name])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_two_zips_that_differ_only_in_time_and_order_come_out_identical(tmp_path: Path) -> None:
    files = {"extension/out/a.js": b"console.log(1)\n" * 50, "[Content_Types].xml": b"<Types/>",
             "extension/package.json": b'{"name": "x"}'}
    a, b = tmp_path / "a.vsix", tmp_path / "b.vsix"
    _zip(a, files, when=(2026, 9, 28, 20, 5, 2), order=list(files))
    _zip(b, files, when=(2026, 9, 28, 21, 7, 40), order=list(reversed(files)))
    assert _sha(a) != _sha(b)
    _normalize(a)
    _normalize(b)
    assert _sha(a) == _sha(b), "same contents must give the same bytes"
    with zipfile.ZipFile(a) as z:
        assert z.testzip() is None
        assert z.namelist() == sorted(files)
        assert {n: z.read(n) for n in z.namelist()} == files, "contents are unchanged"
    manifest = json.loads((tmp_path / "a.vsix.manifest.json").read_text(encoding="utf-8"))
    assert manifest["vsix_sha256"] == _sha(a)
    assert manifest["files"]["extension/package.json"]["sha256"] == hashlib.sha256(files["extension/package.json"]).hexdigest()
    _normalize(a)
    assert _sha(a) == manifest["vsix_sha256"], "normalizing twice changes nothing"


def _vsce() -> Path | None:
    for name in ("vsce.cmd", "vsce"):
        p = EXT_DIR / "node_modules" / ".bin" / name
        if p.exists():
            return p
    return None


@pytest.mark.skipif(shutil.which("node") is None or _vsce() is None,
                    reason="run `npm ci` in vscode-frontier-insight/ first")
def test_packaging_the_extension_twice_gives_the_same_file(tmp_path: Path) -> None:
    compiled = EXT_DIR / "out" / "extension.js"
    if not compiled.exists():
        npm = shutil.which("npm") or "npm"
        assert subprocess.run([npm, "run", "compile"], cwd=str(EXT_DIR), capture_output=True,
                              timeout=180).returncode == 0
    shas = []
    for n in range(2):
        out = tmp_path / f"build{n}.vsix"
        # A fresh compile rewrites out/ with new times: the case that used to change the bytes.
        now = time.time() + n * 3600
        for f in (EXT_DIR / "out").glob("*.js"):
            os.utime(f, (now, now))
        proc = subprocess.run([str(_vsce()), "package", "--out", str(out)], cwd=str(EXT_DIR),
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        _normalize(out)
        shas.append(_sha(out))
    assert shas[0] == shas[1], shas
