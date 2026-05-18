"""Unit tests for ``launch._install_tectonic_from_local``.

The airgapped install path doesn't touch the network; these tests drive
it with synthetic archives + fake binaries to pin the four contracts:

1. Already-installed → no-op (early return).
2. Path-doesn't-exist → rc=1 with a clear error.
3. Directory → finds and uses the archive / binary inside.
4. Wrong-arch binary (no recognised exec header) → rc=1, refuses
   to install garbage.
"""

from __future__ import annotations

import importlib
import io
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def install_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import ``launch`` with the repo root redirected to ``tmp_path``
    so the helper writes its ``tools/`` directory inside the test
    sandbox instead of polluting the real repo."""
    import launch as launch_mod
    # The helper computes ``Path(__file__).resolve().parent`` for the
    # repo root. Patch ``__file__`` so it points at a temp file under
    # ``tmp_path`` and the helper's tools/ lands inside the sandbox.
    fake_launch = tmp_path / "launch.py"
    fake_launch.write_text("# sandbox stub\n", encoding="utf-8")
    monkeypatch.setattr(launch_mod, "__file__", str(fake_launch))
    return launch_mod._install_tectonic_from_local


def _write_elf_stub(path: Path) -> None:
    """Minimal ELF header so ``_install_tectonic_from_local``'s magic-
    bytes sanity check passes. Real Linux binaries start with ``\\x7fELF``
    followed by class / data / version bytes; we just need the first 4."""
    path.write_bytes(b"\x7fELF" + b"\x00" * 60)


def _write_pe_stub(path: Path) -> None:
    """Minimal Windows PE header (MZ prefix)."""
    path.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)


def _exe_name() -> str:
    return "tectonic.exe" if sys.platform == "win32" else "tectonic"


# ---------------------------------------------------------------------------


def test_already_installed_short_circuits(install_helper, tmp_path: Path) -> None:
    """When ``tools/tectonic`` already exists, the helper returns 0
    without touching the source — re-running ``--install-tectonic-from``
    is idempotent and won't clobber a working install."""
    tools = tmp_path / "tools"
    tools.mkdir()
    existing = tools / _exe_name()
    _write_elf_stub(existing)
    original = existing.read_bytes()

    # ``src`` doesn't need to exist — the early-return fires first.
    rc = install_helper(tmp_path / "ignored.tar.gz")
    assert rc == 0
    assert existing.read_bytes() == original  # untouched


def test_missing_path_returns_error(install_helper, tmp_path: Path) -> None:
    """A bogus source path is rejected with rc=1 and a clear message."""
    rc = install_helper(tmp_path / "does-not-exist.tar.gz")
    assert rc == 1


def test_extract_from_tarball(install_helper, tmp_path: Path, capsys) -> None:
    """The .tar.gz path mirrors what ``_install_tectonic`` downloads from
    GitHub — same archive shape, just locally sourced. The helper
    extracts the binary and atomic-replaces into ``tools/``."""
    exe = _exe_name()
    # Build a tar.gz containing a fake tectonic binary.
    archive = tmp_path / "tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz"
    bin_bytes = b"\x7fELF" + b"\x00" * 1000
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=exe)
        info.size = len(bin_bytes)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(bin_bytes))
    archive.write_bytes(buf.getvalue())

    rc = install_helper(archive)
    assert rc == 0
    landed = tmp_path / "tools" / exe
    assert landed.is_file()
    assert landed.read_bytes().startswith(b"\x7fELF")


def test_extract_from_zip(install_helper, tmp_path: Path) -> None:
    """The .zip path is the Windows release shape."""
    exe = _exe_name()
    archive = tmp_path / "tectonic-0.16.9-x86_64-pc-windows-msvc.zip"
    bin_bytes = b"MZ\x90\x00" + b"\x00" * 1000
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(exe, bin_bytes)

    rc = install_helper(archive)
    assert rc == 0
    landed = tmp_path / "tools" / exe
    assert landed.is_file()
    assert landed.read_bytes().startswith(b"MZ")


def test_extract_from_bare_binary(install_helper, tmp_path: Path) -> None:
    """The user can also point us at an already-extracted binary
    (e.g., downloaded once and copied across hosts via USB)."""
    exe = _exe_name()
    src = tmp_path / "downloads" / exe
    src.parent.mkdir()
    if sys.platform == "win32":
        _write_pe_stub(src)
    else:
        _write_elf_stub(src)

    rc = install_helper(src)
    assert rc == 0
    landed = tmp_path / "tools" / exe
    assert landed.is_file()


def test_directory_finds_archive_or_binary(install_helper, tmp_path: Path) -> None:
    """A directory source is scanned — the helper picks the first
    matching archive or binary it finds, so the user can just hand
    over a ``Downloads/`` folder."""
    exe = _exe_name()
    drop = tmp_path / "drop"
    drop.mkdir()
    # Leave one unrelated file + the real binary.
    (drop / "README.txt").write_text("nothing useful", encoding="utf-8")
    target = drop / exe
    if sys.platform == "win32":
        _write_pe_stub(target)
    else:
        _write_elf_stub(target)

    rc = install_helper(drop)
    assert rc == 0
    assert (tmp_path / "tools" / exe).is_file()


def test_directory_empty_returns_error(install_helper, tmp_path: Path) -> None:
    """An empty directory yields no candidates — refuse the install."""
    drop = tmp_path / "empty"
    drop.mkdir()
    rc = install_helper(drop)
    assert rc == 1


def test_wrong_arch_binary_rejected(install_helper, tmp_path: Path) -> None:
    """A "binary" that doesn't start with ELF/Mach-O/PE magic is
    rejected — guards against the user pointing at a README or the
    wrong-arch tarball where the inner file is a shell script
    placeholder."""
    src = tmp_path / "garbage"
    src.write_bytes(b"#!/usr/bin/env bash\necho 'not tectonic'\n")
    rc = install_helper(src)
    assert rc == 1
    # Nothing landed in tools/.
    landed = tmp_path / "tools" / _exe_name()
    assert not landed.is_file()
