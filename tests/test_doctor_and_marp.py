"""`--doctor` environment report and Marp CLI discovery / install.

Doctor exists because paper_pdf had a pre-flight but slides and poster had
none, so a missing renderer only surfaced AFTER the pipeline had spent its
LLM calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import launch
from generation import _marp


# ---------------------------------------------------------------- find_marp

def test_path_wins_over_tools_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_marp.shutil, "which", lambda _n: "/usr/bin/marp")
    assert _marp.find_marp() == "/usr/bin/marp"


def test_falls_back_to_tools_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of --install-marp: a PATH-only lookup would ignore
    the binary the installer just downloaded."""
    monkeypatch.setattr(_marp.shutil, "which", lambda _n: None)
    tools = tmp_path / "tools"
    tools.mkdir()
    name = "marp.exe" if sys.platform == "win32" else "marp"
    (tools / name).write_text("binary")
    assert _marp.find_marp(tmp_path) == str(tools / name)


def test_ignores_wrong_platform_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_marp.shutil, "which", lambda _n: None)
    tools = tmp_path / "tools"
    tools.mkdir()
    wrong = "marp" if sys.platform == "win32" else "marp.exe"
    (tools / wrong).write_text("binary")
    assert _marp.find_marp(tmp_path) is None


def test_returns_none_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_marp.shutil, "which", lambda _n: None)
    assert _marp.find_marp(tmp_path) is None


# ------------------------------------------------------------ marp install

def test_asset_table_covers_this_platform() -> None:
    """Every platform FI runs on must have a download mapped, or
    --install-marp is dead weight exactly where it's needed."""
    import platform
    arch = {"aarch64": "arm64", "AMD64": "AMD64",
            "x86_64": "x86_64", "arm64": "arm64"}.get(
        platform.machine(), platform.machine())
    assert (sys.platform, arch) in launch._MARP_ASSET_NAMES


def test_asset_names_match_published_release() -> None:
    """Pinned against the real marp-cli release asset names."""
    for name in launch._MARP_ASSET_NAMES.values():
        assert name.startswith(f"marp-cli-v{launch._MARP_VERSION}-")
        assert name.endswith((".zip", ".tar.gz"))


def test_install_from_missing_file_is_an_error(tmp_path: Path) -> None:
    assert launch._install_marp_from_local(tmp_path / "nope.zip") == 1


def test_install_from_zip_places_binary(tmp_path: Path, monkeypatch) -> None:
    import zipfile
    archive = tmp_path / "marp-cli-v4.5.1-win.zip"
    exe = "marp.exe" if sys.platform == "win32" else "marp"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("LICENSE", "MIT")
        zf.writestr(exe, "#!binary")
    tools = tmp_path / "tools"
    assert launch._place_marp_binary(archive, tools) == 0
    assert (tools / exe).is_file()
    assert (tools / exe).read_text() == "#!binary"
    # No leftover partial files from the atomic-replace dance.
    assert not list(tools.glob(".*partial*"))


def test_install_from_archive_without_binary_fails(tmp_path: Path) -> None:
    import zipfile
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("README.md", "nothing here")
    assert launch._place_marp_binary(archive, tmp_path / "tools") == 1


# ----------------------------------------------------------------- doctor

def test_doctor_reports_everything_missing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The locked-down-laptop case: doctor must name each gap AND the fix,
    and still exit 0 -- 'you cannot make a poster here' is a successful
    diagnosis, not a tool failure."""
    monkeypatch.setattr("generation._pandoc.find_pandoc", lambda *a, **k: None)
    monkeypatch.setattr("generation._pdf_engine.find_pdf_engine", lambda *a, **k: None)
    monkeypatch.setattr("generation._html_pdf.find_html_browser", lambda *a, **k: None)
    monkeypatch.setattr("generation._marp.find_marp", lambda *a, **k: None)
    monkeypatch.setattr("core.passages._embed_model", lambda: None)

    assert launch._doctor() == 0
    out = capsys.readouterr().out
    assert "MISS" in out
    # paper_md needs nothing, and pptx is rendered in-process -- both must
    # still report OK on a machine with no tooling at all.
    assert "paper_md" in out and "slides (.pptx)" in out
    # Each gap carries its remedy.
    assert "pypandoc_binary" in out
    assert "--install-tectonic" in out
    assert "--install-marp" in out
    assert "sentence-transformers" in out
    # The silent-failure mode that started all this.
    assert "relevance filter INACTIVE" in out


def test_doctor_reports_all_present(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("generation._pandoc.find_pandoc", lambda *a, **k: "/p/pandoc")
    monkeypatch.setattr(
        "generation._pdf_engine.find_pdf_engine", lambda *a, **k: ("/x", "pdflatex"))
    monkeypatch.setattr(
        "generation._html_pdf.find_html_browser", lambda *a, **k: ("edge", "/e/edge"))
    monkeypatch.setattr("generation._marp.find_marp", lambda *a, **k: "/m/marp")
    monkeypatch.setattr("core.passages._embed_model", lambda: object())

    assert launch._doctor() == 0
    out = capsys.readouterr().out
    assert "MISS" not in out
    assert "Everything checked is available" in out
