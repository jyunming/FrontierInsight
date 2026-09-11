"""Tiered pandoc discovery (``generation/_pandoc.py``).

The tiers exist so a locked-down host — no admin, no PATH edit — can still
render paper.pdf with nothing but ``pip install pypandoc_binary``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from generation import _pandoc


def test_path_wins_over_other_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit system install is what the user most likely intends."""
    monkeypatch.setattr(_pandoc.shutil, "which", lambda _n: "/usr/bin/pandoc")
    monkeypatch.setattr(_pandoc, "_pypandoc_pandoc", lambda: "/bundled/pandoc")
    assert _pandoc.find_pandoc() == "/usr/bin/pandoc"


def test_falls_back_to_repo_local_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tools/pandoc[.exe]`` mirrors the documented tectonic install path."""
    monkeypatch.setattr(_pandoc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(_pandoc, "_pypandoc_pandoc", lambda: None)
    tools = tmp_path / "tools"
    tools.mkdir()
    name = "pandoc.exe" if sys.platform == "win32" else "pandoc"
    (tools / name).write_text("binary")
    assert _pandoc.find_pandoc(tmp_path) == str(tools / name)


def test_ignores_wrong_platform_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stray binary for the other OS must not be returned: exec'ing it
    fails with a format error, which is far more confusing than "not found".
    """
    monkeypatch.setattr(_pandoc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(_pandoc, "_pypandoc_pandoc", lambda: None)
    tools = tmp_path / "tools"
    tools.mkdir()
    wrong = "pandoc" if sys.platform == "win32" else "pandoc.exe"
    (tools / wrong).write_text("binary")
    assert _pandoc.find_pandoc(tmp_path) is None


def test_falls_back_to_pypandoc_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pip-only tier — the whole point of the change."""
    monkeypatch.setattr(_pandoc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(_pandoc, "_pypandoc_pandoc", lambda: "/site/pypandoc/files/pandoc")
    assert _pandoc.find_pandoc(tmp_path) == "/site/pypandoc/files/pandoc"


def test_returns_none_when_nothing_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_pandoc.shutil, "which", lambda _n: None)
    monkeypatch.setattr(_pandoc, "_pypandoc_pandoc", lambda: None)
    assert _pandoc.find_pandoc(tmp_path) is None


def test_pypandoc_bare_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pypandoc.get_pandoc_path()`` returns the bare string "pandoc" when it
    resolved through PATH. We only reach this tier *after* PATH failed, so a
    bare name means "nothing new to offer" — returning it would hand callers
    an unrunnable command. Regression test: the first implementation shipped
    this bug and it only surfaced on a machine with a system pandoc.
    """
    fake = type(sys)("pypandoc")
    fake.__file__ = str(Path(__file__).parent / "no_such_pkg" / "__init__.py")
    fake.get_pandoc_path = lambda: "pandoc"  # bare name, not a path
    monkeypatch.setitem(sys.modules, "pypandoc", fake)
    assert _pandoc._pypandoc_pandoc() is None
