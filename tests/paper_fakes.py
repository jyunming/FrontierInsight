"""Shared fakes for the paper generator's tests."""

from __future__ import annotations

from typing import Any, Callable


def fake_pandoc(monkeypatch: Any, fake_run: Callable[..., Any]) -> None:
    """Replace the paper generator's one pandoc run (``PaperGenerator._run_pandoc``) with
    ``fake_run(cmd, cwd=..., timeout=...)``, which returns what ``subprocess.run`` would
    (``returncode``, ``stdout``, ``stderr``) or raises as it would."""
    from generation.paper import PaperGenerator

    monkeypatch.setattr(
        PaperGenerator, "_run_pandoc",
        staticmethod(lambda cmd, out_dir, timeout_s: fake_run(cmd, cwd=str(out_dir), timeout=timeout_s)),
    )


def fake_html_run(monkeypatch: Any, fake_run: Callable[..., Any]) -> None:
    """Replace the browser fallback's pandoc and browser runs (``_html_pdf._run``) with
    ``fake_run(cmd, cwd=..., timeout=...)``."""
    import generation._html_pdf as html_pdf

    monkeypatch.setattr(html_pdf, "_run", lambda cmd, *, timeout, cwd=None: fake_run(cmd, cwd=cwd, timeout=timeout))


def fake_poster_run(monkeypatch: Any, fake_run: Callable[..., Any]) -> None:
    """Replace the poster's LaTeX run (``generation.poster._run``) with
    ``fake_run(cmd, cwd=..., timeout=...)``."""
    import generation.poster as poster

    monkeypatch.setattr(poster, "_run", lambda cmd, *, timeout, cwd=None: fake_run(cmd, cwd=cwd, timeout=timeout))
