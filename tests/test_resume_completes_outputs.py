"""Resume completes MISSING outputs without clobbering existing ones (PR-8b).

A --resume re-runs the generator pass. Previously it re-invoked every
generator unconditionally — re-running the LLM for slides/poster/speech that
had already rendered and overwriting good decks. Now the resume pass skips a
kind whose final deliverable is already on disk and generates only what's
missing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import launch as fi_launch
from core.engine import QuestArtifacts


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(
            kinds=["paper_pdf", "slides", "poster", "speech"],
            require_pdf=False,
        )
    )


def _install_fake_generators(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr(fi_launch, "_apply_paper_venue_override", lambda c, a: None)

    def mk(name: str, is_async: bool, retkey: str, fname: str):
        if is_async:
            class _G:
                def __init__(self, cfg):  # noqa: ANN001
                    pass

                async def generate(self, art, out_dir, *, supervisor):  # noqa: ANN001
                    calls.append(name)
                    return {retkey: out_dir / fname}
        else:
            class _G:  # type: ignore[no-redef]
                def __init__(self, cfg):  # noqa: ANN001
                    pass

                def generate(self, art, out_dir):  # noqa: ANN001
                    calls.append(name)
                    return {retkey: out_dir / fname}
        return _G

    monkeypatch.setattr(fi_launch, "PaperGenerator", mk("paper", False, "paper_pdf", "paper.pdf"))
    monkeypatch.setattr(fi_launch, "SlideGenerator", mk("slides", True, "slides_pdf", "slides.pdf"))
    monkeypatch.setattr(fi_launch, "PosterGenerator", mk("poster", True, "poster_pdf", "poster.pdf"))
    monkeypatch.setattr(fi_launch, "SpeechGenerator", mk("speech", True, "speech", "talk.md"))


@pytest.mark.asyncio
async def test_resume_skips_existing_and_completes_missing(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    _install_fake_generators(monkeypatch, calls)

    art = QuestArtifacts(
        quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md",
    )
    # Paper + slides already rendered; poster + speech are missing.
    (tmp_path / "paper.pdf").write_text("pdf", encoding="utf-8")
    (tmp_path / "slides.pdf").write_text("pdf", encoding="utf-8")

    written = await fi_launch._run_generators(
        _cfg(), art, supervisor=MagicMock(), skip_existing=True,
    )

    # Only the missing kinds were (re)generated.
    assert calls == ["poster", "speech"]
    # Existing deliverables are carried through untouched.
    assert written["paper_pdf"] == tmp_path / "paper.pdf"
    assert written["slides"] == tmp_path / "slides.pdf"


@pytest.mark.asyncio
async def test_fresh_run_regenerates_everything(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    _install_fake_generators(monkeypatch, calls)

    art = QuestArtifacts(
        quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md",
    )
    # Even with a stale paper.pdf on disk, a fresh run (skip_existing=False)
    # regenerates unconditionally — no behaviour change for non-resume runs.
    (tmp_path / "paper.pdf").write_text("stale", encoding="utf-8")

    await fi_launch._run_generators(
        _cfg(), art, supervisor=MagicMock(), skip_existing=False,
    )
    assert calls == ["paper", "slides", "poster", "speech"]


def test_existing_output_detects_final_deliverables(tmp_path: Path):
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path)
    assert fi_launch._existing_output(art, "paper_pdf") is None
    (tmp_path / "paper.pdf").write_text("x", encoding="utf-8")
    assert fi_launch._existing_output(art, "paper_pdf") == tmp_path / "paper.pdf"
    # slides accepts either the pdf or the html render.
    (tmp_path / "slides.html").write_text("x", encoding="utf-8")
    assert fi_launch._existing_output(art, "slides") == tmp_path / "slides.html"
    # An intermediate (slides.md) does NOT count as the final deliverable.
    assert fi_launch._existing_output(art, "poster") is None
