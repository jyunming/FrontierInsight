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


def _cfg(visual_check: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        output=SimpleNamespace(
            kinds=["paper_pdf", "slides", "poster", "speech"],
            require_pdf=False,
            visual_check=visual_check,
        )
    )


def _record_checks(monkeypatch) -> list[tuple[str, str]]:
    checked: list[tuple[str, str]] = []

    async def fake_check(cfg, kind, pdf, quest_root, *, supervisor):  # noqa: ANN001
        checked.append((kind, pdf.name))
        return {"kind": kind, "transport": "images", "findings": [], "measured": {"findings": []}}

    async def fake_check_and_redo(cfg, kind, pdf, quest_root, regenerate, *, supervisor):  # noqa: ANN001
        return await fake_check(cfg, kind, pdf, quest_root, supervisor=supervisor)

    async def fake_check_pptx(cfg, pptx, quest_root, *, supervisor):  # noqa: ANN001
        return await fake_check(cfg, "slides_pptx", pptx, quest_root, supervisor=supervisor)

    monkeypatch.setattr(fi_launch, "check_pdf", fake_check)
    monkeypatch.setattr(fi_launch, "check_and_redo", fake_check_and_redo)
    monkeypatch.setattr(fi_launch, "check_pptx", fake_check_pptx)
    return checked


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
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5\n...\n%%EOF\n")
    (tmp_path / "slides.pdf").write_bytes(b"%PDF-1.5\n...\n%%EOF\n")

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


@pytest.mark.asyncio
async def test_the_visual_check_runs_on_each_pdf_the_pass_made(tmp_path: Path, monkeypatch, capsys):
    _install_fake_generators(monkeypatch, [])
    checked = _record_checks(monkeypatch)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    await fi_launch._run_generators(_cfg(visual_check=True), art, supervisor=MagicMock())
    assert checked == [("paper", "paper.pdf"), ("slides", "slides.pdf"), ("poster", "poster.pdf")]
    out = capsys.readouterr().out
    assert "[FI] visual check poster: 0 problem(s) seen on the pages, 0 measured" in out


@pytest.mark.asyncio
async def test_a_resume_checks_only_the_outputs_it_made(tmp_path: Path, monkeypatch):
    _install_fake_generators(monkeypatch, [])
    checked = _record_checks(monkeypatch)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5\n...\n%%EOF\n")
    (tmp_path / "slides.pdf").write_bytes(b"%PDF-1.5\n...\n%%EOF\n")
    await fi_launch._run_generators(_cfg(visual_check=True), art, supervisor=MagicMock(), skip_existing=True)
    assert checked == [("poster", "poster.pdf")]


@pytest.mark.asyncio
async def test_the_pptx_is_checked_after_the_slides_have_settled(tmp_path: Path, monkeypatch, capsys):
    _install_fake_generators(monkeypatch, [])

    class _Slides:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        async def generate(self, art, out_dir, *, supervisor, feedback=""):  # noqa: ANN001
            return {"slides_pdf": out_dir / "slides.pdf", "slides_pptx": out_dir / "slides.pptx"}

    monkeypatch.setattr(fi_launch, "SlideGenerator", _Slides)
    checked = _record_checks(monkeypatch)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    await fi_launch._run_generators(_cfg(visual_check=True), art, supervisor=MagicMock())
    assert checked == [
        ("paper", "paper.pdf"), ("slides", "slides.pdf"), ("poster", "poster.pdf"), ("slides_pptx", "slides.pptx"),
    ]
    assert "[FI] visual check slides.pptx: 0 problem(s) seen on the pages, 0 measured" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_slides_and_poster_are_redone_with_the_feedback(
    tmp_path: Path, monkeypatch, capsys,
):
    monkeypatch.setattr(fi_launch, "_apply_paper_venue_override", lambda c, a: None)
    made: list[tuple[str, str]] = []

    def generator(name: str, key: str, fname: str):
        class _G:
            def __init__(self, cfg):  # noqa: ANN001
                pass

            async def generate(self, art, out_dir, *, supervisor, feedback=""):  # noqa: ANN001
                made.append((name, feedback))
                return {key: out_dir / fname}

        return _G

    class _Paper:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        def generate(self, art, out_dir):  # noqa: ANN001
            made.append(("paper", ""))
            return {"paper_pdf": out_dir / "paper.pdf"}

    monkeypatch.setattr(fi_launch, "PaperGenerator", _Paper)
    monkeypatch.setattr(fi_launch, "SlideGenerator", generator("slides", "slides_pdf", "slides.pdf"))
    monkeypatch.setattr(fi_launch, "PosterGenerator", generator("poster", "poster_pdf", "poster.pdf"))
    monkeypatch.setattr(fi_launch, "SpeechGenerator", generator("speech", "speech", "talk.md"))
    checked_only: list[str] = []

    async def fake_check(cfg, kind, pdf, quest_root, *, supervisor):  # noqa: ANN001
        checked_only.append(kind)
        return {"transport": "images", "findings": [], "measured": {"findings": []}}

    redone: list[str] = []

    async def fake_check_and_redo(cfg, kind, pdf, quest_root, regenerate, *, supervisor):  # noqa: ANN001
        redone.append(kind)
        if kind != "paper":  # the paper's repair has its own test
            await regenerate(f"fix the {kind}")
        return {
            "transport": "images", "findings": [], "measured": {"findings": []},
            "attempts": [{"attempt": 0, "kept": False}, {"attempt": 1, "kept": True}],
        }

    monkeypatch.setattr(fi_launch, "check_pdf", fake_check)
    monkeypatch.setattr(fi_launch, "check_and_redo", fake_check_and_redo)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    await fi_launch._run_generators(_cfg(visual_check=True), art, supervisor=MagicMock())
    assert checked_only == [] and redone == ["paper", "slides", "poster"]
    assert ("slides", "fix the slides") in made and ("poster", "fix the poster") in made
    assert "[FI] visual check poster: 0 problem(s) seen on the pages, 0 measured; redone 1 time(s), kept redo 1" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_the_paper_is_repaired_by_recompiling_one_line_taller_per_attempt(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fi_launch, "_apply_paper_venue_override", lambda c, a: None)
    compiled: list[tuple[str, int]] = []

    class _Paper:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        def generate(self, art, out_dir):  # noqa: ANN001
            return {"paper_md": out_dir / "paper.md", "paper_pdf": out_dir / "paper.pdf"}

        def _compile_pdf(self, paper_md, out_dir, *, extra_lines=0):  # noqa: ANN001
            compiled.append((paper_md.name, extra_lines))
            return out_dir / "paper.pdf", None

    class _Nothing:
        def __init__(self, cfg):  # noqa: ANN001
            pass

        async def generate(self, art, out_dir, *, supervisor, feedback=""):  # noqa: ANN001
            return {}

    monkeypatch.setattr(fi_launch, "PaperGenerator", _Paper)
    for name in ("SlideGenerator", "PosterGenerator", "SpeechGenerator"):
        monkeypatch.setattr(fi_launch, name, _Nothing)

    async def fake_check_and_redo(cfg, kind, pdf, quest_root, regenerate, *, supervisor):  # noqa: ANN001
        await regenerate("the last page holds one line")
        await regenerate("the last page holds one line")
        return {"transport": "images", "findings": [], "measured": {"findings": []}}

    monkeypatch.setattr(fi_launch, "check_and_redo", fake_check_and_redo)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "source.md")
    await fi_launch._run_generators(_cfg(visual_check=True), art, supervisor=MagicMock())
    assert compiled == [("paper.md", 1), ("paper.md", 2)]


@pytest.mark.asyncio
async def test_the_visual_check_can_be_turned_off(tmp_path: Path, monkeypatch):
    _install_fake_generators(monkeypatch, [])
    checked = _record_checks(monkeypatch)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, paper_md=tmp_path / "paper.md")
    await fi_launch._run_generators(_cfg(visual_check=False), art, supervisor=MagicMock())
    assert checked == []


def test_existing_output_detects_final_deliverables(tmp_path: Path):
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path)
    assert fi_launch._existing_output(art, "paper_pdf") is None
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-1.5\n...\n%%EOF\n")
    assert fi_launch._existing_output(art, "paper_pdf") == tmp_path / "paper.pdf"
    # slides accepts either the pdf or the html render.
    (tmp_path / "slides.html").write_text("x", encoding="utf-8")
    assert fi_launch._existing_output(art, "slides") == tmp_path / "slides.html"
    # An intermediate (slides.md) does NOT count as the final deliverable.
    assert fi_launch._existing_output(art, "poster") is None


# --- A2: don't trust partial/corrupt artifacts from an interrupted run --------


def test_existing_output_rejects_truncated_or_empty_pdf(tmp_path: Path):
    """A 0-byte or non-%PDF file (killed mid-compile) must NOT count as
    produced — resume regenerates it instead of shipping a broken PDF."""
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"")  # 0-byte
    assert fi_launch._existing_output(art, "paper_pdf") is None
    pdf.write_bytes(b"%!PS truncated not a pdf")  # wrong magic
    assert fi_launch._existing_output(art, "paper_pdf") is None
    pdf.write_bytes(b"%PDF-1.7\n... body ...\n%%EOF\n")  # valid
    assert fi_launch._existing_output(art, "paper_pdf") == pdf


def test_existing_output_text_deliverable_needs_nonempty(tmp_path: Path):
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path)
    talk = tmp_path / "talk.md"
    talk.write_text("", encoding="utf-8")  # empty
    assert fi_launch._existing_output(art, "speech") is None
    talk.write_text("# Talk\n\nHello.\n", encoding="utf-8")
    assert fi_launch._existing_output(art, "speech") == talk
