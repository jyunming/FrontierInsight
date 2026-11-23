"""FI puts no note on the paper, the title slide or the poster header that a paper carries an "(unconfirmed)" mark: a note
an older FI placed is taken out of a kept paper and a kept deck, and the model's own words are never touched."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from core import paper_facts
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from generation import slides as slides_mod

OLD_NOTE = ("<!-- fi:marks-note -->\n> **Not confirmed.** Statements FI could not confirm are marked."
            "\n<!-- /fi:marks-note -->\n")
CLEAN = "# A study\n\n## Abstract\nThe rate is higher than expected.\n"
WORDS_ONLY = "# A study\n\n## Abstract\nThe rate is higher than expected (unconfirmed).\n"
DECK = "---\nmarp: true\n---\n\n# A study\n\n## Subtitle\n\n---\n\n## Second\n\n- point\n"
OLD_DECK = "---\nmarp: true\n---\n\n# A study\n\n" + OLD_NOTE + "\n## Subtitle\n\n---\n\n## Second\n\n- point\n"


def _art(root: Path, paper: str) -> types.SimpleNamespace:
    (root / "paper.md").write_text(paper, encoding="utf-8")
    return types.SimpleNamespace(paper_md=root / "paper.md", quest_root=root)


def test_an_older_note_under_the_title_is_taken_out_of_the_paper_and_the_writers_words_stay() -> None:
    old = CLEAN.replace("## Abstract", OLD_NOTE + "\n## Abstract")
    assert paper_facts.without_marks_note(old) == CLEAN
    assert paper_facts.without_marks_note(WORDS_ONLY) == WORDS_ONLY, "the writer's own words are not FI's note"
    for view in (paper_facts.to_placeholders(old), paper_facts.without_blocks(old)):
        assert "fi:marks-note" not in view and "Not confirmed" not in view and "## Abstract" in view


def test_an_older_note_on_a_kept_deck_is_taken_out_and_a_deck_without_one_is_not_touched(tmp_path: Path) -> None:
    deck = tmp_path / "slides.md"
    deck.write_text(OLD_DECK, encoding="utf-8")
    slides_mod._take_out_marks_note(deck)
    assert deck.read_text(encoding="utf-8") == DECK
    before = deck.stat().st_mtime_ns
    slides_mod._take_out_marks_note(deck)
    assert deck.read_text(encoding="utf-8") == DECK and deck.stat().st_mtime_ns == before


@pytest.mark.asyncio
async def test_a_kept_deck_loses_an_older_note_on_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
    cfg = Config(topic="t", title="t", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(kinds=["slides"], output_dir=tmp_path / "outputs"))
    art = _art(tmp_path, CLEAN)
    art.quest_id, art.figures_dir, art.raw_state = "q", None, {}
    (tmp_path / "slides.md").write_text(OLD_DECK, encoding="utf-8")
    await slides_mod.SlideGenerator(cfg).generate(art, tmp_path, keep_deck=True)
    assert (tmp_path / "slides.md").read_text(encoding="utf-8") == DECK


def test_the_author_line_stays_on_the_first_slide_with_text_even_when_a_later_slide_has_the_h1(tmp_path: Path) -> None:
    deck = "---\nmarp: true\n---\n\n## A study\n\n---\n\n# A later content slide\n"
    out = types.SimpleNamespace(author="A. Author", affiliation="", contact_email="", url="")
    slides = slides_mod._with_author_line(deck, out).split("\n---\n")
    assert "A. Author" in slides[1] and "A. Author" not in slides[2]
