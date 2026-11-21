"""FI's note that a paper carries an "(unconfirmed)" mark stands on the paper, the title slide and the poster header, from
one text (core/result_checks.py::marked_note), and the model never writes or removes it."""
from __future__ import annotations

import json
import types
from pathlib import Path

from core import engine_base as eb
from core import result_checks as rc
from generation import poster as poster_mod
from generation import slides as slides_mod

MARK = "<!-- fi:unconfirmed --> (unconfirmed)<!-- /fi:unconfirmed -->"
MARKED = f"# A study\n\n## Abstract\nThe rate is higher than expected{MARK}.\n"
CLEAN = "# A study\n\n## Abstract\nThe rate is higher than expected.\n"
WORDS_ONLY = "# A study\n\n## Abstract\nThe rate is higher than expected (unconfirmed).\n"
DECK = "---\nmarp: true\n---\n\n# A study\n\n## Subtitle\n\n---\n\n## Second\n\n- point\n"
SENTENCES = "Statements FI could not confirm are marked too."


def _art(root: Path, paper: str) -> types.SimpleNamespace:
    (root / "paper.md").write_text(paper, encoding="utf-8")
    return types.SimpleNamespace(paper_md=root / "paper.md", quest_root=root)


def _held_record(root: Path) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "equation_tests.json").write_text(json.dumps({
        "unconfirmed": {"E1": {"formula": "y = a x", "function": "f", "marked": True, "results": ["q_mean"]}}}),
        encoding="utf-8")


def test_a_sentence_mark_of_the_claim_check_gets_one_plain_note_and_a_clean_paper_none(tmp_path: Path) -> None:
    note = rc.marked_note(tmp_path, MARKED)
    assert note.startswith("**Not confirmed.**") and "statements" in note and "(unconfirmed)" in note
    assert rc.marked_note(tmp_path, CLEAN) == "" and rc.marked_note(tmp_path) == ""
    # only FI's marker counts: the words alone also stand in a draft the writer typed them in
    assert rc.marked_note(tmp_path, WORDS_ONLY) == ""
    # the note FI put under the title is not taken for a mark
    assert rc.marked_note(tmp_path, "<!-- fi:marks-note -->\n> " + note + "\n<!-- /fi:marks-note -->\n" + CLEAN) == ""


def test_the_quantity_text_is_short_and_a_sentence_mark_adds_one_sentence(tmp_path: Path) -> None:
    _held_record(tmp_path)
    quantity_only = rc.marked_note(tmp_path, CLEAN)
    assert "`q_mean`" in quantity_only and "written by hand" in quantity_only and SENTENCES not in quantity_only
    assert len(quantity_only) < 260
    assert rc.marked_note(tmp_path, MARKED) == quantity_only + " " + SENTENCES


def test_the_paper_note_follows_the_sentence_marks_and_clears_with_them(tmp_path: Path) -> None:
    once = eb._mark_unconfirmed_results(MARKED, tmp_path)
    assert once.index("fi:marks-note") < once.index("## Abstract") and "statements" in once
    assert eb._mark_unconfirmed_results(once, tmp_path) == once, "placed once, the same again"
    assert "fi:marks-note" not in eb._mark_unconfirmed_results(once.replace(MARK, ""), tmp_path)
    # the writer's draft that types the words itself gets no note: nothing backs them
    assert eb._mark_unconfirmed_results(WORDS_ONLY, tmp_path) == WORDS_ONLY


def test_a_paper_kept_with_crlf_keeps_it_when_the_note_is_placed_and_taken_out(tmp_path: Path) -> None:
    crlf = MARKED.replace("\n", "\r\n")
    once = eb._mark_unconfirmed_results(crlf, tmp_path)
    assert once.count("\r\n") == once.count("\n") and "fi:marks-note" in once
    assert eb._mark_unconfirmed_results(once.replace(MARK, ""), tmp_path) == CLEAN.replace("\n", "\r\n")


def test_the_title_slide_gets_the_note_once_after_the_author_line_and_the_rest_is_the_same(tmp_path: Path) -> None:
    deck = tmp_path / "slides.md"
    out = types.SimpleNamespace(author="A. Author", affiliation="", contact_email="", url="")
    deck.write_text(slides_mod._with_author_line(DECK, out), encoding="utf-8")
    art = _art(tmp_path, MARKED)
    slides_mod._place_marks_note(deck, art)
    text = deck.read_text(encoding="utf-8")
    note = rc.marked_note(tmp_path, MARKED)
    title = text.split("\n---\n")[1]  # the first slide after the front matter
    assert text.count(note) == 1 and note in title and title.index("A. Author") < title.index(note)
    assert text.replace("\n\n" + note, "", 1) == slides_mod._with_author_line(DECK, out)
    slides_mod._place_marks_note(deck, art)
    assert deck.read_text(encoding="utf-8") == text, "a deck that holds it is not touched"
    # the model wrote the title slide again without it: FI adds it last
    deck.write_text(DECK, encoding="utf-8")
    slides_mod._place_marks_note(deck, art)
    assert note in deck.read_text(encoding="utf-8").split("\n---\n")[1]


def test_a_deck_of_a_paper_with_no_mark_is_not_touched(tmp_path: Path) -> None:
    deck = tmp_path / "slides.md"
    deck.write_bytes(DECK.encode("utf-8"))
    slides_mod._place_marks_note(deck, _art(tmp_path, CLEAN))
    assert deck.read_bytes() == DECK.encode("utf-8")


def test_the_poster_header_holds_the_note_under_the_author_line_and_nothing_when_unmarked(tmp_path: Path) -> None:
    sheet = poster_mod._SHEETS["a1_portrait"]
    args = dict(headline="H", subtitle="", author_terms=("A. Author",), url="", band="", icon_ok=False, cjk_font=None)
    plain = poster_mod.PosterGenerator._template_values(sheet, **args)["authorline"]
    note = rc.marked_note(tmp_path, MARKED)
    with_note = poster_mod.PosterGenerator._template_values(sheet, note=note, **args)["authorline"]
    assert with_note.startswith(plain) and "Not confirmed" in with_note and "`" not in with_note
    assert poster_mod.PosterGenerator._template_values(sheet, note="", **args)["authorline"] == plain


def test_the_deliverables_read_the_papers_own_markers_which_the_delivered_copy_has_lost(tmp_path: Path) -> None:
    """``paper.md`` is delivered without FI's markers; ``paper/paper.md`` keeps them."""
    _held_record(tmp_path)
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / "paper.md").write_text(MARKED, encoding="utf-8")
    assert SENTENCES not in rc.marked_note(tmp_path, WORDS_ONLY)
    assert SENTENCES in rc.marked_note_of_deliverables(tmp_path, WORDS_ONLY)
    deck = tmp_path / "slides.md"
    deck.write_text(DECK, encoding="utf-8")
    slides_mod._place_marks_note(deck, _art(tmp_path, WORDS_ONLY))
    title = deck.read_text(encoding="utf-8").split("\n---\n")[1]
    assert "`q_mean`" in title and SENTENCES in title
