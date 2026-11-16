"""The sentences the page trim takes out of a draft move to the supplement; none is lost.

A draft a little over its page limit loses a few background and discussion sentences (core/paper_trim.py). They used to
be deleted: the delivered paper no longer said them and nothing else did. Now each goes to ``paper/supplement.md``
under the section it came from, the paper's Limitations say how many moved and where, and a later draft gets them back
word for word where the text around them is unchanged (``Engine._supplement_reset``); one that cannot be placed stays
in the supplement. At every point a sentence the trim took is in the paper or in the supplement.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import supplement
from tests.test_paper_trim import _cands, _delivered, _paper, _put, _render_by_words, _with_chat
from tests.test_page_limit import _engine, _quiet_log, _review


def _trimmed(tmp_path: Path, monkeypatch) -> tuple[object, Path, str, list[str]]:  # noqa: ANN001
    """An engine whose draft was a hundred words over and was trimmed by the review; the draft as written, and the
    sentences the trim moved."""
    eng = _engine(tmp_path)
    written = _delivered(_paper(with_lists=False))
    paper = _put(tmp_path, written)
    _render_by_words(monkeypatch, fits_at=len(written.split()) - 100, last_page_empty=0.8)
    _with_chat(eng, json.dumps({"delete": [c.n for c in _cands(written)][::-1]}))
    _quiet_log(eng)
    patch = _review(eng, tmp_path)
    moved = patch["review"]["page_limit"]["trimmed"]["sentences"]
    assert len(moved) >= 2
    return eng, paper, written, moved


def test_a_trimmed_sentence_moves_to_the_supplement_under_its_section_and_the_paper_says_so(
    tmp_path: Path, monkeypatch,
) -> None:
    _eng, paper, written, moved = _trimmed(tmp_path, monkeypatch)
    now = paper.read_text(encoding="utf-8")
    shown = (tmp_path / "paper" / "supplement.md").read_text(encoding="utf-8")
    assert "## Sentences moved for length" in shown
    for sentence in moved:
        assert sentence not in now and sentence in written
        assert f"- {' '.join(sentence.split())}" in shown  # word for word, as a line of its own
    # each under the section it came from
    for record in supplement.sentences(tmp_path):
        for x in record["sentences"]:
            part = shown.split(f"**{x['section']}**")[1].split("**")[0]
            assert x["text"] in part
    # the paper's Limitations count them and point to the supplement
    limits = now.split("### Limitations")[1].split("## Conclusion")[0]
    assert supplement.moved_sentences_line(len(moved)) in limits
    assert supplement.sentence_count(tmp_path) == len(moved)


def test_the_next_draft_gets_the_sentences_back_word_for_word_and_the_record_goes(tmp_path: Path, monkeypatch) -> None:
    eng, paper, written, moved = _trimmed(tmp_path, monkeypatch)
    assert eng._supplement_reset({"paper_md": str(paper)}, paper) is True  # type: ignore[attr-defined]
    back = paper.read_text(encoding="utf-8")
    for sentence in moved:
        assert back.count(sentence) == 1
    assert back == written  # the pointer line goes with them: FI's Limitations are written from the records again
    assert not (tmp_path / supplement.TRIM_RECORD).exists()
    assert not (tmp_path / "paper" / "supplement.md").exists()


def test_a_sentence_whose_text_around_it_changed_stays_in_the_supplement(tmp_path: Path, monkeypatch) -> None:
    eng, paper, _written, moved = _trimmed(tmp_path, monkeypatch)
    places = supplement.sentences(tmp_path)
    assert len({p["step"] for p in places}) >= 2  # taken out in more than one round
    # the next draft rewrote the words just before a place of the last round
    now = paper.read_text(encoding="utf-8")
    first = next(p for p in places if p["step"] == max(q["step"] for q in places) and p["left"].strip()
                 and now.count(p["left"] + p["right"]) == 1)
    left = first["left"]
    paper.write_text(now.replace(left + first["right"], left.upper() + first["right"]), encoding="utf-8")
    eng._supplement_reset({"paper_md": str(paper)}, paper)  # type: ignore[attr-defined]
    back = paper.read_text(encoding="utf-8")
    kept = supplement.sentences(tmp_path)
    assert first["text"] in [p["text"] for p in kept]
    for x in first["sentences"]:
        assert x["text"] not in back  # not in the paper ...
    # ... but still delivered: every moved sentence is in the paper or the supplement, once
    eng._write_supplement({"paper_md": str(paper)}, 4)  # type: ignore[attr-defined]
    shown = (tmp_path / "paper" / "supplement.md").read_text(encoding="utf-8")
    for sentence in moved:
        assert back.count(sentence) + shown.count(f"- {' '.join(sentence.split())}") == 1, sentence
    assert len(kept) < len(places)  # the places whose text around them is unchanged went back


def test_a_place_the_draft_already_holds_is_neither_put_back_nor_kept() -> None:
    place = {"text": "A sentence. ", "left": "Before. ", "right": "After.",
             "sentences": [{"text": "A sentence.", "section": "Introduction"}]}
    assert supplement.put_back_sentences("Before. A sentence. After.", [place]) == ("Before. A sentence. After.", [])
    assert supplement.put_back_sentences("Before. After.", [place]) == ("Before. A sentence. After.", [])
    # the text around it found twice: where it goes is not known, it stays moved
    text = "Before. After. Before. After."
    assert supplement.put_back_sentences(text, [place]) == (text, [place])


def test_two_places_at_one_spot_both_stay_moved() -> None:
    a = {"text": "One. ", "left": "x. ", "right": "y.", "sentences": [{"text": "One.", "section": "S"}]}
    b = {"text": "Two. ", "left": "x. ", "right": "y.", "sentences": [{"text": "Two.", "section": "S"}]}
    assert supplement.put_back_sentences("x. y.", [a, b]) == ("x. y.", [a, b])


def test_the_pointer_line_counts_the_moved_sentences() -> None:
    assert supplement.moved_sentences_line(0) == ""
    one = supplement.moved_sentences_line(1)
    assert "FI moved 1 sentence of" in one and "it stands there" in one and supplement.SHOWN in one
    assert "FI moved 3 sentences of" in supplement.moved_sentences_line(3)
    # with nothing else moved the pointer is that line alone; with FI's items moved too it says the supplement holds both
    assert supplement.limitations_pointer(0, [], moved_sentences=2) == supplement.moved_sentences_line(2)
    both = supplement.limitations_pointer(3, [], moved_sentences=2)
    assert both.endswith(". The supplement also holds 2 sentences of the paper's text, moved there for length.")
    assert "also holds" not in supplement.limitations_pointer(3, [])


def test_a_reset_that_fails_after_putting_sentences_back_leaves_the_paper_and_the_supplement_as_they_were(
    tmp_path: Path, monkeypatch,
) -> None:
    eng, paper, _written, moved = _trimmed(tmp_path, monkeypatch)
    before = {f: f.read_bytes() for f in (paper, tmp_path / "paper" / "supplement.md", tmp_path / supplement.TRIM_RECORD)}

    def _fails(*_a, **_k):  # noqa: ANN002, ANN003, ANN202 - FI's blocks cannot be written again (a full disk)
        raise OSError("disk full")

    monkeypatch.setattr(eng, "_restate_limitations", _fails)
    assert eng._supplement_reset({"paper_md": str(paper)}, paper) is False  # type: ignore[attr-defined]
    assert {f: f.read_bytes() for f in before} == before
    shown = (tmp_path / "paper" / "supplement.md").read_text(encoding="utf-8")
    now = paper.read_text(encoding="utf-8")
    for sentence in moved:  # once, in the supplement only: never in both
        assert now.count(sentence) + shown.count(f"- {' '.join(sentence.split())}") == 1, sentence


def test_a_shortened_paper_that_cannot_be_written_leaves_no_sentence_in_the_supplement(tmp_path: Path, monkeypatch) -> None:
    import inspect

    eng = _engine(tmp_path)
    written = _delivered(_paper(with_lists=False))
    paper = _put(tmp_path, written)
    _render_by_words(monkeypatch, fits_at=len(written.split()) - 100, last_page_empty=0.8)
    _with_chat(eng, json.dumps({"delete": [c.n for c in _cands(written)][::-1]}))
    _quiet_log(eng)
    real = Path.write_text

    def _write(self: Path, *a, **k):  # noqa: ANN002, ANN003, ANN202 - the trimmed paper cannot be written (a full disk)
        if self == paper and inspect.stack()[1].function == "_trim_body_to_fit":
            raise OSError("disk full")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", _write)
    _review(eng, tmp_path)
    now = paper.read_text(encoding="utf-8")
    for c in _cands(written):  # every offered sentence still in the paper, and none in the supplement
        assert c.text in now
    shown = tmp_path / "paper" / "supplement.md"
    assert not shown.exists() or "Sentences moved for length" not in shown.read_text(encoding="utf-8")
    assert supplement.sentence_count(tmp_path) == 0


def test_a_reset_whose_records_cannot_be_cleared_leaves_everything_as_it_was(tmp_path: Path, monkeypatch) -> None:
    eng, paper, _written, _moved = _trimmed(tmp_path, monkeypatch)
    before = {f: f.read_bytes() for f in (paper, tmp_path / "paper" / "supplement.md", tmp_path / supplement.TRIM_RECORD)}

    def _fails(*_a, **_k):  # noqa: ANN002, ANN003, ANN202 - the supplement's record cannot be cleared (access denied)
        raise OSError("access denied")

    monkeypatch.setattr(supplement, "save_sentences", _fails)
    assert eng._supplement_reset({"paper_md": str(paper)}, paper) is False  # type: ignore[attr-defined]
    assert {f: f.read_bytes() for f in before} == before


def test_a_record_without_sentences_counts_none(tmp_path: Path) -> None:
    supplement.save_sentences(tmp_path, [{"text": "x", "left": "", "right": "", "step": 1, "sentences": []},
                                         {"text": "y", "left": "", "right": "", "step": 1}])
    assert supplement.sentence_count(tmp_path) == 0  # the Limitations never count a sentence the supplement does not show
