"""FI's mark beside a sentence its last claim check found unconfirmed (``Engine._mark_unconfirmed``,
``core/paper_facts.py`` ``UNCONFIRMED_MARK``): at the last step before delivery, each sentence the Limitations list as
unconfirmed (a claim still standing) gets a short mark where it stands, placed by the claim's stored passage as the
take-out places a sentence. Nothing is removed and no model is called; every other call takes the marks out, and neither
the writer nor a check of the writer's text ever reads one. No model is called in these tests."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts
from core import receipts

CUT = "The fit stays above 0.9991 in every case."
SOFT = "The tile count clearly rises with the board size [1]"
LISTED = "The sorting order stays stable"
GOOD = "The board held 12 tiles."
PAPER = (f"# T\n\n## Abstract\nThe study counts tiles. {CUT} Done.\n\n## Results\n{SOFT}. {GOOD}\n\n- {LISTED}.\n\n"
         "## Limitations\nFew settings were run.\n\n## References\n1. A.\n")
MARK = " (unconfirmed; see Limitations)"


def _claim(text: str, basis: str, passage: str | None = None, fault: str | None = None) -> dict:
    where = [{"id": "1.1", "section": "S", "passage": passage}] if passage else []
    return {"claim": text, "basis": basis, "fault": fault, "citation_index": None, "quote": "", "evidence": "x",
            "where": where}


def _claims() -> list[dict]:
    return [_claim(CUT, "unsupported", CUT, "number"),
            _claim(SOFT, "unsupported", SOFT + ".", "wording"),
            _claim(LISTED, "unsupported", LISTED + ".", "wording"),
            _claim(GOOD, "experiment")]


def _engine(tmp_path: Path, paper: str = PAPER):
    from tests.test_plan_optimisation import _engine as make

    eng = make(tmp_path, [])
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    path = eng.quest_root / "paper.md"
    path.write_text(paper, encoding="utf-8")
    return eng, path


def _run(eng, path, claims=None, *, mark: bool = True) -> str:
    grounding = {"claims": _claims() if claims is None else claims, "status": "ok"}
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": grounding}, mark=mark)  # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


def _body(text: str) -> str:
    return paper_facts.without_marks(text).split("## Limitations")[0]


def test_each_sentence_listed_as_unconfirmed_is_marked_where_it_stands(tmp_path: Path) -> None:
    """Old code: the two sentences stayed in the paper with nothing beside them; only the Limitations listed them."""
    eng, path = _engine(tmp_path)
    body = _body(_run(eng, path))
    assert SOFT + MARK + "." in body  # before the full stop, after the citation bracket
    assert "- " + LISTED + MARK + "." in body  # a list item
    assert body.count(MARK) == 2
    assert "0.9991" not in body  # the sentence the check found wrong is taken out, not marked
    assert "The study counts tiles. Done." in body and GOOD in body and GOOD + MARK not in body


def test_marking_twice_changes_nothing(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    once = _run(eng, path)
    record = (eng.quest_root / paper_facts.UNSUPPORTED_RECORD).read_bytes()
    assert _run(eng, path) == once
    assert (eng.quest_root / paper_facts.UNSUPPORTED_RECORD).read_bytes() == record


def test_the_mark_is_only_added_and_any_other_call_takes_it_out(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    plain_eng, plain_path = _engine(tmp_path / "other")
    unmarked = _run(plain_eng, plain_path, mark=False)
    marked = _run(eng, path)
    assert paper_facts.without_unconfirmed(marked) == unmarked
    assert _run(eng, path, mark=False) == unmarked


def test_a_sentence_no_longer_listed_loses_its_mark(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    _run(eng, path)
    body = _body(_run(eng, path, [c for c in _claims() if c["claim"] != LISTED]))
    assert SOFT + MARK + "." in body and LISTED + MARK not in body and body.count(MARK) == 1


def test_a_claim_fi_cannot_place_exactly_gets_no_mark_and_stays_listed(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    text = _run(eng, path, [_claim(SOFT, "unsupported", "A sentence the paper does not hold.", "wording")])
    assert MARK not in _body(text)
    assert "read it as unconfirmed" in paper_facts.without_marks(text).split("## Limitations")[1]


def test_two_claims_on_one_sentence_give_one_mark(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    body = _body(_run(eng, path, [_claim(SOFT, "unsupported", SOFT + ".", "wording"),
                                  _claim("The count rises", "unsupported", SOFT + ".", "wording")]))
    assert body.count(MARK) == 1


def test_a_sentence_taken_out_is_not_marked_even_with_a_second_claim_on_it(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    body = _body(_run(eng, path, [_claim(CUT, "unsupported", CUT, "number"),
                                  _claim("The fit stays high", "unsupported", CUT, "wording")]))
    assert "0.9991" not in body and MARK not in body


def test_the_same_sentence_twice_is_marked_twice(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path, PAPER.replace("The study counts tiles.", f"The study counts tiles. {SOFT}."))
    body = _body(_run(eng, path, [_claim(SOFT, "unsupported", SOFT + ".", "wording")]))
    assert body.count(SOFT + MARK + ".") == 2


def test_no_writer_and_no_check_of_the_writers_text_reads_the_mark(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    marked = _run(eng, path)
    assert MARK.strip() not in paper_facts.to_placeholders(marked)  # the passage editor and every rewrite
    assert MARK.strip() not in paper_facts.without_blocks(marked)  # the claim, statistics and record checks
    assert eng._kept_draft({"paper_md": str(path)}).count("unconfirmed;") == 0  # type: ignore[arg-type]


def test_a_rendered_paper_keeps_its_marks(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    marked = _run(eng, path)
    assert paper_facts.render(marked, eng.quest_root).text.count(paper_facts.UNCONFIRMED_MARK) == 2


def test_the_receipts_move_to_the_marked_paper(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    _run(eng, path, mark=False)
    before = path.read_bytes()
    for check in ("claim_check", "citation_check"):
        receipts.write(eng.quest_root, check, status="pass", producer=check, started_at=receipts.now(),
                       inputs={"paper": before}, output="judged")
    _run(eng, path)
    after = path.read_bytes()
    assert after != before
    for check in ("claim_check", "citation_check"):
        rec = json.loads(receipts.path(eng.quest_root, check).read_text(encoding="utf-8"))
        assert rec["input_hashes"]["paper"] == receipts.sha256(after)


def test_the_last_step_before_delivery_marks_the_paper(tmp_path: Path, monkeypatch) -> None:
    """The citation check's step is where the paper is delivered from: its call marks (every earlier one does not)."""
    eng, path = _engine(tmp_path)

    async def no_check(state):  # noqa: ANN001
        return {"citation_next": "done"}

    monkeypatch.setattr(eng, "_check_citations", no_check)
    monkeypatch.setattr("core.engine_write.resolve_page_limit", lambda cfg: None)
    import asyncio

    asyncio.run(eng._node_citation_check({"paper_md": str(path), "claim_grounding": {"claims": _claims(),  # type: ignore[arg-type]
                                                                                       "status": "ok"}}))
    assert _body(path.read_text(encoding="utf-8")).count(MARK) == 2


def test_taking_a_sentence_out_moves_the_citation_checks_receipt_too(tmp_path: Path) -> None:
    """Old code: only the claim check's receipt moved, so the citation check, which had just read the paper, read as not
    run on the final draft whenever FI took a sentence out at the last step (6 of 33 stored studies)."""
    eng, path = _engine(tmp_path)
    before = path.read_bytes()
    for check in ("claim_check", "citation_check"):
        receipts.write(eng.quest_root, check, status="pass", producer=check, started_at=receipts.now(),
                       inputs={"paper": before}, output="judged")
    _run(eng, path, mark=False)
    after = path.read_bytes()
    assert b"0.9991" not in after  # a sentence was taken out
    for check in ("claim_check", "citation_check"):
        rec = json.loads(receipts.path(eng.quest_root, check).read_text(encoding="utf-8"))
        assert rec["input_hashes"]["paper"] == receipts.sha256(after)


def test_a_receipt_for_an_earlier_draft_does_not_move(tmp_path: Path) -> None:
    """Only a receipt for the paper as it was read moves: one for an earlier draft still reads as not for this paper."""
    eng, path = _engine(tmp_path)
    receipts.write(eng.quest_root, "citation_check", status="pass", producer="citation_check", started_at=receipts.now(),
                   inputs={"paper": b"an earlier draft"}, output="judged")
    _run(eng, path)
    rec = json.loads(receipts.path(eng.quest_root, "citation_check").read_text(encoding="utf-8"))
    assert rec["input_hashes"]["paper"] == receipts.sha256(b"an earlier draft")
