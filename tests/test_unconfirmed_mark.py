"""The paper as delivered carries nothing of FI's about a sentence its last claim check found unconfirmed
(``Engine._state_unsupported_claims``): a ``wording`` claim stays as written, unmarked, and the record QUALITY.md reads
lists it; a sentence whose claim's fault takes it out is taken out. The marks and the note under the title an older FI put
in the paper are taken out (``Engine._take_out_old_marks``), and the receipts move with the paper. No model is called."""

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
OLD_MARK = "<!-- fi:unconfirmed --> (unconfirmed)<!-- /fi:unconfirmed -->"
OLD_NOTE = "<!-- fi:marks-note -->\n> **Not confirmed.** Some statements are marked.\n<!-- /fi:marks-note -->\n\n"


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


def _run(eng, path, claims=None) -> str:
    grounding = {"claims": _claims() if claims is None else claims, "status": "ok"}
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": grounding})  # type: ignore[arg-type]
    return path.read_text(encoding="utf-8")


def _old(paper: str = PAPER) -> str:
    """The paper as an older FI left it: its note under the title and a mark beside each standing claim."""
    return (paper.replace("## Abstract", OLD_NOTE + "## Abstract")
            .replace(SOFT + ".", SOFT + OLD_MARK + ".").replace(LISTED + ".", LISTED + OLD_MARK + "."))


def test_a_standing_claim_stays_as_written_unmarked_and_the_record_lists_it(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    text = _run(eng, path)
    assert SOFT + "." in text and "- " + LISTED + "." in text, "the sentences stay exactly as the writer wrote them"
    assert "(unconfirmed)" not in text and "fi:unconfirmed" not in text and "fi:marks-note" not in text
    assert "0.9991" not in text, "the sentence the check found wrong is taken out"
    assert "The study counts tiles. Done." in text and GOOD in text
    record = json.loads((eng.quest_root / paper_facts.UNSUPPORTED_RECORD).read_text(encoding="utf-8"))
    assert [c["claim"] for c in record["claims"]] == [SOFT, LISTED]


def test_a_second_pass_changes_nothing(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path)
    once = _run(eng, path)
    record = (eng.quest_root / paper_facts.UNSUPPORTED_RECORD).read_bytes()
    assert _run(eng, path) == once
    assert (eng.quest_root / paper_facts.UNSUPPORTED_RECORD).read_bytes() == record


def test_an_older_fis_marks_and_note_are_taken_out_byte_for_byte(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path, _old())
    plain_eng, plain_path = _engine(tmp_path / "other")
    assert _run(eng, path) == _run(plain_eng, plain_path)
    crlf_eng, crlf_path = _engine(tmp_path / "crlf")
    crlf_path.write_bytes(_old().replace("\n", "\r\n").encode("utf-8"))
    crlf_eng._take_out_old_marks(crlf_path)
    assert crlf_path.read_bytes() == PAPER.replace("\n", "\r\n").encode("utf-8"), "line endings stay as they were"


def test_a_call_with_no_answer_from_the_check_still_takes_an_older_fis_marks_out(tmp_path: Path) -> None:
    """A call returns early when the claim check has no answer: it takes the old marks out first."""
    eng, path = _engine(tmp_path, _old())
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": {"status": "failed"}})  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == PAPER


def test_no_writer_and_no_check_of_the_writers_text_reads_an_older_mark(tmp_path: Path) -> None:
    eng, _path = _engine(tmp_path)
    old = _old()
    for view in (paper_facts.to_placeholders(old), paper_facts.without_blocks(old)):
        assert "(unconfirmed)" not in view and "fi:marks-note" not in view and "Not confirmed" not in view


def test_taking_old_marks_out_moves_the_receipts_to_the_paper(tmp_path: Path) -> None:
    eng, path = _engine(tmp_path, _old())
    before = path.read_bytes()
    for check in ("claim_check", "citation_check"):
        receipts.write(eng.quest_root, check, status="pass", producer=check, started_at=receipts.now(),
                       inputs={"paper": before}, output="judged")
    eng._take_out_old_marks(path)
    after = path.read_bytes()
    assert after != before
    for check in ("claim_check", "citation_check"):
        rec = json.loads(receipts.path(eng.quest_root, check).read_text(encoding="utf-8"))
        assert rec["input_hashes"]["paper"] == receipts.sha256(after)


def test_taking_a_sentence_out_moves_the_citation_checks_receipt_too(tmp_path: Path) -> None:
    """Old code: only the claim check's receipt moved, so the citation check, which had just read the paper, read as not
    run on the final draft whenever FI took a sentence out at the last step (6 of 33 stored studies)."""
    eng, path = _engine(tmp_path)
    before = path.read_bytes()
    for check in ("claim_check", "citation_check"):
        receipts.write(eng.quest_root, check, status="pass", producer=check, started_at=receipts.now(),
                       inputs={"paper": before}, output="judged")
    _run(eng, path)
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
