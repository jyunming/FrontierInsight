"""A paper on disk with Windows line endings: the steps after the claim check that only take text out of the checked
draft, or add a sentence FI states itself, keep its line endings and move the checks' receipts to the paper they leave.
The p3 kinetics papers left the typed-number take-out with CR CR LF line ends, the next step read each as one blank line
more and rewrote the paper, and the page-limit sentence went in after the claim and citation checks without moving their
receipts: the delivered paper was not the one the receipts name. The line endings only go wrong on Windows (text mode
writes LF as-is elsewhere); the receipts go wrong everywhere. No model is called."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import receipts
from tests.test_plan_optimisation import _engine
from tests.test_unfilled_keys import PAPER


def _quest(tmp_path: Path, text: str, checks: tuple[str, ...] = ("claim_check",)) -> tuple[Any, Path]:
    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    paper = root / "paper.md"
    paper.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    for check in checks:
        receipts.write(root, check, status="pass", producer=check, started_at=receipts.now(),
                       inputs={"paper": paper.read_bytes()}, output={"claims": []})
    return eng, paper


def _receipt_names(eng: Any, paper: Path, check: str = "claim_check") -> bool:
    _status, record, problem = receipts.read(eng.quest_root, check)
    return not problem and record is not None and record["input_hashes"]["paper"] == receipts.sha256(paper.read_bytes())


def _one_line_ending(paper: Path) -> bool:
    data = paper.read_bytes()
    return b"\r\r\n" not in data and data.count(b"\r\n") == data.count(b"\n")


def test_taking_out_values_fi_could_not_fill_keeps_the_line_endings_and_moves_the_receipt(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, PAPER)
    assert eng._take_out_unfilled_keys({"paper_md": str(paper)})
    assert _one_line_ending(paper)
    assert _receipt_names(eng, paper)


def test_taking_out_typed_numbers_keeps_the_line_endings_and_moves_the_receipt(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, "# T\n\n## Results\n\nThe headline score was 0.987 across the settings.\n")
    taken = eng._take_out_typed_numbers({"paper_md": str(paper)}, {"grid": {"m": [20, 200]}})
    assert [n for n, _sentence in taken] == ["0.987"]
    assert "[number removed]" in paper.read_text(encoding="utf-8")
    assert _one_line_ending(paper)
    assert _receipt_names(eng, paper)


async def test_the_page_limit_sentence_after_the_checks_moves_their_receipts(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, "# T\n\n## Methods\nM.\n\n## Limitations\nFew.\n\n## References\n1. A.\n",
                        checks=("claim_check", "citation_check"))

    async def measure(path: Path, state: dict) -> dict:
        return {"pages": 6}

    eng._measure_draft_pages = measure
    assert await eng._state_page_overrun({}, paper, 4, 6) == 6
    assert "This paper is 6 pages, over the 4-page limit set for it" in paper.read_text(encoding="utf-8")
    for check in ("claim_check", "citation_check"):
        assert _receipt_names(eng, paper, check), check
        assert "page count" in json.loads(receipts.path(eng.quest_root, check).read_text(encoding="utf-8"))["detail"]
