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


async def test_the_page_overrun_leaves_the_paper_and_the_receipts_of_the_checks_alone(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, "# T\n\n## Methods\nM.\n\n## Limitations\nFew.\n\n## References\n1. A.\n",
                        checks=("claim_check", "citation_check"))
    before = paper.read_bytes()
    assert await eng._state_page_overrun({}, paper, 4, 6) == 6
    assert paper.read_bytes() == before
    for check in ("claim_check", "citation_check"):
        assert _receipt_names(eng, paper, check), check


def _ledger(eng: Any, claims: list[str]) -> Path:
    path = eng.quest_root / "paper" / "claims.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"claim": c, "basis": "experiment", "evidence": ""} for c in claims]
    path.write_text(json.dumps({"claims": rows, "total": len(rows), "grounded": len(rows), "unsupported": [],
                                "status": "ok", "summary": ""}), encoding="utf-8")
    return path


_SENT = "Settling time rises with pond depth in every tested pond."
_KEEP = "The pond was sampled at dawn by a trained volunteer."


def test_a_claim_whose_sentence_fi_took_out_leaves_the_ledger(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, f"# Pond\n\n## Results\n{_SENT} {_KEEP}\n\n## References\n1. A.\n")
    ledger = _ledger(eng, [_SENT.rstrip("."), _KEEP.rstrip(".")])
    eng._prune_ledger_after_take_out([_SENT], paper.read_text(encoding="utf-8").replace(_SENT + " ", ""))
    rows = [r["claim"] for r in json.loads(ledger.read_text(encoding="utf-8"))["claims"]]
    assert rows == [_KEEP.rstrip(".")]  # the taken-out claim is gone, the one the paper still says stays
    assert "Settling time" not in (eng.quest_root / "paper" / "CLAIMS.md").read_text(encoding="utf-8")


def test_a_claim_still_in_the_paper_stays_on_the_ledger(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, f"# Pond\n\n## Results\n{_SENT}\n\n## References\n1. A.\n")
    ledger = _ledger(eng, [_SENT.rstrip(".")])
    before = ledger.read_text(encoding="utf-8")
    eng._prune_ledger_after_take_out(["An unrelated sentence taken out."], paper.read_text(encoding="utf-8"))
    assert ledger.read_text(encoding="utf-8") == before
    eng._prune_ledger_after_take_out([_SENT], paper.read_text(encoding="utf-8"))  # another sentence still says it
    assert ledger.read_text(encoding="utf-8") == before


def test_the_record_take_out_prunes_the_ledger_through_its_own_path(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, f"# Pond\n\n## Results\n{_SENT} Ponds were sampled.\n\n## References\n1. A.\n")
    ledger = _ledger(eng, [_SENT.rstrip(".")])
    assert eng._take_out_contradicted({"paper_md": str(paper)}, [("the record contradicts it", _SENT)]) == [_SENT]
    assert json.loads(ledger.read_text(encoding="utf-8"))["claims"] == []


def test_the_unfilled_take_out_prunes_the_ledger_too(tmp_path: Path) -> None:
    eng, paper = _quest(tmp_path, PAPER)
    ledger = _ledger(eng, ["The headline setting is within tolerance"])
    removed = [x for x in eng._take_out_unfilled_keys({"paper_md": str(paper)}) if x["how"] == "sentence removed"]
    assert removed
    _ledger(eng, [removed[0]["sentence"].strip(), "The headline setting is within tolerance"])
    eng._prune_ledger_after_take_out([removed[0]["sentence"]], paper.read_text(encoding="utf-8"))
    assert [r["claim"] for r in json.loads(ledger.read_text(encoding="utf-8"))["claims"]] == [
        "The headline setting is within tolerance"]
