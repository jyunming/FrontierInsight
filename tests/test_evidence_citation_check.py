"""Every citation of the final draft is read against its source's retrieved text, last: a draft that changed after the
check (a citation put in by a later rewrite) was not read, and the evidence level says so."""
from __future__ import annotations

from pathlib import Path

from core import evidence
from core import receipts as _receipts
from tests.test_evidence import ON, _ready_quest, _state


def test_a_citation_check_that_did_not_read_the_final_draft_is_a_gap(tmp_path: Path) -> None:
    root = _ready_quest(tmp_path)
    paper = root / "paper" / "paper.md"
    paper.write_text("the final draft", encoding="utf-8")
    on = {**ON, "citation_check": "on"}
    for check in ("claim_check", "citation_check"):
        _receipts.write(root, check, status="pass", producer=check, started_at=_receipts.now(),
                        inputs={"paper": paper.read_bytes()}, output={"claims": []})
    assert evidence.assess(root, _state(paper_md=str(paper)), settings=on)["status"] == "publication_ready"
    paper.write_text("the final draft, and a citation a later rewrite added [4].", encoding="utf-8")
    _receipts.write(root, "claim_check", status="pass", producer="claim_check", started_at=_receipts.now(),
                    inputs={"paper": paper.read_bytes()}, output={"claims": []})
    got = evidence.assess(root, _state(paper_md=str(paper)), settings=on)
    assert got["status"] == "statistically_adequate"
    assert any("citation check did not run on the final draft" in g for g in got["gaps"]), got["gaps"]
    _receipts.path(root, "citation_check").unlink()
    got = evidence.assess(root, _state(paper_md=str(paper)), settings=on)
    assert any("citation check is not shown to have run" in g for g in got["gaps"]), got["gaps"]
    _receipts.write(root, "citation_check", status="unknown", producer="citation_check", started_at=_receipts.now(),
                    inputs={"paper": paper.read_bytes()}, output={}, error="the reader gave no verdict")
    got = evidence.assess(root, _state(paper_md=str(paper)), settings=on)
    assert any("could not read every citation" in g for g in got["gaps"]), got["gaps"]


def test_a_quest_that_does_not_run_the_citation_check_is_not_asked_for_it(tmp_path: Path) -> None:
    root = _ready_quest(tmp_path)
    assert evidence.assess(root, _state(), settings={**ON, "citation_check": "off"})["status"] == "publication_ready"
