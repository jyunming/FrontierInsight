"""A resume after the citation check wrote its record but before the graph saved the state: the paper on disk is renumbered,
the state's sources are not. The resumed pass puts the state back as a first pass leaves it, asks nothing it already
asked, and runs the claim check again only because its receipt is not for the text as it is."""
from __future__ import annotations

import json

from core import citation_support as cs
from core import receipts as _receipts
from tests.test_citation_check_node import Replies, _engine, _literature, _paper, _run


def test_a_resume_after_the_record_was_written_puts_the_state_back_and_asks_nothing_again(tmp_path) -> None:
    eng = _engine(tmp_path)
    paper = _paper(tmp_path)
    _state, first = _run(eng, paper, Replies())
    final = paper.read_text(encoding="utf-8")
    titles = [i["metadata"]["title"] for i in first["literature"]]
    # the stop: the state holds the sources as they were before the check, the claim check's receipt is the one before it
    _receipts.write(tmp_path, "claim_check", status="pass", producer="claim_check", started_at=_receipts.now(),
                    inputs={"paper": b"the draft before the check"}, output={"claims": []})
    _receipts.path(tmp_path, "citation_check").unlink()
    rec = json.loads((eng.fi_dir / cs.FILE).read_text(encoding="utf-8"))
    rec["done"] = False
    (eng.fi_dir / cs.FILE).write_text(json.dumps(rec), encoding="utf-8")

    again = Replies()
    _state, out = _run(eng, paper, again, literature=_literature())
    assert paper.read_text(encoding="utf-8") == final
    assert [i["metadata"]["title"] for i in out["literature"]] == titles, "the sources are in the order the paper numbers them"
    assert again.calls == ["claim_check"], "no reader, no fixer; the claim check only, because its receipt was for another text"
    for check in ("claim_check", "citation_check"):
        _status, record, problem = _receipts.read(tmp_path, check)
        assert not problem and record["input_hashes"]["paper"] == _receipts.sha256(paper.read_bytes()), check
    assert json.loads((eng.fi_dir / cs.FILE).read_text(encoding="utf-8"))["done"] is True
    # and once everything is settled, a further pass asks for nothing at all
    third = Replies()
    _run(eng, paper, third, literature=out["literature"])
    assert third.calls == []
