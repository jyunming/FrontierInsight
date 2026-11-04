"""A sentence of the writer's that FI's claim check judged unsupported and that states a number is never delivered: it is
taken out of the paper, leaves the claim list, and the Limitations and QUALITY.md say so (``Engine._state_unsupported_claims``,
``core/record_claims.py::take_out_unsupported_numbers``). A sentence the check judged supported stays, and so does an
unsupported one with no number (it is stated as unconfirmed, as before). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts

MARKED = ("At a step of 0.0025 s the error was <!-- fi:v {{v:err @ dt=all | range}} -->1.374e-11 to 2.164e-07"
          "<!-- /fi:v --> across the settings.")
TYPED = "The fit stays above 0.9991 in every case."
SOFT = "The period grows without bound near the top of the range"
GOOD = "The settle time was 4.2 s."
PAPER = (f"# T\n\n## Abstract\nThe study measures a period. {MARKED} {TYPED} Done.\n\n## Results\n{SOFT}. {GOOD}\n\n"
         "## Limitations\nFew settings were run.\n\n## References\n1. A.\n")


def _claim(text: str, basis: str, evidence: str = "x") -> dict:
    return {"claim": text, "basis": basis, "citation_index": None, "quote": "", "evidence": evidence,
            "where": []}


def _grounding(status: str = "ok") -> dict:
    claims = [
        _claim("At a step of 0.0025 s the error was 1.374e-11 to 2.164e-07 across the settings", "unsupported",
               "the range at that step is 1.374e-11 to 3.122e-11; this one was pooled over all steps"),
        _claim("The fit stays above 0.9991 in every case", "unsupported", "the table shows 0.9990875"),
        _claim(SOFT, "unsupported", "no source or result shows it"),
        _claim("The settle time was 4.2 s", "experiment"),
    ]
    return {"claims": claims, "total": len(claims), "status": status,
            "unsupported": [c["claim"] for c in claims if c["basis"] == "unsupported"]}


def _paper(tmp_path: Path):
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    path = eng.quest_root / "paper.md"
    path.write_text(PAPER, encoding="utf-8")
    return eng, path


def _run(eng, path):
    return eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()})  # type: ignore[arg-type]


def test_a_sentence_with_an_unsupported_number_does_not_reach_the_final_paper(tmp_path: Path) -> None:
    """Old code: both sentences stayed in the abstract and were only listed as unconfirmed in the Limitations."""
    eng, path = _paper(tmp_path)
    _run(eng, path)
    text = path.read_text(encoding="utf-8")
    assert "2.164e-07" not in text.split("## Limitations")[0] and "0.9991" not in text.split("## Limitations")[0]
    assert "The study measures a period." in text and "Done." in text  # the rest of the abstract is as it was
    assert "2 sentences of the draft" in text.split("## Limitations")[1] and "were taken out of this paper" in text
    # the Limitations do not quote the number that was not supported
    assert "2.164e-07" not in text and "0.9991" not in text


def test_the_taken_out_sentences_are_in_the_record_and_leave_the_claim_list(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    ledger = _grounding()
    eng._write_claims_ledger({**ledger, "grounded": 1})
    _run(eng, path)
    record = json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text(encoding="utf-8"))
    assert [r["claim"][:20] for r in record["removed"]] == ["At a step of 0.0025 ", "The fit stays above "]
    assert "pooled over all steps" in record["removed"][0]["evidence"]
    kept = json.loads((eng.quest_root / "paper" / "claims.json").read_text(encoding="utf-8"))["claims"]
    assert not [c for c in kept if "2.164e-07" in c["claim"] or "0.9991" in c["claim"]]


def test_a_supported_sentence_stays(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    _run(eng, path)
    assert GOOD in path.read_text(encoding="utf-8")


def test_an_unsupported_sentence_without_a_number_stays_and_is_stated_as_unconfirmed(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = _run(eng, path)
    text = path.read_text(encoding="utf-8")
    assert n == 1 and f"{SOFT}." in text.split("## Limitations")[0]
    assert "not found supported" in text.split("## Limitations")[1] and f'"{SOFT}"' in text


def test_a_second_pass_keeps_what_the_first_took_out_and_says_it_once(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    _run(eng, path)
    first = path.read_text(encoding="utf-8")
    _run(eng, path)  # the paper no longer holds the two sentences: nothing more to take out
    again = path.read_text(encoding="utf-8")
    assert again.count("were taken out of this paper") == 1 and again == first


def test_a_claim_check_that_gave_no_answer_takes_nothing_out(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    assert eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding("unknown")}) == 0  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == PAPER


def test_the_limitations_sentence_and_the_quality_report_name_the_taken_out_claims(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".fi").mkdir()
    assert paper_facts.write_unsupported_record(root, [], removed=[{"claim": "A number sentence", "evidence": "pooled"}])
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.UNSUPPORTED_CLAIMS]
    assert item.sentence.startswith("One sentence of the draft") and "was taken out of this paper" in item.sentence
    assert "A number sentence" not in item.sentence
