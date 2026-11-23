"""The sentences of the delivered paper that FI's last claim check found supported by neither the run's results nor a
cited source are kept in FI's own record, which QUALITY.md lists (core/paper_facts.py ``UNSUPPORTED_CLAIMS``,
``Engine._state_unsupported_claims``); the paper carries no mark beside them, and while one stands it ends its body with one
plain sentence that points to the quality report (core/quality_pointer.py). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts

FIRST = "The cooling rate doubles whenever the fan speed is halved"
SECOND = "Every tested design settles faster than the baseline [3]"
PAPER = (f"# T\n\n## Results\n{FIRST}. The settle time was 4.2 s. {SECOND}.\n\n"
         "## Limitations\nFew settings were run.\n\n## References\n1. A.\n")


def _grounding(*unsupported: str, status: str = "ok") -> dict:
    claims = [{"claim": c, "basis": "unsupported", "citation_index": None, "quote": "", "evidence": "no result shows it"}
              for c in unsupported]
    claims.append({"claim": "The settle time was 4.2 s", "basis": "experiment", "citation_index": None, "quote": "",
                   "evidence": "result_json"})
    return {"claims": claims, "total": len(claims), "unsupported": list(unsupported), "status": status}


POINTER = "Statements that the study's results and the cited sources do not support are listed in the accompanying quality report."


def _pointers(text: str) -> int:
    return " ".join(paper_facts.without_marks(text).split()).count("listed in the accompanying quality report")


def _listed(eng) -> str:
    """What QUALITY.md says of the claims, from FI's record."""
    path = eng.quest_root / paper_facts.UNSUPPORTED_RECORD
    return paper_facts.unsupported_claims_text(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else ""


def _paper(tmp_path: Path, text: str = PAPER):
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    path = eng.quest_root / "paper.md"
    path.write_text(text, encoding="utf-8")
    return eng, path


def test_the_record_gives_one_sentence_that_quotes_each_claim_and_drops_the_reference_numbers(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".fi").mkdir()
    assert not [i for i in paper_facts.required_items(root) if i.kind == paper_facts.UNSUPPORTED_CLAIMS]
    assert paper_facts.write_unsupported_record(
        root, [{"claim": FIRST, "evidence": "x"}, {"claim": SECOND, "evidence": "y"}])
    (item,) = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.UNSUPPORTED_CLAIMS]
    assert item.sentence.startswith("2 sentences of this paper were not found supported by the run's results or a cited source")
    assert "QUALITY.md" not in item.sentence and "unconfirmed" not in item.sentence, "the sentence is QUALITY.md's own"
    assert f'"{FIRST}"' in item.sentence and '"Every tested design settles faster than the baseline"' in item.sentence
    assert "[3]" not in item.sentence
    assert json.loads((root / ".fi" / "unsupported_claims.json").read_text())["claims"][0]["evidence"] == "x"


def test_a_long_list_is_cut_to_a_few_and_says_how_many_more(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    paper_facts.write_unsupported_record(tmp_path, [{"claim": f"Claim number {n} says something."} for n in range(11)])
    (item,) = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.UNSUPPORTED_CLAIMS]
    assert "Claim number 7 says" in item.sentence and "Claim number 8" not in item.sentence
    assert "and 3 more" in item.sentence


def test_two_unsupported_claims_are_listed_and_the_paper_ends_its_body_with_one_sentence_that_points_there(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    assert n == 2
    text = paper_facts.without_marks(path.read_text(encoding="utf-8"))
    assert " ".join(text.replace(POINTER, "").split()) == " ".join(PAPER.split()), "only the one sentence is added; no mark"
    assert _pointers(text) == 1 and text.index(POINTER) < text.index("## References"), "once, before the reference list"
    assert "unconfirmed" not in text and "quality report" in text and "QUALITY.md" not in text
    said = _listed(eng)
    assert f'"{FIRST}"' in said and "Every tested design settles faster than the baseline" in said
    # Listed again, not added again: the sentence is there once after every round.
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    assert _listed(eng) == said and _pointers(path.read_text(encoding="utf-8")) == 1


def test_a_paper_with_no_unsupported_claim_gets_no_such_sentence(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    assert eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()}) == 0  # type: ignore[arg-type]
    assert "not found supported" not in _listed(eng) and path.read_text(encoding="utf-8") == PAPER


def test_a_claim_the_paper_no_longer_holds_is_not_stated(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding("The pump never failed during any trial of the study")})
    assert n == 0 and "not found supported" not in _listed(eng)


def test_a_later_clean_check_empties_the_record_and_takes_the_sentence_away(tmp_path: Path) -> None:
    """A stale listing must not survive: the record is emptied, and the paper ends as the writer wrote it."""
    eng, path = _paper(tmp_path)
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    assert "not found supported" in _listed(eng) and _pointers(path.read_text(encoding="utf-8")) == 1
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()})  # type: ignore[arg-type]
    assert "not found supported" not in _listed(eng)
    assert json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text()) == {"claims": []}
    assert _pointers(path.read_text(encoding="utf-8")) == 0
    assert " ".join(paper_facts.without_marks(path.read_text(encoding="utf-8")).split()) == " ".join(PAPER.split())


def test_a_claim_check_that_gave_no_answer_leaves_the_record_as_it_was(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST)})  # type: ignore[arg-type]
    listed = _listed(eng)
    assert eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": {}}) == 0  # type: ignore[arg-type]
    assert eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding(status="unknown")}) == 0
    assert _listed(eng) == listed and f'"{FIRST}"' in listed


async def test_the_last_step_before_delivery_lists_the_claims_whatever_route_brought_the_paper(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)

    async def no_citations(state: dict) -> dict:
        return {"citation_next": "done", "claim_grounding": _grounding(FIRST, SECOND)}

    eng._check_citations = no_citations  # type: ignore[method-assign]
    out = await eng._node_citation_check({"paper_md": str(path)})  # type: ignore[arg-type]
    assert out["citation_next"] == "done"
    assert f'"{FIRST}"' in _listed(eng) and f'"{FIRST}"' not in path.read_text(encoding="utf-8")


# A claim kept by the exact passage the check stored for it, not by its wording (backlog -434, G32).
_PASSAGE = "The cooling rate doubles whenever the fan speed is halved"
_REWORDED = "Halving fan speed leads to a twofold gain in how fast heat leaves"


def _grounding_with_passage(claim: str, passage: str) -> dict:
    g = _grounding(claim)
    g["claims"][0]["where"] = [{"passage": passage, "section": "Results"}]
    return g


def test_a_claim_reworded_after_the_check_is_still_listed_by_its_stored_passage(tmp_path: Path) -> None:
    """Old code: the claim's wording was matched with the paper, so a reworded claim was dropped from the list."""
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding_with_passage(_REWORDED, _PASSAGE)})
    assert n == 1 and f'"{_REWORDED}"' in _listed(eng) and _pointers(path.read_text(encoding="utf-8")) == 1


def test_a_claim_whose_stored_passage_is_gone_from_the_paper_is_not_listed(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path),
         "claim_grounding": _grounding_with_passage(_REWORDED, "The pump never failed during any trial of the study")})
    assert n == 0 and "not found supported" not in _listed(eng)


def test_a_claim_quoted_only_in_an_older_fis_limitations_block_is_not_found_by_its_passage(tmp_path: Path) -> None:
    old = PAPER.replace("Few settings were run.\n", f'Few settings were run.\n\n<!-- fi:fact-limitations -->\n- "{FIRST}"\n'
                        "<!-- /fi:fact-limitations -->\n").replace(f"{FIRST}. ", "", 1)
    eng, path = _paper(tmp_path, old)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding_with_passage(_REWORDED, FIRST)})
    assert n == 0 and "not found supported" not in _listed(eng)


def test_an_unchanged_claim_with_a_passage_is_listed_as_before(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding_with_passage(FIRST, FIRST)})
    assert n == 1 and f'"{FIRST}"' in _listed(eng) and FIRST in path.read_text(encoding="utf-8")
    assert _pointers(path.read_text(encoding="utf-8")) == 1


def test_an_earlier_unsupported_claim_is_carried_by_its_stored_passage_when_its_wording_is_not_in_the_paper() -> None:
    """A claim the check wrote as a paraphrase, with the paper's own sentence stored as its passage, is carried over to the
    next check by that passage: a later check that leaves it out cannot clear it while the passage is still there."""
    from core.engine import _unsupported_still_in

    earlier = [{"claim": "The method applies to every design", "basis": "unsupported", "evidence": "no result shows it",
                "where": [{"passage": FIRST}]}]
    kept = _unsupported_still_in(PAPER, earlier, [])
    assert [c["claim"] for c in kept] == ["The method applies to every design"]
    assert _unsupported_still_in(PAPER.replace(FIRST, "The fan was run at three speeds"), earlier, []) == []
