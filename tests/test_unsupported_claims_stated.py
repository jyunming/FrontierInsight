"""The sentences of the delivered paper that FI's last claim check found supported by neither the run's results nor a
cited source are stated in the paper's Limitations, from FI's own record (core/paper_facts.py ``UNSUPPORTED_CLAIMS``,
``Engine._state_unsupported_claims``). Before, such findings went to QUALITY.md only and the paper said nothing. No model
is called."""

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
    assert "not found supported by the run's results or a cited source" in item.sentence
    assert f'"{FIRST}"' in item.sentence and '"Every tested design settles faster than the baseline"' in item.sentence
    assert "[3]" not in item.sentence
    assert json.loads((root / ".fi" / "unsupported_claims.json").read_text())["claims"][0]["evidence"] == "x"


def test_a_long_list_is_cut_to_a_few_and_says_how_many_more(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    paper_facts.write_unsupported_record(tmp_path, [{"claim": f"Claim number {n} says something."} for n in range(11)])
    (item,) = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.UNSUPPORTED_CLAIMS]
    assert "Claim number 7 says" in item.sentence and "Claim number 8" not in item.sentence
    assert "and 3 more" in item.sentence


def test_a_paper_with_two_unsupported_claims_says_so_in_its_limitations(tmp_path: Path) -> None:
    """Old code: the final claim check's findings went to QUALITY.md only; the delivered paper said nothing."""
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    text = path.read_text(encoding="utf-8")
    assert n == 2
    assert f'"{FIRST}"' in text and "Every tested design settles faster than the baseline" in text.split("## Limitations")[1]
    assert text.index("## Limitations") < text.index("not found supported") < text.index("## References")
    assert text.count("not found supported") == 1
    # Nothing is taken out of the paper: both sentences are still in the results.
    assert text.split("## Limitations")[0].count("fan speed is halved") == 1
    # Stated again, not added again.
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8").count("not found supported") == 1


def test_a_paper_with_no_unsupported_claim_gets_no_such_sentence(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    assert eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()}) == 0  # type: ignore[arg-type]
    assert "not found supported" not in path.read_text(encoding="utf-8")


def test_a_claim_the_paper_no_longer_holds_is_not_stated(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    n = eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding("The pump never failed during any trial of the study")})
    assert n == 0 and "not found supported" not in path.read_text(encoding="utf-8")


def test_a_later_clean_check_removes_the_sentence_an_earlier_pass_placed(tmp_path: Path) -> None:
    """A stale disclosure must not survive: the record is emptied and the sentence goes (it must not turn into 'not
    recorded' either)."""
    eng, path = _paper(tmp_path)
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST, SECOND)})  # type: ignore[arg-type]
    assert "not found supported" in path.read_text(encoding="utf-8")
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()})  # type: ignore[arg-type]
    text = path.read_text(encoding="utf-8")
    assert "not found supported" not in text and "not recorded" not in text and "fi:fact" not in text
    assert json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text()) == {"claims": []}
    assert "Few settings were run." in text


def test_a_claim_check_that_gave_no_answer_leaves_the_record_as_it_was(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding(FIRST)})  # type: ignore[arg-type]
    placed = path.read_text(encoding="utf-8")
    assert eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": {}}) == 0  # type: ignore[arg-type]
    assert eng._state_unsupported_claims(  # type: ignore[arg-type]
        {"paper_md": str(path), "claim_grounding": _grounding(status="unknown")}) == 0
    assert path.read_text(encoding="utf-8") == placed


async def test_the_last_step_before_delivery_states_the_claims_whatever_route_brought_the_paper(tmp_path: Path) -> None:
    eng, path = _paper(tmp_path)

    async def no_citations(state: dict) -> dict:
        return {"citation_next": "done", "claim_grounding": _grounding(FIRST, SECOND)}

    eng._check_citations = no_citations  # type: ignore[method-assign]
    out = await eng._node_citation_check({"paper_md": str(path)})  # type: ignore[arg-type]
    assert out["citation_next"] == "done"
    assert f'"{FIRST}"' in path.read_text(encoding="utf-8")
