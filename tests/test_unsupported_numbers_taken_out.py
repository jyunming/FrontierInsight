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


def _claim(text: str, basis: str, evidence: str = "x", passage: str | None = None, fault: str | None = None) -> dict:
    where = [{"id": "1.1", "section": "S", "passage": passage}] if passage else []
    return {"claim": text, "basis": basis, "fault": fault, "citation_index": None, "quote": "", "evidence": evidence,
            "where": where}


def _grounding(status: str = "ok") -> dict:
    claims = [
        _claim("At a step of 0.0025 s the error was 1.374e-11 to 2.164e-07 across the settings", "unsupported",
               "the range at that step is 1.374e-11 to 3.122e-11; this one was pooled over all steps",
               "At a step of 0.0025 s the error was {{v:err @ dt=all | range}} across the settings.", "scope"),
        _claim("The fit stays above 0.9991 in every case", "unsupported", "the table shows 0.9990875",
               "The fit stays above 0.9991 in every case.", "number"),
        _claim(SOFT, "unsupported", "no source or result shows it", SOFT + ".", "wording"),
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


def test_a_claim_record_written_before_faults_were_named_is_taken_out_as_it_was_then(tmp_path: Path) -> None:
    """A quest resumed on this FI keeps the claim record an older FI wrote, with no ``fault`` on any claim: a sentence it
    found unsupported that states a number (one FI filled in, or one the writer typed that FI cannot trace) is taken out,
    as that FI took it out. Old code: no fault, so both stayed, stated only as unconfirmed."""
    eng, path = _paper(tmp_path)
    g = _grounding()
    for c in g["claims"]:
        c.pop("fault")
    n = eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": g})  # type: ignore[arg-type]
    body = path.read_text(encoding="utf-8").split("## Limitations")[0]
    assert "2.164e-07" not in body and "0.9991" not in body
    # known positive: an unsupported sentence with no number stays and is stated as unconfirmed, as it was then too
    assert n == 1 and f"{SOFT}." in body and GOOD in body
    record = json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text(encoding="utf-8"))
    assert [r["claim"][:20] for r in record["removed"]] == ["At a step of 0.0025 ", "The fit stays above "]


def test_a_claim_of_this_fi_that_its_fault_keeps_is_not_cut_for_its_number(tmp_path: Path) -> None:
    """The older rule is for a record with no fault only: a claim of this FI whose fault keeps its sentence (``wording``) stays,
    whatever number it states."""
    eng, path = _paper(tmp_path)
    g = _grounding()
    g["claims"][1]["fault"] = "wording"  # the typed-number sentence, judged a wording fault by this FI
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": g})  # type: ignore[arg-type]
    body = path.read_text(encoding="utf-8").split("## Limitations")[0]
    assert TYPED in body and "2.164e-07" not in body


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


def test_what_was_taken_out_of_an_earlier_draft_is_not_stated_about_a_new_one(tmp_path: Path) -> None:
    """A person's refine writes a new draft after the delivery step: the removed list belongs to the draft it was made on."""
    eng, path = _paper(tmp_path)
    _run(eng, path)
    record = eng.quest_root / ".fi" / "unsupported_claims.json"
    assert json.loads(record.read_text(encoding="utf-8")).get("removed")
    paper_facts.forget_removed_unsupported(eng.quest_root)
    kept = json.loads(record.read_text(encoding="utf-8"))
    assert "removed" not in kept and kept["claims"]
    paper_facts.forget_removed_unsupported(tmp_path / "no_such_quest")  # nothing there: no error


# --- exact identification, markdown, one consistent change, and the record's life --------------------------------------

def _record_claims_run(paper: str, claims: list[dict]) -> tuple[str, list[dict]]:
    from core import record_claims

    return record_claims.take_out_unsupported_numbers(
        paper, claims, same=lambda a, b: True,  # a loose test: likeness decides nothing
        holds_number=lambda s: True, plain=lambda s: " ".join(s.split()), keyed=lambda s: s)


def test_same_wording_with_other_numbers_only_the_unsupported_sentence_goes() -> None:
    paper = "# T\n\n## Results\nThe error was 7 units across all settings. The error was 3 units across all settings.\n"
    claims = [_claim("The error was 7 units across all settings", "unsupported",
                     passage="The error was 7 units across all settings."),
              _claim("The error was 3 units across all settings", "experiment",
                     passage="The error was 3 units across all settings.")]
    text, removed = _record_claims_run(paper, claims)
    assert "7 units" not in text and "The error was 3 units across all settings." in text and len(removed) == 1


def test_minus_seven_is_not_taken_for_seven() -> None:
    paper = "# T\n\n## Results\nThe shift was -7 units. The shift was 7 units.\n"
    text, removed = _record_claims_run(paper, [_claim("The shift was 7 units", "unsupported",
                                                      passage="The shift was 7 units.")])
    assert text == "# T\n\n## Results\nThe shift was -7 units.\n" and len(removed) == 1


def test_a_claim_with_no_passage_cuts_nothing() -> None:
    paper = "# T\n\n## Results\nThe error was 7 units across all settings.\n"
    text, removed = _record_claims_run(paper, [_claim("The error was 7 units across all settings", "unsupported")])
    assert text == paper and removed == []
    text, removed = _record_claims_run(paper, [_claim("The error was 7 units across all settings", "unsupported",
                                                      passage="A different sentence entirely.")])
    assert text == paper and removed == []


def test_the_claims_own_passage_identifies_the_sentence() -> None:
    paper = "# T\n\n## Results\nThe error was 7 units across all settings. The error was 3 units across all settings.\n"
    claim = _claim("The error was 7 units across all settings", "unsupported",
                   passage="The error was 3 units across all settings.")
    text, removed = _record_claims_run(paper, [claim])
    assert "3 units" not in text and "7 units across all settings." in text and len(removed) == 1


def test_fenced_code_is_never_touched_and_a_list_item_leaves_no_bare_marker() -> None:
    sentence = "The error was 7 units across all settings."
    claims = [_claim(sentence[:-1], "unsupported", passage=sentence)]
    fenced = f"# T\n\n```\n{sentence}\nmore\n```\n\nText.\n"
    text, removed = _record_claims_run(fenced, claims)
    assert text == fenced and removed == []
    listed = f"# T\n\n- Keep this one.\n- {sentence}\n- Also kept.\n"
    text, removed = _record_claims_run(listed, claims)
    assert text == "# T\n\n- Keep this one.\n- Also kept.\n" and len(removed) == 1
    last = f"# T\n\n- Keep.\n- {sentence}"
    text, _ = _record_claims_run(last, claims)
    assert "\n-" not in text.split("Keep.")[1]
    both = f"# T\n\n- Lead sentence. {sentence}\n"
    text, _ = _record_claims_run(both, claims)
    assert text == "# T\n\n- Lead sentence.\n"


def test_a_failed_write_puts_every_file_back_and_reports_nothing_removed(tmp_path: Path, monkeypatch) -> None:
    from core import receipts

    eng, path = _paper(tmp_path)
    eng._write_claims_ledger({**_grounding(), "grounded": 1})
    ledger_before = (eng.quest_root / "paper" / "claims.json").read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(receipts, "carry_over", boom)
    n = _run(eng, path)
    assert n == 3  # every unsupported claim is stated as unconfirmed, none as taken out
    text = path.read_text(encoding="utf-8")
    assert "0.9991" in text.split("## Limitations")[0] and "were taken out" not in text and "was taken out" not in text
    assert (eng.quest_root / "paper" / "claims.json").read_bytes() == ledger_before
    record = json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text(encoding="utf-8"))
    assert "removed" not in record


async def test_a_writer_that_fails_leaves_the_earlier_drafts_removed_list(tmp_path: Path) -> None:
    import pytest

    eng, path = _paper(tmp_path)
    _run(eng, path)
    record = eng.quest_root / ".fi" / "unsupported_claims.json"
    kept = record.read_bytes()

    async def nothing(*a, **k):
        return None

    async def fails(*a, **k):
        raise RuntimeError("the writer failed")

    eng._check_code_project = nothing  # type: ignore[method-assign]
    eng._patch_flagged_passages = nothing  # type: ignore[method-assign]
    eng._write_sections = nothing  # type: ignore[method-assign]
    eng._write_whole_paper = fails  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await eng._node_write({"paper_md": str(path)})  # type: ignore[arg-type]
    assert record.read_bytes() == kept


def test_a_fence_closes_only_on_a_fence_of_its_own_kind_and_length() -> None:
    sentence = "The error was 7 units across all settings."
    claims = [_claim(sentence[:-1], "unsupported", passage=sentence)]
    # a three-backtick line inside a four-backtick fence does not close it
    paper = f"# T\n\n````\n```\n{sentence}\n```\n````\n\nText.\n"
    text, removed = _record_claims_run(paper, claims)
    assert text == paper and removed == []
    # a tilde line does not close a backtick fence, and a longer fence does
    paper = f"# T\n\n```\n~~~\n{sentence}\n`````\n\n{sentence}\n"
    text, removed = _record_claims_run(paper, claims)
    assert text.count(sentence) == 1 and len(removed) == 1


def test_a_failure_at_the_claim_list_or_the_limitations_puts_every_file_back(tmp_path: Path, monkeypatch) -> None:
    for what in ("claims_md", "restate"):
        eng, path = _paper(tmp_path / what)
        eng._write_claims_ledger({**_grounding(), "grounded": 1})
        files = [path, eng.quest_root / "paper" / "claims.json", eng.quest_root / "paper" / "CLAIMS.md"]
        before = [f.read_bytes() for f in files]
        if what == "claims_md":
            real = Path.write_text

            def write_text(self, data, *a, **k):
                if self.name == "CLAIMS.md":
                    raise OSError("disk full")
                return real(self, data, *a, **k)

            monkeypatch.setattr(Path, "write_text", write_text)
        else:
            def restate(state, why, strict=False):
                if strict:
                    raise OSError("disk full")
                return False

            monkeypatch.setattr(eng, "_restate_limitations", restate)
        eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()})  # type: ignore[arg-type]
        monkeypatch.undo()
        assert [f.read_bytes() for f in files[1:]] == before[1:], what
        kept = path.read_text(encoding="utf-8")  # the paper keeps every sentence and says none was taken out
        assert "0.9991" in kept.split("## Limitations")[0] and "taken out of this paper" not in kept, what
        record = json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text(encoding="utf-8"))
        assert "removed" not in record, what


async def test_a_render_that_fails_leaves_the_earlier_drafts_removed_list(tmp_path: Path) -> None:
    import pytest

    eng, path = _paper(tmp_path)
    _run(eng, path)
    record = eng.quest_root / ".fi" / "unsupported_claims.json"
    kept = record.read_bytes()

    async def nothing(*a, **k):
        return None

    async def writes(*a, **k):
        return "# New draft\n\n## Abstract\nA new sentence.\n"

    def render_fails(markdown, state):
        raise RuntimeError("render failed")

    eng._check_code_project = nothing  # type: ignore[method-assign]
    eng._patch_flagged_passages = nothing  # type: ignore[method-assign]
    eng._write_sections = nothing  # type: ignore[method-assign]
    eng._write_whole_paper = writes  # type: ignore[method-assign]
    eng._render_paper_facts = render_fails  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await eng._node_write({"paper_md": str(path)})  # type: ignore[arg-type]
    assert record.read_bytes() == kept


def test_a_claim_kept_only_by_its_stored_passage_is_listed_but_its_sentence_is_not_cut(tmp_path: Path) -> None:
    """The passage match widens the Limitations listing only: a reworded claim whose passage holds a number is stated as
    unconfirmed, and its sentence stays in the paper (the take-out gets the claims it got before)."""
    eng, path = _paper(tmp_path)
    g = _grounding()
    g["claims"] = [_claim("A fit quality of at least 0.9991 holds throughout", "unsupported", "x", TYPED)]
    n = eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": g})  # type: ignore[arg-type]
    text = path.read_text(encoding="utf-8")
    assert n == 1 and TYPED in text.split("## Limitations")[0]
    assert "A fit quality of at least 0.9991 holds throughout" in text.split("## Limitations")[1]
