"""The claim check names what is wrong with each claim (``fault``), and only the fault decides what becomes of its
sentence (``Engine._node_claim_check``, ``Engine._state_unsupported_claims``):

- ``null``: supported; the sentence stays.
- ``number`` (a typed number not in the record), ``scope`` (a true value under a wrong setting, label, interval or "every
  setting"): the sentence is taken out and the Limitations say so.
- ``wording``: the words say more than the record; the sentence stays and is stated in the Limitations.
- ``no_source`` (FI's own title-only and retracted rules) and ``unread`` (a verdict FI could not read after one ask): out.

Before, the take-out guessed from the sentence (any number in it, typed or filled by FI), so a true filled number was cut
with a wording claim, and a wrong-setting claim with no number in it stayed. No model is called."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from core.engine import (
    _CLAIM_FAULTS_TAKEN_OUT,
    _apply_title_only_rule,
    _claim_choice_problem,
    _claim_faults_from_sources,
    _unsupported_still_in,
)

# --- the take-out reads only the fault ---------------------------------------------------------------------------------

FILLED = "<!-- fi:v {{v:wait @ load=3}} -->0.19900<!-- /fi:v -->"
WORDING = f"At a load of 3 the mean wait is {FILLED} and close to 0.2 at every load."
SETTING = f"At a load of 5 the mean wait is {FILLED} minutes."
INTERVAL = f"The mean wait at a load of 3 is {FILLED} (95% interval 0.18 to 0.22)."
EVERY = "The mean wait stays under a quarter of a minute at every load of the sweep."
NO_SOURCE = "Earlier work found that a toll queue settles within a few arrivals of the start of the sweep."
UNREAD = "The queue is shorter on the second lane than on the first for every arrival pattern."
OLD = "The booth model reproduces the arrival pattern of a real toll plaza in the sweep."
GOOD = f"The mean wait at a load of 3 is {FILLED} minutes."
BODY = [WORDING, SETTING, INTERVAL, EVERY, NO_SOURCE, UNREAD, OLD, GOOD]
PAPER = ("# T\n\n## Results\n\n" + " ".join(BODY) + "\n\n## Limitations\nFew loads were run.\n\n## References\n1. A.\n")


def _keyed(sentence: str) -> str:
    return sentence.replace(FILLED, "{{v:wait @ load=3}}")


def _plain(sentence: str) -> str:
    return sentence.replace(FILLED, "0.19900").rstrip(".")


def _claim(sentence: str, basis: str, fault: Any = "absent") -> dict[str, Any]:
    c: dict[str, Any] = {"claim": _plain(sentence), "basis": basis, "citation_index": None, "quote": "",
                         "evidence": "e", "where": [{"id": "3.1", "section": "Results", "passage": _keyed(sentence)}]}
    if fault != "absent":
        c["fault"] = fault
    return c


def _grounding() -> dict[str, Any]:
    claims = [_claim(WORDING, "unsupported", "wording"), _claim(SETTING, "unsupported", "scope"),
              _claim(INTERVAL, "unsupported", "scope"), _claim(EVERY, "unsupported", "scope"),
              _claim(NO_SOURCE, "unsupported", "no_source"), _claim(UNREAD, "unsupported", "unread"),
              _claim(OLD, "unsupported"), _claim(GOOD, "experiment", None)]
    return {"claims": claims, "total": len(claims), "status": "ok",
            "unsupported": [c["claim"] for c in claims if c["basis"] == "unsupported"]}


def _deliver(tmp_path: Path) -> tuple[str, dict[str, Any]]:
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    path = eng.quest_root / "paper.md"
    path.write_text(PAPER, encoding="utf-8")
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": _grounding()})  # type: ignore[arg-type]
    record = json.loads((eng.quest_root / ".fi" / "unsupported_claims.json").read_text(encoding="utf-8"))
    # FI fills its values again when it writes the Limitations (here from no record): the body is compared by its keys.
    body = path.read_text(encoding="utf-8").split("## Limitations")[0]
    return re.sub(r"<!-- fi:v (\{\{.*?\}\}) -->.*?<!-- /fi:v -->", r"\1", body), record


def test_a_wording_claim_keeps_its_sentence_and_its_filled_number(tmp_path: Path) -> None:
    """Old code: the sentence held a value FI filled from its record, so it was cut with the true 0.19900."""
    body, record = _deliver(tmp_path)
    assert _keyed(WORDING) in body
    assert _plain(WORDING) in [r["claim"] for r in record["claims"]]  # stated in the Limitations, not cut


def test_a_true_value_under_a_wrong_setting_interval_or_every_setting_is_taken_out(tmp_path: Path) -> None:
    """Old code: the every-setting sentence holds no number, so it stayed."""
    body, record = _deliver(tmp_path)
    for sentence in (SETTING, INTERVAL, EVERY):
        assert _keyed(sentence) not in body, sentence
    assert {_plain(s) for s in (SETTING, INTERVAL, EVERY)} <= {r["claim"] for r in record["removed"]}


def test_a_claim_with_no_source_or_an_unread_verdict_is_taken_out(tmp_path: Path) -> None:
    body, record = _deliver(tmp_path)
    assert NO_SOURCE not in body and UNREAD not in body
    assert {_plain(NO_SOURCE), _plain(UNREAD)} <= {r["claim"] for r in record["removed"]}


def test_a_supported_claim_and_an_old_record_without_a_fault_stay(tmp_path: Path) -> None:
    """Known positive: a supported claim with a filled number stays; an unsupported claim recorded before the check named
    faults is never cut on a guess, only stated."""
    body, record = _deliver(tmp_path)
    assert _keyed(GOOD) in body and OLD in body
    assert _plain(OLD) in [r["claim"] for r in record["claims"]]
    assert _plain(GOOD) not in [r["claim"] for r in record["claims"]]


def test_the_faults_that_take_a_sentence_out() -> None:
    assert _CLAIM_FAULTS_TAKEN_OUT == {"number", "scope", "no_source", "unread"}


# --- reading the check's answer ----------------------------------------------------------------------------------------

def test_a_basis_and_a_fault_are_closed_choices() -> None:
    assert _claim_choice_problem({"basis": "experiment", "fault": None}) == ""
    assert _claim_choice_problem({"basis": "citation"}) == ""  # a supported claim with no fault key: null
    assert _claim_choice_problem({"basis": " Unsupported ", "fault": "SCOPE"}) == ""
    assert "basis 'partly'" in _claim_choice_problem({"basis": "partly", "fault": None})
    assert "gives no fault" in _claim_choice_problem({"basis": "unsupported"})
    assert "fault 'setting'" in _claim_choice_problem({"basis": "unsupported", "fault": "setting"})
    assert "fault 'null'" in _claim_choice_problem({"basis": "unsupported", "fault": "null"})
    assert "fault 'wording' is given" in _claim_choice_problem({"basis": "experiment", "fault": "wording"})
    assert _claim_choice_problem({"basis": "unsupported", "fault": "no_source"})  # FI's to set, not the model's


def _engine(tmp_path: Path, replies: list[str]):
    from tests.test_claim_check import _engine as claim_engine

    eng = claim_engine(tmp_path)
    calls: list[str] = []

    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        calls.append(prompt)
        return replies[min(len(calls), len(replies)) - 1]

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    return eng, calls


QUEUE_PAPER = ("# Toll booth waits\n\n## Results\n\nThe mean wait at a load of 3 is 0.199 minutes. "
               "The wait is lowest at the first load.\n")


def _state(tmp_path: Path) -> dict[str, Any]:
    p = tmp_path / "paper" / "paper.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(QUEUE_PAPER, encoding="utf-8")
    return {"topic": "toll booth waits", "paper_md": str(p), "result_json": {"wait": 0.199}}


def _reply(*claims: dict[str, Any]) -> str:
    return json.dumps({"claims": list(claims), "summary": "s"})


GOOD_CLAIM = {"claim": "The mean wait at a load of 3 is 0.199 minutes", "basis": "experiment", "fault": None,
              "evidence": "wait=0.199"}
LOW = "The wait is lowest at the first load"


def test_the_check_records_each_claims_fault_and_keys(tmp_path: Path) -> None:
    eng, calls = _engine(tmp_path, [_reply(GOOD_CLAIM, {"claim": LOW, "basis": "unsupported", "fault": "scope",
                                                        "keys": ["{{v:wait @ load=1}}"], "evidence": "lowest at 2"})])
    g = asyncio.run(eng._node_claim_check(_state(tmp_path)))["claim_grounding"]  # type: ignore[arg-type]
    assert len(calls) == 1
    assert [(c["basis"], c["fault"]) for c in g["claims"]] == [("experiment", None), ("unsupported", "scope")]
    assert g["claims"][1]["keys"] == ["v:wait @ load=1"]
    ledger = json.loads((tmp_path / "paper" / "claims.json").read_text(encoding="utf-8"))
    assert ledger["claims"][1]["fault"] == "scope"  # nothing leaves the receipt's record
    assert "[unsupported: scope]" in (tmp_path / "paper" / "CLAIMS.md").read_text(encoding="utf-8")


def test_an_unreadable_choice_is_asked_once_for_that_claim_only(tmp_path: Path) -> None:
    """Old code: an unknown basis silently became unsupported, with no ask and no fault."""
    first = _reply(GOOD_CLAIM, {"claim": LOW, "basis": "partly", "evidence": "x"})
    second = _reply({"claim": LOW, "basis": "unsupported", "fault": "wording", "evidence": "only one load is lower"})
    eng, calls = _engine(tmp_path, [first, second])
    g = asyncio.run(eng._node_claim_check(_state(tmp_path)))["claim_grounding"]  # type: ignore[arg-type]
    assert len(calls) == 2
    assert "basis 'partly'" in calls[1] and "only these 1 claim(s)" in calls[1]
    assert "0.199 minutes'" not in calls[1].split("could not be used:")[1].split("Answer again")[0]  # only that claim
    assert [(c["basis"], c["fault"]) for c in g["claims"]] == [("experiment", None), ("unsupported", "wording")]


def test_a_choice_still_unreadable_after_the_ask_is_unread_and_fails_the_receipt(tmp_path: Path) -> None:
    bad = _reply(GOOD_CLAIM, {"claim": LOW, "basis": "unsupported", "fault": "maybe", "evidence": "x"})
    eng, calls = _engine(tmp_path, [bad, bad])
    out = asyncio.run(eng._node_claim_check(_state(tmp_path)))  # type: ignore[arg-type]
    g = out["claim_grounding"]
    assert len(calls) == 2 and g["status"] == "ok" and not out["claim_check_failed"]
    unread = g["claims"][1]
    assert unread["basis"] == "unsupported" and unread["fault"] == "unread" and g["unsupported"] == [LOW]
    assert "could not read the check's verdict" in unread["evidence"]
    receipt = json.loads((tmp_path / "needs" / "receipts" / "claim_check.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "fail"


def test_a_failed_ask_leaves_the_claim_unread_not_the_check_failed(tmp_path: Path) -> None:
    eng, calls = _engine(tmp_path, [_reply(GOOD_CLAIM, {"claim": LOW, "basis": "unsupported", "evidence": "x"})])
    answer = eng._chat

    async def failing_second(prompt: str, *, node: str = "") -> str:
        if calls:
            calls.append(prompt)
            raise TimeoutError("the provider did not answer")
        return await answer(prompt, node=node)

    eng._chat = failing_second  # type: ignore[assignment,method-assign]
    out = asyncio.run(eng._node_claim_check(_state(tmp_path)))  # type: ignore[arg-type]
    assert len(calls) == 2 and not out["claim_check_failed"]
    assert [c["fault"] for c in out["claim_grounding"]["claims"]] == [None, "unread"]


def test_a_quote_fi_cannot_find_in_the_source_is_kept_as_wording(tmp_path: Path) -> None:
    """FI's own quote look-up: an unconfirmed citation is stated in the Limitations, never cut on that alone."""
    state = _state(tmp_path)
    state["literature"] = [{"content": "Arrivals at the booth were counted every minute for a year of operation.",
                            "metadata": {"title": "Booth arrivals", "doi": "10.1/q", "source": "crossref"}}]
    claim = {"claim": LOW, "basis": "citation", "fault": None, "citation_index": 1,
             "quote": "the wait is lowest at the first load of the sweep", "evidence": "x"}
    eng, _calls = _engine(tmp_path, [_reply(GOOD_CLAIM, claim)])
    g = asyncio.run(eng._node_claim_check(state))["claim_grounding"]  # type: ignore[arg-type]
    assert (g["claims"][1]["basis"], g["claims"][1]["fault"]) == ("unsupported", "wording")


# --- FI's own source rules set the fault --------------------------------------------------------------------------------

TITLE = "Queueing at Toll Plazas"
SOURCES = {"1": ({"title": TITLE, "authors": ["A. Writer"], "work_type": "book", "year": 2001}, TITLE),
           "2": ({"title": "Booth study", "retraction": "retracted"}, "Arrivals at the booth were counted.")}
SPECIFIC = "Toll plaza queues settle within four arrivals of opening once the second lane is staffed [1]."


def test_a_claim_resting_on_a_title_only_source_is_no_source() -> None:
    cited = {"claim": SPECIFIC, "basis": "citation", "citation_index": 1, "quote": TITLE, "evidence": "",
             "fault": None}
    out, changed = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [cited])
    assert changed == 1 and (out[0]["basis"], out[0]["fault"]) == ("unsupported", "no_source")
    added, _n = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [])
    assert [(c["basis"], c["fault"]) for c in added] == [("unsupported", "no_source")]


def test_a_retracted_source_is_no_source_and_a_typed_untraced_number_is_number() -> None:
    claims = [{"claim": "a", "basis": "unsupported", "citation_index": 2, "evidence": "retracted"},
              {"claim": "b", "basis": "unsupported", "fault": "wording", "where": [{"passage": "It took 7.25 min."}]},
              {"claim": "c", "basis": "unsupported", "fault": "wording", "where": [{"passage": "It is {{v:wait}}."}]},
              {"claim": "d", "basis": "unsupported", "fault": "scope", "citation_index": 2},
              {"claim": "e", "basis": "experiment", "fault": None, "where": [{"passage": "It took 7.25 min."}]}]
    out = _claim_faults_from_sources(claims, SOURCES, lambda passage: "7.25" in passage)
    assert [c["fault"] for c in out] == ["no_source", "number", "wording", "scope", None]
    assert "fault" not in claims[0], "the caller's claims are not changed in place"


def test_an_earlier_fault_is_carried_with_its_unchanged_sentence() -> None:
    earlier = [{"claim": "The wait halves at the third load", "basis": "unsupported", "fault": "scope",
                "evidence": "it halves at the fourth"}]
    (carried,) = _unsupported_still_in("# T\n\nThe wait halves at the third load.\n", earlier, [])
    assert carried["fault"] == "scope"


# --- review pass 1 ------------------------------------------------------------------------------------------------------

def test_a_sentence_fi_adds_for_a_title_only_source_is_taken_out(tmp_path: Path) -> None:
    """Review finding: a claim FI's rule added had no stored passage, so its no_source fault cut nothing."""
    from tests.test_plan_optimisation import _engine as plan_engine

    added, _n = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [])
    (claim,) = _claim_faults_from_sources(added, SOURCES, lambda _p: False)
    assert claim["fault"] == "no_source" and claim["where"][0]["passage"] == SPECIFIC
    eng = plan_engine(tmp_path, [])
    (eng.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "paper").mkdir(parents=True, exist_ok=True)
    path = eng.quest_root / "paper.md"
    path.write_text(f"# T\n\n## Introduction\n\nQueues form at booths. {SPECIFIC} They clear later.\n\n"
                    "## Limitations\nFew loads.\n\n## References\n1. A.\n", encoding="utf-8")
    g = {"claims": [claim], "total": 1, "status": "ok", "unsupported": [claim["claim"]]}
    eng._state_unsupported_claims({"paper_md": str(path), "claim_grounding": g})  # type: ignore[arg-type]
    body = path.read_text(encoding="utf-8").split("## Limitations")[0]
    assert "second lane is staffed" not in body and "Queues form at booths." in body


def test_a_claim_the_check_found_unsupported_on_a_title_only_source_is_no_source() -> None:
    """Review finding: the title-only rule skipped a sentence the check had already found unsupported (fault wording)."""
    claim = {"claim": SPECIFIC, "basis": "unsupported", "fault": "wording", "citation_index": 1, "quote": "",
             "evidence": "x"}
    out, changed = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [claim])
    assert changed == 1 and [(c["basis"], c["fault"]) for c in out] == [("unsupported", "no_source")]
    assert claim["fault"] == "wording", "the caller's claims are not changed in place"
    run = {**claim, "basis": "experiment", "fault": None}
    kept, n = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [run])
    assert n == 0 and kept[0]["fault"] is None  # a claim the run grounds is left alone


def test_answers_are_matched_by_order_only_when_none_matches_by_words(tmp_path: Path) -> None:
    """Review finding: two answers both named the first claim; the second was put on the other claim by order."""
    first = _reply({"claim": "The wait halves at the third load", "basis": "partly"},
                   {"claim": LOW, "basis": "unsupported", "fault": "maybe"})
    second = _reply({"claim": "The wait halves at the third load", "basis": "unsupported", "fault": "scope"},
                    {"claim": "The wait halves at the third load", "basis": "unsupported", "fault": "wording"})
    eng, calls = _engine(tmp_path, [first, second])
    g = asyncio.run(eng._node_claim_check(_state(tmp_path)))["claim_grounding"]  # type: ignore[arg-type]
    assert len(calls) == 2
    assert [(c["claim"][:12], c["fault"]) for c in g["claims"]] == [("The wait hal", "scope"), ("The wait is ", "unread")]


# --- review pass 2 ------------------------------------------------------------------------------------------------------

def test_two_answers_with_other_words_are_never_put_on_claims_by_order(tmp_path: Path) -> None:
    first = _reply({"claim": "The wait halves at the third load", "basis": "partly"},
                   {"claim": LOW, "basis": "unsupported", "fault": "maybe"})
    second = _reply({"claim": "The wait halved at load three", "basis": "unsupported", "fault": "scope"},
                    {"claim": "The wait halved at load three", "basis": "unsupported", "fault": "wording"})
    eng, _calls = _engine(tmp_path, [first, second])
    g = asyncio.run(eng._node_claim_check(_state(tmp_path)))["claim_grounding"]  # type: ignore[arg-type]
    assert [c["fault"] for c in g["claims"]] == ["unread", "unread"]


def test_one_answer_with_other_words_is_taken_for_the_one_claim_asked(tmp_path: Path) -> None:
    first = _reply(GOOD_CLAIM, {"claim": LOW, "basis": "partly"})
    second = _reply({"claim": "The wait is lowest at load one", "basis": "unsupported", "fault": "wording"})
    eng, _calls = _engine(tmp_path, [first, second])
    g = asyncio.run(eng._node_claim_check(_state(tmp_path)))["claim_grounding"]  # type: ignore[arg-type]
    assert [c["fault"] for c in g["claims"]] == [None, "wording"]


def test_a_shorter_claim_on_a_title_only_source_is_placed_by_the_papers_sentence() -> None:
    """Review finding: a model claim with no stored passage, marked no_source, would have been placed by its own shorter
    words, which equal no sentence, so nothing was cut."""
    short = {"claim": "Toll plaza queues settle within four arrivals of opening", "basis": "unsupported",
             "fault": "wording", "citation_index": 1, "quote": "", "evidence": "x"}
    out, _n = _apply_title_only_rule(f"# T\n\n## Introduction\n\n{SPECIFIC}\n", SOURCES, [short])
    assert out[0]["fault"] == "no_source" and out[0]["where"] == [{"passage": SPECIFIC}]
