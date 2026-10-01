"""Who accepted the result, and the question a person answers before they do (core/acceptance.py).

The user's rule: an accept that no person looked at is marked "not reviewed by a person" and reaches at most one level
below ``publication_ready``; before a person accepts, FI shows what the result does not guarantee and its gaps and
asks one question (have they reviewed the evidence record, and do they accept these claims and the limits listed?).
Only an explicit yes reaches ``publication_ready``; "I did not check" and "partly" (which needs a note) leave a gap;
"no" does not accept. The answer is recorded as a receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from core import acceptance, audit_log, evidence
from core.engine import _review_decision
from tests.test_evidence import ON, _quest, _state
from tests.test_pause_answers import _review_engine, _saved_state, no_llm  # noqa: F401 -- a fixture

PERSON = {"by": "person", "via": "cli", "question": acceptance.QUESTION, "answer": "yes"}


# --- the evidence --------------------------------------------------------------------------------------------------


def test_an_automatic_accept_stops_one_level_below_publication_ready_and_says_why(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    for automatic in ({}, {"acceptance": acceptance.automatic("auto_accept_on_pass")}):
        record = evidence.assess(root, _state(acceptance=automatic.get("acceptance")), settings=ON)
        assert record["status"] == "statistically_adequate", record["gaps"]
        assert record["gaps"] == [acceptance.NO_PERSON_GAP]
        assert record["accepted_by"] == "automatic" and record["acceptance"]["by"] == "automatic"
        # Marked on every surface that shows the one line.
        assert "Not reviewed by a person" not in evidence.summary_line(record)  # the gap itself says it
        assert acceptance.NO_PERSON_GAP in evidence.summary_line(record)


def test_the_mark_is_shown_even_when_another_gap_comes_first(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    record = evidence.assess(root, _state(acceptance=None), precision_missed=["p"], settings=ON)
    assert record["status"] == "independently_validated" and record["accepted_by"] == "automatic"
    assert "Not reviewed by a person" in evidence.summary_line(record)


def test_only_a_persons_explicit_yes_reaches_publication_ready(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    record = evidence.assess(root, _state(acceptance=PERSON), settings=ON)
    assert record["status"] == "publication_ready", record["gaps"]
    assert record["accepted_by"] == "person" and record["acceptance"]["answer"] == "yes"
    assert "Not reviewed by a person" not in evidence.summary_line(record)


def test_i_did_not_check_finishes_but_keeps_a_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    unchecked = evidence.assess(root, _state(acceptance={**PERSON, "answer": "not_checked"}), settings=ON)
    assert unchecked["status"] == "statistically_adequate", unchecked["gaps"]
    assert unchecked["gaps"] == [acceptance.NOT_CHECKED_GAP]
    assert acceptance.NOT_CHECKED_GAP == "no person reviewed the evidence before accepting"
    # Still a person's accept, recorded as such.
    assert unchecked["accepted_by"] == "person" and unchecked["acceptance"]["answer"] == "not_checked"
    # A person's record without an explicit yes (an unknown or missing answer) is not a yes either.
    for odd in ({**PERSON, "answer": None}, {"by": "person", "via": "cli"}):
        assert evidence.assess(root, _state(acceptance=odd), settings=ON)["status"] == "statistically_adequate"


def test_partly_keeps_a_gap_that_shows_the_note(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    note = "the 10x rate claim in table 2"
    partly = evidence.assess(root, _state(acceptance={**PERSON, "answer": "partly", "note": note}), settings=ON)
    assert partly["status"] == "statistically_adequate", partly["gaps"]
    (gap,) = partly["gaps"]
    assert gap.startswith(acceptance.PARTLY_GAP) and note in gap
    assert partly["acceptance"]["note"] == note
    assert note in evidence.summary_line(partly)


def test_the_accept_mark_is_shown_even_when_another_gap_comes_first(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    note = "the fit in figure 3"
    partly = evidence.assess(root, _state(acceptance={**PERSON, "answer": "partly", "note": note}),
                             precision_missed=["p"], settings=ON)
    assert partly["status"] == "independently_validated" and note in evidence.summary_line(partly)
    unchecked = evidence.assess(root, _state(acceptance={**PERSON, "answer": "not_checked"}), precision_missed=["p"],
                                settings=ON)
    assert "did not review the evidence" in evidence.summary_line(unchecked)


def test_a_persons_accept_of_an_earlier_paper_does_not_cover_the_paper_there_is_now(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    record = evidence.assess(root, _state(acceptance={**PERSON, "paper_sha256": "0" * 64}), settings=ON)
    assert record["status"] == "statistically_adequate" and record["gaps"] == [acceptance.NO_PERSON_GAP]
    assert record["acceptance"]["earlier"]["by"] == "person"


def test_a_persons_accept_covers_the_paper_it_was_given_for(tmp_path: Path) -> None:
    from core import receipts

    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    paper = receipts.sha256((root / "paper" / "paper.md").read_bytes())
    same = evidence.assess(root, _state(acceptance={**PERSON, "paper_sha256": paper}), settings=ON)
    assert same["status"] == "publication_ready" and same["accepted_by"] == "person"
    nothing = evidence.assess(root, _state(acceptance={**PERSON, "paper_sha256": ""}), settings=ON)
    assert nothing["accepted_by"] == "automatic" and nothing["gaps"] == [acceptance.NO_PERSON_GAP]


def test_a_person_who_accepts_against_the_reviewers_verdict_is_still_recorded(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    record = evidence.assess(root, _state(review={"verdict": "revise", "must_flag_hits": []}, acceptance=PERSON),
                             settings=ON)
    assert record["status"] == "statistically_adequate" and record["gaps"] == ["the review verdict is revise"]
    assert record["accepted_by"] == "person" and record["acceptance"]["answer"] == "yes"
    # A reject or refine leaves no acceptance: nothing is recorded as accepted.
    rejected = evidence.assess(root, _state(review={"verdict": "rejected", "must_flag_hits": []}, acceptance={}),
                               settings=ON)
    assert "accepted_by" not in rejected and "acceptance" not in rejected


def test_the_trace_says_who_accepted_in_plain_words() -> None:
    person = audit_log.describe({"kind": "result_accepted", "node": "human_feedback", **PERSON, "who": "ada"})
    assert "a person accepted the result (via the terminal, by ada)" in person and acceptance.QUESTION in person
    assert person.endswith("they answered yes")
    partly = audit_log.describe({"kind": "result_accepted", **PERSON, "answer": "partly", "note": "table 2"})
    assert partly.endswith("they answered partly; not accepted: table 2")
    auto = audit_log.describe({"kind": "result_accepted", "by": "automatic", "via": "auto_accept_on_pass"})
    assert "accepted automatically (automatic accept of a clean review)" in auto and "no person" in auto


def test_an_evidence_assessment_that_failed_is_not_shown_as_no_results() -> None:
    for failed in (None, evidence.unassessed("ValueError: x")):
        block = acceptance.shown(failed)
        assert block["reached"] == "" and "could not be worked out" in block["not_guaranteed"][0]


def test_an_exploration_accepted_by_a_person_is_still_not_publication_ready(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    record = evidence.assess(root, _state(acceptance=PERSON), settings={**ON, "result_use": "explore"})
    assert record["status"] != "publication_ready"
    assert any("set up to explore" in g for g in record["gaps"])


def test_a_review_waiting_for_a_decision_is_not_yet_accepted_by_anyone(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / ".fi").mkdir()
    (root / ".fi" / "human_review.json").write_text("{}", encoding="utf-8")
    record = evidence.assess(root, _state(acceptance=None), settings=ON)
    assert record["gaps"] == [acceptance.WAITING_GAP] and "accepted_by" not in record


# --- what is shown before an accept -------------------------------------------------------------------------------


def test_the_list_shown_before_an_accept_is_capped_and_has_no_duplicates() -> None:
    record = {
        "status": "internally_reconciled",
        "all_gaps": {
            "protocol_runtime_matched": ["gap one", "gap one", "Gap one.", acceptance.WAITING_GAP, "gap two"],
            "independently_validated": ["gap two", "gap three", "gap four"],
            "publication_ready": [acceptance.NO_PERSON_GAP, acceptance.NOT_CHECKED_GAP,
                                  acceptance.partly_gap("an earlier note"), "gap five"],
        },
        "gaps": ["gap one"],
    }
    block = acceptance.shown(record)
    assert block["gaps"] == ["gap one", "gap two", "gap three"]
    assert block["more_gaps"] == 2  # gap four, gap five; the decision's own gaps are never shown
    assert 1 <= len(block["not_guaranteed"]) <= acceptance.MAX_NOT_GUARANTEED
    assert block["not_guaranteed"][0].startswith("The script itself may be wrong")
    assert block["question"] == acceptance.QUESTION
    assert [c["id"] for c in block["choices"]] == ["yes", "partly", "no", "not_checked"]
    # The evidence record the person is asked about is named by its fingerprint.
    assert block["evidence_sha256"] == acceptance.evidence_sha256(record) and len(block["evidence_sha256"]) == 64
    assert acceptance.shown(None)["evidence_sha256"] == ""
    assert all(len(g) <= 240 for g in block["gaps"] + block["not_guaranteed"])
    assert acceptance.shown({"status": "executed", "all_gaps": {"internally_reconciled": ["x" * 900]}})["gaps"][0].endswith("…")
    assert acceptance.shown(None)["not_guaranteed"]  # no record: still says nothing is claimed


@pytest.mark.parametrize(("typed", "expect"), [
    ("yes", "yes"), ("Y", "yes"), ("partly", "partly"), ("no", "no"), ("not-checked", "not_checked"),
    ("I did not check", "not_checked"), ("unchecked", "not_checked"), ("", None), ("maybe", None),
])
def test_the_answers_a_person_may_type(typed: str, expect: str | None) -> None:
    assert acceptance.parse_answer(typed) == expect


def test_an_accept_needs_an_answer_and_no_is_not_an_accept() -> None:
    assert _review_decision({"action": "accept", "answer": "yes"})
    assert _review_decision({"action": "accept", "answer": "not_checked"})
    assert not _review_decision({"action": "accept", "feedback": ""})
    assert not _review_decision({"action": "accept", "answer": "no"})
    assert acceptance.problem({"action": "accept", "answer": "no"}) == acceptance.NOT_ACCEPTED_ON_NO
    assert _review_decision({"action": "reject"}) and _review_decision({"action": "refine", "feedback": "x"})


def test_partly_without_a_note_is_not_an_accept() -> None:
    for note in (None, "", "   "):
        answer = {"action": "accept", "answer": "partly", **({} if note is None else {"note": note})}
        assert not _review_decision(answer)
        assert acceptance.problem(answer) == acceptance.NOTE_NEEDED
    assert _review_decision({"action": "accept", "answer": "partly", "note": "table 2"})


def test_the_question_asks_about_the_evidence_not_what_was_expected() -> None:
    assert acceptance.QUESTION == ("Have you reviewed the evidence record, and do you accept these claims and the "
                                   "limits listed?")
    assert "expected" not in acceptance.QUESTION.lower()
    assert dict(acceptance.CHOICES)["not_checked"] == "I did not check"


def test_the_acceptance_receipt_binds_what_was_accepted_and_who_when_how() -> None:
    record = {"status": "executed", "all_gaps": {"internally_reconciled": ["the number check reported 2 finding(s)"]}}
    block = acceptance.shown(record)
    snap = {"verdict": "accept", "before_accept": block}
    stamped = acceptance.stamp({"action": "accept", "answer": "partly", "note": "  table 2  ", "via": "web",
                                "who": "ada", "at": "2026-09-30T08:00:00+00:00",
                                "acceptance": {"by": "automatic"}}, "callback", snap)
    receipt = stamped["acceptance"]
    assert receipt["by"] == "person" and receipt["via"] == "web" and receipt["who"] == "ada"
    assert receipt["at"] == "2026-09-30T08:00:00+00:00"
    assert receipt["question"] == acceptance.QUESTION and receipt["answer"] == "partly" and receipt["note"] == "table 2"
    assert receipt["evidence_sha256"] == block["evidence_sha256"] and len(receipt["evidence_sha256"]) == 64
    # The limits that were listed with the question, word for word.
    assert receipt["limits_shown"] == acceptance.lines(block)
    assert any("the number check reported 2 finding(s)" in line for line in receipt["limits_shown"])
    # What the interface did not say is filled in, never left to look like a decision nobody made.
    bare = acceptance.stamp({"action": "accept", "answer": "yes"}, "callback")["acceptance"]
    assert bare["who"] == "not given" and bare["via"] == "callback" and bare["at"].endswith("+00:00")
    assert bare["note"] == "" and bare["limits_shown"] == [] and bare["evidence_sha256"] == ""
    # A time that is not a time is not taken.
    assert acceptance.stamp({"action": "accept", "answer": "yes", "at": "yesterday"}, "x")["acceptance"]["at"] != "yesterday"


def test_a_note_is_kept_only_with_partly() -> None:
    stamped = acceptance.stamp({"action": "accept", "answer": "yes", "note": "table 2 is wrong"}, "answer file")
    assert stamped["acceptance"]["note"] == "" and acceptance.review_gap(stamped["acceptance"]) is None


def test_the_receipt_keeps_the_fingerprint_that_was_shown_and_says_when_it_changed() -> None:
    block = acceptance.shown({"status": "executed"})
    snap = {"before_accept": block}
    same = acceptance.stamp({"action": "accept", "answer": "yes", "shown_evidence_sha256": block["evidence_sha256"]},
                            "x", snap)["acceptance"]
    assert same["evidence_sha256"] == block["evidence_sha256"] and "evidence_sha256_at_accept" not in same
    other = "a" * 64
    moved = acceptance.stamp({"action": "accept", "answer": "yes", "shown_evidence_sha256": other}, "x", snap)
    assert moved["acceptance"]["evidence_sha256"] == other
    assert moved["acceptance"]["evidence_sha256_at_accept"] == block["evidence_sha256"]
    # The limits worked out now are not the ones the person read: kept under their own name.
    assert moved["acceptance"]["limits_shown"] == [] and moved["acceptance"]["limits_at_accept"] == acceptance.lines(block)
    # Not a fingerprint: the one handed to the interface is kept.
    junk = acceptance.stamp({"action": "accept", "answer": "yes", "shown_evidence_sha256": "x"}, "x", snap)
    assert junk["acceptance"]["evidence_sha256"] == block["evidence_sha256"]


def test_a_time_still_to_come_is_not_taken() -> None:
    assert acceptance.stamp({"action": "accept", "answer": "yes", "at": "2999-01-01T00:00:00+00:00"},
                            "x")["acceptance"]["at"] != "2999-01-01T00:00:00+00:00"


def test_an_accept_given_to_the_earlier_question_is_not_a_yes_to_this_one(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    earlier = {**PERSON, "question": "Do the main numbers match what you expected?"}
    record = evidence.assess(root, _state(acceptance=earlier), settings=ON)
    assert record["status"] == "statistically_adequate" and record["gaps"] == [acceptance.EARLIER_QUESTION_GAP]


def test_a_record_worked_out_before_this_rule_is_not_described_by_it() -> None:
    # needs/EVIDENCE.json read back as it was written: publication_ready with an older person's accept.
    old = {"status": "publication_ready", "gaps": [], "all_gaps": {}, "accepted_by": "person",
           "acceptance": {"by": "person", "via": "cli", "answer": "not_checked"}}
    line = evidence.summary_line(old)
    assert "did not review" not in line and "only in part" not in line


def test_the_interactive_prompt_records_the_approve_as_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    replies = iter(["accept", "yes"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    callback = launch._pick_human_feedback_callback(None, None, True, "Ada Lovelace")  # type: ignore[arg-type]
    assert asyncio.run(callback({"verdict": "accept"}))["who"] == "Ada Lovelace"


def test_a_note_with_yes_is_refused_rather_than_dropped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    with pytest.raises(SystemExit) as exc:
        launch._apply_review_decision(_args(accept=["yes", "table 2 is wrong"]), tmp_path)
    assert exc.value.code == 2 and not (fi / "human_review_answer.json").exists()
    assert '--accept partly "table 2 is wrong"' in capsys.readouterr().err


@pytest.mark.parametrize("words", [["not", "checked"], ["I", "did", "not", "check"]])
def test_an_answer_typed_as_several_words_is_the_answer(tmp_path: Path, words: list[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    launch._apply_review_decision(_args(accept=words), tmp_path)
    staged = _staged(fi)
    assert staged["answer"] == "not_checked" and staged["note"] == ""


def test_an_answer_cannot_claim_to_be_automatic_or_by_someone_else() -> None:
    stamped = acceptance.stamp({"action": "accept", "answer": "yes", "via": "web",
                                "acceptance": {"by": "automatic"}}, "callback")
    assert stamped["acceptance"]["by"] == "person" and stamped["acceptance"]["via"] == "web"
    assert "acceptance" not in acceptance.stamp({"action": "reject", "acceptance": {"by": "person"}}, "x")


# --- the review pause, through the real run loop ------------------------------------------------------------------


async def test_a_persons_accept_through_the_callback_records_their_answer(tmp_path: Path, no_llm: None) -> None:
    seen: list[dict] = []

    async def person(snapshot: dict) -> Any:
        seen.append(snapshot)
        return {"action": "accept", "feedback": "", "answer": "partly", "note": "the rate in table 2",
                "via": "vscode", "who": "ada"}

    eng = _review_engine(tmp_path)
    await asyncio.wait_for(eng.run(human_feedback_callback=person), timeout=120)
    # The person was shown what the result does not guarantee and asked the question.
    shown = seen[0]["before_accept"]
    assert shown["question"] == acceptance.QUESTION and shown["not_guaranteed"]
    assert len(shown["evidence_sha256"]) == 64
    state = await _saved_state(eng)
    receipt = state["acceptance"]
    assert receipt["by"] == "person" and receipt["via"] == "vscode" and receipt["who"] == "ada"
    assert receipt["answer"] == "partly" and receipt["note"] == "the rate in table 2"
    # Bound to the paper and the evidence record that were shown, and to the limits listed.
    # (This review-only engine writes no paper, so its hash is empty; a real paper's is checked in test_research_acceptance.)
    assert "paper_sha256" in receipt and receipt["evidence_sha256"] == shown["evidence_sha256"]
    assert receipt["limits_shown"] == acceptance.lines(shown) and receipt["at"]
    events = [e for e in audit_log.read(eng.fi_dir / "audit.jsonl") if e["kind"] == "result_accepted"]
    assert len(events) == 1 and events[0]["by"] == "person" and events[0]["answer"] == "partly"
    assert events[0]["note"] == "the rate in table 2" and events[0]["evidence_sha256"] == shown["evidence_sha256"]
    from core import attempt_records as ar

    (end,) = [r for r in ar.read(eng.fi_dir, ar.ATTEMPTS) if r.get("kind") == "quest"]
    assert end["person"] == "accept"


async def test_a_staged_accept_from_the_command_line_or_the_web_records_its_answer(tmp_path: Path,
                                                                                  no_llm: None) -> None:
    eng = _review_engine(tmp_path)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "human_review_answer.json").write_text(
        json.dumps({"action": "accept", "feedback": "", "answer": "yes", "via": "web"}), encoding="utf-8")
    await asyncio.wait_for(eng.run(), timeout=120)
    state = await _saved_state(eng)
    assert state["human_feedback"]["action"] == "accept"
    assert state["acceptance"]["by"] == "person" and state["acceptance"]["via"] == "web"


@pytest.mark.parametrize("answer", [
    {"action": "accept", "feedback": ""},                       # no answer to the question
    {"action": "accept", "feedback": "", "answer": "no"},       # "no" does not accept
    {"action": "accept", "feedback": "", "answer": "partly"},   # "partly" needs its note
])
async def test_an_accept_without_an_answer_or_with_no_stops_instead(tmp_path: Path, no_llm: None,
                                                                    answer: dict) -> None:
    async def person(snapshot: dict) -> Any:
        return answer

    eng = _review_engine(tmp_path)
    await asyncio.wait_for(eng.run(human_feedback_callback=person), timeout=120)
    state = await _saved_state(eng)
    assert "human_feedback" not in state and "acceptance" not in state
    assert (eng.fi_dir / "human_review.json").is_file()
    card = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert acceptance.QUESTION in card and "--accept yes" in card


async def test_an_automatic_accept_is_recorded_as_automatic(tmp_path: Path, no_llm: None) -> None:
    eng = _review_engine(tmp_path)
    eng.auto_accept_on_pass = True
    # A reviewer's own clean accept, so auto_accept_on_pass takes it.
    await asyncio.wait_for(eng.run(), timeout=120)
    state = await _saved_state(eng)
    assert state["human_feedback"]["action"] == "accept"
    assert state["acceptance"]["by"] == "automatic" and state["acceptance"]["via"] == "auto_accept_on_pass"
    events = [e for e in audit_log.read(eng.fi_dir / "audit.jsonl") if e["kind"] == "result_accepted"]
    assert [e["by"] for e in events] == ["automatic"]
    # The quest's own record does not say a person decided.
    from core import attempt_records as ar

    (end,) = [r for r in ar.read(eng.fi_dir, ar.ATTEMPTS) if r.get("kind") == "quest"]
    assert end["person"] is None and end["accepted_by"] == "automatic"


def test_resuming_after_the_review_pause_keeps_the_environment_the_experiment_ran_on(tmp_path: Path) -> None:
    """A person's accept resumes the quest; the run's start must not replace the record made after the installs."""
    eng = _review_engine(tmp_path)
    env = eng.quest_root / "needs" / "ENVIRONMENT.json"
    env.parent.mkdir(parents=True, exist_ok=True)
    kept = {"recorded": "after installing the experiment's packages", "packages": ["numpy==2.0"], "isolated": True}
    env.write_text(json.dumps(kept), encoding="utf-8")
    asyncio.run(eng._record_environment())
    assert json.loads(env.read_text(encoding="utf-8")) == kept


# --- the command line ---------------------------------------------------------------------------------------------


def _paused(tmp_path: Path) -> Path:
    fi = tmp_path / "q1" / ".fi"
    fi.mkdir(parents=True)
    (fi / "pause.json").write_text(json.dumps({"kind": "review"}), encoding="utf-8")
    (fi / "human_review.json").write_text(json.dumps({"before_accept": acceptance.shown(
        {"status": "executed", "all_gaps": {"internally_reconciled": ["the number check reported 2 finding(s)"]}})}),
        encoding="utf-8")
    return fi


def _args(**over: Any) -> argparse.Namespace:
    return argparse.Namespace(**{"accept": None, "reject": False, "refine": None, "resume": "q1", "approve_as": "",
                                 **over})


def _staged(fi: Path) -> dict:
    staged = json.loads((fi / "human_review_answer.json").read_text(encoding="utf-8"))
    assert staged.pop("at")  # when the person answered
    # The fingerprint of the evidence record whose limits were printed goes back with the answer.
    shown = json.loads((fi / "human_review.json").read_text(encoding="utf-8"))["before_accept"]
    assert staged.pop("shown_evidence_sha256") == shown["evidence_sha256"]
    return staged


def test_accept_with_an_answer_stages_it_after_showing_the_limits(tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    launch._apply_review_decision(_args(accept=["not-checked"], approve_as="ada"), tmp_path)
    assert _staged(fi) == {"action": "accept", "feedback": "", "answer": "not_checked", "note": "", "via": "cli",
                           "who": "ada"}
    # The limits were listed before the answer was taken, also when it came with the flag.
    assert "Gap: the number check reported 2 finding(s)" in capsys.readouterr().out


def test_accept_names_the_login_when_no_name_is_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    fi = _paused(tmp_path)
    monkeypatch.setattr("getpass.getuser", lambda: "jdoe")
    launch._apply_review_decision(_args(accept=["yes"]), tmp_path)
    assert _staged(fi)["who"] == "jdoe"


def test_accept_partly_carries_its_note(tmp_path: Path) -> None:
    import launch

    fi = _paused(tmp_path)
    args = launch.parse_args(["--config", "x.yaml", "--resume", "q1", "--accept", "partly", "the rate in table 2"])
    launch._apply_review_decision(args, tmp_path)
    staged = _staged(fi)
    assert staged["answer"] == "partly" and staged["note"] == "the rate in table 2"


def test_accept_partly_without_a_note_and_no_terminal_says_how(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: False)
    with pytest.raises(SystemExit) as exc:
        launch._apply_review_decision(_args(accept=["partly"]), tmp_path)
    assert exc.value.code == 2 and not (fi / "human_review_answer.json").exists()
    assert '--accept partly "' in capsys.readouterr().err


def test_accept_partly_at_a_terminal_asks_for_the_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    fi = _paused(tmp_path)
    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: True)
    replies = iter(["", "figure 3's fit"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    launch._apply_review_decision(_args(accept=["partly"]), tmp_path)
    assert _staged(fi)["note"] == "figure 3's fit"


def test_accept_no_is_refused_and_says_what_to_do(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    with pytest.raises(SystemExit) as exc:
        launch._apply_review_decision(_args(accept=["no"]), tmp_path)
    assert exc.value.code == 2 and not (fi / "human_review_answer.json").exists()
    assert "--refine" in capsys.readouterr().err


def test_accept_without_an_answer_and_no_terminal_shows_the_list_and_the_question(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    fi = _paused(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit) as exc:
        launch._apply_review_decision(_args(accept=[]), tmp_path)
    err = capsys.readouterr().err
    assert exc.value.code == 2 and not (fi / "human_review_answer.json").exists()
    assert acceptance.QUESTION in err and "the number check reported 2 finding(s)" in err and "--accept yes" in err


def test_accept_without_an_answer_at_a_terminal_asks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    fi = _paused(tmp_path)
    monkeypatch.setattr(launch, "_stdin_is_terminal", lambda: True)
    replies = iter(["partly", "table 2"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    launch._apply_review_decision(_args(accept=[]), tmp_path)
    staged = _staged(fi)
    assert staged["answer"] == "partly" and staged["note"] == "table 2"


def test_the_interactive_prompt_asks_before_an_accept(monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    replies = iter(["accept", "yes"])
    prompts: list[str] = []

    def reply(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", reply)
    monkeypatch.setattr("getpass.getuser", lambda: "jdoe")
    block = acceptance.shown(
        {"status": "executed", "all_gaps": {"internally_reconciled": ["the number check reported 2 finding(s)"]}})
    got = asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept", "before_accept": block}))
    assert got.pop("at")
    assert got == {"action": "accept", "feedback": "", "answer": "yes", "note": "", "via": "cli", "who": "jdoe",
                   "shown_evidence_sha256": block["evidence_sha256"]}
    assert acceptance.QUESTION in prompts[-1]
    out = capsys.readouterr().out
    assert "Not guaranteed: Nothing says the results are right" in out
    assert "Gap: the number check reported 2 finding(s)" in out


def test_the_interactive_prompt_asks_for_the_note_after_partly(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    replies = iter(["accept", "partly", "", "the rate in table 2"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    got = asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"}))
    assert got["answer"] == "partly" and got["note"] == "the rate in table 2"
    # Nobody writes the note: nothing is accepted.
    replies = iter(["accept", "partly", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    assert asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"})) == {}


def test_the_interactive_prompt_offers_refine_after_no(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    replies = iter(["", "no", "refine", "the rate in table 2 is off by 10x"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    got = asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"}))
    assert got == {"action": "refine", "feedback": "the rate in table 2 is off by 10x"}


def test_the_interactive_prompt_stops_to_look_again_after_no(monkeypatch: pytest.MonkeyPatch) -> None:
    import launch

    replies = iter(["accept", "no", "look"])
    monkeypatch.setattr("builtins.input", lambda *_a: next(replies))
    assert asyncio.run(launch._cli_human_feedback_callback({"verdict": "accept"})) == {}


# --- the web page -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("body", "status"), [
    ({"action": "accept", "feedback": ""}, 400),                    # no answer: refused
    ({"action": "accept", "feedback": "", "answer": "no"}, 400),    # "no" does not accept
    ({"action": "accept", "feedback": "", "answer": "partly"}, 400),  # "partly" needs its note
    ({"action": "accept", "feedback": "", "answer": "yes", "note": "table 2"}, 400),  # a note only with partly
    ({"action": "accept", "feedback": "", "answer": "not_checked"}, 200),
    ({"action": "accept", "feedback": "", "answer": "partly", "note": " table 2 ", "who": "ada"}, 200),
])
def test_the_web_accept_needs_an_answer_and_records_it(tmp_path: Path, body: dict, status: int) -> None:
    from fastapi.testclient import TestClient

    from tests.test_web_server import _mk_quest_dir
    from web.server import make_app

    q = _mk_quest_dir(tmp_path, "qacc")
    (q / ".fi" / "human_review.json").write_text(json.dumps({"verdict": "accept"}), encoding="utf-8")
    r = TestClient(make_app(tmp_path)).post("/api/quests/qacc/human-review", json=body)
    assert r.status_code == status, r.text
    staged = q / ".fi" / "human_review_answer.json"
    if status == 200:
        got = json.loads(staged.read_text(encoding="utf-8"))
        assert got.pop("at")
        assert got == {"action": "accept", "feedback": "", "answer": body["answer"],
                       "note": "table 2" if body["answer"] == "partly" else "", "via": "web",
                       "who": body.get("who", "")}
    else:
        assert not staged.exists()
        if body.get("answer") == "no":
            assert "refine" in r.text
        if body.get("answer") == "partly":
            assert "what you do not accept" in r.text


def test_the_web_accept_sends_back_the_fingerprint_it_showed(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from tests.test_web_server import _mk_quest_dir
    from web.server import make_app

    q = _mk_quest_dir(tmp_path, "qfp")
    (q / ".fi" / "human_review.json").write_text(json.dumps({"verdict": "accept"}), encoding="utf-8")
    r = TestClient(make_app(tmp_path)).post("/api/quests/qfp/human-review", json={
        "action": "accept", "feedback": "", "answer": "yes", "shown_evidence_sha256": "b" * 64})
    assert r.status_code == 200, r.text
    staged = json.loads((q / ".fi" / "human_review_answer.json").read_text(encoding="utf-8"))
    assert staged["shown_evidence_sha256"] == "b" * 64
    page = (Path(__file__).resolve().parents[1] / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "shown_evidence_sha256: (humanReviewBeforeAccept && humanReviewBeforeAccept.evidence_sha256)" in page


def test_the_web_review_panel_asks_before_it_accepts() -> None:
    page = (Path(__file__).resolve().parents[1] / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert 'id="hr-before-accept"' in page and "showBeforeAccept()" in page
    assert acceptance.QUESTION in page and "I did not check" in page
    # The Accept button opens the question; only an answer submits it, and "no" points at refine.
    assert "btn.dataset.action === 'accept'" in page and "answer === 'no'" in page
    # "Partly" needs the note; the name is optional and sent with it.
    assert 'id="hr-accept-note"' in page and 'id="hr-accept-who"' in page and "answer === 'partly'" in page


# --- VS Code ------------------------------------------------------------------------------------------------------


def test_the_vscode_bridge_passes_the_answer_on_and_names_the_interface() -> None:
    from core.vscode_bridge import VSCodeBridgeClient

    async def go(msg: dict) -> Any:
        client = VSCodeBridgeClient(host="127.0.0.1", port=1)
        fut = asyncio.get_running_loop().create_future()
        client._pending[3] = fut
        client._dispatch({"type": "human_review_response", "id": 3, **msg})
        return await fut

    got = asyncio.run(go({"action": "accept", "feedback": "", "answer": "partly", "note": "table 2", "who": "ada"}))
    assert got.pop("at")
    assert got == {"action": "accept", "feedback": "", "answer": "partly", "note": "table 2", "via": "vscode",
                   "who": "ada"}
    assert _review_decision(got)
    # An older extension that sends no answer: no decision, so nothing is accepted.
    assert not _review_decision(asyncio.run(go({"action": "accept", "feedback": ""})))
    # ... nor a "partly" with no note.
    assert not _review_decision(asyncio.run(go({"action": "accept", "feedback": "", "answer": "partly"})))
    # The fingerprint of the evidence record the chat showed goes back with the accept.
    sent = asyncio.run(go({"action": "accept", "feedback": "", "answer": "yes", "shown_evidence_sha256": "c" * 64}))
    assert sent["shown_evidence_sha256"] == "c" * 64


def test_the_vscode_review_asks_the_question_and_no_does_not_accept() -> None:
    src = (Path(__file__).resolve().parents[1] / "vscode-frontier-insight" / "src" / "bridge.ts").read_text(
        encoding="utf-8")
    assert acceptance.QUESTION in src and "I did not check" in src
    assert "beforeAcceptMarkdown(block)" in src and 'picked.id === "no"' in src
    assert '"Look again"' in src and "human_review_cancelled" in src
    # "Partly" asks for the note and will not take an empty one; the login name goes with the answer.
    assert 'picked.id === "partly"' in src and "validateInput" in src and "os.userInfo().username" in src
    assert "shown_evidence_sha256: snap.before_accept?.evidence_sha256" in src
