"""A code finding whose test fails and that FI's records cannot tie to a reported result stops the quest, in plain words. It is
neither kept with a sentence nor does it leave every result out. Where the records do place it (a rerun that moved one number, a
function of one result), only that result is left out, or nothing when the rerun shows nothing changed. Neutral made-up study (the
one of tests/test_shown_reach.py); no model is called."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from core.rerun_check import Verdict
from tests.test_shown_reach import DERIVE_COVER, _engine, _quest, _run, _stops

#: A helper that only the trial calls: the call graph ties it to every result the trial makes, to none in particular.
HELPER = "from model import helper_for_trial\n\n\ndef test_x():\n    assert helper_for_trial(1) == 2, 'wrong'\n"
TRIAL_KEY = "import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r['est_b'] == 1.0, 'wrong'\n"


def _helper_gate(tmp_path: Path, verdict: Verdict | None, headline: tuple[str, ...] = ("cover",),
                 quantity: str = "zz_unknown") -> tuple[Any, dict[str, Any]]:
    engine = _engine(tmp_path)

    async def answers(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return _run(engine, HELPER, file="model.py", function="helper_for_trial", quantity=quantity, headline=headline)


def _missing_gate(tmp_path: Path, name: str) -> tuple[Any, dict[str, Any]]:
    """A finding whose only claim is that the code does not produce the plan's quantity ``name``; its test fails."""
    engine = _engine(tmp_path)
    root = _quest(engine)
    rel = ".fi/finding_tests/F-1_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(TRIAL_KEY, encoding="utf-8")
    issue = {"id": "F-1", "file": "", "function": "", "quantity": name, "kind": "quantity", "changes_result": True,
             "demonstrated": True, "test": rel, "text": f"the {name} is not computed", "outcome": "not repaired",
             "failure_kind": "assertion", "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1}
    record = {"issues": {"F-1": issue},
              "read_problems": [{"id": "R1", "kind": "metric", "status": "missing", "name": name, "fid": "F-1", "text": "x"}]}
    state = {"design": None, "exec_result": {"returncode": 0}}
    from tests.test_shown_reach import RESULT

    state["result_json"] = RESULT
    out = asyncio.run(engine._shown_fault_check([issue], record, RESULT, ["est_a"], state))
    return out, record


# ---- a finding the records cannot place stops ------------------------------------------------------------------------------------
def test_a_helper_only_the_trial_reaches_is_not_placed_and_stops_instead_of_leaving_every_trial_result_out(tmp_path: Path) -> None:
    # the finding's words name no result; the main result (cover) is not made by the trial: the old gate left est_a, est_b and
    # what is worked out from them out and went on with `cover` alone
    out, record = _helper_gate(tmp_path, None)
    assert _stops(out)
    assert "at_gate" not in record["issues"]["F-1"]
    assert not list(tmp_path.rglob("RESULT_CHECK.json"))  # nothing was left out


def test_the_stop_says_in_plain_words_which_finding_why_and_what_the_person_can_do(tmp_path: Path) -> None:
    out, _ = _helper_gate(tmp_path, None)
    text = out["rationale"]
    assert "its test still fails" in text or "failing test" in text
    assert "cannot find which of the results it changes" in text
    assert "fix the code" in text and "does not affect the results" in text
    assert "helper_for_trial" in text  # the code finding is named


def test_a_part_not_done_whose_failing_test_no_record_places_stops_instead_of_being_kept(tmp_path: Path) -> None:
    # a quantity the plan does not declare, called missing by the reader: the old gate kept the results and wrote
    # "part not done"; the failing test could stand for a value reported under another name
    out, record = _missing_gate(tmp_path, "relative_zz")
    assert _stops(out)
    assert "cannot find which of the results it changes" in out["rationale"]
    assert "at_gate" not in record["issues"]["F-1"]


# ---- known positives: what FI's records place keeps today's behaviour ---------------------------------------------------------------
def test_a_rerun_that_moved_one_number_leaves_only_that_result_out_and_stops_nothing(tmp_path: Path) -> None:
    moved = Verdict("changed", metrics=["est_b"], keys=["trials/noise=0.1/est_b"])
    out, record = _helper_gate(tmp_path, moved)
    assert out is None
    keys = record["issues"]["F-1"]["at_gate"]["keys"]
    assert "est_b" in keys and "est_a" not in keys and "cover" not in keys


def test_a_rerun_that_moved_nothing_stops_nothing_and_leaves_nothing_out(tmp_path: Path) -> None:
    out, record = _helper_gate(tmp_path, Verdict("unchanged", mode="whole study"))
    assert out is None
    issue = record["issues"]["F-1"]
    assert issue["rerun_unchanged"] is True and "at_gate" not in issue


def test_a_test_that_reads_a_result_s_key_literally_places_the_finding_and_leaves_only_that_result_out(tmp_path: Path) -> None:
    out, record = _run(_engine(tmp_path), TRIAL_KEY, file="simulate.py", function="trial", quantity="est_b")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "est_b"


def test_a_function_of_one_result_places_the_finding(tmp_path: Path) -> None:
    out, record = _run(_engine(tmp_path), DERIVE_COVER, quantity="cover", headline=("est_a",))
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"] == ["cover"]
