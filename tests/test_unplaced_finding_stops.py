"""A code finding whose test fails and that FI's records cannot tie to a reported result stops the quest, in plain words. It is
neither kept with a sentence nor does it leave every result out. Where the records do place it (a rerun that moved one number, a
function of one result), only that result is left out, or nothing when the rerun shows nothing changed. Neutral made-up study (the
one of tests/test_shown_reach.py); no model is called."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.rerun_check import Verdict
from tests.test_shown_reach import DERIVE_COVER, _engine, _quest, _run, _stops

#: A helper that only the trial calls, in a trial whose result keys are built in a loop (the code does not say which key it feeds):
#: the call graph ties it to every result the trial makes, to none in particular.
LOOPED_TRIAL = {"simulate.py": (
    "import model\n\n\ndef trial(cell, ctx):\n    out = {}\n    for k in ('est_a', 'est_b', 'se_a', 'lo_a', 'hi_a'):\n"
    "        out[k] = model.helper_for_trial(1)\n    return out\n")}
HELPER = "from model import helper_for_trial\n\n\ndef test_x():\n    assert helper_for_trial(1) == 2, 'wrong'\n"
TRIAL_KEY = "import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r['est_b'] == 1.0, 'wrong'\n"


#: A failing test that reads no key of the results.
PLAIN = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert 'zz_none' in r, 'wrong'\n")


def _helper_gate(tmp_path: Path, verdict: Verdict | None, headline: tuple[str, ...] = ("cover",),
                 quantity: str = "zz_unknown") -> tuple[Any, dict[str, Any]]:
    engine = _engine(tmp_path)

    async def answers(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return _run(engine, HELPER, file="model.py", function="helper_for_trial", quantity=quantity, headline=headline, files=LOOPED_TRIAL)


def _missing_gate(tmp_path: Path, name: str, not_computed: tuple[str, ...] = (),
                  test: str = "") -> tuple[Any, dict[str, Any]]:
    """A finding whose only claim is that the code does not produce the plan's quantity ``name``; its test fails."""
    engine = _engine(tmp_path)
    root = _quest(engine)
    if not_computed:  # FI's record of the analysis: no value was worked out for these ids
        import json

        path = root / "data" / "results" / "derived.json"
        derived = json.loads(path.read_text(encoding="utf-8"))
        derived["complete"] = False
        derived["not_computed"] = [{"id": i, "status": "no_data", "why": "no trial gave a value"} for i in not_computed]
        for i in not_computed:
            derived["derived"].pop(i, None)
        path.write_text(json.dumps(derived), encoding="utf-8")
    rel = ".fi/finding_tests/F-1_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(test or TRIAL_KEY, encoding="utf-8")
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
    out, record = _missing_gate(tmp_path, "relative_zz", test=PLAIN)
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


# ---- review pass 1 ---------------------------------------------------------------------------------------------------------------
def test_an_unplaced_finding_is_not_placed_by_the_readers_quantity_word(tmp_path: Path) -> None:
    # the reader's word names a result the run holds (est_b), but no record of FI's ties the helper to it alone
    out, record = _helper_gate(tmp_path, None, quantity="est_b")
    assert _stops(out)
    assert "at_gate" not in record["issues"]["F-1"]


def test_a_part_not_done_is_kept_only_when_the_failing_test_reads_no_reported_key_that_holds_a_value(tmp_path: Path) -> None:
    # known positive: FI's record says no value was worked out for the id and the test reads no reported key: a part not done
    plain = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n"
             "    assert 'zz_none' in r, 'wrong'\n")
    out, record = _missing_gate(tmp_path / "a", "corr_ab", ("corr_ab",), plain)
    assert out is None and record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"
    # the same record, but the failing test reads a key the run reports a number for: the finding is placed on that key
    out, record = _missing_gate(tmp_path / "b", "corr_ab", ("corr_ab",), TRIAL_KEY)
    gate = record["issues"]["F-1"].get("at_gate")
    assert _stops(out) or (gate and gate["outcome"] == "left_out" and gate["keys"][0] == "est_b")


def test_a_helper_test_that_reads_a_reported_key_places_the_finding_on_that_key(tmp_path: Path) -> None:
    src = ("import simulate\nfrom model import helper_for_trial\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n"
           "    assert helper_for_trial(1) == 2 and r['est_b'] == 1.0, 'wrong'\n")
    out, record = _run(_engine(tmp_path), src, file="model.py", function="helper_for_trial", quantity="zz_unknown",
                       headline=("cover",))
    assert out is None
    keys = record["issues"]["F-1"]["at_gate"]["keys"]
    assert "est_b" in keys and "cover" not in keys


def test_a_helper_called_by_the_trial_and_by_one_result_s_function_is_placed_on_that_result(tmp_path: Path) -> None:
    files = {"derive.py": "import model\n\n\ndef derive_cover(table):\n    return model.helper_for_trial(1)\n\n\n"
                          "def derive_corr_ab(table):\n    return 0.5\n"}
    out, record = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="zz_unknown",
                       headline=("cover",), files=files)
    # the call graph links it to cover (and, through the trial, to the trial's results): placed by a record, so the stop is
    # about the main result, not the unplaced one
    assert _stops(out) and "main result" in out["rationale"] and "cannot find which of the results" not in out["rationale"]


def test_the_recorded_stop_still_names_the_finding_and_what_the_person_can_do(tmp_path: Path) -> None:
    from core.engine import _one_line

    long_how = "its test still fails on it; " + "the study could not be run again with a repair of it (" + "x" * 300 + ")"
    engine = _engine(tmp_path)
    from core.engine import _UNPLACED_FINDING

    issue = {"id": "F-1", "file": "model.py", "function": "helper_for_trial", "text": "the rate is computed twice " * 6}
    out = engine._shown_fault_stop(issue, long_how + "; " + _UNPLACED_FINDING)
    kept = _one_line(out["rationale"], 400)  # what the stop record keeps
    assert "helper_for_trial" in kept and "fix the code" in kept and "does not affect the results" in kept


# ---- review pass 2: the reader's word places nothing with a frozen plan ---------------------------------------------------------
def test_the_readers_quantity_word_never_leaves_out_a_result_the_test_does_not_read(tmp_path: Path) -> None:
    # the test calls the trial and reads only a key the run never holds; the reader names an existing result (est_b) for it
    src = "import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r.get('zz_missing') is not None, 'wrong'\n"
    engine = _engine(tmp_path)
    out, record = _run(engine, src, file="simulate.py", function="trial", quantity="est_b", headline=("cover",))
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "not_computed" and "est_b" not in gate["names"]  # the test's own key is what is not done
    assert not (engine.quest_root / "needs" / "RESULT_CHECK.json").exists() or "est_b" not in json.loads(
        (engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8")).get("removed_quantities", [])


def test_the_stop_of_a_placed_finding_also_leads_with_the_finding(tmp_path: Path) -> None:
    # a long failing-test description must not push the finding's name out of the first 400 characters of the record
    engine = _engine(tmp_path)
    out = engine._shown_fault_stop({"id": "F-1", "file": "model.py", "function": "helper_for_trial", "text": "a made-up fault"},
                                   "x" * 600)
    assert out["rationale"][:400].count("helper_for_trial") == 1
