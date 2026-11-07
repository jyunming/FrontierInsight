"""A known-answer check the gate's short side run could not decide for lack of time is judged again on the delivered run.

The real case (backlog -385): a false-discovery study whose gate measured three checks on 320-5,120 trials and left them
"not decided" (``mc.why == "time"``), while the delivered run held 8,000 trials of the same settings. FI's own rule
(core/mc_error.py) decides two of them on those trials. Nothing re-judged them, so the paper said "not validated".

Here the record and the trials are small, made-up ones with neutral names; the real record is replayed in
``C:/fi_runs/replays/2026-11-07_b385_rejudge_delivered``. Nothing here calls a model or runs a simulation.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from core import accepted_checks as ac, mc_error as mc, oracle_check as oc

VERSION = "code-v1"
PROTOCOL: dict[str, Any] = {
    "runs_per_setting": 4000,
    "metrics": [{"id": "hit", "kind": "mean", "estimand": "the mean hit score", "unit": "one run"},
                {"id": "gain", "kind": "mean", "estimand": "the mean gain", "unit": "one run"}],
    "oracles": [
        {"name": "hit_at_ten", "case": {"size": 10}, "measure": "mean(hit)", "expected": 0.5, "tolerance": 0.02},
        {"name": "gain_at_ten", "case": {"size": 10}, "measure": "mean(gain)", "expected": 0.2, "tolerance": 0.02},
        {"name": "hit_at_forty", "case": {"size": 40}, "measure": "mean(hit)", "expected": 0.5, "tolerance": 0.02},
        {"name": "hit_close", "case": {"size": 10}, "measure": "mean(hit)", "expected": 0.51, "tolerance": 0.01},
    ],
}
N_GATE, N_DELIVERED = 100, 4000


def _cell(size: int, n: int = N_DELIVERED, failed: int = 0) -> dict[str, Any]:
    rng = random.Random(size)
    ids = list(range(n))
    return {"cell": {"size": size}, "key": f"size={size}", "planned": n + failed, "ok": n, "failed": failed,
            "metrics": {"hit": {"values": [rng.gauss(0.5, 0.1) for _ in ids], "trials": ids},
                        "gain": {"values": [rng.gauss(0.2, 0.1) for _ in ids], "trials": ids}}}


def _trials(*sizes: int, n: int = N_DELIVERED) -> dict[str, Any]:
    return {"schema": "fi.trials/v1", "thresholds": {}, "cells": [_cell(s, n) for s in sizes]}


def _undecided(oracle: dict[str, Any], *, why: str = "time", version: str = VERSION, verdict: str = "undecided",
               trials: int = N_GATE, basis: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """One check as the gate left it: its judged entry and the entry of the list of checks the quest went on with."""
    expected, limit, _ = oc.limit_of(oracle)
    stat = {"trials": trials, "se": 0.01, "on": "means", "varies": True, "across": True}
    judged = {"name": oracle["name"], "value": 0.5, "expected": expected, "tolerance": oracle["tolerance"],
              "mode": "absolute", "limit": limit, "passed_by_engine": None, "script_said": None, "measured_by": "engine",
              "mc": mc.record(stat, read_as="mean", verdict=verdict, why=why, failed=0, started=trials), "undecided": True}
    entry = {"name": oracle["name"], "fingerprint": ac.fingerprint(oracle), "expected": expected, "limit": limit,
             "tolerance": oracle["tolerance"], "tolerance_mode": "absolute", "case": oracle["case"],
             "measure": oracle["measure"], "measured": 0.5, "measured_by": "engine",
             "mc_words": mc.words(judged["mc"]), "undecided": True,
             "undecided_words": mc.undecided_words(judged["mc"], limit), "by": ac.AUTOMATIC, "via": ac.AUTOMATIC,
             "script": "code/simulate.py", "script_version": version,
             "trial_basis": oc.trial_basis(PROTOCOL) if basis is None else basis}
    return judged, entry


def _record(*pairs: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    problems = [f"the oracle {j['name']!r} was not decided: FI measured 0.5" for j, _ in pairs]
    entries = [e for _, e in pairs]
    return {"status": ac.WENT_ON, "judged_by": "engine", "problems": list(problems),
            "attempts": [{"attempt": 0, "oracles": [j["name"] for j, _ in pairs], "problems": list(problems),
                          "checks": [{"name": j["name"], "value": 0.5, "measured_by": "engine", "mc": j["mc"]}
                                     for j, _ in pairs],
                          "judged": [j for j, _ in pairs],
                          "triage": [{"check": j["name"], "kind": "trials", "verdict": "undecided"} for j, _ in pairs]}],
            "set_aside": {j["name"]: "not decided" for j, _ in pairs},
            "went_on": entries, "summary": "; ".join(ac.gap(e) for e in entries), "explained": {"leaning": "check"}}


def _oracle(name: str, **changes: Any) -> dict[str, Any]:
    return {**next(o for o in PROTOCOL["oracles"] if o["name"] == name), **changes}


def _protocol(*oracles: dict[str, Any]) -> dict[str, Any]:
    return {**PROTOCOL, "oracles": list(oracles)}


def _judged(record: dict[str, Any], name: str) -> dict[str, Any]:
    return next(j for j in record["attempts"][-1]["judged"] if j["name"] == name)


def test_a_check_undecided_for_time_whose_case_is_a_delivered_setting_becomes_a_pass() -> None:
    oracle = _oracle("hit_at_ten")
    record = _record(_undecided(oracle))
    new, names = oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION)
    assert names == ["hit_at_ten"]
    j = _judged(new, "hit_at_ten")
    assert j["passed_by_engine"] is True and not j.get("undecided")
    assert j["judged_on"] == oc.DELIVERED_RUN == j["mc"]["judged_on"]
    assert j["mc"]["trials"] == N_DELIVERED and j["mc"]["verdict"] == "pass" and j["mc"]["why"] == ""
    # nothing is left to go on with: the record says the checks passed, and every sentence that follows it agrees
    assert new["status"] == "ok" and "went_on" not in new and "summary" not in new and "set_aside" not in new
    assert new["problems"] == [] and new["attempts"][-1]["problems"] == []
    assert not [t for t in new["attempts"][-1]["triage"] if t["kind"] == "trials"]
    assert ac.failing_disclosure(new) == ""
    note = oc.analysis_note(oc.last_judged(new))
    assert "passed" in note and "judged on the delivered run" in note and "not decided" not in note
    assert oc.engine_passed(oc.last_judged(new)) == ["hit_at_ten"]


def test_the_record_given_is_not_changed() -> None:
    oracle = _oracle("hit_at_ten")
    record = _record(_undecided(oracle))
    before = copy.deepcopy(record)
    oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION)
    assert record == before


def test_a_planted_wrong_expected_value_is_a_fail_and_stays_a_failure() -> None:
    oracle = _oracle("hit_at_ten", expected=0.6)  # the delivered mean is 0.5
    record = _record(_undecided(oracle))
    new, names = oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION)
    assert names == ["hit_at_ten"]
    assert _judged(new, "hit_at_ten")["passed_by_engine"] is False
    assert new["status"] == ac.WENT_ON
    (entry,) = new["went_on"]
    assert not entry.get("undecided") and "undecided_words" not in entry
    assert "failed" in ac.gap(entry) and "judged on the delivered run" in ac.gap(entry)
    assert "FAILED" in ac.failing_disclosure(new)
    assert any("failed on the delivered run" in p for p in new["problems"])


def test_a_case_that_is_not_a_setting_of_the_delivered_run_stays_undecided() -> None:
    oracle = _oracle("hit_at_forty")
    record = _record(_undecided(oracle))
    new, names = oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION)  # the run only held size 10
    assert (new, names) == (None, [])
    # a setting that holds more than the case names is not the case either
    wider = _trials(10)
    wider["cells"][0]["cell"] = {"size": 10, "shape": "flat"}
    assert oc.rejudge_on_delivered(_record(_undecided(_oracle("hit_at_ten"))), wider, _protocol(_oracle("hit_at_ten")),
                                   VERSION) == (None, [])


def test_a_check_undecided_for_another_reason_is_not_touched() -> None:
    oracle = _oracle("hit_at_ten")
    for why in ("most_trials", "failed_trials"):
        record = _record(_undecided(oracle, why=why))
        assert oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION) == (None, [])


def test_code_that_changed_since_the_gate_leaves_the_check_undecided() -> None:
    oracle = _oracle("hit_at_ten")
    record = _record(_undecided(oracle, version="code-v0"))
    assert oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION) == (None, [])


def test_a_check_changed_since_the_gate_leaves_it_undecided() -> None:
    gate_oracle = _oracle("hit_at_ten")
    record = _record(_undecided(gate_oracle))
    assert oc.rejudge_on_delivered(record, _trials(10), _protocol(_oracle("hit_at_ten", expected=0.5001)),
                                   VERSION) == (None, [])


def test_a_delivered_run_with_no_more_trials_or_with_failed_trials_decides_nothing() -> None:
    oracle = _oracle("hit_at_ten")
    record = _record(_undecided(oracle))
    assert oc.rejudge_on_delivered(record, _trials(10, n=N_GATE), _protocol(oracle), VERSION) == (None, [])
    failing = {"cells": [_cell(10, failed=3)]}
    assert oc.rejudge_on_delivered(record, failing, _protocol(oracle), VERSION) == (None, [])


def test_a_measure_that_is_not_a_declared_mean_is_not_read_from_the_trials() -> None:
    oracle = _oracle("hit_at_ten")
    protocol = {**_protocol(oracle), "metrics": PROTOCOL["metrics"][1:]}  # "hit" is not declared
    record = _record(_undecided(oracle))
    assert oc.rejudge_on_delivered(record, _trials(10), protocol, VERSION) == (None, [])


def test_a_check_the_delivered_trials_still_cannot_decide_stays_undecided_and_the_others_are_judged() -> None:
    close, clear = _oracle("hit_close"), _oracle("gain_at_ten")
    record = _record(_undecided(close), _undecided(clear))
    new, names = oc.rejudge_on_delivered(record, _trials(10), _protocol(close, clear), VERSION)
    assert names == ["gain_at_ten"]
    assert _judged(new, "hit_close").get("undecided") is True and _judged(new, "hit_close")["mc"]["why"] == "time"
    assert _judged(new, "gain_at_ten")["passed_by_engine"] is True
    assert new["status"] == ac.WENT_ON and [e["name"] for e in new["went_on"]] == ["hit_close"]
    assert "gain_at_ten" not in new["summary"] and "hit_close" in new["summary"]
    assert "gain_at_ten" not in ac.failing_disclosure(new) and "hit_close" in ac.failing_disclosure(new)
    assert list(new["set_aside"]) == ["hit_close"]


def test_a_record_that_went_on_with_nothing_or_stopped_is_left_alone() -> None:
    oracle = _oracle("hit_at_ten")
    for status in ("ok", "stopped", "warned"):
        record = {**_record(_undecided(oracle)), "status": status}
        assert oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION) == (None, [])
    assert oc.rejudge_on_delivered(None, _trials(10), _protocol(oracle), VERSION) == (None, [])
    assert oc.rejudge_on_delivered(_record(_undecided(oracle)), None, _protocol(oracle), VERSION) == (None, [])


def test_a_delivered_run_with_a_longer_list_compared_inside_each_run_stays_undecided() -> None:
    oracle = _oracle("hit_at_ten")
    gate = {**PROTOCOL, "in_each_run": {"level": [1, 2]}}
    record = _record(_undecided(oracle, basis=oc.trial_basis(gate)))
    # the same list: judged again (the control that shows the test can see a re-judgment)
    new, names = oc.rejudge_on_delivered(record, _trials(10), gate, VERSION)
    assert names == ["hit_at_ten"]
    # the person extended the list for the main run: the same setting is not the same trials
    extended = {**PROTOCOL, "in_each_run": {"level": [1, 2, 3]}}
    assert oc.rejudge_on_delivered(record, _trials(10), extended, VERSION) == (None, [])


def test_an_entry_the_gate_wrote_without_the_trial_basis_is_never_judged_again() -> None:
    oracle = _oracle("hit_at_ten")
    judged, entry = _undecided(oracle)
    entry.pop("trial_basis")
    assert oc.rejudge_on_delivered(_record((judged, entry)), _trials(10), _protocol(oracle), VERSION) == (None, [])


def test_the_check_keeps_only_the_delivered_measurement() -> None:
    oracle = _oracle("hit_at_ten")
    record = _record(_undecided(oracle))
    record["attempts"][-1]["checks"][0]["returned"] = {"hit": 0.48, "gain": 0.19, "n_trials": 100.0}
    new, _ = oc.rejudge_on_delivered(record, _trials(10), _protocol(oracle), VERSION)
    check = new["attempts"][-1]["checks"][0]
    cell = _cell(10)
    assert check["returned"]["n_trials"] == float(N_DELIVERED)
    assert abs(check["returned"]["hit"] - sum(cell["metrics"]["hit"]["values"]) / N_DELIVERED) < 1e-12
    assert check["returned"]["hit"] != 0.48 and check["value"] == _judged(new, "hit_at_ten")["value"]


def test_the_main_run_is_read_again_whenever_it_finished_and_no_script_is_sent_back() -> None:
    from core.engine import _rejudges_oracles_after as after

    for status in ("ok", "warned", "extended_by_person", "off", "not_applicable"):
        assert after(status, None) is True, status
        assert after(status, "simulate.py") is False, status  # a script sent back: the run is not the delivered one
    for status in ("repairing", "stopped", "pending", "not_run", "single_script", "", None):
        assert after(status, None) is False, status


def test_the_engine_rewrites_the_record_after_the_main_run_and_frees_the_check(tmp_path: Any) -> None:
    import asyncio
    import json
    import logging
    import types

    from core.engine import Engine

    oracle = _oracle("hit_at_ten")
    root = tmp_path / "quest"
    (root / "needs").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "raw" / "trials.json").write_text(json.dumps(_trials(10)), encoding="utf-8")
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(_record(_undecided(oracle))), encoding="utf-8")
    written: list[dict[str, Any]] = []
    stub = types.SimpleNamespace(
        quest_root=root, _log=logging.getLogger("test"), _oracle_undecided={"hit_at_ten"},
        _oracle_set_aside={"hit_at_ten": "not decided"}, _protocol_block=lambda _state: _protocol(oracle),
        _measuring_code_version=lambda _seed: VERSION,
        _oracle_record_read=lambda: json.loads((root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8")),
        _oracle_record=lambda payload: written.append(payload))
    names = asyncio.run(Engine._rejudge_oracles_on_delivered_run(stub, {}, root / "code" / "simulate.py"))
    assert names == ["hit_at_ten"] and written and written[0]["status"] == "ok"
    assert not stub._oracle_undecided and not stub._oracle_set_aside
    # code that changed since the gate: the record is not rewritten
    stub._measuring_code_version = lambda _seed: "code-v2"
    written.clear()
    assert asyncio.run(Engine._rejudge_oracles_on_delivered_run(stub, {}, root / "code" / "simulate.py")) == []
    assert not written


def test_the_node_that_runs_the_study_makes_the_call_through_that_condition() -> None:
    import inspect

    from core.engine import Engine

    source = inspect.getsource(Engine._node_execute_steps)
    assert "_rejudges_oracles_after(manifest_status, failed_script)" in source
    assert "_rejudge_oracles_on_delivered_run(state, seed_path)" in source
