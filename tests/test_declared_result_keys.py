"""A declared result a rerun of the study compared and found unchanged is not left out for the likeness of its name to one
that moved: a fault shown in ``x_spread`` whose rerun moves only ``x_spread`` leaves out ``x_spread``, not ``x`` whose name it
begins with. Without a rerun the names still group them (an uncertainty worked out from a wrong value inside one trial has
no other record that ties it to the value). Neutral made-up study (the fixtures of tests/test_shown_reach.py with one more
declared result). No model is called."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review as cr
from core.rerun_check import Verdict
from tests import test_shown_reach as base


def test_a_result_shown_apart_is_not_the_key_of_another_whose_name_begins_with_it() -> None:
    result = {"drift_rate": 0.12, "drift_rate_stderr": 0.01, "drift_rate_ci95": [0.1, 0.14]}
    # without a record that sets them apart the names alone decide (as before)
    assert cr.named_in_results("drift_rate_stderr", result) == {"drift_rate", "drift_rate_stderr", "drift_rate_ci95"}
    assert cr.named_in_results("drift_rate", result) == {"drift_rate", "drift_rate_stderr", "drift_rate_ci95"}
    assert cr.named_in_results("drift_rate_stderr", result, ["drift_rate"]) == {"drift_rate_stderr"}
    assert cr.named_in_results("drift_rate", result, ["drift_rate_stderr"]) == {"drift_rate", "drift_rate_ci95"}
    assert cr.named_in_results("drift_rate", result, ["drift_rate", "", None]) == {"drift_rate", "drift_rate_stderr", "drift_rate_ci95"}


SPREAD = {"id": "corr_ab_spread", "kind": "mean", "estimand": "spread of the correlation", "unit": "one setting", "role": "result",
          "returned": "afterwards"}
DERIVE_SPREAD = "import derive\n\n\ndef test_x():\n    assert derive.derive_corr_ab_spread(None) == 0.9, 'wrong'\n"


@pytest.fixture
def spread_study(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = copy.deepcopy(base.PROTOCOL)
    protocol["metrics"].append(SPREAD)
    monkeypatch.setattr(base, "PROTOCOL", protocol)
    monkeypatch.setattr(base, "RESULT", {**base.RESULT, "derived": {"corr_ab": 0.42, "corr_ab_spread": 0.03}})
    files = {**base.FILES, "derive.py": base.FILES["derive.py"] + "\n\ndef derive_corr_ab_spread(table):\n    return 0.03\n"}
    monkeypatch.setattr(base, "FILES", files)
    made = base._quest

    def quest(engine, files=None):  # type: ignore[no-untyped-def]
        root = made(engine, files)
        path = root / "data" / "results" / "derived.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["derived"]["corr_ab_spread"] = {"status": "ok", "by": "derive_corr_ab_spread", "inputs": [], "value": 0.03}
        path.write_text(json.dumps(record), encoding="utf-8")
        return root

    monkeypatch.setattr(base, "_quest", quest)


def _rerun_moves(engine: Any, metrics: list[str]) -> None:
    async def moved(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {str(i.get("id")): Verdict("changed", metrics=list(metrics), keys=[f"derived/derived/{m}" for m in metrics],
                                          mode="analysis only") for i, _how in verdicts}

    engine._rerun_shown = moved  # this one engine only


def test_a_rerun_that_moves_one_result_leaves_out_that_one_not_the_one_its_name_begins_with(tmp_path: Path, spread_study: None) -> None:
    """Old code: the gate left out ``corr_ab`` with ``corr_ab_spread`` (``corr_ab_spread`` begins with ``corr_ab``), though
    the rerun showed only ``corr_ab_spread`` moved."""
    engine = base._engine(tmp_path)
    _rerun_moves(engine, ["corr_ab_spread"])
    out, record = base._run(engine, DERIVE_SPREAD, quantity="corr_ab_spread")
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and gate["keys"] == ["corr_ab_spread"]
    removed = base._removed(engine)["removed_quantities"]
    assert "corr_ab_spread" in removed and "corr_ab" not in removed


def test_without_a_rerun_the_names_still_group_the_two(tmp_path: Path, spread_study: None) -> None:
    """Known positive of the old rule: nothing measured sets them apart, so both go out."""
    engine = base._engine(tmp_path)
    out, record = base._run(engine, DERIVE_SPREAD, quantity="corr_ab_spread")
    assert out is None
    assert "corr_ab" in record["issues"]["F-1"]["at_gate"]["keys"]


def test_a_rerun_that_moves_both_leaves_out_both(tmp_path: Path, spread_study: None) -> None:
    engine = base._engine(tmp_path)
    _rerun_moves(engine, ["corr_ab", "corr_ab_spread"])
    out, record = base._run(engine, base.DERIVE_CORR, quantity="corr_ab")
    assert out is None
    keys = record["issues"]["F-1"]["at_gate"]["keys"]
    assert "corr_ab" in keys and "corr_ab_spread" in keys


def _rerun_says(engine: Any, verdict: Verdict) -> None:
    async def said(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {str(i.get("id")): verdict for i, _how in verdicts}

    engine._rerun_shown = said


@pytest.mark.parametrize("verdict", [
    Verdict("could_not", "the repaired study gave no number"),
    Verdict("changed", metrics=["corr_ab_spread"], keys=["derived/derived/corr_ab_spread"], mode="analysis only",
            unwritten=["data/results/other.json"]),
    Verdict("changed", metrics=["corr_ab_spread", "failed trials"], keys=["derived/derived/corr_ab_spread"], mode="whole study"),
], ids=["could_not", "unwritten", "failed_trials"])
def test_a_rerun_that_did_not_compare_everything_sets_nothing_apart(tmp_path: Path, spread_study: None, verdict: Verdict) -> None:
    engine = base._engine(tmp_path)
    _rerun_says(engine, verdict)
    out, record = base._run(engine, DERIVE_SPREAD, quantity="corr_ab_spread")
    assert out is None
    assert "corr_ab" in record["issues"]["F-1"]["at_gate"]["keys"]


def test_an_analysis_only_rerun_never_shows_a_trial_result_unmoved(tmp_path: Path) -> None:
    """A rerun of the analysis alone reads the recorded trials: a result a trial returns cannot move in it."""
    from core import shown_reach

    engine = base._engine(tmp_path)
    root = base._quest(engine)
    reach = shown_reach.Reach(root, base.PROTOCOL, ["est_a"])
    assert "est_a" in reach.unmoved(["corr_ab"]) and "cover" in reach.unmoved(["corr_ab"])
    assert "est_a" not in reach.unmoved(["corr_ab"], analysis_only=True)
    assert "cover" in reach.unmoved(["corr_ab"], analysis_only=True)


def test_a_reader_claim_whose_rerun_moves_one_result_leaves_out_that_one_not_the_one_its_name_begins_with(
        tmp_path: Path, spread_study: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same rule for a second model's claim that a result is computed differently: a test shows it, the rerun with a
    repair moves only ``corr_ab_spread``, so ``corr_ab`` stays in. Old code: both went out."""
    import asyncio

    from core import frozen_protocol, shown_reach

    engine = base._engine(tmp_path)
    root = base._quest(engine)
    check = {"kind": "code_reader", "note": "x", "quantity": "corr_ab_spread", "keys": ["corr_ab_spread"]}
    rc = {"removed_quantities": ["corr_ab_spread"], "removed_figures": [], "removed_values": {}, "checks": [check],
          "note": "x", "say": "x"}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")
    monkeypatch.setattr(cr, "differing_quantities", lambda *a, **k: [
        {"name": "corr_ab_spread", "fid": "F-9", "kind": "dependent variable", "status": "different", "note": "x"}])

    async def demonstrate(issue: Any, record: Any, state: Any) -> bool:
        issue.update(demonstrated=True, test=".fi/finding_tests/F-9_1.py")
        return True

    async def answers(verdicts: Any, record: Any, state: Any, strict: bool = False) -> dict[str, Any]:
        return {str(i["id"]): Verdict("changed", metrics=["corr_ab_spread"], keys=["derived/derived/corr_ab_spread"],
                                      mode="analysis only") for i, _h in verdicts}

    engine._demonstrate_reader_finding = demonstrate
    engine._rerun_shown = answers
    record = {"issues": {"F-9": {"id": "F-9", "file": "derive.py", "function": "", "quantity": "corr_ab_spread",
                                 "kind": "quantity", "changes_result": True, "demonstrated": None, "outcome": "open",
                                 "text": "computed another way", "severity": "high"}}}
    reach = shown_reach.Reach(root, frozen_protocol.protocol_of(root), ["est_a"], [])
    state = {"design": None, "result_json": base.RESULT, "exec_result": {"returncode": 0}}
    assert asyncio.run(engine._measure_reader_findings(record, state, reach, ["est_a"])) is None
    removed = json.loads((root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))["removed_quantities"]
    assert "corr_ab_spread" in removed and "corr_ab" not in removed
