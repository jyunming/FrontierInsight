"""FI's checks of the trials (core/trial_checks.py) on real quests, as they are: each quest's own frozen plan and its own
trial record, with no declaration added by hand.

tests/fixtures/replay/trial_checks_real/ holds them: ``raw/trials.json`` of each quest with its
``needs/FROZEN_PROTOCOL.json``, unchanged except that each value keeps only its first 40 trials (fdr) or 30 (kinetics, a
few values only); ``s2_kinetics_full`` keeps every trial of the two values the calibration reads. Built by reading the
quests' folders (``source`` in each file). Where the quests' own folders are on this machine, the same checks also run on
the full records (skipped elsewhere)."""

from __future__ import annotations

import glob
import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

from core import paper_facts
from core import trial_checks as tc

FIX = Path(__file__).parent / "fixtures" / "replay" / "trial_checks_real"
FDR = ["s1_fdr", "s2_fdr", "r14b_fdr", "r13_fdr", "E1_fdr"]
FULL = {
    "s1_fdr": r"C:\fi_runs\suite\rounds\s1\false_discovery\out\*",
    "s2_fdr": r"C:\fi_runs\suite\rounds\s2\false_discovery\out\*",
    "r14b_fdr": r"C:\fi_runs\suite\rounds\r14b\false_discovery\out\*",
    "r13_fdr": r"C:\fi_runs\suite\rounds\r13\false_discovery\out\*",
    "E1_fdr": r"C:\fi_compare\exp\E1\out\*",
}


def _load(name: str) -> dict[str, Any]:
    with gzip.open(FIX / f"{name}.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _full(name: str) -> dict[str, Any]:
    found = glob.glob(FULL[name])
    if not found:
        pytest.skip(f"the quest folder of {name} is not on this machine")
    root = Path(found[0])
    data = json.loads((root / "raw" / "trials.json").read_text(encoding="utf-8"))
    proto = json.loads((root / "needs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))["protocol"]
    return {"protocol": proto, "thresholds": data.get("thresholds") or {}, "cells": data["cells"],
            "reasons": tc.ledger_reasons(root)}


def _judge(d: dict[str, Any], **kw: Any) -> tc.Verdict:
    return tc.judge(d["cells"], d["protocol"], d["thresholds"] or None, d.get("reasons") or {},
                    repairs_left=kw.pop("repairs_left", False), **kw)


# ---- a value with no number at a setting is not a failed trial ---------------------------------------------------------


def _check_fdr(v: tc.Verdict) -> None:
    assert v.action == tc.OK, v.say
    assert not [f for f in v.fails if f.failed], [f.says() for f in v.fails if f.failed]
    rec = v.record()
    undefined = rec["undefined"]
    # power (or the power difference) has no value where there is no true alternative: pi1 = 0, in every trial there
    assert undefined and all(u["all"] for u in undefined), undefined
    assert all("pi1=0" in u["key"] for u in undefined), undefined


@pytest.mark.parametrize("name", FDR)
def test_a_correct_fdr_quest_does_not_stop_on_power_having_no_value_at_pi1_0(name: str) -> None:
    _check_fdr(_judge(_load(name)))
    # with repairs left, nothing is sent back to be repaired either
    assert _judge(_load(name), repairs_left=True).action == tc.OK


@pytest.mark.parametrize("name", FDR)
def test_the_full_record_of_a_correct_fdr_quest_does_not_stop(name: str) -> None:
    _check_fdr(_judge(_full(name)))


def _cells(values: dict[str, dict[str, list[float]]], cell_of: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, metrics in values.items():
        n = max(len(v) for v in metrics.values())
        out.append({"key": key, "cell": cell_of[key], "planned": n, "failed": 0,
                    "metrics": {m: {"values": v, "trials": list(range(len(v)))} for m, v in metrics.items()}})
    return out


def test_a_value_with_no_number_in_some_trials_is_counted_and_is_a_required_item(tmp_path: Path) -> None:
    nan = float("nan")
    cells = _cells({"d=0": {"power": [nan] * 20, "fdr": [0.04] * 20},
                    "d=1": {"power": [nan] * 3 + [0.5] * 17, "fdr": [0.04] * 20},
                    "d=2": {"power": [0.7] * 20, "fdr": [0.04] * 20}},
                   {"d=0": {"d": 0}, "d=1": {"d": 1}, "d=2": {"d": 2}})
    proto = {"metrics": [{"id": "power", "estimand": "power", "kind": "mean", "unit": "one trial"},
                         {"id": "fdr", "estimand": "fdr", "kind": "mean", "unit": "one trial"}]}
    v = tc.judge(cells, proto, None, {}, repairs_left=False)
    assert v.action == tc.OK and not [f for f in v.fails if f.failed]
    rec = v.record()
    assert rec["undefined"] == [{"key": "d=0", "metric": "power", "trials": 20, "ran": 20, "all": True},
                                {"key": "d=1", "metric": "power", "trials": 3, "ran": 20, "all": False}]
    root = tmp_path / "q"
    tc.write_record(root, rec)
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.UNDEFINED]
    assert [i.id for i in items] == ["undefined-power"]
    said = items[0].sentence
    assert "`power` has no value at d=0 (in every trial there)" in said
    assert "in 3 of 20 trials at d=1" in said
    # a number the plan says has no value there is not said again
    proto["metrics"][0]["undefined_when"] = {"d": [0]}
    rec = tc.judge(cells, proto, None, {}, repairs_left=False).record()
    assert [(u["key"], u["all"]) for u in rec["undefined"]] == [("d=1", False)]


def test_where_the_truth_is_declared_a_trial_with_no_estimate_is_a_miss_of_the_calibration() -> None:
    """The r12 repair that returned NaN in place of a failing fit: with the true value declared, the trials with no estimate
    are misses of the calibration, so the quest stops (it is not a value that has no meaning there)."""
    nan = float("nan")
    est = [1.0 + 0.01 * ((i % 7) - 3) for i in range(40)]
    lo, hi = [e - 0.05 for e in est], [e + 0.05 for e in est]
    bad = {"x_hat": [nan] * 30 + est[:10], "x_lo": [nan] * 30 + lo[:10], "x_hi": [nan] * 30 + hi[:10]}
    good = {"x_hat": est, "x_lo": lo, "x_hi": hi}
    cells = _cells({f"s={i}": (bad if i < 2 else good) for i in range(6)}, {f"s={i}": {"s": i} for i in range(6)})
    spec = {"id": "x", "estimand": "x", "kind": "mean", "unit": "one trial",
            "known_value": {"true_value": 1.0, "estimate": "x_hat", "interval": ["x_lo", "x_hi"], "level": 0.95}}
    v = tc.judge(cells, {"metrics": [spec]}, None, {}, repairs_left=False)
    assert v.action == tc.STOP and v.calibration.faults, v.say
    assert not v.record()["undefined"], "a value the truth is declared for is not 'undefined': it is a miss"
