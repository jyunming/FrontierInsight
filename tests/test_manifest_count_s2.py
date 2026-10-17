"""FI's count of what the analysis reports, replayed on the s2 false-discovery quest, which stopped on it.

Grid m x pi1 (9 settings, 10,000 runs each), three procedures in each run (``protocol.in_each_run``). FI's record holds
270,000 ``power`` values per procedure and setting: 180,000 with a value and 90,000 with none, and those 90,000 are every
trial of the three pi1 = 0 settings (no real effect, so no power), not one of them with a value. The analysis lists the
180,000 values under ``by_setting/m=..,pi1=..,procedure=..`` and, at pi1 = 0, gives ``power: null`` and no list. FI asked
three times for a ``power_undefined`` count or the 90,000 nulls and stopped. Where a metric has no value in any trial of a
setting there is nothing to drop: the analysis may leave that setting's list out.

``tests/fixtures/run_size/s2_fdr/manifest_check.json``: the analysis's RESULT_JSON (the final experiment.py run on FI's
stored trials) with each per-trial list replaced by its counts, FI's per-setting record as counts, the protocol and the
manifest.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from core import run_manifest
from tests.test_manifest_count_r14b import _lists

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "run_size" / "s2_fdr" / "manifest_check.json"


def _inputs() -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    per_cell = {key: {name: Counter({0.5: d, math.nan: u}) for name, (d, u) in by.items()}
                for key, by in fx["recorded_by_cell"].items()}
    recorded: dict[str, Counter] = {}
    for by in per_cell.values():
        for name, c in by.items():
            recorded.setdefault(name, Counter()).update(c)
    return fx["protocol"], {"successful_per_cell": fx["successful_per_cell"]}, _lists(fx["result_json"]), recorded, per_cell


def _found(rj: Any) -> list[str]:
    protocol, manifest, _rj, recorded, per_cell = _inputs()
    return [s for s, _ in run_manifest._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=recorded,
                                                             recorded_by_cell=per_cell)]


def test_fis_record_has_no_power_in_any_trial_at_pi1_0() -> None:
    _protocol, _manifest, _rj, recorded, per_cell = _inputs()
    assert run_manifest.recorded_counts(recorded, "power") == (180000, 90000)
    nowhere = [k for k, by in per_cell.items() if run_manifest.recorded_counts(by, "power") == (0, 10000)]
    assert len(nowhere) == 9 and all("pi1=0.0" in k for k in nowhere)


def test_the_s2_analysis_passes_the_count_check() -> None:
    """Before: "the analysis accounts for 180000 value(s) + 0 undefined = 180000, and RESULT_JSON has no `power_undefined`
    count and no null entries for them"."""
    _protocol, _manifest, rj, _recorded, _per_cell = _inputs()
    assert _found(rj) == []


def test_undefined_trials_at_a_setting_that_has_values_must_still_be_reported() -> None:
    protocol, manifest, rj, _recorded, per_cell = _inputs()
    key = "m=20,pi1=0.05,procedure=bonferroni"
    by = per_cell[key]["power"]
    by[math.nan] += 100  # 100 trials of that setting returned no power, and the analysis does not say so
    by[0.5] -= 100
    entry = rj["by_setting"][key]
    entry["power_values"] = entry["power_values"][:-100]
    recorded: dict[str, Counter] = {}
    for b in per_cell.values():
        for name, c in b.items():
            recorded.setdefault(name, Counter()).update(c)
    found = [s for s, _ in run_manifest._value_count_findings(protocol, manifest, rj, trial_mode=True, recorded=recorded,
                                                              recorded_by_cell=per_cell)]
    assert any("`power`" in s and "no `power_undefined` count" in s for s in found), found


def test_the_nulls_of_a_setting_with_no_value_still_count_when_listed() -> None:
    _protocol, _manifest, rj, _recorded, _per_cell = _inputs()
    for key in ("m=20,pi1=0.0,procedure=uncorrected", "m=200,pi1=0.0,procedure=bonferroni"):
        rj["by_setting"][key]["power_values"] = [None] * 10000
    assert _found(rj) == []


def test_a_list_left_out_where_values_exist_is_still_caught() -> None:
    _protocol, _manifest, rj, _recorded, _per_cell = _inputs()
    # The pi1 = 0.2 settings have power in every trial; the analysis drops their lists everywhere it printed them.
    for key, entry in rj["by_setting"].items():
        if "pi1=0.2" in key:
            entry.pop("power_values", None)
    for key, cell in rj["cells"].items():
        if "pi1=0.2" in key:
            for entry in cell["by_procedure"].values():
                entry.pop("power_values", None)
    rj.pop("power_values", None)
    found = _found(rj)
    assert any("`power`" in s and "accounts for 90000 value(s)" in s for s in found), found
