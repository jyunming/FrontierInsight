"""Replay of a real stop: a study whose trials measured an edge that never formed in many of them (the value is NaN there),
and whose protocol declares means the trials never return under their own name (a quantity per setting, a time). The
run-manifest check counted real numbers against every successful trial and asked the analysis, twice, for lists that could
not exist; the quest stopped with no paper. The evidence is the per-setting counts of the real record of that quest
(tests/fixtures/replay/undefined_trials/). No model is called."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from core import metric_spec
from core import run_manifest as rm

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "undefined_trials"
RECORD = json.loads((FIXTURE / "record_counts.json").read_text(encoding="utf-8"))
STUDY = json.loads((FIXTURE / "study.json").read_text(encoding="utf-8"))
METRICS, WHY = metric_spec.normalize(yaml.safe_load((FIXTURE / "protocol_metrics.yaml").read_text(encoding="utf-8"))["metrics"])
assert METRICS is not None, WHY
PROTOCOL: dict[str, Any] = {"grid": STUDY["grid"], "runs_per_setting": STUDY["runs_per_setting"], "metrics": METRICS}
MANIFEST: dict[str, Any] = {
    "successful_per_cell": {c["key"]: c["ok"] for c in RECORD["cells"]},
    "attempted_per_cell": {c["key"]: c["planned"] for c in RECORD["cells"]},
}


def _recorded() -> dict[str, Counter]:
    """What FI's record holds per name run_trial returned: a count per value (every defined value distinct), 'nan' for none."""
    out: dict[str, Counter] = {}
    n = 0
    for cell in RECORD["cells"]:
        for name, (defined, undefined) in cell["metrics"].items():
            counts = out.setdefault(name, Counter())
            for _ in range(defined):
                n += 1
                counts[f"{n}.5"] += 1
            counts["nan"] += undefined
    return out


def _analysis(*, nulls: bool = True, undefined_count: bool = False, keep: float = 1.0) -> dict[str, Any]:
    """The RESULT_JSON the analysis printed: per-setting lists of the metrics the trials returned, the rest as numbers."""
    per_setting: dict[str, Any] = {}
    for cell in RECORD["cells"]:
        entry: dict[str, Any] = {}
        for name in ("ler", "lwr"):
            if name not in cell["metrics"]:
                continue
            defined, undefined = cell["metrics"][name]
            values: list[Any] = [1.0] * int(defined * keep)
            if nulls:
                values += [None] * undefined
            entry[f"{name}_values"] = values
            if undefined_count:
                entry[f"{name}_undefined"] = undefined
        per_setting[cell["key"]] = entry
    derived = {name: 1.0 for name in ("dose_to_size", "tail_uncertainty", "ler_shot_noise", "lwr_shot_noise",
                                      "abs_photon_variance", "runtime")}
    return {"per_setting": per_setting, **derived}


def test_the_evidence_is_what_the_stop_says() -> None:
    stop = json.loads((FIXTURE / "stuck_excerpt.json").read_text(encoding="utf-8"))
    assert "only has 16989 real per-trial value(s)" in stop["problem"]
    assert any("`dose_to_size_values` list at all" in t for t in stop["tried"])
    ler = [sum(c["metrics"]["ler"][0] for c in RECORD["cells"] if "ler" in c["metrics"])]
    assert ler == [16989]


def test_before_the_fix_the_check_asked_for_lists_that_cannot_exist() -> None:
    found = rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(), trial_mode=True)  # no record: the old reading
    text = " | ".join(s for s, _ in found)
    assert "`ler` only has 16989 real per-trial value(s)" in text and "`dose_to_size_values` list at all" in text


def test_defined_plus_undefined_is_accepted_and_the_derived_means_are_not_asked_for_trial_lists() -> None:
    recorded = _recorded()
    assert rm.recorded_counts(recorded, "ler") == (16989, 23011)
    assert set(rm.derived_metrics(PROTOCOL, recorded)) == {
        "dose_to_size", "tail_uncertainty", "ler_shot_noise", "lwr_shot_noise", "abs_photon_variance", "runtime"}
    assert rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(), trial_mode=True, recorded=recorded) == []
    # The undefined ones may also be given as a count.
    assert rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(nulls=False, undefined_count=True),
                                    trial_mode=True, recorded=recorded) == []


def test_undefined_trials_are_never_hidden() -> None:
    recorded = _recorded()
    found = rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(nulls=False), trial_mode=True, recorded=recorded)
    assert found and all(analysis for _, analysis in found), "the analysis's to fix, so experiment.py is sent back"
    text = found[0][0]
    assert "16989 with a value and 23011 with none" in text and "16989 value(s) + 0 undefined" in text
    assert "`ler_undefined`" in text
    assert set(rm.missing_keys([s for s, _ in found])) >= {"ler_undefined"}
    asked = rm.analysis_directive([s for s, _ in found])
    assert "ler_undefined" in asked and "FI_TRIALS" in asked and "`null`" in asked and "experiment.py (not simulate.py)" in asked


def test_a_thin_list_is_still_a_mismatch() -> None:
    recorded = _recorded()
    found = rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(nulls=False, undefined_count=True, keep=0.1),
                                     trial_mode=True, recorded=recorded)
    assert found and "accounts for" in found[0][0]


def test_an_undefined_count_above_FIs_record_is_flagged() -> None:
    recorded = _recorded()
    analysis = _analysis(nulls=False, undefined_count=True)
    first = next(iter(analysis["per_setting"].values()))
    first["ler_undefined"] += 1_000_000  # the honest count plus a million that no trial returned
    found = rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=recorded)
    assert found and all(analysis_fix for _, analysis_fix in found), "the analysis's to fix"
    assert "reports more than that" in found[0][0] and "undefined where FI's record has 23011" in found[0][0]
    # the same lie as a bare top-level count beside the honest nulls
    analysis = _analysis()
    analysis["ler_undefined"] = 1_000_000
    assert rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=recorded)


def test_a_values_list_longer_than_the_trials_with_a_value_is_flagged() -> None:
    recorded = _recorded()
    found = rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(keep=1.5), trial_mode=True, recorded=recorded)
    assert found and "reports more than that" in found[0][0] and "value(s) where FI's record has 16989" in found[0][0]
    # one extra value anywhere is above the record: counts are integers, there is no slack above it
    analysis = _analysis()
    next(iter(analysis["per_setting"].values()))["ler_values"].append(1.0)
    assert rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=recorded)


def test_the_honest_report_is_not_flagged_by_the_upper_bound() -> None:
    recorded = _recorded()
    assert rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(), trial_mode=True, recorded=recorded) == []
    assert rm._value_count_findings(PROTOCOL, MANIFEST, _analysis(nulls=False, undefined_count=True),
                                    trial_mode=True, recorded=recorded) == []
