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


def _recorded_by_cell() -> dict[str, dict[str, Counter]]:
    """The same record per setting (the cell key of each setting), as FI keeps it."""
    out: dict[str, dict[str, Counter]] = {}
    n = 0
    for cell in RECORD["cells"]:
        for name, (defined, undefined) in cell["metrics"].items():
            counts = out.setdefault(cell["key"], {}).setdefault(name, Counter())
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


def _findings(analysis: dict[str, Any]) -> list[tuple[str, bool]]:
    return rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=_recorded(),
                                    recorded_by_cell=_recorded_by_cell())


def _pooled_copy(analysis: dict[str, Any], keep: float = 1.0) -> None:
    """The same lists once more, pooled over the settings, beside the per-setting ones."""
    for name in ("ler", "lwr"):
        values: list[Any] = []
        for entry in analysis["per_setting"].values():
            values += entry.get(f"{name}_values", [])
        analysis[f"{name}_values"] = values[: int(len(values) * keep)] if keep < 1 else values


def test_per_setting_lists_beside_an_honest_pooled_copy_are_not_flagged() -> None:
    analysis = _analysis()
    _pooled_copy(analysis)
    assert _findings(analysis) == []
    # and without the per-setting record the totals alone agree too
    assert rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=_recorded()) == []


def test_a_half_length_list_copied_to_reach_the_bound_is_still_flagged() -> None:
    analysis = _analysis(keep=0.5)
    assert _findings(analysis), "half of the values alone is short"
    copy = {k: dict(v) for k, v in analysis["per_setting"].items()}
    analysis["per_setting_again"] = copy  # the same half once more, under another name
    _pooled_copy(analysis, keep=0.5)  # and a pooled copy of the half
    found = _findings(analysis)
    assert found and "accounts for" in found[0][0], found
    assert all(analysis_fix for _, analysis_fix in found)
    # the same setting listed twice inside one report counts once
    twice = _analysis(keep=0.5)
    twice["per_setting"] = {**twice["per_setting"], **{f"{k} ": v for k, v in twice["per_setting"].items()}}
    assert _findings(twice)


def test_an_inflated_pooled_list_is_flagged_beside_honest_per_setting_ones() -> None:
    analysis = _analysis()
    _pooled_copy(analysis)
    analysis["ler_values"] = analysis["ler_values"] + [1.0] * 5000
    found = _findings(analysis)
    assert found and "reports more than that" in found[0][0], found
    # an inflated list for one setting is held to that setting's own trials
    analysis = _analysis()
    first = next(iter(analysis["per_setting"]))
    analysis["per_setting"][first]["ler_values"] = analysis["per_setting"][first]["ler_values"] + [1.0]
    found = _findings(analysis)
    assert found and f"`{first}`" in found[0][0]


def _keyed_by_index(analysis: dict[str, Any]) -> dict[str, Any]:
    """The same per-setting report under keys the grid cannot read as settings ("0", "1", ...)."""
    return {**{k: v for k, v in analysis.items() if k != "per_setting"},
            "by_setting": {str(i): entry for i, entry in enumerate(analysis["per_setting"].values())}}


def test_an_honest_report_keyed_by_index_is_not_taken_for_its_largest_part() -> None:
    analysis = _keyed_by_index(_analysis())
    assert _findings(analysis) == []
    assert _findings(_keyed_by_index(_analysis(nulls=False, undefined_count=True))) == []
    # with an honest pooled copy beside it, too
    pooled = _analysis()
    _pooled_copy(pooled)
    assert _findings(_keyed_by_index(pooled)) == []


def test_values_given_as_a_mapping_of_lists_are_counted() -> None:
    base = _analysis(nulls=False, undefined_count=True)
    analysis: dict[str, Any] = {k: v for k, v in base.items() if k != "per_setting"}
    for name in ("ler", "lwr"):
        analysis[f"{name}_values"] = {str(i): e[f"{name}_values"] for i, e in enumerate(base["per_setting"].values())
                                      if f"{name}_values" in e}
        analysis[f"{name}_undefined"] = {str(i): e[f"{name}_undefined"]
                                         for i, e in enumerate(base["per_setting"].values()) if f"{name}_undefined" in e}
    assert _findings(analysis) == []
    # the same mapping under the settings' own keys is the per-setting copy, and is held to each setting's trials
    keyed = {k: v for k, v in base.items() if k != "per_setting"}
    for name in ("ler", "lwr"):
        keyed[f"{name}_values"] = {k: e[f"{name}_values"] for k, e in base["per_setting"].items()
                                   if f"{name}_values" in e}
        keyed[f"{name}_undefined"] = {k: e[f"{name}_undefined"] for k, e in base["per_setting"].items()
                                      if f"{name}_undefined" in e}
    assert _findings(keyed) == []


def test_a_short_report_keyed_by_index_is_still_flagged() -> None:
    found = _findings(_keyed_by_index(_analysis(keep=0.5)))
    assert found and "accounts for" in found[0][0], found
    # one index's list longer than the whole record is still above it
    analysis = _keyed_by_index(_analysis())
    analysis["by_setting"]["0"]["ler_values"] = [1.0] * 20000
    found = _findings(analysis)
    assert found and "reports more than that" in found[0][0], found
