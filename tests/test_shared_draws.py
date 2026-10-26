"""Settings share random draws only when the plan asks for it, and an interval over settings that did share them is
not built from all their trials as if they were independent repeats."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from core import code_project, mc_error, trial_runner
from core import paper_values as pv
from core import stats as _stats

METRIC = {"id": "x", "kind": "mean", "estimand": "the mean of x", "unit": "one run", "paired": True}


def _protocol(shared: bool | None = None, policy: str | None = None) -> dict:
    p: dict = {"grid": {"n": [5, 12]}, "runs_per_setting": 4, "metrics": [dict(METRIC)]}
    if shared is not None:
        p["shared_draws"] = shared
    if policy is not None:
        p["seed_policy"] = policy
    return p


def _study(tmp_path: Path, protocol: dict) -> dict:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "simulate.py").write_text("def run_trial(setting, trial, seed):\n    return {}\n", encoding="utf-8")
    study = code_project.study_of(code, protocol)
    assert study is not None
    return study


def _seeds(study: dict) -> dict[str, list[int]]:
    base = study["base_seed"]
    return {key: [trial_runner.trial_seed(base, "" if study["paired"] else key, t) for t in range(4)]
            for key in ("n=5", "n=12")}


def test_a_paired_metric_alone_gives_each_setting_its_own_seeds(tmp_path):
    # A metric that pairs two estimates of one trial's data says nothing about the draws of another setting.
    for protocol in (_protocol(), _protocol(False), _protocol(None, "independent stream per (setting, trial)")):
        study = _study(tmp_path, protocol)
        seeds = _seeds(study)
        assert study["paired"] is False and not mc_error.paired(protocol)
        assert not set(seeds["n=5"]) & set(seeds["n=12"])


def test_the_words_of_the_seed_policy_alone_never_decide_it(tmp_path):
    for words in ("common random numbers across settings", "CRN", "the same random numbers across arms",
                  "no common random numbers"):
        protocol = _protocol(None, words)
        assert _study(tmp_path, protocol)["paired"] is False and not mc_error.paired(protocol)
    # and a policy of independent streams does not undo what the plan states
    assert mc_error.paired(_protocol(True, "independent streams"))


def test_an_explicit_request_for_common_draws_still_shares_them(tmp_path):
    study = _study(tmp_path, _protocol(True))
    seeds = _seeds(study)
    assert study["paired"] is True and mc_error.paired(_protocol(True))
    assert seeds["n=5"] == seeds["n=12"]
    # the field alone makes the settings share their draws; no metric needs to be marked paired
    assert mc_error.paired({**_protocol(True), "metrics": []})
    assert _study(tmp_path, {**_protocol(True), "metrics": []})["paired"] is True


def test_the_paired_contrast_of_the_analysis_follows_the_same_decision():
    from core import metric_spec

    assert [s["paired"] for s in metric_spec.declared(_protocol())] == [False]
    assert [s["paired"] for s in metric_spec.declared(_protocol(True))] == [True]


def test_a_quest_begun_before_the_field_keeps_the_seeds_it_recorded(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "study.json").write_text(json.dumps({"paired": True}), encoding="utf-8")
    assert mc_error.paired(_protocol(), tmp_path)             # an older plan: no field, recorded as shared
    assert not mc_error.paired(_protocol(False), tmp_path)    # a plan that states it is believed
    assert not mc_error.paired(_protocol())                   # no quest folder, nothing recorded
    # the analysis's paired contrast follows the very same answer (the seeds and the contrast never disagree)
    from core import metric_spec

    assert [m["paired"] for m in metric_spec.declared(_protocol(), tmp_path)] == [True]
    assert [m["paired"] for m in metric_spec.declared(_protocol(False), tmp_path)] == [False]
    assert [m["paired"] for m in metric_spec.declared(_protocol())] == [False]
    a = [float(i % 7) for i in range(40)]
    b = [x + 0.3 + (0.01 * (i % 3)) for i, x in enumerate(a)]
    reps = [{"_seed": k, "by_g": {"a": {"m_values": a}, "b": {"m_values": b}}} for k in range(2)]
    legacy = {"metrics": [{"id": "m", "kind": "mean", "paired": True, "estimand": "e", "unit": "u"}]}
    got = metric_spec.statistics(reps, legacy, quest_root=tmp_path)["contrasts"][0]["method"]
    assert got == "paired_permutation_bootstrap"                       # the old quest keeps its paired contrast
    assert metric_spec.statistics(reps, legacy)["contrasts"][0]["method"] == "bootstrap_permutation"
    # study_of keeps the recorded value when it writes the study again from such a plan
    (tmp_path / "code" / "simulate.py").write_text("def run_trial(setting, trial, seed):\n    return {}\n", encoding="utf-8")
    assert code_project.study_of(tmp_path / "code", _protocol())["paired"] is True


def _cells(shared: bool) -> tuple[list[dict], dict[str, dict[int, int]]]:
    rng = random.Random(3)
    base = [rng.gauss(0, 1) for _ in range(50)]
    cells, seeds = [], {}
    for i, key in enumerate(("n=5", "n=12", "n=20")):
        # settings that share draws move together: the same draws plus a small setting-specific part
        values = [b + (0.05 * rng.gauss(0, 1)) for b in base] if shared else [rng.gauss(0, 1) for _ in range(50)]
        cells.append({"cell": {"n": 5 + i}, "key": key, "planned": 50, "ok": 50, "failed": 0,
                      "metrics": {"x": {"values": values, "trials": list(range(50))}}})
        seeds[key] = {t: (1000 + t if shared else 1000 * (i + 1) + t) for t in range(50)}
    return cells, seeds


KEY = "x:mean_ci @ n=all"


def test_an_interval_over_settings_that_share_draws_is_not_the_naive_pooled_one():
    cells, seeds = _cells(shared=True)
    rec = pv.from_cells(cells)
    naive = pv.resolve(KEY, rec)
    rec.seeds = seeds
    got = pv.resolve(KEY, rec)
    assert got.filled and "50 different random draws" in got.text
    per_draw = np.mean([c["metrics"]["x"]["values"] for c in cells], axis=0)
    allv = np.concatenate([c["metrics"]["x"]["values"] for c in cells])
    half = _stats._t95(49) * np.std(per_draw, ddof=1) / math.sqrt(50)
    assert f"{per_draw.mean() - half:.2f} to {per_draw.mean() + half:.2f}" in got.text
    # the naive interval counts 150 independent trials; the settings' trials are copies of 50 draws
    naive_half = _stats._t95(149) * allv.std(ddof=1) / math.sqrt(150)
    assert half > 1.5 * naive_half and got.text != naive.text
    se = pv.resolve("x:se @ n=all", rec)
    assert math.isclose(float(se.text), np.std(per_draw, ddof=1) / math.sqrt(50), rel_tol=5e-3)


def test_settings_with_their_own_draws_are_pooled_as_before():
    cells, seeds = _cells(shared=False)
    rec = pv.from_cells(cells)
    before = pv.resolve(KEY, rec).text
    rec.seeds = seeds
    assert pv.resolve(KEY, rec).text == before and "share random" not in before


def test_every_way_of_loading_the_records_reads_the_seeds(tmp_path):
    cells, seeds = _cells(shared=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    rows = [{"event": "trial", "cell": k, "trial": t, "seed": sd, "status": "ok"} for k, per in seeds.items() for t, sd in per.items()]
    (tmp_path / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rec = pv.load(tmp_path)
    assert rec is not None and rec.share_draws(rec.cells)
    assert "different random draws" in pv.resolve(KEY, rec).text


# --- every reducer and interval form counts the draws, with the inputs the review used ----------------------------


def _record(per_setting: dict[str, dict[int, float]], seeds: dict[str, dict[int, int]], extra=None, metric="x"):
    cells = []
    for i, (key, vals) in enumerate(per_setting.items()):
        ids = sorted(vals)
        metrics = {metric: {"values": [vals[t] for t in ids], "trials": ids}}
        for name, other in (extra or {}).items():
            metrics[name] = {"values": [other[key][t] for t in ids], "trials": ids}
        cells.append({"cell": {"n": i + 1}, "key": key, "planned": len(ids), "ok": len(ids), "failed": 0, "metrics": metrics})
    rec = pv.from_cells(cells)
    rec.seeds = seeds
    return rec


def test_the_mean_over_settings_with_unequal_trials_is_the_mean_of_the_draws():
    # setting A ran draws 100 and 101, B only 101 (a trial failed): the draws' means are 0 and 10, so 5 (not 6.667)
    rec = _record({"a": {0: 0.0, 1: 10.0}, "b": {1: 10.0}}, {"a": {0: 100, 1: 101}, "b": {1: 101}})
    assert pv.resolve("x:mean @ n=all", rec).text == "5.000"


def test_sd_ci_and_corr_ci_count_the_draws_and_give_none_for_one_draw():
    seeds = {"a": {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105}, "b": {0: 100, 1: 101, 2: 102, 3: 103, 4: 104, 5: 105}}
    vals = {"a": {t: float(t * t % 7) for t in range(6)}, "b": {t: float(t * t % 7) + 0.1 for t in range(6)}}
    other = {k: {t: v + t for t, v in d.items()} for k, d in vals.items()}
    rec = _record(vals, seeds, {"y": other})
    shared = pv.resolve("x:sd_ci @ n=all", rec)
    rec.seeds = {}
    naive = pv.resolve("x:sd_ci @ n=all", rec)
    rec.seeds = seeds
    assert "6 different random draws" in shared.text and "different random draws" not in naive.text
    assert shared.numbers[2] - shared.numbers[1] > naive.numbers[2] - naive.numbers[1]   # fewer degrees of freedom: wider
    corr = pv.resolve("x:corr_ci:y @ n=all", rec)
    assert corr.filled and "6 different random draws" in corr.text    # 6 pairs, not 12 trials
    one = _record({"a": {0: 1.0, 1: 2.0}, "b": {0: 1.0, 1: 2.0}}, {"a": {0: 5, 1: 5}, "b": {0: 5, 1: 5}}, {"y": {"a": {0: 1.0, 1: 2.0}, "b": {0: 1.0, 1: 2.0}}})
    for key in ("x:mean_ci @ n=all", "x:se @ n=all", "x:sd_ci @ n=all", "x:corr_ci:y @ n=all"):
        assert not pv.resolve(key, one).filled, key


def test_count_where_uses_the_t_interval_over_draws_not_a_wilson_interval_of_all_trials():
    protocol = {"grid": {"n": [1, 2], "m": [1, 2]}, "thresholds": {"r": 0.75},
                "metrics": [{"id": "b", "reference": {"value": "b", "formula": "r"}}]}
    cells, seeds = [], {}
    for n in (1, 2):
        for m in (1, 2):
            key = f"n={n},m={m}"
            vals = [1.0] * 5 + [0.0] * 5
            cells.append({"cell": {"n": n, "m": m}, "key": key, "planned": 10, "ok": 10, "failed": 0,
                          "metrics": {"b": {"values": vals, "trials": list(range(10))}}})
            seeds[key] = {t: 100 * n + t for t in range(10)}      # the two m settings of one n share their 10 draws
    rec = pv.from_cells(cells, protocol)
    key = "b @ n=each, m=all | count_where:differs"
    naive = pv.resolve(key, rec)
    rec.seeds = seeds
    shared = pv.resolve(key, rec)
    assert naive.filled and shared.filled and naive.text != shared.text, (naive.text, shared.text)


def test_sd_ci_under_shared_draws_is_the_spread_of_the_draws_values():
    # the same two draws in both settings: per-draw values [0, 15], whose SD is 10.607 (the pooled raw SD is 11.18)
    seeds = {"a": {0: 100, 1: 101}, "b": {0: 100, 1: 101}}
    rec = _record({"a": {0: 0.0, 1: 10.0}, "b": {0: 0.0, 1: 20.0}}, seeds)
    got = pv.resolve("x:sd_ci @ n=all", rec)
    assert got.filled and "2 different random draws" in got.text
    assert math.isclose(got.numbers[0], 10.6066, rel_tol=1e-4), got.numbers


def test_values_are_paired_by_trial_number_never_by_position():
    seeds = {k: {0: 100, 1: 101, 2: 102} for k in ("a", "b")}
    cells = []
    for i, key in enumerate(("a", "b")):
        cells.append({"cell": {"n": i + 1}, "key": key, "planned": 3, "ok": 3, "failed": 0, "metrics": {
            "x": {"values": [0.0, 1.0, 2.0], "trials": [0, 1, 2]},
            "y": {"values": [2.2, 0.1, 1.3], "trials": [2, 0, 1]}}})
    rec = pv.from_cells(cells)
    rec.seeds = seeds
    got = pv.resolve("x:corr:y @ n=all", rec)
    # matched by trial number, y of trials 0, 1, 2 is 0.1, 1.3, 2.2: nearly a line with x; by position it would not be
    assert got.filled and got.numbers[0] > 0.99, got
    # numbers the record does not give: no pairing at all
    for c in cells:
        del c["metrics"]["y"]["trials"]
    assert not pv.resolve("x:corr_ci:y @ n=all", rec).filled
    assert not pv.resolve("x:corr:y @ n=all", rec).filled


def test_a_study_that_recorded_shared_seeds_keeps_them_even_with_no_metric_marked_paired(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "study.json").write_text(json.dumps({"paired": True}), encoding="utf-8")
    assert mc_error.paired({"grid": {"n": [1, 2]}}, tmp_path) is True
    assert mc_error.paired({"grid": {"n": [1, 2]}, "shared_draws": False}, tmp_path) is False
