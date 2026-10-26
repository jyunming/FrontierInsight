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
    # stating it without a paired metric changes nothing
    assert not mc_error.paired({**_protocol(True), "metrics": []})
    assert _study(tmp_path, {**_protocol(True), "metrics": []})["paired"] is False


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
