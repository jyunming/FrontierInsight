"""Replay of a real stop: the pilot read 28 trials at 22 settings and found nothing wrong, then the full run (120 settings)
gave proportions that were exactly 0 or exactly 1 in every trial, so nothing could be measured. The evidence is the pilot's own
trials from the real per-trial record (tests/fixtures/replay/pilot_stuck_proportions/). A proportion at a bound in every
pilot trial is only a finding when at least two different proportions are, because the protocol has no structured field that
tells a rare-event probability (0 is its expected pilot outcome) from any other. Names are read from the record."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from core import run_estimate as est
from tests.test_replay_pilot_undefined import Answers, _engine_and_runner, _kinds, _observations

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "pilot_stuck_proportions"


def _record() -> dict[str, Any]:
    return json.loads((FIXTURE / "pilot_trials.json").read_text(encoding="utf-8"))


def _proportions(record: dict[str, Any]) -> list[str]:
    return [m["id"] for m in record["metrics"] if m["kind"] == "proportion"]


def _sampled(record: dict[str, Any]) -> dict[str, Any]:
    """The same record where one proportion is a rare event that stays 0 and the others are sampled (0 or 1 by trial)."""
    out = copy.deepcopy(record)
    props = _proportions(record)
    rare = next(m for m in props if any(m in t for c in out["cells"] for t in c["trials"]))
    for c in out["cells"]:
        for i, trial in enumerate(c["trials"]):
            for m in props:
                if m in trial:
                    trial[m] = 0.0 if m == rare else float(i % 2)
    return out


def test_the_evidence_is_what_the_run_log_says() -> None:
    assert "the pilot read 28 trial(s) at 22 setting(s)" in (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")
    record = _record()
    assert len(record["cells"]) == 22


def test_the_real_pilot_trials_are_sent_back_with_each_stuck_proportion_named() -> None:
    record = _record()
    pilot = est.pilot_problems(_observations(record), _kinds(record))
    stuck = [p for p in pilot.problems if p.kind == "stuck"]
    props = set(_proportions(record))
    assert len(stuck) >= 2 and {p.metric for p in stuck} <= props
    first = stuck[0]
    assert f"the proportion `{first.metric}` was exactly" in first.says and "pilot trials, at" in first.says
    assert "each gave the same outcome in every trial" in first.says
    assert est.directive(pilot).startswith("FI ran 44 trials of the simulation at 22 settings")
    assert "expected" not in est.directive(pilot).lower()


def test_one_proportion_at_a_bound_is_never_a_finding_and_nor_is_an_expected_constant() -> None:
    record = _record()
    obs = _observations(record)
    kinds = _kinds(record)
    stuck = [p.metric for p in est.pilot_problems(obs, kinds).problems if p.kind == "stuck"]
    # leave only one of them stuck: the others are sampled
    one = _sampled(record)
    for c in one["cells"]:
        for t in c["trials"]:
            if stuck[0] in t:
                t[stuck[0]] = 0.0
    assert [p for p in est.pilot_problems(_observations(one), kinds).problems if p.kind == "stuck"] == []
    # all stuck, but the plan expects them all to stay put
    exempt = est.pilot_problems(obs, kinds, exempt=set(stuck))
    assert [p for p in exempt.problems if p.kind == "stuck"] == []


def test_an_honest_record_with_a_rare_event_at_zero_and_other_proportions_in_range_passes() -> None:
    record = _sampled(_record())
    pilot = est.pilot_problems(_observations(record), _kinds(record))
    assert [p for p in pilot.problems if p.kind == "stuck"] == []


def test_proportions_at_opposite_bounds_that_only_flip_between_settings_are_not_a_finding_on_their_own() -> None:
    """A sharp but honest transition sampled by a few settings looks the same, so this is deliberately not a rule."""
    cells = [{"a": a} for a in (1, 2, 3, 4)]
    obs = []
    for c in cells:
        high = float(c["a"] >= 3)
        for t in range(2):
            obs.append((dict(c), {"x": float(c["a"]) + 0.1 * t, "p": high, "q": 1.0 - high}))
    pilot = est.pilot_problems(obs, {"x": "mean", "p": "proportion", "q": "proportion"})
    assert pilot.problems == []


@pytest.mark.asyncio
async def test_the_engine_sends_the_real_record_back_and_lets_the_sampled_one_go_on(tmp_path: Path) -> None:
    record = _record()
    eng, runner, ex = _engine_and_runner(tmp_path / "a", record)
    patch = await eng._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner, "python", None)
    assert patch is not None and "the proportion `" in patch["exec_result"]["stderr_tail"]
    assert patch["exec_result"]["failed_script"] == "simulate.py" and ex.calls <= 30
    assert est.read_pilot_record(eng.quest_root)["outcome"] == "repairing"
    eng2, runner2, _ex2 = _engine_and_runner(tmp_path / "b", _sampled(record))
    assert await eng2._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner2, "python", None) is None
    assert est.read_pilot_record(eng2.quest_root)["outcome"] == "passed"
