"""Replay of a real stop: the pilot read 28 trials at 22 settings and found nothing wrong, then the full run (120 settings)
gave proportions that were exactly 0 or exactly 1 in every trial, so nothing could be measured. The evidence is the pilot's own
trials from the real per-trial record (tests/fixtures/replay/pilot_stuck_proportions/). A proportion at a bound in every
pilot trial is only a finding when at least two different proportions are, because the protocol has no structured field that
tells a rare-event probability (0 is its expected pilot outcome) from any other. Names are read from the record.

Trials that did not change are an observation, not a fault, so a stuck proportion is sent back only with evidence from the
simulation's own code that its value is built from no setting (core/flat_output.py). The recorded trials are real; the original
simulation text was not kept, so these tests give it a stand-in that returns the proportions without reading the setting, and
show that the same trials from code that does read the setting are not sent back."""
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


def _ignoring(record: dict[str, Any]) -> str:
    """A simulation text that returns every metric of the record without reading the setting it is given."""
    ids = ", ".join(f'"{m["id"]}": 0.5' for m in record["metrics"])
    return "def run_trial(cell, trial, seed):\n    return {" + ids + "}\n"


def _reading(record: dict[str, Any]) -> str:
    """The same names, each made from the setting."""
    axis = next(iter(record["grid"]))
    ids = ", ".join(f'"{m["id"]}": cell["{axis}"] * 0.0' for m in record["metrics"])
    return "def run_trial(cell, trial, seed):\n    return {" + ids + "}\n"


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
    pilot = est.pilot_problems(_observations(record), _kinds(record), source=_ignoring(record))
    stuck = [p for p in pilot.problems if p.kind == "stuck"]
    props = set(_proportions(record))
    assert len(stuck) >= 2 and {p.metric for p in stuck} <= props
    first = stuck[0]
    assert f"the proportion `{first.metric}` was exactly" in first.says and "pilot trials, at" in first.says
    assert "each gave the same outcome in every trial" in first.says
    assert est.directive(pilot).startswith("FI ran 44 trials of the simulation at 22 settings")
    assert "expected" not in est.directive(pilot).lower()


def test_the_same_trials_are_not_sent_back_without_evidence_from_the_code() -> None:
    """Values that stayed at a bound are an observation: no code to read, code that reads the setting, or code that cannot be
    followed with certainty are none of them a fault."""
    record = _record()
    obs, kinds = _observations(record), _kinds(record)
    assert est.pilot_problems(obs, kinds).problems == []
    assert est.pilot_problems(obs, kinds, source=_reading(record)).problems == []
    unclear = "def run_trial(cell, trial, seed):\n    return helper(cell)\n"
    assert est.pilot_problems(obs, kinds, source=unclear).problems == []
    assert est.pilot_problems(obs, kinds, source="this is not python (").problems == []


def test_one_proportion_at_a_bound_is_never_a_finding_and_nor_is_an_expected_constant() -> None:
    record = _record()
    obs = _observations(record)
    kinds = _kinds(record)
    source = _ignoring(record)
    stuck = [p.metric for p in est.pilot_problems(obs, kinds, source=source).problems if p.kind == "stuck"]
    # leave only one of them stuck at each bound: the rare event `_sampled` keeps at 0, and one other at 1
    one = _sampled(record)
    rare = next(m for m in _proportions(record) if any(m in t for c in one["cells"] for t in c["trials"]))
    other = next(m for m in stuck if m != rare)
    for c in one["cells"]:
        for t in c["trials"]:
            if other in t:
                t[other] = 1.0
    assert [p for p in est.pilot_problems(_observations(one), kinds, source=source).problems if p.kind == "stuck"] == []
    # all stuck, but the plan expects them all to stay put
    exempt = est.pilot_problems(obs, kinds, exempt=set(stuck), source=source)
    assert [p for p in exempt.problems if p.kind == "stuck"] == []


def test_an_honest_record_with_a_rare_event_at_zero_and_other_proportions_in_range_passes() -> None:
    record = _sampled(_record())
    pilot = est.pilot_problems(_observations(record), _kinds(record), source=_ignoring(record))
    assert [p for p in pilot.problems if p.kind == "stuck"] == []


def test_proportions_at_opposite_bounds_that_only_flip_between_settings_are_not_a_finding_on_their_own() -> None:
    """A sharp but honest transition sampled by a few settings looks the same, so this is deliberately not a rule."""
    cells = [{"a": a} for a in (1, 2, 3, 4)]
    obs = []
    for c in cells:
        high = float(c["a"] >= 3)
        for t in range(2):
            obs.append((dict(c), {"x": float(c["a"]) + 0.1 * t, "p": high, "q": 1.0 - high}))
    src = 'def run_trial(cell, trial, seed):\n    return {"x": 1.0, "p": 1.0, "q": 0.0}\n'
    pilot = est.pilot_problems(obs, {"x": "mean", "p": "proportion", "q": "proportion"}, source=src)
    assert pilot.problems == []



def test_a_proportion_at_one_beside_its_complement_at_zero_is_not_a_finding() -> None:
    """A deterministic study: a success fraction always 1 and its failure fraction always 0 are two proportions, but
    at opposite bounds; only proportions stuck at the SAME bound count together."""
    obs = []
    for a in (1, 2, 3):
        for t in range(2):
            obs.append(({"a": a}, {"x": float(a) + 0.1 * t, "ok": 1.0, "fail": 0.0}))
    src = 'def run_trial(cell, trial, seed):\n    return {"x": cell["a"], "ok": 1.0, "fail": 0.0, "fail2": 0.0}\n'
    pilot = est.pilot_problems(obs, {"x": "mean", "ok": "proportion", "fail": "proportion"}, source=src)
    assert [p for p in pilot.problems if p.kind == "stuck"] == []
    # two at the same bound still are
    obs2 = [(c, {**v, "fail2": 0.0}) for c, v in obs]
    pilot2 = est.pilot_problems(obs2, {"x": "mean", "ok": "proportion", "fail": "proportion", "fail2": "proportion"},
                                source=src)
    assert sorted(p.metric for p in pilot2.problems if p.kind == "stuck") == ["fail", "fail2"]

@pytest.mark.asyncio
async def test_the_engine_sends_the_real_record_back_and_lets_the_sampled_one_go_on(tmp_path: Path) -> None:
    record = _record()
    eng, runner, ex = _engine_and_runner(tmp_path / "a", record)
    (eng.quest_root / "code" / "simulate.py").write_text(_ignoring(record), encoding="utf-8")
    patch = await eng._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner, "python", None)
    assert patch is not None and "the proportion `" in patch["exec_result"]["stderr_tail"]
    assert patch["exec_result"]["failed_script"] == "simulate.py" and ex.calls <= 30
    assert est.read_pilot_record(eng.quest_root)["outcome"] == "repairing"
    eng2, runner2, _ex2 = _engine_and_runner(tmp_path / "b", _sampled(record))
    (eng2.quest_root / "code" / "simulate.py").write_text(_ignoring(record), encoding="utf-8")
    assert await eng2._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner2, "python", None) is None
    assert est.read_pilot_record(eng2.quest_root)["outcome"] == "passed"
    # The real trials from code that reads the setting are not sent back either: they are an observation.
    eng3, runner3, _ex3 = _engine_and_runner(tmp_path / "c", record)
    (eng3.quest_root / "code" / "simulate.py").write_text(_reading(record), encoding="utf-8")
    assert await eng3._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner3, "python", None) is None
    assert est.read_pilot_record(eng3.quest_root)["outcome"] == "passed"
