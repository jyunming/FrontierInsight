"""The speed goal of the improve loop (core/improve_speed.py, ``RunChecksMixin._speed_up``) and the two small switches of the
correctness loop (retry after a round not kept; a check that now meets its bar is better). A fake model answers; FI runs
the sample itself on this interpreter. No real model is called."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import frozen_protocol as fp, improve, improve_speed as sp, plan
from core import run_estimate as est
from core import trial_runner as tr
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine

SLOW = '''import numpy as np


def run_trial(cell, trial_id, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=2000)
    total = 0.0
    for v in x:
        total += v * v
    return {"mean_square": total / len(x), "scale": float(cell["s"])}
'''
FAST = ("    total = float(np.sum(x * x))\n")
LOOP = "    total = 0.0\n    for v in x:\n        total += v * v\n"
CHEAT = "    x = x[:200]\n    total = 0.0\n    for v in x:\n        total += v * v\n"
ANALYSIS = "print('RESULT_JSON: {}')\n"


class _Model:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001
        self.prompts.append(messages[-1]["content"])
        assert messages[-1]["content"].startswith("# Make the Simulation Faster")
        return self.replies.pop(0) if self.replies else "{}"

    async def aclose(self) -> None:
        return None


def _edit(find: str, replace: str) -> str:
    return json.dumps({"file": "simulate.py", "find": find, "replace": replace, "why": "test"})


def _engine(tmp_path: Path, rounds: int = 3) -> Engine:
    cfg = Config(
        topic="sum of squares", title="speed", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=1, pilot_run=False, improve_rounds=rounds),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False, review="off"),
    )
    engine = Engine(cfg)
    root = engine.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_bytes(SLOW.encode("utf-8"))
    (root / "code" / "experiment.py").write_bytes(ANALYSIS.encode("utf-8"))
    protocol = plan.normalize_protocol({"grid": {"s": [1, 2, 3]}, "runs_per_setting": 50, "oracles": [], "criteria": [],
                                        "metrics": [{"id": "ms", "kind": "mean", "estimand": "m", "unit": "run"}]})[0]
    fp.freeze(root, protocol, approved_by="human: test", source="plan.md")
    engine._trial_mode = True
    return engine


async def _speed(engine: Engine, model: _Model) -> dict[str, Any]:
    engine._client = model
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=False)
    grid = {"s": [1, 2, 3]}
    big = SimpleNamespace(seconds=10.0)
    return await engine._speed_up({}, runner, engine.executor.python_path(root), None, grid, 50, {}, {}, big, 10.0)


def test_the_sample_must_return_the_same_numbers_and_take_clearly_less_time() -> None:
    a = sp.Sample(seconds=1.0, values={"k": [{"x": 1.0, "n": {"y": 2.0}}]})
    same = sp.Sample(seconds=0.5, values={"k": [{"x": 1.0 + 1e-12, "n": {"y": 2.0}}]})
    off = sp.Sample(seconds=0.5, values={"k": [{"x": 1.001, "n": {"y": 2.0}}]})
    gone = sp.Sample(seconds=0.5, values={"k": [{"x": 1.0}]})
    assert sp.disagreements(a, same) == []
    assert len(sp.disagreements(a, off)) == 1 and "x" in sp.disagreements(a, off)[0]
    assert sp.disagreements(a, gone) and "only in one" in sp.disagreements(a, gone)[0]
    assert sp.fast_enough(1.0, 0.7) and not sp.fast_enough(1.0, 0.8) and not sp.fast_enough(0.0, 0.0)


@pytest.mark.asyncio
async def test_a_change_that_alters_a_number_is_not_kept_and_one_that_keeps_them_and_is_faster_is_kept(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    model = _Model([_edit(LOOP, CHEAT), _edit(LOOP, FAST)])
    out = await _speed(engine, model)
    text = (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert out["rounds"][0]["outcome"] == "not kept" and "number(s) of the sample changed" in out["rounds"][0]["reason"]
    assert out["rounds"][1]["outcome"] == "kept" and out["kept"] is True and out["calls"] == 2
    assert "np.sum(x * x)" in text and "x[:200]" not in text
    assert out["speedup"] > 1.5


@pytest.mark.asyncio
async def test_a_change_that_is_not_faster_is_put_back(tmp_path: Path) -> None:
    engine = _engine(tmp_path, rounds=1)
    out = await _speed(engine, _Model([_edit("    total = 0.0\n", "    total = 0.0 + 0.0\n")]))
    assert out["kept"] is False and out["rounds"][0]["outcome"] == "not kept"
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SLOW


def _row(name: str, value: float, met: bool, *, target: float = 0.0, tolerance: float = 1e-3, **more: Any) -> dict[str, Any]:
    return {"name": name, "value": value, "met": met, "direction": "lower", "target": target, "tolerance": tolerance,
            "counts": True, **more}


def test_a_check_that_now_meets_its_bar_is_better_however_small_the_step() -> None:
    before = [_row("c", 1.1e-16, False, tolerance=1e-12)]
    after = [_row("c", 0.0, True, tolerance=1e-12)]
    assert improve.compare(before, after)[0]["change"] == "better"
    assert improve.compare(after, before)[0]["change"] == "worse"  # and one that stops meeting it is still worse
    assert improve.verdict(improve.compare(before, after))["best"] is True


def test_the_loop_does_not_start_for_a_flagged_check_or_one_already_within_its_tolerance() -> None:
    flagged = _row("a", 0.0, False, why=improve.FLAGGED_WHY + " that the code is flagged (x)")
    noise = _row("b", 0.0012, False, tolerance=0.0025)
    assert improve.only_noise([flagged]) and improve.only_noise([noise]) and improve.only_noise([flagged, noise])
    # a check that fails by more than its tolerance, one with no bar, or a mix that holds one of them, still starts it
    assert improve.only_noise([_row("c", 0.5, False)]) == ""
    assert improve.only_noise([noise, _row("c", 0.5, False)]) == ""
    assert improve.only_noise([{**_row("d", 3.0, False), "target": None}]) == ""
    assert improve.only_noise([_row("e", 0.0, True)]) == ""
    assert improve.only_noise([_row("f", 5.0, False, direction="higher", target=5.001, tolerance=0.01)])


SECOND = """def loop_count(p):
    return sum(1 for x in p if x < 0.5)


def run_trial(cell, trial_id, seed):
    p = [0.1, 0.9]
    out = {}
    out["R"] = len([x for x in p if x < 0.5])
    out["R_loop"] = float(loop_count(p))
    return out
"""


def test_a_speed_up_may_not_make_an_independent_check_read_the_other_side() -> None:
    oracle = {"name": "agree", "kind": "second_implementation", "expected": 0, "case": {"m": 20},
              "measure": "abs(R - R_loop)"}
    expected, measures, settings = improve.guards({}, [oracle], [], set())
    assert {"R", "R_loop"} <= measures and settings == {"m": [20]}
    files = {"simulate.py": SECOND}
    bend = improve.Edit(file="simulate.py", find='out["R_loop"] = float(loop_count(p))',
                        replace='out["R_loop"] = float(out["R"])', why="faster")
    new, why = improve.check_edit(files, bend, ["simulate.py"], tried=set(), expected=expected, measures=measures,
                                  settings=settings)
    assert new is None and "worked out where the simulation returns it" in why
    honest = improve.Edit(file="simulate.py", find="    return sum(1 for x in p if x < 0.5)",
                          replace="    return len([x for x in p if x < 0.5])", why="faster")
    new, why = improve.check_edit(files, honest, ["simulate.py"], tried=set(), expected=expected, measures=measures,
                                  settings=settings)
    assert new is not None, why


def test_estimate_module_still_exposes_what_the_speed_loop_uses() -> None:
    assert callable(est.probe_cells) and callable(est.plain_duration) and est.SAFETY > 1


@pytest.mark.asyncio
async def test_a_quest_whose_unmet_check_is_inside_its_own_tolerance_makes_no_change_and_says_why(tmp_path: Path) -> None:
    from tests import test_improve as ti

    noisy = {**ti.C_ERR, "tolerance": 0.05}  # Euler's error at dt=0.1 is about 0.018: not met, but inside 0.05 of the bar
    engine, model = await ti._quest(tmp_path, [ti._edit(ti.EULER, ti.RK4)], criteria=[noisy])
    out = await engine._node_improve(dict(ti.STATE))
    assert out == {"improve_rerun": False} and model.prompts == []
    assert "already within its own tolerance of its bar" in ti._log(engine)
    assert ti.EULER in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


def _probe(cell: dict, seconds: float, at_least: bool = False, axis: Any = "s") -> est.Probe:
    return est.Probe(axis, 0, dict(cell), seconds, at_least)


def test_the_sample_takes_only_settings_the_timing_saw_finish_and_fits_the_time_for_timing() -> None:
    """A live quest's speed step gave every sampled setting the whole study's limit (8 h), including the base setting the
    timing had not seen finish: one setting of the sample ran for half an hour and the step could take hours."""
    cells = [{"s": 2}, {"s": 1}, {"s": 3}]
    probes = [_probe({"s": 2}, 1000.0, at_least=True, axis=None), _probe({"s": 1}, 2.0), _probe({"s": 3}, 500.0)]
    kept, timeouts, why = sp.bounded_sample(cells, probes, runs=50, deterministic=False, budget_s=600.0, samples=4)
    # the base did not finish when timed (left out); s=3 costs 12/50 * 500 = 120 s, over 600 / 4 samples / 2 repeats
    assert kept == [{"s": 1}] and why == ""
    assert timeouts == {tr.cell_key({"s": 1}): 60}  # three times its 0.48 s, at least a minute
    # a setting the timing never ran is costed as the base; a known-answer case's setting is one of these
    more, more_t, _ = sp.bounded_sample([{"s": 1}, {"s": 9}], [_probe({"s": 1}, 50.0, axis=None)], runs=50,
                                        deterministic=False, budget_s=10_000.0, samples=1)
    assert more == [{"s": 1}, {"s": 9}] and more_t[tr.cell_key({"s": 9})] == 60
    # nothing fits, or nothing finished: the step is skipped and says why
    none, _, why = sp.bounded_sample(cells, probes, runs=50, deterministic=False, budget_s=1.0, samples=4)
    assert none == [] and "more than the" in why
    unseen, _, why = sp.bounded_sample(cells, [_probe({"s": 2}, 9.0, at_least=True, axis=None)], runs=50,
                                       deterministic=False, budget_s=600.0, samples=4)
    assert unseen == [] and "none of the sample's settings finish" in why


@pytest.mark.asyncio
async def test_a_setting_the_timing_could_not_finish_is_not_sampled(tmp_path: Path) -> None:
    import time

    engine = _engine(tmp_path, rounds=1)
    slow = SLOW.replace("    rng = np.random.default_rng(seed)\n",
                        "    rng = np.random.default_rng(seed)\n    if cell['s'] == 2:\n        import time; time.sleep(60)\n")
    (engine.quest_root / "code" / "simulate.py").write_bytes(slow.encode("utf-8"))
    engine._client = _Model(["{}"])  # no edit: only the sample before any change runs (a kept edit times the study again)
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=False)
    probes = [_probe({"s": 2}, 300.0, at_least=True, axis=None), _probe({"s": 1}, 1.0), _probe({"s": 3}, 1.0)]
    started = time.monotonic()
    record = await engine._speed_up({}, runner, engine.executor.python_path(root), None, {"s": [1, 2, 3]}, 50, {}, {},
                                    SimpleNamespace(seconds=10.0), 10.0, budget_s=600.0, probes=probes)
    assert time.monotonic() - started < 50, "the setting that sleeps a minute was sampled"
    assert record["rounds"] and record["rounds"][0]["outcome"] == "refused"  # the sample ran, and the model was asked
