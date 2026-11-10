"""The speed goal of the improve loop (core/improve_speed.py, ``RunChecksMixin._speed_up``) and the two small switches of the
correctness loop (retry after a round not kept; a check that now meets its bar is better). A fake model answers; FI runs
the sample itself on this interpreter. No real model is called."""

from __future__ import annotations

import json
import re
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


#: The same simulation, which also reports a number that is different on every run of the same code.
NOISY = SLOW.replace("import numpy as np\n", "import time\n\nimport numpy as np\n", 1).replace(
    '"scale": float(cell["s"])}', '"scale": float(cell["s"]), "elapsed": time.perf_counter()}')


def _engine(tmp_path: Path, rounds: int = 3, source: str = SLOW) -> Engine:
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
    (root / "code" / "simulate.py").write_bytes(source.encode("utf-8"))
    (root / "code" / "experiment.py").write_bytes(ANALYSIS.encode("utf-8"))
    protocol = plan.normalize_protocol({"grid": {"s": [1, 2, 3]}, "runs_per_setting": 50, "oracles": [], "criteria": [],
                                        "metrics": [{"id": "ms", "kind": "mean", "estimand": "m", "unit": "run"}]})[0]
    fp.freeze(root, protocol, approved_by="human: test", source="plan.md")
    engine._trial_mode = True
    return engine


async def _speed(engine: Engine, model: _Model, seconds: float = 10.0) -> dict[str, Any]:
    """The speed-up on a study FI estimates at ``seconds`` (unpadded) for 10 s allowed: too slow to fit. A long one
    (``seconds`` in hours) lets the rounds go on, since a round then costs far less than a tenth of the study."""
    engine._client = model
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=False)
    grid = {"s": [1, 2, 3]}
    big = SimpleNamespace(seconds=seconds)
    return await engine._speed_up({}, runner, engine.executor.python_path(root), None, grid, 50, {}, {}, big, 10.0)


def test_the_sample_must_return_the_same_numbers_and_take_clearly_less_time() -> None:
    a = sp.Sample(seconds=1.0, values={"k": [{"x": 1.0, "n": {"y": 2.0}}]})
    same = sp.Sample(seconds=0.5, values={"k": [{"x": 1.0 + 1e-12, "n": {"y": 2.0}}]})
    off = sp.Sample(seconds=0.5, values={"k": [{"x": 1.001, "n": {"y": 2.0}}]})
    gone = sp.Sample(seconds=0.5, values={"k": [{"x": 1.0}]})
    assert sp.disagreements(a, same) == []
    assert len(sp.disagreements(a, off)) == 1 and "x" in sp.disagreements(a, off)[0]
    assert sp.disagreements(a, gone) and "only in one" in sp.disagreements(a, gone)[0]
    assert sp.fast_enough(1.0, 0.7) and not sp.fast_enough(0.0, 0.0)


def test_a_change_is_kept_when_it_saves_a_tenth_and_more_than_two_runs_of_the_same_code_differ() -> None:
    """The step kept a change only at a quarter less time, so a study that a change made 15% faster ran at its old speed.
    The rule is now a tenth, and more than the noise FI measured: the two samples of the unchanged code."""
    assert sp.fast_enough(1.0, 0.85) and sp.fast_enough(1.0, 0.89)
    assert not sp.fast_enough(1.0, 0.95)
    # a saving of 0.15 s is not a saving when two runs of the same code already differ by 0.2 s
    assert not sp.fast_enough(1.0, 0.85, 0.2) and sp.fast_enough(1.0, 0.85, 0.1)


def test_a_study_is_sped_up_when_fi_expects_it_to_take_ten_minutes_or_more_whether_it_fits_or_not() -> None:
    """A study that fit its time but took hours was never sped up: the step ran only when FI's estimate said it would not
    fit. The gate is now the padded whole-study estimate alone."""
    assert sp.worth_speeding_up(600.0) and sp.worth_speeding_up(5 * 3600.0)
    assert not sp.worth_speeding_up(599.0) and not sp.worth_speeding_up(None)
    # a study that does not fit is sped up before it is cut, however short
    assert sp.worth_speeding_up(30.0, fits=False) and not sp.worth_speeding_up(30.0, fits=True)
    assert sp.not_worth_it_line(300.0) == ("The study fits and is expected to take about 5 minutes, under 10 minutes, so FI "
                                           "does not try to make the simulation faster.")


@pytest.mark.asyncio
async def test_a_change_that_alters_a_number_is_not_kept_and_one_that_keeps_them_and_is_faster_is_kept(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    model = _Model([_edit(LOOP, CHEAT), _edit(LOOP, FAST)])
    out = await _speed(engine, model, seconds=10_000.0)
    text = (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert out["rounds"][0]["outcome"] == "not kept" and "number(s) of the sample changed" in out["rounds"][0]["reason"]
    # a change not kept does not end the step: the model is asked again, with the reason
    assert out["rounds"][1]["outcome"] == "kept" and out["kept"] is True and out["calls"] >= 2
    assert "number(s) of the sample changed" in model.prompts[1]
    assert "np.sum(x * x)" in text and "x[:200]" not in text
    assert out["speedup"] > 1.5


def test_numbers_left_out_are_exactly_the_ones_named_and_another_change_still_counts() -> None:
    a = sp.Sample(seconds=1.0, values={"k": [{"x": 1.0, "t": 5.0}, {"x": 1.0, "t": 6.0}]})
    b = sp.Sample(seconds=1.0, values={"k": [{"x": 1.0, "t": 7.0}, {"x": 1.5, "t": 8.0}]})
    both = {("k", 0, "/t"), ("k", 1, "/t")}
    assert len(sp.disagreements(a, b)) == 3
    assert sp.disagreements(a, b, ignore=both) == ["k trial 1 /x: 1 -> 1.5"]
    assert sp.varying(a, b) == both | {("k", 1, "/x")}
    assert sp.varying(a, a) == set() and sp.count_numbers(a) == 4


@pytest.mark.asyncio
async def test_a_number_that_changes_between_two_runs_of_the_same_code_does_not_stop_a_faster_edit_being_kept(tmp_path: Path) -> None:
    engine = _engine(tmp_path, source=NOISY)
    out = await _speed(engine, _Model([_edit(LOOP, FAST)]))
    assert out["rounds"][0]["outcome"] == "kept" and out["kept"] is True
    assert out["varying"] and all("elapsed" in v for v in out["varying"])
    assert "np.sum(x * x)" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_an_edit_that_changes_a_real_result_is_not_kept_though_a_varying_number_is_left_out(tmp_path: Path) -> None:
    engine = _engine(tmp_path, rounds=1, source=NOISY)
    out = await _speed(engine, _Model([_edit(LOOP, CHEAT)]))
    assert out["kept"] is False and out["rounds"][0]["outcome"] == "not kept"
    assert "mean_square" in out["rounds"][0]["reason"] and "elapsed" not in out["rounds"][0]["reason"]
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == NOISY


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
    engine, model = await ti._quest(tmp_path, [ti._edit(ti.FIRST_ORDER, ti.RK4)], criteria=[noisy])
    out = await engine._node_improve(dict(ti.STATE))
    assert out == {"improve_rerun": False} and model.prompts == []
    assert "already within its own tolerance of its bar" in ti._log(engine)
    assert ti.FIRST_ORDER in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


def _probe(cell: dict, seconds: float, at_least: bool = False, axis: Any = "s") -> est.Probe:
    return est.Probe(axis, 0, dict(cell), seconds, at_least)


def test_the_sample_takes_only_settings_the_timing_saw_finish_and_fits_the_time_for_timing() -> None:
    """A live quest's speed step gave every sampled setting the whole study's limit (8 h), including the base setting the
    timing had not seen finish: one setting of the sample ran for half an hour and the step could take hours."""
    cells = [{"s": 2}, {"s": 1}, {"s": 3}]
    probes = [_probe({"s": 2}, 1000.0, at_least=True, axis=None), _probe({"s": 1}, 2.0), _probe({"s": 3}, 500.0)]
    kept, timeouts, why = sp.bounded_sample(cells, probes, runs=50, deterministic=False, budget_s=600.0, samples=4)
    # the base did not finish when timed (left out); s=3 costs 12/50 * 500 = 120 s, over 600 / 4 samples / 2 repeats
    assert kept == [{"s": 1}]
    assert why == ("2 of the sample's 3 settings are left out (FI's timing did not see 1 finish; 1 would not fit in "
                   "the time left for timing)")
    assert timeouts == {tr.cell_key({"s": 1}): 60}  # three times its 0.48 s, at least a minute
    # a setting the timing never ran is not costed as another one (the step times it first, untimed_cells)
    more, more_t, more_why = sp.bounded_sample([{"s": 1}, {"s": 9}], [_probe({"s": 1}, 50.0, axis=None)], runs=50,
                                        deterministic=False, budget_s=10_000.0, samples=1)
    assert more == [{"s": 1}] and tr.cell_key({"s": 9}) not in more_t and "did not see 1 finish" in more_why
    assert sp.untimed_cells([{"s": 1}, {"s": 9}], [_probe({"s": 1}, 50.0, axis=None)]) == [{"s": 9}]
    # nothing fits, or nothing finished: the step is skipped and says why
    none, _, why = sp.bounded_sample(cells, probes, runs=50, deterministic=False, budget_s=1.0, samples=4)
    assert none == [] and "more than the" in why
    unseen, _, why = sp.bounded_sample(cells, [_probe({"s": 2}, 9.0, at_least=True, axis=None)], runs=50,
                                       deterministic=False, budget_s=600.0, samples=4)
    assert unseen == [] and "did not see any of the sample's settings finish" in why


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


@pytest.mark.asyncio
async def test_the_model_is_shown_where_the_time_goes_measured_by_fi(tmp_path: Path) -> None:
    """The model guessed from the code where a simulation spends its time; a real one spent nine tenths of it in one
    function called a hundred thousand times. FI measures it and shows the costliest functions with their call counts."""
    engine = _engine(tmp_path)
    model = _Model([_edit(LOOP, FAST)])
    out = await _speed(engine, model)
    assert out["profiled"] is True
    shown = model.prompts[0]
    assert "Where the time goes, measured by FI" in shown
    assert re.search(r"run_trial \(simulate\.py:\d+\): [\d.]+ s of its own \(\d+% of the setting\), called \d+ times", shown)


@pytest.mark.asyncio
async def test_a_refused_change_is_kept_with_its_refusal(tmp_path: Path) -> None:
    """A guard refused a real speed-up and only its reason was kept, so whether the refusal was right could not be judged."""
    engine = _engine(tmp_path)
    out = await _speed(engine, _Model([_edit("this text is not in the file", "x")]))
    entry = out["rounds"][0]
    assert entry["outcome"] == "refused" and "this text is not in the file" in entry["diff"]
    assert entry["file"] == "simulate.py"


@pytest.mark.asyncio
async def test_an_answer_that_cannot_be_read_is_kept_with_its_refusal(tmp_path: Path) -> None:
    """A live answer that rewrote a whole function was refused as "not the JSON object asked for", and nothing of it was
    kept: whether the answer or the reading was at fault could not be judged."""
    engine = _engine(tmp_path)
    out = await _speed(engine, _Model(["here is the new function: def f(x): return x"]))
    entry = out["rounds"][0]
    assert entry["outcome"] == "refused" and "here is the new function" in entry["reply"]


@pytest.mark.asyncio
async def test_an_answer_with_a_first_try_and_a_correction_is_read_as_the_correction(tmp_path: Path) -> None:
    """A live answer wrote one edit with a placeholder, then "Correction, the exact-text version:" and the edit it meant;
    the two objects were read as one and the answer was refused as "not the JSON object asked for"."""
    engine = _engine(tmp_path)
    first = json.loads(_edit("this text is not in the file", ""))
    first["why"] = "placeholder"
    reply = json.dumps(first) + "\n\nCorrection, the exact-text version:\n\n" + _edit("this text is not in the file", "x")
    out = await _speed(engine, _Model([reply]))
    entry = out["rounds"][0]
    assert "not the JSON object" not in entry["reason"], entry["reason"]
    assert "this text is not in the file" in entry["diff"] and entry["why"] != "placeholder"


@pytest.mark.asyncio
async def test_a_setting_the_timing_never_ran_is_timed_first_and_its_limit_is_its_own(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A live quest's only speed-up try never ran: a known-answer case's setting, finer than the base, was costed as the
    base (about 5 s) and given a minute; it did not finish, and the step stopped. Such a setting is timed on its own first,
    and its limit in the sample comes from that timing."""
    engine = _engine(tmp_path, rounds=1)
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=True)
    timed: list[dict[str, Any]] = []

    async def time_one(executor, python, quest_root, module, cell, **kw):  # noqa: ANN001, ANN003
        timed.append(dict(cell))
        return est.Probe(None, None, dict(cell), 100.0, fixed_s=100.0, per_trial_s=0.0, timed_trials=1), ""

    seen: dict[str, Any] = {}

    async def run_sample(*a: Any, **k: Any) -> sp.Sample:
        seen.update(cells=a[4], timeouts=k.get("timeouts"))
        return sp.Sample(failed="stop here")

    monkeypatch.setattr(est, "_time_one", time_one)
    monkeypatch.setattr(sp, "run_sample", run_sample)
    probes = [_probe({"s": 1}, 2.0, axis=None), _probe({"s": 2}, 2.0), _probe({"s": 3}, 2.0)]
    record = await engine._speed_up({}, runner, engine.executor.python_path(root), None, {"s": [1, 2, 3]}, 1, {}, {},
                                    SimpleNamespace(seconds=10_000.0), 100_000.0, budget_s=100_000.0, probes=probes,
                                    deterministic=True, extra_cells=[{"s": 9}])
    assert timed == [{"s": 9}], "only the setting the timing never ran is timed first"
    assert {"s": 9} in seen["cells"]
    assert seen["timeouts"][tr.cell_key({"s": 9})] == 300  # three times its own 100 s, not a minute from the base
    assert "stop here" in record["stopped"]


SOMETIMES_FAILS = SLOW.replace("    rng = np.random.default_rng(seed)\n",
                               "    rng = np.random.default_rng(seed)\n    if cell['s'] == 2:\n"
                               "        raise ValueError('no stable answer at this setting')\n")


@pytest.mark.asyncio
async def test_a_sample_that_does_not_run_says_why(tmp_path: Path) -> None:
    """A live quest's sample stopped at '0 of 1 trials ran at <setting>' and kept no reason: the trial's error and the
    time limit it reached were both dropped. The reason now reaches the record (and run.log, which says the same)."""
    engine = _engine(tmp_path, source=SOMETIMES_FAILS)
    out = await _speed(engine, _Model([]))
    assert "trials ran at s=2" in out["stopped"]
    assert "a trial failed" in out["stopped"] and "no stable answer at this setting" in out["stopped"]
    saved = json.loads((engine.quest_root / ".fi" / "improve_speed.json").read_text(encoding="utf-8"))
    assert saved["stopped"] == out["stopped"]
    slow = SLOW.replace("    rng = np.random.default_rng(seed)\n",
                        "    rng = np.random.default_rng(seed)\n    if cell['s'] == 2:\n        import time; time.sleep(30)\n")
    root = engine.quest_root
    (root / "code" / "simulate.py").write_bytes(slow.encode("utf-8"))
    sample = await sp.run_sample(engine.executor, engine.executor.python_path(root), root,
                                 "code/simulate.py", [{"s": 2}], deterministic=False, thresholds={}, in_each_run={},
                                 env=None, timeout_s=60, timeouts={tr.cell_key({"s": 2}): 3})
    assert "reached its time limit of 3 seconds" in sample.failed


@pytest.mark.asyncio
async def test_the_rounds_go_on_while_each_saves_a_tenth_even_when_the_study_already_fits(tmp_path: Path) -> None:
    """The rounds stopped as soon as the study would fit the time allowed; a study that already fits was never made
    faster. They now go on to the cap of engine.improve_rounds, a round not kept included."""
    engine = _engine(tmp_path, rounds=3)
    model = _Model([_edit(LOOP, FAST), _edit("this text is not in the file", "x")])
    engine._client = model
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=False)
    # FI expects about 4 h for a limit of 28 h: it fits, and it is long
    out = await engine._speed_up({}, runner, engine.executor.python_path(root), None, {"s": [1, 2, 3]}, 50, {}, {},
                                 SimpleNamespace(seconds=10_000.0), 100_000.0)
    assert out["rounds"][0]["outcome"] == "kept" and out["calls"] == 3
    assert out["rounds"][1]["outcome"] == "refused" and out["rounds"][2]["outcome"] == "refused"
    assert out["stopped"] == "all 3 round(s) of engine.improve_rounds were used"
    assert out["noise_s"] >= 0 and out["rounds"][0]["saves_on_study_s"] > 0
    assert "It fits, but it is long" in model.prompts[0] and "times faster to fit" not in model.prompts[0]


@pytest.mark.asyncio
async def test_the_rounds_stop_when_a_round_costs_more_time_than_it_saves_on_the_study(tmp_path: Path) -> None:
    """A round's wall time (the model's answer and the sample) is set against what it saves on the study: on a study of a
    second and a half, a kept change saves less than the round took, and the rounds stop there."""
    engine = _engine(tmp_path, rounds=3)
    engine._client = _Model([_edit(LOOP, FAST), _edit(LOOP, FAST)])
    root = engine.quest_root
    runner = SimpleNamespace(simulate=root / "code" / "simulate.py", deterministic=False)
    out = await engine._speed_up({}, runner, engine.executor.python_path(root), None, {"s": [1, 2, 3]}, 50, {}, {},
                                 SimpleNamespace(seconds=1.0), 100.0)
    assert out["rounds"][0]["outcome"] == "kept" and len(out["rounds"]) == 1
    assert "more than the" in out["stopped"] and "it saves on the study" in out["stopped"]


@pytest.mark.asyncio
async def test_a_refused_first_change_is_followed_by_a_second_ask_and_a_kept_second_change_is_kept(tmp_path: Path) -> None:
    """A round that is not kept (here a change whose text is not in the file) does not end the step: FI asks again with the
    reason, and a second change that keeps every number and is faster is kept."""
    engine = _engine(tmp_path, rounds=3)
    model = _Model([_edit("this text is not in the file", "x"), _edit(LOOP, FAST)])
    out = await _speed(engine, model, seconds=10_000.0)
    assert out["rounds"][0]["outcome"] == "refused"
    assert len(model.prompts) >= 2 and "this text is not in the file" in model.prompts[1]
    assert out["rounds"][1]["outcome"] == "kept" and out["kept"] is True
    assert "np.sum(x * x)" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
