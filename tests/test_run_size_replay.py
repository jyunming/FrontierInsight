"""The run's size against the topic's own time limit, replayed on the records of two real quests.

r12 and r13 false-discovery quests (``tests/fixtures/run_size``: their simulation code, frozen protocol, topic constraints,
FI's recorded estimate and a per-setting summary of FI's trial ledger) both asked for "under 60 seconds of runtime". FI
estimated about 2 and 0.5 minutes (with its margin) and the runs took 860 s and 104 s: the harness timed each trial with a
clock that moves in steps of about 16 ms on Windows, and read the cost of a setting off one later trial; and FI sized the run
to ``execution.timeout_s`` (15 minutes), not to the topic's 60 seconds.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import paper_facts, plan, run_estimate
from core import trial_runner as _tr
from core.execution import SharedInterpreterExecutor
from tests.test_run_checks import _log, _sizing

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "run_size"


def _fixture(name: str) -> dict[str, Any]:
    root = FIXTURES / name
    return {"root": root,
            "protocol": json.loads((root / "protocol.json").read_text(encoding="utf-8")),
            "ledger": json.loads((root / "ledger_summary.json").read_text(encoding="utf-8")),
            "recorded": json.loads((root / "run_estimate.recorded.json").read_text(encoding="utf-8")),
            "constraints": json.loads((root / "topic_requirements.json").read_text(encoding="utf-8"))["constraints"]}


#: The wall time of the first full run of the trials and of an analysis-only rerun (from each quest's run.log), seconds.
WALL = {"r12_fdr": (864.8, 11.2), "r13_fdr": (104.0, 6.2)}


def _ledger_probes(name: str) -> tuple[dict[str, list[Any]], int, list[run_estimate.Probe]]:
    """What the real run cost per setting, as probes FI would have made had it timed each setting exactly: each setting's
    process start (the run's wall time less its trials and its analysis, shared out) and its mean time per trial."""
    fx = _fixture(name)
    grid, runs = fx["protocol"]["grid"], int(fx["protocol"]["runs_per_setting"])
    cells = fx["ledger"]["cells"]
    wall, analysis = WALL[name]
    fixed = (wall - analysis - sum(c["trial_seconds"] for c in cells.values())) / len(cells)

    def probe(axis: Any, index: Any, cell: dict[str, Any]) -> run_estimate.Probe:
        c = cells[_tr.cell_key(cell)]
        per = c["trial_seconds"] / c["trials"]
        return run_estimate.Probe(axis, index, cell, fixed + runs * per, fixed_s=fixed, per_trial_s=per)

    probes = [probe(a, i, c) for a, i, c in run_estimate.probe_cells(grid)]
    costly = run_estimate.costly_ends(grid, probes)
    if len(costly) >= 2:
        probes.append(probe(run_estimate.CORNER, None, run_estimate.corner_cell(grid, costly)))
    return grid, runs, probes


def test_the_recorded_estimates_were_far_below_the_real_runs() -> None:
    for name in ("r12_fdr", "r13_fdr"):
        fx = _fixture(name)
        wall, _analysis = WALL[name]
        assert fx["recorded"]["estimate_s"] < wall / 3, name
        # The ledger's trial times show the coarse clock: most are exactly 0.
        zeros = sum(c["zero_times"] for c in fx["ledger"]["cells"].values())
        assert zeros > 0.5 * sum(c["trials"] for c in fx["ledger"]["cells"].values()), name


def test_the_real_costs_added_up_give_the_real_run_time() -> None:
    """The estimate's arithmetic was right; its inputs were not. With each setting's real cost it gives the real run."""
    for name in ("r12_fdr", "r13_fdr"):
        grid, runs, probes = _ledger_probes(name)
        est = run_estimate.estimate(grid, runs, False, probes)
        wall, analysis = WALL[name]
        assert est is not None and 0.75 * (wall - analysis) < est.seconds < 1.5 * (wall - analysis), (name, est.seconds)


def test_r12_cannot_meet_60_seconds_at_any_runs_and_keeps_the_plans_minimum() -> None:
    grid, runs, probes = _ledger_probes("r12_fdr")
    protocol = _fixture("r12_fdr")["protocol"]
    # Its 27 settings each start a process (about 1.7 s in the real run): with FI's margin that alone is over 60 s.
    assert run_estimate.fit_runs(grid, runs, False, probes, protocol, limit_s=60.0, topic_limit=True) is None
    # The plan's own fewest: its minimum is 1000, its precision target (±0.03 on a proportion) needs 1068.
    assert run_estimate.runs_floor(protocol, runs) == 1068
    at_floor = run_estimate.seconds_at(grid, 1068, runs, False, probes)
    assert at_floor is not None and at_floor * run_estimate.SAFETY < 900.0 < 860.0 * run_estimate.SAFETY


def test_r13_fits_60_seconds_with_fewer_runs_than_the_plans_minimum() -> None:
    grid, runs, probes = _ledger_probes("r13_fdr")
    protocol = _fixture("r13_fdr")["protocol"]
    fit = run_estimate.fit_runs(grid, runs, False, probes, protocol, limit_s=60.0, topic_limit=True)
    assert fit is not None and fit.below_plan and fit.plan_floor == 4096
    assert 300 < fit.runs < 900, fit.runs
    assert fit.seconds * run_estimate.SAFETY <= 60.0
    # Never below the plan for FI's own limit: 15 minutes holds the whole plan.
    assert run_estimate.fit_runs(grid, runs, False, probes, protocol, limit_s=900.0, topic_limit=False) is None
    precision = run_estimate.expected_precision(protocol, fit.runs)
    assert precision is not None and precision["target"] == 0.015 and 0.03 < precision["half_width"] < 0.06


R12_TOPIC = ("TOPIC. When many hypotheses are tested at once, how do Bonferroni correction and the Benjamini-Hochberg procedure "
             "compare? CONSTRAINTS. CPU only; under 60 seconds of runtime; fixed random seeds.")


async def _replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> tuple[Any, dict[str, Any] | None]:
    """FI's sizing step on the quest's own protocol, with the timing replaced by what each setting really cost."""
    fx = _fixture(name)
    grid, runs, probes = _ledger_probes(name)
    keep = {k: fx["protocol"][k] for k in ("min_runs_per_setting", "precision", "metrics")}
    eng, model, runner, _ex = await _sizing(tmp_path, timeout_s=900, grid=grid, runs=runs, **keep)
    (eng.quest_root / ".fi" / "topic_requirements.json").write_text(
        json.dumps({"constraints": fx["constraints"]}), encoding="utf-8")

    async def measured(*a: Any, runs: int, **k: Any) -> Any:
        return [p.at(runs, int(fx["protocol"]["runs_per_setting"])) for p in probes], ""

    monkeypatch.setattr(run_estimate, "measure", measured)
    patch = await eng._size_the_run({"iteration": 0, "topic": R12_TOPIC}, runner, sys.executable, None)
    assert model.asked == []
    return eng, patch


@pytest.mark.asyncio
async def test_replay_r13_lowers_the_runs_below_the_plans_minimum_to_fit_the_topics_minute(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, patch = await _replay(tmp_path, monkeypatch, "r13_fdr")
    assert patch is None, "the run goes on"
    design = plan.load_design(eng.quest_root)[0]
    runs = design["protocol"]["runs_per_setting"]
    assert design["protocol"]["grid"] == _fixture("r13_fdr")["protocol"]["grid"] and 300 < runs < 900
    record = run_estimate.read_record(eng.quest_root)
    assert record["limit"] == "topic" and record["limit_s"] == 60.0 and record["hard_limit_s"] == 900.0 and record["fits"]
    reduced = record["reduced"]
    assert reduced["below_plan_minimum"] and reduced["plan_minimum"] == 4096 and reduced["to"]["runs_per_setting"] == runs
    assert reduced["precision"]["target"] == 0.015
    note = reduced["note"]
    assert "fits the topic's limit (60 seconds)" in note and f"from 4096 to {runs}" in note and "all 27 setting(s)" in note
    assert "a part of the study not fully done" in note and "the plan's target is ±0.015" in note
    assert "the topic's limit is 60 seconds" in _log(eng)
    # The paper must say so by name, or it is sent back.
    items = [i for i in paper_facts.required_items(eng.quest_root) if i.kind == paper_facts.FEWER_RUNS]
    assert [i.name for i in items] == [paper_facts.FEWER_RUNS_NAME]
    # with the numbers: the runs against the plan's, below its minimum, and the precision those runs give
    assert f"{runs} runs per setting instead of the 4096 the plan asked for" in items[0].text
    assert "below the plan's minimum of 4096" in items[0].text and "against the plan's target of ±0.015" in items[0].text


@pytest.mark.asyncio
async def test_replay_r12_runs_at_the_plans_minimum_over_the_topics_minute_and_says_why(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, patch = await _replay(tmp_path, monkeypatch, "r12_fdr")
    assert patch is None, "over the topic's limit but within FI's own: the run goes on, no stop"
    design = plan.load_design(eng.quest_root)[0]
    assert design["protocol"]["runs_per_setting"] == 1068 and design["protocol"]["grid"] == _fixture("r12_fdr")["protocol"]["grid"]
    reduced = run_estimate.read_record(eng.quest_root)["reduced"]
    assert reduced["over_topic_limit"] and not reduced["below_plan_minimum"]
    assert "cannot be met by fewer runs per setting" in reduced["note"] and "over the topic's limit" in reduced["note"]
    assert "from 2000 to the plan's minimum of 1068" in reduced["note"]
    # Every run FI made smaller is named in the paper, a cut to the plan's own minimum too (2000 asked, 1068 run).
    items = [i for i in paper_facts.required_items(eng.quest_root) if i.kind == paper_facts.FEWER_RUNS]
    assert [i.name for i in items] == [paper_facts.FEWER_RUNS_NAME]
    assert "1068 runs per setting instead of the 2000 the plan asked for" in items[0].text
    assert "still over the topic's time limit" in items[0].text and "below the plan's minimum" not in items[0].text


@pytest.mark.asyncio
async def test_without_a_limit_in_the_topic_the_run_is_sized_to_fis_own_limit_as_before(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fx = _fixture("r13_fdr")
    grid, runs, probes = _ledger_probes("r13_fdr")
    eng, _model, runner, _ex = await _sizing(tmp_path, timeout_s=900, grid=grid, runs=runs,
                                             min_runs_per_setting=2048)

    async def measured(*a: Any, runs: int, **k: Any) -> Any:
        return [p.at(runs, int(fx["protocol"]["runs_per_setting"])) for p in probes], ""

    monkeypatch.setattr(run_estimate, "measure", measured)
    assert await eng._size_the_run({"iteration": 0, "topic": "Compare Bonferroni and BH."}, runner, sys.executable, None) is None
    record = run_estimate.read_record(eng.quest_root)
    assert record["fits"] and record["limit"] == "execution.timeout_s" and record["topic_limit_s"] is None
    assert plan.load_design(eng.quest_root)[0]["protocol"]["runs_per_setting"] == runs


# ---- the real code, timed ---------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["r12_fdr", "r13_fdr"])
def test_timing_the_real_simulation_measures_what_one_trial_costs(name: str, tmp_path: Path) -> None:
    """FI's own timing run of the quest's real simulate.py. Before, one later trial on the coarse clock (0 or 16 ms) gave
    estimates of 111 s and 21 s for runs of about 850 s and 98 s. The bounds are wide: the machine may be busy."""
    pytest.importorskip("numpy")
    if name == "r12_fdr":
        pytest.importorskip("scipy")
        pytest.importorskip("matplotlib")
    import shutil

    fx = _fixture(name)
    root = tmp_path / "quest"
    shutil.copytree(fx["root"] / "code", root / "code")
    grid, runs = fx["protocol"]["grid"], int(fx["protocol"]["runs_per_setting"])
    probes, why = asyncio.run(run_estimate.measure(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", grid, runs=runs,
        deterministic=False, limit_s=900.0, budget_s=300.0, thresholds=fx["protocol"].get("thresholds")))
    assert why == "" and probes, why
    base = probes[0]
    # The quest's own ledger cannot say what one trial cost (its durations were read on the coarse clock, most of them 0),
    # so the reference is the real run's time without the analysis.
    assert base.timed_trials >= 3 and base.per_trial_s is not None and base.per_trial_s > 0
    est = run_estimate.estimate(grid, runs, False, probes)
    wall, analysis = WALL[name]
    assert est is not None and 0.5 * (wall - analysis) < est.seconds < 3 * (wall - analysis), (est.seconds, base)
    assert not (root / "raw").exists(), "the timed trials are not results"
