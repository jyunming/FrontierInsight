"""A search whose evaluation budget ran out while its best design was still improving goes on by itself.

The rule uses only FI's own record of the search: it stopped because its budget ran out, it is not frozen, the last
tenth of its evaluations gave more than 5% of the whole improvement over the baseline, and one more round (one
start's share of the budget) fits in what is left of the call's time limit. The earlier evaluations are answered from
the record, so only the new round runs the simulation.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from core import optimise
from core import optimise_refine as refine
from tests.test_optimise_runner import LocalExecutor

SIMULATE = '''\
def run_cell(cell):
    with open(r"__COUNT__", "a", encoding="utf-8") as fh:
        fh.write("1\\n")
    x, y = cell["x"], cell["y"]
    return {"f": (x - 3.0) ** 2 + 2.0 * (y - 1.5) ** 2 + 10.0 + 0.5 * cell["mesh"] ** 2, "g": x}
'''
ANALYSIS = '''\
import json, os
best = json.load(open(os.environ["FI_BEST_DESIGN"], encoding="utf-8"))
print("RESULT_JSON: " + json.dumps({"evaluated": best["evaluations"]["search"]}))
'''


# --- the rule on its own ------------------------------------------------------------------------------------------------


def _rows(values: list[float], baseline: float) -> list[dict[str, Any]]:
    rows = [{"event": "evaluation", "n": 1, "stage": "baseline", "objective": baseline, "status": "ok",
             "feasible": True, "counted": True}]
    for i, v in enumerate(values):
        rows.append({"event": "evaluation", "n": i + 2, "stage": "search", "objective": v, "status": "ok",
                     "feasible": True, "counted": True})
    return rows


def test_a_trace_flat_then_rising_in_the_last_tenth_fires_for_maximise_and_minimise() -> None:
    up = [0.0] * 90 + [float(i + 1) for i in range(10)]
    fired = optimise.still_improving(_rows(up, -0.5), "maximise")
    assert fired is not None and fired["total"] == pytest.approx(10.5) and fired["gain_last_tenth"] == pytest.approx(10.0)
    assert fired["share"] == pytest.approx(10.0 / 10.5)
    down = [-v for v in up]
    assert optimise.still_improving(_rows(down, 0.5), "minimise") is not None
    # A rise is no improvement of a quantity made as low as possible.
    assert optimise.still_improving(_rows(up, -0.5), "minimise") is None


def test_a_baseline_that_breaks_a_limit_counts_from_the_first_design_that_meets_every_limit() -> None:
    rows = _rows([0.0] * 18 + [1.0, 2.0], -0.5)
    rows[0]["feasible"] = False  # the baseline breaks a limit: its value is not a start to improve on
    fired = optimise.still_improving(rows, "maximise")
    assert fired is not None and fired["total"] == 2.0 and fired["gain_last_tenth"] == 2.0
    rows = _rows([3.0] + [3.0] * 19, -0.5)
    rows[0]["feasible"] = False
    assert optimise.still_improving(rows, "maximise") is None


def test_a_first_design_that_meets_every_limit_in_the_last_tenth_fires() -> None:
    rows = _rows([0.0] * 18 + [1.0, 2.0], -0.5)
    for row in rows[:19]:
        row["feasible"] = False  # nothing met every limit before the last tenth
    fired = optimise.still_improving(rows, "maximise")
    assert fired is not None and fired["total"] == 1.0 and fired["share"] == 1.0


def test_a_converged_trace_does_not_fire() -> None:
    values = [float(min(i, 50)) for i in range(100)]
    assert optimise.still_improving(_rows(values, -1.0), "maximise") is None


def test_no_improvement_at_all_does_not_fire() -> None:
    assert optimise.still_improving(_rows([1.0] * 40, 1.0), "maximise") is None
    assert optimise.still_improving(_rows([0.5] * 40, 1.0), "maximise") is None  # worse than the baseline


def test_a_float_wiggle_in_the_last_tenth_does_not_fire() -> None:
    values = [1.0] * 99 + [1.0 + 1e-10]
    assert optimise.still_improving(_rows(values, 0.0), "maximise") is None


def test_infeasible_and_scan_rows_do_not_count() -> None:
    rows = _rows([1.0] * 90 + [1.0] * 10, 0.0)
    rows.append({"event": "evaluation", "n": 200, "stage": "scan", "objective": 99.0, "status": "ok",
                 "feasible": True, "counted": False})
    rows[-2]["objective"], rows[-2]["feasible"] = 99.0, False
    assert optimise.still_improving(rows, "maximise") is None


# --- add_round ------------------------------------------------------------------------------------------------------------


def _block(**changes: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "f", "direction": "minimise", "unit": "K", "meaning": "the peak temperature"},
        "design_variables": [{"name": "x", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"},
                             {"name": "y", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"}],
        "constraints": [{"quantity": "g", "limit": "<= 5.5"}],
        "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
        "numerical_settings": {"mesh": {"search": 0.4}},
        "evaluation_budget": {"starts": 1, "per_start": 8},
        "search_method": "bounded_local",
    }
    block.update(copy.deepcopy(changes))
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _level_rows(level: dict[str, Any], values: list[float], baseline: float, first_n: int) -> list[dict[str, Any]]:
    return [dict(r, level=level, n=r["n"] + first_n) for r in _rows(values, baseline)]


def _compared(parts: list[tuple[dict[str, Any], list[float], str]]) -> optimise.SearchRun:
    rows: list[dict[str, Any]] = []
    for level, values, _ in parts:
        rows += _level_rows(level, values, 0.0, len(rows))
    record = {"evaluations": {"stopped_because": "budget"},
              "compared": {"levels": [{"level": level, "stopped_because": stop} for level, _, stop in parts]}}
    return optimise.SearchRun(record=record, rows=rows)


def test_a_compared_search_is_judged_level_by_level() -> None:
    rising = [0.0] * 18 + [1.0, 2.0]   # still improving at the end of its own budget
    done = [5.0] + [5.0] * 19          # its best at the first design
    # The level still improving comes first: the trace of the last level alone would say the search had converged.
    run = _compared([({"family": "a"}, rising, "budget"), ({"family": "b"}, done, "budget")])
    assert optimise.still_improving(run.rows[len(rising) + 1:], "maximise") is None
    fired = optimise._still_improving_somewhere(run, "maximise")
    assert fired is not None and fired["level"] == {"family": "a"}
    # A level that stopped for another reason is not continued for its trace.
    run = _compared([({"family": "a"}, rising, "converged"), ({"family": "b"}, done, "budget")])
    assert optimise._still_improving_somewhere(run, "maximise") is None
    # The baselines of later levels do not count as a level's own improvement.
    run = _compared([({"family": "a"}, done, "budget"), ({"family": "b"}, done, "budget")])
    assert optimise._still_improving_somewhere(run, "maximise") is None


def test_add_round_keeps_auto_and_does_not_add_the_same_automatic_round_twice(tmp_path: Path) -> None:
    block = _block()
    first = refine.add_round(tmp_path, block, {"added": 6, "target": None, "asked": "FI: more", "auto": 1})
    again = refine.add_round(tmp_path, block, {"added": 6, "target": None, "asked": "FI: more", "auto": 1})
    assert first == again and len(again) == 1 and again[0]["auto"] == 1
    second = refine.add_round(tmp_path, block, {"added": 6, "target": None, "asked": "FI: more", "auto": 2})
    assert [r["auto"] for r in second] == [1, 2]
    stored = json.loads((tmp_path / optimise.ROUNDS_PATH).read_text(encoding="utf-8"))["rounds"]
    assert [r["auto"] for r in stored] == [1, 2]


def test_a_persons_round_has_no_auto_key(tmp_path: Path) -> None:
    block = _block()
    rounds = refine.add_round(tmp_path, block, {"added": 6, "target": None, "asked": "push further", "refine": 1})
    assert "auto" not in rounds[0] and rounds[0]["refine"] == 1
    assert len(refine.add_round(tmp_path, block, {"added": 6, "target": None, "asked": "push further", "refine": 1})) == 1


# --- through FI's own search --------------------------------------------------------------------------------------------


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(SIMULATE.replace("__COUNT__", str(root.parent / "count.txt")),
                                               encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    return root


def _runner(root: Path, block: dict[str, Any]) -> optimise.OptimisationRunner:
    return optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                       simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")


async def _go(runner: optimise.OptimisationRunner, root: Path, timeout_s: int = 300) -> None:
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=timeout_s)
    assert result.returncode == 0, result.stderr[-2000:]


def _count(root: Path) -> int:
    path = root.parent / "count.txt"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def _rounds(root: Path) -> list[dict[str, Any]]:
    path = root / optimise.ROUNDS_PATH
    return json.loads(path.read_text(encoding="utf-8"))["rounds"] if path.is_file() else []


def _best(root: Path) -> dict[str, Any]:
    return json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_search_still_improving_when_its_budget_ran_out_continues_once_without_rerunning_what_it_did(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    block = _block()  # eight evaluations do not reach the optimum at x = 3, y = 1.5; the last tenth of them still gains
    # What six evaluations alone give, for the comparison.
    plain_root = _quest(tmp_path / "plain")
    monkeypatch.setattr(optimise, "still_improving", lambda rows, direction: None)
    await _go(_runner(plain_root, block), plain_root)
    plain_count, plain_best = _count(plain_root), _best(plain_root)["best"]["objective"]
    assert _rounds(plain_root) == []
    monkeypatch.undo()

    root = _quest(tmp_path / "auto")
    runner = _runner(root, block)
    fired = []
    real = optimise.still_improving
    monkeypatch.setattr(optimise, "still_improving", lambda rows, direction: fired.append(1) or real(rows, direction))
    await _go(runner, root)
    assert fired, "the rule was asked"
    rounds = _rounds(root)
    assert rounds and rounds[0]["auto"] == 1 and rounds[0]["added"] == 8 and rounds[0]["target"] is None
    assert [r["auto"] for r in rounds] == list(range(1, len(rounds) + 1))
    assert "still improving" in rounds[0]["asked"]
    best = _best(root)
    assert best["best"]["objective"] < plain_best, "the continued search found a better design"
    # The record says FI continued it, never that a person asked.
    assert all(r.get("by") == "FI" for r in best["continued"]) and best["continued"]
    assert "by FI" in json.dumps(best) and "at the person's request" not in json.dumps(best)
    from core import best_design_report
    text = best_design_report.section(root, block)
    assert "Continued by FI, still improving when its budget ran out (round 1)" in text
    assert "at the person's request" not in text
    # Each evaluation of the whole record ran once: the first eight were answered from the record, not run again.
    ledger = [json.loads(x) for x in (root / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    counted = [r for r in ledger if r.get("event") == "evaluation" and r.get("counted")]
    assert len(counted) == best["evaluations"]["search"] == 8 * (1 + len(rounds))
    first_calls = plain_count  # the first round alone
    assert _count(root) < 3 * first_calls * (1 + len(rounds)), "not a search run again from its start"
    # FI's rounds file is the one the runner keeps put back.
    assert runner._rounds == (root / optimise.ROUNDS_PATH).read_bytes()


@pytest.mark.asyncio
async def test_earlier_evaluations_are_not_run_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    block = _block()
    monkeypatch.setattr(optimise, "still_improving", lambda rows, direction: None)
    await _go(_runner(root, block), root)
    first = _count(root)  # the first round alone, the check at finer settings included
    monkeypatch.undo()
    root2 = _quest(tmp_path / "again")
    await _go(_runner(root2, block), root2)
    rounds = _rounds(root2)
    assert rounds
    # The search's own evaluations of the first round are in the count once, whatever the number of rounds.
    ledger = [json.loads(x) for x in (root2 / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    designs = [json.dumps(r["design"], sort_keys=True) for r in ledger if r.get("event") == "evaluation"
               and r.get("counted") and r.get("stage") != "scan"]
    assert len(designs) == len(set(designs)), "no design was evaluated twice"
    assert _count(root2) - first == 8 * len(rounds), "only the new rounds ran the simulation"


@pytest.mark.asyncio
async def test_a_search_that_converged_gets_no_automatic_round(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    block = _block(evaluation_budget={"starts": 1, "per_start": 80})
    await _go(_runner(root, block), root)
    assert _best(root)["evaluations"]["stopped_because"] != "budget"
    assert _rounds(root) == []


@pytest.mark.asyncio
async def test_no_round_when_it_does_not_fit_in_the_time_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    monkeypatch.setattr(optimise, "_seconds_per_evaluation", lambda run: 1e6)
    await _go(_runner(root, _block()), root)
    assert _best(root)["evaluations"]["stopped_because"] == "budget"
    assert _rounds(root) == []


@pytest.mark.asyncio
async def test_a_frozen_study_never_continues(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    block = _block()
    await _go(_runner(root, block), root)  # exploration (continues itself)
    before = _rounds(root)
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py",
                                         frozen=True)
    await _go(runner, root)
    assert _rounds(root) == before


@pytest.mark.asyncio
async def test_a_round_costs_only_the_levels_that_were_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The plan compares three values of a condition that changes no result the search judges (only a number the plan
    # does not name): the second and third give the first one's results and are not searched, so a round costs one
    # level's share of evaluations, not three.
    root = _quest(tmp_path)
    simulate = root / "code" / "simulate.py"
    simulate.write_text(simulate.read_text(encoding="utf-8").replace(
        '"g": x}', '"g": x, "h": {"cup": 0.0, "mug": 1.0, "bowl": 2.0}[cell["shape"]]}'), encoding="utf-8")
    block = _block(fixed={"shape": "cup"}, grid={"shape": ["cup", "mug", "bowl"]})
    monkeypatch.setattr(optimise, "_seconds_per_evaluation", lambda run: 20.0)  # 8 x 20 s fits in 300 s; 3 x 160 not
    await _go(_runner(root, block), root, timeout_s=300)
    levels = _best(root)["compared"]["levels"]
    assert [bool(p.get("same_as")) for p in levels] == [False, True, True]
    assert _rounds(root) and _rounds(root)[0]["auto"] == 1


@pytest.mark.asyncio
async def test_a_continued_round_still_skips_an_alternative_that_gives_the_first_ones_results(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The plan names a result besides the objective and the limit. The round's earlier evaluations are answered from
    # FI's record: they must carry that result too, or the second and third alternatives look different from the first
    # (one side without the result) and are searched in full in the round.
    root = _quest(tmp_path)
    simulate = root / "code" / "simulate.py"
    simulate.write_text(simulate.read_text(encoding="utf-8").replace(
        '"g": x}', '"g": x, "mass": 2.0 * x + y, "h": {"cup": 0.0, "mug": 1.0, "bowl": 2.0}[cell["shape"]]}'),
        encoding="utf-8")
    block = _block(fixed={"shape": "cup"}, grid={"shape": ["cup", "mug", "bowl"]})
    protocol = {"optimisation": block,
                "metrics": [{"id": "mass", "kind": "mean", "estimand": "the part's mass", "unit": "one design"}]}
    monkeypatch.setattr(optimise, "_seconds_per_evaluation", lambda run: 20.0)
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol=protocol,
                                         simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")
    await _go(runner, root, timeout_s=300)
    assert _rounds(root) and _rounds(root)[0]["auto"] == 1
    levels = _best(root)["compared"]["levels"]
    assert [bool(p.get("same_as")) for p in levels] == [False, True, True], levels
    ledger = [json.loads(x) for x in (root / "raw" / "optimisation_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all("mass" in (r.get("results") or {}) for r in ledger if r.get("event") == "evaluation" and r.get("status") == "ok")
