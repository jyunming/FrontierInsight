"""FI checks the best design its search found, at finer numerical settings, before anything is written about it.

The check is the ENGINE's (core/optimum_check.py): it evaluates the search's best designs and the baseline again at the
finer levels of the plan's numerical settings, estimates each design's numerical error from how its value changes, and
reports plainly what holds and what does not. A check that fails is the study's result: the quest goes on, and the
evidence level is limited with a plain sentence.

The toy problems run in-process (``check_sync``): the check asks for evaluations and is sent the answers, as FI's
harness answers it in a quest.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from core import evidence
from core import optimisation_plan as op
from core import optimise
from core import optimise_search as osearch
from core import optimum_check as oc


def _block(**changes: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "f", "direction": "minimise", "unit": "K"},
        "design_variables": [
            {"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"},
            {"name": "y", "low": -5.0, "high": 5.0, "kind": "continuous"},
        ],
        "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
        "numerical_settings": {"mesh": {"search": 0.4}},
        "evaluation_budget": {"starts": 3, "per_start": 80},
        "search_method": "bounded_local",
    }
    block.update(copy.deepcopy(changes))
    fixed, why = op.normalize({k: v for k, v in block.items() if v is not None})
    assert why is None, why
    return fixed


def _search_then_check(block: dict[str, Any], sim, *, seed: int = 0) -> dict[str, Any]:
    """The search as FI runs it, then the check, both calling ``sim(cell)``."""
    outcome = osearch.run_sync(block, sim, seed=seed)
    record = osearch.best_design(outcome, block, seed=seed)
    return oc.check_sync(block, record, outcome["rows"], sim)


def _bowl(cell: dict[str, Any]) -> dict[str, float]:
    # A real optimum at (2, -1); the mesh adds the same second-order error to every design.
    x, y = cell["x"], cell["y"]
    return {"f": (x - 2) ** 2 + (y + 1) ** 2 + 0.5 * cell["mesh"] ** 2, "g": x + y}


# --- (a) a real optimum survives refinement ----------------------------------------------------------------------------


def test_a_real_optimum_survives_the_finer_settings() -> None:
    record = _search_then_check(_block(), _bowl)
    assert record["verdict"] == "verified", record["says"]
    assert record["checks"]["refinement"]["status"] == "passed"
    assert record["checks"]["improvement"]["status"] == "passed"
    assert record["checks"]["neighbourhood"]["status"] == "passed", record["checks"]["neighbourhood"]
    assert record["checks"]["starts"]["status"] == "passed", record["checks"]["starts"]
    best = record["best"]
    assert abs(best["design"]["x"] - 2) < 0.05 and abs(best["design"]["y"] + 1) < 0.05
    # The finer levels come from FI's fixed rule (half, then a quarter of the search's mesh), and the record says so.
    assert record["settings"]["levels"] == [{"mesh": 0.2}, {"mesh": 0.1}]
    assert record["settings"]["where"] == {"mesh": "rule"}
    # The observed order of the mesh error is the one put in (2), and the error is small against the improvement.
    assert best["observed_order"] == pytest.approx(2.0, abs=1e-6)
    imp = record["improvement"]
    assert imp["value"] == pytest.approx(5.0, abs=0.01) and imp["rule"] == "rule"
    assert imp["numerical_error"] < 0.05 and imp["beyond_numerical_error"] is True
    assert record["evaluations"]["check"] <= record["evaluations"]["planned"]
    assert "holds at finer numerical settings" in record["says"]


def test_the_check_is_not_taken_from_the_search_budget_and_its_count_is_planned() -> None:
    block = _block()
    count = op.budget(block)
    # (3 best + the baseline) × 2 finer levels + 2 × 2 nudges = 12
    assert count["check"] == 12
    text = "\n".join(op.plan_lines({"study_type": "find_best_design", "protocol": {"optimisation": block}}))
    assert "12 evaluations at most" in text and "not taken from the search's budget" in text
    assert "does not yet recompute" not in text


# --- (b) an optimum that exists only because the search's mesh is coarse ----------------------------------------------


def _artefact(cell: dict[str, Any]) -> dict[str, float]:
    # The true objective is x² (the baseline, x = 0, is best); a coarse mesh digs a false valley at x = 3.
    x = cell["x"]
    return {"f": x ** 2 - 60.0 * cell["mesh"] ** 2 * math.exp(-((x - 3.0) ** 2))}


def test_a_gain_that_only_a_coarse_mesh_shows_disappears_and_is_reported() -> None:
    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    outcome = osearch.run_sync(block, _artefact, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    assert record["best"]["design"] == {"x": 3} and record["improvement"]["better"] is True, "the search is fooled"
    check = oc.check_sync(block, record, outcome["rows"], _artefact)
    assert check["verdict"] == "improvement_not_shown", check["says"]
    assert "disappears at finer settings" in check["checks"]["improvement"]["says"]
    assert check["improvement"]["value"] < 0 and check["improvement"]["search"] > 0
    assert "disappears at finer settings" in check["says"]


# --- (c) a limit broken at finer settings ---------------------------------------------------------------------------------


def _rows(block: dict[str, Any], designs: list[tuple[int, dict[str, Any], float, dict[str, float]]]) -> tuple[
        dict[str, Any], list[dict[str, Any]]]:
    """A search's record by hand: ``(start, design, objective at the search's settings, constraints)`` per row, the
    baseline first."""
    rows = []
    sign = 1.0 if block["objective"]["direction"] == "minimise" else -1.0
    for n, (start, design, value, constraints) in enumerate(designs, start=1):
        judged = osearch.judge(block, {"f": value, **constraints})
        rows.append({"event": "evaluation", "n": n, "stage": "baseline" if n == 1 else "local", "start": start,
                     "design": design, "objective": value, "constraints": dict(constraints),
                     "feasible": judged["feasible"], "status": "ok", "method": "bounded_local", "counted": True})
    feasible = [r for r in rows if r["feasible"]]
    best = min(feasible, key=lambda r: sign * r["objective"])
    base = rows[0]
    record = {"best": {"design": best["design"], "objective": best["objective"], "constraints": best["constraints"],
                       "feasible": True},
              "baseline": {"design": base["design"], "objective": base["objective"], "constraints": base["constraints"],
                           "feasible": base["feasible"]},
              "evaluations": {"starts": 2, "search": len(rows), "budget": 100, "stopped_because": "converged"},
              "method": {"used": "bounded_local"}}
    return record, rows


def test_a_limit_the_best_design_breaks_at_finer_settings_moves_the_answer_to_the_next_best(tmp_path: Path) -> None:
    # A coarse mesh underestimates the mass: design A (just under the limit) meets it at the search's settings and breaks
    # it at finer ones; design B, a little warmer, meets it at every setting.
    block = _block(constraints=[{"quantity": "mass", "limit": "<= 120"}])

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        x = cell["x"]
        f = {0.0: 10.0, 1.0: 6.0, 2.0: 6.5}[x]
        mass = {0.0: 100.0, 1.0: 119.9, 2.0: 110.0}[x] + (0.4 - cell["mesh"]) * 2.0  # heavier at a finer mesh
        return {"f": f, "mass": mass}

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.0, {"mass": 100.0}),
                                 (0, {"x": 1.0, "y": 0.0}, 6.0, {"mass": 119.9}),
                                 (1, {"x": 2.0, "y": 0.0}, 6.5, {"mass": 110.0})])
    check = oc.check_sync(block, record, rows, sim)
    constraints = check["checks"]["constraints"]
    assert constraints["status"] == "failed"
    assert "breaks the limit on mass at finer settings" in constraints["says"] and "next best" in constraints["says"]
    assert check["best"]["design"] == {"x": 2.0, "y": 0.0} and check["best"]["replaces_search_best"] is True
    assert check["verdict"] in ("verified", "not_local_optimum")
    gaps = _gaps(tmp_record=check, block=block, root=tmp_path / 'q')
    assert any("breaks the limit on mass" in g for g in gaps.get("independently_validated", []))


def test_no_design_that_meets_every_limit_at_finer_settings_is_infeasible() -> None:
    block = _block(constraints=[{"quantity": "g", "limit": ">= 2"}], baseline={"values": {"x": 2.0, "y": 1.0},
                                                                                "source": "s"})

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        # x + y is overestimated by a coarse mesh: the search hugs a limit it does not really meet.
        x, y = cell["x"], cell["y"]
        return {"f": (x - 2) ** 2 + (y + 1) ** 2, "g": x + y + 0.5 * cell["mesh"]}

    check = _search_then_check(block, sim)
    assert check["checks"]["constraints"]["status"] == "failed"
    assert "breaks the limit on g at finer settings" in check["checks"]["constraints"]["says"]
    assert check["verdict"] == "infeasible", check["says"]


# --- (d) an optimum on the edge of its range ------------------------------------------------------------------------------


def test_an_optimum_on_the_edge_of_its_range_is_flagged(tmp_path: Path) -> None:
    block = _block(design_variables=[{"name": "t", "low": 0.4, "high": 2.0}], baseline={"values": {"t": 1.0},
                                                                                        "source": "s"},
                   evaluation_budget={"starts": 2, "per_start": 40})

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": (cell["t"] - 3.0) ** 2 + 0.1 * cell["mesh"] ** 2}  # the best is outside the range, at t = 3

    check = _search_then_check(block, sim)
    hood = check["checks"]["neighbourhood"]
    assert check["best"]["design"]["t"] == pytest.approx(2.0)
    assert hood["status"] == "failed" and hood["at_bound"][0]["edge"] == "high"
    assert "upper edge of the range allowed for t" in hood["says"]
    assert check["verdict"] == "verified", "the improvement itself holds"
    assert check["passed"] is False
    gaps = _gaps(tmp_record=check, block=block, root=tmp_path / 'q')
    assert any("upper edge" in g for g in gaps["publication_ready"])


def test_a_variable_the_simulation_ignores_is_named() -> None:
    def sim(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": (cell["x"] - 2) ** 2 + 0.5 * cell["mesh"] ** 2}  # y is never read

    check = _search_then_check(_block(), sim)
    assert "y" in check["checks"]["neighbourhood"]["no_effect"]
    assert "nudging y changes nothing" in check["checks"]["neighbourhood"]["says"]


def test_a_nearby_better_design_means_the_search_had_not_finished() -> None:
    block = _block()

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": (cell["x"] - 2) ** 2 + (cell["y"] + 1) ** 2}

    # The search stopped at (1.5, -1): its record says so, and a nudge of x finds better.
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.0, {}), (0, {"x": 1.5, "y": -1.0}, 0.25, {})])
    check = oc.check_sync(block, record, rows, sim)
    assert check["verdict"] == "not_local_optimum", check["says"]
    assert "the search had not finished" in check["checks"]["neighbourhood"]["says"]


def test_starting_points_that_end_at_different_designs_are_reported() -> None:
    block = _block()

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        x, y = cell["x"], cell["y"]
        return {"f": min((x - 2) ** 2 + (y + 1) ** 2, (x + 3) ** 2 + (y - 3) ** 2 + 1.0)}

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.0, {}), (0, {"x": 2.0, "y": -1.0}, 0.0, {}),
                                 (1, {"x": -3.0, "y": 3.0}, 1.0, {})])
    check = oc.check_sync(block, record, rows, sim)
    assert check["checks"]["starts"]["status"] == "failed"
    assert "different designs" in check["checks"]["starts"]["says"]


def test_a_search_from_one_starting_point_is_not_called_agreement() -> None:
    check = _search_then_check(_block(evaluation_budget={"starts": 1, "per_start": 80}), _bowl)
    assert check["checks"]["starts"]["status"] == "not_checked"
    assert "one starting point" in check["checks"]["starts"]["says"]


def test_a_search_that_ran_out_of_budget_is_reported() -> None:
    check = _search_then_check(_block(evaluation_budget={"starts": 2, "per_start": 6}), _bowl)
    assert check["checks"]["budget"]["status"] == "failed"
    assert "whole budget" in check["checks"]["budget"]["says"] or "share" in check["checks"]["budget"]["says"]


# --- (e) the plan's threshold, and FI's default rule ------------------------------------------------------------------------


def _two_designs(slope: float, tolerance: Any = None) -> dict[str, Any]:
    """The baseline (x = 0) and one better design (x = 1), 0.5 K apart at every mesh, each with a first-order mesh
    error: with three levels (0.4, 0.2, 0.1) each design's error is 0.125 × slope."""
    block = _block(improvement_tolerance=tolerance) if tolerance is not None else _block()

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        return {"f": (10.0 if cell["x"] == 0 else 9.5) + slope * cell["mesh"]}

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.0 + slope * 0.4, {}),
                                 (0, {"x": 1.0, "y": 0.0}, 9.5 + slope * 0.4, {})])
    return oc.check_sync(block, record, rows, sim)


def test_without_a_threshold_better_means_more_than_the_numerical_error_of_the_two_designs() -> None:
    small = _two_designs(1.6)  # errors 0.2 + 0.2 = 0.4 < 0.5
    assert small["improvement"]["numerical_error"] == pytest.approx(0.4)
    assert small["improvement"]["rule"] == "rule" and small["improvement"]["shown"] is True
    assert small["checks"]["improvement"]["status"] == "passed"
    large = _two_designs(2.4)  # errors 0.3 + 0.3 = 0.6 > 0.5
    assert large["improvement"]["shown"] is False and large["verdict"] == "improvement_not_shown"
    assert "within the numerical error of the two designs" in large["checks"]["improvement"]["says"]


def test_the_plans_threshold_wins_when_it_gives_one_and_the_numerical_error_is_still_recorded(tmp_path: Path) -> None:
    strict = _two_designs(1.6, tolerance={"value": 1.0, "mode": "absolute"})
    assert strict["improvement"]["rule"] == "plan" and strict["improvement"]["shown"] is False
    assert "not more than the plan's threshold of 1" in strict["checks"]["improvement"]["says"]
    lenient = _two_designs(2.4, tolerance=0.1)
    assert lenient["improvement"]["shown"] is True and lenient["verdict"] == "verified"
    assert lenient["improvement"]["beyond_numerical_error"] is False
    # The level says so: above the plan's threshold, but within the numerical error.
    gaps = _gaps(tmp_record=lenient, block=_block(improvement_tolerance=0.1), root=tmp_path / 'q')
    assert any("within the numerical error" in g for g in gaps["statistically_adequate"])


# --- randomness -------------------------------------------------------------------------------------------------------------


def test_the_checks_seeds_are_fresh() -> None:
    check, search, key = oc.check_seeds(7, runs=20, search_runs=20)
    assert not set(check) & set(search) and len(set(check)) == 20 and key.startswith(oc.SEED_KEY)


def test_with_randomness_the_improvement_is_the_checked_one_with_its_interval() -> None:
    import random

    block = _block(runs_per_evaluation=4, check_runs=12)

    def sim(cell: dict[str, Any]) -> list[dict[str, float]]:
        # Twelve fresh runs, the same random numbers for every design (paired), a true gap of 1 K.
        rng = random.Random(1)
        noise = [rng.gauss(0.0, 0.3) for _ in range(12)]
        true = 10.0 if cell["x"] == 0 else 9.0
        return [{"f": true + e + 0.1 * cell["mesh"] ** 2} for e in noise]

    # At the search's settings the better design was lucky: it looked 3 K better.
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.0, {}), (0, {"x": 1.0, "y": 0.0}, 7.0, {})])
    check = oc.check_sync(block, record, rows, sim, noisy=True, check_runs=12)
    imp = check["improvement"]
    assert imp["search"] == pytest.approx(3.0) and imp["value"] == pytest.approx(1.0, abs=1e-6)
    assert imp["interval"]["ci_lower"] <= 1.0 <= imp["interval"]["ci_upper"]
    assert check["evaluations"]["runs_each"] == 12


# --- the numerical error --------------------------------------------------------------------------------------------------


def test_the_numerical_error_follows_the_grid_convergence_index() -> None:
    three = oc.numerical_error([1.08, 1.02, 1.005], [2.0, 2.0])
    assert three["order"] == pytest.approx(2.0) and three["error"] == pytest.approx(1.25 * 0.015 / 3.0)
    two = oc.numerical_error([1.08, 1.02], [2.0])
    assert two["error"] == pytest.approx(3.0 * 0.06) and "two levels only" in two["how"]
    swinging = oc.numerical_error([1.0, 1.2, 1.1], [2.0, 2.0])
    assert "swings" in swinging["how"] and swinging["error"] > 0
    growing = oc.numerical_error([1.0, 1.01, 1.2], [2.0, 2.0])
    assert "not converging" in growing["how"]
    assert oc.numerical_error([1.0], [])["error"] is None


def test_without_a_numerical_setting_nothing_is_checked_at_finer_settings(tmp_path: Path) -> None:
    block = _block(numerical_settings=None)
    check = _search_then_check(block, lambda c: {"f": (c["x"] - 2) ** 2 + (c["y"] + 1) ** 2})
    assert check["checks"]["refinement"]["status"] == "not_checked"
    assert check["verdict"] == "unverified"
    gaps = _gaps(tmp_record=check, block=block, root=tmp_path / 'q')
    assert any("names no numerical setting" in g for g in gaps["independently_validated"])


# --- the evidence ladder --------------------------------------------------------------------------------------------------


def _gaps(*, tmp_record: dict[str, Any], block: dict[str, Any], root: Path) -> dict[str, list[str]]:
    """The check's gaps for a quest (at ``root``) whose files hold ``tmp_record`` (a record from ``check_sync``)."""
    ledger = "\n".join(json.dumps(r) for r in [{"event": "search"}]) + "\n"
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "needs").mkdir(parents=True, exist_ok=True)
    (root / optimise.LEDGER_PATH).write_bytes(ledger.encode("utf-8"))
    text = json.dumps({**tmp_record, "ledger_sha256": oc._sha(ledger.encode("utf-8"))})
    (root / oc.CHECK_PATH).write_bytes(text.encode("utf-8"))
    # FI's best-design file names the check it recorded, by its hash.
    (root / optimise.BEST_PATH).write_text(json.dumps({"check": oc.attach_summary(tmp_record, text)}), encoding="utf-8")
    return oc.evidence_gaps(root, {"optimisation": block})


def test_the_evidence_level_needs_the_check_and_a_failed_check_limits_it(tmp_path: Path) -> None:
    block = _block()
    good = _search_then_check(block, _bowl)
    root = tmp_path / "q"
    assert _gaps(tmp_record=good, block=block, root=root) == {}
    # No check at all: the level stops below independently_validated.
    (root / oc.CHECK_PATH).unlink()
    gaps = oc.evidence_gaps(root, {"optimisation": block})
    assert any("not checked at finer numerical settings" in g for g in gaps["independently_validated"])
    # FI's record of the search changed after the check: not held to what ran.
    _gaps(tmp_record=good, block=block, root=root)
    (root / optimise.LEDGER_PATH).write_text('{"event": "search"}\n{"event": "evaluation", "design": {"x": 9, "y": 0}}\n',
                                            encoding="utf-8")
    gaps = oc.evidence_gaps(root, {"optimisation": block})
    assert any("changed after FI checked" in g for g in gaps["protocol_runtime_matched"])
    assert any("outside the plan's ranges" in g for g in gaps["protocol_runtime_matched"])
    # A measurement has none of this.
    assert oc.evidence_gaps(root, {"grid": {"x": [1]}}) == {}


def test_evidence_assess_puts_the_checks_sentence_where_it_limits_the_level(tmp_path: Path) -> None:
    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    outcome = osearch.run_sync(block, _artefact, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    check = oc.check_sync(block, record, outcome["rows"], _artefact)
    root = tmp_path / "q"
    _gaps(tmp_record=check, block=block, root=root)
    assessed = evidence.assess(root, {"design": {"protocol": {"optimisation": block}}})
    assert any("disappears at finer settings" in g for g in assessed["all_gaps"]["statistically_adequate"])
    rung = next(r for r in assessed["ladder"] if r["level"] == "independently_validated")
    assert any("within the plan's model" in s for s in rung["known_blind_spots"])
    assert "needs/OPTIMUM_CHECK.json" in rung["evidence_artifacts"]
    # A measurement's ladder carries none of the search's words.
    plain = evidence.assess(tmp_path / "m", {"design": {"protocol": {"grid": {"x": [1]}}}})
    rung = next(r for r in plain["ladder"] if r["level"] == "independently_validated")
    assert not any("within the plan's model" in s for s in rung["known_blind_spots"])


# --- through FI's harness: the search, the check, then the analysis ------------------------------------------------------

ARTEFACT_SIM = '''\
import math


def run_cell(cell):
    with open("count.txt", "a", encoding="utf-8") as fh:
        fh.write("1\\n")
    x = cell["x"]
    return {"f": x ** 2 - 60.0 * cell["mesh"] ** 2 * math.exp(-((x - 3.0) ** 2))}
'''
# The analysis reads FI's check and also tries to overwrite it: FI puts its own copy back.
ANALYSIS = '''\
import json, os
check = json.load(open(os.environ["FI_OPTIMUM_CHECK"], encoding="utf-8"))
best = json.load(open(os.environ["FI_BEST_DESIGN"], encoding="utf-8"))
with open(os.environ["FI_OPTIMUM_CHECK"], "w", encoding="utf-8") as fh:
    fh.write(json.dumps({"verdict": "verified"}))
print("RESULT_JSON: " + json.dumps({"verdict_seen": check["verdict"], "best_f": best["best"]["objective"]}))
'''


def _runner_quest(tmp_path: Path, simulate: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    return root


@pytest.mark.asyncio
async def test_a_failed_check_is_reported_and_the_analysis_still_runs(tmp_path: Path) -> None:
    from tests.test_optimise_runner import LocalExecutor

    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    root = _runner_quest(tmp_path, ARTEFACT_SIM)
    logs: list[str] = []

    class Log:
        def info(self, msg: str, *args: Any) -> None:
            logs.append(msg % args if args else msg)

        warning = info

    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                         simulate=root / "code" / "simulate.py",
                                         analysis=root / "code" / "experiment.py", log=Log())
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert '"verdict_seen": "improvement_not_shown"' in result.stdout, "the analysis read FI's check"
    check = json.loads((root / "needs" / "OPTIMUM_CHECK.json").read_text(encoding="utf-8"))
    assert check["verdict"] == "improvement_not_shown" and check["schema"] == oc.SCHEMA, "FI's copy was put back"
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    assert best["check"]["verdict"] == "improvement_not_shown" and best["checked_at_finer_settings"] is True
    assert "Checked at finer numerical settings: The improvement over the baseline disappears" in best["says"]
    text = "\n".join(logs)
    assert "[optimise] check at finer numerical settings: The improvement over the baseline disappears" in text
    assert "changed FI's record of the search" in text
    # The check's evaluations went through FI's harness too: the simulation counted them.
    count = len((root / "count.txt").read_text(encoding="utf-8").splitlines())
    assert count == best["evaluations"]["search"] + check["evaluations"]["check"]
    # A second run (an analysis repair) uses the search and the check FI already ran.
    await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=120)
    assert len((root / "count.txt").read_text(encoding="utf-8").splitlines()) == count
    assert any("the check at finer settings FI already ran is used" in line for line in logs)
    # The evidence level is limited, with the check's plain sentence.
    gaps = oc.evidence_gaps(root, {"optimisation": block})
    assert any("disappears at finer settings" in g for g in gaps["statistically_adequate"])


NOISY_SIM = '''\
import random


def run_trial(cell, trial_id, seed):
    with open("seeds.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{seed}\\n")
    rng = random.Random(seed)
    x, y = cell["x"], cell["y"]
    return {"f": (x - 2) ** 2 + (y + 1) ** 2 + rng.gauss(0.0, 0.05) + 0.5 * cell["mesh"] ** 2}
'''


@pytest.mark.asyncio
async def test_with_randomness_the_check_runs_with_seeds_the_search_never_used(tmp_path: Path) -> None:
    from tests.test_optimise_runner import LocalExecutor

    block = _block(runs_per_evaluation=2, check_runs=4, evaluation_budget={"starts": 1, "per_start": 12})
    root = _runner_quest(tmp_path, NOISY_SIM)
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", {"optimisation": block},
                                    base_seed=0, timeout_s=300)
    search_seeds = {int(s) for s in (root / "seeds.txt").read_text(encoding="utf-8").split()}
    check, _text, _key = await oc.run_check(LocalExecutor(), sys.executable, root, "code/simulate.py",
                                      {"optimisation": block}, run, base_seed=0, timeout_s=300)
    all_seeds = [int(s) for s in (root / "seeds.txt").read_text(encoding="utf-8").split()]
    check_seeds = set(all_seeds[len(all_seeds) - check["evaluations"]["check"] * 4:])
    assert check_seeds == set(check["seeds"]["check"]) and len(check_seeds) == 4
    assert not check_seeds & search_seeds, "the check's runs use seeds the search never used"
    assert check["evaluations"]["runs_each"] == 4
    assert check["improvement"]["interval"] is not None and check["verdict"] == "verified", check["says"]


@pytest.mark.asyncio
async def test_a_check_its_time_limit_cuts_short_is_unverified_never_a_pass(tmp_path: Path) -> None:
    from tests.test_optimise_runner import LocalExecutor

    slow = ARTEFACT_SIM.replace("def run_cell(cell):\n", "def run_cell(cell):\n    import time\n"
                                                         "    if cell['mesh'] < 0.5:\n        time.sleep(1.5)\n")
    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    root = _runner_quest(tmp_path, slow)
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", {"optimisation": block},
                                    base_seed=0, timeout_s=300)
    check, _text, _key = await oc.run_check(LocalExecutor(), sys.executable, root, "code/simulate.py",
                                      {"optimisation": block}, run, base_seed=0, timeout_s=2)
    assert check["verdict"] == "unverified" and check["finished"] is False
    assert "time limit" in check["says"]
    gaps = oc.evidence_gaps(root, {"optimisation": block})
    assert any("was not finished" in g for g in gaps["independently_validated"])


# --- what the check does NOT count as a pass ------------------------------------------------------------------------------


def test_a_design_that_failed_at_a_finer_level_does_not_get_a_numerical_error_of_zero() -> None:
    def sim(cell: dict[str, Any]) -> dict[str, float]:
        if (cell["x"], cell["y"]) != (0.0, 0.0) and cell["mesh"] == 0.2:
            raise RuntimeError("the solver diverged")
        return _bowl(cell)

    block = _block()
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.08, {}), (0, {"x": 2.0, "y": -1.0}, 0.08, {})])
    check = oc.check_sync(block, record, rows, sim)
    assert check["checks"]["refinement"]["status"] == "failed"
    assert check["improvement"]["numerical_error"] is None
    assert check["verdict"] == "unverified", check["says"]
    assert "could not be estimated" in check["checks"]["improvement"]["says"]


def test_with_randomness_and_one_finer_level_the_numerical_error_is_not_taken_as_zero() -> None:
    block = _block(numerical_settings={"mesh": {"search": 0.4, "check": [0.2]}}, runs_per_evaluation=2, check_runs=6)

    def sim(cell: dict[str, Any]) -> list[dict[str, float]]:
        return [{"f": (9.0 if cell["x"] else 10.0) + 0.01 * t + cell["mesh"]} for t in range(6)]

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.4, {}), (0, {"x": 1.0, "y": 0.0}, 9.4, {})])
    check = oc.check_sync(block, record, rows, sim, noisy=True, check_runs=6)
    assert check["improvement"]["beyond_numerical_error"] is None
    assert check["verdict"] == "unverified" and "second finer" in check["says"]


def test_with_randomness_one_fresh_run_is_not_a_test_of_the_improvement() -> None:
    block = _block()

    def sim(cell: dict[str, Any]) -> list[dict[str, float]]:
        return [{"f": (9.0 if cell["x"] else 10.0) + cell["mesh"] ** 2}]

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.16, {}), (0, {"x": 1.0, "y": 0.0}, 9.16, {})])
    check = oc.check_sync(block, record, rows, sim, noisy=True, check_runs=1)
    assert check["checks"]["improvement"]["status"] == "not_checked"
    assert check["verdict"] == "unverified" and "check_runs" in check["says"]


def test_best_designs_that_all_failed_at_the_finest_level_are_not_called_infeasible() -> None:
    def sim(cell: dict[str, Any]) -> dict[str, float]:
        if cell["x"] != 0.0 and cell["mesh"] < 0.15:
            raise RuntimeError("the solver diverged")
        return _bowl(cell)

    block = _block(constraints=[{"quantity": "g", "limit": "<= 10"}])
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.08, {"g": 0.0}), (0, {"x": 2.0, "y": -1.0}, 0.08, {"g": 1.0})])
    check = oc.check_sync(block, record, rows, sim)
    assert check["verdict"] == "unverified", check["says"]
    assert "diverged" in check["says"]


def test_a_check_cut_short_during_the_nudges_keeps_what_it_measured() -> None:
    block = _block()
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.08, {}), (0, {"x": 1.5, "y": -1.0}, 0.33, {})])
    gen = oc.check(block, record, rows)
    request, answered = next(gen), 0
    try:
        while True:
            # Every design at every level (2 designs × 2 levels), then the time is up during the nudges.
            reply = {"values": _bowl(request["cell"])} if answered < 4 else {"stop": "time"}
            answered += 1
            request = gen.send(reply)
    except StopIteration as done:
        check = done.value
    assert check["finished"] is False and check["verdict"] == "unverified"
    assert check["best"]["design"] == {"x": 1.5, "y": -1.0}, "the measured best design is kept"
    assert check["checks"]["refinement"]["status"] == "passed"
    assert "nudge" in check["says"]


def test_the_one_sentence_says_when_the_design_reported_is_not_the_searchs() -> None:
    block = _block(constraints=[{"quantity": "mass", "limit": "<= 120"}])

    def sim(cell: dict[str, Any]) -> dict[str, float]:
        f = {0.0: 10.0, 1.0: 6.0, 2.0: 6.5}[cell["x"]]
        return {"f": f, "mass": {0.0: 100.0, 1.0: 119.9, 2.0: 110.0}[cell["x"]] + (0.4 - cell["mesh"]) * 2.0}

    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 10.0, {"mass": 100.0}),
                                 (0, {"x": 1.0, "y": 0.0}, 6.0, {"mass": 119.9}),
                                 (1, {"x": 2.0, "y": 0.0}, 6.5, {"mass": 110.0})])
    check = oc.check_sync(block, record, rows, sim)
    assert "not the one the search chose (x = 1, y = 0), which breaks a limit there" in check["says"]
    # The improvement is the reported design's, at the search's settings too (10 - 6.5), not the search's best's.
    assert check["improvement"]["search"] == pytest.approx(3.5)


def test_without_finer_settings_the_best_design_file_does_not_say_it_was_checked_at_finer_settings(
        tmp_path: Path) -> None:
    block = _block(numerical_settings=None)
    outcome = osearch.run_sync(block, lambda c: {"f": (c["x"] - 2) ** 2}, seed=0)
    record = osearch.best_design(outcome, block, seed=0)
    check = oc.check_sync(block, record, outcome["rows"], lambda c: {"f": (c["x"] - 2) ** 2})
    ledger = "\n".join(osearch.ledger_lines(outcome, block)) + "\n"
    best_text = json.dumps(record)
    saved = json.dumps({"key": "k", "ledger": {"text": ledger}, "best": {"text": best_text}})
    run = optimise.SearchRun(record=record, rows=outcome["rows"],
                             files=optimise._files(ledger, best_text, saved))
    optimise.attach_check(tmp_path, run, check, json.dumps(check))
    assert run.record["checked_at_finer_settings"] is False
    assert "could not be checked at finer numerical settings" in run.record["check"]["says"]


@pytest.mark.asyncio
async def test_a_check_a_script_rewrote_on_disk_is_never_reused(tmp_path: Path) -> None:
    from tests.test_optimise_runner import LocalExecutor

    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    root = _runner_quest(tmp_path, ARTEFACT_SIM)
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                         simulate=root / "code" / "simulate.py",
                                         analysis=root / "code" / "experiment.py")
    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    await runner.execute(cmd, cwd=root, timeout_s=120)
    # A later script forges FI's kept copy of the check (keeping its key) to say "verified".
    kept_path = root / oc.RECORD
    kept = json.loads(kept_path.read_text(encoding="utf-8"))
    forged = {**json.loads(kept["text"]), "verdict": "verified", "says": "forged"}
    kept_path.write_text(json.dumps({"key": kept["key"], "text": json.dumps(forged)}), encoding="utf-8")
    (root / oc.CHECK_PATH).write_text(json.dumps(forged), encoding="utf-8")
    assert any("changed after FI wrote it" in g for g in oc.evidence_gaps(root, {"optimisation": block})[
        "independently_validated"]), "the ladder does not take a changed check on its word"
    await runner.execute(cmd, cwd=root, timeout_s=120)
    check = json.loads((root / oc.CHECK_PATH).read_text(encoding="utf-8"))
    assert check["verdict"] == "improvement_not_shown", "the forged check was not reused: FI ran its own again"
    assert not any("changed after FI wrote it" in g
                   for g in oc.evidence_gaps(root, {"optimisation": block}).get("independently_validated", []))


def test_with_a_threshold_a_design_that_failed_at_a_finer_level_is_still_not_verified() -> None:
    def sim(cell: dict[str, Any]) -> dict[str, float]:
        if (cell["x"], cell["y"]) != (0.0, 0.0) and cell["mesh"] == 0.2:
            raise RuntimeError("the solver diverged")
        return _bowl(cell)

    block = _block(improvement_tolerance=0.1)
    record, rows = _rows(block, [(0, {"x": 0.0, "y": 0.0}, 5.08, {}), (0, {"x": 2.0, "y": -1.0}, 0.08, {})])
    check = oc.check_sync(block, record, rows, sim)
    assert check["verdict"] == "unverified" and "diverged" in check["says"]


def test_without_a_numerical_setting_no_numerical_error_of_zero_is_claimed() -> None:
    check = _search_then_check(_block(numerical_settings=None), lambda c: {"f": (c["x"] - 2) ** 2 + (c["y"] + 1) ** 2})
    says = check["checks"]["improvement"]["says"]
    assert "numerical error was not estimated" in says and "(0 K)" not in says


@pytest.mark.asyncio
async def test_a_new_search_removes_an_earlier_searchs_check_and_other_scripts_get_it_put_back(tmp_path: Path) -> None:
    from tests.test_optimise_runner import LocalExecutor

    block = _block(design_variables=[{"name": "x", "low": -1, "high": 5, "kind": "integer"}],
                   baseline={"values": {"x": 0}, "source": "s"}, numerical_settings={"mesh": {"search": 0.5}},
                   evaluation_budget={"starts": 1, "per_start": 20}, search_method="exhaustive")
    root = _runner_quest(tmp_path, ARTEFACT_SIM)
    runner = optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                         simulate=root / "code" / "simulate.py",
                                         analysis=root / "code" / "experiment.py")
    await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=120)
    good = (root / oc.CHECK_PATH).read_bytes()
    # Another script run through the runner (a replot, say) overwrites the check: FI's copy is put back after it.
    (root / "code" / "replot.py").write_text(
        "open('needs/OPTIMUM_CHECK.json', 'w').write('{\"verdict\": \"verified\"}')\n", encoding="utf-8")
    await runner.execute([sys.executable, str(root / "code" / "replot.py")], cwd=root, timeout_s=60)
    assert (root / oc.CHECK_PATH).read_bytes() == good
    # A new search (the simulation changed) removes the earlier search's check before anything else.
    (root / "code" / "simulate.py").write_text(ARTEFACT_SIM.replace("60.0", "61.0"), encoding="utf-8")
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", {"optimisation": block},
                                    base_seed=0, timeout_s=300)
    assert run.reused is False and not (root / oc.CHECK_PATH).exists() and not (root / oc.RECORD).exists()


# --- run.log --------------------------------------------------------------------------------------------------------------


def test_the_log_lines_are_plain_and_say_where_the_finer_settings_come_from() -> None:
    record = _search_then_check(_block(numerical_settings={"mesh": {"search": 0.4, "check": [0.25, 0.1]}}), _bowl)
    text = "\n".join(oc.summary_lines(record))
    assert "check at finer numerical settings: The improvement over the baseline holds" in text
    assert "mesh: the plan" in text
    assert "apart from the search's budget" in text
    for label in ("finer settings", "limits", "better than the baseline", "starting points", "nearby designs",
                  "search budget"):
        assert f"{label}:" in text
