"""An oracle with a case is measured by the engine calling the simulation on it, not by a number the script reports.

The fixtures are the three RK4 variants of a real demo: a correct RK4, an Euler method named RK4, and an RK4 whose third
stage uses the wrong slope (order 2). All three pass a loose ``oracle()`` tolerance; only a run of the simulation on a
case with a tight tolerance tells them apart. Everything goes through the real trial runner in its own process."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import evidence, oracle_check as oc, plan, trial_runner
from core.execution import SharedInterpreterExecutor

_COMMON = '''
import math

def _integrate(step, dt):
    y, t = 1.0, 0.0
    n = int(round(1.0 / dt))
    for _ in range(n):
        y = step(y, dt)
    return y

def _f(y):
    return -y
'''

CORRECT = _COMMON + '''
def _rk4(y, h):
    k1 = _f(y); k2 = _f(y + h * k1 / 2); k3 = _f(y + h * k2 / 2); k4 = _f(y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def run_cell(cell):
    return {"error": abs(_integrate(_rk4, cell["dt"]) - math.exp(-1.0))}
'''

EULER_NAMED_RK4 = _COMMON + '''
def _rk4(y, h):  # named RK4, is Euler
    return y + h * _f(y)

def run_cell(cell):
    return {"error": abs(_integrate(_rk4, cell["dt"]) - math.exp(-1.0))}
'''

K3_BUG = _COMMON + '''
def _rk4(y, h):  # the third stage reuses k1: order 2
    k1 = _f(y); k2 = _f(y + h * k1 / 2); k3 = _f(y + h * k1 / 2); k4 = _f(y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def run_cell(cell):
    return {"error": abs(_integrate(_rk4, cell["dt"]) - math.exp(-1.0))}
'''

# The dishonest script: a wrong simulation whose own oracle() hands back the closed form without simulating.
CLOSED_FORM_ORACLE = EULER_NAMED_RK4 + '''
def oracle():
    return {"rk4 error": 0.0}
'''

TIGHT = {"name": "rk4 error", "kind": "closed_form", "check": "error at t=1 of y'=-y against exp(-1)", "expected": 0.0,
         "tolerance": 1e-5, "case": {"dt": 0.1}, "measure": "error", "order": 4}
LOOSE = {**TIGHT, "tolerance": 0.05}
NO_CASE = {k: v for k, v in TIGHT.items() if k not in ("case", "measure", "order")} | {"tolerance": 0.05}


def _measure(tmp_path: Path, source: str, oracles: list[dict[str, Any]], **kw: Any) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    checks, problems, _ = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", oracles, timeout_s=60, **kw))
    reported = {"checks": checks, "engine_measured": True}
    return checks, problems, oc.with_run_problems(problems, oc.problems(oracles, reported, 0), oracles)


@pytest.mark.parametrize("source, passes", [(CORRECT, True), (EULER_NAMED_RK4, False), (K3_BUG, False)])
def test_the_three_rk4_variants_are_told_apart_by_the_simulation_the_engine_runs(tmp_path: Path, source: str, passes: bool) -> None:
    checks, problems, found = _measure(tmp_path, source, [TIGHT])
    assert [c["measured_by"] for c in checks] == ["engine"] and not problems
    assert (found == []) is passes, (checks, found)


def test_a_loose_tolerance_passes_all_three_which_is_why_it_is_warned_about(tmp_path: Path) -> None:
    for i, source in enumerate((CORRECT, EULER_NAMED_RK4, K3_BUG)):
        _, _, found = _measure(tmp_path / str(i), source, [LOOSE])
        assert found == [], (i, found)
    assert "first-order method would pass" in (oc.loose_tolerance(LOOSE) or "")
    assert oc.loose_tolerance(TIGHT) is None


def test_a_script_whose_oracle_returns_the_closed_form_without_simulating_fails_once_the_oracle_has_a_case(tmp_path: Path) -> None:
    _, _, found = _measure(tmp_path / "with_case", CLOSED_FORM_ORACLE, [{**TIGHT, "tolerance": 1e-5}])
    assert len(found) == 1 and "failed" in found[0], found
    # Sabotage check: without a case the same script still passes, on its own word, and the record says whose word it was.
    checks, problems, found = _measure(tmp_path / "no_case", CLOSED_FORM_ORACLE, [NO_CASE])
    assert not problems and found == [] and [c["measured_by"] for c in checks] == ["script"]
    judged = oc.judged([NO_CASE], {"checks": checks})
    assert oc.script_measured(judged) == ["rk4 error"]


def test_an_oracle_the_simulation_cannot_answer_is_named_not_guessed(tmp_path: Path) -> None:
    _, problems, found = _measure(tmp_path / "a", CORRECT, [{**TIGHT, "measure": "no_such_number"}])
    assert len(problems) == 1 and "'no_such_number'" in problems[0] and "error" in problems[0]
    assert found == problems, "a missing value is reported by the case problem, not twice"
    _, problems, _ = _measure(tmp_path / "b", CORRECT, [{**TIGHT, "case": {"wrong_key": 1}}])
    assert len(problems) == 1 and "could not be run on its case" in problems[0] and "KeyError" in problems[0]


def test_cases_and_scripted_oracles_can_be_mixed(tmp_path: Path) -> None:
    checks, problems, found = _measure(tmp_path, CLOSED_FORM_ORACLE, [{**TIGHT, "name": "engine one"}, {**NO_CASE, "name": "rk4 error"}])
    by = {c["name"]: c["measured_by"] for c in checks}
    assert by == {"engine one": "engine", "rk4 error": "script"} and not problems
    assert len(found) == 1 and "'engine one'" in found[0], found


def test_the_looseness_warning_needs_a_claimed_order_a_step_and_a_tolerance_above_the_middle() -> None:
    assert oc.loose_tolerance({**LOOSE, "order": 1}) is None
    assert oc.loose_tolerance({k: v for k, v in LOOSE.items() if k != "order"}) is None
    assert oc.loose_tolerance({**LOOSE, "case": {"n": 10}}) is None
    assert oc.loose_tolerance({**LOOSE, "tolerance": 1e-4}) is None
    assert oc.loose_tolerance({**LOOSE, "tolerance": 0.004}) is not None  # above 0.1 ** 2.5 = 0.0032


def test_the_plan_keeps_a_case_and_leaves_out_a_field_that_is_not_usable_without_refusing_the_plan() -> None:
    kept, why = plan.normalize_protocol({"oracles": [TIGHT]})
    assert why is None and kept["oracles"][0]["case"] == {"dt": 0.1} and kept["oracles"][0]["order"] == 4
    kept, why = plan.normalize_protocol({"oracles": [{**TIGHT, "case": {"dt": 0.1, "adaptive": True, "y0": [1, 0]}}]})
    assert why is None and kept["oracles"][0]["case"]["adaptive"] is True
    for field, bad in (("case", [1, 2]), ("case", "a 10x10 lattice"), ("measure", 3), ("order", "four"), ("order", True)):
        kept, why = plan.normalize_protocol({"oracles": [{**TIGHT, field: bad}, {**TIGHT, "name": "other"}]})
        assert why is None and len(kept["oracles"]) == 2 and field not in kept["oracles"][0], (field, bad)


ENV_DETECTING = _COMMON + '''
import os

def run_cell(cell):
    if os.environ.get("FI_ORACLE") == "1":
        return {"error": 0.0}
    y = _integrate(lambda y, h: y + h * _f(y), cell["dt"])
    return {"error": abs(y - math.exp(-1.0))}
'''


def test_a_simulation_that_answers_differently_when_it_sees_the_oracle_variable_still_fails(tmp_path: Path) -> None:
    _, _, found = _measure(tmp_path, ENV_DETECTING, [TIGHT], env={"FI_ORACLE": "1"})
    assert len(found) == 1 and "failed" in found[0], found


def test_a_failed_case_is_not_filled_in_by_the_scripts_own_value_for_the_same_name(tmp_path: Path) -> None:
    checks, problems, _ = _measure(tmp_path, CLOSED_FORM_ORACLE, [{**TIGHT, "case": {"wrong_key": 1}}])
    assert problems and checks == []


def test_a_case_that_times_out_does_not_hide_the_failure_of_another_oracle(tmp_path: Path) -> None:
    slow = EULER_NAMED_RK4 + """
import time
_run = run_cell
def run_cell(cell):
    if cell['dt'] == 0.5:
        time.sleep(30)
    return _run(cell)
"""
    oracles = [{**TIGHT, "name": "wrong one"}, {**TIGHT, "name": "slow one", "case": {"dt": 0.5}}]
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(slow, encoding="utf-8")
    checks, problems, timed_out = asyncio.run(trial_runner.measure_oracles(
        SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py", oracles, timeout_s=5))
    found = oc.with_run_problems(problems, oc.problems(oracles, {"checks": checks, "engine_measured": True}, 0), oracles)
    assert any("'slow one'" in f for f in found) and any("'wrong one'" in f and "failed" in f for f in found), found


def test_only_the_engine_can_mark_a_value_as_engine_measured() -> None:
    spoofed = {"checks": [{"name": "x", "value": 0.0, "measured_by": "engine"}]}
    assert oc.judged([{"name": "x", "expected": 0.0, "tolerance": 1.0}], spoofed)[0]["measured_by"] == "script"


def _root(tmp_path: Path, record: dict[str, Any]) -> Path:
    from tests.test_evidence import PROTOCOL, _quest
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", protocol=PROTOCOL)
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def _judged(by: str) -> dict[str, Any]:
    return {"status": "ok", "judged_by": "engine", "contract": "trial",
            "attempts": [{"judged": [{"name": "closed form", "value": 0.0, "passed_by_engine": True, "measured_by": by}]}]}


def test_evidence_says_what_was_not_checked_when_the_value_came_from_the_script(tmp_path: Path) -> None:
    from tests.test_evidence import ON, _state
    scripted = evidence.assess(_root(tmp_path / "s", _judged("script")), _state(), settings=ON)
    assert scripted["levels"]["independently_validated"] is False
    assert any("closed form" in g and "not from the engine running the simulation" in g for g in scripted["all_gaps"]["independently_validated"])
    measured = evidence.assess(_root(tmp_path / "m", _judged("engine")), _state(), settings=ON)
    assert measured["levels"]["independently_validated"] is True


def test_a_value_the_script_reported_is_not_independent_on_a_one_script_quest_either(tmp_path: Path) -> None:
    """A one-script quest has no trial contract: its oracle value is what the script printed under FI_ORACLE=1. The rule is
    the same for every quest: only a value the engine measured counts, and the gap says how to get one."""
    from tests.test_evidence import ON, _state
    one_script = {k: v for k, v in _judged("script").items() if k != "contract"}
    got = evidence.assess(_root(tmp_path / "o", one_script), _state(), settings=ON)
    assert got["levels"]["independently_validated"] is False
    gap = " ".join(got["all_gaps"]["independently_validated"])
    assert "closed form" in gap and "execution.split_analysis: true" in gap
    unmarked = {"status": "ok", "judged_by": "engine",
                "attempts": [{"judged": [{"name": "closed form", "value": 0.0, "passed_by_engine": True}]}]}
    assert evidence.assess(_root(tmp_path / "u", unmarked), _state(), settings=ON)["levels"]["independently_validated"] is False


@pytest.mark.parametrize("attempts", [
    None,  # no attempts at all
    [],
    [{"judged": []}],  # the last attempt judged nothing
    [{"judged": [{"name": "closed form", "value": None, "passed_by_engine": None, "measured_by": "engine"}]}],  # no value
    [{"judged": [{"name": "closed form", "value": 0.0, "passed_by_engine": False, "measured_by": "engine"}]}],  # it failed
    [{"judged": [{"name": "closed form", "value": None, "passed_by_engine": None, "measured_by": "script"}]}],  # script, no value
    [{"judged": [{"value": 0.0, "passed_by_engine": True, "measured_by": "engine"}]}],  # an entry with no name is not read
    # an earlier attempt judged a value; the last one, the one the run went on from, judged nothing
    [{"judged": [{"name": "closed form", "value": 0.0, "passed_by_engine": True, "measured_by": "engine"}]}, {"judged": []}],
])
def test_an_ok_oracle_check_that_judged_no_value_is_not_independent_evidence(tmp_path: Path, attempts: Any) -> None:
    """An oracle record that says ``ok`` but holds no value the engine measured and passed checked nothing against the
    simulation: it must not reach independently validated, and the gap says so in plain words."""
    from tests.test_evidence import ON, _state
    record: dict[str, Any] = {"status": "ok", "judged_by": "engine", "contract": "trial"}
    if attempts is not None:
        record["attempts"] = attempts
    got = evidence.assess(_root(tmp_path, record), _state(), settings=ON)
    assert got["levels"]["independently_validated"] is False
    gaps = got["all_gaps"]["independently_validated"]
    if any(j.get("name") for j in (attempts or [{}])[-1].get("judged", [])):
        assert any("'closed form' has no measured value, or its value is outside" in g for g in gaps), gaps
    else:
        assert any("judged no value" in g for g in gaps), gaps


def test_every_declared_oracle_must_be_measured_not_just_one(tmp_path: Path) -> None:
    """A record that covers only some of the protocol's oracles (written before one was added to the plan) does not validate
    the ones it never measured."""
    from tests.test_evidence import ON, _quest, _state
    protocol = {"runs_per_setting": 300, "oracles": [{"name": "closed form"}, {"name": "limit"}]}
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", protocol=protocol)
    design = {"hypothesis": "h", "protocol": protocol}
    from tests.test_evidence import _write_passes
    _write_passes(root, design)
    got = evidence.assess(root, _state(design=design), settings=ON)
    assert got["levels"]["independently_validated"] is False
    assert any("did not measure 'limit'" in g for g in got["all_gaps"]["independently_validated"]), got["all_gaps"]
    # Both measured and passed (the name matched regardless of case and surrounding spaces): validated.
    both = {"status": "ok", "judged_by": "engine", "contract": "trial", "attempts": [{"judged": [
        {"name": "closed form", "value": 0.0, "passed_by_engine": True, "measured_by": "engine"},
        {"name": " LIMIT ", "value": 1.0, "passed_by_engine": True, "measured_by": "engine"},
    ]}]}
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(both), encoding="utf-8")
    assert evidence.assess(root, _state(design=design), settings=ON)["levels"]["independently_validated"] is True


def test_one_engine_measured_pass_among_several_oracles_is_still_needed_and_enough(tmp_path: Path) -> None:
    """Every value the engine measured and passed counts; an oracle with no value beside one that passed does not hide it,
    and a failed one beside a pass is a gap."""
    from tests.test_evidence import ON, _state
    passed = {"name": "closed form", "value": 0.0, "passed_by_engine": True, "measured_by": "engine"}
    failed = {"name": "limit", "value": 2.0, "passed_by_engine": False, "measured_by": "engine"}
    ok = {"status": "ok", "judged_by": "engine", "contract": "trial", "attempts": [{"judged": [passed]}]}
    assert evidence.assess(_root(tmp_path / "a", ok), _state(), settings=ON)["levels"]["independently_validated"] is True
    mixed = {**ok, "attempts": [{"judged": [passed, failed]}]}
    got = evidence.assess(_root(tmp_path / "b", mixed), _state(), settings=ON)
    assert got["levels"]["independently_validated"] is False
    assert any("limit" in g for g in got["all_gaps"]["independently_validated"])
