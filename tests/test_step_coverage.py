"""Step coverage (core/step_coverage.py): is each metric the simulation reports computed through the model's package?
Checked on the s1 decay / screening code (tests/fixtures/agent_cheats) and its fake-truth cheats."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from core import step_coverage as sc

FIXTURES = Path(__file__).parent / "fixtures" / "agent_cheats"
PACKAGES = {"decay": "decay_model", "screening": "screening_strict_vs_adaptive"}


def _inputs(topic: str, simulate: Path | None = None) -> tuple[str, list[str], list]:
    clean = FIXTURES / topic / "clean"
    model = (clean / PACKAGES[topic] / "model.py").read_text(encoding="utf-8")
    functions = [n.name for n in ast.parse(model).body if isinstance(n, ast.FunctionDef)]
    metrics = json.loads((FIXTURES / topic / "protocol.json").read_text(encoding="utf-8"))["metrics"]
    return (simulate or clean / "simulate.py").read_text(encoding="utf-8"), functions, metrics


@pytest.mark.parametrize("topic", ["decay", "screening"])
def test_the_clean_s1_code_raises_no_alarm(topic: str) -> None:
    assert sc.check(*_inputs(topic)) == []


def test_the_fake_truth_cheat_is_caught_metric_by_metric() -> None:
    warnings = sc.check(*_inputs("decay", FIXTURES / "cheats" / "fake_truth" / "code" / "simulate.py"))
    caught = {w.split("`")[1] for w in warnings}
    # Every metric the plan asks for, the ones run_trial returns and the ones run_cell builds from them.
    assert caught == {"coverage_Ea", "coverage_lnA", "Ea_mean", "lnA_mean", "Ea_sd", "corr_Ea_lnA"}
    assert all("without calling any function of the model's package" in w for w in warnings)
    screening = sc.check(*_inputs("screening", FIXTURES / "cheats" / "fake_truth_screening" / "code" / "simulate.py"))
    assert {w.split("`")[1] for w in screening} >= {"fhr_strict", "power_strict", "power_gap_pp"}


def test_a_typed_metric_is_named_as_typed() -> None:
    warnings = sc.check(*_inputs("decay", FIXTURES / "cheats" / "typed_result" / "code" / "simulate.py"))
    assert warnings == ["the metric `coverage_Ea` is returned as a number written into the code, not computed"]


def test_a_value_unpacked_from_a_package_call_counts_as_computed_through_it() -> None:
    """The false alarm the clean s1 code raised before: ``a, b = pkg.f(...)`` and ``d[key] = v`` were not followed."""
    source = (
        "from pkg import model\n\n"
        "def run_trial(cell, trial_id, seed):\n"
        "    a, b = model.fit(cell)\n"
        "    tally = {}\n"
        "    for name in ('x', 'y'):\n"
        "        _v, t = model.count(name)\n"
        "        tally[name] = t\n"
        "    return {'a': a, 'gap': tally['x'] - tally['y'], 'typed': 0.5, 'raw': cell['n'] * 2.0}\n")
    warnings = sc.check(source, ["fit", "count"], ["a", "gap", "typed", "raw"])
    assert [w.split("`")[1] for w in warnings] == ["typed", "raw"]


def test_one_token_call_is_enough_to_satisfy_it() -> None:
    """The residual risk the design names: step coverage reads the syntax, so a package call multiplied by 0 passes."""
    assert sc.check(*_inputs("decay", FIXTURES / "cheats" / "fake_truth_token_call" / "code" / "simulate.py")) == []


@pytest.mark.parametrize("topic", ["decay", "screening"])
def test_the_clean_s1_code_is_bound_to_its_package_by_its_imports(topic: str) -> None:
    """With the package's folder name, a call counts only through an import (the form the engine uses); the clean code, which
    also assigns attributes of the module (`model.C0 = ...`), still raises no alarm."""
    source, functions, metrics = _inputs(topic)
    assert sc.check(source, functions, metrics, package_modules=[PACKAGES[topic]]) == []
    cheat = "fake_truth" if topic == "decay" else "fake_truth_screening"
    fake = (FIXTURES / "cheats" / cheat / "code" / "simulate.py").read_text(encoding="utf-8")
    assert sc.check(fake, functions, metrics, package_modules=[PACKAGES[topic]])
