"""An equation whose output is a random draw is tested on its distribution (core/equation_tests.py): the mean of fixed-seed
draws against the mean the plan states, within five standard errors. Neutral models only: a count of arrivals, a noisy
reading, a straight-line rate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from core import equation_tests as eqt


def _protocol(formula: str, example: dict[str, Any], eid: str = "E1") -> dict[str, Any]:
    return {"model": {"summary": "a counting process", "equations": [
        {"id": eid, "formula": formula, "role": "generates", "source": "derivation", "example": example}]}}


COUNT = _protocol("N = Poisson(rate * t)", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t",
                                            "random": True})


def _results(tmp_path: Path, protocol: dict[str, Any], source: str, function: str = "arrivals") -> list[dict[str, Any]]:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "model.py").write_text(source, encoding="utf-8")
    rows = eqt.example_rows(protocol)
    located = {rows[0]["id"]: {"module": "pkg.model", "function": function, "file": "pkg/model.py"}}
    eqt.write(tmp_path, eqt.cases(rows, located))
    done = subprocess.run([sys.executable, "-B", str(tmp_path / eqt.TEST_PATH), "--json"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=240)
    found = eqt.parse_results(done.stdout)
    assert found is not None, done.stderr[-800:]
    return found


def test_a_random_equation_is_marked_by_the_plan_or_by_a_formula_that_names_a_distribution() -> None:
    assert eqt.example_rows(COUNT)[0]["random"] is True
    named = _protocol("N = Poisson(rate * t)", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t"})
    assert eqt.example_rows(named)[0]["random"] is True, "a formula that names a distribution is read as a draw"
    plain = _protocol("y = rate * t", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t"})
    row = eqt.example_rows(plain)[0]
    assert row["random"] is False and row["variance"] is None


def test_the_stated_variance_is_computed_by_fi_and_a_bad_one_is_left_out() -> None:
    good = _protocol("N = Poisson(rate * t)", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t",
                                               "random": True, "variance_formula": "rate * t"})
    assert eqt.example_rows(good)[0]["variance"] == 10.0
    bad = _protocol("N = Poisson(rate * t)", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t",
                                              "random": True, "variance_formula": "rate * nothing"})
    row = eqt.example_rows(bad)[0]
    assert row["state"] == "ok" and row["variance"] is None, "an unusable variance only means it is not checked"


def test_a_random_equation_without_a_usable_mean_is_not_tested_this_way_and_is_not_failed() -> None:
    none = _protocol("N = Poisson(rate * t)", {"inputs": {"rate": 4.0, "t": 2.5}, "random": True})
    row = eqt.example_rows(none)[0]
    assert row["state"] == "unusable" and "random draw" in row["why"] and "mean" in row["why"]
    notes = eqt.untested_notes([row], {})
    assert notes and "not tested before the run" in notes[0]
    assert eqt.cases([row], {"E1": {"module": "m", "function": "f", "file": "m.py"}}) == []
    stated = _protocol("N = Poisson(rate * t)", {"untestable": "no mean in closed form"})
    assert eqt.example_rows(stated)[0]["state"] == "unprotected", "untestable with no other check of the equation"
    covered = {**stated, "oracles": [{"name": "rate_limit", "kind": "special_case", "check": "E1 at t=0 gives 0",
                                      "expected": 0.0, "reference": "derivation: N = 0 when t = 0"}]}
    row = eqt.example_rows(covered)[0]
    assert row["state"] == "untestable" and row["covered_by"] == "rate_limit"


def test_a_correct_sampler_passes_whether_it_takes_a_generator_or_a_seed(tmp_path: Path) -> None:
    source = ("import random\n\n\n# E1\ndef arrivals(rate, t, rng):\n    return float(rng.poisson(rate * t))\n\n\n"
              "# E1\ndef by_seed(rate, t, seed):\n    r = random.Random(seed)\n    n, clock = 0, r.expovariate(rate)\n"
              "    while clock < t:\n        n += 1\n        clock += r.expovariate(rate)\n    return float(n)\n")
    assert _results(tmp_path, COUNT, source)[0]["status"] == eqt.OK
    assert _results(tmp_path, COUNT, source, "by_seed")[0]["status"] == eqt.OK


def test_a_sampler_with_the_wrong_mean_fails_and_the_message_never_holds_the_expected_value(tmp_path: Path) -> None:
    source = "# E1\ndef arrivals(rate, t, rng):\n    return float(rng.poisson(1.1 * rate * t))\n"
    [result] = _results(tmp_path, COUNT, source)
    assert result["status"] == eqt.MISMATCH and "standard errors" in result["detail"]
    text = eqt.failure_message({**result, "function": "arrivals"}, "N = Poisson(rate * t)")
    assert "standard errors" in text and "10.0" not in text and "4.0" not in text and "2.5" not in text


def test_a_stated_variance_is_held_too(tmp_path: Path) -> None:
    with_variance = _protocol("X = Normal(mu, s)", {"inputs": {"mu": 3.0, "s": 2.0}, "expected_formula": "mu",
                                                    "random": True, "variance_formula": "s ** 2"})
    right = "# E1\ndef reading(mu, s, rng):\n    return float(rng.normal(mu, s))\n"
    wide = "# E1\ndef reading(mu, s, rng):\n    return float(rng.normal(mu, 1.5 * s))\n"
    assert _results(tmp_path, with_variance, right, "reading")[0]["status"] == eqt.OK
    [bad] = _results(tmp_path, with_variance, wide, "reading")
    assert bad["status"] == eqt.MISMATCH and "variance" in bad["detail"]


def test_a_deterministic_function_is_never_judged_by_the_statistical_rule(tmp_path: Path) -> None:
    # The formula names a distribution, so the equation is read as a draw; the function does not vary, so the strict
    # relative 1e-6 rule applies: a result off by 1e-3 of its size fails, however many draws are taken.
    density = _protocol("p = Normal(x, mu, s)", {"inputs": {"x": 1.0, "mu": 0.0}, "expected_formula": "x - mu"})
    assert eqt.example_rows(density)[0]["random"] is True
    exact = "# E1\ndef level(x, mu):\n    return x - mu\n"
    off = "# E1\ndef level(x, mu):\n    return (x - mu) * 1.001\n"
    assert _results(tmp_path, density, exact, "level")[0]["status"] == eqt.OK
    assert _results(tmp_path, density, off, "level")[0]["status"] == eqt.MISMATCH


def test_a_plain_deterministic_equation_is_unchanged(tmp_path: Path) -> None:
    plain = _protocol("y = rate * t", {"inputs": {"rate": 4.0, "t": 2.5}, "expected_formula": "rate * t"})
    assert _results(tmp_path, plain, "# E1\ndef arrivals(rate, t):\n    return rate * t\n")[0]["status"] == eqt.OK
    assert _results(tmp_path, plain, "# E1\ndef arrivals(rate, t):\n    return rate * t * 1.0001\n")[0]["status"] == eqt.MISMATCH


def test_the_plan_is_told_how_to_mark_a_random_equation() -> None:
    assert "\"random\": true" in eqt.EXAMPLE_RULE and "MEAN" in eqt.EXAMPLE_RULE and "variance_formula" in eqt.EXAMPLE_RULE
