"""Replay of a real quest that stopped because a random equation was tested as a deterministic one.

In tests/fixtures/replay/random_term_exact/ the plan's first equation said a term follows a distribution (``epsilon_j~N(0,1)``)
and gave its mean as the worked example (``expected_formula: mu``, ``variance_formula: '1'``); the plan wrote the flag
``random: true`` beside the example instead of inside it. FI read it as a deterministic function, wrote the case with
``"random": false`` and ``"expected": 2.5``, and a correct implementation (the input plus one standard normal draw) could
never pass: the function was repaired 6 times, then the whole script, then the design, and the log gave no reason. No model
is called."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from core import equation_tests as eqt

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "random_term_exact"
PROTOCOL = yaml.safe_load((FIXTURE / "model_block.yaml").read_text(encoding="utf-8"))
REAL_CASE = json.loads((FIXTURE / "test_case.json").read_text(encoding="utf-8"))
REAL_FUNCTION = (FIXTURE / "z_statistic.py").read_text(encoding="utf-8")


def _results(tmp_path: Path, source: str, protocol: dict[str, Any] = PROTOCOL, eid: str = "E1") -> dict[str, Any]:
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "model.py").write_text(source, encoding="utf-8")
    rows = eqt.example_rows(protocol)
    located = {eid: {"module": "pkg.model", "function": "z_statistic", "file": "pkg/model.py"}}
    case_list = eqt.cases(rows, located)
    eqt.write(tmp_path, case_list)
    eqt.write_values(tmp_path, case_list)   # a person's copy, delivered at the end of a quest
    done = subprocess.run([sys.executable, "-B", str(tmp_path / eqt.TEST_PATH), "--json"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=240)
    found = eqt.parse_results(done.stdout)
    assert found is not None, done.stderr[-800:]
    assert "2.5" not in done.stdout, "the runner's output names no number of the worked example"
    from tests.equation_judge_helpers import fi_verdict

    fi = next(r for r in fi_verdict(case_list, tmp_path) if r["id"] == eid)
    person = next(r for r in found if r["id"] == eid)
    assert person["status"] == fi["status"], "a person's copy of the test agrees with FI's verdict"
    return fi


def test_the_evidence_is_what_the_real_quest_held() -> None:
    assert REAL_CASE["random"] is False and REAL_CASE["expected"] == 2.5 and REAL_CASE["variance"] is None
    first = PROTOCOL["model"]["equations"][0]
    assert "~N(0,1)" in first["formula"] and first["random"] is True and "random" not in first["example"]


def test_the_equation_that_states_a_distribution_is_a_random_one_with_its_stated_mean_and_variance() -> None:
    row = eqt.example_rows(PROTOCOL)[0]
    assert row["state"] == "ok" and row["random"] is True
    assert row["expected"] == 2.5 and row["variance"] == 1.0


def test_a_formula_that_says_a_term_follows_a_distribution_is_read_as_random_without_any_flag() -> None:
    for formula in ("y = a + e, e ~ N(0, 1)", "y = a + e where e ~ Normal", "y = a + u, u ~ U(0, 1)",
                    "y = a + e with e∼Poisson(2)", "y = a + e, e is drawn from a wide band", "y = a + random noise",
                    "y = a + e, e distributed as a standard normal"):
        assert eqt.is_random({"formula": formula}), formula
    for formula in ("y = a * b", "y = exp(-k * t)", "FDR_BH = pi0 * 0.05", "p = 1 - Phi(z)",
                    "y = a * b with the random seed fixed", "n = number of non-random walks"):
        assert not eqt.is_random({"formula": formula}), formula


def test_a_correct_random_implementation_passes(tmp_path: Path) -> None:
    assert _results(tmp_path, REAL_FUNCTION)["status"] == "ok"


def test_a_deterministic_implementation_that_is_off_by_one_still_fails(tmp_path: Path) -> None:
    result = _results(tmp_path, "# E1\ndef z_statistic(mu):\n    return mu + 1.0\n")
    assert result["status"] == "mismatch" and result["got"] == 3.5 and result["expected"] == 2.5


def test_a_wrong_mean_fails(tmp_path: Path) -> None:
    source = ("import numpy as np\n\n# E1\ndef z_statistic(mu, rng=None):\n    rng = rng or np.random.default_rng()\n"
              "    return float(mu + 0.5 + rng.standard_normal())\n")
    assert _results(tmp_path, source)["status"] == "mismatch"


def test_a_wrong_spread_fails_when_the_variance_is_stated(tmp_path: Path) -> None:
    source = ("import numpy as np\n\n# E1\ndef z_statistic(mu, rng=None):\n    rng = rng or np.random.default_rng()\n"
              "    return float(mu + 2.0 * rng.standard_normal())\n")
    assert _results(tmp_path, source)["status"] == "mismatch"


def test_a_function_that_never_varies_is_held_to_the_exact_value(tmp_path: Path) -> None:
    ok = _results(tmp_path, "# E1\ndef z_statistic(mu):\n    return mu\n")
    assert ok["status"] == "ok", "a function that returns exactly the mean does not vary: the strict rule, and it agrees"


def test_a_random_equation_without_a_stated_mean_is_not_held_to_an_exact_value() -> None:
    protocol = yaml.safe_load(yaml.safe_dump(PROTOCOL))
    del protocol["model"]["equations"][0]["example"]["expected_formula"]
    row = eqt.example_rows(protocol)[0]
    assert row["state"] == "unusable" and "exact" in row["why"] and "mean" in row["why"]
    assert eqt.cases([row], {"E1": {"module": "m", "function": "f", "file": "m.py"}}) == []


def test_the_failure_reason_states_the_expected_value_the_value_got_and_the_tolerance(tmp_path: Path) -> None:
    result = _results(tmp_path, "# E1\ndef z_statistic(mu):\n    return mu + 1.0\n")
    reason = eqt.failure_reason(result)
    assert "got 3.5" in reason and "expected 2.5" in reason and "tolerance" in reason
    # the request to the model that repairs the function still never carries the expected value
    message = eqt.failure_message(result, "Z = mu + e")
    assert "2.5" not in message and "3.5" not in message
    assert eqt.failure_reason({"status": "error", "detail": "KeyError: 'x'"}) == "error: KeyError: 'x'"
