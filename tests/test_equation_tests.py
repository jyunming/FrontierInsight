"""A small test per equation of the plan's model, written by FI from the plan's own worked example
(core/equation_tests.py). Neutral models only: a cooling cup, a damped spring."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import equation_tests as eqt

# A damped spring: the quality factor Q = sqrt(m k) / c. The wrong version divides by a different quantity.
SPRING = {"summary": "a mass on a damped spring", "equations": [
    {"id": "E1", "formula": "Q = sqrt(m * k) / c", "role": "generates", "source": "derivation",
     "derivation": "the ratio of stored to lost energy per radian",
     "example": {"inputs": {"m": 2.0, "k": 8.0, "c": 0.5}, "expected_formula": "sqrt(m * k) / c"}},
    {"id": "E2", "formula": "T = 2 * pi / sqrt(k / m)", "role": "analyses", "source": "derivation",
     "example": {"untestable": "the period is read from a simulated trace, not computed in closed form"}},
    {"id": "E3", "formula": "x(t) = x0 * exp(-g * t)", "role": "generates", "source": "derivation"},
    {"id": "E4", "formula": "loss = c * v", "role": "narrates", "source": "derivation"},
]}
PROTOCOL = {"model": SPRING}


def _row(rows: list[dict[str, Any]], eid: str) -> dict[str, Any]:
    return next(r for r in rows if r["id"] == eid)


def test_a_worked_example_is_computed_by_fi_and_what_is_missing_or_unusable_is_named() -> None:
    rows = eqt.example_rows(PROTOCOL)
    assert [r["id"] for r in rows] == ["E1", "E2", "E3"], "an equation with another role gets no example"
    ok = _row(rows, "E1")
    assert ok["state"] == "ok" and ok["expected"] == pytest.approx(8.0) and ok["inputs"] == {"m": 2.0, "k": 8.0, "c": 0.5}
    assert _row(rows, "E2")["state"] == "untestable" and "simulated trace" in _row(rows, "E2")["why"]
    assert _row(rows, "E3")["state"] == "missing"
    bad = {"model": {"equations": [
        {"id": "E1", "formula": "y = a * b", "role": "generates",
         "example": {"inputs": {"a": 2.0}, "expected_formula": "a * b"}},  # b is not an input
        {"id": "E2", "formula": "y = a", "role": "generates", "example": {"inputs": {"a": "two"}, "expected_formula": "a"}},
        {"id": "E3", "formula": "y = a", "role": "generates", "example": "see the paper"},
    ]}}
    rows = eqt.example_rows(bad)
    assert [r["state"] for r in rows] == ["unusable"] * 3
    assert "uses b, which are not among its `inputs`" in rows[0]["why"]
    text = eqt.request([r for r in rows if r["state"] != "ok"])
    assert "relative 1e-6" in text and "E1" in text and "Change only these equations' `example`" in text
    assert eqt.request(eqt.example_rows({"model": {"equations": SPRING["equations"][:1]}})) == ""


def test_the_tolerance_is_a_rule_a_closed_form_is_relative_1e_6_and_a_method_brings_its_own() -> None:
    def row(example: dict[str, Any]) -> dict[str, Any]:
        base = {"inputs": {"a": 2.0}, "expected_formula": "a ** 2"}
        return eqt.example_rows({"model": {"equations": [
            {"id": "E1", "formula": "y = a^2", "role": "generates", "example": {**base, **example}}]}})[0]

    closed = row({})
    assert closed["rel"] == eqt.RELATIVE == 1e-6 and closed["abs"] == eqt.ABSOLUTE_FLOOR
    assert row({"tolerance": 1e-3})["rel"] == 1e-6, "a stated tolerance without a method never loosens the test"
    assert row({"tolerance": 1e-9})["rel"] == 1e-9, "but it may tighten it"
    stepped = row({"method": "explicit Euler, dt = 0.01", "tolerance": 5e-3, "tolerance_mode": "relative"})
    assert stepped["state"] == "ok" and stepped["rel"] == 5e-3 and stepped["method"].startswith("explicit Euler")
    absolute = row({"method": "bisection", "tolerance": 1e-4, "tolerance_mode": "absolute"})
    assert absolute["rel"] == 0.0 and absolute["abs"] == 1e-4
    no_tol = row({"method": "explicit Euler, dt = 0.01"})
    assert no_tol["state"] == "unusable" and "no tolerance of its own" in no_tol["why"], "the method's setting is never guessed"


def _write_package(code: Path, model_source: str) -> None:
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "spring" / "model.py").write_text(model_source, encoding="utf-8")


CORRECT = "import math\n\n\n# E1\ndef quality(m, k, c):\n    return math.sqrt(m * k) / c\n"
WRONG = "import math\n\n\n# E1\ndef quality(m, k, c):\n    return math.sqrt(m / k) / c\n"  # a different normalisation
RENAMED = "import math\n\n\n# E1\ndef quality(mass, k, c):\n    return math.sqrt(mass * k) / c\n"


def _run(code: Path) -> tuple[list[dict[str, Any]], str]:
    shutil.rmtree(code / "spring" / "__pycache__", ignore_errors=True)  # the same size, rewritten in the same second
    done = subprocess.run([sys.executable, "-B", eqt.TEST_PATH, "--json"], cwd=code, capture_output=True, text=True, timeout=60)
    return eqt.parse_results(done.stdout) or [], done.stdout + done.stderr


def _cases(code: Path) -> list[dict[str, Any]]:
    sources = {"spring/model.py": (code / "spring" / "model.py").read_text(encoding="utf-8")}
    located = eqt.locate(PROTOCOL, sources)
    assert located["E1"] == {"module": "spring.model", "function": "quality", "file": "spring/model.py"}
    assert located["E2"] is None and located["E3"] is None
    return eqt.cases(eqt.example_rows(PROTOCOL), located)


def test_an_equation_implemented_wrongly_is_caught_and_a_correct_one_passes(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_package(code, CORRECT)
    eqt.write(code, _cases(code))
    results, _out = _run(code)
    assert [(r["id"], r["status"]) for r in results] == [("E1", "ok")]
    (code / "spring" / "model.py").write_text(WRONG, encoding="utf-8")
    results, out = _run(code)
    assert [(r["id"], r["status"]) for r in results] == [("E1", "mismatch")]
    assert "8.0" not in out and "0.5" not in out and "expected" not in out.lower().replace("expected_formula", ""), (
        "the runner's output names no number of the worked example")
    (code / "spring" / "model.py").write_text(RENAMED, encoding="utf-8")
    results, _out = _run(code)
    assert results[0]["status"] == "contract" and "m" in results[0]["detail"].split("take ")[1]
    (code / "spring" / "model.py").write_text("def quality(m, k, c):\n    # E1\n    raise ValueError('nope')\n", encoding="utf-8")
    assert _run(code)[0][0]["status"] == "error"


def test_the_generated_file_is_also_a_pytest_file_and_is_fi_s_not_the_model_s(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_package(code, WRONG)
    path = eqt.write(code, _cases(code))
    text = path.read_text(encoding="utf-8")
    assert "Written by Frontier Insight (never by the model" in text and "def test_e1():" in text
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)], cwd=code,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode != 0 and "mismatch" in done.stdout, done.stdout[-800:]
    assert eqt.write(code, []) is None and not path.exists(), "no case: no file"


def test_a_failure_is_reported_by_its_function_and_equation_never_by_a_number() -> None:
    result = {"id": "E1", "function": "quality", "status": "mismatch", "detail": ""}
    text = eqt.failure_message(result, "Q = sqrt(m * k) / c")
    assert "`quality`" in text and "equation E1" in text and "Q = sqrt(m * k) / c" in text
    assert "8.0" not in text and "2.0" not in text
    contract = eqt.failure_message({**result, "status": "contract", "detail": "the function must take c as arguments, named so"}, "f")
    assert "must take c" in contract


def test_what_cannot_be_tested_is_said_plainly() -> None:
    rows = eqt.example_rows(PROTOCOL)
    notes = eqt.untested_notes(rows, {"E1": None})
    assert any("E1" in n and "no function of the code that can be imported" in n for n in notes)
    assert any("E2" in n and "no closed form" in n for n in notes)
    assert any("E3" in n and "no worked example" in n for n in notes)


def test_a_function_in_the_scenario_script_is_found_and_a_method_is_not() -> None:
    sources = {"simulate.py": "# E1\ndef quality(m, k, c):\n    return 1\n\n\nclass A:\n    # E3\n    def f(self):\n        pass\n"}
    found = eqt.locate(PROTOCOL, sources)
    assert found["E1"]["module"] == "simulate" and found["E3"] is None
    assert eqt.module_of("pkg/model.py") == "pkg.model" and eqt.module_of("pkg/__init__.py") == "pkg"


def test_the_design_prompt_and_the_outline_prompt_state_the_contract() -> None:
    from core import engine

    assert "worked example" in engine._PLAN_DIRECTIVE and '"example": {"inputs"' in engine._PLAN_DIRECTIVE
    outline = (Path(__file__).resolve().parents[1] / "agents" / "implement_outline.md").read_text(encoding="utf-8")
    assert "`implements`" in outline and "`depends_on`" in outline and "example.inputs" in outline
    json.dumps(eqt.EXAMPLE_RULE)
