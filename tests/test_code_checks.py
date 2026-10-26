"""One call judges a code folder with FI's own checks and repairs nothing (core/code_checks.py).

A tiny neutral quest (a mass on a damped spring): a model package whose two functions carry the plan's equations, a
simulation that calls them, a plotting script and a protocol with a grid, two worked examples and one known-answer check.
The clean folder passes every check; each planted fault is one edit of the clean folder and fails the check it is for.
The verdicts are looked up by name, so a check dropped from ``check_code`` fails the lookup, not just an assertion
about a count. Nothing here calls a model."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from core import code_checks
from core.execution import SharedInterpreterExecutor

# Values the checks must never say: the examples' expected outputs (5.0 and 12.0), the known-answer check's expected
# value (8.0) and its tolerance.
EXPECTED_VALUES = ("5.0", "12.0", "8.0")
TOLERANCE = 1e-6

PROTOCOL: dict[str, Any] = {
    "grid": {"c": [0.5, 1.0], "extra": [1, 2]},
    "model": {"summary": "a mass on a damped spring", "equations": [
        {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton's law",
         "example": {"inputs": {"k": 50.0, "m": 2.0}, "expected_formula": "sqrt(k / m)"}},
        {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation", "derivation": "energy ratio",
         "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}},
    ]},
    "oracles": [{"name": "quality at one damping", "kind": "closed_form", "check": "Q = m * w / c at c = 0.75",
                 "expected": 8.0, "tolerance": TOLERANCE, "case": {"c": 0.75, "extra": 1}, "measure": "q"}],
}

MODEL = '''import math


# E1
def frequency(k, m):
    return math.sqrt(k / m)


# E2
def quality(m, w, c):
    return m * w / c
'''
SIMULATE = '''from spring import model


def run_cell(cell):
    return {"q": model.quality(2.0, 3.0, cell["c"]) * cell["extra"]}
'''
PLOT = '''import json

import matplotlib.pyplot as plt

values = json.load(open("results.json"))
plt.plot(values, values)
plt.savefig("figure.png")
'''

CLEAN = {"spring/__init__.py": "from . import model\n", "spring/model.py": MODEL, "simulate.py": SIMULATE, "plot.py": PLOT}


def _with(**edits: tuple[str, str]) -> dict[str, str]:
    """The clean folder with ``old -> new`` applied once in each named file (``simulate``, ``model``, ``plot``)."""
    files = dict(CLEAN)
    names = {"simulate": "simulate.py", "model": "spring/model.py", "plot": "plot.py"}
    for key, (old, new) in edits.items():
        rel = names[key]
        assert old in files[rel], (rel, old)
        files[rel] = files[rel].replace(old, new, 1)
    return files


# One planted fault per check: the folder, and the checks that fail on it (no other).
FAULTS: dict[str, tuple[dict[str, str], set[str]]] = {
    "typed_results": (_with(simulate=('"q": model.quality(2.0, 3.0, cell["c"]) * cell["extra"]}',
                                      '"q": model.quality(2.0, 3.0, cell["c"]) * cell["extra"], "drag": 0.91827}')),
                      {"typed_results"}),
    "grid_keys": (_with(simulate=('* cell["extra"]}', '* cell["extra"] + 0.0 * cell.get("extr", 0.0)}')), {"grid_keys"}),
    "typed_constants": (_with(model=("    return math.sqrt(k / m)\n",
                                     "    pad = 0.0 * 0.37 + 0.0 * 0.81 + 0.0 * 0.29\n    return math.sqrt(k / m) + pad\n")),
                        {"typed_constants"}),
    "small_run": (_with(simulate=("from spring import model\n", "from spring import model\nimport module_that_is_not_there\n")),
                  {"small_run", "known_answers"}),
    "trial_checks": (_with(simulate=('    return {"q"', '    if cell["extra"] == 2:\n        raise ValueError("no solution here")\n    return {"q"')),
                     {"trial_checks"}),
    "equation_tests": (_with(model=("math.sqrt(k / m)", "math.sqrt(k * m)")), {"equation_tests"}),
    "known_answers": (_with(simulate=('cell["c"])', 'cell["c"] * 2)')), {"known_answers"}),
    "unread_settings": (_with(simulate=(' * cell["extra"]}', '}')), {"unread_settings"}),
    "figure_data": (_with(plot=("plt.plot(values, values)", "plt.plot([1, 2, 3], [4, 5, 6])")), {"figure_data"}),
}
ROSTER = ["typed_results", "undefined_names", "grid_keys", "typed_constants", "independent_check", "small_run", "trial_checks",
          "equation_tests", "known_answers", "unread_settings", "figure_data"]


def _judge(root: Path, files: dict[str, str]) -> list[code_checks.Verdict]:
    code = root / "code"
    for rel, text in files.items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
    return asyncio.run(code_checks.check_code(root, PROTOCOL, executor=SharedInterpreterExecutor(python_version="3.11"),
                                              python=sys.executable, timeout_s=90))


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[code_checks.Verdict]]:
    out = {"clean": _judge(tmp_path_factory.mktemp("clean"), CLEAN)}
    for name, (files, _failing) in FAULTS.items():
        out[name] = _judge(tmp_path_factory.mktemp(name), files)
    return out


def _verdict(verdicts: list[code_checks.Verdict], name: str) -> code_checks.Verdict:
    found = [v for v in verdicts if v.check == name]
    assert len(found) == 1, f"the check {name} gave {len(found)} verdicts: {[v.check for v in verdicts]}"
    return found[0]


def test_the_clean_folder_passes_every_check(runs: dict[str, list[code_checks.Verdict]]) -> None:
    verdicts = runs["clean"]
    assert [v.check for v in verdicts] == ROSTER
    for name in ROSTER:
        got = _verdict(verdicts, name)
        assert got.passed is True, (name, got.says, got.details)
        assert got.says and "\n" not in got.says


@pytest.mark.parametrize("fault", list(FAULTS))
def test_a_planted_fault_fails_its_check_and_no_other(runs: dict[str, list[code_checks.Verdict]], fault: str) -> None:
    verdicts = runs[fault]
    assert [v.check for v in verdicts] == ROSTER
    failing = {v.check for v in verdicts if v.passed is False}
    named = FAULTS[fault][1]
    assert failing == named, {v.check: (v.passed, v.says) for v in verdicts}
    for name in named:
        assert _verdict(verdicts, name).passed is False


def test_a_load_error_leaves_the_failed_trials_unjudged(runs: dict[str, list[code_checks.Verdict]]) -> None:
    got = _verdict(runs["small_run"], "trial_checks")
    assert got.passed is None and "no trials" in got.says


def test_the_failed_trials_are_named_by_setting(runs: dict[str, list[code_checks.Verdict]]) -> None:
    says = _verdict(runs["trial_checks"], "trial_checks").says
    assert "extra=2" in says and "no solution here" in says


def test_the_equation_that_fails_is_named_with_the_value_it_gave(runs: dict[str, list[code_checks.Verdict]]) -> None:
    got = _verdict(runs["equation_tests"], "equation_tests")
    assert "equation E1" in got.says and "`frequency`" in got.says and "10" in got.says
    assert [f["id"] for f in got.details["failed"]] == ["E1"]


def test_a_known_answer_names_the_check_and_the_value_measured(runs: dict[str, list[code_checks.Verdict]]) -> None:
    got = _verdict(runs["known_answers"], "known_answers")
    assert "quality at one damping" in got.says and "measured 4" in got.says


def test_no_verdict_says_an_expected_value_or_a_tolerance(runs: dict[str, list[code_checks.Verdict]]) -> None:
    for name, verdicts in runs.items():
        for v in verdicts:
            text = v.says + " " + json.dumps(v.details, default=str)
            for value in EXPECTED_VALUES:
                assert not re.search(rf"(?<![\d.]){re.escape(value)}(?!\d)", text), (name, v.check, value, text[:300])
            assert not re.search(r"1e-0?6|0\.000001", text), (name, v.check, text[:300])


def test_the_judging_changes_nothing_in_the_code_folder(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    before = {}
    code = root / "code"
    for rel, text in FAULTS["equation_tests"][0].items():
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text(text, encoding="utf-8")
        before[rel] = text
    asyncio.run(code_checks.check_code(root, PROTOCOL, executor=SharedInterpreterExecutor(python_version="3.11"),
                                       python=sys.executable, timeout_s=90))
    after = {p.relative_to(code).as_posix(): p.read_text(encoding="utf-8") for p in code.rglob("*.py")
             if "__pycache__" not in p.parts}
    assert after == before


def test_a_folder_with_nothing_to_judge_gives_verdicts_that_say_so(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    verdicts = asyncio.run(code_checks.check_code(root, None, executor=SharedInterpreterExecutor(python_version="3.11"),
                                                  python=sys.executable))
    assert [v.check for v in verdicts] == ROSTER
    assert all(v.passed is not False for v in verdicts), [(v.check, v.says) for v in verdicts]


def test_a_check_that_raises_is_a_verdict_that_could_not_be_judged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("broken reader")

    monkeypatch.setattr(code_checks._figdata, "typed_series_in", boom)
    verdicts = _judge(tmp_path / "quest", CLEAN)
    got = _verdict(verdicts, "figure_data")
    assert got.passed is None and "broken reader" in got.says
    assert _verdict(verdicts, "typed_results").passed is True
