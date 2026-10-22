"""One source for a function's interface (core/equation_tests.py, core/function_steps.py, core/code_layout.py,
core/oracle_check.py, core/run_estimate.py, core/second_impl.py).

A function's interface used to have several sources that disagreed: the outline's signature, the plan's worked example,
the code's own parameters, and a `# E<n>` comment. Equation tests and code repairs then fought over them. Each test below
builds the disagreement from a neutral model (a mass on a spring, a cooling cup) and shows the one source winning, and
each has the planted fault that must still be caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core import code_layout as cl, equation_measure as em, equation_tests as eqt, function_steps as fs
from core import oracle_check as oc, run_estimate as re_, second_impl as si
from tests.test_equation_gate import GOOD, OUTLINE, PROTOCOL

OUTLINE_NAMES = {"E1": "frequency", "E2": "quality"}


# --- the outline's function is the one source of which function an equation test calls; no label is looked for ----------


def test_an_equation_the_outline_names_a_function_for_is_never_looked_for_by_label() -> None:
    renamed = GOOD["frequency"] + "\n\n# E2\ndef q_factor(m, w, c):\n    return m * w / c\n"
    sources = {"spring/model.py": renamed}
    located = eqt.locate(PROTOCOL, sources, OUTLINE_NAMES)
    assert located["E1"]["function"] == "frequency"
    assert located["E2"] is None, "the outline's `quality` is not in the code; the label on `q_factor` is not a second source"
    # Known positive: an equation the outline names no function for is still found by its label.
    assert eqt.locate(PROTOCOL, sources, {"E1": "frequency"})["E2"]["function"] == "q_factor"
    rows = eqt.example_rows(PROTOCOL)
    notes = eqt.untested_notes(rows, located, OUTLINE_NAMES)
    assert notes == ["equation E2 is not tested before the run: the code has no function `quality`, the one planned for it"]


def test_the_outline_names_the_function_so_no_label_is_asked_for_and_none_is_missing() -> None:
    bare = {"spring/model.py": "def frequency(k, m):\n    return (k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n"}
    assert oc.unlabelled_equations(PROTOCOL, bare) == ["E1", "E2"], "without the outline: labels are missing, as before"
    assert oc.unlabelled_equations(PROTOCOL, bare, OUTLINE_NAMES) == []
    assert oc.label_gaps(PROTOCOL, bare, "model.py", OUTLINE_NAMES) == []
    # Known positive: a function the outline names that the code does not define still needs its label.
    assert oc.unlabelled_equations(PROTOCOL, bare, {"E1": "frequency", "E2": "q_factor"}) == ["E2"]
    rows = cl.equation_map(PROTOCOL, bare, OUTLINE_NAMES)
    assert [r["function"] for r in rows] == ["frequency", "quality"]


def test_the_layout_check_does_not_ask_for_labels_where_the_outline_names_the_functions(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "spring" / "model.py").write_text(
        "def frequency(k, m):\n    return (k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n", encoding="utf-8")
    (code / "simulate.py").write_text("from spring import model\n", encoding="utf-8")
    without = [p for p in cl.check(code, PROTOCOL, "spring") if "not labelled" in p]
    assert without and "E1, E2" in without[0]
    named = cl.check(code, PROTOCOL, "spring", OUTLINE_NAMES)
    assert not [p for p in named if "not labelled" in p or "planned for equation" in p]
    absent = cl.check(code, PROTOCOL, "spring", {"E1": "frequency", "E2": "q_factor"})
    assert any("planned for equation E2" in p and "not in code/spring/" in p for p in absent)


# --- a repair is held to ONE interface: the outline's where the test cannot call the code yet, else the code's ------------

# The code was first written with the example's parameter NAMES in another ORDER than the signature built from the example
# (the shape of a recorded round: the tests pass, and every whole-file repair was refused on order alone).
REORDERED = "def frequency(m, k):\n    return (k / m) ** 0.5\n\n\ndef quality(c, w, m):\n    return m * w / c\n"
OUTLINE_SIGS = {"frequency": "def frequency(*, k, m) -> float:", "quality": "def quality(*, m, w, c) -> float:"}


def _cases(source: str) -> list[dict]:
    sources = {"spring/model.py": source}
    return eqt.cases(eqt.example_rows(PROTOCOL), eqt.locate(PROTOCOL, sources, OUTLINE_NAMES))


def test_code_the_test_can_already_call_keeps_its_own_def_even_when_only_the_order_differs() -> None:
    sources = {"spring/model.py": REORDERED}
    held = fs.held_signatures(OUTLINE_SIGS, sources, _cases(REORDERED))
    assert held == {"frequency": "def frequency(m, k):", "quality": "def quality(c, w, m):"}
    # The whole-file repair the quest needed (a fix elsewhere, the parameters as written) is no longer refused.
    assert fs.changed_signatures(REORDERED.replace("** 0.5", "**0.5"), held) == []
    # Old: held to the outline's order, the same repair was refused for both functions.
    assert len(fs.changed_signatures(REORDERED, OUTLINE_SIGS)) == 2


def test_code_the_test_cannot_call_is_held_to_the_outlines_names_and_a_renaming_repair_is_still_refused() -> None:
    wrong_names = "def frequency(spring_k, m):\n    return (spring_k / m) ** 0.5\n\n\ndef quality(m, w, c):\n    return m * w / c\n"
    sources = {"spring/model.py": wrong_names}
    held = fs.held_signatures(OUTLINE_SIGS, sources, _cases(wrong_names))
    assert held["frequency"] == OUTLINE_SIGS["frequency"], "the test calls `k`: the repair must give it"
    assert held["quality"] == "def quality(m, w, c):"
    # Known positive: a repair that renames or drops a parameter of a function the test calls is still refused.
    renamed = wrong_names.replace("def quality(m, w, c)", "def quality(m, w, damping)").replace("m * w / c", "m * w / damping")
    refused = fs.changed_signatures(renamed, held)
    assert len(refused) == 2 and any("`quality`" in r for r in refused)


def test_a_function_the_code_does_not_define_keeps_the_outlines_line() -> None:
    held = fs.held_signatures(OUTLINE_SIGS, {"spring/model.py": "def other():\n    pass\n"}, [])
    assert held == OUTLINE_SIGS


# --- the pilot reads a run_trial that is a whole cell --------------------------------------------------------------------------

CELL_AS_TRIAL = (
    "def run_trial(cell, trial_id=0, seed=0):\n    if isinstance(cell, dict):\n        return run_cell(cell)\n    return {'x': 1.0}\n\n\n"
    "def run_cell(cell):\n    return {'x': sum(run_trial(1.0, i, i)['x'] for i in range(1000)) / 1000}\n")
CELL_FROM_TRIALS = (
    "def run_trial(cell, trial_id=0, seed=0):\n    return {'x': 1.0}\n\n\n"
    "def run_cell(cell):\n    return {'x': sum(run_trial(cell, i, i)['x'] for i in range(1000)) / 1000}\n")


def test_the_pilot_names_a_run_trial_that_returns_the_whole_cell() -> None:
    found = re_.pilot_problems([], {}, source=CELL_AS_TRIAL, stochastic=True)
    assert [p.kind for p in found.problems] == ["aggregate"] and "ONE trial" in found.problems[0].says
    # Old behaviour (nothing said before the run) when the study does not run trials, or the reading is not asked for.
    assert re_.pilot_problems([], {}, source=CELL_AS_TRIAL, stochastic=False).problems == []
    assert re_.pilot_problems([], {}, source=CELL_AS_TRIAL).problems == []
    # Known negative: a run_cell that is built from run_trial is how a cell is made, never a finding.
    assert re_.pilot_problems([], {}, source=CELL_FROM_TRIALS, stochastic=True).problems == []


