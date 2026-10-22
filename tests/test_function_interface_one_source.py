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


