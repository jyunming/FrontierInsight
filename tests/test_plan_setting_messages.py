"""What a plan is told when a numerical setting's tolerance or a held element's baseline cannot be used."""
from __future__ import annotations

from core import optimisation_plan as op


def test_a_tolerance_written_as_a_percentage_is_refused_with_what_to_write() -> None:
    for bad in (5, "5%", 0, 1.5):
        setting, why = op._setting("dx", {"search": 2.0, "tolerance": bad})
        assert setting is None and "between 0 and 1" in why
    setting, why = op._setting("dx", {"search": 2.0, "tolerance": 0.02})
    assert why is None and setting["tolerance"] == 0.02


def test_a_baseline_off_a_held_element_says_the_value_to_write() -> None:
    variable = {"name": "source", "shape": [2, 2], "kind": "continuous", "low": [[0, 0], [0, 0]],
                "high": [[1, 1], [1, 0]]}
    why = op.array_problem(variable, [[0.5, 0.5], [0.5, 0.3]])
    assert "not the value 0 this element is held at" in why and "write 0 there" in why
    assert op.array_problem(variable, [[0.5, 0.5], [0.5, 0.0]]) is None
