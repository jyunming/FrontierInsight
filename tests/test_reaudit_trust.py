"""Regression contracts from an external re-audit's reproductions (each began as a probe asserting the defect; every one
now asserts the correct behaviour). Neutral names throughout: nothing here depends on a topic."""
from __future__ import annotations

from types import SimpleNamespace

from core import flat_output


def test_the_same_numbers_in_other_units_get_the_same_flat_decision() -> None:
    def cells(scale: float) -> list[SimpleNamespace]:
        return [SimpleNamespace(cell={"setting": i}, rows=[{"status": "ok", "values": {"size": i * scale}}])
                for i in (1, 2, 3)]
    # 1, 2 and 3 of a small unit written in the base unit, and in the small unit: one decision for both.
    assert flat_output.flat_sweep(cells(1e-12)) is None
    assert flat_output.flat_sweep(cells(1.0)) is None
    assert flat_output.flat_sweep(cells(1e12)) is None
