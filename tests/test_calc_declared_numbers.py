"""``{{calc:}}`` counts a number the plan itself declares (a value under ``protocol.constants`` or ``protocol.thresholds``)
as the plan's number, however small. A small whole number that only a setting, a run count or a seed holds does not."""
from __future__ import annotations

import pytest

from core import paper_values
from core.typed_numbers import declared_numbers

NOT = paper_values.NOT_RECORDED
TOPIC = "A counter moves up or down by one. How many steps on average?"


@pytest.mark.parametrize("protocol, expr, text", [
    ({"constants": {"lo": 4, "hi": 9}}, "4 * 9", "36"),
    ({"constants": {"lo": 4, "hi": 9}}, "4 * 9 * (4**2 + 9**2 - 2) / 3", "1140"),
    ({"thresholds": {"alpha": 0.05, "m": 6}}, "6 * 0.05", "0.3"),
    ({"constants": {"limits": {"upper": 3}}}, "3 ** 2", "9"),
    ({"constants": {"floor": -4}}, "4 * 2", "8"),
])
def test_declared_small_numbers_are_the_plans(protocol, expr, text) -> None:
    value = paper_values.calc(expr, protocol, TOPIC)
    assert value.hit is None and value.text == text, value


@pytest.mark.parametrize("protocol, expr", [
    ({"grid": {"n": [3, 4]}}, "3 * 4"),                    # settings are not declared constants
    ({"runs_per_setting": 5, "seed_policy": {"base": 7}}, "5 * 7"),
    ({"constants": {"lo": 4}}, "7 * 11"),                  # 11 is in neither
    ({"constants": {"lo": 4}}, "6 / 7"),                   # whole numbers alone make any number
    ({"constants": {"lo": 4}}, "2"),
])
def test_other_small_numbers_do_not_make_a_number_the_plan_fixes(protocol, expr) -> None:
    value = paper_values.calc(expr, protocol, TOPIC)
    assert value.text == NOT and value.hit is not None, value


def test_known_positive_unchanged() -> None:
    assert paper_values.calc("1 - 0.05 / 2", {"thresholds": {"alpha": 0.05}}, "").text == "0.975"


def test_declared_numbers_reads_only_constants_and_thresholds() -> None:
    got = declared_numbers({"constants": {"a": 3, "b": {"c": [2, True]}}, "thresholds": {"t": 0.5}, "grid": {"g": [8]}})
    assert {float(x) for x in got} == {3.0, 2.0, 0.5}
