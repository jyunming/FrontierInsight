"""``{{calc:}}`` counts a number the plan itself declares (a value under ``protocol.constants`` or ``protocol.thresholds``)
as the plan's number, however small. A small whole number that only a setting, a run count or a seed holds does not."""
from __future__ import annotations

import pytest

from core import paper_values
from core.typed_numbers import declared_names

NOT = paper_values.NOT_RECORDED
TOPIC = "A counter moves up or down by one. How many steps on average?"


@pytest.mark.parametrize("protocol, expr, text", [
    ({"constants": {"lo": 4, "hi": 9}}, "lo * hi", "36"),
    ({"constants": {"lo": 4, "hi": 9}}, "lo * hi * (lo**2 + hi**2 - 2) / 3", "1140"),
    ({"thresholds": {"alpha": 0.05, "m": 6}}, "m * alpha", "0.3"),
    ({"constants": {"limits": {"upper": 3}}}, "upper ** 2", "9"),
    ({"constants": {"bottom": -4}}, "-bottom * 2", "8"),
])
def test_declared_names_are_the_plans(protocol, expr, text) -> None:
    value = paper_values.calc(expr, protocol, TOPIC)
    assert value.hit is None and value.text == text, value


def test_a_typed_digit_equal_to_a_declared_value_is_not_the_plans() -> None:
    # Finding: 2/3 was admitted because a constant happens to be 2. Provenance is the name, not the digit.
    for protocol, expr in [({"constants": {"x": 2}}, "2 / 3"), ({"constants": {"lo": 4, "hi": 9}}, "4 * 9")]:
        value = paper_values.calc(expr, protocol, TOPIC)
        assert value.text == NOT and value.hit is not None, value
    assert paper_values.calc("x / 3", {"constants": {"x": 2}}, TOPIC).text == "0.6667"


def test_a_name_two_values_share_or_a_setting_is_not_named() -> None:
    for protocol in ({"constants": {"a": {"k": 2}, "b": {"k": 3}}}, {"grid": {"k": [3, 4]}}):
        value = paper_values.calc("k * 2", protocol, TOPIC)
        assert value.text == NOT and value.hit is not None, value


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


def test_declared_names_reads_only_constants_and_thresholds() -> None:
    got = declared_names({"constants": {"a": 3, "b": {"c": [2, True], "d": {"e": 1.5}}}, "thresholds": {"t": 0.5}, "grid": {"g": [8]}})
    assert got == {"a": 3.0, "e": 1.5, "t": 0.5}
