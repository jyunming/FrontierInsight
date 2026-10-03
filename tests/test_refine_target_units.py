"""A refine's target is read with its unit: a percent is relative to the best design so far, a prefixed unit is
converted, a unit that does not fit the objective is refused, and a bare number is in the objective's own unit."""

from __future__ import annotations

import pytest

from core import optimise_refine as refine

BLOCK = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "K", "meaning": "the temperature of the device"},
    "constraints": [],
    "design_variables": [{"name": "x"}],
    "evaluation_budget": {"per_start": 6},
}
BEST = {"baseline": {"objective": 19.08}, "best": {"objective": 10.0, "design": {"x": 1}},
        "check": {"objective": 10.0, "baseline_objective": 19.0, "design": {"x": 1}}}
# A search that has hardly improved on its baseline: an improvement over the baseline is not yet reached.
SHORT = {"baseline": {"objective": 19.0}, "best": {"objective": 18.9, "design": {"x": 1}},
         "check": {"objective": 18.9, "baseline_objective": 19.0, "design": {"x": 1}}}


def _read(note: str, block: dict = BLOCK, best: dict = BEST) -> dict:
    return refine.read_request(note, block, best)


def test_a_percent_is_relative_to_the_current_best_not_read_in_the_objectives_unit() -> None:
    request = _read("I want f at most 5%")
    assert request["kind"] == refine.SEARCH
    assert request["round"]["target"] == pytest.approx(9.5)
    assert request["round"]["read_as"] == "f at most 9.5 K (5% below the current best 10 K at the finest settings)"
    assert _read("f at most 5 %")["round"]["target"] == pytest.approx(9.5)
    assert _read("f at most 5 percent")["round"]["target"] == pytest.approx(9.5)


def test_a_percent_of_a_quantity_made_high_is_above_the_best() -> None:
    high = {**BLOCK, "objective": {**BLOCK["objective"], "direction": "maximise"}}
    request = _read("f at least 10%", high)
    assert request["round"]["target"] == pytest.approx(11.0)
    assert "10% above the current best" in request["round"]["read_as"]


def test_a_percent_with_no_recorded_best_is_asked_again() -> None:
    request = _read("f at most 5%", BLOCK, {"baseline": {"objective": 19.0}})
    assert request["kind"] == refine.UNCLEAR and "Nothing was changed" in request["says"]


def test_a_percent_of_a_quantity_with_no_unit_is_asked_again() -> None:
    plain = {**BLOCK, "objective": {**BLOCK["objective"], "unit": ""}}
    assert _read("f at least 85%", plain)["kind"] == refine.UNCLEAR


def test_a_quantity_measured_in_percent_takes_the_number_as_written() -> None:
    pct = {**BLOCK, "objective": {**BLOCK["objective"], "unit": "percent"}}
    best = {"baseline": {"objective": 40.0}, "best": {"objective": 30.0}}
    assert _read("f at most 25%", pct, best)["round"]["target"] == pytest.approx(25.0)


def test_the_objectives_own_unit_and_a_bare_number_are_taken_as_written() -> None:
    assert _read("f at most 9.5 K")["round"]["target"] == pytest.approx(9.5)
    assert _read("f at most 9.5 kelvin")["round"]["target"] == pytest.approx(9.5)
    assert _read("f at most 9.5")["round"]["target"] == pytest.approx(9.5)
    assert _read("f at most 9.5")["round"]["read_as"] == "f at most 9.5 K"


def test_a_prefixed_unit_is_converted_to_the_objectives_unit() -> None:
    request = _read("f at most 9500 mK")
    assert request["kind"] == refine.SEARCH
    assert request["round"]["target"] == pytest.approx(9.5)
    assert "9500 mK" in request["round"]["read_as"] and "9.5 K" in request["round"]["read_as"]


def test_a_unit_that_does_not_fit_the_objective_is_refused_not_guessed() -> None:
    for note in ("f at most 9.5 W", "f at most 9.5 Pa", "the f target is at most 280 C", "f at most 280 °C",
                 "f at most 9.5 K/s", "f at most 280 celsius"):
        request = _read(note)
        assert request["kind"] == refine.UNCLEAR, note
        assert "Nothing was changed" in request["says"] and "in K" in request["says"], note


def test_an_unrelated_number_with_a_unit_is_not_a_target() -> None:
    assert _read("Shorten the abstract to at most 200 words") == {}
    assert _read("Use a font of at most 10 pt in the figures") == {}
    block = {**BLOCK, "objective": {**BLOCK["objective"], "quantity": "device temperature"}}
    assert _read("Make the device temperature plot's x-axis at most 100 s", block) == {}
    assert _read("Keep the device temperature PDF below 10 MB", block) == {}


def test_an_improvement_over_the_baseline_has_its_unit_read_too() -> None:
    assert _read("I want it 3000 mK better than the baseline", BLOCK, SHORT)["round"]["target"] == pytest.approx(16.0)
    assert _read("I want it 5% better than the baseline", BLOCK, SHORT)["round"]["target"] == pytest.approx(18.05)
    assert _read("I want it 5 percent better than the baseline", BLOCK, SHORT)["round"]["target"] == pytest.approx(18.05)
    refused = _read("I want it 6 W better than the baseline", BLOCK, SHORT)
    assert refused["kind"] == refine.UNCLEAR and "in K" in refused["says"]


def test_a_unit_named_in_words_an_unknown_unit_and_a_note_about_the_paper() -> None:
    named = {**BLOCK, "objective": {**BLOCK["objective"], "unit": "kelvin"}}
    assert _read("f at most 9.5 kelvin", named)["round"]["target"] == pytest.approx(9.5)
    assert _read("f at most 9.5 K", named)["round"]["target"] == pytest.approx(9.5)
    sec = {**BLOCK, "objective": {**BLOCK["objective"], "unit": "s"}}
    refused = _read("I want it 2 min better than the baseline", sec, SHORT)
    assert refused["kind"] == refine.UNCLEAR and "min" in refused["says"]
    assert _read("Keep the PDF below 10 MB, the target for upload") == {}
    assert _read("Keep the f PDF below 10 MB so reviewers can reach them") == {}
