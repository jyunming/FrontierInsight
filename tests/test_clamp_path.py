"""A cap counts inside the range only for the value whose whole path it is written at (the 2026-09-27 delta re-audit, F6).

The check used to match the last segment of the path: ``{"train": {"rmse": min(train_rmse, 10.0)}}`` made a test RMSE
of exactly 10 look capped, though nothing capped it.
"""

from __future__ import annotations

from core.plausibility import Assertion, direct_caps, violations

RMSE = [Assertion(path="rmse", min=0.0, max=100.0)]
TEST_RMSE = [Assertion(path="test.rmse", min=0.0, max=100.0)]


def test_a_cap_on_train_is_not_a_cap_on_test() -> None:
    code = 'out = {"train": {"rmse": min(train_rmse, 10.0)}, "test": {"rmse": test_rmse}}\n'
    result = {"train": {"rmse": 3.0}, "test": {"rmse": 10.0}}
    assert violations(result, TEST_RMSE, code=code) == [], "the re-audit's counterexample"
    assert violations(result, RMSE, code=code) == [], "the assertion names only the leaf: still test's own path counts"
    # The train value on its own cap is caught.
    (found,) = violations({"train": {"rmse": 10.0}, "test": {"rmse": 2.0}}, RMSE, code=code)
    assert found.kind == "clamped" and found.path == "train.rmse"


def test_a_literal_that_holds_only_part_of_the_path_cannot_say_which_value_it_caps() -> None:
    code = 'inner = {"rmse": min(rmse, 10.0)}\nout = {"test": inner}\n'
    assert violations({"test": {"rmse": 10.0}}, TEST_RMSE, code=code) == []
    # At the top level the literal is the whole path: caught as before.
    (found,) = violations({"rmse": 10.0}, RMSE, code='out = {"rmse": min(rmse, 10.0)}\n')
    assert found.kind == "clamped"


def test_a_list_of_records_is_matched_position_by_position() -> None:
    code = 'out = {"runs": [{"rmse": min(r, 10.0)} for r in rs]}\n'
    (found,) = violations({"runs": [{"rmse": 10.0}]}, RMSE, code=code)
    assert found.kind == "clamped" and found.path.startswith("runs")
    assert direct_caps(code, "runs[0].rmse") == {10.0}
    assert direct_caps(code, "other[0].rmse") == set()


def test_the_path_can_be_given_as_tokens_or_a_string() -> None:
    code = 'out = {"test": {"rmse": min(x, 7.5)}}\n'
    assert direct_caps(code, "test.rmse") == {7.5}
    assert direct_caps(code, (("k", "test"), ("k", "rmse"))) == {7.5}
    assert direct_caps(code, "train.rmse") == set()
    assert direct_caps("not python (", "rmse") == set()
