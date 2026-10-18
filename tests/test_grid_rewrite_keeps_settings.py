"""A rewrite of the plan's settings never drops a setting the plan compared, and a sub-study states what it holds fixed."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import optimise_search as search
from core import plan
from tests.test_grid_size_at_plan import BIG, NAMES, PARTS, Model, _draft, _engine, TOPIC

GRID = {"kind": ["a", "b"], "size": [1, 2, 3], "depth": [10, 20], "noise": [0.1, 0.2, 0.3]}
SUBS = [{"name": "kinds", "vary": ["kind", "size"], "hold": {"depth": 10, "noise": 0.1}},
        {"name": "depths", "vary": ["depth"], "hold": {"kind": "a", "size": 1, "noise": 0.1}},
        {"name": "noises", "vary": ["noise"], "hold": {"kind": "b", "size": 1, "depth": 10}}]


def test_a_setting_with_several_values_may_not_leave_the_plan_a_single_valued_one_may_become_fixed() -> None:
    before = {"grid": {"a": [1, 2], "b": [1], "c": [1, 2]}, "in_each_run": {"m": ["x", "y"]}}
    assert plan.lost_settings(before, {"grid": {"a": [1, 2], "c": [1, 2]}, "in_each_run": {"m": ["x", "y"]}, "constants": {"b": 1}}) == []
    assert plan.lost_settings(before, {"grid": {"a": [1, 2]}, "in_each_run": {"m": ["x", "y"], "c": [1, 2]}}) == ["b"]
    lost = plan.lost_settings(before, {"grid": {"a": [1, 2]}, "constants": {"b": 1, "c": 1}})
    assert lost == ["c", "m"], lost  # `c` had two values, `m` was compared inside each run
    assert "`c`, `m` are in the plan now" in plan.lost_settings_reason(lost)
    assert plan.lost_settings({}, {"grid": {"a": [1]}}) == []


@pytest.mark.asyncio
async def test_a_rewrite_that_drops_a_setting_is_refused_with_its_name_and_the_ask_goes_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(BIG, sub_studies=PARTS))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    fewer = {k: v for k, v in BIG.items() if k != NAMES[-1]}
    parts = [p for p in PARTS if p["name"] != "part5"]
    with pytest.raises(ValueError, match=f"`{NAMES[-1]}` is in the plan now"):
        eng._keep_only_the_parts(text, {"protocol": {"grid": fewer, "sub_studies": parts}}, ["grid", "sub_studies"])
    # the same settings kept (all of them, in another division) are taken
    kept = eng._keep_only_the_parts(text, {"protocol": {"grid": BIG, "sub_studies": PARTS[::-1]}}, ["grid", "sub_studies"])
    assert plan.parse(kept).design["protocol"]["grid"] == BIG


def _read(subs):
    return plan.normalize_protocol({"grid": GRID, "sub_studies": subs})


def test_every_part_says_the_value_it_holds_every_setting_it_does_not_vary_at() -> None:
    fixed, why = _read(SUBS)
    assert why is None, why

    loose = [{"name": "kinds", "vary": ["kind", "size"]}, {"name": "d", "vary": ["depth"]}, {"name": "n", "vary": ["noise"]}]
    fixed, why = _read(loose)
    assert fixed is None and "does not say the value it holds `depth`, `noise`" in why
    assert "EVERY other setting" in plan.grid_request("t", "p")


def test_a_setting_is_compared_when_a_part_varies_it_or_the_holds_between_them_cover_all_its_values() -> None:
    grid = {"kind": ["x", "y"], "size": [1, 2, 3]}
    # no part varies `kind`; each holds it at one value: both values are run
    by_kind = [{"name": "x", "vary": ["size"], "hold": {"kind": "x"}}, {"name": "y", "vary": ["size"], "hold": {"kind": "y"}}]
    fixed, why = plan.normalize_protocol({"grid": grid, "sub_studies": by_kind})
    assert why is None, why
    assert search.grid_size(search.study_grid(fixed)) == 6
    # known positive: `y` is neither varied nor held by any part, so it would never be run
    only_x = [{"name": "x", "vary": ["size"], "hold": {"kind": "x"}}]
    fixed, why = plan.normalize_protocol({"grid": grid, "sub_studies": only_x})
    assert fixed is None and "`kind`" in why and "'y'" in why
