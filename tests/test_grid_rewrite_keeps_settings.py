"""A rewrite of the plan's settings never drops a setting the plan compared, and the ways a sub-study can state what it holds
fixed (compared by the environment variable FI_SUB_STUDY_HOLD; the default is today's) read and count as described."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import optimise_search as search
from core import plan
from tests.test_grid_size_at_plan import BIG, PARTS, Model, _draft, _engine, TOPIC

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
    fewer = {k: v for k, v in BIG.items() if k != "source_parametrisation"}
    parts = [p for p in PARTS if p["name"] != "parametrisation"]
    with pytest.raises(ValueError, match="`source_parametrisation` is in the plan now"):
        eng._keep_only_the_parts(text, {"protocol": {"grid": fewer, "sub_studies": parts}}, ["grid", "sub_studies"])
    # the same settings kept (all of them, in another division) are taken
    kept = eng._keep_only_the_parts(text, {"protocol": {"grid": BIG, "sub_studies": PARTS[::-1]}}, ["grid", "sub_studies"])
    assert plan.parse(kept).design["protocol"]["grid"] == BIG


def _read(subs, mode, monkeypatch):
    monkeypatch.setenv("FI_SUB_STUDY_HOLD", mode)
    return plan.normalize_protocol({"grid": GRID, "sub_studies": subs})


def test_every_way_reads_a_complete_division_and_only_the_default_fills_a_missing_hold(monkeypatch) -> None:
    for mode in ("default", "G1", "G2", "G4"):
        fixed, why = _read(SUBS, mode, monkeypatch)
        assert why is None, (mode, why)
    loose = [{"name": "kinds", "vary": ["kind", "size"]}, {"name": "d", "vary": ["depth"]}, {"name": "n", "vary": ["noise"]}]
    for mode in ("default", "G1"):
        assert _read(loose, mode, monkeypatch)[1] is None
    fixed, why = _read(loose, "G2", monkeypatch)
    assert fixed is None and "does not say the value it holds `depth`, `noise`" in why
    fixed, why = _read(loose, "G4", monkeypatch)  # only a setting whose values are names must be held
    assert fixed is None and "`kind`" in why and "noise" not in why
    names_held = [{"name": "kinds", "vary": ["kind", "size"]}, {"name": "d", "vary": ["depth"], "hold": {"kind": "a"}},
                  {"name": "n", "vary": ["noise"], "hold": {"kind": "b"}}]
    assert _read(names_held, "G4", monkeypatch)[1] is None
    for mode, word in (("G1", "name that value in the part's `hold`"), ("G2", "EVERY other setting"), ("G3", "only the settings that part uses"),
                       ("G4", "whose values are names")):
        monkeypatch.setenv("FI_SUB_STUDY_HOLD", mode)
        assert word in plan.grid_request("t", "p"), mode
    monkeypatch.delenv("FI_SUB_STUDY_HOLD")
    assert "the one in its `hold`" in plan.grid_request("t", "p")


def test_own_settings_arm_reads_counts_and_refuses_what_the_others_do(monkeypatch) -> None:
    monkeypatch.setenv("FI_SUB_STUDY_HOLD", "G3")
    own = [{"name": "kinds", "grid": {"kind": ["a", "b"], "size": [1, 2, 3]}}, {"name": "depths", "grid": {"depth": [10, 20]}},
           {"name": "noises", "grid": {"noise": [0.1, 0.3]}}]
    fixed, why = plan.normalize_protocol({"grid": GRID, "sub_studies": own})
    assert why is None
    grid = search.study_grid(fixed)
    assert search.grid_size(grid) == 6 + 2 + 2  # no setting is held: a part's settings are only its own
    bad = [{"name": "x", "grid": {"kind": ["zzz"]}}, *own[1:]]
    assert "must be some of its values" in plan.normalize_protocol({"grid": GRID, "sub_studies": bad})[1]
    missing = own[:2]
    assert "`noise` is a setting of `protocol.grid` that no entry" in plan.normalize_protocol({"grid": GRID, "sub_studies": missing})[1]
