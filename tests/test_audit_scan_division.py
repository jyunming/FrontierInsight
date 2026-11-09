"""The settings to compare and the parts of the study are written against each other: FI does not invent a crossed part on
top of the parts an answer declares, and an audit's change to the settings and to the parts is made together or not at
all (with the real reason kept when it is not). Neutral made-up settings; fake model only."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from core import audit_open_items as audit_open
from core import optimise_search as search
from core import plan
from core import topic_requirements as topic
from tests.test_design_critique_record import DRAFT, _engine as _critique_engine
from tests.test_plan_optimisation import HEAT_SINK, _engine as _search_engine

# -- a setting added for the topic to a plan that has no parts (topic_requirements.joined_sub_studies) -------------------

OLD = {"grid": {"alpha": [1, 2, 3], "beta": [1, 2], "gamma": ["x", "y"]}}


def _cells(protocol: dict[str, Any]) -> int:
    fixed, why = plan.normalize_protocol(protocol)
    assert fixed is not None, why
    return search.grid_size(search.study_grid(fixed))


def test_the_old_grid_is_not_crossed_again_with_settings_the_answers_parts_vary() -> None:
    grid = {**OLD["grid"], "delta": [1, 2, 3]}
    answer = [{"name": "bd", "vary": ["beta", "delta"]}, {"name": "g", "vary": ["gamma"]}]
    subs = topic.joined_sub_studies(OLD, grid, answer)
    unnamed = [s for s in subs if s["name"] == ""]
    assert [s["vary"] for s in unnamed] == [["alpha"]], "only the setting no answer part varies stays in the old part"
    assert not any({"alpha", "beta"} <= set(s["vary"]) for s in subs)
    # 3 (alpha alone) + 6 (beta x delta) + 2 (gamma) minus the base cell they share
    assert _cells({"grid": grid, "sub_studies": subs}) == 3 + 6 + 2 - 2


def test_no_old_part_is_made_when_the_answers_parts_vary_every_old_setting() -> None:
    grid = {**OLD["grid"], "delta": [1, 2, 3]}
    answer = [{"name": "ab", "vary": ["alpha", "beta", "delta"]}, {"name": "g", "vary": ["gamma"]}]
    subs = topic.joined_sub_studies(OLD, grid, answer)
    assert [s["name"] for s in subs] == ["ab", "g"]
    assert _cells({"grid": grid, "sub_studies": subs}) == 18 + 2 - 1


def test_known_positive_parts_that_do_not_touch_the_old_settings_leave_the_old_grid_crossed() -> None:
    """The plan had no parts and the answer's parts vary only the added setting: the old grid stays one crossed part, as
    before (nothing the answer's parts vary was in it)."""
    grid = {**OLD["grid"], "delta": [1, 2, 3]}
    subs = topic.joined_sub_studies(OLD, grid, [{"name": "d", "vary": ["delta"]}])
    assert subs[0] == {"name": "", "vary": ["alpha", "beta", "gamma"], "hold": {"delta": 1}}
    assert [s["name"] for s in subs] == ["", "d"]
    # and no answer for the parts at all: the setting joins the one grid, nothing is divided
    assert topic.joined_sub_studies(OLD, grid, None) is None


# -- an audit's change to the settings and to the parts (Engine._hold_audit_repairs_to_the_schema) ---------------------

PARTS = [
    {"name": "p1", "vary": ["alpha", "beta"], "hold": {"gamma": "x"}},
    {"name": "p2", "vary": ["gamma"], "hold": {"alpha": 1, "beta": 1}},
]


def _measure(grid: dict[str, Any], subs: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {**DRAFT, "protocol": {"runs_per_setting": 300, "grid": grid, **({"sub_studies": subs} if subs else {})}}


def test_a_grid_change_that_would_drop_the_plans_parts_is_refused_with_its_parts(tmp_path: Path) -> None:
    eng = _critique_engine(tmp_path, [])
    before = _measure(OLD["grid"], PARTS)
    amended = _measure({"alpha": [1, 2, 3]}, PARTS)  # the settings the parts vary are gone
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), amended)
    assert design["protocol"]["grid"] == OLD["grid"]
    assert [s["name"] for s in design["protocol"]["sub_studies"]] == ["p1", "p2"], "the parts are not dropped"
    keys = {n["key"]: n["reason"] for n in not_applied}
    assert "protocol.grid" in keys and "protocol.sub_studies" not in keys, "only the change the audit made is named"
    assert "together" in keys["protocol.grid"] and "`beta`" in keys["protocol.grid"]


def test_known_positive_a_grid_and_parts_change_that_read_together_is_made(tmp_path: Path) -> None:
    """Neither half reads alone (a new setting no old part varies; a part that varies a setting the old grid lacks), both
    read together: the audit's change is made whole."""
    eng = _critique_engine(tmp_path, [])
    before = _measure(OLD["grid"], PARTS)
    grid = {**OLD["grid"], "delta": [1, 2]}
    subs = [*PARTS, {"name": "p3", "vary": ["delta"], "hold": {"alpha": 1, "beta": 1, "gamma": "x"}}]
    subs = [{**s, "hold": {**s["hold"], **({"delta": 1} if "delta" not in s["vary"] else {})}} for s in subs]
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), _measure(grid, subs))
    assert design["protocol"]["grid"] == grid and [s["name"] for s in design["protocol"]["sub_studies"]] == ["p1", "p2", "p3"]
    assert not not_applied


def test_known_positive_a_parts_free_grid_change_is_still_made(tmp_path: Path) -> None:
    eng = _critique_engine(tmp_path, [])
    design, not_applied = eng._hold_audit_repairs_to_the_schema(
        _measure(OLD["grid"], None), _measure({**OLD["grid"], "alpha": [1, 2, 3, 4]}, None))
    assert design["protocol"]["grid"]["alpha"] == [1, 2, 3, 4] and not not_applied


def test_the_audits_parts_that_vary_settings_its_own_grid_change_removed_are_refused_with_that_reason(tmp_path: Path) -> None:
    eng = _critique_engine(tmp_path, [])
    before = _measure(OLD["grid"], PARTS)
    other_parts = [{"name": "q", "vary": ["beta"], "hold": {"alpha": 1, "gamma": "x"}}]
    design, not_applied = eng._hold_audit_repairs_to_the_schema(
        copy.deepcopy(before), _measure({"alpha": [1, 2, 3], "gamma": ["x", "y"]}, other_parts))
    assert design["protocol"] == before["protocol"], "both halves are put back"
    reasons = eng._audit_reasons(not_applied)
    assert "`beta`" in reasons["protocol.sub_studies"] and reasons["protocol.grid"] == reasons["protocol.sub_studies"]


# -- the same for a search's scan (the block's own grid and sub_studies) ------------------------------------------------

SCAN = {"fin_spacing": [2.0, 3.0, 4.0], "fin_thickness": [0.5, 1.0], "fin_count": [10, 20]}
SCAN_PARTS = [
    {"name": "spacing", "vary": ["fin_spacing", "fin_thickness"], "hold": {"fin_count": 10}},
    {"name": "count", "vary": ["fin_count"], "hold": {"fin_spacing": 2.0, "fin_thickness": 0.5}},
]


def _search(grid: dict[str, Any], subs: list[dict[str, Any]] | None, **more: Any) -> dict[str, Any]:
    design = copy.deepcopy(HEAT_SINK)
    block = design["protocol"]["optimisation"]
    block["grid"] = grid
    if subs is not None:
        block["sub_studies"] = subs
    block.update(more)
    return design


def test_a_search_scan_cut_to_one_setting_does_not_drop_its_parts(tmp_path: Path) -> None:
    eng = _search_engine(tmp_path, [])
    before = _search(SCAN, SCAN_PARTS)
    assert plan.normalize_design(copy.deepcopy(before))[0] is not None
    design, not_applied = eng._hold_audit_repairs_to_the_schema(copy.deepcopy(before), _search({"fin_spacing": [2.0, 3.0, 4.0]}, SCAN_PARTS))
    block = design["protocol"]["optimisation"]
    assert block["grid"] == SCAN and [s["name"] for s in block["sub_studies"]] == ["spacing", "count"]
    reasons = eng._audit_reasons(not_applied)
    assert "together" in reasons["protocol.optimisation.grid"] and "`fin_thickness`" in reasons["protocol.optimisation.grid"]
    assert "protocol.optimisation.sub_studies" not in reasons, "the audit did not change the parts"


def test_known_positive_a_search_scan_and_parts_cut_together_are_made(tmp_path: Path) -> None:
    eng = _search_engine(tmp_path, [])
    cut_parts = [{"name": "spacing", "vary": ["fin_spacing"], "hold": {}}]
    design, not_applied = eng._hold_audit_repairs_to_the_schema(
        _search(SCAN, SCAN_PARTS), _search({"fin_spacing": [2.0, 3.0, 4.0]}, cut_parts))
    block = design["protocol"]["optimisation"]
    assert block["grid"] == {"fin_spacing": [2.0, 3.0, 4.0]} and [s["vary"] for s in block["sub_studies"]] == [["fin_spacing"]]
    assert not not_applied


def test_a_part_left_out_of_a_partly_applied_block_keeps_its_own_reason(tmp_path: Path) -> None:
    """A rewrite of the search block applied except one part: the open item of that part (not the block's others) carries
    the reader's reason, not the generic 'FI did not apply it'."""
    eng = _search_engine(tmp_path, [])
    bad_budget = {"starts": 4, "per_start": 60, "seconds_per_evaluation": "measured on a few evaluations first"}
    _design, not_applied = eng._hold_audit_repairs_to_the_schema(
        _search(SCAN, SCAN_PARTS), _search(SCAN, SCAN_PARTS, evaluation_budget=bad_budget, baseline={
            "values": {"fin_spacing": 2.5, "fin_thickness": 1.2, "fin_count": 18}, "source": "[4] Table 1"}))
    reasons = eng._audit_reasons(not_applied)
    assert "protocol.optimisation" not in reasons, "the block as a whole was applied"
    path = "protocol.optimisation.evaluation_budget"
    assert path in reasons and "could not be read" in reasons[path]
    props = [{"path": path, "value": bad_budget, "objection": "o", "fix": "f", "check": "c"}]
    held, missing = audit_open.split(props, _search(SCAN, SCAN_PARTS))
    assert not held and missing
    audit_open.record(tmp_path, held, missing, reasons=reasons, default_reason="", iteration=0)
    items = audit_open._load(tmp_path)["open"]
    assert items and "could not be read" in items[0]["reason"] and items[0]["reason"] != "FI did not apply it"


def test_the_audit_may_change_how_the_study_is_divided_not_what_it_is_judged_by(tmp_path: Path) -> None:
    """``sub_studies`` goes with ``grid`` as something the audit may change, so a compatible pair is taken together."""
    eng = _critique_engine(tmp_path, [])
    assert eng._audit_may_not_change("protocol.sub_studies") is False
    before = _measure(OLD["grid"], PARTS)
    grid = {**OLD["grid"], "delta": [1, 2]}
    subs = [{**s, "hold": {**s["hold"], "delta": 1}} for s in PARTS] + [
        {"name": "p3", "vary": ["delta"], "hold": {"alpha": 1, "beta": 1, "gamma": "x"}}]
    got, changed = eng._keep_what_the_results_are_judged_by(before, _measure(grid, subs))
    assert got["protocol"]["grid"] == grid and [s["name"] for s in got["protocol"]["sub_studies"]] == ["p1", "p2", "p3"]
    assert not changed
