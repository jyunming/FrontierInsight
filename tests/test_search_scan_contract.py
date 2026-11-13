"""A search for the best design runs a coarse scan first and records every setting of it as a measurement's grid is
recorded. The plan writes that scan inside its search block (``protocol.optimisation.grid`` and ``.sub_studies``); the
contract and the plan-time questions read the settings the runs differ in from there, by one reader."""
from __future__ import annotations

from typing import Any

from core import analysis_frame as _frame
from core import contract as _contract
from core import contract_parts as _parts
from core import dependent_map as _dependents


def _protocol() -> dict[str, Any]:
    """A bracket: the search varies its thickness; the scan compares two materials over three spans."""
    return {
        "form": "contract",
        "metrics": [
            {"id": "sag", "estimand": "sag at mid-span", "unit": "one bracket", "role": "result", "returned": "each_run",
             "depends_on": ["material", "span", "thickness"]},
            {"id": "cost", "estimand": "cost of the bracket", "unit": "one bracket", "role": "result", "returned": "each_run",
             "depends_on": ["material"]},
        ],
        "comparisons": [{"id": "sag_by_material", "result": "sag", "across": "material", "how": "difference"}],
        "optimisation": {
            "objective": {"quantity": "sag", "direction": "minimise"},
            "design_variables": [{"name": "thickness", "kind": "continuous", "low": 1, "high": 8}],
            "fixed": {"material": "steel", "span": 20},
            "baseline": {"values": {"thickness": 2}},
            "grid": {"material": ["steel", "timber"], "span": [10, 20, 30], "thickness": [2, 4]},
            "sub_studies": [{"name": "steel", "vary": ["span", "thickness"], "hold": {"material": "steel"}},
                            {"name": "timber", "vary": ["span"], "hold": {"material": "timber", "thickness": 2}}],
            "in_each_run": {"load_case": ["static", "gust"]},
        },
    }


_FIGURE = {"shows": ["sag"], "x": "span", "series": "material", "panels": 1}


def _design(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    p = protocol if protocol is not None else _protocol()
    return {"hypothesis": "a thicker bracket sags less", "method": "a beam model", "figures_planned": ["sag_by_span.png"],
            "protocol": {**p, "figure_declarations": {"sag_by_span.png": dict(_FIGURE)}}}


def test_the_contract_reads_the_scan_of_a_search_as_the_settings_the_runs_differ_in() -> None:
    """The failure: three stored searches with a scan delivered none of their planned figures. The contract held no setting
    of the scan, so what the plan said of them (a result changes with them, a comparison across one, a figure along one)
    was left out with a gap."""
    for fi_analysis in (True, False):
        c = _contract.from_design(_design(), fi_analysis=fi_analysis)
        assert c.settings.compared == {"material": ("steel", "timber"), "span": (10, 20, 30), "thickness": (2, 4)}
        assert [s["name"] for s in c.settings.sub_studies] == ["steel", "timber"]
        assert c.settings.in_each_run == {"load_case": ("static", "gust")}
        assert {r.id: r.depends_on for r in c.results} == {"sag": ("material", "span", "thickness"), "cost": ("material",)}
        assert [x.id for x in c.comparisons] == ["sag_by_material"]
        assert [(f.x, f.series, f.declared) for f in c.figures] == [("span", "material", True)]
        assert c.gaps == ()


def test_a_name_that_is_no_setting_of_the_scan_is_still_left_out_and_said() -> None:
    p = _protocol()
    p["metrics"][1]["depends_on"] = ["material", "weather"]
    p["comparisons"] = [{"id": "sag_by_weather", "result": "sag", "across": "weather", "how": "difference"}]
    c = _contract.from_design({"protocol": p})
    assert c.result("cost").depends_on == ("material",) and c.comparisons == ()
    assert any("`cost` is said to depend on weather, which are not varied settings" in g for g in c.gaps)
    assert any("comparison `sag_by_weather` is left out" in g for g in c.gaps)


def test_the_search_blocks_own_scan_wins_and_a_measurements_grid_is_read_as_before() -> None:
    p = _protocol()
    p["grid"], p["sub_studies"] = {"other": [1, 2]}, [{"name": "elsewhere", "vary": ["other"], "hold": {}}]
    c = _contract.from_design({"protocol": p})
    assert set(c.settings.compared) == {"material", "span", "thickness"}
    assert [s["name"] for s in c.settings.sub_studies] == ["steel", "timber"]
    # A search that writes no scan: a grid beside it is still read (as its `in_each_run` is).
    del p["optimisation"]["grid"], p["optimisation"]["sub_studies"]
    c = _contract.from_design({"protocol": p})
    assert set(c.settings.compared) == {"other"} and [s["name"] for s in c.settings.sub_studies] == ["elsewhere"]
    # A measurement.
    measure = {"grid": {"dose": [1, 2]}, "sub_studies": [{"name": "low", "vary": ["dose"], "hold": {}}],
               "metrics": [{"id": "yield", "estimand": "yield", "unit": "one plot"}]}
    c = _contract.from_design({"protocol": measure})
    assert c.settings.compared == {"dose": (1, 2)} and [s["name"] for s in c.settings.sub_studies] == ["low"]
    assert _contract.varied_settings(measure) == ["dose"]


def test_the_plan_time_checks_hold_a_search_to_the_settings_of_its_scan() -> None:
    """Before, every setting a search plan named was refused ("the settings are none"), also the one its own request listed."""
    p = _protocol()
    assert _contract.varied_settings(p) == ["material", "span", "thickness", "load_case"]
    roles = {"sag": {"role": "result", "depends_on": ["material", "span", "load_case"], "returned": "each_run"},
             "cost": {"role": "result", "depends_on": ["weather"], "returned": "each_run"}}
    assert _contract.check_part("roles", roles, {"protocol": p}) == [
        "`cost`: weather is not a setting that varies (the settings are material, span, thickness, load_case)"]
    across = {"result": "sag", "across": "material", "values": ["steel", "timber"]}
    assert _contract.check_part("comparisons", {"comparisons": [across]}, {"protocol": p}) == []
    assert _contract.check_part("comparisons", {"comparisons": [{**across, "values": ["steel", "glass"]}]}, {"protocol": p}) == [
        "comparison 1: `values` must be values of `material` in the grid"]
    assert _contract.check_part("comparisons", {"comparisons": [{**across, "across": "weather"}]}, {"protocol": p}) == [
        "comparison 1: `across` must be a setting that varies (material, span, thickness, load_case)"]


def test_the_requests_show_the_scans_settings_and_its_names_are_declared() -> None:
    p = _protocol()
    assert _parts._settings_lines(p).splitlines() == [
        '- material: ["steel", "timber"]', "- span: [10, 20, 30]", "- thickness: [2, 4]",
        '- load_case: ["static", "gust"] (every run goes through all of these together)']
    assert {"material", "span", "thickness", "steel", "timber", "load_case"} <= _contract._declared_names(p)
    # A new derived result may not take the name of a setting of the scan.
    assert _dependents._new_result_problems("D1", {"id": "span", "inputs": ["sag"], "what": "w"}, {"protocol": p}, {}) == [
        "`D1`: `span` is a setting the study changes; give the new result a name of its own"]


def test_a_planned_figure_is_declared_along_the_scan_and_the_frame_carries_it_with_the_scans_parts() -> None:
    design = _design()
    asked = {"sag_by_span.png": ""}
    _lines, several, _single = _parts._figure_holds(design)
    assert several == ["load_case", "material", "span", "thickness"]
    assert _parts._check_figures({"sag_by_span.png": dict(_FIGURE)}, asked, design) == []
    # Known positive: along the setting every run goes through together, a number recorded once per run is still refused,
    # and the way forward is now the scan's settings.
    problems = _parts._check_figures({"sag_by_span.png": {"shows": ["cost"], "x": "load_case", "panels": 1}}, asked, design)
    assert len(problems) == 1 and "`x` is `load_case`, which every run goes through together" in problems[0]
    assert problems[0].endswith("draw it along material, span, thickness")
    plan = _frame.plan_of(design)
    assert plan["parts"] == {"steel": {"material": "steel"}, "timber": {"material": "timber", "thickness": 2}}
    assert plan["compared"] == {"load_case": ["static", "gust"]}
    assert {k: plan["figures"]["fig1"][k] for k in ("file", "shows", "x", "series")} == {
        "file": "sag_by_span.png", "shows": ["sag"], "x": "span", "series": "material"}
