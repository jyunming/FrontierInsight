"""What the roles and figures questions after the plan also ask (core/contract_parts.py, core/contract.py): whether each
result is one number, a curve or a map; a result's physical bound; the studied system (its parts, what each carries, what
touches what); and the reference lines of each figure. Each is checked from FI's own records, asked again once when it cannot
be used, then left out alone with a note; none of them changes the plan's form or stops a study. The fake model here is a
function, never a real one. Neutral made-up study: a water tank drained through a valve."""

from __future__ import annotations

import asyncio
import copy
import json
import string
from pathlib import Path
from typing import Any

from core import contract, contract_parts as cp, frozen_protocol, plausibility, quality_report

ROOT = Path(__file__).parent.parent
TEMPLATES = {n: string.Template((ROOT / "agents" / f"{n}.md").read_text(encoding="utf-8"))
             for n in ("plan_roles", "plan_names", "plan_comparisons", "plan_formulas", "plan_figures")}

DESIGN: dict[str, Any] = {
    "hypothesis": "h", "method": "Simulate the water level of a tank that drains through a valve while an inlet fills it.",
    "figures_planned": ["level_drop.png (the level drop against the valve opening for each inlet rate)"],
    "result_assertions": [{"path": "level_drop", "min": -5, "max": 50, "unit": "cm", "reason": "the tank is 50 cm deep"}],
    "protocol": {
        "grid": {"valve": [0.2, 0.5, 1.0], "inlet": [1, 2]},
        "constants": {"tank_depth_cm": 50},
        "runs_per_setting": 20,
        "metrics": [
            {"id": "level_drop", "kind": "mean", "estimand": "drop of the water level after one hour", "unit": "one run",
             "physical_unit": "cm"},
            {"id": "overflow", "kind": "proportion", "estimand": "P(the tank overflows)", "unit": "one run"},
            {"id": "level_trace", "kind": "mean", "estimand": "the water level over the hour", "unit": "one run"},
            {"id": "drop_theory", "kind": "mean", "estimand": "the drop by the closed form", "unit": "one run",
             "reference": {"formula": "valve * 10", "kind": "value"}},
        ],
        "oracles": [{"name": "closed valve", "kind": "special_case", "check": "no outflow", "expected": 0.0,
                     "tolerance": 0.01, "case": {"valve": 0.2, "inlet": 1}, "measure": "overflow"}],
    },
}

ROW = {"role": "result", "depends_on": ["valve", "inlet"], "returned": "each_run"}
PLAIN = {"results": {"level_drop": dict(ROW), "overflow": dict(ROW), "level_trace": dict(ROW),
                     "drop_theory": {**ROW, "role": "reference"}}}
SYSTEM = {"parts": [{"id": "tank", "name": "water tank", "kind": "part", "carries": ["tank_depth_cm", "level_drop"]},
                    {"id": "valve_out", "name": "drain valve", "kind": "boundary", "carries": ["valve"]},
                    {"id": "inflow", "name": "inlet pipe", "kind": "source", "carries": ["inlet"]}],
          "links": [{"from": "inflow", "to": "tank", "what": "water"}, {"from": "tank", "to": "valve_out", "what": "water"}]}


def full() -> dict[str, Any]:
    out = copy.deepcopy(PLAIN)
    out["results"]["overflow"]["bound"] = {"min": 0, "max": 1, "why": "a probability"}
    out["results"]["level_drop"]["bound"] = {"max": 50, "unit": "cm", "why": "the level cannot drop below the tank's floor"}
    out["results"]["level_trace"].update({"shape": "curve", "along": ["time"]})
    out["system"] = copy.deepcopy(SYSTEM)
    return out


MARKERS = (("roles", "What kind of number it is"), ("comparisons", "compares between values"),
           ("figures", "FI draws each figure itself"))


class Fake:
    def __init__(self, **replies: list[Any]) -> None:
        self.replies, self.prompts = replies, []

    async def __call__(self, prompt: str) -> str:
        part = next(k for k, marker in MARKERS if marker in prompt)
        self.prompts.append((part, prompt))
        reply = self.replies[part].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


NO_COMPARE = {"comparisons": []}


def run(fake: Fake, design: dict[str, Any] = DESIGN, **kw: Any) -> cp.Outcome:
    return asyncio.run(cp.ask_parts(design, fake, parse, TEMPLATES, **kw))


def metric(out: cp.Outcome, rid: str) -> dict[str, Any]:
    return next(m for m in out.design["protocol"]["metrics"] if m["id"] == rid)


# --- asked, checked, applied, stored ---------------------------------------------------------------------------------------


def test_the_roles_question_asks_shape_bound_and_system_in_the_same_call() -> None:
    fake = Fake(roles=[full()], comparisons=[NO_COMPARE])
    out = run(fake)
    assert [p for p, _ in fake.prompts] == ["roles", "comparisons"]  # no call added
    prompt = fake.prompts[0][1]
    assert "Its shape" in prompt and "physical bound" in prompt and "describe the system" in prompt
    assert "tank_depth_cm" in prompt  # the plan's names an entry may carry
    assert out.form == "contract" and out.notes == []
    p = out.design["protocol"]
    assert metric(out, "level_trace")["shape"] == "curve" and metric(out, "level_trace")["along"] == ["time"]
    assert metric(out, "overflow")["range"] == {"min": 0, "max": 1, "reason": "a probability", "physical": True}
    assert metric(out, "level_drop")["range"]["max"] == 50 and metric(out, "level_drop")["range"]["unit"] == "cm"
    assert p["system"] == SYSTEM
    assert out.parts["roles"]["system"] == "stated" and "left_out" not in out.parts["roles"]


def test_what_is_applied_is_read_into_the_contract_and_frozen_with_the_protocol(tmp_path: Path) -> None:
    out = run(Fake(roles=[full()], comparisons=[NO_COMPARE]))
    c = contract.from_design(out.design)
    trace, flow, drop = (c.result(i) for i in ("level_trace", "overflow", "level_drop"))
    assert (trace.shape, trace.along) == ("curve", ("time",))
    assert flow.range is not None and (flow.range.min, flow.range.max, flow.range.physical) == (0, 1, True)
    # the plan's own bound and the physical one are both held: the values inside both
    assert drop.range is not None and (drop.range.min, drop.range.max) == (-5, 50)
    assert c.system is not None and [x.id for x in c.system.parts] == ["tank", "valve_out", "inflow"]
    assert c.system.links[0].source == "inflow" and c.system.parts[0].carries == ("tank_depth_cm", "level_drop")
    assert not [g for g in c.gaps if "system" in g or "shape" in g]
    record = frozen_protocol.freeze(tmp_path, out.design["protocol"], approved_by="test", source="test")
    assert record["protocol"]["system"] == SYSTEM
    assert next(m for m in record["protocol"]["metrics"] if m["id"] == "overflow")["range"]["physical"] is True


def test_a_plain_result_is_unchanged_and_the_protocol_hash_is_the_old_one() -> None:
    plain = run(Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE]))
    said = copy.deepcopy(PLAIN)
    for row in said["results"].values():
        row.update({"shape": "number", "bound": None})
    stated = run(Fake(roles=[said], comparisons=[NO_COMPARE]))
    for out in (plain, stated):
        assert "system" not in out.design["protocol"]
        assert all(not {"shape", "along", "range"} & set(m) for m in out.design["protocol"]["metrics"])
    assert frozen_protocol.sha256(plain.design["protocol"]) == frozen_protocol.sha256(stated.design["protocol"])
    assert plain.parts["roles"]["system"] == "not given" and plain.parts["roles"]["asks"] == 1
    c = contract.from_design(plain.design)
    assert c.system is None and all(r.shape == "number" and r.along == () for r in c.results)


def test_a_study_with_no_set_up_says_so_and_stores_nothing() -> None:
    answer = {**copy.deepcopy(PLAIN), "system": None}
    out = run(Fake(roles=[answer], comparisons=[NO_COMPARE]))
    assert "system" not in out.design["protocol"] and out.parts["roles"]["system"] == "none" and out.notes == []


# --- checked from FI's records: asked again once, then left out alone ---------------------------------------------------


def test_a_system_carrying_an_undeclared_name_is_asked_again_once() -> None:
    bad = full()
    bad["system"]["parts"][0]["carries"] = ["tank_depth_cm", "salinity"]
    fake = Fake(roles=[bad, full()], comparisons=[NO_COMPARE])
    out = run(fake)
    assert [p for p, _ in fake.prompts] == ["roles", "roles", "comparisons"]
    assert "salinity" in fake.prompts[1][1] and "does not declare" in fake.prompts[1][1]
    assert out.design["protocol"]["system"] == SYSTEM and out.notes == []


def test_a_system_still_unusable_is_left_out_alone_and_the_roles_stand() -> None:
    bad = full()
    bad["system"]["parts"][1]["name"] = "valve 2"  # a number typed in a name
    bad["system"]["links"].append({"from": "tank", "to": "pump", "what": "water"})
    fake = Fake(roles=[bad, copy.deepcopy(bad)], comparisons=[NO_COMPARE])
    out = run(fake)
    assert [p for p, _ in fake.prompts].count("roles") == 2  # asked again once, not three times
    assert out.form == "contract" and out.parts["roles"]["valid"]
    assert "system" not in out.design["protocol"] and out.parts["roles"]["system"] == "left out"
    assert "system" in out.parts["roles"]["left_out"]
    assert any("studied system" in n and "could use" in n for n in out.notes)
    # what could be used of the same answer is kept
    assert metric(out, "overflow")["range"]["physical"] is True and metric(out, "level_trace")["shape"] == "curve"


def test_a_curve_along_a_setting_the_runs_differ_in_is_refused_and_then_read_as_a_number() -> None:
    bad = full()
    bad["results"]["level_drop"].update({"shape": "curve", "along": ["valve"]})
    problems = contract.check_shape("level_drop", bad["results"]["level_drop"], DESIGN["protocol"])
    assert problems and "one run has one value of it" in problems[0]
    fake = Fake(roles=[bad, copy.deepcopy(bad)], comparisons=[NO_COMPARE])
    out = run(fake)
    assert "shape" not in metric(out, "level_drop") and out.parts["roles"]["left_out"]["shape"].keys() == {"level_drop"}
    assert metric(out, "level_trace")["shape"] == "curve"
    assert any("`level_drop`" in n and "one number" in n for n in out.notes)


def test_a_bound_against_the_plans_own_values_is_refused() -> None:
    # the plan's check expects the overflow share to be 0 at a closed valve: a bound above 0 would hold no such value
    assert any("expected value of the check `closed valve`" in x
               for x in contract.check_bound("overflow", {"min": 0.1, "max": 1, "why": "w"}, DESIGN))
    # the plan's own bound on the level drop is -5 to 50 cm: a physical bound of at least 60 lies outside it
    assert any("the plan's own bound" in x for x in contract.check_bound("level_drop", {"min": 60, "why": "w"}, DESIGN))
    # the plan gives the level drop in cm: a bound in another unit is refused
    assert any("give the bound in `cm`" in x
               for x in contract.check_bound("level_drop", {"max": 0.5, "unit": "m", "why": "w"}, DESIGN))
    assert contract.check_bound("level_drop", {"max": 50, "unit": "cm", "why": "w"}, DESIGN) == []
    assert contract.check_bound("overflow", None, DESIGN) == []
    assert contract.check_bound("overflow", {"why": "w"}, DESIGN)  # neither end given
    assert contract.check_bound("overflow", {"min": 1, "max": 0, "why": "w"}, DESIGN)
    bad = full()
    bad["results"]["overflow"]["bound"] = {"min": 0.1, "max": 1, "why": "w"}
    out = run(Fake(roles=[bad, copy.deepcopy(bad)], comparisons=[NO_COMPARE]))
    assert "range" not in metric(out, "overflow") and "overflow" in out.parts["roles"]["left_out"]["bound"]


def test_check_system_holds_ids_names_kinds_and_links() -> None:
    p = DESIGN["protocol"]
    assert contract.check_system(SYSTEM, p) == [] and contract.check_system(None, p) == []
    twice = {"parts": [SYSTEM["parts"][0], SYSTEM["parts"][0]]}
    assert any("given twice" in x for x in contract.check_system(twice, p))
    odd = {"parts": [{**SYSTEM["parts"][0], "kind": "pump"}, {**SYSTEM["parts"][1], "depth": 3}]}
    found = contract.check_system(odd, p)
    assert any("`kind` must be one of" in x for x in found) and any("`depth`" in x for x in found)
    self_link = {"parts": SYSTEM["parts"], "links": [{"from": "tank", "to": "tank"}]}
    assert any("to itself" in x for x in contract.check_system(self_link, p))
    # a digit inside a word is no number; a number of its own is
    assert contract.check_system({"parts": [{"id": "w", "name": "H2O layer", "carries": []}]}, p) == []
    assert contract.check_system({"parts": [{"id": "w", "name": "layer 0.5", "carries": []}]}, p)


def test_a_person_editing_a_bad_system_into_the_plan_is_a_gap_not_a_lost_study() -> None:
    d = copy.deepcopy(DESIGN)
    d["protocol"]["system"] = {"parts": [{"id": "tank", "name": "tank", "carries": ["nope"]}]}
    c = contract.from_design(d)
    assert c.system is None and any("studied system" in g and "left out" in g for g in c.gaps)
    assert {r.id for r in c.results} == {"level_drop", "overflow", "level_trace", "drop_theory"}


# --- the figures' reference lines ------------------------------------------------------------------------------------------


FIG = {"shows": ["level_drop"], "x": "valve", "series": "inlet", "panels": 1}


def test_a_figures_reference_line_is_asked_checked_and_kept_in_its_declaration() -> None:
    good = {"figures": {"level_drop.png": {**FIG, "references": ["drop_theory"]}}}
    fake = Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE], figures=[good])
    out = run(fake, fi_analysis=True)
    assert "drop_theory" in fake.prompts[-1][1] and "references" in fake.prompts[-1][1]
    assert out.design["protocol"]["figure_declarations"]["level_drop.png"]["references"] == ["drop_theory"]
    fig = contract.from_design(out.design, fi_analysis=True).figures[0]
    assert fig.declared and fig.references == ("drop_theory",)
    # the declarations are not frozen, as before
    assert "figure_declarations" not in frozen_protocol.canonical(out.design["protocol"])


def test_a_reference_fi_holds_none_of_is_asked_again_once_then_left_out_and_the_figure_is_declared() -> None:
    bad = {"figures": {"level_drop.png": {**FIG, "references": ["level_trace"]}}}
    fake = Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE], figures=[bad, copy.deepcopy(bad)])
    out = run(fake, fi_analysis=True)
    assert [p for p, _ in fake.prompts].count("figures") == 2
    assert "is no reference FI holds" in fake.prompts[-1][1]
    decl = out.design["protocol"]["figure_declarations"]["level_drop.png"]
    assert decl["shows"] == ["level_drop"] and "references" not in decl
    assert out.parts["figures"]["valid"] and "level_drop.png" in out.parts["figures"]["left_out"]["references"]
    assert any("reference lines" in n for n in out.notes)


def test_a_figure_answer_without_references_is_as_before() -> None:
    fake = Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE], figures=[{"figures": {"level_drop.png": FIG}}])
    out = run(fake, fi_analysis=True)
    assert [p for p, _ in fake.prompts].count("figures") == 1
    assert "references" not in out.design["protocol"]["figure_declarations"]["level_drop.png"]


# --- the existing range check holds every run to a physical bound ------------------------------------------------------


def test_the_range_check_fires_on_a_value_beyond_the_physical_bound() -> None:
    from core.engine_base import _asserting_design, _replicate_assertions

    out = run(Fake(roles=[full()], comparisons=[NO_COMPARE]))
    design = _asserting_design({"design": out.design})
    paths = {(a["path"], a["min"], a["max"]) for a in design["result_assertions"]}
    assert ("overflow", 0, 1) in paths and ("level_drop", None, 50) in paths
    assert any(a.path == "overflow" for a in _replicate_assertions({"design": out.design}))
    beyond = plausibility.check_design({"overflow": 1.2, "level_drop": 10.0}, design)
    assert [v.assertion.path for v in beyond] == ["overflow"]
    assert plausibility.check_design({"overflow": 0.4, "level_drop": 10.0}, design) == []
    # a value exactly on a physical bound is allowed (a share of 1 when every run overflows)
    on_edge = [a for a in plausibility.parse_assertions(design) if a.path == "overflow"]
    assert on_edge and on_edge[0].may_equal_bound
    # without the asked bound, the same value passes: the plan declared none for it
    plain = run(Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE]))
    assert plausibility.check_design({"overflow": 1.2, "level_drop": 10.0}, _asserting_design({"design": plain.design})) == []


def test_the_audit_view_of_the_plan_does_not_hold_what_fi_asked() -> None:
    out = run(Fake(roles=[full()], comparisons=[NO_COMPARE]))
    core_view = cp.without_answers(out.design)["protocol"]
    assert "system" not in core_view
    assert all("shape" not in m and "along" not in m and "range" not in m for m in core_view["metrics"])
    own = copy.deepcopy(DESIGN)
    own["protocol"]["metrics"][1]["range"] = {"min": 0, "max": 1, "reason": "the plan's own"}
    assert cp.without_answers(own)["protocol"]["metrics"][1]["range"]["reason"] == "the plan's own"


def test_an_extra_left_out_is_a_disclosed_row_of_the_quality_report(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "contract_parts.json").write_text(json.dumps({"form": "contract", "parts": {
        "roles": {"asks": 2, "valid": True, "problems": [], "left_out": {"bound": {"overflow": "x"}, "system": "y"}}}}),
        encoding="utf-8")
    rows = [r for r in quality_report._plan_problems(quality_report._Quest(tmp_path))
            if r["source"] == ".fi/contract_parts.json"]
    assert len(rows) == 2 and all(r["status"] == "disclosed" for r in rows)
    assert any("physical bound" in r["text"] and "`overflow`" in r["text"] for r in rows)
    assert not any("older form" in r["text"] for r in rows)


def test_a_curve_may_run_along_what_one_run_steps_through_but_not_a_result_or_a_constant() -> None:
    p = DESIGN["protocol"]
    for along in (["time"], ["x"], ["frequency"]):
        assert contract.check_shape("level_trace", {"shape": "curve", "along": along}, p) == []
    assert contract.check_shape("level_trace", {"shape": "map", "along": ["x", "y"]}, p) == []
    assert "number the study reports" in contract.check_shape("level_trace", {"shape": "curve", "along": ["overflow"]}, p)[0]
    assert "constant of the plan" in contract.check_shape("level_trace", {"shape": "curve", "along": ["tank_depth_cm"]}, p)[0]
    assert contract.check_shape("level_trace", {"shape": "curve", "along": ["step 2"]}, p)
    assert contract.check_shape("level_trace", {"shape": "map", "along": ["x"]}, p)  # a map runs over two
    assert contract.check_shape("level_trace", {"shape": "number", "along": ["time"]}, p)


def test_plan_md_says_the_shape_the_physical_bound_and_the_studied_system_in_plain_words() -> None:
    from core import plan

    out = run(Fake(roles=[full()], comparisons=[NO_COMPARE]))
    text = "\n".join(plan._returns_lines(out.design["protocol"]))
    assert "a curve along `time` in each run" in text and "by its physics between 0 and 1 (a probability)" in text
    assert "The studied system:" in text and "- water tank (`tank`, a part): `tank_depth_cm`, `level_drop`" in text
    assert "- inlet pipe to water tank: water" in text
    plain = run(Fake(roles=[copy.deepcopy(PLAIN)], comparisons=[NO_COMPARE]))
    text = "\n".join(plan._returns_lines(plain.design["protocol"]))
    assert "studied system" not in text and "physics" not in text and "curve" not in text


def test_a_curve_never_runs_along_a_value_every_run_goes_through_together() -> None:
    p = {**DESIGN["protocol"], "in_each_run": {"gauge": ["float", "pressure"]}}
    found = contract.check_shape("level_trace", {"shape": "curve", "along": ["gauge"]}, p)
    assert found and "FI records a number" in found[0] and "depends_on" in found[0]


# --- a plain title for each sub-study and group of results named in code -----------------------------------------------


def coded() -> dict[str, Any]:
    d = copy.deepcopy(DESIGN)
    d["protocol"]["sub_studies"] = [{"name": "valve_scan", "vary": ["valve"], "hold": {"inlet": 1}},
                                    {"name": "Inlet scan", "vary": ["inlet"], "hold": {"valve": 0.5}}]
    for m in d["protocol"]["metrics"][:2]:
        m["family"] = "level_group"
    d["protocol"]["labels"] = {"valve": "Valve opening"}
    return d


def test_only_names_in_code_without_a_reader_name_are_asked_a_title() -> None:
    assert contract.titles_to_ask(coded()["protocol"]) == {"valve_scan": "a sub-study", "level_group": "a group of results"}
    assert contract.titles_to_ask(DESIGN["protocol"]) == {}
    assert contract.check_title("valve_scan", "Scan of the valve opening") == []
    for bad in ("valve scan", "Valve_Scan", "", None, "x" * 81):
        assert contract.check_title("valve_scan", bad), bad


def test_a_title_is_asked_in_the_roles_question_and_kept_as_the_plans_reader_name() -> None:
    answer = {**full(), "titles": {"valve_scan": "Scan of the valve opening", "level_group": "Water level"}}
    fake = Fake(roles=[answer], comparisons=[NO_COMPARE])
    out = run(fake, coded())
    assert [p for p, _ in fake.prompts] == ["roles", "comparisons"]
    assert "- valve_scan (a sub-study)" in fake.prompts[0][1] and "Inlet scan" not in fake.prompts[0][1]
    labels = out.design["protocol"]["labels"]
    assert labels == {"valve": "Valve opening", "valve_scan": "Scan of the valve opening", "level_group": "Water level"}
    assert contract.from_design(out.design).labels["valve_scan"] == "Scan of the valve opening"
    # FI's answer is not part of what the methodology audit judged; the plan's own label is
    assert cp.without_answers(out.design)["protocol"]["labels"] == {"valve": "Valve opening"}
    # a group's title in the labels is never asked about as a name the plan uses without declaring it
    assert "level_group" not in contract.undeclared_names(out.design)


def test_a_title_that_is_the_code_name_again_is_asked_once_more_then_the_name_stays() -> None:
    answer = {**full(), "titles": {"valve_scan": "valve scan", "level_group": "Water level"}}
    fake = Fake(roles=[answer, copy.deepcopy(answer)], comparisons=[NO_COMPARE])
    out = run(fake, coded())
    assert [p for p, _ in fake.prompts].count("roles") == 2 and "name in code again" in fake.prompts[1][1]
    labels = out.design["protocol"]["labels"]
    assert "valve_scan" not in labels and labels["level_group"] == "Water level"
    assert out.parts["roles"]["left_out"]["titles"].keys() == {"valve_scan"}
    assert any("`valve_scan`" in n and "printed as written" in n for n in out.notes)


def test_a_group_named_like_a_result_keeps_that_results_reader_name() -> None:
    d = coded()
    d["protocol"]["metrics"][2]["family"] = "level_trace"  # the group of a result is named as that result's id
    d["protocol"]["labels"]["level_trace"] = "Water level over the hour"
    assert "level_trace" not in contract.titles_to_ask(d["protocol"])
    d["protocol"]["labels"].pop("level_trace")
    assert "level_trace" not in contract.titles_to_ask(d["protocol"])  # never asked, so never written over it
    d["protocol"]["labels"]["level_trace"] = "Water level over the hour"
    assert cp.without_answers(d)["protocol"]["labels"]["level_trace"] == "Water level over the hour"


def test_a_result_with_the_plans_bound_and_a_physical_bound_is_held_to_both() -> None:
    from core.engine_base import _asserting_design

    out = run(Fake(roles=[full()], comparisons=[NO_COMPARE]))
    design = _asserting_design({"design": out.design})
    below = plausibility.check_design({"overflow": 0.5, "level_drop": -6.0}, design)
    beyond = plausibility.check_design({"overflow": 0.5, "level_drop": 51.0}, design)
    assert [v.assertion.min for v in below] == [-5]  # only the plan's own bound reaches below
    assert len(beyond) == 2  # both bounds say 50 cm: each finding names its own reason
