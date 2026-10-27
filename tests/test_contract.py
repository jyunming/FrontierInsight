"""The study contract (core/contract.py): a plan's design read into typed parts by fixed rules, with what could not be
placed named in ``gaps`` instead of guessed."""

from __future__ import annotations

import pytest

from core import contract as C


def _old_plan() -> dict:
    """An older plan: results with no role, a precision target written as a sentence starting with an id, figures
    given only by file name, bounds on a result and on another path."""
    return {
        "hypothesis": "the ratio grows with the load",
        "method": "run each setting many times",
        "figures_planned": ["figures/ratio.png", "figures/spread.png"],
        "result_assertions": [{"path": "ratio", "min": 0, "max": 1, "reason": "a share"},
                              {"path": "other.value", "min": 0, "max": 10}],
        "protocol": {
            "grid": {"load": [1, 2, 4], "method": ["plain", "careful"]},
            "in_each_run": {"variant": ["a", "b"]},
            "constants": {"base_seed": 7, "width": 3.0}, "thresholds": {"limit": 0.5},
            "runs_per_setting": 20, "seed_policy": "one seed per trial",
            "metrics": [{"id": "ratio", "kind": "proportion", "estimand": "share of hits", "unit": "a trial"},
                        {"id": "spread", "kind": "mean", "estimand": "spread", "unit": "a trial", "label": "Spread"}],
            "precision": {"target_half_width": 0.05, "metric": "ratio (at the middle load)", "reason": "enough"},
            "labels": {"ratio": "Hit share"},
            "grid_note": "loads chosen to span the range",
            "steps": [{"id": "S1", "name": "draw"}],
        },
    }


def test_an_older_plan_is_read_by_rule_and_what_it_lacks_is_named() -> None:
    c = C.from_design(_old_plan())
    assert [r.id for r in c.results] == ["ratio", "spread"]
    assert {r.role for r in c.results} == {"result"}
    ratio = c.result("ratio")
    assert ratio is not None and ratio.kind == "proportion" and ratio.observation == "a trial"
    assert ratio.label == "Hit share" and ratio.range == C.Range(min=0, max=1, reason="a share")
    assert c.checks.assertions == ({"path": "other.value", "min": 0, "max": 10},)
    assert c.settings.compared == {"load": (1, 2, 4), "method": ("plain", "careful")}
    assert c.settings.in_each_run == {"variant": ("a", "b")}
    assert c.settings.fixed == {"base_seed": 7, "width": 3.0, "limit": 0.5}
    assert c.budget.runs_per_setting == 20
    assert c.budget.precision == C.Precision(target="ratio", half_width=0.05, reason="enough")
    assert [f.declared for f in c.figures] == [False, False] and c.figures[0].file == "figures/ratio.png"
    assert c.model["steps"] == [{"id": "S1", "name": "draw"}]
    assert c.notes["seed_policy"] == "one seed per trial" and c.notes["grid_note"].startswith("loads")
    assert any("no role" in g for g in c.gaps)
    assert any("2 figure(s) only by file name" in g for g in c.gaps)
    assert not any("kept only as notes" in g for g in c.gaps)  # a *_note key and steps are placed, not a gap


def test_a_precision_target_that_names_no_declared_result_is_left_out_and_said() -> None:
    plan = _old_plan()
    plan["protocol"]["precision"]["metric"] = "the hit share at the middle load"
    c = C.from_design(plan)
    assert c.budget.precision is None and c.budget.runs_per_setting == 20
    assert any("no precision target" in g for g in c.gaps)
    plan["protocol"]["precision"]["metric"] = "ratios"  # starts with an id but goes on with letters: not that id
    assert C.from_design(plan).budget.precision is None


def test_a_plan_in_the_contract_form_is_read_with_its_roles_comparisons_derived_and_figures() -> None:
    plan = _old_plan()
    p = plan["protocol"]
    p["metrics"][0].update(role="physical_bound", depends_on=["load"])
    p["metrics"].append({"id": "wall_s", "kind": "mean", "role": "timing", "estimand": "time of the solve"})
    p["comparisons"] = [{"id": "ratio_gain", "result": "ratio", "across": "method", "values": ["plain", "careful"]}]
    p["derived"] = [{"id": "covered", "inputs": ["spread"], "kind": "proportion"}]
    p["figures"] = [{"file": "figures/ratio.png", "shows": ["ratio", "ratio_gain"], "x": "load", "series": "method"}]
    c = C.from_design(plan)
    assert c.role_of("ratio") == "physical_bound" and c.role_of("wall_s") == "timing"
    assert c.role_of("anything_else") == "diagnostic"
    assert c.result("ratio").depends_on == ("load",)
    assert c.comparisons[0].how == "difference" and c.derived[0].formula is None
    assert c.figures == (C.Figure(id="fig1", file="figures/ratio.png", shows=("ratio", "ratio_gain"), x="load",
                                  series="method"),)
    assert not any("no role" in g or "file name" in g for g in c.gaps)


def test_a_part_that_refers_to_something_undeclared_is_left_out_and_the_rest_kept() -> None:
    plan = _old_plan()
    p = plan["protocol"]
    p["comparisons"] = [{"id": "good", "result": "ratio", "across": "load"},
                        {"id": "bad", "result": "missing_result", "across": "load"}]
    p["figures"] = [{"file": "a.png", "shows": ["ratio"], "x": "load"},
                    {"file": "b.png", "shows": ["ratio"], "x": "not_a_setting"}]
    p["metrics"][1]["depends_on"] = ["load", "nowhere"]
    c = C.from_design(plan)
    assert [x.id for x in c.comparisons] == ["good"]
    assert [f.file for f in c.figures] == ["a.png"]
    assert c.result("spread").depends_on == ("load",)
    assert any("comparison `bad`" in g for g in c.gaps) and any("figure `fig2`" in g for g in c.gaps)
    assert any("nowhere" in g for g in c.gaps)


def test_the_contract_refuses_what_cannot_hold() -> None:
    with pytest.raises(ValueError, match="more than once"):
        C.StudyContract(results=(C.Result(id="x"), C.Result(id="x")))
    with pytest.raises(ValueError, match="above its max"):
        C.Range(min=2, max=1)
    with pytest.raises(ValueError):
        C.Figure(id="f", file="f.png", panels=C.MAX_PANELS + 1)
    with pytest.raises(ValueError, match="precision target"):
        C.StudyContract(results=(C.Result(id="x"),),
                        budget=C.Budget(precision=C.Precision(target="y", half_width=0.1)))
    with pytest.raises(ValueError):
        C.Result(id="x", role="headline")


@pytest.mark.parametrize("design", [None, {}, {"protocol": "not a mapping"},
                                    {"protocol": {"metrics": ["junk", {"id": "x", "kind": "count"}], "grid": [1, 2],
                                                  "precision": {"metric": 3}, "runs_per_setting": 0}}])
def test_a_plan_of_any_shape_gives_a_contract_and_never_raises(design) -> None:
    c = C.from_design(design)
    assert isinstance(c, C.StudyContract)
    if design and isinstance(design.get("protocol"), dict) and design["protocol"].get("metrics"):
        assert any("result `x`" in g for g in c.gaps) and any("run budget" in g for g in c.gaps)
