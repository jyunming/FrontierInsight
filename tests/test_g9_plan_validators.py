"""The request to complete the plan states the sub-study rule the reader holds, one unreadable sub-study entry does not cost
the other entries their answer, the model's raw answer is kept, and the plan prompt says extra runs live only in the grid."""
from __future__ import annotations

from pathlib import Path

import pytest

from core import engine as engine_mod
from core import plan
from tests.test_grid_size_at_plan import BIG, NAMES, PARTS, Model, TOPIC, _draft, _engine

GRID = {"kind": ["a", "b"], "size": [1, 2, 3], "depth": [10, 20]}


def test_the_sub_study_rule_in_the_request_is_the_rule_the_reader_applies() -> None:
    """Old code told the model a setting a part neither varies nor holds "stays at its first value", while the reader
    refuses exactly that (the answer was refused as unreadable). The request now says every other setting must be held."""
    text = plan.part_request("a topic", {"sub_studies": "entry 1 does not say the value it holds `depth` at"})
    assert "stays at its first value" not in text
    assert "EVERY other setting" in text and "refused" in text
    # the reader really does refuse the loose entry the old sentence allowed
    fixed, why = plan.normalize_protocol({"grid": GRID, "sub_studies": [
        {"name": "k", "vary": ["kind", "size"]}, {"name": "d", "vary": ["depth"]}]})
    assert fixed is None and "does not say the value it holds" in why


def test_readable_sub_studies_keeps_the_entries_that_read_and_names_the_one_that_does_not() -> None:
    """Old code had no entry-by-entry reading of sub_studies: one entry varying an unknown setting refused the list."""
    subs = [{"name": "k", "vary": ["kind", "size"], "hold": {"depth": 10}},
            {"name": "d", "vary": ["depth"], "hold": {"kind": "a", "size": 1}},
            {"name": "bad", "vary": ["nowhere"], "hold": {}}]
    kept, reasons = plan.readable_sub_studies(subs, GRID, None)
    assert [s["name"] for s in kept] == ["k", "d"]
    assert len(reasons) == 1 and "`nowhere`" in reasons[0] and "entry 3" in reasons[0]
    fixed, why = plan.normalize_protocol({"grid": GRID, "sub_studies": kept})
    assert why is None, why


def test_a_list_that_leaves_a_value_unrun_is_still_refused_after_the_unreadable_entry_is_left_out() -> None:
    """Known positive: leaving the bad entry out must not make a study that never runs some grid value acceptable."""
    subs = [{"name": "k", "vary": ["kind", "size"], "hold": {"depth": 10}},
            {"name": "bad", "vary": ["nowhere"], "hold": {}}]
    kept, reasons = plan.readable_sub_studies(subs, GRID, None)
    assert reasons
    fixed, why = plan.normalize_protocol({"grid": GRID, "sub_studies": kept})
    assert fixed is None and "`depth`" in why and "never be run" in why


@pytest.mark.asyncio
async def test_one_bad_sub_study_entry_does_not_cost_the_answer_its_other_entries(tmp_path: Path) -> None:
    """Old code: one entry the reader refuses made `_keep_only_the_parts` raise, so the whole list (which was fine) was lost."""
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    answer = {"protocol": {"sub_studies": [*PARTS, {"name": "extra", "vary": ["nowhere"], "hold": {}}]}}
    kept = eng._keep_only_the_parts(text, answer, ["sub_studies"])
    names = [s["name"] for s in plan.parse(kept).design["protocol"]["sub_studies"]]
    assert names == [p["name"] for p in PARTS]
    assert any("`nowhere`" in r for r in eng._answer_refused)


@pytest.mark.asyncio
async def test_a_list_with_only_unreadable_entries_is_still_refused(tmp_path: Path) -> None:
    """Known positive: nothing readable in the answer means the answer is still not used."""
    eng = _engine(tmp_path, [])
    Model(eng, [], _draft(BIG))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="nowhere"):
        eng._keep_only_the_parts(text, {"protocol": {"sub_studies": [{"name": "x", "vary": ["nowhere"], "hold": {}}]}},
                                 ["sub_studies"])


def test_the_plan_models_raw_answers_are_kept_without_save_model_calls(tmp_path: Path) -> None:
    """Old code kept a refused plan answer only when `output.save_model_calls` was on (default off), so a refusal could not be
    read beside the answer. Now every answer is written to `.fi/plan_answers/`, bounded."""
    eng = _engine(tmp_path, [])
    eng._keep_plan_answer("first answer")
    eng._keep_plan_answer("x" * (eng._PLAN_ANSWER_CHARS + 10))
    folder = eng.fi_dir / "plan_answers"
    files = sorted(folder.glob("*.txt"))
    assert [f.name for f in files] == ["001.txt", "002.txt"]
    assert files[0].read_text(encoding="utf-8") == "first answer"
    assert len(files[1].read_text(encoding="utf-8")) == eng._PLAN_ANSWER_CHARS
    for _ in range(eng._PLAN_ANSWERS_KEPT + 5):
        eng._keep_plan_answer("more")
    assert len(list(folder.glob("*.txt"))) == eng._PLAN_ANSWERS_KEPT


def test_the_plan_prompt_says_a_run_the_method_describes_must_be_in_the_grid() -> None:
    """Old prompt let a method text name an extra setting the grid did not hold, and nothing compared them."""
    source = Path(engine_mod.__file__).read_text(encoding="utf-8")
    assert "that run must be a value of a `grid` setting or a part of `sub_studies`" in source


BLOCK = {"objective": {"quantity": "heat_flow", "direction": "maximise", "unit": "W"},
         "design_variables": [{"name": "width", "low": 1.0, "high": 5.0, "kind": "continuous"}],
         "baseline": {"values": {"width": 2.0}, "source": "the topic"},
         "evaluation_budget": {"starts": 2, "per_start": 10},
         "grid": {"kind": ["a", "b"], "size": [1, 2]},
         "fixed": {"size": 1},
         "sub_studies": [{"name": "kinds", "vary": ["kind"], "hold": {"size": 1}},
                         {"name": "sizes", "vary": ["size"], "hold": {"kind": "a"}}]}


def test_the_request_for_a_search_states_the_rules_the_reader_applies() -> None:
    """Old request said only "remove a plain `protocol.grid`": a plan that also had a top-level `sub_studies` was then
    refused ("cannot stand beside `protocol.optimisation`"), and `baseline.values` was described as "one value for each
    design variable", so a model wrote a bare list and was refused ("must give the design to beat")."""
    from core import optimisation_plan as op

    text = op.ask_request("a topic", ["the block is missing"])
    assert op.SCAN_PARTS_RULE in text and op.BASELINE_VALUES_RULE in text
    assert "`.sub_studies`" in text and "a mapping from each design variable's name" in text
    # the reader refuses exactly what the request now tells the model not to write, and quotes the same rule
    _fixed, why = plan.normalize_protocol({"optimisation": BLOCK, "sub_studies": BLOCK["sub_studies"]})
    assert "cannot stand beside `protocol.optimisation`" in why and op.SCAN_PARTS_RULE in why
    bare = {**BLOCK, "baseline": {"values": [2.0], "source": "the topic"}}
    _fixed, why = op.normalize(bare)
    assert "a mapping from each design variable's name" in why


@pytest.mark.asyncio
async def test_a_search_answer_replaces_the_plans_top_level_scan_parts(tmp_path: Path) -> None:
    """Old code dropped only the plan's top-level `grid` when it took the search block in, so the plan's own top-level
    `sub_studies` stayed beside the block and the merged plan was refused ("the plan with your block in it cannot be
    read"), twice, whatever the model answered."""
    eng = _engine(tmp_path, [])
    parts = [{"name": "k", "vary": ["kind", "size"], "hold": {"depth": 10}},
             {"name": "d", "vary": ["depth"], "hold": {"kind": "a", "size": 1}}]
    Model(eng, [], _draft(GRID, sub_studies=parts))
    await eng._node_plan({"topic": TOPIC, "iteration": 0})
    text = plan.plan_path(eng.quest_root).read_text(encoding="utf-8")
    assert "sub_studies" in plan.parse(text).design["protocol"]
    kept = eng._keep_only_the_search(text, {"study_type": "find_best_design", "protocol": {"optimisation": BLOCK}})
    protocol = plan.parse(kept).design["protocol"]
    assert set(protocol) >= {"optimisation"} and not {"grid", "sub_studies", "in_each_run", "numerical_settings"} & set(protocol)
    assert [s["name"] for s in protocol["optimisation"]["sub_studies"]] == ["kinds", "sizes"]


def test_an_edit_FI_would_make_to_its_own_plan_is_not_made_when_the_reader_then_refuses_the_plan() -> None:
    """Old code wrote every recomputed expected value into plan.md without asking the reader again: a metric's reference
    that agreed with the typed number then disagreed with the check, plan.md became unreadable, and the design step dropped
    the whole plan. Now such a check keeps the typed number and the disagreement is returned for the record."""
    from core import oracle_forms
    from tests import reference_values_plans as P

    protocol = P.with_references(P.load("kin_s1", "protocol.json"), {"coverage_Ea": {"formula": "reference_coverage"}})
    names = [o["name"] for o in protocol["oracles"]]
    target = "low_noise_coverage_95"
    assert target in names
    # a typed number the formula disagrees with, and a formula that gives another one
    def with_formula(typed: float, formula: str) -> str:
        oracles = [{**o, "expected": typed, "expected_formula": formula} if o["name"] == target else o
                   for o in protocol["oracles"]]
        return plan.render("a topic", None, {"hypothesis": "h", "protocol": {**protocol, "oracles": oracles}})

    original = next(o for o in protocol["oracles"] if o["name"] == target)["expected"]
    text = with_formula(original, "0.9")  # the metric's reference (reference_coverage) agrees with the typed number
    assert plan.parse(text).design is not None
    refused: list = []
    new_text, done = oracle_forms.apply_formulas(text, refused)
    assert plan.parse(new_text).design is not None  # the plan FI wrote still reads
    assert done == [] and new_text == text
    assert [r["name"] for r in refused] == [target] and refused[0]["typed"] == original and refused[0]["formula_value"] == 0.9
    assert "disagrees with a known-answer check" in refused[0]["why"]


def test_an_edit_that_keeps_the_plan_readable_is_still_made() -> None:
    """Known positive: a recomputed expected value no other part of the plan is held against is still written."""
    from core import oracle_forms
    from tests import reference_values_plans as P

    protocol = P.load("kin_s1", "protocol.json")
    target = next(o["name"] for o in protocol["oracles"] if o["name"] != "low_noise_coverage_95")
    oracles = [{**o, "expected": 123.0, "expected_formula": "2 * 3"} if o["name"] == target else o for o in protocol["oracles"]]
    text = plan.render("a topic", None, {"hypothesis": "h", "protocol": {**protocol, "oracles": oracles}})
    refused: list = []
    new_text, done = oracle_forms.apply_formulas(text, refused)
    assert [d["name"] for d in done] == [target] and done[0]["now"] == 6.0 and refused == []
    assert plan.parse(new_text).design is not None
