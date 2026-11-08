"""A number the method works out from the plan's constants reaches the paper as {{calc:EXPR}}: the whole-paper writer is
told so (not only the review's message), and FI fills it, also inside a formula."""

from __future__ import annotations

from pathlib import Path

from core import paper_values, typed_numbers

PLAN = {"constants": {"flow": 2.5, "area": 40.0, "gain": 1.6}}
TOPIC = "Cool a chip with a heat sink; the pump flow is 2.5 and the area is 40."


def test_the_writer_is_told_about_calc_in_its_rules() -> None:
    assert "{{calc:" in paper_values.WRITER_RULES
    assert "cutoff" in paper_values.WRITER_RULES


def test_the_patch_prompt_and_the_hit_message_name_a_cutoff_from_a_formula() -> None:
    prompt = (Path(__file__).resolve().parents[1] / "agents" / "write_patch.md").read_text(encoding="utf-8")
    assert "cutoff or bound" in prompt and "{{calc:" in prompt
    hit = typed_numbers.hits("Below $a/(b c)-1 = 0.5625$ it fails.", typed_numbers.plan_numbers(PLAN, TOPIC))[0]
    assert "cutoff or bound" in hit and "{{calc:EXPR}}" in hit


def test_a_formula_value_is_filled_from_the_plans_numbers_and_an_unplanned_number_is_not() -> None:
    ok = paper_values.calc("40 / (2.5 * 1.6) - 1", PLAN, TOPIC)
    assert ok.filled and ok.text == "9"
    bad = paper_values.calc("40 / (2.5 * 1.7) - 1", PLAN, TOPIC)  # 1.7 is no number of the plan
    assert not bad.filled
    # a typed result-looking number that no plan number gives is still not allowed
    assert typed_numbers.hits("The cutoff is 9.4 here.", typed_numbers.plan_numbers(PLAN, TOPIC))


def test_the_writer_is_told_every_function_a_calc_may_call() -> None:
    """The writer is told the calculator's whole list (a writer not told of ndtr wrote a wrong closed form instead)."""
    from core import oracle_forms

    prompt = (Path(__file__).resolve().parents[1] / "agents" / "write_patch.md").read_text(encoding="utf-8")
    for name in [*oracle_forms.FUNCTIONS, "ndtr", "ndtri", "t_quantile"]:
        assert name + "(" in paper_values.WRITER_RULES or name + "," in paper_values.WRITER_RULES, name
    for name in ("ndtr", "ndtri", "t_quantile"):
        assert name + "(" in prompt, name
    assert "<CALC_FUNCTIONS>" not in paper_values.WRITER_RULES


def test_a_normal_tail_power_is_filled_from_the_plans_numbers() -> None:
    plan = {"constants": {"alpha": 0.05, "shift": 2.5}}
    got = paper_values.calc("1 - ndtr(ndtri(1 - 0.05) - 2.5)", plan, "")
    assert got.filled and got.text == "0.8038"


def test_a_formula_may_use_the_confidence_level_of_fi_own_intervals() -> None:
    """FI computes its 95% intervals itself, so the level (0.95) and its two-sided quantile levels (0.975, 0.025) are numbers
    FI fixes, not numbers the plan must state; before, ``t_quantile(0.975, 3)`` read ``not recorded``."""
    plan = {"constants": {"n_points": [2, 4, 8]}}
    t = paper_values.calc("t_quantile(0.975, 3)", plan, "")
    assert t.filled and t.text == "3.182"
    z = paper_values.calc("ndtri(0.975)", plan, "")
    assert z.filled and z.text == "1.96"
    assert paper_values.calc("ndtri(0.95)", plan, "").filled
    assert paper_values.calc("ndtri(0.025)", plan, "").filled


def test_only_the_level_of_fi_own_intervals_is_added_to_what_a_formula_may_use() -> None:
    plan = {"constants": {"n_points": [2, 4, 8]}}
    assert not paper_values.calc("t_quantile(0.9, 3)", plan, "").filled  # 0.9 is no level FI uses
    assert not paper_values.calc("7 * 3", plan, "").filled  # no stated number at all
    assert not paper_values.calc("0.97 * 3", plan, "").filled
    assert not paper_values.calc("40 / (2.5 * 1.7) - 1", PLAN, TOPIC).filled


def test_a_trial_count_of_the_record_is_a_number_only_as_degrees_of_freedom() -> None:
    cells = [{"ok": 30, "metrics": {"err": {"values": [0.1 * i for i in range(1, 1001)]}, "bad": 5}}]
    rec = paper_values.Records(cells=cells, protocol={"constants": {"n_points": [2, 4]}})
    got = paper_values.calc("t_quantile(0.975, 29)", rec.protocol, "", rec)
    assert got.filled and got.text == "2.045"
    assert not paper_values.calc("t_quantile(0.975, 29)", rec.protocol, "").filled  # without the record 29 is no number
    for expr in ("29 * 9", "30 * 7", "0.95 * 7", "t_quantile(0.975, 999)", "t_quantile(0.975, 28)", "t_quantile(0.975, 29) * 30"):
        assert not paper_values.calc(expr, rec.protocol, "", rec).filled, expr
