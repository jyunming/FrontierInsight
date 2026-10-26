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
