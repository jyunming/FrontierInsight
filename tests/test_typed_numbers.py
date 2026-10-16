"""Rule R1: the writer may type only a number of the plan or the topic, a label, or a small integer inside math."""
from __future__ import annotations

import inspect
from decimal import Decimal

from core import typed_numbers as tn

PROTOCOL = {
    "grid": {"noise": [0.005, 0.05], "n_points": [5, 10]},
    "runs_per_setting": 300,
    "seed": 1234,
    "thresholds": {"coverage_nominal": 0.95},
    "constants": {"z": 1.96, "R": 8.314},
    "reference": {"expected_formula": "k = 0.25 * exp(-Ea / (R*T))"},
}
TOPIC = "Estimate the activation energy of a first-order reaction at 350 K."


def _plan():
    return tn.plan_numbers(PROTOCOL, TOPIC)


def _typed(text: str) -> list[str]:
    return [n.token for n in tn.typed_outside_the_plan(text, _plan())]


def test_plan_numbers_hold_fields_constants_formula_and_topic():
    plan = _plan()
    for v in ("0.005", "0.05", "5", "10", "0.95", "1.96", "8.314", "0.25", "350"):
        assert Decimal(v) in plan, v


def test_run_facts_are_not_plan_numbers():
    plan = _plan()
    assert Decimal("300") not in plan and Decimal("1234") not in plan


def test_a_plan_number_may_be_typed():
    assert _typed("Noise was 0.05 and R = 8.314 with z = 1.96 at 350 K.") == []


def test_a_percent_of_a_plan_number_is_the_plan_number():
    assert _typed("The nominal level is 95%.") == []


def test_a_number_that_is_not_the_plan_is_found():
    assert _typed("The mean bias was 0.8244 in the main grid.") == ["0.8244"]


def test_there_is_no_tolerance():
    assert _typed("The noise was 0.0500001.") == ["0.0500001"]
    assert "tol" not in inspect.signature(tn.allowed).parameters
    assert "tol" not in inspect.signature(tn.hits).parameters


def test_labels_are_structural():
    assert _typed("See Figure 2, Table 3 and Section 4; Figures 1 and 2 agree [4].") == []


def test_a_small_integer_in_math_is_allowed_but_not_in_prose():
    assert _typed("We fit $y = x^2 + 3x$.") == []
    assert _typed("We fit $y = 73x$.") == ["73"]
    assert _typed("There were 7 groups.") == ["7"]


def test_the_fi_marker_is_not_the_writers_text():
    text = "Coverage was <!-- fi:v {{v:coverage @ noise=0.05}} -->0.9312<!-- /fi:v --> there."
    assert _typed(text) == []
    assert "0.9312" not in tn.writers_text(text)


def test_code_urls_references_and_html_comments_are_left_out():
    text = "Run `x = 4321`. See https://x.org/9876. <!-- 5555 -->\n\n## References\n\n1. Smith 2019, vol 3456."
    assert _typed(text) == []


def test_a_cited_sentence_is_left_to_the_claim_check():
    text = "Prior work found a barrier of 71.3 kJ/mol [3]. We found 12.7 here."
    assert _typed(text) == ["12.7"]
    assert tn.report(text, _plan()).left_to_claim_check == 1


def test_hit_shape_quotes_the_sentence_once_per_sentence():
    out = tn.hits("The bias was 0.8244 and the spread was 0.1234. Noise 0.05 is fine.", _plan())
    assert len(out) == 1
    assert out[0].startswith("paper_record_mismatch: you typed 0.8244 and 0.1234, which are not numbers")
    assert "— “The bias was 0.8244 and the spread was 0.1234." in out[0]


def test_many_sentences_give_a_further_hit_without_a_quote():
    text = " ".join(f"The value was {i + 11}.5{i}." for i in range(tn.MAX_SENTENCES + 3))
    out = tn.hits(text, _plan())
    assert len(out) == tn.MAX_SENTENCES + 1
    assert "a further 3 typed number" in out[-1] and "“" not in out[-1]


def test_remove_replaces_each_number_and_lists_it():
    text = "The bias was 0.8244 at noise 0.05."
    out, taken = tn.remove(text, _plan())
    assert out == f"The bias was {tn.REMOVED} at noise 0.05."
    assert taken == [("0.8244", text)]


def test_a_paper_with_only_plan_numbers_has_no_hits():
    assert tn.hits("Noise 0.005 and 0.05, n = 5 and 10, 95% intervals.", _plan()) == []


def test_only_fis_own_block_hides_a_typed_number():
    own = "<!-- fi:fact-seed -->the seed was 4321<!-- /fi:fact-seed -->"
    other = "<!-- fi:foo -->the mean was 0.7777<!-- /fi:foo -->"
    assert _typed(own) == []
    assert "0.7777" in _typed(other)
