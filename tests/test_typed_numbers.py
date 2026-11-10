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


def test_a_run_figure_and_the_caption_fi_wrote_for_it_are_not_the_writers_text():
    text = ("Text.\n\n![**Figure 1.** Marked: limit at 0.4375.](figures/a.png){width=80%}\n\n"
            "![An image of 0.8125 elsewhere](other/b.png)\n")
    assert "0.4375" not in tn.writers_text(text)
    assert _typed(text) == ["0.8125"], "an image that is not one of the run's figures is still read"


def test_code_urls_references_and_html_comments_are_left_out():
    text = "See https://x.org/9876. <!-- 5555 -->\n\n## References\n\n1. Smith 2019, vol 3456."
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


def test_untraced_lists_each_number_with_its_sentence_and_changes_nothing():
    text = "The bias was 0.8244 at noise 0.05."
    assert tn.untraced(text, _plan()) == [("0.8244", text)]
    assert not hasattr(tn, "remove")
    # Known positive: a number of the plan is traced, so it is not listed.
    assert tn.untraced("Noise 0.05 and n = 10.", _plan()) == []


def test_a_paper_with_only_plan_numbers_has_no_hits():
    assert tn.hits("Noise 0.005 and 0.05, n = 5 and 10, 95% intervals.", _plan()) == []


def test_only_fis_own_block_hides_a_typed_number():
    own = "<!-- fi:fact-seed -->the seed was 4321<!-- /fi:fact-seed -->"
    other = "<!-- fi:foo -->the mean was 0.7777<!-- /fi:foo -->"
    assert _typed(own) == []
    assert "0.7777" in _typed(other)


# --- an inline enumeration "(1) ... (2) ..." is document structure, not a result ---------------------------------------

LIMITATIONS = (
    "**Limitations.** (1) The shared noise across σ levels means pooled intervals, the pooled slope of sd against n, and "
    "cross-n coverage contrasts are overstated in precision. (2) The trial count per setting is below the plan's minimum. "
    "(3) The run used a single seed set with no replication. (4) The time window was set from the true k, so spread is "
    "optimistic. (5) The design covers one Ea, one 1/T span and homoscedastic Gaussian noise on exact temperatures, so "
    "robustness is not shown. (6) No comparison with a linearised ln c versus t fit was made. (7) The checks on tightening "
    "the tolerance to 1e-12 and rescaling the weights were not reported, so they are not verified. (8) The plan's list of "
    "simplifications did not name the per-temperature k fit or the Ea–ln A correlation. (9) The topic asked for one "
    "script; the code is split over several files. (10) The discussion rests on few sources."
)


def _typed_empty_plan(text: str) -> list[str]:
    return [n.token for n in tn.typed_outside_the_plan(text, frozenset())]


def test_enumeration_labels_of_a_real_limitations_paragraph_are_not_numbers():
    left = _typed_empty_plan(LIMITATIONS)
    assert left == ["1", "1e-12"], left  # "1/T" and the tolerance are the paragraph's own numbers, the ten labels are not
    taken = tn.untraced(LIMITATIONS, frozenset())
    assert not any(tok == str(i) and f"({i})" in sent for i in range(2, 11) for tok, sent in taken)
    assert "1e-12" in " ".join(tok for tok, _ in taken)  # a real number of the paragraph is still checked


def test_enumeration_blanking_keeps_offsets():
    assert len(tn.writers_text(LIMITATIONS)) == len(LIMITATIONS)


def test_a_fabricated_result_in_parentheses_is_still_caught():
    assert _typed_empty_plan("Coverage was near nominal (0.93) in every setting.") == ["0.93"]


def test_a_lone_parenthesised_count_is_still_checked():
    assert _typed_empty_plan("The estimator failed in a few runs (6) of the grid.") == ["6"]


def test_a_fabricated_number_inside_an_enumeration_item_is_caught():
    text = "Limits. (1) One seed set. (2) Gaussian noise only. (3) Coverage was 0.871 at n = 5."
    assert _typed_empty_plan(text) == ["0.871", "5"]


def test_a_run_that_is_not_consecutive_is_not_structure():
    text = "Notes. (1) One seed set. (3) Gaussian noise only. (7) Coverage was low."
    assert _typed_empty_plan(text) == ["1", "3", "7"]


def test_a_label_run_does_not_cross_paragraphs():
    text = "(1) First point here.\n\nUnrelated text (2) with a count."
    assert _typed_empty_plan(text) == ["1", "2"]


# --- the rule was too loose at four places and too strict at two -----------------------------------------------------------


def test_a_number_in_inline_code_is_checked():
    """Old code blanked inline code, so a typed result in backticks reached the paper unflagged."""
    assert _typed("The hit rate was `94.3%` overall.") == ["94.3%"]


def test_a_closed_fence_is_skipped_and_an_unclosed_fence_hides_only_its_paragraph():
    """Old code blanked everything after an unclosed fence, so a later typed result went unflagged."""
    assert _typed("```\nfor i in range(1000): seed = 42\n```") == []
    assert _typed("```\nsetup 5555\n\nLater the gain was 7.7 here.") == ["7.7"]
    assert _typed("```\nfor i in range(1000):\n```\n\nThe gain was 7.7.") == ["7.7"]


def test_a_section_after_the_references_is_checked_but_the_list_is_not():
    """Old code skipped everything after the References heading, appendix included."""
    text = "## References\n\n1. Smith 2019, vol 3456.\n\n## Appendix\n\nThe bias was 0.8244."
    assert _typed(text) == ["0.8244"]
    assert _typed("## References\n\n1. Smith 2019, vol 3456.\n\n### Books\n\n2. Lee 2001, p 77.") == []


def test_only_the_number_before_the_citation_in_its_clause_is_left_to_the_claim_check():
    """Old code exempted every number of a sentence that holds a citation."""
    text = "Prior work gets 80% [3]; we get 94.3%."
    assert _typed(text) == ["94.3%"]
    assert tn.report(text, _plan()).left_to_claim_check == 1


def test_a_citation_before_the_number_in_its_clause_covers_it():
    """A review finding: searching only after the number made "As reported in [W1], the rate was 25%." a typed result."""
    assert _typed("As reported in [W1], the rate was 25%.") == []
    assert _typed("As reported in [W1]; the rate was 35%.") == ["35%"]


def test_a_semicolon_inside_a_citation_is_not_a_clause_boundary():
    """A review finding: the ";" of "[W1; W2]" or "(Smith 2020; Lee 2021)" cut the clause, so the cited number looked uncited."""
    assert _typed("As reported in [W1; W2], the rate was 35%.") == []
    assert tn._clause_breaks("As reported (Smith 2020; Lee 2021), then [3]; and more.", []) == [len("As reported (Smith 2020; Lee 2021), then [3]")]
    assert _typed("As reported in [W1; W2]; the rate was 35%.") == ["35%"]


def test_a_cited_prior_number_before_its_marker_is_still_exempt():
    assert _typed("Prior work found a barrier of 71.3 kJ/mol [3].") == []


def test_a_typed_result_in_plain_prose_is_still_found():
    assert _typed("The mean gain was 12.7 in the main grid.") == ["12.7"]


def test_a_listed_number_is_the_number_not_its_operator_or_neighbour():
    """Old code took the operator and the space before the number with it ("a - 2.5", "7-13")."""
    taken = tn.untraced("Value a - 2.5 and range 7-13 and 12 %.", _plan())
    assert [t for t, _s in taken] == ["2.5", "7", "13", "12 %"]


def test_a_percentile_level_is_a_number_to_trace_not_a_label():
    """-425: "the 90th percentile" counted as a label, so a wrong level was never checked. Now its number is traced like any
    other: a level the plan does not state is found; one the plan states (97.5 here) passes by exact equality, as always."""
    assert _typed("We take the 90th percentile of the errors.") == ["90"]
    assert _typed("The 99.9 quantile of the errors was used.") == ["99.9"]
    plan = tn.plan_numbers({"constants": {"level": 97.5}}, "")
    assert [n.token for n in tn.typed_outside_the_plan("We take the 97.5th percentile.", plan)] == []


# --- numbers of FI's own records, the level of its intervals, rounded plan constants, powers of ten ---------------------------

def _ledger(root, cells=3, trials=40):
    import json

    (root / "raw").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in range(cells):
        rows.append({"event": "planned", "cell": f"c{c}", "trials": trials})
        rows += [{"event": "trial", "cell": f"c{c}", "trial": t, "status": "ok", "seed": 7 + t} for t in range(trials)]
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_a_comma_list_of_values_is_not_read_as_one_grouped_number():
    toks = [n.token for n in tn.numbers("The grid is m in {20,200,2000} and 1,234,567 trials were not run.")]
    assert toks == ["20", "200", "2000", "1,234,567"]


def test_the_trials_and_settings_of_the_ledger_and_the_plans_own_run_size_may_be_stated(tmp_path):
    from core import paper_facts

    _ledger(tmp_path, cells=3, trials=40)
    plan = {**PROTOCOL, "runs_per_setting": 500}
    facts = paper_facts.run_numbers(tmp_path, plan)
    text = "Each of the 3 settings ran 40 trials, though the plan asked for 500 trials."
    assert {Decimal(3), Decimal(40), Decimal(500)} <= facts
    assert tn.typed_outside_the_plan(text, tn.plan_numbers(plan, TOPIC), facts) == []
    # without FI's records the same sentence is flagged, as before
    assert {"40", "500"} <= {n.token for n in tn.typed_outside_the_plan(text, tn.plan_numbers(plan, TOPIC))}
    assert paper_facts.run_numbers(None, plan) == frozenset()


def test_a_number_that_is_only_a_run_fact_in_name_is_still_flagged(tmp_path):
    from core import paper_facts

    _ledger(tmp_path, cells=3, trials=40)
    facts = paper_facts.run_numbers(tmp_path, PROTOCOL)
    assert [n.token for n in tn.typed_outside_the_plan("The bias was 41 percent of the truth.", _plan(), facts)] == ["41 percent"]


def test_a_typed_95_percent_is_traced_like_any_number():
    """-423: any "95%" passed, so "success rate of 95%" passed too. Now a 95 passes only as a number of the plan; the
    level of FI's own intervals is written {{interval_level}}% (FI fills it)."""
    empty = tn.plan_numbers({}, "")
    assert [n.token for n in tn.typed_outside_the_plan("They are normal-approximation 95% intervals.", empty)] == ["95%"]
    assert [n.token for n in tn.typed_outside_the_plan("The success rate was 95% overall.", empty)] == ["95%"]
    assert tn.typed_outside_the_plan("They are {{interval_level}}% intervals.", empty) == []
    # kept exception: a number equal to a number of the plan, here the plan's own 0.95 as a percent, passes
    assert tn.typed_outside_the_plan("The nominal level is 95%.", tn.plan_numbers({"constants": {"p": 0.95}}, "")) == []


def test_a_number_that_only_rounds_to_a_plan_constant_is_traced_but_the_exact_one_passes():
    """-424: a mean typed as 16.118 passed because the plan holds 16.11809565. Now only the plan's digits pass."""
    plan = tn.plan_numbers({"constants": {"lnA": 16.11809565, "R": 0.008314462618}}, "")
    for text, flagged in (("ln A = 16.118", ["16.118"]), ("R = 0.008314", ["0.008314"]), ("ln A = 16.119", ["16.119"]),
                          ("ln A = 16.11809565", []), ("R = 0.008314462618", [])):
        assert [n.token for n in tn.typed_outside_the_plan(text, plan)] == flagged


def test_a_power_of_ten_is_read_as_the_number_it_writes():
    plan = tn.plan_numbers({"constants": {"tol": 1e-12}}, "")
    assert [n.token for n in tn.typed_outside_the_plan("A tolerance of $10^{-12}$ was used.", plan)] == []
    assert [n.token for n in tn.typed_outside_the_plan("An error floor of $10^{-11}$ was assumed.", plan)] == ["-11"]
