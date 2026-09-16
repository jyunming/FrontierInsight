"""Every number the paper prints must trace to something the run computed.

The numeric oracle looks for a number sitting NEAR a result without matching
it, and deliberately ignores one that is far from every result. The statistics
claim check asks what a number IS. Neither asks where a number came from, and
a graded paper shipped through the gap: its Table 1 printed eight cells that
match no value the run computed, no replicate, and none of the three-seed
aggregates.

The decisive fact these tests are built around is that a paper does NOT print
the contents of RESULT_JSON — it prints the MEAN OVER THE SEEDS and that
mean's interval, neither of which any single seed recorded. Measured over that
paper, checking against RESULT_JSON alone makes all eleven of its CORRECT
headline numbers untraceable too. So the seed aggregates are part of what a
number may trace to, and every guard below exists to keep the check silent on
prose that is fine.

Each guard comes as a fire / don't-fire pair. The don't-fire half is the one
that matters: a false positive costs a whole rewrite, and FI has already paid
for checks that cried wolf.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.engine import (
    Engine,
    _TEXT_ONLY_HITS,
    _hit_name,
    _hits_need_only_a_rewrite,
    _review_sends_the_experiment_back,
)
from core.number_provenance import check

# A small stand-in for a real run: seed 0's RESULT_JSON.
RESULTS = {
    "deterministic_final_size": 0.5828,
    "by_N": {
        "100": {"outbreak_probability": 0.3848, "conditioned_mean": 0.5600},
        "1000": {"outbreak_probability": 0.3300, "conditioned_mean": 0.5538},
    },
    "final_sizes": [0.11, 0.22, 0.33],
}

# What the engine computes ACROSS the seeds and hands to the paper. 0.3337 and
# its bounds appear nowhere in RESULTS: this is the aggregate a paper reports.
INTERVALS = {
    "by_N.1000.outbreak_probability": {
        "mean": 0.3337, "ci_lower": 0.3126, "ci_upper": 0.3548,
    },
}
AGGREGATE = {
    "by_N.1000.outbreak_probability": {
        "mean": 0.3337, "std": 0.0075, "n": 3, "min": 0.3260, "max": 0.3412,
    },
}


def _tokens(report) -> list[str]:
    return [f.token for f in report.findings]


def _check(text: str, **kw):
    kw.setdefault("result_json", RESULTS)
    kw.setdefault("intervals", INTERVALS)
    kw.setdefault("aggregate", AGGREGATE)
    return check(text, **kw)


# --- the defect this check exists for -----------------------------------------

def test_a_number_the_run_never_computed_is_flagged() -> None:
    """The observed Table 1 cell. 0.7093 matches no result, no replicate and
    no aggregate; the run's own value for that cell was 0.5600."""
    report = _check("| N = 100 | 0.2974 (0.2721 to 0.3227) | 0.7093 (0.6934 to 0.7252) |")
    assert "0.7093" in _tokens(report)
    msg = report.findings[0].message
    assert "nothing in this run accounts for it" in msg


def test_the_finding_names_the_number_and_not_a_result_path() -> None:
    """There IS no result path — that is the point of the check. The message
    must therefore not invent one."""
    report = _check("The conditioned mean was 0.7093 in that cell.")
    assert report.findings[0].value == pytest.approx(0.7093)
    assert report.findings[0].kind == "untraceable_number"


# --- a value the run DID compute ----------------------------------------------

def test_a_value_in_the_results_is_not_flagged() -> None:
    report = _check("The outbreak probability was 0.3848 at N = 100.")
    assert _tokens(report) == []


def test_rounding_to_the_papers_own_precision_is_accepted() -> None:
    """A computed 0.5828 printed as 0.583 is the same number."""
    for written in ("0.583", "0.5828"):
        report = _check(f"The deterministic final size is {written}.")
        assert _tokens(report) == [], written


def test_a_fraction_printed_as_a_percentage_is_accepted() -> None:
    """0.3848 in the results, "38.48%" in the prose."""
    report = _check("The outbreak probability was 38.48% at N = 100.")
    assert _tokens(report) == []


def test_a_complement_is_accepted() -> None:
    """A paper may print 1 - p for a probability p the run computed."""
    report = _check("The epidemic died out in 0.6152 of runs.")
    assert _tokens(report) == []


def test_truncation_is_accepted_as_well_as_rounding() -> None:
    """0.5828... written 0.582 is a truncation, not a number from nowhere.
    A rounding comparison alone would call it untraceable."""
    report = _check("The deterministic final size is 0.582.")
    assert _tokens(report) == []


# --- the aggregate over the seeds: without this the check flags correct prose --

def test_the_mean_over_the_seeds_is_what_a_paper_reports() -> None:
    """0.3337 is in no replicate and no RESULT_JSON — only in the aggregate.
    This is the single fact that makes the check usable rather than a machine
    for flagging correct papers."""
    report = _check("The probability stabilised at 0.3337 as N grew.")
    assert _tokens(report) == []


def test_a_confidence_bound_is_accepted() -> None:
    report = _check("The probability was 0.3337 (95% CI 0.3126 to 0.3548).")
    assert _tokens(report) == []


def test_a_standard_error_is_accepted() -> None:
    """A paper prints "SE 0.0043" beside a mean; that is the SD over root n,
    which the aggregate carries only as a standard deviation."""
    se = 0.0075 / (3 ** 0.5)
    report = _check(f"The probability was 0.3337 (SE {se:.4f}, n = 3).")
    assert _tokens(report) == []


def test_without_the_seed_aggregates_correct_prose_would_be_flagged() -> None:
    """The measurement that justifies including them: the SAME sentence, with
    the aggregates withheld, is flagged. If this ever stops being true the
    aggregates have stopped mattering and the guard can be reconsidered."""
    report = check(
        "The probability stabilised at 0.3337 as N grew.", result_json=RESULTS,
    )
    assert "0.3337" in _tokens(report)


# --- the dominant false positive: a range is not a negative number ------------

def test_a_range_written_with_a_dash_is_not_a_negative_number() -> None:
    """"95% CI 0.3126--0.3548" tokenizes as -0.3548, a value no run computes.
    Measured over the graded papers this one artifact produced the majority of
    all flags, in six runs whose numbers were fine."""
    for dash in ("--", "-", "–", "—"):
        report = _check(f"The probability was 0.3337 (95% CI 0.3126{dash}0.3548).")
        assert _tokens(report) == [], dash


def test_a_genuine_negative_number_still_reads_as_negative() -> None:
    """The range guard keys on a DIGIT before the dash, so a negative value
    introduced by an operator keeps its sign and is still checked."""
    report = _check("The effect size was d = -240.55 for that contrast.")
    assert "-240.55" in _tokens(report)


# --- numbers that are not measurements ----------------------------------------

def test_a_year_is_not_a_measurement() -> None:
    report = _check("The model follows Kermack and McKendrick 1927 throughout.")
    assert _tokens(report) == []


def test_a_version_number_is_not_a_measurement() -> None:
    """"numpy 1.11.4" would otherwise be read as the 3-significant-digit
    measurement 1.11."""
    report = _check("Simulations used numpy 1.11.4 and scipy 1.14.1.")
    assert _tokens(report) == []


def test_a_random_seed_is_not_a_measurement() -> None:
    """A seed is chosen in the code and computed by nothing, so it appears in
    no result. This was the only number flagged in a paper whose numbers the
    grader passed."""
    report = _check("Child streams came from SeedSequence(20260916) per cell.")
    assert _tokens(report) == []


def test_a_citation_marker_is_not_a_measurement() -> None:
    report = _check("This follows from the branching approximation [6, 7].")
    assert _tokens(report) == []


def test_a_figure_or_section_reference_is_not_a_measurement() -> None:
    report = _check("Figure 3.14 and Section 2.11 present the comparison.")
    assert _tokens(report) == []


def test_a_low_precision_number_is_never_asked_to_trace() -> None:
    """The load-bearing guard. "300 runs", "3 seeds", "R0 = 1.5" and "95% CI"
    are rhetorical, conventional or stratum labels; requiring them to trace
    anywhere is how a check starts flagging ordinary sentences."""
    report = _check(
        "We ran 300 realizations across 3 seeds for R0 = 1.5 and N = 5000, "
        "reporting a 95% CI at the 0.05 level over 2 regimes."
    )
    assert _tokens(report) == []


def test_a_low_precision_number_written_to_four_decimals_is_still_checked() -> None:
    """0.0068 carries two significant digits but four decimals: the run either
    computed it or it did not. The real defect included four such bounds."""
    report = _check("The minor-extinction size was 0.0076 (0.0068 to 0.0084).")
    assert "0.0068" in _tokens(report)


# --- derivations --------------------------------------------------------------

def test_arithmetic_the_paper_writes_out_is_accepted() -> None:
    """A paper showing its working prints intermediate values that exist
    nowhere in the results and should not: the derivation is on the page."""
    report = _check(
        "This gives 0.3848 * 0.5600 = 0.2155 for the pooled contribution."
    )
    assert _tokens(report) == []


def test_a_table_row_difference_between_two_results_is_accepted() -> None:
    """A table whose last column is a difference of two columns beside it.
    There is no operator, so both operands must themselves be real results."""
    report = _check("| 1.5 | 100 | 0.5828 | 0.5600 | 0.0228 |")
    assert _tokens(report) == []


def test_two_numbers_that_are_not_results_do_not_explain_a_third() -> None:
    """The strict half of the derivation rule. Without an operator on the page
    the operands must be values the run computed — otherwise any three numbers
    standing near each other would explain away a fourth and the check would
    go blind."""
    report = _check("| 0.7093 | 0.1111 | 0.8204 |")
    assert "0.8204" in _tokens(report)


# --- provenance outside the results -------------------------------------------

def test_a_number_the_user_set_in_the_configuration_is_accepted() -> None:
    """A population size or replicate count comes from the run config, not
    from anything the experiment computed.

    This test said 2700 first, and that made it **vacuous**: 2700 carries two
    significant digits, so the precision floor dropped it before ``config``
    was ever consulted and the assertion held with no config at all. 1250
    survives the floor, so the configuration really is what clears it — which
    the first assertion below now proves.
    """
    text = "Each cell used 1250 independent realizations."
    assert "1250" in _tokens(_check(text)), "without the config this must fire"
    report = _check(text, config={"topic": "Run 1250 realizations per setting."})
    assert _tokens(report) == []


def test_a_reference_value_the_design_declared_is_accepted() -> None:
    """One graded paper prints "as design-time reference values, tau_det =
    0.5828" for a constant the design named and the run never re-derived."""
    text = "As a design-time reference value, tau_det = 0.9405 for R0 = 3.0."
    assert "0.9405" in _tokens(_check(text)), "without the design this must fire"
    report = _check(
        text, design={"method": "Compare against tau_det = 0.9405 at R0 = 3.0."},
    )
    assert _tokens(report) == []


def test_a_figures_own_numbers_are_accepted() -> None:
    """An axis limit printed as a count IS a defect, but stat_claims recomputes
    and names it. Flagging it here too would report one mistake twice."""
    report = _check(
        "The modal bin holds 231 of the runs.",
        figure_records={"fig2.png": {"axes": [{"ylim": [0, 231.0]}]}},
    )
    assert _tokens(report) == []


def test_an_effect_size_from_the_comparison_aggregate_is_accepted() -> None:
    """Cohen's d is computed FOR the paper and is absent from RESULT_JSON, so
    it has nowhere else to trace to."""
    report = _check(
        "The contrast gave Cohen's d = -240.55 at N = 5000.",
        comparison_stats={"effect_sizes": [{"cohens_d": -240.55}]},
    )
    assert _tokens(report) == []


# --- a run with nothing to check against --------------------------------------

def test_a_run_that_recorded_no_results_goes_quiet() -> None:
    """A crashed run whose last RESULT_JSON is empty must not have every
    number in its paper flagged. Several existing engine tests drive the review
    node with an empty result_json and assert must_flag_hits == []."""
    report = check("The conditioned mean was 0.7093 in that cell.", result_json={})
    assert report.skipped and not report.findings
    assert report.skip_reason == "the run recorded no numeric results"


def test_an_empty_array_is_not_a_recorded_result() -> None:
    """The zero length of an empty array would otherwise count as "the run
    computed something" and re-open the whole paper to flagging."""
    report = check(
        "The conditioned mean was 0.7093.", result_json={"final_sizes": []},
        replicates=[],
    )
    assert report.skipped


def test_an_empty_paper_is_skipped() -> None:
    report = _check("   ")
    assert report.skipped and report.skip_reason == "paper text is empty"


def test_the_audit_counts_what_it_checked() -> None:
    report = _check("The conditioned mean was 0.7093 in that cell.")
    body = report.to_dict()
    assert body["paper_numbers_checked"] >= 1
    assert body["traceable_values"] > 0
    assert body["untraceable_numbers"] == 1
    assert body["findings"][0]["token"] == "0.7093"


def test_one_finding_per_distinct_value_and_the_list_is_capped() -> None:
    """A table of wrong cells must not bury every other review finding."""
    text = " ".join(f"value {0.7001 + i / 10000:.4f} here." for i in range(30))
    report = _check(text)
    assert len(report.findings) <= 8
    assert report.untraceable > len(report.findings)
    # ...but the writer must still be told how many more there are, or it
    # fixes the eight it can see and the next round flags the rest, which
    # spends the iteration budget a round at a time.
    assert "A further" in report.findings[-1].message


# --- wiring: the hit routes to a rewrite, never to a re-run -------------------

def test_the_hit_name_is_text_only() -> None:
    hit = "unsourced_number: the paper prints 0.7093, and nothing accounts for it"
    assert _hit_name(hit) == "unsourced_number"
    assert _hit_name(hit) in _TEXT_ONLY_HITS
    assert _hits_need_only_a_rewrite([hit])


def test_the_hit_does_not_send_the_experiment_back() -> None:
    """The experiment ran and recorded its results; the paper quotes a figure
    none of them holds. That is a problem with the prose, so the route is
    ``write`` and the experiment is left alone."""
    review = {
        "verdict": "accept",
        "must_flag_hits": [
            "unsourced_number: the paper prints 0.7093, and nothing in this "
            "run accounts for it"
        ],
        "weaknesses": [], "suggestions": [], "blocking": "",
    }
    state = {"code": "print(1)", "result_json": RESULTS}
    assert _review_sends_the_experiment_back(review, state) == ""  # type: ignore[arg-type]


def _route_engine():
    from types import SimpleNamespace

    return SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=4, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )


def test_the_route_is_rewrite_not_revise_or_re_execute() -> None:
    state = {
        "review": {"verdict": "accept", "must_flag_hits": [
            "unsourced_number: the paper prints 0.7093"]},
        "iteration": 0, "code": "print(1)",
    }
    assert Engine._route_after_review(_route_engine(), state) == "rewrite"  # type: ignore[arg-type]


def _review_engine(tmp_path: Path) -> Engine:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


@pytest.mark.parametrize("panel", [False, True])
def test_review_forces_the_hit_even_when_the_reviewer_accepts(
    tmp_path: Path, panel: bool,
) -> None:
    """Panel mode once skipped the numeric oracle entirely, so asking for more
    reviewers meant fewer checks. Both paths run this one."""
    import asyncio

    eng = _review_engine(tmp_path)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "both accept"})
        return json.dumps({"verdict": "accept", "score": 5,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text(
        "The conditioned mean final size was 0.7093 at N = 100.",
        encoding="utf-8",
    )
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": RESULTS, "figure_records": {},
    }))
    review = patch["review"]
    assert any(str(h).startswith("unsourced_number") for h in review["must_flag_hits"])
    assert patch["iteration"] == 1, "a forced hit spends an iteration"
    assert _review_sends_the_experiment_back(review, {"code": "x"}) == ""  # type: ignore[arg-type]
    assert eng._route_after_review(  # type: ignore[arg-type]
        {"review": review, "iteration": 1, "code": "x"}
    ) == "rewrite"


def test_the_audit_is_written_even_when_the_paper_is_clean(tmp_path: Path) -> None:
    """A clean run must leave evidence the check RAN, so silence never has to
    be read as "it was skipped"."""
    import asyncio

    eng = _review_engine(tmp_path)

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"verdict": "accept", "score": 5,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("The outbreak probability was 0.3848 at N = 100.",
                     encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": RESULTS, "figure_records": {},
    }))
    assert patch["review"]["must_flag_hits"] == []
    audit = tmp_path / "paper" / "provenance_audit.json"
    assert audit.is_file()
    assert json.loads(audit.read_text(encoding="utf-8"))["ok"] is True
