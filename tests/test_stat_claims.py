"""A number must be the QUANTITY the paper says it is.

The numeric oracle checks that a number MATCHES a result; the claim check asks
a model whether a sentence is supported. Neither asks what a number IS, and
five delivered papers shipped through the gap: a Bonferroni threshold printed
as a p-value, "Cohen's d > 16 for all pairwise comparisons" with 17 of 36
below it, three-seed t-intervals labelled "exact binomial", a figure's y-axis
limits printed as bin counts, and a major-outbreak probability called the
chance the disease vanishes.

Every check here is a RECOMPUTATION, so each test comes in a fire / don't-fire
pair: the don't-fire half is the one that matters, because a false positive
costs a whole rewrite and FI has already paid for two checks that cried wolf.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import (
    Engine,
    _TEXT_ONLY_HITS,
    _hit_name,
    _hits_need_only_a_rewrite,
    _review_sends_the_experiment_back,
)
from core.stat_claims import check

# 36 comparisons, as the real quest made; alpha = 0.05/36 = 0.0013888...
COMPARISONS = {
    "effect_sizes": [
        {"factor": "by_R0", "metric": "outbreak_probability",
         "a": "0.9", "b": "1.5", "cohens_d": 24.0, "magnitude": "large"},
        {"factor": "by_R0", "metric": "outbreak_probability",
         "a": "1.5", "b": "3.0", "cohens_d": -9.49, "magnitude": "large"},
    ],
    "comparisons": {"n": 36, "bonferroni_alpha": 0.05 / 36, "many": True},
}


def _kinds(report) -> list[str]:
    return [f.kind for f in report.findings]


# --- 1. an effect-size claim the comparisons contradict -----------------------

def test_effect_size_claim_for_all_comparisons_is_checked_against_every_one() -> None:
    """The g4n1 sentence, LaTeX and all. `$...$` must be stripped or the claim
    never parses."""
    report = check(
        r"The effect was significant (Cohen's $d > 16$ for all pairwise "
        r"comparisons, $p < 0.5$ overall).",
        comparison_stats=COMPARISONS,
    )
    assert "effect_size_overstated" in _kinds(report)
    msg = next(f.message for f in report.findings if f.kind == "effect_size_overstated")
    assert "1 of 2" in msg and "9.49" in msg
    # The finding names the stratum, so the rewrite knows which claim to fix.
    assert "1.5 vs 3.0" in msg


def test_a_single_effect_size_is_not_this_checks_business() -> None:
    """Without the universal quantifier the sentence quotes ONE number, which
    is the numeric oracle's job. Firing here would double-report it."""
    report = check(
        r"The largest effect was substantial (Cohen's $d > 16$ at R0 = 0.9).",
        comparison_stats=COMPARISONS,
    )
    assert _kinds(report) == []


def test_an_effect_size_claim_the_run_supports_is_silent() -> None:
    report = check(
        r"Cohen's $d > 5$ for all pairwise comparisons.",
        comparison_stats=COMPARISONS,
    )
    assert _kinds(report) == []


def test_the_papers_own_rounding_is_not_a_contradiction() -> None:
    """A computed 15.94 does not contradict "d > 16": the paper wrote no
    decimals, and 15.94 rounds to 16 at that precision."""
    stats = {
        "effect_sizes": [{"factor": "f", "metric": "m", "a": "x", "b": "y",
                          "cohens_d": 15.94, "magnitude": "large"}],
        "comparisons": {"n": 2, "bonferroni_alpha": 0.025, "many": False},
    }
    report = check("Cohen's d > 16 for every pairwise comparison.",
                   comparison_stats=stats)
    assert _kinds(report) == []


# --- 2. a correction threshold printed as a p-value ---------------------------

def test_bonferroni_threshold_reported_as_a_pvalue() -> None:
    """0.05/36 = 0.0013888...; the paper wrote 0.001388, a TRUNCATION.
    round(0.0013888, 6) is 0.001389, so a rounding comparison misses this
    entirely — the match has to be one unit in the paper's last decimal."""
    assert round(0.05 / 36, 6) != 0.001388, "the truncation this test exists for"
    report = check(
        r"The difference was significant ($p < 0.001388$ via Bonferroni "
        r"correction).",
        comparison_stats=COMPARISONS,
    )
    assert "threshold_as_pvalue" in _kinds(report)
    msg = next(f.message for f in report.findings if f.kind == "threshold_as_pvalue")
    assert "0.05/36" in msg


def test_a_run_that_computed_pvalues_is_left_alone() -> None:
    """If the run produced p-values, a number equal to the threshold could be
    a coincidence rather than a mislabel."""
    report = check(
        r"The difference was significant ($p < 0.001388$).",
        result_json={"by_R0": {"p_value": 0.002}},
        comparison_stats=COMPARISONS,
    )
    assert "threshold_as_pvalue" not in _kinds(report)


def test_a_threshold_the_paper_calls_a_threshold_is_correct() -> None:
    """Printing the corrected threshold is right; calling it an observed
    p-value is the defect."""
    report = check(
        "We used a Bonferroni-corrected significance threshold of 0.001388.",
        comparison_stats=COMPARISONS,
    )
    assert "threshold_as_pvalue" not in _kinds(report)


def test_an_unrelated_pvalue_is_not_flagged() -> None:
    report = check(r"The difference was significant ($p < 0.01$).",
                   comparison_stats=COMPARISONS)
    assert "threshold_as_pvalue" not in _kinds(report)


# --- 3. an interval labelled as a method that did not produce it --------------

S3_INTERVALS = {
    "by_R0.1.5.by_N.100.p_major": {
        "mean": 0.28, "ci_lower": 0.25133, "ci_upper": 0.30867,
    },
}
S3_TABLE = (
    "| R0 | N | p_major (95% exact binomial CI) | theory |\n"
    "|---|---|---|---|\n"
    "| 1.5 | 100 | 0.280 (0.251–0.309) | 0.333 |\n"
)


def test_a_t_interval_labelled_exact_binomial() -> None:
    """The label sits in the table HEADER and the intervals in the cells, so a
    sentence-only search would never see them together."""
    report = check(S3_TABLE, intervals=S3_INTERVALS, n_seeds=3)
    assert "interval_method_mismatch" in _kinds(report)
    msg = next(f.message for f in report.findings
               if f.kind == "interval_method_mismatch")
    assert "exact binomial" in msg and "3 replicate seeds" in msg


def test_an_unlabelled_interval_is_not_anyones_business() -> None:
    """A paper may print the seed-to-seed interval without naming a method;
    that is honest and must stay silent."""
    table = S3_TABLE.replace(" (95% exact binomial CI)", " (95% CI)")
    report = check(table, intervals=S3_INTERVALS, n_seeds=3)
    assert _kinds(report) == []


def test_a_labelled_interval_that_is_not_the_t_interval_is_left_alone() -> None:
    """A real exact-binomial interval is wider than the t-interval. The check
    proves the label wrong only by reproducing the t-interval exactly."""
    report = check(
        S3_TABLE.replace("0.251–0.309", "0.231–0.333"),
        intervals=S3_INTERVALS, n_seeds=3,
    )
    assert _kinds(report) == []


def test_one_finding_per_labelled_table_not_one_per_cell() -> None:
    """Six cells under one wrong header is one mislabel to fix; six bullets
    would bury every other finding in the review."""
    rows = "".join(
        f"| 1.5 | {n} | 0.280 (0.251–0.309) | 0.333 |\n" for n in (100, 1000, 5000)
    )
    report = check(
        "| R0 | N | p_major (95% exact binomial CI) | theory |\n|---|---|---|---|\n" + rows,
        intervals=S3_INTERVALS, n_seeds=3,
    )
    assert _kinds(report).count("interval_method_mismatch") == 1


# --- 4. a figure's axis limit printed as a count ------------------------------

FIG = {
    "fig2.png": {
        "file": "fig2.png",
        "axes": [{"title": "R0=0.9, N=100", "ylabel": "Runs (of 300)",
                  "ylim": [0.0, 231.0], "series": []}],
    },
}


def test_axis_limit_printed_as_a_bin_count() -> None:
    """matplotlib pads 5% above the tallest bar: the bin held 220, the axis
    stops at 231, and the paper printed 231."""
    report = check("Figure 2's modal bin holds 231 of 300 runs.",
                   figure_records=FIG)
    assert "axis_limit_as_count" in _kinds(report)
    assert "y-axis upper limit" in report.findings[0].message


def test_a_count_the_run_actually_computed_is_not_flagged() -> None:
    """The guard that keeps this check quiet: a number appearing anywhere in
    the results is assumed to be that result, not an axis limit."""
    report = check("Figure 2's modal bin holds 231 of 300 runs.",
                   result_json={"modal_bin": 231}, figure_records=FIG)
    assert _kinds(report) == []


def test_a_citation_marker_is_never_a_count() -> None:
    """The only false positive this check produced over 35 graded papers: the
    7 of "[6, 7]" read as a count of runs."""
    fig = {"f.png": {"axes": [{"title": "t", "ylim": [0.0, 6.8586], "series": []}]}}
    report = check(
        "The method samples the next event from current state probabilities "
        "[6, 7], over 300 runs.",
        figure_records=fig,
    )
    assert _kinds(report) == []


def test_a_number_far_from_the_noun_it_counts_is_not_a_count() -> None:
    """"231" is only "a count of runs" if it sits next to "runs"; a sentence
    that mentions runs 160 characters later does not make it one."""
    far = (
        "The stochastic counterpart was implemented using the Gillespie direct "
        "method with 231 as an unrelated parameter, which provides exact "
        "simulations of the master equation by sampling event times across runs."
    )
    report = check(far, figure_records=FIG)
    assert _kinds(report) == []


def test_the_denominator_of_a_count_is_not_itself_a_count() -> None:
    """"300 runs" is what the counts are out of."""
    fig = {"f.png": {"axes": [{"title": "t", "ylim": [0.0, 300.0], "series": []}]}}
    report = check("The modal bin holds 12 of 300 runs.", figure_records=fig)
    assert _kinds(report) == []


# --- 5. a probability described as its own complement -------------------------

G4N2B_RESULTS = {"by_R0": {"1.5": {"100": {"outbreak_probability": 0.3689}}}}


def test_major_outbreak_probability_called_extinction() -> None:
    report = check(
        r"Knowing the average is insufficient if there is a significant "
        r"probability (e.g., $\sim 34\%$ for $R_0=1.5$) that the disease "
        r"will simply vanish.",
        result_json=G4N2B_RESULTS,
    )
    assert "probability_complement" in _kinds(report)
    msg = next(f.message for f in report.findings if f.kind == "probability_complement")
    assert "MAJOR OUTBREAK" in msg and "0.6311" in msg


def test_a_probability_that_vanishes_is_not_a_disease_that_vanishes() -> None:
    """The sentence from the same paper's abstract. "Vanished" is the verb of
    the PROBABILITY here, not the event it describes — a correct sentence that
    a looser pattern would flag."""
    report = check(
        r"For $R_0 < 1$, the outbreak probability vanished as $N$ increased, "
        r"consistent with early extinction.",
        result_json=G4N2B_RESULTS,
    )
    assert "probability_complement" not in _kinds(report)


def test_a_run_that_computed_extinction_is_quoted_correctly() -> None:
    """When the run computed the extinction probability and the paper quotes
    THAT, there is no contradiction to name."""
    report = check(
        r"There is a probability of 0.37 that the disease will die out.",
        result_json={"by_R0": {"1.5": {"outbreak_probability": 0.3689,
                                       "extinction_probability": 0.3689}}},
    )
    assert "probability_complement" not in _kinds(report)


# --- the quiet cases ----------------------------------------------------------

def test_an_empty_paper_is_skipped() -> None:
    report = check("   ")
    assert report.skipped and report.ok


def test_a_paper_with_nothing_to_check_is_silent() -> None:
    """A survey quest runs no experiment: no replicates, no figures, no
    comparisons. The check must stay out of its way entirely."""
    report = check(
        "The literature reports outbreak probabilities between 0.2 and 0.7 "
        "across 300 runs in several studies.",
    )
    assert report.ok and not report.skipped


# --- wiring: the hit must be able to fire, and must take the rewrite route -----

class _Recorder:
    """Only what the engine helper touches."""

    def __init__(self, tmp: Path) -> None:
        self.quest_root = tmp
        self.msgs: list = []
        self._log = SimpleNamespace(
            warning=lambda *a, **k: self.msgs.append(("warn", a)),
            info=lambda *a, **k: self.msgs.append(("info", a)),
        )

    def hits(self, paper: str, state: dict) -> list[str]:
        return Engine._statistics_claim_hits(self, paper, state)  # type: ignore[arg-type]


def _dirty_state() -> dict:
    return {"result_json": {}, "result_json_replicates": [],
            "figure_records": FIG, "design": {}}


def test_the_helper_returns_a_named_hit(tmp_path: Path) -> None:
    hits = _Recorder(tmp_path).hits(
        "Figure 2's modal bin holds 231 of 300 runs.", _dirty_state(),
    )
    assert len(hits) == 1
    assert hits[0].startswith("mislabelled_statistic:")
    assert "231" in hits[0] and "fig2.png" in hits[0]


def test_the_audit_is_written_clean_and_dirty(tmp_path: Path) -> None:
    """A clean run must leave evidence the check ran — silence would be
    indistinguishable from the check being skipped."""
    rec = _Recorder(tmp_path)
    rec.hits("Nothing statistical here at all.", _dirty_state())
    audit = tmp_path / "paper" / "statistics_audit.json"
    assert json.loads(audit.read_text(encoding="utf-8"))["ok"] is True

    rec2 = _Recorder(tmp_path / "b")
    rec2.hits("Figure 2's modal bin holds 231 of 300 runs.", _dirty_state())
    data = json.loads(
        (tmp_path / "b" / "paper" / "statistics_audit.json").read_text(encoding="utf-8")
    )
    assert data["ok"] is False
    assert data["findings"][0]["kind"] == "axis_limit_as_count"


def test_a_bug_in_the_checker_is_never_quest_fatal(tmp_path: Path, monkeypatch) -> None:
    """Blocking a paper over a defect in the checker is strictly worse than
    not checking."""
    from core import stat_claims

    def boom(*_a, **_k):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(stat_claims, "check", boom)
    rec = _Recorder(tmp_path)
    assert rec.hits("Figure 2's modal bin holds 231 of 300 runs.", _dirty_state()) == []
    assert any(kind == "warn" for kind, _ in rec.msgs)


def test_the_hit_is_classified_as_a_text_problem() -> None:
    """The experiment computed the right numbers; the paper described them
    wrongly. Routing this to ``design`` or re-running the experiment would fix
    the wrong thing."""
    hit = "mislabelled_statistic: the paper reports 231 as a count of runs"
    assert _hit_name(hit) in _TEXT_ONLY_HITS
    assert _hits_need_only_a_rewrite([hit])


def test_the_hit_does_not_send_the_experiment_back() -> None:
    """The finding names a ``result_json`` path, and ``_rerun_evidence``
    matches such names against the review's PROSE. Were the finding copied
    into ``weaknesses``/``suggestions``, a mislabelled caption would re-run the
    whole experiment. It lives in ``must_flag_hits`` only, so it does not."""
    review = {
        "verdict": "accept",
        "must_flag_hits": [
            "mislabelled_statistic: the paper gives 34% as the probability the "
            "disease dies out, but that is `by_R0.1.5.100.outbreak_probability`"
        ],
        "weaknesses": [], "suggestions": [], "blocking": "",
    }
    state = {"code": "print(1)",
             "result_json": {"by_R0": {"1.5": {"outbreak_probability": 0.37}}}}
    assert _review_sends_the_experiment_back(review, state) == ""  # type: ignore[arg-type]


def _route_engine() -> SimpleNamespace:
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
            "mislabelled_statistic: 231 is the y-axis upper limit of fig2.png"]},
        "iteration": 0, "code": "print(1)",
    }
    assert Engine._route_after_review(_route_engine(), state) == "rewrite"  # type: ignore[arg-type]


# --- wiring: it reaches must_flag_hits on both review paths --------------------

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
    paper.write_text("Figure 2's modal bin holds 231 of 300 runs.", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": {}, "figure_records": FIG,
    }))
    review = patch["review"]
    assert any(str(h).startswith("mislabelled_statistic") for h in review["must_flag_hits"])
    assert patch["iteration"] == 1, "a forced hit spends an iteration"
    # and it is a TEXT problem, so the experiment is not re-run
    assert _review_sends_the_experiment_back(review, {"code": "x"}) == ""  # type: ignore[arg-type]
    assert eng._route_after_review(  # type: ignore[arg-type]
        {"review": review, "iteration": 1, "code": "x"}
    ) == "rewrite"


def test_a_clean_paper_adds_no_hit(tmp_path: Path) -> None:
    import asyncio

    eng = _review_engine(tmp_path)

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"verdict": "accept", "score": 5,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("The epidemic was simulated in a closed population.",
                     encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "result_json": {}, "figure_records": FIG,
    }))
    assert patch["review"]["must_flag_hits"] == []
    assert "iteration" not in patch
