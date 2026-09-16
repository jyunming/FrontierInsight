"""A must-flag about something the run computed sends the experiment back to be
written and run again, instead of rewriting the paper around a wrong value.

The case this is built from is real: a quest's review asked to "ensure the
solver is correctly integrated so the 'deterministic_final_size' is not
reported as 0.0", the router answered "all about the text", only ``write``
re-ran, and the paper shipped claiming convergence to a limit whose value was
0.0 at every R0.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    PausesConfig, ProviderConfig,
)
from core.engine import (
    _CODE_REEXECUTES, Engine, _format_review_for_writer, _rerun_evidence,
    _review_sends_the_experiment_back,
)

TOPIC = "Extinction times in a stochastic SIR model"

# The shape of the run's results: the deterministic final size came out 0.0 at
# every R0, and the paper claimed convergence to it anyway.
RESULTS: dict[str, Any] = {
    "by_r0_0.9": {"by_n_100": {
        "outbreak_probability": 0.2166,
        "deterministic_final_size": 0.0,
        "mean_final_size_major": 0.1638,
    }},
    "by_r0_3.0": {"by_n_5000": {
        "outbreak_probability": 0.659,
        "deterministic_final_size": 0.0,
        "mean_final_size_major": 0.9404,
    }},
}
FIGURES = {"stochastic_vs_deterministic_convergence.png": {"series": []}}

# The review as it was actually written: the must-flag is the bare identifier
# the review schema asks for, and what it is about is in the prose.
XG3_REVIEW: dict[str, Any] = {
    "verdict": "revise",
    "score": 3,
    "weaknesses": [
        "The paper acknowledges in the Limitations that the deterministic final "
        "size was not explicitly tabulated in the data, yet claims convergence "
        "to it without providing the actual deterministic target value.",
    ],
    "suggestions": [
        "Reconcile the deterministic results: ensure the solver is correctly "
        "integrated so the 'deterministic_final_size' is not reported as 0.0, "
        "allowing the claim about unique final sizes to be evidence-backed.",
    ],
    "blocking": "The paper contains an unsupported claim regarding the "
                "deterministic model's prediction due to a failure in reporting.",
    "must_flag_hits": ["unsupported_claim"],
}

# The same must-flag identifier, about the text: nothing here names a value.
CITATION_REVIEW: dict[str, Any] = {
    "verdict": "revise",
    "weaknesses": ["Includes an unsupported claim regarding demographic "
                   "stochasticity cited to [2] that is not in the source."],
    "suggestions": ["Remove the specific attribution to [2] for the definition "
                    "of demographic stochasticity, or provide a valid source."],
    "blocking": "The paper contains an unsupported claim.",
    "must_flag_hits": ["unsupported_claim"],
}

CAPTION_REVIEW: dict[str, Any] = {
    "verdict": "revise",
    "suggestions": ["Update the caption for Figure 2 to state that it shows a "
                    "single representative run rather than the distribution."],
    "blocking": "A caption describes what its figure does not show.",
    "must_flag_hits": ["figure_caption: Figure 2 names S(t)"],
}


def _engine(tmp_path: Path, *, max_iterations: int = 2) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=max_iterations, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        pauses=PausesConfig(review="off"),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    return eng


def _state(review: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A quest that ran an experiment, reviewed its first draft."""
    return {
        "topic": TOPIC, "review": review, "iteration": 1,
        "code": "print('RESULT_JSON: {}')",
        "result_json": RESULTS, "figure_records": FIGURES,
        **extra,
    }


# --- the route -----------------------------------------------------------------

def test_a_flag_about_a_computed_value_runs_the_experiment_again(tmp_path: Path) -> None:
    state = _state(XG3_REVIEW)
    assert _rerun_evidence(XG3_REVIEW, state) == "deterministic_final_size"  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(state) == "re_execute"  # type: ignore[arg-type]


def test_a_flag_about_the_experiment_file_runs_the_experiment_again(tmp_path: Path) -> None:
    review = {**CITATION_REVIEW,
              "suggestions": ["Fix the bracketing in experiment.py so the root "
                              "solver does not return the trivial root."]}
    assert _rerun_evidence(review, _state(review)) == "experiment.py"  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "re_execute"  # type: ignore[arg-type]


def test_a_flag_about_a_figures_underlying_data_runs_the_experiment_again(tmp_path: Path) -> None:
    review = {**CAPTION_REVIEW,
              "suggestions": ["Regenerate stochastic_vs_deterministic_convergence.png "
                              "from all 300 trials instead of one seed."]}
    named = _rerun_evidence(review, _state(review))  # type: ignore[arg-type]
    assert named == "stochastic_vs_deterministic_convergence.png"
    assert _engine(tmp_path)._route_after_review(_state(review)) == "re_execute"  # type: ignore[arg-type]


@pytest.mark.parametrize("review", [CITATION_REVIEW, CAPTION_REVIEW])
def test_a_citation_or_caption_flag_still_rewrites_the_paper(
    tmp_path: Path, review: dict[str, Any],
) -> None:
    assert _rerun_evidence(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "rewrite"  # type: ignore[arg-type]


def test_a_leaf_key_that_is_also_an_english_word_does_not_run_the_experiment_again(
    tmp_path: Path,
) -> None:
    """``mean`` / ``n`` / ``ci_upper`` are results keys AND ordinary words; only a
    compound name is evidence that the review meant the value."""
    state = _state(CITATION_REVIEW, result_json={"mean": 0.5, "n": 300, "ci_upper": 0.9},
                   figure_records={})
    review = {**CITATION_REVIEW,
              "suggestions": ["Say what the mean is over, and how many runs n is."]}
    state["review"] = review
    assert _rerun_evidence(review, state) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_an_advisory_numeric_finding_never_forces_a_re_run(tmp_path: Path) -> None:
    """Every numeric-oracle finding quotes a result path. They are advisory
    because a pattern match over prose misreads a DOI, so the scan must not
    read them — otherwise every flagged number would re-run the experiment."""
    review = {**CITATION_REVIEW, "numeric_oracle_warnings": [
        "near_miss: paper says 1 but `by_r0_3.0.by_n_5000."
        "mean_final_size_major` = 0.9404 (off by 5.7%)",
    ]}
    assert _rerun_evidence(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "rewrite"  # type: ignore[arg-type]


def test_a_quest_with_no_experiment_is_never_sent_back_to_implement(tmp_path: Path) -> None:
    """A survey / analyze-only quest has no ``code``: there is nothing to run
    again, whatever the review names."""
    state = _state(XG3_REVIEW)
    del state["code"]
    assert _rerun_evidence(XG3_REVIEW, state) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_a_review_with_no_must_flag_is_untouched_by_the_new_route(tmp_path: Path) -> None:
    review = {**XG3_REVIEW, "verdict": "accept", "must_flag_hits": []}
    assert _engine(tmp_path)._route_after_review(_state(review)) == "done"  # type: ignore[arg-type]


# --- what must NOT be read as a problem with the experiment ----------------------

# The engine writes these two itself, from the figure records, whatever the
# reviewer says — and both quote the figure they checked. Both are text
# problems: the figure is right, the paper's words are not.
ENGINE_CAPTION_HIT = (
    'figure_caption: the caption of figures/stochastic_vs_deterministic_convergence.png '
    'names "deterministic", but the figure draws it flat'
)
ENGINE_MISSING_HIT = (
    "figure_missing: the paper leaves out figures/"
    "stochastic_vs_deterministic_convergence.png, which the design planned and the run drew"
)


@pytest.mark.parametrize("hit", [ENGINE_CAPTION_HIT, ENGINE_MISSING_HIT])
def test_a_hit_the_engine_wrote_about_a_figure_still_rewrites_the_paper(
    tmp_path: Path, hit: str,
) -> None:
    """Scanning the hits themselves would send every one of these to a re-run
    that cannot help: the figure's data is fine, the draft is not."""
    review = {"verdict": "revise", "must_flag_hits": [hit],
              "blocking": "The paper does not show what it discusses."}
    assert _review_sends_the_experiment_back(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "rewrite"  # type: ignore[arg-type]


def test_a_flaw_in_the_design_still_goes_back_to_design(tmp_path: Path) -> None:
    """A hit that already re-opens the design re-runs the experiment on its way
    through implement anyway, and a flawed design is not fixed by writing the
    same experiment again — so re-execution replaces the rewrite, nothing else."""
    review = {**XG3_REVIEW, "must_flag_hits": ["[methodologist] circular_evaluation"]}
    assert _review_sends_the_experiment_back(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "revise"  # type: ignore[arg-type]


def test_a_draft_over_the_page_limit_does_not_spend_the_re_execute(tmp_path: Path) -> None:
    """The shortening rewrites have their own route and their own counter."""
    review = {**XG3_REVIEW,
              "must_flag_hits": ["over_page_limit: 5 pages against a limit of 4"]}
    assert _review_sends_the_experiment_back(review, _state(review)) == ""  # type: ignore[arg-type]
    state = _state(review, page_limit_rewrites=1)
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


# --- the cost cap ---------------------------------------------------------------

@pytest.mark.parametrize(("done", "route"), [
    (0, "re_execute"), (1, "re_execute"), (2, "rewrite"), (3, "rewrite"),
])
def test_one_re_execute_a_quest_then_the_flag_takes_the_text_route(
    tmp_path: Path, done: int, route: str,
) -> None:
    """The review node counts the re-execute before the router reads it, so the
    first flagged review sees 1. A flag that survives the re-run falls back to
    the text route rather than spending the rest of the budget on re-runs."""
    assert _CODE_REEXECUTES == 1
    state = _state(XG3_REVIEW, code_reexecutes=done)
    assert _engine(tmp_path)._route_after_review(state) == route  # type: ignore[arg-type]


def test_the_review_node_counts_the_re_execute(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (tmp_path / "paper" / "paper.md").write_text("# T\n\nThe limit is 0.0.\n", encoding="utf-8")

    async def chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        import json as _json
        return _json.dumps(XG3_REVIEW)

    eng._chat = chat  # type: ignore[method-assign]
    patch = asyncio.run(eng._node_review(_state(  # type: ignore[arg-type]
        XG3_REVIEW, iteration=0, paper_md=str(tmp_path / "paper" / "paper.md"),
    )))
    assert patch["code_reexecutes"] == 1
    assert patch["iteration"] == 1


def test_the_review_node_counts_nothing_when_the_flags_are_about_the_text(
    tmp_path: Path,
) -> None:
    eng = _engine(tmp_path)
    (tmp_path / "paper" / "paper.md").write_text("# T\n\nA claim.\n", encoding="utf-8")

    async def chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        import json as _json
        return _json.dumps(CITATION_REVIEW)

    eng._chat = chat  # type: ignore[method-assign]
    patch = asyncio.run(eng._node_review(_state(  # type: ignore[arg-type]
        CITATION_REVIEW, iteration=0, paper_md=str(tmp_path / "paper" / "paper.md"),
    )))
    assert "code_reexecutes" not in patch


# --- a mixed set ----------------------------------------------------------------

def test_a_mixed_set_runs_the_experiment_again_and_keeps_the_text_flag(
    tmp_path: Path,
) -> None:
    """One code flag and one text flag: the experiment runs again (the text flag
    cannot be fixed around a wrong value either), and the re-run ends at
    ``write``, which is given every must-flag — so the text one is not dropped."""
    review = {
        **XG3_REVIEW,
        "must_flag_hits": ["unsupported_claim", "figure_caption: Figure 2 names S(t)"],
        "suggestions": [*XG3_REVIEW["suggestions"],
                        "Update the caption for Figure 2 to say it is one seed."],
    }
    assert _engine(tmp_path)._route_after_review(_state(review)) == "re_execute"  # type: ignore[arg-type]
    for_writer = _format_review_for_writer({"review": review})  # type: ignore[arg-type]
    assert "unsupported_claim" in for_writer
    assert "figure_caption: Figure 2 names S(t)" in for_writer
    assert "Update the caption for Figure 2" in for_writer


# --- what implement is told -----------------------------------------------------

def test_the_implement_prompt_carries_what_the_review_named(tmp_path: Path) -> None:
    """``implement``'s own prompt has the design and no review, so a re-execute
    would regenerate the same code without this."""
    eng = _engine(tmp_path)
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:
        seen[node] = prompt
        return "```python\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_implement(_state(XG3_REVIEW)))  # type: ignore[arg-type]
    prompt = seen["implement"]
    assert "## The review sent this experiment back" in prompt
    assert "deterministic_final_size" in prompt
    assert "ensure the solver is correctly integrated" in prompt


def test_a_first_implement_is_told_nothing_extra(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:
        seen[node] = prompt
        return "```python\nprint('RESULT_JSON: {}')\n```"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_implement({"topic": TOPIC, "design": {}}))  # type: ignore[arg-type]
    assert "## The review sent this experiment back" not in seen["implement"]


# --- the graph ------------------------------------------------------------------

def test_the_graph_runs_the_experiment_again_and_comes_back_to_review(
    tmp_path: Path,
) -> None:
    eng = _engine(tmp_path)
    visits: list[str] = []
    reviews = iter([
        {"review": XG3_REVIEW, "iteration": 1, "code_reexecutes": 1},
        {"review": {"verdict": "accept", "must_flag_hits": []}},
    ])

    def node(name: str, patch: Any = None) -> Any:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            visits.append(name)
            return patch() if patch else {}
        return run

    for name in ("design", "implement_outline", "implement", "execute",
                 "execute_reflect", "analyze", "cross_check", "evidence_gate",
                 "write", "claim_check"):
        setattr(eng, f"_node_{name}", node(name))
    eng._node_review = node("review", lambda: next(reviews))  # type: ignore[method-assign]
    eng._route_after_execute_reflect = lambda state: "proceed"  # type: ignore[method-assign]
    eng._route_after_cross_check = lambda state: "write"  # type: ignore[method-assign]
    eng._route_after_evidence_gate = lambda state: "write"  # type: ignore[method-assign]
    graph = eng._build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "re-execute"}}
    graph.update_state(config, {
        "topic": TOPIC, "iteration": 1,
        "code": "print('RESULT_JSON: {}')",
        "result_json": RESULTS, "figure_records": FIGURES,
    }, as_node="claim_check")
    asyncio.run(graph.ainvoke(None, config))
    assert visits[0] == "review"
    assert "implement" in visits and "execute" in visits, visits
    # The experiment ran again BEFORE the paper was written again.
    assert visits.index("execute") < visits.index("write")
    # And the run's own design was not re-opened over a code bug.
    assert "design" not in visits
    assert visits[-1] == "review"
