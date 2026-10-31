"""A review's prose never sends the experiment back to be written and run again.

It used to: a must-flag review whose free text merely contained the name of a result key, a figure file or
``experiment.py`` went to ``implement``, so a caption or a source attribution that mentioned a seed, a check or a figure
file ran the whole study again (eight real quests; the last lost three working figures). The name match was a guess
from free text. Every check that measures a computed number as wrong runs before the review (the results check, the
known-answer checks, the run manifest), so a review adds no measurement and goes to ``write``.

The plumbing of the route (``re_execute``, its cap, the count the review node keeps, the prompt ``implement`` is
given) stays, for a measured finding that may be returned by ``_review_sends_the_experiment_back`` later; those tests
make the function return a name.
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
import core.engine as engine_mod
from core.engine import (
    _CODE_REEXECUTES, Engine, _format_review_for_writer, _review_sends_the_experiment_back,
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

@pytest.fixture
def measured_fault(monkeypatch: pytest.MonkeyPatch) -> str:
    """A finding that a check of FI's measured returns as the reason to run the experiment again."""
    monkeypatch.setattr(engine_mod, "_review_sends_the_experiment_back", lambda review, state: "deterministic_final_size")
    return "deterministic_final_size"


def _names(kind: str) -> str:
    return {"key": "deterministic_final_size", "figure": "stochastic_vs_deterministic_convergence.png",
            "file": "experiment.py", "seed": "base_seed"}[kind]


@pytest.mark.parametrize("kind", ["key", "figure", "file", "seed"])
def test_a_review_that_only_names_a_result_key_a_figure_or_the_script_rewrites_the_paper(
    tmp_path: Path, kind: str,
) -> None:
    """The name of a result key, a figure file or experiment.py in the review's prose is no measurement: the paper is
    written again, the study is not run again (the first review of XG3_REVIEW named a key, and so on)."""
    review = {**CITATION_REVIEW,
              "suggestions": [f"Say in the text what {_names(kind)} is, and state the seed the paper uses."]}
    state = _state(review, result_json={**RESULTS, "base_seed": 7})
    assert _review_sends_the_experiment_back(review, state) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_the_review_that_once_sent_the_experiment_back_over_a_key_now_rewrites(tmp_path: Path) -> None:
    """XG3_REVIEW is the real input the old rule was built from; the fault it names (a quantity that is 0.0 at every
    setting) is found by FI's own results check before the review (see the next test)."""
    state = _state(XG3_REVIEW)
    assert _review_sends_the_experiment_back(XG3_REVIEW, state) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


def test_the_measured_check_still_finds_a_quantity_that_is_the_same_at_every_setting() -> None:
    """The known positive: the fault in XG3 (``deterministic_final_size`` 0.0 at every R0) is found by a measurement
    FI makes of the run before the review, not by reading the review's words."""
    from core import degenerate_results as dr

    cells = [{"cell": {"r0": r, "n": n},
              "metrics": {"outbreak_probability": [p], "deterministic_final_size": [0.0], "mean_final_size_major": [m]}}
             for r, n, p, m in [(0.9, 100, 0.21, 0.16), (2.0, 100, 0.5, 0.7), (3.0, 5000, 0.66, 0.94), (1.5, 500, 0.4, 0.5)]]
    found = dr.quantity_problems(cells, {"outbreak_probability", "deterministic_final_size", "mean_final_size_major"})
    assert [(q.name, q.kind) for q in found] == [("deterministic_final_size", "constant")]


def test_a_measured_finding_sends_the_experiment_back(tmp_path: Path, measured_fault: str) -> None:
    state = _state(XG3_REVIEW)
    assert _engine(tmp_path)._route_after_review(state) == "re_execute"  # type: ignore[arg-type]


@pytest.mark.parametrize("review", [CITATION_REVIEW, CAPTION_REVIEW])
def test_a_citation_or_caption_flag_still_rewrites_the_paper(
    tmp_path: Path, review: dict[str, Any],
) -> None:
    assert _review_sends_the_experiment_back(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "rewrite"  # type: ignore[arg-type]


def test_a_quest_with_no_experiment_is_never_sent_back_to_implement(tmp_path: Path) -> None:
    """A survey / analyze-only quest has no ``code``: there is nothing to run
    again, whatever the review names."""
    state = _state(XG3_REVIEW)
    del state["code"]
    assert _review_sends_the_experiment_back(XG3_REVIEW, state) == ""  # type: ignore[arg-type]
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


def test_a_flaw_in_the_design_no_check_confirms_is_stated_not_redesigned(tmp_path: Path) -> None:
    """A reviewer's objection to the design that no check of FI's confirms does not re-run the experiment or change the
    design: the paper is written again with the objection kept in FI's record (QUALITY.md)."""
    review = {**XG3_REVIEW, "must_flag_hits": ["[methodologist] circular_evaluation"]}
    assert _review_sends_the_experiment_back(review, _state(review)) == ""  # type: ignore[arg-type]
    assert _engine(tmp_path)._route_after_review(_state(review)) == "rewrite"  # type: ignore[arg-type]


# --- the cost cap ---------------------------------------------------------------

@pytest.mark.parametrize(("done", "route"), [
    (0, "re_execute"), (1, "re_execute"), (2, "rewrite"), (3, "rewrite"),
])
def test_one_re_execute_a_quest_then_the_flag_takes_the_text_route(
    tmp_path: Path, done: int, route: str, measured_fault: str,
) -> None:
    """The review node counts the re-execute before the router reads it, so the
    first flagged review sees 1. A flag that survives the re-run falls back to
    the text route rather than spending the rest of the budget on re-runs."""
    assert _CODE_REEXECUTES == 1
    state = _state(XG3_REVIEW, code_reexecutes=done)
    assert _engine(tmp_path)._route_after_review(state) == route  # type: ignore[arg-type]


def test_the_review_node_counts_the_re_execute(tmp_path: Path, measured_fault: str) -> None:
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
    tmp_path: Path, measured_fault: str,
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

def test_the_implement_prompt_carries_what_the_review_named(tmp_path: Path, measured_fault: str) -> None:
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
    tmp_path: Path, measured_fault: str,
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
