"""A paper that leaves out a figure its design planned and its run drew goes back
to the writer (``core/engine.py``: ``_missing_planned_figures``, the review
node and the rewrite route)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine,
    _format_review_for_writer,
    _hits_need_only_a_rewrite,
    _missing_planned_figures,
)

DESIGN = {"figures_planned": ["final_size.png", "outbreak_prob.png", "gap_fraction.png"]}
FIGURES = ["gap_fraction.png", "final_size.png", "outbreak_prob.png"]
# The shape of a real paper that kept 1 of its 3 figures.
PAPER = (
    "## Results\n\nThe distributions became bimodal, as shown in Figure 1.\n\n"
    "![**Figure 1.** Final size densities.](figures/final_size.png)\n"
)


def _names(hits: list[str]) -> list[str]:
    return [h.split("figures/", 1)[1].split(",", 1)[0] for h in hits]


def test_a_figure_planned_and_drawn_but_left_out_is_missing() -> None:
    hits = _missing_planned_figures(PAPER, {"design": DESIGN, "figures": FIGURES})
    assert _names(hits) == ["outbreak_prob.png", "gap_fraction.png"]
    assert all(h.startswith("figure_missing: ") for h in hits)


@pytest.mark.parametrize("state", [
    # Planned but never drawn: the writer describes it in prose instead.
    {"design": DESIGN, "figures": ["final_size.png"]},
    # Drawn but never planned: the paper may leave it out.
    {"design": {"figures_planned": ["final_size.png"]}, "figures": FIGURES},
    # No design (a survey), or a plan that is not a list.
    {"figures": FIGURES},
    {"design": {"figures_planned": "outbreak_prob.png"}, "figures": FIGURES},
])
def test_only_a_figure_both_planned_and_drawn_counts(state: dict) -> None:
    assert _missing_planned_figures(PAPER, state) == []


def test_a_paper_with_every_figure_is_clear() -> None:
    paper = PAPER + "![Figure 2.](figures/outbreak_prob.png)\n![Figure 3.](./figures/gap_fraction.png)\n"
    assert _missing_planned_figures(paper, {"design": DESIGN, "figures": FIGURES}) == []


def test_a_missing_figure_is_a_rewrite_not_a_rerun() -> None:
    hits = _missing_planned_figures(PAPER, {"design": DESIGN, "figures": FIGURES})
    assert _hits_need_only_a_rewrite(hits)
    assert _hits_need_only_a_rewrite(["unsupported_claim", *hits])


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


@pytest.mark.parametrize("panel", [False, True])
def test_review_forces_the_rewrite_even_when_the_reviewer_accepts(tmp_path: Path, panel: bool) -> None:
    eng = _engine(tmp_path)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]
    nodes: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        nodes.append(node or "")
        if node == "review_moderator":
            return json.dumps({"rationale": "both reviewers accept"})
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text(PAPER, encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper),
        "design": DESIGN, "figures": FIGURES,
    }))
    review = patch["review"]

    assert any(n.startswith("review_panel.") for n in nodes) == panel
    assert _names(review["must_flag_hits"]) == ["outbreak_prob.png", "gap_fraction.png"]
    assert patch["iteration"] == 1, "a forced rewrite spends budget"
    assert eng._route_after_review({"review": review, "iteration": 1}) == "rewrite"  # type: ignore[arg-type]
    assert "figures/gap_fraction.png" in _format_review_for_writer({"review": review})  # type: ignore[typeddict-item]
