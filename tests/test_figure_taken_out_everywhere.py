"""A figure the evidence gate takes out of the paper's figures leaves every place that would name it (core/engine.py:
``_figures_after_taken_out``, ``_figure_numbers``, ``_place_figures``): no caption, no in-text citation, and the later
figures are numbered again from FI's record of the figures that are left."""
from __future__ import annotations

import json
from pathlib import Path

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _figure_numbers, _place_figures

RECORD = {"axes": [{"ylabel": "value", "series": [{"label": "a", "min": 0.1, "max": 0.9, "shows": "yes"}]}]}
FIGURES = ["fig1_a.png", "fig2_b.png", "fig3_c.png"]
STATE = {
    "design": {"figures_planned": FIGURES}, "figures": list(FIGURES),
    "figure_records": {name: RECORD for name in FIGURES},
}
DRAFT = ("# T\n\n## Results\n\nFigure 1 shows a.\n\nFigure 2 shows b.\n\nFigure 3 shows c.\n\n"
         "## References\n\n1. A cited paper.\n")


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(
        topic="a test topic", title="figure-taken-out", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"))
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    return eng


def _take_out(eng: Engine, name: str) -> None:
    needs = eng.quest_root / "needs"
    needs.mkdir(parents=True, exist_ok=True)
    (needs / "RESULT_CHECK.json").write_text(json.dumps({"removed_figures": [name]}), encoding="utf-8")


def test_the_state_loses_a_figure_the_gate_took_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng._figures_after_taken_out(STATE) == {}  # type: ignore[arg-type]
    _take_out(eng, "fig2_b.png")
    patch = eng._figures_after_taken_out(STATE)  # type: ignore[arg-type]
    assert patch["figures"] == ["fig1_a.png", "fig3_c.png"]
    assert set(patch["figure_records"]) == {"fig1_a.png", "fig3_c.png"}


def test_a_taken_out_figure_has_no_caption_and_no_citation_and_the_later_ones_are_renumbered(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _take_out(eng, "fig2_b.png")
    state = {**STATE, **eng._figures_after_taken_out(STATE)}  # type: ignore[arg-type]
    assert _figure_numbers(state) == {"fig1_a.png": 1, "fig3_c.png": 2}  # type: ignore[arg-type]
    out, placed, _ = _place_figures("# T\n\n## Results\n\nText.\n\n## References\n\n1. A.\n", state)  # type: ignore[arg-type]
    assert placed == ["fig1_a.png", "fig3_c.png"]
    assert "fig2_b.png" not in out
    assert "**Figure 3." not in out and "(Figure 3)" not in out
    assert "![**Figure 1.**" in out and "![**Figure 2.**" in out and out.count("](figures/") == 2


def test_the_gate_node_hands_the_writer_only_the_figures_that_are_left(tmp_path: Path) -> None:
    import asyncio

    eng = _engine(tmp_path)
    _take_out(eng, "fig2_b.png")

    class _Fake:
        async def chat(self, *a, **k):  # noqa: ANN002, ANN003
            return json.dumps({"verdict": "sufficient", "rationale": "ok", "gaps": []})

    eng._client = _Fake()  # type: ignore[assignment]
    patch = asyncio.run(eng._node_evidence_gate({"topic": "a test topic", **STATE}))  # type: ignore[arg-type]
    assert patch["figures"] == ["fig1_a.png", "fig3_c.png"]
