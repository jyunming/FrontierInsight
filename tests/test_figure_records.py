"""What each experiment figure draws reaches the writer, and a caption that
names a series its figure does not show reaches the reviewer (``core/engine.py``
figure records, recorded by ``core/plot_style.py``)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine,
    _figure_caption_findings,
    _figure_list_for_prompt,
    _format_figure_check,
    _read_figure_records,
)

# The validation quest's figure 3, as the bootstrap records it.
DRIFT = {
    "file": "long_term_energy_drift.png",
    "axes": [{
        "title": "Long-term Energy Drift (h=0.01)", "xlabel": "Time (s)", "ylabel": "Energy Difference ΔE (J)",
        "yscale": "linear", "ylim": [-0.022, 0.462],
        "series": [
            {"label": "forward_euler", "min": 0.0, "max": 0.44, "shows": "yes"},
            {"label": "rk4", "min": -1.1e-08, "max": 1.2e-08, "shows": "flat"},
            {"label": "velocity_verlet", "min": 0.0, "max": 3.9e-05, "shows": "flat"},
        ],
    }],
}
PAPER = (
    "# P\n\n## Results\n\n"
    "![**Figure 1.** Error of Forward Euler and RK4.](figures/error.png)\n\n"
    "![**Figure 3.** Long-term energy drift rates for RK4 and Velocity-Verlet.](figures/long_term_energy_drift.png)\n"
)


def test_records_are_read_by_figure_file_name(tmp_path: Path) -> None:
    (tmp_path / "long_term_energy_drift.json").write_text(json.dumps(DRIFT), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    records = _read_figure_records(tmp_path, ["long_term_energy_drift.png", "broken.png", "missing.png"])
    assert list(records) == ["long_term_energy_drift.png"]


def test_the_writer_sees_what_each_figure_draws() -> None:
    listing = _figure_list_for_prompt({
        "figures": ["error.png", "long_term_energy_drift.png"],
        "figure_records": {"long_term_energy_drift.png": DRIFT},
    })
    lines = listing.splitlines()
    assert lines[0] == "- figures/error.png"
    assert lines[1].startswith('- figures/long_term_energy_drift.png — "Long-term Energy Drift (h=0.01)", '
                               'y axis "Energy Difference ΔE (J)" (linear, -0.022 to 0.462), series: ')
    assert "forward_euler 0 to 0.44;" in lines[1]
    assert "rk4 -1.1e-08 to 1.2e-08, FLAT on this axis" in lines[1]
    assert "velocity_verlet 0 to 3.9e-05, FLAT on this axis" in lines[1]
    assert "cannot describe how such a series changes" in lines[2]
    # No record, no note.
    assert _figure_list_for_prompt({"figures": ["a.png"]}) == "- figures/a.png"


def test_a_caption_naming_a_flat_series_is_found() -> None:
    records = {"long_term_energy_drift.png": DRIFT, "error.png": DRIFT | {"file": "error.png"}}
    axis = 'on its axis "Energy Difference ΔE (J)"'
    # In the paper's order; "Velocity-Verlet" in a caption is the series velocity_verlet.
    assert _figure_caption_findings(PAPER, records) == [
        f'figure_caption: the caption of figures/error.png names "rk4", but the figure draws it flat at one value {axis}',
        f'figure_caption: the caption of figures/long_term_energy_drift.png names "rk4", but the figure '
        f"draws it flat at one value {axis}",
        f'figure_caption: the caption of figures/long_term_energy_drift.png names "velocity_verlet", but the '
        f"figure draws it flat at one value {axis}",
    ]
    # A caption about the series that shows, or a label inside another word, is fine.
    assert _figure_caption_findings("![Forward Euler spikes; rk45 differs.](figures/long_term_energy_drift.png)",
                                    {"long_term_energy_drift.png": DRIFT}) == []


def test_the_review_prompt_carries_the_figure_check() -> None:
    state = {"figure_records": {"long_term_energy_drift.png": DRIFT}}
    block = _format_figure_check(PAPER, state)
    assert block.startswith("CAPTIONS THAT DESCRIBE WHAT THEIR FIGURE DOES NOT SHOW (2):")
    assert "`figure_caption`" in block and "must_flag_hits" in block
    assert _format_figure_check("![Forward Euler.](figures/long_term_energy_drift.png)", state).startswith("No caption")
    assert _format_figure_check(PAPER, {}) == "(no record of what the figures draw)"


def test_review_sends_the_figure_check_and_keeps_its_findings(tmp_path: Path) -> None:
    cfg = Config(
        topic="integrators", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    paper = tmp_path / "paper.md"
    paper.write_text(PAPER, encoding="utf-8")
    prompts: list[str] = []

    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        prompts.append(prompt)
        return json.dumps({"verdict": "revise", "score": 3, "must_flag_hits": ["figure_caption"]})

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    state = {"topic": "integrators", "paper_md": str(paper),
             "figure_records": {"long_term_energy_drift.png": DRIFT}}
    review = asyncio.run(eng._node_review(state))["review"]  # type: ignore[arg-type]
    assert "## Figures (what each figure draws" in prompts[0]
    assert 'names "velocity_verlet"' in prompts[0]
    assert len(review["figure_caption_warnings"]) == 2
