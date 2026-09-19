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


# A measured quest's figure: the deterministic reference varies with N in the first two
# panels (0.002 to 0.073, 0.583 to 0.594) and is flat only in the third (0.9405 to 0.9412
# on an axis of 0 to 1). The y axis is labelled on the first panel only.
THREE_PANELS = {
    "file": "conditional_final_size_vs_deterministic.png",
    "axes": [
        {"title": "R0=0.9", "ylabel": "Conditional final infected fraction", "ylim": [0.0, 1.0],
         "series": [{"label": "deterministic", "min": 0.00198, "max": 0.0727, "shows": "yes"}]},
        {"title": "R0=1.5", "ylabel": "", "ylim": [0.0, 1.0],
         "series": [{"label": "deterministic", "min": 0.583, "max": 0.594, "shows": "yes"}]},
        {"title": "R0=3", "ylabel": "", "ylim": [0.0, 1.0],
         "series": [{"label": "deterministic", "min": 0.9405, "max": 0.9412, "shows": "flat"}]},
    ],
}
THREE_PANEL_PAPER = (
    "![**Figure 3.** Conditional final sizes against the deterministic prediction.]"
    "(figures/conditional_final_size_vs_deterministic.png)\n"
)


def test_a_series_flat_in_one_panel_only_is_flagged_with_that_panel_named() -> None:
    """The finding used to read 'draws it flat at one value on its axis "y"' with no
    panel, and the caption was then rewritten to call the reference flat "within each
    R0 panel" -- true of one panel and false of the other two."""
    records = {"conditional_final_size_vs_deterministic.png": THREE_PANELS}
    (finding,) = _figure_caption_findings(THREE_PANEL_PAPER, records)
    assert finding == (
        'figure_caption: the caption of figures/conditional_final_size_vs_deterministic.png names '
        '"deterministic", but the figure draws it flat at one value on its axis '
        '"Conditional final infected fraction" in only some panels: flat in "R0=3" (0.941), but '
        'varying in "R0=0.9" (0.00198 to 0.0727), "R0=1.5" (0.583 to 0.594). '
        "Describe it as flat only where it is flat"
    )


def test_a_series_flat_in_every_panel_keeps_the_plain_finding() -> None:
    """'Flat' is true of the whole figure, so there is nothing to tell the panels apart by."""
    every = {"file": "f.png", "axes": [
        {"title": t, "ylabel": "p", "ylim": [0, 1],
         "series": [{"label": "limit", "min": v, "max": v, "shows": "flat"}]}
        for t, v in (("R0=0.9", 0.0), ("R0=1.5", 0.333), ("R0=3", 0.667))
    ]}
    (finding,) = _figure_caption_findings("![The limit line.](figures/f.png)", {"f.png": every})
    assert finding == (
        'figure_caption: the caption of figures/f.png names "limit", but the figure draws it '
        'flat at one value on its axis "p"'
    )


def test_a_caption_that_says_a_series_lies_flat_is_fine() -> None:
    records = {"long_term_energy_drift.png": DRIFT}
    said = ("![**Figure 3.** Forward Euler drifts to 0.44 J, while RK4 and Velocity-Verlet stay flat "
            "at this scale.](figures/long_term_energy_drift.png)")
    assert _figure_caption_findings(said, records) == []
    # Said of another series, in another clause, it does not count.
    other = "![Forward Euler starts flat, while RK4 climbs.](figures/long_term_energy_drift.png)"
    assert _figure_caption_findings(other, records) == [
        'figure_caption: the caption of figures/long_term_energy_drift.png names "rk4", but the figure '
        'draws it flat at one value on its axis "Energy Difference ΔE (J)"',
    ]


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
