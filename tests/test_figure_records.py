"""What each experiment figure draws reaches the writer, and a caption that
names a series its figure does not show, or calls a series flat that its figure
draws varying, reaches the reviewer (``core/engine.py`` figure records, recorded
by ``core/plot_style.py``)."""

from __future__ import annotations

import asyncio
import copy
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
    _hit_name,
    _hits_need_only_a_rewrite,
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


# A measured quest's figure 3 (outputs of a stored SIR quest), as its recorder wrote it: the
# "deterministic" reference varies over 7.1% of the axis in the R0=0.9 panel (0.002 to 0.073),
# 1.1% in R0=1.5, and is flat in R0=3. The paper's caption calls it a flat reference line
# "within each R0 panel".
CB1_FILE = "conditional_final_size_vs_deterministic.png"
CB1 = {
    "file": CB1_FILE,
    "axes": [
        {"title": "R₀=0.9", "xlabel": "Population size", "ylabel": "Conditional final infected fraction",
         "yscale": "linear", "ylim": [0.0, 1.0], "series": [
             {"label": "threshold 5%", "min": 0.106, "max": 0.14, "shows": "yes"},
             {"label": "deterministic", "min": 0.00198056, "max": 0.0727123, "shows": "yes"}]},
        {"title": "R₀=1.5", "xlabel": "Population size", "ylabel": "",
         "yscale": "linear", "ylim": [0.0, 1.0], "series": [
             {"label": "threshold 5%", "min": 0.428678, "max": 0.584977, "shows": "yes"},
             {"label": "deterministic", "min": 0.583034, "max": 0.593634, "shows": "yes"}]},
        {"title": "R₀=3", "xlabel": "Population size", "ylabel": "",
         "yscale": "linear", "ylim": [0.0, 1.0], "series": [
             {"label": "threshold 5%", "min": 0.919899, "max": 0.940185, "shows": "yes"},
             {"label": "deterministic", "min": 0.940494, "max": 0.941203, "shows": "flat"}]},
    ],
}
CB1_CAPTION = (
    "**Figure 3.** Conditional final infected fractions under 5%, 10%, and 20% major-outbreak "
    "thresholds, with deterministic final-size values drawn as flat reference lines within each "
    r"\(R_0\) panel."
)


def _paper(caption: str, name: str = CB1_FILE) -> str:
    return f"![{caption}](figures/{name})\n"


def test_a_caption_that_calls_a_varying_series_flat_is_flagged_with_its_panel_and_range() -> None:
    """The reverse of the check above: the figure draws the series varying and the caption says
    it is flat. Only the R0=0.9 panel is named: R0=1.5 covers 1.1% of its axis, which is not
    a slope the caption is wrong to call flat."""
    (finding,) = _figure_caption_findings(_paper(CB1_CAPTION), {CB1_FILE: CB1})
    assert finding == (
        f'figure_caption: the caption of figures/{CB1_FILE} calls "deterministic" flat, but the '
        'figure draws it varying in "R₀=0.9" (0.00198 to 0.0727). Say that it varies there, or '
        "describe only the panels where it is flat"
    )
    # Two panels over 5%: both are named, in the figure's order.
    two = copy.deepcopy(CB1)
    two["axes"][1]["series"][1] |= {"min": 0.4, "max": 0.6}
    (finding,) = _figure_caption_findings(_paper(CB1_CAPTION), {CB1_FILE: two})
    assert 'varying in "R₀=0.9" (0.00198 to 0.0727), "R₀=1.5" (0.4 to 0.6). Say that' in finding


def test_a_one_panel_figure_names_the_axis_instead_of_a_panel() -> None:
    (finding,) = _figure_caption_findings(
        "![Forward Euler is drawn as a flat line.](figures/long_term_energy_drift.png)",
        {"long_term_energy_drift.png": DRIFT},
    )
    assert finding == (
        'figure_caption: the caption of figures/long_term_energy_drift.png calls "forward_euler" '
        'flat, but the figure draws it varying (0 to 0.44) on its axis "Energy Difference ΔE (J)". '
        "Say that it varies"
    )


def test_the_same_caption_on_a_figure_that_draws_the_series_flat_everywhere_is_fine() -> None:
    every_flat = copy.deepcopy(CB1)
    for ax, value in zip(every_flat["axes"], (0.0727, 0.5936, 0.9412)):
        ax["series"][1] = {"label": "deterministic", "min": value, "max": value, "shows": "flat"}
    assert _figure_caption_findings(_paper(CB1_CAPTION), {CB1_FILE: every_flat}) == []


def test_a_hedged_clause_is_not_a_flat_claim() -> None:
    """A stored quest's caption said "nearly flat deterministic ODE reference series"; the series
    covers 7.1% of the axis in one panel. Hedged, so it is left to the reader. The same words
    without the hedge are flagged, so the hedge is what the difference rests on."""
    for hedge in ("nearly flat", "Approximately constant", "roughly-flat", "essentially unchanged",
                  "almost perfectly flat", "near-flat"):
        caption = f"Attack rates against the {hedge} deterministic reference within each R0 panel."
        assert _figure_caption_findings(_paper(caption), {CB1_FILE: CB1}) == [], hedge
    # "near" hedges only the word beside it: a flat line placed near something is still a flat line.
    placed = "Attack rates against the deterministic reference drawn near the flat threshold lines."
    (finding,) = _figure_caption_findings(_paper(placed), {CB1_FILE: CB1})
    assert 'calls "deterministic" flat' in finding
    for plain in ("flat", "constant", "unchanged", "horizontal"):
        caption = f"Attack rates against the {plain} deterministic reference within each R0 panel."
        # ("horizontal" also draws the first finding, which does not count it as saying flat.)
        (finding,) = [f for f in _figure_caption_findings(_paper(caption), {CB1_FILE: CB1}) if " calls " in f]
        assert 'calls "deterministic" flat' in finding, plain


def test_a_clause_that_denies_flatness_is_not_a_flat_claim() -> None:
    for said in ("the deterministic reference is not flat at R0=0.9", "a non-constant deterministic reference",
                 "the deterministic reference isn't flat there", "the deterministic curve never stays flat"):
        assert _figure_caption_findings(_paper(f"Attack rates; {said}."), {CB1_FILE: CB1}) == [], said


def test_a_clause_that_confines_flatness_to_part_of_the_run_is_not_a_flat_claim() -> None:
    """"Forward Euler starts flat" (the caption above) says the series is flat for a while, and the
    figure draws it climbing afterwards: nothing the figure contradicts."""
    for said in ("the deterministic reference starts flat", "the deterministic reference is flat until N = 500",
                 "the deterministic reference is flat at first and then falls",
                 "the deterministic reference is flat before the threshold",
                 "the deterministic reference is flat only for large N"):
        assert _figure_caption_findings(_paper(f"Attack rates; {said}."), {CB1_FILE: CB1}) == [], said
    # "stays flat" is a claim about the whole run.
    (finding,) = _figure_caption_findings(_paper("Attack rates; the deterministic reference stays flat."),
                                          {CB1_FILE: CB1})
    assert 'calls "deterministic" flat' in finding


def test_a_flat_claim_in_a_clause_about_another_series_is_not_flagged() -> None:
    records = {"long_term_energy_drift.png": DRIFT}
    said = "![Forward Euler climbs to 0.44 J, while the RK4 line is flat.](figures/long_term_energy_drift.png)"
    assert _figure_caption_findings(said, records) == []
    # The same words in one clause are about Forward Euler, which the figure draws climbing.
    one = "![Forward Euler is flat next to the RK4 line.](figures/long_term_energy_drift.png)"
    (finding,) = _figure_caption_findings(one, records)
    assert 'calls "forward_euler" flat' in finding


def test_a_series_the_recorder_gave_a_range_of_zero_is_not_flagged() -> None:
    """The recorder reads ``ax.hlines`` and ``fill_between`` through ``get_offsets()`` and records
    min = max = 0 "yes" for them (18 of 918 series in the stored quests). Their range is 0, which is
    not over 5% of any axis, and the caption is right: they are horizontal lines."""
    hlines = {"file": "f.png", "axes": [{
        "title": "R₀ = 0.9", "ylabel": "Final size", "yscale": "linear", "ylim": [-0.02, 1.02],
        "series": [{"label": "Large-N final-size root", "min": 0.0, "max": 0.0, "shows": "yes"}],
    }]}
    caption = "![The large-N final-size root is drawn as a flat line.](figures/f.png)"
    assert _figure_caption_findings(caption, {"f.png": hlines}) == []


def test_a_series_varying_under_five_percent_of_the_axis_is_not_flagged() -> None:
    def one_panel(low: float, high: float) -> dict:
        return {"file": "f.png", "axes": [{
            "title": "", "ylabel": "p", "yscale": "linear", "ylim": [0.0, 1.0],
            "series": [{"label": "deterministic", "min": low, "max": high, "shows": "yes"}],
        }]}

    caption = "![The deterministic values are drawn as flat reference lines.](figures/f.png)"
    assert _figure_caption_findings(caption, {"f.png": one_panel(0.50, 0.53)}) == []  # 3%
    (finding,) = _figure_caption_findings(caption, {"f.png": one_panel(0.50, 0.57)})  # 7%
    assert finding.endswith('varying (0.5 to 0.57) on its axis "p". Say that it varies')


def test_the_share_of_the_axis_is_measured_as_the_recorder_measures_flat() -> None:
    """On a log axis in decades, clipped to the axis, as ``core/plot_style.py`` decides "flat"."""
    def one_panel(yscale: str) -> dict:
        return {"file": "f.png", "axes": [{
            "title": "", "ylabel": "p", "yscale": yscale, "ylim": [1e-4, 1.0],
            "series": [{"label": "limit", "min": 1e-3, "max": 3e-3, "shows": "yes"}],
        }]}

    caption = "![The limit is drawn as a flat line.](figures/f.png)"
    # A factor of 3 is 12% of four decades, and 0.2% of the axis on a linear scale.
    assert len(_figure_caption_findings(caption, {"f.png": one_panel("log")})) == 1
    assert _figure_caption_findings(caption, {"f.png": one_panel("linear")}) == []
    # A line that runs off the axis covers only what is on it: 0.98 to 1.0 of an axis of 0 to 1 is 2%,
    # and 0.5 to 1.0 is 50%, whatever the values beyond the top are.
    clipped = one_panel("linear")
    clipped["axes"][0]["ylim"] = [0.0, 1.0]
    clipped["axes"][0]["series"][0] |= {"min": 0.98, "max": 5.0}
    assert _figure_caption_findings(caption, {"f.png": clipped}) == []
    clipped["axes"][0]["series"][0] |= {"min": 0.5}
    assert len(_figure_caption_findings(caption, {"f.png": clipped})) == 1


def test_a_record_that_cannot_give_a_share_is_never_flagged() -> None:
    caption = "![The limit is drawn as a flat line.](figures/f.png)"
    base = {"label": "limit", "min": 0.1, "max": 0.9, "shows": "yes"}
    for axes in (
        [{"title": "", "yscale": "linear", "series": [base]}],                                   # no ylim
        [{"title": "", "yscale": "linear", "ylim": [0.0, 0.0], "series": [base]}],               # empty axis
        [{"title": "", "yscale": "linear", "ylim": [0.0, 1.0], "series": [base | {"min": "x"}]}],
        [{"title": "", "yscale": "linear", "ylim": [0.0, 1.0], "series": [{"label": "limit", "shows": "yes"}]}],
        [{"title": "", "yscale": "log", "ylim": [0.0, 1.0], "series": [base]}],                  # log axis to 0
        [{"title": "", "yscale": "linear", "ylim": [0.0, 1.0], "series": ["limit", None]}],
    ):
        assert _figure_caption_findings(caption, {"f.png": {"file": "f.png", "axes": axes}}) == []


def test_a_flat_claim_is_matched_by_the_exact_label_ignoring_case_and_punctuation() -> None:
    record = {"file": "f.png", "axes": [{
        "title": "", "ylabel": "p", "yscale": "linear", "ylim": [0.0, 1.0],
        "series": [{"label": "Deterministic_ODE", "min": 0.1, "max": 0.6, "shows": "yes"}],
    }]}
    for caption in ("The **DETERMINISTIC-ode** reference is drawn FLAT.",
                    "The `deterministic ode` reference is a Flat line.",
                    "Each panel has a Horizontal deterministic_ode line."):
        assert len(_figure_caption_findings(f"![{caption}](figures/f.png)", {"f.png": record})) == 1, caption
    # Not the exact label: a longer word, or part of the label only ("cr1": "Deterministic ODE"
    # is not "deterministic final-size predictions"), so nothing is named.
    for caption in ("The deterministic odes are flat.", "The deterministic final-size predictions are flat.",
                    "The ODE reference is flat."):
        assert _figure_caption_findings(f"![{caption}](figures/f.png)", {"f.png": record}) == [], caption


def test_only_a_clause_that_says_flat_of_the_series_counts() -> None:
    varying = {"file": "f.png", "axes": [{
        "title": "", "ylabel": "p", "yscale": "linear", "ylim": [0.0, 1.0],
        "series": [{"label": "constant N", "min": 0.1, "max": 0.6, "shows": "yes"}],
    }]}
    # The series is labelled "constant N": naming it does not call it constant.
    assert _figure_caption_findings("![The constant N runs against the mean.](figures/f.png)", {"f.png": varying}) == []
    assert len(_figure_caption_findings("![The constant N runs are flat.](figures/f.png)", {"f.png": varying})) == 1
    # "horizontal" as the axis is not a claim about a series.
    axis = "![Constant N against population size on the horizontal axis.](figures/f.png)"
    assert _figure_caption_findings(axis, {"f.png": varying}) == []


def test_a_caption_that_scopes_flat_to_the_panels_where_it_is_flat_is_fine() -> None:
    """The finding asks for this: "describe only the panels where it is flat". A clause that names
    a panel where the series is not varying says where it is flat; naming only a panel where it
    varies does not."""
    for scoped in (
        "the deterministic values are flat at R₀=3 and fall with N at R₀=0.9",
        "the deterministic values are flat in the R0 = 3 panel",  # spacing is not the title's
        r"the deterministic values are flat for \(R_0=3\)",
        "the deterministic values are flat at R₀=1.5 and R₀=3",
    ):
        assert _figure_caption_findings(_paper(f"Attack rates; {scoped}."), {CB1_FILE: CB1}) == [], scoped
    wrong = "Attack rates; the deterministic values are flat at R₀=0.9."
    (finding,) = _figure_caption_findings(_paper(wrong), {CB1_FILE: CB1})
    assert 'draws it varying in "R₀=0.9"' in finding
    # A panel with no title is named by its place, as the finding names it.
    untitled = copy.deepcopy(CB1)
    for ax in untitled["axes"]:
        ax["title"] = ""
    assert _figure_caption_findings(_paper("Attack rates; the deterministic values are flat in panel 3."),
                                    {CB1_FILE: untitled}) == []


def test_the_two_findings_about_a_series_flat_in_some_panels_can_both_be_answered() -> None:
    """Naming it without saying flat is the first finding; saying flat of all panels is the second.
    The rewrite that satisfies both is naming the panel where it is flat."""
    plain = "Attack rates against the deterministic reference within each R0 panel."
    (first,) = _figure_caption_findings(_paper(plain), {CB1_FILE: CB1})
    assert "draws it flat at one value" in first and "calls" not in first
    (second,) = _figure_caption_findings(_paper(CB1_CAPTION), {CB1_FILE: CB1})
    assert 'calls "deterministic" flat' in second
    both = "Attack rates against the deterministic reference that is flat at R₀=3 and falls elsewhere."
    assert _figure_caption_findings(_paper(both), {CB1_FILE: CB1}) == []


def test_the_new_finding_reaches_the_reviewer_and_sends_the_paper_back_to_be_rewritten() -> None:
    state = {"figure_records": {CB1_FILE: CB1}}
    block = _format_figure_check(_paper(CB1_CAPTION), state)
    assert block.startswith("CAPTIONS THAT DESCRIBE WHAT THEIR FIGURE DOES NOT SHOW (1):")
    assert 'calls "deterministic" flat' in block and "`figure_caption`" in block and "must_flag_hits" in block
    (finding,) = _figure_caption_findings(_paper(CB1_CAPTION), {CB1_FILE: CB1})
    # The hit's name is what routes it: text-only, so the experiment is not run again.
    assert _hit_name(finding) == "figure_caption"
    assert _hits_need_only_a_rewrite([finding])
    assert _format_figure_check(_paper("Attack rates by population size."), state).startswith("No caption")


def test_a_caption_that_says_a_series_lies_flat_is_fine_after_the_reverse_check_too() -> None:
    """The reverse check does not reach the caption the first one accepts."""
    said = ("![**Figure 3.** Forward Euler drifts to 0.44 J, while RK4 and Velocity-Verlet stay flat "
            "at this scale.](figures/long_term_energy_drift.png)")
    assert _figure_caption_findings(said, {"long_term_energy_drift.png": DRIFT}) == []


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

