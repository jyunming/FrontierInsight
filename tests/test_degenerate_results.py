"""Results and figures that show nothing are sent back, and left out when they stay (core/degenerate_results.py, made after the
run through core/result_checks.py). Neutral toy records; the recorder test runs a real plot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import degenerate_results as dr
from core import reported_settings as rs
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.execution import ExecutionResult
from core.plot_style import write_boot

RATES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def _cells(**metrics: Any) -> list[dict[str, Any]]:
    """Eight settings of one setting; each keyword is a quantity, as a function of the setting or a fixed number."""
    cells = []
    for r in RATES:
        cells.append({"cell": {"rate": r}, "metrics": {
            name: [fn(r) if callable(fn) else fn] for name, fn in metrics.items()}})
    return cells


REPORTED = {"gain", "loss", "level", "capped", "counted", "rate"}


def _flags(cells: list[dict[str, Any]], reported: set[str] = REPORTED, exempt: set[str] = frozenset()) -> list[tuple[str, str]]:
    return [(p.name, p.kind) for p in dr.quantity_problems(cells, reported, exempt)]


# ---- quantities -----------------------------------------------------------------------------------------------------------


def test_a_quantity_that_is_the_same_at_every_setting_while_others_change_is_caught() -> None:
    found = dr.quantity_problems(_cells(gain=lambda r: 3 * r, level=2.5), REPORTED, set())
    assert [(p.name, p.kind, p.value, p.cells) for p in found] == [("level", "constant", 2.5, 8)]
    assert "`level` is 2.5 at every one of the 8 settings although other results change with them" in found[0].says()


def test_a_quantity_stuck_at_its_own_largest_value_for_most_settings_is_caught() -> None:
    found = dr.quantity_problems(_cells(gain=lambda r: r, capped=lambda r: min(r * 2.0, 6.0)), REPORTED, set())
    assert [(p.name, p.kind, p.value, p.at) for p in found] == [("capped", "saturated", 6.0, 6)]
    assert "the largest or smallest value it takes anywhere" in found[0].says()


@pytest.mark.parametrize("name,fn", [
    ("loss", lambda r: 100.0 / r),                   # moves
    ("counted", lambda r: float(r > 3)),             # an indicator sits at 0 or 1 for most settings: allowed
    ("capped", lambda r: min(r, 7.0)),               # reaches its largest value at only two settings of eight
    ("level", lambda r: 0.0 if r > 2 else 1.5),      # mostly zero: allowed
])
def test_quantities_that_may_be_where_they_are_pass(name: str, fn: Any) -> None:
    assert _flags(_cells(gain=lambda r: 3 * r, **{name: fn})) == []


def test_only_what_the_analysis_reports_and_is_not_a_setting_is_judged() -> None:
    cells = _cells(gain=lambda r: 3 * r, diagnostic=1e-16, level=2.5)
    assert _flags(cells, {"gain", "level"}) == [("level", "constant")], "a diagnostic nobody reports is not a result"
    assert _flags(cells, {"gain", "level"}, {"level"}) == [], "a fixed setting echoed back is not a result"
    echo = _cells(gain=lambda r: 3 * r, rate_hz=lambda r: r)
    assert _flags(echo, {"gain", "rate_hz"}) == []
    # a quantity reported under its name with a unit word added is reported
    assert _flags(_cells(gain=lambda r: 3 * r, level_kg=2.5), {"gain", "level"}) == [("level_kg", "constant")]


def test_everything_the_same_is_flat_outputs_and_a_single_setting_has_nothing_to_compare() -> None:
    assert _flags(_cells(gain=1.0, level=2.5)) == [], "when nothing changes, core.flat_output says so"
    assert _flags(_cells(gain=lambda r: r, level=2.5)[:2]) == []
    one = [{"cell": {"rate": 1.0}, "metrics": {"gain": [1.0], "level": [2.0]}}] * 8
    assert _flags(one) == []


# ---- figures --------------------------------------------------------------------------------------------------------------


def _record(directory: Path, stem: str, panels: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(json.dumps({"file": f"{stem}.png", "axes": [
        {"title": p.get("title", ""), "xlabel": p.get("xlabel", ""), "series": p.get("series", [])} for p in panels]}), encoding="utf-8")
    (directory / f"{stem}.seed0.json").write_text(json.dumps({"file": f"{stem}.png", "axes": [
        {"title": p.get("title", ""), "xlabel": p.get("xlabel", ""), "lines": p.get("lines", []),
         **({"draws_other": p["draws_other"]} if "draws_other" in p else {})} for p in panels]}), encoding="utf-8")


def _line(x: list[float], y: list[float], kind: str = "data") -> dict[str, Any]:
    return {"label": "a", "kind": kind, "x": x, "y": y}


def test_a_panel_that_draws_nothing_and_a_series_at_one_x_are_caught(tmp_path: Path) -> None:
    _record(tmp_path, "empty_one", [{"title": "First", "xlabel": "x"}, {"title": "Second", "xlabel": "x"}])
    _record(tmp_path, "one_x", [{"title": "Trend", "lines": [_line([5.0] * 6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]}])
    got = dr.figure_problems(tmp_path, ["empty_one.png", "one_x.png"])
    assert [(p.figure, p.panel, p.kind) for p in got] == [
        ("empty_one.png", "First", "empty"), ("empty_one.png", "Second", "empty"), ("one_x.png", "Trend", "one_x")]
    assert "`empty_one.png`, panel \"First\" draws no data" in got[0].says()
    assert "plots 6 points all at x = 5 (a), so it shows no trend" in got[2].says()


def test_panels_that_draw_something_pass(tmp_path: Path) -> None:
    _record(tmp_path, "fine", [
        {"title": "Line", "lines": [_line([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])], "series": [{"label": "a"}]},
        {"title": "Bars", "draws_other": True},
        {"title": "One marker", "lines": [_line([5.0], [1.0])]},
        {"title": "A vertical reference", "lines": [_line([5.0, 5.0], [0.0, 1.0])]},
        {"title": "A horizontal level with data", "lines": [_line([1.0, 9.0, 4.0], [2.0, 2.0, 2.0], kind="hline"),
                                                           _line([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])]},
        {"title": "Constant y at a few x", "lines": [_line([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])]},
    ])
    assert dr.figure_problems(tmp_path, ["fine.png"]) == []
    assert dr.figure_problems(tmp_path, ["nothing_recorded.png"]) == []
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert dr.figure_problems(tmp_path, ["broken.png"]) == []


def test_the_recorder_says_whether_a_panel_draws_anything_but_lines(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    script = (
        "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
        "fig, axes = plt.subplots(1, 4)\n"
        "axes[0].set_title('empty'); axes[0].set_xlabel('x')\n"
        "axes[1].bar([1, 2, 3], [2, 3, 1]); axes[1].set_title('bars')\n"
        "axes[2].imshow([[0, 1], [1, 0]]); axes[2].set_title('image')\n"
        "axes[3].plot([1, 2, 3], [1, 2, 3], label='line'); axes[3].set_title('line')\n"
        "fig.savefig('fig.png')\n"
    )
    records = tmp_path / "records"
    env = {**{k: v for k, v in os.environ.items()}, "FI_FIGURE_RECORDS": str(records),
           "PYTHONPATH": os.pathsep.join(p for p in (str(write_boot(tmp_path / ".boot", "latex")), os.environ.get("PYTHONPATH", "")) if p)}
    out = subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    panels = json.loads((records / "fig.seed0.json").read_text(encoding="utf-8"))["axes"]
    assert [p["draws_other"] for p in panels] == [False, True, True, False]
    assert [(p.panel, p.kind) for p in dr.figure_problems(records, ["fig.png"])] == [("empty", "empty")]


def _draw(tmp_path: Path, script: str) -> Path:
    """Run ``script`` (matplotlib code) under the plot recorder; the records folder."""
    pytest.importorskip("matplotlib")
    records = tmp_path / "records"
    env = {**os.environ, "FI_FIGURE_RECORDS": str(records),
           "PYTHONPATH": os.pathsep.join(p for p in (str(write_boot(tmp_path / ".boot", "latex")), os.environ.get("PYTHONPATH", "")) if p)}
    out = subprocess.run([sys.executable, "-c", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n" + script],
                         env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return records


THREE_FIGURES = (
    "m = [20, 200, 2000]\n"
    # one panel holds only a reference line (a stray axis), the others hold data
    "fig, axes = plt.subplots(1, 3)\n"
    "axes[0].plot(m, [0.6, 0.7, 0.8], label='a'); axes[0].set_title('with data')\n"
    "axes[1].axhline(0.05, linestyle='--', label='reference'); axes[1].set_title('only the reference'); axes[1].set_xlabel('m')\n"
    "axes[2].plot(m, [0.1, 0.2, 0.3], label='b'); axes[2].axhline(0.05, color='k'); axes[2].set_title('data and reference')\n"
    "fig.savefig('reference_only.png')\n"
    # a value of None at two settings: matplotlib drops the points
    "fig, ax = plt.subplots()\n"
    "ax.plot(m, [0.6, None, None], 'o', label='uncorrected'); ax.plot(m, [0.6, 0.9, 1.0], label='theory'); ax.set_title('Chance')\n"
    "fig.savefig('none_values.png')\n"
    # a series filtered before it was plotted: only the first setting is drawn
    "fig, ax = plt.subplots()\n"
    "kept = [(x, y) for x, y in zip(m, [0.6, None, None]) if y is not None]\n"
    "ax.plot([k[0] for k in kept], [k[1] for k in kept], 'o', label='uncorrected'); ax.plot(m, [0.6, 0.9, 1.0], label='theory')\n"
    "ax.set_title('Chance'); fig.savefig('dropped_before.png')\n"
    # a normal figure: every series at every setting, a data line on a dense axis, a reference line
    "import numpy as np\n"
    "fig, ax = plt.subplots()\n"
    "ax.plot(m, [0.1, 0.2, 0.3], 'o', label='measured'); ax.plot(np.linspace(20, 2000, 100), np.linspace(0.1, 0.3, 100), label='curve')\n"
    "ax.plot(m, [0.2, 0.3, 0.4], label='other'); ax.axhline(0.05, linestyle='--', label='reference'); ax.set_title('Normal')\n"
    "fig.savefig('normal.png')\n"
)


def test_a_panel_with_only_a_reference_line_is_empty_and_a_normal_figure_raises_nothing(tmp_path: Path) -> None:
    records = _draw(tmp_path, THREE_FIGURES)
    names = ["reference_only.png", "none_values.png", "dropped_before.png", "normal.png"]
    assert [(p.figure, p.panel, p.kind) for p in dr.figure_problems(records, names)] == [
        ("reference_only.png", "only the reference", "empty")]
    # a reference line beside data, or a twin axis that carries the data, is not an empty panel
    (tmp_path / "twin").mkdir()
    twin = _draw(tmp_path / "twin", "fig, ax = plt.subplots(); ax.axhline(1.0); ax.set_title('left'); ax.set_ylabel('r')\n"
                                    "ax2 = ax.twinx(); ax2.plot([1, 2, 3], [3, 2, 1]); fig.savefig('twin.png')\n")
    assert dr.figure_problems(twin, ["twin.png"]) == []


def test_a_constant_series_with_markers_a_twin_made_with_add_axes_and_a_legend_panel_are_not_empty(tmp_path: Path) -> None:
    # a measured series that happens to be constant, drawn dashed with markers, is data (each marker a setting)
    (tmp_path / "marked").mkdir()
    marked = _draw(tmp_path / "marked", "fig, ax = plt.subplots(); ax.plot([1, 2, 3], [0.0, 0.0, 0.0], 'o--', label='error')\n"
                                        "ax.set_title('Error'); fig.savefig('marked.png')\n")
    assert dr.figure_problems(marked, ["marked.png"]) == []
    # twin axes on axes made without a grid (add_axes) share the place
    (tmp_path / "boxes").mkdir()
    boxes = _draw(tmp_path / "boxes", "fig = plt.figure(); ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]); ax.axhline(1.0)\n"
                                      "ax.set_title('left'); ax2 = ax.twinx(); ax2.plot([1, 2, 3], [3, 2, 1])\n"
                                      "fig.savefig('boxes.png')\n")
    assert dr.figure_problems(boxes, ["boxes.png"]) == []
    # a panel that only holds the legend the other panels share
    (tmp_path / "legend").mkdir()
    legend = _draw(tmp_path / "legend", "fig, (a, b) = plt.subplots(1, 2); h, = a.plot([1, 2, 3], [1, 2, 3], label='x')\n"
                                        "a.set_title('Data'); b.axis('off'); b.legend(handles=[h]); b.set_title('Legend')\n"
                                        "fig.savefig('legend.png')\n")
    assert dr.figure_problems(legend, ["legend.png"]) == []
    # the same reference line alone, with no data anywhere, is still empty
    (tmp_path / "alone").mkdir()
    alone = _draw(tmp_path / "alone", "fig, ax = plt.subplots(); ax.plot([1, 2, 3], [0.5, 0.5, 0.5], '--')\n"
                                      "ax.set_title('Only the line'); fig.savefig('alone.png')\n")
    assert [p.kind for p in dr.figure_problems(alone, ["alone.png"])] == ["empty"]


def test_a_series_that_misses_settings_the_figure_otherwise_shows_is_counted_from_the_plotted_data(tmp_path: Path) -> None:
    records = _draw(tmp_path, THREE_FIGURES)
    lines = json.loads((records / "none_values.seed0.json").read_text(encoding="utf-8"))["axes"][0]["lines"]
    assert [ln["missing_x"] for ln in lines] == [[200.0, 2000.0], []]
    got = {g.figure: g for g in dr.figure_gaps(records, ["reference_only.png", "none_values.png", "dropped_before.png", "normal.png"])}
    assert set(got) == {"none_values.png", "dropped_before.png"}, "the normal figure, with a dense curve and a reference, raises nothing"
    assert (got["none_values.png"].why, got["none_values.png"].absent) == ("no value", [200.0, 2000.0])
    assert (got["dropped_before.png"].why, got["dropped_before.png"].absent) == ("not plotted", [200.0, 2000.0])
    assert "the series \"uncorrected\" has no value at x = 200, 2000" in got["none_values.png"].says()
    note = dr.gap_note(list(got.values()))
    assert "as shown in Figure N" in note and "caption" in note and "uncorrected" in note


# ---- the words ------------------------------------------------------------------------------------------------------------


def test_the_request_names_each_thing_and_ends_with_the_instruction() -> None:
    q = dr.quantity_problems(_cells(gain=lambda r: 3 * r, level=2.5), REPORTED, set())
    f = [dr.FigureProblem("a.png", "Trend", "empty")]
    text = dr.directive(f, q)
    assert text.startswith("FI looked at what the run produced, and some of it shows nothing:")
    assert "`level` is 2.5" in text and "`a.png`, panel \"Trend\" draws no data" in text
    assert "estimated from the trials' data: compute it from this trial's data" in text
    assert text.rstrip().endswith("Change nothing else.")
    note = dr.writer_note(f, q)
    assert "`level`" in note and "`a.png`" in note and "limitations" in note and "left out" in note


# ---- through the engine ---------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic", title="degenerate", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._trial_mode = True
    named = ["gain", "level", "capped"]   # the plan names these as estimated from the data (a flat one is questioned)
    eng._protocol_block = lambda state: {"metrics": [{"id": n} for n in named]}  # type: ignore[method-assign]
    root = eng.quest_root
    (root / "raw").mkdir(parents=True, exist_ok=True)
    record = {"cells": [{"cell": c["cell"], "metrics": {n: {"values": v} for n, v in c["metrics"].items()}}
                        for c in _cells(gain=lambda r: 3 * r, level=2.5)]}
    (root / "raw" / "trials.json").write_text(json.dumps(record), encoding="utf-8")
    (root / "figures").mkdir(exist_ok=True)
    (root / "figures" / "empty_one.png").write_bytes(b"png")
    (root / "figures" / "fine.png").write_bytes(b"png")
    _record(eng.fi_dir / "figure_records", "empty_one", [{"title": "First", "xlabel": "x"}])
    _record(eng.fi_dir / "figure_records", "fine", [{"title": "Line", "lines": [_line([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])]}])
    return eng


def _run() -> ExecutionResult:
    return ExecutionResult(0, "RESULT_JSON: " + json.dumps({"gain": 3.0, "level": 2.5, "rows": [{"level": 2.5}]}), "", 1.0, False)


@pytest.mark.asyncio
async def test_empty_figures_and_constant_quantities_are_sent_back_to_the_simulation_then_left_out_and_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch)
    state: dict[str, Any] = {"iteration": 0}
    for n in (1, 2):
        result, script = await eng._results_match_run(state, _run(), None, True)
        assert result.returncode == 1 and script == "simulate.py", "a quantity the simulation does not compute: the simulation"
        assert "FI looked at what the run produced" in result.stderr and "`level` is 2.5" in result.stderr
        assert result.stderr.rstrip().endswith("Change nothing else.")
    # (each send-back cleared the figures; the repaired run draws them again)
    (eng.quest_root / "figures" / "empty_one.png").write_bytes(b"png")
    (eng.quest_root / "figures" / "fine.png").write_bytes(b"png")
    result, script = await eng._results_match_run(state, _run(), None, True)
    assert result.returncode == 0
    record = json.loads((eng.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_quantities"] == ["level"] and record["removed_figures"] == ["empty_one.png"]
    assert not (eng.quest_root / "figures" / "empty_one.png").exists() and (eng.quest_root / "figures" / "fine.png").exists()
    assert not (eng.fi_dir / "figure_records" / "empty_one.json").exists() and (eng.fi_dir / "figure_records" / "fine.json").exists()
    assert eng._without_typed_results(json.loads(_run().stdout[len("RESULT_JSON: "):])) == {"gain": 3.0, "rows": [{}]}
    assert "`level`" in eng._run_checks_note() and "limitations" in eng._run_checks_note()


@pytest.mark.asyncio
async def test_a_figure_alone_goes_back_to_the_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch)
    record = {"cells": [{"cell": c["cell"], "metrics": {n: {"values": v} for n, v in c["metrics"].items()}}
                        for c in _cells(gain=lambda r: 3 * r, level=lambda r: r * r)]}
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps(record), encoding="utf-8")
    result, script = await eng._results_match_run({"iteration": 0}, _run(), None, True)
    assert result.returncode == 1 and script == "experiment.py" and "draws no data" in result.stderr
    # once, the figures of the failed run are cleared like any failed run's
    assert not (eng.quest_root / "figures" / "fine.png").exists()


@pytest.mark.asyncio
async def test_a_run_that_saved_no_figure_though_the_plan_has_some_keeps_its_results_and_says_so_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch)
    for p in (eng.quest_root / "figures").iterdir():
        p.unlink()
    record = {"cells": [{"cell": c["cell"], "metrics": {n: {"values": v} for n, v in c["metrics"].items()}}
                        for c in _cells(gain=lambda r: 3 * r, level=lambda r: r * r)]}
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps(record), encoding="utf-8")
    state: dict[str, Any] = {"iteration": 0, "design": {"figures_planned": ["figures/power.png", "figures/fdr.png"]}}
    # Not sent back: a repair that gave up would throw the results away and stop the quest with nothing to write up.
    result, _ = await eng._results_match_run(state, _run(), None, True)
    assert result.returncode == 0
    note = eng._run_checks_note()
    assert "the paper has no figure" in note and "as shown in Figure N" in note and "no figure was drawn" in note
    # a plan without figures asks for none
    (eng.quest_root / "needs" / "RESULT_CHECK.json").unlink()
    result, _ = await eng._results_match_run({"iteration": 5, "design": {"figures_planned": []}}, _run(), None, True)
    assert result.returncode == 0 and not (eng.quest_root / "needs" / "RESULT_CHECK.json").exists()


@pytest.mark.asyncio
async def test_a_figure_that_stays_but_has_no_point_at_some_settings_is_not_dropped_and_the_writer_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch)
    (eng.quest_root / "figures" / "empty_one.png").unlink()
    (eng.fi_dir / "figure_records" / "empty_one.json").unlink()
    (eng.fi_dir / "figure_records" / "empty_one.seed0.json").unlink()
    record = {"cells": [{"cell": c["cell"], "metrics": {n: {"values": v} for n, v in c["metrics"].items()}}
                        for c in _cells(gain=lambda r: 3 * r, level=lambda r: r * r)]}
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps(record), encoding="utf-8")
    gappy = [{"label": "uncorrected", "kind": "data", "x": [20.0], "y": [0.6]},
             {"label": "theory", "kind": "data", "x": [20.0, 200.0, 2000.0], "y": [0.6, 0.9, 1.0]}]
    _record(eng.fi_dir / "figure_records", "fine", [{"title": "Chance", "xlabel": "m", "lines": gappy}])
    result, _ = await eng._results_match_run({"iteration": 0}, _run(), None, True)
    assert result.returncode == 0, "nothing is sent back for a gap"
    assert (eng.quest_root / "figures" / "fine.png").exists()
    saved = json.loads((eng.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert saved["removed_figures"] == [] and saved["removed_quantities"] == []
    note = eng._run_checks_note()
    assert "the series \"uncorrected\" has no point at x = 200, 2000" in note and "as shown in Figure N" in note


# ---- a constant that is expected -----------------------------------------------------------------------------------------


def test_the_directive_asks_for_the_quantity_to_be_computed_and_offers_no_mark() -> None:
    q = dr.quantity_problems(_cells(gain=lambda r: 3 * r, level=2.5), REPORTED, set())
    text = dr.directive([], q)
    assert "estimated from the trials' data: compute it from this trial's data as the plan defines it" in text
    assert "constant by design" not in text and "mark the line" not in text and "or do not return it" not in text


def test_what_the_plan_expects_to_stay_put_is_exempt_when_a_check_measures_that_one_name() -> None:
    protocol = {"oracles": [
        {"name": "c", "kind": "invariant", "measure": "total_drift", "expected": 0, "case": {"rate": 1}},
        {"name": "e", "kind": "invariant", "measure": "abs(total_out - total_in) / total_in", "expected": 0, "case": {"rate": 1}},
        {"name": "l", "kind": "special_case", "measure": "gain", "expected": 1, "case": {"rate": 1}}], "limits": {"ceiling": 5}}
    names = dr.expected_constants(protocol)
    assert names == {"total_drift", "ceiling"}, "an operand of an expression is the plan's own quantity, not a constant"
    assert _flags(_cells(gain=lambda r: 3 * r, total_drift=0.0), REPORTED | {"total_drift"}, names) == []
    assert "gain" in dr.plan_estimates(protocol), "a check with no grid case measures an estimate: unchanged"


FDR_PLAN = {"grid": {"m": [20, 200, 2000], "pi1": [0.0, 0.05, 0.2]}, "in_each_run": {"method": ["uncorrected", "bh"]},
            "metrics": [{"id": "fdr_bh", "kind": "mean"}],
            "oracles": [{"name": "bh_fdr_reference", "kind": "special_case", "measure": "fdr_bh", "expected": 0.04,
                         "case": {"m": 200, "pi1": 0.2}},
                        {"name": "bh_pvector", "kind": "special_case", "measure": "bh_fixed_rejections", "expected": 3,
                         "case": {"m": 4, "pi1": 0.0}}]}


def test_a_check_on_fixed_inputs_is_flat_by_construction_and_a_data_estimate_still_is_not() -> None:
    """The real fdr plan: ``bh_fixed_rejections`` (a special case at ``m = 4``, which the study does not run; fixed p-values,
    expected 3) is 3 at every setting; ``fdr_bh`` (a metric, checked at a setting the study runs) is an estimate."""
    plan = FDR_PLAN
    reported = {"fdr_bh", "bh_fixed_rejections"}
    exempt = dr.expected_constants(plan)
    assert exempt == {"bh_fixed_rejections"} and dr.plan_estimates(plan) == {"fdr_bh"}
    varying = _cells(fdr_bh=lambda r: 0.01 * r, bh_fixed_rejections=3)
    assert dr.split_flat(dr.quantity_problems(varying, reported, exempt), plan) == ([], [])
    assert [q.name for q in dr.quantity_problems(varying, reported, set())] == ["bh_fixed_rejections"]
    planted = _cells(fdr_bh=0.04, bh_fixed_rejections=3, pany_bh=lambda r: r)
    judged, kept = dr.split_flat(dr.quantity_problems(planted, reported | {"pany_bh"}, exempt), plan)
    assert [q.name for q in judged] == ["fdr_bh"] and kept == []


def test_a_check_s_measure_is_a_data_estimate_when_its_case_is_one_of_the_study_s_settings() -> None:
    on_grid = {"grid": {"nsr": [0.01, 0.02, 0.1], "n_pts": [5, 10]}, "in_each_run": {"method": ["a", "b"]},
               "oracles": [{"name": "r", "kind": "special_case", "measure": "ea_est", "case": {"nsr": 0.1 + 1e-12, "n_pts": 10}},
                           {"name": "s", "kind": "published_value", "measure": "other_est", "case": {"method": "A", "alpha": 0.05}}]}
    assert dr.plan_estimates(on_grid) == {"ea_est", "other_est"} and dr.expected_constants(on_grid) == set()
    off = {"grid": {"m": [20, 200]}, "oracles": [{"name": "r", "kind": "published_value", "measure": "x", "case": {"m": 4}}]}
    assert dr.expected_constants(off) == {"x"} and dr.plan_estimates(off) == set()
    listed = off | {"metrics": [{"id": "x"}]}
    assert dr.expected_constants(listed) == set(), "a quantity the plan lists as its own estimate is never exempt"
    for case in (None, {}, {"alpha": 0.05}):
        none = {"grid": {"m": [20, 200]}, "oracles": [{"name": "r", "kind": "special_case", "measure": "x", "case": case}]}
        assert dr.plan_estimates(none) == {"x"} and dr.expected_constants(none) == set(), case


def test_a_planted_constant_on_a_noise_free_check_s_estimate_is_still_sent_back() -> None:
    """The r5k plan: ``ea_est`` is checked at ``nsr = 0`` (the study's noise-free end, beside 0.01 to 0.2): a data estimate."""
    plan = {"grid": {"nsr": [0.01, 0.05, 0.2], "n_pts": [5, 10]}, "metrics": [{"id": "ea_cov"}],
            "oracles": [{"name": "n", "kind": "special_case", "measure": "ea_est", "expected": 50000, "case": {"nsr": 0, "n_pts": 10}}]}
    assert dr.expected_constants(plan) == set() and "ea_est" in dr.plan_estimates(plan)
    planted = _cells(ea_est=50000.0, ea_cov=lambda r: r)
    judged, _kept = dr.split_flat(dr.quantity_problems(planted, {"ea_est", "ea_cov"}, dr.expected_constants(plan)), plan)
    assert [q.name for q in judged] == ["ea_est"]
    fixed = plan | {"grid": {"nsr": [0.01, 0.05, 0.2], "n_pts": [5, 10]}, "oracles": [
        {"name": "n", "kind": "special_case", "measure": "ea_est", "case": {"nsr": 0.5, "n_pts": 10}}]}
    assert dr.expected_constants(fixed) == {"ea_est"}, "a non-zero value the axis does not run is a fixed input"


def test_the_quantities_the_plan_says_are_estimated_from_the_data_are_read_from_fields_it_already_has() -> None:
    names = dr.plan_estimates(KINETICS_PLAN | {"metrics": KINETICS_PLAN["metrics"] + [{"id": "theory_corr_ref", "kind": "mean"}]})
    assert names == {"Ea_hat", "Ea_se", "Ea_lo", "Ea_hi", "coverage_Ea", "theory_corr_ref"}
    assert "Ea_polyfit" not in names, "an operand of a check's expression is not named by the plan as an estimate"
    assert dr.plan_estimates(None) == set() and dr.plan_estimates({}) == set()
    flat = dr.quantity_problems(_cells(Ea_hat=lambda r: r, Ea_hat_known=0.5, theory_corr=0.99, coverage_Ea=0.95),
                                {"Ea_hat", "Ea_hat_known", "theory_corr", "coverage_Ea"}, set())
    judged, kept = dr.split_flat(flat, KINETICS_PLAN)
    assert [q.name for q in judged] == ["coverage_Ea"] and sorted(q.name for q in kept) == ["Ea_hat_known", "theory_corr"]
    assert "`theory_corr` was identical (0.99) at all 8 settings" in dr.kept_note(kept) and "limitations" in dr.kept_note(kept)


# ---- only what the plan says is estimated from the data is questioned -------------------------------------------------------

#: A kinetics-shaped plan: the estimate and its coverage are named, the theory value is not.
KINETICS_PLAN = {
    "metrics": [{"id": "Ea_hat", "known_value": {"estimate": "Ea_hat", "standard_error": "Ea_se", "interval": ["Ea_lo", "Ea_hi"]}},
                {"id": "coverage_Ea"}],
    "precision": {"metric": "coverage_Ea"},
    "oracles": [
        {"name": "coverage_Ea_benign", "kind": "special_case", "measure": "coverage_Ea", "expected": 0.95},
        {"name": "polyfit_agrees", "kind": "second_implementation", "measure": "abs(Ea_hat - Ea_polyfit)", "expected": 0}],
}


def _kinetics_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flat: float) -> Engine:
    """An engine whose last run varied ``Ea_hat`` and ``Ea_bias`` with the setting and held every quantity in ``flat`` at one number."""
    eng = _engine(tmp_path, monkeypatch)
    eng._protocol_block = lambda state: KINETICS_PLAN  # type: ignore[method-assign]
    (eng.quest_root / "figures" / "empty_one.png").unlink()
    cells = _cells(Ea_hat=lambda r: 50.0 + r, Ea_bias=lambda r: 0.1 * r, **flat)
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps({"cells": [
        {"cell": c["cell"], "metrics": {n: {"values": v} for n, v in c["metrics"].items()}} for c in cells]}), encoding="utf-8")
    return eng


def _kinetics_run(*names: str) -> ExecutionResult:
    return ExecutionResult(0, "RESULT_JSON: " + json.dumps({"Ea_hat": 51.0, "Ea_bias": 0.1, **{n: 0.9 for n in names}}), "", 1.0, False)


@pytest.mark.asyncio
async def test_a_flat_quantity_the_plan_does_not_name_is_kept_and_stated_never_sent_back_or_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live s4 case: the theory correlation depends on the fixed design only, so it is the same at every setting."""
    eng = _kinetics_engine(tmp_path, monkeypatch, theory_corr=0.999074)
    result, _ = await eng._results_match_run({"iteration": 0}, _kinetics_run("theory_corr"), None, True)
    assert result.returncode == 0, "nothing is sent back for a number the plan does not say is estimated"
    assert eng._left_out_quantities() == set() and "theory_corr" in result.stdout
    assert eng._without_typed_results({"Ea_hat": 51.0, "theory_corr": 0.999074}) == {"Ea_hat": 51.0, "theory_corr": 0.999074}
    note = eng._run_checks_note()
    assert "`theory_corr`" in note and "identical (0.999074) at all 8 settings" in note and "limitations" in note


@pytest.mark.asyncio
async def test_a_flat_estimate_the_plan_names_is_still_sent_back_then_left_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _kinetics_engine(tmp_path, monkeypatch, coverage_Ea=0.95)
    for _ in (1, 2):
        result, script = await eng._results_match_run({"iteration": 0}, _kinetics_run("coverage_Ea"), None, True)
        assert result.returncode == 1 and script == "simulate.py" and "`coverage_Ea` is 0.95" in result.stderr
    result, _ = await eng._results_match_run({"iteration": 0}, _kinetics_run("coverage_Ea"), None, True)
    assert result.returncode == 0 and eng._left_out_quantities() == {"coverage_Ea"}


@pytest.mark.asyncio
async def test_a_check_measuring_an_expression_does_not_hide_the_plans_own_estimate_in_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planted ``fdr_bh = 0.05`` (the nominal level typed in) passed because ``abs(fdr_bh - 0.05)`` is a check's measure."""
    plan = {"metrics": [{"id": "fdr_bh"}],
            "oracles": [{"name": "fdr_at_level", "kind": "invariant", "measure": "abs(fdr_bh - 0.05)", "expected": 0},
                        {"name": "second_bh", "kind": "second_implementation", "measure": "bh_scipy_disagree", "expected": 0}]}
    assert "fdr_bh" not in dr.expected_constants(plan) and "bh_scipy_disagree" in dr.expected_constants(plan)
    eng = _kinetics_engine(tmp_path, monkeypatch, fdr_bh=0.05, bh_scipy_disagree=0.0)
    eng._protocol_block = lambda state: plan  # type: ignore[method-assign]
    result, script = await eng._results_match_run({"iteration": 0}, _kinetics_run("fdr_bh", "bh_scipy_disagree"), None, True)
    assert result.returncode == 1 and "`fdr_bh` is 0.05" in result.stderr and "bh_scipy_disagree" not in result.stderr


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_dropping_a_figure_removes_exactly_that_file_and_keeps_its_sibling_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng = _engine(tmp_path, monkeypatch)
    folder = eng.quest_root / "figures"
    (folder / "empty_one.svg").write_bytes(b"svg")
    (folder / "empty_one_other.png").write_bytes(b"png")
    eng._drop_figures(["empty_one.png"])
    assert not (folder / "empty_one.png").exists()
    assert (folder / "empty_one.svg").exists() and (folder / "empty_one_other.png").exists() and (folder / "fine.png").exists()
    assert (eng.fi_dir / "figure_records" / "empty_one.json").exists(), "the record stays while another format needs it"
    eng._drop_figures(["empty_one.svg"])
    assert not (eng.fi_dir / "figure_records" / "empty_one.json").exists() and (folder / "empty_one_other.png").exists()
