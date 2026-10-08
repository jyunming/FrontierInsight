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
        {"title": "A horizontal level", "lines": [_line([1.0, 9.0, 4.0], [2.0, 2.0, 2.0], kind="hline")]},
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


# ---- the words ------------------------------------------------------------------------------------------------------------


def test_the_request_names_each_thing_and_ends_with_the_instruction() -> None:
    q = dr.quantity_problems(_cells(gain=lambda r: 3 * r, level=2.5), REPORTED, set())
    f = [dr.FigureProblem("a.png", "Trend", "empty")]
    text = dr.directive(f, q)
    assert text.startswith("FI looked at what the run produced, and some of it shows nothing:")
    assert "`level` is 2.5" in text and "`a.png`, panel \"Trend\" draws no data" in text
    assert "compute each quantity named above from them, with no fixed number and no clamp" in text
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
