"""A line figure of a multi-seed experiment is drawn again as the mean of the
seeds, shaded with its 95% confidence interval, and the checks that compare the
paper with the results accept the mean and its interval (``core/plot_style.py``
records each seed's lines, ``core/engine.py`` computes the means,
``core/replot_figures.py`` draws them)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine, _figure_list_for_prompt, _figure_record_note, _format_figure_check,
    _replicate_line_figure, _replicate_result_intervals,
)
from core.plot_style import write_boot


def _boot_env(tmp_path: Path, records: Path, **extra: str) -> dict[str, str]:
    boot_dir = write_boot(tmp_path / ".boot", "latex")
    return {
        **{k: v for k, v in os.environ.items()
           if k not in ("FI_REPLICATE_SEED", "FI_REPLICATE_INDEX", "FI_REPLOT")},
        "PYTHONPATH": os.pathsep.join(p for p in (str(boot_dir), os.environ.get("PYTHONPATH", "")) if p),
        "FI_FIGURE_RECORDS": str(records),
        **extra,
    }


PLOT = (
    "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
    "fig, (lines, bars) = plt.subplots(1, 2, figsize=(9, 4))\n"
    "lines.plot([1, 10, 100], [0.2, 0.5, float('nan')], marker='o', linestyle='--', label='R0 = 1.5')\n"
    "lines.axhline(0.5, color='k')\n"
    "lines.set_xscale('log'); lines.set_ylabel('Outbreak probability'); lines.legend()\n"
    "bars.hist([1, 2, 2, 3], label='sizes')\n"
    "bars.plot([1, 3], [1, 1], label='mean')\n"
    "fig.savefig('fig.png')\n"
)


# --- the recorder --------------------------------------------------------------

def test_the_recorder_keeps_each_seeds_lines(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = tmp_path / "records"
    out = subprocess.run([sys.executable, "-c", PLOT], env=_boot_env(tmp_path, records, FI_REPLICATE_INDEX="2"),
                         cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in records.iterdir()) == ["fig.json", "fig.seed2.json"]
    data = json.loads((records / "fig.seed2.json").read_text(encoding="utf-8"))
    assert data["file"] == "fig.png" and data["size"] == [9.0, 4.0]
    lines, bars = data["axes"]
    assert lines["grid"] == [1, 2, 0, 1, 0, 1] and bars["grid"] == [1, 2, 0, 1, 1, 2]
    assert lines["line_only"] is True and lines["xscale"] == "log" and lines["legend"] is True
    curve, level = lines["lines"]
    assert curve["label"] == "R0 = 1.5" and curve["kind"] == "data"
    # The point with no y goes with its x, so the two stay paired.
    assert curve["x"] == [1.0, 10.0] and curve["y"] == [0.2, 0.5]
    assert curve["marker"] == "o" and curve["linestyle"] == "--"
    assert level["kind"] == "hline" and level["color"] == "#000000"
    # A line drawn over bars does not make the panel a line figure.
    assert [line["label"] for line in bars["lines"]] == ["mean"] and bars["line_only"] is False


EMPTY_BAND = (
    "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
    "from matplotlib.collections import PolyCollection\n"
    "fig, (empty, band) = plt.subplots(1, 2)\n"
    "empty.plot([1, 2, 3], [0.2, 0.4, 0.6], marker='o', label='N = 100')\n"
    "empty.fill_between([], [], [], alpha=0.2)\n"
    "empty.add_collection(PolyCollection([]))\n"
    "empty.legend()\n"
    "band.plot([1, 2, 3], [0.2, 0.4, 0.6], label='N = 100')\n"
    "band.fill_between([1, 2, 3], [0.1, 0.3, 0.5], [0.3, 0.5, 0.7], alpha=0.2)\n"
    "print('PATHS', [len(c.get_paths()) for c in empty.collections], [len(c.get_paths()) for c in band.collections])\n"
    "fig.savefig('fig.png')\n"
)


def test_a_band_that_draws_nothing_leaves_a_line_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = tmp_path / "records"
    out = subprocess.run([sys.executable, "-c", EMPTY_BAND], env=_boot_env(tmp_path, records),
                         cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    # seaborn leaves an empty band like these per line when each x holds one observation.
    assert "PATHS [0, 0] [1]" in out.stdout
    empty, band = json.loads((records / "fig.seed0.json").read_text(encoding="utf-8"))["axes"]
    assert empty["line_only"] is True
    # A band that is drawn still makes the panel more than lines.
    assert band["line_only"] is False


def test_a_redrawn_figure_keeps_no_seed_lines(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = tmp_path / "records"
    out = subprocess.run([sys.executable, "-c", PLOT], env=_boot_env(tmp_path, records, FI_REPLOT="1"),
                         cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in records.iterdir()) == ["fig.json"]


ERRORBARS = (
    "import os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
    "k = int(os.environ.get('FI_REPLICATE_SEED', 0))\n"
    "fig, ax = plt.subplots()\n"
    "ax.errorbar([1, 2, 3], [0.2 + k / 10, 0.4, 0.6], yerr=0.05, fmt='o-', capsize=4, label='N = 100')\n"
    "ax.axhline(0.5, color='k', label='limit'); ax.set_ylabel('Outbreak probability'); ax.legend()\n"
    "fig.savefig('err.png')\n"
    "fig, axes = plt.subplots(1, 4, figsize=(12, 3))\n"
    "axes[0].errorbar([1, 2, 3], [0.2 + k / 10, 0.4, 0.6], yerr=0.05, fmt='o-', capsize=4)\n"
    "axes[1].errorbar([1, 2, 3], [0.2, 0.4, 0.6], yerr=0.05, fmt='none')\n"
    "axes[2].errorbar([1, 2, 3], [0.2, 0.4, 0.6], xerr=0.1, fmt='o-')\n"
    "axes[3].hist([1, 2, 2, 3 + k])\n"
    "fig.savefig('mixed.png')\n"
)


def _errorbar_records(tmp_path: Path, seeds: tuple[int, ...]) -> Path:
    records = tmp_path / "records"
    for seed in seeds:
        out = subprocess.run([sys.executable, "-c", ERRORBARS],
                             env=_boot_env(tmp_path, records, FI_REPLICATE_SEED=str(seed),
                                           FI_REPLICATE_INDEX=str(seed)),
                             cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
    return records


def test_a_line_with_error_bars_is_recorded_as_a_line(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = _errorbar_records(tmp_path, (0,))
    (panel,) = json.loads((records / "err.seed0.json").read_text(encoding="utf-8"))["axes"]
    assert panel["line_only"] is True
    # The data line, named by its errorbar call; its caps and bars are not series.
    series, limit = panel["lines"]
    assert series["label"] == "N = 100" and series["marker"] == "o" and series["kind"] == "data"
    assert series["x"] == [1.0, 2.0, 3.0] and series["y"] == pytest.approx([0.2, 0.4, 0.6])
    assert limit["label"] == "limit" and limit["kind"] == "hline"
    (shown,) = json.loads((records / "err.json").read_text(encoding="utf-8"))["axes"]
    assert [s["label"] for s in shown["series"]] == ["N = 100", "limit"]
    # Bars with no line, bars across x, and a histogram are more than lines.
    mixed = json.loads((records / "mixed.seed0.json").read_text(encoding="utf-8"))["axes"]
    assert [p["line_only"] for p in mixed] == [True, False, False, False]


def test_a_figure_of_lines_with_error_bars_is_planned_as_their_mean(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = _errorbar_records(tmp_path, (0, 1, 2))

    def runs(stem: str) -> list[Any]:
        return [json.loads((records / f"{stem}.seed{k}.json").read_text(encoding="utf-8")) for k in (0, 1, 2)]

    plan = _replicate_line_figure("err.png", runs("err"), [])
    assert plan is not None and plan["n"] == 3
    series, limit = plan["axes"][0]["lines"]
    assert series["label"] == "N = 100" and limit["kind"] == "hline"
    assert series["mean"] == pytest.approx([0.3, 0.4, 0.6])
    half = T95_2 * 0.1 / 3 ** 0.5
    assert series["lower"][0] == pytest.approx(0.3 - half, abs=1e-3)
    assert series["upper"][0] == pytest.approx(0.3 + half, abs=1e-3)
    # Where every seed agrees there is no band: one run's error bars are gone.
    assert series["lower"][1:] == pytest.approx([0.4, 0.6]) and series["upper"][1:] == pytest.approx([0.4, 0.6])
    # A histogram beside them still keeps the figure from being redrawn.
    assert _replicate_line_figure("mixed.png", runs("mixed"), []) is None


# --- the means -----------------------------------------------------------------

def _run(
    ys: list[float], *, x: tuple[float, ...] = (1.0, 10.0, 100.0), label: str = "R0 = 1.5",
    line_only: bool = True, ylabel: str = "Outbreak probability",
    grid: tuple[int, ...] = (1, 1, 0, 1, 0, 1),
) -> dict[str, Any]:
    return {"file": "p.png", "size": [8.0, 6.0], "suptitle": "", "axes": [{
        "grid": list(grid), "title": "Outbreak probability", "xlabel": "N", "ylabel": ylabel,
        "xscale": "log", "yscale": "linear", "legend": True, "line_only": line_only,
        "lines": [{
            "label": label, "kind": "data", "x": list(x), "y": list(ys), "color": "#0e6e6b",
            "linestyle": "-", "marker": "o", "markersize": 6.0, "markerfacecolor": "#0e6e6b",
            "linewidth": 2.2, "alpha": None, "drawstyle": "default",
        }],
    }]}


T95_2 = 4.303  # the t quantile for two degrees of freedom


def test_each_line_is_drawn_at_its_mean_within_its_interval() -> None:
    runs = [_run([0.10, 0.40, 0.70]), _run([0.20, 0.46, 0.70]), _run([0.30, 0.43, 0.70])]
    plan = _replicate_line_figure("p.png", runs, [])
    assert plan is not None
    assert plan["file"] == "p.png" and plan["n"] == 3 and plan["size"] == [8.0, 6.0]
    (panel,) = plan["axes"]
    assert panel["xscale"] == "log" and panel["legend"] is True and panel["grid"] == [1, 1, 0, 1, 0, 1]
    (line,) = panel["lines"]
    assert line["x"] == [1.0, 10.0, 100.0] and line["marker"] == "o" and line["color"] == "#0e6e6b"
    assert line["mean"] == pytest.approx([0.2, 0.43, 0.7])
    half = T95_2 * 0.1 / 3 ** 0.5
    # A probability's interval stops at 0.
    assert line["lower"][0] == 0.0 and line["upper"][0] == pytest.approx(0.2 + half, abs=1e-3)
    # Every seed drew the same point: no band there.
    assert line["lower"][2] == pytest.approx(0.7) and line["upper"][2] == pytest.approx(0.7)


def test_an_interval_of_a_quantity_that_is_not_a_probability_is_not_cut_at_zero() -> None:
    runs = [_run([0.10, 1, 2], ylabel="Cases"), _run([0.20, 1, 2], ylabel="Cases"), _run([0.30, 1, 3], ylabel="Cases")]
    plan = _replicate_line_figure("p.png", runs, [])
    assert plan is not None
    assert plan["axes"][0]["lines"][0]["lower"][0] == pytest.approx(0.2 - T95_2 * 0.1 / 3 ** 0.5, abs=1e-3)


@pytest.mark.parametrize("runs", [
    pytest.param([_run([1, 2, 3]), _run([1, 2, 4], x=(1.0, 10.0, 1000.0))], id="x differs"),
    pytest.param([_run([1, 2, 3]), _run([1, 2, 4], label="R0 = 3")], id="label differs"),
    pytest.param([_run([1, 2, 3]), _run([1, 2, 4], line_only=False)], id="not only lines"),
    pytest.param([_run([1, 2, 3]), _run([1, 2, 3])], id="nothing varies"),
    pytest.param([_run([1, 2, 3]), None], id="a seed drew no record"),
    pytest.param([_run([1, 2, 3])], id="one seed"),
    pytest.param([_run([1, 2, 3], grid=(1, 2, 0, 1, 0, 1)), _run([1, 2, 4])], id="panel moved"),
])
def test_a_figure_that_is_not_the_same_lines_at_every_seed_is_left_as_it_is(runs: list[Any]) -> None:
    assert _replicate_line_figure("p.png", runs, []) is None


def test_the_writer_is_told_a_figure_is_the_mean_of_the_seeds() -> None:
    record = {
        "file": "p.png", "replicate_mean": {"n": 3},
        "axes": [{"title": "T", "ylabel": "y", "yscale": "linear", "ylim": [0, 1],
                  "series": [{"label": "a", "min": 0.2, "max": 0.7, "shows": "yes"}]}],
    }
    listing = _figure_list_for_prompt({"figures": ["p.png"], "figure_records": {"p.png": record}})  # type: ignore[typeddict-item]
    first, note = listing.splitlines()
    assert first.endswith("— each line is the mean of 3 seeds, shaded with its 95% confidence interval")
    assert note.endswith("and its caption says so.")


def _seed_0_state(replicates: int | None) -> dict[str, Any]:
    """A histogram the seeds could not redraw, beside ``replicates`` seeds' results."""
    record = {
        "file": "sizes.png", "single_seed": 0,
        "axes": [{"title": "Final sizes", "ylabel": "Count", "yscale": "linear", "ylim": [0, 12], "series": []}],
    }
    state: dict[str, Any] = {"figures": ["sizes.png"], "figure_records": {"sizes.png": record}}
    if replicates is not None:
        state["result_json_replicates"] = [{"_seed": k, "m": 0.1 * k} for k in range(replicates)]
    return state


def test_the_writer_and_the_review_are_told_a_figure_shows_seed_0() -> None:
    state = _seed_0_state(None)
    first, note = _figure_list_for_prompt(state).splitlines()  # type: ignore[arg-type]
    assert first.endswith(
        "— shows replicate seed 0 only (every run that seed made), not the mean over the replicate seeds"
    )
    assert note == (
        "A figure that shows replicate seed 0 only could not be drawn as the mean over the replicate "
        "seeds. A replicate seed repeats the whole experiment, so the figure draws every run seed 0 "
        "made: the text and caption about it quote seed 0's values, or say it shows a single "
        "replicate (seed 0), not the means over the seeds."
    )
    check = _format_figure_check("# P\n", state).splitlines()  # type: ignore[arg-type]
    assert check == [
        "figures/sizes.png shows replicate seed 0 only (every run that seed made), not the mean over "
        "the replicate seeds: its caption and the text about it must quote seed 0's values or say it "
        "shows a single replicate (seed 0).",
        "No caption names a series its figure does not show.",
    ]


def test_with_the_seeds_counted_a_figure_shows_seed_0_of_them() -> None:
    state = _seed_0_state(3)
    first, note = _figure_list_for_prompt(state).splitlines()  # type: ignore[arg-type]
    assert first.endswith(
        "— shows replicate seed 0 of 3 only (every run that seed made), not the mean over the 3 replicate seeds"
    )
    assert note.startswith("A figure that shows replicate seed 0 only could not be drawn as the mean over "
                           "the 3 replicate seeds.")
    assert "say it shows a single replicate (seed 0 of 3), not the means over the seeds." in note
    assert _format_figure_check("# P\n", state).splitlines()[0] == (  # type: ignore[arg-type]
        "figures/sizes.png shows replicate seed 0 of 3 only (every run that seed made), not the mean "
        "over the 3 replicate seeds: its caption and the text about it must quote seed 0's values or "
        "say it shows a single replicate (seed 0 of 3)."
    )
    # One seed's results say nothing about how many seeds there were.
    assert "seed 0 of" not in _figure_list_for_prompt(_seed_0_state(1))  # type: ignore[arg-type]


# A seed repeats the whole experiment -- one seed of an SIR study is 300
# simulations -- so a figure of seed 0 is never one run or a single simulation.
_ONE_RUN_RE = re.compile(r"\b(?:one|single)\s+(?:\w+\s+)?(?:run|simulation)s?\b", re.IGNORECASE)


@pytest.mark.parametrize("replicates", [None, 3], ids=["seeds not counted", "3 seeds"])
def test_a_figure_of_seed_0_is_never_called_one_run(replicates: int | None) -> None:
    state = _seed_0_state(replicates)
    record = state["figure_records"]["sizes.png"]
    texts = {
        "record note": _figure_record_note(record, n_seeds=replicates),
        "record note, seeds not passed": _figure_record_note(record),
        "writer figure list": _figure_list_for_prompt(state),  # type: ignore[arg-type]
        "figure check": _format_figure_check("# P\n", state),  # type: ignore[arg-type]
    }
    for where, text in texts.items():
        assert "replicate seed 0" in text, where
        assert "one run" not in text.lower() and "single run" not in text.lower(), where
        assert not _ONE_RUN_RE.search(text), (where, text)
    counted = ("seed 0 of 3" in texts[k] for k in ("record note", "writer figure list", "figure check"))
    assert all(counted) if replicates else not any("seed 0 of" in t for t in texts.values())


# --- the execute node, end to end ------------------------------------------------

EXPERIMENT = """\
import json, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(int(os.environ.get("FI_REPLICATE_SEED", 0)))
xs = [100, 1000, 5000]
prob = [0.3 + 0.1 * random.random() for _ in xs]
os.makedirs("figures", exist_ok=True)
plt.figure()
plt.plot(xs, prob, marker="o", label="R0 = 1.5")
plt.xscale("log"); plt.ylabel("Outbreak probability"); plt.legend()
plt.savefig("figures/outbreak.png"); plt.close()
plt.figure()
plt.errorbar(xs, [p + 0.2 for p in prob], yerr=0.05, fmt="s-", capsize=4, label="R0 = 3")
plt.xscale("log"); plt.ylabel("Outbreak probability"); plt.legend()
plt.savefig("figures/errors.png"); plt.close()
plt.figure()
plt.hist([random.random() for _ in range(50)], bins=5)
plt.savefig("figures/sizes.png"); plt.close()
print("RESULT_JSON: " + json.dumps({"mean_prob": sum(prob) / len(prob)}))
"""


@pytest.mark.asyncio
async def test_execute_draws_the_line_figure_as_the_mean_of_the_seeds(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    eng = Engine(Config(
        topic="replot", title="replot", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=3, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    scripts: list[str] = []
    histograms: dict[str, bytes] = {}

    async def run(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str] | None = None) -> Any:
        scripts.append(Path(cmd[-1]).name)
        start = time.monotonic()
        done = subprocess.run([sys.executable, cmd[-1]], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=timeout_s)
        if scripts[-1] == "experiment.py":
            seed = (env or {}).get("FI_REPLICATE_INDEX", "0")
            histograms[f"png{seed}"] = (cwd / "figures" / "sizes.png").read_bytes()
            histograms[f"json{seed}"] = (eng.fi_dir / "figure_records" / "sizes.json").read_bytes()
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr,
                               duration_s=time.monotonic() - start, timed_out=False)

    eng.executor.execute = run  # type: ignore[method-assign]
    eng.executor.install = AsyncMock(return_value=SimpleNamespace(returncode=0, stderr=""))  # type: ignore[method-assign]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text(EXPERIMENT, encoding="utf-8")

    patch = await eng._node_execute({"deps": []})  # type: ignore[arg-type]

    assert scripts == ["experiment.py"] * 3 + ["replot_figures.py"]
    assert len(patch["result_json_replicates"]) == 3
    records = patch["figure_records"]
    assert records["outbreak.png"]["replicate_mean"] == {"n": 3}
    assert "replicate_mean" not in records["sizes.png"]
    # The line with error bars is redrawn too, named by its errorbar call.
    assert records["errors.png"]["replicate_mean"] == {"n": 3}
    assert [s["label"] for s in records["errors.png"]["axes"][0]["series"]] == ["R0 = 3"]
    assert "single_seed" not in records["outbreak.png"] and "single_seed" not in records["errors.png"]
    # The histogram goes back to seed 0's file and record, and says it shows seed 0 alone.
    assert histograms["png0"] != histograms["png2"] and histograms["json0"] != histograms["json2"]
    assert (eng.quest_root / "figures" / "sizes.png").read_bytes() == histograms["png0"]
    assert records["sizes.png"] == {**json.loads(histograms["json0"]), "single_seed": 0}
    # The figure on disk now draws the mean of the three seeds' lines.
    seed_lines = [
        json.loads((eng.fi_dir / "figure_records" / f"outbreak.seed{k}.json").read_text(encoding="utf-8"))
        ["axes"][0]["lines"][0]["y"]
        for k in range(3)
    ]
    mean = [sum(ys) / 3 for ys in zip(*seed_lines)]
    (series,) = records["outbreak.png"]["axes"][0]["series"]
    assert series["label"] == "R0 = 1.5"
    assert (series["min"], series["max"]) == pytest.approx((min(mean), max(mean)))
    plans = {
        f["file"]: f
        for f in json.loads((eng.quest_root / "code" / "replot_figures.json").read_text(encoding="utf-8"))["figures"]
    }
    assert sorted(plans) == ["errors.png", "outbreak.png"] and plans["outbreak.png"]["n"] == 3
    (bars,) = plans["errors.png"]["axes"][0]["lines"]
    assert bars["label"] == "R0 = 3" and bars["marker"] == "s"
    assert all(lo < m < hi for lo, m, hi in zip(bars["lower"], bars["mean"], bars["upper"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(("values", "runs", "one_run"), [
    pytest.param([0.1, 0.2, 0.3], 3, True, id="the seeds differ"),
    pytest.param([0.1, 0.2, None], 3, True, id="the last replicate crashed after drawing"),
    pytest.param([0.1, 0.1, 0.1], 2, False, id="every seed agrees"),
])
async def test_a_figure_the_seeds_do_not_redraw_shows_seed_0(
    tmp_path: Path, values: list[float | None], runs: int, one_run: bool,
) -> None:
    eng = Engine(Config(
        topic="replot", title="replot", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=3, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    seeds: list[int] = []

    async def run(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str] | None = None) -> Any:
        # The ORDINAL of this replicate, not the seed it draws from: the seeds
        # now stride far apart so no two runs can share a draw.
        seed = int((env or {}).get("FI_REPLICATE_INDEX", 0))
        seeds.append(seed)
        (cwd / "figures").mkdir(exist_ok=True)
        (cwd / "figures" / "sizes.png").write_bytes(f"histogram of seed {seed}".encode())
        records = Path((env or {})["FI_FIGURE_RECORDS"])
        records.mkdir(parents=True, exist_ok=True)
        (records / "sizes.json").write_text(json.dumps({"file": "sizes.png", "axes": [], "seed": seed}), encoding="utf-8")
        value = values[seed]
        return SimpleNamespace(
            returncode=1 if value is None else 0, stderr="", duration_s=0.1, timed_out=False,
            stdout="" if value is None else "RESULT_JSON: " + json.dumps({"m": value}),
        )

    eng.executor.execute = run  # type: ignore[method-assign]
    eng.executor.install = AsyncMock(return_value=SimpleNamespace(returncode=0, stderr=""))  # type: ignore[method-assign]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    # Reads the seed, so "every seed agrees" means the experiment is
    # deterministic rather than unable to see the seed at all.
    (eng.quest_root / "code" / "experiment.py").write_text(
        'import os\nseed = int(os.environ.get("FI_REPLICATE_SEED", 0))\n', encoding="utf-8")

    patch = await eng._node_execute({"deps": []})  # type: ignore[arg-type]

    assert seeds == list(range(runs))
    assert (eng.quest_root / "figures" / "sizes.png").read_bytes() == b"histogram of seed 0"
    record = patch["figure_records"]["sizes.png"]
    assert record["seed"] == 0
    # Only beside means over the seeds does a figure need to say it shows seed 0 alone.
    assert record.get("single_seed") == (0 if one_run else None)


# --- the checks ----------------------------------------------------------------

SEEDS = {
    "result_json": {"p_outbreak": 0.40},
    "result_json_replicates": [
        {"_seed": 0, "p_outbreak": 0.40}, {"_seed": 1, "p_outbreak": 0.46}, {"_seed": 2, "p_outbreak": 0.43},
    ],
}


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    return eng


def test_the_number_check_accepts_the_mean_over_the_seeds_and_its_interval(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    stats = _replicate_result_intervals(SEEDS)  # type: ignore[arg-type]
    assert list(stats) == ["p_outbreak"]
    s = stats["p_outbreak"]
    paper = (
        "# P\n\n## Results\n\nThe outbreak probability was "
        f"{s['mean']:.2f} (95% CI {s['ci_lower']:.3f} to {s['ci_upper']:.3f}).\n"
    )
    assert eng._numeric_oracle_hits(paper, {"result_json": SEEDS["result_json"]})  # type: ignore[arg-type]
    assert eng._numeric_oracle_hits(paper, SEEDS) == []  # type: ignore[arg-type]


def test_the_claim_check_sees_the_mean_over_the_seeds(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen["prompt"] = prompt
        return json.dumps({"claims": [], "summary": ""})

    eng._chat = chat  # type: ignore[method-assign]
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text("# P\n\nThe outbreak probability was 0.43.\n", encoding="utf-8")
    asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(paper), **SEEDS}))  # type: ignore[arg-type]
    s = _replicate_result_intervals(SEEDS)["p_outbreak"]  # type: ignore[arg-type]
    means = (
        "Mean over the 3 seeds, with its 95% CI:\n"
        f"- p_outbreak: {s['mean']:.4g} (95% CI {s['ci_lower']:.4g} to {s['ci_upper']:.4g})"
    )
    assert means in seen["prompt"]
    # After the results, which keep their own share of the evidence budget.
    # The results are now written under their own heading instead of as a raw
    # ``"result_json"`` key, so locate them by that heading. What this test
    # protects is the ORDER: the seed means read as a note on the results and
    # must not come before them.
    results_heading = "The run's results (result_json)"
    assert results_heading in seen["prompt"]
    assert seen["prompt"].index(results_heading) < seen["prompt"].index(means)
