"""A figure is drawn from the run's results, never from numbers typed into the plotting code (core/figure_data_check.py).

The scripts are read with ``ast`` (never run). A typed-in series is sent back to be drawn from the saved results, a bounded
number of times; one that persists leaves its figure out of the paper, with a plain note. Neutral toy topics only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import figure_data_check as fdc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

TYPED = '''\
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

flows = [1, 2, 3, 4]
res = json.load(open("data/results/cooling.json"))
fig, ax = plt.subplots()
ax.plot(flows, [4, 5, 6, 8])
fig.savefig("figures/cooling.png")
print("RESULT_JSON: " + json.dumps({"n": len(res)}))
'''
FROM_RESULTS = '''\
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("data/results/cooling.json"))
fig, ax = plt.subplots()
ax.plot(res["flow"], res["drop"])
ax.axhline(5.0, linestyle="--")
ax.axvline(2.5)
ax.set_xlim(0, 5)
ax.set_xticks([1, 2, 3, 4])
ax.plot([0, 1], [0, 1], "k:")
ax.annotate("limit", xy=(1.0, 5.0))
fig.savefig("figures/cooling.png")
print("RESULT_JSON: " + json.dumps({"n": len(res)}))
'''


def _lines(src: str) -> list[int]:
    return [f.line for f in fdc.typed_series(src)]


def test_a_typed_in_series_is_caught_with_its_call_and_line() -> None:
    found = fdc.typed_series(TYPED, "experiment.py")
    assert len(found) == 1
    f = found[0]
    assert f.line == 9 and f.call == "ax.plot" and f.count == 4 and f.saved_as == {"cooling.png"}
    assert "experiment.py line 9: ax.plot() draws argument 2, 4 numbers typed into the code" in f.says()


def test_a_figure_drawn_from_results_passes_and_reference_lines_do_not_trigger() -> None:
    assert fdc.typed_series(FROM_RESULTS) == []


@pytest.mark.parametrize("source, expect", [
    ("ax.plot(xs, ys)", False),
    ("ax.plot(xs, [1, 2, 3])", True),
    ("ax.bar([0, 1, 2], heights)", False),                      # the settings along the axis, the numbers from the run
    ("ax.bar([0, 1, 2], [3, 4, 5])", True),
    ("ax.bar(labels, height=[3, 4, 5])", True),
    ("ax.scatter(res.x, np.array([1.0, 2.0, 3.0, 4.0]))", True),
    ("ax.errorbar(x, y, yerr=[0.1, 0.1, 0.1])", True),
    ("ax.errorbar(x, y, yerr=0.1)", False),
    ("ax.plot([0, 1], [3, 4])", False),                          # two points: a reference line
    ("ax.plot([0, 1, 2], [0, 1, 2], 'k--')", False),             # y = x
    ("ax.plot(x, y, color=[0.1, 0.2, 0.3])", False),             # a colour is no data
    ("ax.plot(x, y, linewidth=2); ax.set_ylim([0, 1, 2])", False),
    ("ax.hist(values, bins=[0, 1, 2, 3])", False),
    ("ax.imshow([[1, 2, 3], [4, 5, 6]])", True),
    ("ax.axhline(5); ax.axvline(2); ax.text(1, 2, 'a')", False),
    ("plt.pie([30, 40, 30])", True),
])
def test_what_counts_as_a_typed_in_series(source: str, expect: bool) -> None:
    assert bool(fdc.typed_series(source)) is expect, source


def test_a_name_bound_only_to_a_literal_is_typed_but_one_that_is_read_or_rebound_is_not() -> None:
    assert _lines("ys = [4, 5, 6, 8]\nax.plot(x, ys)\n")
    assert _lines("def f(x):\n    ys = [4, 5, 6, 8]\n    ax.plot(x, ys)\n")
    assert not _lines("ys = [4, 5, 6, 8]\nys = load()\nax.plot(x, ys)\n")
    assert not _lines("ys = [4, 5, 6, 8]\nys += extra\nax.plot(x, ys)\n")
    assert not _lines("def f(x, ys):\n    ax.plot(x, ys)\n")
    assert not _lines("ys = [r['t'] for r in rows]\nax.plot(x, ys)\n")
    assert not _lines("ys = [4, 5, k]\nax.plot(x, ys)\n")
    assert not _lines("for ys in ([1, 2, 3],):\n    ax.plot(x, ys)\n")


def test_a_script_that_does_not_parse_reports_nothing() -> None:
    assert fdc.typed_series("ax.plot(x, [1, 2, 3") == []


def test_the_figure_a_call_is_saved_into_is_named_and_otherwise_every_figure_is() -> None:
    src = ('fig, ax = plt.subplots()\nax.plot(x, [1, 2, 3])\nfig.savefig("figures/a.png")\n'
           'fig2, ax2 = plt.subplots()\nax2.plot(x, y)\nfig2.savefig("figures/b.png")\n')
    found = fdc.typed_series(src)
    assert fdc.figures_to_drop(found, ["a.png", "b.png"]) == ["a.png"]
    unnamed = fdc.typed_series('ax.plot(x, [1, 2, 3])\nfig.savefig(path)\n')
    assert fdc.figures_to_drop(unnamed, ["a.png", "b.png"]) == ["a.png", "b.png"]
    assert fdc.figures_to_drop([], ["a.png"]) == []


# --- the engine's use of it ----------------------------------------------------------------------------------------


class _Client:
    """Answers each chat call with the next reply (the last one again when they run out) and remembers the prompts."""

    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic about a cooling cup", title="typed", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text(TYPED, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_typed_in_series_is_sent_back_and_a_repair_that_uses_the_results_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": FROM_RESULTS, "patch_summary": "drew from results"})])
    used, rewrote = await eng._figures_from_results({})
    assert (used, rewrote) == (1, True) and len(asked) == 1
    # The request names the call and the line, and says where the numbers should come from.
    assert "experiment.py line 9: ax.plot() draws argument 2" in asked[0] and "saved results" in asked[0]
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == FROM_RESULTS
    # Nothing is left to drop: no record, the figures stay.
    figs = eng.quest_root / "figures"
    figs.mkdir()
    (figs / "cooling.png").write_bytes(b"x")
    assert eng._drop_typed_figures(["cooling.png"], eng.fi_dir / "figure_records") == ["cooling.png"]
    assert (figs / "cooling.png").is_file() and eng._typed_figures_note() == ""


@pytest.mark.asyncio
async def test_a_typed_in_series_that_persists_leaves_its_figure_out_with_a_plain_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = TYPED.replace('fig.savefig("figures/cooling.png")', 'fig.savefig("figures/cooling.png")\n'
                          'fig2, ax2 = plt.subplots()\nax2.plot(res["x"], res["y"])\nfig2.savefig("figures/other.png")')
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": other, "patch_summary": "no change"})])
    (eng.quest_root / "code" / "experiment.py").write_text(other, encoding="utf-8")
    used, rewrote = await eng._figures_from_results({})
    assert used == Engine._FIGURE_DATA_REPAIRS == len(asked) and rewrote is False, "asked twice, kept as written"
    figs = eng.quest_root / "figures"
    figs.mkdir()
    records = eng.fi_dir / "figure_records"
    records.mkdir(parents=True)
    for name in ("cooling.png", "cooling.pdf", "other.png"):
        (figs / name).write_bytes(b"x")
    (records / "cooling.json").write_text("{}", encoding="utf-8")
    kept = eng._drop_typed_figures(["cooling.pdf", "cooling.png", "other.png"], records)
    assert kept == ["other.png"], "only the figure that call is saved into (and its other formats) is left out"
    assert not (figs / "cooling.png").exists() and not (figs / "cooling.pdf").exists() and not (records / "cooling.json").exists()
    assert (figs / "other.png").is_file()
    record = json.loads((eng.quest_root / "needs" / "FIGURE_DATA_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_figures"] == ["cooling.pdf", "cooling.png"] and "typed into the code" in record["note"]
    note = eng._typed_figures_note()
    assert "left out" in note and "limitations" in note and "figures/cooling.png" in note
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "The paper does not use them and says so in its limitations" in log
    # The next pass's own record replaces it: a pass with nothing typed in leaves none.
    (eng.quest_root / "code" / "experiment.py").write_text(FROM_RESULTS, encoding="utf-8")
    await eng._figures_from_results({"figure_data_repairs": 2})
    assert eng._drop_typed_figures(["other.png"], records) == ["other.png"]
    assert not (eng.quest_root / "needs" / "FIGURE_DATA_CHECK.json").exists() and eng._typed_figures_note() == ""


@pytest.mark.asyncio
async def test_a_repair_that_is_not_the_same_script_is_not_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for reply in ("{}", json.dumps({"code": "print('hello')\n"}), json.dumps({"code": "def broken(:\n"})):
        eng, asked = _engine(tmp_path / reply[:3].strip("{}\"") , monkeypatch, [reply])
        used, rewrote = await eng._figures_from_results({"figure_data_repairs": 1})
        assert rewrote is False and len(asked) == 1
        assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == TYPED, reply


@pytest.mark.asyncio
async def test_no_call_is_made_for_a_script_with_no_typed_in_series_or_no_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, ["{}"])
    (eng.quest_root / "code" / "experiment.py").write_text(FROM_RESULTS, encoding="utf-8")
    assert await eng._figures_from_results({}) == (0, False) and asked == []
    (eng.quest_root / "code" / "experiment.py").write_text(TYPED, encoding="utf-8")
    assert await eng._figures_from_results({"figure_data_repairs": 2}) == (2, False) and asked == []
    # A script that saves no figure is not a figure script.
    (eng.quest_root / "code" / "experiment.py").write_text(TYPED.replace('fig.savefig("figures/cooling.png")', ""), encoding="utf-8")
    assert await eng._figures_from_results({}) == (0, False)


def test_the_note_reaches_the_writers_prompt_only_when_a_figure_was_left_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, _ = _engine(tmp_path, monkeypatch, ["{}"])
    assert eng._typed_figures_note() == ""
    (eng.quest_root / "needs").mkdir(exist_ok=True)
    (eng.quest_root / "needs" / "FIGURE_DATA_CHECK.json").write_text(
        json.dumps({"removed_figures": ["a.png"], "note": "figures/a.png was left out: it used typed numbers."}), encoding="utf-8")
    assert "figures/a.png was left out" in eng._typed_figures_note()


def test_the_implement_prompt_says_to_draw_from_results() -> None:
    text = (Path(__file__).resolve().parent.parent / "agents" / "implement.md").read_text(encoding="utf-8")
    assert "never from numbers typed into the plotting code" in text
