"""The nine seeded faults of the figure-claims design (section 11.2), each planted in a known-good figure and run through the
merged code path: the drawing calls (``core/frame_figure.py``, ``core/analysis_figures.py``), the call record they write
(``<stem>.calls.json``), the claims and the menu (``core/figure_claims.py``), the caption, and the write node's pointer rule and
take-out with its disclosure. Nothing is mocked but the model: a reply of the writer is a fixed text. Made-up study: a colony whose
``growth`` is read at settings of ``size``.

A fault is *caught* by the condition of design 11.2 (named in each test). A fault the merged code does not catch is a finding, not
something these tests are loosened for: its test is ``xfail(strict=True)`` with the reason, so the day it is caught it fails loudly."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from core import figure_claims, figure_facts, plot_style, skeleton  # noqa: E402
from core.engine import _figure_caption_text, _figure_overlap_findings, _place_figures  # noqa: E402
from tests.test_analysis_figures import _CallTable, _calls_run, frame5  # noqa: E402,F401  (frame5 is a fixture)
from tests.test_fi_places_figures import DISCLOSURE, _paper, _write  # noqa: E402
from tests.test_frame_figure import _Host  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def frame_record(root: Path) -> dict:
    from core import analysis_frame

    return analysis_frame.read_record(root)


def _calls(root: Path, stem: str = "g") -> dict:
    calls = figure_claims.load_calls(root, stem)
    assert calls is not None, "the call record of the figure was not written"
    return calls


def _drawn_record(tmp_path: Path) -> dict:
    """The call record of a figure drawn through the real drawing calls (``run_figures``, the ``frame5`` fixture): one panel, the
    series ``fast``."""
    def body(fig, table):
        fig.line(0, "fast", [1, 2], [table.mean("growth", "size=1"), table.mean("growth", "size=2")], "growth")

    assert _calls_run(tmp_path, body, panels=1)["status"] == "ok"
    return _calls(tmp_path)


def _offered(calls: dict, plan: dict | None = None) -> list[tuple[str, dict]]:
    return [(o["claim"], o["args"]) for o in figure_claims.menu(calls, plan, stem="g")]


# ---- 1. a body reads an id the plan does not declare ----------------------------------------------------------------------------

UNDECLARED = "G_obs/G0 - 1"


def test_mutation_1_undeclared_read_gets_default_figure(tmp_path: Path, monkeypatch) -> None:
    """Caught: the dry run records a warning that names the id; and one connected chain: the repairs are spent, the engine's last
    run draws the figure the body lost from the plan's own description (a body that read an undeclared id), that figure's record
    is the engine's own, and the paper has the figure under a number with its caption."""
    from core.engine import _read_figure_records  # noqa: F401  (the engine reads the records it passes on)
    from tests import test_figure_failure_keeps_results as keeps
    from tests.test_code_checks import GOOD_BODY, _dry, _quest_with_figures

    # (a) before the long run: a warning, never a stop
    dry = tmp_path / "dry"
    _quest_with_figures(dry, GOOD_BODY.replace("table.mean('signal', k)", f"table.mean('{UNDECLARED}', k)"))
    got = _dry(dry)
    assert got.passed is None and got.says.startswith("warning:") and UNDECLARED in got.says, got
    # (b) the connected chain
    plot = f"fig.axes[0].plot([1, 2], [table.mean('{UNDECLARED}', k) for k in table.keys()][:2])"
    body = "def figure_fig1(fig, table):\n    " + plot + "\n"
    monkeypatch.setattr(keeps, "_RAISES", body)
    eng, root, calls = keeps._give_up_quest(tmp_path / "run", monkeypatch)  # the first run: fig1 failed, fig2 drew
    first = frame_record(root)
    assert {o["id"]: o["status"] for o in first["figures"]}["fig1"] == "failed" and UNDECLARED in str(first["figures"])
    assert not (root / "figures" / "growth.png").exists(), "the lost figure is not there before the engine acts"
    eng.config.engine.exec_reflect_max_iterations = 0
    state = {"exec_reflect_iter": 0, "exec_result": {"returncode": 1, "results_produced": True},
             "figure_records": {"other.png": {"axes": []}, "growth.png": {"axes": []}}}  # the records of the first run
    assert not eng._repair_budget_left(state), "no repair is left"
    out = asyncio.run(eng._figures_after_give_up(state))
    assert calls and calls[0]["FI_FIGURE_FALLBACK"] == "only", "the engine's last run asks for the plan's own figure"
    assert (root / "figures" / "growth.png").stat().st_size > 0
    assert out["figure_records"]["growth.png"]["fallback"] is True and "fallback" not in out["figure_records"]["other.png"]
    paper_state = {"design": {"figures_planned": ["growth.png", "other.png"]}, "quest_root": str(root), **out}
    draft = "# T\n\n## Results\n\nFigure 1 shows the growth. Figure 2 shows another.\n"
    text, placed, _ = _place_figures(draft, paper_state)
    assert placed == ["growth.png", "other.png"] and "](figures/growth.png)" in text and "**Figure 1.**" in text, text


# ---- 2. a sentence names a series the figure does not draw ----------------------------------------------------------------------

DRAFT_HEAD = "# T\n\n## Abstract\nWe compare schemes.\n\n## Results\n\n"
DRAFT_TAIL = "\n## Discussion\nIt is small.\n"


def test_mutation_2_a_series_the_figure_does_not_draw_is_dropped_with_the_disclosure(tmp_path: Path, monkeypatch, frame5) -> None:
    """Caught: the sentence is dropped, and the paper carries the disclosure; a true statement of the same figure stays."""
    ghost = "{{figure:error_vs_step:drawn @ series=ghost, panel=1}}"
    real = "{{figure:error_vs_step:drawn @ series=fast, panel=1}}"
    draft = DRAFT_HEAD + f"**Accuracy.** {ghost} (Figure 1).\n\n**Cost.** {real} (Figure 1).\n" + DRAFT_TAIL
    _eng, state, _ = _write(tmp_path / "q", monkeypatch, [draft], _drawn_record(tmp_path))
    paper = _paper(state)
    assert "ghost" not in paper and "not recorded" not in paper
    assert "fast is drawn" in paper, "the same claim for a series the figure draws is filled"
    assert paper.count(DISCLOSURE) == 1
    from core import paper_facts

    gone = [d["key"] for d in paper_facts.kept_check(tmp_path / "q")["taken_out_values"] if str(d.get("key", "")).startswith("figure:")]
    assert "figure:error_vs_step:drawn @ series=ghost, panel=1" in gone, gone
    assert not [k for k in gone if "series=fast" in k], gone


# ---- 3. a quantity worked out in the body --------------------------------------------------------------------------------------


def test_mutation_3_a_quantity_computed_in_the_body_is_never_said_not_drawn(tmp_path: Path, frame5) -> None:
    """Caught: the caption names the series as drawn (``what`` or ``other``), and no statement says it is not drawn."""
    def body(fig, table):
        a, b = (table.mean("growth", k) for k in ("size=1", "size=2"))
        fig.line(0, "fast", [1, 2], [b / a, a / b], "other")  # a ratio the body worked out, no declared result

    assert _calls_run(tmp_path, body, panels=1)["status"] == "ok"
    calls = _calls(tmp_path)
    assert calls["panels"][0]["y"]["what"] == "other"
    caption = _figure_caption_text("g.png", None, n_seeds=None, calls=calls)
    assert "draws fast" in caption and "not drawn" not in caption and "not visible" not in caption, caption
    plan = {"series": [{"id": "fast", "label": ""}]}
    assert figure_claims.evaluate(calls, "not_drawn", {"series": "fast"}, plan=plan) is None
    assert figure_claims.evaluate(calls, "drawn", {"series": "fast", "panel": "1"})[0].startswith("fast is drawn")
    assert ("not_drawn", {"series": "fast"}) not in _offered(calls, plan)


# ---- 4. a number the body typed ------------------------------------------------------------------------------------------------


def test_mutation_4_a_typed_number_in_a_custom_panel_is_quoted_in_the_caption(tmp_path: Path, frame5) -> None:
    """Caught (raw panel): the typed text is quoted in the caption as not checked, and FI offers no claim on that panel."""
    def body(fig, table):
        fig.axes[0].plot([1, 2], [table.mean("growth", "size=1"), table.mean("growth", "size=2")])
        fig.axes[0].text(0.5, 0.5, "r = 0.9991")

    assert _calls_run(tmp_path, body, panels=1)["status"] == "ok"
    calls = _calls(tmp_path)
    assert calls["panels"][0]["raw"] is True
    caption = _figure_caption_text("g.png", None, n_seeds=None, calls=calls)
    assert "custom drawing" in caption and "not checked by FI: 'r = 0.9991'" in caption, caption
    assert figure_claims.menu(calls, None, stem="g") == []


def test_mutation_4_a_typed_number_in_a_note_is_refused_and_nothing_typed_survives(tmp_path: Path, frame5) -> None:
    """Caught (verified panel): ``fig.note`` with a typed number is refused, the figure is then drawn by FI from the plan's own
    description, and the number is in no record, caption or statement."""
    def body(fig, table):
        fig.line(0, "fast", [1, 2], [table.mean("growth", "size=1"), table.mean("growth", "size=2")], "growth")
        fig.note(0, what="r = 0.9991 {growth}", key="size=1")

    out = _calls_run(tmp_path, body, panels=1)
    assert out["status"] == "failed" and "digits" in out["why"], out
    assert not list(frame5.glob("*.calls.json")) or "0.9991" not in (frame5 / "g.calls.json").read_text(encoding="utf-8")
    # the run that ends the repairs draws the default figure instead, in which the number is nowhere
    from tests.test_analysis_frame import _design, _quest
    from tests.test_figure_failure_keeps_results import _analysis_run

    typed = ("def figure_fig1(fig, table):\n    fig.line(0, 'dose_1', [1, 2], [table.mean('growth', k) for k in table.keys()][:2], 'growth')\n"
             "    fig.note(0, what='r = 0.9991 {growth}')\n")
    quest = _quest(tmp_path / "run", _design(), typed)
    code, _printed, derived = _analysis_run(quest, last_run=True)
    fig = next(o for o in derived["figures"] if o["id"] == "fig1")
    assert code == 0 and fig["status"] == "ok" and fig["fallback"] is True and "0.9991" not in json.dumps(derived)


# ---- 5. a legend moved off the canvas ------------------------------------------------------------------------------------------


def _legend_inside_canvas(host) -> bool:
    host.fig.canvas.draw()
    box = host.axes[0].get_legend().get_window_extent(host.fig.canvas.get_renderer())
    return box.x0 >= -0.5 and box.y0 >= -0.5 and box.x1 <= host.fig.bbox.width + 0.5 and box.y1 <= host.fig.bbox.height + 0.5


def test_mutation_5_a_legend_moved_off_the_canvas_is_placed_by_fi(tmp_path: Path) -> None:
    """Caught (verified panel): FI writes the legend, inside the canvas, and the panel stays verified."""
    h = _Host(panels=1)
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.line(0, "slow", [1, 2, 3], [3.0, 2.0, 1.0], "growth")
    h.axes[0].legend(["fast", "slow"], loc="upper left", bbox_to_anchor=(1.6, 1.5))  # the body's own legend, off the canvas
    assert not _legend_inside_canvas(h), "the planted fault is there before FI writes the legend"
    h.finish(h.labels, {})
    assert _legend_inside_canvas(h)
    assert h.record()["panels"][0]["raw"] is False


def _recorded_by_the_house_style(tmp_path: Path, body: str) -> tuple[dict, Path]:
    """A one-panel figure whose function is ``body`` (the text of ``figure_g``), saved by the real figure code under the real
    recorder of the quest's interpreter; returns the recorder's record of the figure and the records folder."""
    code = tmp_path / "code"
    code.mkdir(parents=True)
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.FIGURE_CALLS_FILE).write_text(skeleton.FRAME_FIGURE_SOURCE, encoding="utf-8")
    boot = plot_style.write_boot(tmp_path / ".fi")
    records = tmp_path / ".fi" / plot_style.RECORDS_DIRNAME
    records.mkdir(parents=True)
    script = tmp_path / "run_one.py"
    script.write_text(textwrap.dedent("""
        import sys, types
        sys.path.insert(0, %r)
        sys.path.insert(0, %r)
        from core import analysis_figures as af
        class T:
            declared = ["growth"]
            def keys(self): return ["size=1", "size=2"]
            def setting(self, key): return {"size": float(key.split("=")[1])}
            def mean(self, ident, key): return {"size=1": 1.0, "size=2": 2.0}[key]
        %s
        ns = types.SimpleNamespace(figure_g=figure_g, COMPARED={"speed": ["fast", "slow"]})
        out = af.run_figures([{"id": "g", "file": "g.png", "panels": 1, "x": "size", "series": "fast"}], ns, T(), %r)
        print(out[0]["status"], out[0].get("why"))
    """) % (str(Path(__file__).resolve().parents[1]), str(code), textwrap.indent(textwrap.dedent(body), "        ").lstrip(),
            str(tmp_path / "figures")), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(records), "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=240,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert done.stdout.startswith("ok"), done.stdout + done.stderr[-1200:]
    return json.loads((records / "g.json").read_text(encoding="utf-8")), records


def test_mutation_5_a_legend_off_the_canvas_of_a_custom_panel_is_a_finding(tmp_path: Path) -> None:
    """Caught (raw panel): the panel is drawn by the body, so FI leaves the legend and records a finding that sends the run back
    once."""
    record, records = _recorded_by_the_house_style(tmp_path, """
        def figure_g(fig, table):
            ax = fig.raw(0)
            ax.plot([1, 2], [1.0, 2.0], label="fast")
            ax.plot([1, 2], [2.0, 1.0], label="slow")
            ax.legend(loc="upper left", bbox_to_anchor=(1.6, 1.5))
    """)
    assert json.loads((records / "g.calls.json").read_text(encoding="utf-8"))["panels"][0]["raw"] is True
    assert any("reaches outside the figure" in s for s in figure_facts.geometry_findings(record)), record.get("axes")
    found = _figure_overlap_findings({"figures": ["g.png"], "figure_records": {"g.png": record}})
    assert found and found[0].startswith("figures/g.png: ") and "legend" in found[0]
    assert _figure_overlap_findings({"figures": ["g.png"], "figure_records": {"g.png": record}, "figure_overlap_repaired": True}) == []
    # known good: the same custom panel with its legend where matplotlib puts it gives no finding
    good, _ = _recorded_by_the_house_style(tmp_path / "good", """
        def figure_g(fig, table):
            ax = fig.raw(0)
            ax.plot([1, 2], [1.0, 2.0], label="fast")
            ax.plot([1, 2], [2.0, 1.0], label="slow")
            ax.legend(loc="upper left")
    """)
    assert figure_facts.geometry_findings(good) == []


# ---- 6. a series copied under a second name -----------------------------------------------------------------------------------


def test_mutation_6_a_series_copied_under_a_second_name_is_merged_and_said_identical() -> None:
    """Caught: the two are one line to the reader (merged in the record, one legend entry), ``identical`` is offered, and the
    caption names them together."""
    h = _Host(panels=1, identity=True)
    h.labels["slow"] = "Slow colony"
    ys = [1.0, 2.0, 3.0]
    h.line(0, "fast", [1, 2, 3], ys, "growth")
    h.line(0, "slow", [1, 2, 3], list(ys), "growth")  # the same data under a second name
    h.finish(h.labels, {})
    calls = h.record()
    assert calls["panels"][0]["series"][0]["merged_with"] == ["slow"]
    assert [t.get_text() for t in h.axes[0].get_legend().get_texts()] == ["Fast colony = Slow colony"]
    got = figure_claims.evaluate(calls, "identical", {"a": "fast", "b": "slow", "panel": "1"})
    assert got is not None and "Fast colony" in got[0] and "Slow colony" in got[0], got
    assert ("identical", {"a": "fast", "b": "slow", "panel": "1"}) in _offered(calls)
    assert "Fast colony = Slow colony" in _figure_caption_text("g.png", None, n_seeds=None, calls=calls)


# ---- 7. limits set to the means, with sd bars -------------------------------------------------------------------------------


def test_mutation_7_ylim_set_to_the_means_widens_to_the_bars(tmp_path: Path) -> None:
    """Caught (verified panel): the limits are FI's and include the bar ends; a bar of all-zero widths gives no ``errorbars``
    claim."""
    h = _Host(panels=1)
    means, sds = [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]
    h.line(0, "fast", [1, 2, 3], means, "growth", yerr=sds, yerr_meaning="sd")
    h.axes[0].set_ylim(min(means), max(means))  # the body's own limits: exactly the means, so the bars are cut off
    assert h.axes[0].get_ylim() == (1.0, 3.0), "the planted fault is there before FI sets the limits"
    h.finish(h.labels, {})
    lo, hi = h.axes[0].get_ylim()
    assert lo <= min(means) - max(sds) and hi >= max(means) + max(sds), (lo, hi)
    assert h.record()["panels"][0]["raw"] is False
    flat = _Host(panels=1)
    flat.line(0, "fast", [1, 2, 3], means, "growth", yerr=[0.0, 0.0, 0.0], yerr_meaning="sd")
    flat.finish(flat.labels, {})
    assert figure_claims.evaluate(flat.record(), "errorbars", {"series": "fast", "panel": "1"}) is None
    assert not [c for c, _a in _offered(flat.record()) if c == "errorbars"]


def test_mutation_7_cut_off_error_bars_of_a_custom_panel_are_a_finding(tmp_path: Path) -> None:
    """Caught (raw panel): the bars reach past the limits the body set, which is a finding with the remedy of the repair round."""
    record, _records = _recorded_by_the_house_style(tmp_path, """
        def figure_g(fig, table):
            ax = fig.raw(0)
            ax.errorbar([1, 2, 3], [1.0, 2.0, 3.0], yerr=[0.5, 0.5, 0.5])
            ax.set_ylim(1.0, 3.0)
    """)
    assert any("error bars" in s and "cut off" in s for s in figure_facts.geometry_findings(record)), record.get("axes")
    found = _figure_overlap_findings({"figures": ["g.png"], "figure_records": {"g.png": record}})
    assert any("error bars" in s for s in found)
    # known good: limits that hold the bars give no finding
    good, _ = _recorded_by_the_house_style(tmp_path / "good", """
        def figure_g(fig, table):
            ax = fig.raw(0)
            ax.errorbar([1, 2, 3], [1.0, 2.0, 3.0], yerr=[0.5, 0.5, 0.5])
            ax.set_ylim(0.0, 4.0)
    """)
    assert figure_facts.geometry_findings(good) == []


# ---- 8. a writer sentence about a figure FI has no statement for -------------------------------------------------------------


def test_mutation_8_a_writer_sentence_about_the_figure_is_dropped_with_the_disclosure(tmp_path: Path, monkeypatch, frame5) -> None:
    """Caught: the sentence is gone (the figure draws no such series), the disclosure is in the paper and in the record, and the
    figure keeps FI's own statement."""
    draft = (DRAFT_HEAD + "**Accuracy.** Figure 1 shows the paired gain between the two schemes.\n\n"
             "**Cost.** {{figure:error_vs_step:drawn @ series=fast, panel=1}} (Figure 1).\n" + DRAFT_TAIL)
    _eng, state, _ = _write(tmp_path / "q", monkeypatch, [draft], _drawn_record(tmp_path))
    paper = _paper(state)
    assert "paired gain" not in paper and "fast is drawn" in paper
    assert paper.count(DISCLOSURE) == 1
    from core import paper_facts

    left_out = [d for d in paper_facts.kept_check(tmp_path / "q")["taken_out_values"] if str(d.get("key", "")).startswith("figure:")]
    assert left_out and {d["figure"] for d in left_out} == {"1"}


# ---- 9. the y label of another quantity ------------------------------------------------------------------------------------


def test_mutation_9_a_typed_y_label_never_appears(tmp_path: Path, frame5) -> None:
    """Caught: the label is FI's, from the plan; the body's text is in no axis label, record, caption or statement."""
    typed = "Temperature (K)"
    h = _Host(panels=1, identity=True)  # the table's own mean at size=v is v: what is drawn is the result `growth`
    h.line(0, "fast", [1, 2, 3], [1.0, 2.0, 3.0], "growth")
    h.axes[0].set_ylabel(typed)
    h.axes[0].set_title(typed)
    assert h.axes[0].get_ylabel() == typed, "the planted fault is there before FI writes the labels"
    h.finish(h.labels, {})
    assert h.axes[0].get_ylabel() == "Growth" and h.axes[0].get_title() == ""
    calls = h.record()
    assert calls["panels"][0]["y"]["label"] == "Growth"
    caption = _figure_caption_text("g.png", None, n_seeds=None, calls=calls)
    texts = [o["text"] for o in figure_claims.menu(calls, None, stem="g")]
    assert texts and "Temperature" not in json.dumps(calls) + caption + " ".join(texts), (caption, texts)
    # the same through the figure code a run uses (it writes the call record a paper is checked against)

    def body(fig, table):
        fig.line(0, "fast", [1, 2], [table.mean("growth", "size=1"), table.mean("growth", "size=2")], "growth")
        fig.axes[0].set_ylabel(typed)

    assert _calls_run(tmp_path, body, panels=1)["status"] == "ok"
    assert typed not in (frame5 / "g.calls.json").read_text(encoding="utf-8")
    assert _calls(tmp_path)["panels"][0]["raw"] is False, "an axis label is scaffolding FI rewrites, not a custom drawing"
