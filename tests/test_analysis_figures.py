"""The figures of the analysis FI writes (core/analysis_figures.py): FI creates, saves and closes each figure and a body only draws;
a body that saves or shows itself is refused; one body's failure does not stop the others; FI's own figures of a search come from
FI's record and say a missing best design or a check that did not run in a plain sentence. Made-up topic (a colony); the copied
record of a recorded search is used when present."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

from core import analysis_figures as af
from core import plot_style

RECORD = Path("C:/fi_runs/c1/skeleton/s4c_work/d34")


def _search_record(root: Path, *, best: Any = "make", checked: bool = False, scan: bool = False, starts: int = 2) -> Path:
    """A small made-up record of a search: results/best_design.json and raw/optimisation_ledger.jsonl under ``root``."""
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    objective = {"quantity": "growth_rate", "direction": "maximise", "unit": "1/h"}
    record: dict[str, Any] = {
        "objective": objective,
        "best": {"design": {"dose": 3.0}, "objective": 1.8} if best == "make" else best,
        "baseline": {"design": {"dose": 1.0}, "objective": 1.0},
        "improvement": {"value": 0.8, "unit": "1/h"},
        "checked_at_finer_settings": checked,
        "check": {"verdict": "unverified", "says": "The best design could not be checked at finer numerical settings: time ran out."},
    }
    (root / "results" / "best_design.json").write_text(json.dumps(record), encoding="utf-8")
    rows: list[dict[str, Any]] = [{"event": "search", "objective": objective}]
    n = 0
    for start in range(starts):
        for k in range(6):
            n += 1
            rows.append({"event": "evaluation", "n": n, "stage": "local", "start": start, "design": {"dose": float(k)},
                         "objective": 1.0 + 0.1 * k * (start + 1) - (0.05 if k == 3 else 0), "status": "ok", "feasible": True})
    if scan:
        for dose in (1, 2, 3):
            for rate in (0.1, 0.2):
                rows.append({"event": "evaluation", "stage": "scan", "design": {"dose": dose, "rate": rate},
                             "objective": dose * rate, "status": "ok"})
    (root / "raw" / "optimisation_ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return root


class _Table:
    """The little a body here reads."""

    def mean(self, ident: str, key: str) -> float:
        return {"a": 1.0, "b": 2.0}[key]

    def keys(self) -> list[str]:
        return ["a", "b"]


def _good(fig, table):
    fig.axes[0].bar(table.keys(), [table.mean("growth", k) for k in table.keys()])


def _namespace(**bodies: Any) -> types.SimpleNamespace:
    return types.SimpleNamespace(**bodies)


# ---- the static refusal --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("call", ["fig.savefig('x.png')", "plt.savefig('x.png')", "plt.show()", "plt.close(fig)", "plt.figure()",
                                  "plt.subplots()", "savefig('x.png')"])
def test_a_body_that_saves_shows_or_opens_a_figure_is_refused_with_a_plain_reason(call):
    source = "def figure_growth(fig, table):\n    fig.axes[0].plot([1, 2])\n    %s\n" % call
    reason = af.figure_body_problem(source)
    assert "figure_growth" in reason and "FI creates the figure" in reason
    assert "AST" not in reason and "Traceback" not in reason


def test_an_honest_body_and_other_functions_are_not_refused():
    source = ("def helper(fig):\n    fig.savefig('x.png')\n\n"
              "def figure_growth(fig, table):\n    ax = fig.axes[0]\n    ax.plot([1, 2])\n    ax.legend()\n")
    assert af.figure_body_problem(source) == ""  # a helper outside a figure body is not this check's
    assert af.figure_body_problem("def figure_x(:\n") == ""  # a syntax error is the gate's to say


def test_a_refused_body_is_not_run_and_the_others_are_drawn(tmp_path):
    drew = []

    def figure_bad(fig, table):  # pragma: no cover - never reached
        drew.append("bad")
        fig.savefig("x.png")

    out = af.run_figures([{"id": "bad", "file": "bad.png"}, {"id": "good", "file": "good.png"}],
                         _namespace(figure_bad=figure_bad, figure_good=_good), _Table(), str(tmp_path))
    assert [o["status"] for o in out] == ["refused", "ok"]
    assert drew == [] and not (tmp_path / "bad.png").exists() and (tmp_path / "good.png").is_file()
    assert "FI creates the figure" in out[0]["said"] and af.exit_code(out) == 1


# ---- a declared figure ----------------------------------------------------------------------------------------------------------
def test_a_body_that_raises_does_not_stop_the_other_figures(tmp_path):
    def figure_broken(fig, table):
        raise ValueError("no such column")

    figures = [{"id": "one", "file": "one.png"}, {"id": "broken", "file": "broken.png"}, {"id": "three", "file": "three.png"}]
    out = af.run_figures(figures, _namespace(figure_one=_good, figure_broken=figure_broken, figure_three=_good), _Table(), str(tmp_path))
    assert [o["status"] for o in out] == ["ok", "failed", "ok"]
    assert (tmp_path / "one.png").is_file() and (tmp_path / "three.png").is_file() and not (tmp_path / "broken.png").exists()
    assert "ValueError" in out[1]["why"] and "no such column" in out[1]["why"] and af.exit_code(out) == 1
    assert af.said(out).startswith("2 of 3 figure(s) drawn.")


def test_a_stub_is_said_and_is_not_a_failure(tmp_path):
    class NotYet(Exception):
        pass

    def figure_later(fig, table):
        raise NotYet("not written")

    out = af.run_figures([{"id": "later", "file": "later.png"}, {"id": "missing", "file": "m.png"}],
                         _namespace(figure_later=figure_later), _Table(), str(tmp_path))
    assert [o["status"] for o in out] == ["not_written", "not_written"] and af.exit_code(out) == 0
    assert "not written yet" in out[0]["said"]


def test_a_body_that_draws_nothing_is_said_and_no_empty_file_is_kept(tmp_path):
    out = af.run_figures([{"id": "blank", "file": "blank.png"}], _namespace(figure_blank=lambda fig, table: None), _Table(), str(tmp_path))
    assert out[0]["status"] == "failed" and "drew nothing" in out[0]["said"] and not (tmp_path / "blank.png").exists()


def test_fi_creates_one_axes_per_panel_and_closes_the_figure(tmp_path):
    import matplotlib.pyplot as plt
    seen: list[Any] = []

    def figure_two(fig, table):
        seen.append((len(fig.axes), fig.x, fig.series, fig.file))
        for ax in fig.axes:
            ax.plot([1, 2, 3])

    before = len(plt.get_fignums())
    out = af.run_figures([{"id": "two", "file": "two.png", "panels": 2, "x": "dose", "series": "rate"}],
                         _namespace(figure_two=figure_two), _Table(), str(tmp_path))
    assert out[0]["status"] == "ok" and seen == [(2, "dose", "rate", "two.png")]
    assert len(plt.get_fignums()) == before  # FI closed it


def test_a_figure_is_saved_through_the_recorder_of_the_house_style(tmp_path):
    """FI saves with the quest's own interpreter in the house style, so the figure recorder notes what the figure draws."""
    boot = plot_style.write_boot(tmp_path / ".fi")
    records = tmp_path / ".fi" / plot_style.RECORDS_DIRNAME
    records.mkdir(parents=True)
    script = tmp_path / "run_one.py"
    script.write_text(textwrap.dedent("""
        import sys, types
        sys.path.insert(0, %r)
        from core import analysis_figures as af
        def figure_growth(fig, table):
            fig.axes[0].plot([1, 2, 3], label="growth")
            fig.axes[0].legend()
        out = af.run_figures([{"id": "growth", "file": "growth.png"}], types.SimpleNamespace(figure_growth=figure_growth), None, %r)
        print(out[0]["status"])
    """) % (str(Path(__file__).resolve().parents[1]), str(tmp_path / "figures")), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(records), "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=240,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert done.stdout.strip() == "ok", done.stderr[-800:]
    assert (tmp_path / "figures" / "growth.png").is_file()
    assert any(p.name.startswith("growth") for p in records.glob("*.json"))


# ---- FI's own figures of a search -------------------------------------------------------------------------------------------------
def test_the_search_figures_are_drawn_from_a_small_record_and_the_files_exist(tmp_path):
    root = _search_record(tmp_path / "q", scan=True)
    out = af.search_figures(str(root), str(tmp_path / "figs"))
    assert [(o["file"], o["status"]) for o in out] == [(af.PROGRESS_FILE, "ok"), (af.BEST_VS_BASELINE_FILE, "ok"), (af.SCAN_FILE, "ok")]
    for o in out:
        assert (tmp_path / "figs" / o["file"]).stat().st_size > 1000
    assert af.exit_code(out) == 0
    assert "not checked" in af.said(out).lower() or "could not be checked" in af.said(out)  # the check did not run: said


def test_one_line_per_start_of_the_best_so_far():
    rows = [{"n": 1, "objective": 1.0, "start": 0}, {"n": 2, "objective": 3.0, "start": 0}, {"n": 3, "objective": 2.0, "start": 0},
            {"n": 4, "objective": 5.0, "start": 1}, {"n": 5, "objective": 4.0, "start": None}]
    lines = af._best_so_far(rows, "maximise")
    assert set(lines) == {"start 0", "start 1", "global search"}
    assert lines["start 0"] == [(1.0, 1.0), (2.0, 3.0), (3.0, 3.0)]  # the best so far never goes down; the line runs to its last row
    assert af._best_so_far(rows, "minimise")["start 0"] == [(1.0, 1.0), (3.0, 1.0)]


def test_no_best_design_and_no_check_are_said_in_a_plain_sentence_not_a_crash(tmp_path):
    root = _search_record(tmp_path / "q", best=None)
    out = af.search_figures(str(root), str(tmp_path / "figs"))
    assert af.exit_code(out) == 0 and all(o["status"] != "failed" for o in out)
    sentence = af.said(out)
    assert "no design that meets every limit" in sentence and "no best design to show" in sentence
    assert "checked at finer numerical settings" in sentence
    assert (tmp_path / "figs" / af.BEST_VS_BASELINE_FILE).is_file()  # the baseline is still shown


def test_a_checked_best_design_has_no_such_note(tmp_path):
    root = _search_record(tmp_path / "q", checked=True)
    assert af.best_sentence(json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))) == ""


def test_a_missing_record_is_said_and_not_a_crash(tmp_path):
    out = af.search_figures(str(tmp_path / "nothing"), str(tmp_path / "figs"))
    assert out and all(o["status"] == "skipped" for o in out) and af.exit_code(out) == 0
    assert "not drawn" in af.said(out)


def test_a_scan_with_one_setting_is_a_line_and_without_a_scan_there_is_no_map(tmp_path):
    root = _search_record(tmp_path / "q")
    assert [o["file"] for o in af.search_figures(str(root), str(tmp_path / "f1"))] == [af.PROGRESS_FILE, af.BEST_VS_BASELINE_FILE]
    data = {"scan": [{"design": {"dose": d}, "objective": d * 2.0, "level": None} for d in (1, 2, 3)]}
    assert af._scan_figure(data, str(tmp_path / "f2"))["status"] == "ok"
    assert af._scan_figure({"scan": [{"design": {"dose": 1}, "objective": 1.0}]}, str(tmp_path / "f3"))["status"] == "skipped"


@pytest.mark.skipif(not (RECORD / "results" / "best_design.json").is_file(), reason="the copied record of the recorded search is absent")
def test_the_recorded_search_with_its_best_removed_ends_with_exit_0_and_says_so(tmp_path):
    root = tmp_path / "q"
    shutil.copytree(RECORD, root)
    path = root / "results" / "best_design.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["best"] = None
    record["checked_at_finer_settings"] = False
    path.write_text(json.dumps(record), encoding="utf-8")
    out = af.search_figures(str(root), str(tmp_path / "figs"))
    assert af.exit_code(out) == 0
    assert (tmp_path / "figs" / af.PROGRESS_FILE).is_file() and (tmp_path / "figs" / af.BEST_VS_BASELINE_FILE).is_file()
    assert "no best design to show" in af.said(out) and "checked at finer numerical settings" in af.said(out)


@pytest.mark.skipif(not (RECORD / "results" / "best_design.json").is_file(), reason="the copied record of the recorded search is absent")
def test_the_recorded_search_as_it_is_draws_its_progress_and_says_the_check_did_not_run(tmp_path):
    out = af.search_figures(str(RECORD), str(tmp_path / "figs"))
    assert [o["status"] for o in out] == ["ok", "ok"] and af.exit_code(out) == 0
    assert "could not be checked" in af.said(out)
