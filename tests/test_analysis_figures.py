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
        "best": {"design": {"dose": 3.0}, "objective": 2.0} if best == "make" else best,
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


# ---- the runtime guard ----------------------------------------------------------------------------------------------------------
def _run(tmp_path, body, name="x"):
    out = af.run_figures([{"id": name, "file": name + ".png"}], {"figure_" + name: body}, _Table(), str(tmp_path))
    return out[0]


def _saved_alias():
    import matplotlib.pyplot as plt
    return plt.savefig  # bound before the guard, the way a top-level `from matplotlib.pyplot import savefig` is


_EARLY_SAVE = _saved_alias()


def _helper_that_saves(fig):
    fig.savefig("helper.png")


def _bodies():
    def aliased(fig, table):
        fig.axes[0].plot([1, 2])
        _EARLY_SAVE("early.png")

    def helper(fig, table):
        fig.axes[0].plot([1, 2])
        _helper_that_saves(fig.axes[0].figure)

    def canvas(fig, table):
        fig.axes[0].plot([1, 2])
        fig.axes[0].figure.canvas.print_figure("canvas.png")

    def run_text(fig, table):
        fig.axes[0].plot([1, 2])
        exec("import matplotlib.pyplot as p\np.savefig('exec.png')")

    def shows(fig, table):
        fig.axes[0].plot([1, 2])
        __import__("matplotlib.pyplot").pyplot.show()

    def makes(fig, table):
        fig.axes[0].plot([1, 2])
        __import__("matplotlib.pyplot").pyplot.figure()

    def swallows(fig, table):
        fig.axes[0].plot([1, 2])
        try:
            fig.axes[0].figure.savefig("swallowed.png")
        except Exception:
            pass

    return {"aliased": aliased, "helper": helper, "canvas": canvas, "exec": run_text, "shows": shows, "makes": makes,
            "swallows": swallows}


@pytest.mark.parametrize("which", ["aliased", "helper", "canvas", "exec", "shows", "makes", "swallows"])
def test_a_body_that_saves_shows_or_makes_a_figure_is_refused_at_run_time_with_a_plain_reason(tmp_path, which, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = _run(tmp_path, _bodies()[which], which)
    assert out["status"] == "refused", out
    assert "FI creates the figure" in out["why"] and "Traceback" not in out["why"]
    assert not list(tmp_path.glob("*.png"))  # nothing was saved by the body, and FI drew nothing


def test_the_guard_is_lifted_after_the_body_and_a_figure_made_in_it_is_closed(tmp_path):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    saved, shown = Figure.savefig, plt.show
    before = len(plt.get_fignums())
    _run(tmp_path, _bodies()["makes"], "makes")
    assert Figure.savefig is saved and plt.show is shown and len(plt.get_fignums()) == before


def test_an_honest_body_with_a_file_or_connection_close_is_not_refused(tmp_path):
    class Handle:
        closed = False

        def close(self):
            self.closed = True

        def show(self):
            return "rows"

    def figure_honest(fig, table):
        fh = Handle()
        fh.show()
        fig.axes[0].plot([1, 2, 3])
        fig.axes[0].figure.subplots_adjust(left=0.2)
        fh.close()

    assert _run(tmp_path, figure_honest, "honest")["status"] == "ok"


def test_a_refused_body_is_not_kept_and_the_others_are_drawn(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = af.run_figures([{"id": "bad", "file": "bad.png"}, {"id": "good", "file": "good.png"}],
                         _namespace(figure_bad=_bodies()["canvas"], figure_good=_good), _Table(), str(tmp_path))
    assert [o["status"] for o in out] == ["refused", "ok"] and af.exit_code(out) == 1
    assert not (tmp_path / "bad.png").exists() and (tmp_path / "good.png").is_file()


def test_a_figure_that_fails_does_not_leave_the_file_of_an_earlier_run(tmp_path):
    (tmp_path / "later.png").write_bytes(b"old figure")

    def figure_later(fig, table):
        raise ValueError("now broken")

    out = af.run_figures([{"id": "later", "file": "later.png"}], _namespace(figure_later=figure_later), _Table(), str(tmp_path))
    assert out[0]["status"] == "failed" and not (tmp_path / "later.png").exists()
    (tmp_path / "later.png").write_bytes(b"old figure")
    assert af.run_figures([{"id": "later", "file": "later.png"}], _namespace(), _Table(), str(tmp_path))[0]["status"] == "not_written"
    assert not (tmp_path / "later.png").exists()


def test_one_empty_panel_of_two_is_a_failure_of_that_figure(tmp_path):
    def figure_half(fig, table):
        fig.axes[0].plot([1, 2])

    out = af.run_figures([{"id": "half", "file": "half.png", "panels": 2}], _namespace(figure_half=figure_half), _Table(), str(tmp_path))
    assert out[0]["status"] == "failed" and "panel 2" in out[0]["why"] and not (tmp_path / "half.png").exists()


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


def _rows(level, objectives, *, n0=0, feasible=None):
    out = []
    for k, obj in enumerate(objectives):
        out.append({"event": "evaluation", "n": n0 + k + 1, "stage": "baseline" if k == 0 else "local", "start": 0,
                    "design": {"d": k}, "objective": obj, "feasible": True if feasible is None else feasible[k], "status": "ok",
                    "level": level})
    return out


def _write(root, rows, record, direction="maximise"):
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    if record is not None:
        (root / "results" / "best_design.json").write_text(json.dumps(record), encoding="utf-8")
    head = [{"event": "search", "objective": {"quantity": "q", "direction": direction, "unit": "u"}}] if direction else []
    (root / "raw" / "optimisation_ledger.jsonl").write_text("\n".join(json.dumps(r) for r in head + rows) + "\n", encoding="utf-8")


def _ends(root):
    data = af._read_record(str(root))
    sign = 1.0 if data["objective"].get("direction") == "minimise" else -1.0
    return {a["label"]: (a["line"][-1][1] if a["line"] else None) for a in af._alternatives(data, sign)}


def test_the_level_rules_equal_the_search_s_own():
    from core import optimise_search
    for level in ({"a": 1, "b": "x"}, {}, None):
        assert af.level_key(level) == optimise_search.level_key(level)
    assert af.level_says({"a": 1, "b": "x"}) == optimise_search.level_says({"a": 1, "b": "x"})


def test_an_infeasible_better_row_does_not_move_the_best_so_far(tmp_path):
    rows = _rows(None, [5.0, 4.0, 1.0, 3.0], feasible=[True, True, False, True])  # minimise: 1.0 breaks a limit
    _write(tmp_path / "q", rows, {"objective": {"direction": "minimise"}, "best": {"design": {}, "objective": 3.0}}, "minimise")
    assert _ends(tmp_path / "q") == {"search": 3.0}  # the best of the rows that meet every limit, not the 1.0


def test_two_alternatives_are_two_lines_each_ending_at_its_own_best(tmp_path):
    a, b = {"family": "a"}, {"family": "b"}
    rows = _rows(a, [1.0, 3.0, 2.0]) + _rows(b, [1.0, 5.0], n0=3)
    record = {"objective": {"direction": "maximise"}, "best": {"design": {}, "objective": 5.0},
              "compared": {"levels": [{"level": a, "best": {"objective": 3.0}}, {"level": b, "best": {"objective": 5.0}}]}}
    _write(tmp_path / "q", rows, record)
    assert _ends(tmp_path / "q") == {"family = a": 3.0, "family = b": 5.0}  # A's 3.0 is not hidden by B's 5.0; the winner ends at the file's best
    out = af.search_figures(str(tmp_path / "q"), str(tmp_path / "f"))
    assert out[0]["status"] == "ok" and af.fi_faults(out) == []


def test_a_line_that_does_not_end_at_the_recorded_best_is_an_fi_fault_not_the_models(tmp_path):
    _write(tmp_path / "q", _rows(None, [1.0, 3.0]), {"objective": {"direction": "maximise"}, "best": {"design": {}, "objective": 4.0}})
    out = af.search_figures(str(tmp_path / "q"), str(tmp_path / "f"))
    assert out[0]["status"] == "fi_fault" and af.fi_faults(out) and af.exit_code(out) == 0
    assert not (tmp_path / "f" / af.PROGRESS_FILE).exists()


def test_a_minimise_search_without_the_best_design_file_takes_its_direction_from_the_ledger(tmp_path):
    _write(tmp_path / "q", _rows(None, [5.0, 2.0, 3.0]), None, "minimise")
    assert _ends(tmp_path / "q") == {"search": 2.0}
    assert af.search_figures(str(tmp_path / "q"), str(tmp_path / "f"))[0]["status"] == "ok"


def test_no_direction_anywhere_draws_no_best_line(tmp_path):
    _write(tmp_path / "q", _rows(None, [5.0, 2.0]), None, direction="")
    out = af.search_figures(str(tmp_path / "q"), str(tmp_path / "f"))
    assert out[0]["status"] == "skipped" and "higher or lower" in out[0]["said"]


def test_a_baseline_row_numbered_start_0_does_not_join_the_start_0_line():
    rows = [{"n": 1, "objective": 9.0, "start": 0, "stage": "baseline", "feasible": True, "level": None},
            {"n": 2, "objective": 1.0, "start": 0, "stage": "local", "feasible": True, "level": None}]
    alt = af._alternatives({"rows": rows, "record": {}, "objective": {}}, 1.0)[0]
    assert alt["line"][0] == (1.0, 9.0)  # the baseline counts for the best so far (FI's rule) ...
    assert alt["starts"] == [[(2.0, 1.0)]]  # ... but not for start 0's own line


def test_a_baseline_that_breaks_a_limit_is_drawn_and_said(tmp_path):
    root = _search_record(tmp_path / "q")
    path = root / "results" / "best_design.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["baseline"]["feasible"] = False
    path.write_text(json.dumps(record), encoding="utf-8")
    assert af._best_vs_baseline_figure(af._read_record(str(root)), str(tmp_path / "f"))["status"] == "ok"


def test_the_scan_map_orders_its_axes_by_count_then_name(tmp_path):
    data = {"scan": [{"design": {"b": b, "a": a}, "objective": a + b} for a in (1, 2) for b in (1, 2)]}
    assert af._scan_figure(data, str(tmp_path))["status"] == "ok"


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


# ---- fix round 2 ------------------------------------------------------------------------------------------------------------------
def test_a_body_that_picks_one_of_the_two_panels_with_pyplot_sca_is_not_refused(tmp_path):
    import matplotlib.pyplot as plt

    def figure_pick(fig, table):
        plt.sca(fig.axes[1])
        plt.plot([1, 2, 3])
        fig.axes[0].plot([3, 2, 1])

    out = af.run_figures([{"id": "pick", "file": "pick.png", "panels": 2}], _namespace(figure_pick=figure_pick), _Table(), str(tmp_path))
    assert out[0]["status"] == "ok", out


def test_an_alternative_that_was_not_searched_gives_no_false_fault_and_is_said(tmp_path):
    a, b = {"family": "a"}, {"family": "b"}
    rows = _rows(a, [1.0, 3.0]) + _rows(b, [1.0], n0=2)  # b: probe rows only
    record = {"objective": {"direction": "maximise"}, "best": {"design": {}, "objective": 3.0},
              "compared": {"levels": [{"level": a, "best": {"objective": 3.0}},
                                      {"level": b, "best": {"objective": 3.0, "evaluation": 2}, "same_as": a, "probes": 1}]}}
    _write(tmp_path / "q", rows, record)
    out = af.search_figures(str(tmp_path / "q"), str(tmp_path / "f"))
    assert out[0]["status"] == "ok" and af.fi_faults(out) == []
    assert "family = b gave the same results as family = a" in out[0]["note"]


@pytest.mark.parametrize("where", ["handle", "folder"])
def test_a_fault_of_fi_s_own_lines_is_fi_s_and_never_the_model_s(tmp_path, monkeypatch, where):
    if where == "handle":
        monkeypatch.setattr(af, "_figure_handle", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("frame broke")))
    else:
        monkeypatch.setattr(af, "_folder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("folder broke")))
    out = af.run_figures([{"id": "g", "file": "g.png"}], _namespace(figure_g=_good), _Table(), str(tmp_path))
    assert out[0]["status"] == "fi_fault" and af.exit_code(out) == 0 and af.fi_faults(out)


def test_an_fi_fault_after_the_body_is_fi_s_too(tmp_path, monkeypatch):
    calls = {"n": 0}
    real = af._folder

    def folder(out_dir):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("folder broke after the body")
        return real(out_dir)

    monkeypatch.setattr(af, "_folder", folder)
    out = af.run_figures([{"id": "g", "file": "g.png"}], _namespace(figure_g=_good), _Table(), str(tmp_path))
    assert out[0]["status"] == "fi_fault" and af.exit_code(out) == 0


def test_the_copied_level_rules_equal_the_search_s_own_for_empty_levels_too():
    from core import optimise_search
    assert af.level_says({}) == optimise_search.level_says({}) == ""
    assert af.level_key({}) == optimise_search.level_key({}) == "" and af.level_key(None) == optimise_search.level_key(None)


def test_a_file_the_body_wrote_in_the_figure_folder_is_removed_and_refused(tmp_path):
    def figure_png(fig, table):
        fig.axes[0].plot([1, 2])
        with open(str(tmp_path / "side.png"), "wb") as handle:
            fig.axes[0].figure.canvas.print_png(handle)

    (tmp_path / "other.png").write_bytes(b"another figure of an earlier step")
    out = af.run_figures([{"id": "png", "file": "png.png"}], _namespace(figure_png=figure_png), _Table(), str(tmp_path))
    assert out[0]["status"] == "refused" and "side.png" in out[0]["why"]
    assert not (tmp_path / "side.png").exists() and not (tmp_path / "png.png").exists()
    assert (tmp_path / "other.png").read_bytes() == b"another figure of an earlier step"


# ---- a figure that drew nothing from the trial record ----------------------------------------------------------------------------
def _real_table(declared=("r1", "r2")):
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = [{"cell": {"a": a}, "key": "a=%d" % a, "planned": 2, "ok": 2, "failed": 0,
              "metrics": {"r1": {"values": [1.0 * a, 2.0 * a]}, "r2": {"values": [float("nan")]}}} for a in (1, 2, 3)]
    return ns["Table"](cells, list(declared))


def _shown(tmp_path, body, shows=("r1",), table=None):
    spec = {"id": "x", "file": "x.png"}
    if shows != "missing":
        spec["shows"] = list(shows)
    return af.run_figures([spec], {"figure_x": body}, table or _real_table(), str(tmp_path))[0]


def test_a_figure_that_hides_the_error_of_its_reads_and_draws_only_curves_is_failed_with_the_error_text(tmp_path):
    def figure_x(fig, table):
        import numpy as np

        for p in range(1, 3):
            try:
                table.mean("r1", table.at(a=p))  # a list, not a key
            except Exception:
                continue
        fig.axes[0].plot(np.linspace(1, 3, 5), np.linspace(1, 3, 5) ** 2)
        fig.axes[0].axhline(0.05)

    out = _shown(tmp_path, figure_x)
    assert out["status"] == "failed" and "drew nothing from the trial record" in out["why"]
    assert "TypeError: " in out["why"] and "list of settings" in out["why"] and "`r1`" not in out["why"].split("\n")[0]  # the error line names no result; the read-form lines do
    assert not (tmp_path / "x.png").exists()


def test_a_figure_that_reads_only_a_result_with_no_value_is_failed(tmp_path):
    def figure_x(fig, table):
        got = [table.mean("r2", k) for k in table.keys()]
        fig.axes[0].plot([1, 2, 3], [g for g in got if g is not None] or [0, 0, 0])

    assert _shown(tmp_path, figure_x)["status"] == "failed"


def test_a_figure_that_reads_and_plots_a_result_is_ok(tmp_path):
    def figure_x(fig, table):
        fig.axes[0].plot([1, 2, 3], [table.mean("r1", k) for k in table.keys()])

    assert _shown(tmp_path, figure_x)["status"] == "ok"


def test_what_the_figure_is_declared_to_show_does_not_decide_it(tmp_path):
    """A shown id may be a derived result with no declared inputs, or a comparison: the body reads the results it is made of."""
    def figure_x(fig, table):
        fig.axes[0].plot([1, 2, 3], [table.mean("r1", k) for k in table.keys()])

    for shows in (["slope"], ["diff_between_settings"], [], "missing"):
        assert _shown(tmp_path, figure_x, shows=shows)["status"] == "ok", shows


def test_a_read_inside_a_derive_the_body_calls_counts(tmp_path):
    def derive_slope(table):
        return [table.mean("r1", k) for k in table.keys()]

    def figure_x(fig, table):
        fig.axes[0].plot([1, 2, 3], derive_slope(table))

    assert _shown(tmp_path, figure_x, shows=["slope"])["status"] == "ok"


def test_only_a_reference_a_count_of_undefined_or_trial_numbers_is_not_a_read_of_a_result(tmp_path):
    def figure_x(fig, table):
        table.reference("r1", "a=1")
        table.undefined("r1", "a=1")
        table.trials("r1", "a=1")
        fig.axes[0].plot([1, 2], [1, 2])

    assert _shown(tmp_path, figure_x)["status"] == "failed"


def test_the_reads_are_forgotten_between_figures(tmp_path):
    table = _real_table()

    def figure_one(fig, table):
        fig.axes[0].plot([1, 2, 3], [table.mean("r1", k) for k in table.keys()])

    def figure_two(fig, table):
        fig.axes[0].plot([1, 2], [1, 2])

    out = af.run_figures([{"id": "one", "file": "one.png", "shows": ["r1"]}, {"id": "two", "file": "two.png", "shows": ["r1"]}],
                         {"figure_one": figure_one, "figure_two": figure_two}, table, str(tmp_path))
    assert [o["status"] for o in out] == ["ok", "failed"]


def test_a_table_that_does_not_record_its_reads_is_not_held_to_the_check(tmp_path):
    def figure_x(fig, table):
        fig.axes[0].axhline(0.05)

    assert _shown(tmp_path, figure_x, table=_Table())["status"] == "ok"


def test_the_failure_text_survives_what_a_model_may_read_when_a_quantity_is_left_out(tmp_path, monkeypatch):
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    import asyncio

    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]

    def figure_x(fig, table):
        try:
            table.mean("zzz", "a=1")  # not declared: the error would list the declared results, among them the left-out one
        except Exception:
            pass
        fig.axes[0].plot([1, 2], [1, 2])

    out = _shown(tmp_path, figure_x, shows=(key,), table=_real_table(declared=(key,)))
    first, _nl, listed = out["said"].partition("\n")
    assert out["status"] == "failed" and key not in first and key in listed  # the declared ids are on a line of their own
    seen = eng._stderr_for_models(af.said([out]))
    # the line that lists the declared results names a left-out quantity and is taken out; the first line, which names the id
    # that was read, stays
    assert "drew nothing from the trial record" in seen and "`zzz` is not a result the plan declares" in seen
    assert "[FI NOTE]" in seen and f"`{key}`" in seen.split("[FI NOTE]")[1] and f"results that can be read: {key}" not in seen


def test_a_figure_that_declares_no_result_is_not_checked_and_one_that_declares_results_is(tmp_path):
    def layout(fig, table):
        fig.axes[0].plot([1, 2], [1, 2])  # a layout or a reference curve: reads nothing, says it shows no result

    assert _shown(tmp_path, layout, shows=[])["status"] == "ok"
    assert _shown(tmp_path, layout, shows="missing")["status"] == "ok"
    assert _shown(tmp_path, layout, shows=["r1"])["status"] == "failed"


def test_a_levels_error_that_names_the_result_is_cut_to_its_type_and_the_line_survives_what_a_model_reads(tmp_path, monkeypatch):
    from tests.test_run_checks import _typed_engine
    import asyncio

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cells = [{"cell": {"a": 1}, "key": "a=1", "planned": 1, "ok": 1, "failed": 0, "metrics": {key: {"values": [1.0]}}},
             {"cell": {"a": 1}, "level": {"m": 1, "n": 2}, "key": "a=1", "metrics": {key: {"values": [1.0]}}}]
    table = ns["Table"](cells, [key])

    def figure_x(fig, table):
        try:
            table.levels(key, "m", "a=1")  # the run also compares another factor: the error names the result
        except Exception:
            pass
        fig.axes[0].plot([1, 2], [1, 2])

    out = _shown(tmp_path, figure_x, table=table)
    assert out["status"] == "failed" and key not in out["said"] and "raised ValueError." in out["why"]
    assert "drew nothing from the trial record" in eng._stderr_for_models(af.said([out]))


def test_an_undeclared_read_names_the_id_and_the_results_that_can_be_read(tmp_path):
    def figure_x(fig, table):
        try:
            table.values("r1_lo", "a=1")  # the interval end of a result: FI's record holds it, the plan does not declare it
        except Exception:
            pass
        fig.axes[0].plot([1, 2], [1, 2])

    out = _shown(tmp_path, figure_x, table=_real_table(declared=("r1", "r2")))
    assert out["status"] == "failed"
    assert "The last read raised UnknownResult: `r1_lo` is not a result the plan declares." in out["why"]
    assert "The results that can be read: r1." in out["why"] and "r2" not in out["why"].split("can be read")[1]


def test_a_known_result_is_read_by_its_coverage_from_the_record(tmp_path):
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    two = {"values": [0.0, 1.0, 2.0, 3.0], "trials": [0, 1, 2, 3]}
    cells = [{"cell": {"s": 1}, "key": "s=1", "planned": 4, "ok": 4, "failed": 0, "metrics": {"est": two}},
             {"cell": {"s": 1, "m": "a"}, "key": "s=1,m=a", "level": {"m": "a"},
              "metrics": {"lo": {"values": [0.0, 1.0, 1.0, 9.0], "trials": [0, 1, 2, 3]},
                          "hi": {"values": [2.0, 3.0, 1.5, 9.5], "trials": [0, 1, 2, 3]}}},
             {"cell": {"s": 1, "m": "b"}, "key": "s=1,m=b", "level": {"m": "b"},
              "metrics": {"lo": {"values": [0.0, 1.0], "trials": [0, 2]}, "hi": {"values": [5.0], "trials": [0]}}}]
    table = ns["Table"](cells, ["est", "other"], known={"est": {"interval": ["lo", "hi"], "truth": 1.2}})
    assert table.coverage("est", "s=1", m="a") == [1.0, 1.0, 1.0, 0.0]  # the truth 1.2 is inside the first three intervals only
    assert table.coverage("est", "s=1", m="b") == [1.0, 0.0]  # trial 2 reported no upper end: a miss, not left out
    assert table.coverage("est", "s=1", m="c") == [] and "est" in table.gave_values
    with pytest.raises(ns["UnknownResult"], match="no true value and interval"):
        table.coverage("other", "s=1", m="a")
    with pytest.raises(ns["UnknownResult"], match="`zzz` is not a result the plan declares"):
        table.coverage("zzz", "s=1")


# ---- a panel's title is FI's, from the setting the body declared for it ----------------------------------------------------------
def test_a_declared_panel_is_titled_by_fi_from_its_setting_not_by_the_body(tmp_path):
    def figure_grid(fig, table):
        for i, size in enumerate((20, 200)):
            fig.axes[i].plot([1, 2])
            fig.axes[i].set_title("(dashed: a curve this panel does not draw)")
            fig.panel(i, size=size)

    out = af.run_figures([{"id": "grid", "file": "grid.png", "panels": 2}], _namespace(figure_grid=figure_grid), _Table(), str(tmp_path))
    assert out[0]["status"] == "ok"
    assert [p["title"] for p in out[0]["panels"]] == ["size = 20", "size = 200"]
    record = json.loads((tmp_path / ".fi" / "figure_records" / "grid.json").read_text(encoding="utf-8")) if (tmp_path / ".fi").exists() else None
    if record:  # the house style's recorder is what the caption reads
        assert [a["title"] for a in record["axes"]] == ["size = 20", "size = 200"]


def test_a_panel_given_nothing_keeps_the_body_title_and_a_one_panel_figure_is_unchanged(tmp_path):
    def figure_one(fig, table):
        fig.axes[0].plot([1, 2])
        fig.axes[0].set_title("growth over time")

    import matplotlib.pyplot as plt
    seen = {}
    real = af._set_panel_titles

    def spy(handle, axes, labels):
        got = real(handle, axes, labels)
        seen["title"], seen["record"] = axes[0].get_title(), got
        return got

    af._set_panel_titles = spy
    try:
        out = af.run_figures([{"id": "one", "file": "one.png"}], _namespace(figure_one=figure_one), _Table(), str(tmp_path))
    finally:
        af._set_panel_titles = real
    assert out[0]["status"] == "ok" and out[0]["panels"] == [] and seen == {"title": "growth over time", "record": []}
    plt.close("all")


def test_a_panel_declared_for_a_setting_or_value_the_study_does_not_have_is_refused_naming_what_may_be_used(tmp_path):
    def figure_grid(fig, table):
        fig.axes[0].plot([1, 2])
        fig.axes[1].plot([1, 2])
        fig.panel(0, size=20)
        fig.panel(1, size=999)

    spec = {"id": "grid", "file": "grid.png", "panels": 2, "settings": {"size": [20, 200], "kind": ["a", "b"]}, "labels": {"size": "Colony size"}}
    out = af.run_figures([spec], _namespace(figure_grid=figure_grid), _Table(), str(tmp_path))
    assert out[0]["status"] == "failed" and "UnknownSetting" in out[0]["why"] and "size = 20, 200" in out[0]["why"] and "kind = a, b" in out[0]["why"]
    ok = af.run_figures([dict(spec, id="g2", file="g2.png")], _namespace(figure_g2=lambda fig, table: [a.plot([1, 2]) for a in fig.axes] and fig.panel(1, size=200)), _Table(), str(tmp_path))
    assert ok[0]["status"] == "ok" and ok[0]["panels"][0]["title"] == "size = 200".replace("size", "Colony size")


def test_a_figure_that_draws_nothing_is_its_own_failure_and_the_others_are_still_saved(tmp_path) -> None:
    class Tbl:
        def __init__(self):
            self.gave_values = set()

    def empty(fig, table):
        fig.axes[0].axvline(1.0)  # a reference line only: no read of a result gave data

    def fine(fig, table):
        fig.axes[0].plot([0, 1], [0, 1])
        table.gave_values.add("g")

    specs = [{"id": "a", "file": "a.png", "shows": ["g"], "panels": 1}, {"id": "b", "file": "b.png", "shows": ["g"], "panels": 1}]
    out = af.run_figures(specs, _namespace(figure_a=empty, figure_b=fine), Tbl(), str(tmp_path))
    assert [o["status"] for o in out] == ["failed", "ok"]
    assert (tmp_path / "b.png").is_file() and not (tmp_path / "a.png").exists()


# ---- the repair note and the record of what a figure drew, from the table's own tracking ------------------------------------------
def _level_table(references=None, declared=("r1", "r2", "r3", "r4")):
    """A study whose result ``r1`` is recorded for each of several long-named methods inside each run, not once per setting."""
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    methods = ["method_with_a_long_name_%d" % i for i in range(1, 7)]
    cells = [{"cell": {"a": 1}, "key": "a=1", "planned": 2, "ok": 2, "failed": 0, "metrics": {"r2": {"values": [1.0, 2.0]}}}]
    for m in methods:
        cells.append({"cell": {"a": 1, "method": m}, "level": {"method": m}, "key": "a=1,method=%s" % m, "planned": 2, "ok": 2,
                      "failed": 0, "metrics": {"r1": {"values": [0.5, 0.6]}}})
    return ns["Table"](cells, list(declared), references)


def test_the_repair_note_of_a_figure_keeps_how_each_shown_result_is_read_when_the_error_is_long(tmp_path) -> None:
    def figure_x(fig, table):
        table.values("r1", "a=1")  # r1 is recorded for each method, not once for the setting

    out = _shown(tmp_path, figure_x, shows=("r1", "r2"), table=_level_table())
    assert out["status"] == "failed"
    assert "table.level_rows('r1', key)" in out["why"] and "table.level_means('r1', key)" in out["why"]
    assert "How the results this figure shows are read" in out["why"] and "read it with `table.values('r2', key)`" in out["why"]


def test_a_long_read_error_that_the_body_swallowed_is_not_cut_before_it_says_how_to_read(tmp_path) -> None:
    def figure_x(fig, table):
        fig.axes[0].axvline(1.0)
        try:
            table.mean("r1", "a=1")
        except Exception:
            pass  # swallowed: FI's record of the last read is what the repair note is built from

    out = _shown(tmp_path, figure_x, shows=("r1",), table=_level_table())
    first = out["why"].split("\n")[0]
    assert "drew nothing from the trial record" in first and len(first) > 240
    assert "read it with `table.level_rows('r1', key)`" in first and "UnknownResult" in first


def test_a_body_that_invents_a_read_is_sent_back_with_the_real_read_forms(tmp_path) -> None:
    def figure_x(fig, table):
        fig.axes[0].plot([1, 2])
        table.level_ro("r1", "a=1")  # no such read

    out = _shown(tmp_path, figure_x, shows=("r1",), table=_level_table())
    assert out["status"] == "failed" and "no attribute 'level_ro'" in out["why"]
    assert "table.level_rows('r1', key)" in out["why"]


def test_the_record_of_a_drawn_figure_says_which_declared_results_the_body_drew(tmp_path, monkeypatch) -> None:
    records = tmp_path / "records"
    records.mkdir()
    (records / "x.json").write_text(json.dumps({"file": "x.png", "axes": []}), encoding="utf-8")
    monkeypatch.setenv("FI_FIGURE_RECORDS", str(records))

    def figure_x(fig, table):
        fig.axes[0].plot(table.values("r1", "a=1"))
        try:
            table.values("r2", "a=1")  # r2 has no finite value: nothing drawn
        except Exception:
            pass

    table = _real_table()
    spec = {"id": "x", "file": "x.png", "shows": ["r1", "r2"], "result_labels": {"r1": "First quantity"}}
    out = af.run_figures([spec], {"figure_x": figure_x}, table, str(tmp_path / "figs"))[0]
    assert out["status"] == "ok"
    want = {"shows": ["r1", "r2"], "drawn": ["r1"], "not_drawn": ["r2"], "labels": {"r1": "First quantity"}}
    assert out["declared"] == want
    assert json.loads((records / "x.json").read_text(encoding="utf-8"))["declared"] == want


def test_a_body_that_reads_only_with_level_rows_or_reference_has_drawn_that_result(tmp_path) -> None:
    def figure_x(fig, table):
        rows = table.level_rows("r1", "a=1")
        fig.axes[0].plot([v for _k, _t, vals in rows for v in vals])
        table.reference("r3", "a=1")

    table = _level_table(references={"r3": {"a=1": 0.0}})
    out = _shown(tmp_path, figure_x, shows=("r1", "r3"), table=table)
    assert out["status"] == "ok", out["why"]
    assert out["declared"]["drawn"] == ["r1", "r3"] and out["declared"]["not_drawn"] == []


def test_a_read_form_is_given_only_for_a_way_that_works_for_that_kind_of_result(tmp_path) -> None:
    def figure_x(fig, table):
        fig.axes[0].axvline(1.0)

    out = _shown(tmp_path, figure_x, shows=("r2", "r3", "r4"), table=_level_table(references={"r3": {"a=1": 0.5}}))
    why = out["why"]
    assert "`r2` is recorded once for each setting" in why
    assert "`r3` is a reference value" in why and "table.reference('r3', key)" in why
    assert "`r4`" not in why  # recorded nowhere, no reference: FI has no way to read it, so it says none
    assert not (set(_level_table().gave_values))


def test_the_head_of_a_long_error_is_capped_but_the_read_forms_are_not(tmp_path) -> None:
    def figure_x(fig, table):
        fig.axes[0].axvline(1.0)
        try:
            table.values("n" * 3000, "a=1")
        except Exception:
            pass

    out = _shown(tmp_path, figure_x, shows=("r1",), table=_level_table())
    first = out["why"].split("\n")[0]
    assert len(first) < 1000 and "table.level_rows('r1', key)" in out["why"]


# ---- the repair note is never cut, and the plan's default figure ----------------------------------------------------------------
class _Grid:
    """A table of a made-up study: settings ``size`` (3 values) and ``rate`` (2 values); results ``gain`` and ``loss`` per setting."""

    def __init__(self, error: str = ""):
        self.gave_values: set[str] = set()
        self.last_error = {"gain": error} if error else {}
        self._cells = {"size=%d,rate=%s" % (s, r): {"size": s, "rate": r} for r in (0.1, 0.2) for s in (10, 20, 30)}

    def keys(self) -> list[str]:
        return list(self._cells)

    def setting(self, key: str) -> dict[str, Any]:
        return dict(self._cells[key])

    def mean(self, name: str, key: str) -> float:
        self.gave_values.add(name)
        cell = self._cells[key]
        return cell["size"] * (1 if name == "gain" else 2) + cell["rate"]

    def per_level(self, name: str, key: str) -> list[str]:
        return []


def _default(tmp_path, **spec):
    return af.default_figure({"id": "f1", "file": "f1.png", **spec}, _Grid(), str(tmp_path))


def test_repair_note_is_not_cut_at_240(tmp_path):
    long_error = "RuntimeError: " + ", ".join("form %d reads the value of level %d" % (i, i) for i in range(30))
    assert len(long_error) > 600
    out = af.run_figures([{"id": "x", "file": "x.png", "shows": ["gain"]}], {"figure_x": lambda fig, table: fig.axes[0].axhline(1)},
                         _Grid(long_error), str(tmp_path))[0]
    assert out["status"] == "failed" and long_error.rstrip(".") in out["why"]


def test_default_figure_draws_declared_shows_over_x(tmp_path):
    out = _default(tmp_path, shows=["gain", "loss"], x="size")
    assert out["status"] == "ok" and (tmp_path / "f1.png").stat().st_size > 0
    assert out["declared"]["drawn"] == ["gain", "loss"]


def test_default_figure_draws_two_lines_of_three_points(tmp_path, monkeypatch):
    seen = {}
    real = af._new_figure

    def spy(panels):
        plt, fig, axes = real(panels)
        seen["axes"] = axes
        return plt, fig, axes

    monkeypatch.setattr(af, "_new_figure", spy)
    _default(tmp_path, shows=["gain", "loss"], x="size", series="rate")
    lines = seen["axes"][0].lines
    assert len(lines) == 4 and all(len(line.get_xdata()) == 3 for line in lines)  # 2 results x 2 values of the series
    seen.clear()
    _default(tmp_path, shows=["gain", "loss"], x="size")
    lines = seen["axes"][0].lines
    # rate is neither x nor series: it is held at its first value (0.1), so a line holds one point for each size
    assert [list(l.get_xdata()) for l in lines] == [[10, 20, 30]] * 2
    assert [list(l.get_ydata()) for l in lines] == [[10.1, 20.1, 30.1], [20.1, 40.1, 60.1]]


def test_default_figure_marks_fallback(tmp_path):
    out = _default(tmp_path, shows=["gain"], x="size")
    assert out["fallback"] is True and out["status"] == "ok" and out["file"] == "f1.png" and out["id"] == "f1"
    assert out["said"] == af._outcome("f1", "f1.png", "ok")["said"]


@pytest.mark.parametrize("spec", [{}, {"shows": ["gain"]}, {"x": "size"}, {"shows": [], "x": "size"}])
def test_default_figure_is_dropped_with_reason_when_shows_undeclared(tmp_path, spec):
    (tmp_path / "f1.png").write_bytes(b"old")
    out = _default(tmp_path, **spec)
    assert out["status"] == "dropped" and out["why"].strip() and out["fallback"] is True
    assert not (tmp_path / "f1.png").exists() and out["said"] == out["why"]


def test_default_figure_one_panel_per_declared_panel(tmp_path, monkeypatch):
    seen = {}
    real = af._new_figure

    def spy(panels):
        plt, fig, axes = real(panels)
        seen["axes"] = axes
        return plt, fig, axes

    monkeypatch.setattr(af, "_new_figure", spy)
    out = _default(tmp_path, shows=["gain"], x="size", panels=2, settings={"rate": [0.1, 0.2]}, labels={"rate": "learning rate"})
    assert out["status"] == "ok" and len(seen["axes"]) == 2
    assert [p["title"] for p in out["panels"]] == ["learning rate = 0.1", "learning rate = 0.2"]
    assert [a.get_title() for a in seen["axes"]] == ["learning rate = 0.1", "learning rate = 0.2"]
    assert [len(a.lines) for a in seen["axes"]] == [1, 1]


class _Levels(_Grid):
    """``gain`` is recorded once per setting; ``lev`` once for each of two values compared inside the runs (so ``mean`` raises)."""

    def mean(self, name: str, key: str) -> float:
        if name == "lev":
            self.last_error["lev"] = "UnknownResult: `lev` is recorded for each value"
            raise LookupError("`lev` is recorded for each value of `mode` (a, b)")
        return super().mean(name, key)

    def per_level(self, name: str, key: str) -> list[str]:
        return ["a", "b"] if name == "lev" else []

    def level_means(self, name: str, key: str) -> dict[str, float]:
        self.gave_values.add(name)
        size = self._cells[key]["size"]
        return {"a": size + 1.0, "b": size + 2.0}


def _spy_axes(monkeypatch):
    seen = {}
    real = af._new_figure

    def spy(panels):
        plt, fig, axes = real(panels)
        seen["axes"] = axes
        return plt, fig, axes

    monkeypatch.setattr(af, "_new_figure", spy)
    return seen


def test_default_figure_draws_a_result_recorded_per_compared_value_one_line_each(tmp_path, monkeypatch):
    seen = _spy_axes(monkeypatch)
    out = af.default_figure({"id": "f1", "file": "f1.png", "shows": ["lev"], "x": "size", "result_labels": {"lev": "response"}},
                            _Levels(), str(tmp_path))
    assert out["status"] == "ok" and out["fallback"] is True
    lines = seen["axes"][0].lines
    assert [l.get_label() for l in lines] == ["response, a", "response, b"]
    assert [list(l.get_ydata()) for l in lines] == [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]


def test_default_figure_holds_every_other_setting_and_says_which(tmp_path, monkeypatch):
    seen = _spy_axes(monkeypatch)
    out = _default(tmp_path, shows=["gain"], x="size", labels={"rate": "learning rate"})
    assert out["held"] == {"rate": 0.1}
    ax = seen["axes"][0]
    assert list(ax.lines[0].get_ydata()) == [10.1, 20.1, 30.1] and ax.get_title() == "learning rate = 0.1"
    assert ax.get_ylabel() == "gain" and ax.get_legend() is not None  # one line still gets a y label and a legend


def test_default_figure_says_when_it_has_fewer_panels_than_asked(tmp_path):
    out = _default(tmp_path, shows=["gain"], x="size", panels=3, settings={"rate": [0.1, 0.2]})
    assert out["status"] == "ok" and len(out["panels"]) == 2 and "2 panel(s) instead of the 3" in out["why"] and out["note"] == out["why"]
    one = _default(tmp_path, shows=["gain"], x="size", panels=2)
    assert "1 panel(s) instead of the 2" in one["why"]


def test_default_figure_categorical_x_keeps_one_axis_for_every_line(tmp_path, monkeypatch):
    seen = _spy_axes(monkeypatch)

    class Cat(_Grid):
        def __init__(self):
            super().__init__()
            self._cells = {"m=%s" % m: {"m": m} for m in ("p", "q", "r")}

        def mean(self, name, key):
            self.gave_values.add(name)
            return float(len(key))

    af.default_figure({"id": "f1", "file": "f1.png", "shows": ["gain", "loss"], "x": "m"}, Cat(), str(tmp_path))
    ax = seen["axes"][0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["p", "q", "r"] and all(list(l.get_xdata()) == [0, 1, 2] for l in ax.lines)


def test_default_figure_drops_when_no_value_is_readable_and_does_not_leave_a_stale_error(tmp_path):
    class Empty(_Grid):
        def mean(self, name, key):
            self.last_error[name] = "RuntimeError: nothing"
            return None

    table = Empty()
    out = af.default_figure({"id": "f1", "file": "f1.png", "shows": ["gain"], "x": "size"}, table, str(tmp_path))
    assert out["status"] == "dropped" and "no value of `gain` along `size`" in out["why"] and out["fallback"] is True
    assert not (tmp_path / "f1.png").exists() and table.last_error == {}


def test_default_figure_own_fault_is_fi_fault_with_fallback_and_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(af, "_new_figure", lambda panels: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _default(tmp_path, shows=["gain"], x="size")
    assert out["status"] == "fi_fault" and out["fallback"] is True and "boom" in out["why"] and not (tmp_path / "f1.png").exists()


def test_unknown_result_tail_lists_only_the_figures_own_results_uncut(tmp_path):
    ids = ["r%03d" % i for i in range(120)]  # a tail well over the old 600 characters
    table = _Grid("UnknownResult: `zz` is not a result the plan declares; the declared results are: " + ", ".join(ids))
    out = af.run_figures([{"id": "x", "file": "x.png", "shows": ["r007", "r110"]}], {"figure_x": lambda fig, t: fig.axes[0].axhline(1)},
                         table, str(tmp_path))[0]
    assert "The results that can be read: r007, r110." in out["why"] and "r008" not in out["why"]


# ---- the figure handle with the drawing calls: the call record, and what is drawn outside the calls ---------------------------------
class _CallTable:
    """A small made-up record of a study: two sizes, one result."""

    declared = ["growth"]

    def __init__(self):
        self.gave_values = set()

    def keys(self):
        return ["size=1", "size=2"]

    def setting(self, key):
        return {"size": float(key.split("=")[1])}

    def mean(self, ident, key):
        self.gave_values.add(ident)
        return {"size=1": 1.0, "size=2": 2.0}[key]

    def n(self, ident, key):
        return 10

    def interval(self, ident, key):
        return (self.mean(ident, key) - 0.5, self.mean(ident, key) + 0.5)

    def reset_reads(self):
        self.gave_values.clear()


@pytest.fixture
def frame5(tmp_path, monkeypatch):
    """The frame of the drawing calls, written beside the figure code the way a quest has it, and a records folder."""
    from core import skeleton

    code = tmp_path / "code"
    code.mkdir()
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.FIGURE_CALLS_FILE).write_text(skeleton.FRAME_FIGURE_SOURCE, encoding="utf-8")
    for name in ("fi_frame", "fi_figure_calls"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(code))
    records = tmp_path / ".fi" / "figure_records"
    records.mkdir(parents=True)
    monkeypatch.setenv("FI_FIGURE_RECORDS", str(records))
    yield records
    for name in ("fi_frame", "fi_figure_calls"):
        sys.modules.pop(name, None)


def _calls_run(tmp_path, body, panels=2, table=None):
    ns = types.SimpleNamespace(figure_g=body, COMPARED={"speed": ["fast", "slow"]})
    spec = {"id": "g", "file": "g.png", "panels": panels, "x": "size", "series": "fast", "shows": ["growth"]}
    return af.run_figures([spec], ns, table or _CallTable(), str(tmp_path / "figures"))[0]


def _drawn(fig, table, i):
    return fig.line(i, "fast", [1, 2], [table.mean("growth", "size=1"), table.mean("growth", "size=2")], "growth")


def _both_panels(fig, table):
    for i in range(len(fig.axes)):
        _drawn(fig, table, i)


def test_primitive_only_body_has_no_raw_panel(tmp_path, frame5):
    out = _calls_run(tmp_path, _both_panels)
    assert out["status"] == "ok", out
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    assert [p["raw"] for p in record["panels"]] == [False, False] and record["fallback"] is False
    assert out["declared"]["raw_panels"] == [] and out["declared"]["calls"]["series"] == 2


def test_bare_pyplot_call_makes_panel_raw_and_figure_is_saved(tmp_path, frame5):
    def body(fig, table):
        _drawn(fig, table, 0)
        fig.axes[1].plot([1, 2, 3], [3, 2, 1])
        fig.axes[1].text(0.5, 0.5, "slope 0.37")

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok" and (tmp_path / "figures" / "g.png").is_file()
    panels = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]
    assert [p["raw"] for p in panels] == [False, True]
    assert panels[1]["raw_reason"] and "slope 0.37" in panels[1]["texts_raw"] and panels[0]["texts_raw"] == []
    assert out["declared"]["raw_panels"] == [2]


def test_fig_raw_call_makes_panel_raw_even_when_nothing_is_untagged(tmp_path, frame5):
    def body(fig, table):
        _drawn(fig, table, 0)
        fig.raw(1).text(0.5, 0.5, "free text")

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok" and out["declared"]["raw_panels"] == [2]
    assert json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][1]["raw_reason"] == "fig.raw"


def test_call_record_written_beside_recorder_files(tmp_path, frame5):
    (frame5 / "g.calls.json").write_text("{}", encoding="utf-8")  # an earlier run's record
    assert _calls_run(tmp_path, lambda fig, table: None)["status"] == "failed"  # drew nothing: the old record goes with the old figure
    assert not (frame5 / "g.calls.json").exists()
    assert _calls_run(tmp_path, _both_panels)["status"] == "ok"
    record = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))
    assert record["version"] == 1 and len(record["panels"]) == 2
    assert record["panels"][0]["series"][0]["id"] == "fast"


def test_the_handle_is_given_the_plan_and_the_table(tmp_path, frame5):
    seen = {}

    def body(fig, table):
        seen.update(settings=fig.settings, series=fig.series_ids, table=fig.table is table, derived=fig.derived_ids)
        _both_panels(fig, table)

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok"
    assert seen["settings"] == ["size", "speed"] and seen["series"] == ["fast", "slow"] and seen["table"] is True


def test_an_inconsistency_in_FIs_own_calls_is_fi_fault(tmp_path, frame5):
    def body(fig, table):
        _both_panels(fig, table)
        fig._fc_extra[1]["series"].pop()  # the calls kept one artist fewer than the record holds: FI's own fault

    out = _calls_run(tmp_path, body)
    assert out["status"] == "fi_fault" and "panel 2" in out["why"] and af.fi_faults([out])
    assert not (frame5 / "g.calls.json").exists()


def test_log_line_counts_checked_and_custom_panels(tmp_path, frame5):
    def body(fig, table):
        for i in (0, 1, 3):
            _drawn(fig, table, i)
        fig.axes[2].scatter([1, 2], [2, 1])

    out = _calls_run(tmp_path, body, panels=4)
    assert out["note"] == "figure g: 3 of 4 panels checked by FI, panel 3 is a custom drawing"
    assert out["note"] in af.said([out])
    assert _calls_run(tmp_path, _both_panels)["note"] == "figure g: 2 of 2 panels checked by FI"


def test_body_without_primitives_on_old_frame_has_no_call_record(tmp_path, monkeypatch):
    records = tmp_path / "rec"
    records.mkdir()
    monkeypatch.setenv("FI_FIGURE_RECORDS", str(records))
    stub = tmp_path / "old"
    stub.mkdir()
    (stub / "fi_frame.py").write_text(
        "class Figure:\n    def __init__(self, axes, x=None, series=None, shows=None, file=None):\n"
        "        self.axes, self.x, self.series, self.shows, self.file = list(axes), x, series, shows, file\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "fi_frame", raising=False)
    old = _calls_run(tmp_path, lambda fig, table: [a.plot([table.mean("growth", "size=1"), 2]) for a in fig.axes], panels=1)  # no frame at all: the plain holder
    monkeypatch.syspath_prepend(str(stub))
    monkeypatch.delitem(sys.modules, "fi_frame", raising=False)
    v4 = _calls_run(tmp_path, lambda fig, table: [a.plot([table.mean("growth", "size=1"), 2]) for a in fig.axes], panels=1)  # a frame of an earlier version
    sys.modules.pop("fi_frame", None)
    for out in (old, v4):
        assert out["status"] == "ok" and "note" not in out and "calls" not in out.get("declared", {})
    assert list(records.glob("*.calls.json")) == []


def test_calls_are_recorded_with_the_real_recorder_and_agree_with_it(tmp_path):
    """The house-style recorder of the quest's interpreter saves the figure; error bars (their caps are lines too) do not disagree."""
    from core import skeleton

    code = tmp_path / "code"
    code.mkdir()
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
        def figure_g(fig, table):
            fig.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.2])
            fig.points(0, "slow", [1, 2], [2.0, 1.0], "growth")
            fig.band(0, "fast", [1, 2], [0.5, 1.5], [1.5, 2.5], "growth", "other")
            fig.ref(0, value=1.5)
            fig.bars(1, "fast", [1, 2], [1.0, 2.0], "growth")
            fig.axes[1].plot([0, 1], [0, 1])
        ns = types.SimpleNamespace(figure_g=figure_g, COMPARED={"speed": ["fast", "slow"]})
        out = af.run_figures([{"id": "g", "file": "g.png", "panels": 2, "x": "size", "series": "fast"}], ns, T(), %r)
        print(out[0]["status"], out[0].get("note"), out[0].get("why"))
    """) % (str(Path(__file__).resolve().parents[1]), str(code), str(tmp_path / "figures")), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(records), "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=240,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert done.stdout.strip() == "ok figure g: 1 of 2 panels checked by FI, panel 2 is a custom drawing", done.stdout + done.stderr[-800:]
    assert json.loads((records / "g.calls.json").read_text(encoding="utf-8"))["panels"][1]["raw"] is True


def test_typed_text_on_a_panel_makes_it_a_custom_drawing(tmp_path, frame5):
    def body(fig, table):
        _drawn(fig, table, 0)
        fig.axes[0].text(0.5, 0.5, "r = 0.9991")
        _drawn(fig, table, 1)
        fig.axes[1].annotate("slope 0.37", (1, 1))

    out = _calls_run(tmp_path, body)
    panels = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]
    assert [p["raw"] for p in panels] == [True, True] and panels[0]["raw_reason"] == "text placed outside FI's calls"
    assert "r = 0.9991" in panels[0]["texts_raw"] and out["declared"]["raw_panels"] == [1, 2]


def test_a_body_using_only_fi_notes_stays_verified(tmp_path, frame5):
    def body(fig, table):
        _both_panels(fig, table)
        fig.note(0, what="mean growth: {growth}", key="size=1")

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok" and out["declared"]["raw_panels"] == []


def test_a_figure_level_text_makes_every_panel_a_custom_drawing(tmp_path, frame5):
    def body(fig, table):
        _both_panels(fig, table)
        fig.axes[0].figure.text(0.5, 0.01, "n = 12")

    assert _calls_run(tmp_path, body)["declared"]["raw_panels"] == [1, 2]


def test_a_one_panel_figure_is_labelled_from_the_plans_words_and_units(tmp_path, frame5):
    spec = {"id": "g", "file": "g.png", "panels": 1, "x": "size", "series": "fast", "shows": ["growth"],
            "labels": {"size": "Colony size"}, "result_labels": {"growth": "Growth"},
            "result_info": {"growth": {"unit": "1/h", "description": "growth rate of a colony"}}}
    ns = types.SimpleNamespace(figure_g=_both_panels, COMPARED={"speed": ["fast"]})
    out = af.run_figures([spec], ns, _CallTable(), str(tmp_path / "figures"))[0]
    assert out["status"] == "ok"
    panel = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][0]
    assert panel["x"]["label"] == "Colony size" and panel["y"]["label"] == "Growth (1/h)"


def test_result_info_reaches_the_figure_spec_from_the_plan():
    from core import analysis_frame

    c = types.SimpleNamespace(results=[types.SimpleNamespace(id="growth", unit="1/h", estimand="growth rate", label="Growth")], derived=[])
    assert analysis_frame._result_info(c) == {"result_info": {"growth": {"label": "Growth", "unit": "1/h", "description": "growth rate"}}}
    assert analysis_frame._result_info(types.SimpleNamespace(results=[types.SimpleNamespace(id="g", unit="", estimand="")], derived=[])) == {}


def test_a_figure_of_several_panels_names_its_x_and_series_settings():
    from core import analysis_frame

    c = types.SimpleNamespace(labels={"size": "Colony size", "speed": "Speed", "other": "Other"},
                              settings=types.SimpleNamespace(compared={"speed": [1, 2]}, in_each_run={}))
    got = analysis_frame._panel_words(c, 2, ("size", "speed"))
    assert got["labels"] == {"speed": "Speed", "size": "Colony size"} and list(got["settings"]) == ["speed"]


def test_a_two_panel_figure_is_labelled_with_the_plans_units_on_every_axis(tmp_path, frame5):
    spec = {"id": "g", "file": "g.png", "panels": 2, "x": "size", "series": "fast", "shows": ["growth"],
            "labels": {"size": "Colony size"}, "result_info": {"growth": {"label": "Growth", "unit": "1/h"}}}
    ns = types.SimpleNamespace(figure_g=_both_panels, COMPARED={"speed": ["fast"]})
    assert af.run_figures([spec], ns, _CallTable(), str(tmp_path / "figures"))[0]["status"] == "ok"
    for panel in json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]:
        assert panel["x"]["label"] == "Colony size" and panel["y"]["label"] == "Growth (1/h)"


def _raw_after(tmp_path, frame5, extra, panels=2):
    """Run a body that draws on every panel through FI's calls, then does ``extra(fig)``; the raw flags of the panels."""
    def body(fig, table):
        _both_panels(fig, table)
        extra(fig)

    out = _calls_run(tmp_path, body, panels=panels)
    assert out["status"] == "ok", out
    return [p["raw"] for p in json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]]


@pytest.mark.parametrize("name,extra,expected", [
    ("manual tick labels", lambda fig: (fig.axes[0].set_xticks([1, 2]), fig.axes[0].set_xticklabels(["low", "r = 0.99"])), [True, False]),
    ("a table", lambda fig: fig.axes[1].table(cellText=[["slope 0.4"]], loc="bottom"), [False, True]),
    ("an inset", lambda fig: fig.axes[0].inset_axes([0.5, 0.5, 0.3, 0.3]).text(0.1, 0.1, "n = 7"), [True, False]),
    ("a twin axes title", lambda fig: fig.axes[1].twinx().set_title("p = 0.03"), [False, True]),
    ("a twin axes legend", lambda fig: (lambda t: (t.plot([1, 2], [3, 4], label="x"), t.legend()))(fig.axes[0].twinx()), [True, False]),
    ("a colorbar text", lambda fig: fig.axes[0].figure.colorbar(fig.axes[0].imshow([[1, 2], [3, 4]]), ax=fig.axes[0], label="bound 0.5"), [True, True]),
    ("a figure legend", lambda fig: fig.axes[0].figure.legend(["a"], loc="upper center"), [True, True]),
    ("a figure text", lambda fig: fig.axes[0].figure.text(0.5, 0.01, "n = 12"), [True, True]),
    ("a suptitle", lambda fig: fig.axes[0].figure.suptitle("shown: 2.5"), [True, True]),
    ("a spoofed tag", lambda fig: fig.axes[1].text(0.5, 0.5, "r = 0.9").set_gid("fi:1"), [False, True]),
    ("a body line with a spoofed tag", lambda fig: fig.axes[0].plot([1, 2], [9, 9], gid="fi:2"), [True, False]),
])
def test_every_artist_FI_did_not_make_makes_its_panel_a_custom_drawing(tmp_path, frame5, name, extra, expected):
    assert _raw_after(tmp_path, frame5, extra) == expected, name


def test_a_suptitle_stays_on_a_custom_drawing(tmp_path, frame5):
    held = []
    _raw_after(tmp_path, frame5, lambda fig: held.append(fig.axes[0].figure) or fig.axes[0].figure.suptitle("shown: 2.5"))
    assert held[0]._suptitle.get_text() == "shown: 2.5"  # FI clears a title only when it writes the words itself


def test_an_fi_line_whose_data_the_body_changed_is_a_custom_drawing(tmp_path, frame5):
    def change(fig):
        fig.axes[0].lines[0].set_ydata([5.0, 6.0])

    assert _raw_after(tmp_path, frame5, change) == [True, False]
    reasons = [p["raw_reason"] for p in json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]]
    assert reasons == ["drawing changed after FI's call", None]
    assert _raw_after(tmp_path, frame5, lambda fig: (fig.line(1, "slow", [1, 2], [2.0, 1.0], "growth"), fig.axes[1].lines[0].remove()))         == [False, True]


def test_categories_numeric_strings_and_large_arrays_are_verified_without_a_fault(tmp_path, frame5):
    big = list(range(25000))

    def body(fig, table):
        table.mean("growth", "size=1")
        fig.line(0, "fast", ["a", "b"], [1.0, 2.0], "growth")
        fig.line(1, "fast", ["1", "2"], [1.0, 2.0], "growth")
        fig.bars(1, "slow", ["1", "2"], [3.0, 4.0], "growth")

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok" and out["declared"]["raw_panels"] == []
    out = _calls_run(tmp_path, lambda fig, table: (table.mean("growth", "size=1"), fig.line(0, "fast", big, [float(i % 5) for i in big], "growth"),
                                                    fig.line(1, "fast", [1, 2], [1.0, None], "growth")))
    assert out["status"] == "ok" and out["declared"]["raw_panels"] == []


def test_a_body_using_only_fi_calls_stays_verified(tmp_path, frame5):
    def body(fig, table):
        _drawn(fig, table, 0)
        fig.line(0, "slow", [1, 2], [2.0, 1.0], "growth", yerr=[0.1, 0.2])
        fig.band(0, "fast", [1, 2], [0.5, 1.5], [1.5, 2.5], "growth", "other")
        fig.curve(0, "trend", [1, 2], [1.0, 2.0])
        fig.ref(0, value=1.5)
        fig.note(0, what="mean growth: {growth}", key="size=1")
        _drawn(fig, table, 1)
        fig.bars(1, "slow", [1, 2], [1.0, 2.0], "growth")
        fig.image(1, [[1, 2], [3, 4]])

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok" and out["declared"]["raw_panels"] == [], out


@pytest.mark.parametrize("category", [True, False])
def test_an_honest_body_the_recorder_cannot_read_is_not_a_fi_fault(tmp_path, category):
    """A category x and a series of over 20000 points are lines the figure recorder cannot hold: the check skips them."""
    from core import skeleton

    code = tmp_path / "code"
    code.mkdir()
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    (code / skeleton.FIGURE_CALLS_FILE).write_text(skeleton.FRAME_FIGURE_SOURCE, encoding="utf-8")
    boot = plot_style.write_boot(tmp_path / ".fi")
    records = tmp_path / ".fi" / plot_style.RECORDS_DIRNAME
    records.mkdir(parents=True)
    draw = ('fig.line(0, "fast", ["a", "b"], [1.0, 2.0], "growth")' if category else
            'fig.line(0, "fast", list(range(20001)), [float(i % 7) for i in range(20001)], "growth")')
    script = tmp_path / "run_one.py"
    script.write_text(textwrap.dedent("""
        import sys, types
        sys.path.insert(0, %r)
        sys.path.insert(0, %r)
        from core import analysis_figures as af
        class T:
            declared = ["growth"]
            def keys(self): return ["size=1"]
            def setting(self, key): return {"size": 1.0}
            def mean(self, i, k): return 1.0
        def figure_g(fig, table):
            table.mean("growth", "size=1")
            %s
        out = af.run_figures([{"id": "g", "file": "g.png", "panels": 1, "x": "size", "series": "fast"}],
                             types.SimpleNamespace(figure_g=figure_g, COMPARED={}), T(), %r)
        print(out[0]["status"], out[0].get("why"))
    """) % (str(Path(__file__).resolve().parents[1]), str(code), draw, str(tmp_path / "figures")), encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONPATH": str(boot), "FI_FIGURE_RECORDS": str(records), "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=240,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert done.stdout.split()[:1] == ["ok"], done.stdout + done.stderr[-800:]
    assert (records / "g.calls.json").is_file()


# ---- a panel of the drawing calls says which setting it shows, and FI writes that title --------------------------------------------
def _hold_run(tmp_path, body, labels=None):
    ns = types.SimpleNamespace(figure_g=body, COMPARED={"speed": ["fast", "slow"]})
    spec = {"id": "g", "file": "g.png", "panels": 2, "x": "size", "series": "fast", "shows": ["growth"],
            "settings": {"size": [1, 2]}, "labels": labels or {"size": "Colony size"}}
    return af.run_figures([spec], ns, _CallTable(), str(tmp_path / "figures"))[0]


def test_a_panel_declared_with_fig_panel_keeps_its_setting_in_the_record_and_fi_writes_its_title(tmp_path, frame5):
    def body(fig, table):
        fig.panel(0, size=1)
        fig.panel(1, size=2)
        _both_panels(fig, table)

    out = _hold_run(tmp_path, body)
    assert out["status"] == "ok", out
    panels = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]
    assert [p["hold"] for p in panels] == [{"size": 1}, {"size": 2}]
    assert [p["title"] for p in out["panels"]] == ["Colony size = 1", "Colony size = 2"]


def test_a_title_typed_after_fig_panel_is_replaced_by_fi_and_a_panel_without_a_hold_has_none(tmp_path, frame5):
    def body(fig, table):
        _both_panels(fig, table)
        fig.panel(0, size=2)
        fig.axes[0].set_title("a curve this panel does not draw")
        fig.axes[1].set_title("typed")

    out = _hold_run(tmp_path, body)
    assert out["status"] == "ok", out
    panels = json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"]
    assert panels[0]["hold"] == {"size": 2} and "hold" not in panels[1]
    assert [p["title"] for p in out["panels"]] == ["Colony size = 2"]


def test_the_title_of_a_hold_is_written_by_finish_from_the_plans_label(tmp_path, frame5):
    seen = {}

    def body(fig, table):
        fig.panel(0, size=1)
        _both_panels(fig, table)
        fig.axes[0].set_title("typed late")
        fig.finish({"size": "Colony size"}, {})
        seen["title"], seen["other"] = fig.axes[0].get_title(), fig.axes[1].get_title()

    ns = types.SimpleNamespace(figure_g=body)
    spec = {"id": "g", "file": "g.png", "panels": 2, "x": "size", "series": "fast", "shows": ["growth"],
            "settings": {"size": [1, 2]}, "labels": {"size": "Colony size"}}
    af.run_figures([spec], ns, _CallTable(), str(tmp_path / "figures"))
    assert seen == {"title": "Colony size = 1", "other": ""}


def test_fig_panel_with_a_setting_or_value_the_plan_does_not_have_is_refused_on_the_calls_frame(tmp_path, frame5):
    out = _hold_run(tmp_path, lambda fig, table: fig.panel(0, speed=5))
    assert out["status"] == "failed" and "UnknownSetting" in out["why"] and "size = 1, 2" in out["why"]
    out = _hold_run(tmp_path, lambda fig, table: fig.panel(1, size=999))
    assert out["status"] == "failed" and "UnknownSetting" in out["why"] and "size = 1, 2" in out["why"]
