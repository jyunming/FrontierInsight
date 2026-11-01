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

    out = _shown(tmp_path, figure_x, table=_real_table(declared=(key,)))
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
    assert "The results that can be read: r1, r2." in out["why"]


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
