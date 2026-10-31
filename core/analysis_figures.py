"""The figures of the analysis FI writes: FI creates each figure, the model's body only draws on the axes it is given, and FI
saves it (through the figure recorder the house style installs) and closes it. A search's own figures are FI's, drawn from
FI's record of the search, never from a model's script.

What the writer of ``experiment.py`` (``core/analysis_frame.py``) calls, all of it here, in this file's own words:

* ``run_figures(figures, namespace, table, out_dir) -> list[dict]`` draws every declared figure. ``figures`` is the plan's
  ``contract.Figure`` list as dicts (``id``, ``file``, ``panels``, ``x``, ``series``, ``shows``); ``namespace`` is the module
  of ``derive.py`` (a body is its attribute ``figure_<id>``); ``table`` is the ``Table`` of the run. One figure's failure never
  stops the others. Each outcome is ``{"id", "file", "status": ok | not_written | failed | refused | fi_fault, "why", "said"}``; ``dropped`` is set after the run, by ``result_checks`` (an empty figure FI took out of the results), and is never ok. While a
  body runs FI's own figure is the only one it may draw on: saving or showing a figure, or making one, raises a plain error
  at that moment (``refused``), whatever way the body reaches it, and a file the body wrote in the figure folder is removed and refused. A fault of FI's own lines is ``fi_fault``, never the model's. The file of an earlier run is removed before a figure is
  drawn, so a figure that fails is never the old one.
* ``search_figures(root, out_dir) -> list[dict]`` draws FI's own figures of a search from ``results/best_design.json`` and
  ``raw/optimisation_ledger.jsonl`` (the files ``FI_BEST_DESIGN`` and ``FI_OPTIMISATION`` name): the best objective so far
  against the evaluation number (one line for each compared alternative, drawn by the rule FI's own search picks its best by,
  so a line ends at the best FI recorded for that alternative), the baseline against the best design (a table), and the scan
  map when the search began with a coarse scan. No best design, or a check that did not run, is said in a plain sentence and
  never a crash. Outcomes are the same shape; ``status`` may also be ``skipped``, or ``fi_fault`` when FI's own record and
  its own figure disagree or a drawing of FI's own stopped: that is FI's bug, never the model's.
* ``said(outcomes) -> str`` is the plain sentence for the run's log; ``exit_code(outcomes) -> int`` is 1 when a figure body
  failed or was refused, 0 otherwise (a stub not written yet, and a search with no best design, are said and are not failures);
  ``fi_faults(outcomes) -> list[str]`` lists FI's own faults, which the run reports plainly and never sends to the model.

Standard library only, with matplotlib read when a figure is drawn and nothing of FI's imported, so this file's text can be
carried into the quest's code folder the way ``core/frame_table.py`` is; the house style (``core/plot_style.py``) is already
on matplotlib in the quest's interpreter, so every figure here is drawn in it without naming a colour or a font. The two small
rules copied from ``core/optimise_search.py`` (``level_key``, ``level_says``) are held equal to it by the tests."""

from __future__ import annotations

import contextlib
import json
import math
import os
import traceback
from typing import Any

BEST_FILE = os.path.join("results", "best_design.json")
LEDGER_FILE = os.path.join("raw", "optimisation_ledger.jsonl")
PROGRESS_FILE = "search_progress.png"
BEST_VS_BASELINE_FILE = "search_best_vs_baseline.png"
SCAN_FILE = "search_scan_map.png"


# ---- the runtime guard ------------------------------------------------------------------------------------------------------
class FigureMisuse(Exception):
    """A figure body saved, showed or made a figure itself (FI does that)."""


def _misuse_text(name: str, what: str) -> str:
    return ("`%s` tried to %s itself; FI creates the figure, saves it and closes it. Draw only on the axes the function is given "
            "(`fig.axes`)." % (name, what))


@contextlib.contextmanager
def _guard(name: str, plt: Any):
    """While a body runs, saving, showing and making figures raise ``FigureMisuse`` naming what the body did; what it made
    anyway (a figure made through a name bound before the guard) is closed afterwards and counted. Yields the list of what the
    body did. Patches are made on matplotlib's classes and ``pyplot`` and put back as they were (the recorder's ``savefig``
    of the house style included) when the body ends."""
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.figure import Figure

    did: list[str] = []

    def refuse(what: str, original=None):
        def blocked(*args, **kwargs):
            if original is not None:  # pyplot.sca and the like name a figure that is already open: that is not making one
                num = args[0] if args else kwargs.get("num")
                if (isinstance(num, Figure) and num.number in plt.get_fignums()) or (isinstance(num, int) and num in plt.get_fignums()):
                    return original(*args, **kwargs)
            if what not in did:
                did.append(what)
            raise FigureMisuse(_misuse_text(name, what))
        return blocked

    targets = [(Figure, "savefig", "save a figure"), (Figure, "show", "show a figure"),
               (FigureCanvasBase, "print_figure", "save a figure"),
               (plt, "savefig", "save a figure"), (plt, "show", "show a figure"), (plt, "figure", "make a figure"),
               (plt, "subplots", "make a figure"), (plt, "close", "close a figure")]
    saved = [(owner, attr, getattr(owner, attr)) for owner, attr, _ in targets]
    before = set(plt.get_fignums())
    for owner, attr, what in targets:
        setattr(owner, attr, refuse(what, getattr(owner, attr) if (owner, attr) == (plt, "figure") else None))
    try:
        yield did
    finally:
        for owner, attr, original in saved:
            setattr(owner, attr, original)
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
            if "make a figure" not in did:
                did.append("make a figure")


# ---- a declared figure ------------------------------------------------------------------------------------------------------
def _pyplot():
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt
    return plt


def _new_figure(panels: int):
    """A figure of ``panels`` axes at the house style's size (one row up to two panels, then two columns)."""
    plt = _pyplot()
    panels = max(1, int(panels or 1))
    columns = panels if panels <= 2 else 2
    rows = int(math.ceil(panels / columns))
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(rows, columns, figsize=(width * (1 if columns == 1 else 1.5), height * rows), squeeze=False)
    flat = list(axes.flat)
    for extra in flat[panels:]:
        extra.set_visible(False)
    return plt, fig, flat[:panels]


def _last_frame(text: str) -> str:
    """``derive.py line N`` of the last frame of a traceback that is in the model's body, else the last frame there is."""
    frames = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("File ")]
    mine = [f for f in frames if "derive.py" in f]
    pick = (mine or frames or [""])[-1]
    try:
        path = pick.split('"')[1]
        line = pick.split("line ")[1].split(",")[0]
        return "%s line %s" % (os.path.basename(path), line)
    except IndexError:
        return pick


def _outcome(fid: str, file: str, status: str, why: str = "") -> dict[str, Any]:
    said = {"ok": "Figure `%s` was drawn (%s)." % (fid, file),
            "not_written": "Figure `%s` was not drawn: its `figure_%s` is not written yet." % (fid, fid),
            "failed": "Figure `%s` was not drawn: %s" % (fid, why),
            "refused": "Figure `%s` was not drawn: %s" % (fid, why)}.get(status, why)
    return {"id": fid, "file": file, "status": status, "why": why, "said": said}


def _remove(out_dir: str, file: str) -> None:
    """The file of an earlier run goes first: a figure that is not drawn now must not leave the old one for the paper."""
    try:
        os.remove(os.path.join(out_dir, file))
    except OSError:
        pass


def _empty_panels(axes: list[Any]) -> list[int]:
    return [i + 1 for i, a in enumerate(axes) if a.get_visible() and not (a.has_data() or a.texts)]


def _folder(out_dir: str) -> dict[str, tuple[int, int]]:
    """Every file in the figure folder with its time and size: what a body must leave as it found it."""
    seen = {}
    try:
        for entry in os.scandir(out_dir):
            if entry.is_file():
                info = entry.stat()
                seen[entry.name] = (info.st_mtime_ns, info.st_size)
    except OSError:
        pass
    return seen


def _last_read_error(table: Any) -> str:
    """The most recent exception a read of the table raised, cut to its type and first clause (an ``UnknownResult`` lists the
    declared results, so only its type is kept: this text must not name a result)."""
    errors = getattr(table, "last_error", None)
    last = list(errors.values())[-1] if isinstance(errors, dict) and errors else ""
    if last.startswith("UnknownResult"):
        return "UnknownResult: a result the plan does not declare"
    if last.startswith(("ValueError", "KeyError")):  # `levels` names the result in its ValueError: only the type is kept
        return last.split(":", 1)[0]
    return last[:240]


def _drew_nothing(spec: dict[str, Any], table: Any) -> str:
    """Why a figure drew nothing from the trial record (``""`` when some read of a result gave data, or when the table does not
    record its reads). FI's table notes each read that returned data, so this reads no text of the body. It names no result:
    the line goes to a model, and a line naming a left-out quantity is taken out of what a model reads."""
    gave = getattr(table, "gave_values", None)
    shows = spec.get("shows")
    if not isinstance(gave, (set, frozenset)) or gave or not shows or not isinstance(shows, (list, tuple)):
        return ""  # a figure that declares no result draws what it says it draws (a layout, a reference curve): not checked
    text = "`figure_%s` drew nothing from the trial record: no read of a result returned a value." % spec.get("id")
    err = _last_read_error(table)
    return text + (" The last read raised %s." % err if err else "")


def _draw_one(spec: dict[str, Any], body: Any, table: Any, out_dir: str) -> dict[str, Any]:
    """Draw one declared figure. A fault of the model's body is ``failed`` / ``refused`` (its repair); a fault of FI's own lines
    (making the figure, handing it over, the folder) is ``fi_fault`` and is never sent to the model."""
    fid, file = str(spec.get("id")), str(spec.get("file") or "%s.png" % spec.get("id"))
    plt = fig = None
    try:
        _remove(out_dir, file)
        if body is None:
            return _outcome(fid, file, "not_written")
        name = getattr(body, "__name__", "figure_%s" % fid)
        plt, fig, axes = _new_figure(spec.get("panels") or 1)
        handle = _figure_handle(axes, spec, file)
        reset = getattr(table, "reset_reads", None)
        if callable(reset):
            reset()  # what the reads of this one body give is recorded from here
        plt.sca(axes[0])  # a body that draws with the bare pyplot calls still draws on its first axes
        os.makedirs(out_dir, exist_ok=True)
        folder = _folder(out_dir)
    except Exception as err:
        if plt is not None and fig is not None:
            plt.close(fig)
        return _outcome(fid, file, "fi_fault", "FI could not make the figure `%s` (%s: %s)." % (fid, type(err).__name__, err))
    did: list[str] = []
    try:
        try:
            with _guard(name, plt) as did:
                body(handle, table)
        except FigureMisuse as err:
            return _outcome(fid, file, "refused", str(err))
        except Exception as err:
            if type(err).__name__ == "NotYet":
                return _outcome(fid, file, "not_written")
            if did:
                return _outcome(fid, file, "refused", _misuse_text(name, did[0]))
            where = _last_frame(traceback.format_exc())
            return _outcome(fid, file, "failed", "`figure_%s` stopped with %s: %s (%s)." % (fid, type(err).__name__, err, where))
        if did:
            return _outcome(fid, file, "refused", _misuse_text(name, did[0]))
        after = _folder(out_dir)
        wrote = sorted(n for n, v in after.items() if folder.get(n) != v)
        if wrote:  # a body that reached a way of saving the guard does not know: whatever it wrote is removed
            for n in wrote:
                _remove(out_dir, n)
            return _outcome(fid, file, "refused", "`%s` wrote %s itself; FI creates the figure, saves it and closes it. Draw only on "
                            "the axes the function is given (`fig.axes`)." % (name, ", ".join(wrote)))
        empty = _empty_panels(axes)
        if empty:
            return _outcome(fid, file, "failed", "`figure_%s` drew nothing on panel %s." % (fid, " and ".join(map(str, empty))))
        none_drawn = _drew_nothing(spec, table)
        if none_drawn:
            return _outcome(fid, file, "failed", none_drawn)
        panel_record = _set_panel_titles(handle, axes, spec.get("labels") or {})
        try:
            fig.savefig(os.path.join(out_dir, file))  # the recorder of the house style notes what the figure draws
        except Exception as err:
            return _outcome(fid, file, "failed", "`figure_%s` drew something that cannot be saved (%s: %s)." % (fid, type(err).__name__, err))
        return dict(_outcome(fid, file, "ok"), panels=panel_record)
    except Exception as err:
        return _outcome(fid, file, "fi_fault", "FI stopped while drawing the figure `%s` (%s: %s)." % (fid, type(err).__name__, err))
    finally:
        plt.close(fig)


def _figure_handle(axes: list[Any], spec: dict[str, Any], file: str) -> Any:
    """What the body is given: the frame's ``Figure`` when it is carried beside this file, else a plain holder."""
    try:
        from fi_frame import Figure  # the frame's runtime, written beside this file in the quest's code folder
    except ImportError:
        class Figure:  # type: ignore[no-redef]
            def __init__(self, axes, x=None, series=None, shows=None, file=None):
                self.axes, self.x, self.series, self.shows, self.file = list(axes), x, series, shows, file
                self.declared = {}

            def panel(self, which, **settings):
                self.declared[which if isinstance(which, int) else self.axes.index(which)] = dict(settings)
    handle = Figure(axes, x=spec.get("x"), series=spec.get("series"), shows=spec.get("shows"), file=file)
    if not hasattr(handle, "panel"):  # a runtime written before `panel` existed (a resumed quest keeps its own)
        handle.declared = {}
        handle.panel = lambda which, **settings: handle.declared.__setitem__(
            which if isinstance(which, int) else handle.axes.index(which), dict(settings))
    return handle


def _panel_title(settings: dict[str, Any], labels: dict[str, Any]) -> str:
    """The title FI writes for a panel from the settings the body declared for it: ``label = value``, the label the plan
    gives the setting, else its name."""
    return ", ".join("%s = %s" % (labels.get(k) or k, "%.4g" % v if isinstance(v, float) else v) for k, v in settings.items())


def _set_panel_titles(handle: Any, axes: list[Any], labels: dict[str, Any]) -> list[dict[str, Any]]:
    """Give each panel the body declared its settings for FI's title in place of whatever the body typed there (the body's
    title can name a line the panel does not draw); a panel it declared nothing for keeps the body's own. Returns the record."""
    record = []
    for index, settings in sorted((getattr(handle, "declared", None) or {}).items()):
        if not isinstance(index, int) or not 0 <= index < len(axes) or not isinstance(settings, dict) or not settings:
            continue
        title = _panel_title(settings, labels)
        for loc in ("left", "center", "right"):
            axes[index].set_title("", loc=loc)
        axes[index].set_title(title)
        record.append({"panel": index + 1, "settings": settings, "title": title})
    return record


def run_figures(figures: list[dict[str, Any]], namespace: Any, table: Any, out_dir: str) -> list[dict[str, Any]]:
    """Draw every declared figure from the body ``figure_<id>`` of ``namespace`` (a module, or a dict); a figure's failure is its
    own and the others are still drawn."""
    out = []
    for spec in figures or []:
        name = "figure_%s" % spec.get("id")
        body = namespace.get(name) if isinstance(namespace, dict) else getattr(namespace, name, None)
        out.append(_draw_one(spec, body, table, out_dir))
    return out


# ---- FI's own figures of a search ---------------------------------------------------------------------------------------------
def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any) -> str:
    return "%.4g" % value if _number(value) else "none"


def level_key(level: Any) -> str:
    """A short text naming an alternative (the rule of ``optimise_search.level_key``)."""
    return json.dumps(level, sort_keys=True) if level else ""


def level_says(level: Any) -> str:
    """The alternative in words (the rule of ``optimise_search.level_says``): empty for no alternative."""
    return ", ".join("%s = %s" % (k, v) for k, v in level.items()) if isinstance(level, dict) else ""


def _read_record(root: str) -> dict[str, Any]:
    """FI's record of the search: the best-design file (``{}`` when missing), the ledger's head line (the objective, with its
    direction) and the evaluations, kept light (the design of a row is kept for the scan only; a real ledger is large)."""
    best_path = os.path.join(root, os.environ.get("FI_BEST_DESIGN") or BEST_FILE)
    ledger_path = os.path.join(root, os.environ.get("FI_OPTIMISATION") or LEDGER_FILE)
    try:
        with open(best_path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = {}
    head: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    scan: list[dict[str, Any]] = []
    try:
        with open(ledger_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("event") == "search":
                    head = row
                    continue
                if row.get("event", "evaluation") != "evaluation" or row.get("status", "ok") != "ok" or not _number(row.get("objective")):
                    continue
                light = {"n": row.get("n"), "objective": row["objective"], "start": row.get("start"), "stage": row.get("stage"),
                         "feasible": bool(row.get("feasible", True)), "level": row.get("level")}
                rows.append(light)
                if row.get("stage") == "scan":
                    scan.append(dict(light, design=row.get("design")))
    except OSError:
        pass
    record = record if isinstance(record, dict) else {}
    objective = record.get("objective") if isinstance(record.get("objective"), dict) else {}
    if not objective.get("direction") and isinstance(head.get("objective"), dict):
        objective = head["objective"]
    return {"record": record, "rows": rows, "scan": scan, "objective": objective}


def _best_so_far(rows: list[dict[str, Any]], sign: float) -> list[tuple[float, float]]:
    """The best objective so far over ``rows`` by the rule FI's own search picks its best by (``optimise_search``: a row that
    succeeded and is feasible, the better objective, the earlier row on a tie): the points where it changed, and the last row
    so the line runs to the end. ``sign`` is 1 to minimise, -1 to maximise."""
    pts: list[tuple[float, float]] = []
    last = None
    for i, row in enumerate(sorted(rows, key=lambda r: (r["n"] if _number(r.get("n")) else math.inf))):
        x = float(row["n"]) if _number(row.get("n")) else float(i + 1)
        last = x
        if row["feasible"] and (not pts or sign * row["objective"] < sign * pts[-1][1]):
            pts.append((x, float(row["objective"])))
    if pts and last is not None and last != pts[-1][0]:
        pts.append((last, pts[-1][1]))
    return pts


def _alternatives(data: dict[str, Any], sign: float) -> list[dict[str, Any]]:
    """Each compared alternative (one when the plan compares none): its label, the best-so-far line over its rows, one thinner line
    for each of its starts, and the best FI recorded for it (``compared.levels`` of the best-design file; ``best`` for a search of
    one alternative)."""
    order: dict[str, list[dict[str, Any]]] = {}
    levels: dict[str, Any] = {}
    for row in data["rows"]:
        key = level_key(row.get("level"))
        order.setdefault(key, []).append(row)
        levels.setdefault(key, row.get("level"))
    record = data["record"]
    recorded: dict[str, Any] = {}
    compared = record.get("compared") if isinstance(record.get("compared"), dict) else {}
    same: dict[str, str] = {}
    for entry in compared.get("levels") or []:
        if isinstance(entry, dict) and isinstance(entry.get("same_as"), dict):
            same[level_key(entry.get("level"))] = level_says(entry["same_as"])  # only probes; its best is the first one's
        elif isinstance(entry, dict):
            best = entry.get("best") if isinstance(entry.get("best"), dict) else {}
            recorded[level_key(entry.get("level"))] = best.get("objective") if _number(best.get("objective")) else None
    if not recorded and "" in order and "best" in record:
        best = record["best"] if isinstance(record.get("best"), dict) else {}
        recorded[""] = best.get("objective") if _number(best.get("objective")) else None
    out = []
    for key, rows in order.items():
        starts: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("start") is not None and row.get("stage") not in ("baseline", "scan"):
                starts.setdefault(row["start"], []).append(row)
        out.append({"key": key, "label": level_says(levels[key]) or "search", "line": _best_so_far(rows, sign),
                    "starts": [_best_so_far(v, sign) for _, v in sorted(starts.items(), key=lambda kv: str(kv[0]))],
                    "recorded": recorded.get(key, "unknown"), "same_as": same.get(key)})
    return out


def _disagrees(alternative: dict[str, Any]) -> bool:
    """Whether the line drawn for an alternative does not end at the best FI recorded for it (nothing recorded: no line)."""
    recorded, line = alternative["recorded"], alternative["line"]
    if recorded == "unknown":
        return False
    if recorded is None:
        return bool(line)
    return not line or not math.isclose(line[-1][1], recorded, rel_tol=1e-9, abs_tol=1e-12)


def _progress_figure(data: dict[str, Any], out_dir: str) -> dict[str, Any]:
    fid, file = "search_progress", PROGRESS_FILE
    _remove(out_dir, file)
    objective = data["objective"]
    direction = objective.get("direction")
    if not data["rows"]:
        return _outcome(fid, file, "skipped", "The search's record holds no evaluation, so the search's progress is not drawn.")
    if direction not in ("minimise", "maximise"):
        return _outcome(fid, file, "skipped", "The search's record does not say whether the objective is made higher or lower, so "
                        "the best design so far is not drawn.")
    sign = 1.0 if direction == "minimise" else -1.0
    alternatives = _alternatives(data, sign)
    for a in alternatives:
        if a["same_as"]:
            a["line"], a["starts"] = [], []  # an alternative that was not searched has probe rows only: not drawn
    drawn = [a for a in alternatives if a["line"]]
    wrong = [a["label"] for a in alternatives if not a["same_as"] and _disagrees(a)]
    if wrong:
        return _outcome(fid, file, "fi_fault", "FI's drawing of the search does not end at the best FI recorded for: %s." % "; ".join(wrong))
    if not drawn:
        return _outcome(fid, file, "skipped", "No design met every limit, so there is no best design so far to draw.")
    plt, fig, (ax,) = _new_figure(1)
    try:
        for a in drawn:
            line, = ax.step([p[0] for p in a["line"]], [p[1] for p in a["line"]], where="post",
                            label=a["label"] if a["label"] != "search" else "best so far")
            for one in a["starts"]:
                if one:
                    ax.step([p[0] for p in one], [p[1] for p in one], where="post", color=line.get_color(), alpha=0.35, linewidth=0.8)
        unit = objective.get("unit")
        ax.set_xlabel("evaluation number")
        ax.set_ylabel("best %s so far%s" % (objective.get("quantity") or "objective", " (%s)" % unit if unit else ""))
        if len(drawn) <= 6:
            ax.legend()
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, file))
        out = _outcome(fid, file, "ok")
        none = [a["label"] for a in alternatives if not a["line"] and not a["same_as"]]
        same = ["%s gave the same results as %s, so it is not drawn" % (a["label"], a["same_as"]) for a in alternatives if a["same_as"]]
        notes = (["No design met every limit for: %s." % "; ".join(none)] if none else []) + [s + "." for s in same]
        if notes:
            out["note"] = " ".join(notes)
        return out
    finally:
        plt.close(fig)


def best_sentence(record: dict[str, Any]) -> str:
    """One plain sentence on whether the search found a best design and whether it was checked at finer settings."""
    best = record.get("best") if isinstance(record.get("best"), dict) else {}
    parts = []
    if not best or best.get("design") is None or not _number(best.get("objective")):
        parts.append("FI found no design that meets every limit, so there is no best design to show.")
    check = record.get("check") if isinstance(record.get("check"), dict) else {}
    if record.get("checked_at_finer_settings") is not True:
        parts.append(str(check.get("says") or "The best design was not checked at finer numerical settings."))
    return " ".join(parts)


def _best_vs_baseline_figure(data: dict[str, Any], out_dir: str) -> dict[str, Any]:
    fid, file = "search_best_vs_baseline", BEST_VS_BASELINE_FILE
    _remove(out_dir, file)
    record = data["record"]
    objective = data["objective"]
    base = record.get("baseline") if isinstance(record.get("baseline"), dict) else {}
    baseline = base.get("objective")
    best = record.get("best") if isinstance(record.get("best"), dict) else {}
    best_value = best.get("objective") if _number(best.get("objective")) and best.get("design") is not None else None
    if not _number(baseline) and best_value is None:
        return _outcome(fid, file, "skipped", "The search's record holds neither a baseline nor a best design, so the comparison is not drawn.")
    improvement = (record.get("improvement") or {}).get("value") if isinstance(record.get("improvement"), dict) else None
    unit = objective.get("unit") or ""
    plt, fig, (ax,) = _new_figure(1)
    try:
        ax.axis("off")
        name = "%s%s" % (objective.get("quantity") or "objective", " (%s)" % unit if unit else "")
        breaks = base.get("feasible") is False
        cells = [[name, _fmt(baseline) + (" (breaks a limit)" if breaks else ""),
                  _fmt(best_value) if best_value is not None else "no best design", _fmt(improvement) if best_value is not None else "none"]]
        table = ax.table(cellText=cells, colLabels=["", "baseline", "best design", "improvement"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.scale(1, 1.8)
        note = best_sentence(record)
        if note:
            ax.text(0.5, 0.12, note, ha="center", va="top", wrap=True, transform=ax.transAxes)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, file))
        out = _outcome(fid, file, "ok")
        out["note"] = note
        return out
    finally:
        plt.close(fig)


def _scalar_levels(design: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in (design or {}).items() if _number(v)} if isinstance(design, dict) else {}


def _scan_figure(data: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """The coarse scan's map: the objective at each scanned design, over the two settings that vary most (one setting: a line)."""
    fid, file = "search_scan_map", SCAN_FILE
    _remove(out_dir, file)
    points = [(_scalar_levels(r["design"]), r["objective"]) for r in data["scan"]]
    points = [p for p in points if p[0]]
    count = {k: len({p[k] for p, _ in points if k in p}) for k in {k for p, _ in points for k in p}}
    names = sorted((k for k, c in count.items() if c > 1), key=lambda k: (-count[k], k))
    if not points or not names:
        return _outcome(fid, file, "skipped", "The coarse scan has no setting that changed, so its map is not drawn.")
    plt, fig, (ax,) = _new_figure(1)
    try:
        if len(names) == 1:
            pts = sorted((p[names[0]], v) for p, v in points)
            ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o")
            ax.set_xlabel(names[0])
            ax.set_ylabel("objective")
        else:
            x, y = names[0], names[1]
            shown = [(p[x], p[y], v) for p, v in points if x in p and y in p]
            sc = ax.scatter([s[0] for s in shown], [s[1] for s in shown], c=[s[2] for s in shown], s=60)
            fig.colorbar(sc, ax=ax, label="objective")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, file))
        return _outcome(fid, file, "ok")
    finally:
        plt.close(fig)


def search_figures(root: str, out_dir: str) -> list[dict[str, Any]]:
    """FI's own figures of a search, from FI's files under ``root``; each one's failure is its own, and is FI's (``fi_fault``)."""
    data = _read_record(str(root))
    out = []
    for fid, file, draw in (("search_progress", PROGRESS_FILE, lambda: _progress_figure(data, out_dir)),
                            ("search_best_vs_baseline", BEST_VS_BASELINE_FILE, lambda: _best_vs_baseline_figure(data, out_dir)),
                            ("search_scan_map", SCAN_FILE, lambda: _scan_figure(data, out_dir) if data["scan"] else None)):
        try:
            result = draw()
        except Exception as err:
            result = _outcome(fid, file, "fi_fault", "FI's drawing of `%s` stopped with %s: %s." % (file, type(err).__name__, err))
        if result:
            out.append(result)
    return out


# ---- what is said and the exit code ---------------------------------------------------------------------------------------------
def said(outcomes: list[dict[str, Any]]) -> str:
    """The figure step in plain words: how many were drawn and, one sentence each, every one that was not and why."""
    if not outcomes:
        return ""
    drawn = [o for o in outcomes if o.get("status") == "ok"]
    lines = ["%d of %d figure(s) drawn." % (len(drawn), len(outcomes))]
    lines += [o["said"] for o in outcomes if o.get("status") != "ok" and o.get("said")]
    lines += [o["note"] for o in outcomes if o.get("note")]
    return " ".join(lines)


def exit_code(outcomes: list[dict[str, Any]]) -> int:
    """1 when a model's figure body failed or was refused (its repair); FI's own faults are not counted (``fi_faults``)."""
    return 1 if any(o.get("status") in ("failed", "refused") for o in outcomes) else 0


def fi_faults(outcomes: list[dict[str, Any]]) -> list[str]:
    """What FI's own figures could not do because of FI's own record or code: said plainly, never sent to the model."""
    return [o["said"] for o in outcomes if o.get("status") == "fi_fault"]
