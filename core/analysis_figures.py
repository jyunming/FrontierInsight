"""The figures of the analysis FI writes: FI creates each figure, the model's body only draws on the axes it is given, and FI
saves it (through the figure recorder the house style installs) and closes it. A search's own figures are FI's, drawn from
FI's record of the search, never from a model's script.

What the writer of ``experiment.py`` (``core/analysis_frame.py``) calls, all of it here, in this file's own words:

* ``run_figures(figures, namespace, table, out_dir) -> list[dict]`` draws every declared figure. ``figures`` is the plan's
  ``contract.Figure`` list as dicts (``id``, ``file``, ``panels``, ``x``, ``series``, ``shows``); ``namespace`` is the module
  of ``derive.py`` (a body is its attribute ``figure_<id>``); ``table`` is the ``Table`` of the run. One figure's failure never
  stops the others. Each outcome is ``{"id", "file", "status": ok | not_written | failed | refused, "why", "said"}``.
* ``search_figures(root, out_dir) -> list[dict]`` draws FI's own figures of a search from ``results/best_design.json`` and
  ``raw/optimisation_ledger.jsonl`` (the files ``FI_BEST_DESIGN`` and ``FI_OPTIMISATION`` name): the best objective so far
  against the evaluation number (one line per start), the baseline against the best design (a table), and the scan map when
  the search began with a coarse scan. No best design, or a check that did not run, is said in a plain sentence and never
  a crash. Outcomes are the same shape; ``status`` may also be ``skipped``.
* ``figure_body_problem(source, name=None) -> str`` is the static refusal: ``""``, or the plain reason a ``figure_<id>`` body
  (every one in ``source`` when ``name`` is None) is refused for saving, showing, closing or opening a figure itself.
* ``said(outcomes) -> str`` is the plain sentence for the run's log; ``exit_code(outcomes) -> int`` is 1 when a figure body
  failed or was refused, 0 otherwise (a stub not written yet, and a search with no best design, are said and are not failures).

Standard library only, with matplotlib read when a figure is drawn and nothing of FI's imported, so this file's text can be
carried into the quest's code folder the way ``core/frame_table.py`` is; the house style (``core/plot_style.py``) is already
on matplotlib in the quest's interpreter, so every figure here is drawn in it without naming a colour or a font."""

from __future__ import annotations

import ast
import json
import math
import os
import traceback
from typing import Any

#: Calls a ``figure_<id>`` body may not make: FI creates, saves, shows and closes the figure.
_REFUSED_ATTRIBUTES = ("savefig", "show", "close", "figure", "subplots", "subplot", "gcf", "clf")
_REFUSED_NAMES = ("savefig", "show", "subplots")

BEST_FILE = os.path.join("results", "best_design.json")
LEDGER_FILE = os.path.join("raw", "optimisation_ledger.jsonl")
PROGRESS_FILE = "search_progress.png"
BEST_VS_BASELINE_FILE = "search_best_vs_baseline.png"
SCAN_FILE = "search_scan_map.png"


# ---- the static refusal ----------------------------------------------------------------------------------------------------
def figure_body_problem(source: str, name: str | None = None) -> str:
    """The plain reason a ``figure_<id>`` body in ``source`` is refused, ``""`` when none is. A body draws on the axes it is
    given; it does not create, save, show or close a figure."""
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return ""  # a syntax error is the gate's to say, not this check's
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("figure_") or (name and node.name != name):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            used = fn.attr if isinstance(fn, ast.Attribute) and fn.attr in _REFUSED_ATTRIBUTES else (
                fn.id if isinstance(fn, ast.Name) and fn.id in _REFUSED_NAMES else "")
            if used:
                return ("`%s` calls `%s`; FI creates the figure, saves it and closes it. Draw only on the axes the function is "
                        "given (`fig.axes`)." % (node.name, used))
    return ""


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
            "refused": "Figure `%s` was not drawn: %s" % (fid, why),
            "skipped": why}.get(status, why)
    return {"id": fid, "file": file, "status": status, "why": why, "said": said}


def _draw_one(spec: dict[str, Any], body: Any, table: Any, out_dir: str, source: str | None) -> dict[str, Any]:
    fid, file = str(spec.get("id")), str(spec.get("file") or "%s.png" % spec.get("id"))
    if body is None:
        return _outcome(fid, file, "not_written")
    if source is None:
        try:
            import inspect
            import textwrap
            source = textwrap.dedent(inspect.getsource(body))
        except (OSError, TypeError):
            source = ""
    problem = figure_body_problem(source, getattr(body, "__name__", None))
    if problem:
        return _outcome(fid, file, "refused", problem)
    plt = fig = None
    try:
        plt, fig, axes = _new_figure(spec.get("panels") or 1)
        handle = _figure_handle(axes, spec, file)
        plt.sca(axes[0])  # a body that draws with the bare pyplot calls still draws on its first axes
        body(handle, table)
        if not any(a.has_data() or a.texts for a in axes):
            return _outcome(fid, file, "failed", "`figure_%s` drew nothing on its axes." % fid)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, file))  # the recorder of the house style notes what the figure draws
        return _outcome(fid, file, "ok")
    except Exception as err:
        if type(err).__name__ == "NotYet":
            return _outcome(fid, file, "not_written")
        where = _last_frame(traceback.format_exc())
        return _outcome(fid, file, "failed", "`figure_%s` stopped with %s: %s (%s)." % (fid, type(err).__name__, err, where))
    finally:
        if plt is not None and fig is not None:
            plt.close(fig)


def _figure_handle(axes: list[Any], spec: dict[str, Any], file: str) -> Any:
    """What the body is given: the frame's ``Figure`` when it is carried beside this file, else a plain holder."""
    try:
        from fi_frame import Figure  # the frame's runtime, written beside this file in the quest's code folder
        return Figure(axes, x=spec.get("x"), series=spec.get("series"), shows=spec.get("shows"), file=file)
    except Exception:
        class _Figure:
            pass
        handle = _Figure()
        handle.axes, handle.x, handle.series = axes, spec.get("x"), spec.get("series")
        handle.shows, handle.file = spec.get("shows"), file
        return handle


def run_figures(figures: list[dict[str, Any]], namespace: Any, table: Any, out_dir: str, source: str | None = None) -> list[dict[str, Any]]:
    """Draw every declared figure from the body ``figure_<id>`` of ``namespace`` (a module, or a dict); a figure's failure is its
    own and the others are still drawn. ``source`` is the text of ``derive.py`` when the caller has it (else each body's own)."""
    out = []
    for spec in figures or []:
        name = "figure_%s" % spec.get("id")
        body = namespace.get(name) if isinstance(namespace, dict) else getattr(namespace, name, None)
        out.append(_draw_one(spec, body, table, out_dir, source))
    return out


# ---- FI's own figures of a search ---------------------------------------------------------------------------------------------
def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any) -> str:
    return "%.4g" % value if _number(value) else "none"


def _read_record(root: str) -> dict[str, Any]:
    """FI's record of the search: the best-design file (``{}`` when missing) and the ledger's evaluations, kept light (the design
    of each row is dropped; a real ledger is large)."""
    best_path = os.path.join(root, os.environ.get("FI_BEST_DESIGN") or BEST_FILE)
    ledger_path = os.path.join(root, os.environ.get("FI_OPTIMISATION") or LEDGER_FILE)
    try:
        with open(best_path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        record = {}
    rows: list[dict[str, Any]] = []
    scan: list[dict[str, Any]] = []
    try:
        with open(ledger_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or row.get("event", "evaluation") != "evaluation" or row.get("status", "ok") != "ok":
                    continue
                if not _number(row.get("objective")):
                    continue
                if row.get("stage") == "scan":
                    scan.append({"design": row.get("design"), "objective": row["objective"], "level": row.get("level")})
                else:
                    rows.append({"n": row.get("n"), "objective": row["objective"], "start": row.get("start"), "stage": row.get("stage"),
                                 "feasible": row.get("feasible", True)})
    except OSError:
        pass
    return {"record": record if isinstance(record, dict) else {}, "rows": rows, "scan": scan}


def _best_so_far(rows: list[dict[str, Any]], direction: str) -> dict[str, list[tuple[float, float]]]:
    """Per start (the global stage, which has none, is one line), the points where the best objective so far changed, and the
    last evaluation so the line runs to the end."""
    better = (lambda a, b: a > b) if direction != "minimise" else (lambda a, b: a < b)
    lines: dict[str, list[tuple[float, float]]] = {}
    last: dict[str, tuple[float, float]] = {}
    for i, row in enumerate(rows):
        label = "global search" if row.get("start") is None else "start %s" % row["start"]
        x = float(row["n"]) if _number(row.get("n")) else float(i + 1)
        pts = lines.setdefault(label, [])
        if not pts or better(row["objective"], pts[-1][1]):
            pts.append((x, float(row["objective"])))
        last[label] = (x, pts[-1][1])
    for label, pts in lines.items():
        if last[label][0] != pts[-1][0]:
            pts.append(last[label])
    return lines


def _progress_figure(data: dict[str, Any], objective: dict[str, Any], out_dir: str) -> dict[str, Any]:
    if not data["rows"]:
        return _outcome("search_progress", PROGRESS_FILE, "skipped", "The search's record holds no evaluation, so the search's progress is not drawn.")
    plt, fig, (ax,) = _new_figure(1)
    try:
        for label, pts in _best_so_far(data["rows"], str(objective.get("direction") or "maximise")).items():
            ax.step([p[0] for p in pts], [p[1] for p in pts], where="post", label=label)
        unit = objective.get("unit")
        ax.set_xlabel("evaluation number")
        ax.set_ylabel("best %s so far%s" % (objective.get("quantity") or "objective", " (%s)" % unit if unit else ""))
        ax.legend()
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, PROGRESS_FILE))
        return _outcome("search_progress", PROGRESS_FILE, "ok")
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
    record = data["record"]
    objective = record.get("objective") if isinstance(record.get("objective"), dict) else {}
    baseline = (record.get("baseline") or {}).get("objective") if isinstance(record.get("baseline"), dict) else None
    best = record.get("best") if isinstance(record.get("best"), dict) else {}
    best_value = best.get("objective") if _number(best.get("objective")) and best.get("design") is not None else None
    if not _number(baseline) and best_value is None:
        return _outcome("search_best_vs_baseline", BEST_VS_BASELINE_FILE, "skipped",
                        "The search's record holds neither a baseline nor a best design, so the comparison is not drawn.")
    improvement = (record.get("improvement") or {}).get("value") if isinstance(record.get("improvement"), dict) else None
    unit = objective.get("unit") or ""
    plt, fig, (ax,) = _new_figure(1)
    try:
        ax.axis("off")
        name = "%s%s" % (objective.get("quantity") or "objective", " (%s)" % unit if unit else "")
        cells = [[name, _fmt(baseline), _fmt(best_value) if best_value is not None else "no best design",
                  _fmt(improvement) if best_value is not None else "none"]]
        table = ax.table(cellText=cells, colLabels=["", "baseline", "best design", "improvement"], loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.scale(1, 1.8)
        note = best_sentence(record)
        if note:
            ax.text(0.5, 0.12, note, ha="center", va="top", wrap=True, transform=ax.transAxes)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, BEST_VS_BASELINE_FILE))
        out = _outcome("search_best_vs_baseline", BEST_VS_BASELINE_FILE, "ok")
        out["note"] = note
        return out
    finally:
        plt.close(fig)


def _scalar_levels(design: Any) -> dict[str, float]:
    return {str(k): float(v) for k, v in (design or {}).items() if _number(v)} if isinstance(design, dict) else {}


def _scan_figure(data: dict[str, Any], out_dir: str) -> dict[str, Any]:
    """The coarse scan's map: the objective at each scanned design, over the two settings that vary most (one setting: a line)."""
    points = [(_scalar_levels(r["design"]), r["objective"]) for r in data["scan"]]
    points = [p for p in points if p[0]]
    names = sorted({k for p, _ in points for k in p}, key=lambda k: -len({p[k] for p, _ in points if k in p}))
    names = [k for k in names if len({p[k] for p, _ in points if k in p}) > 1]
    if not points or not names:
        return _outcome("search_scan_map", SCAN_FILE, "skipped", "The coarse scan has no setting that changed, so its map is not drawn.")
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
        fig.savefig(os.path.join(out_dir, SCAN_FILE))
        return _outcome("search_scan_map", SCAN_FILE, "ok")
    finally:
        plt.close(fig)


def search_figures(root: str, out_dir: str) -> list[dict[str, Any]]:
    """FI's own figures of a search, from FI's files under ``root``; each one's failure is its own."""
    data = _read_record(str(root))
    objective = data["record"].get("objective") if isinstance(data["record"].get("objective"), dict) else {}
    out = []
    for draw in (lambda: _progress_figure(data, objective, out_dir), lambda: _best_vs_baseline_figure(data, out_dir),
                 lambda: _scan_figure(data, out_dir) if data["scan"] else None):
        try:
            result = draw()
        except Exception as err:
            result = _outcome("search", "", "failed", "a figure of the search stopped with %s: %s." % (type(err).__name__, err))
        if result:
            out.append(result)
    return out


# ---- what is said and the exit code ---------------------------------------------------------------------------------------------
def said(outcomes: list[dict[str, Any]]) -> str:
    """The figure step in plain words: how many were drawn and, one sentence each, every one that was not and why."""
    drawn = [o for o in outcomes if o.get("status") == "ok"]
    lines = ["%d of %d figure(s) drawn." % (len(drawn), len(outcomes))]
    lines += [o["said"] for o in outcomes if o.get("status") != "ok" and o.get("said")]
    lines += [o["note"] for o in outcomes if o.get("note")]
    return " ".join(lines)


def exit_code(outcomes: list[dict[str, Any]]) -> int:
    return 1 if any(o.get("status") in ("failed", "refused") for o in outcomes) else 0
