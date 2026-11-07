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
import inspect
import json
import math
import numbers
import os
import re
import sys
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
    fig, axes = plt.subplots(rows, columns, figsize=(width * (1 if columns == 1 else 1.5), height * rows), squeeze=False,
                             layout="constrained")  # the layout makes room for each panel's labels, titles and the legend
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


def _last_read_error(table: Any, shows: Any = None) -> tuple[str, str]:
    """The most recent exception a read of the table raised: ``(what, which results can be read)``. An ``UnknownResult`` keeps the
    id that was read (``g_lo`` is not a result the plan declares) and, apart from it, the ids the plan declares, so a repair
    knows which name to change. A ``ValueError`` or ``KeyError`` of a read names a result in its text: only the type is kept. The head of
    an ``UnknownResult`` echoes the id the body typed, so it stays capped; FI's own text (the list of results, any other error) is
    never cut: a note shortened mid-sentence left a repair without the form it needed."""
    errors = getattr(table, "last_error", None)
    last = list(errors.values())[-1] if isinstance(errors, dict) and errors else ""
    if last.startswith("UnknownResult"):
        kind, _colon, said = last.partition(": ")
        said = said[1:-1] if len(said) > 1 and said[0] == said[-1] and said[0] in "'\"" else said  # a KeyError shows its text quoted
        head, tail = (re.split(r"; the (?:declared results|results with one) are: ", said, maxsplit=1) + [""])[:2]  # FI's own message
        if isinstance(shows, (list, tuple)):  # the model is handed the ids of the figure's own ``shows`` only
            tail = ", ".join(i.strip() for i in tail.split(",") if i.strip() in {str(r) for r in shows})
        return f"{kind}: {head.strip()[:600]}", ("The results that can be read: %s." % tail.strip()) if tail.strip() else ""
    if last.startswith(("ValueError", "KeyError")):
        return last.split(":", 1)[0], ""
    return last, ""  # FI's own message is never cut: a repair is written from it


def _read_forms(spec: dict[str, Any], table: Any) -> str:
    """How each result the figure ``shows`` is read from the table, one line each, from the table's own record of the results
    (``Table.keys`` and ``Table.per_level``, which every frame's table has); empty when the figure shows nothing or the table does
    not say. These lines are FI's own words and are never shortened: they are what a repair needs when a body read a result the
    wrong way."""
    shows = spec.get("shows")
    if not isinstance(shows, (list, tuple)) or not callable(getattr(table, "per_level", None)) or not callable(getattr(table, "keys", None)):
        return ""
    try:
        keys = list(table.keys())
    except Exception:  # noqa: BLE001 -- a table that cannot say is not a reason to lose the original error
        return ""
    lines = []
    # Probing the table for a form must not change what it records of the body's own reads.
    saved = (list(getattr(table, "served", []) or []), set(getattr(table, "gave_values", set()) or ()),
             dict(getattr(table, "last_error", {}) or {}))
    try:
        for rid in shows:
            name, levels = str(rid), []
            for key in keys[:200]:
                try:
                    levels = [str(t) for t in table.per_level(name, key)]
                except Exception:  # noqa: BLE001
                    levels = []
                if levels:
                    break
            if levels:
                lines.append("- `%s` is recorded for each value compared inside the runs (%s), not once for a setting: read it with "
                             "`table.level_rows(%r, key)` (the record key, the value and the trials' values of each) or "
                             "`table.level_means(%r, key)`; `table.values` does not read it." % (name, ", ".join(levels), name, name))
            elif _has(table, keys, "n", name):
                lines.append("- `%s` is recorded once for each setting: read it with `table.values(%r, key)`, `table.mean(%r, key)` or "
                             "`table.trials(%r, key)`, with `key` one of `table.keys()`." % (name, name, name, name))
            elif _has(table, keys, "reference", name):
                lines.append("- `%s` is a reference value FI works out from the plan, not a recorded result: read it with "
                             "`table.reference(%r, key)`, with `key` one of `table.keys()`." % (name, name))
    finally:
        if hasattr(table, "served"):
            table.served[:] = saved[0]
        if isinstance(getattr(table, "gave_values", None), set):
            table.gave_values.clear()
            table.gave_values.update(saved[1])
        if isinstance(getattr(table, "last_error", None), dict):
            table.last_error.clear()
            table.last_error.update(saved[2])
    return ("How the results this figure shows are read:\n" + "\n".join(lines)) if lines else ""


def _has(table: Any, keys: list[Any], read: str, name: str) -> bool:
    """Whether read ``read`` (``n`` or ``reference``) of ``name`` returns something for some setting."""
    fn = getattr(table, read, None)
    if not callable(fn):
        return False
    for key in keys[:200]:
        try:
            got = fn(name, key)
        except Exception:  # noqa: BLE001
            continue
        if got:
            return True
    return False


def _with_forms(why: str, spec: dict[str, Any], table: Any) -> str:
    forms = _read_forms(spec, table)
    return why + ("\n" + forms if forms and forms not in why else "")


def _drew_nothing(spec: dict[str, Any], table: Any) -> str:
    """Why a figure drew nothing from the trial record (``""`` when some read of a result gave data, or when the table does not
    record its reads). FI's table notes each read that returned data, so this reads no text of the body. When the last read
    was of an id the plan does not declare, the id and the declared ones are said, on a line of their own (a line that names a
    quantity a run left out is taken out of what a model reads, and the first line must survive that)."""
    gave = getattr(table, "gave_values", None)
    shows = spec.get("shows")
    if not isinstance(gave, (set, frozenset)) or gave or not shows or not isinstance(shows, (list, tuple)):
        return ""  # a figure that declares no result draws what it says it draws (a layout, a reference curve): not checked
    text = "`figure_%s` drew nothing from the trial record: no read of a result returned a value." % spec.get("id")
    err, readable = _last_read_error(table, shows)
    return _with_forms(text + (" The last read raised %s." % err.rstrip(".") if err else "") + ("\n" + readable if readable else ""), spec, table)


def _declared_record(spec: dict[str, Any], table: Any, out_dir: str, file: str, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the figure was declared to show next to what its body drew, from the table's own tracking of the reads that gave
    values (``Table.gave_values``): ``{"shows", "drawn", "not_drawn", "labels"}``, the labels being the plan's own for each
    declared result. It is added to the figure's record (``FI_FIGURE_RECORDS/<stem>.json``, written when the figure was saved)
    so the caption is written from it and not from the words the body typed on an axis. A figure drawn through FI's calls adds
    ``calls`` (how many series, bands, reference lines and notes were drawn) and ``raw_panels`` (the panels drawn outside the
    calls); ``summary`` is that. Empty when the figure declares no result and was not drawn through the calls, or the table does
    not track its reads."""
    shows, gave = spec.get("shows"), getattr(table, "gave_values", None)
    got: dict[str, Any] = {}
    if isinstance(shows, (list, tuple)) and shows and isinstance(gave, (set, frozenset)):
        ids = [str(r) for r in shows]
        labels = spec.get("result_labels") if isinstance(spec.get("result_labels"), dict) else {}
        got = {"shows": ids, "drawn": [r for r in ids if r in gave], "not_drawn": [r for r in ids if r not in gave],
               "labels": {r: str(labels[r]) for r in ids if labels.get(r)}}
    if summary:
        got = dict(got, calls=summary["calls"], raw_panels=summary["raw_panels"])
    if not got:
        return {}
    folder = os.environ.get("FI_FIGURE_RECORDS")
    if folder:
        path = os.path.join(folder, os.path.splitext(file)[0] + ".json")
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(dict(record, declared=got), handle)
        except (OSError, ValueError):
            pass  # a figure saved without the recorder has no record to add to; the outcome still carries it
    return got


def _draw_one(spec: dict[str, Any], body: Any, table: Any, out_dir: str, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    """Draw one declared figure. A fault of the model's body is ``failed`` / ``refused`` (its repair); a fault of FI's own lines
    (making the figure, handing it over, the folder) is ``fi_fault`` and is never sent to the model."""
    fid, file = str(spec.get("id")), str(spec.get("file") or "%s.png" % spec.get("id"))
    plt = fig = None
    try:
        _remove(out_dir, file)
        _remove_calls(file)  # the call record of an earlier run goes with its figure
        if body is None:
            return _outcome(fid, file, "not_written")
        name = getattr(body, "__name__", "figure_%s" % fid)
        plt, fig, axes = _new_figure(spec.get("panels") or 1)
        handle = _figure_handle(axes, spec, file, plan, table)
        if _has_calls(handle):
            handle.fig = fig  # the drawing calls clear a title the body put over the figure
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
            return _outcome(fid, file, "failed", _with_forms(
                "`figure_%s` stopped with %s: %s (%s)." % (fid, type(err).__name__, err, where), spec, table))
        if did:
            return _outcome(fid, file, "refused", _misuse_text(name, did[0]))
        after = _folder(out_dir)
        wrote = sorted(n for n, v in after.items() if folder.get(n) != v)
        if wrote:  # a body that reached a way of saving the guard does not know: whatever it wrote is removed
            for n in wrote:
                _remove(out_dir, n)
            return _outcome(fid, file, "refused", "`%s` wrote %s itself; FI creates the figure, saves it and closes it. Draw only on "
                            "the axes the function is given (`fig.axes`)." % (name, ", ".join(wrote)))
        calls = _has_calls(handle)
        left_out = list((handle.record().get("removed") or []) if calls else [])  # lines left out in the last run
        emptied = {r.get("panel") for r in left_out if isinstance(r, dict)}
        empty = _empty_panels(axes)
        if empty and (len(empty) == len(axes) or not set(empty) <= emptied):
            return dict(_outcome(fid, file, "failed", _with_forms(
                "`figure_%s` drew nothing on panel %s." % (fid, " and ".join(map(str, empty))), spec, table)),
                        **({"removed": left_out} if left_out else {}))
        none_drawn = _drew_nothing(spec, table)
        if none_drawn:
            return _outcome(fid, file, "failed", none_drawn)
        if calls:  # a quest on the frame with the drawing calls: what was drawn outside them is marked, FI writes the words
            fault = _audit(handle, fig, axes)
            if fault:
                return _outcome(fid, file, "fi_fault", "FI's record of the calls of figure `%s` and the calls it kept disagree: %s." % (fid, fault))
            handle.finish((plan or {}).get("labels") or {}, (plan or {}).get("result_info") or {},
                          {k: (plan or {}).get(k + "_scale") for k in ("x", "y")})
        panel_record = _set_panel_titles(handle, axes, spec.get("labels") or {})
        try:
            fig.savefig(os.path.join(out_dir, file))  # the recorder of the house style notes what the figure draws
        except Exception as err:
            return _outcome(fid, file, "failed", "`figure_%s` drew something that cannot be saved (%s: %s)." % (fid, type(err).__name__, err))
        summary: dict[str, Any] = {}
        removed: list[Any] = []
        if calls:
            _write_calls(handle, file)
            drawn = handle.record()
            summary = _calls_summary(drawn)
            removed = list(drawn.get("removed") or [])  # lines left out in the last run: FI could not say what they are
        declared = _declared_record(spec, table, out_dir, file, summary)
        done = dict(_outcome(fid, file, "ok"), panels=panel_record, **({"declared": declared} if declared else {}),
                    **({"removed": removed} if removed else {}))
        if summary:
            done["note"] = _checked_line(fid, summary)
        return done
    except Exception as err:
        return _outcome(fid, file, "fi_fault", "FI stopped while drawing the figure `%s` (%s: %s)." % (fid, type(err).__name__, err))
    finally:
        plt.close(fig)


def _figure_handle(axes: list[Any], spec: dict[str, Any], file: str, plan: dict[str, Any] | None = None, table: Any = None) -> Any:
    """What the body is given: the frame's ``Figure`` when it is carried beside this file, else a plain holder. A frame with the
    drawing calls (version 5) is also given what the calls check against, all of it from FI's own records (``plan``, see
    ``_plan_for``, and the ``table`` of the run); an older frame's ``Figure`` takes none of it and is built as before."""
    try:
        from fi_frame import Figure  # the frame's runtime, written beside this file in the quest's code folder
    except ImportError:
        class Figure:  # type: ignore[no-redef]
            def __init__(self, axes, x=None, series=None, shows=None, file=None):
                self.axes, self.x, self.series, self.shows, self.file = list(axes), x, series, shows, file
    given: dict[str, Any] = {}
    try:
        takes = set(inspect.signature(Figure).parameters)
    except (TypeError, ValueError):
        takes = set()
    if "series_ids" in takes:
        plan = plan or {}
        given = {"table": table, "settings": plan.get("settings"), "series_ids": plan.get("series_ids"),
                 "references": plan.get("references"), "labels": plan.get("labels"), "fallback": plan.get("fallback", False),
                 "derived_ids": plan.get("derived_ids")}
    handle = Figure(axes, x=spec.get("x"), series=spec.get("series"), shows=spec.get("shows"), file=file, **given)
    if given:
        handle.derived_values = (plan or {}).get("derived_values") or {}  # set beside the frame's own arguments, so an older frame still builds
    handle.axes = _Panels(handle.axes, str(spec.get("id")))
    handle.declared = {}
    handle.panel = _panel_declarer(handle, spec.get("settings") or {})
    return handle


class _Panels(list):
    """``fig.axes``: the axes FI made, one for each panel of the declared figure. A place past the last panel raises an
    ``IndexError`` that says how many panels FI made, so the body is repaired from the count and not from a bare index."""

    def __init__(self, axes: Any, figure: str):
        super().__init__(axes)
        self.figure = figure

    def __getitem__(self, index: Any) -> Any:
        try:
            return super().__getitem__(index)
        except IndexError:
            if not isinstance(index, int):
                raise
            advice = (", and put several settings, series or results on one axes as lines" if len(self) == 1 else "")
            raise IndexError("`fig.axes[%d]`: FI made %d panel(s) for `figure_%s` (the plan's count), so `fig.axes` holds the axes "
                             "0 to %d; draw on those%s" % (index, len(self), self.figure, len(self) - 1, advice)) from None


class UnknownSetting(KeyError):
    """A panel declared for a setting the study does not vary, or a value it does not take."""


def _panel_declarer(handle: Any, allowed: dict[str, Any]):
    """``fig.panel(which, **settings)``: say which setting a panel shows (``which`` is the panel's place in ``fig.axes``, or the
    axes). FI then writes the panel's title. A setting the plan does not vary, or a value outside its declared values, raises
    ``UnknownSetting`` naming what may be used (``allowed`` is empty for a plan that did not record its settings: not checked)."""
    def panel(which: Any, **settings: Any) -> None:
        index = which if isinstance(which, int) else handle.axes.index(which)
        if not 0 <= index < len(handle.axes):
            raise IndexError("panel %r: this figure has %d panel(s)" % (which, len(handle.axes)))
        calls = _has_calls(handle)  # on the drawing calls' frame the check is closed: the plan's settings and values only
        if not calls:  # an older frame: as it always was
            for name, value in settings.items():
                if allowed and (name not in allowed or value not in allowed[name]):
                    raise UnknownSetting(_not_a_setting(index, name, value, allowed))
            handle.declared[index] = dict(settings)
            return
        held: dict[str, Any] = {}
        for name, value in settings.items():
            found = _in_grid(value, allowed.get(name)) if name in allowed else _NOT_IN
            if found is _NOT_IN:
                raise UnknownSetting(_not_a_setting(index, name, value, allowed))
            held[name] = found  # the plan's own value, not the body's object
        record = handle._panel(index)[1]  # the call record keeps which setting the panel holds; `finish` writes the title from it
        if record.get("hold") not in (None, held):  # also a bare fig.panel(i) after a hold
            raise ValueError("panel %d was already declared to show %s; one panel shows one setting, so declare it once" %
                             (index, ", ".join("%s = %s" % kv for kv in record["hold"].items())))
        if held:
            record["hold"] = held
        handle.declared[index] = dict(held)
    return panel


def _not_a_setting(index: int, name: str, value: Any, allowed: dict[str, Any]) -> str:
    return ("panel %d: `%s = %s` is not a setting this study varies; the settings and values a panel may use are: %s"
            % (index, name, value, "; ".join("%s = %s" % (k, ", ".join(map(str, v))) for k, v in allowed.items()) or "none"))


_NOT_IN = object()


def _in_grid(value: Any, grid: Any) -> Any:
    """The plan's own value in ``grid`` that ``value`` equals, else ``_NOT_IN``. Numbers match by value (``1`` and ``1.0``
    match), a bool matches only a bool, and a string only a string."""
    value = _plain(value)

    def kind(v: Any) -> str:
        return "b" if isinstance(v, bool) else "n" if isinstance(v, numbers.Real) else "s" if isinstance(v, str) else "?"

    for item in grid or []:
        if kind(item) == kind(value) != "?" and item == value:
            return item
    return _NOT_IN


def _plain(value: Any) -> Any:
    """A setting's value as a plain number or text (a numpy scalar is unwrapped), so the call record can be written."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


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
        rec = handle._panel(index)[1] if _has_calls(handle) else None
        if rec is not None and not rec["raw"]:
            record.append({"panel": index + 1, "settings": settings, "title": axes[index].get_title()})  # `finish` wrote it
            continue
        title = handle._hold_title(rec) if rec is not None and rec.get("hold") else _panel_title(settings, labels)
        for loc in ("left", "center", "right"):
            axes[index].set_title("", loc=loc)
        axes[index].set_title(title)
        record.append({"panel": index + 1, "settings": settings, "title": title})
    return record


# ---- the drawing calls: the plan they check against, the custom-drawing mark, the call record ---------------------------------
def _number_text(number: Any) -> str:
    """A plan number as a person writes it: 200, 0.05, 6.283."""
    value = float(number)
    return str(int(value)) if value.is_integer() and abs(value) < 1e15 else "%.4g" % value


def _from(namespace: Any, name: str, default: Any) -> Any:
    got = namespace.get(name) if isinstance(namespace, dict) else getattr(namespace, name, None)
    return default if got is None else got


def _plan_for(spec: dict[str, Any], namespace: Any, table: Any) -> dict[str, Any]:
    """What the drawing calls of one figure check against, read from FI's own records and never from the body: the study's
    settings (the names its trial cells carry, what each run compares, what a sub-study holds), the ids a series may name (the
    figure's own series and the values the study compares), the plan's references that hold one number at every setting, the
    plan's words for ids, whether FI drew the figure itself, and the quantities the study derives. Whatever a record does not
    hold is left out: a call then checks against less and says what is missing."""
    compared = _from(namespace, "COMPARED", {})
    compared = compared if isinstance(compared, dict) else {}
    parts = _from(namespace, "PARTS", {})
    settings: list[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name and name not in settings:
            settings.append(name)

    keys: list[Any] = []
    try:
        keys = list(table.keys()) if table is not None else []
    except Exception:  # noqa: BLE001 -- a table that cannot list its settings has none to offer
        pass
    for key in keys:
        try:
            for name in (table.setting(key) or {}):
                add(name)
        except Exception:  # noqa: BLE001
            continue
    held_by_parts = [n for hold in (parts.values() if isinstance(parts, dict) else []) if isinstance(hold, dict) for n in hold]
    for name in list(compared) + held_by_parts + list(spec.get("settings") or {}):
        add(name)
    series_ids: list[str] = []
    own = spec.get("series")
    for item in ([own] if isinstance(own, str) else list(own) if isinstance(own, (list, tuple)) else []):
        if isinstance(item, str) and item not in series_ids:
            series_ids.append(item)
    for values in compared.values():
        for value in (values if isinstance(values, (list, tuple)) else []):
            if str(value) not in series_ids:
                series_ids.append(str(value))
    results = spec.get("result_labels") if isinstance(spec.get("result_labels"), dict) else {}
    info = spec.get("result_info") if isinstance(spec.get("result_info"), dict) else {}
    named = {str(k): v["label"] for k, v in info.items() if isinstance(v, dict) and v.get("label")}
    labels: dict[str, Any] = {str(k): str(v) for k, v in {**named, **(spec.get("labels") or {}), **results}.items() if v}
    references: dict[str, Any] = {}
    held = getattr(table, "_references", None)  # read only: Table.reference() would also count the result as read
    for ident, per_setting in (held.items() if isinstance(held, dict) else []):
        numbers = {float(v) for v in (per_setting.values() if isinstance(per_setting, dict) else []) if _number(v)}
        if len(numbers) == 1:  # a reference that differs by setting is not one line
            references[str(ident)] = {"value": next(iter(numbers)), "label": labels.get(str(ident), str(ident))}
    limits = getattr(table, "_thresholds", None)  # the run's record of the plan's thresholds (nominal levels, pass/fail limits)
    for ident, number in (limits.items() if isinstance(limits, dict) else []):
        # a threshold of the plan (a significance level, a coverage target) is a line a figure may draw, named by the plan's key
        if str(ident) not in references and _number(number) and not isinstance(number, bool):
            references[str(ident)] = {"value": float(number), "label": "%s = %s" % (ident, _number_text(number)), "fixed": True}
    nominal = spec.get("nominal") if isinstance(spec.get("nominal"), dict) else {}
    held_numbers = [float(r["value"]) for r in references.values() if _number(r.get("value"))]
    for ident, level in nominal.items():
        # the level the plan declares for a result's interval (``known_value.level``) is a line a figure may draw too, one for each result
        # the figure draws and named by it; it is offered only while no reference or threshold of the plan already holds that number
        if not _number(level) or isinstance(level, bool) or any(abs(v - float(level)) <= 1e-12 for v in held_numbers):
            continue
        references["%s_nominal" % ident] = {"value": float(level), "fixed": True,
                                            "label": "Nominal level of %s = %s" % (labels.get(str(ident), str(ident)), _number_text(level))}
    derived = _from(namespace, "DERIVED", {})
    return {"settings": settings, "series_ids": series_ids, "references": references, "labels": labels,
            "result_info": info,
            **{k: spec[k] for k in ("x_scale", "y_scale") if spec.get(k) in ("linear", "log")},
            "fallback": bool(spec.get("fallback")), "derived_ids": [str(k) for k in derived] if isinstance(derived, dict) else []}


def _has_calls(handle: Any) -> bool:
    """True for the frame's ``Figure`` with the drawing calls (version 5). An older frame's, and the plain holder, have none, and
    a quest on them gets no call record (the legacy path)."""
    try:
        return all(callable(getattr(handle, name)) for name in ("finish", "record", "write", "_panel"))
    except Exception:  # noqa: BLE001 -- the frame says a missing file by raising on any call name
        return False


def _records_folder() -> str:
    return os.environ.get("FI_FIGURE_RECORDS") or ""


def _remove_calls(file: str) -> None:
    folder = _records_folder()
    if folder:
        try:
            os.remove(os.path.join(folder, os.path.splitext(file)[0] + ".calls.json"))
        except OSError:
            pass


#: The tick-label formatters matplotlib picks itself (and the one a category axis gets); a tick label under any other formatter
#: (a fixed list of words, a function) is text the body typed.
_AUTO_FORMATTERS = frozenset({"ScalarFormatter", "LogFormatter", "LogFormatterExponent", "LogFormatterMathtext",
                              "LogFormatterSciNotation", "LogitFormatter", "NullFormatter", "StrCategoryFormatter", "FiSlotFormatter"})


def _all_axes(fig: Any) -> tuple[list[Any], dict[int, Any]]:
    """Every axes of the figure, insets included, and the parent of each inset."""
    out: list[Any] = []
    parent: dict[int, Any] = {}
    stack = list(fig.get_axes())
    while stack:
        ax = stack.pop(0)
        if any(ax is seen for seen in out):
            continue
        out.append(ax)
        for child in getattr(ax, "child_axes", None) or []:
            parent[id(child)] = ax
            stack.append(child)
    return out, parent


def _panel_of(ax: Any, axes: list[Any], parent: dict[int, Any]) -> int | None:
    """The panel an axes belongs to: itself, the panel it is an inset of, or the panel it shares an axis with (a twin)."""
    while ax is not None:
        for i, panel in enumerate(axes):
            if ax is panel:
                return i
        for i, panel in enumerate(axes):
            if any(ax is other for group in (panel.get_shared_x_axes(), panel.get_shared_y_axes())
                                  for other in group.get_siblings(panel)):
                return i
        ax = parent.get(id(ax))
    return None


def _scaffolding(fig: Any, all_axes: list[Any], axes: list[Any], owner: dict[int, Any]) -> set[int]:
    """The ids of what matplotlib draws by itself and FI rewrites: the figure and axes backgrounds, spines, ticks and the tick
    labels of a default formatter, and, on a panel, the title, axis labels and legend (``Figure.finish`` writes those)."""
    ok = {id(fig), id(fig.patch)}
    for ax in all_axes:
        ok.update((id(ax), id(ax.patch)))
        ok.update(id(sp) for sp in ax.spines.values())
        is_panel = any(ax is p for p in axes)
        for axis in (ax.xaxis, ax.yaxis):
            ok.update((id(axis), id(axis.offsetText)))
            if is_panel:
                ok.add(id(axis.label))
            manual = {type(axis.get_major_formatter()).__name__, type(axis.get_minor_formatter()).__name__} - _AUTO_FORMATTERS
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                ok.update((id(tick), id(tick.tick1line), id(tick.tick2line), id(tick.gridline)))
                owner.update({id(tick.label1): ax, id(tick.label2): ax})  # a tick label does not know its axes
                if not manual:
                    ok.update((id(tick.label1), id(tick.label2)))
        if is_panel:
            ok.update(id(t) for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)) if t is not None)
            legend = ax.get_legend()
            if legend is not None:
                ok.update(id(o) for o in [legend] + list(legend.findobj()))
    return ok


def _gap(v: Any) -> bool:
    return v is None or (isinstance(v, numbers.Real) and math.isnan(v))


def _same_numbers(a: Any, b: Any) -> bool:
    """Two sequences as stored: numbers equal, a gap equal to a gap, anything else (a category, a numeric string) equal as text."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if _gap(u) or _gap(v):
            if not (_gap(u) and _gap(v)):
                return False
        elif isinstance(u, numbers.Real) and isinstance(v, numbers.Real) and not isinstance(u, bool) and not isinstance(v, bool):
            if not math.isclose(float(u), float(v), rel_tol=1e-12, abs_tol=1e-300):
                return False
        elif str(u) != str(v):
            return False
    return True


def _drawn_differently(art: Any, entry: dict[str, Any], owned: dict[int, Any], panel_ax: Any) -> bool:
    """True when the artist a call drew no longer holds what the call recorded (the body changed it, or took it out)."""
    try:
        if entry["kind"] == "image":
            return False  # compared by the caller, which knows which image belongs to which entry
        if art is None or getattr(art, "axes", None) is None:
            return True
        if entry["kind"] == "bars":
            group = [p for p in owned[id(art)][1] if hasattr(p, "get_height")]
            return not _same_numbers([p.get_height() for p in group], [float("nan") if y is None else y for y in entry["y"]])
        return not (_same_numbers(art.get_xdata(orig=True), entry["x"])
                    and _same_numbers(art.get_ydata(orig=True), [float("nan") if y is None else y for y in entry["y"]]))
    except Exception:  # noqa: BLE001 -- an artist this check cannot read is not held to the record
        return False


def _audit(handle: Any, fig: Any, axes: list[Any]) -> str:
    """After the body returned, before FI writes labels, legend and limits. Every visible artist of the figure that the drawing
    calls did not create (FI keeps their identities; a tag the body could set itself proves nothing) and that is not
    matplotlib's own scaffolding is the body's: the panel it sits on is a custom drawing, and one that sits on no panel (a
    ``fig.text``, ``fig.legend``, a suptitle, a colorbar) makes every panel one. A drawing call's own artist whose data the body
    changed makes its panel one too. Returns a text only for FI's own inconsistency (the record and the calls it keeps disagree)."""
    from matplotlib.image import AxesImage

    owned = getattr(handle, "_fc_owned", None) or {}
    all_axes, parent = _all_axes(fig)
    where_is: dict[int, Any] = {}
    for ax in all_axes:  # a part of a table, a legend or a tick label may not know its axes: the axes it hangs from do
        for child in ax.get_children():
            for obj in child.findobj():
                where_is.setdefault(id(obj), ax)
    ok = _scaffolding(fig, all_axes, axes, where_is) | set(owned)
    seen: dict[int, Any] = {}
    for root in [fig] + all_axes:
        for obj in root.findobj():
            seen[id(obj)] = obj
    why: dict[Any, str] = {}
    for obj in seen.values():
        if id(obj) in ok or not obj.get_visible():
            continue
        text = hasattr(obj, "get_text")
        if text and not str(obj.get_text()).strip():
            continue
        home = where_is.get(id(obj)) or getattr(obj, "axes", None)
        where = _panel_of(home, axes, parent) if home is not None else None
        why.setdefault(where, "text placed outside FI's calls" if text else "drawn outside FI's calls")
    for i, ax in enumerate(axes):
        _ax, rec = handle._panel(i)
        extra = getattr(handle, "_fc_extra", {}).get(i) or {"series": [], "refs": []}
        if len(extra["series"]) != len(rec["series"]) or len(extra["refs"]) != len(rec["refs"]):
            return "panel %d: FI's record holds %d series and %d reference lines but the calls kept %d and %d" % (
                i + 1, len(rec["series"]), len(rec["refs"]), len(extra["series"]), len(extra["refs"]))
        images = [a for a, _grp in owned.values() if isinstance(a, AxesImage) and getattr(a, "axes", None) is ax]
        for (art, _what), entry in zip(extra["series"], rec["series"]):
            if entry["kind"] == "image":
                shown = images.pop(0) if images else None
                if shown is None or not _same_numbers([v for row in shown.get_array().astype(float).filled(float("nan")).tolist() for v in row],
                                                      [float("nan") if v is None else v for row in entry["z"] for v in row]):
                    why.setdefault(i, "drawing changed after FI's call")
            elif _drawn_differently(art, entry, owned, ax):
                why.setdefault(i, "drawing changed after FI's call")
        for line, ref in zip(extra["refs"], rec["refs"]):
            data = line.get_ydata() if ref["axis"] == "y" else line.get_xdata()
            if getattr(line, "axes", None) is None or not len(data) or not _same_numbers([data[0]], [ref["value"]]):
                why.setdefault(i, "drawing changed after FI's call")
    for i in range(len(axes)):
        _ax, rec = handle._panel(i)
        reason = why.get(i) or why.get(None)
        if reason and not rec["raw"]:
            rec["raw"], rec["raw_reason"] = True, reason
    return ""


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _digit_texts(index: int, ax: Any, written: Any) -> list[str]:
    """The words a custom drawing prints that hold a digit, from the figure recorder (its title, labels, texts and legend
    entries), else read from the axes."""
    panel = written[index] if isinstance(written, list) and index < len(written) and isinstance(written[index], dict) else None
    if panel is not None:
        found = [panel.get("title"), panel.get("xlabel"), panel.get("ylabel")] + list(panel.get("texts") or [])
    else:
        found = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()] + [t.get_text() for t in ax.texts]
    out: list[str] = []
    for text in found:
        text = " ".join(str(text or "").split())
        if text and any(ch.isdigit() for ch in text) and text not in out:
            out.append(text)
    return out


def _write_calls(handle: Any, file: str) -> None:
    """Write ``<FI_FIGURE_RECORDS>/<stem>.calls.json`` after the figure was saved: what the calls drew, and for a custom drawing
    the digit-bearing words the recorder saw. Without a records folder there is nowhere to write."""
    folder = _records_folder()
    if not folder:
        return
    written = (_read_json(os.path.join(folder, os.path.splitext(file)[0] + ".json")) or {}).get("axes")
    for i, ax in enumerate(handle.axes):
        _ax, rec = handle._panel(i)
        if rec["raw"]:
            rec["texts_raw"] = _digit_texts(i, ax, written)
    handle.write(os.path.join(folder, file))


def _calls_summary(record: dict[str, Any]) -> dict[str, Any]:
    panels = record.get("panels") or []
    return {"calls": {"series": sum(len(p["series"]) for p in panels), "bands": sum(len(p["bands"]) for p in panels),
                      "refs": sum(len(p["refs"]) for p in panels), "notes": sum(len(p["notes"]) for p in panels)},
            "raw_panels": [p["panel"] for p in panels if p["raw"]], "panels": len(panels)}


def _checked_line(fid: str, summary: dict[str, Any]) -> str:
    """The line of the run log for one figure: how many panels FI checked and which are a custom drawing."""
    total, raw = summary["panels"], summary["raw_panels"]
    line = "figure %s: %d of %d panels checked by FI" % (fid, total - len(raw), total)
    if raw:
        line += ", panel%s %s %s a custom drawing" % ("s" if len(raw) > 1 else "", " and ".join(map(str, raw)), "are" if len(raw) > 1 else "is")
    return line


def _derived_values(derived: Any) -> dict[str, dict[str, float]]:
    """``{id: {setting key: number}}`` of FI's record of the derived values (``{id: record}``), for the ones that hold a number for
    each setting (``by_setting``) and were worked out; whatever else a record holds is left out, so a call checks against less."""
    out: dict[str, dict[str, float]] = {}
    for ident, record in (derived.items() if isinstance(derived, dict) else ()):
        if isinstance(record, dict) and record.get("status") == "ok" and isinstance(record.get("by_setting"), dict):
            got = {str(k): float(v) for k, v in record["by_setting"].items() if _plain_number(v)}
            if got:
                out[str(ident)] = got
    return out


def run_figures(figures: list[dict[str, Any]], namespace: Any, table: Any, out_dir: str, derived: Any = None) -> list[dict[str, Any]]:
    """Draw every declared figure from the body ``figure_<id>`` of ``namespace`` (a module, or a dict); a figure's failure is its
    own and the others are still drawn. ``derived``: FI's record of the derived values, which a call that draws a derived quantity
    is checked against (as a result of the table is checked against the table)."""
    out = []
    held = _derived_values(derived)
    for spec in figures or []:
        name = "figure_%s" % spec.get("id")
        body = namespace.get(name) if isinstance(namespace, dict) else getattr(namespace, name, None)
        plan = _plan_for(spec, namespace, table)
        plan["derived_values"] = held
        out.append(_draw_one(spec, body, table, out_dir, plan))
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


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _derived_points(record: Any, key: str) -> dict[str, float]:
    """The number FI holds for a derived quantity at setting ``key``: ``{"": number}`` from a record of a value for each setting
    (``by_setting``), ``{}`` for anything else (one number for the whole study, a value for each compared value, a quantity that
    was not worked out, a setting it gives none for): no line is made of a number that is not along the setting."""
    if not isinstance(record, dict) or record.get("status") != "ok" or not isinstance(record.get("by_setting"), dict):
        return {}
    got = record["by_setting"].get(key)
    return {"": float(got)} if _plain_number(got) else {}


def _default_points(table: Any, rid: str, key: str, derived: Any = None) -> dict[str, float]:
    """The numbers the table holds for result ``rid`` at setting ``key``: ``{"": mean}`` for a result recorded once for the setting,
    ``{level: mean}`` for one recorded for each value compared inside the runs (``level_means``); ``{}`` when it holds none. A
    derived quantity (``derived``: FI's record of the derived values, ``{id: record}``) is read from that record."""
    if isinstance(derived, dict) and rid in derived:
        return _derived_points(derived[rid], key)
    try:
        got = table.mean(rid, key)
    except Exception:  # noqa: BLE001 -- a result recorded per compared value is read by level below
        got = None
        try:
            if table.per_level(rid, key):
                return {str(t): float(m) for t, m in table.level_means(rid, key).items() if _plain_number(m)}
        except Exception:  # noqa: BLE001
            pass
    return {"": float(got)} if _plain_number(got) else {}


def _default_lines(spec: dict[str, Any], table: Any, fixed: dict[str, Any], derived: Any = None
                   ) -> list[tuple[str, list[Any], list[float], str, str, Any, str]]:
    """The lines of the default figure with the settings in ``fixed`` held: ``(label, x values, y values, result id, series id, value of
    the series setting or None, level compared inside the runs or "")``,
    one per result in ``shows``, per value of the ``series`` setting and, for a result recorded per compared value, per value
    compared. No line holds points of two settings. The series id is made by FI from the plan's ids and the recorded values
    (``gain, rate = 0.1``); the label is the plan's words and is apart from it."""
    x_name, s_name = str(spec.get("x")), spec.get("series")
    names = spec.get("result_labels") if isinstance(spec.get("result_labels"), dict) else {}
    cells = {k: dict(table.setting(k)) for k in table.keys()}
    values = list(dict.fromkeys(c[str(s_name)] for c in cells.values() if str(s_name) in c)) if s_name else [None]
    by_compared = False  # the series is a factor compared inside the runs: only a result recorded for each of its values is drawn
    if s_name and not values:  # no setting of that name: a factor FI recorded as compared inside the runs is drawn as one line per level
        compared = getattr(table, "compared", None)  # an older table has none: the series stays unknown and the figure is dropped
        if callable(compared) and s_name in compared():
            s_name, values, by_compared = None, [None], True
    lines = []
    for rid in (str(r) for r in spec["shows"]):
        for value in values:
            by_level: dict[str, list[tuple[Any, float]]] = {}
            for k, cell in cells.items():
                if x_name not in cell or any(cell.get(a) != v for a, v in fixed.items()) or (s_name and cell.get(str(s_name)) != value):
                    continue
                for level, mean in _default_points(table, rid, k, derived).items():
                    by_level.setdefault(level, []).append((cell[x_name], mean))
            for level, points in by_level.items():
                if by_compared and not level:  # recorded once for the setting: a line of it would not be split by the series
                    continue
                if all(_plain_number(p[0]) for p in points):
                    points.sort(key=lambda p: p[0])
                tail = (["%s = %s" % (s_name, value)] if s_name else []) + ([level] if level else [])
                lines.append((", ".join([str(names.get(rid) or rid)] + tail), [p[0] for p in points], [p[1] for p in points], rid,
                              ", ".join([rid] + tail), value if s_name else None, level))
    return lines


def _grid_order(name: str, values: list[Any], spec: dict[str, Any], compared: Any) -> list[Any]:
    """The values of setting ``name`` in the plan's order: as the plan lists them for what a run compares or for the figure's own
    settings; any other (the plan's grid is not carried) ascending, numbers first. Never in the order of the table's rows."""
    plan: list[Any] = []
    for source in (compared.get(name) if isinstance(compared, dict) else None,
                   spec["settings"].get(name) if isinstance(spec.get("settings"), dict) else None):
        if isinstance(source, (list, tuple)):
            plan = list(source)
            break

    def rank(v: Any) -> tuple[Any, ...]:
        if v in plan:
            return (0, plan.index(v), 0, "")
        return (1, 0, v, "") if _plain_number(v) else (2, 0, 0, str(v))

    return sorted(values, key=rank)


def _default_hold(spec: dict[str, Any], table: Any, held: list[dict[str, Any]], skip: set[str], namespace: Any,
                  derived: Any = None) -> dict[str, Any]:
    """Where the default figure holds every other setting the study varies (not ``x``, the series or the panel setting). Not at the
    first value the table records (that can be a combination nothing was run at: a blank panel). Each setting is held at the value
    that leaves the most points with a value to draw (a line's points, over the panels); of two equal ones, the first in the plan's
    order (``_grid_order``). The settings are settled in the figure's own order of them, else by name, never in the order the
    table names them. No sub-study's hold is used: the plan does not say which sub-study a figure belongs to."""
    cells = [dict(table.setting(k)) for k in table.keys()]
    others: dict[str, list[Any]] = {}
    for cell in cells:
        for a, v in cell.items():
            if a not in skip and v not in others.setdefault(a, []):
                others[a].append(v)
    varied = {a: vs for a, vs in others.items() if len(vs) > 1}
    if not varied:
        return {}
    compared = _from(namespace, "COMPARED", {})

    def points(hold: dict[str, Any]) -> int:
        return sum(len(line[1]) for one in held for line in _default_lines(spec, table, {**one, **hold}, derived))

    chosen: dict[str, Any] = {}
    own = [str(k) for k in (spec.get("settings") if isinstance(spec.get("settings"), dict) else {})]
    for a in sorted(varied, key=lambda n: (own.index(n) if n in own else len(own), n)):
        vs = varied[a]
        top, pick = -1, vs[0]
        for v in _grid_order(a, vs, spec, compared):
            got = points({**chosen, a: v})
            if got > top:
                top, pick = got, v
        chosen[a] = pick
    return chosen


def _frame_has_calls() -> bool:
    """True when the frame carried beside this file has the drawing calls (version 5)."""
    try:
        from fi_frame import Figure
        return all(callable(getattr(Figure, name, None)) for name in ("line", "finish", "record", "write"))
    except Exception:  # noqa: BLE001 -- no frame, or an older one: the default is drawn without the calls
        return False


def default_figure(spec: dict[str, Any], table: Any, out_dir: str, namespace: Any = None, derived: Any = None) -> dict[str, Any]:
    """The plan's own figure, drawn by FI from the table when a figure body cannot be repaired: one line for each result the figure
    ``shows`` over its ``x`` setting, one line for each value of its ``series`` setting (and of a factor a result is recorded per) and
    one panel for each of its ``panels`` (the panel setting is the first setting the plan lets a panel show that is neither ``x`` nor
    ``series``; the panels hold its values in turn; without one, or with fewer values than panels, there are fewer panels and the
    outcome says so). Any other setting the study varies is held (``_default_hold``: at the sub-study's hold, else where most points
    are), and the title says which. It is drawn with the calls a figure body uses (``fig.line``) when the frame has them, so its
    call record is marked ``fallback`` and holds no custom drawing; ``namespace`` is the carried plan (``COMPARED``, ``PARTS``,
    ``DERIVED``). A result the plan derives is drawn from ``derived``, FI's record of the derived values (this run's, or in the
    run that only draws the figures that failed, the earlier run's); a line FI holds no value for is left out and said on stderr,
    never made up. The drawing is lines only: no bars and no error bars. A plan that declares no ``shows`` or no ``x``, or a table that holds no value to draw, gives no default: the
    outcome is ``dropped`` and says why, and no file is kept."""
    fid, file = str(spec.get("id")), str(spec.get("file") or "%s.png" % spec.get("id"))
    _remove(out_dir, file)
    _remove_calls(file)
    shows = spec.get("shows")
    if not isinstance(shows, (list, tuple)) or not shows or not spec.get("x"):
        return dict(_outcome(fid, file, "dropped", "Figure `%s` is not drawn: the plan does not say which results it shows over which "
                             "setting, so there is nothing to draw in its place." % fid), fallback=True)
    plt = fig = None
    try:
        x_name, s_name = str(spec["x"]), (str(spec["series"]) if spec.get("series") else None)
        settings = spec.get("settings") if isinstance(spec.get("settings"), dict) else {}
        panel_by = next((str(k) for k in settings if str(k) not in (x_name, s_name)), None)
        wanted = int(spec.get("panels") or 1)
        held, note = [{}], ""
        if wanted > 1 and panel_by:
            held = [{panel_by: v} for v in list(settings[panel_by])[:wanted]]
        by_panel = panel_by if len(held) > 1 else None
        hold = _default_hold(spec, table, held, {a for a in (x_name, s_name, by_panel) if a}, namespace, derived)
        labels = spec.get("labels") if isinstance(spec.get("labels"), dict) else {}
        reset = getattr(table, "reset_reads", None)
        if callable(reset):
            reset()
        drawn = [_default_lines(spec, table, {**one, **hold}, derived) for one in held]
        for rid in (str(r) for r in shows):
            if isinstance(derived, dict) and rid in derived and not any(line[3] == rid for lines in drawn for line in lines):
                sys.stderr.write("fi: the default figure `%s` leaves out `%s`: FI holds no value of it for each setting along `%s`\n" % (
                    fid, rid, x_name))
        if not any(drawn):
            return dict(_outcome(fid, file, "dropped", "Figure `%s` is not drawn: the table holds no value of %s along `%s`." % (
                fid, ", ".join("`%s`" % r for r in shows), spec["x"])), fallback=True)
        if len(held) > 1:  # a panel the table holds nothing for is left out, and said
            held, drawn = [h for h, d in zip(held, drawn) if d], [d for d in drawn if d]
        if len(held) < wanted:
            note = "Figure `%s` has %d panel(s) instead of the %d the plan asks for: the table gives no more settings to show one on each." % (
                fid, len(held), wanted)
        if _frame_has_calls():
            return _default_with_calls(dict(spec, panels=len(held)), table, out_dir, namespace, held, hold, drawn, note, derived)
        plt, fig, axes = _new_figure(len(held))
        record = []
        for index, (one, lines) in enumerate(zip(held, drawn)):
            ax = axes[index]
            cats = list(dict.fromkeys(v for _l, xs, _y, _r, _i, _v, _c in lines for v in xs if not _plain_number(v)))
            for label, xs, ys, _rid, _sid, _value, _lvl in lines:
                ax.plot([cats.index(v) if v in cats else v for v in xs], ys, marker="o", label=label)
            if cats:
                ax.set_xticks(range(len(cats)), [str(v) for v in cats])
            ax.set_xlabel(str(labels.get(x_name) or x_name))
            if len(shows) == 1:
                ax.set_ylabel(str((spec.get("result_labels") or {}).get(str(shows[0])) or shows[0]))
            if lines:
                ax.legend()
            title = ", ".join(t for t in (_panel_title(one, labels) if one else "", _panel_title(hold, labels) if hold else "") if t)
            if title:
                ax.set_title(title)
            if one:
                record.append({"panel": index + 1, "settings": one, "title": title})
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, file))  # the recorder of the house style notes what the figure draws
        declared = _declared_record(spec, table, out_dir, file)
        out = dict(_outcome(fid, file, "ok", note), fallback=True, **({"panels": record} if record else {}),
                   **({"declared": declared} if declared else {}), **({"held": hold} if hold else {}))
        if note:
            out["note"] = note
        return out
    except Exception as err:
        _remove(out_dir, file)
        _remove_calls(file)
        return dict(_outcome(fid, file, "fi_fault", "FI's drawing of the default figure `%s` stopped with %s: %s." % (fid, type(err).__name__, err)),
                    fallback=True)
    finally:
        if isinstance(getattr(table, "last_error", None), dict):
            table.last_error.clear()  # the reads FI made here are not the body's: a later repair note must not quote them
        if plt is not None and fig is not None:
            plt.close(fig)


def _default_with_calls(spec: dict[str, Any], table: Any, out_dir: str, namespace: Any, held: list[dict[str, Any]],
                        hold: dict[str, Any], drawn: list[list[tuple[str, list[Any], list[float], str, str, Any, str]]], note: str,
                        derived: Any = None) -> dict[str, Any]:
    """Draw the default figure through ``fig.line`` as a figure body would, with the plan marked ``fallback`` so the call record is.
    A figure whose plan draws one curve per value of a setting passes that value as each line's series, exactly as a body does, so
    every curve is a level of the setting: the legend condenses and the colours run along the setting's values like a body's. A line
    of a result recorded per value compared inside the runs is named by an id FI makes of the plan's ids and the recorded values
    (``gain, rate = 0.1``; added to the ids the calls accept, with the plan's words as its label), and so is every line of a figure
    that has both (a setting's value and a compared value name no single curve). The held settings of a panel are declared for FI's
    title. The outcome is the one of ``_draw_one``, marked ``fallback``; a fault of these calls is FI's (``fi_fault``), never the model's."""
    fid, file = str(spec.get("id")), str(spec.get("file") or "%s.png" % spec.get("id"))
    plan = _plan_for(dict(spec, fallback=True), namespace, table)
    all_lines = [line for lines in drawn for line in lines]
    by_setting = isinstance(spec.get("series"), str) and spec["series"] in plan["settings"]
    as_levels = by_setting and all(line[5] is not None and not line[6] for line in all_lines)
    mixed = by_setting and not as_levels  # a setting and a compared value together: each curve is one value of each, named by FI's id
    for lines in drawn:
        for label, _x, _y, rid, sid, _value, level in lines:
            if not as_levels and (level or mixed) and sid not in plan["series_ids"]:
                plan["series_ids"].append(sid)
                plan["labels"][sid] = label

    def body(fig: Any, tbl: Any) -> None:
        for index, one in enumerate(held):
            lines = _default_lines(spec, tbl, {**one, **hold}, derived)  # read again: the reads are recorded from the start of the body
            for _label, xs, ys, rid, sid, value, level in lines:
                name = value if as_levels else sid if (level or mixed) else rid
                if isinstance(derived, dict) and rid in derived and isinstance(getattr(tbl, "gave_values", None), set):
                    tbl.gave_values.add(rid)  # FI read this value from its derived record, not from the table
                for art in fig.line(index, name, xs, ys, rid):
                    art.set_marker("o")
            if {**one, **hold}:
                fig._panel(index)[1]["hold"] = {**one, **hold}  # FI's values from its table: `finish` titles the panel from them
                fig.declared[index] = {**one, **hold}

    out = _draw_one(dict(spec, series=None) if mixed else spec, body, table, out_dir, plan)  # the lines above are read with the plan's own series
    if out.get("status") != "ok":
        _remove(out_dir, file)
        _remove_calls(file)
        return dict(_outcome(fid, file, "fi_fault", "FI's drawing of the default figure `%s` stopped: %s" % (
            fid, out.get("why") or out.get("said"))), fallback=True)
    out = dict(out, fallback=True, why=note, **({"held": hold} if hold else {}))
    if note:
        out["note"] = note
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
    """What FI's own figures could not do because of FI's own record or code: said plainly, never sent to the model . The default figure that
    failed in place of a body that failed (``fallback_fault`` on the outcome) is not one: a figure never stops the run."""
    return [o["said"] for o in outcomes if o.get("status") == "fi_fault"]
