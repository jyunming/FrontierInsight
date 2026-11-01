"""Replay the recorded figure functions of earlier runs through FI's drawing calls, and report what would be stated about them.

    python dev/evaluation/figure_replay.py --corpus C:/fi_runs/suite/rounds --out <scratch folder>

A local script (CI does not run it). The corpus is only read: every recorded quest's ``code`` folder is copied into
``--out`` and run there. For each ``figure_<id>`` body found in ``sk4*/*/out/*/code/derive.py``:

* the body runs on a *stub table* built from that quest's plan (the grid of ``study.json``, the results ``derive.py``
  declares): synthetic finite values at every declared setting, no real model and no recorded trial;
* the body draws on *proxy axes*. A deterministic proxy maps the matplotlib calls ``plot, errorbar, scatter, axhline,
  axvline, annotate, text, legend, set_*, loglog`` onto ``core.frame_figure.FigureCalls`` (the drawing calls). A call
  outside that set, a number typed in a text, an error bar with unequal sides, a reference line at a value the plan does
  not hold, or a call the drawing calls refuse, makes the panel *raw*, and the call that forced it is kept with the panel;
* ``core.figure_claims.menu`` lists the claims true on the resulting call record, and each one is checked against a plain
  recomputation from the record's arrays (written here, sharing no code with ``figure_claims``).

It prints the bodies counted, panels (verified / raw, with the reason), bodies that fail (the frame would draw FI's default
figure), and the claim agreement. ``--out`` also receives one json file per quest.

The proxy's guesses (which result a ``plot`` draws, which plan id an x is, whether a dashed line is an analytic overlay)
are the rules below, stated once; they are a stand-in for a body written against the drawing calls, so the raw share is a
measure of how much of today's drawing the calls cover, not of how a future body will look.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = Path("C:/fi_runs/suite/rounds")
CODE_FILES = ("derive.py", "fi_frame.py", "fi_table.py", "study.json")
STAMP = re.compile(r"^sk4")
MAPPED = ("plot", "errorbar", "scatter", "axhline", "axvline", "annotate", "text", "legend", "loglog", "semilogx", "semilogy",
          "fill_between", "get_legend_handles_labels")
#: attributes of an axes that are values, not drawing (a transform handed to ``text`` / ``annotate``): reading them is not a call
PLAIN_ATTRIBUTES = ("transAxes", "transData", "transScale")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
TIMEOUT_S = 240
PANEL_NOOP = False


# ---------------------------------------------------------------------------------------------------------------------
# The worker: runs the bodies of one quest in its own process (its fi_frame / fi_table / derive must not meet another's).
# ---------------------------------------------------------------------------------------------------------------------

def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _floats(v):
    """A list of floats (NaN for a gap) from a number, a list or an array; ``None`` when it is not numeric."""
    try:
        import numpy as np

        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError, ImportError):
        return None
    if arr.ndim > 1:
        return None
    return [float(x) for x in arr.reshape(-1)]


def _stub_cells(plan, declared, kinds):
    """Synthetic trial cells: every combination of the plan's grid, every declared result with 30 finite values."""
    import itertools
    import random

    grid = plan.get("grid") if isinstance(plan.get("grid"), dict) else {}
    axes = {k: v for k, v in grid.items() if isinstance(v, list) and v}
    names = sorted(axes)
    combos = list(itertools.product(*[axes[n] for n in names]))[:400] or [()]
    inside = plan.get("in_each_run") if isinstance(plan.get("in_each_run"), dict) else {}
    levels = [(f, v) for f, vs in inside.items() if isinstance(vs, list) for v in vs]
    cells = []
    for c, combo in enumerate(combos):
        cell = dict(zip(names, combo))
        for lvl in [None] + levels:
            metrics = {}
            for r, rid in enumerate(declared):
                rng = random.Random(f"{rid}/{c}/{lvl}")
                if kinds.get(rid) == "proportion":
                    p = 0.05 + 0.9 * ((c * 0.37 + r * 0.11 + (0.05 if lvl else 0)) % 1.0)
                    vals = [1.0 if rng.random() < p else 0.0 for _ in range(30)]
                else:
                    base = 1.0 + (r % 5) + 0.5 * c + (0.2 if lvl else 0.0)
                    vals = [base * (1 + 0.1 * rng.gauss(0, 1)) for _ in range(30)]
                metrics[rid] = {"values": vals, "trials": list(range(30))}
            key = ",".join(f"{n}={cell[n]}" for n in names) or "all"
            entry = {"cell": cell, "key": key, "planned": 30, "ok": 30, "failed": 0, "metrics": metrics}
            if lvl:
                entry = {**entry, "key": key + "|" + f"{lvl[0]}={lvl[1]}", "level": {lvl[0]: lvl[1]}}
            cells.append(entry)
    return cells, axes


def _stand_in():
    """What a call that drew nothing hands back, so a body that keeps the artist (``line, = ax.plot(...)``) goes on."""
    from matplotlib.lines import Line2D

    return [Line2D([], [])]


def _container(line=None):
    """What ``ax.errorbar`` hands back (a body may read ``.lines``)."""
    from matplotlib.container import ErrorbarContainer

    return ErrorbarContainer((line or _stand_in()[0], (), ()), has_xerr=False, has_yerr=True)


class _Run:
    """One body's run: the host that holds the drawing calls, and what the proxy decided."""

    def __init__(self, host, table, plan, axes_grid, spec, refs, held=()):
        self.host, self.table, self.plan, self.grid, self.spec, self.refs = host, table, plan, axes_grid, spec, refs
        self.held = [float(v) for v in held if _number(v)]  # the setting values the study holds fixed for the figure
        self.pending_texts = []  # (panel, index of its cause, [(number, read?)]) of the digit texts that went raw, sorted at the end
        self.reads = []          # (value, result id, setting key, "mean"|"n"|"low"|"high") of every table read the body made
        self.causes = {}         # panel index -> [(kind, call, detail), ...], every cause that made the panel raw
        self.text_classes = collections.Counter()  # what the digit-bearing texts hold
        self.curves = []         # (panel, label, xs, ys) analytic overlays, drawn after the data
        self.scales = {}         # (panel, "x"|"y") -> "log"

    def raw(self, i, kind, call, detail=""):
        """Every cause is kept; the panel is raw from the first."""
        first = i not in self.causes
        self.causes.setdefault(i, []).append((kind, call, detail))
        if first:
            self.host.raw(i)
            self.host._state()[i]["raw_reason"] = f"{kind}: {call}"

    def own_values(self, i):
        """The setting values panel ``i`` itself shows or holds: the x values of its series that sit on a setting axis, and
        the values held for the figure. Not the other values of the study's grid: a small round number (1, 2, 10, 0.5) is
        on some axis of almost every study, so a match with the whole grid would call a typed number a setting value."""
        rec = self.host._state().get(i) or {}
        if (rec.get("x") or {}).get("kind") != "setting":
            return list(self.held)
        xs = [v for s in rec.get("series", []) if s.get("kind") != "curve" for v in s["x"] if _number(v)]
        return xs + self.held

    def sort_texts(self):
        """Class every digit text that went raw, once the body has drawn all its series (a text may come before the data it
        sits next to): a number the table gave is a read; a number equal to a setting value of the panel is a setting value;
        any other number was typed by the body."""
        for i, idx, call, numbers in self.pending_texts:
            own = self.own_values(i)
            grid = [any(_close(n, g) for g in own) for n, _r in numbers]
            reads = [r for _n, r in numbers]
            if any(not r and not g for r, g in zip(reads, grid)):
                kind, label = "text with a number the body typed", "number typed by the body"
            elif len(numbers) > 1 and any(reads):
                kind, label = "text with several numbers (a note carries one)", "several numbers, some read from the table"
            else:
                kind, label = "text with a setting value in it", "number is a setting value"
            self.text_classes[label] += 1
            old = self.causes[i][idx]
            self.causes[i][idx] = (kind, old[1], old[2])
            if idx == 0:
                self.host._state()[i]["raw_reason"] = f"{kind}: {old[1]}"
        self.pending_texts = []

    def mean_reads(self):
        return [(v, r, k) for (v, r, k, f) in self.reads if f == "mean"]

    def read_matches(self, token):
        """The table reads (value, result id, setting key, form) whose value shows as ``token`` at the token's own precision."""
        shown = float(token)
        mantissa = token.lower().split("e")[0]
        digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
        out = []
        for v, rid, key, form in self.reads:
            if digits == 0:
                if _close(v, shown):  # a whole number is a read only when the read is that number
                    out.append((v, rid, key, form))
            elif abs(round(v, digits) - shown) <= 1e-9 * max(1.0, abs(shown)):
                out.append((v, rid, key, form))
        return out

    # ---- guesses of the proxy -------------------------------------------------------------------------------------
    def what_of(self, ys):
        finite = [y for y in ys if _number(y)]
        hit = set()
        for y in finite:
            m = next((r for (v, r, _k) in self.mean_reads() if _close(v, y)), None)
            if m is None:
                return "other"
            hit.add(m)
        return hit.pop() if len(hit) == 1 else "other"

    def x_id_of(self, xs):
        if not xs or not all(_number(x) for x in xs):
            return None
        want = [self.spec.get("x")] + sorted(self.grid)
        for name in want:
            if name in self.grid and all(any(_close(x, g) for g in self.grid[name] if _number(g)) for x in xs):
                return name
        for x in xs[:1]:
            hit = next((r for (v, r, _k) in self.mean_reads() if _close(v, x)), None)
            if hit:
                return hit
        return self.spec.get("x") if self.spec.get("x") in self.grid or self.spec.get("x") in self.host._results() else (sorted(self.grid) or [None])[0]

    def series_id(self, i, what, label):
        used = {s["id"] for s in (self.host._state().get(i) or {"series": []})["series"]}
        base = what if what != "other" else "computed"
        sid, k = base, 1
        while sid in used:
            k += 1
            sid = f"{base}#{k}"
        if sid not in self.host.series_ids:
            self.host.series_ids.append(sid)
        self.host.labels[sid] = label or sid
        return sid

    def yerr_meaning(self, ys, err):
        """``ci95`` when each half-width is the table's own 95% interval half-width of the mean it sits on, ``sd`` when it is
        the standard deviation of that setting's trials, else ``other``."""
        import statistics

        kinds = {"ci95": True, "sd": True}
        try:
            for y, e in zip(ys, err):
                if not _number(y) or not _number(e):
                    continue
                rid, key = next((r, k) for (v, r, k) in self.mean_reads() if _close(v, y))
                lo, hi = self.table.interval(rid, key)
                kinds["ci95"] &= lo is not None and _close((hi - lo) / 2, abs(e))
                vals = self.table.values(rid, key)
                kinds["sd"] &= len(vals) > 1 and _close(statistics.stdev(vals), abs(e))
        except Exception:  # noqa: BLE001 -- a frame without these reads, or a value no read gave
            return "other"
        return next((k for k, ok in kinds.items() if ok), "other")


class _Panel:
    """Stands in for one matplotlib ``Axes``: the mapped calls draw through the drawing calls, anything else makes the panel raw."""

    def __init__(self, run, i, ax):
        self._run, self._i, self._ax = run, i, ax

    def __getattr__(self, name):
        if name.startswith("__"):  # numpy and copy probe these; they are not drawing
            raise AttributeError(name)
        run = self._run
        if name.startswith("set_"):
            if name in ("set_xscale", "set_yscale"):
                return lambda *a, **k: self._scale(name[4], a[0] if a else k.get("value", k.get("scale", "linear")))
            return lambda *a, **k: None
        if name in MAPPED:
            return getattr(self, "_m_" + name)
        if name not in PLAIN_ATTRIBUTES:
            run.raw(self._i, "unmapped call", name)
        return getattr(self._ax, name)

    def _scale(self, axis, scale):
        """The scale the body sets on an axis: the real axes take it (the claims are checked against them) and the record
        gets it when it is log."""
        if scale == "linear":
            return None
        if scale != "log":
            return self._run.raw(self._i, "an axis scale the calls do not hold", "set_%sscale" % axis, str(scale))
        self._run.scales[(self._i, axis)] = "log"
        getattr(self._ax, "set_%sscale" % axis)("log")
        return None

    # ---- argument reading -----------------------------------------------------------------------------------------
    @staticmethod
    def _xy(args):
        """The (x, y, fmt) triples of a ``plot`` call's positional arguments."""
        out, i, args = [], 0, list(args)
        while i < len(args):
            if isinstance(args[i], str):
                i += 1
                continue
            a = args[i]
            b = args[i + 1] if i + 1 < len(args) and not isinstance(args[i + 1], str) else None
            if b is None:
                fmt = args[i + 1] if i + 1 < len(args) else ""
                out.append((None, a, fmt))
                i += 2 if fmt else 1
            else:
                fmt = args[i + 2] if i + 2 < len(args) and isinstance(args[i + 2], str) else ""
                out.append((a, b, fmt))
                i += 3 if fmt else 2
        return out

    @staticmethod
    def _points_style(fmt, kw):
        ls = kw.get("linestyle", kw.get("ls"))
        if ls in ("none", "None", "", " "):
            return True
        return bool(fmt) and not any(c in fmt for c in "-:") and any(c in fmt for c in "o.sv^<>*+xD")

    @staticmethod
    def _dashed(fmt, kw):
        ls = kw.get("linestyle", kw.get("ls")) or ""
        return any(t in str(ls) + (fmt or "") for t in ("--", ":", "-."))

    # ---- the mapped calls -----------------------------------------------------------------------------------------
    def _series(self, call, x, y, fmt, kw, yerr=None):
        run, i, host = self._run, self._i, self._run.host
        ys = _floats(y)
        xs = _floats(x) if x is not None else None
        if x is not None and xs is None:
            try:
                xs = [str(v) for v in x]
            except TypeError:
                xs = None
        if ys is None or (x is not None and xs is None):
            run.raw(i, "values that are not a plain list of numbers", call)
            return _stand_in()
        if x is None:
            xs = [float(k) for k in range(len(ys))]
        if len(xs) != len(ys):
            run.raw(i, "x and y of different length", call)
            return _stand_in()
        label = kw.get("label")
        what = run.what_of(ys)
        if what == "other" and self._dashed(fmt, kw) and yerr is None:
            run.curves.append((i, str(label or "overlay"), xs, ys))
            return _stand_in()
        xid = run.x_id_of(xs) if all(_number(v) for v in xs) else run.spec.get("x")
        if xid is None:
            run.raw(i, "x is not a setting or result of the plan", call)
            return _stand_in()
        sid = run.series_id(i, what, str(label) if label else None)
        extra = {"yerr": yerr[0], "yerr_meaning": yerr[1]} if yerr else {}
        try:
            draw = host.points if self._points_style(fmt, kw) else host.line
            return list(draw(i, sid, (xid, xs), ys, what, **extra))
        except (ValueError, IndexError) as err:
            run.raw(i, "drawing call refused", call, str(err)[:120])
        return _stand_in()

    def _m_plot(self, *args, **kw):
        triples = self._xy(args)
        if not triples:
            self._run.raw(self._i, "unreadable arguments", "plot")
            return _stand_in()
        got = []
        for x, y, fmt in triples:
            got += self._series("plot", x, y, fmt, kw) or []
        return got

    def _m_loglog(self, *args, **kw):
        self._scale("x", "log")
        self._scale("y", "log")
        return self._m_plot(*args, **kw)

    def _m_semilogx(self, *args, **kw):
        self._scale("x", "log")
        return self._m_plot(*args, **kw)

    def _m_semilogy(self, *args, **kw):
        self._scale("y", "log")
        return self._m_plot(*args, **kw)

    def _m_get_legend_handles_labels(self, *a, **kw):
        return self._ax.get_legend_handles_labels(*a, **kw)  # FI owns the legend; reading the handles draws nothing

    def _m_fill_between(self, x, y1, y2=0, *a, **kw):
        run, i, host = self._run, self._i, self._run.host
        xs, lo, hi = _floats(x), _floats(y1), _floats(y2)
        if xs is None or lo is None or hi is None:
            run.raw(i, "values that are not a plain list of numbers", "fill_between")
            return _stand_in()[0]
        lo = lo * len(xs) if len(lo) == 1 else lo
        hi = hi * len(xs) if len(hi) == 1 else hi
        drawn = [e for e in (host._state().get(i) or {"series": []})["series"] if e["kind"] != "curve" and e["x"] == xs]
        if not drawn or len(lo) != len(xs) or len(hi) != len(xs):
            run.raw(i, "a band without a drawn series at its x", "fill_between")
            return _stand_in()[0]
        s = drawn[-1]
        half = [(h - l) / 2 if _number(h) and _number(l) else None for l, h in zip(lo, hi)]
        centred = all(not _number(y) or h is None or _close((l + hh) / 2, y) for y, l, hh, h in zip(s["y"], lo, hi, half))
        meaning = run.yerr_meaning(s["y"], half) if centred else "other"
        rec = host._state()[i]
        try:
            host.band(i, s["id"], (rec["x"]["id"], xs), lo, hi, rec["y"]["what"].split(" / ")[0], meaning)
        except (ValueError, IndexError) as err:
            run.raw(i, "drawing call refused", "fill_between", str(err)[:120])
        return _stand_in()[0]

    def _m_scatter(self, x, y, *a, **kw):
        if any(k in kw for k in ("c", "cmap")) and _floats(kw.get("c")) not in (None, [] ) and len(_floats(kw["c"])) > 1:
            self._run.raw(self._i, "points coloured by a value", "scatter")
            return _stand_in()
        return self._series("scatter", x, y, "o", {**kw, "linestyle": "none"})

    def _m_errorbar(self, x, y, yerr=None, xerr=None, fmt="", **kw):
        run, i = self._run, self._i
        if xerr is not None:
            run.raw(i, "x error bars", "errorbar")
            return _container()
        ys = _floats(y)
        err = None
        if yerr is not None:
            try:
                import numpy as np

                arr = np.asarray(yerr, dtype=float)
            except (TypeError, ValueError):
                run.raw(i, "error bars that are not numbers", "errorbar")
                return _container()
            if arr.ndim == 2:
                if arr.shape[0] != 2 or not np.allclose(arr[0], arr[1], rtol=1e-9, atol=1e-12, equal_nan=True):
                    run.raw(i, "error bars with unequal sides", "errorbar")
                    return _container()
                arr = arr[0]
            err = [float(v) for v in np.broadcast_to(arr, (len(ys or []),)).reshape(-1)] if ys is not None else None
        got = self._series("errorbar", x, y, fmt, kw, yerr=(err, run.yerr_meaning(ys, err)) if err is not None and ys is not None else None)
        return _container(got[0] if got else None)

    def _line_at(self, call, value, axis, kw):
        run, i = self._run, self._i
        v = _floats(value)
        if v is None or len(v) != 1:
            return run.raw(i, "reference line at a value that is not one number", call)
        hit = next((rid for rid, r in run.refs.items() if _close(r["value"], v[0])), None)
        if hit is None:
            return run.raw(i, "reference line at a value the plan does not hold", call)
        try:
            run.host.ref(i, hit, axis=axis)
        except (ValueError, IndexError) as err:
            run.raw(i, "drawing call refused", call, str(err)[:120])
        return None

    def _m_axhline(self, y=0, *a, **kw):
        self._line_at("axhline", y, "y", kw)
        return _stand_in()[0]

    def _m_axvline(self, x=0, *a, **kw):
        self._line_at("axvline", x, "x", kw)
        return _stand_in()[0]

    def _text(self, call, s, x, y, in_data=True):
        """A text. Words only: the calls draw no such text, so the panel is raw. A text with one number that is a value the body
        read from the table is a ``note`` (FI writes the number); any other number is typed by the body (or is a setting value
        of the panel) and the panel is raw. ``in_data`` is False when the body placed the text in axes or figure coordinates:
        ``note`` takes data coordinates, and the limits FI will set are not known yet, so such a note is raw."""
        run, i = self._run, self._i
        s = str(s)
        tokens = list(NUMBER.finditer(s))
        if not tokens:
            run.raw(i, "words-only text", call)
            return None
        reads = [run.read_matches(m.group()) for m in tokens]
        if len(tokens) == 1 and reads[0] and not in_data:
            run.text_classes["number read from the table, placed in axes coordinates"] += 1
            run.raw(i, "a note placed in axes coordinates", call)
            return None
        if len(tokens) == 1 and reads[0]:
            run.text_classes["number read from the table (a note)"] += 1
            t = tokens[0]
            _v, rid, key, form = reads[0][0]
            what = s[:t.start()] + "{" + rid + "}" + s[t.end():]
            try:
                run.host.note(i, x, y, what, form=form, key=key)
            except (ValueError, IndexError) as err:
                run.raw(i, "drawing call refused", call, str(err)[:120])
            return None
        run.raw(i, "text with a number", call)  # which kind of number is decided by ``sort_texts``
        run.pending_texts.append((i, len(run.causes[i]) - 1, call, [(float(m.group()), bool(r)) for m, r in zip(tokens, reads)]))
        return None

    @staticmethod
    def _at(value):
        v = _floats(value) if value is not None else None
        return v[0] if v and len(v) == 1 and _number(v[0]) else 0.0

    def _in_data(self, transform):
        """True when a ``transform=`` keyword is absent or is the data transform of these axes."""
        return transform is None or transform is self._ax.transData  # not ``==``: two transforms of empty axes compare equal

    def _m_text(self, x=None, y=None, s="", *a, **kw):
        return self._text("text", kw.get("s", s), self._at(x), self._at(y), self._in_data(kw.get("transform")))

    def _m_annotate(self, text="", xy=None, *a, **kw):
        xy = kw.get("xy", xy)
        try:
            x, y = xy
        except (TypeError, ValueError):
            x = y = None
        return self._text("annotate", kw.get("text", kw.get("s", text)), self._at(x), self._at(y),
                          str(kw.get("xycoords", "data")) == "data")

    def _m_legend(self, *a, **kw):
        return None


class _FigureProxy:
    """What the recorded frame's ``Figure`` carries: ``axes`` and what the plan says the figure draws. Nothing else."""

    def __init__(self, panels, spec, panel_noop=False):
        if panel_noop:  # the recorded frame has no ``panel`` (bodies call it all the same): count the bodies as running
            self.panel = lambda *a, **k: None
        self.axes = panels
        self.x, self.series, self.shows, self.file = spec.get("x"), spec.get("series"), spec.get("shows"), spec.get("file")


def _fmt(v):
    return "%.4g" % v


# ---- the independent recomputation (shares no code with core/figure_claims) --------------------------------------------

_MEANING_WORDS = {"ci95": "95% confidence interval", "sd": "standard deviation", "range": "range"}


def _pairs(s):
    return [(x, y) for x, y in zip(s["x"], s["y"]) if y is not None and x is not None and not isinstance(x, str)]


def _numbers_in(text, calls):
    """The numbers a sentence states: its text without the labels of the figure (which may hold digits) and without the
    panel suffix."""
    labels = set()
    for p in calls["panels"]:
        labels.update(str(s.get(k) or "") for s in p["series"] for k in ("label", "id"))
        labels.update(str((p.get(a) or {}).get("label") or "") for a in ("x", "y"))
        labels.update(str(r.get("label") or "") for r in p["refs"])
    for label in sorted(labels, key=len, reverse=True):
        if label:
            text = text.replace(label, " ")
    text = re.sub(r"in panel \d+", " ", text)
    return [float(m.group()) for m in NUMBER.finditer(text)]


def _states(tokens, wanted):
    """True when every wanted number is stated by a distinct number of the sentence (the 4 digits the sentence prints)."""
    pool = list(tokens)
    for n in wanted:
        hit = next((k for k, t in enumerate(pool) if (t == 0 if n == 0 else abs(t - n) <= 1e-3 * abs(n))), None)
        if hit is None:
            return False
        pool.pop(hit)
    return True


def _agree(calls, offer, grid, scale_of):
    """``("agree"|"disagree"|"not recomputed", why)`` for one offered claim, from the record's arrays alone. Numbers are matched
    by value (relative 1e-3, the 4 digits a sentence prints) among the numbers the sentence states, each used once. ``scale_of``
    reads the axis scale from the real axes the body set, not from the record."""
    cid, a, text = offer["claim"], offer["args"], offer["text"]
    panel = next((p for p in calls["panels"] if str(p["panel"]) == a.get("panel")), None)
    sers = {s["id"]: s for s in (panel or {}).get("series", [])}
    tol = 5e-4
    tokens = _numbers_in(text, calls)

    def says(*nums):
        return _states(tokens, nums)

    def tie(p, q):
        return abs(p - q) <= tol * max(abs(p), abs(q))

    def ok(good, why):
        return ("agree", "") if good else ("disagree", why)

    try:
        if cid == "not_drawn":
            every = {s["id"] for p in calls["panels"] for s in p["series"]} | {m for p in calls["panels"] for s in p["series"] for m in s["merged_with"]}
            return ok(a["series"] not in every and not any(p["raw"] for p in calls["panels"]), "it is drawn, or a panel is raw")
        if cid == "scale":
            return ok(scale_of(int(a["panel"]), a["axis"]) == ("logarithmic" in text and "log" or "linear"), "axis scale of the real axes")
        if cid == "reference":
            r = next(r for r in panel["refs"] if r["id"] == a["ref"])
            return ok(says(r["value"]) and f"{r['axis']} = " in text, "value or axis")
        if cid == "note":
            n = next(n for n in panel["notes"] if n["id"] == a["note"])
            return ok(n["text"] == text, "text")
        if cid == "band":
            s = sers[a["series"]]
            b = next(b for b in panel["bands"] if b["series"] == a["series"])
            pairs = [(l, h) for l, h in zip(b["lo"], b["hi"]) if l is not None and h is not None]
            good = (b["x"] == s["x"] and len(b["lo"]) == len(b["hi"]) == len(s["x"]) and pairs and all(l <= h for l, h in pairs)
                    and _MEANING_WORDS.get(b["meaning"], "\0") in text)
            return ok(good, "band x, lo/hi or meaning")
        if cid in ("crossing", "no_crossing", "identical"):
            sa, sb = sers[a["a"]], sers[a["b"]]
            ya, yb = dict(_pairs(sa)), dict(_pairs(sb))
            if cid == "identical":
                return ok(sorted(ya) == sorted(yb) and all(_close(ya[k], yb[k]) for k in ya), "values differ")
            common = sorted(set(ya) & set(yb))
            signs = [(ya[x] - yb[x]) > 0 for x in common]
            crossed = any(p != q for p, q in zip(signs, signs[1:]))
            return ok(crossed == (cid == "crossing"), "sign changes %s" % crossed)
        s = sers.get(a.get("series", ""))
        if cid == "drawn":
            good = s is not None and text.startswith(str(s["label"])) and (
                any(v is not None for v in s["y"]) or (s["kind"] == "image" and any(v is not None for row in s["z"] for v in row)))
            return ok(good, "series or values")
        if cid == "range":
            if s is not None and s["kind"] == "image":
                vals = [v for row in s["z"] for v in row if v is not None]
            else:
                chosen = [s] if s else [t for t in panel["series"] if t["kind"] != "image"]
                vals = [x for t in chosen for x, _y in _pairs(t)] if a.get("axis", "y") == "x" else [y for t in chosen for y in t["y"] if y is not None]
            return ok(vals and says(min(vals), max(vals)), "min/max")
        if cid == "value_at":
            want = a["x"]
            ys = {y for x, y in zip(s["x"], s["y"]) if y is not None and x is not None and (
                x == want if isinstance(x, str) else _close(x, float(want)))}
            nums = [next(iter(ys))] + ([] if isinstance(next(iter(s["x"])), str) else [float(want)]) if len(ys) == 1 else None
            return ok(nums is not None and says(*nums), "y at x")
        if cid in ("trend", "steps"):
            pts = sorted(_pairs(s))
            if len(pts) < 2:
                return ("disagree", "few points")
            first, last = pts[0][1], pts[-1][1]
            if cid == "steps":
                word = "rises" if last > first else "falls"
                return ok(s["step"] and word in text and says(first, last, pts[0][0], pts[-1][0]), "step, direction or end values")
            ups = sum(1 for (_a, p), (_b, q) in zip(pts, pts[1:]) if q > p and not tie(p, q))
            downs = sum(1 for (_a, p), (_b, q) in zip(pts, pts[1:]) if q < p and not tie(p, q))
            if "increases" in text:
                good = ups > 0 and downs == 0 and last > first
            elif "decreases" in text:
                good = downs > 0 and ups == 0 and last < first
            else:
                good = (ups > 0 and downs > 0) or tie(first, last) or ups + downs == 0
            return ok(good, "direction")
        if cid == "order":
            want = a["at"]
            vals = []
            for t in panel["series"]:
                if t["kind"] in ("curve", "image"):
                    continue
                ys = {y for x, y in zip(t["x"], t["y"]) if y is not None and x is not None and (
                    x == want if isinstance(x, str) else _close(x, float(want)))}
                if len(ys) == 1:
                    vals.append((next(iter(ys)), t["label"]))
            vals.sort(key=lambda v: -v[0])
            # the labels must follow one another in the sentence, each as a whole item of the list (one label can open another:
            # "computed#2" opens "computed#29"), so each is searched for after the end of the one before
            at, found = text.find("highest first, are"), len(vals) >= 2
            for _v, label in vals:
                hit = re.compile(re.escape(label) + r"(?=,|\s+in panel|$)").search(text, max(at, 0))
                found = found and hit is not None
                at = hit.end() if hit else at
            return ok(found, "order of the values")
        if cid == "errorbars":
            e = s["yerr"] if s else None
            good = (e is not None and len(e) == len(s["y"]) and all(v is None or v >= 0 for v in e) and any((v or 0) > 0 for v in e)
                    and _MEANING_WORDS.get(s.get("yerr_meaning"), "\0") in text)
            return ok(good, "bar widths or meaning")
        if cid == "missing_x":
            drawn = [x for x, _y in _pairs(s)]
            gone = [g for g in grid if not any(_close(g, x) for x in drawn)]
            return ok(s["kind"] == "points" and gone and says(*gone), "grid")
    except Exception as err:  # noqa: BLE001
        return ("disagree", "recompute failed: %s" % err)
    return ("not recomputed", cid)


def _artists_match(host):
    """Mismatches between the record's series and the lines/bars the host really drew (by legend label): a check that the
    record states what the axes hold."""
    bad = 0
    for p in host.record()["panels"]:
        ax = host.axes[p["panel"] - 1]
        for s in p["series"]:
            if s["kind"] not in ("line", "points"):
                continue
            ys = [None if v is None else float(v) for v in s["y"]]
            found = False
            for ln in ax.lines:
                if ln.get_label() != s["label"] or not str(ln.get_gid() or "").startswith("fi:"):
                    continue
                got = [None if v != v else float(v) for v in ln.get_ydata()]
                found = found or got == ys
            bad += not found
    return bad


# ---- the worker entry --------------------------------------------------------------------------------------------------

def _worker(workdir: Path, result_path: Path) -> None:
    import importlib

    global PANEL_NOOP
    PANEL_NOOP = bool(os.environ.get("FI_REPLAY_PANEL_NOOP"))

    sys.dont_write_bytecode = True
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(workdir))
    sys.path.insert(1, str(ROOT))
    from core import figure_claims, frame_figure

    derive = importlib.import_module("derive")
    plan = json.loads((workdir / "study.json").read_text(encoding="utf-8"))
    declared = list(getattr(derive, "RESULTS", ())) + [r for r in getattr(derive, "AFTERWARDS", ()) if r not in getattr(derive, "RESULTS", ())]
    cells, grid = _stub_cells(plan, declared, getattr(derive, "KINDS", {}))
    import fi_frame

    refs = {}
    consts = {}
    for src in (getattr(derive, "HELD", {}), getattr(derive, "FIXED", {}), plan.get("thresholds") or {}):
        consts.update({k: v for k, v in src.items() if _number(v)})
    for k, v in consts.items():
        refs[k] = {"value": float(v), "label": k}
    out = []
    for fid, spec in (getattr(derive, "FIGURES", {}) or {}).items():
        item = {"figure": fid, "status": "ok", "error": None, "panels": [], "offers": []}
        out.append(item)
        body = getattr(derive, "figure_%s" % fid, None)
        if body is None:
            item.update(status="failed", error="no body figure_%s" % fid)
            continue
        table = fi_frame.Table(cells, declared, parts=getattr(derive, "PARTS", {}))
        fig, axes = plt.subplots(1, max(1, int(spec.get("panels") or 1)), squeeze=False)
        host = type("Host", (frame_figure.FigureCalls,), {})()
        host.axes, host.table, host.x = list(axes[0]), table, spec.get("x")
        host.settings, host.series_ids, host.references = sorted(grid), [], refs
        host.labels = {}
        for r in declared:
            host.labels[r] = r
        run = _Run(host, table, plan, grid, spec, refs, held=list((getattr(derive, "HELD", {}) or {}).values()))
        # what the y check does: every series call, with the result the proxy asked for and the one the frame recorded
        y_log, pending = [], []
        checked = host._checked_what

        def counted(what, form, x_id, xs, ys):
            pending.append(checked(what, form, x_id, xs, ys))
            return pending[-1]

        added = host._add_series

        def add_series(panel, kind, series, x, y, what, *rest, **more):
            pending.clear()
            got = added(panel, kind, series, x, y, what, *rest, **more)
            if pending:
                y_log.append((panel, series, what, pending[-1]))
            return got

        host._checked_what, host._add_series = counted, add_series

        class _Logged:
            def __getattr__(self, name):
                got = getattr(table, name)
                if name in ("mean", "n"):
                    def read(rid, key):
                        v = got(rid, key)
                        if _number(v):
                            run.reads.append((float(v), rid, key, name))
                        return v
                    return read
                if name == "interval":
                    def interval(rid, key):
                        lo, hi = got(rid, key)
                        run.reads += [(float(v), rid, key, form) for v, form in ((lo, "low"), (hi, "high")) if _number(v)]
                        return lo, hi
                    return interval
                return got

        proxy = _FigureProxy([_Panel(run, i, ax) for i, ax in enumerate(host.axes)], spec, PANEL_NOOP)
        try:
            body(proxy, _Logged())
        except BaseException as err:  # noqa: BLE001 -- a body that raises is a body the frame would replace
            item.update(status="failed", error="%s: %s" % (type(err).__name__, str(err)[:160]))
            item["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
        if item["status"] == "failed":
            plt.close(fig)
            continue
        run.sort_texts()
        for i, label, xs, ys in run.curves:
            if i in run.causes or not host._state().get(i, {}).get("series"):
                continue  # a raw panel states nothing; a panel without data stays empty (the frame replaces such a body)
            try:
                host.curve(i, label, xs, ys)
            except (ValueError, IndexError) as err:
                run.raw(i, "drawing call refused", "curve", str(err)[:120])
        st = host._state()
        for (i, axis), scale in run.scales.items():
            rec = st.get(i)
            if rec and rec[axis] is not None:
                rec[axis]["scale"] = scale
        calls = host.record()
        empty = []
        for p in calls["panels"]:
            i = p["panel"] - 1
            if p["raw"]:
                causes = run.causes.get(i) or [("fig.raw", "", "")]
                item["panels"].append({"panel": p["panel"], "state": "raw", "causes": sorted({(k, c) for k, c, _d in causes}),
                                       "detail": causes[0][2]})
            elif not (p["series"] or p["refs"] or p["notes"]):
                empty.append(p["panel"])
                item["panels"].append({"panel": p["panel"], "state": "empty"})
            else:
                item["panels"].append({"panel": p["panel"], "state": "verified"})
        item["text_classes"] = dict(run.text_classes)
        item["record_vs_artists_mismatches"] = _artists_match(host)
        item["scales"] = {str(i + 1): {a: ax.get_xscale() if a == "x" else ax.get_yscale() for a in ("x", "y")}
                          for i, ax in enumerate(host.axes)}
        if empty:
            item.update(status="failed", error="drew nothing on panel %s" % " and ".join(map(str, empty)))
        gridx = [float(g) for g in grid.get(spec.get("x"), []) if _number(g)]
        offers = figure_claims.menu(calls, {"series": list(spec.get("shows") or []), "grid": gridx}, _fmt, stem=fid)
        item["offers_capped"] = len(figure_claims.cap_menu(offers, calls))
        # the y check: the series whose result the frame set to ``other`` because the drawn y is not the table's, and the
        # claims the same record would have offered if the result had been kept
        asked = [(p, sid, req) for p, sid, req, got in y_log if req != "other"]
        down = [(p, sid, req) for p, sid, req, got in y_log if req != "other" and got == "other"]
        item["y_check"] = {"series_asked": len(asked), "downgraded": len(down)}
        if down:
            import copy

            kept = copy.deepcopy(calls)
            for p, sid, req in down:
                panel = kept["panels"][p]
                for s in panel["series"]:
                    if s["id"] == sid:
                        s["what"] = req
                panel["y"]["what"] = " / ".join(dict.fromkeys(r for q, _i, r, _g in y_log if q == p))
            key = lambda o: (o["claim"], json.dumps(o["args"], sort_keys=True))  # noqa: E731
            now = {key(o) for o in offers}
            more = figure_claims.menu(kept, {"series": list(spec.get("shows") or []), "grid": gridx}, _fmt, stem=fid)
            item["withdrawn_by_y_check"] = [o["claim"] for o in more if key(o) not in now]
        for o in offers:
            verdict, why = _agree(calls, o, gridx, lambda n, axis: item["scales"][str(n)][axis])
            item["offers"].append({"claim": o["claim"], "args": o["args"], "text": o["text"], "agree": verdict, "why": why})
        # the same sentence must come back from evaluate (the writer's mark)
        for o, off in zip(offers, item["offers"]):
            again = figure_claims.evaluate(calls, o["claim"], dict(o["args"]), _fmt, plan={"series": list(spec.get("shows") or []), "grid": gridx})
            off["same_on_evaluate"] = bool(again) and again[0] == o["text"]
        plt.close(fig)
    result_path.write_text(json.dumps({"figures": out, "declared": declared, "grid": {k: len(v) for k, v in grid.items()}}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------------------------------------------------

def _quests(corpus: Path):
    for rnd in sorted(p for p in corpus.iterdir() if p.is_dir() and STAMP.match(p.name)):
        for derive in sorted(rnd.glob("*/out/*/code/derive.py")):
            yield rnd.name, derive.parents[2].parent.name, derive.parent


def _run_quest(rnd: str, topic: str, code: Path, out: Path):
    name = f"{rnd}__{topic}"
    work = out / "work" / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for f in CODE_FILES:
        if (code / f).is_file():
            shutil.copy2(code / f, work / f)
    text = (code / "derive.py").read_text(encoding="utf-8")
    bodies = len(re.findall(r"^def figure_", text, re.M))
    result = out / f"{name}.json"
    if result.exists():
        result.unlink()
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
    try:
        done = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(work), str(result)],
                              capture_output=True, text=True, timeout=TIMEOUT_S, creationflags=flags, cwd=str(work))
        error = None if result.is_file() else (done.stderr.strip().splitlines() or ["no result"])[-1]
    except subprocess.TimeoutExpired:
        error = "timed out"
    if error:
        return {"name": name, "bodies": bodies, "error": error, "figures": []}
    data = json.loads(result.read_text(encoding="utf-8"))
    return {"name": name, "bodies": bodies, "error": None, "figures": data["figures"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="folder holding the rounds (sk4a, sk4b, ...)")
    ap.add_argument("--out", type=Path, required=True, help="scratch folder for the copies and the per-quest json")
    ap.add_argument("--panel-noop", action="store_true",
                    help="let a body call fig.panel(...), which the recorded frame lacks, as a call that does nothing")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N quests (a trial)")
    args = ap.parse_args(argv)
    corpus, out = args.corpus.resolve(), args.out.resolve()
    if corpus == out or corpus in out.parents:
        ap.error("--out must not be the corpus or inside it: the corpus is only read")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["FI_REPLAY_PANEL_NOOP"] = "1" if args.panel_noop else ""
    quests = list(_quests(corpus))
    if args.limit:
        quests = quests[:args.limit]
    done = [_run_quest(r, t, c, out) for r, t, c in quests]

    figs = [(q["name"], f) for q in done for f in q["figures"]]
    with_bodies = sum(1 for q in done if q["bodies"])
    print(f"files counted: {len(done)}; files with figure bodies: {with_bodies}; bodies (def figure_): {sum(q['bodies'] for q in done)}; "
          f"bodies run: {len(figs)}")
    crashed = [q for q in done if q["error"]]
    if crashed:
        print(f"quests whose worker did not finish: {len(crashed)}")
        for q in crashed:
            print(f"  {q['name']}: {q['error']}")
    panels = [(n, f["figure"], p) for n, f in figs for p in f["panels"]]
    counts = collections.Counter(p["state"] for _n, _f, p in panels)
    total = len(panels)
    share = f" ({100 * counts['raw'] / total:.0f}%)" if total else ""
    print(f"panels (of bodies that ran to the end): {total}; verified {counts['verified']}; raw {counts['raw']}{share}; empty {counts['empty']}")
    raw = [p for _n, _f, p in panels if p["state"] == "raw"]
    per_cause = collections.Counter(c for p in raw for c in {tuple(x) for x in p["causes"]})
    alone = collections.Counter(tuple(p["causes"][0]) for p in raw if len(p["causes"]) == 1)
    print("causes of raw panels, counted per panel (a panel may have several, so the numbers add to more than the raw panels);"
          " 'alone' = panels whose only cause it is, which one change to it would free:")
    for (kind, call), k in per_cause.most_common():
        print(f"  {k:3d} panels (alone {alone[(kind, call)]:3d})  {kind}" + (f" ({call})" if call else ""))
    classes = collections.Counter()
    for _n, f in figs:
        classes.update(f.get("text_classes", {}))
    print("texts with a number, by where the number comes from: " + ("; ".join(f"{k} {v}" for k, v in classes.most_common()) or "none"))
    failing = [(n, f) for n, f in figs if f["status"] == "failed"]
    print(f"bodies that fail (the frame would draw the default figure): {len(failing)} of {len(figs)}")
    ft = collections.Counter((f["error"] or "(no message)")[:90] for _n, f in failing)
    for text, k in ft.most_common(12):
        print(f"  {k:3d}  {text}")
    offers = [o for _n, f in figs for o in f["offers"]]
    by = collections.Counter(o["agree"] for o in offers)
    print(f"claims offered: {len(offers)}; recomputed and agree {by['agree']}; disagree {by['disagree']}; not recomputed {by['not recomputed']}; "
          f"same sentence from evaluate: {sum(1 for o in offers if o['same_on_evaluate'])}")
    print("  per claim: offered / agree / disagree (numbers matched by value; scale read from the real axes; band, error bars, "
          "drawn check the arrays)")
    for claim, k in collections.Counter(o["claim"] for o in offers).most_common():
        good = sum(1 for o in offers if o["claim"] == claim and o["agree"] == "agree")
        bad = sum(1 for o in offers if o["claim"] == claim and o["agree"] == "disagree")
        print(f"  {claim:12s} {k:4d} / {good:4d} / {bad:3d}")
    for n, f in figs:
        for o in f["offers"]:
            if o["agree"] == "disagree":
                print(f"  DISAGREE {n} {f['figure']}: {o['claim']} {o['args']} -> {o['text']!r} ({o['why']})")
    capped = sum(f.get("offers_capped", 0) for _n, f in figs)
    sys.path.insert(0, str(ROOT))
    from core import figure_claims

    print(f"claims offered per figure, cut to the menu limit of {figure_claims.MENU_LIMIT}: {len(offers)} before, {capped} after; "
          f"figures over the limit: {sum(1 for _n, f in figs if len(f['offers']) > figure_claims.MENU_LIMIT)}")
    asked = sum(f.get("y_check", {}).get("series_asked", 0) for _n, f in figs)
    down = sum(f.get("y_check", {}).get("downgraded", 0) for _n, f in figs)
    gone = collections.Counter(c for _n, f in figs for c in f.get("withdrawn_by_y_check", []))
    print(f"the y check: series drawn as a result of the plan {asked}; set to 'other' (drawn y is not the table's) {down}; "
          f"claims withdrawn by it: {sum(gone.values())}" + ("".join(f"; {c} {k}" for c, k in gone.most_common())))
    print("fi_fault: not measured here (the replay draws through the proxy, not through the figure code of a run, which is what "
          "raises it); the nearest check is the line below")
    print(f"series whose record differs from the line really drawn: {sum(f.get('record_vs_artists_mismatches', 0) for _n, f in figs)}")
    shared = [f for _n, f in figs if f["status"] == "ok" and any(p["state"] != "empty" for p in f["panels"])]
    print(f"bodies that draw (no fallback): {len(shared)}; of them with every panel verified: "
          f"{sum(1 for f in shared if all(p['state'] == 'verified' for p in f['panels']))}")
    (out / "summary.json").write_text(json.dumps({"quests": done}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    raise SystemExit(main())
