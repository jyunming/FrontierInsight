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
MAPPED = ("plot", "errorbar", "scatter", "axhline", "axvline", "annotate", "text", "legend", "loglog")
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

    def __init__(self, host, table, plan, axes_grid, spec, refs):
        self.host, self.table, self.plan, self.grid, self.spec, self.refs = host, table, plan, axes_grid, spec, refs
        self.reads = []          # (mean, result id, setting key) of every table.mean the body made
        self.raw_by = {}         # panel index -> (kind, call, detail), the first reason
        self.curves = []         # (panel, label, xs, ys) analytic overlays, drawn after the data
        self.scales = {}         # (panel, "x"|"y") -> "log"
        self.dropped_texts = 0
        self.counter = collections.Counter()

    def raw(self, i, kind, call, detail=""):
        self.counter[(kind, call)] += 1
        if i not in self.raw_by:
            self.raw_by[i] = (kind, call, detail)
            self.host.raw(i)
            self.host._state()[i]["raw_reason"] = f"{kind}: {call}"

    # ---- guesses of the proxy -------------------------------------------------------------------------------------
    def what_of(self, ys):
        finite = [y for y in ys if _number(y)]
        hit = set()
        for y in finite:
            m = next((r for (v, r, _k) in self.reads if _close(v, y)), None)
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
            hit = next((r for (v, r, _k) in self.reads if _close(v, x)), None)
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
        """``ci95`` when each half-width is the table's own 95% interval half-width of the mean it sits on."""
        try:
            for y, e in zip(ys, err):
                if not _number(y):
                    continue
                rid, key = next((r, k) for (v, r, k) in self.reads if _close(v, y))
                lo, hi = self.table.interval(rid, key)
                if lo is None or not _close((hi - lo) / 2, abs(e)):
                    return "other"
            return "ci95"
        except Exception:  # noqa: BLE001
            return "other"


class _Panel:
    """Stands in for one matplotlib ``Axes``: the mapped calls draw through the drawing calls, anything else makes the panel raw."""

    def __init__(self, run, i, ax):
        self._run, self._i, self._ax = run, i, ax

    def __getattr__(self, name):
        run = self._run
        if name.startswith("set_"):
            if name in ("set_xscale", "set_yscale"):
                return lambda scale="linear", *a, **k: run.scales.__setitem__((self._i, name[4]), scale) if scale == "log" else None
            return lambda *a, **k: None
        if name in MAPPED:
            return getattr(self, "_m_" + name)
        run.raw(self._i, "unmapped call", name)
        return getattr(self._ax, name)

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
        self._run.scales[(self._i, "x")] = self._run.scales[(self._i, "y")] = "log"
        return self._m_plot(*args, **kw)

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

    def _text(self, call, s):
        if re.search(r"\d", str(s)):
            return self._run.raw(self._i, "text with a number in it", call)
        self._run.dropped_texts += 1  # words only: nothing in the calls draws them, and nothing is claimed about them
        return None

    def _m_text(self, x=None, y=None, s="", *a, **kw):
        return self._text("text", kw.get("s", s))

    def _m_annotate(self, text="", *a, **kw):
        return self._text("annotate", kw.get("text", kw.get("s", text)))

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

def _pairs(s):
    return [(x, y) for x, y in zip(s["x"], s["y"]) if y is not None and x is not None and not isinstance(x, str)]


def _agree(calls, offer, grid):
    """``("agree"|"disagree"|"not recomputed", why)`` for one offered claim, from the record's arrays alone."""
    cid, a, text = offer["claim"], offer["args"], offer["text"]
    panel = next((p for p in calls["panels"] if str(p["panel"]) == a.get("panel")), None)
    sers = {s["id"]: s for s in (panel or {}).get("series", [])}
    tol = 5e-4

    def has(*nums):
        return all(_fmt(n) in text for n in nums)

    def tie(p, q):
        return abs(p - q) <= tol * max(abs(p), abs(q))

    try:
        if cid == "not_drawn":
            every = [s["id"] for p in calls["panels"] for s in p["series"]]
            return ("agree", "") if a["series"] not in every and not any(p["raw"] for p in calls["panels"]) else ("disagree", "drawn")
        if cid == "scale":
            word = "logarithmic" if panel[a["axis"]]["scale"] == "log" else "linear"
            return ("agree", "") if word in text else ("disagree", word)
        if cid == "reference":
            r = next(r for r in panel["refs"] if r["id"] == a["ref"])
            return ("agree", "") if has(r["value"]) else ("disagree", "value")
        if cid == "note":
            n = next(n for n in panel["notes"] if n["id"] == a["note"])
            return ("agree", "") if n["text"] in text else ("disagree", "text")
        if cid == "band":
            return ("agree", "") if any(b["series"] == a["series"] and b["meaning"] != "other" for b in panel["bands"]) else ("disagree", "no band")
        if cid in ("crossing", "no_crossing", "identical"):
            sa, sb = sers[a["a"]], sers[a["b"]]
            ya, yb = dict(_pairs(sa)), dict(_pairs(sb))
            if cid == "identical":
                return ("agree", "") if ya == yb or (sorted(ya) == sorted(yb) and all(_close(ya[k], yb[k]) for k in ya)) else ("disagree", "differ")
            common = sorted(set(ya) & set(yb))
            signs = [(ya[x] - yb[x]) > 0 for x in common]
            crossed = any(p != q for p, q in zip(signs, signs[1:]))
            return ("agree", "") if crossed == (cid == "crossing") else ("disagree", "sign changes %s" % crossed)
        s = sers.get(a.get("series", ""))
        if cid == "drawn":
            ok = s is not None and (any(y is not None for y in s["y"]) or s["kind"] == "image")
            return ("agree", "") if ok else ("disagree", "no values")
        if cid == "range":
            if s is not None and s["kind"] == "image":
                vals = [v for row in s["z"] for v in row if v is not None]
            else:
                chosen = [s] if s else [t for t in panel["series"] if t["kind"] != "image"]
                if a.get("axis", "y") == "x":
                    vals = [x for t in chosen for x, _y in _pairs(t)]
                else:
                    vals = [y for t in chosen for y in t["y"] if y is not None]
            return ("agree", "") if vals and has(min(vals), max(vals)) else ("disagree", "min/max")
        if cid == "value_at":
            want = float(a["x"])
            ys = {y for x, y in _pairs(s) if _close(x, want)}
            return ("agree", "") if len(ys) == 1 and has(next(iter(ys))) else ("disagree", "y at x")
        if cid in ("trend", "steps"):
            pts = sorted(_pairs(s))
            if len(pts) < 2:
                return ("disagree", "few points")
            first, last = pts[0][1], pts[-1][1]
            if cid == "steps":
                word = "rises" if last > first else "falls"
                return ("agree", "") if word in text and has(first, last) else ("disagree", "first/last")
            ups = sum(1 for (_a, p), (_b, q) in zip(pts, pts[1:]) if q > p and not tie(p, q))
            downs = sum(1 for (_a, p), (_b, q) in zip(pts, pts[1:]) if q < p and not tie(p, q))
            if "increases" in text:
                ok = ups > 0 and downs == 0 and last > first
            elif "decreases" in text:
                ok = downs > 0 and ups == 0 and last < first
            else:
                ok = (ups > 0 and downs > 0) or tie(first, last) or ups + downs == 0
            return ("agree", "") if ok else ("disagree", "direction")
        if cid == "order":
            want = float(a["at"])
            vals = []
            for t in panel["series"]:
                if t["kind"] in ("curve", "image"):
                    continue
                ys = {y for x, y in _pairs(t) if _close(x, want)}
                if len(ys) == 1:
                    vals.append((next(iter(ys)), t["label"]))
            vals.sort(key=lambda v: -v[0])
            where = [text.find(label) for _v, label in vals]
            return ("agree", "") if len(vals) >= 2 and all(w >= 0 for w in where) and where == sorted(where) else ("disagree", "order")
        if cid == "errorbars":
            ok = s is not None and s.get("yerr") and any((e or 0) > 0 for e in s["yerr"]) and s.get("yerr_meaning") in ("ci95", "sd", "range")
            return ("agree", "") if ok else ("disagree", "no error bars")
        if cid == "missing_x":
            drawn = [x for x, _y in _pairs(s)]
            gone = [g for g in grid if not any(_close(g, x) for x in drawn)]
            return ("agree", "") if gone and has(*gone) else ("disagree", "grid")
    except Exception as err:  # noqa: BLE001
        return ("disagree", "recompute failed: %s" % err)
    return ("not recomputed", cid)


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
        run = _Run(host, table, plan, grid, spec, refs)

        class _Logged:
            def __getattr__(self, name):
                got = getattr(table, name)
                if name == "mean":
                    def mean(rid, key):
                        v = got(rid, key)
                        if _number(v):
                            run.reads.append((v, rid, key))
                        return v
                    return mean
                return got

        proxy = _FigureProxy([_Panel(run, i, ax) for i, ax in enumerate(host.axes)], spec, PANEL_NOOP)
        try:
            body(proxy, _Logged())
        except BaseException as err:  # noqa: BLE001 -- a body that raises is a body the frame would replace
            item.update(status="failed", error="%s: %s" % (type(err).__name__, str(err)[:160]))
            item["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
        plt.close(fig)
        if item["status"] == "failed":
            continue
        for i, label, xs, ys in run.curves:
            if i in run.raw_by or not host._state().get(i, {}).get("series"):
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
                kind, call, detail = run.raw_by.get(i, ("fig.raw", "", ""))
                item["panels"].append({"panel": p["panel"], "state": "raw", "reason": kind, "call": call, "detail": detail})
            elif not (p["series"] or p["refs"] or p["notes"]):
                empty.append(p["panel"])
                item["panels"].append({"panel": p["panel"], "state": "empty"})
            else:
                item["panels"].append({"panel": p["panel"], "state": "verified"})
        item["dropped_texts"] = run.dropped_texts
        if empty:
            item.update(status="failed", error="drew nothing on panel %s" % " and ".join(map(str, empty)))
        gridx = [float(g) for g in grid.get(spec.get("x"), []) if _number(g)]
        offers = figure_claims.menu(calls, {"series": list(spec.get("shows") or []), "grid": gridx}, _fmt, stem=fid)
        for o in offers:
            verdict, why = _agree(calls, o, gridx)
            item["offers"].append({"claim": o["claim"], "args": o["args"], "text": o["text"], "agree": verdict, "why": why})
        # the same sentence must come back from evaluate (the writer's mark)
        for o, off in zip(offers, item["offers"]):
            again = figure_claims.evaluate(calls, o["claim"], dict(o["args"]), _fmt, plan={"series": list(spec.get("shows") or []), "grid": gridx})
            off["same_on_evaluate"] = bool(again) and again[0] == o["text"]
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
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["FI_REPLAY_PANEL_NOOP"] = "1" if args.panel_noop else ""
    quests = list(_quests(args.corpus))
    if args.limit:
        quests = quests[:args.limit]
    done = [_run_quest(r, t, c, args.out) for r, t, c in quests]

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
    print(f"panels (of bodies that ran to the end): {len(panels)}; verified {counts['verified']}; raw {counts['raw']}; empty {counts['empty']}")
    why = collections.Counter((p["reason"], p["call"]) for _n, _f, p in panels if p["state"] == "raw")
    for (reason, call), k in why.most_common():
        print(f"  raw {k:3d}  {reason}" + (f" ({call})" if call else ""))
    failing = [(n, f) for n, f in figs if f["status"] == "failed"]
    print(f"bodies that fail (the frame would draw the default figure): {len(failing)} of {len(figs)}")
    ft = collections.Counter((f["error"] or "(no message)")[:90] for _n, f in failing)
    for text, k in ft.most_common(12):
        print(f"  {k:3d}  {text}")
    offers = [o for _n, f in figs for o in f["offers"]]
    by = collections.Counter(o["agree"] for o in offers)
    print(f"claims offered: {len(offers)}; recomputed and agree {by['agree']}; disagree {by['disagree']}; not recomputed {by['not recomputed']}; "
          f"same sentence from evaluate: {sum(1 for o in offers if o['same_on_evaluate'])}")
    for claim, k in collections.Counter(o["claim"] for o in offers).most_common():
        bad = sum(1 for o in offers if o["claim"] == claim and o["agree"] == "disagree")
        print(f"  {claim:12s} {k:4d}" + (f"  disagree {bad}" if bad else ""))
    for n, f in figs:
        for o in f["offers"]:
            if o["agree"] == "disagree":
                print(f"  DISAGREE {n} {f['figure']}: {o['claim']} {o['args']} -> {o['text']!r} ({o['why']})")
    shared = [f for _n, f in figs if f["status"] == "ok" and any(p["state"] != "empty" for p in f["panels"])]
    print(f"bodies that draw (no fallback): {len(shared)}; of them with every panel verified: "
          f"{sum(1 for f in shared if all(p['state'] == 'verified' for p in f['panels']))}")
    print(f"texts with words only that the calls do not draw: {sum(f.get('dropped_texts', 0) for _n, f in figs)}")
    (args.out / "summary.json").write_text(json.dumps({"quests": done}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]), Path(sys.argv[3]))
        raise SystemExit(0)
    raise SystemExit(main())
