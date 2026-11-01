"""The drawing calls of a figure the analysis FI frames, and the record of exactly what they drew.

A ``figure_<id>`` body draws only through these calls (``fig.line``, ``fig.points``, ``fig.bars``, ``fig.band``, ``fig.ref``,
``fig.note``, ``fig.curve``, ``fig.image``; ``fig.raw`` hands back the real axes for a drawing no call covers). Each call
draws on FI's own axes, tags every artist it creates with ``gid = "fi:<n>"`` (a data artist without such a tag, found later,
is a drawing the calls did not make) and appends one entry to the call record. FI states facts about a figure only from that
record: ``record()`` returns it, ``write(path)`` writes ``<stem>.calls.json`` beside the figure's other records.

This file is carried by text into the quest's code folder, the way ``core/frame_table.py`` is, and the ``Figure`` class of the
frame's runtime mixes ``FigureCalls`` in. It therefore uses the standard library only and imports nothing of FI's; matplotlib
objects come from the caller's axes.

What the host (``Figure``) must hold, by these names:

* ``axes`` -- the list of matplotlib ``Axes``, one per panel (calls take the 0-based index, the record numbers panels from 1);
* ``table`` -- the read-only ``Table``: ``declared`` (the plan's result ids), ``keys()`` (the settings), and ``mean(id, key)``,
  ``n(id, key)``, ``interval(id, key)`` for ``note``;
* ``x`` -- the id of the setting or result the plan puts on this figure's x axis (a call may name another with
  ``x=(id, values)``);
* ``settings`` -- the names of the plan's settings (an ``x=`` may name one);
* ``series_ids`` -- the plan ids a series may name besides the declared results (for example the values the study compares);
* ``references`` -- ``{reference id: value}`` or ``{reference id: {"value": number, "label": text}}``;
* ``labels`` -- ``{id: label}`` or ``{id: {"label": text, "scale": "linear"|"log"}}``, the plan's words (set by FI; absent
  means the id stands in for the label and the scale is linear);
* ``fallback`` -- True when FI drew the figure itself (absent means False).

Every one but ``axes`` and ``table`` may be missing: the call then has less to check against and says so only where it must.
The mixin has no ``__init__``; it keeps its state in two private attributes it creates on first use."""

from __future__ import annotations

import contextlib
import copy
import json
import math
from pathlib import Path

MEANINGS = ("ci95", "sd", "range", "other")
NOTE_FORMS = ("mean", "n", "low", "high")


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_BAD = object()


def _clean(v, strict=True):
    """A value as the record keeps it: a finite number as a float; None or a non-finite number as None (a gap); text only where
    ``strict`` is False (an x of categories); anything else is ``_BAD``."""
    if v is None:
        return None
    if isinstance(v, str):
        return _BAD if strict else v
    if isinstance(v, bool):
        return _BAD
    try:  # numbers, numpy scalars
        f = float(v)
    except (TypeError, ValueError):
        return _BAD
    return f if math.isfinite(f) else None


def _vector(values, what, strict=True):
    """The values of a call as the record keeps them. Missing values (None, NaN, inf) stay None and are drawn as gaps; a value
    that is not a number is refused before anything is recorded or drawn (only x, ``strict=False``, may hold category names)."""
    try:
        raw = list(values)
    except TypeError:
        raise ValueError("`%s` must be a list of numbers, not %r" % (what, type(values).__name__)) from None
    out = [_clean(v, strict) for v in raw]
    bad = [raw[i] for i, v in enumerate(out) if v is _BAD]
    if bad:
        raise ValueError("`%s` must hold numbers (use None for a missing value, which is drawn as a gap); got %r" % (what, bad[0]))
    return out


def _gaps(values):
    """The values with each gap as NaN, for matplotlib."""
    return [float("nan") if v is None else v for v in values]


@contextlib.contextmanager
def _undone(ax, rec):
    """If a call fails part way, leave the record and the axes as they were: the record holds only what was drawn."""
    saved = copy.deepcopy(rec)
    before = {id(a) for a in ax.get_children()}
    try:
        yield
    except BaseException:
        rec.clear()
        rec.update(saved)
        for a in ax.get_children():
            if id(a) not in before and hasattr(a, "remove"):
                try:
                    a.remove()
                except Exception:  # noqa: BLE001 -- best effort; the record is already restored
                    pass
        raise


class FigureCalls:
    """The drawing calls of ``fig`` and the call record (see the module text for what the host holds)."""

    # ---- state and lookups -------------------------------------------------------------------------------------------

    def _state(self):
        if not hasattr(self, "_fc_panels"):
            self._fc_panels = {}
            self._fc_gid = 0
        return self._fc_panels

    def _next_gid(self):
        self._state()
        self._fc_gid += 1
        return "fi:%d" % self._fc_gid

    def _tag(self, *artists):
        """Give each artist the tag of one call (a list or container is walked)."""
        gid = self._next_gid()

        def walk(a):
            if a is None:
                return
            if isinstance(a, (list, tuple)):
                for b in a:
                    walk(b)
            elif hasattr(a, "set_gid"):
                a.set_gid(gid)
            else:  # an errorbar or bar container
                for b in list(getattr(a, "patches", [])) + list(getattr(a, "lines", []) or []) + list(getattr(a, "errorbar", None) or []):
                    walk(b)

        for a in artists:
            if hasattr(a, "lines") and hasattr(a, "has_xerr"):  # ErrorbarContainer: (data line, caplines, barlinecols)
                walk(a.lines)
            else:
                walk(a)
        return gid

    def _panel(self, panel):
        axes = list(getattr(self, "axes", []) or [])
        if not isinstance(panel, int) or isinstance(panel, bool) or not 0 <= panel < len(axes):
            raise IndexError("there is no panel %r: the figure has %d panel(s), numbered from 0" % (panel, len(axes)))
        st = self._state()
        if panel not in st:
            st[panel] = {"panel": panel + 1, "raw": False, "raw_reason": None, "x": None, "y": None, "series": [], "bands": [],
                         "refs": [], "notes": [], "texts_raw": []}
        return axes[panel], st[panel]

    def _label_of(self, ident):
        got = (getattr(self, "labels", None) or {}).get(ident)
        if isinstance(got, dict):
            return str(got.get("label") or ident)
        return str(got) if got else str(ident)

    def _scale_of(self, ident):
        got = (getattr(self, "labels", None) or {}).get(ident)
        return "log" if isinstance(got, dict) and got.get("scale") == "log" else "linear"

    def _results(self):
        return list(getattr(getattr(self, "table", None), "declared", []) or [])

    def _series_ok(self, series):
        known = list(getattr(self, "series_ids", []) or []) + [r for r in self._results() if r not in (getattr(self, "series_ids", []) or [])]
        if series not in known:
            raise ValueError("series `%s` is not an id of the plan; a series names a plan id (never legend text), and the plan's ids "
                             "are: %s" % (series, ", ".join(sorted(map(str, known))) or "(none)"))
        return series

    def _what_ok(self, what):
        results = self._results()
        derived = list(getattr(self, "derived_ids", []) or [])
        if what != "other" and what not in results and what not in derived:
            raise ValueError("`what=%r` is not a result of the plan; pass one of: %s, or \"other\" for a quantity the plan does not "
                             "declare" % (what, ", ".join(sorted(results + derived)) or "(none)"))
        return what

    def _set_x(self, rec, x_id):
        settings = list(getattr(self, "settings", []) or [])
        if x_id in self._results():
            kind = "result"
        elif x_id in settings:
            kind = "setting"
        else:
            raise ValueError("`x=%r` is neither a setting nor a result of the plan; the settings are: %s; the results are: %s"
                             % (x_id, ", ".join(settings) or "(none)", ", ".join(self._results()) or "(none)"))
        new = {"id": x_id, "kind": kind, "label": self._label_of(x_id), "scale": self._scale_of(x_id)}
        if rec["x"] is not None and rec["x"]["id"] != x_id:
            raise ValueError("panel %d already has `%s` on its x axis; one panel has one x axis (use another panel for `%s`)"
                             % (rec["panel"], rec["x"]["id"], x_id))
        rec["x"] = new

    def _set_y(self, rec, what):
        if rec["y"] is None:
            rec["y"] = {"what": what, "label": "computed value" if what == "other" else self._label_of(what),
                        "scale": "linear" if what == "other" else self._scale_of(what)}
        elif what not in rec["y"]["what"].split(" / "):  # two results on one axis: both are named
            rec["y"]["what"] += " / " + what
            rec["y"]["label"] += " / " + ("computed value" if what == "other" else self._label_of(what))

    # ---- series ------------------------------------------------------------------------------------------------------

    def _xsplit(self, x):
        """``(x id, values)`` of the ``x`` of a call: a plain list is on the x the plan draws (``fig.x``), ``(id, values)`` names a
        setting or a result of the plan (result A against result B, a time axis, a parametric curve)."""
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
            return x[0], x[1]
        ident = getattr(self, "x", None)
        if not isinstance(ident, str):
            raise ValueError("this figure's plan gives no x: pass x=(\"id of a setting or a result\", values) to say what x is")
        return ident, x

    def _add_series(self, panel, kind, series, x, y, what, yerr, yerr_meaning, step=False):
        """Check a series call and draw it; the record gets the series only if it was drawn."""
        ax, rec = self._panel(panel)
        if yerr_meaning not in MEANINGS:
            raise ValueError("`yerr_meaning=%r` is not one of: %s" % (yerr_meaning, ", ".join(MEANINGS)))
        x_id, x = self._xsplit(x)
        self._series_ok(series)
        self._what_ok(what)
        xs, ys = _vector(x, "x", strict=False), _vector(y, "y")
        if len(xs) != len(ys):
            raise ValueError("x has %d values and y has %d: a series needs one y for each x" % (len(xs), len(ys)))
        errs = None
        if yerr is not None:
            errs = [None if e is None else abs(e) for e in _vector(yerr, "yerr")]
            if len(errs) != len(ys):
                raise ValueError("yerr has %d values and y has %d: one bar half-width for each y" % (len(errs), len(ys)))
        legend = self._label_of(series)
        with _undone(ax, rec):
            self._set_x(rec, x_id)
            self._set_y(rec, what)
            arts = self._draw_series(ax, kind, xs, ys, legend, step)
            self._tag(arts)
            if errs is not None:
                self._tag(self._errorbar(ax, xs, ys, errs))
            rec["series"].append({"id": series, "label": legend, "kind": kind, "step": bool(step), "x": xs, "y": ys, "yerr": errs,
                                  "yerr_meaning": yerr_meaning if errs is not None else None, "merged_with": []})
        return arts

    @staticmethod
    def _draw_series(ax, kind, xs, ys, legend, step):
        if kind == "bars":
            return list(ax.bar(xs, _gaps(ys), label=legend).patches)
        if kind == "points":
            return ax.plot(xs, _gaps(ys), linestyle="none", marker="o", label=legend)
        return ax.plot(xs, _gaps(ys), label=legend, **({"drawstyle": "steps-post"} if step else {}))

    def line(self, panel, series, x, y, what, yerr=None, step=False, yerr_meaning="other"):
        """A line (a step curve with ``step=True``) of the plan's ``series``, ``y`` being the result ``what``. ``yerr`` are the
        half-widths of error bars and ``yerr_meaning`` says what they are: one of ci95, sd, range, other."""
        return self._add_series(panel, "line", series, x, y, what, yerr, yerr_meaning, step)

    def points(self, panel, series, x, y, what, yerr=None, yerr_meaning="other"):
        """Markers only."""
        return self._add_series(panel, "points", series, x, y, what, yerr, yerr_meaning)

    def bars(self, panel, series, x, y, what, yerr=None, yerr_meaning="other"):
        """Bars."""
        return self._add_series(panel, "bars", series, x, y, what, yerr, yerr_meaning)

    @staticmethod
    def _errorbar(ax, xs, ys, errs):
        nan = float("nan")
        return ax.errorbar(xs, [nan if v is None else v for v in ys], yerr=[nan if e is None else e for e in errs],
                           fmt="none", ecolor="0.3", capsize=2)

    def curve(self, panel, label, xs, ys, source="computed in the figure"):
        """An analytic overlay (a theory curve, a slope, a fitted line): marked with ``source``; claims about it stay few."""
        ax, rec = self._panel(panel)
        x, y = _vector(xs, "xs", strict=False), _vector(ys, "ys")
        if len(x) != len(y):
            raise ValueError("xs has %d values and ys has %d: a curve needs one y for each x" % (len(x), len(y)))
        if rec["x"] is None:
            raise ValueError("draw the data of panel %d first: a curve is drawn on the axes the data's x names" % (panel,))
        text = "%s (%s)" % (label, source)
        with _undone(ax, rec):
            arts = ax.plot(x, _gaps(y), linestyle="--", color="0.35", label=text)
            self._tag(arts)
            n = sum(1 for e in rec["series"] if e["kind"] == "curve") + 1
            rec["series"].append({"id": "curve%d" % n, "label": text, "kind": "curve", "step": False, "x": x, "y": y, "yerr": None,
                                  "yerr_meaning": None, "merged_with": [], "source": str(source)})
        return arts

    def image(self, panel, z, x=None, y=None, what="other"):
        """A heat map or raster of ``z`` (rows of values); ``x`` and ``y`` are the coordinates of its columns and rows."""
        ax, rec = self._panel(panel)
        self._what_ok(what)
        rows = [_vector(r, "z") for r in z]
        if not rows or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("`z` must be rows of equal length")
        cols = _vector(x, "x", strict=False) if x is not None else [float(i) for i in range(len(rows[0]))]
        rws = _vector(y, "y") if y is not None else [float(i) for i in range(len(rows))]
        if len(cols) != len(rows[0]) or len(rws) != len(rows):
            raise ValueError("`x` needs one value per column of `z` (%d) and `y` one per row (%d)" % (len(rows[0]), len(rows)))
        nums = [v for r in rows for v in r if _finite(v)]
        extent = None
        if all(_finite(v) for v in cols + rws) and cols and rws:
            extent = [min(cols), max(cols), max(rws), min(rws)]
        with _undone(ax, rec):
            if rec["x"] is None:
                rec["x"] = {"id": "column", "kind": "setting", "label": "column", "scale": "linear"}
            self._set_y(rec, what)
            art = ax.imshow([_gaps(r) for r in rows], aspect="auto", extent=extent,
                            vmin=min(nums) if nums else None, vmax=max(nums) if nums else None)
            self._tag(art)
            rec["series"].append({"id": what, "label": self._label_of(what), "kind": "image", "step": False, "x": cols, "y": rws,
                                  "yerr": None, "yerr_meaning": None, "merged_with": [], "z": rows})
        return art

    # ---- band, reference, note ---------------------------------------------------------------------------------------

    def band(self, panel, series, x, lo, hi, what, meaning):
        """A shaded interval around ``series``; ``meaning`` says what the interval is: one of ci95, sd, range, other."""
        if meaning not in MEANINGS:
            raise ValueError("`meaning=%r` is not one of: %s" % (meaning, ", ".join(MEANINGS)))
        ax, rec = self._panel(panel)
        self._series_ok(series)
        self._what_ok(what)
        x_id, x = self._xsplit(x)
        xs, los, his = _vector(x, "x", strict=False), _vector(lo, "lo"), _vector(hi, "hi")
        if not len(xs) == len(los) == len(his):
            raise ValueError("x, lo and hi must have the same length (%d, %d, %d)" % (len(xs), len(los), len(his)))
        drawn = [e for e in rec["series"] if e["id"] == series and e["kind"] != "curve"]
        if not drawn:
            raise ValueError("series `%s` has not been drawn on panel %d: draw it with line, points or bars before its band; the "
                             "series on this panel are: %s" % (series, panel, ", ".join(sorted({e["id"] for e in rec["series"]})) or "(none)"))
        if not any(e["x"] == xs for e in drawn):
            raise ValueError("the band's x does not match the x of series `%s` on panel %d: a band is drawn at the points of its "
                             "series" % (series, panel))
        with _undone(ax, rec):
            self._set_x(rec, x_id)
            self._set_y(rec, what)
            art = ax.fill_between(xs, _gaps(los), _gaps(his), alpha=0.25, linewidth=0)
            self._tag(art)
            rec["bands"].append({"series": series, "x": xs, "lo": los, "hi": his, "meaning": meaning})
        return art

    def ref(self, panel, ref_id, axis="y"):
        """A reference line at a value FI holds (a plan reference), labelled with the plan's words."""
        ax, rec = self._panel(panel)
        refs = getattr(self, "references", None) or {}
        if ref_id not in refs:
            raise ValueError("`%s` is not a reference the plan holds; the references are: %s" % (ref_id, ", ".join(sorted(map(str, refs))) or "(none)"))
        if axis not in ("x", "y"):
            raise ValueError("`axis=%r` must be \"x\" or \"y\"" % (axis,))
        got = refs[ref_id]
        value = got.get("value") if isinstance(got, dict) else got
        if not _finite(value):
            raise ValueError("the reference `%s` has no number to draw" % ref_id)
        label = str(got.get("label")) if isinstance(got, dict) and got.get("label") else self._label_of(ref_id)
        draw = ax.axhline if axis == "y" else ax.axvline
        art = draw(float(value), linestyle=":", color="0.2", label=label)
        self._tag(art)
        rec["refs"].append({"id": ref_id, "axis": axis, "value": float(value), "label": label})
        return art

    def note(self, panel, x, y, what, form="mean", key=None):
        """Text in the figure: ``what`` is digit-free words with one ``{result id}`` where FI writes the number it reads from the
        table (``"mean growth: {growth}"``); ``form`` is mean, n, low or high; ``key`` is the setting (optional when the study
        has one)."""
        ax, rec = self._panel(panel)
        if not (_finite(x) and _finite(y)):
            raise ValueError("a note's x and y must be finite numbers, got %r and %r" % (x, y))
        if form not in NOTE_FORMS:
            raise ValueError("`form=%r` is not one of: %s" % (form, ", ".join(NOTE_FORMS)))
        text = str(what)
        if text.count("{") != 1 or text.count("}") != 1 or text.index("{") > text.index("}"):
            raise ValueError("a note holds its words and exactly one `{result id}` where the number goes, for example "
                             "'mean growth: {growth}'; got %r" % text)
        ident = text[text.index("{") + 1:text.index("}")]
        words = text[:text.index("{")] + text[text.index("}") + 1:]
        if any(ch.isdigit() for ch in words):
            raise ValueError("a note's words cannot hold digits (%r): FI writes the number from the table; write the words and "
                             "`{%s}` where the number goes" % (words.strip(), ident))
        table = getattr(self, "table", None)
        if ident not in self._results():
            raise ValueError("`%s` is not a result of the plan; the results are: %s" % (ident, ", ".join(sorted(self._results())) or "(none)"))
        keys = list(table.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError("say which setting the note reads with `key=`; the settings are: %s" % (", ".join(keys) or "(none)"))
            key = keys[0]
        if form == "mean":
            value = table.mean(ident, key)
        elif form == "n":
            value = table.n(ident, key)
        else:
            value = table.interval(ident, key)[0 if form == "low" else 1]
        if value is None or not _finite(value):
            raise ValueError("the table has no %s of `%s` at setting %s to write" % (form, ident, key))
        number = "%d" % value if form == "n" else "%.4g" % value
        shown = text[:text.index("{")] + number + text[text.index("}") + 1:]
        art = ax.text(float(x), float(y), shown)
        self._tag(art)
        rec["notes"].append({"id": ident, "text": shown, "x": float(x), "y": float(y)})
        return art

    # ---- raw, record -------------------------------------------------------------------------------------------------

    def raw(self, panel):
        """The real ``Axes`` of panel; FI then states nothing about what is drawn on it."""
        ax, rec = self._panel(panel)
        rec["raw"], rec["raw_reason"] = True, "fig.raw"
        return ax

    def record(self):
        """The call record: what every call drew (see the plan's schema, version 1)."""
        self._state()
        panels = []
        for i in range(len(getattr(self, "axes", []) or [])):
            _ax, rec = self._panel(i)
            panels.append(json.loads(json.dumps(rec)))
        return {"version": 1, "fallback": bool(getattr(self, "fallback", False)), "panels": panels}

    def write(self, path):
        """Write the record as ``<stem>.calls.json`` (``path`` is the figure's file or a ``.calls.json`` path); returns the path."""
        p = Path(path)
        out = p if p.name.endswith(".calls.json") else p.with_name(p.stem + ".calls.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.record(), indent=1, allow_nan=False), encoding="utf-8")
        return out
