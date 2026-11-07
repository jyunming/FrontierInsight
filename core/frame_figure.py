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
* ``labels`` -- ``{id: label}`` or ``{id: {"label": text}}``, the plan's words (set by FI; absent means the id stands in
  for the label);
* ``fallback`` -- True when FI drew the figure itself (absent means False).

Every one but ``axes`` and ``table`` may be missing: the call then has less to check against and says so only where it must.
The mixin has no ``__init__``; it keeps its state in two private attributes it creates on first use."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import numbers
import os
from pathlib import Path

MEANINGS = ("ci95", "sd", "range", "other")
NOTE_FORMS = ("mean", "n", "low", "high")
LOG_NOTE = " (log scale)"  # said on the label of a log axis
LOG_RATIO = 1e3  # FI's rule for an axis the plan does not scale: log when all values are above zero and span this ratio
IDENTICAL_RTOL = 1e-12  # two series equal within this relative tolerance are one line to the reader
Y_RTOL = 1e-3  # a drawn y matches the table's value within this relative tolerance (plus 1e-12 absolute)
Y_FORMS = ("mean", "sd", "se", "min", "max", "range", "max_abs")  # which statistic of the trials at a setting y may be (``form``)
BAR_RTOL = 1e-3  # error-bar widths the body typed match the table's own widths within this relative tolerance (plus 1e-12 absolute)
RECORD_VERSION = 2  # 2: a curve's name and y are FI's, series carry ``x_checked`` / ``neutral``, "range" bars are checked
NEUTRAL_X = "computed x"  # the x axis of a panel that draws values FI did not find among the setting's values
#: Set by FI for the last run, when no repair of a figure is left: a line FI cannot name is then left out of the figure and
#: recorded in ``removed``, where before it is refused (sent to the figure's repair).
LAST_RUN_ENV = "FI_FIGURE_FALLBACK"
NEUTRAL_Y = "computed value"  # the neutral name of a quantity FI did not tie to a result of the table
LEGEND_LEVELS = 12  # a figure with one curve per level of a setting names every level in its legend up to this many; more: the lowest and the highest
LEGEND_SHARE = 0.2  # a legend whose box lies this much over a panel it does not belong to is over it (the share the layout check of plot_style uses)
LEGEND_INSET_PX = 3  # a legend's box is measured this many pixels in from its frame, so a line that only grazes the frame is not under it
REF_STATED = "stated in the figure"  # the source of a reference line drawn at a literal value


def _typed(v):
    """A number as it was typed (0.05, 3, 1e-06): the words FI uses for a line it names by its value."""
    text = repr(float(v))
    return text[:-2] if text.endswith(".0") else text


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


_BAD = object()


def _same(a, b):
    """Two value lists equal within ``IDENTICAL_RTOL`` (relative); a gap equals only a gap; category names must be equal."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    for u, v in zip(a, b):
        if u is None or v is None or isinstance(u, str) or isinstance(v, str):
            if u != v:
                return False
        elif abs(u - v) > IDENTICAL_RTOL * max(abs(u), abs(v)):
            return False
    return True


def hold_value(value):
    """A held setting's value as text: the shortest form that gives the same value back (``repr`` for a float), so two values
    of the plan never print the same."""
    return repr(value) if isinstance(value, float) else str(value)


def hold_text(panel):
    """The setting a panel was declared to show, as ``label = value`` (the plan's label when the record carries it, else the
    setting's name); empty when it has none. The title, the claims and the caption all say it with this."""
    hold = panel.get("hold")
    if not isinstance(hold, dict):
        return ""
    labels = panel.get("hold_label") if isinstance(panel.get("hold_label"), dict) else {}
    return ", ".join("%s = %s" % (labels.get(k) or k, hold_value(v)) for k, v in hold.items())


def _short(description):
    """A short form of a plan description: up to its first stop, colon, semicolon or bracket, at most 48 characters."""
    text = str(description or "").strip()
    for stop in (".", ";", ":", "(", "\n"):
        text = text.split(stop)[0]
    text = text.strip(" ,")
    if len(text) > 48:
        text = text[:48].rsplit(" ", 1)[0].rstrip(" ,")
    return text[:1].upper() + text[1:]


def _padded(lo, hi, scale):
    """Axis limits around ``lo..hi``: 5% of the span on each side (in decades on a log axis); a single value gets a fixed margin."""
    if scale == "log":
        a, b = math.log10(lo), math.log10(hi)
        pad = 0.05 * (b - a) if b > a else 0.08
        return 10 ** (a - pad), 10 ** (b + pad)
    pad = 0.05 * (hi - lo) if hi > lo else (0.1 * abs(lo) or 0.5)
    return lo - pad, hi + pad


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


def _is_pair(values):
    """True for ``(lo_widths, hi_widths)``: exactly two members that are themselves sequences (a plain two-point list is not)."""
    try:
        seq = list(values)
    except TypeError:
        return False
    return len(seq) == 2 and all(hasattr(m, "__len__") and not isinstance(m, (str, bytes)) for m in seq)


def _sides(entry):
    """``(lower widths, upper widths)`` of a series' error bars from its record, ``(None, None)`` without any (an older record
    holds only the symmetric ``yerr``)."""
    if entry.get("yerr_lo") is not None or entry.get("yerr_hi") is not None:
        return entry.get("yerr_lo"), entry.get("yerr_hi")
    if entry.get("yerr") is not None:
        return entry["yerr"], entry["yerr"]
    return None, None


def _same_number(a, b):
    """Two numbers equal within ``Y_RTOL`` of the larger one, with no absolute allowance: a small number is not zero and a number
    of another size is not the same (zero equals only zero)."""
    return abs(a - b) <= Y_RTOL * max(abs(a), abs(b))


def _close_y(a, b):
    return abs(a - b) <= Y_RTOL * max(abs(a), abs(b)) + 1e-12


def _close_width(a, b):
    """Two bar widths equal within ``BAR_RTOL``; a gap equals only a gap."""
    if a is None or b is None:
        return a is b
    return abs(a - b) <= BAR_RTOL * max(abs(a), abs(b)) + 1e-12


def _sample_sd(values):
    """Sample standard deviation (n - 1) of the finite values; None for fewer than two."""
    v = [float(x) for x in values if _finite(x)]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


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
            self._fc_extra = {}  # panel -> {"series": [(first artist, what)], "refs": [artist]}, parallel to the record's lists
            self._fc_owned = {}  # id(artist) -> (artist, the artists of the same call): everything these calls drew, by identity (never by gid)
        return self._fc_panels

    def _remember(self, panel, part, artist, what=None):
        """Keep the artist of a record entry (and the result it shows) beside the record, for the legend FI builds in ``finish``."""
        extra = self._fc_extra.setdefault(panel, {"series": [], "refs": []})
        extra[part].append((artist, what) if part == "series" else artist)

    def _next_gid(self):
        self._state()
        self._fc_gid += 1
        return "fi:%d" % self._fc_gid

    def _tag(self, *artists):
        """Give each artist the tag of one call (a list or container is walked)."""
        gid = self._next_gid()
        group = []

        def walk(a):
            if a is None:
                return
            if isinstance(a, (list, tuple)):
                for b in a:
                    walk(b)
            elif hasattr(a, "set_gid"):
                a.set_gid(gid)
                group.append(a)
            else:  # an errorbar or bar container
                for b in list(getattr(a, "patches", [])) + list(getattr(a, "lines", []) or []) + list(getattr(a, "errorbar", None) or []):
                    walk(b)

        for a in artists:
            if hasattr(a, "lines") and hasattr(a, "has_xerr"):  # ErrorbarContainer: (data line, caplines, barlinecols)
                walk(a.lines)
            else:
                walk(a)
        for a in group:
            self._fc_owned[id(a)] = (a, group)
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

    def _hold_title(self, rec):
        """The title of a panel that holds a setting: the plan's label and the value, kept in the record so a claim and the
        caption name the panel the same."""
        rec["hold_label"] = {k: self._label_of(k) for k in rec["hold"]}
        return hold_text(rec)

    def _label_of(self, ident):
        got = (getattr(self, "labels", None) or {}).get(ident)
        if isinstance(got, dict):
            return str(got.get("label") or ident)
        return str(got) if got else str(ident)

    def _reference_label(self, ident):
        """The name of a plan reference: the plan's own label; a reference that is the plan's formula for one of the study's
        results is named as that result's expected value, so its line and the result's curve never share a name."""
        got = (getattr(self, "references", None) or {}).get(ident)
        own = str(got["label"]) if isinstance(got, dict) and got.get("label") else None
        if ident in self._results() and own in (None, self._label_of(ident)):
            return "Expected %s (plan formula)" % self._label_of(ident)
        return own or self._label_of(ident)

    def _results(self):
        return list(getattr(getattr(self, "table", None), "declared", []) or [])

    def _series_setting(self):
        """The setting the figure's plan draws one curve per level of (the frame's ``series``), else None. FI's own default figure
        is held to it like a body: it passes the setting's value as each curve's series."""
        ident = getattr(self, "series", None)
        if not isinstance(ident, str) or ident not in (getattr(self, "settings", None) or []):
            return None
        return ident

    def _series_levels(self, setting):
        """The levels of ``setting`` in FI's table, as text, in the table's order."""
        table = getattr(self, "table", None)
        out = []
        for key in (list(table.keys()) if table is not None else []):
            got = self._key_settings(key).get(setting)
            if got is not None and got not in out:
                out.append(got)
        return out

    def _level_series(self, series, what, setting):
        """``(id, info)`` of the curve a call names by the level of the figure's series setting: the level is the call's ``series``
        (a value of the setting in FI's table). The id is FI's (``<result>@<setting>:<level>``: no comma or equals sign, so a claim can name it); a result's id or any other
        text is no level and says what to pass instead."""
        label = self._label_of(setting)
        levels = self._series_levels(setting)
        if isinstance(series, numbers.Real) and not isinstance(series, bool):  # any real number, numpy's too, as a plain number
            series = int(series) if isinstance(series, numbers.Integral) else float(series)
        numeric = isinstance(series, (int, float)) and not isinstance(series, bool)
        if (not numeric and not isinstance(series, str)) or (levels and not self._keys_at(setting, series)):
            shown = ", ".join(levels[:8]) + (", ..." if len(levels) > 8 else "")
            if isinstance(series, str):
                raise ValueError("`%s` %s. This figure has one curve per %s: pass each curve's %s as its series, "
                                 "for example fig.line(0, <%s>, x, y, what)%s" % (
                                     series, "names a result, not a curve" if series in self._results() else "is not a value of " + label,
                                     label, label, setting, "; its values are: " + shown if shown else ""))
            raise ValueError("`%r` is not a value of %s; its values are: %s" % (series, label, shown or "(none)"))
        text = ("%.6g" % series) if isinstance(series, float) else str(series)
        info = {"setting": setting, "setting_label": label, "value": series, "text": text, "result": what,
                "result_label": NEUTRAL_Y if what == "other" else self._label_of(what)}
        return "%s@%s:%s" % (what, setting, text), info

    def _series_ok(self, series):
        ids = list(getattr(self, "series_ids", []) or [])
        known = ids + [r for r in self._results() + list(getattr(self, "derived_ids", []) or []) if r not in ids]
        if not known:
            raise ValueError("FI did not supply the study's series and results to this figure, so no series can be drawn; this is a "
                             "fault of FI's files, not of the figure function (series `%s`)" % (series,))
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

    def _set_x(self, rec, x_id, checked=True):
        settings = list(getattr(self, "settings", []) or [])
        if x_id in self._results():
            kind = "result"
        elif x_id in settings:
            kind = "setting"
        else:
            raise ValueError("`x=%r` is neither a setting nor a result of the plan; the settings are: %s; the results are: %s"
                             % (x_id, ", ".join(settings) or "(none)", ", ".join(self._results()) or "(none)"))
        new = {"id": x_id, "kind": kind, "label": self._label_of(x_id), "scale": "linear"}
        if rec["x"] is not None and rec["x"]["id"] != x_id:
            raise ValueError("panel %d already has `%s` on its x axis; one panel has one x axis (use another panel for `%s`)"
                             % (rec["panel"], rec["x"]["id"], x_id))
        if rec["x"] is not None and rec["x"].get("neutral"):
            return  # one series whose x FI could not tie to the setting makes the axis neutral for the whole panel
        if not checked:
            new.update(label=NEUTRAL_X, neutral=True)
        rec["x"] = new

    def _set_y(self, rec, what, named=None):
        """The y axis of a panel. ``named``: the derived quantity the series was found (by FI's record of the derived values) to be
        the number of, whose name the axis has while ``what`` stays ``other`` (no value of the table is claimed for it)."""
        name = named or what
        label = NEUTRAL_Y if name == "other" else self._label_of(name)
        if rec["y"] is None:
            rec["y"] = {"what": what, "label": label, "scale": "linear", **({"names": name} if named else {})}
        else:
            if name not in rec["y"].get("names", rec["y"]["what"]).split(" / "):  # two results on one axis: both are named
                rec["y"]["label"] += " / " + label
                rec["y"]["names"] = rec["y"].get("names", rec["y"]["what"]) + " / " + name
            if what not in rec["y"]["what"].split(" / "):
                rec["y"]["what"] += " / " + what
            if "names" in rec["y"] and rec["y"]["names"] == rec["y"]["what"]:
                del rec["y"]["names"]

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

    def _keys_at(self, x_id, x):
        """Every key of the table whose setting holds ``x_id`` at the value ``x`` (a number compared as a number)."""
        table = getattr(self, "table", None)
        return [key for key in (list(table.keys()) if table is not None else []) if self._key_holds(key, x_id, x)]

    @staticmethod
    def _key_holds(key, x_id, x):
        """Whether the setting key (``a=1,b=2``) holds setting ``x_id`` at the value ``x`` (a number compared as a number)."""
        for piece in str(key).split(","):
            name, _, text = piece.partition("=")
            if name.strip() != x_id:
                continue
            try:
                # Relative only: a fixed absolute tolerance would join distinct values on a small scale (1e-12 vs 3e-12).
                same = math.isclose(float(text), float(x), rel_tol=1e-9, abs_tol=0.0)
            except (TypeError, ValueError):
                same = text.strip() == str(x)
            if same:
                return True
        return False

    def _setting_key(self, x_id, x, what=None, y=None):
        """The table's key of the setting where ``x_id`` is ``x``. When other settings also vary (several keys share that x), the
        key is the one whose mean of ``what`` is the plotted ``y`` (the bars belong to the point drawn); None when no key, or
        still more than one, fits."""
        found = self._setting_keys(x_id, x, what, y)
        return found[0] if len(found) == 1 else None

    def _setting_keys(self, x_id, x, what=None, y=None):
        """Every key of the table at ``x`` of ``x_id`` whose mean of ``what`` is the plotted ``y`` (all keys at x when ``y`` is not
        given or only one key is there)."""
        found = self._keys_at(x_id, x)
        table = getattr(self, "table", None)
        if len(found) > 1 and what is not None and _finite(y):
            fits = []
            for key in found:
                try:
                    m = table.mean(what, key)
                except Exception:  # noqa: BLE001 -- a key whose mean cannot be read does not fit
                    continue
                if _finite(m) and _close_width(float(m), float(y)):
                    fits.append(key)
            found = fits
        return found

    def _table_widths(self, meaning, what, x_id, xs, ys, edges=False):
        """``(lower, upper)`` lists of the widths FI reads from the table for ``meaning`` (ci95 or sd) of result ``what`` at each
        x (a setting's value); an entry is None where the table gives none. ``None`` when the table cannot say at all. With ``edges``
        the drawn y must also be the table's mean at the key (a check of drawn bars or a band: their edges are the table's bounds
        only around the table's own mean), else the key gives no width."""
        table = getattr(self, "table", None)
        if table is None or what not in self._results() or x_id not in (getattr(self, "settings", None) or []):
            return None
        lo, hi = [], []
        for x, y in zip(xs, ys):
            a = b = None
            reads = []  # the widths of each key that fits; several keys that fit are one answer only when they agree
            for key in ([] if x is None else self._setting_keys(x_id, x, what, y)):
                c = d = None
                try:
                    if edges and _finite(y) and not _close_width(float(table.mean(what, key)), float(y)):
                        reads.append((None, None))
                        continue
                    if meaning == "ci95":
                        m, (u, v) = table.mean(what, key), table.interval(what, key)
                        if _finite(m) and _finite(u) and _finite(v):
                            c, d = m - u, v - m
                    else:
                        sd = table.sd(what, key) if hasattr(table, "sd") else _sample_sd(table.values(what, key))
                        c = d = float(sd) if _finite(sd) else None
                except Exception:  # noqa: BLE001 -- a read the table refuses is a width it does not give
                    c = d = None
                reads.append((c, d))
            if reads and all(r[0] is not None and r[1] is not None and _close_width(r[0], reads[0][0])
                             and _close_width(r[1], reads[0][1]) for r in reads):
                a, b = reads[0]
            lo.append(a)
            hi.append(b)
        return (lo, hi) if any(v is not None for v in lo + hi) else None

    def _table_stats(self, what, key, form):
        """The numbers the table gives for ``form`` of result ``what`` at ``key``: one, or one for each value a sub-study compared
        inside the setting (the level-aware read). Empty when the table gives none."""
        table = self.table
        if form == "mean":
            try:
                m = table.mean(what, key)
                return [float(m)] if _finite(m) else []
            except Exception:  # noqa: BLE001 -- a sub-study records a result for each compared value, not for the setting
                try:
                    return [float(m) for m in table.level_means(what, key).values() if _finite(m)]
                except Exception:  # noqa: BLE001 -- a read the table refuses gives nothing to compare with
                    return []
        try:
            groups = [list(table.values(what, key))]
        except Exception:  # noqa: BLE001 -- as above: the values of each compared value
            try:
                groups = [list(v) for _k, _t, v in table.level_rows(what, key)]
            except Exception:  # noqa: BLE001
                return []
        out = []
        for vals in groups:
            vals = [float(v) for v in vals if _finite(v)]
            if not vals:
                continue
            if form in ("sd", "se"):
                sd = _sample_sd(vals)
                if _finite(sd):
                    out.append(float(sd) / (len(vals) ** 0.5 if form == "se" else 1.0))
            elif form == "min":
                out.append(min(vals))
            elif form == "max":
                out.append(max(vals))
            elif form == "range":
                out.append(max(vals) - min(vals))
            elif form == "max_abs":
                out.append(max(abs(v) for v in vals))
        return out

    def _y_matches(self, form, what, x_id, xs, ys):
        """Whether every drawn y (a gap is skipped) equals, within ``Y_RTOL``, the table's ``form`` of result ``what`` at a setting
        whose ``x_id`` is the drawn x (when other settings also vary, at any of them: a y that is a real value of another slice
        is not told apart). A point with no x, no such setting or no table value cannot be checked, which counts as not matching;
        so does a series with no finite y at all (nothing was checked) and a figure with no table."""
        return self._y_comparison(form, what, x_id, xs, ys) == "matches"

    def _y_comparison(self, form, what, x_id, xs, ys):
        """``matches`` (the rule of ``_y_matches``), ``differs`` (FI read the table's value at a drawn point and it is not the drawn
        y) or ``not compared`` (FI could not read a value to compare at some point, or there was nothing to compare)."""
        if getattr(self, "table", None) is None:
            return "not compared"
        checked = 0
        for x, y in zip(xs, ys):
            if y is None or not _finite(y):
                continue
            if x is None:
                return "not compared"
            stats = [t for key in self._keys_at(x_id, x) for t in self._table_stats(what, key, form)]
            if not stats:
                return "not compared"
            if not any(_close_y(float(y), t) for t in stats):
                return "differs"
            checked += 1
        return "matches" if checked > 0 else "not compared"

    def _derived_comparison(self, what, form, x_id, xs, ys, level=None, hold=None):
        """``matches``, ``differs`` or ``not compared`` for a series drawn as a quantity the plan derives (``what`` is no result of the
        table): FI holds the number it worked out for each setting (``derived_values``, from its own record of the derived values).
        Only the keys the curve can show are read: those at its level of the setting and at the panel's held values. The curve
        matches when ONE combination of the other settings gives, at every drawn point, a number equal to the drawn y (a relative
        tolerance of the two numbers; zero equals only zero). One point that no key at its x gives the drawn y for is ``differs``
        whatever the other points are; the points that match only at different combinations are ``differs`` too; a point with no
        key at all (and no point that differs) is ``not compared``. No record, another statistic, an x that is not a setting: ``not compared``."""
        held = (getattr(self, "derived_values", None) or {}).get(what)
        if not isinstance(held, dict) or form != "mean" or x_id not in (getattr(self, "settings", None) or []):
            return "not compared"
        fixed = [(str(k), v) for k, v in (hold or {}).items()]
        if level:
            fixed.append((level["setting"], level["value"]))
        keys = {}
        for key in held:
            pieces = dict((n.strip(), t.strip()) for n, _, t in (piece.partition("=") for piece in str(key).split(",")))
            if all(n not in pieces or self._key_holds(key, n, v) for n, v in fixed):
                keys[key] = ",".join("%s=%s" % kv for kv in sorted(pieces.items()) if kv[0] != x_id)  # the other settings of the key
        combos, missing, checked = None, False, 0
        for x, y in zip(xs, ys):
            if y is None or not _finite(y):
                continue
            if x is None:
                missing = True
                continue
            here = [(rest, held[key]) for key, rest in keys.items() if self._key_holds(key, x_id, x)]
            if not here:
                missing = True
                continue
            fits = {rest for rest, v in here if _same_number(float(y), v)}
            if not fits:
                return "differs"
            combos = fits if combos is None else combos & fits
            checked += 1
        if missing or not checked:
            return "not compared"
        return "matches" if combos else "differs"

    def _names_a_result(self, series):
        """Whether a series id, or its plan label, is a result of the plan's (id or label): such a name says what the y is."""
        results = self._results() + list(getattr(self, "derived_ids", []) or [])
        return series in results or self._label_of(series) in {self._label_of(r) for r in results}

    def _x_checked(self, x_id, xs, ys):
        """Whether every drawn x is a value of the setting ``x_id`` in FI's table (a number compared as a number): then the axis
        may carry the setting's name and the claims that read x may be offered. An x that is a result, a transform of the setting
        (1/N), a point with no x, a figure without a table, or no x at all cannot be checked, which counts as not (fail closed)."""
        if getattr(self, "table", None) is None or x_id not in (getattr(self, "settings", None) or []):
            return False
        pts = [x for x, y in zip(xs, ys) if y is not None and _finite(y)]  # a gap has no point, so its x says nothing
        return bool(pts) and all(x is not None and self._keys_at(x_id, x) for x in pts)

    def _checked_what(self, what, form, x_id, xs, ys):
        """``what`` when the drawn y is the table's own value, else ``other``: the series keeps its shape but FI states no value
        from it. A declared ``other`` stays. Everything FI cannot compare is ``other`` as well (fail closed): a quantity that is
        not a result of the plan (a derived one: FI holds no value of it here), an x that is a result and not a setting, a figure
        without a table, a point the table cannot say, a series with no finite y. ``form`` (default the mean) says which statistic
        of the trials y is. No case refuses the call."""
        return self._check_of_what(what, form, x_id, xs, ys)[0]

    def _check_of_what(self, what, form, x_id, xs, ys):
        """``(checked what, how)``: how is ``matches``, ``differs`` (FI compared and the values are not the table's) or
        ``not compared`` (FI had nothing to compare the series with), from what the check did."""
        if what == "other" or what not in self._results() or x_id not in (getattr(self, "settings", None) or []):
            return "other", "not compared"
        if form not in Y_FORMS:
            return "other", "not compared"
        how = self._y_comparison(form, what, x_id, xs, ys)
        return (what if how == "matches" else "other"), how

    def _checked_meaning(self, meaning, what, x_id, xs, ys, lo, hi):
        """What error bars of typed widths may be called. ``ci95``/``sd`` stay only when the widths equal the table's own for
        that meaning at EVERY drawn point (``BAR_RTOL``): one point the table gives no width for, or one that differs, an unreadable
        table, all give ``other`` (the bars are drawn, no claim about their meaning is offered). ``range`` stays only when the
        drawn edges are the table's minimum and maximum of the trials (``_range_matches``). The table's sd is its ``sd`` read, else the sample sd (n - 1) of its ``values``."""
        if meaning == "range":
            return "range" if self._range_matches(what, x_id, xs, ys, lo, hi) else "other"
        if meaning != "ci95" and meaning != "sd":
            return meaning  # ``other`` states nothing
        got = self._table_widths(meaning, what, x_id, xs, ys, edges=True)
        if got is None:
            return "other"
        drawn = False
        for i, v in enumerate(ys):
            if v is None:  # a gap has no bar
                continue
            drawn = True
            for mine, theirs in ((lo, got[0]), (hi, got[1])):
                if theirs[i] is None or not _close_width(mine[i], theirs[i]):  # one unverifiable or unmatched bar: no claim
                    return "other"
        return meaning if drawn else "other"

    def _range_matches(self, what, x_id, xs, ys, lo, hi):
        """Whether the drawn lower and upper edges (the drawn y less ``lo`` and plus ``hi``) are the table's minimum and maximum of
        the trials at EVERY drawn x (``BAR_RTOL`` on the widths below and above y, as the other bar checks). A point with no key at
        its x, no minimum and maximum, or an edge that is a gap, the keys of one x disagreeing, a result that is not the table's
        or an x that is not a setting: not the range (no claim)."""
        if getattr(self, "table", None) is None or what not in self._results() or x_id not in (getattr(self, "settings", None) or []):
            return False
        drawn = False
        for i, (x, y) in enumerate(zip(xs, ys)):
            if y is None:  # a gap has no bar
                continue
            drawn = True
            if x is None or lo[i] is None or hi[i] is None or not _finite(y):
                return False
            keys = self._setting_keys(x_id, x, what, y)
            if not keys:
                return False
            for key in keys:
                lows, highs = self._table_stats(what, key, "min"), self._table_stats(what, key, "max")
                if not any(_close_width(float(y) - a, lo[i]) and _close_width(b - float(y), hi[i]) for a, b in zip(lows, highs)):
                    return False
        return drawn

    @staticmethod
    def _key_settings(key):
        """``{setting: text}`` of a table key such as ``a=1,b=2``."""
        out = {}
        for piece in str(key).split(","):
            k, _, text = piece.partition("=")
            out[k.strip()] = text.strip()
        return out

    @staticmethod
    def _evenly_placed(values):
        """Whether the coordinates of a heat map's columns or rows are where ``imshow`` puts them: categories (text) by position,
        numbers ascending and evenly spaced."""
        if all(isinstance(v, str) for v in values):
            return True
        if any(not _finite(v) for v in values):
            return False
        steps = [b - a for a, b in zip(values, values[1:])]
        return all(d > 0 and abs(d - steps[0]) <= 1e-6 * max(abs(d), abs(steps[0])) for d in steps)

    def _image_matches(self, what, x_id, y_id, cols, rows, z):
        """Whether every drawn colour value is the table's mean of result ``what`` at the one setting its column (``x_id``) and row
        (``y_id``) name: for each cell exactly one key at that column and row has that mean (within ``Y_RTOL``), and the keys
        of all cells agree on every other setting (one slice of the table). A cell with no such key, or with more than one,
        cannot be compared and counts as not matching; so does an image with no finite value at all."""
        if getattr(self, "table", None) is None:
            return False
        checked = 0
        slice_of = None
        for r, line in zip(rows, z):
            for c, v in zip(cols, line):
                if v is None or not _finite(v):
                    continue
                if c is None or r is None:
                    return False
                keys = [k for k in set(self._keys_at(x_id, c)) & set(self._keys_at(y_id, r))
                        if any(_close_y(float(v), t) for t in self._table_stats(what, k, "mean"))]
                if len(keys) != 1:
                    return False
                rest = {k: t for k, t in self._key_settings(keys[0]).items() if k not in (x_id, y_id)}
                if slice_of is None:
                    slice_of = rest
                elif rest != slice_of:
                    return False
                checked += 1
        return checked > 0

    def _checked_image_what(self, what, x_id, y_id, cols, rows, z):
        """``what`` when every cell of the image is the table's own value, else ``other`` (the image keeps its shape and states
        no value). Whatever FI cannot tie to the table is ``other`` as well: a quantity that is not a plan result, a column or a
        row that is not a setting of the plan (two different settings), coordinates that are not ascending and evenly spaced."""
        settings = getattr(self, "settings", None) or []
        if what == "other" or what not in self._results() or x_id not in settings or y_id not in settings or x_id == y_id:
            return "other"
        if not (self._evenly_placed(cols) and self._evenly_placed(rows)):
            return "other"
        return what if self._image_matches(what, x_id, y_id, cols, rows, z) else "other"

    def _add_series(self, panel, kind, series, x, y, what, yerr, yerr_meaning, step=False, form="mean"):
        """Check a series call and draw it; the record gets the series only if it was drawn."""
        ax, rec = self._panel(panel)
        if yerr_meaning not in MEANINGS:
            raise ValueError("`yerr_meaning=%r` is not one of: %s" % (yerr_meaning, ", ".join(MEANINGS)))
        x_id, x = self._xsplit(x)
        setting = self._series_setting()
        level = None
        if setting:
            series, level = self._level_series(series, what, setting)
        else:
            self._series_ok(series)
        self._what_ok(what)
        self._once(rec, panel, kind, series)
        xs, ys = _vector(x, "x", strict=False), _vector(y, "y")
        if len(xs) != len(ys):
            raise ValueError("x has %d values and y has %d: a series needs one y for each x" % (len(xs), len(ys)))
        lo = hi = None
        if isinstance(yerr, str):
            if yerr not in ("ci95", "sd"):
                raise ValueError("`yerr=%r`: pass bar widths, a (lower, upper) pair of widths, or \"ci95\" or \"sd\" for FI to read "
                                 "them from the table" % (yerr,))
            if yerr_meaning not in ("other", yerr):
                raise ValueError("`yerr=%r` already says what the bars are; drop `yerr_meaning=%r`" % (yerr, yerr_meaning))
            if what not in self._results() or x_id not in (getattr(self, "settings", None) or []):
                raise ValueError("`yerr=%r` reads the table: `what` must be a result of the plan and x a setting; give the widths "
                                 "yourself otherwise" % (yerr,))
            got = self._table_widths(yerr, what, x_id, xs, ys)
            if got is None:
                raise ValueError("the table gives no %s of `%s` at these x values" % (yerr, what))
            (lo, hi), yerr_meaning = got, yerr
        elif yerr is not None:
            pair = _is_pair(yerr)
            lo = [None if e is None else abs(e) for e in _vector(yerr[0] if pair else yerr, "yerr")]
            hi = [None if e is None else abs(e) for e in _vector(yerr[1], "yerr")] if pair else list(lo)
            if len(lo) != len(ys) or len(hi) != len(ys):
                raise ValueError("yerr has %d values and y has %d: one bar half-width for each y (a pair has one list of lower and "
                                 "one of upper widths)" % (len(lo), len(ys)))
            yerr_meaning = self._checked_meaning(yerr_meaning, what, x_id, xs, ys, lo, hi)
        declared = what
        what, how = self._check_of_what(what, form, x_id, xs, ys)
        # A quantity the plan derives is not in the table; FI holds the number it worked out for it at each setting, and a series
        # that is those numbers is that quantity (named so), though no value of the table is claimed for it (``what`` stays ``other``).
        tied = None
        if what == "other" and declared != "other" and declared not in self._results():
            how = self._derived_comparison(declared, form, x_id, xs, ys, level, rec.get("hold"))
            if how == "matches":
                tied = {"id": declared, "label": self._label_of(declared), "checked": how, "from": "record of the derived values"}
        x_ok = self._x_checked(x_id, xs, ys)
        # A series the body gave as a result of the plan that FI did not find to be the table's value of it keeps neither that
        # result's name (nor the series' own plan label, which names the same thing): its name is the neutral one, numbered below.
        neutral = what == "other" and not tied and (declared != "other" or (level is None and self._names_a_result(series)))
        # The caption and the axis come from one fact, the quantity FI's check found: a level that carried the declared result's
        # label no longer does, and the declared id stays beside it in the record.
        declared_as = None
        if what == "other" and declared != "other" and not tied:
            declared_as = {"id": declared, "label": self._label_of(declared), "checked": how}
            if level:
                level.update(result_label=NEUTRAL_Y, declared_as=declared, declared_label=declared_as["label"], checked=how)  # "result" stays the curve's identity
        legend = "%s = %s" % (level["setting_label"], level["text"]) if level else NEUTRAL_Y if neutral else self._label_of(series)
        with _undone(ax, rec):
            self._set_x(rec, x_id, x_ok)
            self._set_y(rec, what, declared if tied else None)
            arts = self._draw_series(ax, kind, xs, ys, legend, step)
            self._tag(arts)
            if lo is not None:
                bars = self._errorbar(ax, xs, ys, lo, hi)
                self._tag(bars)
                # kept to share the curve's colour and, for bars, to move with the bar, by the position the series takes in the record
                self._fc_extra.setdefault(panel, {"series": [], "refs": []}).setdefault("bars", {})[len(rec["series"])] = bars
            rec["series"].append({"id": series, "label": legend, "kind": kind, "step": bool(step), "x": xs, "y": ys,
                                  "yerr": lo if lo is not None and _same(lo, hi) else None, "yerr_lo": lo, "yerr_hi": hi,
                                  "yerr_meaning": yerr_meaning if lo is not None else None, "what": what, "merged_with": [],
                                  "x_checked": x_ok, **({"neutral": True} if neutral else {}), **({"declared_as": declared_as} if declared_as else {}),
                                  **({"named_as": tied} if tied else {}), **({"level": level} if level else {})})
            loose = [e for e in rec["series"] if e.get("neutral") and not e.get("level")]
            for k, e in enumerate(loose, 1):
                e["label"] = NEUTRAL_Y if len(loose) == 1 else "%s %d" % (NEUTRAL_Y, k)
            levelled = [e for e in rec["series"] if e.get("level")]
            # Curves FI found not to be the result they were declared as are all "a computed value"; when the panel has curves
            # declared as different results, each such result's curves are numbered, so no two groups share a name or a count.
            unchecked = list(dict.fromkeys(e["level"]["declared_as"] for e in levelled if e["level"].get("declared_as")))
            # a number is skipped when a label of the figure (a plan label, a series' name) is already that name
            taken = {str(v.get("label") if isinstance(v, dict) else v) for v in (getattr(self, "labels", None) or {}).values()}
            taken |= {str(r.get("label")) for st in self._state().values() for r in st["series"] if not r.get("declared_as")}
            names, n = {}, 0
            for ident in unchecked:  # in the order the panel first draws them
                n += 1
                while "%s %d" % (NEUTRAL_Y, n) in taken:
                    n += 1
                names[ident] = "%s %d" % (NEUTRAL_Y, n)
            for e in levelled:
                if e["level"].get("declared_as"):
                    e["level"]["result_label"] = names[e["level"]["declared_as"]] if len(unchecked) > 1 else NEUTRAL_Y
            several = len({(e["level"]["result"], e["level"]["result_label"]) for e in levelled}) > 1  # two results on one panel: each curve says which
            for e in levelled:
                lv = e["level"]
                e["label"] = ("%s, " % lv["result_label"] if several else "") + "%s = %s" % (lv["setting_label"], lv["text"])
            self._remember(panel, "series", arts[0] if arts else None, what)
            self._fc_extra[panel].setdefault("drawn", {})[len(rec["series"]) - 1] = list(arts or [])  # every artist of the call, for its colour
            self._group_bars(panel)  # bars of the panel: a series of another kind changes how they are placed
        return arts

    def _once(self, rec, panel, kind, series):
        """A panel draws a series once with a call of a kind: a second line (or points, or bars) of the same id would show the
        reader two curves under one name."""
        if any(e["id"] == series and e["kind"] == kind for e in rec["series"]):
            raise ValueError("the figure draws `%s` a second time on panel %d with the same name, so the legend would show two series "
                             "under one name; give each curve its own series (a figure with one curve per level of a setting "
                             "passes the level as the series)" % (series, panel))

    def _group_bars(self, panel):
        """The bar series of a panel side by side. Bars over a setting are categories: each distinct x value is one slot, the slots
        are evenly spaced in the order of the values and named by them on the axis, and the series of a slot take 0.8 of it between
        them (0.8 over the number of bar series each), so no bar hides another and a sparse x value is no thinner than a crowded one.
        A panel that also draws something else along x (a line, a band, a reference at an x) keeps its x axis: the bars are grouped
        around their x values, a slot being 0.8 of the smallest gap between two distinct values. An error bar and a text placed at a
        bar move with it. Run after every bars call (a later series changes the slots of the earlier ones); the record's data is not
        touched, only where the bars sit."""
        ax, rec = self._panel(panel)
        extra = self._fc_extra[panel]
        drawn = extra.get("drawn") or {}
        mine = [k for k, e in enumerate(rec["series"]) if e["kind"] == "bars" and drawn.get(k)]
        if not mine or ax.get_xscale() != "linear":
            return
        if getattr(ax, "_fi_bar_slots", None):  # the slots of an earlier call: the axis is made again below
            from matplotlib import ticker
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        values = {k: [ax.xaxis.convert_units(v) for v in rec["series"][k]["x"]] for k in mine}
        if any(len(values[k]) != len(drawn[k]) for k in mine):
            return
        numeric = all(_finite(v) for k in mine for v in values[k])
        alone = all(e["kind"] == "bars" for e in rec["series"]) and not rec["bands"] and not any(r["axis"] == "x" for r in rec["refs"])
        slots = sorted({round(v, 12) for k in mine for v in values[k]}) if numeric and alone else None
        if slots is not None:
            width = 0.8 / len(mine)
            where = {k: [slots.index(round(v, 12)) for v in values[k]] for k in mine}
        else:
            base = {k: values[k] for k in mine}
            distinct = sorted({round(v, 12) for vs in base.values() for v in vs if _finite(v)})
            gap = min((b - a for a, b in zip(distinct, distinct[1:])), default=1.0)
            width = 0.8 * gap / len(mine)
            where = base
        moved = []  # (old centre, new centre) of every bar laid out before
        laid = extra.setdefault("bar_laid", set())
        positions = {}
        for j, k in enumerate(mine):
            shift = (j - (len(mine) - 1) / 2) * width
            centres = []
            for p, c in zip(drawn[k], where[k]):
                old = p.get_x() + p.get_width() / 2
                new = c + shift
                p.set_x(new - width / 2)
                p.set_width(width)
                if k in laid:  # a bar of an earlier call: texts at it follow; a bar just drawn has only the default place
                    moved.append((old, new))
                centres.append(new)
            positions[k] = centres
            container = (extra.get("bars") or {}).get(k)
            if container is not None and len(container.lines) > 2:
                for cap in container.lines[1] or ():
                    if len(cap.get_xdata()) == len(centres):
                        cap.set_xdata(centres)
                for lc in container.lines[2] or ():
                    segs = lc.get_segments()
                    if len(segs) == len(centres):
                        lc.set_segments([[(c, y) for _x, y in seg] if len(seg) else [(c, math.nan), (c, math.nan)] for c, seg in zip(centres, segs)])  # a missing value draws nothing
        for t in ax.texts:  # a text a body put at a bar moves with it (matched by the centre the bar had)
            x0, y0 = t.get_position()
            near = [new for old, new in moved if _finite(x0) and abs(old - x0) <= 1e-9]
            if len({round(n, 12) for n in near}) == 1 and abs(near[0] - x0) > 1e-12:
                t.set_position((near[0], y0))
        laid.update(mine)
        ax._fi_bar_x = positions
        if slots is not None:
            ax._fi_bar_slots = len(slots)
            from matplotlib import ticker

            class FiSlotFormatter(ticker.Formatter):  # the name the audit of the figure step knows as FI's own (not typed text)
                def __call__(self, x, pos=None):
                    return ("%.6g" % slots[int(round(x))]) if abs(x - round(x)) < 1e-9 and 0 <= round(x) < len(slots) else ""

            ax.xaxis.set_major_locator(ticker.FixedLocator(list(range(len(slots)))))
            ax.xaxis.set_major_formatter(FiSlotFormatter())
        else:
            ax._fi_bar_slots = None
        ax.relim()
        ax.autoscale_view()

    @staticmethod
    def _draw_series(ax, kind, xs, ys, legend, step):
        if kind == "bars":
            return list(ax.bar(xs, _gaps(ys), label=legend).patches)
        if kind == "points":
            return ax.plot(xs, _gaps(ys), linestyle="none", marker="o", label=legend)
        return ax.plot(xs, _gaps(ys), label=legend, **({"drawstyle": "steps-post"} if step else {}))

    def line(self, panel, series, x, y, what, yerr=None, step=False, yerr_meaning="other", form="mean"):
        """A line (a step curve with ``step=True``) of the plan's ``series``, ``y`` being the result ``what``. ``yerr`` are the
        half-widths of error bars: one list (both sides), a pair ``(lower, upper)`` of lists, or ``"ci95"`` / ``"sd"`` for FI to
        read the widths from the table at each x (x a setting). ``yerr_meaning`` says what typed widths are (ci95, sd, range,
        other); FI keeps ci95 or sd only if the widths equal the table's own (relative 1e-3), else the meaning is ``other``."""
        return self._add_series(panel, "line", series, x, y, what, yerr, yerr_meaning, step, form)

    def points(self, panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean"):
        """Markers only."""
        return self._add_series(panel, "points", series, x, y, what, yerr, yerr_meaning, form=form)

    def bars(self, panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean"):
        """Bars."""
        return self._add_series(panel, "bars", series, x, y, what, yerr, yerr_meaning, form=form)

    @staticmethod
    def _errorbar(ax, xs, ys, lo, hi):
        nan = float("nan")
        return ax.errorbar(xs, [nan if v is None else v for v in ys],
                           yerr=[[nan if e is None else e for e in lo], [nan if e is None else e for e in hi]],
                           fmt="none", ecolor="0.3", capsize=2)

    def curve(self, panel, label, xs, ys, source="computed in the figure"):
        """Refused: a curve whose numbers the figure computes itself has no identity in FI's records, so FI could not name it in
        the legend. A theory curve is a reference of the plan (``ref``), which FI evaluates and names itself."""
        return self._no_identity(
            panel, "a curve the figure computed itself", "FI has no record of what it is",
            "FI does not draw a curve the figure computes itself: it could not say what the curve is. A theory or reference "
            "curve is a reference of the plan, drawn with `fig.ref(panel, \"<reference id>\")`; a quantity of the study is "
            "computed in the analysis and drawn from the table")

    def _no_identity(self, panel, what, why, refusal):
        """A line FI cannot name. While a repair of the figure is left it is refused (``ValueError``: the body's repair is told
        what to do); in the last run it is left out of the figure, never drawn under a made-up name, and recorded in
        ``removed`` (FI's QUALITY.md names it). Returns nothing drawn."""
        if os.environ.get(LAST_RUN_ENV) != "1":
            raise ValueError(refusal)
        self._state()
        entry = {"panel": int(panel) + 1, "what": what, "why": why}
        if entry not in self.__dict__.setdefault("_fc_removed", []):
            self._fc_removed.append(entry)
        return []

    def _reference_curve(self, panel, ax, rec, ref_id):
        """A plan reference that changes along the panel's x, drawn as a curve through FI's own values of it at the settings the
        panel shows (the table's references: FI's evaluation of the plan's formula), named by the plan. Refused when the panel
        has no x of a setting yet, or when the reference also changes with a setting the panel does not hold."""
        if rec["x"] is None or rec["x"].get("kind") != "setting":
            raise ValueError("draw the data of panel %d first, along a setting: a reference curve is drawn at the settings the "
                             "panel's x names" % (panel,))
        x_id = rec["x"]["id"]
        per_setting = getattr(getattr(self, "table", None), "_references", {}).get(ref_id) or {}
        by_x, other = {}, set()
        for key in self._held_keys(rec):
            if key not in per_setting or not _finite(per_setting[key]):
                continue
            pieces = dict(p.partition("=")[::2] for p in str(key).split(","))
            try:
                xv = float(pieces[x_id])
            except (KeyError, TypeError, ValueError):
                continue
            y = float(per_setting[key])
            if xv in by_x and not math.isclose(by_x[xv], y, rel_tol=1e-9, abs_tol=0.0):
                other |= {n for n in pieces if n != x_id}
            by_x.setdefault(xv, y)
        if other:
            raise ValueError("the reference `%s` also changes with %s, which this panel does not hold: declare the panel's "
                             "setting with `fig.panel` first" % (ref_id, ", ".join(sorted(other))))
        if len(by_x) < 2:
            raise ValueError("the reference `%s` has fewer than two values along `%s` in this panel" % (ref_id, x_id))
        x = sorted(by_x)
        y = [by_x[v] for v in x]
        label = self._reference_label(ref_id)
        with _undone(ax, rec):
            arts = ax.plot(x, y, linestyle="--", color="0.35", label=label)
            self._tag(arts)
            rec["series"].append({"id": ref_id, "label": label, "kind": "curve", "step": False, "x": x, "y": y, "yerr": None,
                                  "yerr_lo": None, "yerr_hi": None, "yerr_meaning": None, "merged_with": [], "source": "plan",
                                  "what": "other"})
            self._remember(panel, "series", arts[0] if arts else None, "other")
            self._group_bars(panel)  # a curve along x keeps the bars of the panel at their x values
        return arts

    def image(self, panel, z, x=None, y=None, what="other"):
        """A heat map or raster of ``z`` (rows of values); ``x`` and ``y`` are the coordinates of its columns and rows."""
        ax, rec = self._panel(panel)
        self._what_ok(what)
        x_id = y_id = None
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
            x_id, x = x
        if isinstance(y, tuple) and len(y) == 2 and isinstance(y[0], str):
            y_id, y = y
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
        what = self._checked_image_what(what, x_id, y_id, cols, rws, rows) if x is not None and y is not None else "other"
        with _undone(ax, rec):
            if rec["x"] is None:
                rec["x"] = {"id": "column", "kind": "setting", "label": "column", "scale": "linear"}
            self._set_y(rec, what)
            art = ax.imshow([_gaps(r) for r in rows], aspect="auto", extent=extent,
                            vmin=min(nums) if nums else None, vmax=max(nums) if nums else None)
            self._tag(art)
            label = NEUTRAL_Y if what == "other" else self._label_of(what)  # the neutral name an unchecked quantity's axis has
            rec["series"].append({"id": what, "label": label, "kind": "image", "step": False, "x": cols, "y": rws,
                                  "yerr": None, "yerr_lo": None, "yerr_hi": None, "yerr_meaning": None, "merged_with": [], "z": rows, "what": what})
            self._remember(panel, "series", None, what)
        return art

    # ---- band, reference, note ---------------------------------------------------------------------------------------

    def _checked_band_meaning(self, meaning, what, x_id, xs, los, his, drawn):
        """What a band may be called. ``ci95``/``sd`` stay only when the band's edges are the table's own interval / sd around the
        drawn series at EVERY x (the error-bar check, ``BAR_RTOL``): the widths below and above the drawn y are compared with the
        table's. A band with no drawn series at its x, or an edge that is a gap where the series has a y, gives ``other``."""
        series = [e for e in drawn if e["x"] == xs]
        if not series:
            return "other"
        ys = series[0]["y"]
        lo, hi = [], []
        for y, a, b in zip(ys, los, his):
            if y is None:
                lo.append(None)
                hi.append(None)
            elif a is None or b is None:
                return "other"
            else:
                lo.append(y - a)
                hi.append(b - y)
        return self._checked_meaning(meaning, what, x_id, xs, ys, lo, hi)

    def band(self, panel, series, x, lo, hi, what, meaning):
        """A shaded interval around ``series``; ``meaning`` says what the interval is: one of ci95, sd, range, other."""
        if meaning not in MEANINGS:
            raise ValueError("`meaning=%r` is not one of: %s" % (meaning, ", ".join(MEANINGS)))
        ax, rec = self._panel(panel)
        setting = self._series_setting()
        if setting:
            series, _level = self._level_series(series, what, setting)
        else:
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
        if meaning in ("ci95", "sd", "range"):
            meaning = self._checked_band_meaning(meaning, what, x_id, xs, los, his, drawn)
        with _undone(ax, rec):
            self._set_x(rec, x_id, any(e["x"] == xs and e.get("x_checked") for e in drawn))
            self._set_y(rec, what)
            stepped = any(e["step"] for e in drawn if e["x"] == xs)  # the edges of a band follow the steps of its line
            art = ax.fill_between(xs, _gaps(los), _gaps(his), alpha=0.25, linewidth=0, step="post" if stepped else None)
            self._tag(art)
            rec["bands"].append({"series": series, "x": xs, "lo": los, "hi": his, "meaning": meaning})
            self._fc_extra.setdefault(panel, {"series": [], "refs": []}).setdefault("bands", []).append((series, art))
        self._group_bars(panel)
        return art

    def _next_line_id(self, rec):
        return "refline%d" % (sum(1 for r in rec["refs"] if r.get("source") != "plan") + 1)

    def _table_read(self, ident, form, key):
        """``(value, key)``: the ``form`` (mean, n, low, high) of result ``ident`` at setting ``key`` from the table (``key``
        optional when the study has one setting); raises a plain ``ValueError`` where the table has none."""
        table = getattr(self, "table", None)
        known = self._results() + list(getattr(self, "derived_ids", []) or [])
        if ident not in known:
            raise ValueError("`%s` is not a result of the plan; the results are: %s" % (ident, ", ".join(sorted(known)) or "(none)"))
        if form not in NOTE_FORMS:
            raise ValueError("`form=%r` is not one of: %s" % (form, ", ".join(NOTE_FORMS)))
        keys = list(table.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError("say which setting is read with `key=`; the settings are: %s" % (", ".join(keys) or "(none)"))
            key = keys[0]
        if form == "mean":
            value = table.mean(ident, key)
        elif form == "n":
            value = table.n(ident, key)
        else:
            value = table.interval(ident, key)[0 if form == "low" else 1]
        if value is None or not _finite(value):
            raise ValueError("the table has no %s of `%s` at setting %s to write" % (form, ident, key))
        return value, key

    def _setting_read(self, name, key):
        """The value setting ``name`` has at setting key ``key`` (optional when the study has one), written as text: from the
        table's ``setting(key)`` when the host has it, else from the key's own ``name=value`` pieces."""
        table = getattr(self, "table", None)
        keys = list(table.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError("say which setting key the note reads with `key=`; the keys are: %s" % (", ".join(keys) or "(none)"))
            key = keys[0]
        elif key not in keys:  # a typed key would put a typed number in the figure
            raise ValueError("`key=%r` is not a setting key of the study; the keys are: %s" % (key, ", ".join(keys) or "(none)"))
        got = None
        if hasattr(table, "setting"):
            try:
                got = (table.setting(key) or {}).get(name)
            except Exception:  # noqa: BLE001 -- fall back to the key's own text
                got = None
        if got is None:
            for piece in str(key).split(","):
                n, _, text = piece.partition("=")
                if n.strip() == name and text.strip():
                    got = text.strip()
        if got is None:
            raise ValueError("setting key %s holds no value of `%s`" % (key, name))
        return ("%.4g" % got) if _finite(got) else str(got)

    def _held_keys(self, rec):
        """The table's setting keys this panel shows: every key when the panel holds no setting, else the keys whose
        ``name=value`` pieces agree with the panel's ``hold`` (a number compared as a number)."""
        table = getattr(self, "table", None)
        keys = list(table.keys()) if table is not None else []
        hold = rec.get("hold") if isinstance(rec.get("hold"), dict) else {}
        out = []
        for key in keys:
            pieces = dict(p.partition("=")[::2] for p in str(key).split(","))
            ok = True
            for name, want in hold.items():
                got = pieces.get(str(name))
                if got is None:
                    continue
                try:
                    ok = math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=0.0)
                except (TypeError, ValueError):
                    ok = str(got).strip() == str(want)
                if not ok:
                    break
            if ok:
                out.append(key)
        return out

    def _plan_values(self, rec):
        """``{reference id: [numbers]}``: each plan reference's numbers over the settings this panel shows. A reference that
        holds one number everywhere (``references``) gives that number; one that differs by setting gives its numbers at the
        panel's settings, read from the table's references (FI's own evaluation of the plan's formula)."""
        out = {}
        for ident, got in (getattr(self, "references", None) or {}).items():
            number = got.get("value") if isinstance(got, dict) else got
            if _finite(number):
                out[str(ident)] = [float(number)]
        held = getattr(getattr(self, "table", None), "_references", None)
        if isinstance(held, dict):
            keys = self._held_keys(rec)
            for ident, per_setting in held.items():
                if str(ident) in out or not isinstance(per_setting, dict):
                    continue
                numbers = [float(per_setting[k]) for k in keys if k in per_setting and _finite(per_setting[k])]
                if numbers:
                    out[str(ident)] = numbers
        return out

    def _plan_match(self, value, rec):
        """The plan references that hold one number over this panel's settings, equal to ``value`` within the digits the
        body typed (half a unit of its last digit): the only way a typed constant gets a name."""
        text = repr(float(value))
        mantissa, _, exponent = text.partition("e")
        digits = len(mantissa.partition(".")[2])
        tol = 0.5 * 10.0 ** (int(exponent or 0) - digits)
        # a threshold of the plan is one more reference here: two with the value leave the line without a name
        return [ident for ident, numbers in self._plan_values(rec).items()
                if max(numbers) - min(numbers) <= tol and abs(numbers[0] - float(value)) <= tol]

    def _plan_value(self, ident, rec):
        return self._plan_values(rec)[ident][0]

    def ref(self, panel, ref_id=None, axis="y", *, at=None, value=None, key=None, form="mean"):
        """A reference line, one of three: a plan reference ``ref_id`` (labelled with the plan's words); ``at=<result id>`` (FI
        reads the line's value from the table: ``form`` mean, n, low or high at setting ``key``); or ``value=<number>``, a
        constant drawn as stated (recorded as "stated in the figure", labelled "Reference line" by FI, and claimed only as a
        line at that coordinate)."""
        ax, rec = self._panel(panel)
        if axis not in ("x", "y"):
            raise ValueError("`axis=%r` must be \"x\" or \"y\"" % (axis,))
        if sum(v is not None for v in (ref_id, at, value)) != 1:
            raise ValueError("a reference line is one of: a reference id of the plan, `at=<result id>` (FI reads the value from "
                             "the table), or `value=<number>` for a constant")
        if ref_id is not None and ref_id not in (getattr(self, "references", None) or {}) and ref_id in self._plan_values(rec):
            numbers = self._plan_values(rec)[ref_id]
            if max(numbers) - min(numbers) > 0:
                if axis == "x":
                    raise ValueError("`%s` changes along the panel's x, so FI draws it as a curve over x; it has no single "
                                     "x to draw a line at: draw it with axis=\"y\"" % ref_id)
                return self._reference_curve(panel, ax, rec, ref_id)
        if ref_id is not None:
            refs = getattr(self, "references", None) or {}
            if ref_id not in refs and ref_id in self._plan_values(rec):
                refs = {ref_id: {"value": self._plan_value(ref_id, rec), "label": self._reference_label(ref_id)}}
            if ref_id not in refs:
                raise ValueError("`%s` is not a reference the plan holds; the references are: %s" % (ref_id, ", ".join(sorted(map(str, refs))) or "(none)"))
            got = refs[ref_id]
            number = got.get("value") if isinstance(got, dict) else got
            if not _finite(number):
                raise ValueError("the reference `%s` has no number to draw" % ref_id)
            label = self._reference_label(ref_id)
            ident, source = ref_id, "plan"
        elif at is not None:
            number, used = self._table_read(at, form, key)
            words = {"mean": "Mean", "n": "Number of runs", "low": "Lower 95% bound", "high": "Upper 95% bound"}[form]
            label = "%s of %s at %s" % (words, self._label_of(at), used)
            ident, source = self._next_line_id(rec), "table"
        else:
            if not _finite(value):
                raise ValueError("`value=%r` must be a finite number" % (value,))
            if axis == "x" and any(isinstance(v, str) for e in rec["series"] for v in e["x"]):
                raise ValueError("panel %d has categories on its x axis, which have no coordinate to draw a line at; use axis=\"y\"" % panel)
            hits = self._plan_match(value, rec)
            if len(hits) != 1:
                return self._no_identity(
                    panel, "a line at %s" % _typed(value),
                    "its value matches several references of the plan" if hits
                    else "FI has no record of where its value comes from",
                    "the constant %r %s, so FI cannot say what the line is. If it is a theory value, add it to the plan as "
                    "a named reference; if it is a result's value, draw it with `at=<result id>`"
                    % (value, "matches several references of the plan (%s)" % ", ".join(sorted(hits)) if hits
                       else "matches no reference of the plan at this panel's setting"))
            ident = hits[0]
            number, source = self._plan_value(ident, rec), "plan"
            label = self._reference_label(ident)
        if source == "plan" and any(r["id"] == ident and r["axis"] == axis for r in rec["refs"]):
            # the same reference of the plan drawn again (by its id and as a typed number): one line, one legend entry
            return None
        draw = ax.axhline if axis == "y" else ax.axvline
        art = draw(float(number), linestyle=":", color="0.2", label=label)
        self._tag(art)
        rec["refs"].append({"id": ident, "axis": axis, "value": float(number), "label": label, "source": source})
        self._remember(panel, "refs", art)
        self._group_bars(panel)
        return art

    def note(self, panel, x=None, y=None, what=None, form="mean", key=None):
        """Text in the figure: ``what`` is digit-free words with one ``{result id}`` where FI writes the number it reads from the
        table (``"mean growth: {growth}"``); ``form`` is mean, n, low or high; ``key`` is the setting (optional when the study
        has one). ``x`` and ``y`` (data coordinates) are optional: left out, FI puts the note in the corner of the axes the data
        and the legend leave free. The id may also be a derived quantity the host lists."""
        ax, rec = self._panel(panel)
        if what is None and isinstance(x, str) and y is None:  # note(panel, "words {id}")
            x, what = None, x
        if what is None:
            raise ValueError("a note needs its words: `what=\"mean growth: {growth}\"`")
        placed = x is not None or y is not None
        if placed and not (_finite(x) and _finite(y)):
            raise ValueError("a note's x and y must be finite numbers (or both left out), got %r and %r" % (x, y))
        text = str(what)
        if text.count("{") != 1 or text.count("}") != 1 or text.index("{") > text.index("}"):
            raise ValueError("a note holds its words and exactly one `{result id}` where the number goes, for example "
                             "'mean growth: {growth}'; got %r" % text)
        ident = text[text.index("{") + 1:text.index("}")]
        words = text[:text.index("{")] + text[text.index("}") + 1:]
        if any(ch.isdigit() for ch in words):
            raise ValueError("a note's words cannot hold digits (%r): FI writes the number from the table; write the words and "
                             "`{%s}` where the number goes" % (words.strip(), ident))
        settings = list(getattr(self, "settings", None) or [])
        if ident in settings and ident not in self._results():
            number = self._setting_read(ident, key)
        else:
            if ident not in self._results() + list(getattr(self, "derived_ids", []) or []):
                raise ValueError("`{%s}` is neither a setting nor a result of the plan; the settings are: %s; the results are: %s"
                                 % (ident, ", ".join(settings) or "(none)", ", ".join(sorted(self._results())) or "(none)"))
            value, _used = self._table_read(ident, form, key)
            number = "%d" % value if form == "n" else "%.4g" % value
        shown = text[:text.index("{")] + number + text[text.index("}") + 1:]
        art = ax.text(float(x), float(y), shown) if placed else ax.text(0.5, 0.5, shown, transform=ax.transAxes)
        self._tag(art)
        self._state()
        self._fc_extra.setdefault(panel, {"series": [], "refs": []}).setdefault("notes", []).append(art)
        entry = {"id": ident, "text": shown, "x": float(x) if placed else None, "y": float(y) if placed else None}
        if ident in settings and ident not in self._results():
            entry.update(source="setting", setting=ident)
        if not placed:
            entry["corner"] = None  # FI chooses it in ``finish``
        rec["notes"].append(entry)
        return art

    # ---- labels, legend and limits: FI's, never the body's --------------------------------------------------------------

    def finish(self, spec_labels, result_info, axis_scales=None):
        """Write what the figure's body must not: every axis label, the legend and the limits, from the plan and the record.

        Called once, after the body returned and before the figure is saved. A title, label, legend, limit or axis scale the body
        set is replaced, and the replacement is logged once for the whole figure (never raw, never an error). A raw panel
        (``fig.raw``) is left as the body drew it: there is no record to write from.

        ``spec_labels`` is ``{id: {"label": text, "unit": text}}`` or ``{id: text}`` (a text is a label
        that already holds its unit), for the ids of the plan (settings, results, series); every key is optional.
        ``result_info`` is ``{result id: {"description": text, "unit": text}}`` (the plan's description of each result and its
        unit); every key is optional. ``axis_scales`` is ``{"x": "linear"|"log", "y": ...}``, the scale the plan declares for the
        figure's axes (``Figure.x_scale`` / ``y_scale`` of the contract); an axis the plan is silent on is scaled by FI's rule
        (log when every value drawn along it is above zero and the largest is at least 1000 times the smallest). The record of an
        axis says which one set the scale (``scale_source``: "plan" or "FI rule"); a declared log axis with a value at or below
        zero is drawn linear and ``scale_note`` says why.

        An axis label is, in this order: the plan's label of the id, with its unit ("Growth (1/h)"); else a short form of the
        result's plan description (up to its first stop or bracket, at most 48 characters); else "value" with the unit. The id
        itself is never a label. Two results on one axis are joined with " / "; ``what="other"`` is "computed value". The x label
        comes from the id of the call's ``x=``. In an image panel the vertical axis is "row".

        Series whose x, y, error bars, kind and result are equal within ``IDENTICAL_RTOL`` (relative) are one line to the reader:
        the later ids go to the first one's ``merged_with`` (both entries stay in the record, since both were drawn) and the
        legend has one entry reading "a = b".

        The legend is built from the record, never from the artists' own labels, one entry per drawn series (and per reference
        line). A figure with entries on more than one panel has ONE legend, outside the axes: the entries of every panel, a series
        once (by its id, and the level of a setting for a curve of one), below the panels in as many columns as fit the width of
        the figure, or to their right when that would be more than four rows and the legend is narrow. A figure with entries on
        one panel has its legend inside the axes, in the corner (upper right, upper left, lower right, lower left, in this order of
        preference) that no drawn point, error bar, band or reference line reaches, a corner being 40% of the axes wide and as high
        as its rows need; when every corner is taken or there are more than six entries, that legend goes outside the axes like
        the figure's. The constrained layout makes the room (the figure is given it), so no panel is moved by hand. A series is
        one colour in every panel, along the house style's colour cycle in the order the series first appear in. A y label longer
        than its panel and an x label or a note wider than its panel are broken between words onto as many lines as they need (no text is cut). The record's ``legend``
        says where the legend is.

        Limits are set from the series, error bar ends, bands, reference lines and curves together (bars also reach their
        baseline, 0, on a linear axis). A linear axis is padded by 5% of the span on both sides, a log axis by the same share of
        the span in decades, and a single value gets a fixed margin; on a log axis only values above zero count, so a band that
        reaches 0 still gives finite limits (a log axis with no positive value is drawn linear, and the record says so)."""
        spec_labels, result_info = spec_labels or {}, result_info or {}
        axis_scales = axis_scales if isinstance(axis_scales, dict) else {}
        self._state()
        changed = []
        live = []
        for i in range(len(getattr(self, "axes", []) or [])):
            ax, rec = self._panel(i)
            if rec["raw"]:
                continue
            live.append(i)
            self._merge_identical(i, rec)
        scales = self._figure_scales(live, axis_scales)  # one scale for each axis for the whole figure
        for i in live:
            ax, rec = self._panel(i)
            self._write_limits(ax, rec, changed, scales)
            self._write_labels(ax, rec, spec_labels, result_info, changed)
        fig = self._figure_of_panels()
        if fig is not None and live:
            self._use_constrained_layout(fig)
        self._share_colours(live)
        keyed = {}
        for i in live:
            ax, rec = self._panel(i)
            old = ax.get_legend()
            if old is not None:
                changed.append("legend")
                old.remove()
            self._level_colours(i, rec)
            keyed[i] = self._entries_keyed(i, rec, spec_labels, result_info)
        for old in list(getattr(fig, "legends", []) or []) if fig is not None else []:
            if getattr(old, "_fi_legend", False):
                old.remove()
        self._fc_legend = None
        with_entries = [i for i in live if keyed[i]]
        if len(with_entries) > 1 and fig is not None:  # one legend for the figure: every panel's entries, each series once
            self._figure_legend(fig, with_entries, keyed)
        else:
            outside = []
            for i in live:
                ax, rec = self._panel(i)
                placed = self._write_legend(ax, i, rec, keyed[i])
                if placed is not None:
                    outside.append(placed)
            for ax, entries in outside:
                self._legend_outside(ax, entries)
        self._keep_legend_off_data(fig, live, keyed)
        for i in live:
            ax, rec = self._panel(i)
            self._place_notes(ax, i, rec)
        self._wrap_to_panels(live)
        sup = getattr(getattr(self, "fig", None), "_suptitle", None)
        if sup is not None and sup.get_text() and not any(self._panel(i)[1]["raw"] for i in range(len(self.axes))):  # a custom drawing keeps its own
            sup.set_text("")
            changed.append("title")
        if changed and not getattr(self, "_fc_logged", False):
            self._fc_logged = True
            logging.getLogger("frame_figure").warning(
                "the figure's drawing code set its own %s; FI wrote them from the plan instead", ", ".join(dict.fromkeys(changed)))

    def _merge_identical(self, panel, rec):
        store = self._fc_extra.setdefault(panel, {"series": [], "refs": []})
        store["twins"] = set()
        extra = store["series"]

        def what_of(k):
            return extra[k][1] if k < len(extra) else None

        for e in rec["series"]:
            e["merged_with"] = []
        firsts = []
        for i, e in enumerate(rec["series"]):
            if e["kind"] in ("curve", "image"):
                continue
            for j in firsts:
                f = rec["series"][j]
                if (f["kind"], f["step"], what_of(j)) == (e["kind"], e["step"], what_of(i)) \
                        and _same(f["x"], e["x"]) and _same(f["y"], e["y"]) \
                        and _same(_sides(f)[0], _sides(e)[0]) and _same(_sides(f)[1], _sides(e)[1]):
                    f["merged_with"].append(e["id"])
                    store["twins"].add(i)
                    break
            else:
                firsts.append(i)

    @staticmethod
    def _name_of(ident, spec_labels, result_info):
        sl, ri = spec_labels.get(ident), result_info.get(ident) or {}
        sl = sl if isinstance(sl, dict) else ({"label": sl} if sl else {})
        unit = sl.get("unit") or ri.get("unit")
        text = str(sl.get("label") or _short(ri.get("description")) or "value")
        return "%s (%s)" % (text, unit) if unit and str(unit) not in text else text

    def _write_labels(self, ax, rec, spec_labels, result_info, changed):
        for loc in ("center", "left", "right"):
            if ax.get_title(loc):
                ax.set_title("", loc=loc)
                changed.append("title")
        if rec.get("hold"):  # the setting the panel was declared to show (`fig.panel`): the plan's label and the value, FI's words
            ax.set_title(self._hold_title(rec))
        if rec["x"] is None:
            return
        image = any(e["kind"] == "image" for e in rec["series"])
        x = rec["x"]
        if x.get("neutral"):
            x["label"] = NEUTRAL_X
        elif not (image and x["id"] == "column" and "column" not in spec_labels):
            x["label"] = self._name_of(x["id"], spec_labels, result_info)
        if rec["y"] is not None:
            parts = rec["y"].get("names", rec["y"]["what"]).split(" / ")
            names = dict.fromkeys(NEUTRAL_Y if w == "other" else self._name_of(w, spec_labels, result_info) for w in parts)
            rec["y"]["label"] = " / ".join(names)
        if x.get("scale") == "log":
            x["label"] += LOG_NOTE
        if rec["y"] is not None and rec["y"].get("scale") == "log":
            rec["y"]["label"] += LOG_NOTE
        ytext = "row" if image else (rec["y"] or {}).get("label", "")
        for old, text, setter in ((ax.get_xlabel(), x["label"], ax.set_xlabel), (ax.get_ylabel(), ytext, ax.set_ylabel)):
            if old and old != text:
                changed.append("label")
            setter(text)

    @staticmethod
    def _axis_scale(declared, centre, full, bars_y):
        """``(scale, source, note)`` of one axis: what the plan declares, else FI's rule over the values drawn along it (``centre``:
        the points and lines; ``full``: those and the ends of every error bar, band and reference line). A log axis cannot show a
        value at or below zero: where the plan declares one (or FI's rule would choose one) over such values the axis is linear and
        the note says why."""
        blocked = not full or min(full) <= 0
        why = "nothing is drawn along it" if not full else "a value drawn along it (an error bar or interval end included) is zero or negative"
        if declared in ("linear", "log"):
            if declared == "log" and blocked:
                return "linear", "plan", "the plan asked for a log axis but " + why
            return declared, "plan", None
        spans = bool(centre) and min(centre) > 0 and max(centre) / min(centre) >= LOG_RATIO and not bars_y
        if spans and blocked:
            return "linear", "FI rule", "FI's rule would draw a log axis but " + why
        return ("log" if spans else "linear"), "FI rule", None

    @staticmethod
    def _panel_values(ax, rec):
        """``(centre, full, skip)`` of a panel: for each axis the values of its points and lines, and those with the ends of the error
        bars, bands and reference lines; ``skip`` is the axes a panel does not scale (x of bars in slots, a categorical x)."""
        series = rec["series"]
        categorical = any(isinstance(v, str) for e in series for v in e["x"]) or any(isinstance(v, str) for b in rec["bands"] for v in b["x"])
        yv, xv = [], []
        for e in series:
            yv += [v for v in e["y"] if v is not None]
            lo, hi = _sides(e)
            if lo is not None:
                for v, dl, dh in zip(e["y"], lo, hi):
                    if v is not None and dl is not None:
                        yv.append(v - dl)
                    if v is not None and dh is not None:
                        yv.append(v + dh)
            xv += [v for v in e["x"] if _finite(v)]
        centre = {"x": list(xv), "y": [v for e in series for v in e["y"] if v is not None]}
        for b in rec["bands"]:
            yv += [v for v in b["lo"] + b["hi"] if v is not None]
            xv += [v for v in b["x"] if _finite(v)]
        for r in rec["refs"]:
            (yv if r["axis"] == "y" else xv).append(r["value"])
            centre["y" if r["axis"] == "y" else "x"].append(r["value"])
        skip = set()
        if getattr(ax, "_fi_bar_slots", None) or categorical:
            skip.add("x")
        return centre, {"x": xv, "y": yv}, skip

    def _figure_scales(self, live, axis_scales):
        """``{axis: (scale, source, note)}``, decided once for the whole figure over the values of every panel that scales that axis:
        the plan declares one scale for a figure, and panels that share an axis must be read on the same one, so a panel that cannot
        be drawn on a log axis makes every panel linear."""
        centre, full, bars = {"x": [], "y": []}, {"x": [], "y": []}, False
        for i in live:
            ax, rec = self._panel(i)
            if rec["x"] is None or rec["y"] is None or any(e["kind"] == "image" for e in rec["series"]):
                continue
            c, f, skip = self._panel_values(ax, rec)
            for axis in ("x", "y"):
                if axis not in skip:
                    centre[axis] += [v for v in c[axis] if _finite(v)]
                    full[axis] += [v for v in f[axis] if _finite(v)]
            bars = bars or any(e["kind"] == "bars" for e in rec["series"])
        return {axis: self._axis_scale((axis_scales or {}).get(axis), centre[axis], full[axis], bars and axis == "y") for axis in ("x", "y")}

    def _write_limits(self, ax, rec, changed, scales):
        if rec["x"] is None or rec["y"] is None:
            return
        if not ax.get_autoscalex_on() or not ax.get_autoscaley_on():
            changed.append("limits")
        if any(e["kind"] == "image" for e in rec["series"]):
            if (ax.get_xscale(), ax.get_yscale()) != ("linear", "linear"):
                changed.append("scale")
            return
        before = (ax.get_xscale(), ax.get_yscale())
        series = rec["series"]
        _centre, full, skip = self._panel_values(ax, rec)
        bars = [e for e in series if e["kind"] == "bars"]
        for axis in ("x", "y"):
            vals = full[axis]
            if axis == "x" and getattr(ax, "_fi_bar_slots", None):  # bars over a setting: one slot for each value
                ax.set_xlim(-0.5, ax._fi_bar_slots - 0.5)
                continue
            if axis == "x" and "x" in skip:  # a categorical x
                ax.relim()
                ax.autoscale(enable=True, axis="x")
                continue
            scale, source, note = scales[axis]
            rec[axis].pop("scale_note", None)
            rec[axis].update(scale=scale, scale_source=source)
            if note:
                rec[axis]["scale_note"] = note
            vals = [v for v in vals if _finite(v)]
            if bars and axis == "y" and scale == "linear":
                vals.append(0.0)
            if axis == "x" and scale == "linear":  # bars have width: half the closest spacing on each side
                for e in bars:
                    pts = sorted({v for v in e["x"] if _finite(v)})
                    if pts:
                        half = min((b - a for a, b in zip(pts, pts[1:])), default=1.0) / 2
                        vals += [pts[0] - half, pts[-1] + half]
            if scale == "log":
                vals = [v for v in vals if v > 0]  # (a bar's foot at zero is not drawn on a log axis)
            (ax.set_xscale if axis == "x" else ax.set_yscale)(scale)
            if vals:
                (ax.set_xlim if axis == "x" else ax.set_ylim)(*_padded(min(vals), max(vals), scale))
        if before != ("linear", "linear") and before != (ax.get_xscale(), ax.get_yscale()):
            changed.append("scale")

    @staticmethod
    def _level_ident(e):
        """The curve of a level a record entry belongs to: its result and the level's text (a curve drawn as a line and as points
        is one curve). The result is the one the caption names: its id with the label FI's check left it, so a series that is not
        the table's value of its declared result is its own curve beside one that is."""
        return (FigureCalls._result_key(e), e["level"]["text"])

    @staticmethod
    def _result_key(e):
        return (e["level"]["result"], e["level"].get("result_label"))

    @staticmethod
    def _condensed(rec):
        """The results (``_result_key``) of a panel whose levels the legend condenses: those that draw more than ``LEGEND_LEVELS`` distinct levels of a
        setting, all of them numbers (an ordered scale: a colour ramp tells the levels between the named ones apart). Counted for
        each result on its own; levels that are names (methods, arms) are never condensed."""
        out = set()
        for result in {FigureCalls._result_key(e) for e in rec["series"] if e.get("level")}:
            mine = [e for e in rec["series"] if e.get("level") and FigureCalls._result_key(e) == result]
            if len({FigureCalls._level_ident(e) for e in mine}) > LEGEND_LEVELS and all(_finite(e["level"]["value"]) for e in mine):
                out.add(result)
        return out

    @staticmethod
    def _shown_levels(rec):
        """The positions of the series a legend names when a result of the panel draws more than ``LEGEND_LEVELS`` distinct numeric
        levels of a setting: the lowest and the highest level of that result (the caption says the range), every curve of any other
        result; None when the legend names them all."""
        condensed = FigureCalls._condensed(rec)
        if not condensed:
            return None
        keep = set()
        for result in {FigureCalls._result_key(e) for e in rec["series"] if e.get("level")}:
            mine = [(k, e["level"]["value"]) for k, e in enumerate(rec["series"]) if e.get("level") and FigureCalls._result_key(e) == result]
            if result in condensed:
                ends = {min(v for _k, v in mine), max(v for _k, v in mine)}
                keep |= {k for k, v in mine if v in ends}
            else:
                keep |= {k for k, _v in mine}
        return keep

    @staticmethod
    def _recolour(art, colour):
        """Give a drawn artist, or the container or list of the artists of one call, one colour."""
        if art is None:
            return
        if isinstance(art, (list, tuple)):
            for a in art:
                FigureCalls._recolour(a, colour)
        elif hasattr(art, "set_color"):
            art.set_color(colour)
        elif hasattr(art, "set_facecolor"):
            art.set_facecolor(colour)
        else:  # an errorbar container: caps and bar lines (the data line is none)
            FigureCalls._recolour(list(getattr(art, "lines", None) or []), colour)

    def _level_colours(self, panel, rec):
        """One colour for each level: the curve's line, its markers, error bars and band share it. More than ``LEGEND_LEVELS``
        numeric levels are coloured along one scale in the order of the level, so the two named in the legend say where the rest
        lie; a failing colour map leaves the default colours."""
        extra = self._fc_extra.get(panel, {"series": []})
        arts = extra["series"]
        groups = {}
        for k, e in enumerate(rec["series"]):
            if e.get("level") and k < len(arts) and arts[k][0] is not None:
                groups.setdefault(self._level_ident(e), []).append(k)
        if not groups:
            return
        colour = {}
        condensed = self._condensed(rec)
        if condensed:
            try:
                from matplotlib import colormaps
                cmap = colormaps["viridis"]
            except Exception:  # noqa: BLE001 -- no colour map: the curves keep the colours they were drawn with
                cmap = None
            span = getattr(self, "_fc_level_span", None) or {}
            for result in {r for r, _t in groups if r in condensed}:
                mine = [(rec["series"][ks[0]]["level"]["value"], ident) for ident, ks in groups.items() if ident[0] == result]
                if cmap is None or not all(_finite(v) for v, _i in mine) or len({v for v, _i in mine}) < 2:
                    continue
                lo, hi = min(v for v, _i in mine), max(v for v, _i in mine)
                setting = rec["series"][groups[mine[0][1]][0]]["level"].get("setting")
                if setting in span and span[setting][1] > span[setting][0]:
                    lo, hi = span[setting]  # the scale every panel shares: one value, one colour
                colour.update({ident: cmap((v - lo) / (hi - lo)) for v, ident in mine})
        for ident, ks in groups.items():
            if ident not in colour:  # the colour the first drawn of the level got from the colour cycle
                first = arts[ks[0]][0]
                if not hasattr(first, "get_color"):
                    continue
                colour[ident] = first.get_color()
        for ident, ks in groups.items():
            if ident not in colour:
                continue
            for k in ks:
                art = arts[k][0]
                if rec["series"][k]["kind"] in ("line", "points") and hasattr(art, "set_color"):
                    art.set_color(colour[ident])
                self._recolour((extra.get("bars") or {}).get(k), colour[ident])
        for sid, art in extra.get("bands") or []:
            for k, e in enumerate(rec["series"]):
                if e["id"] == sid and e.get("level") and e["kind"] != "curve":
                    self._recolour(art, colour.get(self._level_ident(e)))
                    break

    def _entries_keyed(self, panel, rec, spec_labels, result_info):
        """``[(key, artist, text)]`` of the legend, the key being what makes two panels' entries one (the series id, with the
        level of a setting when the curve is one; a reference line's id). The entries are: one per drawn series (the merged ones read "a = b") and per reference line. A text
        is the plan's label, else the record's label when it is not the bare id, else a short form of the plan's description,
        else "value": never the id."""
        extra = self._fc_extra.get(panel, {"series": [], "refs": []})
        twins = extra.get("twins", set())
        recorded = {}
        for e in rec["series"]:
            recorded.setdefault(e["id"], e["label"])

        def name(ident, record_label=None):
            sl = spec_labels.get(ident)
            got = sl.get("label") if isinstance(sl, dict) else sl
            record_label = record_label if record_label is not None else recorded.get(ident)
            if not got and record_label and record_label != ident:
                got = record_label
            return str(got or _short((result_info.get(ident) or {}).get("description")) or "value")

        # FI's neutral name, never the plan's label of a series whose values FI did not find, whichever was drawn first
        neutral_by_id = {s["id"]: s["label"] for s in rec["series"] if s.get("neutral")}
        shown = self._shown_levels(rec)
        named = set()  # a level drawn as a line and as points is named once
        out = []
        for k, e in enumerate(rec["series"]):
            if e["kind"] == "image" or k >= len(extra["series"]) or extra["series"][k][0] is None or k in twins:
                continue
            if shown is not None and e.get("level") and k not in shown:
                continue
            if e.get("level"):
                if self._level_ident(e) in named:
                    continue
                named.add(self._level_ident(e))
            plan_curve = e["kind"] == "curve" and e.get("source") == "plan" and e["id"] in self._results()
            first = e["label"] if e.get("neutral") or plan_curve else name(e["id"], e["label"])
            text = " = ".join([first] + [neutral_by_id.get(i) or name(i) for i in e["merged_with"]])
            out.append((self._colour_key(e), extra["series"][k][0], text))
        for r, art in zip(rec["refs"], extra["refs"]):
            # a reference that is the plan's formula for a result keeps the name ref() gave it, never the result's own name
            text = r["label"] if r.get("source") == "plan" and r["id"] in self._results() else name(r["id"], r["label"])
            out.append((("ref", r["id"]), art, text))
        return out

    @staticmethod
    def _colour_key(e):
        """What gives a series its colour: the curve of a level of a setting, else the series id, so one series is one colour in
        every panel."""
        return FigureCalls._level_ident(e) if e.get("level") else (e["id"],)

    def _figure_of_panels(self):
        fig = getattr(self, "fig", None)
        if fig is None:
            axes = list(getattr(self, "axes", []) or [])
            fig = axes[0].figure if axes else None
        return fig

    @staticmethod
    def _use_constrained_layout(fig):
        """The layout engine that makes room for the labels, the legend and the titles of every panel (the figure is made with it;
        a body that set another one gets this one back)."""
        try:
            engine = fig.get_layout_engine()
            if engine is None or type(engine).__name__ != "ConstrainedLayoutEngine":
                fig.set_layout_engine("constrained")
        except Exception:  # noqa: BLE001 -- a figure that cannot take the engine keeps its own layout
            pass

    def _share_colours(self, panels):
        """One colour for one series in every panel (a panel's colour cycle starts again, so the second panel's first series would
        otherwise take the first panel's colour): by the order the series first appear in, along the house style's cycle. The
        line, markers, bars and band of a series share it; a curve of a level of a setting is coloured again by
        ``_level_colours`` after this."""
        try:
            import matplotlib
            cycle = list(matplotlib.rcParams["axes.prop_cycle"].by_key().get("color") or [])
        except Exception:  # noqa: BLE001 -- no cycle to read: the series keep the colours they were drawn with
            return
        if not cycle:
            return
        # a value of a setting is one colour in every panel, unless some panel draws two results at that value (then each
        # result keeps its own colour there, so the two curves of one panel stay apart)
        results_at = {}
        for i in panels:
            for e in self._panel(i)[1]["series"]:
                if e.get("level") and e["kind"] != "image":
                    results_at.setdefault((i, e["level"].get("setting"), e["level"]["text"]), set()).add(FigureCalls._result_key(e))
        shared = all(len(r) == 1 for r in results_at.values())
        span = {}  # the range of each setting's levels over every panel, for the one colour scale they then share
        if shared:
            for i in panels:
                for e in self._panel(i)[1]["series"]:
                    v = (e.get("level") or {}).get("value")
                    if e.get("level") and _finite(v):
                        lo, hi = span.get(e["level"].get("setting"), (v, v))
                        span[e["level"].get("setting")] = (min(lo, v), max(hi, v))
        self._fc_level_span = span

        def key_of(e):
            if shared and e.get("level"):
                return ("level", e["level"].get("setting"), e["level"]["text"])
            return self._colour_key(e)

        colour = {}
        for i in panels:
            _ax, rec = self._panel(i)
            extra = self._fc_extra.get(i, {"series": []})
            arts = extra["series"]
            for k, e in enumerate(rec["series"]):
                if e["kind"] in ("curve", "image") or k >= len(arts) or arts[k][0] is None:
                    continue
                key = key_of(e)
                if key not in colour:
                    colour[key] = cycle[len(colour) % len(cycle)]
                self._recolour((extra.get("drawn") or {}).get(k) or arts[k][0], colour[key])
            for sid, art in extra.get("bands") or []:
                for e in rec["series"]:
                    if e["id"] == sid and e["kind"] != "curve" and key_of(e) in colour:
                        self._recolour(art, colour[key_of(e)])
                        break

    def _write_legend(self, ax, panel, rec, entries):
        """Draw the legend in a free corner; if it needs room outside, return ``(ax, entries)`` for ``_legend_outside``; ``entries`` are the keyed
        entries of ``_entries_keyed``."""
        if not entries:
            return None
        handles, texts = [a for _k, a, _t in entries], [t for _k, _a, t in entries]
        corner = self._free_corner(ax, rec, len(entries)) if len(entries) <= 6 else None
        if corner is not None:
            ax.legend(handles, texts, loc=corner, fontsize="small", framealpha=0.85)
            self._fc_extra.setdefault(panel, {"series": [], "refs": []})["legend_corner"] = corner
            self._fc_legend = {"scope": "panel", "panel": panel + 1, "where": corner, "entries": list(texts)}
            return None
        return ax, entries

    def _legend_clash(self, fig, live):
        """What the drawn legends cover, measured on the canvas the figure will be saved from: ``{"over_data": how many drawn
        points, and samples along a drawn line (one every 4 pixels), lie under a legend's box, "over_panel": the largest share of a
        legend's box that lies over a panel it does not belong to}``; None when the canvas cannot be measured."""
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            own = [(self._panel(i)[0], self._panel(i)[0].get_legend()) for i in live]
            own = [(ax, leg) for ax, leg in own if leg is not None and leg.get_visible()]
            loose = [(None, leg) for leg in fig.legends if leg.get_visible()]
            over_data, over_panel = 0, 0.0
            for owner, leg in own + loose:
                box = leg.get_window_extent(renderer)
                if box.width <= 2 * LEGEND_INSET_PX or box.height <= 2 * LEGEND_INSET_PX:
                    continue
                x0, y0 = box.x0 + LEGEND_INSET_PX, box.y0 + LEGEND_INSET_PX
                x1, y1 = box.x1 - LEGEND_INSET_PX, box.y1 - LEGEND_INSET_PX
                for i in live:
                    ax = self._panel(i)[0]
                    area = ax.get_window_extent(renderer)
                    if ax is not owner:
                        w, h = min(box.x1, area.x1) - max(box.x0, area.x0), min(box.y1, area.y1) - max(box.y0, area.y0)
                        if w > 0 and h > 0:
                            over_panel = max(over_panel, w * h / (box.width * box.height))
                    lo_x, hi_x, lo_y, hi_y = max(x0, area.x0), min(x1, area.x1), max(y0, area.y0), min(y1, area.y1)
                    if hi_x <= lo_x or hi_y <= lo_y:
                        continue
                    for art in list(ax.get_lines()) + list(ax.collections):
                        if not art.get_visible():
                            continue
                        if hasattr(art, "get_xydata"):
                            if art.get_transform() != ax.transData:
                                continue
                            raw, trans = art.get_xydata(), ax.transData
                            joined = str(art.get_linestyle()) not in ("None", "none", "", " ")
                        else:
                            raw, trans, joined = art.get_offsets(), art.get_offset_transform(), False
                        pts = [(float(a), float(b)) for a, b in trans.transform(raw) if math.isfinite(a) and math.isfinite(b)]
                        for k, (px, py) in enumerate(pts):
                            over_data += lo_x <= px <= hi_x and lo_y <= py <= hi_y
                            if joined and k + 1 < len(pts):
                                qx, qy = pts[k + 1]
                                steps = int(math.hypot(qx - px, qy - py) // 4)
                                over_data += sum(1 for t in range(1, steps + 1)
                                                 if lo_x <= px + (qx - px) * t / (steps + 1) <= hi_x
                                                 and lo_y <= py + (qy - py) * t / (steps + 1) <= hi_y)
            return {"over_data": int(over_data), "over_panel": round(over_panel, 2)}
        except Exception:  # noqa: BLE001 -- a figure that cannot be measured keeps its legend where it is
            return None

    @staticmethod
    def _clashes(found):
        return found is not None and (found["over_data"] > 0 or found["over_panel"] >= LEGEND_SHARE)

    def _keep_legend_off_data(self, fig, live, keyed):
        """A legend that covers the data it names, or a panel it does not belong to, is put outside the axes: FI measures the
        legend it drew and, when it clashes (``_legend_clash``: any drawn point or line under it, or ``LEGEND_SHARE`` of its box
        over another panel), draws it again outside, below the panels or to their right, whichever clears more, and measures
        again. The record's ``legend`` says what was measured and where the legend went (``adjusted``); a legend that no
        placement clears stays where it clears most and the record says so."""
        if fig is None or not live:
            return
        before = self._legend_clash(fig, live)
        if not self._clashes(before) or not getattr(self, "_fc_legend", None):
            return
        panels = [i for i in live if keyed.get(i)]
        if not panels:
            return
        args = ([None], {None: keyed[panels[0]]}) if len(panels) == 1 else (panels, keyed)
        for i in live:  # a legend inside a panel's axes goes: the figure's own legend, outside, takes its entries
            old = self._panel(i)[0].get_legend()
            if old is not None:
                old.remove()
        best = None
        for side in ("below", "right"):
            self._figure_legend(fig, *args, prefer=side)
            found = self._legend_clash(fig, live)
            if found is None:
                continue
            score = (found["over_data"] > 0, found["over_panel"])
            if best is None or score < best[0]:
                best = (score, side, found)
            if not self._clashes(found):
                break
        if best is None:
            return
        self._figure_legend(fig, *args, prefer=best[1])
        self._fc_legend["adjusted"] = {"was": before, "moved_outside": best[1], "after": best[2], "cleared": not self._clashes(best[2])}

    def _legend_outside(self, ax, entries):
        """A single panel's legend that fits in no corner goes outside the axes, as the one legend of the figure; its entries keep
        their own keys, so two series (or reference lines) that share a label stay two entries."""
        self._figure_legend(ax.figure, [None], {None: entries})

    def _figure_legend(self, fig, panels, keyed, prefer=None):
        """One legend for the figure: the entries of every panel in ``panels``, each series once (by its key), outside the axes
        (below them, in as many columns as fit the width of the figure; to their right when that would be more than four rows
        and the legend is narrow). The constrained layout makes the room, so no panel is moved by hand."""
        seen, handles, texts, covers, alike = set(), [], [], [], {}
        for i in panels:
            for key, art, text in keyed[i]:
                is_ref = bool(key) and key[0] == "ref"
                look = None if is_ref else self._look(art)
                # one reference of the plan drawn in several panels is one line; a series is one entry with an earlier one
                # only when it is drawn the same (a series whose look cannot be read keeps its own entry)
                mark = (key, text) if is_ref else (key, text, look) if look is not None else (key, text, id(art))
                if mark in seen:
                    continue
                seen.add(mark)
                if look is not None and (text, look) in alike:  # same name, drawn the same: a reader sees one thing
                    covers[alike[(text, look)]] += 1
                    continue
                if look is not None:
                    alike[(text, look)] = len(texts)
                handles.append(art)
                texts.append(text)
                covers.append(1)
                try:
                    art.set_label(text)  # the artist carries the name the legend shows, for the recorder of the saved figure
                except Exception:  # noqa: BLE001
                    pass
        try:
            renderer = fig.canvas.get_renderer()
        except AttributeError:  # a canvas that cannot measure: a legend inside the first panel is still better than none
            fig.axes[0].legend(handles, texts, loc="upper right", fontsize="small", framealpha=0.85)
            return

        def make(loc, ncol):
            for old in list(fig.legends):
                if getattr(old, "_fi_legend", False):
                    old.remove()
            leg = fig.legend(handles, texts, loc=loc, ncol=ncol, fontsize="small", frameon=False)
            leg._fi_legend = True
            return leg

        room = 0.95 * fig.bbox.width
        for ncol in range(len(texts), 0, -1):
            leg = make("outside lower center", ncol)
            if leg.get_window_extent(renderer).width <= room or ncol == 1:
                break
        where = "below"
        if prefer == "right" or (prefer is None and int(math.ceil(len(texts) / ncol)) > 4):
            side = make("outside right upper", 1)
            if prefer == "right" or side.get_window_extent(renderer).width <= 0.25 * fig.bbox.width:
                where = "right"
            else:
                make("outside lower center", ncol)
        self._fc_legend = {"scope": "figure", "where": where, "entries": list(texts), "covers": covers}

    @staticmethod
    def _look(art):
        """How a legend entry's artist looks (colour, marker, line style, as matplotlib draws it), or ``None`` when it cannot be
        read: two entries are one only when both their name and their look are the same."""
        try:
            from matplotlib.colors import to_hex
            first = art
            if hasattr(art, "lines") and not hasattr(art, "get_color"):  # an error-bar container: its data line
                first = art.lines[0] if art.lines and art.lines[0] is not None else None
            elif hasattr(art, "patches") and getattr(art, "patches", None):  # bars
                first = art.patches[0]
            if first is None:
                return None
            if hasattr(first, "get_marker"):
                return ("line", to_hex(first.get_color()), str(first.get_marker()), str(first.get_linestyle()))
            if hasattr(first, "get_facecolor"):
                fc = first.get_facecolor()
                fc = fc[0] if len(fc) and hasattr(fc[0], "__len__") else fc
                shape = ()
                paths = first.get_paths() if hasattr(first, "get_paths") else []
                if paths:  # the mark of a scatter: a circle and a cross of one colour are two looks
                    shape = tuple(round(float(c), 3) for c in paths[0].vertices.ravel()[:40])
                return ("patch", to_hex(fc), type(first).__name__, shape)
        except Exception:  # noqa: BLE001 -- a look that cannot be read never merges two entries
            return None
        return None

    @staticmethod
    def _wrapped(text, limit, length):
        """``text`` broken between words onto lines no longer than ``limit`` by ``length(line)`` (a word longer than that keeps a
        line to itself: nothing is cut); the lines it already has stay apart."""
        out = []
        for line in str(text).split(chr(10)):
            current = []
            for word in line.split():
                if current and length(" ".join(current + [word])) > limit:
                    out.append(" ".join(current))
                    current = [word]
                else:
                    current.append(word)
            out.append(" ".join(current))
        return chr(10).join(out)

    def _wrap_to_panels(self, panels):
        """A y label longer than its panel's height and an x label or a note wider than its panel's width are broken between words onto as many
        lines as they need (the text is never cut): first to the size a panel is expected to get, then, after the layout has
        drawn the figure, to the size each panel got (unless the layout left it next to nothing), until nothing changes. A text
        that was already wrapped is wrapped again from its words."""
        axes = [self._panel(i)[0] for i in panels]
        if not axes:
            return
        fig = axes[0].figure
        try:
            from matplotlib.text import Text
            renderer = fig.canvas.get_renderer()
            guess = {}
            for ax in axes:  # before any layout: about 60% of the height and 80% of the width of the panel's cell in the grid
                try:
                    grid = ax.get_subplotspec().get_gridspec()
                    rows, cols = grid.nrows, grid.ncols
                except Exception:  # noqa: BLE001
                    rows = cols = 1
                guess[ax] = (0.8 * fig.bbox.width / max(1, cols), 0.6 * fig.bbox.height / max(1, rows))
            room = dict(guess)
            for turn in range(3):
                if turn:  # the size the layout gave each panel, unless it gave one next to nothing
                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    for ax in axes:
                        box = ax.get_window_extent(renderer)
                        room[ax] = (box.width if box.width >= 0.4 * guess[ax][0] else guess[ax][0],
                                    box.height if box.height >= 0.4 * guess[ax][1] else guess[ax][1])
                changed = False
                for k, ax in enumerate(axes):
                    texts = [(ax.yaxis.label, 90, 1), (ax.xaxis.label, 0, 0)] + [(art, 0, 0) for art in self._fc_extra.get(panels[k], {}).get("notes", [])]
                    for art, rotation, along in texts:
                        words = " ".join(art.get_text().split()) if rotation else art.get_text()
                        if not words.strip():
                            continue
                        probe = Text(0, 0, "", fontproperties=art.get_fontproperties(), rotation=rotation)
                        probe.set_figure(fig)

                        def length(line, probe=probe, along=along):
                            probe.set_text(line)
                            box = probe.get_window_extent(renderer)
                            return box.height if along else box.width

                        # the width a note may take: its panel's, less a margin; the label's: the panel's height
                        limit = room[ax][1] if along else 0.9 * room[ax][0]
                        new = self._wrapped(words, limit, length)
                        if new != art.get_text():
                            art.set_text(new)
                            changed = True
                if turn and not changed:
                    break
        except Exception:  # noqa: BLE001 -- a text that cannot be measured stays as it is
            return

    def _free_corner(self, ax, rec, rows):
        """The first corner (upper right, upper left, lower right, lower left) the drawn data does not reach, or None."""
        counts = self._corner_counts(ax, rec, rows)
        return next((name for name, n in (counts or {}).items() if n == 0), None)

    def _corner_counts(self, ax, rec, rows):
        """``{corner: how many drawn points, step samples, error-bar ends and reference lines reach it}`` in the order of
        preference, a corner being 40% of the axes wide and as high as ``rows`` of text need; None for an image panel."""
        w, h = 0.4, min(0.5, 0.04 + 0.07 * rows)
        boxes = {"upper right": (1 - w, 1, 1 - h, 1), "upper left": (0, w, 1 - h, 1),
                 "lower right": (1 - w, 1, 0, h), "lower left": (0, w, 0, h)}
        to_axes = ax.transAxes.inverted()
        pts = []

        def along(xs, ys):  # the points, and 16 samples along each step between them (in the axes' own fractions)
            fr = []
            for x, y in zip(xs, ys):
                if _finite(x) and y is not None and _finite(y):
                    fx, fy = to_axes.transform(ax.transData.transform((x, y)))
                    if math.isfinite(fx) and math.isfinite(fy):
                        fr.append((fx, fy))
            pts.extend(fr)
            for a, b in zip(fr, fr[1:]):
                pts.extend((a[0] + (b[0] - a[0]) * t / 16, a[1] + (b[1] - a[1]) * t / 16) for t in range(1, 16))

        placed = getattr(ax, "_fi_bar_x", None) or {}
        for k, e in enumerate(rec["series"]):
            if e["kind"] == "image":
                return None  # an image fills its axes
            xs = placed.get(k) if e["kind"] == "bars" and placed.get(k) and len(placed[k]) == len(e["x"]) else e["x"]
            along(xs, e["y"])
            lo, hi = _sides(e)
            if lo is not None:
                along(xs, [None if v is None or d is None else v + d for v, d in zip(e["y"], hi)])
                along(xs, [None if v is None or d is None else v - d for v, d in zip(e["y"], lo)])
            if e["kind"] == "bars":
                along(xs, [0.0] * len(xs))
        for b in rec["bands"]:
            along(b["x"], b["lo"])
            along(b["x"], b["hi"])
        lines = []
        for r in rec["refs"]:
            point = (r["value"], 1.0) if r["axis"] == "x" else (1.0, r["value"])
            fx, fy = to_axes.transform(ax.transData.transform(point))
            lines.append((r["axis"], fx if r["axis"] == "x" else fy))
        out = {}
        for name, (x0, x1, y0, y1) in boxes.items():
            out[name] = sum(1 for x, y in pts if x0 <= x <= x1 and y0 <= y <= y1) \
                + sum(1 for a, v in lines if (a == "x" and x0 <= v <= x1) or (a == "y" and y0 <= v <= y1))
        return out

    def _place_notes(self, ax, panel, rec):
        """Put each note drawn without x and y in a corner the data and the legend leave free; with none free, in the corner
        the fewest drawn points reach. An untaken corner is preferred; two notes in one corner stack."""
        extra = self._fc_extra.get(panel, {})
        arts = extra.get("notes", [])
        taken = [extra["legend_corner"]] if extra.get("legend_corner") else []
        for n, art in zip(rec["notes"], arts):
            if n["x"] is not None:
                continue
            counts = self._corner_counts(ax, rec, 1 + str(n["text"]).count("\n"))
            order = list(counts) if counts else ["upper right", "upper left", "lower right", "lower left"]
            fresh = [c for c in order if c not in taken] or order
            free = [c for c in fresh if not counts or counts[c] == 0]
            corner = free[0] if free else min(fresh, key=lambda c: counts[c] if counts else 0)
            stack = taken.count(corner)
            taken.append(corner)
            top, right = corner.startswith("upper"), corner.endswith("right")
            step = 0.07 * stack
            art.set_transform(ax.transAxes)
            art.set_position((0.98 if right else 0.02, (0.97 - step) if top else (0.03 + step)))
            art.set_ha("right" if right else "left")
            art.set_va("top" if top else "bottom")
            n["corner"] = corner

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
        out = {"version": RECORD_VERSION, "fallback": bool(getattr(self, "fallback", False)), "panels": panels}
        if getattr(self, "_fc_removed", None):  # the lines left out in the last run: FI could not say what they are
            out["removed"] = json.loads(json.dumps(self._fc_removed))
        if getattr(self, "_fc_legend", None):  # where the figure's legend is: one for the figure, or the panel's own
            out["legend"] = json.loads(json.dumps(self._fc_legend))
        return out

    def write(self, path):
        """Write the record as ``<stem>.calls.json`` (``path`` is the figure's file or a ``.calls.json`` path); returns the path."""
        p = Path(path)
        out = p if p.name.endswith(".calls.json") else p.with_name(p.stem + ".calls.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.record(), indent=1, allow_nan=False), encoding="utf-8")
        return out
