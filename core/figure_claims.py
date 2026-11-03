"""The statements FI can make about a figure from its own record of what the figure draws.

A writer who describes a figure in words says what the picture "should" show. The root fix is the one ``{{seed}}`` uses:
the fact comes from FI's records. Every figure drawn through FI's drawing calls leaves a *call record*
(``<stem>.calls.json`` beside the recorder files, written by ``core/frame_figure.py``) holding, per panel, the axes, each
drawn series with its points, bands, reference lines and notes. This module holds the catalogue of claims over that record.

* :func:`menu` lists the claims that are TRUE on the record, as marks the writer may place
  (``{{figure:fig1:trend @ series=a, panel=1}}``).
* :func:`evaluate` re-computes any claim the writer names at fill time and returns FI's own sentence, or ``None`` when the
  claim is not in the catalogue, names a series, panel or x the record does not hold, sits on a raw panel, or is not true.
  ``None`` always means fail closed: the mark stays unfilled.

All comparisons use drawn points only; nothing is interpolated. A claim is defined over the geometry of drawn data, never
over free text. Numbers in a sentence go through an injected ``fmt(float) -> str`` (the engine passes the result-mark
formatter), so this module needs no ``paper_values`` import.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import figure_facts
from .frame_figure import RECORD_VERSION, hold_text as _hold_text
from . import paper_values as _values

#: Two drawn values closer than this (relative) are equal to the rounding of a drawn value; a trend may run through them.
TREND_TIE_RTOL = 5e-4
#: Two series whose points agree to this (relative) are the same data (they are merged by construction).
IDENTICAL_RTOL = 1e-12
#: ``value_at`` is offered at every drawn x of a series with at most this many points, else at the first and the last.
_MAX_VALUE_AT = 6
#: ``order`` is offered at the first, middle and last x shared by two or more series.
_MAX_ORDER_X = 3

Fmt = Callable[[float], str]
Offer = dict[str, Any]


def _default_fmt(value: float) -> str:
    return "%.4g" % value


def load_calls(quest_root: Path | str, stem: str) -> dict[str, Any] | None:
    """The call record of figure ``stem`` (``<quest_root>/.fi/figure_records/<stem>.calls.json``), or ``None`` when the file
    is missing, unreadable or not a call record."""
    name = str(stem)
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    try:
        data = json.loads((Path(quest_root) / figure_facts.RECORDS_DIR / f"{name}.calls.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("panels"), list) and _version(data) >= RECORD_VERSION:
        return data
    return None  # a record from before the checks of the current version is not read: the figure has no call record


def _version(data: dict[str, Any]) -> int:
    v = data.get("version")
    return v if isinstance(v, int) and not isinstance(v, bool) else 0


#: A value a mark carries as it is: none of the characters the mark is built from (the comma between arguments, ``=``, a
#: quote, a backslash, a brace, a control character) and no space at either end. What else a series id or a level can hold
#: (``@``, ``:``, a space inside) needs no quoting.
_PLAIN_VALUE = re.compile(r"[^,=\"\\{}\x00-\x1f\s]+(?:[ ]+[^,=\"\\{}\x00-\x1f\s]+)*")


def _quote_value(value: str) -> str:
    """A value as the mark carries it: as it is when plain, else a JSON string with its two braces written ``\\u007b`` and
    ``\\u007d``, so the mark still ends at its own closing braces."""
    if _PLAIN_VALUE.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False).replace("{", "\\u007b").replace("}", "\\u007d")


def format_args(args: dict[str, str]) -> str:
    """The arguments of a claim as a mark carries them, ``"series=a, panel=1"`` (the panel last): the inverse of
    :func:`parse_args`. A value with a comma, an equals sign, a quote, a backslash, a brace or a line break is quoted, so a
    series id of any text reads back whole; a value without them is written as it is."""
    return ", ".join(f"{k}={_quote_value(str(v))}" for k, v in sorted(args.items(), key=lambda kv: (kv[0] == "panel", kv[0])))


def parse_args(text: str) -> dict[str, str]:
    """``"series=a, panel=1"`` as ``{"series": "a", "panel": "1"}``; a value in quotes (:func:`format_args`) is read back
    whole. A piece that is not ``key=value`` (or a key given twice) is kept under the key ``""``, a key no claim accepts, so
    that :func:`evaluate` fails closed instead of ignoring it."""
    out: dict[str, str] = {}
    text = str(text)
    if not text.strip():
        return out
    decoder = json.JSONDecoder()
    pos, n = 0, len(text)
    while pos <= n:
        eq, comma = text.find("=", pos), text.find(",", pos)
        if eq != -1 and (comma == -1 or eq < comma) and text[eq + 1:].lstrip().startswith('"'):
            key = text[pos:eq].strip()
            try:
                value, end = decoder.raw_decode(text, text.index('"', eq + 1))
            except ValueError:
                value, end = None, n
            rest = text[end:].lstrip()
            if isinstance(value, str) and key and value and key not in out and (not rest or rest.startswith(",")):
                out[key] = value
                pos = n + 1 if not rest else text.index(",", end) + 1
                continue
        end = comma if comma != -1 else n
        piece = text[pos:end]
        key, sep, value = piece.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value or key in out:
            out[""] = piece.strip()
        else:
            out[key] = value
        pos = end + 1
    return out


# --- reading the record -------------------------------------------------------------------------------------------------

def _finite(v: Any) -> bool:
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except OverflowError:  # an integer too large for a float is not a drawn value
        return False


def _is_cat(series: dict[str, Any]) -> bool:
    """The series' x values are categories (text), e.g. bars over methods."""
    xs = series.get("x")
    return isinstance(xs, list) and any(isinstance(x, str) for x in xs)


def _pairs(series: dict[str, Any]) -> list[tuple[Any, float]]:
    """The drawn (x, y) pairs with a finite y and an x that is a finite number or a category; a missing (null or NaN)
    value is skipped. A series whose x and y differ in length has no readable points."""
    xs, ys = series.get("x"), series.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
        return []
    return [(float(a) if _finite(a) else a, float(b)) for a, b in zip(xs, ys)
            if _finite(b) and (_finite(a) or (isinstance(a, str) and a))]


def _points(series: dict[str, Any]) -> list[tuple[float, float]]:
    """The drawn (x, y) points of a series on a numeric x axis with both values finite, in drawn order. A series over
    categories has none (it supports no claim that needs an order or a distance on x)."""
    if series.get("kind") == "image" or _is_cat(series):
        return []
    return [(a, b) for a, b in _pairs(series)]


def _at(series: dict[str, Any], raw: str) -> list[tuple[Any, float]]:
    """The drawn pairs of a series at the x (or category) named by ``raw``; no interpolation."""
    if series.get("kind") == "image":
        return []
    if _is_cat(series):
        return [(x, y) for x, y in _pairs(series) if x == raw]
    xv = _x_arg(raw)
    return [(x, y) for x, y in _pairs(series) if xv is not None and _same_x(x, xv)]


def _xtext(x: Any, fmt: Fmt) -> str:
    return x if isinstance(x, str) else fmt(x)


def _xarg(x: Any) -> str:
    return x if isinstance(x, str) else "%.12g" % x


def _same_x(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _tie(a: float, b: float) -> bool:
    return abs(a - b) <= TREND_TIE_RTOL * max(abs(a), abs(b))


def _panel(calls: dict[str, Any], args: dict[str, str]) -> dict[str, Any] | None:
    """The named panel (1-based), or the only panel when the figure has one and none is named. ``None`` for a raw panel."""
    panels = calls.get("panels")
    if not isinstance(panels, list) or not panels:
        return None
    raw = args.get("panel")
    if raw is None:
        if len(panels) != 1:
            return None
        number = panels[0].get("panel") if isinstance(panels[0], dict) else None
    else:
        try:
            number = int(raw)
        except ValueError:
            return None
    found = [p for p in panels if isinstance(p, dict) and p.get("panel") == number]
    if len(found) != 1 or found[0].get("raw"):
        return None
    return found[0]


def _series(panel: dict[str, Any], sid: str | None) -> dict[str, Any] | None:
    found = [s for s in panel.get("series") or [] if isinstance(s, dict) and s.get("id") == sid]
    return found[0] if len(found) == 1 else None


def _label(item: dict[str, Any]) -> str:
    if item.get("kind") == "curve":  # the body's words never name a curve, whatever a record holds
        return "a curve"
    return str(item.get("label") or item.get("id") or "")


def hold_text(panel: dict[str, Any]) -> str:
    """The setting a panel was declared to show (``fig.panel``), as "Mass = 200"; empty when it has none. The same words and
    the same value form as the panel's title (``core/frame_figure.py``)."""
    return _hold_text(panel)


def panel_name(panel: dict[str, Any]) -> str:
    """How a panel is named in a sentence: by the setting it shows ("the panel at m = 200") when it has one, else "panel 2"."""
    held = hold_text(panel)
    return f"the panel at {held}" if held else f"panel {panel.get('panel')}"


def _where(calls: dict[str, Any], panel: dict[str, Any]) -> str:
    if hold_text(panel):
        return f" in {panel_name(panel)}"
    return f" in panel {panel.get('panel')}" if len(calls.get("panels") or []) > 1 else ""


def _norm(panel: dict[str, Any], args: dict[str, str], **extra: str) -> dict[str, str]:
    out = {k: v for k, v in args.items() if k != "panel"}
    out.update(extra)
    out["panel"] = str(panel.get("panel"))
    return out


def _axis(panel: dict[str, Any], axis: str | None) -> dict[str, Any] | None:
    return panel.get(axis) if axis in ("x", "y") and isinstance(panel.get(axis), dict) else None


def _value_axis_ok(panel: dict[str, Any], axis: str, series: dict[str, Any] | None = None) -> bool:
    """Values on the y axis of an ``other`` quantity are not claimed (the label is neutral). A series FI found not to be the
    table's value of its result carries ``what: other`` itself and states no value, whatever the panel's axis says."""
    ax = _axis(panel, axis)
    return ax is not None and not (axis == "y" and (ax.get("what") == "other" or (series or {}).get("what") == "other"
                                                    or (series or {}).get("kind") == "curve"))


def _x_ok(series: dict[str, Any]) -> bool:
    """Whether a series' drawn x values are values of the x setting in FI's table (``x_checked`` of the call record). A claim that
    reads x (a trend along it, an order at an x, a crossing, a span of x, a missing x, a step) is offered only then; a curve's x
    is never FI's. A series with no flag reads as unchecked (fail closed)."""
    return series.get("kind") != "curve" and series.get("x_checked") is True


def _x_arg(raw: str | None) -> float | None:
    try:
        v = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --- the evaluators -----------------------------------------------------------------------------------------------------
# Each takes (calls, panel, args, fmt, ctx) with the args already checked against the claim's argument names and returns
# (sentence, extra normalised args) or None. ``ctx`` carries the optional plan and recorder files.

def _drawn(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None:
        return None
    if s.get("kind") == "image":
        if not _image_values(s):
            return None
    elif not _pairs(s):
        return None
    return f"{_label(s)} is drawn{_where(calls, panel)}", {}


def _image_values(s: dict[str, Any]) -> list[float]:
    """The finite values of an image series (its ``z`` rows: the colour quantity)."""
    z = s.get("z")
    return [float(v) for row in z if isinstance(row, list) for v in row if _finite(v)] if isinstance(z, list) else []


def _not_drawn(calls, panel, a, fmt, ctx):
    plan_ids = _plan_series(ctx.get("plan"))
    if a["series"] not in plan_ids:
        return None
    # A raw panel may draw anything, so absence cannot be stated while one exists. An id another series absorbed
    # (``merged_with``: identical data drawn once) is drawn. An unreadable ``merged_with`` states nothing.
    for p in calls.get("panels") or []:
        if not isinstance(p, dict) or p.get("raw"):
            return None
        for s in p.get("series") or []:
            if not isinstance(s, dict):
                return None
            merged = s.get("merged_with")
            if s.get("id") == a["series"] or (merged is not None and (
                    not isinstance(merged, list) or a["series"] in merged)):
                return None
    return f"{plan_ids[a['series']] or a['series']} is not drawn in this figure", {}


def _range(calls, panel, a, fmt, ctx):
    axis = a.get("axis", "y")
    if not _value_axis_ok(panel, axis):
        return None
    if "series" in a:
        s = _series(panel, a["series"])
        chosen = [s] if s is not None and s.get("kind") != "curve" else []  # a curve's numbers are not FI's
        who = _label(s) if s is not None else ""
    else:
        # an image's rows and columns are positions, not a quantity: only a named image gives its colour values; a curve's
        # numbers are the body's own, so it is never part of "the series"
        chosen = [s for s in panel.get("series") or [] if isinstance(s, dict) and s.get("kind") not in ("image", "curve")]
        who = "the series"
    if axis == "y" and any(not _value_axis_ok(panel, "y", s) for s in chosen):
        return None  # one series whose y is not the table's value: the span of "the series" would state it
    if axis == "x" and any(not _x_ok(s) for s in chosen):
        return None  # one series on an x FI did not check
    if not chosen:
        return None
    if chosen[0].get("kind") == "image":
        values = _image_values(chosen[0]) if axis == "y" else []
        if not values:
            return None
        return f"{who} has values from {fmt(min(values))} to {fmt(max(values))}{_where(calls, panel)}", {"axis": "y"}
    if axis == "x" and any(_is_cat(s) for s in chosen):
        return None
    values = [p[0 if axis == "x" else 1] for s in chosen for p in (_points(s) if axis == "x" else _pairs(s))]
    if not values:
        return None
    lo, hi = min(values), max(values)
    return (f"{who} spans {fmt(lo)} to {fmt(hi)} on the {axis} axis{_where(calls, panel)}",
            {"axis": axis})


def _value_at(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not _value_axis_ok(panel, "y", s) or not _x_ok(s):
        return None
    got = _at(s, a.get("x", ""))
    if len({y for _, y in got}) != 1:  # not a drawn x (no interpolation), or two different y at one x
        return None
    x, y = got[0]
    return (f"{_label(s)} is {fmt(y)} at {_axis_name(panel, 'x')} {_xtext(x, fmt)}{_where(calls, panel)}",
            {"x": _xarg(x)})


def _axis_name(panel: dict[str, Any], axis: str) -> str:
    ax = _axis(panel, axis) or {}
    return str(ax.get("label") or axis)


def _sorted_unique(s: dict[str, Any]) -> list[tuple[float, float]] | None:
    """The points in x order, or ``None`` when one x holds two points (the order of the line there is not readable)."""
    pts = sorted(_points(s))
    xs = [p[0] for p in pts]
    return pts if all(not _same_x(b, a) for a, b in zip(xs, xs[1:])) else None


def _trend(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or s.get("step") or _axis(panel, "y") is None or not _x_ok(s):
        return None  # a shape claim: no y number, so no checked ``what`` needed; but it reads x, which FI must have checked
    pts = _sorted_unique(s)
    if pts is None or len(pts) < 3:
        return None
    ups = downs = 0
    for (_, y0), (_, y1) in zip(pts, pts[1:]):
        if _tie(y0, y1):
            continue
        ups, downs = ups + (y1 > y0), downs + (y1 < y0)
    if not ups and not downs:
        return None  # flat to the rounding of the values: the catalogue has no word for it
    # A rise (fall) is only stated when the whole drawn run ends higher (lower) than it began, so a chain of steps each
    # inside the rounding cannot hide a movement the other way.
    first, last = pts[0][1], pts[-1][1]
    word = ("increases" if not downs and last > first and not _tie(first, last) else
            "decreases" if not ups and last < first and not _tie(first, last) else "does not change monotonically")
    return f"{_label(s)} {word} over the drawn points{_where(calls, panel)}", {}


def _steps(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not s.get("step") or not _value_axis_ok(panel, "y", s) or not _x_ok(s):
        return None
    pts = _sorted_unique(s)
    if pts is None or len(pts) < 2 or _tie(pts[0][1], pts[-1][1]):
        return None
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    word = "rises" if y1 > y0 else "falls"
    return (f"{_label(s)} {word} from {fmt(y0)} to {fmt(y1)} over {_axis_name(panel, 'x')} {fmt(x0)} to {fmt(x1)}"
            f"{_where(calls, panel)}", {})


def _order(calls, panel, a, fmt, ctx):
    if _axis(panel, "y") is None:  # a shape claim: it names the order, never a y number
        return None
    got: list[tuple[float, Any, dict[str, Any]]] = []
    for s in panel.get("series") or []:
        if not isinstance(s, dict) or not _plain(s):
            continue
        if not _x_ok(s):
            return None  # one series on an x FI did not check: the order at that x is not FI's to state
        at = _at(s, a["at"])
        if len({y for _, y in at}) > 1:
            return None
        if at:
            got.append((at[0][1], at[0][0], s))
    if len(got) < 2:
        return None
    got.sort(key=lambda t: -t[0])
    if any(_tie(p[0], q[0]) for p, q in zip(got, got[1:])):
        return None
    drawn_x = got[0][1]
    names = ", ".join(_label(s) for _, _, s in got)
    return (f"at {_axis_name(panel, 'x')} {_xtext(drawn_x, fmt)} the drawn values, highest first, are {names}"
            f"{_where(calls, panel)}", {"at": _xarg(drawn_x)})


def _crossings(panel: dict[str, Any], a: dict[str, str], same_grid: bool = False):
    """(series a, series b, [(x1, x2), ...]) or None when the pair cannot be compared. ``same_grid`` also requires both
    series to be drawn at the same x points (two lines on different grids can cross between the points they share)."""
    if a["a"] == a["b"] or _axis(panel, "y") is None:  # shape claims: the text names x only
        return None
    sa, sb = _series(panel, a["a"]), _series(panel, a["b"])
    if sa is None or sb is None or not _plain(sa) or not _plain(sb) or not _x_ok(sa) or not _x_ok(sb):
        return None
    ya, yb = dict(_points(sa)), dict(_points(sb))
    if len(ya) != len(_points(sa)) or len(yb) != len(_points(sb)):
        return None
    common = sorted(x for x in ya if x in yb)
    if same_grid and (len(common) != len(ya) or len(common) != len(yb)):
        return None
    if len(common) < 2:
        return None
    diffs = [ya[x] - yb[x] for x in common]
    if any(_tie(ya[x], yb[x]) for x in common):
        return None  # the two touch at a drawn x: the sign there is not readable
    spans = [(x0, x1) for (x0, d0), (x1, d1) in zip(zip(common, diffs), zip(common[1:], diffs[1:])) if (d0 > 0) != (d1 > 0)]
    return sa, sb, spans


def _crossing(calls, panel, a, fmt, ctx):
    got = _crossings(panel, a)
    if got is None or not got[2]:
        return None
    sa, sb, spans = got
    where = " and ".join(f"between {_axis_name(panel, 'x')} {fmt(x0)} and {fmt(x1)}" for x0, x1 in spans)
    return f"{_label(sa)} and {_label(sb)} cross {where}{_where(calls, panel)}", {}


def _no_crossing(calls, panel, a, fmt, ctx):
    got = _crossings(panel, a, same_grid=True)
    if got is None or got[2]:
        return None
    return f"{_label(got[0])} and {_label(got[1])} do not cross{_where(calls, panel)}", {}


def _identical(calls, panel, a, fmt, ctx):
    if a["a"] == a["b"]:
        return None
    sa, sb = _series(panel, a["a"]), _series(panel, a["b"])
    if sa is None and sb is None:
        return None
    if sa is not None and sb is None and a["b"] in (sa.get("merged_with") or []):
        return f"{_label(sa)} and {a['b']} are the same data, drawn once{_where(calls, panel)}", {}
    if sb is not None and sa is None and a["a"] in (sb.get("merged_with") or []):
        return f"{_label(sb)} and {a['a']} are the same data, drawn once{_where(calls, panel)}", {}
    if sa is None or sb is None:
        return None
    pa, pb = _points(sa), _points(sb)
    if not pa or len(pa) != len(pb) or len(pa) != len(sa.get("x") or []) or len(pb) != len(sb.get("x") or []):
        return None
    same = all(abs(p - q) <= IDENTICAL_RTOL * max(abs(p), abs(q)) for ra, rb in zip(pa, pb) for p, q in zip(ra, rb))
    if not same:
        return None
    return f"{_label(sa)} and {_label(sb)} are the same data{_where(calls, panel)}", {}


def _reference(calls, panel, a, fmt, ctx):
    found = [r for r in panel.get("refs") or [] if isinstance(r, dict) and r.get("id") == a["ref"]]
    if len(found) != 1 or not _finite(found[0].get("value")) or found[0].get("axis") not in ("x", "y"):
        return None
    r = found[0]
    if r.get("source") == "stated in the figure":  # a constant the drawing typed: where the line is, and nothing about why
        return f"a line is drawn at {r['axis']} = {fmt(float(r['value']))}{_where(calls, panel)}", {}
    return (f"a reference line{' for ' + _label(r) if r.get('label') else ''} is drawn at "
            f"{r['axis']} = {fmt(float(r['value']))}{_where(calls, panel)}", {})


_MEANING = {"ci95": "95% confidence interval", "sd": "standard deviation", "range": "range"}


def _errorbars(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not _plain(s):
        return None
    meaning = s.get("yerr_meaning")  # the drawing call states it (FI checked it against the table); absent or ``other`` is not a claim
    if meaning not in _MEANING:
        return None
    if isinstance(s.get("yerr_lo"), list) and isinstance(s.get("yerr_hi"), list):
        lo, hi = s["yerr_lo"], s["yerr_hi"]
    elif isinstance(s.get("yerr"), list):  # a record from before the two sides were kept
        lo = hi = s["yerr"]
    else:
        return None
    lo_w = [abs(float(w)) for w in lo if _finite(w)]
    hi_w = [abs(float(w)) for w in hi if _finite(w)]
    if not any(w > 0 for w in lo_w + hi_w):
        return None
    base = f"{_label(s)} is drawn with error bars showing the {_MEANING[meaning]}"
    if lo == hi:
        return f"{base}{_where(calls, panel)}", {}
    # uneven bars: how far they reach below and above the points, at most. These are y numbers: a series FI found not to be the
    # table's value of its result (``what: other``) states none.
    if s.get("what") == "other":
        return None
    return (f"{base}, reaching from -{fmt(max(lo_w, default=0.0))} to +{fmt(max(hi_w, default=0.0))} around the points at most"
            f"{_where(calls, panel)}", {})


def _band(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not _plain(s):
        return None
    found = [b for b in panel.get("bands") or [] if isinstance(b, dict) and b.get("series") == a["series"]]
    if not found or len({b.get("meaning") for b in found}) != 1 or found[0].get("meaning") not in _MEANING:
        return None
    b = found[0]
    lo, hi = b.get("lo"), b.get("hi")
    if not isinstance(lo, list) or not isinstance(hi, list) or not any(_finite(v) for v in lo + hi):
        return None
    return f"a band showing the {_MEANING[b['meaning']]} is drawn around {_label(s)}{_where(calls, panel)}", {}


def _missing_x(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    grid = _plan_grid(ctx.get("plan"))
    plan = ctx.get("plan")
    grid_id = plan.get("grid_id") if isinstance(plan, dict) else None
    if grid_id is not None and (panel.get("x") or {}).get("id") != grid_id:
        return None  # the grid is another setting's values
    if s is None or s.get("kind") != "points" or not grid or _is_cat(s) or not _x_ok(s):
        return None
    drawn = [x for x, _ in _points(s)]
    gone = [g for g in grid if not any(_same_x(g, x) for x in drawn)]
    if not gone:
        return None
    return (f"{_label(s)} has no point at {_axis_name(panel, 'x')} {', '.join(fmt(g) for g in gone)}"
            f"{_where(calls, panel)}", {})


def _scale(calls, panel, a, fmt, ctx):
    ax = _axis(panel, a["axis"])
    if ax is None or ax.get("scale") not in ("linear", "log"):
        return None
    word = "logarithmic" if ax["scale"] == "log" else "linear"
    return f"the {a['axis']} axis is {word}{_where(calls, panel)}", {}


def _layout(calls, panel, a, fmt, ctx):
    return _sentence_of(figure_facts.layout_sentence, ctx.get("record"), ctx.get("lines_data"))


def _lines(calls, panel, a, fmt, ctx):
    return _sentence_of(figure_facts.lines_sentence, ctx.get("lines_data"))


def _sentence_of(fn, *given):
    if given[0] is None:
        return None
    text = fn(*given)
    return (text, {}) if text else None


def _note(calls, panel, a, fmt, ctx):
    found = [n for n in panel.get("notes") or [] if isinstance(n, dict) and n.get("id") == a["note"]]
    text = str(found[0].get("text") or "").strip() if len(found) == 1 else ""
    return (text, {}) if text else None


def _plain(series: dict[str, Any]) -> bool:
    """A series other than a ``curve`` or an ``image``, which support only ``drawn`` and ``range``."""
    return series.get("kind") in ("line", "points", "bars")


def _plan_series(plan: Any) -> dict[str, str]:
    """``{id: label}`` of the series a plan declares (``plan["series"]``: ids, or dicts with ``id`` and ``label``)."""
    out: dict[str, str] = {}
    items = plan.get("series") if isinstance(plan, dict) else None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            out[item] = ""
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            out[item["id"]] = str(item.get("label") or "")
    return out


def _plan_grid(plan: Any) -> list[float]:
    """The x values the plan declares for its settings (``plan["grid"]``), or ``[]`` when it declares none."""
    grid = plan.get("grid") if isinstance(plan, dict) else None
    return [float(g) for g in grid if _finite(g)] if isinstance(grid, list) else []


def plan_of(protocol: Any) -> dict[str, Any] | None:
    """The plan the claims read (``{"series": [{"id": ...}], "grid": [...], "grid_id": ...}``) from the study's protocol:
    ``series`` are the metric ids it declares; ``grid`` is the list of values of the one numeric setting it varies
    (``grid_id`` names it), left out when it varies none or more than one, so that no claim reads a grid it cannot place.
    ``None`` without a protocol."""
    if not isinstance(protocol, dict) or not protocol:
        return None
    metrics = protocol.get("metrics")
    out: dict[str, Any] = {"series": [{"id": str(m["id"]), "label": ""} for m in metrics
                                      if isinstance(m, dict) and m.get("id")] if isinstance(metrics, list) else []}
    grid = protocol.get("grid")
    if isinstance(grid, dict) and len(grid) == 1:
        (name, values), = grid.items()
        if isinstance(values, list) and values and all(_finite(v) for v in values):
            out["grid"], out["grid_id"] = [float(v) for v in values], str(name)
    return out


@dataclass(frozen=True)
class ClaimSpec:
    """One claim: the argument names it takes, the function that evaluates it, and whether it needs a panel and series."""
    id: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    fn: Callable[..., tuple[str, dict[str, str]] | None]
    on_curve: bool = False   # also evaluable on a ``curve`` or ``image`` series
    figure_level: bool = False  # about the whole figure: no panel argument


def _spec(cid, fn, required=(), optional=(), **kw) -> ClaimSpec:
    return ClaimSpec(cid, tuple(required), tuple(optional), fn, **kw)


CLAIMS: dict[str, ClaimSpec] = {s.id: s for s in (
    _spec("drawn", _drawn, ["series"], ["panel"], on_curve=True),
    _spec("not_drawn", _not_drawn, ["series"], [], figure_level=True),
    _spec("range", _range, [], ["axis", "series", "panel"], on_curve=True),
    _spec("value_at", _value_at, ["series", "x"], ["panel"]),
    _spec("trend", _trend, ["series"], ["panel"]),
    _spec("order", _order, ["at"], ["panel"]),
    _spec("crossing", _crossing, ["a", "b"], ["panel"]),
    _spec("no_crossing", _no_crossing, ["a", "b"], ["panel"]),
    _spec("identical", _identical, ["a", "b"], ["panel"]),
    _spec("reference", _reference, ["ref"], ["panel"]),
    _spec("errorbars", _errorbars, ["series"], ["panel"]),
    _spec("band", _band, ["series"], ["panel"]),
    _spec("missing_x", _missing_x, ["series"], ["panel"]),
    _spec("scale", _scale, ["axis"], ["panel"]),
    _spec("layout", _layout, [], [], figure_level=True),
    _spec("lines", _lines, [], [], figure_level=True),
    _spec("note", _note, ["note"], ["panel"]),
    _spec("steps", _steps, ["series"], ["panel"]),
)}


def evaluate(calls: Any, claim_id: str, args: dict[str, str], fmt: Fmt = _default_fmt, *, plan: Any = None,
             record: Any = None, lines_data: Any = None) -> tuple[str, dict[str, str]] | None:
    """FI's sentence for claim ``claim_id`` with ``args`` on the call record, with the arguments as evaluated (the panel
    always numbered, an x as the drawn x); ``None`` when the claim cannot be stated: unknown id, an argument the claim does
    not take or lacks, a series, panel or x the record does not hold, a raw panel, no record, a claim on a curve other than
    ``drawn`` or ``range``, or a claim that is not true of the drawn points. Never raises on a malformed record.

    ``plan`` (``{"series": [...], "grid": [...]}``) is read by ``not_drawn`` and ``missing_x``; ``record`` and
    ``lines_data`` are the recorder's files of the figure (see :mod:`core.figure_facts`), read by ``layout`` and ``lines``."""
    spec = CLAIMS.get(claim_id) if isinstance(claim_id, str) else None
    if spec is None or not isinstance(args, dict) or not isinstance(calls, dict):
        return None
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in args.items()):
        return None
    allowed = set(spec.required) | set(spec.optional)
    if set(args) - allowed or any(r not in args for r in spec.required):
        return None
    try:
        if spec.figure_level:
            got = spec.fn(calls, {}, dict(args), fmt, {"plan": plan, "record": record, "lines_data": lines_data})
            return (got[0], dict(args)) if got else None
        panel = _panel(calls, args)
        if panel is None:
            return None
        if not spec.on_curve:
            for key in ("series", "a", "b"):
                s = _series(panel, args.get(key)) if key in args else None
                if s is not None and not _plain(s):
                    return None
        got = spec.fn(calls, panel, dict(args), fmt, {"plan": plan})
        if got is None or not isinstance(got[0], str) or not got[0]:
            return None
        return got[0], _norm(panel, args, **got[1])
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, ArithmeticError):
        return None


# --- the menu ------------------------------------------------------------------------------------------------------------

def _safe(fn, *args) -> list[tuple[str, dict[str, str]]]:
    """The candidates one unit of the record gives, or none when that unit is malformed: the menu never raises."""
    try:
        return list(fn(*args))
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, ArithmeticError):
        return []


def _series_candidates(s: dict[str, Any], p: dict[str, str]):
    sid = {"series": s["id"], **p}
    yield "drawn", sid
    for axis in ("x", "y"):
        yield "range", {**sid, "axis": axis}
    xs = list(dict.fromkeys(x for x, _ in _pairs(s))) if _plain(s) else []
    xs = xs if _is_cat(s) else sorted(xs)
    for x in (xs if len(xs) <= _MAX_VALUE_AT else [xs[0], xs[-1]]):
        yield "value_at", {**sid, "x": _xarg(x)}
    for cid in ("trend", "steps", "errorbars", "band", "missing_x"):
        yield cid, sid


def _shared_candidates(series: list[dict[str, Any]], p: dict[str, str]):
    shared: list[list[Any]] = []   # [x, number of series drawn there]
    for s in series:
        if not _plain(s):
            continue
        for x in dict.fromkeys(x for x, _ in _pairs(s)):
            hit = next((e for e in shared if (isinstance(x, str) and e[0] == x) or (
                not isinstance(x, str) and not isinstance(e[0], str) and _same_x(e[0], x))), None)
            if hit is None:
                shared.append([x, 1])
            else:
                hit[1] += 1
    common = [e[0] for e in shared if e[1] >= 2]
    if common and not any(isinstance(x, str) for x in common):
        common.sort()
    for x in dict.fromkeys([common[0], common[len(common) // 2], common[-1]] if common else []):
        yield "order", {"at": _xarg(x), **p}


def _pair_candidates(series: list[dict[str, Any]], p: dict[str, str]):
    for i, sa in enumerate(series):
        for sb in series[i + 1:]:
            for cid in ("crossing", "no_crossing", "identical"):
                yield cid, {"a": sa["id"], "b": sb["id"], **p}
        merged = sa.get("merged_with")
        for other in merged if isinstance(merged, list) else []:
            yield "identical", {"a": sa["id"], "b": str(other), **p}


def _extras_candidates(panel: dict[str, Any], p: dict[str, str]):
    for r in panel.get("refs") or []:
        if isinstance(r, dict) and isinstance(r.get("id"), str):
            yield "reference", {"ref": r["id"], **p}
    for n in panel.get("notes") or []:
        if isinstance(n, dict) and isinstance(n.get("id"), str):
            yield "note", {"note": n["id"], **p}
    for axis in ("x", "y"):
        yield "scale", {"axis": axis, **p}


def _candidates(calls: dict[str, Any], plan: Any, record: Any, lines_data: Any):
    """(claim id, args) to try, over the non-raw panels. The menu keeps only those that evaluate."""
    if record is not None:
        yield "layout", {}
    if lines_data is not None:
        yield "lines", {}
    for sid in _safe(lambda: [("not_drawn", {"series": k}) for k in _plan_series(plan)]):
        yield sid
    for panel in calls.get("panels") or []:
        if not isinstance(panel, dict) or panel.get("raw"):
            continue
        p = {"panel": str(panel.get("panel"))}
        listed = panel.get("series")
        series = [s for s in listed if isinstance(s, dict) and isinstance(s.get("id"), str)] \
            if isinstance(listed, list) else []
        for s in series:
            yield from _safe(_series_candidates, s, p)
        yield from _safe(_shared_candidates, series, p)
        yield from _safe(_pair_candidates, series, p)
        yield from _safe(_extras_candidates, panel, p)


def _mark(stem: str, claim: str, args: dict[str, str]) -> str:
    tail = format_args(args)
    return "{{figure:%s:%s%s}}" % (stem, claim, f" @ {tail}" if tail else "")


def menu(calls: Any, plan: dict[str, Any] | None = None, fmt: Fmt = _default_fmt, *, stem: str = "<stem>",
         record: Any = None, lines_data: Any = None) -> list[Offer]:
    """The claims that are true on the call record, as ``{"claim", "args", "mark", "text"}`` in catalogue order. Every offer
    is one :func:`evaluate` returns the same sentence for. ``stem`` names the figure in the mark; ``record`` and
    ``lines_data`` (the recorder's files) switch on ``layout`` and ``lines``."""
    if not isinstance(calls, dict) or not isinstance(calls.get("panels"), list):
        return []
    tried: set[tuple[str, tuple]] = set()
    found: list[Offer] = []
    try:
        candidates = list(_candidates(calls, plan, record, lines_data))
    except (TypeError, ValueError, KeyError, AttributeError, IndexError, ArithmeticError):
        return []
    for cid, args in candidates:
        key = (cid, tuple(sorted(args.items())))
        if key in tried:
            continue
        tried.add(key)
        spec = CLAIMS[cid]
        args = {k: v for k, v in args.items() if k in spec.required or k in spec.optional}
        got = evaluate(calls, cid, args, fmt, plan=plan, record=record, lines_data=lines_data)
        if got is not None:
            found.append({"claim": cid, "args": got[1], "mark": _mark(stem, cid, got[1]), "text": got[0]})
    order = {cid: i for i, cid in enumerate(CLAIMS)}
    found.sort(key=lambda o: order[o["claim"]])
    return found


# --- the pointer rule ------------------------------------------------------------------------------------------------

#: A closing "(Figure N)" is a pointer by position: an opening bracket before the mention, the closing one after it.
_OPEN_PAREN = re.compile(r"\(\s*$")
_CLOSE_PAREN = re.compile(r"^\s*\)[\s.,;:!?*]*$")
_TOKEN_AT = re.compile(r"\x00(\d+)\x00\.?")
_OWN_POINTER = re.compile(r"(?:\*\*)?[ \t]*\([ \t]*")
#: What joins one more figure number to a pointer: "(Figures 2 and 3)", "(Figures 2, 3)", "(Figures 2-4)".
_LIST_JOIN = re.compile(r"\s*(?:,\s*(?:and|&)|,|and|&|[-\u2013])\s*", re.IGNORECASE)
#: No paper has this many figures: a written range wider than this still covers every figure FI could have drawn.
_MAX_RANGE = 1000
_BARE_NUMBER = re.compile(r"\d+")
#: An image embed is its own span, never part of a sentence: it is masked with a \x01 token (FI's marks use \x00).
#: ``![alt](target "title")``: the alt may hold one level of brackets and the target one level of parentheses, as Markdown reads it.
_IMAGE_EMBED = re.compile(r"!\[(?:[^\[\]\n]|\[[^\[\]\n]*\])*\]\((?:[^()\n]|\([^()\n]*\))*\)")
_IMAGE_TOKEN = re.compile(r"\x01\d+\x01")
_PERIOD_SPACE = re.compile(r"\. |[.!?]\*\*(?= )")
#: A bold span that opens a sentence.
_LEAD_BOLD = re.compile(r"\A\s*\*\*([^*\n]+)\*\*")
#: A full stop after one of these (or after a single letter, an initial) does not end a sentence.
_ABBREVIATIONS = frozenset({"e.g", "i.e", "al", "fig", "figs", "cf", "vs", "eq", "eqs", "sec", "ref", "refs", "approx"})


def _listed(text: str, first: re.Match[str], mention_re: re.Pattern[str]) -> tuple[list[int], int]:
    """``(the figure numbers a mention opens, where the list ends)``: "Figure 2" gives ``[2]``, "Figures 2 and 3" gives
    ``[2, 3]``, "Figures 2-4" gives ``[2, 3, 4]``."""
    nums = [int(first.group(1))]
    pos = first.end()
    while True:
        join = _LIST_JOIN.match(text, pos)
        more = None
        if join:
            more = mention_re.match(text, join.end()) or _BARE_NUMBER.match(text, join.end())
        if more is None:
            break
        n = int(more.group(1) if more.re is mention_re else more.group(0))
        if join.group(0).strip() in ("-", "\u2013") and n != nums[-1]:
            lo, hi = sorted((nums[-1], n))
            nums += [k for k in range(lo, min(hi, lo + _MAX_RANGE) + 1) if k != nums[-1]]
        else:
            nums.append(n)
        pos = more.end()
    return nums, pos


def _pointer(text: str, mention_re: re.Pattern[str]) -> tuple[int, list[int]] | None:
    """``(position of the opening bracket, figure numbers listed)`` when every figure named in ``text`` stands in one closing
    parenthetical "(Figure N)", "(Figures N and M)", "(Fig. N and Fig. M)" at the end of the sentence, else ``None``."""
    first = mention_re.search(text)
    if first is None or not _OPEN_PAREN.search(text[:first.start()]):
        return None
    nums, pos = _listed(text, first, mention_re)
    if not _CLOSE_PAREN.match(text[pos:]) or any(m.start() >= pos for m in mention_re.finditer(text)):
        return None
    return _OPEN_PAREN.search(text[:first.start()]).start(), nums


def _sentence_break(flat: str, i: int) -> bool:
    """Whether the ". " at ``i`` ends a sentence: always, except after a single letter (an initial) or a common abbreviation
    ("e.g.", "et al.", "Fig."). The next character is never looked at."""
    if flat[i - 1:i] == "\x00":
        return True
    word = re.search(r"[A-Za-z.]+$", flat[:i])
    if word is None:
        return True
    word = word.group(0).lower().lstrip(".")
    return len(word) > 1 and word not in _ABBREVIATIONS


def _sentence_at(flat: str, line_start: int, line_end: int, pos: int) -> tuple[int, int]:
    """The running sentence of the line ``flat[line_start:line_end]`` that holds ``pos``: ``(start, end)``."""
    a = line_start
    breaks: list[tuple[int, int]] = []  # (where a sentence ends, where the next one starts)
    for m in _PERIOD_SPACE.finditer(flat, line_start, line_end):
        if _sentence_break(flat, m.start()):
            end = m.end() if m.group(0).endswith("**") else m.start() + 1  # ".** ": ends after the closing "**"
            breaks.append((end, end))
    for m in _IMAGE_TOKEN.finditer(flat, line_start, line_end):  # an image embed stands between two sentences
        breaks.append((m.start(), m.end()))
    for end, nxt in sorted(breaks):
        if nxt <= pos:
            a = nxt
        else:
            return a, end
    return a, line_end


def _lead_heading(sentence: str, mention_re: re.Pattern[str]) -> int:
    """How far a leading bold span reaches when it is a heading, else 0. It is a heading when, with its ``**`` and trailing
    punctuation taken off, it names no figure or is nothing but a figure label ("Figure 1", "Figures 1 and 2"). Any other
    leading bold span is part of the sentence. FI's own statements (a mark inside the span) are not read here."""
    lead = _LEAD_BOLD.match(sentence)
    if lead is None or "\x00" in lead.group(1):
        return 0
    raw = lead.group(1).strip()
    after = sentence[lead.end():lead.end() + 1]
    if not (raw.endswith((".", ":")) or after in (".", ":")):
        return 0  # no label closing mark: "**Figure 1** shows ..." is a sentence with a bold word, not a heading
    reach = lead.end() + (1 if after in (".", ":") and not raw.endswith((".", ":")) else 0)
    inner = raw.rstrip(".:;,!? \t")
    if mention_re.search(inner) is None:
        return reach
    left = mention_re.sub(" ", inner)
    left = _LIST_JOIN.sub(" ", _BARE_NUMBER.sub(" ", left))
    return reach if not re.sub(r"[\s.,;:&\-\u2013]+", "", left) else 0


def _unfilled_entries(marks: list[re.Match[str]], which: list[int], sentence: str) -> list[dict[str, Any]]:
    """The result keys FI could not fill in a sentence the figure pass removed, listed as the take-out of unfilled keys lists
    them (``key``, ``sentence``, ``how``), so that the number removed with the sentence is still on record."""
    return [{"key": marks[i].group(1).strip(), "sentence": sentence, "how": "sentence removed", "at_write": True}
            for i in which]


def apply_pointer_rule(markdown: str, numbers: dict[str, int], drawn: dict[int, set[str]],
                       mention_re: re.Pattern[str]) -> tuple[str, list[dict[str, str]]]:
    """``markdown`` without the running sentences that point at a figure and are not FI's to stand behind:
    ``(text, taken_out)``, ``taken_out`` having ``key`` (the figure the sentence named), ``sentence`` (as it read) and
    ``how`` for each, like :func:`core.paper_facts.take_out_unfilled`.

    By position and marks only, no wording. A running sentence that names a figure (``mention_re``) is kept when every
    figure it names is one FI issued (``numbers``) and either none of them has a call record (``drawn`` lacks its number: an
    earlier version drew it, nothing is checked), or the sentence is a figure claim FI composed, or it reports results
    (FI value marks) and points at one figure in a closing parenthesis, "(Figure N)", whose record draws (``drawn[N]``) the
    metric of at least one of those marks (one figure, or several listed in it: "(Figures 2 and 3)", then the metric need only
    be drawn in one of them, and every one listed must be a figure FI issued). Every other sentence naming a figure is
    removed. A sentence ends at ". " except after an initial or a common abbreviation
    ("e.g.", "et al.", "Fig."). Like :func:`core.paper_facts.take_out_unfilled`, a line is read alone (a sentence wrapped over
    two lines is two) and captions, table rows, headings, list items, quotes, code and the text inside FI's own markers are
    never touched."""
    from . import paper_facts as _facts  # late: paper_facts may itself reach this module

    marks = [m for m in _values._MARK.finditer(markdown or "")]
    marks += [m for m in _facts._MARK.finditer(markdown or "")]
    marks += [m for m in _values._SPAN.finditer(markdown or "")]
    marks += [m for m in _IMAGE_EMBED.finditer(markdown or "")]
    marks.sort(key=lambda m: m.start())
    kept: list[re.Match[str]] = []  # the outermost marks: one inside another is part of it
    for m in marks:
        if not kept or m.start() >= kept[-1].end():
            kept.append(m)
    marks = kept
    if not marks and not mention_re.search(markdown or ""):
        return markdown, []
    parts: list[str] = []
    spans: list[tuple[int, int, int, int]] = []  # (token start, end, original start, end)
    kinds: list[str] = []  # "v" | "claim" | "other"
    last = size = 0
    for i, m in enumerate(marks):
        before = markdown[last:m.start()]
        claim = m.re is _facts._MARK and m.group(1).startswith("figure-")
        # FI's sentence about a figure ends with its own "(Figure N)": the claim and that pointer are one sentence.
        opening = _OWN_POINTER.match(markdown, m.end()) if claim else None
        own_pointer = opening is not None and mention_re.match(markdown, opening.end()) is not None
        # A statement of its own has its full stop; one the writer's words lead into is one sentence with them.
        full_stop = claim and not own_pointer and _facts._starts_sentence(markdown[:m.start()]) \
            and markdown[m.end():m.end() + 1] not in (".", "!", "?") \
            and re.match(r"[ \t]*[a-z,;]", markdown[m.end():m.end() + 8]) is None
        is_image = m.re is _IMAGE_EMBED
        token = f"\x01{i}\x01" if is_image else f"\x00{i}\x00" + ("." if full_stop else "")
        parts += [before, token]
        spans.append((size + len(before), size + len(before) + len(token), m.start(), m.end()))
        kinds.append("v" if m.re is _values._MARK else "claim" if claim else "image" if is_image else "other")
        size += len(before) + len(token)
        last = m.end()
    parts.append(markdown[last:])
    flat = "".join(parts)
    issued = set(numbers.values())

    def original(p: int) -> int:
        shift = 0
        for _bs, be, _os, oe in spans:
            if be <= p:
                shift = oe - be
        return p + shift

    def metrics(a: int, b: int) -> set[str]:
        found: set[str] = set()
        for i, sp in enumerate(spans):
            if kinds[i] != "v" or not (a <= sp[0] and sp[1] <= b):
                continue
            if marks[i].group(2).strip() == _values.NOT_RECORDED:
                continue
            try:
                found.add(_values.parse(marks[i].group(1)).metric)
            except Exception:  # noqa: BLE001 -- a key that does not parse reports nothing
                pass
        return found

    def keeps(a: int, b: int, hits: list[re.Match[str]]) -> bool:
        text = flat[a:b]
        point = _pointer(text, mention_re)
        named = sorted({n for h in hits for n in _listed(flat, h, mention_re)[0]})  # "Figures 1 and 2" names both
        nums = sorted(set(point[1] if point else []) | set(named))
        if all(n not in drawn for n in nums):
            return True  # only figures with no call record (or none FI numbered): nothing is checked, as before
        if point and any(n in drawn and n not in point[1] for n in named):
            return False  # a checked figure named outside the closing pointer (a run-in label): the pointer does not cover it
        if any(n not in issued for n in nums):
            return False
        inside = [i for i, sp in enumerate(spans) if a <= sp[0] and sp[1] <= b]
        rest = text
        if point:
            rest = text[:point[0]] + text[text.index(")", point[0]) + 1:]
        if any(kinds[i] == "claim" for i in inside) \
                and not re.sub(r"[\s.,;:!?()*\-\u2013\u2014]+", "", _TOKEN_AT.sub("", rest)):
            return True  # a figure claim FI composed, with at most its pointer
        shown = set().union(*(drawn.get(n, set()) for n in nums))
        return point is not None and bool(metrics(a, b) & shown)

    edits: list[tuple[int, int, str]] = []
    taken: list[dict[str, str]] = []
    seen: set[tuple[int, int]] = set()
    fence: str | None = None
    at = 0
    for line in flat.split("\n"):
        line_start, at = at, at + len(line) + 1
        fence_line = re.match(r"^\s*(```|~~~)", line)
        if fence_line:
            fence = None if fence == fence_line.group(1) else (fence or fence_line.group(1))
            continue
        if fence is not None or _facts._NOT_PROSE.match(line):
            continue
        if "![" in line:  # every embed that parses is masked above: a "![" left is one that does not, and its line is left as it is
            continue
        at_claims = [sp[0] - line_start for i, sp in enumerate(spans) if kinds[i] == "claim"
                     and line_start <= sp[0] < line_start + len(line)]
        for pos in sorted({f.start() for f in mention_re.finditer(line)} | set(at_claims)):
            a, b = _sentence_at(flat, line_start, line_start + len(line), line_start + pos)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            lead = _facts._RUN_IN.match(flat[a:b])
            heading = _lead_heading(flat[a:b], mention_re)
            label_at = None  # where a run-in figure label starts, when the sentence after it is judged as naming that figure
            label_hits: list[re.Match[str]] = []
            if heading:  # a bare label ("**Results.**", "**Figure 1.**"): neither judged nor cut on its own
                label_hits = [h for h in mention_re.finditer(flat, a, a + heading)]
                label_start = a
                a += heading
                if not flat[a:b].strip():
                    # a figure label with prose after it on its line: that prose is a sentence that names the figure. A label
                    # alone on its line (nothing after it before the line ends, or an image) stays a heading
                    nxt = b
                    while nxt < line_start + len(line) and flat[nxt] in " \t":
                        nxt += 1
                    if not label_hits or nxt >= line_start + len(line):
                        continue
                    na, nb = _sentence_at(flat, line_start, line_start + len(line), nxt)
                    if not flat[na:nb].strip() or flat[nxt] == "\x01":
                        continue
                    a, b = na, nb
                    seen.add((a, b))
                if label_hits:
                    label_at = label_start
                lead = None
                seen.add((a, b))
            else:
                # any other bold span (a result mark or FI's own statement in it too) is part of the sentence: judged and cut
                # with it, so FI's statement inside it is kept by the rule below and only the writer's words around it go
                lead = None
            hits = label_hits + [h for h in mention_re.finditer(flat, a, b)]
            body_at = a + (lead.end() if lead else 0)
            own = [i for i, sp in enumerate(spans) if kinds[i] == "claim" and body_at <= sp[0] and sp[1] <= b]
            if own:
                # A sentence with a statement of FI's is judged whether or not it names a figure, and before any result
                # mark can keep it: what is not FI's statement (and its closing pointer) is the writer's, and goes.
                rest = flat[body_at:b]
                for i in reversed(own):
                    rest = rest[:spans[i][0] - body_at] + rest[spans[i][1] - body_at:]
                point = _pointer(rest, mention_re)
                if point:
                    rest = rest[:point[0]] + rest[rest.index(")", point[0]) + 1:]
                if not re.sub(r"[\s.,;:!?()*\-\u2013\u2014]+", "", rest):
                    continue
            elif not hits or keeps(a, b, hits):
                continue
            sentence = re.sub(r"\s+", " ", _values.without_marks(markdown[original(a):original(b)])).strip()
            figures = sorted({n for h in hits for n in _listed(flat, h, mention_re)[0]})
            unfilled = [i for i, sp in enumerate(spans) if kinds[i] == "v" and a <= sp[0] and sp[1] <= b
                        and marks[i].group(2).strip() == _values.NOT_RECORDED]
            if own:  # FI's own statement is never removed: only the words the writer put into its sentence go
                said = " ".join(markdown[spans[i][2]:spans[i][3]] for i in own)
                stems = {Path(str(n)).stem: k for n, k in numbers.items()}
                shown_in = sorted({stems[parts[0]] for i in own
                                   if (parts := _facts._figure_parts(marks[i].group(1))) and parts[0] in stems})
                pointer = ("" if not shown_in else " (Figure %d)" % shown_in[0] if len(shown_in) == 1
                           else " (Figures %s and %d)" % (", ".join(map(str, shown_in[:-1])), shown_in[-1]))
                gap = re.match(r"\s*", flat[body_at:b]).group(0)
                seg_at = original(body_at)
                seg = markdown[seg_at:original(b)]
                for i in reversed(own):
                    seg = seg[:spans[i][2] - seg_at] + seg[spans[i][3] - seg_at:]
                sentence = re.sub(r"\s+", " ", _values.without_marks(seg)).strip()
                stop = "." if any(kinds[i] == "claim" and spans[i][3] == original(body_at) for i in range(len(spans))) else ""
                edits.append((original(body_at), original(b), stop + gap + said + pointer + "."))
                figures = sorted(set(figures) | set(shown_in))
                taken.append({"key": f"Figure {figures[0] if figures else '?'}", "sentence": sentence,
                              "how": "writer's words removed", "figures": figures})
                taken += _unfilled_entries(marks, unfilled, sentence)
                continue
            if label_at is not None:  # the figure label goes with the sentence it led: no label is left alone
                body_at = label_at
            start, end = original(body_at), original(b)
            if flat[line_start:body_at].strip() == "":  # it opens its line: the space it leaves goes with it
                while end < len(markdown) and markdown[end] in " \t":
                    end += 1
                if not lead and (end >= len(markdown) or markdown[end] == "\n"):  # nothing is left of the line
                    start, end = original(line_start), min(end + 1, len(markdown))
            else:  # the space before it goes with it
                while start > 0 and markdown[start - 1] in " \t":
                    start -= 1
            # FI's statement before it had no full stop of its own (the rule reads it as a sentence): it gets one
            stop = "." if any(kinds[i] == "claim" and spans[i][3] == start for i in range(len(spans))) else ""
            edits.append((start, end, stop))
            taken.append({"key": f"Figure {hits[0].group(1)}", "sentence": sentence, "how": "sentence removed",
                          "figures": figures})
            taken += _unfilled_entries(marks, unfilled, sentence)
    out = markdown
    for start, end, text in sorted(edits, reverse=True):
        out = out[:start] + text + out[end:]
    return out, taken


#: The most statements offered to the writer for one figure.
MENU_LIMIT = 40
#: What is cut last is first here: the order statements are kept in when a figure offers more than MENU_LIMIT.
_KEEP_ORDER = ("drawn", "range:x", "value_at", "steps", "range:y", "trend", "order", "crossing", "errorbars", "band", "note")


def cap_menu(offers: list[Offer], calls: Any, limit: int = MENU_LIMIT) -> list[Offer]:
    """At most ``limit`` of ``offers``, in their own order. What is cut is decided by the kind of statement, not by the
    figure: kept first are ``drawn``, the x range, ``value_at``, ``steps``, the y range, ``trend``, ``order``, ``crossing``,
    error bars and bands and notes (a statement of a kind not named there follows them), and ``no_crossing`` goes last,
    the pairs of lines next to each other in the legend before the others. Deterministic."""
    if len(offers) <= limit:
        return offers
    legend: dict[str, list[str]] = {}
    for p in calls.get("panels") or [] if isinstance(calls, dict) else []:
        if isinstance(p, dict):
            legend[str(p.get("panel"))] = [str(s.get("id")) for s in p.get("series") or [] if isinstance(s, dict)]

    def rank(i: int) -> tuple[int, int, int]:
        o = offers[i]
        name = o["claim"] + (":" + str(o["args"].get("axis")) if o["claim"] == "range" else "")
        if o["claim"] == "no_crossing":
            ids = legend.get(str(o["args"].get("panel")), [])
            a, b = o["args"].get("a"), o["args"].get("b")
            gap = abs(ids.index(b) - ids.index(a)) if a in ids and b in ids else len(ids)
            return len(_KEEP_ORDER) + 1, gap, i
        return (_KEEP_ORDER.index(name) if name in _KEEP_ORDER else len(_KEEP_ORDER)), 0, i

    kept = sorted(sorted(range(len(offers)), key=rank)[:limit])
    return [offers[i] for i in kept]
