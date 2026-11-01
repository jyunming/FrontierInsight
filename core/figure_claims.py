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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import figure_facts

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
    if isinstance(data, dict) and isinstance(data.get("panels"), list):
        return data
    return None


def parse_args(text: str) -> dict[str, str]:
    """``"series=a, panel=1"`` as ``{"series": "a", "panel": "1"}``. A piece that is not ``key=value`` (or a key given twice)
    is kept under the key ``""``, a key no claim accepts, so that :func:`evaluate` fails closed instead of ignoring it."""
    out: dict[str, str] = {}
    if not str(text).strip():
        return out
    for piece in str(text).split(","):
        key, sep, value = piece.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value or key in out:
            out[""] = piece.strip()
            continue
        out[key] = value
    return out


# --- reading the record -------------------------------------------------------------------------------------------------

def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _points(series: dict[str, Any]) -> list[tuple[float, float]]:
    """The drawn (x, y) points of a series with both values finite, in drawn order. A series whose x and y differ in length
    has no readable points."""
    xs, ys = series.get("x"), series.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
        return []
    return [(float(a), float(b)) for a, b in zip(xs, ys) if _finite(a) and _finite(b)]


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
    return str(item.get("label") or item.get("id") or "")


def _where(calls: dict[str, Any], panel: dict[str, Any]) -> str:
    return f" in panel {panel.get('panel')}" if len(calls.get("panels") or []) > 1 else ""


def _norm(panel: dict[str, Any], args: dict[str, str], **extra: str) -> dict[str, str]:
    out = {k: v for k, v in args.items() if k != "panel"}
    out.update(extra)
    out["panel"] = str(panel.get("panel"))
    return out


def _axis(panel: dict[str, Any], axis: str | None) -> dict[str, Any] | None:
    return panel.get(axis) if axis in ("x", "y") and isinstance(panel.get(axis), dict) else None


def _value_axis_ok(panel: dict[str, Any], axis: str) -> bool:
    """Values on the y axis of an ``other`` quantity are not claimed (the label is neutral)."""
    ax = _axis(panel, axis)
    return ax is not None and not (axis == "y" and ax.get("what") == "other")


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
    if s is None or not _points(s):
        return None
    return f"{_label(s)} is drawn{_where(calls, panel)}", {}


def _not_drawn(calls, panel, a, fmt, ctx):
    plan_ids = _plan_series(ctx.get("plan"))
    if a["series"] not in plan_ids:
        return None
    # A raw panel may draw anything, so absence cannot be stated while one exists.
    for p in calls.get("panels") or []:
        if not isinstance(p, dict) or p.get("raw") or any(
                isinstance(s, dict) and s.get("id") == a["series"] for s in p.get("series") or []):
            return None
    return f"{plan_ids[a['series']] or a['series']} is not drawn in this figure", {}


def _range(calls, panel, a, fmt, ctx):
    axis = a.get("axis", "y")
    if not _value_axis_ok(panel, axis):
        return None
    if "series" in a:
        s = _series(panel, a["series"])
        chosen = [s] if s is not None else []
        who = _label(s) if s is not None else ""
    else:
        chosen = [s for s in panel.get("series") or [] if isinstance(s, dict)]
        who = "the series"
    values = [p[0 if axis == "x" else 1] for s in chosen for p in _points(s)]
    if not chosen or not values:
        return None
    lo, hi = min(values), max(values)
    return (f"{who} spans {fmt(lo)} to {fmt(hi)} on the {axis} axis{_where(calls, panel)}",
            {"axis": axis})


def _value_at(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    x = _x_arg(a.get("x"))
    if s is None or x is None or not _value_axis_ok(panel, "y"):
        return None
    ys = {y for px, y in _points(s) if _same_x(px, x)}
    if len(ys) != 1:  # not a drawn x (no interpolation), or two different y at one x
        return None
    drawn_x = next(px for px, _ in _points(s) if _same_x(px, x))
    return (f"{_label(s)} is {fmt(next(iter(ys)))} at {_axis_name(panel, 'x')} {fmt(drawn_x)}{_where(calls, panel)}",
            {"x": "%.12g" % drawn_x})


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
    if s is None or s.get("step") or not _value_axis_ok(panel, "y"):
        return None
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
    word = "increases" if not downs else "decreases" if not ups else "does not change monotonically"
    return f"{_label(s)} {word} over the drawn points{_where(calls, panel)}", {}


def _steps(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not s.get("step") or not _value_axis_ok(panel, "y"):
        return None
    pts = _sorted_unique(s)
    if pts is None or len(pts) < 2 or _tie(pts[0][1], pts[-1][1]):
        return None
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    word = "rises" if y1 > y0 else "falls"
    return (f"{_label(s)} {word} from {fmt(y0)} to {fmt(y1)} over {_axis_name(panel, 'x')} {fmt(x0)} to {fmt(x1)}"
            f"{_where(calls, panel)}", {})


def _order(calls, panel, a, fmt, ctx):
    x = _x_arg(a.get("at"))
    if x is None or not _value_axis_ok(panel, "y"):
        return None
    got: list[tuple[float, dict[str, Any]]] = []
    for s in panel.get("series") or []:
        if not isinstance(s, dict) or not _plain(s):
            continue
        ys = {y for px, y in _points(s) if _same_x(px, x)}
        if len(ys) > 1:
            return None
        if ys:
            got.append((next(iter(ys)), s))
    if len(got) < 2:
        return None
    got.sort(key=lambda t: -t[0])
    if any(_tie(p[0], q[0]) for p, q in zip(got, got[1:])):
        return None
    drawn_x = next(px for px, _ in _points(got[0][1]) if _same_x(px, x))
    names = ", ".join(_label(s) for _, s in got)
    return (f"at {_axis_name(panel, 'x')} {fmt(drawn_x)} the drawn values, highest first, are {names}{_where(calls, panel)}",
            {"at": "%.12g" % drawn_x})


def _crossings(panel: dict[str, Any], a: dict[str, str]):
    """(series a, series b, [(x1, x2), ...]) or None when the pair cannot be compared."""
    if a["a"] == a["b"] or not _value_axis_ok(panel, "y"):
        return None
    sa, sb = _series(panel, a["a"]), _series(panel, a["b"])
    if sa is None or sb is None or not _plain(sa) or not _plain(sb):
        return None
    ya, yb = dict(_points(sa)), dict(_points(sb))
    if len(ya) != len(_points(sa)) or len(yb) != len(_points(sb)):
        return None
    common = sorted(x for x in ya if x in yb)
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
    got = _crossings(panel, a)
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
    return (f"a reference line{' for ' + _label(r) if r.get('label') else ''} is drawn at "
            f"{r['axis']} = {fmt(float(r['value']))}{_where(calls, panel)}", {})


_MEANING = {"ci95": "95% confidence interval", "sd": "standard deviation", "range": "range"}


def _errorbars(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    if s is None or not _plain(s) or not isinstance(s.get("yerr"), list):
        return None
    meaning = s.get("yerr_meaning")  # optional in the pinned record; absent means the drawing call stated none
    if meaning is not None and meaning not in _MEANING:
        return None
    widths = [abs(float(w)) for w in s["yerr"] if _finite(w)]
    if not widths or not any(w > 0 for w in widths):
        return None
    what = f" showing the {_MEANING[meaning]}" if meaning else ""
    return f"{_label(s)} is drawn with error bars{what}{_where(calls, panel)}", {}


def _band(calls, panel, a, fmt, ctx):
    found = [b for b in panel.get("bands") or [] if isinstance(b, dict) and b.get("series") == a["series"]]
    if not found or len({b.get("meaning") for b in found}) != 1 or found[0].get("meaning") not in _MEANING:
        return None
    b = found[0]
    lo, hi = b.get("lo"), b.get("hi")
    if not isinstance(lo, list) or not isinstance(hi, list) or not any(_finite(v) for v in lo + hi):
        return None
    s = _series(panel, a["series"])
    name = _label(s) if s is not None else a["series"]
    return f"a band showing the {_MEANING[b['meaning']]} is drawn around {name}{_where(calls, panel)}", {}


def _missing_x(calls, panel, a, fmt, ctx):
    s = _series(panel, a["series"])
    grid = _plan_grid(ctx.get("plan"))
    if s is None or s.get("kind") != "points" or not grid:
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
    for item in (plan.get("series") if isinstance(plan, dict) else None) or []:
        if isinstance(item, str):
            out[item] = ""
        elif isinstance(item, dict) and isinstance(item.get("id"), str):
            out[item["id"]] = str(item.get("label") or "")
    return out


def _plan_grid(plan: Any) -> list[float]:
    """The x values the plan declares for its settings (``plan["grid"]``), or ``[]`` when it declares none."""
    grid = plan.get("grid") if isinstance(plan, dict) else None
    return [float(g) for g in grid if _finite(g)] if isinstance(grid, list) else []


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

def _candidates(calls: dict[str, Any], plan: Any, record: Any, lines_data: Any):
    """(claim id, args) to try, over the non-raw panels. The menu keeps only those that evaluate."""
    if record is not None:
        yield "layout", {}
    if lines_data is not None:
        yield "lines", {}
    for sid in _plan_series(plan):
        yield "not_drawn", {"series": sid}
    for panel in calls.get("panels") or []:
        if not isinstance(panel, dict) or panel.get("raw"):
            continue
        p = {"panel": str(panel.get("panel"))}
        series = [s for s in panel.get("series") or [] if isinstance(s, dict) and isinstance(s.get("id"), str)]
        for s in series:
            sid = {"series": s["id"], **p}
            yield "drawn", sid
            for axis in ("x", "y"):
                yield "range", {**sid, "axis": axis}
            xs = sorted({x for x, _ in _points(s)})
            for x in (xs if len(xs) <= _MAX_VALUE_AT else [xs[0], xs[-1]]):
                yield "value_at", {**sid, "x": "%.12g" % x}
            for cid in ("trend", "steps", "errorbars", "band", "missing_x"):
                yield cid, sid
        shared: dict[float, int] = {}
        for s in series:
            for x in {x for x, _ in _points(s)}:
                key = next((k for k in shared if _same_x(k, x)), x)
                shared[key] = shared.get(key, 0) + 1
        common = sorted(x for x, n in shared.items() if n >= 2)
        for x in {common[0], common[len(common) // 2], common[-1]} if common else ():
            yield "order", {"at": "%.12g" % x, **p}
        for i, sa in enumerate(series):
            for sb in series[i + 1:]:
                for cid in ("crossing", "no_crossing", "identical"):
                    yield cid, {"a": sa["id"], "b": sb["id"], **p}
            for other in sa.get("merged_with") or []:
                yield "identical", {"a": sa["id"], "b": str(other), **p}
        for r in panel.get("refs") or []:
            if isinstance(r, dict) and isinstance(r.get("id"), str):
                yield "reference", {"ref": r["id"], **p}
        for n in panel.get("notes") or []:
            if isinstance(n, dict) and isinstance(n.get("id"), str):
                yield "note", {"note": n["id"], **p}
        for axis in ("x", "y"):
            yield "scale", {"axis": axis, **p}


def _mark(stem: str, claim: str, args: dict[str, str]) -> str:
    tail = ", ".join(f"{k}={v}" for k, v in sorted(args.items(), key=lambda kv: (kv[0] == "panel", kv[0])))
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
    for cid, args in _candidates(calls, plan, record, lines_data):
        key = (cid, tuple(sorted(args.items())))
        if key in tried:
            continue
        tried.add(key)
        spec = CLAIMS[cid]
        args = {k: v for k, v in args.items() if k in spec.required or k in spec.optional}
        got = evaluate(calls, cid, args, fmt, plan=plan, record=record, lines_data=lines_data)
        if got is not None:
            found.append({"claim": cid, "args": got[1], "mark": _mark(stem, cid, got[1]),
                          "text": got[0]})
    order = {cid: i for i, cid in enumerate(CLAIMS)}
    found.sort(key=lambda o: order[o["claim"]])
    return found
