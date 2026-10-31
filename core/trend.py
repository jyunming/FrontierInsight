"""How a result changes along one setting, and how two results compare setting by setting: the words FI writes itself.

A paper says a result "rises", "falls" or "stays flat" along a setting, that an error series is "irregular at the finest
steps", or that a spread is "larger than the reported error". Each of those is a fact about a series FI holds in its own
trial record, so FI computes it (no model is called) and the writer places the key instead of typing the word:

* ``{{trend:<result> over <axis> @ <other>=each|<value>, ...}}`` -- the direction and size of the change of ``<result>``
  from the first to the last value of ``<axis>``, for each setting of the other axes (:func:`trend_text`). Add
  ``| order`` (the axis counts steps or points: finer is larger) or ``| order_size`` (the axis is a step size: finer is
  smaller) for the pairwise orders of an error series. Which way the axis runs is a fact of the record, so FI reads it from
  where the mean error is lower (:func:`_finer_is_larger`); the writer's word is used for "coarsest" and "finest" only when
  the record agrees, and otherwise the pairs are named by where the error is lower, never by a guess.
* ``{{comparison:<result> <form> vs <other result> <form> @ <axis>=each, ...}}`` -- the ratio of one result's form to
  another's inside each setting (:func:`comparison_text`), forms ``mean``, ``sd`` and ``rms`` (a spread against a spread,
  a mean against a mean).

No tolerance is invented: a change is called a rise or a fall only when the 95% interval of last minus first excludes 0,
and otherwise it is "no change beyond its 95% interval". With one trial per setting there is no interval and only the
change itself is worded, and that is said. The interval is the one the rest of the paper's numbers use
(:mod:`core.paper_values`): a difference of the two settings' means, paired by random draw when the settings ran on the
same draws (FI's seed ledger), else a Welch interval from their trials.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import paper_values as _pv
from . import stats as _stats
from . import trial_table as _tt

_Unfilled = _pv._Unfilled
Pick = _pv.Pick

#: Above this many settings of the other axes a trend is told by counts and extremes, not setting by setting.
MAX_SCOPES_LISTED = 6
#: Up to this many neighbouring pairs the order of each is listed.
MAX_ORDERS_LISTED = 8
ORDER, ORDER_SIZE = "order", "order_size"


@dataclass
class _Point:
    x: Any
    mean: float
    sd: float
    n: int
    cells: list[dict[str, Any]]


@dataclass
class _Change:
    """The last-minus-first difference of two settings' means: ``lower``/``upper`` are ``None`` without an interval."""

    diff: float
    lower: float | None
    upper: float | None
    how: str
    #: The change has no spread, so it is exact and has no interval; ``exact`` says why, in words.
    exact: str = ""
    #: Said after the interval when it is over fewer trials than the means printed beside it.
    note: str = ""

    @property
    def word(self) -> str:
        if self.lower is None or self.upper is None:
            return "point"
        if self.exact:
            return "rises" if self.diff > 0 else "falls" if self.diff < 0 else "none"
        if self.lower > 0:
            return "rises"
        if self.upper < 0:
            return "falls"
        return "none"


def _names(rec: _pv.Records, setting: dict[str, Any]) -> str:
    return ", ".join(f"{rec.short(a)} = {v}" for a, v in setting.items())


def _numeric(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) for v in values)


def _split(body: str) -> tuple[str, str, list[str]]:
    """``(head, where text, the parts after |)`` of a key body."""
    segments = [s.strip() for s in body.split("|")]
    head, at, where = segments[0].partition("@")
    if not at:
        raise _Unfilled("no '@ <where>': name every other setting the study varies, each as name=each or name=<value>")
    return head.strip(), where, segments[1:]


def _where_of(rec: _pv.Records, text: str) -> dict[str, Pick]:
    return {rec.real(a): p for a, p in _pv._where(text).items()}


def _need_result(rec: _pv.Records, name: str) -> None:
    if name in rec.left_out:
        note = rec.left_out[name].strip().rstrip(".")
        raise _Unfilled(f"FI left {name} out of the results" + (f": {note}" if note else "")
                        + "; say in the limitations that it was left out, do not report it")
    if name not in rec.metrics():
        raise _Unfilled(f"{name} is not a value the trials recorded (they recorded: {', '.join(rec.metrics())})")


def _no_pooling(rec: _pv.Records, where: dict[str, Pick], what: str, metrics: list[str]) -> None:
    """Refuse =all on a setting the results in question actually vary over (an axis their cells do not carry pools nothing)."""
    for axis, pick in where.items():
        if pick.kind != "all":
            continue
        seen = {_pv._exact(c["cell"][axis]) for m in metrics for c in rec.carrying(m) if axis in c["cell"]}
        if len(seen) > 1:
            raise _Unfilled(f"{rec.short(axis)}=all pools settings that differ in {rec.short(axis)}, so {what} has no single "
                            f"value over them; name each setting: {rec.short(axis)}=each (or one value of it)")


def _scopes(rec: _pv.Records, metrics: list[str], where: dict[str, Pick]) -> list[dict[str, Any]]:
    """The combinations of values of the axes marked =each that the cells carrying all of ``metrics`` hold, in the record's
    order (``[{}]`` when no axis is marked)."""
    each = [a for a, p in where.items() if p.kind == "each"]
    fixed = {a: p.value for a, p in where.items() if p.kind == "value"}
    found: list[dict[str, Any]] = []
    for c in rec.carrying(metrics[0]):
        cell = c["cell"]
        if any(m not in c["metrics"] for m in metrics[1:]):
            continue
        if any(a in cell and not _pv._same(cell[a], v) for a, v in fixed.items()):
            continue
        combo = {a: cell.get(a) for a in each}
        if combo not in found:
            found.append(combo)
    if not found:
        raise _Unfilled(f"no trials of {', '.join(metrics)} at that combination of settings")
    return found


def _point(rec: _pv.Records, metric: str, where: dict[str, Pick], scope: dict[str, Any], axis: str, x: Any) -> _Point | None:
    named = {a: p for a, p in where.items() if p.kind == "value"}
    named.update({a: Pick("value", v) for a, v in scope.items()})
    named[axis] = Pick("value", x)
    try:
        cells = _pv._matching(rec, named, metric)
    except _Unfilled as why:
        if "no trials" in str(why):
            return None
        raise
    obs, _shared = _pv._observations(rec, cells, metric)
    s = _tt.summary(obs)
    if not s["n"]:
        return None
    return _Point(x, s["mean"], s["sd"], s["n"], cells)


def _change(rec: _pv.Records, metric: str, a: _Point, b: _Point) -> _Change:
    """``b`` minus ``a``: paired by random draw when the two settings ran on the same draws, else two independent samples."""
    da, db = _pv._draw_values(rec, a.cells, metric), _pv._draw_values(rec, b.cells, metric)
    common = [k for k in da if k in db]
    if len(common) >= 2:
        diffs = [db[k] - da[k] for k in common]
        s = _tt.summary(diffs)
        half = _stats.t_quantile(0.975, len(diffs) - 1) * s["sd"] / math.sqrt(len(diffs))
        # A spread at the rounding of the numbers themselves is no spread.
        if half <= 1e-12 * max([abs(d) for d in diffs] + [abs(da[k]) for k in common]):
            half = 0.0
        note = (f"; the means are over all {a.n} and {b.n} trials, the change and its interval over the {len(diffs)} paired draws"
                if len(diffs) < a.n or len(diffs) < b.n else "")
        return _Change(s["mean"], s["mean"] - half, s["mean"] + half, f"paired over {len(diffs)} shared random draws",
                       exact=(f"each of the {len(diffs)} shared random draws changed by the same amount, so the change "
                              "has no spread and no 95% interval") if half == 0 else "", note=note)
    diff = b.mean - a.mean
    if a.n < 2 or b.n < 2:
        return _Change(diff, None, None, "each setting ran once" if a.n < 2 and b.n < 2 else "one setting ran once")
    va, vb = a.sd ** 2 / a.n, b.sd ** 2 / b.n
    se = math.sqrt(va + vb)
    if se == 0:
        return _Change(diff, diff, diff, "two samples, no spread",
                       exact="the trials of both settings are identical, so the values are exact and there is no 95% interval")
    df = (va + vb) ** 2 / ((va ** 2 / (a.n - 1)) + (vb ** 2 / (b.n - 1)))
    half = _stats.t_quantile(0.975, df) * se
    return _Change(diff, diff - half, diff + half, "two independent samples (Welch)")


def _signed(ch: _Change) -> str:
    assert ch.lower is not None and ch.upper is not None
    text = _pv.fmt(_pv.Measured(ch.diff, "interval", ch.lower, ch.upper, "", None, symmetric=True), "pm")
    return text if text.startswith("-") else "+" + text


def _ratio(first: float, last: float) -> str:
    """The last over the first, only when both are above 0 (a ratio of a signed or zero value says nothing)."""
    return f"; the last is {_pv.sig(last / first, 3)} times the first" if first > 0 and last > 0 else ""


def _series(rec: _pv.Records, metric: str, axis: str, where: dict[str, Pick], scope: dict[str, Any],
            strict: bool = True) -> list[_Point] | None:
    xs = sorted(rec.axes()[axis], key=float)
    points = [p for x in xs if (p := _point(rec, metric, where, scope, axis, x)) is not None]
    if len(points) < 2:
        if not strict:
            return None
        raise _Unfilled(f"{rec.short(axis)} has fewer than two values with trials of {metric}"
                        + (f" at {_names(rec, scope)}" if scope else ""))
    return points


def _shape(rec: _pv.Records, axis: str, points: list[_Point]) -> str:
    if len(points) < 3:
        return ""
    steps = [b.mean - a.mean for a, b in zip(points, points[1:])]
    if all(s <= 0 for s in steps) and any(s < 0 for s in steps):
        return ", the means decrease at every step" if all(s < 0 for s in steps) else ", the means never rise (some steps are level)"
    if all(s >= 0 for s in steps) and any(s > 0 for s in steps):
        return ", the means increase at every step" if all(s > 0 for s in steps) else ", the means never fall (some steps are level)"
    return _not_monotone(rec, axis, points)


def _not_monotone(rec: _pv.Records, axis: str, points: list[_Point]) -> str:
    steps = [b.mean - a.mean for a, b in zip(points, points[1:])]
    if len(points) < 3 or all(s <= 0 for s in steps) or all(s >= 0 for s in steps):
        return ""
    lo, hi = min(points, key=lambda p: p.mean), max(points, key=lambda p: p.mean)
    return f", the means are not monotone (lowest at {rec.short(axis)} = {lo.x}, highest at {rec.short(axis)} = {hi.x})"


def _direction(rec: _pv.Records, metric: str, axis: str, points: list[_Point]) -> tuple[str, str, float | None]:
    """``(word, the scope's sentence, the last-over-first ratio)``."""
    first, last = points[0], points[-1]
    ch = _change(rec, metric, first, last)
    ratio = last.mean / first.mean if first.mean > 0 and last.mean > 0 else None
    ends = f"from {_pv.sig(first.mean)} at {rec.short(axis)} = {first.x} to {_pv.sig(last.mean)} at {rec.short(axis)} = {last.x}"
    if ch.word == "point":
        lead = "changes"
        detail = (f" {ends} (the change is {_pv.sig(ch.diff)}{_ratio(first.mean, last.mean)}); {ch.how}, so there is no 95% "
                  "interval and FI does not say whether the change is more than noise")
        return "point", lead + detail, ratio
    lead = {"rises": "rises", "falls": "falls", "none": "no change beyond its 95% interval"}[ch.word]
    if ch.exact:  # no spread on either side: the values are exact, a zero-width interval is not an interval
        lead = "no change" if ch.word == "none" else lead
        detail = f", {ends} (change {_pv.sig(ch.diff)}{_ratio(first.mean, last.mean)}; {ch.exact})"
        return ch.word, lead + detail + _shape(rec, axis, points), ratio
    detail = f", {ends} (change {_signed(ch)}, 95% interval{_ratio(first.mean, last.mean)}{ch.note})"
    return ch.word, lead + detail + _shape(rec, axis, points), ratio


def _finer_is_larger(points: list[_Point]) -> bool | None:
    """Which end of the axis the error falls toward, read from the record: ``True`` when the mean error is lower at the largest
    value of the axis than at the smallest, ``False`` for the reverse, ``None`` when the two are equal or undefined. A writer's
    ``| order`` / ``| order_size`` is a guess at the same fact (is the axis a count of steps or a step size?) and is only used
    for the words coarse and fine when this record agrees with it."""
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda p: float(p.x))
    lo, hi = ordered[0].mean, ordered[-1].mean
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
        return None
    return hi < lo


def _orders(points: list[_Point], finer_larger: bool | None) -> list[tuple[Any, Any, float | None]]:
    """``(first x, second x, order)`` for each neighbouring pair, from the end where the error is higher to the end where it is
    lower (``finer_larger`` False: from the largest x down); the order from the means, ``ln(e1 / e2) / |ln(x2 / x1)|``, ``None``
    where it is not defined (a zero or negative error, a non-positive x). With ``finer_larger`` ``None`` the pairs run from the
    smallest x up and the order is its absolute value, since the record does not say which way the error falls."""
    ordered = sorted(points, key=lambda p: float(p.x), reverse=finer_larger is False)
    out = []
    for a, b in zip(ordered, ordered[1:]):
        ok = a.mean > 0 and b.mean > 0 and float(a.x) > 0 and float(b.x) > 0 and float(a.x) != float(b.x)
        value = math.log(a.mean / b.mean) / abs(math.log(float(b.x) / float(a.x))) if ok else None
        out.append((a.x, b.x, abs(value) if value is not None and finer_larger is None else value))
    return out


def _order_words(rec: _pv.Records, axis: str, pairs: list[tuple[Any, Any, float | None]], labels: str) -> tuple[str, list[float]]:
    """``labels``: ``coarse`` (the writer's reading of the axis agrees with the record: coarsest and finest pair), ``lower``
    (the pairs are named by where the error is lower, no claim what the axis counts) or ``unknown`` (the record does not say
    which way the error falls: absolute orders and no verdict)."""
    name = rec.short(axis)
    defined = [(a, b, p) for a, b, p in pairs if p is not None]
    if not defined:
        return "no order is defined (an error is 0 or a value of the setting is not positive)", []
    first, last = ((" (the coarsest pair)", " (the finest pair)") if labels == "coarse"
                   else (" (the pair at the high-error end)", " (the pair at the low-error end)") if labels == "lower"
                   else ("", ""))

    def where(i: int) -> str:
        a, b, _p = pairs[i]
        end = first if i == 0 else last if i == len(pairs) - 1 else ""
        return f"{name} = {a} to {b}{end}"

    values = [p for _a, _b, p in defined]
    low = min(range(len(pairs)), key=lambda i: math.inf if pairs[i][2] is None else pairs[i][2])
    high = max(range(len(pairs)), key=lambda i: -math.inf if pairs[i][2] is None else pairs[i][2])
    note = ("" if labels != "unknown" else " (absolute values: the mean error is the same at both ends, so the record does "
            "not say which way it falls)")
    if len(pairs) <= MAX_ORDERS_LISTED:
        shown = "; ".join(f"{name} = {a} to {b}: " + ("not defined" if p is None else _pv.sig(p, 3)) for a, b, p in pairs)
        text = f"orders between neighbouring values, from the mean at each: {shown}{note}"
    else:
        text = (f"orders between {len(pairs)} neighbouring pairs, from the mean at each: {_pv.sig(min(values), 3)} to "
                f"{_pv.sig(max(values), 3)}; lowest {_pv.sig(pairs[low][2], 3)} at {where(low)}, highest "
                f"{_pv.sig(pairs[high][2], 3)} at {where(high)}{note}")
    if labels != "unknown":
        not_falling = [i for i, (_a, _b, p) in enumerate(pairs) if p is not None and p <= 0]
        if not_falling:
            text += "; the error does not fall at " + ", ".join(where(i) for i in not_falling[:4])
    return text, values


def trend_text(rec: _pv.Records, body: str) -> tuple[str, list[float], dict[str, Any]]:
    """``(the words, the numbers behind them, what they were computed from)`` for the key
    ``<result> over <axis> @ <where> [| order | order_size]`` (without its ``trend:`` prefix)."""
    head, where_text, parts = _split(body)
    if " over " not in head:
        raise _Unfilled("say what changes along what: trend:<result> over <axis> @ <other settings>")
    metric, _, axis_name = (s.strip() for s in head.partition(" over "))
    _need_result(rec, metric)
    order = ""
    for part in parts:
        if part in (ORDER, ORDER_SIZE) and not order:
            order = part
        else:
            raise _Unfilled(f"unknown part '{part}' (a trend takes at most one of: {ORDER}, {ORDER_SIZE})")
    axis = rec.real(axis_name)
    if axis not in rec.axes() or axis in rec.unread:
        raise _Unfilled(f"{axis_name} is not a setting of the record (its settings: {', '.join(rec.short(a) for a in rec.varied()) or 'none'})")
    values = rec.axes()[axis]
    if len(values) < 2 or not _numeric(values):
        raise _Unfilled(f"{rec.short(axis)} has no ordered numeric values to follow a change along "
                        f"(its values: {', '.join(str(v) for v in values)})")
    where = _where_of(rec, where_text)
    if axis in where:
        raise _Unfilled(f"{rec.short(axis)} is the setting the change is followed along; leave it out of '@'")
    _no_pooling(rec, where, f"{'an order' if order else 'a trend'} of {metric}", [metric])
    _pv._check_where(rec, {**where, axis: Pick("each")})
    label_axes = [a for a, p in where.items() if p.kind == "each"]
    unit = f"values of {rec.short(label_axes[0])}" if len(label_axes) == 1 else "settings"
    results: list[tuple[dict[str, Any], str, str, float | None]] = []
    skipped: list[dict[str, Any]] = []
    order_numbers: list[float] = []
    for scope in _scopes(rec, [metric], where):
        try:
            points = _series(rec, metric, axis, where, scope, strict=False)
            if points is None:  # fewer than two values of the axis gave a number there (every trial there is not a number)
                skipped.append(scope)
                continue
            if order:
                measured = _finer_is_larger(points)
                said = None if order == "" else (order == ORDER)
                labels = "unknown" if measured is None else "coarse" if measured == said else "lower"
                text, nums = _order_words(rec, axis, _orders(points, measured), labels)
                results.append((scope, "", text, min(nums) if nums else None))
                order_numbers += [min(nums), max(nums)] if nums else []
            else:
                word, text, ratio = _direction(rec, metric, axis, points)
                results.append((scope, word, text, ratio))
        except _Unfilled as why:
            raise _Unfilled(f"not computable{(' at ' + _names(rec, scope)) if scope else ''}: {why}") from None
    if not results:
        raise _Unfilled(f"{metric} has numbers at fewer than two values of {rec.short(axis)} wherever it was run")
    k = len(results)
    inputs = {"value": metric, "along": axis, "settings": "each" if label_axes else "fixed"}
    fixed = {a: p.value for a, p in where.items() if p.kind == "value"}

    def at(scope: dict[str, Any]) -> str:
        return _names(rec, {**fixed, **scope})

    left = (f"; {metric} has no numbers at {'; '.join(at(s) for s in skipped[:3])}" + (f" and {len(skipped) - 3} more" if len(skipped) > 3 else "")
            if skipped else "")
    counts = {w: sum(1 for r in results if r[1] == w) for w in ("falls", "rises", "none", "point")}
    said = {"falls": "falls", "rises": "rises", "none": "shows no change beyond its 95% interval",
            "point": "changes (no interval)"}
    summary = ", ".join(f"{said[w]} at {c} of {k} {unit}" for w, c in counts.items() if c)
    if order:
        if k <= MAX_SCOPES_LISTED:
            detail = "; ".join((f"at {at(s)}: " if at(s) else "") + t for s, _w, t, _r in results)
        else:
            low = min((r for r in results if r[3] is not None), key=lambda r: r[3], default=None)
            falling = [s for s, _w, t, _r in results if "the error does not fall" in t]
            detail = (f"orders computed at {k} {unit}"
                      + (f"; the lowest order is {_pv.sig(low[3], 3)} at {at(low[0])}" if low else "")
                      + (f"; the error does not fall between neighbouring values at {len(falling)} of them, among them "
                         + "; ".join(at(s) for s in falling[:4]) if falling else "; at none of them does the error fail to fall"))
        return (f"{metric} over {rec.short(axis)}, pairwise {detail}{left}", order_numbers, inputs)
    ratios = [r[3] for r in results if r[3] is not None]
    if k == 1:
        where_words = at(results[0][0])
        text = f"{metric} over {rec.short(axis)}" + (f" at {where_words}" if where_words else "") + f": {results[0][2]}"
    elif k <= MAX_SCOPES_LISTED:
        text = (f"{metric} over {rec.short(axis)}: {summary}. "
                + "; ".join(("At" if i == 0 else "at") + f" {at(s)}: {t}" for i, (s, _w, t, _r) in enumerate(results)))
    else:
        text = (f"{metric} over {rec.short(axis)}: {summary}"
                + (f"; the last over the first value is {_pv.sig(min(ratios), 3)} to {_pv.sig(max(ratios), 3)}" if ratios else ""))
    return text + left, ratios, inputs


# --- a result against another result, inside each setting ------------------------------------------------------------

_FORMS = ("mean", "sd", "rms")
_SPREAD = ("sd", "rms")


def _form_value(rec: _pv.Records, metric: str, form: str, cells: list[dict[str, Any]]) -> float:
    values = _pv._trials(cells, metric)
    if not values:
        raise _pv._NoNumbers(f"{metric} has no numeric trials at that setting")
    if form == "mean":
        return sum(values) / len(values)
    if form == "rms":
        return math.sqrt(sum(v * v for v in values) / len(values))
    return _pv._sd_of(rec, cells, metric)[0]


def _form_name(form: str, metric: str) -> str:
    return {"mean": f"the mean of {metric}", "sd": f"the SD of {metric}", "rms": f"the root-mean-square of {metric}"}[form]


def comparison_text(rec: _pv.Records, body: str) -> tuple[str, list[float], dict[str, Any]]:
    """The key ``<result> <form> vs <other result> <form> @ <axis>=each, ...`` (without its ``comparison:`` prefix): the
    ratio of the first to the second inside each setting, over the settings. A mean goes against a mean and a spread (``sd``
    or ``rms``) against a spread; an axis pooled with =all is refused."""
    head, where_text, parts = _split(body)
    if parts:
        raise _Unfilled("a comparison takes no | part")
    words = head.split()
    if len(words) != 5 or words[2] != "vs":
        raise _Unfilled("say it as comparison:<result> <form> vs <other result> <form> @ <settings>, forms mean, sd or rms")
    a_name, a_form, _vs, b_name, b_form = words
    if a_form not in _FORMS or b_form not in _FORMS:
        raise _Unfilled(f"the forms of a comparison are {', '.join(_FORMS)}")
    if (a_form == "mean") != (b_form == "mean"):
        raise _Unfilled("a mean and a spread are not like for like: compare a mean with a mean, or an sd with an sd or with the root-mean-square "
            "(rms) of a reported error (the like of an SD is the root-mean-square of the errors, not their mean)")
    if a_name == b_name and a_form == b_form:
        raise _Unfilled("a result compared with itself")
    for name in (a_name, b_name):
        _need_result(rec, name)
    where = _where_of(rec, where_text)
    _no_pooling(rec, where, f"{_form_name(a_form, a_name)} against {_form_name(b_form, b_name)}", [a_name, b_name])
    _pv._check_where(rec, where)
    if not any(p.kind == "each" for p in where.values()):
        raise _Unfilled("a comparison goes setting by setting: mark at least one setting =each")
    ratios: list[float] = []
    where_of: list[dict[str, Any]] = []
    inside = tested = 0
    for scope in _scopes(rec, [a_name, b_name], where):
        fixed = {a: (Pick("value", scope[a]) if a in scope else p) for a, p in where.items()}
        here = _names(rec, scope)
        try:
            cells_a, cells_b = _pv._matching(rec, fixed, a_name), _pv._matching(rec, fixed, b_name)
            fa, fb = _form_value(rec, a_name, a_form, cells_a), _form_value(rec, b_name, b_form, cells_b)
            if fb == 0:
                raise _Unfilled(f"{_form_name(b_form, b_name)} is 0")
        except _Unfilled as why:
            raise _Unfilled(f"not computable at {here}: {why}") from None
        ratios.append(fa / fb)
        where_of.append(scope)
        if a_form in ("sd", "mean"):
            try:
                m = _pv._measure(rec, a_name, "sd_ci" if a_form == "sd" else "mean_ci", "", fixed)
            except _Unfilled:
                continue
            if m.lower is not None and m.upper is not None:
                tested += 1
                inside += m.lower <= fb <= m.upper
    k = len(ratios)
    low, high = min(range(k), key=lambda i: ratios[i]), max(range(k), key=lambda i: ratios[i])
    text = (f"{_form_name(a_form, a_name)} divided by {_form_name(b_form, b_name)}, setting by setting over {k} settings: "
            f"{_pv.sig(ratios[low], 3)} to {_pv.sig(ratios[high], 3)} (lowest at {_names(rec, where_of[low])}, highest at "
            f"{_names(rec, where_of[high])})")
    if tested:
        text += (f"; {_form_name(b_form, b_name)} lies inside the 95% interval of {_form_name(a_form, a_name)} at {inside} of "
                 f"{tested} settings")
    return text, [ratios[low], ratios[high]], {"value": a_name, "against": b_name, "settings": "each"}
