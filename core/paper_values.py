"""Result numbers FI fills in from its own records, so the paper's writer never types one.

The writer places a key such as ``{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}}``; FI computes it from the
trial record (``raw/trials.json``) and the frozen plan, rounds it once (:func:`fmt`), and wraps it so the key survives::

    <!-- fi:v {{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}} -->0.8244%<!-- /fi:v -->

The grammar::

    {{v:<metric>[:<form>] @ <where> [| <op> <operand>] [| <reducer>] [| %]}}

    <where>    every axis the trials vary, as ``axis=<value>`` (one setting), ``axis=all`` (the trials of every setting
               of that axis pooled) or ``axis=each`` (kept apart, for a reducer)
    <form>     mean (default) | mean_ci | ci | pm | sd | sd_ci | se | n | count | failed | ref | corr:<metric2> |
               corr_ci:<metric2>
    <op>       diff (A - B) | ratio (A / B) | rel_diff (A / B - 1); the operand is ``ref``, ``@ <where2>`` (the same
               metric and form there) or ``<metric2>[:<form2>]`` (at the same where)
    <reducer>  min | max | range | max_abs | argmin | argmax | count_where:<word> (over the ``each`` axes)
    vs         ``| vs ref`` or ``| vs <metric2>`` (a value the plan or each setting sets): the comparison itself, worded by
               FI setting by setting, e.g. ``below <metric2> at all 5 values of sigma (0.914 to 0.927 times <metric2>)``;
               "within" means the reference lies inside the measured value's 95% interval at that setting

Naming every axis is deliberate: a key that left one out would render a pooled number that reads like a per-setting
one. Pooling is defined once: the trials of the matching settings are concatenated (Wilson interval for a 0/1 value,
t interval otherwise, :mod:`core.trial_table`); ``sd`` is the pooled within-setting SD, sqrt(sum((n_i - 1) s_i^2) /
sum(n_i - 1)), not the SD of the concatenated trials.

A key FI cannot fill (an unknown metric, form or axis value, an axis left out, a record that lacks the data) renders
``not recorded`` and is reported as a :class:`Hit`; it is never filled with a guess. No number is read from the
analysis script's output: only the trials and the plan.

Nothing here knows what a metric or an axis means; the names are the plan's.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import in_run as _in_run
from . import reference_values as _refs
from . import results_by_setting as _rbs
from . import stats as _stats
from . import trial_table as _tt

NOT_RECORDED = "not recorded"
FORMS = ("mean", "mean_ci", "ci", "pm", "sd", "sd_ci", "se", "n", "count", "failed", "ref", "corr", "corr_ci")
OPS = ("diff", "ratio", "rel_diff")
REDUCERS = ("min", "max", "range", "max_abs", "argmin", "argmax", "count_where")
VS = "vs"
#: The forms ``vs`` can judge, and the interval form it judges them by (a comparison is decided by the interval, never by a
#: fixed tolerance).
_VS_INTERVAL = {"mean": "mean_ci", "sd": "sd_ci", "corr": "corr_ci"}
#: Forms that are one number, so an op or a reducer can work on them.
_SCALAR = frozenset({"mean", "sd", "se", "n", "failed", "ref", "corr"})
_INTERVAL_FORMS = frozenset({"mean_ci", "ci", "pm", "sd_ci", "corr_ci"})
#: The rendered text a key opens and closes with (markers never nest).
_OPEN = "<!-- fi:v {{{{v:{key}}}}} -->"
CLOSE = "<!-- /fi:v -->"
_MARK = re.compile(r"<!-- fi:v \{\{v:(.*?)\}\} -->(.*?)<!-- /fi:v -->", re.S)
_PLACEHOLDER = re.compile(r"\{\{v:([^{}]{1,300})\}\}")
#: A number of the method FI computes from the plan's own numbers: the writer types ``{{calc:1 - 0.05 / 2}}``, FI keeps it
#: as the key ``calc:1 - 0.05 / 2`` (filled, marked and taken out like every other key) and shows it to the writer as typed.
CALC = "calc:"
_CALC_TYPED = re.compile(r"\{\{\s*calc\s*:\s*([^{}\n]{1,300}?)\s*\}\}")
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


class _Unfilled(Exception):
    """A key the records cannot support; the message is the reason."""


# --- rounding: one function ---------------------------------------------------------------------------------------


def _clean(text: str) -> str:
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def sig(x: float, digits: int = 4) -> str:
    """``x`` to ``digits`` significant figures, trailing zeros kept (6.050, not 6.05: the zero is precision)."""
    if x == 0:
        return "0"
    if not math.isfinite(x):
        raise _Unfilled("the value is not a finite number")
    exp = math.floor(math.log10(abs(x)))
    if exp >= 6 or exp < -4:
        return f"{x:.{digits - 1}e}"
    rounded = float(f"{x:.{digits - 1}e}")  # 0.99996 rounds up to 1: the exponent is that of the rounded number
    exp = math.floor(math.log10(abs(rounded)))
    place = exp - (digits - 1)
    return _clean(f"{round(x, -place):.{max(0, -place)}f}")


def interval_places(value: float, lower: float, upper: float) -> tuple[str, str, str]:
    """``(value, lower, upper)`` all to one decimal place: that of the larger half-width at two significant figures."""
    half = max(value - lower, upper - value, 0.0)
    if half == 0:
        return sig(value, 6), sig(lower, 6), sig(upper, 6)
    place = math.floor(math.log10(half)) - 1
    if math.floor(math.log10(round(half, -place))) - 1 > place:  # 0.0996 rounds up to 0.10
        place += 1
    decimals = max(0, -place)
    return tuple(_clean(f"{round(x, -place):.{decimals}f}") for x in (value, lower, upper))  # type: ignore[return-value]


@dataclass
class Measured:
    """A computed number as FI holds it, before it is written: ``kind`` says which rounding it gets."""

    value: float
    kind: str = "plain"  # plain | ratio | int | count | interval | words
    lower: float | None = None
    upper: float | None = None
    method: str = ""  # "95% Wilson interval"
    n: int | None = None  # trials behind it
    successes: int | None = None
    words: str = ""
    symmetric: bool = False  # a t interval: ``pm`` may print it as value +/- half-width
    parts: list[float] = field(default_factory=list)  # a range's two ends


def fmt(m: Measured, form: str = "mean", percent: bool = False) -> str:
    """The one rounding function. A value with an interval: the larger half-width at two significant figures, the value
    and both ends to that place; the ends are always written out, and ``+/-`` only for a symmetric (t) interval under
    ``pm``. A value without an interval: four significant figures, trailing zeros kept; a ratio three; a count exact."""
    if m.kind == "words":
        return m.words
    if m.kind == "int":
        return f"{int(m.value)}"
    if m.kind == "count":
        return f"{m.successes} of {m.n}"
    scale = 100.0 if percent else 1.0
    tail = "%" if percent else ""
    if m.kind == "interval":
        assert m.lower is not None and m.upper is not None
        v, lo, hi = interval_places(m.value * scale, m.lower * scale, m.upper * scale)
        if form == "ci":
            return f"{lo}{tail} to {hi}{tail}"
        if form == "pm" and m.symmetric:
            return f"{v}{tail} ± {interval_places(0.0, -(m.upper - m.value) * scale, (m.upper - m.value) * scale)[2]}{tail}"
        if form == "pm":
            return f"{lo}{tail} to {hi}{tail}"
        trials = f", {m.n} trials" if m.n else ""
        return f"{v}{tail} ({m.method} {lo}{tail} to {hi}{tail}{trials})"
    digits = 3 if m.kind == "ratio" else 4
    return sig(m.value * scale, digits) + tail


# --- the key ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pick:
    """What a key says about one axis: ``kind`` is ``value``, ``all`` or ``each``."""

    kind: str
    value: Any = None


@dataclass(frozen=True)
class Key:
    metric: str
    form: str
    where: dict[str, Pick]
    op: str = ""
    operand: str = ""  # "ref" | "where" | "metric"
    operand_where: dict[str, Pick] | None = None
    operand_metric: str = ""
    operand_form: str = "mean"
    reducer: str = ""
    word: str = ""  # count_where:<word>
    percent: bool = False
    form_arg: str = ""  # corr:<metric2>
    text: str = ""


@dataclass(frozen=True)
class Hit:
    """A key FI could not fill; the writer fixes the key in the next round."""

    key: str
    reason: str

    def text(self) -> str:
        return f"{placeholder(self.key)} is {NOT_RECORDED}: {self.reason}"


@dataclass
class Value:
    key: str
    text: str
    hit: Hit | None = None
    numbers: list[float] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.hit is None

    @property
    def marked(self) -> str:
        return _OPEN.format(key=self.key) + self.text + CLOSE


def _number(text: str) -> Any:
    try:
        return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    except ValueError:
        return text


def _where(text: str) -> dict[str, Pick]:
    out: dict[str, Pick] = {}
    for part in (p for p in text.split(",") if p.strip()):
        axis, eq, value = part.partition("=")
        axis, value = axis.strip(), value.strip()
        if not eq or not axis or not value:
            raise _Unfilled(f"'{part.strip()}' is not axis=value, axis=all or axis=each")
        if axis in out:
            raise _Unfilled(f"axis {axis} is named twice")
        out[axis] = Pick(value) if value in ("all", "each") else Pick("value", _number(value))
    return out


def _metric_form(text: str) -> tuple[str, str, str]:
    metric, _, form = text.strip().partition(":")
    form = form.strip() or "mean"
    base, _, arg = form.partition(":")
    if base not in FORMS:
        raise _Unfilled(f"unknown form '{base}' (the forms are {', '.join(FORMS)})")
    if (base in ("corr", "corr_ci")) != bool(arg.strip()):
        raise _Unfilled("corr and corr_ci name the other metric as corr:<metric>; no other form takes an argument")
    if not metric.strip():
        raise _Unfilled("no metric")
    return metric.strip(), base, arg.strip()


def parse(text: str) -> Key:
    """The key of a placeholder (``{{v:...}}`` or the text inside it); raises :class:`_Unfilled` with the reason."""
    body = text.strip()
    if body.startswith("{{"):
        body = body[2:-2] if body.endswith("}}") else body[2:]
    if body.startswith("v:"):
        body = body[2:]
    segments = [s.strip() for s in body.split("|")]
    head, at, where = segments[0].partition("@")
    if not at:
        raise _Unfilled("no '@ <where>': every key says where in the settings it is")
    metric, form, arg = _metric_form(head)
    kw: dict[str, Any] = {"metric": metric, "form": form, "where": _where(where), "form_arg": arg}
    for seg in segments[1:]:
        if seg == "%":
            kw["percent"] = True
            continue
        word, _, rest = seg.partition(" ")
        if word in OPS or word == VS:
            if kw.get("op"):
                raise _Unfilled("at most one op")
            kw["op"] = word
            rest = rest.strip()
            if rest == "ref":
                kw["operand"] = "ref"
            elif rest.startswith("@"):
                kw.update(operand="where", operand_where=_where(rest[1:]))
            elif rest:
                m2, f2, a2 = _metric_form(rest)
                kw.update(operand="metric", operand_metric=m2, operand_form=f2 + (f":{a2}" if a2 else ""))
            else:
                raise _Unfilled(f"{word} needs an operand: ref, @ <where> or <metric>")
            continue
        name, _, arg2 = seg.partition(":")
        if name.strip() in REDUCERS:
            if kw.get("reducer"):
                raise _Unfilled("at most one reducer")
            if (name.strip() == "count_where") != bool(arg2.strip()):
                raise _Unfilled("count_where takes a word (count_where:<word>), no other reducer does")
            kw.update(reducer=name.strip(), word=arg2.strip())
            continue
        raise _Unfilled(f"unknown part '{seg}' (an op {', '.join(OPS)}, {VS}; a reducer {', '.join(REDUCERS)}; or %)")
    if kw.get("op") == VS and (kw.get("reducer") or kw.get("percent")):
        raise _Unfilled("vs already gives the whole comparison in words; add no reducer and no %")
    if kw.get("op") == VS and kw.get("operand") == "where":
        raise _Unfilled("vs compares with ref or with a value each setting sets (vs ref, vs <metric>), not with another setting")
    return Key(text=body, **kw)


# --- the records --------------------------------------------------------------------------------------------------


@dataclass
class Records:
    """What a key is computed from: the cells of ``raw/trials.json`` that carry metrics (a cell with none is an empty
    per-value record of a study that compares values inside each run, and is never counted), the frozen plan, and the
    axes nothing reads (their cells are pooled and their copies counted once, as :func:`core.results_by_setting.distinct`)."""

    cells: list[dict[str, Any]]
    protocol: dict[str, Any] | None = None
    unread: tuple[str, ...] = ()
    #: The quantities FI left out of the results (name -> the check's note); no key reads one (:func:`left_out_of`).
    left_out: dict[str, str] = field(default_factory=dict)
    _refs_cache: dict[str, Any] = field(default_factory=dict)
    _alias_cache: dict[str, str] | None = None
    #: The topic as the person wrote it: its numbers may be used in a ``{{calc:...}}``, as they may be typed.
    topic: str = ""
    #: What :meth:`axes`, :meth:`metrics`, :meth:`in_run_axes` and :meth:`carrying` found, worked out once (the cells
    #: never change; the axes depend on ``unread`` too). A study with values compared inside each run has a record per
    #: value: a real one had 455,292, and each key worked these out again over all of them.
    _derived: dict[Any, Any] = field(default_factory=dict)

    def _once(self, name: str, make: Any) -> Any:
        key = (name, self.unread)
        if key not in self._derived:
            self._derived[key] = make()
        return self._derived[key]

    def carrying(self, metric: str | None) -> list[dict[str, Any]]:
        """The cells that carry ``metric`` (every cell for ``None``), in the record's order."""
        if metric is None:
            return self.cells
        def make() -> dict[str, list[dict[str, Any]]]:
            out: dict[str, list[dict[str, Any]]] = {}
            for c in self.cells:
                for m in c["metrics"]:
                    out.setdefault(m, []).append(c)
            return out
        return self._once("carrying", make).get(metric, [])

    def at_setting(self, metric: str, values: dict[str, Any]) -> list[dict[str, Any]] | None:
        """The cells carrying ``metric`` whose setting may be the one ``values`` names (an axis -> value for every axis
        those cells carry), found by lookup; ``None`` when ``values`` does not name exactly those axes. A candidate
        only: the caller still compares each value as :func:`_same` does."""
        def make() -> dict[str, Any]:
            return {}
        index = self._once("at_setting", make)
        if metric not in index:
            cells = self.carrying(metric)
            names = tuple(sorted({a for c in cells for a in c["cell"]}))
            table: dict[tuple, list[dict[str, Any]]] = {}
            for c in cells:
                if len(c["cell"]) == len(names):
                    table.setdefault(tuple(_norm(c["cell"][a]) for a in names), []).append(c)
            index[metric] = (names, table, sum(len(v) for v in table.values()) == len(cells))
        names, table, whole = index[metric]
        if not whole or set(values) != set(names):
            return None
        return table.get(tuple(_norm(values[a]) for a in names), [])

    def at_value(self, metric: str | None, axis: str, value: Any) -> list[dict[str, Any]]:
        """The cells carrying ``metric`` (any cell for ``None``) whose ``axis`` may hold ``value``, found by lookup (a
        candidate only, as :meth:`at_setting`'s)."""
        index = self._once("at_value", dict)
        if (metric, axis) not in index:
            table: dict[tuple[bool, str], list[dict[str, Any]]] = {}
            for c in self.carrying(metric):
                if axis in c["cell"]:
                    table.setdefault(_norm(c["cell"][axis]), []).append(c)
            index[(metric, axis)] = table
        return index[(metric, axis)].get(_norm(value), [])

    def levels(self, metric: str, axis: str) -> list[list[dict[str, Any]]]:
        """The cells carrying ``metric`` grouped by their value of ``axis`` (as its JSON text), in the record's order."""
        index = self._once("levels", dict)
        if (metric, axis) not in index:
            groups: dict[str, list[dict[str, Any]]] = {}
            for c in self.carrying(metric):
                if axis in c["cell"]:
                    groups.setdefault(json.dumps(c["cell"][axis], default=str), []).append(c)
            index[(metric, axis)] = list(groups.values())
        return index[(metric, axis)]

    def position(self, cell: dict[str, Any]) -> int:
        return self._once("position", lambda: {id(c): i for i, c in enumerate(self.cells)})[id(cell)]

    def aliases(self) -> dict[str, str]:
        """FI's short name for each axis whose own name a key cannot hold (a name with a comma, ``=``, ``@``, ``|`` or a brace,
        such as a whole sentence naming the values compared inside each run): ``compared`` for a value the plan compares
        inside each run, else its first words as a slug, made unique among the axes. Every other axis keeps its name. The sheet shows the short name; a key may use either."""
        if self._alias_cache is None:
            names: list[str] = []
            for c in self.cells:
                names.extend(k for k in c["cell"] if k not in names)
            taken = {n for n in names if not _unkeyable(n)}
            out: dict[str, str] = {}
            for n in names:
                if not _unkeyable(n):
                    out[n] = n
                    continue
                inside = (self.protocol or {}).get("in_each_run") if isinstance(self.protocol, dict) else None
                base = ("compared" if isinstance(inside, dict) and n in inside
                        else "_".join(re.findall(r"[A-Za-z0-9]+", n)[:3]).lower()[:30] or "axis")
                short, i = base, 2
                while short in taken:
                    short, i = f"{base}_{i}", i + 1
                taken.add(short)
                out[n] = short
            self._alias_cache = out
        return self._alias_cache

    def short(self, axis: str) -> str:
        return self.aliases().get(axis, axis)

    def real(self, name: str) -> str:
        """The axis a key's axis name stands for (its own name, or the short name FI gave it)."""
        for axis, short in self.aliases().items():
            if name == short:
                return axis
        return name

    def axes(self) -> dict[str, list[Any]]:
        """Every axis the cells carry and its values, in the record's order (an unread axis is not one)."""
        def make() -> dict[str, list[Any]]:
            found: dict[str, list[Any]] = {}
            for c in self.cells:
                for k, v in c["cell"].items():
                    if k not in self.unread and v not in found.setdefault(k, []):
                        found[k].append(v)
            return found
        return {k: list(vs) for k, vs in self._once("axes", make).items()}

    def varied(self) -> list[str]:
        return [k for k, vs in self.axes().items() if len(vs) > 1]

    def metrics(self) -> list[str]:
        return list(self._once("metrics", lambda: list(dict.fromkeys(m for c in self.cells for m in c["metrics"]))))

    def in_run_axes(self) -> list[str]:
        """The axes whose values are compared inside each run (the plan's ``in_each_run``, and the ``level`` a value's own
        record carries): the values of such an axis come from the same trials, so no key pools them."""
        def make() -> list[str]:
            names: list[str] = [str(k) for k in (_in_run.of(self.protocol) or {})]
            for c in self.cells:
                names.extend(k for k in (c.get("level") or {}) if k not in names)
            axes = self.axes()
            return [k for k in names if k in axes]
        return list(self._once("in_run_axes", make))


def _unkeyable(name: str) -> bool:
    return bool(re.search(r"[,|{}=@]", name)) or len(name) > 40


def from_cells(cells: list[Any], protocol: dict[str, Any] | None = None, unread: list[str] | tuple[str, ...] = (),
               left_out: dict[str, str] | None = None) -> Records:
    """The records of ``cells``; a quantity in ``left_out`` is not kept in them (it is not offered and no key fills it)."""
    gone = left_out or {}
    kept = []
    for c in cells:
        if not (isinstance(c, dict) and isinstance(c.get("metrics"), dict) and isinstance(c.get("cell"), dict)):
            continue
        metrics = {k: v for k, v in c["metrics"].items() if k not in gone}
        if metrics:
            kept.append({**c, "metrics": metrics})
    return Records(kept, protocol, tuple(unread), dict(gone))


def load(quest_root: Path | str, protocol: dict[str, Any] | None = None,
         unread: list[str] | tuple[str, ...] = (), left_out: dict[str, str] | None = None) -> Records | None:
    """The quest's records; ``None`` without a readable ``raw/trials.json`` (the paper's keys are then all not recorded)."""
    try:
        data = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cells = data.get("cells") if isinstance(data, dict) else None
    return from_cells(cells, protocol, unread, left_out) if isinstance(cells, list) else None


def _norm(value: Any) -> tuple[bool, str]:
    """``value`` as :func:`_same` compares it, to look a setting up by: a number to 9 significant digits, else its text."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, format(float(value), ".9g")
    return False, str(value).strip().lower()


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return str(a).strip().lower() == str(b).strip().lower()


def _check_where(rec: Records, where: dict[str, Pick], *, allow_each: bool = True) -> None:
    axes = rec.axes()
    for name, pick in where.items():
        if name in rec.unread:
            raise _Unfilled(f"{rec.short(name)} is an axis nothing reads; FI pools it, do not name it")
        if name not in axes:
            raise _Unfilled(f"{name} is not an axis of the record (its axes: {', '.join(rec.short(a) for a in axes) or 'none'})")
        if pick.kind == "value" and not any(_same(pick.value, v) for v in axes[name]):
            raise _Unfilled(f"{rec.short(name)}={pick.value} is not a setting of the record (it has "
                            f"{', '.join(str(v) for v in axes[name])})")
        if pick.kind == "each" and not allow_each:
            raise _Unfilled("each is only for a key with a reducer")
    missing = [a for a in rec.varied() if a not in where]
    if missing:
        raise _Unfilled(f"the key does not name every axis the study varies; missing: {', '.join(rec.short(a) for a in missing)} "
                        "(say =<value>, =all or =each)")


def _pooled_inside(rec: Records, axis: str, names: str) -> str:
    return (f"{rec.short(axis)} is compared inside each run: its values ({names}) are computed from the same trials, "
            f"so FI does not pool them (the same trials would be counted once per value). Name one value of it, "
            f"e.g. {rec.short(axis)}={names.split(', ')[0]}")


def _matching(rec: Records, where: dict[str, Pick], metric: str) -> list[dict[str, Any]]:
    """The cells that carry ``metric`` at the setting ``where`` names; each setting's copies (the same trials under an
    axis nothing reads) once."""
    def at(c: dict[str, Any]) -> bool:
        return all(p.kind != "value" or (a in c["cell"] and _same(c["cell"][a], p.value)) for a, p in where.items())

    for axis in rec.in_run_axes():
        # A key that pools an axis compared inside each run is refused below; told here from the first matching cell of
        # each of its values, before the whole record is read for it (a real record had 450,840 cells of one quantity).
        if where.get(axis, Pick("value")).kind == "all":
            firsts = [c for c in (next((c for c in group if at(c)), None) for group in rec.levels(metric, axis)) if c]
            if len(firsts) > 1:
                names = ", ".join(str(c["cell"][axis]) for c in sorted(firsts, key=rec.position))
                raise _Unfilled(_pooled_inside(rec, axis, names))
    named = {a: p.value for a, p in where.items() if p.kind == "value"}
    # A key that names every axis by its value is one setting: found by lookup, since a study with values compared
    # inside each run has a record per value (a real one had 455,292) and each key scanned them all.
    found = rec.at_setting(metric, named) if len(named) == len(where) else None
    if found is None and named:  # one setting of some axes: the fewest cells any of them names
        found = min((rec.at_value(metric, a, v) for a, v in named.items()), key=len)
    carrying = [c for c in (found or rec.carrying(metric)) if at(c)]
    if not carrying:
        some = min((rec.at_value(None, a, v) for a, v in named.items()), key=len) if named else rec.cells
        raise _Unfilled(f"no trials of {metric} at that setting" if any(at(c) for c in (some or rec.cells))
                        else "no trials at that combination of settings")
    for axis in rec.in_run_axes():
        if where.get(axis, Pick("value")).kind != "all":
            continue
        levels = list(dict.fromkeys(json.dumps(c["cell"][axis], default=str) for c in carrying if axis in c["cell"]))
        if len(levels) > 1:
            names = ", ".join(str(next(c["cell"][axis] for c in carrying if axis in c["cell"]
                                       and json.dumps(c["cell"][axis], default=str) == lv)) for lv in levels)
            raise _Unfilled(_pooled_inside(rec, axis, names))
    if rec.unread:
        seen: set[tuple[str, str]] = set()
        out = []
        for c in carrying:
            ident = (json.dumps({k: v for k, v in c["cell"].items() if k not in rec.unread}, default=str),
                     _tt.summary(_tt._values(c["metrics"][metric]))["digest"])
            if ident not in seen:
                seen.add(ident)
                out.append(c)
        return out
    return carrying


def _trials(cells: list[dict[str, Any]], metric: str) -> list[float]:
    return [v for c in cells for v in _tt._values(c["metrics"][metric])]


def _setting_values(cells: list[dict[str, Any]], metric: str) -> list[float] | None:
    """The one value per setting when ``metric`` is the same in every trial of every setting in ``cells`` (at least two
    trials each, so a value that was only measured once is never taken for one), else ``None``. A value like that is
    set by the setting, not measured in it: a theory line, a reference, an input echoed back."""
    values: list[float] = []
    for c in cells:
        got = _tt._values(c["metrics"][metric]) if metric in c["metrics"] else []
        if len(got) < 2 or not all(math.isclose(v, got[0], rel_tol=1e-9, abs_tol=1e-12) for v in got):
            return None
        values.append(got[0])
    return values or None


def _differs_between_settings(cells: list[dict[str, Any]], metric: str) -> bool:
    """Whether ``metric`` is a value each setting sets (:func:`_setting_values`) and the settings in ``cells`` do not agree
    on it. Its mean over them is then a number no setting has, set by how the grid happens to be laid out."""
    values = _setting_values(cells, metric)
    return values is not None and not all(math.isclose(v, values[0], rel_tol=1e-9, abs_tol=1e-12) for v in values)


def set_by_setting(rec: Records) -> list[str]:
    """The quantities of the record that are values each setting sets and that differ between settings, in the record's
    order (the theory lines): the ones that have no value over several settings."""
    return [m for m in rec.metrics()
            if _differs_between_settings(rec.carrying(m), m)]


# --- statistics ---------------------------------------------------------------------------------------------------


def _gamma_p(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function P(a, x)."""
    if x <= 0:
        return 0.0
    log_front = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(log_front)
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(log_front) * h


def chi2_quantile(p: float, dof: float) -> float:
    """The ``p`` quantile of the chi-square distribution with ``dof`` degrees of freedom."""
    lo, hi = 0.0, max(10.0, dof * 4)
    while _gamma_p(dof / 2, hi / 2) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _gamma_p(dof / 2, mid / 2) < p else (lo, mid)
    return (lo + hi) / 2


def _pooled_sd(cells: list[dict[str, Any]], metric: str) -> tuple[float, int]:
    """``(the within-setting pooled SD, its degrees of freedom)``."""
    num = dof = 0.0
    for c in cells:
        _, sd, n = _stats.mean_std(_tt._values(c["metrics"][metric]))
        if n >= 2:
            num += (n - 1) * sd * sd
            dof += n - 1
    if dof <= 0:
        raise _Unfilled("no setting here has two trials, so there is no SD")
    return math.sqrt(num / dof), int(dof)


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        raise _Unfilled("one of the two values is the same in every trial, so it has no correlation")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def _paired(cells: list[dict[str, Any]], a: str, b: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for c in cells:
        if a not in c["metrics"] or b not in c["metrics"]:
            raise _Unfilled(f"{b} is not recorded in every setting that records {a}")
        va, vb = _tt._values(c["metrics"][a]), _tt._values(c["metrics"][b])
        if len(va) != len(vb):
            raise _Unfilled(f"{a} and {b} do not have one value per trial in the same trials, so they cannot be paired")
        xs.extend(va)
        ys.extend(vb)
    if len(xs) < 4:
        raise _Unfilled("fewer than four paired trials")
    return xs, ys


# --- one setting's measure ----------------------------------------------------------------------------------------


def _reference(rec: Records, metric: str, cells: list[dict[str, Any]]) -> tuple[float, str]:
    """The plan's reference value for ``metric`` over ``cells`` (its ``kind``); one value, or the settings pooled must
    agree on it."""
    if not rec.protocol:
        raise _Unfilled("there is no frozen plan to take a reference value from")
    found: list[tuple[float, str, str]] = []
    for c in cells:
        ident = json.dumps({"metric": metric, "cell": c["cell"]}, sort_keys=True, default=str)
        if ident not in rec._refs_cache:
            computed = _refs.compute(rec.protocol, [dict(c["cell"])], {metric})
            kinds = {q["value"]: q.get("kind", "value") for q in computed["quantities"]
                     if q.get("status") == "computed" and q.get("value") == metric}
            row = next((r for r in computed["by_setting"] if r.get("value") == metric and r.get("reference") is not None), None)
            formula = next((q.get("formula") for q in computed["quantities"] if q.get("value") == metric), "")
            rec._refs_cache[ident] = (None if row is None or metric not in kinds else (row["reference"], kinds[metric], formula))
        got = rec._refs_cache[ident]
        if got is None:
            raise _Unfilled(f"the plan gives no reference for {metric} at that setting")
        found.append(got)
    if not all(math.isclose(f[0], found[0][0], rel_tol=1e-9, abs_tol=1e-12) for f in found):
        raise _Unfilled(f"the reference for {metric} differs between the settings pooled; name one setting")
    return found[0][0], found[0][1]


_LITERAL = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")


def _measure(rec: Records, metric: str, form: str, arg: str, where: dict[str, Pick]) -> Measured:
    cells = _matching(rec, where, metric)
    values = _trials(cells, metric)
    if not values and form not in ("failed",):
        raise _Unfilled(f"{metric} has no numeric trials at that setting")
    if form not in ("ref", "n", "failed", "count") and _differs_between_settings(cells, metric):
        raise _Unfilled(
            f"{metric} is a value each setting sets, and it differs between the settings pooled here, so it has no single "
            f"value over them (a pooled spread is not comparable to the mean of such values); name one setting, or compare "
            f"the measured value with it setting by setting: {{{{v:<measured>:sd @ <axis>=each, ... | ratio {metric} | range}}}}")
    if form == "ref":
        value, _kind = _reference(rec, metric, cells)
        return Measured(value)
    if form == "n":
        return Measured(len(values), "int")
    if form == "failed":
        counts = [c.get("failed") for c in cells]
        if not all(isinstance(k, int) and not isinstance(k, bool) for k in counts):
            raise _Unfilled("the record does not say how many trials failed there")
        return Measured(sum(counts), "int")
    summary = _tt.summary(values)
    if form == "count":
        if not summary["binary"]:
            raise _Unfilled(f"{metric} is not a 0/1 value, so there is no count of successes")
        return Measured(summary["mean"], "count", n=summary["n"], successes=int(summary["successes"]))
    if form == "mean":
        return Measured(summary["mean"])
    if form in ("mean_ci", "ci", "pm"):
        lo, hi, how = _tt.interval(summary)
        if lo is None or hi is None:
            raise _Unfilled("too few trials for an interval")
        words = "95% Wilson interval" if how == "Wilson" else f"95% {how}"
        return Measured(summary["mean"], "interval", lo, hi, words, summary["n"], symmetric=how != "Wilson")
    if form == "se":
        if summary["n"] < 2:
            raise _Unfilled("fewer than two trials, so there is no standard error")
        return Measured(summary["sd"] / math.sqrt(summary["n"]))
    if form in ("sd", "sd_ci"):
        sd, dof = _pooled_sd(cells, metric)
        if form == "sd":
            return Measured(sd)
        return Measured(sd, "interval", sd * math.sqrt(dof / chi2_quantile(0.975, dof)),
                        sd * math.sqrt(dof / chi2_quantile(0.025, dof)), "95% chi-square interval", summary["n"])
    xs, ys = _paired(cells, metric, arg)
    r = _pearson(xs, ys)
    if form == "corr":
        return Measured(r)
    if len(xs) < 4:
        raise _Unfilled("too few paired trials for an interval")
    z, half = math.atanh(r), 1.96 / math.sqrt(len(xs) - 3)
    return Measured(r, "interval", math.tanh(z - half), math.tanh(z + half), "95% Fisher interval", len(xs))


def _combine(key: Key, a: Measured, b: float) -> Measured:
    if key.op == "diff":
        return Measured(a.value - b)
    if b == 0:
        raise _Unfilled("the value it is compared with is 0")
    return Measured(a.value / b - 1 if key.op == "rel_diff" else a.value / b,
                    "plain" if key.op == "rel_diff" else "ratio")


def _operand(rec: Records, key: Key, where: dict[str, Pick]) -> float:
    if key.operand == "ref":
        return _reference(rec, key.metric, _matching(rec, where, key.metric))[0]
    if key.operand == "where":
        assert key.operand_where is not None
        _check_where(rec, key.operand_where, allow_each=False)
        form = key.form
        return _measure(rec, key.metric, form, key.form_arg, key.operand_where).value
    form, _, arg = key.operand_form.partition(":")
    return _measure(rec, key.operand_metric, form, arg, where).value


def _one(rec: Records, key: Key, where: dict[str, Pick]) -> Measured:
    got = _measure(rec, key.metric, key.form, key.form_arg, where)
    if not key.op:
        return got
    if key.form not in _SCALAR:
        raise _Unfilled(f"{key.op} works on one number; {key.form} is not (use {', '.join(sorted(_SCALAR))})")
    return _combine(key, got, _operand(rec, key, where))


def _number_word(k: int) -> str:
    return _WORDS[k] if 0 <= k < len(_WORDS) else str(k)


def _setting_words(setting: dict[str, Any]) -> str:
    return ", ".join(f"{a} = {v}" for a, v in setting.items())


def _reduce(rec: Records, key: Key) -> tuple[Measured, str]:
    each = [a for a, p in key.where.items() if p.kind == "each"]
    if not each:
        raise _Unfilled(f"{key.reducer} works over axes marked =each; none is")
    if key.form not in _SCALAR and not (key.reducer == "count_where" and key.form == "mean"):
        raise _Unfilled(f"{key.reducer} works on one number per setting; {key.form} is not")
    combos: list[tuple[Any, ...]] = []
    for c in rec.carrying(key.metric):
        combo = tuple(c["cell"].get(a) for a in each)
        if combo not in combos:
            combos.append(combo)
    if not combos:
        raise _Unfilled(f"no trials of {key.metric}")
    settings: list[dict[str, Any]] = []
    results: list[Measured] = []
    for combo in combos:
        fixed = {a: (Pick("value", combo[each.index(a)]) if a in each else p) for a, p in key.where.items()}
        try:
            results.append(_one(rec, key, fixed))
        except _Unfilled as why:
            raise _Unfilled(f"not computable at {_setting_words(dict(zip(each, combo)))}: {why}") from None
        settings.append(dict(zip(each, combo)))
    xs = [m.value for m in results]
    r = key.reducer
    kind = results[0].kind if key.op == "" else ("ratio" if key.op == "ratio" else "plain")
    if r == "count_where":
        if key.form != "mean" or key.op:
            raise _Unfilled("count_where compares each setting's mean with the plan's reference; use form mean, no op")
        k = 0
        for combo, m in zip(combos, results):
            fixed = {a: (Pick("value", combo[each.index(a)]) if a in each else p) for a, p in key.where.items()}
            cells = _matching(rec, fixed, key.metric)
            ref, ref_kind = _reference(rec, key.metric, cells)
            s = _tt.summary(_trials(cells, key.metric))
            lo, hi, _how = _tt.interval(s)
            k += _rbs.comparison(ref_kind, ref, m.value, lo, hi, s["n"]) == key.word
        return Measured(k, "words", words=_number_word(k)), "count"
    if r in ("min", "max"):
        return Measured((min if r == "min" else max)(xs), kind), r
    if r == "max_abs":
        return Measured(max(abs(x) for x in xs), kind), r
    if r == "range":
        return Measured(min(xs), kind, parts=[min(xs), max(xs)]), r
    pick = xs.index(min(xs) if r == "argmin" else max(xs))
    return Measured(xs[pick], "words", words=f"at {_setting_words(settings[pick])}"), r


def _plural(k: int, unit: str) -> str:
    return "at that setting" if k == 1 else f"at all {k} {unit}"


def _versus_instead(rec: Records, key: Key, why: str) -> str:
    """What the writer can do when ``| vs ref`` has no reference to judge by: the plan gave none for this value, so there is
    no key that compares it (no other value is offered: whether a value is the one to compare with is the plan's to say, not
    a guess from the record), and the comparison is left out of the paper or said in words with no number."""
    if key.operand != "ref" or "gives no reference" not in why:
        return ""
    return (f". FI has no reference value for {key.metric} to compare it with, so no key can make this comparison: leave it "
            "out of the paper, or say it in words with no number")


def _versus(rec: Records, key: Key) -> tuple[str, list[float]]:
    """The words for ``| vs <reference>``: FI judges the measured value against the reference at each setting kept apart by
    =each, by that setting's own 95% interval (above: the whole interval over the reference; below: under it; otherwise the
    reference is inside it), and counts the outcomes. Nothing is left to the writer's choice of word."""
    each = [a for a, p in key.where.items() if p.kind == "each"]
    if not each:
        raise _Unfilled("vs compares setting by setting; mark at least one axis =each")
    if key.form not in _VS_INTERVAL:
        raise _Unfilled(f"vs judges a mean, an sd or a corr by the 95% interval of the measured value; {key.form} has none "
                        f"(use form {', '.join(_VS_INTERVAL)})")
    combos: list[tuple[Any, ...]] = []
    for c in rec.carrying(key.metric):
        combo = tuple(c["cell"].get(a) for a in each)
        if combo not in combos:
            combos.append(combo)
    if not combos:
        raise _Unfilled(f"no trials of {key.metric}")
    verdicts: list[str] = []
    ratios: list[float] = []
    for combo in combos:
        fixed = {a: (Pick("value", combo[each.index(a)]) if a in each else p) for a, p in key.where.items()}
        here = _setting_words(dict(zip(each, combo)))
        try:
            got = _measure(rec, key.metric, _VS_INTERVAL[key.form], key.form_arg, fixed)
            ref = (_reference(rec, key.metric, _matching(rec, fixed, key.metric))[0] if key.operand == "ref"
                   else _measure(rec, key.operand_metric, "mean", "", fixed).value)
            if key.operand == "metric":
                if _setting_values(_matching(rec, fixed, key.operand_metric), key.operand_metric) is None:
                    raise _Unfilled(f"{key.operand_metric} is measured, not set by the setting, so it has no single value to "
                                    f"compare with")
        except _Unfilled as why:
            raise _Unfilled(f"not computable at {here}: {why}" + _versus_instead(rec, key, str(why))) from None
        assert got.lower is not None and got.upper is not None
        verdicts.append("above" if got.lower > ref else "below" if got.upper < ref else "within")
        if ref != 0:
            ratios.append(got.value / ref)
    name = "the plan's reference" if key.operand == "ref" else key.operand_metric
    unit = f"values of {each[0]}" if len(each) == 1 else "settings"
    k = len(verdicts)
    count = {v: verdicts.count(v) for v in ("above", "within", "below")}
    said = {"above": "above", "below": "below", "within": "consistent with"}
    present = [v for v in ("above", "within", "below") if count[v]]
    if len(present) == 1:
        v = present[0]
        head = f"{said[v]} {name}" + (" (inside the measured 95% interval)" if v == "within" else "") + f" {_plural(k, unit)}"
    else:
        first, rest = present[0], present[1:]
        head = (f"{said[first]} {name} at {count[first]} of {k} {unit}, "
                + ", ".join(f"{said[v]} it at {count[v]}" for v in rest))
    if len(ratios) == k:
        head += f" ({sig(min(ratios), 3)} to {sig(max(ratios), 3)} times it)"
    return head, ([min(ratios), max(ratios)] if len(ratios) == k else [])


def _inputs(rec: Records, key: Key) -> dict[str, Any]:
    """What a filled key was computed from: the value's name, how many settings and trials it pools (the settings it
    averages over; a key with ``each`` lists none)."""
    try:
        if any(p.kind == "each" for p in key.where.values()):
            return {"value": key.metric, "settings": "each"}
        cells = _matching(rec, key.where, key.metric)
        return {"value": key.metric, "settings": len(cells), "trials": len(_trials(cells, key.metric))}
    except _Unfilled:
        return {}


def _in_axis_names(rec: Records, key: Key) -> Key:
    """``key`` with each axis named by the short name FI gave it turned back to the axis's own name."""
    def back(where: dict[str, Pick] | None) -> dict[str, Pick] | None:
        return None if where is None else {rec.real(a): p for a, p in where.items()}

    return replace(key, where=back(key.where) or {}, operand_where=back(key.operand_where))


def _check_left_out(rec: Records, key: Key) -> None:
    """A key that names a quantity FI left out of the results is not recorded, and says why."""
    named = [key.metric, key.operand_metric] + ([key.form_arg] if key.form in ("corr", "corr_ci") else [])
    for name in named:
        if name and name in rec.left_out:
            note = rec.left_out[name].strip().rstrip(".")
            raise _Unfilled(f"FI left {name} out of the results" + (f": {note}" if note else "")
                            + "; say in the limitations that it was left out, do not report it")


def placeholder(key: str) -> str:
    """The key as the writer types it: ``{{calc:...}}`` for an expression, ``{{v:...}}`` for a result."""
    return "{{" + key + "}}" if key.startswith(CALC) else "{{v:" + key + "}}"


def keyed_calcs(markdown: str) -> str:
    """``markdown`` with each ``{{calc:EXPR}}`` the writer typed made the key ``{{v:calc:EXPR}}``, so FI fills it as a key."""
    return _CALC_TYPED.sub(lambda m: "{{v:" + CALC + m.group(1) + "}}", markdown or "")


def _calc_text(x: float) -> str:
    """An expression's value as a method's number is written: four significant figures, no trailing zeros (``1.96``,
    ``0.975``, ``3.182``, ``400``)."""
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    text = sig(x, 4)
    return _clean(text.rstrip("0").rstrip(".")) if "." in text and "e" not in text else text


def calc(expression: str, protocol: dict[str, Any] | None, topic: str = "") -> Value:
    """``{{calc:expression}}`` computed by FI. Every number in it must be one the writer may type (a number the plan or the
    topic states, or a whole number from 0 to 10; :mod:`core.typed_numbers`), so an expression can only work out what the
    plan already fixes (a quantile, a level, a reference value, a count from the grid), never carry a number of its own.
    It may call FI's calculator functions, ``ndtr``/``ndtri`` (the standard normal distribution and its quantile) and
    ``t_quantile(p, df)`` (the Student-t quantile). An expression FI does not compute reads ``not recorded`` with why."""
    from decimal import Decimal

    from . import oracle_forms as _forms
    from . import typed_numbers as _typed

    key = CALC + expression
    try:
        if not isinstance(protocol, dict) or not protocol:
            raise _Unfilled("FI has no plan to compute it from")
        plan = _typed.plan_numbers(protocol, topic)
        # The numbers are read from the parsed expression, not from its text: ``.97``, ``1_234`` or ``0x1F`` is a number of
        # its own however it is spelled.
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as e:
            raise _Unfilled(f"FI cannot read it ({e.__class__.__name__})") from None
        own: list[str] = []
        of_the_plan = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or isinstance(node.value, bool) \
                    or not isinstance(node.value, (int, float)):
                continue
            try:
                d = Decimal(repr(node.value))
            except ArithmeticError:
                d = None
            # A constant is never negative in the tree (``-0.5`` is a minus before 0.5), so a negative plan number counts.
            small = d is not None and d.is_finite() and d == d.to_integral_value() and 0 <= d <= 10
            if d is not None and d.is_finite() and (d in plan or -d in plan):
                # A whole number from 0 to 10 is allowed anyway: only another plan number shows the plan fixes the value.
                of_the_plan += 0 if small else 1
            elif not small:
                own.append(repr(node.value))
        if own:
            raise _Unfilled(f"it uses {', '.join(dict.fromkeys(own))}, which the plan and the topic do not state; write it "
                            "from the plan's own numbers as the plan writes them (a two-sided level is 1 - alpha / 2 with "
                            "the plan's alpha), or say the number in words")
        # Whole numbers alone make any number (6 / 7): an expression works out something from at least one of the plan's
        # or the topic's numbers. A bound, not a proof: 0.05 * 9 * 9 still passes.
        if not of_the_plan:
            raise _Unfilled("it uses none of the plan's or the topic's numbers, so it is not a number the plan fixes; type "
                            "a whole number as it is, or say the number in words")
        ev = _forms.evaluate(expression, {}, special={"ndtr": _forms._ndtr, "ndtri": _forms._ndtri,
                                                      "t_quantile": _stats.t_quantile})
        if ev.value is None:
            why = (f"it names {', '.join(ev.missing)}, which is no number" if ev.missing
                   else ev.unreadable or ev.problem or "it cannot be computed")
            raise _Unfilled(f"FI cannot compute it ({why})")
        return Value(key, _calc_text(float(ev.value)), numbers=[float(ev.value)], inputs={"expression": expression})
    except _Unfilled as why:
        return Value(key, NOT_RECORDED, Hit(key, str(why)))


def resolve(text: str, rec: Records | None) -> Value:
    """The key's value rendered, or ``not recorded`` with a :class:`Hit` saying why. Never a guess."""
    shown = text.strip()
    if shown.startswith("{{"):
        shown = shown[2:-2] if shown.endswith("}}") else shown[2:]
    shown = shown[2:] if shown.startswith("v:") else shown
    if shown.startswith(CALC):
        return calc(shown[len(CALC):].strip(), rec.protocol if rec else None, rec.topic if rec else "")
    try:
        key = parse(shown)
        if rec is None or not rec.cells:
            raise _Unfilled("this study has no per-setting trial record")
        key = _in_axis_names(rec, key)
        _check_left_out(rec, key)
        if key.metric not in rec.metrics():
            raise _Unfilled(f"{key.metric} is not a value the trials recorded (they recorded: {', '.join(rec.metrics())})")
        _check_where(rec, key.where)
        if key.percent and key.form in ("n", "count", "failed"):
            raise _Unfilled("% does not apply to a count")
        if not key.reducer and key.op != VS and any(p.kind == "each" for p in key.where.values()):
            raise _Unfilled("=each keeps settings apart for a reducer (min, max, range, ...); this key has none")
        if key.op == VS:
            words, nums = _versus(rec, key)
            return Value(shown, words, numbers=nums, inputs=_inputs(rec, key))
        if key.reducer:
            measured, how = _reduce(rec, key)
            tail = "%" if key.percent else ""
            if how == "range" and measured.parts:
                scale = 100.0 if key.percent else 1.0
                digits = 3 if measured.kind == "ratio" else 4
                out = " to ".join(sig(x * scale, digits) + tail for x in measured.parts)
            else:
                out = fmt(measured, key.form, key.percent and measured.kind != "words")
            return Value(shown, out, numbers=measured.parts or [measured.value], inputs=_inputs(rec, key))
        measured = _one(rec, key, key.where)
        if key.form == "sd" and not key.op:  # a spread is written with its own interval, as a mean's is
            measured = _measure(rec, key.metric, "sd_ci", key.form_arg, key.where)
        out = fmt(measured, key.form, key.percent)
        return Value(shown, out, numbers=[measured.value] + ([measured.lower, measured.upper] if measured.lower is not None else []),
                     inputs=_inputs(rec, key))
    except _Unfilled as why:
        return Value(shown, NOT_RECORDED, Hit(shown, str(why)))


# --- the paper -----------------------------------------------------------------------------------------------------


#: Where FI keeps every key the last draft used, its value, and the keys it could not fill.
RECORD = Path(".fi") / "paper_values.json"


def fill_values(markdown: str, rec: Records | None, plain: Any = None) -> tuple[str, list[Value]]:
    """``markdown`` with every ``{{v:...}}`` filled and wrapped in its marker, and each key's :class:`Value`. ``plain``:
    a function of a text offset that says the key sits where no reader sees a marker as markup (math, code); its value is
    then written bare."""
    values: list[Value] = []

    def one(m: re.Match[str]) -> str:
        value = resolve(m.group(1), rec)
        values.append(value)
        return value.text if plain is not None and plain(m.start()) else value.marked

    # A {{calc:...}} turned into its key keeps the length a caller's ``plain`` offsets were taken at only when none is
    # typed, so the caller makes the keys first (core/paper_facts.py:render); here they are made for any other caller.
    return _PLACEHOLDER.sub(one, keyed_calcs(markdown)), values


def fill(markdown: str, rec: Records | None) -> tuple[str, list[Hit]]:
    """``markdown`` with every ``{{v:...}}`` filled and wrapped in its marker, and the keys FI could not fill."""
    text, values = fill_values(markdown, rec)
    return text, [v.hit for v in values if v.hit]


def keys_in(markdown: str) -> list[str]:
    """The keys of a paper, filled or not, in order of first use."""
    found = [m.group(1).strip() for m in _MARK.finditer(markdown or "")]
    found += [m.group(1).strip() for m in _PLACEHOLDER.finditer(keyed_calcs(_MARK.sub("", markdown or "")))]
    return list(dict.fromkeys(found))


def unfilled_in(markdown: str) -> list[str]:
    """The keys of a rendered paper that read ``not recorded``."""
    return list(dict.fromkeys(m.group(1).strip() for m in _MARK.finditer(markdown or "")
                              if m.group(2).strip() == NOT_RECORDED))


def write_record(quest_root: Path | str, values: list[Value]) -> None:
    """Keep every key the paper used: its value, the numbers behind it, what it was computed from, and (for a key FI could
    not fill) why. This is the audit of "where did this number come from", and what the slides and poster can read."""
    path = Path(quest_root) / RECORD
    try:
        if not values:
            path.unlink(missing_ok=True)
            return
        seen: dict[str, Value] = {}
        for v in values:
            seen.setdefault(v.key, v)
        rows = [{"key": v.key, "text": v.text, "numbers": v.numbers, "inputs": v.inputs,
                 **({"not_recorded": v.hit.reason} if v.hit else {})} for v in seen.values()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"keys": rows, "not_recorded": [r["key"] for r in rows if "not_recorded" in r]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError):
        pass


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def unavailable_reasons(quest_root: Path | str) -> dict[str, str]:
    """Why each key of the last draft that FI could not fill was not filled."""
    return {r["key"]: str(r["not_recorded"]) for r in read_record(quest_root).get("keys") or []
            if isinstance(r, dict) and r.get("not_recorded")}


def records_of(quest_root: Path | str, protocol: dict[str, Any] | None) -> Records | None:
    """The quest's records for the keys: its trial record, the frozen plan, and the grid axes the simulation never reads
    (their cells are pooled, as the tables of every setting pool them). ``None`` without a trial record."""
    root = Path(quest_root)
    from . import result_checks as _checks

    rec = load(root, protocol, left_out=_checks.left_out_by_record(root))
    if rec is None or not rec.cells:
        return rec
    try:
        from . import grid_keys as _grid_keys

        formula_names: set[str] = set()
        metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
        for m in metrics if isinstance(metrics, list) else []:
            ref = m.get("reference") if isinstance(m, dict) else None
            if isinstance(ref, dict) and ref.get("formula") is not None:
                from . import oracle_forms as _forms

                formula_names.update(_forms.names(str(ref["formula"]), {**_forms.FUNCTIONS, **_forms.SPECIAL_FUNCTIONS}))
        unread, _why = _rbs.unread_axes(protocol, _grid_keys.code_files(root / "code"), formula_names)
        rec.unread = tuple(unread)
    except Exception:  # noqa: BLE001 -- an axis FI cannot tell is unread counts as read: nothing is pooled by guess
        rec.unread = ()
    return rec


def to_placeholders(markdown: str) -> str:
    """A filled paper turned back into keys, for the next writer round."""
    return _MARK.sub(lambda m: placeholder(m.group(1)), markdown or "")


def without_marks(markdown: str) -> str:
    """The reader's text: the markers gone, the values kept."""
    return _MARK.sub(lambda m: m.group(2), markdown or "")


# --- the sheet the writer is shown --------------------------------------------------------------------------------

#: The sheet lists at most this many lines before it falls back to pooled lines and each metric's extremes.
SHEET_MAX_LINES = 150
#: The most lines the fallback lists. Past studies' fallbacks were up to 676 lines and are listed whole; a study with 447
#: quantities over 455,292 records had 44,811 (16 MB) for the writer to read.
SHEET_MAX_SHOWN = 700
SHEET_GRAMMAR = (
    "Any key of this form can be placed, even one whose value is not listed here; FI fills it from its records: "
    "{{v:<value>[:<form>] @ axis=<value>|all, ...}}. Forms: " + ", ".join(FORMS) + ". Ops: " + ", ".join(OPS) + ". Comparison with a reference or theory value, worded by FI: | vs ref  or  | vs <value>" +
    ". Reducers over axes marked =each: " + ", ".join(REDUCERS) + ". Add | % for a percent. Name every axis.")


def _text_of(value: Any) -> str:
    return format(value, ".15g") if isinstance(value, float) else str(value)


def _addressable(value: Any) -> bool:
    return not re.search(r"[,|{}=@]", str(value)) and str(value).strip() == str(value) and str(value) != ""


def _line_key(metric: str, form: str, where: dict[str, Any], short: dict[str, str] | None = None) -> str:
    short = short or {}
    return "{{v:" + metric + (f":{form}" if form != "mean" else "") + " @ " + ", ".join(
        f"{short.get(a, a)}={_text_of(v) if not isinstance(v, Pick) else v.kind}" for a, v in where.items()) + "}}"


def _sheet_line(rec: Records, metric: str, where: dict[str, Any], sets: frozenset[str] = frozenset()) -> str | None:
    short = rec.aliases()
    key = _line_key(metric, "mean" if metric in sets else "mean_ci", where, short)  # a value a setting sets has no interval
    shown = resolve(key, rec)
    if shown.hit:
        return None
    parts = [f"{key} -> {shown.text}"]
    if metric in sets:
        return "- " + parts[0]
    sd = resolve(_line_key(metric, "sd", where, short), rec)
    if sd.filled:
        parts.append(f"sd {_line_key(metric, 'sd', where, short)} -> {sd.text}")
    ref = resolve(_line_key(metric, "ref", where, short), rec)
    if ref.filled:
        parts.append(f"reference {_line_key(metric, 'ref', where, short)} -> {ref.text}")
    return "- " + "; ".join(parts)


def _set_by_setting_line(rec: Records, varied: list[str]) -> str:
    varied = [rec.short(a) for a in varied]
    """The sheet's line on the quantities each setting sets (a theory line): no pooled line is offered for them, so this
    says how to compare a measured value with one. ``""`` when the record has none."""
    names = set_by_setting(rec)
    if not names or not varied:
        return ""
    where = ", ".join(f"{a}=each" if i == 0 else f"{a}=<value>|all" for i, a in enumerate(varied))
    return ("Compare with a value each setting sets (" + ", ".join(names) + "): it has no value over several settings, and "
            "a pooled measured value is not comparable to it. Compare setting by setting: "
            "{{v:<measured>:sd @ " + where + " | vs " + names[0] + "}}; FI words the comparison itself (above, below or "
            "consistent with it, at how many settings, and the ratio range), so type no comparison word beside it. "
            "{{... | ratio " + names[0] + " | range}} gives only the range.\n")


def _metric_lines(rec: Records, metric: str, axes: dict[str, list[Any]], varied: list[str],
                  usable: dict[str, list[Any]], sets: frozenset[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """One quantity's part of the sheet: its line per setting (headed by its name), its lowest and highest mean (headed),
    and its pooled lines, each with the axis it is a line of (``""`` for one pinned to the values compared in a run)."""
    lines: list[tuple[float, str]] = []
    seen: set[str] = set()
    for c in rec.carrying(metric):
        where = {a: v for a, v in c["cell"].items() if a in axes and a not in rec.unread}
        if not all(_addressable(v) for v in where.values()) or json.dumps(where, default=str) in seen:
            continue
        seen.add(json.dumps(where, default=str))
        line = _sheet_line(rec, metric, where, sets)
        if line:
            mean = _tt.summary(_trials([c], metric))["mean"] if _tt._values(c["metrics"][metric]) else 0.0
            lines.append((mean, line))
    full = [f"{metric}:"] + [line for _, line in lines]
    extremes: list[str] = []
    if lines:
        extremes = [f"{metric}: lowest and highest mean", min(lines, key=lambda x: x[0])[1],
                    max(lines, key=lambda x: x[0])[1]]
    pooled: list[tuple[str, str]] = []
    for axis in varied:
        for value in usable[axis]:
            where = {a: (value if a == axis else Pick("all")) for a in varied}
            line = _sheet_line(rec, metric, where, sets)
            if line:
                pooled.append((axis, line))
    if varied:
        # A pooled line over the grid's axes: an axis compared inside each run is never pooled, so where this metric's
        # cells hold more than one of its values it is pinned to each in turn (the refusal in _matching); a metric that
        # is one number per run has no such axis in its cells and is pooled whole.
        carrying = rec.carrying(metric)
        inside = [a for a in varied if a in rec.in_run_axes()
                  and len({json.dumps(c["cell"][a], default=str) for c in carrying if a in c["cell"]}) > 1]
        combos: list[dict[str, Any]] = [{}]
        for a in inside:
            combos = [dict(base, **{a: v}) for base in combos for v in usable[a]]
        for pin in combos:
            line = _sheet_line(rec, metric, {a: (pin[a] if a in pin else Pick("all")) for a in varied}, sets)
            if line:
                pooled.append(("", line))
    return full, extremes, pooled


def _planned(rec: Records, names: list[str]) -> list[str]:
    """The quantities of the record the plan names (its ``metrics``, by ``id``), in the plan's order."""
    metrics = rec.protocol.get("metrics") if isinstance(rec.protocol, dict) else None
    ids = [str(m.get("id")) for m in metrics if isinstance(m, dict) and m.get("id")] if isinstance(metrics, list) else []
    return [m for m in dict.fromkeys(ids) if m in names]


def _keys(lines: list[str]) -> int:
    return sum(1 for line in lines if line.startswith("- "))


def sheet(rec: Records | None, max_lines: int = SHEET_MAX_LINES, max_shown: int = SHEET_MAX_SHOWN) -> str:
    """The "Results you can place" block: for each value the trials recorded, a line per setting (the key and the value
    as FI will write it, with its SD and the plan's reference where there are any), then the pooled line of each axis.
    A pure function of the records. Over ``max_lines``, only the pooled lines and each value's lowest and highest
    setting are listed, and the grammar line says any key can still be placed. When those are over ``max_shown``, the
    lowest and highest of each quantity come first, the plan's quantities before the others (at most half the lines),
    then pooled lines taken in turn from each of the plan's quantities and axes, then from the others, ``max_shown``
    lines in all."""
    if rec is None or not rec.cells:
        return ("Results you can place: none. This study has no per-setting trial record, so FI cannot fill in "
                "any result number; do not type one.")
    axes = rec.axes()
    varied = rec.varied()
    usable = {a: [v for v in vs if _addressable(v)] for a, vs in axes.items()}
    names = [m for m in rec.metrics()]
    sets = frozenset(m for m in names if _setting_values(rec.carrying(m), m) is not None)
    planned = _planned(rec, names)
    order = planned + [m for m in names if m not in planned]
    parts: dict[str, tuple[list[str], list[str], list[tuple[str, str]]]] = {}

    def part(metric: str) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        if metric not in parts:
            parts[metric] = _metric_lines(rec, metric, axes, varied, usable, sets)
        return parts[metric]

    full_n = pooled_n = extremes_n = 0
    capped = False
    for i, metric in enumerate(order):
        full, ext, pool = part(metric)
        full_n, extremes_n = full_n + _keys(full), extremes_n + _keys(ext)
        pooled_n += _keys(list(dict.fromkeys(line for _, line in pool)))
        # Over both limits, with every planned quantity worked out: the lines of the rest are worked out only
        # while there is room left to show them.
        if full_n + pooled_n > max_lines and pooled_n + extremes_n > max_shown and i + 1 >= len(planned):
            capped = True
            break
    head = ("Results you can place (FI fills each key in from its records; never type a result). "
            "A line shows the key and what it will read as.")
    if capped:
        body = ["Too many settings to list; the lowest and highest mean of each quantity (the plan's first), then pooled "
                "lines, as many as fit (any other key can still be placed):"]
        room, ext_room = max_shown, max_shown // 2  # the extremes take at most half: pooled lines are kept too
        for metric in order:
            if min(room, ext_room) < 2:
                break
            ext = part(metric)[1]
            if ext and min(room, ext_room) >= _keys(ext):
                body.extend(ext)
                room, ext_room = room - _keys(ext), ext_room - _keys(ext)
        listed = set(body)
        for group in (planned, [m for m in order if m not in planned]):
            queues = []  # one per quantity and axis, taken in turn
            for metric in group:
                if room <= 0:
                    break
                for axis in dict.fromkeys(a for a, _ in part(metric)[2]):
                    queues.append([ln for a, ln in part(metric)[2] if a == axis])
            while room > 0 and any(queues):
                for q in queues:
                    if q and room > 0:
                        line = q.pop(0)
                        if line not in listed:  # a line pinned to one value can also be the line of its own axis
                            listed.add(line)
                            body.append(line)
                            room -= 1
        return head + "\n" + "\n".join(body) + "\n" + _set_by_setting_line(rec, varied) + SHEET_GRAMMAR
    full = [line for m in names for line in parts[m][0]]
    pooled = list(dict.fromkeys(line for m in names for _, line in parts[m][2]))  # a line pinned to one value can also be the line of that value's own axis
    extremes = [line for m in names for line in parts[m][1]]
    body = full + ["Pooled (the trials of the other axes together; sd is the within-setting pooled SD):"] + pooled
    if sum(1 for line in body if line.startswith("- ")) > max_lines:
        body = ["Too many settings to list; the pooled lines and each value's extremes:"] + pooled + extremes
    return head + "\n" + "\n".join(body) + "\n" + _set_by_setting_line(rec, varied) + SHEET_GRAMMAR


# --- what the prompts say -------------------------------------------------------------------------------------------

WRITER_RULES = """\
RESULT NUMBERS ARE FILLED IN BY FI. YOU DO NOT TYPE THEM.
Every number that is a result of this study (anything the run computed: a mean, an interval, an SD, a bias, a coverage, a
rate, a power, a correlation, a count of outcomes, a difference or ratio of these, the smallest or largest over settings) goes
into the paper as a key in double braces, never as digits. FI replaces each key with the value from its own records when the
paper is saved, so the reader sees the number.
You may type only: a number the plan or the topic states, written exactly as the plan writes it (a setting, a true value, a
nominal level, a constant of the method); the number of a figure, table, section or citation (Figure 2, Section 3, [4]); and a
whole number from 0 to 10 inside a formula. A typed result, or any other typed number, is sent back to you.
How to write a key: {{v:<value> @ <where>}}. The value is a quantity from the list below. <where> names every setting the
study varied, each as name=<value it took> (one setting), name=all (the trials of all its settings together) or name=each (kept
apart, for a smallest, largest or range). Add :<form> after the value for another form of it (default is the mean). Add | % to
write a percent. Examples: {{v:coverage @ noise=0.05, n=all}}, {{v:coverage:mean_ci @ noise=0.05, n=20}},
{{v:bias @ noise=each, n=each | range | %}}.
If FI cannot fill a key (a value the study did not record, a setting it did not run, a varied setting left out), it is sent back
to you. Write the key; FI computes the number, so never work one out yourself from the list.
To say how a measured value compares with a theory or reference value, place {{v:<value>:<form> @ <axis>=each, ... | vs
<theory value>}} (or | vs ref for the plan's reference) as the comparison itself: FI writes the words ("below ... at all 5
values of sigma (0.914 to 0.927 times ...)", or "consistent with" where the value's 95% interval holds it), setting by setting.
Do not type above, below, larger, smaller, exceeds, within or equal for such a comparison; never set one pooled number beside
another.
Numbers that are not available: a number that only the analysis script computed, and that no key on this list gives, is not
available. The Analysis block above shows such numbers; do not copy any of them as digits. Use the key for the same quantity
when the list has one, otherwise say it in words or leave it out."""


def writer_block(rec: Records | None) -> str:
    """The write prompt's "Results you can place" part: the rule in plain words, the quantities the trials recorded and the
    settings they vary, the sheet of keys with their values, and the grammar. ``""`` when there is no trial record (the paper
    then has no result for FI to fill, and the rule is not applied: :func:`has_keys`)."""
    if not has_keys(rec):
        return ""
    assert rec is not None
    varied = ", ".join(rec.short(a) for a in rec.varied()) or "none (one setting)"
    head = (f"The study's quantities: {', '.join(rec.metrics())}. The settings it varied: {varied}"
            + (f". Settings the simulation never reads, pooled by FI: {', '.join(rec.short(a) for a in rec.unread)}" if rec.unread else "") + ".")
    return WRITER_RULES + "\n" + head + "\n\n" + sheet(rec)


def has_keys(rec: Records | None) -> bool:
    """Whether this study has trial records FI can fill keys from (else the paper is not held to the rule)."""
    return rec is not None and bool(rec.cells)
