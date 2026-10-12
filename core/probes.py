"""Behaviour probes: does the simulation respond to its settings the way the plan says it must?

A probe is a *relation* between runs of the simulation, never an expected output number. The plan (read by a model, once)
states relations such as "the error shrinks when the step is halved" or "the result does not depend on the seed label"; FI
runs the cells itself and judges the numbers with plain statistics. No model ever supplies a number to compare against.

Everything here is pure and deterministic: no I/O, no model call, no clock, no module state. The running of cells lives
elsewhere; this module decides *which* cells (:func:`plan_runs`), reads what came back (:func:`judge`, :func:`saturation`),
and, for FI's own measurement of its false-stop rate, writes faulty copies of a simulation (:func:`mutants`).

The closed vocabulary (:data:`KINDS`):

======================  =========================================================================================
kind                    held when
======================  =========================================================================================
sensitivity             the metric differs between two values of the setting (paired runs; more than 4 SE)
invariance              the metric is the same at two values of the setting (within 4 SE)
monotone                the metric rises (or falls) over at least 3 values; a wrong step is tolerated within 2 SE
equal_when              two named cells give the same metric (a null case, a symmetry, a limit) within 4 SE
compare                 the metric is larger in one named cell than in the other (direction only, no value)
scaling                 multiplying the setting by c multiplies the metric by c**p (ratio within noise)
convergence             refining the knob shrinks the error; the fitted order is at least half the stated one
======================  =========================================================================================

plus ``saturation``, which needs no plan: NaN, a value pinned at 0 or 1, or one constant over a sweep.
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

KINDS = ("sensitivity", "invariance", "monotone", "equal_when", "compare", "scaling", "convergence")
AUTOMATIC = ("saturation",)
STRENGTHS = ("core", "supporting")
MAX_RELATIONS = 8
DEFAULT_SEEDS = 8
MAX_SEEDS = 20
#: A difference counts as real beyond this many standard errors; a monotone step may go the wrong way within MONOTONE_SE.
DIFFERENCE_SE = 4.0
MONOTONE_SE = 2.0
#: Two numbers from a deterministic run are "the same" within this relative amount.
SAME = 1e-9
#: A ratio from a deterministic run matches c**p within this relative amount.
RATIO_TOL = 1e-6
#: The fitted order of convergence must reach this share of the stated order.
ORDER_SHARE = 0.5
#: The fit of a few noisy levels may fall this share short of the half-order line without counting as broken.
ORDER_SLACK = 0.05
MIN_LEVELS = 3
SATURATION_MIN_CELLS = 3
#: A 0/1 indicator (a coverage, a success: several 0/1 draws in every cell) that sits at its bound in every cell is called stuck only with at least this many
#: draws (pooled; at the top bound 1, in EACH cell: a coverage or success rate is allowed to be all ones): a proportion near 0.95 comes out all-ones in a handful of draws by chance (0.95**36 is 16%, 0.95**72 is 2.5%).
INDICATOR_MIN_VALUES = 60

_UP = {"up", "increase", "increasing", "rise", "rises", "rising"}
_DOWN = {"down", "decrease", "decreasing", "fall", "falls", "falling"}
_EXPECTED_FIELDS = ("expected", "expected_value", "expected_values", "value", "values", "result", "number", "target")


# ---------------------------------------------------------------------------------------------------------- data classes


@dataclass
class Relation:
    """One relation the plan states. Carries no expected output value, by construction."""

    id: str
    kind: str
    metric: str
    setting: str | None = None
    strength: str = "supporting"
    quote: str = ""
    ledger_id: str | None = None
    direction: str | None = None      # monotone: "up" | "down"
    factor: float | None = None       # scaling: the setting is multiplied by this
    exponent: float | None = None     # scaling: the metric is multiplied by factor ** exponent
    order: float | None = None        # convergence: the stated order of the error in the knob
    refine: str | None = None         # convergence: "decrease" (a step size) | "increase" (a count)
    of: str = "mean"                  # convergence: "mean" (error against the finest level) | "spread" (seed-to-seed spread)
    a: dict[str, Any] = field(default_factory=dict)  # equal_when / compare: the first cell's settings
    b: dict[str, Any] = field(default_factory=dict)  # equal_when / compare: the second cell's settings
    larger: str | None = None         # compare: "a" or "b", the cell the plan says gives the larger metric

    def to_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if v not in (None, {}, "")}
        out.setdefault("strength", self.strength)
        return out


@dataclass
class Run:
    """One cell to run, with the number of paired seeds."""

    label: str
    cell: dict[str, Any]
    seeds: int
    relations: list[str] = field(default_factory=list)


@dataclass
class RunPlan:
    runs: list[Run]
    uses: dict[str, list[str]]            # relation id -> labels of the runs that judge it
    unrunnable: list[dict[str, str]]      # {"id", "reason"}: relations FI cannot test on this grid


@dataclass
class Observation:
    """What one cell returned for one metric: one value per paired seed (index i is the same seed in every cell)."""

    cell: dict[str, Any]
    values: list[float | None]


@dataclass
class Verdict:
    relation_id: str
    verdict: str   # "held" | "broken" | "unconfirmed"
    numbers: dict[str, Any]
    says: str

    def to_dict(self) -> dict[str, Any]:
        return {"relation_id": self.relation_id, "verdict": self.verdict, "numbers": self.numbers, "says": self.says}


# ------------------------------------------------------------------------------------------------------------ small maths


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _isnum(v: Any) -> bool:
    return _num(v) is not None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _se(xs: list[float]) -> float:
    return _sd(xs) / math.sqrt(len(xs)) if len(xs) >= 2 else 0.0


def _g(x: float) -> str:
    return f"{x:.4g}"


def _clean(values: list[Any]) -> list[float] | None:
    """The values as floats, or None when any is missing or not finite (a probe cannot judge a hole)."""
    out = [_num(v) for v in values]
    return None if not out or any(v is None for v in out) else [float(v) for v in out]  # type: ignore[arg-type]


def _paired(a: list[float], b: list[float]) -> list[float] | None:
    return None if len(a) != len(b) else [y - x for x, y in zip(a, b)]


def _scale(*xs: float) -> float:
    return max([abs(x) for x in xs] + [1e-300])


def noise(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation and standard error of the mean of one cell's values (a helper for callers and tests)."""
    if not values:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"), "n": 0}
    return {"mean": _mean(values), "sd": _sd(values), "se": _se(values), "n": len(values)}


def _cell_key(cell: dict[str, Any]) -> str:
    return json.dumps(cell, sort_keys=True, default=str)


# ----------------------------------------------------------------------------------------------------------------- parse


def _loads(reply: Any) -> Any:
    if isinstance(reply, (dict, list)):
        return reply
    text = str(reply or "")
    text = re.sub(r"```[a-zA-Z]*", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    dec = json.JSONDecoder()
    best: tuple[int, Any] = (0, None)
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if end - m.start() > best[0]:
            best = (end - m.start(), obj)
    return best[1]


def _protocol_parts(protocol: dict[str, Any] | None) -> tuple[dict[str, list[Any]], set[str]]:
    p = protocol if isinstance(protocol, dict) else {}
    if "grid" not in p and isinstance(p.get("protocol"), dict):
        p = p["protocol"]
    grid = {str(k): list(v) for k, v in (p.get("grid") or {}).items() if isinstance(v, list) and v} if isinstance(p.get("grid"), dict) else {}
    metrics: set[str] = set()
    for m in p.get("metrics") or []:
        if isinstance(m, dict) and m.get("id") is not None:
            metrics.add(str(m["id"]))
        elif isinstance(m, str):
            metrics.add(m)
    return grid, metrics


def _allowed(values: list[Any], v: Any) -> bool:
    """A value a cell may take: inside [min, max] for a numeric axis, one of the names for a categorical axis."""
    nums = [_num(x) for x in values]
    if all(n is not None for n in nums):
        f = _num(v)
        if f is None:
            return False
        lo, hi = min(nums), max(nums)  # type: ignore[type-var]
        return lo - 1e-12 * _scale(lo, hi) <= f <= hi + 1e-12 * _scale(lo, hi)
    return v in values


def _build(raw: dict[str, Any], n: int, grid: dict[str, list[Any]], metrics: set[str]) -> tuple[Relation | None, str]:
    rid = str(raw.get("id") or f"R{n}")
    kind = str(raw.get("kind") or "").strip().lower()
    if kind not in KINDS:
        return None, "unknown kind"
    metric = str(raw.get("metric") or "")
    if metric not in metrics:
        return None, "metric not in the protocol"
    quote = str(raw.get("from_plan") or raw.get("quote") or "").strip()
    if not quote:
        return None, "no quote from the plan"
    strength = str(raw.get("strength") or "supporting").lower()
    strength = strength if strength in STRENGTHS else "supporting"
    ledger = raw.get("ledger_id")
    rel = Relation(id=rid, kind=kind, metric=metric, strength=strength, quote=quote[:400],
                   ledger_id=str(ledger) if ledger not in (None, "") else None)
    if kind in ("equal_when", "compare"):
        for side in ("a", "b"):
            cell = raw.get(side)
            if not isinstance(cell, dict) or not cell:
                return None, f"{kind} needs two cells"
            for name, v in cell.items():
                if str(name) not in grid:
                    return None, "setting not in the protocol's grid"
                if not _allowed(grid[str(name)], v):
                    return None, "value outside the grid's range"
            setattr(rel, side, {str(k): v for k, v in cell.items()})
        if _cell_key(rel.a) == _cell_key(rel.b):
            return None, f"{kind} cells are the same"
        if kind == "compare":
            larger = str(raw.get("larger") or "").strip().lower()
            if larger not in ("a", "b"):
                return None, "compare needs `larger` = a or b"
            rel.larger = larger
        return rel, ""
    setting = str(raw.get("setting") or "")
    if setting not in grid:
        return None, "setting not in the protocol's grid"
    rel.setting = setting
    if kind == "monotone":
        d = str(raw.get("direction") or "").lower()
        if d in _UP:
            rel.direction = "up"
        elif d in _DOWN:
            rel.direction = "down"
        else:
            return None, "monotone needs a direction"
    elif kind == "scaling":
        c, p = _num(raw.get("factor")), _num(raw.get("exponent"))
        if c is None or c <= 0 or abs(c - 1) < 1e-9 or p is None:
            return None, "scaling needs a factor (positive, not 1) and an exponent"
        rel.factor, rel.exponent = c, p
    elif kind == "convergence":
        o = _num(raw.get("order"))
        r = str(raw.get("refine") or "decrease").lower()
        if o is None or o <= 0:
            return None, "convergence needs a stated order above 0"
        if r not in ("decrease", "increase"):
            return None, "convergence needs refine = decrease or increase"
        rel.order, rel.refine = o, r
        rel.of = "spread" if str(raw.get("of") or "").lower() == "spread" else "mean"
    return rel, ""


def parse(reply: Any, protocol: dict[str, Any] | None) -> tuple[list[Relation], list[dict[str, str]]]:
    """Read a reader model's reply. Strict: a relation whose setting is not in the protocol's grid, whose metric is not in its
    metrics, whose kind is unknown or that has no quote from the plan is dropped, with its reason (``dropped`` is a list of
    ``{"id", "reason"}``; :func:`drop_counts` counts the reasons). Any expected-output field is ignored. At most
    :data:`MAX_RELATIONS` are kept, core ones first."""
    grid, metrics = _protocol_parts(protocol)
    data = _loads(reply)
    items = data.get("relations") if isinstance(data, dict) else data
    items = items if isinstance(items, list) else []
    kept: list[Relation] = []
    dropped: list[dict[str, str]] = []
    seen: set[str] = set()
    for n, raw in enumerate(items, 1):
        if not isinstance(raw, dict):
            dropped.append({"id": f"R{n}", "reason": "not an object"})
            continue
        rel, why = _build({k: v for k, v in raw.items() if k not in _EXPECTED_FIELDS}, n, grid, metrics)
        if rel is None:
            dropped.append({"id": str(raw.get("id") or f"R{n}"), "reason": why})
            continue
        if rel.id in seen:
            rel.id = f"{rel.id}_{n}"
        seen.add(rel.id)
        kept.append(rel)
    if len(kept) > MAX_RELATIONS:
        ranked = sorted(range(len(kept)), key=lambda i: (kept[i].strength != "core", i))[:MAX_RELATIONS]
        keep_idx = set(ranked)
        dropped += [{"id": r.id, "reason": f"over the limit of {MAX_RELATIONS}"} for i, r in enumerate(kept) if i not in keep_idx]
        kept = [r for i, r in enumerate(kept) if i in keep_idx]
    return kept, dropped


def from_dict(d: Any) -> Relation | None:
    """A relation kept in ``.fi/probes.json`` (see :meth:`Relation.to_dict`) read back; None when it cannot be."""
    if not isinstance(d, dict):
        return None
    names = set(Relation.__dataclass_fields__)
    try:
        return Relation(**{k: v for k, v in d.items() if k in names})
    except TypeError:
        return None


def describe(rel: Relation | None, rid: str = "") -> str:
    """What a check was about, in the words of a scientist: ``that `T` changes `rate```. Used in the plain lines FI prints, and
    in the sentence the paper's limitations carry when the check did not hold."""
    if rel is None:
        name = rid.split(":", 1)[1] if rid.startswith("saturation:") else rid
        return f"that `{name}` varies over the settings run" if name else "of the simulation"
    m, s = f"`{rel.metric}`", f"`{rel.setting}`"
    return {
        "sensitivity": f"that {s} changes {m}",
        "invariance": f"that {s} does not change {m}",
        "monotone": f"that {m} {'rises' if rel.direction == 'up' else 'falls'} with {s}",
        "equal_when": f"that {m} is the same in the two cases the plan says match",
        "compare": f"that {m} is larger in one case than in the other, as the plan says",
        "scaling": f"that {m} scales with {s} as the plan says",
        "convergence": f"that {m} converges as {s} is refined",
    }.get(rel.kind, f"that {m} behaves as the plan says")


def drop_counts(dropped: Iterable[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dropped:
        out[d["reason"]] = out.get(d["reason"], 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------- plan_runs


def _base(grid: dict[str, list[Any]]) -> dict[str, Any]:
    """The base cell: each setting at its middle value (the same base the run estimate times)."""
    return {name: values[len(values) // 2] for name, values in grid.items() if values}


def _sorted_values(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda v: _num(v)) if all(_isnum(v) for v in values) else list(values)


def _spread(values: list[Any], k: int) -> list[Any]:
    """Up to k values spread evenly over the sorted axis."""
    vs = _sorted_values(values)
    if len(vs) <= k:
        return vs
    idx = sorted({round(i * (len(vs) - 1) / (k - 1)) for i in range(k)})
    return [vs[i] for i in idx]


def _levels(values: list[Any], k: int, geometric: bool) -> list[Any]:
    """At least MIN_LEVELS levels of an axis: its own values when it has enough, else values inside [min, max]."""
    vs = _spread(values, k)
    if len(vs) >= MIN_LEVELS or not all(_isnum(v) for v in values):
        return vs if len(vs) >= MIN_LEVELS else []
    lo, hi = min(_num(v) for v in values), max(_num(v) for v in values)  # type: ignore[type-var]
    if hi <= lo:
        return []
    ints = all(float(_num(v)).is_integer() for v in values)  # type: ignore[arg-type]
    pts = ([lo * (hi / lo) ** (i / (MIN_LEVELS - 1)) for i in range(MIN_LEVELS)] if geometric and lo > 0
           else [lo + (hi - lo) * i / (MIN_LEVELS - 1) for i in range(MIN_LEVELS)])
    if ints:
        pts = [float(round(p)) for p in pts]
        pts = sorted(set(pts))
        return [int(p) for p in pts] if len(pts) >= MIN_LEVELS else []
    return pts


def _scaling_pair(values: list[Any], c: float) -> tuple[Any, Any] | None:
    nums = [(float(_num(v)), v) for v in values if _isnum(v)]  # type: ignore[arg-type]
    if len(nums) != len(values):
        return None
    best = None
    for x, xv in nums:
        for y, yv in nums:
            if x > 0 and y > 0 and abs(y / x - c) <= 0.01 * c and (best is None or abs(y / x - c) < best[0]):
                best = (abs(y / x - c), xv, yv)
    if best:
        return best[1], best[2]
    lo, hi = min(n for n, _ in nums), max(n for n, _ in nums)
    ints = all(n.is_integer() for n, _ in nums)

    def ok(v: float) -> bool:
        return (not ints) or abs(v - round(v)) < 1e-9

    if lo > 0 and lo * c <= hi * (1 + 1e-12) and ok(lo * c):
        return (int(lo) if ints else lo), (int(round(lo * c)) if ints else lo * c)
    if hi / c >= lo * (1 - 1e-12) and hi / c > 0 and ok(hi / c):
        return (int(round(hi / c)) if ints else hi / c), (int(hi) if ints else hi)
    return None


def plan_runs(relations: list[Relation], grid: dict[str, list[Any]], seeds: int = DEFAULT_SEEDS,
              deterministic: bool = False) -> RunPlan:
    """Which cells to run for ``relations``, and how many paired seeds each (``seeds`` clamped to 1..20; 1 when the simulation
    is ``deterministic``). Setting values are the grid's own values where possible, else inside [min, max]; the other
    settings stay at the base cell. Identical cells shared by several relations run once."""
    n_seeds = 1 if deterministic else max(1, min(MAX_SEEDS, int(seeds)))
    base = _base(grid)
    runs: dict[str, Run] = {}
    uses: dict[str, list[str]] = {}
    unrunnable: list[dict[str, str]] = []

    def want(rel: Relation, overrides: list[dict[str, Any]]) -> None:
        labels = []
        for ov in overrides:
            cell = {**base, **ov}
            key = _cell_key(cell)
            if key not in runs:
                runs[key] = Run(label=f"c{len(runs) + 1}", cell=cell, seeds=n_seeds)
            if rel.id not in runs[key].relations:
                runs[key].relations.append(rel.id)
            labels.append(runs[key].label)
        uses[rel.id] = labels

    for rel in relations:
        values = grid.get(rel.setting or "", [])
        if rel.kind in ("equal_when", "compare"):
            want(rel, [rel.a, rel.b])
            continue
        if not values:
            unrunnable.append({"id": rel.id, "reason": "setting not in the grid"})
            continue
        s = rel.setting or ""
        if rel.kind in ("sensitivity", "invariance"):
            vs = _sorted_values(values)
            if len({_cell_key({"v": v}) for v in vs}) < 2:
                unrunnable.append({"id": rel.id, "reason": "the setting has one value in the grid"})
                continue
            want(rel, [{s: vs[0]}, {s: vs[-1]}])
        elif rel.kind == "monotone":
            lv = _levels(values, 5, geometric=False)
            if not lv:
                unrunnable.append({"id": rel.id, "reason": "fewer than 3 levels of the setting are available"})
                continue
            want(rel, [{s: v} for v in lv])
        elif rel.kind == "scaling":
            pair = _scaling_pair(values, rel.factor or 1.0)
            if pair is None:
                unrunnable.append({"id": rel.id, "reason": "no two values of the setting in range differ by the factor"})
                continue
            want(rel, [{s: pair[0]}, {s: pair[1]}])
        elif rel.kind == "convergence":
            lv = _levels(values, 5, geometric=True)
            if not lv:
                unrunnable.append({"id": rel.id, "reason": "fewer than 3 levels of the knob are available"})
                continue
            want(rel, [{s: v} for v in lv])
    return RunPlan(runs=list(runs.values()), uses=uses, unrunnable=unrunnable)


# ------------------------------------------------------------------------------------------------------------------ judge


def _obs(o: Any) -> tuple[dict[str, Any], list[Any]]:
    if isinstance(o, Observation):
        return o.cell, o.values
    if isinstance(o, dict):
        return dict(o.get("cell") or {}), list(o.get("values") or [])
    cell, values = o
    return dict(cell), list(values)


def _unconfirmed(rel: Relation, why: str, **numbers: Any) -> Verdict:
    return Verdict(rel.id, "unconfirmed", numbers, f"Could not check this: {why}.")


def _matches(cell: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(k in cell and (cell[k] == v or (_isnum(cell[k]) and _isnum(v) and math.isclose(float(cell[k]), float(v), rel_tol=1e-9)))
               for k, v in want.items())


def _by_setting(rel: Relation, observations: list[Any]) -> list[tuple[Any, list[float]]] | None:
    """(setting value, clean values) sorted by the setting's value; None when a cell has no usable values or the setting is
    missing."""
    rows: dict[str, tuple[Any, list[float]]] = {}
    for o in observations:
        cell, raw = _obs(o)
        if rel.setting not in cell:
            return None
        vals = _clean(raw)
        if vals is None:
            return None
        rows.setdefault(_cell_key({"v": cell[rel.setting]}), (cell[rel.setting], vals))
    out = list(rows.values())
    if all(_isnum(v) for v, _ in out):
        out.sort(key=lambda r: float(_num(r[0])))  # type: ignore[arg-type]
    return out


def _diff(a: list[float], b: list[float]) -> tuple[float, float, float, list[float]] | None:
    """(mean paired difference b - a, its SE, the scale of the two means, the differences)."""
    d = _paired(a, b)
    if d is None:
        return None
    return _mean(d), _se(d), _scale(_mean(a), _mean(b)), d


def _same(md: float, se: float, scale: float) -> bool:
    """Whether a difference is nothing: within 4 SE of noise, or (no noise) within SAME of the values' size."""
    return abs(md) <= (DIFFERENCE_SE * se if se > 0 else SAME * scale)


def judge(relation: Relation, observations: list[Any]) -> Verdict:
    """Judge one relation from the runs. ``observations``: one per cell run, each ``Observation`` (or ``(cell, values)``) with
    the metric's value per paired seed. The verdict is ``held``, ``broken`` or ``unconfirmed`` (the noise is too large, a cell
    returned no number, there are too few levels); ``says`` is a plain sentence with the numbers."""
    fn = _JUDGES.get(relation.kind)
    if fn is None:
        return _unconfirmed(relation, f"{relation.kind} is not a known kind of check")
    try:
        return fn(relation, observations)
    except (ZeroDivisionError, ValueError, OverflowError):
        return _unconfirmed(relation, "the numbers could not be compared")


def _pick_ab(rel: Relation, observations: list[Any]) -> tuple[list[float], list[float]] | None:
    """The clean values of the two named cells of an equal_when / compare relation; None when either returned no number."""
    named = set(rel.a) | set(rel.b)
    found: dict[str, list[tuple[dict[str, Any], list[float]]]] = {"a": [], "b": []}
    for side, want in (("a", rel.a), ("b", rel.b)):
        for o in observations:
            cell, raw = _obs(o)
            vals = _clean(raw)
            if _matches(cell, want) and vals is not None:
                found[side].append((cell, vals))
    # the two cells of one relation are the same cell except where the relation names a difference: a cell that merely
    # contains the named settings but sits elsewhere in the grid is another relation's
    for ca, va in found["a"]:
        for cb, vb in found["b"]:
            if all(_cell_key({"v": ca.get(k)}) == _cell_key({"v": cb.get(k)}) for k in (set(ca) | set(cb)) - named):
                return va, vb
    return None


def _compare(rel: Relation, observations: list[Any]) -> Verdict:
    """Direction only: the plan says one named cell gives the larger metric. Held when it does by more than 2 SE of the paired
    difference; broken when the other cell is larger by more than 4 SE (or by more than rounding when nothing varies);
    otherwise a tie or noise, which FI cannot call a fault."""
    got = _pick_ab(rel, observations)
    if got is None:
        return _unconfirmed(rel, "one of the two cells returned no number")
    va, vb = got
    r = _diff(va, vb)
    if r is None:
        return _unconfirmed(rel, "the two cells ran a different number of seeds")
    md, se, scale, d = r      # md = mean(b - a)
    sign = 1.0 if rel.larger == "b" else -1.0    # sign * md > 0 means the plan's larger cell is larger
    n = len(va)
    nums = {"mean_a": _mean(va), "mean_b": _mean(vb), "mean_difference": md, "se": se, "n_pairs": n, "larger": rel.larger}
    m = f"`{rel.metric}`"
    big, small = (rel.b, rel.a) if rel.larger == "b" else (rel.a, rel.b)
    where = f"{_fmt_cell(big)} against {_fmt_cell(small)}"
    plan_mean, other_mean = (_mean(vb), _mean(va)) if rel.larger == "b" else (_mean(va), _mean(vb))
    tol_hold = MONOTONE_SE * se if se > 0 else SAME * scale
    tol_break = DIFFERENCE_SE * se if se > 0 else SAME * scale
    if sign * md > tol_hold:
        return Verdict(rel.id, "held", nums, f"{m} is larger for {where} ({_g(plan_mean)} against {_g(other_mean)}), as the plan says.")
    if sign * md < -tol_break:
        return Verdict(rel.id, "broken", nums, f"{m} is smaller for {where} ({_g(plan_mean)} against {_g(other_mean)}, difference {_g(abs(md))}, "
                                               f"SE {_g(se)}, {n} paired runs), but the plan says it is larger.")
    return _unconfirmed(rel, f"{m} for {where} is {_g(plan_mean)} against {_g(other_mean)}, a tie or within the noise (SE {_g(se)}) of {n} paired runs", **nums)


def _pair_judge(rel: Relation, observations: list[Any]) -> Verdict:
    if rel.kind == "equal_when":
        got = _pick_ab(rel, observations)
        if got is None:
            return _unconfirmed(rel, "one of the two cells returned no number")
        va, vb = got
        where = f"{_fmt_cell(rel.a)} and {_fmt_cell(rel.b)}"
    else:
        rows = _by_setting(rel, observations)
        if rows is None or len(rows) < 2:
            return _unconfirmed(rel, f"`{rel.setting}` was not run at two values with numbers back")
        (xa, va), (xb, vb) = rows[0], rows[-1]
        where = f"`{rel.setting}` = {xa} and {xb}"
    r = _diff(va, vb)
    if r is None:
        return _unconfirmed(rel, "the two cells ran a different number of seeds")
    md, se, scale, d = r
    n = len(va)
    nums = {"mean_a": _mean(va), "mean_b": _mean(vb), "mean_difference": md, "se": se, "n_pairs": n}
    m = f"`{rel.metric}`"
    if rel.kind == "sensitivity":
        if max(abs(x) for x in d) <= SAME * scale:
            return Verdict(rel.id, "broken", nums, f"{m} is exactly the same ({_g(_mean(va))}) at {where} in all {n} paired runs, but the plan says it responds to `{rel.setting}`.")
        if se > 0 and abs(md) > DIFFERENCE_SE * se or se == 0:
            return Verdict(rel.id, "held", nums, f"{m} changes from {_g(_mean(va))} to {_g(_mean(vb))} between {where}, as the plan says.")
        return _unconfirmed(rel, f"{m} moves by {_g(md)} between {where}, which is within the noise (SE {_g(se)}) of {n} paired runs", **nums)
    # invariance / equal_when
    if _same(md, se, scale):
        return Verdict(rel.id, "held", nums, f"{m} is the same within noise at {where} ({_g(_mean(va))} and {_g(_mean(vb))}), as the plan says.")
    return Verdict(rel.id, "broken", nums, f"{m} differs between {where}: {_g(_mean(va))} against {_g(_mean(vb))} (difference {_g(md)}, SE {_g(se)}, {n} paired runs), but the plan says they match.")


def _fmt_cell(cell: dict[str, Any]) -> str:
    return ", ".join(f"`{k}` = {v}" for k, v in cell.items())


def _monotone(rel: Relation, observations: list[Any]) -> Verdict:
    rows = _by_setting(rel, observations)
    if rows is None or len(rows) < MIN_LEVELS:
        return _unconfirmed(rel, f"`{rel.setting}` needs at least {MIN_LEVELS} values with numbers back")
    sign = 1.0 if rel.direction == "up" else -1.0
    word = "rise" if rel.direction == "up" else "fall"
    steps = []
    wrong = []
    for (x0, v0), (x1, v1) in zip(rows, rows[1:]):
        r = _diff(v0, v1)
        if r is None:
            return _unconfirmed(rel, "the cells ran different numbers of seeds")
        md, se, scale, _d = r
        steps.append({"from": x0, "to": x1, "change": md, "se": se})
        tol = MONOTONE_SE * se if se > 0 else SAME * scale
        if sign * md < -tol:
            wrong.append((x0, x1, md, se))
    net = _diff(rows[0][1], rows[-1][1])
    assert net is not None
    nmd, nse, nscale, nd = net
    nums = {"steps": steps, "net_change": nmd, "net_se": nse}
    m = f"`{rel.metric}`"
    if wrong:
        x0, x1, md, se = wrong[0]
        return Verdict(rel.id, "broken", nums, f"{m} should {word} with `{rel.setting}`, but goes the other way from {x0} to {x1} (change {_g(md)}, SE {_g(se)}).")
    if max(abs(x) for x in nd) <= SAME * nscale:
        return Verdict(rel.id, "broken", nums, f"{m} stays at {_g(_mean(rows[0][1]))} over `{rel.setting}` from {rows[0][0]} to {rows[-1][0]}, but the plan says it should {word}.")
    if sign * nmd > (MONOTONE_SE * nse if nse > 0 else SAME * nscale):
        return Verdict(rel.id, "held", nums, f"{m} {word}s with `{rel.setting}` ({_g(_mean(rows[0][1]))} to {_g(_mean(rows[-1][1]))} over {len(rows)} values), as the plan says.")
    return _unconfirmed(rel, f"{m} moves by {_g(nmd)} over `{rel.setting}`, which is within the noise (SE {_g(nse)})", **nums)


def _scaling(rel: Relation, observations: list[Any]) -> Verdict:
    rows = _by_setting(rel, observations)
    if rows is None or len(rows) < 2:
        return _unconfirmed(rel, f"`{rel.setting}` was not run at two values with numbers back")
    (xa, va), (xb, vb) = rows[0], rows[-1]
    xa_f, xb_f = _num(xa), _num(xb)
    if xa_f is None or xb_f is None or xa_f <= 0 or xb_f <= 0:
        return _unconfirmed(rel, "scaling needs positive numeric values of the setting")
    c = xb_f / xa_f
    target = c ** (rel.exponent or 0.0)
    if len(va) != len(vb):
        return _unconfirmed(rel, "the two cells ran a different number of seeds")
    z = [y - target * x for x, y in zip(va, vb)]
    scale = _scale(_mean(vb), target * _mean(va))
    tol = DIFFERENCE_SE * _se(z) + RATIO_TOL * scale
    ma = _mean(va)
    nums = {"factor": c, "expected_ratio": target, "ratio": _mean(vb) / ma if abs(ma) > 1e-300 else None, "mean_a": ma, "mean_b": _mean(vb), "se": _se(z)}
    m = f"`{rel.metric}`"
    if abs(_mean(z)) <= tol:
        return Verdict(rel.id, "held", nums, f"{m} scales as the plan says: `{rel.setting}` x{_g(c)} (from {xa} to {xb}) gives x{_g(target)} (exponent {_g(rel.exponent or 0.0)}); observed {_g(_mean(vb))} against {_g(target * ma)}.")
    return Verdict(rel.id, "broken", nums, f"{m} does not scale as the plan says: `{rel.setting}` x{_g(c)} should give x{_g(target)} (exponent {_g(rel.exponent or 0.0)}), but {_g(ma)} became {_g(_mean(vb))} (x{_g(_mean(vb) / ma) if abs(ma) > 1e-300 else 'n/a'}).")


def _convergence(rel: Relation, observations: list[Any]) -> Verdict:
    rows = _by_setting(rel, observations)
    if rows is None or len(rows) < MIN_LEVELS or not all(_isnum(x) and float(_num(x)) > 0 for x, _ in rows):  # type: ignore[arg-type]
        return _unconfirmed(rel, f"`{rel.setting}` needs at least {MIN_LEVELS} positive levels with numbers back")
    fine_first = rel.refine == "decrease"   # sorted ascending: the finest level is first for a step size
    ordered = rows if fine_first else list(reversed(rows))   # finest first
    pts: list[tuple[float, float]] = []      # (knob, error)
    noisy = 0
    if rel.of == "spread":
        for x, v in ordered:
            pts.append((float(_num(x)), _sd(v)))  # type: ignore[arg-type]
    else:
        # successive differences between neighbouring levels: for levels in a fixed ratio they follow the same power of the knob
        # as the error does, whatever the (unknown) limit is
        for (_xf, fine_v), (xc, coarse_v) in zip(ordered, ordered[1:]):
            r = _diff(fine_v, coarse_v)
            if r is None:
                return _unconfirmed(rel, "the cells ran different numbers of seeds")
            md, se, scale, _d = r
            if se > 0 and abs(md) <= MONOTONE_SE * se:
                noisy += 1
                continue
            pts.append((float(_num(xc)), abs(md)))  # type: ignore[arg-type]
        scale_all = _scale(*[_mean(v) for _x, v in ordered])
        if pts and all(e <= SAME * scale_all for _x, e in pts):
            return Verdict(rel.id, "held", {"errors": pts}, f"`{rel.metric}` is already the same at every level of `{rel.setting}` (differences below {SAME:g} of its size): converged.")
    pos = [(x, e) for x, e in pts if e > 0]
    if len(pos) < 2:
        if rel.of != "spread" and len(pts) >= 1 and not pos:
            return Verdict(rel.id, "held", {"errors": pts}, f"`{rel.metric}` does not change between levels of `{rel.setting}`: converged.")
        return _unconfirmed(rel, f"the changes in `{rel.metric}` between levels of `{rel.setting}` are within the noise or too few to fit an order", errors=pts, noisy_levels=noisy)
    lx, ly = [math.log(x) for x, _ in pos], [math.log(e) for _, e in pos]
    mx, my = _mean(lx), _mean(ly)
    var = sum((a - mx) ** 2 for a in lx)
    if var <= 0:
        return _unconfirmed(rel, "the levels of the knob are all the same")
    slope = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / var
    order = slope if rel.refine == "decrease" else -slope
    stated = rel.order or 0.0
    # error must shrink toward the finest level: the coarsest has the largest error
    by_coarse = sorted(pos, key=lambda p: p[0], reverse=(rel.refine == "increase"))   # finest first
    shrinks = by_coarse[-1][1] > by_coarse[0][1]
    nums = {"observed_order": order, "stated_order": stated, "levels": len(pos), "errors": pos}
    what = "spread between runs" if rel.of == "spread" else "change between neighbouring levels"
    if shrinks and order >= ORDER_SHARE * stated * (1 - ORDER_SLACK):
        return Verdict(rel.id, "held", nums, f"The {what} of `{rel.metric}` shrinks as `{rel.setting}` is refined, with a fitted order of {_g(order)} over {len(pos)} levels (the plan says {_g(stated)}).")
    if not shrinks:
        return Verdict(rel.id, "broken", nums, f"The {what} of `{rel.metric}` does not shrink as `{rel.setting}` is refined (fitted order {_g(order)}); the plan says it converges at order {_g(stated)}.")
    return Verdict(rel.id, "broken", nums, f"The {what} of `{rel.metric}` shrinks too slowly as `{rel.setting}` is refined: fitted order {_g(order)} over {len(pos)} levels, but the plan says {_g(stated)} (at least {_g(ORDER_SHARE * stated)} is needed).")


_JUDGES = {"sensitivity": _pair_judge, "invariance": _pair_judge, "equal_when": _pair_judge, "compare": _compare, "monotone": _monotone,
           "scaling": _scaling, "convergence": _convergence}


# ------------------------------------------------------------------------------------------------------------ saturation


def exempt_names(protocol: dict[str, Any] | None) -> set[str]:
    """The names the plan expects to stay put (the same exemptions the degenerate-results check uses)."""
    from . import degenerate_results as _dr

    return _dr.expected_constants(protocol)


#: Words that make a quantity a check value (which is meant to sit at 0) or a flag (which is meant to sit at 1), not a result.
_ZERO_BY_DESIGN = ("error", "residual", "violation", "drift", "conservation", "mismatch", "defect")
_ONE_BY_DESIGN = ("_defined", "_ok", "_valid", "_flag", "_converged", "_passed")


def _diagnostic(name: str, bound: float) -> bool:
    """Whether ``name`` at ``bound`` is a diagnostic sitting where it is meant to (an error term at 0, a flag at 1)."""
    low = name.lower()
    if bound == 0.0:
        return any(w in low for w in _ZERO_BY_DESIGN)
    return low.startswith(("is_", "has_")) or any(low.endswith(w) or w + "_" in low for w in _ONE_BY_DESIGN)


def saturation(cells: list[dict[str, Any]], exempt: Iterable[str] = (), headline: Iterable[str] = ()) -> list[Verdict]:
    """The generic test, needing no plan and no model. ``cells``: one per setting run, each ``{"metrics": {name: [values]}}``
    (the same shape the degenerate-results check reads). A metric is ``broken`` when any value is NaN or missing, when every
    cell sits at exactly 0 (or at exactly 1; a 0/1 indicator needs :data:`INDICATOR_MIN_VALUES` draws (at the top bound, in every setting: 8 of 8 is a good coverage, not a fault); a check value or flag sitting at its bound by design is only ``unconfirmed``), or when everything
    the run returns is one constant over a sweep of at least :data:`SATURATION_MIN_CELLS` cells; it is ``unconfirmed``
    when it alone is constant over such a sweep while something else moves (it may be a fixed control). Names in ``exempt`` (see :func:`exempt_names`) may be
    constant or pinned by design, but never NaN."""
    skip = set(exempt)
    protect = set(headline)   # reported / headline quantities: never excused as a check value or a flag
    names = sorted(set.intersection(*[set((c.get("metrics") or {}).keys()) for c in cells])) if cells else []
    out: list[Verdict] = []
    means: dict[str, list[float]] = {}
    n_values: dict[str, int] = {}
    thinnest: dict[str, int] = {}
    indicator: set[str] = set()
    for name in names:
        bad = 0
        total = 0
        ms: list[float] = []
        for c in cells:
            vals = list(c["metrics"][name])
            total += len(vals)
            clean = [v for v in (_num(x) for x in vals) if v is not None]
            bad += len(vals) - len(clean)
            if clean:
                ms.append(_mean(clean))
        if bad:
            out.append(Verdict(f"saturation:{name}", "broken", {"missing": bad, "of": total},
                               f"`{name}` is not a number in {bad} of {total} values over the {len(cells)} settings run."))
        elif ms:
            means[name] = ms
            n_values[name] = total
            thinnest[name] = min(len(c["metrics"][name]) for c in cells)
            if all(len(c["metrics"][name]) >= 2 and all(_num(x) in (0.0, 1.0) for x in c["metrics"][name]) for c in cells):
                indicator.add(name)
    moving = {n for n, ms in means.items() if len({round(m, 12) for m in ms}) > 1 or (max(ms) - min(ms)) > SAME * _scale(*ms)}
    for name, ms in means.items():
        if name in skip or len(ms) < 2:
            continue
        vid = f"saturation:{name}"
        pinned = next((b for b in (0.0, 1.0) if all(abs(m - b) <= 1e-12 for m in ms)), None)
        if pinned is not None:
            said = f"`{name}` is exactly {pinned:g} at all {len(ms)} settings run, a bound it cannot leave."
            thin = name in indicator and (n_values[name] < INDICATOR_MIN_VALUES
                                          or (pinned == 1.0 and thinnest[name] < INDICATOR_MIN_VALUES))
            if thin:
                # a coverage or success rate comes out all ones at a handful of draws a setting: not judged, and said so
                out.append(Verdict(vid, "unconfirmed", {"bound": pinned, "cells": len(ms), "draws_per_setting": thinnest[name]},
                                   said[:-1] + f", but with {thinnest[name]} draw(s) a setting that is not enough to call a "
                                   f"bound (a 0/1 proportion needs {INDICATOR_MIN_VALUES}); the bound was not judged."))
            elif name not in protect and _diagnostic(name, pinned):
                out.append(Verdict(vid, "unconfirmed", {"bound": pinned, "cells": len(ms)},
                                   said[:-1] + " (it looks like a check value or a flag, which may sit at its bound by design)."))
            else:
                out.append(Verdict(vid, "broken", {"bound": pinned, "cells": len(ms)}, said))
        else:
            flat = (max(ms) - min(ms)) <= SAME * _scale(*ms)
            if flat and len(ms) >= SATURATION_MIN_CELLS and (moving or len(means) == len([n for n in means if n not in skip])):
                # Constant while other results move is "unconfirmed": a control the study holds fixed on purpose (the temperatures
                # of the kinetics study, a true rate) looks the same, and FI cannot tell from the numbers. Everything constant is broken.
                out.append(Verdict(vid, "unconfirmed" if moving else "broken", {"value": ms[0], "cells": len(ms)},
                                   f"`{name}` is {_g(ms[0])} at every one of the {len(ms)} settings run"
                                   + (", although other results change with them (fine if it is a fixed control of the study)." if moving else ", and so is everything else the run returns.")))
    return out


# --------------------------------------------------------------------------------------------------------------- mutators


def _compiles(source: str) -> bool:
    try:
        compile(source, "<mutant>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _is_key(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == name


class _IgnoreSetting(ast.NodeTransformer):
    """Reads of a setting (``cell["x"]``, ``cell.get("x", d)``, or a parameter called ``x``) become a fixed value."""

    def __init__(self, setting: str, value: Any) -> None:
        self.setting, self.value, self.changed = setting, value, 0

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load) and _is_key(node.slice, self.setting):
            self.changed += 1
            return ast.copy_location(ast.Constant(self.value), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args and _is_key(node.args[0], self.setting):
            self.changed += 1
            return ast.copy_location(ast.Constant(self.value), node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.setting:
            self.changed += 1
            return ast.copy_location(ast.Constant(self.value), node)
        return node


class _Metric(ast.NodeTransformer):
    """Edits the expression a returned dict (or a ``out["m"] = ...`` assignment) gives a metric."""

    def __init__(self, metric: str, edit: Any) -> None:
        self.metric, self.edit, self.changed = metric, edit, 0

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        self.generic_visit(node)
        for i, k in enumerate(node.keys):
            if _is_key(k, self.metric):
                node.values[i] = self.edit(node.values[i])
                self.changed += 1
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        for t in node.targets:
            if isinstance(t, ast.Subscript) and _is_key(t.slice, self.metric):
                node.value = self.edit(node.value)
                self.changed += 1
                break
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            for kw in node.keywords:
                if kw.arg == self.metric:
                    kw.value = self.edit(kw.value)
                    self.changed += 1
        return node


def _finish(tree: ast.AST, changed: int) -> str | None:
    if not changed:
        return None
    ast.fix_missing_locations(tree)
    try:
        text = ast.unparse(tree)
    except (ValueError, RecursionError):
        return None
    return text + "\n" if _compiles(text) else None


def ignore_setting(source: str, setting: str, default: Any) -> str | None:
    """The simulation with ``setting`` never read: every use is the fixed ``default`` instead. None when it is not read."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    t = _IgnoreSetting(setting, default)
    return _finish(t.visit(tree), t.changed)


def hard_code_metric(source: str, metric: str, value: float = 0.5) -> str | None:
    """The simulation returning the constant ``value`` for ``metric``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    t = _Metric(metric, lambda _old: ast.Constant(value))
    return _finish(t.visit(tree), t.changed)


def clip_metric(source: str, metric: str, bound: float = 1.0) -> str | None:
    """The simulation with ``metric`` pinned at ``bound`` (it can never leave it)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    def edit(old: ast.expr) -> ast.expr:
        new = ast.parse(f"min(max(0, {bound!r}), {bound!r})", mode="eval").body
        new.args[0].args[0] = old  # type: ignore[attr-defined]
        return new

    t = _Metric(metric, edit)
    return _finish(t.visit(tree), t.changed)


def swap_sign(source: str, metric: str) -> str | None:
    """The simulation with a sign swapped in the expression returned for ``metric``: its first ``+``/``-`` (or ``*``/``/``)
    exchanged, else the whole expression negated."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    pairs = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}

    def edit(old: ast.expr) -> ast.expr:
        for node in ast.walk(old):
            if isinstance(node, ast.BinOp) and type(node.op) in pairs:
                node.op = pairs[type(node.op)]()
                return old
        return ast.UnaryOp(op=ast.USub(), operand=old)

    t = _Metric(metric, edit)
    return _finish(t.visit(tree), t.changed)


def coarse_equals_fine(source: str, setting: str, fine: Any) -> str | None:
    """The simulation in which every level of the refinement knob ``setting`` is read as its finest value ``fine``, so a
    coarse and a fine run are identical."""
    return ignore_setting(source, setting, fine)


def mutants(source: str, settings: dict[str, Any], metrics: Iterable[str], knobs: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """Every faulty copy of ``source`` FI can make, as ``(name, source)``: each setting ignored (``settings``: name -> the
    default to use), each metric hard-coded, clipped to a bound, with a sign swapped, and each refinement knob (``knobs``:
    name -> its finest value) made identical at every level. A mutant that does not change the code or does not compile is
    left out. Deterministic: the same inputs give the same list."""
    out: list[tuple[str, str]] = []

    def add(name: str, text: str | None) -> None:
        if text and text.strip() != source.strip():
            out.append((name, text))

    for s, default in settings.items():
        add(f"ignore_setting:{s}", ignore_setting(source, s, default))
    for m in metrics:
        add(f"hard_code:{m}", hard_code_metric(source, m))
        add(f"clip:{m}", clip_metric(source, m))
        add(f"swap_sign:{m}", swap_sign(source, m))
    for k, fine in (knobs or {}).items():
        add(f"coarse_equals_fine:{k}", coarse_equals_fine(source, k, fine))
    return out
