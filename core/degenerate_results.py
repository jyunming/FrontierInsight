"""Results and figures that show nothing are broken, not findings.

After the run, before the paper, from FI's own records (the per-cell record and the figure records the run's plotting
recorder wrote; never from the script's claims):

* a figure panel that draws no data (no line, no bar, no image), and a plotted series whose points are all at one x
  although it has several (a trend drawn against a column that never varied);
* a reported quantity that is exactly the same at every setting while the study varies its settings and other reported
  quantities do vary (a quantity the simulation does not compute from its inputs);
* a reported quantity stuck at its own largest or smallest value for most settings (a clamp, not a measurement).

Only what the analysis reports is judged (a quantity nobody reports is a diagnostic), a setting echoed back is not a
result, and a value of exactly 0 or 1 is never read as stuck at a bound (an indicator or a probability is allowed to be
there). A quantity the plan expects to stay put (what an invariant, symmetry or second-implementation check measures when
that is one bare name, a check of any other kind whose bare measure the plan does not list as a metric, a declared bound) is
exempt.

Only a quantity the plan itself says is estimated from the data (:func:`plan_estimates`: the plan's metrics, the names of a
known value, the metric the run's size is set for) is questioned: sent back, then left
out. Any other flat quantity (a theory value that depends on the fixed design only, a design constant, a count) is kept in the
results and the limitations say, from FI's record, that it was identical at all the settings (:func:`kept_note`). The
all-identical case is :mod:`core.flat_output`'s; this is the quantity that is flat while others are not.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import reported_settings as _rs

#: A series is read only when it has at least this many points (a two-point line is a reference line).
MIN_POINTS = 3
#: A quantity is read only when the study has this many settings.
MIN_CELLS = 3
#: A bound must be reached by at least this share of the settings, and at least this many of them.
SATURATED_SHARE = 0.5
SATURATED_MIN_CELLS = 6
SATURATED_MIN_DISTINCT = 3


@dataclass
class FigureProblem:
    figure: str
    panel: str
    kind: str  # "empty" | "one_x"
    detail: str = ""

    def says(self, supplied: list[str] | None = None) -> str:
        where = f"`{self.figure}`" + (f", panel \"{self.panel}\"" if self.panel else "")
        if self.kind == "empty":
            return f"{where} draws no data"
        return f"{where} plots {self.detail}, so it shows no trend"


@dataclass
class FigureGap:
    """A series of a drawn figure that has no point at some settings the figure otherwise shows (or has no value there)."""

    figure: str
    panel: str
    series: str
    absent: list[float]  # the x values with no point
    why: str  # "no value" (None / NaN at that x) or "not plotted" (the panel's other series have a point there)

    def says(self) -> str:
        where = f"`{self.figure}`" + (f", panel \"{self.panel}\"" if self.panel else "")
        at = ", ".join(f"{x:g}" for x in self.absent[:8]) + (" ..." if len(self.absent) > 8 else "")
        what = f"the series \"{self.series}\"" if self.series and not self.series.startswith("_") else "a series"
        if self.why == "no value":
            return f"{where}: {what} has no value at x = {at}, so the figure shows no point there"
        return f"{where}: {what} has no point at x = {at}, where the panel's other series do"


@dataclass
class QuantityProblem:
    name: str
    kind: str  # "constant" | "saturated"
    value: float
    cells: int
    at: int

    def says(self, supplied: list[str] | None = None) -> str:
        if self.kind == "constant":
            return (f"`{self.name}` is {self.value:g} at every one of the {self.cells} settings although other results "
                    "change with them")
        return (f"`{self.name}` is {self.value:g} at {self.at} of the {self.cells} settings, the largest or smallest value it "
                "takes anywhere")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


_DASHED = ("--", ":", "-.", "dashed", "dotted", "dashdot")


def _is_reference(line: Any) -> bool:
    """A recorded line that is a reference drawn on purpose: ``axhline`` / ``axvline``, or a dashed line at one value
    with no markers. A constant drawn with markers is a measured series (each marker is a setting), not a reference."""
    if not isinstance(line, dict):
        return False
    if line.get("kind") in ("hline", "vline"):
        return True
    y = line.get("y")
    return (line.get("kind") == "data" and line.get("linestyle") in _DASHED and isinstance(y, list) and len(y) > 1
            and str(line.get("marker") or "None") in ("None", "none", "", " ")
            and len(_rs._distinct([float(v) for v in y])) == 1)


def _has_data(panel: Any) -> bool:
    if not isinstance(panel, dict):
        return False
    lines = panel.get("lines") if isinstance(panel.get("lines"), list) else []
    return bool(panel.get("draws_other")) or any(not _is_reference(line) for line in lines)


def _shares_place_with_data(panels: list[Any], i: int) -> bool:
    """Another panel in the same place holds data (a twin axis): the same cell of a grid, or the same box on the figure
    for axes made without a grid."""
    me = panels[i] if isinstance(panels[i], dict) else {}
    keys = [k for k in ("grid", "place") if me.get(k) is not None]
    return bool(keys) and any(j != i and isinstance(p, dict) and any(p.get(k) == me[k] for k in keys) and _has_data(p)
                              for j, p in enumerate(panels))


#: A panel's series are compared point for point only when the longer of the two has at most this many points (a grid of
#: settings, not a curve drawn on a dense axis).
MAX_GRID_POINTS = 12


def _numbers(values: Any) -> list[float]:
    out: list[float] = []
    for v in values if isinstance(values, list) else []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f not in out:
            out.append(f)
    return out


def figure_gaps(records_dir: Path, figures: list[str]) -> list[FigureGap]:
    """Where a series of a figure has no point at a setting: from the figure's own record (``<stem>.seed0.json``), two ways.

    * the line holds a missing value (None / NaN) at an x: matplotlib draws nothing there (``missing_x``, counted by the plot
      recorder from the plotted data);
    * the series' x values are a strict part of another series' x values in the same panel, both few enough to be a grid of
      settings (at most :data:`MAX_GRID_POINTS`): the panel's other series have a point where this one has none.

    Reference lines are never compared. Nothing here changes the figure or the results; it says what the figure does not show."""
    out: list[FigureGap] = []
    for name in sorted(set(figures)):
        panels = _read(records_dir / f"{Path(name).stem}.seed0.json").get("axes")
        for panel in panels if isinstance(panels, list) else []:
            if not isinstance(panel, dict):
                continue
            lines = [ln for ln in (panel.get("lines") if isinstance(panel.get("lines"), list) else [])
                     if isinstance(ln, dict) and ln.get("kind") == "data" and not _is_reference(ln)]
            title = str(panel.get("title") or "")
            # Compared at 9 significant digits: the same setting written as 0.1 and as np.arange's 0.1000000000000001.
            xs = [{float(f"{v:.9g}") for v in _numbers(ln.get("x"))} for ln in lines]
            for k, ln in enumerate(lines):
                label = str(ln.get("label") or "")
                gone = _numbers(ln.get("missing_x"))
                if gone:
                    out.append(FigureGap(Path(name).name, title, label, gone, "no value"))
                    continue
                if not xs[k] or label.startswith("_"):
                    continue
                lacking: set[float] = set()
                for j, other in enumerate(lines):
                    if j != k and not str(other.get("label") or "").startswith("_") and xs[k] < xs[j] and len(xs[j]) <= MAX_GRID_POINTS:
                        lacking |= xs[j] - xs[k]
                if lacking:
                    out.append(FigureGap(Path(name).name, title, label, sorted(lacking), "not plotted"))
    return out


def gap_note(gaps: list[FigureGap]) -> str:
    """For the writer: where a figure has no point, and what that means for the sentences about it."""
    shown = "; ".join(g.says() for g in gaps[:6])
    return (f"Where the figures show no point: {shown}. Do not say a figure shows, confirms or reaches a value at those "
            "settings, and do not write \"as shown in Figure N\" for them: where the results hold the value, give it in the text or a "
            "table; where they do not, say it was not computed; and say in the caption which settings the figure has no "
            "point for.")


def figure_problems(records_dir: Path, figures: list[str]) -> list[FigureProblem]:
    """The empty panels and the single-x series of the figures in ``figures`` (file names), from their records."""
    out: list[FigureProblem] = []
    for name in sorted(set(figures)):
        stem = Path(name).stem
        main, lines = _read(records_dir / f"{stem}.json"), _read(records_dir / f"{stem}.seed0.json")
        panels = lines.get("axes") if isinstance(lines.get("axes"), list) else []
        shown = main.get("axes") if isinstance(main.get("axes"), list) else []
        if not panels and not shown:
            continue
        for i, panel in enumerate(panels or [{} for _ in shown]):
            labelled = shown[i] if i < len(shown) and isinstance(shown[i], dict) else {}
            title = str(panel.get("title") or labelled.get("title") or "")
            named = bool(title or panel.get("xlabel") or panel.get("ylabel") or labelled.get("xlabel"))
            drawn = panel.get("lines") if isinstance(panel.get("lines"), list) else []
            # A panel that only holds a legend (shared by the other panels) is not empty.
            if (not drawn and not labelled.get("series") and not panel.get("draws_other") and not panel.get("legend")
                    and named):
                out.append(FigureProblem(Path(name).name, title, "empty"))
                continue
            # A panel whose only lines are reference lines (a threshold, a known value) holds no result: the axes are
            # there and the data are not. Read from the lines themselves, since an unlabelled data line is not among the
            # labelled series; and not when another panel sits in the same place (a twin axis carries the data).
            if (drawn and all(_is_reference(line) for line in drawn) and not panel.get("draws_other") and named
                    and not _shares_place_with_data(panels, i)):
                out.append(FigureProblem(Path(name).name, title, "empty"))
                continue
            for line in drawn:
                x, y = line.get("x"), line.get("y")
                if line.get("kind") != "data" or not isinstance(x, list) or len(x) < MIN_POINTS or not isinstance(y, list):
                    continue
                if len(_rs._distinct([float(v) for v in x])) == 1 and len(_rs._distinct([float(v) for v in y])) > 1:
                    label = str(line.get("label") or "")
                    out.append(FigureProblem(Path(name).name, title, "one_x",
                                             f"{len(x)} points all at x = {float(x[0]):g}"
                                             + (f" ({label})" if label and not label.startswith("_") else "")))
    return out


def _means(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Each returned quantity's mean at every setting, only for those every setting returned."""
    names = set.intersection(*[set(c["metrics"]) for c in cells]) if cells else set()
    return {n: [sum(c["metrics"][n]) / len(c["metrics"][n]) for c in cells] for n in sorted(names)}


def keys_in(value: Any) -> set[str]:
    """Every dictionary key, at any depth, of what the analysis reported."""
    out: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(str(k))
            out |= keys_in(v)
    elif isinstance(value, list):
        for v in value[:300]:
            out |= keys_in(v)
    return out


def quantity_problems(cells: list[dict[str, Any]], reported: set[str], exempt: set[str] | frozenset[str] = frozenset()
                      ) -> list[QuantityProblem]:
    """The constant and the stuck quantities among those the analysis reported (``reported``: the keys it printed or wrote).
    A number per run is judged over the settings and a number per value over the settings and values, each over the records
    that hold it (:func:`core.in_run.groups`)."""
    from . import in_run as _in_run

    return [q for group in _in_run.groups(cells) for q in _quantity_problems_in(group, reported, exempt)]


def _quantity_problems_in(cells: list[dict[str, Any]], reported: set[str], exempt: set[str] | frozenset[str]
                          ) -> list[QuantityProblem]:
    varying = _rs._varying(cells)
    seen: dict[str, set[str]] = {}
    for c in cells:   # the values compared inside each run vary a record too, numbers or not (a procedure)
        for k, v in (c.get("level") or {}).items():
            seen.setdefault(str(k), set()).add(str(v))
    for k, vs in seen.items():
        if len(vs) > 1:
            varying.setdefault(k, sorted(vs))  # type: ignore[arg-type]
    if len(cells) < MIN_CELLS or not varying:
        return []
    means = {n: v for n, v in _means(cells).items() if all(math.isfinite(x) for x in v)}
    axes = list(varying)

    def is_reported(name: str) -> bool:
        return any(name == k or _rs.names_setting(name, k) or _rs.names_setting(k, name) for k in reported)

    kept = {n: v for n, v in means.items()
            if n not in exempt and is_reported(n) and not any(_rs.names_setting(n, a) for a in axes)}
    moving = [n for n, v in kept.items() if len(_rs._distinct(v)) > 1]
    out: list[QuantityProblem] = []
    for name, values in kept.items():
        distinct = _rs._distinct(values)
        if len(distinct) == 1:
            if moving:
                out.append(QuantityProblem(name, "constant", distinct[0], len(values), len(values)))
            continue
        lo, hi = min(values), max(values)
        for bound in (hi, lo):
            at = sum(1 for v in values if _rs.same(v, bound))
            if (len(values) >= SATURATED_MIN_CELLS and len(distinct) >= SATURATED_MIN_DISTINCT
                    and at / len(values) >= SATURATED_SHARE and not (_rs.same(bound, 0.0) or _rs.same(bound, 1.0))):
                out.append(QuantityProblem(name, "saturated", bound, len(values), at))
                break
    return out


#: The kinds of check whose measured quantity is expected to be the same everywhere (a worst violation of an invariant).
_EXPECTED_KINDS = ("invariant", "symmetry", "second_implementation")
#: The kinds of check whose measured quantity is a value the plan expects to learn from the data (a special case, a published value).
_MEASURED_KINDS = ("special_case", "published_value")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _same_value(a: Any, b: Any) -> bool:
    if (isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return str(a).strip().lower() == str(b).strip().lower()


def _is_off(value: Any) -> bool:
    """A numeric 0 (no noise, no spread): the axis switched off."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _runs_on_fixed_inputs(oracle: dict[str, Any], protocol: dict[str, Any] | None) -> bool:
    """Whether a check's ``case`` names a value of a study axis (``grid`` or ``in_each_run``) that the axis does not run: the
    check then runs on its own fixed inputs, not on a setting of the study, and what it measures is the same whatever the
    setting. A case that is one of the study's settings (each axis it names at a value the axis runs), one with no case, and
    one that names no axis are not. One exception, an added rule and not a derived fact: a 0 on an axis whose values are all
    above 0 (``nsr = 0`` beside 0.01 to 0.2) is that axis switched off, the noise-free end of the study itself, and the
    estimate it measures there is still computed by the study's own pipeline (``ea_est``); it is not told apart from a
    setting."""
    case = oracle.get("case")
    p = protocol if isinstance(protocol, dict) else {}
    if not isinstance(case, dict):
        return False
    for block in ("grid", "in_each_run"):
        axes = p.get(block)
        for axis, values in (axes.items() if isinstance(axes, dict) else ()):
            if axis not in case or not isinstance(values, (list, tuple)) or any(_same_value(case[axis], v) for v in values):
                continue
            numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if _is_off(case[axis]) and numbers and len(numbers) == len(values) and all(v > 0 for v in numbers):
                continue
            return True
    return False


def expected_constants(protocol: dict[str, Any] | None) -> set[str]:
    """Names the plan itself expects to stay put: what a check of an invariant, a symmetry or a second implementation measures
    when the measure IS that one bare name (``bh_scipy_disagree``); what a special-case or published-value check measures when
    that is one bare name and the check runs on its own fixed inputs, a case that is no setting of the study
    (:func:`_runs_on_fixed_inputs`: ``bh_fixed_rejections`` at ``m = 4`` when the study runs ``m`` at 20, 200 and 2000), unless
    the plan lists the name as an estimate of its own; and the keys of the plan's declared bounds or limits. A name that is only
    an operand of a measure's expression (``abs(Ea_hat - Ea_polyfit)``) is not exempt: what the check measures is the
    expression, and ``Ea_hat`` in it is still the plan's own estimate."""
    from . import oracle_check as _oracle

    own = _own_estimates(protocol)
    out: set[str] = set()
    for o in _oracle.declared(protocol):
        measure = str(o.get("measure") or "").strip()
        if not _IDENT.fullmatch(measure):
            continue
        kind = _oracle.kind_of(o)
        if kind in _EXPECTED_KINDS or (kind in _MEASURED_KINDS and measure not in own and _runs_on_fixed_inputs(o, protocol)):
            out.add(measure)
    for key in ("bounds", "limits"):
        block = (protocol or {}).get(key) if isinstance(protocol, dict) else None
        if isinstance(block, dict):
            out |= {str(k) for k in block}
    return out


def _own_estimates(protocol: dict[str, Any] | None) -> set[str]:
    """The names the plan lists as estimates itself: each metric's id, the names of a metric's declared known value (its
    estimate, its standard error, the two ends of its interval) and the metric the run's size is set for."""
    p = protocol if isinstance(protocol, dict) else {}
    out: set[str] = set()
    for m in p.get("metrics") if isinstance(p.get("metrics"), list) else []:
        if not isinstance(m, dict):
            continue
        if m.get("id"):
            out.add(str(m["id"]))
        known = m.get("known_value")
        if isinstance(known, dict):
            out |= {str(known[k]) for k in ("estimate", "standard_error") if known.get(k)}
            out |= {str(v) for v in known.get("interval") or [] if v}
    precision = p.get("precision")
    if isinstance(precision, dict) and precision.get("metric"):
        out.add(str(precision["metric"]))
    return out


def plan_estimates(protocol: dict[str, Any] | None) -> set[str]:
    """Names the frozen plan says are estimated or measured from the trials, from fields it already has: what it lists itself
    (:func:`_own_estimates`) and what a special-case or published-value check measures when that is one bare name (an
    expression is not a name) and the check's case is a setting of the study or names none; a check on its own fixed inputs
    measures a constant (:func:`expected_constants`). A theory value, a design constant or a count is not among them."""
    from . import oracle_check as _oracle

    out = _own_estimates(protocol)
    for o in _oracle.declared(protocol):
        measure = str(o.get("measure") or "").strip()
        if _oracle.kind_of(o) in _MEASURED_KINDS and _IDENT.fullmatch(measure) and not _runs_on_fixed_inputs(o, protocol):
            out.add(measure)
    return out


def split_flat(quantities: list[QuantityProblem], protocol: dict[str, Any] | None
               ) -> tuple[list[QuantityProblem], list[QuantityProblem]]:
    """``(judged, kept)``: a flat quantity is judged (sent back, then left out) only when the plan says it is estimated from
    the data (:func:`plan_estimates`, the same name matching as for a setting); any other flat quantity is kept and stated in
    the limitations from FI's record. A theory value, a design constant and a count are flat by nature, and a name the plan
    never ties to the data is not a name FI can say is wrong."""
    named = plan_estimates(protocol)
    judged = [q for q in quantities if any(q.name == n or _rs.names_setting(q.name, n) or _rs.names_setting(n, q.name) for n in named)]
    return judged, [q for q in quantities if q not in judged]


def directive(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    """What to tell the repair: plain, naming each figure and quantity. The instruction is the last thing in it."""
    lines = [f"- {p.says()}" for p in (*quantities[:6], *figures[:6])]
    text = "FI looked at what the run produced, and some of it shows nothing:\n" + "\n".join(lines)
    if quantities:
        text += ("\n\nThe plan says each quantity named above is estimated from the trials' data: compute it from this trial's "
                 "data as the plan defines it (no fixed number, no clamp).")
    if figures:
        text += ("\n\nDraw each panel from the saved results (every panel needs data, each point at the setting it was run at); "
                 "a panel with nothing to show is removed, not left empty.")
    return text + " Change nothing else."


def kept_says(q: QuantityProblem) -> str:
    """One kept flat quantity, in plain words, from FI's own record of the trials."""
    if q.kind == "constant":
        return f"`{q.name}` was identical ({q.value:g}) at all {q.cells} settings"
    return f"`{q.name}` was at {q.value:g}, the largest or smallest value it takes anywhere, at {q.at} of the {q.cells} settings"


def kept_note(kept: list[QuantityProblem]) -> str:
    """For the writer: flat quantities the plan does not say are estimated from the data stay in the results; the limitations
    say what FI saw."""
    shown = "; ".join(kept_says(q) for q in kept[:6])
    one = len(kept) == 1
    return (f"FI's record of the trials shows that {shown}. The plan does not say {'it is' if one else 'they are'} estimated from "
            f"the data, so {'it stays' if one else 'they stay'} in the results. Say plainly, in the limitations, "
            f"what the record shows of {'it' if one else 'each of them'}, as stated here.")


def plain_note(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    return "the run's results show nothing in places (" + "; ".join(p.says() for p in (*quantities[:3], *figures[:3])) + ")"


def writer_note(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    gone = []
    if quantities:
        gone.append("the quantities " + ", ".join(f"`{q.name}`" for q in quantities[:6]))
    if figures:
        gone.append("the figures " + ", ".join(sorted({f"`{p.figure}`" for p in figures})))
    return (f"{plain_note(figures, quantities)}. {' and '.join(gone).capitalize()} {'are' if len(gone) > 1 or len(quantities) > 1 else 'is'} "
            "left out of the results. Do not report, describe or quote anything from "
            f"{'them' if len(gone) > 1 or len(quantities) + len({p.figure for p in figures}) > 1 else 'it'}: say plainly, in the "
            "limitations, that it was left out because it showed nothing.")
