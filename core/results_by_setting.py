"""FI's results table: every setting the study computed, each quantity as value +/- uncertainty beside its reference.

The writer chose which settings to report, and a 125-cell study printed 9 of them with dashes for the rest: its trial
table had 2,001 rows, more than the writer is given (core/trial_table.py ``WRITER_MAX_ROWS``), so the writer saw only
the analysis. This module builds, from FI's own records and with no model in between:

* **The distinct settings.** A grid cell is not always a distinct setting: a grid axis the simulation never reads
  (a quest's ``temperature_K``: each run fits all five temperatures, so the axis only copied every setting five times)
  gives cells that differ in nothing the simulation sees. :func:`unread_axes` reads the code with ``ast``
  (:func:`core.grid_keys.reads`, which follows a read inside a closure or a lambda; FI's own runner, ``code/run.py``,
  is not the study's code) and returns the grid axes it found no read of. Only when the reading is sure: when the code
  uses a setting in a way that cannot be followed (passes it whole to an unknown function, spreads it with ``**``,
  reads it with a computed key) or a file does not parse, every axis counts as read. :func:`distinct` pools the cells
  that differ only in unread axes -- only when their means, their spread and their shares of failed runs also agree
  within the pooled uncertainty (a read the reading missed shows up there), and counting cells that gave the very same
  trials (the same seeds) once.
* **One table per quantity** (:func:`tables`): a row per distinct setting, its settings, ``n``, the mean +/- the half-width
  of its 95% interval (Wilson for a 0/1 value, else t, as core/trial_table.py), the SD, the reference FI computed from
  the frozen plan (core/reference_values.py) and the comparison (:func:`comparison`): ``agrees`` (the reference is
  inside the interval) or ``differs``; for a limit, ``within bound``, ``within bound, within uncertainty`` (the
  interval reaches the limit), ``above bound`` or ``above bound, within uncertainty`` -- a bound is never said to
  agree; and ``too few trials to judge`` instead of an agreement when the interval is too wide to tell. A quantity is
  a value of a declared metric the trials return (its reference names the value); a metric computed from several
  values (an SD, a correlation) is not a value of the trials and is named in a note.
* **Renderings** (pure functions): :func:`markdown` for the checks and the writer, :func:`latex` (booktabs) for the
  PDF, :func:`csv_text` for every row, :func:`reference_section` / :func:`reference_section_latex` for the paragraph
  of reference values in Methods.
* **What the writer gets** (:func:`writer_note`): the full trial table while it fits; otherwise these tables while they
  fit; otherwise a slice (the first, middle and last value of each axis) with the file named; otherwise only the file.
"""

from __future__ import annotations

import ast
import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import grid_keys as _grid_keys
from . import oracle_forms as _forms
from . import reference_values as _refs
from . import trial_table as _trial_table

#: A cell whose mean is further than this many standard errors from the pooled mean is not pooled with the others: the
#: reading of the code said the axis is not read, the numbers say it matters.
POOL_Z = 5.0
#: FI's own scripts at the top of code/: they run the study, they are not the study's code (a file of the same name in a
#: package of the study is).
_FI_SCRIPTS = frozenset({"run.py", "fi_search.py"})
_HARMLESS_CALLS = frozenset({"print", "repr", "str", "isinstance", "len", "id", "type", "bool", "sorted"})


# --- which grid axes the simulation reads ----------------------------------------------------------------------------


def _followed(node: ast.AST, parents: dict[int, ast.AST], defs: dict[str, list[str]]) -> bool:
    """Whether a use of a setting is one the reading of the code follows: a read by a literal key, a copy into a
    name, a hand-over to the trial contract's entry point or to a function whose parameter is a setting, or a use that
    reads no value (printed, a presence test)."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str)
    if isinstance(parent, ast.Attribute) and parent.value is node:
        call = parents.get(id(parent))
        if parent.attr in ("get", "pop", "setdefault") and isinstance(call, ast.Call) and call.func is parent:
            return bool(call.args) and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
        if parent.attr == "copy" and isinstance(call, ast.Call):
            return isinstance(parents.get(id(call)), ast.Assign)
        return False
    if isinstance(parent, ast.Call) and node in parent.args:
        callee = parent.func.id if isinstance(parent.func, ast.Name) else None
        if callee == "dict":
            return isinstance(parents.get(id(parent)), ast.Assign)
        if callee in _HARMLESS_CALLS or callee in _grid_keys.ENTRIES:
            return True
        params = defs.get(callee or "")
        index = parent.args.index(node)  # type: ignore[arg-type]
        return params is not None and index < len(params) and params[index] in _grid_keys.SETTING_NAMES
    if isinstance(parent, ast.Assign) and parent.value is node:
        return all(isinstance(t, ast.Name) for t in parent.targets)
    if isinstance(parent, ast.Compare):
        return all(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
    return isinstance(parent, (ast.FormattedValue, ast.Expr))


def opaque_uses(source: str, script: str) -> list[str]:
    """Where ``source`` uses a setting in a way :func:`core.grid_keys.reads` cannot follow (``script line N``)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    defs = {n.name: [a.arg for a in (*n.args.posonlyargs, *n.args.args)]
            for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    out: list[str] = []
    for scope, names in _grid_keys._scopes(tree):
        names = set(names)
        nodes = sorted(_grid_keys._own_nodes(scope), key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
        for node in nodes:
            if isinstance(node, ast.Assign) and _grid_keys._setting_like(node.value, names):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id in _grid_keys.SETTING_NAMES:
                names.add(node.target.id)
        for node in nodes:
            holder = (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in names) \
                or _grid_keys._is_record_get(node)
            if holder and not _followed(node, parents, defs):
                out.append(f"{script} line {getattr(node, 'lineno', 0)}")
    return out


def unread_axes(protocol: dict[str, Any] | None, files: dict[str, str],
                also_read: set[str] | None = None) -> tuple[list[str], str]:
    """``(the grid axes FI found no read of, why every axis counts as read when it is empty)``. ``files``: the study's
    code (:func:`core.grid_keys.code_files`); FI's own scripts at the top of code/ are skipped. ``also_read``: names that
    count as read anyway (an axis a reference formula uses)."""
    axes = list(_refs.grid_axes(protocol))
    if not axes:
        return [], "the plan has no grid"
    study = {k: v for k, v in files.items() if Path(k).as_posix() not in _FI_SCRIPTS}
    if not study:
        return [], "there is no code to read"
    for script, text in study.items():
        try:
            ast.parse(text)
        except (SyntaxError, ValueError):
            return [], f"FI could not read {script}"
    opaque = [where for script, text in study.items() for where in opaque_uses(text, script)]
    if opaque:
        return [], f"the code uses a setting in a way FI cannot follow ({', '.join(opaque[:3])})"
    read: set[str] = set(also_read or ())
    for script, text in study.items():
        for r in _grid_keys.reads(text, script):
            read.update(r.keys)
    return [a for a in axes if a not in read], ""


# --- pooling cells into distinct settings ----------------------------------------------------------------------------


def pooled(items: list[dict[str, Any]]) -> dict[str, Any]:
    """One summary (core/trial_table.py :func:`~core.trial_table.summary`) of independent sets of trials: the counts
    added, the mean weighted by them, the SD of all the trials together."""
    items = [s for s in items if s and s.get("n")]
    if not items:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "binary": False, "successes": None, "index": False, "constant": True,
                "digest": ""}
    if len(items) == 1:
        return dict(items[0])
    n = sum(int(s["n"]) for s in items)
    mean = sum(s["n"] * s["mean"] for s in items) / n
    ss = sum((s["n"] - 1) * s["sd"] ** 2 + s["n"] * (s["mean"] - mean) ** 2 for s in items)
    binary = all(s.get("binary") for s in items)
    return {"n": n, "mean": mean, "sd": math.sqrt(ss / (n - 1)) if n > 1 else 0.0, "binary": binary,
            "successes": sum(s["successes"] for s in items) if binary else None, "index": False,
            "constant": all(s.get("constant") for s in items) and len({s["mean"] for s in items}) == 1,
            "digest": "+".join(sorted(s["digest"] for s in items))}


def _differs(items: list[dict[str, Any]], whole: dict[str, Any]) -> bool:
    """Whether cells' summaries of one value differ by more than chance: a mean further than :data:`POOL_Z` standard
    errors from the pooled mean, or (for a value that is not 0/1, whose spread its mean fixes) an SD further than
    :data:`POOL_Z` standard errors of an SD from the cells' pooled SD (an axis such as a noise level changes the spread
    and not the mean)."""
    for s in items:
        se = whole["sd"] / math.sqrt(s["n"])
        if se == 0:
            if s["mean"] != whole["mean"]:
                return True
        elif abs(s["mean"] - whole["mean"]) / se > POOL_Z:
            return True
    spread = [s for s in items if s["n"] > 1 and not s.get("binary")]
    dof = sum(s["n"] - 1 for s in spread)
    if len(spread) > 1 and dof > 0:
        within = math.sqrt(sum((s["n"] - 1) * s["sd"] ** 2 for s in spread) / dof)
        for s in spread:
            if within == 0:
                if s["sd"] != 0:
                    return True
            elif abs(s["sd"] - within) / (within / math.sqrt(2 * (s["n"] - 1))) > POOL_Z:
                return True
    return False


def _failures_differ(members: list[dict[str, Any]]) -> bool:
    """Whether the cells' shares of failed runs differ by more than :data:`POOL_Z` standard errors of the pooled share
    (cells whose records give no counts are not compared)."""
    counted = [(int(c["failed"]), int(c["planned"])) for c in members
               if isinstance(c.get("failed"), int) and isinstance(c.get("planned"), int) and c["planned"] > 0]
    if len(counted) < 2:
        return False
    share = sum(f for f, _ in counted) / sum(p for _, p in counted)
    if share in (0.0, 1.0):
        return False
    return any(abs(f / p - share) / math.sqrt(share * (1 - share) / p) > POOL_Z for f, p in counted)


def _label(setting: dict[str, Any]) -> str:
    from .trial_runner import _fmt

    return ",".join(f"{k}={_fmt(v)}" for k, v in setting.items()) or "the one setting"


@dataclass
class Setting:
    """One distinct setting: its settings (the read axes, and a value compared inside each run), the grid cells it
    pools, and a pooled summary per value."""

    setting: dict[str, Any]
    label: str
    cells: list[str]
    metrics: dict[str, dict[str, Any]]
    copies: int = 0  # cells left out of the counts because they gave the very same trials as another


@dataclass
class Distinct:
    settings: list[Setting]
    unread: list[str]
    notes: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)


def distinct(summarised: list[dict[str, Any]], protocol: dict[str, Any] | None, unread: list[str]) -> Distinct:
    """The distinct settings of a trial record (:func:`core.trial_table.summaries`), pooling the cells that differ only
    in ``unread`` axes (see the module docstring)."""
    records = [c for c in summarised if c.get("metrics")]
    values = _trial_table.kept_values(records)
    unread_set = set(unread)
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in records:
        setting = {k: v for k, v in (c.get("cell") or {}).items() if k not in unread_set}
        groups.setdefault(json.dumps(setting, sort_keys=False, default=str), []).append(c)
    notes: list[str] = []
    out: list[Setting] = []
    kept_apart: list[str] = []
    copied = joined = 0
    for members in groups.values():
        setting = {k: v for k, v in (members[0].get("cell") or {}).items() if k not in unread_set}
        merged: dict[str, dict[str, Any]] = {}
        apart = len(members) > 1 and _failures_differ(members)
        dropped = 0
        for name in values if not apart else ():
            seen: dict[str, dict[str, Any]] = {}
            for c in members:
                s = c["metrics"].get(name)
                if s and s.get("n") and s.get("digest") not in seen:
                    seen[s["digest"]] = s
            items = list(seen.values())
            dropped = max(dropped, sum(1 for c in members if c["metrics"].get(name, {}).get("n")) - len(items))
            whole = pooled(items)
            if len(items) > 1 and _differs(items, whole):
                apart = True
                break
            merged[name] = whole
        if apart and len(members) > 1:
            kept_apart.append(_label(setting))
            for c in members:
                own = dict(c.get("cell") or {})
                out.append(Setting(own, _label(own), [c["key"]], {n: dict(c["metrics"][n]) for n in values
                                                                  if c["metrics"].get(n)}))
            continue
        copied += dropped
        joined += len(members) > 1
        out.append(Setting(setting, _label(setting), [c["key"] for c in members], merged, dropped))
    axes = " and ".join(f"`{a}`" for a in unread)
    if joined:
        notes.append(f"FI found no read of {axes} in the simulation's code, so grid cells that differ only there are one "
                     "setting here: their trials are pooled and n counts them all.")
    if copied:
        notes.append("Some pooled cells gave exactly the same trials as another cell (the same random seeds); each such "
                     "set of trials is counted once.")
    if kept_apart:
        lead = (f"The cells of {len(kept_apart)} setting(s) were not pooled" if joined else
                f"FI found no read of {axes} in the simulation's code, but its cells were not pooled")
        notes.append(f"{lead}, because their means, spread or failed runs differ by more than {POOL_Z:g} standard "
                     f"errors: {', '.join(kept_apart[:3])}.")
    return Distinct(out, list(unread), notes, values)


# --- the tables --------------------------------------------------------------------------------------------------------


@dataclass
class Table:
    """A table as data, rendered by :func:`markdown` and :func:`latex`."""

    label: str
    caption: str
    columns: list[str]
    align: list[str]
    rows: list[list[str]]
    value: str = ""


def _sig(x: float, digits: int = 4) -> str:
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    if exp >= 6 or exp < -4:
        return f"{x:.{digits - 1}e}"
    decimals = max(0, digits - 1 - exp)
    text = f"{x:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_pm(mean: float, lower: float | None, upper: float | None) -> str:
    """``mean +/- half-width``: the half-width (the larger side of an uneven interval) to two significant figures, the
    mean to the same place."""
    if lower is None or upper is None:
        return _sig(mean)
    half = max(mean - lower, upper - mean, 0.0)
    if half == 0:
        return f"{_sig(mean, 6)} ± 0"
    place = math.floor(math.log10(half)) - 1  # two significant figures of the half-width
    half_r = round(half, -place)
    if half_r and math.floor(math.log10(half_r)) - 1 > place:  # 0.0996 rounds up to 0.10
        place += 1
        half_r = round(half, -place)
    decimals = max(0, -place)
    return f"{round(mean, -place):.{decimals}f} ± {half_r:.{decimals}f}"


#: A mean is not said to agree with its reference, or to be within a bound, from fewer trials than this ...
MIN_TRIALS_TO_JUDGE = 10
#: ... or when its 95% interval reaches further than this share of the reference either side of the mean (a reference
#: of 0 has no scale: there only the count of trials applies).
MAX_HALF_WIDTH_SHARE = 0.25
TOO_FEW = "too few trials to judge"


def comparison(kind: str, ref: float | None, mean: float, lower: float | None, upper: float | None,
               n: int | None = None) -> str:
    """The plain verdict of a mean and its 95% interval (Wilson for a 0/1 value) from ``n`` trials against a reference.

    ``agrees`` / ``differs`` for a value; for a limit (``bound``: at most, ``lower_bound``: at least) ``within bound``
    when the whole interval is on the right side of it, ``within bound, within uncertainty`` when the mean is and the
    interval reaches past it, and ``above bound`` / ``below bound`` (with ``, within uncertainty`` when the interval
    reaches back). An agreement, or a mean within a bound, from too few trials or with too wide an interval is
    :data:`TOO_FEW` instead: the interval cannot tell it from a disagreement."""
    if ref is None:
        return "no reference"
    if lower is None or upper is None:
        return "not judged (no interval)"
    too_wide = (n is not None and n < MIN_TRIALS_TO_JUDGE) or (
        ref != 0 and max(mean - lower, upper - mean) > MAX_HALF_WIDTH_SHARE * abs(ref))
    # Rounding: a Wilson interval of 4000 successes in 4000 ends a hair under 1, and 1 - 0.95**2000 is 1 in floats.
    slack = 1e-12 * max(1.0, abs(ref))
    lower, upper = lower - slack, upper + slack
    if kind == "bound":
        if mean > ref:
            return "above bound" if lower > ref else "above bound, within uncertainty"
        return TOO_FEW if too_wide else ("within bound" if upper - 2 * slack <= ref else
                                         "within bound, within uncertainty")
    if kind == "lower_bound":
        if mean < ref:
            return "below bound" if upper < ref else "below bound, within uncertainty"
        return TOO_FEW if too_wide else ("within bound" if lower + 2 * slack >= ref else
                                         "within bound, within uncertainty")
    if lower <= ref <= upper:
        return TOO_FEW if too_wide else "agrees"
    return "differs"


def _ref_text(kind: str, ref: float | None) -> str:
    if ref is None:
        return "no reference"
    return {"bound": "≤ ", "lower_bound": "≥ "}.get(kind, "") + _sig(ref)


def _axis_text(v: Any) -> str:
    from .trial_runner import _fmt

    if isinstance(v, float) and not v.is_integer():
        return _sig(v, 6)
    return _fmt(v)


@dataclass
class Built:
    """Everything this module builds for one quest."""

    distinct: Distinct
    references: dict[str, Any]
    tables: list[Table]
    notes: list[str]
    rows: list[dict[str, Any]]  # one per quantity and distinct setting, every number (for the CSV and the checks)


def _quantities(protocol: dict[str, Any] | None, values: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """``([(value name, metric id)], metric ids that are no value of the trials)``. Without declared metrics, or when
    none is a value of the trials, every value the trial table keeps."""
    from . import metric_spec

    out: list[tuple[str, str]] = []
    left: list[str] = []
    for m in metric_spec.declared(protocol):
        name = _refs.value_name(m)
        if name in values and all(name != v for v, _ in out):
            out.append((name, m["id"]))
        elif name not in values:
            left.append(m["id"])
    if not out:
        return [(v, "") for v in values], left
    return out, left


def build(summarised: list[dict[str, Any]], protocol: dict[str, Any] | None, files: dict[str, str],
          judged: list[dict[str, Any]] | None = None) -> Built:
    """The distinct settings, the references at each and one table per quantity, from the trial record's summaries,
    the frozen protocol, the study's code and the engine's verdicts on the known-answer checks."""
    metrics = (protocol or {}).get("metrics") if isinstance(protocol, dict) else None
    formula_names: set[str] = set()
    for m in metrics if isinstance(metrics, list) else []:
        ref = m.get("reference") if isinstance(m, dict) else None
        if isinstance(ref, dict) and ref.get("formula") is not None:
            formula_names.update(_forms.names(str(ref["formula"]), {**_forms.FUNCTIONS, **_forms.SPECIAL_FUNCTIONS}))
    unread, why = unread_axes(protocol, files, formula_names)
    found = distinct(summarised, protocol, unread)
    notes = list(found.notes)
    if why and _refs.grid_axes(protocol):
        notes.append(f"Every grid axis is treated as read: {why}.")
    quantities, left = _quantities(protocol, found.values)
    computed = _refs.compute(protocol, [s.setting for s in found.settings], set(found.values), judged)
    by_metric = {q["metric"]: q for q in computed["quantities"]}
    at: dict[tuple[str, str], dict[str, Any]] = {}
    for r in computed["by_setting"]:
        at[(r["metric"], json.dumps(r["setting"], default=str))] = r
    if left:
        notes.append("Not in these tables, because no trial returns them (they are computed from several values): "
                     + ", ".join(f"`{m}`" for m in left) + ".")
    axes = [a for a in _refs.grid_axes(protocol) if a not in set(found.unread)]
    levels = list(dict.fromkeys(k for s in found.settings for k in s.setting if k not in axes))
    columns_axes = axes + levels
    tables: list[Table] = []
    rows: list[dict[str, Any]] = []
    for value, metric in quantities:
        q = by_metric.get(metric) if metric else None
        kind = str((q or {}).get("kind") or "value")
        stats = []
        for s in found.settings:
            summary = s.metrics.get(value)
            if summary and summary.get("n"):
                lower, upper, method = _trial_table.interval(summary)
            else:
                lower = upper = None
                method = ""
            stats.append((s, summary, lower, upper, method))
        status = (q or {}).get("status", "no reference")
        why_none = (q or {}).get("why") or ("the plan declares no metric for it" if not metric else "")
        if q and status == "computed":
            raw: Any = next((m.get("reference") for m in metrics or [] if isinstance(m, dict) and m.get("id") == metric),
                            {})
            covered = []
            for s, summary, lower, upper, _ in stats:
                r = at.get((metric, json.dumps(s.setting, default=str)))
                if r and r.get("reference") is not None and summary and summary.get("n") and lower is not None:
                    covered.append((summary["mean"], max(summary["mean"] - lower, upper - summary["mean"]),
                                    float(r["reference"])))
            if _refs.copies_measured(raw if isinstance(raw, dict) else {}, covered, _refs.inputs(protocol)):
                status = "no reference"
                why_none = ("its reference gives the measured means more closely than they are known, so it was read "
                            "off the run, not worked out")
                q.update(status=status, why=why_none)
                computed["by_setting"] = [r for r in computed["by_setting"] if r["metric"] != metric]
        binary = all((summary or {}).get("binary", True) for _, summary, *_ in stats if summary and summary.get("n"))
        has_ref = status == "computed"
        cols = [*columns_axes, "n", f"{value}", *([] if binary else ["SD"]), *(["reference", "comparison"] if has_ref else [])]
        align = ["r"] * len(columns_axes) + ["r", "r"] + ([] if binary else ["r"]) + (["r", "l"] if has_ref else [])
        body: list[list[str]] = []
        methods: set[str] = set()
        for s, summary, lower, upper, method in stats:
            cells = [_axis_text(s.setting.get(a, "")) if a in s.setting else "" for a in columns_axes]
            ref_entry = at.get((metric, json.dumps(s.setting, default=str))) if metric else None
            ref = ref_entry.get("reference") if (ref_entry and has_ref) else None
            if summary and summary.get("n"):
                methods.add(method)
                mean = summary["mean"]
                cells += [str(summary["n"]), format_pm(mean, lower, upper)]
                if not binary:
                    cells.append(_sig(summary["sd"]))
                verdict = comparison(kind, ref, mean, lower, upper, int(summary["n"]))
            else:
                cells += ["0", "no value"]
                if not binary:
                    cells.append("")
                mean = None
                verdict = "no value" if ref is not None else "no reference"
            if has_ref:
                cells += [_ref_text(kind, ref), verdict]
            body.append(cells)
            rows.append({"value": value, "metric": metric, "setting": s.label, **s.setting,
                         "n": (summary or {}).get("n", 0), "mean": mean, "sd": (summary or {}).get("sd"),
                         "ci_lower": lower, "ci_upper": upper, "interval": method, "reference": ref,
                         "reference_kind": kind if ref is not None else "", "comparison": verdict})
        how = " and ".join(sorted(m for m in methods if m)) or "no interval"
        caption = (f"`{value}` at each of the {len(found.settings)} settings: mean ± half-width of its 95% interval "
                   f"({how}) over the n trials that returned it.")
        if has_ref:
            ref_q = q or {}
            caption += (f" Reference: `{ref_q.get('formula')}`"
                        f"{' (' + ref_q['note'] + ')' if ref_q.get('note') else ''}, computed by FI from the plan frozen "
                        f"before the run; {ref_q.get('checked', {}).get('why') or 'not cross-checked'}.")
            caption += (" `agrees`: the reference lies inside the interval (about 1 setting in 20 differs by chance "
                        "alone)." if kind == "value" else
                        " A bound is a limit, not a value: `within bound` means the whole interval stays on its side, "
                        "`within bound, within uncertainty` that the mean does and the interval reaches past it.")
            caption += (f" `{TOO_FEW}`: fewer than {MIN_TRIALS_TO_JUDGE} trials, or an interval wider than "
                        f"{MAX_HALF_WIDTH_SHARE:.0%} of the reference either side.")
        else:
            caption += f" No reference: {why_none}."
        tables.append(Table(label=f"fi-results-{re.sub(r'[^A-Za-z0-9]+', '-', value).strip('-').lower()}",
                            caption=caption, columns=cols, align=align, rows=body, value=value))
    if not quantities:
        notes.append("The trials returned no value that varies, so there is nothing to tabulate.")
    return Built(found, computed, tables, notes, rows)


# --- renderings (pure) -------------------------------------------------------------------------------------------------


def _md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def markdown(table: Table) -> str:
    """One table in Markdown: its caption line, then the table."""
    lines = [f"**Table ({table.value}).** {table.caption}", "",
             "| " + " | ".join(_md_cell(c) for c in table.columns) + " |",
             "|" + "|".join("---:" if a == "r" else "---" for a in table.align) + "|"]
    lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in table.rows]
    return "\n".join(lines) + "\n"


_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"$\pm$", "≤": r"$\le$",
                "≥": r"$\ge$", "×": r"$\times$", "π": r"$\pi$"}


def tex_escape(text: str) -> str:
    """``text`` as LaTeX prints it literally (a backticked name becomes ``\\texttt``)."""
    out = []
    parts = re.split(r"(`[^`]*`)", str(text))
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            out.append(r"\texttt{" + "".join(_TEX_SPECIAL.get(ch, ch) for ch in part[1:-1]) + "}")
        else:
            out.append("".join(_TEX_SPECIAL.get(ch, ch) for ch in part))
    return "".join(out)


def latex(table: Table) -> str:
    """One table as a LaTeX ``table`` float with booktabs rules (needs ``\\usepackage{booktabs}``)."""
    spec = "".join(table.align)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             r"\caption{" + tex_escape(table.caption) + "}", r"\label{tab:" + table.label + "}",
             r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in table.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in row) + r" \\" for row in table.rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def csv_text(built: Built) -> str:
    """Every row of every table, every number at full precision."""
    keys: list[str] = []
    for r in built.rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for r in built.rows:
        writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return buf.getvalue()


def _reference_sentences(built: Built) -> list[str]:
    out: list[str] = []
    for q in built.references.get("quantities") or []:
        name = q.get("value") or q.get("metric")
        if q.get("status") == "computed":
            kind = {"bound": "an upper limit", "lower_bound": "a lower limit"}.get(q.get("kind"), "the value it should take")
            note = f", {q['note']}" if q.get("note") else ""
            out.append(f"`{name}`: `{q.get('formula')}` ({kind}{note}); {q.get('checked', {}).get('why')}.")
        else:
            out.append(f"`{name}`: no reference ({q.get('why')}).")
    return out


def reference_section(built: Built) -> str:
    """The paragraph of reference values for Methods, in Markdown; ``""`` when the plan declares no metrics."""
    sentences = _reference_sentences(built)
    if not sentences:
        return ""
    return ("**Reference values.** FI computed the value each quantity should take at each setting from formulas fixed "
            "in the plan before the run, from the settings, fixed settings and thresholds alone (never from the "
            "results). " + " ".join(sentences) + "\n")


def reference_section_latex(built: Built) -> str:
    """:func:`reference_section` as a LaTeX paragraph."""
    text = reference_section(built)
    if not text:
        return ""
    body = text.replace("**Reference values.** ", "", 1).strip()
    return r"\paragraph{Reference values.} " + tex_escape(body) + "\n"


def _slice(built: Built) -> list[Table]:
    """The tables cut to the settings at the first, middle and last value of each axis that takes more than three."""
    keep_values: dict[str, set[str]] = {}
    axes: dict[str, list[Any]] = {}
    for s in built.distinct.settings:
        for k, v in s.setting.items():
            axes.setdefault(k, [])
            if v not in axes[k]:
                axes[k].append(v)
    for k, vals in axes.items():
        try:
            ordered = sorted(vals)
        except TypeError:
            ordered = list(vals)
        picked = ordered if len(ordered) <= 3 else [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        keep_values[k] = {json.dumps(v, default=str) for v in picked}
    keep = [all(json.dumps(v, default=str) in keep_values[k] for k, v in s.setting.items()) for s in built.distinct.settings]
    return [Table(t.label, t.caption + " Only the first, middle and last value of each setting are shown here.",
                  t.columns, t.align, [r for r, k in zip(t.rows, keep) if k], t.value) for t in built.tables]


def writer_note(trial_rows: list[dict[str, Any]], built: Built | None, cap: int | None = None) -> str:
    """What the writer is given of FI's tables: the full trial table while it has at most ``cap`` rows (default
    :data:`core.trial_table.WRITER_MAX_ROWS`); otherwise the per-setting tables while they fit; otherwise their slice at
    the first, middle and last value of each setting, naming the file with every row; otherwise only the file."""
    cap = _trial_table.WRITER_MAX_ROWS if cap is None else cap
    if trial_rows and len(trial_rows) <= cap:
        return _trial_table.write_note(trial_rows)
    if built is None or not built.tables:
        return ""
    head = (f"FI's tables of every setting the study computed ({len(built.distinct.settings)} distinct settings; "
            "also in paper/results_by_setting.md, every number in paper/results_by_setting.csv). Each value is the mean "
            "± the half-width of its 95% interval; the reference is computed by FI from the frozen plan. Take a number "
            "about these values from here rather than working it out again.")
    extra = "\n".join(built.notes)
    total = sum(len(t.rows) for t in built.tables)
    if total <= cap:
        return head + ("\n\n" + extra if extra else "") + "\n\n" + "\n".join(markdown(t) for t in built.tables)
    cut = _slice(built)
    if 0 < sum(len(t.rows) for t in cut) <= cap:
        return (head + f" They have {total} rows, too many for this note, so here is a slice." +
                ("\n\n" + extra if extra else "") + "\n\n" + "\n".join(markdown(t) for t in cut))
    return (head + f" They have {total} rows, too many for this note even as a slice: quote a setting's numbers only "
            "from those files." + ("\n\n" + extra if extra else ""))


# --- the files, and the pools for the provenance check -----------------------------------------------------------------


def from_quest(quest_root: Path, protocol: dict[str, Any] | None,
               judged: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], Built | None]:
    """``(FI's trial table rows, what this module builds)`` from the quest's trial record, read once; ``([], None)``
    without one."""
    cells = _trial_table._record(Path(quest_root))
    if not cells:
        return [], None
    summarised = _trial_table.summaries(cells)
    built = build(summarised, protocol, _grid_keys.code_files(Path(quest_root) / "code"), judged)
    return _trial_table.rows_of(summarised), built


def write(quest_root: Path, protocol: dict[str, Any] | None, judged: list[dict[str, Any]] | None = None) -> Built | None:
    """Build from the quest's records and write ``paper/results_by_setting.md`` / ``.tex`` / ``.csv`` and
    ``paper/reference_values.json``; ``None`` without a trial record."""
    _rows, built = from_quest(quest_root, protocol, judged)
    if built is None:
        return None
    paper = Path(quest_root) / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    notes = "\n".join(built.notes)
    (paper / "results_by_setting.md").write_text(
        "# Every setting the study computed\n\n" + (reference_section(built) + "\n" if reference_section(built) else "")
        + (notes + "\n\n" if notes else "") + "\n".join(markdown(t) for t in built.tables), encoding="utf-8")
    (paper / "results_by_setting.tex").write_text(
        reference_section_latex(built) + "\n".join(latex(t) for t in built.tables), encoding="utf-8")
    (paper / "results_by_setting.csv").write_text(csv_text(built), encoding="utf-8")
    (paper / "reference_values.json").write_text(json.dumps(built.references, indent=2, default=str) + "\n",
                                                 encoding="utf-8")
    return built


def traceable(built: Built) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """``(the pooled settings' numbers by ``label.value``, each label's settings that tell it apart)`` for the
    provenance check's per-setting table pool (:func:`core.number_provenance.check` ``trial_table`` and
    ``trial_settings``)."""
    numbers: dict[str, dict[str, Any]] = {}
    for r in built.rows:
        if r.get("mean") is None:
            continue
        numbers[f"{r['setting']}.{r['value']}"] = {k: r[k] for k in ("n", "mean", "sd", "ci_lower", "ci_upper")
                                                   if r.get(k) is not None}
    return numbers, varying_settings({s.label: s.setting for s in built.distinct.settings})


def provenance_inputs(trial_rows: list[dict[str, Any]], built: Built | None) -> dict[str, Any]:
    """The keyword arguments of :func:`core.number_provenance.check` that hold FI's own per-setting numbers:
    ``trial_table`` (every grid cell, and every pooled setting), ``trial_settings`` (each label's settings that tell it
    apart) and ``references`` (the reference values FI computed)."""
    table = _trial_table.traceable(trial_rows)
    settings = varying_settings(_trial_table.settings(trial_rows))
    references: dict[str, float] = {}
    if built is not None:
        numbers, pooled_settings = traceable(built)
        table = {**table, **numbers}
        settings = {**settings, **pooled_settings}
        references = _refs.pool(built.references)
    return {"trial_table": table, "trial_settings": settings, "references": references}


def varying_settings(settings: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each label's settings, cut to those that take more than one value across ``settings`` (only they tell a row
    apart)."""
    values: dict[str, set[str]] = {}
    for s in settings.values():
        for k, v in s.items():
            values.setdefault(k, set()).add(json.dumps(v, default=str))
    varying = {k for k, vs in values.items() if len(vs) > 1}
    return {label: {k: v for k, v in s.items() if k in varying} for label, s in settings.items()}
