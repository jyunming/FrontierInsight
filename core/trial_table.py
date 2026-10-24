"""FI's own table of what every trial gave, per setting, from its trial record (``raw/trials.json``).

The analysis script and the analysis model both summarise the trials, and the writer copies numbers from the analysis:
two model copies between a trial's value and the paper. A real quest's analysis gave an SD of 1,920 where the trials
give 1,849.0, and the paper printed it. This table is computed by FI from the trial values themselves, with no model
in between, so a number the paper quotes from it is a number the run produced.

What it holds, and what it does not: for each setting and each value ``run_trial`` returned, the trials that gave it
(n), their mean and SD, and a 95% interval of the mean -- a Wilson interval when every trial gave 0 or 1 (a
proportion), otherwise a t interval. A quantity the analysis derives from several values (a coverage from an estimate
and its truth, a power from two counts) is not here; the table covers what the trials themselves recorded. A value
that is the same in every trial of every setting (a setting echoed back, a constant) and a trial index are left out:
they have no spread to report.

Written to ``paper/results_table.md`` beside the paper and given to the writer while it is short enough.
"""

from __future__ import annotations

import hashlib
import json
import math
from array import array
from pathlib import Path
from typing import Any

from core import in_run as _in_run
from core import stats as _stats
from core.result_checks import left_out_by_record as _left_out_by_record

#: The table goes into the writer's evidence only up to this many rows; a longer one stays in its file.
WRITER_MAX_ROWS = 200

#: A row's ``status`` when no number stands in it. "not run": no trial ran for that setting (the record holds no value
#: for it and no trial that succeeded), so there is nothing to describe, not even a failure. "no finite value": trials
#: ran and every value they gave was NaN, infinite or missing (an overflow, a divergence); the row carries how many
#: and the reasons the runs gave. Told apart here, from FI's own record, because a model reading the analysis sees
#: ``None`` for both and has labelled a setting that never ran as an overflow.
NOT_RUN = "not run"
NO_FINITE = "no finite value"


def _record(quest_root: Path) -> list[dict[str, Any]]:
    """The cells of ``raw/trials.json`` that carry metrics, without a quantity FI left out of the results."""
    try:
        data = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list):
        return []
    gone = set(_left_out_by_record(quest_root))  # the one source of what FI left out (core/result_checks.py)
    out = [c for c in cells if isinstance(c, dict) and isinstance(c.get("metrics"), dict)]
    if not gone:
        return out
    # FI's own record (raw/trials.json) is unchanged; what the writer and the reader are shown has no left-out quantity.
    return [{**c, "metrics": {k: v for k, v in c["metrics"].items() if str(k) not in gone}} for c in out]


def _values(metric: Any) -> list[float]:
    raw = metric.get("values") if isinstance(metric, dict) else metric
    if not isinstance(raw, (list, tuple)):
        return []
    return [float(v) for v in raw
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]


def _is_index(values: list[float]) -> bool:
    return len(values) > 1 and sorted(values) == [float(i) for i in range(len(values))]


def summary(values: list[float]) -> dict[str, Any]:
    """What the table needs of one value's trials in one setting, without the trials themselves: ``n``, ``mean``,
    ``sd``, whether every trial gave 0 or 1 (``binary``, with their ``successes``), whether the values are a trial index
    (``index``), whether they are all one number (``constant``), and a ``digest`` of the values (two settings with the
    same digest gave the very same trials: copies, not independent runs)."""
    mean, sd, n = _stats.mean_std(values)
    binary = bool(values) and all(v in (0.0, 1.0) for v in values)
    return {"n": n, "mean": mean, "sd": sd, "binary": binary, "successes": sum(values) if binary else None,
            "index": _is_index(values), "constant": len(set(values)) <= 1,
            "digest": hashlib.sha256(array("d", values).tobytes()).hexdigest()[:16]}


def missing(metric: Any) -> dict[str, Any]:
    """How many of a result's runs gave no number (``missing``) and why, as the runs said (``why_missing``:
    ``{reason: runs}``, :data:`core.in_run.WHY_MISSING`); ``{}`` when every run gave one."""
    raw = metric.get("values") if isinstance(metric, dict) else metric
    if not isinstance(raw, (list, tuple)):
        return {}
    count = sum(1 for v in raw if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))))
    if not count:
        return {}
    why = metric.get(_in_run.WHY_MISSING) if isinstance(metric, dict) else None
    return {"missing": count, **({"why_missing": dict(why)} if isinstance(why, dict) and why else {})}


def status_of(s: dict[str, Any], counts: dict[str, Any]) -> str:
    """``""`` for a value some trials gave; :data:`NO_FINITE` when trials ran and none gave a finite number (``missing``
    says how many, ``why_missing`` why); :data:`NOT_RUN` when the setting has no value at all and no trial that
    succeeded or failed (``counts``: the record's ``ok`` and ``failed``, both present and 0). A setting with no value
    and no such counts, or with trials that succeeded or failed, has no status."""
    if s.get("n"):
        return ""
    if s.get("missing"):
        return NO_FINITE
    ok, failed = counts.get("ok"), counts.get("failed")
    # "Not run" claims that no trial ran, so only the record's own counts may say it: ok and failed both present, both 0.
    # Without counts (an older record) a setting with no value may as well be one whose every trial failed.
    return NOT_RUN if ok == 0 and failed == 0 and isinstance(ok, int) and isinstance(failed, int) else ""


def not_run_summary() -> dict[str, Any]:
    """The summary of a value for a setting that never ran (see :data:`NOT_RUN`)."""
    return {**summary([]), "status": NOT_RUN}


def summaries(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each record of ``raw/trials.json`` as ``{"cell", "key", "level", "metrics": {name: summary}}`` (see
    :func:`summary`), in the record's order, with the record's counts of runs (``planned``, ``ok``, ``failed``) when it
    has them. A record with no metrics (the per-value records a study that compares values inside each run writes with
    nothing in them) is kept, with an empty ``metrics``."""
    out: list[dict[str, Any]] = []
    for c in cells:
        cell, level = c.get("cell"), c.get("level")
        setting: dict[str, Any] = dict(cell) if isinstance(cell, dict) else {}
        label = str(c.get("key") or ", ".join(f"{k}={v}" for k, v in setting.items()) or "the one setting")
        entry: dict[str, Any] = {
            "cell": setting, "key": label, "level": dict(level) if isinstance(level, dict) and level else None,
            "metrics": {}}
        entry.update({k: c[k] for k in ("planned", "ok", "failed")
                      if isinstance(c.get(k), int) and not isinstance(c.get(k), bool)})
        for name, metric in c["metrics"].items():
            one = {**summary(_values(metric)), **missing(metric)}
            one["status"] = status_of(one, entry)
            entry["metrics"][str(name)] = one
        # The record says in numbers that no trial of this setting succeeded or failed: it never ran, whatever values
        # (none) it holds. It then stands in every value's table as "not run", though it names none of them.
        entry["not_run"] = entry.get("ok") == 0 and entry.get("failed") == 0
        out.append(entry)
    return out


def interval(s: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """``(lower, upper, method)`` of the 95% interval of the mean from a :func:`summary`: Wilson when every trial gave
    0 or 1, otherwise a t interval (the same numbers :func:`core.stats.wilson_interval` and
    :func:`core.stats.confidence_interval` give from the values)."""
    n = int(s.get("n") or 0)
    if n <= 0:
        return None, None, ""
    if s.get("binary"):
        ci = _stats.wilson_interval(s["successes"], n)
        return (ci[0], ci[1], "Wilson") if ci else (None, None, "")
    if n < 2:
        return None, None, ""
    half = _stats._t95(n - 1) * (s["sd"] / math.sqrt(n))
    return s["mean"] - half, s["mean"] + half, "t interval of the mean"


def per_setting_precision(quest_root: Path, metric: str, target: float) -> dict[str, Any] | None:
    """How the plan's precision target (a 95% interval of the mean no wider than ``target`` either side) holds for
    ``metric`` in each setting of FI's trial record, the interval worked out from that setting's own trials: ``{"target",
    "metric", "settings"`` (every setting of the record, so a setting that never gave a value is in the count),
    ``"met"`` (those within the target), ``"widest"`` (the widest half-width, ``-1`` when no setting has an interval),
    ``"widest_setting"``, ``"without_interval"`` (settings whose trials give none: a single trial), ``"without_value"``
    (names of the settings that ran no trial or gave no finite value for it: never counted as met)``}``.
    Settings are never pooled: their true values differ, and one interval over them all would be narrower than any of
    theirs. ``None`` when no setting has an interval for it."""
    met = settings = without = 0
    no_value: list[str] = []
    widest, widest_at = -1.0, ""
    judged = 0
    for c in summaries(_record(quest_root)):
        s = c["metrics"].get(metric)
        if not c["metrics"] and not c.get("planned"):
            continue  # a per-value record with nothing in it is not a setting
        settings += 1
        if not s or not s["n"]:
            no_value.append(c["key"])
            continue
        lower, upper, _method = interval(s)
        if lower is None or upper is None:
            without += 1
            continue
        half = (upper - lower) / 2.0
        judged += 1
        met += half <= target
        if half > widest:
            widest, widest_at = half, c["key"]
    if not judged:
        return None
    return {"target": float(target), "metric": metric, "settings": settings, "met": met, "widest": widest,
            "widest_setting": widest_at, "without_interval": without, "without_value": no_value}


def kept_values(summarised: list[dict[str, Any]]) -> list[str]:
    """The value names the table reports, in the order they first appear: not a trial index, not one number in every
    trial of every setting (a setting echoed back, a constant)."""
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for c in summarised:
        for name, s in c["metrics"].items():
            by_metric.setdefault(name, []).append(s)
    out = []
    for name, per_cell in by_metric.items():
        shown = [s for s in per_cell if s["n"]]
        if not shown:  # no setting gave a number: kept only when the table can say why (not run / no finite value)
            if any(s.get("status") for s in per_cell):
                out.append(name)
        elif not all(s["index"] for s in shown) and not all(s["constant"] for s in shown):
            out.append(name)
    return out


def rows_of(summarised: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """:func:`rows` from :func:`summaries`."""
    out: list[dict[str, Any]] = []
    for name in kept_values(summarised):
        for c in summarised:
            s = c["metrics"].get(name)
            if s is None and c.get("not_run"):
                s = not_run_summary()  # the setting never ran, so it names none of the values it would have given
            if not s or not (s["n"] or s.get("status")):
                continue
            lower, upper, method = interval(s)
            row = {"setting": c["key"], "metric": name, "n": s["n"], "mean": s["mean"] if s["n"] else None,
                   "sd": s["sd"] if s["n"] else None, "ci_lower": lower, "ci_upper": upper,
                   "method": method if lower is not None else "", "cell": c["cell"], "status": s.get("status") or ""}
            if row["status"] == NO_FINITE:
                row.update({k: s[k] for k in ("missing", "why_missing") if k in s})
            out.append(row)
    return out


def rows(quest_root: Path) -> list[dict[str, Any]]:
    """One row per setting and per value the trials returned: ``{"setting", "metric", "n", "mean", "sd",
    "ci_lower", "ci_upper", "method", "cell"}``; ``[]`` without a trial record."""
    return rows_of(summaries(_record(quest_root)))


def _num(x: Any) -> str:
    return "" if x is None else f"{x:.6g}"


def _status_text(r: dict[str, Any]) -> str:
    """The mean cell of a row with no number: ``not run``, or ``no finite value (overflow: 3)`` -- the reasons the runs
    gave, each with the runs it covers (``no finite value (3 runs)`` when they gave none)."""
    if r.get("status") != NO_FINITE:
        return str(r.get("status") or "")
    why = r.get("why_missing") or {}
    if why:
        return f"{NO_FINITE} (" + "; ".join(f"{k}: {v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])) + ")"
    count = int(r.get("missing") or 0)
    return f"{NO_FINITE} ({count} run{'s' if count != 1 else ''})" if count else NO_FINITE


def markdown(table: list[dict[str, Any]]) -> str:
    """The rows as a Markdown table, with a line saying what they are; ``""`` for no rows."""
    if not table:
        return ""
    lines = [
        "Computed by FI from the value every trial returned (raw/trials.json), with no model in between: per setting, "
        "the trials that gave the value (n), their mean and SD, and a 95% interval of the mean (Wilson when every trial "
        "gave 0 or 1, otherwise a t interval). Quantities the analysis derives from several values are not here.",
        "",
        "| setting | value | n | mean | SD | 95% interval of the mean | interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in table:
        interval = f"{_num(r['ci_lower'])} to {_num(r['ci_upper'])}" if r["ci_lower"] is not None else ""
        mean = _status_text(r) if r.get("status") else _num(r["mean"])
        lines.append(f"| {r['setting']} | {r['metric']} | {r['n']} | {mean} | {_num(r['sd'])} | {interval} "
                     f"| {r['method']} |")
    return "\n".join(lines) + "\n"


def write(quest_root: Path) -> list[dict[str, Any]]:
    """Write ``paper/results_table.md`` from the trial record and return its rows; nothing is written without one."""
    table = rows(quest_root)
    if table:
        paper = Path(quest_root) / "paper"
        paper.mkdir(parents=True, exist_ok=True)
        (paper / "results_table.md").write_text("# What every trial gave, per setting\n\n" + markdown(table),
                                                encoding="utf-8")
    return table


def write_note(table: list[dict[str, Any]]) -> str:
    """The writer's note: the table itself, while it has at most ``WRITER_MAX_ROWS`` rows; ``""`` otherwise (a longer
    one stays in its file)."""
    if not table or len(table) > WRITER_MAX_ROWS:
        return ""
    return ("FI's table of what every trial gave, per setting (also in paper/results_table.md). Take a number about "
            "these values from here rather than working it out again:\n\n" + markdown(table))
