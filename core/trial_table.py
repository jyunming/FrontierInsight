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

from core import stats as _stats
from core.result_checks import left_out_by_record as _left_out_by_record

#: The table goes into the writer's evidence only up to this many rows; a longer one stays in its file.
WRITER_MAX_ROWS = 200


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
            "metrics": {str(name): summary(_values(metric)) for name, metric in c["metrics"].items()}}
        entry.update({k: c[k] for k in ("planned", "ok", "failed")
                      if isinstance(c.get(k), int) and not isinstance(c.get(k), bool)})
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


def kept_values(summarised: list[dict[str, Any]]) -> list[str]:
    """The value names the table reports, in the order they first appear: not a trial index, not one number in every
    trial of every setting (a setting echoed back, a constant)."""
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for c in summarised:
        for name, s in c["metrics"].items():
            by_metric.setdefault(name, []).append(s)
    return [name for name, per_cell in by_metric.items()
            if not all(s["index"] for s in per_cell if s["n"]) and not all(s["constant"] for s in per_cell)]


def rows_of(summarised: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """:func:`rows` from :func:`summaries`."""
    out: list[dict[str, Any]] = []
    for name in kept_values(summarised):
        for c in summarised:
            s = c["metrics"].get(name)
            if not s or not s["n"]:
                continue
            lower, upper, method = interval(s)
            out.append({"setting": c["key"], "metric": name, "n": s["n"], "mean": s["mean"], "sd": s["sd"],
                        "ci_lower": lower, "ci_upper": upper, "method": method if lower is not None else "",
                        "cell": c["cell"]})
    return out


def rows(quest_root: Path) -> list[dict[str, Any]]:
    """One row per setting and per value the trials returned: ``{"setting", "metric", "n", "mean", "sd",
    "ci_lower", "ci_upper", "method", "cell"}``; ``[]`` without a trial record."""
    return rows_of(summaries(_record(quest_root)))


def _num(x: Any) -> str:
    return "" if x is None else f"{x:.6g}"


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
        lines.append(f"| {r['setting']} | {r['metric']} | {r['n']} | {_num(r['mean'])} | {_num(r['sd'])} | {interval} "
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
