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

Written to ``paper/results_table.md`` beside the paper, given to the writer while it is short enough, and read by
the provenance check as numbers the run computed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core import stats as _stats

#: The table goes into the writer's evidence only up to this many rows; a longer one stays in its file.
WRITER_MAX_ROWS = 200


def _record(quest_root: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list):
        return []
    return [c for c in cells if isinstance(c, dict) and isinstance(c.get("metrics"), dict)]


def _values(metric: Any) -> list[float]:
    raw = metric.get("values") if isinstance(metric, dict) else metric
    if not isinstance(raw, (list, tuple)):
        return []
    return [float(v) for v in raw
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]


def _is_index(values: list[float]) -> bool:
    return len(values) > 1 and sorted(values) == [float(i) for i in range(len(values))]


def rows(quest_root: Path) -> list[dict[str, Any]]:
    """One row per setting and per value the trials returned: ``{"setting", "metric", "n", "mean", "sd",
    "ci_lower", "ci_upper", "method"}``; ``[]`` without a trial record."""
    cells = _record(quest_root)
    by_metric: dict[str, list[tuple[str, list[float]]]] = {}
    for c in cells:
        setting = c.get("cell") if isinstance(c.get("cell"), dict) else {}
        label = str(c.get("key") or ", ".join(f"{k}={v}" for k, v in setting.items()) or "the one setting")
        for name, metric in c["metrics"].items():
            by_metric.setdefault(str(name), []).append((label, _values(metric)))
    out: list[dict[str, Any]] = []
    for name, per_cell in by_metric.items():
        if all(_is_index(v) for _label, v in per_cell if v):
            continue
        if all(len(set(v)) <= 1 for _label, v in per_cell):
            continue
        for label, values in per_cell:
            if not values:
                continue
            mean, sd, n = _stats.mean_std(values)
            if all(v in (0.0, 1.0) for v in values):
                ci = _stats.wilson_interval(sum(values), n)
                lower, upper, method = (ci[0], ci[1], "Wilson") if ci else (None, None, "")
            else:
                ci = _stats.confidence_interval(values)
                lower, upper, method = ci["ci_lower"], ci["ci_upper"], "t interval of the mean"
            out.append({"setting": label, "metric": name, "n": n, "mean": mean, "sd": sd,
                        "ci_lower": lower, "ci_upper": upper, "method": method if lower is not None else ""})
    return out


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


def traceable(table: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The rows by ``setting.value``, for the provenance check: numbers the run computed."""
    return {f"{r['setting']}.{r['metric']}": {k: r[k] for k in ("n", "mean", "sd", "ci_lower", "ci_upper")
                                               if r[k] is not None}
            for r in table}


def write_note(table: list[dict[str, Any]]) -> str:
    """The writer's note: the table itself, while it has at most ``WRITER_MAX_ROWS`` rows; ``""`` otherwise (a longer
    one stays in its file and in the provenance check)."""
    if not table or len(table) > WRITER_MAX_ROWS:
        return ""
    return ("FI's table of what every trial gave, per setting (also in paper/results_table.md). Take a number about "
            "these values from here rather than working it out again:\n\n" + markdown(table))
