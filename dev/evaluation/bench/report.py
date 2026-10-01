"""The benchmark's report: ``self_benchmark.json`` (one object; what a web or VS Code page will show) and
``self_benchmark.md``."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import catalogue
from . import score as _score

RESULTS_SCHEMA = "fi.bench.results/v1"


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def results(outcomes: list[dict[str, Any]], *, note: str = "") -> dict[str, Any]:
    """Everything the report shows, as one JSON-ready object."""
    return {"schema": RESULTS_SCHEMA, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "fi_commit": _commit(),
            "note": note, "summary": _score.summarize(outcomes), "runs": [_score.public(o) for o in outcomes]}


def _pct(rate: dict[str, Any]) -> str:
    if not rate.get("n"):
        return "no runs"
    ci = rate.get("ci95")
    span = f" (95% CI {ci[0]:.0%} to {ci[1]:.0%})" if ci else ""
    return f"{rate['k']}/{rate['n']} = {rate['rate']:.0%}{span}"


def markdown(data: dict[str, Any], *, title: str = "FI self-benchmark") -> str:
    s = data["summary"]
    lines = [f"# {title}", "",
             f"Generated {data['generated_at']}" + (f" on FI {data['fi_commit']}" if data.get("fi_commit") else "")
             + f"; {s['runs']} run(s).", ""]
    if data.get("note"):
        lines += [data["note"], ""]
    lines += [
        "**Wrong results let through** (a planted error that changes the answer, and FI still would publish): "
        + _pct(s["false_pass"]["total"]) + ".",
        "",
        "**Right results held back** (a clean run with the right answer that FI would not publish): "
        + _pct(s["false_block"]) + ".",
        "",
        "**Wrong among the published** (clean runs FI would publish whose answer is wrong): "
        + _pct(s["error_after_publication"]) + ".",
        "",
    ]
    if s["false_pass"]["not_valid"]:
        lines += [f"{s['false_pass']['not_valid']} planted run(s) are not counted (the last column of the runs table "
                  "says why: an equivalent change, no usable control, or a replay that did not follow its recording).", ""]
    if s["false_pass"]["by_error"]:
        lines += ["## By planted error", "", "| Error | What was planted | Let through |", "|---|---|---|"]
        for err, rate in s["false_pass"]["by_error"].items():
            lines.append(f"| {err} | {catalogue.get(err).name} | {_pct(rate)} |")
        lines.append("")
    if s["detection"]:
        gates = [g for g in catalogue.GATES if any(g in row for row in s["detection"].values())]
        extra = [k for k in ("control_also", "none") if any(k in row for row in s["detection"].values())]
        lines += ["## Where each error was first caught", "",
                  "`control_also`: held back only by a check its control failed the same way; `none`: no check "
                  "flagged it.", "",
                  "| Error | " + " | ".join(gates + extra) + " | caught at all |",
                  "|---|" + "---|" * (len(gates) + len(extra) + 1)]
        for err, row in s["detection"].items():
            total = sum(v for k, v in row.items() if k != "any")
            lines.append(f"| {err} | " + " | ".join(str(row.get(g, 0)) for g in gates + extra)
                         + f" | {row.get('any', 0)}/{total} |")
        lines.append("")
    if s["calibration"]:
        lines += ["## How often the answer is right at each evidence level", "", "| Level | Runs | Right | Rate |",
                  "|---|---|---|---|"]
        for level, c in s["calibration"].items():
            rate = f"{c['rate']:.0%}" if "rate" in c else "(fewer than 10 runs: counts only)"
            lines.append(f"| {level} | {c['n']} | {c['correct']} | {rate} |")
        lines.append("")
    lines += ["## Runs", "", "| Run | Task | Planted | Mode | Answer | Evidence level | Would publish | First caught by | Counted |",
              "|---|---|---|---|---|---|---|---|---|"]
    for o in data["runs"]:
        counted = ("yes" if o["valid"] else f"no: {o.get('not_counted_because') or 'not judged'}") if o["error"] else "-"
        planted = o["error"] or ("(control)" if o["role"] == "control" else "(clean)")
        lines.append(f"| {o['run']} | {o['task']} | {planted} | {o['mode'] or ''} | "
                     f"{_score.answer_line(o).replace('|', '/')} | {o['evidence_level'] or ('stopped: ' + o['stopped'][:60] if o['stopped'] else '-')} | "
                     f"{'yes' if o['would_publish'] else 'no'} | {o['first_gate'] or '-'} | {counted} |")
    c = s["cost"]
    lines += ["", f"Model calls: {c['calls']}; tokens: {c['tokens']:,}; time: {c['seconds']:.0f} s; "
              f"calls a replay had no recording for: {c['divergences']}.", ""]
    return "\n".join(lines)


def write(outcomes: list[dict[str, Any]], out_dir: Path, *, name: str = "self_benchmark", note: str = "",
          title: str = "FI self-benchmark") -> tuple[Path, Path]:
    """``<out_dir>/<name>.json`` and ``<out_dir>/<name>.md``."""
    data = results(outcomes, note=note)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / f"{name}.json", out_dir / f"{name}.md"
    js.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    md.write_text(markdown(data, title=title), encoding="utf-8")
    return js, md
