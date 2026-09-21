"""What a simulation says it actually did, compared with the protocol that was frozen for it.

``core/protocol_check.py`` reads the scripts and looks for a grid, a run count and thresholds that contradict the protocol: a static
lint, which cannot see what the script does when it runs (a constant that is never used, a loop that runs 30 times under
``NUM_RUNS = 300``, a grid that a helper overrides, trials dropped after a failure). This module reads the other side: the
``run_manifest.json`` that ``simulate.py`` writes into its raw-data folder, listing the grid it really swept, the trials it
attempted and completed in every cell, the trials that failed and the thresholds it used, and compares it with the frozen
protocol. A disagreement is a difference the run itself reports, not one a search of the source may or may not find.

The manifest is the script's own statement: it shows that the run says it matched the protocol, and a script can write anything.
What it adds over the lint is that a run which contradicts the protocol can no longer pass by containing the right constants, and
that a mismatch is a difference between two records, which the engine can act on without reading code.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "fi.run-manifest/v1"
NAME = "run_manifest.json"


def read(raw_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """The manifest a simulation wrote into ``raw_dir``, or ``(None, why)``."""
    path = raw_dir / NAME
    if not path.is_file():
        return None, f"{NAME} was not written into the folder FI_RAW_DIR names"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return None, f"{NAME} could not be read as JSON ({e})"
    if not isinstance(data, dict):
        return None, f"{NAME} is not a JSON object"
    if data.get("schema") != SCHEMA:
        return None, f"{NAME} does not say `\"schema\": \"{SCHEMA}\"` (it says {data.get('schema')!r})"
    return data, ""


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _same(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _fmt(values: list[Any]) -> str:
    return "[" + ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values) + "]"


def checkable(protocol: dict[str, Any] | None) -> bool:
    """Whether the protocol fixes anything a manifest can be compared with."""
    if not isinstance(protocol, dict):
        return False
    grid = protocol.get("grid")
    runs = protocol.get("runs_per_setting")
    thresholds = protocol.get("thresholds")
    return bool(
        (isinstance(grid, dict) and grid)
        or (_num(runs) is not None)
        or (isinstance(thresholds, dict) and thresholds)
    )


def problems(protocol: dict[str, Any], manifest: dict[str, Any] | None, why: str = "") -> list[str]:
    """How the run differs from the protocol, one sentence each; empty when the manifest matches it."""
    if manifest is None:
        return [why or f"{NAME} is missing"]
    out: list[str] = []

    realized = manifest.get("realized_grid")
    realized = realized if isinstance(realized, dict) else {}
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    for axis, values in grid.items():
        want = values if isinstance(values, list) else []
        if not want:
            continue
        got = realized.get(axis)
        if not isinstance(got, list):
            out.append(f"the protocol fixes {axis} at {_fmt(want)}, and the run reports no realized values for it")
            continue
        left_out = [v for v in want if not any(_same(v, g) for g in got)]
        extra = [g for g in got if not any(_same(g, v) for v in want)]
        if left_out or extra:
            parts = ([f"leaves out {_fmt(left_out)}"] if left_out else []) + ([f"adds {_fmt(extra)}"] if extra else [])
            out.append(f"the protocol fixes {axis} at {_fmt(want)}; the run swept {_fmt(got)}: it {' and '.join(parts)}")

    runs = _num(protocol.get("runs_per_setting"))
    attempted = manifest.get("attempted_per_cell")
    if runs is not None:
        if not isinstance(attempted, dict) or not attempted:
            out.append(f"the protocol fixes {runs:g} runs per setting, and the run reports no attempted trials per cell (`attempted_per_cell`)")
        else:
            cells = 1
            for values in grid.values():
                cells *= len(values) if isinstance(values, list) and values else 1
            # Only when the run swept nothing the protocol does not list: another axis multiplies the settings.
            if grid and set(realized) <= set(grid) and len(attempted) != cells:
                out.append(f"the protocol's grid has {cells} settings; the run reports {len(attempted)}")
            short = {c: n for c, n in attempted.items() if _num(n) is None or not math.isclose(float(n), runs)}
            if short:
                shown = ", ".join(f"{c}: {n}" for c, n in list(short.items())[:4])
                out.append(f"the protocol fixes {runs:g} runs per setting; {len(short)} setting(s) ran another number ({shown})")

    if isinstance(attempted, dict) and attempted:
        completed = manifest.get("successful_per_cell")
        completed = completed if isinstance(completed, dict) else {}
        lost = 0
        for cell, n in attempted.items():
            done = _num(completed.get(cell))
            total = _num(n)
            if total is not None and (done is None or done > total):
                out.append(f"setting {cell} attempted {n} trials and reports {completed.get(cell)!r} completed (`successful_per_cell`)")
                break
            if total is not None and done is not None:
                lost += int(total - done)
        listed = manifest.get("failed_trials")
        listed = listed if isinstance(listed, list) else []
        if lost and len(listed) != lost:
            out.append(f"{lost} trial(s) did not complete, and the run lists {len(listed)} of them (`failed_trials`): a failure must be listed, not dropped")

    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    used = manifest.get("thresholds_used")
    used = used if isinstance(used, dict) else {}
    for name, value in thresholds.items():
        if name not in used:
            out.append(f"the protocol fixes the threshold {name} = {value}, and the run reports none (`thresholds_used`)")
        elif not _same(value, used[name]):
            out.append(f"the protocol fixes the threshold {name} = {value}; the run used {used[name]}")
    return out


def failure_count(manifest: dict[str, Any] | None) -> int:
    """How many trials the run says did not complete (0 when it says nothing)."""
    if not isinstance(manifest, dict):
        return 0
    attempted = manifest.get("attempted_per_cell")
    completed = manifest.get("successful_per_cell")
    if not isinstance(attempted, dict) or not isinstance(completed, dict):
        return 0
    lost = 0
    for cell, n in attempted.items():
        total, done = _num(n), _num(completed.get(cell))
        if total is not None and done is not None and done <= total:
            lost += int(total - done)
    return lost


def contract() -> str:
    """The part of the two-script directive that asks ``simulate.py`` for the manifest."""
    return (
        "**run_manifest.json.** When it has finished, simulate.py also writes `run_manifest.json` into the same folder, "
        "as JSON, saying what it ACTUALLY ran (not what it meant to run): "
        '{"schema": "fi.run-manifest/v1", "realized_grid": {"<parameter>": [<every value it swept>]}, '
        '"attempted_per_cell": {"<parameter>=<value>,<parameter>=<value>": <trials it started in that setting>}, '
        '"successful_per_cell": {"<same keys>": <trials that completed>}, '
        '"failed_trials": [{"cell": "<same key>", "trial": <number>, "reason": "<short>"}], '
        '"thresholds_used": {"<name>": <value>}}. '
        "Build it from what the loops did (count in the loop, not from a constant declared above it), list EVERY trial that "
        "failed (a solver that did not converge, an exception) instead of dropping it, and read the grid, the run count and "
        "the thresholds it uses from ONE place in the script, so the manifest and the simulation cannot disagree. "
    )


def directive(found: list[str], protocol: dict[str, Any]) -> str:
    """What stands where a traceback would in the repair request."""
    return (
        "This run finished, but its run_manifest.json does not match the frozen protocol:\n"
        + "\n".join(f"- {line}" for line in found[:8])
        + "\n\nThe frozen protocol (it cannot change here):\n"
        + json.dumps({k: protocol.get(k) for k in ("grid", "runs_per_setting", "thresholds", "failure_policy") if protocol.get(k) is not None}, separators=(",", ":"))
        + "\n\nRewrite simulate.py so that it runs EXACTLY that design and writes run_manifest.json from what its loops did "
        "(realized_grid, attempted_per_cell, successful_per_cell, failed_trials, thresholds_used, schema `fi.run-manifest/v1`). "
        "Do not edit the manifest to say what the protocol says: change what the simulation does. Keep everything else."
    )


_IMPORTS_SIMULATE = re.compile(
    r"(^\s*(?:import|from)\s+simulate\b)|(runpy\.[\w_]+\(\s*['\"][^'\"]*simulate)|(subprocess\.[\w_]+\([^)]*simulate\.py)|(os\.system\([^)]*simulate\.py)|(exec\(\s*open\([^)]*simulate)",
    re.MULTILINE,
)


def split_lint(scripts: dict[str, str]) -> list[str]:
    """Where the two-script contract is broken in the source: the analysis must not import or run the simulation, and the
    simulation must not print the results a paper is written from (``RESULT_JSON``) or draw the figures."""
    out: list[str] = []
    analysis = scripts.get("experiment.py") or ""
    simulation = scripts.get("simulate.py") or ""
    if analysis and _IMPORTS_SIMULATE.search(analysis):
        out.append("experiment.py imports or runs simulate.py: the analysis must read only the files in FI_RAW_DIR")
    if simulation and re.search(r"\bRESULT_JSON\s*:", simulation):
        out.append("simulate.py prints or names RESULT_JSON: results and statistics belong to experiment.py")
    if simulation and re.search(r"savefig\s*\(", simulation):
        out.append("simulate.py draws figures: figures belong to experiment.py")
    return out
