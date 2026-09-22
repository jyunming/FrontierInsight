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
from itertools import product
from pathlib import Path
from typing import Any

SCHEMA = "fi.run-manifest/v1"
NAME = "run_manifest.json"
_NO_MATCH = object()  # a cell key's value that names none of the protocol's own values for that axis


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


def _coerce_number(raw: str) -> Any:
    """A cell key's value as the number it names, when it is one (``"0.9"`` -> ``0.9``); the string itself otherwise."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _match_grid_value(raw: str, allowed: list[Any]) -> Any:
    """The value in ``allowed`` that a cell key's ``raw`` text names (matched numerically, so ``"0.9"`` finds ``0.9`` and a
    ``0.9`` written back with float rounding still finds it), or :data:`_NO_MATCH`."""
    for v in allowed:
        if isinstance(v, str) and raw == v:
            return v
    number = _num(_coerce_number(raw))
    if number is not None:
        for v in allowed:
            vn = _num(v)
            if vn is not None and math.isclose(vn, number, rel_tol=1e-9, abs_tol=1e-12):
                return v
    return _NO_MATCH


def _parse_cell(key: Any) -> dict[str, str] | None:
    """``"R0=0.9,N=100"`` -> ``{"R0": "0.9", "N": "100"}``. ``None`` when ``key`` is not a string shaped that way, or
    names the same axis twice."""
    if not isinstance(key, str) or not key:
        return None
    parsed: dict[str, str] = {}
    for part in key.split(","):
        if "=" not in part:
            return None
        axis, _, value = part.partition("=")
        axis = axis.strip()
        if not axis or axis in parsed:
            return None
        parsed[axis] = value.strip()
    return parsed


def _canonicalize_cell(key: Any, grid: dict[str, list[Any]]) -> tuple[frozenset[tuple[str, Any]] | None, str | None]:
    """The setting a manifest's cell key names, as a hashable set of ``(axis, the protocol's own value object)`` pairs so
    two keys for the same setting compare equal regardless of key order or number formatting — or ``(None, why)`` when the
    key cannot be placed in the protocol's grid at all (the identity check :func:`problems` needs before it trusts any
    count against it)."""
    parsed = _parse_cell(key)
    if parsed is None:
        return None, f"cell key {key!r} does not look like `axis=value[,axis=value...]`"
    grid_axes, key_axes = set(grid), set(parsed)
    if key_axes != grid_axes:
        parts = []
        if missing := sorted(grid_axes - key_axes):
            parts.append(f"is missing {', '.join(missing)}")
        if extra := sorted(key_axes - grid_axes):
            parts.append(f"names {', '.join(extra)}, which the protocol's grid does not have")
        return None, f"cell key {key!r} {' and '.join(parts)}"
    canon: list[tuple[str, Any]] = []
    for axis, raw in parsed.items():
        matched = _match_grid_value(raw, grid[axis])
        if matched is _NO_MATCH:
            return None, f"cell key {key!r} sets {axis}={raw!r}, and the protocol's grid for {axis} is {_fmt(grid[axis])}"
        canon.append((axis, matched))
    return frozenset(canon), None


def _all_cells(grid: dict[str, list[Any]]) -> set[frozenset[tuple[str, Any]]]:
    """Every setting the protocol's grid names — its full Cartesian product, canonically keyed the same way
    :func:`_canonicalize_cell` keys a manifest's cells, so the two sets can be compared for exact equality."""
    axes = list(grid.items())
    return {frozenset(zip((a for a, _ in axes), combo)) for combo in product(*(v for _, v in axes))}


def problems(protocol: dict[str, Any], manifest: dict[str, Any] | None, why: str = "") -> list[str]:
    """How the run differs from the protocol, one sentence each; empty when the manifest matches it.

    A cell (a setting the grid names) is identified by parsing its key and snapping each axis's value to the protocol's own
    value object (:func:`_canonicalize_cell`), never by comparing key strings: a key naming an axis the protocol does not
    have, missing one it does, or naming a value outside the protocol's list for an axis is a difference on its own, and the
    run's cells must then match the protocol's full Cartesian product exactly — sweeping one axis or value beyond the
    protocol is not absorbed as "extra", it is the same kind of difference as leaving one out. Reaching a wider grid this
    way is always available: ask for it as a protocol amendment.
    """
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
    if grid and (extra_axes := sorted(set(realized) - set(grid))):
        out.append(
            f"the run's realized_grid sweeps {', '.join(extra_axes)}, which the protocol's grid does not fix: "
            "an axis beyond the protocol needs a protocol amendment, not a wider run"
        )

    runs = _num(protocol.get("runs_per_setting"))
    attempted = manifest.get("attempted_per_cell")
    cells_trustworthy = False
    if grid and isinstance(attempted, dict) and attempted:
        canon_of: dict[str, frozenset[tuple[str, Any]]] = {}
        raw_of: dict[frozenset[tuple[str, Any]], list[str]] = {}
        cell_errors: list[str] = []
        for raw_key in attempted:
            canon, err = _canonicalize_cell(raw_key, grid)
            if err:
                cell_errors.append(err)
                continue
            canon_of[raw_key] = canon
            raw_of.setdefault(canon, []).append(raw_key)
        if cell_errors:
            out.extend(cell_errors[:4])
            if len(cell_errors) > 4:
                out.append(f"...and {len(cell_errors) - 4} more cell key(s) that do not fit the protocol's grid")
        if duplicates := {c: ks for c, ks in raw_of.items() if len(ks) > 1}:
            shown = "; ".join(f"{ks[0]!r} and {ks[1]!r} name the same setting" for ks in list(duplicates.values())[:3])
            out.append(f"more than one cell key names the same setting ({shown})")
        cells_trustworthy = not cell_errors and not duplicates
        if cells_trustworthy:
            all_cells = _all_cells(grid)
            covered = set(canon_of.values())
            if covered != all_cells:
                out.append(
                    f"the protocol's grid has {len(all_cells)} setting(s) exactly ({_fmt_cells(all_cells)}); "
                    f"the run's cells are {_fmt_cells(covered)}"
                )
    if runs is not None:
        if not isinstance(attempted, dict) or not attempted:
            out.append(f"the protocol fixes {runs:g} runs per setting, and the run reports no attempted trials per cell (`attempted_per_cell`)")
        elif not grid or cells_trustworthy:
            short = {c: n for c, n in attempted.items() if _num(n) is None or not math.isclose(float(n), runs)}
            if short:
                shown = ", ".join(f"{c}: {n}" for c, n in list(short.items())[:4])
                out.append(f"the protocol fixes {runs:g} runs per setting; {len(short)} setting(s) ran another number ({shown})")

    listed = manifest.get("failed_trials")
    listed = listed if isinstance(listed, list) else []
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
        if lost and len(listed) != lost:
            out.append(f"{lost} trial(s) did not complete, and the run lists {len(listed)} of them (`failed_trials`): a failure must be listed, not dropped")

    if listed and (not grid or cells_trustworthy):
        seen: set[tuple[Any, Any]] = set()
        bad: list[str] = []
        for i, entry in enumerate(listed):
            if not isinstance(entry, dict):
                bad.append(f"failed_trials[{i}] is not an object")
                continue
            cell = entry.get("cell")
            if not isinstance(cell, str) or (grid and _canonicalize_cell(cell, grid)[0] is None):
                bad.append(f"failed_trials[{i}] names no valid cell of the protocol's grid ({cell!r})")
            if "trial" not in entry:
                bad.append(f"failed_trials[{i}] has no `trial` id")
            elif (cell, entry["trial"]) in seen:
                bad.append(f"failed_trials names the same cell and trial id twice ({cell!r}, {entry['trial']!r})")
            else:
                seen.add((cell, entry["trial"]))
            if not str(entry.get("reason") or "").strip():
                bad.append(f"failed_trials[{i}] has no `reason`")
        if bad:
            out.extend(bad[:4])
            if len(bad) > 4:
                out.append(f"...and {len(bad) - 4} more problem(s) in `failed_trials`")

    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    used = manifest.get("thresholds_used")
    used = used if isinstance(used, dict) else {}
    for name, value in thresholds.items():
        if name not in used:
            out.append(f"the protocol fixes the threshold {name} = {value}, and the run reports none (`thresholds_used`)")
        elif not _same(value, used[name]):
            out.append(f"the protocol fixes the threshold {name} = {value}; the run used {used[name]}")
    return out


def _fmt_cells(cells: set[frozenset[tuple[str, Any]]]) -> str:
    """A short, deterministic description of a set of canonical cells, for a difference message."""
    def render(cell: frozenset[tuple[str, Any]]) -> str:
        return ",".join(f"{a}={v:g}" if isinstance(v, float) else f"{a}={v}" for a, v in sorted(cell))

    shown = sorted(render(c) for c in cells)
    if len(shown) <= 6:
        return f"{{{', '.join(shown)}}} ({len(shown)})"
    return f"{{{', '.join(shown[:6])}, ...}} ({len(shown)})"


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
