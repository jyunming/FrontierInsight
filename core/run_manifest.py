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

**The trial counts don't have to be self-reported.** ``simulate.py`` also appends one real line per trial, as it finishes,
to ``trial_ledger.jsonl`` (:func:`read_ledger`) in the same folder; :func:`manifest_from_ledger` derives the count fields
from those lines instead of trusting a summary. Two designs considered and rejected for this, so a future change doesn't
re-litigate them without new information:

* **One subprocess per trial**, so the engine itself drives every trial instead of the script looping internally. This is
  what a literal reading of "the engine assigns each trial" suggests, and it is the most rigorous option in principle —
  but measured against a realistic script (one that imports numpy, the floor for anything in this codebase), a single
  subprocess spawn costs roughly 130ms; at ``runs_per_setting: 300`` across even a small grid that is minutes of pure
  process-spawn overhead added to quests that complete in seconds today. Not viable for FI's actual workload shape.
* **An RNG "fingerprint"** — the engine assigns each trial a seed, and the script must report a hash of the RNG's first
  few draws from that seed, which the engine re-derives and checks. This looks like independent verification, but it
  is not: a fingerprint computed purely from a known seed requires no real trial computation to produce, so it is exactly
  as cheap to fabricate 300 of them as it is to verify them. It would not have caught the concrete bypass this exists
  for.

What ships instead trades a weaker guarantee for something actually enforceable: the engine counts real, appended lines,
so lying about a trial count now costs writing that many lines (and, per the ``result_json`` check below, that many real
per-trial values downstream) rather than one number. It does not prove any single trial's SCIENCE is correct — no
mechanism here re-executes the simulation — the same limit every other gate in this codebase already lives with (the
oracle check proves agreement with a closed form the plan declared, not that the simulator is right).
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
LEDGER_NAME = "trial_ledger.jsonl"
_NO_MATCH = object()  # a cell key's value that names none of the protocol's own values for that axis


def read_ledger(raw_dir: Path) -> tuple[list[dict[str, Any]] | None, str]:
    """The per-trial ledger a simulation appended to in ``raw_dir`` (one JSON object per line: ``{"cell": "<axis=value,...>",
    "trial": <id>, "status": "ok"|"failed", "reason": "<only for failed>"}``), or ``(None, why)``.

    Where this exists, :func:`manifest_from_ledger` derives the trial counts from it instead of trusting a self-reported
    summary: a script can still write anything into ``run_manifest.json`` (see the module docstring), but it cannot make
    the engine COUNT trials it never appended a line for."""
    path = raw_dir / LEDGER_NAME
    if not path.is_file():
        return None, f"{LEDGER_NAME} was not written into the folder FI_RAW_DIR names"
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return None, f"{LEDGER_NAME} could not be read ({e})"
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            return None, f"{LEDGER_NAME} line {i} is not valid JSON"
        if not isinstance(row, dict):
            return None, f"{LEDGER_NAME} line {i} is not a JSON object"
        rows.append(row)
    return rows, ""


def _norm_trial(trial: Any) -> Any:
    """A trial id in one canonical form, so ``1`` (an int, the usual case) and ``"1"`` (a string) name the same trial
    for dedup — a script that replays a trial under the other JSON type must not slip past the duplicate check."""
    if isinstance(trial, bool):
        return trial
    if isinstance(trial, int):
        return trial
    if isinstance(trial, float):
        return int(trial) if trial.is_integer() else trial
    if isinstance(trial, str):
        try:
            return int(trial)
        except ValueError:
            pass
        try:
            f = float(trial)
            return int(f) if f.is_integer() else f
        except ValueError:
            pass
    return trial


def manifest_from_ledger(
    protocol: dict[str, Any], ledger: list[dict[str, Any]], thresholds_used: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """``(manifest, problems)`` — the manifest a ledger of real per-trial rows implies, standing in for the fields a
    script would otherwise self-report (``realized_grid``, ``attempted_per_cell``, ``successful_per_cell``,
    ``failed_trials``): each is counted from ledger rows the engine can see, not read from a summary the script wrote.
    ``thresholds_used`` still comes from the caller (a ledger row has no natural place for a run-wide setting).

    ``problems`` names what's wrong with the ledger ITSELF, before the derived manifest is even compared with the
    protocol (:func:`problems` does that part): a row naming no cell the protocol's grid has, one missing ``trial`` or
    ``status``, or two rows claiming the same (cell, trial) — the exact "duplicate id" / "replayed trial" and
    "out-of-protocol cell" shapes a script's self-report could otherwise hide."""
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    row_problems: list[str] = []
    attempted: dict[str, int] = {}
    successful: dict[str, int] = {}
    failed: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    realized: dict[str, set[Any]] = {axis: set() for axis in grid}
    for i, row in enumerate(ledger):
        cell_key = row.get("cell")
        canon, err = _canonicalize_cell(cell_key, grid) if grid else (None, None)
        if grid and err:
            row_problems.append(f"ledger row {i}: {err}")
            continue
        display = cell_key if isinstance(cell_key, str) else repr(cell_key)
        if "trial" not in row:
            row_problems.append(f"ledger row {i} (cell {display!r}) has no `trial` id")
            continue
        trial = row["trial"]
        dedup_key = (canon if canon is not None else display, _norm_trial(trial))
        if dedup_key in seen:
            row_problems.append(f"more than one ledger row claims cell {display!r}, trial {trial!r}")
            continue
        seen.add(dedup_key)
        status = str(row.get("status") or "").strip().lower()
        if status not in ("ok", "failed"):
            row_problems.append(f"ledger row {i} (cell {display!r}, trial {trial!r}) has no `status` of \"ok\" or \"failed\"")
            continue
        attempted[display] = attempted.get(display, 0) + 1
        if status == "ok":
            successful[display] = successful.get(display, 0) + 1
        else:
            failed.append({"cell": display, "trial": trial, "reason": str(row.get("reason") or "").strip() or "(no reason given)"})
        if canon is not None:
            for axis, value in canon:
                realized[axis].add(value)
    manifest = {
        "schema": SCHEMA,
        "realized_grid": {axis: sorted(values, key=lambda v: (isinstance(v, str), v)) for axis, values in realized.items()},
        "attempted_per_cell": attempted,
        "successful_per_cell": successful,
        "failed_trials": failed,
        "thresholds_used": dict(thresholds_used or {}),
    }
    return manifest, row_problems


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


def _total_values_reported(result_json: Any, metric_id: str) -> int:
    """The total length of every list found anywhere in ``result_json`` under a key named ``f"{metric_id}_values"`` — the
    real per-trial numbers the analysis actually had to work with, regardless of which setting held them."""
    key = f"{metric_id}_values"
    total = 0

    def walk(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            for k, v in node.items():
                if k == key and isinstance(v, list):
                    total += sum(1 for x in v if isinstance(x, (int, float)) and not isinstance(x, bool))
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(result_json)
    return total


def _values_listed(result_json: Any, metric_id: str) -> bool:
    """Whether ``result_json`` holds a ``f"{metric_id}_values"`` list anywhere at all (an empty one counts)."""
    key = f"{metric_id}_values"

    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            return any((k == key and isinstance(v, list)) or walk(v) for k, v in node.items())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return walk(result_json)


def _value_count_findings(
    protocol: dict[str, Any], manifest: dict[str, Any], result_json: Any,
) -> list[tuple[str, bool]]:
    """``(sentence, the analysis listed no values at all)`` for each `kind: "mean"` metric whose claimed trial count is
    not backed by the per-trial values the analysis printed.

    The two shapes point at different scripts. A `<id>_values` list that exists but is far shorter than the count claimed
    is the audit's own fabrication case (10 real trials, 300 claimed): the simulation is sent back. No such list at all
    is the analysis not printing what the contract asks of it, while the simulation's raw files may hold every trial
    (a live quest's did: 100 final sizes per setting on disk, and the rewrite of simulate.py this used to ask for broke
    the oracle checks it had passed). That one is the analysis's to fix."""
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    runs = _num(protocol.get("runs_per_setting"))
    if result_json is None or not grid or runs is None:
        return []
    # The baseline is trials the manifest itself says SUCCEEDED, not the protocol's raw runs_per_setting x cells: a
    # design with a genuine, declared failure rate (failed_trials, a failure_policy) legitimately has fewer real values
    # than the nominal total, and that is not fabrication. A low threshold (rather than requiring an exact match) leaves
    # room for a metric that, by the design's own nature, only applies to a subset of cells (e.g. a "time to extinction"
    # that only exists where an outbreak happened) -- this check is aimed at the audit's own example (10 real trials,
    # 300 claimed), not at flagging every partial or conditional metric.
    successful = manifest.get("successful_per_cell")
    successful = successful if isinstance(successful, dict) else {}
    expected_total = sum(_num(n) or 0 for n in successful.values()) or runs * len(_all_cells(grid))
    out: list[tuple[str, bool]] = []
    for spec in protocol.get("metrics") or []:
        if not isinstance(spec, dict) or spec.get("kind") != "mean" or not spec.get("id"):
            continue
        metric = str(spec["id"])
        given = str(spec.get("given") or "").strip()
        # A mean over a subset of the trials (the final size of the runs that became major outbreaks) is backed by the
        # size of that subset, which the proportion it is `given` counts, not by every trial: a live quest's 1022 final
        # sizes of 2700 runs were its 1022 major outbreaks, and this check read them as 1678 trials gone missing.
        base, over = expected_total, f"{expected_total:g} successful trial(s)"
        if given:
            if not _counts_listed(result_json, f"{given}_count"):
                out.append((
                    f"the metric `{metric}` is a mean over the trials `{given}` counts, but the analysis's RESULT_JSON "
                    f"prints no `{given}_count`, so the subset it averages over cannot be counted",
                    True,
                ))
                continue
            base, over = _total_counts_reported(result_json, f"{given}_count", beside=f"{metric}_values"), \
                f"the trials `{given}` counts"
            given_total = _total_counts_reported(result_json, f"{given}_total", beside=f"{metric}_values")
            if given_total < expected_total * 0.5:
                out.append((
                    f"the manifest reports {expected_total:g} successful trial(s) in total, but `{given}`, the "
                    f"proportion `{metric}` is a mean over, counts only {given_total:g} trial(s) in its `{given}_total` "
                    "— the claimed trial count is not backed by the data the analysis actually used",
                    False,
                ))
                continue
        if not _values_listed(result_json, metric):
            out.append((
                f"the metric `{metric}` is a mean over {over} ({base:g}), but the analysis's "
                f"RESULT_JSON has no `{metric}_values` list at all — the analysis must print the per-trial values it "
                "averaged, so the claimed trial count can be checked against them",
                True,
            ))
            continue
        got = _total_values_reported(result_json, metric)
        if got < base * 0.5:
            out.append((
                (
                    f"the metric `{metric}` is a mean over {over} ({base:g}), but it only has {got} real per-trial "
                    "value(s) in this run's own analysis output"
                    if given else
                    f"the manifest reports {expected_total:g} successful trial(s) in total, but the metric "
                    f"`{metric}` only has {got} real per-trial value(s) in this run's own analysis output"
                )
                + " — the claimed trial count is not backed by the data the analysis actually used"
                + ("" if given else (
                    ". If this mean is over a subset of the trials by design (the runs that became major outbreaks, "
                    "say), the protocol's metric spec says so with `given`: the id of the proportion that counts them"
                )),
                False,
            ))
    return out


def _total_counts_reported(result_json: Any, key: str, *, beside: str | None = None) -> float:
    """The sum of the numbers in ``result_json`` under ``key`` (a ``<id>_count`` or ``<id>_total``).

    With ``beside`` (a ``<metric>_values`` key), only the counts that sit in the same object as those values are summed:
    they are what the values are backed by. A copy of one setting's count elsewhere in the result (a headline repeated
    at the top level) was added to the per-setting ones: a live run's 933 + 96 was printed in its paper as 1,029 trials.
    When no count sits beside the values, every count found is summed, as before."""
    total, paired, found_paired = 0.0, 0.0, False

    def number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def walk(node: Any) -> None:
        nonlocal total, paired, found_paired
        if isinstance(node, dict):
            if beside and key in node and number(node[key]) and isinstance(node.get(beside), list):
                paired += float(node[key])
                found_paired = True
            for k, v in node.items():
                if k == key and number(v):
                    total += float(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(result_json)
    return paired if found_paired else total


def _counts_listed(result_json: Any, key: str) -> bool:
    """Whether ``result_json`` holds a number under ``key`` anywhere."""
    if isinstance(result_json, dict):
        return any(
            (k == key and isinstance(v, (int, float)) and not isinstance(v, bool)) or _counts_listed(v, key)
            for k, v in result_json.items()
        )
    if isinstance(result_json, list):
        return any(_counts_listed(v, key) for v in result_json)
    return False


def analysis_output_problems(
    protocol: dict[str, Any], manifest: dict[str, Any] | None, result_json: Any,
) -> list[str]:
    """The differences :func:`problems` reports that are the analysis's to fix, not the simulation's: a mean metric the
    analysis printed no per-trial values for. The caller sends experiment.py back for these, simulate.py for the rest."""
    if not isinstance(manifest, dict):
        return []
    return [sentence for sentence, analysis in _value_count_findings(protocol, manifest, result_json) if analysis]


def problems(
    protocol: dict[str, Any], manifest: dict[str, Any] | None, why: str = "",
    result_json: dict[str, Any] | None = None,
) -> list[str]:
    """How the run differs from the protocol, one sentence each; empty when the manifest matches it.

    A cell (a setting the grid names) is identified by parsing its key and snapping each axis's value to the protocol's own
    value object (:func:`_canonicalize_cell`), never by comparing key strings: a key naming an axis the protocol does not
    have, missing one it does, or naming a value outside the protocol's list for an axis is a difference on its own, and the
    run's cells must then match the protocol's full Cartesian product exactly — sweeping one axis or value beyond the
    protocol is not absorbed as "extra", it is the same kind of difference as leaving one out. Reaching a wider grid this
    way is always available: ask for it as a protocol amendment.

    ``result_json`` (the same seed's own analysis output, when the caller has it) backs the trial counts against data the
    analysis actually used: a manifest can claim any ``attempted_per_cell`` it likes (:mod:`core.run_manifest`'s own
    docstring says so), but a `kind: "mean"` metric's `<id>_values` array is real per-trial numbers the paper's statistics
    are computed from, so a manifest whose claimed trial count is not backed by anywhere near that many real values is
    caught here, independent of whatever the manifest's own summary says. This does not prove every individual trial is
    genuine (a script could still fabricate that many values) — it raises the bar from writing one number to fabricating
    that much of the data the paper stands on, which is the same kind of commitment the rest of the evidence ladder relies
    on rather than a cryptographic guarantee.
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
        out.extend(sentence for sentence, _ in _value_count_findings(protocol, manifest, result_json))

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
    """The part of the two-script directive that asks ``simulate.py`` for the manifest and the per-trial ledger."""
    return (
        f"**{LEDGER_NAME}.** As EACH trial finishes (not at the end), append one line to `{LEDGER_NAME}` in the same "
        'folder: a single JSON object, `{"cell": "<parameter>=<value>,<parameter>=<value>", "trial": <number>, '
        '"status": "ok" | "failed", "reason": "<only when failed>"}`, then flush. This is the record FI actually counts '
        "trials from — one real line per trial, written as it happens, not a total computed afterwards. "
        "**run_manifest.json.** When it has finished, simulate.py also writes `run_manifest.json` into the same folder, "
        "as JSON, saying what it ACTUALLY ran (not what it meant to run): "
        '{"schema": "fi.run-manifest/v1", "realized_grid": {"<parameter>": [<every value it swept>]}, '
        '"attempted_per_cell": {"<parameter>=<value>,<parameter>=<value>": <trials it started in that setting>}, '
        '"successful_per_cell": {"<same keys>": <trials that completed>}, '
        '"failed_trials": [{"cell": "<same key>", "trial": <number>, "reason": "<short>"}], '
        '"thresholds_used": {"<name>": <value>}}. '
        "Build both from what the loops did (count in the loop, not from a constant declared above it), list EVERY "
        "trial that failed (a solver that did not converge, an exception) instead of dropping it, and read the grid, "
        "the run count and the thresholds it uses from ONE place in the script, so the two files and the simulation "
        "cannot disagree. "
    )


def directive(found: list[str], protocol: dict[str, Any]) -> str:
    """What stands where a traceback would in the repair request."""
    return (
        "This run finished, but what it actually did does not match the frozen protocol:\n"
        + "\n".join(f"- {line}" for line in found[:8])
        + "\n\nThe frozen protocol (it cannot change here):\n"
        + json.dumps({k: protocol.get(k) for k in ("grid", "runs_per_setting", "thresholds", "failure_policy") if protocol.get(k) is not None}, separators=(",", ":"))
        + f"\n\nRewrite simulate.py so that it runs EXACTLY that design: appends one real line to `{LEDGER_NAME}` as EACH "
        "trial finishes (cell, trial, status, and reason when failed), and writes run_manifest.json from what its loops "
        "did (realized_grid, attempted_per_cell, successful_per_cell, failed_trials, thresholds_used, schema "
        "`fi.run-manifest/v1`). Do not edit either file to say what the protocol says: change what the simulation does "
        f"— FI counts trials from `{LEDGER_NAME}`'s own lines, not from a number written beside it. Keep everything else."
    )


def analysis_directive(found: list[str]) -> str:
    """The repair request for :func:`analysis_output_problems`: the simulation stands, the analysis must print its values."""
    return (
        "This run finished, and the simulation's raw files stand, but the analysis did not print what the contract asks:\n"
        + "\n".join(f"- {line}" for line in found[:8])
        + "\n\nRewrite experiment.py (not simulate.py) so that, for each of those metrics, RESULT_JSON carries a "
        "`<metric>_values` list: the per-trial values, read from the raw files, that the metric's mean is computed from "
        "(per setting when the metric is reported per setting). Do not invent values: every number must come from the raw "
        "files. A metric that is not a mean over trials at all (a closed-form or deterministic value) has no per-trial "
        "values to list; say so plainly in the analysis rather than printing a made-up list. Keep everything else."
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
