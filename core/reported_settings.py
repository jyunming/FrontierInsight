"""What the analysis reports for a setting must be what ran at that setting.

The analysis reads FI's per-cell record (``raw/trials.json``) and prints a row, a series or a list per setting. When it
reads a setting under a name the record does not have, every row carries the default (``pitch_nm`` = 80 for fourteen
cells that ran at fourteen different pitches), and the paper reports an axis nobody varied. This compares what the
analysis reported with the cells that ran, after the run, from FI's own record and never from the script's claims:

* every table of rows in what the analysis reported (the printed result and the summaries it wrote under
  ``data/results/``: a list of dictionaries, or a dictionary of equal-length lists) is read;
* a column whose name is a setting of the grid (the same words, or the same words with a short unit word) is a setting
  column;
* each row's other numbers that FI's cells also returned are matched to the cells that returned them, and the row's setting
  columns are compared with those cells' settings: a row that carries one cell's results under another cell's setting is a
  mismatch. A table in which most verifiable rows do this is flagged.
* when no row can be matched to a cell, a table with one row per cell whose setting column has a single value while the
  grid varies that setting is flagged as a collapsed axis.

Rows that cannot be matched to a cell (a mean over several cells, a derived number) are not judged.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAME_TOLERANCE = 1e-9
#: A table is read only when it has this many rows.
MIN_ROWS = 2
#: A row is matched to a cell only on at least this many returned numbers, one of which tells the cells apart.
MIN_SHARED = 2
#: The most cells a row may match for the match to say anything.
MAX_CANDIDATES = 3


@dataclass
class Mismatch:
    """A table of the analysis's report whose setting column is not what ran."""

    path: str
    column: str
    axis: str
    reported: list[float]
    ran: list[float]
    rows: int
    wrong: int
    collapsed: bool = False
    top: str = ""

    def says(self, supplied: list[str] | None = None) -> str:
        shown = ", ".join(f"{v:g}" for v in self.reported[:3])
        ran = ", ".join(f"{v:g}" for v in self.ran[:6])
        if self.collapsed:
            return (f"`{self.path}` reports `{self.column}` as {shown} in all {self.rows} rows, but the study varied `{self.axis}` "
                    f"over {ran}")
        return (f"`{self.path}` reports `{self.column}` as {shown} in {self.wrong} of {self.rows} rows whose results ran at "
                f"`{self.axis}` = {ran}")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= SAME_TOLERANCE * (1.0 + max(abs(float(a)), abs(float(b))))


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name)).lower()) if t]


def names_setting(column: str, axis: str) -> bool:
    """Whether a column is named for a grid setting: the same words, or the same words with one short word added to either
    (a unit: ``pitch`` and ``pitch_nm``)."""
    c, a = _tokens(column), _tokens(axis)
    if not c or not a:
        return False
    if c == a:
        return True
    longer, shorter = (c, a) if len(c) > len(a) else (a, c)
    if len(longer) != len(shorter) + 1 or not all(t in longer for t in shorter):
        return False
    extra = [t for t in longer if t not in shorter]
    return len(extra) == 1 and len(extra[0]) <= 3


def cells_of(record: Any) -> list[dict[str, Any]]:
    """``[{"cell": {...}, "metrics": {name: [values]}}]`` from FI's per-cell record (``raw/trials.json``)."""
    out = []
    for entry in (record.get("cells") if isinstance(record, dict) else None) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("cell"), dict):
            continue
        metrics = {}
        for name, m in (entry.get("metrics") or {}).items():
            values = [float(v) for v in (m.get("values") if isinstance(m, dict) else m or []) if _is_number(v)]
            if values:
                metrics[str(name)] = values
        out.append({"cell": entry["cell"], "metrics": metrics})
    return out


def read_record(quest_root: Path) -> list[dict[str, Any]]:
    try:
        return cells_of(json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def tables(value: Any, path: str = "") -> list[tuple[str, list[dict[str, Any]]]]:
    """Every table in ``value``: a list of dictionaries, or a dictionary of equal-length lists of numbers."""
    out: list[tuple[str, list[dict[str, Any]]]] = []
    if isinstance(value, list):
        rows = [r for r in value if isinstance(r, dict)]
        if len(rows) >= MIN_ROWS and len(rows) == len(value):
            out.append((path or "(result)", rows))
        for i, v in enumerate(value[:50]):
            out.extend(tables(v, f"{path}[{i}]" if len(value) > 1 else path))
    elif isinstance(value, dict):
        columns = {k: v for k, v in value.items() if isinstance(v, list) and v and all(_is_number(x) for x in v)}
        sizes = {len(v) for v in columns.values()}
        if len(columns) >= 2 and len(sizes) == 1 and next(iter(sizes)) >= MIN_ROWS:
            n = next(iter(sizes))
            out.append((path or "(result)", [{k: columns[k][i] for k in columns} for i in range(n)]))
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                out.extend(tables(v, f"{path}.{k}" if path else str(k)))
    return out


def _varying(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    """The grid's numeric settings that take more than one value, with the values that ran."""
    axes: dict[str, set[float]] = {}
    for c in cells:
        for a, v in c["cell"].items():
            if _is_number(v):
                axes.setdefault(str(a), set()).add(float(v))
    return {a: sorted(vs) for a, vs in axes.items() if len(vs) > 1}


def _distinct(values: list[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        if not any(same(v, o) for o in out):
            out.append(v)
    return out


def _candidates(row: dict[str, Any], cells: list[dict[str, Any]], skip: set[str]) -> list[dict[str, Any]] | None:
    """The cells that returned the numbers this row carries, or ``None`` when the row cannot be matched to any."""
    shared = [k for k, v in row.items() if k not in skip and _is_number(v) and any(k in c["metrics"] for c in cells)]
    if len(shared) < MIN_SHARED:
        return None
    matched = [c for c in cells if all(
        k not in c["metrics"] or any(same(row[k], v) for v in c["metrics"][k]) or
        same(row[k], sum(c["metrics"][k]) / len(c["metrics"][k])) for k in shared)
        and any(k in c["metrics"] for k in shared)]
    discriminating = [k for k in shared if len(_distinct([v for c in cells for v in c["metrics"].get(k, [])])) > 1]
    if not matched or len(discriminating) < 1 or len(matched) > MAX_CANDIDATES:
        return None
    return matched


def problems(cells: list[dict[str, Any]], reported: list[tuple[str, Any]]) -> list[Mismatch]:
    """The tables in ``reported`` (``[(where, value)]``: the printed result, the summaries) whose setting columns are not what
    ran, judged against ``cells`` (:func:`cells_of`)."""
    varying = _varying(cells)
    if len(cells) < 3 or not varying:
        return []
    out: list[Mismatch] = []
    for where, value in reported:
        for path, rows in tables(value, where if where else ""):
            # Only a table of the printed result can be taken out of it, by the key it sits under.
            root = "" if where else path.split(".")[0].split("[")[0]
            columns = {}
            for k in {k for r in rows for k in r}:
                axis = next((a for a in varying if names_setting(k, a)), None)
                if axis and all(_is_number(r.get(k)) for r in rows):
                    columns[k] = axis
            if not columns:
                continue
            for column, axis in sorted(columns.items()):
                verified = wrong = 0
                reported_values: list[float] = []
                ran_values: list[float] = []
                for row in rows:
                    near = _candidates(row, cells, set(columns))
                    if near is None:
                        continue
                    verified += 1
                    actual = [float(c["cell"][axis]) for c in near if _is_number(c["cell"].get(axis))]
                    if actual and not any(same(row[column], a) for a in actual):
                        wrong += 1
                        reported_values.append(float(row[column]))
                        ran_values.extend(actual)
                if verified and wrong >= max(1, math.ceil(0.5 * verified)):
                    out.append(Mismatch(path, column, axis, _distinct(reported_values), _distinct(ran_values), len(rows), wrong,
                                        top=root))
                elif not verified and len(rows) == len(cells):
                    values = _distinct([float(r[column]) for r in rows])
                    if len(values) == 1:
                        out.append(Mismatch(path, column, axis, values, varying[axis], len(rows), len(rows), collapsed=True,
                                            top=root))
    return out


def reported_of(quest_root: Path, result_json: Any) -> list[tuple[str, Any]]:
    """What the analysis reported: the printed result, and the small JSON summaries it wrote under ``data/results/``."""
    out: list[tuple[str, Any]] = []
    if result_json:
        out.append(("", result_json))
    folder = Path(quest_root) / "data" / "results"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            if path.stat().st_size <= 2_000_000:
                out.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def directive(found: list[Mismatch], cells: list[dict[str, Any]]) -> str:
    """What to tell the repair of the analysis: plain, naming each table and the setting it reads wrongly. The instruction is
    the last thing in it."""
    lines = [f"- {m.says()}" for m in found[:5]]
    axes = ", ".join(f"`{a}`" for a in list(_varying(cells))[:6])
    return (
        "FI compared what the analysis reports for each setting with the settings that really ran, and they differ:\n"
        + "\n".join(lines)
        + f"\n\nThe study's settings are {axes}. Read each row's setting from the cell its results come from "
        "(`entry[\"cell\"][<setting>]`, by the name the study gives it), never from a default, so every row carries "
        "the setting it was run at. Change nothing else."
    )


def plain_note(found: list[Mismatch]) -> str:
    return ("what the analysis reports for a setting is not what ran at it ("
            + "; ".join(m.says() for m in found[:3]) + ")")


def writer_note(found: list[Mismatch]) -> str:
    names = ", ".join(f"`{m.path}`" for m in found[:4])
    return (f"{plain_note(found)}. The values reported in {names} are left out. Do not report, describe or quote a value for a "
            "setting from them: say plainly, in the limitations, that the per-setting results were reported under the wrong "
            "setting and are left out.")


def keys_to_leave_out(found: list[Mismatch]) -> list[str]:
    """The names to take out of the printed result: the top-level key of each table and the wrongly read column."""
    keys = []
    for m in found:
        for name in (m.top, m.column):
            if name and name not in keys:
                keys.append(name)
    return keys
