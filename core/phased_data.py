"""Which rows of a table are held back for a confirm run, and how a data quest's confirm numbers compare with exploration's.

``core/phased.py`` keeps the files, the record and the two stages; this module only decides. Rows are never held back
one by one unless something says the rows are independent: rows of one subject, site, family or device measured more
than once would otherwise land in both parts, and the confirm run would not be on unseen data. What decides, in order:

1. **The plan's protocol** (``protocol.split`` in plan.md's design block, :func:`declared`)::

       split:
         strategy: group        # row | group | time | spatial
         unit: subject_id       # group / spatial: the column naming the unit rows belong to (also group_column)
         time_column: date      # time: the column that orders the rows
         embargo: 1             # time: periods left out between exploration's part and the held-back part
         stratify_by: arm       # row / group: about the same share of each value held back
         min_groups: 10         # group / spatial: the fewest units the table must have
         seed: 7                # what picks the units (the quest's id when not given)

2. **Your answer in plan.md** (:func:`plan_answer`): the line ``Rows that belong together: <column>``, or
   ``Rows that belong together: independent`` when every row is a separate case.
3. **The table's own column names**, only when they leave one reading: a single time column (``date``, ``year``,
   ``time``, ``month``, ``quarter``, ``period``, ``timestamp``, or a name ending in one) and no unit column holds back the
   latest period; a single unit column (``subject``, ``patient``, ``participant``, ``site``, ``school``, ``family``,
   ``household``, ``device``, ``sensor``, ``cluster``, ``country``, ``id`` or a name ending in ``_id``, ...) and no time
   column holds back whole units.
4. Otherwise FI cannot tell (:attr:`Decision.ask`): a research quest asks you once; any other quest holds nothing back
   and says so. Spatial units are never guessed from coordinates: name the block column.

The rules: **row**, each row by a hash of its own text (stratified: the same share of each stratum); **group** and
**spatial**, about :data:`HOLD_BACK_FRACTION` of the units, picked by a hash of each unit's value, with all of a unit's
rows on one side (at least ``min_groups`` units, default :data:`MIN_UNITS`, and :data:`MIN_HELD_UNITS` held back);
**time**, the latest periods until about :data:`HOLD_BACK_FRACTION` of the rows are held back, never more than half, with
``embargo`` periods before them left out of both parts. Every split carries a manifest (:attr:`Decision.manifest`) whose
``overlap`` counts the units found in both parts; ``core/phased.py`` refuses a split with any.

:func:`compare` sets a data quest's confirm numbers beside exploration's. A disagreement is reported, never hidden and
never a gap: the confirm run's numbers are the ones reported.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The share of rows (or units) held back.
HOLD_BACK_FRACTION = 0.3
#: A group or spatial split needs at least this many units ...
MIN_UNITS = 10
#: ... and at least this many of them held back (and as many left for exploration).
MIN_HELD_UNITS = 3
#: A column named like one of the plan's variables is a stratum when it takes at most this many values.
MAX_STRATA = 20
#: Two numbers agree when they have the same sign and differ by at most this share of the larger one.
TOLERANCE = 0.25

ROW, GROUP, TIME, SPATIAL = "row", "group", "time", "spatial"
STRATEGIES = (ROW, GROUP, TIME, SPATIAL)

#: A time column by its name: ``time`` alone, or a date-like name (``reaction_time`` is a measurement, not a clock).
_TIME_NAME = re.compile(r"^(?:time|date|datetime|timestamp|year|month|quarter|period)$"
                        r"|[ _\-.](?:date|datetime|timestamp|year|month|quarter|period)$", re.I)
_UNIT_NAME = re.compile(
    r"(?:^|[ _\-.])(?:id|subject|patient|participant|person|individual|respondent|site|centre|center|clinic|hospital|"
    r"school|family|household|device|sensor|station|cluster|animal|mouse|plot|user|customer|firm|company|country|"
    r"county|city|village|farm|herd|litter)(?:[ _\-.]?id)?$", re.I)
_INDEPENDENT = re.compile(r"^(?:rows?(?: are)? independent|independent(?: rows?)?|each row(?: is)?(?: its own| a separate)?"
                          r" case|none)\.?$", re.I)
_ANSWER_LINE = re.compile(r"^[ \t>*_-]*Rows that belong together[*_]*:[*_]*[ \t]*(.*?)[ \t]*$", re.I | re.M)
_QUARTER = re.compile(r"(\d{4})[-_ ]?[Qq]([1-4])")
_YEAR_MONTH = re.compile(r"(\d{4})-(\d{1,2})")

#: The line plan.md carries while nothing says which rows belong together (:func:`plan_answer` reads it back).
QUESTION_LINE = ("Rows that belong together: ? (write the column that names the subject, site, device or other unit "
                 "several rows can share, or `independent` if every row is a separate case)")


@dataclass
class Decision:
    """What a table's split is. ``rule`` is stored in ``.fi/phased.json`` and applied again the same way at each start."""

    rule: dict[str, Any] = field(default_factory=dict)
    held: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    why: str = ""  # the table cannot be split by the rule that applies (plain words)
    ask: bool = False  # nothing says which rows belong together, and FI cannot tell
    manifest: dict[str, Any] = field(default_factory=dict)


def _cells(line: str, delimiter: str) -> list[str]:
    return next(csv.reader([line], delimiter=delimiter), [])


def _norm(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _digest(salt: str, value: str) -> bytes:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()


def _fraction(salt: str, value: str) -> float:
    return int.from_bytes(_digest(salt, value)[:8], "big") / 2**64


def declared(protocol: Any) -> dict[str, Any] | None:
    """The plan's ``protocol.split`` (a mapping, or a bare strategy name), or None."""
    split = protocol.get("split") if isinstance(protocol, dict) else None
    if isinstance(split, str) and split.strip():
        return {"strategy": split.strip().lower()}
    return dict(split) if isinstance(split, dict) and split else None


def plan_answer(plan_text: str) -> str | None:
    """Your answer on plan.md's line ``Rows that belong together: ...`` (None while it is ``?`` or missing)."""
    for m in _ANSWER_LINE.finditer(plan_text or ""):
        answer = re.sub(r"\s*\(write the column.*$", "", m.group(1), flags=re.I).strip().strip("`'\"*_ ")
        if answer and answer != "?":
            return answer
    return None


def grouping_names(design: Any) -> list[str]:
    """The names of the plan's variables (independent first, then controls): a column named like one is a stratum."""
    variables = design.get("variables") if isinstance(design, dict) else None
    if not isinstance(variables, dict):
        return []
    out: list[str] = []
    for key in ("independent", "controls"):
        values = variables.get(key)
        for v in values if isinstance(values, list) else [values] if values else []:
            name = v.get("name") if isinstance(v, dict) else v
            if isinstance(name, str) and name.strip() and name not in out:
                out.append(name.strip())
    return out


def _column(names: list[str], wanted: Any) -> int | None:
    key = _norm(wanted)
    return next((i for i, n in enumerate(names) if key and _norm(n) == key), None)


def _time_key(value: str) -> tuple[str, float] | None:
    """``(kind, position)`` of one time value, or None when it cannot be put in order."""
    v = value.strip()
    if not v:
        return None
    try:
        number = float(v)
        return ("number", number) if math.isfinite(number) else None
    except ValueError:
        pass
    if m := _QUARTER.fullmatch(v):
        when = datetime(int(m.group(1)), 3 * int(m.group(2)) - 2, 1)
    elif (m := _YEAR_MONTH.fullmatch(v)) and 1 <= int(m.group(2)) <= 12:
        when = datetime(int(m.group(1)), int(m.group(2)), 1)
    else:
        try:
            when = datetime.fromisoformat(v)
        except ValueError:
            return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return "date", when.timestamp()


def decide(file: str, header: str, rows: list[str], delimiter: str, quest_id: str, *,
           declared_split: dict[str, Any] | None = None, answer: str | None = None,
           grouping: list[str] | None = None) -> Decision:
    """The split of one table (see the module docstring for the order). The rows are the file's lines after the header."""
    names = _cells(header, delimiter)
    cells = [_cells(r, delimiter) for r in rows]
    if declared_split:
        spec = {**declared_split, "source": declared_split.get("source") or "plan"}
    elif answer:
        if _INDEPENDENT.match(answer):
            spec = {"strategy": ROW, "source": "answer"}
        elif _column(names, answer) is not None:
            spec = {"strategy": GROUP, "unit": names[_column(names, answer)], "source": "answer"}  # type: ignore[index]
        else:
            return Decision(why=(f"the answer on plan.md's line 'Rows that belong together' ({answer!r}) names no column "
                                 f"of {file} (its columns: {', '.join(names[:12])})"), ask=True)
    else:
        times = [n for n in names if _TIME_NAME.search(n.strip())]
        units = [n for n in names if _UNIT_NAME.search(n.strip()) and n not in times]
        if len(times) == 1 and not units:
            spec = {"strategy": TIME, "time_column": times[0], "source": "columns"}
        elif len(units) == 1 and not times:
            spec = {"strategy": GROUP, "unit": units[0], "source": "columns"}
        else:
            found = ", ".join(f"'{n}'" for n in times + units)
            return Decision(ask=True, why=(
                f"FI cannot tell from {file}'s columns which rows belong to the same subject, site or other unit "
                + (f"(several could: {found})" if found else "(no column names one)")))
    strategy = str(spec.get("strategy") or "").strip().lower()
    salt = f"seed:{spec['seed']}" if spec.get("seed") is not None else quest_id
    if strategy not in STRATEGIES:
        return Decision(why=f"plan.md's protocol.split.strategy is {strategy!r}; it must be one of {', '.join(STRATEGIES)}")
    if any(len(c) != len(names) for c in cells) and strategy != ROW:
        return Decision(why=f"{file} has rows with a different number of values than its header, so its columns "
                            "cannot be read")
    stratum = _stratum(names, cells, spec, grouping or [])
    if stratum is not None:
        # Kept in the rule, so the split applied again at a later start is the same one.
        spec = {**spec, "stratify_by": names[stratum]}
    if strategy == TIME:
        return _time(file, names, rows, cells, spec)
    if strategy == ROW:
        return _rows_split(rows, cells, spec, salt, stratum)
    return _group(file, names, rows, cells, {**spec, "strategy": strategy}, salt, stratum)


def _stratum(names: list[str], cells: list[list[str]], spec: dict[str, Any], grouping: list[str]) -> int | None:
    if spec.get("stratify_by"):
        return _column(names, spec["stratify_by"])
    for g in grouping:
        i = _column(names, g)
        if i is not None and 2 <= len({c[i].strip() for c in cells if len(c) > i}) <= MAX_STRATA:
            return i
    return None


def _rule(spec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    keep = ("strategy", "unit", "time_column", "embargo", "stratify_by", "min_groups", "seed", "source")
    return {**{k: spec[k] for k in keep if spec.get(k) is not None}, **extra}


def _rows_split(rows: list[str], cells: list[list[str]], spec: dict[str, Any], salt: str,
                stratum: int | None) -> Decision:
    if stratum is None:
        held = {r for r in rows if _fraction(salt, r) < HOLD_BACK_FRACTION}
    else:
        held = set()
        for value in {c[stratum].strip() for c in cells if len(c) > stratum}:
            texts = sorted({r for r, c in zip(rows, cells) if len(c) > stratum and c[stratum].strip() == value},
                           key=lambda t: _digest(salt, t))
            want = round(HOLD_BACK_FRACTION * len(texts))
            held.update(texts[:want])
    rule = _rule(spec)
    explore = {r for r in rows if r not in held}
    return Decision(rule=rule, held=held, manifest={
        "strategy": ROW, "unit": "row", "units": len(set(rows)), "units_explore": len(explore),
        "units_held_back": len(held), "overlap": len(explore & held)})


def _group(file: str, names: list[str], rows: list[str], cells: list[list[str]], spec: dict[str, Any], salt: str,
           stratum: int | None) -> Decision:
    unit_name = spec.get("unit") or spec.get("group_column") or spec.get("split_unit")
    col = _column(names, unit_name) if unit_name else None
    strategy = spec["strategy"]
    if col is None:
        what = "spatial block (a region or tile)" if strategy == SPATIAL else "unit"
        return Decision(why=(f"plan.md's protocol.split names no column of {file} for the {what} rows belong to "
                             f"(protocol.split.unit: {unit_name!r}; its columns: {', '.join(names[:12])})"))
    values = [c[col].strip() for c in cells]
    units = sorted(set(values))
    least = int(spec.get("min_groups") or MIN_UNITS)
    if len(units) < max(least, 2 * MIN_HELD_UNITS):
        return Decision(why=(f"{file} has {len(units)} different values of '{names[col]}', fewer than the "
                             f"{max(least, 2 * MIN_HELD_UNITS)} a split by {names[col]} needs"))
    by_stratum: dict[str, list[str]] = {}
    for unit in units:
        first = next(c for c in cells if c[col].strip() == unit)
        by_stratum.setdefault(first[stratum].strip() if stratum is not None else "", []).append(unit)
    held_units: set[str] = set()
    for members in by_stratum.values():
        members.sort(key=lambda u: _digest(salt, u))
        held_units.update(members[:round(HOLD_BACK_FRACTION * len(members))])
    if len(held_units) < MIN_HELD_UNITS or len(units) - len(held_units) < MIN_HELD_UNITS:
        return Decision(why=(f"a split of {file} by '{names[col]}' would leave fewer than {MIN_HELD_UNITS} of its values "
                             "on one side"))
    held = {r for r, v in zip(rows, values) if v in held_units}
    explore_units = {v for r, v in zip(rows, values) if r not in held}
    rule = _rule(spec, unit=names[col])
    return Decision(rule=rule, held=held, manifest={
        "strategy": strategy, "unit": names[col], "units": len(units), "units_explore": len(explore_units),
        "units_held_back": len(held_units), "overlap": len(explore_units & held_units)})


def _time(file: str, names: list[str], rows: list[str], cells: list[list[str]], spec: dict[str, Any]) -> Decision:
    name = spec.get("time_column") or next((n for n in names if _TIME_NAME.search(n.strip())), None)
    col = _column(names, name) if name else None
    if col is None:
        return Decision(why=f"{file} has no time column to split by (protocol.split.time_column: {name!r})")
    keys = [_time_key(c[col]) for c in cells]
    if any(k is None for k in keys) or len({k[0] for k in keys if k}) > 1:
        return Decision(why=(f"{file}'s time column ('{names[col]}') has values that cannot all be put in order "
                             "(numbers, ISO dates such as 2024-03-01, or quarters such as 2024Q1 can), so its latest "
                             "period cannot be held back, and a time series is never split at random"))
    try:
        embargo = max(0, int(spec.get("embargo") or 0))
    except (TypeError, ValueError):
        return Decision(why=f"plan.md's protocol.split.embargo must be a whole number of periods ({spec.get('embargo')!r})")
    counts = Counter(k[1] for k in keys if k)
    periods = sorted(counts)
    if len(periods) < 2 + embargo:
        return Decision(why=(f"{file}'s time column ('{names[col]}') has {len(periods)} period(s), too few to hold back "
                             f"a later one" + (f" with {embargo} left out between" if embargo else "")))
    target = HOLD_BACK_FRACTION * len(rows)
    held_periods: list[float] = []
    count = 0
    for p in reversed(periods):
        if count >= target:
            break
        held_periods.append(p)
        count += counts[p]
    if count * 2 > len(rows) or len(held_periods) + embargo >= len(periods):
        return Decision(why=(f"holding back the latest period of {file} (by '{names[col]}') would take more than half "
                             "of its rows, leaving too few for exploration"))
    first = min(held_periods)
    gap = [p for p in periods if p < first][-embargo:] if embargo else []
    held = {r for r, k in zip(rows, keys) if k and k[1] in held_periods}
    dropped = {r for r, k in zip(rows, keys) if k and k[1] in gap}
    since = next(c[col].strip() for c, k in zip(cells, keys) if k and k[1] == first)
    rule = _rule(spec, time_column=names[col], since=since)
    explore_periods = {k[1] for r, k in zip(rows, keys) if k and r not in held and r not in dropped}
    return Decision(rule=rule, held=held, dropped=dropped, manifest={
        "strategy": TIME, "unit": f"period of {names[col]}", "units": len(periods),
        "units_explore": len(explore_periods), "units_held_back": len(held_periods), "embargo_periods": len(gap),
        "embargo_rows": len(dropped), "overlap": len(explore_periods & set(held_periods))})


def overlap(header: str, explore_rows: list[str], held_rows: list[str], rule: dict[str, Any] | None,
            delimiter: str) -> int:
    """How many units are in both parts as they are on disk (rows for a row split; the unit's values, or the periods,
    for the others): the zero-overlap check, run on the parts written, not on the decision."""
    rule = rule or {}
    strategy = rule.get("strategy")
    if strategy in (GROUP, SPATIAL, TIME):
        names = _cells(header, delimiter)
        col = _column(names, rule.get("unit") if strategy != TIME else rule.get("time_column"))
        if col is None:
            return 0
        def values(rows: list[str]) -> set[str]:
            out = set()
            for r in rows:
                c = _cells(r, delimiter)
                if len(c) > col:
                    out.add(c[col].strip())
            return out
        return len(values(explore_rows) & values(held_rows))
    return len(set(explore_rows) & set(held_rows))


def how(rule: dict[str, Any] | None) -> str:
    """The rule in plain words, for run.log and plan.md."""
    rule = rule or {}
    source = {"plan": "as plan.md's protocol says", "answer": "as you answered in plan.md",
              "columns": "decided from the table's columns"}.get(str(rule.get("source") or ""), "")
    strategy = rule.get("strategy")
    if strategy == TIME:
        text = f"the latest period, rows from {rule.get('time_column')} {rule.get('since')} on"
        if rule.get("embargo"):
            text += f", with {rule.get('embargo')} period(s) before it left out of both parts"
    elif strategy in (GROUP, SPATIAL):
        text = f"whole {rule.get('unit')} values picked at random, so each {rule.get('unit')} is on one side only"
    elif strategy == ROW:
        text = "rows picked at random (the rows are independent)"
    else:
        return "rows picked at random, set aside until it is decided which rows belong together"
    if rule.get("stratify_by") and strategy != TIME:
        text += f", the same share of each {rule.get('stratify_by')}"
    return f"{text}; {source}" if source else text


def how_in_paper(rule: dict[str, Any] | None) -> str:
    """The rule in words for the paper (no names or numbers from the data: every number in the paper is held to the
    results)."""
    strategy = (rule or {}).get("strategy")
    if strategy == TIME:
        return "its latest period"
    if strategy == SPATIAL:
        return "whole spatial blocks picked at random"
    if strategy == GROUP:
        return "whole units picked at random, each unit on one side only"
    return "rows picked at random"


# ---- the confirm run's numbers beside exploration's -----------------------------------------------------------------

#: Numbers that count rows or files: the held-back part is smaller than exploration's by design, so they are not compared.
_COUNTS = re.compile(r"^(?:n|count|rows|n_.*|.*_count|num_.*|number_of_.*)$", re.I)


def numbers(result: Any, prefix: str = "") -> dict[str, float]:
    """Every number in a result, by its path (``measurements.trust``; an object with a ``value`` counts as that value).
    Lists, counts of rows or files, and source ids are left out."""
    out: dict[str, float] = {}
    if not isinstance(result, dict):
        return out
    for key, value in result.items():
        name = str(key)
        if _COUNTS.match(name) or name in ("source_file_ids", "primary_sources"):
            continue
        path = f"{prefix}{name}"
        if isinstance(value, dict) and "value" in value and not isinstance(value["value"], (dict, list)):
            value = value["value"]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[path] = float(value)
        elif isinstance(value, dict):
            out.update(numbers(value, f"{path}."))
    return out


def compare(explore: Any, confirm: Any) -> dict[str, Any]:
    """``{"compared": n, "differs": [{"name", "explore", "confirm"}, ...]}`` for the numbers both results report. Two
    numbers differ when their signs differ or they are further apart than :data:`TOLERANCE` of the larger one."""
    a, b = numbers(explore), numbers(confirm)
    shared = [k for k in a if k in b]
    differs = []
    for k in shared:
        x, y = a[k], b[k]
        if x == y:
            continue
        if x * y < 0 or abs(x - y) > TOLERANCE * max(abs(x), abs(y)):
            differs.append({"name": k, "explore": x, "confirm": y})
    return {"compared": len(shared), "differs": differs}


def compare_lines(result: dict[str, Any]) -> list[str]:
    """What run.log says about the comparison (with the numbers: run.log is not held to the results)."""
    n = int(result.get("compared") or 0)
    differs = list(result.get("differs") or [])
    if not n:
        return ["confirm stage: the confirm run and exploration report no number by the same name, so their numbers "
                "could not be set side by side"]
    if not differs:
        return [f"confirm stage: the {n} number(s) both runs report agree (same sign, within a quarter of each other)"]
    listed = "; ".join(f"{d['name']}: exploration {d['explore']:g}, confirm {d['confirm']:g}" for d in differs[:8])
    more = f" and {len(differs) - 8} more" if len(differs) > 8 else ""
    return [f"confirm stage: on the held-back rows, {len(differs)} of the {n} number(s) both runs report differ from "
            f"exploration's by more than a quarter, or in sign ({listed}{more}); the paper reports the confirm run's "
            "numbers and says they differ (a total or a count over the rows is smaller on the held-back part by design)"]
