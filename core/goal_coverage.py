"""Does the experiment use the numbers the topic asks for?

A run graded on a topic that named its parameters ("for R0 in {0.9, 1.5, 3.0} and N in {100, 1000, 5000}") changed
them on its own: one drew its grid at {0.8, 1.2, 2.0, 3.0}, and another drew five figures for a topic that asked for
three. No later step compares the experiment with the topic, so the paper described a study nobody asked for and the
goals were met on paper only.

This module holds the parts that need no model and no engine state, and its output is advice, never a must-fix hit: a
topic is prose, a number in it may not be a parameter, and sending the experiment back costs a whole cycle.

* :func:`asked_numbers` reads the numbers a topic sets as parameters: the members of a set written in braces
  (``{0.9, 1.5, 3.0}``) and a count of runs (``300 stochastic runs``). A number in running prose (a wavelength, a
  threshold, a page limit) is not read, and neither is an integer below 10, which every script holds.
* :func:`code_numbers` reads the numbers an experiment script contains, and the values its ``linspace``, ``arange``,
  ``logspace``, ``geomspace`` and ``range`` calls with literal arguments generate.
* :func:`asked_figure_count` reads how many figures a topic asks for, and whether that is exact.
* :func:`notes` compares them.

A number counts as present when the script, or the keys of its results, holds that value. A number the topic gives
with a unit or a percent sign (193 nm, 5%) also counts at any power of ten, since a script works in metres or in
fractions: 193 nm is 1.93e-7, and 5% is 0.05.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "a single": 1,
}
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
# A set of settings: {0.9, 1.5, 3.0}, also after "in" or a membership sign.
_SET_RE = re.compile(r"\{(?P<body>[^{}]{3,160})\}")
# "300 stochastic runs per setting", "1,000 samples": a count of repeated runs.
_RUN_COUNT_RE = re.compile(
    r"(?<![\w.])(?P<n>\d{1,3}(?:,\d{3})+|\d+)\s+(?:[\w-]+\s+){0,2}"
    r"(?:runs?|samples?|trials?|replicates?|replications?|realisations?|realizations?|simulations?|iterations?|seeds?)\b",
    re.IGNORECASE,
)
# A unit or a percent sign right after a number or a set: the script may hold it at another power of ten.
_UNIT_AFTER_RE = re.compile(r"^\s?(?:%|[a-zA-Zµμ]{1,3}(?![a-zA-Z]))")
_NOT_A_UNIT = {"x", "d", "th", "st", "nd", "rd", "of", "in", "at", "to", "or", "is", "as", "an", "by", "on", "and", "the", "for"}
_FIGURES_RE = re.compile(
    r"(?P<before>\b(?:at least|at most|up to|no more than|no fewer than|not more than|maximum of|minimum of"
    r"|a maximum of|a minimum of)\s+)?"
    r"\b(?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten|a single)\s+(?:main\s+|separate\s+|distinct\s+)?figures?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Asked:
    """A number the topic states."""

    value: float
    text: str  # as the topic writes it
    where: str  # the words around it
    scaled: bool  # given with a unit or a percent sign, so any power of ten counts


def _value(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _mantissa(x: float) -> float | None:
    if x == 0 or not math.isfinite(x):
        return None
    return round(abs(x) / 10 ** math.floor(math.log10(abs(x))), 6)


def _phrase(text: str, start: int, end: int, tail: int = 0) -> str:
    """The words of ``text`` around ``start:end``, from the start of its sentence or line (a list marker left off)."""
    head = text[max(0, start - 70):start]
    for mark in ("\n", ". "):
        cut = head.rfind(mark)
        if cut >= 0:
            head = head[cut + len(mark):]
    head = re.sub(r"^\s*(?:\d{1,2}[.)]|[-*+])\s+", "", head)
    return " ".join((head + text[start:end + tail]).split())


def asked_numbers(topic: str) -> list[Asked]:
    """The numbers of ``topic`` that are parameters of an experiment, in the order they are written, once each: the
    members of a set written in braces (``{0.9, 1.5, 3.0}``, at least two numbers) and a count of runs (``300 stochastic
    runs``). A number in running prose (a wavelength in the context, a threshold, a target) is left alone: it is not a
    setting the experiment must sweep, and reporting it every time would drown the ones that are."""
    found: list[Asked] = []
    seen: set[float] = set()

    def add(token: str, where: str, scaled: bool) -> None:
        value = _value(token)
        if value is None or value == 0 or value in seen:
            return
        if abs(value) < 10 and "." not in token and "e" not in token.lower():
            return  # a small integer is in every script
        seen.add(value)
        found.append(Asked(value, token, " ".join(where.split()), scaled))

    text = topic or ""
    for group in _SET_RE.finditer(text):
        tokens = _NUMBER_RE.findall(group.group("body"))
        if len(tokens) < 2:
            continue
        body = group.group("body")
        unit = _UNIT_AFTER_RE.match(text[group.end():group.end() + 8])
        after_set = bool(unit) and unit.group(0).strip().lower() not in _NOT_A_UNIT
        where = _phrase(text, group.start(), group.end())
        for number in _NUMBER_RE.finditer(body):
            unit = _UNIT_AFTER_RE.match(body[number.end():number.end() + 6])
            own_unit = bool(unit) and unit.group(0).strip().lower() not in _NOT_A_UNIT
            add(number.group(0), where, after_set or own_unit)
    for run in _RUN_COUNT_RE.finditer(text):
        add(run.group("n").replace(",", ""), _phrase(text, run.start(), run.end(), 12), False)
    return found


def _literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    return None


def _generated(name: str, args: list[float]) -> list[float]:
    """The values a numeric range call with literal arguments produces (at most 200)."""
    try:
        if name in ("linspace", "logspace", "geomspace") and len(args) >= 3:
            a, b, n = args[0], args[1], int(args[2])
            if not 1 < n <= 200:
                return []
            if name == "linspace":
                return [a + (b - a) * i / (n - 1) for i in range(n)]
            if name == "logspace":
                return [10 ** (a + (b - a) * i / (n - 1)) for i in range(n)]
            return [a * (b / a) ** (i / (n - 1)) for i in range(n)] if a > 0 and b > 0 else []
        if name in ("arange", "range") and args:
            start, stop, step = (0.0, args[0], 1.0) if len(args) == 1 else (args[0], args[1], args[2] if len(args) > 2 else 1.0)
            count = int(math.ceil((stop - start) / step)) if step else 0
            return [start + i * step for i in range(count)] if 0 < count <= 200 else []
    except (ValueError, ZeroDivisionError, OverflowError):
        return []
    return []


def code_numbers(source: str) -> set[float]:
    """The numbers ``source`` (Python) contains: its numeric literals and the values its range calls with literal
    arguments generate. A script that does not parse gives every number in its text."""
    values: list[float] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {v for t in _NUMBER_RE.findall(source) if (v := _value(t)) is not None}
    for node in ast.walk(tree):
        lit = _literal(node)
        if lit is not None:
            values.append(lit)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name in ("linspace", "logspace", "geomspace", "arange", "range"):
                args = [_literal(a) for a in node.args]
                if args and all(a is not None for a in args):
                    values += _generated(name, [a for a in args if a is not None])
    return set(values)


def result_key_numbers(result: Any) -> set[float]:
    """The numbers in the keys of a results dictionary, at any depth (``R0_1.5_N_1000``)."""
    values: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                # An underscore is a word character to the number pattern, and a key is written R0_1.5_N_1000.
                values.update(v for t in _NUMBER_RE.findall(str(key).replace("_", " ")) if (v := _value(t)) is not None)
                walk(child)
        elif isinstance(node, list):
            for child in node[:200]:
                walk(child)

    walk(result)
    return values


def _present(asked: Asked, values: set[float]) -> bool:
    if any(math.isclose(asked.value, v, rel_tol=1e-9, abs_tol=1e-12) for v in values):
        return True
    if not asked.scaled:
        return False
    target = _mantissa(asked.value)
    return target is not None and any(_mantissa(v) == target for v in values)


def asked_figure_count(topic: str) -> tuple[int, str] | None:
    """``(n, kind)`` for the figures a topic asks for, where ``kind`` is ``"exact"``, ``"at least"`` or ``"at most"``;
    ``None`` when it names no number of figures or names two different ones."""
    counts: set[tuple[int, str]] = set()
    for m in _FIGURES_RE.finditer(topic or ""):
        raw = m.group("n").lower()
        n = int(raw) if raw.isdigit() else _WORD_NUMBERS.get(raw)
        if not n:
            continue
        before = (m.group("before") or "").strip().lower()
        kind = "at least" if before.startswith(("at least", "no fewer", "minimum", "a minimum")) else (
            "at most" if before else "exact")
        counts.add((n, kind))
    return next(iter(counts)) if len(counts) == 1 else None


def notes(topic: str, scripts: list[str], result: Any = None, figure_count: int | None = None) -> list[str]:
    """What the experiment does differently from what the topic asks, one line each; empty when it follows the topic
    or the topic asks for nothing checkable. ``scripts`` are the experiment's Python sources: a number the topic
    states that appears in none of them, nor in the keys of ``result``, is reported, and so is a figure count that
    differs from what the topic asks for."""
    out: list[str] = []
    if scripts:
        present: set[float] = set()
        for source in scripts:
            present |= code_numbers(source)
        present |= result_key_numbers(result)
        missing = [a for a in asked_numbers(topic) if not _present(a, present)]
        for a in missing[:8]:
            out.append(
                f"The topic gives {a.text} (\"...{a.where}...\"), and neither the experiment's code nor the keys of "
                "its results contain that number."
            )
        if len(missing) > 8:
            out.append(f"{len(missing) - 8} more numbers of the topic are not in the experiment either.")
    asked = asked_figure_count(topic)
    if asked is not None and figure_count:  # a run that drew none failed, which other checks report
        n, kind = asked
        if (kind == "exact" and figure_count != n) or (kind == "at least" and figure_count < n) or (
            kind == "at most" and figure_count > n
        ):
            what = {"exact": "", "at least": "at least ", "at most": "at most "}[kind]
            out.append(f"The topic asks for {what}{n} figure{'s' if n != 1 else ''}; the run drew {figure_count}.")
    return out
