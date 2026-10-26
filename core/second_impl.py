"""A second, independently written version of the key functions, and the engine's comparison of the two.

The model's package is written one function at a time (:mod:`core.function_steps`), each checked against a worked example
FI computes from the plan's own formula. A function with no example, or one whose example happens to agree with a
misreading, can still be wrong. Here a second writer, who sees only the description of the function (its card or
equation, the literature passages, the signatures of the functions it calls, the example INPUTS) and never the first
version or any expected output, writes the function again. FI then calls both on a small set of inputs it builds itself
and compares the answers:

* ``agree``: the two give the same results on every input both can take. Weaker evidence when both writers are the same
  model family (``same_family`` in the record says so).
* ``disagree``: a judge reads the description and both versions and says ``main_wrong`` (the main version is repaired,
  and the repair is kept only if the function still passes its own tests), ``reference_wrong`` (the second version is
  rewritten once) or ``card_ambiguous``. At most :data:`MAX_ROUNDS` rounds; what is still open is ``unresolved``.
* ``unconfirmed``: the comparison could not be made (no inputs, a version that would not load, a request that was lost,
  the call limit). Never a repair, never a stop.

The inputs come from the plan's worked example when it has one; otherwise from FI's own record of what the function is
called with when the simulation runs once at a setting of the plan (:func:`capture_inputs`, through FI's trial harness),
so a function the plan gives no example for is still compared, on the very inputs the study gives it.

``unresolved`` and ``unconfirmed`` come with one plain sentence for QUALITY.md and the paper's limitations
(:func:`limitations`); an ``unresolved`` one is an item FI fills into the paper's Limitations itself and checks is there
(:mod:`core.paper_facts`). Nothing here stops a quest and nothing asks a person.

The model is reached through injected callbacks, so the engine's per-step providers decide who writes and who judges:
``chat(node, messages)`` with the nodes ``second_impl`` and ``second_impl.adjudicate``; ``run(argv)`` runs
``python <argv>`` from the quest folder in the quest's environment (as :class:`core.function_steps.FunctionFiller` does);
``repair_main`` and ``still_passes`` belong to the main implementation. A model never supplies an expected number.

Bounds and resume. Every request is counted in ``.fi/second_impl.json`` BEFORE it is sent (``calls``, at most
:func:`max_calls_for`), and the judge's verdict is saved as soon as it is read, so a resume reuses finished functions,
repeats no request, and a request sent whose answer was not saved is not sent again (that function is ``unconfirmed``).

The second versions live in ``<quest>/.fi/second_impl/<function>.py``, never in ``code/`` (the gate receipt hashes
``code/``). Pure helpers (key function choice, inputs, comparison) are importable and tested on their own.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
import string
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import equation_tests as _eqt
from . import function_steps as _fs
from . import oracle_forms as _forms

RECORD = Path(".fi") / "second_impl.json"
DIR = Path(".fi") / "second_impl"
#: What each key function was called with when FI ran the simulation once (:func:`capture_inputs`).
CAPTURED = DIR / "captured_inputs.json"
#: The calls kept per function, and the most numbers kept from one call.
CAPTURE_CALLS = 3
CAPTURE_ELEMENTS = 20000
#: Fixed limits (not settings).
MAX_KEY = 6
MAX_ROUNDS = 2
MAX_INPUTS = 8
MIN_INPUTS = 4
REL = 1e-6
#: Below this, a result that is exactly 0 on one side agrees with a round-off result on the other.
ZERO_FLOOR = 1e-12
MAX_REFERENCE_LINES = 120

NODE_WRITE, NODE_ADJUDICATE = "second_impl", "second_impl.adjudicate"
AGREE, DISAGREE, UNCONFIRMED, UNRESOLVED, SKIPPED, WORKING = "agree", "disagree", "unconfirmed", "unresolved", "skipped", "working"
FINAL = (AGREE, UNRESOLVED, UNCONFIRMED, SKIPPED)
MAIN_WRONG, REFERENCE_WRONG, AMBIGUOUS = "main_wrong", "reference_wrong", "card_ambiguous"
_RANDOM_PARAMS = {*_eqt.RNG_PARAMS, _eqt.SEED_PARAM}

Message = dict[str, str]
ChatFn = Callable[[str, list[Message]], Awaitable[str]]
RunFn = Callable[[list[str]], Awaitable[tuple[int, str, str]]]
#: ``repair_main(name, problem, cases)``: try once to repair the main function in ``code/``; ``True`` when a new version
#: was written there. ``problem`` is plain text for the repairing prompt; ``cases`` are the disagreeing inputs.
RepairFn = Callable[[str, str, list[dict[str, Any]]], Awaitable[bool]]
#: ``still_passes(name)``: the function's own example and relation tests, run again on what is now in ``code/``.
PassesFn = Callable[[str], Awaitable[bool]]
FamilyFn = Callable[[str], "str | None"]


def max_calls_for(cap: int = MAX_KEY) -> int:
    """The most requests the step may make in a quest: ``3 * cap + 6``."""
    return 3 * int(cap) + 6


def _text(value: Any, limit: int = 400) -> str:
    return " ".join(str(value if value is not None else "").split())[:limit]


# --- which functions ------------------------------------------------------------------------------------------------------


def _is_random(spec: _fs.Spec, protocol: dict[str, Any] | None) -> str:
    """Why a function is skipped for being random, or ``""``."""
    try:
        params = {p.lstrip("*") for p in spec.params()}
    except SyntaxError:
        params = set()
    if params & _RANDOM_PARAMS:
        return f"its result is a random draw (it takes {', '.join(sorted(params & _RANDOM_PARAMS))}); two draws cannot be compared exactly"
    unset = (None, False, "", 0, "false", "False")
    if spec.card and spec.card.get("random") not in unset:
        return "the step's card says its result is a random draw; two draws cannot be compared exactly"
    if spec.kind == _fs.EQUATION:
        eq = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == spec.equation), None)
        if eq is not None and _eqt.is_random(eq):
            return "the equation's result is a random draw; two draws cannot be compared exactly"
    return ""


def _metric_text(protocol: dict[str, Any] | None) -> tuple[str, str]:
    """``(headline metric text, text of all metrics)`` from the protocol: ``precision.metric`` is the headline, the
    ``metrics`` list holds the rest (every string in an entry counts)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    precision = protocol.get("precision") if isinstance(protocol.get("precision"), dict) else {}
    headline = _text(precision.get("metric"), 200)
    parts: list[str] = []
    metrics = protocol.get("metrics")
    for m in metrics if isinstance(metrics, list) else []:
        parts.append(json.dumps(m, default=str) if isinstance(m, (dict, list)) else str(m))
    allm = " ".join(parts)
    if headline:
        # a metric entry whose id is the headline contributes its whole text to the headline
        head_parts = [p for p in parts if headline.lower() in p.lower()]
        return (headline + " " + " ".join(head_parts)).strip(), (allm + " " + headline).strip()
    return "", allm


def _mentions(text: str, spec: _fs.Spec) -> bool:
    low = text.lower()
    if not low.strip():
        return False
    keys = [spec.name]
    if spec.kind == _fs.EQUATION:
        keys.append(spec.equation)
    elif spec.card and spec.card.get("name"):
        keys.append(str(spec.card["name"]))
    for k in keys:
        k = str(k).strip().lower()
        if k and re.search(r"(?<![a-z0-9_])" + re.escape(k) + r"(?![a-z0-9_])", low):
            return True
    return False


def feeds(specs: list[_fs.Spec], protocol: dict[str, Any] | None) -> dict[str, int]:
    """``{function: 0 feeds the headline metric, 1 feeds another metric}``; absent = feeds none. A function feeds a metric
    when the metric's text names it (its name, its equation id or its step's name), or when a function that does depends
    on it (``depends_on``) or, for a step, uses its equation (``uses``)."""
    headline, everything = _metric_text(protocol)
    by_name = {s.name: s for s in specs}
    by_eq = {s.equation: s for s in specs if s.kind == _fs.EQUATION}
    out: dict[str, int] = {}
    for rank, text in ((1, everything), (0, headline)):  # the headline is written last so it wins
        todo = [s for s in specs if _mentions(text, s)]
        seen: set[str] = set()
        while todo:
            s = todo.pop()
            if s.name in seen:
                continue
            seen.add(s.name)
            out[s.name] = rank
            for d in s.depends_on:
                if d in by_name:
                    todo.append(by_name[d])
            for u in (s.card or {}).get("uses") or []:
                if _eqt._text(u) in by_eq:
                    todo.append(by_eq[_eqt._text(u)])
    return out


def key_functions(specs: list[_fs.Spec], protocol: dict[str, Any] | None, cap: int = MAX_KEY
                  ) -> tuple[list[_fs.Spec], dict[str, str]]:
    """``(the functions to write a second version of, {skipped name: why})``. Order: steps with no usable worked example,
    then functions that feed the headline metric, then those that feed another metric, then equations, then the other steps.
    A function whose result is random is skipped (and said). At most ``cap`` are returned."""
    skipped: dict[str, str] = {}
    live: list[_fs.Spec] = []
    for s in specs:
        why = _is_random(s, protocol)
        if why:
            skipped[s.name] = why
        else:
            live.append(s)
    feed = feeds(live, protocol)

    def tier(s: _fs.Spec) -> tuple[int, int]:
        if s.kind == _fs.STEP and (s.card or {}).get("example_state") != "ok":
            return 0, 0
        if s.name in feed:
            return 1, feed[s.name]
        if s.kind == _fs.EQUATION:
            return 2, 0
        return 3, 0

    ordered = sorted(range(len(live)), key=lambda i: (*tier(live[i]), i))
    return [live[i] for i in ordered][: max(0, int(cap))], skipped


# --- inputs (built by FI, deterministic) -----------------------------------------------------------------------------------


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_array(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(_is_num(x) or isinstance(x, list) for x in v)


def _scale(v: Any, f: float) -> Any:
    if isinstance(v, list):
        return [_scale(x, f) for x in v]
    if isinstance(v, int):
        return v * 2 if f == 2.0 else v  # an integer stays an integer: only doubling keeps it one
    return float(v) * f if v != 0 else f / 2.0


def _flat(v: Any) -> list[float]:
    return [y for x in v for y in _flat(x)] if isinstance(v, list) else [float(v)]


def _shift(v: Any, d: float) -> Any:
    if isinstance(v, list):
        return [_shift(x, d) for x in v]
    return v + int(d) if isinstance(v, int) else float(v) + d


def domain_of(spec: _fs.Spec, protocol: dict[str, Any] | None, base: dict[str, Any] | None) -> Callable[[dict[str, Any]], bool] | None:
    """Whether the plan's own worked example still defines the function at an input: a predicate on a case, or ``None``
    when there is none to apply. The example's ``expected_formula`` is the plan's statement of the function's value in
    terms of its inputs, and FI's calculator computes it (:func:`core.oracle_forms.formula_value`); an input at which the
    formula has no real value (a square root of a negative number, a division by zero) is outside the function's domain,
    where both versions are free to answer anything, so a difference there says nothing about either. Only used when the
    formula is computable at the example's own inputs (so it is read, and its names are the inputs'), and only for
    inputs that are all plain numbers; every other case is taken as defined."""
    example: Any = None
    if spec.card:
        example = spec.card.get("example")
    else:
        eq = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == spec.equation), None)
        example = (eq or {}).get("example")
    formula = example.get("expected_formula") if isinstance(example, dict) else None
    formulas = list(formula.values()) if isinstance(formula, dict) else [formula]
    if (not formulas or any(f is None or not str(f).strip() for f in formulas) or not isinstance(base, dict)
            or not all(_is_num(v) for v in base.values())):
        return None

    def value(case: dict[str, Any], f: Any) -> float | None:
        return _forms.formula_value({"expected_formula": f, "case": case})[0]

    if any(value(base, f) is None for f in formulas):
        return None  # not readable from the example's inputs: nothing says where the function is defined

    def defined(case: dict[str, Any]) -> bool:
        if not all(_is_num(v) for v in case.values()):
            return True
        return all(value(case, f) is not None for f in formulas)

    return defined


def build_inputs(base: dict[str, Any], ranges: dict[str, Any] | None = None, limit: int = MAX_INPUTS,
                 defined: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
    """The inputs both versions are called with: the example inputs; every scalar at x0.5, x2 and x0.9; arrays rescaled and
    shifted; the edges of a known range. Deterministic; duplicates dropped; at most ``limit``. Strings and booleans are
    kept as they are. A function with only such inputs is called on the example alone. ``defined`` (:func:`domain_of`):
    a case outside the function's domain is left out (the example itself always stays), so the scaled copies of the
    example never ask a function a question its own formula cannot answer."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(case: dict[str, Any], *, first: bool = False) -> None:
        key = json.dumps(case, sort_keys=True, default=str)
        if key not in seen and len(out) < limit and (first or defined is None or defined(case)):
            seen.add(key)
            out.append(case)

    add(dict(base), first=True)
    scalars = [k for k, v in base.items() if _is_num(v)]
    arrays = [k for k, v in base.items() if _is_array(v)]
    for f in (0.5, 2.0, 0.9):
        if scalars:
            add({k: (_scale(v, f) if k in scalars else v) for k, v in base.items()})
    if arrays:
        add({k: (_scale(v, 2.0) if k in arrays else v) for k, v in base.items()})
        sizes = [max(abs(x) for x in _flat(base[k])) or 1.0 for k in arrays]
        d = 0.1 * max(sizes)
        add({k: (_shift(v, d) if k in arrays else v) for k, v in base.items()})
    edges = {k: r for k, r in (ranges or {}).items()
             if k in base and isinstance(r, (list, tuple)) and len(r) == 2 and all(_is_num(x) for x in r)}
    if edges:
        for side in (0, 1):
            add({k: (edges[k][side] if k in edges else v) for k, v in base.items()})
    if len(out) < MIN_INPUTS:  # too few distinct cases: one scalar at a time, doubled
        for k in scalars:
            add({kk: (_scale(v, 2.0) if kk == k else v) for kk, v in base.items()})
    return out


def _signature_defaults(signature: str) -> dict[str, Any] | None:
    """A case from the signature's own defaults, when every parameter has a plain one; else ``None``."""
    try:
        node = ast.parse(signature.rstrip().rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return None
    if not isinstance(node, ast.FunctionDef):
        return None
    a = node.args
    pos = [*a.posonlyargs, *a.args]
    defaults: dict[str, Any] = {}
    for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        defaults[arg.arg] = d
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            defaults[arg.arg] = d
    case: dict[str, Any] = {}
    for arg in [*pos, *a.kwonlyargs]:
        node_d = defaults.get(arg.arg)
        if node_d is None:
            return None
        try:
            case[arg.arg] = ast.literal_eval(node_d)
        except (ValueError, SyntaxError):
            return None
    return case or None


def example_inputs(spec: _fs.Spec, protocol: dict[str, Any] | None,
                   captured: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any] | None:
    """The inputs of the plan's worked example (the INPUTS only, never the expected value), else the first call FI
    recorded when the simulation ran (``captured``, :func:`capture_inputs`), else the signature's defaults when it has
    them for every parameter, else ``None``."""
    row: dict[str, Any] | None = None
    if spec.kind == _fs.STEP and spec.card and spec.card.get("example") is not None:
        pseudo = {"model": {"equations": [{"id": spec.equation, "formula": spec.card.get("does") or spec.card.get("name") or "step",
                                           "role": "analyses", "example": spec.card["example"],
                                           "random": spec.card.get("random")}]}}
        row = next((r for r in _eqt.example_rows(pseudo) if r["id"] == spec.equation), None)
    elif spec.kind == _fs.EQUATION:
        row = next((r for r in _eqt.example_rows(protocol) if r["id"] == spec.equation), None)
    if row is not None and row.get("state") == "ok" and row.get("inputs"):
        try:
            params = {p for p in spec.params() if not p.startswith("*")}
        except SyntaxError:
            params = set()
        case = {k: v for k, v in row["inputs"].items() if k in params}
        if case:
            return case
    calls = (captured or {}).get(spec.name) or []
    if calls and isinstance(calls[0], dict) and calls[0]:
        return dict(calls[0])
    return _signature_defaults(spec.signature)


async def capture_inputs(run: Callable[[list[str]], Awaitable[Any]], quest_root: Path | str, package: str,
                         names: list[str], cells: list[dict[str, Any]], *, deterministic: bool,
                         in_each_run: dict[str, list[Any]] | None = None,
                         thresholds: dict[str, Any] | None = None,
                         budget_s: float | None = None) -> dict[str, list[dict[str, Any]]]:
    """FI runs the simulation at each of ``cells`` in turn (settings of the plan; each part of a study takes other paths
    through the code) through its own trial harness, with each of ``names`` (functions of the package's model) recording
    the distinct calls it gets, each argument by its parameter's name, kept only when every argument is plain data
    (numbers, text, real arrays of at most :data:`CAPTURE_ELEMENTS` numbers); it stops once every function has
    :data:`CAPTURE_CALLS` calls. ``run(argv)`` runs ``python <argv>`` from the quest folder. Saved in :data:`CAPTURED`;
    ``{}`` when no run could be made. Never an expected value: only what each function is given."""
    from . import trial_runner as _trials

    root = Path(quest_root)
    folder = root / DIR / "capture"
    folder.mkdir(parents=True, exist_ok=True)
    (root / _trials.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
    (root / CAPTURED).unlink(missing_ok=True)
    out: dict[str, list[dict[str, Any]]] = {}
    called_at: dict[str, list[int]] = {}
    ran: list[int] = []
    returned: dict[str, set[str]] = {}
    deadline = time.monotonic() + budget_s if budget_s else None
    for index, cell in enumerate(cells):
        if all(len(out.get(n, [])) >= CAPTURE_CALLS for n in names) and all(n in called_at for n in names):
            break
        if deadline is not None and index and time.monotonic() > deadline:
            break  # the time given is spent: the settings run so far are the record
        got = await _run_instrumented(run, root, package, names, cell, f"calls{index}", deterministic=deterministic,
                                      in_each_run=in_each_run, thresholds=thresholds)
        if got is None:
            continue
        if (got.get("row") or {}).get("status") == "ok":
            ran.append(index)
        for n, count in (got.get("counts") or {}).items():
            if str(n) in names and isinstance(count, int) and count > 0:
                called_at.setdefault(str(n), []).append(index)
        for n, keys in (got.get("returned") or {}).items():
            if str(n) in names and isinstance(keys, list):
                returned.setdefault(str(n), set()).update(str(k) for k in keys)
        for n, calls in (got.get("cases") or {}).items():
            if str(n) not in names or not isinstance(calls, list):
                continue
            kept = out.setdefault(str(n), [])
            for c in calls:
                if isinstance(c, dict) and c and c not in kept and len(kept) < CAPTURE_CALLS:
                    kept.append(c)
    out = {n: calls for n, calls in out.items() if calls}
    (root / CAPTURED).write_text(json.dumps({"cases": out, "called_at": called_at, "ran": ran,
                                             "returned": {n: sorted(k) for n, k in returned.items()},
                                             "cells": [dict(c) for c in cells]}, default=str), encoding="utf-8")
    return out


async def _run_instrumented(run: Callable[[list[str]], Awaitable[Any]], root: Path, package: str, names: list[str],
                            cell: dict[str, Any], tag: str, *, deterministic: bool,
                            in_each_run: dict[str, list[Any]] | None, thresholds: dict[str, Any] | None,
                            shift: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One run of the simulation at ``cell`` through FI's trial harness with ``names`` instrumented (and, with ``shift``
    ``{"name", "scale"}``, that function's result changed by that share): ``{"cases", "counts", "shifted", "row"}`` (the
    trial's row as the harness wrote it), ``None`` when the run could not be made."""
    from . import trial_runner as _trials

    folder = root / DIR / "capture"
    plan = _trials._plan(root, "code/simulate.py", {k: [v] for k, v in cell.items()}, runs_per_setting=1, base_seed=0,
                         deterministic=deterministic, folder=folder, out_name=tag + "-{index}.out.jsonl",
                         thresholds=thresholds, in_each_run=in_each_run)
    if not plan:
        return None
    task = plan[0]
    spec_path = root / task["spec"]
    part = DIR / "capture" / f"{tag}.json"
    (root / part).unlink(missing_ok=True)
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        data["capture"] = {"module": f"{package}.model", "names": list(names), "path": part.as_posix(),
                           "limit": CAPTURE_CALLS, "max_elements": CAPTURE_ELEMENTS, "shift": shift}
        spec_path.write_text(json.dumps(data, default=str), encoding="utf-8")
        # The record's path is named on the command line too, so a run made from a copy of the quest gives it back.
        await run([_trials.HARNESS_PATH.as_posix(), task["spec"], task["out"], part.as_posix()])
        got = json.loads((root / part).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- no record of the run
        return None
    if not isinstance(got, dict):
        return None
    row: dict[str, Any] = {}
    try:
        for line in (root / task["out"]).read_text(encoding="utf-8").splitlines():
            item = json.loads(line) if line.strip().startswith("{") else None
            if isinstance(item, dict) and item.get("nonce") == task["nonce"] and "trial" in item:
                row = item
    except (OSError, ValueError):
        pass
    return {**got, "row": row}


def read_captured(quest_root: Path | str) -> dict[str, Any]:
    """The record :func:`capture_inputs` saved: ``cases`` (calls by function), ``called_at`` (the indices of the cells
    each function was called at), ``ran`` (the indices of the cells that ran to the end), ``returned`` (for a function that
    returned a dict, every key it returned) and ``cells``; ``{}`` when there is none."""
    try:
        data = json.loads((Path(quest_root) / CAPTURED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and "cases" in data else {}


#: Where FI keeps which checked functions the headline results depend on (:func:`dependence`).
DEPENDENCE = DIR / "dependence.json"
#: The verdicts of :func:`dependence`. ``stops_when_changed``: the simulation stops when the function's result is changed
#: (it may compare its own number with it); neither shown to feed the headline results nor shown not to.
FEEDS, NOT_FEEDING, NOT_CALLED, NOT_RUN, STOPS = ("feeds", "does_not_feed", "not_called", "could_not_run",
                                                 "stops_when_changed")
#: The shares by which a checked function's result is changed, in turn: the second only when the run stops at the first.
SHIFTS = (0.05, 0.005)
#: At most this many of the settings a function was called at are tried (the first, the last, one between).
DEPENDENCE_CELLS = 3


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_value(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if _is_num(a) and _is_num(b):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return fa == fb or abs(fa - fb) <= 1e-12 * max(1.0, abs(fa), abs(fb))
    return a == b


def _has_finite(v: Any) -> bool:
    """Whether ``v`` holds at least one finite number (a number, or a dict or list of them)."""
    if isinstance(v, dict):
        return any(_has_finite(x) for x in v.values())
    if isinstance(v, list):
        return any(_has_finite(x) for x in v)
    return _is_num(v) and math.isfinite(float(v))


def _spread_cells(where: list[int]) -> list[int]:
    """The first, the last and one between of the settings a function was called at."""
    if len(where) <= DEPENDENCE_CELLS:
        return list(where)
    return sorted({where[0], where[len(where) // 2], where[-1]})


async def dependence(run: Callable[[list[str]], Awaitable[Any]], quest_root: Path | str, package: str,
                     names: list[str], headline: set[str], *, deterministic: bool,
                     in_each_run: dict[str, list[Any]] | None = None,
                     thresholds: dict[str, Any] | None = None, budget_s: float | None = None) -> dict[str, Any]:
    """Whether the headline results are computed with each checked function of ``names``. At up to
    :data:`DEPENDENCE_CELLS` settings where the function was called (:func:`read_captured`), FI runs the simulation once
    as it is and once with every real number the function returns changed by up to :data:`SHIFTS` (5%; 0.5% when the run
    stops at 5%). A headline result that changes depends on it (``feeds``), once a second run as it is gives the same
    numbers as the first (a result that differs between the two is left out of the comparison; when none repeats,
    ``could_not_run``). Headline results that come out the very
    same at every setting tried do not (``does_not_feed``: the checks of that function do not check them). A run that
    stops whenever the result is changed is ``stops_when_changed`` (neither). A function never called at the settings
    that ran is ``not_called``. ``headline``: the names of the headline results, compared as the headline check compares
    them; only finite numbers of the run as it is are compared (every result it returns when a headline result is not
    among them, since that one is computed later from them; said so).
    Only references to the function through its module are reached (a copy kept before the run, in a table, is not).
    Saved in :data:`DEPENDENCE` with the hash of the code it was made on. Never a stop."""
    from . import criteria as _criteria
    from . import hidden_check as _hidden

    root = Path(quest_root)
    record = read_captured(root)
    cells = record.get("cells") or []
    called_at = record.get("called_at") or {}
    ran = [i for i in record.get("ran") or [] if isinstance(i, int)]
    keys = {_criteria._key(h) for h in headline}
    plain_runs: dict[tuple[int, int], dict[str, Any]] = {}
    deadline = time.monotonic() + budget_s if budget_s else None

    def late() -> bool:
        return deadline is not None and time.monotonic() > deadline

    async def plain(i: int, again: int = 0) -> dict[str, Any]:
        if (i, again) not in plain_runs:
            got = await _run_instrumented(run, root, package, [], cells[i], f"plain{i}" + ("b" if again else ""),
                                          deterministic=deterministic, in_each_run=in_each_run, thresholds=thresholds)
            plain_runs[(i, again)] = (got or {}).get("row") or {}
        return plain_runs[(i, again)]

    def _whole(value: Any) -> bool:
        items = value if isinstance(value, (list, tuple)) else [value]
        nums = [float(v) for v in items if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        return bool(nums) and all(v == int(v) for v in nums)

    async def at_cell(name: str, i: int) -> dict[str, Any]:
        if late() and (i, 0) not in plain_runs:
            return {"status": NOT_RUN, "why": "the time given to this check was spent"}
        base = await plain(i)
        if base.get("status") != "ok":
            return {"status": NOT_RUN, "why": "the setting did not run as it is"}
        values = base.get("values") or {}
        finite = [k for k in values if _has_finite(values[k])]
        judged = [k for k in finite if _criteria._key(k) in keys]
        # A headline result the run does not return is computed later from what it returns (an average over runs, a
        # fit across settings): then every result the run returns may feed it, and all of them are compared.
        fallback = not keys or not keys <= {_criteria._key(k) for k in values}
        judged = finite if fallback else judged
        if not judged:
            return {"status": NOT_RUN, "why": "the setting gave no number to compare"}
        stopped = False
        for scale in SHIFTS:
            if late():
                return {"status": NOT_RUN, "why": "the time given to this check was spent"}
            got = await _run_instrumented(run, root, package, [name], cells[i], f"shift_{name}_{i}",
                                          deterministic=deterministic, in_each_run=in_each_run, thresholds=thresholds,
                                          shift={"name": name, "scale": scale})
            row = (got or {}).get("row") or {}
            if not row:
                return {"status": NOT_RUN, "why": "the run with its result changed gave no result"}
            if not (got.get("counts") or {}).get(name):
                return {"status": NOT_RUN, "why": "it was not called when the setting was run again"}
            if not (got.get("shifted") or {}).get(name):
                return {"status": NOT_RUN, "why": "what it returns holds no number other than zero that FI can change"}
            if row.get("status") != "ok":
                stopped = True
                continue
            changed = [k for k in judged if not _same_value(values.get(k), (row.get("values") or {}).get(k))]
            if not changed and all(_whole(values.get(k)) for k in judged):
                return {"status": NOT_RUN, "why": "every result compared is a whole number (a count, a yes/no, a step "
                                                  "of a grid), which a change of a few percent need not move"}
            if not changed:
                return {"status": NOT_FEEDING, "results": judged[:8], "scale": scale, "all_results": fallback}
            if late():
                return {"status": NOT_RUN, "why": "the time given to this check was spent"}
            # A result that differs between two runs of the setting as it is (a time it took, an unseeded draw) says
            # nothing either way: only the results that repeat are judged.
            again = await plain(i, 1)
            if again.get("status") != "ok":
                return {"status": NOT_RUN, "why": "the setting did not run as it is a second time"}
            noisy = {k for k in judged if not _same_value(values.get(k), (again.get("values") or {}).get(k))}
            steady = [k for k in judged if k not in noisy]
            real = [k for k in changed if k not in noisy]
            if real:
                # Which of them are headline results by name: with the fallback, a value that moved may be one the
                # analysis never uses, so the record keeps the two apart.
                return {"status": FEEDS, "scale": scale, "all_results": fallback,
                        "headline_moved": [k for k in real if _criteria._key(k) in keys][:5],
                        "how": "changed with its result changed: " + ", ".join(real[:5])}
            if not steady:
                return {"status": NOT_RUN, "why": "the results change between two runs of the same setting"}
            if all(_whole(values.get(k)) for k in steady):
                return {"status": NOT_RUN, "why": "every result compared is a whole number (a count, a yes/no, a step "
                                                  "of a grid), which a change of a few percent need not move"}
            return {"status": NOT_FEEDING, "results": steady[:8], "scale": scale, "all_results": fallback,
                    "not_repeating": sorted(noisy)[:8]}
        return {"status": STOPS} if stopped else {"status": NOT_RUN, "why": "no run was made"}

    out: dict[str, Any] = {}
    for name in names:
        if not ran:
            out[name] = {"status": NOT_RUN, "why": "the simulation did not run to the end at the settings FI tried"}
            continue
        where = [i for i in called_at.get(name) or [] if isinstance(i, int) and 0 <= i < len(cells)]
        if not where:
            out[name] = {"status": NOT_CALLED, "settings_run": len(ran)}
            continue
        tried: list[dict[str, Any]] = []
        for i in _spread_cells(where):
            if late() and tried:
                break
            entry = {**await at_cell(name, i), "cell": cells[i]}
            tried.append(entry)
            if entry["status"] == FEEDS:
                break
        states = [e["status"] for e in tried]
        if FEEDS in states:
            out[name] = next(e for e in tried if e["status"] == FEEDS)
        elif states and all(s == NOT_FEEDING for s in states):
            out[name] = {**tried[0], "settings_tried": len(tried)}
        elif STOPS in states:
            out[name] = {"status": STOPS, "settings_tried": len(tried)}
        else:
            out[name] = next((e for e in tried if e["status"] == NOT_RUN), {"status": NOT_RUN, "why": "no run was made"})
    (root / DEPENDENCE).parent.mkdir(parents=True, exist_ok=True)
    (root / DEPENDENCE).write_text(json.dumps({"code_sha256": _hidden.code_sha(root), "names": list(names),
                                               "functions": out}, indent=1, default=str), encoding="utf-8")
    return out


def read_dependence(quest_root: Path | str) -> dict[str, Any]:
    """The verdicts :func:`dependence` saved, by function; ``{}`` when there is none or it was made on other code."""
    from . import hidden_check as _hidden

    try:
        data = json.loads((Path(quest_root) / DEPENDENCE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("code_sha256") != _hidden.code_sha(Path(quest_root)):
        return {}
    functions = data.get("functions")
    return functions if isinstance(functions, dict) else {}


def dependence_gaps(quest_root: Path | str) -> list[str]:
    """Evidence gaps: a checked function the headline results were shown not to use (its tests say nothing about them).
    A function never called at the settings FI ran is said, not a gap: a reference formula or a function of the analysis
    is not called by the simulation, and that is no fault."""
    return [line for name, e in read_dependence(quest_root).items() if (e or {}).get("status") == NOT_FEEDING
            for line in dependence_sentences({name: e})]


def dependence_sentences(record: dict[str, Any]) -> list[str]:
    """One plain sentence per checked function the headline results were not shown to use."""
    out = []
    for name, e in (record or {}).items():
        status = (e or {}).get("status")
        if status == NOT_FEEDING:
            share = f"{100 * float(e.get('scale') or SHIFTS[0]):g}%"
            what = "results" if e.get("all_results") else "headline results"
            out.append(f"the {what} do not use `{name}`: with its result changed by up to {share} where the simulation calls "
                       f"it, they came out the same, so the tests of `{name}` say nothing about them")
        elif status == NOT_CALLED:
            out.append(f"`{name}`, a function FI tested, was not called at the {e.get('settings_run')} setting(s) FI "
                       "ran, so FI could not show that the headline results are computed with it")
    return out


def ranges_of(spec: _fs.Spec, protocol: dict[str, Any] | None) -> dict[str, Any]:
    """``{input: [low, high]}`` when the card or equation (or its example) states a range for it, else ``{}``."""
    sources: list[Any] = []
    if spec.card:
        sources += [spec.card.get("ranges"), (spec.card.get("example") or {}).get("ranges") if isinstance(spec.card.get("example"), dict) else None]
    else:
        eq = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == spec.equation), None)
        if eq:
            sources += [eq.get("ranges"), (eq.get("example") or {}).get("ranges") if isinstance(eq.get("example"), dict) else None]
    for s in sources:
        if isinstance(s, dict):
            return {str(k): v for k, v in s.items()}
    return {}


def tolerance_for(spec: _fs.Spec, protocol: dict[str, Any] | None) -> float:
    """Relative tolerance of the comparison: :data:`REL`, or the example's own tolerance when it names a numerical method
    (an iterative method cannot be held to 1e-6 by its own tolerance). Never looser otherwise."""
    example: Any = None
    if spec.card:
        example = spec.card.get("example")
    else:
        eq = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == spec.equation), None)
        example = (eq or {}).get("example")
    if isinstance(example, dict) and _text(example.get("method")):
        t = _eqt._number(example.get("tolerance"))
        mode = _text(example.get("tolerance_mode")).lower() or "relative"
        if t is not None and t > 0 and mode == "relative":
            return max(REL, float(t))
    return REL


# --- comparing two results -------------------------------------------------------------------------------------------------


def decode(x: Any) -> Any:
    """A result as the comparison script wrote it (tagged non-finite floats, complex numbers, dicts, other types) back to
    Python values."""
    if isinstance(x, dict):
        if set(x) == {"f"}:
            return float(x["f"])
        if set(x) == {"c"}:
            return complex(decode(x["c"][0]), decode(x["c"][1]))
        if set(x) == {"d"}:
            return {k: decode(v) for k, v in x["d"].items()}
        if set(x) == {"r"}:
            return _Other(str(x["r"]))
    if isinstance(x, list):
        return [decode(i) for i in x]
    return x


class _Other(str):
    """A result of a type the script could not carry; two of the same type name are taken as equal."""


def _close(a: float, b: float, rel: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    if a == b:
        return True
    lo = min(abs(a), abs(b))
    if lo == 0.0:
        return max(abs(a), abs(b)) <= ZERO_FLOOR
    return abs(a - b) <= rel * lo


def same(a: Any, b: Any, rel: float = REL) -> bool:
    """Whether two decoded results agree: numbers within ``rel`` (relative; NaN with NaN; infinities by sign), integers,
    strings and booleans exactly, lists and dicts element by element with the same shape."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, _Other) or isinstance(b, _Other):
        return isinstance(a, _Other) and isinstance(b, _Other) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if _is_num(a) and _is_num(b):
        return _close(float(a), float(b), rel)
    if isinstance(a, complex) or isinstance(b, complex):
        try:
            ca, cb = complex(a), complex(b)
        except (TypeError, ValueError):
            return False
        return _close(ca.real, cb.real, rel) and _close(ca.imag, cb.imag, rel)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k], rel) for k in a)
    if a is None and b is None:
        return True
    return False


def compare_shared(a: Any, b: Any, rel: float = REL, path: str = "") -> tuple[int, bool, list[str], list[str]]:
    """Compare what both versions return, matched by name (dict keys present in both, at every depth) or position (lists
    of the same length). ``(values compared, whether one of them differs, notes, missing)``. ``missing`` holds the path of
    every key the first version (``a``) returns that the second lacks: it was not compared, so the step cannot be agree.
    A key only the second version returns is a note. Everything that is not a dict or a list of the same length as the
    other is judged by :func:`same`, so a real difference is judged as before."""
    if isinstance(a, dict) and isinstance(b, dict):
        notes: list[str] = []
        missing = [f"{path}.{k}" if path else str(k) for k in a if k not in b]
        extra = sorted(str(k) for k in b if k not in a)
        if extra:
            notes.append(f"{path or 'the result'}: only the second version returns {', '.join(extra)}; those were not compared")
        n, differs = 0, False
        for k in a:
            if k in b:
                cn, cd, cnotes, cmiss = compare_shared(a[k], b[k], rel, f"{path}.{k}" if path else str(k))
                n, differs = n + cn, differs or cd
                notes += cnotes
                missing += cmiss
        return n, differs, notes, missing
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and any(isinstance(x, (dict, list)) for x in a + b):
        n, differs, notes, missing = 0, False, [], []
        for i, (x, y) in enumerate(zip(a, b)):
            cn, cd, cnotes, cmiss = compare_shared(x, y, rel, f"{path}[{i}]")
            n, differs = n + cn, differs or cd
            notes += cnotes
            missing += cmiss
        return n, differs, notes, missing
    return 1, not same(a, b, rel), [], []


def short(value: Any, limit: int = 120) -> str:
    """A result in few characters for a sentence or a prompt."""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        shown = ", ".join(short(v, 30) for v in value[:6]) + (f", ... ({len(value)} values)" if len(value) > 6 else "")
        return "[" + shown + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {short(v, 30)}" for k, v in list(value.items())[:4]) + "}"
    return str(value)[:limit]


def judge_rows(rows: list[dict[str, Any]], inputs: list[dict[str, Any]], rel: float) -> dict[str, Any]:
    """The verdict on the rows the script returned (one per input: ``main`` / ``ref`` results, or ``main_err`` /
    ``ref_err``). Both raising excludes the input; one raising is a ``domain`` disagreement; otherwise the values are
    compared. ``{"status": agree|disagree|unconfirmed, "compared", "excluded", "disagreements": [{inputs, main, other, kind}]}``."""
    compared = excluded = unshared = 0
    missing: list[str] = []
    bad: list[dict[str, Any]] = []
    notes: list[str] = []
    for case, row in zip(inputs, rows):
        m_err, r_err = row.get("main_err"), row.get("ref_err")
        if m_err and r_err:
            excluded += 1
            continue
        compared += 1
        if m_err or r_err:
            bad.append({"inputs": case, "kind": "domain",
                        "main": f"raised {m_err}" if m_err else short(decode(row.get("main"))),
                        "other": f"raised {r_err}" if r_err else short(decode(row.get("ref")))})
        else:
            shared, differs, case_notes, case_missing = compare_shared(decode(row.get("main")), decode(row.get("ref")), rel)
            missing += [m for m in case_missing if m not in missing]
            notes += [n for n in case_notes if n not in notes]
            if differs:
                bad.append({"inputs": case, "kind": "value", "main": short(decode(row.get("main"))),
                            "other": short(decode(row.get("ref")))})
            elif shared == 0:
                unshared += 1
    if bad:
        status = DISAGREE
    elif compared == 0 or unshared == compared or missing:
        status = UNCONFIRMED
    else:
        status = AGREE
    out: dict[str, Any] = {"status": status, "compared": compared - unshared, "excluded": excluded, "disagreements": bad}
    if notes:
        out["shape_notes"] = notes[:3]
    if status == UNCONFIRMED and missing:
        out["why"] = f"the second version does not return {', '.join(missing[:3])}, so that was not compared"
    elif status == UNCONFIRMED and unshared:
        out["why"] = "the two versions return no value in common to compare"
    return out


_SCRIPT = r'''
import importlib, importlib.util, json, math, sys, traceback, warnings
sys.path.insert(0, @CODE@)
try:
    import numpy as np
except Exception:
    np = None


def enc(x):
    if np is not None:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, np.generic):
            x = x.item()
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else {"f": repr(x)}
    if isinstance(x, complex):
        return {"c": [enc(x.real), enc(x.imag)]}
    if isinstance(x, (list, tuple)):
        return [enc(i) for i in x]
    if isinstance(x, dict):
        return {"d": {str(k): enc(v) for k, v in x.items()}}
    return {"r": type(x).__name__}


def dec_in(v):
    if isinstance(v, list):
        if np is not None:
            try:
                return np.array(v)
            except Exception:
                pass
        return list(v)
    return v


def done(obj):
    print("SECOND_IMPL: " + json.dumps(obj))
    sys.exit(0)


try:
    main_mod = importlib.import_module(@MODULE@)
    main_fn = getattr(main_mod, @NAME@)
except BaseException:
    done({"main_import_error": traceback.format_exc()[-600:]})
try:
    sp = importlib.util.spec_from_file_location("second_impl_reference", @REF@)
    ref_mod = importlib.util.module_from_spec(sp)
    own = @OWN@
    for n in @USES@:
        if n in own:
            dsp = importlib.util.spec_from_file_location("second_impl_reference_" + n, own[n])
            dmod = importlib.util.module_from_spec(dsp)
            for m in @USES@:
                if m != n and hasattr(main_mod, m):
                    setattr(dmod, m, getattr(main_mod, m))
            dsp.loader.exec_module(dmod)
            setattr(ref_mod, n, getattr(dmod, n))
        elif hasattr(main_mod, n):
            setattr(ref_mod, n, getattr(main_mod, n))
    sp.loader.exec_module(ref_mod)
    ref_fn = getattr(ref_mod, @NAME@)
except BaseException:
    done({"ref_import_error": traceback.format_exc()[-600:]})
with open(@INPUTS@, encoding="utf-8") as fh:
    cases = json.load(fh)
rows = []
for case in cases:
    row = {}
    for tag, fn in (("main", main_fn), ("ref", ref_fn)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row[tag] = enc(fn(**{k: dec_in(v) for k, v in case.items()}))
        except BaseException as e:
            row[tag + "_err"] = type(e).__name__ + ": " + str(e)[:160]
    rows.append(row)
done({"rows": rows})
'''


def compare_script(*, code_dir: str, module: str, name: str, reference: str, inputs: str, uses: list[str],
                   own: dict[str, str] | None = None) -> str:
    """The ``-c`` script that imports both versions and calls each on every input; it prints one ``SECOND_IMPL: {json}`` line."""
    text = _SCRIPT
    for token, value in (("@CODE@", code_dir), ("@MODULE@", module), ("@NAME@", name), ("@REF@", reference),
                         ("@INPUTS@", inputs), ("@USES@", list(uses)), ("@OWN@", dict(own or {}))):
        text = text.replace(token, repr(value))
    return text


def parse_output(out: str) -> dict[str, Any] | None:
    for line in reversed((out or "").splitlines()):
        if line.startswith("SECOND_IMPL: "):
            try:
                data = json.loads(line[len("SECOND_IMPL: "):])
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


# --- the reference version ---------------------------------------------------------------------------------------------


def reference_problem(code: str, spec: _fs.Spec, package: str) -> str:
    """Why a reply cannot serve as the second version (``""`` when it can): it must parse, define the function with the same
    parameter names, and not import the study's own package."""
    if not code.strip():
        return "the reply held no Python"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"the reply is not valid Python ({e.msg})"
    node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == spec.name), None)
    if node is None:
        return f"the reply does not define {spec.name}"
    if _fs._param_names(node) != spec.params():
        return f"the reply's parameters ({', '.join(_fs._param_names(node))}) are not the signature's"
    if len(code.splitlines()) > MAX_REFERENCE_LINES:
        return "the reply is far longer than the function should be"
    for n in ast.walk(tree):
        mods = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module or ""] if isinstance(n, ast.ImportFrom) else [])
        if any(m.split(".")[0] == package for m in mods if package):
            return "the reply imports the study's own package"
    return ""


def label(spec: _fs.Spec) -> str:
    if spec.kind == _fs.STEP and spec.card:
        return f"step {spec.card['id']} ({spec.card.get('name') or spec.name})"
    return f"equation {spec.equation} ({spec.name})"


def _spec_block(spec: _fs.Spec, protocol: dict[str, Any] | None) -> str:
    """What the function is, for the writer and the judge: the card or the equation (never an expected value)."""
    if spec.kind == _fs.STEP and spec.card:
        c = spec.card
        lines = [f"{c['id']} (a step of the method): {c.get('name') or ''}".rstrip(), f"what it does: {c.get('does') or '(not stated)'}"]
        if c.get("inputs"):
            lines.append("inputs: " + "; ".join(f"{k} ({v})" if v else k for k, v in c["inputs"].items()))
        if c.get("output"):
            lines.append(f"returns: {c['output']}")
        for u in c.get("uses") or []:
            e = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == _eqt._text(u)), None)
            if e:
                lines.append(f"uses equation {u}: {_eqt._text(e.get('formula'))}")
        if c.get("source"):
            lines.append(f"source: {c['source']}")
        if c.get("relations"):
            lines.append("properties its result must have: " + "; ".join(c["relations"]))
        return "\n".join(lines)
    e = next((e for e in _eqt.eligible(protocol) if _eqt._text(e["id"]) == spec.equation), {"id": spec.equation, "formula": ""})
    lines = [f"{spec.equation}: {_eqt._text(e.get('formula'))}"]
    for key in ("units", "unit", "variables", "symbols", "source", "derivation"):
        v = e.get(key)
        if v:
            lines.append(f"{key}: " + (json.dumps(v, default=str)[:400] if isinstance(v, (dict, list)) else _eqt._text(v)[:400]))
    lines.append(f"purpose: {spec.purpose}")
    return "\n".join(lines)


def write_prompt(spec: _fs.Spec, *, template: string.Template, protocol: dict[str, Any] | None, specs: list[_fs.Spec],
                 passages: str, example: dict[str, Any] | None, topic_block: str = "", rewrite: str = "",
                 main_source: str = "", return_keys: list[str] | None = None) -> str:
    """The request that writes the second version. It holds the description, the signature, the passages, the signatures of
    the functions it calls and the example INPUTS. Raises ``ValueError`` if a line of the main implementation ended up in it."""
    by_name = {s.name: s for s in specs}
    related = "\n".join(f"- {by_name[d].signature.strip()}" for d in spec.depends_on if d in by_name) or "(none)"
    literature = ("# What the literature says (passages from the source the plan cites)\n\n" + passages.strip() + "\n") if passages.strip() else ""
    given = ", ".join(f"{k} = {short(v)}" for k, v in (example or {}).items())
    example_block = f"# Inputs it will be called with\n\nFI will call the function with, among others: {given}.\n" if given else ""
    rewrite_block = (f"# A first attempt was judged wrong\n\n{rewrite}\n") if rewrite else ""
    return_block = ("# What it must return\n\nWhen it returns a dict: across the inputs FI ran, the first version's dict results carried these keys: "
                    + ", ".join(return_keys) + ". A key may appear only for some inputs; for the same input return the same keys "
                    "under exactly these names.\n") if return_keys else ""
    prompt = template.substitute(topic_block=topic_block or "(the topic is not available)", step_block=_spec_block(spec, protocol),
                                 return_block=return_block,
                                 signature=spec.signature.strip(), related_block=related, literature_block=literature,
                                 example_block=example_block, rewrite_block=rewrite_block)
    sig = spec.signature.strip()
    for line in (ln.strip() for ln in main_source.splitlines()):
        if len(line) >= 30 and line != sig and not line.startswith(("def ", "#", '"""', "'''")) and line in prompt:
            raise ValueError("a line of the main implementation is in the second writer's request")
    return prompt


def _parse_json(text: str) -> dict[str, Any] | None:
    """The first balanced ``{...}`` of a reply that parses as an object."""
    text = text or ""
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except ValueError:
                    break
                if isinstance(obj, dict):
                    return obj
                break
    return None


def parse_verdict(reply: str) -> tuple[str, str]:
    """``(verdict, reason)``; anything unreadable is ``card_ambiguous``."""
    obj = _parse_json(reply) or {}
    verdict = _text(obj.get("verdict"), 40).lower().replace("-", "_").replace(" ", "_")
    if verdict not in (MAIN_WRONG, REFERENCE_WRONG, AMBIGUOUS):
        return AMBIGUOUS, _text(obj.get("reason"), 300) or "the judge's reply could not be read"
    return verdict, _text(obj.get("reason"), 300)


def cases_text(cases: list[dict[str, Any]]) -> str:
    return "\n".join("- " + ", ".join(f"{k} = {short(v)}" for k, v in c["inputs"].items())
                     + f": version 1 returned {c['main']}; version 2 returned {c['other']}" for c in cases)


def repair_problem(reason: str, cases: list[dict[str, Any]]) -> str:
    """The problem text given to the repair of the main function. It says another, separately written version returned other
    values; it never calls those values expected or correct."""
    return ("A second version of this function, written separately from the same description, returned different results "
            "on these inputs (neither is known to be right):\n" + cases_text(cases)
            + (f"\nA reader of the description says: {reason}" if reason else "")
            + "\nRead the description again and find out from it which result is right; if this function is wrong, correct it, "
              "and do not write any of these numbers into it.")


def limitation_unresolved(spec: _fs.Spec, cases: list[dict[str, Any]]) -> str:
    where = "; ".join(", ".join(f"{k} = {short(v, 30)}" for k, v in c["inputs"].items()) for c in cases[:2]) or "some inputs"
    return (f"Two independently written versions of {label(spec)} gave different results on {where}; "
            "the number depends on which is right.")


def limitation_unconfirmed(spec: _fs.Spec, why: str) -> str:
    return f"A second, independently written version of {label(spec)} could not be compared with the first ({why})."


# --- the record -----------------------------------------------------------------------------------------------------------


def load(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save(quest_root: Path | str, record: dict[str, Any]) -> None:
    path = Path(quest_root) / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def limitations(record: dict[str, Any]) -> list[str]:
    """The plain sentences for QUALITY.md and the paper's limitations: every ``unresolved`` or ``unconfirmed`` function's."""
    return [str(e["limitation"]) for e in (record.get("functions") or {}).values()
            if isinstance(e, dict) and e.get("status") in (UNRESOLVED, UNCONFIRMED) and e.get("limitation")]


def counts(record: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in (record.get("functions") or {}).values():
        if isinstance(e, dict):
            out[str(e.get("status"))] = out.get(str(e.get("status")), 0) + 1
    return out


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _function_source(model_path: Path, name: str) -> str:
    try:
        text = model_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        return ""
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(text, n) or ""
    return ""


# --- the step ---------------------------------------------------------------------------------------------------------------


class SecondImpl:
    """Runs the second-version check over the key functions. Nothing here knows about the engine."""

    def __init__(self, *, quest_root: Path | str, package: str, specs: list[_fs.Spec], protocol: dict[str, Any] | None,
                 prompts: dict[str, string.Template], chat: ChatFn, run: RunFn, repair_main: RepairFn, still_passes: PassesFn,
                 passages: dict[str, str] | None = None, family_of: FamilyFn | None = None, topic_block: str = "",
                 cap: int = MAX_KEY, log: Any = None,
                 verify_package: Callable[[], Awaitable[str]] | None = None,
                 captured: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.root, self.package, self.specs, self.protocol = Path(quest_root), package, list(specs), protocol
        self.prompts, self.chat, self.run = prompts, chat, run
        self.repair_main, self.still_passes = repair_main, still_passes
        #: ``verify_package()``: the model's other tests, run once on the package a repair left; ``""`` when none fails, else
        #: what fails (the repair is then undone).
        self.verify_package = verify_package
        self.passages = {str(k): str(v) for k, v in (passages or {}).items() if str(v).strip()}
        self.family_of, self.topic_block = family_of, topic_block
        self.cap, self.max_calls = int(cap), max_calls_for(cap)
        self.log = log or logging.getLogger(__name__)
        self.model_path = self.root / "code" / package / "model.py"
        self.record: dict[str, Any] = {}
        #: What each function was called with when FI ran the simulation once (:func:`capture_inputs`).
        self.captured = {str(k): list(v) for k, v in (captured or {}).items() if isinstance(v, list)}

    # -- bookkeeping --

    def _save(self) -> None:
        save(self.root, self.record)

    def _funcs(self) -> dict[str, Any]:
        return self.record.setdefault("functions", {})

    def _spend(self, entry: dict[str, Any], kind: str) -> bool:
        """Count one request BEFORE it is sent, and say which one is in flight. ``False`` when the limit is reached."""
        if int(self.record.get("calls") or 0) >= self.max_calls:
            return False
        self.record["calls"] = int(self.record.get("calls") or 0) + 1
        entry["pending"] = {"kind": kind, "round": int(entry.get("rounds") or 0)}
        self._save()
        return True

    def _done_call(self, entry: dict[str, Any]) -> None:
        entry.pop("pending", None)
        self._save()

    def _same_family(self) -> bool | None:
        if self.family_of is None:
            return None
        try:
            a, b = self.family_of(NODE_WRITE), self.family_of("implement")
        except Exception:  # noqa: BLE001 -- identity is a note, never a reason to stop
            return None
        return None if not a or not b else a == b

    def _finish(self, spec: _fs.Spec, entry: dict[str, Any], status: str, why: str = "", cases: list[dict[str, Any]] | None = None,
                recheck: bool = False) -> None:
        entry["status"] = status
        entry.pop("pending", None)
        entry.pop("recheck", None)
        if status == UNRESOLVED:
            entry["limitation"] = limitation_unresolved(spec, cases or entry.get("disagreements") or [])
            entry["why"] = why
        elif status == UNCONFIRMED:
            entry["limitation"] = limitation_unconfirmed(spec, why)
            entry["why"] = why
            if recheck:
                entry["recheck"] = True
        else:
            entry.pop("limitation", None)
            entry["why"] = why
        shared = entry.get("deps_from_first_version")
        if shared and status in (UNRESOLVED, UNCONFIRMED) and entry.get("limitation"):
            entry["limitation"] += (" The second version called " + ", ".join(shared) + " from the first version, so a fault "
                                    "in that function would not show.")
        self.log.info("[second_impl] %s: %s%s", label(spec), status, f" ({why})" if why else "")
        self._save()

    def _orig_path(self, name: str) -> Path:
        return self.root / DIR / f"{name}.orig.py"

    def _restore_unverified_repair(self, spec: _fs.Spec, entry: dict[str, Any]) -> None:
        """A repair of the first version was begun and the run ended before it was judged: the module as it was before the
        repair is put back (the repair was never shown to pass its tests)."""
        if not entry.get("repair_pending"):
            return
        orig = self._orig_path(spec.name)
        try:
            if orig.is_file():
                self.model_path.write_text(orig.read_text(encoding="utf-8"), encoding="utf-8")
                self.log.info("[second_impl] an unfinished repair of %s was undone", spec.name)
                self.record["code_sha"] = _sha(self.model_path)
                orig.unlink()
        except OSError:
            pass
        entry.pop("repair_pending", None)
        self._save()

    # -- the reference version --

    def _ref_path(self, name: str) -> Path:
        return self.root / DIR / f"{name}.py"

    def _spec_passages(self, spec: _fs.Spec) -> str:
        return self.passages.get(spec.equation, "")

    async def _write_reference(self, spec: _fs.Spec, entry: dict[str, Any], *, rewrite: str = "", kind: str = "write") -> str:
        """Ask for the second version and save it; ``""`` on success, else why not (nothing is saved)."""
        main_source = _function_source(self.model_path, spec.name)
        try:
            prompt = write_prompt(spec, template=self.prompts["write"], protocol=self.protocol, specs=self.specs,
                                  passages=self._spec_passages(spec), example=example_inputs(spec, self.protocol, self.captured),
                                  topic_block=self.topic_block, rewrite=rewrite, main_source=main_source,
                                  return_keys=self._return_keys(spec))
        except ValueError as e:
            return str(e)
        if not self._spend(entry, kind):
            return "the limit on requests for second versions was reached"
        try:
            reply = await self.chat(NODE_WRITE, [{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001 -- a request that failed is spent; the function is left unconfirmed
            self._done_call(entry)
            return f"the request failed ({type(e).__name__})"
        self._done_call(entry)
        code = _fs.reply_code(reply)
        bad = reference_problem(code, spec, self.package)
        if bad:
            return bad
        path = self._ref_path(spec.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code.rstrip() + "\n", encoding="utf-8")
        return ""

    # -- comparing --

    def _inputs_problem(self, spec: _fs.Spec) -> str:
        """Why the function cannot be called with inputs FI knows (``""`` when it can): checked before a second version is
        asked for, so no request is spent on a function that can only end unconfirmed."""
        base = example_inputs(spec, self.protocol, self.captured)
        if not base:
            return "the plan gives no example inputs to call it with, and FI's run of the simulation did not call it"
        if not all(p in base for p in _required(spec.signature)):
            return "the example inputs do not cover the function's parameters"
        return ""

    async def _compare(self, spec: _fs.Spec, entry: dict[str, Any]) -> dict[str, Any]:
        """One subprocess runs both versions on every input. ``status`` unconfirmed with ``why`` when it cannot be judged."""
        why = self._inputs_problem(spec)
        if why:
            return {"status": UNCONFIRMED, "why": why, "disagreements": [], "retry": False}
        base = example_inputs(spec, self.protocol, self.captured)
        inputs = build_inputs(base, ranges_of(spec, self.protocol), defined=domain_of(spec, self.protocol, base))
        # Every other call FI recorded is an input too: the settings the study really gives the function.
        for call in self.captured.get(spec.name, [])[1:]:
            if len(inputs) < MAX_INPUTS + CAPTURE_CALLS and call not in inputs:
                inputs.append(dict(call))
        (self.root / DIR).mkdir(parents=True, exist_ok=True)
        inputs_path = DIR / f"{spec.name}.inputs.json"
        (self.root / inputs_path).write_text(json.dumps(inputs), encoding="utf-8")
        uses = [d for d in spec.depends_on if d != spec.name]
        # A dependency that has a second version of its own is called through it; the others are the first version's, which
        # makes the evidence weaker (a fault in such a dependency is shared) and is said in the record.
        own = {d: (DIR / f"{d}.py").as_posix() for d in uses if (self.root / DIR / f"{d}.py").is_file()}
        shared = [d for d in uses if d not in own]
        if shared:
            entry["deps_from_first_version"] = shared
        else:
            entry.pop("deps_from_first_version", None)
        script = compare_script(code_dir="code", module=f"{self.package}.model", name=spec.name,
                                reference=(DIR / f"{spec.name}.py").as_posix(), inputs=inputs_path.as_posix(), uses=uses,
                                own=own)
        out = None
        err = ""
        for _try in range(2):  # a harness that printed nothing is run once more: it says nothing about the function
            try:
                rc, text, err = await self.run(["-c", script])
            except Exception as e:  # noqa: BLE001
                rc, text, err = -1, "", repr(e)
            out = parse_output(text)
            if out is not None:
                break
        entry["inputs_tried"] = len(inputs)
        if out is None:
            return {"status": UNCONFIRMED, "why": f"the comparison could not be run here ({_text(err, 120) or 'no output'})",
                    "disagreements": [], "retry": True}
        if "main_import_error" in out:
            return {"status": UNCONFIRMED, "why": "the first version could not be imported", "disagreements": [], "retry": True}
        if "ref_import_error" in out:
            return {"status": UNCONFIRMED, "why": "the second version could not be loaded", "disagreements": [], "retry": False}
        rows = out.get("rows") if isinstance(out.get("rows"), list) else []
        if len(rows) != len(inputs):
            return {"status": UNCONFIRMED, "why": "the comparison returned fewer results than inputs", "disagreements": [], "retry": True}
        verdict = judge_rows(rows, inputs, tolerance_for(spec, self.protocol))
        if verdict["status"] == UNCONFIRMED:
            verdict.update(why=verdict.get("why") or "both versions raised on every input", retry=False)
        return verdict

    def _return_keys(self, spec: _fs.Spec) -> list[str] | None:
        """Every key of the first version's dict results in the calls FI recorded when it ran the simulation; ``None`` when
        FI has no record."""
        every = (read_captured(self.root).get("returned") or {}).get(spec.name)
        return [str(k) for k in every] if isinstance(every, list) and every else None

    # -- one function --

    async def _one(self, spec: _fs.Spec, entry: dict[str, Any]) -> None:
        entry["same_family"] = self._same_family()
        self._restore_unverified_repair(spec, entry)
        if entry.get("pending"):  # a request was sent and its answer was not saved: it is not sent again
            kind = entry["pending"].get("kind")
            self._finish(spec, entry, UNCONFIRMED, f"a request ({kind}) was sent but its answer was not saved; it is not asked again")
            return
        if not self._ref_path(spec.name).is_file():
            no_inputs = self._inputs_problem(spec)
            if no_inputs:
                self._finish(spec, entry, UNCONFIRMED, no_inputs)
                return
            why = await self._write_reference(spec, entry)
            if why:
                self._finish(spec, entry, UNCONFIRMED, f"the second version could not be written: {why}")
                return
        while True:
            adj = entry.get("adjudication")
            if isinstance(adj, dict) and not adj.get("acted"):
                outcome = await self._act(spec, entry, adj)
                adj["acted"] = True
                self._save()
                if outcome:
                    self._finish(spec, entry, outcome[0], outcome[1], entry.get("disagreements"))
                    return
                continue
            res = await self._compare(spec, entry)
            entry["disagreements"] = res["disagreements"][:3]
            entry["compared"] = res.get("compared", 0)
            entry["excluded"] = res.get("excluded", 0)
            if res.get("shape_notes"):
                entry["shape_note"] = "; ".join(res["shape_notes"])
            else:
                entry.pop("shape_note", None)
            if res["status"] == AGREE:
                self._finish(spec, entry, AGREE, f"{res.get('compared', 0)} input(s) compared"
                             + (f"; {entry['shape_note']}" if entry.get("shape_note") else ""))
                return
            if res["status"] == UNCONFIRMED:
                self._finish(spec, entry, UNCONFIRMED, res.get("why", ""), recheck=bool(res.get("retry")))
                return
            if int(entry.get("rounds") or 0) >= MAX_ROUNDS:
                self._finish(spec, entry, UNRESOLVED, "still different after the rounds allowed", entry["disagreements"])
                return
            entry["rounds"] = int(entry.get("rounds") or 0) + 1
            verdict = await self._adjudicate(spec, entry)
            if verdict is None:
                self._finish(spec, entry, UNCONFIRMED, "the judge could not be asked")
                return
            self._save()

    async def _adjudicate(self, spec: _fs.Spec, entry: dict[str, Any]) -> tuple[str, str] | None:
        if not self._spend(entry, "adjudicate"):
            return None
        prompt = self.prompts["adjudicate"].substitute(
            step_block=_spec_block(spec, self.protocol), main_code=_function_source(self.model_path, spec.name) or spec.signature,
            reference_code=self._ref_path(spec.name).read_text(encoding="utf-8").strip(), cases_block=cases_text(entry["disagreements"]))
        try:
            reply = await self.chat(NODE_ADJUDICATE, [{"role": "user", "content": prompt}])
        except Exception:  # noqa: BLE001 -- spent; read as ambiguous
            reply = ""
        verdict, reason = parse_verdict(reply)
        entry["adjudication"] = {"round": entry["rounds"], "verdict": verdict, "reason": reason, "acted": False}
        self._done_call(entry)
        return verdict, reason

    async def _act(self, spec: _fs.Spec, entry: dict[str, Any], adj: dict[str, Any]) -> tuple[str, str] | None:
        """Do what the judge's verdict says. ``None`` = compare again; else ``(final status, why)``."""
        verdict, reason = adj.get("verdict"), str(adj.get("reason") or "")
        if verdict == MAIN_WRONG:
            before = self.model_path.read_text(encoding="utf-8") if self.model_path.is_file() else None
            if before is not None:  # saved before anything is written, so a run that ends in the middle can be undone
                self._orig_path(spec.name).parent.mkdir(parents=True, exist_ok=True)
                self._orig_path(spec.name).write_text(before, encoding="utf-8")
            entry["repair_pending"] = True
            if not self._spend(entry, "repair_main"):
                entry.pop("repair_pending", None)
                return UNRESOLVED, "no requests left to repair the first version"
            ok = False
            try:
                ok = bool(await self.repair_main(spec.name, repair_problem(reason, entry.get("disagreements") or []),
                                                 list(entry.get("disagreements") or [])))
            except Exception as e:  # noqa: BLE001
                self.log.info("[second_impl] repair of %s failed: %r", spec.name, e)
            self._done_call(entry)
            accepted, why_not = False, "the repair did not pass the function's own tests"
            if ok:
                try:
                    accepted = bool(await self.still_passes(spec.name))
                except Exception:  # noqa: BLE001
                    accepted = False
            if accepted and self.verify_package is not None:
                try:
                    broke = await self.verify_package()
                except Exception:  # noqa: BLE001 -- a check that cannot be made does not accept the repair
                    broke = "the model's other tests could not be run"
                if broke:
                    accepted, why_not = False, f"the repair broke another check of the model ({broke})"
            if not accepted:
                if before is not None and self.model_path.is_file() and self.model_path.read_text(encoding="utf-8") != before:
                    self.model_path.write_text(before, encoding="utf-8")  # the repair is not kept
                entry.pop("repair_pending", None)
                self._orig_path(spec.name).unlink(missing_ok=True)
                adj["outcome"] = "repair not accepted"
                return UNRESOLVED, f"the judge found the first version wrong, but {why_not}"
            entry.pop("repair_pending", None)
            self._orig_path(spec.name).unlink(missing_ok=True)
            adj["outcome"] = "repair accepted"
            self.record["code_sha"] = _sha(self.model_path)
            return None
        if verdict == REFERENCE_WRONG:
            if int(entry.get("rewrites") or 0) >= 1:
                return UNRESOLVED, "the second version was rewritten once and still differs"
            entry["rewrites"] = 1
            old = self._ref_path(spec.name).read_text(encoding="utf-8").strip() if self._ref_path(spec.name).is_file() else ""
            note = (f"Your earlier version:\n\n```python\n{old}\n```\n\nA reader of the description judged it wrong: {reason}\n"
                    "Write the function again from the description.")
            why = await self._write_reference(spec, entry, rewrite=note, kind="rewrite")
            if why:
                return UNCONFIRMED, f"the second version could not be rewritten: {why}"
            adj["outcome"] = "reference rewritten"
            return None
        return UNRESOLVED, f"the description allows both readings ({reason})" if reason else "the description allows both readings"

    # -- all --

    async def run_all(self) -> dict[str, Any]:
        """Check the key functions and return (and save) the record."""
        self.record = load(self.root)
        sha = _sha(self.model_path)
        stale = bool(self.record.get("code_sha")) and self.record.get("code_sha") != sha
        self.record["code_sha"] = sha
        self.record.setdefault("calls", 0)
        selected, skipped = key_functions(self.specs, self.protocol, self.cap)
        funcs = self._funcs()
        for name, why in skipped.items():
            funcs.setdefault(name, {"status": SKIPPED, "rounds": 0, "inputs_tried": 0, "disagreements": [], "same_family": None, "why": why})
        self._save()
        for spec in selected:
            entry = funcs.get(spec.name)
            if entry is None:
                entry = funcs[spec.name] = {"status": WORKING, "rounds": 0, "inputs_tried": 0, "disagreements": [], "same_family": None}
            elif entry.get("status") in FINAL:
                again = entry.get("recheck") or (stale and entry.get("status") in (AGREE, UNRESOLVED))
                if not again or not self._ref_path(spec.name).is_file():
                    continue  # finished: a resume reuses it and asks nothing
                entry["status"] = WORKING
                entry["rounds"] = 0
                entry["rewrites"] = 0
                entry.pop("adjudication", None)  # the first version changed: an earlier verdict is about other code
            try:
                await self._one(spec, entry)
            except Exception as e:  # noqa: BLE001 -- one function's trouble never stops the others
                self.log.info("[second_impl] %s: %r", label(spec), e)
                self._finish(spec, entry, UNCONFIRMED, f"the check failed to run ({type(e).__name__})")
        self._save()
        return self.record


def _required(signature: str) -> list[str]:
    """Names of the parameters with no default."""
    try:
        node = ast.parse(signature.rstrip().rstrip(":") + ":\n    pass").body[0]
    except SyntaxError:
        return []
    if not isinstance(node, ast.FunctionDef):
        return []
    a = node.args
    pos = [*a.posonlyargs, *a.args]
    need = [x.arg for x in pos[: len(pos) - len(a.defaults)]]
    need += [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
    return need


async def second_versions(**kwargs: Any) -> dict[str, Any]:
    """``SecondImpl(**kwargs).run_all()``: the record. See :class:`SecondImpl` for the arguments."""
    return await SecondImpl(**kwargs).run_all()

