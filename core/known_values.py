"""A commissioned study's known values, each computed from its source before the plan, with the answer kept from the model.

Per value the model writes one program from the source's text (the value withheld); FI runs it apart from the quest and
judges what it printed (``.fi/known_values.json``). Only when a second, independent program agrees with a first that
missed the value is the study handed back (``known_value_mismatch``); what FI cannot judge goes on, said to the plan.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import commission as _commission
from .oracle_triage import num as _number

LINE = "KNOWN_JSON:"
RECORD = "known_values.json"  # in .fi/
WITHHELD = "[value withheld]"
EXCERPT_CHARS = 12000  # how much of the source text the program's writer is shown
_HEAD_CHARS = 2000  # ... of which the opening of the text (its definitions), when the passage is further in
# A display equation whose formula did not survive the conversion to text: a numbered row with nothing in it.
_LOST_EQUATION = re.compile(r"^\s*\|\s*\(\d+\)\s*\|\s*$", re.M)
_SECOND = ("Another program computed this value before yours, and FI is checking it: write yours on your own, from "
           "the definitions in the text, without guessing what the other found.")

Ask = Callable[[str, bool], Awaitable[str]]  # (prompt, second program) -> the reply
Execute = Callable[[Path], Awaitable[tuple[int, str, str]]]  # the program -> (returncode, stdout, stderr)


def _decimal(value: float) -> Decimal:  # a commission's value is a finite number (core/commission.py)
    return Decimal(repr(float(value))).normalize()


def decisive(value: float) -> bool:
    """Whether a number typed in a program can be told from a computed one: not 0 or 1, and two digits or more."""
    return abs(float(value)) not in (0.0, 1.0) and len(_decimal(value).as_tuple().digits) >= 2


def _half_unit(value: float) -> float:
    """Half a unit of the value's last written digit (1.2571: 0.00005)."""
    return 0.5 * 10.0 ** int(_decimal(value).as_tuple().exponent)


def mask(text: Any, value: float) -> str:
    """``text`` with every number that writes ``value`` (to two digits or more, either sign) replaced by
    :data:`WITHHELD`; unchanged for a value that is not :func:`decisive` (masking 0.5 would blank every 0.5)."""
    text = str(text or "")
    if not decisive(value):
        return text

    def one(m: re.Match[str]) -> str:
        digits, exp = m.group(2).replace(",", ""), int((m.group(3) or "0").replace("−", "-"))
        written = digits.replace(".", "").lstrip("0")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        near = abs(float(digits) * 10.0 ** exp - abs(float(value))) <= 0.5 * 10.0 ** (exp - decimals) * (1 + 1e-9)
        return WITHHELD if len(written) >= 2 and near else m.group(0)

    return _commission._NUMBER.sub(one, text)


def excerpt(text: str, passage: str) -> str:
    """The part of the source text the program's writer is shown: all of it when short, else its opening and the
    stretch around the passage, at most :data:`EXCERPT_CHARS` characters."""
    if len(text) <= EXCERPT_CHARS:
        return text
    words = str(passage or "").split()[:12]
    found = re.search(r"\s+".join(map(re.escape, words)), text, re.IGNORECASE) if words else None
    if found is None or found.start() < EXCERPT_CHARS - _HEAD_CHARS:
        return text[:EXCERPT_CHARS]
    width = EXCERPT_CHARS - _HEAD_CHARS
    start = max(_HEAD_CHARS, found.start() - width // 2)
    return text[:_HEAD_CHARS] + "\n[...]\n" + text[start:start + width]


def prompt(template: Any, kv: dict[str, Any], shown: str, *, again: str = "") -> str:
    """The request for the program (``agents/known_value_program.md``), with the value withheld everywhere in it."""
    value = float(kv["value"])
    return template.substitute(
        name=mask(kv.get("name"), value), case=mask(kv.get("case"), value), passage=mask(kv.get("passage"), value),
        title=mask((kv.get("source") or {}).get("title"), value), excerpt=mask(shown, value), line=LINE,
        again=again or "")


def program(reply: Any, value: float, tolerance: float, given: str) -> tuple[str, str]:
    """``(the program, "")`` from the reply, or ``("", why)``: no fenced block, a forbidden call (reading or running
    anything, core/oracle_challenge.py), under 5 or over 400 lines, or a number within the tolerance of a
    :func:`decisive` value typed in that the request did not give (``given``: the masked text it was shown)."""
    from .oracle_challenge import numeric_script

    script, why = numeric_script(reply)
    if not script:
        return "", why
    lines = len(script.splitlines())
    if not 5 <= lines <= 400:
        return "", f"the program has {lines} lines; write one of 5 to 400"
    if decisive(value):
        try:
            tree = ast.parse(script)
        except SyntaxError as e:
            return "", f"the program is not valid Python ({e.msg}, line {e.lineno})"
        typed = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                 and not isinstance(n.value, bool) and abs(abs(float(n.value)) - abs(value)) <= tolerance
                 and not _commission.number_in_text(n.value, given)]
        if typed:
            return "", "the program types in a number close to the value withheld; it must compute the value"
    return script, ""


def printed(stdout: str) -> tuple[list[dict[str, Any]], str]:
    """``(the routes, what it says it cannot do)`` from the program's ``KNOWN_JSON`` lines."""
    routes: list[dict[str, Any]] = []
    cannot = ""
    for line in str(stdout or "").splitlines():
        if not line.startswith(LINE):
            continue
        try:
            data = json.loads(line[len(LINE):])
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if str(data.get("cannot") or "").strip():
            cannot = " ".join(str(data["cannot"]).split())[:400]
        elif (v := _number(data.get("value"))) is not None:
            routes.append({"route": str(data.get("route") or f"route {len(routes) + 1}").strip().lower()[:40], "value": v,
                           "error": abs(_number(data.get("error")) or 0.0), "how": " ".join(str(data.get("how") or "").split())[:300]})
    return routes, cannot


def _g(x: float) -> str:
    return f"{x:.6g}"


def judge(routes: list[dict[str, Any]], value: float, tolerance: float) -> tuple[str, str]:
    """``(verdict, why)``: ``match``, ``no_match`` or ``cannot_judge``. Two routes of one program disagree when they are
    further apart than three times their combined error (plus half a unit of the value's last digit); a route reproduces
    the value within the larger of the tolerance and three times its own error. One route alone is a match only for a
    :func:`decisive` value."""
    for a, b in combinations(routes, 2):
        if abs(a["value"] - b["value"]) > 3 * math.hypot(a["error"], b["error"]) + _half_unit(value):
            return "no_match", (f"its {a['route']} route gives {_g(a['value'])} (error {_g(a['error'])}) and its "
                                f"{b['route']} route {_g(b['value'])} (error {_g(b['error'])}), further apart than their errors")
    off = [r for r in routes if abs(r["value"] - value) > max(tolerance, 3 * r["error"])]
    if off:
        r = off[0]
        return "no_match", (f"its {r['route']} route gives {_g(r['value'])} (error {_g(r['error'])}), not "
                            f"{_g(value)} within {_g(tolerance)}")
    said = ", ".join(f"{r['route']} {_g(r['value'])}" for r in routes)
    if len(routes) < 2 and not decisive(value):
        return "cannot_judge", f"one route only ({said}), and the value is too round to tell a computed number from a typed one"
    return "match", f"reproduced: {said}"


def agree(first: list[dict[str, Any]], second: list[dict[str, Any]], tolerance: float) -> bool:
    """Whether two programs give the same numbers: each route both give, within the larger of the tolerance and three
    times their combined error."""
    by = {r["route"]: r for r in second}
    shared = [(r, by[r["route"]]) for r in first if r["route"] in by]
    return bool(shared) and all(abs(a["value"] - b["value"]) <= max(tolerance, 3 * math.hypot(a["error"], b["error"]))
                                for a, b in shared)


def _source_text(quest_root: Path, name: str) -> str:
    from . import pdf_text

    path = Path(quest_root) / "inputs" / "papers" / name
    try:
        return ((pdf_text.extract(path.read_bytes()).text or "") if path.suffix.lower() == ".pdf"
                else path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return ""


async def _program(kv: dict[str, Any], shown: str, path: Path, *, template: Any, ask: Ask, execute: Execute,
                   second: bool) -> dict[str, Any]:
    """One program for one value: asked (once more, with the reason, when it cannot be used), run, and judged."""
    value, tolerance = float(kv["value"]), float(kv["tolerance"])
    given = " ".join(mask(kv.get(k), value) for k in ("case", "passage")) + " " + mask(shown, value)
    note = _SECOND if second else ""
    script, why = "", ""
    for _ in range(2):
        try:
            reply = await ask(prompt(template, kv, shown, again=note), second)
        except Exception as e:  # noqa: BLE001 -- a program that could not be had leaves the value unconfirmed
            return {"verdict": "cannot_judge", "why": f"the program could not be asked for ({str(e)[:160] or type(e).__name__})"}
        script, why = program(reply, value, tolerance, given)
        if script:
            break
        note = (f"{_SECOND} " if second else "") + f"Your last reply could not be used: {why}. Write the program again."
    if not script:
        return {"verdict": "cannot_judge", "why": f"no usable program after two requests ({why})"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    code, out, err = await execute(path)
    routes, cannot = printed(out)
    ran = {"program": path.name, "sha256": _commission.sha256(path), "returncode": code, "stdout_tail": out[-1500:],
           "routes": routes}
    if not routes:
        failed = (err or "").strip().splitlines()[-1][:200] if (err or "").strip() else f"exit code {code}"
        why = (f"the program says the text does not give what it needs: {cannot}" if cannot else
               f"the program printed no `{LINE}` value ({failed})")
        return {**ran, "verdict": "cannot_judge", "why": why}
    verdict, why = judge(routes, value, tolerance)
    return {**ran, "verdict": verdict, "why": why}


async def _one(kv: dict[str, Any], text: str, folder: Path, **how: Any) -> dict[str, Any]:
    out = {"name": kv["name"], "value": float(kv["value"]), "tolerance": float(kv["tolerance"]), "case": kv["case"],
           "passage": kv["passage"], "source": kv["source"], "programs": []}
    shown = excerpt(text, str(kv["passage"]))
    if len(_LOST_EQUATION.findall(shown)) >= 2:
        return {**out, "verdict": "cannot_judge", "why": "the full text lost its equations (numbered rows with no formula)"}
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(kv["name"]))[:60] or "value"
    first = await _program(kv, shown, folder / f"{slug}_1.py", second=False, **how)
    out["programs"].append(first)
    if first["verdict"] != "no_match":
        return {**out, "verdict": first["verdict"], "why": first["why"]}
    second = await _program(kv, shown, folder / f"{slug}_2.py", second=True, **how)
    out["programs"].append(second)
    if second["verdict"] == "no_match" and agree(first["routes"], second["routes"], out["tolerance"]):
        return {**out, "verdict": "no_match", "why": f"{first['why']}; a second, independent program gives the same numbers"}
    said = ("reproduced it" if second["verdict"] == "match" else
            "gave other numbers" if second.get("routes") else f"could not compute it ({second['why']})")
    return {**out, "verdict": "cannot_judge",
            "why": f"a first program did not reproduce it ({first['why']}), and a second, independent one {said}"}


async def check(data: dict[str, Any], quest_root: Path, *, template: Any, ask: Ask, execute: Execute) -> dict[str, Any]:
    """Every known value of the commission ``data`` computed and judged, recorded in ``.fi/known_values.json``; a record
    already there for the same commission is read back instead (a resume asks nothing again)."""
    path = Path(quest_root) / ".fi" / RECORD
    sha = _commission.recorded_sha(Path(quest_root))
    try:
        kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        kept = None
    if isinstance(kept, dict) and sha and kept.get("commission_sha256") == sha:
        return kept
    names = _commission.staged_names(data)
    texts = {raw: _source_text(Path(quest_root), name) for raw, name in names.items()}
    folder = Path(quest_root) / ".fi" / "known_values"
    values = [await _one(kv, texts.get(str(kv["source"]["file"]), ""), folder, template=template, ask=ask,
                         execute=execute) for kv in data.get("known_values") or []]
    record = {"commission_sha256": sha, "at": time.time(), "values": values}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return record


def mismatches(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [v for v in (record or {}).get("values") or [] if v.get("verdict") == "no_match"]


def problem(wrong: list[dict[str, Any]]) -> str:
    """The return's sentence: which known value two programs found to be other than the commission says."""
    return "; ".join(f"the known value {v['name']!r} ({_g(v['value'])} within {_g(v['tolerance'])}) was computed from "
                     f"its source by two independent programs, and {v['why']}" for v in wrong)


def evidence(wrong: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What QC needs to decide: per value the commission's number, tolerance, passage and source, and each program's
    file, hash and routes (value and error)."""
    return [{k: v.get(k) for k in ("name", "value", "tolerance", "case", "passage", "source", "why")}
            | {"programs": [{k: p.get(k) for k in ("program", "sha256", "routes")} for p in v.get("programs") or []]}
            for v in wrong]


def plan_block(record: dict[str, Any] | None) -> str:
    """What FI's own computation of each known value found, for the plan's model."""
    values = (record or {}).get("values") or []
    if not values:
        return ""
    lines = ["\n\nWHAT FI COMPUTED OF THE KNOWN VALUES (before this plan, each from its source, by a program FI ran)"]
    for v in values:
        lines.append(f"- {v['name']!r} ({_g(v['value'])}): " + (
            f"confirmed: {v['why']}" if v["verdict"] == "match" else f"not confirmed by FI's own computation: {v['why']}"))
    return "\n".join(lines) + "\n"


def _fold(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip().lower()


def labels(literature: list[Any], data: dict[str, Any]) -> dict[str, str]:
    """Each supplied full text (its commission file) -> its label in the literature."""
    from .engine_base import _labelled_sources

    by_name = {name: raw for raw, name in _commission.staged_names(data).items()}
    return {by_name[Path(str(meta.get("path") or "")).name]: label for label, meta, _item in _labelled_sources(literature)
            if meta.get("source") == "user_supplied" and Path(str(meta.get("path") or "")).name in by_name}


def fill_checks(design: dict[str, Any], data: dict[str, Any], source_labels: dict[str, str]) -> dict[str, Any]:
    """The plan's checks named for a known value, with FI writing their kind, expected value, tolerance and source
    from the commission: the plan's model gives only the case and the measure."""
    from .oracle_check import _KIND_KEYS, _REFERENCE_KEYS

    protocol = design.get("protocol") if isinstance(design, dict) else None
    if not isinstance(protocol, dict) or not isinstance(protocol.get("oracles"), list):
        return design
    by = {_fold(kv.get("name")): kv for kv in data.get("known_values") or []}
    oracles = []
    for o in protocol["oracles"]:
        kv = by.get(_fold(o.get("name"))) if isinstance(o, dict) else None
        if kv is not None:
            label = source_labels.get(str(kv["source"]["file"]))
            where = f"[{label}]" if label else f"the supplied full text {Path(str(kv['source']['file'])).name}"
            o = {**{k: v for k, v in o.items() if k not in (*_KIND_KEYS, *_REFERENCE_KEYS, "expected_formula")},
                 "kind": "published_value", "expected": float(kv["value"]), "tolerance": float(kv["tolerance"]),
                 "tolerance_mode": "absolute", "reference": f"{where}, where it states: \"{kv['passage']}\""}
        oracles.append(o)
    return {**design, "protocol": {**protocol, "oracles": oracles}}
