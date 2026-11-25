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
ROUTES = ("formula", "definition")  # the two ways a program computes a value (agents/known_value_program.md)
RECORD = "known_values.json"  # in .fi/
WITHHELD = "[value withheld]"
EXCERPT_CHARS = 12000  # how much of the source text the program's writer is shown
_HEAD_CHARS = 2000  # ... of which the opening of the text, when the passage is further in
_BEFORE_CHARS = 2000  # ... and of the text before the passage: the rest is what follows it (its definitions)
# A display equation whose formula did not survive the conversion to text: a numbered row with nothing in it.
_LOST_EQUATION = re.compile(r"^\s*\|\s*\(\d+\)\s*\|\s*$", re.M)
_SECOND = ("Another program computed this value before yours, and FI is checking it: write yours on your own, from "
           "the definitions in the text, without guessing what the other found.")

Ask = Callable[[str, bool], Awaitable[str]]  # (prompt, second program) -> the reply
Execute = Callable[[Path], Awaitable[tuple[int, str, str]]]  # the program -> (returncode, stdout, stderr)


def _decimal(value: float | Decimal) -> Decimal:  # a commission's value is a finite number (core/commission.py)
    return value if isinstance(value, Decimal) else Decimal(repr(float(value))).normalize()


def written(kv: dict[str, Any]) -> Decimal:
    """A known value as its passage writes it: 0.400 keeps the three digits that the number read from the file (0.4)
    lost; the number's own digits when the passage writes it with no more."""
    value = float(kv["value"])
    best = _decimal(value)
    for m in _commission._NUMBER.finditer(str(kv.get("passage") or "")):
        found = _commission.number_value(m)
        if found is None or not math.isclose(abs(found), abs(value), rel_tol=1e-9, abs_tol=0.0):
            continue
        form = Decimal(m.group(2).replace(",", "")).scaleb(int((m.group(3) or "0").replace("−", "-")))
        form = form.copy_sign(best)  # the commission's sign (-0.400 is -0.4), whatever the passage's own sign
        if len(form.as_tuple().digits) > len(best.as_tuple().digits):
            best = form
    return best


def decisive(value: float | Decimal) -> bool:
    """Whether a number typed in a program can be told from a computed one: not 0 or 1, and two digits or more."""
    return abs(float(value)) not in (0.0, 1.0) and len(_decimal(value).as_tuple().digits) >= 2


def _half_unit(value: float | Decimal) -> float:
    """Half a unit of the value's last written digit (2.3846: 0.00005)."""
    return 0.5 * 10.0 ** int(_decimal(value).as_tuple().exponent)


def withhold(text: Any, value: float | Decimal) -> str:
    """``text`` with every number that writes ``value`` (either sign: the value itself in any form, 0.4 for 0.400, or
    the value rounded to two digits or more) replaced by :data:`WITHHELD`; unchanged for a value that is not
    :func:`decisive` (withholding 0.5 would blank every 0.5). ``value`` may be :func:`written`, whose digits count."""
    text = str(text or "")
    if not decisive(value):
        return text

    def one(m: re.Match[str]) -> str:
        found = _commission.number_value(m)
        if found is None:  # an exponent no float holds: not the value
            return m.group(0)
        digits, exp = m.group(2).replace(",", ""), int((m.group(3) or "0").replace("−", "-"))
        shown = digits.replace(".", "").lstrip("0")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        same = math.isclose(abs(found), abs(float(value)), rel_tol=1e-9, abs_tol=0.0)
        near = abs(abs(found) - abs(float(value))) <= 0.5 * 10.0 ** (exp - decimals) * (1 + 1e-9)
        return WITHHELD if same or (len(shown) >= 2 and near) else m.group(0)

    return _commission._NUMBER.sub(one, text)


def excerpt(text: str, passage: str) -> str:
    """The part of the source text the program's writer is shown, at most :data:`EXCERPT_CHARS` characters: all of it
    when short, else a little before the passage and what follows it (where a table's or a result's definitions are),
    with the text's opening first when the passage is further in."""
    if len(text) <= EXCERPT_CHARS:
        return text
    words = str(passage or "").split()[:12]
    found = re.search(r"\s+".join(map(re.escape, words)), text, re.IGNORECASE) if words else None
    start = max(0, found.start() - _BEFORE_CHARS) if found is not None else 0
    head = text[:_HEAD_CHARS] + "\n[...]\n" if start > _HEAD_CHARS else ""
    return head + text[start if head else 0:][:EXCERPT_CHARS - len(head)]


def prompt(template: Any, kv: dict[str, Any], shown: str, *, again: str = "") -> str:
    """The request for the program (``agents/known_value_program.md``), with the value withheld everywhere in it."""
    value = written(kv)
    return template.substitute(
        name=withhold(kv.get("name"), value), case=withhold(kv.get("case"), value), passage=withhold(kv.get("passage"), value),
        title=withhold((kv.get("source") or {}).get("title"), value), excerpt=withhold(shown, value), line=LINE,
        again=again or "")


def _constants(tree: ast.AST) -> list[float]:
    """Every number a program types: its number constants and each number written in a text constant
    (``float("2.3846")``)."""
    out: list[float] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            out.append(float(n.value))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            out += [x for m in _commission._NUMBER.finditer(n.value) if (x := _commission.number_value(m)) is not None]
    return out


def program(reply: Any, value: float | Decimal, tolerance: float, given: str) -> tuple[str, str]:
    """``(the program, "")`` from the reply, or ``("", why)``: no fenced block, a forbidden call (reading or running
    anything, core/oracle_challenge.py), under 5 or over 400 lines, or a number within the tolerance of a
    :func:`decisive` value typed in that the request did not give (``given``: the withheld_text text it was shown)."""
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
        typed = [x for x in _constants(tree) if abs(abs(x) - abs(float(value))) <= tolerance
                 and not _commission.number_in_text(x, given)]
        if typed:
            return "", "the program types in a number close to the value withheld; it must compute the value"
    return script, ""


def printed(stdout: str) -> tuple[list[dict[str, Any]], str]:
    """``(the routes, what it says it cannot do)`` from the program's ``KNOWN_JSON`` lines: one route per name of
    :data:`ROUTES` (its first line), so a line printed twice or under another name is never a second route."""
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
        elif ((v := _number(data.get("value"))) is not None and (name := str(data.get("route") or "").strip().lower()) in ROUTES
              and name not in {r["route"] for r in routes}):
            routes.append({"route": name, "value": v, "error": abs(_number(data.get("error")) or 0.0),
                           "how": " ".join(str(data.get("how") or "").split())[:300]})
    return routes, cannot


def _g(x: float) -> str:
    return f"{x:.6g}"


def judge(routes: list[dict[str, Any]], value: float | Decimal, tolerance: float) -> tuple[str, str]:
    """``(verdict, why)``: ``match``, ``no_match`` or ``cannot_judge``. Two routes of one program disagree when they are
    further apart than three times their combined error (plus half a unit of the value's last digit); a route reproduces
    the value within the larger of the tolerance and three times its own error. One route alone is a match only for a
    :func:`decisive` value."""
    for a, b in combinations(routes, 2):
        if abs(a["value"] - b["value"]) > 3 * math.hypot(a["error"], b["error"]) + _half_unit(value):
            return "no_match", (f"its {a['route']} route gives {_g(a['value'])} (error {_g(a['error'])}) and its "
                                f"{b['route']} route {_g(b['value'])} (error {_g(b['error'])}), further apart than their errors")
    off = [r for r in routes if abs(r["value"] - float(value)) > max(tolerance, 3 * r["error"])]
    if off:
        r = off[0]
        return "no_match", (f"its {r['route']} route gives {_g(r['value'])} (error {_g(r['error'])}), not "
                            f"{_g(float(value))} within {_g(tolerance)}")
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
    value, tolerance = written(kv), float(kv["tolerance"])
    given = " ".join(withhold(kv.get(k), value) for k in ("case", "passage")) + " " + withhold(shown, value)
    note = _SECOND if second else ""
    script, why = "", ""
    for _ in range(2):
        try:
            reply = await ask(prompt(template, kv, shown, again=note), second)
        except Exception as e:  # noqa: BLE001 -- a program that could not be had leaves the value unconfirmed
            return {"verdict": "cannot_judge", "unasked": True,
                    "why": f"the program could not be asked for ({str(e)[:160] or type(e).__name__})"}
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
    if code != 0 or not routes:  # a program that stopped with an error: what it printed before is not judged
        failed = (err or "").strip().splitlines()[-1][:200] if (err or "").strip() else f"exit code {code}"
        why = (f"the program stopped with an error ({failed}); what it printed before is not judged" if code != 0 else
               f"the program says the text does not give what it needs: {cannot}" if cannot else
               f"the program printed no `{LINE}` value ({failed})")
        return {**ran, "routes": [], "printed_before_it_stopped": routes if code != 0 else [], "verdict": "cannot_judge",
                "why": why, "crashed": code != 0 or not cannot}
    verdict, why = judge(routes, value, tolerance)
    return {**ran, "verdict": verdict, "why": why}


async def _one(kv: dict[str, Any], text: str, folder: Path, **how: Any) -> dict[str, Any]:
    out = {"name": kv["name"], "value": float(kv["value"]), "tolerance": float(kv["tolerance"]), "case": kv["case"],
           "passage": kv["passage"], "source": kv["source"], "programs": []}
    if not decisive(written(kv)):  # 0, 1, 0.5: no text can be shown without it, so no program would be blind to it
        return {**out, "verdict": "cannot_judge",
                "why": "too common a number to hide from the program's writer (one digit, or 0 or 1), so it is not computed"}
    shown = excerpt(text, str(kv["passage"]))
    if len(_LOST_EQUATION.findall(shown)) >= 2:
        return {**out, "verdict": "cannot_judge", "why": "the full text lost its equations (numbered rows with no formula)"}
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", str(kv["name"]))[:60] or "value"
    first = await _program(kv, shown, folder / f"{slug}_1.py", second=False, **how)
    out["programs"].append(first)
    if first["verdict"] != "no_match" and not first.get("crashed"):
        return {**out, "verdict": first["verdict"], "why": first["why"]}
    second = await _program(kv, shown, folder / f"{slug}_2.py", second=True, **how)  # never shown the first's output
    out["programs"].append(second)
    if first.get("crashed"):  # a program that gave no value says nothing of the value: the second one decides alone
        return {**out, "verdict": "match" if second["verdict"] == "match" else "cannot_judge",
                "why": f"a first program gave no value ({first['why']}); a second one: {second['why']}"}
    if second["verdict"] == "no_match" and agree(first["routes"], second["routes"], out["tolerance"]):
        return {**out, "verdict": "no_match", "why": f"{first['why']}; a second, independent program gives the same numbers"}
    said = ("reproduced it" if second["verdict"] == "match" else
            "gave other numbers" if second.get("routes") else f"could not compute it ({second['why']})")
    return {**out, "verdict": "cannot_judge",
            "why": f"a first program did not reproduce it ({first['why']}), and a second, independent one {said}"}


async def check(data: dict[str, Any], quest_root: Path, *, template: Any, ask: Ask, execute: Execute) -> dict[str, Any]:
    """Every known value of the commission ``data`` computed and judged, recorded in ``.fi/known_values.json`` as each
    one is done; a value already recorded for the same commission is read back instead (a resume asks nothing again for
    it). A value whose program could not be asked for is not kept, so the next pass asks again."""
    path = Path(quest_root) / ".fi" / RECORD
    sha = _commission.recorded_sha(Path(quest_root))
    try:
        kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        kept = None
    done = ({str(v.get("name")): v for v in kept.get("values") or [] if isinstance(v, dict)}
            if isinstance(kept, dict) and sha and kept.get("commission_sha256") == sha else {})
    names = _commission.staged_names(data)
    texts = {raw: _source_text(Path(quest_root), name) for raw, name in names.items()}
    folder = Path(quest_root) / ".fi" / "known_values"
    values: list[dict[str, Any]] = []
    for kv in data.get("known_values") or []:
        value = done.get(str(kv["name"]))
        if value is None:
            value = await _one(kv, texts.get(str(kv["source"]["file"]), ""), folder, template=template, ask=ask,
                               execute=execute)
        values.append(value)
        keep = [v for v in values if not any(p.get("unasked") for p in v["programs"])]
        if keep:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"commission_sha256": sha, "at": time.time(), "values": keep}, indent=1,
                                       default=str), encoding="utf-8")
    return {"commission_sha256": sha, "at": time.time(), "values": values}


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
    from the commission: the plan's model gives only the case and the measure. A known value the plan has no check
    for gets one from FI (its name, the commission's case in words, and the same four parts), so every known value
    is checked; plan.md says which (``commission.oracle_gaps``, read before this)."""
    from .oracle_check import _KIND_KEYS, _REFERENCE_KEYS

    protocol = design.get("protocol") if isinstance(design, dict) else None
    if not isinstance(protocol, dict):
        return design
    by = {_fold(kv.get("name")): kv for kv in data.get("known_values") or []}

    def fi_part(kv: dict[str, Any]) -> dict[str, Any]:
        label = source_labels.get(str(kv["source"]["file"]))
        where = f"[{label}]" if label else f"the supplied full text {Path(str(kv['source']['file'])).name}"
        return {"kind": "published_value", "expected": float(kv["value"]), "tolerance": float(kv["tolerance"]),
                "tolerance_mode": "absolute", "reference": f"{where}, where it states: \"{kv['passage']}\""}

    oracles, named = [], set()
    for o in protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else []:
        if isinstance(o, str) and _fold(o) in by:  # a check written as its name alone (oracle_check.declared)
            o = {"name": o.strip(), "check": o.strip()}
        kv = by.get(_fold(o.get("name"))) if isinstance(o, dict) else None
        if kv is not None:
            named.add(_fold(kv.get("name")))
            o = {**{k: v for k, v in o.items() if k not in (*_KIND_KEYS, *_REFERENCE_KEYS, "expected_formula")}, **fi_part(kv)}
        oracles.append(o)
    oracles += [{"name": kv["name"], "check": f"the commission's known value for {kv['case']}", **fi_part(kv)}
                for key, kv in by.items() if key not in named]
    return {**design, "protocol": {**protocol, "oracles": oracles}}
