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


_FOLD_DEPTH = 100  # deeper than this an expression is not folded (nothing is raised)
_FOLD_CHARS = 1000  # nor is one whose text is longer


def _text_of(node: ast.AST, depth: int = 0) -> str | None:
    """The text an expression of constants is worth before it runs (``"2." + "3846"``, ``"2." * 1``, ``f"{0.2}{5}"``),
    else ``None``."""
    if depth > _FOLD_DEPTH:
        return None
    out: str | None = None
    if isinstance(node, ast.Constant):
        out = node.value if isinstance(node.value, str) else None
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _text_of(node.left, depth + 1), _text_of(node.right, depth + 1)
        out = left + right if left is not None and right is not None else None
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        text = _text_of(node.left, depth + 1)
        times = node.right.value if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) else None
        out = (text * times if text is not None and times is not None and 0 <= times <= _FOLD_CHARS
               and len(text) * times <= _FOLD_CHARS else None)
    elif isinstance(node, ast.JoinedStr):
        parts: list[str | None] = []
        for v in node.values:
            if not isinstance(v, ast.FormattedValue):
                parts.append(_text_of(v, depth + 1))
            elif v.format_spec is not None or v.conversion not in (-1, 115, 114):
                parts.append(None)
            elif isinstance(v.value, ast.Constant) and not isinstance(v.value.value, str):
                parts.append(repr(v.value.value) if v.conversion == 114 else str(v.value.value))
            else:
                inner = _text_of(v.value, depth + 1)
                parts.append(repr(inner) if inner is not None and v.conversion == 114 else inner)
        out = "".join(parts) if all(x is not None for x in parts) else None  # type: ignore[arg-type]
    return out if out is not None and len(out) <= _FOLD_CHARS else None


def _constants(tree: ast.AST) -> list[float]:
    """Every number a program types: its number constants and each number written in a text expression of constants
    (``float("2." + "3846")``), the whole expression and never a part of it (``"0." + "25" + "7"`` is 0.257)."""
    out: list[float] = []
    todo = [tree]
    while todo:  # not recursive: a long chain of "+" must not run out of stack
        n = todo.pop()
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            out.append(float(n.value))
        elif (text := _text_of(n)) is not None:
            out += [x for m in _commission._NUMBER.finditer(text) if (x := _commission.number_value(m)) is not None]
        else:
            todo.extend(ast.iter_child_nodes(n))
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


def judge(routes: list[dict[str, Any]], value: float | Decimal, tolerance: float, bound: str = "") -> tuple[str, str]:
    """``(verdict, why)``: ``match``, ``no_match`` or ``cannot_judge``. Two routes of one program disagree when they are
    further apart than three times their combined error (plus half a unit of the value's last digit); a route reproduces
    the value within the larger of the tolerance and three times its own error, and a limit (``bound``: ``at_least`` /
    ``at_most``) when it is not beyond that on the wrong side: any number above a floor agrees with it, so a floor's match
    says only that the result does not fall short. One route alone is a match only for a :func:`decisive` value."""
    def beyond(r: dict[str, Any]) -> bool:
        gap = r["value"] - float(value)
        side = -gap if bound == "at_least" else gap if bound == "at_most" else abs(gap)
        return side > max(tolerance, 3 * r["error"])

    for a, b in combinations(routes, 2):
        if abs(a["value"] - b["value"]) > 3 * math.hypot(a["error"], b["error"]) + _half_unit(value):
            said = (f"its {a['route']} route gives {_g(a['value'])} (error {_g(a['error'])}) and its "
                    f"{b['route']} route {_g(b['value'])} (error {_g(b['error'])}), further apart than their errors")
            if bound and not any(beyond(r) for r in routes):  # both within the limit: nothing found against it
                return "cannot_judge", f"{said}, though neither is beyond the limit {_g(float(value))}"
            return "no_match", said

    off = [r for r in routes if beyond(r)]
    if off:
        r = off[0]
        return "no_match", (f"its {r['route']} route gives {_g(r['value'])} (error {_g(r['error'])}), not "
                            f"{_commission.stated(float(value), tolerance, bound)}")
    said = ", ".join(f"{r['route']} {_g(r['value'])}" for r in routes)
    if len(routes) < 2 and not decisive(value):
        return "cannot_judge", f"one route only ({said}), and the value is too round to tell a computed number from a typed one"
    if bound:
        return "match", f"{bound.replace('_', ' ')} {_g(float(value))} within {_g(tolerance)}, as the source states: {said}"
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


_ANY_NUMBER = re.compile(r"\.?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?|[¼-¾⅐-⅞]")  # every digit run in an error line, even inside a name


async def _written(kv: dict[str, Any], shown: str, *, template: Any, ask: Ask, second: bool, value: Any,
                   tolerance: float, given: str, note: str, tries: int = 2) -> tuple[str, str, bool]:
    """A usable program for one value: asked, and (with ``tries=2``) once more with the reason when the reply cannot
    be used. Returns ``(script, why, unasked)``; ``script`` is ``""`` when none could be had."""
    script, why = "", ""
    for _ in range(tries):
        try:
            reply = await ask(prompt(template, kv, shown, again=note), second)
        except Exception as e:  # noqa: BLE001 -- a program that could not be had leaves the value unconfirmed
            return "", f"the program could not be asked for ({str(e)[:160] or type(e).__name__})", True
        script, why = program(reply, value, tolerance, given)
        if script:
            return script, "", False
        note = (f"{_SECOND} " if second else "") + f"Your last reply could not be used: {why}. Write the program again."
    return "", f"no usable program after {'two requests' if tries == 2 else 'one request'} ({why})", False


async def _ran(path: Path, script: str, execute: Execute) -> dict[str, Any]:
    """The program written to ``path`` and run: what it printed, its exit code and, when it stopped with an error,
    the last line of that error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    code, out, err = await execute(path)
    routes, cannot = printed(out)
    failed = (err or "").strip().splitlines()[-1][:200] if (err or "").strip() else f"exit code {code}"
    return {"program": path.name, "sha256": _commission.sha256(path), "returncode": code, "stdout_tail": out[-1500:],
            "routes": routes, "_cannot": cannot, "_failed": failed}


def _settled(ran: dict[str, Any], value: Any, tolerance: float, bound: str) -> dict[str, Any]:
    """The verdict on one run program; a program that stopped with an error is not judged on what it printed before."""
    ran = dict(ran)
    cannot, failed, code, routes = ran.pop("_cannot"), ran.pop("_failed"), ran["returncode"], ran["routes"]
    if code != 0 or not routes:
        why = (f"the program stopped with an error ({failed}); what it printed before is not judged" if code != 0 else
               f"the program says the text does not give what it needs: {cannot}" if cannot else
               f"the program printed no `{LINE}` value ({failed})")
        return {**ran, "routes": [], "printed_before_it_stopped": routes if code != 0 else [], "verdict": "cannot_judge",
                "why": why, "crashed": code != 0 or not cannot}
    verdict, why = judge(routes, value, tolerance, bound)
    return {**ran, "verdict": verdict, "why": why}


async def _program(kv: dict[str, Any], shown: str, path: Path, *, template: Any, ask: Ask, execute: Execute,
                   second: bool) -> dict[str, Any]:
    """One program for one value: asked (once more, with the reason, when it cannot be used), run, and judged. A
    program that stops with an error is written once more (one request), told only the last line of that error with
    every number in it taken out (a number the program computed is not shown to the model that writes it again): a
    function the installed library no longer has, or an overflow, leaves the value unjudged though the method is
    sound; the record of the program that stopped is kept whole under ``stopped_first``."""
    value, tolerance, bound = written(kv), float(kv["tolerance"]), _commission.bound_of(kv)
    given = " ".join(withhold(kv.get(k), value) for k in ("case", "passage")) + " " + withhold(shown, value)
    how = dict(template=template, ask=ask, second=second, value=value, tolerance=tolerance, given=given)
    script, why, unasked = await _written(kv, shown, note=_SECOND if second else "", **how)
    if not script:
        return {"verdict": "cannot_judge", "why": why, **({"unasked": True} if unasked else {})}
    ran = await _ran(path, script, execute)
    if ran["returncode"] == 0:
        return _settled(ran, value, tolerance, bound)
    stopped = _settled(ran, value, tolerance, bound)
    error = _ANY_NUMBER.sub("<number>", ran["_failed"])
    note = ((f"{_SECOND} " if second else "") + f"Your last program stopped with an error before it finished: {error}. "
            "Write the program again so that it runs to the end with the installed libraries.")
    script, why, _ = await _written(kv, shown, note=note, tries=1, **how)
    if not script:
        return {**stopped, "why": f"{stopped['why']}; written once more, the reply could not be used ({why})"}
    again = _settled(await _ran(path.with_name(f"{path.stem}_again{path.suffix}"), script, execute), value, tolerance,
                     bound)
    return {**again, "stopped_first": stopped}


async def _one(kv: dict[str, Any], text: str, folder: Path, **how: Any) -> dict[str, Any]:
    out = {"name": kv["name"], "value": float(kv["value"]), "tolerance": float(kv["tolerance"]), "case": kv["case"],
           "passage": kv["passage"], "source": kv["source"], "programs": [],
           **({"bound": _commission.bound_of(kv)} if _commission.bound_of(kv) else {})}
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
    met, missed = (("found it within the limit", "found it beyond the limit") if out.get("bound") else
                   ("reproduced it", "did not reproduce it"))
    said = (met if second["verdict"] == "match" else
            "gave other numbers" if second.get("routes") else f"could not compute it ({second['why']})")
    return {**out, "verdict": "cannot_judge",
            "why": f"a first program {missed} ({first['why']}), and a second, independent one {said}"}


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
        if not any(p.get("unasked") for p in value["programs"]):  # kept beside every value already on disk
            done[str(kv["name"])] = value
            order = [str(k["name"]) for k in data.get("known_values") or []]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"commission_sha256": sha, "at": time.time(),
                                        "values": [done[n] for n in order if n in done]}, indent=1, default=str),
                            encoding="utf-8")
    return {"commission_sha256": sha, "at": time.time(), "values": values}


def mismatches(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [v for v in (record or {}).get("values") or [] if v.get("verdict") == "no_match"]


def problem(wrong: list[dict[str, Any]]) -> str:
    """The return's sentence: which known value two programs found to be other than the commission says."""
    return "; ".join(f"the known value {v['name']!r} ({_commission.stated(v['value'], v['tolerance'], v.get('bound') or '')}) "
                     "was computed from "
                     f"its source by two independent programs, and {v['why']}" for v in wrong)


def evidence(wrong: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What QC needs to decide: per value the commission's number, tolerance, passage and source, and each program's
    file, hash and routes (value and error)."""
    return [{k: v.get(k) for k in ("name", "value", "tolerance", "bound", "case", "passage", "source", "why") if k in v}
            | {"programs": [{k: p.get(k) for k in ("program", "sha256", "routes")} for p in v.get("programs") or []]}
            for v in wrong]


def plan_block(record: dict[str, Any] | None) -> str:
    """What FI's own computation of each known value found, for the plan's model."""
    values = (record or {}).get("values") or []
    if not values:
        return ""
    lines = ["\n\nWHAT FI COMPUTED OF THE KNOWN VALUES (before this plan, each from its source, by a program FI ran)"]
    for v in values:
        said = (f"{str(v['bound']).replace('_', ' ')} " if v.get("bound") else "") + _g(v["value"])
        lines.append(f"- {v['name']!r} ({said}): " + (
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


def wrap(measure: str, value: float, bound: str) -> str:
    """The number FI checks for a known value given as a limit: how far ``measure`` falls beyond ``value`` on the wrong
    side (``max(0, value - (measure))`` for ``at_least``, ``max(0, (measure) - value)`` for ``at_most``), 0 when it
    does not."""
    from .oracle_forms import _literal

    lit = _literal(value)
    return f"max(0, {lit} - ({measure}))" if bound == "at_least" else f"max(0, ({measure}) - {lit})"


def unwrap(measure: Any, value: float, bound: str) -> str | None:
    """The quantity a check measures, with FI's own :func:`wrap` for this limit taken off (a plan written again keeps
    the form FI gave it, which must not be wrapped twice); ``None`` when there is no measure."""
    text = measure.strip() if isinstance(measure, str) else ""
    if not text:
        return None
    try:
        body = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return text
    if (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id == "max" and not body.keywords
            and len(body.args) == 2 and isinstance(body.args[0], ast.Constant) and body.args[0].value == 0
            and isinstance(body.args[1], ast.BinOp) and isinstance(body.args[1].op, ast.Sub)):
        left, right = body.args[1].left, body.args[1].right
        if bound == "at_least" and _same(left, value):
            return _unclip(ast.unparse(right), value)
        if bound == "at_most" and _same(right, value):
            return _unclip(ast.unparse(left), value)
    return _unclip(text, value)


def _same(node: ast.AST, value: float) -> bool:
    """Whether ``node`` is the number ``value`` written as a constant (a negative one is a minus sign before it)."""
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign, node = (-1.0 if isinstance(node.op, ast.USub) else 1.0), node.operand
    return (isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
            and math.isclose(sign * float(node.value), value, rel_tol=1e-12, abs_tol=0.0))


def _unclip(text: str, value: float) -> str:
    """``text`` with an outer ``min(x, value)`` or ``max(x, value)`` taken off: a quantity already clipped at the limit
    cannot show it crossed it (``min(x, 0.25)`` is never above 0.25), so the limit is checked on ``x`` itself."""
    try:
        body = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return text
    if (isinstance(body, ast.Call) and isinstance(body.func, ast.Name) and body.func.id in ("min", "max")
            and not body.keywords and len(body.args) == 2):
        a, b = body.args
        if _same(b, value) and not _same(a, value):
            return ast.unparse(a)
        if _same(a, value) and not _same(b, value):
            return ast.unparse(b)
    return text


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

    def fi_part(kv: dict[str, Any], measure: Any = None) -> dict[str, Any]:
        label = source_labels.get(str(kv["source"]["file"]))
        where = f"[{label}]" if label else f"the supplied full text {Path(str(kv['source']['file'])).name}"
        part = {"kind": "published_value", "expected": float(kv["value"]), "tolerance": float(kv["tolerance"]),
                "tolerance_mode": "absolute", "reference": f"{where}, where it states: \"{kv['passage']}\""}
        bound = _commission.bound_of(kv)
        if not bound:
            return part
        # A limit is checked as how far the quantity falls on its wrong side, expecting 0 (the form a violation takes,
        # core/oracle_forms.py): every check of a number then judges it one-sided, and the shortfall stays the number.
        inner = unwrap(measure, float(kv["value"]), bound)
        part["bound"] = bound
        part["reference"] += (f" (a limit: FI checks that the quantity is {bound.replace('_', ' ')} "
                              f"{float(kv['value']):g}; its number is how far it falls beyond, 0 when it does not)")
        if inner is None:  # nothing to measure the quantity by: the check names no number FI can judge
            del part["expected"]
            return part
        return {**part, "measure": wrap(inner, float(kv["value"]), bound), "expected": 0.0}

    oracles, named = [], set()
    for o in protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else []:
        if isinstance(o, str) and _fold(o) in by:  # a check written as its name alone (oracle_check.declared)
            o = {"name": o.strip(), "check": o.strip()}
        kv = by.get(_fold(o.get("name"))) if isinstance(o, dict) else None
        if kv is not None:
            named.add(_fold(kv.get("name")))
            part = fi_part(kv, o.get("measure"))
            gone = ("expected", "measure") if _commission.bound_of(kv) and "expected" not in part else ()
            o = {**{k: v for k, v in o.items()
                    if k not in (*_KIND_KEYS, *_REFERENCE_KEYS, "expected_formula", "bound", *gone)}, **part}
        oracles.append(o)
    oracles += [{"name": kv["name"], "check": f"the commission's known value for {kv['case']}", **fi_part(kv)}
                for key, kv in by.items() if key not in named]
    criteria = protocol.get("criteria")
    if isinstance(criteria, list) and any(_commission.bound_of(kv) for kv in by.values()):
        kept, _left_out = limit_criteria(criteria, data)
        return {**design, "protocol": {**protocol, "oracles": oracles, "criteria": kept}}
    return {**design, "protocol": {**protocol, "oracles": oracles}}


def limit_criteria(criteria: list[Any], data: dict[str, Any]) -> tuple[list[Any], list[str]]:
    """``(the criteria, the names of those left out)``: the check of a commission's limit measures how far the quantity
    falls beyond it, so a criterion on the check's value would read that shortfall as the quantity. One that says the
    limit itself (``higher`` than a floor, ``lower`` than a ceiling, at the limit) is read on the shortfall, lower is
    better, met at 0; any other bar on the quantity's value cannot be read from the check and is left out."""
    limits = {_fold(kv.get("name")): kv for kv in data.get("known_values") or [] if _commission.bound_of(kv)}
    kept: list[Any] = []
    left_out: list[str] = []
    for c in criteria:
        kv = limits.get(_fold(c.get("oracle"))) if isinstance(c, dict) and c.get("use") == "value" else None
        if kv is None:
            kept.append(c)
            continue
        same = c.get("direction") == ("higher" if _commission.bound_of(kv) == "at_least" else "lower")
        target = _number(c.get("target"))
        if same and target is not None and math.isclose(target, float(kv["value"]), rel_tol=1e-12, abs_tol=0.0):
            kept.append({**c, "use": "error", "direction": "lower", "target": 0})
        else:
            left_out.append(str(c.get("name") or c.get("oracle")))
    return kept, left_out
