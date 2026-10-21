"""When the simulation and the plan's own working disagree, each side questions the other, and a third model judges.

A check against a known answer whose expected value FI computed itself from the plan's ``expected_formula``
(:func:`core.oracle_triage.formula_value`) is still only the plan's working: a plan once squared the real part of a
complex field instead of its modulus, and FI then refused the simulation's correct value twice and marked correct code
as failing. So a repair may question such a check, and the plan must answer:

1. **the code questions the plan** -- the repair names the step of the plan's working it disputes and gives the expected
   value as a formula FI can compute (:func:`challenge_of`). A challenge whose formula FI cannot compute, or that does
   not give the value the repair claims, is not taken (a number alone cannot be told apart from the measurement);
2. **the plan answers** -- the plan's model is shown the check and the objection, never the measured value, and either
   concedes with a corrected formula or stands and names what is wrong in the code (:func:`plan_prompt`);
3. **a third model judges** when the plan stands -- shown the check, both workings and the plan's answer, never the
   measured value (:func:`judge_prompt`).

FI computes every formula itself; a side's word alone settles nothing (:func:`settle`). The outcome is ``corrected``
(the plan conceded, or the judge sided with the code, with a formula FI computes to the challenge's value within the
check's own tolerance), ``code`` (the plan, or the judge, names a fault in the code: one more repair, told why), or
``unsettled`` (the check is set aside and goes on unconfirmed, with every side's reason). A corrected check is still the
plan's own working: it never counts as validation against a published value (:mod:`core.evidence`).
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import oracle_check as _oracle
from . import oracle_forms as _forms

#: The prompts' nodes: the plan's model answers the objection (``plan_revise`` picks it, as for every plan change), and
#: the judge is the model named for ``oracle_review`` (a dot name falls back to its first part), else the main model.
PLAN_NODE = "plan_revise"
JUDGE_NODE = "oracle_review.judge"
#: Where FI keeps each answer (in ``.fi/oracle_triage.json``), so a resume never asks again.
RECORD_KEY = "challenge"


#: How close (relative) a number written in a formula or a text may come to the measured value before FI reads it as the
#: measurement itself: three significant digits.
_MEASURED_REL = 5e-3
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _near_measured(x: float, measured: float | None) -> bool:
    return (measured is not None and x == x and abs(x - measured) <= _MEASURED_REL * max(abs(measured), 1e-12)
            and not (x == 0 and measured != 0))


def without_measured(text: Any, measured: float | None) -> str:
    """``text`` with every number in it that is the measured value (to three significant digits) taken out: a blind
    model is shown the code's words, never what the simulation measured (FI knows that value, so this is exact)."""
    text = " ".join(str(text or "").split())
    if measured is None:
        return text

    def one(m: re.Match[str]) -> str:
        try:
            return "(a value not shown)" if _near_measured(float(m.group(0)), measured) else m.group(0)
        except ValueError:
            return m.group(0)

    return _NUMBER_RE.sub(one, text)


def _same_formula(a: Any, b: Any) -> bool:
    return bool(a) and "".join(str(a).split()).lower() == "".join(str(b or "").split()).lower()


def computes(oracle: dict[str, Any], formula: Any, measured: float | None = None) -> float | None:
    """What ``formula`` gives on the check's own case, computed by FI, or ``None`` when FI cannot read it surely (a
    number restated, a name its case does not set, more than one reading). With ``measured`` (a formula written by a
    side that saw the measurement): a formula made of numbers alone, or one that writes the measured value as a number,
    is the measurement restated, not a working."""
    from . import oracle_triage as _triage

    if not isinstance(formula, str) or not formula.strip():
        return None
    probe = {**oracle, "expected_formula": formula}
    try:
        value, _why, unset, used = _forms._formula_result(probe, None)
    except Exception:  # noqa: BLE001 -- a formula FI cannot compute decides nothing
        return None
    if value is None or unset or _triage.num(value) is None:
        return None
    import ast

    text = used or formula
    if _triage.ambiguous(text) or _triage._degrees_unconverted(text):
        return None
    tree, _ = _forms._tree(text, {**_forms.FUNCTIONS, **_forms.SPECIAL_FUNCTIONS})
    if tree is None or not any(isinstance(n, (ast.BinOp, ast.Call, ast.Name)) for n in ast.walk(tree)):
        return None  # a number restated is not a working
    if measured is not None:
        called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - called
        numbers = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                   and not isinstance(n.value, bool)]
        if not names or any(_near_measured(float(x), measured) for x in numbers):
            return None
    return float(value)


def _within(oracle: dict[str, Any], a: float | None, b: float | None) -> bool:
    """Whether two values lie within the check's own tolerance of each other (never widened)."""
    if a is None or b is None:
        return False
    _expected, limit, _mode = _oracle.limit_of({**oracle, "expected": b})
    return limit is not None and abs(a - b) <= limit


def challenge_of(oracle: dict[str, Any], proposal: dict[str, Any],
                 measured: float | None = None) -> tuple[dict[str, Any] | None, str]:
    """The repair's question to the plan, as FI takes it: ``({step, formula, value, reason, measured}, "")``, or
    ``(None, why)`` when it cannot be taken (no step named, no formula FI computes, a formula that restates the measured
    value, or one that does not give the value claimed). ``measured``: what the simulation measured on the check's case;
    kept with the challenge for FI's own use, and taken out of the step a blind model is shown."""
    step = without_measured(proposal.get("disputed_step"), measured)[:600]
    formula = str(proposal.get("expected_formula") or "").strip()
    if not step or not formula:
        return None, ("it does not name the step of the plan's working it disputes and give the expected value as a "
                      "formula (`disputed_step`, `expected_formula`)")
    value = computes(oracle, formula, measured)
    if value is None:
        return None, (f"FI cannot compute its formula ({formula[:120]}) on the check's case as a working: it must use the "
                      "case's settings and may not write the measured value as a number")
    claimed = proposal.get("expected")
    if isinstance(claimed, (int, float)) and not isinstance(claimed, bool) and not _within(oracle, value, float(claimed)):
        return None, (f"its formula gives {_oracle.fmt_digits(value)}, not the {_oracle.fmt_digits(claimed)} it claims")
    return {"step": step, "formula": formula, "value": value, "measured": measured,
            "reason": " ".join(str(proposal.get("reason") or "").split())[:600]}, ""


def _check_lines(oracle: dict[str, Any]) -> str:
    """The check as both the plan and the judge see it: its statement, case, measure, reference and the plan's formula.
    Never the measured value."""
    case = oracle.get("case")
    return "\n".join([
        f"- Name: `{str(oracle.get('name') or '').strip()}`",
        f"- Kind: {_oracle.kind_of(oracle) or _oracle.kind_written(oracle) or '(not given)'}",
        f"- What it checks: {' '.join(str(oracle.get('check') or '').split())[:1200] or '(not given)'}",
        f"- The case it runs on: {json.dumps(case, default=str) if isinstance(case, dict) and case else '(not given)'}",
        f"- How its number is computed from what the simulation returns: `{str(oracle.get('measure') or '').strip()}`",
        f"- The plan's working (its reference): {' '.join(str(_oracle.reference_of(oracle) or '').split())[:2000]}",
        f"- The plan's expected value as a formula: `{str(oracle.get('expected_formula') or '').strip()}`",
    ])


def plan_prompt(template: Any, *, topic: str, oracle: dict[str, Any], challenge: dict[str, Any]) -> str:
    """The plan's model asked to answer the code's objection (prompt ``oracle_challenge_plan``)."""
    return template.substitute(
        topic=" ".join(str(topic or "").split())[:1500], check=_check_lines(oracle),
        step=challenge["step"], formula=challenge["formula"], language=_forms.EXPECTED_FORMULA_LANGUAGE)


def judge_prompt(template: Any, *, topic: str, oracle: dict[str, Any], challenge: dict[str, Any],
                 answer: dict[str, Any]) -> str:
    """The judge asked to decide between the plan's working and the code's objection (prompt ``oracle_challenge_judge``)."""
    return template.substitute(
        topic=" ".join(str(topic or "").split())[:1500], check=_check_lines(oracle),
        step=challenge["step"], formula=challenge["formula"],
        plan_answer=str(answer.get("why") or "(no reason given)")[:1500],
        plan_fault=str(answer.get("code_fault") or "(none named)")[:800],
        language=_forms.EXPECTED_FORMULA_LANGUAGE)


def parse_plan(parsed: Any) -> dict[str, Any]:
    """The plan's answer: ``{"answer": "concede" | "stand", "formula", "code_fault", "why"}``. A reply that is not one
    JSON object, or names neither, stands (the plan's working is kept as it was; only the judge is then asked)."""
    if not isinstance(parsed, dict):
        return {"answer": "stand", "formula": "", "code_fault": "", "why": "the plan's model gave no usable answer"}
    answer = str(parsed.get("answer") or "").strip().lower()
    return {"answer": "concede" if answer.startswith("concede") else "stand",
            "formula": str(parsed.get("expected_formula") or "").strip(),
            "code_fault": " ".join(str(parsed.get("code_fault") or "").split())[:800],
            "why": " ".join(str(parsed.get("why") or "").split())[:1500]}


def parse_judge(parsed: Any) -> dict[str, Any]:
    """The judge's verdict: ``{"side": "plan" | "code" | "undecided", "formula", "why"}`` (anything else is undecided)."""
    if not isinstance(parsed, dict):
        return {"side": "undecided", "formula": "", "why": "the judge gave no usable answer"}
    side = str(parsed.get("side") or "").strip().lower()
    return {"side": side if side in ("plan", "code") else "undecided",
            "formula": str(parsed.get("expected_formula") or "").strip(),
            "why": " ".join(str(parsed.get("why") or "").split())[:1500]}


def settle(oracle: dict[str, Any], challenge: dict[str, Any], answer: dict[str, Any],
           verdict: dict[str, Any] | None) -> dict[str, Any]:
    """What the exchange decides, FI computing every formula itself: ``{"outcome", "expected"?, "by", "text"}``.

    - ``corrected``: the plan conceded, or the judge sided with the code, AND that side's formula gives the challenge's
      value within the check's own tolerance: the check's expected value becomes that value;
    - ``code``: the plan stood and named a fault in the code, and the judge sided with the plan with a formula FI
      computes to the plan's value: the code is repaired once more, told why;
    - ``unsettled``: anything else, including a judge whose own working gives a third value (the plan and the code may
      both be wrong); the check is set aside and goes on unconfirmed, with every side's reason."""
    name = str(oracle.get("name") or "").strip()
    target = challenge["value"]
    plan_value = computes(oracle, answer.get("formula")) if answer.get("answer") == "concede" else None
    echoed = []
    if answer.get("answer") == "concede" and _same_formula(answer.get("formula"), challenge["formula"]):
        echoed.append("the plan's model conceded with the code's own formula, copied")
        plan_value = None
    if answer.get("answer") == "concede" and _within(oracle, plan_value, target):
        return {"outcome": "corrected", "expected": target, "by": "the plan's model conceded",
                "text": (f"the plan's model, asked about the step '{challenge['step'][:160]}' and never shown the "
                         f"measured value, corrected its working to {_oracle.fmt_digits(plan_value)}")}
    judged = verdict or {}
    side = judged.get("side")
    judge_value = computes(oracle, judged.get("formula"))
    copied = side == "code" and _same_formula(judged.get("formula"), challenge["formula"])
    if copied:
        echoed.append("the judge sided with the code with the code's own formula, copied")
    if side == "code" and not copied and _within(oracle, judge_value, target):
        return {"outcome": "corrected", "expected": target, "by": "the judge",
                "text": (f"a third model, never shown the measured value, sided with the code: "
                         f"{str(judged.get('why') or '')[:300]} (its formula gives {_oracle.fmt_digits(judge_value)})")}
    plan_now = computes(oracle, oracle.get("expected_formula"))
    if side == "plan" and answer.get("answer") == "stand" and answer.get("code_fault") and _within(oracle, judge_value,
                                                                                                    plan_now):
        return {"outcome": "code", "by": "the judge",
                "text": (f"the plan's model stands by its working and names a fault in the code "
                         f"({answer['code_fault'][:300]}); a third model sided with the plan: "
                         f"{str(judged.get('why') or '')[:300]}")}
    reasons = [f"the code: {challenge.get('reason') or challenge['step']}"[:400],
               f"the plan: {answer.get('why') or answer.get('code_fault') or 'no reason given'}"[:400], *echoed]
    if verdict is not None:
        reasons.append(f"the judge ({side}): {verdict.get('why') or 'no reason given'}"[:400])
        if judge_value is not None and not _within(oracle, judge_value, target) and not _within(oracle, judge_value,
                                                                                               plan_now):
            # A third value: the plan's working and the code may both be wrong.
            reasons.append(f"the judge's own working gives {_oracle.fmt_digits(judge_value)}, neither the code's "
                           f"{_oracle.fmt_digits(target)} nor the plan's "
                           f"{_oracle.fmt_digits(plan_now) if plan_now is not None else 'value'}: the plan's working and "
                           "the code may both be wrong")
    return {"outcome": "unsettled", "by": "", "text": (
        f"the simulation and the plan's own working disagree on the check '{name}' and neither is confirmed: "
        + "; ".join(reasons))}


# --- the last referee: the value computed numerically from the definitions, not from anyone's working ------------------

#: The node of the request for the numerical computation: the model named for ``oracle_review`` answers it.
NUMERIC_NODE = "oracle_review.numeric"
#: The line the numerical script prints last.
NUMERIC_LINE = "NUMERIC_JSON:"
#: How long FI lets the numerical script run.
NUMERIC_TIMEOUT_S = 300
# What a numerical script may not do: use the simulation's own code (it would agree with the code by construction).
_FORBIDDEN = ("import simulate", "from simulate", "from code", "import code", "sys.path", "importlib", "__import__",
              "open(", "subprocess", "os.system", "os.popen", "pathlib", "read_text", "read_bytes", "glob", "listdir",
              "os.walk", "scandir", "shutil", "ctypes", "exec(", "eval(")


def _model_lines(protocol: dict[str, Any] | None) -> str:
    """The model behind the numbers, as its equations' definitions only: never a check's working or expected value."""
    view = _oracle.model_view((protocol or {}).get("model")) if isinstance(protocol, dict) else None
    if not view:
        return "(the plan gives no model)"
    rows = [f"- {view['summary']}"] if view.get("summary") else []
    for eq in view.get("equations") or []:
        if isinstance(eq, dict):
            text = " ".join(str(eq.get("formula") or eq.get("equation") or eq.get("text") or "").split())
            if text:
                rows.append(f"- {str(eq.get('id') or '').strip()}: {text}"[:600])
    return "\n".join(rows)[:5000] or "(the plan gives no model)"


def numeric_prompt(template: Any, *, topic: str, oracle: dict[str, Any], protocol: dict[str, Any] | None) -> str:
    """The request for a small script that computes the check's number on its case numerically from the definitions
    (prompt ``oracle_numeric_reference``). Shown: the topic, the model's equations, the check's statement, case and
    measure. Never shown: the check's reference or working, its expected value, either side's value, the measurement."""
    case = oracle.get("case")
    return template.substitute(
        topic=" ".join(str(topic or "").split())[:1500], model=_model_lines(protocol),
        name=str(oracle.get("name") or "").strip(),
        check=" ".join(str(oracle.get("check") or "").split())[:1200] or "(not given)",
        case=json.dumps(case, default=str) if isinstance(case, dict) and case else "(not given)",
        measure=str(oracle.get("measure") or "").strip() or "(not given)", line=NUMERIC_LINE)


def numeric_script(reply: Any) -> tuple[str, str]:
    """``(the script, "")`` from the reply's one fenced Python block, or ``("", why)`` when there is none or it uses the
    simulation's own code (or reads or runs anything else)."""
    import re

    text = str(reply or "")
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    script = (blocks[0] if blocks else "").strip()
    if not script:
        return "", "the reply held no Python script"
    bad = next((f for f in _FORBIDDEN if f in script), None)
    if bad:
        return "", f"the script uses `{bad}`: it must compute the number from the definitions alone"
    return script + "\n", ""


def numeric_value(stdout: str) -> tuple[float | None, str]:
    """``(the value, how)`` from the script's last ``NUMERIC_JSON:`` line, or ``(None, why)``."""
    from . import oracle_triage as _triage

    for line in reversed(str(stdout or "").splitlines()):
        if line.startswith(NUMERIC_LINE):
            try:
                data = json.loads(line[len(NUMERIC_LINE):])
            except ValueError:
                return None, "its last line could not be read"
            value = _triage.num(data.get("value")) if isinstance(data, dict) else None
            how = " ".join(str((data or {}).get("how") or "").split())[:400] if isinstance(data, dict) else ""
            return (value, how) if value is not None else (None, "it gave no finite value")
    return None, f"it printed no `{NUMERIC_LINE}` line"


def settle_numeric(oracle: dict[str, Any], challenge: dict[str, Any], verdict: dict[str, Any] | None,
                   numeric: float | None, how: str, before: dict[str, Any]) -> dict[str, Any]:
    """The exchange decided by the number FI computed numerically (``before``: what :func:`settle` said without it).

    - it agrees with the code: ``corrected`` to the code's value;
    - it agrees with the plan's working: ``code`` (the code is repaired, told the value);
    - it agrees with the judge's third value: ``both`` -- the plan's working and the code were both wrong; the check is
      corrected to that value (two independent workings agree on it) and the code is repaired towards it;
    - anything else: ``unsettled``, as before, with the number said."""
    if numeric is None:
        return {**before, "text": before["text"] + f"; the numerical computation gave no value ({how})"}
    target = challenge["value"]
    plan_now = computes(oracle, oracle.get("expected_formula"))
    judge_formula = str((verdict or {}).get("formula") or "")
    judge_value = computes(oracle, judge_formula)
    said = f"an independent numerical computation from the definitions gives {_oracle.fmt_digits(numeric)} ({how})"
    if _within(oracle, numeric, target):
        return {"outcome": "corrected", "expected": target, "by": "a numerical computation",
                "text": f"{said}, as the code does: the plan's working was wrong"}
    if _within(oracle, numeric, plan_now):
        return {"outcome": "code", "by": "a numerical computation",
                "text": f"{said}, as the plan's working does: the code is wrong"}
    if _within(oracle, numeric, judge_value) and judge_value is not None:
        return {"outcome": "both", "expected": judge_value, "formula": judge_formula, "by": "a numerical computation",
                "text": (f"{said}, as the judge's working does, and neither the code's {_oracle.fmt_digits(target)} nor "
                         f"the plan's {_oracle.fmt_digits(plan_now) if plan_now is not None else 'value'}: the plan's "
                         "working and the code were both wrong")}
    return {**before, "text": before["text"] + f"; {said}, which agrees with none of them"}
