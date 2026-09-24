"""An oracle the experiment must pass before its main run: something independent of the script's own numbers.

Every check that ran before this looked at internal consistency. The bounds a design declares (``result_assertions``) only
say a number is legal; the numeric, statistics and provenance audits only say the paper copied the script's output
faithfully; the reviewer reads prose. A simulator that uses the wrong model, the wrong parameter or the wrong estimator
still prints legal numbers, and every one of those checks is green. What was missing is a check against something the
script did not produce: a closed form, a limiting case, an invariant that must hold, a small case whose exact answer is
known, or a second implementation.

The plan's protocol therefore declares its **oracles** (``protocol.oracles``: a ``name``, a ``check`` that says what is
compared with what, a numeric ``expected`` and ``tolerance`` that the check is judged by, optionally a ``tolerance_mode``
(``absolute``, the default, or ``relative``), a ``kind`` and the ``reference`` the expected value comes from), and the script is
written to *measure* them. When the environment variable ``FI_ORACLE`` is ``1`` the script does not run its sweep: it computes
the value of each declared check on a small fast case, prints one line

    ORACLE_JSON: {"checks": [{"name": "...", "value": 0.98, "diagnostics": {"solver_success": true}}]}

and exits 0. **The engine judges**: it takes ``expected`` and ``tolerance`` from the (frozen) protocol, never from the script,
and computes ``abs(value - expected) <= tolerance`` (or ``tolerance * abs(expected)``) itself. A ``passed`` the script prints,
and any ``expected`` or ``tolerance`` it prints, are not read as a verdict: a script that writes its own pass/fail, its own
expected value and its own tolerance can always be made to pass, which is what an oracle is for not being. (A script that
reports a check as failed itself is still a problem.) For an invariant (conservation, monotonicity) the value is the worst
violation observed and the expected value is 0.

A **problem** is any of: the design declares no oracle; a declared oracle fixes no numeric ``expected`` and ``tolerance`` (the
engine cannot judge it); the script printed no ``ORACLE_JSON`` line; a declared oracle does not appear among the checks or
reports no finite numeric ``value``; the value is outside the tolerance; the script reports the check as failed itself; the
script exited non-zero.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

_LINE_RE = re.compile(r"^\s*ORACLE_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


def declared(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The oracles the protocol declares, each as a mapping with at least a ``name``."""
    items = protocol.get("oracles") if isinstance(protocol, dict) else None
    out: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip(), "check": item.strip()})
    return out


def parse(stdout: str) -> dict[str, Any] | None:
    """The last ``ORACLE_JSON`` line of ``stdout`` as a mapping; ``None`` when there is none or it is not JSON."""
    found = None
    for match in _LINE_RE.finditer(stdout or ""):
        try:
            value = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(value, dict):
            found = value
    return found


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def limit_of(oracle: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """``(expected, limit, mode)`` an oracle is judged by: ``limit`` is the largest difference from ``expected`` that counts as
    agreeing (``None`` when the oracle cannot be judged: no numeric expected or tolerance, or a relative tolerance around 0)."""
    expected, tolerance = _num(oracle.get("expected")), _num(oracle.get("tolerance"))
    mode = "relative" if str(oracle.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
    if expected is None or tolerance is None or tolerance < 0:
        return None, None, mode
    if mode == "relative":
        if expected == 0:
            return expected, None, mode
        return expected, tolerance * abs(expected), mode
    return expected, tolerance, mode


def unjudgeable(oracles: list[dict[str, Any]]) -> list[str]:
    """The declared oracles the engine cannot judge, one sentence each: they fix no numeric ``expected`` and ``tolerance``."""
    out: list[str] = []
    for oracle in oracles:
        expected, limit, mode = limit_of(oracle)
        name = str(oracle["name"]).strip()
        if limit is None and expected is not None and mode == "relative":
            out.append(f"the oracle {name!r} has a relative tolerance around an expected value of 0 (use an absolute tolerance)")
        elif limit is None:
            out.append(
                f"the oracle {name!r} fixes no numeric `expected` and `tolerance` in the protocol, so the engine cannot judge it "
                "(the script's own pass/fail is not evidence)"
            )
    return out


def judged(oracles: list[dict[str, Any]], reported: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What the engine made of each declared oracle: its value as the script reported it, what the protocol expects and allows,
    and the verdict computed here. For the record in ``needs/ORACLE_CHECK.json``."""
    checks = (reported or {}).get("checks")
    by_name = {str(c.get("name") or "").strip().lower(): c for c in checks if isinstance(c, dict)} if isinstance(checks, list) else {}
    out: list[dict[str, Any]] = []
    for oracle in oracles:
        name = str(oracle["name"]).strip()
        check = by_name.get(name.lower())
        expected, limit, mode = limit_of(oracle)
        value = _num(check.get("value")) if check else None
        verdict = None if (value is None or limit is None) else abs(value - expected) <= limit  # type: ignore[operator]
        out.append({
            "name": name, "value": value, "expected": expected, "tolerance": oracle.get("tolerance"), "mode": mode,
            "limit": limit, "passed_by_engine": verdict, "script_said": (check or {}).get("passed"),
        })
    return out


def last_judged(record: Any) -> list[dict[str, Any]]:
    """The engine's verdicts from the last attempt of a ``needs/ORACLE_CHECK.json`` record that let the main run go on
    (``ok`` or ``warned``); empty for anything else."""
    if not isinstance(record, dict) or record.get("status") not in ("ok", "warned"):
        return []
    attempts = record.get("attempts")
    last = attempts[-1] if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else {}
    return [j for j in last.get("judged") or [] if isinstance(j, dict) and j.get("name")]


def analysis_note(judged_list: list[dict[str, Any]]) -> str:
    """What the analysis is told about the oracles the engine judged before the main run ('' when there were none).

    A script often re-judges its own oracles in its results with its own copy of each tolerance, and that copy can be
    out of date: in one real quest a person corrected a tolerance in the plan, the engine's check passed under it, and
    the analysis still reported the oracle as failed because the script's results carried the old value."""
    lines = []
    for j in judged_list:
        value, expected, limit = _num(j.get("value")), _num(j.get("expected")), _num(j.get("limit"))
        verdict = {True: "passed", False: "failed"}.get(j.get("passed_by_engine"), "not judged")
        if value is None or expected is None or limit is None:
            lines.append(f"- {j['name']}: {verdict}")
        else:
            lines.append(f"- {j['name']}: measured {_fmt(value)}, expected {_fmt(expected)} within {_fmt(limit)}: {verdict}")
    if not lines:
        return ""
    return (
        "[FI NOTE] Before the main run the engine checked the protocol's oracles against the expected values and "
        "tolerances the protocol fixes:\n" + "\n".join(lines) + "\n"
        "These verdicts are the ones that count. If the results below also judge these oracles (a pass/fail flag, or the "
        "script's own copy of an expected value or tolerance), that copy can be out of date: report the engine's verdicts "
        "and numbers above, and do not report an oracle as failed or passed on the script's word.\n\n"
    )


def problems(oracles: list[dict[str, Any]], reported: dict[str, Any] | None, returncode: int, timed_out: bool = False) -> list[str]:
    """What is wrong with the oracle run, one sentence each; empty when every declared oracle ran and passed."""
    if not oracles:
        return [
            "the protocol declares no oracle, so nothing independent of the script's own numbers checks that they are right "
            "(add `oracles` to the protocol: a closed form, a limiting case, an invariant, an exact small case)"
        ]
    if timed_out:
        return ["the script did not finish the oracle checks in time (run with FI_ORACLE=1: each check must be small and fast)"]
    if reported is None:
        return [
            "the script printed no `ORACLE_JSON:` line when run with FI_ORACLE=1 "
            f"(exit code {returncode}); it must run the declared oracle checks and print one"
        ]
    checks = reported.get("checks")
    checks = checks if isinstance(checks, list) else []
    by_name = {str(c.get("name") or "").strip().lower(): c for c in checks if isinstance(c, dict)}
    out: list[str] = []
    for oracle in oracles:
        name = str(oracle["name"]).strip()
        check = by_name.get(name.lower())
        expected, limit, mode = limit_of(oracle)
        if check is None:
            out.append(f"the declared oracle {name!r} was not checked (the script reported: {', '.join(sorted(by_name)) or 'nothing'})")
        elif limit is None:
            out.append(unjudgeable([oracle])[0])
        elif _num(check.get("value")) is None:
            out.append(f"the oracle {name!r} reported no finite numeric `value` (it reported {check.get('value')!r}): the script measures, the engine judges")
        else:
            value = _num(check.get("value"))
            if abs(value - expected) > limit:  # type: ignore[operator]
                out.append(
                    f"the oracle {name!r} failed: the script measured {_fmt(value)}, the protocol expects {_fmt(expected)} "
                    f"within {_fmt(limit)} ({mode} tolerance {_fmt(_num(oracle.get('tolerance')))})"
                )
            elif check.get("passed") is False:
                out.append(f"the oracle {name!r} is within its tolerance, but the script reports the check as failed itself: find out why")
    if not out and returncode != 0:
        out.append(f"every declared oracle passed, but the script exited with code {returncode}")
    return out


def directive(oracles: list[dict[str, Any]], found: list[str]) -> str:
    """What stands where a traceback would in the repair request."""
    declared_block = json.dumps(oracles, indent=2)
    return (
        "This script has NOT run its experiment yet: it was run with the environment variable FI_ORACLE=1 to check its "
        "oracles, and that check did not pass. The account of a crash above does not apply.\n\n"
        "The oracles the design declares (independent of the script's own numbers):\n" + declared_block + "\n\n"
        "What went wrong:\n" + "\n".join(f"- {p}" for p in found) + "\n\n"
        "The contract: when FI_ORACLE is 1 the script must NOT run its sweep. It MEASURES each declared oracle on a small, fast "
        "case (seconds) and prints ONE line `ORACLE_JSON: {\"checks\": [{\"name\": <the declared name>, \"value\": <the "
        "number it measured>, \"diagnostics\": {...}}, ...]}`, then exits 0. It does NOT decide pass or fail and it does not "
        "state the expected value or the tolerance: the engine judges the value against the `expected` and `tolerance` the "
        "protocol fixes above (for an invariant the value is the worst violation observed, and the expected value is 0).\n\n"
        "If a value is outside its tolerance, find out which is wrong before changing anything: the simulator or estimator (fix "
        "it) or the way the value is measured (fix that), and say which in `patch_summary`. Never make a check pass by "
        "measuring something else, skipping it or hard-coding its value: a check the script can always pass is not an oracle. If "
        "the checks were missing, add them for every declared oracle.\n\n"
        "The check itself can be what is wrong: an `expected` or `tolerance` the method cannot reach on that case (below its "
        "known error at that step or sample size), or a measurement that is not well defined (a convergence order read far "
        "from the asymptotic regime, two methods compared on different quantities). You cannot change the protocol, and you must "
        "not bend the script to hide it. Instead, besides `code`, return `oracle_change`: a list of {\"name\": <the declared "
        "name>, \"expected\": <number>, \"tolerance\": <number>, \"tolerance_mode\": \"absolute\" | \"relative\", \"check\": "
        "<the corrected check, only if the measurement itself must change>, \"reason\": <the method's known error or the flaw, "
        "with the numbers>}. A person decides whether to accept it; nothing changes without them.\n\n"
        "Keep everything else unchanged: the same functions, outputs and figures, the handling of FI_PILOT and "
        "FI_REPLICATE_SEED, and the same final RESULT_JSON line. Return the whole script in `code`, one sentence in "
        "`patch_summary`, and leave `give_up_reason` empty."
    )


def proposals(raw: Any, oracles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The usable entries of a repair's ``oracle_change``: each names a declared oracle, gives a finite ``expected`` and a
    non-negative ``tolerance``, and says why. Anything else is dropped, never guessed at."""
    names = {str(o.get("name")).strip().lower(): str(o.get("name")).strip() for o in oracles}
    out: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = names.get(str(item.get("name") or "").strip().lower())
        expected, tolerance = _num(item.get("expected")), _num(item.get("tolerance"))
        reason = str(item.get("reason") or "").strip()
        if name is None or expected is None or tolerance is None or tolerance < 0 or not reason:
            continue
        mode = "relative" if str(item.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
        entry: dict[str, Any] = {"name": name, "expected": expected, "tolerance": tolerance, "tolerance_mode": mode, "reason": reason[:600]}
        check = str(item.get("check") or "").strip()
        if check:
            entry["check"] = check[:600]
        out.append(entry)
    return out


def proposal_request(proposal: dict[str, Any]) -> str:
    """The ``--revise-plan`` request that applies one proposal to the plan, word for word.

    It is shown inside a double-quoted command a person copies into a shell, and the check and the reason are the model's
    own words: a quote, ``$``, a backtick or a backslash in them could end the argument or run something when pasted
    (``$(...)`` in bash, a backtick escape in PowerShell). Those become plain characters, so the pasted command only
    ever carries text."""
    change = f"expected {_fmt(proposal['expected'])}, tolerance {_fmt(proposal['tolerance'])} ({proposal['tolerance_mode']})"
    if proposal.get("check"):
        change += f", and its check reads: {proposal['check']}"
    return _shell_safe(f"Change the oracle '{proposal['name']}' to {change}. Reason: {proposal['reason']} Change nothing else.")


def _shell_safe(text: str) -> str:
    """``text`` with nothing a shell acts on inside double quotes: ``"`` and the curly double quotes PowerShell also
    ends a string on become ``'``, ``!`` (bash history) becomes ``.``, ``$``, backticks and backslashes are dropped, and
    line breaks become spaces."""
    text = re.sub(r'["“”„‟]', "'", text).replace("!", ".")
    text = re.sub(r"[$`\\]", "", text)
    return re.sub(r"\s+", " ", text).strip()
