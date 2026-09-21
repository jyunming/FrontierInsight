"""An oracle the experiment must pass before its main run: something independent of the script's own numbers.

Every check that ran before this looked at internal consistency. The bounds a design declares (``result_assertions``) only
say a number is legal; the numeric, statistics and provenance audits only say the paper copied the script's output
faithfully; the reviewer reads prose. A simulator that uses the wrong model, the wrong parameter or the wrong estimator
still prints legal numbers, and every one of those checks is green. What was missing is a check against something the
script did not produce: a closed form, a limiting case, an invariant that must hold, a small case whose exact answer is
known, or a second implementation.

The plan's protocol therefore declares its **oracles** (``protocol.oracles``: a ``name``, a ``check`` that says what is
compared with what, and optionally a ``kind`` and a ``tolerance``), and the script is written to answer them. When the
environment variable ``FI_ORACLE`` is ``1`` the script does not run its sweep: it runs the declared checks, each on a small
fast case, prints one line

    ORACLE_JSON: {"checks": [{"name": "...", "passed": true, "value": 0.98, "expected": 1.0, "tolerance": 0.05}]}

and exits 0 when every check passed, 1 when one did not. The engine runs it that way before the pilot and the main run, and
this module reads the answer. It needs no model and no engine state; the engine decides what to do about a problem (ask for a
repair, then stop the quest).

A **problem** is any of: the design declares no oracle; the script printed no ``ORACLE_JSON`` line; a declared oracle does not
appear among the checks; a check reports ``passed`` other than true; the script exited non-zero.
"""

from __future__ import annotations

import json
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
        if check is None:
            out.append(f"the declared oracle {name!r} was not checked (the script reported: {', '.join(sorted(by_name)) or 'nothing'})")
        elif check.get("passed") is not True:
            detail = ", ".join(
                f"{key} {_fmt(check[key])}" for key in ("value", "expected", "tolerance") if key in check
            )
            out.append(f"the oracle {name!r} failed" + (f" ({detail})" if detail else ""))
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
        "The contract: when FI_ORACLE is 1 the script must NOT run its sweep. It runs each declared oracle check on a small, "
        "fast case (seconds), prints ONE line `ORACLE_JSON: {\"checks\": [{\"name\": <the declared name>, \"passed\": true or "
        "false, \"value\": ..., \"expected\": ..., \"tolerance\": ...}, ...]}` and exits 0 if every check passed, 1 if not.\n\n"
        "If a check FAILED, find out which is wrong before changing anything: the simulator or estimator (fix it) or the check "
        "itself (fix the check), and say which in `patch_summary`. Never make a check pass by loosening its tolerance, "
        "skipping it or hard-coding its result: a check the script can always pass is not an oracle. If the checks were "
        "missing, add them for every declared oracle.\n\n"
        "Keep everything else unchanged: the same functions, outputs and figures, the handling of FI_PILOT and "
        "FI_REPLICATE_SEED, and the same final RESULT_JSON line. Return the whole script in `code`, one sentence in "
        "`patch_summary`, and leave `give_up_reason` empty."
    )
