"""The card a person reads when the known-answer checks stop a quest: why it stopped, in numbers, and what to do.

The oracle gate (``Engine._oracle_gate``) judges each check FI declared against a known answer, repairs the script a
bounded number of times, and stops before the main run when they still do not pass. Everything a person needs to
decide who is wrong (the simulation, or the check) is already in hand at that point: each check's expected value and
where it comes from, the measured value and who measured it, the tolerance, the case it ran on, what the repairs did
and said. Before this card, the terminal showed none of it and the web and VS Code showed a generic sentence.

:func:`build` turns what the gate has into ONE structured payload (a plain mapping, kept in ``.fi/pause.json`` and
``.fi/todo.json``); :mod:`core.todo` renders it into ``NEXT_STEP.md`` and the terminal, and the web page and the VS Code
chat show the same text and add buttons from its ``actions``. Nothing here judges or changes a verdict: it only says what
was judged. No model is called.

The payload (``type: "known_answer_check"``):

- ``summary``: one plain sentence of what happened.
- ``checks``: one entry per check that did not pass: its plain name (from its ``check``) and ``id``, ``kind``,
  ``expected`` with its ``reference``, ``measured`` and who measured it, ``tolerance``, the ``gap`` (absolute, relative,
  times the tolerance, the ratio to the expected value), ``case`` and ``measure``, ``where`` in the script the number is
  computed (file, line, a short excerpt, when it can be found reliably), the ``error`` when nothing was measured, and the
  repair's ``proposal`` when it called the check wrong.
- ``causes``: the most likely causes, each with the evidence FI has for it.
- ``tried``: what FI already did (repairs, and what each changed).
- ``actions``: two or three one-step ways on, each with the exact command per interface. A consumer must ignore an
  action ``id`` it does not know: more kinds of action can be added without changing the others.
- ``notes``: what holds for this quest (a frozen protocol, a research quest).
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

from . import oracle_check as _oracle
from . import oracle_forms as _forms

TYPE = "known_answer_check"
#: The one plain name a person reads for an oracle, on every surface.
NAME = "known-answer check"

_REPAIR_SAID = {
    "applied": "changed the script",
    "set_aside_disputed": "said the check itself is wrong; its code was not used",
    "not_usable": "gave no usable script; nothing was changed",
    "no_code": "changed nothing",
    "call_failed": "the model did not answer; nothing was changed",
    "reverted_disputed_changed": "made a disputed check pass by changing the script, so FI put the script back",
}

# A line of a Python traceback that names the exception ("KeyError: 'FI_RAW_DIR'", "ValueError", "numpy.linalg.LinAlgError: ...").
_TB_LINE_RE = re.compile(r"^\s*(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit|Interrupt|Warning|Fault)\b(?::.*)?$")
# The same, written inside a sentence ("... could not be run on its case (KeyError: 'FI_RAW_DIR' (at code/simulate.py line 12))").
_EXC_INLINE_RE = re.compile(r"\b(?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit|Fault)\b")
_AT_RE = re.compile(r"\(at ([^()]+ line \d+)\)")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _short(text: Any, most: int = 160) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= most else flat[: most - 1].rstrip() + "…"


def _about(x: float) -> str:
    """A factor said plainly: 317, 20, 1.6."""
    return f"{x:.0f}" if x >= 10 else f"{x:.2g}"


def gap(value: Any, expected: Any, limit: Any) -> dict[str, Any] | None:
    """How far a measured value is from its expected value: ``absolute``, ``relative`` (to the expected value, when it is
    not 0), ``times_tolerance`` and ``ratio`` (measured / expected, both non-zero), and ``text``, one plain sentence."""
    v, e, lim = _num(value), _num(expected), _num(limit)
    if v is None or e is None:
        return None
    d = abs(v - e)
    out: dict[str, Any] = {"absolute": d}
    parts = [f"off by {d:.4g}"]
    if lim is not None and lim > 0:
        out["times_tolerance"] = d / lim
        parts[0] += f" ({_about(d / lim)} times the tolerance)"
    if e != 0:
        out["relative"] = d / abs(e)
    if e != 0 and v != 0:
        ratio = v / e
        out["ratio"] = ratio
        if ratio < 0:
            parts.append("the measured value has the opposite sign of the expected one")
        elif ratio >= 1.5:
            parts.append(f"the measured value is about {_about(ratio)} times the expected one")
        elif ratio <= 1 / 1.5:
            parts.append(f"the measured value is about 1/{_about(1 / ratio)} of the expected one")
        else:
            parts.append(f"{d / abs(e) * 100:.3g}% of the expected value")
    elif e != 0 and v == 0:
        parts.append("the measured value is exactly 0")
    out["text"] = "; ".join(parts)
    return out


def last_error(text: str) -> str:
    """The last exception a traceback or a sentence names ("KeyError: 'FI_RAW_DIR'"), or ``""``."""
    for line in reversed((text or "").splitlines()):
        if _TB_LINE_RE.match(line) and not line.strip().startswith(("File ", "During handling")):
            return _short(line.strip(), 240)
    found = list(_EXC_INLINE_RE.finditer(text or ""))
    if not found:
        return ""
    rest = (text or "")[found[-1].start():].splitlines()[0]
    rest = _AT_RE.sub("", rest).rstrip(" ;.")
    while rest.endswith(")") and rest.count(")") > rest.count("("):
        rest = rest[:-1].rstrip()
    return _short(rest, 240)


def error_at(text: str, quest_root: Path | str | None = None) -> str:
    """Where the last error happened in the quest's own code (``code/simulate.py line 12``), or ``""``."""
    from .trial_runner import last_frame

    inline = _AT_RE.findall(text or "")
    if inline:
        return inline[-1]
    return last_frame(text or "", quest_root)


def _excerpt(lines: list[str], line: int, around: int = 2) -> list[str]:
    start, end = max(1, line - around), min(len(lines), line + around)
    return [f"{n:>4} | {lines[n - 1].rstrip()[:120]}" for n in range(start, end + 1)]


def _key_lines(tree: ast.AST, names: set[str]) -> list[tuple[int, str]]:
    """``(line, name)`` for each place ``tree`` writes one of ``names`` as a returned key: a dict literal's key, an
    assignment to ``x["name"]``, or ``dict(name=...)``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in names:
                    found.append((key.lineno, str(key.value)))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                        and t.slice.value in names):
                    found.append((t.lineno, str(t.slice.value)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            for kw in node.keywords:
                if kw.arg in names:
                    found.append((kw.value.lineno, str(kw.arg)))
    return sorted(found)


def locate(script: Path, quest_root: Path, oracle: dict[str, Any], *, trial: bool) -> dict[str, Any] | None:
    """Where in ``script`` the number a check reads is computed: ``{file, line, what, excerpt}``, or ``None`` when it
    cannot be found reliably (the script cannot be read or parsed, or names nothing the check reads)."""
    try:
        source = Path(script).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return None
    lines = source.splitlines()
    try:
        rel = Path(script).resolve().relative_to(Path(quest_root).resolve()).as_posix()
    except ValueError:
        rel = Path(script).name
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    name = str(oracle.get("name") or "").strip()
    own = _oracle.case_of(oracle)

    def at(line: int, what: str) -> dict[str, Any]:
        return {"file": rel, "line": line, "what": what, "excerpt": _excerpt(lines, line)}

    if trial and own is not None:
        measure = own[1]
        wanted = set([measure] if _forms.is_name(measure) else _forms.names(measure))
        entry = next((e for e in ("run_trial", "run_cell") if e in functions), None)
        if entry is not None and wanted:
            inside = _key_lines(functions[entry], wanted)
            if inside:
                line, key = inside[0]
                return at(line, f"{entry}(), where it returns `{key}`, the number this check reads")
        anywhere = _key_lines(tree, wanted) if wanted else []
        if anywhere:
            line, key = anywhere[0]
            return at(line, f"where the simulation returns `{key}`, the number this check reads")
        if entry is not None:
            return at(functions[entry].lineno, f"{entry}(), the function FI calls on this check's case (it reads "
                                               f"`{measure}` from what it returns)")
        return None
    if trial:
        if "oracle" not in functions:
            return None
        node = functions["oracle"]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.strip().lower() == name.lower():
                return at(sub.lineno, "oracle(), where it returns this check's value")
        return at(node.lineno, "oracle(), which returns the value of each check that has no case")
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value.strip().lower() == name.lower():
            return at(sub.lineno, "where the script reports this check (its FI_ORACLE=1 branch)")
    first = next((i for i, text in enumerate(lines, 1) if "FI_ORACLE" in text), None)
    return at(first, "the script's FI_ORACLE=1 branch, which reports the checks") if first else None


def _plain_name(oracle: dict[str, Any]) -> str:
    return _short(oracle.get("check"), 140) or str(oracle.get("name") or "").strip()


def _case_text(case: Any) -> str:
    if not isinstance(case, dict) or not case:
        return ""
    return ", ".join(f"{k}={v}" for k, v in case.items())


def _matching(found: list[str], name: str) -> list[str]:
    return [f for f in found if repr(name) in f]


def build(
    *,
    quest_id: str,
    quest_root: Path,
    script: Path,
    found: list[str],
    oracles: list[dict[str, Any]],
    judged: list[dict[str, Any]] | None = None,
    attempts: list[dict[str, Any]] | None = None,
    proposals: list[dict[str, Any]] | None = None,
    disputed: list[str] | None = None,
    trial: bool = False,
    frozen: bool = False,
    research: bool = False,
    interview_made: bool = False,
    repairs: int = 2,
    kept: str | None = None,
    stderr_tail: str = "",
) -> dict[str, Any]:
    """The card for a stop at the known-answer checks (see the module docstring). Pure: reads ``script`` to say where a
    number is computed, nothing else."""
    attempts = [a for a in attempts or [] if isinstance(a, dict)]
    judged = [j for j in judged or [] if isinstance(j, dict)]
    by_judged = {str(j.get("name") or "").strip().lower(): j for j in judged}
    proposals = [p for p in proposals or [] if isinstance(p, dict) and p.get("name")]
    by_proposal = {str(p["name"]).strip().lower(): p for p in proposals}
    disputed_set = {str(n).strip().lower() for n in disputed or []}
    last_checks = attempts[-1].get("checks") if attempts else None
    reported_by_name = {str(c.get("name") or "").strip().lower(): c for c in last_checks or [] if isinstance(c, dict)}
    entry = next((e for e in ("run_trial", "run_cell") if trial and e in _entries(script)), "run_cell")
    run_text = "; ".join(found) + "\n" + (stderr_tail or "")

    checks: list[dict[str, Any]] = []
    causes: list[dict[str, str]] = []
    for oracle in oracles:
        name = str(oracle.get("name") or "").strip()
        key = name.lower()
        j = by_judged.get(key, {})
        if j.get("passed_by_engine") is True:
            continue
        expected, limit, mode = _oracle.limit_of(oracle)
        value = _num(j.get("value"))
        engine = j.get("measured_by") == "engine"
        if limit is None:
            status = "cannot_judge"
        elif value is None:
            status = "not_measured"
        else:
            status = "failed"
        if value is not None and expected is not None:
            shown_value, shown_expected = _oracle.fmt_pair(value, expected, limit)
        else:
            shown_value = _oracle.fmt_digits(value) if value is not None else ""
            shown_expected = _oracle.fmt_digits(expected) if expected is not None else ""
        kind = _oracle.kind_of(oracle)
        problems = _matching(found, name)
        own = _oracle.case_of(oracle)
        check: dict[str, Any] = {
            "id": name,
            "name": _plain_name(oracle),
            "kind": kind or _oracle.kind_written(oracle) or "",
            "kind_words": _oracle.KINDS.get(kind or "", ""),
            "status": status,
            "disputed": key in disputed_set,
            "expected": expected,
            "expected_text": shown_expected,
            "reference": _oracle.reference_of(oracle),
            "measured": value,
            "measured_text": shown_value,
            "measured_by": ("fi" if engine else "script") if value is not None else "",
            "measured_by_text": (
                (f"FI, which called {entry}() on the check's case itself" if engine else
                 "the script's own oracle() (FI did not run the simulation for it)" if trial else
                 "the script itself, run with FI_ORACLE=1") if value is not None else ""),
            "tolerance": _num(oracle.get("tolerance")),
            "tolerance_mode": mode,
            "limit": limit,
            "limit_text": _oracle.fmt_digits(limit) if limit is not None else "",
            "gap": gap(value, expected, limit) if status == "failed" else None,
            "case": own[0] if own else (oracle.get("case") if isinstance(oracle.get("case"), dict) else None),
            "measure": own[1] if own else str(oracle.get("measure") or ""),
            "where": locate(script, quest_root, oracle, trial=trial),
            "problems": problems,
        }
        raw = (reported_by_name.get(key) or {}).get("value")
        if value is None and raw is not None:
            # A value was reported but is not a finite number (nan, inf, text): say what it was, not "nothing".
            check["reported"] = _short(repr(raw), 60)
        if status == "not_measured":
            text = "; ".join(problems) or run_text
            error = last_error(text)
            if error:
                check["error"] = error
                at = error_at(text if problems else stderr_tail or text, quest_root)
                if at:
                    check["error_at"] = at
        proposal = by_proposal.get(key)
        if proposal is not None:
            check["proposal"] = dict(proposal)
        checks.append(check)
        causes += _causes_for(check, oracle, proposal, reported_by_name.get(key), attempts)

    # A run that names no check (a crash before any check, no ORACLE_JSON line): its error is the cause for all of them.
    run_error = ""
    if any(c["status"] == "not_measured" for c in checks) and not any(c.get("error") for c in checks):
        run_error = last_error(stderr_tail) or last_error("; ".join(found))
        if run_error:
            at = error_at(stderr_tail, quest_root) or error_at("; ".join(found), quest_root)
            causes.insert(0, {
                "text": "The simulation stopped with an error before it reported the checks, so nothing was measured: "
                        "the script is what needs fixing here, not the checks' expected values.",
                "evidence": run_error + (f" (at {at})" if at else "")
                + _same_each_time(attempts, run_error),
            })
    for sentence in _oracle.duplicate_names(oracles, {"checks": last_checks} if last_checks else None):
        causes.append({"text": "Two checks share one name, so a value may have been judged against the other "
                               "check's expected value.", "evidence": sentence[0].upper() + sentence[1:] + "."})

    n_failed = sum(1 for c in checks if c["status"] == "failed")
    if not oracles:
        summary = ("FI stopped before the main run: the plan has no known-answer check, so nothing independent of the "
                   "script's own numbers shows they are right.")
    elif checks and all(c["status"] == "cannot_judge" for c in checks):
        summary = (f"FI stopped before the main run: {len(checks)} known-answer check(s) give no number to compare "
                   "with, so FI cannot judge them.")
    elif checks and all(c["status"] != "failed" for c in checks):
        summary = ("FI stopped before the main run: the known-answer checks could not be measured, so nothing was "
                   "compared with its expected value.")
    elif not checks:
        # Every check passed, but the run itself has a problem (it exited non-zero, it reports a check as failed itself).
        summary = ("FI stopped before the main run: the known-answer checks did not pass. "
                   + " ".join(_plain(f) for f in found)[:400])
        error = last_error(stderr_tail)
        if error:
            at = error_at(stderr_tail, quest_root)
            causes.insert(0, {"text": "The script stopped with an error.", "evidence": error + (f" (at {at})" if at else "")})
    else:
        summary = (f"FI stopped before the main run: {n_failed} of {len(oracles)} known-answer check(s) did not pass"
                   + (" and FI's repair says the check itself is wrong" if kept else "")
                   + (f"; {len(checks) - n_failed} more could not be measured or judged" if len(checks) > n_failed
                      else "")
                   + ".")

    for c in checks:
        # What the gate found about a check that has no measured number, in its own words (the numbers of a failed
        # check are on the card already).
        c["found"] = [_plain(p) for p in c.get("problems") or []] if c["status"] != "failed" else []
    return {
        "type": TYPE,
        "summary": summary,
        "checks": checks,
        # What the gate found that is about no check listed above: a passing check the script reports as failed itself,
        # a non-zero exit. Nothing the record says is left off the card.
        "also_found": [_plain(f) for f in found if not any(repr(c["id"]) in f for c in checks)] if checks else [],
        "causes": causes,
        "tried": _tried(attempts, kept=kept, script=script),
        "actions": _actions(quest_id, checks, oracles, frozen=frozen, research=research, interview_made=interview_made,
                            repairs=repairs, kept=kept, quest_root=quest_root, script=script),
        "notes": _notes(frozen=frozen, research=research) + [
            # After the freeze a proposal cannot be applied from here: say the one way it can be used.
            ("This research quest cannot take the repair's proposed change (its protocol is frozen); to use it, start "
             "a new quest whose plan states it: " if research else
             "Accepting the repair's proposed change needs an amendment: ask, at the review, for this change: ")
            + _oracle.proposal_request(p) for p in proposals if frozen],
    }


def _entries(script: Path) -> set[str]:
    from .trial_runner import entries

    try:
        return entries(Path(script))
    except ValueError:  # a script that is not UTF-8 text: the card names run_cell
        return set()


def _same_each_time(attempts: list[dict[str, Any]], error: str) -> str:
    """A sentence when every attempt stopped on the same error, so the repairs did not change it."""
    errors = [last_error("; ".join(a.get("problems") or [])) for a in attempts]
    applied = sum(1 for a in attempts if a.get("repair") == "applied")
    if applied and len(attempts) >= 2 and error and all(e == error for e in errors):
        return f"; the same error came back after each of FI's {applied} repair(s) that changed the script"
    return ""


def _causes_for(check: dict[str, Any], oracle: dict[str, Any], proposal: dict[str, Any] | None,
                reported: dict[str, Any] | None, attempts: list[dict[str, Any]]) -> list[dict[str, str]]:
    name = check["id"]
    out: list[dict[str, str]] = []
    if proposal is not None:
        value = check.get("measured")
        expected, limit, _mode = _oracle.limit_of(proposal)
        verdict = ""
        if value is not None and expected is not None and limit is not None:
            verdict = (f" The script measured {_oracle.fmt_digits(value)}: accepting this makes that measurement pass, "
                       "so check the reason, not the result." if abs(value - expected) <= limit else
                       f" The script measured {_oracle.fmt_digits(value)}, which would still fail it.")
        out.append({
            "text": f"The check may be what is wrong: FI's repair judged the check `{name}` itself wrong.",
            "evidence": f"It proposes expected {_oracle.fmt_digits(proposal['expected'])}, tolerance "
                        f"{_oracle.fmt_digits(proposal['tolerance'])} ({proposal.get('tolerance_mode') or 'absolute'})"
                        + (f", check: {proposal['check']}" if proposal.get("check") else "")
                        + f". Its reason: {_short(proposal.get('reason'), 400)}.{verdict}",
        })
    if check["status"] == "not_measured":
        error = check.get("error")
        if error:
            at = check.get("error_at")
            out.append({
                "text": f"The simulation failed on the case of `{name}`, so nothing was measured: the script is what "
                        "needs fixing here, not the expected value.",
                "evidence": error + (f" (at {at})" if at else "") + _same_each_time(attempts, error),
            })
        for problem in check.get("problems") or []:
            if "must return" in problem or "did not return" in problem or "takes nothing the simulation returns" in problem:
                out.append({
                    "text": f"What `{name}` reads is not among what the simulation returns: either the simulation "
                            "must return it, or the check's `measure` names the wrong quantity.",
                    "evidence": _plain(problem),
                })
                break
    if check["status"] == "failed":
        engine = (reported or {}).get("measured_by") == "engine"
        why = _forms.mismatch(oracle, check.get("measured"),
                              str((reported or {}).get("formula_problem") or "") if engine else "", engine=engine)
        if why:
            out.append({"text": f"The plan and the simulation may mean different things by `{name}` (a unit, a "
                                "representation, or a different quantity).",
                        "evidence": why[0].upper() + why[1:] + "."})
        if not out:
            ref = check.get("reference")
            out.append({
                "text": f"Nothing FI has points to one side for `{name}`: either the simulation computes this quantity "
                        "wrongly, or the check's expected value or tolerance is wrong.",
                "evidence": ("Check the expected value against its source first: " + _short(ref, 200) + "."
                             if ref else "The check does not say where its expected value comes from."),
            })
    return out


def _plain(sentence: str) -> str:
    """A sentence of the gate's record as a person reads it: "the oracle 'x'" is the known-answer check 'x'."""
    text = re.sub(r"\bthe (?:declared )?oracle\b", "the known-answer check", sentence.strip())
    return text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")


def _tried(attempts: list[dict[str, Any]], *, kept: str | None, script: Path) -> list[str]:
    out: list[str] = []
    if any(a.get("test_run_mismatches") for a in attempts):
        out.append("Before any repair, FI asked the plan once to look at the checks' definitions (the numbers of a "
                   "test run did not fit them).")
    asked_plan = sum(1 for a in attempts[:-1] if any("declares no oracle" in p or "cannot judge it" in p
                                                       for p in a.get("problems") or []))
    if asked_plan:
        out.append(f"FI asked the plan {asked_plan} time(s) to add or complete a known-answer check.")
    n = 0
    for a in attempts:
        outcome = a.get("repair")
        if not outcome:
            continue
        n += 1
        said = _REPAIR_SAID.get(str(outcome), str(outcome))
        summary = _short(a.get("patch_summary"), 200)
        files = ", ".join(sorted(a.get("package_files") or {}))
        out.append(f"Repair {n}: {said}" + (f" ({Path(script).name}" + (f", {files}" if files else "") + ")"
                                             if outcome == "applied" else "")
                   + (f" — “{summary}”" if summary and outcome == "applied" else "") + ".")
    if kept == "as_it_was":
        out.append("The script was kept as it was: the repair says the failing checks are what is wrong, not the script.")
    elif kept == "for_disputed":
        out.append("After the repair disputed the failing check(s), the script was not changed for them.")
    if not n and not out:
        out.append("No repair was made.")
    return out


def _shell(text: str) -> str:
    return _oracle._shell_safe(text)  # noqa: SLF001 -- the one place text a person pastes is made safe


def _revise_text(checks: list[dict[str, Any]], oracles: list[dict[str, Any]]) -> str:
    """What to ask the plan for, word for word: never the measured value as the new expected one (that is the bend the
    gate exists to stop); a re-derivation from the check's own source."""
    if not oracles:
        return ("Add a known-answer check to the protocol: a special or limiting case with a known answer, a conserved "
                "quantity, a symmetry, a convergence rate, a published value or a second implementation, with a "
                "numeric expected value, a tolerance, and where the value comes from. Change nothing else.")
    proposals = [c["proposal"] for c in checks if c.get("proposal")]
    if proposals:
        return _oracle.proposal_request(proposals[0])
    parts = []
    for c in checks[:3]:
        if c["status"] == "cannot_judge":
            parts.append(f"give the known-answer check '{c['id']}' a numeric expected value and tolerance, and say "
                         "where the value comes from")
        elif c["status"] == "not_measured":
            parts.append(f"make sure the case and the measure of the known-answer check '{c['id']}' fit what the "
                         "simulation returns")
        else:
            case = _case_text(c.get("case"))
            ref = _short(c.get("reference"), 160)
            parts.append(f"re-derive the expected value of the known-answer check '{c['id']}'"
                         + (f" on its case ({case})" if case else "")
                         + (f" from its source ({ref})" if ref else "")
                         + ", and the tolerance the method can reach there; correct them only if the derivation gives "
                           "other numbers")
    text = "; ".join(parts)
    return _shell(text[0].upper() + text[1:] + ". Change nothing else.") if text else ""


def _actions(quest_id: str, checks: list[dict[str, Any]], oracles: list[dict[str, Any]], *, frozen: bool,
             research: bool, interview_made: bool, repairs: int, kept: str | None, quest_root: Path,
             script: Path) -> list[dict[str, Any]]:
    resume_cli = f"python launch.py --resume {quest_id}"
    actions: list[dict[str, Any]] = []
    nothing_measured = bool(checks) and all(c["status"] == "not_measured" for c in checks)
    # A check with no numbers (or no check at all) is the plan's to complete: the gate asks the plan, not a repair of
    # the script, and after the freeze it asks nobody (Engine._oracle_gate).
    plan_incomplete = not oracles or any(c["status"] == "cannot_judge" for c in checks)
    # 1. Resume as it is. Each resume gives the gate a fresh budget; the repair's earlier view that a check is wrong is
    # not carried into it, so say both. After the freeze an incomplete plan stops here again: no such action then.
    if plan_incomplete and not frozen:
        again = (f"FI asks the plan to {'add a known-answer check' if not oracles else 'give the check its numbers'} "
                 f"(up to {repairs} time(s)) and then checks the script with it.")
    else:
        again = (f"FI measures the checks again and, if one still does not pass, repairs the script up to {repairs} "
                 "more time(s).")
    if kept:
        again += (" The repair's earlier view that the check is wrong is not carried over, so a new repair may change "
                  "the script towards the declared value: read the reason above first.")
    if not (plan_incomplete and frozen):
        actions.append({"id": "resume", "label": "Let FI try again", "detail": "Resume as it is: " + again,
                        "cli": resume_cli, "web": "Resume", "vscode": f"@fi /resume {quest_id}"})
    # 2. Fix it yourself: the script at the line that computes the number, or the check in plan.md (before the freeze).
    where = next((c.get("where") for c in checks if c.get("where")), None)
    error_at = next((c.get("error_at") for c in checks if c.get("error_at")), "")
    target = error_at or (f"{where['file']} line {where['line']}" if where else _rel(script, quest_root))
    plan_part = "" if frozen else ", or the check in plan.md (the `oracles` list of the protocol)"
    if plan_incomplete:
        what = ("add a known-answer check" if not oracles else
                "give each check listed above a numeric expected value and tolerance")
        detail = (f"In plan.md (the `oracles` list of the protocol), {what}, with where the value comes from, then "
                  "resume." if not frozen else
                  "The protocol is frozen, so a check can be added or completed only through an amendment approved "
                  "at the review.")
    else:
        detail = (f"Change the simulation at {target}{plan_part}, then resume: the checks run again before anything "
                  "else.")
    actions.append({
        "id": "edit", "label": "Fix it yourself",
        "detail": detail,
        "cli": resume_cli, "web": "Resume", "vscode": f"@fi /resume {quest_id}",
        "file": (error_at.rsplit(" line ", 1)[0] if error_at else where["file"] if where else _rel(script, quest_root)),
        **({"line": int(error_at.rsplit(" line ", 1)[1])} if error_at and error_at.rsplit(" line ", 1)[-1].isdigit()
           else {"line": where["line"]} if where else {}),
    })
    # 3. Change the check: before the freeze, a request to the plan with the text prefilled; after it, the paths that
    # exist (an amendment at the review for a default quest; a new quest for a research one).
    if not frozen:
        text = _revise_text(checks, oracles)
        if text and not (nothing_measured and not any("must return" in p or "did not return" in p
                                                      for c in checks for p in c.get("problems") or [])):
            proposal = next((c["proposal"] for c in checks if c.get("proposal")), None)
            actions.append({
                "id": "accept_proposal" if proposal else "revise_check",
                "label": "Accept the repair's proposed change" if proposal else "Have the check worked out again",
                "detail": ("Ask the plan for the change the repair proposes (check its reason, not only that the "
                           "measured value would pass), then resume." if proposal else
                           "Ask the plan to work the check out again from its own source, then resume. FI does not "
                           "put the measured value in as the expected one."),
                "prefill": text,
                "cli": f'python launch.py --resume {quest_id} --revise-plan "{text}"',
                "web": "Ask for this change (the Plan box), then Resume",
                "vscode": f"@fi /plan {quest_id} {text}",
            })
    elif not research:
        how = ("set `engine.oracle_check: warn` in the quest's config.yaml, then approve that change: "
               f"`python launch.py --update {quest_id}` (web: the quest page's Update settings; VS Code: `@fi /update "
               f"{quest_id}`), which records it and resumes (this quest's settings were approved when it started, so "
               "a plain resume would stop again to ask)") if interview_made else (
               "set `engine.oracle_check: warn` in the quest's YAML and resume")
        actions.append({
            "id": "go_on_recorded", "label": "Go on with the failure recorded",
            "detail": f"The protocol is frozen, so editing plan.md does not change it. To go on: {how}. The failure is "
                      "recorded, the result does not count as checked against known answers, and the check can be "
                      "changed later through an amendment: ask for the change when the quest is refined at the review "
                      "and approve the amendment it asks for.",
            "cli": f"python launch.py --update {quest_id}" if interview_made else resume_cli,
            "web": "Update settings" if interview_made else "Resume",
            "vscode": f"@fi /update {quest_id}" if interview_made else f"@fi /resume {quest_id}",
        })
    return actions


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _notes(*, frozen: bool, research: bool) -> list[str]:
    if frozen and research:
        from .todo import research_instead

        return [research_instead("oracle", frozen=True)]
    if research:
        from .todo import research_instead

        return [research_instead("oracle", frozen=False)]
    if frozen:
        return ["The protocol is frozen, so editing plan.md does not change it."]
    return []
