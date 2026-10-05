"""Going on with checks whose expected value has no stated source: the person's choice, recorded.

Under ``rigor_profile: research`` a quest stops at the plan when a check against a known answer (an oracle) does not say
where the value it expects comes from (``core/oracle_check.py::source_gaps``). Filling that in is the way on FI recommends,
and FI asks the model once itself before it stops. Some expected values need no source a person has to hunt for, and some
people know the value is right and have no time to write why. So the stop offers a third way: go on as it is.

That choice is an act of its own, like approving a change to a frozen protocol (``core/frozen_protocol.py``): it needs a
name (``--accept-checks <quest> --approve-as <you>``, the web page's *Go on as it is* button, ``@fi /accept-checks``),
and it covers only the checks the stop named. Nothing about the check itself is relaxed: each still runs and is still
judged; the evidence record keeps the gap below *independently validated* (marked "source not confirmed", with who chose
to go on), the audit trace records the choice, and the paper is told to say so plainly.

Two records under ``needs/``: ``UNSOURCED_CHECKS.json`` (which checks, why, the plan version) and
``UNSOURCED_CHECKS_ACCEPTED.json`` (who went on with them).

Where a check's expected value comes from is not something a person is asked to work out: when FI's own ask of a model
(core/engine.py ``_fill_plan_sources``) finds no source, the quest goes on by itself (:func:`go_on_by_itself`),
recorded as :data:`AUTOMATIC`, with each check marked "source not confirmed" exactly as a person's choice marks it. A
person can still make the choice under their own name (``--accept-checks``) for a stop an older FI wrote.

The same choice, with the same command, exists for a check that WAS measured and failed (the stop at the known-answer
checks, ``Engine._oracle_gate``): *mark it unconfirmed and go on*. It is offered only when every problem the stop found is
a check with a measured number outside its tolerance (:func:`offer`); a check that measured nothing (the script crashed,
printed nothing) has no failure to record, and the card says so. The choice binds to the check's conditions (its
:func:`fingerprint`) AND to the version of the code that computed the number (:func:`script_version`: the script the
check ran, the helper modules beside it and the model's package in ``code/``); when either changes, the choice no longer applies and the check is
judged again. ``needs/ORACLE_CHECK.json`` holds what the stop offered (``go_on``) and, once the quest went on, what it
went on with (status ``went_on_failing``, ``went_on``); ``needs/FAILED_CHECKS_ACCEPTED.json`` holds the person's choice.
An exploration quest (not ``rigor_profile: research``, ``result_use`` explore) goes on by itself in the same way, recorded
as automatic, never as a person's choice. Either way the evidence keeps the result below *independently validated*
(:func:`gap`), so never publication-ready, and the paper says so (:func:`failing_disclosure`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PENDING_NAME = "UNSOURCED_CHECKS.json"
ACCEPTED_NAME = "UNSOURCED_CHECKS_ACCEPTED.json"
#: The words a person reads for a check they went on with.
NOT_CONFIRMED = "source not confirmed"
#: Who went on, when nobody chose it: FI went on by itself (an exploration quest with a failed check, or checks whose
#: source FI could not find).
AUTOMATIC = "automatic"


def _needs(quest_root: Path) -> Path:
    return Path(quest_root) / "needs"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write(path: Path, record: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(name: Any) -> str:
    return " ".join(str(name or "").split()).lower()


def fingerprint(oracle: dict[str, Any]) -> str:
    """What a choice to go on binds to: the check's name and the numbers it is judged by (``expected``, ``tolerance``,
    ``tolerance_mode``, ``case``, ``measure``). A check changed after the choice is a check nobody chose to go on with."""
    import hashlib

    fields = {"name": _key(oracle.get("name")),
              **{k: oracle.get(k) for k in ("expected", "tolerance", "tolerance_mode", "case", "measure")}}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def pending(quest_root: Path) -> dict[str, Any] | None:
    """What the stop named (``checks``: ``[{name, why, expected, fingerprint}]``, ``plan_version``), or ``None`` when the
    quest is not stopped for it."""
    record = _read(_needs(quest_root) / PENDING_NAME)
    return record if record and isinstance(record.get("checks"), list) and record["checks"] else None


def write_pending(quest_root: Path, checks: list[dict[str, Any]], *, plan_version: int | None,
                  model_missing: list[str] | None = None) -> dict[str, Any]:
    """Record what the stop names, and return the record with ``previous``: what the stop before this one named and at
    which plan version (``None`` the first time), so the stop can say what changed since."""
    earlier = pending(quest_root)
    record = {
        "checks": [{"name": str(c.get("name") or ""), "why": str(c.get("why") or ""), "expected": c.get("expected"),
                    "fingerprint": str(c.get("fingerprint") or "")}
                   for c in checks],
        "model_missing": list(model_missing or []),
        "plan_version": plan_version,
        "at": _now(),
        "previous": ({"checks": earlier.get("checks"), "plan_version": earlier.get("plan_version")}
                     if earlier else None),
    }
    _write(_needs(quest_root) / PENDING_NAME, record)
    return record


def clear_pending(quest_root: Path) -> None:
    try:
        (_needs(quest_root) / PENDING_NAME).unlink(missing_ok=True)
    except OSError:
        pass


def accepted(quest_root: Path) -> dict[str, Any] | None:
    """The person's choice to go on (``by``, ``via``, ``at``, ``checks``: the names), or ``None``."""
    record = _read(_needs(quest_root) / ACCEPTED_NAME)
    return record if record and record.get("by") and isinstance(record.get("checks"), list) else None


def accept(quest_root: Path, who: str, *, via: str) -> tuple[bool, str]:
    """Record that ``who`` goes on with the checks the stop named, as they are. ``(ok, what to tell the person)``.

    One command for both stops it serves: the quest stopped at its known-answer checks (a check measured and failed:
    :func:`accept_failing`), or at the plan for checks with no stated source. Which one is read from the live stop
    (``.fi/pause.json``), so an older record of the other stop never takes the choice."""
    who = " ".join(str(who or "").split())
    if not who:
        return False, ("say who is choosing to go on (--approve-as <you>): going on with a check that failed, or with "
                       "checks whose expected value has no stated source, is a choice a person makes, and it is "
                       "recorded with their name")
    paused = _paused_kind(quest_root)
    if paused == "oracle" or (paused is None and pending(quest_root) is None and _stopped_record(quest_root)):
        return accept_failing(quest_root, who, via=via)
    record = pending(quest_root)
    if record is None:
        return False, ("this quest is not stopped for checks without a stated source (needs/UNSOURCED_CHECKS.json is "
                       "not there): nothing to go on with")
    named = [c for c in record["checks"] if str(c.get("name") or "").strip() and c.get("fingerprint")]
    if not named:
        return False, ("the stop did not name its checks, so there is nothing to go on with yet: resume the quest, and "
                       "the stop names them")
    earlier = accepted(quest_root)
    # Each choice binds to the check as the stop showed it (its name and numbers), and is kept per check.
    chosen = dict((earlier or {}).get("chosen") or {})
    for c in named:
        chosen[_key(c["name"])] = {"name": c["name"], "fingerprint": c["fingerprint"], "by": who, "via": via,
                                   "at": _now(), "why": c.get("why"), "expected": c.get("expected")}
    names = [str(c["name"]) for c in named]
    entry = {
        "by": who, "via": via, "at": _now(), "checks": [v["name"] for v in chosen.values()], "chosen": chosen,
        "plan_version": record.get("plan_version"),
        "history": [*((earlier or {}).get("history") or []),
                    {"by": who, "via": via, "at": _now(), "checks": names}],
    }
    if not _write(_needs(quest_root) / ACCEPTED_NAME, entry):
        return False, "the choice could not be written to needs/ (check the folder can be written to)"
    listed = ", ".join(repr(n) for n in names)
    return True, (f"recorded: {who} goes on with {len(names)} check(s) whose expected value has no stated source "
                  f"({listed}). They still run and are still judged; each is marked “{NOT_CONFIRMED}”, and the result "
                  "and the paper say so. A check whose numbers change after this stops the quest again. Resume the "
                  "quest to go on.")


def go_on_by_itself(quest_root: Path, checks: list[dict[str, Any]], *,
                    plan_version: int | None) -> dict[str, Any] | None:
    """Record that FI goes on by itself with ``checks`` (``[{name, why, expected, fingerprint}]``: the checks whose
    expected value still has no source after FI asked a model for one), each marked "source not confirmed". Returns
    the record (as :func:`accepted` reads it), or ``None`` when there is nothing to record. A check a person already
    chose to go on with, as it is now, keeps the person's name; nothing else about any check changes."""
    named = [c for c in checks if str(c.get("name") or "").strip() and c.get("fingerprint")]
    if not named:
        return None
    write_pending(quest_root, named, plan_version=plan_version)
    earlier = accepted(quest_root)
    chosen = dict((earlier or {}).get("chosen") or {})
    for c in named:
        entry = chosen.get(_key(c["name"]))
        if isinstance(entry, dict) and entry.get("fingerprint") == c["fingerprint"]:
            continue  # a person's choice about this check, as it is now, stays theirs
        chosen[_key(c["name"])] = {"name": c["name"], "fingerprint": c["fingerprint"], "by": AUTOMATIC,
                                   "via": AUTOMATIC, "at": _now(), "why": c.get("why"), "expected": c.get("expected")}
    names = [str(c["name"]) for c in named]
    entry = {
        "by": (earlier or {}).get("by") or AUTOMATIC, "via": (earlier or {}).get("via") or AUTOMATIC, "at": _now(),
        "checks": [v["name"] for v in chosen.values()], "chosen": chosen, "plan_version": plan_version,
        "history": [*((earlier or {}).get("history") or []),
                    {"by": AUTOMATIC, "via": AUTOMATIC, "at": _now(), "checks": names}],
    }
    if not _write(_needs(quest_root) / ACCEPTED_NAME, entry):
        return None
    clear_pending(quest_root)
    return {**entry, "by": AUTOMATIC}


def who_text(by: Any) -> str:
    """Who went on with a check, as a sentence's subject: a person's name, or FI when it went on by itself."""
    return "FI (no source could be found)" if str(by or "") == AUTOMATIC else str(by or "")


def chose(quest_root: Path, oracle: dict[str, Any]) -> str | None:
    """Who chose to go on with ``oracle`` exactly as it is now (its name and numbers), or ``None``."""
    record = accepted(quest_root)
    entry = ((record or {}).get("chosen") or {}).get(_key(oracle.get("name")))
    return str(entry.get("by")) if isinstance(entry, dict) and entry.get("fingerprint") == fingerprint(oracle) else None


def changed_since(quest_root: Path, oracle: dict[str, Any]) -> bool:
    """Whether someone chose to go on with a check of this name whose numbers have changed since."""
    record = accepted(quest_root)
    entry = ((record or {}).get("chosen") or {}).get(_key(oracle.get("name")))
    return isinstance(entry, dict) and entry.get("fingerprint") != fingerprint(oracle)


def covers(quest_root: Path, oracles: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The choice to go on when it covers every check in ``oracles`` as they are now, else ``None``."""
    record = accepted(quest_root)
    if record is None or not oracles:
        return None
    return record if all(chose(quest_root, o) for o in oracles) else None


def disclosure(quest_root: Path, names: list[str]) -> str:
    """What the paper must say about ``names`` (the checks gone on with that still have no stated source), or ``""``."""
    record = accepted(quest_root)
    if record is None or not names:
        return ""
    chosen = record.get("chosen") or {}
    who = sorted({who_text((chosen.get(_key(n)) or {}).get("by") or record["by"]) for n in names})
    listed = ", ".join(repr(str(n)) for n in names)
    return (
        f"The expected values of these checks against known answers have no stated source: {listed}. "
        f"{' and '.join(who)} went on without one. Say so plainly where the checks are described (and in the "
        "limitations): each check shows the code agrees with the value the plan expected, not that the value itself is "
        "right. Do not describe these checks as validated against an independent source."
    )


# --- A check that was measured and failed: mark it unconfirmed and go on -------------------------------------------

ORACLE_RECORD = "ORACLE_CHECK.json"
FAILED_ACCEPTED_NAME = "FAILED_CHECKS_ACCEPTED.json"
#: The status of ``needs/ORACLE_CHECK.json`` when the quest went on although a check failed.
WENT_ON = "went_on_failing"
#: The word a person reads for a failed check the quest went on with.
UNCONFIRMED = "unconfirmed"


def _paused_kind(quest_root: Path) -> str | None:
    record = _read(Path(quest_root) / ".fi" / "pause.json")
    return str(record.get("kind")) if record and record.get("kind") else None


def _stopped_record(quest_root: Path) -> dict[str, Any] | None:
    record = _read(_needs(quest_root) / ORACLE_RECORD)
    return record if record and record.get("status") == "stopped" else None


def script_version(files: dict[str, str]) -> str:
    """The version of the code that computed a check's number: one hash of each file's path and text (the script the
    check ran, the helper modules beside it and the model's package in ``code/``). Any change to any of them is a new version."""
    import hashlib

    body = json.dumps(sorted((str(k), str(v)) for k, v in (files or {}).items()), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def short_version(version: Any) -> str:
    return str(version or "")[:12]


def offer(found: list[str], oracles: list[dict[str, Any]],
          judged: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str]:
    """Whether the stop may offer "mark it unconfirmed and go on": ``(checks, "")`` when every problem in ``found`` is a
    check that was measured and is outside its tolerance (each check: its name, conditions, fingerprint and the measured
    value), else ``([], why not)``, in words a person reads on the card. The one rule the engine and the card share."""
    from . import oracle_check as _oracle

    if not oracles:
        return [], "the plan has no known-answer check, so there is no failed check to go on with"
    if _oracle.unjudgeable(oracles):
        return [], "a check gives no number to compare with, so nothing was judged and there is no failure to record"
    duplicate = _oracle.duplicate_names(oracles)
    if duplicate:
        return [], f"{duplicate[0]}, so a failure cannot be told apart from the other check's"
    by_name = {str(j.get("name") or "").strip().lower(): j for j in judged or [] if isinstance(j, dict)}
    failing: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    for oracle in oracles:
        name = str(oracle.get("name") or "").strip()
        j = by_name.get(name.lower()) or {}
        value = _oracle._num(j.get("value"))  # noqa: SLF001 -- the one finite-number test the checks use
        if value is None:
            unmeasured.append(name)
        elif j.get("passed_by_engine") is False:
            expected, limit, mode = _oracle.limit_of(oracle)
            failing.append({
                "name": name, "fingerprint": fingerprint(oracle), "expected": expected, "limit": limit,
                "tolerance": oracle.get("tolerance"), "tolerance_mode": mode, "case": oracle.get("case"),
                "measure": oracle.get("measure"), "measured": value, "measured_by": j.get("measured_by") or "",
            })
    if unmeasured:
        return [], (f"nothing was measured for {', '.join(repr(n) for n in unmeasured)} (the run stopped before it "
                    "reported a number, or reported no number), so there is no failure to record: the script has to "
                    "be fixed first")
    names = [repr(c["name"]) for c in failing]
    other = [f for f in found or [] if not any(n in f for n in names)]
    if other:
        return [], f"the run has a problem besides the failed checks ({other[0][:200]}), which going on would not record"
    if not failing:
        return [], "no check failed with a measured number"
    return failing, ""


def went_on_entries(found: list[str], oracles: list[dict[str, Any]],
                    judged: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Every check the gate could not confirm, for going on by itself when :func:`offer` offers nothing (a check that
    measured nothing, a check with no number, no check at all): the measured failures as :func:`offer` gives them, and
    each other one with ``measured`` ``None`` and ``unmeasured``: why, in the gate's words. Never empty when ``found``
    is not: a run with no check to name gets one entry with no name."""
    from . import oracle_check as _oracle

    by_name = {str(j.get("name") or "").strip().lower(): j for j in judged or [] if isinstance(j, dict)}
    out: list[dict[str, Any]] = []
    for oracle in oracles or []:
        name = str(oracle.get("name") or "").strip()
        j = by_name.get(name.lower()) or {}
        if j.get("passed_by_engine") is True:
            continue
        expected, limit, mode = _oracle.limit_of(oracle)
        value = _oracle._num(j.get("value"))  # noqa: SLF001
        said = next((f for f in found or [] if repr(name) in f or f"'{name}'" in f), "")
        if value is None and not said and j:
            said = "it reported no number"
        if value is None and not said:
            continue
        out.append({"name": name, "fingerprint": fingerprint(oracle), "expected": expected, "limit": limit,
                    "tolerance": oracle.get("tolerance"), "tolerance_mode": mode, "case": oracle.get("case"),
                    "measure": oracle.get("measure"), "measured": value, "measured_by": j.get("measured_by") or "",
                    **({"unmeasured": said[:300]} if value is None else {})})
    if not out and found:
        out.append({"name": "", "measured": None, "unmeasured": str(found[0])[:300]})
    return out


def conditions_text(check: dict[str, Any]) -> str:
    """A failed check's conditions in one line: expected ± limit, case, measure, and what was measured."""
    from . import oracle_check as _oracle

    expected, measured, limit = check.get("expected"), check.get("measured"), check.get("limit")
    try:
        shown_measured, shown_expected = _oracle.fmt_pair(measured, expected, limit)
    except (OverflowError, ValueError, TypeError):
        shown_measured, shown_expected = str(measured), str(expected)
    case = check.get("case")
    case_text = ", ".join(f"{k}={v}" for k, v in case.items()) if isinstance(case, dict) and case else ""
    parts = [f"expected {shown_expected} within ±{_oracle.fmt_digits(limit)} ({check.get('tolerance_mode') or 'absolute'})"]
    if case_text:
        parts.append(f"case {case_text}")
    if check.get("measure"):
        parts.append(f"measure `{check['measure']}`")
    parts.append(f"measured {shown_measured}")
    return "; ".join(parts)


def failing_pending(quest_root: Path) -> dict[str, Any] | None:
    """What the stop at the known-answer checks offers to go on with (``needs/ORACLE_CHECK.json``'s ``go_on``: ``checks``,
    ``script``, ``script_version``), or ``None`` when the quest is not stopped there or the stop offers nothing."""
    # The same rule `accept` follows: only while the live stop is this one (an older stopped record must not offer a
    # choice the endpoint would then record against the plan's stop).
    paused = _paused_kind(quest_root)
    if paused not in ("oracle", None) or (paused is None and pending(quest_root) is not None):
        return None
    record = _stopped_record(quest_root)
    go_on = (record or {}).get("go_on")
    if isinstance(go_on, dict) and go_on.get("offered") and isinstance(go_on.get("checks"), list) and go_on["checks"]:
        return go_on
    return None


def failing_accepted(quest_root: Path) -> dict[str, Any] | None:
    """The person's choices to go on with a failed check (``chosen``: per check), or ``None``."""
    record = _read(_needs(quest_root) / FAILED_ACCEPTED_NAME)
    return record if record and isinstance(record.get("chosen"), dict) else None


def accept_failing(quest_root: Path, who: str, *, via: str) -> tuple[bool, str]:
    """Record that ``who`` goes on although the checks the stop names failed: each marked unconfirmed, bound to its
    conditions and to the version of the code that measured it. ``(ok, what to tell the person)``."""
    record = _stopped_record(quest_root)
    if record is None:
        return False, ("this quest is not stopped at a known-answer check that failed (needs/ORACLE_CHECK.json does not "
                       "say so): nothing to go on with")
    go_on = record.get("go_on") if isinstance(record.get("go_on"), dict) else {}
    if not go_on.get("offered") or not go_on.get("checks"):
        why = str(go_on.get("why_not") or "the stop did not say which checks failed; resume the quest and it says")
        return False, f"going on with the check marked {UNCONFIRMED} is not offered at this stop: {why}"
    earlier = failing_accepted(quest_root)
    chosen = dict((earlier or {}).get("chosen") or {})
    at = _now()
    for c in go_on["checks"]:
        chosen[_key(c.get("name"))] = {
            **{k: c.get(k) for k in ("name", "fingerprint", "expected", "limit", "tolerance", "tolerance_mode", "case",
                                     "measure", "measured")},
            "script": go_on.get("script"), "script_version": go_on.get("script_version"),
            "by": who, "via": via, "at": at,
        }
    names = [str(c.get("name")) for c in go_on["checks"]]
    entry = {"chosen": chosen,
             "history": [*((earlier or {}).get("history") or []), {"by": who, "via": via, "at": at, "checks": names}]}
    if not _write(_needs(quest_root) / FAILED_ACCEPTED_NAME, entry):
        return False, "the choice could not be written to needs/ (check the folder can be written to)"
    listed = "; ".join(f"'{c.get('name')}' ({conditions_text(c)})" for c in go_on["checks"])
    return True, (f"recorded: {who} goes on although {len(names)} known-answer check(s) failed: {listed}, measured by "
                  f"{go_on.get('script') or 'the script'} (version {short_version(go_on.get('script_version'))}). Each "
                  f"is marked {UNCONFIRMED}: the result does not count as checked against known answers, and the paper "
                  "says so. If the check's expected value, tolerance, case or measure changes, or that code changes "
                  "(by you since the stop, or by a later fix FI makes), the check is judged again and FI asks again. "
                  "Resume the quest to go on.")


def went_on_by(quest_root: Path, check: dict[str, Any], version: str) -> dict[str, Any] | None:
    """The person's choice that covers ``check`` (an entry of :func:`offer`) as it is now, measured by the code at
    ``version``; ``None`` when nobody chose it, or its conditions or the code changed since."""
    entry = ((failing_accepted(quest_root) or {}).get("chosen") or {}).get(_key(check.get("name")))
    if (isinstance(entry, dict) and entry.get("fingerprint") == check.get("fingerprint")
            and entry.get("script_version") == version):
        return entry
    return None


def no_longer_applies(quest_root: Path, check: dict[str, Any], version: str) -> str:
    """Why a choice to go on with ``check`` no longer applies ("" when there is none, or it still applies)."""
    entry = ((failing_accepted(quest_root) or {}).get("chosen") or {}).get(_key(check.get("name")))
    if not isinstance(entry, dict):
        return ""
    if entry.get("fingerprint") != check.get("fingerprint"):
        return "its expected value, tolerance, case or measure changed since"
    if entry.get("script_version") != version:
        return "the code that measures it changed since"
    return ""


def gap(entry: dict[str, Any]) -> str:
    """The evidence's sentence for one failed check the quest went on with."""
    name = str(entry.get("name") or "")
    if entry.get("measured") is None and "unmeasured" in entry:
        if not name:
            return f"FI went on by itself although no known-answer check could be judged ({entry.get('unmeasured')})"
        return (f"FI went on by itself although the known-answer check '{name}' could not be judged "
                f"({entry.get('unmeasured')}); it is marked {UNCONFIRMED}")
    how = conditions_text(entry)
    if entry.get("by") == AUTOMATIC:
        return (f"FI went on by itself although the known-answer check '{name}' failed ({how}); "
                f"it is marked {UNCONFIRMED}")
    return f"{entry.get('by')} chose to go on although the known-answer check '{name}' failed ({how})"


def failing_disclosure(record: Any) -> str:
    """What the paper must say about the failed checks the quest went on with (``needs/ORACLE_CHECK.json``), or ``""``."""
    if not isinstance(record, dict) or record.get("status") != WENT_ON:
        return ""
    entries = [e for e in record.get("went_on") or [] if isinstance(e, dict) and (e.get("name") or e.get("unmeasured"))]
    if not entries:
        return ""
    lines = "; ".join(gap(e) for e in entries)
    return (f"These known-answer checks FAILED and the run went on anyway: {lines}. Say so plainly where the checks are "
            "described and in the limitations: the result is not validated against these known answers, and nothing "
            "here may be described as checked against them.")
