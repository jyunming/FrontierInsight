"""The words of ``code/CHANGELOG.md``: what kind of change each entry is, and whether the results changed.

Each entry :func:`core.code_project.record_change` writes names one of four kinds of change (Keep a Changelog's
categories, in plain words): **added** (something new: the first code, a measurement a refine asked for), **changed** (a
different computation: a redesign, a round of the improve loop, a version put back), **fixed** (a repair: a crash fix,
the check against known answers repairing the script, a number a review found wrong) and **tidied** (a clean-up that does
not change what the code computes). The step that made the change decides it, in code; the model is never asked.

Each entry also has one line, "Did the results change": the checks of correctness (``core/criteria.py``) before and
after, and whether the study's results changed. Only the names of the results that changed are written, never their
values (the improve loop's rule: a result never chooses the code). When the run after a change has not finished yet, the
line says "not measured yet" and is filled in when a run of that code produces results (:func:`run_line`); a change
undone before any run says so. Nothing here is guessed."""

from __future__ import annotations

from typing import Any

CATEGORIES = ("added", "changed", "fixed", "tidied")

HEADER = ("# What changed in this code\n\n"
          "Each entry says whether it added something, changed or fixed what the code computes, or only tidied it (the "
          "same results), and whether the results changed.\n\n")

RESULTS_LEAD = "Did the results change: "
PENDING = RESULTS_LEAD + "not measured yet (filled in when a run of this code finishes with results)."

# The project files FI keeps for a person (core/code_project.py, core/code_layout.py) that FI's own runs of the study
# never read: a change to these alone is a tidy-up whose results are known without a run. (fi_search.py is not here: a
# script could import it.)
DOCUMENTATION = frozenset({"README.md", "requirements.txt", "run.py", "study.json", "METHODS.md",
                           "tests/test_oracles.py"})

_MAX_NAMES = 8


def category(value: Any, *, log: Any = None) -> str:
    """One of :data:`CATEGORIES`; anything else becomes ``changed`` (and says so in the log)."""
    word = str(value or "").strip().lower()
    if word in CATEGORIES:
        return word
    if log is not None:
        log.warning("[code] %r is not a kind of change (%s); the CHANGELOG entry says 'changed'", value,
                    ", ".join(CATEGORIES))
    return "changed"


def heading(when: str, kind: str, note: str) -> str:
    return f"## {when} - {kind.capitalize()}: {note}"


def results(text: str) -> str:
    """The results line with ``text`` after its lead."""
    return RESULTS_LEAD + text.strip()


def documentation_only(names: list[str]) -> bool:
    """Whether every file a change touched is one FI's own runs never read (:data:`DOCUMENTATION`)."""
    return bool(names) and all(n in DOCUMENTATION for n in names)


def documentation_line(names: list[str]) -> str:
    return results(f"no: only {', '.join(sorted(names))} changed, which FI's own runs of the study do not read.")


def not_measured(reason: str) -> str:
    return results(f"not measured: {reason.strip()}")


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    v = _num(value)
    return "not measured" if v is None else f"{v:.6g}"


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:_MAX_NAMES])
    return shown + (f" and {len(names) - _MAX_NAMES} more" if len(names) > _MAX_NAMES else "")


def _met(row: dict[str, Any]) -> str:
    return {True: "met", False: "not met"}.get(row.get("met"), "")


def checks_compared(before: list[dict[str, Any]] | None, after: list[dict[str, Any]]) -> tuple[str, bool]:
    """``(each check of correctness from its value before to its value after, whether any value moved)``. A value is
    compared as it is written (six significant figures); one measured on one side only is shown, not counted as moved."""
    b = {str(r.get("name")): r for r in before or [] if isinstance(r, dict) and r.get("name")}
    a = {str(r.get("name")): r for r in after or [] if isinstance(r, dict) and r.get("name")}
    parts, differs = [], False
    for name in [*a, *[n for n in b if n not in a]]:
        rb, ra = b.get(name), a.get(name)
        now = _fmt(ra.get("value")) if ra else "not in this run"
        notes = [_met(ra)] if ra else []
        if before is None or rb is None:
            text = f"{name}: {now}"
        else:
            was = _fmt(rb.get("value"))
            if was == now:
                text = f"{name}: {now}"
                notes.insert(0, "same")
            else:
                differs = differs or (_num(rb.get("value")) is not None and ra is not None
                                      and _num(ra.get("value")) is not None)
                text = f"{name}: from {was} to {now}"
        notes = [n for n in notes if n]
        parts.append(text + (f" ({', '.join(notes)})" if notes else ""))
    return ("; ".join(parts) or "none in the plan"), differs


def _full_runs_before(history: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    n = row.get("n")
    return [r for r in history if isinstance(r, dict) and not r.get("improve") and isinstance(r.get("n"), int)
            and isinstance(n, int) and r["n"] < n and isinstance(r.get("result_digest"), dict) and r["result_digest"]]


def run_line(history: list[dict[str, Any]], row: dict[str, Any]) -> str:
    """The results line for the changes a run measured: ``row`` is this run's row of ``.fi/criteria_history.jsonl``
    (with ``result_digest``), compared with the last earlier full run that produced results."""
    after_digest = row.get("result_digest") if isinstance(row.get("result_digest"), dict) else {}
    earlier = _full_runs_before(history, row)
    if not earlier:
        checks, _ = checks_compared(None, row.get("criteria") or [])
        return results(f"not compared: no earlier run's results are on record. Checks of correctness now: {checks}.")
    before = earlier[-1]
    before_digest = before["result_digest"]
    changed = sorted(k for k in set(before_digest) | set(after_digest) if before_digest.get(k) != after_digest.get(k))
    study = f"yes: the study's results changed ({_names(changed)})." if changed else "The study's results: unchanged."
    sha_b, sha_a = before.get("protocol_sha256"), row.get("protocol_sha256")
    if sha_b and sha_a and sha_b != sha_a:
        checks, _ = checks_compared(None, row.get("criteria") or [])
        return results(f"{study if changed else 'no. ' + study} The plan was amended between the two runs, so its checks "
                       f"of correctness are not compared (now: {checks}).")
    checks, moved = checks_compared(before.get("criteria") or [], row.get("criteria") or [])
    if changed:
        lead = study
    elif moved:
        lead = "the study's results did not; a check of correctness did."
    else:
        lead = "no. " + study
    return results(f"{lead} Checks of correctness: {checks}.")


def round_not_kept(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    """The results line of an improve round whose version was not kept (the next entry puts the earlier one back)."""
    checks, _ = checks_compared(before, after)
    return not_measured(f"this version was not kept, so the study's results were never computed for it. What its checks "
                        f"of correctness measured: {checks}.")


def put_back(which: str, *, ran_in_full: bool = True) -> str:
    """The results line of an entry that puts back an earlier version: one a full run used (``ran_in_full``), or an
    improve round's version whose own entry still waits for the full run after the loop."""
    if ran_in_full:
        return results(f"no: this puts back {which}, which a full run already measured.")
    return results(f"no change from {which}, which this puts back; the full run after the improve loop says whether the "
                   "study's results changed (in that version's own entry).")


def together(line: str, others: int) -> str:
    """``line`` for one of several changes a single run measured."""
    if others <= 0:
        return line
    return (line.rstrip() + f" (One run measured this change together with {others} other change"
            + ("s that were" if others > 1 else " that was") + " waiting for a run.)")
