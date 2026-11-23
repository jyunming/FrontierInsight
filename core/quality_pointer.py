"""The one plain sentence that ends the paper's body when something the reader could not otherwise see stands in it.

The paper carries no mark beside a value and no note about how FI got it. Five cases still matter to a reader: a printed value
outside the range the plan expected (FI kept it, :func:`core.result_checks.kept_bound_breaks`), a result whose check against a known answer did not pass and the study went on
(:func:`has_failed_check`) or a control of the plan that failed on the full run (:func:`has_failed_control`), a result that rests on an
equation not confirmed before the run (:func:`core.equation_tests.unconfirmed_marks`), a result from fewer runs than the plan
asked for (:func:`core.result_checks.fewer_runs_marks`) and a sentence the final claim check found unsupported that stays in
the paper (``claims`` of :data:`core.paper_facts.UNSUPPORTED_RECORD`). Only then the
paper ends its body with a single sentence that points to the quality report, and it names only the kinds that occur: no
count, no identifier, no file name. The sentence is FI's block ``{{quality_pointer}}`` (core/paper_facts.py ``BLOCK_NAMES``):
put in after every write round and replaced, never repeated, and gone when the last case is gone. QUALITY.md lists every
value and sentence it stands for.

This module builds the text and finds the place; the engine writes the file (``Engine._place_blocks``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

#: Where the engine keeps the sentence (the block's file, core/paper_facts.py ``_BLOCK_FILES``). Absent: no case.
FILE = Path(".fi") / "quality_pointer.md"

_PLACEHOLDER = re.compile(r"\{\{\s*quality_pointer\s*\}\}")
_BLOCK = re.compile(r"<!-- fi:fact-quality_pointer -->.*?<!-- /fi:fact-quality_pointer -->", re.S)
#: The sections that follow the paper's body: the reference list (and what a paper puts next to it).
_AFTER_BODY = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:references?|bibliography|works cited|further reading|acknowledg)", re.I)


def has_unsupported(quest_root: Path | str) -> bool:
    """Whether the last claim check left a sentence in the paper it found unsupported (``claims`` of the record)."""
    from . import paper_facts as _facts

    try:
        record = json.loads((Path(quest_root) / _facts.UNSUPPORTED_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    claims = record.get("claims") if isinstance(record, dict) else None
    return any(isinstance(c, dict) and str(c.get("claim") or "").strip() for c in claims or [])


def has_outside_range(quest_root: Path | str) -> bool:
    """Whether a quantity FI kept although its values break a range the plan expected is in the results (never one FI left
    out): FI's tables of every setting print every such value."""
    from . import result_checks as _checks

    try:
        return bool(_checks.kept_bound_breaks(quest_root))
    except Exception:  # noqa: BLE001 -- a record that cannot be read names no case
        return False


def has_failed_check(quest_root: Path | str) -> bool:
    """Whether a known-answer check did not pass (failed, disagreed with the plan's own working, or could not be decided) and
    the study went on (``went_on`` of ``needs/ORACLE_CHECK.json``, core/accepted_checks.py), so the results are not checked
    against that known answer."""
    from . import accepted_checks as _accepted

    try:
        record = json.loads((Path(quest_root) / "needs" / _accepted.ORACLE_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict) or record.get("status") != _accepted.WENT_ON:
        return False
    return any(isinstance(e, dict) and (e.get("name") or e.get("unmeasured")) for e in record.get("went_on") or [])


def has_failed_control(quest_root: Path | str) -> bool:
    """Whether a control of the plan (a setting whose answer is known before the run, core/controls.py) failed on the full
    run: a result checked against a known answer that did not pass, named by the same words. A control not decided is
    the quality report's only."""
    from . import controls as _controls

    try:
        return _controls.has_failed(quest_root)
    except Exception:  # noqa: BLE001 -- a record that cannot be read names no case
        return False


def has_unconfirmed_equation(quest_root: Path | str) -> bool:
    """Whether a result of the plan rests on an equation the test could not confirm before the run, and the study went on
    with it (:func:`core.equation_tests.unconfirmed_marks`; never a quantity FI left out)."""
    from . import equation_tests as _eq

    try:
        return bool(_eq.unconfirmed_marks(quest_root))
    except Exception:  # noqa: BLE001 -- a record that cannot be read names no case
        return False


def has_fewer_runs(quest_root: Path | str) -> bool:
    """Whether the run had fewer runs per setting than the plan's own minimum because the time allowed did not fit it
    (:func:`core.result_checks.fewer_runs_marks`)."""
    from . import result_checks as _checks

    try:
        return bool(_checks.fewer_runs_marks(quest_root))
    except Exception:  # noqa: BLE001 -- a record that cannot be read names no case
        return False


def sentence(outside_range: bool, unsupported: bool, equation: bool = False, fewer_runs: bool = False,
             failed_check: bool = False) -> str:
    """The closing sentence for the kinds that occur, in this order: values outside the expected range, results whose check against a known answer did not pass, results resting on an
    equation not confirmed before the run, results from fewer runs than planned, statements not supported. One kind is named
    plainly; several are a short list joined with commas and "and". ``""`` when none occurs."""
    kinds = [text for on, text in (
        (outside_range, "values that fell outside the range the study expected"),
        (failed_check, "results whose check against a known answer did not pass"),
        (equation, "results that rest on an equation not confirmed before the run"),
        (fewer_runs, "results from fewer runs than planned"),
        (unsupported, "statements that the study's results and the cited sources do not support")) if on]
    if not kinds:
        return ""
    what = kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + ", and " + kinds[-1] + ","
    return what[0].upper() + what[1:] + " are listed in the accompanying quality report."


def text(quest_root: Path | str) -> str:
    """The sentence FI's records call for now, ``""`` with no case."""
    return sentence(has_outside_range(quest_root), has_unsupported(quest_root),
                    has_unconfirmed_equation(quest_root), has_fewer_runs(quest_root),
                    has_failed_check(quest_root) or has_failed_control(quest_root))


def in_paper(markdown: str) -> str:
    """The sentence the paper carries now (the text of the filled block), ``""`` when it carries none."""
    m = re.search(r"<!-- fi:fact-quality_pointer -->(.*?)<!-- /fi:fact-quality_pointer -->", markdown, re.S)
    return " ".join(m.group(1).split()) if m else ""


def with_block(markdown: str) -> str:
    """``markdown`` with the placeholder ``{{quality_pointer}}`` on a line of its own at the end of the paper's body: right
    before the reference list (and what is put beside it), else at the end of the paper. Nothing is changed when the draft
    holds the block already, as a placeholder or filled in."""
    if _PLACEHOLDER.search(markdown) or _BLOCK.search(markdown):
        return markdown
    from . import best_design_report as _best

    lines = markdown.split("\n")
    level = _best.heading_level(markdown)
    hit = next((i for i, lv, t in _best._headings(lines) if lv <= level and _AFTER_BODY.match(t)), None)
    at = len(lines) if hit is None else _best._before_rule(lines, hit)
    return _best._insert_at(lines, at, ["{{quality_pointer}}"])
