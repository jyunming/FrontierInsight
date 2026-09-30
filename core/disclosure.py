"""What the paper says about how its result was reached: how often the design changed, and how many runs were made.

The records are kept as the quest runs: ``needs/DESIGN_HISTORY.json`` (each version of the design; every one after the
first was made after the experiment had run, ``post_hoc``) and ``.fi/attempts.jsonl`` (one ``run`` line per complete
run of the experiment, with its outcome). A reader of the finished paper cannot see either, and a study whose design was
changed after its first results looks the same as one whose design was fixed in advance, so the engine writes one
paragraph into the paper's methods from those records (:func:`paragraph`, :func:`mark_paper`). The model never writes
it: whatever the writer put between the markers is removed, and the engine's paragraph is put in after the writer has
finished.

The paragraph is built from a closed set of phrases and whole-number counts only (never a raw reason from the record),
so a reader of the paper can tell it apart from the text the model wrote, and so can the number checks
(``core/numeric_oracle.py``, ``core/stat_claims.py``, ``core/number_provenance.py``): they hold every number in the paper
to the run's results, and these counts are about the quest, not results. :func:`strip_for_checks` removes the paragraph
before they read the paper, but only when every word of it is the engine's grammar (:data:`_CONTENT`): a paragraph
between the markers that says anything else (a result-like number, a sentence the engine never writes) is left in and
checked like the rest of the paper.

:func:`unconfirmed_gap` is what the evidence ladder (``core/evidence.py``) reads: a design changed after the experiment
had run, with no confirm run on data or seeds exploration never saw after the last change (``core/phased.py``), keeps
the result below ``publication_ready``. It reads whether the confirm run happened (``.fi/phased.json``), not whether
``engine.phased`` is set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BEGIN = "<!-- fi:attempts -->"
END = "<!-- /fi:attempts -->"

LEAD = "**How this result was reached.**"

#: When the design changed, for a quest that runs an experiment (False) and one that analyses data (True). Neither says
#: "after the results were seen": a run that produced no result also sends the quest back to the design.
_AFTER = {
    False: "after the experiment had first been run",
    True: "after the data had first been analysed",
}

#: The reason a design was changed, in plain words. Only these phrases are ever written (the record's own text is not),
#: so the paragraph stays inside the grammar :func:`strip_for_checks` recognises.
REASON_ANOTHER_EXPERIMENT = "the analysis asked for another experiment"
REASON_MORE_LITERATURE = "the analysis asked for more literature"
REASON_REVIEW = "the review asked for changes"
REASON_OTHER = "a later step asked for a new design"
REASONS = (REASON_ANOTHER_EXPERIMENT, REASON_MORE_LITERATURE, REASON_REVIEW, REASON_OTHER)

_RUNS_NOTE = "(a run includes all its seeds)"
_EXPLORATORY = "Numbers from before the last design change are exploratory."

#: Run outcomes (``core/attempt_records.OUTCOMES``) that mean the run was thrown away, and how the paragraph says so. Any
#: other outcome finished; of the finished runs, every one but the last was replaced by a later run.
_FAILED = {
    "process_error": "failed or gave no result",
    "protocol_mismatch": "did not follow the protocol",
    "oracle_failure": "failed a known-answer check",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def design_history(quest_root: Path) -> list[dict[str, Any]]:
    """Every entry of ``needs/DESIGN_HISTORY.json`` (empty when there is none or it cannot be read)."""
    history = _read_json(Path(quest_root) / "needs" / "DESIGN_HISTORY.json")
    return [e for e in history if isinstance(e, dict)] if isinstance(history, list) else []


def revisions(quest_root: Path) -> list[dict[str, Any]]:
    """The design versions made after the experiment had run (``post_hoc``), oldest first."""
    return [e for e in design_history(quest_root) if e.get("post_hoc")]


def reason_phrase(reason: Any) -> str:
    """The plain phrase for a revision's recorded ``reason`` (one of :data:`REASONS`)."""
    text = str(reason or "")
    if "next_step=re_experiment" in text:
        return REASON_ANOTHER_EXPERIMENT
    if "next_step=broaden_lit" in text:
        return REASON_MORE_LITERATURE
    verdict = re.search(r"review verdict=(\w+)", text)
    if verdict and verdict.group(1).lower() in ("revise", "reject", "rejected"):
        return REASON_REVIEW
    return REASON_OTHER


def runs(quest_root: Path) -> dict[str, int]:
    """The complete runs of the experiment in ``.fi/attempts.jsonl``: ``total``, ``discarded`` and each reason for it
    (``process_error``, ``protocol_mismatch``, ``oracle_failure``, ``replaced``: a run that finished but a later one
    took its place)."""
    from . import attempt_records as _attempts

    records = [r for r in _attempts.read(Path(quest_root) / ".fi", _attempts.ATTEMPTS) if r.get("kind") == "run"]
    counts = {key: 0 for key in _FAILED}
    finished = 0
    for record in records:
        outcome = str(record.get("outcome") or "")
        if outcome in counts:
            counts[outcome] += 1
        else:
            finished += 1
    counts["replaced"] = max(finished - 1, 0)
    counts["total"] = len(records)
    counts["discarded"] = sum(counts[k] for k in _FAILED) + counts["replaced"]
    return counts


def _times(n: int) -> str:
    return "1 time" if n == 1 else f"{n} times"


def _design_sentence(history: list[dict[str, Any]], *, no_simulation: bool) -> str:
    changed = [e for e in history if e.get("post_hoc")]
    after = _AFTER[bool(no_simulation)]
    if not changed:
        return f"The design was not changed {after}."
    reasons = list(dict.fromkeys(reason_phrase(e.get("reason")) for e in changed))
    return f"The design was changed {_times(len(changed))} {after} (reasons: {'; '.join(reasons)})."


def _runs_sentence(counts: dict[str, int]) -> str:
    total = counts["total"]
    head = f"The complete experiment was run {'once' if total == 1 else f'{total} times'} {_RUNS_NOTE}"
    discarded = counts["discarded"]
    if not discarded:
        return head + "."
    items = [f"{counts[key]} {words}" for key, words in _FAILED.items() if counts[key]]
    if counts["replaced"]:
        items.append(f"{counts['replaced']} {'was' if counts['replaced'] == 1 else 'were'} replaced by a later run")
    which = "1 of these runs was" if discarded == 1 else f"{discarded} of these runs were"
    return f"{head}; {which} discarded ({', '.join(items)})."


def paragraph(quest_root: Path, *, no_simulation: bool = False) -> str:
    """The paragraph the engine writes into the paper's methods, or ``""`` when there is nothing to say (no run of the
    experiment and no change of the design: a literature survey, say). With no change of the design it still says so,
    in one short sentence: otherwise a reader cannot tell a design that was not changed from one whose changes were not
    disclosed."""
    history = design_history(quest_root)
    counts = runs(quest_root)
    changed = any(e.get("post_hoc") for e in history)
    if not changed and not counts["total"]:
        return ""
    parts = [LEAD, _design_sentence(history, no_simulation=no_simulation)]
    if counts["total"]:
        parts.append(_runs_sentence(counts))
    if changed:
        parts.append(_EXPLORATORY)
    return " ".join(parts)


# ---- the paper -----------------------------------------------------------------------------------------------------

#: Any block between the markers, whoever wrote it (with the blank lines around it), and a marker left on its own.
_ANY_BLOCK = re.compile(r"\n*[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[ \t]*\n*", re.S)
_STRAY_MARKER = re.compile(r"[ \t]*(?:" + re.escape(BEGIN) + "|" + re.escape(END) + r")[ \t]*\n?")

_METHODS = re.compile(
    r"^(#{1,6})[ \t]+(?:\d+(?:\.\d+)*\.?[ \t]+)?(?:materials[ \t]+and[ \t]+methods|methods?|methodology|"
    r"experimental[ \t]+(?:setup|design)|study[ \t]+design|approach)\b",
    re.I,
)
_HEADING = re.compile(r"^(#{1,6})[ \t]+\S")


def without_block(markdown: str) -> str:
    """``markdown`` with every block between the markers, and every marker left on its own, taken out."""
    def gap(m: re.Match[str]) -> str:
        # What was before and after the block is left one blank line apart, as two paragraphs are.
        return "" if m.start() == 0 else "\n" if m.end() == len(m.string) else "\n\n"

    text = _ANY_BLOCK.sub(gap, markdown)
    return _STRAY_MARKER.sub("", text)


def _insert_at(lines: list[str]) -> int:
    """The line the paragraph goes before: the end of the methods section (before the next heading of its level or
    higher), else just under the title, else the top (after any YAML front matter). Headings inside a fenced code block
    do not count."""
    start = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if end is not None:
            start = end + 1
    fenced = False
    methods_level = 0
    title_at = None
    for i in range(start, len(lines)):
        line = lines[i]
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        heading = _HEADING.match(line)
        if not heading:
            continue
        level = len(heading.group(1))
        if methods_level and level <= methods_level:
            return i
        if not methods_level and _METHODS.match(line):
            methods_level = level
            continue
        if title_at is None and level == 1:
            title_at = i + 1
    if methods_level:
        return len(lines)
    return title_at if title_at is not None else start


def mark_paper(markdown: str, text: str) -> str:
    """``markdown`` with the engine's paragraph ``text`` (:func:`paragraph`) in its methods, once. Whatever was between
    the markers before (an earlier pass's paragraph, or anything the writer put there) is taken out first; an empty
    ``text`` leaves no paragraph."""
    markdown = without_block(markdown)
    if not text:
        return markdown
    lines = markdown.split("\n")
    at = _insert_at(lines)
    head = lines[:at]
    while head and not head[-1].strip():
        head.pop()
    tail = lines[at:]
    while tail and not tail[0].strip():
        tail.pop(0)
    block = [BEGIN, text, END]
    return "\n".join([*head, *([""] if head else []), *block, *([""] if tail else []), *tail])


# ---- what the number checks leave out ------------------------------------------------------------------------------

_NUM = r"[1-9]\d*"
_REASON = "(?:" + "|".join(re.escape(r) for r in REASONS) + ")"
_DESIGN = (
    r"The design was (?:not changed|changed (?:1 time|" + _NUM + r" times)) (?:"
    + "|".join(re.escape(a) for a in _AFTER.values())
    + r")(?: \(reasons: " + _REASON + r"(?:; " + _REASON + r")*\))?\."
)
_ITEM = (
    r"(?:" + _NUM + r" (?:" + "|".join(re.escape(w) for w in _FAILED.values()) + r")"
    r"|(?:1 was|" + _NUM + r" were) replaced by a later run)"
)
_RUNS = (
    r"The complete experiment was run (?:once|" + _NUM + r" times) " + re.escape(_RUNS_NOTE)
    + r"(?:; (?:1 of these runs was|" + _NUM + r" of these runs were) discarded \(" + _ITEM + r"(?:, " + _ITEM
    + r")*\))?\."
)
#: Everything the engine's paragraph can say, and nothing else.
_CONTENT = re.compile(
    re.escape(LEAD) + " " + _DESIGN + "(?: " + _RUNS + ")?(?: " + re.escape(_EXPLORATORY) + ")?"
)
_BLOCK = re.compile(re.escape(BEGIN) + r"[ \t]*\n(.*?)\n[ \t]*" + re.escape(END), re.S)


def is_engine_paragraph(text: str) -> bool:
    """Whether ``text`` is something :func:`paragraph` writes (its grammar, whole)."""
    return bool(_CONTENT.fullmatch(text.strip()))


def strip_for_checks(text: str) -> str:
    """``text`` with the engine's paragraph taken out, for the checks that hold the paper's numbers to the results.
    Only a block whose every word is the engine's grammar is taken out; anything else between the markers stays and
    is checked."""
    if BEGIN not in text:
        return text
    return _BLOCK.sub(lambda m: " " if is_engine_paragraph(m.group(1)) else m.group(0), text)


# ---- the evidence ladder -------------------------------------------------------------------------------------------


def confirmed_after_last_change(quest_root: Path) -> bool:
    """Whether a confirm run (``core/phased.py``) finished, once and untouched, after the design's last change: the
    record says ``confirmed`` and the design history had no more entries when the confirm stage began than it has
    now. A record that does not say how long the history was then (written before it did) cannot show it."""
    from . import phased as _phased

    record = _phased.load(Path(quest_root))
    if _phased.status(record) != _phased.CONFIRMED:
        return False
    at_confirm = (record or {}).get("design_revisions_at_confirm")
    if not isinstance(at_confirm, int) or isinstance(at_confirm, bool):
        return False
    return len(design_history(quest_root)) <= at_confirm


def unconfirmed_gap(quest_root: Path, *, no_simulation: bool = False) -> str:
    """The ``publication_ready`` gap for a design changed after the experiment had run with no confirm run after the
    last change, or ``""``."""
    changed = revisions(quest_root)
    if not changed or confirmed_after_last_change(quest_root):
        return ""
    return (
        f"the design was changed {_times(len(changed))} {_AFTER[bool(no_simulation)]}, and no confirm run on data or "
        "seeds the earlier runs never saw came after the last change, so the numbers are exploratory (explore, then "
        "confirm: `engine.phased: true`)"
    )
