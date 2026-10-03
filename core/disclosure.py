"""What the paper says about how its result was reached: how often the design was revised, and how many runs were made.

The records are kept as the quest runs: ``needs/DESIGN_HISTORY.json`` (each version of the design; a later one is
``post_hoc``, and ``after_results`` says whether the experiment had run, or the data had been analysed, by then) and
``.fi/attempts.jsonl`` (one ``run`` line per complete run of the experiment, all its seeds together, with its outcome).
A reader of the finished paper cannot see either, and a study whose design was revised after its first results looks
the same as one whose design was fixed in advance, so the engine writes one paragraph into the paper's methods from
those records (:func:`paragraph`, :func:`mark_paper`). When the study was confirmed more than once (a version changed
after its confirmation is a new one, confirmed on its own: ``.fi/confirmations.jsonl``, ``core/confirmations.py``),
the paragraph also says how many versions were confirmed and how many of those confirmations did not hold (failed ones
included). The model never writes it: whatever the writer put between the
markers, or under the paragraph's lead words, is removed, and the engine's paragraph is put in after the writer has
finished.

The paragraph is built from a closed set of phrases and whole-number counts only (never a raw reason from the record),
so the number checks (``core/numeric_oracle.py``, ``core/stat_claims.py``, ``core/number_provenance.py``), which hold
every number in the paper to the run's results, can leave it out: these counts are about the quest, not results.
:func:`strip_for_checks` removes it only when every word of it is the engine's grammar (:data:`_CONTENT`); a block
between the markers that says anything else (a result-like number, a sentence the engine never writes) is left in and
checked like the rest of the paper.

:func:`unconfirmed_gap` is what the evidence ladder (``core/evidence.py``) reads: a design revised after the experiment
had run, with no confirm run on data or seeds exploration never saw after the last revision (``core/phased.py``),
keeps the result below ``publication_ready``. It reads whether the confirm run happened (``.fi/phased.json``), not
whether ``engine.phased`` is set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BEGIN = "<!-- fi:attempts -->"
END = "<!-- /fi:attempts -->"

LEAD = "**How this result was reached.**"

SIMULATION, DATA, SURVEY = "simulation", "data", "survey"

#: When the design was revised, by the kind of study. None says "after the results were seen": a run that produced no
#: result also sends the quest back to the design.
_AFTER = {
    SIMULATION: "after the experiment had first been run",
    DATA: "after the data had first been analysed",
    SURVEY: "after the literature had first been analysed",
}

#: The reason a design was revised, in plain words. Only these phrases are ever written (the record's own text is not),
#: so the paragraph stays inside the grammar :func:`strip_for_checks` recognises.
REASON_ANOTHER_EXPERIMENT = "the analysis asked for another experiment"
REASON_MORE_LITERATURE = "the analysis asked for more literature"
REASON_REVIEW = "the review asked for changes"
REASON_OTHER = "a later step asked for a new design"
REASONS = (REASON_ANOTHER_EXPERIMENT, REASON_MORE_LITERATURE, REASON_REVIEW, REASON_OTHER)

_RUNS_NOTE = "(a run includes all its seeds)"
_EXPLORATORY = "Numbers from before the last design revision are exploratory."

#: Run outcomes (``core/attempt_records.OUTCOMES``) of a run that was not kept, and how the paragraph says it. The last
#: run is the one kept (its result is the one analysed, even when a check only warned about it); an earlier run that
#: finished was replaced by a later one.
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


def kind(*, no_simulation: bool = False, survey: bool = False) -> str:
    """The kind of study, as the paragraph words it."""
    return SURVEY if survey else DATA if no_simulation else SIMULATION


def design_history(quest_root: Path) -> list[dict[str, Any]]:
    """Every entry of ``needs/DESIGN_HISTORY.json`` (empty when there is none or it cannot be read)."""
    history = _read_json(Path(quest_root) / "needs" / "DESIGN_HISTORY.json")
    return [e for e in history if isinstance(e, dict)] if isinstance(history, list) else []


def revisions(quest_root: Path) -> list[dict[str, Any]]:
    """The design versions made after the experiment had run (``post_hoc``), oldest first. A version recorded with
    ``after_results: false`` (a rerun of the design before anything had run) is not one; an older record without the
    field counts."""
    return [e for e in design_history(quest_root) if e.get("post_hoc") and e.get("after_results", True) is not False]


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
    """The complete runs of the experiment in ``.fi/attempts.jsonl``: ``total``, ``discarded`` (every run but the last,
    which is the one kept) and why each was (``process_error``, ``protocol_mismatch``, ``oracle_failure``,
    ``replaced``: it finished, and a later run took its place)."""
    from . import attempt_records as _attempts

    records = [r for r in _attempts.read(Path(quest_root) / ".fi", _attempts.ATTEMPTS) if r.get("kind") == "run"]
    counts = {key: 0 for key in _FAILED}
    counts["replaced"] = 0
    for record in records[:-1]:
        outcome = str(record.get("outcome") or "")
        counts[outcome if outcome in _FAILED else "replaced"] += 1
    counts["total"] = len(records)
    counts["discarded"] = max(len(records) - 1, 0)
    return counts


def _times(n: int) -> str:
    return "1 time" if n == 1 else f"{n} times"


def _design_sentence(changed: list[dict[str, Any]], study: str) -> str:
    after = _AFTER[study]
    if not changed:
        return f"The design was not revised {after}."
    reasons = list(dict.fromkeys(reason_phrase(e.get("reason")) for e in changed))
    return f"The design was revised {_times(len(changed))} {after} (reasons: {'; '.join(reasons)})."


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


#: How a study confirmed more than once (a version changed after its confirmation is a new one) says it.
_CONFIRMS_HEAD = "confirmed once on data or seeds that exploration never saw (a version changed after its " \
                 "confirmation is confirmed on its own)"
_ALL_HELD = "every confirmation held"
_LAST_UNCONFIRMED = "The current version was changed after the last confirmation and is not confirmed itself."


def _versions(n: int) -> str:
    return "1 version of the study was" if n == 1 else f"{n} versions of the study were"


def confirmations(quest_root: Path) -> dict[str, int]:
    """The versions of the study that reached a confirm run (``.fi/confirmations.jsonl``): ``versions``, ``failed``
    (whose confirmation did not hold, or was run again after its result was seen) and ``current_unconfirmed`` (1 when the current version came
    after the last confirmed one and has no confirmation of its own)."""
    from . import confirmations as _confirmations
    from . import phased as _phased

    done = _confirmations.candidates(quest_root)
    if not done:  # nothing was ever confirmed (and a quest without the two stages never reads their record)
        return {"versions": 0, "failed": 0, "current_unconfirmed": 0}
    current = _phased.candidate(_phased.load(Path(quest_root)))
    last = max((c["candidate"] for c in done), default=0)
    return {"versions": len(done), "failed": sum(1 for c in done if c["verdict"] != _confirmations.CONFIRMED),
            "current_unconfirmed": int(bool(done) and current > last)}


def _confirms_sentence(counts: dict[str, int]) -> str:
    """Said only when there is more than one confirmed version, a confirmation that did not hold, or a version changed
    after its confirmation: a study confirmed once, as planned, has the stage note under its title for that."""
    n, failed, after = counts["versions"], counts["failed"], counts["current_unconfirmed"]
    if not n or (n == 1 and not failed and not after):
        return ""
    held = _ALL_HELD if not failed else (
        f"{'1 confirmation did not hold or was' if failed == 1 else f'{failed} confirmations did not hold or were'} "
        "run again, and only the last version's own confirmation counts for the result")
    return f"{_versions(n)} {_CONFIRMS_HEAD}; {held}." + (f" {_LAST_UNCONFIRMED}" if after else "")


def paragraph(quest_root: Path, *, no_simulation: bool = False, survey: bool = False) -> str:
    """The paragraph the engine writes into the paper's methods, or ``""`` when there is nothing to say (no run of the
    experiment and no revision of the design). With no revision it still says so, in one short sentence: otherwise a
    reader cannot tell a design that was not revised from one whose revisions were not disclosed. A survey has no
    numbers of its own, so its paragraph does not call any exploratory. A study confirmed more than once (each changed
    version on its own, ``core/confirmations.py``) says how many versions were confirmed and how many did not hold."""
    study = kind(no_simulation=no_simulation, survey=survey)
    changed = revisions(quest_root)
    counts = runs(quest_root)
    if not changed and not counts["total"]:
        return ""
    parts = [LEAD, _design_sentence(changed, study)]
    if counts["total"]:
        parts.append(_runs_sentence(counts))
    if study != SURVEY and (confirms := _confirms_sentence(confirmations(quest_root))):
        parts.append(confirms)
    if changed and study != SURVEY:
        parts.append(_EXPLORATORY)
    return " ".join(parts)


# ---- the paper -----------------------------------------------------------------------------------------------------

#: Any block between the markers, whoever wrote it (with the blank lines around it), a marker left on its own, and a
#: paragraph under the engine's lead words outside the markers (the passage editor can drop a marker and keep the rest).
_ANY_BLOCK = re.compile(r"\n*[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[ \t]*\n*", re.S)
_STRAY_MARKER = re.compile(r"(?:" + re.escape(BEGIN) + "|" + re.escape(END) + r")")
_LEAD_LINE = re.compile(r"\n*^[ \t]*" + re.escape(LEAD) + r"[^\n]*(?:\n|\Z)\n*", re.M)

_SECTION_NUMBER = r"(?:(?:\d+(?:\.\d+)*|[IVXLC]+)\.?[ \t]+)?"
_METHODS = re.compile(
    r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r"(?:.*\bmethods?\b|.*\bmethodology\b|(?:simulation|experimental|numerical)"
    r"[ \t]+(?:setup|design)\b|study[ \t]+design\b|approach\b|setup\b)",
    re.I,
)
#: A heading that is the methods section by name alone ("Methods", "2. Methodology", "Materials and Methods").
_EXACT_METHODS = re.compile(
    r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r"(?:materials[ \t]+and[ \t]+)?(?:methods?|methodology)[ \t]*$", re.I)
#: A heading that names methods but is another section (a results, introduction or discussion that compares methods).
_NOT_METHODS = re.compile(
    r"\b(?:results?|findings|introduction|background|related|prior|previous|discussion|conclusions?|limitations?)\b", re.I)
_RESULTS = re.compile(r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r"(?:results|findings)\b", re.I)
_ABSTRACT = re.compile(r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r"abstract\b", re.I)
_HEADING = re.compile(r"^(#{1,6})[ \t]+\S")
_FRONT_MATTER_KEY = re.compile(r"^[A-Za-z_][\w-]*[ \t]*:")


def without_block(markdown: str) -> str:
    """``markdown`` with every block between the markers, every marker left on its own, and every paragraph under the
    engine's lead words taken out. Repeated until nothing is left to take out, so taking one piece out cannot join
    what is around it into a new marker."""
    def gap(m: re.Match[str]) -> str:
        # What was before and after the block is left one blank line apart, as two paragraphs are.
        return "" if m.start() == 0 else "\n" if m.end() == len(m.string) else "\n\n"

    text = markdown
    while True:
        before = text
        text = _ANY_BLOCK.sub(gap, text)
        text = _STRAY_MARKER.sub("", text)
        text = _LEAD_LINE.sub(gap, text)
        if text == before:
            return text


def _headings(lines: list[str], start: int) -> list[tuple[int, int, str]]:
    """``(line, level, text)`` of each heading outside a fenced code block."""
    out: list[tuple[int, int, str]] = []
    fenced = False
    for i in range(start, len(lines)):
        line = lines[i]
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        heading = None if fenced else _HEADING.match(line)
        if heading:
            out.append((i, len(heading.group(1)), line))
    return out


def _section_end(headings: list[tuple[int, int, str]], at: int, level: int, total: int) -> int:
    return next((i for i, lv, _t in headings if i > at and lv <= level), total)


def _insert_at(lines: list[str]) -> int:
    """The line the paragraph goes before: the end of the methods section (the shallowest heading that names methods,
    a methodology, a setup, a study design or an approach; to before the next heading of its level or higher), else
    before the results, else at the end of the abstract, else under the title, else the top (after any YAML front
    matter)."""
    start = 0
    if lines and lines[0].strip() == "---" and len(lines) > 1 and _FRONT_MATTER_KEY.match(lines[1]):
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if end is not None:
            start = end + 1
    headings = _headings(lines, start)
    # Never the title (the first level-1 heading), never a heading that is another section; a heading that is the
    # methods by name alone comes before one that only mentions them, then the shallowest, then the first.
    title_line = next((i for i, lv, _t in headings if lv == 1), None)
    methods = [h for h in headings
               if h[0] != title_line and _METHODS.match(h[2]) and not _NOT_METHODS.search(h[2])]
    if methods:
        at, level, _text = min(methods, key=lambda h: (not _EXACT_METHODS.match(h[2]), h[1], h[0]))
        return _section_end(headings, at, level, len(lines))
    results = next((h for h in headings if _RESULTS.match(h[2])), None)
    if results:
        return results[0]
    abstract = next((h for h in headings if _ABSTRACT.match(h[2])), None)
    if abstract:
        return _section_end(headings, abstract[0], abstract[1], len(lines))
    title = next((h for h in headings if h[1] == 1), None)
    return title[0] + 1 if title else start


def mark_paper(markdown: str, text: str) -> str:
    """``markdown`` with the engine's paragraph ``text`` (:func:`paragraph`) in its methods, once. Whatever was between
    the markers before, or under the paragraph's lead words (an earlier pass's paragraph, or anything the writer put
    there), is taken out first; an empty ``text`` leaves no paragraph."""
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
    r"The design was (?:not revised|revised (?:1 time|" + _NUM + r" times)) (?:"
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
_CONFIRMS = (
    r"(?:1 version of the study was|" + _NUM + r" versions of the study were) " + re.escape(_CONFIRMS_HEAD)
    + r"; (?:" + re.escape(_ALL_HELD) + r"|(?:1 confirmation did not hold or was|" + _NUM + r" confirmations did not hold or "
    r"were) run again, and only the last version's own confirmation counts for the result)\.(?: " + re.escape(_LAST_UNCONFIRMED) + r")?"
)
#: Everything the engine's paragraph can say, and nothing else.
_CONTENT = re.compile(
    re.escape(LEAD) + " " + _DESIGN + "(?: " + _RUNS + ")?(?: " + _CONFIRMS + ")?(?: " + re.escape(_EXPLORATORY) + ")?"
)
_BLOCK = re.compile(re.escape(BEGIN) + r"[ \t]*\r?\n(.*?)\r?\n[ \t]*" + re.escape(END), re.S)


def is_engine_paragraph(text: str) -> bool:
    """Whether ``text`` is something :func:`paragraph` writes (its grammar, whole)."""
    return bool(_CONTENT.fullmatch(text.strip()))


def strip_for_checks(text: str) -> str:
    """``text`` with the engine's paragraph taken out, for the checks that hold the paper's numbers to the results and
    for the claim check. Only a block whose every word is the engine's grammar is taken out; anything else between
    the markers stays and is checked."""
    if BEGIN not in text:
        return text
    return _BLOCK.sub(lambda m: " " if is_engine_paragraph(m.group(1)) else m.group(0), text)


# ---- the evidence ladder -------------------------------------------------------------------------------------------


def confirmed_after_last_change(quest_root: Path) -> bool:
    """Whether a confirm run (``core/phased.py``) finished, once and untouched, after the design's last revision: the
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


def unconfirmed_gap(quest_root: Path, *, no_simulation: bool = False, survey: bool = False) -> str:
    """The ``publication_ready`` gap for a design revised after the experiment had run with no confirm run shown to
    have come after the last revision, or ``""``. A survey has no numbers to confirm, so it has none. A revision that
    an approved protocol amendment made after results were seen is that amendment's gap already
    (``frozen_protocol.post_hoc``), so only the revisions beyond those amendments count here."""
    from . import frozen_protocol as _frozen

    if survey:
        return ""
    changed = len(revisions(quest_root)) - len(_frozen.post_hoc(Path(quest_root)))
    if changed <= 0 or confirmed_after_last_change(quest_root):
        return ""
    study = kind(no_simulation=no_simulation)
    how = ("a study that analyses data is confirmed only when part of one table it was given can be held back before "
           "the data is first analysed: explore, then confirm, `engine.phased: true`" if study == DATA
           else "explore, then confirm: `engine.phased: true`")
    return (
        f"the design was revised {_times(changed)} {_AFTER[study]}, and no confirm run on data or seeds the earlier "
        f"runs never saw is shown to have come after the last revision, so the numbers are exploratory ({how})"
    )
