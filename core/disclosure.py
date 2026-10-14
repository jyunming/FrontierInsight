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

The same module holds the second engine-written block, the paper's open code findings (:func:`mark_findings`, its text
built by ``core/code_review.py``: what a second model's reading of the code left unresolved, put into the limitations
after the writer has finished, so the paper states it whatever the writer did with the prompt's note). It is between
``<!-- fi:code-findings:<checksum> -->`` markers. Its words are the findings' own texts, which carry numbers (a
tolerance, a size, a count), so the grammar check above cannot recognise it. Instead the opening marker carries a
checksum of the block's text, and :func:`strip_for_checks` leaves the block out of the number checks only while the text
still matches it: a block that was edited, or one a writer typed with a guessed checksum, is checked like the rest of
the paper. (The engine removes every block between these markers from a draft before it puts its own in, so a writer's
block never survives to the checks in the first place.)
"""

from __future__ import annotations

import hashlib
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


# ---- the run record ------------------------------------------------------------------------------------------------

_RECORD_LEAD = "**What FI recorded about this run.**"
_SHOWN = 3  # sentences of one record listed in the paper; the rest are counted
_LIMIT_PHRASE = re.compile(
    r"\b(?:under|within|at most|less than|no more than|below)\s+(?:about\s+|roughly\s+|approximately\s+)?"
    r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|s|h)\b", re.I)
#: A word that says the limit is about how long the run takes (not about, say, a time inside the study).
_RUN_WORD = re.compile(r"\b(?:run[ \t-]?times?|runtimes?|runs?|running|comput\w*|execut\w*|wall[ \t-]?clock)\b", re.I)
_UNIT_SECONDS = (("h", 3600.0), ("m", 60.0), ("s", 1.0))


def _whole(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _plain_seconds(seconds: float) -> str:
    """A duration the way the paper says it: ``73.6 s``, ``60 s``, ``1,200 s``."""
    return f"{seconds:,.0f} s" if float(seconds).is_integer() else f"{seconds:,.1f} s"


def _limits_in(text: str) -> set[float]:
    """The distinct run-time limits (in seconds) that ``text`` states: a phrase such as "under 60 seconds" with a word
    about the run itself ("runtime", "runs", "computing") within a few words of it."""
    found: set[float] = set()
    for m in _LIMIT_PHRASE.finditer(text):
        around = text[max(0, m.start() - 60):m.end() + 60]
        if not _RUN_WORD.search(around):
            continue
        scale = next(s for u, s in _UNIT_SECONDS if m.group(2).lower().startswith(u))
        found.add(round(float(m.group(1)) * scale, 6))
    return found


def _topic_limit(constraints: list[str], topic: str) -> float | None:
    """The one run-time limit the topic states (in seconds), or ``None`` when it states none or several different ones.
    The topic's constraints are read first; its text only when they state none."""
    for source in ("\n".join(constraints), topic):
        limits = _limits_in(source)
        if limits:
            return next(iter(limits)) if len(limits) == 1 else None
    return None


def _items(value: Any) -> list[str]:
    """FI's sentences from a record's list (plain sentences, or objects that hold one under ``sentence``)."""
    out: list[str] = []
    for item in value if isinstance(value, list) else []:
        text = item.get("sentence") if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip() and text.strip() not in out:
            out.append(text.strip())
    return out


def _listed(lead: str, sentences: list[str]) -> str:
    """``lead`` and the first few of ``sentences`` (each a sentence of its own), then how many more there are."""
    shown = [s[:1].upper() + s[1:] + ("" if s.endswith((".", "!", "?")) else ".") for s in sentences[:_SHOWN]]
    more = len(sentences) - _SHOWN
    return " ".join([lead, *shown, *([f"And {more} more."] if more > 0 else [])])


def _ledger_rows(quest_root: Path) -> tuple[dict[str, int], dict[tuple[str, int], tuple[str, int | None]]] | None:
    """What FI's trial ledger holds: the trials planned per setting, and each trial's ``(status, seed)`` (a later line
    for the same trial replaces an earlier one). ``None`` when there is no readable ledger or it holds no trial."""
    try:
        lines = (Path(quest_root) / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    planned: dict[str, int] = {}
    trials: dict[tuple[str, int], tuple[str, int | None]] = {}
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("cell"), str):
            continue
        if row.get("event") == "planned" and (n := _whole(row.get("trials"))) is not None:
            planned[row["cell"]] = n
        elif row.get("event") == "trial" and (t := _whole(row.get("trial"))) is not None:
            trials[(row["cell"], t)] = (str(row.get("status")), _whole(row.get("seed")))
    return (planned, trials) if trials else None


def recorded_seeds(quest_root: Path) -> tuple[int | None, frozenset[int]]:
    """What the run's own records say its seeds were: the base seed in the study record (``code/study.json``) and
    every trial seed in FI's ledger (``raw/ledger.jsonl``); ``None`` and an empty set where a record is missing."""
    study = _read_json(Path(quest_root) / "code" / "study.json")
    base = _whole(study.get("base_seed")) if isinstance(study, dict) else None
    rows = _ledger_rows(quest_root)
    trial = frozenset(seed for _status, seed in rows[1].values() if seed is not None) if rows else frozenset()
    return base, trial


def _seeds_paired(trials: dict[tuple[str, int], tuple[str, int | None]]) -> bool | None:
    """Whether the ledger shows every setting using the same seed for the same trial number (``True``), or no two
    settings sharing one (``False``); ``None`` when it shows neither (one setting only, or a mixture)."""
    by_trial: dict[int, list[int]] = {}
    for (_cell, t), (_status, seed) in trials.items():
        if seed is not None:
            by_trial.setdefault(t, []).append(seed)
    shared = [seeds for seeds in by_trial.values() if len(seeds) > 1]
    if not shared:
        return None
    if all(len(set(seeds)) == 1 for seeds in shared):
        return True
    if all(len(set(seeds)) == len(seeds) for seeds in shared):
        return False
    return None


def _seed_sentence(quest_root: Path, rows: Any) -> str:
    study = _read_json(Path(quest_root) / "code" / "study.json")
    base = _whole(study.get("base_seed")) if isinstance(study, dict) else None
    if base is None:
        return ""
    stated = study.get("paired") if isinstance(study.get("paired"), bool) else None
    shown = _seeds_paired(rows[1]) if rows else None
    if shown is not None and shown != stated:
        # The ledger is what happened; the study's own setting does not override it.
        if shown:
            return (f"Each trial's random seed is derived from base seed {base}; the run's ledger shows that every "
                    "setting used the same seeds for the same trial number (common random numbers), so the settings' "
                    "results are paired, not independent.")
        return (f"Each trial's random seed is derived from base seed {base}; the run's ledger shows that different "
                "settings used different seeds for the same trial number.")
    if stated is True:
        return (f"Each trial's random seed is derived from base seed {base} and the trial number; every setting uses "
                "the same seeds for the same trial number (common random numbers), so the settings' results are "
                "paired, not independent.")
    if stated is False:
        return (f"Each trial's random seed is derived from base seed {base}, the setting and the trial number, so "
                "different settings use different seeds.")
    return f"Each trial's random seed is derived from base seed {base}."


def _time_sentence(quest_root: Path, topic: str, duration_s: float | None) -> str:
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not 0 <= duration_s < float("inf"):
        return ""
    required = _read_json(Path(quest_root) / ".fi" / "topic_requirements.json")
    constraints = [c if isinstance(c, str) else c.get("stated", "") if isinstance(c, dict) else ""
                   for c in (required.get("constraints") if isinstance(required, dict) else None) or []]
    limit = _topic_limit([c for c in constraints if isinstance(c, str)], topic or "")
    ran = f"The experiment ran for {_plain_seconds(float(duration_s))}"
    if limit is None:
        return f"{ran}; FI did not check it against a limit in the topic."
    return f"{ran}; the topic sets a limit of {_plain_seconds(limit)}, and the run was {'within' if duration_s <= limit else 'over'} it."


def _count_sentence(rows: Any) -> str:
    if not rows:
        return ""
    planned, trials = rows
    cells = sorted({cell for cell, _t in trials} | set(planned))
    counts: dict[str, int] = {}
    for c, _t in trials:
        counts[c] = counts.get(c, 0) + 1
    per = sorted(set(planned.values())) or sorted({counts.get(cell, 0) for cell in cells})
    each = f"{per[0]:,}" if len(per) == 1 else f"{per[0]:,} to {per[-1]:,}"
    finished = sum(1 for status, _s in trials.values() if status == "ok")
    failed = sum(1 for status, _s in trials.values() if status == "failed")
    settings = f"{len(cells):,} setting{'' if len(cells) == 1 else 's'}"
    return (f"The run had {settings} with {each} trial{'' if per == [1] else 's'} per setting; "
            f"{finished:,} trial{'' if finished == 1 else 's'} finished and {failed:,} failed.")


_UNTESTED_ID = re.compile(r"\bequation\s+(\S+?)\s+is not tested( at all)?\b", re.I)


def _code_faults_sentence(quest_root: Path) -> str:
    """The quantities FI left out because a test it ran on the code that produced the results showed them computed
    wrongly, and the plan's quantities the code does not compute (the evidence gate's record of each, in
    ``.fi/code_review.json``)."""
    record = _read_json(quest_root / ".fi" / "code_review.json")
    issues = record.get("issues") if isinstance(record, dict) and isinstance(record.get("issues"), dict) else {}
    wrong: list[str] = []
    missing: list[str] = []
    for issue in issues.values():
        gate = issue.get("at_gate") if isinstance(issue, dict) else None
        if not isinstance(gate, dict):
            continue
        if gate.get("outcome") == "left_out" and str(gate.get("quantity") or "").strip():
            name = str(gate["quantity"]).strip()
            if name not in wrong:
                wrong.append(name)
        elif gate.get("outcome") == "not_computed":
            for n in gate.get("names") if isinstance(gate.get("names"), list) else []:
                if str(n).strip() and str(n).strip() not in missing:
                    missing.append(str(n).strip())
    said = []
    if wrong:
        said.append("Tests FI ran on the code that produced these results showed it computes " + _names(wrong)
                    + " wrongly, so " + ("this quantity is" if len(wrong) == 1 else "these quantities are")
                    + " left out of the results reported here.")
    if missing:
        said.append("The code does not compute " + _names(missing) + ", which the plan asks for.")
    return " ".join(said)


def _names(names: list[str]) -> str:
    shown = [f"`{n}`" for n in names[:_SHOWN]]
    more = len(names) - _SHOWN
    if more > 0:
        return ", ".join(shown) + f" and {more} more"
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _untested_sentence(notes: list[str]) -> str:
    """The equations FI did not test before the run, by id, in one short sentence (the paper has little room; the reasons
    are in the quality report, from the same record). A note whose id cannot be read is kept as written."""
    ids, never, other = [], [], []
    for note in notes:
        m = _UNTESTED_ID.search(note)
        if not m:
            other.append(note)
            continue
        if m.group(1) not in ids:
            ids.append(m.group(1))
        if m.group(2) and m.group(1) not in never:
            never.append(m.group(1))
    text = ""
    if ids:
        text = (("Equation " + ids[0] + " was") if len(ids) == 1 else
                ("Equations " + ", ".join(ids[:-1]) + " and " + ids[-1] + " were")) + \
            " not tested against a worked example before the run" + \
            (" (" + ", ".join(never) + (" is" if len(never) == 1 else " are") + " not tested at all)" if never else "") + \
            "; the quality report says why."
    if other:
        text = (text + " " + _listed("", other)).strip()
    return text


def run_record(quest_root: Path, *, topic: str = "", duration_s: float | None = None) -> str:
    """One paragraph (no heading) of what FI recorded about the run, for the paper's limitations: the seeds, the run
    time against any limit the topic states, the trial counts, a reduced size, the topic's requirements that were not
    done and the equations not tested before the run. Each sentence is written only from a record that exists and can
    be read; a missing record leaves its sentence out. ``duration_s`` is the measured wall time of the experiment
    run; ``topic`` is the topic's text (its limits are read from it when the topic's constraints state none).
    Nothing here is asked of a model."""
    quest_root = Path(quest_root)
    rows = _ledger_rows(quest_root)
    parts = [_seed_sentence(quest_root, rows), _time_sentence(quest_root, topic, duration_s), _count_sentence(rows)]
    estimate = _read_json(quest_root / ".fi" / "run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    if isinstance(reduced, dict) and isinstance(reduced.get("note"), str) and reduced["note"].strip():
        parts.append(reduced["note"].strip())
    covered = _read_json(quest_root / ".fi" / "topic_covered.json")
    left = _items(covered.get("left")) if isinstance(covered, dict) else []
    if left:
        parts.append(_listed("FI did not do all of the topic's requirements.", left))
    parts.append(_code_faults_sentence(quest_root))
    tests = _read_json(quest_root / ".fi" / "equation_tests.json")
    untested = _items(tests.get("not_tested")) if isinstance(tests, dict) else []
    if untested:
        parts.append(_untested_sentence(untested))
    parts = [p for p in parts if p]
    return f"{_RECORD_LEAD} " + " ".join(parts) if parts else ""


# ---- the paper -----------------------------------------------------------------------------------------------------

#: Any block between the markers, whoever wrote it (with the blank lines around it), a marker left on its own, and a
#: paragraph under the engine's lead words outside the markers (the passage editor can drop a marker and keep the rest).
_ANY_BLOCK = re.compile(r"\n*[ \t]*" + re.escape(BEGIN) + r".*?" + re.escape(END) + r"[ \t]*\n*", re.S)
_STRAY_MARKER = re.compile(r"(?:" + re.escape(BEGIN) + "|" + re.escape(END) + r")")
_RECORD_LEAD_LINE = re.compile(r"\n*^[ \t]*" + re.escape(_RECORD_LEAD) + r"[^\n]*(?:\n|\Z)\n*", re.M)
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


#: The open code findings block (see the module docstring): the checksum of its text is in the opening marker.
FINDINGS_END = "<!-- /fi:code-findings -->"
_FINDINGS_OPEN = r"<!-- fi:code-findings(?::[0-9a-f]*)? -->"
_FINDINGS_ANY = re.compile(r"\n*[ \t]*" + _FINDINGS_OPEN + r".*?" + re.escape(FINDINGS_END) + r"[ \t]*\n*", re.S)
_FINDINGS_STRAY = re.compile(r"(?:" + _FINDINGS_OPEN + "|" + re.escape(FINDINGS_END) + r")")
_FINDINGS_BLOCK = re.compile(
    r"<!-- fi:code-findings:([0-9a-f]{12}) -->[ \t]*\r?\n(.*?)\r?\n[ \t]*" + re.escape(FINDINGS_END), re.S)


#: The run record block (:func:`run_record`): the same scheme as the findings block, its own markers.
RUN_RECORD_END = "<!-- /fi:run-record -->"
_RECORD_OPEN = r"<!-- fi:run-record(?::[0-9a-f]*)? -->"
_RECORD_ANY = re.compile(r"\n*[ \t]*" + _RECORD_OPEN + r".*?" + re.escape(RUN_RECORD_END) + r"[ \t]*\n*", re.S)
_RECORD_STRAY = re.compile(r"(?:" + _RECORD_OPEN + "|" + re.escape(RUN_RECORD_END) + r")")
_RECORD_BLOCK = re.compile(
    r"<!-- fi:run-record:([0-9a-f]{12}) -->[ \t]*\r?\n(.*?)\r?\n[ \t]*" + re.escape(RUN_RECORD_END), re.S)


def findings_checksum(text: str) -> str:
    """The checksum in the opening marker of the open code findings block: 12 hex digits of the SHA-256 of its text
    (line ends and outer blank lines not counted)."""
    return hashlib.sha256(text.replace("\r\n", "\n").strip().encode("utf-8")).hexdigest()[:12]


def _gap(m: re.Match[str]) -> str:
    # What was before and after a block is left one blank line apart, as two paragraphs are.
    return "" if m.start() == 0 else "\n" if m.end() == len(m.string) else "\n\n"


def without_block_findings(markdown: str) -> str:
    """``markdown`` with only the open code findings block (and any marker of it left on its own) taken out."""
    text = markdown
    while True:
        before = text
        text = _FINDINGS_ANY.sub(_gap, text)
        text = _FINDINGS_STRAY.sub("", text)
        if text == before:
            return text


def without_block_run_record(markdown: str) -> str:
    """``markdown`` with only the run record block (and any marker of it left on its own) taken out, and a paragraph
    under the record's lead words that lost its markers: a copy the writer carried over is FI's text, not the
    writer's, and is put back by :func:`mark_run_record` with its markers."""
    text = markdown
    while True:
        before = text
        text = _RECORD_ANY.sub(_gap, text)
        text = _RECORD_STRAY.sub("", text)
        text = _RECORD_LEAD_LINE.sub(_gap, text)
        if text == before:
            return text


def without_block(markdown: str) -> str:
    """``markdown`` with every block between the markers, every marker left on its own, and every paragraph under the
    engine's lead words taken out. Repeated until nothing is left to take out, so taking one piece out cannot join
    what is around it into a new marker. The open code findings block (:func:`mark_findings`) and the run record block (:func:`mark_run_record`) go too."""
    text = markdown
    while True:
        before = text
        text = _ANY_BLOCK.sub(_gap, text)
        text = _STRAY_MARKER.sub("", text)
        text = _LEAD_LINE.sub(_gap, text)
        text = without_block_findings(text)
        text = without_block_run_record(text)
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


_LIMITS = re.compile(
    r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r".*\b(?:limitations?|threats[ \t]+to[ \t]+validity)\b", re.I)
_BEFORE_END = re.compile(
    r"^#{1,6}[ \t]+" + _SECTION_NUMBER + r"(?:references|bibliography|further[ \t]+reading|acknowledg|appendix|appendices)",
    re.I)


def mark_findings(markdown: str, body: str, *, heading: str = "Limitations") -> str:
    """``markdown`` with the open code findings ``body`` (``core/code_review.py:findings_block``) once, between its
    checksummed markers, at the end of the paper's limitations section (the last heading that names limitations or
    threats to validity). A paper with none gets a ``heading`` section of its own, before the references (else at the
    end). Whatever an earlier pass or the writer put between those markers is taken out first; an empty ``body``
    leaves no block."""
    markdown = without_block_findings(markdown)
    if not body.strip():
        return markdown
    return _put_block(markdown, body, "code-findings", FINDINGS_END, heading, heading_inside=True)


def _put_block(markdown: str, text: str, name: str, end: str, heading: str, *, heading_inside: bool) -> str:
    """``markdown`` with ``text`` between ``<!-- fi:NAME:checksum -->`` and ``end`` at the end of the paper's limitations
    section (the last heading that names limitations or threats to validity). A paper with none gets a ``heading``
    section of its own, before the references (else at the end); that heading is inside the block (and its checksum)
    when ``heading_inside``, else just above it. The block is not in ``markdown`` yet."""
    lines = markdown.split("\n")
    start = 0
    if lines and lines[0].strip() == "---" and len(lines) > 1 and _FRONT_MATTER_KEY.match(lines[1]):
        end_fm = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        start = end_fm + 1 if end_fm is not None else 0
    headings = _headings(lines, start)
    title_line = next((i for i, lv, _t in headings if lv == 1), None)
    levels = [lv for i, lv, _t in headings if i != title_line]
    level = min(levels) if levels else 2
    text = text.strip("\n")
    own = [h for h in headings if h[0] != title_line and _LIMITS.match(h[2])]
    lead = ""
    if own:
        at, lv, _t = own[-1]
        place = _section_end(headings, at, lv, len(lines))
    else:
        lead = "#" * level + f" {heading}\n\n"
        if heading_inside:
            text, lead = lead + text, ""
        refs = next((h for h in headings if h[0] != title_line and h[1] <= level and _BEFORE_END.match(h[2])), None)
        place = refs[0] if refs else len(lines)
    if place < len(lines):
        # The engine's references follow a rule line: the block goes before it.
        while place > 0 and not lines[place - 1].strip():
            place -= 1
        if place > 0 and lines[place - 1].strip() == "---":
            place -= 1
    head = lines[:place]
    while head and not head[-1].strip():
        head.pop()
    tail = lines[place:]
    while tail and not tail[0].strip():
        tail.pop(0)
    block = [f"<!-- fi:{name}:{findings_checksum(text)} -->", text, end]
    if lead:
        block = [lead.rstrip("\n"), "", *block]
    return "\n".join([*head, *([""] if head else []), *block, *([""] if tail else []), *tail]) + ("" if tail else "\n")


_FINDINGS_INNER = re.compile(_FINDINGS_OPEN + r"[ \t]*\r?\n?(.*?)\r?\n?[ \t]*" + re.escape(FINDINGS_END), re.S)
_OWN_HEADING = re.compile(r"\A(#{1,6}[ \t]+[^\n]*)\n\n(.*)\Z", re.S)


def mark_run_record(markdown: str, text: str, *, heading: str = "Limitations") -> str:
    """``markdown`` with the run record ``text`` (:func:`run_record`) once, between its checksummed markers, at the end
    of the paper's limitations section, the place :func:`mark_findings` uses. The run record always comes before the
    open code findings: a findings block already in ``markdown`` is taken out and put back after it. Whatever was
    between the run record markers before is taken out first; an empty ``text`` leaves no block."""
    markdown = without_block_run_record(markdown)
    if not text.strip():
        return markdown
    body = ""
    found = _FINDINGS_INNER.search(markdown)
    if found:
        body = found.group(1).strip("\n")
        # A limitations heading the findings block made for itself stays in the paper, above both blocks.
        own = _OWN_HEADING.match(body)
        keep = ""
        if own:
            keep, body = own.group(1), own.group(2)
        markdown = _FINDINGS_ANY.sub(lambda m: f"\n\n{keep}\n\n" if keep else _gap(m), markdown, count=1)
        markdown = without_block_findings(markdown)
    out = _put_block(markdown, text, "run-record", RUN_RECORD_END, heading, heading_inside=False)
    return mark_findings(out, body, heading=heading) if body.strip() else out


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
    the markers stays and is checked. The open code findings block is left out while its text matches the checksum in
    its opening marker (:func:`findings_checksum`); so is the run record block (:func:`run_record`)."""
    if "fi:code-findings:" in text:
        text = _FINDINGS_BLOCK.sub(
            lambda m: " " if findings_checksum(m.group(2)) == m.group(1) else m.group(0), text)
    if "fi:run-record:" in text:
        text = _RECORD_BLOCK.sub(
            lambda m: " " if findings_checksum(m.group(2)) == m.group(1) else m.group(0), text)
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
