"""The paper states what FI recorded about the run; FI checks the paper against its records.

FI used to write three blocks into the paper itself (how often the design changed, what FI recorded about the run, the
open findings in the code). They carried false facts in graded papers (a run time of 2.3 s for a run of 223 s, "base seed 0"
for code that used a number typed into it, reviewer quotes about text no longer in the paper) and states a reader cannot use,
so they are no longer in the paper: they are in the quality report (``QUALITY.md``, ``core/quality_report.py``).

What the blocks made sure of was that the paper says what the study did not do. That is now the writer's job, and FI checks it:

* :func:`writer_section` gives the writer FI's facts about the run (the seed the code used, the measured run time against the
  topic's limit, trials per setting and trials that failed) and a list of **required items** for the limitations, each with a
  short name: a part of the topic the study did not do, a quantity left out of the results because a test showed the code
  computes it wrongly, a quantity the plan asks for that the code does not compute, a precision target that was missed.
* :func:`check` reads the finished paper against the same records, with no model: a run time it states must agree with the
  measured one, a claim that the run was within the topic's limit must agree with it, and every required item's short
  name must appear in the paper (case, spacing and underscores ignored). A failure is a must-fix hit named
  :data:`HIT`; it sends the paper back to be written again (the text route; never the design, never a stop). The seed the
  paper states is checked in ``core/stat_claims.py`` against :func:`core.run_facts.seed_verdict`.
* An item still missing when the review rounds are used up is kept in ``.fi/paper_record_check.json``, which the quality
  report prints, and is said in the run log.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import disclosure as _disclosure
from . import run_facts as _run_facts

HIT = "paper_record_mismatch"
RECORD = Path(".fi") / "paper_record_check.json"

NOT_DONE, LEFT_OUT, NOT_COMPUTED, PRECISION = "not_done", "left_out", "not_computed", "precision"
_KIND_WORDS = {
    NOT_DONE: "a part of the topic the study did not do",
    LEFT_OUT: "a quantity left out of the results because a test showed the code computes it wrongly",
    NOT_COMPUTED: "a quantity the plan asks for that the code does not compute",
    PRECISION: "a precision target that was not reached",
}


@dataclass(frozen=True)
class Item:
    """One thing the paper must say it did not do. ``name`` is the short name the paper is checked for."""

    kind: str
    name: str
    text: str


def _read(root: Path, rel: str) -> Any:
    try:
        return json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _norm(text: str) -> str:
    """Lower case, with backticks and quotes gone, underscores and runs of space as one space."""
    return " ".join(re.sub(r"[`'\"_*]", " ", str(text)).lower().split())


# ---- the required items --------------------------------------------------------------------------------------------


def required_items(quest_root: Path | str) -> list[Item]:
    """Every item the limitations must name, from FI's records. A missing or unreadable record gives none."""
    root = Path(quest_root)
    items: list[Item] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: Any, text: str = "") -> None:
        name = " ".join(str(name or "").split())
        key = (kind, _norm(name))
        if name and key not in seen:
            seen.add(key)
            items.append(Item(kind, name, " ".join(str(text or "").split())))

    covered = _read(root, ".fi/topic_covered.json")
    for gap in (covered.get("left") if isinstance(covered, dict) else None) or []:
        if isinstance(gap, dict):
            add(NOT_DONE, gap.get("name"), gap.get("sentence") or "")
    record = _read(root, ".fi/code_review.json")
    issues = record.get("issues") if isinstance(record, dict) and isinstance(record.get("issues"), dict) else {}
    from . import code_findings as _code_findings

    for issue in issues.values():
        gate = issue.get("at_gate") if isinstance(issue, dict) else None
        if not isinstance(gate, dict) or not _code_findings.flagged_now(issue):
            continue
        if gate.get("outcome") == "left_out":
            add(LEFT_OUT, gate.get("quantity"), "a test showed the code computes it wrongly")
        elif gate.get("outcome") == "not_computed":
            for n in gate.get("names") if isinstance(gate.get("names"), list) else []:
                add(NOT_COMPUTED, n, "the plan asks for it and the code does not compute it")
    checked = _read(root, "needs/RESULT_CHECK.json")
    for c in (checked.get("checks") if isinstance(checked, dict) else None) or []:
        if isinstance(c, dict) and c.get("quantity"):
            add(LEFT_OUT, c["quantity"], str(c.get("note") or "")[:200])
    missed = _read(root, str(_disclosure.PRECISION_RECORD))
    for m in missed if isinstance(missed, list) else []:
        if isinstance(m, dict) and m.get("name"):
            try:
                said = (f"a 95% interval of ±{float(m['half_width']):.3g} was reached, the plan's target is ±{float(m['target']):.3g}"
                        + (f", from {int(m['trials']):,} trials" if isinstance(m.get("trials"), (int, float)) else ""))
            except (KeyError, TypeError, ValueError):
                said = "the plan's precision target was not reached"
            add(PRECISION, m["name"], said)
    return items


def missing_items(paper_md: str, items: list[Item]) -> list[Item]:
    """The items whose short name does not appear in the paper (case, spacing, underscores and backticks ignored)."""
    text = _norm(_disclosure.without_block(paper_md))
    return [i for i in items if _norm(i.name) not in text]


# ---- the run time ---------------------------------------------------------------------------------------------------


def _ledger_trial_seconds(root: Path) -> float | None:
    """The sum of the per-trial times in FI's ledger: the trials alone took at least this long (every setting also runs in
    a process of its own). ``None`` when there is no ledger."""
    path = root / "raw" / "ledger.jsonl"
    if not path.is_file():
        return None
    total = 0.0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if '"duration_s"' not in line:
                    continue
                try:
                    d = json.loads(line).get("duration_s")
                except ValueError:
                    continue
                if isinstance(d, (int, float)) and not isinstance(d, bool):
                    total += float(d)
    except OSError:
        return None
    return round(total, 3)


def measured_run_time(quest_root: Path | str, fallback_s: Any = None) -> tuple[float | None, float | None]:
    """``(wall seconds of the run that produced the stored trials, the least the trials alone took)``, each ``None`` when FI
    does not know it. The wall time is the record (``core/run_facts.py``); a study without FI's trials has none but
    ``fallback_s``, the time of its last execution. A quest whose trials were run before the record existed has only the
    least time, from the ledger: the time of the last execution is not the experiment's."""
    root = Path(quest_root)
    wall = _run_facts.run_seconds(root)
    least = _ledger_trial_seconds(root)
    if wall is None and least is None and not isinstance(_run_facts.read(root).get("trials_run"), dict):
        wall = _run_facts.run_seconds(root, fallback_s)
    return wall, least


def _plain(seconds: float) -> str:
    return _disclosure._plain_seconds(seconds)


# ---- what the writer is given ---------------------------------------------------------------------------------------


def seed_fact(quest_root: Path | str) -> str:
    """What the code used as a seed, for the writer; ``""`` when the records say nothing."""
    root = Path(quest_root)
    verdict = _run_facts.record_seed(root)
    base, _trial = _disclosure.recorded_seeds(root)
    rows = _disclosure._ledger_rows(root)
    study = _disclosure._read_json(root / "code" / "study.json")
    literals = verdict.get("code_seeds") or []
    if verdict.get("fi_seed_reaches") is False:
        where: dict[int, list[str]] = {}
        for lit in literals:
            where.setdefault(lit["value"], []).append(lit["where"])
        used = "; ".join(f"{v} ({', '.join(w[:3])})" for v, w in list(where.items())[:4])
        return (f"Seed: the seed FI gives each trial does NOT reach the simulation ({verdict.get('why')}). Its generators are "
                f"seeded from numbers written in the code: {used or 'FI found no seed number in the code'}. State that seed "
                "as the run's seed" + (f"; do not say the trials' seeds are derived from base seed {base}" if base is not None else "")
                + ", and say that the seeds are fixed by the code, not assigned per trial.")
    if base is None:
        return ""
    sentence = _disclosure._seed_sentence(root, rows) if isinstance(study, dict) else ""
    return "Seed: " + (sentence or f"each trial's random seed is derived from base seed {base}.")


def run_time_fact(quest_root: Path | str, topic: str, fallback_s: Any = None) -> str:
    """The measured run time against the topic's limit, for the writer; ``""`` when FI has no time to give."""
    wall, least = measured_run_time(quest_root, fallback_s)
    limit = _disclosure.topic_limit(Path(quest_root), topic)
    if wall is None:
        if least is None:
            return ""
        return (f"Run time: FI did not record how long the whole experiment took; the trials alone took at least "
                f"{_plain(least)}. Do not state a run time.")
    text = f"Run time: the experiment ran for {_plain(wall)} (measured)."
    if limit is not None:
        text += (f" The topic sets a limit of {_plain(limit)}; the run was {'within' if wall <= limit else 'OVER'} it"
                 + ("" if wall <= limit else ", so do not say the limit was met") + ".")
    return text


def writer_section(quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> str:
    """The record section of the write prompt: the facts the Methods' reproducibility statement is made of, and the items the
    limitations must name. ``""`` when FI has nothing to give."""
    root = Path(quest_root)
    facts: list[str] = []
    for part in (seed_fact(root), run_time_fact(root, topic, fallback_s)):
        if part:
            facts.append(part)
    count = _disclosure._count_sentence(_disclosure._ledger_rows(root))
    if count:
        facts.append("Trials: " + count[:1].lower() + count[1:])
    estimate = _read(root, ".fi/run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    if isinstance(reduced, dict) and isinstance(reduced.get("note"), str) and reduced["note"].strip():
        facts.append("Reduced size: " + reduced["note"].strip())
    items = required_items(root)
    if not facts and not items:
        return ""
    lines = ["FI'S RECORD OF THIS RUN. These are facts FI measured or read from its own records. Put them in the paper in your "
             "own words; a statement that disagrees with one, or a required item left out, is found by a check and the paper "
             "is sent back to be written again."]
    if facts:
        lines += ["In the Methods, a short reproducibility statement (the seed the code used, the measured run time against "
                  "the topic's limit if there is one, the trials per setting and how many failed) built from these facts:",
                  *[f"- {f}" for f in facts]]
    if items:
        lines += ["In the Limitations, say each of these by its short name (the name in quotes, spelled as given) and what it "
                  "means in plain words. A paper that leaves a part of the topic out without saying so is not trusted:",
                  *[f"- \"{i.name}\" ({_KIND_WORDS[i.kind]})" + (f": {i.text}" if i.text else "") for i in items]]
    return "\n".join(lines)


# ---- the check ------------------------------------------------------------------------------------------------------

_RUN_WORD = re.compile(
    r"\b(?:ran|run|runs|runtime|run[ -]time|wall[ -]?clock|took|takes|completed|finished|elapsed|executed|execution|"
    r"computing time|compute time|cpu time)\b", re.I)
#: A time that is not the whole experiment's: per trial, per setting, an average, one analysis step.
_PER_WORD = re.compile(
    r"\b(?:per|each|every|mean|average|median|typical|typically|single|one trial|one run|analysis|plotting|figure|"
    r"startup|start-up|import)\b", re.I)
_DURATION = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|s|h)(?![\w])", re.I)
#: Words just before a duration: it is a limit being stated, or a claim that the run stayed within it.
_LIMIT_BEFORE = re.compile(
    r"\b(?:limit(?: of| is| was)?|budget(?: of)?|cap(?: of)?|ceiling(?: of)?)\s*[:=]?\s*(?:about |roughly |~)?\s*$", re.I)
_LIMIT_AFTER = re.compile(r"^\s*(?:run[ -]?time |wall[ -]clock |time )?(?:limit|budget|cap|ceiling|constraint)\b", re.I)
_WITHIN_BEFORE = re.compile(
    r"(?:\bunder|\bwithin|\bbelow|\bless than|\bat most|\bno more than|\bup to|\bnot (?:to )?exceed\w*|<=?|≤)\s*"
    r"(?:about |roughly |approximately |~)?\s*$", re.I)
#: Words that say the sentence admits the limit was missed.
_ADMITS = re.compile(
    r"(?<!not )(?<!never )\b(?:exceed\w*|over|above|longer|violat\w*|fail\w*|missed|overran|slower|too long|unmet)\b"
    r"|\bnot (?:met|within|under|satisfied)\b|\b(?:did|does|do) not (?:meet|satisfy)\b|n't (?:meet|satisfy)", re.I)
_MET = re.compile(
    r"\b(?:met|meets|meeting|satisf\w+|respect\w+|within|under|inside|below|complies|complied|compliant)\b[^.]{0,40}"
    r"\b(?:limit|budget|constraint|bound|requirement|ceiling)\b", re.I)
_TIME_WORD = re.compile(r"\b(?:time|runtime|run[ -]time|second|seconds|minute|minutes|wall|cpu|computing|compute)\b", re.I)
_SCALE = (("h", 3600.0), ("m", 60.0), ("s", 1.0))
#: How far a stated run time may be from the measured one (a rounding such as "about 4 minutes" for 223 s is within it).
_TOLERANCE = 0.3


def _seconds(num: str, unit: str) -> float | None:
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    return value * next(s for u, s in _SCALE if unit.lower().startswith(u))


def _sentences(paper_md: str) -> list[str]:
    from .record_claims import _sentences as prose

    return prose(_disclosure.without_block(paper_md))


def _shown(sentence: str) -> str:
    one = " ".join(sentence.split())
    return one if len(one) <= 160 else one[:157] + "..."


def time_hits(paper_md: str, quest_root: Path | str, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Must-fix hits for a run time the paper states that disagrees with the measured one, and for a claim that the run was
    within the topic's limit when the measurement says it was not. Silent when FI measured nothing."""
    root = Path(quest_root)
    wall, least = measured_run_time(root, fallback_s)
    if wall is None and least is None:
        return []
    limit = _disclosure.topic_limit(root, topic)
    hits: list[str] = []
    for s in _sentences(paper_md):
        has_run = bool(_RUN_WORD.search(s))
        admits = bool(_ADMITS.search(s))
        flagged = False
        for m in _DURATION.finditer(s):
            seconds = _seconds(m.group(1), m.group(2))
            if seconds is None:
                continue
            before = s[max(0, m.start() - 32):m.start()]
            if _LIMIT_BEFORE.search(before) or _LIMIT_AFTER.match(s[m.end():m.end() + 24]):
                continue  # the sentence states a limit; whether the run met it is judged below
            if _WITHIN_BEFORE.search(before):
                if has_run and not admits and wall is not None and wall > seconds * 1.0001:
                    hits.append(f"{HIT}: the paper says the run stayed within {_plain(seconds)} (\"{_shown(s)}\"), but FI "
                                f"measured {_plain(wall)} for the run that produced the trials. State the measured time and "
                                "do not say a limit was met that was not.")
                    flagged = True
                continue
            if not has_run or _PER_WORD.search(s):
                continue
            if wall is not None and abs(seconds - wall) > max(1.0, _TOLERANCE * wall):
                hits.append(f"{HIT}: the paper says the run took {_plain(seconds)} (\"{_shown(s)}\"), but FI measured "
                            f"{_plain(wall)} for the run that produced the trials. State the measured time.")
                flagged = True
            elif wall is None and least is not None and seconds < least * 0.95:
                hits.append(f"{HIT}: the paper says the run took {_plain(seconds)} (\"{_shown(s)}\"), but the trials alone "
                            f"took at least {_plain(least)} in FI's ledger. State a run time FI measured, or none.")
                flagged = True
        if (limit is not None and wall is not None and wall > limit and not flagged and not admits
                and _MET.search(s) and _TIME_WORD.search(s)):
            hits.append(f"{HIT}: the paper says the run met the topic's limit (\"{_shown(s)}\"), but FI measured "
                        f"{_plain(wall)} against a limit of {_plain(limit)}. Say the limit was not met.")
    return list(dict.fromkeys(hits))


def item_hits(paper_md: str, quest_root: Path | str) -> list[str]:
    """Must-fix hits for each required item the paper does not name."""
    return [f"{HIT}: the paper does not mention \"{i.name}\", which FI's records list as {_KIND_WORDS[i.kind]}"
            + (f" ({i.text})" if i.text else "") + ". Say it in the limitations, by that name."
            for i in missing_items(paper_md, required_items(quest_root))]


def check(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Every must-fix hit of the records check for ``paper_md`` (run time, limit claim, required items). Never raises."""
    try:
        return [*time_hits(paper_md, quest_root, topic, fallback_s), *item_hits(paper_md, quest_root)]
    except Exception:  # noqa: BLE001 -- a check that cannot be made flags nothing
        return []


def keep_check(quest_root: Path | str, hits: list[str], paper_md: str) -> None:
    """Keep what the last review found (``.fi/paper_record_check.json``): the hits and the required items the paper still
    leaves out. The quality report prints it, so an item still missing when the rounds are used up is on record."""
    path = Path(quest_root) / RECORD
    try:
        unmet = [{"kind": i.kind, "name": i.name, "text": i.text}
                 for i in missing_items(paper_md, required_items(quest_root))]
        if not hits and not unmet:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"at": time.time(), "hits": list(hits), "unmet_items": unmet}, indent=1) + "\n",
                        encoding="utf-8")
    except (OSError, ValueError):
        pass


def kept_check(quest_root: Path | str) -> dict[str, Any]:
    data = _read(Path(quest_root), str(RECORD))
    return data if isinstance(data, dict) else {}
