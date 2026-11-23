"""The facts of the run in the paper come from FI's records, through placeholders; FI checks that they are there.

FI used to write three blocks into the paper itself (how often the design changed, what FI recorded about the run, the
open findings in the code). They carried false facts in graded papers (a run time of 2.3 s for a run of 223 s, "base seed 0"
for code that used a number typed into it, reviewer quotes about text no longer in the paper), so they are no longer in the
paper: they are in the quality report (``QUALITY.md``, ``core/quality_report.py``). A later attempt asked the writer to
state the facts in its own words and parsed the prose to check them; any sentence could slip past such a parser (a
"per setting" or a "failed" elsewhere in the sentence switched the check off), and real sentences about simulated time were
flagged. So the writer no longer types these facts at all:

* **The run's facts** are one paragraph FI writes and places itself, ``{{run_facts}}`` (:func:`run_facts_text`): the
  number of settings and trials and the measured run time. FI puts it at the end of the Methods (:func:`place_blocks`);
  the writer does not state these facts. The seed the code used, the delivered code's run time and the verdict against
  the topic's time limit are in the quality report only. The single placeholders of older drafts (:data:`FACT_NAMES`:
  ``{{run_time}}``, ``{{time_limit}}``, ``{{time_limit_met}}``, ``{{trials_per_setting}}``, ``{{settings}}``,
  ``{{trials_failed}}``) are still filled, but no longer offered or asked for. One is offered: ``{{interval_level}}``, the
  percent level of every interval FI computes ("95"), so a method sentence that names the level ("a {{interval_level}}%
  interval") types no 95 of its own (:func:`core.typed_numbers.allowed` no longer lets a typed 95% pass).
* **Placeholders.** Each thing the study did not do is one item (:func:`required_items`) with one plain sentence from
  FI's record; the quality report lists them, the paper carries none. A ``{{not_done:<id>}}`` of an older draft is still
  filled.
* **Blocks.** ``{{results_table}}`` (:data:`BLOCK_NAMES`) is a whole block FI writes: its tables of every setting with
  each value's interval and reference (core/results_by_setting.py, saved in ``.fi/results_table.md``). The writer puts
  it on a line of its own in the Results; when it does not, FI places it before the discussion (:func:`place_blocks`).
  The checks of the writer's own text leave it out (:func:`without_blocks`, :func:`mask`).
* **The table of the asked results.** ``{{asked_results}}`` (core/asked_results.py, saved in ``.fi/asked_results.md`` by
  ``Engine._place_blocks``) lists the quantities the topic asks for that FI holds a value for, each value a ``{{v:...}}`` key.
  :func:`place_blocks` puts it at the end of the section that reports the results after every write round, once.
* **The closing sentence.** ``{{quality_pointer}}`` (core/quality_pointer.py, saved in ``.fi/quality_pointer.md`` by
  ``Engine._place_blocks``) is one plain sentence at the end of the body, before the reference list, that points to the
  quality report. It is there only when a value outside the range the plan expected, or a sentence the final claim check found
  unsupported, stands in the paper; it names only the kinds that occur, and is put in once and gone when the last case is.
* **Rendering.** :func:`render` fills each placeholder from the records when the paper is saved, between markers
  (``<!-- fi:fact-run_time -->223.2 s<!-- /fi:fact-run_time -->``), so ``paper.md`` and every check that reads it hold the
  values, never a raw placeholder. The markers let FI put the values back after any step that edits the text (the passage
  editor gets placeholders again, :func:`to_placeholders`; the page trim never offers a sentence holding one; a resumed
  quest re-renders), and they are dropped wherever text leaves FI (:func:`without_marks`). A placeholder FI cannot fill
  reads "not recorded", is said in the run log and is listed in the quality report; it is never left raw and never guessed.
* **No block of FI's audit records.** FI writes no Limitations block, no answers block and no mark beside a sentence
  into the paper: what the study did not do, the sentences the claim check could not confirm and the points of the last
  review are in the quality report only. A draft an older FI wrote with such a block (``{{answers}}``,
  ``{{limitations}}``, :data:`_RETIRED_BLOCKS`) loses it, with nothing in its place.
* **The check** (:func:`check`, no model) is then simple: no name between double braces is unknown to FI
  (:func:`placement_hits`); a run time typed as a number or a claim that the run kept to the topic's limit when it did
  not is left to the typed-number rule and the claim check, which receive FI's record (the seed in it comes from
  :func:`core.run_facts.seed_verdict`). A failure is a must-fix hit named :data:`HIT`; it sends the paper back to be written again (the text route; never the
  design, never a stop). What is still unmet when the review rounds are used up is kept in ``.fi/paper_record_check.json``,
  which the quality report prints, and is said in the run log.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import disclosure as _disclosure
from . import figure_claims as _claims
from . import figure_facts as _figures
from . import frozen_protocol as _frozen
from . import math_safe_cut as _math_safe
from . import supplement as _supplement
from . import asked_results as _asked
from . import quality_pointer as _pointer
from . import paper_values as _values
from . import run_facts as _run_facts
from . import run_manifest as _run_manifest
from . import typed_numbers as _typed

_LOG = logging.getLogger(__name__)

HIT = "paper_record_mismatch"
RECORD = Path(".fi") / "paper_record_check.json"
NOT_RECORDED = "not recorded"

#: The single placeholders of older drafts: still filled, no longer offered (the paragraph ``{{run_facts}}`` replaces them).
FACT_NAMES = ("run_time", "time_limit", "time_limit_met", "trials_per_setting", "settings", "trials_failed",
              "interval_level")
#: FI's blocks a placeholder can name: a whole block of the paper FI writes from its records, on a line of its own.
#: Not a fact: its numbers are accounted for setting by setting (core/results_by_setting.py), never as FI's record.
#: ``run_facts`` has no file: its paragraph is composed from :func:`facts` (:func:`run_facts_text`).
BLOCK_NAMES = ("results_table", "run_facts", "asked_results", "quality_pointer")
#: Blocks an older FI wrote into the paper: a draft that still holds one loses it, with nothing in its place
#: (:func:`render`, :func:`place_blocks`).
_RETIRED_BLOCKS = ("answers", "limitations")
_BLOCK_FILES = {"results_table": Path(".fi") / "results_table.md",  # = core.results_by_setting.PAPER_BLOCK
                "asked_results": Path(".fi") / "asked_results.md",  # = core.asked_results.FILE
                "quality_pointer": Path(".fi") / "quality_pointer.md"}  # = core.quality_pointer.FILE
#: ``{{figure:<file stem>:layout}}`` and ``{{figure:<file stem>:lines}}``: what a saved figure draws, from the recorder's
#: files (:mod:`core.figure_facts`), not from the writer. The marker key is ``figure-<kind>-<stem>``.
#: ``{{figure:<file stem>:<claim> @ <args>}}``: a statement about what the figure draws (:mod:`core.figure_claims`), worded
#: and checked by FI against its call record of the drawing. Its marker key is ``figure-<claim>-<stem>-args<hex>``, the
#: hex being the arguments as written, so that a re-render evaluates the same claim again.
_FIGURE_KEY = "figure-"
_ALIASES = {"runtime": "run_time", "wall_time": "run_time", "run_seconds": "run_time", "limit": "time_limit",
            "runtime_limit": "time_limit", "limit_met": "time_limit_met", "within_limit": "time_limit_met",
            "trials_each_setting": "trials_per_setting", "failed_trials": "trials_failed"}

NOT_DONE, LEFT_OUT, NOT_COMPUTED, PRECISION = "not_done", "left_out", "not_computed", "precision"
SECOND_VERSION, UNTESTED, CODE_FINDING, FAILED_TRIALS = "second_version", "untested", "code_finding", "failed_trials"
SETTING_LEFT_OUT, CALIBRATION, TRIALS_UNCHECKED = "setting_left_out", "calibration", "trials_unchecked"
UNDEFINED = "undefined"
#: A quantity kept although its values are outside the range the plan expected (core/result_checks.py ``kept_bound_breaks``).
OUT_OF_RANGE = "out_of_range"
FEWER_RUNS = "fewer_runs"
CUT_SHORT = "cut_short"
RUN_DIFFERENCE = "run_difference"
SHARED_SEEDS = "shared_seeds"
UNSUPPORTED_CLAIMS = "unsupported_claims"
UNRESOLVED_REVIEW = "unresolved_review"
FIGURE_LEFT_OUT = "figure_left_out"
FIGURE_PASS_FAULT = "figure_pass_fault"
FIGURE_REFERENCE = "figure_reference"
METHOD_DIFFERS = "method_differs"
#: A check of the code against a known answer that the quest went on without (``needs/ORACLE_CHECK.json``: it failed,
#: could not be run on its case, or the trials could not decide it).
KNOWN_ANSWER = "known_answer"
#: Kept while the check of the sentences about the figures stopped with an error of FI's own on the paper last written.
FIGURE_PASS_RECORD = Path(".fi") / "figure_pass_fault.json"
FIGURE_PASS_SENTENCE = ("FI could not check the sentences about the figures: its check stopped with an error of its own, so "
                        "they stand as the writer wrote them.")
LAST_REVIEW = "last_review"
#: The sentences of the delivered paper the last claim check found supported by neither the run's results nor a cited
#: source (``Engine._state_unsupported_claims``): ``{"claims": [{"claim", "evidence"}]}``. Empty when the last check found none.
UNSUPPORTED_RECORD = Path(".fi") / "unsupported_claims.json"
#: FI's measurement of the finished paper against the topic's page limit (``Engine._page_limit_review``):
#: ``{"pages", "limit", "final"}``. ``final``: no further shortening rewrite will follow, so the paper stays this long.
PAGE_RECORD = Path(".fi") / "page_limit.json"
#: The name of the item the paper must place when what the analysis reports differs from FI's record of the run and the
#: quest went on (``engine.run_manifest_check: warn``).
RUN_DIFFERENCE_NAME = "difference from the run record"
#: The name of the item the paper must place when FI ran fewer runs per setting than the plan asked for (to fit a time limit).
FEWER_RUNS_NAME = "fewer runs than the plan asked for"
#: The name of the item the paper must place when FI ran the plan's settings only up to the time limit.
CUT_SHORT_NAME = "study cut at the time limit"
_ID_PREFIX = {NOT_DONE: "topic", LEFT_OUT: "left-out", NOT_COMPUTED: "not-computed", PRECISION: "precision",
              SECOND_VERSION: "second-version", UNTESTED: "equation", CODE_FINDING: "finding", FAILED_TRIALS: "failed-trials",
              SETTING_LEFT_OUT: "setting-left-out", CALIBRATION: "calibration", TRIALS_UNCHECKED: "trials-unchecked",
              UNDEFINED: "undefined",
              FEWER_RUNS: "fewer", CUT_SHORT: "cut-short", RUN_DIFFERENCE: "difference", SHARED_SEEDS: "shared-seeds",
              UNSUPPORTED_CLAIMS: "unsupported", UNRESOLVED_REVIEW: "review", LAST_REVIEW: "last-review",
              FIGURE_LEFT_OUT: "figure-left-out", FIGURE_PASS_FAULT: "figure-pass-fault",
              FIGURE_REFERENCE: "figure-reference", METHOD_DIFFERS: "method-differs",
              KNOWN_ANSWER: "known-answer", OUT_OF_RANGE: "out-of-range"}
#: More settings left out than this are placed as one item.
_MAX_SETTING_ITEMS = 6


@dataclass(frozen=True)
class Item:
    """One thing the paper must say the study did not do: ``id`` is what the writer places as ``{{not_done:<id>}}``,
    ``sentence`` what FI puts there."""

    kind: str
    id: str
    name: str
    sentence: str
    key: str = ""

    @property
    def text(self) -> str:  # the plain sentence, for the quality report
        return self.sentence


@dataclass(frozen=True)
class Fact:
    """One fact of the run: ``text`` is what FI puts where ``{{name}}`` is."""

    name: str
    text: str


@dataclass(frozen=True)
class Rendered:
    text: str
    unfilled: tuple[str, ...]  # the placeholders that read "not recorded"
    values: tuple[_values.Value, ...] = ()  # every ``{{v:...}}`` key and what FI made of it


def _read(root: Path, rel: str) -> Any:
    try:
        return json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _slug(text: Any) -> str:
    """A short id from ``text``: lower-case words joined by hyphens, cut after a whole word at about 40 characters."""
    words = [w for w in re.split(r"[^a-z0-9]+", str(text or "").lower()) if w]
    out = ""
    for w in words:
        if out and len(out) + 1 + len(w) > 40:
            break
        out = f"{out}-{w}" if out else w[:40]
    return out or "item"


def _one(text: Any) -> str:
    return " ".join(str(text or "").split())


def _whole(text: Any, cap: int) -> str:
    """``text`` as one line when it fits in ``cap`` characters, else ``""``: a reason written by a model is quoted whole or
    left out, never cut in the middle of a word or a bracket (a cut gave "hard-coded to )")."""
    one = _one(text)
    return one if len(one) <= cap else ""


def _sentence(text: str) -> str:
    text = _one(text).replace("<!--", "").replace("-->", "")
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


# ---- what the study did not do ---------------------------------------------------------------------------------------


def _topic_sentence(gap: Any) -> tuple[str, str]:
    """``(name, sentence)`` for one part of the topic FI's check did not find in the plan (an object of
    ``topic_covered.left``, or a plain sentence of an older record). What FI found is said, not that the study did not do
    it: the check misses a setting held under another name or form. When the plan's own map named a part of the plan for
    it (``named``), which one model's reading did not accept, that is said too."""
    name, said = _topic_sentence_found(gap)
    if isinstance(gap, dict) and _one(gap.get("named")) and said:
        from .topic_requirements import named_sentence
        said = f"{said} {named_sentence(str(gap.get('kind') or ''), _one(gap.get('named')))}"
    return name, said


def _topic_sentence_found(gap: Any) -> tuple[str, str]:
    """:func:`_topic_sentence` without what the plan's own map named."""
    if isinstance(gap, str):
        said = _one(gap).rstrip(".")
        if re.match(r"(?i)the topic (asks|says|states|wants|requires)\b", said):
            return gap, _sentence(f"{said}; FI's check did not find it in the study's plan")
        return gap, _sentence(f"The topic asks for this, and FI's check did not find it in the study's plan: {said}")
    if not isinstance(gap, dict):
        return "", ""
    name = _one(gap.get("name"))
    kind = gap.get("kind")
    said = _one(gap.get("sentence"))
    if kind == "factor":
        return name, _sentence(f"The topic asks to vary {name}; FI's check did not find a setting of the study's plan "
                               "that varies it")
    if kind == "quantity":
        return name, _sentence(f"The topic asks to report {name}; FI's check did not find it among what the study's "
                               "plan measures")
    if kind == "level":
        return name, _sentence(f"The topic states which values of {name} to use; FI's check did not find them among "
                               "the study's settings")
    if kind == "search":
        return name, _sentence(f"The topic asks to search for {name}; FI's check did not find a search for it in the "
                               "study's plan")
    formula = re.search(r"`([^`]+)`", said)
    if kind == "formula" or formula:
        return name, _sentence(f"The topic states the formula `{formula.group(1) if formula else name}`; FI's check did "
                               "not find it in the study's model")
    return name or said, _sentence(f"{said.rstrip('.')}; this study did not do it" if said else
                                   f"The topic asks for {name}; this study did not do it")


def _said(root: Path, name: str) -> str:
    """A quantity the way the paper names it: the plan's label for it, else its name, in quotes (never a code identifier
    in backticks)."""
    metric, label = _metric_of(root, name)
    return f'"{label or name}"'


def _code_files_sentence(root: Path) -> str:
    """The code is in more files than the topic asked for, said for a reader of the paper from the layout FI recorded
    (``.fi/code_layout.json``), never from the sentence FI stored for its own log."""
    from . import code_layout as _layout

    decision = _layout.load(root) or {}
    asked = decision.get("topic_code_files")
    parts = ("the simulation, the analysis, the model's functions and their tests" if decision.get("shape") == _layout.PACKAGE
             else "the simulation and the analysis")
    if not isinstance(asked, int) or asked < 1:
        return f"The code is kept in separate parts ({parts}), in more files than the topic asked for"
    wanted = "the single file" if asked == 1 else f"the {asked} files"
    return f"The code is kept in separate parts ({parts}), not in {wanted} the topic asked for"


def _labelled_precision(root: Path, reduced: dict[str, Any]) -> dict[str, Any]:
    """``reduced`` with the precision's metric named the way the paper names it: when the record holds the id of one of the
    plan's metrics, the plan's label for it (the record keeps the id; a reader of the paper knows the label)."""
    precision = reduced.get("precision")
    if not isinstance(precision, dict):
        return reduced
    for m in _plan_metrics(root):
        if str(m.get("id") or "") == str(precision.get("metric") or "") and _one(m.get("label")):
            return {**reduced, "precision": {**precision, "metric": _one(m["label"])}}
    return reduced


def _listed_labels(labels: list[str], cap: int = 4) -> str:
    shown = labels[:cap]
    if len(labels) > cap:
        shown.append(f"{len(labels) - cap} more")
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _second_not_used(root: Path, function: str) -> bool:
    """Whether FI's dependence record shows the code throws the result of ``function`` away wherever it is called: such a
    disagreement changes how no reported number reads, so it is in QUALITY.md only and not in the paper. A change of its
    result that moved nothing does not show this (a decision need not flip), so it stays in the paper."""
    from . import second_impl as _second

    verdict = _second.read_dependence(root).get(function)
    return isinstance(verdict, dict) and verdict.get("status") == _second.DISCARDED


def second_version_text(root: Path, function: str) -> str:
    """The Limitations sentence for a key function whose two independently written versions still disagree, in the
    paper's words: what the reported results owe to it, from FI's own dependence record (``core/second_impl.py::dependence``:
    the results that changed when the function's result was changed, by the plan's labels), never the function's name, the
    plan's step id or the inputs the versions were compared on (those are in QUALITY.md and ``.fi/second_impl.json``).
    A function the record does not show as used says that."""
    from . import second_impl as _second

    verdict = _second.read_dependence(root).get(function)
    verdict = verdict if isinstance(verdict, dict) else {}
    lead = "Two independently written versions of one calculation in the code gave different results on a test input"
    if verdict.get("status") == _second.FEEDS:
        known = {str(m.get("id")): _one(m.get("label")) or str(m.get("id")) for m in _plan_metrics(root) if m.get("id")}
        labels = [f'"{known[i]}"' for i in dict.fromkeys(str(x) for x in verdict.get("headline_moved") or []) if i in known]
        behind = _listed_labels(labels) if labels else "the reported results"
        return (f"{lead}, and the run did not settle which is right; {behind} "
                f"{'depends' if len(labels) == 1 else 'depend'} on that calculation, so "
                f"{'it is' if len(labels) == 1 else 'they are'} not independently confirmed")
    return f"{lead}, and the run did not show whether the reported results depend on it"


def required_items(quest_root: Path | str) -> list[Item]:
    """Every thing the study did not do, from FI's records (QUALITY.md lists them). A missing or unreadable record gives
    none."""
    root = Path(quest_root)
    items: list[Item] = []
    ids: set[str] = set()
    seen: set[tuple[str, str]] = set()

    def add(kind: str, name: Any, sentence: str, key: Any = None) -> None:
        name = _one(name)
        if not name or not sentence:
            return
        k = (kind, _one(key or name).lower())
        if k in seen:
            return
        seen.add(k)
        base = f"{_ID_PREFIX[kind]}-{_slug(key or name)}"
        ident, n = base, 2
        while ident in ids:
            ident, n = f"{base}-{n}", n + 1
        ids.add(ident)
        items.append(Item(kind, ident, name, sentence, _one(key)))

    covered = _read(root, ".fi/topic_covered.json")
    for gap in (covered.get("left") if isinstance(covered, dict) else None) or []:
        name, said = _topic_sentence(gap)
        add(NOT_DONE, name, said)
    # What the study did not do as the plan wrote it (``not_done``, kept by the engine with a ``kind``: a comparison
    # inside each run that could not be used, code not in the files the topic asked for): its sentence is FI's own.
    for gap in (covered.get("not_done") if isinstance(covered, dict) else None) or []:
        if isinstance(gap, dict) and gap.get("kind") == "code_files":
            add(NOT_DONE, gap.get("name"), _sentence(_code_files_sentence(root)))
        elif isinstance(gap, dict) and _one(gap.get("sentence")):
            add(NOT_DONE, gap.get("name"), _sentence(_one(gap.get("sentence")).rstrip(".")))
    record = _read(root, ".fi/code_review.json")
    issues = record.get("issues") if isinstance(record, dict) and isinstance(record.get("issues"), dict) else {}
    from . import code_findings as _code_findings

    gated: set[str] = set()
    for fid, issue in issues.items():
        gate = issue.get("at_gate") if isinstance(issue, dict) else None
        if not isinstance(gate, dict) or not _code_findings.flagged_now(issue):
            continue
        if gate.get("outcome") == "left_out":
            gated.add(str(fid))
            q = _one(gate.get("quantity"))
            if gate.get("rerun") == "changed":
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a test of the code failed, and a repair that "
                                           "passes the test changes the reported number"))
            else:
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a test of the code failed, and FI did not "
                                           "establish whether the value is wrong or missing"))
        elif gate.get("outcome") == "not_computed":
            gated.add(str(fid))
            for n in gate.get("names") if isinstance(gate.get("names"), list) else []:
                add(NOT_COMPUTED, n, _sentence(f"The plan asks for {_said(root, _one(n))}; the code does not compute it, so "
                                               "this study does not report it"))
    # The analysis FI writes (core/analysis_frame.py): each declared quantity its last run did not work out, from FI's own record
    # of that run (``data/results/derived.json``), never a null left silent.
    from . import analysis_frame as _analysis

    reasons = {"no_data": ("no trial gave a value for it", "no trial gave a value for any of them"),
               "not_written": ("the study's code to work it out was not written", "the study's code to work them out was not written"),
               "blocked": ("a quantity it is worked out from was not computed", "a quantity each is worked out from was not computed"),
               "missing": ("the study's code to work it out gave no value at any setting",
                           "the study's code to work them out gave no value at any setting"),
               "held_back": ("FI's record could not have moved the value the study's code gave (it was worked out along a setting "
                             "no trial read, or from results that never differed)",
                             "FI's record could not have moved the values the study's code gave (each was worked out along a "
                             "setting no trial read, or from results that never differed)")}
    other = ("the study's code to work it out gave no usable value", "the study's code to work them out gave no usable value")
    by_reason: dict[str, list[str]] = {}
    for item in _analysis.not_computed(_read(root, _analysis.DERIVED_PATH.as_posix()) or {}):
        by_reason.setdefault(item["status"] if item["status"] in reasons else "other", []).append(item["id"])
    # One sentence for the quantities that failed for the same reason, naming each (a list of one line each repeated the
    # same words); a single one keeps its own sentence and item.
    for status, qs in by_reason.items():
        qs = [q for q in qs if (NOT_COMPUTED, _one(q).lower()) not in seen]
        if not qs:
            continue
        one, many = reasons.get(status, other)
        if len(qs) == 1:
            add(NOT_COMPUTED, qs[0], _sentence(f"The plan asks for {_said(root, qs[0])}; {one}, so this study does not report it"))
            continue
        said = [_said(root, q) for q in qs]
        listed = ", ".join(said[:-1]) + f" and {said[-1]}"
        add(NOT_COMPUTED, "; ".join(qs), _sentence(f"The plan asks for {listed}; {many}, so this study does not report them"),
            key=f"{status}-{len(qs)}-quantities")
    from . import result_checks as _result_checks

    checked = _read(root, "needs/RESULT_CHECK.json")
    for c in (checked.get("checks") if isinstance(checked, dict) else None) or []:
        if isinstance(c, dict) and c.get("quantity") and _result_checks.takes_out(checked, c):
            q = _one(c["quantity"])
            note = _whole(c.get("note"), 400).rstrip(".")
            if c.get("kind") == "copy_of_result":  # not a wrong value: the same numbers as an earlier result, reported once
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported separately: {note or 'it was the same number as an earlier result in every trial'}"))
                continue
            if c.get("kind") == "code_reader":  # a second model's word, not a check of the results: never "the value is wrong"
                what_fi_knows = {"twice": "; the plan's text gives it two different formulas, and FI holds no check of the value "
                                          "that passed",
                                 "plan_problem": "; the finding is marked as being about the plan, and FI holds no check of the value "
                                                 "that passed"}.get(str(c.get("contested") or ""), "")
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a second model that read the code found it computed "
                                           "differently from the plan" + what_fi_knows + (f" ({note})" if note else "")))
                continue
            if c.get("kind") == _result_checks.PHYSICAL_KIND:  # a value beyond the bound the plan flags as physical
                bound = _one(c.get("bound"))
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a value of it is outside the physical bound the "
                                           "plan gives it" + (f" ({bound})" if bound else "") + ", and the repairs did not "
                                           "change it"))
                continue
            if c.get("kind") == "shown_fault":  # a failing test of the code; with no repair's rerun on record, nothing more is known
                add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a test of the code failed, and FI did not establish "
                                           "whether the value is wrong or missing"))
                continue
            add(LEFT_OUT, q, _sentence(f"{_said(root, q)} is not reported: a check of the results showed the value is wrong"
                                       + (f" ({note})" if note else "")))
    # A quantity kept although its values break a range the plan only expected: reported as it came out, and said here with the range and the furthest value, from FI's record.
    for q, r in _result_checks.kept_bound_breaks(root).items():
        try:
            count, furthest = int(r.get("count") or 1), f"{float(r['value']):g}"
        except (KeyError, TypeError, ValueError):
            continue
        many = f"{count} of its values are" if count > 1 else "a value of it is"
        add(OUT_OF_RANGE, q, _sentence(
            f"For {_said(root, q)}, {many} outside the range the plan expected for it ({_one(r.get('bound'))}); the furthest is "
            f"{furthest}. FI did not establish whether the value or the plan's expected range is wrong, so the value is "
            f"reported as it came out"))
    # A name the plan appears to define two ways (a comparison of the plan's text: core/code_review.py::plan_defines_twice), said as
    # that. Which definition the reported value follows is said only when FI holds a check it measured, that passed and that reads
    # the quantity; a name that is left out already has its one sentence (above), and is not given a second one.
    try:
        from . import code_review as _review

        twice = _review.plan_defined_twice(root)
        left_out = {_review._norm(_review.leading_id(c.get("quantity"))) for c in (checked.get("checks") if isinstance(checked, dict) else None) or []
                    if isinstance(c, dict) and c.get("quantity")}
    except Exception:  # noqa: BLE001 -- a record that cannot be read adds nothing
        twice, left_out = {}, set()
    for key, found in twice.items():
        if key in left_out:
            continue
        held = _review.passing_checks(root, found["id"])
        # the two definitions come from a comparison of the plan's text, and a passing check that reads the quantity does not
        # show which definition every reported value follows: FI says what it compared and which check passed, nothing more
        two_ways = (f"The plan's text gives {_said(root, found['id'])} two different formulas ({found['dependent']} in the "
                    f"variables, {found['metric']} in the protocol)")
        add(METHOD_DIFFERS, found["id"], _sentence(
            f"{two_ways}; FI's check \"{held[0]}\", which reads it, passed, but FI did not establish which formula every "
            "reported value follows" if held else
            f"{two_ways}; FI could not tell which of the two the value reported follows"),
            key=f"{found['id']}-defined-twice")
    for d in (checked.get("kept_differences") if isinstance(checked, dict) else None) or []:
        if isinstance(d, dict) and d.get("quantity") and not set(map(str, d.get("keys") or [])) <= set(map(str, checked.get("removed_quantities") or [])):
            q = _one(d["quantity"])  # a second model's finding that no measurement showed to move a number, for keys that are back
            if (d.get("measured") or {}).get("status") == "unchanged":
                add(METHOD_DIFFERS, q, _sentence(f"A second model claims that the code computes {_said(root, q)} by a method that "
                                                 "differs from the plan's definition. A test written for the claim failed on the code "
                                                 "that produced the results and passed on a repair, and the numbers reported with that "
                                                 "repair did not move beyond their allowance, so the value is kept"))
            else:
                add(METHOD_DIFFERS, q, _sentence(f"A second model claims that the code computes {_said(root, q)} by a method that "
                                                 "differs from the plan's definition; no test FI wrote could show it, so the claim is "
                                                 "not shown and the value is kept"))
    for key, name, said in known_answer_items(root):
        add(KNOWN_ANSWER, name, said, key=key)
    _precision_items(root, add)
    # A key function whose second, independently written version still gives different results (not one that could not
    # be compared: that is in the quality report only).
    try:
        from . import second_impl as _second

        for fname, entry in ((_second.load(root).get("functions") or {}).items()):
            if isinstance(entry, dict) and entry.get("status") == _second.UNRESOLVED and not _second_not_used(root, fname):
                add(SECOND_VERSION, fname, _sentence(second_version_text(root, fname)))
    except Exception:  # noqa: BLE001 -- a record that cannot be read adds nothing
        pass
    # Equations not tested against a worked example before the run.
    tests = _read(root, ".fi/equation_tests.json")
    for note in (tests.get("not_tested") if isinstance(tests, dict) else None) or []:
        text = note.get("sentence") if isinstance(note, dict) else note
        if not isinstance(text, str) or not text.strip():
            continue
        m = _disclosure._UNTESTED_ID.search(text)
        if m:
            add(UNTESTED, m.group(1), _sentence(f"Equation {m.group(1)} was not tested "
                                                + ("at all" if m.group(2) else "against a worked example")
                                                + " before the run"))
        else:
            add(UNTESTED, text[:60], _sentence(text))
    # A fault a test showed by its own assertion, whose test errored on the code that produced the results (the evidence
    # gate keeps where, ``gate_error``): nothing shows it fixed, so the paper says so. Read from the issues themselves:
    # a repair since the reading changes the code, and the reading's own list is then empty (core/code_review.py).
    for fid, issue in issues.items():
        err = issue.get("gate_error") if isinstance(issue, dict) else None
        if (not isinstance(err, dict) or str(fid) in gated or issue.get("changes_result") is not True
                or not _code_findings.flagged_now(issue) or not _code_findings.shown_by_assertion(issue)):
            continue
        text = _one(issue.get("text")).rstrip(".")
        where = _one(issue.get("function") or issue.get("file"))
        detail = _whole(err.get("detail"), 160).rstrip(".")
        add(CODE_FINDING, text[:80] or str(fid), _sentence(
            "A test of the code showed a fault that can change a reported result"
            + (f" (in {where})" if where else "") + (f": {text}" if text else "")
            + "; the test no longer runs on the code that produced the results, so it is not confirmed that the fault is "
            "fixed"), key=fid)
    # Open problems in the code, flagged by a second reading, that can change a reported result (one about a function the
    # run never calls cannot: core/code_review.py:open_findings leaves those out).
    try:
        from . import code_review as _code_review

        found = _code_review.open_findings(root, root / "code")
        for v in [*found["missing"], *found["different"], *found["flagged"], *found["unshown"], *found["not_rechecked"]]:
            fid = str(v.get("fid") or "")
            if fid in gated:
                continue
            issue = issues.get(fid) if fid else None
            changes = (issue.get("changes_result") if isinstance(issue, dict) and issue.get("changes_result") is not None
                       else _code_findings.changes_result(v))
            if not changes:
                continue
            text = _code_review._plain_text(v.get("text"))
            if text and v in found["not_rechecked"]:
                # The code was changed after the review, and every test written for it errored: only the review's word.
                add(CODE_FINDING, text[:80], _sentence(
                    f"A review of the code flagged a possible problem with {text}; no test could show it (every test "
                    "written for it failed to run), and the code was changed afterwards and not checked again for it"),
                    key=fid or text)
            elif text:
                add(CODE_FINDING, text[:80], _sentence("A review of the code flagged a problem that can change a "
                                                       f"reported result and was not resolved: {text}"), key=fid or text)
    except Exception:  # noqa: BLE001
        pass
    rows = _disclosure._ledger_rows(root)
    if rows:
        statuses = [status for status, _seed in rows[1].values()]
        late = _out_of_time(root)
        failed = max(0, sum(1 for s in statuses if s == "failed") - late)
        if failed or late:
            add(FAILED_TRIALS, "failed trials", _sentence(
                (f"{failed:,} of {len(statuses):,} trials failed; their results are not in the reported numbers"
                 + _failed_where(root) if failed else "")
                + ("; " if failed and late else "")
                + (f"{late:,} trials were not run because the study's time limit ran out" if late else "")), key="all")
    estimate = _read(root, ".fi/run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    if isinstance(reduced, dict) and (said := fewer_runs_text(_labelled_precision(root, reduced))):
        add(FEWER_RUNS, FEWER_RUNS_NAME, _sentence(f"The experiment ran {said}"), key="runs")
    if (cut_said := _disclosure.cut_short_text(root)):
        add(CUT_SHORT, CUT_SHORT_NAME, _sentence(cut_said), key="cut")
    manifest = _read(root, "needs/RUN_MANIFEST_CHECK.json")
    if (isinstance(manifest, dict) and manifest.get("status") in ("warned", "differs_in_replicates")
            and run_difference_text(manifest)):
        add(RUN_DIFFERENCE, RUN_DIFFERENCE_NAME, _sentence(run_difference_text(manifest)), key="record")
    _left_out_setting_items(root, add)
    _trial_check_items(root, add)
    if (said_seeds := shared_seeds_text(root)):
        add(SHARED_SEEDS, "seeds shared across settings", _sentence(said_seeds), key="seeds")
    from . import disclosure as _disc

    unresolved = _disc.unresolved_sentence(_read(root, str(_disc.OBJECTIONS_RECORD)))
    if unresolved:
        add(UNRESOLVED_REVIEW, "unresolved review reason", _sentence(unresolved), key="reason")
    said = unsupported_claims_text(_read(root, str(UNSUPPORTED_RECORD)))
    if said:
        add(UNSUPPORTED_CLAIMS, "unsupported sentences", _sentence(said), key="claims")
    for stem, number in _figure_left_out_items(kept_check(root).get("taken_out_values")):
        add(FIGURE_LEFT_OUT, f"figure {number} statements", figure_left_out_sentence(number), key=stem)
    if (Path(root) / FIGURE_PASS_RECORD).is_file():
        add(FIGURE_PASS_FAULT, "figure sentences not checked", FIGURE_PASS_SENTENCE, key="figures")
    for key, name, said in _figure_reference_items(root):
        add(FIGURE_REFERENCE, name, said, key=key)
    points = _disclosure.last_review_points(root)
    if points:
        add(LAST_REVIEW, "final review points", _disclosure.last_review_sentence(points), key="points")
    return items


def _number_said(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def went_on_checks(root: Path) -> list[dict[str, Any]]:
    """The entries of the known-answer checks the quest went on without (``needs/ORACLE_CHECK.json`` with status
    ``went_on_failing``); ``[]`` when there is none. An entry with no name says no check of the plan could be judged."""
    from . import accepted_checks as _accepted

    record = _read(root, f"needs/{_accepted.ORACLE_RECORD}")
    if not isinstance(record, dict) or record.get("status") != _accepted.WENT_ON:
        return []
    return [e for e in record.get("went_on") or [] if isinstance(e, dict) and (e.get("name") or e.get("unmeasured"))]


def known_answer_items(root: Path) -> list[tuple[str, str, str]]:
    """``(key, name, sentence)`` for each check of the code against a known answer the quest went on without, from FI's own
    record (:func:`went_on_checks`), in the paper's words: the plan's own words for what the check compares (the first
    sentence of its ``check``), FI's numbers, and what happened (it failed, it could not be run on its case, the trials
    could not decide it, or the simulation and the plan's own working of the known value disagree; an entry with no name:
    no check could be judged). The writer is told the same, but whether the paper says it never rests on the writer. More
    than :data:`_MAX_SETTING_ITEMS` are one item; the line the Limitations keep counts the checks, not the items."""
    entries = went_on_checks(root)
    if not entries:
        return []
    frozen = _read(root, "needs/FROZEN_PROTOCOL.json")
    oracles = (frozen.get("protocol") or {}).get("oracles") if isinstance(frozen, dict) else None
    # the plan's first sentence of what the check compares (the rest is how the plan sized it), else the check's name
    words = {str(o.get("name")): re.split(r"(?<=\.)\s+(?=[A-Z])", _one(o.get("check")))[0].rstrip(".")
             for o in oracles or [] if isinstance(o, dict) and o.get("name")}
    out: list[tuple[str, str, str]] = []
    kinds: list[str] = []
    # an entry with no name is recorded when no check of the plan could be judged at all: said on its own, never counted
    # or folded in as a check
    none_judged = [("none", "known-answer checks not judged", _sentence(
        "No check of the code against a known answer could be judged, so the results are not validated against one"))
        for e in entries if not e.get("name")][:1]
    for e in entries:
        if not e.get("name"):
            continue
        name = str(e["name"])
        what = words.get(name) or name
        lead = f'The check of the code against a known answer "{what}"'
        tol = e.get("tolerance")
        known = (f"the known value is {_number_said(e.get('expected'))}"
                 + (f" within {_number_said(tol)}" + (" (relative)" if e.get("tolerance_mode") == "relative" else "")
                    if tol is not None else ""))
        gave = f"the run gave {_number_said(e.get('measured'))}, {known}" if e.get("measured") is not None else known
        if e.get("disagreement"):
            kinds.append("could not be judged")
            said = (f"{lead} could not be judged: the simulation and the plan's own working of the known value disagree, and "
                    "neither is confirmed")
        elif e.get("undecided"):
            kinds.append("could not be decided with the trials the study ran")
            said = f"{lead} could not be decided with the trials the study ran ({gave})"
        elif e.get("measured") is None:
            kinds.append("could not be run on its case")
            said = f"{lead} could not be run on its case"
        else:
            kinds.append("did not pass")
            said = f"{lead} did not pass ({gave})"
        out.append((name, f"known-answer check {name}", _sentence(f"{said}; the results are not validated against it")))
    if len(out) > _MAX_SETTING_ITEMS:
        listed = "; ".join(f'"{words.get(k) or k}"' for k, _n, _s in out)
        # each kind counted from what happened to it, so a check that was not run or not decided is never said to fail
        counts = ", ".join(f"{n} {k.replace('its case', 'their cases') if n > 1 else k}"
                           for k in dict.fromkeys(kinds) for n in [kinds.count(k)])
        out = [("checks", "known-answer checks", _sentence(
            f"{len(out)} checks of the code against a known answer did not confirm the results ({counts}), and the study "
            f"went on: {listed}; the results are not validated against them (the quality report gives each with its "
            "numbers)"))]
    return out + none_judged


def shared_seeds_text(root: Path) -> str:
    """The sentence FI's record gives when its own trial record shows the settings sharing seeds: trial ``t`` of
    every setting ran on one seed (``paired: true``, common random numbers; ``core/trial_runner.py::trial_seed`` with an
    empty key). ``""`` when the ledger shows the settings drew different seeds, or there is one setting only.
    Only a plan that states ``protocol.shared_draws`` has the settings share
    seeds; the interval across them is FI's own, from the different draws (``core/paper_values.py``).

    Why it is stated: settings that share their random draws move together, so their results are one correlated
    fluctuation, not independent repeats. A difference between two settings is sharper for it, but an interval pooled
    across settings, or a count of "agreeing" settings, then overstates the precision, and a reader cannot tell this
    from the paper unless it says so. The fact comes from the ledger of the seeds the trials ran with, never from the
    paper's words."""
    rows = _disclosure._ledger_rows(root)
    if not rows:
        return ""
    planned, trials = rows
    if len({cell for cell, _t in trials}) < 2:
        return ""
    shown = _disclosure._seeds_paired(trials)
    if shown is None:
        study = _read(root, "code/study.json")
        shown = bool(isinstance(study, dict) and study.get("paired") is True and len(planned) > 1) or None
    if not shown:
        return ""
    return ("In every setting, trial t ran on the same random seed (common random numbers), so the settings' results are "
            "correlated, not independent: a difference between two settings is paired, so an interval over several "
            "settings counts the different random draws, not the trials of every setting pooled, and a count of settings "
            "that agree is not a count of independent repeats")


_MAX_UNSUPPORTED_LISTED = 8
_MAX_UNSUPPORTED_CHARS = 300
_CITE_MARK = re.compile(r"\s*\[(?:W?\d+(?:\s*[,;–-]\s*W?\d+)*)\]")


def _quoted_claim(text: Any) -> str:
    """One claim as FI's record quotes it: on one line, without the reference numbers it carried (they point at
    the paper's list, not at a source of this sentence), without double braces (no placeholder), cut after a whole word
    when long."""
    one = _one(_CITE_MARK.sub("", str(text or ""))).replace("{{", "").replace("}}", "").replace('"', "'")
    if len(one) > _MAX_UNSUPPORTED_CHARS:
        word = one[:_MAX_UNSUPPORTED_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:")
        one = _math_safe.head_before_math(one, len(word)).rstrip(" ,;:") + " ..."
    return one


def unsupported_claims_text(record: Any) -> str:
    """What the quality report says of the sentences FI's last claim check found supported by neither the run's results
    nor a cited source (:data:`UNSUPPORTED_RECORD`): how many, and each one quoted (at most :data:`_MAX_UNSUPPORTED_LISTED`,
    and how many more there are). The sentences stay in the paper as written. Sentences of the writer's whose claim the
    check found wrong against the record were taken out of the paper (``removed`` in the record): that is said too,
    without quoting them (what they said is what was not supported). ``""`` when the record holds neither."""
    rows = record.get("claims") if isinstance(record, dict) else None
    quoted: list[str] = []
    for row in rows or []:
        q = _quoted_claim(row.get("claim") if isinstance(row, dict) else row)
        if q and q not in quoted:
            quoted.append(q)
    said = ""
    if quoted:
        shown = quoted[:_MAX_UNSUPPORTED_LISTED]
        more = len(quoted) - len(shown)
        n = len(quoted)
        said = (("One sentence of this paper was" if n == 1 else f"{n} sentences of this paper were")
                + " not found supported by the run's results or a cited source: "
                + "; ".join(f'"{q}"' for q in shown) + (f"; and {more} more" if more else ""))
    gone = [r for r in (record.get("removed") if isinstance(record, dict) else None) or [] if isinstance(r, dict)]
    if gone:
        n = len(gone)
        out = (("One sentence" if n == 1 else f"{n} sentences") + " of the draft that FI's check of the claims could not "
               "confirm against the run's results or its sources (a number not in the results, a true value under the "
               "wrong setting, a source that backs nothing, or a verdict FI could not read) " + ("was" if n == 1 else "were")
               + " taken out of this paper")
        said = f"{said}. {out}" if said else out
    return said


def write_unsupported_record(quest_root: Path | str, claims: list[dict[str, Any]],
                             removed: list[dict[str, Any]] | None = None) -> bool:
    """Keep the claims the last claim check judged unsupported, with its evidence words (:data:`UNSUPPORTED_RECORD`); an
    empty list empties the record, so a sentence an earlier pass listed is gone from the quality report.
    ``removed``: the claims whose sentence FI took out of the paper (``claim``, ``evidence``, ``sentence``, and why it went:
    the claim check's ``fault`` and the ``keys`` of FI's values it is about, when given), added to those an earlier pass
    took out of this draft, never replaced. Returns whether the record could be written."""
    path = Path(quest_root) / UNSUPPORTED_RECORD
    rows = [{"claim": _one(c.get("claim")), "evidence": _one(c.get("evidence"))} for c in claims if _one(c.get("claim"))]
    record: dict[str, Any] = {"claims": rows}
    try:
        earlier = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        earlier = {}
    kept = [r for r in (earlier.get("removed") if isinstance(earlier, dict) else None) or [] if isinstance(r, dict)]
    by_sentence = {str(r.get("sentence")): r for r in kept}
    for r in removed or []:
        row = {"claim": _one(r.get("claim")), "evidence": _one(r.get("evidence")), "sentence": _one(r.get("sentence"))}
        if _one(r.get("fault")):
            row["fault"] = _one(r.get("fault"))
        keys = [_one(k) for k in r.get("keys") or [] if _one(k)] if isinstance(r.get("keys"), list) else []
        if keys:
            row["keys"] = keys
        earlier_row = by_sentence.get(row["sentence"])
        if earlier_row is None:
            by_sentence[row["sentence"]] = row
            kept.append(row)
        else:
            # The same sentence again: the earlier entry stays, and gains why it went if it did not say.
            for field in ("fault", "keys"):
                if field in row and not earlier_row.get(field):
                    earlier_row[field] = row[field]
    if kept:
        record["removed"] = kept
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


def forget_removed_unsupported(quest_root: Path | str) -> None:
    """A new draft is written: the sentences FI took out of the earlier draft (``removed`` in :data:`UNSUPPORTED_RECORD`) are
    not stated about this one. The claims the last check found are left to the next check, which writes them again."""
    path = Path(quest_root) / UNSUPPORTED_RECORD
    try:
        record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        if isinstance(record, dict) and record.pop("removed", None) is not None:
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + chr(10), encoding="utf-8")
    except (OSError, ValueError):
        pass


_MAX_DIFFERENCE_CHARS = 500


def run_difference_text(record: dict[str, Any]) -> str:
    """The sentence FI's record gives for a difference it recorded and went on with
    (``needs/RUN_MANIFEST_CHECK.json``), for a reader of the paper: that the numbers the analysis reports do not all match
    the trials that were run, how many places, and that the results rest on the analysis. ``""`` when the record holds no
    such difference: a finding that only says the analysis printed no per-trial list is not one (it is in QUALITY.md and
    run.log). The technical findings stay in the record."""
    headline = " ".join(str(record.get("headline") or "").replace("`", "").split())
    problems = [p for p in record.get("problems") or [] if str(p).strip() and not _run_manifest.is_unprinted(p)]
    seeds = record.get("seeds") if isinstance(record.get("seeds"), dict) else {}
    lead = "The numbers the analysis reports do not all match the record of the trials that were run"
    if headline:
        said = f"{lead}. {headline.rstrip('.')}"
    elif problems:
        said = lead + ": they differ in " + ("1 place" if len(problems) == 1 else f"{len(problems)} places")
    elif seeds:
        said = ("A repeat of the experiment with another random seed differs from the plan"
                + ("" if len(seeds) == 1 else f" ({len(seeds)} repeats)"))
    else:
        return ""
    return said + ". The numbers reported here rest on that analysis"


def _runs(part: Any) -> int | None:
    n = part.get("runs_per_setting") if isinstance(part, dict) else None
    return int(n) if isinstance(n, (int, float)) and not isinstance(n, bool) else None


def fewer_runs_text(reduced: dict[str, Any]) -> str:
    """What the paper must say of a run FI made smaller (``.fi/run_estimate.json`` ``reduced``): the runs per setting it
    ran against the plan's, why, whether that is below the plan's own minimum, and the 95% interval those runs give (or
    that it was not known before the run). ``""`` when the record shows no fewer runs."""
    ran, asked = _runs(reduced.get("to")), _runs(reduced.get("from"))
    if ran is None or asked is None or ran >= asked:
        return ""
    over = reduced.get("over_topic_limit")
    if over == "cut_still_over":
        why = "so that the run fits the topic's time limit; timed again, it is still expected to be over that limit"
    elif over:  # "at_plan_minimum", or True in a record written before the two were told apart
        why = "the plan's minimum, the fewest it allows; the run is still over the topic's time limit"
    elif reduced.get("limit") == "topic":
        why = "so that the run fits the topic's time limit"
    else:
        why = "so that the run fits the time limit for one run"
    said = f"{ran} runs per setting instead of the {asked} the plan asked for ({why})"
    if reduced.get("below_plan_minimum") and isinstance(reduced.get("plan_minimum"), (int, float)):
        said += f", below the plan's minimum of {int(reduced['plan_minimum'])}"
    elif (reduced.get("limit") == "topic" and reduced.get("plan_minimum") is None and not reduced.get("over_topic_limit")
          and reduced.get("precision_known", True) is not False):
        said += "; the plan states no fewest number of runs"
    precision = reduced.get("precision")
    if isinstance(precision, dict) and isinstance(precision.get("half_width"), (int, float)):
        said += (f"; with {ran} runs per setting the 95% interval of {precision.get('metric') or 'the result'} is about "
                 f"±{float(precision['half_width']):g}"
                 + (f" against the plan's target of ±{float(precision['target']):g}"
                    if isinstance(precision.get("target"), (int, float)) else ""))
    else:
        said += "; the 95% interval those runs give was not known before the run"
    return said


def _digits(value: float, other: float) -> str:
    """``value`` with the fewest significant digits (3 to 6) that tell it from ``other`` when both are written the same way."""
    for n in range(3, 7):
        text = f"{value:.{n}g}"
        if text != f"{other:.{n}g}" or n == 6:
            return text
    return f"{value:.6g}"


def _precision_items(root: Path, add: Any) -> None:
    """One item per quantity whose 95% interval was wider than the plan's target, with the range of half-widths over the
    settings it missed in. A record names a quantity by its place in the analysis's output (``by_m.20.pi1_0.05.any_false_bh``):
    the quantity is the plan's metric the name ends with (:func:`_metric_of`), said by the plan's own label; a name that
    ends in none of them is its own item, as before."""
    missed = _read(root, str(_disclosure.PRECISION_RECORD))
    rows = [m for m in missed if isinstance(m, dict) and m.get("name")] if isinstance(missed, list) else []
    groups: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for m in rows:
        metric, label = _metric_of(root, str(m["name"]))
        groups.setdefault(metric or str(m["name"]), []).append(m)
        labels[metric or str(m["name"])] = f"\"{label}\"" if label else f"`{_one(m['name'])}`"
    for key, members in groups.items():
        try:
            target = float(members[0]["target"])
            widths = [float(m["half_width"]) for m in members]
            trials = {int(m["trials"]) for m in members if isinstance(m.get("trials"), (int, float))}
            low, high = min(widths), max(widths)
            where = "" if len(members) == 1 and key == str(members[0]["name"]) else (
                f" in {len(members)} of the settings" if len(members) > 1 else " in one setting")
            span = (f"±{_digits(low, target)}" if low == high else
                    f"between ±{_digits(low, target)} and ±{_digits(high, target)}")
            from_trials = f" (from {next(iter(trials)):,} trials each)" if len(trials) == 1 and len(members) > 1 else (
                f" (from {next(iter(trials)):,} trials)" if len(trials) == 1 else "")
            said = (f"The 95% interval for {labels[key]} is {span}{where}, wider than the plan's target of "
                    f"±{target:g}{from_trials}")
        except (KeyError, TypeError, ValueError):
            said = f"The plan's precision target for {labels[key]} was not reached"
        add(PRECISION, labels[key], _sentence(said), key=key)


_TRAILING_ID_RE = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)$")


def _plan_metrics(root: Path) -> list[dict[str, Any]]:
    """The metrics of the frozen plan; ``[]`` when there is no plan to read."""
    try:
        from . import frozen_protocol as _frozen

        frozen = _frozen.load(root)
        protocol = frozen.get("protocol") if isinstance(frozen, dict) else None
        metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    except Exception:  # noqa: BLE001 -- no plan to read: the name stands as it is
        return []
    return [m for m in metrics if isinstance(m, dict)] if isinstance(metrics, list) else []


def _metric_of(root: Path, name: str) -> tuple[str, str]:
    """``(the metric, its label)`` that ``name`` (a path in the analysis's output, such as ``by_sigma.sigma=0.01.coverage_Ea``)
    is about. The plan's metric the name ends with (the longest id when several fit) with the plan's own label; else the
    name the path ends in, which is the quantity's own name and never the cell it was measured in (a label is then ``""``).
    ``("", "")`` when the name ends in no name at all."""
    best: dict[str, Any] | None = None
    for m in _plan_metrics(root):
        mid = str(m.get("id") or "")
        if mid and (name == mid or name.endswith("." + mid)) and (best is None or len(mid) > len(str(best["id"]))):
            best = m
    if best is not None:
        return str(best["id"]), _one(best.get("label")) or str(best["id"])
    tail = _TRAILING_ID_RE.search(name.strip())
    return (tail.group(1), "") if tail else ("", "")


def _label_of_names(root: Path, names: list[str], fallback: str) -> str:
    """What a reader calls the quantity whose columns in the trials are ``names`` (``Ea_lo``, ``Ea_hi``): the label of the
    plan's metric that declares one of them as its estimate, standard error or interval; else ``fallback``."""
    wanted = {str(n) for n in names if str(n).strip()}
    for m in _plan_metrics(root):
        known = m.get("known_value") if isinstance(m.get("known_value"), dict) else {}
        mine = {str(known.get(k)) for k in ("estimate", "standard_error") if known.get(k)}
        interval = known.get("interval")
        mine |= {str(x) for x in interval} if isinstance(interval, (list, tuple)) else set()
        if wanted & mine and _one(m.get("label")):
            return _one(m.get("label"))
    return fallback


def _undefined_items(root: Path, rec: dict[str, Any], add: Any) -> None:
    """Each value of the plan with no number in SOME trials of a setting (core/trial_checks.py:undefined), one item per
    value. A value with none in EVERY trial of a setting has none there by its definition (a power where there is no true
    effect): it is a plain line in QUALITY.md and run.log, and the writer may say it anywhere, so it is not an item."""
    by: dict[str, list[dict[str, Any]]] = {}
    for u in rec.get("undefined") or []:
        if isinstance(u, dict) and u.get("metric") and not u.get("all"):
            by.setdefault(str(u["metric"]), []).append(u)
    for metric, some in by.items():
        said = f"{_said(root, metric)} has no value in " + "; ".join(
            f"{u.get('trials')} of {u.get('ran')} trials at {u.get('key') or 'the one setting'}" for u in some[:6]) + (
            f"; and at {len(some) - 6} more settings" if len(some) > 6 else "")
        add(UNDEFINED, metric, _sentence(said), key=metric)


def _planned_runs(root: Path) -> int:
    """The trials per setting the frozen plan asks for (0 when there is no frozen plan or it gives none)."""
    from . import frozen_protocol as _frozen

    record = _frozen.load(root) or {}
    protocol = record.get("protocol") if isinstance(record.get("protocol"), dict) else {}
    runs = protocol.get("runs_per_setting")
    return int(runs) if isinstance(runs, (int, float)) and not isinstance(runs, bool) else 0


def _trial_check_items(root: Path, add: Any) -> None:
    """What FI's checks of the trials (core/trial_checks.py) could not show, for a plan with more than one trial a setting:
    that the trials were not FI's to check at all, that the reported intervals and standard errors were not measured
    against the true values (and why), and each one measured below its level in some settings (beyond chance, but not
    grossly: a gross one is disclosed by :func:`_disclosed_item`)."""
    from . import trial_checks as _trial_checks

    rec = _trial_checks.read_record(root)
    if not rec:
        return
    _undefined_items(root, rec, add)
    if rec.get("not_run"):
        if _planned_runs(root) <= 1:
            return
        add(TRIALS_UNCHECKED, "trials not checked", _sentence(
            "The simulation ran its trials by itself, so its failed trials were not counted and its reported intervals and "
            "standard errors were not checked against known true values"), key="all")
        return
    cal = rec.get("calibration") if isinstance(rec.get("calibration"), dict) else None
    # The trials FI ran (its ledger): a simulation FI found deterministic ran each setting once whatever the plan asked.
    rows = _disclosure._ledger_rows(root)
    ran = max(rows[0].values(), default=0) if rows else _planned_runs(root)
    if cal is None or ran <= 1:
        return
    reasons = [str(r) for r in cal.get("not_checked") or [] if str(r).strip()]
    if reasons:
        # Why (a plan that names no true value, too few trials) is in the quality report; the paper says what was not
        # checked, and the one reason a reader needs: a search's intervals are not compared with true values.
        from . import optimise as _optimise

        search = any(r == _optimise.NO_CALIBRATION for r in reasons)
        said = ("The coverage of the reported intervals, and the accuracy of their standard errors, were not checked "
                "against known true values" if not cal.get("checked") else
                "The coverage of some of the reported intervals, and the accuracy of some of their standard errors, were "
                "not checked against known true values")
        add(CALIBRATION, "calibration not checked", _sentence(
            said + ("; the study searched for the best design, and the intervals of a search are not compared with true "
                    "values" if search else "")), key="all")
    for q in cal.get("quantities") or []:
        if isinstance(q, dict) and q.get("fault") and rec.get("action") == _trial_checks.DISCLOSE:
            _disclosed_item(root, q, add)
    for q in cal.get("counted_as_misses") or []:
        if isinstance(q, dict):
            _counted_as_misses_item(root, q, add)
    for q in [*(cal.get("quantities") or []), *(cal.get("inside") or [])]:
        if not isinstance(q, dict) or q.get("fault") or not q.get("below"):
            continue
        vals = [s.get("value") for s in q.get("settings") or [] if isinstance(s, dict)
                and isinstance(s.get("value"), (int, float))]
        plain = _label_of_names(root, q.get("names") or [], "")
        label = _one(q.get("label"))
        shown = f'"{plain}"' if plain else f"`{label}`"
        n, of = int(q.get("below") or 0), int(q.get("judged") or 0)
        if q.get("kind") == "coverage":
            level = q.get("level")
            pct = f"{float(level):.0%}" if isinstance(level, (int, float)) else "their stated level"
            said = (f"The {pct} intervals reported for {shown} contain the true value less often than {pct} in {n} of "
                    f"{of} settings, by more than chance explains"
                    + (f" (lowest coverage {min(vals):.2f})" if vals else ""))
        elif q.get("kind") == "bias":
            said = (f"The estimates of {shown} lie off the true value in {n} of {of} settings, by more than chance "
                    "explains" + (f" (by {min(vals):+.2f} to {max(vals):+.2f} of their own spread)" if vals else ""))
        elif q.get("kind") == "inside":
            said = (f"The estimate {shown} lies inside the interval the same trial reported in fewer trials than it should in {n} of "
                    f"{of} settings" + (f" (as few as {min(vals):.0%} of a setting's trials; 100% when the rows of the record belong "
                                        "together)" if vals else "")
                    + ", so its rows may not belong to the same trials; a correlation or any other statistic that pairs this "
                    "estimate with another across trials may not be right")
        elif q.get("kind") == "no_estimate":
            said = (f"{shown}, the estimate of a known true value, has no number in some trials at {n} settings"
                    + (f" (up to {max(vals):.0%} of a setting's trials)" if vals else ""))
        else:
            said = (f"The standard error reported for {shown} differs from the spread of the estimates over the trials "
                    f"in {n} of {of} settings, by more than chance explains"
                    + (f" (ratio {min(vals):.2g} to {max(vals):.2g}; 1 when right)" if vals else ""))
        add(CALIBRATION, plain or label or "a reported uncertainty", _sentence(said), key=f"{q.get('kind')}-{label}")
    _power_limit_item(root, cal, add)


def _disclosed_item(root: Path, q: dict[str, Any], add: Any) -> None:
    """A reported standard error or interval FI measured not to match the trials, which the repairs did not remove: the
    quest went on (core/trial_checks.py ``DISCLOSE``), and the paper says at which settings it was measured, from FI's own
    record: the number measured and how many trials, and that the interval reported there is not a valid interval of its
    stated level. The settings where it was measured to match are not named as wrong."""
    settings = [s for s in q.get("settings") or [] if isinstance(s, dict) and isinstance(s.get("value"), (int, float))]
    wrong = [s for s in settings if s.get("gross")]
    pooled = q.get("pooled") if isinstance(q.get("pooled"), dict) else None
    plain = _label_of_names(root, q.get("names") or [], "")
    label = _one(q.get("label"))
    shown = f'"{plain}"' if plain else f"`{label}`"
    coverage = q.get("kind") == "coverage"
    level = q.get("level")
    pct = f"{float(level):.0%}" if isinstance(level, (int, float)) else "its stated level"

    def measured(s: dict[str, Any]) -> str:
        v = f"{float(s['value']):.3g}" if not coverage else f"{float(s['value']):.2f}"
        return f"{s.get('key') or 'the one setting'}: {v} from {int(s.get('n') or 0):,} trials"

    if wrong:
        where = ("every setting" if len(wrong) == len(settings) else f"{len(wrong)} of {len(settings)} settings")
        listed = "; ".join(measured(s) for s in wrong[:6]) + (f"; and {len(wrong) - 6} more" if len(wrong) > 6 else "")
    elif pooled is not None and pooled.get("gross"):
        where = "the settings taken together"
        listed = f"{float(pooled['value']):.3g} from {int(pooled.get('n') or 0):,} trials"
    else:
        return
    if coverage:
        what = (f"The {pct} intervals reported for {shown} were measured to contain the true value in "
                f"far fewer trials than {pct} in {where} ({listed}; the share of trials whose interval holds the true value)")
    else:
        what = (f"The standard error reported for {shown} was measured not to match the spread of the estimates over the "
                f"trials in {where} ({listed}; the spread divided by the reported error, 1 when they agree)")
    right = [s for s in settings if not s.get("gross")] if wrong else []
    off = [s for s in right if s.get("below")]
    kept = ""
    if right:
        kept = (f". At the other {len(right) - len(off)} setting{'s' if len(right) - len(off) != 1 else ''} the value measured "
                "was within the range FI accepts" if len(right) > len(off) else "")
        if off:
            kept += (f"{'; at' if kept else '. At'} {len(off)} other setting{'s' if len(off) != 1 else ''} it was off by "
                     "more than chance explains, though not grossly (" + "; ".join(measured(s) for s in off[:4])
                     + (f"; and {len(off) - 4} more" if len(off) > 4 else "") + ")")
    # The settings whose error and interval FI keeps back are exactly the ones measured grossly off, named above; a setting
    # that was off but not grossly stays as reported and is said apart.
    held = ("the setting named above" if len(wrong) == 1 else "the settings named above") if right else where
    if wrong and off:
        kept += (f"; the reported error and interval of {'that setting' if len(off) == 1 else 'those settings'} are kept "
                 "as reported")
    if coverage:
        valid = (f"valid {pct} intervals" if pct != "its stated level" else "valid intervals of their stated level")
        tail = f"The intervals reported for {held} are therefore not presented as {valid}"
    else:
        tail = (f"The standard error reported for {held} is therefore not presented as the error of the estimate there, "
                "and no interval formed from it is presented as a valid interval of its stated level")
    add(CALIBRATION, f"{plain or label or 'a reported uncertainty'}: measured not to match the trials", _sentence(
        what + kept + f". {tail}; FI does not hand on the reported error or interval of {held} (they are not in the results "
        "table or in the numbers the paper can quote), and the estimates themselves are reported as they were computed"),
        key=f"disclosed-{q.get('kind')}-{label}")


def _counted_as_misses_item(root: Path, q: dict[str, Any], add: Any) -> None:
    """The coverage of a reported interval counted over every trial that ran, each trial that gave no interval a miss, where
    it is grossly low and the coverage FI judged the intervals on (the trials that gave an estimate) is not
    (core/trial_checks.py ``counted_as_misses``): the study went on, and the paper states FI's count, with the trials, beside
    the coverage over the trials that gave an estimate. The results table may show only the study's own coverage, over the
    trials that gave a value (and leaves it out where it is the same everywhere), so this is where FI's count reaches the
    paper."""
    settings = [s for s in q.get("settings") or [] if isinstance(s, dict) and isinstance(s.get("value"), (int, float))]
    marked = [s for s in settings if s.get("gross")]
    pooled = q.get("pooled") if isinstance(q.get("pooled"), dict) else None
    plain = _label_of_names(root, q.get("names") or [], "")
    label = _one(q.get("label"))
    shown = f'"{plain}"' if plain else f"`{label}`"
    level = q.get("level")
    pct = f"{float(level):.0%}" if isinstance(level, (int, float)) else "their stated level"

    def counted(s: dict[str, Any], key: str) -> str:
        n = int(s.get("n") or 0)
        v = float(s["value"])
        given = s.get("given")
        rest = (f", {float(given):.2f} of those that gave an estimate" if isinstance(given, (int, float)) else
                ", too few of them gave an estimate to judge those alone")
        return f"{key}: {v:.2f}, {round(v * n):,} of {n:,} trials{rest}"

    if marked:
        where = "every setting" if len(marked) == len(settings) else f"{len(marked)} of {len(settings)} settings"
        listed = "; ".join(counted(s, s.get("key") or "the one setting") for s in marked[:6]) + (
            f"; and {len(marked) - 6} more" if len(marked) > 6 else "")
    elif pooled is not None and pooled.get("gross") and isinstance(pooled.get("value"), (int, float)):
        where = "the settings taken together"
        listed = counted(pooled, "all settings together")
    else:
        return
    add(CALIBRATION, f"{plain or label or 'a reported interval'}: trials with no interval counted as misses", _sentence(
        f"Counting each trial that gave no interval as a miss, the {pct} intervals reported for {shown} contain the true "
        f"value in far fewer trials than {pct} in {where} ({listed}). FI judged the intervals on the trials that gave an "
        "estimate; the trials that gave none are stated apart"), key=f"counted-misses-{label}")


def _power_limit_item(root: Path, cal: dict[str, Any], add: Any) -> None:
    """The check of the reported standard errors passed, and the paper says how small a miss it could not have seen.

    The check compares the spread of the estimates over the trials with the standard error the trials report; with a few
    trials a setting it flags only a gross miss, so a standard error that is too small by less than the factor the trials
    allow (``detects_from`` in the record, :func:`core.trial_checks.ratio_detection_floor`) passes it. A passed check is
    not a proof of a right standard error, and the paper does not claim one. Said once for the worst (largest) limit of
    the standard errors that passed; nothing when one is flagged (that has its own item), or the record has no limit."""
    rows = [q for q in cal.get("quantities") or []
            if isinstance(q, dict) and q.get("kind") == "se_ratio" and not q.get("fault") and not q.get("below")
            and isinstance(q.get("detects_from"), (int, float)) and q.get("judged")]
    if not rows:
        return
    worst = max(rows, key=lambda q: float(q["detects_from"]))
    factor = float(worst["detects_from"])
    plain = _label_of_names(root, worst.get("names") or [], "")
    label = _one(worst.get("label"))
    shown = f'"{plain}"' if plain else f"`{label}`"
    many = len(rows) > 1
    add(CALIBRATION, "power of the standard-error check", _sentence(
        f"The standard error{'s' if many else ''} reported for {shown}{' and others' if many else ''} agreed with the spread "
        f"of the estimates over the trials, but with this many trials the check could only have flagged one too small by "
        f"a factor of about {factor:.2g} or more; a smaller error in the standard error is not ruled out"), key="se-power")


def _failed_where(root: Path, most: int = 4) -> str:
    """`` (key: failed of planned; ...)``: the settings where trials failed, most first, from FI's check of the trials
    (core/trial_checks.py; trials the time limit cut off are not in it). ``""`` when it has none."""
    from . import trial_checks as _trial_checks

    rows = [f for f in _trial_checks.read_record(root).get("failures") or [] if isinstance(f, dict) and f.get("failed")]
    if not rows:
        return ""
    rows.sort(key=lambda f: -int(f.get("failed") or 0))
    shown = "; ".join(f"{f.get('key') or 'the one setting'}: {f.get('failed')} of {f.get('planned')}" for f in rows[:most])
    return f" ({shown}" + (f"; and {len(rows) - most} more settings" if len(rows) > most else "") + ")"


def _left_out_setting_items(root: Path, add: Any) -> None:
    """The settings FI's check of the trials left out (most of their trials failed and the repairs were spent;
    core/trial_checks.py), each by its setting; more than :data:`_MAX_SETTING_ITEMS` are one item."""
    from . import trial_checks as _trial_checks

    left = _trial_checks.left_out_settings(root)

    def said(f: dict[str, Any]) -> str:
        reason = next((_whole(r[0], 160) for r in f.get("reasons") or [] if isinstance(r, list) and r), "")
        return f"{f.get('failed')} of {f.get('planned')} trials failed" + (f" ({reason})" if reason else "")

    if len(left) > _MAX_SETTING_ITEMS:
        add(SETTING_LEFT_OUT, "settings left out",
            _sentence(f"{len(left)} settings are left out of the results because most of their trials failed: "
                      + "; ".join(f"{f['key']}: {said(f)}" for f in left[:12])), key="all")
        return
    for f in left:
        add(SETTING_LEFT_OUT, str(f["key"]) or "the one setting",
            _sentence(f"The setting {f['key']} is left out of the results: {said(f)}"), key=f["key"])


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


_search_seconds = _run_facts.search_seconds


def measured_run_time(quest_root: Path | str, fallback_s: Any = None) -> tuple[float | None, float | None]:
    """``(wall seconds of the run that produced the stored trials, the least the trials and the search took)``, each ``None`` when FI
    does not know it. The wall time is the record (``core/run_facts.py``); a study without FI's trials has none but
    ``fallback_s``, the time of its last execution. A quest whose trials were run before the record existed has only the
    least time, from the ledger: the time of the last execution is not the experiment's."""
    root = Path(quest_root)
    wall = _run_facts.run_seconds(root)
    least = _ledger_trial_seconds(root)
    searched = _search_seconds(root)
    if searched:
        least = round((least or 0.0) + searched, 3)  # the search's own time counts: the scan alone is minutes of hours
    if wall is None and least is None and not isinstance(_run_facts.read(root).get("trials_run"), dict):
        wall = _run_facts.run_seconds(root, fallback_s)
    return wall, least


def _plain(seconds: float) -> str:
    """A duration in plain units: ``121.3 s``, ``223.2 s (about 3.7 minutes)``, ``9,000 s (about 2.5 hours)``."""
    said = _disclosure._plain_seconds(seconds)
    if seconds >= 2 * 3600:
        return f"{said} (about {seconds / 3600:.1f} hours)"
    if seconds >= 180:
        return f"{said} (about {seconds / 60:.1f} minutes)"
    return said


# ---- the facts ------------------------------------------------------------------------------------------------------


def _drew_record(root: Path) -> bool | None:
    """Whether the trials drew random numbers, from what the frame measured and the harness wrote into each trial's row
    (``drew``, core/skeleton.py): ``True`` when every trial that ran and carries the record drew, ``False`` when none did, ``None``
    when no row carries it (a quest from before it was kept) or the trials differ."""
    seen: set[bool] = set()
    for path in sorted((root / ".fi" / "trials").glob("*.jsonl")):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if '"drew"' not in line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("status") == "ok" and isinstance(row.get("drew"), bool):
                        seen.add(row["drew"])
        except (OSError, ValueError):
            return None
        if len(seen) > 1:
            return None
    return seen.pop() if seen else None


def seed_meaning_text(root: Path) -> str:
    """What the seed record means for the intervals, said to the writer as FI's fact (``""`` when the records do not show
    it): trials that each drew from their own seed make the interval over trials include the variation between seeds, and
    trials that share one seed, or never drew, do not. Without it the writer takes "one base seed" for "one seed" and
    types that the intervals leave seed-to-seed variation out."""
    study = _read(root, "code/study.json")
    if isinstance(study, dict) and study.get("entry") == "run_cell":
        return ""
    base, trial_seeds = _disclosure.recorded_seeds(root)
    drew = _drew_record(root)
    if drew is False:
        return ("No trial drew random numbers, so the results do not depend on a seed and the intervals are not about "
                "seed variation.")
    verdict = _run_facts.seed_verdict(root)
    state = verdict.get("verdict")
    if state == _run_facts.NO_ENTRY:
        return ""
    if len(trial_seeds) > 1 and (state == _run_facts.REACHES or (state == _run_facts.UNKNOWN and drew is True)):
        origin = f" derived from base seed {base}" if base is not None else ""
        return (f"Each of the trials ran on its own seed{origin}; the trial seeds are all different. The intervals over "
                "the trials therefore include the variation between seeds. Do not call the study single-seed or say "
                "the intervals leave seed-to-seed variation out.")
    if len(trial_seeds) == 1 and state in (_run_facts.REACHES, _run_facts.UNKNOWN):
        return ("Every trial ran on the same seed, so the trials do not vary the seed and the intervals do not include "
                "seed-to-seed variation.")
    return ""


def _trial_counts(root: Path) -> tuple[str, str, str]:
    """``(trials per setting, settings, trials failed)`` from FI's ledger, each ``""`` when there is none."""
    rows = _disclosure._ledger_rows(root)
    if not rows:
        return "", "", ""
    planned, trials = rows
    cells = sorted({cell for cell, _t in trials} | set(planned))
    counts: dict[str, int] = {}
    for c, _t in trials:
        counts[c] = counts.get(c, 0) + 1
    per = sorted(set(planned.values())) or sorted({counts.get(cell, 0) for cell in cells})
    each = f"{per[0]:,}" if len(per) == 1 else f"{per[0]:,} to {per[-1]:,}"
    failed = max(0, sum(1 for status, _s in trials.values() if status == "failed") - _out_of_time(root))
    return each, f"{len(cells):,}", f"{failed:,}"


def _out_of_time(root: Path) -> int:
    """The trials the study's time limit cut off (FI's ledger records them as failed): they are not failures of the code
    (core/trial_checks.py keeps them apart, ``out_of_time``)."""
    from . import trial_checks as _trial_checks

    rows = _trial_checks.read_record(root).get("out_of_time") or []
    return sum(int(r.get("trials") or 0) for r in rows if isinstance(r, dict))


def _limit_and_met(root: Path, topic: str, wall: float | None, least: float | None,
                   reader: dict[str, Any] | None) -> tuple[float | None, str]:
    """``(the topic's run-time limit, whether the run kept to it)``. The delivered code runs the settings one after
    another: when FI ran them at the same time, that run's time (a lower bound) is judged, and "within" is said only when
    no lower bound is in question. ``""`` when FI cannot say."""
    limit = _disclosure.topic_limit(root, topic)
    if limit is None:
        return None, ""
    if reader is not None:
        return limit, _run_facts.reader_verdict(reader, limit)  # the same verdict the run record gives
    if wall is not None:
        return limit, "within" if wall <= limit else "over"
    return limit, "over" if least is not None and least > limit else ""


def interval_level_percent() -> str:
    """The percent level of every interval FI computes, as ``{{interval_level}}`` fills it (``95``): from the one level
    :data:`core.paper_values.INTERVAL_LEVEL`, so the paper's words and FI's intervals cannot differ."""
    from decimal import Decimal

    return format((Decimal(_values.INTERVAL_LEVEL) * 100).normalize(), "f")


def facts(quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> dict[str, Fact]:
    """The run's facts FI holds, by placeholder name (a fact FI does not hold is not in it). The paper gets them as one
    paragraph FI writes itself (:func:`run_facts_text`); these are what an older draft's ``{{run_time}}`` and the like are
    filled with."""
    root = Path(quest_root)
    out: dict[str, Fact] = {}
    out["interval_level"] = Fact("interval_level", interval_level_percent())
    wall, least = measured_run_time(root, fallback_s)
    reader_known = _run_facts.reader_run(root)
    stage, others = _run_facts.run_stage(root)
    if wall is not None:
        said = _plain(wall)
        if stage == "confirm":
            explore = (others.get("explore") or {}).get("wall_s")
            said += (" for the confirming run, whose results are reported"
                     + (f" (the exploring run took {_plain(float(explore))})"
                        if isinstance(explore, (int, float)) and not isinstance(explore, bool) else ""))
        elif stage == "explore":
            said += " for the exploring run, whose results are reported"
        if reader_known is not None:
            reader = reader_known
            said += (f", {_run_facts.reader_clause(reader)} (run one after another, as the delivered "
                     f"code does, the settings take {_run_facts.reader_words(reader, _plain)})")
        out["run_time"] = Fact("run_time", said)
    elif least is not None:
        what = ("the settings run, the search and its check at finer settings; the rest was not timed"
                if _search_seconds(root) else "the trials alone; the whole run was not timed")
        out["run_time"] = Fact("run_time", f"at least {_plain(least)} ({what})")
    limit, met = _limit_and_met(root, topic, wall, least, reader_known)
    if limit is not None:
        out["time_limit"] = Fact("time_limit", _plain(limit))
        if met:
            out["time_limit_met"] = Fact("time_limit_met", met)
    each, settings, failed = _trial_counts(root)
    for name, value in (("trials_per_setting", each), ("settings", settings), ("trials_failed", failed)):
        if value:
            out[name] = Fact(name, value)
    return out


# ---- the paragraph of the run's facts ---------------------------------------------------------------------------------


def _plural(n: str, word: str) -> str:
    """``n`` (a formatted count) with ``word`` in the right number: ``1 setting``, ``3 settings``."""
    return f"{n} {word}{'' if n == '1' else 's'}"


def _counts_sentences(root: Path) -> list[str]:
    each, settings, failed = _trial_counts(root)
    out: list[str] = []
    if settings and each:
        out.append(f"The study ran {_plural(settings, 'setting')}, {_plural(each, 'trial')} in each.")
    elif settings:
        out.append(f"The study ran {_plural(settings, 'setting')}.")
    elif each:
        out.append(f"Each setting ran {_plural(each, 'trial')}.")
    late = _out_of_time(root)
    cut = f"{late:,} {'was' if late == 1 else 'were'} cut off by the time limit" if late else ""
    if failed:
        said = "No trial failed" if failed == "0" else f"{_plural(failed, 'trial')} failed"
        out.append(f"{said}{'; ' + cut if cut else ''}.")
    return out


def _cells_run(root: Path) -> str:
    """How many settings FI's run record counts (``trials_run.cells``), else ``""``."""
    run = _run_facts.read(root).get("trials_run")
    n = run.get("cells") if isinstance(run, dict) else None
    return f"{n:,}" if isinstance(n, int) and not isinstance(n, bool) and n > 0 else ""


def _run_time_sentences(root: Path, wall: float | None, least: float | None,
                        reader: dict[str, Any] | None, settings: str) -> list[str]:
    """The measured run time as whole short sentences, from the same records as the ``{{run_time}}`` fact."""
    stage, others = _run_facts.run_stage(root)
    if wall is None:
        if least is None:
            return []
        if _search_seconds(root):
            return [f"The settings run, the search and its check at finer settings took at least {_plain(least)}.",
                    "The rest of FI's run was not timed."]
        return [f"The trials alone took at least {_plain(least)}.", "FI's whole run was not timed."]
    took = {"confirm": "the confirming run, whose results are reported, took",
            "explore": "the exploring run, whose results are reported, took"}.get(stage or "", "the run took")
    n = settings or _cells_run(root)
    sets = f"the {_plural(n, 'setting')}" if n else "the settings"
    if reader is None:
        lead = f"{took[:1].upper()}{took[1:]} {_plain(wall)}."
    elif reader["at_once"] == 1:
        lead = f"{sets[:1].upper()}{sets[1:]} ran one at a time; {took} {_plain(wall)}."
    elif reader["at_once"] is not None:
        lead = f"{sets[:1].upper()}{sets[1:]} ran {reader['at_once']} at a time; {took} {_plain(wall)}."
    elif reader["overlap"]:
        lead = f"Several of {sets} ran at the same time (how many was not counted); {took} {_plain(wall)}."
    else:
        lead = f"It was not recorded whether {sets} ran at the same time; {took} {_plain(wall)}."
    out = [lead]
    if stage == "confirm":
        explore = (others.get("explore") or {}).get("wall_s")
        if isinstance(explore, (int, float)) and not isinstance(explore, bool):
            out.append(f"The exploring run took {_plain(float(explore))}.")
    if reader is not None and reader["at_once"] != 1 and reader["least"] is not None:
        sum_said = ("The trials and the search add up to" if _search_seconds(root)
                    else "The trials' own computation adds up to")
        out.append(f"{sum_said} {_disclosure._plain_seconds(reader['least'])}.")
    return out


def _unit_words(conflict: dict[str, Any]) -> list[str]:
    """``[the result's unit, the unit asked]`` in words (a pure number is said so)."""
    return [("a pure number (no unit)" if str(conflict.get(k) or "").strip() in ("", "1") else str(conflict.get(k)).strip())
            for k in ("result", "asked")]


def unit_conflicts(quest_root: Path | str) -> list[dict[str, Any]]:
    """The questions the plan answers with a result in another unit than the question asks in, from the plan's own
    ``topic_map`` (``unit_conflict``): ``{"name", "conflict", "metric", "valued"}``. ``conflict`` is the record (a pair of units, or
    the old text of a plan written before the pair); ``valued``: FI holds a value for the result. The quality report reads
    it."""
    from . import topic_requirements as _topic_req

    root = Path(quest_root)
    req = _read(root, f".fi/{_topic_req.FILE}")
    protocol = plan_protocol(root)
    asked = [q for q in (req.get("quantities") if isinstance(req, dict) else None) or [] if isinstance(q, dict) and q.get("main") is True]
    if not asked or not isinstance(protocol, dict):
        return []
    plan = _topic_req.view({}, protocol)
    rec = _values.records_of(root, protocol)
    out: list[dict[str, Any]] = []
    for item in asked:
        name = " ".join(str(item.get("name") or "").split())
        conflict = next((m["unit_conflict"] for m in _topic_req.map_entries(req, plan, "quantities", item) if m.get("unit_conflict")), None)
        if not name or not conflict:
            continue
        mapped = _topic_req.mapping_of(req, plan, "quantities", item)
        metric = mapped[1] if mapped is not None and mapped[0] in ("ok", "judge") else ""
        valued = (isinstance(conflict, dict) and rec is not None and bool(rec.cells) and bool(metric) and metric not in rec.by_run
                  and (metric in rec.metrics() or metric in rec.answers))
        out.append({"name": name, "conflict": conflict, "metric": metric, "valued": valued})
    return out


def other_candidates(quest_root: Path | str) -> list[dict[str, str]]:
    """The questions for which the answers part named a better result than the plan's map (``other_candidate`` on the plan's
    ``topic_map`` entry, a derived result FI's record shows is worked out from the mapped one): ``{"name", "planned", "candidate"}``.
    The plan's map is what the paper answers by; this is read by the quality report alone."""
    from . import topic_requirements as _topic_req

    root = Path(quest_root)
    req = _read(root, f".fi/{_topic_req.FILE}")
    protocol = plan_protocol(root)
    asked = [q for q in (req.get("quantities") if isinstance(req, dict) else None) or [] if isinstance(q, dict) and q.get("main") is True]
    if not asked or not isinstance(protocol, dict):
        return []
    plan = _topic_req.view({}, protocol)
    out: list[dict[str, str]] = []
    for item in asked:
        name = " ".join(str(item.get("name") or "").split())
        entry = next((m for m in _topic_req.map_entries(req, plan, "quantities", item) if m.get("other_candidate")), None)
        if name and entry:
            out.append({"name": name, "planned": str(entry.get("by") or ""), "candidate": str(entry["other_candidate"])})
    return out


def printed_result_ids(paper_md: str) -> list[str]:
    """The ids of the plan's results the paper prints, in order of first use: the first word of each ``{{v:...}}`` key (a
    ``calc``, ``trend`` or ``comparison`` key names no result)."""
    out: list[str] = []
    for m in _values._PLACEHOLDER.finditer(_values.to_placeholders(paper_md or "")):
        head = re.match(r"\s*([A-Za-z_]\w*)", m.group(1))
        if head and head.group(1) not in out and head.group(1) not in ("calc", "trend", "comparison"):
            out.append(head.group(1))
    return out


def run_facts_text(quest_root: Path | str, *, fallback_s: Any = None) -> str:
    """The paragraph ``{{run_facts}}`` stands for, composed from FI's records: how many settings and trials, and the
    measured run time, each only when FI holds it. ``""`` when it holds none."""
    root = Path(quest_root)
    parts = _counts_sentences(root)
    wall, least = measured_run_time(root, fallback_s)
    _each, settings, _failed = _trial_counts(root)
    parts += _run_time_sentences(root, wall, least, _run_facts.reader_run(root), settings)
    return "**Run.** " + " ".join(parts) if parts else ""


# ---- the placeholders and their markers ------------------------------------------------------------------------------

#: Any name between double braces, with extra braces around it taken in (``{{{{seed}}}}``, ``{{ {seed} }}``).
_PLACEHOLDER = re.compile(r"\{\{[\s{]*(?!v:)([A-Za-z][^{}\n:]{0,59}?)\s*(?::\s*([^{}\n\s][^{}\n]{0,999}?))?[\s}]*\}\}")
#: ``{{figure:...}}`` as far as the first ``}}`` on its line, whatever is inside it: what is left of a figure mark after
#: :data:`_PLACEHOLDER` has filled the ones it can read.
_STRAY_FIGURE = re.compile(r"\{\{[\s{]*figure\s*:([^\n]*?)\}\}+", re.I)
#: The rest of a figure mark whose value held an unescaped ``}}``: text glued to the end of a filled or recorded mark (no space
#: between them) up to the next ``}}``. Text after a space is the paper's own prose and is never touched.
_MARK_TAIL = re.compile(r"(<!-- /fi:fact-figure-[\w-]+ -->)[^\s{}<][^\n{}<]*\}\}+")
_STRAY_PARTS = re.compile(r"\s*([\w-]+)(?:\.\w+)?\s*:\s*([\w-]+)\s*(?:@(.*))?\Z", re.S)
_MARK = re.compile(r"<!-- fi:fact-([\w-]+) -->(.*?)<!-- /fi:fact-\1 -->", re.S)
_TAG = re.compile(r"<!-- /?fi:fact-[\w-]+ -->|<!-- fi:limitations-heading -->[ \t]*\n?|<!-- /?fi:unconfirmed -->")
#: FI's mark beside a sentence its claim check could not confirm, as an older FI placed it: FI places none now (the
#: quality report lists such sentences), and every view of a paper takes an old one out (:func:`without_unconfirmed`).
_UNCONFIRMED = re.compile(r"<!-- fi:unconfirmed -->.*?<!-- /fi:unconfirmed -->", re.S)
_MARKS_NOTE = re.compile(r"<!-- fi:marks-note -->.*?<!-- /fi:marks-note -->(?:\r?\n)*", re.S)
_ITEM_KEY = "not_done-"
#: Where no reader sees a placeholder as text: a code block or inline code, math, an HTML comment (not FI's own markers).
_HIDDEN = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)[^\n]*\n.*?^[ \t]*(?P=fence)[^\n]*$"
    r"|^[ \t]*(?:```|~~~).*\Z"
    r"|<!--(?!\s*/?fi:(?:fact-|v\b)).*?-->"
    r"|(?P<ticks>`+)(?:(?!\n[ \t]*\n)[^`])+?(?P=ticks)"
    r"|\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)"
    r"|(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)",
    re.M | re.S)
#: A double brace right after a LaTeX command or a closing brace is LaTeX (``\textbf{{x}}``), not a placeholder.
_LATEX_BEFORE = re.compile(r"(?:\\[A-Za-z]+\*?\s*|[}^_])$")
#: ...but a name that is FI's own right after ``}``, ``^`` or ``_`` (``{{seed}}`` after ``snake_case`` or after another
#: placeholder) is still FI's: only a command makes it LaTeX there.
_LATEX_COMMAND_BEFORE = re.compile(r"\\[A-Za-z]+\*?\s*$")


_FIGURE_ARGS = re.compile(r"\A(.+)-args([0-9a-f]*)\Z")


def _figure_parts(key: str) -> tuple[str, str, str | None] | None:
    """``(stem, kind, args)`` of a figure marker key (``figure-<kind>-<stem>``, with ``-args<hex>`` after a claim's
    arguments; ``args`` is ``None`` without them), else ``None``. A kind FI does not know is accepted only with arguments."""
    if not key.startswith(_FIGURE_KEY):
        return None
    kind, _, rest = key[len(_FIGURE_KEY):].partition("-")
    args = None
    m = _FIGURE_ARGS.match(rest)
    if m:
        try:
            args, rest = bytes.fromhex(m.group(2)).decode("utf-8"), m.group(1)
        except ValueError:
            args = None
    known = kind in _figures.KINDS or kind in _claims.CLAIMS
    return (rest, kind, args) if rest and kind and (known or args is not None) else None


def _figure_key(key: str) -> str | None:
    """``<stem>:<kind>`` (``<stem>:<claim> @ <args>`` for a claim with arguments) of a figure marker key, else ``None``."""
    parts = _figure_parts(key)
    if parts is None:
        return None
    stem, kind, args = parts
    return f"{stem}:{kind}" + (f" @ {args}" if args is not None else "")


def _figure_claim_text(root: Path, stem: str, claim: str, args: str, plan: Any = None) -> str | None:
    """FI's sentence for ``claim`` on the figure ``stem`` with ``args``, from its call record of the drawing; ``None`` when
    it cannot be established (no record, a claim FI does not make, a series or panel the figure does not hold, a raw panel,
    a malformed argument). Never raises."""
    try:
        calls = _claims.load_calls(root, stem)
        got = _claims.evaluate(calls, claim, _claims.parse_args(args), _values.sig, plan=plan) if calls is not None else None
    except Exception:  # noqa: BLE001 -- a claim that cannot be evaluated is left out, never guessed
        return None
    return got[0] if got else None


def _figure_facts(root: Path) -> dict[str, str]:
    """The figure facts FI holds (:func:`core.figure_facts.facts`), only for file stems a marker can carry."""
    try:
        return {k: v for k, v in _figures.facts(root).items() if re.fullmatch(r"[\w-]+:\w+", k)}
    except Exception:  # noqa: BLE001 -- a record that cannot be read is a fact FI does not hold
        return {}


def _norm_name(raw: str) -> str:
    name = re.sub(r"[\s.\-]+", "_", raw.strip().strip("{} ").strip().lower())
    return _ALIASES.get(name, name)


def _key(name: str) -> str:
    """The marker key of a name that is no placeholder of FI's: the name as written, in the characters a marker holds."""
    return re.sub(r"[^\w-]+", "_", name).strip("_")[:60] or "unnamed"


def _hidden_spans(markdown: str) -> list[tuple[int, int]]:
    return [m.span() for m in _HIDDEN.finditer(markdown)]


def _inside(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def _is_item(name: str) -> bool:
    return name in ("not_done", "notdone", "not_done_item")


def without_unconfirmed(markdown: str) -> str:
    """``markdown`` without the marks an older FI put beside unconfirmed sentences, exactly as it was before they were
    placed."""
    return _UNCONFIRMED.sub("", markdown or "")


def without_marks_note(markdown: str) -> str:
    """``markdown`` without the note an older FI put under the title (``<!-- fi:marks-note -->`` block), with the blank
    lines after it: FI's words, never the writer's. FI places none now; a paper that still carries one loses it."""
    return _MARKS_NOTE.sub("", markdown or "")


def to_placeholders(markdown: str) -> str:
    """``markdown`` with every value FI filled in (between its markers) turned back into its placeholder: what the passage
    editor is given, so its edits are made on the placeholders and FI fills them again. FI's marks beside unconfirmed
    sentences and its note under the title, which an older FI placed, are taken out: FI places none now."""
    return _placeholders(without_marks_note(without_unconfirmed(markdown)))


def _placeholders(markdown: str) -> str:
    """:func:`to_placeholders` with FI's marks beside unconfirmed sentences kept as they are (what :func:`render` reads)."""
    def back(m: re.Match[str]) -> str:
        key = m.group(1)
        if key.startswith(_ITEM_KEY):
            return "{{not_done:" + key[len(_ITEM_KEY):] + "}}"
        figure = _figure_key(key)
        if figure is not None:
            return "{{figure:" + figure + "}}"
        return "{{" + key + "}}"

    return _MARK.sub(back, _values.to_placeholders(markdown or ""))


def without_marks(markdown: str) -> str:
    """``markdown`` with FI's markers (of its facts and of its filled result numbers) taken out and the values kept: what a
    reader, the slides and the poster get."""
    return _TAG.sub("", _values.without_marks(markdown or ""))


def block_text(quest_root: Path | str, name: str, *, fallback_s: Any = None, full: bool = False) -> str:
    """FI's block ``name`` (:data:`BLOCK_NAMES`) as it goes into the paper, from its file (``run_facts``: composed from
    FI's records, :func:`run_facts_text`); ``""`` when there is none. The tables, which FI sends to the supplement
    (core/supplement.py), are their pointer there, unless ``full`` (what the supplement holds)."""
    if not full and name == _supplement.TABLES and name in _supplement.moved(quest_root):
        return _supplement.table_pointer() if block_text(quest_root, name, full=True) else ""
    if name == "run_facts":
        return run_facts_text(quest_root, fallback_s=fallback_s)
    rel = _BLOCK_FILES.get(name)
    if rel is None:
        return ""
    try:
        return (Path(quest_root) / rel).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


#: The line an older FI put above the heading (or run-in line) of its Limitations block, so the heading goes with it.
HEADING_MARK = "<!-- fi:limitations-heading -->"
_CUT = "\x00cut\x00"


def _limitations_in(text: str) -> bool:
    """Whether the Limitations block an older FI wrote is in ``text``: its marked block, or its placeholder outside code."""
    if any(m.group(1) == "limitations" for m in _MARK.finditer(text)):
        return True
    hidden = _hidden_spans(text)
    return any(_norm_name(m.group(1)) == "limitations" and m.group(2) is None and not _inside(m.start(), hidden)
               for m in _PLACEHOLDER.finditer(text))


def _drop_limitations(text: str) -> str:
    """``text`` without FI's Limitations block (marked, or the placeholder on its own), and without the heading or the
    run-in line FI put above it when nothing else is under it."""
    text = _MARK.sub(lambda m: _CUT if m.group(1) == "limitations" else m.group(0), text)
    hidden = _hidden_spans(text)
    for a, b in reversed([m.span() for m in _PLACEHOLDER.finditer(text)
                          if _norm_name(m.group(1)) == "limitations" and m.group(2) is None
                          and not _inside(m.start(), hidden)]):
        text = text[:a] + _CUT + text[b:]
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == HEADING_MARK:
            # FI's own heading: it goes with the block when nothing else is under it; else the writer has put words
            # there and it is the writer's heading now, so only the mark goes
            nxt = next((x for x in lines[i + 2:] if x.replace(_CUT, "").strip()), None)
            if nxt is None or re.match(r"^#{1,6}[ \t]", nxt) or nxt.strip() == "---":
                i += 2
            else:
                i += 1
            out.append(_CUT)
            continue
        out.append(lines[i])
        i += 1
    return _tidy_cuts(_without_emptied_headings(out))


_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]")


def _without_emptied_headings(lines: list[str]) -> list[str]:
    """``lines`` without the heading of a section the cut left empty (nothing but the cut and blank lines before the next
    heading of its level or above, a rule, or the end): a paper never keeps an empty section heading. A section with
    any other line, or a deeper heading, under it keeps its heading; code blocks are not read."""
    fenced, heads = False, []
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and (m := _HEADING_LINE.match(ln)):
            heads.append((i, len(m.group(1))))
    out = list(lines)
    for i, level in heads:
        cut = False
        for ln in lines[i + 1:]:
            rest = ln.replace(_CUT, "").strip()
            if _CUT in ln:
                cut = True
            if not rest:
                continue
            m = _HEADING_LINE.match(rest)
            if (m and len(m.group(1)) <= level) or rest == "---":
                break
            cut = False  # a line of text or a deeper heading: the section is not empty
            break
        if cut:
            out[i] = _CUT
    return out


def _tidy_cuts(lines: list[str]) -> str:
    """The lines joined, with each run of blank lines that touches a cut made to one blank line (none at the top): the
    rest of the paper, code blocks and tables included, is not touched."""
    lines = [ln.replace(_CUT, "") if ln.replace(_CUT, "").strip() else ln for ln in lines]
    done: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() and _CUT not in lines[i]:
            done.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and (not lines[j].strip() or _CUT in lines[j]):
            j += 1
        group = lines[i:j]
        if any(_CUT in g for g in group):
            if done:
                done.append("")
        else:
            done.extend(group)
        i = j
    return "\n".join(done)


#: The blocks only FI places: a draft's placeholder or filled block of one of these is taken out wherever it stands (a table
#: cell, the reference list, a second copy) and FI puts exactly one in its own place (:func:`place_blocks`).
FI_PLACED = ("asked_results", "quality_pointer")


def strip_block(markdown: str, name: str) -> str:
    """``markdown`` without any placeholder ``{{name}}`` and any filled block of ``name``, wherever they stand; a placeholder
    or block that fills a line of its own goes with that line and the blank line after it."""
    mark = r"<!-- fi:fact-" + re.escape(name) + r" -->.*?<!-- /fi:fact-" + re.escape(name) + r" -->"
    holder = r"\{\{[\s{]*" + re.escape(name) + r"[\s}]*\}\}"
    out = markdown or ""
    for pat in (mark, holder):
        out = re.sub(r"(?m)^[ \t]*(?:" + pat + r")[ \t]*\r?\n(?:[ \t]*\r?\n)?", "", out, flags=re.S)
        out = re.sub(pat, "", out, flags=re.S)
    return out


def place_blocks(markdown: str, quest_root: Path | str, *, fallback_s: Any = None) -> str:
    """``markdown`` with the placeholder of each of FI's blocks that has text and that the writer did not place put in
    on a line of its own. The tables go before the discussion that follows the results (else the conclusions, the
    limitations, the references; else at the end); the run's facts go at the end of the Methods (else before the
    Results, else after the abstract, else at the top). A block already in the draft, as a placeholder or filled in
    (the review sends FI's own text back), is not put in twice; a line that opens like FI's run paragraph
    (``**Run.** ``) but has lost its markers is replaced by the placeholder, so the paragraph is always FI's current
    one. The Limitations block an older FI wrote is taken out, with the heading FI put above it
    (:func:`_drop_limitations`). :func:`render` then fills the rest."""
    text = markdown or ""
    if _limitations_in(text):
        # the writer's own paragraphs an older FI moved to the supplement with this block come back first: they are the
        # writer's text, not FI's, and the block alone is what goes
        text = _drop_limitations(_supplement.with_writer_paragraphs(text, quest_root))
    for name in BLOCK_NAMES:
        if name in FI_PLACED:  # FI's own place only: every earlier one, wherever it stands, goes first
            text = strip_block(text, name)
        if not block_text(quest_root, name, fallback_s=fallback_s):
            continue
        if any(_norm_name(m.group(1)) == name for m in _PLACEHOLDER.finditer(text)) or any(
                m.group(1) == name for m in _MARK.finditer(text)):
            continue
        if name == "run_facts" and re.search(r"(?m)^\*\*Run\.\*\* ", text):
            # FI's paragraph with its markers gone (or a run line of the writer's own): FI's current one takes its place
            text = re.sub(r"(?m)^\*\*Run\.\*\* .*$", "{{run_facts}}", text, count=1)
            continue
        if name == "run_facts":
            lines = text.split("\n")
            at = _disclosure._insert_at(lines)
            head = lines[:at]
            while head and not head[-1].strip():
                head.pop()
            tail = lines[at:]
            while tail and not tail[0].strip():
                tail.pop(0)
            text = "\n".join([*head, *([""] if head else []), "{{run_facts}}", *([""] if tail else []), *tail]) + (
                "" if tail else "\n")
            continue
        if name == "asked_results":  # at the end of the section that reports the results
            text = _asked.with_block(text)
            continue
        if name == "quality_pointer":  # the one closing sentence, at the end of the body, before the reference list
            text = _pointer.with_block(text)
            continue
        from . import best_design_report as _best

        text = _best.insert_before_discussion(text, ["{{" + name + "}}"])
    return text


def page_limit_text(quest_root: Path | str) -> str:
    """The sentence for the quality report when the finished paper is longer than the topic's page limit, and the ones
    that say which of FI's own parts are in the supplement (core/supplement.py): its tables of every setting and its
    Further reading always (and, in a record an older FI wrote, the paper's Limitations paragraphs for the page limit);
    ``""`` with neither. The limit is a guideline: the
    paper does not carry these sentences (QUALITY.md and run.log do)."""
    pages = _read(Path(quest_root), str(PAGE_RECORD))
    over = ""
    if (isinstance(pages, dict) and pages.get("final") and isinstance(pages.get("pages"), int)
            and isinstance(pages.get("limit"), int) and pages["pages"] > pages["limit"]):
        over = _sentence(f"This paper is {pages['pages']} pages, over the {pages['limit']}-page limit set for it")
    moved = _supplement.moved(quest_root)
    if not moved or not (Path(quest_root) / _supplement.FILE).is_file():
        return over
    names = {_supplement.TABLES: "its tables of every setting", _supplement.LIMITATIONS: "the paper's own Limitations "
             "paragraphs", _supplement.FURTHER_READING: "its Further reading"}

    def listed(parts: list[str]) -> str:
        said = [names[m] for m in parts]
        return said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]

    always = [m for m in moved if m in _supplement.ALWAYS]
    for_limit = [m for m in moved if m not in _supplement.ALWAYS]
    kept = _sentence(f"FI keeps {listed(always)} in the supplement, `{_supplement.SHOWN}`, unchanged, as it does "
                     "for every paper") if always else ""
    limit = (_sentence(f"For the page limit, FI moved {listed(for_limit)} there too") if always else
             _sentence(f"For the page limit, FI moved {listed(for_limit)} to the supplement, `{_supplement.SHOWN}`, "
                       "unchanged")) if for_limit else ""
    return " ".join(s for s in (over, kept, limit) if s)


def write_page_record(quest_root: Path | str, pages: int, limit: int, *, final: bool) -> None:
    """Keep FI's measurement of the paper against the page limit (:data:`PAGE_RECORD`)."""
    path = Path(quest_root) / PAGE_RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pages": int(pages), "limit": int(limit), "final": bool(final)}) + "\n",
                        encoding="utf-8")
    except OSError:
        pass


def without_blocks(markdown: str) -> str:
    """``markdown`` with each of FI's blocks (:data:`BLOCK_NAMES`, and those an older FI wrote) taken out, markers and
    all: what the checks of the writer's own text read (the block is FI's record, printed, not a claim of the writer's).
    The marks and the note an older FI put in are taken out too: they are FI's words, not the writer's."""
    names = "|".join(re.escape(n) for n in (*BLOCK_NAMES, *_RETIRED_BLOCKS))
    return re.sub(r"\n?<!-- fi:fact-(" + names + r") -->.*?<!-- /fi:fact-\1 -->\n?", "\n",
                  without_marks_note(without_unconfirmed(markdown)), flags=re.S)


def mask(markdown: str, token: str = " FI-FACT ") -> str:
    """``markdown`` with each value FI filled in replaced by ``token``: what the writer typed, and nothing of FI's."""
    return _MARK.sub(token, _PLACEHOLDER.sub(token, _values._MARK.sub(token, markdown or "")))


def render(markdown: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None,
           records: _values.Records | None = None, plan: Any = None) -> Rendered:
    """``markdown`` with every placeholder (and every value FI filled in before, so a re-render takes the records as they
    are now) filled from FI's records, between markers. A placeholder FI cannot fill reads "not recorded" and is in
    ``unfilled``; an unknown name that looks like a placeholder (``{{run_seconds}}``, ``{{not_done:x}}``) is treated the
    same. Any other name between double braces is no placeholder of FI's: it reads "not recorded" between markers, so it
    is listed in QUALITY.md and the check asks for one of FI's names. Extra braces around a name are taken in.

    Inside math, code or an HTML comment, a placeholder of FI's gets its plain value with no markers (a marker there would
    print as text), and anything else between double braces is left as written (LaTeX's ``{{x}}``); so is a double brace
    right after a LaTeX command.

    A result number is a key ``{{v:...}}`` (:mod:`core.paper_values`): FI fills it from ``records`` (the quest's trial record
    and plan) after the facts, between its own marker; a key it cannot fill reads "not recorded" and is in ``values`` with
    the reason. ``records=None`` fills every key as not recorded."""
    root = Path(quest_root)
    known = facts(root, topic=topic, fallback_s=fallback_s)
    items = {i.id.lower(): i for i in required_items(root)}
    figure_known = _figure_facts(root)
    unfilled: list[str] = []
    # A {{calc:...}} is a key FI fills with the result numbers below, not a fact of the run.
    text = _values.keyed_calcs(_placeholders(markdown))  # FI's marks beside unconfirmed sentences stay as they are
    hidden = _hidden_spans(text)

    def fill(m: re.Match[str]) -> str:
        raw, arg = m.group(1), m.group(2)
        name = _norm_name(raw)
        plain = _inside(m.start(), hidden)
        before = text[max(0, m.start() - 40):m.start()]
        mine = (_is_item(name) or arg is not None and name.startswith("not") or name in BLOCK_NAMES
                or name in _RETIRED_BLOCKS or name in FACT_NAMES or name == "figure" and arg is not None)
        if _LATEX_COMMAND_BEFORE.search(before) if mine else _LATEX_BEFORE.search(before):
            return m.group(0)
        if _is_item(name) or arg is not None and name.startswith("not"):
            ident = (arg or "").strip()
            item = items.get(ident.lower())
            if plain:
                return item.sentence if item else m.group(0)
            key = _ITEM_KEY + (item.id if item else _slug(ident))
            if item is None:
                unfilled.append(f"not_done:{_slug(ident)}")
            return f"<!-- fi:fact-{key} -->{item.sentence if item else NOT_RECORDED}<!-- /fi:fact-{key} -->"
        if name == "figure" and arg is not None:
            head, at, tail = (arg or "").partition("@")
            stem, _, kind = head.strip().rpartition(":")
            stem, kind, args = stem.strip(), kind.strip().lower(), tail.strip()
            stem = re.sub(r"\.(?:png|jpe?g|svg|pdf|gif|webp|tiff?)\Z", "", stem, flags=re.I)  # the file name as typed
            if at:  # a claim with arguments: worded by FI from its record of the drawing, or not at all
                text_ = _figure_claim_text(root, stem, kind, args, plan) if re.fullmatch(r"[\w-]+", stem) else None
            else:
                text_ = figure_known.get(f"{stem}:{kind}")
                if text_ is None and kind in _claims.CLAIMS and kind not in _figures.KINDS:
                    text_ = _figure_claim_text(root, stem, kind, "", plan) if re.fullmatch(r"[\w-]+", stem) else None
            if text_ is None:
                unfilled.append(f"figure:{stem}:{kind}")
            if plain:
                return text_ if text_ else m.group(0)
            if at:  # always a figure key, so that a statement FI cannot check is found and taken out
                kind_k = re.sub(r"\W+", "_", kind) or "claim"
                stem_k = re.sub(r"[^\w-]+", "_", stem) or "figure"
                key = f"{_FIGURE_KEY}{kind_k}-{stem_k}-args{args.encode('utf-8').hex()}"
            elif re.fullmatch(r"[\w-]+", stem) and (kind in _figures.KINDS or kind in _claims.CLAIMS):
                key = f"{_FIGURE_KEY}{kind}-{stem}"
            else:
                key = _key(f"figure_{stem}_{kind}")
            return f"<!-- fi:fact-{key} -->{text_ or NOT_RECORDED}<!-- /fi:fact-{key} -->"
        if name in BLOCK_NAMES:
            if plain:
                return m.group(0)
            block = block_text(root, name, fallback_s=fallback_s)
            if name == "asked_results":  # a table of FI's with no row left leaves no trace; its number fits the paper's
                if not block:
                    return ""
                block = block.replace(_asked.NUMBER, str(_asked.number_at(text, m.start())))
            if not block and name == "quality_pointer":  # no case, no sentence, and no trace of one a round put in before
                return ""
            if not block:
                unfilled.append(name)
                return f"<!-- fi:fact-{name} -->{NOT_RECORDED}<!-- /fi:fact-{name} -->"
            return f"<!-- fi:fact-{name} -->\n{block}\n<!-- /fi:fact-{name} -->"
        if name in _RETIRED_BLOCKS:
            return m.group(0) if plain else ""  # an older FI's block: nothing in its place, and no "not recorded"
        if name in FACT_NAMES:
            fact = known.get(name)
            if fact is None:
                unfilled.append(name)
            value = fact.text if fact else NOT_RECORDED
            return value if plain else f"<!-- fi:fact-{name} -->{value}<!-- /fi:fact-{name} -->"
        if plain:
            return m.group(0)
        key = _key(name)
        unfilled.append(key)
        return f"<!-- fi:fact-{key} -->{NOT_RECORDED}<!-- /fi:fact-{key} -->"

    out = _PLACEHOLDER.sub(fill, text)
    # A figure mark the pattern above could not read (a brace inside it, a name too long) is still a mark and never prints:
    # it is an unfilled figure statement, so it is taken out with its sentence and listed like any other.
    stray_hidden = _hidden_spans(out)

    def stray(m: re.Match[str]) -> str:
        if _inside(m.start(), stray_hidden) or _LATEX_COMMAND_BEFORE.search(out[max(0, m.start() - 40):m.start()]):
            return m.group(0)
        parts = _STRAY_PARTS.match(m.group(1))
        stem, kind, args = (parts.group(1), parts.group(2), (parts.group(3) or "").strip()) \
            if parts else ("figure", "claim", m.group(1).strip())
        unfilled.append(f"figure:{stem}:{kind}")
        kind_k = re.sub(r"\W+", "_", kind.lower()) or "claim"
        key = f"{_FIGURE_KEY}{kind_k}-{stem}-args{args.encode('utf-8').hex()}"
        return f"<!-- fi:fact-{key} -->{NOT_RECORDED}<!-- /fi:fact-{key} -->"

    out = _STRAY_FIGURE.sub(stray, out)
    # What an unescaped brace left of a mark's arguments (``...a}}b, panel=1}}``) is part of the mark, not of the paper.
    out = _MARK_TAIL.sub(lambda m: m.group(1), out)
    # A placeholder for a whole sentence followed by the writer's own full stop: one full stop.
    out = re.sub(r"([.!?])(<!-- /fi:fact-[\w-]+ -->)\.", r"\1\2", out)
    # The result numbers go in last, from the trial record: a key inside math or code gets its bare value.
    keyed_hidden = _hidden_spans(out)
    def math_span(pos: int) -> tuple[int, int] | None:
        return next((sp for sp in keyed_hidden if sp[0] <= pos < sp[1] and out[sp[0]] in "$\\"), None)

    out, values = _values.fill_values(out, records, plain=lambda pos: _inside(pos, keyed_hidden), span_of=math_span)
    return Rendered(out, tuple(dict.fromkeys(unfilled)), tuple(values))


def unfilled_in(markdown: str) -> list[str]:
    """The placeholders and result keys of a rendered paper that read "not recorded"."""
    out = []
    for m in _MARK.finditer(markdown or ""):
        if m.group(2).strip() == NOT_RECORDED:
            key = m.group(1)
            out.append("not_done:" + key[len(_ITEM_KEY):] if key.startswith(_ITEM_KEY) else key)
    out += ["v:" + k for k in _values.unfilled_in(markdown)]
    return list(dict.fromkeys(out))


# ---- what the writer is given ---------------------------------------------------------------------------------------


def writer_section(quest_root: Path | str, *, fallback_s: Any = None) -> str:
    """The record section of the write prompt: the paragraph of the run's facts FI puts into the Methods (what it reads
    now), the tables FI writes, and what FI's records say the writer must not claim. ``""`` when FI holds nothing."""
    root = Path(quest_root)
    run_text = run_facts_text(root, fallback_s=fallback_s)
    lines: list[str] = []
    estimate = _read(root, ".fi/run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    reduced_note = (_one(reduced["note"]) if isinstance(reduced, dict) and isinstance(reduced.get("note"), str)
                    and reduced["note"].strip() else "")
    has_table = bool(block_text(root, "results_table"))
    over = estimate.get("over_topic_limit") if isinstance(estimate, dict) else None
    over_note = _one(over["note"]) if isinstance(over, dict) and isinstance(over.get("note"), str) and over["note"].strip() else ""
    figure_known = _figure_facts(root)
    known_answer = any(i.kind == KNOWN_ANSWER for i in required_items(root))
    if not run_text and not reduced_note and not over_note and not has_table and not figure_known and not known_answer:
        return ""
    lines.append("FI'S RECORD OF THIS RUN. The facts below come from FI's own records. FI fills them in; do not type "
                 "them yourself.")
    if run_text:
        lines.append("Run facts: FI puts a paragraph of the run's facts into the Methods when the paper is saved. It "
                     f"reads now: \"{run_text}\" Do not state these facts yourself (the number of settings and trials, "
                     "the run time), and do not refer to the paragraph: it is part of the Methods, in FI's own words. A "
                     "run time or a count of settings or trials typed as a number is sent back to you, and so is a claim "
                     "about the topic's time limit that the record contradicts.")
    if (seed_meaning := seed_meaning_text(root)):
        lines.append(f"Seed: {seed_meaning}")
    if figure_known:
        lines.append("A figure's panel layout and its line styles are in the caption FI writes under it, from FI's "
                     "record of the saved picture. You have not seen the picture: do not describe them in your own words.")
    if reduced_note:
        lines.append(f"Reduced size: {reduced_note} Say this in the paper, with the numbers.")
    if has_table:
        lines.append("{{results_table}}: FI's tables of every setting the study computed (each value with its 95% "
                     "interval beside the reference FI worked out from the plan). Put the placeholder on a line of its "
                     "own in the Results, where the tables belong; left out, FI puts it at the end of the Results. Discuss "
                     "the tables, but do not copy them into a table of your own.")
    if over_note:
        lines.append(f"Time limit: {over_note} Say this in the paper.")
    if known_answer:
        lines.append("Known answers: a check of the code against a known answer did not pass. Do not state in your own "
                     "sentence that the results or the code were validated, verified or confirmed against known answers.")
    lines.append("A placeholder FI cannot fill is printed as \"not recorded\", so use only the ones listed here.")
    return "\n".join(lines)


# ---- the check ------------------------------------------------------------------------------------------------------


def _unknown_hits(markdown: str, quest_root: Path | str) -> list[str]:
    """A hit for each name between double braces that is no placeholder of FI's (it reads "not recorded")."""
    ids = {i.id for i in required_items(quest_root)}
    hits: list[str] = []
    for key in dict.fromkeys(m.group(1) for m in _MARK.finditer(markdown)):
        if key.startswith(_ITEM_KEY):
            if key[len(_ITEM_KEY):] not in ids:
                listed = ", ".join("{{not_done:" + i + "}}" for i in sorted(ids))
                hits.append(f"{HIT}: {{{{not_done:{key[len(_ITEM_KEY):]}}}}} is not one of the parts FI recorded as not "
                            "done, so the paper reads \"not recorded\" there. "
                            + (f"Use one of {listed}, or take it out." if listed else "Take it out."))
        elif _figure_key(key) is not None:
            if _figure_parts(key)[1] not in _figures.KINDS:
                continue  # a claim FI cannot establish is left out and said once (take_out_unfilled), not sent back
            if _figure_key(key) not in _figure_facts(Path(quest_root)):
                ref = _figure_key(key)
                hits.append(f"{HIT}: {{{{figure:{ref}}}}} is not a figure FI has a record of, so the paper reads "
                            "\"not recorded\" there. Name a figure file as listed in FI's record, or describe "
                            "the figure without a placeholder.")
        elif key not in FACT_NAMES and key not in BLOCK_NAMES and key not in _RETIRED_BLOCKS:
            hits.append(f"{HIT}: {{{{{key}}}}} is not one of FI's placeholders, so the paper reads \"not recorded\" there. "
                        "Write it in your own words without double braces.")
    return hits


def placement_hits(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Must-fix hits for a name between double braces that is none of FI's. The run's facts are not asked of the writer
    any more: FI writes them as one paragraph in the Methods (:func:`run_facts_text`), so a missing one is no fault of the
    paper."""
    return _unknown_hits(_disclosure.without_block(paper_md), quest_root)


def value_hits(paper_md: str, quest_root: Path | str) -> list[str]:
    """A must-fix hit for each result key FI could not fill (it reads "not recorded"), naming the key, why, and the
    writer's sentence (as the passage editor locates it): the writer fixes the key or leaves the number out."""
    bare = to_placeholders(paper_md)
    reasons = _values.unavailable_reasons(quest_root)
    out: list[str] = []
    for key in _values.unfilled_in(paper_md):
        shown = _values.placeholder(key)
        at = bare.find(shown)
        why = reasons.get(key) or "FI has no record for it"
        sentence = _typed.sentence_around(bare, at, at + len(shown)) if at >= 0 else ""
        out.append(f"{HIT}: {shown} is not recorded ({why}). Fix the key so FI can fill it, or leave the number out."
                   + (f" \u2014 \u201c{sentence[:400]}\u201d" if sentence else ""))
    return out


#: A bold run-in heading that opens a paragraph (``**Power.** ...``): kept when the sentence after it is taken out.
_RUN_IN = re.compile(r"\A\s*\*\*[^*\n]+\*\*")
#: A line that is not running prose (a table row, heading, figure, list item, quote): a sentence in it is never cut out.
_NOT_PROSE = re.compile(r"^\s*(\||#|!\[|[-+*]\s|\d+[.)]\s|>)")
_TOKEN = re.compile(r"\x00\d+\x00")


def _whole_sentence(sentence: str) -> bool:
    """Whether ``sentence`` reads as a sentence by itself: it opens with a capital, a digit, a mark or a formula, ends with a
    full stop and closes every bracket it opens (so ``e.g.``, an abbreviation or a split inside brackets is not one)."""
    t = sentence.strip()
    return bool(t) and t.endswith(".") and (t[0].isupper() or t[0] in "*_$(0123456789") and all(
        t.count(a) == t.count(b) for a, b in ("()", "[]", "{}"))


_SENTENCE_START = re.compile(r"(?:\A|\n|[.!?]|\*\*)[ \t]*\Z")


def _starts_sentence(before: str) -> bool:
    """Whether the text ``before`` a mark ends a sentence, a line or a bold run-in heading: the mark opens its own sentence."""
    return bool(_SENTENCE_START.search(before))


_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)")


def _figure_number(markdown: str, stem: str) -> str:
    """The number of the figure whose image file is ``stem`` (the order of the paper's images), else the stem itself."""
    stems = [re.sub(r"\.\w+\Z", "", re.split(r"[/\\]", m.group(1))[-1]) for m in _IMAGE.finditer(markdown or "")]
    stems = list(dict.fromkeys(stems))
    return str(stems.index(stem) + 1) if stem in stems else stem


def _figure_reference_items(root: Path) -> list[tuple[str, str, str]]:
    """``(key, name, sentence)`` for each reference the plan gives for a result a figure draws and the figure does not
    draw (:func:`core.figure_text_check.quest_missing_references`: FI's record of the drawing against the frozen protocol). The
    sentence is FI's own; a figure with no record, or a study with no protocol, gives none."""
    try:
        from . import figure_text_check as _ftc

        found = _ftc.quest_missing_references(root, plan_protocol(root))
    except Exception:  # noqa: BLE001 -- a check that cannot be made names nothing
        return []
    out = []
    for f in found:
        where = f"Figure {f['number']}" if f.get("number") else f"The figure {Path(str(f['file'])).stem.replace('_', ' ')}"
        how = f" ({f['formula']})" if f.get("formula") else ""
        out.append((f"{Path(str(f['file'])).stem}:{f['result']}", f"{where} reference for {f['label']}",
                    _sentence((f"{where} draws a line that FI did not take from the plan, so FI cannot confirm it is the plan's reference for "
                     f"{_one(f['label'])}{how}" if f.get("other_line") else
                     f"{where} draws no line for the plan's reference for {_one(f['label'])}{how}"))))
    return out


def figure_left_out_sentence(figure: str) -> str:
    return (f"Figure {figure}: some statements about this figure could not be checked against what it draws and were "
            "left out.")


def _figure_left_out_items(taken: Any) -> list[tuple[str, str]]:
    """``(figure stem, figure number)`` once per figure that has a statement taken out (``take_out_unfilled``'s entries)."""
    found: dict[str, str] = {}
    for t in taken if isinstance(taken, list) else []:
        key = t.get("key") if isinstance(t, dict) else None
        if isinstance(key, str) and key.startswith("figure:") and t.get("figure"):
            found.setdefault(key.split(":")[1], str(t["figure"]))
    return list(found.items())


def take_out_unfilled(markdown: str, *, figures_only: bool = False) -> tuple[str, list[dict[str, str]]]:
    """``markdown`` without the result keys FI could not fill (they read "not recorded"), for a paper no round is left to
    rewrite: ``(text, taken_out)``, ``taken_out`` having ``key``, ``sentence`` (as it read) and ``how`` for each.

    The whole unit that holds an unfilled result key goes, so no stub is left where a number stood: in running text its
    sentence (a bold run-in heading before it stays; its filled values go with it), anywhere else its line (a table row, a
    list item, a heading, a caption). A table left with no row of values goes whole, and so does one whose header or rule
    held the key. Nothing is filled with a guess. With ``figures_only`` only the figure statements are looked at (the
    result keys are left for the round that takes them out)."""
    # A figure statement is a mark of the second shape (<!-- fi:fact-figure-... -->): a whole sentence FI wrote or, unfilled, none.
    marks = sorted([*([] if figures_only else _values._MARK.finditer(markdown or "")),
                    *(m for m in _MARK.finditer(markdown or "") if _figure_parts(m.group(1)) is not None)],
                   key=lambda m: m.start())
    if not any(m.group(2).strip() == _values.NOT_RECORDED for m in marks):
        return markdown, []
    is_figure = [m.re is _MARK for m in marks]
    full_stop = [False] * len(marks)  # whether the token carries a full stop of its own
    # The text with each mark made one token, so a sentence is found as it reads: (token start, end, original start, end).
    parts: list[str] = []
    spans: list[tuple[int, int, int, int]] = []
    last = size = 0
    for i, m in enumerate(marks):
        before = markdown[last:m.start()]
        token = f"\x00{i}\x00"
        if is_figure[i] and _starts_sentence(markdown[:m.start()]) and not markdown[m.end():m.end() + 1] in (".", "!", "?"):
            token += "."  # a figure statement of its own (not inside the writer's sentence) has its full stop
        full_stop[i] = token.endswith(".")
        parts += [before, token]
        spans.append((size + len(before), size + len(before) + len(token), m.start(), m.end()))
        size += len(before) + len(token)
        last = m.end()
    parts.append(markdown[last:])
    flat = "".join(parts)

    def original(p: int) -> int:  # a position of the token text that is outside every token
        shift = 0
        for _bs, be, _os, oe in spans:
            if be <= p:
                shift = oe - be
        return p + shift

    groups: dict[tuple[int, int], list[int]] = {}
    for i, m in enumerate(marks):
        if m.group(2).strip() == _values.NOT_RECORDED:
            groups.setdefault(_typed._sentence(flat, spans[i][0], spans[i][1] - full_stop[i]), []).append(i)
    edits: list[tuple[int, int]] = []
    gone_rows: set[int] = set()  # the lines that go whole (by number)
    rows_of: list[tuple[dict[str, str], int]] = []  # each result key's entry and the line that went with it (-1: none)
    taken: list[dict[str, str]] = []
    for (a, b), idx in groups.items():
        if b > a and flat[b - 1] == "\n":
            b -= 1  # the sentence ends its line: the line break is not its own
        inside = [i for i, sp in enumerate(spans) if a <= sp[0] and sp[1] <= b]
        line_start = flat.rfind("\n", 0, a + 1) + 1
        line_end = flat.find("\n", b)
        line = flat[line_start:line_end if line_end != -1 else len(flat)]
        sentence = re.sub(r"\s+", " ", _values.without_marks(markdown[original(a):original(b)])).strip()
        lead = _RUN_IN.match(flat[a:b])
        body_at = a + (lead.end() if lead else 0)
        # A bold run-in heading followed by nothing but the unfilled key and its punctuation has no sentence to keep.
        key_only = bool(lead) and not re.sub(r"[\s.,;:!?()\-\u2013\u2014]+", "", _TOKEN.sub("", flat[body_at:b]))
        # A figure statement is a whole sentence by itself, so it stands for one that opens with a capital.
        stand_in = lambda t: "X" if is_figure[int(t.group(0).strip("\x00"))] else "x"  # noqa: E731
        drop = (all(i in idx for i in inside if spans[i][0] >= body_at) and not _NOT_PROSE.match(line)
                and (key_only or _whole_sentence(_TOKEN.sub(stand_in, flat[body_at:b]))))
        # A result key FI could not fill that is not the whole of a sentence still goes with its unit: the sentence in
        # running text, the line anywhere else (a table row, a list item, a heading).
        unit = "sentence removed" if drop else ""
        row = -1
        if not drop and any(not is_figure[i] for i in idx):
            if _NOT_PROSE.match(line):
                row = markdown.count(chr(10), 0, marks[idx[0]].start())
                gone_rows.add(row)
                unit = "table row removed" if line.lstrip().startswith("|") else "line removed"
            else:
                unit = "sentence removed"
        if drop and key_only and not flat[b:line_end if line_end != -1 else len(flat)].strip():
            # Nothing else on its line: the heading goes with it (a heading over nothing reads as a gap).
            start, end = original(a), original(b)
            if flat[line_start:a].strip() == "":
                start, end = original(line_start), min(end + 1, len(markdown))
            edits.append((start, end))
        elif unit == "sentence removed":
            start, end = original(body_at), original(b)
            if flat[line_start:body_at].strip() == "":  # it opened its line: the space it leaves goes with it
                while end < len(markdown) and markdown[end] in " \t":
                    end += 1
                if not lead and (line_end == -1 or end >= original(line_end)):  # nothing is left of the line
                    start, end = original(line_start), min(end + 1, len(markdown))
            edits.append((start, end))
        if figures_only and drop:  # a result key FI could not fill goes with the sentence: it is listed like any other
            for v in _values._MARK.finditer(markdown, original(a), original(b)):
                if v.group(2).strip() == _values.NOT_RECORDED:
                    taken.append({"key": v.group(1).strip(), "sentence": sentence, "how": "sentence removed",
                                  "at_write": True})
        for i in idx:
            if is_figure[i]:
                stem, kind, args = _figure_parts(marks[i].group(1))  # type: ignore[misc]
                if not unit:
                    edits.append((marks[i].start(), marks[i].end()))
                taken.append({"key": f"figure:{stem}:{kind}" + (f" @ {args}" if args else ""), "sentence": sentence,
                              "how": unit or "removed", "figure": _figure_number(markdown, stem)})
                continue
            taken.append({"key": marks[i].group(1).strip(), "sentence": sentence, "how": unit})
            rows_of.append((taken[-1], row))
    lines = markdown.split(chr(10))
    at = [0]
    for ln in lines:
        at.append(at[-1] + len(ln) + 1)
    tables_gone: set[int] = set()
    for row in sorted(gone_rows):
        first = last_row = row
        while first > 0 and lines[first - 1].lstrip().startswith("|"):
            first -= 1
        while last_row + 1 < len(lines) and lines[last_row + 1].lstrip().startswith("|"):
            last_row += 1
        table = range(first, last_row + 1)
        if lines[row].lstrip().startswith("|") and (row < first + 2 or all(r in gone_rows for r in table[2:])):
            # its header or rule, or every row of values: the table goes whole
            tables_gone.update(table)
    for t, row in rows_of:
        if row in tables_gone:
            t["how"] = "table removed"
    for row in gone_rows | tables_gone:
        edits.append((at[row], min(at[row + 1], len(markdown))))
    # every edit takes text out: overlapping ones are one cut
    cuts: list[list[int]] = []
    for start, end in sorted(edits):
        if cuts and start <= cuts[-1][1]:
            cuts[-1][1] = max(cuts[-1][1], end)
        else:
            cuts.append([start, end])
    out = markdown
    for start, end in reversed(cuts):
        out = out[:start] + out[end:]
    return out, taken


def run_numbers(quest_root: Path | str | None, protocol: dict[str, Any] | None) -> frozenset[Any]:
    """The numbers of FI's own records of the run that the writer may state as FI states them: the trials per setting, the
    settings and the failed trials of the trial ledger (the figures of the ``Run`` paragraph), the trials the plan asked for
    (its ``runs_per_setting``, which the plan's numbers leave out because the paper gets them from FI), the base seed, and the
    sizes of the plan's grid (levels of each axis, settings in all). Nothing here is read from prose. Empty without a quest."""

    found: set[Any] = set()
    if quest_root is None:
        return frozenset()
    root = Path(quest_root)

    def take(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        d = _typed._decimal(x)
        if d is not None and d.is_finite():
            found.add(d)

    try:
        for text in _trial_counts(root):
            for m in re.finditer(r"\d[\d,]*", text or ""):
                take(m.group(0).replace(",", ""))
        take(_cells_run(root).replace(",", "") or None)
        base, _trial = _disclosure.recorded_seeds(root)
        take(base)
    except Exception:  # noqa: BLE001 -- a record that cannot be read gives no number
        pass
    plan = protocol if isinstance(protocol, dict) else {}
    for key in ("runs_per_setting", "min_runs_per_setting"):
        take(plan.get(key))
    grid = plan.get("grid")
    if isinstance(grid, dict):
        sizes = [len(v) for v in grid.values() if isinstance(v, list) and v]
        # A count of 10 or fewer is also a small number a sentence says for other reasons (3 half-widths): only the larger
        # sizes are the plan's grid.
        total = 1
        for size in sizes:
            total *= size
            if size > 10:
                take(size)
        if sizes and total > 10:
            take(total)
    return frozenset(found)


def typed_hits(paper_md: str, protocol: dict[str, Any] | None, topic: str = "",
               quest_root: Path | str | None = None) -> list[str]:
    """Rule R1 (:mod:`core.typed_numbers`) on the writer's own text: a hit for each sentence that types a number which is not
    a number of the plan or the topic, of FI's records of the run (:func:`run_numbers`), or a label. Run only on a paper
    written under the rule."""
    return _typed.hits(to_placeholders(_disclosure.without_block(paper_md)), _typed.plan_numbers(protocol, topic), HIT,
                       run_numbers(quest_root, protocol))


def untraced_typed(paper_md: str, protocol: dict[str, Any] | None, topic: str = "",
                   quest_root: Path | str | None = None) -> list[tuple[str, str]]:
    """``(number, sentence)`` for each number the writer typed that FI cannot trace to the plan, the topic or its own records
    (the numbers :func:`typed_hits` reports), the paper left as it is."""
    return _typed.untraced(to_placeholders(_disclosure.without_block(paper_md)), _typed.plan_numbers(protocol, topic),
                           run_numbers(quest_root, protocol))


def plan_protocol(quest_root: Path | str, protocol: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The plan the typed numbers are held to: the one given, else the frozen protocol on disk."""
    return protocol if isinstance(protocol, dict) and protocol else _frozen.protocol_of(Path(quest_root))


def check(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None, keyed: bool = False,
          protocol: dict[str, Any] | None = None, typed: bool = True) -> list[str]:
    """Every must-fix hit of the records check for ``paper_md``: a required placeholder missing or misplaced and, for a
    paper written under the rule that FI fills every result number (``keyed``), a result key FI could not fill and a number
    typed that is not a number of the plan or the topic (``typed=False`` leaves those out: with no round left to write the
    paper again they are listed as not traced, :func:`keep_check`). Never raises."""
    try:
        hits = placement_hits(paper_md, quest_root, topic=topic, fallback_s=fallback_s)
        if keyed:
            hits += value_hits(paper_md, quest_root)
            if typed:
                hits += typed_hits(paper_md, plan_protocol(quest_root, protocol), topic, quest_root)
        return hits
    except Exception:  # noqa: BLE001 -- a check that cannot be made flags nothing
        return []


def keep_check(quest_root: Path | str, hits: list[str], paper_md: str, *, untraced: list[tuple[str, str]] | None = None,
               not_filled: str = "", dropped: list[dict[str, str]] | None = None) -> None:
    """Keep what the last review found (``.fi/paper_record_check.json``): the hits, the placeholders that read "not
    recorded", the typed numbers FI cannot trace that stay in the paper when the rewrite rounds are used up (``untraced``:
    ``(number, sentence)``), the result keys FI could not fill that were taken out the same way (``dropped``: ``key``,
    ``sentence``, ``how``), and why the paper is not held to the rule that FI fills every result number (``not_filled``:
    ``no_records`` or ``older_paper``). The quality report prints it, so what is still unmet when the rounds are used up
    is on record. What the study did not do (:func:`required_items`) is no fault of the paper: the paper does not carry it,
    and the quality report lists it from the records themselves."""
    path = Path(quest_root) / RECORD
    try:
        unfilled = unfilled_in(paper_md)
        if not any(_is_figure_entry(d) for d in dropped or []):
            # statements about a figure were left out when the paper was written: they stay on record until it is written again
            dropped = list(dropped or []) + [d for d in _kept_figure_entries(kept_check(quest_root))]
        if not hits and not unfilled and not untraced and not not_filled and not dropped:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"at": time.time(), "hits": list(hits), "unfilled": unfilled}
        if untraced:
            record["untraced_numbers"] = [{"number": n, "sentence": sent} for n, sent in untraced]
        if dropped:
            record["taken_out_values"] = [dict(d) for d in dropped]
        if not_filled:
            record["not_filled_by_fi"] = not_filled
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        pass


def _is_figure_entry(d: Any) -> bool:
    """A take-out made when the paper was written: a statement about a figure, or a result key FI could not fill that went
    with a sentence the figure pass removed."""
    return isinstance(d, dict) and (str(d.get("key", "")).startswith("figure:") or bool(d.get("at_write")))


def _kept_figure_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    values = data.get("taken_out_values")
    return [d for d in values if _is_figure_entry(d)] if isinstance(values, list) else []


def keep_figure_left_out(quest_root: Path | str, entries: list[dict[str, str]]) -> bool:
    """Record, for the paper just written, the statements about a figure that were left out of it (``entries`` as
    ``take_out_unfilled`` and the pointer rule list them): they replace those of the paper before, so QUALITY.md says
    what this paper left out and nothing older. Returns whether the record changed. Never raises."""
    path = Path(quest_root) / RECORD
    try:
        data = kept_check(quest_root)
        before = _kept_figure_entries(data)
        new = [dict(e) for e in entries if _is_figure_entry(e)]
        if before == new:
            return False
        others = [d for d in (data.get("taken_out_values") or []) if not _is_figure_entry(d)]
        record = dict(data) if data else {"at": time.time(), "hits": [], "unfilled": []}
        if others + new:
            record["taken_out_values"] = others + new
        else:
            record.pop("taken_out_values", None)
        if not any(record.get(k) for k in ("hits", "unfilled", "untraced_numbers", "taken_out_values",
                                           "not_filled_by_fi")):
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return True
    except (OSError, ValueError, TypeError):
        return False


def keep_figure_pass_fault(quest_root: Path | str, failed: bool) -> bool:
    """Keep (``failed``) or drop the record that the check of the sentences about the figures stopped on the paper last
    written; the Limitations block says it while it stands. Returns whether the record changed. Never raises."""
    path = Path(quest_root) / FIGURE_PASS_RECORD
    try:
        if failed == path.is_file():
            return False
        if failed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"at": time.time()}) + "\n", encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def kept_check(quest_root: Path | str) -> dict[str, Any]:
    data = _read(Path(quest_root), str(RECORD))
    return data if isinstance(data, dict) else {}
