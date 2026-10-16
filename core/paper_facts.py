"""The facts of the run in the paper come from FI's records, through placeholders; FI checks that they are there.

FI used to write three blocks into the paper itself (how often the design changed, what FI recorded about the run, the
open findings in the code). They carried false facts in graded papers (a run time of 2.3 s for a run of 223 s, "base seed 0"
for code that used a number typed into it, reviewer quotes about text no longer in the paper), so they are no longer in the
paper: they are in the quality report (``QUALITY.md``, ``core/quality_report.py``). A later attempt asked the writer to
state the facts in its own words and parsed the prose to check them; any sentence could slip past such a parser (a
"per setting" or a "failed" elsewhere in the sentence switched the check off), and real sentences about simulated time were
flagged. So the writer no longer types these facts at all:

* **Placeholders.** The writer writes the sentences and puts a placeholder where a fact goes (:data:`FACT_NAMES`):
  ``{{seed}}`` (the seed the code actually used, from :func:`core.run_facts.seed_verdict`), ``{{run_time}}`` (the measured
  wall time of the run whose trials are reported), ``{{time_limit}}`` and ``{{time_limit_met}}`` ("within" / "over"),
  ``{{trials_per_setting}}``, ``{{settings}}`` and ``{{trials_failed}}``. Each thing the study did not do is placed with
  ``{{not_done:<id>}}`` (:func:`required_items`), which FI turns into one plain sentence from its record.
* **Rendering.** :func:`render` fills each placeholder from the records when the paper is saved, between markers
  (``<!-- fi:fact-run_time -->223.2 s<!-- /fi:fact-run_time -->``), so ``paper.md`` and every check that reads it hold the
  values, never a raw placeholder. The markers let FI put the values back after any step that edits the text (the passage
  editor gets placeholders again, :func:`to_placeholders`; the page trim never offers a sentence holding one; a resumed
  quest re-renders), and they are dropped wherever text leaves FI (:func:`without_marks`). A placeholder FI cannot fill
  reads "not recorded", is said in the run log and is listed in the quality report; it is never left raw and never guessed.
* **The check** (:func:`check`, no model) is then simple: every required placeholder is in the paper, each ``not_done``
  one inside the Limitations section (:func:`placement_hits`); and a run time typed as a number outside a placeholder that
  disagrees with the measurement, or a claim that the run kept to the topic's limit when it did not, is a mismatch to
  rewrite (:func:`stated_hits`). That second part is kept narrow: simulated or physical time, a time per trial or per
  setting, the analysis's time and another run's are left alone, and anything it is unsure of is left to the claim check,
  which receives FI's record. A seed typed outside ``{{seed}}`` is held to the code's seed by ``core/stat_claims.py``.
  A failure is a must-fix hit named :data:`HIT`; it sends the paper back to be written again (the text route; never the
  design, never a stop). What is still unmet when the review rounds are used up is kept in ``.fi/paper_record_check.json``,
  which the quality report prints, and is said in the run log.
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
NOT_RECORDED = "not recorded"

#: The run's facts a placeholder can name, with what each is, for the writer.
FACT_NAMES = ("seed", "run_time", "time_limit", "time_limit_met", "trials_per_setting", "settings", "trials_failed")
_FACT_WORDS = {
    "seed": "the seed the code used to draw the trials' random numbers",
    "run_time": "the measured wall time of the experiment (the run whose trials are reported)",
    "time_limit": "the run-time limit the topic sets",
    "time_limit_met": "\"within\" or \"over\": whether the run kept to the topic's limit",
    "trials_per_setting": "how many trials each setting ran",
    "settings": "how many settings (combinations of the plan's parameters) were run",
    "trials_failed": "how many trials failed",
}
_ALIASES = {"runtime": "run_time", "wall_time": "run_time", "run_seconds": "run_time", "limit": "time_limit",
            "runtime_limit": "time_limit", "limit_met": "time_limit_met", "within_limit": "time_limit_met",
            "trials_each_setting": "trials_per_setting", "failed_trials": "trials_failed", "random_seed": "seed"}

NOT_DONE, LEFT_OUT, NOT_COMPUTED, PRECISION = "not_done", "left_out", "not_computed", "precision"
SECOND_VERSION, UNTESTED, CODE_FINDING, FAILED_TRIALS = "second_version", "untested", "code_finding", "failed_trials"
FEWER_RUNS = "fewer_runs"
#: The name of the item the paper must place when FI ran fewer runs per setting than the plan's minimum.
FEWER_RUNS_NAME = "fewer runs than the plan's minimum"
_KIND_WORDS = {
    NOT_DONE: "a part of the topic the study did not do",
    LEFT_OUT: "a quantity left out of the results because a test showed the code computes it wrongly",
    NOT_COMPUTED: "a quantity the plan asks for that the code does not compute",
    PRECISION: "a precision target that was not reached",
    SECOND_VERSION: "a key function whose second, independently written version still gives different results",
    UNTESTED: "an equation not tested against a worked example before the run",
    CODE_FINDING: "an open problem in the code that can change a reported result",
    FAILED_TRIALS: "trials that failed",
    FEWER_RUNS: "a part of the plan not fully done, to fit the topic's time limit",
}
_ID_PREFIX = {NOT_DONE: "topic", LEFT_OUT: "left-out", NOT_COMPUTED: "not-computed", PRECISION: "precision",
              SECOND_VERSION: "second-version", UNTESTED: "equation", CODE_FINDING: "finding", FAILED_TRIALS: "failed-trials",
              FEWER_RUNS: "fewer"}


@dataclass(frozen=True)
class Item:
    """One thing the paper must say the study did not do: ``id`` is what the writer places as ``{{not_done:<id>}}``,
    ``sentence`` what FI puts there."""

    kind: str
    id: str
    name: str
    sentence: str

    @property
    def text(self) -> str:  # the plain sentence, for the quality report
        return self.sentence


@dataclass(frozen=True)
class Fact:
    """One fact of the run: ``text`` is what FI puts where ``{{name}}`` is; ``required`` when the paper must state it."""

    name: str
    text: str
    required: bool = False


@dataclass(frozen=True)
class Rendered:
    text: str
    unfilled: tuple[str, ...]  # the placeholders that read "not recorded"


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


def _sentence(text: str) -> str:
    text = _one(text).replace("<!--", "").replace("-->", "")
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


# ---- what the study did not do ---------------------------------------------------------------------------------------


def _topic_sentence(gap: Any) -> tuple[str, str]:
    """``(name, sentence)`` for one part of the topic the study did not do (an object of ``topic_covered.left``, or a plain
    sentence of an older record)."""
    if isinstance(gap, str):
        said = _one(gap).rstrip(".")
        if re.match(r"(?i)the topic (asks|says|states|wants|requires)\b", said):
            return gap, _sentence(f"{said}; this study did not do it")
        return gap, _sentence(f"The topic asks for this, and this study did not do it: {said}")
    if not isinstance(gap, dict):
        return "", ""
    name = _one(gap.get("name"))
    kind = gap.get("kind")
    said = _one(gap.get("sentence"))
    if kind == "factor":
        return name, _sentence(f"The topic asks to vary {name}; this study did not vary it")
    if kind == "quantity":
        return name, _sentence(f"The topic asks to report {name}; this study did not measure it")
    if kind == "level":
        return name, _sentence(f"The topic states which values of {name} to use; this study did not use them")
    formula = re.search(r"`([^`]+)`", said)
    if kind == "formula" or formula:
        return name, _sentence(f"The topic states the formula `{formula.group(1) if formula else name}`; this study's model "
                               "does not use it")
    return name or said, _sentence(f"{said.rstrip('.')}; this study did not do it" if said else
                                   f"The topic asks for {name}; this study did not do it")


def required_items(quest_root: Path | str) -> list[Item]:
    """Every item the Limitations must place, from FI's records. A missing or unreadable record gives none."""
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
        items.append(Item(kind, ident, name, sentence))

    covered = _read(root, ".fi/topic_covered.json")
    for gap in (covered.get("left") if isinstance(covered, dict) else None) or []:
        name, said = _topic_sentence(gap)
        add(NOT_DONE, name, said)
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
            add(LEFT_OUT, q, _sentence(f"`{q}` is not reported: a test FI ran on the code showed that the code computes it "
                                       "wrongly"))
        elif gate.get("outcome") == "not_computed":
            gated.add(str(fid))
            for n in gate.get("names") if isinstance(gate.get("names"), list) else []:
                add(NOT_COMPUTED, n, _sentence(f"The plan asks for `{_one(n)}`; the code does not compute it, so this study "
                                               "does not report it"))
    checked = _read(root, "needs/RESULT_CHECK.json")
    for c in (checked.get("checks") if isinstance(checked, dict) else None) or []:
        if isinstance(c, dict) and c.get("quantity"):
            q = _one(c["quantity"])
            note = _one(c.get("note"))[:200].rstrip(".")
            add(LEFT_OUT, q, _sentence(f"`{q}` is not reported: a check FI ran showed the value is wrong"
                                       + (f" ({note})" if note else "")))
    missed = _read(root, str(_disclosure.PRECISION_RECORD))
    for m in missed if isinstance(missed, list) else []:
        if isinstance(m, dict) and m.get("name"):
            try:
                said = (f"The 95% interval for `{_one(m['name'])}` is ±{float(m['half_width']):.3g}, wider than the plan's "
                        f"target of ±{float(m['target']):.3g}"
                        + (f" (from {int(m['trials']):,} trials)" if isinstance(m.get("trials"), (int, float)) else ""))
            except (KeyError, TypeError, ValueError):
                said = f"The plan's precision target for `{_one(m['name'])}` was not reached"
            add(PRECISION, m["name"], _sentence(said))
    # A key function whose second, independently written version still gives different results (not one that could not
    # be compared: that is in the quality report only).
    try:
        from . import second_impl as _second

        for fname, entry in ((_second.load(root).get("functions") or {}).items()):
            if isinstance(entry, dict) and entry.get("status") == _second.UNRESOLVED:
                said = _one(entry.get("limitation")) or (f"Two independently written versions of `{fname}` gave different "
                                                         "results; the number depends on which is right")
                add(SECOND_VERSION, fname, _sentence(said))
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
    # Open problems in the code, flagged by a second reading, that can change a reported result (one about a function the
    # run never calls cannot: core/code_review.py:open_findings leaves those out).
    try:
        from . import code_review as _code_review

        found = _code_review.open_findings(root, root / "code")
        for v in [*found["missing"], *found["different"], *found["flagged"], *found["unshown"]]:
            fid = str(v.get("fid") or "")
            if fid in gated:
                continue
            issue = issues.get(fid) if fid else None
            changes = (issue.get("changes_result") if isinstance(issue, dict) and issue.get("changes_result") is not None
                       else _code_findings.changes_result(v))
            if not changes:
                continue
            text = _code_review._plain_text(v.get("text"))
            if text:
                add(CODE_FINDING, text[:80], _sentence("A second reading of the code flagged a problem that can change a "
                                                       f"reported result and was not resolved: {text}"), key=fid or text)
    except Exception:  # noqa: BLE001
        pass
    rows = _disclosure._ledger_rows(root)
    if rows:
        statuses = [status for status, _seed in rows[1].values()]
        failed = sum(1 for s in statuses if s == "failed")
        if failed:
            add(FAILED_TRIALS, "failed trials", _sentence(f"{failed:,} of {len(statuses):,} trials failed; their results are "
                                                          "not in the reported numbers"), key="all")
    estimate = _read(root, ".fi/run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    if isinstance(reduced, dict) and reduced.get("below_plan_minimum"):
        to = reduced.get("to") if isinstance(reduced.get("to"), dict) else {}
        add(FEWER_RUNS, FEWER_RUNS_NAME,
            _sentence(f"The experiment ran {to.get('runs_per_setting', '?')} runs per setting against the plan's minimum of "
                      f"{reduced.get('plan_minimum', '?')}, so that the run fits the topic's time limit"), key="runs")
    return items


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
    """A duration in plain units: ``121.3 s``, ``223.2 s (about 3.7 minutes)``, ``9,000 s (about 2.5 hours)``."""
    said = _disclosure._plain_seconds(seconds)
    if seconds >= 2 * 3600:
        return f"{said} (about {seconds / 3600:.1f} hours)"
    if seconds >= 180:
        return f"{said} (about {seconds / 60:.1f} minutes)"
    return said


# ---- the facts ------------------------------------------------------------------------------------------------------


def _seed_text(root: Path) -> str:
    """What ``{{seed}}`` says: the seed the code used, from the code's own reading (:func:`core.run_facts.seed_verdict`)."""
    verdict = _run_facts.seed_verdict(root)
    state = verdict.get("verdict")
    if state == _run_facts.NO_ENTRY:
        return ""
    if state == _run_facts.REACHES:
        base, _trial = _disclosure.recorded_seeds(root)
        return (f"a separate seed for each trial, derived from base seed {base}" if base is not None
                else "a separate seed for each trial, passed to the simulation by FI")
    if state == _run_facts.UNKNOWN:
        return "a seed passed to each trial (FI could not tell from the code whether the simulation uses it)"
    driving = [d for d in verdict.get("driving_seeds") or [] if isinstance(d, dict)]
    if state == _run_facts.PARTLY:
        # Both are true and both drive the trials: FI's per-trial seed and a number typed into the code.
        base, _trial = _disclosure.recorded_seeds(root)
        listed = "; ".join(f"{d['value']}, at {d['where']}" for d in driving[:3])
        return ("FI's own seed for each trial" + (f" (derived from base seed {base})" if base is not None else "")
                + (", and a seed fixed in the code" if len(driving) == 1 else ", and seeds fixed in the code")
                + f" ({listed})")
    unseeded = [str(u).split(" (", 1)[0] for u in verdict.get("unseeded") or []]
    if driving:
        listed = "; ".join(f"{d['value']}, at {d['where']}" for d in driving[:3])
        text = f"a seed fixed in the code ({listed})" if len(driving) == 1 else f"seeds fixed in the code ({listed})"
        if unseeded:
            text += f", and no seed at {', '.join(unseeded[:2])}"
        return text
    if unseeded:
        return f"no fixed seed for part of its random numbers (the code draws them without one at {', '.join(unseeded[:2])})"
    return ("no seed FI can name (the simulation does not use the seed FI passes each trial, and FI found none written "
            f"into the code that runs a trial, {verdict.get('entry') or 'code/simulate.py'})")


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
    failed = sum(1 for status, _s in trials.values() if status == "failed")
    return each, f"{len(cells):,}", f"{failed:,}"


def facts(quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> dict[str, Fact]:
    """The run's facts FI holds, by placeholder name (a fact FI does not hold is not in it). ``required`` marks the ones
    the paper must state: the seed, the run time, and whether the run kept to the topic's limit when it sets one."""
    root = Path(quest_root)
    out: dict[str, Fact] = {}
    seed = _seed_text(root)
    if seed:
        out["seed"] = Fact("seed", seed, required=True)
    wall, least = measured_run_time(root, fallback_s)
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
        out["run_time"] = Fact("run_time", said, required=True)
    elif least is not None:
        out["run_time"] = Fact("run_time", f"at least {_plain(least)} (the trials alone; FI did not time the whole run)",
                               required=True)
    limit = _disclosure.topic_limit(root, topic)
    if limit is not None:
        out["time_limit"] = Fact("time_limit", _plain(limit))
        met = ("within" if wall <= limit else "over") if wall is not None else ("over" if least is not None
                                                                               and least > limit else "")
        if met:
            out["time_limit_met"] = Fact("time_limit_met", met, required=True)
    each, settings, failed = _trial_counts(root)
    for name, value in (("trials_per_setting", each), ("settings", settings), ("trials_failed", failed)):
        if value:
            out[name] = Fact(name, value)
    return out


# ---- the placeholders and their markers ------------------------------------------------------------------------------

#: Any name between double braces, with extra braces around it taken in (``{{{{seed}}}}``, ``{{ {seed} }}``).
_PLACEHOLDER = re.compile(r"\{\{[\s{]*([A-Za-z][^{}\n:]{0,59}?)\s*(?::\s*([^{}\n]{1,80}?)\s*)?[\s}]*\}\}")
_MARK = re.compile(r"<!-- fi:fact-([\w-]+) -->(.*?)<!-- /fi:fact-\1 -->", re.S)
_TAG = re.compile(r"<!-- /?fi:fact-[\w-]+ -->")
_ITEM_KEY = "not_done-"
#: Where no reader sees a placeholder as text: a code block or inline code, math, an HTML comment (not FI's own markers).
_HIDDEN = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)[^\n]*\n.*?^[ \t]*(?P=fence)[^\n]*$"
    r"|^[ \t]*(?:```|~~~).*\Z"
    r"|<!--(?!\s*/?fi:fact-).*?-->"
    r"|(?P<ticks>`+)(?:(?!\n[ \t]*\n)[^`])+?(?P=ticks)"
    r"|\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)"
    r"|(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)",
    re.M | re.S)
#: A double brace right after a LaTeX command or a closing brace is LaTeX (``\textbf{{x}}``), not a placeholder.
_LATEX_BEFORE = re.compile(r"(?:\\[A-Za-z]+\*?\s*|[}^_])$")
#: The sections that list sources: a placeholder there is not where a reader looks for the run's facts.
_SOURCES = re.compile(r"^#{1,6}[ \t]+" + _disclosure._SECTION_NUMBER
                      + r"(?:references|bibliography|further[ \t]+reading|works[ \t]+cited|sources)\b", re.I)


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


def to_placeholders(markdown: str) -> str:
    """``markdown`` with every value FI filled in (between its markers) turned back into its placeholder: what the passage
    editor is given, so its edits are made on the placeholders and FI fills them again."""
    def back(m: re.Match[str]) -> str:
        key = m.group(1)
        if key.startswith(_ITEM_KEY):
            return "{{not_done:" + key[len(_ITEM_KEY):] + "}}"
        return "{{" + key + "}}"

    return _MARK.sub(back, markdown or "")


def without_marks(markdown: str) -> str:
    """``markdown`` with FI's fact markers taken out and the values kept: what a reader, the slides and the poster get."""
    return _TAG.sub("", markdown or "")


def mask(markdown: str, token: str = " FI-FACT ") -> str:
    """``markdown`` with each value FI filled in replaced by ``token``: what the writer typed, and nothing of FI's."""
    return _MARK.sub(token, _PLACEHOLDER.sub(token, markdown or ""))


def render(markdown: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> Rendered:
    """``markdown`` with every placeholder (and every value FI filled in before, so a re-render takes the records as they
    are now) filled from FI's records, between markers. A placeholder FI cannot fill reads "not recorded" and is in
    ``unfilled``; an unknown name that looks like a placeholder (``{{run_seconds}}``, ``{{not_done:x}}``) is treated the
    same. Any other name between double braces is no placeholder of FI's: it reads "not recorded" between markers, so it
    is listed in QUALITY.md and the check asks for one of FI's names. Extra braces around a name are taken in.

    Inside math, code or an HTML comment, a placeholder of FI's gets its plain value with no markers (a marker there would
    print as text), and anything else between double braces is left as written (LaTeX's ``{{x}}``); so is a double brace
    right after a LaTeX command."""
    root = Path(quest_root)
    known = facts(root, topic=topic, fallback_s=fallback_s)
    items = {i.id.lower(): i for i in required_items(root)}
    unfilled: list[str] = []
    text = to_placeholders(markdown)
    hidden = _hidden_spans(text)

    def fill(m: re.Match[str]) -> str:
        raw, arg = m.group(1), m.group(2)
        name = _norm_name(raw)
        plain = _inside(m.start(), hidden)
        if _LATEX_BEFORE.search(text[max(0, m.start() - 40):m.start()]):
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
    # A placeholder for a whole sentence followed by the writer's own full stop: one full stop.
    out = re.sub(r"([.!?])(<!-- /fi:fact-[\w-]+ -->)\.", r"\1\2", out)
    return Rendered(out, tuple(dict.fromkeys(unfilled)))


def unfilled_in(markdown: str) -> list[str]:
    """The placeholders of a rendered paper that read "not recorded"."""
    out = []
    for m in _MARK.finditer(markdown or ""):
        if m.group(2).strip() == NOT_RECORDED:
            key = m.group(1)
            out.append("not_done:" + key[len(_ITEM_KEY):] if key.startswith(_ITEM_KEY) else key)
    return list(dict.fromkeys(out))


def fact_numbers(quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> dict[str, float]:
    """Every number FI puts into the paper through a placeholder, by where it comes from: FI's own record, which the
    number checks must not count as a result nothing accounts for."""
    out: dict[str, float] = {}
    texts = [(f"fact.{f.name}", f.text) for f in facts(quest_root, topic=topic, fallback_s=fallback_s).values()]
    texts += [(f"item.{i.id}", i.sentence) for i in required_items(quest_root)]
    for where, text in texts:
        for k, m in enumerate(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text)):
            try:
                out[f"{where}.{k}"] = float(m.group(0).replace(",", ""))
            except ValueError:
                continue
    return out


# ---- what the writer is given ---------------------------------------------------------------------------------------


def writer_section(quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> str:
    """The record section of the write prompt: the placeholders the paper's facts go through, what each will say now, one
    example sentence, and the ``not_done`` placeholders the Limitations must hold. ``""`` when FI holds nothing."""
    root = Path(quest_root)
    known = facts(root, topic=topic, fallback_s=fallback_s)
    items = required_items(root)
    lines: list[str] = []
    estimate = _read(root, ".fi/run_estimate.json")
    reduced = estimate.get("reduced") if isinstance(estimate, dict) else None
    reduced_note = (_one(reduced["note"]) if isinstance(reduced, dict) and isinstance(reduced.get("note"), str)
                    and reduced["note"].strip() else "")
    over = estimate.get("over_topic_limit") if isinstance(estimate, dict) else None
    over_note = _one(over["note"]) if isinstance(over, dict) and isinstance(over.get("note"), str) and over["note"].strip() else ""
    if not known and not items and not reduced_note and not over_note:
        return ""
    lines.append("FI'S RECORD OF THIS RUN. The facts below come from FI's own records. Do not type them: write the "
                 "placeholder (double braces, exactly as shown) where the fact goes, and FI puts the value in from its "
                 "records when the paper is saved. A seed or a run time typed as a number, or a claim about the topic's "
                 "time limit that the record contradicts, is found by a check and the paper is sent back to be written "
                 "again.")
    if known:
        lines.append("Placeholders (with what FI will put in for each, as the records stand now):")
        lines += [f"- {{{{{f.name}}}}}: {_FACT_WORDS[f.name]}. Now: \"{f.text}\"" for f in known.values()]
        needed = [f"{{{{{f.name}}}}}" for f in known.values() if f.required]
        example = [p for p in ("Random numbers were drawn from {{seed}}" if "seed" in known else "",
                               "the experiment ran {{trials_per_setting}} trials in each of {{settings}} settings"
                               if "settings" in known and "trials_per_setting" in known else "",
                               "and took {{run_time}}" if "run_time" in known else "") if p]
        if example:
            sentence = "; ".join(example[:2]) + (" " + example[2] if len(example) > 2 else "")
            if "time_limit_met" in known and "time_limit" in known:
                sentence += ", {{time_limit_met}} the topic's limit of {{time_limit}}"
            lines.append(f"Example (in the Methods): \"{sentence}.\"")
        if needed:
            lines.append(f"Required: the paper must hold {', '.join(needed)} (in the Methods, or where the run is described).")
    if reduced_note:
        lines.append(f"Reduced size: {reduced_note} Say this in the paper, with the numbers.")
    if over_note:
        lines.append(f"Time limit: {over_note} Say this in the paper.")
    if items:
        lines.append("In the Limitations section (a section headed \"Limitations\"), put each of these placeholders on its "
                     "own, as a sentence; FI replaces it with the plain sentence shown, from its record. You may add your "
                     "own words before or after it, but do not move it out of the Limitations, change it or leave it out. "
                     "A paper that leaves out a part of the topic without saying so is not trusted:")
        lines += [f"- {{{{not_done:{i.id}}}}} ({_KIND_WORDS[i.kind]}): \"{i.sentence}\"" for i in items]
    lines.append("A placeholder FI cannot fill is printed as \"not recorded\", so use only the ones listed here.")
    return "\n".join(lines)


# ---- the check ------------------------------------------------------------------------------------------------------


def _title_span(markdown: str) -> tuple[int, int]:
    m = re.search(r"^#[ \t]+[^\n]*$", markdown, re.M)
    return (m.start(), m.end()) if m else (0, 0)


def _limitations_spans(markdown: str) -> list[tuple[int, int]]:
    """The character spans of the paper's Limitations sections (a heading that names limitations or threats to validity,
    to the next heading of its level or higher)."""
    lines = markdown.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    heads = _disclosure._headings(lines, 0)
    title = next((i for i, lv, _t in heads if lv == 1), None)
    spans = []
    for i, lv, text in heads:
        if i != title and _disclosure._LIMITS.match(text):
            end = _disclosure._section_end(heads, i, lv, len(lines))
            spans.append((offsets[i], offsets[end] if end < len(lines) else len(markdown)))
    return spans


def _sources_spans(markdown: str) -> list[tuple[int, int]]:
    """The character spans of the sections that list sources (References, Bibliography, Further reading)."""
    lines = markdown.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    heads = _disclosure._headings(lines, 0)
    spans = []
    for i, lv, text in heads:
        if _SOURCES.match(text):
            end = _disclosure._section_end(heads, i, lv, len(lines))
            spans.append((offsets[i], offsets[end] if end < len(lines) else len(markdown)))
    return spans


def _placed(markdown: str, key: str) -> list[int]:
    """Where FI's value for ``key`` is in text a reader sees: not inside an HTML comment, code or math, and not in the
    list of sources."""
    away = _hidden_spans(markdown) + _sources_spans(markdown)
    return [m.start() for m in _MARK.finditer(markdown) if m.group(1) == key and not _inside(m.start(), away)]


def _unknown_hits(markdown: str, quest_root: Path | str, known: dict[str, Fact]) -> list[str]:
    """A hit for each name between double braces that is no placeholder of FI's (it reads "not recorded")."""
    ids = {i.id for i in required_items(quest_root)}
    names = ", ".join("{{" + n + "}}" for n in (known or dict.fromkeys(FACT_NAMES)))
    hits: list[str] = []
    for key in dict.fromkeys(m.group(1) for m in _MARK.finditer(markdown)):
        if key.startswith(_ITEM_KEY):
            if key[len(_ITEM_KEY):] not in ids:
                listed = ", ".join("{{not_done:" + i + "}}" for i in sorted(ids))
                hits.append(f"{HIT}: {{{{not_done:{key[len(_ITEM_KEY):]}}}}} is not one of the parts FI recorded as not "
                            "done, so the paper reads \"not recorded\" there. "
                            + (f"Use one of {listed}, or take it out." if listed else "Take it out."))
        elif key not in FACT_NAMES:
            hits.append(f"{HIT}: {{{{{key}}}}} is not one of FI's placeholders, so the paper reads \"not recorded\" there. "
                        f"Use one of {names}, or write it in your own words without double braces.")
    return hits


def placement_hits(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Must-fix hits for a required placeholder the paper does not hold: a required fact of the run anywhere but the
    title, each ``not_done`` item inside the Limitations section."""
    text = _disclosure.without_block(paper_md)
    t0, t1 = _title_span(text)
    known = facts(quest_root, topic=topic, fallback_s=fallback_s)
    hits: list[str] = _unknown_hits(text, quest_root, known)
    for fact in known.values():
        if not fact.required:
            continue
        where = [p for p in _placed(text, fact.name) if not t0 <= p < t1]
        if not where:
            # The hit names the placeholder, not its value: the value (a measured time) differs from run to run, and the
            # writer is given it in FI's record section anyway.
            hits.append(f"{HIT}: the paper does not state {{{{{fact.name}}}}} ({_FACT_WORDS[fact.name]}). Put the "
                        f"placeholder {{{{{fact.name}}}}} in the Methods; do not type the value.")
    spans = _limitations_spans(text)
    for item in required_items(quest_root):
        where = _placed(text, _ITEM_KEY + item.id)
        inside = [p for p in where if any(a <= p < b for a, b in spans)]
        if inside:
            continue
        if where:
            what = ("in the title" if any(t0 <= p < t1 for p in where) else "outside the Limitations section")
            hits.append(f"{HIT}: {{{{not_done:{item.id}}}}} ({_KIND_WORDS[item.kind]}) is {what}. Move it into the "
                        "Limitations section.")
        else:
            hits.append(f"{HIT}: the paper does not say {_KIND_WORDS[item.kind]}: \"{item.sentence}\" Put "
                        f"{{{{not_done:{item.id}}}}} in the Limitations section"
                        + ("" if spans else " (the paper has no Limitations section: add one)") + ".")
    return hits


# What (b) reads: a run time typed as a number outside a placeholder.
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30,
                 "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90}
_NUM = r"(?:\d[\d,]*(?:\.\d+)?|" + "|".join(sorted(_WORD_NUMBERS, key=len, reverse=True)) + r")"
_UNIT = r"(?:milliseconds?|ms|seconds?|secs?|minutes?|mins?|hours?|hrs?|hr|s|h)"
_DURATION = re.compile(
    rf"(?<![\w.])(?:(?P<lo>{_NUM})\s*(?:-|–|to|and)\s*)?(?P<n>{_NUM})(?P<sep>\s*-\s*|\s*)(?P<u>{_UNIT})(?![\w])", re.I)
#: Where a clause ends: what is on the other side is about something else.
_BREAK = re.compile(r";|\bof which\b|\bwhereas\b|\bwhile\b|,\s*but\b|\bbut\b|—|\s–\s")
#: A word that says the clause is about how long the experiment ran.
_ANCHOR = re.compile(
    r"\b(?:ran|run|runs|running|runtime|run[- ]time|took|takes|taking|completed|completes|finished|finishes|lasted|"
    r"lasts|needed|needs|required|requires|elapsed|wall[- ]?clock|wall[- ]time|execution|executed|computation|"
    r"compute[- ]time|computing[- ]time|cpu[- ]time|in total)\b", re.I)
#: The time is simulated or physical, not the computer's (``600 s of simulated time``, ``t = 0 to 100 s``, a half-life).
_PHYSICAL_BEFORE = re.compile(
    r"(?:\bt\s*(?:=|∈|from|in)|\btimes?[- ]steps?\b|\bstep(?:\s+size)?\s+(?:of|=)|\bdt\s*=|Δt|\bhalf[- ]li(?:fe|ves)\b|"
    r"\blifetimes?\b|\btime[- ]constants?\b|\bhorizons?\b|\bevery\b|\bintervals?\b|\bperiods?\b|\bdelays?\b|"
    r"\bresidence\b|\buntil\b|\bsimulated\b|\breaction\b|\bphysical\b|\bat\s+t\b|\bsampl\w+)", re.I)
_PHYSICAL_AFTER = re.compile(r"^\s*(?:of\s+)?(?:\w+\s+)?(?:simulated|simulation|reaction|physical|model|process|virtual)\b"
                             r"|^\s*of\s+(?:\w+\s+)?time\b|^\s*(?:after|into)\b", re.I)
# "The simulation" or "the model" as the subject is the computer's run as often as not ("The simulation took 2.3 s"): a
# clause about simulated time says so itself (``simulated``, ``t =``, checked above). Only a physical subject is physical.
_PHYSICAL_SUBJECT = re.compile(r"\b(?:reactions?|integration|ode|trajector(?:y|ies)|particles?|decay)\b\W*(?:\w+\W+){0,2}$",
                               re.I)
#: Another study's or another implementation's time, not this run's: a citation, a possessive name, "et al.", or a word
#: that names another work ("the reference implementation", "prior work", "the original 2019 study", "reported by").
_OTHER_STUDY = re.compile(
    r"\[\s*@?\w[^\]]*\]|\bet\s+al\b|\b[A-Z][a-z]+['’]s\b|\b(?:reference|baseline|original|prior|previous|earlier|"
    r"published|their|others?|existing|literature|competing|reported|benchmarked)\b|\b(?:19|20)\d\d\b(?!s)")
#: A part of the run, or another run: a time per trial or setting, an average, the analysis, a discarded or earlier run.
_PART_BEFORE = re.compile(
    r"\b(?:(?:each|every|per|one|a\s+single|single|individual|average|mean|median|typical)\s+(?:\w+\s+)?"
    r"(?:trials?|settings?|cells?|replicates?|famil(?:y|ies)|seeds?|simulations?|iterations?|runs?|steps?|fits?|tasks?)|"
    r"analys[ie]s|analysing|analyzing|post-?processing|plotting|plots?|figures?|start-?up|imports?|loading|compil\w+|"
    r"(?:unit\s+)?tests?|bootstrap\w*|trials?\s+alone|"
    r"(?:first|earlier|previous|prior|discarded|pilot|exploratory|exploring|initial|warm-?up|probe|calibration|"
    r"smaller|reduced|short|dry)[\s,]+(?:\w+\s+)?(?:runs?|attempts?|versions?|passes|pass))\b", re.I)
_PART_AFTER = re.compile(r"^\s*(?:\w+\s+){0,2}?(?:per|each|apiece|on\s+average|a\s+piece)\b", re.I)
_LINK = re.compile(r",|;|\band\b|\bwith\b|\bbut\b|\bthen\b|\bwhich\b|\bthat\b|\bwhere\b|\bso\b", re.I)
_UPPER = re.compile(r"(?:\bunder|\bwithin|\bbelow|\bless\s+than|\bfewer\s+than|\bat\s+most|\bno\s+more\s+than|"
                    r"\bnot\s+more\s+than|\bup\s+to|\binside|\bshorter\s+than|\bfaster\s+than|<=?|≤)\s*"
                    r"(?:about\s+|roughly\s+|approximately\s+|~\s*|the\s+)?$", re.I)
_LOWER = re.compile(r"(?:\bover|\bmore\s+than|\babove|\bat\s+least|\blonger\s+than|\bexceed\w*|\bbeyond|>=?|≥)\s*"
                    r"(?:about\s+|roughly\s+|approximately\s+|~\s*|the\s+)?$", re.I)
_LIMIT_AFTER = re.compile(r"^\s*(?:\w+\s+){0,1}(?:limit|budget|cap|ceiling|constraint|allowance|maximum|deadline)\b", re.I)
_LIMIT_BEFORE = re.compile(r"\b(?:limit|budget|cap|ceiling|maximum|allowance|deadline)(?:\s+(?:of|is|was|at|set\s+at))?"
                           r"\s*[:=]?\s*(?:about\s+|roughly\s+|~\s*)?$|\b(?:allowed|allotted|permitted)\s*$", re.I)
_APPROX = re.compile(r"(?:about|around|roughly|approximately|nearly|almost|some|~)\s*$", re.I)
#: How far a typed run time may be from the measured one (a rounding such as "about 4 minutes" for 223 s is within it).
_TOLERANCE = 0.3
#: A claim that the run kept to a limit (the met-word before the limit, or the limit then "was met"); a negation right
#: before the met-word, or between the limit and "met", undoes it.
_MET = re.compile(
    r"(?P<neg>\b(?:not|n't|never|failed\s+to|fail\s+to|without)\s+(?:\w+\s+){0,1})?"
    r"\b(?:within|under|inside|below|met|meet|meets|meeting|satisf\w+|respect\w*|compl(?:y|ies|ied)\s+with|"
    r"kept\s+(?:to|within)|stay(?:ed|s)?\s+(?:within|under)|fit(?:s|ted)?\s+(?:in|within)|honou?r\w*)\s+"
    r"(?P<what>(?:[\w'’-]+\s+){0,5}?)(?P<word>limit|budget|constraint|cap|ceiling|requirement|allowance|deadline)s?\b"
    r"|(?P<what2>(?:[\w'’-]+\s+){0,2})(?P<word2>limit|budget|constraint|cap|ceiling|requirement|allowance|deadline)s?\s+"
    r"(?:\w+\s+){0,2}?(?:was|were|is|are|has\s+been|had\s+been)\s+(?P<neg2>not\s+)?"
    r"(?:met|satisfied|respected|honou?red|kept|observed)\b",
    re.I)
#: Words that make a limit a limit on time.
_TIME_WORD = re.compile(r"\b(?:time|runtime|run[- ]time|wall|clock|cpu|compute|computing|duration|seconds?|minutes?|"
                        r"hours?|FI-FACT)\b|\d\s*(?:s|sec|min|h)\b", re.I)
#: A verb that says the run finished: with it, "inside the limit" is about time even when the limit's words do not say so.
_DONE_VERB = re.compile(r"\b(?:ran|run|runs|completed|completes|finished|finishes|took|executed|stayed)\b", re.I)
_STRONG_LIMIT = ("limit", "budget", "cap", "ceiling", "allowance", "deadline")
_OTHER_LIMIT = re.compile(r"\b(?:accuracy|error|tolerance|memory|precision|page|word|length|size|sample|storage|"
                          r"disk|budget\s+of\s+\d+\s+trials|iteration|step)\b", re.I)


def _seconds(num: str, unit: str) -> float | None:
    num = num.strip().lower()
    value = float(_WORD_NUMBERS[num]) if num in _WORD_NUMBERS else None
    if value is None:
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            return None
    u = unit.lower()
    if u.startswith("ms") or u.startswith("milli"):
        return value / 1000.0
    if u.startswith("h"):
        return value * 3600.0
    if u.startswith("m"):
        return value * 60.0
    return value


_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _prose(markdown: str) -> list[str]:
    """The sentences of the writer's own text, with FI's filled values masked: prose, a figure's caption, and each cell of
    a table read with its column's header ("Run time: 2.3 s"), or a two-column row as "name: value". No code or headings."""
    out: list[str] = []
    fenced = False
    header: list[str] = []
    for line in mask(_disclosure.without_block(markdown)).splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if stripped.startswith("|") and not fenced:
            cells = _cells(stripped)
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # the line under a table's header
            if not header:
                header = cells
                continue
            out.extend(f"{h}: {c}." for h, c in zip(header, cells) if h and c)
            if len(cells) == 2 and cells[0] and cells[1]:
                out.append(f"{cells[0]}: {cells[1]}.")
            continue
        header = []
        if fenced or not stripped or stripped.startswith(("#", "<!--", "---")):
            continue
        line = _IMAGE.sub(lambda m: m.group(1) + " ", line)
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[*`\"])", line) if s.strip())
    return out


def _clause(sentence: str, start: int, end: int) -> tuple[str, int, str]:
    """The clause around ``sentence[start:end]`` (to the nearest break on each side), with every bracketed aside that
    does not hold it blanked out, where the span starts in it, and the break that opens it (``""`` at the start)."""
    opened = None
    for m in _BREAK.finditer(sentence, 0, start):
        opened = m
    left = opened.end() if opened else 0
    right = next((m.start() for m in _BREAK.finditer(sentence, end)), len(sentence))
    text = sentence[left:right]
    rel = start - left

    def blank(m: re.Match[str]) -> str:
        return m.group(0) if m.start() <= rel < m.end() else " " * len(m.group(0))

    return re.sub(r"\([^()]*\)", blank, text), rel, opened.group(0).strip().lower() if opened else ""


def _attached(before: str, pattern: re.Pattern[str]) -> bool:
    """Whether ``pattern`` is found in ``before`` with no linking word (a comma, "and", "with") between it and the end."""
    for m in pattern.finditer(before):
        if not _LINK.search(before[m.end():]):
            return True
    return False


def _shown(sentence: str) -> str:
    one = _one(sentence).replace(" FI-FACT ", " [FI's value] ")
    return one if len(one) <= 160 else one[:157] + "..."


def stated_hits(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Must-fix hits for a run time typed outside a placeholder that disagrees with the measured one, and for a claim that
    the run kept to the topic's limit when the measurement says it did not. Silent when FI measured nothing; anything it is
    unsure of is left alone (the claim check reads the paper against FI's record)."""
    root = Path(quest_root)
    wall, least = measured_run_time(root, fallback_s)
    if wall is None and least is None:
        return []
    limit = _disclosure.topic_limit(root, topic)
    over = limit is not None and ((wall is not None and wall > limit) or (wall is None and least is not None and least > limit))
    hits: list[str] = []
    measured = (f"FI measured {_plain(wall)} for the run whose trials are reported" if wall is not None
                else f"the trials alone took at least {_plain(least)} in FI's ledger")  # type: ignore[arg-type]
    for s in _prose(paper_md):
        flagged = False
        for m in _DURATION.finditer(s):
            num, unit = m.group("n"), m.group("u")
            if m.group("sep").strip() == "" and m.group("sep") == "" and unit.lower() in ("s", "h") and "." not in num:
                if num.replace(",", "").isdigit() and int(num.replace(",", "")) >= 1000:
                    continue  # "1990s"
            seconds = _seconds(num, unit)
            if seconds is None:
                continue
            lo = _seconds(m.group("lo"), unit) if m.group("lo") else None
            clause, rel, opened = _clause(s, m.start(), m.end())
            before, after = clause[:rel], clause[rel + (m.end() - m.start()):]
            if not _ANCHOR.search(clause) or opened == "of which":
                continue  # not about how long something ran, or a part of the time just given ("..., of which the trials took 33 s")
            if _PHYSICAL_BEFORE.search(" ".join(before.split()[-6:])) or _PHYSICAL_AFTER.search(after) or \
                    re.search(r"\b(?:simulated|simulation[- ]time|reaction[- ]time|physical[- ]time|model[- ]time)\b",
                              clause, re.I):
                continue
            anchor = next(iter(_ANCHOR.finditer(clause)), None)
            if anchor is not None and _PHYSICAL_SUBJECT.search(clause[:anchor.start()]):
                continue
            # Another study's time: named in the subject ("Smith's solver took", "Prior work needed"), or right after the
            # number up to the next comma ("the 22 s reported by Smith [3]", "in an earlier study [3]").
            subject = clause[:anchor.start()] if anchor is not None and anchor.start() < rel else before
            near_after = re.split(r",|\.\s", after, maxsplit=1)[0]
            if _OTHER_STUDY.search(subject) or _OTHER_STUDY.search(near_after):
                continue
            if _attached(before, _PART_BEFORE) or _PART_AFTER.search(after):
                continue
            if _LIMIT_AFTER.search(after) or _LIMIT_BEFORE.search(before):
                continue  # the limit itself; whether the run kept to it is the claim below
            if _UPPER.search(before):
                if wall is not None and wall > seconds * 1.0001 or wall is None and least is not None and least > seconds:
                    hits.append(f"{HIT}: the paper says the run took less than {_plain(seconds)} (\"{_shown(s)}\"), but "
                                f"{measured}. Write {{{{run_time}}}} instead of a typed time.")
                    flagged = True
                continue
            if _LOWER.search(before):
                if wall is not None and wall < seconds * 0.9999:
                    hits.append(f"{HIT}: the paper says the run took more than {_plain(seconds)} (\"{_shown(s)}\"), but "
                                f"{measured}. Write {{{{run_time}}}} instead of a typed time.")
                    flagged = True
                continue
            if wall is not None:
                low, high = (min(lo, seconds), max(lo, seconds)) if lo is not None else (seconds, seconds)
                slack = max(1.0, _TOLERANCE * wall)
                if not (low - slack <= wall <= high + slack):
                    hits.append(f"{HIT}: the paper says the run took {_plain(seconds) if lo is None else m.group(0)} "
                                f"(\"{_shown(s)}\"), but {measured}. Write {{{{run_time}}}} instead of a typed time.")
                    flagged = True
            elif least is not None and seconds < least * 0.95 and not _APPROX.search(before):
                hits.append(f"{HIT}: the paper says the run took {_plain(seconds)} (\"{_shown(s)}\"), but {measured}. "
                            "Write {{run_time}} instead of a typed time.")
                flagged = True
        if not over or flagged:
            continue
        for m in _MET.finditer(s):
            if m.group("neg") or (m.group("neg2") or "").strip():
                continue
            what = m.group("what") if m.group("word") else m.group("what2")
            word = (m.group("word") or m.group("word2") or "").lower()
            clause, rel, _opened = _clause(s, m.start(), m.end())
            if _OTHER_LIMIT.search(what or ""):
                continue  # an accuracy, memory or page limit
            near = " ".join([*clause[:rel].split()[-3:], m.group(0), *clause[rel + len(m.group(0)):].split()[:4]])
            about_time = bool(_TIME_WORD.search(near)) or (word in _STRONG_LIMIT and bool(_DONE_VERB.search(clause[:rel + 1])))
            if not about_time:
                continue
            if re.search(r"\b(?:not|n't|never|did\s+not|failed|exceed\w*|over)\b", clause[max(0, rel - 25):rel], re.I):
                continue
            hits.append(f"{HIT}: the paper says the run kept to the topic's time limit (\"{_shown(s)}\"), but {measured}, "
                        f"over the limit of {_plain(limit)}. Write {{{{time_limit_met}}}} (FI puts in \"over\"); do "
                        "not say the limit was met.")  # type: ignore[arg-type]
            break
    return list(dict.fromkeys(hits))


def check(paper_md: str, quest_root: Path | str, *, topic: str = "", fallback_s: Any = None) -> list[str]:
    """Every must-fix hit of the records check for ``paper_md``: a required placeholder missing or misplaced, a run time
    typed outside a placeholder that disagrees with the measurement, a limit claim the measurement contradicts. Never
    raises."""
    try:
        return [*placement_hits(paper_md, quest_root, topic=topic, fallback_s=fallback_s),
                *stated_hits(paper_md, quest_root, topic=topic, fallback_s=fallback_s)]
    except Exception:  # noqa: BLE001 -- a check that cannot be made flags nothing
        return []


def keep_check(quest_root: Path | str, hits: list[str], paper_md: str) -> None:
    """Keep what the last review found (``.fi/paper_record_check.json``): the hits, the required items the paper still does
    not place in its Limitations, and the placeholders that read "not recorded". The quality report prints it, so what is
    still unmet when the rounds are used up is on record."""
    path = Path(quest_root) / RECORD
    try:
        text = _disclosure.without_block(paper_md)
        spans = _limitations_spans(text)
        unmet = [{"kind": i.kind, "id": i.id, "name": i.name, "text": i.sentence}
                 for i in required_items(quest_root)
                 if not any(any(a <= p < b for a, b in spans) for p in _placed(text, _ITEM_KEY + i.id))]
        unfilled = unfilled_in(paper_md)
        if not hits and not unmet and not unfilled:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"at": time.time(), "hits": list(hits), "unmet_items": unmet, "unfilled": unfilled},
                                   indent=1) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        pass


def kept_check(quest_root: Path | str) -> dict[str, Any]:
    data = _read(Path(quest_root), str(RECORD))
    return data if isinstance(data, dict) else {}
