"""The one to-do card: everything a quest is waiting for the person to do, in one place.

When a quest stops for the person, the card says why it stopped, what there is to decide, what FI recommends and
what else can be done, then the one command that goes on. Things that did not stop the quest but are worth a look (a
check that warned, a protocol change waiting for approval, papers that could not be fetched, a run that failed) are
listed under it, so nothing waiting is only in ``run.log``.

The card is written as ``NEXT_STEP.md`` (to read) and ``.fi/todo.json`` (for the web page and VS Code), and the CLI
prints it when the quest stops. Each pause has a recommendation and alternatives written here, by kind; the place that
pauses can pass better ones (the oracle stop names the change it proposes).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TODO_NAME = "todo.json"

# Per pause kind: (what there is to decide, the recommendation, the alternatives).
_ADVICE: dict[str, tuple[str, str, list[str]]] = {
    "clarify": (
        "Is the research setup right?",
        "Check each suggested answer and accept the ones that are right.",
        ["Change any answer before going on."],
    ),
    "papers": (
        "Do you want to add the papers FI could not download?",
        "Download the listed papers (most relevant first) into inputs/papers/, then go on.",
        ["Go on without them: those papers are used from their abstracts only."],
    ),
    "plan": (
        "Is the plan what you want to run?",
        "Read plan.md; if it says what you want, go on.",
        ["Edit the design block in plan.md and save it, then go on.",
         "Ask for a change in words: `--revise-plan \"<what to change>\"`."],
    ),
    "model_output": (
        "A model's answer was cut off at its output limit, or withheld by the provider's filter. How should that "
        "step answer?",
        "Give the step a larger output limit (`provider.node_max_tokens.<step>`) or another model "
        "(`provider.node_models.<step>`), then go on.",
        ["Change what the quest asks (the topic or the plan), then go on."],
    ),
    "code_project": (
        "You edited files in code/ that FI would otherwise update. Keep your version?",
        "Go on: your edited files stay as you wrote them, and FI does not touch them.",
        ["Rename or delete the file you edited, then go on: FI writes its own version again."],
    ),
    "figures": (
        "Which model should read the figures in the papers?",
        "Choose a model that can read images for `provider.node_models.figures`, then go on.",
        ["Turn figure reading off: `knowledge.read_figures: false`."],
    ),
    "supply": (
        "Do you want to add papers, data or example files before FI goes on?",
        "Go on: adding files here is optional.",
        ["Add files to inputs/papers/, inputs/data/ or inputs/examples/ first."],
    ),
    "data": (
        "Which dataset should be analysed?",
        "Put the dataset in data/ (data/README.md says what is expected), then go on.",
        [],
    ),
    "data_split": (
        "Which rows of your data belong together (the same subject, site or device)?",
        "In plan.md, on the line `Rows that belong together:`, name the column that identifies them, then go on.",
        ["Write `independent` on that line if every row is a separate case.",
         "Turn explore-then-confirm off (`engine.phased: false`): the result is then not confirmed on unseen data."],
    ),
    "split": (
        "The simulation and the analysis did not come as the two scripts this quest keeps. Ask for them again, or run "
        "one script?",
        "Go on: the two scripts are asked for again.",
        ["Let the quest run as one script: set `execution.split_failure: warn`."],
    ),
    "manifest": (
        "The run's record does not match the plan. Fix it, or accept it as it is?",
        "Fix the script as listed, then go on.",
        ["Accept it: set `engine.run_manifest_check: warn` (the difference is recorded in the evidence)."],
    ),
    "amendment": (
        "Approve the change to the frozen protocol?",
        "Approve it if it is what you meant: `--approve-amendment` (see the command below).",
        ["Go on without approving: the frozen protocol stays as it was."],
    ),
    "numeric": (
        "The run printed numerical warnings. Fix the script, or accept them?",
        "Fix experiment.py (a changed script is run again), then go on.",
        ["Go on unchanged: the warnings are accepted and recorded, and the evidence then does not count the run as "
         "holding to its protocol."],
    ),
    "oracle": (
        "A known-answer check did not pass. Is the simulation wrong, or the check?",
        "Compare the measured value with the expected one and where it comes from, fix whichever is wrong, then go on.",
        ["Change the check's expected value or tolerance in the plan: `--revise-plan \"...\"`."],
    ),
    "improve": (
        "A change FI made to the simulation made a check of correctness worse. Go on with the version kept so far?",
        "Read code/CHANGELOG.md (every round and its values), then go on: the version kept so far is in code/ and FI "
        "makes no more changes to it.",
        ["Change the simulation yourself first (code/ is yours to edit), then go on: your version is run in full."],
    ),
    "code_layout": (
        "The model's equations are not yet in a package of their own in code/ (or its unit tests or equation list are "
        "missing). Fix it yourself, or keep two scripts?",
        "Fix what is listed in code/, then go on: the folder is read again before anything runs.",
        ["Keep two scripts: set `execution.code_package: false` in the quest's YAML, then go on."],
    ),
    "equation_labels": (
        "The simulation does not say where it implements each equation of the plan. Label it?",
        "Add a comment with each missing equation's id (`# E1`) above the code that computes it, then go on.",
        ["If the equation does not produce the data (it is used on the results), change its role to `analyses` in the "
         "plan while it is not yet frozen: `--revise-plan \"...\"`."],
    ),
    "protocol": (
        "The experiment does not follow the plan. Fix the script, or change the plan?",
        "Fix the named script(s) so they follow the plan, then go on.",
        ["Change the plan's protocol instead (`--revise-plan`), unless it is frozen."],
    ),
    "replicate_seed_unrepairable": (
        "The script ignores the seed FI gives each repeat. Fix it, or run without repeats?",
        "Fix the script so every random draw uses FI_REPLICATE_SEED, then go on.",
        ["Set `rigor_profile: default`: the result is then reported as a single run."],
    ),
    "review_unavailable": (
        "The paper is written but could not be reviewed. Try again later?",
        "Go on later: the review is tried again.",
        [],
    ),
    "review": (
        "Accept the paper, ask for another revision, or reject it?",
        "Read the paper and the review, then accept it or ask for a revision.",
        ["Accept, answering whether you reviewed the evidence record and accept the claims and the limits listed: "
         "`--accept yes` (only yes can make it publication-ready), `--accept partly \"what you do not accept\"` or "
         "`--accept not-checked` (if you do not accept them, ask for a revision instead).",
         "Ask for a revision: `--refine \"<what to change>\"`.", "Reject: `--reject`."],
    ),
    "results": (
        "The experiment runs as a background job. Wait for it here?",
        "Let FI watch the job and go on when it is done: `--watch`.",
        ["Go on yourself once the job is done: `--resume`."],
    ),
    "review_models_collapsed": (
        "The reviewers were set up on different models, but one model answered them all. Wait for the other model, or go on?",
        "Resume once the other model can be reached again: the review runs from the start.",
        ["Go on with one model's review: add `one_model_review: true` under `engine:` in the quest's config.yaml and "
         "resume. The result is then not publication-ready, and says why."],
    ),
    "review_models": (
        "Which model should one reviewer use? A research result needs a reviewer on a model other than the rest.",
        "Choose a model for one reviewer (the steps say where), then go on.",
        ["Run it without rigor_profile: research: the panel may then share one model."],
    ),
    "plan_changed": (
        "Settings that decide how strictly the quest is checked changed after you approved it. Keep the change?",
        "If you meant the change, approve it: `--update` (it shows the settings and records them).",
        ["Put the setting back in config.yaml, then go on."],
    ),
}
_CHECK_UNKNOWN = (
    "A required check could not judge. Fix what stopped it, or go on without it?",
    "Look at the check's entry in .fi/run.log, fix what stopped it, then go on: it is tried once more.",
    ["Go on as it is: the missing check is recorded as a gap in the evidence."],
)


@dataclass
class Item:
    kind: str
    why: str
    decide: str = ""
    recommended: str = ""
    alternatives: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    blocking: bool = False
    # The "why it stopped" card (core/oracle_card.py): one structured payload, rendered here into NEXT_STEP.md and the
    # terminal, and shown by the web page and VS Code from .fi/pause.json. ``None`` for a pause that has none.
    card: dict[str, Any] | None = None


def advice(kind: str) -> tuple[str, str, list[str]]:
    """What there is to decide at a pause of ``kind``, the recommendation and the alternatives."""
    if kind.endswith("_unknown"):
        return _CHECK_UNKNOWN
    decide, rec, alts = _ADVICE.get(kind, ("", "", []))
    return decide, rec, list(alts)


_SETTING_RE = re.compile(r"`(?:(?P<section>engine|execution|pauses)\.)?(?P<key>\w+):\s*(?P<value>[^`]+?)\s*`")

#: What a person under ``rigor_profile: research`` can do instead of relaxing a check the profile holds fixed, by
#: pause kind, before (``False``) and after (``True``) the protocol is frozen. A frozen protocol cannot be changed from
#: a stop before the review, so after the freeze nothing here points at the plan.
_RESEARCH_INSTEAD: dict[str, dict[bool, str]] = {
    "oracle": {
        False: "This quest is set up for research, so the known-answer checks cannot be relaxed: fix the script and "
               "resume, or, if a check itself is wrong, change it in the plan (`--revise-plan`) and resume; or, if it "
               "was measured and you judge the check wrong, mark it unconfirmed and go on (with your name: the result "
               "then says the check failed and is never publication-ready).",
        True: "This quest is set up for research and its protocol is frozen, so the known-answer checks cannot be "
              "relaxed or changed inside this quest: fix the script and resume; or, if a check was measured and you "
              "judge the check itself wrong, mark it unconfirmed and go on (with your name: the result then says the "
              "check failed and is never publication-ready); or start a new quest whose plan states the right check.",
    },
    "split": {
        False: "This quest is set up for research, so it always keeps the simulation and the analysis apart: resume, "
               "and the two scripts are asked for again.",
        True: "This quest is set up for research, so it always keeps the simulation and the analysis apart: resume, "
              "and the two scripts are asked for again.",
    },
    "improve": dict.fromkeys(
        (False, True),
        "This quest is set up for research, so a change that made a check of correctness worse stops it: the version "
        "kept so far is back in code/; resume to go on with it.",
    ),
    "equation_labels": dict.fromkeys(
        (False, True),
        "This quest is set up for research, so the simulation must say where it implements each equation of the plan: "
        "add the labels and resume.",
    ),
    "code_layout": dict.fromkeys(
        (False, True),
        "This quest is set up for research, so its code keeps the model's equations apart, with unit tests and an "
        "equation list: fix what is listed in code/ and resume, or set `execution.code_package: false` to keep two "
        "scripts.",
    ),
}
_RESEARCH_INSTEAD_ANY = {
    False: "This quest is set up for research, so that check cannot be relaxed: fix what it found and resume (FI "
           "repairs the script), change the plan (`--revise-plan`), or start a new quest set up to explore.",
    True: "This quest is set up for research and its protocol is frozen, so that check cannot be relaxed: fix what it "
          "found and resume (FI repairs the script), or start a new quest if the protocol itself is wrong.",
}


def research_instead(kind: str, *, frozen: bool) -> str:
    """What to do instead of relaxing a check, under ``rigor_profile: research``."""
    return _RESEARCH_INSTEAD.get(kind, _RESEARCH_INSTEAD_ANY)[frozen]


#: Kept for callers that name no pause: the text before the freeze.
RESEARCH_INSTEAD = _RESEARCH_INSTEAD_ANY[False]


def refused_settings(text: str, profile: str) -> list[str]:
    """The settings ``text`` suggests (``engine.run_manifest_check: warn``, ...) that ``profile`` refuses."""
    if profile != "research":
        return []
    from .config import _RESEARCH_PROFILE  # the one table of what research holds fixed

    out = []
    for m in _SETTING_RE.finditer(text or ""):
        sections = [m["section"]] if m["section"] else list(_RESEARCH_PROFILE)
        for section in sections:
            forced = (_RESEARCH_PROFILE.get(section) or {}).get(m["key"])
            if forced is None or isinstance(forced, list):
                continue
            if _plain(m["value"]) != _plain(forced):
                out.append(m.group(0))
    return out


def _plain(value: Any) -> str:
    return str(value).strip().strip("'\"").strip().lower()


def _without_refused(lines: list[str], profile: str) -> tuple[list[str], bool]:
    """``lines`` with every sentence or clause (split at ``.``, ``;``, ``:``) that suggests a refused setting dropped
    (a line left empty goes); and whether any was dropped."""
    kept, dropped = [], False
    for line in lines:
        if not refused_settings(line, profile):
            kept.append(line)
            continue
        dropped = True
        parts = re.split(r"(?<=[.:])\s+(?=[A-Z(`])|;\s+", line)
        rest = [p.strip() for p in parts if p.strip() and not refused_settings(p, profile)]
        text = "; ".join(rest).strip()
        if text and text[-1] not in ".!?":
            text += "."
        if text:
            kept.append(text[0].upper() + text[1:])
    return kept, dropped


def pause_item(kind: str, headline: str, steps: list[str], *, recommended: str | None = None,
               alternatives: list[str] | None = None, profile: str = "default", frozen: bool = False,
               card: dict[str, Any] | None = None) -> Item:
    """The item for the pause that stopped the quest. Under ``profile`` ``research`` no line suggests a setting the
    profile refuses (it would be refused when the quest is resumed); what the person can do instead is said, for this
    kind of pause and whether the protocol is ``frozen``."""
    decide, rec, alts = advice(kind)
    rec = recommended or rec
    alts = list(alternatives) if alternatives is not None else alts
    steps, dropped_steps = _without_refused(list(steps), profile)
    alts, dropped_alts = _without_refused(alts, profile)
    if profile == "research" and frozen:
        # After the freeze the plan no longer changes the protocol: a suggestion to rewrite it would contradict the card.
        alts = [a for a in alts if "--revise-plan" not in a]
    instead = research_instead(kind, frozen=frozen)
    if refused_settings(rec, profile):
        rec, dropped_alts = instead, True
    elif dropped_steps or dropped_alts:
        alts.append(instead)
    return Item(kind=kind, why=headline, decide=decide, recommended=rec, alternatives=alts, steps=steps, blocking=True,
                card=card)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _supplied_since(folder: Path, record: Path) -> bool:
    """Whether a file was put in ``folder`` after ``record`` was written: the papers asked for were supplied."""
    try:
        written = record.stat().st_mtime
        return any(p.is_file() and p.stat().st_mtime > written for p in folder.iterdir())
    except OSError:
        return False


#: What the card says when the current version's confirmation on unseen data did not hold (``core/confirmations.py``).
CONFIRM_FAILED = ("The confirmation on unseen data did not agree; this version cannot be confirmed again — a changed "
                  "version is a new candidate and is confirmed on its own.")


def confirm_item(quest_root: Path) -> Item | None:
    """The card's line when the current version of the study had its one confirm run and it did not hold (or it was
    run again): the result stays exploratory, and only a changed version, confirmed on its own, can do better."""
    from . import confirmations as _confirmations
    from . import phased as _phased

    if not _confirmations.path(Path(quest_root)).is_file():
        return None  # nothing confirmed yet (and a quest without the two stages never reads their record)
    record = _phased.load(Path(quest_root))
    if record is None:
        return None
    mine = _confirmations.for_candidate(_confirmations.read(Path(quest_root)), _phased.candidate(record))
    verdict = _confirmations.worst(mine)
    if not mine or verdict == _confirmations.CONFIRMED:
        return None
    return Item("confirm", f"{CONFIRM_FAILED} ({verdict.replace('_', ' ')}; .fi/confirmations.jsonl)",
                recommended="Nothing to do if the exploratory result is enough. To try a changed version, say what to "
                            "change (`--refine \"...\"`): it is run, explored and confirmed once on its own; the "
                            "earlier confirmation stays in the record and the paper counts it.")


def waiting(quest_root: Path) -> list[Item]:
    """Things that did not stop the quest but are waiting for the person: each from a record the quest already keeps."""
    root = Path(quest_root)
    needs = root / "needs"
    out: list[Item] = []
    failed = root / "quest_failed.md"
    if failed.is_file():
        from . import crash_kind as _crash_kind

        plain = _crash_kind.read(root / ".fi")
        if plain:
            out.append(Item("failed", f"The last run stopped: {plain['say']}",
                            recommended=f"{plain.get('do') or ''} (`--resume` continues from the step that stopped; the "
                                        "details for a bug report are in quest_failed.md.)".strip()))
        else:
            out.append(Item("failed", "The last run stopped with an error (quest_failed.md says where).",
                            recommended="Read quest_failed.md; once the cause is fixed, go on: the quest continues "
                                        "from the step that failed."))
    pending = _read_json(needs / "PROTOCOL_AMENDMENT_PENDING.json")
    if isinstance(pending, dict):
        changes = pending.get("changes") or []
        out.append(Item("amendment", f"A change to the frozen protocol is waiting for your approval ({len(changes)} "
                                     f"change(s), needs/PROTOCOL_AMENDMENT_PENDING.json).",
                        decide=_ADVICE["amendment"][0], recommended=_ADVICE["amendment"][1],
                        alternatives=list(_ADVICE["amendment"][2])))
    for name, what in (("PROTOCOL_CHECK.json", "the plan's protocol"), ("ORACLE_CHECK.json", "the known-answer checks"),
                       ("RUN_MANIFEST_CHECK.json", "the run's record against the plan")):
        record = _read_json(needs / name)
        if isinstance(record, dict) and record.get("status") == "warned":
            out.append(Item("warned", f"The check of {what} found differences and was set to warn, so the quest went "
                                      f"on (needs/{name}).",
                            recommended="Read the differences; nothing to do if they are expected."))
    oracle = _read_json(needs / "ORACLE_CHECK.json")
    if isinstance(oracle, dict) and oracle.get("status") == "went_on_failing":
        from .accepted_checks import gap as _went_on_gap

        from .oracle_check import proposal_request

        sentences = [_went_on_gap(e) for e in oracle.get("went_on") or [] if isinstance(e, dict) and e.get("name")]
        # A change to a check that FI or a repair proposed, kept for the person to accept or not.
        proposed = [p for p in oracle.get("proposed_changes") or [] if isinstance(p, dict) and p.get("name")]
        offers = []
        for p in proposed:
            try:
                offers.append(f"proposed for '{p['name']}' (to accept it: `--revise-plan \"{proposal_request(p)}\"`)")
            except Exception:  # noqa: BLE001 -- a proposal that cannot be shown is still in the record
                continue
        out.append(Item("went_on", "A known-answer check failed and the quest went on: "
                                   + ("; ".join(sentences) or "see needs/ORACLE_CHECK.json") + ".",
                        recommended="Compare the measured value with the expected one and where it comes from: if the "
                                    "check is right, the simulation is wrong; if the check is wrong, change it. Until "
                                    "then the result does not count as checked against known answers."
                                    + (" A change to the check was " + "; ".join(offers) + " (once the protocol is "
                                       "frozen, this becomes an amendment you approve)." if offers else "")))
    wanted = needs / "WANTED_PAPERS.md"
    if wanted.is_file() and not _supplied_since(root / "inputs" / "papers", wanted):
        out.append(Item("papers", "Some papers could not be downloaded (needs/WANTED_PAPERS.md lists them, most "
                                  "relevant first).",
                        recommended="Nothing to do unless one of them matters: put its PDF in inputs/papers/ and go on."))
    declined = _read_json(root / ".fi" / "papers_declined.json")
    if isinstance(declined, list) and declined:
        out.append(Item("papers_declined", f"{len(declined)} paper(s) you went on without are read from their "
                                           "abstracts only (.fi/papers_declined.json).",
                        recommended="Nothing to do unless one of them matters: put its PDF in inputs/papers/; it is "
                                    "read in full the next time the quest searches the literature."))
    knowledge = _read_json(root / ".fi" / "knowledge_problem.json")
    if isinstance(knowledge, dict) and knowledge.get("problem"):
        out.append(Item("knowledge", "The knowledge base's copy of this result is not current: "
                                     + str(knowledge["problem"]) + " (.fi/knowledge_problem.json).",
                        recommended="Once the Axon service is reachable, run `fi tools tidy-knowledge`, then resume "
                                    "the quest to write it back again."))
    failures =_read_json(root / ".fi" / "source_failures.json")
    # The record is written on every run; only one that counts a failure is worth a look.
    if (isinstance(failures, dict) and failures.get("total")) or (isinstance(failures, list) and failures):
        out.append(Item("sources", "Some literature sources could not be reached (.fi/source_failures.json).",
                        recommended="Nothing to do unless the literature looks thin: go on later, or add papers to "
                                    "inputs/papers/."))
    from .retractions import retracted_in_record

    retracted = retracted_in_record(root / ".fi")
    if retracted:
        named = "; ".join(t[:80] for t in retracted[:3]) + (f"; and {len(retracted) - 3} more" if len(retracted) > 3
                                                            else "")
        out.append(Item("retracted", f"{len(retracted)} source(s) the literature search found have been retracted "
                                     f"({named}). The paper must not cite them.",
                        recommended="Make sure the paper does not cite them; the claim check marks any sentence that "
                                    "does as unsupported (.fi/literature_queries.json lists the retraction notices)."))
    confirm = confirm_item(root)
    if confirm is not None:
        out.append(confirm)
    from .evidence import read as _read_evidence  # with its seal checked, as every surface shows it

    evidence = _read_evidence(root)
    if isinstance(evidence, dict) and evidence.get("gaps") and evidence.get("next_level"):
        from .evidence import summary_line  # the level's own plain sentence, never its identifier

        out.append(Item("evidence", "What this result shows, and what stands in the way of more: "
                                    + summary_line(evidence) + " (needs/EVIDENCE.json)",
                        recommended="Nothing to do unless you need more than this."))
    return out


def resume_lines(quest_id: str) -> list[str]:
    return [
        f"- **CLI:** `python launch.py --resume {quest_id}` (or `fi --resume {quest_id}`)",
        "- **Web / VS Code:** open the quest and click **Resume**, or `@fi /resume " + quest_id + "`.",
    ]


def _check_lines(c: dict[str, Any], *, markdown: bool) -> list[str]:
    """One check of a "why it stopped" card: what it is, its numbers, where they come from and where in the script."""
    kind = f", {c['kind_words'] or c['kind']}" if (c.get("kind_words") or c.get("kind")) else ""
    head = (f"### Check “{c.get('name') or c.get('id')}” (`{c.get('id')}`{kind})" if markdown else
            f"Check “{c.get('name') or c.get('id')}” ({c.get('id')}{kind})")
    rows: list[str] = []
    if c.get("expected_text"):
        rows.append(f"Expected: {c['expected_text']} — from: {c.get('reference') or 'the check does not say'}")
    if c.get("measured_text"):
        rows.append(f"Measured: {c['measured_text']} — by {c.get('measured_by_text')}")
    elif c.get("reported"):
        rows.append(f"Measured: {c['reported']}, which is not a finite number, so it cannot be compared")
    elif c.get("status") == "not_measured":
        rows.append("Measured: nothing (the value could not be measured)")
    elif c.get("status") == "cannot_judge":
        rows.append("Measured: not run (the check gives no number to compare with)")
    elif c.get("status") == "not_run":
        rows.append("Measured: not run yet (FI runs the checks once every check has its numbers)")
    if c.get("limit_text"):
        mode = c.get("tolerance_mode") or "absolute"
        tol = c.get("tolerance")
        rows.append(f"Tolerance: ±{c['limit_text']} ({mode}" + (f", {tol:g} of the expected value" if mode == "relative"
                                                                  and isinstance(tol, (int, float)) else "") + ")")
    if c.get("gap"):
        rows.append(f"Gap: {c['gap']['text']}")
    if c.get("disputed"):
        rows.append("FI's repair disputes this check's expected value (nobody has approved a change, so it counts as "
                    "failed).")
    case = ", ".join(f"{k}={v}" for k, v in (c.get("case") or {}).items())
    if case or c.get("measure"):
        rows.append("Case: " + (case or "none (the script's own value)")
                    + (f"; measure: `{c['measure']}`" if c.get("measure") else ""))
    if c.get("error"):
        rows.append(f"Error: {c['error']}" + (f" (at {c['error_at']})" if c.get("error_at") else ""))
    rows += [f"Found: {f}" for f in c.get("found") or []]
    where = c.get("where")
    if where:
        rows.append(f"In the script: {where['file']} line {where['line']} — {where['what']}")
    out = [head]
    out += [f"- {r}" for r in rows] if markdown else [f"  {r}" for r in rows]
    if where and where.get("excerpt") and markdown:
        # The fence at the start of the line: the web page's renderer reads only that one as a code block.
        out += ["", "```python", *where["excerpt"], "```"]
    return out


def card_lines(card: dict[str, Any], *, markdown: bool = True) -> list[str]:
    """The "why it stopped" card (core/oracle_card.py) as lines: Markdown for NEXT_STEP.md (``markdown``) or plain text
    for a terminal. One rendering of one payload, so every surface says the same thing."""
    if not isinstance(card, dict):
        return []
    checks = [c for c in card.get("checks") or [] if isinstance(c, dict)]
    causes = [c for c in card.get("causes") or [] if isinstance(c, dict)]
    actions = [a for a in card.get("actions") or [] if isinstance(a, dict)]
    out: list[str] = []
    if markdown:
        out += ["## Why it stopped", str(card.get("summary") or ""), ""]
        for c in checks:
            out += [*_check_lines(c, markdown=True), ""]
        if card.get("also_found"):
            out += ["**Also found:**", *(f"- {f}" for f in card["also_found"]), ""]
        if causes:
            out += ["## Most likely cause", *(f"{n}. {c.get('text')} {c.get('evidence') or ''}".rstrip()
                                              for n, c in enumerate(causes, 1)), ""]
        if card.get("tried"):
            out += ["## What FI already tried", *(f"- {t}" for t in card["tried"]), ""]
        if actions:
            out += ["## What you can do"]
            for n, a in enumerate(actions, 1):
                ways = [f"CLI: `{a['cli']}`" if a.get("cli") else "", f"web: **{a['web']}**" if a.get("web") else "",
                        f"VS Code: `{a['vscode']}`" if a.get("vscode") else ""]
                said = "; ".join(w for w in ways if w)
                out.append(f"{n}. **{a.get('label')}.** {a.get('detail') or ''}" + (f" ({said})" if said else ""))
            out.append("")
        if card.get("notes"):
            out += [*(f"> {n}" for n in card["notes"]), ""]
        return out
    out.append(str(card.get("summary") or ""))
    for c in checks:
        out += _check_lines(c, markdown=False)
    out += [f"Also found: {f}" for f in card.get("also_found") or []]
    if causes:
        out += ["Most likely: " + causes[0].get("text", "") + " " + (causes[0].get("evidence") or "")]
        out += [f"  also: {c.get('text')}" for c in causes[1:3]]
    if card.get("tried"):
        out += ["FI already tried: " + " ".join(card["tried"])]
    if actions:
        out += ["You can:"]
        for n, a in enumerate(actions, 1):
            out.append(f"  {n}. {a.get('label')}: {a.get('detail') or ''}"
                       + (f" `{a['cli']}`" if a.get("cli") and a.get("id") != "edit" else ""))
    out += [f"Note: {n}" for n in card.get("notes") or []]
    return out


def render(quest_id: str, items: list[Item]) -> str:
    """The card as Markdown: the pause first (why, what to decide, the recommendation, the alternatives, what to do),
    then everything else waiting, then how to go on. A pause with a "why it stopped" card shows the card instead of
    the generic parts (its actions are the recommendation and the alternatives, with their commands)."""
    blocking = [i for i in items if i.blocking]
    rest = [i for i in items if not i.blocking]
    lines: list[str] = []
    if blocking and blocking[0].card:
        first = blocking[0]
        lines += [f"# Action needed — {first.why}", "", f"Quest **{quest_id}** is paused and waiting for you.", ""]
        lines += card_lines(first.card, markdown=True)
    elif blocking:
        first = blocking[0]
        lines += [f"# Action needed — {first.why}", "", f"Quest **{quest_id}** is paused and waiting for you.", ""]
        if first.decide:
            lines += ["## What to decide", first.decide, ""]
        if first.recommended:
            lines += ["## Recommended", first.recommended, ""]
        if first.alternatives:
            lines += ["## Or", *(f"- {a}" for a in first.alternatives), ""]
        if first.steps:
            lines += ["## What to do", *(f"{n}. {s}" for n, s in enumerate(first.steps, 1)), ""]
    else:
        lines += [f"# Waiting for you — quest {quest_id}", ""]
    if rest:
        lines += ["## Also waiting for you" if blocking else "## Worth a look",
                  *(f"- **{i.why}** {i.recommended}".rstrip() for i in rest), ""]
    lines += ["## Then go on", *resume_lines(quest_id), ""]
    return "\n".join(lines)


def write(quest_root: Path, fi_dir: Path, quest_id: str, pause: Item | None) -> list[Item]:
    """Write the card for ``pause`` and everything else waiting, and return the items. At a pause the card is
    ``NEXT_STEP.md`` and ``.fi/todo.json``; with no pause (a finished quest) only ``.fi/todo.json``, since a
    ``NEXT_STEP.md`` means "paused" to every interface, and none at all when nothing is waiting. Best-effort: a card
    that cannot be written never stops a quest."""
    todo = Path(fi_dir) / TODO_NAME
    try:
        items = ([pause] if pause else []) + waiting(quest_root)
        if pause:
            (Path(quest_root) / "NEXT_STEP.md").write_text(render(quest_id, items), encoding="utf-8")
        if items:
            Path(fi_dir).mkdir(parents=True, exist_ok=True)
            todo.write_text(
                json.dumps({"quest_id": quest_id, "items": [asdict(i) for i in items]}, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            todo.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 -- a card that cannot be written must never stop a quest or hide its error
        items = [pause] if pause else []
    return items


def read(fi_dir: Path) -> list[dict[str, Any]]:
    """The items of ``.fi/todo.json`` (empty when there is none)."""
    record = _read_json(Path(fi_dir) / TODO_NAME)
    items = record.get("items") if isinstance(record, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def text(fi_dir: Path) -> str:
    """The card as plain text for a terminal ('' when nothing is waiting)."""
    items = read(fi_dir)
    if not items:
        return ""
    out: list[str] = []
    for item in items:
        if item.get("blocking") and isinstance(item.get("card"), dict):
            # The numbers a person needs to decide, here and not only in a file (NEXT_STEP.md has the same card).
            out += [f"[FI] Waiting for you: {item.get('why')}"]
            out += [f"     {line}" for line in card_lines(item["card"], markdown=False)]
            out += ["     (The same card, with the script excerpt: NEXT_STEP.md)"]
        elif item.get("blocking"):
            out += [f"[FI] Waiting for you: {item.get('why')}"]
            if item.get("decide"):
                out += [f"     To decide: {item['decide']}"]
            if item.get("recommended"):
                out += [f"     Recommended: {item['recommended']}"]
            out += [f"     Or: {a}" for a in item.get("alternatives") or []]
        else:
            out += [f"[FI] Also: {item.get('why')} {item.get('recommended') or ''}".rstrip()]
    return "\n".join(out)
