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
    "split": (
        "The simulation does not do what the plan fixed. Fix it, or accept it as it is?",
        "Fix the script as listed, then go on.",
        ["Accept it: set `execution.split_failure: warn` (the result is then marked as the script's own record)."],
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
        ["Go on unchanged: the warnings are accepted and recorded."],
    ),
    "oracle": (
        "The script has not passed its oracle checks (a result with a known answer). Fix the script, or the check?",
        "Fix the script, then go on.",
        ["Change the check's expected value or tolerance in the plan: `--revise-plan \"...\"`."],
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
        ["Accept: `--accept`.", "Ask for a revision: `--refine \"<what to change>\"`.", "Reject: `--reject`."],
    ),
    "results": (
        "The experiment runs as a background job. Wait for it here?",
        "Let FI watch the job and go on when it is done: `--watch`.",
        ["Go on yourself once the job is done: `--resume`."],
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


def advice(kind: str) -> tuple[str, str, list[str]]:
    """What there is to decide at a pause of ``kind``, the recommendation and the alternatives."""
    if kind.endswith("_unknown"):
        return _CHECK_UNKNOWN
    decide, rec, alts = _ADVICE.get(kind, ("", "", []))
    return decide, rec, list(alts)


def pause_item(kind: str, headline: str, steps: list[str], *, recommended: str | None = None,
               alternatives: list[str] | None = None) -> Item:
    """The item for the pause that stopped the quest."""
    decide, rec, alts = advice(kind)
    return Item(kind=kind, why=headline, decide=decide, recommended=recommended or rec,
                alternatives=list(alternatives) if alternatives is not None else alts, steps=list(steps), blocking=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def waiting(quest_root: Path) -> list[Item]:
    """Things that did not stop the quest but are waiting for the person: each from a record the quest already keeps."""
    root = Path(quest_root)
    needs = root / "needs"
    out: list[Item] = []
    failed = root / "quest_failed.md"
    if failed.is_file():
        out.append(Item("failed", "The last run stopped with an error (quest_failed.md says where).",
                        recommended="Read quest_failed.md; once the cause is fixed, go on: the quest continues from "
                                    "the step that failed."))
    pending = _read_json(needs / "PROTOCOL_AMENDMENT_PENDING.json")
    if isinstance(pending, dict):
        changes = pending.get("changes") or []
        out.append(Item("amendment", f"A change to the frozen protocol is waiting for your approval ({len(changes)} "
                                     f"change(s), needs/PROTOCOL_AMENDMENT_PENDING.json).",
                        decide=_ADVICE["amendment"][0], recommended=_ADVICE["amendment"][1],
                        alternatives=list(_ADVICE["amendment"][2])))
    for name, what in (("PROTOCOL_CHECK.json", "the plan's protocol"), ("ORACLE_CHECK.json", "the oracle checks"),
                       ("RUN_MANIFEST_CHECK.json", "the run's record against the plan")):
        record = _read_json(needs / name)
        if isinstance(record, dict) and record.get("status") == "warned":
            out.append(Item("warned", f"The check of {what} found differences and was set to warn, so the quest went "
                                      f"on (needs/{name}).",
                            recommended="Read the differences; nothing to do if they are expected."))
    wanted = needs / "WANTED_PAPERS.md"
    if wanted.is_file():
        out.append(Item("papers", "Some papers could not be downloaded (needs/WANTED_PAPERS.md lists them, most "
                                  "relevant first).",
                        recommended="Nothing to do unless one of them matters: put its PDF in inputs/papers/ and go on."))
    failures = _read_json(root / ".fi" / "source_failures.json")
    if isinstance(failures, (list, dict)) and failures:
        out.append(Item("sources", "Some literature sources could not be reached (.fi/source_failures.json).",
                        recommended="Nothing to do unless the literature looks thin: go on later, or add papers to "
                                    "inputs/papers/."))
    evidence = _read_json(needs / "EVIDENCE.json")
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


def render(quest_id: str, items: list[Item]) -> str:
    """The card as Markdown: the pause first (why, what to decide, the recommendation, the alternatives, what to do),
    then everything else waiting, then how to go on."""
    blocking = [i for i in items if i.blocking]
    rest = [i for i in items if not i.blocking]
    lines: list[str] = []
    if blocking:
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
    items = ([pause] if pause else []) + waiting(quest_root)
    todo = Path(fi_dir) / TODO_NAME
    try:
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
    except OSError:
        pass
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
        if item.get("blocking"):
            out += [f"[FI] Waiting for you: {item.get('why')}"]
            if item.get("decide"):
                out += [f"     To decide: {item['decide']}"]
            if item.get("recommended"):
                out += [f"     Recommended: {item['recommended']}"]
            out += [f"     Or: {a}" for a in item.get("alternatives") or []]
        else:
            out += [f"[FI] Also: {item.get('why')} {item.get('recommended') or ''}".rstrip()]
    return "\n".join(out)
