"""Rerun a quest from a chosen step (``--resume <quest_id> --from <step>``).

The quest's checkpoint history (LangGraph's ``state.sqlite``) holds the state before every step it ran. Rerunning from
a step continues from the latest checkpoint taken just before that step ran, so everything the quest decided up to
there is kept and everything from there on is done again. The outputs of that step and the ones after it are first
moved to ``.fi/previous/<time>/``, so the new ones never mix with the old and both can be compared.

Every step a person can pick has three things, kept together here: the graph nodes it starts at (``STEPS``), one
sentence saying what redoing it does and what it keeps (``REDOES``), and the files it and the later steps wrote
(``OUTPUTS``, moved aside first). A graph node with no such list and sentence is not offered.

Steps up to the design (``NEEDS_APPROVAL``) change the plan or the frozen protocol, so the engine asks for a named
approval (``--approve-as <you>``) before it does them, and says in plain words that the old protocol is replaced. The
``skills`` step is not one of them: it picks the skills again and leaves the plan and the protocol alone.

Graph nodes that are not a step, and why:

* ``clarify``: redoing it means starting a new quest; ``ideas`` is the first step.
* ``pause_after_literature`` and ``wait_for_data``: pauses. They write nothing of their own.
* ``execute_reflect``: the repair of a crashed script, inside ``run`` (it has its own budget).
* ``human_feedback``: a person's decision, not something to redo.
* ``implement_outline`` / ``implement`` (the ``code`` step), ``auto_collect_data`` / ``data_load`` (the ``run`` step),
  ``web_plots`` / ``web_figures`` (the ``figures`` step) and ``select_skills`` (the ``skills`` step) are parts of a
  step, and their names are accepted as that step.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

# Plain name -> the graph nodes the step is made of (run in that order). The node names are accepted too.
# The order is the graph's order, so the list reads top to bottom as the quest runs.
STEPS: dict[str, tuple[str, ...]] = {
    "ideas": ("ideate",),
    "literature": ("literature",),
    "plan": ("plan",),
    "design": ("design",),
    # The skills are picked again just before the code (or, with no simulation, the data) is made: the same place in
    # the graph as ``code``, so the checkpoint is the same; the pick itself is redone by the engine (``Engine.run``).
    "skills": ("implement_outline", "auto_collect_data"),
    "code": ("implement_outline", "implement"),
    # The no-simulation path's run is the data: collected, waited for, loaded.
    "run": ("execute", "auto_collect_data", "wait_for_data", "data_load"),
    # The no-simulation path draws its plots and figures from the loaded data, before the analysis.
    "figures": ("web_plots", "web_figures"),
    "analysis": ("analyze",),
    "crosscheck": ("cross_check",),
    "evidence": ("evidence_gate",),
    "writing": ("write",),
    "claims": ("claim_check",),
    "review": ("review",),
}
# Nodes a step loops through without leaving it: the run's repair of a crashed script goes back to execute.
_WITHIN: dict[str, tuple[str, ...]] = {"run": ("execute_reflect",)}

# The big blocks a map of the quest shows, and the steps in each.
GROUPS: dict[str, tuple[str, ...]] = {
    "Ideas & literature": ("ideas", "literature"),
    "Skills & plan": ("plan", "skills"),
    "Design": ("design",),
    "Code": ("code",),
    "Run": ("run", "figures"),
    "Analysis": ("analysis", "crosscheck", "evidence"),
    "Writing": ("writing", "claims"),
    "Review": ("review",),
}
# Steps that change the plan or the frozen protocol: they need a named approval.
NEEDS_APPROVAL: frozenset[str] = frozenset({"ideas", "literature", "plan", "design"})

# What running the quest again from each step does, in one sentence a person reads before choosing.
_REPLACES = "the experiment plan (frozen) is replaced by the new design's"
REDOES: dict[str, str] = {
    "ideas": "comes up with the research ideas again, then redoes the literature, plan, design, code and everything "
             f"after; {_REPLACES}, and the old one is kept in .fi/previous/",
    "literature": "searches the literature again, then redoes the plan, design, code and everything after; "
                  f"the ideas are kept, {_REPLACES}, and the old one is kept in .fi/previous/",
    "plan": "writes plan.md again from the same literature and waits for you to read it, then redoes the design, code "
            f"and everything after; {_REPLACES}, and the old one is kept in .fi/previous/",
    "design": "takes the design written in plan.md again (edit plan.md first to change it; `plan` writes it anew), then "
              f"redoes the code and everything after; the literature and the plan are kept, {_REPLACES}, and the old "
              "one is kept in .fi/previous/",
    "skills": "picks the skills again from the quest's current config, writes the code again, then runs it and does "
              "everything after; the literature and the plan are not redone",
    "code": "writes the experiment's code again, then runs it and does everything after",
    "run": "runs the same code again for new results, then analyses, writes and reviews them",
    "figures": "draws the figures again from the same data (a quest with no simulation), then analyses, writes and "
               "reviews",
    "analysis": "analyses the same results again, then writes and reviews",
    "crosscheck": "checks the analysis against the literature and the design again, then weighs the evidence, writes "
                  "and reviews",
    "evidence": "weighs the evidence for the results again, then writes and reviews",
    "writing": "writes the paper again from the same analysis, then reviews it",
    "claims": "checks every claim in the paper against the results again, then reviews it (the paper is not written "
              "again)",
    "review": "reviews the same paper again (and makes the slides and poster from it again)",
}
_ALIASES = {
    "ideate": "ideas", "idea": "ideas", "ideas & literature": "ideas",
    "literature review": "literature", "lit": "literature",
    "select_skills": "skills", "skill": "skills",
    "implement": "code", "implement_outline": "code", "execute": "run", "data_load": "run",
    "auto_collect_data": "run", "web_plots": "figures", "web_figures": "figures", "plots": "figures",
    "analyze": "analysis",
    "cross_check": "crosscheck", "cross-check": "crosscheck", "evidence_gate": "evidence",
    "write": "writing", "paper": "writing", "claim_check": "claims",
}

# What each step and every step after it write, relative to the quest folder. A step's own inputs are not here: a
# rerun from the writing keeps the figures the run drew. An entry with ``*`` matches several files.
_DELIVERED = ["slides.md", "slides.html", "slides.pdf", "slides.pptx", "poster.tex", "poster.pdf", "poster.pptx",
              "poster.html", "speech.md", "frontier_insight_summary.json", "NEXT_STEP.md"]
_PAPER = ["paper", "paper.md", "paper.pdf", "paper.html", "paper_html_body.md", "paper_html_source.html",
          "paper_html_theme.css", "paper_pdf_source.md", *_DELIVERED]
# What the claim check writes into paper/ (the folder itself is the writing step's), and its receipt.
_CLAIM_CHECK = ["paper/claims.json", "paper/CLAIMS.md", "paper/goal_coverage.json", "paper/numeric_audit.json",
                "paper/statistics_audit.json", "paper/provenance_audit.json", "needs/receipts/claim_check.json"]
# What the evidence gate writes.
_EVIDENCE = ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json"]
# What the run and its checks leave under needs/ (the run writes them again).
_RUN_RECORDS = ["needs/RUN_MANIFEST_CHECK.json", "needs/ORACLE_CHECK.json", "needs/PROTOCOL_CHECK.json",
                "needs/ENVIRONMENT.json"]
# The frozen protocol: what a redesign replaces. The amendments and the saved versions stay, so the new protocol is
# frozen as a change made after results were seen and the paper says so.
_PROTOCOL = ["needs/FROZEN_PROTOCOL.json", "needs/PROTOCOL_AMENDMENT_PENDING.json", "needs/AMENDMENT_APPROVAL.json"]
# From the design on: the protocol, the design's audit, and everything the run and the writing made from it.
_FROM_DESIGN = [*_PROTOCOL, "needs/DESIGN_CRITIQUE.json", "needs/receipts", *_RUN_RECORDS, "needs/EVIDENCE.json",
                "code", "figures", "raw", "results.json", "data/auto_collected", *_PAPER]

_OWN: dict[str, list[str]] = {
    # The review writes no file of its own; what is made from the reviewed paper is made again after it.
    "review": _DELIVERED,
    "claims": [*_CLAIM_CHECK, *_DELIVERED],
    "writing": [*_PAPER, *_CLAIM_CHECK],
    "evidence": [*_EVIDENCE, *_PAPER, *_CLAIM_CHECK],
    "crosscheck": [*_EVIDENCE, *_PAPER, *_CLAIM_CHECK],
    "analysis": _PAPER,
    "figures": ["figures", "code/web_plots.py", *_EVIDENCE, *_PAPER, *_CLAIM_CHECK],
    # The mean-over-seeds redraw is written into code/ by the run.
    "run": ["figures", "raw", "results.json", "code/replot_figures.py", "code/replot_figures.json", "code/web_plots.py",
            "data/auto_collected", "needs/RUN_MANIFEST_CHECK.json", "needs/ENVIRONMENT.json", *_PAPER],
    "code": ["code", "figures", "raw", "results.json", *_RUN_RECORDS, *_PAPER],
    "skills": ["code", "figures", "raw", "results.json", *_RUN_RECORDS, *_PAPER],
    # A redesign starts the protocol over: the old one is moved aside, and the run's records with it.
    "design": _FROM_DESIGN,
    "plan": ["plan.md", ".fi/paused_at_plan.flag", "needs/receipts/design_audit.json", *_FROM_DESIGN],
    "literature": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", *_FROM_DESIGN],
    "ideas": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", *_FROM_DESIGN],
}
# A step redoes everything after it, so it moves aside everything the later steps write too (a rerun that stops halfway
# must not leave the old receipts for the evidence level to read).
OUTPUTS: dict[str, list[str]] = {
    step: list(dict.fromkeys(f for later in list(STEPS)[list(STEPS).index(step):] for f in _OWN[later]))
    for step in STEPS
}
# The graph node that leads into a step's first node with a plain edge, for the steps up to the design: the state the
# engine writes before running them again (the design history, see ``Engine.run``) is written as that node's.
LEADS_INTO: dict[str, str] = {"ideas": "clarify", "literature": "ideate", "plan": "select_skills", "design": "plan"}


def resolve(name: str) -> str | None:
    """The plain step name ``name`` stands for, or ``None`` when it names no step that can be rerun from."""
    key = (name or "").strip().lower()
    return key if key in STEPS else _ALIASES.get(key)


def choices() -> str:
    return ", ".join(STEPS)


def needs_approval(step: str) -> bool:
    """Whether running again from ``step`` replaces the plan or the frozen protocol (so it needs a named approval)."""
    return step in NEEDS_APPROVAL


def group_of(step: str) -> str:
    return next((group for group, members in GROUPS.items() if step in members), "")


def outputs_of(step: str) -> list[str]:
    """What running again from ``step`` moves to ``.fi/previous/<time>/`` (patterns spelled out as written)."""
    return list(dict.fromkeys(OUTPUTS[step]))


def step_info(step: str, *, reached: bool) -> dict[str, Any]:
    """One step as a map or a list shows it: its name, block, plain sentence, whether it needs approval, whether the
    quest reached it, and what would be moved aside."""
    return {"name": step, "group": group_of(step), "sentence": REDOES[step], "needs_approval": needs_approval(step),
            "reached": reached, "outputs": outputs_of(step)}


# The graph's nodes in the order the quest runs them, each with its block on the map and one plain sentence. Every node
# is shown; only the ones that resolve to a step (see ``resolve``) and were reached can be restarted from.
NODES: dict[str, tuple[str, str]] = {
    "clarify": ("Ideas & literature", "Checks the setup and asks what is unclear."),
    "ideate": ("Ideas & literature", "Chooses the research angle to take."),
    "literature": ("Ideas & literature", "Searches the literature."),
    "pause_after_literature": ("Skills & plan", "Waits for you to add papers the search could not get in full."),
    "select_skills": ("Skills & plan", "Chooses which tools (skills) the experiment may use."),
    "plan": ("Skills & plan", "Writes the plan and waits for you to read it."),
    "design": ("Design", "Designs the experiment and freezes the protocol it will be judged by."),
    "implement_outline": ("Code", "Sketches the experiment's code."),
    "implement": ("Code", "Writes the experiment's code."),
    "execute": ("Run", "Runs the experiment."),
    "execute_reflect": ("Run", "Repairs a script that crashed, then runs it again."),
    "auto_collect_data": ("Run", "Looks for data (a quest with no simulation)."),
    "wait_for_data": ("Run", "Waits for you to add data (a quest with no simulation)."),
    "data_load": ("Run", "Loads the data (a quest with no simulation)."),
    "web_plots": ("Run", "Plans the plots from the loaded data (a quest with no simulation)."),
    "web_figures": ("Run", "Draws the figures from the loaded data (a quest with no simulation)."),
    "analyze": ("Analysis", "Analyses the results."),
    "cross_check": ("Analysis", "Checks the analysis against the literature and the design."),
    "evidence_gate": ("Analysis", "Weighs how strong the evidence for the results is."),
    "write": ("Writing", "Writes the paper."),
    "claim_check": ("Writing", "Checks every claim in the paper against the results."),
    "review": ("Review", "Reviews the paper."),
    "human_feedback": ("Review", "Waits for your decision on the paper."),
}


def node_map(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every graph node as a map shows it, in graph order: name, block, sentence, the step a restart from it means (or
    ``""``), and whether it can be clicked. ``steps`` is what ``Engine.rerun_steps()`` returns. A node is clickable only
    when it names a step that has its backup list and sentence and the quest reached that step."""
    by_step = {s["name"]: s for s in steps}
    out = []
    for node, (block, sentence) in NODES.items():
        step = resolve(node) or ""
        info = by_step.get(step) if step in OUTPUTS and step in REDOES else None
        out.append({"node": node, "block": block, "sentence": sentence, "step": step if info else "",
                    "clickable": bool(info and info["reached"]),
                    "needs_approval": bool(info and info["needs_approval"]),
                    "redoes": info["sentence"] if info else "",
                    "outputs": info["outputs"] if info else []})
    return out


async def _current_branch(graph: Any, run_config: dict[str, Any]):
    """The checkpoints of the quest's present line, newest first. After a run from an earlier step the history also
    holds the line it left, whose later steps the quest no longer has; only the newest checkpoint and its parents (each
    one named by ``parent_config``) are the quest as it is."""
    expected: str | None = None
    first = True
    async for snapshot in graph.aget_state_history(run_config):
        cid = str(((snapshot.config or {}).get("configurable") or {}).get("checkpoint_id") or "")
        if not first and cid != expected:
            continue
        first = False
        parent = getattr(snapshot, "parent_config", None)
        expected = str(((parent or {}).get("configurable") or {}).get("checkpoint_id") or "") or None
        yield snapshot
        if expected is None:
            return


async def checkpoint_before(graph: Any, run_config: dict[str, Any], step: str) -> dict[str, Any] | None:
    """The config of the checkpoint taken just before ``step`` last began, or ``None`` when the quest never reached it.

    The quest's present line is walked newest first. The newest checkpoint about to run one of the step's nodes is found, then the walk
    goes on to older ones for as long as each is still about to run one of them: a step of several nodes (the outline
    then the script; the data collected, waited for, loaded) is done again from its first node, and never from a node
    of an earlier pass."""
    starts = set(STEPS[step])
    inside = starts | set(_WITHIN.get(step, ()))
    best = None
    async for snapshot in _current_branch(graph, run_config):
        about_to = set(snapshot.next or ())
        if about_to & starts:
            best = snapshot
        elif best is not None and not about_to & inside:
            break
    return best.config if best is not None else None


async def reached(graph: Any, run_config: dict[str, Any]) -> list[str]:
    """The steps the quest reached, in their order: the ones ``--from`` can run it again from (a step it never reached
    has no checkpoint before it)."""
    seen: set[str] = set()
    async for snapshot in _current_branch(graph, run_config):
        seen.update(snapshot.next or ())
    return [step for step, nodes in STEPS.items() if seen & set(nodes)]


def picks_skills(step: str) -> bool:
    """Whether running again from ``step`` picks the quest's skills again."""
    return step == "skills"


def listing(quest_id: str, steps: list[str] | list[dict[str, Any]]) -> str:
    """The reached steps as a person reads them, with how to use one. ``steps`` is the names ``reached`` returns, or the
    dicts ``Engine.rerun_steps()`` returns (only the ones the quest reached are shown)."""
    names = [s if isinstance(s, str) else s["name"] for s in steps if isinstance(s, str) or s.get("reached", True)]
    if not names:
        return (f"Quest {quest_id} has not reached a step it can be run again from yet. "
                f"`--resume {quest_id}` goes on from where it stopped; to change the plan, use --revise-plan.")
    width = max(len(step) for step in names)
    lines = [f"Quest {quest_id} can be run again from:"]
    lines += [f"  {step.ljust(width)}  {REDOES[step]}" for step in names]
    lines.append(f"Run: --resume {quest_id} --from <step>. What that step and the later ones made is first moved to "
                 ".fi/previous/<time>/.")
    if any(needs_approval(step) for step in names):
        lines.append("ideas, literature, plan and design replace the experiment plan (frozen), so they also need "
                     "--approve-as <you>.")
    return "\n".join(lines)


def _matches(quest_root: Path, rel: str) -> list[str]:
    """The paths (relative, with ``/``) that ``rel`` names: itself, or every match when it has a ``*``."""
    if "*" not in rel:
        return [rel] if (quest_root / rel).exists() else []
    return sorted(p.relative_to(quest_root).as_posix() for p in quest_root.glob(rel))


def _move(src: Path, target: Path) -> None:
    """Move ``src`` to ``target``. A folder that is already there (a file inside it was moved first, because a step lists
    both) is merged into, never nested inside itself."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and target.is_dir():
        for child in list(src.iterdir()):
            _move(child, target / child.name)
        src.rmdir()
        return
    shutil.move(str(src), str(target))


def _restore(quest_root: Path, dest: Path, moved: list[str]) -> None:
    """Put back what a move that stopped half way had already taken, so a failed backup leaves the files where they were."""
    for rel in reversed(moved):
        held = dest / rel
        if held.exists() and not (quest_root / rel).exists():
            try:
                _move(held, quest_root / rel)
            except OSError:
                pass


def back_up(quest_root: Path, step: str) -> tuple[Path | None, list[str]]:
    """Move what ``step`` and the steps after it wrote into ``.fi/previous/<time>/``. Returns the folder and what was
    moved (relative paths); ``(None, [])`` when there was nothing to move."""
    quest_root = Path(quest_root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = quest_root / ".fi" / "previous" / stamp
    n = 2
    while dest.exists():
        dest = quest_root / ".fi" / "previous" / f"{stamp}-{n}"
        n += 1
    moved: list[str] = []
    for pattern in dict.fromkeys(OUTPUTS[step]):
        for rel in _matches(quest_root, pattern):
            src = quest_root / rel
            if not src.exists() or rel in moved:
                continue
            target = dest / rel
            try:
                _move(src, target)
            except OSError:
                _restore(quest_root, dest, moved)
                raise
            moved.append(rel)
    # The folders every step expects to find are put back empty.
    for folder in ("figures", "code", "paper"):
        (quest_root / folder).mkdir(parents=True, exist_ok=True)
    return (dest if moved else None), moved
