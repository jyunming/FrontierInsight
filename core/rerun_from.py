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
* ``improve``: the changes to the simulation against its checks of correctness, inside ``run`` (``engine.improve_rounds``).
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
_WITHIN: dict[str, tuple[str, ...]] = {"run": ("execute_reflect", "improve")}

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
_EVIDENCE = ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json", "needs/STUCK.json"]
# What the run and its checks leave under needs/ (the run writes them again).
_RUN_RECORDS = ["needs/RUN_MANIFEST_CHECK.json", "needs/ORACLE_CHECK.json", "needs/PROTOCOL_CHECK.json",
                "needs/ENVIRONMENT.json", "needs/HIDDEN_CHECK.json"]
# The tests of the model's equations follow the code.
_FUNCTION_STEPS = [".fi/equation_tests.json"]
# The frozen protocol: what a redesign replaces. The amendments and the saved versions stay, so the new protocol is
# frozen as a change made after results were seen and the paper says so.
_PROTOCOL = ["needs/FROZEN_PROTOCOL.json", "needs/PROTOCOL_AMENDMENT_PENDING.json", "needs/AMENDMENT_APPROVAL.json"]
# From the design on: the protocol, the design's audit, and everything the run and the writing made from it.
# A search for the best design (core/optimise.py): its best design and FI's record of the search go with the run.
_SEARCH = ["results/best_design.json", ".fi/optimisation", "needs/OPTIMUM_CHECK.json"]
_FROM_DESIGN = [*_PROTOCOL, "needs/DESIGN_CRITIQUE.json", "needs/receipts", *_RUN_RECORDS, "needs/EVIDENCE.json",
                # The test run of the checks before the study is read once per protocol: a new one reads it again.
                ".fi/oracle_dry_run.json",
                "code", *_FUNCTION_STEPS, "figures", "raw", "results.json", *_SEARCH, "data/auto_collected", *_PAPER]
# The once-per-plan look at the plan's checks (core/oracle_forms.py): a new plan is looked at again.
_PLAN_LOOK = [".fi/oracle_guidance.json", ".fi/oracle_review.json", ".fi/oracle_corrections.json"]

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
    "run": ["figures", "raw", "results.json", *_SEARCH, "code/replot_figures.py", "code/replot_figures.json", "code/web_plots.py",
            "data/auto_collected", "needs/RUN_MANIFEST_CHECK.json", "needs/ENVIRONMENT.json",
            # The improve loop's record and saved copies belong to the run they started from.
            ".fi/improve.json", ".fi/improve", *_PAPER],
    "code": ["code", *_FUNCTION_STEPS, "figures", "raw", "results.json", *_SEARCH, *_RUN_RECORDS, *_PAPER],
    "skills": ["code", *_FUNCTION_STEPS, "figures", "raw", "results.json", *_SEARCH, *_RUN_RECORDS, *_PAPER],
    # A redesign starts the protocol over: the old one is moved aside, and the run's records with it.
    "design": _FROM_DESIGN,
    # The record of the oracles the engine added to plan.md goes with plan.md: a design rerun keeps both.
    "plan": ["plan.md", ".fi/paused_at_plan.flag", ".fi/oracles_added.json", *_PLAN_LOOK, "needs/receipts/design_audit.json",
             *_FROM_DESIGN],
    "literature": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", ".fi/oracles_added.json", *_PLAN_LOOK,
                   *_FROM_DESIGN],
    "ideas": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", ".fi/oracles_added.json", *_PLAN_LOOK, *_FROM_DESIGN],
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


def in_quest(path: Any, quest_root: Path) -> str:
    """``path`` as it is in ``quest_root``: a quest folder that was copied or moved keeps the absolute paths of its old
    folder in its checkpoints (the paper's path, written by the writing step), so a rerun from a later step would read
    and judge the OLD folder's paper. A path inside a folder named like this quest, but not this one, is taken to the
    same place in this one; any other path is returned as it is."""
    from pathlib import PurePosixPath, PureWindowsPath

    text = str(path or "")
    if not text:
        return text
    root = Path(quest_root)
    try:
        Path(text).relative_to(root)
        return text
    except ValueError:
        pass
    # A path written on Windows is read as one on any system (a quest copied from Windows to Linux).
    windows = "\\" in text or (len(text) > 1 and text[1] == ":")
    parts = (PureWindowsPath(text) if windows else PurePosixPath(text)).parts
    at = [i for i, part in enumerate(parts) if part == root.name]
    if not at:
        return text
    # The quest's name can appear twice (a folder inside it named the same): the place that exists in this quest wins,
    # else the last one.
    candidates = [root.joinpath(*parts[i + 1:]) for i in reversed(at)]
    return str(next((c for c in candidates if c.exists()), candidates[0]))


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


# The blocks of the quest map, in the order the quest runs them: key -> (plain name, one line, on the real-data path
# instead of the simulation path). The key of a block is what a node names in ``NODES``.
MAP_BLOCKS: dict[str, tuple[str, str, bool]] = {
    "setup": ("Understand the question", "Check the setup and pick an angle.", False),
    "lit": ("Read the literature", "Search papers and pause for any you must download.", False),
    "plan": ("Plan the work", "Choose tools, write the plan, design the experiment.", False),
    "run": ("Build and run the experiment", "Write the code, run it, repair it if it crashes.", False),
    "data": ("Or: use real data", "For questions with no simulation: collect and load data.", True),
    "judge": ("Judge the result", "Analyze, check against papers, decide if it is strong enough.", False),
    "write": ("Write and review", "Write the paper, check every claim, review it.", False),
}

# The graph's nodes in the order the quest runs them. Each has its block on the map (a key of ``MAP_BLOCKS``), a plain
# title, one sentence of what it does, what it reads and writes, and, when the graph can send the quest back from it, a
# note about that loop. Every node is shown; only the ones that resolve to a step (see ``resolve``) and were reached can
# be restarted from. The node's own name is shown small beside its title, so a person can match it to the log.
NODES: dict[str, dict[str, str]] = {
    "clarify": {"block": "setup", "title": "Check the setup",
                "sentence": "Reads your topic and settings, asks about anything unclear.",
                "reads": "topic, config.yaml", "writes": "clarified topic"},
    "ideate": {"block": "setup", "title": "Pick an angle",
               "sentence": "Chooses the research angle, using what earlier quests learned.",
               "reads": "topic, earlier quests", "writes": "chosen angle"},
    "literature": {"block": "lit", "title": "Search papers",
                   "sentence": "Finds papers. Only free ones are downloaded automatically.",
                   "reads": "angle", "writes": "literature notes, papers needed by hand"},
    "pause_after_literature": {"block": "lit", "title": "Wait for papers",
                               "sentence": "Stops so you can add paywalled papers. Skipped when none are needed.",
                               "reads": "needs/WANTED_PAPERS.md", "writes": "inputs/papers/"},
    "select_skills": {"block": "plan", "title": "Choose tools",
                      "sentence": "Picks which tools (skills) the code may use.",
                      "reads": "literature, approved tools", "writes": "chosen tools"},
    "plan": {"block": "plan", "title": "Write the plan",
             "sentence": "Writes plan.md. This is what you read and approve before compute is spent.",
             "reads": "angle, literature", "writes": "plan.md"},
    "design": {"block": "plan", "title": "Design the experiment",
               "sentence": "Fixes the method, the settings and the pass/fail checks before any run.",
               "reads": "plan", "writes": "frozen protocol"},
    "implement_outline": {"block": "run", "title": "Outline the code",
                          "sentence": "Sketches the structure of the experiment's script.",
                          "reads": "design", "writes": "outline"},
    "implement": {"block": "run", "title": "Write the code",
                  "sentence": "Writes the experiment's script.",
                  "reads": "outline, chosen tools", "writes": "code/"},
    "execute": {"block": "run", "title": "Run it",
                "sentence": "Runs the script in the quest's own environment.",
                "reads": "code/", "writes": "results, figures/"},
    "execute_reflect": {"block": "run", "title": "Repair a crash",
                        "sentence": "If the run crashed, reads the error and fixes the script, then runs it again.",
                        "reads": "error log", "writes": "fixed code",
                        "loop": "crash: back to Run it (its own retry budget)"},
    "improve": {"block": "run", "title": "Improve the code",
                "sentence": "If a check of correctness is not met, changes the simulation one step at a time and keeps "
                            "a change only when no check got worse.",
                "reads": "checks of correctness, code/", "writes": "code/CHANGELOG.md, criteria history",
                "loop": "a change kept: back to Run it once, in full (engine.improve_rounds)"},
    "auto_collect_data": {"block": "data", "title": "Find data",
                          "sentence": "Looks for public datasets.",
                          "reads": "plan", "writes": "data/auto_collected/"},
    "wait_for_data": {"block": "data", "title": "Wait for your data",
                      "sentence": "Stops so you can add data by hand.",
                      "reads": "needs/", "writes": "inputs/"},
    "data_load": {"block": "data", "title": "Load the data",
                  "sentence": "Reads the files into the analysis.",
                  "reads": "data files", "writes": "tables"},
    "web_plots": {"block": "data", "title": "Plan the plots",
                  "sentence": "Plans the plots from the loaded data.",
                  "reads": "tables", "writes": "code/web_plots.py"},
    "web_figures": {"block": "data", "title": "Draw the figures",
                    "sentence": "Draws the figures from the loaded data.",
                    "reads": "tables", "writes": "figures/"},
    "analyze": {"block": "judge", "title": "Analyze",
                "sentence": "Turns the results into findings and says what to do next.",
                "reads": "results", "writes": "findings",
                "loop": "weak result: back to Design (shares the redo budget with the review)"},
    "cross_check": {"block": "judge", "title": "Compare with papers",
                    "sentence": "Checks the findings against the literature and the design.",
                    "reads": "findings, literature", "writes": "cross-check"},
    "evidence_gate": {"block": "judge", "title": "Evidence check",
                      "sentence": "Weighs how strong the evidence for each result is.",
                      "reads": "findings", "writes": "evidence level"},
    "stuck_no_findings": {"block": "judge", "title": "Stop without a paper",
                          "sentence": "Ends the quest with no paper when the experiment left no findings to write up.",
                          "reads": "findings", "writes": "the reason it stopped"},
    "write": {"block": "write", "title": "Write the paper",
              "sentence": "Drafts paper.md and the outputs made from it.",
              "reads": "findings", "writes": "paper.md"},
    "replot_layout": {"block": "write", "title": "Rearrange the figures",
                      "sentence": "Redraws the figures you asked to be arranged differently, from the saved numbers.",
                      "reads": "figures, saved results", "writes": "figures/"},
    "claim_check": {"block": "write", "title": "Check every claim",
                    "sentence": "Ties each sentence of the paper to a result.",
                    "reads": "paper.md", "writes": "claim ledger"},
    "review": {"block": "write", "title": "Review",
               "sentence": "Reviews the paper. May send it back to Design.",
               "reads": "paper.md", "writes": "review",
               "loop": "revise: back to Design"},
    "human_feedback": {"block": "write", "title": "Your feedback",
                       "sentence": "Stops for your decision on the paper when the review pauses are on.",
                       "reads": "review", "writes": "your notes"},
}
# A node that shows a setting a person can change in config.yaml before the restart.
NODE_HINTS: dict[str, str] = {
    "select_skills": "To keep a tool out, add it to engine.skills_exclude in this quest's config.yaml, then restart "
                     "from the skills step.",
}


def map_blocks(*, no_simulation: bool = False) -> list[dict[str, Any]]:
    """The blocks a map shows, in order: id, plain name, one line, and whether the quest does not take that path (the
    real-data blocks on a simulation quest, the simulation ones when the quest has no simulation)."""
    return [{"id": key, "name": name, "desc": desc,
             "off": (key == "run") if no_simulation else off}
            for key, (name, desc, off) in MAP_BLOCKS.items()]


def node_map(steps: list[dict[str, Any]], *, finished: bool = False, no_simulation: bool = False,
             at: list[str] | tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Every graph node as a map shows it, in graph order: name, plain title, block, sentence, reads, writes, loop note,
    the step a restart from it means (or ``""``), whether it can be clicked, and its status: ``done``, ``now`` (where the
    quest stopped), ``todo`` or ``off`` (the other path). ``steps`` is what ``Engine.rerun_steps()`` returns; ``at`` and
    ``finished`` are what ``Engine.stopped_at()`` returns. The node the quest stopped in is the first of ``at`` on the
    quest's path (the pause it waits in, the node that failed); without one it is the first node of the last step the
    quest reached. A node is clickable only when it names a step that has its backup list and sentence, the quest
    reached that step, the node is on the quest's path, and the quest got as far as the node itself (two nodes can
    share one step)."""
    by_step = {s["name"]: s for s in steps}
    off_blocks = {b["id"] for b in map_blocks(no_simulation=no_simulation) if b["off"]}
    on_path = [n for n in NODES if NODES[n]["block"] not in off_blocks]
    stop_idx = min((on_path.index(n) for n in at if n in on_path), default=-1)
    if stop_idx < 0:
        reached_steps = [s["name"] for s in steps if s["reached"]]
        stop_step = reached_steps[-1] if reached_steps else ""
        stop_idx = next((i for i, n in enumerate(on_path) if stop_step and resolve(n) == stop_step), -1)
    status: dict[str, str] = {}
    for i, node in enumerate(on_path):
        if finished and stop_idx >= 0:
            status[node] = "done"
        elif stop_idx < 0:
            status[node] = "todo"
        else:
            status[node] = "done" if i < stop_idx else "now" if i == stop_idx else "todo"
    out = []
    for node, meta in NODES.items():
        step = resolve(node) or ""
        info = by_step.get(step) if step in OUTPUTS and step in REDOES else None
        off = node not in on_path
        out.append({"node": node, "title": meta["title"], "block": meta["block"], "sentence": meta["sentence"],
                    "reads": meta["reads"], "writes": meta["writes"], "loop": meta.get("loop", ""),
                    "hint": NODE_HINTS.get(node, ""), "status": "off" if off else status[node],
                    "step": step if info else "",
                    "clickable": bool(info and info["reached"] and not off and status[node] != "todo"),
                    "needs_approval": bool(info and info["needs_approval"]),
                    "redoes": info["sentence"] if info else "",
                    "outputs": info["outputs"] if info else []})
    return out


def map_payload(steps: list[dict[str, Any]], *, finished: bool = False, no_simulation: bool = False,
                at: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """What the quest map draws, in one piece: ``blocks``, ``finished`` and ``nodes``. The web page and the VS Code panel
    both get exactly this (built by ``Engine.quest_map()``)."""
    return {"blocks": map_blocks(no_simulation=no_simulation), "finished": finished,
            "nodes": node_map(steps, finished=finished, no_simulation=no_simulation, at=at)}


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
    history = dest / "code" / ".git"
    if history.is_dir() and not (quest_root / "code" / ".git").exists():
        try:  # code/'s change history carries on through a re-run of the code step
            shutil.copytree(history, quest_root / "code" / ".git")
        except OSError:
            pass
    return (dest if moved else None), moved
