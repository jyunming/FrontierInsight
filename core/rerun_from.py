"""Rerun a quest from a chosen step (``--resume <quest_id> --from <step>``).

The quest's checkpoint history (LangGraph's ``state.sqlite``) holds the state before every step it ran. Rerunning from
a step continues from the latest checkpoint taken just before that step ran, so everything the quest decided up to
there is kept and everything from there on is done again. The outputs of that step and the ones after it are first
moved to ``.fi/previous/<time>/``, so the new ones never mix with the old and both can be compared.

The steps offered start at the code: an earlier change is a change to the plan, which the frozen protocol holds, and
goes through ``--revise-plan``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

# Plain name -> the graph nodes the step is made of (run in that order). The node names are accepted too.
STEPS: dict[str, tuple[str, ...]] = {
    "code": ("implement_outline", "implement"),
    # The no-simulation path's run is the data: collected, waited for, loaded.
    "run": ("execute", "auto_collect_data", "wait_for_data", "data_load"),
    "analysis": ("analyze",),
    "writing": ("write",),
    "review": ("review",),
}
# Nodes a step loops through without leaving it: the run's repair of a crashed script goes back to execute.
_WITHIN: dict[str, tuple[str, ...]] = {"run": ("execute_reflect",)}
# What running the quest again from each step does, in one sentence a person reads before choosing.
REDOES: dict[str, str] = {
    "code": "writes the experiment's code again, then runs it and does everything after",
    "run": "runs the same code again for new results, then analyses, writes and reviews them",
    "analysis": "analyses the same results again, then writes and reviews",
    "writing": "writes the paper again from the same analysis, then reviews it",
    "review": "reviews the same paper again (and makes the slides and poster from it again)",
}
_ALIASES = {
    "implement": "code", "implement_outline": "code", "execute": "run", "data_load": "run", "auto_collect_data": "run", "analyze": "analysis",
    "write": "writing", "paper": "writing",
}

# What each step and every step after it write, relative to the quest folder. A step's own inputs are not here: a
# rerun from the writing keeps the figures the run drew.
_DELIVERED = ["slides.md", "slides.html", "slides.pdf", "slides.pptx", "poster.tex", "poster.pdf", "poster.pptx",
              "poster.html", "speech.md", "frontier_insight_summary.json", "NEXT_STEP.md"]
_PAPER = ["paper", "paper.md", "paper.pdf", "paper.html", "paper_html_body.md", "paper_html_source.html",
          "paper_html_theme.css", "paper_pdf_source.md", *_DELIVERED]
OUTPUTS: dict[str, list[str]] = {
    # The review writes no file of its own; what is made from the reviewed paper is made again after it.
    "review": _DELIVERED,
    "writing": _PAPER,
    "analysis": _PAPER,
    # The mean-over-seeds redraw is written into code/ by the run.
    "run": ["figures", "raw", "results.json", "code/replot_figures.py", "code/replot_figures.json", *_PAPER],
    "code": ["code", "figures", "raw", "results.json", *_PAPER],
}


def resolve(name: str) -> str | None:
    """The plain step name ``name`` stands for, or ``None`` when it names no step that can be rerun from."""
    key = (name or "").strip().lower()
    return key if key in STEPS else _ALIASES.get(key)


def choices() -> str:
    return ", ".join(STEPS)


async def checkpoint_before(graph: Any, run_config: dict[str, Any], step: str) -> dict[str, Any] | None:
    """The config of the checkpoint taken just before ``step`` last began, or ``None`` when the quest never reached it.

    The history is newest first. The newest checkpoint about to run one of the step's nodes is found, then the walk
    goes on to older ones for as long as each is still about to run one of them: a step of several nodes (the outline
    then the script; the data collected, waited for, loaded) is done again from its first node, and never from a node
    of an earlier pass."""
    starts = set(STEPS[step])
    inside = starts | set(_WITHIN.get(step, ()))
    best = None
    async for snapshot in graph.aget_state_history(run_config):
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
    async for snapshot in graph.aget_state_history(run_config):
        seen.update(snapshot.next or ())
    return [step for step, nodes in STEPS.items() if seen & set(nodes)]


def listing(quest_id: str, steps: list[str]) -> str:
    """The reached steps as a person reads them, with how to use one."""
    if not steps:
        return (f"Quest {quest_id} has not reached the code step yet, so there is no step to run it again from. "
                f"`--resume {quest_id}` goes on from where it stopped; to change the plan, use --revise-plan.")
    width = max(len(step) for step in steps)
    lines = [f"Quest {quest_id} can be run again from:"]
    lines += [f"  {step.ljust(width)}  {REDOES[step]}" for step in steps]
    lines.append(f"Run: --resume {quest_id} --from <step>. What that step and the later ones made is first moved to "
                 ".fi/previous/<time>/. To change the plan, use --revise-plan.")
    return "\n".join(lines)


def back_up(quest_root: Path, step: str) -> tuple[Path | None, list[str]]:
    """Move what ``step`` and the steps after it wrote into ``.fi/previous/<time>/``. Returns the folder and what was
    moved (relative paths); ``(None, [])`` when there was nothing to move."""
    quest_root = Path(quest_root)
    dest = quest_root / ".fi" / "previous" / time.strftime("%Y%m%d-%H%M%S")
    moved: list[str] = []
    for rel in dict.fromkeys(OUTPUTS[step]):
        src = quest_root / rel
        if not src.exists():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))
        moved.append(rel)
    # The folders every step expects to find are put back empty.
    for folder in ("figures", "code", "paper"):
        (quest_root / folder).mkdir(parents=True, exist_ok=True)
    return (dest if moved else None), moved
