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

# Plain name -> the graph nodes that begin that step, in the order to look for them. The node names are accepted too.
STEPS: dict[str, tuple[str, ...]] = {
    "code": ("implement_outline", "implement"),
    "run": ("execute", "data_load"),
    "analysis": ("analyze",),
    "writing": ("write",),
    "review": ("review",),
}
_ALIASES = {
    "implement": "code", "implement_outline": "code", "execute": "run", "data_load": "run", "analyze": "analysis",
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
    "run": ["figures", "raw", "results.json", *_PAPER],
    "code": ["code", "figures", "raw", "results.json", *_PAPER],
}


def resolve(name: str) -> str | None:
    """The plain step name ``name`` stands for, or ``None`` when it names no step that can be rerun from."""
    key = (name or "").strip().lower()
    return key if key in STEPS else _ALIASES.get(key)


def choices() -> str:
    return ", ".join(STEPS)


async def checkpoint_before(graph: Any, run_config: dict[str, Any], step: str) -> dict[str, Any] | None:
    """The config of the latest checkpoint taken just before ``step`` ran, or ``None`` when the quest never reached it.

    The history is newest first; the first node of a step is looked for before the next one ("code" is the outline
    when there is one, else the script), so the whole step is done again."""
    history = [s async for s in graph.aget_state_history(run_config)]
    for node in STEPS[step]:
        for snapshot in history:
            if node in (snapshot.next or ()):
                return snapshot.config
    return None


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
