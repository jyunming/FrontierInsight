"""The engine's source as one text, for the tests that look for a line of its code.

The engine is ``core/engine.py`` (the class, its run loop and graph), ``core/engine_base.py`` (what the stages share) and
one ``core/engine_<stage>.py`` per stage."""
from __future__ import annotations

from pathlib import Path

_CORE = Path(__file__).resolve().parent.parent / "core"
STAGE_FILES = ("engine_ideas.py", "engine_plan.py", "engine_data.py", "engine_code.py", "engine_improve.py",
               "engine_run.py", "engine_analyze.py", "engine_write.py", "engine_review.py")


def engine_text() -> str:
    """``engine_base.py``, ``engine.py`` and the stage files, in the order the code had in one file."""
    return chr(10).join((_CORE / name).read_text(encoding="utf-8") for name in ("engine_base.py", "engine.py", *STAGE_FILES))
