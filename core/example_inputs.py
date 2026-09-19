"""Example files the user supplies for the experiment.

A simulation setup, an input deck, a config, a script, a document: whatever the
user has that shows what the experiment should look like. They are copied into
one folder of the quest, the design and the code-writing steps are shown what is
there (names, sizes, and the text of the small ones), and the experiment finds
the folder in ``FI_INPUT_DIR``.

Any file type is accepted. The older ``inputs/data/`` drop zone only takes
tabular data files and only the analysis reads them, so a ``.in`` or ``.yaml``
setup never reached the code that had to use it.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

EXAMPLES_DIRNAME = "examples"
ENV_VAR = "FI_INPUT_DIR"

#: What a quest may copy in, in total. A mesh or a result set can be large; this
#: stops a path typo (a whole drive) from filling the disk.
MAX_COPY_BYTES = 500 * 1024 * 1024

#: How much of a text file the prompts show, per file and in all.
PER_FILE_CHARS = 4000
TOTAL_CHARS = 16000

_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", ".idea"})
_NUL = b"\x00"
_REPLACEMENT_CHAR = "�"


def examples_dir(quest_root: Path) -> Path:
    return quest_root / "inputs" / EXAMPLES_DIRNAME


def _walk(path: Path):
    """Files under ``path``, without the folders nobody means to supply."""
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        if set(p.relative_to(path).parts[:-1]) & _SKIP_DIRS:
            continue
        yield p


def stage_inputs(
    sources: list[str], quest_root: Path, log: logging.Logger,
) -> list[str]:
    """Copy each source file or folder into ``inputs/examples/``. Returns the
    files now there, relative to the quest. Copying again is harmless: a file
    is only rewritten when it differs.

    Raises ``FileNotFoundError`` for a path that does not exist, and
    ``ValueError`` when the total is larger than ``MAX_COPY_BYTES``, because
    an ``inputs`` entry that cannot be used should stop the quest before any
    LLM call, not be skipped and found missing at the end.
    """
    dest = examples_dir(quest_root)
    plan: list[tuple[Path, Path]] = []
    total = 0
    for raw in sources:
        src = Path(raw).expanduser()
        if not src.exists():
            raise FileNotFoundError(
                f"execution.inputs: {raw!r} does not exist. Give a file or a folder."
            )
        base = src.parent  # a folder keeps its own name under examples/
        for f in _walk(src):
            total += f.stat().st_size
            plan.append((f, dest / f.relative_to(base)))
    if total > MAX_COPY_BYTES:
        raise ValueError(
            f"execution.inputs is {total / 1e6:.0f} MB, over the "
            f"{MAX_COPY_BYTES / 1e6:.0f} MB the quest will copy. Point it at the "
            f"files the experiment needs, not the folder around them."
        )
    # A resume stages again. What decides a copy is whether the SOURCE changed
    # since it was last staged, not whether the copy differs from it: a file
    # edited by hand in the quest while it was paused must survive the resume.
    staged_path = quest_root / ".fi" / "examples_staged.json"
    try:
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        staged = {}
    copied = 0
    for f, target in plan:
        key = target.relative_to(quest_root).as_posix()
        st = f.stat()
        signature = [st.st_size, st.st_mtime_ns]
        if target.exists() and staged.get(key) == signature:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        staged[key] = signature
        copied += 1
    if copied:
        try:
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(json.dumps(staged, indent=1), encoding="utf-8")
        except OSError as exc:
            log.debug("[inputs] could not record what was staged: %s", exc)
    if plan:
        log.info(
            "[inputs] %d example file(s) from execution.inputs in %s (%d copied, %.1f MB)",
            len(plan), dest, copied, total / 1e6,
        )
    return list_inputs(quest_root)


def list_inputs(quest_root: Path) -> list[str]:
    """Every example file in the quest, relative to it, whether it was copied
    from ``execution.inputs`` or dropped there by hand during a pause."""
    dest = examples_dir(quest_root)
    if not dest.is_dir():
        return []
    return [
        p.relative_to(quest_root).as_posix()
        for p in sorted(dest.rglob("*"))
        if p.is_file() and not p.name.startswith(".")
    ]


def _read_text(path: Path, limit: int) -> tuple[str, bool] | None:
    """The start of a text file and whether it was cut, or None for a binary one."""
    try:
        with path.open("rb") as fh:
            head = fh.read(limit + 1)
    except OSError:
        return None
    if _NUL in head[:2048]:
        return None
    text = head[:limit].decode("utf-8", errors="replace")
    if text.count(_REPLACEMENT_CHAR) > max(3, len(text) // 50):
        return None
    return text, len(head) > limit


_INTRO = (
    "The user supplied these example files. They are the user's own material: "
    "simulation settings, inputs, a config, a script or a document. Use them as "
    "the starting point: combine what they show with the skills above to build a "
    "NEW simulation for this design, rather than re-running them unchanged. They "
    "are read-only and live in the folder named by the environment variable "
    f"`{ENV_VAR}` (an absolute path) — open them from there in the code, never "
    "by a path written into it."
)


def render_block(quest_root: Path) -> str:
    """The text the design and code-writing prompts carry about the example
    files. Empty when there are none."""
    files = list_inputs(quest_root)
    if not files:
        return ""
    dest = examples_dir(quest_root)
    listing: list[str] = []
    bodies: list[str] = []
    shown = 0
    for rel in files:
        path = quest_root / rel
        name = path.relative_to(dest).as_posix()
        size = path.stat().st_size
        listing.append(f"- {name} ({size:,} bytes)")
        if shown >= TOTAL_CHARS:
            continue
        read = _read_text(path, PER_FILE_CHARS)
        if read is None:
            bodies.append(f"### {name}\n(binary file, {size:,} bytes; not shown)")
            continue
        text, cut_short = read
        room = text[: TOTAL_CHARS - shown]
        shown += len(room)
        more = "\n... (truncated)" if cut_short or len(room) < len(text) else ""
        bodies.append(f"### {name}\n```\n{room}{more}\n```")
    parts = [_INTRO, "", "Files (relative to that folder):", *listing]
    if bodies:
        parts += ["", "Contents (small text files in full, larger ones cut):", *bodies]
    return "\n".join(parts)
