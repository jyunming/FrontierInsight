"""Read a quest's ``.fi/state.sqlite`` as text.

The checkpoint is a binary LangGraph store. When a quest runs on a machine
nothing can be copied off, the only way to see what it held is to print it, so
this reads it directly — read-only, no ``config.yaml``, no ``Engine`` — and
prints the state and the path the quest took.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

_PREVIEW_CHARS = 100


def _locate(path: Path) -> Path:
    db = path if path.suffix == ".sqlite" else path / ".fi" / "state.sqlite"
    if not db.is_file():
        raise FileNotFoundError(
            f"no checkpoint at {db}. Give the quest directory (it contains "
            f".fi/state.sqlite) or the state.sqlite file itself."
        )
    return db


def _describe(value: Any) -> str:
    if isinstance(value, str):
        one_line = " ".join(value.split())
        cut = one_line[:_PREVIEW_CHARS] + ("..." if len(one_line) > _PREVIEW_CHARS else "")
        return f"str, {len(value)} chars: {cut!r}"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}, {len(value)} items"
    if isinstance(value, dict):
        keys = ", ".join(str(k) for k in list(value)[:6])
        more = ", ..." if len(value) > 6 else ""
        return f"dict, {len(value)} keys: {keys}{more}"
    return f"{type(value).__name__}: {str(value)[:_PREVIEW_CHARS]}"


def dump_state(path: Path) -> str:
    from langgraph.checkpoint.sqlite import SqliteSaver

    db = _locate(Path(path))
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        threads = [r[0] for r in conn.execute("select distinct thread_id from checkpoints")]
        if not threads:
            return f"{db}: the checkpoint store is empty (the quest never reached its first node)."
        saver = SqliteSaver(conn)
        lines: list[str] = []
        for thread in threads:
            cfg = {"configurable": {"thread_id": thread}}
            history = list(saver.list(cfg))
            latest = history[0]
            values = latest.checkpoint.get("channel_values") or {}
            lines.append(f"quest: {thread}")
            lines.append(
                f"checkpoints: {len(history)}   latest step: "
                f"{latest.metadata.get('step')}   at {latest.checkpoint.get('ts')}"
            )
            lines.append("")
            lines.append(f"state ({len(values)} keys):")
            width = max((len(k) for k in values), default=0)
            for key in sorted(values):
                lines.append(f"  {key.ljust(width)}  {_describe(values[key])}")
            lines.append("")
            lines.append("path (oldest first; 'wrote' = state keys that node set):")
            for cp in reversed(history):
                channels = cp.checkpoint.get("updated_channels") or []
                nxt = [c.split(":", 2)[-1] for c in channels if c.startswith("branch:to:")]
                wrote = [c for c in channels if not c.startswith(("branch:", "__"))]
                stamp = str(cp.checkpoint.get("ts") or "")[11:19]
                lines.append(
                    f"  step {str(cp.metadata.get('step')).rjust(2)}  {stamp}  "
                    f"wrote: {', '.join(wrote) or '-'}  ->  "
                    f"{', '.join(nxt) or '(end)'}"
                )
        return "\n".join(lines)
    finally:
        conn.close()
