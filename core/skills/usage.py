"""Record that a quest used a skill, so the library learns from its own use.

This is the return leg of the compounding loop. Selection ranks partly on a
skill's track record, and distillation needs to know which quests a skill took
part in — neither is possible unless use is written down.

Two constraints shape it:

**Writing must not lapse approval.** ``provenance.json`` is excluded from a
skill's content hash (see ``base.py``), precisely so recording use does not
look like the skill changing. That exclusion exists for this module.

**``--fleet`` runs quests in parallel.** Several may finish with the same skill
at the same moment, so every write takes a lock and re-reads before appending.
A last-writer-wins update would silently drop records, and a dropped record is
invisible: the count is simply lower than the truth, with nothing to notice.

Nothing here is allowed to fail a quest. A quest that produced an accepted
paper has already succeeded; losing its bookkeeping is a smaller harm than
turning that success into an error.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.skills.base import PROVENANCE_JSON

_log = logging.getLogger("fi.skills.usage")

#: A quest that finished should not wait long on bookkeeping. If a lock is
#: held longer than this something is wrong, and skipping the record beats
#: stalling the caller.
LOCK_TIMEOUT_S = 10


@dataclass(frozen=True)
class UsageRecord:
    quest: str
    outcome: str
    at: str

    def to_dict(self) -> dict[str, str]:
        return {"quest": self.quest, "outcome": self.outcome, "at": self.at}


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def record_use(
    skill_path: Path,
    quest_id: str,
    outcome: str,
    *,
    timeout_s: int = LOCK_TIMEOUT_S,
) -> bool:
    """Append one usage record under a lock. True if it was written.

    Idempotent per (quest, skill): re-running or resuming a quest updates the
    existing entry rather than adding a second, so a resumed quest cannot
    inflate a skill's track record.
    """
    provenance = skill_path / PROVENANCE_JSON
    entry = UsageRecord(
        quest=str(quest_id),
        outcome=str(outcome or "unknown"),
        at=time.strftime("%Y-%m-%d"),
    )

    try:
        from filelock import FileLock, Timeout
    except ImportError:  # pragma: no cover - declared in pyproject deps
        _log.warning("filelock unavailable; skipping usage record")
        return False

    lock = FileLock(str(provenance) + ".lock", timeout=timeout_s)
    try:
        with lock:
            data = _read(provenance)
            history = data.get("taught_by_projects")
            if not isinstance(history, list):
                history = []
            # Replace an existing record for this quest rather than appending
            # a duplicate — otherwise `--resume` would count twice.
            history = [
                h for h in history
                if not (isinstance(h, dict) and h.get("quest") == entry.quest)
            ]
            history.append(entry.to_dict())
            data["taught_by_projects"] = history
            provenance.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(provenance, data)
        return True
    except Timeout:
        _log.warning(
            "could not lock %s within %ss; usage not recorded", provenance, timeout_s,
        )
        return False
    except OSError as e:
        _log.warning("could not record usage in %s: %s", provenance, e)
        return False


def record_quest(
    skill_names: list[str],
    quest_id: str,
    outcome: str,
    *,
    skills_dir: Path | None = None,
) -> list[str]:
    """Record a finished quest against every skill it used.

    Returns the names actually written. Never raises: a quest that produced an
    accepted paper has already succeeded, and losing its bookkeeping is a far
    smaller harm than turning that success into an error.
    """
    if not skill_names:
        return []
    try:
        from core.skills.registry import discover

        found = {s.name: s for s in discover(skills_dir)}
    except Exception as e:  # noqa: BLE001
        _log.warning("skills unavailable; usage not recorded: %s", e)
        return []

    written: list[str] = []
    for name in skill_names:
        skill = found.get(name)
        if skill is None:
            _log.warning("skill %r vanished before its usage could be recorded", name)
            continue
        if record_use(skill.path, quest_id, outcome):
            written.append(name)
    return written
