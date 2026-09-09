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

import hashlib
import json
import logging
import os
import tempfile
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


def _lock_path(provenance: Path) -> Path:
    """Where the lock for one ``provenance.json`` lives — outside the skill.

    The obvious place is ``provenance.json.lock`` beside the file, and that
    is what this used to do. It cost the invariant this module exists to
    keep. ``filelock`` deletes the lock file on release on Windows but
    leaves it behind on POSIX, so on Linux the first recorded use dropped a
    new file into the skill directory, the content hash covered it, and the
    skill's approval lapsed — a quest un-trusting the skill it had just
    used successfully, on one platform only.

    Excluding ``*.lock`` from the hash would have fixed the symptom and
    re-opened the hole the hash was widened to close: anything a skill
    ships under a name the hash skips is unsigned content. So the lock goes
    somewhere the hash never looks instead. The name is derived from the
    resolved path, so two processes reaching the same skill by different
    routes still serialise on the same lock.
    """
    key = hashlib.sha256(
        str(provenance.resolve()).casefold().encode("utf-8")
    ).hexdigest()[:32]
    return Path(tempfile.gettempdir()) / "fi-skill-locks" / f"{key}.lock"


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


#: Windows can refuse a rename while a handle on the destination is still
#: closing — a scanner's, or the previous writer's. Under ``--fleet`` those
#: renames land back to back, so a few short retries turn a lost record into
#: a slightly later one.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = 0.05


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write through a sibling file, then rename it over the target.

    The sibling is named per process. A fixed ``.tmp`` is every writer's
    file and not just this one's, and a crashed writer's leftover is worse
    than untidy: anything left behind inside the skill counts towards its
    content hash, so a stray temp file lapses the skill's approval exactly
    the way a stray lock file did. Hence the ``finally``.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                tmp.replace(path)
                return
            except OSError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_BACKOFF_S)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - only if the directory vanished
            pass


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

    lock_file = _lock_path(provenance)
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log.warning("could not create the lock directory %s: %s", lock_file.parent, e)
        return False
    lock = FileLock(str(lock_file), timeout=timeout_s)
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
