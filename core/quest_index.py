"""Every quest FI has run on this computer, so a quest can be found by its id from any folder.

A person runs several studies, each in its own folder, sometimes at the same time. Each quest still lives where it
always did (``<working folder>/outputs/<quest id>``, or the YAML's ``output_dir``); this file only remembers where:
``~/.frontier-insight/quests.json`` (``core/fi_home.py``), one entry per quest id::

    {"version": 1, "quests": {"1790003131-energy-drift-479b06": {
        "quest_root": "C:\\\\studies\\\\drift\\\\outputs\\\\1790003131-energy-drift-479b06",
        "config": ".../config.yaml", "working_folder": "C:\\\\studies\\\\drift",
        "title": "Energy drift in symplectic integrators",
        "created_at": "2026-09-30T10:00:00+00:00", "last_seen": "2026-09-30T11:00:00+00:00"}}}

Written when a quest starts or resumes (``launch.py:run_one``) and when it is renamed (``core/quest_title.py``). Several
FI processes may write it at once (different folders, ``--fleet``, several VS Code windows, the web page): every write
reads, changes one entry and writes the whole file again under a file lock, through a temporary file and a rename, so
no writer loses another's entries. Readers take no lock; the rename is atomic.

:func:`find` is the one lookup every place that takes a quest id uses (CLI ``--resume``/``--from``/``--watch``/
``fi tools rename``/``--why``/``--trace``, the web quest page and its API, and — through the CLI — VS Code): the
local outputs folder first, as before, then this index, then a plain error naming close matches. A quest can be named
by its full id or by any unique start or end of it — the six characters after the last dash (:func:`short_id`) are
what lists show. An entry whose folder is gone is dropped when it is looked up, or by ``fi tools quests --prune``;
resuming a moved quest from its new folder records the new place.

The VS Code extension reads this file (never writes it): ``vscode-frontier-insight/src/quest-index.ts``.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.fi_home import fi_home

INDEX_NAME = "quests.json"
VERSION = 1
#: The fewest characters a shortened id may have: fewer would match by accident.
MIN_SHORT = 4
_LOCK_TIMEOUT_S = 30.0
_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")
_NONCE_RE = re.compile(r"-([0-9a-f]{6})$")


def index_path() -> Path:
    return fi_home() / INDEX_NAME


def short_id(quest_id: str) -> str:
    """The six characters after the last dash of an id FI minted (``<time>-<topic>-<6 hex>``); the whole id otherwise."""
    m = _NONCE_RE.search(quest_id)
    return m.group(1) if m else quest_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- reading and writing the file ------------------------------------------------------------------------------------


def _read(path: Path) -> dict[str, dict[str, Any]]:
    """The entries, or ``{}`` when there is no file. Raises ``ValueError`` when the file is there but unreadable."""
    for attempt in range(10):
        try:
            text = path.read_text(encoding="utf-8")
            break
        except FileNotFoundError:
            return {}
        except PermissionError:
            # Windows: another process is replacing it this instant.
            if attempt == 9:
                raise
            time.sleep(0.05)
    data = json.loads(text) if text.strip() else {}
    quests = data.get("quests") if isinstance(data, dict) else None
    if not isinstance(quests, dict):
        raise ValueError(f"{path} is not a quest list")
    return {str(k): v for k, v in quests.items() if isinstance(v, dict) and _ID_RE.match(str(k))}


def load() -> dict[str, dict[str, Any]]:
    """Every entry, keyed by quest id (``{}`` when there is no index or it cannot be read)."""
    try:
        return _read(index_path())
    except (OSError, ValueError):
        return {}


def _write(path: Path, quests: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"version": VERSION, "quests": quests}, indent=2, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())  # on disk before it replaces the old list: a power cut leaves one whole list or the other
        for attempt in range(40):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows refuses to replace a file another process has open for reading (VS Code, the web page, a
                # list): it is open for a moment only.
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _update(change: Any) -> dict[str, dict[str, Any]]:
    """Read the index, apply ``change(quests)`` (which edits the dict in place) and write it back, all under the lock.

    A file whose content is not a quest list is kept beside the new one (``quests.json.unreadable-<time>``) rather than
    written over, so nothing in it is lost for good. A file that cannot be *read* (held open too long, a network folder
    that failed) raises ``OSError`` and nothing is written: its entries are still good."""
    from filelock import FileLock

    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # thread_local=False: the web server may take it from worker threads; a release from another thread must count.
    with FileLock(str(path.with_name(INDEX_NAME + ".lock")), timeout=_LOCK_TIMEOUT_S, thread_local=False):
        try:
            quests = _read(path)
        except ValueError:
            # Set aside first; if even that fails, raise rather than write over it.
            os.replace(path, path.with_name(f"{INDEX_NAME}.unreadable-{int(time.time())}"))
            quests = {}
        change(quests)
        _write(path, quests)
        return quests


# ---- recording quests ------------------------------------------------------------------------------------------------


def register(quest_root: Path, *, config: Path | None = None, working_folder: Path | None = None,
             title: str | None = None) -> None:
    """Record (or refresh) where the quest is: called when it starts or resumes. Keeps the first ``created_at``; a
    quest resumed from a new folder is recorded at the new place."""
    root = Path(quest_root).resolve()
    qid = root.name
    if not _ID_RE.match(qid):
        return

    def change(quests: dict[str, dict[str, Any]]) -> None:
        old = quests.get(qid) or {}
        now = _now()
        quests[qid] = {
            "quest_root": str(root),
            "config": str(Path(config).resolve()) if config else str(root / "config.yaml"),
            "working_folder": str(Path(working_folder).resolve()) if working_folder else old.get("working_folder", ""),
            "title": (title or "").strip() or old.get("title", ""),
            "created_at": old.get("created_at") or now,
            "last_seen": now,
        }

    _update(change)


def set_title(quest_root: Path, title: str) -> None:
    """The quest was renamed: change its title here too (and record it when it was not listed yet)."""
    root = Path(quest_root).resolve()
    qid = root.name

    def change(quests: dict[str, dict[str, Any]]) -> None:
        entry = quests.get(qid)
        if entry is None:
            now = _now()
            entry = {"quest_root": str(root), "config": str(root / "config.yaml"), "working_folder": "",
                     "created_at": now, "last_seen": now}
        entry["title"] = title
        entry["quest_root"] = str(root)
        quests[qid] = entry

    if _ID_RE.match(qid):
        _update(change)


def _missing(entry: dict[str, Any]) -> bool:
    """The quest's folder cannot be seen now (deleted, moved, or on a drive that is not there at the moment)."""
    root = str(entry.get("quest_root") or "")
    return not root or not (Path(root) / ".fi").is_dir()


def _gone(entry: dict[str, Any]) -> bool:
    """The quest's folder is really gone: missing while the outputs folder that held it is there. A quest on a disk or
    share that is only disconnected (or unmounted) is kept, and listed again once it is back; one whose whole study
    folder went is only left out of lists, until ``fi tools quests --prune``."""
    if not _missing(entry):
        return False
    root = str(entry.get("quest_root") or "")
    if not root:
        return True
    try:
        return Path(root).parent.is_dir()
    except OSError:
        return False


def _prunable(entry: dict[str, Any]) -> bool:
    """For ``--prune``, which a person asks for: missing, and its drive (a Windows drive letter or share) is there."""
    if not _missing(entry):
        return False
    root = str(entry.get("quest_root") or "")
    if not root:
        return True
    try:
        return Path(Path(root).anchor or "/").exists()
    except OSError:
        return False


def prune() -> list[str]:
    """Drop every entry whose quest folder no longer exists (on a drive that is there); returns the ids dropped."""
    dropped: list[str] = []

    def change(quests: dict[str, dict[str, Any]]) -> None:
        for qid in [q for q, e in quests.items() if _prunable(e)]:
            dropped.append(qid)
            del quests[qid]

    _update(change)
    return dropped


def _drop_if_gone(ids: Iterable[str]) -> None:
    """Best effort: a lookup found these gone. Re-checked under the lock (another process may have re-recorded one)."""
    wanted = set(ids)
    if not wanted:
        return

    def change(quests: dict[str, dict[str, Any]]) -> None:
        for qid in wanted:
            if qid in quests and _gone(quests[qid]):
                del quests[qid]

    try:
        _update(change)
    except Exception:  # noqa: BLE001 -- a lookup never fails because the list could not be tidied
        pass


# ---- listing ---------------------------------------------------------------------------------------------------------


def status(quest_root: Path) -> str:
    """Where the quest is, in plain words, from its files only (cheap: no checkpoint is opened)."""
    from core import quest_title

    fi = quest_root / ".fi"
    if (fi / "pause.json").is_file() or (quest_root / "NEXT_STEP.md").is_file():
        return "waiting for you"
    try:
        if quest_title.looks_running(quest_root):
            return "running"
    except Exception:  # noqa: BLE001
        pass
    if (quest_root / "quest_failed.md").is_file():
        return "failed"
    if (quest_root / "frontier_insight_summary.json").is_file():
        return "finished"
    return "stopped"


@dataclass
class Entry:
    quest_id: str
    quest_root: Path
    config: Path
    working_folder: str
    title: str
    created_at: str
    last_seen: str

    @property
    def short(self) -> str:
        return short_id(self.quest_id)

    def as_dict(self, *, with_status: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "quest_id": self.quest_id, "short_id": self.short, "title": self.title,
            "quest_root": str(self.quest_root), "config": str(self.config),
            "working_folder": self.working_folder, "created_at": self.created_at, "last_seen": self.last_seen,
        }
        if with_status:
            d["status"] = status(self.quest_root)
        return d


def _entry(qid: str, e: dict[str, Any]) -> Entry:
    root = Path(str(e.get("quest_root") or ""))
    return Entry(
        quest_id=qid, quest_root=root, config=Path(str(e.get("config") or root / "config.yaml")),
        working_folder=str(e.get("working_folder") or ""), title=str(e.get("title") or ""),
        created_at=str(e.get("created_at") or ""), last_seen=str(e.get("last_seen") or ""),
    )


def entries(*, drop_gone: bool = True) -> list[Entry]:
    """The recorded quests whose folders can be seen now, most recently seen first (the really gone ones are dropped
    from the file when ``drop_gone``; one on a disconnected drive is only left out)."""
    raw = load()
    if drop_gone:
        _drop_if_gone([q for q, e in raw.items() if _gone(e)])
    live = [_entry(q, e) for q, e in raw.items() if not _missing(e)]
    live.sort(key=lambda x: (x.last_seen or x.created_at, x.quest_id), reverse=True)
    return live


# ---- finding a quest by id -------------------------------------------------------------------------------------------


@dataclass
class Found:
    root: Path
    #: "folder" (the text was the quest's folder), "local" (under a local outputs folder) or "index".
    via: str
    entry: Entry | None = None

    @property
    def quest_id(self) -> str:
        return self.root.name

    @property
    def working_folder(self) -> Path | None:
        """The folder the quest was started from, when it still exists (the index knows it; a local quest's is the
        current folder)."""
        wf = self.entry.working_folder if self.entry else ""
        return Path(wf) if wf and Path(wf).is_dir() else None


class QuestNotFound(LookupError):
    """No quest by that id: the message is a sentence to show the person as it is."""


class AmbiguousQuest(QuestNotFound):
    """The shortened id matches more than one quest: the message lists them."""

    def __init__(self, message: str, candidates: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.candidates = candidates


def matching(text: str, ids: Iterable[str]) -> list[str]:
    """The ids ``text`` names: itself when it is one of them, else every id that starts or ends with it (at least
    :data:`MIN_SHORT` characters; letter case ignored)."""
    ids = list(ids)
    if text in ids:
        return [text]
    t = text.lower()
    if len(t) < MIN_SHORT:
        return []
    return sorted({i for i in ids if i.lower().startswith(t) or i.lower().endswith(t)})


def _local_quests(local_dirs: Iterable[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in local_dirs:
        try:
            children = list(Path(d).iterdir())
        except OSError:
            continue
        for c in children:
            if not c.name.startswith(("_", ".")) and c.name not in found and (c / ".fi").is_dir():
                found[c.name] = c
    return found


def _candidate(qid: str, root: Path, title: str = "") -> dict[str, str]:
    from core import quest_title

    if not title:
        try:
            title = quest_title.current_title(root) or ""
        except Exception:  # noqa: BLE001
            title = ""
    return {"quest_id": qid, "short_id": short_id(qid), "title": title, "quest_root": str(root)}


def _ambiguous(text: str, cands: list[dict[str, str]]) -> AmbiguousQuest:
    lines = [f"{text!r} matches more than one quest; give more of its id:"]
    for c in cands:
        lines.append(f"  {c['quest_id']}  {c['title'] or '(no title yet)'}  ({c['quest_root']})")
    return AmbiguousQuest("\n".join(lines), cands)


def find(text: str, local_dirs: Iterable[Path] = (), *, allow_folder: bool = True, tidy: bool = True) -> Found:
    """The quest ``text`` names: its folder (``allow_folder``); else ``<local dir>/<id>``; else the index's quest with
    that full id; else the one quest — under the local dirs or in the index — whose id starts or ends with ``text``
    (a local one is used when both name the same id). Raises :class:`AmbiguousQuest` when a shortened id matches more
    than one quest, and :class:`QuestNotFound` (naming close matches) when nothing does. Entries whose folder is
    really gone are dropped from the index on the way when ``tidy`` (the web server passes ``False``: it answers
    without waiting on the file lock)."""
    text = (text or "").strip().strip("\"'")
    local_dirs = [Path(d) for d in local_dirs]
    if not text:
        raise QuestNotFound("give a quest id (`fi tools quests` lists them)")
    if allow_folder:
        as_path = Path(text).expanduser()
        if (as_path / ".fi").is_dir():
            return Found(as_path.resolve(), "folder", _index_entry(as_path.resolve().name, as_path))
    if not _ID_RE.match(text):
        raise QuestNotFound(f"no quest folder at {text} (a quest id has letters, digits, - _ . only)")
    for d in local_dirs:
        if (d / text / ".fi").is_dir():
            root = (d / text).resolve()
            return Found(root, "local", _index_entry(text, root))

    raw = load()
    local = _local_quests(local_dirs)
    # Only the entries this text names have their folder looked at: one on a share that does not answer must not slow
    # down every lookup.
    named = matching(text, set(local) | set(raw))
    missing = [q for q in named if q not in local and q in raw and _missing(raw[q])]
    gone = [q for q in missing if _gone(raw[q])]
    if gone and tidy:
        _drop_if_gone(gone)
    live_index = {q: raw[q] for q in named if q in raw and q not in missing}
    if text in live_index:
        e = _entry(text, live_index[text])
        return Found(e.quest_root.resolve(), "index", e)
    # A shortened id: one quest among this folder's and every other one FI knows of, or it is ambiguous.
    hits = [q for q in named if q in local or q in live_index]
    if len(hits) == 1:
        q = hits[0]
        if q in local:
            root = local[q].resolve()
            return Found(root, "local", _index_entry(q, root))
        e = _entry(q, live_index[q])
        return Found(e.quest_root.resolve(), "index", e)
    if len(hits) > 1:
        raise _ambiguous(text, [
            _candidate(q, local[q]) if q in local
            else _candidate(q, _entry(q, live_index[q]).quest_root, live_index[q].get("title") or "")
            for q in hits
        ])

    where = ", ".join(str(d) for d in local_dirs) or "this folder"
    msg = f"no quest {text!r} in {where} or among the quests FI has run on this computer"
    if gone:
        msg += f" (its folder {raw[gone[0]].get('quest_root')} is gone; resume it from where it is now to record it again)"
    elif missing:
        msg += (f" that can be reached now (its folder {raw[missing[0]].get('quest_root')} cannot be seen: a disk or "
                "network folder that is not connected?)")
    if len(text) < MIN_SHORT and not missing:
        msg += f"; give at least {MIN_SHORT} characters of its id"
    pool = {q: str(e.get("quest_root") or "") for q, e in raw.items() if q not in missing}
    pool.update({q: str(p) for q, p in local.items()})
    close = difflib.get_close_matches(text, list(pool), n=3, cutoff=0.5)
    close += [q for q in pool if text.lower() in q.lower() and q not in close][: max(0, 3 - len(close))]
    if close:
        msg += ". Close matches: " + "; ".join(f"{q} ({pool[q]})" for q in close)
    msg += ". `fi tools quests` lists every quest FI knows of."
    raise QuestNotFound(msg)


def _index_entry(qid: str, root: Path) -> Entry | None:
    e = load().get(qid)
    if e is None:
        return None
    entry = _entry(qid, e)
    try:
        same = entry.quest_root.resolve() == Path(root).resolve()
    except OSError:
        same = False
    return entry if same else None


def listing(items: list[Entry]) -> str:
    """``fi tools quests``: one line per quest (short id, where it is, its title) and its folder under it."""
    if not items:
        return ("FI has no quests recorded on this computer yet. A quest is recorded when it starts or resumes, "
                "wherever it runs.")
    lines = [f"Quests FI has run on this computer ({len(items)}, most recent first):", ""]
    for e in items:
        lines.append(f"  {e.short:<8} {status(e.quest_root):<16} {e.title or '(no title yet)'}")
        lines.append(f"  {'':<8} {e.quest_root}")
    lines += ["", "Go on with one from any folder: fi --resume <its short id>   (the full id works too)"]
    return "\n".join(lines)


__all__ = [
    "AmbiguousQuest", "Entry", "Found", "INDEX_NAME", "MIN_SHORT", "QuestNotFound", "entries", "find", "index_path",
    "listing", "load", "matching", "prune", "register", "set_title", "short_id", "status",
]
