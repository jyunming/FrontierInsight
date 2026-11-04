"""FI's own record of whether a quest's process is running, and how an earlier one ended.

A quest process can be stopped from outside Python (a kill, a restart, the system's low-memory stop): it then leaves no
traceback and no log line. Without a record of its own, the next session cannot tell that from a clean end. The marker
is that record, ``.fi/process.json``: the process id and start time, a sign of life every :data:`BEAT_S` seconds, and
at the end the way it ended. A session that finds an earlier record with no end and no process behind it knows the
earlier session was stopped from outside, says so, and clears what that session could not: the locks and temporary
files it left in the shared cache (``core/outside_run.py``), which no later setting would ever see finished.

Every session appends one line to ``.fi/process_history.jsonl`` (started, ended, or found stopped), so what happened to
a quest is a list a person or a later check can read.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

RECORD = ".fi/process.json"
HISTORY = ".fi/process_history.jsonl"
#: Seconds between the signs of life.
BEAT_S = 30.0


def _clock(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "unknown"


def _started_at(pid: int) -> float | None:
    """When the process ``pid`` started, or ``None`` when no such process runs (or it cannot be told)."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 -- no process, no access, no psutil: all "not known to run"
        return None


def is_running(record: dict[str, Any]) -> bool:
    """Whether the process a record names is still the one running (the same id, started at the same time)."""
    try:
        pid = int(record.get("pid"))
        began = float(record.get("started"))
    except (TypeError, ValueError):
        return False
    now = _started_at(pid)
    return now is not None and abs(now - began) < 5.0


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


class RunMarker:
    """One session's marker for the quest at ``quest_root``: :meth:`start`, then :meth:`finish`."""

    def __init__(self, quest_root: Path | str, *, log: Any = None, beat_s: float = BEAT_S) -> None:
        self.quest_root = Path(quest_root)
        self._log = log if log is not None else logging.getLogger(f"frontier_insight.{self.quest_root.name}")
        self._beat_s = beat_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._record: dict[str, Any] = {}

    @property
    def path(self) -> Path:
        return self.quest_root / RECORD

    def _history(self, line: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):
            with (self.quest_root / HISTORY).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": _clock(time.time()), **line}) + "\n")

    def start(self) -> dict[str, Any] | None:
        """Begin this session. Returns what the earlier session's record says when it was stopped from outside (its
        ``pid``, ``started``, ``seen_at``), else ``None``."""
        try:
            earlier = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            earlier = None
        found: dict[str, Any] | None = None
        if isinstance(earlier, dict) and not earlier.get("ended"):
            if is_running(earlier) and int(earlier.get("pid") or 0) != os.getpid():
                self._log.warning("[run] this quest already runs in another process (id %s, started %s); two runs "
                                  "of one quest can overwrite each other's files", earlier.get("pid"),
                                  _clock(earlier.get("started")))
            else:
                found = earlier
                self._log.warning(
                    "[run] the earlier run of this quest (started %s) ended without recording an end; its last sign of "
                    "life was at %s. It was stopped from outside (a kill, a restart, or the system running out of "
                    "memory); this run goes on from the saved state", _clock(earlier.get("started")),
                    _clock(earlier.get("seen_at")))
                self._history({"event": "found_stopped", "pid": earlier.get("pid"), "started": earlier.get("started"),
                               "seen_at": earlier.get("seen_at")})
                self._clear_leftovers()
        now = time.time()
        pid = os.getpid()
        self._record = {"pid": pid, "started": _started_at(pid) or now, "seen_at": now, "ended": None}
        _write(self.path, self._record)
        self._history({"event": "started", "pid": pid})
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="fi-run-marker", daemon=True)
        self._thread.start()
        return found

    def _clear_leftovers(self) -> None:
        """The locks and temporary files the stopped session left in the shared cache: no process is left to finish them,
        and a later setting would wait on a lock for ever. A finished value stays (core/outside_run.unfinished_shared)."""
        from . import outside_run as _outside

        left: list[str] = []
        names = _outside.unfinished_shared(_outside._shared_names(self.quest_root))
        if names:
            gone = _outside._remove_files(self.quest_root, names, left)
            _outside.say_removed(self._log, gone, left, cause="the earlier run, stopped from outside,")

    def _beat(self) -> None:
        while not self._stop.wait(self._beat_s):
            self._record["seen_at"] = time.time()
            with contextlib.suppress(OSError):
                _write(self.path, self._record)

    def finish(self, how: str = "finished") -> None:
        """End this session: ``how`` is recorded (``finished``, ``stopped``, or what failed)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if not self._record:
            return
        self._record.update(seen_at=time.time(), ended=str(how)[:200] or "finished")
        with contextlib.suppress(OSError):
            _write(self.path, self._record)
        self._history({"event": "ended", "pid": self._record.get("pid"), "how": self._record["ended"]})
