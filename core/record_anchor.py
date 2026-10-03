"""FI's own note of how far it wrote each of the quest's record files, so a line another program added is told apart.

A simulation script runs as the same user as FI. It could append a well-formed line to the quest's decision trace
(``.fi/audit.jsonl``, hash-chained, so a line chained onto the last one verifies) or to one of the record files the
evidence level is read from, and then kill FI: on resume, FI would have taken the line as its own. To tell them apart,
FI notes, for each record file (:data:`ANCHORED`), how far it wrote it (its size, its line count and a running digest of
its lines) in ``.fi/record_heads.sqlite``: before every line it appends the note moves on and says the line is on its
way (``pending``), and once the line is written it says it landed. At the end of every step the engine copies the note
into its own checkpoint (``.fi/state.sqlite``, the ``record_anchor`` state key).

* Before each line FI appends (and when a process first opens the trace), the file's size is compared with the note:
  bytes past it are moved aside to a named file (``<name>.outside_fi.jsonl`` beside it) and never used, so FI never
  writes onto them; lines of FI's that are gone are recorded. Both are kept in the note (:func:`evidence_gaps`).
* When a quest starts again (:func:`check_on_start`), every file is compared with the note line by line (and with the
  checkpoint's copy): what FI did not write is moved aside; a change to lines FI wrote is reported. The engine says
  each in run.log and records it in the trace (:data:`EVENT`).
* The evidence (:func:`evidence_gaps`) reads those records, any moved-aside file, and compares the files with the note
  again, so the result is not publication-ready after any of them.

A line FI noted but never wrote (stopped before or while writing it) is no alarm: a part-written last line of the trace
is left for the trace's own repair (``audit_repair``), of another record file it is cut off.

What this does not do: a program running as the same user can also rewrite the note and the checkpoint. It stops a
script that changes the record files without them; the script can reach the quest's folder under every sandbox FI
has today.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: FI's note of how far it wrote each record file (in the quest's ``.fi/``). Not ``state.sqlite``: several parts of FI
#: read that file's existence as "this quest has run before", and the first record is written before it exists.
HEADS = "record_heads.sqlite"
#: How many lines FI wrote without managing to note them first (the note could not be written), over every run.
HEADS_LOST = "record_heads.lost"

#: The record files FI notes (names in ``.fi/``): the decision trace and the record files the evidence reads and the
#: final seal names (core/evidence.py ``SEALED_LEDGERS``), plus the model-call record kept after a seal.
ANCHORED = ("audit.jsonl", "attempts.jsonl", "branch_ledger.jsonl", "model_calls.jsonl", "model_calls.after_seal.jsonl",
            "shadow_recommendations.jsonl")
_TRACE = "audit.jsonl"

#: The trace event that says a record file was changed outside FI.
EVENT = "record_changed_outside_fi"

#: The marker in a moved-aside file's name: ``audit.jsonl`` -> ``audit.outside_fi.jsonl`` (``.2.jsonl`` ... after).
MOVED_MARK = ".outside_fi"

# The size a process left a record file at after writing a line it could not note first (path -> size): FI's own
# line, so the next note goes on from it instead of setting it aside. Entries are per file and removed once used.
_UNNOTED: dict[str, int] = {}

# One lock per quest folder, so the note and the line are written together and in order even when model calls are
# recorded from worker threads; quests of a --fleet do not wait for each other.
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(fi_dir: Path) -> threading.RLock:
    key = str(fi_dir.resolve()).lower()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _step(digest: str, line: bytes) -> str:
    """The running digest after one more line (``line`` without its newline)."""
    return hashlib.sha256(digest.encode("ascii") + b"\n" + line).hexdigest()


@dataclass(frozen=True)
class Head:
    size: int = 0
    lines: int = 0
    digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"size": self.size, "lines": self.lines, "digest": self.digest}

    @classmethod
    def from_any(cls, value: Any) -> "Head | None":
        if not isinstance(value, dict):
            return None
        try:
            return cls(int(value["size"]), int(value["lines"]), str(value["digest"]))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Row:
    now: Head
    prev: Head
    pending: bool  # ``now`` is a line FI noted and has not said it wrote


@dataclass
class Finding:
    """One record file that was changed outside FI."""
    file: str            # the file's name in .fi/
    reason: str          # plain words: what was found
    lines: int = 0       # lines moved aside (0: nothing was moved)
    moved_to: str = ""   # the file they were moved to (name in .fi/), "" when nothing was moved

    def line(self) -> str:
        where = f"; moved aside to .fi/{self.moved_to}, not used" if self.moved_to else ""
        return f"the quest's record was changed outside FI: .fi/{self.file} {self.reason}{where}"


# ---- the note ---------------------------------------------------------------------------------------------------------


def _connect(fi_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(fi_dir / HEADS), timeout=15, isolation_level=None)
    try:
        # The default rollback journal: a killed process leaves the note whole. No sync to disk: only a power cut could
        # lose the last note, and that shows as a finding, never as a line taken for FI's.
        con.execute("PRAGMA synchronous=OFF")
        con.execute(
            "CREATE TABLE IF NOT EXISTS heads (name TEXT PRIMARY KEY, size INTEGER, lines INTEGER, digest TEXT, "
            "prev_size INTEGER, prev_lines INTEGER, prev_digest TEXT, pending INTEGER)")
        con.execute(
            "CREATE TABLE IF NOT EXISTS found (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, reason TEXT, "
            "lines INTEGER, moved_to TEXT, reported INTEGER DEFAULT 0)")
    except BaseException:
        con.close()
        raise
    return con


def _rows(con: sqlite3.Connection) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for name, size, lines, digest, psize, plines, pdigest, pending in con.execute(
            "SELECT name, size, lines, digest, prev_size, prev_lines, prev_digest, pending FROM heads"):
        out[str(name)] = Row(Head(int(size), int(lines), str(digest)), Head(int(psize), int(plines), str(pdigest)),
                             bool(pending))
    return out


def _put(con: sqlite3.Connection, name: str, now: Head, prev: Head, pending: bool) -> None:
    con.execute(
        "INSERT OR REPLACE INTO heads (name, size, lines, digest, prev_size, prev_lines, prev_digest, pending) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (name, now.size, now.lines, now.digest, prev.size, prev.lines, prev.digest, int(pending)))


def _record(con: sqlite3.Connection, f: Finding, *, reported: bool = False) -> None:
    con.execute("INSERT INTO found (name, reason, lines, moved_to, reported) VALUES (?, ?, ?, ?, ?)",
                (f.file, f.reason, f.lines, f.moved_to, int(reported)))


def _walk(data: bytes) -> tuple[list[Head], bytes]:
    """The head after each complete line of ``data`` (index 0: nothing yet) and the part-written tail after the last
    newline."""
    heads = [Head()]
    cut = data.rfind(b"\n") + 1
    at, digest = 0, ""
    for line in data[:cut].split(b"\n")[:-1]:
        at += len(line) + 1
        digest = _step(digest, line)
        heads.append(Head(at, len(heads), digest))
    return heads, data[cut:]


def _checkpoint_copy(fi_dir: Path) -> dict[str, Any] | None:
    """The copy of the note in FI's last checkpoint (``.fi/state.sqlite``), ``None`` when there is none (a new quest,
    a quest from before the note, or a checkpoint that cannot be read)."""
    db = fi_dir / "state.sqlite"
    if not db.is_file():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from langgraph.checkpoint.sqlite import SqliteSaver
        con = sqlite3.connect(str(db), timeout=15, check_same_thread=False)
        try:
            last = SqliteSaver(con).get_tuple({"configurable": {"thread_id": fi_dir.parent.name}})
        finally:
            con.close()
    except Exception:  # noqa: BLE001 -- an unreadable checkpoint: the note in .fi/ is still checked
        return None
    if last is None:
        return None
    kept = (last.checkpoint.get("channel_values") or {}).get("record_anchor")
    return kept if isinstance(kept, dict) and kept else None


def _start_note(con: sqlite3.Connection, fi_dir: Path, kept: dict[str, Any] | None = None) -> None:
    """The note is new (none in ``.fi/``). With no copy in FI's checkpoint (a new quest, or one from before the note)
    the record files there now are taken as they are. With one, the note was removed: only what the copy names is
    FI's, and that is recorded."""
    kept = kept or _checkpoint_copy(fi_dir)
    if kept is None:
        for name in ANCHORED:
            try:
                heads, torn = _walk((fi_dir / name).read_bytes())
            except OSError:
                continue
            head = heads[-1]
            if torn and name == _TRACE:
                # A line half-written when an earlier FI stopped: noted as FI's line on its way, for the trace's repair.
                _put(con, name, Head(head.size + len(torn) + 1, head.lines + 1, ""), head, True)
                continue
            if torn:
                try:
                    with (fi_dir / name).open("r+b") as fh:
                        fh.truncate(head.size)  # an earlier FI's half-written line
                except OSError:
                    pass
            _put(con, name, head, head, False)
        return
    for name, value in kept.items():
        head = Head.from_any(value)
        if name in ANCHORED and head is not None:
            _put(con, name, head, head, False)
    _record(con, Finding(HEADS, "(FI's note of how far it wrote the records) was removed; only what its last "
                                "checkpoint names is taken as FI's"))


def _open_note(fi_dir: Path, kept: dict[str, Any] | None = None) -> sqlite3.Connection:
    """The note, started if it is new, inside an open write transaction (``BEGIN IMMEDIATE``)."""
    fi_dir.mkdir(parents=True, exist_ok=True)
    new = not (fi_dir / HEADS).is_file()
    con = _connect(fi_dir)
    try:
        con.execute("BEGIN IMMEDIATE")
        if new and not con.execute("SELECT 1 FROM heads LIMIT 1").fetchone():
            _start_note(con, fi_dir, kept)
    except BaseException:
        con.close()
        raise
    return con


def _moved_name(fi_dir: Path, name: str) -> str:
    stem, dot, ext = name.rpartition(".")
    stem = stem if dot else name
    candidate = f"{stem}{MOVED_MARK}.{ext}" if dot else f"{name}{MOVED_MARK}"
    n = 2
    while (fi_dir / candidate).exists():
        candidate = f"{stem}{MOVED_MARK}.{n}.{ext}" if dot else f"{name}{MOVED_MARK}.{n}"
        n += 1
    return candidate


def _move_tail(path: Path, at: int, tail: bytes) -> str:
    """Move ``tail`` (the file's bytes from ``at`` on) to a named file beside it and cut the file back to ``at``."""
    target = _moved_name(path.parent, path.name)
    (path.parent / target).write_bytes(tail)
    with path.open("r+b") as fh:
        fh.truncate(at)
    return target


def _lines_in(tail: bytes) -> int:
    return tail.count(b"\n") + (0 if tail.endswith(b"\n") or not tail else 1)


def _reconcile(con: sqlite3.Connection, path: Path, row: Row | None, *, leave_torn: bool,
               unnoted: int | None = None) -> Head:
    """Compare the file's size with the note before FI writes to it, and return where FI goes on from. Bytes past the
    note are moved aside (never written onto); FI's own part-written last line is cut off (or, for the trace, left to
    its reopening, ``leave_torn``); lines of FI's that are gone are recorded. O(1) unless something is found.
    ``unnoted``: the size this process left the file at after a line it wrote without noting it first (the note could
    not be written; counted in :data:`HEADS_LOST`): a file still at that size is FI's as it is."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    if unnoted is not None and size == unnoted:
        try:
            return _walk(path.read_bytes())[0][-1]
        except FileNotFoundError:
            return Head()
    if row is None:
        # FI has not written this file since it began keeping the note: anything in it now is not FI's.
        row = Row(Head(), Head(), False)
    now, prev, pending = row.now, row.prev, row.pending
    if size == now.size:
        return now
    if pending and size == prev.size:
        return prev  # the last line FI noted never landed
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    if pending and prev.size < size < now.size and b"\n" not in data[prev.size:]:
        if leave_torn:
            return prev  # FI's own part-written line: the trace's reopening drops it and records that
        with path.open("r+b") as fh:
            fh.truncate(prev.size)
        return prev
    base = prev if pending and size < now.size else now
    if size > base.size:
        tail = data[base.size:]
        target = _move_tail(path, base.size, tail)
        _record(con, Finding(path.name, f"had {_lines_in(tail)} line(s) after the last one FI wrote",
                             _lines_in(tail), target))
        return base
    _record(con, Finding(path.name, "is shorter than what FI wrote: lines FI wrote were removed"))
    # Recorded once (the evidence and the next start read it): FI goes on from the file as it now is.
    heads, torn = _walk(data)
    if torn:
        target = _move_tail(path, heads[-1].size, torn)
        _record(con, Finding(path.name, "had a part-written line after the lines FI wrote", 1, target))
    return heads[-1]


@contextmanager
def appending(path: Path, line: bytes) -> Iterator[None]:
    """Note that FI is about to append ``line`` (with its newline) to ``path``, let the caller write it, then note it
    landed. A write that fails is cut back and the note put back. Files outside :data:`ANCHORED` pass through untouched;
    a note that cannot be written never stops the write (it is counted, :data:`HEADS_LOST`)."""
    if path.name not in ANCHORED:
        yield
        return
    fi_dir = path.parent
    body = line[:-1] if line.endswith(b"\n") else line
    with _lock(fi_dir):
        con: sqlite3.Connection | None = None
        base: Head | None = None
        try:
            con = _open_note(fi_dir)
            try:
                base = _reconcile(con, path, _rows(con).get(path.name), leave_torn=False,
                                  unnoted=_UNNOTED.pop(str(path), None))
                nxt = Head(base.size + len(body) + 1, base.lines + 1, _step(base.digest, body))
                _put(con, path.name, nxt, base, True)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        except (OSError, sqlite3.Error, ValueError):
            if con is not None:
                con.close()
                con = None
            base = None
            _count_lost(fi_dir)
        try:
            yield
            if con is None:
                try:
                    _UNNOTED[str(path)] = path.stat().st_size
                except OSError:
                    pass
        except BaseException:
            if con is not None and base is not None:
                try:
                    if path.is_file() and path.stat().st_size > base.size:
                        with path.open("r+b") as fh:
                            fh.truncate(base.size)  # a line half-written before the error
                    _put(con, path.name, base, base, False)
                except (OSError, sqlite3.Error):
                    pass
            raise
        else:
            if con is not None:
                try:
                    con.execute("UPDATE heads SET pending = 0 WHERE name = ?", (path.name,))
                except sqlite3.Error:
                    pass
        finally:
            if con is not None:
                con.close()


def before_reading(path: Path) -> None:
    """Before a process reads the trace to go on from its last line (``AuditLog._open``): bytes past FI's note are moved
    aside first, so FI never chains onto them. A part-written last line of FI's own is left for the reopening."""
    if path.name not in ANCHORED or not path.parent.is_dir():
        return
    with _lock(path.parent):
        try:
            con = _open_note(path.parent)
        except (OSError, sqlite3.Error):
            return
        try:
            try:
                row = _rows(con).get(path.name)
                if row is not None or path.is_file():
                    base = _reconcile(con, path, row, leave_torn=True, unnoted=_UNNOTED.pop(str(path), None))
                    size = path.stat().st_size if path.is_file() else 0
                    if row is not None and base != row.now and size == base.size:
                        _put(con, path.name, base, base, False)
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        except (OSError, sqlite3.Error):
            pass
        finally:
            con.close()


def _count_lost(fi_dir: Path) -> None:
    try:
        target = fi_dir / HEADS_LOST
        target.write_text(str(lost(fi_dir) + 1), encoding="utf-8")
    except OSError:
        pass


def lost(fi_dir: Path) -> int:
    """How many lines FI wrote without managing to note them first, over every run of the quest."""
    try:
        return int((fi_dir / HEADS_LOST).read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def snapshot(fi_dir: Path) -> dict[str, Any]:
    """FI's note as it is now (name -> size, lines, digest; a line not yet landed is left out), for the engine's
    checkpoint at the end of a step."""
    if not (fi_dir / HEADS).is_file():
        return {}
    try:
        with _lock(fi_dir):
            con = _connect(fi_dir)
            try:
                return {name: (r.prev if r.pending else r.now).as_dict() for name, r in _rows(con).items()}
            finally:
                con.close()
    except (OSError, sqlite3.Error):
        return {}


# ---- the full comparison ------------------------------------------------------------------------------------------


def _find(heads: list[Head], target: Head | None) -> int | None:
    if target is None or target.lines >= len(heads):
        return None
    got = heads[target.lines]
    return target.lines if (got.size, got.digest) == (target.size, target.digest) else None


def check(fi_dir: Path, checkpoint: dict[str, Any] | None = None, *, repair: bool = False) -> list[Finding]:
    """Compare each record file with FI's note line by line (and, when given, with the copy in FI's last checkpoint).

    ``repair`` (a quest starting again, before it writes anything): move what FI did not write to a named file beside
    it, cut FI's own part-written last line off a record file (the trace's is left for its reopening), and settle the
    note to the file as it now is. Without ``repair`` nothing is changed and nothing is created (the evidence uses
    that)."""
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    findings: list[Finding] = []
    if not fi_dir.is_dir() or (not repair and not (fi_dir / HEADS).is_file()):
        return findings
    with _lock(fi_dir):
        try:
            con = _open_note(fi_dir, checkpoint or None) if repair else _connect(fi_dir)
        except (OSError, sqlite3.Error):
            return findings
        try:
            rows = _rows(con)
            for name in ANCHORED:
                f = _check_one(con, fi_dir, name, rows.get(name), Head.from_any(checkpoint.get(name)), repair)
                if f is not None:
                    findings.append(f)
            if repair:
                con.execute("COMMIT")
        except BaseException:
            if repair:
                try:
                    con.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        finally:
            con.close()
    return findings


def _check_one(con: sqlite3.Connection, fi_dir: Path, name: str, noted: Row | None, kept: Head | None,
               repair: bool) -> Finding | None:
    path = fi_dir / name
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = b""
    except OSError:
        return None
    heads, torn = _walk(data)
    last = len(heads) - 1
    reason = ""
    benign_torn = False
    if noted is not None:
        cut = _find(heads, noted.now)
        if cut is None and noted.pending:
            cut = _find(heads, noted.prev)  # the last line FI noted never landed (or only part of it did)
            benign_torn = cut is not None and cut == last and 0 < len(torn) < noted.now.size - noted.prev.size
    elif last == 0 and not torn:
        return None  # a file FI has not written, and nobody else has either
    else:
        cut, reason = 0, "(FI never wrote to it) "
    if cut is not None and kept is not None and (kept.lines > cut or _find(heads, kept) is None):
        # The note and the checkpoint's copy disagree: the note was put back, or the lines before it were rewritten.
        return Finding(name, reason + "does not match what FI's last checkpoint says it wrote")
    if cut is None:
        return Finding(name, reason + "no longer holds the lines FI wrote (lines were changed, removed or put in "
                                      "between them)")
    extra = last - cut
    if extra <= 0 and (not torn or benign_torn):
        if repair:
            if torn and name == _TRACE:
                return None  # FI's own part-written line, left (with its note) for the trace's reopening
            if torn:
                with path.open("r+b") as fh:
                    fh.truncate(heads[cut].size)  # FI's own part-written last line
            _put(con, name, heads[cut], heads[cut], False)
        return None
    tail = data[heads[cut].size:]
    count = _lines_in(tail)
    if not repair:
        return Finding(name, f"{reason}has {count} line(s) after the last one FI wrote", count)
    try:
        target = _move_tail(path, heads[cut].size, tail)
    except OSError as e:
        return Finding(name, f"{reason}has {count} line(s) after the last one FI wrote, which could not be moved "
                             f"aside ({e})")
    _put(con, name, heads[cut], heads[cut], False)
    return Finding(name, f"{reason}had {count} line(s) after the last one FI wrote", count, target)


def check_on_start(fi_dir: Path) -> list[Finding]:
    """What a quest starting again does before it writes anything: :func:`check` with ``repair`` against the note and
    its copy in FI's last checkpoint, plus what FI found and moved aside while the quest last ran and has not said yet.
    Each finding is kept in the note, so the evidence reads it."""
    if not fi_dir.is_dir():
        return []
    found = check(fi_dir, _checkpoint_copy(fi_dir), repair=True)
    with _lock(fi_dir):
        try:
            con = _connect(fi_dir)
        except (OSError, sqlite3.Error):
            return found
        try:
            earlier = [Finding(str(n), str(r), int(c or 0), str(m or "")) for n, r, c, m in con.execute(
                "SELECT name, reason, lines, moved_to FROM found WHERE reported = 0 ORDER BY id")]
            con.execute("UPDATE found SET reported = 1 WHERE reported = 0")
            for f in found:
                _record(con, f, reported=True)
        except sqlite3.Error:
            earlier = []
        finally:
            con.close()
    return earlier + found


def moved_files(fi_dir: Path) -> list[str]:
    """Files lines were moved aside to (names in ``.fi/``)."""
    try:
        return sorted(p.name for p in fi_dir.iterdir() if MOVED_MARK in p.name and p.is_file())
    except OSError:
        return []


def _recorded(fi_dir: Path) -> list[Finding]:
    if not (fi_dir / HEADS).is_file():
        return []
    try:
        con = _connect(fi_dir)
        try:
            return [Finding(str(n), str(r), int(c or 0), str(m or "")) for n, r, c, m in con.execute(
                "SELECT name, reason, lines, moved_to FROM found ORDER BY id")]
        finally:
            con.close()
    except (OSError, sqlite3.Error):
        return []


def evidence_gaps(fi_dir: Path, events: list[dict[str, Any]] | None = None) -> list[str]:
    """What keeps the result from being publication-ready because a record file was changed outside FI: a change found
    while the quest ran or when it started again (kept in the note, in the trace, or a moved-aside file), a file that
    does not match FI's note now, or lines FI could not note."""
    gaps: list[str] = []
    found = [Finding(str(e.get("file")), str(e.get("reason") or ""), int(e.get("lines") or 0), str(e.get("moved_to") or ""))
             for e in (events or []) if isinstance(e, dict) and e.get("kind") == EVENT and e.get("file")]
    found += _recorded(fi_dir)
    for line in dict.fromkeys(f.line() for f in found):
        gaps.append(line)
    named = {f.moved_to for f in found}
    unexplained = [m for m in moved_files(fi_dir) if m not in named]
    if unexplained:
        gaps.append("the quest's record was changed outside FI: lines FI did not write were set aside to " +
                    ", ".join(f".fi/{m}" for m in unexplained))
    try:
        live = check(fi_dir)
    except Exception:  # noqa: BLE001 -- a check that cannot run is not a finding
        live = []
    gaps.extend(f.line() for f in live)
    n = lost(fi_dir)
    if n:
        gaps.append(f"FI could not keep its note of how far it wrote the quest's records for {n} line(s), so lines "
                    "another program added cannot be told apart from them")
    return gaps
