"""FI runs the quest's code from a copy outside the quest folder, so FI's own records are not next to it.

Code a model or a coding agent wrote runs on this computer as the person (there is no sandbox on the normal path). Run
from the quest folder, it could open ``needs/FROZEN_PROTOCOL.json`` (the checks' expected values and tolerances),
``.fi/checks/`` or ``.fi/state.sqlite`` by a plain relative path. :class:`OutsideCopy` makes a scratch folder under the
system's temporary folder, named by a random id (never by the quest's name), with:

* a copy of ``code/`` (no ``.git`` history, no ``.venv``, no caches, no value file an older FI left in ``code/tests/``),
* the folders the run reads: for a cell of the study, a check case or an equation measurement only ``inputs/`` and
  ``data/`` (:data:`CELL_INPUTS`, without the results written under ``data/results/``); for the analysis every folder
  of the quest but FI's own (:data:`ANALYSIS_INPUTS`: ``raw/``, ``results/`` and whatever else the cells wrote),
* the files and folders FI itself names for this one run, on the command line or in the environment (its harness, the
  spec of a setting, the file a result goes to);

and never ``needs/``, ``.fi/checks/`` or the rest of ``.fi/``. After the run FI copies back everything the run wrote,
created or removed, except into FI's own folders (:data:`FI_OWNED`: ``needs/``, ``.fi/`` other than the paths it named,
``code/`` and ``paper/``): those are dropped, and each dropped path is said in the run log. Paths that name the copy are written
back as the quest's in what the run printed and in FI's own result files, so a traceback still points at ``code/...``.

One folder is shared, not copied: ``data/results/cache/`` (:data:`SHARED_CACHE`), where a computation several settings
use is kept. A cell's copy reaches the quest's own folder through a link (a directory junction on Windows), so a value one
setting worked out is there for the next, and settings run at the same time see each other's finished files. What a cell
writes there is the quest's at once, so it is never copied back or removed by the copy-back. The folder belongs to one
version of the simulation's code (``simulate.py`` and its model package): when it changes (a repair, a speed-up), what
an earlier version kept there is moved aside to
``.fi/previous/`` before anything runs, so no value worked out by other code is read (:func:`prepare_shared_cache`).

This is not a sandbox. Code that searches the disk by absolute path, or starts from its own interpreter (the quest's
``.venv/`` stays in the quest folder), can still find the quest; the copy only stops reads and writes by relative path.
:class:`OutsideExecutor` wraps an executor so a run whose working folder is the quest folder runs from such a copy.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

#: The folder under the system's temporary folder that holds every copy (one sub-folder per run, a random id).
SCRATCH_DIR = "fi_run"
CODE = "code"
#: What a cell of the study, a check case or an equation measurement is given besides ``code/``: the person's inputs and
#: the data the quest collected, without what the study writes under ``data/`` (:data:`CELL_SKIP`).
CELL_INPUTS = ("inputs", "data")
CELL_SKIP = ("data/results",)
#: The one folder every cell of the study shares while it runs (see the module docstring): linked, not copied.
SHARED_CACHE = "data/results/cache"
#: FI's record of which version of ``code/`` filled :data:`SHARED_CACHE`.
SHARED_CACHE_RECORD = ".fi/shared_cache.json"
#: What the analysis is given: every folder of the quest but FI's own (:data:`FI_OWNED`), so it reads
#: whatever the cells wrote (``raw/``, ``results/``, a ``cache/`` of their own); never a file at the top of the quest
#: (``plan.md`` holds the checks' expected values).
EVERY_FOLDER = "*"
ANALYSIS_INPUTS = (EVERY_FOLDER,)
#: FI's own folders: what a run writes there in its copy never comes back (except a path FI named for the run).
FI_OWNED = ("needs", ".fi", "code", "paper")
#: Never in a copy and never written back, even when named.
NEVER = ("needs", ".fi/checks", ".venv", ".git", ".fi/state.sqlite")
#: FI's own records inside the folders a harness writes in (core/execution.py keeps them read-only in a container).
RECORD_FILES = (".fi/trials/run.json", ".fi/trials/cluster.json", ".fi/optimisation/run.json",
                ".fi/optimisation/check.json")
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
#: Files an older FI left in ``code/tests/`` with expected values in them (core/equation_tests.STALE_NAMES).
_STALE_TEST_FILES = {"equation_numbers.json", "equation_values.json", "oracle_values.json"}
_SMALL = 2_000_000
_TEXT_SUFFIXES = (".json", ".jsonl", ".txt", ".log")


def scratch_base(base: Path | str | None = None) -> Path:
    return (Path(base) if base is not None else Path(tempfile.gettempdir()) / SCRATCH_DIR).resolve()


def new_scratch(base: Path | str | None = None) -> Path:
    """A new, empty folder for one run: ``<temp>/fi_run/<random id>``."""
    root = scratch_base(base) / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


#: Folders a trial killed for its time is not checked in: FI's own records (written by FI, never by the trial) and the
#: quest's environment and history.
NOT_CHECKED_AFTER_KILL = ("needs", ".fi", "paper", "raw", ".venv", ".git")


def working_files(quest_root: Path | str) -> set[str]:
    """The names (relative, with ``/``) of the files under the quest's working folders (``code/``, ``data/``, ``inputs/``
    and any other folder but :data:`NOT_CHECKED_AFTER_KILL`): the list taken when a trial starts. Names only, nothing is
    read, so it stays cheap beside a large data folder."""
    root = Path(quest_root)
    out: set[str] = set()
    if not root.is_dir():
        return out
    for top in root.iterdir():
        if not top.is_dir() or top.name in NOT_CHECKED_AFTER_KILL or top.name in _SKIP_DIRS:
            continue
        for here, names, leaves in os.walk(top):
            names[:] = [n for n in names if n not in _SKIP_DIRS]
            out.update((Path(here) / leaf).relative_to(root).as_posix() for leaf in leaves if not leaf.endswith(".pyc"))
    return out


#: FI's record of the files each setting that FINISHED wrote (path -> size and modification time at that moment).
FINISHED_FILES_RECORD = ".fi/finished_files.json"
#: Where a file a stopped setting left half-written is moved (not deleted), under ``.fi/``.
STOPPED_ASIDE = ".fi/previous"
_record_lock = threading.Lock()


def _file_state(quest_root: Path, rel: str) -> list[int] | None:
    """A file's size and modification time (ns) as FI records it, or None when it is not there."""
    try:
        st = (quest_root / rel).stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def finished_files(quest_root: Path | str) -> dict[str, list[int]]:
    """FI's record of what finished settings wrote (:data:`FINISHED_FILES_RECORD`); empty when there is none."""
    import json

    try:
        data = json.loads((Path(quest_root) / FINISHED_FILES_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, list)} if isinstance(data, dict) else {}


def working_states(quest_root: Path | str, names: set[str] | None = None) -> dict[str, list[int]]:
    """The size and modification time of the working files (or of ``names``): the snapshot taken when a setting starts,
    so a file it wrote over is told by its state changing, not by a clock."""
    root = Path(quest_root)
    return {r: s for r in (working_files(root) if names is None else names) if (s := _file_state(root, r)) is not None}


def _rewritten(quest_root: Path, states: dict[str, list[int]]) -> set[str]:
    """Of the files in ``states``, the ones whose state now differs from it (written over, or gone-and-back)."""
    return {r for r, s in states.items() if _file_state(quest_root, r) not in (None, s)}


#: The runs now running in each quest (by its folder), each with whether another ran at any time beside it.
_active: dict[str, dict[int, bool]] = {}


def begin_run(quest_root: Path | str) -> int:
    """Note that a run began in this quest; the token to give :func:`end_run`. Every run already running is marked as
    overlapped, and so is this one when another runs."""
    with _record_lock:
        runs = _active.setdefault(str(Path(quest_root).resolve()), {})
        for other in runs:
            runs[other] = True
        token = uuid.uuid4().int
        runs[token] = bool(runs)
        return token


def run_overlapped(quest_root: Path | str, token: int) -> bool:
    """Whether another run overlapped this one so far (the run stays registered)."""
    with _record_lock:
        return _active.get(str(Path(quest_root).resolve()), {}).get(token, False)


def end_run(quest_root: Path | str, token: int) -> bool:
    """Note that the run ended; whether another run overlapped it at any time."""
    with _record_lock:
        key = str(Path(quest_root).resolve())
        runs = _active.get(key, {})
        overlapped = runs.pop(token, False)
        if not runs:
            _active.pop(key, None)
        return overlapped


def record_finished(quest_root: Path | str, start_states: dict[str, list[int]], overlapped: bool = False) -> list[str]:
    """After a setting FINISHED: record the state of every working file that is new or changed since ``start_states``
    (the snapshot at its start). A later setting that is stopped for its time can then tell a file it left half-written
    (its state is not one a finished setting recorded) from one a finished setting wrote whole, even when both are under
    the file's final name. When another setting ran at the same time (``overlapped``) nothing is recorded: FI cannot tell
    which of them wrote a changed file, and a file the other left half-written must not be taken for a finished one. Not
    recording errs toward moving a file aside, which loses nothing."""
    import json

    root = Path(quest_root)
    if overlapped:
        return []
    now = working_states(root)
    wrote = {r: st for r, st in now.items() if start_states.get(r) != st}
    if not wrote:
        return []
    with _record_lock:
        record = finished_files(root)
        record.update(wrote)
        path = root / FINISHED_FILES_RECORD
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            return []
    return sorted(wrote)


def _not_a_finished_state(quest_root: Path, rels: set[str]) -> set[str]:
    """Of these files, the ones whose state now is not one a finished setting recorded."""
    record = finished_files(quest_root)
    return {r for r in rels if record.get(r) != _file_state(quest_root, r)}


def _move_aside(quest_root: Path | str, rels: list[str], left: list[str] | None = None) -> list[str]:
    """Move these files of the quest to :data:`STOPPED_ASIDE` (nothing is deleted); the ones moved. A file that cannot
    be moved is tried once more, then added to ``left``."""
    import time

    root = Path(quest_root)
    aside = root / STOPPED_ASIDE / f"stopped_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    moved: list[str] = []
    for rel in rels:
        for attempt in range(2):
            try:
                target = aside / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(root / rel, target)
                moved.append(rel)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == 0:
                    time.sleep(0.5)
                elif left is not None:
                    left.append(rel)
    return moved


def remove_created_since(quest_root: Path | str, before: set[str], left: list[str] | None = None, *,
                         states: dict[str, list[int]] | None = None) -> list[str]:
    """Remove the files under the quest's working folders that were not in ``before``; the paths removed. For a trial
    FI stopped for its time: it did not finish, so what it was writing (a lock, a half-written cache entry) is not
    something the next trial may trust. A file that cannot be removed (on Windows a handle of the stopped process can
    outlive it for a moment) is tried once more, then added to ``left``.

    With ``states`` (:func:`working_states`, taken when the stopped setting began) two more things hold. A file that was there before and was written
    over during the stopped setting (code that saves straight under the final name, and was cut off part way) is moved
    aside to :data:`STOPPED_ASIDE`, not left to be read as a result; and a file, new or rewritten, whose state is one a
    finished setting recorded (:func:`record_finished`) is kept: a finished setting wrote it whole."""
    root = Path(quest_root)
    created = working_files(root) - before
    # A finished value in the shared folder stays (unfinished_shared): only its locks and temporary files go.
    shared = {r for r in created if r == SHARED_CACHE or r.startswith(SHARED_CACHE + "/")}
    doomed = (created - shared) | set(unfinished_shared(shared))
    rewritten: set[str] = set()
    if states is not None:
        # FI's own folders (code/) are not written by a setting; a rewrite there is not a result.
        rewritten = {r for r in _rewritten(root, states)
                     if r.split("/", 1)[0] not in FI_OWNED}
        doomed = _not_a_finished_state(root, doomed)
        rewritten = _not_a_finished_state(root, rewritten)
    gone = _remove_files(root, sorted(doomed), left)
    return gone + _move_aside(root, sorted(rewritten), left)


def unfinished_shared(rels: list[str] | set[str]) -> list[str]:
    """Of these paths, the ones in :data:`SHARED_CACHE` that a stopped run left unfinished. The code writes every file
    there whole, to a temporary name it then renames, and marks a value being worked out with ``<name>.lock`` (the code
    writer is told so): a file under its own name is a finished value and is kept for the next setting; a lock or a
    temporary file (a name ending in ``.lock``, or in ``.tmp`` with or without digits after it, or with ``.tmp.`` in
    it, as a library adding its own extension to ``<name>.tmp`` writes; not ``x.tmp_table.npz``) is not. An exception by name, stated: FI cannot
    tell a lock from a value otherwise, and a value it removes is worked out again by every later setting. The exception
    covers files the stopped run CREATED; a file that was already there and was written over is judged by the record of
    finished settings instead (:func:`remove_created_since`, ``states``), because a rewrite under the final name is
    exactly what a cut-off save looks like."""
    out = []
    for rel in sorted(rels):
        if not (rel == SHARED_CACHE or rel.startswith(SHARED_CACHE + "/")):
            continue
        name = rel.rsplit("/", 1)[-1]
        if name.endswith(".lock") or re.search(r"\.tmp(\d*$|\.)", name):
            out.append(rel)
    return out


def _remove_files(quest_root: Path | str, rels: list[str], left: list[str] | None = None) -> list[str]:
    """Remove these files of the quest; the ones removed. A file that cannot be removed is tried once more, then added
    to ``left``."""
    import time

    root = Path(quest_root)
    gone: list[str] = []
    for rel in rels:
        for attempt in range(2):
            try:
                (root / rel).unlink()
                gone.append(rel)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == 0:
                    time.sleep(0.5)
                elif left is not None:
                    left.append(rel)
    return gone


def say_removed(log: Any, gone: list[str], left: list[str] | None = None, *, cause: str = "a trial stopped for its time") -> None:
    if gone:
        shown = ", ".join(gone[:5]) + (f" and {len(gone) - 5} more" if len(gone) > 5 else "")
        log.warning("[execute] %s left %d file%s it was writing (%s); removed, so the next "
                    "trial does not wait on %s", cause, len(gone), "" if len(gone) == 1 else "s", shown,
                    "it" if len(gone) == 1 else "them")
    if left:
        shown = ", ".join(left[:5]) + (f" and {len(left) - 5} more" if len(left) > 5 else "")
        log.warning("[execute] %s left %d file%s FI could not remove (%s); a later trial that "
                    "waits on %s may not finish", cause, len(left), "" if len(left) == 1 else "s", shown,
                    "it" if len(left) == 1 else "them")


async def execute_keeping_only_finished(executor: Any, cmd: list[str], *, cwd: Path, timeout_s: int,
                                        env: dict[str, str] | None = None, log: Any = None) -> Any:
    """``executor.execute(...)`` for a trial that runs in the quest folder itself: when FI stopped it for its time, the
    files it created are removed (see :func:`remove_created_since`); a trial that finished keeps what it wrote."""
    root = Path(cwd)
    await asyncio.to_thread(prepare_shared_cache, root, log)
    states = await asyncio.to_thread(working_states, root)
    before = set(states)
    token = begin_run(root)
    # The run stays registered until its files are recorded: a setting that begins after this one's process ended but
    # before the recording could be the writer of a file the recording would take.
    try:
        result = await executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        if getattr(result, "timed_out", False):
            left: list[str] = []
            gone = await asyncio.to_thread(lambda: remove_created_since(root, before, left, states=states))
            say_removed(log if log is not None else logging.getLogger(f"frontier_insight.{root.resolve().name}"), gone, left)
        elif getattr(result, "returncode", 0) in (0, None):
            await asyncio.to_thread(lambda: record_finished(root, states, run_overlapped(root, token)))
        return result
    finally:
        end_run(root, token)


def code_version(quest_root: Path | str) -> str:
    """A hash of the simulation's code as it is now: ``code/simulate.py`` and the model package beside it, the code each
    setting runs and the code that decides a set of trials (``core.trial_runner._run_key``). Also every other file the
    simulation may read (:func:`core.code_layout.simulation_inputs`): a helper module beside ``simulate.py``, a parameter
    file in ``code/``, a file the person put in the quest's ``inputs/``. Changing any of them changes what the simulation
    computes, so the cached values of the old version are not reused. The analysis script, the tests and the notes in
    ``code/`` work out nothing kept in :data:`SHARED_CACHE`: when a real run's analysis script was repaired after every
    setting had run, the values its unchanged simulation had kept were moved aside and worked out again."""
    from . import code_layout

    h = hashlib.sha256()
    root = Path(quest_root)
    code = root / CODE
    with contextlib.suppress(OSError):
        h.update(b"simulate.py\0" + (code / "simulate.py").read_bytes() + b"\0")
    if code.is_dir():
        for rel, body in sorted(code_layout.package_sources(code).items()):
            h.update(rel.encode("utf-8") + b"\0" + body.encode("utf-8") + b"\0")
    for path in code_layout.simulation_inputs(root):
        with contextlib.suppress(OSError):
            h.update(code_layout.input_label(path, root).encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


#: How long a setting waits for another to finish preparing :data:`SHARED_CACHE`, and the age at which FI's own lock on it
#: is taken to be left by a process that died while holding it.
PREPARE_WAIT_S = 60.0
PREPARE_STALE_S = 120.0


@contextlib.contextmanager
def _preparing(quest_root: Path):
    """One run at a time prepares the quest's shared folder: settings start at the same time, and a run that read the
    record before another wrote it would move aside the folder the other's settings already write into. A lock file in
    ``.fi/`` (atomic create), so threads and processes alike wait; one older than :data:`PREPARE_STALE_S` is broken."""
    import time

    lock = Path(quest_root) / ".fi" / "shared_cache.prepare.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    give_up = time.monotonic() + PREPARE_WAIT_S
    held = False
    while not held:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held = True
        except (FileExistsError, PermissionError):  # Windows: PermissionError while its holder is removing it
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > PREPARE_STALE_S:
                    lock.unlink()
                    continue
            if time.monotonic() > give_up:
                break  # prepared without the lock rather than not at all; the move below still checks what it finds
            time.sleep(0.05)
    try:
        yield
    finally:
        if held:
            with contextlib.suppress(OSError):
                lock.unlink()


def prepare_shared_cache(quest_root: Path | str, log: Any = None) -> Path | None:
    """The quest's :data:`SHARED_CACHE` folder, made if missing, holding only what the current version of ``code/`` kept
    there: when FI's record names another version, the folder is moved aside to ``.fi/previous/`` first (what is in it
    was worked out by other code). Returns the folder, or None when the folder of another version could not be moved
    aside (a file in it held open): then nothing may share it, and FI's record still names the other version."""
    import json
    import time

    root = Path(quest_root)
    folder = root / SHARED_CACHE
    record = root / SHARED_CACHE_RECORD
    version = code_version(root)
    with _preparing(root):
        try:
            kept = json.loads(record.read_text(encoding="utf-8")).get("code")
        except (OSError, ValueError, AttributeError):
            kept = None
        if kept != version:
            if folder.is_dir() and any(folder.iterdir()):
                aside = root / ".fi" / "previous" / f"shared_cache_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
                aside.parent.mkdir(parents=True, exist_ok=True)
                moved = False
                for attempt in range(2):
                    try:
                        os.replace(folder, aside)
                        moved = True
                        if log is not None:
                            log.info("[run] the code changed, so what the earlier code kept in %s/ was moved to %s (no "
                                     "value worked out by other code is read)", SHARED_CACHE,
                                     aside.relative_to(root).as_posix())
                        break
                    except OSError:
                        if attempt == 0:
                            time.sleep(0.5)
                if not moved and folder.is_dir() and any(folder.iterdir()):
                    if log is not None:
                        log.warning("[run] the code changed, but FI could not move aside what the earlier code kept in "
                                    "%s/ (a file in it is open), so this run does not share it", SHARED_CACHE)
                    return None
            record.parent.mkdir(parents=True, exist_ok=True)
            tmp = record.with_name(record.name + f".{uuid.uuid4().hex[:6]}.tmp")
            tmp.write_text(json.dumps({"code": version}), encoding="utf-8")
            os.replace(tmp, record)
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def _shared_names(quest_root: Path) -> set[str]:
    """The files in the quest's :data:`SHARED_CACHE` (relative to the quest, with ``/``)."""
    folder = Path(quest_root) / SHARED_CACHE
    out: set[str] = set()
    if folder.is_dir():
        for here, _names, leaves in os.walk(folder):
            out.update((Path(here) / leaf).relative_to(quest_root).as_posix() for leaf in leaves)
    return out


def _link_folder(link: Path, target: Path) -> bool:
    """Make ``link`` reach ``target`` (a directory junction on Windows, a symbolic link elsewhere); False when it could not."""
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, ImportError, AttributeError):
        return False


def _unlink_folder(link: Path) -> None:
    """Remove the link itself, never what it reaches."""
    with contextlib.suppress(OSError):
        if os.name == "nt":
            os.rmdir(link)  # a junction: removes the link, not its target's files
        else:
            os.unlink(link)


def _never(rel: str) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in NEVER) or rel in RECORD_FILES


def copy_code(code_dir: Path, dest: Path) -> None:
    """Copy ``code_dir`` into ``dest``: no history, environment, caches or stale value files."""
    def ignore(folder: str, names: list[str]) -> set[str]:
        skip = {n for n in names if n in _SKIP_DIRS or n.endswith(".pyc")}
        if Path(folder).name == "tests":
            skip |= {n for n in names if n in _STALE_TEST_FILES}
        return skip

    if Path(code_dir).is_dir():
        shutil.copytree(code_dir, dest, ignore=ignore, dirs_exist_ok=True)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)


def _sig(path: Path) -> Any:
    st = path.stat()
    if st.st_size <= _SMALL:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return (st.st_size, st.st_mtime_ns)


def _files(folder: Path, leave: tuple[str, ...] = ()) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not folder.is_dir():
        return out
    for here, names, leaves in os.walk(folder):
        rel_here = Path(here).relative_to(folder).as_posix()
        # a linked folder (``leave``) is the quest's own: never walked as part of the copy
        names[:] = [n for n in names if n not in _SKIP_DIRS
                    and (f"{rel_here}/{n}" if rel_here != "." else n) not in leave]
        for leaf in leaves:
            path = Path(here) / leaf
            with contextlib.suppress(OSError):
                out[path.relative_to(folder).as_posix()] = _sig(path)
    return out


class OutsideCopy:
    """One run's copy of the quest (see the module docstring). ``named``: paths relative to the quest that FI names for
    this run (copied in when they are there, written back when the run changed them)."""

    def __init__(self, quest_root: Path | str, *, named: list[str] | tuple[str, ...] = (), base: Path | str | None = None,
                 inputs: tuple[str, ...] = CELL_INPUTS, skip: tuple[str, ...] = CELL_SKIP, share: bool = True,
                 only_named: bool = False) -> None:
        self.quest_root = Path(quest_root).resolve()
        #: Whether :data:`SHARED_CACHE` may be linked into the copy (False: it was not prepared, or the run cannot follow
        #: a link to the quest's own folder); unlinked, the run works out its shared values in its own copy.
        self.share = share
        #: Only the paths FI named for the run come back (a check's run whose other files must not reach the quest).
        self.only_named = only_named
        self.named = sorted({n.strip("/") for n in named if n and not _never(n.strip("/"))})
        self.inputs = tuple(inputs)
        self.skip = tuple(skip)
        self._base = base
        self.root: Path | None = None
        self._before: dict[str, Any] = {}
        self.brought_back: list[str] = []
        self.left_out: list[str] = []
        self.unfinished: list[str] = []
        #: The folders linked to the quest's own (:data:`SHARED_CACHE`), relative to the copy.
        self.linked: list[str] = []
        #: :data:`SHARED_CACHE` could not be linked, so this run works out its shared values again.
        self.not_shared = False

    def _named(self, rel: str) -> bool:
        return any(rel == n or rel.startswith(n + "/") for n in self.named)

    def make(self) -> Path:
        root = self.root = new_scratch(self._base)
        copy_code(self.quest_root / CODE, root / CODE)
        inputs = self.inputs
        if EVERY_FOLDER in inputs and self.quest_root.is_dir():
            inputs = tuple(sorted(p.name for p in self.quest_root.iterdir() if p.is_dir() and p.name not in FI_OWNED
                                  and p.name not in _SKIP_DIRS))

        def ignore(folder: str, names: list[str]) -> set[str]:
            here = Path(folder).resolve().relative_to(self.quest_root).as_posix()
            return {n for n in names if n in _SKIP_DIRS or n.endswith(".pyc") or f"{here}/{n}" in self.skip}

        for rel in (*inputs, *self.named):
            src = self.quest_root / rel
            if rel == CODE or rel.startswith(CODE + "/") or _never(rel) or rel in self.skip:
                continue
            if src.is_dir():
                shutil.copytree(src, root / rel, ignore=ignore, dirs_exist_ok=True)
            elif src.is_file():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, root / rel)
            elif "/" in rel:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
        # The study's shared folder, when the copy would not hold it: linked to the quest's own, not copied.
        if any(SHARED_CACHE == s or SHARED_CACHE.startswith(s + "/") for s in self.skip):
            target = self.quest_root / SHARED_CACHE
            if self.share and target.is_dir() and _link_folder(root / SHARED_CACHE, target):
                self.linked.append(SHARED_CACHE)
            else:
                self.not_shared = True
        self._before = _files(root, tuple(self.linked))
        return root

    def _comes_back(self, rel: str) -> bool:
        if _never(rel):
            return False
        if any(rel == n or rel.startswith(n + "/") for n in self.named):
            return not (rel == CODE or rel.startswith(CODE + "/"))
        return not self.only_named and rel.split("/", 1)[0] not in FI_OWNED

    def _forms(self) -> list[tuple[str, str]]:
        assert self.root is not None
        pairs = []
        for a, b in ((str(self.root), str(self.quest_root)), (self.root.as_posix(), self.quest_root.as_posix())):
            pairs += [(a.replace("\\", "\\\\"), b.replace("\\", "\\\\")), (a, b)]
        return pairs

    def to_quest(self, text: str) -> str:
        """``text`` with every path into the copy written as the same path in the quest."""
        if not text or self.root is None:
            return text
        for a, b in self._forms():
            text = text.replace(a, b)
        return text

    def bring_back(self, *, stopped: bool = False) -> list[str]:
        """Copy into the quest what the run wrote (created, changed, removed), except into FI's own folders
        (:data:`FI_OWNED`, other than the named paths): those are left out and listed in :attr:`left_out`. The paths
        written back. ``stopped``: FI stopped the run for its time, so it did not finish; a file it created (not a path
        FI named for it) is not written back, and is listed in :attr:`unfinished`."""
        assert self.root is not None
        now = _files(self.root, tuple(self.linked))
        back: list[str] = []
        for rel in sorted(set(now) | set(self._before)):
            if now.get(rel) == self._before.get(rel):
                continue
            if not self._comes_back(rel):
                self.left_out.append(rel)
                continue
            if stopped and rel in now and rel not in self._before and not self._named(rel):
                self.unfinished.append(rel)
                continue
            dst = self.quest_root / rel
            if rel not in now:
                with contextlib.suppress(OSError):
                    dst.unlink()
                back.append(rel)
                continue
            src = self.root / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if rel.startswith(".fi/") and rel.endswith(_TEXT_SUFFIXES) and src.stat().st_size <= _SMALL:
                    # FI's own result files: a traceback in them names the quest's file, not the copy's.
                    dst.write_text(self.to_quest(src.read_text(encoding="utf-8", errors="replace")), encoding="utf-8")
                else:
                    shutil.copy2(src, dst)
                back.append(rel)
            except OSError:
                self.left_out.append(rel)
        self.brought_back = back
        return back

    def remove(self) -> None:
        if self.root is not None:
            for rel in self.linked:  # the link first: the quest's own folder behind it is never touched
                _unlink_folder(self.root / rel)
            shutil.rmtree(self.root, ignore_errors=True)


def _exists(path: Path) -> bool:
    """``path.exists()``, False for a path the user may not look at (a value such as ``PSModulePath`` naming ``/root``)."""
    try:
        return path.exists()
    except OSError:
        return False


def named_paths(quest_root: Path, values: list[str]) -> list[str]:
    """The paths into the quest folder that ``values`` (command-line arguments, environment values) name, relative to
    it: an absolute path under the quest folder, or a relative one that is there (or whose folder is there)."""
    root = Path(quest_root).resolve()
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 400 or "\n" in value or value.startswith("-"):
            continue
        for part in value.split(os.pathsep) if os.pathsep in value and not _exists(Path(value)) else [value]:
            try:
                path = Path(part)
                full = path.resolve() if path.is_absolute() else (root / path).resolve()
                if not full.is_relative_to(root) or full == root:
                    continue
                rel = full.relative_to(root).as_posix()
                if path.is_absolute() or full.exists() or (("/" in part or "\\" in part) and full.parent.is_dir()):
                    if not _never(rel):
                        out.append(rel)
            except (OSError, ValueError):
                continue
    return out


class OutsideExecutor:
    """An executor whose runs from the quest folder run from an :class:`OutsideCopy` instead; everything else is the
    wrapped executor's. The interpreter is the quest's own (its ``.venv/`` is not copied); absolute paths into the quest
    on the command line and in the environment are written as the copy's."""

    def __init__(self, inner: Any, quest_root: Path | str, *, base: Path | str | None = None,
                 inputs: tuple[str, ...] = CELL_INPUTS, skip: tuple[str, ...] = CELL_SKIP, log: Any = None,
                 share: bool = True, only_named: bool = False) -> None:
        self.inner = inner
        # ``share`` False: the run never sees the quest's shared folder (it works out every value itself); ``only_named``:
        # only the paths FI named for it come back. Both for a check whose run must neither read what the study kept nor
        # leave anything in the quest (core/second_impl.py changes a function's result to see what depends on it).
        self._share = share
        self._only_named = only_named
        self.quest_root = Path(quest_root).resolve()
        self._base = base
        self._inputs = inputs
        self._skip = skip
        # Without a log of its own, the quest's run log (core/engine.py names it frontier_insight.<quest id>).
        self._log = log if log is not None else logging.getLogger(f"frontier_insight.{self.quest_root.name}")
        self.last: OutsideCopy | None = None
        # The runs of this executor now running: each one's flag says whether another ran beside it at any time (then
        # what it left in the shared folder cannot be told from what the other wrote).
        self._running: dict[int, bool] = {}
        self._said_not_shared = False
        # A container sees only the folder mounted into it, so a link from the copy to the quest's own folder would not
        # resolve there: each setting then works out its shared values in its own copy.
        from .execution import DockerExecutor

        self._links = not isinstance(inner, DockerExecutor)
        #: The shared folder's files when this executor's first run started, and whether a stopped run's leftovers were
        #: kept because another ran beside it (removed by :meth:`settle` once none runs).
        self._shared_start: set[str] | None = None
        self._kept_beside = False
        #: Told the largest memory (MB) of every run that finished and had one measured (``SettingsGate.note``).
        self.meter: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _map(self, value: str, root: Path) -> str:
        venv = self.quest_root / ".venv"
        for a, b in ((str(self.quest_root), str(root)), (self.quest_root.as_posix(), root.as_posix())):
            if value.startswith(a) and not value.startswith(str(venv)) and not value.startswith(venv.as_posix()):
                return b + value[len(a):]
        return value

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        if Path(cwd).resolve() != self.quest_root:
            return await self.inner.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        named = named_paths(self.quest_root, [*cmd[1:], *(env or {}).values()])
        shares = self._share and self._links and any(SHARED_CACHE == s or SHARED_CACHE.startswith(s + "/")
                                                     for s in self._skip)
        if shares:
            shares = await asyncio.to_thread(prepare_shared_cache, self.quest_root, self._log) is not None
        box = OutsideCopy(self.quest_root, named=named, base=self._base, inputs=self._inputs, skip=self._skip,
                          share=shares, only_named=self._only_named)
        root = await asyncio.to_thread(box.make)
        self.last = box
        if box.not_shared and self._share and not self._said_not_shared:
            self._said_not_shared = True
            self._log.warning("[run] the settings cannot share %s/ (FI could not link it into the copy the code runs "
                              "from), so each setting works out what is kept there again", SHARED_CACHE)
        token = id(box)
        for other in self._running:
            self._running[other] = True
        self._running[token] = bool(self._running)
        shared_states = await asyncio.to_thread(working_states, self.quest_root, _shared_names(self.quest_root)) if box.linked else {}
        shared_before = set(shared_states)
        start_states = await asyncio.to_thread(working_states, self.quest_root)
        run_token = begin_run(self.quest_root)
        if box.linked and self._shared_start is None:
            self._shared_start = set(shared_before)
        try:
            argv = [cmd[0], *(self._map(a, root) for a in cmd[1:])]
            run_env = {k: self._map(v, root) for k, v in env.items()} if env is not None else None
            result = await self.inner.execute(argv, cwd=root, timeout_s=timeout_s, env=run_env)
            stopped = bool(getattr(result, "timed_out", False))
            if self.meter is not None:
                self.meter(getattr(result, "peak_mb", None))
            await asyncio.to_thread(box.bring_back, stopped=stopped)
            if box.unfinished:
                say_removed(self._log, box.unfinished)
            if stopped and box.linked:
                await self._clean_shared(shared_before, beside=self._running.get(token, False), states=shared_states)
            elif not stopped and getattr(result, "returncode", 0) in (0, None):
                await asyncio.to_thread(lambda: record_finished(self.quest_root, start_states, run_overlapped(self.quest_root, run_token)))
            dropped = [r for r in box.left_out if not r.endswith(".pyc") and "/__pycache__/" not in f"/{r}"]
            if dropped and not self._only_named:  # a check's run keeps only what FI named, by design
                self._log.warning("[run] the code ran from a copy outside the quest folder and wrote into FI's own "
                                  "folders (%s), which is not kept: %s", ", ".join(f"{d}/" for d in FI_OWNED),
                                  ", ".join(dropped))
            if dataclasses.is_dataclass(result):
                return dataclasses.replace(result, stdout=box.to_quest(result.stdout or ""),
                                           stderr=box.to_quest(result.stderr or ""))
            return result
        finally:
            end_run(self.quest_root, run_token)
            self._running.pop(token, None)
            await asyncio.to_thread(box.remove)

    async def _clean_shared(self, before: set[str], *, beside: bool, states: dict[str, list[int]] | None = None) -> None:
        """After a run FI stopped for its time: the files that appeared in the shared folder while it ran. Alone, they are
        its own: the locks and temporary files among them (:func:`unfinished_shared`) are removed, so the next setting does
        not wait on them, and a value it finished is kept. With another run beside it, FI cannot tell which run wrote
        which, so they are kept and said."""
        # Only what it left unfinished (a lock, a temporary file): a value it finished is kept (unfinished_shared).
        now = await asyncio.to_thread(_shared_names, self.quest_root)
        created = unfinished_shared(now - before)
        if states is not None and not beside:
            # A value that was there before and was written over while this run ran, and is not what a finished setting
            # wrote (code that saves under the final name was cut off part way): moved aside, not read as a result.
            rewritten = await asyncio.to_thread(
                lambda: _not_a_finished_state(self.quest_root, _rewritten(self.quest_root, states)))
            if rewritten:
                left_over: list[str] = []
                moved = await asyncio.to_thread(_move_aside, self.quest_root, sorted(rewritten), left_over)
                say_removed(self._log, moved, left_over)
        if not created:
            return
        if beside:
            shown = ", ".join(created[:5]) + (f" and {len(created) - 5} more" if len(created) > 5 else "")
            self._log.warning("[execute] a setting stopped for its time while others ran beside it; %d file%s appeared in "
                              "%s/ meanwhile (%s) and are kept, as FI cannot tell which setting wrote them", len(created),
                              "" if len(created) == 1 else "s", SHARED_CACHE, shown)
            self._kept_beside = True
            return
        left: list[str] = []
        gone = await asyncio.to_thread(_remove_files, self.quest_root, created, left)
        say_removed(self._log, gone, left)

    async def settle(self) -> None:
        """Once none of this executor's runs is running: the locks and temporary files that appeared in the shared folder
        since its first run started, when a run stopped for its time beside others left them (:meth:`_clean_shared` could
        not tell whose they were then). No one is left to finish them, and a later run would wait on them."""
        if not self._kept_beside or self._running or self._shared_start is None:
            return
        self._kept_beside = False
        created = unfinished_shared(await asyncio.to_thread(_shared_names, self.quest_root) - self._shared_start)
        if created:
            left: list[str] = []
            gone = await asyncio.to_thread(_remove_files, self.quest_root, created, left)
            say_removed(self._log, gone, left)
