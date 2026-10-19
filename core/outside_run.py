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
import shutil
import tempfile
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


def remove_created_since(quest_root: Path | str, before: set[str], left: list[str] | None = None) -> list[str]:
    """Remove the files under the quest's working folders that were not in ``before``; the paths removed. For a trial
    FI stopped for its time: it did not finish, so what it was writing (a lock, a half-written cache entry) is not
    something the next trial may trust. A file that cannot be removed (on Windows a handle of the stopped process can
    outlive it for a moment) is tried once more, then added to ``left``."""
    import time

    root = Path(quest_root)
    gone: list[str] = []
    created = working_files(root) - before
    # A finished value in the shared folder stays (unfinished_shared): only its locks and temporary files go.
    shared = {r for r in created if r == SHARED_CACHE or r.startswith(SHARED_CACHE + "/")}
    for rel in sorted((created - shared) | set(unfinished_shared(shared))):
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


def unfinished_shared(rels: list[str] | set[str]) -> list[str]:
    """Of these paths, the ones in :data:`SHARED_CACHE` that a stopped run left unfinished. The code writes every file
    there whole, to a temporary name it then renames, and marks a value being worked out with ``<name>.lock`` (the code
    writer is told so): a file under its own name is a finished value and is kept for the next setting; a lock or a
    temporary file (a name with ``.lock`` at its end, or ``.tmp`` in it) is not. An exception by name, stated: FI cannot
    tell a lock from a value otherwise, and a value it removes is worked out again by every later setting."""
    out = []
    for rel in sorted(rels):
        if not (rel == SHARED_CACHE or rel.startswith(SHARED_CACHE + "/")):
            continue
        name = rel.rsplit("/", 1)[-1]
        if name.endswith(".lock") or ".tmp" in name:
            out.append(rel)
    return out


def _remove_files(quest_root: Path | str, rels: list[str], left: list[str] | None = None) -> list[str]:
    """Remove these files of the quest (each tried twice, as in :func:`remove_created_since`); the ones removed."""
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


def say_removed(log: Any, gone: list[str], left: list[str] | None = None) -> None:
    if gone:
        shown = ", ".join(gone[:5]) + (f" and {len(gone) - 5} more" if len(gone) > 5 else "")
        log.warning("[execute] a trial stopped for its time left %d file%s it was writing (%s); removed, so the next "
                    "trial does not wait on %s", len(gone), "" if len(gone) == 1 else "s", shown,
                    "it" if len(gone) == 1 else "them")
    if left:
        shown = ", ".join(left[:5]) + (f" and {len(left) - 5} more" if len(left) > 5 else "")
        log.warning("[execute] a trial stopped for its time left %d file%s FI could not remove (%s); a later trial that "
                    "waits on %s may not finish", len(left), "" if len(left) == 1 else "s", shown,
                    "it" if len(left) == 1 else "them")


async def execute_keeping_only_finished(executor: Any, cmd: list[str], *, cwd: Path, timeout_s: int,
                                        env: dict[str, str] | None = None, log: Any = None) -> Any:
    """``executor.execute(...)`` for a trial that runs in the quest folder itself: when FI stopped it for its time, the
    files it created are removed (see :func:`remove_created_since`); a trial that finished keeps what it wrote."""
    root = Path(cwd)
    await asyncio.to_thread(prepare_shared_cache, root, log)
    before = await asyncio.to_thread(working_files, root)
    result = await executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
    if getattr(result, "timed_out", False):
        left: list[str] = []
        gone = await asyncio.to_thread(remove_created_since, root, before, left)
        say_removed(log if log is not None else logging.getLogger(f"frontier_insight.{root.resolve().name}"), gone, left)
    return result


def code_version(quest_root: Path | str) -> str:
    """A hash of the simulation's code as it is now: ``code/simulate.py`` and the model package beside it, the code each
    setting runs and the code that decides a set of trials (``core.trial_runner._run_key``). The analysis script, the
    tests and the notes in ``code/`` work out nothing kept in :data:`SHARED_CACHE`: when a real run's analysis script was
    repaired after every setting had run, the values its unchanged simulation had kept were moved aside and worked out
    again."""
    from . import code_layout

    h = hashlib.sha256()
    code = Path(quest_root) / CODE
    with contextlib.suppress(OSError):
        h.update(b"simulate.py\0" + (code / "simulate.py").read_bytes() + b"\0")
    if code.is_dir():
        for rel, body in sorted(code_layout.package_sources(code).items()):
            h.update(rel.encode("utf-8") + b"\0" + body.encode("utf-8") + b"\0")
    return h.hexdigest()


def prepare_shared_cache(quest_root: Path | str, log: Any = None) -> Path:
    """The quest's :data:`SHARED_CACHE` folder, made if missing, holding only what the current version of ``code/`` kept
    there: when FI's record names another version, the folder is moved aside to ``.fi/previous/`` first (what is in it
    was worked out by other code). Returns the folder."""
    import json
    import time

    root = Path(quest_root)
    folder = root / SHARED_CACHE
    record = root / SHARED_CACHE_RECORD
    version = code_version(root)
    try:
        kept = json.loads(record.read_text(encoding="utf-8")).get("code")
    except (OSError, ValueError, AttributeError):
        kept = None
    if kept != version:
        if folder.is_dir() and any(folder.iterdir()):
            aside = root / ".fi" / "previous" / f"shared_cache_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(folder, aside)
                if log is not None:
                    log.info("[run] the code changed, so what the earlier code kept in %s/ was moved to %s (no value "
                             "worked out by other code is read)", SHARED_CACHE, aside.relative_to(root).as_posix())
            except OSError:
                pass  # another run moved it first
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
                 inputs: tuple[str, ...] = CELL_INPUTS, skip: tuple[str, ...] = CELL_SKIP) -> None:
        self.quest_root = Path(quest_root).resolve()
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
            if target.is_dir() and _link_folder(root / SHARED_CACHE, target):
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
        return rel.split("/", 1)[0] not in FI_OWNED

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
                 inputs: tuple[str, ...] = CELL_INPUTS, skip: tuple[str, ...] = CELL_SKIP, log: Any = None) -> None:
        self.inner = inner
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
        shares = any(SHARED_CACHE == s or SHARED_CACHE.startswith(s + "/") for s in self._skip)
        if shares:
            await asyncio.to_thread(prepare_shared_cache, self.quest_root, self._log)
        box = OutsideCopy(self.quest_root, named=named, base=self._base, inputs=self._inputs, skip=self._skip)
        root = await asyncio.to_thread(box.make)
        self.last = box
        if box.not_shared and not self._said_not_shared:
            self._said_not_shared = True
            self._log.warning("[run] the settings cannot share %s/ (FI could not link it into the copy the code runs "
                              "from), so each setting works out what is kept there again", SHARED_CACHE)
        token = id(box)
        for other in self._running:
            self._running[other] = True
        self._running[token] = bool(self._running)
        shared_before = await asyncio.to_thread(_shared_names, self.quest_root) if box.linked else set()
        try:
            argv = [cmd[0], *(self._map(a, root) for a in cmd[1:])]
            run_env = {k: self._map(v, root) for k, v in env.items()} if env is not None else None
            result = await self.inner.execute(argv, cwd=root, timeout_s=timeout_s, env=run_env)
            stopped = bool(getattr(result, "timed_out", False))
            await asyncio.to_thread(box.bring_back, stopped=stopped)
            if box.unfinished:
                say_removed(self._log, box.unfinished)
            if stopped and box.linked:
                await self._clean_shared(shared_before, beside=self._running.get(token, False))
            dropped = [r for r in box.left_out if not r.endswith(".pyc") and "/__pycache__/" not in f"/{r}"]
            if dropped:
                self._log.warning("[run] the code ran from a copy outside the quest folder and wrote into FI's own "
                                  "folders (%s), which is not kept: %s", ", ".join(f"{d}/" for d in FI_OWNED),
                                  ", ".join(dropped))
            if dataclasses.is_dataclass(result):
                return dataclasses.replace(result, stdout=box.to_quest(result.stdout or ""),
                                           stderr=box.to_quest(result.stderr or ""))
            return result
        finally:
            self._running.pop(token, None)
            await asyncio.to_thread(box.remove)

    async def _clean_shared(self, before: set[str], *, beside: bool) -> None:
        """After a run FI stopped for its time: the files that appeared in the shared folder while it ran. Alone, they are
        its own: the locks and temporary files among them (:func:`unfinished_shared`) are removed, so the next setting does
        not wait on them, and a value it finished is kept. With another run beside it, FI cannot tell which run wrote
        which, so they are kept and said."""
        # Only what it left unfinished (a lock, a temporary file): a value it finished is kept (unfinished_shared).
        created = unfinished_shared(await asyncio.to_thread(_shared_names, self.quest_root) - before)
        if not created:
            return
        if beside:
            shown = ", ".join(created[:5]) + (f" and {len(created) - 5} more" if len(created) > 5 else "")
            self._log.warning("[execute] a setting stopped for its time while others ran beside it; %d file%s appeared in "
                              "%s/ meanwhile (%s) and are kept, as FI cannot tell which setting wrote them", len(created),
                              "" if len(created) == 1 else "s", SHARED_CACHE, shown)
            return
        left: list[str] = []
        gone = await asyncio.to_thread(_remove_files, self.quest_root, created, left)
        say_removed(self._log, gone, left)
