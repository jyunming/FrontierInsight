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

This is not a sandbox. Code that searches the disk by absolute path, or starts from its own interpreter (the quest's
``.venv/`` stays in the quest folder), can still find the quest; the copy only stops reads and writes by relative path.
:class:`OutsideExecutor` wraps an executor so a run whose working folder is the quest folder runs from such a copy.
"""

from __future__ import annotations

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


def _files(folder: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not folder.is_dir():
        return out
    for here, names, leaves in os.walk(folder):
        names[:] = [n for n in names if n not in _SKIP_DIRS]
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
        self._before = _files(root)
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

    def bring_back(self) -> list[str]:
        """Copy into the quest what the run wrote (created, changed, removed), except into FI's own folders
        (:data:`FI_OWNED`, other than the named paths): those are left out and listed in :attr:`left_out`. The paths
        written back."""
        assert self.root is not None
        now = _files(self.root)
        back: list[str] = []
        for rel in sorted(set(now) | set(self._before)):
            if now.get(rel) == self._before.get(rel):
                continue
            if not self._comes_back(rel):
                self.left_out.append(rel)
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _map(self, value: str, root: Path) -> str:
        venv = self.quest_root / ".venv"
        for a, b in ((str(self.quest_root), str(root)), (self.quest_root.as_posix(), root.as_posix())):
            if value.startswith(a) and not value.startswith(str(venv)) and not value.startswith(venv.as_posix()):
                return b + value[len(a):]
        return value

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        import asyncio

        if Path(cwd).resolve() != self.quest_root:
            return await self.inner.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
        named = named_paths(self.quest_root, [*cmd[1:], *(env or {}).values()])
        box = OutsideCopy(self.quest_root, named=named, base=self._base, inputs=self._inputs, skip=self._skip)
        root = await asyncio.to_thread(box.make)
        self.last = box
        try:
            argv = [cmd[0], *(self._map(a, root) for a in cmd[1:])]
            run_env = {k: self._map(v, root) for k, v in env.items()} if env is not None else None
            result = await self.inner.execute(argv, cwd=root, timeout_s=timeout_s, env=run_env)
            await asyncio.to_thread(box.bring_back)
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
            await asyncio.to_thread(box.remove)
