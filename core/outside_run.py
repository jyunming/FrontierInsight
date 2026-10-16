"""FI runs the quest's code from a copy outside the quest folder, so FI's own records are not next to it.

Code a model or a coding agent wrote runs on this computer as the person (there is no sandbox on the normal path). Run
from the quest folder, it could open ``needs/FROZEN_PROTOCOL.json`` (the checks' expected values and tolerances),
``.fi/checks/`` or ``.fi/state.sqlite`` by a plain relative path. :class:`OutsideCopy` makes a scratch folder under the
system's temporary folder, named by a random id (never by the quest's name), with:

* a copy of ``code/`` (no ``.git`` history, no ``.venv``, no caches, no value file an older FI left in ``code/tests/``),
* the output folders the experiment reads back and adds to (:data:`OUTPUT_DIRS`: ``raw/``, ``figures/``, ``results/``,
  ``data/``),
* the files and folders FI itself names for this one run, on the command line or in the environment (its harness, the
  spec of a setting, the file a result goes to);

and never ``needs/``, ``.fi/checks/`` or the rest of ``.fi/``. After the run FI copies back, by its own list, what the
run wrote under those output folders and those named paths, and nothing else (a file the run wrote anywhere else in the
copy, or into ``code/``, is dropped with the copy). Paths that name the copy are written back as the quest's in what
the run printed and in FI's own result files, so a traceback still points at ``code/...``.

This is not a sandbox. Code that searches the disk by absolute path, or starts from its own interpreter (the quest's
``.venv/`` stays in the quest folder), can still find the quest; the copy only stops reads and writes by relative path.
:class:`OutsideExecutor` wraps an executor so a run whose working folder is the quest folder runs from such a copy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

#: The folder under the system's temporary folder that holds every copy (one sub-folder per run, a random id).
SCRATCH_DIR = "fi_run"
CODE = "code"
#: Folders of the quest the experiment reads back and adds to: copied in, and what the run wrote there copied back.
OUTPUT_DIRS = ("raw", "figures", "results", "data")
#: Never in a copy and never written back, even when named.
NEVER = ("needs", ".fi/checks", ".venv", ".git", ".fi/state.sqlite", ".fi/agent_coder.json")
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
                 outputs: tuple[str, ...] = OUTPUT_DIRS) -> None:
        self.quest_root = Path(quest_root).resolve()
        self.named = sorted({n.strip("/") for n in named if n and not _never(n.strip("/"))})
        self.outputs = tuple(outputs)
        self._base = base
        self.root: Path | None = None
        self._before: dict[str, Any] = {}
        self.brought_back: list[str] = []
        self.left_out: list[str] = []

    def make(self) -> Path:
        root = self.root = new_scratch(self._base)
        copy_code(self.quest_root / CODE, root / CODE)
        for rel in (*self.outputs, *self.named):
            src = self.quest_root / rel
            if rel == CODE or rel.startswith(CODE + "/") or _never(rel):
                continue
            if src.is_dir():
                shutil.copytree(src, root / rel, ignore=shutil.ignore_patterns(*_SKIP_DIRS, "*.pyc"), dirs_exist_ok=True)
            elif src.is_file():
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, root / rel)
            elif "/" in rel:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
        self._before = _files(root)
        return root

    def _comes_back(self, rel: str) -> bool:
        if _never(rel) or rel == CODE or rel.startswith(CODE + "/"):
            return False
        if rel.split("/", 1)[0] in self.outputs:
            return True
        return any(rel == n or rel.startswith(n + "/") for n in self.named)

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
        """Copy into the quest what the run wrote (created, changed, removed) under the output folders and the named
        paths; anything else it wrote is left out (and listed in :attr:`left_out`). The paths written back."""
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


def named_paths(quest_root: Path, values: list[str]) -> list[str]:
    """The paths into the quest folder that ``values`` (command-line arguments, environment values) name, relative to
    it: an absolute path under the quest folder, or a relative one that is there (or whose folder is there)."""
    root = Path(quest_root).resolve()
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 400 or "\n" in value or value.startswith("-"):
            continue
        for part in value.split(os.pathsep) if os.pathsep in value and not Path(value).exists() else [value]:
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
                 outputs: tuple[str, ...] = OUTPUT_DIRS, log: Any = None) -> None:
        self.inner = inner
        self.quest_root = Path(quest_root).resolve()
        self._base = base
        self._outputs = outputs
        self._log = log
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
        box = OutsideCopy(self.quest_root, named=named, base=self._base, outputs=self._outputs)
        root = await asyncio.to_thread(box.make)
        self.last = box
        try:
            argv = [cmd[0], *(self._map(a, root) for a in cmd[1:])]
            run_env = {k: self._map(v, root) for k, v in env.items()} if env is not None else None
            result = await self.inner.execute(argv, cwd=root, timeout_s=timeout_s, env=run_env)
            await asyncio.to_thread(box.bring_back)
            dropped = [r for r in box.left_out if not r.endswith(".pyc") and "/__pycache__/" not in f"/{r}"]
            if dropped and self._log is not None:
                self._log.warning("[run] the code ran from a copy outside the quest folder and wrote %s there, outside "
                                  "the folders FI keeps (%s); not kept", ", ".join(dropped[:6])
                                  + (f" and {len(dropped) - 6} more" if len(dropped) > 6 else ""),
                                  ", ".join(f"{d}/" for d in self._outputs))
            if dataclasses.is_dataclass(result):
                return dataclasses.replace(result, stdout=box.to_quest(result.stdout or ""),
                                           stderr=box.to_quest(result.stderr or ""))
            return result
        finally:
            await asyncio.to_thread(box.remove)
