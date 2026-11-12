"""Which functions of the study's code were really called in FI's checking runs: FI's record of calls.

Reading the code can say that nothing names a function, but not that a function ran: a name chosen at run time, a function
nested in another, a method handed to a library. So in FI's own checking runs (the small run, a check's run of the
simulation, the analysis gate's small copy and a finding's failing test) the program is started through this file, which
notes the first call of each function the study's ``code/`` folder defines: its file, its qualified name (``Class.method``,
``outer.<locals>.inner``) and the function that called it. It is never on in the study's own trials: noting every call
costs between a half and two times the run's time.

What the record can say is only "this function ran". A function missing from it may sit in a setting the small run did not
take, so a missing function is "cannot tell", never "never run" (:mod:`core.code_reach`).

This file is standard library only and is copied as text to ``.fi/call_record.py`` next to the harness (like
:mod:`core.in_run`); the checking run's command line is ``python .fi/call_record.py <result file> <program> [arguments]``
(or ``-c <source>`` for a program given as text). The program's code folder is the ``code/`` beside ``.fi/``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

#: Where FI keeps what the checking runs of a quest have shown (one file per quest, the union of the runs).
RECORD = Path(".fi") / "calls.json"
#: Where the copy of this file that starts a checking run is written, and the folder a run writes its own file in.
RUNNER = Path(".fi") / "call_record.py"
RUN_DIR = Path(".fi") / "calls"
VERSION = 1


class Recorder:
    """Notes the first call of every function defined under ``root``; :meth:`write` merges what it saw into ``path``."""

    def __init__(self, root: str, path: str) -> None:
        self.root = os.path.normcase(os.path.realpath(root)) + os.sep
        self.path = path
        self.seen: set[Any] = set()
        self.calls: dict[tuple[str, str], list[Any]] = {}

    def _where(self, code: Any) -> tuple[str, str] | None:
        name = code.co_filename
        if not name or name.startswith("<"):
            return None
        full = os.path.normcase(os.path.realpath(name))
        if not full.startswith(self.root):
            return None
        return full[len(self.root):].replace(os.sep, "/"), getattr(code, "co_qualname", code.co_name)

    def _on(self, frame: Any, event: str, arg: Any) -> None:
        if event != "call":
            return
        code = frame.f_code
        if code in self.seen:
            return
        self.seen.add(code)
        try:
            here = self._where(code)
            if here is None or here in self.calls:
                return
            back = frame.f_back
            caller = self._where(back.f_code) if back is not None else None
            self.calls[here] = [here[0], here[1], code.co_firstlineno, caller[0] if caller else None,
                                caller[1] if caller else None]
        except Exception:  # noqa: BLE001 -- noting a call must never change the run
            return

    def start(self) -> None:
        sys.setprofile(self._on)
        threading.setprofile(self._on)

    def stop(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)
        self.write()

    def write(self) -> None:
        merge_into(self.path, list(self.calls.values()))


def merge_into(path: str | Path, calls: list[list[Any]]) -> None:
    """Add ``calls`` (``[file, qualname, line, caller file, caller qualname]``) to the record in ``path``: a function already
    there keeps its first entry. Written to a new file and put in place, so a reader never sees half a record."""
    have = _read(path)
    known = {(c[0], c[1]) for c in have}
    merged = have + [c for c in calls if (c[0], c[1]) not in known]
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": VERSION, "calls": merged}, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _read(path: str | Path) -> list[list[Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data.get("calls") if isinstance(data, dict) and data.get("version") == VERSION else None
    return [list(r) for r in rows if isinstance(r, list) and len(r) >= 2] if isinstance(rows, list) else []


def main(argv: list[str]) -> None:
    out, program = argv[1], argv[2:]
    here = os.path.dirname(os.path.abspath(__file__))
    rec = Recorder(os.path.join(os.path.dirname(here), "code"), out)
    text = None
    if program and program[0] == "-c":
        text, sys.argv = program[1], ["-c", *program[3:]] if len(program) > 2 else ["-c"]
        sys.path[0] = ""
    else:
        sys.argv = program
        sys.path[0] = os.path.dirname(os.path.abspath(program[0]))
    rec.start()
    try:
        if text is not None:
            exec(compile(text, "<string>", "exec"), {"__name__": "__main__", "__builtins__": __builtins__})
        else:
            import runpy

            runpy.run_path(program[0], run_name="__main__")
    finally:
        rec.stop()


# --- FI's side: reading and keeping the record -------------------------------------------------------------------------------


def load(quest_root: Path | str) -> set[tuple[str, str]] | None:
    """``{(file, qualified name)}`` of every function the checking runs of this quest saw called (``file`` relative to the
    quest's ``code/``), or ``None`` when no checking run has been recorded."""
    path = Path(quest_root) / RECORD
    if not path.is_file():
        return None
    return {(str(c[0]), str(c[1])) for c in _read(path)}


_LOCK = threading.Lock()


def absorb(quest_root: Path | str, run_file: Path | str) -> None:
    """Move one checking run's own file into the quest's record (and remove it). Nothing to move when the run wrote none."""
    run_file = Path(run_file)
    rows = _read(run_file)
    if rows:
        with _LOCK:
            merge_into(Path(quest_root) / RECORD, rows)
    run_file.unlink(missing_ok=True)


def prepare(quest_root: Path | str, tag: str) -> tuple[str, Path]:
    """Write the runner into ``quest_root`` and name the file one run writes: ``(the runner's path, the run's file)``, both
    relative to the quest (the form a command line gives). ``tag`` keeps two runs at the same time apart."""
    root = Path(quest_root)
    (root / RUN_DIR).mkdir(parents=True, exist_ok=True)
    (root / RUNNER).write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)[:60]
    return RUNNER.as_posix(), RUN_DIR / f"{safe}_{os.getpid()}_{threading.get_ident()}_{next(_COUNTER)}.json"


def _counter():
    n = 0
    while True:
        n += 1
        yield n


_COUNTER = _counter()


if __name__ == "__main__":
    main(sys.argv)
