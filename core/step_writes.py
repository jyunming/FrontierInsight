"""Which step of the study wrote each result file: FI's own record, made where FI starts the steps.

FI runs a study in steps it starts itself: the simulation (or the search) and then the analysis. Around each run of them
(``core/split_run.py``, ``core/trial_runner.py``, ``core/optimise.py``) FI looks at the result files (``data/results/*.json``)
before the simulation, between the two steps and after the analysis, and keeps in ``.fi/step_writes.json`` which step
wrote or rewrote each file: ``{file name: step}``. A rerun of the study (``core/rerun_check.py``) reads it to tell which
delivered result files a changed step can affect. A file this record has no writer for is not known to be safe.

When two runs overlap in one quest FI cannot tell which of them wrote a changed file, so nothing is recorded (as for
``core/outside_run.py::record_finished``)."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import outside_run as _outside

RECORD = ".fi/step_writes.json"
RESULTS_DIR = "data/results"
SIMULATION = "simulation"
ANALYSIS = "analysis"


def snapshot(quest_root: Path | str) -> dict[str, tuple[int, int]]:
    """``{file name: (size, modification time ns)}`` of the result files (the ``*.json`` of the results folder)."""
    out: dict[str, tuple[int, int]] = {}
    folder = Path(quest_root) / RESULTS_DIR
    try:
        for p in folder.glob("*.json"):
            st = p.stat()
            out[p.name] = (st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    return out


def read(quest_root: Path | str) -> dict[str, str]:
    """FI's record: ``{file name: step that wrote it}``; empty when FI made none."""
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)} if isinstance(data, dict) else {}


class Watch:
    """The result files seen at each step boundary of one run."""

    def __init__(self, quest_root: Path | str) -> None:
        self.root = Path(quest_root)
        self._last = snapshot(self.root)
        self.wrote: dict[str, str] = {}
        self.done = False

    def mark(self, step: str) -> None:
        """The files new or changed since the last mark were written by ``step``. After the analysis the run is complete."""
        now = snapshot(self.root)
        for name, state in now.items():
            if self._last.get(name) != state:
                self.wrote[name] = step
        self._last = now
        if step == ANALYSIS:
            self.done = True


@contextmanager
def watching(quest_root: Path | str) -> Iterator[Watch]:
    """Watch one run of the steps; when it ran to the analysis and no other run overlapped it, keep what it wrote."""
    token = _outside.begin_run(quest_root)
    watch = Watch(quest_root)
    try:
        yield watch
    finally:
        overlapped = _outside.end_run(quest_root, token)
        if watch.done and not overlapped and watch.wrote:
            _save(watch.root, watch.wrote)


def _save(root: Path, wrote: dict[str, str]) -> None:
    record = read(root)
    record.update(wrote)
    path = root / RECORD
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
