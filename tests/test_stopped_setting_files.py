"""A setting FI stopped for its time may have left a file half-written under its final name.

Code that saves straight under the final name (no temporary name renamed afterwards) and is cut off part way leaves a
truncated file. When that file was already there (an earlier finished setting wrote it, or it is shared), a list of
"what is new" does not see it. FI keeps a record of the files each finished setting wrote
(``core.outside_run.record_finished``); a file written during the stopped setting whose state is not in that record is
unfinished and is moved aside.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from core import outside_run
from core.execution import ExecutionResult


class _Run:
    """An executor whose run does ``action(cwd)`` and then reports a clean end or a stop for its time."""

    def __init__(self, action, timed_out: bool = False) -> None:  # noqa: ANN001
        self.action = action
        self.timed_out = timed_out

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        self.action(Path(cwd))
        return ExecutionResult(returncode=-9 if self.timed_out else 0, stdout="", stderr="", duration_s=1.0,
                               timed_out=self.timed_out)


def _quest(base: Path) -> Path:
    root = base / "quest"
    (root / "data").mkdir(parents=True)
    return root


def _go(root: Path, action, *, timed_out: bool):  # noqa: ANN001, ANN202
    return asyncio.run(outside_run.execute_keeping_only_finished(_Run(action, timed_out), ["python", "x.py"], cwd=root,
                                                                 timeout_s=5))


def _finished_setting_writes_whole_table(root: Path) -> None:
    _go(root, lambda cwd: (cwd / "data" / "table.bin").write_bytes(b"0123456789" * 100), timed_out=False)
    assert (root / "data" / "table.bin").stat().st_size == 1000


def test_a_shared_file_cut_off_by_a_stopped_setting_is_not_kept_under_its_name(tmp_path: Path) -> None:
    """Old code: a file that existed before the stopped setting began was never looked at, so the truncated table stayed
    under its final name and a later setting read it as a result."""
    root = _quest(tmp_path)
    _finished_setting_writes_whole_table(root)
    _go(root, lambda cwd: (cwd / "data" / "table.bin").write_bytes(b"0123"), timed_out=True)
    assert not (root / "data" / "table.bin").exists()
    aside = list((root / ".fi" / "previous").rglob("table.bin"))
    assert len(aside) == 1 and aside[0].read_bytes() == b"0123", "moved aside, not deleted"


def test_a_file_a_finished_setting_wrote_and_the_stopped_one_did_not_touch_is_kept(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    _finished_setting_writes_whole_table(root)
    _go(root, lambda cwd: (cwd / "data" / "partial.tmp").write_text("x"), timed_out=True)
    assert (root / "data" / "table.bin").read_bytes() == b"0123456789" * 100
    assert not (root / "data" / "partial.tmp").exists()


def test_a_file_created_only_during_the_stopped_setting_is_removed(tmp_path: Path) -> None:
    """The earlier behaviour stays: a file the stopped setting created is not a result."""
    root = _quest(tmp_path)
    (root / "data" / "inputs.csv").write_text("a\n1\n")
    _go(root, lambda cwd: (cwd / "data" / "made.bin").write_bytes(b"abc"), timed_out=True)
    assert not (root / "data" / "made.bin").exists()
    assert (root / "data" / "inputs.csv").exists(), "what the quest had before, untouched, is kept"


def test_a_file_the_quest_had_before_and_the_stopped_setting_rewrote_is_moved_aside_too(tmp_path: Path) -> None:
    """No finished setting recorded this file, so a rewrite during a stopped setting cannot be told from a cut-off save."""
    root = _quest(tmp_path)
    (root / "data" / "inputs.csv").write_text("a\n1\n2\n3\n")
    _go(root, lambda cwd: (cwd / "data" / "inputs.csv").write_text("a\n"), timed_out=True)
    assert not (root / "data" / "inputs.csv").exists()
    assert list((root / ".fi" / "previous").rglob("inputs.csv"))


def test_a_file_rewritten_to_the_state_a_finished_setting_recorded_is_kept(tmp_path: Path) -> None:
    """Known positive for the record: a stopped setting that wrote nothing new over a finished setting's file (here it
    only rewrote the same bytes after the finished one recorded them) is judged by the file's state, and an untouched
    state matches."""
    root = _quest(tmp_path)
    _finished_setting_writes_whole_table(root)
    record = outside_run.finished_files(root)
    assert "data/table.bin" in record and record["data/table.bin"][0] == 1000
    gone = outside_run.remove_created_since(root, outside_run.working_files(root), states=outside_run.working_states(root))
    assert gone == [] and (root / "data" / "table.bin").exists()


def test_a_half_written_file_is_not_recorded_as_finished_by_a_setting_that_ran_beside_the_stopped_one(tmp_path: Path) -> None:
    """Old code: ``record_finished`` scanned the whole quest, so setting A finishing while setting B was half-way
    through writing ``data/value.bin`` recorded B's truncated file as a finished one, and B's cleanup then kept it."""
    root = _quest(tmp_path)
    (root / "data" / "value.bin").write_bytes(b"x" * 1000)

    async def scenario() -> None:
        b_wrote, a_done = asyncio.Event(), asyncio.Event()

        class _A:
            async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
                await b_wrote.wait()
                a_done.set()
                return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=1.0)

        class _B:
            async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
                (Path(cwd) / "data" / "value.bin").write_bytes(b"x" * 10)  # cut off under the final name
                b_wrote.set()
                await a_done.wait()
                return ExecutionResult(returncode=-9, stdout="", stderr="", duration_s=1.0, timed_out=True)

        await asyncio.gather(
            outside_run.execute_keeping_only_finished(_A(), ["python", "a.py"], cwd=root, timeout_s=5),
            outside_run.execute_keeping_only_finished(_B(), ["python", "b.py"], cwd=root, timeout_s=5))

    asyncio.run(scenario())
    assert "data/value.bin" not in outside_run.finished_files(root)
    assert not (root / "data" / "value.bin").exists(), "the cut-off file is moved aside"
    assert list((root / ".fi" / "previous").rglob("value.bin"))


def test_a_setting_that_begins_after_anothers_process_ended_but_before_its_files_are_recorded_overlaps_it(tmp_path: Path) -> None:
    """Old code: a run left the registry the moment its process ended, so a setting that began and half-wrote
    ``data/value.bin`` before the first one's files were recorded was not counted as beside it, and the half-written
    file was recorded as finished."""
    root = _quest(tmp_path)
    (root / "data" / "value.bin").write_bytes(b"x" * 1000)

    class _Result:
        returncode = 0

        @property
        def timed_out(self) -> bool:
            # Read by FI right after A's process ended, before it records A's files: B begins and writes half a file.
            self.b_token = outside_run.begin_run(root)
            (root / "data" / "value.bin").write_bytes(b"x" * 10)
            return False

    class _A:
        async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
            return _Result()

    asyncio.run(outside_run.execute_keeping_only_finished(_A(), ["python", "a.py"], cwd=root, timeout_s=5))
    assert "data/value.bin" not in outside_run.finished_files(root)
