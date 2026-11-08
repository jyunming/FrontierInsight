"""A compared setting that no trial of the whole run asked for is a fact FI states from its own record of what the code
read (core.in_run.Cell), however much seed noise moves the numbers (core.trial_runner.unread_axes)."""
from __future__ import annotations

from core import flat_output as flat
from core import trial_runner as T


def _run(cells: list[tuple[dict, list[list[str]], bool]], *, status: str = "ok") -> list[T.CellRun]:
    """``(cell, [names read in each trial], took the whole cell)``; every trial returns a different number (seed noise)."""
    out = []
    for i, (cell, reads, whole) in enumerate(cells):
        rows = [{"status": status, "values": {"y": 0.1 * i + 0.01 * t}, "read": r, "unknown_reads": [], "read_all": whole}
                for t, r in enumerate(reads)]
        out.append(T.CellRun(key=str(i), cell=cell, planned=len(rows), rows=rows))
    return out


def _grid(read: list[str], whole: bool = False) -> list[T.CellRun]:
    cells = [{"a": a, "b": b} for a in (1, 2) for b in (10, 20)]
    return _run([(c, [read, read], whole) for c in cells])


def test_a_varied_setting_no_trial_read_is_reported_even_when_the_numbers_move() -> None:
    assert T.unread_axes(_grid(["a"])) == ["b"]


def test_a_setting_that_is_read_is_not_reported_whatever_its_effect() -> None:
    assert T.unread_axes(_grid(["a", "b"])) == []


def test_no_claim_when_the_code_took_the_cell_as_a_whole() -> None:
    assert T.unread_axes(_grid(["a"], whole=True)) == []


def test_some_failed_trials_do_not_silence_an_ignored_setting() -> None:
    runs = _grid(["a"])
    for c in runs[:3]:
        c.rows[0]["status"] = "failed"
    assert T.unread_axes(runs) == ["b"]


def test_a_read_recorded_by_a_failed_trial_counts() -> None:
    runs = _grid(["a"])
    runs[0].rows[0].update(status="failed", read=["a", "b"])
    assert T.unread_axes(runs) == []


def test_a_value_with_no_finished_trial_makes_no_claim_for_that_setting_alone() -> None:
    runs = _grid(["a"])
    for c in runs:
        if c.cell["b"] == 20:
            for r in c.rows:
                r["status"] = "failed"
    assert T.unread_axes(runs) == []
    runs = _grid(["a"])
    for c in runs:
        if c.cell["a"] == 2:
            for r in c.rows:
                r["status"] = "failed"
    assert T.unread_axes(runs) == ["b"]


def test_a_numerical_setting_left_out_by_the_caller_is_not_listed() -> None:
    assert T.unread_axes(_grid(["a"]), skip={"b"}) == []


def test_the_note_says_the_code_never_read_the_setting_and_keeps_the_no_effect_advice() -> None:
    note = flat.unchanged_axes_note([], unread=["b"])
    assert "never read `b`" in note or "never read b" in note.replace("`", "")
    assert "cannot show an effect" in note
    both = flat.unchanged_axes_note([("a", 3)], unread=["b"])
    assert "never read" in both and "changed no number" in both
    assert flat.unchanged_axes_note([]) == ""
