"""A layout-only refine redraws the figures from the saved numbers. A real quest's redraw guessed an array key
(``hs_euler``) the archive did not have, failed, and left the figures as they were while the quest went on to write.
The model is now shown what each saved file holds, a failed script is written once more with its error, and when that
fails too the person is told at the review that their request was not applied."""

from __future__ import annotations

import asyncio
import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from core.data_shape import describe_data_file, describe_data_files
from tests.test_refine_extend import STATE, _engine

KEYS = ["h_grid", "euler_errors", "rk4_errors"]
WRONG = ("import numpy as np\nd = np.load('data/results/convergence_data.npz')\nx = d['hs_euler']\n"
         "open('figures/fig1.png', 'wb').write(b'new')\nprint('REPLOTTED: fig1.png')")
RIGHT = ("import numpy as np\nd = np.load('data/results/convergence_data.npz')\nx = d['h_grid']\n"
         "open('figures/fig1.png', 'wb').write(('new-%d' % len(x)).encode())\nprint('REPLOTTED: fig1.png')")


def _quest(tmp_path: Path) -> Any:
    eng = _engine(tmp_path)
    root = eng.quest_root
    (root / "code").mkdir(parents=True)
    (root / "data" / "results").mkdir(parents=True)
    (root / "figures").mkdir()
    (root / "code" / "experiment.py").write_text("hs = [0.1, 0.05]\n", encoding="utf-8")
    np.savez(root / "data" / "results" / "convergence_data.npz",
             h_grid=np.array([0.1, 0.05, 0.025, 0.0125]), euler_errors=np.ones(5), rk4_errors=np.ones(5))
    (root / "figures" / "fig1.png").write_bytes(b"old-1")
    (root / "figures" / "fig2.png").write_bytes(b"old-2")
    return eng


def _replot(eng: Any, scripts: list[str]) -> tuple[Any, list[str]]:
    """Run the layout step with a model that answers ``scripts`` in turn; each script is run for real."""
    prompts: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return f"```python\n{scripts[min(len(prompts), len(scripts)) - 1]}\n```"

    async def execute(cmd: list[str], **kw: Any) -> Any:
        proc = subprocess.run([sys.executable, cmd[-1]], cwd=str(kw["cwd"]), capture_output=True, text=True)
        return SimpleNamespace(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, duration_s=0.1,
                               timed_out=False)

    eng._chat = chat
    eng.executor = SimpleNamespace(python_path=lambda q: Path(sys.executable), execute=execute)
    out = asyncio.run(eng._node_replot_layout({**STATE, "refine_layout": ["make the axis labels larger"]}))
    return out, prompts


def test_a_redraw_that_guessed_a_key_is_written_again_and_the_figure_changes(tmp_path: Path) -> None:
    eng = _quest(tmp_path)
    out, prompts = _replot(eng, [WRONG, RIGHT])
    assert (eng.quest_root / "figures" / "fig1.png").read_bytes() == b"new-4", "the layout request was applied"
    assert (eng.quest_root / "figures" / "fig2.png").read_bytes() == b"old-2"
    assert out["layout_missed"] == [] and out["layout_not_redrawn"] == []
    assert len(prompts) == 2
    for key in KEYS:
        assert f"`{key}` (float64" in prompts[0], "the model is shown the real keys of the archive"
    assert "Your previous script failed" not in prompts[0]
    assert "Your previous script failed" in prompts[1] and "hs_euler" in prompts[1], "the retry sees the error"
    assert not (eng.fi_dir / "layout_backup").exists()


def test_a_redraw_that_fails_twice_tells_the_person_it_was_not_applied(tmp_path: Path) -> None:
    eng = _quest(tmp_path)
    out, prompts = _replot(eng, [WRONG, WRONG])
    assert len(prompts) == 2
    assert (eng.quest_root / "figures" / "fig1.png").read_bytes() == b"old-1"
    assert out["layout_missed"] == ["make the axis labels larger"]
    assert out["layout_not_redrawn"] == ["make the axis labels larger"]
    assert eng._route_after_replot({**STATE, **out}) == "write"
    card: dict[str, Any] = {}

    def pause(**kw: Any) -> Any:
        card.update(kw)
        return {"action": "accept", "answer": "yes"}

    eng._pause_for_human = pause
    after = asyncio.run(eng._node_human_feedback({**STATE, **out, "review": {"verdict": "accept", "score": 7}}))
    assert "NOT applied" in card["steps"][0] and "make the axis labels larger" in card["steps"][0]
    assert card["payload"]["human_review"]["layout_not_redrawn"] == ["make the axis labels larger"]
    assert after["layout_not_redrawn"] == [], "told once, then cleared"
    # A passing review is not accepted for the person while their figure request was not applied.
    from core.engine import _auto_accepts
    snap = {"verdict": "accept", "review_status": "ok", "must_flag_hits": []}
    assert _auto_accepts(snap) and not _auto_accepts({**snap, "layout_not_redrawn": ["x"]})


def test_the_review_card_file_says_the_request_was_not_applied(tmp_path: Path) -> None:
    """The line reaches NEXT_STEP.md, which the web page and VS Code show."""
    eng = _quest(tmp_path)
    state = {**STATE, "layout_not_redrawn": ["make the axis labels larger"], "review": {"verdict": "accept"}}
    with pytest.raises(RuntimeError, match="runnable context"):
        asyncio.run(eng._node_human_feedback(state))  # the pause itself raises outside a running graph
    card = (eng.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "Your figure request was NOT applied: make the axis labels larger" in card


def test_one_note_the_text_can_answer_does_not_excuse_another_that_changed_nothing(tmp_path: Path) -> None:
    eng = _quest(tmp_path)
    prompts: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return "```python\nprint('NOT_DRAWN: put Figure 2 first')\n```" if len(prompts) == 1 else f"```python\n{RIGHT}\n```"

    async def execute(cmd: list[str], **kw: Any) -> Any:
        proc = subprocess.run([sys.executable, cmd[-1]], cwd=str(kw["cwd"]), capture_output=True, text=True)
        return SimpleNamespace(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, timed_out=False)

    eng._chat = chat
    eng.executor = SimpleNamespace(python_path=lambda q: Path(sys.executable), execute=execute)
    out = asyncio.run(eng._node_replot_layout({**STATE, "refine_layout": ["larger labels", "put Figure 2 first"]}))
    assert len(prompts) == 2
    assert (eng.quest_root / "figures" / "fig1.png").read_bytes() == b"new-4"
    assert out["layout_not_redrawn"] == []


def test_a_redraw_that_runs_but_changes_no_figure_is_tried_again(tmp_path: Path) -> None:
    eng = _quest(tmp_path)
    out, prompts = _replot(eng, ["print('REPLOTTED: fig1.png')", RIGHT])
    assert len(prompts) == 2 and "changed no figure" in prompts[1]
    assert (eng.quest_root / "figures" / "fig1.png").read_bytes() == b"new-4"
    assert out["layout_not_redrawn"] == []


def test_a_note_only_the_text_can_answer_is_not_retried(tmp_path: Path) -> None:
    eng = _quest(tmp_path)
    out, prompts = _replot(eng, ["print('NOT_DRAWN: put Figure 2 first in the paper')"])
    assert len(prompts) == 1
    assert out["layout_missed"] == ["put Figure 2 first in the paper"] and out["layout_not_redrawn"] == []


# ---- the structure helper -------------------------------------------------------------------------------------------


def test_npz_npy_json_csv_and_other_files_are_described(tmp_path: Path) -> None:
    np.savez_compressed(tmp_path / "a.npz", x=np.zeros((3, 2), dtype=np.int32), label=np.array("hi"))
    np.save(tmp_path / "b.npy", np.zeros(7, dtype=np.float32))
    (tmp_path / "c.json").write_text(json.dumps({"cells": [1, 2], "fits": {"euler": 1}, "n": 3}), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps([{"n": 1, "t": 2.0}, {"n": 2, "t": 3.0}]), encoding="utf-8")
    (tmp_path / "e.csv").write_text("n,t\n8,1\n16,2\n", encoding="utf-8")
    (tmp_path / "f.tsv").write_text("a\tb c\n1\t2\n", encoding="utf-8")
    (tmp_path / "g.parquet").write_bytes(b"PAR1")
    assert describe_data_file(tmp_path / "a.npz") == (
        "arrays under the keys `x` (int32, shape (3, 2)), `label` (str (up to 2 chars), a single value)")
    assert describe_data_file(tmp_path / "b.npy") == "an array: float32, shape (7,)"
    assert describe_data_file(tmp_path / "c.json") == (
        "an object with the keys `cells`: list of 2; `fits`: object with keys `euler`; `n`: number")
    assert describe_data_file(tmp_path / "d.json") == "a list of 2 records; the first has the keys `n`, `t`"
    assert describe_data_file(tmp_path / "e.csv") == "a table with the columns `n`, `t` (first line), 2 line(s) below it"
    assert "`b c`" in describe_data_file(tmp_path / "f.tsv")
    assert describe_data_file(tmp_path / "g.parquet") == ""
    block = describe_data_files(tmp_path, ["g.parquet", "e.csv"])
    assert block.splitlines() == ["- g.parquet", "- e.csv: a table with the columns `n`, `t` (first line), 2 line(s) below it"]
    # An Excel-written CSV starts with a byte-order mark, which is not part of the first column's name.
    (tmp_path / "h.csv").write_bytes("﻿x,y\n1,2\n".encode("utf-8"))
    assert describe_data_file(tmp_path / "h.csv").startswith("a table with the columns `x`, `y`")


def test_every_key_of_a_large_archive_is_named(tmp_path: Path) -> None:
    np.savez(tmp_path / "big.npz", **{f"series_number_{i:03d}": np.zeros(3) for i in range(120)})
    block = describe_data_files(tmp_path, ["big.npz"])
    for i in range(120):
        assert f"`series_number_{i:03d}`" in block
    assert "(cut)" not in block


def test_a_broken_file_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "a.npz").write_bytes(b"not a zip")
    (tmp_path / "b.json").write_text("{nope", encoding="utf-8")
    (tmp_path / "c.npy").write_bytes(b"\x93NUMPY\x01\x00\x05\x00{'x'")
    header = b"{[1]: 2}"  # a literal whose evaluation raises TypeError, not SyntaxError
    (tmp_path / "d.npy").write_bytes(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header)
    # A member whose compressed data is corrupt, next to a good one: the archive is still listed.
    np.savez_compressed(tmp_path / "e.npz", bad=np.arange(4000), good=np.zeros(2))
    raw = bytearray((tmp_path / "e.npz").read_bytes())
    with zipfile.ZipFile(tmp_path / "e.npz") as zf:
        bad = zf.getinfo("bad.npy")
    start = bad.header_offset + 30 + len(bad.filename.encode()) + len(bad.extra)
    raw[start: start + 40] = b"\xff" * 40
    (tmp_path / "e.npz").write_bytes(bytes(raw))
    # A member marked as encrypted cannot be opened without a password.
    with zipfile.ZipFile(tmp_path / "f.npz", "w") as zf:
        zf.writestr("x.npy", b"\x93NUMPY")
    raw = bytearray((tmp_path / "f.npz").read_bytes())
    raw[6] |= 1  # local header flag
    cd = raw.rfind(b"PK\x01\x02")
    raw[cd + 8] |= 1  # central directory flag
    (tmp_path / "f.npz").write_bytes(bytes(raw))
    assert describe_data_file(tmp_path / "a.npz").startswith("could not be read")
    assert describe_data_file(tmp_path / "b.json") == "not readable as JSON"
    assert describe_data_file(tmp_path / "c.npy") == "not a readable .npy file"
    assert describe_data_file(tmp_path / "d.npy") == "not a readable .npy file"
    assert describe_data_file(tmp_path / "e.npz") == (
        "arrays under the keys `bad` (not read), `good` (float64, shape (2,))")
    assert describe_data_file(tmp_path / "f.npz") == "arrays under the keys `x` (not read)"


def test_a_name_is_shown_exactly_as_the_file_has_it(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("time  (s), n,`q`\n1,2,3\n", encoding="utf-8")
    assert describe_data_file(tmp_path / "a.csv").startswith(
        "a table with the columns 'time  (s)', ' n', '`q`' (first line)")
    keys = {f"k{i:03d}": 1 for i in range(300)}
    (tmp_path / "b.json").write_text(json.dumps(keys), encoding="utf-8")
    text = describe_data_file(tmp_path / "b.json")
    assert "`k239`" in text and "and 60 more (not listed here; read them in the script)" in text
    (tmp_path / "c.csv").write_text(",".join(f"c{i}" for i in range(20000)) + "\n1\n", encoding="utf-8")
    assert "too long to list its columns" in describe_data_file(tmp_path / "c.csv")
    (tmp_path / "d.csv").write_text("x,y", encoding="utf-8")  # a short header with no newline is still listed
    assert describe_data_file(tmp_path / "d.csv") == "a table with the columns `x`, `y` (first line), 0 line(s) below it"


def test_the_block_is_bounded_and_says_so(tmp_path: Path) -> None:
    names = [f"f{i:03d}.csv" for i in range(200)]
    for n in names:
        (tmp_path / n).write_text(",".join(f"column_number_{j}" for j in range(30)) + "\n1\n", encoding="utf-8")
    block = describe_data_files(tmp_path, names, max_chars=3000)
    assert len(block) < 3000 + 200
    assert block.splitlines()[-1].startswith("- ... and ") and "the list was cut to fit" in block


class _Boom:
    """Unpickling this writes a file: the helper must never unpickle an object array."""

    def __init__(self, target: str) -> None:
        self.target = target

    def __reduce__(self) -> Any:
        return (Path.write_text, (Path(self.target), "unpickled"))


def test_an_object_array_is_never_unpickled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = tmp_path / "unpickled.txt"
    arr = np.empty(1, dtype=object)
    arr[0] = _Boom(str(sentinel))
    np.savez(tmp_path / "p.npz", obj=arr, ok=np.arange(3))
    np.save(tmp_path / "p.npy", arr, allow_pickle=True)
    assert not sentinel.exists()

    def refuse(*a: Any, **kw: Any) -> Any:
        raise AssertionError("the helper unpickled a file")

    monkeypatch.setattr(pickle, "load", refuse)
    monkeypatch.setattr(pickle, "loads", refuse)
    monkeypatch.setattr(np, "load", refuse)
    text = describe_data_file(tmp_path / "p.npz")
    assert "`obj` (object (not read), shape (1,))" in text and "`ok` (int64, shape (3,))" in text
    assert describe_data_file(tmp_path / "p.npy") == "an array: object (not read), shape (1,)"
    assert not sentinel.exists()
    with zipfile.ZipFile(tmp_path / "p.npz") as zf:
        assert sorted(zf.namelist()) == ["obj.npy", "ok.npy"]
