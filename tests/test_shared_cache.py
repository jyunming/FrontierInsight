"""The settings of a study share one folder, ``data/results/cache/`` (core/outside_run.py): a value one setting worked
out is read by the next, not worked out again, and nothing a setting kept there is lost to another's copy."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from core import outside_run
from core import trial_runner as tr
from core.execution import ExecutionResult, SharedInterpreterExecutor
from core.outside_run import ANALYSIS_INPUTS, SHARED_CACHE, OutsideCopy, OutsideExecutor

SIM = '''
import os, json, time
CACHE = os.path.join("data", "results", "cache")

def run_cell(cell):
    path = os.path.join(CACHE, "shared.json")
    computed = 0.0
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        time.sleep(0.3)
        tmp = path + ".%d.tmp" % os.getpid()
        with open(tmp, "w") as fh:
            json.dump({"v": 7}, fh)
        os.replace(tmp, path)
        computed = 1.0
    with open(path) as fh:
        v = json.load(fh)["v"]
    return {"y": float(cell["n"]) * v, "computed_shared": computed}
'''


def _quest(base: Path, source: str = SIM) -> Path:
    root = base / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (root / "data").mkdir()
    return root


def _run(root: Path, grid: dict[str, list[Any]]) -> dict[str, dict[str, float]]:
    asyncio.run(tr.run_trials(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py",
                              grid, runs_per_setting=1, base_seed=1, deterministic=True, timeout_s=120))
    out = {}
    for path in sorted((root / ".fi" / "trials").glob("cell*.out.jsonl")):
        row = next(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if '"values"' in x)
        out[path.name] = row["values"]
    return out


def test_a_value_several_settings_use_is_worked_out_once(tmp_path: Path) -> None:
    values = _run(_quest(tmp_path / "a"), {"n": [1, 2, 3, 4]})
    assert len(values) == 4 and sum(v["computed_shared"] for v in values.values()) == 1.0, values
    # the same numbers as when every setting works it out itself (no folder shared)
    alone = {}
    for n in (1, 2, 3, 4):
        alone.update({f"n{n}": _run(_quest(tmp_path / f"b{n}"), {"n": [n]})})
    assert sorted(v["y"] for v in values.values()) == sorted(v["y"] for d in alone.values() for v in d.values())


def test_the_copy_back_never_removes_a_file_another_setting_kept(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    outside_run.prepare_shared_cache(root)
    box = OutsideCopy(root, base=tmp_path / "s")
    copy = box.make()
    try:
        assert box.linked == [SHARED_CACHE]
        # another setting, running beside this one, keeps a value while this one runs
        (root / SHARED_CACHE / "sibling.json").write_text("{}", encoding="utf-8")
        (copy / SHARED_CACHE / "mine.json").write_text("{}", encoding="utf-8")  # this one's, through the link
        back = box.bring_back()
        assert not any(r.startswith(SHARED_CACHE) for r in back), back
    finally:
        box.remove()
    assert (root / SHARED_CACHE / "sibling.json").is_file() and (root / SHARED_CACHE / "mine.json").is_file()
    assert not copy.exists()


def test_the_analysis_copy_still_holds_its_own_copy_of_the_folder(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / SHARED_CACHE).mkdir(parents=True)
    (root / SHARED_CACHE / "v.json").write_text("{}", encoding="utf-8")
    box = OutsideCopy(root, base=tmp_path / "s", inputs=ANALYSIS_INPUTS, skip=())
    copy = box.make()
    try:
        assert box.linked == [] and (copy / SHARED_CACHE / "v.json").is_file()
    finally:
        box.remove()
    assert (root / SHARED_CACHE / "v.json").is_file()


def test_when_no_link_can_be_made_each_setting_works_it_out_and_the_log_says_so(tmp_path: Path, monkeypatch,
                                                                                  caplog) -> None:  # noqa: ANN001
    monkeypatch.setattr(outside_run, "_link_folder", lambda link, target: False)
    with caplog.at_level(logging.WARNING):
        values = _run(_quest(tmp_path), {"n": [1, 2]})
    assert sum(v["computed_shared"] for v in values.values()) == 2.0
    assert sum("cannot share data/results/cache/" in r.getMessage() for r in caplog.records) == 1


def test_what_other_code_kept_there_is_moved_aside_before_anything_runs(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    outside_run.prepare_shared_cache(root)
    (root / SHARED_CACHE / "shared.json").write_text('{"v": 1}', encoding="utf-8")  # worked out by this code
    outside_run.prepare_shared_cache(root)
    assert (root / SHARED_CACHE / "shared.json").is_file(), "the same code keeps what it worked out"
    (root / "code" / "simulate.py").write_text(SIM.replace("* v", "* v + 0"), encoding="utf-8")  # a repair
    outside_run.prepare_shared_cache(root)
    assert not any((root / SHARED_CACHE).iterdir()), "nothing worked out by the earlier code is read"
    moved = list((root / ".fi" / "previous").glob("shared_cache_*/shared.json"))
    assert len(moved) == 1 and moved[0].read_text(encoding="utf-8") == '{"v": 1}'


def test_a_changed_analysis_script_or_note_keeps_what_the_simulation_worked_out(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    outside_run.prepare_shared_cache(root)
    (root / SHARED_CACHE / "shared.json").write_text('{"v": 1}', encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('repaired summary')\n", encoding="utf-8")
    (root / "code" / "CHANGELOG.md").write_text("- the summary reads every value\n", encoding="utf-8")
    outside_run.prepare_shared_cache(root)
    assert (root / SHARED_CACHE / "shared.json").is_file(), "the simulation did not change, so its values stay"
    assert not (root / ".fi" / "previous").exists()
    (root / "code" / "model").mkdir()
    (root / "code" / "model" / "__init__.py").write_text("", encoding="utf-8")
    (root / "code" / "model" / "physics.py").write_text("def f(v):\n    return v\n", encoding="utf-8")
    outside_run.prepare_shared_cache(root)
    assert not any((root / SHARED_CACHE).iterdir()), "the model package is the simulation's code too"


class _Writes:
    """Writes a file in the shared folder of the folder it runs in; stopped for its time when told, after ``wait``."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        name = cmd[-1]
        (Path(cwd) / SHARED_CACHE).mkdir(parents=True, exist_ok=True)
        (Path(cwd) / SHARED_CACHE / f"{name}.lock").write_text("x", encoding="utf-8")
        gate = self.gates.get(name)
        if gate is not None:
            await gate.wait()
        return ExecutionResult(returncode=-9, stdout="", stderr="", duration_s=1.0, timed_out=True)


def test_a_stopped_setting_alone_leaves_nothing_and_beside_others_keeps_what_it_cannot_tell(tmp_path: Path,
                                                                                         caplog) -> None:  # noqa: ANN001
    root = _quest(tmp_path)
    inner = _Writes()
    ex = OutsideExecutor(inner, root, base=tmp_path / "s", log=logging.getLogger("t"))
    with caplog.at_level(logging.WARNING):
        asyncio.run(ex.execute(["python", "alone"], cwd=root, timeout_s=5))
    assert not (root / SHARED_CACHE / "alone.lock").exists(), "its own unfinished file is removed"
    assert "removed, so the next trial does not wait" in caplog.text

    async def both() -> None:
        inner.gates = {"first": asyncio.Event(), "second": asyncio.Event()}
        a = asyncio.create_task(ex.execute(["python", "first"], cwd=root, timeout_s=5))
        b = asyncio.create_task(ex.execute(["python", "second"], cwd=root, timeout_s=5))
        await asyncio.sleep(0.5)
        inner.gates["first"].set()
        inner.gates["second"].set()
        await asyncio.gather(a, b)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        asyncio.run(both())
    assert (root / SHARED_CACHE / "first.lock").exists() and (root / SHARED_CACHE / "second.lock").exists()
    assert "cannot tell which setting wrote them" in caplog.text


class _WritesValue:
    """Finishes one value in the shared folder (written whole under its own name), then is stopped for its time while
    working out a second one, leaving that one's lock and temporary file."""

    async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
        cache = Path(cwd) / SHARED_CACHE
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "done.json").write_text('{"v": 7}', encoding="utf-8")
        (cache / "next.json.lock").write_text("x", encoding="utf-8")
        (cache / "next.json.tmp123").write_text("{", encoding="utf-8")
        return ExecutionResult(returncode=-9, stdout="", stderr="", duration_s=1.0, timed_out=True)


def test_a_value_a_stopped_setting_finished_is_kept_for_the_next(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    ex = OutsideExecutor(_WritesValue(), root, base=tmp_path / "s", log=logging.getLogger("t"))
    asyncio.run(ex.execute(["python", "x"], cwd=root, timeout_s=5))
    cache = root / SHARED_CACHE
    assert (cache / "done.json").read_text(encoding="utf-8") == '{"v": 7}', "a finished value is kept"
    assert not (cache / "next.json.lock").exists() and not (cache / "next.json.tmp123").exists()


def test_a_stopped_trials_finished_shared_values_survive_its_cleanup(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    before = outside_run.working_files(root)
    cache = root / SHARED_CACHE
    cache.mkdir(parents=True)
    for name in ("a.npz", "b.json", "b.json.lock", "c.npz.tmp99", "d.12.tmp"):
        (cache / name).write_text("x", encoding="utf-8")
    (root / "data" / "made.tmp").write_text("x", encoding="utf-8")
    gone = outside_run.remove_created_since(root, before)
    assert sorted(gone) == sorted([f"{SHARED_CACHE}/b.json.lock", f"{SHARED_CACHE}/c.npz.tmp99",
                                   f"{SHARED_CACHE}/d.12.tmp", "data/made.tmp"])
    assert (cache / "a.npz").exists() and (cache / "b.json").exists()


def test_a_folder_of_other_code_that_cannot_be_moved_aside_is_not_shared(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    root = _quest(tmp_path)
    outside_run.prepare_shared_cache(root)
    (root / SHARED_CACHE / "shared.json").write_text('{"v": 1}', encoding="utf-8")
    record = (root / outside_run.SHARED_CACHE_RECORD).read_text(encoding="utf-8")
    (root / "code" / "simulate.py").write_text(SIM.replace("* v", "* v + 0"), encoding="utf-8")  # a repair
    real = outside_run.os.replace

    def held_open(src, dst):  # noqa: ANN001, ANN202
        if Path(src) == root / SHARED_CACHE:
            raise PermissionError("a file in it is open")
        return real(src, dst)

    monkeypatch.setattr(outside_run.os, "replace", held_open)
    monkeypatch.setattr(outside_run, "_link_folder", lambda link, target: (_ for _ in ()).throw(AssertionError("linked")))
    assert outside_run.prepare_shared_cache(root) is None
    assert (root / outside_run.SHARED_CACHE_RECORD).read_text(encoding="utf-8") == record, "still the earlier code's"
    ex = OutsideExecutor(SharedInterpreterExecutor(python_version="3.11"), root, base=tmp_path / "s",
                         log=logging.getLogger("t"))
    asyncio.run(ex.execute([sys.executable, "-c", "print(1)"], cwd=root, timeout_s=30))
    assert ex.last is not None and ex.last.linked == [] and ex.last.not_shared


def test_settings_starting_at_once_move_aside_only_the_earlier_code_s_folder(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import threading
    import time

    root = _quest(tmp_path)
    outside_run.prepare_shared_cache(root)
    (root / SHARED_CACHE / "old.json").write_text("{}", encoding="utf-8")
    (root / "code" / "simulate.py").write_text(SIM.replace("* v", "* v + 0"), encoding="utf-8")  # a repair
    version = outside_run.code_version(root)
    start = threading.Barrier(6)

    def late(r):  # noqa: ANN001, ANN202
        # all six start together; five read FI's record while the first is still moving the folder aside
        if start.wait() != 0:
            time.sleep(0.1)
        return version

    monkeypatch.setattr(outside_run, "code_version", late)
    real = outside_run.os.replace
    first = threading.Event()

    def slow_move(src, dst):  # noqa: ANN001, ANN202
        out = real(src, dst)
        if Path(src) == root / SHARED_CACHE and not first.is_set():
            first.set()
            (root / SHARED_CACHE).mkdir()
            (root / SHARED_CACHE / "fresh.json").write_text("{}", encoding="utf-8")  # a setting already writing there
            time.sleep(0.3)
        return out

    monkeypatch.setattr(outside_run.os, "replace", slow_move)
    threads = [threading.Thread(target=outside_run.prepare_shared_cache, args=(root,)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(list((root / ".fi" / "previous").glob("shared_cache_*"))) == 1
    assert (root / SHARED_CACHE / "fresh.json").is_file(), "the folder the new code writes into stays where it is"


def test_what_settings_stopped_together_left_unfinished_is_removed_once_none_runs(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    inner = _Writes()
    ex = OutsideExecutor(inner, root, base=tmp_path / "s", log=logging.getLogger("t"))

    async def both() -> None:
        inner.gates = {"first": asyncio.Event(), "second": asyncio.Event()}
        a = asyncio.create_task(ex.execute(["python", "first"], cwd=root, timeout_s=5))
        b = asyncio.create_task(ex.execute(["python", "second"], cwd=root, timeout_s=5))
        await asyncio.sleep(0.5)
        inner.gates["first"].set()
        inner.gates["second"].set()
        await asyncio.gather(a, b)
        await ex.settle()

    outside_run.prepare_shared_cache(root)
    (root / SHARED_CACHE / "before.lock").write_text("x", encoding="utf-8")  # not this batch's: left alone
    asyncio.run(both())
    assert not (root / SHARED_CACHE / "first.lock").exists() and not (root / SHARED_CACHE / "second.lock").exists()
    assert (root / SHARED_CACHE / "before.lock").exists()


def test_a_container_run_does_not_get_a_link_it_cannot_follow(tmp_path: Path) -> None:
    from core.execution import DockerExecutor

    root = _quest(tmp_path)
    (root / SHARED_CACHE).mkdir(parents=True)
    inner = object.__new__(DockerExecutor)

    async def fake(cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN202
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    inner.execute = fake  # type: ignore[method-assign]
    ex = OutsideExecutor(inner, root, base=tmp_path / "s", log=logging.getLogger("t"))
    asyncio.run(ex.execute(["python", "x"], cwd=root, timeout_s=5))
    assert ex.last is not None and ex.last.linked == [] and ex.last.not_shared


def test_only_temporary_names_count_as_unfinished() -> None:
    names = ["a.npz", "x.tmp_table.npz", "b.json.lock", "c.npz.tmp99", "d.12.tmp", "e.tmp.npy", "f.tmp"]
    got = outside_run.unfinished_shared({f"{SHARED_CACHE}/{n}" for n in names})
    assert sorted(r.rsplit("/", 1)[-1] for r in got) == ["b.json.lock", "c.npz.tmp99", "d.12.tmp", "e.tmp.npy", "f.tmp"]
