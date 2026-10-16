"""FI runs the quest's code from a copy outside the quest folder (core/outside_run.py)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from core.execution import ExecutionResult
from core.outside_run import ANALYSIS_INPUTS, OutsideCopy, OutsideExecutor, named_paths


class _Real:
    def __init__(self) -> None:
        self.cwd: Path | None = None

    async def execute(self, cmd, *, cwd, timeout_s, env=None):
        self.cwd = Path(cwd)
        done = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, env=env)
        return ExecutionResult(done.returncode, done.stdout, done.stderr, 0.0)


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "1791991314-kinetics-quest"
    (root / "code" / "tests").mkdir(parents=True)
    (root / "code" / ".git").mkdir()
    (root / "code" / ".git" / "HEAD").write_text("ref\n", encoding="utf-8")
    (root / "code" / "tests" / "equation_numbers.json").write_text('{"E1": {"expected": 6.0}}', encoding="utf-8")
    (root / "code" / "simulate.py").write_text("def f():\n    raise ValueError('boom')\n", encoding="utf-8")
    (root / "needs").mkdir()
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text("{}", encoding="utf-8")
    (root / ".fi" / "checks").mkdir(parents=True)
    (root / ".fi" / "checks" / "oracle_values.json").write_text("{}", encoding="utf-8")
    (root / "raw").mkdir()
    (root / "raw" / "old.txt").write_text("old", encoding="utf-8")
    return root


def test_the_copy_holds_code_and_inputs_only_and_is_named_by_a_random_id(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    for rel in ("inputs/a.csv", "data/collected.csv", "data/results/r.csv", "results/s.json", "figures/f.png"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    box = OutsideCopy(root, named=[".fi/trials/spec.json", "needs/FROZEN_PROTOCOL.json"], base=tmp_path / "s")
    copy = box.make()
    try:
        assert root.name not in str(copy) and len(copy.name) == 32
        files = {p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()}
        # A cell, a check case or an equation measurement: code/, inputs/ and data/ (not the results written there).
        assert files == {"code/simulate.py", "inputs/a.csv", "data/collected.csv"}, files
    finally:
        box.remove()
    assert not copy.exists()
    # The analysis reads what the cells wrote: every folder but FI's own (never needs/, .fi/, paper/, a top-level file).
    (root / "cache").mkdir()
    (root / "cache" / "traj_1.txt").write_text("x", encoding="utf-8")
    (root / "paper").mkdir()
    (root / "paper" / "paper.md").write_text("x", encoding="utf-8")
    (root / "plan.md").write_text("expected 0.6415", encoding="utf-8")
    box = OutsideCopy(root, base=tmp_path / "s", inputs=ANALYSIS_INPUTS, skip=())
    copy = box.make()
    try:
        files = {p.relative_to(copy).as_posix() for p in copy.rglob("*") if p.is_file()}
        assert files == {"code/simulate.py", "inputs/a.csv", "data/collected.csv", "data/results/r.csv",
                         "raw/old.txt", "results/s.json", "figures/f.png", "cache/traj_1.txt"}, files
    finally:
        box.remove()


async def test_outputs_come_back_by_fis_list_and_a_traceback_names_the_quests_file(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / ".fi" / "trials").mkdir()
    script = ("import sys, traceback, pathlib; sys.path.insert(0, 'code')\n"
              "pathlib.Path('raw').mkdir(exist_ok=True); pathlib.Path('raw/new.txt').write_text('new')\n"
              "pathlib.Path('needs').mkdir(exist_ok=True)\n"
              "for i in range(8): pathlib.Path('needs/forged%d.json' % i).write_text('{}')\n"
              "pathlib.Path('code/stray.py').write_text('x')\n"
              "pathlib.Path('stray.txt').write_text('x')\n"
              "import simulate\n"
              "try:\n    simulate.f()\nexcept ValueError:\n"
              "    pathlib.Path(sys.argv[1]).write_text(traceback.format_exc())\n    traceback.print_exc()\n")
    inner = _Real()
    log = logging.getLogger("t-outside")
    seen: list[str] = []
    log.warning = lambda msg, *a: seen.append(msg % a)  # type: ignore[method-assign]
    ran = await OutsideExecutor(inner, root, base=tmp_path / "s", log=log).execute(
        [sys.executable, "-B", "-c", script, ".fi/trials/out.txt"], cwd=root, timeout_s=60)
    assert ran.returncode == 0, ran.stderr
    assert inner.cwd is not None and not inner.cwd.exists()
    assert (root / "raw" / "new.txt").read_text() == "new" and (root / "raw" / "old.txt").exists()
    # Anything the run wrote comes back, except into FI's own folders (needs/, .fi/, code/): those are dropped, and
    # every dropped path is in the log.
    assert (root / "stray.txt").exists()
    assert not any((root / "needs" / f"forged{i}.json").exists() for i in range(8))
    assert not (root / "code" / "stray.py").exists()
    assert str(root.resolve() / "code" / "simulate.py") in ran.stderr
    assert str(root.resolve() / "code" / "simulate.py") in (root / ".fi" / "trials" / "out.txt").read_text()
    said = "\n".join(seen)
    assert all(f"needs/forged{i}.json" in said for i in range(8)) and "code/stray.py" in said, said


def test_a_path_named_on_the_command_line_or_in_the_environment_is_found(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / ".fi" / "trials").mkdir()
    got = named_paths(root, ["-B", ".fi/trials/cell0.out.jsonl", "raw", "3", str(root / "raw" / "old.txt"),
                             "needs/FROZEN_PROTOCOL.json", str(root / ".venv" / "python.exe")])
    assert got == [".fi/trials/cell0.out.jsonl", "raw", "raw/old.txt"]
