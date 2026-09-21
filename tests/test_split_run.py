"""``core/split_run.py``: the simulation once, the analysis as often as needed.

A quest with ``execution.split_analysis`` keeps ``code/simulate.py`` (writes raw files) apart from
``code/experiment.py`` (reads them). These pin the two things that make the split worth having and
safe: raw files are used again exactly while the record of them fits (the simulation script's hash, the
seed, the files and their sizes), and a failure names the script it is in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import split_run as sr
from core.execution import ExecutionResult

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)


# --- where the raw files live -------------------------------------------------


def test_the_raw_folder_is_raw_in_the_quest_unless_configured(tmp_path: Path) -> None:
    assert sr.raw_root_of(tmp_path) == tmp_path / "raw"
    assert sr.raw_root_of(tmp_path, "  ") == tmp_path / "raw"
    assert sr.raw_root_of(tmp_path, "scratch/out") == tmp_path / "scratch" / "out"
    absolute = tmp_path.parent / "elsewhere"
    assert sr.raw_root_of(tmp_path, str(absolute)) == absolute


def test_the_scripts_get_a_path_relative_to_the_quest_when_it_can_be(tmp_path: Path) -> None:
    inside = sr.raw_dir_for(tmp_path / "raw", 2)
    assert sr.env_value(inside, tmp_path) == "raw/seed2"
    outside = tmp_path.parent / "elsewhere" / "seed0"
    assert sr.env_value(outside, tmp_path) == str(outside)


# --- the record of a seed's raw files -----------------------------------------


def _written(raw: Path, *, sha: str = "a" * 64, seed: int = 0) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "values.json").write_text('{"x": [1, 2, 3]}', encoding="utf-8")
    (raw / "nested").mkdir()
    (raw / "nested" / "more.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    sr.write_manifest(raw, sr.build_manifest(raw, simulate_sha=sha, seed=seed, index=0))


def test_the_manifest_lists_every_file_with_its_size_and_hash(tmp_path: Path) -> None:
    raw = tmp_path / "seed0"
    _written(raw)
    manifest = sr.read_manifest(raw)
    assert manifest and manifest["simulate_sha256"] == "a" * 64 and manifest["seed"] == 0
    files = {f["path"]: f for f in manifest["files"]}
    assert set(files) == {"values.json", "nested/more.csv"}  # the manifest is not listed in itself
    assert files["values.json"]["bytes"] == len('{"x": [1, 2, 3]}')
    assert files["values.json"]["sha256"] == sr.sha256_of(raw / "values.json")


def test_raw_files_are_current_while_their_record_fits(tmp_path: Path) -> None:
    raw = tmp_path / "seed0"
    _written(raw)
    assert sr.stale_reason(raw, simulate_sha="a" * 64, seed=0) is None


@pytest.mark.parametrize(
    "change,expected",
    [
        (lambda raw: None, None),
        (lambda raw: (raw / "values.json").unlink(), "values.json is missing"),
        (lambda raw: (raw / "values.json").write_text("{}", encoding="utf-8"), "not the size"),
        (lambda raw: (raw / sr.MANIFEST_NAME).unlink(), "no raw files"),
        (lambda raw: (raw / sr.MANIFEST_NAME).write_text("not json", encoding="utf-8"), "no raw files"),
    ],
)
def test_what_makes_raw_files_stale(tmp_path: Path, change, expected) -> None:
    raw = tmp_path / "seed0"
    _written(raw)
    change(raw)
    why = sr.stale_reason(raw, simulate_sha="a" * 64, seed=0)
    assert (why is None) if expected is None else (expected in why)


def test_a_rewritten_simulation_or_another_seed_makes_them_stale(tmp_path: Path) -> None:
    raw = tmp_path / "seed0"
    _written(raw)
    assert "simulate.py has changed" in sr.stale_reason(raw, simulate_sha="b" * 64, seed=0)
    assert "seed 0, not 5" in sr.stale_reason(raw, simulate_sha="a" * 64, seed=5)


def test_a_record_that_lists_no_files_is_not_used(tmp_path: Path) -> None:
    raw = tmp_path / "seed0"
    raw.mkdir()
    sr.write_manifest(raw, sr.build_manifest(raw, simulate_sha="a" * 64, seed=0, index=0))
    assert "lists no files" in sr.stale_reason(raw, simulate_sha="a" * 64, seed=0)


def test_a_file_too_large_to_hash_is_recorded_by_size_alone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sr, "HASH_LIMIT_BYTES", 4)
    raw = tmp_path / "seed0"
    _written(raw)
    files = {f["path"]: f for f in sr.read_manifest(raw)["files"]}
    assert files["values.json"]["sha256"] is None and files["values.json"]["bytes"] > 4
    assert sr.stale_reason(raw, simulate_sha="a" * 64, seed=0) is None


def test_the_listing_names_the_files_for_a_repair(tmp_path: Path) -> None:
    raw = tmp_path / "seed0"
    _written(raw)
    text = sr.listing(raw)
    assert "- values.json (16 bytes)" in text and "nested/more.csv" in text
    assert sr.listing(tmp_path / "nothing") == ""


# --- reading the code-writing reply ------------------------------------------


def _reply(sim: str = "print('sim')", ana: str = "print('ana')", *, deps: str = "DEPS: numpy") -> str:
    return f"```python\n# file: simulate.py\n{sim}\n```\n\n```python\n# file: experiment.py\n{ana}\n```\n{deps}\n"


def test_both_scripts_are_found_by_their_headers_in_either_order() -> None:
    got = sr.parse_split_response(_reply(), FENCE)
    assert got == {"simulate": "print('sim')", "analysis": "print('ana')"}  # the header names a script, it is not in it
    swapped = (
        "```python\n# file: experiment.py\nA\n```\n```python\n# file: simulate.py\nS\n```\n"
    )
    got = sr.parse_split_response(swapped, FENCE)
    assert got["simulate"].endswith("S") and got["analysis"].endswith("A")


def test_two_unmarked_blocks_are_read_in_order_and_anything_else_is_not_a_split() -> None:
    two = "```python\nprint('first')\n```\n```python\nprint('second')\n```\n"
    got = sr.parse_split_response(two, FENCE)
    assert got == {"simulate": "print('first')", "analysis": "print('second')"}
    assert sr.parse_split_response("```python\nprint(1)\n```\n", FENCE) is None
    assert sr.parse_split_response("", FENCE) is None
    one_marked = "```python\n# file: simulate.py\nS\n```\n```python\nprint('unmarked')\n```\n"
    assert sr.parse_split_response(one_marked, FENCE) is None


def test_a_script_given_back_with_other_line_endings_is_the_same_script() -> None:
    assert sr.same_script("a = 1\nb = 2\n", "a = 1  \r\nb = 2")
    assert not sr.same_script("a = 1\nb = 2\n", "a = 1\nb = 3\n")


# --- the runner ---------------------------------------------------------------


class _Inner:
    """A stand-in executor. ``simulate`` and ``analysis`` are what the two scripts do."""

    def __init__(self, quest: Path) -> None:
        self.quest = quest
        self.calls: list[str] = []
        self.envs: list[dict] = []
        self.simulate_rc, self.simulate_writes, self.analysis_rc = 0, True, 0

    async def execute(self, cmd, *, cwd, timeout_s, env=None):
        name = Path(cmd[1]).name if len(cmd) > 1 else cmd[0]
        self.calls.append(name)
        self.envs.append(dict(env or {}))
        if name == "simulate.py":
            if self.simulate_writes and self.simulate_rc == 0:
                target = Path(cwd) / env["FI_RAW_DIR"]
                target.mkdir(parents=True, exist_ok=True)
                (target / "values.json").write_text(json.dumps({"seed": env["FI_REPLICATE_SEED"]}), encoding="utf-8")
            return ExecutionResult(self.simulate_rc, "sim out", "sim err" if self.simulate_rc else "", 2.0)
        if name == "experiment.py":
            return ExecutionResult(self.analysis_rc, 'RESULT_JSON: {"a": 1}', "boom" if self.analysis_rc else "", 0.5)
        return ExecutionResult(0, "passed through", "", 0.1)


@pytest.fixture()
def quest(tmp_path: Path):
    (tmp_path / "code").mkdir()
    simulate, analysis = tmp_path / "code" / "simulate.py", tmp_path / "code" / "experiment.py"
    simulate.write_text("# the simulation\n", encoding="utf-8")
    analysis.write_text("# the analysis\n", encoding="utf-8")
    inner = _Inner(tmp_path)
    log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    runner = sr.SplitRunner(
        inner, quest_root=tmp_path, raw_root=tmp_path / "raw", simulate=simulate, analysis=analysis, log=log,
    )
    return runner, inner, simulate, analysis, tmp_path


async def _run(runner, analysis: Path, quest: Path, *, index: int = 0):
    env = {"FI_REPLICATE_INDEX": str(index), "FI_REPLICATE_SEED": str(index * 1_000_000)}
    return await runner.execute(["python", str(analysis)], cwd=quest, timeout_s=60, env=env)


@pytest.mark.asyncio
async def test_a_run_simulates_then_analyses_and_records_the_raw_files(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    result = await _run(runner, analysis, root)
    assert inner.calls == ["simulate.py", "experiment.py"]
    assert result.returncode == 0 and "RESULT_JSON" in result.stdout
    assert result.duration_s == pytest.approx(2.5)  # the two runs together
    assert inner.envs[0]["FI_RAW_DIR"] == "raw/seed0" == inner.envs[1]["FI_RAW_DIR"]
    manifest = sr.read_manifest(root / "raw" / "seed0")
    assert manifest["simulate_sha256"] == sr.sha256_of(simulate)
    assert runner.failed_script is None and (runner.simulated, runner.reused) == (1, 0)


@pytest.mark.asyncio
async def test_the_analysis_alone_runs_while_the_simulation_is_unchanged(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    await _run(runner, analysis, root)
    analysis.write_text("# the analysis, rewritten\n", encoding="utf-8")  # a repair of the analysis
    await _run(runner, analysis, root)
    assert inner.calls == ["simulate.py", "experiment.py", "experiment.py"]
    assert (runner.simulated, runner.reused) == (1, 1)


@pytest.mark.asyncio
async def test_a_rewritten_simulation_runs_again_and_leaves_none_of_the_old_files(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    await _run(runner, analysis, root)
    (root / "raw" / "seed0" / "leftover.csv").write_text("old", encoding="utf-8")
    simulate.write_text("# a different simulation\n", encoding="utf-8")
    await _run(runner, analysis, root)
    assert inner.calls == ["simulate.py", "experiment.py", "simulate.py", "experiment.py"]
    assert not (root / "raw" / "seed0" / "leftover.csv").exists()


@pytest.mark.asyncio
async def test_a_missing_raw_file_runs_the_simulation_again(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    await _run(runner, analysis, root)
    (root / "raw" / "seed0" / "values.json").unlink()
    await _run(runner, analysis, root)
    assert inner.calls.count("simulate.py") == 2


@pytest.mark.asyncio
async def test_every_seed_has_its_own_folder_and_record(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    await _run(runner, analysis, root, index=0)
    await _run(runner, analysis, root, index=2)
    assert inner.envs[2]["FI_RAW_DIR"] == "raw/seed2"
    assert sr.read_manifest(root / "raw" / "seed2")["seed"] == 2_000_000
    # the first seed's files are not touched by the third's
    assert sr.stale_reason(root / "raw" / "seed0", simulate_sha=sr.sha256_of(simulate), seed=0) is None


@pytest.mark.asyncio
async def test_a_failing_simulation_is_named_and_the_analysis_does_not_run(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    inner.simulate_rc = 3
    result = await _run(runner, analysis, root)
    assert result.returncode == 3 and inner.calls == ["simulate.py"]
    assert runner.failed_script == "simulate.py"
    assert "simulate.py (the simulation) failed" in result.stderr and "sim err" in result.stderr
    assert sr.read_manifest(root / "raw" / "seed0") is None  # nothing recorded for a failed run


@pytest.mark.asyncio
async def test_a_simulation_that_saves_nothing_fails_in_its_own_script(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    inner.simulate_writes = False
    result = await _run(runner, analysis, root)
    assert result.returncode == 1 and inner.calls == ["simulate.py"]
    assert runner.failed_script == "simulate.py"
    assert "wrote no file into the folder the environment variable FI_RAW_DIR names (raw/seed0 for this run)" in result.stderr
    assert 'os.environ["FI_RAW_DIR"]' in result.stderr


@pytest.mark.asyncio
async def test_a_failing_analysis_is_named_and_keeps_the_raw_files_for_the_next_try(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    inner.analysis_rc = 1
    result = await _run(runner, analysis, root)
    assert result.returncode == 1 and runner.failed_script == "experiment.py"
    inner.analysis_rc = 0
    await _run(runner, analysis, root)
    assert inner.calls == ["simulate.py", "experiment.py", "experiment.py"]
    assert runner.failed_script is None


@pytest.mark.asyncio
async def test_a_command_that_is_not_the_analysis_passes_straight_through(quest) -> None:
    runner, inner, simulate, analysis, root = quest
    result = await runner.execute(["python", "-c", "import numpy"], cwd=root, timeout_s=60)
    assert result.stdout == "passed through" and inner.calls == ["-c"]
    assert not (root / "raw").exists()
