"""The analysis gate in the engine (``engine.pipeline_gate``): where it is called and that nothing at all changes with the
switch off. Neutral made-up study: a tank whose fill time is measured at a few pump rates; the analysis averages the
recorded trials of every setting. The gate itself is tested in tests/test_pipeline_gate.py; here a change that breaks
the analysis is refused at each place the engine adopts a change, with the files put back."""

from __future__ import annotations

import asyncio
import json
import logging
import string
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import finding_repair as fr
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_pipeline_gate import make_quest

SIMULATE = "SCALE = 1.0\n\n\ndef run_cell(cell):\n    return {'fill_s': cell['rate'] * SCALE}\n"
EXPERIMENT = '''import json, os

import simulate


def fill_time(m):
    return sum(m["values"]) / len(m["values"]) + 1.0


def main():
    data = json.load(open(os.environ["FI_TRIALS"]))
    out = {c["key"]: fill_time(c["metrics"]["fill_s"]) * simulate.SCALE for c in data["cells"]}
    print("RESULT_JSON: " + json.dumps(out))


if __name__ == "__main__":
    main()
'''
GOOD_FILL = 'def fill_time(m):\n    return sum(m["values"]) / len(m["values"])\n'
# passes the finding's own test (two values) and breaks the analysis (three trials of every setting): AttributeError
BAD_FILL = ('def fill_time(m):\n    if len(m["values"]) == 2:\n        return 3.0\n    return m.mean_value()\n')
FINDING_TEST = ('import experiment\n\n\ndef test_fill():\n'
                '    assert experiment.fill_time({"values": [2.0, 4.0]}) == 3.0, "one second too many"\n')
NO_SCALE = "def run_cell(cell):\n    return {'fill_s': cell['rate']}\n"  # drops SCALE, which the analysis reads


def _block(text: str) -> str:
    return f"```python\n{text}```"


def _engine(tmp_path: Path, gate: bool, **engine: Any) -> Engine:
    cfg = Config(
        topic="tank fill", title="gatewire", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False,
                            pipeline_gate=gate, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"))
    eng = Engine(cfg)
    make_quest(eng.quest_root)
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (code / "experiment.py").write_text(EXPERIMENT, encoding="utf-8")
    eng._protocol_block = lambda state, **kw: {}  # type: ignore[method-assign]
    return eng


def _quiet(eng: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing(*a: Any, **k: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(eng, "_small_run", nothing)
    eng._small_run_ctx = ({}, sys.executable, {})  # type: ignore[attr-defined]


def _gate_files(eng: Engine) -> bool:
    return (eng.fi_dir / "pipeline_gate.json").exists()


# --- the switch ------------------------------------------------------------------------------------------------------


def test_the_switch_is_off_by_default_and_a_plain_config_stays_a_plain_config() -> None:
    assert EngineConfig().pipeline_gate is False


@pytest.mark.asyncio
async def test_with_the_switch_off_nothing_is_run_written_or_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    caplog: pytest.LogCaptureFixture) -> None:
    eng = _engine(tmp_path, gate=False)
    _quiet(eng, monkeypatch)

    async def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("the gate was called with the switch off")

    monkeypatch.setattr("core.pipeline_gate.analysis_gate", never)
    (eng.quest_root / "code" / "experiment.py").write_text(EXPERIMENT.replace("fill_time(m)", "fill_time(m, x)"), encoding="utf-8")
    with caplog.at_level(logging.DEBUG):
        assert await eng._small_run_again() == ""
        assert await eng._simulation_repair_fault({}, SIMULATE) == ""
        assert await eng._gate_first({}) is None
        assert await eng._gate_refusal({}) == ""
    assert not _gate_files(eng) and "[gate]" not in caplog.text
    assert "_gate_last" not in eng.__dict__ and "_gate_cache" not in eng.__dict__


@pytest.mark.asyncio
async def test_no_recorded_raw_and_no_simulation_mean_no_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=True)
    assert await eng._analysis_gate({"no_simulation_resolved": True}) is None
    (eng.quest_root / "raw" / "trials.json").unlink()
    assert await eng._analysis_gate({}) is None and not _gate_files(eng)


# --- a finding repair (RunChecksMixin._small_run_again, through core/finding_repair.py) -------------------------------


class _Log:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


async def _finding_repair(eng: Engine, monkeypatch: pytest.MonkeyPatch, fill: str) -> tuple[dict[str, Any], list[str]]:
    _quiet(eng, monkeypatch)
    root, code = eng.quest_root, eng.quest_root / "code"
    issue = {"id": "F-1", "kind": "quantity", "file": "experiment.py", "function": "fill_time", "text": "one second too many",
             "why": "w", "fix": "f", "severity": "high", "changes_result": True, "readings": [{"n": 1, "status": "open"}],
             "demonstrated": None, "outcome": "open", "repairs": [], "tests": 0, "calls": {"test": 0, "repair": 0, "other": 0},
             "rounds": 0, "wordings": []}
    record: dict[str, Any] = {"issues": {"F-1": issue}, "readings": [{"n": 1}]}
    replies = {"code:test": [_block(FINDING_TEST)], "code:repair": [_block(fill)]}

    async def ask(prompt: str, who: str, kind: str) -> str:
        queue = replies.get(f"{who}:{kind}") or []
        return queue.pop(0) if queue else ""

    async def run_tests(paths: list[str]) -> dict[str, Any]:
        out = {}
        for p in paths:
            done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(code), [str(root / p)])],
                                  capture_output=True, text=True, cwd=root, timeout=60)
            out[p] = fr.parse_runs(done.stdout, [str(root / p)])[str(root / p)]
        return out

    async def snapshot() -> dict[str, str]:
        return {}

    prompts = {"code_finding_test": string.Template("TEST $finding\n$code\n$problem"),
               "code_review_repair": string.Template("REPAIR $finding\n$code\n$test\n$failure\n$how\n$problem")}
    env = fr.Env(root=root, code=code, record=record, save=lambda: None, ask=ask, run_tests=run_tests, snapshot=snapshot,
                 static=dict, keeps=lambda old, new, rel: "", prompts=prompts, topic="a tank", equations="[]",
                 scrub=lambda t: t, log=_Log(), package="", code_model="writer", small_run=eng._small_run_again,
                 notes=[])
    eng._gate_notes = env.notes  # as the engine does
    changed = await fr.repair_issues(env)
    return issue, changed


@pytest.mark.asyncio
async def test_a_finding_repair_that_breaks_the_analysis_is_refused_with_the_gates_sentence_and_the_file_put_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=True)
    assert await eng._gate_first({}) is None and eng._gate_last.ok  # type: ignore[attr-defined]  # the code before the repair
    issue, changed = await _finding_repair(eng, monkeypatch, BAD_FILL)
    assert changed == []
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == EXPERIMENT
    why = json.dumps(issue["repairs"])
    assert "the analysis no longer runs" in why and "has an error in its code" in why and "running the analysis" in why
    assert json.loads((eng.fi_dir / "pipeline_gate.json").read_text(encoding="utf-8"))["kind"] == "fault"


@pytest.mark.asyncio
async def test_the_same_repair_is_kept_as_before_with_the_switch_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=False)
    issue, changed = await _finding_repair(eng, monkeypatch, BAD_FILL)
    assert changed == ["experiment.py"]
    assert "m.mean_value()" in (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    assert not _gate_files(eng)


@pytest.mark.asyncio
async def test_a_good_finding_repair_is_kept_with_the_switch_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=True)
    assert await eng._gate_first({}) is None
    issue, changed = await _finding_repair(eng, monkeypatch, GOOD_FILL)
    assert changed == ["experiment.py"] and issue["outcome"] == "fixed"
    assert eng._gate_last.ok  # type: ignore[attr-defined]


# --- _simulation_repair_fault (one place for its callers) ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_simulation_repair_that_breaks_the_analysis_is_a_fault_and_one_caller_puts_the_files_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=True)

    async def entry_fine(state: Any, new_code: str) -> str:
        return ""

    monkeypatch.setattr(eng, "_simulation_repair_fault_uncached", entry_fine)
    assert await eng._gate_first({}) is None and eng._gate_last.ok  # type: ignore[attr-defined]
    path = eng.quest_root / "code" / "simulate.py"
    path.write_text(NO_SCALE, encoding="utf-8")
    fault = await eng._simulation_kept_or_back({}, path, SIMULATE, {}, "improve")
    assert fault.startswith("the analysis no longer runs")
    assert path.read_text(encoding="utf-8") == SIMULATE
    # the same text again (the entry check remembers it) is refused again, from the gate's own kept result
    path.write_text(NO_SCALE, encoding="utf-8")
    assert await eng._simulation_repair_fault({}, NO_SCALE) == fault
    # a repair that keeps what the analysis reads is kept
    path.write_text(SIMULATE.replace("1.0", "2.0"), encoding="utf-8")
    assert await eng._simulation_kept_or_back({}, path, SIMULATE, {}, "improve") == ""


@pytest.mark.asyncio
async def test_the_same_simulation_repair_is_kept_with_the_switch_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=False)

    async def entry_fine(state: Any, new_code: str) -> str:
        return ""

    monkeypatch.setattr(eng, "_simulation_repair_fault_uncached", entry_fine)
    path = eng.quest_root / "code" / "simulate.py"
    path.write_text(NO_SCALE, encoding="utf-8")
    assert await eng._simulation_kept_or_back({}, path, SIMULATE, {}, "improve") == ""
    assert path.read_text(encoding="utf-8") == NO_SCALE and not _gate_files(eng)


# --- the first gate (Engine._gate_pass) -----------------------------------------------------------------------------------


class _Reached(Exception):
    pass


def _before_the_gate(eng: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    async def none(*a: Any, **k: Any) -> None:
        return None

    for name in ("_results_from_computation", "_static_gate", "_small_run_gate"):
        monkeypatch.setattr(eng, name, none)
    monkeypatch.setattr(eng, "_cannot_repair_stop", lambda: None)
    monkeypatch.setattr(eng, "_code_problems_stop", lambda state: None)

    async def reached(*a: Any, **k: Any) -> None:
        raise _Reached

    monkeypatch.setattr(eng, "_code_review_gate", reached)


BROKEN_ANALYSIS = EXPERIMENT.replace("simulate.SCALE for", "simulate.SCALING for")


@pytest.mark.asyncio
async def test_a_first_version_whose_analysis_does_not_run_goes_to_the_crash_repair_and_a_good_one_goes_on(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=True)
    _before_the_gate(eng, monkeypatch)
    (eng.quest_root / "code" / "experiment.py").write_text(BROKEN_ANALYSIS, encoding="utf-8")
    stop, repaired, rewrote = await eng._gate_pass({"exec_reflect_iter": 0}, sys.executable, {}, eng.quest_root / "code" / "simulate.py")
    assert repaired is None and rewrote is False
    assert stop["exec_result"]["directive"] is True and stop["exec_result"]["failed_script"] == "experiment.py"
    assert "SCALING" in stop["exec_result"]["stderr_tail"] or "running the analysis" in stop["exec_result"]["stderr_tail"]
    assert eng._gate_last.kind == "fault"  # type: ignore[attr-defined]
    # no repair left: shown, not a stop
    with pytest.raises(_Reached):
        await eng._gate_pass({"exec_reflect_iter": 3}, sys.executable, {}, eng.quest_root / "code" / "simulate.py")
    # a version that runs goes on to the next check
    (eng.quest_root / "code" / "experiment.py").write_text(EXPERIMENT, encoding="utf-8")
    with pytest.raises(_Reached):
        await eng._gate_pass({"exec_reflect_iter": 0}, sys.executable, {}, eng.quest_root / "code" / "simulate.py")
    assert eng._gate_last.ok  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_first_gate_does_nothing_with_the_switch_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, gate=False)
    _before_the_gate(eng, monkeypatch)
    (eng.quest_root / "code" / "experiment.py").write_text(BROKEN_ANALYSIS, encoding="utf-8")
    with pytest.raises(_Reached):
        await eng._gate_pass({"exec_reflect_iter": 0}, sys.executable, {}, eng.quest_root / "code" / "simulate.py")
    assert not _gate_files(eng)


# --- the crash repair of experiment.py (Engine._node_execute_reflect) -------------------------------------------------------


async def _reflect(eng: Engine, new_code: str) -> tuple[dict[str, Any], str]:
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps({"code": new_code, "patch_summary": "fix"}))})()
    state = {"exec_result": {"returncode": 1, "stdout_tail": "", "failed_script": "experiment.py",
                             "stderr_tail": "ZeroDivisionError: division by zero"},
             "result_json": None, "code": EXPERIMENT, "design": {"hypothesis": "h"}}
    patch = await eng._node_execute_reflect(state)
    return patch, (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_crash_repair_of_experiment_py_that_breaks_the_analysis_is_not_used(tmp_path: Path) -> None:
    eng = _engine(tmp_path, gate=True)
    assert await eng._gate_first({}) is None
    broken = EXPERIMENT.replace("simulate.SCALE for", "simulate.SCALING for")
    patch, on_disk = await _reflect(eng, broken)
    assert on_disk == EXPERIMENT and not patch.get("exec_patch_pending")
    assert "the analysis no longer runs" in patch["exec_reflect_history"][0]["patch_summary"]
    good = EXPERIMENT.replace("+ 1.0", "+ 0.0")
    patch, on_disk = await _reflect(eng, good)
    assert on_disk == good and patch["exec_patch_pending"] is True


@pytest.mark.asyncio
async def test_the_same_crash_repair_is_written_with_the_switch_off(tmp_path: Path) -> None:
    eng = _engine(tmp_path, gate=False)
    broken = EXPERIMENT.replace("simulate.SCALE for", "simulate.SCALING for")
    patch, on_disk = await _reflect(eng, broken)
    assert on_disk == broken and patch["exec_patch_pending"] is True and not _gate_files(eng)


@pytest.mark.asyncio
async def test_a_gate_run_does_not_touch_the_code_history(tmp_path: Path) -> None:
    eng = _engine(tmp_path, gate=True)
    code = eng.quest_root / "code"
    subprocess.run(["git", "init", "-q"], cwd=code, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"], cwd=code, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "v1"], cwd=code, check=True)

    def log() -> str:
        return subprocess.run(["git", "log", "--all", "--oneline"], cwd=code, capture_output=True, text=True).stdout

    before = log()
    assert await eng._gate_first({}) is None
    assert log() == before
