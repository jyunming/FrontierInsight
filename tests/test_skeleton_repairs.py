"""Repairs keep the frame of simulate.py (core/skeleton.py): with FI's frame in the file, a repair may change the function
``trial`` and the helpers it uses and nothing FI wrote; a change to the frame is refused before it runs, with a plain
reason. Without a frame (the default) every repair path behaves as it did. Made-up topic: a colony's growth."""

from __future__ import annotations

import json
import string
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import finding_repair as fr
from core import improve, skeleton, trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

DESIGN: dict[str, Any] = {"protocol": {
    "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
    "metrics": [{"id": "growth", "kind": "mean"}]}}

BODY = '''
def _scale(x):
    return x * 1.0

def trial(cell, ctx):
    out = TrialResult()
    out["growth"] = _scale(cell["rate"] * cell["dose"])
    return out
'''

FRAMED = skeleton.fill(skeleton.frame_source(DESIGN, stochastic=True), BODY)[0]
assert FRAMED is not None

#: The same simulation as a model wrote it before there was a frame (the default today).
UNFRAMED = '''import random


def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    return {"growth": cell["rate"] * cell["dose"] + rng.random() * 0.0}
'''


# --- skeleton.repair_problem ---------------------------------------------------------------------------------------------

def test_an_honest_repair_of_the_trial_body_is_kept_and_a_change_to_the_frame_is_refused() -> None:
    honest = FRAMED.replace("return x * 1.0", "return x * 1.5")
    assert skeleton.repair_problem(FRAMED, honest, "simulate.py") == ""
    honest_trial = FRAMED.replace('out["growth"] = _scale(', 'out["growth"] = 0.5 + _scale(')
    assert skeleton.repair_problem(FRAMED, honest_trial, "simulate.py") == ""
    results = FRAMED.replace('RESULTS = (', 'RESULTS = (\'extra\', ')
    assert skeleton.repair_problem(FRAMED, results, "simulate.py").startswith("the frame FI wrote was changed")
    entry = FRAMED.replace("trial(cell, Ctx(seed=seed, trial=trial_id))", "{'growth': 0.0}")
    assert skeleton.repair_problem(FRAMED, entry, "simulate.py").startswith("the frame FI wrote was changed")
    body_of_entry = FRAMED.replace("return TrialResult.finish(", "return (")
    said = skeleton.repair_problem(FRAMED, body_of_entry, "simulate.py")  # names the line that differs, not the unchanged `def`
    assert "return TrialResult.finish(" in said and "is now `return trial(" in said
    params = FRAMED.replace("def trial(cell, ctx):", "def trial(cell, ctx, extra=1):")
    assert "parameters of `trial`" in skeleton.repair_problem(FRAMED, params, "simulate.py")


def test_a_whole_script_of_the_models_own_is_refused_when_the_file_has_a_frame() -> None:
    why = skeleton.repair_problem(FRAMED, UNFRAMED, "simulate.py")
    assert why.startswith("the frame FI wrote was changed") and "missing" in why
    after_the_frame = FRAMED + "\n\ndef run_trial(cell, trial_id, seed):\n    return {}\n"  # a second entry point
    assert "run_trial" in skeleton.repair_problem(FRAMED, after_the_frame, "simulate.py")


def test_nothing_is_checked_for_a_file_without_a_frame_or_for_another_file() -> None:
    assert skeleton.repair_problem(UNFRAMED, "anything at all", "simulate.py") == ""
    assert skeleton.repair_problem(FRAMED, "anything", "experiment.py") == ""
    assert skeleton.repair_note(UNFRAMED) == "" and skeleton.repair_note("") == ""
    note = skeleton.repair_note(FRAMED)
    assert note.startswith("FIRST") and "`trial(cell, ctx)`" in note and note.endswith("\n\n")


def test_fi_frame_py_is_never_edited_by_a_repair() -> None:
    assert skeleton.repair_problem(skeleton.RUNTIME, skeleton.RUNTIME, "fi_frame.py") == ""
    assert skeleton.repair_problem(skeleton.RUNTIME, skeleton.RUNTIME + "\nx = 1\n", "code/fi_frame.py").startswith(
        "the frame FI wrote was changed")


# --- the contract check every whole-script repair path asks ---------------------------------------------------------------

def test_the_contract_check_refuses_a_changed_frame_and_keeps_its_old_answers_without_one() -> None:
    honest = FRAMED.replace("return x * 1.0", "return x * 1.5")
    assert trial_runner.repair_breaks_contract(FRAMED, honest, "simulate.py") == ""
    why = trial_runner.repair_breaks_contract(FRAMED, UNFRAMED, "simulate.py")
    assert why.startswith("the frame FI wrote was changed")
    # No frame in the file: exactly the answers it gave before (a lost entry point, a stand-in word).
    assert trial_runner.repair_breaks_contract(UNFRAMED, UNFRAMED.replace("0.0", "0.5"), "simulate.py") == ""
    assert "does not define run_trial" in trial_runner.repair_breaks_contract(UNFRAMED, "x = 1\n", "simulate.py")
    assert trial_runner.repair_breaks_contract(UNFRAMED, "placeholder", "simulate.py").startswith("it holds no code")


# --- improve and the speed-up (one find/replace edit) ----------------------------------------------------------------------

def _edit(find: str, replace: str, file: str = "simulate.py") -> improve.Edit:
    return improve.Edit(file=file, find=find, replace=replace, why="test")


def test_an_edit_that_touches_the_frame_is_refused_before_it_runs_and_a_body_edit_is_not() -> None:
    files = {"simulate.py": FRAMED, "fi_frame.py": skeleton.RUNTIME}
    allowed = ["simulate.py"]
    new, why = improve.check_edit(files, _edit("return x * 1.0", "return x * 1.5"), allowed, tried=set(), expected=[])
    assert new is not None and why == ""
    new, why = improve.check_edit(files, _edit("TrialResult.declare(RESULTS, strict=False)", "pass"), allowed, tried=set(),
                                  expected=[])
    assert new is None and why.startswith("the frame FI wrote was changed")
    new, why = improve.check_edit(files, _edit("x = 1", "x = 2", "fi_frame.py"), allowed, tried=set(), expected=[])
    assert new is None and "fi_frame.py" in why


def test_without_a_frame_the_same_edit_is_judged_as_before() -> None:
    files = {"simulate.py": UNFRAMED}
    new, why = improve.check_edit(files, _edit("rng.random() * 0.0", "rng.random() * 0.5"), ["simulate.py"], tried=set(),
                                  expected=[])
    assert new is not None and why == ""


def test_fi_frame_py_is_not_among_the_files_an_edit_may_change(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text(FRAMED, encoding="utf-8")
    (code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
    assert improve.editable(tmp_path) == ["simulate.py"]


# --- the repair of a finding (test first, one function) ----------------------------------------------------------------------

class _Log:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


def _block(text: str) -> str:
    return f"```python\n{text}\n```"


class _Run:
    def __init__(self, tmp_path: Path, simulate: str, replies: dict[str, list[str]], function: str) -> None:
        self.root, self.code = tmp_path, tmp_path / "code"
        self.code.mkdir(parents=True)
        (self.code / "simulate.py").write_text(simulate, encoding="utf-8")
        if skeleton.has_frame(simulate):
            (self.code / "fi_frame.py").write_text(skeleton.RUNTIME, encoding="utf-8")
        self.record: dict[str, Any] = {"issues": {"F-1": {
            "id": "F-1", "kind": "quantity", "file": "simulate.py", "function": function, "text": "growth is wrong", "why": "w",
            "fix": "f", "severity": "high", "changes_result": True, "readings": [{"n": 1, "status": "open"}],
            "demonstrated": None, "outcome": "open", "repairs": [], "tests": 0,
            "calls": {"test": 0, "repair": 0, "other": 0}, "rounds": 0, "wordings": []}}, "readings": [{"n": 1}]}
        self.replies = {k: list(v) for k, v in replies.items()}
        self.prompts: list[str] = []
        prompts = {"code_finding_test": string.Template("TEST $finding\n$code\n$problem"),
                   "code_review_repair": string.Template("REPAIR $finding\n$code\n$test\n$failure\n$how\n$problem")}
        self.env = fr.Env(
            root=tmp_path, code=self.code, record=self.record, save=lambda: None, ask=self._ask, run_tests=self._run_tests,
            snapshot=self._snapshot, static=lambda: {"grid_keys": 0}, keeps=lambda old, new, rel: "", prompts=prompts,
            topic="a colony", equations="[]", scrub=lambda text: text, log=_Log(), package="", other="", code_model="writer")

    async def _ask(self, prompt: str, who: str, kind: str) -> str:
        self.prompts.append(prompt)
        queue = self.replies.get(f"{who}:{kind}") or []
        return queue.pop(0) if queue else ""

    async def _run_tests(self, paths: list[str]) -> dict[str, fr.TestRun]:
        out = {}
        for p in paths:
            done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(self.code), [str(self.root / p)])],
                                  capture_output=True, text=True, cwd=self.root, timeout=60)
            out[p] = fr.parse_runs(done.stdout, [str(self.root / p)])[str(self.root / p)]
        return out

    async def _snapshot(self) -> dict[str, str]:
        return {"equation:E1": "pass"}

    def go(self) -> list[str]:
        import asyncio

        return asyncio.run(fr.repair_issues(self.env))


TEST = ("import simulate\n\ndef test_growth():\n    out = simulate.trial({'rate': 2.0, 'dose': 3.0}, None)\n"
        "    assert out['growth'] == 7.0, 'growth lacks the offset'\n")
FIXED_TRIAL = BODY.replace('_scale(cell["rate"] * cell["dose"])', '1.0 + _scale(cell["rate"] * cell["dose"])')


def test_a_repair_of_the_trial_function_is_kept_in_a_framed_file_and_asked_with_the_note_first(tmp_path: Path) -> None:
    run = _Run(tmp_path, FRAMED, {"code:test": [_block(TEST)], "code:repair": [_block(FIXED_TRIAL)]}, "trial")
    assert run.go() == ["simulate.py"]
    assert run.record["issues"]["F-1"]["outcome"] == "fixed"
    text = (run.code / "simulate.py").read_text(encoding="utf-8")
    assert "1.0 + _scale(" in text and skeleton.frame_intact(FRAMED, text) == ""
    assert run.prompts[-1].startswith(skeleton.REPAIR_NOTE) and "REPAIR" in run.prompts[-1]


def test_a_whole_file_repair_that_changes_the_frame_is_refused_with_the_reason(tmp_path: Path) -> None:
    # Only the body of the entry function changes: no module-level statement and no other function, so the scope rules of the
    # finding repair let it through and the frame check alone refuses it.
    changed = FRAMED.replace("return TrialResult.finish(trial(cell, Ctx(seed=seed, trial=trial_id)))",
                             "return trial(cell, Ctx(seed=seed, trial=trial_id))")
    assert changed != FRAMED
    run = _Run(tmp_path, FRAMED, {"code:test": [_block(TEST)], "code:repair": [_block("# file: simulate.py" + chr(10) + changed)]}, "")
    assert run.go() == []
    why = run.record["issues"]["F-1"]["repairs"][0]
    assert why["outcome"] == "rejected" and why["why"].startswith("the frame FI wrote was changed")
    assert (run.code / "simulate.py").read_text(encoding="utf-8") == FRAMED


def test_without_a_frame_the_finding_repair_asks_and_accepts_exactly_as_before(tmp_path: Path) -> None:
    fixed = UNFRAMED.replace("0.0", "1.0")
    test = ("import simulate\n\ndef test_growth():\n    out = simulate.run_trial({'rate': 2.0, 'dose': 3.0}, 0, 1)\n"
            "    assert out['growth'] > 6.0, 'growth lacks the offset'\n")
    run = _Run(tmp_path, UNFRAMED, {"code:test": [_block(test)], "code:repair": [_block("# file: simulate.py\n" + fixed)]}, "")
    assert run.go() == ["simulate.py"]
    assert all(skeleton.REPAIR_NOTE not in p for p in run.prompts)
    assert (run.code / "simulate.py").read_text(encoding="utf-8") == fixed


# --- the repair after a crash of the simulation (execute_reflect) -----------------------------------------------------------

def _cfg(tmp_path: Path) -> Config:
    return Config(topic="repairs keep the frame", title="frame-repairs", provider=ProviderConfig(name="openai"),
                  engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False),
                  execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
                  knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"))


def _crash_state() -> dict[str, Any]:
    return {"exec_result": {"returncode": 1, "stdout_tail": "", "stderr_tail": "NameError: name 'offset' is not defined",
                            "failed_script": "simulate.py"},
            "result_json": None, "code": "print('x')", "design": {"hypothesis": "h"}, "deps": []}


def _engine(tmp_path: Path, simulate: str, reply: dict[str, Any]) -> tuple[Engine, AsyncMock]:
    eng = Engine(_cfg(tmp_path))
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(simulate, encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    chat = AsyncMock(return_value=json.dumps(reply))
    eng._client = type("Stub", (), {"chat": chat})()
    return eng, chat


@pytest.mark.asyncio
async def test_a_crash_repair_of_a_framed_simulation_keeps_the_frame_and_puts_the_note_first(tmp_path: Path) -> None:
    broken = FRAMED.replace('_scale(cell["rate"]', 'offset + _scale(cell["rate"]')
    fixed_fn = BODY
    eng, chat = _engine(tmp_path, broken, {"functions": fixed_fn, "patch_summary": "removed the undefined name"})
    patch = await eng._node_execute_reflect(_crash_state())
    on_disk = (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert patch["exec_reflect_iter"] == 1 and "offset" not in on_disk
    assert skeleton.frame_intact(FRAMED, on_disk) == ""
    prompt = chat.await_args.args[0][-1]["content"]
    assert prompt.startswith(skeleton.REPAIR_NOTE)


@pytest.mark.asyncio
async def test_a_crash_repair_that_rewrites_the_frame_is_refused_before_it_runs(tmp_path: Path) -> None:
    broken = FRAMED.replace('_scale(cell["rate"]', 'offset + _scale(cell["rate"]')
    whole = UNFRAMED  # the model's own whole script, with its own run_trial
    eng, _chat = _engine(tmp_path, broken, {"code": whole, "patch_summary": "wrote the script again"})
    patch = await eng._node_execute_reflect(_crash_state())
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == broken
    summary = patch["exec_reflect_history"][0]["patch_summary"]
    assert "the repair was not used" in summary and "the frame FI wrote was changed" in summary


@pytest.mark.asyncio
async def test_without_a_frame_a_crash_repair_is_asked_and_written_as_before(tmp_path: Path) -> None:
    broken = UNFRAMED.replace("cell[\"rate\"]", "offset + cell[\"rate\"]")
    fixed = UNFRAMED
    eng, chat = _engine(tmp_path, broken, {"code": fixed, "patch_summary": "removed the undefined name"})
    patch = await eng._node_execute_reflect(_crash_state())
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == fixed
    assert patch["exec_reflect_iter"] == 1
    assert skeleton.REPAIR_NOTE not in chat.await_args.args[0][-1]["content"]
