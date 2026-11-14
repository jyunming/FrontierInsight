"""A study in its second stage (the design was chosen on data one stage saw and is now checked on new data) that cannot use
the search that chose its design ends in a plain stop: it says what happened and what to do, in words a scientist already
uses, and it never goes to the repair step (nothing in any script can change it). Neutral toy simulation; no real model.

Cases (core/optimise.py::run_search): a search of an older FI that this FI cannot check, no saved search, a search made for
another simulation or plan, a search that found no design, a search whose scan file is gone.
"""
from __future__ import annotations

from tests._engine_text import engine_text
import sys
from pathlib import Path
from typing import Any

import pytest

from core import optimise, trial_runner
from core.engine import Engine
from tests.test_optimise_runner import SIMULATE, LocalExecutor, _protocol
from tests.test_plan_optimisation import _engine

#: Words of FI's own that a person is never told: the second stage's internal names, and what the record is kept by.
INTERNAL = ("exploration", "confirm seeds", "confirm run", "frozen", "study key", "harness", "search code")


class Said:
    """A logger that keeps what run.log would be given."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, text: str, *args: Any) -> None:
        self.lines.append(text % args if args else text)

    info = warning


def _quest(tmp_path: Path) -> tuple[Engine, Path, dict[str, Any]]:
    eng = _engine(tmp_path, [])
    root = eng.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(
        SIMULATE.replace("__COUNT__", str(tmp_path / "count.txt")).replace("__CWD__", str(tmp_path / "cwd.txt")),
        encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    return eng, root, protocol


async def _search(root: Path, protocol: dict[str, Any], **kw: Any) -> optimise.SearchRun:
    return await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py", protocol, base_seed=0,
                                     timeout_s=300, **kw)


def _record(root: Path) -> dict[str, Any]:
    import json

    return json.loads((root / optimise.RECORD).read_text(encoding="utf-8"))


def _write_record(root: Path, record: dict[str, Any]) -> None:
    import json

    (root / optimise.RECORD).write_text(json.dumps(record), encoding="utf-8")


def _as_an_older_fi_wrote_it(root: Path, protocol: dict[str, Any]) -> None:
    """The record as an FI before the reuse keys wrote it: its one combined key, no reuse block."""
    block = optimise._plan.normalize(protocol["optimisation"])[0]
    record = _record(root)
    record.pop("reuse")
    record["study_key"] = optimise._legacy_study_key(root, root / "code" / "simulate.py", block, "run_cell", None)
    _write_record(root, record)


def _plain(said: str) -> None:
    low = said.lower()
    assert not any(word in low for word in INTERNAL), said
    assert said.endswith("."), said


# --- the messages ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_search_of_an_older_fi_that_this_fi_cannot_check_says_the_cause_and_what_to_do(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _eng, root, protocol = _quest(tmp_path)
    await _search(root, protocol)
    _as_an_older_fi_wrote_it(root, protocol)
    # FI was updated since: the harness the old record was made with is not the one this FI has
    monkeypatch.setattr(trial_runner, "LEGACY_HARNESS_SOURCE", trial_runner.LEGACY_HARNESS_SOURCE + "\n# another\n")
    log = Said()
    run = await _search(root, protocol, frozen=True, log=log)
    assert run.problems, "the stop stays: a frozen study keeps the rules it was written with"
    said = run.problems[0]
    assert "earlier version of FI" in said and "cannot be checked by this one" in said
    assert "finish the study with the FI version that started it, or start the study again" in said
    _plain(said)
    # FI's own detail (which record, which key) is in run.log, not in the message
    assert any("older FI" in line and optimise.RECORD.as_posix() in line for line in log.lines), log.lines
    assert run.frozen_stop is True and run.reused is False


@pytest.mark.asyncio
async def test_a_study_with_no_saved_search_says_so_and_what_to_do(tmp_path: Path) -> None:
    _eng, root, protocol = _quest(tmp_path)
    run = await _search(root, protocol, frozen=True)
    assert run.problems and "no saved result of the search" in run.problems[0]
    assert "start the study again" in run.problems[0]
    _plain(run.problems[0])
    assert run.frozen_stop is True


@pytest.mark.asyncio
async def test_a_search_made_for_another_simulation_says_so_and_what_to_do(tmp_path: Path) -> None:
    _eng, root, protocol = _quest(tmp_path)
    await _search(root, protocol)
    sim = root / "code" / "simulate.py"
    sim.write_text(sim.read_text(encoding="utf-8").replace("** 2", "** 3"), encoding="utf-8")
    run = await _search(root, protocol, frozen=True)
    assert run.problems and "no longer the one this study's search was made for" in run.problems[0]
    assert "put them back as they were, or start the study again" in run.problems[0]
    _plain(run.problems[0])
    assert run.frozen_stop is True


@pytest.mark.asyncio
async def test_a_search_that_found_no_design_says_so_and_what_to_do(tmp_path: Path) -> None:
    _eng, root, protocol = _quest(tmp_path)
    await _search(root, protocol)
    record = _record(root)
    record["ledger"]["text"] = ""  # no evaluation that worked is left in the record
    _write_record(root, record)
    run = await _search(root, protocol, frozen=True)
    assert run.problems and "found no design that worked" in run.problems[0]
    assert "start the study again" in run.problems[0]
    _plain(run.problems[0])
    assert run.frozen_stop is True


@pytest.mark.asyncio
async def test_a_search_whose_scan_file_is_gone_names_the_file_and_what_to_do(tmp_path: Path) -> None:
    import shutil

    _eng, root, protocol = _quest(tmp_path)
    first = await _search(root, protocol)
    assert first.scan
    (root / "raw" / "trials.json").unlink()
    shutil.rmtree(root / optimise.SCAN_DIR)
    run = await _search(root, protocol, frozen=True)
    assert run.problems and "raw/trials.json" in run.problems[0] and "is missing" in run.problems[0]
    assert "put the file back, or start the study again" in run.problems[0]
    _plain(run.problems[0])
    assert run.frozen_stop is True


@pytest.mark.asyncio
async def test_a_plan_that_cannot_drive_a_search_is_not_a_frozen_stop(tmp_path: Path) -> None:
    """That one is unchanged: a failed run with its own message, which is not the plain stop of a frozen study."""
    _eng, root, protocol = _quest(tmp_path)
    run = await optimise.run_search(LocalExecutor(), sys.executable, root, "code/simulate.py",
                                    {"optimisation": {"objective": {"quantity": "f"}}}, base_seed=0, timeout_s=60,
                                    frozen=True)
    assert run.problems and run.frozen_stop is False


# --- the quest ends there, and no repair is asked for ---------------------------------------------------------------


async def _failed_run(eng: Engine, root: Path, protocol: dict[str, Any]) -> optimise.OptimisationRunner:
    """The run step's runner, frozen for its second stage, run on a quest whose search was made by an older FI."""
    runner = optimise.OptimisationRunner(
        LocalExecutor(), quest_root=root, protocol=lambda: protocol, simulate=root / "code" / "simulate.py",
        analysis=root / "code" / "experiment.py", log=eng._log, frozen=lambda: True)
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60)
    assert result.returncode == 1 and runner.failed_script is None
    return runner


@pytest.mark.asyncio
async def test_the_stop_skips_the_repair_and_ends_the_quest_in_the_plain_words(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    eng, root, protocol = _quest(tmp_path)
    await _search(root, protocol)
    _as_an_older_fi_wrote_it(root, protocol)
    monkeypatch.setattr(trial_runner, "LEGACY_HARNESS_SOURCE", trial_runner.LEGACY_HARNESS_SOURCE + "\n# another\n")
    asked: list[Any] = []

    async def no_model(self: Any, *a: Any, **k: Any) -> str:
        asked.append(a)
        raise AssertionError("a repair was asked for a run that never started")

    monkeypatch.setattr("core.engine.LLMClient.chat", no_model)
    eng._client = type("Stub", (), {"chat": no_model})()
    runner = await _failed_run(eng, root, protocol)
    assert runner.plain_stop and "earlier version of FI" in runner.plain_stop

    stop = eng._stop_for_unusable_search(runner)
    assert stop is not None
    result = stop["exec_result"]
    state = {**stop, "iteration": 0, "exec_reflect_iter": 0, "result_json": {}}
    # the repair step does nothing and the route is the end, with no model asked
    assert await eng._node_execute_reflect(state) == {}
    assert eng._route_after_execute_reflect(state) == "stuck"
    verdict = eng._no_results_verdict(state)
    assert verdict is not None and verdict["stuck"] is True and verdict["stuck_reason"] == "no_search"
    out = await eng._node_stuck_no_findings({"evidence_assessment": {"stuck_reason": "no_search"}, "exec_result": result,
                                             "exec_reflect_iter": 0})
    assert asked == []
    said = capsys.readouterr().out
    assert "earlier version of FI" in said and "finish the study with the FI version that started it" in said
    for wrong in ("repair", "Nothing for you to fix", "revise-plan", "node_models", "so nothing was run for the study"):
        assert wrong not in said, said
    assert "earlier version of FI" in out["stuck"]["say"] and "repair" not in out["stuck"]["say"]
    assert (root / "needs" / "STUCK.json").is_file()


@pytest.mark.asyncio
async def test_a_run_that_stopped_for_another_reason_is_not_taken_for_this_one(tmp_path: Path) -> None:
    eng, root, protocol = _quest(tmp_path)
    runner = optimise.OptimisationRunner(
        LocalExecutor(), quest_root=root, protocol=lambda: protocol, simulate=root / "code" / "simulate.py",
        analysis=root / "code" / "experiment.py", log=eng._log)
    assert eng._stop_for_unusable_search(runner) is None  # nothing run yet
    assert eng._stop_for_unusable_search(LocalExecutor()) is None  # not a search runner


def test_the_run_step_ends_the_quest_before_it_could_retry_or_repair() -> None:
    """Wiring: the runner's plain stop is read right after the run, before the retry of a run that failed at once and
    before everything that follows a failed run (the repair step reads the run's record)."""
    src = engine_text()
    node = src[src.index("async def _node_execute_steps("):]
    after_run = node[node.index("runner.execute(\n"):]
    assert "self._stop_for_unusable_search(runner)" in after_run
    assert after_run.index("self._stop_for_unusable_search(runner)") < after_run.index("suspicious fast-fail")


# --- a record that main wrote is still read by this FI ----------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "main_search_record"
TOY = "def run_cell(cell):\n    return {'f': (cell['x'] - 2) ** 2 + (cell['y'] + 1) ** 2}\n"
TOY_BLOCK: dict[str, Any] = {
    "objective": {"quantity": "f", "direction": "minimise", "unit": "J"},
    "design_variables": [{"name": "x", "low": -5.0, "high": 5.0, "kind": "continuous"},
                         {"name": "y", "low": -5.0, "high": 5.0, "kind": "continuous"}],
    "fixed": {}, "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
    "evaluation_budget": {"starts": 1, "per_start": 6}, "search_method": "bounded_local"}


def _quest_with_a_record_of_main(tmp_path: Path, form: str) -> tuple[Path, dict[str, Any]]:
    """A quest whose search record is one FI's main wrote (a real search of this toy simulation, run by main's code before the
    recording of opened files, with the search file checked out with ``\n`` (``lf``) or ``\r\n`` (``crlf``))."""
    from core import optimisation_plan as op

    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(TOY, encoding="utf-8")
    (root / optimise.RECORD).parent.mkdir(parents=True)
    (root / optimise.RECORD).write_bytes((FIXTURES / f"{form}_run.json").read_bytes())
    block, why = op.normalize(dict(TOY_BLOCK))
    assert why is None, why
    return root, {"optimisation": block}


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["lf", "crlf"])
async def test_a_search_record_main_wrote_is_kept_by_the_confirm_stage(tmp_path: Path, form: str) -> None:
    """The legacy key is made from the texts the record was written with, not from today's files: this FI's harness and
    search code differ from main's, and the record of a search main ran must still be the study's own."""
    root, protocol = _quest_with_a_record_of_main(tmp_path, form)
    run = await _search(root, protocol, frozen=True)
    assert run.reused is True and not run.problems and run.ok_evaluations > 0, run.problems


@pytest.mark.asyncio
async def test_a_search_record_main_wrote_for_another_simulation_still_stops(tmp_path: Path) -> None:
    """The known positive: the same record, with the simulation changed since, is a stranger and says so plainly."""
    root, protocol = _quest_with_a_record_of_main(tmp_path, "lf")
    (root / "code" / "simulate.py").write_text(TOY.replace("** 2", "** 3"), encoding="utf-8")
    run = await _search(root, protocol, frozen=True)
    assert run.reused is False and run.problems and "earlier version of FI" in run.problems[0]


# --- the whole run step ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_run_step_of_a_study_with_no_saved_search_asks_no_model_and_ends_the_quest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real run step (``_node_execute``) of a study in its second stage with no saved search: before, the failed run went
    to the repair step, which asked the model to rewrite experiment.py (and went on to the analysis when it gave up)."""
    from unittest.mock import AsyncMock, MagicMock

    from core import phased
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    cfg = Config(topic="toy search", title="toy", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=2, review_loop=False, clarify_mode="off", phased=True),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
                 knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(kinds=["paper_md"], output_dir=tmp_path / "out"))
    eng = Engine(cfg)
    root = eng.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    eng.executor.install = AsyncMock(return_value=type("IR", (), {"returncode": 0, "stderr": ""})())  # type: ignore
    (root / "code" / "simulate.py").write_text(
        SIMULATE.replace("__COUNT__", str(tmp_path / "count.txt")).replace("__CWD__", str(tmp_path / "cwd.txt")),
        encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    eng._protocol_block = lambda state: protocol  # type: ignore[method-assign]
    phased.prepare(root, "q1")  # the second stage, and no search saved
    phased.seed_env(root, {"FI_REPLICATE_SEED": "0", "FI_REPLICATE_INDEX": "0"})
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    asked: list[Any] = []
    client = MagicMock()
    client.last_usage, client.last_model = None, "test-model"
    client.endpoint = MagicMock()
    client.endpoint.provider_name = "openai"

    async def chat(messages: Any, **kw: Any) -> str:
        asked.append(messages)
        return "{}"

    client.chat = chat
    eng._client = client
    state: dict[str, Any] = {"deps": [], "design": {"protocol": protocol}, "iteration": 0, "exec_reflect_iter": 0}
    out = await eng._node_execute(state)
    result = out["exec_result"]
    assert result["returncode"] == 1 and "no saved result of the search" in result["no_search"]
    asked.clear()
    after = {**state, **out}
    assert await eng._node_execute_reflect(after) == {}
    assert asked == [], "the repair step asked a model about a run that never started"
    assert eng._route_after_execute_reflect(after) == "stuck"
