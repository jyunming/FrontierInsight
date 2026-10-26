"""FI never asks a person to change code, a script or the quest's YAML to go on. A problem in the code is sent to the
existing repair with a plain directive; when the repairs are used up the quest ends in the honest stop (no paper), with
what FI tried. One test per kind of stop that used to ask for it. Fake model only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from core import run_manifest as rm
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine


def _cfg(tmp_path: Path, *, research: bool = True, **engine: Any) -> Config:
    cfg = Config(
        topic="a neutral topic about a cooling curve", title="cannot-repair", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return cfg.model_copy(update={"rigor_profile": "research"}) if research else cfg


def _engine(tmp_path: Path, **kw: Any) -> Engine:
    eng = Engine(_cfg(tmp_path, **kw))
    eng.quest_root.mkdir(parents=True, exist_ok=True)

    def asked(**kwargs: Any) -> None:
        raise AssertionError(f"a person was asked to go on: {kwargs.get('kind')}: {kwargs.get('steps')}")

    eng._pause_for_human = asked  # type: ignore[method-assign]
    return eng


async def _ends_in_the_honest_stop(eng: Engine, patch: dict[str, Any], said: str) -> dict[str, Any]:
    """The run step's patch -> the evidence gate's ruling -> the stop with its record."""
    result = patch["exec_result"]
    assert result["cannot_repair"] and said in result["cannot_repair"]
    ruled = eng._no_results_verdict({"exec_result": result, "result_json": patch.get("result_json", {}), "iteration": 1})
    assert ruled is not None and ruled["stuck"] is True and ruled["stuck_reason"] == "cannot_repair"
    out = await eng._node_stuck_no_findings({
        "evidence_assessment": {"stuck_reason": "cannot_repair"}, "exec_result": result, "exec_reflect_iter": 1})
    record = out["stuck"]
    assert said in record["problem"] and "did not remove it" in record["problem"]
    assert record["tried"] and (eng.quest_root / "needs" / "STUCK.json").is_file()
    return record


# --- labels and layout (research): repaired like a failed run, then the honest stop -------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("which", ["labels", "layout"])
async def test_a_missing_label_or_layout_is_sent_to_the_repair_then_recorded_and_the_quest_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str,
) -> None:
    eng = _engine(tmp_path)
    sentence = ("the plan's equation E1 is not marked in code/simulate.py" if which == "labels"
                else "the package code/model_pkg/ is missing")
    monkeypatch.setattr(eng, "_check_equation_labels", lambda state: [sentence] if which == "labels" else [])
    monkeypatch.setattr(eng, "_check_code_layout", lambda state: [sentence] if which == "layout" else [])
    patch = eng._code_problems_stop({"exec_reflect_iter": 0})
    assert patch is not None and "cannot_repair" not in patch["exec_result"]
    assert patch["exec_result"]["returncode"] == 1 and sentence in patch["exec_result"]["stderr_tail"]
    assert "Keep everything else as it is" in patch["exec_result"]["stderr_tail"]
    # What the repairs leave is read from the code: recorded, said in the paper, and the quest goes on.
    last = eng._code_problems_stop({"exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations})
    assert last is None and eng._cannot_repair_stop() is None
    assert sentence in eng._run_checks_note()


def test_without_research_a_missing_label_only_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, research=False)
    monkeypatch.setattr(eng, "_check_equation_labels", lambda state: ["E1 is not marked"])
    monkeypatch.setattr(eng, "_check_code_layout", lambda state: [])
    assert eng._code_problems_stop({"exec_reflect_iter": 0}) is None


# --- the run's record differs from the protocol -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_record_that_still_differs_after_the_repair_ends_in_the_honest_stop(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    found = ["the metric `rate` is a mean over 3500 successful trial(s) (3500), but the analysis's RESULT_JSON has no "
             "`rate_values` list at all"]
    eng._manifest_analysis_problems = list(found)  # type: ignore[attr-defined]
    patch = eng._manifest_stop(found, 1)
    record = await _ends_in_the_honest_stop(eng, patch, "rate_values")
    assert any("sent experiment.py back to be repaired 1 time(s)" in t for t in record["tried"])
    assert "Change `" not in json.dumps(record)


def test_the_analysis_repair_names_the_missing_lists_and_where_the_values_are() -> None:
    found = ["the metric `rate` is a mean over 3500 successful trial(s) (3500), but the analysis's RESULT_JSON has no "
             "`rate_values` list at all — x",
             "the metric `size` is a mean over 12 successful trial(s) (12), but the analysis's RESULT_JSON has no "
             "`size_values` list at all — x"]
    assert rm.missing_keys(found) == ["rate_values", "size_values"]
    text = rm.analysis_directive(found)
    assert "`rate_values`, `size_values`" in text and "FI_TRIALS" in text and '["values"]' in text
    assert "Rewrite experiment.py (not simulate.py)" in text
    assert "must be added" not in rm.analysis_directive(["the analysis printed something else"])


# --- the script differs from the plan's protocol / its randomness cannot be traced to the seed / split ----------------


@pytest.mark.asyncio
async def test_a_script_that_still_differs_from_the_protocol_is_recorded_and_the_quest_goes_on(tmp_path: Path) -> None:
    # The comparison reads the script's text, so a difference that stands is said in the paper, not a stop.
    eng = _engine(tmp_path)
    mismatch = mock.Mock()
    mismatch.message.return_value = "n_runs is 5 in simulate.py, line 3, but the protocol fixed 30"
    eng._stop_for_protocol([mismatch])
    assert eng._cannot_repair_stop() is None
    assert "n_runs is 5" in eng._run_checks_note()
    eng._forget_cannot_repair()  # the code is written again
    assert eng._cannot_repair_stop() is None


@pytest.mark.asyncio
async def test_a_reply_without_both_scripts_ends_in_the_honest_stop_in_block_mode(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.execution.split_failure = "block"
    eng._split_contract_broken(["the reply did not hold both scripts"], what="the code-writing step did not return both scripts",
                               source="measured")
    patch = eng._cannot_repair_stop()
    assert patch is not None
    await _ends_in_the_honest_stop(eng, patch, "did not hold both scripts")


# --- numeric warnings --------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_numeric_warnings_the_repairs_did_not_remove_end_in_the_honest_stop(tmp_path: Path) -> None:
    eng = _engine(tmp_path, numeric_warnings="block", exec_reflect_max_iterations=1)
    state = {
        "exec_result": {"returncode": 0, "numeric_warnings": [{"kind": "convergence", "text": "the solver did not converge"}]},
        "result_json": {"x": 1.5}, "exec_reflect_iter": 1, "iteration": 1,
    }
    patch = await eng._node_execute_reflect(state)  # type: ignore[arg-type]
    assert patch and "the solver did not converge" in patch["exec_result"]["cannot_repair"]
    await _ends_in_the_honest_stop(eng, patch, "the solver did not converge")
