"""Under rigor_profile: research, the checks against known answers count as independent evidence only when a second,
different model read them (core/oracle_review.py::independence_gaps, read from the quest's record of its model calls)
and, for the kinds where it is well defined, they also held at a setting the code never saw (core/hidden_check.py).
Outside research nothing changes. No real model is called."""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

from core import evidence, hidden_check as hc, oracle_review as orv
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from core.execution import SharedInterpreterExecutor
from core.provider import LAST_CALL
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_oracle_review import PROTOCOL, REVIEW

USABLE = {"lines": ["", "- read"], "verdicts": [{"name": "power_conservation", "appropriate": True}],
          "call_id": "r1"}


def _row(node: str, model: str | None, *, reported: bool = True, call_id: str = "", outcome: str = "ok",
         provider: str = "openai") -> dict[str, Any]:
    return {"node": node, "served_model": model, "reported": reported, "call_id": call_id or f"{node}-{model}",
            "outcome": outcome, "provider": provider}


# --- the rule, on the record of the calls ------------------------------------------------------------------------------


def test_the_same_model_reading_its_own_checks_is_a_gap_that_says_how_to_name_another() -> None:
    calls = [_row("plan", "gpt-5.6-luna"), _row("oracle_review", "gpt-5.6-luna", call_id="r1")]
    gaps = orv.independence_gaps(USABLE, calls, configured=False)
    assert len(gaps) == 1 and gaps[0].startswith(orv.NOT_REVIEWED)
    assert "gpt-5.6-luna wrote them and read them again" in gaps[0]
    assert "`oracle_review: <another model your provider offers>` under `provider: node_models:`" in gaps[0]


def test_a_different_model_with_a_usable_answer_is_no_gap() -> None:
    calls = [_row("plan", "planner-model"), _row("plan_revise", "planner-model"),
             _row("oracle_review", "reviewer-model", call_id="r1")]
    assert orv.independence_gaps(USABLE, calls, configured=True) == []


@pytest.mark.parametrize("record, why", [
    ({"lines": ["x"], "error": "the call failed (TimeoutError)"}, "no usable answer (the call failed"),
    ({"lines": ["x"]}, "no usable answer (it judged none of the checks)"),
    ({}, "no second reading of them is recorded"),
    (None, "no second reading of them is recorded"),
])
def test_a_review_that_failed_or_never_ran_is_a_gap(record: Any, why: str) -> None:
    calls = [_row("plan", "planner-model"), _row("oracle_review", "reviewer-model", call_id="r1")]
    gaps = orv.independence_gaps(record, calls, configured=False)
    assert len(gaps) == 1 and why in gaps[0] and gaps[0].startswith(orv.NOT_REVIEWED)


@pytest.mark.parametrize("writer, reader", [
    ("gpt-5.6-luna", "gpt-5.6-luna-2026-09"),        # a dated id the connection reports
    ("claude-opus-4-5-20251101", "claude-opus-4-5"),
    ("openai/gpt-5", "GPT-5"),                         # a vendor prefix, another case
    ("gemma4:latest", "gemma4"),
    ("gemini-2.5-pro-preview-05-06", "gemini-2.5-pro"),
    ("gpt-5.6", "gpt_5_6"),
])
def test_the_same_model_under_another_name_is_the_same_model(writer: str, reader: str) -> None:
    assert orv.same_model(writer, reader)
    calls = [_row("plan", writer, provider="openai"), _row("oracle_review", reader, call_id="r1", provider="other")]
    assert "wrote them and read them again" in orv.independence_gaps(USABLE, calls, configured=True)[0]


@pytest.mark.parametrize("a, b", [("gpt-5", "gpt-5-mini"), ("claude-opus-4-5", "claude-sonnet-4-5"),
                                  ("llama3:8b", "llama3:70b"), ("gemini-2.5-pro", "gemini-2.5-flash")])
def test_different_models_stay_different(a: str, b: str) -> None:
    assert not orv.same_model(a, b)


def test_any_writer_of_the_checks_on_the_reader_s_model_is_a_gap() -> None:
    # The plan was written by one model, but a rewrite of it (FI's request about the checks) by the reader's own model.
    calls = [_row("plan", "planner-model"), _row("plan_revise", "reviewer-model"),
             _row("oracle_review", "reviewer-model", call_id="r1")]
    gaps = orv.independence_gaps(USABLE, calls, configured=True)
    assert gaps and "the model set for `oracle_review` is the one that wrote the plan" in gaps[0]


def test_a_model_the_connection_did_not_name_is_not_shown_different() -> None:
    reader_unnamed = [_row("plan", "planner-model"), _row("oracle_review", None, reported=False, call_id="r1")]
    assert "did not say which model read them" in orv.independence_gaps(USABLE, reader_unnamed, configured=True)[0]
    writer_unnamed = [_row("plan", None, reported=False), _row("oracle_review", "reviewer-model", call_id="r1")]
    assert "did not say which model wrote them" in orv.independence_gaps(USABLE, writer_unnamed, configured=True)[0]
    # A failed attempt of the plan is not a writer; the record of calls must still name the writer.
    no_writer = [_row("plan", "x", outcome="TimeoutError"), _row("oracle_review", "reviewer-model", call_id="r1")]
    assert "does not show which model wrote them" in orv.independence_gaps(USABLE, no_writer, configured=True)[0]


def test_the_reading_counted_is_the_one_the_record_names() -> None:
    # Two readings in the record (a rerun of the plan): the one .fi/oracle_review.json names is the one compared.
    calls = [_row("plan", "planner-model"), _row("oracle_review", "planner-model", call_id="old"),
             _row("oracle_review", "reviewer-model", call_id="r1")]
    assert orv.independence_gaps(USABLE, calls, configured=True) == []
    assert orv.independence_gaps({**USABLE, "call_id": "old"}, calls, configured=True)


# --- the engine: the reading at plan time, and the evidence that reads it ---------------------------------------------


def _config(tmp_path: Path, *, research: bool, node_models: dict[str, str] | None = None) -> Config:
    cfg = Config(
        topic="power through a lossless network", title="independence", provider=ProviderConfig(
            name="openai", model="planner-model", node_models=node_models or {}),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False),
    )
    return cfg.model_copy(update={"rigor_profile": "research"}) if research else cfg


def _fake(review: Any, *, fail_review: bool = False):
    async def chat(self, messages, **kw):  # noqa: ANN001
        # An HTTP API names the model that answered: the one asked for, else the config's.
        LAST_CALL.set({"provider": "openai", "model": kw.get("model") or "planner-model", "reported": True})
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith("# Second Opinion on the Checks"):
            if fail_review:
                raise TimeoutError("no answer")
            return json.dumps(review)
        if "You are revising the plan" in prompt:
            return prompt.split("# The plan as it stands", 1)[1].split("# What the person asked for", 1)[0].strip()
        if _classify(prompt) == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = PROTOCOL
            body["plan"] = {"in_short": "x", "literature": [], "gap": "g", "success_criteria": ["s"], "risks": ["r"],
                            "out_of_scope": ["o"]}
            return json.dumps(body)
        return _fake_response_for(prompt)

    return chat


async def _plan(engine: Engine) -> None:
    await engine._connect_llm()
    try:
        await engine._node_plan({"topic": engine.config.topic, "literature": []})
    finally:
        await engine._client.aclose()


def _review_gaps(engine: Engine) -> list[str]:
    return [g for g in engine._independence_gaps(PROTOCOL) if g.startswith(orv.NOT_REVIEWED)]


FINE = {**REVIEW, "equations_not_tested": [], "add": []}


@pytest.mark.asyncio
async def test_research_a_reviewer_on_another_model_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(FINE))
    engine = Engine(_config(tmp_path, research=True, node_models={"oracle_review": "reviewer-model"}))
    await _plan(engine)
    record = json.loads((engine.fi_dir / "oracle_review.json").read_text(encoding="utf-8"))
    assert record.get("call_id") and record["verdicts"]
    rows = [json.loads(line) for line in (engine.fi_dir / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["served_model"] for r in rows if r["node"] == "oracle_review"} == {"reviewer-model"}
    assert _review_gaps(engine) == []


@pytest.mark.asyncio
async def test_research_the_planner_s_own_model_is_a_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(FINE))
    engine = Engine(_config(tmp_path, research=True))
    await _plan(engine)
    gaps = _review_gaps(engine)
    assert len(gaps) == 1 and "planner-model wrote them and read them again" in gaps[0]
    assert "provider: node_models:" in gaps[0]
    assert "below *independently validated*" in (engine.quest_root / "plan.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_research_an_alias_of_the_planner_s_model_is_the_same_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def chat(self, messages, **kw):  # noqa: ANN001
        reply = await _fake(FINE)(self, messages, **kw)
        # The config names "reviewer-alias", and the connection reports that the planner's model answered it.
        if messages[-1]["content"].lstrip().startswith("# Second Opinion on the Checks"):
            LAST_CALL.set({"provider": "openai", "model": "planner-model-2026-09", "reported": True})
        return reply

    monkeypatch.setattr("core.engine.LLMClient.chat", chat)
    engine = Engine(_config(tmp_path, research=True, node_models={"oracle_review": "reviewer-alias"}))
    await _plan(engine)
    gaps = _review_gaps(engine)
    assert len(gaps) == 1 and "planner-model-2026-09 wrote them and read them again" in gaps[0]


@pytest.mark.asyncio
async def test_research_a_failed_review_goes_on_and_keeps_the_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(FINE, fail_review=True))
    engine = Engine(_config(tmp_path, research=True, node_models={"oracle_review": "reviewer-model"}))
    await _plan(engine)  # no stop, no crash
    assert (engine.quest_root / "plan.md").is_file()
    gaps = _review_gaps(engine)
    assert len(gaps) == 1 and "no usable answer" in gaps[0] and "node_models" not in gaps[0]
    assert "keeps the result below independently validated" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_outside_research_nothing_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(FINE, fail_review=True))
    engine = Engine(_config(tmp_path, research=False))
    await _plan(engine)
    assert engine._independence_gaps(PROTOCOL) == []
    assert "independently validated" not in (engine.quest_root / "plan.md").read_text(encoding="utf-8")


def test_the_gaps_hold_the_level_below_independently_validated(tmp_path: Path) -> None:
    from tests.test_engine_measured_oracle import _judged, _root
    from tests.test_evidence import ON, _state

    root = _root(tmp_path, _judged("engine"))
    assert evidence.assess(root, _state(), settings=ON)["levels"]["independently_validated"] is True
    got = evidence.assess(root, _state(), settings=ON, independence_gaps=[f"{orv.NOT_REVIEWED}: x"])
    assert got["levels"]["independently_validated"] is False
    assert any(g.startswith(orv.NOT_REVIEWED) for g in got["all_gaps"]["independently_validated"])


# --- a check at a setting the code never saw ----------------------------------------------------------------------------


INVARIANT = {"name": "power_conservation", "kind": "invariant", "check": "P_out equals P_in", "expected": 0,
             "tolerance": 1e-9, "case": {"n": 4}, "measure": "abs(P_out - P_in)"}
GRID_PROTOCOL = {"grid": {"n": [4, 8]}, "oracles": [INVARIANT]}
HONEST = "def run_cell(cell):\n    n = cell['n']\n    return {'P_in': float(n), 'P_out': float(n)}\n"
# Passes the case it was shown, and nowhere else.
SPECIAL_CASED = "def run_cell(cell):\n    return {'P_in': 1.0, 'P_out': 1.0 if cell['n'] == 4 else 1.25}\n"


def test_a_hidden_setting_is_well_defined_only_for_checks_whose_expected_value_holds_everywhere() -> None:
    rng = random.Random(0)
    case, changed = hc.derive(INVARIANT, GRID_PROTOCOL, rng)
    assert case == {"n": 8} and "n = 8" in changed
    stepped = {**INVARIANT, "kind": "symmetry", "case": {"n": 4, "dt": 0.1}}
    case, changed = hc.derive(stepped, GRID_PROTOCOL, rng)
    assert case["n"] == 4 and case["dt"] in (0.05, 0.1 / 3) and "dt" in changed
    special = {**INVARIANT, "kind": "special_case", "expected": 2.0}
    assert hc.candidates({"grid": {"n": [4, 8]}, "oracles": [special]}) == []
    nowhere = {**INVARIANT, "case": {"m": 3}}
    assert hc.derive(nowhere, GRID_PROTOCOL, rng)[0] is None
    # Never a setting another declared check already uses as its case (the code was shown that one).
    other = {**INVARIANT, "name": "at eight", "case": {"n": 8}}
    assert hc.derive(INVARIANT, {**GRID_PROTOCOL, "oracles": [INVARIANT, other]}, rng)[0] is None


def _quest(tmp_path: Path, simulate: str) -> Path:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    return root


def _run(root: Path, protocol: dict[str, Any], **kw: Any) -> dict[str, Any]:
    record = asyncio.run(hc.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, protocol,
                                timeout_s=60, env={}, engine_callable=True, rng=random.Random(1), **kw))
    hc.write(root, record)
    return record


def test_a_hidden_check_that_passes_is_no_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, HONEST)
    record = _run(root, GRID_PROTOCOL)
    assert record["status"] == "passed" and record["cases"][0]["case"] == {"n": 8}
    assert hc.evidence_gaps(root, GRID_PROTOCOL) == []
    # The code changed after it: the record no longer covers it.
    (root / "code" / "simulate.py").write_text(HONEST + "\n# changed\n", encoding="utf-8")
    assert "changed after FI ran its checks" in hc.evidence_gaps(root, GRID_PROTOCOL)[0]


def test_code_that_passes_only_the_case_it_was_shown_fails_the_hidden_check(tmp_path: Path) -> None:
    root = _quest(tmp_path, SPECIAL_CASED)
    record = _run(root, GRID_PROTOCOL)
    assert record["status"] == "failed" and record["cases"][0]["passed"] is False
    gaps = hc.evidence_gaps(root, GRID_PROTOCOL)
    assert len(gaps) == 1 and "passed at its own case but not at a setting the code never saw (n = 8" in gaps[0]


def test_a_hidden_check_that_cannot_run_is_a_gap_and_kinds_not_covered_are_none(tmp_path: Path) -> None:
    root = _quest(tmp_path, "def run_cell(cell):\n    if cell['n'] != 4:\n        raise ValueError('no')\n"
                            "    return {'P_in': 1.0, 'P_out': 1.0}\n")
    _run(root, GRID_PROTOCOL)
    assert "could not be run at a setting the code never saw" in hc.evidence_gaps(root, GRID_PROTOCOL)[0]
    special = {"grid": {"n": [4, 8]}, "oracles": [{**INVARIANT, "kind": "special_case", "expected": 2.0}]}
    record = _run(_quest(tmp_path / "s", HONEST), special)
    assert record["status"] == "not_covered" and "special or limiting case" in record["not_covered"][0]
    assert hc.evidence_gaps(tmp_path / "s" / "q", special) == []
    # A quest whose simulation FI cannot call on one case: said, and a gap.
    root = _quest(tmp_path / "n", HONEST)
    hc.write(root, asyncio.run(hc.run(None, sys.executable, root, GRID_PROTOCOL, timeout_s=5, env={},
                                      engine_callable=False)))
    assert "could not run the checks at a setting the code never saw" in hc.evidence_gaps(root, GRID_PROTOCOL)[0]
    # No record at all.
    assert "has not run the checks" in hc.evidence_gaps(tmp_path / "none", GRID_PROTOCOL)[0]


@pytest.mark.parametrize("simulate, passed", [(HONEST, True), (SPECIAL_CASED, False)])
def test_the_engine_runs_the_hidden_check_after_the_run_and_the_evidence_reads_it(
        tmp_path: Path, simulate: str, passed: bool) -> None:
    engine = Engine(_config(tmp_path, research=True, node_models={"oracle_review": "reviewer-model"}))
    engine.executor = SharedInterpreterExecutor(python_version="3.11")
    engine._trial_mode = True
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    state = {"design": {"protocol": GRID_PROTOCOL}}
    asyncio.run(engine._hidden_check(state, sys.executable, {}))
    record = hc.load(engine.quest_root)
    assert record is not None and record["cases"][0]["passed"] is passed
    hidden = [g for g in engine._independence_gaps(GRID_PROTOCOL) if "a setting the code never saw" in g]
    assert (hidden == []) is passed, hidden
    if not passed:
        assert "a setting the code never saw" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    # Outside research there is no hidden check and no such gap.
    plain = Engine(_config(tmp_path / "plain", research=False))
    assert plain._independence_gaps(GRID_PROTOCOL) == []
