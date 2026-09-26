"""What a quest tried and under which conditions (core/attempt_records.py): the explore/confirm work's first phase.

Records only: the quest writes them as it runs and nothing that decides a route reads them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import attempt_records as ar
from core.config import Config
from core.engine import Engine
from tests.test_engine_smoke import _fake_response_for, smoke_config  # noqa: F401 -- the fixture


def test_a_run_s_outcome() -> None:
    assert ar.run_outcome(returncode=1, has_result=False) == "process_error"
    assert ar.run_outcome(returncode=0, has_result=False) == "process_error"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="repairing") == "protocol_mismatch"
    assert ar.run_outcome(returncode=0, has_result=True, oracle_status="failed") == "oracle_failure"
    assert ar.run_outcome(returncode=0, has_result=True, manifest_status="ok") == "inconclusive"


def test_a_quest_s_outcome_is_accepted_only_when_a_reviewer_accepted_it() -> None:
    ran = {"result_json": {"x": 1}}
    accepted = {"verdict": "accept", "status": "ok"}
    assert ar.quest_outcome({}, {"status": "not_executed"}) == "process_error"
    assert ar.quest_outcome({**ran, "review": accepted}, {"status": "executed"}) == "accepted_positive"
    assert ar.quest_outcome({**ran, "review": accepted, "analysis": {"hypothesis_supported": "refuted"}},
                            {"status": "executed"}) == "accepted_negative"
    assert ar.quest_outcome({**ran, "review": {"verdict": "accept", "status": "error"}, "analysis": {}},
                            {"status": "executed"}) == "inconclusive", "an accept no reviewer gave is not one"
    assert ar.quest_outcome({**ran, "analysis": {"verdict": "not supported by the data"}},
                            {"status": "executed"}) == "contradicted"
    assert ar.quest_outcome({**ran, "analysis": {"conclusion": "no significant difference"}},
                            {"status": "executed"}) == "null_result"
    assert all(o in ar.OUTCOMES for o in ("process_error", "accepted_positive", "null_result"))


def test_the_context_names_what_a_failure_depends_on(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "provider": {"name": "openai", "model": "m1"},
                                 "knowledge": {"enabled": False}})
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "ENVIRONMENT.json").write_text('{"packages": ["a==1"]}', encoding="utf-8")
    state = {"design": {"protocol": {"runs_per_setting": 30, "metrics": [{"id": "x"}]}}, "iteration": 1}
    ctx = ar.context_fingerprint(cfg, tmp_path, state, prompts={"design": "prompt text"})
    assert ctx["model"] == "m1" and ctx["provider"] == "openai"
    assert ctx["environment_sha256"] and ctx["metric_specs_sha256"] and ctx["budget"]["runs_per_setting"] == 30
    assert ctx["lineage"]["iteration"] == 1
    other = ar.context_fingerprint(cfg.model_copy(update={"provider": cfg.provider.model_copy(update={"model": "m2"})}),
                                   tmp_path, state, prompts={"design": "prompt text"})
    assert other["model"] != ctx["model"], "a different model is a different context"
    assert ar.context_fingerprint(cfg, tmp_path, state, prompts={"design": "changed"})["prompts_sha256"] != ctx["prompts_sha256"]


def test_records_append_and_a_torn_line_is_skipped(tmp_path: Path) -> None:
    assert ar.append(tmp_path, ar.ATTEMPTS, {"kind": "run", "outcome": "inconclusive"})
    with (tmp_path / ar.ATTEMPTS).open("a", encoding="utf-8") as fh:
        fh.write('{"torn": ')
    assert [r["kind"] for r in ar.read(tmp_path, ar.ATTEMPTS)] == ["run"]


@pytest.mark.asyncio
async def test_a_quest_records_its_ideas_design_runs_and_end(smoke_config: Config, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(smoke_config)
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    ledger = ar.read(engine.fi_dir, ar.LEDGER)
    kinds = [r["kind"] for r in ledger]
    assert "ideas" in kinds and "design" in kinds, kinds
    ideas = next(r for r in ledger if r["kind"] == "ideas")
    assert ideas["candidates"] and ideas["chosen"] and ideas["rule"]
    attempts = ar.read(engine.fi_dir, ar.ATTEMPTS)
    runs = [r for r in attempts if r["kind"] == "run"]
    (end,) = [r for r in attempts if r["kind"] == "quest"]
    assert runs and all(r["outcome"] in ar.OUTCOMES for r in runs)
    assert end["outcome"] in ar.OUTCOMES and end["context"]["provider"] == "openai"
    # Records only: nothing is written to the knowledge base, and the routes are unchanged.
    assert not (engine.quest_root / "needs" / "attempts.jsonl").exists()
