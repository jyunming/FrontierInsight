"""Going on although a known-answer check failed (core/accepted_checks.py): FI goes on by itself, recorded as
automatic, the check marked unconfirmed. A person is never offered the choice. The evidence keeps a gap in plain words
and the paper says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import accepted_checks as ac, evidence, oracle_check as oc, todo
from core.engine import Engine
from tests.test_oracle_gate import _FAILING, _PROTOCOL, ORACLE, _cfg, _fake

REPORTED = {"checks": [{"name": ORACLE["name"], "value": 0.5}]}


def _offer() -> list[dict[str, Any]]:
    checks, why_not = ac.offer(oc.problems([ORACLE], REPORTED, 0), [ORACLE], oc.judged([ORACLE], REPORTED))
    assert checks and not why_not
    return checks


def _stopped_record(quest: Path, *, version: str = "v1", offered: bool = True, why_not: str = "") -> None:
    (quest / ".fi").mkdir(parents=True, exist_ok=True)
    (quest / "needs").mkdir(parents=True, exist_ok=True)
    (quest / ".fi" / "pause.json").write_text(json.dumps({"kind": "oracle"}), encoding="utf-8")
    record = {"status": "stopped", "judged_by": "engine", "problems": ["x"],
              "go_on": {"offered": offered, "why_not": why_not, "checks": _offer() if offered else [],
                        "script": "code/experiment.py", "script_version": version}}
    (quest / "needs" / "ORACLE_CHECK.json").write_text(json.dumps(record), encoding="utf-8")


# --- what may be offered -------------------------------------------------------------------------------------------


def test_only_a_check_that_was_measured_and_failed_is_offered() -> None:
    (check,) = _offer()
    assert check["measured"] == 0.5 and check["expected"] == 1.0 and check["limit"] == 0.05
    assert check["fingerprint"] == ac.fingerprint(ORACLE)
    # S5: the script crashed or printed nothing -- no number, so no failure to record.
    nothing, why = ac.offer(oc.problems([ORACLE], None, 1), [ORACLE], oc.judged([ORACLE], None))
    assert nothing == [] and why.startswith("nothing was measured for 'final size closed form'")
    # A problem besides the failed check (another check unmeasured) is not covered by going on with the failed one.
    other = {**ORACLE, "name": "second"}
    both = {"checks": [{"name": ORACLE["name"], "value": 0.5}]}
    nothing, why = ac.offer(oc.problems([ORACLE, other], both, 0), [ORACLE, other], oc.judged([ORACLE, other], both))
    assert nothing == [] and "'second'" in why
    assert ac.offer([], [], [])[0] == []
    # Two checks under one name: a failure cannot be told apart from the other's.
    twin = {**ORACLE, "expected": 0.5}
    nothing, why = ac.offer(oc.problems([ORACLE, twin], REPORTED, 0), [ORACLE, twin], oc.judged([ORACLE, twin], REPORTED))
    assert nothing == [] and "cannot tell them apart" in why


# --- the three surfaces record a name, the conditions and the code version ------------------------------------------


# --- a person is never offered the choice -------------------------------------------------------------------------


def test_a_person_cannot_go_on_with_a_failed_check_and_the_engine_has_no_such_path(tmp_path: Path) -> None:
    """An old stop at the checks no longer takes a person's choice: the command says the quest is not stopped for
    unsourced checks and writes nothing; the engine and the module have no path that reads such a choice."""
    quest = tmp_path / "q-1"
    _stopped_record(quest)
    ok, message = ac.accept(quest, "Jun", via="cli")
    assert not ok and "not stopped for checks without a stated source" in message
    assert not (quest / "needs" / "FAILED_CHECKS_ACCEPTED.json").exists()
    assert not any(hasattr(ac, n) for n in ("accept_failing", "failing_pending", "failing_accepted", "went_on_by"))
    assert not hasattr(Engine, "_chosen_go_on")


def test_the_measuring_code_version_follows_the_script_and_its_helper_modules(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    script = engine.quest_root / "code" / "experiment.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_FAILING, encoding="utf-8")
    # A helper module beside the script is part of the code that measured the check; FI's own scripts are not.
    before = engine._measuring_code_version(script)
    (script.parent / "run.py").write_text("# FI's own runner\n", encoding="utf-8")
    assert engine._measuring_code_version(script) == before
    (script.parent / "helpers.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert engine._measuring_code_version(script) != before


# --- the evidence, the to-do card and the paper ---------------------------------------------------------------------


def test_the_evidence_names_who_went_on_and_is_never_publication_ready() -> None:
    (check,) = _offer()
    person = ac.gap({**check, "by": "Jun"})
    assert person.startswith("Jun chose to go on although the known-answer check 'final size closed form' failed")
    auto = ac.gap({**check, "by": ac.AUTOMATIC})
    assert auto.startswith("FI went on by itself although the known-answer check") and "unconfirmed" in auto
    record = {"status": ac.WENT_ON, "went_on": [{**check, "by": "Jun"}]}
    assert "These known-answer checks FAILED and the run went on anyway: Jun chose" in ac.failing_disclosure(record)
    assert ac.failing_disclosure({"status": "ok"}) == ""
    assert oc.last_judged({"status": ac.WENT_ON, "attempts": [{"judged": oc.judged([ORACLE], REPORTED)}]}), \
        "the analysis is told the engine's verdicts on a run that went on"


@pytest.mark.asyncio
async def test_a_decision_quest_goes_on_by_itself_and_never_asks_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # Whether the check or the simulation is wrong is FI's to work out: after its repairs, the quest goes on with the
    # check marked unconfirmed, recorded as FI's decision; the result counts as exploratory and the paper says so.
    calls: list[str] = []
    prompts: list[str] = []
    fake = _fake(calls, implement=_FAILING, repair=_FAILING, protocol={**_PROTOCOL, "oracles": [ORACLE]})

    async def recording(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return await fake(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", recording)
    cfg = _cfg(tmp_path)  # result_use: decision
    engine = Engine(cfg)
    artifacts = await engine.run()
    assert calls.count("OracleRepair") == 2, "FI still tries its repairs first"
    assert artifacts.paper_md is not None and artifacts.paper_md.exists(), "no stop to ask"
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == ac.WENT_ON and record["went_on"][0]["by"] == ac.AUTOMATIC
    assert record["went_on"][0]["script"] == "code/experiment.py" and len(record["went_on"][0]["script_version"]) == 64
    assert record.get("explained", {}).get("why"), "why, in plain words, with the decision"
    out = capsys.readouterr().out
    assert "FI could not confirm 'final size closed form'" in out and "counts as exploratory" in out
    assert not (engine.quest_root / "NEXT_STEP.md").exists() or "known-answer" not in (
        engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    gaps = json.dumps(evidence.read(engine.quest_root) or {})
    assert "FI went on by itself although the known-answer check 'final size closed form' failed" in gaps
    level = (evidence.read(engine.quest_root) or {}).get("level")
    assert level not in ("independently_validated", "publication_ready"), "nothing about the check is relaxed"
    assert any("These known-answer checks FAILED and the run went on anyway" in p for p in prompts), \
        "the paper is told to say so"


@pytest.mark.asyncio
async def test_an_exploration_goes_on_by_itself_with_an_automatic_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat",
                        _fake(calls, implement=_FAILING, repair=_FAILING, protocol={**_PROTOCOL, "oracles": [ORACLE]}))
    explore = _cfg(tmp_path / "e").model_copy(update={"result_use": ""})  # unsaid: an exploration
    engine = Engine(explore)
    artifacts = await engine.run()
    assert calls.count("OracleRepair") == 2, "FI still tries its repairs first"
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == ac.WENT_ON and record["went_on"][0]["by"] == ac.AUTOMATIC
    assert "FI could not confirm" in capsys.readouterr().out
    items = todo.read(engine.fi_dir)
    assert any(i["kind"] == "went_on" and "FI went on by itself" in i["why"] for i in items), "listed on the to-do card"
    assert not (engine.quest_root / "NEXT_STEP.md").exists() or "Action needed" not in (
        engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert engine._goes_on_by_itself()


def test_every_quest_goes_on_by_itself_with_the_checks_on(tmp_path: Path) -> None:
    from core.config import Config, EngineConfig, OutputConfig

    def goes_on(**kw: Any) -> bool:
        return Engine(Config(topic="t", output=OutputConfig(output_dir=tmp_path / "o"), **kw))._goes_on_by_itself()

    assert goes_on() and goes_on(result_use="explore")
    assert goes_on(result_use="decision") and goes_on(result_use="research")
    assert goes_on(rigor_profile="research"), "a research quest too: no person is asked whether a check is right"
    assert not goes_on(engine=EngineConfig(oracle_check="warn")), "warn records the failure its own way"
