"""Going on although a known-answer check failed (core/accepted_checks.py): the person's named choice, or automatic for
an exploration.

A check that was measured and failed used to leave two ways on: fix the script, or (after the freeze) start a new
quest, or change `engine.oracle_check` -- which, on a quest the interview wrote, stopped it a second time. Now one step,
`--accept-checks <quest> --approve-as <you>` (the web page's button, `@fi /accept-checks`), marks the check unconfirmed
and goes on. The choice binds to the check's conditions and to the code that measured it: a change to either and the
check is judged again. It is never offered when nothing was measured. An exploration quest goes on by itself, recorded as
automatic; a research quest still stops. Either way the evidence keeps a gap in plain words and the paper says so.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from core import accepted_checks as ac, evidence, oracle_check as oc, todo
from core.engine import Engine
from tests.test_oracle_gate import _FAILING, _PROTOCOL, ORACLE, _cfg, _fake

ROOT = Path(__file__).resolve().parent.parent
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


def test_the_web_page_offers_it_only_while_the_quest_is_stopped_there(tmp_path: Path) -> None:
    """A stale record of the stop at the checks must not show the failed-check choice while the quest is stopped at the
    plan: the endpoint would record the plan's choice, and the person would have read one thing and signed another."""
    quest = tmp_path / "q-1"
    _stopped_record(quest)
    assert ac.failing_pending(quest) is not None
    (quest / ".fi" / "pause.json").write_text(json.dumps({"kind": "plan"}), encoding="utf-8")
    assert ac.failing_pending(quest) is None
    (quest / ".fi" / "pause.json").unlink()
    assert ac.failing_pending(quest) is not None
    ac.write_pending(quest, [{"name": "a", "why": "w", "expected": 1.0, "fingerprint": "f"}], plan_version=1)
    assert ac.failing_pending(quest) is None, "with no live stop, the plan's own record decides, as accept() does"


# --- the three surfaces record a name, the conditions and the code version ------------------------------------------


def test_the_cli_records_the_name_the_conditions_and_the_code_version(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    quest = tmp_path / "q-1"
    _stopped_record(quest, version="abc" * 20)
    assert launch._accept_checks("q-1", "", tmp_path) == 2, "a choice with no name is refused"
    assert launch._accept_checks("q-1", "Jun", tmp_path) == 0
    out = capsys.readouterr().out
    for needle in ("Jun goes on although 1 known-answer check(s) failed", "expected 1 within ±0.05",
                   "measured 0.5", "code/experiment.py (version abcabcabcabc)", "judged again"):
        assert needle in out, needle
    chosen = ac.failing_accepted(quest)["chosen"]["final size closed form"]
    assert chosen["by"] == "Jun" and chosen["via"] == "cli" and chosen["script_version"] == "abc" * 20
    assert chosen["fingerprint"] == ac.fingerprint(ORACLE) and chosen["measured"] == 0.5
    # The unsourced-checks stop is not what this quest stopped at: nothing written there.
    assert ac.accepted(quest) is None


def test_nothing_measured_is_refused_with_the_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import launch

    _stopped_record(tmp_path / "q-1", offered=False, why_not="nothing was measured for 'x'")
    assert launch._accept_checks("q-1", "Jun", tmp_path) == 1
    assert "not offered at this stop: nothing was measured for 'x'" in capsys.readouterr().out
    assert ac.failing_accepted(tmp_path / "q-1") is None


def test_the_web_page_records_it_through_the_same_endpoint(tmp_path: Path) -> None:
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover
        pytest.skip("fastapi not installed")
    from web.server import make_app

    _stopped_record(tmp_path / "q1")
    client = TestClient(make_app(tmp_path))
    assert client.post("/api/quests/q1/plan/accept-checks", json={"who": ""}).status_code == 400
    ok = client.post("/api/quests/q1/plan/accept-checks", json={"who": "Ana"})
    assert ok.status_code == 200 and "Ana goes on although" in ok.json()["message"]
    assert ac.failing_accepted(tmp_path / "q1")["chosen"]["final size closed form"]["via"] == "web"
    page = (ROOT / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "a.id === 'go_on_failing'" in page and "Go on with it marked unconfirmed" in page
    assert "failed.script_version" in page, "the box repeats the code the choice is bound to"
    assert "fi_accepted.failing_pending(quest_root)" in (ROOT / "web" / "server.py").read_text(encoding="utf-8")


def test_vs_code_turns_the_card_action_into_accept_checks(tmp_path: Path) -> None:
    ext = ROOT / "vscode-frontier-insight"
    skills = (ext / "src" / "skills.ts").read_text(encoding="utf-8")
    # The command reads the stop's offer and shows the conditions and the code before it asks for the name, and runs
    # the same launch.py --accept-checks.
    assert 'record.go_on' in skills and "failed.script_version" in skills and "failed.why_not" in skills
    assert '["--accept-checks", questId, "--approve-as", who.trim()' in skills
    compiled, source = ext / "out" / "stop-card.js", ext / "src" / "stop-card.ts"
    node = shutil.which("node")
    if node is None or not compiled.is_file() or compiled.stat().st_mtime < source.stat().st_mtime:
        pytest.skip("node, or a compiled vscode-frontier-insight/out/stop-card.js as new as its source, is not here")
    pause = {"kind": "oracle", "card": {"actions": [{"id": "go_on_failing", "label": "Mark the check unconfirmed and go on",
                                                     "detail": "d", "vscode": "@fi /accept-checks q-1"}]}}
    script = ("const m = require(process.argv[1]); const p = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "console.log(JSON.stringify(m.cardButtons(m.cardOf(p), 'q-1')));")
    done = subprocess.run([node, "-e", script, str(compiled)], input=json.dumps(pause), capture_output=True,
                          text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [{"title": "Mark the check unconfirmed and go on",
                                        "query": "@fi /accept-checks q-1", "send": True, "tooltip": "d"}]


# --- the choice binds to the check's conditions and the code -------------------------------------------------------


def test_a_changed_expected_value_or_changed_code_undoes_the_choice(tmp_path: Path) -> None:
    quest = tmp_path / "q-1"
    _stopped_record(quest, version="v1")
    assert ac.accept(quest, "Jun", via="cli")[0]
    (check,) = _offer()
    assert ac.went_on_by(quest, check, "v1")["by"] == "Jun"
    assert ac.went_on_by(quest, check, "v2") is None
    assert ac.no_longer_applies(quest, check, "v2") == "the code that measures it changed since"
    (moved,) = ac.offer(oc.problems([{**ORACLE, "expected": 0.9}], REPORTED, 0), [{**ORACLE, "expected": 0.9}],
                        oc.judged([{**ORACLE, "expected": 0.9}], REPORTED))[0]
    assert ac.went_on_by(quest, moved, "v1") is None
    assert "expected value, tolerance, case or measure changed" in ac.no_longer_applies(quest, moved, "v1")


def test_the_engine_judges_again_when_the_script_or_the_check_changed(tmp_path: Path) -> None:
    engine = Engine(_cfg(tmp_path))
    script = engine.quest_root / "code" / "experiment.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_FAILING, encoding="utf-8")
    _stopped_record(engine.quest_root, version=engine._measuring_code_version(script))
    assert ac.accept(engine.quest_root, "Jun", via="cli")[0]
    found, judged = oc.problems([ORACLE], REPORTED, 1), oc.judged([ORACLE], REPORTED)
    (went,) = engine._chosen_go_on(found, [ORACLE], judged, script)
    assert went["by"] == "Jun" and went["via"] == "cli"
    changed = {**ORACLE, "tolerance": 0.06}
    assert engine._chosen_go_on(oc.problems([changed], REPORTED, 1), [changed], oc.judged([changed], REPORTED), script) == []
    # A helper module beside the script is part of the code that measured the check; FI's own scripts are not.
    before = engine._measuring_code_version(script)
    (script.parent / "run.py").write_text("# FI's own runner\n", encoding="utf-8")
    assert engine._measuring_code_version(script) == before
    (script.parent / "helpers.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert engine._measuring_code_version(script) != before
    script.write_text(_FAILING + "\n# edited\n", encoding="utf-8")
    assert engine._chosen_go_on(found, [ORACLE], judged, script) == []
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "no longer applies: the code that measures it changed since; it is judged again" in log
    assert "no longer applies: its expected value, tolerance, case or measure changed since" in log


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
