"""Acceptance of ``rigor_profile: research`` end to end: a validated research config, the fake model, the real graph.

Every other research test turns the profile on with ``model_copy``, which skips the validator, so what the profile forces
(the plan pause, the two scripts, an isolated interpreter, the review panel, the blocking checks) never ran together.
Here one quest runs through all of it and reaches ``publication_ready``; then each fault the 2026-09-26 re-audit named
is put into a copy of that quest (or run as its own quest) and must bring the evidence down, or stop the quest, with a
reason a person can read. Slow: each quest makes its own virtual environment.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import audit_log, plan_settings, receipts
from core.config import Config
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
from tests.test_run_manifest import ANALYSIS_TRIAL, PROTOCOL, SIM_TRIAL, _reply

pytestmark = pytest.mark.slow

METRIC = {"id": "final_size", "kind": "mean", "estimand": "mean final epidemic size", "unit": "trial"}
RESEARCH_PROTOCOL = {
    **PROTOCOL,
    "oracles": [{"name": "half", "check": "a known case", "expected": 0.5, "tolerance": 0.01}],
    "metrics": [METRIC],
    "contrasts": [{"metric": "final_size", "a": "R0=0.9", "b": "R0=3.0"}],
}
ORACLE = '\n\ndef oracle():\n    return {"half": 0.5004}\n'
SIM = SIM_TRIAL + ORACLE


def _config(root: Path, **over: Any) -> Config:
    """A research config built the way a YAML is: through the validator, so the profile's settings are applied."""
    data: dict[str, Any] = {
        "topic": "smoke topic for the research acceptance run", "title": "research-acceptance",
        "rigor_profile": "research",
        "provider": {"name": "openai"},
        "engine": {"max_iterations": 1, "review_loop": False, "auto_accept_on_pass": True, "execute_replicates": 3,
                   "pilot_run": False},
        "execution": {"sandbox": "venv", "timeout_s": 300},
        "knowledge": {"enabled": False},
        "output": {"output_dir": str(root)},
        **over,
    }
    return Config.model_validate(data)


def _fake(protocol: dict[str, Any], simulate: str, analysis: str, calls: list[str]):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(simulate, analysis)
        if prompt.lstrip().startswith("**Persona:"):  # a review-panel member
            return _FAKE_RESPONSES["review"]
        return _fake_response_for(prompt)

    return fake_chat


def _run_through_pauses(config: Config, *, quest_id: str | None = None, from_step: str | None = None,
                        passes: int = 5, interview: bool = False) -> tuple[Engine, Any, list[str]]:
    """Run a quest, answering the pauses a person would (read the plan, then go on); returns the last engine, its
    artifacts and the headline of each pause met on the way. ``interview``: the quest's config.yaml is the one the
    interview writes, whose settings FI records as approved (a hand-written YAML is its own record)."""
    pauses: list[str] = []
    engine = Engine(config, resume_quest_id=quest_id) if quest_id else Engine(config)
    if interview:
        import yaml

        engine.quest_root.mkdir(parents=True, exist_ok=True)
        (engine.quest_root / "config.yaml").write_text(
            f"{plan_settings.INTERVIEW_MARK} Frontier Insight\n" + yaml.safe_dump(config.model_dump(mode="json")),
            encoding="utf-8",
        )
    artifacts = asyncio.run(engine.run(from_step=from_step) if from_step else engine.run())
    for _ in range(passes):
        nxt = engine.quest_root / "NEXT_STEP.md"
        if not nxt.is_file():
            break
        text = nxt.read_text(encoding="utf-8")
        pauses.append(text.splitlines()[0])
        if "read and edit the plan" not in text:
            break  # anything else is a stop the test is about
        nxt.unlink()
        engine = Engine(config, resume_quest_id=engine.quest_id)
        artifacts = asyncio.run(engine.run())
    return engine, artifacts, pauses


def _evidence(root: Path) -> dict[str, Any]:
    return json.loads((root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))


def _all_gaps(record: dict[str, Any]) -> list[str]:
    return [g for gaps in (record.get("all_gaps") or {}).values() for g in gaps] + list(record.get("gaps") or [])


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("research")
    calls: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("core.engine.LLMClient.chat", _fake(RESEARCH_PROTOCOL, SIM, ANALYSIS_TRIAL, calls))
        config = _config(root)
        engine, artifacts, pauses = _run_through_pauses(config, interview=True)
    return {"config": config, "engine": engine, "artifacts": artifacts, "pauses": pauses, "calls": calls, "root": root}


def _copy(baseline: dict[str, Any], dest: Path) -> tuple[Config, Path]:
    """A copy of the finished quest in its own output folder, and the config pointing at it."""
    src = baseline["engine"].quest_root
    target = dest / src.name
    shutil.copytree(src, target, ignore=shutil.ignore_patterns(".venv"))
    return baseline["config"].model_copy(update={"output": baseline["config"].output.model_copy(
        update={"output_dir": dest})}), target


def _reassess(baseline: dict[str, Any], config: Config, quest_id: str, **state_over: Any) -> dict[str, Any]:
    """The evidence of a (damaged) copy, assessed by the engine's own method with the quest's final state."""
    engine = Engine(config, resume_quest_id=quest_id)
    state = {**dict(baseline["artifacts"].raw_state), **state_over}
    record = engine._write_evidence(state)
    assert record is not None
    return record


# --- the whole profile, once ----------------------------------------------------------------------------------------


def test_a_research_quest_reaches_publication_ready_only_through_every_gate(baseline: dict[str, Any]) -> None:
    config, engine, artifacts = baseline["config"], baseline["engine"], baseline["artifacts"]
    root = engine.quest_root
    # What the profile forces, applied by the validator (not by a shortcut).
    assert config.pauses.plan == "ask" and config.execution.split_analysis is True
    assert config.execution.shared_interpreter is False and config.engine.oracle_check == "block"
    assert {"methodologist", "statistician", "reproducibility"} <= set(config.engine.review_panel)
    # It stopped for the plan first, and nothing else stopped it.
    assert baseline["pauses"] == ["# Action needed — read and edit the plan"], baseline["pauses"]
    assert artifacts.paper_md is not None
    # The oracle passed before the main run; FI ran every trial and wrote the ledger itself.
    assert json.loads((root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))["status"] == "ok"
    trials = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [t for t in trials if t.get("event") == "trial"]
    assert len(trials) == 900 and len({t["seed"] for t in trials}) == 900
    # The environment the experiment ran on, isolated and recorded after its installs.
    env = json.loads((root / "needs" / "ENVIRONMENT.json").read_text(encoding="utf-8"))
    assert env["isolated"] is True and env["recorded"].startswith("after installing") and "packages" in env
    # Each check left a passing receipt, the trace still verifies, and the review left no stale "waiting" behind.
    for check in ("evidence_gate", "claim_check", "design_audit"):
        status, record, problem = receipts.read(root, check)
        assert status == "pass", (check, status, problem)
    assert audit_log.verify(root / ".fi" / "audit.jsonl").ok
    assert not (root / ".fi" / "human_review.json").exists()
    # The settings the interview approved are recorded, and their hash is in the trace.
    assert (root / ".fi" / plan_settings.NAME).is_file()
    assert any(e.get("kind") == "plan_settings_recorded" for e in audit_log.read(root / ".fi" / "audit.jsonl"))
    record = _evidence(root)
    assert record["status"] == "publication_ready", record.get("gaps")


# --- faults put into a copy of the finished quest ---------------------------------------------------------------------


@pytest.mark.parametrize("check", ["evidence_gate", "claim_check", "design_audit"])
@pytest.mark.parametrize("damage", ["deleted", "corrupt"])
def test_a_missing_or_corrupt_receipt_is_a_gap(baseline: dict[str, Any], tmp_path: Path, check: str,
                                                damage: str) -> None:
    config, root = _copy(baseline, tmp_path)
    path = receipts.path(root, check)
    if damage == "deleted":
        path.unlink()
    else:
        path.write_text("{not json", encoding="utf-8")
    record = _reassess(baseline, config, root.name)
    assert record["status"] != "publication_ready"
    assert any(check.replace("_", " ") in g or check in g for g in _all_gaps(record)), _all_gaps(record)


def test_an_evidence_gate_receipt_for_an_earlier_analysis_is_a_gap(baseline: dict[str, Any], tmp_path: Path) -> None:
    config, root = _copy(baseline, tmp_path)
    analysis = {**dict(baseline["artifacts"].raw_state.get("analysis") or {}), "headline": "changed after the gate"}
    record = _reassess(baseline, config, root.name, analysis=analysis)
    assert record["status"] != "publication_ready"
    assert any("evidence gate judged earlier inputs" in g for g in _all_gaps(record))


def test_a_package_list_that_could_not_be_made_is_a_gap(baseline: dict[str, Any], tmp_path: Path) -> None:
    config, root = _copy(baseline, tmp_path)
    path = root / "needs" / "ENVIRONMENT.json"
    env = json.loads(path.read_text(encoding="utf-8"))
    env.pop("packages", None)
    env["packages_error"] = "pip freeze exited 1"
    path.write_text(json.dumps(env), encoding="utf-8")
    record = _reassess(baseline, config, root.name)
    assert record["status"] != "publication_ready"
    assert any("could not be listed" in g for g in _all_gaps(record))


def test_an_exploration_is_never_publication_ready(baseline: dict[str, Any], tmp_path: Path) -> None:
    config, root = _copy(baseline, tmp_path)
    record = _reassess(baseline, config.model_copy(update={"result_use": "explore"}), root.name)
    assert record["status"] != "publication_ready"
    assert any("set up to explore" in g for g in _all_gaps(record))


@pytest.mark.parametrize("edit", ["corrupt", "weaker"])
def test_an_edited_record_of_the_approved_settings_stops_the_next_resume(
    baseline: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: str,
) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(RESEARCH_PROTOCOL, SIM, ANALYSIS_TRIAL, []))
    config, root = _copy(baseline, tmp_path)
    record = root / ".fi" / "approved_plan.json"
    if edit == "corrupt":
        record.write_text("{not json", encoding="utf-8")
    else:
        data = json.loads(record.read_text(encoding="utf-8"))
        data["settings"]["engine.oracle_check"] = "warn"
        record.write_text(json.dumps(data), encoding="utf-8")
    engine = Engine(config, resume_quest_id=root.name)
    asyncio.run(engine.run())
    text = (root / "NEXT_STEP.md").read_text(encoding="utf-8")
    # Either is a stop before anything runs: the record's hash in the trace no longer matches (checked first), or the
    # record cannot be read.
    assert "changed after you approved it" in text and ("changed after it was approved" in text or "cannot be read" in text), text[:800]


def test_a_clean_rerun_from_the_run_redoes_the_trials_and_keeps_the_old_outputs(
    baseline: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(RESEARCH_PROTOCOL, SIM, ANALYSIS_TRIAL, calls))
    config, root = _copy(baseline, tmp_path)
    engine, artifacts, pauses = _run_through_pauses(config, quest_id=root.name, from_step="run")
    assert pauses == [] and artifacts.paper_md is not None
    assert "Implementation" not in calls, "the code is kept: the run starts from the same scripts"
    previous = list((root / ".fi" / "previous").iterdir())
    assert len(previous) == 1 and (previous[0] / "raw" / "ledger.jsonl").is_file()
    assert (root / "raw" / "ledger.jsonl").is_file()
    assert _evidence(root)["status"] == "publication_ready", _evidence(root).get("gaps")
    assert audit_log.verify(root / ".fi" / "audit.jsonl").ok


# --- faults that need a quest of their own ----------------------------------------------------------------------------

PAIRED = {**RESEARCH_PROTOCOL, "metrics": [{**METRIC, "paired": True}]}
ANALYSIS_WITH_PAIR_IDS = ANALYSIS_TRIAL.replace(
    '"final_size_count": m["count"]}', '"final_size_count": m["count"], "final_size_pair_id": m["trials"]}')


def test_a_paired_metric_printed_without_pair_ids_stops_the_research_quest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(PAIRED, SIM, ANALYSIS_TRIAL, []))
    engine, artifacts, pauses = _run_through_pauses(_config(tmp_path))
    assert pauses and "read and edit the plan" not in pauses[-1], pauses
    text = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "final_size_pair_id" in text, text[:1200]
    assert _evidence(engine.quest_root)["status"] != "publication_ready"


def test_a_paired_metric_with_pair_ids_pairs_trials_that_share_a_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(PAIRED, SIM, ANALYSIS_WITH_PAIR_IDS, []))
    engine, artifacts, pauses = _run_through_pauses(_config(tmp_path))
    assert artifacts.paper_md is not None and pauses == ["# Action needed — read and edit the plan"], pauses
    root = engine.quest_root
    trials = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    seeds: dict[int, set[int]] = {}
    for t in (t for t in trials if t.get("event") == "trial"):
        seeds.setdefault(t["trial"], set()).add(t["seed"])
    assert seeds and all(len(s) == 1 for s in seeds.values()), "trial i of every setting gets one seed"
    record = _evidence(root)
    assert record["status"] == "publication_ready", record.get("gaps")
    assert not any("by their position" in g for g in _all_gaps(record))


SIM_SOME_FAIL = SIM_TRIAL.replace(
    "    rng = random.Random(seed)\n",
    "    if trial_id % 50 == 7:\n        raise RuntimeError('this trial did not converge')\n    rng = random.Random(seed)\n",
) + ORACLE


def test_trials_that_fail_with_no_failure_policy_are_a_gap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(RESEARCH_PROTOCOL, SIM_SOME_FAIL, ANALYSIS_TRIAL, []))
    engine, artifacts, pauses = _run_through_pauses(_config(tmp_path))
    root = engine.quest_root
    trials = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [t for t in trials if t.get("event") == "trial" and t.get("status") != "ok"]
    assert len(failed) == 18, "6 of the 300 trials in each of the 3 settings"
    record = _evidence(root)
    assert record["status"] != "publication_ready"
    assert any("failure_policy" in g for g in _all_gaps(record)), _all_gaps(record)
