"""How much of a quest's result has been checked against something other than itself (core/evidence.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import evidence, frozen_protocol
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "evidence_fr1"
PROTOCOL = {"runs_per_setting": 300, "oracles": [{"name": "closed form"}]}
ACCEPT = {"verdict": "accept", "must_flag_hits": []}


def _quest(tmp_path: Path, *, audits: bool = True, protocol_status: str | None = None, oracle_status: str | None = None,
          warnings: list[Any] | None = None, protocol: dict[str, Any] | None = PROTOCOL, freeze: bool = True,
          manifest_status: str | None = "ok") -> Path:
    root = tmp_path / "quest"
    (root / "needs").mkdir(parents=True)
    if freeze:  # the protocol the run was held to is the frozen one (None: the study froze without one)
        frozen_protocol.freeze(root, protocol, approved_by="test", source="plan.md")
    if audits:
        shutil.copytree(FIXTURE / "paper", root / "paper")
    if manifest_status and (protocol_status or oracle_status):  # a run whose gates passed also matched its manifest
        (root / "needs" / "RUN_MANIFEST_CHECK.json").write_text(json.dumps({"status": manifest_status}), encoding="utf-8")
    if protocol_status:
        (root / "needs" / "PROTOCOL_CHECK.json").write_text(json.dumps({"status": protocol_status}), encoding="utf-8")
    if oracle_status:
        (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": oracle_status, "judged_by": "engine"}), encoding="utf-8")
    if warnings is not None:
        (root / "needs" / "NUMERIC_WARNINGS.json").write_text(json.dumps(warnings), encoding="utf-8")
    return root


def _state(**over: Any) -> dict[str, Any]:
    return {"result_json": {"p": 0.3}, "exec_result": {"returncode": 0}, "design": {"hypothesis": "h", "protocol": PROTOCOL},
            "review": ACCEPT, **over}


ON = {"protocol_check": "block", "oracle_check": "block", "numeric_warnings": "block", "run_manifest_check": "block"}


def test_a_quest_that_has_not_run_has_only_the_first_gap(tmp_path: Path) -> None:
    got = evidence.assess(_quest(tmp_path, audits=False), {}, settings=ON)
    assert got["status"] == "not_executed" and got["next_level"] == "executed"
    assert got["levels"] == {level: False for level in evidence.LEVELS}
    assert got["gaps"] == ["the experiment has not produced results (it has not run, or it failed)"]


def test_the_run_of_the_audit_passed_every_check_it_had_and_is_still_only_internally_consistent(tmp_path: Path) -> None:
    """fr1 of the stored SIR runs: its number, statistics and provenance audits all passed, and nothing held it to a protocol
    or an oracle, so all that green said was that the paper copied what the script printed."""
    got = evidence.assess(_quest(tmp_path, protocol=None), _state(design={"hypothesis": "h"}), settings=ON)
    assert got["status"] == "internally_consistent" and got["next_level"] == "validated_against_oracle"
    assert got["levels"] == {"executed": True, "internally_consistent": True, "validated_against_oracle": False,
                             "publication_ready": False}
    assert any("plan fixes no protocol" in g for g in got["gaps"])


def test_an_audit_that_found_something_keeps_the_quest_at_executed(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    (root / "paper" / "numeric_audit.json").write_text(json.dumps({"ok": False, "findings": [1, 2]}), encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["status"] == "executed" and got["gaps"] == ["the number check reported 2 finding(s)"]
    empty = evidence.assess(_quest(tmp_path / "e", audits=False), _state(), settings=ON)
    assert empty["status"] == "executed" and empty["gaps"] == ["the paper has not been checked against the results yet"]


def test_a_skipped_audit_is_not_a_pass_and_not_a_failure(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    for name in ("numeric_audit", "statistics_audit", "provenance_audit"):
        (root / "paper" / f"{name}.json").write_text(json.dumps({"ok": True, "skipped": True}), encoding="utf-8")
    assert evidence.assess(root, _state(), settings=ON)["status"] == "executed"


def test_holding_to_the_protocol_passing_an_oracle_and_no_accepted_warning_validates_it(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["status"] == "publication_ready" and got["next_level"] is None and got["gaps"] == []
    assert all(got["levels"].values()) and got["all_gaps"] == {}


@pytest.mark.parametrize("protocol_status, oracle_status, expect", [
    (None, "ok", "protocol check: not run"),
    ("stopped", "ok", "protocol check: stopped"),
    ("warned", "ok", "protocol check: warned"),
    ("ok", None, "oracle check: not run"),
    ("ok", "warned", "oracle check: warned"),
    ("ok", "stopped", "oracle check: stopped"),
])
def test_a_check_that_was_not_passed_says_what_state_it_was_in(tmp_path: Path, protocol_status, oracle_status, expect) -> None:
    got = evidence.assess(_quest(tmp_path, protocol_status=protocol_status, oracle_status=oracle_status), _state(), settings=ON)
    assert got["status"] == "internally_consistent" and any(expect in g for g in got["gaps"]), got["gaps"]


def test_a_check_that_was_turned_off_is_a_gap_and_not_a_pass(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    for key in ("protocol_check", "oracle_check", "numeric_warnings"):
        got = evidence.assess(root, _state(), settings={**ON, key: "off"})
        assert got["status"] == "internally_consistent" and any("turned off" in g for g in got["gaps"]), key


def test_warnings_that_were_accepted_or_only_recorded_keep_it_from_validated(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", warnings=[{"warnings": [{"kind": "runtime"}]}])
    accepted = evidence.assess(root, _state(numeric_warnings_accepted=True), settings=ON)
    assert accepted["status"] == "internally_consistent" and any("accepted" in g for g in accepted["gaps"])
    recorded = evidence.assess(root, _state(), settings={**ON, "numeric_warnings": "warn"})
    assert recorded["status"] == "internally_consistent" and any("only recorded" in g for g in recorded["gaps"])
    repaired = evidence.assess(root, _state(), settings=ON)  # warned once, then repaired: block mode
    assert repaired["status"] == "publication_ready"


def test_a_missed_precision_target_and_an_unaccepted_review_keep_it_from_publication_ready(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    got = evidence.assess(root, _state(), precision_missed=["p"], settings=ON)
    assert got["status"] == "validated_against_oracle" and got["gaps"] == ["the target precision was not reached for p"]
    revise = evidence.assess(root, _state(review={"verdict": "revise", "must_flag_hits": [1]}), settings=ON)
    assert revise["status"] == "validated_against_oracle"
    assert revise["gaps"] == ["the review verdict is revise", "the review left 1 must-fix finding(s)"]
    unreviewed = evidence.assess(root, _state(review={}), settings=ON)
    assert unreviewed["gaps"] == ["the paper has not been reviewed yet"]


def test_a_quest_waiting_for_the_reviews_human_decision_is_not_publication_ready(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / ".fi").mkdir()
    (root / ".fi" / "human_review.json").write_text("{}", encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["status"] == "validated_against_oracle"
    assert got["gaps"] == ["the review is waiting for your decision (accept, reject or refine)"]
    (root / ".fi" / "human_review_answer.json").write_text("{}", encoding="utf-8")
    assert evidence.assess(root, _state(), settings=ON)["status"] == "publication_ready"


def test_a_level_needs_the_one_before_it(tmp_path: Path) -> None:
    """Passing the oracle does not make a run whose paper failed its audit 'validated'."""
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / "paper" / "provenance_audit.json").write_text(json.dumps({"ok": False, "findings": [1]}), encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["status"] == "executed" and not got["levels"]["validated_against_oracle"]
    assert "validated_against_oracle" not in got["gaps"] and got["next_level"] == "internally_consistent"


def test_the_one_line_for_the_cli_and_the_chat(tmp_path: Path) -> None:
    got = evidence.assess(_quest(tmp_path, protocol=None), _state(design={"hypothesis": "h"}), settings=ON)
    line = evidence.summary_line(got)
    assert line.startswith("internally_consistent; to reach validated_against_oracle: the plan fixes no protocol")
    assert evidence.summary_line({"status": "publication_ready", "gaps": []}) == "publication_ready"
    many = evidence.summary_line({"status": "executed", "next_level": "internally_consistent", "gaps": ["a", "b", "c"]})
    assert many == "executed; to reach internally_consistent: a (+2 more)"


# --- the engine writes it, and the surfaces show it ------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="smoke topic for evidence", title="evidence-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1, pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_a_finished_quest_leaves_its_evidence_and_says_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    record = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert record["levels"]["executed"] is True and record["status"] in ("executed", "internally_consistent")
    assert any("plan fixes no protocol" in g for g in record["all_gaps"]["validated_against_oracle"])
    assert not record["levels"]["validated_against_oracle"] and not record["levels"]["publication_ready"]
    assert "[evidence] " in (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_quest_that_held_to_a_protocol_and_passed_an_oracle_reaches_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ok_script = (
        "import os, json\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "if os.environ.get('FI_ORACLE') == '1':\n"
        "    print('ORACLE_JSON: ' + json.dumps({'checks': [{'name': 'closed form', 'value': 1.0}]}))\n    raise SystemExit(0)\n"
        "os.makedirs('figures', exist_ok=True)\nplt.figure(); plt.plot([0, 1], [0, 1]); plt.savefig('figures/result.png', dpi=72)\n"
        "print('RESULT_JSON: {\"score\": 0.987}')\n"
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = {"oracles": [{"name": "closed form", "check": "x", "expected": 1.0, "tolerance": 0.05}]}
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": ok_script, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    record = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert "validated_against_oracle" not in record["all_gaps"], record["all_gaps"]
    assert record["levels"]["validated_against_oracle"] == record["levels"]["internally_consistent"]


def test_the_quest_page_the_cli_summary_and_the_chat_carry_it() -> None:
    root = Path(__file__).resolve().parent.parent
    page = (root / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    for needle in ('id="evidence-banner"', "renderEvidence(data.evidence)", "validated_against_oracle", "publication_ready"):
        assert needle in page, needle
    assert '"evidence": _read_json_or_none' in (root / "web" / "server.py").read_text(encoding="utf-8")
    assert 'summary["evidence"] = evidence' in (root / "launch.py").read_text(encoding="utf-8")
    assert "evidence: (.+)" in (root / "vscode-frontier-insight" / "src" / "extension.ts").read_text(encoding="utf-8")


# --- the frozen protocol is what the level reads --------------------------------------------------------------------------


def test_the_frozen_protocol_counts_even_when_the_design_in_the_state_has_none(tmp_path: Path) -> None:
    """A redesign left the protocol out of the design in the state; the gates had held the run to the frozen one."""
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    got = evidence.assess(root, _state(design={"hypothesis": "redesigned, and no protocol"}), settings=ON)
    assert got["levels"]["validated_against_oracle"] is True


def test_a_run_that_was_never_frozen_cannot_be_validated_and_says_why(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", freeze=False)
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is False
    assert any("was not frozen before the run" in g for g in got["all_gaps"]["validated_against_oracle"])


def test_an_edited_freeze_record_is_a_gap(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    path = frozen_protocol.frozen_path(root)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["protocol"]["runs_per_setting"] = 3
    path.write_text(json.dumps(record), encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is False
    assert any("does not match its own SHA-256" in g for g in got["all_gaps"]["validated_against_oracle"])


def test_an_amendment_made_after_results_were_seen_keeps_it_from_publication_ready(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    pending = frozen_protocol.propose(root, {**PROTOCOL, "runs_per_setting": 600}, None, source="redesign", reason="more runs", results_seen=True)
    frozen_protocol.approve(root, "Jun", via="test")
    frozen_protocol.apply(root, pending, frozen_protocol.approval_for(root, pending), raw_root=root / "raw")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is True and got["levels"]["publication_ready"] is False
    assert any("amended after results were seen" in g and "runs_per_setting" in g for g in got["all_gaps"]["publication_ready"])


# --- the run manifest ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("status, expect", [
    ("single_script", "ran as one script, which writes no run manifest"),
    ("differs", "run manifest check: differs"),
    ("stopped", "run manifest check: stopped"),
    ("differs_in_replicates", "run manifest check: differs_in_replicates"),
    (None, "run manifest check: not run"),
])
def test_a_run_not_shown_to_match_its_manifest_cannot_be_validated(tmp_path: Path, status, expect) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", manifest_status=status)
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is False
    assert any(expect in g for g in got["all_gaps"]["validated_against_oracle"]), got["all_gaps"]


def test_switching_the_manifest_check_off_is_a_gap_not_a_pass(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    got = evidence.assess(root, _state(), settings={**ON, "run_manifest_check": "off"})
    assert any("run manifest check was turned off" in g for g in got["all_gaps"]["validated_against_oracle"])


def test_a_protocol_with_nothing_a_manifest_can_be_compared_with_does_not_ask_for_one(tmp_path: Path) -> None:
    only_oracles = {"oracles": [{"name": "closed form"}]}
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", protocol=only_oracles, manifest_status=None)
    got = evidence.assess(root, _state(design={"hypothesis": "h", "protocol": only_oracles}), settings=ON)
    assert got["levels"]["validated_against_oracle"] is True


def test_a_study_that_needs_no_manifest_is_not_asked_for_one(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok", manifest_status="not_applicable")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is True


def test_a_failed_trial_the_protocol_has_no_policy_for_is_a_gap_of_the_evidence_level(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / "needs" / "RUN_MANIFEST_CHECK.json").write_text(json.dumps({"status": "ok", "failed_trials": 3}), encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is False
    assert any("3 trial(s) failed" in g and "failure_policy" in g for g in got["all_gaps"]["validated_against_oracle"])
    frozen_with_policy = _quest(tmp_path / "b", protocol_status="ok", oracle_status="ok", protocol={**PROTOCOL, "failure_policy": "counted as failures"})
    (frozen_with_policy / "needs" / "RUN_MANIFEST_CHECK.json").write_text(json.dumps({"status": "ok", "failed_trials": 3}), encoding="utf-8")
    assert evidence.assess(frozen_with_policy, _state(), settings=ON)["levels"]["validated_against_oracle"] is True


def test_an_oracle_verdict_the_script_wrote_itself_is_not_independent_evidence(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    got = evidence.assess(root, _state(), settings=ON)
    assert got["levels"]["validated_against_oracle"] is False
    assert any("the script's own" in g for g in got["all_gaps"]["validated_against_oracle"])
