"""The frozen protocol and its amendments (core/frozen_protocol.py), and what the engine does with them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import frozen_protocol as fp
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

ORACLE = {"name": "final size closed form", "kind": "closed_form", "check": "small-N final size against the closed form", "expected": 1.0, "tolerance": 0.05}
P1 = {"runs_per_setting": 300, "seed_policy": "an independent stream per setting", "oracles": [ORACLE]}
P2 = {"runs_per_setting": 500, "seed_policy": "common random numbers across settings", "oracles": [ORACLE]}


# --- the record ---------------------------------------------------------------------------------------------------------


def test_the_hash_does_not_depend_on_key_order() -> None:
    assert fp.sha256({"a": 1, "b": [1, 2]}) == fp.sha256({"b": [1, 2], "a": 1})
    assert fp.sha256({"a": 1}) != fp.sha256({"a": 2})


def test_freezing_records_who_when_and_the_hash_and_is_idempotent(tmp_path: Path) -> None:
    assert fp.load(tmp_path) is None and fp.protocol_of(tmp_path) is None
    first = fp.freeze(tmp_path, P1, approved_by="human: test", source="plan.md")
    assert first["sha256"] == fp.sha256(P1) and first["run_id"] == "run_1" and first["version"] == 1 and first["amendments"] == 0
    again = fp.freeze(tmp_path, P2, approved_by="someone else", source="elsewhere")
    assert again["sha256"] == fp.sha256(P1) and again["approved_by"] == "human: test"  # a second freeze changes nothing
    assert fp.protocol_of(tmp_path) == P1
    assert (tmp_path / "needs" / "protocol_versions" / "v1.json").is_file()


def test_a_study_frozen_without_a_protocol_says_so_and_returns_none(tmp_path: Path) -> None:
    record = fp.freeze(tmp_path, None, approved_by="auto", source="plan.md")
    assert record["protocol"] is None and fp.protocol_of(tmp_path) is None and fp.load(tmp_path) is not None


def test_an_edited_record_is_reported_and_its_protocol_is_still_what_it_says(tmp_path: Path) -> None:
    fp.freeze(tmp_path, P1, approved_by="auto", source="plan.md")
    path = fp.frozen_path(tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["protocol"]["runs_per_setting"] = 30
    path.write_text(json.dumps(record), encoding="utf-8")
    loaded = fp.load(tmp_path)
    assert "does not match its own SHA-256" in loaded["problem"] and loaded["protocol"]["runs_per_setting"] == 30


def test_what_changed_is_one_line_per_key() -> None:
    lines = fp.diff(P1, P2)
    assert "runs_per_setting: 300 -> 500" in lines
    assert 'seed_policy: "an independent stream per setting" -> "common random numbers across settings"' in lines
    assert fp.diff(P1, P1) == []
    assert fp.diff(None, {"a": 1}) == ["a: <absent> -> 1"]


# --- amendments ---------------------------------------------------------------------------------------------------------


def _quest_with_results(tmp_path: Path) -> Path:
    (tmp_path / "raw" / "seed0").mkdir(parents=True)
    (tmp_path / "raw" / "seed0" / "outcomes.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text("print('old')\n", encoding="utf-8")
    (tmp_path / "paper.md").write_text("# old paper\n", encoding="utf-8")
    return tmp_path


def test_an_approval_of_another_request_does_not_count_and_nobody_can_approve_anonymously(tmp_path: Path) -> None:
    fp.freeze(tmp_path, P1, approved_by="auto", source="plan.md")
    assert fp.approve(tmp_path, "Jun", via="cli")[0] is False  # nothing is waiting
    pending = fp.propose(tmp_path, P2, {"hypothesis": "h"}, source="redesign", reason="wider grid", results_seen=False)
    ok, message = fp.approve(tmp_path, "  ", via="cli")
    assert not ok and "who approves" in message and fp.approval_for(tmp_path, pending) is None
    assert fp.approve(tmp_path, "Jun", via="cli")[0] is True
    assert fp.approval_for(tmp_path, pending)["approved_by"] == "Jun"
    other = fp.propose(tmp_path, {"runs_per_setting": 5}, None, source="redesign", reason="", results_seen=False)
    assert fp.approval_for(tmp_path, other) is None  # the approval was for the earlier request


def test_an_approved_amendment_before_any_result_is_prespecified_and_archives_nothing(tmp_path: Path) -> None:
    fp.freeze(tmp_path, P1, approved_by="auto", source="plan.md")
    pending = fp.propose(tmp_path, P2, {"hypothesis": "h"}, source="redesign", reason="wider grid", results_seen=False)
    fp.approve(tmp_path, "Jun", via="web")
    record = fp.apply(tmp_path, pending, fp.approval_for(tmp_path, pending), raw_root=tmp_path / "raw")
    assert record["prespecified"] is True and record["archived_to"] is None and record["new_run"] == "run_2"
    assert fp.protocol_of(tmp_path) == P2 and fp.run_id(tmp_path) == "run_2"
    assert fp.load(tmp_path)["amendments"] == 1 and fp.post_hoc(tmp_path) == []
    assert not fp.pending_path(tmp_path).exists() and not fp.approval_path(tmp_path).exists()


def test_an_amendment_after_results_archives_the_old_run_and_is_recorded_as_post_hoc(tmp_path: Path) -> None:
    quest = _quest_with_results(tmp_path)
    fp.freeze(quest, P1, approved_by="auto", source="plan.md")
    pending = fp.propose(quest, P2, {"hypothesis": "h"}, source="redesign at iteration 1", reason="the grid was too thin", results_seen=True)
    fp.approve(quest, "Jun", via="cli")
    record = fp.apply(quest, pending, fp.approval_for(quest, pending), raw_root=quest / "raw")
    assert record["prespecified"] is False and record["archived_to"] == "archive/run_1" and record["previous_run"] == "run_1"
    assert not (quest / "raw").exists(), "the raw outcomes must not be reusable by the next run"
    assert (quest / "archive" / "run_1" / "raw" / "seed0" / "outcomes.csv").is_file()
    assert (quest / "archive" / "run_1" / "code" / "experiment.py").read_text(encoding="utf-8") == "print('old')\n"
    assert (quest / "archive" / "run_1" / "paper.md").is_file() and (quest / "code" / "experiment.py").is_file()
    saved = json.loads(fp.amendment_path(quest, 1).read_text(encoding="utf-8"))
    assert saved["approved_by"] == "Jun" and saved["from_sha256"] == fp.sha256(P1) and saved["to_sha256"] == fp.sha256(P2)
    assert [a["n"] for a in fp.post_hoc(quest)] == [1]
    said = fp.disclosure(quest)
    assert "AMENDED" in said and "post-hoc" in said and "Jun" in said and "the grid was too thin" in said
    assert fp.disclosure(tmp_path / "nothing") == ""


def test_a_request_resumed_past_without_approval_is_kept_as_a_declined_record(tmp_path: Path) -> None:
    fp.freeze(tmp_path, P1, approved_by="auto", source="plan.md")
    pending = fp.propose(tmp_path, P2, None, source="redesign", reason="", results_seen=True)
    fp.decline(tmp_path, pending)
    assert not fp.pending_path(tmp_path).exists() and fp.protocol_of(tmp_path) == P1 and fp.amendments(tmp_path) == []
    assert json.loads((tmp_path / "needs" / "PROTOCOL_AMENDMENT_1_declined.json").read_text(encoding="utf-8"))["declined"] is True


# --- the engine, through the real graph ---------------------------------------------------------------------------------

_HEAD = "import os, json\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
_SCRIPT = _HEAD + (
    "if os.environ.get('FI_ORACLE') == '1':\n"
    "    print('ORACLE_JSON: ' + json.dumps({'checks': [{'name': 'final size closed form', 'passed': True, 'value': 0.99, 'expected': 1.0, 'tolerance': 0.05}]}))\n"
    "    raise SystemExit(0)\n"
    "os.makedirs('figures', exist_ok=True)\n"
    "plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)\n"
    "print('RESULT_JSON: {\"score\": 0.987}')\n"
)


def _cfg(tmp_path: Path, **engine: Any) -> Config:
    return Config(
        topic="smoke topic for the frozen protocol", title="frozen-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=True, auto_accept_on_pass=True, execute_replicates=1, pilot_run=False, **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    )


def _fake(prompts: list[str], *, second_design: dict[str, Any], first_protocol: dict[str, Any] | None = None,
          critique_protocol: dict[str, Any] | None = None):
    """A model whose first design carries ``first_protocol``, whose second design is ``second_design`` merged over the
    first, and whose first review asks for a revision (so the quest is redesigned once, after its results)."""
    seen = {"design": 0, "review": 0, "critique": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        kind = _classify(prompt)
        if "Apply this checklist to the draft design" in prompt:
            # The methodology audit: from the second pass on it amends the protocol in place, as a real model did.
            seen["critique"] += 1
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = critique_protocol if critique_protocol and seen["critique"] > 1 else (first_protocol or P1)
            body.update({k: v for k, v in second_design.items() if v is not None})
            return json.dumps({"objections_addressed": ["a precision target is missing"], "amended_design": body})
        if kind == "Experiment Design":
            seen["design"] += 1
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = first_protocol or P1
            if seen["design"] > 1:
                body.update(second_design)
                for key in [k for k, v in second_design.items() if v is None]:
                    body.pop(key, None)
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": _SCRIPT, "deps": ["matplotlib"]})
        if kind == "Review":
            seen["review"] += 1
            body = json.loads(_FAKE_RESPONSES["review"])
            if seen["review"] == 1:
                body.update({"verdict": "revise", "score": 2, "weaknesses": ["the grid is too thin"], "suggestions": ["widen the grid"]})
            return json.dumps(body)
        return _fake_response_for(prompt)

    return fake_chat


def _frozen(engine: Engine) -> dict[str, Any]:
    return json.loads(fp.frozen_path(engine.quest_root).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_the_protocol_is_frozen_before_the_first_full_run_with_the_oracle_the_gate_settled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={}))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    record = _frozen(engine)
    assert record["schema"] == "fi.frozen-protocol/v1" and record["run_id"] == "run_1" and record["source"] == "plan.md"
    assert record["approved_by"].startswith("auto:") and record["protocol"]["oracles"][0]["name"] == ORACLE["name"]
    assert record["sha256"] == fp.sha256(record["protocol"])
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[protocol] frozen before the first full run" in log
    assert log.index("[oracle] 1 oracle(s) passed") < log.index("[protocol] frozen before the first full run")


@pytest.mark.asyncio
async def test_a_redesign_that_leaves_the_protocol_out_keeps_the_frozen_one_for_every_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the freeze existed the second design pass carried no protocol, and the protocol and oracle gates then saw
    nothing and let the run through."""
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={"protocol": None}))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert artifacts.raw_state["iteration"] == 1, "the quest was redesigned once"
    assert (artifacts.raw_state["design"] or {}).get("protocol") == _frozen(engine)["protocol"]
    assert engine._protocol_block({"design": {"hypothesis": "h"}, "iteration": 1}) == _frozen(engine)["protocol"]
    assert not list((engine.quest_root / "needs").glob("PROTOCOL_AMENDMENT_*"))
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert log.count("[oracle] 1 oracle(s) passed") == 2, "the oracle gate must have run again, against the frozen protocol"
    assert any("FROZEN PROTOCOL" in p for p in prompts), "the redesign is told what is frozen"


@pytest.mark.asyncio
async def test_a_redesign_that_asks_for_a_different_protocol_stops_the_quest_for_a_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    ask = {"protocol_amendment": {"protocol": P2, "reason": "the grid is too thin to show a threshold"}}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design=ask))
    first = Engine(_cfg(tmp_path))
    await first.run()

    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[amendment] paused" in log and "[FI] paused for the protocol amendment" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "amendment" and descriptor["interaction"] == "supply"
    pending = fp.load_pending(first.quest_root)
    assert pending["proposed_protocol"] == P2 and pending["results_seen"] is True and pending["reason"].startswith("the grid is too thin")
    assert "runs_per_setting: 300 -> 500" in pending["changes"]
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "--approve-amendment" in text and "Resuming WITHOUT approving keeps the frozen protocol" in text
    assert _frozen(first)["sha256"] == fp.sha256(P1), "nothing changed until a person approves"


@pytest.mark.asyncio
async def test_resuming_without_approving_keeps_the_frozen_protocol_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    ask = {"protocol_amendment": {"protocol": P2, "reason": "wider grid"}}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design=ask))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    design_calls = sum(1 for p in prompts if _classify(p) == "Experiment Design")

    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None
    assert sum(1 for p in prompts if _classify(p) == "Experiment Design") == design_calls, "no new design was asked for"
    assert _frozen(second)["sha256"] == fp.sha256(P1) and fp.amendments(second.quest_root) == []
    assert (second.quest_root / "needs" / "PROTOCOL_AMENDMENT_1_declined.json").is_file()
    assert not fp.pending_path(second.quest_root).exists()
    assert (artifacts.raw_state["design"] or {}).get("protocol") == P1


@pytest.mark.asyncio
async def test_an_approved_amendment_after_results_archives_the_run_and_the_paper_says_it_was_post_hoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    ask = {"protocol_amendment": {"protocol": P2, "reason": "the grid is too thin to show a threshold"}}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design=ask))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    assert fp.load_pending(first.quest_root) is not None
    first_code = (first.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")

    ok, _ = fp.approve(first.quest_root, "Jun", via="test")
    assert ok
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None

    record = json.loads(fp.amendment_path(second.quest_root, 1).read_text(encoding="utf-8"))
    assert record["prespecified"] is False and record["approved_by"] == "Jun" and record["archived_to"] == "archive/run_1"
    assert _frozen(second)["protocol"] == P2 and _frozen(second)["run_id"] == "run_2"
    assert (second.quest_root / "archive" / "run_1" / "code" / "experiment.py").read_text(encoding="utf-8") == first_code
    assert (artifacts.raw_state["design"] or {}).get("protocol") == P2
    writer = [p for p in prompts if "AMENDED after it was frozen" in p]
    assert writer, "the paper's writer must be told the protocol was amended after results were seen"
    assert "post-hoc" in writer[0] and "the grid is too thin to show a threshold" in writer[0]
    evidence = json.loads((second.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert any("amended after results were seen" in g for g in evidence["all_gaps"].get("publication_ready", []))
    assert "amended after results were seen" in " ".join(evidence["all_gaps"].get("publication_ready", []))


@pytest.mark.asyncio
async def test_a_plan_edit_after_the_freeze_does_not_change_what_the_gates_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import plan

    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={}))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    design = plan.load_design(engine.quest_root)[0]
    design["protocol"] = {**design["protocol"], "runs_per_setting": 30}
    (engine.quest_root / "plan.md").write_text(plan.render("t", {}, design), encoding="utf-8")
    assert engine._protocol_block({"design": {}, "iteration": 0})["runs_per_setting"] == 300
    said: list[str] = []
    monkeypatch.setattr(engine._log, "warning", lambda msg, *a, **k: said.append(msg % a if a else msg))
    engine._warn_if_plan_diverges()
    assert said and "differs from the frozen protocol" in said[0] and "runs_per_setting: 300 -> 30" in said[0]


# --- the three surfaces: CLI, web, VSCode -------------------------------------------------------------------------------------


def _quest_waiting_for_approval(root: Path, quest_id: str = "q-amend") -> Path:
    quest = root / quest_id
    (quest / ".fi").mkdir(parents=True)
    fp.freeze(quest, P1, approved_by="auto", source="plan.md")
    fp.propose(quest, P2, {"hypothesis": "h"}, source="redesign at iteration 1", reason="wider grid", results_seen=True)
    return quest


def test_the_command_line_approves_by_quest_id_and_needs_a_name(tmp_path: Path, capsys) -> None:
    import launch

    quest = _quest_waiting_for_approval(tmp_path)
    args = launch.parse_args(["--approve-amendment", quest.name, "--approve-as", "Jun", "--output-root", str(tmp_path)])
    assert args.approve_amendment == quest.name and args.approve_as == "Jun"
    assert launch._approve_amendment(quest.name, "", tmp_path) == 2
    assert "who approves" in capsys.readouterr().out and not fp.approval_path(quest).exists()
    assert launch._approve_amendment("no-such-quest", "Jun", tmp_path) == 1
    assert launch._approve_amendment(quest.name, "Jun", tmp_path) == 0
    out = capsys.readouterr().out
    assert "approved amendment 1" in out and "--resume q-amend" in out
    assert fp.approval_for(quest, fp.load_pending(quest))["via"] == "cli"
    assert launch._approve_amendment(str(quest), "Jun", tmp_path / "elsewhere") == 0  # the folder itself works too


def test_the_command_line_says_when_nothing_is_waiting(tmp_path: Path, capsys) -> None:
    import launch

    quest = tmp_path / "q-none"
    (quest / ".fi").mkdir(parents=True)
    assert launch._approve_amendment("q-none", "Jun", tmp_path) == 1
    assert "no protocol amendment waiting" in capsys.readouterr().out


def test_the_web_page_shows_the_request_and_approves_it_as_an_act_of_its_own(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from web.server import make_app

    root = tmp_path / "outputs"
    root.mkdir()
    client = TestClient(make_app(root))
    quest = _quest_waiting_for_approval(root, "q-web")
    (quest / "config.yaml").write_text("topic: x", encoding="utf-8")

    body = client.get("/api/quests/q-web/amendment").json()
    assert body["frozen"]["run_id"] == "run_1" and body["pending"]["results_seen"] is True and body["pending"]["approved"] is False
    assert "runs_per_setting: 300 -> 500" in body["pending"]["changes"] and "design" not in body["pending"]
    assert client.get("/api/quests/q-web").json()["amendment"]["pending"]["reason"] == "wider grid"

    assert client.post("/api/quests/q-web/amendment/approve", json={"who": " "}).status_code == 400
    assert not fp.approval_path(quest).exists(), "an approval without a name is refused"
    done = client.post("/api/quests/q-web/amendment/approve", json={"who": "Jun"})
    assert done.status_code == 200 and done.json()["pending"]["approved"] is True
    assert fp.approval_for(quest, fp.load_pending(quest))["via"] == "web"
    other = root / "q-idle"
    (other / ".fi").mkdir(parents=True)
    assert client.post("/api/quests/q-idle/amendment/approve", json={"who": "Jun"}).status_code == 400


def test_the_quest_page_and_the_chat_have_the_approve_action() -> None:
    root = Path(__file__).resolve().parent.parent
    page = (root / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    for needle in ('id="amendment-banner"', "renderAmendment(data.amendment)", "/amendment/approve", "Resuming without approving", "resuming without approving"):
        assert needle.lower() in page.lower(), needle
    ext = root / "vscode-frontier-insight"
    assert 'cmd === "approve-amendment"' in (ext / "src" / "extension.ts").read_text(encoding="utf-8")
    skills = (ext / "src" / "skills.ts").read_text(encoding="utf-8")
    assert "export async function runApproveAmendment" in skills and '"--approve-amendment"' in skills and "showInputBox" in skills
    assert '"name": "approve-amendment"' in (ext / "package.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_protocol_the_methodology_audit_rewrote_does_not_stop_the_quest_and_does_not_replace_the_frozen_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit amends the design's protocol in place (a real model added a precision target and a stream policy). After the
    freeze that is not an amendment request: the frozen protocol stands, and the log says what the audit wanted."""
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={}, critique_protocol=P2))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and artifacts.raw_state["iteration"] == 1
    assert not fp.pending_path(engine.quest_root).exists() and fp.amendments(engine.quest_root) == []
    assert _frozen(engine)["sha256"] == fp.sha256(P1)
    assert (artifacts.raw_state["design"] or {}).get("protocol") == P1
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "changed the protocol without asking for an amendment" in log and "the frozen protocol stands" in log
    assert "runs_per_setting: 300 -> 500" in log


@pytest.mark.asyncio
async def test_the_stop_message_names_the_quest_folder_so_the_command_can_be_pasted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    ask = {"protocol_amendment": {"protocol": P2, "reason": "wider grid"}}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design=ask))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    text = (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert f'--approve-amendment "{engine.quest_root}"' in text and f'--approve-amendment "{engine.quest_root}"' in log


# --- a hand-edited frozen record (P0-3) -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_frozen_record_edited_by_hand_blocks_the_quest_until_a_person_approves_restoring_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real bypass: `needs/FROZEN_PROTOCOL.json` no longer matches its own hash, which `load()` already reported as a
    `problem` — but every gate read the edited content anyway. It must instead block, under every profile, and recover
    only from the version saved right when the protocol was locked, approved the same way any other change to it is."""
    prompts: list[str] = []
    engine_box: dict[str, Any] = {}
    tampered: dict[str, bool] = {"once": False}

    def _tamper() -> None:
        path = fp.frozen_path(engine_box["engine"].quest_root)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["protocol"]["runs_per_setting"] = 999999  # edited by hand; the sha256 field is left as it was
        path.write_text(json.dumps(record), encoding="utf-8")

    base = _fake(prompts, second_design={})

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        reply = await base(self, messages, **kw)
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Experiment Design" and "Apply this checklist" not in prompt:
            if sum(1 for p in prompts if _classify(p) == "Experiment Design") == 2 and not tampered["once"]:
                tampered["once"] = True
                _tamper()
        return reply

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    first = Engine(_cfg(tmp_path))
    engine_box["engine"] = first
    await first.run()  # pauses at the redesign (iteration 1); iteration 0 already wrote a paper.md, which is expected

    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "amendment"
    pending = fp.load_pending(first.quest_root)
    assert pending["source"] == fp.TAMPER_SOURCE and pending["proposed_protocol"] == P1 and pending["results_seen"] is False
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "changed after it was locked" in text and "--approve-amendment" in text and "does not go on" in text
    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[amendment] paused" in log
    on_disk = json.loads(fp.frozen_path(first.quest_root).read_text(encoding="utf-8"))
    assert on_disk["protocol"]["runs_per_setting"] == 999999, "nothing is silently fixed without an approval"

    # Resuming without approving does not go on (unlike a declined ordinary amendment): it pauses again, the same way.
    cfg = _cfg(tmp_path)
    unapproved = Engine(cfg, resume_quest_id=first.quest_id)
    engine_box["engine"] = unapproved
    await unapproved.run()
    assert fp.load_pending(unapproved.quest_root)["source"] == fp.TAMPER_SOURCE
    assert not fp.approval_path(unapproved.quest_root).exists()

    ok, message = fp.approve(unapproved.quest_root, "the auditor", via="test")
    assert ok and "restor" not in message  # approve() itself doesn't know the domain; it just names the request
    second = Engine(cfg, resume_quest_id=first.quest_id)
    engine_box["engine"] = second
    artifacts = await second.run()
    assert artifacts.paper_md is not None
    record = _frozen(second)
    assert record["protocol"]["runs_per_setting"] == 300 and record["version"] == 2, "restored, not the tampered content"
    assert not fp.pending_path(second.quest_root).exists() and not fp.approval_path(second.quest_root).exists()
    amend = fp.amendments(second.quest_root)
    assert len(amend) == 1 and amend[0]["source"] == fp.TAMPER_SOURCE and amend[0]["prespecified"] is True
    assert (artifacts.raw_state["design"] or {}).get("protocol", {}).get("runs_per_setting") == 300


# --- evidence_gate's own "supply" pause (unrelated to the frozen protocol, but this file already has
# the real-graph + fake-LLM infrastructure the other pause/resume tests above use, and test_audit_log.py
# already borrows it for an unrelated concern too) ------------------------------------------------------


async def test_an_unknown_evidence_gate_under_research_profile_pauses_then_retries_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real graph proof, not a mocked ``_pause_for_human``: an unparseable evidence-gate reply under
    rigor_profile: research genuinely pauses the quest via LangGraph's own interrupt(), and --resume
    (a fresh Engine with resume_quest_id, exactly like the tamper-recovery test above) re-enters the
    SAME node and retries the call — proceeding once it succeeds. A reviewer flagged the risk that
    LangGraph might resume execution "after" the interrupt() call instead of re-running the node from
    the top, which would mean the retry never actually happens; the tamper-recovery test above already
    proves resume-without-fix pauses again through the real graph (not a fake interrupt), and this test
    proves the matching positive case: resume-WITH-a-fix proceeds, because the call was genuinely
    retried, not skipped."""
    calls = {"evidence_gate": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "EvidenceGate":
            calls["evidence_gate"] += 1
            if calls["evidence_gate"] <= 2:
                return "not json at all"  # garbage the first two calls: status becomes unknown
            return json.dumps({"verdict": "sufficient", "rationale": "fine now", "gaps": []})
        # Everything else gets the plain "accept, no revise" default -- this test is about the
        # evidence_gate pause in isolation, not the redesign loop _fake() also drives.
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path).model_copy(update={"rigor_profile": "research"})

    first = Engine(cfg)
    await first.run()
    assert calls["evidence_gate"] == 1, "must have tried the gate exactly once before pausing"
    assert not (first.quest_root / "paper" / "paper.md").exists(), "must pause before write, not after"
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "evidence_gate_unknown" and descriptor["interaction"] == "supply"

    # Resume without anything having changed: the SAME node re-enters and pauses again (this half of
    # the proof already has a sibling above, for the tamper case; kept here too as a tight bracket
    # around the positive case just below, on this exact pause).
    still_broken = Engine(cfg, resume_quest_id=first.quest_id)
    await still_broken.run()
    assert calls["evidence_gate"] == 2, "resume must retry the call, not skip it"
    assert not (still_broken.quest_root / "paper" / "paper.md").exists()

    # Resume again: this time the call succeeds. The retry was real, so the quest proceeds to write.
    fixed = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await fixed.run()
    assert calls["evidence_gate"] == 3
    assert artifacts.paper_md is not None
    assert artifacts.raw_state["evidence_assessment"]["status"] == "ok"



# --- the unconditional tamper check at execute's entry (P1-6) -------------------------------------------


async def test_a_tampered_protocol_pauses_at_execute_even_with_every_named_check_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """protocol_check / oracle_check / run_manifest_check all being "off" used to mean nothing ever
    called _resolved_frozen() at all -- a hand-edited needs/FROZEN_PROTOCOL.json ran completely
    unchecked. _node_execute now calls it unconditionally, regardless of those three flags."""
    cfg = _cfg(tmp_path, protocol_check="off", oracle_check="off", run_manifest_check="off")
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    fp.freeze(eng.quest_root, P1, approved_by="auto", source="plan.md")
    record = json.loads(fp.frozen_path(eng.quest_root).read_text(encoding="utf-8"))
    record["protocol"]["runs_per_setting"] = 999999  # edited by hand; sha256 left as it was
    fp.frozen_path(eng.quest_root).write_text(json.dumps(record), encoding="utf-8")

    paused = {}

    def fake_pause(self, **kwargs):  # noqa: ANN001
        paused.update(kwargs)
        raise RuntimeError("paused")

    monkeypatch.setattr(Engine, "_pause_for_human", fake_pause)
    with pytest.raises(RuntimeError, match="paused"):
        await eng._node_execute({})  # type: ignore[arg-type]
    assert paused["kind"] == "amendment" and "changed after it was locked" in paused["headline"]


async def test_an_intact_protocol_does_not_pause_at_execute_with_every_named_check_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(tmp_path, protocol_check="off", oracle_check="off", run_manifest_check="off")
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    fp.freeze(eng.quest_root, P1, approved_by="auto", source="plan.md")

    def fail_if_paused(self, **kwargs):  # noqa: ANN001
        raise AssertionError("must not pause on an untampered protocol")

    monkeypatch.setattr(Engine, "_pause_for_human", fail_if_paused)
    # Stop right after the tamper check (before the real subprocess machinery this minimal state
    # can't drive) by making the executor's install step raise something recognisable.
    async def boom_install(*a, **k):  # noqa: ANN001, ARG001
        raise RuntimeError("stop-here")

    eng.executor.install = boom_install  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stop-here"):
        await eng._node_execute({"deps": ["numpy"]})  # type: ignore[arg-type]
