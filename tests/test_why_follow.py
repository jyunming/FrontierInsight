"""Why a quest did what it did (core/why.py, --why), following it live (--trace --follow), and keeping every model
call (output.save_model_calls)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from core import audit_log, why
from core.provider import append_cost_row, set_model_call_archive

REPO = Path(__file__).resolve().parent.parent


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "q1"
    log = audit_log.AuditLog(root / ".fi" / "audit.jsonl", quest_id="q1")
    log.append("node_started", node="review")
    log.append("model_claim", node="review", provenance=audit_log.MODEL_CLAIM, topic="review_verdict",
               claim="revise: two claims are unsupported")
    log.append("route_decision", node="review", chosen="rewrite",
               facts={"verdict": "revise", "must_flag_hits": ["unsupported_claim"]})
    log.append("node_started", node="execute")
    log.append("check_result", node="execute", check="oracle", status="failed", problems=["N=2 mean is 1.3, not 1.5"])
    (root / "needs").mkdir(parents=True)
    (root / "needs" / "EVIDENCE.json").write_text(json.dumps(
        {"status": "executed", "next_level": "internally_reconciled", "gaps": ["the audits did not run", "b"]}),
        encoding="utf-8")
    return root


def test_why_answers_the_review_the_evidence_and_any_step_from_the_records(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    answer = why.explain(root)
    assert "Why it stopped" not in answer, "not paused: no pause section"
    assert "Decided: next is rewrite because must_flag_hits=['unsupported_claim'], verdict=revise." in answer
    assert "The model's own reasons (what it said, not something FI checked):" in answer
    assert "revise: two claims are unsupported" in answer
    assert "Next: the audits did not run (+1 more)" in answer and "internally_reconciled" not in answer
    step = why.explain(root, "execute")
    assert "Check oracle: failed" in step and "N=2 mean is 1.3, not 1.5" in step
    assert why.explain(root, "stopped") == "Why it stopped: it is not waiting for you."
    assert "nothing from the design step" in why.explain(root, "design")
    (root / ".fi" / "pause.json").write_text(json.dumps({"kind": "oracle", "headline": "the oracle failed"}),
                                            encoding="utf-8")
    (root / "NEXT_STEP.md").write_text("# Action needed — the oracle failed\n\n## What to do\n1. Fix it.\n\n## Then resume\n- x\n",
                                       encoding="utf-8")
    paused = why.explain(root)
    assert paused.startswith("Why it stopped:\n  It is waiting for you: the oracle failed.") and "1. Fix it." in paused
    assert "Then resume" not in paused


def _launch(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(REPO / "launch.py"), *args, "--no-axon-sidecar"],
                          capture_output=True, text=True, timeout=timeout, cwd=str(REPO),
                          env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "FI_SKIP_BOOTSTRAP": "1"})


def test_the_cli_why_and_follow(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    out = _launch("--why", str(root), "evidence")
    assert out.returncode == 0, out.stderr
    assert "Why the evidence is at this level:" in out.stdout and "Decided" not in out.stdout

    # --follow prints what is there, then what is added, and ends when the quest stops for a person.
    def later() -> None:
        time.sleep(4)
        audit_log.AuditLog(root / ".fi" / "audit.jsonl", quest_id="q1").append("node_started", node="write")
        (root / ".fi" / "pause.json").write_text("{}", encoding="utf-8")

    threading.Thread(target=later, daemon=True).start()
    out = _launch("--trace", str(root), "--follow", "--trace-detail", "debug", timeout=90)
    assert out.returncode == 0, out.stderr
    assert "execute: check oracle: failed" in out.stdout and "write: started" in out.stdout
    assert out.stdout.rstrip().endswith("it stopped for you: see NEXT_STEP.md.")


def test_model_calls_are_kept_only_when_asked_and_without_image_bytes(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    append_cost_row(fi, node="write", model="m", usage=None, messages=[{"role": "user", "content": "p"}], response="r")
    assert not (fi / "io").exists(), "off by default"
    set_model_call_archive(fi, True)
    image = "data:image/png;base64," + "A" * 5000
    append_cost_row(fi, node="visual_check", model="m", usage={"prompt_tokens": 3}, response="looks fine",
                    messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}])
    kept = [p for p in (fi / "io").iterdir() if p.suffix == ".json"]
    assert len(kept) == 1 and kept[0].name.endswith("-visual_check.json")
    record = json.loads(kept[0].read_text(encoding="utf-8"))
    assert record["response"] == "looks fine" and record["usage"] == {"prompt_tokens": 3}
    assert record["messages"][0]["content"][0]["image_url"]["url"] == "[image, 5,022 characters]"
    set_model_call_archive(fi, False)
    append_cost_row(fi, node="write", model="m", usage=None, messages=[], response="r")
    assert len([p for p in (fi / "io").iterdir() if p.suffix == ".json"]) == 1, "off again: nothing more kept"
    assert len((fi / "cost.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_an_engine_call_is_kept_whole_when_the_quest_keeps_its_model_calls(tmp_path: Path) -> None:
    import asyncio

    from core.config import Config, EngineConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine

    cfg = Config(topic="t", title="t", provider=ProviderConfig(name="openai"), engine=EngineConfig(max_iterations=1),
                 knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "out", save_model_calls=True))
    eng = Engine(cfg)

    class _Client:
        last_usage, last_model, last_provider = {"prompt_tokens": 5}, "m1", "openai"

        async def chat(self, messages, **kw):  # noqa: ANN001
            return "the whole answer"

    eng._client = _Client()  # type: ignore[assignment]
    set_model_call_archive(eng.fi_dir, cfg.output.save_model_calls)  # what Engine.run does first
    asyncio.run(eng._chat("the whole prompt", node="design"))
    (kept,) = [p for p in (eng.fi_dir / "io").iterdir() if p.suffix == ".json"]
    record = json.loads(kept.read_text(encoding="utf-8"))
    assert record["node"] == "design" and record["model"] == "m1" and record["response"] == "the whole answer"
    assert record["messages"] == [{"role": "user", "content": "the whole prompt"}]


def test_the_web_answers_why_like_the_cli(tmp_path: Path) -> None:
    import pytest

    testclient = pytest.importorskip("fastapi.testclient")
    from web.server import make_app

    outputs = tmp_path / "outputs"
    root = _quest(outputs)
    client = testclient.TestClient(make_app(outputs))
    res = client.get(f"/api/quests/{root.name}/why?about=execute")
    assert res.status_code == 200 and res.json()["text"] == why.explain(root, "execute")
    assert "Why?" in (REPO / "web" / "static" / "quest.html").read_text(encoding="utf-8")
