"""``Engine._audit_claims`` turns a node's stated reasoning into ``model_claim`` events (core/audit_log.py).

Design and review already fed this; item 5 of the 2026-09-23 re-audit asked for the evidence gate's verdict and
analyze's ``next_step`` reason to do the same, and for each claim to carry which provider/model actually produced it
plus a hash of the exact prompt/reply — so a claim can be checked against the real call, not taken on faith.

Two things this file is careful about, both caught before any code was written:

* The evidence gate can be decided by a hardcoded engine rule (``core.engine._evidence_gate_rule``) with no model
  call at all, or come back with ``status: "unknown"`` (a provider failure, an unparseable reply) with no real
  rationale. Recording either as a ``model_claim`` would misattribute an engine decision (or nothing at all) to the
  model, which ``core/audit_log.py``'s own contract forbids ("never mixed with a check result"). Only a genuinely
  model-decided, successfully-parsed verdict is recorded.
* Provenance (``provider``/``model``/``prompt_hash``/``response_hash``) comes from ``Engine._chat_provenance``, which
  reads ``self._last_chat`` — populated by ``_chat()`` after the real call, not from static config, since a fallback
  provider can serve the call the config named a different one for (core/provider.py::FallbackLLMClient.last_model /
  last_provider already reflect whoever actually answered; see tests/test_provider_fallback.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import audit_log as al
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine


def _engine(tmp_path: Path, *, knowledge_enabled: bool = False) -> Engine:
    cfg = Config(
        topic="decision trace", title="decision-claims",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=knowledge_enabled),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    return eng


def _claims(eng: Engine) -> list[dict[str, Any]]:
    return [e for e in al.read(eng.audit.path) if e["kind"] == "model_claim"]


# ---- design / review: existing behaviour, now with provenance -------------


def test_design_claims_are_unchanged_and_gain_provenance(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._last_chat["design"] = {
        "provider": "openai", "model": "gpt-5", "prompt_hash": "p" * 64, "response_hash": "r" * 64,
    }
    out = eng._audit_claims("design", {
        "design": {
            "hypothesis": "h",
            "rationale": {
                "assumptions": ["trials are independent"],
                "alternatives_considered": [{"option": "Wilson pooled", "decision": "rejected", "reason": "assumes iid"}],
            },
        },
    })

    # Regression: rationale is stripped from state so later prompts don't carry it.
    assert "rationale" not in out["design"]

    claims = _claims(eng)
    assert len(claims) == 2
    assumption, alternative = claims
    assert assumption["topic"] == "assumption" and assumption["claim"] == "trials are independent"
    assert alternative["topic"] == "alternative"
    assert alternative["decision"] == "rejected" and alternative["reason"] == "assumes iid"
    for c in claims:
        assert c["provider"] == "openai" and c["model"] == "gpt-5"
        assert c["prompt_hash"] == "p" * 64 and c["response_hash"] == "r" * 64
        assert c["provenance"] == al.MODEL_CLAIM


def test_review_claims_are_unchanged_and_gain_provenance(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._last_chat["review"] = {"provider": "claude_cli", "model": "opus", "prompt_hash": "a" * 64, "response_hash": "b" * 64}
    eng._audit_claims("review", {"review": {"verdict": "revise", "blocking": "no oracle", "weaknesses": ["thin lit"]}})

    claims = _claims(eng)
    assert [c["topic"] for c in claims] == ["review_verdict", "review_weakness"]
    assert claims[0]["claim"] == "revise: no oracle"
    for c in claims:
        assert c["provider"] == "claude_cli" and c["model"] == "opus"


# ---- evidence_gate: only a model-decided, successfully-parsed verdict -----


def test_evidence_gate_model_decided_records_verdict_and_gaps(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._last_chat["evidence_gate"] = {"provider": "openai", "model": "gpt-5"}
    eng._audit_claims("evidence_gate", {
        "evidence_assessment": {
            "decided_by": "model", "status": "ok", "verdict": "broaden",
            "rationale": "only 2 of 6 sources are on-topic", "gaps": ["on-topic sources", "a supporting finding"],
        },
    })

    claims = _claims(eng)
    assert len(claims) == 3
    verdict, gap1, gap2 = claims
    assert verdict["topic"] == "verdict" and verdict["decision"] == "broaden"
    assert verdict["claim"] == "only 2 of 6 sources are on-topic"
    assert [gap1["claim"], gap2["claim"]] == ["on-topic sources", "a supporting finding"]
    assert gap1["topic"] == "gap"
    for c in claims:
        assert c["provider"] == "openai"


def test_evidence_gate_rule_decided_records_nothing(tmp_path: Path) -> None:
    """A hardcoded engine rule (core.engine._evidence_gate_rule) decided this — no model was asked, so recording its
    rationale as a model_claim would misattribute it."""
    eng = _engine(tmp_path)
    eng._audit_claims("evidence_gate", {
        "evidence_assessment": {
            "decided_by": "rule", "status": "ok", "verdict": "broaden",
            "rationale": "no source with readable text was retrieved.", "gaps": ["sources on the research question"],
        },
    })

    assert _claims(eng) == []


def test_evidence_gate_unknown_status_records_nothing(tmp_path: Path) -> None:
    """A provider failure or an unparseable reply — there is no real rationale to record either way."""
    eng = _engine(tmp_path)
    eng._audit_claims("evidence_gate", {
        "evidence_assessment": {
            "decided_by": "model", "status": "unknown", "verdict": "sufficient",
            "rationale": "", "gaps": [],
        },
    })

    assert _claims(eng) == []


# ---- analyze: the model's stated reason for next_step ----------------------


def test_analyze_next_step_reason_becomes_a_model_claim(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._last_chat["analyze"] = {"provider": "openai", "model": "gpt-5"}
    eng._audit_claims("analyze", {
        "analysis": {"next_step": "re_experiment", "next_step_reason": "the effect is within noise at this sample size"},
    })

    claims = _claims(eng)
    assert len(claims) == 1
    assert claims[0]["topic"] == "next_step" and claims[0]["decision"] == "re_experiment"
    assert claims[0]["claim"] == "the effect is within noise at this sample size"
    assert claims[0]["provider"] == "openai"


def test_analyze_without_a_reason_records_nothing(tmp_path: Path) -> None:
    """Older prompts / a parse failure default next_step without a reason — nothing to claim."""
    eng = _engine(tmp_path)
    eng._audit_claims("analyze", {"analysis": {"next_step": "publish"}})

    assert _claims(eng) == []


# ---- _chat() populates the provenance _audit_claims reads -------------------


class _FakeChatClient:
    """No last_provider — the oldest test stubs pre-date it (see core/engine.py::_chat's getattr fallback)."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_usage = None
        self.last_model = "stub-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return self.reply


async def test_chat_populates_last_chat_for_audit_claims_to_read(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._client = _FakeChatClient("the reply text")  # type: ignore[assignment]

    reply = await eng._chat("the prompt text", node="analyze")

    assert reply == "the reply text"
    prov = eng._chat_provenance("analyze")
    assert prov["model"] == "stub-model"
    assert prov["provider"] == "openai"  # falls back to config.provider.name — the client stub has no last_provider
    import hashlib
    assert prov["prompt_hash"] == hashlib.sha256(b"the prompt text").hexdigest()
    assert prov["response_hash"] == hashlib.sha256(b"the reply text").hexdigest()
    # A key nothing was ever called under stays absent, not a KeyError or an empty-but-present dict.
    assert eng._chat_provenance("design") == {}


# ---- real _node_evidence_gate, not a hand-built patch -----------------------


async def test_node_evidence_gate_end_to_end_feeds_audit_claims(tmp_path: Path) -> None:
    """Runs the REAL node (not a hand-built ``evidence_assessment`` dict) so a wrong assumption about its output
    shape — not just a bug in ``_audit_claims`` — would show up here."""
    eng = _engine(tmp_path, knowledge_enabled=False)  # retrieval_on=False forces the model branch, never the rule
    eng._client = _FakeChatClient(json.dumps({
        "verdict": "broaden", "rationale": "no on-topic sources were retrieved", "gaps": ["on-topic sources"],
    }))  # type: ignore[assignment]

    patch = await eng._node_evidence_gate({"topic": "a test topic"})
    assessment = patch["evidence_assessment"]
    assert assessment["decided_by"] == "model" and assessment["status"] == "ok"

    eng._audit_claims("evidence_gate", patch)
    claims = _claims(eng)
    assert len(claims) == 2
    assert claims[0]["topic"] == "verdict" and claims[0]["decision"] == "broaden"
    assert claims[0]["provider"] == "openai"  # stub has no last_provider; falls back to config.provider.name


# ---- design_self_critique: a direct self._audit() call, not through _audit_claims -----------


async def test_design_self_critique_claims_gain_provenance(tmp_path: Path) -> None:
    """``_record_design_critique`` calls ``self._audit(...)`` directly (it is not reached through ``_audit_claims``,
    since it fires from inside ``_audit_design``, not from a node's returned patch) — an oversight an external review
    caught: unlike every other model_claim call site, it did not merge in ``_chat_provenance``."""
    eng = _engine(tmp_path)
    eng._client = _FakeChatClient(json.dumps({
        "objections_addressed": [{"check": "circular_evaluation", "objection": "same simulator scores and evaluates", "fix": "use a second simulator"}],
    }))  # type: ignore[assignment]

    design = {"hypothesis": "h", "dependencies": []}
    await eng._audit_design({"topic": "t", "iteration": 0}, design)

    claims = _claims(eng)
    assert len(claims) == 1
    claim = claims[0]
    assert claim["topic"] == "design_self_critique/circular_evaluation"
    assert claim["claim"] == "same simulator scores and evaluates"
    assert claim["fix"] == "use a second simulator"
    assert claim["provider"] == "openai"  # stub has no last_provider; falls back to config.provider.name
    assert claim["model"] == "stub-model"
