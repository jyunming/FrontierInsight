"""Tests for the claim-grounding stage (claim_check node + helpers).

Grounds each substantive paper claim to evidence (experiment / citation /
unsupported), writes a paper/claims.json + paper/CLAIMS.md ledger, and surfaces
unsupported claims to the reviewer so they force a bounded revise.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _format_claim_grounding


def _engine(tmp_path: Path, *, claim_grounding: bool = True) -> Engine:
    cfg = Config(
        topic="overlay metrology",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False,
                            claim_grounding=claim_grounding),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def _set_chat(eng: Engine, payload: str) -> None:
    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        return payload
    eng._chat = fake_chat  # type: ignore[assignment,method-assign]


# --- _format_claim_grounding ----------------------------------------------


def test_format_claim_grounding_empty() -> None:
    assert _format_claim_grounding({}) == "(claim grounding not run)"


def test_format_claim_grounding_flags_unsupported() -> None:
    g = {"grounded": 2, "total": 3, "summary": "mostly ok",
         "unsupported": ["the 40% speedup"]}
    out = _format_claim_grounding({"claim_grounding": g})
    assert "2/3" in out
    assert "the 40% speedup" in out
    # The reviewer is instructed to must-flag it.
    assert "must_flag_hits" in out and "revise" in out


def test_format_claim_grounding_clean() -> None:
    g = {"grounded": 3, "total": 3, "summary": "all grounded", "unsupported": []}
    out = _format_claim_grounding({"claim_grounding": g})
    assert "No unsupported claims" in out


# --- _node_claim_check ------------------------------------------------------


def _paper(tmp_path: Path, text: str = "# Paper\n\nWe find X.\n") -> str:
    p = tmp_path / "paper" / "paper.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_claim_check_disabled_is_passthrough(tmp_path: Path) -> None:
    eng = _engine(tmp_path, claim_grounding=False)
    state = {"topic": "t", "paper_md": _paper(tmp_path)}
    out = asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    assert out == {}
    assert not (tmp_path / "paper" / "claims.json").exists()


def test_claim_check_grounds_and_writes_ledger(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _set_chat(eng, json.dumps({
        "claims": [
            {"claim": "Overlay error is 2.1 nm", "basis": "experiment",
             "citation_index": None, "evidence": "result_json overlay=2.1"},
            {"claim": "EUV outperforms DUV here", "basis": "citation",
             "citation_index": 1, "evidence": "ref 1 reports the same"},
            {"claim": "This is the best method ever", "basis": "unsupported",
             "citation_index": None, "evidence": "no result or source backs it"},
            {"claim": "junk", "basis": "nonsense-basis", "evidence": ""},
        ],
        "summary": "Two of three grounded.",
    }))
    state = {
        "topic": "overlay", "paper_md": _paper(tmp_path),
        "analysis": {"key_findings": ["Overlay error is 2.1 nm"]},
        "result_json": {"overlay": 2.1}, "literature": [],
    }
    out = asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    g = out["claim_grounding"]
    assert g["total"] == 4 and g["grounded"] == 2
    assert g["unsupported"] == ["This is the best method ever", "junk"]
    # Unknown basis normalized to unsupported.
    assert g["claims"][3]["basis"] == "unsupported"
    # Ledger artifacts written next to the paper.
    ledger = json.loads((tmp_path / "paper" / "claims.json").read_text(encoding="utf-8"))
    assert ledger["grounded"] == 2
    claims_md = (tmp_path / "paper" / "CLAIMS.md").read_text(encoding="utf-8")
    assert "2 of 4" in claims_md
    assert "cite [1]" in claims_md  # citation basis shows its index


def test_claim_check_no_paper_skips(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    out = asyncio.run(eng._node_claim_check({"topic": "t"}))  # type: ignore[arg-type]
    assert out == {}


def test_claim_check_unparseable_llm_is_safe(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _set_chat(eng, "not json at all")
    state = {"topic": "t", "paper_md": _paper(tmp_path), "literature": []}
    out = asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    g = out["claim_grounding"]
    assert g["total"] == 0 and g["unsupported"] == []
