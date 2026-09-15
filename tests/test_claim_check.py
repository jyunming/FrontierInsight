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
             "citation_index": 1, "evidence": "ref 1 reports the same",
             "quote": "EUV overlay beat DUV overlay on every wafer"},
            {"claim": "This is the best method ever", "basis": "unsupported",
             "citation_index": None, "evidence": "no result or source backs it"},
            {"claim": "junk", "basis": "nonsense-basis", "evidence": ""},
        ],
        "summary": "Two of three grounded.",
    }))
    state = {
        "topic": "overlay", "paper_md": _paper(tmp_path),
        "analysis": {"key_findings": ["Overlay error is 2.1 nm"]},
        "result_json": {"overlay": 2.1},
        # One real reference so citation_index=1 is valid.
        "literature": [{"content": "In our fab, EUV overlay beat DUV overlay on every wafer.", "metadata": {
            "title": "EUV vs DUV overlay", "doi": "10.1/x", "source": "crossref"}}],
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


def test_claim_check_citation_without_valid_index_is_unsupported(tmp_path: Path) -> None:
    """A `citation` claim that points at no real reference (missing /
    non-numeric / out-of-range index) is downgraded to unsupported — a claim
    naming no source isn't grounded."""
    eng = _engine(tmp_path)
    _set_chat(eng, json.dumps({"claims": [
        {"claim": "A", "basis": "citation", "citation_index": None, "evidence": ""},
        {"claim": "B", "basis": "citation", "citation_index": 9, "evidence": ""},
        {"claim": "C", "basis": "citation", "citation_index": "x", "evidence": ""},
    ], "summary": ""}))
    # Only one real reference exists → index must be 1.
    state = {"topic": "t", "paper_md": _paper(tmp_path),
             "literature": [{"content": "x", "metadata": {
                 "title": "Only ref", "doi": "10.1/x", "source": "crossref"}}]}
    g = asyncio.run(eng._node_claim_check(state))["claim_grounding"]  # type: ignore[arg-type]
    assert g["grounded"] == 0
    assert sorted(g["unsupported"]) == ["A", "B", "C"]
    assert all(c["basis"] == "unsupported" and c["citation_index"] is None
               for c in g["claims"])


MCLACHLAN = (
    "Geometric integration using discrete gradients\n\n"
    "This paper discusses the discrete analogue of the gradient of a function and shows how "
    "discrete gradients can be used in the numerical integration of ordinary differential "
    "equations. The method applies to all Hamil-\ntonian, Poisson and gradient systems, and also "
    "to many dissipative systems (those with a known first integral or Lyapunov function)."
)


def _literature() -> list[dict]:
    return [
        {"content": MCLACHLAN, "metadata": {
            "title": "Geometric integration using discrete gradients", "doi": "10.1098/rsta.1999.0363",
            "source": "openalex"}},
        {"content": "An unrelated abstract about plasma echoes.", "metadata": {
            "title": "Vlasov simulation", "doi": "10.1/v", "source": "openalex"}},
        {"content": "Forum answer: explicit Euler adds energy to an undamped oscillator every step.", "metadata": {
            "title": "Energy of a damped oscillator grows", "url": "https://forum.example/q/1",
            "source": "web_search"}},
    ]


def test_a_citation_counts_only_with_a_quote_from_the_sources_text(tmp_path: Path) -> None:
    """The validation quest cited McLachlan et al. [2] for "the instability of
    explicit Euler methods in oscillatory systems"; its abstract never
    mentions Euler, and the check, which saw only titles, grounded it."""
    eng = _engine(tmp_path)
    _set_chat(eng, json.dumps({"claims": [
        {"claim": "Discrete gradients cover dissipative systems [1]", "basis": "citation", "citation_index": 1,
         "quote": "applies to all Hamiltonian, Poisson and gradient systems, and also to “many dissipative systems”"},
        {"claim": "Explicit Euler is unstable for oscillators [1]", "basis": "citation", "citation_index": 1,
         "quote": "explicit Euler methods are unstable in oscillatory systems"},
        {"claim": "Euler is conditionally stable [1]", "basis": "citation", "citation_index": 1},
        {"claim": "Too short a quote [1]", "basis": "citation", "citation_index": 1, "quote": "gradient systems"},
        {"claim": "Euler adds energy [W1]", "basis": "citation", "citation_index": "W1",
         "quote": "explicit Euler adds energy to an undamped oscillator ... every step"},
    ], "summary": ""}))
    state = {"topic": "t", "paper_md": _paper(tmp_path, "# P\n\nClaims [1] and [W1].\n"),
             "literature": _literature()}
    claims = asyncio.run(eng._node_claim_check(state))["claim_grounding"]["claims"]  # type: ignore[arg-type]
    assert [c["basis"] for c in claims] == ["citation", "unsupported", "unsupported", "unsupported", "citation"]
    assert "the quote is not in the text of [1]" in claims[1]["evidence"]
    assert "no quote from [1] given" in claims[2]["evidence"]
    assert claims[4]["citation_index"] == "W1"
    ledger = (tmp_path / "paper" / "CLAIMS.md").read_text(encoding="utf-8")
    assert 'quoted: "explicit Euler methods are unstable in oscillatory systems"' in ledger


def test_the_check_sees_the_text_of_each_source_the_paper_cites(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: list[str] = []

    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen.append(prompt)
        return json.dumps({"claims": [], "summary": ""})

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    paper = "# P\n\nEuler is unstable in oscillatory systems [1]. See also [W1].\n\n## References\n\n2. Vlasov [2]\n"
    state = {"topic": "t", "paper_md": _paper(tmp_path, paper), "literature": _literature()}
    asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    prompt = seen[0]
    assert "[1] Geometric integration using discrete gradients · DOI:10.1098/rsta.1999.0363\nText:" in prompt
    assert "many dissipative systems" in prompt and "explicit Euler adds energy" in prompt
    # Cited only in the paper's own reference list: title only.
    assert "[2] Vlasov simulation · DOI:10.1/v" in prompt and "plasma echoes" not in prompt


def test_citing_sentences_spell_out_lists_and_ranges() -> None:
    from core.engine import _citing_sentences

    paper = (
        "# T\n\nFirst claim [1, 3]. Second claim [2–4] and a page [W2].\n"
        "Third [5-5].\n\n## References\n\n1. Something [9].\n"
    )
    found = _citing_sentences(paper)
    assert sorted(found) == ["1", "2", "3", "4", "5", "W2"]
    assert found["3"] == ["First claim [1, 3].", "Second claim [2–4] and a page [W2]."]
    assert "9" not in found


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


def test_claim_check_provider_failure_is_non_fatal(tmp_path: Path) -> None:
    """A transient provider outage during the grounding call (e.g. a Copilot
    bridge stall) must NOT abort the quest. claim_check runs after the paper is
    already written; it degrades to "grounding not run" (empty result, no
    ledger) so review + output generation still proceed on resume-free."""
    eng = _engine(tmp_path)

    async def boom_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        raise RuntimeError(
            "Copilot backend was unavailable across 6 retry attempts "
            "(bridge stalled: no part for 180 s)"
        )
    eng._chat = boom_chat  # type: ignore[assignment,method-assign]

    state = {"topic": "t", "paper_md": _paper(tmp_path), "literature": []}
    # Does not raise — the node records the provider failure instead.
    out = asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    assert out["claim_grounding"] == {}
    assert "Copilot backend was unavailable" in out["claim_check_failed"]
    # The reviewer is told the check failed, not that it was never meant to run.
    assert "FAILED" in _format_claim_grounding(out)
    # No ledger is written when grounding never produced a result.
    assert not (tmp_path / "paper" / "claims.json").exists()


# --- how much of each cited source the check sees -----------------------------

# A quest's text of Brauer 2017, "Mathematical epidemiology: Past, present, and
# future", as FI stored it (inline formulas already blank), trimmed to its
# opening and its "Stochastic models" section. The paper below is that quest's
# six sentences citing it, renumbered [1]. Its claim check saw 1,500 characters
# of the source: the abstract and the Galton-Watson paragraph. It rejected the
# "critical mass" sentence as "truncated", though the source says so further on.
BRAUER_2017 = {
    "title": "Mathematical epidemiology: Past, present, and future",
    "doi": "10.1016/j.idm.2017.02.001",
    "source": "openalex",
}
BRAUER_PAPER = (
    "# Stochastic and deterministic SIR\n\n## Introduction\n\n"
    "However, deterministic models assume a continuous population and homogeneous mixing, which are often "
    "inaccurate descriptions of the early stages of an outbreak when the number of infected individuals is "
    "very small [1]. In these regimes, transmission is a stochastic event.\n\n"
    "For $R_0 > 1$, branching process theory suggests that the probability of a minor outbreak (extinction) "
    "is $1/R_0$, meaning the probability of a major outbreak is $1 - 1/R_0$ [1]. Once an outbreak escapes "
    "this initial stochastic phase and reaches a critical mass, its trajectory is expected to converge toward "
    "the deterministic prediction [1].\n\n## Discussion\n\n"
    "The convergence of the outbreak probability to $1 - 1/R_0$ for $R_0 > 1$ aligns with the theory of "
    "Galton-Watson branching processes [1]. In these models, the probability of extinction for a single "
    "infective is the smallest root of the generating function, which for the simple SIR model simplifies to "
    "$1/R_0$ [1].\n\n"
    "However, our results show that the deterministic prediction does not represent the *average* outcome of "
    "all stochastic realizations, but rather the *conditional* outcome given that the disease avoids early "
    "extinction [1].\n"
)
# Where the source supports the "critical mass" sentence.
BRAUER_SUPPORT = "make a transition to a compartmental model when the epidemic has become established"


def _brauer_text() -> str:
    return (Path(__file__).parent / "fixtures" / "claim_check_brauer_2017.txt").read_text(encoding="utf-8")


def test_the_source_excerpt_is_as_long_as_the_budget_allows(monkeypatch) -> None:
    from core import engine as E

    text = _brauer_text()
    sentences = E._citing_sentences(BRAUER_PAPER)["1"]
    lengths = {}
    for budget in (1500, 6000):
        monkeypatch.setattr(E, "_CLAIM_SOURCE_CHARS", budget)
        excerpt = E._claim_source_block("1", BRAUER_2017, text, sentences).split("\nText:\n", 1)[1]
        assert len(excerpt) <= budget
        lengths[budget] = len(excerpt)
    assert lengths[1500] < lengths[6000]


def test_the_check_sees_the_passage_that_supports_a_citation(tmp_path: Path) -> None:
    from core.engine import _normalized_text

    text = _brauer_text()
    # The passage is past the first 1,500 characters, and the excerpt of the
    # sentences' most related passages did not reach it at that size.
    assert _normalized_text(text).find(BRAUER_SUPPORT) > 1500
    eng = _engine(tmp_path)
    seen: list[str] = []

    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen.append(prompt)
        return json.dumps({"claims": [], "summary": ""})

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    state = {"topic": "t", "paper_md": _paper(tmp_path, BRAUER_PAPER),
             "literature": [{"content": text, "metadata": BRAUER_2017}]}
    asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    prompt = _normalized_text(seen[0])
    assert BRAUER_SUPPORT in prompt
    assert "we define the generating function" in prompt
    assert "distinction between a minor outbreak and a major epidemic" in prompt
