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


# --- a formula the source and the paper render differently ------------------

# Allen & Lahodny 2012 as FI really stored it, trimmed to the passages that
# matter. One source, one sentence, two renderings of the same formula: the
# abstract writes the reproduction number "ℛ(0)" and the exponent as "( i )",
# and the full text writes the subscript broken onto its own line ("R\n0") and
# the exponent as a bare "i". The paper quoting it wrote "R0" and "^i". None of
# those is a substring of another, so the check called a verbatim quotation
# unsupported and two runs spent their whole review loop on it.
ALLEN_2012 = {
    "title": "Extinction thresholds in deterministic and stochastic epidemic models",
    "doi": "10.1080/17513758.2012.665502",
    "source": "openalex",
}
_ALLEN_TAIL = (
    " and i infectious individuals are introduced into a susceptible population, "
    "then the probability of a major outbreak is approximately "
)
# How four real quests quoted that sentence. Every one is a correct quotation.
ALLEN_QUOTES = {
    "R0, parenthesised exponent": "if R0 > 1" + _ALLEN_TAIL + "1-(1/R0)( i )",
    "R0, caret exponent": "In the case of a single infectious group, if R0 > 1"
                          + _ALLEN_TAIL + "1-(1/R0)^i",
    "LaTeX font command": r"if $\mathcal{R}(0)>1$" + _ALLEN_TAIL
                          + r"1-(1/$\mathcal{R}(0)$)( i )",
    "minus sign, bare exponent": "In the case of a single infectious group, if R0 > 1"
                                 + _ALLEN_TAIL + "1 − (1/R0)i.",
    "verbatim, script R": "In the case of a single infectious group, if ℛ(0)>1"
                          + _ALLEN_TAIL + "1-(1/ℛ(0))( i ).",
}
# And quotations that are NOT what the source says. Each changes exactly one
# thing the fold must never fold away.
ALLEN_FABRICATIONS = {
    "the exponent is a different number": "if R0 > 1" + _ALLEN_TAIL + "1-(1/R0)( 2 )",
    "the threshold is a different number": "if R0 > 2" + _ALLEN_TAIL + "1-(1/R0)^i",
    "the operator is a plus": "if R0 > 1" + _ALLEN_TAIL + "1+(1/R0)( i )",
    "the inequality is reversed": "if R0 < 1" + _ALLEN_TAIL + "1-(1/R0)^i",
    "the same tokens in another order": "then the probability of a major outbreak is "
        "approximately 1-(1/R0)( i ) if R0 > 1 and i infectious individuals are "
        "introduced into a susceptible population",
    "a word replaced by its translation": "The basic reproduction number, R0, one of the "
        "most well-known thresholds in deterministic epidemic theory, ამიტომ a disease "
        "outbreak if R0>1.",
    "invented outright": "the probability of a major outbreak is exactly one minus the "
        "reciprocal of the basic reproduction number in every stochastic epidemic model",
}


def _allen_text() -> str:
    return (Path(__file__).parent / "fixtures"
            / "claim_check_allen_lahodny_2012.txt").read_text(encoding="utf-8")


def test_a_formula_the_source_renders_differently_is_still_quoted_from_it() -> None:
    """Each of these is the source's own sentence, written the way the quoting
    paper renders mathematics. All of them are in the source."""
    from core.engine import _quote_in_source

    text = _allen_text()
    # The two renderings really are both in the one stored source, and really
    # are different strings — otherwise this test proves nothing. Compared with
    # the line breaks flattened, because the subscript of the second one is a
    # line break and git hands this file to Windows with a different one.
    flat = " ".join(text.split())
    assert "1-(1/ℛ(0))( i )" in flat
    assert "1 − (1/R 0)i" in flat
    for name, quote in ALLEN_QUOTES.items():
        assert _quote_in_source(quote, text), name


def test_a_quotation_that_changes_the_mathematics_is_not_in_the_source() -> None:
    """The fold must buy nothing for a quote the source does not support: a
    changed number, a changed operator, a reordering, a translated word and an
    invention all stay rejected. If this test ever goes green the check has
    stopped protecting the paper from a fabricated citation."""
    from core.engine import _quote_in_source

    text = _allen_text()
    for name, quote in ALLEN_FABRICATIONS.items():
        assert not _quote_in_source(quote, text), name


def test_folding_keeps_the_numbers_the_operators_and_the_order() -> None:
    from core.engine import _math_folded

    # A subscript, however the PDF flattened it, is the same symbol — and
    # however the paper quoting it wrote the subscript in LaTeX.
    assert _math_folded("r(0)>1") == _math_folded("r0 > 1") == _math_folded("r 0>1")
    assert _math_folded("r_{0}") == _math_folded("r_0") == _math_folded("r0")
    assert _math_folded("r^{i}") == _math_folded("r^i") == _math_folded("ri")
    # An exponent, spaced, parenthesised, carried or bare.
    assert _math_folded("(1/r0)( i )") == _math_folded("(1/r0)^i") == _math_folded("(1/r0)i")
    # A group that is not a script keeps its parentheses: the divisor here is
    # an expression, not a subscript, and must not collapse into the token
    # before it.
    assert "(1/r0)" in _math_folded("1-(1/r0)i")
    # The two mathematical letters NFKC does not fold to an ASCII letter (it
    # leaves the Weierstrass p alone and folds the reduced Planck constant to a
    # stroked h, which is still not "h").
    assert _math_folded("℘") == "p"
    assert _math_folded("ħ") == "h"
    # What the formula says still separates these.
    assert _math_folded("1-(1/r0)( i )") != _math_folded("1-(1/r0)( 2 )")
    assert _math_folded("1-(1/r0)i") != _math_folded("1+(1/r0)i")
    assert _math_folded("r0>1") != _math_folded("r0<1")


def test_the_grounding_accepts_the_real_quotation_and_rejects_the_fabricated_one(
    tmp_path: Path,
) -> None:
    """End to end through the node, on the stored source: the quest whose
    CLAIMS.md said "the quote is not in the text of [3]" about a formula that
    was in the text of [3]."""
    eng = _engine(tmp_path)
    _set_chat(eng, json.dumps({"claims": [
        {"claim": "The outbreak probability is 1 - (1/R0)^i [1]", "basis": "citation",
         "citation_index": 1, "quote": ALLEN_QUOTES["R0, parenthesised exponent"]},
        {"claim": "The same formula, quoted with a caret [1]", "basis": "citation",
         "citation_index": 1, "quote": ALLEN_QUOTES["R0, caret exponent"]},
        {"claim": "The outbreak probability is 1 - (1/R0)^2 [1]", "basis": "citation",
         "citation_index": 1, "quote": ALLEN_FABRICATIONS["the exponent is a different number"]},
    ], "summary": ""}))
    state = {"topic": "t", "paper_md": _paper(tmp_path, "# P\n\nThe threshold [1].\n"),
             "literature": [{"content": _allen_text(), "metadata": ALLEN_2012}]}
    claims = asyncio.run(eng._node_claim_check(state))["claim_grounding"]["claims"]  # type: ignore[arg-type]
    assert [c["basis"] for c in claims] == ["citation", "citation", "unsupported"]
    assert "the quote is not in the text of [1]" in claims[2]["evidence"]


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


# --- how much of the quest's OWN evidence the check sees ----------------------

# A real SIR run's shape. Its analysis produced 14 key findings, the last of
# them an exact finite-state validation, and results that open with an
# 80,000-character parameter sweep. The findings, the supported claims and the
# results were serialised together and cut at 4,000 characters, which showed
# 3.5% of the 113,460 the run had: the cut fell inside the 12th finding and the
# 14th never appeared. The check called that validation unsupported, the
# rewrite deleted the table that proved it, and the next review asked for the
# table back.
VALIDATION_FINDING = (
    "[validation:N=100] Exact CTMC threshold probabilities were 0.229, 0.131, and 0.0617 "
    "for R0=0.9; 0.422, 0.357, and 0.313 for R0=1.5; and 0.672, 0.660, and 0.658 for R0=3 "
    "across the three thresholds. Each fell within the corresponding Gillespie Wilson "
    "interval; exact and Gillespie mean final-size fractions were also close."
)


def _sweep_findings(n: int = 16) -> list[str]:
    """Findings shaped like the run's per-stratum ones, long enough together
    that the old 4,000-character cut could not reach what follows them."""
    return [
        f"[by_R0:{i}] With branching reference 1/3, probabilities at N=100 for tau=0.05, "
        f"0.10, and 0.20 were 0.427 (Wilson 95% CI 0.372-0.483), 0.360, and 0.313; the "
        f"paired bootstrap comparison across thresholds stayed consistent with finite-N "
        f"sampling in stratum {i}."
        for i in range(n)
    ]


def _evidence_block(prompt: str) -> str:
    """The evidence section of a claim-check prompt."""
    return prompt.split("## This quest's own evidence", 1)[1].split("## The paper's sources", 1)[0]


def _capture(eng: Engine) -> list[str]:
    seen: list[str] = []

    async def fake_chat(prompt: str, *, node: str = "") -> str:  # noqa: ARG001
        seen.append(prompt)
        return json.dumps({"claims": [], "summary": ""})

    eng._chat = fake_chat  # type: ignore[assignment,method-assign]
    return seen


def test_every_key_finding_reaches_the_check(tmp_path: Path) -> None:
    """The finding that proves a claim is often the last one the analysis
    wrote. Nothing may be dropped for its position, and no finding may be cut
    in half."""
    findings = [*_sweep_findings(), VALIDATION_FINDING]
    # The state the old cut could not show: the findings alone exceed 4,000.
    assert len(json.dumps(findings, indent=2)) > 4000
    eng = _engine(tmp_path)
    seen = _capture(eng)
    state = {"topic": "t", "paper_md": _paper(tmp_path),
             "analysis": {"key_findings": findings,
                          "claims_supported": [{"claim": "The Gillespie implementation is "
                                                "consistent with the exact finite-state CTMC "
                                                "at N=100.", "evidence": "All nine exact "
                                                "threshold probabilities were inside their "
                                                "Wilson intervals."}]},
             "literature": []}
    asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    block = _evidence_block(seen[0])
    assert VALIDATION_FINDING in block
    for finding in findings:
        assert finding in block
    # The claim the analysis says the run supports reaches it too.
    assert "All nine exact threshold probabilities were inside" in block


def test_the_results_branch_the_findings_name_reaches_the_check(tmp_path: Path) -> None:
    """The results open with a sweep far larger than any budget, so every
    leading slice of them is that sweep. The branch the findings actually talk
    about has to be chosen, not waited for."""
    sweep = {str(i): {"conditional_major_final_size_fraction":
                      {"mean": 0.5 + i / 1000, "iqr": [0.1, 0.2], "n": 300}}
             for i in range(200)}
    validation = {"N": 100, "by_R0": {"0.9": {
        "exact_outbreak_probability": 0.22879608811337607,
        "exact_mean_final_size_fraction": 0.0468407125395854,
        "exact_final_size_pmf": [i / 1000 for i in range(101)],
    }}}
    eng = _engine(tmp_path)
    seen = _capture(eng)
    state = {"topic": "t", "paper_md": _paper(tmp_path),
             "analysis": {"key_findings": [VALIDATION_FINDING]},
             "result_json": {"by_R0": sweep, "validation": validation},
             "literature": []}
    asyncio.run(eng._node_claim_check(state))  # type: ignore[arg-type]
    block = _evidence_block(seen[0])
    assert "exact_outbreak_probability" in block
    assert "0.2287960881" in block
    # The sweep did not fit, so none of its contents crowd the block.
    assert "conditional_major_final_size_fraction" not in block
    # And the array behind the validation is named and bounded, not printed.
    assert "exact_final_size_pmf" in block
    assert "[101 values, 0 to 0.1]" in block


def test_a_finding_is_shown_whole_or_not_at_all() -> None:
    """Under a budget too small for everything, an item is left out rather
    than severed — the old cut ended one finding mid-sentence."""
    from core.engine import _claim_distilled_block

    findings = _sweep_findings(6)
    block, dropped = _claim_distilled_block({"key_findings": findings}, 1200)
    assert dropped
    assert len(block) <= 1200
    kept = [f for f in findings if f in block]
    assert kept, "at least one finding should fit"
    assert len(kept) == len(findings) - dropped
    # No partial item: every finding in the block is there in full.
    for finding in findings:
        head = finding[:60]
        assert head not in block or finding in block


def test_a_long_numeric_array_is_summarised_not_printed() -> None:
    from core.engine import _summarise_long_arrays

    out = _summarise_long_arrays({
        "pmf": [i / 100 for i in range(101)],
        "wilson95": [0.372, 0.483],
        "labels": ["a"] * 40,
        "nested": {"deep": list(range(50))},
    })
    assert out["pmf"] == "[101 values, 0 to 1]"
    assert out["nested"]["deep"] == "[50 values, 0 to 49]"
    # A short array, and one that isn't all numbers, are left alone.
    assert out["wilson95"] == [0.372, 0.483]
    assert out["labels"] == ["a"] * 40


def test_the_results_are_shown_whole_when_they_fit() -> None:
    from core.engine import _claim_results_block

    block = _claim_results_block({"a": 1, "b": 2}, query="a b", budget=14000)
    assert '"a": 1' in block and '"b": 2' in block
    # Nothing to say about branches when every one of them is there.
    assert "did not fit" not in block


def test_the_evidence_block_stays_within_its_budget() -> None:
    from core.engine import _claim_distilled_block, _claim_results_block

    findings = _sweep_findings(40)
    budget = 6000
    distilled, _ = _claim_distilled_block({"key_findings": findings}, budget)
    results = _claim_results_block(
        {str(i): {"x": list(range(20))} for i in range(400)},
        query=distilled, budget=budget - len(distilled),
    )
    # A sweep of hundreds of branches: none of them fits whole, and naming
    # every one of them in the heading would itself overrun the budget.
    assert len(distilled) <= budget
    assert len(distilled) + len(results) <= budget
