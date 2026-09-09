"""Cross-quest memory write-back validation.

Replaces the in-engine Knowledge with a stub that records every
add_quest_artifacts call, then runs a fake-LLM quest and confirms the
paper was written back with the expected tag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


_FAKE_EXPERIMENT_CODE = """\
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1]); plt.savefig('figures/r.png', dpi=72)
print('RESULT_JSON: {"v": 1}')
"""


_FAKE = {
    "Ideation": json.dumps({
        "ideas": [{"title": "f", "summary": "x", "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "f", "rationale": "only"},
    }),
    "Experiment Design": json.dumps({
        "hypothesis": "h",
        "variables": {"independent": [], "dependent": [], "controls": []},
        "method": "m",
        "expected_outcome": "e",
        "figures_planned": ["r.png"],
        "dependencies": ["matplotlib"],
    }),
    "Implementation": json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"]}),
    "Analysis": json.dumps({
        "summary": "Two points plotted.",
        "key_findings": ["linear"],
        "claims_supported": [],
        "claims_unsupported": [],
        "limitations": [],
    }),
    "Writing": "# f\n\nResults.\n",
    "Review": json.dumps({
        "verdict": "accept", "score": 4,
        "strengths": [], "weaknesses": [], "suggestions": [], "blocking": "",
    }),
}


def _fake_response(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE.items():
        if tag in head:
            return resp
    return "{}"


@pytest.mark.slow  # builds a real venv (sandbox="venv") — not fast-tier hermetic
@pytest.mark.asyncio
async def test_quest_writeback_invokes_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = Config(
        topic="memory writeback test",
        title="memory-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # These tests pin write-back behaviour against fake LLM
            # responses that can include verdict=revise; the human-
            # feedback gate (default "after_review") would otherwise
            # intercept the route. Gate off so the legacy revise/done
            # routing decides outcome, and write-back behaviour is
            # what's being verified.
            human_feedback_gate="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=True, write_back_quests=True),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    # Stand-in for Axon: capture add_quest_artifacts calls.
    captured: list[dict] = []

    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({
            "quest_id": quest_id,
            "paper_md_exists": paper_md_path.exists(),
            "summary": summary,
            "metadata": metadata or {},
        })
        return True

    async def fake_asearch(self, query, *, top_k=None, external_top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []

    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
    # Force enabled even without axon installed.
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    art = await engine.run()

    assert art.paper_md is not None and art.paper_md.exists()
    assert len(captured) == 1
    assert captured[0]["quest_id"] == engine.quest_id
    assert captured[0]["paper_md_exists"] is True
    assert "Two points plotted." in captured[0]["summary"]
    assert "- linear" in captured[0]["summary"]
    # Rich metadata threaded through: verdict + structured findings.
    meta = captured[0]["metadata"]
    assert meta["verdict"] == "accept"
    assert meta["key_findings"] == ["linear"]
    assert meta["provider"] == "openai"
    assert "result_json" in meta
    # Path stored relative to quest_root, NOT absolute. Storing
    # `/home/<user>/.../paper.md` in Axon would leak the user's
    # directory layout into the long-term corpus and break corpus
    # portability across machines.
    assert "paper_md_relpath" in meta
    assert "paper_md_path" not in meta  # the absolute-path field is gone
    assert not Path(meta["paper_md_relpath"]).is_absolute()
    assert meta["paper_md_relpath"].endswith("paper.md")


# Build a "revise"-verdict variant of the fake responses so the engine
# terminates with verdict != accept and the accept-gate fires.
_FAKE_REVISE = dict(_FAKE)
_FAKE_REVISE["Review"] = json.dumps({
    "verdict": "revise", "score": 2,
    "strengths": [], "weaknesses": ["thin"], "suggestions": ["redo"], "blocking": "n",
})


def _fake_response_revise(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE_REVISE.items():
        if tag in head:
            return resp
    return "{}"


@pytest.mark.slow  # builds a real venv (sandbox="venv") — not fast-tier hermetic
@pytest.mark.asyncio
async def test_quest_writeback_skipped_when_verdict_revise_and_accept_gate_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `write_back_only_on_accept=True` (default), a quest whose
    final review verdict is `revise` must NOT be ingested."""
    cfg = Config(
        topic="revise-verdict test",
        title="revise-test",
        provider=ProviderConfig(name="openai"),
        # max_iterations=1 + review_loop=False means a single pass; the
        # review-node verdict goes straight to "done" regardless.
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # These tests pin write-back behaviour against fake LLM
            # responses that can include verdict=revise; the human-
            # feedback gate (default "after_review") would otherwise
            # intercept the route. Gate off so the legacy revise/done
            # routing decides outcome, and write-back behaviour is
            # what's being verified.
            human_feedback_gate="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(
            enabled=True, write_back_quests=True, write_back_only_on_accept=True,
        ),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_revise(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    captured: list[dict] = []
    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({"quest_id": quest_id})
        return True
    async def fake_asearch(self, q, *, top_k=None, external_top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []
    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    await engine.run()

    # Accept-gate fired: no write-back happened despite write_back_quests=True.
    assert captured == []


@pytest.mark.slow  # builds a real venv (sandbox="venv") — not fast-tier hermetic
@pytest.mark.asyncio
async def test_quest_writeback_runs_on_revise_when_accept_gate_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `write_back_only_on_accept=False`, every finished quest
    lands (used for bootstrapping an empty corpus)."""
    cfg = Config(
        topic="ungated writeback test",
        title="ungated-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # These tests pin write-back behaviour against fake LLM
            # responses that can include verdict=revise; the human-
            # feedback gate (default "after_review") would otherwise
            # intercept the route. Gate off so the legacy revise/done
            # routing decides outcome, and write-back behaviour is
            # what's being verified.
            human_feedback_gate="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(
            enabled=True, write_back_quests=True, write_back_only_on_accept=False,
        ),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_revise(messages[-1]["content"])
    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    captured: list[dict] = []
    def fake_add(self, *, quest_id, paper_md_path, summary, metadata=None) -> bool:  # noqa: ANN001
        captured.append({"quest_id": quest_id, "verdict": (metadata or {}).get("verdict")})
        return True
    async def fake_asearch(self, q, *, top_k=None, external_top_k=None, chosen_idea=None, chat_fn=None):  # noqa: ANN001
        return []
    monkeypatch.setattr("core.knowledge.Knowledge.add_quest_artifacts", fake_add)
    monkeypatch.setattr("core.knowledge.Knowledge.asearch", fake_asearch)
    monkeypatch.setattr(
        "core.knowledge.Knowledge.__init__",
        lambda self, c: setattr(self, "cfg", c) or setattr(self, "enabled", True)
        or setattr(self, "_brain", object()) or setattr(self, "_retriever", None),
    )

    engine = Engine(cfg)
    await engine.run()

    assert len(captured) == 1
    assert captured[0]["verdict"] == "revise"


# ---------------------------------------------------------------------------
# Document ids must identify documents (issue #237)
#
# A normalized title is not a key. Six substantively different papers were
# found sharing one id in a live corpus; because the id is the vector store's
# primary key, only one of the six was retrievable and the other five were
# silently absent — present in BM25, missing from vector search, which is a
# confusing state to debug from the outside.
# ---------------------------------------------------------------------------


def test_same_title_different_papers_get_different_ids() -> None:
    """The exact collision from the field: one shared title, six abstracts."""
    from core.knowledge import _paper_short_id

    base = {"title": "Optical proximity effect and correction"}
    abstracts = [
        "increased mask complexity and runtime relative to no OPC",
        "rule-based OPC and the limits of its correction table",
        "model-based OPC convergence under low k1 imaging",
        "inverse lithography compared against conventional OPC",
        "OPC recipe tuning for sub-16 nm half-pitch line and space",
        "a sixth paper that merely shares the same title",
    ]
    ids = {_paper_short_id({**base, "abstract": a}) for a in abstracts}
    assert len(ids) == len(abstracts), (
        "same-title papers collapsed to one id — the vector store keys on "
        "this, so every collision is a document that cannot be retrieved"
    )


def test_an_external_identifier_still_wins() -> None:
    """DOI, arXiv and PMID exist to identify papers; the hash is only for
    when none of them is available."""
    from core.knowledge import _paper_short_id

    assert _paper_short_id(
        {"title": "t", "doi": "10.1000/AbC", "abstract": "x"}
    ) == "doi:10.1000/abc"
    assert _paper_short_id(
        {"title": "t", "arxiv_id": "2504.08066", "abstract": "x"}
    ) == "arxiv:2504.08066"
    assert _paper_short_id({"title": "t", "pmid": "12345", "abstract": "x"}) == "pmid:12345"


def test_the_id_is_stable_across_re_ingest() -> None:
    """The property that makes re-ingesting an update rather than a duplicate.
    A counter or a timestamp would break it."""
    from core.knowledge import _paper_short_id

    meta = {"title": "A paper", "abstract": "the same body text"}
    assert _paper_short_id(meta) == _paper_short_id(dict(meta))


def test_distinguishing_material_falls_back_through_weaker_fields() -> None:
    """Abstract identifies a document; url identifies where it came from;
    year+authors is weakest because a preprint and its published version
    share both. Any of them beats colliding."""
    from core.knowledge import _paper_short_id

    t = {"title": "Shared title"}
    by_url = {_paper_short_id({**t, "url": u}) for u in ("http://a/x", "http://b/y")}
    assert len(by_url) == 2
    by_year = {
        _paper_short_id({**t, "year": y, "authors": "Chen"}) for y in ("2024", "2025")
    }
    assert len(by_year) == 2


def test_a_title_with_nothing_else_collides_honestly() -> None:
    """Two documents that differ in no metadata at all genuinely cannot be
    told apart, so sharing an id is the correct outcome — they dedupe. That
    is different from losing five of six that DO differ."""
    from core.knowledge import _paper_short_id

    a = _paper_short_id({"title": "Manual"})
    b = _paper_short_id({"title": "Manual"})
    assert a == b == "title:manual"


def test_the_id_does_not_repeat_the_title_twice() -> None:
    """``tag`` is minted as ``fi-paper:<paper_id>``, so including both wrote
    the title into the id twice — 164-235 character ids carrying no more
    information than the first copy."""
    from core.knowledge import Knowledge, _paper_short_id

    pid = _paper_short_id(
        {"title": "Optical proximity effect and correction", "abstract": "body"}
    )
    doc_id = Knowledge._mint_doc_id(
        "fi_external_ref_spine", {"paper_id": pid, "tag": f"fi-paper:{pid}"}
    )
    assert doc_id.count("optical-proximity") == 1, doc_id
    assert len(doc_id) < 120, f"{len(doc_id)} chars: {doc_id}"


def test_a_genuinely_distinct_discriminator_is_still_kept() -> None:
    """The de-duplication above must not swallow a discriminator that carries
    real information — ``rel_path`` is what separates summary-input docs that
    share one summary_id."""
    from core.knowledge import Knowledge

    a = Knowledge._mint_doc_id(
        "fi_summary_input", {"summary_id": "s1", "rel_path": "a/one.md"}
    )
    b = Knowledge._mint_doc_id(
        "fi_summary_input", {"summary_id": "s1", "rel_path": "b/two.md"}
    )
    assert a != b
