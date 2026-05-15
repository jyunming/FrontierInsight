"""Direct unit tests for `core.knowledge.Knowledge`.

These tests do NOT require `axon` to be installed. The disabled-path
tests use `KnowledgeConfig(enabled=False)` so the axon import flag is
irrelevant. The fallback-chain tests construct a `Knowledge` instance
with `enabled=False` then monkeypatch `enabled=True` and inject a fake
brain object — this exercises the fallback dispatch in
`add_quest_artifacts` without requiring the real axon API surface.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from core.config import KnowledgeConfig
from core.knowledge import Knowledge, RetrievedDoc


# --- disabled-path contract --------------------------------------------------


def test_search_returns_empty_when_disabled_and_no_external_fallback() -> None:
    # With Axon disabled AND no external sources, search returns [].
    # (With external sources configured, the router still runs — the
    # arxiv/openalex/crossref fallback tests cover that path.)
    k = Knowledge(KnowledgeConfig(enabled=False, external_fallback="none"))
    assert k.enabled is False
    assert k.search("any query") == []
    assert k.search("any query", top_k=10) == []


def test_ingest_returns_false_when_disabled(tmp_path: Path) -> None:
    k = Knowledge(KnowledgeConfig(enabled=False))
    f = tmp_path / "doc.txt"
    f.write_text("hello", encoding="utf-8")
    assert k.ingest(f) is False
    assert k.ingest("https://example.com/x") is False


def test_add_quest_artifacts_returns_false_when_disabled(tmp_path: Path) -> None:
    k = Knowledge(KnowledgeConfig(enabled=False))
    paper = tmp_path / "paper.md"
    paper.write_text("# paper", encoding="utf-8")
    assert k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="s") is False


def test_add_quest_artifacts_returns_false_when_writeback_disabled(tmp_path: Path) -> None:
    """Even if enabled, write_back_quests=False short-circuits to False."""
    k = Knowledge(KnowledgeConfig(enabled=False))
    # Force enabled with a stub brain but disable write-back.
    k.enabled = True
    k._brain = object()
    k.cfg = KnowledgeConfig(enabled=True, write_back_quests=False)
    paper = tmp_path / "paper.md"
    paper.write_text("# paper", encoding="utf-8")
    assert k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="s") is False


# --- fallback chain in add_quest_artifacts -----------------------------------


class _BrainAddText:
    """Mock brain that captures ``brain.ingest(documents)`` calls.

    The real AxonBrain API is ``ingest(documents: list[dict])`` where
    each doc has ``id`` / ``text`` / ``metadata``. The class is named
    ``_BrainAddText`` for historical continuity with the older test
    suite which mocked a non-existent ``add_text`` method.

    ``self.calls`` is the flattened list of ``(text, metadata)``
    tuples — one entry per document — so existing test bodies that
    index by kind via ``_calls_by_kind`` keep working without
    knowing about the batch-call shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.batches: list[list[dict]] = []
        self.finalize_calls: int = 0

    def ingest(self, documents: list[dict]) -> None:
        self.batches.append(list(documents))
        for doc in documents:
            self.calls.append((doc.get("text", ""), doc.get("metadata", {}) or {}))

    def finalize_ingest(self) -> None:
        self.finalize_calls += 1


def _enabled_knowledge_with(brain: object, *, write_back: bool = True) -> Knowledge:
    """Build a Knowledge with enabled=True bypassing the axon import path."""
    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(enabled=True, write_back_quests=write_back)
    k._brain = brain
    return k


def _calls_by_kind(brain: object) -> dict[str, tuple[str, dict]]:
    """Index a brain's recorded calls by metadata['kind']. The new
    write-back lays down a small bundle per quest (spine, body, summary,
    topic event, + per-ref spines); tests should assert by kind, not by
    positional index."""
    out: dict[str, tuple[str, dict]] = {}
    for text, meta in brain.calls:  # type: ignore[attr-defined]
        kind = (meta or {}).get("kind", "")
        out.setdefault(kind, (text, meta))
    return out


def test_add_quest_artifacts_uses_add_text_when_available(tmp_path: Path) -> None:
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("# paper body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="q42", paper_md_path=paper, summary="key finding")

    assert ok is True
    # Per quest the structured ingest lays down four docs: spine, body,
    # structured summary, topic event. (Plus per-ref spines when the
    # caller supplied `external_refs` — empty here, so just the four.)
    assert len(brain.calls) == 4
    by_kind = _calls_by_kind(brain)
    assert set(by_kind) == {
        "fi_paper_spine", "fi_quest_paper", "fi_quest_summary", "fi_topic_event",
    }

    spine_text, spine_meta = by_kind["fi_paper_spine"]
    assert spine_meta["paper_id"] == "quest:q42"
    assert spine_meta["origin"] == "quest_paper"
    assert "TITLE:" in spine_text
    assert "key finding" in spine_text  # summary became abstract

    body_text, body_meta = by_kind["fi_quest_paper"]
    # Body is the paper.md content with a 1-line citation header prepended.
    assert body_text.endswith("# paper body")
    assert body_text.startswith("[")  # header bracket
    assert body_meta["tag"] == "fi-quest:q42"
    assert body_meta["quest_id"] == "q42"
    assert "ingested_at" in body_meta

    summary_text, summary_meta = by_kind["fi_quest_summary"]
    assert "key finding" in summary_text
    assert "---structured-findings---" in summary_text
    assert summary_meta["quest_id"] == "q42"

    topic_text, topic_meta = by_kind["fi_topic_event"]
    assert topic_meta["topic_id"]  # slug present even for empty topic
    assert "QUEST: q42" in topic_text


def test_add_quest_artifacts_always_writes_summary_doc(tmp_path: Path) -> None:
    """Even with empty summary, the full 4-doc bundle (spine, body,
    summary, topic event) still lands so future retrievals can find the
    quest by its rich metadata even if the LLM produced a thin analysis."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="")

    assert ok is True
    assert len(brain.calls) == 4
    by_kind = _calls_by_kind(brain)
    assert "fi_quest_summary" in by_kind
    # The summary doc carries a structured-findings JSON envelope even
    # when the analysis summary itself is empty.
    summary_text, _ = by_kind["fi_quest_summary"]
    assert "---structured-findings---" in summary_text


def test_add_quest_artifacts_includes_caller_metadata(tmp_path: Path) -> None:
    """Rich metadata from the engine (title, topic, verdict, score, etc.)
    must land on the paper-body doc; the same fields are encoded in the
    structured-findings JSON; key_findings show up in the spine's
    `KEY CLAIMS` bullets."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("body", encoding="utf-8")

    meta = {
        "title": "BV under noise",
        "topic": "Bernstein-Vazirani depolarizing",
        "verdict": "accept",
        "score": 4,
        "hypothesis": "Pr ≈ (1-p)^cn",
        "key_findings": ["fitted c=3.0", "p*=10% at n=8"],
        "provider": "codex_cli",
    }

    ok = k.add_quest_artifacts(
        quest_id="q99",
        paper_md_path=paper,
        summary="Curve confirmed.",
        metadata=meta,
    )

    assert ok is True
    by_kind = _calls_by_kind(brain)

    body_text, body_meta = by_kind["fi_quest_paper"]
    assert body_meta["title"] == "BV under noise"
    assert body_meta["verdict"] == "accept"
    assert body_meta["provider"] == "codex_cli"
    # The citation header on the body chunk carries the title forward
    # into every retrieved fragment, regardless of where chunking lands.
    assert "BV under noise" in body_text

    summary_text, _ = by_kind["fi_quest_summary"]
    assert "BV under noise" in summary_text
    assert "fitted c=3.0" in summary_text

    spine_text, _ = by_kind["fi_paper_spine"]
    # Spine packs title + key claims into a single retrieval target.
    assert "BV under noise" in spine_text
    assert "fitted c=3.0" in spine_text
    assert "p*=10% at n=8" in spine_text

    topic_text, topic_meta = by_kind["fi_topic_event"]
    assert topic_meta["topic_id"] == "bernstein-vazirani-depolarizing"
    assert "Bernstein-Vazirani depolarizing" in topic_text
    assert "Verdict: accept" in topic_text


# ---------------------------------------------------------------------------
# External fallback — arXiv
# ---------------------------------------------------------------------------


def _arxiv_atom(*entries: dict) -> str:
    """Minimal arXiv Atom XML for the search-result envelope."""
    items = "".join(
        f"""
        <entry>
          <id>http://arxiv.org/abs/{e.get("id", "9999.99999")}v1</id>
          <title>{e.get("title", "T")}</title>
          <summary>{e.get("summary", "S")}</summary>
          <published>{e.get("published", "2020-01-01T00:00:00Z")}</published>
          <author><name>{e.get("author", "A. Person")}</name></author>
          <link title="pdf" href="http://arxiv.org/pdf/{e.get("id", "9999.99999")}.pdf"/>
        </entry>
        """
        for e in entries
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">{items}</feed>"""


def test_arxiv_fallback_fires_when_axon_returns_empty(monkeypatch) -> None:
    """When Axon is enabled but returns 0 results, search() falls
    through to the configured external fallback (arxiv by default)."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True  # bypass axon import
    # Pin fallback to arxiv-only so the test doesn't have to stub other
    # sources' JSON responses.
    k.cfg = KnowledgeConfig(enabled=True, external_fallback=["arxiv"], top_k=3)
    class _Empty:
        def invoke(self, _): return []
        def with_overrides(self, _): return self
    k._retriever = _Empty()

    atom = _arxiv_atom(
        {"id": "2401.00001", "title": "BV under noise", "summary": "study", "author": "X. Yu"},
        {"id": "2401.00002", "title": "Depolarizing channels", "summary": "review"},
    )
    captured: dict = {}

    class _FakeClient:
        def __init__(self, *a, **kw): captured["timeout"] = kw.get("timeout")
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):  # **_ accepts headers=, etc.
            captured["url"] = url
            captured["params"] = params
            r = MagicMock()
            r.text = atom
            r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _FakeClient)
    docs = k.search("Bernstein-Vazirani depolarizing", top_k=3)

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "arxiv"
    # arXiv URLs include a version suffix (vN); the parser keeps it.
    assert docs[0].metadata["arxiv_id"].startswith("2401.00001")
    assert docs[0].metadata["pdf_url"].endswith(".pdf")
    assert "X. Yu" in docs[0].metadata["authors"]
    assert "BV under noise" in docs[0].content
    assert captured["params"]["max_results"] == "3"
    assert "all:" in captured["params"]["search_query"]


def test_arxiv_fallback_disabled_when_external_fallback_none(monkeypatch) -> None:
    """`external_fallback: none` keeps an empty Axon result empty —
    no network call happens."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(enabled=True, external_fallback="none")
    class _Empty:
        def invoke(self, _): return []
        def with_overrides(self, _): return self
    k._retriever = _Empty()

    # Sentinel: if arxiv were called, the test should fail.
    def boom(*a, **kw):  # pragma: no cover
        raise AssertionError("arxiv must not be called when fallback=none")

    monkeypatch.setattr("core.knowledge.httpx.Client", boom)
    assert k.search("anything") == []


def test_arxiv_fallback_skipped_when_axon_returns_results(monkeypatch) -> None:
    """If Axon already returned docs, no fallback HTTP call happens —
    Axon's ranking wins."""
    from core.knowledge import Knowledge, RetrievedDoc as RD

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(enabled=True, external_fallback="arxiv")
    class _NonEmpty:
        def invoke(self, _):
            doc = MagicMock()
            doc.page_content = "axon hit"
            doc.metadata = {"src": "axon"}
            return [doc]
        def with_overrides(self, _): return self
    k._retriever = _NonEmpty()

    def boom(*a, **kw):  # pragma: no cover
        raise AssertionError("arxiv must not be called when axon returned results")

    monkeypatch.setattr("core.knowledge.httpx.Client", boom)
    docs = k.search("anything")
    assert len(docs) == 1
    assert docs[0].metadata["src"] == "axon"


# ---------------------------------------------------------------------------
# Local paper loader — manual feed of paywalled PDFs / MDs the user
# downloaded out-of-band.
# ---------------------------------------------------------------------------


def test_local_paper_load_markdown(tmp_path: Path) -> None:
    from core.knowledge import _load_local_paper

    f = tmp_path / "grenville_2015_inpria.md"
    f.write_text("# Inpria MOR\n\nText body about metal-oxide resists.", encoding="utf-8")
    doc = _load_local_paper(f)
    assert doc is not None
    assert "Inpria MOR" in doc.content
    assert doc.metadata["source"] == "local_paper"
    assert doc.metadata["filename"] == "grenville_2015_inpria.md"
    # Underscores/hyphens in filename → spaces in displayable title.
    assert doc.metadata["title"] == "grenville 2015 inpria"
    # Absolute path is intentionally NOT stored — leaks home directory
    # layout into Axon and makes exported corpora non-portable. Only
    # `filename` is preserved.
    assert "path" not in doc.metadata


def test_local_paper_load_txt(tmp_path: Path) -> None:
    from core.knowledge import _load_local_paper

    f = tmp_path / "note.txt"
    f.write_text("plain text reference body", encoding="utf-8")
    doc = _load_local_paper(f)
    assert doc is not None and "plain text" in doc.content


def test_local_paper_missing_file_returns_none(tmp_path: Path) -> None:
    from core.knowledge import _load_local_paper
    assert _load_local_paper(tmp_path / "missing.md") is None


def test_local_paper_unknown_extension_skips(tmp_path: Path) -> None:
    from core.knowledge import _load_local_paper

    f = tmp_path / "weird.docx"
    f.write_bytes(b"not a real docx")
    assert _load_local_paper(f) is None


def test_local_papers_pinned_to_head_of_asearch(tmp_path: Path, monkeypatch) -> None:
    """A local paper must appear FIRST in asearch results, ahead of any
    Axon hits or external-source hits."""
    from core.knowledge import Knowledge, RetrievedDoc as RD

    # Two local papers.
    a = tmp_path / "local_a.md"; a.write_text("AAA local one", encoding="utf-8")
    b = tmp_path / "local_b.md"; b.write_text("BBB local two", encoding="utf-8")

    cfg = KnowledgeConfig(enabled=False, source_routing="manual",
                          external_fallback="none", local_papers=[a, b], top_k=5)
    k = Knowledge(cfg)
    # No Axon, no external; asearch returns just the pinned papers.
    import asyncio
    docs = asyncio.run(k.asearch("anything"))
    assert len(docs) == 2
    assert docs[0].metadata["filename"] == "local_a.md"
    assert docs[1].metadata["filename"] == "local_b.md"


def test_local_papers_count_against_topk(tmp_path: Path) -> None:
    """If the user supplies more local papers than top_k, only the
    first top_k are returned and external sources are NOT queried (no
    network traffic). Belt-and-suspenders the user against accidentally
    requesting 20 papers and getting flooded."""
    from core.knowledge import Knowledge

    paths = []
    for i in range(8):
        p = tmp_path / f"p{i}.md"
        p.write_text(f"local paper {i}", encoding="utf-8")
        paths.append(p)

    cfg = KnowledgeConfig(enabled=False, source_routing="manual",
                          external_fallback="none", local_papers=paths, top_k=5)
    k = Knowledge(cfg)
    import asyncio
    docs = asyncio.run(k.asearch("query"))
    assert len(docs) == 5
    assert all(d.metadata["source"] == "local_paper" for d in docs)


def test_local_papers_supplement_external_when_few(tmp_path: Path, monkeypatch) -> None:
    """One local paper + top_k=4 means we return [local, ext1, ext2, ext3]."""
    from core.knowledge import Knowledge, RetrievedDoc as RD

    local = tmp_path / "manual.md"
    local.write_text("user-supplied note", encoding="utf-8")

    # Stub _route_external to return 3 fake external hits.
    async def fake_router(query, top_k, sources, **kw):
        return [
            RD(content=f"ext {i}", metadata={"source": "fake", "title": f"E{i}"})
            for i in range(top_k)
        ]
    monkeypatch.setattr("core.knowledge._route_external", fake_router)

    cfg = KnowledgeConfig(enabled=False, source_routing="manual",
                          external_fallback=["arxiv"], local_papers=[local], top_k=4)
    k = Knowledge(cfg)
    import asyncio
    docs = asyncio.run(k.asearch("test"))
    assert len(docs) == 4
    assert docs[0].metadata["source"] == "local_paper"
    assert all(d.metadata["source"] == "fake" for d in docs[1:])


# ---------------------------------------------------------------------------
# Phase 2 — opportunistic full-text fetch from publisher URLs.
# Login walls / non-PDF responses must NOT augment the doc content.
# ---------------------------------------------------------------------------


_FAKE_PDF = (
    # Minimal valid PDF: header + EOF marker. pypdf accepts it but
    # returns no pages — for tests of the magic-bytes check we don't
    # need a real document. Tests that need extracted text patch pypdf.
    b"%PDF-1.4\n%\xc0\xc1\xc2\xc3\n"
    b"1 0 obj\n<< >>\nendobj\nxref\n0 1\n0000000000 65535 f \n"
    b"trailer << /Root 1 0 R /Size 1 >>\nstartxref\n0\n%%EOF\n"
)


def test_looks_like_pdf_magic_check() -> None:
    from core.knowledge import _looks_like_pdf

    assert _looks_like_pdf("application/pdf", b"%PDF-1.7") is True
    # HTML login page declaring it's a PDF? Magic-bytes catches it.
    assert _looks_like_pdf("application/pdf", b"<!DOCTYPE html>") is False
    # No content-type but valid magic: trust the bytes.
    assert _looks_like_pdf(None, b"%PDF-1.4") is True
    # Wrong both: definitely not.
    assert _looks_like_pdf("text/html", b"<html>") is False


def test_find_pdf_url_in_html() -> None:
    """Citation_pdf_url meta tag is the cross-publisher convention.
    Parser must handle attribute ordering variations."""
    from core.knowledge import _find_pdf_url_in_html

    html_a = b'<html><head><meta name="citation_pdf_url" content="https://example.com/p.pdf"></head>...'
    assert _find_pdf_url_in_html(html_a) == "https://example.com/p.pdf"

    # Order swapped, single quotes, extra attrs.
    html_b = b"<meta content='/files/x.pdf' name='citation_pdf_url' scheme='dc' />"
    assert _find_pdf_url_in_html(html_b) == "/files/x.pdf"

    assert _find_pdf_url_in_html(b"<html>no meta here</html>") is None


def test_fetch_pdf_bytes_returns_none_on_login_wall(monkeypatch) -> None:
    """Login-wall response (200 OK, HTML body) must be rejected."""
    from core.knowledge import _fetch_pdf_bytes

    class _LoginWall:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            r = MagicMock()
            r.status_code = 200
            r.headers = {"content-type": "text/html"}
            r.content = b"<!DOCTYPE html>\n<html>Please log in</html>"
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _LoginWall)
    assert _fetch_pdf_bytes("https://paywalled.example/article", timeout_s=5) is None


def test_fetch_pdf_bytes_accepts_real_pdf(monkeypatch) -> None:
    from core.knowledge import _fetch_pdf_bytes

    class _OkPdf:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            r = MagicMock()
            r.status_code = 200
            r.headers = {"content-type": "application/pdf"}
            r.content = _FAKE_PDF
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _OkPdf)
    body = _fetch_pdf_bytes("https://open.example/p.pdf", timeout_s=5)
    assert body is not None and body.startswith(b"%PDF-")


@pytest.mark.asyncio
async def test_enrich_with_full_text_login_wall_skipped(monkeypatch) -> None:
    """When the publisher returns a login wall instead of a PDF, the
    original doc is returned unchanged — never half-broken."""
    from core.knowledge import _enrich_with_full_text, RetrievedDoc as RD

    docs = [RD(
        content="title only",
        metadata={"source": "crossref", "title": "t", "url": "https://paywall/x"},
    )]

    def fake_fetch(doc, *, timeout_s, max_kb):
        return None  # simulate login-wall rejection
    monkeypatch.setattr("core.knowledge._fetch_full_text", fake_fetch)

    out = await _enrich_with_full_text(
        docs, timeout_s=5, total_budget_s=30, max_kb=64,
    )
    assert len(out) == 1
    assert out[0].content == "title only"
    assert out[0].metadata.get("fetched_full_text") is not True


@pytest.mark.asyncio
async def test_enrich_with_full_text_augments_on_success(monkeypatch) -> None:
    from core.knowledge import _enrich_with_full_text, RetrievedDoc as RD

    docs = [RD(
        content="abstract only",
        metadata={"source": "openalex", "pdf_url": "https://open.example/p.pdf"},
    )]

    def fake_fetch(doc, *, timeout_s, max_kb):
        return "Section 1. Introduction. Lorem ipsum body of paper text."
    monkeypatch.setattr("core.knowledge._fetch_full_text", fake_fetch)

    out = await _enrich_with_full_text(
        docs, timeout_s=5, total_budget_s=30, max_kb=64,
    )
    assert out[0].metadata["fetched_full_text"] is True
    assert "abstract only" in out[0].content
    assert "FULL TEXT (fetched)" in out[0].content
    assert "Lorem ipsum" in out[0].content


@pytest.mark.asyncio
async def test_enrich_with_full_text_skips_docs_without_url(monkeypatch) -> None:
    """Docs without pdf_url or url are passed through untouched (no
    network call attempted)."""
    from core.knowledge import _enrich_with_full_text, RetrievedDoc as RD

    docs = [RD(content="just text", metadata={"source": "pubmed"})]

    def boom(*a, **kw):  # pragma: no cover
        raise AssertionError("must not fetch when no URL")
    monkeypatch.setattr("core.knowledge._fetch_full_text", boom)

    out = await _enrich_with_full_text(
        docs, timeout_s=5, total_budget_s=30, max_kb=64,
    )
    assert out == docs


@pytest.mark.asyncio
async def test_enrich_with_full_text_respects_total_budget(monkeypatch) -> None:
    """If the batch exceeds total_budget_s, return whatever's done plus
    the rest unchanged. Don't raise."""
    from core.knowledge import _enrich_with_full_text, RetrievedDoc as RD

    docs = [
        RD(content=f"d{i}", metadata={"source": "x", "url": f"https://e/{i}"})
        for i in range(3)
    ]

    def slow_fetch(doc, *, timeout_s, max_kb):
        import time as _t
        _t.sleep(10)  # exceeds the 0.01s budget below
        return "should never arrive"
    monkeypatch.setattr("core.knowledge._fetch_full_text", slow_fetch)

    out = await _enrich_with_full_text(
        docs, timeout_s=5, total_budget_s=0.01, max_kb=64,
    )
    # Budget exceeded → returns originals unchanged, no raise.
    assert all(d.metadata.get("fetched_full_text") is not True for d in out)


@pytest.mark.asyncio
async def test_enrich_with_full_text_returns_partial_results_on_budget(
    monkeypatch,
) -> None:
    """If some fetches complete and others run long, the budget timer
    must NOT discard the completed work. The previous implementation
    used `wait_for(gather(...))` which cancels every in-flight task and
    drops their results on timeout — partial enrichment was a fiction.
    The current `as_completed` loop applies finished results as they
    land and only abandons the still-running ones at the deadline."""
    import time as _t
    from core.knowledge import _enrich_with_full_text, RetrievedDoc as RD

    docs = [
        RD(content="fast", metadata={"source": "x", "url": "https://e/fast"}),
        RD(content="slow", metadata={"source": "x", "url": "https://e/slow"}),
    ]

    def mixed_fetch(doc, *, timeout_s, max_kb):
        # First doc returns immediately; second blocks past the budget.
        if doc.metadata["url"].endswith("fast"):
            return "fetched body for the fast doc"
        _t.sleep(2.0)
        return "this should be abandoned"
    monkeypatch.setattr("core.knowledge._fetch_full_text", mixed_fetch)

    out = await _enrich_with_full_text(
        docs, timeout_s=5, total_budget_s=0.5, max_kb=64,
    )
    # Fast doc gets enriched; slow doc is left untouched.
    assert out[0].metadata.get("fetched_full_text") is True
    assert "fetched body for the fast doc" in out[0].content
    assert out[1].metadata.get("fetched_full_text") is not True
    assert out[1].content == "slow"


def test_source_catalog_known_searchable_names_match_registry() -> None:
    """Every catalog entry with `has_search_adapter=True` must name an
    adapter that exists in `_SOURCE_REGISTRY`. Catches drift if someone
    adds a catalog entry without wiring the adapter (or vice versa)."""
    from core.knowledge import _SOURCE_CATALOG, _SOURCE_REGISTRY

    for entry in _SOURCE_CATALOG:
        if entry["has_search_adapter"]:
            assert entry["name"] in _SOURCE_REGISTRY, (
                f"catalog entry {entry['name']!r} claims has_search_adapter "
                f"but is not in _SOURCE_REGISTRY"
            )


def test_source_catalog_seeds_axon_when_enabled(monkeypatch) -> None:
    """When `seed_source_catalog=True` (default) and Axon has an
    `add_text` method, one document per catalog entry lands at construction."""
    from core.knowledge import _SOURCE_CATALOG

    brain = _BrainAddText()
    cfg = KnowledgeConfig(enabled=True, seed_source_catalog=True)
    # Bypass the AxonBrain build path and inject our stub.
    monkeypatch.setattr("core.knowledge._AXON_AVAILABLE", True)
    monkeypatch.setattr(
        "core.knowledge.Knowledge._build_brain",
        staticmethod(lambda c: brain),
    )
    # Skip the AxonRetriever ctor.
    monkeypatch.setattr(
        "core.knowledge.AxonRetriever",
        lambda **kw: type("R", (), {"invoke": lambda s, q: [], "with_overrides": lambda s, _: s})(),
    )

    k = Knowledge(cfg)
    assert k.enabled is True
    assert len(brain.calls) == len(_SOURCE_CATALOG)
    # First doc is a catalog entry, not a quest paper.
    first_text, first_meta = brain.calls[0]
    assert first_meta["kind"] == "fi_source_catalog"
    assert first_meta["source_name"] in {e["name"] for e in _SOURCE_CATALOG}


@pytest.mark.asyncio
async def test_source_router_picks_sources_from_catalog() -> None:
    """The LLM-routing path: agent reads the catalog + topic and
    returns a list of source names. Only names that exist in
    `_SOURCE_REGISTRY` are honored."""
    from core.knowledge import _route_sources_with_llm

    async def chat(messages, **kw):
        return (
            '{"sources_to_query": ["arxiv", "openalex", "spie"], '
            '"noteworthy_venues": ["nature_portfolio"], '
            '"rationale": "physics + paywalled venues"}'
        )

    picked = await _route_sources_with_llm(
        topic="Bernstein-Vazirani depolarizing",
        chosen_idea={"title": "noise floor"},
        chat_fn=chat,
        fallback_sources=["openalex"],
    )
    # "spie" is in the catalog but has no adapter → filtered out.
    # arxiv and openalex are kept.
    assert picked == ["arxiv", "openalex"]


@pytest.mark.asyncio
async def test_source_router_falls_back_on_parse_failure() -> None:
    """If the LLM response is unparseable, fall back to the config list."""
    from core.knowledge import _route_sources_with_llm

    async def garbage(messages, **kw):
        return "I dunno, try arxiv I guess"

    picked = await _route_sources_with_llm(
        topic="x", chosen_idea=None, chat_fn=garbage,
        fallback_sources=["openalex", "crossref"],
    )
    assert picked == ["openalex", "crossref"]


@pytest.mark.asyncio
async def test_source_router_falls_back_when_no_valid_sources() -> None:
    """If LLM only suggests unsearchable venues, fall back to config."""
    from core.knowledge import _route_sources_with_llm

    async def only_unsearchable(messages, **kw):
        return '{"sources_to_query": ["spie", "ieee_xplore"]}'

    picked = await _route_sources_with_llm(
        topic="EUV lithography", chosen_idea=None,
        chat_fn=only_unsearchable, fallback_sources=["openalex"],
    )
    # Both are catalog-only, so we fall back.
    assert picked == ["openalex"]


def test_arxiv_fallback_tolerates_network_failure(monkeypatch) -> None:
    """A network error during the fallback returns [] (no raise) —
    literature retrieval is best-effort."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = False  # Axon disabled entirely
    k.cfg = KnowledgeConfig(enabled=True, external_fallback="arxiv")

    class _Bad:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, *a, **kw): raise httpx.ConnectError("DNS down")

    monkeypatch.setattr("core.knowledge.httpx.Client", _Bad)
    assert k.search("anything") == []


def test_add_quest_artifacts_calls_brain_ingest_once_per_quest(tmp_path: Path) -> None:
    """Real Axon API is ``brain.ingest(documents)`` taking a LIST of
    docs. The full bundle (spine, body, summary, topic event) lands
    in a SINGLE call so the embed/index flush — driven by
    ``finalize_ingest`` — runs once for the whole quest, not five
    times. Earlier versions of this code mistakenly called a per-doc
    ``add_text`` that didn't exist on the brain at all."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("# body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="qX", paper_md_path=paper, summary="s")
    assert ok is True
    assert len(brain.batches) == 1, (
        "the structured bundle must batch into a single ingest call"
    )
    assert brain.finalize_calls == 1, (
        "finalize_ingest must run once so embeddings flush after the batch"
    )


def test_add_quest_artifacts_handles_missing_paper_with_add_text(tmp_path: Path) -> None:
    """If the paper file is missing, the body doc's payload is just the
    citation header (no body content) — the spine, summary, and topic
    event still land normally so the quest remains discoverable."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "missing.md"  # not created

    ok = k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="x")

    assert ok is True
    by_kind = _calls_by_kind(brain)
    body_text, _ = by_kind["fi_quest_paper"]
    # `_prepend_paper_header(empty, meta)` → "<header>\n" (no body).
    assert body_text.startswith("[")
    assert body_text.endswith("\n") or body_text.count("\n") == 1


# --- ingest -----------------------------------------------------------------


def test_ingest_reads_file_and_calls_brain_ingest(tmp_path: Path) -> None:
    """``Knowledge.ingest(file_path)`` reads the file and calls the
    real ``brain.ingest([{id, text, metadata}])`` API, then flushes
    via ``finalize_ingest``. The previous version called a
    non-existent ``ingest(str)``/``add_document(str)`` path and
    silently returned False."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    f = tmp_path / "doc.txt"
    f.write_text("hello payload", encoding="utf-8")
    assert k.ingest(f) is True
    assert len(brain.batches) == 1
    doc = brain.batches[0][0]
    assert doc["text"] == "hello payload"
    assert doc["id"] == "doc.txt"
    assert doc["metadata"]["kind"] == "fi_local_paper"
    assert brain.finalize_calls == 1


def test_ingest_returns_false_when_brain_has_no_ingest_api(tmp_path: Path) -> None:
    brain = object()  # no ingest method
    k = _enabled_knowledge_with(brain)
    f = tmp_path / "doc.txt"
    f.write_text("x", encoding="utf-8")
    assert k.ingest(f) is False


# --- RetrievedDoc decoupling -------------------------------------------------


def test_retrieved_doc_is_a_plain_dataclass() -> None:
    """Sanity-check: callers can build it without importing langchain."""
    d = RetrievedDoc(content="hi", metadata={"k": 1})
    assert d.content == "hi"
    assert d.metadata == {"k": 1}


# ---------------------------------------------------------------------------
# Structured-ingest helpers (Patterns A spine / B header / C topic event)
# ---------------------------------------------------------------------------


def test_slugify_topic_normalizes_and_truncates() -> None:
    from core.knowledge import _slugify_topic

    assert _slugify_topic("EUV-MOR SE yield, 2026!") == "euv-mor-se-yield-2026"
    assert _slugify_topic("") == "untitled-topic"
    assert _slugify_topic("   ") == "untitled-topic"
    long = "x" * 200
    assert len(_slugify_topic(long)) <= 80


def test_paper_short_id_prefers_doi_then_arxiv_then_pmid_then_title() -> None:
    from core.knowledge import _paper_short_id

    assert _paper_short_id({"doi": "10.1117/12.X", "arxiv_id": "2401.1"}) == "doi:10.1117/12.x"
    assert _paper_short_id({"arxiv_id": "2401.00001"}) == "arxiv:2401.00001"
    assert _paper_short_id({"pmid": "12345"}) == "pmid:12345"
    assert _paper_short_id({"title": "Some Big Title"}) == "title:some-big-title"
    assert _paper_short_id({}) == "unknown"


def test_paper_header_line_compact_citation() -> None:
    from core.knowledge import _paper_header_line

    line = _paper_header_line({
        "title": "T", "authors": ["A. Lastname", "B. Other"],
        "year": 2024, "venue": "Proc. SPIE", "doi": "10.1/abc",
    })
    assert line.startswith("[") and line.endswith("]")
    assert "T" in line and "A. Lastname et al." in line
    assert "2024" in line and "Proc. SPIE" in line and "DOI:10.1/abc" in line


def test_prepend_paper_header_keeps_body_intact() -> None:
    from core.knowledge import _prepend_paper_header

    out = _prepend_paper_header(
        "body content...", {"title": "Pap", "authors": ["X"], "year": 2020},
    )
    assert out.startswith("[Pap ·")
    assert out.endswith("body content...")
    assert "\n" in out  # header is its own line


def test_render_paper_spine_packs_title_authors_doi_and_claims() -> None:
    from core.knowledge import _render_paper_spine

    text = _render_paper_spine(
        {
            "title": "EUV-MOR sensitization",
            "authors": ["Grenville", "Stowers"],
            "year": 2015, "venue": "Proc. SPIE 9425",
            "doi": "10.1117/12.X",
            "topic": "EUV resist sensitization",
        },
        abstract="Fano factor characterization at 92 eV.",
        key_claims=["F_SE ≈ 2.3", "dose-to-clear scales sub-linearly"],
    )
    assert "TITLE: EUV-MOR sensitization" in text
    assert "AUTHORS: Grenville, Stowers" in text
    assert "YEAR: 2015" in text
    assert "VENUE: Proc. SPIE 9425" in text
    assert "DOI: 10.1117/12.X" in text
    assert "TOPIC: EUV resist sensitization" in text
    assert "Fano factor" in text
    assert "KEY CLAIMS:" in text
    assert "- F_SE ≈ 2.3" in text


def test_render_topic_event_lists_quest_and_paper_refs() -> None:
    from core.knowledge import _render_topic_event

    text = _render_topic_event(
        topic="EUV-MOR sensitization",
        topic_id="euv-mor-sensitization",
        quest_id="q_test",
        quest_title="MOR dose nonlinearity",
        verdict="accept", score=4,
        paper_refs=[
            {"title": "Paper A", "doi": "10.1/a"},
            {"title": "Paper B", "arxiv_id": "2401.5"},
        ],
    )
    assert "TOPIC_ID: euv-mor-sensitization" in text
    assert "QUEST: q_test" in text
    assert "Title: MOR dose nonlinearity" in text
    assert "Verdict: accept" in text
    assert "Score: 4" in text
    assert "PAPERS CITED (2):" in text
    assert "Paper A" in text and "DOI:10.1/a" in text
    assert "Paper B" in text and "arXiv:2401.5" in text


# ---------------------------------------------------------------------------
# add_quest_artifacts: external-ref spines + topic-event scoping
# ---------------------------------------------------------------------------


def test_add_quest_artifacts_writes_external_ref_spines(tmp_path: Path) -> None:
    """When `metadata['external_refs']` lists papers the quest consumed,
    one `fi_external_ref_spine` doc lands per ref, each tagged with the
    quest's topic_id so a topic-scoped query can recover the references."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("body", encoding="utf-8")

    meta = {
        "title": "MOR dose nonlinearity",
        "topic": "EUV-MOR sensitization",
        "verdict": "accept",
        "score": 4,
        "external_refs": [
            {"title": "Grenville 2015", "authors": ["Grenville"],
             "year": 2015, "venue": "Proc. SPIE 9425",
             "doi": "10.1117/12.X", "abstract": "Fano factor at 92 eV."},
            {"title": "Hinsberg 2017", "authors": ["Hinsberg", "Meyers"],
             "year": 2017, "venue": "Proc. SPIE 10146",
             "arxiv_id": "", "abstract": "MOR imaging model."},
            # Malformed entry (no identifying field) — still allowed: the
            # ingest path doesn't filter; the spine will key on title hash.
            {"title": "Stub", "abstract": "x"},
        ],
    }

    ok = k.add_quest_artifacts(
        quest_id="q_euv", paper_md_path=paper,
        summary="Confirmed sub-linear dose response.", metadata=meta,
    )
    assert ok is True

    # 4 core docs + 3 external-ref spines = 7 calls.
    assert len(brain.calls) == 7

    ref_calls = [c for c in brain.calls if c[1].get("kind") == "fi_external_ref_spine"]
    assert len(ref_calls) == 3

    # Topic id used as the linking key on every ref spine.
    for text, m in ref_calls:
        assert m["topic_id"] == "euv-mor-sensitization"
        assert m["consumed_by_quests"] == ["q_euv"]
        assert m["tag"].startswith("fi-paper:")

    # First ref carries its DOI through into both text and metadata.
    grenville = next((t, m) for t, m in ref_calls if "Grenville" in t)
    assert "DOI: 10.1117/12.X" in grenville[0]
    assert grenville[1]["doi"] == "10.1117/12.X"
    assert grenville[1]["paper_id"] == "doi:10.1117/12.x"

    # The structured-summary doc records paper_refs in its metadata (the
    # short-id list, not the full ref blobs — JSON-safe and compact).
    by_kind = _calls_by_kind(brain)
    _, summary_meta = by_kind["fi_quest_summary"]
    assert "doi:10.1117/12.x" in summary_meta["paper_refs"]
    # And the JSON envelope in the summary text carries the full refs.
    summary_text, _ = by_kind["fi_quest_summary"]
    assert "Fano factor at 92 eV" in summary_text

    # Topic event lists all three refs by title.
    topic_text, topic_meta = by_kind["fi_topic_event"]
    assert "Grenville 2015" in topic_text
    assert "Hinsberg 2017" in topic_text
    assert "PAPERS CITED (3):" in topic_text
    assert topic_meta["topic_id"] == "euv-mor-sensitization"


def test_add_quest_artifacts_ingests_no_ref_spines_when_no_external_refs(
    tmp_path: Path,
) -> None:
    """No `external_refs` → no `fi_external_ref_spine` docs. The four
    core docs (spine, body, summary, topic event) still land."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("body", encoding="utf-8")

    ok = k.add_quest_artifacts(
        quest_id="q_solo", paper_md_path=paper, summary="s",
        metadata={"title": "T", "topic": "alpha", "verdict": "accept"},
    )
    assert ok is True
    kinds = [c[1].get("kind") for c in brain.calls]
    assert kinds.count("fi_external_ref_spine") == 0
    assert set(kinds) == {
        "fi_paper_spine", "fi_quest_paper", "fi_quest_summary", "fi_topic_event",
    }


def test_local_paper_ingestion_writes_spine_and_body() -> None:
    """`_ingest_local_papers_into_axon` now lays down BOTH a spine and a
    header-prepended body per local paper, so a future title query lands
    on the spine and a fact-level query lands on a body chunk that still
    carries provenance."""
    import tempfile
    brain = _BrainAddText()

    # Build Knowledge with a tmpfile local paper and inject the brain.
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "grenville_2015_test.md"
        f.write_text("body of the paper", encoding="utf-8")

        k = Knowledge(KnowledgeConfig(enabled=False))
        k.enabled = True
        k.cfg = KnowledgeConfig(enabled=True, local_papers=[f])
        k._brain = brain
        # Manually load and run the ingest step (Knowledge.__init__ did
        # not see the brain because we forced enabled=False at construct).
        from core.knowledge import _load_local_papers
        k._local_papers = _load_local_papers([f])
        k._ingest_local_papers_into_axon()

    # Spine + body per local paper.
    assert len(brain.calls) == 2
    kinds = {c[1].get("kind") for c in brain.calls}
    assert kinds == {"fi_paper_spine", "fi_local_paper"}
    by_kind = _calls_by_kind(brain)

    spine_text, spine_meta = by_kind["fi_paper_spine"]
    assert "TITLE:" in spine_text
    assert spine_meta["origin"] == "local_paper"
    assert spine_meta["paper_id"].startswith("title:")  # no DOI on raw local md

    body_text, body_meta = by_kind["fi_local_paper"]
    assert body_text.startswith("[")  # citation header
    assert "body of the paper" in body_text
    # Spine and body share the same paper_id so a chunk can be walked
    # back to the spine via metadata.
    assert body_meta["paper_id"] == spine_meta["paper_id"]
    assert body_meta["tag"] == spine_meta["tag"]
