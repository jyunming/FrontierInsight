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
from core.knowledge import Knowledge, RetrievedDoc, _apply_offline_env


# --- offline env application -------------------------------------------------


_HF_ENV_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME")


@pytest.fixture
def clean_hf_env():
    """Snapshot + restore HF env around a test. ``_apply_offline_env``
    mutates ``os.environ`` directly (it's process-global by design), so a
    plain monkeypatch can't undo it — without this, the offline vars leak
    into later tests in the same pytest process. Also snapshots and clears
    the module's ``_APPLIED_OFFLINE_ENV`` tracker so the first-quest-wins
    conflict detection starts fresh each test instead of seeing a value a
    previous test recorded."""
    import os
    from core import knowledge as _k
    saved = {k: os.environ.get(k) for k in _HF_ENV_KEYS}
    saved_applied = dict(_k._APPLIED_OFFLINE_ENV)
    for k in _HF_ENV_KEYS:
        os.environ.pop(k, None)
    _k._APPLIED_OFFLINE_ENV.clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _k._APPLIED_OFFLINE_ENV.clear()
        _k._APPLIED_OFFLINE_ENV.update(saved_applied)


def test_apply_offline_env_sets_hf_vars(clean_hf_env) -> None:
    """offline=True + models_dir → HF offline vars + HF_HOME, so model
    loads stay local and never hit huggingface.co."""
    import os
    _apply_offline_env(
        KnowledgeConfig(enabled=False, offline=True, models_dir=Path("/tmp/m"))
    )
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HOME"] == str(Path("/tmp/m"))


def test_apply_offline_env_noop_when_off(clean_hf_env) -> None:
    """offline=False, no models_dir → leaves HF env untouched.

    ``models_dir`` is pinned to ``None`` explicitly: the field otherwise
    defaults from ``FI_MODELS_DIR``, so a runner that exports it would make
    this test set ``HF_HOME`` and flip the assertion. With it pinned the
    config is deterministic regardless of the ambient environment, and we
    assert ``HF_HOME`` stays unset alongside the offline flags."""
    import os
    _apply_offline_env(KnowledgeConfig(enabled=False, offline=False, models_dir=None))
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_HOME" not in os.environ


def test_apply_offline_env_first_wins_on_divergent_models_dir(
    clean_hf_env, caplog
) -> None:
    """The --fleet cross-talk case: two quests with different ``models_dir``.
    HF env vars are process-global, so the second apply must NOT clobber the
    first quest's ``HF_HOME`` (a brain may still be lazily loading from it).
    First-wins, and the divergence is logged so the misconfiguration is
    visible instead of silently nondeterministic."""
    import logging
    import os
    _apply_offline_env(
        KnowledgeConfig(enabled=False, offline=True, models_dir=Path("/tmp/a"))
    )
    with caplog.at_level(logging.WARNING, logger="frontier_insight.knowledge"):
        _apply_offline_env(
            KnowledgeConfig(enabled=False, offline=True, models_dir=Path("/tmp/b"))
        )
    # First quest's cache root is preserved; the matching offline flags
    # (both "1") are not a conflict and stay set.
    assert os.environ["HF_HOME"] == str(Path("/tmp/a"))
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert "offline-env conflict" in caplog.text


# --- disabled-path contract --------------------------------------------------


def test_search_returns_empty_when_disabled_and_no_external_fallback() -> None:
    # With Axon disabled AND no external sources AND web search off, search
    # returns []. (web_search is the always-on layer — it's disabled here to
    # isolate the academic/Axon path; its own tests cover the web layer.)
    k = Knowledge(KnowledgeConfig(
        enabled=False, external_fallback="none", web_search=False,
    ))
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
    through to the configured external fallback (arxiv by default).
    The external layer uses ``external_top_k`` (not ``top_k``) so a
    web miss can return broader coverage than the RAG cap allows."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True  # bypass axon import
    # Pin fallback to arxiv-only so the test doesn't have to stub other
    # sources' JSON responses. external_top_k=3 keeps the test
    # deterministic — without setting it the default is 20 and the
    # arxiv stub would need to return 20 entries.
    k.cfg = KnowledgeConfig(
        enabled=True, external_fallback=["arxiv"],
        top_k=3, external_top_k=3,
    )
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


def test_external_top_k_used_when_caller_explicitly_passes_it(monkeypatch) -> None:
    """The literature node is the one caller that wants broad external
    retrieval when Axon misses — it passes ``external_top_k`` explicitly,
    so the web layer fetches the bigger config cap instead of being
    silently bounded by the caller's per-call ``top_k``."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, external_fallback=["arxiv"],
        top_k=5, external_top_k=20,
    )
    class _Empty:
        def invoke(self, _): return []
        def with_overrides(self, _): return self
    k._retriever = _Empty()

    captured: dict = {}
    atom = _arxiv_atom(*[
        {"id": f"24{i:02d}.0001", "title": f"paper {i}", "summary": "x"}
        for i in range(20)
    ])

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):
            captured["params"] = params
            r = MagicMock()
            r.text = atom
            r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _FakeClient)
    import asyncio as _asyncio
    _asyncio.run(k.asearch("anything", top_k=5, external_top_k=20))
    assert captured["params"]["max_results"] == "20"


def test_disabled_knowledge_does_not_web_search(monkeypatch) -> None:
    """knowledge.enabled=False is the master switch — it must NOT reach web
    search, even with web_search=True (audit #8). Previously want_web looked
    only at web_search (default on), so disabling the layer still hit the net."""
    from core.knowledge import Knowledge
    import asyncio

    called = {"web": 0}

    def fake_web(*a, **k):  # noqa: ANN001
        called["web"] += 1
        return []

    monkeypatch.setattr("core.knowledge._web_search", fake_web)

    # enabled=False + web_search=True → still no web.
    k = Knowledge(KnowledgeConfig(enabled=False, web_search=True))
    k.cfg = KnowledgeConfig(enabled=False, web_search=True)
    asyncio.run(k.asearch("anything"))
    assert called["web"] == 0, "enabled=False must not web-search"

    # Control: enabled=True + web_search=True DOES web-search, even without
    # an Axon corpus (web is independent of the Axon brain when enabled).
    k.cfg = KnowledgeConfig(enabled=True, web_search=True, external_fallback=[])
    asyncio.run(k.asearch("anything"))
    assert called["web"] == 1, "enabled=True + web_search must web-search"


def test_external_top_k_honours_caller_per_call_cap(monkeypatch) -> None:
    """When the caller passes ``top_k=3`` (e.g. ideate seeding, cross_check)
    WITHOUT an explicit ``external_top_k``, the external layer must
    respect the per-call cap. PR #113's first cut regressed this:
    an ideate seed asking for 3 papers got 20 web hits instead."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, external_fallback=["arxiv"],
        top_k=5, external_top_k=20,
    )
    class _Empty:
        def invoke(self, _): return []
        def with_overrides(self, _): return self
    k._retriever = _Empty()

    captured: dict = {}
    atom = _arxiv_atom(
        {"id": "2401.0001", "title": "a", "summary": "x"},
        {"id": "2401.0002", "title": "b", "summary": "x"},
        {"id": "2401.0003", "title": "c", "summary": "x"},
    )

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):
            captured["params"] = params
            r = MagicMock()
            r.text = atom
            r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _FakeClient)
    import asyncio as _asyncio
    _asyncio.run(k.asearch("ideate seed", top_k=3))
    assert captured["params"]["max_results"] == "3"


def test_external_top_k_uses_config_when_no_caller_caps(monkeypatch) -> None:
    """When ``asearch`` is called with no caps at all, the external
    layer falls through to ``cfg.external_top_k`` (the broad default)."""
    from core.knowledge import Knowledge

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, external_fallback=["arxiv"],
        top_k=8, external_top_k=20,
    )
    class _Empty:
        def invoke(self, _): return []
        def with_overrides(self, _): return self
    k._retriever = _Empty()

    captured: dict = {}
    atom = _arxiv_atom(*[
        {"id": f"24{i:02d}.0001", "title": f"paper {i}", "summary": "x"}
        for i in range(20)
    ])

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):
            captured["params"] = params
            r = MagicMock()
            r.text = atom
            r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _FakeClient)
    import asyncio as _asyncio
    _asyncio.run(k.asearch("anything"))
    assert captured["params"]["max_results"] == "20"


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
                          external_fallback="none", web_search=False,
                          local_papers=[a, b], top_k=5)
    k = Knowledge(cfg)
    # No Axon, no external, no web; asearch returns just the pinned papers.
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

    # Pin external_top_k too so the test stays deterministic — the
    # default (20) would let the fake router return up to 19 external
    # docs alongside the pinned paper; the test only cares about the
    # head-of-list invariant (local paper first, then external).
    # enabled=True: this test exercises the external supplement path, which
    # is network retrieval and now (audit #8) requires the layer enabled.
    cfg = KnowledgeConfig(
        enabled=True, source_routing="manual",
        external_fallback=["arxiv"], web_search=False, local_papers=[local],
        top_k=4, external_top_k=4,
    )
    k = Knowledge(cfg)
    import asyncio
    docs = asyncio.run(k.asearch("test"))
    assert len(docs) == 4
    assert docs[0].metadata["source"] == "local_paper"
    assert all(d.metadata["source"] == "fake" for d in docs[1:])


# ---------------------------------------------------------------------------
# General web search — Brave / DuckDuckGo adapters + always-parallel asearch
# ---------------------------------------------------------------------------


def _brave_payload(*results: dict) -> dict:
    return {"web": {"results": list(results)}}


def test_brave_search_parses_results(monkeypatch) -> None:
    from core.knowledge import _brave_search

    payload = _brave_payload(
        {"title": "SpaceX revenue 2023", "url": "https://e.com/a",
         "description": "SpaceX booked <strong>$8B</strong> in 2023",
         "meta_url": {"hostname": "e.com"}, "page_age": "2024-01-02"},
        {"title": "Falcon 9", "url": "https://x.org/b", "description": "launch"},
    )

    class _Ok:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, headers=None, **_):
            assert "brave.com" in url
            assert headers["X-Subscription-Token"] == "BSAkey"
            r = MagicMock(); r.json = lambda: payload; r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _Ok)
    docs = _brave_search("spacex revenue", 5, api_key="BSAkey")
    assert len(docs) == 2
    d = docs[0]
    assert d.metadata["source"] == "web_search" and d.metadata["backend"] == "brave"
    assert d.metadata["url"] == "https://e.com/a"
    assert d.metadata["site"] == "e.com"
    # <strong> markup stripped from the snippet.
    assert d.metadata["snippet"] == "SpaceX booked $8B in 2023"


def test_brave_search_returns_empty_without_key() -> None:
    from core.knowledge import _brave_search
    assert _brave_search("anything", 5, api_key="") == []


def test_web_search_auto_falls_back_to_ddg_without_key(monkeypatch) -> None:
    from core.knowledge import RetrievedDoc as RD
    import core.knowledge as kn

    called = {}
    def fake_ddg(q, tk, *, timeout_s=10.0):
        called["ddg"] = True
        return [RD(content="x", metadata={"source": "web_search", "backend": "duckduckgo"})]
    monkeypatch.setattr("core.knowledge._ddg_search", fake_ddg)

    out = kn._web_search("q", 3, backend="auto", api_key="")
    assert called.get("ddg") is True
    assert out and out[0].metadata["backend"] == "duckduckgo"


def test_ddg_decodes_redirect_links() -> None:
    from core.knowledge import _ddg_decode
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fa%3D1&rut=x"
    assert _ddg_decode(href) == "https://example.com/page?a=1"
    assert _ddg_decode("https://plain.example/p") == "https://plain.example/p"


def _force_fallback_extractor(monkeypatch) -> None:
    """Make ``import trafilatura`` fail so _html_to_text uses its built-in
    tag-strip fallback — the path under test here (and the one a no-extras
    install actually runs)."""
    import sys
    monkeypatch.setitem(sys.modules, "trafilatura", None)


def test_html_to_text_strips_chrome_and_tags(monkeypatch) -> None:
    _force_fallback_extractor(monkeypatch)
    from core.knowledge import _html_to_text
    html = (
        "<html><head><style>.x{}</style></head><body>"
        "<nav>menu</nav><h1>Heading</h1><p>Body <strong>text</strong> here.</p>"
        "<ul><li>one</li><li>two</li></ul><script>evil()</script></body></html>"
    )
    out = _html_to_text(html)
    assert "menu" not in out and "evil" not in out and ".x{}" not in out
    assert "Heading" in out and "Body text here." in out
    assert "- one" in out and "- two" in out


def test_html_to_text_appends_figure_captions_and_prefers_article(monkeypatch) -> None:
    _force_fallback_extractor(monkeypatch)
    from core.knowledge import _html_to_text
    html = (
        "<html><body><nav>chrome junk</nav>"
        "<article><h1>EV Report</h1><p>BEV share hit 18%.</p>"
        "<figure><img src='x.png' alt='EV sales by region 2024'>"
        "<figcaption>Figure 1: regional EV shares</figcaption></figure>"
        "</article><footer>more junk</footer></body></html>"
    )
    out = _html_to_text(html)
    assert "BEV share hit 18%." in out
    assert "chrome junk" not in out          # nav dropped
    assert "[FIGURES IN SOURCE]" in out
    assert "Figure 1: regional EV shares" in out
    assert "EV sales by region 2024" in out  # img alt surfaced


def test_html_to_text_prefers_trafilatura_when_available(monkeypatch) -> None:
    """When trafilatura is importable, _html_to_text returns its extraction
    (plus the figure note) rather than the tag-strip fallback."""
    import sys, types
    fake = types.ModuleType("trafilatura")
    fake.extract = lambda html, **kw: "TRAFILATURA BODY"
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    from core.knowledge import _html_to_text
    out = _html_to_text("<html><body><p>raw tag-strip body</p></body></html>")
    assert "TRAFILATURA BODY" in out
    assert "raw tag-strip body" not in out


def test_brave_search_folds_extra_snippets(monkeypatch) -> None:
    from core.knowledge import _brave_search
    payload = _brave_payload({
        "title": "EV outlook", "url": "https://e.com/x",
        "description": "BEV share rose.",
        "extra_snippets": ["China led at 35%.", "Europe held ~21%."],
    })

    class _Ok:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, headers=None, **_):
            r = MagicMock(); r.json = lambda: payload; r.raise_for_status = MagicMock()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _Ok)
    docs = _brave_search("ev", 3, api_key="BSAk")
    snip = docs[0].metadata["snippet"]
    assert "BEV share rose." in snip
    assert "China led at 35%." in snip and "Europe held ~21%." in snip


def test_low_quality_domains_filtered() -> None:
    from core.knowledge import _is_low_quality_domain
    assert _is_low_quality_domain("https://www.marketsandmarkets.com/Market-Reports/ev-209.html")
    assert _is_low_quality_domain("https://grandviewresearch.com/industry-analysis/ev-market")
    assert _is_low_quality_domain("https://evwire.com/p/x")
    assert not _is_low_quality_domain("https://www.iea.org/reports/global-ev-outlook-2024/x")
    assert not _is_low_quality_domain("https://ourworldindata.org/electric-car-sales")
    assert not _is_low_quality_domain("")


def test_web_search_drops_low_quality(monkeypatch) -> None:
    import core.knowledge as kn
    from core.knowledge import RetrievedDoc as RD
    monkeypatch.setattr("core.knowledge._ddg_search", lambda q, tk, *, timeout_s=10.0: [
        RD("good", {"source": "web_search", "url": "https://iea.org/x"}),
        RD("junk", {"source": "web_search", "url": "https://marketsandmarkets.com/Market-Reports/y"}),
    ])
    out = kn._web_search("ev", 5, backend="duckduckgo")
    assert [d.metadata["url"] for d in out] == ["https://iea.org/x"]


def test_fetch_web_page_text_headless_fallback_on_403(monkeypatch) -> None:
    import core.knowledge as kn
    from core.knowledge import RetrievedDoc as RD

    class _Blocked:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **_):
            r = MagicMock(); r.status_code = 403; return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _Blocked)
    # Realistic recovered article body (> _MIN_FULL_TEXT_CHARS) — a real
    # render returns a full article, not a one-liner, and the fetcher now
    # rejects sub-threshold stubs as block pages.
    _body = "Recovered full text 18%. " + (
        "The renderer cleared the challenge and returned the real article "
        "body with substantive content for the writer to quote. " * 6
    )
    monkeypatch.setattr(
        "core.knowledge._playwright_fetch_html",
        lambda url, *, timeout_s: f"<html><body><article><p>{_body}</p></article></body></html>",
    )
    doc = RD("", {"url": "https://iea.example/blocked"})
    # headless on → recovers via the renderer.
    out = kn._fetch_web_page_text(doc, timeout_s=5, max_kb=32, headless=True)
    assert out and "Recovered full text 18%." in out
    # headless off → blocked stays blocked (snippet only).
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=32, headless=False) is None


def test_looks_like_bot_challenge() -> None:
    import core.knowledge as kn
    assert kn._looks_like_bot_challenge("")                       # empty
    assert kn._looks_like_bot_challenge(
        "Checking your browser - reCAPTCHA. Click here if not redirected."
    )
    assert kn._looks_like_bot_challenge(
        "Please enable JavaScript and cookies to continue."
    )
    # A long real article that merely MENTIONS recaptcha is not a challenge.
    article = "This paper studies how reCAPTCHA affects user behaviour. " * 60
    assert not kn._looks_like_bot_challenge(article)
    # Short text with no challenge marker is not flagged here (the
    # snippet-relative gate in _fetch_web_page_text drops stubs that don't
    # beat the snippet).
    assert not kn._looks_like_bot_challenge("A legitimate short abstract sentence.")


def test_fetch_web_page_text_rejects_200_challenge_page(monkeypatch) -> None:
    """A 200 that's actually a reCAPTCHA / cookie wall (PMC, MDPI, …) must
    be rejected, not stored as 165-byte 'full text'."""
    import core.knowledge as kn
    from core.knowledge import RetrievedDoc as RD

    challenge = (
        "<html><body>Checking your browser - reCAPTCHA. "
        "Click here if you are not automatically redirected.</body></html>"
    )

    class _Chal:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **_):
            r = MagicMock()
            r.status_code = 200
            r.headers = {"content-type": "text/html"}
            r.content = challenge.encode()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _Chal)
    # Non-academic URL → the open-access API fallback is a no-op, so the
    # result is None (the caller keeps the real search snippet).
    doc = RD("snippet", {"url": "https://example.com/blocked"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=32, headless=False) is None


def test_keep_fetched_text_is_snippet_relative() -> None:
    """The fetched-text gate is relative to the snippet we already have:
    keep page text only when it beats the snippet (so a legitimately short
    page survives), drop a stub no longer than the snippet, and always drop
    a bot-challenge regardless of length."""
    import core.knowledge as kn
    snippet = "A short search-result snippet we already have on file"
    # Longer than the snippet → kept, even though it's well under the old
    # absolute 300-char floor (a real short press release / stats page).
    assert kn._keep_fetched_text(snippet + " plus genuinely new content.", snippet)
    # No longer than the snippet → no improvement, drop it (keep snippet).
    assert not kn._keep_fetched_text("shorter than the snippet", snippet)
    # A challenge marker is dropped however long relative to the snippet.
    assert not kn._keep_fetched_text(
        "Checking your browser - reCAPTCHA. " + "x" * 40, snippet
    )
    # With no snippet to beat, fall back to the substance floor.
    assert not kn._keep_fetched_text("tiny", "")
    assert kn._keep_fetched_text("y" * (kn._MIN_FULL_TEXT_CHARS + 1), "")
    assert not kn._keep_fetched_text(None, snippet)


def test_fetch_web_page_text_keeps_short_page_beating_snippet(monkeypatch) -> None:
    """A legit short page (press release / stats page) that's under the old
    300-char floor but longer than the search snippet is now KEPT, not
    dropped — the gate is snippet-relative, not an absolute length floor."""
    import core.knowledge as kn
    from core.knowledge import RetrievedDoc as RD

    page_text = (
        "Renewable capacity rose 12 percent in 2025, the agency said in a "
        "brief release, reaching a new annual record across all regions."
    )
    body = f"<html><body><p>{page_text}</p></body></html>"

    class _Ok:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **_):
            r = MagicMock()
            r.status_code = 200
            r.headers = {"content-type": "text/html"}
            r.content = body.encode()
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _Ok)
    doc = RD("Short snippet.", {"url": "https://example.com/press"})
    out = kn._fetch_web_page_text(doc, timeout_s=5, max_kb=32, headless=False)
    assert out and "12 percent" in out
    assert len(out) < kn._MIN_FULL_TEXT_CHARS  # would've been dropped before


def test_xml_to_text_decodes_entities() -> None:
    """JATS flattening decodes named AND numeric entities (em dash, accents)
    via html.unescape, not a 3-entity allowlist."""
    import core.knowledge as kn
    xml = (
        "<p>Effect size&#x2014;large&#8212;was r &gt; 0.5 for caf&eacute; "
        "&amp; tea in the H&#246;lzel study.</p>"
    )
    out = kn._xml_to_text(xml)
    assert "—" in out        # &#x2014; / &#8212; → em dash
    assert "r > 0.5" in out       # &gt;
    assert "café" in out     # &eacute;
    assert "Hölzel" in out   # &#246;
    assert "& tea" in out         # &amp; preserved as literal ampersand
    assert "&#" not in out and "&amp;" not in out  # no leftover entity tokens


def test_sanitize_search_query_collapses_and_caps() -> None:
    """A multi-line, paragraph-length topic is collapsed to a single line
    and trimmed under Brave's length cap (Brave 422s on newlines / very
    long queries; DuckDuckGo returns nothing), so web search still works."""
    import core.knowledge as kn
    topic = "Line one about greenness\nand mortality.\n\n  Extra   spaces here.\n"
    q = kn._sanitize_search_query(topic)
    assert "\n" not in q and "  " not in q
    assert q == "Line one about greenness and mortality. Extra spaces here."
    # A long topic is capped under the limit at a word boundary (no half word).
    capped = kn._sanitize_search_query("word " * 200, max_chars=380)
    assert len(capped) <= 380
    assert capped.split()[-1] == "word"  # cut on a boundary, not mid-word
    # A short query passes through untouched.
    assert kn._sanitize_search_query("greenness and health") == "greenness and health"


def test_web_search_sanitizes_query_before_dispatch(monkeypatch) -> None:
    """_web_search collapses the multi-line topic before handing it to a
    backend, so the backend never sees newlines (which 422 Brave)."""
    import core.knowledge as kn
    seen = {}

    def _fake_ddg(query, top_k, *, timeout_s=10.0):
        seen["query"] = query
        return []

    monkeypatch.setattr("core.knowledge._ddg_search", _fake_ddg)
    kn._web_search("multi\nline\n\ntopic   blob", 5, backend="duckduckgo")
    assert seen["query"] == "multi line topic blob"


def test_fetch_web_page_text_routes_pdf_to_extractor(monkeypatch) -> None:
    from core.knowledge import _fetch_web_page_text, RetrievedDoc as RD

    class _PdfResp:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **_):
            r = MagicMock()
            r.status_code = 200
            r.headers = {"content-type": "application/pdf"}
            r.content = b"%PDF-1.7\nfake"
            return r

    monkeypatch.setattr("core.knowledge.httpx.Client", _PdfResp)
    monkeypatch.setattr(
        "core.knowledge._pdf_bytes_to_text",
        lambda body, *, cap: "EXTRACTED PDF TEXT",
    )
    doc = RD("", {"url": "https://iea.example/report.pdf"})
    assert _fetch_web_page_text(doc, timeout_s=5, max_kb=32) == "EXTRACTED PDF TEXT"


def _fake_retriever(docs):
    class _R:
        def invoke(self, _q):
            out = []
            for d in docs:
                m = MagicMock(); m.page_content = d.content; m.metadata = d.metadata
                out.append(m)
            return out
        def with_overrides(self, _):
            return self
    return _R()


def test_asearch_runs_web_in_parallel_and_merges_with_axon(monkeypatch) -> None:
    """web_search runs alongside Axon for every query and the two are
    merged (Axon first, then web), even when Axon already returned hits."""
    from core.knowledge import Knowledge, RetrievedDoc as RD

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, web_search=True, web_fetch_pages=False,
        external_fallback="none", top_k=8, external_top_k=10,
    )
    k._retriever = _fake_retriever([RD(content="axon hit", metadata={"src": "axon"})])

    def fake_web(query, top_k, *, backend, api_key, timeout_s):
        return [
            RD(content="w1", metadata={"source": "web_search", "url": "https://a/1"}),
            RD(content="w2", metadata={"source": "web_search", "url": "https://a/2"}),
        ]
    monkeypatch.setattr("core.knowledge._web_search", fake_web)

    import asyncio
    docs = asyncio.run(k.asearch("anything", top_k=8, external_top_k=10))
    srcs = [d.metadata.get("src") or d.metadata.get("source") for d in docs]
    assert srcs == ["axon", "web_search", "web_search"]


def test_asearch_web_fills_when_axon_empty_and_no_academic(monkeypatch) -> None:
    """The non-academic case (e.g. 'SpaceX revenue'): Axon empty, academic
    fallback 'none' — web still supplies real results instead of nothing."""
    from core.knowledge import Knowledge, RetrievedDoc as RD

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, web_search=True, web_fetch_pages=False,
        external_fallback="none", top_k=8, external_top_k=10,
    )
    k._retriever = _fake_retriever([])  # Axon returns nothing

    def fake_web(query, top_k, *, backend, api_key, timeout_s):
        return [RD(content="real spacex page",
                   metadata={"source": "web_search", "url": "https://sec.example/x"})]
    monkeypatch.setattr("core.knowledge._web_search", fake_web)

    import asyncio
    docs = asyncio.run(k.asearch("SpaceX revenue", top_k=8, external_top_k=10))
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "web_search"


def test_asearch_dedupes_web_by_url(monkeypatch) -> None:
    from core.knowledge import Knowledge, RetrievedDoc as RD

    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(
        enabled=True, web_search=True, web_fetch_pages=False,
        external_fallback="none", top_k=8, external_top_k=10,
    )
    k._retriever = _fake_retriever([])

    def fake_web(query, top_k, *, backend, api_key, timeout_s):
        # Same page from two backends (trailing slash + scheme differ).
        return [
            RD(content="a", metadata={"source": "web_search", "url": "https://site.com/p/"}),
            RD(content="b", metadata={"source": "web_search", "url": "http://site.com/p"}),
        ]
    monkeypatch.setattr("core.knowledge._web_search", fake_web)

    import asyncio
    docs = asyncio.run(k.asearch("q", top_k=8, external_top_k=10))
    assert len(docs) == 1  # collapsed to one


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
    via ``finalize_ingest``. Assertion shape is loose — ``id`` must
    be non-empty and derived from the source path, but the exact
    format stays implementation-detail (pinning ``doc.txt`` was too
    coupled and broke when the id strategy tightened against
    same-basename collisions)."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    f = tmp_path / "doc.txt"
    f.write_text("hello payload", encoding="utf-8")
    assert k.ingest(f) is True
    assert len(brain.batches) == 1
    doc = brain.batches[0][0]
    assert doc["text"] == "hello payload"
    assert doc["id"], "doc id must be non-empty"
    assert "doc.txt" in doc["id"]
    assert str(tmp_path.resolve()).replace("\\", "/") in doc["id"], (
        "doc id must carry the resolved parent-path so distinct files "
        "with the same basename don't collide on ingest"
    )
    assert doc["metadata"]["kind"] == "fi_local_paper"
    assert brain.finalize_calls == 1


def test_ingest_distinct_files_with_same_basename_dont_collide(
    tmp_path: Path,
) -> None:
    """Two files named ``data.csv`` from different subdirs ingest as
    distinct documents — bare ``path.name`` would have caused the
    second ingest to overwrite the first."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    a = tmp_path / "a" / "data.csv"
    b = tmp_path / "b" / "data.csv"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text("from a", encoding="utf-8")
    b.write_text("from b", encoding="utf-8")
    assert k.ingest(a) is True
    assert k.ingest(b) is True
    ids = [doc["id"] for batch in brain.batches for doc in batch]
    assert len(set(ids)) == 2, f"expected 2 distinct ids; got {ids}"


def test_mint_doc_id_uses_paper_id_and_rel_path_for_disambiguation() -> None:
    """``_mint_doc_id`` previously only checked the top-level
    quest_id / proposal_id / etc. keys, so ``fi_summary_input`` docs
    that share a summary_id but live at different paths would all
    collide to the same id. New behaviour: composite key including
    ``paper_id``, ``rel_path``, and ``tag`` discriminators."""
    from core.knowledge import Knowledge
    # Same summary_id but different rel_path → distinct ids.
    a = Knowledge._mint_doc_id("fi_summary_input", {
        "summary_id": "abc", "rel_path": "papers/x.pdf",
    })
    b = Knowledge._mint_doc_id("fi_summary_input", {
        "summary_id": "abc", "rel_path": "papers/y.pdf",
    })
    assert a != b, "rel_path must disambiguate fi_summary_input docs"
    # fi_external_ref_spine carries paper_id, not quest_id —
    # previously this would have fallen through to time+uuid.
    c = Knowledge._mint_doc_id("fi_external_ref_spine", {
        "paper_id": "doi:10.1/x", "tag": "fi-paper:doi:10.1/x",
    })
    assert "doi:10.1/x" in c, "paper_id must be the stable id when present"


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


# ---------------------------------------------------------------------------
# OA full-text acquisition cascade (Phase 1) + paywall/stub gate (Phase 2)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status=200, content=b"x", text=None, json_data=None, headers=None):
        self.status_code = status
        self.content = content if isinstance(content, bytes) else content.encode()
        self.text = text if text is not None else (
            content.decode() if isinstance(content, bytes) else content
        )
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _fake_httpx_get(routes):
    """routes: list of (url-substring, _FakeResp); first match wins, else 404."""
    def _get(url, **kw):
        for sub, resp in routes:
            if sub in url:
                return resp
        return _FakeResp(status=404, content=b"")
    return _get


def test_normalize_pmcid_and_arxiv_id() -> None:
    from core.knowledge import _normalize_pmcid, _arxiv_id_from
    assert _normalize_pmcid("https://pmc.ncbi.nlm.nih.gov/articles/PMC6406785/") == "PMC6406785"
    assert _normalize_pmcid("PMC0006406785") == "PMC6406785"   # leading zeros dropped
    assert _normalize_pmcid("nothing here") == ""
    assert _arxiv_id_from("https://arxiv.org/abs/2401.01234", {}) == "2401.01234"
    assert _arxiv_id_from("https://arxiv.org/pdf/2401.01234v3", {}) == "2401.01234"
    assert _arxiv_id_from("", {"arxiv_id": "2401.01234v2"}) == "2401.01234"
    assert _arxiv_id_from("https://example.com/x", {}) == ""


def test_is_academic_source() -> None:
    from core.knowledge import _is_academic_source
    assert _is_academic_source("https://pmc.ncbi.nlm.nih.gov/articles/PMC1/")
    assert _is_academic_source("https://www.sciencedirect.com/science/article/pii/X")
    assert _is_academic_source("https://arxiv.org/abs/2401.01234")
    assert _is_academic_source("https://link.springer.com/article/x")
    assert _is_academic_source("https://journals.plos.org/plosone/article?id=x")
    assert not _is_academic_source("https://www.bbc.com/news/x")
    assert not _is_academic_source("https://example.com/")
    assert not _is_academic_source("")


def test_pmc_bioc_fulltext_assembles_and_skips_refs(monkeypatch) -> None:
    import core.knowledge as kn
    bioc = [{"documents": [{"passages": [
        {"infons": {"section_type": "TITLE"}, "text": "Greenspace and mortality"},
        {"infons": {"section_type": "ABSTRACT"},
         "text": "We studied greenness and all-cause mortality in a cohort. " * 8},
        {"infons": {"section_type": "RESULTS"},
         "text": "Hazard ratio 0.94 per IQR increase in NDVI. " * 5},
        {"infons": {"section_type": "REF"}, "text": "Smith J. 2019. Some Journal 1:2."},
    ]}]}]
    monkeypatch.setattr(kn.httpx, "get", _fake_httpx_get([("pmcoa.cgi", _FakeResp(json_data=bioc))]))
    out = kn._pmc_bioc_fulltext("PMC123", timeout_s=5, cap=100000)
    assert out and "all-cause mortality" in out and "Hazard ratio 0.94" in out
    assert "Smith J. 2019" not in out          # REF section dropped


def test_europepmc_fulltext_uses_corrected_url(monkeypatch) -> None:
    import core.knowledge as kn
    captured = {}

    def _get(url, **kw):
        captured["url"] = url
        return _FakeResp(status=200, text="<article><p>" + ("Full body text. " * 40) + "</p></article>")

    monkeypatch.setattr(kn.httpx, "get", _get)
    out = kn._europepmc_fulltext("PMC123", timeout_s=5, cap=100000)
    assert out and "Full body text" in out
    # The corrected path: PMC-prefixed id, NO /PMC/ segment (that 404s).
    assert captured["url"].endswith("/PMC123/fullTextXML")
    assert "/PMC/PMC123" not in captured["url"]


def test_resolve_ids_cross_resolves_doi_from_pmc_url(monkeypatch) -> None:
    import core.knowledge as kn
    search = {"resultList": {"result": [{"pmcid": "PMC6406785", "doi": "10.3390/x"}]}}
    monkeypatch.setattr(kn.httpx, "get", _fake_httpx_get([("europepmc", _FakeResp(json_data=search))]))
    ids = kn._resolve_ids(
        RetrievedDoc("", {"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6406785/"}),
        timeout_s=5,
    )
    assert ids["pmcid"] == "PMC6406785" and ids["doi"] == "10.3390/x"


def test_resolve_ids_arxiv_no_network(monkeypatch) -> None:
    import core.knowledge as kn
    # An arXiv URL resolves locally with no DOI/PMCID → no cross-resolve call.
    def _boom(*a, **k):
        raise AssertionError("should not hit the network for a bare arXiv id")
    monkeypatch.setattr(kn.httpx, "get", _boom)
    ids = kn._resolve_ids(
        RetrievedDoc("", {"url": "https://arxiv.org/abs/2401.01234"}), timeout_s=5,
    )
    assert ids == {"doi": "", "pmcid": "", "arxiv_id": "2401.01234"}


def test_fetch_via_open_apis_cascade_stops_at_first_hit(monkeypatch) -> None:
    import core.knowledge as kn
    monkeypatch.setattr(
        kn, "_resolve_ids",
        lambda doc, *, timeout_s: {"doi": "10.1/x", "pmcid": "PMC9", "arxiv_id": ""},
    )
    calls: list[str] = []

    def _bioc(p, *, timeout_s, cap):
        calls.append("bioc"); return None        # miss
    def _epmc(p, *, timeout_s, cap):
        calls.append("epmc"); return "E" * 400    # hit
    def _unpaywall(d, *, timeout_s, cap):
        calls.append("unpaywall"); return "U" * 400

    monkeypatch.setattr(kn, "_pmc_bioc_fulltext", _bioc)
    monkeypatch.setattr(kn, "_europepmc_fulltext", _epmc)
    monkeypatch.setattr(kn, "_unpaywall_fulltext", _unpaywall)
    out = kn._fetch_via_open_apis(RetrievedDoc("", {"url": "x"}), timeout_s=5, cap=100000)
    assert out == "E" * 400
    assert calls == ["bioc", "epmc"]   # stopped at the first hit; unpaywall untouched


def test_is_paywall_or_stub_flags_and_spares() -> None:
    import core.knowledge as kn
    pay = (
        '<html><head><script type="application/ld+json">'
        '{"isAccessibleForFree":false}</script></head><body><p>teaser</p></body></html>'
    )
    assert kn._is_paywall_or_stub(pay, "short teaser body")
    vend = (
        '<html><head><script src="https://cdn.tinypass.com/api.js"></script>'
        '</head><body><p>teaser</p></body></html>'
    )
    assert kn._is_paywall_or_stub(vend, "short teaser body")
    assert kn._is_paywall_or_stub("<html></html>", "Sign in to read the full article now")
    # A long real article body is NOT flagged even if the page carries the flag.
    assert not kn._is_paywall_or_stub(pay, "Real article content. " * 400)
    assert not kn._is_paywall_or_stub("", "anything")


def test_preprint_fulltext_arxiv_html(monkeypatch) -> None:
    import core.knowledge as kn
    body = "<html><body><p>" + ("ArXiv full text body. " * 40) + "</p></body></html>"

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            if "arxiv.org/html" in url:
                return _FakeResp(status=200, content=body.encode())
            return _FakeResp(status=404, content=b"")

    monkeypatch.setattr(kn.httpx, "Client", _Client)
    out = kn._preprint_fulltext({"arxiv_id": "2401.01234", "doi": ""}, timeout_s=5, cap=100000)
    assert out and "ArXiv full text body" in out


def test_local_papers_accepts_directory(tmp_path) -> None:
    """knowledge.local_papers may point at a FOLDER — it's scanned
    recursively for .pdf/.md/.txt, deduped against explicit file entries."""
    from core.knowledge import _expand_local_paper_paths
    (tmp_path / "a.md").write_text("paper a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("paper b", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("x", encoding="utf-8")    # unsupported
    (tmp_path / ".hidden.md").write_text("x", encoding="utf-8")    # dotfile
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("x", encoding="utf-8")  # hidden dir
    files = _expand_local_paper_paths([tmp_path, tmp_path / "a.md"])
    names = sorted(p.name for p in files)
    # recursive; csv / dotfile / hidden-dir contents / duplicate all excluded
    assert names == ["a.md", "b.txt"]
