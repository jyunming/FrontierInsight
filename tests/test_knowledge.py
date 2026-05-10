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

import pytest

from core.config import KnowledgeConfig
from core.knowledge import Knowledge, RetrievedDoc


# --- disabled-path contract --------------------------------------------------


def test_search_returns_empty_when_disabled() -> None:
    k = Knowledge(KnowledgeConfig(enabled=False))
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
    """Mock brain exposing only `add_text` (the most idiomatic path)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def add_text(self, text: str, *, metadata: dict) -> None:
        self.calls.append((text, metadata))


class _BrainIngestText:
    """Mock brain exposing only `ingest_text` (the secondary path)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def ingest_text(self, text: str, *, metadata: dict) -> None:
        self.calls.append((text, metadata))


class _BrainIngestOnly:
    """Mock brain exposing only `ingest` (the file-path fallback)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def ingest(self, source: str) -> None:
        self.calls.append(source)


def _enabled_knowledge_with(brain: object, *, write_back: bool = True) -> Knowledge:
    """Build a Knowledge with enabled=True bypassing the axon import path."""
    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled = True
    k.cfg = KnowledgeConfig(enabled=True, write_back_quests=write_back)
    k._brain = brain
    return k


def test_add_quest_artifacts_uses_add_text_when_available(tmp_path: Path) -> None:
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("# paper body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="q42", paper_md_path=paper, summary="key finding")

    assert ok is True
    assert len(brain.calls) == 2  # paper body + summary
    body_text, body_meta = brain.calls[0]
    assert body_text == "# paper body"
    assert body_meta == {"tag": "fi-quest:q42", "kind": "fi_quest"}
    summary_text, summary_meta = brain.calls[1]
    assert summary_text == "key finding"
    assert summary_meta == {"tag": "fi-quest:q42", "kind": "fi_quest_summary"}


def test_add_quest_artifacts_skips_summary_when_empty(tmp_path: Path) -> None:
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="")

    assert ok is True
    assert len(brain.calls) == 1  # no summary call


def test_add_quest_artifacts_uses_ingest_text_when_add_text_missing(tmp_path: Path) -> None:
    """Confirms the `add_text` -> `ingest_text` fallback step."""
    brain = _BrainIngestText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("# body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="qX", paper_md_path=paper, summary="s")

    assert ok is True
    assert len(brain.calls) == 2
    assert brain.calls[0][0] == "# body"
    assert brain.calls[0][1]["tag"] == "fi-quest:qX"


def test_add_quest_artifacts_falls_through_to_ingest(tmp_path: Path) -> None:
    """When neither add_text nor ingest_text exist, falls back to ingest(path)."""
    brain = _BrainIngestOnly()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "paper.md"
    paper.write_text("# body", encoding="utf-8")

    ok = k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="s")

    assert ok is True
    assert brain.calls == [str(paper)]


def test_add_quest_artifacts_handles_missing_paper_with_add_text(tmp_path: Path) -> None:
    """If the paper file is missing, add_text receives an empty body."""
    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    paper = tmp_path / "missing.md"  # not created

    ok = k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="x")

    assert ok is True
    assert brain.calls[0][0] == ""


# --- ingest fallback ---------------------------------------------------------


def test_ingest_uses_ingest_method_first(tmp_path: Path) -> None:
    brain = _BrainIngestOnly()
    k = _enabled_knowledge_with(brain)
    f = tmp_path / "doc.txt"
    f.write_text("x", encoding="utf-8")
    assert k.ingest(f) is True
    assert brain.calls == [str(f)]


class _BrainAddDocument:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def add_document(self, source: str) -> None:
        self.calls.append(source)


def test_ingest_falls_back_to_add_document(tmp_path: Path) -> None:
    brain = _BrainAddDocument()
    k = _enabled_knowledge_with(brain)
    f = tmp_path / "doc.txt"
    f.write_text("x", encoding="utf-8")
    assert k.ingest(f) is True
    assert brain.calls == [str(f)]


def test_ingest_returns_false_when_brain_has_no_ingest_api(tmp_path: Path) -> None:
    brain = object()  # neither ingest nor add_document
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
