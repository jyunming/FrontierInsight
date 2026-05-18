"""Unit tests for the literature node's pause-for-user-papers gate.

When ``knowledge.pause_for_user_papers: true``, the literature node:
- writes a ``needs/<slug>.json`` stub per abstract-only retrieved doc;
- creates ``inputs/papers/README.md`` with resume instructions;
- raises ``interrupt()`` so ``Engine.run`` can exit rc=0 cleanly.

On resume after the user dropped PDFs, the node walks
``inputs/papers/`` and appends them to the literature list as
``source=user_supplied`` entries.

These tests poke the helpers + node logic directly (no real LLM, no
real retriever) so they run in milliseconds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.engine import (
    _ingest_user_dropped_papers,
    _is_abstract_only,
    _papers_dir_has_files,
    _write_paper_need_stubs,
)
from core.knowledge import RetrievedDoc


def _log() -> logging.Logger:
    return logging.getLogger("test_literature_pause")


# ---------------------------------------------------------------------------
# _is_abstract_only — heuristic + explicit-flag interaction
# ---------------------------------------------------------------------------


def test_abstract_only_explicit_flag_wins() -> None:
    """When the retriever sets ``metadata.abstract_only``, the helper
    trusts it regardless of content length."""
    doc = RetrievedDoc(
        content="x" * 10_000,  # would otherwise be "long enough" not-abstract
        metadata={"abstract_only": True},
    )
    assert _is_abstract_only(doc) is True


def test_abstract_only_short_content_flagged() -> None:
    """Without an explicit flag, short content (< 1500 chars) is
    treated as abstract-only — covers today's arxiv/openalex/crossref
    returns, which sit in the 800-1400 char range."""
    doc = RetrievedDoc(content="x" * 800, metadata={})
    assert _is_abstract_only(doc) is True


def test_abstract_only_long_content_not_flagged() -> None:
    """Long content is full-text, no pause needed."""
    doc = RetrievedDoc(content="x" * 5000, metadata={})
    assert _is_abstract_only(doc) is False


def test_abstract_only_fetched_full_text_overrides_length() -> None:
    """An enriched doc carrying ``fetched_full_text: True`` is NOT
    abstract-only even if the post-extraction content happens to be
    short (e.g. a 1-page poster)."""
    doc = RetrievedDoc(
        content="short on purpose",
        metadata={"fetched_full_text": True},
    )
    assert _is_abstract_only(doc) is False


def test_abstract_only_local_paper_source_never_flagged() -> None:
    """User-supplied + local_paper sources are never abstract-only —
    they're the resolution, not the problem."""
    for src in ("local_paper", "user_supplied"):
        doc = RetrievedDoc(content="short", metadata={"source": src})
        assert _is_abstract_only(doc) is False


# ---------------------------------------------------------------------------
# _papers_dir_has_files — gate predicate
# ---------------------------------------------------------------------------


def test_papers_dir_has_files_false_for_missing_dir(tmp_path: Path) -> None:
    """A quest that never paused has no inputs/papers dir — the gate
    must say "no files" so the first pause check fires normally."""
    assert _papers_dir_has_files(tmp_path) is False


def test_papers_dir_has_files_ignores_readme(tmp_path: Path) -> None:
    """The README FI writes counts as zero papers — otherwise the
    pause would never re-fire after we created the dir."""
    d = tmp_path / "inputs" / "papers"
    d.mkdir(parents=True)
    (d / "README.md").write_text("# stub", encoding="utf-8")
    assert _papers_dir_has_files(tmp_path) is False


def test_papers_dir_has_files_true_after_pdf_drop(tmp_path: Path) -> None:
    """A real PDF (.pdf / .md / .txt outside README) counts."""
    d = tmp_path / "inputs" / "papers"
    d.mkdir(parents=True)
    (d / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    assert _papers_dir_has_files(tmp_path) is True


# ---------------------------------------------------------------------------
# _write_paper_need_stubs — JSON shape
# ---------------------------------------------------------------------------


def test_write_need_stubs_creates_per_paper_json(tmp_path: Path) -> None:
    """One stub per missing paper, named by slug, carrying the
    metadata FI knows so the user can resolve the citation."""
    docs = [
        RetrievedDoc(
            content="Abstract of Smith 2024",
            metadata={
                "title": "Smith 2024 quantum thingy",
                "authors": "Smith, J.",
                "doi": "10.1234/abc",
                "url": "https://example.com/smith.pdf",
                "source": "arxiv",
            },
        ),
        RetrievedDoc(
            content="Abstract of Liu 2025",
            metadata={
                "title": "Liu 2025 measurement",
                "doi": "10.5678/xyz",
                "source": "openalex",
            },
        ),
    ]
    _write_paper_need_stubs(tmp_path, docs, _log())
    needs_dir = tmp_path / "needs"
    assert needs_dir.is_dir()
    stub_paths = sorted(needs_dir.glob("*.json"))
    assert len(stub_paths) == 2
    parsed = [json.loads(p.read_text(encoding="utf-8")) for p in stub_paths]
    titles = [p["title"] for p in parsed]
    assert "Smith 2024 quantum thingy" in titles
    assert "Liu 2025 measurement" in titles
    smith = next(p for p in parsed if "Smith" in (p["title"] or ""))
    assert smith["doi"] == "10.1234/abc"
    assert smith["url"] == "https://example.com/smith.pdf"
    assert smith["source"] == "arxiv"


def test_write_need_stubs_creates_papers_readme(tmp_path: Path) -> None:
    """``inputs/papers/README.md`` is written so the user knows what
    to drop and where. Includes the quest_id so a copy-pasted resume
    command is unambiguous."""
    docs = [RetrievedDoc(content="abstract", metadata={"title": "x"})]
    quest = tmp_path / "1779131901-test-quest-abc123"
    quest.mkdir()
    _write_paper_need_stubs(quest, docs, _log())
    readme = quest / "inputs" / "papers" / "README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    assert "fi --resume" in text
    assert quest.name in text


def test_write_need_stubs_does_not_overwrite_existing(tmp_path: Path) -> None:
    """If a stub exists from a prior pause, preserve user notes —
    rewriting on every pass would silently destroy them."""
    docs = [RetrievedDoc(content="abstract", metadata={"title": "smith"})]
    needs_dir = tmp_path / "needs"
    needs_dir.mkdir()
    pre_existing = needs_dir / "smith.json"
    pre_existing.write_text('{"user_note": "downloaded already"}\n', encoding="utf-8")
    _write_paper_need_stubs(tmp_path, docs, _log())
    parsed = json.loads(pre_existing.read_text(encoding="utf-8"))
    assert parsed.get("user_note") == "downloaded already"


# ---------------------------------------------------------------------------
# _ingest_user_dropped_papers — resume-side pickup
# ---------------------------------------------------------------------------


def test_ingest_user_papers_picks_up_md_and_txt(tmp_path: Path) -> None:
    """User dropped .md / .txt files get content-extracted and appended
    to the literature list as source=user_supplied entries."""
    d = tmp_path / "inputs" / "papers"
    d.mkdir(parents=True)
    md_text = "# Paper title\n\n" + ("Real content. " * 50)  # > 200 chars
    (d / "smith2024.md").write_text(md_text, encoding="utf-8")
    txt_text = "Another paper. " * 50
    (d / "liu2025.txt").write_text(txt_text, encoding="utf-8")
    merged: list[dict] = []
    seen: set[str] = set()
    out, added = _ingest_user_dropped_papers(tmp_path, merged, seen, _log())
    assert added == 2
    assert len(out) == 2
    sources = {e["metadata"]["source"] for e in out}
    assert sources == {"user_supplied"}
    filenames = {e["metadata"]["filename"] for e in out}
    assert filenames == {"smith2024.md", "liu2025.txt"}


def test_ingest_user_papers_skips_short_and_readme(tmp_path: Path) -> None:
    """The README we wrote on pause is ignored; files under 200 chars
    are skipped as accidental empties."""
    d = tmp_path / "inputs" / "papers"
    d.mkdir(parents=True)
    (d / "README.md").write_text("# instructions", encoding="utf-8")
    (d / "stub.md").write_text("tiny", encoding="utf-8")  # < 200 chars
    merged: list[dict] = []
    seen: set[str] = set()
    _, added = _ingest_user_dropped_papers(tmp_path, merged, seen, _log())
    assert added == 0


def test_ingest_user_papers_respects_seen_set(tmp_path: Path) -> None:
    """Caller's ``seen`` dedup set is honoured — re-running ingest on
    the same dir doesn't duplicate entries."""
    d = tmp_path / "inputs" / "papers"
    d.mkdir(parents=True)
    md_text = "Hello there. " * 50
    (d / "p1.md").write_text(md_text, encoding="utf-8")
    merged: list[dict] = []
    seen: set[str] = set()
    _ingest_user_dropped_papers(tmp_path, merged, seen, _log())
    # Second call with the seen-set already populated → no new adds.
    _, added2 = _ingest_user_dropped_papers(tmp_path, merged, seen, _log())
    assert added2 == 0


def test_ingest_user_papers_no_dir_returns_zero(tmp_path: Path) -> None:
    """No inputs/papers dir → nothing to do."""
    merged: list[dict] = []
    seen: set[str] = set()
    out, added = _ingest_user_dropped_papers(tmp_path, merged, seen, _log())
    assert out is merged  # identity, not copy
    assert added == 0
