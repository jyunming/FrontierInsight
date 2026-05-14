"""Unit tests for the Phase D3 Wikipedia adapter.

All HTTP calls are mocked at the ``_http_get_text`` / ``_http_get_json_sync``
boundary so the test suite runs with no network access.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

import core.datasets.wikipedia as wiki_mod
from core.datasets import ADAPTER_REGISTRY, WikipediaAdapter


# ---- registry / contract -------------------------------------------------


def test_wikipedia_adapter_registered() -> None:
    """``WikipediaAdapter`` is registered under the name
    ``"wikipedia"`` so YAML configs can opt in by name."""
    assert "wikipedia" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["wikipedia"] is WikipediaAdapter
    assert WikipediaAdapter.name == "wikipedia"


# ---- query compression ---------------------------------------------------


def test_build_search_query_compresses_to_top_keywords() -> None:
    """A full-sentence research question is too long for Wikipedia's
    opensearch — compress to top informative keywords. The function
    keeps the FIRST top_k informative keywords (preserving the
    LLM's intent of which terms come first), drops stop-words, and
    caps at 6 tokens."""
    out = wiki_mod._build_search_query(
        "Belgium Taiwan cultural comparison: collectivism and trust dynamics",
    )
    # Cap at 6 tokens.
    assert len(out.split()) <= 6
    # Stop-words excluded.
    for stop in ("and", "the", "in", "across", "of"):
        assert f" {stop} " not in f" {out} "
    # First-listed informative tokens survive.
    assert "belgium" in out.lower()
    assert "taiwan" in out.lower()


def test_build_search_query_falls_back_to_raw_on_no_keywords() -> None:
    """Edge case: a query that's all stop-words / short tokens →
    fall back to the raw query rather than emitting empty (which
    would make opensearch return nothing)."""
    out = wiki_mod._build_search_query("a the of by an in")
    # Falls back to the raw string (trimmed).
    assert out == "a the of by an in"


# ---- WikipediaAdapter.search --------------------------------------------


def _mock_wikipedia_responses(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opensearch_titles: list[str],
    summaries: dict[str, dict[str, Any]],
    opensearch_raises: Exception | None = None,
    summary_raises: dict[str, Exception] | None = None,
) -> None:
    """Patch both HTTP helpers in one call. ``opensearch_titles``
    becomes the second element of the opensearch response.
    ``summaries`` is a title→summary dict."""
    summary_raises = summary_raises or {}

    def fake_get_text(url: str, *, timeout_s: float = 8.0) -> str:
        if "opensearch" in url:
            if opensearch_raises is not None:
                raise opensearch_raises
            payload = ["", opensearch_titles, [], []]
            return json.dumps(payload)
        raise AssertionError(f"unexpected text URL: {url}")

    def fake_get_json(url: str, *, timeout_s: float = 8.0) -> Any:
        # /page/summary/<title>
        if "/page/summary/" not in url:
            raise AssertionError(f"unexpected JSON URL: {url}")
        # Reverse-decode the URL-encoded title.
        import urllib.parse as up
        title_underscored = url.rsplit("/", 1)[-1]
        title_url_decoded = up.unquote(title_underscored)
        title = title_url_decoded.replace("_", " ")
        if title in summary_raises:
            raise summary_raises[title]
        return summaries.get(title, {})

    monkeypatch.setattr(wiki_mod, "_http_get_text", fake_get_text)
    monkeypatch.setattr(wiki_mod, "_http_get_json_sync", fake_get_json)


@pytest.mark.asyncio
async def test_wikipedia_returns_dataset_rows_on_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """opensearch returns 2 titles → both summaries fetched → 2
    DatasetRows returned with extract/description/canonical URL."""
    summaries = {
        "Belgian culture": {
            "type": "standard",
            "title": "Belgian culture",
            "description": "Cultural traditions of Belgium",
            "extract": (
                "Belgian culture is shaped by Flemish, Walloon, and German "
                "communities, with deeply rooted traditions in cuisine, "
                "fine arts, and a strong emphasis on regional autonomy. "
                "Catholic heritage and the legacy of postwar federalism "
                "continue to influence civic life and trust dynamics."
            ),
            "canonicalurl": "https://en.wikipedia.org/wiki/Belgian_culture",
        },
        "Taiwanese culture": {
            "type": "standard",
            "title": "Taiwanese culture",
            "description": "Cultural traditions of Taiwan",
            "extract": (
                "Taiwanese culture is a hybrid of Han Chinese heritage, "
                "indigenous Austronesian traditions, and modern influences "
                "from Japanese occupation and democratic reform. Religion, "
                "Confucian ethics, and rapid modernization shape "
                "intergenerational dynamics."
            ),
            "canonicalurl": "https://en.wikipedia.org/wiki/Taiwanese_culture",
        },
    }
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=list(summaries.keys()),
        summaries=summaries,
    )

    rows = await WikipediaAdapter().search(
        "Compare Belgian culture and Taiwanese culture trust dynamics",
        top_k=2,
    )

    assert len(rows) == 2
    titles_in_rows = {r.metadata["title"] for r in rows}
    assert titles_in_rows == {"Belgian culture", "Taiwanese culture"}
    for r in rows:
        assert r.metadata["source"] == "wikipedia"
        assert r.metadata["url"].startswith("https://en.wikipedia.org/wiki/")
        # Body has the heading + a substantive extract + Source line.
        assert r.content.startswith("## ")
        assert "Source: Wikipedia" in r.content
        # Extract IS in the body.
        assert r.metadata["title"] in r.content


@pytest.mark.asyncio
async def test_wikipedia_skips_disambiguation_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A summary with ``type: "disambiguation"`` is useless to the
    paper — drop it, log INFO."""
    summaries = {
        "Belgium": {
            "type": "disambiguation",
            "title": "Belgium",
            "extract": "Belgium may refer to: country, beer, etc...",
            "canonicalurl": "https://en.wikipedia.org/wiki/Belgium_(disambiguation)",
        },
        "Belgium (country)": {
            "type": "standard",
            "title": "Belgium (country)",
            "description": "Country in Western Europe",
            "extract": (
                "Belgium is a country in Western Europe with a population "
                "of about 12 million. It has three official languages "
                "(Dutch, French, German), three regions, and a federal "
                "constitutional monarchy. It hosts the EU institutions "
                "in Brussels and has a strong tradition of consociational "
                "democracy."
            ),
            "canonicalurl": "https://en.wikipedia.org/wiki/Belgium_(country)",
        },
    }
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=list(summaries.keys()),
        summaries=summaries,
    )

    rows = await WikipediaAdapter().search("Belgium overview", top_k=2)
    # Disambiguation dropped → only one row.
    assert len(rows) == 1
    assert rows[0].metadata["title"] == "Belgium (country)"


@pytest.mark.asyncio
async def test_wikipedia_skips_short_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stubs (extract < 200 chars) are not paper-citation-worthy.
    Drop them, log INFO."""
    summaries = {
        "Stubby": {
            "type": "standard",
            "title": "Stubby",
            "extract": "A short stub.",  # < 200 chars
            "canonicalurl": "https://en.wikipedia.org/wiki/Stubby",
        },
        "Real article": {
            "type": "standard",
            "title": "Real article",
            "description": "A proper article",
            "extract": "X" * 250,  # >= 200 chars
            "canonicalurl": "https://en.wikipedia.org/wiki/Real_article",
        },
    }
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=list(summaries.keys()),
        summaries=summaries,
    )

    rows = await WikipediaAdapter().search("anything", top_k=2)
    assert len(rows) == 1
    assert rows[0].metadata["title"] == "Real article"


@pytest.mark.asyncio
async def test_wikipedia_returns_empty_when_opensearch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """opensearch network failure → return ``[]``, log WARNING.
    Don't crash the auto-collect flow."""
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=[],
        summaries={},
        opensearch_raises=ConnectionError("network down"),
    )

    rows = await WikipediaAdapter().search("anything", top_k=3)
    assert rows == []


@pytest.mark.asyncio
async def test_wikipedia_individual_summary_failure_skips_that_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If summary fetch fails for ONE title but succeeds for others,
    the adapter must keep the survivors — not abort the whole
    search."""
    summaries = {
        "Good article": {
            "type": "standard",
            "title": "Good article",
            "description": "A working summary",
            "extract": "X" * 250,
            "canonicalurl": "https://en.wikipedia.org/wiki/Good_article",
        },
    }
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=["Bad article", "Good article"],
        summaries=summaries,
        summary_raises={"Bad article": TimeoutError("slow mirror")},
    )

    rows = await WikipediaAdapter().search("anything", top_k=2)
    assert len(rows) == 1
    assert rows[0].metadata["title"] == "Good article"


@pytest.mark.asyncio
async def test_wikipedia_zero_top_k_returns_empty() -> None:
    """``top_k=0`` short-circuits — no network call, no rows."""
    rows = await WikipediaAdapter().search("anything", top_k=0)
    assert rows == []


@pytest.mark.asyncio
async def test_wikipedia_no_titles_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """opensearch returned successfully but with zero candidate
    titles — no summary fetches, return ``[]``."""
    _mock_wikipedia_responses(
        monkeypatch, opensearch_titles=[], summaries={},
    )
    rows = await WikipediaAdapter().search("zxqwt nonsense", top_k=3)
    assert rows == []


@pytest.mark.asyncio
async def test_wikipedia_falls_back_to_synthesized_url_when_canonical_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a summary lacks ``canonicalurl`` AND ``content_urls.desktop.page``,
    fall back to a synthesized URL from the title so the row still
    has provenance (no broken-link state)."""
    summaries = {
        "Orphan article": {
            "type": "standard",
            "title": "Orphan article",
            "extract": "X" * 250,
            # No canonicalurl, no content_urls.
        },
    }
    _mock_wikipedia_responses(
        monkeypatch,
        opensearch_titles=["Orphan article"],
        summaries=summaries,
    )

    rows = await WikipediaAdapter().search("anything", top_k=1)
    assert len(rows) == 1
    assert rows[0].metadata["url"].startswith("https://en.wikipedia.org/wiki/")
    assert "Orphan_article" in rows[0].metadata["url"]
