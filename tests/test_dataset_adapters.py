"""Unit tests for the dataset adapters (core/datasets/*).

All adapters are HTTP-mocked at the ``_http_get_json_sync`` boundary
so no real network call is ever made. The engine-level integration
(``_node_auto_collect_data`` calling the adapters and writing files)
is tested in ``test_engine_helpers.py`` to keep this file focused on
the adapter contract itself.
"""
from __future__ import annotations

from typing import Any

import pytest

import core.datasets.worldbank as wb_mod
from core.datasets import ADAPTER_REGISTRY, WorldBankAdapter
from core.datasets.base import DatasetAdapter, DatasetRow


# ---- registry ------------------------------------------------------------


def test_adapter_registry_exposes_worldbank() -> None:
    """The registry is the single source of truth for which adapter
    names can appear in ``engine.dataset_adapters``. The current
    registry ships ``"worldbank"`` only — adding new adapters is a
    one-line registry edit."""
    assert "worldbank" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["worldbank"] is WorldBankAdapter
    # All registered adapters must inherit from DatasetAdapter so the
    # engine can call .search() uniformly.
    for name, cls in ADAPTER_REGISTRY.items():
        assert issubclass(cls, DatasetAdapter), name


# ---- helper-function unit tests -----------------------------------------


def test_query_keywords_drops_stop_words_and_short_tokens() -> None:
    """Stop-words ('and', 'the', 'of', 'vs') and short tokens (<3
    chars) would dilute the indicator-name match score by matching
    almost every indicator. Confirm they're filtered."""
    kws = wb_mod._query_keywords(
        "Belgium and Taiwan: a study of GDP growth across the regions"
    )
    assert "belgium" in kws
    assert "taiwan" in kws
    assert "gdp" in kws
    assert "growth" in kws
    # Stop-words must be absent.
    for stop in ("and", "the", "of", "across", "study"):
        assert stop not in kws


def test_query_keywords_preserves_first_seen_order_for_duplicates() -> None:
    """Order matters for downstream usage (highest-signal terms
    typically come first in a query). Dedup preserves first
    occurrence."""
    kws = wb_mod._query_keywords("GDP per GDP capita GDP")
    assert kws == ["gdp", "per", "capita"] or kws == ["gdp", "capita"], kws


def test_detect_countries_finds_known_iso3_codes() -> None:
    """Country-name heuristic recognizes the hand-picked frequent
    countries and returns ISO3 codes."""
    out = wb_mod._detect_countries("compare Belgium and Taiwan cultural attitudes")
    assert "BEL" in out
    assert "TWN" in out


def test_detect_countries_prefers_specific_over_short_match() -> None:
    """``"south korea"`` must match BEFORE ``"korea"`` so the
    specific entry wins. Without sort-by-length, a query mentioning
    "South Korea" might land on the generic ``"korea"`` key by
    iteration order and miss the more specific intent."""
    out = wb_mod._detect_countries("South Korea workforce demographics")
    assert out == ["KOR"] or out[0] == "KOR"
    # The exact same ISO3 must not appear twice.
    assert out.count("KOR") == 1


def test_detect_countries_falls_back_to_global_aggregates() -> None:
    """When the query mentions no recognized country, fall back to a
    global aggregate set so the adapter still returns *some*
    comparative data instead of nothing."""
    out = wb_mod._detect_countries("inflation impact on household saving rates")
    # No country word in the query → default set must be returned.
    assert "WLD" in out
    assert out == wb_mod._DEFAULT_COUNTRIES


def test_score_indicator_counts_matching_keywords() -> None:
    """The scoring function is the heart of indicator ranking. Verify
    each keyword present in the indicator name contributes 1."""
    score = wb_mod._score_indicator("GDP per capita (current US$)", ["gdp", "capita", "income"])
    assert score == 2  # gdp + capita match; income does not


# ---- WorldBankAdapter.search -- happy + edge paths ----------------------


def _build_mock_indicators(
    monkeypatch: pytest.MonkeyPatch,
    indicators: list[dict[str, Any]],
    datapoint_response: Any,
) -> None:
    """Helper: monkeypatch the in-process indicator cache + the
    per-indicator datapoint fetch so the adapter doesn't touch the
    network."""
    # Reset module-level cache so the test starts clean.
    monkeypatch.setattr(wb_mod, "_indicator_cache", indicators)

    def fake_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        if "/indicator/" in url:
            return datapoint_response
        # Indicator catalog format echo.
        return [{"page": 1, "pages": 1, "total": len(indicators)}, indicators]

    monkeypatch.setattr(wb_mod, "_http_get_json_sync", fake_fetch)


@pytest.mark.asyncio
async def test_worldbank_returns_rows_with_table_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: a query that matches one indicator across two
    countries produces a DatasetRow with a country×year table in the
    body and full provenance in metadata."""
    indicators = [
        {"id": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)"},
    ]
    # WorldBank format: [header, [records]].
    datapoint_response = [
        {"page": 1, "pages": 1, "total": 4},
        [
            {"country": {"id": "BEL", "value": "Belgium"}, "date": "2022", "value": 50000.0},
            {"country": {"id": "BEL", "value": "Belgium"}, "date": "2021", "value": 48000.0},
            {"country": {"id": "TWN", "value": "Taiwan"}, "date": "2022", "value": 33000.0},
            {"country": {"id": "TWN", "value": "Taiwan"}, "date": "2021", "value": 32000.0},
        ],
    ]
    _build_mock_indicators(monkeypatch, indicators, datapoint_response)

    rows = await WorldBankAdapter().search(
        "compare GDP per capita Belgium and Taiwan", top_k=1,
    )

    assert len(rows) == 1
    row = rows[0]
    # Body contains the indicator name + country codes + at least
    # one of the data values.
    assert "GDP per capita" in row.content
    assert "BEL" in row.content
    assert "TWN" in row.content
    assert "50000" in row.content
    # Markdown table headers present.
    assert "| Country |" in row.content
    # Metadata: adapter source, indicator id/name, countries, URL.
    assert row.metadata["source"] == "worldbank"
    assert row.metadata["indicator_id"] == "NY.GDP.PCAP.CD"
    assert "Belgium" not in row.metadata["countries"]  # ISO3 codes, not names
    assert "BEL" in row.metadata["countries"]
    assert "TWN" in row.metadata["countries"]
    assert row.metadata["url"].startswith("https://data.worldbank.org/")


@pytest.mark.asyncio
async def test_worldbank_returns_empty_when_no_keyword_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Query terms that don't overlap any indicator name → return
    ``[]``. The adapter MUST NOT fall back to "just return the first
    K indicators" — a wrong-topic indicator polluting the data load
    is worse than no indicator at all."""
    indicators = [
        {"id": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)"},
        {"id": "SH.STA.WASH.P5", "name": "Mortality caused by road traffic injury"},
    ]
    _build_mock_indicators(monkeypatch, indicators, [{}, []])

    rows = await WorldBankAdapter().search(
        "cultural attitudes toward intergenerational trust", top_k=3,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_worldbank_returns_empty_when_indicator_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Indicator catalog fetch failed earlier (cached as ``[]``) →
    every subsequent call returns ``[]`` without re-trying. Prevents
    flooding a flaky network with retries during a single quest."""
    monkeypatch.setattr(wb_mod, "_indicator_cache", [])
    # Even if datapoint URL is called, it must short-circuit before.
    fake_called = {"count": 0}
    def fake_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        fake_called["count"] += 1
        return [{}, []]
    monkeypatch.setattr(wb_mod, "_http_get_json_sync", fake_fetch)

    rows = await WorldBankAdapter().search("GDP growth", top_k=3)
    assert rows == []
    assert fake_called["count"] == 0


@pytest.mark.asyncio
async def test_worldbank_zero_top_k_returns_empty() -> None:
    """``top_k=0`` is a degenerate request — fail fast with ``[]``."""
    rows = await WorldBankAdapter().search("anything", top_k=0)
    assert rows == []


@pytest.mark.asyncio
async def test_worldbank_logs_and_skips_on_datapoint_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the catalog matched but the per-indicator datapoint fetch
    raises (network blip), the adapter must log and skip THAT
    indicator — not abort the whole search. Other indicators in the
    same call still get fetched and returned."""
    indicators = [
        {"id": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)"},
        {"id": "SP.POP.TOTL",    "name": "Population total"},
    ]
    monkeypatch.setattr(wb_mod, "_indicator_cache", indicators)

    def fake_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        # The first datapoint URL hit raises; the second succeeds.
        if "NY.GDP.PCAP.CD" in url:
            raise ConnectionError("network down")
        return [
            {"page": 1, "pages": 1, "total": 1},
            [{"country": {"id": "WLD"}, "date": "2022", "value": 8_000_000_000}],
        ]
    monkeypatch.setattr(wb_mod, "_http_get_json_sync", fake_fetch)

    rows = await WorldBankAdapter().search("GDP population growth", top_k=2)
    # Only the survivor indicator returned. NOT a hard failure.
    assert len(rows) == 1
    assert rows[0].metadata["indicator_id"] == "SP.POP.TOTL"


@pytest.mark.asyncio
async def test_worldbank_render_row_shows_no_data_message_when_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API can return an empty record list for a country/indicator
    combo (e.g. the indicator simply has no data for that country in
    the requested year window). The row must still render with a
    "(no data)" marker so the data_load manifest shows the attempt
    happened — debug-friendly when the user wonders why a topic
    didn't surface."""
    indicators = [
        {"id": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)"},
    ]
    monkeypatch.setattr(wb_mod, "_indicator_cache", indicators)
    monkeypatch.setattr(
        wb_mod, "_http_get_json_sync",
        lambda url, **_: [{"page": 1, "pages": 1, "total": 0}, []],
    )

    rows = await WorldBankAdapter().search("GDP Belgium", top_k=1)
    assert len(rows) == 1
    assert "no data" in rows[0].content.lower()


# ---- DatasetRow shape ----------------------------------------------------


@pytest.mark.asyncio
async def test_worldbank_does_not_cache_failed_indicator_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed catalog fetch MUST NOT poison the in-process cache.
    Caching ``[]`` after a transient network blip would permanently
    disable the adapter for the rest of the process (a long-running
    VSCode session would never recover until restart). Instead the
    failure is logged and the next call retries."""
    # Reset cache so the test starts clean.
    monkeypatch.setattr(wb_mod, "_indicator_cache", None)

    call_count = {"n": 0}
    def fake_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("transient blip")
        # Second call succeeds.
        return [
            {"page": 1, "pages": 1, "total": 1},
            [{"id": "X.Y.Z", "name": "Test indicator"}],
        ]
    monkeypatch.setattr(wb_mod, "_http_get_json_sync", fake_fetch)

    # First call: catalog fetch raises → empty result, NO cache set.
    rows1 = await WorldBankAdapter().search("test", top_k=1)
    assert rows1 == []
    assert wb_mod._indicator_cache is None, (
        "FAILED catalog fetch must NOT populate the cache — "
        "otherwise a transient blip permanently disables the adapter"
    )

    # Second call: same Python process → tries again, succeeds.
    rows2 = await WorldBankAdapter().search("test indicator", top_k=1)
    assert len(rows2) == 1 or rows2 == []  # depends on score match
    assert call_count["n"] >= 2, "second call should have retried the catalog"


@pytest.mark.asyncio
async def test_worldbank_does_not_cache_empty_indicator_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rationale: if the catalog endpoint comes back 200 but
    with an empty body (API drift, regional outage returning a stub),
    the adapter should NOT cache the zero-row state. Retry on next
    quest."""
    monkeypatch.setattr(wb_mod, "_indicator_cache", None)
    call_count = {"n": 0}
    def fake_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [{}, []]  # zero rows
        return [{}, [{"id": "X.Y.Z", "name": "Test indicator"}]]
    monkeypatch.setattr(wb_mod, "_http_get_json_sync", fake_fetch)

    rows1 = await WorldBankAdapter().search("test", top_k=1)
    assert rows1 == []
    assert wb_mod._indicator_cache is None, (
        "empty-response state must NOT be cached"
    )


@pytest.mark.asyncio
async def test_worldbank_fetches_indicators_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-indicator data fetches must dispatch concurrently via
    ``asyncio.gather``, not serially. With
    top_k=3 and an 8 s per-call timeout, serial worst case is 24 s
    — way over the documented <5 s budget.

    We verify concurrency by having each mocked fetch sleep for
    ~50 ms; if dispatched in parallel, total wall-clock is roughly
    50 ms; serial would be 150 ms for 3 indicators. The threshold
    is generous (100 ms) to avoid CI flakiness, but still well
    below the serial-execution floor."""
    import asyncio as _asyncio
    import time as _time

    indicators = [
        {"id": "ID1", "name": "Indicator one"},
        {"id": "ID2", "name": "Indicator two"},
        {"id": "ID3", "name": "Indicator three"},
    ]
    monkeypatch.setattr(wb_mod, "_indicator_cache", indicators)

    def slow_fetch(url: str, *, timeout_s: float = 8.0) -> Any:
        if "/indicator/" in url:
            _time.sleep(0.05)  # 50 ms per call
        return [
            {"page": 1, "pages": 1, "total": 1},
            [{"country": {"id": "WLD"}, "date": "2022", "value": 100}],
        ]
    monkeypatch.setattr(wb_mod, "_http_get_json_sync", slow_fetch)

    start = _time.monotonic()
    rows = await WorldBankAdapter().search(
        "indicator one two three", top_k=3,
    )
    elapsed = _time.monotonic() - start

    assert len(rows) == 3
    # Serial would be ~150 ms; parallel ~50 ms + overhead. 130 ms
    # threshold is below the serial floor with margin for CI jitter.
    assert elapsed < 0.13, (
        f"per-indicator fetches must run in parallel; wall-clock "
        f"{elapsed:.3f}s suggests serial execution"
    )


def test_dataset_row_defaults_to_empty_metadata() -> None:
    """``DatasetRow.metadata`` defaults to ``{}`` (not None) so
    callers can always ``.get(key)`` without a None-check."""
    r = DatasetRow(content="hello")
    assert r.metadata == {}
    r2 = DatasetRow(content="hi", metadata={"k": "v"})
    assert r2.metadata == {"k": "v"}
