"""World Bank Open Data adapter for the auto-collect path.

The World Bank publishes a free, key-less JSON API at
``https://api.worldbank.org/v2/`` covering ~1500 indicators across
~270 countries (GDP, inflation, schooling years, life expectancy,
Gini, ...). It's the cleanest free macro / dev data source for
qualitative comparative studies, which is exactly what
no-simulation mode targets (cultural / social / historical topics
that need real numbers).

How this adapter maps a free-form query to API calls:

1. Use the indicator-search endpoint
   (``/v2/indicator?source=2&format=json&per_page=10000``) to
   find indicator IDs whose name matches words from the query.
   Cached in-process to avoid re-downloading the ~3 MB JSON on
   every quest.
2. For each candidate indicator, score by how many query keywords
   the indicator name contains. Take top_k.
3. For each picked indicator, fetch the latest data point for a
   sensible set of countries (when the query mentions specific
   countries) or for a default short list (World, OECD, EU,
   high-income / low-income aggregates) when no country is named.
4. Render one ``DatasetRow`` per indicator carrying the values as
   a small Markdown table.

Why heuristic keyword matching and not semantic search: a quest's
auto-collect path runs once per node invocation, must be fast
(<5 s budget), and the World Bank's indicator names are short and
self-describing. Embedding-based ranking would add an embedding
model dependency for marginal gain.

The adapter NEVER raises on transient API errors — it logs and
returns ``[]``. The engine treats that as "this adapter had
nothing" and moves on.
"""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
import urllib.request
from typing import Any

from .base import DatasetAdapter, DatasetRow

_log = logging.getLogger("frontier_insight.datasets.worldbank")

_INDICATOR_LIST_URL = (
    "https://api.worldbank.org/v2/indicator"
    "?source=2&format=json&per_page=10000"
)
_DATAPOINT_URL = (
    "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    "?format=json&per_page=5&date={start}:{end}"
)

# Short list of country codes the adapter falls back on when the
# query doesn't name a specific country. Picks high-signal global
# aggregates so the LLM gets *some* comparative data even on a
# generic query.
_DEFAULT_COUNTRIES = ["WLD", "OED", "EUU", "HIC", "LIC"]

# Country-name → ISO3 lookup for the most-searched-for countries.
# Intentionally hand-picked instead of pulling a 200-entry table:
# the adapter is a sketch, not a country-normalization library, and
# a country mentioned by an unusual name (e.g. "Czech Republic" vs
# "Czechia") gracefully falls through to the default set rather
# than producing wrong results.
_COUNTRY_HINTS: dict[str, str] = {
    "belgium": "BEL", "taiwan": "TWN", "china": "CHN", "japan": "JPN",
    "korea": "KOR", "south korea": "KOR", "germany": "DEU", "france": "FRA",
    "italy": "ITA", "spain": "ESP", "portugal": "PRT", "netherlands": "NLD",
    "united kingdom": "GBR", "uk": "GBR", "england": "GBR",
    "united states": "USA", "usa": "USA", "us": "USA", "america": "USA",
    "canada": "CAN", "mexico": "MEX", "brazil": "BRA", "argentina": "ARG",
    "australia": "AUS", "new zealand": "NZL", "india": "IND", "indonesia": "IDN",
    "thailand": "THA", "vietnam": "VNM", "singapore": "SGP", "philippines": "PHL",
    "russia": "RUS", "poland": "POL", "ukraine": "UKR", "turkey": "TUR",
    "egypt": "EGY", "saudi arabia": "SAU", "israel": "ISR", "uae": "ARE",
    "south africa": "ZAF", "nigeria": "NGA", "kenya": "KEN", "ethiopia": "ETH",
    "morocco": "MAR", "iran": "IRN", "iraq": "IRQ",
    "sweden": "SWE", "norway": "NOR", "denmark": "DNK", "finland": "FIN",
    "switzerland": "CHE", "austria": "AUT", "ireland": "IRL",
    "greece": "GRC", "czechia": "CZE", "hungary": "HUN", "romania": "ROU",
}


_indicator_cache: list[dict[str, Any]] | None = None
_indicator_cache_lock = asyncio.Lock()


def _http_get_json_sync(url: str, *, timeout_s: float = 8.0) -> Any:
    """Plain blocking urllib fetch — adapter implementations call
    this via ``asyncio.to_thread`` to keep the event loop free.

    Returns the parsed JSON or raises ``urllib.error.URLError`` /
    ``json.JSONDecodeError`` on failure. The caller (always one of
    this file's async helpers) is responsible for catching.
    """
    import json
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "frontier-insight/0.1 (+https://github.com/jyunming/FrontierInsight)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


async def _fetch_indicators() -> list[dict[str, Any]]:
    """Fetch the full indicator catalog (~3 MB JSON, ~1500 rows).

    Cached in-process on SUCCESS only; the lock prevents a
    thundering-herd of concurrent fetches on first call when
    multiple quests fire at once.

    Critical: failures are NOT cached. A single transient DNS /
    network blip at the start of a long-running Python process
    (e.g. a VSCode extension session) would otherwise permanently
    disable the adapter for that process. Returning ``[]`` on
    failure without setting the cache means the next quest tries
    again — which is the right policy for an opt-in adapter where
    transient unavailability shouldn't be sticky.
    """
    global _indicator_cache
    if _indicator_cache is not None:
        return _indicator_cache
    async with _indicator_cache_lock:
        if _indicator_cache is not None:
            return _indicator_cache
        try:
            data = await asyncio.to_thread(_http_get_json_sync, _INDICATOR_LIST_URL)
        except Exception as e:  # noqa: BLE001 — defensive
            _log.warning(
                "worldbank indicator fetch failed: %s — NOT cached, "
                "next call will retry", e,
            )
            return []
        # WorldBank's response format is ``[header, [rows...]]``.
        rows: list[dict[str, Any]] = []
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            rows = [r for r in data[1] if isinstance(r, dict)]
        if not rows:
            # Empty response is suspicious (API drift?) — log and
            # also DO NOT cache, so the next quest re-fetches.
            _log.warning(
                "worldbank indicator response had zero rows — NOT "
                "cached; next call will retry",
            )
            return []
        _indicator_cache = rows
        _log.info("worldbank: cached %d indicators", len(rows))
        return rows


def _score_indicator(indicator_name: str, keywords: list[str]) -> int:
    """Number of keywords that appear in the indicator name (case-
    insensitive substring). Bigger is better; ties are broken by
    indicator catalog order (which roughly correlates with prominence)."""
    name_l = indicator_name.lower()
    return sum(1 for kw in keywords if kw in name_l)


# Stop-words trimmed from a query before keyword-matching. Cheap —
# we don't need a NLP library, just a denylist of words that would
# match every indicator (e.g. "and", "the", "of") and dilute the
# top_k ranking.
_STOP_WORDS = frozenset({
    "and", "or", "the", "of", "in", "on", "for", "to", "a", "an",
    "is", "are", "was", "were", "be", "been", "vs", "versus",
    "between", "across", "by", "with", "without", "this", "that",
    "these", "those", "study", "studies", "research", "analysis",
})


def _query_keywords(query: str) -> list[str]:
    """Split a free-form query into lowercase 3+ char keywords with
    stop-words removed. Order preserved so the highest-signal terms
    (typically the topic head) come first."""
    raw = re.findall(r"[A-Za-z]{3,}", query)
    seen: set[str] = set()
    out: list[str] = []
    for token in raw:
        low = token.lower()
        if low in _STOP_WORDS or low in seen:
            continue
        seen.add(low)
        out.append(low)
    return out


def _detect_countries(query: str) -> list[str]:
    """Scan the query for known country names → return ISO3 codes.
    Falls back to ``_DEFAULT_COUNTRIES`` when none matched, so the
    adapter always returns *some* comparative data.

    Multi-word names (``"south korea"``, ``"new zealand"``) are
    matched before single-word names so ``"korea"`` doesn't shadow
    ``"south korea"``.

    Matching uses word boundaries (``\\b``) — without this, short
    hints like ``"us"`` would match inside ``"household"`` /
    ``"because"`` / ``"thus"`` and falsely add ``USA`` to every
    query containing those words. Tested in
    ``tests/test_dataset_adapters.py::
    test_detect_countries_falls_back_to_global_aggregates``.
    """
    q = query.lower()
    found: list[str] = []
    # Sort hint keys by length DESC so longer (more specific) matches
    # land first.
    for hint in sorted(_COUNTRY_HINTS, key=len, reverse=True):
        pattern = r"\b" + re.escape(hint) + r"\b"
        if re.search(pattern, q) and _COUNTRY_HINTS[hint] not in found:
            found.append(_COUNTRY_HINTS[hint])
    return found or list(_DEFAULT_COUNTRIES)


class WorldBankAdapter(DatasetAdapter):
    """Adapter for the World Bank Open Data API.

    Search strategy:
      1. Get/cache the full indicator catalog (~3 MB on first call).
      2. Tokenize the query and keep keywords with >=3 chars, no
         stop-words.
      3. Score each indicator by how many keywords appear in its
         ``name``; keep the top ``top_k`` scored indicators (ties
         broken by catalog order).
      4. Detect country mentions in the query (``"Belgium and
         Taiwan"`` → ``["BEL", "TWN"]``); fall back to global
         aggregates when none found.
      5. Fetch the latest 5 years of data per indicator × country
         and render one ``DatasetRow`` per indicator with a small
         Markdown table.

    Net result: a no-simulation quest like *"compare Belgium and
    Taiwan public-trust dynamics"* gets 5 World Bank rows like
    "government effectiveness" + "GDP per capita" + "tertiary
    enrollment" + "% of public sector employment", each with the
    actual 2018-2023 values for BEL and TWN. The LLM then has a
    real factual baseline to write the comparative analysis from
    instead of inventing numbers.

    The adapter NEVER raises — every failure path logs and falls
    through to ``return []`` so the engine treats the adapter as
    "had nothing useful" and moves on.
    """

    name = "worldbank"

    # 5-year window. The LLM doesn't need 30 years of history for
    # a comparative paper; recent values + maybe one historical
    # reference is enough, and a wider window blows out the
    # data_load prompt budget.
    _YEAR_WINDOW = 5

    async def search(self, query: str, *, top_k: int) -> list[DatasetRow]:
        from datetime import date

        if top_k <= 0:
            return []
        try:
            indicators = await _fetch_indicators()
        except Exception as e:  # noqa: BLE001
            _log.warning("worldbank: indicator catalog unavailable: %s", e)
            return []
        if not indicators:
            return []

        keywords = _query_keywords(query)
        if not keywords:
            _log.info("worldbank: query had no usable keywords; skipping")
            return []
        countries = _detect_countries(query)

        # Score + pick top_k.
        scored: list[tuple[int, dict[str, Any]]] = []
        for ind in indicators:
            name = str(ind.get("name", ""))
            if not name:
                continue
            score = _score_indicator(name, keywords)
            if score > 0:
                scored.append((score, ind))
        # Sort by score DESC, preserving original catalog order on ties.
        scored.sort(key=lambda t: -t[0])
        top = scored[:top_k]
        if not top:
            _log.info(
                "worldbank: no indicator matched keywords=%s; "
                "returning empty", keywords,
            )
            return []

        current_year = date.today().year
        start_year = current_year - self._YEAR_WINDOW
        country_path = ";".join(countries)

        # Parallel per-indicator fetches. Each call is an independent
        # HTTP round-trip; serializing them would push the worst-case
        # latency to ``top_k × per-call-timeout`` (e.g. 3 × 8 s = 24 s
        # for top_k=3) and blow the "<5 s budget" documented in the
        # module header. ``asyncio.gather(return_exceptions=True)``
        # lets a single failing indicator NOT take down the whole
        # search — its slot just produces an exception we log+skip.
        async def fetch_one(score_ind: tuple[int, dict[str, Any]]):
            score, ind = score_ind
            ind_id = str(ind.get("id", ""))
            if not ind_id:
                return None
            url = _DATAPOINT_URL.format(
                country=urllib.parse.quote(country_path, safe=";"),
                indicator=urllib.parse.quote(ind_id),
                start=start_year, end=current_year,
            )
            data = await asyncio.to_thread(_http_get_json_sync, url)
            return (score, ind, data)

        results = await asyncio.gather(
            *(fetch_one(t) for t in top), return_exceptions=True,
        )

        rows: list[DatasetRow] = []
        for r, (orig_score, orig_ind) in zip(results, top):
            ind_id = str(orig_ind.get("id", ""))
            ind_name = str(orig_ind.get("name", ""))
            if isinstance(r, Exception):
                _log.warning(
                    "worldbank: fetch failed for %s × %s: %s",
                    ind_id, country_path, r,
                )
                continue
            if r is None:
                continue
            score, ind, data = r
            rows.append(self._render_row(score, ind_id, ind_name, countries, data))
        return rows

    def _render_row(
        self,
        score: int,
        ind_id: str,
        ind_name: str,
        countries: list[str],
        api_data: Any,
    ) -> DatasetRow:
        """Render one indicator's API response as a DatasetRow.

        The Markdown body looks like::

            ## GDP per capita (current US$) (NY.GDP.PCAP.CD)

            | Country | 2019 | 2020 | 2021 | 2022 | 2023 |
            |---------|------|------|------|------|------|
            | BEL     | …    | …    | …    | …    | …    |
            | TWN     | …    | …    | …    | …    | …    |

            Source: World Bank Open Data.

        Missing values render as ``-``. If the API returned no rows
        we still emit a row with a "(no data)" line so the manifest
        records the attempt — useful for debugging "why didn't this
        indicator show up in the paper".
        """
        # Parse the WorldBank ``[header, [rows]]`` shape defensively.
        records: list[dict[str, Any]] = []
        if isinstance(api_data, list) and len(api_data) >= 2 and isinstance(api_data[1], list):
            records = [r for r in api_data[1] if isinstance(r, dict)]

        # Pivot: build {country_iso: {year: value}}.
        pivot: dict[str, dict[int, Any]] = {c: {} for c in countries}
        years_seen: set[int] = set()
        for rec in records:
            country_obj = rec.get("country", {})
            iso = ""
            if isinstance(country_obj, dict):
                iso = str(country_obj.get("id", ""))
            date_s = str(rec.get("date", ""))
            try:
                year = int(date_s)
            except ValueError:
                continue
            value = rec.get("value")
            if iso in pivot:
                pivot[iso][year] = value
                years_seen.add(year)
        years = sorted(years_seen)

        # Render table.
        lines = [f"## {ind_name} ({ind_id})", ""]
        if not years:
            lines.append("(no data returned by API for the queried country set)")
        else:
            header = "| Country | " + " | ".join(str(y) for y in years) + " |"
            sep = "|---------|" + "------|" * len(years)
            lines.append(header)
            lines.append(sep)
            for country in countries:
                cells = []
                for y in years:
                    v = pivot[country].get(y)
                    cells.append("-" if v is None else f"{v}")
                lines.append(f"| {country} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Source: World Bank Open Data.")
        content = "\n".join(lines)
        url = f"https://data.worldbank.org/indicator/{ind_id}"
        return DatasetRow(
            content=content,
            metadata={
                "source": "worldbank",
                "indicator_id": ind_id,
                "indicator_name": ind_name,
                "countries": ",".join(countries),
                "url": url,
                "title": ind_name,
                "score": score,
            },
        )
