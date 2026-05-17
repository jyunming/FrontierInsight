"""Wikipedia adapter for free-form web-fetch data collection.

Where this fits: Axon retrieval covers corpus-shaped queries and
the structured-data adapters (WorldBank etc.) cover tabular macros.
Many no-simulation topics fall in NEITHER bucket — *"compare the
1968 student protests in Paris and Mexico City"* isn't in the
user's Axon corpus by default and isn't tabular data. Wikipedia is
the realistic free, key-less, well-covered source for that long-tail.

API surface used (both key-less, anonymous-friendly):

* ``GET /w/api.php?action=opensearch&search=<q>&limit=N&format=json``
  — title autocomplete. Returns up to N candidate article titles
  for a query string. Cheap (single round-trip), permissive
  (User-Agent only).
* ``GET /api/rest_v1/page/summary/<title>`` — per-article structured
  summary (extract, description, canonical URL, thumbnail). One
  request per matched title.

The total request budget is therefore ``1 + top_k`` HTTP calls per
quest. With ``dataset_adapter_top_k=3`` (default) that's 4 calls —
acceptable for a one-shot retrieval. Each call has a small
per-call timeout so a slow Wikipedia mirror doesn't stall the
whole quest.

Adapter NEVER raises on transient API errors — caught, logged,
return ``[]``. The engine treats that as "Wikipedia had nothing
useful" and falls through to the user-data pause.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
import urllib.request
from typing import Any

from .base import DatasetAdapter, DatasetRow
from .worldbank import _http_get_json_sync, _query_keywords

_log = logging.getLogger("frontier_insight.datasets.wikipedia")

# Anonymous Wikipedia API requires a non-empty User-Agent identifying
# the client; absent or generic UAs get rate-limited or 403'd.
_USER_AGENT = (
    "frontier-insight/0.1 "
    "(+https://github.com/jyunming/FrontierInsight) "
    "research-agent"
)
_OPENSEARCH_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=opensearch&search={q}&limit={n}&format=json&namespace=0"
)
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def _http_get_text(url: str, *, timeout_s: float = 8.0) -> str:
    """Plain blocking GET that returns the raw text body. Mirrors the
    ``_http_get_json_sync`` helper but for endpoints that return
    HTML or non-JSON (the opensearch endpoint returns JSON but in
    array form, easier to consume as text+parse). Always sends the
    Wikipedia-required User-Agent."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _build_search_query(query: str) -> str:
    """Wikipedia's opensearch matches on title prefix + redirects, so
    a verbatim research-question query (full sentence with stop-
    words and punctuation) tends to miss everything. Compress to
    the most-informative tokens via ``_query_keywords`` (the same
    stop-word filter the WorldBank adapter uses) and join with
    spaces — the opensearch backend treats that as a phrase search.

    Cap at 6 tokens so the query stays under Wikipedia's effective
    relevance window (longer queries hit the long-tail and return
    nothing)."""
    kws = _query_keywords(query)[:6]
    if not kws:
        return query.strip()
    return " ".join(kws)


class WikipediaAdapter(DatasetAdapter):
    """Wikipedia summary adapter — the free-form web-fetch slot.

    Search strategy:
      1. Compress the query to its top informative keywords.
      2. Hit ``opensearch`` to get up to ``top_k`` candidate titles.
      3. For each title, fetch ``/page/summary/<title>`` and render
         the article extract + description + URL as a DatasetRow.

    Skipped titles (logged + counted): disambiguation pages,
    missing-page responses, and rows where the extract is < 200
    characters (usually stubs not worth a paper citation).
    """

    name = "wikipedia"

    # Don't bother citing articles with extracts shorter than this —
    # they're typically stubs or one-line disambiguation entries.
    _MIN_EXTRACT_CHARS = 200

    async def search(self, query: str, *, top_k: int) -> list[DatasetRow]:
        import json
        if top_k <= 0:
            return []

        search_q = _build_search_query(query)
        if not search_q:
            return []
        opensearch_url = _OPENSEARCH_URL.format(
            q=urllib.parse.quote(search_q), n=top_k,
        )
        try:
            text = await asyncio.to_thread(
                _http_get_text, opensearch_url, timeout_s=8.0,
            )
            parsed = json.loads(text)
        except Exception as e:  # noqa: BLE001
            _log.warning("wikipedia opensearch failed: %s", e)
            return []

        # opensearch returns ``[query, [titles], [descs], [urls]]``.
        titles: list[str] = []
        if isinstance(parsed, list) and len(parsed) >= 2 and isinstance(parsed[1], list):
            titles = [str(t) for t in parsed[1] if t]
        if not titles:
            _log.info("wikipedia: opensearch returned no titles for %r", search_q)
            return []

        rows: list[DatasetRow] = []
        for title in titles:
            summary = await self._fetch_summary(title)
            if summary is None:
                continue
            row = self._render_row(title, summary)
            if row is not None:
                rows.append(row)
        return rows

    async def _fetch_summary(self, title: str) -> dict[str, Any] | None:
        """Fetch ``/page/summary/<title>``. Returns parsed JSON dict
        on success, ``None`` on any failure — caller skips."""
        url = _SUMMARY_URL.format(title=urllib.parse.quote(title.replace(" ", "_")))
        try:
            data = await asyncio.to_thread(_http_get_json_sync, url, timeout_s=8.0)
        except Exception as e:  # noqa: BLE001
            _log.warning("wikipedia summary fetch failed for %r: %s", title, e)
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _render_row(self, title: str, summary: dict[str, Any]) -> DatasetRow | None:
        """Render one Wikipedia summary as a DatasetRow.

        Returns ``None`` (caller skips) when:
          - ``type`` is ``"disambiguation"`` (no concrete content).
          - ``extract`` is shorter than ``_MIN_EXTRACT_CHARS``.
          - ``title`` / ``extract`` are missing.

        Body format::

            ## <article title>

            <description (one line)>

            <extract — the article's intro paragraph>

            Source: Wikipedia (<canonical URL>). Retrieved <date>.
        """
        page_type = str(summary.get("type", ""))
        if page_type == "disambiguation":
            _log.info(
                "wikipedia: skipping disambiguation page %r", title,
            )
            return None
        extract = str(summary.get("extract", "")).strip()
        if not extract or len(extract) < self._MIN_EXTRACT_CHARS:
            _log.info(
                "wikipedia: skipping %r — extract too short "
                "(%d chars)", title, len(extract),
            )
            return None
        description = str(summary.get("description", "")).strip()
        # canonicalurl > content_urls.desktop.page > fallback synth
        content_urls = summary.get("content_urls") or {}
        desktop = content_urls.get("desktop") if isinstance(content_urls, dict) else None
        url = (
            str(summary.get("canonicalurl"))
            if summary.get("canonicalurl")
            else (str(desktop.get("page")) if isinstance(desktop, dict) and desktop.get("page") else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}")
        )
        article_title = str(summary.get("title") or title)

        body_lines = [
            f"## {article_title}",
            "",
        ]
        if description:
            body_lines.append(description)
            body_lines.append("")
        body_lines.append(extract)
        body_lines.append("")
        body_lines.append(f"Source: Wikipedia ({url}).")
        body = "\n".join(body_lines)

        return DatasetRow(
            content=body,
            metadata={
                "source": "wikipedia",
                "title": article_title,
                "url": url,
                "wikipedia_type": page_type or "standard",
            },
        )
