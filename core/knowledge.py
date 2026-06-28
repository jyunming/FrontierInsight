"""Research knowledge layer — Axon primary, multi-source external router.

Two responsibilities:

1. **Retrieval** (`Knowledge.asearch`) for the engine's `ideate` and
   `literature` nodes. Three layers, in order:
     a. Pinned `local_papers` — files the user dropped into the config;
        always at the head of the result.
     b. Axon — the FI long-term, curated store.
     c. External router — fires when Axon returns empty. With
        `source_routing="auto"` the LLM picks 1–5 sources from a
        12-entry catalog (extensible via Axon-ingested `fi_source_catalog`
        entries); otherwise uses the YAML `external_fallback` list
        verbatim. Adapters run in parallel; results are merged and
        de-duplicated by DOI / arXiv-id / PMID / normalized title.
   When `try_fetch_full_text` is true, external hits are augmented
   with publisher-PDF text where the host network has access (login
   walls are rejected by a Content-Type + `%PDF-` magic-bytes check).

2. **Ingest** (`Knowledge.add_quest_artifacts`) for the post-quest
   write-back, gated on `verdict == "accept"` by default. Finished
   research lands as a *structured bundle*, NOT a flat blob, so the
   chunked-RAG corpus stays title-searchable and topic-linkable:
     - `fi_paper_spine`        — single-chunk card-catalog entry per
                                 paper (title + authors + DOI + topic
                                 + abstract + key claims).
     - `fi_quest_paper`        — full paper body with a 1-line
                                 `[Title · Year · Venue · DOI]` citation
                                 header prepended to every chunk.
     - `fi_quest_summary`      — analysis summary + structured JSON
                                 (hypothesis / findings / result_json
                                 / verdict / score / provider / model).
                                 Carries `paper_refs` (short-ids) in
                                 metadata.
     - `fi_topic_event`        — per-accept pointer keyed by topic slug
                                 listing this quest + the papers it
                                 cited. Enables topic-scoped rollups.
     - `fi_external_ref_spine` — curated card-catalog entries for the
                                 external papers an accepted quest
                                 consumed (only refs from accepted
                                 quests persist).

External sources currently implemented (all free, all best-effort —
network failures log and return [] rather than raising):
  - openalex          — broadest single open index, ~200M works
  - arxiv             — physics / CS / math / quant-ph / q-bio / stats
  - crossref          — DOI metadata across paywalled publishers
                        (Springer, Elsevier, IEEE, ACM, SPIE, ACS, …)
  - semantic_scholar  — broad coverage with citation graph
  - pubmed            — biomedical (NCBI E-utilities)
  - core              — 240M open-access papers (requires CORE_API_KEY)
  - google_scholar    — EXPERIMENTAL via `scholarly`; no official API,
                        rate-limited / sometimes blocked by Google

Each adapter implements `search(query, top_k) -> list[RetrievedDoc]`.
The router parallelizes calls via `asyncio.to_thread` + `asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import html as _htmlmod
import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from .config import KnowledgeConfig

_log = logging.getLogger("frontier_insight.knowledge")

# trafilatura logs an ERROR ("empty HTML tree", "wrong data type") whenever
# it's handed an empty / non-HTML stub — which is routine here (we fall back
# to the built-in extractor on any miss). Silence its non-actionable noise so
# the quest log stays readable; real failures are handled by the fallback.
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

try:
    from axon import AxonBrain, AxonConfig  # type: ignore[import-not-found]
    from axon.integrations.langchain import AxonRetriever  # type: ignore[import-not-found]

    _AXON_AVAILABLE = True
except Exception as e:
    _AXON_AVAILABLE = False
    _AXON_IMPORT_ERROR: Exception = e
    AxonBrain = None  # type: ignore[assignment]
    AxonConfig = None  # type: ignore[assignment]
    AxonRetriever = None  # type: ignore[assignment]


# Serializes the read-modify-write of the process-global HF env vars in
# ``_apply_offline_env``. In ``--fleet`` mode several engines build their
# Knowledge layer around the same time; without the lock two quests could
# interleave their env mutations, and a quest with a divergent
# ``offline`` / ``models_dir`` could silently flip another quest's HF
# config mid-build. The lock also gives us a safe point to detect that
# divergence and warn instead of clobbering.
_OFFLINE_ENV_LOCK = threading.Lock()

# HF env values this process has already applied via ``_apply_offline_env``.
# Lets a later quest's divergent config be recognized as ``--fleet``
# cross-talk (warn + keep the first FI value) WITHOUT mistaking the user's
# own pre-existing shell HF env for a conflict — the first apply still
# overwrites the ambient value so an explicit FI offline/models_dir config
# always takes effect on a single quest.
_APPLIED_OFFLINE_ENV: dict[str, str] = {}


def _apply_offline_env(cfg: KnowledgeConfig) -> None:
    """Set Hugging Face offline / local-cache env vars from FI config.

    Enforced at the HF library level (below Axon), so sentence-transformers
    and transformers load the embedding + reranker weights from the local
    cache with no network call:

    - ``offline`` → ``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1``. This
      sidesteps the ``huggingface_hub`` closed-client crash that a missing
      or flaky network triggers at quest start.
    - ``models_dir`` → ``HF_HOME`` points at the shipped HF-cache root
      (``<models_dir>/hub/models--...``). Produce one on a connected
      machine with ``launch.py --export-models <dir>``.

    Only sets a var when the corresponding knob is active, so a process
    that already exported its own HF env (and runs with FI defaults off)
    is left untouched. Process-global by nature — also inherited by the
    Axon sidecar via ``os.environ.copy()``.

    These vars are process-global, so they can't be made per-quest: a
    ``--fleet`` run shares one HF env across every quest in the process.
    To keep that safe and deterministic the mutation runs under
    ``_OFFLINE_ENV_LOCK`` and is *first-quest-wins*: the first apply still
    overwrites whatever the shell exported (so an explicit FI config always
    takes effect), but once FI has applied a value, a later quest whose
    config would point the same var somewhere different keeps the existing
    value and logs a warning — rather than yanking the cache out from under
    a brain another quest may still be lazily loading. The practical
    contract: a fleet must use a uniform ``offline`` / ``models_dir`` across
    its quests; a divergent one is surfaced as a warning instead of silent
    nondeterminism. (Tracking is keyed off ``_APPLIED_OFFLINE_ENV`` so a
    pre-existing shell HF var is never mistaken for an FI-set one.)
    """
    def _set(key: str, value: str) -> None:
        prior = _APPLIED_OFFLINE_ENV.get(key)
        if prior is not None and prior != value:
            _log.warning(
                "offline-env conflict on %s: this process already applied "
                "%r for an earlier quest but the current config wants %r. "
                "HF env vars are process-global, so a --fleet run must use a "
                "uniform offline/models_dir across its quests; keeping %r.",
                key, prior, value, prior,
            )
            return
        os.environ[key] = value
        _APPLIED_OFFLINE_ENV[key] = value

    with _OFFLINE_ENV_LOCK:
        if cfg.offline:
            _set("HF_HUB_OFFLINE", "1")
            _set("TRANSFORMERS_OFFLINE", "1")
        if cfg.models_dir:
            _set("HF_HOME", str(cfg.models_dir))


# Dedicated Axon project name for FI's corpus. Quest write-back and
# retrieval both operate inside this namespace so FI's documents
# don't mingle with whatever else the user does in Axon (e.g. a
# personal default project). All AxonBrain instances built via
# `_build_brain` switch to this project immediately after
# construction. To override (e.g. for testing), set the env var
# ``FI_AXON_PROJECT`` before importing this module.
#
# Axon enforces lowercase project names (letters / digits / hyphens
# / underscores, 1-50 chars, must start with a letter or digit).
# We use `frontier-insight` instead of the camelCase `FrontierInsight`
# you might expect — the trailing UI / docs still display the
# product name correctly; only Axon's on-disk directory uses the
# slug form.
import os as _os
FI_AXON_PROJECT = _os.environ.get("FI_AXON_PROJECT", "frontier-insight").strip() or "frontier-insight"


@dataclass
class RetrievedDoc:
    """Loose mirror of LangChain `Document`, decoupled so callers don't
    need to import langchain just to consume retrieval output."""

    content: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# External source adapters — each is a sync function with the signature
# `(query: str, top_k: int, *, timeout_s: float) -> list[RetrievedDoc]`,
# best-effort, never raises. The router invokes them off the event loop
# via `asyncio.to_thread`.
# ---------------------------------------------------------------------------


def _http_get_json(url: str, params: dict | None, timeout_s: float) -> dict | None:
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as c:
            r = c.get(url, params=params, headers={"User-Agent": "FrontierInsight/1.0"})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        _log.info("http GET %s failed: %s", url, e)
        return None


def _http_get_text(url: str, params: dict | None, timeout_s: float) -> str | None:
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as c:
            r = c.get(url, params=params, headers={"User-Agent": "FrontierInsight/1.0"})
            r.raise_for_status()
            return r.text
    except Exception as e:
        _log.info("http GET %s failed: %s", url, e)
        return None


def _arxiv_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    if not query.strip():
        return []
    params = {
        "search_query": f"all:{query.strip()}",
        "start": "0",
        "max_results": str(max(1, min(top_k, 20))),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    xml = _http_get_text("http://export.arxiv.org/api/query", params, timeout_s)
    if not xml:
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out: list[RetrievedDoc] = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        arxiv_url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        m = re.search(r"abs/([^/?#]+)$", arxiv_url)
        arxiv_id = m.group(1) if m else ""
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        pdf_url = next(
            (link.get("href", "") for link in entry.findall("a:link", ns) if link.get("title") == "pdf"),
            "",
        )
        out.append(RetrievedDoc(
            content=f"{title}\n\n{summary}".strip(),
            metadata={
                "source": "arxiv", "title": title, "authors": authors,
                "published": published, "arxiv_id": arxiv_id,
                "url": arxiv_url, "pdf_url": pdf_url,
            },
        ))
    return out


def _openalex_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    if not query.strip():
        return []
    params = {
        "search": query.strip(),
        "per-page": str(max(1, min(top_k, 25))),
    }
    data = _http_get_json("https://api.openalex.org/works", params, timeout_s)
    if not data or "results" not in data:
        return []
    out: list[RetrievedDoc] = []
    for w in data.get("results", []):
        title = w.get("title") or ""
        # OpenAlex returns an inverted index for abstracts; reconstruct.
        abstract = _openalex_reconstruct_abstract(w.get("abstract_inverted_index"))
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        authors = [
            (a.get("author", {}) or {}).get("display_name", "")
            for a in (w.get("authorships") or [])
        ]
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        oa_pdf = ((w.get("primary_location") or {}).get("pdf_url")) or ""
        out.append(RetrievedDoc(
            content=f"{title}\n\n{abstract}".strip(),
            metadata={
                "source": "openalex", "title": title, "authors": authors,
                "venue": venue, "year": w.get("publication_year"),
                "doi": doi, "url": w.get("id") or "", "pdf_url": oa_pdf,
                "cited_by": w.get("cited_by_count"),
                "open_access": (w.get("open_access") or {}).get("is_oa"),
            },
        ))
    return out


def _openalex_reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as `{word: [positions]}` to evade
    copyright scrapers. Reconstruct a readable sentence."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs or []:
            positions.append((i, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _crossref_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    """DOI metadata across all major publishers (paywalled or not).
    Abstracts present when the publisher submitted them; many won't
    have one but title + venue + author + year is still useful."""
    if not query.strip():
        return []
    params = {
        "query": query.strip(),
        "rows": str(max(1, min(top_k, 25))),
        "select": "DOI,title,abstract,author,container-title,published-print,published-online,publisher,URL",
    }
    data = _http_get_json("https://api.crossref.org/works", params, timeout_s)
    if not data:
        return []
    items = (data.get("message") or {}).get("items") or []
    out: list[RetrievedDoc] = []
    for it in items:
        title_list = it.get("title") or []
        title = title_list[0] if title_list else ""
        # Crossref returns abstract wrapped in JATS XML; strip the tags.
        raw_abs = it.get("abstract") or ""
        abstract = re.sub(r"<[^>]+>", " ", raw_abs).strip()
        authors = [
            f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}".strip()
            for a in (it.get("author") or [])
        ]
        venue_list = it.get("container-title") or []
        venue = venue_list[0] if venue_list else ""
        pub = it.get("publisher", "")
        year = None
        for k in ("published-print", "published-online"):
            dp = (it.get(k) or {}).get("date-parts") or []
            if dp and dp[0]:
                year = dp[0][0]
                break
        out.append(RetrievedDoc(
            content=f"{title}\n\n{abstract}".strip() if abstract else title,
            metadata={
                "source": "crossref", "title": title, "authors": authors,
                "venue": venue, "publisher": pub, "year": year,
                "doi": it.get("DOI", ""), "url": it.get("URL", ""),
            },
        ))
    return out


def _semantic_scholar_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    if not query.strip():
        return []
    params = {
        "query": query.strip(),
        "limit": str(max(1, min(top_k, 25))),
        "fields": "title,abstract,authors,year,venue,externalIds,openAccessPdf,url",
    }
    data = _http_get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search", params, timeout_s,
    )
    if not data or "data" not in data:
        return []
    out: list[RetrievedDoc] = []
    for p in data.get("data", []):
        title = p.get("title") or ""
        abstract = p.get("abstract") or ""
        authors = [a.get("name", "") for a in (p.get("authors") or [])]
        ext = p.get("externalIds") or {}
        out.append(RetrievedDoc(
            content=f"{title}\n\n{abstract}".strip(),
            metadata={
                "source": "semantic_scholar", "title": title, "authors": authors,
                "venue": p.get("venue", ""), "year": p.get("year"),
                "doi": ext.get("DOI", ""), "arxiv_id": ext.get("ArXiv", ""),
                "url": p.get("url", ""),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            },
        ))
    return out


def _pubmed_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    """Biomedical via NCBI E-utilities. Two-step: esearch → esummary."""
    if not query.strip():
        return []
    ids_data = _http_get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "pubmed", "term": query.strip(),
         "retmax": str(max(1, min(top_k, 25))), "retmode": "json"},
        timeout_s,
    )
    if not ids_data:
        return []
    ids = ((ids_data.get("esearchresult") or {}).get("idlist")) or []
    if not ids:
        return []
    sum_data = _http_get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        timeout_s,
    )
    if not sum_data:
        return []
    out: list[RetrievedDoc] = []
    result = sum_data.get("result") or {}
    for pmid in ids:
        rec = result.get(pmid) or {}
        title = rec.get("title") or ""
        authors = [a.get("name", "") for a in (rec.get("authors") or [])]
        venue = rec.get("fulljournalname") or rec.get("source", "")
        year = rec.get("pubdate", "")[:4]
        doi = ""
        for aid in rec.get("articleids") or []:
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
                break
        out.append(RetrievedDoc(
            content=title,  # esummary doesn't return abstract; would need efetch
            metadata={
                "source": "pubmed", "title": title, "authors": authors,
                "venue": venue, "year": year, "pmid": pmid, "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            },
        ))
    return out


def _core_search(query: str, top_k: int, *, timeout_s: float = 10.0) -> list[RetrievedDoc]:
    """CORE (https://core.ac.uk) — 240M+ open-access papers. Requires
    a free API key via `CORE_API_KEY` env var; degrades to [] if unset.
    Covers all fields with full-text where available."""
    import os
    api_key = os.environ.get("CORE_API_KEY", "").strip()
    if not api_key or not query.strip():
        return []
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as c:
            r = c.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": query.strip(), "limit": str(max(1, min(top_k, 25)))},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "FrontierInsight/1.0",
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        _log.info("core.ac.uk fallback failed: %s", e)
        return []
    out: list[RetrievedDoc] = []
    for w in data.get("results") or []:
        title = w.get("title") or ""
        abstract = w.get("abstract") or ""
        authors = [a.get("name", "") for a in (w.get("authors") or [])]
        out.append(RetrievedDoc(
            content=f"{title}\n\n{abstract}".strip(),
            metadata={
                "source": "core", "title": title, "authors": authors,
                "year": w.get("yearPublished"),
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "url": w.get("downloadUrl") or w.get("sourceFulltextUrls", [""])[0],
                "pdf_url": w.get("downloadUrl", ""),
                "venue": (w.get("publisher") or ""),
            },
        ))
    return out


def _google_scholar_search(query: str, top_k: int, *, timeout_s: float = 30.0) -> list[RetrievedDoc]:
    """Google Scholar has no official API. This adapter uses the
    `scholarly` PyPI package, which scrapes Scholar's HTML and is
    explicitly rate-limited / occasionally blocked by Google's
    anti-bot defenses. Mark as EXPERIMENTAL. If `scholarly` isn't
    installed or Google blocks the call, returns []. For production
    workloads, prefer OpenAlex / Semantic Scholar / CORE (which is
    what Google Scholar's open-data equivalents are designed to be)."""
    if not query.strip():
        return []
    try:
        from scholarly import scholarly  # type: ignore[import-not-found]
    except Exception:
        _log.info(
            "google_scholar source requested but `scholarly` not installed. "
            "Run `pip install scholarly`. Google has no official API; "
            "this adapter is best-effort and may be rate-limited."
        )
        return []
    out: list[RetrievedDoc] = []
    try:
        gen = scholarly.search_pubs(query.strip())
        for _ in range(max(1, min(top_k, 10))):
            try:
                pub = next(gen)
            except StopIteration:
                break
            bib = pub.get("bib", {}) or {}
            title = bib.get("title") or ""
            abstract = bib.get("abstract") or ""
            authors = bib.get("author") or []
            if isinstance(authors, str):
                authors = [authors]
            out.append(RetrievedDoc(
                content=f"{title}\n\n{abstract}".strip(),
                metadata={
                    "source": "google_scholar", "title": title, "authors": authors,
                    "venue": bib.get("venue", ""), "year": bib.get("pub_year"),
                    "url": pub.get("pub_url", ""),
                    "cited_by": pub.get("num_citations"),
                    # No reliable DOI from scholarly — title-based dedup applies.
                },
            ))
    except Exception as e:
        _log.info("google_scholar fallback failed: %s", e)
    return out


# ---------------------------------------------------------------------------
# General web search — Brave (keyed) with a keyless DuckDuckGo fallback.
#
# The academic adapters above only cover scholarly literature. A
# non-academic question ("SpaceX revenue by year", "Belgium vs Taiwan work
# culture") has no arXiv/Crossref match, so those adapters return
# irrelevant nearest-neighbours. These adapters add a general-web layer so
# such topics retrieve real sources. Every hit carries its ``url`` / ``site``
# / ``title`` so it can be cited in the paper / poster / slides References.
# ---------------------------------------------------------------------------


def _url_host(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


# Hosts whose clean full text is better obtained from the OA APIs
# (PMC BioC / Europe PMC / Unpaywall / preprint servers) than by scraping
# the publisher HTML — which is either a reCAPTCHA / cookie wall or a
# boilerplate-laden article page. For these we run the OA cascade *first*.
_ACADEMIC_HOST_RE = re.compile(
    r"(?:^|\.)(?:"
    r"ncbi\.nlm\.nih\.gov|europepmc\.org|arxiv\.org|biorxiv\.org|medrxiv\.org|"
    r"doi\.org|sciencedirect\.com|springer\.com|wiley\.com|tandfonline\.com|"
    r"sagepub\.com|mdpi\.com|nature\.com|plos\.org|frontiersin\.org|"
    r"ieee\.org|acm\.org|oup\.com|cambridge\.org|jstor\.org|bmj\.com|"
    r"cell\.com|pnas\.org|ssrn\.com|researchgate\.net|semanticscholar\.org|"
    r"academic\.oup\.com"
    r")$",
    re.I,
)


def _is_academic_source(url: str) -> bool:
    """True for URLs (PMC, preprint servers, DOI resolvers, major academic
    publishers) whose clean full text the OA cascade recovers better than an
    HTML scrape."""
    host = _url_host(url)
    return bool(host) and bool(_ACADEMIC_HOST_RE.search(host))


# Domains that rank well on commercial queries but are low-signal: SEO
# "market report" / press-release farms whose pages are vendor pitches,
# "request a free sample" gates, and recycled forecasts rather than
# primary data. Dropping them keeps the citation list to real sources
# (IEA, BNEF, government, OWID, journals, company filings). Curated and
# conservative — only well-known offenders, matched by registered domain.
_LOW_QUALITY_DOMAINS = frozenset({
    "marketsandmarkets.com", "grandviewresearch.com", "mordorintelligence.com",
    "fortunebusinessinsights.com", "alliedmarketresearch.com",
    "precedenceresearch.com", "marketresearchfuture.com", "imarcgroup.com",
    "researchandmarkets.com", "futuremarketinsights.com", "marknteladvisors.com",
    "verifiedmarketresearch.com", "polarismarketresearch.com",
    "globenewswire.com", "prnewswire.com", "businesswire.com",
    "evwire.com", "recharged.com", "expertmarketresearch.com",
    "coherentmarketinsights.com", "datamintelligence.com",
})


def _is_low_quality_domain(url: str) -> bool:
    """True for known SEO/market-report domains or tell-tale URL paths."""
    if not url:
        return False
    host = _url_host(url).lower()
    if host.startswith("www."):
        host = host[4:]
    if any(host == d or host.endswith("." + d) for d in _LOW_QUALITY_DOMAINS):
        return True
    low = url.lower()
    return any(
        s in low for s in (
            "/market-report", "market-size", "market-share-report",
            "request-sample", "request-a-sample", "-market-report-",
        )
    )


# A realistic browser header set. Many authoritative sites (IEA, S&P
# Global, etc.) return 403 to a request whose User-Agent advertises a bot
# (our old ``compatible; FrontierInsight/1.0``). These headers look like an
# ordinary Chrome request and clear the simple UA filters — NOT Cloudflare-
# grade JS challenges, which need a real browser and are out of scope.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "application/pdf;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


def _pdf_bytes_to_text(body: bytes, *, cap: int) -> str | None:
    """Extract text from in-memory PDF bytes via pypdf, capped at ``cap``
    bytes. Returns None when pypdf is missing or the parse fails."""
    try:
        import pypdf  # type: ignore[import-not-found]
        from io import BytesIO
    except ImportError:
        _log.info("pypdf not installed; cannot extract fetched PDF text")
        return None
    try:
        reader = pypdf.PdfReader(BytesIO(body))
    except Exception as e:
        _log.info("fetched PDF parse failed: %s", e)
        return None
    parts: list[str] = []
    total = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        if not txt.strip():
            continue
        parts.append(txt)
        total += len(txt.encode("utf-8", errors="replace"))
        if total >= cap:
            break
    if not parts:
        return None
    return "\n\n".join(parts)[:cap]


def _main_region(html: str) -> str:
    """Inner HTML of the first sizeable ``<article>`` / ``<main>`` element
    (drops site chrome) when present; else the whole document."""
    for tag in ("article", "main"):
        m = re.search(rf"(?is)<{tag}\b[^>]*>(.*?)</{tag}>", html)
        if m and len(m.group(1)) > 200:
            return m.group(1)
    return html


def _figure_note(html: str) -> str:
    """Collect figure captions + image alt-text from the page so the writer
    knows what visuals the source carried — datapoints often live in chart
    captions. We surface them as text; we do NOT download the images
    (licensing + relevance make auto-embedding unsafe). '' when none."""
    import html as _htmlmod
    caps: list[str] = []
    for m in re.finditer(r"(?is)<figcaption\b[^>]*>(.*?)</figcaption>", html):
        t = _htmlmod.unescape(re.sub(r"<[^>]+>", " ", m.group(1))).strip()
        if t:
            caps.append(t)
    for m in re.finditer(r'(?is)<img\b[^>]*\balt=["\']([^"\']{8,200})["\']', html):
        caps.append(m.group(1).strip())
    seen: set[str] = set()
    uniq: list[str] = []
    for c in caps:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
        if len(uniq) >= 8:
            break
    if not uniq:
        return ""
    return "\n\n[FIGURES IN SOURCE]\n" + "\n".join(f"- {c}" for c in uniq)


def _html_to_text(html: str) -> str:
    """HTML → readable text. Prefers ``trafilatura`` main-content extraction
    when it's installed (clean article body, drops boilerplate); otherwise
    isolates the ``<article>`` / ``<main>`` region and strips tags. Appends
    a short list of the source's figure captions / image alt-text so the
    writer is aware of the visuals (we don't fetch the images themselves)."""
    figs = _figure_note(html)
    # 1. Best path: trafilatura main-content extraction, if available.
    #    favor_recall keeps short paragraphs / captions / results sentences
    #    that scientific bodies depend on (the precision default drops them);
    #    older trafilatura without the kwarg falls back to the plain call.
    try:
        import trafilatura  # type: ignore[import-not-found]
        try:
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=True,
                favor_recall=True,
            )
        except TypeError:
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=True,
            )
        if extracted and extracted.strip():
            return (extracted.strip() + figs).strip()
    except Exception:
        pass
    # 2. Fallback: prefer the main article region, then strip tags.
    import html as _htmlmod
    region = _main_region(html)
    text = re.sub(
        r"(?is)<(script|style|noscript|nav|header|footer|aside|form|svg)\b.*?</\1>",
        " ", region,
    )
    text = re.sub(r"(?i)<li\b[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr|/table|/ul|/ol)\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)          # strip all remaining tags
    text = _htmlmod.unescape(text)
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return (text.strip() + figs).strip()


def _brave_search(
    query: str, top_k: int, *, api_key: str, timeout_s: float = 10.0,
) -> list[RetrievedDoc]:
    """Brave Search API (https://brave.com/search/api/). Mirrors the
    request shape Axon uses (``X-Subscription-Token`` header). Best-effort:
    any error / missing key returns []."""
    if not query.strip() or not api_key:
        return []
    count = max(1, min(top_k, 20))
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True) as c:
            r = c.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query.strip(), "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                    "User-Agent": "FrontierInsight/1.0",
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        _log.info("brave web search failed: %s", e)
        return []
    out: list[RetrievedDoc] = []
    for item in ((data.get("web") or {}).get("results") or [])[:count]:
        title = item.get("title") or ""
        url = item.get("url") or ""
        # Brave wraps matched terms in <strong>; strip the markup.
        snippet = re.sub(r"<[^>]+>", "", item.get("description") or "").strip()
        # Brave returns up to ~5 ``extra_snippets`` per result — extra
        # sentences pulled from the page. Fold them in so even a result
        # whose page later 403s the fetch still contributes several
        # datapoints to the research.
        extras = item.get("extra_snippets") or []
        extra_text = "\n".join(
            re.sub(r"<[^>]+>", "", e).strip() for e in extras if e
        ).strip()
        full_snippet = f"{snippet}\n{extra_text}".strip() if extra_text else snippet
        site = (item.get("meta_url") or {}).get("hostname") or _url_host(url)
        published = item.get("page_age") or item.get("age") or ""
        out.append(RetrievedDoc(
            content=f"{title}\n\n{full_snippet}".strip(),
            metadata={
                "source": "web_search", "backend": "brave", "kind": "web_page",
                "title": title, "url": url, "snippet": full_snippet,
                "site": site, "published": published,
            },
        ))
    _log.info("brave web search returned %d results for %r", len(out), query[:60])
    return out


def _ddg_decode(href: str) -> str:
    """DuckDuckGo HTML results wrap targets in a ``/l/?uddg=<encoded>``
    redirect; unwrap to the real URL."""
    from urllib.parse import urlparse, parse_qs, unquote
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        try:
            q = parse_qs(urlparse(href).query)
            if q.get("uddg"):
                return unquote(q["uddg"][0])
        except Exception:
            return href
    return href


def _ddg_search(
    query: str, top_k: int, *, timeout_s: float = 10.0,
) -> list[RetrievedDoc]:
    """Keyless DuckDuckGo HTML endpoint — no API key, but scraped and
    rate-limited/blockable. The no-key fallback so web search still works
    out of the box. Best-effort: any error returns []."""
    if not query.strip():
        return []
    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True, headers=_BROWSER_HEADERS,
        ) as c:
            r = c.post("https://html.duckduckgo.com/html/", data={"q": query.strip()})
            r.raise_for_status()
            html = r.text
    except Exception as e:
        _log.info("duckduckgo search failed: %s", e)
        return []
    anchors = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S,
    )
    snippets = re.findall(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html, re.S,
    )
    out: list[RetrievedDoc] = []
    for i, (href, title_html) in enumerate(anchors[:top_k]):
        url = _ddg_decode(href)
        title = _html_to_text(title_html).strip()
        snippet = _html_to_text(snippets[i]).strip() if i < len(snippets) else ""
        out.append(RetrievedDoc(
            content=f"{title}\n\n{snippet}".strip(),
            metadata={
                "source": "web_search", "backend": "duckduckgo", "kind": "web_page",
                "title": title, "url": url, "snippet": snippet,
                "site": _url_host(url), "published": "",
            },
        ))
    _log.info("duckduckgo search returned %d results for %r", len(out), query[:60])
    return out


def _sanitize_search_query(query: str, *, max_chars: int = 380) -> str:
    """Collapse a (possibly multi-line, paragraph-length) ``topic:`` into a
    single-line web-search query. Brave's API rejects embedded newlines
    with a 422 and caps query length (~400 chars); DuckDuckGo's HTML
    endpoint returns nothing for the same blob. Collapse all whitespace to
    single spaces and trim to a word boundary under the limit, so a rich
    multi-line topic still yields real web results to fetch."""
    q = re.sub(r"\s+", " ", query or "").strip()
    if len(q) <= max_chars:
        return q
    cut = q[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).strip()


def _web_search(
    query: str, top_k: int, *,
    backend: str = "auto", api_key: str = "", timeout_s: float = 10.0,
) -> list[RetrievedDoc]:
    """Dispatch a general-web search. ``auto`` uses Brave when a key is
    present (falling back to DuckDuckGo if Brave errors/returns empty),
    else the keyless DuckDuckGo endpoint. Low-signal SEO / market-report
    domains are dropped from the results so the paper cites real sources."""
    backend = (backend or "auto").lower()
    query = _sanitize_search_query(query)
    if backend == "duckduckgo":
        docs = _ddg_search(query, top_k, timeout_s=timeout_s)
    elif backend == "brave":
        if not api_key:
            _log.info(
                "web_search backend=brave but no BRAVE_API_KEY / "
                "knowledge.brave_api_key set — skipping web search",
            )
            return []
        docs = _brave_search(query, top_k, api_key=api_key, timeout_s=timeout_s)
    else:  # auto
        docs = _brave_search(query, top_k, api_key=api_key, timeout_s=timeout_s) if api_key else []
        if not docs:
            if api_key:
                _log.info("brave returned nothing; falling back to keyless DuckDuckGo")
            docs = _ddg_search(query, top_k, timeout_s=timeout_s)
    filtered = [
        d for d in docs if not _is_low_quality_domain(d.metadata.get("url", ""))
    ]
    if len(filtered) != len(docs):
        _log.info(
            "web_search dropped %d low-signal SEO/market-report result(s)",
            len(docs) - len(filtered),
        )
    return filtered


def _web_search_source(
    query: str, top_k: int, *, timeout_s: float = 10.0,
) -> list[RetrievedDoc]:
    """Registry/​router entry point — reads the Brave key from the
    environment (``Knowledge.asearch`` uses the config-aware path that also
    honours a YAML-set ``knowledge.brave_api_key``)."""
    import os
    return _web_search(
        query, top_k, backend="auto",
        api_key=os.environ.get("BRAVE_API_KEY", "").strip(), timeout_s=timeout_s,
    )


class _FetchBlocked(Exception):
    """Internal sentinel: the direct HTTP fetch was blocked (4xx)."""


# Specific full-page interstitial phrases — safe to match against an
# entire rendered HTML document (the headless path) without false
# positives, because they only appear on a challenge page, never in a
# real article's markup. "checking your browser" also covers PMC's
# reCAPTCHA wall ("Checking your browser - reCAPTCHA").
_INTERSTITIAL_MARKERS = (
    "just a moment",
    "performing security verification",
    "challenge-platform",
    "checking your browser",
    "cf-browser-verification",
    "request unsuccessful. incapsula",
)

# Broader bot-challenge / CAPTCHA phrases used ONLY for the length-gated
# text check below. These (recaptcha / captcha / "verify you are human")
# routinely appear in a real page's comment-widget markup, so they must
# NOT be matched against full HTML — only against already-extracted text
# that is also suspiciously short.
_BOT_CHALLENGE_MARKERS = _INTERSTITIAL_MARKERS + (
    "recaptcha",
    "captcha",
    "enable javascript and cookies",
    "please enable javascript",
    "verify you are human",
    "are you a robot",
    "ddos protection",
    "access denied",
)

# Extracted text shorter than this is treated as a failed fetch — a real
# article is far longer; a sub-threshold result is a block page, a cookie
# wall, or an empty render, none of which should be stored as full text.
# Kept conservative (the search snippet is ~250 chars, so this only
# rejects results no better than the snippet we already have).
_MIN_FULL_TEXT_CHARS = 300


def _looks_like_bot_challenge(text: str) -> bool:
    """True when ``text`` is a bot-challenge / CAPTCHA interstitial (or an
    otherwise-empty block page) rather than real article content. A genuine
    paper may *mention* "captcha"/"recaptcha" in its body, so the marker
    match is gated on the text being short — a challenge page is tiny, an
    article is not."""
    if not text or not text.strip():
        return True
    low = text.lower()
    if len(text) < 1500 and any(m in low for m in _BOT_CHALLENGE_MARKERS):
        return True
    return False


def _keep_fetched_text(text: str | None, snippet: str) -> bool:
    """Decide whether fetched page text is worth keeping over the search
    snippet we already have. Rejects bot-challenge / empty pages, then
    keeps the text only when it actually improves on the snippet — a
    snippet-relative bar rather than a flat length floor, so a legitimately
    short page (a press release, a stats page) is kept when it beats the
    snippet, while a block stub no longer than the snippet is dropped. With
    no snippet to compare against, requires a couple of sentences of
    substance (``_MIN_FULL_TEXT_CHARS``)."""
    if not text or _looks_like_bot_challenge(text):
        return False
    snippet_len = len((snippet or "").strip())
    if snippet_len:
        return len(text) > snippet_len
    return len(text) >= _MIN_FULL_TEXT_CHARS


# Paywall / metered-content infrastructure: the presence of one of these
# vendor scripts in the HTML means the page is gated, so the extractable
# text is a teaser, not the article.
_PAYWALL_SCRIPT_MARKERS = (
    "piano.io", "tinypass.com", "poool.fr", "sophi.io", "evolok.net",
    "js.pelcro.com", "cxense.com", "blueconic.net", "steadyhq.com",
    "/leaky-paywall/",
)
# Paywall call-to-action phrases. Real long articles can mention these in a
# footer, so a match only counts alongside a short body (see below).
_PAYWALL_TEXT_MARKERS = (
    "get access to the full", "purchase pdf", "purchase access", "buy article",
    "buy this article", "subscribe to read", "subscribe to view",
    "institutional login", "access through your institution",
    "rent this article", "sign in to read the full",
    "to read the full-text of this research",
)
# schema.org paywall flag (Google-endorsed) — the cleanest single signal.
_NOT_FREE_RE = re.compile(
    r'"isaccessibleforfree"\s*:\s*'
    r'(?:false|"false"|"https?://schema\.org/false")',
    re.I,
)


def _is_paywall_or_stub(html: str, text: str) -> bool:
    """Is this fetched page a paywall landing / abstract-only stub rather
    than real article content? Combines schema.org ``isAccessibleForFree:
    false``, known paywall-vendor scripts, and paywall call-to-action text.
    Conservative: every signal is corroborated by a short extracted body, so
    a genuine long article that merely links 'institutional login' in its
    header (or carries a partial-paywall ``hasPart`` flag) is NOT dropped."""
    if not html:
        return False
    low_html = html.lower()
    # A real, fully-extracted article body comfortably clears this; a teaser
    # / abstract-only stub does not.
    short = len(text) < 4000
    if short and _NOT_FREE_RE.search(low_html):
        return True
    if short and any(m in low_html for m in _PAYWALL_SCRIPT_MARKERS):
        return True
    if len(text) < 1800 and any(m in text.lower() for m in _PAYWALL_TEXT_MARKERS):
        return True
    return False


def _playwright_fetch_html(url: str, *, timeout_s: float) -> str | None:
    """Render ``url`` in a headless Chromium via Playwright and return its
    HTML. This executes JavaScript and clears most anti-bot challenges
    (Cloudflare) that block a plain HTTP GET — recovering the full article
    for authoritative sites (IEA, S&P, …). Optional dependency: returns
    None (with an actionable one-time hint) when Playwright or its Chromium
    browser isn't installed. Uses the SYNC API, which is valid here because
    this runs inside an ``asyncio.to_thread`` worker (no event loop in the
    thread)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        if not getattr(_playwright_fetch_html, "_warned", False):
            _log.info(
                "knowledge.headless_fetch is on but Playwright isn't installed; "
                "blocked pages fall back to snippets. Enable full rendering with "
                "`pip install playwright && playwright install chromium`.",
            )
            _playwright_fetch_html._warned = True  # type: ignore[attr-defined]
        return None
    _CF_MARKERS = _INTERSTITIAL_MARKERS
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                # Mask the most obvious automation signal (navigator.webdriver)
                # so a JS challenge is more likely to auto-clear.
                args=["--disable-blink-features=AutomationControlled"],
            )
            try:
                ctx = browser.new_context(
                    user_agent=_BROWSER_HEADERS["User-Agent"],
                    viewport={"width": 1280, "height": 900},
                    locale="en-US",
                )
                page = ctx.new_page()
                page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
                # Wait (best-effort) for a Cloudflare-style interstitial to
                # resolve into the real page before grabbing content.
                deadline = time.monotonic() + min(timeout_s, 20.0)
                html = page.content()
                while time.monotonic() < deadline:
                    low = html.lower()
                    if not any(m in low for m in _CF_MARKERS):
                        break
                    page.wait_for_timeout(1000)
                    html = page.content()
                # If a strict managed challenge (IEA-class Cloudflare) never
                # cleared, the "Just a moment…" interstitial is NOT content —
                # return None so the caller keeps the search snippet instead
                # of storing the challenge page as the source text.
                if any(m in html.lower() for m in _CF_MARKERS):
                    _log.info(
                        "playwright: %s stayed on a bot-challenge page; "
                        "keeping snippet", url,
                    )
                    return None
                return html
            finally:
                browser.close()
    except Exception as e:
        _log.info("playwright render %s failed: %s", url, e)
        return None


def _xml_to_text(xml: str) -> str:
    """Flatten JATS / XML full-text markup to readable plain text — strip
    tags + decode HTML entities, collapse whitespace. Good enough to feed
    the writer + plot steps; we don't need structural fidelity."""
    no_tags = re.sub(r"<[^>]+>", " ", xml)
    # html.unescape decodes named *and* numeric entities (&#x2014;, &eacute;,
    # …) so JATS punctuation/symbols survive, unlike a hand-rolled allowlist.
    decoded = _htmlmod.unescape(no_tags)
    return re.sub(r"\s+", " ", decoded).strip()


def _normalize_pmcid(raw: str) -> str:
    """Canonical ``PMC<digits>`` form for any PMCID-bearing string, or ''."""
    m = re.search(r"PMC\s*0*(\d+)", raw or "", re.I)
    return f"PMC{m.group(1)}" if m else ""


def _arxiv_id_from(url: str, md: dict) -> str:
    """Extract an arXiv id from metadata or an arxiv.org URL (version
    suffix stripped)."""
    raw = str(md.get("arxiv_id") or md.get("arxivId") or "").strip()
    if raw:
        return re.sub(r"v\d+$", "", raw)
    m = re.search(
        r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
        url or "", re.I,
    )
    return m.group(1) if m else ""


def _resolve_ids(doc: RetrievedDoc, *, timeout_s: float) -> dict:
    """Best-effort ``{doi, pmcid, arxiv_id}`` for a doc, from metadata, the
    source URL, and one Europe PMC cross-resolution call (DOI<->PMCID).
    Academic web-search hits routinely carry only a PMC link or only a DOI;
    filling in the missing id gives the OA cascade keys to work with."""
    md = doc.metadata
    url = str(md.get("url") or "")
    doi = str(md.get("doi") or "").strip().lower()
    pmcid = _normalize_pmcid(str(md.get("pmcid") or "")) or _normalize_pmcid(url)
    arxiv_id = _arxiv_id_from(url, md)
    if not doi:  # a bare DOI (incl. bioRxiv/medRxiv 10.1101/…) in the URL
        m = re.search(r"10\.\d{4,9}/[^\s\"'<>?#]+", url)
        if m:
            doi = m.group(0).rstrip(".").lower()
    # Cross-resolve DOI<->PMCID via one Europe PMC search when exactly one is
    # known — the search result carries both ids.
    if bool(doi) != bool(pmcid):
        query = f"DOI:{doi}" if doi else f"PMCID:{pmcid}"
        try:
            r = httpx.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": query, "format": "json",
                        "resultType": "lite", "pageSize": 1},
                headers=_BROWSER_HEADERS, timeout=timeout_s, follow_redirects=True,
            )
            if r.status_code == 200:
                res = ((r.json().get("resultList") or {}).get("result") or [{}])[0]
                doi = doi or str(res.get("doi") or "").lower()
                pmcid = pmcid or _normalize_pmcid(str(res.get("pmcid") or ""))
        except Exception:
            pass
    return {"doi": doi, "pmcid": pmcid, "arxiv_id": arxiv_id}


# BioC passage section types that are reference dumps / boilerplate — we
# drop them so the recovered body is article prose, not a citation list.
_BIOC_SKIP_SECTIONS = {"REF", "REFERENCES", "ACK_FUND", "COMP_INT", "SUPPL"}


def _pmc_bioc_fulltext(pmcid: str, *, timeout_s: float, cap: int) -> str | None:
    """Pull clean, structured full text from the PMC BioC API — the
    highest-quality PMC route (no HTML nav boilerplate, no reCAPTCHA wall).
    Assembles the article body from BioC passages, skipping reference-list /
    funding sections. None when the article isn't in the PMC OA subset."""
    if not pmcid:
        return None
    url = (
        "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
        f"pmcoa.cgi/BioC_json/{pmcid}/unicode"
    )
    try:
        r = httpx.get(
            url, headers=_BROWSER_HEADERS, timeout=timeout_s, follow_redirects=True,
        )
        if r.status_code != 200 or not r.content:
            return None
        data = r.json()
    except Exception as e:
        _log.info("pmc bioc %s failed: %s", pmcid, e)
        return None
    collections = data if isinstance(data, list) else [data]
    parts: list[str] = []
    for coll in collections:
        for d in (coll or {}).get("documents", []):
            for p in d.get("passages", []):
                sect = str((p.get("infons") or {}).get("section_type") or "").upper()
                if sect in _BIOC_SKIP_SECTIONS:
                    continue
                t = (p.get("text") or "").strip()
                if t:
                    parts.append(t)
    text = "\n".join(parts).strip()
    if len(text) >= _MIN_FULL_TEXT_CHARS:
        _log.info("pmc bioc: recovered %d chars for %s", len(text), pmcid)
        return text[:cap]
    return None


def _europepmc_fulltext(pmcid: str, *, timeout_s: float, cap: int) -> str | None:
    """OA full text as JATS XML from Europe PMC — second choice after BioC
    (the XML→text flatten loses structure). Only OA-subset articles expose
    ``fullTextXML``; others 404.

    NOTE: the path is ``/{PMCID}/fullTextXML`` with the ``PMC``-prefixed id
    and NO ``/PMC/`` path segment — ``/PMC/{id}/fullTextXML`` 404s for every
    article, OA or not."""
    if not pmcid:
        return None
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        r = httpx.get(
            url, headers=_BROWSER_HEADERS, timeout=timeout_s, follow_redirects=True,
        )
        if r.status_code != 200 or not r.content:
            return None
        text = _xml_to_text(r.text)
        if len(text) >= _MIN_FULL_TEXT_CHARS:
            _log.info("europepmc: recovered %d chars for %s", len(text), pmcid)
            return text[:cap]
    except Exception as e:
        _log.info("europepmc fulltext %s failed: %s", pmcid, e)
    return None


def _pdf_or_html_text(resp: Any, *, cap: int) -> str | None:
    """Extract text from a fetched response that may be a PDF or HTML,
    rejecting bot-challenge pages. Shared by the OA-PDF routes."""
    body = resp.content
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/pdf" in ctype or body[:5] == b"%PDF-":
        return _pdf_bytes_to_text(body, cap=cap)
    if "html" in ctype or b"<html" in body[:2048].lower():
        t = _html_to_text(body.decode("utf-8", errors="replace"))
        return None if _looks_like_bot_challenge(t) else t
    return None


def _preprint_fulltext(ids: dict, *, timeout_s: float, cap: int) -> str | None:
    """Full text for arXiv / bioRxiv / medRxiv preprints (always OA). arXiv:
    the HTML render (``arxiv.org/html``) then the PDF; bioRxiv/medRxiv: the
    ``.full.pdf`` for the ``10.1101/…`` DOI."""
    arxiv_id = ids.get("arxiv_id") or ""
    doi = ids.get("doi") or ""
    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True, headers=_BROWSER_HEADERS,
        ) as c:
            if arxiv_id:
                rr = c.get(f"https://arxiv.org/html/{arxiv_id}")
                if rr.status_code == 200 and b"<html" in rr.content[:2048].lower():
                    t = _html_to_text(rr.text)
                    if len(t) >= _MIN_FULL_TEXT_CHARS:
                        _log.info("arxiv: recovered HTML full text for %s", arxiv_id)
                        return t[:cap]
                rr = c.get(f"https://arxiv.org/pdf/{arxiv_id}")
                if rr.status_code == 200 and rr.content[:5] == b"%PDF-":
                    t = _pdf_bytes_to_text(rr.content, cap=cap)
                    if t and len(t) >= _MIN_FULL_TEXT_CHARS:
                        _log.info("arxiv: recovered PDF full text for %s", arxiv_id)
                        return t[:cap]
            if doi.startswith("10.1101/"):  # bioRxiv / medRxiv share the prefix
                for server in ("biorxiv", "medrxiv"):
                    try:
                        rr = c.get(f"https://www.{server}.org/content/{doi}v1.full.pdf")
                    except Exception:
                        continue
                    if rr.status_code == 200 and rr.content[:5] == b"%PDF-":
                        t = _pdf_bytes_to_text(rr.content, cap=cap)
                        if t and len(t) >= _MIN_FULL_TEXT_CHARS:
                            _log.info("%s: recovered full text for %s", server, doi)
                            return t[:cap]
    except Exception as e:
        _log.info("preprint fetch failed (%s / %s): %s", arxiv_id, doi, e)
    return None


def _unpaywall_fulltext(doi: str, *, timeout_s: float, cap: int) -> str | None:
    """Recover open-access full text via Unpaywall. The publisher's OA copy
    is often itself bot-walled (MDPI / SAGE / … 403), so we try EVERY
    ``oa_location`` — published version + direct-PDF + repository copies
    first (university repos rarely block) — and return the first that yields
    real content. Unpaywall needs a real contact email
    (``FI_UNPAYWALL_EMAIL`` / ``FI_CONTACT_EMAIL`` env, else a no-reply)."""
    if not doi:
        return None
    email = (
        os.environ.get("FI_UNPAYWALL_EMAIL")
        or os.environ.get("FI_CONTACT_EMAIL")
        or "frontier-insight@users.noreply.github.com"
    )

    def _loc_rank(loc: dict) -> tuple:
        # published > accepted > submitted; direct PDF before landing page;
        # repository host before publisher (publisher OA PDFs 403 most).
        ver = {"publishedVersion": 0, "acceptedVersion": 1}.get(
            loc.get("version") or "", 2,
        )
        has_pdf = 0 if loc.get("url_for_pdf") else 1
        repo = 0 if loc.get("host_type") == "repository" else 1
        return (ver, has_pdf, repo)

    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True, headers=_BROWSER_HEADERS,
        ) as c:
            r = c.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email})
            if r.status_code != 200:
                return None
            locs = sorted(r.json().get("oa_locations") or [], key=_loc_rank)
            candidates: list[str] = []
            for loc in locs:
                for key in ("url_for_pdf", "url"):
                    u = loc.get(key)
                    if u and u not in candidates:
                        candidates.append(u)
            for u in candidates:
                try:
                    rr = c.get(u)
                except Exception:
                    continue
                if rr.status_code != 200 or not rr.content:
                    continue
                text = _pdf_or_html_text(rr, cap=cap)
                if text and len(text) >= _MIN_FULL_TEXT_CHARS:
                    _log.info(
                        "unpaywall: recovered OA full text for DOI %s via %s",
                        doi, u[:70],
                    )
                    return text[:cap]
    except Exception as e:
        _log.info("unpaywall fulltext %s failed: %s", doi, e)
    return None


def _s2_oa_pdf(doi: str, *, timeout_s: float, cap: int) -> str | None:
    """Semantic Scholar ``openAccessPdf`` locator (best-effort). The shared
    unauth pool 429s readily — set ``SEMANTIC_SCHOLAR_API_KEY`` for a stable
    1 req/s lane; without a key this silently no-ops on a 429."""
    if not doi:
        return None
    headers = dict(_BROWSER_HEADERS)
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    try:
        r = httpx.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            headers=headers, timeout=timeout_s, follow_redirects=True,
        )
        if r.status_code != 200:
            return None
        pdf = (r.json().get("openAccessPdf") or {}).get("url") or ""
        if not pdf:
            return None
        rr = httpx.get(
            pdf, headers=_BROWSER_HEADERS, timeout=timeout_s, follow_redirects=True,
        )
        if rr.status_code == 200:
            text = _pdf_or_html_text(rr, cap=cap)
            if text and len(text) >= _MIN_FULL_TEXT_CHARS:
                _log.info("s2: recovered OA full text for DOI %s", doi)
                return text[:cap]
    except Exception as e:
        _log.info("s2 oa pdf %s failed: %s", doi, e)
    return None


def _core_fulltext(doi: str, *, timeout_s: float, cap: int) -> str | None:
    """CORE v3 aggregated full text (deepest green-OA repository coverage).
    Requires a free ``CORE_API_KEY`` — without it CORE returns the literal
    'Not available for public API users', so this no-ops when unset."""
    key = os.environ.get("CORE_API_KEY", "").strip()
    if not doi or not key:
        return None
    try:
        r = httpx.get(
            "https://api.core.ac.uk/v3/search/works",
            params={"q": f'doi:"{doi}"', "limit": 1},
            headers={"Authorization": f"Bearer {key}", **_BROWSER_HEADERS},
            timeout=timeout_s, follow_redirects=True,
        )
        if r.status_code != 200:
            return None
        results = r.json().get("results") or []
        if results:
            ft = (results[0].get("fullText") or "").strip()
            if len(ft) >= _MIN_FULL_TEXT_CHARS:
                _log.info("core: recovered %d chars for DOI %s", len(ft), doi)
                return ft[:cap]
    except Exception as e:
        _log.info("core fulltext %s failed: %s", doi, e)
    return None


def _fetch_via_open_apis(
    doc: RetrievedDoc, *, timeout_s: float, cap: int,
) -> str | None:
    """Open-access full-text cascade for academic sources whose publisher
    HTML is bot-walled or paywalled. Resolves ids once, then tries the
    routes in descending order of text quality / reliability, stopping at
    the first that yields real full text:

        PMC BioC → Europe PMC XML → preprint (arXiv/bioRxiv/medRxiv) →
        Unpaywall OA copy → Semantic Scholar → CORE (last two env-gated).

    Returns the recovered text, or None when no OA copy is reachable."""
    ids = _resolve_ids(doc, timeout_s=timeout_s)
    pmcid, doi, arxiv_id = ids["pmcid"], ids["doi"], ids["arxiv_id"]
    routes: list[Any] = []
    if pmcid:
        routes.append(lambda: _pmc_bioc_fulltext(pmcid, timeout_s=timeout_s, cap=cap))
        routes.append(lambda: _europepmc_fulltext(pmcid, timeout_s=timeout_s, cap=cap))
    if arxiv_id or doi.startswith("10.1101/"):
        routes.append(lambda: _preprint_fulltext(ids, timeout_s=timeout_s, cap=cap))
    if doi:
        routes.append(lambda: _unpaywall_fulltext(doi, timeout_s=timeout_s, cap=cap))
        routes.append(lambda: _s2_oa_pdf(doi, timeout_s=timeout_s, cap=cap))
        routes.append(lambda: _core_fulltext(doi, timeout_s=timeout_s, cap=cap))
    for route in routes:
        try:
            text = route()
        except Exception:
            text = None
        if text and len(text) >= _MIN_FULL_TEXT_CHARS:
            return text
    return None


def _fetch_web_page_text(
    doc: RetrievedDoc, *, timeout_s: float, max_kb: int, headless: bool = False,
) -> str | None:
    """Fetch a web result's page and extract readable text so the writer
    can quote real content (not just the snippet) and the plot step can
    pull numbers out of it. Handles HTML, plain text, AND PDFs (a web
    result is often a report PDF — e.g. an IEA outlook). Uses browser-like
    headers; when ``headless`` is set, a blocked (403) or empty direct
    fetch is retried by rendering the page in headless Chromium via
    Playwright (clears Cloudflare/JS). Best-effort: returns None on any
    failure."""
    url = doc.metadata.get("url") or ""
    if not url:
        return None
    cap = max_kb * 1024
    snippet = doc.content or ""
    # Academic sources (PMC, preprints, DOIs, major publishers): the OA
    # cascade gives cleaner full text than the publisher HTML (a wall or
    # boilerplate-laden page), so try it FIRST and only scrape on a miss.
    academic = _is_academic_source(url)
    if academic:
        api_text = _fetch_via_open_apis(doc, timeout_s=timeout_s, cap=cap)
        if api_text:
            return api_text
    blocked = False
    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True, headers=_BROWSER_HEADERS,
        ) as c:
            r = c.get(url)
            if r.status_code >= 400:
                blocked = r.status_code in (401, 403, 429)
                raise _FetchBlocked()
            ctype = (r.headers.get("content-type") or "").lower()
            body = r.content
    except _FetchBlocked:
        body = None
    except Exception as e:
        _log.info("web page fetch %s failed: %s", url, e)
        body = None
        blocked = True

    if body is not None:
        head = body[:2048].lstrip().lower()
        if "application/pdf" in ctype or body[:5] == b"%PDF-":
            text = _pdf_bytes_to_text(body, cap=cap)
            if text:
                return text
        elif "html" in ctype or head.startswith(b"<!doctype html") or b"<html" in head:
            raw = body.decode("utf-8", errors="replace")
            text = _html_to_text(raw)
            # A 200 can still be a bot-challenge / cookie wall (PMC's
            # reCAPTCHA, MDPI, …) whose extracted text is tiny garbage, or a
            # paywall / abstract-only stub. Reject those, and any stub no
            # better than the snippet, so we fall through to the headless
            # render / API fallbacks instead of storing it.
            if _keep_fetched_text(text, snippet) and not _is_paywall_or_stub(raw, text):
                return text[:cap]
        elif ctype.startswith("text/") or not ctype:
            text = body.decode("utf-8", errors="replace")
            if _keep_fetched_text(text, snippet):
                return text[:cap]

    # Direct fetch was blocked / empty / a challenge page. Try the headless
    # renderer (which clears Cloudflare and JS-only pages).
    if headless:
        html = _playwright_fetch_html(url, timeout_s=timeout_s)
        if html:
            text = _html_to_text(html)
            if _keep_fetched_text(text, snippet) and not _is_paywall_or_stub(html, text):
                return text[:cap]

    # Last resort for pages we didn't classify as academic: the OA cascade
    # can still resolve a DOI from a publisher URL and recover full text.
    # (Academic URLs already tried the cascade first, above.)
    if not academic:
        api_text = _fetch_via_open_apis(doc, timeout_s=timeout_s, cap=cap)
        if _keep_fetched_text(api_text, snippet):
            return api_text
    return None


_SOURCE_REGISTRY = {
    "arxiv": _arxiv_search,
    "openalex": _openalex_search,
    "crossref": _crossref_search,
    "semantic_scholar": _semantic_scholar_search,
    "pubmed": _pubmed_search,
    "core": _core_search,
    "google_scholar": _google_scholar_search,
    "web_search": _web_search_source,
}


# ---------------------------------------------------------------------------
# Local paper loader — for manually-downloaded paywalled PDFs etc.
# ---------------------------------------------------------------------------


def _load_local_paper(path: Path) -> RetrievedDoc | None:
    """Load a single PDF / Markdown / plain-text file into a RetrievedDoc.
    Returns None (with a warning) if the file is missing or unreadable.
    PDF support requires the optional `pypdf` dependency — without it
    PDFs are skipped with an actionable warning."""
    path = Path(path).expanduser()
    if not path.exists():
        _log.warning("local_paper not found: %s", path)
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in (".md", ".txt", ".rst"):
            content = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            content = _extract_pdf_text(path)
            if content is None:
                return None
        else:
            _log.warning(
                "local_paper %s: unsupported extension %r (expected .pdf/.md/.txt)",
                path.name, suffix,
            )
            return None
    except Exception as e:
        _log.warning("local_paper %s: read failed: %s", path.name, e)
        return None
    return RetrievedDoc(
        content=content[:50000],  # cap so the prompt doesn't explode
        metadata={
            "source": "local_paper",
            "title": path.stem.replace("_", " ").replace("-", " "),
            # `filename` only — we intentionally do NOT store the absolute
            # filesystem path. That leaks the user's home directory layout
            # into Axon's long-term store and makes the corpus non-portable
            # (an exported / shared Axon corpus would reveal `/home/<user>/`
            # or `C:\Users\<user>\...` paths). The filename is sufficient
            # to reconstruct provenance during review.
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "kind": "local_paper",
            # No `doi` field on purpose — local files don't dedup against
            # external sources unless the user has set the title to match.
        },
    )


def _extract_pdf_text(path: Path) -> str | None:
    """Best-effort PDF text extraction via `pypdf`. Returns None when
    the dep is missing — engine continues with whatever did load."""
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        _log.warning(
            "pypdf not installed; PDF %s skipped. "
            "Install with `pip install pypdf` to enable PDF ingestion, "
            "or convert to .md/.txt first.",
            path.name,
        )
        return None
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        _log.warning("PDF %s: pypdf parse failed: %s", path.name, e)
        return None
    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


_LOCAL_PAPER_SUFFIXES = (".pdf", ".md", ".txt", ".rst")


def _expand_local_paper_paths(paths: list[Path]) -> list[Path]:
    """Resolve each ``local_papers`` entry to concrete files. An entry may be
    a FILE (.pdf / .md / .txt / .rst) or a DIRECTORY — a directory is scanned
    recursively for supported files, so a quest can point ``local_papers`` at
    a folder that already holds the research material. De-duped, order-stable."""
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            candidates = [
                f for f in sorted(p.rglob("*"))
                if f.is_file()
                and f.suffix.lower() in _LOCAL_PAPER_SUFFIXES
                # Skip dotfiles AND anything under a hidden dir (.git/.venv/…),
                # so pointing at a repo root doesn't walk huge trees.
                and not any(part.startswith(".") for part in f.relative_to(p).parts)
            ]
            if not candidates:
                _log.warning("local_papers dir %s holds no .pdf/.md/.txt files", p)
        else:
            candidates = [p]
        for f in candidates:
            try:
                key = f.resolve()
            except OSError:
                key = f
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def _load_local_papers(paths: list[Path]) -> list[RetrievedDoc]:
    out: list[RetrievedDoc] = []
    for p in _expand_local_paper_paths(paths):
        doc = _load_local_paper(p)
        if doc is not None:
            out.append(doc)
    if out:
        _log.info(
            "loaded %d local_paper(s): %s",
            len(out), [d.metadata["filename"] for d in out],
        )
    return out


# ---------------------------------------------------------------------------
# Phase 2: full-text fetch from publisher URLs (requires host network
# access to the paywalled content; gracefully skips login walls).
# ---------------------------------------------------------------------------


_PDF_MAGIC = b"%PDF-"
# Citation-PDF meta tag (Highwire Press / Google Scholar convention).
# Supported by SPIE, Elsevier, Springer, Wiley, IEEE Xplore, ACS,
# Nature, OUP, Sage, T&F, etc. The simplest cross-publisher heuristic.
# Attribute order varies in the wild — match either name-then-content
# or content-then-name. We pick the first <meta> that contains both.
_CITATION_PDF_RES = (
    re.compile(
        rb'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        rb'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        re.IGNORECASE,
    ),
)


def _looks_like_pdf(content_type: str | None, body_prefix: bytes) -> bool:
    """Cheap two-factor check: server says it's a PDF AND the bytes
    actually start with the PDF magic. Login walls fail both — they
    return text/html with a <body>...</body> regardless of the URL."""
    if content_type and "application/pdf" in content_type.lower():
        return body_prefix.startswith(_PDF_MAGIC)
    # Some publishers omit Content-Type or send octet-stream. Trust the
    # magic bytes alone in those edge cases.
    return body_prefix.startswith(_PDF_MAGIC)


def _find_pdf_url_in_html(html_bytes: bytes) -> str | None:
    """Look for the `citation_pdf_url` meta tag in a landing page,
    handling both attribute orderings. We parse a fixed prefix so we
    don't load a 5 MB page into memory."""
    head = html_bytes[:200_000]
    for pat in _CITATION_PDF_RES:
        m = pat.search(head)
        if m:
            return m.group(1).decode("utf-8", errors="replace")
    return None


def _fetch_pdf_bytes(url: str, *, timeout_s: float) -> bytes | None:
    """GET `url`; return the body iff it's a real PDF (Content-Type +
    magic-bytes check). Login walls / HTML pages / 4xx all return None."""
    try:
        with httpx.Client(
            timeout=timeout_s, follow_redirects=True,
            headers={"User-Agent": "FrontierInsight/1.0"},
        ) as c:
            r = c.get(url)
            if r.status_code >= 400:
                return None
            body = r.content
            if not _looks_like_pdf(r.headers.get("content-type"), body[:8]):
                return None
            return body
    except Exception as e:
        _log.info("full-text GET %s failed: %s", url, e)
        return None


def _fetch_full_text(
    doc: RetrievedDoc, *, timeout_s: float, max_kb: int,
) -> str | None:
    """For one external-search hit, try to obtain the full-text PDF
    using whatever network access the host has. Best-effort: returns
    extracted text on success, None on any failure (login wall, no
    PDF, parse error, missing pypdf). Honors:

      1. doc.metadata["pdf_url"] (direct PDF URL) — preferred.
      2. doc.metadata["url"] (landing page) — GET to scrape for the
         citation_pdf_url <meta> tag, then GET that.

    Validation: Content-Type AND %PDF-magic. Login walls fail both.
    """
    pdf_url = doc.metadata.get("pdf_url") or ""
    landing = doc.metadata.get("url") or ""

    pdf_bytes: bytes | None = None
    if pdf_url:
        pdf_bytes = _fetch_pdf_bytes(pdf_url, timeout_s=timeout_s)
    if pdf_bytes is None and landing:
        # Scrape the landing page for a citation_pdf_url.
        try:
            with httpx.Client(
                timeout=timeout_s, follow_redirects=True,
                headers={"User-Agent": "FrontierInsight/1.0"},
            ) as c:
                page = c.get(landing)
                if page.status_code < 400:
                    candidate = _find_pdf_url_in_html(page.content)
                    if candidate:
                        # Resolve relative URLs against the landing page.
                        if not candidate.startswith(("http://", "https://")):
                            from urllib.parse import urljoin
                            candidate = urljoin(str(page.url), candidate)
                        pdf_bytes = _fetch_pdf_bytes(candidate, timeout_s=timeout_s)
        except Exception as e:
            _log.info("full-text landing-page %s failed: %s", landing, e)
            return None

    if not pdf_bytes:
        # No direct publisher PDF — fall back to the open-access cascade
        # (PMC BioC / Europe PMC / preprint / Unpaywall / …) which recovers
        # clean full text by PMCID / DOI / arXiv-id.
        return _fetch_via_open_apis(doc, timeout_s=timeout_s, cap=max_kb * 1024)

    # Extract text via pypdf. Cap to max_kb so we don't blow up the
    # downstream prompt.
    try:
        import pypdf  # type: ignore[import-not-found]
        from io import BytesIO
    except ImportError:
        _log.info("pypdf not installed; cannot extract fetched PDF text")
        return None
    try:
        reader = pypdf.PdfReader(BytesIO(pdf_bytes))
    except Exception as e:
        _log.info("fetched PDF parse failed: %s", e)
        return None
    parts: list[str] = []
    total = 0
    cap = max_kb * 1024
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        if not txt.strip():
            continue
        parts.append(txt)
        total += len(txt.encode("utf-8", errors="replace"))
        if total >= cap:
            break
    if not parts:
        return None
    return "\n\n".join(parts)[:cap]


async def _enrich_with_full_text(
    docs: list[RetrievedDoc],
    *,
    timeout_s: float,
    total_budget_s: float,
    max_kb: int,
    fetch_fn: Any = None,
) -> list[RetrievedDoc]:
    """Batch-enrich a list of external-source docs with fetched full
    text where possible. Runs fetches in parallel (each in a thread so
    httpx-sync doesn't block the loop). Respects a total wall-clock
    budget; docs not enriched within the budget are returned unchanged.
    Login walls / missing PDFs return the original doc untouched.

    ``fetch_fn(doc, *, timeout_s, max_kb) -> str | None`` is the per-doc
    fetcher — defaults to the academic PDF path (``_fetch_full_text``);
    the web layer passes ``_fetch_web_page_text`` to pull HTML page text
    instead. Resolved at call time (not bound as a default) so a test that
    monkeypatches ``_fetch_full_text`` is honoured."""
    if fetch_fn is None:
        fetch_fn = _fetch_full_text
    if not docs:
        return docs
    targets = [
        i for i, d in enumerate(docs)
        if d.metadata.get("pdf_url") or d.metadata.get("url")
    ]
    if not targets:
        return docs

    async def fetch_one(idx: int) -> tuple[int, str | None]:
        text = await asyncio.to_thread(
            fetch_fn,
            docs[idx],
            timeout_s=timeout_s,
            max_kb=max_kb,
        )
        return idx, text

    start = time.monotonic()
    enriched: list[RetrievedDoc] = list(docs)
    successes = 0

    # Use `as_completed` with a per-task deadline rather than
    # `wait_for(gather(...))`. On budget timeout the gather form
    # cancels every in-flight task AND discards any already-completed
    # results — so partial enrichment doesn't actually return partial
    # results. With as_completed we apply any result that lands before
    # the deadline and only abandon the *still-running* ones when the
    # budget expires.
    #
    # Caveat: `asyncio.to_thread` tasks cannot truly be cancelled
    # mid-blocking-call — the underlying thread will keep running its
    # synchronous httpx GET to completion. Since the per-doc HTTP
    # timeout (`timeout_s`) bounds that, the leaked threads are
    # short-lived. This is the best we can do without a custom
    # cancellable HTTP layer.
    pending = {asyncio.create_task(fetch_one(i)) for i in targets}
    deadline = start + total_budget_s
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                # Wait returned because the timeout fired with nothing
                # finishing — bail out of the loop.
                break
            for task in done:
                try:
                    idx, text = task.result()
                except Exception as e:
                    _log.info("full-text fetch task raised: %s", e)
                    continue
                if not text:
                    continue
                original = enriched[idx]
                new_meta = {
                    **original.metadata,
                    "fetched_full_text": True,
                    "full_text_bytes": len(text.encode("utf-8", errors="replace")),
                }
                enriched[idx] = RetrievedDoc(
                    content=f"{original.content}\n\n---FULL TEXT (fetched)---\n\n{text}",
                    metadata=new_meta,
                )
                successes += 1
    finally:
        # Cancel anything still running. The underlying threads will
        # finish their bounded HTTP call but won't deliver results.
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    if pending:
        _log.info(
            "full-text fetch budget %.1fs exceeded; "
            "enriched %d/%d docs (%d abandoned)",
            total_budget_s, successes, len(targets), len(pending),
        )
    else:
        _log.info(
            "full-text fetch: enriched %d/%d docs in %.1fs",
            successes, len(targets), time.monotonic() - start,
        )
    return enriched


# Catalog of sources/venues the agent should know about, including
# unsearchable-but-named ones (so the LLM can say "this topic belongs at
# SPIE — even though we can't query it, you should at least know which
# journals/conferences matter"). Searchable entries have
# `has_search_adapter=True` and a `name` matching `_SOURCE_REGISTRY`.
# Entries are seeded into Axon as `kind="fi_source_catalog"` on engine
# construction (config: `seed_source_catalog: true`), and the routing
# step reads them back so users can extend the catalog by ingesting
# their own knowledge into Axon (e.g., domain-specific venues).
_SOURCE_CATALOG: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "title": "General web search (Brave, or keyless DuckDuckGo)",
        "fields": ["all", "news", "business", "finance", "current events",
                   "culture", "general reference", "non-academic"],
        "access": "open (Brave key optional; DuckDuckGo keyless fallback)",
        "has_search_adapter": True,
        "when_to_use": "ANY non-academic or current-events question (company financials, "
                       "markets, products, history, culture, how-to) where scholarly "
                       "databases have no relevant match. Returns live web pages with "
                       "citable URLs; runs in parallel with the academic sources.",
    },
    {
        "name": "openalex",
        "title": "OpenAlex",
        "fields": ["all"],
        "access": "open",
        "has_search_adapter": True,
        "when_to_use": "broadest single open index (~200M works). Strong default for any topic.",
    },
    {
        "name": "arxiv",
        "title": "arXiv preprint server",
        "fields": ["physics", "computer science", "mathematics", "quantum computing",
                   "quantitative biology", "statistics", "electrical engineering"],
        "access": "open",
        "has_search_adapter": True,
        "when_to_use": "physics, CS, math, quantum, anything posted as a preprint before peer review.",
    },
    {
        "name": "crossref",
        "title": "Crossref DOI index",
        "fields": ["all"],
        "access": "metadata-only (full text behind publisher paywalls)",
        "has_search_adapter": True,
        "when_to_use": "to surface paywalled journal papers from Springer, Elsevier, IEEE, ACM, SPIE, ACS, Wiley, etc. by DOI metadata.",
    },
    {
        "name": "semantic_scholar",
        "title": "Semantic Scholar",
        "fields": ["all"],
        "access": "open (rate-limited without API key)",
        "has_search_adapter": True,
        "when_to_use": "broad coverage with abstracts + citation graph. Good for follow-the-citations workflows.",
    },
    {
        "name": "pubmed",
        "title": "PubMed (NCBI E-utilities)",
        "fields": ["biomedicine", "life sciences", "clinical", "genomics", "epidemiology"],
        "access": "open",
        "has_search_adapter": True,
        "when_to_use": "any biomedical / health-sciences topic.",
    },
    {
        "name": "core",
        "title": "CORE open-access aggregator",
        "fields": ["all"],
        "access": "open (free API key required: CORE_API_KEY env var)",
        "has_search_adapter": True,
        "when_to_use": "240M+ open-access papers, useful when seeking PDFs not just abstracts.",
    },
    {
        "name": "google_scholar",
        "title": "Google Scholar (unofficial, via `scholarly` package)",
        "fields": ["all"],
        "access": "open but no official API; scraped via scholarly; rate-limited / sometimes blocked",
        "has_search_adapter": True,
        "when_to_use": "last-resort broad search when other sources miss something. Prefer openalex/semantic_scholar.",
    },
    # ----- Unsearchable-but-named venues (catalog only, no adapter) -----
    {
        "name": "spie",
        "title": "SPIE Digital Library",
        "fields": ["optics", "photonics", "lithography", "EUV", "semiconductor patterning"],
        "access": "paywalled (search via crossref)",
        "has_search_adapter": False,
        "when_to_use": "EUV / DUV lithography, photoresist, optical metrology. Key venues: Proc. SPIE, JM3, J. Vac. Sci. Technol. B.",
    },
    {
        "name": "ieee_xplore",
        "title": "IEEE Xplore",
        "fields": ["electrical engineering", "electronics", "signal processing", "communications", "computing"],
        "access": "paywalled (search via crossref)",
        "has_search_adapter": False,
        "when_to_use": "EE / electronics topics. Key journals: IEEE TPAMI, TIT, TIP; key conferences: ICASSP, INFOCOM, IROS.",
    },
    {
        "name": "acm_digital_library",
        "title": "ACM Digital Library",
        "fields": ["computer science", "human-computer interaction", "systems"],
        "access": "paywalled (search via crossref)",
        "has_search_adapter": False,
        "when_to_use": "CS systems / HCI venues: TOCS, TOPLAS, SIGGRAPH, CHI, OSDI, SOSP.",
    },
    {
        "name": "nature_portfolio",
        "title": "Nature portfolio journals",
        "fields": ["all sciences"],
        "access": "paywalled (search via crossref)",
        "has_search_adapter": False,
        "when_to_use": "high-impact general-science findings. Nature, Nat. Phys., Nat. Chem., Nat. Mach. Intell., etc.",
    },
    {
        "name": "biorxiv",
        "title": "bioRxiv preprint server",
        "fields": ["biology"],
        "access": "open",
        "has_search_adapter": False,
        "when_to_use": "biology preprints. (Not yet wired as a search adapter — covered via openalex + pubmed for now.)",
    },
]


def _source_catalog_text() -> str:
    """Render the built-in catalog as a compact text block for the
    LLM-routing prompt."""
    lines: list[str] = []
    for s in _SOURCE_CATALOG:
        searchable = "yes" if s["has_search_adapter"] else "no (catalog-only)"
        fields = ", ".join(s["fields"])
        lines.append(
            f"- {s['name']}: {s['title']}. Fields: {fields}. "
            f"Access: {s['access']}. Searchable by FI: {searchable}. "
            f"When to use: {s['when_to_use']}"
        )
    return "\n".join(lines)


async def _route_sources_with_llm(
    topic: str,
    chosen_idea: dict | None,
    chat_fn: Any,
    fallback_sources: list[str],
) -> list[str]:
    """Call the agent to pick sources from the catalog for this topic.
    On any failure (parse error, empty list, LLM error), return the
    `fallback_sources` list. Only source names that exist in
    `_SOURCE_REGISTRY` are honored — catalog-only entries (e.g. SPIE)
    are logged so the user can see "this topic should also be checked
    on SPIE manually" even though FI can't programmatically search it.
    """
    try:
        catalog_text = _source_catalog_text()
        idea_text = ""
        if chosen_idea:
            idea_text = f"\n# Chosen research direction\n{chosen_idea.get('title','')}\n{chosen_idea.get('summary','') or chosen_idea.get('rationale','')}\n"
        prompt = (
            "You are picking which literature sources to query for a research topic. "
            "Choose AT LEAST one and AT MOST five sources from the catalog below. "
            "Prefer searchable sources (FI can actually query them); you may also "
            "note 1–2 catalog-only venues for the user's awareness.\n\n"
            "# Topic\n"
            f"{topic[:1500]}\n"
            f"{idea_text}\n"
            "# Source catalog\n"
            f"{catalog_text}\n\n"
            "Respond with a single JSON object, no prose:\n"
            "{\n"
            '  "sources_to_query":   ["<name>", ...],   // searchable sources FI will hit\n'
            '  "noteworthy_venues":  ["<name>", ...],   // catalog-only; for awareness only\n'
            '  "rationale": "<one sentence>"\n'
            "}"
        )
        text = await chat_fn([{"role": "user", "content": prompt}], temperature=0.0)
        # Lenient JSON parse — strip fences and find balanced braces.
        s = text.strip()
        if s.startswith("```"):
            nl = s.find("\n")
            if nl > 0 and s.endswith("```"):
                s = s[nl + 1 : -3].strip()
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("no JSON object in router response")
        parsed = json.loads(s[a : b + 1])
        raw = parsed.get("sources_to_query") or []
        chosen = [n for n in raw if n in _SOURCE_REGISTRY]
        noteworthy = parsed.get("noteworthy_venues") or []
        if noteworthy:
            _log.info(
                "source-router noted catalog-only venues for this topic: %s "
                "(rationale: %s)",
                noteworthy, parsed.get("rationale", "")[:120],
            )
        if not chosen:
            _log.info(
                "source-router returned no valid searchable sources; "
                "falling back to config: %s",
                fallback_sources,
            )
            return fallback_sources
        _log.info(
            "source-router picked %s (rationale: %s)",
            chosen, parsed.get("rationale", "")[:120],
        )
        return chosen
    except Exception as e:
        _log.info("source-router failed: %s; falling back to config", e)
        return fallback_sources


# ---------------------------------------------------------------------------
# Dedup + router
# ---------------------------------------------------------------------------


def _doc_dedup_key(d: RetrievedDoc) -> str:
    """Strongest available identifier first, falling back to a
    normalized title hash. Prevents the same paper appearing twice when
    multiple sources return it."""
    m = d.metadata
    if m.get("doi"):
        return f"doi:{m['doi'].lower()}"
    if m.get("arxiv_id"):
        return f"arxiv:{m['arxiv_id']}"
    if m.get("pmid"):
        return f"pmid:{m['pmid']}"
    # Web pages have no scholarly id — dedupe on the URL (normalized to
    # ignore the scheme + a trailing slash) so the same page from Brave +
    # DuckDuckGo collapses to one hit.
    if m.get("source") == "web_search" and m.get("url"):
        u = re.sub(r"^https?://", "", str(m["url"]).lower()).rstrip("/")
        return f"url:{u}"
    title = (m.get("title") or "").lower().strip()
    normalized = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", title))
    return f"title:{normalized}" if normalized else f"id:{id(d)}"


async def _route_external(
    query: str, top_k: int, sources: list[str], *, timeout_s: float = 10.0,
) -> list[RetrievedDoc]:
    """Run all requested adapters in parallel (each off the event loop)
    and merge. Returns up to top_k de-duplicated docs, preserving the
    original source-list order on collisions (first source wins)."""
    unknown = [s for s in sources if s not in _SOURCE_REGISTRY]
    if unknown:
        _log.warning(
            "unknown literature source(s) %s ignored; known: %s",
            unknown, list(_SOURCE_REGISTRY),
        )
    valid = [s for s in sources if s in _SOURCE_REGISTRY]
    if not valid:
        return []

    async def run_one(name: str) -> tuple[str, list[RetrievedDoc]]:
        fn = _SOURCE_REGISTRY[name]
        try:
            docs = await asyncio.to_thread(fn, query, top_k, timeout_s=timeout_s)
        except Exception as e:
            _log.info("source %s raised: %s", name, e)
            docs = []
        return name, docs

    results = await asyncio.gather(*(run_one(n) for n in valid))

    # Source-priority order = order of `sources` list. Dedup preserves
    # the first occurrence — so higher-priority sources win on collision.
    by_source: dict[str, list[RetrievedDoc]] = {n: d for n, d in results}
    seen: set[str] = set()
    merged: list[RetrievedDoc] = []
    for name in valid:
        for doc in by_source.get(name, []):
            key = _doc_dedup_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= top_k:
                _log.info(
                    "external retrieval %r: %d docs (sources tried: %s)",
                    query[:60], len(merged), valid,
                )
                return merged
    if merged:
        _log.info(
            "external retrieval %r: %d docs (sources tried: %s)",
            query[:60], len(merged), valid,
        )
    return merged


# ---------------------------------------------------------------------------
# Structured-ingest helpers — Patterns A (spine), B (header), C (topic event)
#
# Axon is a RAG store: chunked retrieval is great for "what did anyone say
# about X" but useless for "find Hinsberg 2017" or "what papers did quest Y
# cite?". These helpers add a small index-layer on top of plain ingestion:
#
#   • _render_paper_spine — one tight chunk per paper with the
#     title/authors/venue/DOI/topic/abstract/key-claims — the "card-catalog
#     entry". Title-shaped queries hit this directly.
#   • _prepend_paper_header — every body chunk gets a 1-line citation
#     prefix so the model always sees provenance, and BM25 / dense hits on
#     title-author terms land somewhere.
#   • _render_topic_event — one packed pointer per accepted quest, scoped
#     to the topic slug. Walks from a topic-query back to quests + papers.
# ---------------------------------------------------------------------------


def _slugify_topic(topic: str) -> str:
    """Stable, file-name-safe topic id used as `topic_id` and Axon tag."""
    s = (topic or "").lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = s.strip("-")
    return s[:80] or "untitled-topic"


def _paper_short_id(meta: dict[str, Any]) -> str:
    """Strongest available cross-quest paper identifier."""
    if meta.get("doi"):
        return f"doi:{str(meta['doi']).lower()}"
    if meta.get("arxiv_id"):
        return f"arxiv:{meta['arxiv_id']}"
    if meta.get("pmid"):
        return f"pmid:{meta['pmid']}"
    title = (meta.get("title") or "").lower().strip()
    norm = re.sub(r"\s+", "-", re.sub(r"[^\w\s]", "", title))[:60]
    return f"title:{norm}" if norm else "unknown"


def _paper_header_line(meta: dict[str, Any]) -> str:
    """Pattern B: the one-line citation prefix prepended to every body chunk.
    Compact enough not to bloat embeddings; rich enough that a retrieved
    chunk is self-describing without a second lookup."""
    title = meta.get("title") or "(untitled)"
    authors = meta.get("authors") or []
    if isinstance(authors, list) and authors:
        first = str(authors[0]).strip()
        author_str = f"{first}{' et al.' if len(authors) > 1 else ''}"
    else:
        author_str = str(authors or "").strip()
    year = meta.get("year") or (meta.get("published") or "")[:4]
    venue = meta.get("venue") or meta.get("publisher") or ""
    doi = meta.get("doi") or ""
    bits = [str(title).strip()]
    if author_str:
        bits.append(author_str)
    if year:
        bits.append(str(year))
    if venue:
        bits.append(str(venue))
    if doi:
        bits.append(f"DOI:{doi}")
    return "[" + " · ".join(bits) + "]"


def _prepend_paper_header(content: str, meta: dict[str, Any]) -> str:
    return f"{_paper_header_line(meta)}\n{content}"


def _render_paper_spine(
    meta: dict[str, Any],
    *,
    abstract: str = "",
    key_claims: list[str] | None = None,
) -> str:
    """Pattern A: a single-chunk 'card catalog entry' for a paper. The
    text intentionally repeats title/authors/DOI in label-prefixed lines
    so BM25-style retrieval has a target and dense embeddings get a
    concentrated paper-identity signal."""
    title = meta.get("title") or "(untitled)"
    authors = meta.get("authors") or []
    if isinstance(authors, list):
        authors_text = ", ".join(str(a) for a in authors[:8])
        if len(authors) > 8:
            authors_text += ", et al."
    else:
        authors_text = str(authors or "")
    year = meta.get("year") or (meta.get("published") or "")[:4]
    venue = meta.get("venue") or meta.get("publisher") or ""
    doi = meta.get("doi") or ""
    arxiv_id = meta.get("arxiv_id") or ""
    topic = meta.get("topic", "")

    lines = [f"TITLE: {title}"]
    if authors_text:
        lines.append(f"AUTHORS: {authors_text}")
    bib: list[str] = []
    if year:
        bib.append(f"YEAR: {year}")
    if venue:
        bib.append(f"VENUE: {venue}")
    if doi:
        bib.append(f"DOI: {doi}")
    if arxiv_id:
        bib.append(f"ARXIV: {arxiv_id}")
    if bib:
        lines.append("   ".join(bib))
    if topic:
        lines.append(f"TOPIC: {topic}")
    abs_text = (abstract or meta.get("abstract") or "").strip()
    if abs_text:
        lines.append("")
        lines.append("ABSTRACT:")
        lines.append(abs_text[:1200])
    if key_claims:
        lines.append("")
        lines.append("KEY CLAIMS:")
        for c in list(key_claims)[:10]:
            lines.append(f"  - {c}")
    return "\n".join(lines)


def _render_topic_event(
    *,
    topic: str,
    topic_id: str,
    quest_id: str,
    quest_title: str,
    verdict: str,
    score: Any,
    paper_refs: list[dict[str, Any]],
) -> str:
    """Pattern C: one packed pointer per accepted quest. Searchable by
    topic words; the body lists this quest's identity + the papers it
    cited. A future 'what do we know about X' query retrieves these
    events for topic X and the LLM can roll them up."""
    lines = [
        f"# Topic event: {topic[:200]}",
        f"TOPIC_ID: {topic_id}",
        "",
        f"QUEST: {quest_id}",
        f"  Title: {quest_title}",
        f"  Verdict: {verdict}",
    ]
    if score is not None:
        lines.append(f"  Score: {score}")
    if paper_refs:
        lines.append("")
        lines.append(f"PAPERS CITED ({len(paper_refs)}):")
        for r in paper_refs:
            t = r.get("title") or "(untitled)"
            ident_bits: list[str] = []
            if r.get("doi"):
                ident_bits.append(f"DOI:{r['doi']}")
            if r.get("arxiv_id"):
                ident_bits.append(f"arXiv:{r['arxiv_id']}")
            if r.get("pmid"):
                ident_bits.append(f"PMID:{r['pmid']}")
            ident = "  " + " ".join(ident_bits) if ident_bits else ""
            lines.append(f"  - {t}{ident}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Knowledge facade
# ---------------------------------------------------------------------------


class Knowledge:
    def __init__(self, cfg: KnowledgeConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled and _AXON_AVAILABLE
        self._brain: Any | None = None
        self._retriever: Any | None = None
        if cfg.enabled and not _AXON_AVAILABLE:
            _log.warning(
                "axon not importable (%s); knowledge layer disabled. "
                "External sources (%s) will still run if configured.",
                _AXON_IMPORT_ERROR, cfg.external_fallback,
            )
        if self.enabled:
            self._brain = self._build_brain(cfg)
            self._retriever = AxonRetriever(brain=self._brain, top_k=cfg.top_k)
            if cfg.seed_source_catalog:
                self._seed_source_catalog()

        # Manually-supplied papers (paywalled PDFs the user downloaded,
        # MD notes, etc.). Loaded once at construction so the engine
        # always sees them, even if Axon is disabled and external
        # sources all 404.
        self._local_papers: list[RetrievedDoc] = _load_local_papers(cfg.local_papers)
        # Also ingest them permanently into Axon when available, so
        # future quests find them via standard retrieval.
        if self._local_papers and self.enabled and self._brain is not None:
            self._ingest_local_papers_into_axon()

    def _ingest_local_papers_into_axon(self) -> None:
        """Write each loaded local paper into Axon as TWO documents:

        1. `fi_paper_spine` — one-chunk card-catalog entry, title-searchable.
        2. `fi_local_paper` — full body with a 1-line citation header
           prepended to every chunk so retrieval-time hits carry provenance.

        Both share `paper_id = _paper_short_id(meta)` so the spine and the
        body chunks can be reconciled later. Idempotent only when Axon
        deduplicates on identical content.
        """
        if self._brain is None:
            return
        docs: list[dict[str, Any]] = []
        for doc in self._local_papers:
            paper_id = _paper_short_id(doc.metadata)
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            spine_meta = {
                **doc.metadata,
                "kind": "fi_paper_spine",
                "paper_id": paper_id,
                "origin": "local_paper",
                "tag": f"fi-paper:{paper_id}",
                "ingested_at": now,
            }
            body_meta = {
                **doc.metadata,
                "kind": "fi_local_paper",
                "paper_id": paper_id,
                "tag": f"fi-paper:{paper_id}",
                "ingested_at": now,
            }
            docs.append({
                "id": self._mint_doc_id("fi_paper_spine", spine_meta),
                "text": _render_paper_spine(doc.metadata, abstract=doc.content[:1200]),
                "metadata": spine_meta,
            })
            docs.append({
                "id": self._mint_doc_id("fi_local_paper", body_meta),
                "text": _prepend_paper_header(doc.content, doc.metadata),
                "metadata": body_meta,
            })
        if not docs:
            return
        try:
            self._brain.ingest(docs)
            self._finalize_ingest_if_supported()
        except Exception as e:
            _log.warning("local_papers Axon ingest failed: %s", e)

    def _seed_source_catalog(self) -> None:
        """Ingest the built-in source catalog into Axon as one document
        per entry, tagged `kind="fi_source_catalog"`. Idempotent across
        engine reruns when Axon de-duplicates on identical content; if
        the brain has no de-dup, this can grow over time — surface as a
        config knob if it becomes a problem."""
        if self._brain is None:
            return
        try:
            docs: list[dict[str, Any]] = []
            for entry in _SOURCE_CATALOG:
                meta = {
                    "kind": "fi_source_catalog",
                    "tag": f"fi-source:{entry['name']}",
                    "source_name": entry["name"],
                    "fields": entry["fields"],
                    "access": entry["access"],
                    "has_search_adapter": entry["has_search_adapter"],
                    "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                text = (
                    f"# {entry['title']} ({entry['name']})\n"
                    f"Fields: {', '.join(entry['fields'])}\n"
                    f"Access: {entry['access']}\n"
                    f"When to use: {entry['when_to_use']}"
                )
                docs.append({
                    "id": f"fi_source_catalog:{entry['name']}",
                    "text": text,
                    "metadata": meta,
                })
            self._brain.ingest(docs)
            self._finalize_ingest_if_supported()
            _log.info("seeded %d source-catalog entries into Axon", len(_SOURCE_CATALOG))
        except Exception as e:
            _log.warning("source-catalog seed failed: %s", e)

    @staticmethod
    def _build_brain(cfg: KnowledgeConfig) -> Any:
        # Force offline / local-model loading BEFORE Axon constructs the
        # embedding model — otherwise transformers makes a network HEAD
        # check to huggingface.co that crashes on air-gapped machines.
        _apply_offline_env(cfg)
        if cfg.axon_config is None:
            brain = AxonBrain(AxonConfig())  # type: ignore[misc]
        elif isinstance(cfg.axon_config, Path):
            brain = AxonBrain(AxonConfig.from_yaml(cfg.axon_config))  # type: ignore[misc]
        elif isinstance(cfg.axon_config, dict):
            ac = AxonConfig.model_validate(cfg.axon_config)  # type: ignore[union-attr]
            brain = AxonBrain(ac)  # type: ignore[misc]
        else:
            ac = AxonConfig.model_validate(yaml.safe_load(yaml.safe_dump(cfg.axon_config)))  # type: ignore[union-attr]
            brain = AxonBrain(ac)  # type: ignore[misc]
        # Pin the FI corpus to its own project so quest
        # write-back / retrieval doesn't mingle with whatever else
        # the user does in Axon. `default` is where AxonBrain
        # initially lands; create the FI project if it doesn't
        # exist yet, then switch. ``ensure_project`` is idempotent
        # — second + subsequent calls are no-ops.
        try:
            from axon.projects import ensure_project  # type: ignore[import-not-found]
            ensure_project(
                FI_AXON_PROJECT,
                description="Frontier Insight quest papers + retrieval corpus",
            )
            brain.switch_project(FI_AXON_PROJECT)
        except Exception as e:
            _log.warning(
                "axon project setup for %r failed: %s; "
                "falling back to default project",
                FI_AXON_PROJECT, e,
            )
        return brain

    # ---- retrieval --------------------------------------------------------

    async def asearch(
        self,
        query: str,
        *,
        top_k: int | None = None,
        external_top_k: int | None = None,
        chosen_idea: dict | None = None,
        chat_fn: Any | None = None,
    ) -> list[RetrievedDoc]:
        """Async retrieval. Layers, merged + de-duplicated:

        1. **Pinned local papers** — files in `knowledge.local_papers`
           (PDF/MD/TXT the user manually placed). Always at the head,
           never dedupped away.
        2. **Axon** — the FI long-term store. Fast in-process call.
        3. **General web search** — Brave / DuckDuckGo. Runs IN PARALLEL
           with Axon for *every* query when ``knowledge.web_search`` is on,
           so non-academic topics (which have no scholarly match) get real
           sources instead of irrelevant nearest-neighbour papers. Web hits
           carry citable URLs.
        4. **Academic router** — `crossref` / `openalex` / etc., either
           agent-routed (`source_routing="auto"`) or YAML-configured. Fires
           when Axon is empty (the scholarly-breadth fallback), still in
           parallel with the web layer.

        Cap policy:
        * ``top_k`` (or ``self.cfg.top_k``) bounds the Axon RAG layer
          and the pinned-papers head — small-k because dense hits are
          precise.
        * The external (web + academic) layer is bounded by, in order of
          priority: an explicit ``external_top_k`` kwarg → otherwise the
          caller's ``top_k`` → otherwise ``self.cfg.external_top_k``.
        The merged result is capped at ``max(k, external_k)`` so a broad
        caller (the literature node) keeps web + Axon + academic breadth
        while a narrow caller (an ideate seed asking for 3) stays small.
        Local papers count toward the cap so the user isn't flooded."""
        k = top_k if top_k is not None else self.cfg.top_k
        if external_top_k is not None:
            external_k = external_top_k
        elif top_k is not None:
            # Caller pinned a per-call cap (e.g. ideate seed wants 3
            # papers) — honour it for external too. Otherwise the cheap
            # callers would silently trigger a 20-result fetch.
            external_k = top_k
        else:
            external_k = self.cfg.external_top_k

        # Layer 1: pinned local papers, always first.
        pinned = list(self._local_papers[:k])
        if len(pinned) >= k:
            return pinned

        # Layer 2: Axon — fast in-process call, done up front so the
        # academic fallback can be gated on whether it returned anything.
        remaining = k - len(pinned)
        axon_docs = self._search_axon(query, top_k=remaining)

        # Decide the academic source list (router or YAML fallback). The
        # web layer is handled separately (always-on), so strip any
        # ``web_search`` entry the router/fallback may name.
        fallback = self._fallback_sources()
        if self.cfg.source_routing == "auto" and chat_fn is not None:
            sources = await _route_sources_with_llm(
                topic=query, chosen_idea=chosen_idea,
                chat_fn=chat_fn, fallback_sources=fallback,
            )
        else:
            sources = fallback
        academic_sources = [s for s in sources if s != "web_search"]

        # Network layer, run concurrently:
        #   • web search — always when enabled (the new always-on breadth),
        #   • academic router — only when Axon came back empty (preserves
        #     the cost profile for corpus-backed academic quests).
        tasks: list[Any] = []
        want_web = bool(self.cfg.web_search)
        if want_web:
            tasks.append(asyncio.to_thread(
                _web_search, query, self.cfg.web_search_top_k,
                backend=self.cfg.web_search_backend,
                api_key=self.cfg.brave_api_key,
                timeout_s=self.cfg.full_text_fetch_timeout_s,
            ))
        run_academic = (not axon_docs) and bool(academic_sources)
        if run_academic:
            tasks.append(_route_external(query, external_k, academic_sources))

        web_docs: list[RetrievedDoc] = []
        academic_docs: list[RetrievedDoc] = []
        if tasks:
            results = await asyncio.gather(*tasks)
            idx = 0
            if want_web:
                web_docs = results[idx] or []
                idx += 1
            if run_academic:
                academic_docs = results[idx] or []

        # Enrichment: web hits get their page text (so the writer can quote
        # real content and the plot step can mine numbers); academic hits
        # opt into PDF full-text via the existing knob.
        if want_web and web_docs and self.cfg.web_fetch_pages:
            import functools
            web_docs = await _enrich_with_full_text(
                web_docs,
                timeout_s=self.cfg.full_text_fetch_timeout_s,
                total_budget_s=self.cfg.full_text_fetch_total_s,
                max_kb=self.cfg.full_text_max_kb,
                fetch_fn=functools.partial(
                    _fetch_web_page_text, headless=self.cfg.headless_fetch,
                ),
            )
        if self.cfg.try_fetch_full_text and academic_docs:
            academic_docs = await _enrich_with_full_text(
                academic_docs,
                timeout_s=self.cfg.full_text_fetch_timeout_s,
                total_budget_s=self.cfg.full_text_fetch_total_s,
                max_kb=self.cfg.full_text_max_kb,
            )

        # Merge pinned → Axon → web → academic, de-duplicated. Web ahead of
        # academic so that on a non-academic topic (Axon empty) the real
        # web hits lead and any irrelevant scholarly nearest-neighbours
        # trail (the relevance guard prunes those downstream).
        seen: set[str] = set()
        merged: list[RetrievedDoc] = []
        cap = max(k, external_k)
        for doc in (*pinned, *axon_docs, *web_docs, *academic_docs):
            key = _doc_dedup_key(doc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= cap:
                break
        return merged

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedDoc]:
        """Synchronous wrapper for non-async callers (tests, scripts).
        Spins a fresh event loop. Do NOT call from inside an already-
        running event loop — use `asearch()` there instead."""
        return asyncio.run(self.asearch(query, top_k=top_k))

    def _fallback_sources(self) -> list[str]:
        raw = self.cfg.external_fallback
        if isinstance(raw, str):
            return [raw] if raw and raw != "none" else []
        return list(raw or [])

    def _search_axon(self, query: str, *, top_k: int) -> list[RetrievedDoc]:
        if not self.enabled or self._retriever is None:
            return []
        retriever = self._retriever
        if top_k != self.cfg.top_k:
            retriever = retriever.with_overrides({"top_k": top_k})
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            _log.warning("axon retrieval failed for query=%r: %s", query[:60], e)
            return []
        return [
            RetrievedDoc(
                content=getattr(d, "page_content", str(d)),
                metadata=getattr(d, "metadata", {}) or {},
            )
            for d in docs
        ]

    # ---- ingest -----------------------------------------------------------

    def ingest(self, source: Path | str) -> bool:
        """Ingest a file (or URL) into Axon. The Axon API is
        ``brain.ingest(documents: list[dict])`` where each doc has
        ``id`` / ``text`` / ``metadata``. The previous version of this
        helper was written against an imagined ``ingest(str)`` /
        ``add_document(str)`` API that doesn't exist — so calls
        silently returned False and ``--ingest`` never actually wrote
        anything. Read the file, build the right shape, call the real
        API, then ``finalize_ingest`` so embeddings/index are flushed.
        Returns True on success.

        Document ``id`` is the absolute path string for files (so two
        files named ``data.csv`` from different directories don't
        collide), or the raw source string for URLs."""
        if not self.enabled or self._brain is None:
            return False
        try:
            path = Path(source)
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                abs_str = str(path.resolve())
                # Forward slashes so the same id is generated on
                # Windows + POSIX when the file is reachable through
                # equivalent paths.
                doc_id = f"fi_local_paper:{abs_str.replace(chr(92), '/')}"
                source_str = abs_str
            else:
                text = str(source)
                doc_id = f"fi_local_paper:{source}"
                source_str = str(source)
            self._brain.ingest([{
                "id": doc_id,
                "text": text,
                "metadata": {
                    "source": source_str,
                    "kind": "fi_local_paper",
                },
            }])
            self._finalize_ingest_if_supported()
            return True
        except Exception as e:
            _log.warning("axon ingest failed for %s: %s", source, e)
            return False

    def add_text(
        self, *, text: str, kind: str, metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Ingest a pre-formatted text document into Axon. Returns
        True iff the call succeeded.

        Used by the post-quest write-back path (``add_quest_artifacts``)
        and by every PM-command (``/proposal``, ``/digest``,
        ``/portfolio``, ``/critique``, ``/summarize``) to land their
        respective ``fi_*`` documents. Tolerates a disabled or missing
        Axon brain — returns False silently in those cases.

        Calls the real Axon API: ``brain.ingest([{id, text, metadata}])``
        followed by ``finalize_ingest`` so the embeddings + index are
        flushed and the document is queryable on the next ``asearch``.
        ``kind`` is folded into ``metadata['kind']`` so ``fi_*``
        documents stay filterable downstream."""
        if not self.enabled or self._brain is None:
            return False
        try:
            doc_id = self._mint_doc_id(kind, metadata)
            doc = {
                "id": doc_id,
                "text": text,
                "metadata": {**(metadata or {}), "kind": kind},
            }
            self._brain.ingest([doc])
            self._finalize_ingest_if_supported()
            return True
        except Exception as e:
            _log.warning("axon add_text(kind=%s) failed: %s", kind, e)
            return False

    @staticmethod
    def _mint_doc_id(kind: str, metadata: dict[str, Any] | None) -> str:
        """Stable doc id derived from metadata when possible so
        re-ingesting the same logical doc updates rather than
        duplicates. Order of preference: explicit ``id`` in metadata,
        ``quest_id`` (post-quest writebacks), ``proposal_id`` (from
        /proposal), ``critique_id``, ``digest_id``, ``summary_id``,
        ``portfolio_id``. Falls back to ``<kind>-<epoch>-<uuid4[:8]>``
        when none are present."""
        md = metadata or {}
        # Stable-key preference list. For per-quest writebacks the
        # quest_id is canonical; for local papers + external-ref
        # spines the paper_id is. ``rel_path`` discriminates
        # ``fi_summary_input`` docs that share a parent summary_id
        # but live at different paths. ``tag`` is the wildcard
        # discriminator the topic-event + external-ref-spine paths
        # use. Composite keys (kind + most-specific-id + most-
        # specific-discriminator) prevent the "two docs collide
        # under one id and the later one overwrites the earlier"
        # failure mode.
        primary_keys = (
            "id", "quest_id", "proposal_id", "critique_id",
            "digest_id", "summary_id", "portfolio_id",
        )
        primary: str | None = None
        for key in primary_keys:
            v = md.get(key)
            if v:
                primary = str(v)
                break
        # Discriminators: included even when primary is set, so docs
        # like ``fi_summary_input`` (shared summary_id, distinct
        # rel_path) get distinct ids.
        secondary_keys = ("paper_id", "rel_path", "tag")
        secondary_parts: list[str] = []
        for key in secondary_keys:
            v = md.get(key)
            if v:
                secondary_parts.append(str(v))
        if primary and secondary_parts:
            return f"{kind}:{primary}:{':'.join(secondary_parts)}"
        if primary:
            return f"{kind}:{primary}"
        if secondary_parts:
            return f"{kind}:{':'.join(secondary_parts)}"
        # Last resort — no stable identifier at all. Time + uuid is
        # the documented fallback; re-ingesting the same logical doc
        # WILL duplicate here, so callers should pass at least one
        # of the keys above when they know it.
        import time
        import uuid
        return f"{kind}:{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def _finalize_ingest_if_supported(self) -> None:
        """Some Axon builds require ``finalize_ingest`` after a batch to
        actually flush embeddings to the store; older builds do it
        inside ``ingest``. We call it when present and swallow
        AttributeError so a downgrade doesn't break the path."""
        if self._brain is None:
            return
        fin = getattr(self._brain, "finalize_ingest", None)
        if callable(fin):
            try:
                fin()
            except Exception as e:
                _log.warning("axon finalize_ingest failed: %s", e)

    def add_quest_artifacts(
        self,
        *,
        quest_id: str,
        paper_md_path: Path,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Write a finished, accepted quest into Axon as a structured
        bundle so RAG retrieval can answer title-, topic-, and
        provenance-shaped queries — not just "what did anyone say
        about X". Five documents land per quest:

        1. **`fi_paper_spine`** — single tight chunk holding title, authors,
           topic, DOI (if any), abstract, and the analysis key-findings as
           bullet "key claims". This is what surfaces for title queries.
        2. **`fi_quest_paper`** — full `paper.md` content with a 1-line
           citation header prepended (so any chunk that hits in retrieval
           carries its own provenance).
        3. **`fi_quest_summary`** — analysis summary + structured JSON
           (hypothesis / findings / result_json / verdict / score / model).
           Backwards-compat tag-shape; same as before but now carries
           `paper_refs` listing external papers this quest consumed.
        4. **`fi_topic_event`** — one packed pointer keyed by `topic_id`
           (a slug of the topic). Lists this quest + the papers it cited,
           so a topic-scoped retrieval surfaces all related accepted work.
        5. **`fi_external_ref_spine` × N** — thin card-catalog entries for
           each external paper this quest consumed (passed in via
           `metadata['external_refs']`). Only papers that contributed to
           an accepted quest land here, keeping Axon curated rather than
           a raw web cache.

        Returns True iff every doc above wrote successfully.
        """
        if not self.enabled or self._brain is None:
            return False
        if not self.cfg.write_back_quests:
            return False

        external_refs = list((metadata or {}).get("external_refs") or [])
        meta_no_refs = {k: v for k, v in (metadata or {}).items() if k != "external_refs"}

        topic = str(meta_no_refs.get("topic", "")).strip()
        topic_id = _slugify_topic(topic)
        paper_id = f"quest:{quest_id}"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        base_meta: dict[str, Any] = {
            **meta_no_refs,
            "tag": f"fi-quest:{quest_id}",
            "quest_id": quest_id,
            "topic_id": topic_id,
            "paper_id": paper_id,
            "ingested_at": now,
        }

        try:
            paper_md = (
                paper_md_path.read_text(encoding="utf-8")
                if paper_md_path.exists()
                else ""
            )
            quest_paper_meta_for_helpers = {
                "title": meta_no_refs.get("title", ""),
                "authors": meta_no_refs.get("authors") or [],
                "year": meta_no_refs.get("year") or now[:4],
                "venue": "Frontier Insight (internal quest)",
                "topic": topic,
            }

            # Batch ALL five docs into a single ``brain.ingest`` call
            # so the embeddings + index are flushed once at the end —
            # one ``finalize_ingest`` for the whole bundle. The
            # previous shape sent five separate ``add_text`` calls
            # which (a) didn't exist on the Axon API at all and (b)
            # would have over-flushed if they had.
            docs: list[dict[str, Any]] = []

            def _enq(kind: str, text: str, extra_meta: dict[str, Any]) -> None:
                merged = {**extra_meta, "kind": kind}
                docs.append({
                    "id": self._mint_doc_id(kind, merged),
                    "text": text,
                    "metadata": merged,
                })

            # (1) Spine — title-searchable card-catalog entry.
            spine_text = _render_paper_spine(
                quest_paper_meta_for_helpers,
                abstract=summary[:1200],
                key_claims=list(meta_no_refs.get("key_findings") or []),
            )
            _enq("fi_paper_spine", spine_text, {**base_meta, "origin": "quest_paper"})

            # (2) Body — full paper with 1-line citation header prepended.
            body_text = _prepend_paper_header(paper_md, quest_paper_meta_for_helpers)
            _enq("fi_quest_paper", body_text, base_meta)

            # (3) Summary — backwards-compat kind; gains paper_refs in metadata.
            summary_payload = {
                "summary": summary,
                **{k: v for k, v in meta_no_refs.items() if k != "summary"},
                "external_refs": external_refs,
            }
            summary_text = (
                f"{summary}\n\n"
                f"---structured-findings---\n"
                f"{json.dumps(summary_payload, indent=2, default=str)}"
            )
            _enq("fi_quest_summary", summary_text, {
                **base_meta,
                "paper_refs": [_paper_short_id(r) for r in external_refs],
            })

            # (4) Topic event — pointer doc scoped by topic slug.
            topic_text = _render_topic_event(
                topic=topic,
                topic_id=topic_id,
                quest_id=quest_id,
                quest_title=str(meta_no_refs.get("title", "")),
                verdict=str(meta_no_refs.get("verdict", "")),
                score=meta_no_refs.get("score"),
                paper_refs=external_refs,
            )
            _enq("fi_topic_event", topic_text, {
                "tag": f"fi-topic:{topic_id}",
                "topic_id": topic_id,
                "topic": topic,
                "quest_id": quest_id,
                "verdict": meta_no_refs.get("verdict", ""),
                "score": meta_no_refs.get("score"),
                "paper_refs": [_paper_short_id(r) for r in external_refs],
                "ingested_at": now,
            })

            # (5) External-ref spines — curated card-catalog entries for
            # papers that actually contributed to an accepted quest.
            for ref in external_refs:
                ref_paper_id = _paper_short_id(ref)
                ref_spine_text = _render_paper_spine(
                    {**ref, "topic": topic},
                    abstract=ref.get("abstract", ""),
                )
                _enq("fi_external_ref_spine", ref_spine_text, {
                    **{k: v for k, v in ref.items() if k != "abstract"},
                    "paper_id": ref_paper_id,
                    "tag": f"fi-paper:{ref_paper_id}",
                    "topic_id": topic_id,
                    "consumed_by_quests": [quest_id],
                    "ingested_at": now,
                })

            self._brain.ingest(docs)
            self._finalize_ingest_if_supported()
            _log.info(
                "axon writeback: %d docs ingested for quest %s "
                "(kinds: %s)",
                len(docs), quest_id,
                ", ".join(sorted({d["metadata"]["kind"] for d in docs})),
            )
            return True
        except Exception as e:
            _log.warning("axon write-back failed for quest %s: %s", quest_id, e)
            return False
