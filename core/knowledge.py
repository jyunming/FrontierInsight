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
import json
import logging
import re
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


_SOURCE_REGISTRY = {
    "arxiv": _arxiv_search,
    "openalex": _openalex_search,
    "crossref": _crossref_search,
    "semantic_scholar": _semantic_scholar_search,
    "pubmed": _pubmed_search,
    "core": _core_search,
    "google_scholar": _google_scholar_search,
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


def _load_local_papers(paths: list[Path]) -> list[RetrievedDoc]:
    out: list[RetrievedDoc] = []
    for p in paths:
        doc = _load_local_paper(Path(p))
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
        return None

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
) -> list[RetrievedDoc]:
    """Batch-enrich a list of external-source docs with fetched full
    text where possible. Runs fetches in parallel (each in a thread so
    httpx-sync doesn't block the loop). Respects a total wall-clock
    budget; docs not enriched within the budget are returned unchanged.
    Login walls / missing PDFs return the original doc untouched."""
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
            _fetch_full_text,
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
        if cfg.axon_config is None:
            return AxonBrain(AxonConfig())  # type: ignore[misc]
        if isinstance(cfg.axon_config, Path):
            return AxonBrain(AxonConfig.from_yaml(cfg.axon_config))  # type: ignore[misc]
        if isinstance(cfg.axon_config, dict):
            ac = AxonConfig.model_validate(cfg.axon_config)  # type: ignore[union-attr]
            return AxonBrain(ac)  # type: ignore[misc]
        ac = AxonConfig.model_validate(yaml.safe_load(yaml.safe_dump(cfg.axon_config)))  # type: ignore[union-attr]
        return AxonBrain(ac)  # type: ignore[misc]

    # ---- retrieval --------------------------------------------------------

    async def asearch(
        self,
        query: str,
        *,
        top_k: int | None = None,
        chosen_idea: dict | None = None,
        chat_fn: Any | None = None,
    ) -> list[RetrievedDoc]:
        """Async retrieval. Three layers, in order:

        1. **Pinned local papers** — files in `knowledge.local_papers`
           (PDF/MD/TXT the user manually placed). Always at the head,
           never dedupped away.
        2. **Axon** — the FI long-term store. Fast in-process call.
        3. **External router** — `crossref` / `openalex` / etc., either
           agent-routed (`source_routing="auto"`) or YAML-configured.

        Local papers count toward `top_k` so the user doesn't get
        flooded — if they supplied 8 papers and top_k=5, only the first
        5 are returned (and external sources aren't queried)."""
        k = top_k if top_k is not None else self.cfg.top_k

        # Layer 1: pinned local papers, always first.
        pinned = list(self._local_papers[:k])
        if len(pinned) >= k:
            return pinned

        # Layer 2: Axon. If Axon returns anything, take it next.
        remaining = k - len(pinned)
        axon_docs = self._search_axon(query, top_k=remaining)
        if axon_docs:
            return pinned + axon_docs[:remaining]

        # Layer 3: external router.
        remaining = k - len(pinned)
        fallback = self._fallback_sources()
        if self.cfg.source_routing == "auto" and chat_fn is not None:
            sources = await _route_sources_with_llm(
                topic=query, chosen_idea=chosen_idea,
                chat_fn=chat_fn, fallback_sources=fallback,
            )
        else:
            sources = fallback
        if not sources:
            return pinned
        external = await _route_external(query, remaining, sources)
        # Phase 2: opportunistic full-text fetch for external hits. Off
        # by default; opt-in via `knowledge.try_fetch_full_text: true`.
        # Login walls fail the Content-Type + %PDF magic-bytes check
        # so the quest is unaffected when the host lacks access.
        if self.cfg.try_fetch_full_text and external:
            external = await _enrich_with_full_text(
                external,
                timeout_s=self.cfg.full_text_fetch_timeout_s,
                total_budget_s=self.cfg.full_text_fetch_total_s,
                max_kb=self.cfg.full_text_max_kb,
            )
        return pinned + external[:remaining]

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
        Returns True on success."""
        if not self.enabled or self._brain is None:
            return False
        try:
            path = Path(source)
            text = path.read_text(encoding="utf-8") if path.is_file() else str(source)
            doc_id = path.name if path.is_file() else str(source)
            self._brain.ingest([{
                "id": doc_id,
                "text": text,
                "metadata": {
                    "source": str(path) if path.is_file() else str(source),
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
        for key in ("id", "quest_id", "proposal_id", "critique_id",
                    "digest_id", "summary_id", "portfolio_id"):
            v = md.get(key)
            if v:
                return f"{kind}:{v}"
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
