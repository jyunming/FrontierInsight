"""Topic-relevant figure enrichment from license-clean web sources.

Plots chart the numbers a quest gathered; this adds *illustrative* figures —
a topic diagram, or a real figure from a cited open-access paper — so a
study (especially a qualitative one) carries a visual point of view, not
only bar charts. Two sources, both gated on a FREE / OPEN license so the
generated paper / poster / slides never embed copyrighted images:

  1. **Wikimedia Commons** — topic diagrams / schematics / photos under a
     Creative-Commons or public-domain license, via the MediaWiki API with a
     rasterised thumbnail (works even for SVG originals).
  2. **arXiv** — a real figure from a CC-licensed paper the quest already
     cited, lifted from the paper's HTML rendering (only papers whose authors
     chose a redistributable CC license; the default arXiv license forbids
     reuse, so those are skipped).

Each call returns a :class:`WebFigure` (downloaded image + caption + source
URL + license + attribution) or nothing. Best-effort throughout: any
network / parse error, a non-free license, or a missing image yields no
figure rather than raising — enrichment never blocks a quest.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

_log = logging.getLogger("frontier_insight.figures")

# Polite, policy-compliant UA — Wikimedia 403s generic agents.
_UA = (
    "FrontierInsight/1.0 "
    "(https://github.com/jyunming/FrontierInsight; research assistant)"
)
_HEADERS = {"User-Agent": _UA}

# Licenses we will embed: redistributable Creative-Commons + public domain.
# Deliberately EXCLUDES NonCommercial (NC) and NoDerivatives (ND) — FI output
# may be used commercially, and we'd rather skip an image than mis-license it.
_FREE_LICENSE_RE = re.compile(
    r"(?:^|\b)("
    r"cc[ \-]?by(?:[ \-]?sa)?(?:[ \-]?\d(?:\.\d)?)?"  # CC BY / CC BY-SA (+ ver)
    r"|cc0"
    r"|public[ \-]?domain"
    r"|pd[ \-]?(?:us|old)?"
    r")(?:\b|$)",
    re.I,
)
_NONFREE_RE = re.compile(r"\b(nc|nd|noncommercial|no[ \-]?deriv)", re.I)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp")
_MAX_IMAGE_BYTES = 12 * 1024 * 1024  # per-figure cap so enrichment stays light

# A redistributable CC license on an arXiv abstract page (the default arXiv
# license is NOT redistributable, so a paper without one of these is skipped).
_ARXIV_CC_RE = re.compile(
    r"creativecommons\.org/"
    r"(?:licenses/(by(?:-sa)?)/([0-9.]+)|(publicdomain/zero)/([0-9.]+))",
    re.I,
)


@dataclass
class WebFigure:
    """A downloaded, license-cleared illustrative figure."""

    local_path: Path
    caption: str
    source_url: str
    license: str
    attribution: str
    kind: str  # "commons" | "oa_paper"


def _is_free_license(text: str) -> bool:
    t = (text or "").strip()
    if not t or _NONFREE_RE.search(t):
        return False
    return bool(_FREE_LICENSE_RE.search(t))


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def _terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 3}


def _relevance(caption: str, query: str) -> int:
    """Cheap caption<->topic overlap for ranking candidate figures."""
    q = _terms(query)
    return len(q & _terms(caption)) if q else 0


def _slug(s: str, n: int = 40) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-") or "figure")[:n]


# ---------------------------------------------------------------------------
# Wikimedia Commons
# ---------------------------------------------------------------------------

def fetch_commons_figures(
    query: str, *, out_dir: Path, max_n: int = 2, timeout_s: float = 20.0,
) -> list[WebFigure]:
    """Search Wikimedia Commons for topic images under a free license and
    download a rasterised thumbnail of each. Returns up to ``max_n``."""
    if not query.strip():
        return []
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": query.strip(),
        "gsrnamespace": "6", "gsrlimit": "12",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|mime", "iiurlwidth": "1100",
    }
    try:
        r = httpx.get(
            "https://commons.wikimedia.org/w/api.php",
            params=params, headers=_HEADERS, timeout=timeout_s, follow_redirects=True,
        )
        if r.status_code != 200:
            return []
        pages = (r.json().get("query") or {}).get("pages") or {}
    except Exception as e:  # noqa: BLE001 — best-effort
        _log.info("commons search failed: %s", e)
        return []

    out: list[WebFigure] = []
    for page in pages.values():
        if len(out) >= max_n:
            break
        ii = (page.get("imageinfo") or [{}])[0]
        if not str(ii.get("mime") or "").startswith("image/"):
            continue  # skip PDFs / video
        em = ii.get("extmetadata") or {}
        lic = (em.get("LicenseShortName") or {}).get("value", "")
        if not _is_free_license(lic):
            continue
        thumb = ii.get("thumburl") or ii.get("url")
        if not thumb:
            continue
        img = _download_image(thumb, timeout_s=timeout_s)
        if img is None:
            continue
        title = str(page.get("title") or "").removeprefix("File:")
        artist = _strip_html((em.get("Artist") or {}).get("value", "")) or "Wikimedia Commons"
        path = out_dir / f"webfig_commons_{_slug(title)}{_ext_of(thumb)}"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img)
        except OSError:
            continue
        out.append(WebFigure(
            local_path=path,
            caption=_strip_html(title.rsplit(".", 1)[0].replace("_", " ")),
            source_url=ii.get("descriptionurl") or thumb,
            license=lic, attribution=artist, kind="commons",
        ))
        _log.info("commons: kept %s (%s)", title[:50], lic)
    return out


# ---------------------------------------------------------------------------
# arXiv (a real figure from a cited CC-licensed paper's HTML rendering)
# ---------------------------------------------------------------------------

def fetch_arxiv_figure(
    arxiv_id: str, query: str, *, out_dir: Path, timeout_s: float = 25.0,
) -> WebFigure | None:
    """Pull the most topic-relevant figure from a CC-licensed arXiv paper's
    HTML rendering, with its caption. None unless the paper carries a
    redistributable CC license (the default arXiv license forbids reuse) or
    no figure is recoverable."""
    aid = re.sub(r"v\d+$", "", (arxiv_id or "").strip())
    if not aid:
        return None
    try:
        with httpx.Client(
            headers=_HEADERS, timeout=timeout_s, follow_redirects=True,
        ) as c:
            ab = c.get(f"https://arxiv.org/abs/{aid}")
            m = _ARXIV_CC_RE.search(ab.text) if ab.status_code == 200 else None
            if not m:
                return None  # default arXiv license is not redistributable
            if m.group(1):
                lic = f"CC {m.group(1).upper()} {m.group(2)}".strip()
            else:
                lic = f"CC0 {m.group(4)}".strip()

            h = c.get(f"https://arxiv.org/html/{aid}")
            if h.status_code != 200 or "<figure" not in h.text:
                return None
            cands: list[tuple[str, str]] = []  # (caption, img_src)
            for fig in re.findall(r"<figure\b.*?</figure>", h.text, re.S):
                src = re.search(
                    r'<img[^>]+src="([^"]+\.(?:png|jpg|jpeg))"', fig, re.I,
                )
                cap = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", fig, re.S)
                if src:
                    cands.append((_strip_html(cap.group(1)) if cap else "",
                                  src.group(1)))
            if not cands:
                return None
            cands.sort(key=lambda c2: _relevance(c2[0], query), reverse=True)
            caption, src = cands[0]
            if src.startswith("http"):
                url = src
            elif src.startswith("/"):
                url = "https://arxiv.org" + src
            else:
                url = f"https://arxiv.org/html/{aid}/{src}"
            img = _download_image(url, timeout_s=timeout_s)
            if img is None:
                return None
            path = out_dir / f"webfig_arxiv_{aid.replace('/', '_')}{_ext_of(url)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(img)
            _log.info("arxiv: kept figure from %s (%s)", aid, lic)
            return WebFigure(
                local_path=path,
                caption=(caption or f"Figure from arXiv:{aid}")[:300],
                source_url=f"https://arxiv.org/abs/{aid}",
                license=lic, attribution=f"arXiv:{aid}", kind="oa_paper",
            )
    except Exception as e:  # noqa: BLE001 — best-effort
        _log.info("arxiv figure %s failed: %s", aid, e)
        return None


def _download_image(url: str, *, timeout_s: float) -> bytes | None:
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=timeout_s, follow_redirects=True)
    except Exception:
        return None
    if r.status_code != 200 or len(r.content) > _MAX_IMAGE_BYTES:
        return None
    ctype = (r.headers.get("content-type") or "").lower()
    if not (ctype.startswith("image/") or r.content[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"GIF8")):
        return None
    return r.content


def _ext_of(url: str) -> str:
    m = re.search(r"\.(jpg|jpeg|png|gif|webp)(?:$|\?)", url, re.I)
    return f".{m.group(1).lower()}" if m else ".png"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect_topic_figures(
    topic: str,
    arxiv_ids: list[str],
    *,
    out_dir: Path,
    max_n: int = 3,
    timeout_s: float = 25.0,
) -> list[WebFigure]:
    """Gather a few license-clean illustrative figures for ``topic``: one
    real figure from the most relevant cited CC-licensed arXiv paper (first
    of ``arxiv_ids`` that yields one), plus Wikimedia Commons diagrams to
    fill up to ``max_n``. Best-effort and order-stable; [] on a total miss."""
    figs: list[WebFigure] = []
    for aid in arxiv_ids:
        if len(figs) >= 1:  # one paper figure is enough
            break
        f = fetch_arxiv_figure(aid, topic, out_dir=out_dir, timeout_s=timeout_s)
        if f is not None:
            figs.append(f)
    remaining = max(0, max_n - len(figs))
    if remaining:
        figs.extend(fetch_commons_figures(
            topic, out_dir=out_dir, max_n=remaining, timeout_s=timeout_s,
        ))
    return figs[:max_n]
