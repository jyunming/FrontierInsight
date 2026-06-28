"""Relevance-ranked passage selection over fetched full text.

A quest's fetched corpus is often many documents of tens of KB each — far
more than fits in any single node's prompt. Rather than truncate each
document to its first N characters (which is usually just the abstract +
page nav), we chunk it and keep the passages most relevant to the quest's
question, so the writer and plotter see the *relevant* content, not the
*first* content. This is the Retrieval -> Contextual-Summarisation idea
from PaperQA2, scoped to a single document at prompt-build time.

Ranking uses a dependency-free lexical term-overlap score by default, and
sentence-transformer embedding cosine when ``mode`` selects it (the model
is loaded lazily and cached for the process, so quests with no long text to
select never pay for it). Either way the chosen passages are returned in
original document order, with the leading chunk (title + abstract) always
kept for orientation, so the reading flow the LLM sees stays coherent.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

_log = logging.getLogger("frontier_insight.passages")

# Separator inserted between non-adjacent kept passages so the LLM can see
# that text was elided between them.
_GAP = "\n\n[...]\n\n"

# Common English function words — dropped from the relevance query so a
# chunk isn't ranked highly just for containing "the" / "of" / "and".
_STOPWORDS = frozenset(
    """a an the of to in on for and or but with without within into from by as
    at is are was were be been being this that these those it its their his her
    our your they we you he she how why what which when where who whom whose can
    could may might will would shall should do does did done has have had not no
    nor than then so such over under between among across per via about against
    during before after above below up down out off only own same other more
    most some any all each both few many much several use using used based study
    studies paper results result effect effects associated association""".split()
)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _content_terms(s: str) -> set[str]:
    return {t for t in _tokens(s) if len(t) > 2 and t not in _STOPWORDS}


def chunk_text(text: str, *, chunk_chars: int = 1200) -> list[str]:
    """Split ``text`` into ~``chunk_chars`` chunks, packing whole paragraphs
    greedily and hard-splitting any paragraph longer than the window.
    Order is preserved."""
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()] or [text]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > chunk_chars:  # hard-split an over-long paragraph
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:chunk_chars])
            p = p[chunk_chars:]
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= chunk_chars:
            cur += "\n\n" + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


_EMBED_MODEL = None
_EMBED_TRIED = False


def _embed_model():
    """Lazily load + cache the sentence-transformer model, or None when
    sentence-transformers isn't installed."""
    global _EMBED_MODEL, _EMBED_TRIED
    if _EMBED_TRIED:
        return _EMBED_MODEL
    _EMBED_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _log.info("passages: loaded all-MiniLM-L6-v2 for semantic ranking")
    except Exception as e:  # noqa: BLE001 — best-effort; lexical fallback
        _log.info("passages: embeddings unavailable (%s); using lexical ranking", e)
        _EMBED_MODEL = None
    return _EMBED_MODEL


def _embed_scores(chunks: list[str], query: str) -> list[float] | None:
    model = _embed_model()
    if model is None:
        return None
    try:
        import numpy as np
        vecs = model.encode([query] + chunks, normalize_embeddings=True)
        q, cs = vecs[0], vecs[1:]
        return [float(np.dot(q, c)) for c in cs]
    except Exception as e:  # noqa: BLE001 — fall back to lexical
        _log.info("passages: embedding scoring failed (%s); lexical fallback", e)
        return None


def _lexical_scores(chunks: list[str], query: str) -> list[float]:
    """Term-overlap score per chunk: distinct query terms present dominate,
    with a length-normalised term-frequency tie-breaker."""
    qterms = _content_terms(query)
    if not qterms:
        return [0.0] * len(chunks)
    scores: list[float] = []
    for ch in chunks:
        toks = _tokens(ch)
        if not toks:
            scores.append(0.0)
            continue
        tf = Counter(t for t in toks if t in qterms)
        distinct = len(tf)
        density = sum(tf.values()) / math.sqrt(len(toks))
        scores.append(distinct + 0.1 * density)
    return scores


def select_relevant_excerpt(
    content: str,
    query: str,
    *,
    budget_chars: int = 4000,
    chunk_chars: int = 1200,
    mode: str = "lexical",
) -> str:
    """Return up to ``budget_chars`` of the passages in ``content`` most
    relevant to ``query``, in original document order (reading flow
    preserved). The leading chunk (title + abstract) is always kept for
    orientation. Falls back to ``content[:budget_chars]`` when there's
    nothing to rank — short content, no query, or a single chunk.

    ``mode``: ``"lexical"`` (default, fast, dependency-free term overlap),
    ``"embed"`` (sentence-transformer cosine), or ``"auto"`` (embed when
    available, else lexical)."""
    content = content or ""
    if not query or not query.strip() or len(content) <= budget_chars:
        return content[:budget_chars]
    chunks = chunk_text(content, chunk_chars=chunk_chars)
    if len(chunks) <= 1:
        return content[:budget_chars]

    scores: list[float] | None = None
    if mode in ("auto", "embed"):
        scores = _embed_scores(chunks, query)
    if scores is None:
        scores = _lexical_scores(chunks, query)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    chosen = {0}  # always keep the head (title + abstract) for orientation
    total = len(chunks[0])
    for i in order:
        if i in chosen:
            continue
        if total + len(chunks[i]) + len(_GAP) > budget_chars:
            continue
        chosen.add(i)
        total += len(chunks[i]) + len(_GAP)
    out = _GAP.join(chunks[i] for i in sorted(chosen))
    return out[:budget_chars]
