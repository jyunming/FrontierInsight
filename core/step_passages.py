"""The literature a function is written from: a few passages for one step card or equation, found without a model call.

``for_card`` asks the knowledge layer for the three best hits for the card's name and what it does (Axon and the pinned
papers: no web search, no source routing, so no model is asked which source to use), keeps only hits that belong to the
card's ``source`` when the card names one, cuts each to the passages that matter with the existing selector
(:func:`core.passages.select_relevant_excerpt`, term overlap, no embedding model) and puts the paper's citation line in
front. The sourced definitions of the metrics the step computes (``protocol.definitions``, each a quote the plan took from
a retrieved source) are added as they are. The result is cached in a JSON file the caller gives, so a resumed quest does not
retrieve again; a failed retrieval is not cached.

No source and no hits give ``""``. Async because retrieval is; makes no model calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

#: Characters of passages in one block, and hits asked for.
BUDGET_CHARS = 1500
TOP_K = 3
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{3,}|(?:19|20)\d{2}|10\.\d{4,9}/\S+")
#: Words in a source line that say nothing about which source it is.
_SOURCE_NOISE = {"from", "with", "that", "this", "paper", "papers", "eqn", "equation", "section", "chapter", "page",
                 "derivation", "first", "principles", "heuristic", "standard", "textbook", "method", "et", "al"}


def _text(value: Any, limit: int = 400) -> str:
    return " ".join(str(value if value is not None else "").split())[:limit]


def _source_terms(source: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(source or "") if t.lower() not in _SOURCE_NOISE]


def belongs_to(source: str, meta: dict[str, Any], content: str = "") -> bool:
    """Whether a retrieved hit is the card's source: a word, a year or a DOI of the source line occurs in the hit's title,
    authors, venue, year, DOI or URL, or in the start of its text. A source line with no such word (``[3]``, ``heuristic``)
    names nothing to filter by, so every hit belongs to it."""
    terms = _source_terms(source)
    if not terms:
        return True
    authors = meta.get("authors")
    hay = " ".join([_text(meta.get("title")), " ".join(map(str, authors)) if isinstance(authors, list) else _text(authors),
                    _text(meta.get("venue")), _text(meta.get("year")), _text(meta.get("doi")), _text(meta.get("url")),
                    _text(content, 300)]).lower()
    return any(t in hay for t in terms)


def _header(meta: dict[str, Any]) -> str:
    try:
        from .knowledge import _paper_header_line

        return _paper_header_line(meta)
    except Exception:  # noqa: BLE001 -- a header that cannot be built is a title line instead
        return f"[{_text(meta.get('title')) or '(untitled)'}]"


def definition_lines(definitions: Any, card: dict[str, Any], metrics: list[str] | None = None) -> list[str]:
    """The sourced definitions that apply to the card: those of ``metrics`` when given, else of every metric the card's
    name, purpose or output names. One line each: the metric, the source, and the quote word for word."""
    text = " ".join(_text(card.get(k), 600) for k in ("name", "does", "output", "formula")).lower()
    wanted = {m.lower() for m in metrics or []}
    out: list[str] = []
    for d in definitions if isinstance(definitions, list) else []:
        if not isinstance(d, dict):
            continue
        metric, quote = _text(d.get("metric"), 80), _text(d.get("quote"), 600)
        if not metric or not quote:
            continue
        hit = metric.lower() in wanted if metrics is not None else bool(
            re.search(rf"(?<![A-Za-z0-9_]){re.escape(metric.lower())}(?![A-Za-z0-9_])", text))
        if hit:
            src = _text(d.get("source"), 80)
            out.append(f"Definition of {metric}" + (f" ({src})" if src else "") + f': "{quote}"')
    return out


def _query(card: dict[str, Any]) -> str:
    return " ".join(p for p in (_text(card.get("name"), 100), _text(card.get("does") or card.get("formula"), 300),
                                _text(card.get("source"), 200)) if p)


def _signature(card: dict[str, Any], defs: list[str]) -> str:
    blob = json.dumps([_query(card), defs, BUDGET_CHARS, TOP_K], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(path: Path, data: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    except OSError:
        pass


async def for_card(card: dict[str, Any], knowledge: Any, cache_path: Path | str | None, *, definitions: Any = None,
                   metrics: list[str] | None = None) -> str:
    """The literature block for one step card (or equation: ``id``, ``name`` / ``formula``, ``source``). ``knowledge`` is
    the quest's :class:`core.knowledge.Knowledge` (or ``None``); ``cache_path`` the JSON file the block is kept in."""
    cid = _text(card.get("id"), 40)
    defs = definition_lines(definitions, card, metrics)
    sig = _signature(card, defs)
    cache = _load(Path(cache_path)) if cache_path else {}
    held = cache.get(cid)
    if cid and isinstance(held, dict) and held.get("sig") == sig and isinstance(held.get("text"), str):
        return held["text"]
    source = _text(card.get("source"), 300)
    blocks: list[str] = []
    searched = True
    if knowledge is not None and source and _text(card.get("name") or card.get("does") or card.get("formula")):
        try:
            hits = await knowledge.asearch(_query(card), top_k=TOP_K, sources=[], web=False)
        except Exception:  # noqa: BLE001 -- retrieval that failed leaves the block to the definitions, and is asked again later
            hits, searched = [], False
        hits = [h for h in hits or [] if belongs_to(source, dict(getattr(h, "metadata", None) or {}),
                                                    str(getattr(h, "content", "") or ""))][:TOP_K]
        if hits:
            from .passages import select_relevant_excerpt

            room = max(200, BUDGET_CHARS // len(hits))
            for h in hits:
                meta = dict(getattr(h, "metadata", None) or {})
                body = select_relevant_excerpt(str(getattr(h, "content", "") or ""), _query(card), budget_chars=room,
                                               mode="lexical").strip()
                if body:
                    blocks.append(f"{_header(meta)}\n{body}")
    text = "\n\n".join([*blocks, *defs]).strip()
    if cid and cache_path and searched:
        cache[cid] = {"sig": sig, "text": text}
        _save(Path(cache_path), cache)
    return text
