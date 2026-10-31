"""Does the prior work already answer the study's question? (disclosure only)

After the literature step, one model call judges whether the retrieved sources already answer the question the study
asks, and what the study adds. The answer is *disclosed*: it is written to ``.fi/literature_novelty.json``, the plan
prompt gets one line of it, and the writer's introduction guidance gets the suggested framing. It never stops the
quest, never sends it back to ideate and never changes the design.

FI checks the answer against its own record of the sources: an evidence item whose source is not in the list the model
was given, or whose quoted sentence is not in that source's text, is dropped (and counted). The model's words are never
taken as the evidence; the source's text is.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

RECORD = Path(".fi") / "literature_novelty.json"
VERDICTS = ("yes", "partly", "no")
#: A quote shorter than this would match almost any text, so it cannot show that a source says something.
MIN_QUOTE_CHARS = 20
#: Characters of each source's text the model reads (the leading part, as the prototype that was measured did).
EXCERPT_CHARS = 900

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-", "‑": "-",
                         "‒": "-", "–": "-", "—": "-", "−": "-"})


def normalise(text: Any) -> str:
    """Text as compared: HTML entities decoded, quotes and dashes made plain, whitespace collapsed, case ignored."""
    s = html.unescape(str(text or ""))
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _label(value: Any) -> str:
    """A source id as the model may write it: ``[3]``, ``3``, ``W2``."""
    return str(value or "").strip().strip("[]() ").upper()


def format_sources(sources: list[dict[str, Any]]) -> str:
    """The listed sources as the model reads them: ``[id] (year) title`` and the leading part of the text."""
    out = []
    for s in sources:
        year = f"({s['year']}) " if s.get("year") else ""
        out.append(f"[{s['id']}] {year}{s.get('title') or '(no title)'}\n    {s.get('excerpt', '')}")
    return "\n".join(out)


def check(reply: Any, sources: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """FI's check of the model's answer against the sources it was given. ``sources`` are dicts with ``id``, ``title``
    and ``text`` (the whole text). Returns ``(record, counts)``: the record is None when the answer has no usable
    verdict; ``counts`` says how many evidence items were given, kept, and dropped for an unknown source or a quote
    that is not in that source's text."""
    counts = {"given": 0, "kept": 0, "dropped_unknown_source": 0, "dropped_quote_not_found": 0}
    if not isinstance(reply, dict):
        return None, counts
    verdict = str(reply.get("already_answered") or "").strip().lower()
    if verdict not in VERDICTS:
        return None, counts
    by_id = {_label(s["id"]): s for s in sources}
    normalised: dict[str, str] = {}
    kept: list[dict[str, str]] = []
    raw = reply.get("evidence")
    for item in raw if isinstance(raw, list) else []:
        counts["given"] += 1
        if not isinstance(item, dict):
            counts["dropped_unknown_source"] += 1
            continue
        src = by_id.get(_label(item.get("source", item.get("id"))))
        if src is None:
            counts["dropped_unknown_source"] += 1
            continue
        quote = str(item.get("quote", item.get("sentence")) or "").strip()
        q = normalise(quote)
        if len(q) < MIN_QUOTE_CHARS:
            counts["dropped_quote_not_found"] += 1
            continue
        text = normalised.setdefault(src["id"], normalise(src.get("text")))
        if q not in text:
            counts["dropped_quote_not_found"] += 1
            continue
        kept.append({"source": src["id"], "title": src.get("title") or "", "quote": quote})
    counts["kept"] = len(kept)
    return {
        "already_answered": verdict,
        "evidence": kept,
        "what_this_adds": str(reply.get("what_this_adds") or "").strip(),
        "suggested_framing": str(reply.get("suggested_framing") or "").strip(),
        # A "yes" or "partly" that no checked passage backs is kept in the record but not repeated to the plan or the
        # writer: FI does not pass on a claim about the sources that it could not find in them.
        "backed": verdict == "no" or bool(kept),
        "counts": counts,
    }, counts


def _ids(record: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for e in record.get("evidence") or []:
        if e["source"] not in seen:
            seen.append(e["source"])
    return seen


def design_line(record: dict[str, Any] | None) -> str:
    """The one short line the plan prompt gets: what the literature already answers and what this study adds."""
    if not record or not record.get("backed"):
        return ""
    adds = record.get("what_this_adds") or ""
    tail = f" This study adds: {adds}" if adds else ""
    verdict = record["already_answered"]
    if verdict == "yes":
        head = "The sources retrieved for this study already answer its question."
    elif verdict == "partly":
        head = "The sources retrieved for this study answer part of its question."
    else:
        head = "None of the sources retrieved for this study answers its question."
    return f"{head}{tail}"


def design_block(record: dict[str, Any] | None) -> str:
    line = design_line(record)
    return (f"## What the prior work already answers\n{line} Plan the study so that what it adds is measured; "
            "do not plan work whose answer the sources already give.\n") if line else ""


def write_block(record: dict[str, Any] | None) -> str:
    """The introduction guidance the writer gets: the framing, and the only sources it may cite as having answered."""
    if not record or not record.get("backed"):
        return ""
    ids = _ids(record)
    framing = record.get("suggested_framing") or ""
    lines = ["## How the introduction should frame the study"]
    if framing:
        lines.append(f"Frame the Introduction like this: {framing}")
    lines.append(design_line(record))
    if ids:
        lines.append("Cite a source as having already answered part of the question only if it is one of these "
                     "(FI checked that each says it): " + ", ".join(f"[{i}]" for i in ids) + ".")
    else:
        lines.append("Do not cite any source as having answered the question.")
    return "\n".join(x for x in lines if x) + "\n"


def read_record(quest_root: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_record(quest_root: Path, record: dict[str, Any], *, call_id: str | None = None,
                 iteration: int | None = None, model: str | None = None) -> None:
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record, "call_id": call_id, "iteration": iteration, "model": model},
                               indent=2, ensure_ascii=False), encoding="utf-8")
