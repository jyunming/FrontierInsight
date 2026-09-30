"""Whether a retrieved source has been retracted, looked up in Crossref.

Crossref holds the Retraction Watch database and publishers' own notices as ``updated-by`` entries on the retracted
work (``{"type": "retraction", "DOI": <the notice>, "source": "retraction-watch", "updated": {...}}``). After the
literature search de-duplicates what it found, every DOI not yet looked up is asked about in one request per
:data:`BATCH_SIZE` DOIs (``/works?filter=doi:A,doi:B&select=DOI,updated-by``), one request at a time: Crossref's public
pool allows one request at a time and five a second. No email goes with the request (FI sends none to a search
service).

Each source's ``metadata["retraction"]`` becomes one of:

- ``retracted``: an ``updated-by`` notice of a type in :data:`RETRACTED_TYPES`; ``metadata["retraction_note"]`` says
  which notice and when.
- ``not_retracted``: Crossref answered, holds the DOI, and lists no such notice.
- ``not_checked``: no answer (a network error, a timeout, an error status, an answer that could not be read), or
  Crossref does not hold the DOI (an arXiv DOI is not sent at all: Crossref holds no arXiv record). Never read as "not
  retracted"; the next literature pass asks again.
- ``no_doi``: nothing to look up.

A retracted source is marked ``[retracted]`` in every prior-work block (``core.engine._format_lit_header``) and in the
claim check's source list, and :func:`apply_to_claims` makes a claim the check grounded in one ``unsupported``. The rows
go to ``.fi/literature_queries.json`` (stage ``retractions``), the sealed record of the search, and :func:`summary_line`
is the one line ``run.log`` gets. Nothing here raises: a lookup that fails leaves the sources ``not_checked`` and the
quest goes on.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx

CROSSREF_WORKS = "https://api.crossref.org/works"
RETRACTED = "retracted"
NOT_RETRACTED = "not_retracted"
NOT_CHECKED = "not_checked"
NO_DOI = "no_doi"
#: Crossref update types that withdraw the work itself. A correction, an expression of concern or a partial retraction
#: is kept in the record's notices but does not mark the source.
RETRACTED_TYPES = frozenset({"retraction", "withdrawal", "removal"})
#: DOIs per request: well inside a URL's length, and one request for a usual quest's sources.
BATCH_SIZE = 40
_TIMEOUT_S = 15.0
#: Between two requests, so the public pool's five a second is never reached.
_GAP_S = 0.25
#: One more try after a "too many requests" answer, after at most this long.
_RETRY_WAIT_S = 2.0
_MAX_RETRY_WAIT_S = 10.0
_USER_AGENT = "FrontierInsight/1.0"
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
_ARXIV_PREFIX = "10.48550/"


async def _sleep(seconds: float) -> None:
    """The wait between requests (tests replace it)."""
    await asyncio.sleep(seconds)


def normalize_doi(value: Any) -> str:
    """A DOI in Crossref's form (lower case, no resolver prefix), or "" when ``value`` is not a DOI. A DOI with a comma
    is not one either: the comma separates the filter's DOIs."""
    doi = str(value or "").strip()
    doi = re.sub(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi).strip().lower()
    return doi if _DOI_RE.match(doi) and "," not in doi else ""


def _notice(update: dict[str, Any]) -> dict[str, str]:
    updated = update.get("updated") or {}
    date = str(updated.get("date-time") or "")[:10]
    if not date:
        parts = (updated.get("date-parts") or [[]])[0] or []
        date = "-".join(f"{int(p):02d}" if i else str(int(p)) for i, p in enumerate(parts) if str(p).isdigit())
    return {"type": str(update.get("type") or "").lower(), "label": str(update.get("label") or ""),
            "doi": str(update.get("DOI") or ""), "date": date, "source": str(update.get("source") or "")}


def _verdict(item: dict[str, Any]) -> dict[str, Any]:
    notices = [_notice(u) for u in (item.get("updated-by") or []) if isinstance(u, dict)]
    withdrawn = [n for n in notices if n["type"] in RETRACTED_TYPES]
    if not withdrawn:
        return {"status": NOT_RETRACTED, "why": "Crossref lists no retraction", "notices": notices}
    n = withdrawn[-1]
    why = f"{n['label'] or n['type']}" + (f" on {n['date']}" if n["date"] else "")
    why += (f", notice {n['doi']}" if n["doi"] else "") + (f" ({n['source']})" if n["source"] else "")
    return {"status": RETRACTED, "why": why, "notices": notices}


def _not_checked(why: str) -> dict[str, Any]:
    return {"status": NOT_CHECKED, "why": why, "notices": []}


async def _ask(client: httpx.AsyncClient, batch: list[str]) -> tuple[dict[str, dict[str, Any]] | None, str]:
    """Crossref's records for ``batch`` by DOI, or ``None`` and why there is no answer."""
    params = {"filter": ",".join(f"doi:{d}" for d in batch), "rows": str(len(batch) * 2),
              "select": "DOI,updated-by"}
    for attempt in (1, 2):
        try:
            r = await client.get(CROSSREF_WORKS, params=params)
        except httpx.TimeoutException:
            return None, "Crossref did not answer in time"
        except Exception as e:  # noqa: BLE001 -- a lookup never stops a quest
            return None, f"Crossref could not be reached ({type(e).__name__})"
        if r.status_code == 429 and attempt == 1:
            try:
                wait = float(r.headers.get("retry-after") or _RETRY_WAIT_S)
            except ValueError:
                wait = _RETRY_WAIT_S
            if wait <= _MAX_RETRY_WAIT_S:
                await _sleep(max(wait, 0.5))
                continue
        if r.status_code != 200:
            return None, f"Crossref answered with status {r.status_code}"
        try:
            items = r.json()["message"]["items"]
            if not isinstance(items, list):
                raise TypeError("no list of items")
        except Exception:  # noqa: BLE001
            return None, "Crossref's answer could not be read"
        return {normalize_doi(i.get("DOI")): i for i in items if isinstance(i, dict)}, ""
    return None, "Crossref asked for fewer requests"


async def check_dois(
    dois: list[str], *, transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, dict[str, Any]]:
    """``{doi: {"status", "why", "notices"}}`` for every DOI in ``dois`` (normalized, see :func:`normalize_doi`; a value
    that is not a DOI is left out). Never raises: whatever was not answered is ``not_checked``."""
    wanted = list(dict.fromkeys(d for d in (normalize_doi(x) for x in dois) if d))
    out: dict[str, dict[str, Any]] = {}
    ask = []
    for doi in wanted:
        if doi.startswith(_ARXIV_PREFIX):
            out[doi] = _not_checked("an arXiv preprint: Crossref holds no arXiv record")
        else:
            ask.append(doi)
    if not ask:
        return out
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, transport=transport, follow_redirects=True,
                                     headers={"User-Agent": _USER_AGENT}) as client:
            for start in range(0, len(ask), BATCH_SIZE):
                batch = ask[start:start + BATCH_SIZE]
                if start:
                    await _sleep(_GAP_S)
                found, why = await _ask(client, batch)
                for doi in batch:
                    if found is None:
                        out[doi] = _not_checked(why)
                    elif doi in found:
                        out[doi] = _verdict(found[doi])
                    else:
                        out[doi] = _not_checked("Crossref has no record of this DOI")
    except Exception as e:  # noqa: BLE001 -- a lookup never stops a quest
        for doi in ask:
            out.setdefault(doi, _not_checked(f"the lookup failed ({type(e).__name__})"))
    return out


def _parts(entry: Any) -> tuple[dict[str, Any], bool]:
    """An entry's metadata, and whether it is a literature entry at all (a dict, or a RetrievedDoc)."""
    if isinstance(entry, dict):
        return dict(entry.get("metadata") or {}), True
    meta = getattr(entry, "metadata", None)
    return (dict(meta), True) if isinstance(meta, dict) else ({}, False)


def _with_meta(entry: Any, meta: dict[str, Any]) -> Any:
    if isinstance(entry, dict):
        return {**entry, "metadata": meta}
    try:
        return type(entry)(content=getattr(entry, "content", ""), metadata=meta)
    except Exception:  # noqa: BLE001
        return entry


def _source_key(meta: dict[str, Any], doi: str) -> str:
    """The source's identity as the other verdict records in ``literature_queries.json`` name it."""
    if doi:
        return f"doi:{doi}"
    for k in ("url", "source_url", "title"):
        v = str(meta.get(k) or "").strip().lower()
        if v:
            return f"{k}:{v}"
    return ""


async def check_literature(
    entries: list[Any], *, transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """``entries`` with ``metadata["retraction"]`` set on each one not yet looked up (or not answered last time), and
    one record row per source looked up now. New entries replace the changed ones: the caller's (a checkpoint's
    literature) are never changed in place. Never raises."""
    try:
        pending: list[tuple[int, dict[str, Any], str]] = []
        for i, entry in enumerate(entries):
            meta, ok = _parts(entry)
            if not ok or meta.get("retraction") not in (None, NOT_CHECKED):
                continue
            pending.append((i, meta, normalize_doi(meta.get("doi"))))
        if not pending:
            return list(entries), []
        answers = await check_dois([doi for _i, _m, doi in pending if doi], transport=transport)
        out = list(entries)
        rows: list[dict[str, Any]] = []
        for i, meta, doi in pending:
            answer = answers.get(doi) if doi else {"status": NO_DOI, "why": "no DOI to look up", "notices": []}
            answer = answer or _not_checked("the lookup gave no answer")
            meta["retraction"] = answer["status"]
            if answer["status"] == RETRACTED:
                meta["retraction_note"] = answer["why"]
            else:
                meta.pop("retraction_note", None)
            out[i] = _with_meta(entries[i], meta)
            rows.append({"source": _source_key(meta, doi),
                         "title": " ".join(str(meta.get("title") or meta.get("url") or "").split())[:200],
                         "doi": doi, "status": answer["status"], "why": answer["why"],
                         "notices": answer.get("notices") or []})
        return out, rows
    except Exception:  # noqa: BLE001 -- a lookup never stops a quest
        return list(entries), []


def summary_line(rows: list[dict[str, Any]]) -> str:
    """The one plain line ``run.log`` gets for a lookup, "" when nothing was looked up."""
    if not rows:
        return ""
    by = {s: [r for r in rows if r.get("status") == s] for s in (RETRACTED, NOT_RETRACTED, NOT_CHECKED, NO_DOI)}
    looked = len(rows) - len(by[NO_DOI])
    n_none = len(by[NO_DOI])
    tail = f"; {n_none} without a DOI {'was' if n_none == 1 else 'were'} not looked up" if n_none else ""
    if not looked:
        return f"No source had a DOI to check for retractions ({len(rows)} source(s))"
    parts: list[str] = []
    if by[RETRACTED]:
        titles = "; ".join(str(r.get("title") or "?")[:80] for r in by[RETRACTED][:3])
        more = f"; and {len(by[RETRACTED]) - 3} more" if len(by[RETRACTED]) > 3 else ""
        parts.append(f"{len(by[RETRACTED])} retracted ({titles}{more})")
    if by[NOT_CHECKED]:
        parts.append(f"{len(by[NOT_CHECKED])} could not be checked")
    if by[NOT_RETRACTED]:
        parts.append(f"{len(by[NOT_RETRACTED])} not retracted" if parts else "none retracted")
    return f"Checked {looked} source{'s' if looked != 1 else ''} for retractions: {', '.join(parts)}{tail}"


def is_retracted(meta: dict[str, Any] | None) -> bool:
    return bool(meta) and (meta or {}).get("retraction") == RETRACTED


def apply_to_claims(
    claims: list[dict[str, Any]], sources: dict[str, tuple[dict[str, Any], str]],
) -> tuple[list[dict[str, Any]], int]:
    """``claims`` with every claim the check grounded in a retracted source (``basis`` ``citation``) made
    ``unsupported``, and how many were changed. The caller's claims are not changed in place."""
    retracted = {label: meta for label, (meta, _text) in sources.items() if is_retracted(meta)}
    if not retracted:
        return claims, 0
    out = [dict(c) for c in claims]
    changed = 0
    for claim in out:
        label = str(claim.get("citation_index"))
        if claim.get("basis") != "citation" or label not in retracted:
            continue
        note = str(retracted[label].get("retraction_note") or "").strip()
        claim.update(basis="unsupported", quote="",
                     evidence=f"[{label}] has been retracted" + (f" ({note})" if note else "")
                     + ", so it cannot support a claim")
        changed += 1
    return out, changed


def retracted_in_record(fi_dir: Path) -> list[str]:
    """The titles of the sources ``.fi/literature_queries.json`` records as retracted, over every literature pass."""
    try:
        data = json.loads((Path(fi_dir) / "literature_queries.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    seen: dict[str, str] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or entry.get("stage") != "retractions":
            continue
        for row in entry.get("sources") or []:
            if isinstance(row, dict) and row.get("status") == RETRACTED:
                key = str(row.get("source") or row.get("title") or "")
                seen.setdefault(key, str(row.get("title") or row.get("doi") or "a source"))
    return list(seen.values())
