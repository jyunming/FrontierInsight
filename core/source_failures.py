"""Per-quest record of retrieval failures, and where it gets reported.

Why this exists
===============
Every literature and full-text source FI talks to is best-effort: an adapter
that fails returns ``[]`` or ``None`` and the quest carries on. That is the
right behaviour and it had one bad consequence. The failure logged at INFO on
``frontier_insight.knowledge``, a logger with no handler, whose effective
level was the root's WARNING — so it went nowhere. A real quest lost every
arXiv and OpenAlex result (HTTP 429 and 400) and its ``run.log`` said nothing
at all; the only symptom was a bibliography made of whatever was left.

So a failure is now recorded against the quest that caused it, written to that
quest's ``run.log`` as it happens, and summarised per source when the run ends
(``.fi/source_failures.json`` plus one ``[source-failures]`` line).

How the quest is known
======================
Adapters are plain functions running in worker threads with no engine handle.
``Engine.run`` sets :data:`current_quest`; ``asyncio.to_thread`` and task
creation copy context, so every adapter call made on the quest's behalf sees
the id — including under ``--fleet``, where several quests share a process.
Outside a quest (tests, scripts) nothing is ledgered and the line goes to the
module logger instead.

What counts as a failure
========================
Rate limits, other 4xx/5xx answers, timeouts, network errors and unparseable
bodies. A 404/410 is not: "this paper is not in PMC" is an answer, and
counting every miss of the open-access cascade would bury the real outages.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_log = logging.getLogger("frontier_insight.sources")

current_quest: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "fi_current_quest", default=None,
)

_LOCK = threading.Lock()
_COUNTS: dict[str, Counter] = {}
_EXAMPLES: dict[str, list[dict[str, Any]]] = {}
_MAX_EXAMPLES = 50

# Credentials ride in query strings (OpenAlex ``api_key``, Unpaywall
# ``email``) and httpx puts the full URL in its exception text.
_SECRET_RE = re.compile(
    r"(?i)\b((?:api[_-]?key|key|token|access_token|email|mailto)=)[^&\s'\"<>]+",
)
_NOT_FAILURES = frozenset({404, 410})


def redact(text: Any) -> str:
    """Mask credential-bearing query parameters in ``text``."""
    return _SECRET_RE.sub(r"\1***", str(text or ""))


def _host(url: str) -> str:
    try:
        return urlsplit(str(url or "")).netloc.lower()
    except ValueError:
        return ""


def source_for_url(url: str, default: str) -> str:
    """``"arxiv"`` for any arXiv host, else ``default``. Full-text routes
    fetch whatever URL an index handed them, and an arXiv copy found through
    Unpaywall is still arXiv traffic."""
    host = _host(url)
    if host == "arxiv.org" or host.endswith(".arxiv.org"):
        return "arxiv"
    return default


def classify_status(status: int) -> str:
    if status == 429:
        return "http_429"
    if 400 <= status < 500:
        return "http_4xx"
    if status >= 500:
        return "http_5xx"
    return "http_other"


def classify_exception(exc: BaseException) -> tuple[str, int | None]:
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if isinstance(status, int):
            return classify_status(status), status
        return "http_other", None
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", None
    if isinstance(exc, httpx.TransportError):
        return "network", None
    if isinstance(exc, ValueError):  # json.JSONDecodeError included
        return "parse", None
    return "error", None


def record_failure(
    source: str, kind: str, *, status: int | None = None, url: str = "",
    detail: Any = "", count: int = 1,
) -> None:
    """Record ``count`` failures of ``kind`` against ``source``. Never raises."""
    try:
        host = _host(url)
        detail_s = redact(detail)[:200]
        line = f"[source-failure] source={source} kind={kind}"
        if status is not None:
            line += f" status={status}"
        if host:
            line += f" host={host}"
        if count > 1:
            line += f" count={count}"
        if detail_s:
            line += f" detail={detail_s}"
        qid = current_quest.get()
        if qid is None:
            _log.info(line)
            return
        with _LOCK:
            _COUNTS.setdefault(qid, Counter())[(source, kind)] += max(1, count)
            examples = _EXAMPLES.setdefault(qid, [])
            if len(examples) < _MAX_EXAMPLES:
                examples.append({
                    "source": source, "kind": kind, "status": status,
                    "host": host, "detail": detail_s, "count": max(1, count),
                    "at": round(time.time(), 3),
                })
        logging.getLogger(f"frontier_insight.{qid}").info(line)
    except Exception:  # noqa: BLE001 - reporting must never break retrieval
        pass


def record_exception(source: str, exc: BaseException, *, url: str = "") -> None:
    kind, status = classify_exception(exc)
    record_failure(source, kind, status=status, url=url, detail=exc)


def record_response(source: str, response: Any, *, url: str = "") -> None:
    """Record a non-success HTTP answer. Successes, 404/410 and responses
    without an integer status (test doubles) are ignored."""
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or status < 400 or status in _NOT_FAILURES:
        return
    record_failure(
        source, classify_status(status), status=status,
        url=url or str(getattr(response, "url", "") or ""),
    )


def reset(quest_id: str) -> None:
    with _LOCK:
        _COUNTS.pop(quest_id, None)
        _EXAMPLES.pop(quest_id, None)


def snapshot(quest_id: str) -> dict[str, Any]:
    with _LOCK:
        counts = Counter(_COUNTS.get(quest_id) or {})
        examples = list(_EXAMPLES.get(quest_id) or [])
    by_source: dict[str, dict[str, int]] = {}
    for (source, kind), n in counts.items():
        by_source.setdefault(source, {})[kind] = n
    ordered = dict(sorted(
        by_source.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]),
    ))
    return {"total": sum(counts.values()), "by_source": ordered, "examples": examples}


def format_summary(snap: dict[str, Any]) -> str:
    """``openalex 3 (http_429=3); web_page 2 (http_4xx=1, timeout=1)``."""
    if not snap.get("total"):
        return "none"
    parts = []
    for source, kinds in snap["by_source"].items():
        detail = ", ".join(
            f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        parts.append(f"{source} {sum(kinds.values())} ({detail})")
    return "; ".join(parts)


def write_summary(quest_id: str, path: Path, *, started_at: float) -> dict[str, Any]:
    snap = snapshot(quest_id)
    payload = {
        "run_started_at": round(started_at, 3),
        "run_finished_at": round(time.time(), 3),
        "summary": format_summary(snap),
        **snap,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
