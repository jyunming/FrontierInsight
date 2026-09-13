"""One queue for every connection FI makes to arXiv, with a shared backoff and a disk cache.

Why a queue
===========
arXiv asks automated clients for at most one request every three seconds over
a single connection, and it enforces that by answering HTTP 429. FI used to
ignore both: the full-text cascade, web-page enrichment and figure fetching
opened up to twenty arxiv.org connections at once, retried a different arXiv
URL the moment one was refused, and cached nothing — so one quest could spend
dozens of requests re-fetching the same pages, and a rate limit met by the
first fetch was answered with more fetches.

Every arXiv request now goes through :func:`request` (or :func:`acquire_slot`
for callers that don't use httpx, such as the headless renderer):

* **Pacing.** One connection at a time, at least :data:`MIN_INTERVAL_S` apart.
  The state lives in a lock file under the cache directory, so it holds across
  processes too — the web UI and VSCode run each quest in its own process, and
  an in-process lock alone would let four quests hit arXiv at four times the
  allowed rate.
* **Backoff.** A 429 pauses the whole queue for 1, then 2, then 4 minutes. If
  arXiv is still refusing after the third wait, arXiv is paused for the rest of
  the quest that hit it; other sources carry on. A success resets the count.
* **Deadlines.** A caller working to a time budget (full-text enrichment has
  90 s) sets :data:`fetch_deadline`. A request that could not start before it
  returns ``None`` at once instead of sleeping inside a worker thread, so a
  multi-minute backoff never starves the thread pool other sources share.
* **Cache.** Successful arXiv responses are kept on disk for 24 hours
  (``FI_CACHE_DIR``, default ``~/.frontier-insight/cache``; ``FI_ARXIV_CACHE=0``
  disables it). arXiv's own guidance is to cache, and a resumed or re-run quest
  otherwise downloads the same papers again.

This module holds process-wide state on purpose — the second such exception
after ``ProxySupervisor`` — because a rate limit belongs to the machine's IP,
not to any one engine.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from . import source_failures as _sf

_log = logging.getLogger("frontier_insight.arxiv_gate")

MIN_INTERVAL_S = 3.0
BACKOFF_S: tuple[float, ...] = (60.0, 120.0, 240.0)
# A 429 older than this no longer counts towards the backoff sequence.
BACKOFF_DECAY_S = 3600.0
CACHE_TTL_S = 24 * 3600.0
MAX_ENTRY_BYTES = 32 * 1024 * 1024
MAX_CACHE_BYTES = 512 * 1024 * 1024

# time.monotonic() value by which the current caller must be finished.
fetch_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "fi_arxiv_deadline", default=None,
)

# Indirection so tests can drive a fake clock.
_now: Callable[[], float] = time.time
_mono: Callable[[], float] = time.monotonic
_sleep: Callable[[float], None] = time.sleep

_THREAD_LOCK = threading.Lock()
_PAUSED_QUESTS: set[str] = set()
_NO_QUEST = "__no_quest__"
_CHALLENGE_MARKERS = (
    b"just a moment", b"checking your browser", b"captcha", b"rate exceeded",
)


def is_arxiv_url(url: Any) -> bool:
    try:
        host = urlsplit(str(url or "")).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    return host == "arxiv.org" or host.endswith(".arxiv.org")


_VERSIONED_PATH_RE = re.compile(r"^(/(?:abs|pdf|html)/[^/?#]+?)v\d+(?=$|[/?#]|\.pdf$)")


def normalize_url(url: str) -> str:
    """Cache identity of an arXiv URL: https, no www., no trailing slash, no
    version suffix, no ``.pdf`` extension. ``/pdf/2401.01234v2.pdf`` and
    ``/pdf/2401.01234`` are the same document for FI's purposes."""
    parts = urlsplit(str(url))
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = (parts.path or "/").rstrip("/") or "/"
    path = _VERSIONED_PATH_RE.sub(r"\1", path)
    if path.startswith("/pdf/") and path.endswith(".pdf"):
        path = path[:-4]
    query = f"?{parts.query}" if parts.query else ""
    return f"https://{host}{path}{query}"


# --- state shared across processes -----------------------------------------

def _cache_root() -> Path:
    base = os.environ.get("FI_CACHE_DIR", "").strip()
    return (Path(base) if base else Path.home() / ".frontier-insight" / "cache") / "arxiv"


def _state_path() -> Path:
    return _cache_root() / "gate_state.json"


def _read_state() -> dict[str, float]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "last_request": float(data.get("last_request", 0.0)),
                "blocked_until": float(data.get("blocked_until", 0.0)),
                "consecutive_429": float(data.get("consecutive_429", 0.0)),
                "last_429": float(data.get("last_429", 0.0)),
            }
    except (OSError, ValueError, TypeError):
        pass
    return {"last_request": 0.0, "blocked_until": 0.0, "consecutive_429": 0.0, "last_429": 0.0}


def _write_state(state: dict[str, float]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        _log.info("arxiv gate state write failed: %s", e)


def _lock(name: str) -> FileLock:
    root = _cache_root()
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / name))


def _quest_key() -> str:
    return _sf.current_quest.get() or _NO_QUEST


def reset_quest(quest_id: str) -> None:
    """Un-pause arXiv for ``quest_id`` (called when a quest run starts)."""
    _PAUSED_QUESTS.discard(quest_id)


def is_paused() -> bool:
    return _quest_key() in _PAUSED_QUESTS


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - _mono()


# --- the queue ----------------------------------------------------------------

def acquire_slot(url: str) -> Callable[[], None] | None:
    """Wait for the arXiv connection. Returns a ``release`` callable to call
    once the request has finished, or ``None`` when arXiv is paused for this
    quest or the caller's deadline would pass before the request could start.
    Must be called from a worker thread, never on an event loop."""
    if is_paused():
        _sf.record_failure("arxiv", "paused_skip", url=url)
        return None
    deadline = fetch_deadline.get()
    while True:
        state = _read_state()
        now = _now()
        wait = max(
            state["blocked_until"] - now,
            state["last_request"] + MIN_INTERVAL_S - now,
            0.0,
        )
        remaining = _remaining(deadline)
        if remaining is not None and wait > remaining:
            _sf.record_failure(
                "arxiv", "throttled_skip", url=url,
                detail=f"next arXiv slot in {wait:.0f}s, budget left {max(remaining, 0):.0f}s",
            )
            return None
        if wait > 0:
            _sleep(min(wait, 5.0))
            continue
        remaining = _remaining(deadline)
        timeout = -1 if remaining is None else max(0.0, remaining)
        if not _THREAD_LOCK.acquire(timeout=timeout):
            _sf.record_failure("arxiv", "throttled_skip", url=url, detail="queue wait exceeded budget")
            return None
        conn = _lock("connection.lock")
        remaining = _remaining(deadline)
        try:
            conn.acquire(timeout=-1 if remaining is None else max(0.0, remaining))
        except FileLockTimeout:
            _THREAD_LOCK.release()
            _sf.record_failure("arxiv", "throttled_skip", url=url, detail="queue wait exceeded budget")
            return None
        state = _read_state()
        now = _now()
        if max(state["blocked_until"], state["last_request"] + MIN_INTERVAL_S) > now:
            conn.release()
            _THREAD_LOCK.release()
            continue
        state["last_request"] = now
        _write_state(state)

        def release() -> None:
            try:
                conn.release()
            finally:
                _THREAD_LOCK.release()

        return release


def report(url: str, status: Any) -> bool:
    """Feed a response status back to the queue. Returns True when the
    request met a rate limit that the queue will wait out (the caller may
    retry once the wait is over)."""
    if not isinstance(status, int):
        return False
    with _lock("state.lock"):
        state = _read_state()
        now = _now()
        if status == 429:
            if now - state["last_429"] > BACKOFF_DECAY_S:
                state["consecutive_429"] = 0
            state["consecutive_429"] += 1
            state["last_429"] = now
            n = int(state["consecutive_429"])
            if n <= len(BACKOFF_S):
                state["blocked_until"] = now + BACKOFF_S[n - 1]
                _write_state(state)
                _log.info("arXiv rate limit (%d): pausing the queue for %.0fs", n, BACKOFF_S[n - 1])
                return True
            _write_state(state)
            _PAUSED_QUESTS.add(_quest_key())
            _sf.record_failure(
                "arxiv", "arxiv_paused", url=url,
                detail="still rate-limited after waiting "
                + ", ".join(f"{int(s // 60)} min" for s in BACKOFF_S)
                + "; no more arXiv requests this quest",
            )
            return False
        if (status < 400 or status in (404, 410)) and (
            state["consecutive_429"] or state["blocked_until"]
        ):
            state["consecutive_429"] = 0
            state["blocked_until"] = 0.0
            _write_state(state)
    return False


def request(url: str, send: Callable[[], Any]) -> Any:
    """Perform ``send()`` — the caller's own GET of ``url`` — through the
    arXiv queue. Non-arXiv URLs pass straight through. Returns the response,
    a cached response, or ``None`` when the request was skipped (paused or
    over budget)."""
    if not is_arxiv_url(url):
        return send()
    cached = cache_get(url)
    if cached is not None:
        return cached
    while True:
        release = acquire_slot(url)
        if release is None:
            return None
        try:
            response = send()
        finally:
            release()
        status = getattr(response, "status_code", None)
        should_wait = report(url, status)
        if status == 200:
            cache_put(url, response)
        if should_wait:
            _sf.record_failure("arxiv", "http_429", status=429, url=url, detail="queue backing off")
            continue
        return response


# --- cache ----------------------------------------------------------------------

def _cache_enabled() -> bool:
    return os.environ.get("FI_ARXIV_CACHE", "").strip() != "0"


def _cache_paths(url: str) -> tuple[Path, Path]:
    key = hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()
    root = _cache_root() / "responses"
    return root / f"{key}.bin", root / f"{key}.json"


def cache_get(url: str) -> httpx.Response | None:
    if not _cache_enabled() or not is_arxiv_url(url):
        return None
    body_path, meta_path = _cache_paths(url)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if _now() - float(meta.get("fetched_at", 0.0)) > CACHE_TTL_S:
            return None
        body = body_path.read_bytes()
    except (OSError, ValueError, TypeError):
        return None
    try:
        os.utime(body_path, None)
    except OSError:
        pass
    return httpx.Response(
        200, content=body,
        headers={"content-type": str(meta.get("content_type") or "")},
        request=httpx.Request("GET", str(meta.get("url") or url)),
    )


def cache_put(url: str, response: Any) -> None:
    if not _cache_enabled() or not is_arxiv_url(url):
        return
    if getattr(response, "status_code", None) != 200:
        return
    body = getattr(response, "content", None)
    if not isinstance(body, (bytes, bytearray)) or not body or len(body) > MAX_ENTRY_BYTES:
        return
    if any(m in bytes(body[:4096]).lower() for m in _CHALLENGE_MARKERS):
        return
    headers = getattr(response, "headers", {}) or {}
    ctype = headers.get("content-type", "") if hasattr(headers, "get") else ""
    body_path, meta_path = _cache_paths(url)
    try:
        body_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
        tmp_body = body_path.with_suffix(suffix)
        tmp_body.write_bytes(bytes(body))
        os.replace(tmp_body, body_path)
        tmp_meta = meta_path.with_suffix(suffix)
        tmp_meta.write_text(json.dumps({
            "url": url, "content_type": str(ctype), "fetched_at": _now(), "size": len(body),
        }), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    except OSError as e:
        _log.info("arxiv cache write failed: %s", e)
        return
    _evict()


def _evict() -> None:
    root = _cache_root() / "responses"
    try:
        entries = [(p.stat().st_mtime, p.stat().st_size, p) for p in root.glob("*.bin")]
    except OSError:
        return
    total = sum(size for _, size, _ in entries)
    if total <= MAX_CACHE_BYTES:
        return
    for _, size, path in sorted(entries):
        try:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            continue
        total -= size
        if total <= MAX_CACHE_BYTES:
            break
