"""FI's view of Axon through the Axon service that is already running, instead of a second Axon inside every FI process.

Every FI process (a quest, the web server, the ``--summarize`` / ``--digest`` / ``--ingest`` subcommands) used to build its
own in-process ``AxonBrain``: a new embedding stack, the wall of configuration warnings Axon prints at every start, and a
second process writing the vector-store files the Axon service already holds (Axon's own docstring calls two processes on
one store a crash hazard). :class:`AxonHTTPBrain` has the three methods FI uses of a brain, ``ingest``, ``finalize_ingest``
and ``search_raw``, and answers each through the service's HTTP API.

The service has one *active project* at a time, and a request that names another project is refused. FI keeps its corpus in
its own project, so each operation is bracketed: the active project is read, switched to FI's project if it is another one,
the operation runs, and the project that was active is switched back to. A lock, taken across processes as well as across
threads, holds the bracket, so two FI processes cannot switch under each other. Anything that is not FI (the Axon tools of an
editor, say) still sees FI's project for the length of one operation; that is the price of sharing one service, and it is
why the bracket is as short as one operation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_log = logging.getLogger("fi.knowledge")

# One operation is one HTTP call to the service, except an ingest, which embeds its documents before it answers.
_REQUEST_TIMEOUT_S = 300.0
# How long an operation waits for another FI process that is inside its own bracket (an ingest can take minutes).
_LOCK_TIMEOUT_S = 900.0


class AxonUnavailable(RuntimeError):
    """The Axon service could not be reached or refused an operation; the message says why."""


def _api_key() -> str:
    return os.environ.get("RAG_API_KEY", "").strip()


class _ProcessLock:
    """A lock across processes and threads: a thread lock in front of an OS lock on a file that every FI process names the same.

    The OS releases a file lock when its process dies, so a killed quest cannot leave the service locked for the next one.
    """

    def __init__(self, path: Path, timeout: float = _LOCK_TIMEOUT_S) -> None:
        self._path, self._timeout = path, timeout
        self._threads = threading.RLock()
        self._depth = 0
        self._fh: Any = None

    @contextmanager
    def held(self) -> Iterator[bool]:
        """Hold the lock; yields ``True`` for the outermost hold and ``False`` for one nested in it (same thread)."""
        with self._threads:
            outermost = self._depth == 0
            if outermost:
                self._acquire_file()
            self._depth += 1
            try:
                yield outermost
            finally:
                self._depth -= 1
                if outermost:
                    self._release_file()

    def _acquire_file(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self._path, "a+b")  # noqa: SIM115 -- closed in _release_file
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    fh.close()
                    raise AxonUnavailable(
                        f"another FI process has held the Axon project for more than {int(self._timeout)}s "
                        f"(lock file {self._path})",
                    ) from None
                time.sleep(0.2)
        self._fh = fh

    def _release_file(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def _lock_path_for(base_url: str) -> Path:
    """The lock file for one service: named after its address so every FI process that talks to it takes the same one."""
    digest = hashlib.sha1(base_url.rstrip("/").encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"fi-axon-project-{digest}.lock"


class AxonHTTPBrain:
    """The three brain methods FI uses (``ingest``, ``finalize_ingest``, ``search_raw``) over the Axon service at ``base_url``."""

    def __init__(self, base_url: str, project: str, *, lock: _ProcessLock | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.project = project
        self._lock = lock or _ProcessLock(_lock_path_for(self.base_url))
        self._known_project_exists = False

    # -- HTTP ---------------------------------------------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout: float = _REQUEST_TIMEOUT_S) -> Any:
        headers = {"Content-Type": "application/json", "X-Axon-Surface": "fi"}
        if _api_key():
            headers["X-API-Key"] = _api_key()
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 -- the service address is configuration
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = str(json.loads(exc.read().decode("utf-8")).get("detail") or "")
            except Exception:  # noqa: BLE001 -- the reason is best-effort
                pass
            raise AxonUnavailable(f"Axon answered {method} {path} with {exc.code}: {detail or exc.reason}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise AxonUnavailable(f"could not reach Axon at {self.base_url} ({method} {path}): {exc}") from None
        return json.loads(raw) if raw else {}

    def _active_project(self) -> str:
        return str(self._request("GET", "/health/ready", timeout=10.0).get("project") or "")

    def _switch(self, project: str) -> None:
        self._request("POST", "/project/switch", {"project_name": project}, timeout=60.0)

    def _ensure_project_exists(self) -> None:
        if not self._known_project_exists:
            self._request("POST", "/project/new", {"name": self.project, "description": "Frontier Insight quest papers + retrieval corpus"}, timeout=60.0)
            self._known_project_exists = True

    # -- the bracket --------------------------------------------------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[None]:
        """Run what is inside with FI's project active on the service, and put the project that was active back after.

        Nested sessions (an ingest followed by its finalize, in one ``with``) share the outermost one's switch.
        """
        with self._lock.held() as outermost:
            if not outermost:
                yield
                return
            self._ensure_project_exists()
            previous = self._active_project()
            switched = previous != self.project
            if switched:
                self._switch(self.project)
            try:
                yield
            finally:
                if switched and previous:
                    try:
                        self._switch(previous)
                    except AxonUnavailable as exc:
                        _log.warning("could not switch Axon back to %r after an FI operation: %s", previous, exc)

    # -- the brain methods FI uses ------------------------------------------------------------------------------------

    def switch_project(self, name: str) -> None:  # noqa: ARG002 -- a session switches; FI's project is fixed at construction
        return None

    def ingest(self, documents: list[dict[str, Any]]) -> int:
        """Write ``[{id, text, metadata}, ...]``; returns how many the service reports as newly created."""
        docs = [
            {"text": str(d.get("text") or ""), "doc_id": d.get("id"), "metadata": dict(d.get("metadata") or {})}
            for d in documents
            if str(d.get("text") or "").strip()
        ]
        if not docs:
            return 0
        with self.session():
            results = self._request("POST", "/add_texts", {"docs": docs, "project": self.project})
        return sum(1 for r in results if isinstance(r, dict) and r.get("status") == "created")

    def finalize_ingest(self) -> None:
        with self.session():
            self._request("POST", "/graph/finalize", {})

    def search_raw(
        self, query: str, *, filters: dict[str, Any] | None = None, overrides: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], None]:
        """The rows retrieval returns, with the diagnostics the service reports; the third value (a trace) is not carried."""
        body: dict[str, Any] = {"query": query, "project": self.project}
        if filters:
            body["filters"] = filters
        for key in ("top_k", "threshold"):
            if overrides and overrides.get(key) is not None:
                body[key] = overrides[key]
        with self.session():
            out = self._request("POST", "/search/raw", body)
        return list(out.get("results") or []), dict(out.get("diagnostics") or {}), None
