"""Where is Axon listening? — endpoint discovery shared by all FI surfaces.

Why this module exists
======================
Axon's API port is no longer a constant. ``axon-api`` resolves its bind
address as ``--port`` > ``AXON_PORT`` > ``config.yaml``'s ``api.port`` >
``8420``, so two developers running the same command can end up on
different ports, and neither of them has an environment variable that
says so. FI used to hardcode ``127.0.0.1:8000`` — the pre-8420 default —
which meant a perfectly healthy sidecar was reported as "not running".

The fix is discovery rather than a guess. Axon writes a per-store lock
file when its API server boots (``<projects_root>/.axon-api.lock``,
JSON ``{"host", "port", "pid"}``) and removes it on clean shutdown. That
file exists exactly when the question "where is Axon?" has an answer, so
it is the authoritative source. Everything else is a fallback.

Resolution model
================
We build an ordered, de-duplicated list of *candidates* and probe them,
rather than committing to a single precedence winner. A stale
``AXON_PORT=8000`` left in someone's environment then costs one dead
probe instead of hiding a live sidecar:

1. ``AXON_HOST`` / ``AXON_PORT``
2. the store lock file
3. Axon's ``config.yaml`` (``api.host`` / ``api.port``)
4. ``127.0.0.1:8420`` — Axon's current default
5. ``127.0.0.1:8000`` — pre-8420 Axon, for back-compat

Two inputs short-circuit that list instead of joining it: explicit
``host``/``port`` arguments, and ``AXON_API_BASE`` / ``RAG_API_BASE``.
Each names one specific Axon, and two Axon instances hold different
corpora — falling through to a different one because the named one is
down would silently change what a quest reads.

Bind host vs. probe host
========================
The lock file records the address the server *bound*, which is commonly
``0.0.0.0`` (and observed to be ``0.0.0.0`` even when uvicorn actually
bound loopback). You cannot connect to ``0.0.0.0``; rewrite it to
``127.0.0.1`` before probing. Axon's own client does the same thing.

The mirror of this logic for the VSCode extension lives in
``vscode-frontier-insight/src/axon-endpoint.ts``. Keep them in step.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

# Axon's current default port. Only used when nothing else answers —
# config.yaml normally supplies the real one.
DEFAULT_PORT = 8420
# The port FI assumed before Axon moved. Probed last so an older Axon
# install keeps working without configuration.
LEGACY_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

_LOCK_NAME = ".axon-api.lock"
# Wildcard binds are addresses to *listen* on, not addresses to connect
# to. Anything in here becomes loopback for probing purposes.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "*", ""})

# Source labels. Kept as constants because ``preferred_endpoint`` filters
# on one of them and a typo there would silently change spawn behaviour.
SOURCE_EXPLICIT = "explicit"
SOURCE_LOCK = "store lock file"
SOURCE_CONFIG = "axon config.yaml"
SOURCE_DEFAULT = "default"
SOURCE_LEGACY = "legacy default"


@dataclass(frozen=True)
class Candidate:
    """One place Axon might be, and why we think so.

    ``source`` is carried purely for diagnostics — when every candidate
    fails we tell the user which ones we tried and where each came from,
    so "Axon not detected" is debuggable instead of mysterious.
    """

    host: str
    port: int
    source: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def probe_host(host: str | None) -> str:
    """Map a bind address onto something connectable."""
    if host is None:
        return DEFAULT_HOST
    h = host.strip()
    return DEFAULT_HOST if h in _WILDCARD_HOSTS else h


def _split_base_url(raw: str) -> tuple[str, int] | None:
    """Parse an ``AXON_API_BASE``-style value into (host, port).

    Only the host and port survive: probing always speaks plain HTTP, so
    an ``https://`` value resolves its default port (443) but is not
    contacted over TLS. A TLS-fronted Axon therefore fails visibly
    rather than appearing to work.
    """
    raw = raw.strip().rstrip("/")
    if not raw:
        return None
    if "//" not in raw:
        raw = f"http://{raw}"
    try:
        parts = urlsplit(raw)
        if not parts.hostname:
            return None
        port = parts.port or (443 if parts.scheme == "https" else DEFAULT_PORT)
        return probe_host(parts.hostname), int(port)
    except (ValueError, TypeError):
        return None


def _axon_config() -> object | None:
    """Load Axon's config, or ``None`` if Axon isn't importable.

    FI treats Axon as an optional dependency everywhere else, so a
    missing or broken install must degrade to the static fallbacks
    rather than raising out of a health check.
    """
    try:
        from axon.config import AxonConfig

        return AxonConfig.load(os.environ.get("AXON_CONFIG_PATH"))
    except Exception:  # noqa: BLE001 - any import/parse failure means "no config"
        return None


def _config_endpoint(config: object | None) -> tuple[str, int] | None:
    if config is None:
        return None
    host = getattr(config, "api_host", None)
    port = getattr(config, "api_port", None)
    if not port:
        return None
    try:
        return probe_host(host or DEFAULT_HOST), int(port)
    except (TypeError, ValueError):
        return None


def lock_file_endpoint(config: object | None = None) -> tuple[str, int] | None:
    """Read ``<projects_root>/.axon-api.lock``, if a server wrote one.

    Reads the file and nothing more. Axon ships
    ``server_client.find_live_server_for_store``, which additionally
    validates the lock with a ``/health/ready`` request, but we
    deliberately don't call it: building the candidate list would then
    do network I/O, and every caller probes the candidate anyway, so a
    stale lock costs one dead entry either way rather than a wrong
    answer. Keeping enumeration pure also keeps it fast and testable.
    """
    if config is None:
        config = _axon_config()
    if config is None:
        return None

    root = getattr(config, "projects_root", None)
    if not root:
        return None
    try:
        with open(os.path.join(str(root), _LOCK_NAME), encoding="utf-8") as fh:
            info = json.load(fh)
        if not info.get("port"):
            return None
        return probe_host(info.get("host")), int(info["port"])
    except (OSError, ValueError, TypeError):
        return None


def candidates(
    host: str | None = None,
    port: int | None = None,
    *,
    env: Mapping[str, str] | None = None,
    include_legacy: bool = True,
) -> list[Candidate]:
    """Ordered, de-duplicated list of places to look for Axon."""
    if env is None:
        env = os.environ
    config = _axon_config()
    out: list[Candidate] = []
    seen: set[tuple[str, int]] = set()

    def add(h: str | None, p: int | str | None, source: str) -> None:
        if p is None:
            return
        try:
            port_n = int(p)
        except (TypeError, ValueError):
            return
        if not (1 <= port_n <= 65535):
            return
        key = (probe_host(h), port_n)
        if key in seen:
            return
        seen.add(key)
        out.append(Candidate(host=key[0], port=key[1], source=source))

    # An address that names one specific Axon is the *only* candidate.
    # Two Axon instances hold different corpora, so quietly using a
    # different one because the named one is down would change what a
    # quest reads — better to report the named one as unreachable.
    # ``AXON_HOST``/``AXON_PORT`` below are deliberately not treated this
    # way: FI told users to set those for years, so a stale value is
    # likely and worth searching past.
    if host is not None or port is not None:
        add(
            host or DEFAULT_HOST,
            port if port is not None else DEFAULT_PORT,
            SOURCE_EXPLICIT,
        )
        return out

    for var in ("AXON_API_BASE", "RAG_API_BASE"):
        raw = env.get(var)
        if raw:
            parsed = _split_base_url(raw)
            if parsed:
                add(parsed[0], parsed[1], var)
                return out

    # Split host/port env vars — FI's historical knob.
    env_host, env_port = env.get("AXON_HOST"), env.get("AXON_PORT")
    if env_host or env_port:
        add(env_host or DEFAULT_HOST, env_port or DEFAULT_PORT, "AXON_HOST/AXON_PORT")

    # The running server's own lock file — the authoritative answer.
    lock = lock_file_endpoint(config)
    if lock:
        add(lock[0], lock[1], SOURCE_LOCK)

    # Axon's configured API address.
    cfg = _config_endpoint(config)
    if cfg:
        add(cfg[0], cfg[1], SOURCE_CONFIG)

    # Static fallbacks, current default first.
    add(DEFAULT_HOST, DEFAULT_PORT, SOURCE_DEFAULT)
    if include_legacy:
        add(DEFAULT_HOST, LEGACY_PORT, SOURCE_LEGACY)

    return out


def preferred_endpoint(
    host: str | None = None,
    port: int | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[str, int]:
    """Where FI should *start* a sidecar when none is running.

    Deliberately not the same list as :func:`candidates`: the lock file
    describes a server that already exists (nothing to start), and the
    legacy 8000 fallback would plant a new sidecar on a port current
    Axon has moved away from. What's left is explicit args, the env
    vars, config.yaml, then 8420.
    """
    if env is None:
        env = os.environ
    if host is not None and port is not None:
        return probe_host(host), int(port)

    for cand in candidates(host, port, env=env, include_legacy=False):
        if cand.source == SOURCE_LOCK:
            continue
        return cand.host, cand.port
    return DEFAULT_HOST, DEFAULT_PORT
