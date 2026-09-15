"""Which exact skill contents passed their self-tests, and under which Python.

Loading skills for a quest runs every candidate's self-test, and a self-test
is a subprocess that imports real scientific libraries. Measured on a
108-skill library that is 340 s sequentially, with single skills taking 83 s
— paid again by every quest, for skills that had not changed since the last.

So a pass is recorded against everything that decides whether the test
could come out differently:

* the skill's name and its content hash — the same digest approval binds to,
  so any edit to any file in the skill means a new test;
* the Python version and executable that ran it;
* a fingerprint of every installed distribution (``name==version``), so an
  upgrade of any package the skill might import means a new test.

A quest-time load that finds a matching record treats the self-test as
passed. Nothing else about the gate changes: the approval check still runs
against the ledger every time, and on-demand commands (``--skills``,
approval, the web listing) still run the test fresh.

Only passes are recorded. A failing or timed-out self-test is never written
down, so it runs again on every load until it passes — and a self-test that
fails anywhere removes the skill's recorded passes, because the key cannot
see everything a test depends on (an external binary, a network service), and
a failure someone just watched must not be overruled by an older pass.

The record lives beside the approval ledger, never inside a skill directory:
the content hash covers the whole skill folder, so writing there would change
the hash and revoke the approval the test was checking.

It is a cache, not a ledger. A missing, unreadable or corrupt file is an
empty cache, a write that fails is logged and skipped, and deleting the file
is always safe — the only cost is that each skill's self-test runs once more.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.skills import approval

_log = logging.getLogger("fi.skills")

_ENV_CACHE = "FI_SKILLS_SELFTEST_CACHE"
CACHE_FILENAME = "skill_selftest_cache.json"

#: Bumped when the file's shape changes; a file of another format is empty.
_FORMAT = 1

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def cache_path(ledger: Path | None = None) -> Path:
    """Where recorded passes are kept.

    ``FI_SKILLS_SELFTEST_CACHE`` overrides, which is what tests use. Otherwise
    the file sits beside the approval ledger in use — the ledger passed in, or
    the default one — so anything pointed at its own ledger is pointed at its
    own cache too.
    """
    override = os.environ.get(_ENV_CACHE, "").strip()
    if override:
        return Path(override).expanduser()
    return (ledger or approval.ledger_path()).parent / CACHE_FILENAME


def package_fingerprint() -> str:
    """sha256 over the sorted ``name==version`` of every installed distribution."""
    from importlib import metadata

    pairs = sorted(
        f"{d.metadata['Name'] or ''}=={d.version or ''}"
        for d in metadata.distributions()
    )
    return hashlib.sha256("\n".join(pairs).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Environment:
    """The interpreter a self-test ran under, as far as the key can see it."""

    python: str
    executable: str
    packages: str


def current_environment() -> Environment:
    return Environment(
        python=sys.version,
        executable=sys.executable,
        packages=package_fingerprint(),
    )


_SCANNER_VERSION: str | None = None


def scanner_version() -> str:
    """Digest of the static scanner's own source.

    A pass records the scan of the same content, so a quest-time load need
    not parse the skill again. The scan depends on the rules as well as the
    skill, so findings recorded by an older scanner are not reused. Empty when
    the source cannot be read, which disables reusing findings.
    """
    global _SCANNER_VERSION
    if _SCANNER_VERSION is None:
        try:
            from core.skills import scan

            data = Path(scan.__file__).read_bytes()
            _SCANNER_VERSION = hashlib.sha256(data).hexdigest()[:16]
        except (OSError, TypeError, ImportError):
            _SCANNER_VERSION = ""
    return _SCANNER_VERSION


@dataclass(frozen=True)
class CachedPass:
    """A recorded pass that matches the skill and the environment."""

    output: str
    #: The scan taken with the pass, or None when it cannot be reused.
    findings: list[str] | None


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(str(path)))
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def _load(path: Path) -> dict[str, dict[str, Any]]:
    """The entries on disk. Anything unreadable or malformed is empty."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("format") != _FORMAT:
        return {}
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {k: v for k, v in entries.items() if isinstance(k, str) and isinstance(v, dict)}


def _save(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Write atomically: a reader sees the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"format": _FORMAT, "entries": entries}, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update(
    path: Path,
    change: Callable[[dict[str, dict[str, Any]]], bool],
    *,
    warn: bool = True,
) -> bool:
    """Re-read the file, apply ``change``, write it back if it changed.

    Re-reading rather than writing a snapshot keeps what another load recorded
    meanwhile. Callers hold the path's lock, which covers loads in this
    process; two processes writing at once can still lose one's records, which
    costs a re-test and never records a failure. Never raises: returns False
    when the write failed.
    """
    try:
        data = _load(path)
        if change(data):
            _save(path, data)
        return True
    except Exception as e:  # noqa: BLE001 - a cache must never stop a quest
        if warn:
            _log.warning("skills: could not update the self-test cache %s: %s", path, e)
        return False


def _drop_skill(data: dict[str, dict[str, Any]], name: str) -> bool:
    gone = [k for k, e in data.items() if e.get("name") == name]
    for k in gone:
        del data[k]
    return bool(gone)


def forget_skill(name: str, *, ledger: Path | None = None, path: Path | None = None) -> None:
    """Remove every recorded pass for this skill. Never raises."""
    p = path or cache_path(ledger)
    with _lock_for(p):
        _update(p, lambda data: _drop_skill(data, name))


class SelftestCache:
    """Recorded passes for one load, under one environment fingerprint.

    Opened once per ``loadable_skills`` call, so the package fingerprint is
    computed once per call rather than once per skill — and not once per
    process either, because a long-running web server can outlive a
    ``pip install``. Safe to share across the worker threads of that call.
    """

    def __init__(self, path: Path, env: Environment, *, scanner: str = "") -> None:
        self.path = Path(path)
        self.env = env
        self.scanner = scanner
        self._lock = _lock_for(self.path)
        with self._lock:
            self._entries = _load(self.path)
        self.hits = 0
        self.ran = 0
        self._warned = False

    @classmethod
    def open(
        cls, *, ledger: Path | None = None, path: Path | None = None,
    ) -> "SelftestCache | None":
        """A cache for this environment, or None when it cannot be keyed.

        A broken distribution that stops the fingerprint must cost the cache,
        not the skills: without a key every self-test simply runs.
        """
        try:
            env = current_environment()
        except Exception as e:  # noqa: BLE001
            _log.warning(
                "skills: could not fingerprint the Python environment (%s); "
                "self-tests run without the cache", e,
            )
            return None
        return cls(path or cache_path(ledger), env, scanner=scanner_version())

    # ---- key ----

    def _key_fields(self, name: str, content_hash: str) -> tuple[str, ...]:
        """Everything a pass depends on. A change to any of it is a new test."""
        return (
            name,
            content_hash,
            self.env.python,
            self.env.executable,
            self.env.packages,
        )

    def key(self, name: str, content_hash: str) -> str:
        fields = self._key_fields(name, content_hash)
        return hashlib.sha256(json.dumps(fields).encode("utf-8")).hexdigest()

    # ---- read ----

    def lookup(self, name: str, content_hash: str) -> CachedPass | None:
        fields = self._key_fields(name, content_hash)
        entry = self._entries.get(self.key(name, content_hash))
        # The stored fields are compared as well as the digest, so a
        # hand-edited or mismatched entry is a miss rather than a pass.
        if not isinstance(entry, dict) or tuple(entry.get("key") or ()) != fields:
            return None
        output = entry.get("output")
        findings = entry.get("findings")
        reusable = (
            bool(self.scanner)
            and entry.get("scanner") == self.scanner
            and isinstance(findings, list)
            and all(isinstance(f, str) for f in findings)
        )
        with self._lock:
            self.hits += 1
        return CachedPass(
            output=output if isinstance(output, str) else "",
            findings=list(findings) if reusable else None,
        )

    # ---- write ----

    def note_ran(self) -> None:
        with self._lock:
            self.ran += 1

    def record_pass(
        self,
        name: str,
        content_hash: str,
        *,
        output: str,
        findings: list[str] | None,
    ) -> None:
        """Record a pass. Written straight away, so an interrupted load keeps
        the passes it finished. Older records of the same skill under the same
        interpreter are dropped, which keeps the file one entry per skill per
        interpreter."""
        fields = self._key_fields(name, content_hash)
        key = self.key(name, content_hash)
        entry: dict[str, Any] = {
            "key": list(fields),
            "name": name,
            "content_hash": content_hash,
            "python": self.env.python,
            "executable": self.env.executable,
            "packages": self.env.packages,
            "scanner": self.scanner if findings is not None else "",
            "findings": findings,
            "output": output,
            "passed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        def change(data: dict[str, dict[str, Any]]) -> bool:
            for k in [
                k for k, e in data.items()
                if k != key and e.get("name") == name
                and e.get("executable") == self.env.executable
            ]:
                del data[k]
            data[key] = entry
            return True

        with self._lock:
            self._entries[key] = entry
            self._write(change)

    def forget(self, name: str) -> None:
        with self._lock:
            _drop_skill(self._entries, name)
            self._write(lambda data: _drop_skill(data, name))

    def _write(self, change: Callable[[dict[str, dict[str, Any]]], bool]) -> None:
        # Held lock: the caller's. Every write is attempted — a failure can be
        # transient, another process holding the file — but a directory that
        # cannot be written warns once per load, not once per skill.
        if not _update(self.path, change, warn=not self._warned):
            self._warned = True
