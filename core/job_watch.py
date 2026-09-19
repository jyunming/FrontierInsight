"""Experiments that run as background jobs (HPC, a cluster, anything longer than the
wall-time limit).

The contract is one every generated ``experiment.py`` follows when
``execution.background_jobs`` is on, so FI needs no scheduler-specific code: the
selected skill knows how to submit, how to tell finished from failed, and how to read
the results, and the script is written from it.

The script is an idempotent driver that FI runs again and again:

* first run: submit the job, print ``RESULT_JSON: {"fi_job": {"status": "pending", ...}}``
  and exit 0;
* later runs: check the job. Still running -> the same pending line. Finished ->
  collect the results and print the real ``RESULT_JSON`` (no ``fi_job`` key).

On a pending line the quest pauses and exits cleanly (``.fi/pending.json`` says what it
is waiting for). ``--resume`` re-runs the script, which either reports pending again or
delivers. ``--watch`` does the re-running on a timer and resumes the quest by itself
when the script stops saying pending.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

PENDING_FILE = "pending.json"
DEFAULT_POLL_S = 300
MIN_POLL_S = 5

PENDING = "pending"
DONE = "done"
FAILED = "failed"


def job_of(result_json: dict[str, Any] | None) -> dict[str, Any] | None:
    """The ``fi_job`` object of a RESULT_JSON, or None when it carries none.

    A dedicated key rather than a field name a normal result could hold
    (``pending`` is a common one), so an ordinary experiment is never mistaken
    for a job.
    """
    if not isinstance(result_json, dict):
        return None
    job = result_json.get("fi_job")
    if not isinstance(job, dict):
        return None
    status = str(job.get("status") or "").strip().lower()
    return {**job, "status": status or PENDING}


def poll_seconds(job: dict[str, Any] | None, default: int = DEFAULT_POLL_S) -> int:
    """How long to wait between checks: the script's own ``poll_s`` if it gave a
    sane one."""
    try:
        value = int((job or {}).get("poll_s"))
    except (TypeError, ValueError):
        return default
    return max(MIN_POLL_S, value)


def pending_path(fi_dir: Path) -> Path:
    return fi_dir / PENDING_FILE


def read_pending(fi_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(pending_path(fi_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_pending(fi_dir: Path, job: dict[str, Any], script: str) -> dict[str, Any]:
    """Record what the quest is waiting for. ``submitted_at`` and the poll count
    survive re-writes, so the file tells how long the wait has been."""
    previous = read_pending(fi_dir) or {}
    now = time.time()
    info = {
        "script": script,
        "job": job,
        "submitted_at": previous.get("submitted_at") or now,
        "last_checked_at": now,
        "checks": int(previous.get("checks") or 0) + 1,
    }
    fi_dir.mkdir(parents=True, exist_ok=True)
    pending_path(fi_dir).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def clear_pending(fi_dir: Path) -> bool:
    try:
        pending_path(fi_dir).unlink()
        return True
    except OSError:
        return False


def _now() -> float:
    return time.monotonic()


def describe(info: dict[str, Any]) -> str:
    """One line for a log or a banner."""
    job = info.get("job") or {}
    bits = []
    if job.get("id"):
        bits.append(f"id {job['id']}")
    if job.get("note"):
        bits.append(str(job["note"]))
    waited = int(time.time() - float(info.get("submitted_at") or time.time()))
    bits.append(f"waiting {waited // 3600}h{(waited % 3600) // 60:02d}m")
    return ", ".join(bits)


async def watch(
    poll: Callable[[], Any],
    *,
    fi_dir: Path,
    every_s: int,
    max_wait_s: int,
    out: Callable[[str], None] = print,
) -> str:
    """Call ``poll()`` (an async callable returning ``(state, info)``) every
    ``every_s`` seconds until the job is no longer pending. Returns ``"done"``,
    ``"failed"`` or ``"timeout"``. Every check is reported through ``out``: the
    person watching, and the log, are the only monitor there is."""
    started = _now()
    checks = 0
    while True:
        checks += 1
        state, info = await poll()
        stamp = time.strftime("%H:%M:%S")
        note = str((info.get("job") or {}).get("note") or info.get("note") or "")
        out(f"[watch] {stamp} check {checks}: {state}" + (f" - {note}" if note else ""))
        if state != PENDING:
            return DONE if state == DONE else FAILED
        waited = _now() - started
        if max_wait_s and waited >= max_wait_s:
            out(f"[watch] gave up after {waited / 3600:.1f} h; the quest is still paused. "
                f"`--resume` checks once more, `--watch` starts again.")
            return "timeout"
        pending = read_pending(fi_dir) or {}
        wait = max(MIN_POLL_S, int(every_s or poll_seconds(pending.get("job"))))
        await asyncio.sleep(wait)
