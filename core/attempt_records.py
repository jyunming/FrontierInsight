"""What a quest tried, and under which conditions: the records the explore/confirm work is built on.

Two append-only JSONL files in the quest's ``.fi/`` folder, written as the quest runs and read by nothing that decides a
route (this is the "records only" first phase; a later phase reads them in shadow mode, then for real):

- ``attempts.jsonl``: one line per run of the experiment, one when the quest stops for a check it failed, and one when
  the quest ends or fails, each with an ``outcome`` from :data:`OUTCOMES` and the ``context`` it happened under
  (:func:`context_fingerprint`). A failure is a fact about an attempt under a context, not about a method in general: a
  different model, protocol or environment is a different context.
- ``branch_ledger.jsonl``: the choices along the way, with what was chosen from what: the candidate ideas and how one was
  picked, each revision of the design and why, each repair of a script.

Kept apart from the accepted evidence (nothing here is written to the knowledge base, and nothing here supports a
claim). Every write is best-effort: a record that cannot be built or written never touches the quest.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import frozen_protocol as _frozen

#: What an attempt came to. ``process_error``: it crashed or produced no result. ``protocol_mismatch``: it ran something
#: other than what the protocol fixed. ``oracle_failure``: it did not pass the known-answer checks. ``inconclusive``: it
#: ran, but no reviewer accepted the result. ``accepted``: a reviewer accepted it. Whether an accepted result supports,
#: contradicts or finds no effect for the hypothesis is not derived: the analysis does not state it in a field, and
#: guessing from its prose would put false labels on the record.
OUTCOMES = ("process_error", "protocol_mismatch", "oracle_failure", "inconclusive", "accepted")
ATTEMPTS = "attempts.jsonl"
LEDGER = "branch_ledger.jsonl"

#: A stop for one of these checks is an attempt that failed it. Any other stop is not an outcome: the plan waiting for
#: you, papers asked for, a protocol change waiting for approval, or a check that could not be asked (an outage is not
#: a fact about the attempt).
STOP_OUTCOMES = {
    "oracle": "oracle_failure",
    "manifest": "protocol_mismatch", "protocol": "protocol_mismatch",
    "split": "process_error", "numeric": "process_error", "replicate_seed_unrepairable": "process_error",
}

#: Above this size a file is identified by its size and its first and last MiB, not read whole.
_WHOLE_FILE_LIMIT = 64 * 1024 * 1024
_CHUNK = 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str | None:
    """A hash of the file, read in chunks; a file over :data:`_WHOLE_FILE_LIMIT` is hashed by its size and its first
    and last MiB. ``None`` when it cannot be read."""
    try:
        h = hashlib.sha256()
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _WHOLE_FILE_LIMIT:
                h.update(f"size:{size}".encode("ascii"))
                h.update(fh.read(_CHUNK))
                fh.seek(-_CHUNK, 2)
                h.update(fh.read(_CHUNK))
            else:
                for block in iter(lambda: fh.read(_CHUNK), b""):
                    h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _json_sha(value: Any) -> str | None:
    try:
        return _sha(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@functools.lru_cache(maxsize=4)
def _fi_version(repo: Path) -> dict[str, Any] | None:
    """FI's own commit, and whether the checkout had changes, when ``repo`` is the top of a git checkout; ``None``
    otherwise (an installed copy, or a folder inside someone else's repository). Asked once per process: it is the
    code this process loaded."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != repo.resolve():
        return None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)} if commit else None


def _folder_sha(folder: Path, limit: int = 200) -> str | None:
    """A hash over the names and contents of up to ``limit`` files (the inputs a quest was given)."""
    if not folder.is_dir():
        return None
    h = hashlib.sha256()
    for n, path in enumerate(sorted(p for p in folder.rglob("*") if p.is_file())):
        if n >= limit:
            h.update(b"...")
            break
        h.update(path.relative_to(folder).as_posix().encode("utf-8"))
        h.update((_file_sha(path) or "").encode("ascii"))
    return h.hexdigest()


def _environment_sha(needs: Path) -> str | None:
    """A hash of what the environment is (Python, platform, sandbox, isolation, packages), not of where it lives or
    when it was recorded, so two quests with the same packages match."""
    record = _read_json(needs / "ENVIRONMENT.json")
    if not isinstance(record, dict):
        return None
    keep = ("python", "platform", "sandbox", "shared_interpreter", "system_site_packages", "isolated", "packages")
    return _json_sha({k: record.get(k) for k in keep})


def context_fingerprint(config: Any, quest_root: Path, state: dict[str, Any], *, prompts: dict[str, Any] | None = None,
                        fi_repo: Path | None = None) -> dict[str, Any]:
    """The conditions an attempt ran under, as far as the quest knows them. Two attempts share a context only when
    these match: the model and the settings that change its answers, the prompts it was given, FI's own version, the
    selected skills, the environment and its packages, the inputs, the protocol and its metric definitions, the budget,
    and where in the quest's lineage the attempt sits. A field FI could not work out is ``None``."""
    provider = getattr(config, "provider", None)
    engine = getattr(config, "engine", None)
    execution = getattr(config, "execution", None)
    needs = quest_root / "needs"
    frozen = _read_json(needs / "FROZEN_PROTOCOL.json")
    frozen_protocol = frozen.get("protocol") if isinstance(frozen, dict) else None
    design = state.get("design") if isinstance(state.get("design"), dict) else {}
    protocol = frozen_protocol if isinstance(frozen_protocol, dict) else (design.get("protocol") or {})
    if not isinstance(protocol, dict):
        protocol = {}
    try:
        protocol_sha = _frozen.sha256(protocol) if protocol else None
    except (TypeError, ValueError):
        protocol_sha = None
    return {
        "provider": getattr(provider, "name", "") or None,
        "model": getattr(provider, "model", "") or None,
        "settings": {
            "node_models": dict(getattr(provider, "node_models", {}) or {}),
            "reasoning_effort": getattr(provider, "reasoning_effort", None) or None,
            "fixed_temperature": getattr(provider, "fixed_temperature", None),
            "base_url": getattr(provider, "base_url", None),
            "fallback": list(getattr(provider, "fallback", []) or []),
            "node_ensemble": sorted((getattr(provider, "node_ensemble", None) or {}).keys()),
        },
        "prompts_sha256": _json_sha({k: getattr(v, "template", str(v)) for k, v in (prompts or {}).items()}),
        "fi": _fi_version(fi_repo) if fi_repo is not None else None,
        "skills": sorted(str(s) for s in (state.get("selected_skills") or [])),
        "environment_sha256": _environment_sha(needs),
        "dependency_lock_sha256": _file_sha(quest_root / ".fi" / "requirements.lock.txt"),
        "inputs_sha256": _folder_sha(quest_root / "inputs"),
        "protocol_sha256": protocol_sha,
        "metric_specs_sha256": _json_sha(protocol.get("metrics")) if protocol.get("metrics") else None,
        "budget": {
            "max_iterations": getattr(engine, "max_iterations", None),
            "exec_reflect_max_iterations": getattr(engine, "exec_reflect_max_iterations", None),
            "execute_replicates": getattr(engine, "execute_replicates", None),
            "runs_per_setting": protocol.get("runs_per_setting"),
            "timeout_s": getattr(execution, "timeout_s", None),
        },
        "lineage": {
            "iteration": int(state.get("iteration", 0) or 0),
            # The index of the latest entry in DESIGN_HISTORY.json (0 for the first design), None before any.
            "design_revision": (len(state.get("design_history") or []) - 1) if state.get("design_history") else None,
            "repair": int(state.get("exec_reflect_iter", 0) or 0),
        },
    }


def append(fi_dir: Path, name: str, record: dict[str, Any]) -> bool:
    """Append one record (with the time) to ``fi_dir/name``; False when it could not be written."""
    try:
        fi_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"at": time.time(), **record}, sort_keys=True, default=str, ensure_ascii=False)
        with (fi_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return True
    except (OSError, TypeError, ValueError):
        return False


def read(fi_dir: Path, name: str) -> list[dict[str, Any]]:
    """Every record in ``fi_dir/name`` that parses; a torn line is skipped."""
    out: list[dict[str, Any]] = []
    try:
        lines = (fi_dir / name).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def run_outcome(*, returncode: int | None, has_result: bool, manifest_status: str = "", oracle_status: str = "",
                protocol_status: str = "") -> str | None:
    """What one run of the experiment came to; ``None`` for a run that has not finished (a cluster job still
    running), which is recorded when it does."""
    if manifest_status == "pending":
        return None
    if oracle_status in ("failed", "stopped", "warned"):
        return "oracle_failure"
    if manifest_status in ("stopped", "repairing", "warned") or protocol_status in ("stopped", "warned"):
        return "protocol_mismatch"
    if returncode not in (0, None) or not has_result:
        return "process_error"
    return "inconclusive"


def quest_outcome(state: dict[str, Any], evidence: dict[str, Any] | None, *, reviewer_accepted: bool) -> str:
    """What the quest came to: ``process_error`` when nothing ran, ``accepted`` when a reviewer accepted the result
    (``reviewer_accepted`` is the engine's own test of that, the one the automatic accept and the write-back use), and
    ``inconclusive`` otherwise."""
    status = str((evidence or {}).get("status") or "")
    if status == "not_executed" or not state.get("result_json"):
        return "process_error"
    return "accepted" if reviewer_accepted else "inconclusive"
