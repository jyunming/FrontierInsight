"""What a quest tried, and under which conditions: the records the explore/confirm work is built on.

Two append-only JSONL files in the quest's ``.fi/`` folder, written as the quest runs and read by nothing that decides a
route (this is the "records only" first phase; a later phase reads them in shadow mode, then for real):

- ``attempts.jsonl``: one line per run of the experiment and one when the quest ends, each with an ``outcome`` from
  :data:`OUTCOMES` and the ``context`` it happened under (:func:`context_fingerprint`). A failure is a fact about an
  attempt under a context, not about a method in general: a different model, protocol or environment is a different
  context.
- ``branch_ledger.jsonl``: the choices along the way, with what was chosen from what: the candidate ideas and how one was
  picked, each revision of the design and why, each repair of a script.

Kept apart from the accepted evidence (nothing here is written to the knowledge base, and nothing here supports a
claim). Every write is best-effort: a record that cannot be written never touches the quest.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

#: What an attempt came to. ``process_error``: it crashed or produced no result. ``protocol_mismatch``: it ran something
#: other than what the protocol fixed. ``oracle_failure``: it did not pass the known-answer checks. ``inconclusive``: it
#: ran but the result does not decide the question. ``null_result``: no effect where one was tested for.
#: ``contradicted``: the result goes against the hypothesis. ``accepted_negative`` / ``accepted_positive``: a result a
#: reviewer accepted, against or for the hypothesis.
OUTCOMES = (
    "process_error", "protocol_mismatch", "oracle_failure", "inconclusive",
    "null_result", "contradicted", "accepted_negative", "accepted_positive",
)
ATTEMPTS = "attempts.jsonl"
LEDGER = "branch_ledger.jsonl"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str | None:
    try:
        return _sha(path.read_bytes())
    except OSError:
        return None


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


@functools.lru_cache(maxsize=4)
def _fi_commit(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


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


def context_fingerprint(config: Any, quest_root: Path, state: dict[str, Any], *, prompts: dict[str, Any] | None = None,
                        skill_dirs: dict[str, Path] | None = None, fi_repo: Path | None = None) -> dict[str, Any]:
    """The conditions an attempt ran under, as far as the quest knows them. Two attempts share a context only when
    these match: the model (and its settings) that did the work, the prompts it was given, FI's own version, the skills,
    the environment and its packages, the inputs, the protocol and its metric definitions, the budget, and where in the
    quest's lineage the attempt sits."""
    provider = getattr(config, "provider", None)
    engine = getattr(config, "engine", None)
    execution = getattr(config, "execution", None)
    needs = quest_root / "needs"
    protocol = ((state.get("design") or {}).get("protocol") if isinstance(state.get("design"), dict) else None) or {}
    skills = {name: (_file_sha(skill_dirs[name] / "SKILL.md") if name in (skill_dirs or {}) else None)
              for name in sorted(state.get("selected_skills") or [])}
    return {
        "provider": getattr(provider, "name", ""),
        "model": getattr(provider, "model", "") or "",
        "node_models": dict(getattr(provider, "node_models", {}) or {}),
        "reasoning_effort": getattr(provider, "reasoning_effort", "") or "",
        "prompts_sha256": _json_sha({k: getattr(v, "template", str(v)) for k, v in (prompts or {}).items()}),
        "fi_commit": _fi_commit(fi_repo) if fi_repo is not None else "",
        "skills": skills,
        "environment_sha256": _file_sha(needs / "ENVIRONMENT.json"),
        "dependency_lock_sha256": _file_sha(quest_root / ".fi" / "requirements.lock.txt"),
        "inputs_sha256": _folder_sha(quest_root / "inputs"),
        "protocol_sha256": _file_sha(needs / "FROZEN_PROTOCOL.json") or (_json_sha(protocol) if protocol else None),
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
            "design_revision": len(state.get("design_history") or []),
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
                protocol_status: str = "") -> str:
    """What one run of the experiment came to."""
    if oracle_status in ("failed", "stopped"):
        return "oracle_failure"
    if manifest_status in ("stopped", "repairing") or protocol_status == "stopped":
        return "protocol_mismatch"
    if returncode not in (0, None) or not has_result:
        return "process_error"
    return "inconclusive"


def quest_outcome(state: dict[str, Any], evidence: dict[str, Any] | None) -> str:
    """What the quest came to, from its last run, its analysis and its review. A result is ``accepted_*`` only when a
    reviewer accepted it; otherwise an analysis that says the hypothesis failed is ``contradicted`` (or
    ``null_result`` for no effect), and anything else that ran is ``inconclusive``."""
    status = str((evidence or {}).get("status") or "")
    if status == "not_executed" or not state.get("result_json"):
        return "process_error"
    analysis = state.get("analysis") if isinstance(state.get("analysis"), dict) else {}
    said = " ".join(str(analysis.get(k) or "") for k in ("hypothesis_supported", "verdict", "conclusion")).lower()
    against = any(w in said for w in ("refuted", "not supported", "false", "contradict", "rejected"))
    null = any(w in said for w in ("no effect", "null", "no significant", "inconclusive"))
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    accepted = str(review.get("verdict") or "").lower() == "accept" and str(review.get("status") or "ok") == "ok"
    if accepted:
        return "accepted_negative" if against or null else "accepted_positive"
    if against:
        return "contradicted"
    if null:
        return "null_result"
    return "inconclusive"
