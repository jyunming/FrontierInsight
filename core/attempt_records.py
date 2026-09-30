"""What a quest tried, and under which conditions: the records the explore/confirm work is built on.

Two append-only JSONL files in the quest's ``.fi/`` folder, written as the quest runs and read by nothing that decides a
route (this is the "records only" first phase; a later phase reads them in shadow mode, then for real):

- ``attempts.jsonl``: one line per run of the experiment (an ``outcome`` from :data:`OUTCOMES` and the hashes of the
  scripts it ran), one when the quest stops for a check it failed, and one when the quest ends or fails (four separate
  fields: :func:`quest_status`), each with the ``context`` it happened under (:func:`context_fingerprint`). A failure is
  a fact about an attempt under a context, not about a method in general: a different model, protocol, program or
  environment is a different context.
- ``branch_ledger.jsonl``: the choices along the way, with what was chosen from what: the candidate ideas and how one was
  picked, each revision of the design and why, each repair of a script.

Every line carries ``schema`` (:data:`SCHEMA`), its own ``record_id`` and the ``quest_id``. A context says whether it
is ``complete`` (every part of it could be worked out, the inputs hashed whole); only a complete context of the current
schema may ever be treated as the same conditions as another -- an older record or an incomplete one is information
only.

Kept apart from the accepted evidence (nothing here is written to the knowledge base, and nothing here supports a
claim). Every write is best-effort: a record that cannot be built or written never touches the quest, and the quest's
last line counts the ones that could not be written.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import time
import uuid
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
#: One line per model call the quest made (never the prompt or the answer: their hashes). See :func:`model_call_row`.
MODEL_CALLS = "model_calls.jsonl"
MODEL_CALLS_LOST = "model_calls.lost"
#: Calls made after the quest sealed its record (the output generators, a later ``--emit``): kept apart, since the seal
#: names :data:`MODEL_CALLS` as it was when the research record was closed.
MODEL_CALLS_AFTER_SEAL = "model_calls.after_seal.jsonl"
#: Present while the quest's record is sealed (Engine._seal_trace writes it; a new run of the quest removes it).
MODEL_CALLS_CLOSED = "model_calls.closed"
#: Version 2 added record ids, the scripts' hashes, the question, the policy, the models each step was answered by, the
#: inputs' completeness and the quest line's four fields. Version 3: code is hashed whole, a context names its
#: ``context_kind``, and a finished quest's context needs its code, environment and protocol. Version 1 lines carry no
#: ``schema``. Version 4: every model call is a line of :data:`MODEL_CALLS`, and a finished quest's context is compared
#: with it (``model_calls``); after a run and at the end, files are read again rather than taken from a cache.
SCHEMA = 5

#: The quest line's fields (:func:`quest_status`). ``execution_status``: how far it ran. ``review_status``: what the
#: review said. ``evidence_status``: the evidence level (core/evidence.py). ``claim_outcome``: whether the result
#: supports the hypothesis -- not derived yet (the analysis does not state it), so ``None``.
EXECUTION = ("completed", "no_result", "stopped", "crashed", "no_experiment_by_design", "data_analysis")
#: ``rejected`` is the person's (``person``), never a reviewer's: a reviewer answers accept or revise.
REVIEW = ("accepted", "revise", "rejected", "unavailable", "none")

#: Inputs beyond this many files are not all hashed, and the context says it is incomplete.
_INPUT_FILES_LIMIT = 2000

#: A stop for one of these checks is an attempt that failed it. Any other stop is not an outcome: the plan waiting for
#: you, papers asked for, a protocol change waiting for approval, or a check that could not be asked (an outage is not
#: a fact about the attempt).
STOP_OUTCOMES = {
    "oracle": "oracle_failure",
    "manifest": "protocol_mismatch", "protocol": "protocol_mismatch", "equation_labels": "protocol_mismatch",
    "split": "process_error", "numeric": "process_error", "replicate_seed_unrepairable": "process_error",
}

#: Above this size a file is identified by its size and its first and last MiB, not read whole.
_WHOLE_FILE_LIMIT = 64 * 1024 * 1024
_CHUNK = 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path, *, whole: bool = False) -> str | None:
    """A hash of the file, read in chunks; a file over :data:`_WHOLE_FILE_LIMIT` is hashed by its size and its first
    and last MiB unless ``whole`` (code is always read whole: two programs must never share a hash). ``None`` when it
    cannot be read."""
    try:
        h = hashlib.sha256()
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _WHOLE_FILE_LIMIT and not whole:
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
def _fi_source_sha(repo: Path) -> str | None:
    """A hash over FI's own source (``core/**/*.py``, ``generation/**/*.py``, ``agents/*.md``) under ``repo``: what this
    process loaded, the same for an installed copy and a checkout, and different for any edit, committed or not."""
    h = hashlib.sha256()
    found = False
    for pattern in ("core/**/*.py", "generation/**/*.py", "agents/*.md"):
        for path in sorted(repo.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            digest = _file_sha(path)
            if digest is None:
                return None
            h.update(path.relative_to(repo).as_posix().encode("utf-8"))
            h.update(digest.encode("ascii"))
            found = True
    return h.hexdigest() if found else None


def _fi_version(repo: Path) -> dict[str, Any] | None:
    """FI's identity: a hash of its loaded source (:func:`_fi_source_sha`), plus the git commit and whether the checkout
    had changes when ``repo`` is the top of a git checkout. ``None`` when not even the source can be read."""
    source = _fi_source_sha(repo)
    if source is None:
        return None
    info = _git_version(repo)
    return {"source_sha256": source, **(info or {"commit": None, "dirty": None})}


@functools.lru_cache(maxsize=4)
def _git_version(repo: Path) -> dict[str, Any] | None:
    """The git commit, and whether the checkout had changes, when ``repo`` is the top of a git checkout; ``None``
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


def _digest(path: Path, cache: dict | None, *, whole: bool = False) -> tuple[str | None, int | None]:
    """``path``'s hash and size; with ``cache`` (the engine's own, keyed by path, size and modification time) a file
    unchanged since it was last hashed is not read again."""
    try:
        st = path.stat()
    except OSError:
        return None, None
    key = (str(path), st.st_size, st.st_mtime_ns, whole)
    if cache is not None and key in cache:
        return cache[key], st.st_size
    digest = _file_sha(path, whole=whole)
    if cache is not None and digest is not None:
        cache[key] = digest
    return digest, st.st_size


def _folder_manifest(folder: Path, limit: int = _INPUT_FILES_LIMIT, cache: dict | None = None,
                     ) -> dict[str, Any] | None:
    """A hash over the names and contents of the files in ``folder`` (the inputs or the data a quest was given), with
    how many files and bytes there are and whether the hash covers them all: ``complete`` is false when there are more
    than ``limit`` files, a file is too large to read whole (:data:`_WHOLE_FILE_LIMIT`) or one could not be read. One
    aggregate hash, not a per-file list: it says whether the inputs differ, not which one."""
    if not folder.is_dir():
        return None
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and not (folder.name == "data" and p.relative_to(folder).parts[:1] == ("results",))
    )
    h = hashlib.sha256()
    total, complete = 0, len(files) <= limit
    for n, path in enumerate(files):
        if n >= limit:
            try:
                total += path.stat().st_size
            except OSError:
                pass
            continue
        digest, size = _digest(path, cache)
        if size is None:
            complete = False
            continue
        total += size
        if size > _WHOLE_FILE_LIMIT:
            complete = False
        if digest is None:
            complete = False
        h.update(path.relative_to(folder).as_posix().encode("utf-8"))
        h.update((digest or "").encode("ascii"))
    return {"sha256": h.hexdigest(), "files": len(files), "bytes": total, "complete": complete}


def _generated_project_files(quest_root: Path) -> set[str]:
    """The README, requirements, run.py and fi_search.py (FI's own search for the best design) FI wrote into ``code/``
    (core/code_project.py) and a person has not edited: they describe or repeat what FI did, they are not the code an
    attempt ran."""
    try:
        record = json.loads((quest_root / ".fi" / "code_project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(record, dict):
        return set()
    out: set[str] = set()
    for name in ("README.md", "requirements.txt", "run.py", "fi_search.py"):
        try:
            text = (quest_root / "code" / name).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if record.get(name) == hashlib.sha256(text.encode("utf-8")).hexdigest():
            out.add(name)
    return out


def script_hashes(quest_root: Path, cache: dict | None = None) -> dict[str, str]:
    """The hash of every file in the quest's ``code/`` folder, subfolders included (the experiment, the simulation, a
    cluster submit script, whatever they import or read from there), by relative path, each read whole: the code an
    attempt ran. A cluster job's code is also hashed when it is submitted and compared when its results are collected
    (core/trial_runner.py); a change in between is named in ``.fi/trials/cluster.json`` and leaves the context
    incomplete."""
    code = quest_root / "code"
    out: dict[str, str] = {}
    generated = _generated_project_files(quest_root)
    if code.is_dir():
        for path in sorted(p for p in code.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts):
            if path.parent == code and (path.name in generated or path.name == "CHANGELOG.md"):
                continue
            digest, _ = _digest(path, cache, whole=True)
            if digest:
                out[path.relative_to(code).as_posix()] = digest
    return out


def _config_sha(config: Any) -> str | None:
    """A hash of the whole effective configuration, without where the outputs go and the transport wiring: every check
    mode, the sandbox, the split rules -- also for a config that has no approved-settings record."""
    try:
        data = config.model_dump(mode="json", exclude={"output": {"output_dir"}, "provider": {"extra"}})
    except Exception:  # noqa: BLE001 -- not a pydantic config (a test's stand-in)
        return None
    return _json_sha(data)


def _environment_sha(needs: Path) -> str | None:
    """A hash of what the environment is (Python, platform, sandbox, isolation, packages), not of where it lives or
    when it was recorded, so two quests with the same packages match."""
    record = _read_json(needs / "ENVIRONMENT.json")
    if not isinstance(record, dict):
        return None
    keep = ("python", "platform", "sandbox", "shared_interpreter", "system_site_packages", "isolated", "packages")
    return _json_sha({k: record.get(k) for k in keep})


#: When a context is worked out. ``in_progress``: the quest stopped part-way (a check failed, it paused). ``after_run``:
#: an experiment has just run. ``quest_end``: the quest finished; an experiment it was meant to run must have left its
#: code, environment and protocol, unless it runs none by design (a survey, an analysis of data that already exists).
CONTEXT_KINDS = ("in_progress", "after_run", "quest_end")


def _experiment_expected(config: Any, state: dict[str, Any]) -> bool:
    engine = getattr(config, "engine", None)
    return not (state.get("survey_mode_resolved") or state.get("no_simulation_resolved")
                or getattr(engine, "survey_mode", False))


def _cluster_code_changes(quest_root: Path) -> list[str]:
    record = _read_json(quest_root / ".fi" / "trials" / "cluster.json")
    changed = record.get("code_changed_while_queued") if isinstance(record, dict) else None
    return [str(c) for c in changed] if isinstance(changed, list) else []


@functools.lru_cache(maxsize=1)
def _harness_sha() -> str | None:
    try:
        from .trial_runner import HARNESS_SOURCE
    except Exception:  # noqa: BLE001
        return None
    return _sha(HARNESS_SOURCE.encode("utf-8"))


def context_fingerprint(config: Any, quest_root: Path, state: dict[str, Any], *, kind: str,
                        prompts: dict[str, Any] | None = None, fi_repo: Path | None = None,
                        models_used: dict[str, Any] | None = None, cache: dict | None = None,
                        partial: bool = False, model_call_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """The conditions an attempt ran under, as far as the quest knows them. Two attempts share a context only when
    these match: the model and the settings that change its answers, the prompts it was given, FI's own version, the
    selected skills, the environment and its packages, the inputs, the protocol and its metric definitions, the budget,
    and where in the quest's lineage the attempt sits. A field FI could not work out is ``None``. ``kind`` is one of
    :data:`CONTEXT_KINDS`: what must be there for the context to be complete depends on it."""
    if kind not in CONTEXT_KINDS:
        raise ValueError(f"kind must be one of {CONTEXT_KINDS}; got {kind!r}")
    if kind in ("after_run", "quest_end"):
        # What ran and what the quest ended with are read again, whole: a file's size and modification time can be
        # kept while its content changes, so a cache keyed by them is no proof of what is there.
        cache = None
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
    inputs = _folder_manifest(quest_root / "inputs", cache=cache)
    data = _folder_manifest(quest_root / "data", cache=cache)
    code = script_hashes(quest_root, cache)
    engine_cfg = engine
    question = {
        "topic_sha256": _json_sha(str(state.get("topic") or getattr(config, "topic", "") or "")),
        "hypothesis_sha256": _json_sha(str(design.get("hypothesis") or "")) if design else None,
        "design_sha256": _json_sha(design) if design else None,
    }
    policy = {
        "result_use": getattr(config, "effective_result_use", None) or getattr(config, "result_use", None) or None,
        "rigor_profile": getattr(config, "rigor_profile", None),
        "review_panel": list(getattr(engine_cfg, "review_panel", []) or []),
        "approved_plan_sha256": _file_sha(quest_root / ".fi" / "approved_plan.json"),
        "config_sha256": _config_sha(config),
    }
    fi = _fi_version(fi_repo) if fi_repo is not None else None
    environment_sha = _environment_sha(needs)
    # What keeps this context from ever counting as the same conditions as another: it is information only.
    missing: list[str] = []
    if partial:
        missing.append("worked out without the quest's state (it failed before any attempt was recorded)")
    if inputs is not None and not inputs.get("complete"):
        missing.append("inputs not all hashed")
    if data is not None and not data.get("complete"):
        missing.append("data not all hashed")
    if fi is None:
        missing.append("FI's source could not be read")
    if policy["config_sha256"] is None:
        missing.append("the configuration could not be hashed")
    if not models_used:
        missing.append("no model call recorded yet")
    if kind == "after_run" or (kind == "quest_end" and _experiment_expected(config, state)):
        if not code:
            missing.append("no script in code/")
        if environment_sha is None:
            missing.append("no environment record")
        if protocol_sha is None:
            missing.append("no protocol")
    calls = None
    if kind == "quest_end":
        call_gaps, calls = model_call_gaps(quest_root, models_used or {},
                                           model_call_counts if model_call_counts is not None
                                           else state.get("model_call_counts"))
        missing.extend(call_gaps)
    changed_while_queued = _cluster_code_changes(quest_root)
    if changed_while_queued:
        missing.append("the code changed while the cluster job was queued: " + ", ".join(changed_while_queued[:10]))
    return {
        "context_kind": kind,
        "provider": getattr(provider, "name", "") or None,
        "model": getattr(provider, "model", "") or None,
        "settings": {
            "node_models": dict(getattr(provider, "node_models", {}) or {}),
            "node_max_tokens": dict(getattr(provider, "node_max_tokens", {}) or {}),
            "reasoning_effort": getattr(provider, "reasoning_effort", None) or None,
            "fixed_temperature": getattr(provider, "fixed_temperature", None),
            "base_url": getattr(provider, "base_url", None),
            "fallback": list(getattr(provider, "fallback", []) or []),
            "node_ensemble": sorted((getattr(provider, "node_ensemble", None) or {}).keys()),
        },
        "prompts_sha256": _json_sha({k: getattr(v, "template", str(v)) for k, v in (prompts or {}).items()}),
        # Each prompt on its own, so a decision can be compared on the prompt it was made with.
        "prompt_shas": {k: _json_sha(getattr(v, "template", str(v))) for k, v in (prompts or {}).items()},
        # The packages the design asked for, and FI's trial harness (what runs each trial), by hash.
        "deps_sha256": _json_sha(sorted(str(d) for d in (state.get("deps") or []))) if state.get("deps") else None,
        "harness_sha256": _harness_sha(),
        # What the design measures and over which settings, without its wording.
        "design_features": {
            "metrics": sorted(str(m.get("id")) for m in (protocol.get("metrics") or []) if isinstance(m, dict)),
            "grid_axes": sorted(str(k) for k in (protocol.get("grid") or {})) if isinstance(protocol.get("grid"), dict)
            else [],
        },
        "fi": fi,
        "skills": sorted(str(s) for s in (state.get("selected_skills") or [])),
        "environment_sha256": environment_sha,
        "dependency_lock_sha256": _file_sha(quest_root / ".fi" / "requirements.lock.txt"),
        "inputs": inputs,
        "data": data,
        "code": code,
        "question": question,
        "policy": policy,
        # For each step asked through the engine's one-call path, who answered its last call (provider, model, the prompt's
        # and reply's hashes). Every call, of every step (the source router, ensembles and the output generators too),
        # is a line in .fi/model_calls.jsonl.
        "models_used": dict(models_used or {}),
        # At the quest's end: the hash and line count of the quest's record of every model call (.fi/model_calls.jsonl).
        **({"model_calls": calls} if calls is not None else {}),
        "missing": missing,
        "complete": not missing,
        "protocol_sha256": protocol_sha,
        "metric_specs_sha256": _json_sha(protocol.get("metrics")) if protocol.get("metrics") else None,
        "budget": {
            "max_iterations": getattr(engine, "max_iterations", None),
            "exec_reflect_max_iterations": getattr(engine, "exec_reflect_max_iterations", None),
            "execute_replicates": getattr(engine, "execute_replicates", None),
            "runs_per_setting": protocol.get("runs_per_setting"),
            "timeout_s": getattr(execution, "timeout_s", None),
            "replicate_seed_stride": getattr(engine, "replicate_seed_stride", None),
        },
        "lineage": {
            "iteration": int(state.get("iteration", 0) or 0),
            # The index of the latest entry in DESIGN_HISTORY.json (0 for the first design), None before any.
            "design_revision": (len(state.get("design_history") or []) - 1) if state.get("design_history") else None,
            "repair": int(state.get("exec_reflect_iter", 0) or 0),
        },
    }


def append(fi_dir: Path, name: str, record: dict[str, Any]) -> str | None:
    """Append one record (with the time, :data:`SCHEMA` and a new ``record_id``) to ``fi_dir/name``; its id, or
    ``None`` when it could not be written."""
    try:
        fi_dir.mkdir(parents=True, exist_ok=True)
        record_id = str(record.get("record_id") or uuid.uuid4().hex)
        line = json.dumps({"at": time.time(), "schema": SCHEMA, **record, "record_id": record_id},
                          sort_keys=True, default=str, ensure_ascii=False)
        with (fi_dir / name).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record_id
    except (OSError, TypeError, ValueError):
        return None


LOST = "attempts.lost"


def count_lost(fi_dir: Path, name: str = LOST) -> None:
    """Add one to the count of records that could not be built or written, kept in ``fi_dir`` so a later run of the
    quest (after a pause) still sees it. Best-effort."""
    try:
        (fi_dir / name).write_text(str(lost(fi_dir, name) + 1), encoding="utf-8")
    except OSError:
        pass


def lost(fi_dir: Path, name: str = LOST) -> int:
    """How many records of this quest could not be built or written, over every run of it."""
    try:
        return int((fi_dir / name).read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def _text_sha(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return _json_sha(value)
    return _sha(value.encode("utf-8"))


def prompt_sha(messages: Any) -> str | None:
    """The hash of a call's prompt, the same wherever it is written (a line of :data:`MODEL_CALLS`, a step's record,
    a trace event): the SHA-256 of the messages' text, joined by a blank line (a one-message prompt: of its text)."""
    if messages is None:
        return None
    if isinstance(messages, str):
        return _text_sha(messages)
    if isinstance(messages, (list, tuple)) and all(isinstance(m, dict) for m in messages):
        return _text_sha("\n\n".join(str(m.get("content") if m.get("content") is not None else "") for m in messages))
    return _text_sha(messages)


def next_attempt(fi_dir: Path, node: str) -> int:
    """The number the next line of ``node`` in the file calls go to takes (1 for its first)."""
    name = MODEL_CALLS_AFTER_SEAL if (fi_dir / MODEL_CALLS_CLOSED).exists() else MODEL_CALLS
    return 1 + sum(1 for r in read(fi_dir, name) if r.get("node") == node)


def model_call_row(*, node: str, attempt: int, served: dict[str, Any] | None, requested_model: str | None,
                   reports_model: bool, messages: Any, response: Any, outcome: str = "ok",
                   usage: dict[str, Any] | None = None, call_id: str | None = None) -> dict[str, Any]:
    """One line of :data:`MODEL_CALLS`: which step asked (``node``, the ``attempt``-th call under it), which model was
    asked for and which answered (``served_model``, ``reported`` when the connection named it, ``vendor`` when it
    did, ``fallback`` when a fallback provider took the call), hashes of the prompt and the answer (never their
    text), the ``outcome`` (``ok``, ``truncated`` for an answer cut off at its output limit, ``content_filtered``, or
    the error's class), the ``finish_reason`` when the connection gave one, and the token counts when it gave them.
    ``reports_model``: the connection is one that names the model that answered (so ``reported`` false is a gap)."""
    served = dict(served or {})
    return {
        "call_id": call_id or uuid.uuid4().hex,
        "node": node or "",
        "attempt": int(attempt),
        "provider": served.get("provider"),
        "requested_model": requested_model,
        "served_model": served.get("model"),
        **({"vendor": served["vendor"]} if served.get("vendor") else {}),
        "reported": bool(served.get("reported")),
        "reports_model": bool(reports_model),
        "fallback": bool(served.get("fallback")),
        "prompt_sha256": prompt_sha(messages),
        "response_sha256": _text_sha(response),
        "outcome": outcome,
        # Why the answer ended, when the connection said (``stop``, ``length`` for one cut off at its limit, ...).
        **({"finish_reason": served["finish_reason"]} if served.get("finish_reason") else {}),
        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
        if isinstance(usage, dict) else None,
    }


def append_model_call(fi_dir: Path, quest_id: str, row: dict[str, Any]) -> bool:
    """Append one :func:`model_call_row` to the quest's :data:`MODEL_CALLS` (to :data:`MODEL_CALLS_AFTER_SEAL` once the
    quest's record is sealed); a line that cannot be written is counted in :data:`MODEL_CALLS_LOST` (which the quest's
    end reads). Never raises."""
    try:
        name = MODEL_CALLS_AFTER_SEAL if (fi_dir / MODEL_CALLS_CLOSED).exists() else MODEL_CALLS
        if append(fi_dir, name, {"quest_id": quest_id, **row}):
            return True
    except Exception:  # noqa: BLE001 -- a record never touches the quest
        pass
    count_lost(fi_dir, MODEL_CALLS_LOST)
    return False


def model_call_gaps(quest_root: Path, models_used: dict[str, Any],
                    counts: dict[str, int] | None = None) -> tuple[list[str], dict[str, Any]]:
    """What keeps the quest's record of its model calls (:data:`MODEL_CALLS`) from being complete, and that record's
    hash and line count. ``counts``: the engine's own tally of the calls it made, per step (kept in the quest's state,
    so a resumed quest keeps it): a step with fewer lines than calls lost lines. Also: lines that could not be written;
    a step's last call (``models_used``) not among the lines; answered calls on a connection that names the answering
    model whose line names none."""
    fi_dir = Path(quest_root) / ".fi"
    rows = read(fi_dir, MODEL_CALLS)
    try:
        data = (fi_dir / MODEL_CALLS).read_bytes()
        summary: dict[str, Any] = {"sha256": _sha(data), "lines": data.count(b"\n")}
    except OSError:
        summary = {"sha256": None, "lines": 0}
    summary["not_written"] = lost(fi_dir, MODEL_CALLS_LOST)
    gaps: list[str] = []
    if summary["not_written"]:
        gaps.append(f"{summary['not_written']} model call record(s) could not be written")
    per_node: dict[str, int] = {}
    for r in rows:
        per_node[str(r.get("node") or "")] = per_node.get(str(r.get("node") or ""), 0) + 1
    short = sorted(f"{node} ({per_node.get(node, 0)} of {n})" for node, n in (counts or {}).items()
                   if per_node.get(node, 0) < int(n or 0))
    if short:
        gaps.append("the record of model calls has fewer lines than the calls made: " + ", ".join(short[:10]))
    answered = {(str(r.get("node") or ""), r.get("response_sha256")) for r in rows}
    unlisted = sorted(node for node, rec in (models_used or {}).items()
                      if isinstance(rec, dict) and rec.get("response_hash")
                      and (node, rec.get("response_hash")) not in answered)
    if unlisted:
        gaps.append("model calls not in the call record: the last call of " + ", ".join(unlisted[:10]))
    unnamed = sum(1 for r in rows if r.get("outcome") == "ok" and r.get("reports_model")
                  and not r.get("fallback") and not r.get("reported"))
    if unnamed:
        gaps.append(f"{unnamed} answered model call(s) on a connection that names the answering model did not name it")
    return gaps, summary


def file_digests(fi_dir: Path) -> dict[str, Any]:
    """The hash and line count of the record files, for the audit trace to anchor them at the quest's end."""
    out: dict[str, Any] = {}
    for name in (ATTEMPTS, LEDGER, MODEL_CALLS):
        path = fi_dir / name
        try:
            data = path.read_bytes()
        except OSError:
            continue
        out[name] = {"sha256": _sha(data), "lines": data.count(b"\n")}
    return out


def last_id(fi_dir: Path, name: str, kinds: tuple[str, ...] = ()) -> str | None:
    """The ``record_id`` of the last record in ``fi_dir/name`` (of one of ``kinds`` when given): where a resumed quest
    picks up the lineage an earlier run of it recorded."""
    for record in reversed(read(fi_dir, name)):
        if not kinds or record.get("kind") in kinds:
            return record.get("record_id")
    return None


def read(fi_dir: Path, name: str) -> list[dict[str, Any]]:
    """Every record in ``fi_dir/name`` that parses; a torn line is skipped."""
    out: list[dict[str, Any]] = []
    try:
        lines = (fi_dir / name).read_text(encoding="utf-8").split("\n")
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


def quest_status(state: dict[str, Any], evidence: dict[str, Any] | None, *, reviewer_accepted: bool,
                 no_experiment: bool, data_analysis: bool = False) -> dict[str, Any]:
    """What a finished quest came to, as four separate fields rather than one label: a literature survey that runs no
    experiment by design is not a process error, and an accepted result with evidence gaps is not accepted evidence.

    - ``execution_status``: ``no_experiment_by_design`` (``no_experiment``: a survey), ``data_analysis`` (a quest that
      analyses given data instead of simulating), ``completed`` when the experiment gave a result, ``no_result`` when
      the quest finished without one. ``crashed`` is written only by a quest that failed.
    - ``review_status``: ``accepted`` only when ``reviewer_accepted`` (the engine's own test of a real reviewer accept,
      the one the automatic accept and the write-back use); ``unavailable`` for a review that did not really run;
      else the verdict (``revise`` / ``rejected``) or ``none``.
    - ``evidence_status``: the evidence level's status, or ``unknown``.
    - ``claim_outcome``: ``None`` (not derived: the analysis does not state it)."""
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    verdict = str(review.get("verdict") or "").lower()
    if no_experiment:
        execution = "no_experiment_by_design"
    elif data_analysis:
        execution = "data_analysis"
    else:
        execution = "completed" if state.get("result_json") else "no_result"
    if reviewer_accepted:
        review_status = "accepted"
    elif review and str(review.get("status") or "ok") != "ok":
        review_status = "unavailable"
    elif verdict in ("revise", "reject", "rejected"):
        review_status = "revise" if verdict == "revise" else "rejected"
    else:
        review_status = "none"
    return {
        "execution_status": execution,
        "review_status": review_status,
        "evidence_status": str((evidence or {}).get("status") or "unknown"),
        "claim_outcome": None,
    }
