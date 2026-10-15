"""What FI measured about how the run went, kept in one record: ``.fi/run_facts.json``.

Two facts the paper states about the run were wrong when they were read from the wrong place:

* **How long the experiment ran.** ``state["exec_result"]["duration_s"]`` is the time of the LAST execution, and the last
  execution is often only the analysis run again on the trials FI already stored (2.3 s for an experiment that took 223 s).
  The time that counts is the wall time of the run that produced the stored trials. :func:`record_trials` keeps it
  where the trials are run (``core/trial_runner.py``), and an analysis run again on stored trials does not touch it.
  The sum of the per-trial times in the ledger is kept beside it as a cross-check; it is smaller than the wall time (every
  setting runs in its own process) and is not what the paper is checked against.
* **Which seed the code used.** FI gives each trial a seed derived from the study's base seed, but a simulation that never
  uses the ``seed`` its ``run_trial`` is given (it builds its generators from a number typed into the code) is not run
  with that seed. :func:`seed_verdict` reads the code (never runs it): whether ``run_trial`` uses its seed argument, and
  every seed written as a number in the code, with where. The paper's statement of the seed is held to it
  (``core/stat_claims.py``), and the writer is told it (``core/paper_facts.py``).

Nothing here asks a model.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

PATH = Path(".fi") / "run_facts.json"

#: Calls that build or seed a random generator; their first argument is a seed.
_SEEDERS = frozenset({"default_rng", "RandomState", "seed", "manual_seed", "PRNGKey", "Random", "SeedSequence"})
_SKIP_PARTS = {"__pycache__", "site-packages", "tests", "test"}
#: A name that holds a seed (``seed``, ``BASE_SEED``, ``random_state``), not one that merely contains the word.
_SEED_NAME = re.compile(r"^(?:(?:base|random|rng|global|master|root)_)?seed$|^random_state$", re.I)


def read(quest_root: Path | str) -> dict[str, Any]:
    """The record, or ``{}`` when there is none or it cannot be read."""
    try:
        data = json.loads((Path(quest_root) / PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(quest_root: Path | str, **changes: Any) -> None:
    path = Path(quest_root) / PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**read(quest_root), **changes}, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError:
        pass  # a record FI cannot keep leaves the paper without the fact, never stops the quest


def record_trials(quest_root: Path | str, *, key: str, wall_s: float | None, trials: int, cells: int,
                  trial_sum_s: float | None) -> None:
    """Keep the wall time of a run of the trials (``wall_s``: seconds from before the first setting to after the last;
    ``None`` when it is not known, as for a cluster job), how many trials and settings it ran, and the sum of the
    per-trial times in the ledger. ``key`` is the run's own key (``trial_runner._run_key``)."""
    _write(quest_root, trials_run={
        "key": key, "wall_s": None if wall_s is None else round(float(wall_s), 2), "trials": int(trials), "cells": int(cells),
        "trial_sum_s": None if trial_sum_s is None else round(float(trial_sum_s), 2)})


def run_seconds(quest_root: Path | str, fallback: Any = None) -> float | None:
    """The measured wall time of the run that produced the stored trials, in seconds. With no record of one (a study
    that does not run FI's trials: one script, run once) ``fallback``, the time of the last execution, if it is a number."""
    run = read(quest_root).get("trials_run")
    wall = run.get("wall_s") if isinstance(run, dict) else None
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and 0 <= wall < float("inf"):
        return float(wall)
    if isinstance(run, dict):
        return None  # FI ran trials and does not know how long they took (a cluster job): the last execution is not it
    if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) and 0 <= fallback < float("inf"):
        return float(fallback)
    return None


# ---- the seed ------------------------------------------------------------------------------------------------------


def _code_files(code_dir: Path) -> list[Path]:
    if not code_dir.is_dir():
        return []
    out = []
    for path in sorted(code_dir.rglob("*.py")):
        parts = path.relative_to(code_dir).parts
        if any(p.startswith(".") or p in _SKIP_PARTS or p.startswith("test_") for p in parts):
            continue
        out.append(path)
    return out


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _int_of(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _int_of(node.operand)
        return None if inner is None else -inner
    return None


def _literal_seeds(tree: ast.AST, rel: str) -> list[tuple[str, int]]:
    """``("code/x.py:107", 20261015)`` for each seed written as a number in the code: the first argument of a call that
    seeds a generator (or the first item of a list given to ``SeedSequence``), a third argument of a call to
    ``run_trial`` and a name ``..._SEED = N``."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        where = f"{rel}:{line}"
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _SEEDERS and node.args:
                first = node.args[0]
                if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                    first = first.elts[0]
                if (n := _int_of(first)) is not None:
                    found.append((where, n))
            elif name == "run_trial" and len(node.args) >= 3 and (n := _int_of(node.args[2])) is not None:
                found.append((where, n))
            for kw in node.keywords:
                if kw.arg and _SEED_NAME.match(kw.arg) and (n := _int_of(kw.value)) is not None:
                    found.append((where, n))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _SEED_NAME.match(target.id) and (n := _int_of(node.value)) is not None:
                    found.append((where, n))
    return found


def _ignores_its_seed(tree: ast.AST) -> tuple[bool, str]:
    """``(True, the parameter's name)`` when the module's top-level ``run_trial(cell, trial_id, seed)`` never reads its seed
    parameter (the third one, or one named ``seed``); ``(False, "")`` otherwise, including when there is none."""
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_trial":
            params = [a.arg for a in [*node.args.posonlyargs, *node.args.args]]
            name = next((p for p in params if p in ("seed", "rng_seed", "random_seed")), params[2] if len(params) >= 3 else "")
            if not name:
                return False, ""
            used = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)
                       for stmt in node.body for n in ast.walk(stmt))
            return (not used), name
    return False, ""


def seed_verdict(quest_root: Path | str) -> dict[str, Any]:
    """What the code says about the seed, read and not run: ``{"fi_seed_reaches": True | False | None, "why": str,
    "code_seeds": [{"where": "code/simulate.py:107", "value": 20261015}, ...]}``. ``fi_seed_reaches`` is ``False`` when
    ``simulate.py``'s ``run_trial`` never reads the seed FI gives it, ``True`` when it does, ``None`` when the code has no
    ``run_trial`` to read (a study that is not run trial by trial)."""
    root = Path(quest_root)
    code_dir = root / "code"
    literals: list[dict[str, Any]] = []
    reaches: bool | None = None
    why = ""
    for path in _code_files(code_dir):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        literals += [{"where": w, "value": v} for w, v in _literal_seeds(tree, rel)]
        if path.name == "simulate.py" and path.parent == code_dir:
            ignored, name = _ignores_its_seed(tree)
            if name:
                reaches = not ignored
                if ignored:
                    why = f"run_trial takes `{name}` but never uses it, so the seed FI gives each trial does not reach the code"
    return {"fi_seed_reaches": reaches, "why": why, "code_seeds": literals}


def record_seed(quest_root: Path | str) -> dict[str, Any]:
    """Read the code (:func:`seed_verdict`) and keep the verdict in the record; returns it."""
    verdict = seed_verdict(quest_root)
    _write(quest_root, seed=verdict)
    return verdict
