"""The code that runs is the code the checks were made for.

The checks made before a run (typed-in results, one test per equation, the known-answer checks, the code review) are made
on one version of ``code/``. A later rewrite (a figure repair, a person's edit) could change what the run computes without
any of them seeing it. When the checks pass, FI writes a receipt: a hash of every file the study's code is made of (the
scripts, the model's package and its helpers, a parameter file) and a second hash of the same files with their drawing taken
out (:mod:`core.plot_scope`). Before the run it reads the code again:

* the same hash: nothing changed;
* a different hash but the same computation: only the drawing (or a comment, or spacing) changed; the receipt is renewed;
* a different computation: the checks are made again on this version, or the run does not start.

Files FI writes itself (the run script, the figure redraws, the study file, notes) are not part of the study's code.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from . import plot_scope

RECEIPT = Path(".fi") / "gate_receipt.json"
#: FI's own files at the top of ``code/``: written by FI from the plan and the run, not the study's code.
GENERATED = frozenset({"web_plots.py", "replot_figures.py", "replot_figures.json", "replot_layout.py", "run.py", "submit.py",
                       "fi_search.py", "study.json"})
SKIPPED_DIRS = ("__pycache__", ".git", ".pytest_cache")
#: File types a script can name by a literal path. A literal counts only when it resolves to an existing file.
REFERENCE_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".csv", ".txt", ".npy", ".npz", ".py", ".ini", ".cfg")
#: Folders under quest_root that hold outputs and records, not inputs: a literal naming them is not part of the version.
OUTPUT_DIRS = (".fi", "raw", "data/results", "figures", "paper")


def _files(code: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(p for p in code.rglob("*") if p.is_file()):
        rel = path.relative_to(code)
        if (len(rel.parts) == 1 and path.name in GENERATED) or path.suffix.lower() in (".md", ".pyc") \
                or any(part in SKIPPED_DIRS for part in rel.parts):
            continue
        out.append(path)
    return out


def _referenced(code: Path, root: Path) -> dict[str, Path]:
    """Files outside ``code/`` (inside ``root``) that a script names by a literal path, keyed by their path relative to
    ``root``. The scripts are read with :mod:`ast`, never run. Outputs and records are left out."""
    code_r, root_r = code.resolve(), root.resolve()
    found: dict[str, Path] = {}
    for script in _files(code):
        if script.suffix != ".py":
            continue
        try:
            tree = ast.parse(script.read_bytes().decode("utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            literal = node.value.strip()
            if not literal or len(literal) > 512 or "\0" in literal or not literal.lower().endswith(REFERENCE_SUFFIXES):
                continue
            for base in (code_r, root_r):
                try:
                    target = (base / literal).resolve()
                    if target.is_relative_to(code_r) or not target.is_relative_to(root_r) or not target.is_file():
                        continue  # a code file is hashed already; anything outside quest_root is not ours to check
                    rel = target.relative_to(root_r).as_posix()
                except (OSError, ValueError):
                    continue
                if any(rel == d or rel.startswith(d + "/") for d in OUTPUT_DIRS) \
                        or any(part in SKIPPED_DIRS for part in rel.split("/")):
                    continue
                found.setdefault(rel, target)
    return found


def _absorb(full: Any, comp: Any, key: str, path: Path) -> bool:
    """Add one file to both digests. False when it cannot be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    full.update(key.encode("utf-8") + b"\0" + raw + b"\0")
    mark = plot_scope.computation_print(raw.decode("utf-8", errors="replace")) if path.suffix == ".py" else None
    comp.update(key.encode("utf-8") + b"\0" + (mark.encode("utf-8") if mark is not None else raw) + b"\0")
    return True


def fingerprint(quest_root: Path) -> dict[str, Any]:
    """``{"full": hash of every file, "computation": the same with each script's drawing taken out, "files": [names]}``.

    The files are those under ``code/``, then the files outside it that a script names by a literal path (sorted, keyed by
    their path relative to ``quest_root``): a parameter file or a helper is part of the checked version."""
    root = Path(quest_root)
    code = root / "code"
    full, comp = hashlib.sha256(), hashlib.sha256()
    names: list[str] = []
    if code.is_dir():
        for path in _files(code):
            rel = path.relative_to(code).as_posix()
            names.append(rel)
            if not _absorb(full, comp, rel, path):
                return {"full": "", "computation": "", "files": names}
        for rel, path in sorted(_referenced(code, root).items()):
            names.append(rel)
            if not _absorb(full, comp, rel, path):
                return {"full": "", "computation": "", "files": names}
    return {"full": full.hexdigest(), "computation": comp.hexdigest(), "files": names}


def write(quest_root: Path) -> dict[str, Any]:
    """Record the code as it is now as the version the checks were made for."""
    now = fingerprint(quest_root)
    path = Path(quest_root) / RECEIPT
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(now, indent=1), encoding="utf-8")
    except OSError:
        pass
    return now


def read(quest_root: Path) -> dict[str, Any]:
    try:
        value = json.loads((Path(quest_root) / RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


SAME, DRAWING_ONLY, CHANGED, NONE = "same", "drawing_only", "changed", "none"


def verdict(quest_root: Path) -> str:
    """How the code compares with the receipt: ``same``, ``drawing_only`` (the receipt is renewed), ``changed`` (the checks
    were not made on this computation) or ``none`` (there is no receipt, or the code cannot be read)."""
    receipt, now = read(quest_root), fingerprint(quest_root)
    if not receipt.get("full") or not now.get("full"):
        return NONE
    if receipt["full"] == now["full"]:
        return SAME
    if receipt.get("computation") == now["computation"]:
        write(quest_root)
        return DRAWING_ONLY
    return CHANGED
