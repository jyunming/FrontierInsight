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


def _files(code: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(p for p in code.rglob("*") if p.is_file()):
        rel = path.relative_to(code)
        if (len(rel.parts) == 1 and path.name in GENERATED) or path.suffix.lower() in (".md", ".pyc") \
                or any(part in SKIPPED_DIRS for part in rel.parts):
            continue
        out.append(path)
    return out


def fingerprint(quest_root: Path) -> dict[str, Any]:
    """``{"full": hash of every file, "computation": the same with each script's drawing taken out, "files": [names]}``."""
    code = Path(quest_root) / "code"
    full, comp = hashlib.sha256(), hashlib.sha256()
    names: list[str] = []
    if code.is_dir():
        for path in _files(code):
            rel = path.relative_to(code).as_posix()
            try:
                raw = path.read_bytes()
            except OSError:
                return {"full": "", "computation": "", "files": names}
            names.append(rel)
            full.update(rel.encode("utf-8") + b"\0" + raw + b"\0")
            mark = plot_scope.computation_print(raw.decode("utf-8", errors="replace")) if path.suffix == ".py" else None
            comp.update(rel.encode("utf-8") + b"\0" + (mark.encode("utf-8") if mark is not None else raw) + b"\0")
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
