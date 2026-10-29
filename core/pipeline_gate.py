"""The analysis gate: does the analysis script still run from start to end after a code change?

Nothing that accepts a change to a quest's code (a finding repair, a crash repair, a speed-up) ran ``experiment.py``
from start to end before keeping it: the simulation was called once, the analysis first ran in the real run, hours
later. A repair that added a definition after the ``if __name__ == "__main__":`` block, or used a name nothing defines,
was kept and found out there.

:func:`analysis_gate` runs the analysis the way the real run does, on a small copy of the quest's own recorded raw
(:func:`small_raw`: ``TRIALS`` trials of EVERY setting, since an analysis that indexes a setting by name fails on a copy
that leaves the setting out; and, for a study that searches, FI's honest "not checked" record of the search, the shape
that crashed a real analysis), in a scratch copy of the quest. No model is asked. The outcome is a :class:`GateResult`
with a ``kind``:

* ``fault``: the code is wrong whatever the data: a name that is not defined (including a definition placed after the
  main guard), an attribute, import, syntax or type error, a run that ended without its ``RESULT_JSON`` line.
* ``small_data``: the outcome 3 trials can explain (a ``ValueError`` in a fit, a missing setting, a figure that was not
  drawn, a quantity with no usable value) or that the gate cannot judge.
* ``time``: the run did not end in its time.

The kind is decided by the exception's type and where it was raised, never by words in its message.
:func:`gate_verdict` is the decision rule: a change is refused only when the code passed before it and has a new
``fault`` after it; ``small_data`` and ``time`` are shown, never a reason to refuse.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import contract as _contract
from . import trial_runner as _trials
from .code_layout import _is_main_guard

TRIALS = 3
#: Evaluations of a search kept in the small copy of its ledger.
SEARCH_ROWS = 200
FAULT, SMALL_DATA, TIME = "fault", "small_data", "time"
#: The exceptions that say the code is wrong whatever the data.
_FAULTS = frozenset({"NameError", "UnboundLocalError", "AttributeError", "ImportError", "ModuleNotFoundError",
                     "SyntaxError", "IndentationError", "TabError", "TypeError"})
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


@dataclass
class GateResult:
    ok: bool
    stage: str = ""  # "", "static", "derive", "figures" or "facts": where it stopped
    kind: str = ""  # "fault", "small_data" or "time" (empty when ok)
    where: str = ""  # file and line in the quest's own code
    detail: str = ""
    exception: str = ""  # the exception type that stopped the run, if one did
    unfilled: list[str] = field(default_factory=list)
    #: ``file:name`` of each definition after the main guard that the guard reaches (the static stage).
    late: list[str] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"ok": self.ok, "stage": self.stage, "kind": self.kind, "where": self.where, "detail": self.detail,
                "exception": self.exception, "unfilled": self.unfilled, "late": self.late, "seconds": self.seconds}

    def says(self) -> str:
        if self.ok:
            return "the analysis ran from start to end on a small copy of the recorded results"
        return f"the analysis {self.kind} ({self.stage}{', ' + self.where if self.where else ''}): {self.detail}"


@dataclass
class Verdict:
    reject: bool
    why: str = ""
    disclose: str = ""


# --- the code as written: a definition the main guard runs before -------------------------------------------------------


def late_definitions(source: str) -> dict[str, int]:
    """``{name: line}`` of each top-level definition placed after the ``if __name__ == "__main__":`` block that the code
    run before it (the guard's own block, or a module-level statement above it) reaches, directly or through the
    functions defined above the guard. The definition does not exist yet when the guard runs, so the run raises a
    ``NameError``."""
    try:
        body = ast.parse(source).body
    except SyntaxError:
        return {}
    guard = next((i for i, s in enumerate(body) if _is_main_guard(s)), None)
    if guard is None:
        return {}

    def bound(stmts: list[ast.stmt]) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.setdefault(s.name, s.lineno)
            elif isinstance(s, (ast.Import, ast.ImportFrom)):
                for a in s.names:
                    out.setdefault((a.asname or a.name).split(".")[0], s.lineno)
            else:
                for n in ast.walk(s):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        out.setdefault(n.id, s.lineno)
        return out

    early, late = bound(body[:guard]), bound(body[guard + 1:])
    late = {k: v for k, v in late.items() if k not in early}
    if not late:
        return {}
    functions = {s.name: s for s in body[:guard] if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}

    def loads(node: ast.AST, skip_defs: bool) -> set[str]:
        names: set[str] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if skip_defs and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(n.decorator_list)
                stack.extend(d for d in n.args.defaults + n.args.kw_defaults if d is not None)
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
            stack.extend(ast.iter_child_nodes(n))
        return names

    def own(fn: ast.AST) -> set[str]:
        a = fn.args  # type: ignore[attr-defined]
        got = {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if x is not None}
        got |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        return got

    pending = set(loads(body[guard], False))
    for s in body[:guard]:
        if not isinstance(s, (ast.Import, ast.ImportFrom)):
            pending |= loads(s, True)
    seen: set[str] = set()
    reached: dict[str, int] = {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in late:
            reached[name] = late[name]
        if name in functions:
            fn = functions[name]
            pending |= loads(fn, False) - own(fn)
    return reached


def late_definitions_in(code_dir: Path) -> list[tuple[str, str, int]]:
    """``(file, name, line)`` of each :func:`late_definitions` entry in the folder's scripts (not its tests)."""
    out: list[tuple[str, str, int]] = []
    for path in sorted(Path(code_dir).rglob("*.py")):
        rel = path.relative_to(code_dir).as_posix()
        if rel.startswith(("tests/", ".")) or "/." in rel or "__pycache__" in rel or path.name.startswith("test_"):
            continue
        try:
            found = late_definitions(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        out.extend((rel, name, line) for name, line in sorted(found.items(), key=lambda kv: kv[1]))
    return out


# --- the exception a run ended with -------------------------------------------------------------------------------------


def last_exception(stderr: str) -> str:
    """The type of the exception the last traceback in ``stderr`` ends with (``"ValueError"``,
    ``"LinAlgError"``), ``""`` when it holds no traceback (a script that stopped itself with a message)."""
    lines = (stderr or "").splitlines()
    start = max((i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")), default=None)
    if start is None:
        return ""
    for ln in lines[start + 1:]:
        if ln[:1] in (" ", "\t") or not ln.strip():
            continue
        m = _EXC_LINE.match(ln)
        return m.group(1).rsplit(".", 1)[-1] if m else ""
    return ""


def kind_of(exception: str) -> str:
    """``fault`` for an exception that says the code is wrong whatever the data, else ``small_data``."""
    return FAULT if exception in _FAULTS else SMALL_DATA


# --- the small copy of the recorded raw ----------------------------------------------------------------------------------


def _shrink(entry: dict[str, Any], n: int) -> dict[str, Any]:
    """One recorded setting with its first ``n`` trials, its counts recomputed."""
    e = copy.deepcopy(entry)
    for m in (e.get("metrics") or {}).values():
        if not isinstance(m, dict) or not isinstance(m.get("values"), list):
            continue
        m["values"] = m["values"][:n]
        if isinstance(m.get("trials"), list):
            m["trials"] = m["trials"][:n]
        fin = [v for v in m["values"] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        m["count"], m["total"], m["non_finite"] = len(m["values"]), float(sum(fin)), len(m["values"]) - len(fin)
        if isinstance(m.get("why_missing"), dict):
            m["why_missing"] = ({k: min(v, m["non_finite"]) for k, v in m["why_missing"].items()}
                                if m["non_finite"] else {})
    e["planned"] = e["ok"] = min(int(e.get("ok") or n), n)
    e["failed"] = 0
    return e


def small_raw(quest_root: Path, scratch: Path, trials: int = TRIALS) -> bool:
    """Write ``raw/trials.json`` (and FI's ledger, and a short copy of a search's ledger) under ``scratch`` from the
    quest's own recorded raw: ``trials`` trials of every setting, in the record's own format. ``False`` when the quest
    has no readable ``raw/trials.json``."""
    src, dst = Path(quest_root) / "raw", Path(scratch) / "raw"
    try:
        data = json.loads((src / "trials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict) or not isinstance(data.get("cells"), list):
        return False
    dst.mkdir(parents=True, exist_ok=True)
    data["cells"] = [_shrink(c, trials) for c in data["cells"] if isinstance(c, dict)]
    (dst / "trials.json").write_text(json.dumps(data), encoding="utf-8")
    if (src / "ledger.jsonl").is_file():
        with open(src / "ledger.jsonl", encoding="utf-8", errors="replace") as fh, \
                open(dst / "ledger.jsonl", "w", encoding="utf-8") as out:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("event") == "trial" and isinstance(row.get("trial"), int) and row["trial"] >= trials:
                    continue
                if row.get("event") == "planned" and isinstance(row.get("trials"), int):
                    row["trials"] = min(row["trials"], trials)
                out.write(json.dumps(row) + "\n")
    if (src / "optimisation_ledger.jsonl").is_file():
        kept = 0
        with open(src / "optimisation_ledger.jsonl", encoding="utf-8", errors="replace") as fh, \
                open(dst / "optimisation_ledger.jsonl", "w", encoding="utf-8") as out:
            for line in fh:
                if '"event": "evaluation"' in line[:300]:
                    kept += 1
                    if kept > SEARCH_ROWS:
                        continue
                out.write(line if line.endswith("\n") else line + "\n")
    return True


def not_checked_record() -> dict[str, Any]:
    """FI's honest record of a search whose best design was not checked at finer settings (the search used the whole
    time): ``best`` and ``improvement`` are ``null``, the keys are there. An analysis written for a record that always
    has a best design fails on it."""
    return {"schema": "fi.optimum-check/v1", "verdict": "unverified", "passed": False, "finished": False,
            "says": "The best design could not be checked at finer numerical settings: the time limit ran out first.",
            "checks": {}, "best": None, "search_best": None, "baseline": None, "improvement": None, "candidates": [],
            "evaluations": {"check": 0, "failed": 0}, "blind_spots": [], "notes": []}


# --- the gate ----------------------------------------------------------------------------------------------------------


def _copy_code(code_dir: Path, dst: Path) -> None:
    shutil.copytree(code_dir, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "tests", ".venv"))


def _draws(code_dir: Path) -> bool:
    return any("savefig" in p.read_text(encoding="utf-8", errors="replace")
               for p in Path(code_dir).rglob("*.py") if "tests" not in p.relative_to(code_dir).parts)


async def analysis_gate(quest_root: Path | str, code_dir: Path | str, protocol: dict[str, Any] | None = None, *,
                        executor: Any = None, python: Path | str | None = None, trials: int = TRIALS,
                        timeout_s: int = 180, search: bool | None = None, work_dir: Path | str | None = None,
                        keep: bool = False) -> GateResult:
    """Run ``code_dir``'s ``experiment.py`` from start to end in a scratch copy of the quest on a small copy of its
    recorded raw (see the module docstring) and say how it went. ``quest_root`` is only read. ``search`` (default:
    the quest recorded a search) adds FI's "not checked" record of the search. ``executor`` and ``python`` default
    to a plain :class:`~core.execution.VenvExecutor` and this interpreter; a quest's own are passed by the engine."""
    quest_root, code_dir = Path(quest_root), Path(code_dir)
    seconds: dict[str, float] = {}
    began = time.monotonic()
    late = late_definitions_in(code_dir)
    seconds["static"] = round(time.monotonic() - began, 2)
    if late:
        file, name, line = late[0]
        return GateResult(False, "static", FAULT, f"code/{file} line {line}",
                          (", ".join(f"`{n}`" for _f, n, _l in late[:4]) + (" is" if len(late) == 1 else " are")
                           + " defined after the main guard, which runs first and uses it"), "NameError",
                          late=[f"{f}:{n}" for f, n, _ in late], seconds=seconds)
    scratch = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="fi_gate_"))
    try:
        began = time.monotonic()
        scratch.mkdir(parents=True, exist_ok=True)
        _copy_code(code_dir, scratch / "code")
        for name in ("figures", "results", "raw", "needs"):
            (scratch / name).mkdir(exist_ok=True)
        for name in ("inputs", "needs", "results"):
            if (quest_root / name).is_dir():
                shutil.copytree(quest_root / name, scratch / name, dirs_exist_ok=True)
        data = quest_root / "data"
        if data.is_dir() and sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) <= 20 * 1024 * 1024:
            shutil.copytree(data, scratch / "data", dirs_exist_ok=True)
        if not await asyncio.to_thread(small_raw, quest_root, scratch, trials):
            return GateResult(False, "static", SMALL_DATA, "", "the quest has no recorded raw/trials.json to copy",
                              seconds=seconds)
        env = dict(os.environ, FI_TRIALS="raw/trials.json", FI_RAW_DIR="raw", FI_REPLICATE_SEED="0",
                   FI_REPLICATE_INDEX="0", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        if (scratch / "inputs" / "examples").is_dir():
            env["FI_INPUT_DIR"] = str(scratch / "inputs" / "examples")
        searched = (quest_root / "raw" / "optimisation_ledger.jsonl").is_file() if search is None else search
        if searched:
            from . import optimum_check as _check

            (scratch / "needs").mkdir(exist_ok=True)
            (scratch / _check.CHECK_PATH).write_text(json.dumps(not_checked_record()), encoding="utf-8")
            env[_check.CHECK_ENV] = _check.CHECK_PATH.as_posix()
            env["FI_OPTIMISATION"] = "raw/optimisation_ledger.jsonl"
            env["FI_BEST_DESIGN"] = "results/best_design.json"
        seconds["prepare"] = round(time.monotonic() - began, 2)
        if executor is None:
            from .execution import VenvExecutor

            executor = VenvExecutor()
        began = time.monotonic()
        run = await executor.execute([str(python or sys.executable), "code/experiment.py"], cwd=scratch,
                                     timeout_s=timeout_s, env=env)
        seconds["derive"] = round(time.monotonic() - began, 2)
        if getattr(run, "timed_out", False):
            return GateResult(False, "derive", TIME, "", f"the analysis was still running after {timeout_s} s",
                              seconds=seconds)
        err = getattr(run, "stderr", "") or ""
        where = _trials.last_frame(err, scratch)
        exc = last_exception(err)
        if run.returncode != 0:
            tail = [ln for ln in err.strip().splitlines() if ln.strip()][-1:] or [f"exit code {run.returncode}"]
            return GateResult(False, "derive", kind_of(exc) if exc else SMALL_DATA, where, tail[0][:300], exc,
                              seconds=seconds)
        if "RESULT_JSON:" not in (run.stdout or ""):
            return GateResult(False, "derive", FAULT, where, "the analysis ended without its RESULT_JSON line",
                              seconds=seconds)
        began = time.monotonic()
        if _draws(code_dir) and not any(f.is_file() for f in (scratch / "figures").rglob("*")):
            return GateResult(False, "figures", SMALL_DATA, "", "the code draws figures and none was written",
                              seconds=seconds)
        unfilled = await asyncio.to_thread(_unfilled_headlines, scratch, protocol)
        seconds["facts"] = round(time.monotonic() - began, 2)
        if unfilled:
            return GateResult(False, "facts", SMALL_DATA, "",
                              "no value a paper key can fill for: " + ", ".join(unfilled[:6]), unfilled=unfilled,
                              seconds=seconds)
        return GateResult(True, seconds=seconds)
    finally:
        if not keep and not work_dir:
            shutil.rmtree(scratch, ignore_errors=True)


def _unfilled_headlines(scratch: Path, protocol: dict[str, Any] | None) -> list[str]:
    """The plan's result-role quantities that the small raw carries and no paper key can fill (FI's own records, as the
    writer's fact sheet reads them)."""
    if not protocol:
        return []
    from . import paper_values as _values

    records = _values.load(scratch, protocol)
    if records is None:
        return []
    carried = set(records.metrics())
    filled = records.fillable()
    return [m for m, role in _contract.raw_roles(protocol).items() if role == "result" and m in carried and m not in filled]


def gate_verdict(before: GateResult | None, after: GateResult) -> Verdict:
    """Whether to refuse a change: only when the code passed the gate before it (``before.ok``) and has a new ``fault``
    after it. A ``small_data`` or ``time`` outcome is disclosed, never a reason to refuse; a gate that did not pass
    before the change, or that was not run, refuses nothing (the fault is shown to the repair instead).

    One reading of "passed before" for the static stage: it reads the code alone and runs first, so a definition after
    the main guard that the guard reaches and the code before the change did not have is new even when the code before
    stopped later, at a stage that needs data (a run stopped there never reached the place the new fault is in)."""
    if after.ok:
        return Verdict(False)
    shown = after.says()
    new_late = after.stage == "static" and before is not None and bool(set(after.late) - set(before.late))
    if after.kind == FAULT and before is not None and (before.ok or new_late):
        return Verdict(True, why=shown)
    if after.kind == FAULT:
        return Verdict(False, disclose=shown + " (the code did not pass before the change, so nothing is refused)"
                       if before is not None else shown + " (nothing to compare with, so nothing is refused)")
    return Verdict(False, disclose=shown)
