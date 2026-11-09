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
  main guard), an import or syntax error, a run that ended without its ``RESULT_JSON`` line; and a ``TypeError`` or
  ``AttributeError`` that shows at the same place of the code on a larger copy of the record too (3 trials can cause
  one: a fit with more parameters than points, a helper that returns ``None`` for a small ``n``).
* ``small_data``: the outcome 3 trials can explain (a ``ValueError`` in a fit, a missing setting, a figure that was not
  drawn, a quantity with no usable value, a ``TypeError`` that goes away on more trials) or that the gate cannot judge.
* ``time``: the run did not end in its time.
* ``not_checked``: the gate could not run here (a Docker quest: the scratch folder is not visible to the container).

The kind is decided by the exception's type, where it was raised and, for a ``TypeError`` or ``AttributeError``, a
second run on more trials; never by words in its message.
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
from .code_layout import _SCRIPT_NAMES, _is_main_guard

TRIALS = 3
#: Evaluations of a search kept in the small copy of its ledger.
SEARCH_ROWS = 200
#: Trials per setting of the larger copy that confirms a ``TypeError`` or ``AttributeError`` (the recorded trials of
#: each setting up to this many, 3x the small copy) and the longest that confirming run may take: a candidate fault
#: costs at most this much more, so a gate run adds well under a minute.
CONFIRM_TRIALS = 10
CONFIRM_TIMEOUT_S = 40
#: The most the copied folders of the quest may weigh (bytes) in all, and ``data/`` alone; a folder over what is left
#: is left out of the scratch copy (the gate then does not call a missing-input error a fault).
COPY_CAP = 1024 * 1024 * 1024  # the whole copy of the quest's folders
DATA_CAP = 500 * 1024 * 1024  # ``data/`` alone
FAULT, SMALL_DATA, TIME, NOT_CHECKED = "fault", "small_data", "time", "not_checked"
#: The exceptions that say the code is wrong whatever the data. ``UnknownResult`` is FI's own frame refusing a read (a result
#: the plan does not declare, or one read in a form the record does not hold): the record's shape decides it, and the small
#: copy keeps every setting and compared value of that shape, so more trials never change it. The figure calls of the frame raise
#: ``UnknownResult`` for a name the plan does not declare and ``BadArgument`` for a value outside a closed list (core/frame_figure.py).
_HARD_FAULTS = frozenset({"NameError", "UnboundLocalError", "ImportError", "ModuleNotFoundError",
                          "SyntaxError", "IndentationError", "TabError", "UnknownResult", "BadArgument"})
#: The exceptions that say so only if they show on more data too (too little data can raise them).
_MAYBE_FAULTS = frozenset({"AttributeError", "TypeError"})
_FAULTS = _HARD_FAULTS | _MAYBE_FAULTS
_STAGES = {"static": "reading the code", "derive": "running the analysis", "figures": "drawing the figures",
           "facts": "filling the paper's numbers"}
_KINDS = {FAULT: "has an error in its code", SMALL_DATA: "stopped on a small copy of the recorded results, which may have too few trials to tell whether the code is wrong",
          TIME: "did not finish in time", NOT_CHECKED: "was not checked"}
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
    #: Folders of the quest left out of the scratch copy because they were over the size cap.
    left_out: list[str] = field(default_factory=list)
    #: When FI writes the analysis (core/analysis_frame.py): the function of derive.py FI's record of the run names as the
    #: one that failed, was refused or is still FI's stub (``""``: none, or a statement of derive.py outside its functions).
    body: str = ""

    def record(self) -> dict[str, Any]:
        return {"ok": self.ok, "stage": self.stage, "kind": self.kind, "where": self.where, "detail": self.detail,
                "exception": self.exception, "unfilled": self.unfilled, "late": self.late, "seconds": self.seconds,
                "left_out": self.left_out, **({"body": self.body} if self.body else {})}

    def says(self) -> str:
        if self.ok:
            return "the analysis ran from start to end on a small copy of the recorded results"
        what, stage = _KINDS.get(self.kind, "stopped"), _STAGES.get(self.stage, self.stage)
        return f"the analysis {what} ({stage}{', ' + self.where if self.where else ''}): {self.detail}"


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

    # The guard block's own variables (a loop or comprehension target, an assignment, a lambda argument) are not
    # names it needs from elsewhere; a later top-level `for i in ...` does not make `i` a late definition.
    own_guard = {n.id for n in ast.walk(body[guard]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    own_guard |= {a.arg for a in ast.walk(body[guard]) if isinstance(a, ast.arg)}
    pending = set(loads(body[guard], False)) - own_guard
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
    """``(file, name, line)`` of each :func:`late_definitions` entry in the scripts FI runs (``experiment.py``,
    ``simulate.py``, ``submit.py``, ``run.py`` at the top of the folder). A helper module's own guard (a self-test) is
    not run as a script, so it is not looked at."""
    out: list[tuple[str, str, int]] = []
    for path in sorted(Path(code_dir).glob("*.py")):
        rel = path.relative_to(code_dir).as_posix()
        if path.name not in _SCRIPT_NAMES:
            continue
        try:
            found = late_definitions(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        out.extend((rel, name, line) for name, line in sorted(found.items(), key=lambda kv: kv[1]))
    return out


# --- the exception a run ended with -------------------------------------------------------------------------------------


def last_exception(stderr: str) -> str:
    """The type of the exception the script died of (``"ValueError"``, ``"LinAlgError"``): the last traceback in
    ``stderr``, and only when its exception line ends ``stderr`` (apart from indented lines, such as a note). A
    traceback the script handled and printed, followed by anything else, is not what stopped it: ``""``, as for a
    script that stopped itself with a message."""
    lines = (stderr or "").splitlines()
    start = max((i for i, ln in enumerate(lines) if ln.startswith("Traceback (most recent call last)")), default=None)
    if start is None:
        return ""
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if ln[:1] in (" ", "\t") or not ln.strip():
            continue
        m = _EXC_LINE.match(ln)
        if not m:
            return ""
        if any(x[:1] not in (" ", "\t") for x in lines[i + 1:] if x.strip()):
            return ""
        return m.group(1).rsplit(".", 1)[-1]
    return ""


def kind_of(exception: str) -> str:
    """``fault`` for an exception that may say the code is wrong whatever the data (a ``TypeError`` or
    ``AttributeError`` is confirmed on more trials by :func:`analysis_gate`), else ``small_data``."""
    return FAULT if exception in _FAULTS else SMALL_DATA


# --- the small copy of the recorded raw ----------------------------------------------------------------------------------


def _shrink(entry: dict[str, Any], n: int, failed: int = 0) -> dict[str, Any]:
    """One recorded setting with its first ``n`` trials, its counts recomputed. ``failed`` is how many of those trials
    the ledger records as failed; a setting with no good trial stays with none."""
    e = copy.deepcopy(entry)
    for m in (e.get("metrics") or {}).values():
        if not isinstance(m, dict) or not isinstance(m.get("values"), list):
            continue
        if isinstance(m.get("trials"), list) and len(m["trials"]) == len(m["values"]):
            keep = [i for i, t in enumerate(m["trials"]) if not (isinstance(t, int) and t >= n)]
            m["values"], m["trials"] = [m["values"][i] for i in keep], [m["trials"][i] for i in keep]
        else:  # the values are the good trials: the window holds n - failed of them
            m["values"] = m["values"][:max(n - failed, 0)]
            if isinstance(m.get("trials"), list):
                m["trials"] = m["trials"][:len(m["values"])]
        fin = [v for v in m["values"] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
        m["count"], m["total"], m["non_finite"] = len(m["values"]), float(sum(fin)), len(m["values"]) - len(fin)
        if isinstance(m.get("why_missing"), dict):
            m["why_missing"] = ({k: min(v, m["non_finite"]) for k, v in m["why_missing"].items()}
                                if m["non_finite"] else {})
    ok = min(int(e["ok"] if e.get("ok") is not None else n), max(n - failed, 0))
    e["ok"], e["failed"] = ok, failed
    e["planned"] = ok + failed
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
    failed_by_cell: dict[str, int] = {}
    if (src / "ledger.jsonl").is_file():
        with open(src / "ledger.jsonl", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"failed"' not in line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (row.get("event") == "trial" and row.get("status") == "failed" and isinstance(row.get("trial"), int)
                        and row["trial"] < trials):
                    key = str(row.get("cell"))
                    failed_by_cell[key] = failed_by_cell.get(key, 0) + 1
    data["cells"] = [_shrink(c, trials, failed_by_cell.get(str(c.get("key")), 0))
                     for c in data["cells"] if isinstance(c, dict)]
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


def _size_within(path: Path, cap: int) -> int | None:
    """The bytes of the files under ``path``, or ``None`` when they weigh more than ``cap`` (stops counting once over)."""
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
        if total > cap:
            return None
    return total


def _copy_quest_folders(quest_root: Path, scratch: Path) -> list[str]:
    """The quest's inputs, needs, results and data folders, and the subfolders of ``raw/`` (a check's own record, such as
    ``raw/oracle_check/``), at most :data:`COPY_CAP` bytes in all (``data/``: :data:`DATA_CAP`); a folder that does not
    fit is left out and its name returned."""
    left: list[str] = []
    budget = COPY_CAP
    todo = [(n, quest_root / n, scratch / n, DATA_CAP if n == "data" else COPY_CAP)
            for n in ("inputs", "needs", "results", "data")]
    raw = quest_root / "raw"
    if raw.is_dir():
        todo += [(f"raw/{p.name}", p, scratch / "raw" / p.name, COPY_CAP) for p in sorted(raw.iterdir()) if p.is_dir()]
    for name, src, dst, cap in todo:
        if not src.is_dir():
            continue
        size = _size_within(src, min(cap, budget))
        if size is None:
            left.append(name)
            continue
        budget -= size
        shutil.copytree(src, dst, dirs_exist_ok=True)
    return left


def _draws(code_dir: Path) -> bool:
    return any("savefig" in p.read_text(encoding="utf-8", errors="replace")
               for p in Path(code_dir).rglob("*.py") if "tests" not in p.relative_to(code_dir).parts)


def _quest_python(quest_root: Path) -> str:
    """The quest's own venv interpreter when it exists, else this one."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (quest_root / rel).is_file():
            return str(quest_root / rel)
    return sys.executable


def recorded_trials(quest_root: Path | str) -> int:
    """The most trials any setting of the quest's ``raw/trials.json`` records (0 when unreadable)."""
    try:
        cells = json.loads((Path(quest_root) / "raw" / "trials.json").read_text(encoding="utf-8")).get("cells") or []
    except (OSError, ValueError, AttributeError):
        return 0
    best = 0
    for c in cells:
        for m in ((c or {}).get("metrics") or {}).values():
            if isinstance(m, dict) and isinstance(m.get("values"), list):
                best = max(best, len(m["values"]))
    return best


async def analysis_gate(quest_root: Path | str, code_dir: Path | str, protocol: dict[str, Any] | None = None, *,
                        python: Path | str | None = None, trials: int = TRIALS, timeout_s: int = 180,
                        search: bool | None = None, work_dir: Path | str | None = None, keep: bool = False,
                        sandbox: str = "venv") -> GateResult:
    """Run ``code_dir``'s ``experiment.py`` from start to end in a scratch copy of the quest on a small copy of its
    recorded raw (see the module docstring) and say how it went. ``quest_root`` is only read. ``search`` (default:
    the quest recorded a search) adds FI's "not checked" record of the search. The gate always runs the code with its
    own plain executor, never the engine's (whose ``before_run`` would commit the candidate into the quest's code
    history), with ``python`` (default: the quest's own venv interpreter when it has one, else this one). Under
    ``sandbox="docker"`` the scratch folder is not visible to the container: the result is ``not_checked``.

    A ``TypeError`` or ``AttributeError`` can come from too little data, so it is run again on a larger copy of the
    record (up to :data:`CONFIRM_TRIALS` trials of every setting, at most :data:`CONFIRM_TIMEOUT_S` s) and is a
    ``fault`` only when the same exception type shows at the same place at both sizes; ``seconds["confirm"]`` is that
    run's time. A record that holds no more trials than the small copy gives the same run, which counts as both."""
    quest_root, code_dir = Path(quest_root), Path(code_dir)
    seconds: dict[str, float] = {}
    if sandbox == "docker":
        return GateResult(False, "static", NOT_CHECKED, "", "the quest runs its code in Docker, which cannot see the "
                          "gate's scratch copy, so the analysis was not run before the change was kept", seconds=seconds)
    began = time.monotonic()
    late = late_definitions_in(code_dir)
    seconds["static"] = round(time.monotonic() - began, 2)
    if late:
        file, name, line = late[0]
        return GateResult(False, "static", FAULT, f"code/{file} line {line}",
                          (", ".join(f"`{n}`" for _f, n, _l in late[:4]) + (" is" if len(late) == 1 else " are")
                           + " defined after the line that starts the script (`if __name__ == \"__main__\":`), "
                             "which runs first and needs it"), "NameError",
                          late=[f"{f}:{n}" for f, n, _ in late], seconds=seconds)
    py = str(python or _quest_python(quest_root))
    got = await _run_gate(quest_root, code_dir, protocol, py, trials, timeout_s, search, work_dir, keep, seconds)
    if not (got.stage == "derive" and got.kind == FAULT and got.exception in _MAYBE_FAULTS):
        return got
    bigger = min(CONFIRM_TRIALS, recorded_trials(quest_root))
    if bigger <= trials:
        return got  # the record holds no more trials: the larger copy is the same run
    began = time.monotonic()
    again = await _run_gate(quest_root, code_dir, protocol, py, bigger, min(timeout_s, CONFIRM_TIMEOUT_S), search,
                            None, False, {})
    got.seconds["confirm"] = round(time.monotonic() - began, 2)
    if again.stage == "derive" and again.exception == got.exception and again.where == got.where:
        return got
    got.kind = SMALL_DATA
    got.detail = (f"{got.detail} (not the same on {bigger} trials of every setting: too few trials to tell)")[:400]
    return got


async def _run_gate(quest_root: Path, code_dir: Path, protocol: dict[str, Any] | None, python: str, trials: int,
                    timeout_s: int, search: bool | None, work_dir: Path | str | None, keep: bool,
                    seconds: dict[str, float]) -> GateResult:
    scratch = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="fi_gate_"))
    try:
        began = time.monotonic()
        scratch.mkdir(parents=True, exist_ok=True)
        _copy_code(code_dir, scratch / "code")
        for name in ("figures", "results", "raw", "needs"):
            (scratch / name).mkdir(exist_ok=True)
        left_out = await asyncio.to_thread(_copy_quest_folders, quest_root, scratch)
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
        from .execution import VenvExecutor

        executor = VenvExecutor()  # a plain one: no before_run, so nothing is committed to the quest's code history
        began = time.monotonic()
        run = await executor.execute([python, "code/experiment.py"], cwd=scratch, timeout_s=timeout_s, env=env)
        seconds["derive"] = round(time.monotonic() - began, 2)
        if getattr(run, "timed_out", False):
            return GateResult(False, "derive", TIME, "", f"the analysis was still running after {timeout_s} s",
                              seconds=seconds)
        err = getattr(run, "stderr", "") or ""
        from . import analysis_frame as _analysis

        if _analysis.is_fis_file(scratch / "code" / _analysis.EXPERIMENT_FILE):
            # The analysis is FI's: its own record of the run says which function of the study's failed (FI's lines never do).
            return await asyncio.to_thread(_fis_analysis_result, scratch, run, protocol, seconds, left_out)
        where = _trials.last_frame(err, scratch)
        exc = last_exception(err)
        if run.returncode != 0:
            tail = [ln for ln in err.strip().splitlines() if ln.strip()][-1:] or [f"exit code {run.returncode}"]
            kind, detail = (kind_of(exc) if exc else SMALL_DATA), tail[0][:300]
            if kind == FAULT and exc in _MAYBE_FAULTS and left_out:
                # A folder the analysis reads was too large to copy: a missing input can raise these, so it is not
                # shown to be the code's fault.
                kind = SMALL_DATA
                detail += f" (a folder too large to copy was left out of the check: {', '.join(left_out)})"
            return GateResult(False, "derive", kind, where, detail, exc, seconds=seconds, left_out=left_out)
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
                              "no usable value for: " + ", ".join(unfilled[:6]), unfilled=unfilled,
                              seconds=seconds)
        return GateResult(True, seconds=seconds)
    finally:
        if not keep and not work_dir:
            shutil.rmtree(scratch, ignore_errors=True)


def _fis_analysis_result(scratch: Path, run: Any, protocol: dict[str, Any] | None, seconds: dict[str, float],
                         left_out: list[str]) -> GateResult:
    """The gate's outcome for the analysis FI writes (core/analysis_frame.py), read from FI's own record of the run
    (``data/results/derived.json``) rather than from its printed text:

    * a fault of FI's own (exit code 2, or a stop in FI's lines): ``not_checked``, never a fault of the study's code, so
      nothing is refused for it;
    * a function of derive.py that raised: judged by the exception's type as for any analysis (a name defined nowhere is a
      ``fault``); one whose return FI refused (a wrong type, a key that is not a setting): a ``fault``, whatever the data;
      one that gave NaN for a result: ``small_data``;
    * a function still FI's stub for a declared quantity of a counting role: a ``fault`` at the facts stage, since the long
      run would give no value for it (FI's stub is the same on any number of trials)."""
    from . import analysis_frame as _analysis

    record = _analysis.read_record(scratch)
    err = getattr(run, "stderr", "") or ""
    printed = "RESULT_JSON:" in (getattr(run, "stdout", "") or "")
    if run.returncode != 0:
        got = _analysis.failure(record, err, run.returncode, printed, timed_out=bool(getattr(run, "timed_out", False)))
        if got["kind"] == "fi":
            return GateResult(False, "derive", NOT_CHECKED, "", "FI's own analysis could not run on the small copy (FI's "
                              "fault, not the study's code): " + got["why"][:300], seconds=seconds, left_out=left_out)
        why = got["why"]
        # what FI's record of this run says of the function: a derived quantity's record, or a figure's outcome
        body = str(got["body"] or "")
        rec = None
        if body and printed and not record.get("from_earlier_run"):
            rec = next((r for r in (record.get("derived") or {}).values()
                        if isinstance(r, dict) and r.get("by") == body), None)
            if rec is None and body.startswith("figure_"):
                rec = next(({"status": o.get("status"), "why": o.get("why")} for o in record.get("figures") or []
                            if isinstance(o, dict) and f"figure_{o.get('id')}" == body), None)
        said = str(rec.get("why") if rec else "") or why
        # a figure's outcome says "`figure_x` stopped with NameError: ...": the exception's name follows "stopped with"
        m = _EXC_LINE.match(said) or re.search(r"stopped with ([A-Za-z_][\w.]*)(?::|\b)", said)
        exc = last_exception(err) if got["kind"] == "module" else (m.group(1) if m else "")
        status = str(rec.get("status")) if rec else ("failed" if got["kind"] == "module" else "")
        kind = (kind_of(exc) if exc else SMALL_DATA) if status == "failed" else FAULT if status == "refused" else SMALL_DATA
        if kind == FAULT and exc in _MAYBE_FAULTS and left_out:
            kind = SMALL_DATA
            said += f" (a folder too large to copy was left out of the check: {', '.join(left_out)})"
        where = f"code/derive.py, in {got['body']}" if got["body"] else "code/derive.py"
        return GateResult(False, "derive", kind, where, said[:300], exc, seconds=seconds, left_out=left_out,
                          body=got["body"])
    if not printed:
        return GateResult(False, "derive", NOT_CHECKED, "", "FI's own analysis ended without its RESULT_JSON line (FI's "
                          "fault, not the study's code)", seconds=seconds)
    roles = {str(d.get("id")): str(d.get("role") or "result") for d in (protocol or {}).get("derived") or [] if isinstance(d, dict)}
    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    stubs = [(i, str((derived.get(i) or {}).get("by") or "")) for i in [x["id"] for x in _analysis.not_computed(record)
                                                                       if x["status"] == "not_written"]
             if str((derived.get(i) or {}).get("role") or roles.get(i, "result")) in _COUNTING]
    if stubs:
        names = [i for i, _b in stubs]
        return GateResult(False, "facts", FAULT, f"code/derive.py, in {stubs[0][1]}" if stubs[0][1] else "code/derive.py",
                          "; ".join(f"`{i}` is computed by no trial and no function of the study's yet (`{b}` is still FI's "
                                    "stub)" if b else f"`{i}` is computed by no trial and no function of the study's yet"
                                    for i, b in stubs[:4]),
                          unfilled=names, seconds=seconds, body=stubs[0][1])
    began = time.monotonic()
    unfilled = _unfilled_headlines(scratch, protocol)
    seconds["facts"] = round(time.monotonic() - began, 2)
    if unfilled:
        return GateResult(False, "facts", SMALL_DATA, "", "no usable value for: " + ", ".join(unfilled[:6]), unfilled=unfilled,
                          seconds=seconds)
    return GateResult(True, seconds=seconds)


#: The roles a declared quantity counts for (core/numeric_warnings.COUNTING_ROLES).
_COUNTING = ("result", "reference", "physical_bound")


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
