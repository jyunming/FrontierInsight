"""Does a fault a failing test showed change a number the study reports? Answered by running, not by reasoning about the test.

The evidence gate (``Engine._shown_fault_check``) used to stop the paper, or leave results out, when a test written by a model
failed and the code it reaches feeds the main result. A test can check something the plan does not ask for, a faulty branch
may never fire in the run, and a fault may sit in text FI wrote; each of those stopped a paper whose numbers were right.
Here the question is put to the study itself: a repair of the finding is applied to a COPY of the study's code, the study is
run again with the same seeds (the study's own entry, ``code/run.py``; only the analysis again, on the recorded trials, when
only the analysis changed), and the numbers it reports are compared with the delivered ones.

* A number counts as changed only when it moves by more than its Monte Carlo uncertainty. The uncertainty is read from FI's
  own record of the trials (``raw/trials.json``): a mean's spread over the trials divided by their square root, a share's
  binomial error, a standard deviation's own error. It is applied only when the repair changes how random numbers are drawn
  (read from the code); a repair that draws the same numbers moves a number only by what it changes. Counts of trials are
  compared exactly.
* A number the frozen protocol declares as a reference (a known answer, an analytic value) is not computed by the code: a
  repair that changes it repairs the plan, not the code, and it does not count.
* A result the frozen protocol holds to a precision (``precision.target_half_width`` on ``precision.metric``) is not changed by
  a move inside that precision, with or without redrawn numbers: the plan itself accepts that difference. A result with no
  declared precision keeps the rules above.
* A change that only reshapes the same value (a number against ``{"value": ..., "lo": ..., "hi": ...}``) is not a change.
* The rerun is bounded by the quest's own record of how long the study took (:func:`time_allowed`); when it does not fit,
  when the repaired code does not run, or when no result is written, the answer is "could not rerun".

This module only runs and compares; the repair itself is asked for by the engine (``core/finding_repair.py``).
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

#: A rerun may take this much of what the study's own record says it needs.
SLACK = 1.5
#: The study's own entry (``run.py``) runs the settings one after another, each in a process of its own, where FI's run had
#: several at once and the settings' own time (``settings_s``, summed) is what a serial run needs before it pays each process's
#: start. Measured on one study: the serial entry took 1.8 to 2.9 times ``settings_s``; 3 is used. Not measured on other studies.
SERIAL_FACTOR = 3.0
#: Never less than this (starting the interpreter and the packages).
FLOOR_S = 60
#: All the reruns of one check together may take this many times what one rerun of the study is allowed.
TOTAL_FACTOR = 2.0
#: Standard errors a number must move by (the difference of two runs has sqrt(2) times one run's error).
Z = 3.0
#: A number with no uncertainty recorded counts as changed when it moves by more than this share of its value. Added
#: exception: FI's records hold no error for it, so a smaller move is called rounding.
REL_TOL = 5e-4
#: The same, when the repair draws no random number differently: the rerun then repeats the delivered run, so only rounding
#: can differ.
EXACT_TOL = 1e-9
#: Files whose repair cannot change the trials: only the analysis is run again, on the recorded trials.
ANALYSIS_FILES = frozenset({"derive.py", "experiment.py", "fi_figures.py", "fi_table.py", "fi_frame.py", "fi_figure_calls.py"})
_SKIP_KEYS = frozenset({"seconds", "bodies", "figures", "figures_said", "fi_faults", "notes", "wall_s", "elapsed_s", "elapsed",
                        "runtime", "run_time", "time_s", "timestamp", "wall", "wall_time", "duration"})
_COPY_SKIP = shutil.ignore_patterns(".git", ".venv", "__pycache__", "output", "*.pyc", ".pytest_cache")


@dataclass
class Verdict:
    """What the rerun showed. ``status``: ``changed`` (a reported number moved: ``metrics`` names the results, ``keys`` the
    numbers), ``unchanged`` (no reported number moved) or ``could_not`` (``why`` says why)."""

    status: str
    why: str = ""
    metrics: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    mode: str = ""
    seconds: float = 0.0
    source: str = ""
    files: list[str] = field(default_factory=list)
    crashed: bool = False  # could_not because the repaired code did not run (not for time, not for a missing record)
    unplaced: list[str] = field(default_factory=list)  # changed numbers that belong to no result FI can name

    def record(self) -> dict[str, Any]:
        return {"status": self.status, "why": self.why, "metrics": self.metrics[:20], "keys": self.keys[:12],
                "notes": self.notes[:6], "mode": self.mode, "seconds": round(self.seconds, 1), "source": self.source,
                "files": self.files, "crashed": self.crashed, "unplaced": self.unplaced[:8]}


# ---- how long ---------------------------------------------------------------------------------------------------------
def time_allowed(quest_root: Path | str, limit_s: float | None = None) -> float | None:
    """The seconds a rerun may take, from the run record, ``None`` when the record says nothing. The study's entry runs the
    settings one after another, so what it needs is what the run's settings took in all (``settings_s``, times
    :data:`SERIAL_FACTOR`), or the wall time when that is longer; the answer is :data:`SLACK` times that, and never above ``limit_s`` (the quest's own limit for one run)."""
    from . import run_facts as _run_facts

    run = _run_facts.read(quest_root).get("trials_run")

    def seconds(name: str) -> float | None:
        v = run.get(name) if isinstance(run, dict) else None
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float("inf") else None

    wall, settings = seconds("wall_s"), seconds("settings_s")
    seen = [x for x in (wall, None if settings is None else SERIAL_FACTOR * settings) if x is not None]
    if not seen:
        return None
    allowed = max(FLOOR_S, SLACK * max(seen))
    return min(allowed, float(limit_s)) if limit_s else allowed


# ---- reading the results ----------------------------------------------------------------------------------------------
def _flat(x: Any, prefix: str = ""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in _SKIP_KEYS:
                continue
            # a number given as {"value": v, "lo": ..., "hi": ...} is the same number as a plain one
            yield from _flat(v, prefix if (k == "value" and not isinstance(v, (dict, list))) else f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flat(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)):
        yield prefix, float(x)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def results_of(root: Path | str, derived: str = "data/results/derived.json", trials: str = "raw/trials.json") -> dict[str, float]:
    """The numbers a run reported: every number of ``derived.json``, and for each setting the count of trials that finished
    and failed and the mean of each metric of ``trials.json``."""
    root = Path(root)
    out: dict[str, float] = {}
    d = _json(root / derived)
    if d is not None:
        out.update({"derived" + k: v for k, v in _flat(d)})
    for extra in sorted((root / Path(derived).parent).glob("*.json")) if (root / Path(derived).parent).is_dir() else []:
        # the other result files the paper may read (a table, a ledger): compared the same way
        if extra.name != Path(derived).name and extra.stat().st_size <= 8_000_000:
            doc = _json(extra)
            if doc is not None:
                out.update({f"file:{extra.name}" + k: v for k, v in _flat(doc)})
    t = _json(root / trials)
    for c in (t.get("cells") if isinstance(t, dict) else None) or []:
        if not isinstance(c, dict):
            continue
        k = c.get("key")
        out[f"trials/{k}/ok"], out[f"trials/{k}/failed"] = float(c.get("ok") or 0), float(c.get("failed") or 0)
        for m, rec in (c.get("metrics") or {}).items():
            vs = _values(rec)
            if vs and m not in _SKIP_KEYS:
                out[f"trials/{k}/{m}/mean"] = sum(vs) / len(vs)
    return out


def result_files(root: Path | str, derived: str = "data/results/derived.json") -> set[str]:
    """The names of the other result files (``file:<name>`` in :func:`results_of`) a run wrote."""
    folder = Path(root) / Path(derived).parent
    return {p.name for p in folder.glob("*.json")} - {Path(derived).name} if folder.is_dir() else set()


def _values(rec: Any) -> list[float]:
    vs = rec.get("values") if isinstance(rec, dict) else rec
    return [float(v) for v in vs or [] if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))]


def compare(base: dict[str, float], new: dict[str, float], tol: float = REL_TOL) -> list[dict[str, Any]]:
    """The numbers of ``base`` that ``new`` reports differently: ``[{key, old, new}]``. A delivered number the rerun does not
    report is changed (``new`` is ``None``); one only ``new`` has is not (a repair may add a result: nothing delivered changed).
    ``tol``: the share of its value a number may move by (rounding)."""
    out = []
    for k, b in base.items():
        n = new.get(k)
        if n is None:
            out.append({"key": k, "old": b, "new": None})
        elif k.endswith("/ok") or k.endswith("/failed"):
            if n != b:
                out.append({"key": k, "old": b, "new": n})
        elif abs(n - b) > tol * max(abs(b), 1e-12):
            out.append({"key": k, "old": b, "new": n})
    return out


def metric_of(key: str) -> tuple[str | None, str | None]:
    """``(metric, setting)`` a compared key is about."""
    if key.startswith("trials/"):
        parts = key.split("/")
        return (parts[-2] if len(parts) >= 4 else "failed trials"), parts[1]
    m = re.match(r"derived/derived/([^/]+)/by_setting/([^/]+)", key)
    if m:
        return m.group(1), m.group(2)
    m = re.match(r"derived/derived/([^/\[]+)", key)
    return (m.group(1), None) if m else (None, None)


# ---- noise ------------------------------------------------------------------------------------------------------------
def rng_signature(source: str) -> dict[str, list[str]]:
    """Per function (and ``module``): the calls that draw or seed random numbers, in order, as source text."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {"?": ["unparsable"]}

    def calls(node: ast.AST) -> list[str]:
        got = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and re.search(r"rng|random|seed|SeedSequence|spawn|shuffle|permutation", ast.unparse(n.func)):
                got.append((getattr(n, "lineno", 0), getattr(n, "col_offset", 0), ast.unparse(n)))
        return [c[2] for c in sorted(got)]

    out = {"module": calls(ast.Module(body=[s for s in tree.body if not isinstance(
        s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[]))}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = calls(n)
    return out


def draws_changed(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """The functions (``file:function``) whose random draws differ between two versions of the code (``{path: text}``)."""
    why = []
    for rel, text in new.items():
        if not rel.endswith(".py"):
            continue
        so, sn = rng_signature(old.get(rel, "")), rng_signature(text)
        why += [f"{rel}:{k}" for k in sorted(set(so) | set(sn)) if so.get(k, []) != sn.get(k, [])]
    return why


def trial_stats(trials: Any) -> tuple[dict[tuple[str, str], tuple[int, float, float]], dict[str, float]]:
    """``{(setting, metric): (n, mean, sd)}`` and ``{setting: trials that finished}`` from a record of the trials."""
    stats: dict[tuple[str, str], tuple[int, float, float]] = {}
    ok: dict[str, float] = {}
    for c in (trials.get("cells") if isinstance(trials, dict) else None) or []:
        if not isinstance(c, dict):
            continue
        k = str(c.get("key"))
        ok[k] = float(c.get("ok") or 0)
        for m, rec in (c.get("metrics") or {}).items():
            vs = _values(rec)
            if len(vs) > 1:
                mu = sum(vs) / len(vs)
                stats[(k, m)] = (len(vs), mu, math.sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1)))
    return stats, ok


_ESTIMATE = frozenset({"value", "mean", "estimate", "est", "proportion", "share", "rate", "median"})
_ERROR = frozenset({"se", "sem", "stderr", "std_err", "standard_error"})
_SPREAD = frozenset({"sd", "std", "stdev"})
_ENDS = frozenset({"lo", "hi", "low", "high", "lower", "upper", "ci", "ci_lo", "ci_hi", "ci_low", "ci_high", "ci_lower",
                   "ci_upper", "interval", "ci95", "ci95_lo", "ci95_hi"})


def number_kind(key: str) -> str:
    """What a compared number is: ``estimate`` (a mean or a share a trial's values average to), ``error`` (a standard
    error), ``spread`` (a standard deviation), ``end`` (an end of an interval) or ``other`` (a count, a coverage, anything FI
    knows no sampling error of)."""
    if key.startswith("trials/"):
        return "estimate" if key.endswith("/mean") else "other"
    m = re.match(r"derived/derived/[^/]+(?:/by_setting/[^/]+)?(?:/([^/]+))?$", key)
    leaf = re.sub(r"\[\d+\]$", "", (m.group(1) if m and m.group(1) else "")).lower()
    if not leaf or leaf in _ESTIMATE:
        return "estimate"
    return "error" if leaf in _ERROR else "spread" if leaf in _SPREAD else "end" if leaf in _ENDS else "other"


def uncertainty(key: str, old: float, stats: dict, ok: dict[str, float], kinds: dict[str, str]) -> float | None:
    """The Monte Carlo error of the number at ``key`` in the delivered run, when FI's records give one: for a mean or a share
    its standard error, for an end of an interval the same, for a standard error or a deviation its own sampling error (its
    value over sqrt(2(n-1))); ``None`` for a count, a coverage or anything else, which then counts as changed beyond
    :data:`REL_TOL`."""
    metric, setting = metric_of(key)
    if metric is None or setting is None:
        return None
    kind = number_kind(key)
    s = stats.get((setting, metric))
    if kind in ("error", "spread"):
        return abs(old) / math.sqrt(2 * (s[0] - 1)) if s and s[0] > 1 else None
    named = re.search(r"coverage|rate|fraction", key.split("derived/derived/")[-1], re.IGNORECASE) is not None
    if (not key.startswith("trials/") and 0 <= old <= 1 and kind in ("estimate", "other")
            and (kinds.get(metric) == "proportion" or named)):
        # a share (declared, or named a coverage, rate or fraction): its binomial error over the trials
        n = s[0] if s else ok.get(setting)
        return math.sqrt(max(old * (1 - old), 1e-12) / n) if n and n >= 2 else None
    if kind not in ("estimate", "end"):
        return None
    return s[2] / math.sqrt(s[0]) if s else None


def declared_precision(protocol: Any) -> tuple[str, float] | None:
    """``(metric, half width)`` of the precision the frozen protocol holds a result to (``precision.target_half_width`` on
    ``precision.metric``, or on the plan's only metric), in the unit of that result; ``None`` when the plan declares none, or
    names no metric it declares. Read from the record only: nothing is taken from the plan's text."""
    from . import run_estimate as _run_estimate

    if not isinstance(protocol, dict) or not isinstance(protocol.get("precision"), dict):
        return None
    width = protocol["precision"].get("target_half_width")
    metric, _kind = _run_estimate.precision_metric(protocol)
    declared = {str(m.get("id")) for m in protocol.get("metrics") or [] if isinstance(m, dict)}
    if (isinstance(width, bool) or not isinstance(width, (int, float)) or not 0 < float(width) < float("inf")
            or not metric or metric not in declared):
        return None
    return metric, float(width)


def counted(changes: list[dict[str, Any]], protocol: Any, base_trials: Any, redrawn: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Which of the compared changes count, and why the others do not (``notes``). ``redrawn``: the repair changes how random
    numbers are drawn, so every Monte Carlo number moves a little and only a move beyond its uncertainty counts."""
    metrics = [m for m in (protocol.get("metrics") if isinstance(protocol, dict) else None) or [] if isinstance(m, dict)]
    roles = {m.get("id"): m.get("role") for m in metrics}
    kinds = {m.get("id"): m.get("kind") for m in metrics}
    stats, ok = trial_stats(base_trials)
    notes: list[str] = []
    keep: list[dict[str, Any]] = []
    refs = 0
    plan_precision = declared_precision(protocol)
    inside: dict[str, float] = {}  # result -> the largest move that stayed inside the plan's precision
    for c in changes:
        metric, _setting = metric_of(c["key"])
        if metric and roles.get(metric) == "reference":
            refs += 1
            continue
        if (plan_precision and metric == plan_precision[0] and c["new"] is not None and number_kind(c["key"]) in ("estimate", "end")
                and not (c["key"].endswith("/ok") or c["key"].endswith("/failed")) and abs(c["new"] - c["old"]) <= plan_precision[1]):
            # the plan holds this result to a precision and the move is inside it (a count, a coverage or an error is not)
            inside[metric] = max(inside.get(metric, 0.0), abs(c["new"] - c["old"]))
            continue
        if redrawn and c["new"] is not None and not (c["key"].endswith("/ok") or c["key"].endswith("/failed")):
            se = uncertainty(c["key"], c["old"], stats, ok, kinds)
            if se is not None and abs(c["new"] - c["old"]) / (math.sqrt(2) * se) <= Z:
                continue  # inside the Monte Carlo noise
        keep.append(c)
    if refs:
        notes.append(f"{refs} changed value(s) are references the plan declares: that repairs the plan, not the code")
    for m, move in sorted(inside.items()):
        notes.append(f"{m} moves by {move:.3g}, inside the plan's precision {plan_precision[1]:.3g}")  # type: ignore[index]
    if redrawn:
        notes.append("the repair changes how random numbers are drawn: a number counts only beyond its Monte Carlo uncertainty")
    return keep, notes


# ---- running a copy of the study --------------------------------------------------------------------------------------
def copy_code(quest_root: Path | str, dest: Path | str) -> Path:
    """A copy of the quest's ``code/`` (no history, caches or earlier output) under ``dest``; the folder it is in."""
    dest = Path(dest) / "code"
    shutil.copytree(Path(quest_root) / "code", dest, ignore=_COPY_SKIP)
    return dest


def changed_files(quest_root: Path | str, copy: Path | str) -> dict[str, tuple[str, str]]:
    """``{path: (old text, new text)}`` of the ``.py`` files the copy has changed (or added) against the quest's ``code/``."""
    root, copy = Path(quest_root) / "code", Path(copy)
    out: dict[str, tuple[str, str]] = {}
    for p in sorted(copy.rglob("*.py")):
        rel = p.relative_to(copy)
        if {"output", "__pycache__"} & set(rel.parts):
            continue
        new = p.read_text(encoding="utf-8")
        old = (root / rel).read_text(encoding="utf-8") if (root / rel).is_file() else ""
        if new != old:
            out[rel.as_posix()] = (old, new)
    return out


async def _run(executor: Any, python: Path | str, argv: list[str], cwd: Path, timeout_s: int, env: dict[str, str] | None):
    return await executor.execute([str(python), "-B", *argv], cwd=cwd, timeout_s=int(timeout_s), env=env)


async def run_copy(executor: Any, python: Path | str, quest_root: Path | str, copy: Path, files: list[str], timeout_s: float,
                   env: dict[str, str] | None, work: Path, results_root: Path | str | None = None) -> tuple[str, Path | None, str, float, bool]:
    """Run the study of a copy of ``code/``: ``(mode, folder holding data/ and raw/ it wrote or None, why not, seconds, crashed)``.
    Only the analysis, on the recorded trials, when the changed ``files`` are all analysis files; else the study's own entry
    (``run.py``). A run that ends with a non-zero code after writing its results counts as a run (FI's analysis frame ends
    some studies that way)."""
    began = time.monotonic()
    quest_root = Path(quest_root)
    raw_trials = Path(results_root or quest_root) / "raw" / "trials.json"
    analysis_only = bool(files) and all(Path(f).name in ANALYSIS_FILES and "/" not in f for f in files) and raw_trials.is_file()
    if analysis_only:
        mode = "analysis only"
        out = work / "ao"
        (out / "raw").mkdir(parents=True)
        shutil.copy2(raw_trials, out / "raw" / "trials.json")
        shutil.copytree(copy, out / "code", ignore=_COPY_SKIP)
        env = {**(env or {}), "FI_TRIALS": "raw/trials.json", "FI_RAW_DIR": "raw", "FI_REPLICATE_SEED": "0", "FI_REPLICATE_INDEX": "0"}
        ran = await _run(executor, python, ["code/experiment.py"], out, max(1, int(timeout_s)), env)
        wrote = (out / "data/results/derived.json").is_file()
    else:
        mode = "whole study"
        out = copy / "output"
        if not (copy / "run.py").is_file():
            return mode, None, "the code has no run.py to run the study again with", time.monotonic() - began, False
        ran = await _run(executor, python, ["run.py"], copy, max(1, int(timeout_s)), env)
        wrote = (out / "data/results/derived.json").is_file() and (out / "raw/trials.json").is_file()
    secs = time.monotonic() - began
    if getattr(ran, "timed_out", False):
        return mode, None, f"the repaired study did not finish within {int(timeout_s)} s", secs, False
    if ran.returncode != 0 and not wrote:
        return mode, None, f"the repaired code did not run to the end ({_error_of(ran)})", secs, True
    if not wrote:
        return mode, None, "the repaired study wrote no results", secs, True
    return mode, out, "", secs, False


def _error_of(ran: Any) -> str:
    """The error a failed run printed: its last lines that name an error, else the end of what it printed."""
    text = (ran.stderr or "") + "\n" + (ran.stdout or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    named = [ln for ln in lines if re.search(r"Error|Exception|Unknown|Traceback|not declared|undeclared", ln)]
    return " | ".join((named or lines)[-3:])[:400]


async def judge(executor: Any, python: Path | str, quest_root: Path | str, copy: Path, protocol: Any, timeout_s: float | None,
                env: dict[str, str] | None, *, source: str = "", results_root: Path | str | None = None,
                derived: str = "data/results/derived.json") -> Verdict:
    """Run the study of ``copy`` (the quest's code with a repair applied) and compare what it reports with what the quest
    delivered (``results_root``, the quest folder by default)."""
    changed = await asyncio.to_thread(changed_files, quest_root, copy)
    if not changed:
        return Verdict("could_not", "the repair changed no file", source=source)
    files = sorted(changed)
    if timeout_s is None:
        return Verdict("could_not", "FI has no record of how long the study takes, so a rerun cannot be bounded", source=source, files=files)
    work = Path(tempfile.mkdtemp(prefix="fi_rerun_"))
    try:
        mode, out, why, secs, crashed = await run_copy(executor, python, quest_root, copy, files, timeout_s, env, work, results_root)
        if out is None:
            return Verdict("could_not", why, mode=mode, seconds=secs, source=source, files=files, crashed=crashed)
        base_root = Path(results_root or quest_root)
        base = await asyncio.to_thread(results_of, base_root, derived)
        new = await asyncio.to_thread(results_of, out)
        # a result file the repaired run did not write is not compared (an analysis-only rerun writes only derived.json)
        unwritten = await asyncio.to_thread(result_files, base_root, derived)
        unwritten -= await asyncio.to_thread(result_files, out)
        base = {k: v for k, v in base.items() if not any(k.startswith(f"file:{n}") for n in unwritten)}
        # a rerun that wrote no result at all is a rerun that could not be compared
        if not base or not new:
            return Verdict("could_not", "the delivered or the repaired results hold no numbers to compare", mode=mode,
                           seconds=secs, source=source, files=files)
        redrawn = draws_changed({r: o for r, (o, _n) in changed.items()}, {r: n for r, (_o, n) in changed.items()})
        keep, notes = counted(compare(base, new, REL_TOL if redrawn else EXACT_TOL), protocol,
                              _json(base_root / "raw" / "trials.json"), bool(redrawn))
        keys = [c["key"] for c in keep]
        loose = [k for k in keys if metric_of(k)[0] is None]
        metrics = sorted({m for m in (metric_of(k)[0] for k in keys) if m})
        if loose and not metrics:
            # only numbers FI cannot say which result they belong to: not shipped as "unchanged", and not left out as a
            # guess; the gate places the fault the way it does when no rerun could be made
            return Verdict("could_not", "the repair changes numbers FI cannot attribute to a result: " + ", ".join(loose[:4]),
                           keys=keys, notes=notes, mode=mode, seconds=secs, source=source, files=files)
        if loose:  # they add to what the attributed numbers say, never replace it (the gate places them as well)
            notes = [*notes, "numbers FI cannot attribute to a result also differ: " + ", ".join(loose[:4])]
        return Verdict("changed" if keep else "unchanged", "", metrics=metrics, keys=keys, notes=notes, mode=mode, seconds=secs,
                       source=source, files=files, unplaced=loose)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            _LOG.warning("the folder of a rerun could not be removed (%s)", work)
