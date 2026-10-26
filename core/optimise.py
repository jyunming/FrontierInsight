"""FI runs the search for the best design and keeps its record; the simulation only says what one design gives.

For a quest whose plan is a search for the best design (``study_type: find_best_design``, a ``protocol.optimisation``
block: :mod:`core.optimisation_plan`), the simulation defines ``run_cell(cell)`` (or ``run_trial(cell, trial_id, seed)``
for a study with randomness), as for every other study (:mod:`core.trial_runner`). The ENGINE then runs the search:

- which design comes next is decided in FI's process by :mod:`core.optimise_search` (standard library only; a scipy or
  Optuna method named in the plan runs in the quest's own Python when that environment has the library, one step at a
  time, and never evaluates anything itself: FI is told which design it wants and evaluates it);
- every design is evaluated by the same harness the trials use, in a process of its own, with a nonce the simulation
  never sees: the cell is the design, the conditions held fixed and the search's numerical settings; with randomness,
  ``runs_per_evaluation`` trials with FI's seeds, the same seeds for every design, and their mean;
- the budget (``starts × per_start``) is counted by FI, and ``execution.timeout_s`` bounds the whole search;
- an optional coarse scan (``optimisation.grid``) runs first through the trial runner itself (``raw/trials.json``,
  which the analysis can plot), and is counted apart from the search's budget;
- FI, and only FI, writes the record: ``raw/optimisation_ledger.jsonl`` (one line per evaluation: the design, the
  objective, each constrained quantity, whether it met every limit, the method, the starting point, the time) and
  ``results/best_design.json`` (the best design that meets every limit, the baseline at the same settings, the
  improvement, the method, the evaluations used and whether the budget ran out). Both are written after the last
  evaluation, from FI's own memory, so nothing a simulation wrote there stands; and FI keeps their hashes, so a copy the
  analysis script changed is put back (:func:`restore`).

The best design is found and scored at the search's own numerical settings. Then FI checks it (:mod:`core.optimum_check`):
it evaluates the best designs and the baseline again at the finer settings the plan names, nudges the best one, compares
the starting points, and writes ``needs/OPTIMUM_CHECK.json``; ``results/best_design.json`` then carries the check's
verdict under ``check`` (``checked_at_finer_settings`` is true), and run.log says what each check found. The analysis
runs after the check, with ``FI_OPTIMUM_CHECK`` naming the check's file.

A person who sees the result can ask FI to search further, perhaps toward a target value of the objective (a refine;
:mod:`core.optimise_refine` reads the request and writes ``.fi/optimisation/continue.json``). The search is then the same
search followed by the rounds asked for (:mod:`core.optimise_search`): the part already run is answered from FI's own
record of it when the simulation and the plan are unchanged (nothing is evaluated twice), the rounds' evaluations are
new, and the check runs again on the result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import execution as _execution
from . import flat_output as _flat
from . import in_run as _in_run
from . import metric_spec as _metric_spec
from . import optimisation_plan as _plan
from . import optimise_search as _search
from . import outside_run as _outside
from . import trial_runner as _trials

LEDGER_PATH = Path(_trials.RAW_DIRNAME) / "optimisation_ledger.jsonl"
BEST_PATH = Path("results") / "best_design.json"
WORK_DIR = Path(".fi") / "optimisation"
RECORD = WORK_DIR / "run.json"
#: Each evaluation the search has had answered, written as it is answered (the first line names the search's key), so a
#: search stopped before its end (the quest killed or the machine restarted) is continued on resume, not run again from
#: its start. Removed once the search's record is written.
JOURNAL = WORK_DIR / "searched.jsonl"
SEARCH_SOURCE = WORK_DIR / "optimise_search.py"
#: The rounds of a continued search a person asked for (core/optimise_refine.py), kept for the plan they were asked under.
ROUNDS_PATH = WORK_DIR / "continue.json"
#: Written when the simulation was sent back once because the search's objective is counted in whole steps of one of its
#: numerical settings (``stepped_objective``): after it, the fact is only recorded (once per quest).
STEPPED_SENT = WORK_DIR / "stepped_sent.json"
#: A coarse scan in which a compared setting changed no result the simulation returned (``scan_unchanged``) stops the
#: search before its budget and sends the simulation back to be repaired: after it, the fact is only recorded (once per
#: quest).
UNCHANGED_SENT = WORK_DIR / "scan_unchanged_sent.json"
#: The environment variables that tell the analysis script where FI's record of the search is.
LEDGER_ENV = "FI_OPTIMISATION"
BEST_ENV = "FI_BEST_DESIGN"
#: Whether FI recomputes the best design at the finer numerical settings after the search (core/optimum_check.py).
CHECKS_AT_FINER_SETTINGS = True
_STDERR_KEEP = 20000


def block_of(protocol: Any) -> dict[str, Any] | None:
    """The protocol's optimisation block, or ``None`` for a measurement."""
    block = protocol.get("optimisation") if isinstance(protocol, dict) else None
    return block if isinstance(block, dict) else None


def complete_case(protocol: Any, case: dict[str, Any], *, keep_inside: bool = False) -> dict[str, Any]:
    """An oracle's case as the simulation is called with it during the search: the conditions held fixed, the search's
    numerical settings and the baseline design, with what the case itself names on top (a check of the baseline may
    name only some of its variables).

    The values every evaluation goes through together (``in_each_run``, each a whole list) are the run's, not the
    case's: a check's case names one value of such a factor or none, and the trial runner gives the run all of them
    (:func:`core.trial_runner.run_case`). So they are left out here unless the case names them, or ``keep_inside`` (the
    call is a search evaluation, which is given each list whole)."""
    block = block_of(protocol)
    if block is None or not isinstance(case, dict):
        return dict(case or {})
    baseline = (block.get("baseline") or {}).get("values") if isinstance(block.get("baseline"), dict) else None
    full = _search.cell_of(block, dict(baseline) if isinstance(baseline, dict) else {})
    if not keep_inside:
        inside = _search.in_each_run(block)
        full = {k: v for k, v in full.items() if k not in inside or k in case}
    return {**full, **case}


@dataclass
class SearchRun:
    """What one search did, as FI recorded it."""

    record: dict[str, Any]
    rows: list[dict[str, Any]]
    stderr: str = ""
    reused: bool = False
    scan: bool = False
    load_error: str = ""
    problems: list[str] = field(default_factory=list)
    #: FI's own text of each file it wrote (the ledger, the best design, its record), by path relative to the quest:
    #: what :func:`restore` puts back, from memory, after the analysis ran.
    files: dict[str, str] = field(default_factory=dict)

    @property
    def ok_evaluations(self) -> int:
        return sum(1 for r in self.rows if r.get("status") == "ok")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# What a search left unfinished for a reason another attempt may not share (a package installed since, a longer time
# limit, a repaired helper module): such a search is never reused.
# (A search the time limit stopped is kept for a limit no longer than the one it had.)
_NOT_REUSED = {"failures", "the simulation could not be loaded"}
_NOT_THE_SIMULATION = {"experiment.py", "run.py", "fi_search.py", "submit.py", "replot_figures.py", "replot_layout.py",
                       "web_plots.py"}


def block_sha(block: dict[str, Any]) -> str:
    """The plan's optimisation block, as a hash: the rounds of a continued search hold only for the plan they were asked
    under (a new plan is a new study)."""
    return _sha(json.dumps(block, sort_keys=True, default=str).encode("utf-8"))


def read_rounds(quest_root: Path, block: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The rounds of a continued search asked for under this plan (``block``, normalised), in order; none for another
    plan or without a request."""
    try:
        data = json.loads((Path(quest_root) / ROUNDS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or block is None or data.get("block_sha256") != block_sha(block):
        return []
    return [r for r in data.get("rounds") or [] if isinstance(r, dict) and isinstance(r.get("added"), int)
            and r["added"] > 0]


def _key_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"added": int(r.get("added") or 0), "target": r.get("target"),
             **({"search_target": r["search_target"]} if r.get("search_target") is not None else {})} for r in rounds]


def _run_key(quest_root: Path, simulate: Path, block: dict[str, Any], base: int, entry: str, thresholds: Any,
             rounds: list[dict[str, Any]] | None = None, stop_flat: bool = True, stop_stepped: bool = True,
             stop_unchanged: bool = True) -> str:
    """What decides a search: the simulation and the modules beside it, the files below ``code/`` it may use (the
    model's package, a parameter file), the person's files in ``inputs/`` it may read
    (:func:`core.code_layout.simulation_inputs`), the optimisation block, the seed, the entry, the thresholds, and FI's
    own harness and search code. (A package installed since is not in it: a search that got no result is never kept, so a search
    that failed for a missing package runs again anyway.)"""
    parts: list[bytes] = []
    code = Path(quest_root) / "code"
    for path in sorted([Path(simulate), *(p for p in code.glob("*.py") if p.name not in _NOT_THE_SIMULATION
                                          and p.resolve() != Path(simulate).resolve())]):
        try:
            parts.append(path.name.encode() + b"\0" + path.read_bytes())
        except OSError:
            parts.append(path.name.encode() + b"\0")
    # A repair of the model's package changes what the simulation computes: a search of the code before it is another
    # search. (Only files below code/, so a quest with none keeps the key it had.)
    for path in _below_code(code):
        rel = path.relative_to(code).as_posix().encode()
        try:
            parts.append(rel + b"\0" + path.read_bytes())
        except OSError:
            parts.append(rel + b"\0")
    # Every other file the simulation may read (core/code_layout.py::simulation_inputs): a helper module beside it, a
    # parameter file in code/, a file the person put in the quest's inputs/. Changing one is another search.
    from . import code_layout as _code_layout

    for path in _code_layout.simulation_inputs(quest_root):
        rel = _code_layout.input_label(path, quest_root).encode()
        try:
            parts.append(rel + b"\0" + path.read_bytes())
        except OSError:
            parts.append(rel + b"\0")
    parts.append(_trials.HARNESS_SOURCE.encode("utf-8"))
    parts.append(Path(_search.__file__).read_bytes())
    what: list[Any] = [block, base, entry, thresholds]
    if rounds:
        # A continued search is another search; without rounds the key is the one an earlier version of FI kept.
        what.append(_key_rounds(rounds))
    if not stop_flat:
        # A search that goes on although its objective does not move is another search than one that stops there.
        what.append({"stop_flat": False})
    if not stop_stepped:
        what.append({"stop_stepped": False})
    if not stop_unchanged:
        what.append({"stop_unchanged": False})
    return _sha(b"\1".join(parts) + json.dumps(what, sort_keys=True, default=str).encode())


def _below_code(code: Path) -> list[Path]:
    """The files in folders below ``code/`` the simulation may use (the model's package, a parameter file in it): not
    notes (``*.md``), the package's unit tests (``code/tests/``), caches or hidden folders (``__pycache__``, ``.git``,
    ``.pytest_cache``)."""
    out = []
    for p in code.rglob("*"):
        parts = p.relative_to(code).parts
        if (p.is_file() and len(parts) > 1 and p.suffix.lower() != ".md" and parts[0] != "tests"
                and not any(part == "__pycache__" or part.startswith(".") for part in parts)):
            out.append(p)
    return sorted(out)


def _study_key(quest_root: Path, simulate: Path, block: dict[str, Any], entry: str, thresholds: Any) -> str:
    """What a frozen study's search record must have been made from: :func:`_run_key` without the seed, each file read
    the way ``core/phased.py`` decides what is the same version (line endings and comments do not count), so a study
    the confirm stage takes for the same version always finds its search."""
    from . import phased as _phased

    parts: list[bytes] = []
    code = Path(quest_root) / "code"
    for path in sorted([Path(simulate), *(p for p in code.glob("*.py") if p.name not in _NOT_THE_SIMULATION
                                          and p.resolve() != Path(simulate).resolve())]):
        try:
            parts.append(path.name.encode() + b"\0" + _phased._code_digest(path).encode())
        except OSError:
            parts.append(path.name.encode() + b"\0")
    for path in _below_code(code):  # the model's package, as in :func:`_run_key`
        rel = path.relative_to(code).as_posix().encode()
        try:
            parts.append(rel + b"\0" + (_phased._code_digest(path).encode() if path.suffix == ".py"
                                         else path.read_bytes()))
        except OSError:
            parts.append(rel + b"\0")
    # Every other file the simulation may read (core/code_layout.py::simulation_inputs), read the same way as above.
    from . import code_layout as _code_layout

    for path in _code_layout.simulation_inputs(quest_root):
        rel = _code_layout.input_label(path, quest_root).encode()
        try:
            parts.append(rel + b"\0" + (_phased._code_digest(path).encode() if path.suffix == ".py"
                                         else path.read_bytes()))
        except OSError:
            parts.append(rel + b"\0")
    parts.append(_trials.HARNESS_SOURCE.encode("utf-8"))
    parts.append(Path(_search.__file__).read_bytes())
    return _sha(b"\1".join(parts) + json.dumps([block, entry, thresholds], sort_keys=True, default=str).encode())


def _files(ledger_text: str, best_text: str, record_text: str) -> dict[str, str]:
    return {LEDGER_PATH.as_posix(): ledger_text, BEST_PATH.as_posix(): best_text, RECORD.as_posix(): record_text}


def _save(quest_root: Path, key: str, ledger_text: str, best_text: str, stderr: str, scan: bool,
          study_key: str = "") -> str:
    record = {"key": key, "study_key": study_key,
              "ledger": {"sha256": _sha(ledger_text.encode("utf-8")), "text": ledger_text},
              "best": {"sha256": _sha(best_text.encode("utf-8")), "text": best_text},
              "stderr": stderr[-_STDERR_KEEP:], "scan": scan}
    text = json.dumps(record)
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return text


def _load_record(quest_root: Path) -> dict[str, Any] | None:
    try:
        record = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _record_files(record: dict[str, Any], record_text: str) -> dict[str, str]:
    """FI's files as a saved record holds them (the paths are FI's own, never read from the record)."""
    ledger, best = (record.get("ledger") or {}).get("text"), (record.get("best") or {}).get("text")
    if not isinstance(ledger, str) or not isinstance(best, str):
        return {}
    return _files(ledger, best, record_text)


def restore(quest_root: Path, files: dict[str, str] | None = None) -> bool:
    """Put FI's own ledger and best-design file (and its record of them) back where a script changed or removed them.
    ``files`` is FI's copy from memory (:attr:`SearchRun.files`); without it, the record in ``.fi/optimisation/``.
    Only FI's own paths are ever written. ``True`` when one was put back."""
    quest_root = Path(quest_root)
    if files is None:
        try:
            record_text = (quest_root / RECORD).read_text(encoding="utf-8")
            record = json.loads(record_text)
        except (OSError, ValueError):
            return False
        files = _record_files(record, record_text) if isinstance(record, dict) else {}
    restored = False
    for rel in (LEDGER_PATH.as_posix(), BEST_PATH.as_posix(), RECORD.as_posix()):
        text = files.get(rel)
        if not isinstance(text, str):
            continue
        path = quest_root / rel
        try:
            current = path.read_bytes()
        except OSError:
            current = None
        if current != text.encode("utf-8"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            restored = True
    return restored


def searches_itself(texts: dict[str, str]) -> list[str]:
    """What in the simulation's own code may search for a design (an optimiser imported or called): FI runs the search,
    so a simulation that searches inside ``run_cell`` would report the value of a design FI did not ask for. Solving an
    equation (``brentq``, ``fsolve``, ``root``, ``newton``) or fitting a curve is not listed. A physical model can
    minimise on its own (an equilibrium shape by its least energy), so this is a warning on the record, not a stop."""
    import ast

    optimisers = {"minimize", "minimize_scalar", "differential_evolution", "dual_annealing", "basinhopping", "shgo",
                  "direct", "brute"}
    packages = {"optuna", "skopt", "nevergrad", "hyperopt", "bayes_opt", "pyswarms", "deap"}
    found: list[str] = []
    for name, text in texts.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in packages:
                        found.append(f"{name} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in packages:
                    found.append(f"{name} imports {node.module}")
                elif node.module.startswith("scipy.optimize"):
                    found += [f"{name} imports scipy.optimize.{a.name}" for a in node.names if a.name in optimisers]
            elif isinstance(node, ast.Attribute) and node.attr in optimisers:
                base = ast.unparse(node.value) if hasattr(ast, "unparse") else ""
                if base.endswith("optimize") or base in {"so", "opt", "sopt"}:
                    found.append(f"{name} calls {base}.{node.attr}")
    return list(dict.fromkeys(found))


def read(quest_root: Path) -> dict[str, Any] | None:
    """``results/best_design.json`` as FI wrote it (put back first if a script changed it), or ``None``."""
    restore(quest_root)
    try:
        data = json.loads((Path(quest_root) / BEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _mean_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each returned quantity averaged over the runs. A plain number gives a number; a quantity returned per value of what
    each run compares inside it (``{level key: number}``) is averaged per key and stays a mapping (the harness gives every
    run the same keys, so none is left out)."""
    names = set.intersection(*(set(r.get("values") or {}) for r in rows)) if rows else set()
    out: dict[str, Any] = {}
    for name in sorted(names):
        parts = [r["values"][name] for r in rows]
        if isinstance(parts[0], dict):
            keys = [k for k in parts[0] if all(isinstance(p, dict) and k in p for p in parts)]
            out[name] = {k: sum(float(p[k]) for p in parts) / len(parts) for k in keys}
        else:
            out[name] = sum(float(p) for p in parts) / len(parts)
    return out


async def _evaluate(executor: Any, python: Any, quest_root: Path, module: str, entry: str, cell: dict[str, Any], *,
                    runs: int, base_seed: int, timeout_s: int, env: dict[str, str] | None,
                    thresholds: dict[str, Any] | None, seed_key: str = "", name: str = "eval",
                    outside: Any = None, in_each_run: dict[str, list[Any]] | None = None) -> tuple[dict[str, Any], str, str]:
    """One design, in a process of its own through FI's harness: ``(answer for the search, stderr, load error)``. The
    answer's ``trials`` holds each run's own values (in trial order), beside their mean in ``values``. ``seed_key`` picks
    another family of seeds (the check at finer settings uses seeds the search never used). ``name``: the files of this
    evaluation, and ``outside``: the executor several evaluations at once share, with the harness already written
    (:func:`evaluate_many`)."""
    work = quest_root / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    spec, out = work / f"{name}.json", work / f"{name}.out.jsonl"
    out.unlink(missing_ok=True)
    trials = [{"trial": t, "seed": None if entry == "run_cell" else _trials.trial_seed(base_seed, seed_key, t)}
              for t in range(1 if entry == "run_cell" else max(1, runs))]
    nonce = _sha(f"optimise|{time.time_ns()}|{id(trials)}".encode())[:24]
    # The values each run compares inside it are given whole, as the scan and the timing give them, and the run may return
    # a number per value (core/in_run.py::read_returned).
    inside = {str(k): list(v) for k, v in (in_each_run or {}).items()}
    spec.write_text(json.dumps({"module": module, "entry": entry, "cell": {**cell, **inside},
                                "levels": _in_run.levels_json(_in_run.level_cells(inside)), "trials": trials,
                                "nonce": nonce, "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
    if outside is None:
        # Fresh for every evaluation: nothing an earlier evaluation wrote there is run.
        (quest_root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
    started = time.monotonic()
    # From a copy of code/ outside the quest folder (core/outside_run.py), as every cell of the study runs.
    result = await (outside or _outside.OutsideExecutor(executor, quest_root)).execute(
        [str(python), _trials.HARNESS_PATH.as_posix(), spec.relative_to(quest_root).as_posix(),
         out.relative_to(quest_root).as_posix()],
        cwd=quest_root, timeout_s=max(1, math.ceil(timeout_s)), env=env,
    )
    elapsed = round(time.monotonic() - started, 4)
    stderr = getattr(result, "stderr", "") or ""
    if getattr(result, "timed_out", False):
        # Cut off by the search's clock: not a failed design, and not counted.
        return {"stop": "time", "elapsed_s": elapsed}, stderr, ""
    try:
        lines = out.read_text(encoding="utf-8").splitlines() if out.is_file() else []
    except OSError:
        lines = []
    reported: dict[int, dict[str, Any]] = {}
    twice: set[int] = set()
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("nonce") != nonce:
            continue  # not written by the harness FI started for this design
        if row.get("load_error"):
            why = f"the simulation could not be loaded ({row['load_error']})"
            return {"failed": why, "elapsed_s": elapsed}, stderr, why
        if isinstance(row.get("trial"), int):
            if row["trial"] in reported:
                twice.add(row["trial"])
            reported.setdefault(row["trial"], row)
    rows = []
    for t in trials:
        row = reported.get(t["trial"])
        if row is None:
            why = (f"the evaluation's process stopped (exit code {getattr(result, 'returncode', '?')}) before it "
                   "reported")
            # Stopped from outside (memory, a kill) as often as by the design: not an answer a resume may take again.
            return {"failed": why, "elapsed_s": elapsed, "process": True}, stderr, ""
        if t["trial"] in twice:
            return {"failed": "a run of this design was reported more than once", "elapsed_s": elapsed,
                    "process": True}, stderr, ""
        if row.get("status") != "ok" or row.get("seed") != t["seed"]:
            why = str(row.get("reason") or "the reported seed is not the one FI gave") if row.get("status") != "ok" \
                else "the reported seed is not the one FI gave"
            return {"failed": why, "elapsed_s": elapsed}, stderr, ""
        rows.append(row)
    return {"values": _mean_values(rows), "trials": [dict(r.get("values") or {}) for r in rows],
            "elapsed_s": elapsed}, stderr, ""


async def search_entry_returns_objective(
    executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any], cell: dict[str, Any], *,
    timeout_s: int, env: dict[str, str] | None, thresholds: dict[str, Any] | None, log: Any = None,
) -> str:
    """After a repair of the simulation: ``""`` when the search's own evaluation of one design (:func:`_evaluate`, with the
    function the search calls, :func:`core.trial_runner.study_entry`) gives a reply the search reads an objective from
    (:func:`core.optimise_search.judge` finds the objective in it), else a plain sentence. ``""`` also when the evaluation
    failed or did not finish (judged by the runs that follow)."""
    entry = _trials.study_entry(_trials.entries(quest_root / module))
    if not entry:
        return ""
    answer, _stderr, load_error = await _evaluate(executor, python, quest_root, module, entry, dict(cell), runs=1, base_seed=0,
                                                  timeout_s=timeout_s, env=env, thresholds=thresholds, name="entry_check",
                                                  in_each_run=_search.in_each_run(block) or None)
    values = answer.get("values") if isinstance(answer, dict) else None
    if load_error or not isinstance(values, dict):
        if log is not None:
            log.warning("[execute] the repaired code was not judged on the search's baseline design: its own evaluation "
                        "gave no result (%s)", load_error or (answer.get("failed") if isinstance(answer, dict) else None)
                        or ("it ran out of time" if isinstance(answer, dict) and answer.get("stop") else "no reason given"))
        return ""
    quantity = str(block["objective"]["quantity"])
    if quantity in values:
        return ""
    return (f"the repaired code no longer returns the study's results: {entry}() on the search's baseline design returned "
            f"{', '.join(sorted(map(str, values))[:8]) or 'nothing'}, not the objective `{quantity}`")


class EvaluationPool:
    """Designs that do not depend on each other's answers, up to ``at_once`` at the same time, each in a process of its
    own exactly as :func:`_evaluate` runs one (the same seeds, so the answers are those of one after another). A design
    joins the pool the moment it is submitted and a place frees up the moment another finishes: nothing waits for the
    slowest design of a group. :meth:`submit` starts a design, :meth:`wait` returns the ones that have finished, and
    :meth:`close` waits for the rest and clears the shared folder (the locks and temporary files a design stopped for its
    time left there, :meth:`core.outside_run.OutsideExecutor.settle`). A design not started before ``deadline`` is
    answered ``{"stop": "time"}``. ``on_done(id, answer, load error)`` is called as each one finishes (its journal line,
    the progress), so a long run keeps what it finished if the quest is stopped."""

    def __init__(self, executor: Any, python: Any, quest_root: Path, module: str, entry: str, *, at_once: int, runs: int,
                 base_seed: int, deadline: float, env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                 seed_key: str = "", on_done: Any = None, log: Any = None, gate: Any = None,
                 in_each_run: dict[str, list[Any]] | None = None):
        self.executor, self.python, self.quest_root = executor, python, quest_root
        self.in_each_run = in_each_run
        self.module, self.entry, self.runs, self.base_seed = module, entry, runs, base_seed
        self.deadline, self.thresholds, self.seed_key, self.on_done = deadline, thresholds, seed_key, on_done
        (quest_root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
        self.outside = _outside.OutsideExecutor(executor, quest_root, log=log)
        self.log = log
        # Fewer at once when the memory would not hold them. A search passes the one gate it shares by all its batches, so
        # the size measured by the batches before is kept; a gate made here starts again from nothing.
        self.gate = gate if gate is not None else _execution.SettingsGate(max(1, at_once), log=log)
        self.outside.meter = self.gate.note
        self.env = _execution.threads_beside(env, max(1, at_once))  # each process its share of the cores
        self.work = quest_root / WORK_DIR
        self.tasks: dict[asyncio.Task, Any] = {}
        self.serial = 0

    def submit(self, ident: Any, cell: dict[str, Any]) -> None:
        self.serial += 1
        name = f"eval{self.serial}"
        task = asyncio.ensure_future(self._one(ident, name, cell))
        self.tasks[task] = ident

    async def _one(self, ident: Any, name: str, cell: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        async with self.gate.slot():
            left = self.deadline - time.monotonic()
            if left <= 0:
                got: tuple[dict[str, Any], str, str] = ({"stop": "time"}, "", "")
            else:
                try:
                    got = await _evaluate(self.executor, self.python, self.quest_root, self.module, self.entry, cell,
                                          runs=self.runs, base_seed=self.base_seed, timeout_s=left, env=self.env,
                                          thresholds=self.thresholds, seed_key=self.seed_key, name=name,
                                          outside=self.outside, in_each_run=self.in_each_run)
                except Exception as exc:  # noqa: BLE001 -- one design's trouble (a file it could not write) is its answer
                    got = {"failed": f"FI could not run this design ({type(exc).__name__}: {exc})"[:300]}, "", ""
                finally:
                    for left_over in (self.work / f"{name}.json", self.work / f"{name}.out.jsonl"):
                        left_over.unlink(missing_ok=True)
            if self.on_done is not None:
                self.on_done(ident, got[0], got[2])
            return got

    @property
    def running(self) -> int:
        return len(self.tasks)

    async def wait(self) -> list[tuple[Any, tuple[dict[str, Any], str, str]]]:
        """The designs that finished, ``[(id, (answer, stderr, load error))]``: waits until there is one (at least one
        design must be running)."""
        if not self.tasks:
            return []
        done, _pending = await asyncio.wait(list(self.tasks), return_when=asyncio.FIRST_COMPLETED)
        out = []
        for task in [t for t in self.tasks if t in done]:  # in the order they were submitted
            out.append((self.tasks.pop(task), task.result()))
        return out

    async def close(self, discard: bool = False) -> list[tuple[Any, tuple[dict[str, Any], str, str]]]:
        """Every design still running, waited for and returned; then the shared folder is cleared. With ``discard`` (the
        search no longer wants them, or the quest was stopped) the designs still running are not waited for: their
        processes are stopped first, then the folder is cleared."""
        out = []
        try:
            while self.tasks and not discard:
                out.extend(await self.wait())
        finally:
            running = list(self.tasks)  # only left when discarding, or when the wait itself was cut short
            for task in running:
                task.cancel()
            try:
                for task, ended in zip(running, await asyncio.gather(*running, return_exceptions=True)):
                    if isinstance(ended, BaseException) and not isinstance(ended, asyncio.CancelledError):
                        # a real fault of a design's run, not the stop we asked for: say so once, with what it was
                        (getattr(self, "log", None) or logging.getLogger(__name__)).warning(
                            "[optimise] a design that was running when the search stopped failed: %s: %s",
                            type(ended).__name__, str(ended)[:300])
            finally:
                self.tasks.clear()
                await self.outside.settle()
        return out


async def evaluate_many(executor: Any, python: Any, quest_root: Path, module: str, entry: str,
                        cells: list[dict[str, Any]], *, at_once: int, runs: int, base_seed: int, deadline: float,
                        env: dict[str, str] | None, thresholds: dict[str, Any] | None,
                        seed_key: str = "", on_done: Any = None, log: Any = None,
                        gate: Any = None,
                        in_each_run: dict[str, list[Any]] | None = None) -> list[tuple[dict[str, Any], str, str]]:
    """Designs that do not depend on each other's answers, up to ``at_once`` at the same time, each in a process of its
    own exactly as :func:`_evaluate` runs one (the same seeds, so the answers are those of one after another):
    ``[(answer, stderr, load error)]`` in the order of ``cells``. A design not started before ``deadline`` is answered
    ``{"stop": "time"}``. ``on_done(k, answer, load error)`` is called as each one finishes (its journal line, the
    progress), so a long batch keeps what it finished if the quest is stopped."""
    if len(cells) <= 1 or at_once <= 1:
        out = []
        for cell in cells:
            left = deadline - time.monotonic()
            got = (await _evaluate(executor, python, quest_root, module, entry, cell, runs=runs, base_seed=base_seed,
                                   timeout_s=left, env=env, thresholds=thresholds, seed_key=seed_key,
                                   in_each_run=in_each_run)
                   if left > 0 else ({"stop": "time"}, "", ""))
            if on_done is not None:
                on_done(len(out), got[0], got[2])
            out.append(got)
        return out
    pool = EvaluationPool(executor, python, quest_root, module, entry, at_once=at_once, runs=runs, base_seed=base_seed,
                          deadline=deadline, env=env, thresholds=thresholds, seed_key=seed_key, on_done=on_done, log=log,
                          gate=gate, in_each_run=in_each_run)
    for k, cell in enumerate(cells):
        pool.submit(k, cell)
    answers: dict[int, tuple[dict[str, Any], str, str]] = {}
    for ident, got in await pool.close():
        answers[ident] = got
    return [answers[k] for k in range(len(cells))]


def _read_journal(path: Path, key: str, space: Any) -> tuple[dict[str, dict[str, Any]], float]:
    """The answers a stopped search of ``key`` had (keyed as the replay is) and the seconds its evaluations took. A
    journal of another key (other code, plan or seed) is not this search's: nothing is taken from it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}, 0.0
    try:
        head = json.loads(lines[0]) if lines else None
    except ValueError:
        head = None
    if not isinstance(head, dict) or head.get("key") != key:
        return {}, 0.0
    answers: dict[str, dict[str, Any]] = {}
    seconds = 0.0
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # the line being written when the search stopped
        if not (isinstance(row, dict) and isinstance(row.get("design"), dict) and isinstance(row.get("reply"), dict)):
            continue
        try:
            at = _replay_key(space, row["design"], row.get("level"))
        except (KeyError, TypeError, ValueError):
            continue
        answers[at] = row["reply"]
        if isinstance(row["reply"].get("elapsed_s"), (int, float)):
            seconds += float(row["reply"]["elapsed_s"])
    return answers, seconds


def _replay_key(space: Any, design: dict[str, Any], level: dict[str, Any] | None) -> str:
    """What an evaluation already answered is found by: the design, and the alternative it was evaluated under when the
    study compares named alternatives (the same design is another evaluation under each of them)."""
    return f"{_search.level_key(level)}|{space.key(space.canonical(design))}"


def _journal(path: Path, design: dict[str, Any], reply: dict[str, Any], level: dict[str, Any] | None = None) -> None:
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"design": design, "reply": reply, **({"level": level} if level else {})},
                                default=str) + "\n")
    except OSError:
        pass  # the journal only saves time on a resume; the search goes on without it


def _fmt_pace(per_minute: float) -> str:
    return f"{per_minute:.0f}" if per_minute >= 10 else f"{per_minute:.2g}"


async def _drive(executor: Any, python: Any, quest_root: Path, spec: dict[str, Any], *, timeout_s: int,
                 env: dict[str, str] | None) -> dict[str, Any]:
    """One step of a library's method, in the quest's own Python (where the library is, or is not, installed)."""
    work = quest_root / WORK_DIR
    spec_path, out = work / "drive.json", work / "drive.out.json"
    out.unlink(missing_ok=True)
    nonce = _sha(f"drive|{time.time_ns()}".encode())[:24]
    spec_path.write_text(json.dumps({**spec, "nonce": nonce}), encoding="utf-8")
    # FI's own search code, fresh for every step: nothing an evaluation wrote there is run.
    (quest_root / SEARCH_SOURCE).write_text(Path(_search.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    result = await _outside.OutsideExecutor(executor, quest_root).execute(
        [str(python), SEARCH_SOURCE.as_posix(), "--drive", spec_path.relative_to(quest_root).as_posix(),
         out.relative_to(quest_root).as_posix()],
        cwd=quest_root, timeout_s=max(1, math.ceil(timeout_s)), env=env,
    )
    try:
        reply = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = (getattr(result, "stderr", "") or "").strip()[-300:]
        return {"error": f"the method's step did not answer (exit code {getattr(result, 'returncode', '?')}){': ' + tail if tail else ''}"}
    if not isinstance(reply, dict) or reply.get("nonce") != nonce:
        return {"error": "the method's answer was not the one FI asked for"}
    reply.pop("nonce", None)
    return reply


async def run_search(executor: Any, python: Any, quest_root: Path, module: str, protocol: dict[str, Any], *,
                     base_seed: int, timeout_s: int, env: dict[str, str] | None = None, log: Any = None,
                     frozen: bool = False, at_once: int = 1, stop_flat: bool = True,
                     stop_stepped: bool = True, stop_unchanged: bool = True, run_id: str = "") -> SearchRun:
    """Run the search for the best design (see the module docstring) and write its record. ``module`` is the simulation
    file relative to ``quest_root``; ``timeout_s`` bounds the whole search, the coarse scan included.

    ``frozen``: the study is frozen for its confirm run (explore, then confirm: ``core/phased.py``). The search is not
    run again: the best design exploration's search found, for this same simulation and plan, is kept, and only FI's
    check of it (with the confirm run's new seeds, :mod:`core.optimum_check`) runs. Searching again on the confirm seeds
    would choose the design on the data meant only to confirm it. A frozen study whose search record is not there to
    keep cannot be confirmed without searching again: it gets no result.

    ``stop_flat``: whether a search whose objective does not move stops early and hands that fact back (the plan is then
    revised); off, the search runs its budget and only records the fact. ``stop_stepped``: the same for an objective
    counted in whole steps of one of the search's numerical settings (the simulation is then repaired)."""
    quest_root = Path(quest_root)
    raw_block = block_of(protocol)
    if raw_block is None:
        raise ValueError("the protocol has no optimisation block")
    # Read through the plan's own check, so a limit, a budget or a kind means here what plan.md says it means.
    block, why = _plan.normalize(raw_block)
    if block is None or not isinstance(block.get("evaluation_budget"), dict):
        why = (f"the plan's optimisation block cannot drive a search ({why})" if block is None else
               "the plan's optimisation block has no evaluation budget (`evaluation_budget`: `starts` and `per_start`)")
        return SearchRun(record={}, rows=[], problems=[why])
    simulate = quest_root / module
    have = _trials.entries(simulate)
    entry = _trials.study_entry(have)
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
    if not entry:
        why = (f"{Path(module).name} defines neither run_cell(cell) nor run_trial(cell, trial_id, seed): FI calls one of "
               "them for each design the search tries")
        return SearchRun(record={}, rows=[], load_error=why, problems=[why])
    rounds = read_rounds(quest_root, block)
    key = _run_key(quest_root, simulate, block, int(base_seed), entry, thresholds, rounds, stop_flat, stop_stepped,
                   stop_unchanged)
    # The same simulation and plan whatever the seed: what a frozen study's search record must have been made from.
    study_key = _study_key(quest_root, simulate, block, entry, thresholds)
    try:
        record_text = (quest_root / RECORD).read_text(encoding="utf-8")
        record = json.loads(record_text)
    except (OSError, ValueError):
        record_text, record = "", None
    if frozen and isinstance(record, dict) and not record.get("study_key"):
        # A search record from before FI kept its study key: it is exploration's when its key is one exploration's
        # seeds give (core/phased.py keeps them).
        from . import phased as _phased

        seeds = (_phased.load(quest_root) or {}).get("explore_seed_bases") or []
        if record.get("key") in {_run_key(quest_root, simulate, block, int(s), entry, thresholds) for s in seeds}:
            record = {**record, "study_key": study_key}
    if frozen and not (isinstance(record, dict) and record.get("study_key") == study_key):
        why = ("the study is frozen for its confirm run, and the search exploration ran for this simulation and plan is "
               "not kept, so its best design cannot be confirmed without searching again on the confirm seeds")
        return SearchRun(record={}, rows=[], problems=[why])
    # A continued search repeats the search it continues: when FI's record of that one (the same simulation, plan, seed
    # and the rounds before this one) is on disk, its evaluations are answered from the record, not run again.
    replay: dict[str, dict[str, Any]] = {}
    replay_scan: dict[str, list[dict[str, Any]]] | None = None
    scan_carried: dict[str, Any] = {}
    if rounds and not frozen and isinstance(record, dict) and record.get("key") != key and any(
            record.get("key") == _run_key(quest_root, simulate, block, int(base_seed), entry, thresholds, rounds[:k], stop_flat,
                                          stop_stepped, stop_unchanged)
            for k in range(len(rounds))):
        earlier = _record_files(record, record_text).get(LEDGER_PATH.as_posix(), "")
        space = _search.Space(block["design_variables"])
        quantity = str(block["objective"]["quantity"])
        scan_seen: dict[str, list[dict[str, Any]]] = {}
        for line in earlier.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") != "evaluation" or not isinstance(row.get("design"), dict):
                continue
            # The objective, the limits and the plan's named results (the row keeps them), as the simulation gave them.
            answer: dict[str, Any] = ({"values": {quantity: row["objective"], **(row.get("constraints") or {}),
                                                  **(row.get("results") if isinstance(row.get("results"), dict) else {})}}
                                      if row.get("status") == "ok" else {"failed": str(row.get("reason") or "failed")})
            answer["elapsed_s"] = row.get("elapsed_s")
            if row.get("stage") == "scan":
                scan_seen.setdefault(_search.level_key(row.get("level")), []).append(
                    {"design": row["design"], "values": answer.get("values"), "reason": answer.get("failed", ""),
                     "elapsed_s": row.get("elapsed_s")})
            else:
                replay[_replay_key(space, row["design"], row.get("level"))] = answer
        if scan_seen and (quest_root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME).is_file():
            replay_scan = scan_seen
            # The scan is not run again, so what FI measured in it (settings that changed no result, numbers that differ
            # on a second call) is taken from the record of the search being continued: a continued search says it too.
            try:
                earlier_best = json.loads(_record_files(record, record_text).get(BEST_PATH.as_posix(), ""))
            except ValueError:
                earlier_best = None
            if isinstance(earlier_best, dict):
                scan_carried = {k: earlier_best[k] for k in ("scan_settings_without_effect", "unsteady")
                                if earlier_best.get(k)}
        if log is not None and replay:
            log.info("[optimise] continuing the search FI already ran (%d evaluation(s) of it are taken from its record, "
                     "not run again), with %d more evaluation(s) asked for", len(replay),
                     sum(int(r.get("added") or 0) for r in rounds[-1:]))
    if isinstance(record, dict) and (record.get("key") == key or frozen):
        files = _record_files(record, record_text)
        rows = [json.loads(line) for line in files.get(LEDGER_PATH.as_posix(), "").splitlines() if line.strip()]
        try:
            best = json.loads(files.get(BEST_PATH.as_posix(), ""))
        except ValueError:
            best = None
        scan_there = not record.get("scan") or (quest_root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME).is_file()
        worked = any(r.get("event") == "evaluation" and r.get("status") == "ok" for r in rows)
        stopped = ((best or {}).get("evaluations") or {}).get("stopped_because")
        limit = ((best or {}).get("time") or {}).get("limit_seconds")
        # A search the clock stopped stands for the same limit (an analysis repair), not for a longer one.
        timely = stopped != "time" or (isinstance(limit, (int, float)) and int(timeout_s) <= limit)
        if isinstance(best, dict) and worked and scan_there and (frozen or (stopped not in _NOT_REUSED and timely)):
            restore(quest_root, files)
            if log is not None:
                log.info("[optimise] %s", "confirm run: the best design exploration's search found is kept and checked "
                         "once more on new seeds; the search is not run again (that would choose the design on the "
                         "confirm seeds)" if frozen else
                         "the simulation and the plan are unchanged: the search FI already ran is used")
                for line in summary_lines(best):
                    log.info("%s", line)
            return SearchRun(record=best, rows=[r for r in rows if r.get("event") == "evaluation"],
                             stderr=str(record.get("stderr") or ""), reused=True, scan=bool(record.get("scan")),
                             files=files)
    if frozen:
        why = ("the study is frozen for its confirm run, and the search exploration ran found no design to keep, so "
               "nothing can be confirmed without searching again on the confirm seeds")
        if isinstance(record, dict) and record.get("scan") \
                and not (quest_root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME).is_file():
            why = (f"the study is frozen for its confirm run, and the coarse scan of exploration's search "
                   f"({_trials.RAW_DIRNAME}/{_trials.SUMMARY_NAME}) is missing, so the kept search cannot be used as it "
                   "was and nothing can be confirmed without searching again on the confirm seeds")
        return SearchRun(record={}, rows=[], problems=[why])
    work = quest_root / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    # An earlier search's record and files never stand beside this one's (a search cut short is not kept either).
    # The same goes for an earlier search's check at finer settings (core/optimum_check.py).
    for rel in (RECORD, LEDGER_PATH, BEST_PATH, Path("needs") / "OPTIMUM_CHECK.json", WORK_DIR / "check.json",
                Path("results") / "best_design.md"):
        (quest_root / rel).unlink(missing_ok=True)
    (quest_root / _trials.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        texts = {p.name: p.read_text(encoding="utf-8") for p in
                 [simulate, *((quest_root / "code").glob("*.py"))] if p.name not in _NOT_THE_SIMULATION}
        warnings = searches_itself(texts)
    except OSError:
        pass
    if warnings and log is not None:
        log.warning("[optimise] the simulation's own code may search for a design itself (%s): FI runs the search and "
                    "each call must compute the one design it is given", "; ".join(warnings[:5]))
    # A search of this same key stopped before its end: the evaluations it had answered are taken from its journal.
    # The search is the same from the same seed, so it asks for them again in the same order and goes on from there.
    journal = quest_root / JOURNAL
    journaled, before_s = _read_journal(journal, key, _search.Space(block["design_variables"]))
    if journaled:
        replay.update(journaled)
        try:
            with journal.open("rb") as fh:
                fh.seek(-1, 2)
                cut = fh.read(1) != b"\n"
            if cut:
                with journal.open("a", encoding="utf-8") as fh:
                    fh.write("\n")
        except OSError:
            pass
        if log is not None:
            log.info("[optimise] continuing the search that stopped before its end: %d evaluation(s) it had done are "
                     "taken from FI's record of them, not run again", len(journaled))
    else:
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({"key": key}) + "\n", encoding="utf-8")
    started = time.monotonic()
    deadline = started + max(1, int(timeout_s))
    runs = int(block.get("runs_per_evaluation") or 1) if entry == "run_trial" else 1
    stderr_parts: list[str] = []
    method, _where = _plan.effective_method(block)

    scan_rows: list[dict[str, Any]] = []
    scan_note: list[str] = []
    scan_unsteady: set[str] = set()
    scan_unchanged: list[tuple[str, int]] = []  # the compared settings that changed no result (stop_unchanged)
    if not (isinstance(block.get("grid"), dict) and block["grid"]):
        # No coarse scan: an earlier measurement's trial record is not this search's.
        for rel in (Path(_trials.RAW_DIRNAME) / _trials.LEDGER_NAME, Path(_trials.RAW_DIRNAME) / _trials.SUMMARY_NAME,
                    _trials.RUN_RECORD):
            (quest_root / rel).unlink(missing_ok=True)

    scanned: dict[str, dict[str, Any]] = {}

    async def scan(grid: dict[str, list[Any]], scan_block: dict[str, Any] | None = None,
                   level: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """The coarse scan, through the trial runner itself (``raw/trials.json``, which the analysis can plot). It
        measures every alternative of a compared condition together and runs once: each alternative's search is given
        the designs of the scan at its own alternative (``scan_block``)."""
        if replay_scan is not None:
            return replay_scan.get(_search.level_key(level), [])
        grid_key = json.dumps({k: list(v) for k, v in grid.items()}, sort_keys=True, default=str)
        if grid_key in scanned:
            return _search.scan_rows(scanned[grid_key], scan_block or block)
        left = deadline - time.monotonic()
        if left <= 0:
            return []
        trial_run = await _trials.run_trials(
            executor, python, quest_root, module, grid, runs_per_setting=runs, base_seed=int(base_seed),
            deterministic=entry == "run_cell", timeout_s=max(1, math.ceil(left)), env=env,
            thresholds=thresholds, paired=True, in_each_run=_search.in_each_run(block) or None, at_once=at_once,
            run_id=run_id,
        )
        stderr_parts.append(trial_run.stderr())
        summary = json.loads(trial_run.summary_path.read_text(encoding="utf-8"))
        scanned[grid_key] = summary
        reasons = {c.key: next((r.get("reason") for r in c.rows if r.get("reason")), "") for c in trial_run.cells}
        for cell in summary.get("cells") or []:
            cell["reason"] = reasons.get(cell.get("key"), "")
        rows = _search.scan_rows(summary, scan_block or block)
        # A number that differs when the same setting is called again (a timing) is no result of any setting.
        unsteady = await _trials.unsteady_names(
            executor, python, quest_root, module, trial_run.cells,
            timeout_s=max(1, math.ceil(deadline - time.monotonic())), env=env, thresholds=thresholds,
            in_each_run=_search.in_each_run(block) or None,
            log=log) if entry == "run_cell" and deadline - time.monotonic() > 1 else set()
        scan_unsteady.update(unsteady)
        # Every returned number is compared here, not only the plan's declared metrics: a search's results are also its
        # objective and its limits' quantities (not metrics), and an alternative that differs in one of them is not
        # "unchanged" (a level that gives the first one's results is skipped later, not sent back).
        unchanged = _flat.unchanged_axes(trial_run.cells, skip=_trials.numerical_axes(protocol), ignore=unsteady)
        if unchanged:
            scan_note.append(_flat.unchanged_axes_note(unchanged))
            if log is not None:
                log.warning("[optimise] the coarse scan: %s", scan_note[-1])
            if stop_unchanged:
                # FI measured that these settings reach no result: the search would compare one result repeated, so it
                # stops here and the simulation is sent back (OptimiseRunner._unchanged_result).
                scan_unchanged[:] = unchanged
        if log is not None:
            ran = sum(1 for c in summary.get("cells") or [] if c.get("level") is None)
            worked = sum(1 for r in rows if r.get("values") is not None)
            if ran > len(rows):
                log.info("[optimise] the coarse scan ran %d setting(s), not counted in the search's budget: %d of them "
                         "under the conditions the search holds (%d worked) can start the search, the other %d are "
                         "comparisons kept for the paper", ran, len(rows), worked, ran - len(rows))
            else:
                log.info("[optimise] the coarse scan ran %d design(s) after the baseline (%d worked); it is not counted "
                         "in the search's budget", len(rows), worked)
        return rows

    results = watched_results(protocol)
    gen = _search.compare(block, seed=int(base_seed), method=method, rounds=rounds, results=results, batch=at_once > 1,
                          stop_flat=stop_flat, stop_stepped=stop_stepped, stream=at_once > 1)
    replay_space = _search.Space(block["design_variables"]) if replay else None
    load_error = ""
    done_outcome: dict[str, Any] | None = None
    last_note = time.monotonic()
    evaluated = 0
    pool: EvaluationPool | None = None  # the designs running now, when the search is answered as each one finishes
    # One memory gate for the whole search: every batch and pool of it shares the size of a setting measured so far.
    gate = _execution.SettingsGate(max(1, at_once), log=log)
    streamed: dict[Any, dict[str, Any]] = {}  # the request of each design in the pool, by its id

    def finished_streamed(ident: Any, answer: dict[str, Any], loaded: str) -> None:
        nonlocal evaluated, last_note
        asked_one = streamed.get(ident) or {}
        if ("values" in answer or "failed" in answer) and not loaded and not answer.get("process") \
                and isinstance(asked_one.get("design"), dict):
            _journal(journal, asked_one["design"], answer, asked_one.get("level"))
        if "values" in answer or "failed" in answer:
            evaluated += 1
        if log is not None and time.monotonic() - last_note > 60:
            last_note = time.monotonic()
            log.info("[optimise] still searching: %d evaluation(s) so far (%d at the same time)", evaluated, at_once)

    try:
        request = next(gen)
        while True:
            if pool is not None and pool.running and request["kind"] != "evaluate_stream":
                # designs asked for before a stop (or by a start that is over): not needed now, and never taken for the
                # same design of the next alternative
                await pool.close(discard=True)
                pool = None
                streamed.clear()
            if request["kind"] == "scan":
                reply = await scan(request["grid"], request.get("block"), request.get("level"))  # type: ignore[assignment]
                scan_rows = scan_rows or list(reply)
            elif request["kind"] == "evaluate_many":
                # Designs the method chose before reading any of their answers: up to ``at_once`` at the same time.
                asked = request["requests"]
                answers: list[Any] = [None] * len(asked)
                fresh = []
                for k, one in enumerate(asked):
                    replayed = (replay.get(_replay_key(replay_space, one["design"], one.get("level")))
                                if replay_space is not None and isinstance(one.get("design"), dict) else None)
                    if scan_unchanged:
                        answers[k] = {"stop": "scan_unchanged"}
                    elif replayed is not None:
                        answers[k] = dict(replayed)
                    elif load_error:
                        answers[k] = {"stop": "the simulation could not be loaded"}
                    else:
                        fresh.append(k)
                def finished(j: int, answer: dict[str, Any], loaded: str) -> None:
                    nonlocal evaluated, last_note
                    if ("values" in answer or "failed" in answer) and not loaded and not answer.get("process"):
                        _journal(journal, asked[fresh[j]]["design"], answer, asked[fresh[j]].get("level"))
                    if "values" in answer or "failed" in answer:
                        evaluated += 1
                    if log is not None and time.monotonic() - last_note > 60:
                        last_note = time.monotonic()
                        log.info("[optimise] still searching: %d evaluation(s) so far (%d at the same time)", evaluated,
                                 at_once)

                got = await evaluate_many(executor, python, quest_root, module, entry, [asked[k]["cell"] for k in fresh],
                                          at_once=at_once, runs=runs, base_seed=int(base_seed), deadline=deadline,
                                          env=env, thresholds=thresholds, on_done=finished, log=log, gate=gate,
                                          in_each_run=_search.in_each_run(block) or None)
                for k, (answer, err, loaded) in zip(fresh, got):
                    if err:
                        stderr_parts.append(err)
                    # one after another, the designs after the one whose simulation could not be loaded are not run
                    answers[k] = {"stop": "the simulation could not be loaded"} if load_error else answer
                    load_error = load_error or loaded
                reply = answers  # type: ignore[assignment]
            elif request["kind"] == "evaluate_stream":
                # Designs the starts chose, answered as each one finishes: a place in the pool is refilled the moment
                # another design ends, and the search goes on with the ones that finished.
                if pool is None:
                    pool = EvaluationPool(executor, python, quest_root, module, entry, at_once=at_once, runs=runs,
                                          base_seed=int(base_seed), deadline=deadline, env=env, thresholds=thresholds,
                                          on_done=finished_streamed, log=log, gate=gate,
                                          in_each_run=_search.in_each_run(block) or None)
                ready: list[Any] = []
                for one in request["requests"]:
                    replayed = (replay.get(_replay_key(replay_space, one["design"], one.get("level")))
                                if replay_space is not None and isinstance(one.get("design"), dict) else None)
                    if scan_unchanged:
                        ready.append((one["id"], {"stop": "scan_unchanged"}))
                    elif replayed is not None:
                        ready.append((one["id"], dict(replayed)))
                    elif load_error:
                        ready.append((one["id"], {"stop": "the simulation could not be loaded"}))
                    else:
                        streamed[one["id"]] = one
                        pool.submit(one["id"], one["cell"])
                while not ready and pool.running:
                    for ident, (answer, err, loaded) in await pool.wait():
                        if err:
                            stderr_parts.append(err)
                        # one after another, the designs after the one whose simulation could not be loaded are not run
                        ready.append((ident, {"stop": "the simulation could not be loaded"} if load_error else answer))
                        load_error = load_error or loaded
                        streamed.pop(ident, None)
                reply = ready  # type: ignore[assignment]
            elif request["kind"] == "evaluate":
                left = deadline - time.monotonic()
                # A design asked again to see what varies between calls (``repeat``) is never taken from the journal or
                # put in it: it would be the very answer it is compared with.
                again = bool(request.get("repeat"))
                replayed = (replay.get(_replay_key(replay_space, request["design"], request.get("level")))
                            if replay_space is not None and isinstance(request.get("design"), dict) and not again
                            else None)
                if scan_unchanged:
                    reply: dict[str, Any] = {"stop": "scan_unchanged"}
                elif replayed is not None:
                    reply = dict(replayed)
                elif load_error:
                    reply = {"stop": "the simulation could not be loaded"}
                elif left <= 0:
                    reply = {"stop": "time"}
                else:
                    reply, err, load_error = await _evaluate(
                        executor, python, quest_root, module, entry, request["cell"], runs=runs,
                        base_seed=int(base_seed), timeout_s=left, env=env, thresholds=thresholds,
                        in_each_run=_search.in_each_run(block) or None,
                    )
                    if err:
                        stderr_parts.append(err)
                    if (("values" in reply or "failed" in reply) and not load_error and not reply.get("process")
                            and isinstance(request.get("design"), dict) and not again):
                        _journal(journal, request["design"], reply, request.get("level"))
            else:
                left = deadline - time.monotonic()
                reply = (await _drive(executor, python, quest_root, request["spec"], timeout_s=left, env=env)
                         if left > 0 else {"stop": "time"})
                if reply.get("error") and time.monotonic() >= deadline:
                    reply = {"stop": "time"}
            if request["kind"] == "evaluate" and ("values" in reply or "failed" in reply):
                evaluated += 1
            if log is not None and time.monotonic() - last_note > 60:
                last_note = time.monotonic()
                limit = _plan.budget(block).get("search")
                minutes = (time.monotonic() - started) / 60
                pace = (f", about {_fmt_pace((evaluated - len(journaled)) / minutes)} a minute"
                        if minutes > 0 and evaluated > len(journaled) else "")
                log.info("[optimise] still searching: %d%s evaluation(s) so far%s", evaluated,
                         f" of at most {limit}" if isinstance(limit, int) and limit > 0 else "", pace)
            request = gen.send(reply)
    except StopIteration as stop:
        done_outcome = stop.value
    finally:
        if pool is not None:
            await pool.close(discard=True)
    outcome = done_outcome or {}
    elapsed = time.monotonic() - started + before_s
    planned = _plan.budget(block)
    extra = {
        "time": {"seconds": round(elapsed, 3),
                 "seconds_per_evaluation": (round(elapsed / outcome["evaluations"], 4)
                                            if outcome.get("evaluations") else None),
                 "estimated_seconds": planned.get("seconds"),
                 "estimated_seconds_per_evaluation": planned.get("seconds_per_evaluation")},
        "ledger": LEDGER_PATH.as_posix(),
        "runs_per_evaluation": runs,
        "entry": entry,
        **({"run_id": run_id} if run_id else {}),
    }
    if outcome.get("stopped_because") == "time":
        extra["time"]["limit_seconds"] = int(timeout_s)
    if journaled:
        extra["continued"] = {"evaluations_taken_from_the_stopped_search": len(journaled),
                              "seconds_before_the_stop": round(before_s, 3)}
    record = _search.best_design(outcome, block, seed=int(base_seed),
                                 check={str(n): _plan.check_levels(s if isinstance(s, dict) else {"search": s})[0]
                                        for n, s in (block.get("numerical_settings") or {}).items()},
                                 extra=extra)
    if warnings:
        record["simulation_may_search_itself"] = warnings
    record.update({k: v for k, v in scan_carried.items() if k not in record})
    if scan_note:
        record["scan_settings_without_effect"] = scan_note[-1]
    if scan_unchanged:
        record["scan_unchanged"] = {"settings": [a for a, _ in scan_unchanged],
                                    "pairs": min(p for _, p in scan_unchanged)}
    if scan_unsteady:
        record["unsteady"] = sorted(scan_unsteady)
    ledger_text = "\n".join(_search.ledger_lines(outcome, block)) + "\n"
    record["ledger_sha256"] = _sha(ledger_text.encode("utf-8"))
    best_text = json.dumps(record, indent=1) + "\n"
    # Written now, after the last evaluation, from FI's own memory: whatever a simulation wrote to these paths is gone.
    # As bytes, so the file is exactly the text whose hash is kept (no newline translation on Windows).
    (quest_root / LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
    (quest_root / LEDGER_PATH).write_bytes(ledger_text.encode("utf-8"))
    (quest_root / BEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    (quest_root / BEST_PATH).write_bytes(best_text.encode("utf-8"))
    stderr = "\n".join(p for p in stderr_parts if p)[-_STDERR_KEEP:]
    record_text = _save(quest_root, key, ledger_text, best_text, stderr, bool(scan_rows), study_key)
    journal.unlink(missing_ok=True)
    record = json.loads(best_text)
    return SearchRun(record=record, rows=outcome.get("rows") or [], stderr=stderr, scan=bool(scan_rows),
                     load_error=load_error, files=_files(ledger_text, best_text, record_text))


# --- a search still improving when its budget runs out -----------------------------------------------------------------


def still_improving(rows: list[dict[str, Any]], direction: str) -> dict[str, float] | None:
    """Whether the best design was still improving when the search ended, from the search's own ledger rows: the last
    tenth of its counted evaluations (the baseline's is not one of them) gave more than 5% of the whole improvement over
    the baseline (or over the first design that met every limit, when the baseline did not). ``{"gain_last_tenth", "total", "share"}`` (in the objective's unit, the share a fraction) when it
    was, else ``None``."""
    sign = 1.0 if direction == "maximise" else -1.0
    ordered = sorted((r for r in rows if isinstance(r, dict) and r.get("event", "evaluation") == "evaluation"
                      and r.get("counted", True) and r.get("stage") != "scan"),
                     key=lambda r: r.get("n") if isinstance(r.get("n"), (int, float)) else 0)

    def value(row: dict[str, Any]) -> float | None:
        v = row.get("objective")
        ok = row.get("status") == "ok" and row.get("feasible") is not False
        return float(v) if ok and isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    # The improvement is counted from the first design that meets every limit: the baseline when it does, else the
    # first such design the search found (a baseline that breaks a limit has no value to improve on).
    base = next((value(r) for r in ordered if r.get("stage") == "baseline"), None)
    searched = [r for r in ordered if r.get("stage") != "baseline"]
    if not searched:
        return None
    cut = len(searched) - max(1, math.ceil(len(searched) / 10))  # evaluations before the last tenth
    best_at = best_final = base
    for i, row in enumerate(searched):
        if i == cut:
            best_at = best_final
        v = value(row)
        if v is not None and (base is None or sign * (v - best_final) > 0):
            if base is None:
                base = v
            best_final = v
    if base is None:
        return None
    if best_at is None:  # the first design that meets every limit came in the last tenth: all of it is recent
        best_at = base
    total = sign * (best_final - base)
    gain = sign * (best_final - best_at)
    if total > 0 and gain > 0.05 * total:
        return {"gain_last_tenth": gain, "total": total, "share": gain / total}
    return None


def _still_improving_somewhere(run: SearchRun, direction: str) -> dict[str, float] | None:
    """:func:`still_improving` for the search as it was run: a search compared across the levels of a condition
    (``compared``) is one search per level, each with its own baseline, budget and end, so each level whose own budget
    ran out is judged on its own rows; the first one still improving is the answer."""
    levels = (((run.record or {}).get("compared") or {}).get("levels")) or []
    if not levels:
        return still_improving(run.rows, direction)
    for part in levels:
        if not isinstance(part, dict) or part.get("stopped_because") != "budget":
            continue
        mine = [r for r in run.rows if isinstance(r, dict) and r.get("level") == part.get("level")]
        fired = still_improving(mine, direction)
        if fired is not None:
            return {**fired, "level": part.get("level")}
    return None


def _seconds_per_evaluation(run: SearchRun) -> float | None:
    """What one evaluation of this search cost, in wall seconds, from FI's record of it."""
    spent = (run.record or {}).get("time") or {}
    per = spent.get("seconds_per_evaluation")
    if isinstance(per, (int, float)) and per > 0:
        return float(per)
    used = ((run.record or {}).get("evaluations") or {}).get("search")
    seconds = spent.get("seconds")
    if isinstance(seconds, (int, float)) and isinstance(used, int) and used > 0:
        return float(seconds) / used
    return None


# --- run.log ---------------------------------------------------------------------------------------------------------

_STOPPED = {
    "budget": "the evaluation budget ran out",
    "share": "each starting point converged or used its share of the budget",
    "converged": "every starting point converged",
    "finished": "every combination was evaluated",
    "time": "the time limit (execution.timeout_s) was reached",
    "failures": "the first evaluations all failed",
    "not_searched": "its first evaluation returned no objective, as the alternative searched before it did",
    "library_error": "the library's method stopped with an error",
    "the simulation could not be loaded": "the simulation could not be loaded",
    "target": "a design reached the target asked for",
    "not_run": "it did not run: the search before it had stopped (its time limit, or every evaluation failed)",
    "no_effect": "the first designs it tried, which between them changed every number of the design, all gave the "
                 "baseline's results",
    "stepped_objective": "its objective was counted in whole steps of one of its numerical settings, so the search had "
                         "no direction to follow; the simulation was sent back to read it between the steps",
    "scan_unchanged": "its coarse scan found settings it compares that changed no result the simulation returned; the "
                      "simulation was sent back so that each of them reaches the results",
    "flat_objective": "the first designs it tried, which between them changed every number of the design, all gave the "
                      "baseline's value of the objective or one of a few values, so the search had no direction to "
                      "follow (more of the same budget would not have found one)",
}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _design(design: dict[str, Any]) -> str:
    return ", ".join(f"{k} = {_fmt(v)}" for k, v in design.items())


def summary_lines(record: dict[str, Any]) -> list[str]:
    """Plain lines for run.log about a finished search."""
    if not record:
        return []
    ev = record.get("evaluations") or {}
    method = record.get("method") or {}
    objective = record.get("objective") or {}
    unit = f" {objective['unit']}" if objective.get("unit") else ""
    quantity = objective.get("quantity", "the objective")
    stopped = _STOPPED.get(str(ev.get("stopped_because")), str(ev.get("stopped_because")))
    scan = f", after a coarse scan of {ev['scan']} design(s)" if ev.get("scan") else ""
    lines = [f"[optimise] {ev.get('search', 0)} evaluation(s) of the {ev.get('budget', 0)} allowed{scan}, method "
             f"{method.get('used')}; stopped because {stopped}; {ev.get('failed', 0)} failed, {ev.get('infeasible', 0)} "
             "broke a limit"]
    if method.get("why"):
        lines.append(f"[optimise] {method.get('requested')} was asked for, but {method['why']}")
    from . import best_design_report as _report

    if flat := _report.flat_went_on(ev):
        lines.append(f"[optimise] {flat}")
    if stepped := _report.stepped_went_on(ev):
        lines.append(f"[optimise] {stepped}")
    baseline = record.get("baseline") or {}
    base_text = (f"{_fmt(baseline.get('objective'))}{unit}" if baseline.get("objective") is not None
                 else f"no value ({baseline.get('reason') or 'its evaluation failed'})")
    if not baseline.get("feasible") and baseline.get("objective") is not None:
        base_text += ", and it breaks a limit"
    best = record.get("best")
    if best is None:
        lines.append(f"[optimise] no design that meets every limit was found; the baseline "
                     f"({_design(baseline.get('design') or {})}) gives {quantity} = {base_text}")
    else:
        lines.append(f"[optimise] best design: {_design(best['design'])}: {quantity} = {_fmt(best['objective'])}{unit}")
        imp = record.get("improvement")
        change = (f"; {abs(imp['value']):.6g}{unit} {'better' if imp['better'] else 'not better'}" if imp else "")
        if imp and imp.get("better") and imp.get("beyond_threshold") is False:
            change += ", less than the plan's threshold for better"
        lines.append(f"[optimise] the baseline ({_design(baseline.get('design') or {})}) gives {base_text}{change}")
    compared = (record.get("compared") or {}).get("levels") if isinstance(record.get("compared"), dict) else None
    from . import optimise_search as _osearch

    skipped = [p for p in compared or [] if isinstance(p, dict) and p.get("status") == _osearch.NOT_SEARCHED]
    if compared and best is not None:
        same = [p for p in compared if p.get("same_as")]
        lines.append(f"[optimise] {len(compared)} alternatives were "
                     + ("compared" if same or skipped else "each searched on their own") + "; the best design is the "
                     f"one of {_osearch.level_says(best.get('level') or {})}. Best of each: "
                     + "; ".join(f"{_osearch.level_says(p['level'])}: "
                                 + (f"{_fmt(p['best']['objective'])}{unit}" if p.get("best") else "no design that meets every limit")
                                 for p in compared))
        for p in same:
            lines.append(f"[optimise] {_osearch.same_as_says(p)}")
    for p in skipped:  # named whether or not a best design was found
        lines.append(f"[optimise] {_osearch.not_searched_says(p)}")
    for r in record.get("continued") or []:
        target = f", toward {quantity} = {_fmt(r['target'])}{unit}" if r.get("target") is not None else ""
        reached = ("; the target was reached at the search's settings" if r.get("target_reached") else
                   "; the target was not reached at the search's settings" if r.get("target") is not None else "")
        who = ("by FI, still improving when its budget ran out" if r.get("by") == "FI" else "at the person's request")
        lines.append(f"[optimise] continued {who} (round {r.get('round')}): {r.get('evaluations')} of "
                     f"{r.get('added')} more evaluation(s){target}, from {_design(r.get('from') or {})}; stopped "
                     f"because {_STOPPED.get(str(r.get('stopped_because')), r.get('stopped_because'))}{reached}")
    settings = record.get("search_settings") or {}
    at = f" ({_design(settings)})" if settings else ""
    if not CHECKS_AT_FINER_SETTINGS:
        lines.append(f"[optimise] the best design is not checked at finer numerical settings in this version of FI: it "
                     f"was found and scored at the search's own settings{at} only, so part of an improvement may be a "
                     "numerical error")
    elif at:
        lines.append(f"[optimise] these values are at the search's own settings{at}; FI checks them at finer settings "
                     "next")
    return lines


def attach_check(quest_root: Path, run: SearchRun, check: dict[str, Any], text: str | None = None) -> None:
    """Put what FI's check at finer settings found (:func:`core.optimum_check.attach_summary`) into
    ``results/best_design.json`` (``check``, ``checked_at_finer_settings``, and its sentence), and keep FI's record of the
    file in step, with the check's hash, so :func:`restore` puts back this copy and a check kept on disk is reused only
    when it is this one. Idempotent: an earlier check's part is replaced."""
    from . import optimum_check as _check

    quest_root = Path(quest_root)
    best_rel, record_rel = BEST_PATH.as_posix(), RECORD.as_posix()
    try:
        best = json.loads(run.files[best_rel])
        saved = json.loads(run.files[record_rel])
    except (KeyError, ValueError, TypeError):
        return
    summary = _check.attach_summary(check, text)
    best["check"] = summary
    target = best.get("target")
    if isinstance(target, dict):
        # Whether the target holds for the design FI checked, at the finest settings (the number a person relies on).
        direction = str((best.get("objective") or {}).get("direction") or "minimise")
        sign = 1.0 if direction == "minimise" else -1.0
        value = summary.get("objective")
        target["reached_at_finest_settings"] = (
            bool(sign * (float(value) - float(target["value"])) <= 0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and check.get("finished") is not False
            else None)
    # Only a check that finished and had finer settings to use checked anything at finer settings.
    best["checked_at_finer_settings"] = bool(
        check.get("finished") and (check.get("settings") or {}).get("named")
        and (check.get("evaluations") or {}).get("check", 0) > 0)
    if text is not None:
        saved["check_sha256"] = _sha(text.encode("utf-8"))
    says = str(best.get("says") or "")
    for old in (" Found and scored at the search's own numerical settings only; it has not been recomputed at finer "
                "settings.", " Checked at finer numerical settings:"):
        if old in says:
            says = says[:says.index(old)]
    best["says"] = f"{says} Checked at finer numerical settings: {summary['says']}".strip()
    best_text = json.dumps(_search.plain_zero(best), indent=1) + "\n"
    saved["best"] = {"sha256": _sha(best_text.encode("utf-8")), "text": best_text}
    record_text = json.dumps(saved)
    for rel, body in ((BEST_PATH, best_text), (RECORD, record_text)):
        (quest_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (quest_root / rel).write_bytes(body.encode("utf-8"))
    run.files = _files(run.files.get(LEDGER_PATH.as_posix(), ""), best_text, record_text)
    run.record = json.loads(best_text)


# --- the runner the engine uses --------------------------------------------------------------------------------------


#: The one setting the search's own evaluations are judged as by FI's check of the failed trials.
EVALUATIONS_KEY = "the search's evaluations"
NO_CALIBRATION = ("a search for the best design: its intervals and standard errors are not checked against true values "
                  "(only its failed trials are)")


def trial_cells(run: SearchRun, quest_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """What FI's check of the failed trials (core/trial_checks.py) reads for a search: each setting of the coarse scan
    with its own trials (``raw/trials.json``), then the search's evaluations together as one setting, a failed evaluation
    counted as failed. ``(cells, reasons by key)``."""
    from collections import Counter

    from . import trial_checks as _checks

    cells: list[dict[str, Any]] = []
    reasons: dict[str, Any] = {}
    if run.scan:
        try:
            data = json.loads((Path(quest_root) / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        cells = [c for c in data.get("cells") or [] if isinstance(c, dict)]
        reasons = dict(_checks.ledger_reasons(quest_root))
    rows = [r for r in run.rows if r.get("event", "evaluation") == "evaluation" and r.get("stage") != "scan"]
    if rows:
        failed = [r for r in rows if r.get("status") != "ok"]
        cells.append({"key": EVALUATIONS_KEY, "planned": len(rows), "ok": len(rows) - len(failed), "failed": len(failed),
                      "metrics": {}})
        reasons[EVALUATIONS_KEY] = Counter(str(r.get("reason") or "the evaluation failed") for r in failed)
    return cells, reasons


def watched_results(protocol: dict[str, Any]) -> list[str]:
    """The plan's results, watched for a design that changes none of them (core/optimise_search.py, ``no_effect``) and
    for a compared alternative that gives the same results as the first. Not a result the plan itself expects to stay
    put (an invariant, a fixed-input check): its staying put says nothing about the design."""
    from . import degenerate_results as _degenerate

    constant = _degenerate.expected_constants(protocol)
    return [str(s["id"]) for s in _metric_spec.declared(protocol) if s.get("id") and str(s["id"]) not in constant]


class OptimisationRunner:
    """Stands in for the executor when the plan is a search for the best design, as ``trial_runner.TrialsRunner`` does
    for a measurement: running ``experiment.py`` first runs the search (:func:`run_search`), then the analysis, with
    ``FI_OPTIMISATION`` naming FI's ledger and ``FI_BEST_DESIGN`` its best-design file (and ``FI_TRIALS`` the coarse
    scan's per-setting results, when there was one). ``failed_script`` says which script to repair; ``last`` keeps the
    search for the checks after it."""

    def __init__(self, executor: Any, *, quest_root: Path, protocol: Any, simulate: Path, analysis: Path,
                 log: Any = None, frozen: Any = False, repairs_left: Any = None, at_once: int = 1,
                 stop_flat: Any = True) -> None:
        self.executor = executor
        #: How many evaluations that do not depend on each other run at the same time (the coarse scan's settings, the
        #: check's designs at each finer level): ``execution.settings_at_once``.
        self.at_once = max(1, int(at_once or 1))
        #: Seconds of the time limit kept for FI's check of the best design when the study was too long to finish (set by
        #: the run's sizing, core/run_checks.py): the coarse scan and the search stop that much sooner.
        self.check_reserve_s = 0.0
        #: The id of a cut study (set by the run's sizing): written into the scan's ledger and the search's record, so the
        #: disclosure reads only this run's files.
        self.run_id = ""
        #: Whether a repair of the simulation is left (a bool, or a callable asked at each run): FI's check of the failed
        #: trials (core/trial_checks.py) sends a fault back only while one is.
        self._repairs_left = repairs_left
        #: The fault FI's check of the failed trials found on the last run (the engine's stop once the repairs are spent).
        self.trial_fault: dict[str, Any] | None = None
        #: Whether the study is frozen for its confirm run (a bool, or a callable asked at each run): the search is
        #: then not run again (:func:`run_search`).
        self._frozen = frozen
        #: Whether a search whose objective does not move stops and hands that fact back (a bool, or a callable asked at
        #: each run): off once the plan has been revised for it.
        self._stop_flat = stop_flat
        self.quest_root = Path(quest_root)
        self._protocol = protocol
        self.simulate = Path(simulate)
        self.analysis = Path(analysis)
        self.log = log
        self.failed_script: str | None = None
        #: Why the last run was sent back as a simulation that does not use its inputs (core/flat_output.py), or ``None``.
        self.flat: str | None = None
        #: What FI measured of a search that stopped because its objective did not move (the search's fact, with the
        #: results that are timings named ``unsteady``), or ``None``.
        self.flat_objective: dict[str, Any] | None = None
        #: The observation of settings that changed no result (core/flat_output.py), for the analysis and the writer.
        self.no_difference: str | None = None
        self.last: SearchRun | None = None
        #: FI's check at finer settings, from memory: ``(its text, the key it is kept under)``.
        self.last_check: tuple[str | None, str | None] = (None, None)
        #: The rounds a person asked for (``.fi/optimisation/continue.json``) as FI wrote them, before any script ran.
        self._rounds: bytes | None = None
        self._rounds_read = False

    def _put_back_rounds(self) -> bool:
        """Put back the rounds file FI wrote, wherever a script changed or removed it (its budget is the evidence
        ladder's allowance)."""
        if not self._rounds_read:
            return False
        path = self.quest_root / ROUNDS_PATH
        try:
            now = path.read_bytes() if path.is_file() else None
        except OSError:
            now = None
        if now == self._rounds:
            return False
        if self._rounds is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self._rounds)
        return True

    def _is_frozen(self) -> bool:
        return bool(self._frozen() if callable(self._frozen) else self._frozen)

    def put_back(self) -> bool:
        """Put FI's own record of the search and of its check back, from memory, wherever a script changed them."""
        from . import optimum_check as _check

        put_back = self._put_back_rounds()
        put_back = (restore(self.quest_root, self.last.files) if self.last is not None and self.last.files
                    else False) or put_back
        text, key = self.last_check
        if text is not None:
            put_back = _check.restore(self.quest_root, text, key) or put_back
        return put_back

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        from core.execution import ExecutionResult

        if len(cmd) != 2 or Path(cmd[1]).name != self.analysis.name:
            # Any other script (a replot, a helper) runs as it is; what it changed of FI's record is put back after it.
            result = await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)
            if self.put_back() and self.log is not None:
                self.log.warning("[optimise] %s changed FI's record of the search or of its check; FI's own copy was "
                                 "put back", Path(cmd[-1]).name if cmd else "a script")
            return result
        started = time.monotonic()
        self.flat = None
        self.flat_objective = None
        self.trial_fault = None
        protocol = (self._protocol() if callable(self._protocol) else self._protocol) or {}
        if block_of(protocol) is None:
            self.failed_script = None
            return ExecutionResult(returncode=1, stdout="", duration_s=0.0,
                                   stderr="the plan is a search for the best design but its protocol has no "
                                          "optimisation block: nothing can be searched")
        if not self.simulate.is_file():
            self.failed_script = None
            return ExecutionResult(returncode=1, stdout="", duration_s=0.0,
                                   stderr=f"code/{self.simulate.name} is missing: a search for the best design needs the "
                                          "simulation in its own script, defining run_cell(cell) (or run_trial(cell, "
                                          "trial_id, seed)), which FI calls once for each design it tries; the analysis "
                                          f"is {self.analysis.name}. Nothing was searched.")
        base = int((env or {}).get("FI_REPLICATE_SEED") or 0)
        try:
            self._rounds = (self.quest_root / ROUNDS_PATH).read_bytes() if (self.quest_root / ROUNDS_PATH).is_file() \
                else None
        except OSError:
            self._rounds = None
        self._rounds_read = True
        async def search(limit_s: int) -> SearchRun:
            return await run_search(self.executor, cmd[0], self.quest_root,
                                    self.simulate.relative_to(self.quest_root).as_posix(), protocol, base_seed=base,
                                    timeout_s=limit_s, env=env, log=self.log,
                                    frozen=self._is_frozen(), at_once=self.at_once,
                                    stop_flat=bool(self._stop_flat() if callable(self._stop_flat) else self._stop_flat),
                                    stop_stepped=self._stepped_repair_left(),
                                    stop_unchanged=self._unchanged_repair_left(), run_id=self.run_id)

        search_s = int(timeout_s - self.check_reserve_s)
        if search_s < 1:
            # Nothing is left for the scan and the search once the check's time is kept: they do not start, and the check
            # does not lose its time to a search given a second it does not have.
            self.failed_script = None
            return ExecutionResult(
                returncode=1, stdout="", duration_s=time.monotonic() - started, timed_out=True,
                stderr=f"The run's time limit (execution.timeout_s = {timeout_s} s) leaves nothing for the search once the time "
                       "for FI's check of the best design is kept: the search was not started.")
        run = await search(search_s)
        run = await self._continue_while_improving(run, search, protocol, started, search_s)
        self.last = run
        # The coarse scan's settings that changed no result (core/flat_output.py), kept in the search's record so a
        # search used again says it too.
        self.no_difference = str((run.record or {}).get("scan_settings_without_effect") or "") or None
        if self.log is not None and not run.reused:
            for line in summary_lines(run.record):
                self.log.info("%s", line)
        if run.problems and not run.load_error:
            # The plan's block cannot drive a search: nothing in the scripts can fix that.
            self.failed_script = None
            return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                                   stderr="FI's search for the best design could not start: " + "; ".join(run.problems))
        if run.ok_evaluations == 0:
            self.failed_script = self.simulate.name
            stopped = ((run.record or {}).get("evaluations") or {}).get("stopped_because")
            if stopped == "time":
                return ExecutionResult(
                    returncode=1, stdout="", duration_s=time.monotonic() - started, timed_out=True,
                    stderr=f"{run.stderr}\nThe run's time limit (execution.timeout_s = {timeout_s} s) ran out before any "
                           "design was evaluated. Each evaluation of the simulation must fit, many times over, in "
                           "that time: make one evaluation faster, or raise execution.timeout_s.".strip())
            reason = run.load_error or next((r.get("reason") for r in run.rows if r.get("reason")),
                                            "every evaluation failed")
            return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                                   stderr=f"{run.stderr}\nFI's search for the best design got no result from the "
                                          f"simulation: {reason}".strip())
        # More than half of the search's evaluations (or of a scan setting's trials) failed: the best design would rest on
        # the few that survived. Sent back to be repaired while a repair is left, then a stop (core/trial_checks.py).
        if (sent := self._check_trials(run, started)) is not None:
            return sent
        # The first designs of the search, which between them moved every number of the design, all gave the baseline's
        # results (core/optimise_search.py stopped it then, not after its whole budget): FI measured that the design
        # changes nothing at the settings the search holds, whatever the code seems to read.
        no_effect = ((run.record or {}).get("evaluations") or {}).get("no_effect")
        if isinstance(no_effect, dict):
            return self._flat_result(_flat.no_effect_search(no_effect, _plan.normalize(block_of(protocol))[0]
                                                            or block_of(protocol) or {}), run, started)
        # The objective did not move at any of the designs sampled across the whole range (core/optimise_search.py stopped
        # it then): nothing to check and nothing to analyse. The plan is told what FI measured.
        evaluations = (run.record or {}).get("evaluations") or {}
        # The objective is counted in whole steps of one of the search's numerical settings: the simulation is sent back
        # once to read it between the points of that setting's scan.
        unchanged = (run.record or {}).get("scan_unchanged")
        if evaluations.get("stopped_because") == "scan_unchanged" and isinstance(unchanged, dict):
            return self._unchanged_result(unchanged, run, started)
        stepped = evaluations.get("stepped_objective")
        if evaluations.get("stopped_because") == "stepped_objective" and isinstance(stepped, dict):
            return self._stepped_result(stepped, run, started, _plan.normalize(block_of(protocol))[0] or {})
        fact = evaluations.get("flat_objective")
        if evaluations.get("stopped_because") == "flat_objective" and isinstance(fact, dict):
            return self._flat_objective_result(fact, run, started)
        # A search in which every design gave the very same numbers, and whose simulation's code never lets the design reach
        # what it returns, is not a result: the simulation does not use the design it is given. It is sent back to be repaired
        # like one that does not run (core/flat_output.py). The same numbers from a simulation that does read the design are
        # an observed plateau, not a fault.
        norm_block = _plan.normalize(block_of(protocol))[0] or block_of(protocol) or {}
        from . import degenerate_results as _degenerate

        try:
            sim_text = self.simulate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            sim_text = None
        flat = _flat.flat_search(run.rows, str((norm_block.get("objective") or {}).get("quantity") or "the objective"),
                                 source=sim_text, expected=_degenerate.expected_constants(protocol))
        if flat is not None:
            return self._flat_result(flat, run, started)
        # FI checks the best design at finer settings (core/optimum_check.py), within the same time limit as the search. A check that
        # fails is the study's result, reported and carried on with; one that cannot run is recorded as not finished.
        # The check shares the one time limit with the search (``execution.timeout_s`` bounds the whole call): it gets what the
        # search left, not a whole limit of its own, which let a search that used the limit be followed by a check of up to
        # as long again (72 runs past an 8 h limit). A check that finds no time left is recorded as not finished.
        check_left = max(1, int(timeout_s) - int(time.monotonic() - started))
        check_text, check_key = await self._check(cmd[0], run, protocol, base, check_left, env)
        self.last_check = (check_text, check_key)
        try:
            checked = json.loads(check_text) if check_text else None
        except ValueError:
            checked = None
        flat = _flat.flat_search_check(checked, norm_block, source=sim_text)
        if flat is not None:
            return self._flat_result(flat, run, started)
        analysis_env = {**(env or {}), LEDGER_ENV: LEDGER_PATH.as_posix(), BEST_ENV: BEST_PATH.as_posix(),
                        "FI_RAW_DIR": _trials.RAW_DIRNAME}
        if check_text is not None:
            from . import optimum_check as _check

            analysis_env[_check.CHECK_ENV] = _check.CHECK_PATH.as_posix()
        if run.scan:
            analysis_env[_trials.RESULTS_ENV] = (Path(_trials.RAW_DIRNAME) / _trials.SUMMARY_NAME).as_posix()
        result = await self.executor.execute(cmd, cwd=cwd, timeout_s=timeout_s, env=analysis_env)
        # FI's own copy, from memory: a script that also rewrote FI's record of the files cannot make its version stand.
        put_back = self._put_back_rounds()
        put_back = restore(self.quest_root, run.files or None) or put_back
        if check_text is not None:
            from . import optimum_check as _check

            put_back = _check.restore(self.quest_root, check_text, check_key) or put_back
        # (self.last was set by the search; self.last_check just above: put_back() restores both the same way.)
        if put_back and self.log is not None:
            self.log.warning("[optimise] %s changed FI's record of the search (%s, %s, %s, %s or %s); FI's own copy "
                             "was put back", self.analysis.name, LEDGER_PATH.as_posix(), BEST_PATH.as_posix(),
                             RECORD.as_posix(), "needs/OPTIMUM_CHECK.json", ".fi/optimisation/check.json")
        stdout = result.stdout
        if result.returncode == 0:
            stdout, differs = with_fi_record(stdout, run.record)
            if differs and self.log is not None:
                self.log.warning("[optimise] experiment.py reports %s, which FI's record of the search does not hold; "
                                 "FI's own numbers are in its result under `fi_search`", "; ".join(differs[:5]))
        self.failed_script = None if result.returncode == 0 else self.analysis.name
        return ExecutionResult(
            returncode=result.returncode, stdout=stdout, duration_s=time.monotonic() - started,
            stderr=(run.stderr + "\n" + (result.stderr or "")).strip(), timed_out=result.timed_out,
        )

    async def _continue_while_improving(self, run: SearchRun, search: Any, protocol: dict[str, Any], started: float,
                                        timeout_s: int) -> SearchRun:
        """A search that stopped only because its evaluation budget ran out, while its best design was still improving
        (:func:`still_improving`), goes on by itself with one more round (one start's share of the budget) as long as
        the round fits in what is left of this call's time limit. The earlier evaluations are answered from FI's record
        of them, so only the new round runs the simulation. Any other end of a search is left as it is."""
        from . import optimise_refine as _refine

        norm = _plan.normalize(block_of(protocol))[0]
        per_start = int((((norm or {}).get("evaluation_budget") or {}).get("per_start")) or 0)
        if norm is None or per_start <= 0 or run.reused or self._is_frozen():
            return run
        direction = str((norm.get("objective") or {}).get("direction") or "minimise")
        per = _seconds_per_evaluation(run)
        while (((run.record or {}).get("evaluations") or {}).get("stopped_because") == "budget"
               and not run.load_error and not run.problems and not run.reused):
            fired = _still_improving_somewhere(run, direction)
            # A round continues every level of a compared search that was searched: it costs per_start evaluations
            # for each (a level that gave the first level's results is not searched, core/optimise_search.py).
            parts = max(1, sum(1 for part in ((run.record or {}).get("compared") or {}).get("levels") or []
                               if isinstance(part, dict) and not part.get("same_as")
                               and part.get("status") != "not_searched"))
            # Whole seconds, as the search is given them: a round that fits only in a fraction of a second more is not
            # started, and the search is never given more than is left.
            left = int(timeout_s - (time.monotonic() - started))
            if fired is None or per is None or left < 1 or per_start * parts * per > left:
                break
            rounds = read_rounds(self.quest_root, norm)
            number = max([int(r["auto"]) for r in rounds if isinstance(r.get("auto"), int)] or [0]) + 1
            asked = (f"FI: the search was still improving when its budget ran out (the last tenth of its evaluations "
                     f"gave {round(100 * fired['share'])}% of its improvement)")
            _refine.add_round(self.quest_root, norm, {"added": per_start, "target": None, "asked": asked,
                                                      "auto": number})
            if self.log is not None:
                self.log.info("[optimise] the search was still improving when its budget ran out (the last tenth of "
                              "its evaluations gave %d%% of its improvement): FI continues it with %d more "
                              "evaluation(s)%s, %d s left of the time limit", round(100 * fired["share"]), per_start,
                              f" for each of its {parts} compared levels" if parts > 1 else "", int(left))
            left = int(timeout_s - (time.monotonic() - started))  # again: writing the round took time too
            if left < 1:
                break
            round_started = time.monotonic()
            run = await search(left)
            path = self.quest_root / ROUNDS_PATH
            self._rounds = path.read_bytes() if path.is_file() else None
            per = max(per, (time.monotonic() - round_started) / (per_start * parts))
        return run

    def _check_trials(self, run: SearchRun, started: float) -> Any:
        """FI's check of the failed trials on a search (:func:`trial_cells`): a fault is a failed run of the simulation
        while a repair is left, then the engine's stop; a failed design is never chosen, so nothing is left out. The
        intervals are not checked here (the record says so). ``None`` to go on."""
        from core.execution import ExecutionResult

        from . import trial_checks as _checks

        cells, reasons = trial_cells(run, self.quest_root)
        if not cells:
            return None
        left = self._repairs_left() if callable(self._repairs_left) else bool(self._repairs_left)
        verdict = _checks.judge(cells, reasons=reasons, repairs_left=bool(left), no_calibration=NO_CALIBRATION)
        if verdict.action == _checks.LEAVE_OUT:
            # the search never chooses a design whose evaluation failed: nothing to leave out, the failures are stated
            verdict.action, verdict.left_out = _checks.OK, []
        _checks.write_record(self.quest_root, verdict.record())
        if self.log is not None:
            for f in verdict.fails:
                if f.stated and not f.fault:
                    self.log.info("[optimise] %s (recorded; the paper states it)", f.says())
            self.log.info("[optimise] calibration not checked: %s", NO_CALIBRATION)
        if verdict.action == _checks.OK:
            return None
        self.failed_script = self.simulate.name
        self.trial_fault = {"action": verdict.action, "say": verdict.say, "tried": list(verdict.tried),
                            "partial": verdict.partial}
        if self.log is not None:
            self.log.warning("[optimise] FI's check of the failed trials found a fault; %s:\n%s",
                             "sending the simulation back to be repaired" if verdict.action == _checks.REPAIR
                             else "the repairs are spent, the quest stops", verdict.directive[:1500])
        directive = verdict.directive if len(verdict.directive) <= 1800 else verdict.directive[:1797] + "..."
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                               stderr=f"{run.stderr[-4000:]}\n{directive}".strip())

    def _flat_objective_result(self, fact: dict[str, Any], run: SearchRun, started: float) -> Any:
        """The run, handed back with what FI measured: the objective gave one value at every design across the range.
        Numbers that differ when the same design runs twice (timings) are in neither list."""
        from core.execution import ExecutionResult

        unsteady = sorted(str(n) for n in (run.record or {}).get("unsteady") or [])
        self.flat_objective = {**fact, "unsteady": unsteady,
                               "stayed": [n for n in fact.get("stayed") or [] if n not in unsteady],
                               "moved": [n for n in fact.get("moved") or [] if n not in unsteady]}
        self.failed_script = None
        say = ((f"The search's objective moved only between {len(fact['values'])} values over {fact.get('designs')} "
                "designs across the whole range, so the search had no direction to follow. FI sends the design back to "
                "the plan.") if fact.get("values") else
               (f"The search's objective did not move: {fact.get('designs')} designs across the whole range all gave the "
                "starting design's value, so there was nothing to improve. FI sends the design back to the plan."))
        if self.log is not None:
            self.log.warning("[optimise] %s", say)
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started, stderr=say)

    def _stepped_repair_left(self) -> bool:
        """Whether a search whose objective is counted in whole steps of a setting stops and sends the simulation back:
        once per quest, while a repair is left."""
        left = self._repairs_left() if callable(self._repairs_left) else self._repairs_left
        return (left is None or bool(left)) and not (self.quest_root / STEPPED_SENT).is_file()

    def _unchanged_repair_left(self) -> bool:
        """Whether a coarse scan with a compared setting that changed no result stops the search and sends the
        simulation back: once per quest, while a repair is left."""
        left = self._repairs_left() if callable(self._repairs_left) else self._repairs_left
        return (left is None or bool(left)) and not (self.quest_root / UNCHANGED_SENT).is_file()

    def _unchanged_result(self, fact: dict[str, Any], run: SearchRun, started: float) -> Any:
        """The run, sent back to be repaired once: settings the coarse scan compares changed no result the simulation
        returned (core/flat_output.py ``unchanged_axes``, with the numbers that differ on a second call left out)."""
        from core.execution import ExecutionResult

        try:
            (self.quest_root / UNCHANGED_SENT).parent.mkdir(parents=True, exist_ok=True)
            (self.quest_root / UNCHANGED_SENT).write_text(json.dumps(fact, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        self.failed_script = self.simulate.name
        names = ", ".join(str(n) for n in fact.get("settings") or [])
        say = (f"FI's coarse scan found that changing {names} changed no number the simulation returned: in each of at "
               f"least {fact.get('pairs')} pairs of settings that differ in one of them alone, every result was the "
               "same (numbers that differ when the same setting is run again, such as a timing, were left out). The "
               "simulation does not let these settings reach its results, so the study would compare one result "
               f"repeated. Repair simulate.py so that each of {names} changes what the simulation computes as the plan "
               "says it should (a setting that chooses between alternatives must decide which one is computed, even "
               "when the design the search changes is also given). If simulate.py already lets a setting reach what it "
               "computes and the plan's own equations give it no effect there, leave that setting's code as it is; "
               "change nothing else.")
        if self.log is not None:
            self.log.warning("[optimise] %s", say)
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                               stderr=f"{run.stderr[-4000:]}\n{say}".strip())

    def _stepped_result(self, fact: dict[str, Any], run: SearchRun, started: float, block: dict[str, Any]) -> Any:
        """The run, sent back to be repaired once: the search's objective is a whole multiple of the step of one of its
        numerical settings in most of the designs it sampled (core/optimise_search.py, ``stepped_objective``)."""
        from core.execution import ExecutionResult

        try:
            (self.quest_root / STEPPED_SENT).parent.mkdir(parents=True, exist_ok=True)
            (self.quest_root / STEPPED_SENT).write_text(json.dumps(fact, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
        self.failed_script = self.simulate.name
        quantity = str((block.get("objective") or {}).get("quantity") or "the objective")
        unit = f" {fact.get('unit')}" if fact.get("unit") else ""
        say = (f"FI's search for the best design found that {quantity} is a whole multiple of {fact.get('setting')} "
               f"({_fmt(fact.get('step'))}{unit}, the step the search runs at) in {fact.get('on_step')} of the "
               f"{fact.get('nonzero')} designs that gave it a value other than zero, among the first "
               f"{fact.get('designs')} designs sampled across the whole range of the design. It is counted in whole steps "
               f"of {fact.get('setting')}: most small changes of the design leave it where it was, so the search has no "
               f"direction to follow. Repair simulate.py so that {quantity} is read between the points of that scan "
               "(interpolate where the quantity crosses its limit) and changes smoothly with the design; change nothing "
               "else.")
        if self.log is not None:
            self.log.warning("[optimise] %s", say)
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                               stderr=f"{run.stderr[-4000:]}\n{say}".strip())

    def _flat_result(self, why: str, run: SearchRun, started: float) -> Any:
        """The run, sent back: the simulation gives the same numbers whatever design it is given."""
        from core.execution import ExecutionResult

        self.flat = why
        self.failed_script = self.simulate.name
        if self.log is not None:
            self.log.warning("[optimise] the simulation does not use the design it is given: %s", why)
        return ExecutionResult(returncode=1, stdout="", duration_s=time.monotonic() - started,
                               stderr=f"{run.stderr}\nFI's search for the best design got a result that is not usable: "
                                      f"{why}".strip())

    async def _check(self, python: Any, run: SearchRun, protocol: dict[str, Any], base: int, timeout_s: int,
                     env: dict[str, str] | None) -> tuple[str | None, str | None]:
        """Run FI's check of the best design and put its verdict into ``results/best_design.json``: ``(the check's text
        (needs/OPTIMUM_CHECK.json), the key it is kept under)``, or ``(None, None)`` when it could not be written at
        all. Nothing here stops the quest: a check that cannot run, or cannot be recorded, is said in run.log."""
        from . import optimum_check as _check

        key: str | None = None
        try:
            check, text, key = await _check.run_check(
                self.executor, python, self.quest_root, self.simulate.relative_to(self.quest_root).as_posix(),
                protocol, run, base_seed=base, timeout_s=timeout_s, env=env, log=self.log, at_once=self.at_once,
                # The confirm run of a frozen search: the kept best design is checked once more, on the new seeds.
                fresh=self._is_frozen())
        except Exception as exc:  # noqa: BLE001 -- the check reports; it never stops the quest
            if self.log is not None:
                self.log.warning("[optimise] the check at finer numerical settings could not run (%s: %s); the best "
                                 "design is reported as not checked", type(exc).__name__, str(exc)[:300])
            record = _check._nothing_to_check(None, run.record)
            record["finished"] = False
            record["says"] = (f"The best design could not be checked at finer numerical settings: the check could not "
                              f"run ({type(exc).__name__}).")
            try:
                key = ""
                text = _check._write(self.quest_root, record, key)
                check = json.loads(text)
            except Exception:  # noqa: BLE001
                return None, None
        try:
            attach_check(self.quest_root, run, check, text)
            # The same, readable: results/best_design.md (core/best_design_report.py).
            from . import best_design_report as _report

            _report.write_readable(self.quest_root, _plan.normalize(block_of(protocol))[0] or block_of(protocol))
            if self.log is not None:
                for line in _check.summary_lines(check):
                    self.log.info("%s", line)
        except Exception as exc:  # noqa: BLE001 -- e.g. a file another program holds open: said, never a stop
            if self.log is not None:
                self.log.warning("[optimise] the check's verdict could not be written into %s (%s: %s); the check is in "
                                 "needs/OPTIMUM_CHECK.json", BEST_PATH.as_posix(), type(exc).__name__, str(exc)[:300])
        return text, key


def _record_numbers(record: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for part in (record.get("best") or {}, record.get("baseline") or {}):
        if isinstance(part.get("objective"), (int, float)):
            out.append(float(part["objective"]))
        for value in list((part.get("design") or {}).values()) + list((part.get("constraints") or {}).values()):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(float(value))
    imp = record.get("improvement") or {}
    for key in ("value", "relative"):
        if isinstance(imp.get(key), (int, float)):
            out.append(float(imp[key]))
            if key == "relative":
                out.append(float(imp[key]) * 100.0)
    ev = record.get("evaluations") or {}
    out += [float(ev[k]) for k in ("search", "budget", "scan", "planned_budget", "added_budget")
            if isinstance(ev.get(k), (int, float))]
    target = record.get("target") or {}
    if isinstance(target.get("value"), (int, float)) and not isinstance(target.get("value"), bool):
        out.append(float(target["value"]))
    # What FI's check at finer settings measured (results/best_design.json's ``check``).
    check = record.get("check") or {}
    for key in ("objective", "baseline_objective", "improvement", "improvement_numerical_error",
                "best_numerical_error"):
        if isinstance(check.get(key), (int, float)) and not isinstance(check.get(key), bool):
            out.append(float(check[key]))
    for value in (check.get("design") or {}).values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    interval = check.get("interval") or {}
    for key in ("ci_lower", "ci_upper", "diff"):
        if isinstance(interval.get(key), (int, float)):
            out.append(float(interval[key]))
    return out


def with_fi_record(stdout: str, record: dict[str, Any]) -> tuple[str, list[str]]:
    """The analysis's output with FI's own numbers of the search added to its RESULT_JSON (under ``fi_search``), so what
    is analysed and written up next to them is FI's record, not only the script's report of it; and the numbers the
    script reports about the best design, the baseline or the improvement that FI's record does not hold."""
    lines = (stdout or "").splitlines()
    index = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("RESULT_JSON:")), None)
    if index is None:
        return stdout, []
    try:
        result = json.loads(lines[index][len("RESULT_JSON:"):].strip())
    except ValueError:
        return stdout, []
    if not isinstance(result, dict):
        return stdout, []
    known = _record_numbers(record)
    differs = []
    for key, value in result.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not any(
                w in str(key).lower() for w in ("best", "baseline", "improv", "optim")):
            continue
        if not any(math.isclose(float(value), k, rel_tol=1e-6, abs_tol=1e-12) for k in known):
            differs.append(f"{key} = {value}")
    best = record.get("best") or {}
    result["fi_search"] = {
        "best_design": best.get("design"), "best_objective": best.get("objective"),
        "baseline_objective": (record.get("baseline") or {}).get("objective"),
        "improvement": (record.get("improvement") or {}).get("value"),
        "evaluations": (record.get("evaluations") or {}).get("search"),
        "checked_at_finer_settings": record.get("checked_at_finer_settings"),
        **({"target": record["target"]} if isinstance(record.get("target"), dict) else {}),
        **({"not_in_fi_record": differs} if differs else {}),
    }
    check = record.get("check") or {}
    if check:
        # FI's check at finer settings: the verdict, and the numbers at the finest settings (the ones to report).
        result["fi_search"]["check"] = {k: check.get(k) for k in ("verdict", "says", "design", "objective",
                                                                  "baseline_objective", "improvement",
                                                                  "improvement_numerical_error",
                                                                  "best_numerical_error")}
    result = _search.plain_zero(result)
    lines[index] = "RESULT_JSON: " + json.dumps(result, allow_nan=True, default=str)
    return "\n".join(lines) + ("\n" if (stdout or "").endswith("\n") else ""), differs
