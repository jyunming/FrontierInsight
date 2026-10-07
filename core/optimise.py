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

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import flat_output as _flat
from . import optimisation_plan as _plan
from . import optimise_search as _search
from . import trial_runner as _trials

LEDGER_PATH = Path(_trials.RAW_DIRNAME) / "optimisation_ledger.jsonl"
BEST_PATH = Path("results") / "best_design.json"
WORK_DIR = Path(".fi") / "optimisation"
RECORD = WORK_DIR / "run.json"
SEARCH_SOURCE = WORK_DIR / "optimise_search.py"
#: The rounds of a continued search a person asked for (core/optimise_refine.py), kept for the plan they were asked under.
ROUNDS_PATH = WORK_DIR / "continue.json"
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


def complete_case(protocol: Any, case: dict[str, Any]) -> dict[str, Any]:
    """An oracle's case as the simulation is called with it during the search: the conditions held fixed, the search's
    numerical settings and the baseline design, with what the case itself names on top (a check of the baseline may
    name only some of its variables)."""
    block = block_of(protocol)
    if block is None or not isinstance(case, dict):
        return dict(case or {})
    baseline = (block.get("baseline") or {}).get("values") if isinstance(block.get("baseline"), dict) else None
    return {**_search.cell_of(block, dict(baseline) if isinstance(baseline, dict) else {}), **case}


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
             rounds: list[dict[str, Any]] | None = None) -> str:
    """What decides a search: the simulation and the modules beside it, the optimisation block, the seed, the entry,
    the thresholds, and FI's own harness and search code. (A package installed since is not in it: a search that got no
    result is never kept, so a search that failed for a missing package runs again anyway.)"""
    parts: list[bytes] = []
    code = Path(quest_root) / "code"
    for path in sorted([Path(simulate), *(p for p in code.glob("*.py") if p.name not in _NOT_THE_SIMULATION
                                          and p.resolve() != Path(simulate).resolve())]):
        try:
            parts.append(path.name.encode() + b"\0" + path.read_bytes())
        except OSError:
            parts.append(path.name.encode() + b"\0")
    parts.append(_trials.HARNESS_SOURCE.encode("utf-8"))
    parts.append(Path(_search.__file__).read_bytes())
    what: list[Any] = [block, base, entry, thresholds]
    if rounds:
        # A continued search is another search; without rounds the key is the one an earlier version of FI kept.
        what.append(_key_rounds(rounds))
    return _sha(b"\1".join(parts) + json.dumps(what, sort_keys=True, default=str).encode())


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


def _mean_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = set.intersection(*(set(r.get("values") or {}) for r in rows)) if rows else set()
    return {name: sum(float(r["values"][name]) for r in rows) / len(rows) for name in sorted(names)}


async def _evaluate(executor: Any, python: Any, quest_root: Path, module: str, entry: str, cell: dict[str, Any], *,
                    runs: int, base_seed: int, timeout_s: int, env: dict[str, str] | None,
                    thresholds: dict[str, Any] | None, seed_key: str = "") -> tuple[dict[str, Any], str, str]:
    """One design, in a process of its own through FI's harness: ``(answer for the search, stderr, load error)``. The
    answer's ``trials`` holds each run's own values (in trial order), beside their mean in ``values``. ``seed_key`` picks
    another family of seeds (the check at finer settings uses seeds the search never used)."""
    work = quest_root / WORK_DIR
    work.mkdir(parents=True, exist_ok=True)
    spec, out = work / "eval.json", work / "eval.out.jsonl"
    out.unlink(missing_ok=True)
    trials = [{"trial": t, "seed": None if entry == "run_cell" else _trials.trial_seed(base_seed, seed_key, t)}
              for t in range(1 if entry == "run_cell" else max(1, runs))]
    nonce = _sha(f"optimise|{time.time_ns()}|{id(trials)}".encode())[:24]
    spec.write_text(json.dumps({"module": module, "entry": entry, "cell": cell, "trials": trials, "nonce": nonce,
                                "thresholds": dict(thresholds or {})}, default=str), encoding="utf-8")
    # Fresh for every evaluation: nothing an earlier evaluation wrote there is run.
    (quest_root / _trials.HARNESS_PATH).write_text(_trials.HARNESS_SOURCE, encoding="utf-8")
    started = time.monotonic()
    result = await executor.execute(
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
            return {"failed": why, "elapsed_s": elapsed}, stderr, ""
        if t["trial"] in twice:
            return {"failed": "a run of this design was reported more than once", "elapsed_s": elapsed}, stderr, ""
        if row.get("status") != "ok" or row.get("seed") != t["seed"]:
            why = str(row.get("reason") or "the reported seed is not the one FI gave") if row.get("status") != "ok" \
                else "the reported seed is not the one FI gave"
            return {"failed": why, "elapsed_s": elapsed}, stderr, ""
        rows.append(row)
    return {"values": _mean_values(rows), "trials": [dict(r.get("values") or {}) for r in rows],
            "elapsed_s": elapsed}, stderr, ""


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
    result = await executor.execute(
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
                     frozen: bool = False) -> SearchRun:
    """Run the search for the best design (see the module docstring) and write its record. ``module`` is the simulation
    file relative to ``quest_root``; ``timeout_s`` bounds the whole search, the coarse scan included.

    ``frozen``: the study is frozen for its confirm run (explore, then confirm: ``core/phased.py``). The search is not
    run again: the best design exploration's search found, for this same simulation and plan, is kept, and only FI's
    check of it (with the confirm run's new seeds, :mod:`core.optimum_check`) runs. Searching again on the confirm seeds
    would choose the design on the data meant only to confirm it. A frozen study whose search record is not there to
    keep cannot be confirmed without searching again: it gets no result."""
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
    entry = "run_trial" if "run_trial" in have else "run_cell" if "run_cell" in have else ""
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
    if not entry:
        why = (f"{Path(module).name} defines neither run_cell(cell) nor run_trial(cell, trial_id, seed): FI calls one of "
               "them for each design the search tries")
        return SearchRun(record={}, rows=[], load_error=why, problems=[why])
    rounds = read_rounds(quest_root, block)
    key = _run_key(quest_root, simulate, block, int(base_seed), entry, thresholds, rounds)
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
    replay_scan: list[dict[str, Any]] | None = None
    if rounds and not frozen and isinstance(record, dict) and record.get("key") != key and any(
            record.get("key") == _run_key(quest_root, simulate, block, int(base_seed), entry, thresholds, rounds[:k])
            for k in range(len(rounds))):
        earlier = _record_files(record, record_text).get(LEDGER_PATH.as_posix(), "")
        space = _search.Space(block["design_variables"])
        quantity = str(block["objective"]["quantity"])
        scan_seen: list[dict[str, Any]] = []
        for line in earlier.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") != "evaluation" or not isinstance(row.get("design"), dict):
                continue
            answer: dict[str, Any] = ({"values": {quantity: row["objective"], **(row.get("constraints") or {})}}
                                      if row.get("status") == "ok" else {"failed": str(row.get("reason") or "failed")})
            answer["elapsed_s"] = row.get("elapsed_s")
            if row.get("stage") == "scan":
                scan_seen.append({"design": row["design"], "values": answer.get("values"),
                                  "reason": answer.get("failed", ""), "elapsed_s": row.get("elapsed_s")})
            else:
                replay[space.key(space.canonical(row["design"]))] = answer
        if scan_seen and (quest_root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME).is_file():
            replay_scan = scan_seen
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
    started = time.monotonic()
    deadline = started + max(1, int(timeout_s))
    runs = int(block.get("runs_per_evaluation") or 1) if entry == "run_trial" else 1
    stderr_parts: list[str] = []
    method, _where = _plan.effective_method(block)

    scan_rows: list[dict[str, Any]] = []
    if not (isinstance(block.get("grid"), dict) and block["grid"]):
        # No coarse scan: an earlier measurement's trial record is not this search's.
        for rel in (Path(_trials.RAW_DIRNAME) / _trials.LEDGER_NAME, Path(_trials.RAW_DIRNAME) / _trials.SUMMARY_NAME,
                    _trials.RUN_RECORD):
            (quest_root / rel).unlink(missing_ok=True)

    async def scan(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """The coarse scan, through the trial runner itself (``raw/trials.json``, which the analysis can plot)."""
        if replay_scan is not None:
            return replay_scan
        left = deadline - time.monotonic()
        if left <= 0:
            return []
        trial_run = await _trials.run_trials(
            executor, python, quest_root, module, grid, runs_per_setting=runs, base_seed=int(base_seed),
            deterministic=entry == "run_cell", timeout_s=max(1, math.ceil(left)), env=env,
            thresholds=thresholds, paired=True,
        )
        stderr_parts.append(trial_run.stderr())
        summary = json.loads(trial_run.summary_path.read_text(encoding="utf-8"))
        reasons = {c.key: next((r.get("reason") for r in c.rows if r.get("reason")), "") for c in trial_run.cells}
        for cell in summary.get("cells") or []:
            cell["reason"] = reasons.get(cell.get("key"), "")
        rows = _search.scan_rows(summary, block)
        if log is not None:
            log.info("[optimise] the coarse scan ran %d design(s) after the baseline (%d worked); it is not counted in "
                     "the search's budget", len(rows), sum(1 for r in rows if r.get("values") is not None))
        return rows

    gen = _search.search(block, seed=int(base_seed), method=method, rounds=rounds)
    replay_space = _search.Space(block["design_variables"]) if replay else None
    load_error = ""
    done_outcome: dict[str, Any] | None = None
    last_note = time.monotonic()
    evaluated = 0
    try:
        request = next(gen)
        while True:
            if request["kind"] == "scan":
                reply = await scan(request["grid"])  # type: ignore[assignment]
                scan_rows = list(reply)
            elif request["kind"] == "evaluate":
                left = deadline - time.monotonic()
                replayed = (replay.get(replay_space.key(replay_space.canonical(request["design"])))
                            if replay_space is not None and isinstance(request.get("design"), dict) else None)
                if replayed is not None:
                    reply: dict[str, Any] = dict(replayed)
                elif load_error:
                    reply = {"stop": "the simulation could not be loaded"}
                elif left <= 0:
                    reply = {"stop": "time"}
                else:
                    reply, err, load_error = await _evaluate(
                        executor, python, quest_root, module, entry, request["cell"], runs=runs,
                        base_seed=int(base_seed), timeout_s=left, env=env, thresholds=thresholds,
                    )
                    if err:
                        stderr_parts.append(err)
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
                log.info("[optimise] still searching: %d evaluation(s) so far", evaluated)
            request = gen.send(reply)
    except StopIteration as stop:
        done_outcome = stop.value
    outcome = done_outcome or {}
    elapsed = time.monotonic() - started
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
    }
    if outcome.get("stopped_because") == "time":
        extra["time"]["limit_seconds"] = int(timeout_s)
    record = _search.best_design(outcome, block, seed=int(base_seed),
                                 check={str(n): _plan.check_levels(s if isinstance(s, dict) else {"search": s})[0]
                                        for n, s in (block.get("numerical_settings") or {}).items()},
                                 extra=extra)
    if warnings:
        record["simulation_may_search_itself"] = warnings
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
    record = json.loads(best_text)
    return SearchRun(record=record, rows=outcome.get("rows") or [], stderr=stderr, scan=bool(scan_rows),
                     load_error=load_error, files=_files(ledger_text, best_text, record_text))


# --- run.log ---------------------------------------------------------------------------------------------------------

_STOPPED = {
    "budget": "the evaluation budget ran out",
    "share": "each starting point converged or used its share of the budget",
    "converged": "every starting point converged",
    "finished": "every combination was evaluated",
    "time": "the time limit (execution.timeout_s) was reached",
    "failures": "the first evaluations all failed",
    "library_error": "the library's method stopped with an error",
    "the simulation could not be loaded": "the simulation could not be loaded",
    "target": "a design reached the target asked for",
    "not_run": "it did not run: the search before it had stopped (its time limit, or every evaluation failed)",
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
    for r in record.get("continued") or []:
        target = f", toward {quantity} = {_fmt(r['target'])}{unit}" if r.get("target") is not None else ""
        reached = ("; the target was reached at the search's settings" if r.get("target_reached") else
                   "; the target was not reached at the search's settings" if r.get("target") is not None else "")
        lines.append(f"[optimise] continued at the person's request (round {r.get('round')}): {r.get('evaluations')} of "
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


class OptimisationRunner:
    """Stands in for the executor when the plan is a search for the best design, as ``trial_runner.TrialsRunner`` does
    for a measurement: running ``experiment.py`` first runs the search (:func:`run_search`), then the analysis, with
    ``FI_OPTIMISATION`` naming FI's ledger and ``FI_BEST_DESIGN`` its best-design file (and ``FI_TRIALS`` the coarse
    scan's per-setting results, when there was one). ``failed_script`` says which script to repair; ``last`` keeps the
    search for the checks after it."""

    def __init__(self, executor: Any, *, quest_root: Path, protocol: Any, simulate: Path, analysis: Path,
                 log: Any = None, frozen: Any = False) -> None:
        self.executor = executor
        #: Whether the study is frozen for its confirm run (a bool, or a callable asked at each run): the search is
        #: then not run again (:func:`run_search`).
        self._frozen = frozen
        self.quest_root = Path(quest_root)
        self._protocol = protocol
        self.simulate = Path(simulate)
        self.analysis = Path(analysis)
        self.log = log
        self.failed_script: str | None = None
        #: Why the last run was sent back as a simulation that does not use its inputs (core/flat_output.py), or ``None``.
        self.flat: str | None = None
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
        run = await run_search(self.executor, cmd[0], self.quest_root,
                               self.simulate.relative_to(self.quest_root).as_posix(), protocol, base_seed=base,
                               timeout_s=timeout_s, env=env, log=self.log,
                               frozen=self._is_frozen())
        self.last = run
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
        # A search in which every design gave the very same numbers is not a result: the simulation does not use the
        # design it is given. It is sent back to be repaired like one that does not run (core/flat_output.py).
        norm_block = _plan.normalize(block_of(protocol))[0] or block_of(protocol) or {}
        flat = _flat.flat_search(run.rows, str((norm_block.get("objective") or {}).get("quantity") or "the objective"))
        if flat is not None:
            return self._flat_result(flat, run, started)
        # FI checks the best design at finer settings (core/optimum_check.py), with a time limit of its own. A check that
        # fails is the study's result, reported and carried on with; one that cannot run is recorded as not finished.
        check_text, check_key = await self._check(cmd[0], run, protocol, base, timeout_s, env)
        self.last_check = (check_text, check_key)
        try:
            checked = json.loads(check_text) if check_text else None
        except ValueError:
            checked = None
        flat = _flat.flat_search_check(checked, norm_block)
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
                protocol, run, base_seed=base, timeout_s=timeout_s, env=env, log=self.log,
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
