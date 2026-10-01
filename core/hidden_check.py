"""FI's own check at a setting the code never saw (a hidden check), under ``rigor_profile: research``.

Every check against a known answer the plan declares is shown to the model that writes the simulation: its case, its
measure, its expected value. Code can therefore be written to pass the cases it was shown (special-cased, or tuned to
them) and be wrong everywhere else. So after the run, with the simulation's code as it ran, FI makes one more run of
some checks at a setting of FI's own choosing, which no prompt ever named, through the same harness the checks use
(:func:`core.trial_runner.measure_oracles`: FI calls ``run_trial``/``run_cell`` on the case in its own process and
computes the check's number itself), and judges it against the plan's own expected value and tolerance.

Where it is well defined: a check whose expected value does not depend on the setting. Those are the kinds whose
numeric form is "the worst violation, expecting 0" (:mod:`core.oracle_forms`): an **invariant**, a **symmetry** and a
**second implementation**. A conserved quantity is conserved, a symmetry holds and two implementations agree at any
valid setting, so the plan's expected value (0) and tolerance still apply. The setting changed, in this order:

1. a step size of the case (``dt``, ``h``, ``dx``, ...; a decimal number, never a count) made smaller by a factor
   drawn between 1.7 and 3.3: a finer step does not make an honest simulation less accurate (a tolerance near
   rounding error can still be exceeded by the extra steps);
2. otherwise a decimal setting of the case that the study's grid also sweeps, moved to a value drawn between the
   grid's smallest and largest values that is none of them;
3. otherwise a whole-number setting the grid sweeps, moved to another of the grid's values: the main run ran the code
   there, but no check did, and the record says so.

Never a case another declared check uses. The setting is drawn when the check runs, after the code is final.

Not covered: a special or limiting case, a published value and a convergence rate (each expected value belongs to its
own setting, and FI cannot work out the value at another), a check with no case FI can run, a search for the best
design (it has its own check at finer settings, core/optimum_check.py), and a check whose case has no step size and no
setting the grid sweeps. The record says which checks were not covered and why; not being covered is not a gap.

The record (``needs/HIDDEN_CHECK.json``) keeps the hash of the simulation's code (``code/`` but the analysis and the
notes) and of the checks, and FI puts the record's own hash in the quest's trace when it writes it: a record the
simulation's code wrote or changed is not FI's (:func:`evidence_gaps`). A check that failed or could not run, a record
on other code or other checks, and an earlier failure on the same code are gaps below ``independently_validated``. It
never stops the quest. The harness's copy of the case is removed after the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from . import optimise as _optimise
from . import oracle_check as _oracle
from . import trial_runner as _trial_runner

#: Where the record is kept (quest-relative).
RECORD = "needs/HIDDEN_CHECK.json"
#: The kinds of check a hidden setting is well defined for: their expected value (0) holds at every setting.
KINDS = ("invariant", "symmetry", "second_implementation")
#: At most this many checks are run again at a hidden setting (each is one more run of the simulation).
MAX_CASES = 3
_SIMULATE = Path("code") / "simulate.py"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def candidates(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The declared checks a hidden setting is well defined for: one of :data:`KINDS`, with a case FI can run and an
    expected value and tolerance FI can judge by."""
    out = []
    for oracle in _oracle.declared(protocol):
        if (_oracle.kind_of(oracle) in KINDS and _oracle.case_of(oracle) is not None
                and _oracle.limit_of(oracle)[1] is not None):
            out.append(oracle)
    return out


def _not_covered(protocol: dict[str, Any] | None, chosen: list[dict[str, Any]]) -> list[dict[str, str]]:
    """``{name, why}`` for each declared check that is not run at a hidden setting."""
    names = {str(o["name"]).strip() for o in chosen}
    out = []
    for oracle in _oracle.declared(protocol):
        name = str(oracle["name"]).strip()
        if name in names:
            continue
        kind = _oracle.kind_of(oracle)
        if kind not in KINDS:
            why = f"{_oracle.KINDS.get(kind or '', 'a check of no kind FI reads')}; its expected value belongs to its own setting"
        elif _oracle.case_of(oracle) is None:
            why = "it has no case FI can run"
        else:
            why = "FI cannot judge it (no numeric expected value and tolerance)"
        out.append({"name": name, "why": why})
    return out


def _round(value: float, digits: int = 4) -> float:
    """``value`` to ``digits`` significant digits (a setting a person can read in the record)."""
    return float(f"{value:.{digits}g}")


def derive(oracle: dict[str, Any], protocol: dict[str, Any] | None,
           rng: random.Random) -> tuple[dict[str, Any] | None, str]:
    """``(the hidden case, what was changed)`` for one check (:func:`candidates`), or ``(None, why not)``. Never a case
    a declared check already uses. In this order:

    1. a step size of the case (a decimal number, not a count) made smaller by a factor drawn between 1.7 and 3.3;
    2. a decimal setting the study's grid sweeps, moved to a value drawn between the grid's smallest and largest
       values that is none of them (no prompt named it);
    3. a whole-number setting the grid sweeps, moved to another of the grid's values: the code ran there in the main
       run, but no check did (the record says so)."""
    own = _oracle.case_of(oracle)
    if own is None:
        return None, "it has no case FI can run"
    case = own[0]
    shown = [o["case"] for o in _oracle.declared(protocol) if isinstance(o.get("case"), dict)]
    for key in _oracle._STEP_KEYS:
        step = case.get(key)
        if isinstance(step, float) and math.isfinite(step) and step > 0:
            value = _round(step / rng.uniform(1.7, 3.3))
            if {**case, key: value} not in shown:
                return {**case, key: value}, f"{key} = {_fmt(value)} instead of {_fmt(step)}"
    grid = protocol.get("grid") if isinstance(protocol, dict) else None
    between: list[tuple[str, float, float]] = []
    others: list[tuple[str, Any]] = []
    for key, current in case.items():
        values = grid.get(key) if isinstance(grid, dict) else None
        numbers = [v for v in values if _number(v) is not None] if isinstance(values, list) else []
        if _number(current) is None or not numbers:
            continue
        if isinstance(current, float) and all(isinstance(v, float) for v in numbers):
            low, high = min(float(v) for v in numbers + [current]), max(float(v) for v in numbers + [current])
            if high > low:
                between.append((key, low, high))
        others += [(key, v) for v in numbers if _number(v) != _number(current) and {**case, key: v} not in shown]
    if between:
        key, low, high = rng.choice(between)
        taken = {_number(v) for v in grid.get(key) or []} | {_number(case[key])}
        for _ in range(20):
            value = _round(rng.uniform(low, high))
            if low < value < high and value not in taken and {**case, key: value} not in shown:
                return ({**case, key: value},
                        f"{key} = {_fmt(value)} (between the study's settings) instead of {_fmt(case[key])}")
    if others:
        key, value = rng.choice(others)
        return ({**case, key: value}, f"{key} = {_fmt(value)} (another setting of the study's grid, where no check "
                                      f"runs) instead of {_fmt(case[key])}")
    return None, "its case has no step size and no setting the study's grid sweeps, so FI has no safe setting to move"


#: What under ``code/`` does not compute the simulation's numbers: a change to it does not call for the check again.
#: The same files ``Engine._simulation_sources`` does not read as the simulation, and FI's figure redraws (written
#: after every run).
_NOT_THE_SIMULATION = ("analysis.py", "experiment.py", "web_plots.py", "replot_figures.py", "replot_figures.json",
                       "replot_layout.py", "run.py", "submit.py", "fi_search.py")
_NOT_THE_SIMULATION_DIRS = ("__pycache__", ".git")


def code_sha(quest_root: Path) -> str:
    """One hash of the simulation's code as it is now: simulate.py and every other file under ``code/`` it may use
    (the model's package, a parameter file), not the analysis, the notes (``*.md``) or git's own; ``""`` when there is
    no simulate.py."""
    code = Path(quest_root) / "code"
    if not (Path(quest_root) / _SIMULATE).is_file():
        return ""
    digest = hashlib.sha256()
    for path in sorted(p for p in code.rglob("*") if p.is_file()):
        rel = path.relative_to(code)
        if (path.name in _NOT_THE_SIMULATION or path.suffix.lower() == ".md"
                or any(part in _NOT_THE_SIMULATION_DIRS for part in rel.parts)):
            continue
        try:
            digest.update(rel.as_posix().encode("utf-8") + b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            digest.update(b"\0")
        except OSError:
            return ""
    return digest.hexdigest()


def checks_key(protocol: dict[str, Any] | None) -> str:
    """A fingerprint of the checks a hidden setting is derived from: a change to one calls for the check again."""
    return hashlib.sha256(json.dumps(candidates(protocol), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def record_sha(quest_root: Path) -> str:
    """The SHA-256 of the record as it is on disk, ``""`` when there is none. FI puts it in the quest's trace when it
    writes the record, so a record the simulation's own code wrote or changed is told apart."""
    try:
        return hashlib.sha256((Path(quest_root) / RECORD).read_bytes()).hexdigest()
    except OSError:
        return ""


def load(quest_root: Path) -> dict[str, Any] | None:
    try:
        record = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def write(quest_root: Path, record: dict[str, Any]) -> str:
    """Write the record; returns its SHA-256 (for the trace)."""
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return record_sha(quest_root)


def _forget_case(quest_root: Path) -> None:
    """The harness leaves the case it ran in ``.fi/trials/``: removed, so later code cannot read the hidden setting."""
    for name in ("oracle.json", "oracle.out.jsonl"):
        try:
            (Path(quest_root) / ".fi" / "trials" / name).unlink(missing_ok=True)
        except OSError:
            pass


async def run(executor: Any, python: Path | str, quest_root: Path, protocol: dict[str, Any] | None, *, timeout_s: int,
              env: dict[str, str] | None, engine_callable: bool, rng: random.Random | None = None,
              trusted_sha: str = "") -> dict[str, Any]:
    """Run the hidden check on the simulation as it is now and return its record (not written). A record FI itself
    wrote (its SHA-256 is ``trusted_sha``, from the trace) on the same code and the same checks is returned as it is,
    so an analysis repaired on the same run does not run it again. ``engine_callable``: FI can call the simulation on
    one case (``run_trial``/``run_cell``)."""
    quest_root = Path(quest_root)
    simulate = quest_root / _SIMULATE
    sha, key = code_sha(quest_root), checks_key(protocol)
    kept = load(quest_root)
    if (kept and sha and trusted_sha and record_sha(quest_root) == trusted_sha and kept.get("code_sha256") == sha
            and kept.get("checks_key") == key and kept.get("status") in ("passed", "failed", "not_covered")):
        return kept
    chosen = candidates(protocol)
    record: dict[str, Any] = {"code_sha256": sha, "checks_key": key, "cases": [],
                              "not_covered": _not_covered(protocol, chosen)}
    if _optimise.block_of(protocol) is not None:
        record["not_covered"] = [{"name": str(o["name"]).strip(), "why": "a search for the best design is checked at "
                                  "finer settings instead"} for o in _oracle.declared(protocol)]
        return {**record, "status": "not_covered"}
    if not chosen:
        return {**record, "status": "not_covered"}
    if not engine_callable or not simulate.is_file():
        return {**record, "status": "not_run",
                "reason": "FI cannot call the simulation on one case (it needs `run_trial` or `run_cell` in simulate.py)"}
    rng = rng or random.SystemRandom()
    thresholds = protocol.get("thresholds") if isinstance(protocol, dict) and isinstance(protocol.get("thresholds"),
                                                                                        dict) else None
    planned: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for oracle in chosen:
        name = str(oracle["name"]).strip()
        case, changed = derive(oracle, protocol, rng)
        if case is None:
            record["not_covered"].append({"name": name, "why": changed})
        elif len(planned) < MAX_CASES:
            planned.append((oracle, case, changed))
        else:
            record["not_covered"].append({"name": name, "why": f"only {MAX_CASES} checks are run again"})
    try:
        for oracle, case, changed in planned:
            hidden = {**oracle, "case": case}
            try:
                checks, problems, _ = await _trial_runner.measure_oracles(
                    executor, python, quest_root, _SIMULATE.as_posix(), [hidden], timeout_s=timeout_s, env=env,
                    thresholds=thresholds, case_env=env)
            except Exception as e:  # noqa: BLE001 -- a run that cannot start is recorded, never raised
                checks, problems = [], [f"the run could not start: {type(e).__name__}: {str(e)[:200]}"]
            judged = _oracle.judged([hidden], {"checks": checks, "engine_measured": True})[0]
            record["cases"].append({
                "name": str(oracle["name"]).strip(), "kind": _oracle.kind_of(oracle), "case": case, "changed": changed,
                "value": judged["value"], "expected": judged["expected"], "limit": judged["limit"],
                "passed": judged["passed_by_engine"] if judged["measured_by"] == "engine" else None,
                **({"problem": "; ".join(problems)[:400]} if problems else {}),
            })
    finally:
        _forget_case(quest_root)
    if not record["cases"]:
        return {**record, "status": "not_covered"}
    verdicts = [c["passed"] for c in record["cases"]]
    # "failed" only for a value outside the tolerance; a case that gave no value (a timeout, a crash) is run again.
    status = "failed" if False in verdicts else "could_not_run" if None in verdicts else "passed"
    return {**record, "status": status}


def evidence_gaps(quest_root: Path, protocol: dict[str, Any] | None,
                  written: list[dict[str, Any]] | None = None) -> list[str]:
    """Under research: why the hidden check keeps the result below ``independently_validated``, one sentence each.
    Empty when no declared check is of a kind it covers, or when FI's own record (its SHA-256 the last one ``written``
    names: the trace's ``hidden_check`` events, oldest first) says every check run at a hidden setting passed, on the
    code and the checks as they are now, and no earlier run of the same code failed one."""
    chosen = candidates(protocol)
    if not chosen or _optimise.block_of(protocol) is not None:
        return []
    written = [w for w in written or [] if isinstance(w, dict)]
    sha, key = code_sha(quest_root), checks_key(protocol)
    gaps: list[str] = []
    if any(w.get("status") == "failed" and sha and w.get("code_sha256") == sha and w.get("checks_key") == key
           for w in written[:-1]):
        gaps.append("an earlier run of this same code failed a check at a setting the code never saw (see the quest's "
                    "trace, event `hidden_check`); FI checks again after the code is changed and run")
    record = load(quest_root)
    if record is None:
        return [*gaps, "FI did not run the checks at a setting the code never saw (it does so after each run of a "
                       "simulation in its own script, `code/simulate.py`, that FI can call on one case)"]
    if not written or written[-1].get("sha256") != record_sha(quest_root):
        return [*gaps, f"the record of the checks at a setting the code never saw ({RECORD}) is not the one FI wrote"]
    if record.get("status") == "not_run":
        return [*gaps, "FI could not run the checks at a setting the code never saw: "
                       f"{record.get('reason') or 'no reason given'}"]
    if record.get("code_sha256") != sha or record.get("checks_key") != key:
        return [*gaps, "the simulation or its checks changed after FI ran them at a setting the code never saw (run "
                       "the experiment again so FI checks them as they are)"]
    done = {str(c.get("name")) for c in record.get("cases") or [] if isinstance(c, dict)}
    told = {str(n.get("name")) for n in record.get("not_covered") or [] if isinstance(n, dict)}
    for oracle in chosen:
        name = str(oracle["name"]).strip()
        if name not in done and name not in told:
            gaps.append(f"the check {name!r} was not run at a setting the code never saw (run the experiment again)")
    for case in record.get("cases") or []:
        if not isinstance(case, dict) or case.get("passed") is True:
            continue
        name, changed = str(case.get("name")), str(case.get("changed") or "another setting")
        if case.get("passed") is False:
            gaps.append(f"the check {name!r} passed at its own case but not at a setting the code never saw ({changed}): "
                        f"FI measured {_fmt(case.get('value'))}, expected {_fmt(case.get('expected'))} within "
                        f"{_fmt(case.get('limit'))}")
        else:
            gaps.append(f"the check {name!r} could not be run at a setting the code never saw ({changed}): "
                        f"{case.get('problem') or 'no value came back'}")
    return gaps
