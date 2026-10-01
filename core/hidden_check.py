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

1. a step size of the case (``dt``, ``h``, ``dx``, ...) made smaller (divided by 2 or 3): a finer step never makes an
   honest simulation less accurate, so a check that passed should still pass;
2. otherwise one setting of the case that the study's own grid also sweeps, moved to another value of that grid (the
   main run already runs the simulation there), chosen at random.

Not covered: a special or limiting case, a published value and a convergence rate (each expected value belongs to its
own setting, and FI cannot work out the value at another), a check with no case FI can run, a search for the best
design (it has its own check at finer settings, core/optimum_check.py), and a check whose case has no step size and no
setting the grid sweeps. The record says which checks were not covered and why; not being covered is not a gap.

A hidden check that passes is recorded with the hash of the simulation's code (simulate.py and its package); one that fails, or could not run, is a gap below
``independently_validated`` (:func:`evidence_gaps`), as is a record made on another version of the simulation. It never
stops the quest. The case is chosen when the check runs and is written only to ``needs/HIDDEN_CHECK.json`` afterwards.
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


def _not_covered(protocol: dict[str, Any] | None, chosen: list[dict[str, Any]]) -> list[str]:
    """One sentence per declared check that is not run at a hidden setting, saying why."""
    names = {str(o["name"]).strip() for o in chosen}
    out = []
    for oracle in _oracle.declared(protocol):
        name = str(oracle["name"]).strip()
        if name in names:
            continue
        kind = _oracle.kind_of(oracle)
        if kind not in KINDS:
            what = _oracle.KINDS.get(kind or "", "a check of no kind FI reads")
            out.append(f"{name!r}: {what}; its expected value belongs to its own setting")
        elif _oracle.case_of(oracle) is None:
            out.append(f"{name!r}: it has no case FI can run")
        else:
            out.append(f"{name!r}: FI cannot judge it (no numeric expected value and tolerance)")
    return out


def derive(oracle: dict[str, Any], protocol: dict[str, Any] | None,
           rng: random.Random) -> tuple[dict[str, Any] | None, str]:
    """``(the hidden case, what was changed)`` for one check (:func:`candidates`), or ``(None, why not)``. A step size of
    the case is made smaller; else one setting the study's grid sweeps is moved to another of its values; never a
    setting another declared check already uses as its case."""
    own = _oracle.case_of(oracle)
    if own is None:
        return None, "it has no case FI can run"
    case = own[0]
    shown = [c[0] for c in (_oracle.case_of(o) for o in _oracle.declared(protocol)) if c]
    for key in _oracle._STEP_KEYS:
        step = _number(case.get(key))
        if step is not None and step > 0:
            divisor = rng.choice((2, 3))
            value: Any = step / divisor
            return {**case, key: value}, f"{key} = {_fmt(value)} instead of {_fmt(case[key])}"
    grid = protocol.get("grid") if isinstance(protocol, dict) else None
    options: list[tuple[str, Any]] = []
    for key, current in case.items():
        values = grid.get(key) if isinstance(grid, dict) else None
        if _number(current) is None or not isinstance(values, list):
            continue
        for v in values:
            if _number(v) is not None and _number(v) != _number(current) and {**case, key: v} not in shown:
                options.append((key, v))
    if options:
        key, value = rng.choice(options)
        return {**case, key: value}, f"{key} = {_fmt(value)} (a setting of the study) instead of {_fmt(case[key])}"
    return None, "its case has no step size and no setting the study's grid sweeps, so FI has no safe setting to move"


#: Code that does not compute the simulation's numbers: a change to it does not call for the hidden check again.
_NOT_THE_SIMULATION = ("analysis.py", "web_plots.py")


def code_sha(quest_root: Path) -> str:
    """One hash of the simulation's code as it is now: simulate.py and every other Python file under ``code/`` (the
    model's package), not the analysis; ``""`` when there is no simulate.py."""
    code = Path(quest_root) / "code"
    if not (Path(quest_root) / _SIMULATE).is_file():
        return ""
    digest = hashlib.sha256()
    for path in sorted(code.rglob("*.py")):
        if path.name in _NOT_THE_SIMULATION or "__pycache__" in path.parts:
            continue
        try:
            digest.update(path.relative_to(code).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        except OSError:
            return ""
    return digest.hexdigest()


def _checks_key(protocol: dict[str, Any] | None) -> str:
    """A fingerprint of the checks a hidden setting is derived from: a change to one runs the hidden check again."""
    return hashlib.sha256(json.dumps(candidates(protocol), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load(quest_root: Path) -> dict[str, Any] | None:
    try:
        record = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def write(quest_root: Path, record: dict[str, Any]) -> None:
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")


async def run(executor: Any, python: Path | str, quest_root: Path, protocol: dict[str, Any] | None, *, timeout_s: int,
              env: dict[str, str] | None, engine_callable: bool, rng: random.Random | None = None) -> dict[str, Any]:
    """Run the hidden check on the simulation as it is now and return its record (not written). A record already made
    on the same simulation and the same checks is returned as it is (an analysis repaired on the same run does not run
    it again). ``engine_callable``: FI can call the simulation on one case (``run_trial``/``run_cell``)."""
    quest_root = Path(quest_root)
    simulate = quest_root / _SIMULATE
    sha, key = code_sha(quest_root), _checks_key(protocol)
    kept = load(quest_root)
    if kept and sha and kept.get("code_sha256") == sha and kept.get("checks_key") == key and kept.get("status") in (
            "passed", "failed", "not_covered"):
        return kept
    chosen = candidates(protocol)
    record: dict[str, Any] = {"code_sha256": sha, "checks_key": key, "cases": [],
                              "not_covered": _not_covered(protocol, chosen)}
    if _optimise.block_of(protocol) is not None:
        record["not_covered"] = [f"{str(o['name']).strip()!r}: a search for the best design is checked at finer "
                                 "settings instead" for o in _oracle.declared(protocol)]
        return {**record, "status": "not_covered"}
    if not chosen:
        return {**record, "status": "not_covered"}
    if not engine_callable or not simulate.is_file():
        return {**record, "status": "not_run",
                "reason": "FI cannot call the simulation on one case (it needs `run_trial` or `run_cell` in simulate.py)"}
    rng = rng or random.SystemRandom()
    thresholds = protocol.get("thresholds") if isinstance(protocol, dict) and isinstance(protocol.get("thresholds"),
                                                                                        dict) else None
    for oracle in chosen[:MAX_CASES]:
        name = str(oracle["name"]).strip()
        case, changed = derive(oracle, protocol, rng)
        if case is None:
            record["not_covered"].append(f"{name!r}: {changed}")
            continue
        hidden = {**oracle, "case": case}
        try:
            checks, problems, _ = await _trial_runner.measure_oracles(
                executor, python, quest_root, _SIMULATE.as_posix(), [hidden], timeout_s=timeout_s, env=env,
                thresholds=thresholds, case_env=env)
        except Exception as e:  # noqa: BLE001 -- a run that cannot start is recorded, never raised
            checks, problems = [], [f"the run could not start: {type(e).__name__}: {str(e)[:200]}"]
        judged = _oracle.judged([hidden], {"checks": checks, "engine_measured": True})[0]
        record["cases"].append({
            "name": name, "kind": _oracle.kind_of(oracle), "case": case, "changed": changed,
            "value": judged["value"], "expected": judged["expected"], "limit": judged["limit"],
            "passed": judged["passed_by_engine"] if judged["measured_by"] == "engine" else None,
            **({"problem": "; ".join(problems)[:400]} if problems else {}),
        })
    for oracle in chosen[MAX_CASES:]:
        record["not_covered"].append(f"{str(oracle['name']).strip()!r}: only {MAX_CASES} checks are run again")
    if not record["cases"]:
        return {**record, "status": "not_covered"}
    return {**record, "status": "passed" if all(c["passed"] is True for c in record["cases"]) else "failed"}


def evidence_gaps(quest_root: Path, protocol: dict[str, Any] | None) -> list[str]:
    """Under research: why the hidden check keeps the result below ``independently_validated``, one sentence each.
    Empty when no declared check is of a kind it covers, or when every check run at a hidden setting passed on the
    simulation as it is now."""
    chosen = candidates(protocol)
    if not chosen or _optimise.block_of(protocol) is not None:
        return []
    record = load(quest_root)
    if record is None:
        return ["FI has not run the checks at a setting the code never saw (run the experiment again so it does)"]
    if record.get("status") == "not_run":
        return [f"FI could not run the checks at a setting the code never saw: {record.get('reason') or 'no reason given'}"]
    if record.get("code_sha256") != code_sha(quest_root):
        return ["the simulation changed after FI ran its checks at a setting the code never saw (run the experiment "
                "again so FI checks the code as it is)"]
    gaps: list[str] = []
    done = {str(c.get("name")) for c in record.get("cases") or [] if isinstance(c, dict)}
    told = " ".join(str(n) for n in record.get("not_covered") or [])
    for oracle in chosen[:MAX_CASES]:
        name = str(oracle["name"]).strip()
        if name not in done and repr(name) not in told:
            gaps.append(f"the check {name!r} was not run at a setting the code never saw (it was added after FI's run; "
                        "run the experiment again)")
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
