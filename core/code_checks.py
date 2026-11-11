"""FI's own judgement of a code folder, made in one call and without repairing anything.

In agent mode one coding agent writes the whole program (``code/``); afterwards FI judges it with the same checks it uses
for code it wrote itself. Those checks live in the engine's gate pass (``Engine._gate_pass``), where each one also repairs
what it finds. :func:`check_code` is the judging half alone: it calls the existing check functions, in the order of the
gate pass, asks no model, writes nothing into ``code/`` and returns one :class:`Verdict` per check. The caller (an agent
loop, a replay of a recorded quest) decides what to do with them.

Checks composed, in order (the ``check`` name of the verdict in brackets):

1. typed-in results (``typed_results``): a returned value that is a number typed into the code (:mod:`core.typed_results`);
2. the static problems of :func:`core.static_checks.static_checks` (``grid_keys``: a setting the study does not have;
   ``typed_constants``: numbers typed into the model's functions that the plan never gives; ``independent_check``: a
   "second implementation" that compares a value with itself);
3. FI's small run (``small_run``: the simulation can be loaded and called on a few trials at the middle and the corners of
   the grid, :mod:`core.small_run`) and the failed trials of each setting it gave (``trial_checks``:
   :func:`core.trial_checks.judge`);
4. one small test per equation of the plan's model (``equation_tests``: the functions are called in a process of their own
   and judged in FI's, :func:`core.equation_tests.run_and_judge`);
5. the known-answer checks the plan declares (``known_answers``: :func:`core.trial_runner.measure_oracles` runs the
   simulation on each check's case and :func:`core.oracle_check.judged` judges the number);
6. a swept setting of the grid that nothing reads (``unread_settings``: :func:`core.run_checks.unread_findings`);
7. a figure drawn from numbers typed into the plotting code (``figure_data``:
   :func:`core.figure_data_check.typed_series_in`).

``passed`` is ``True`` (the check ran and found nothing), ``False`` (it found a fault) or ``None`` (it could not be
judged: nothing to judge, or the check itself could not be made; ``says`` gives the reason). A fault of the plan or of FI's
own tool is never a ``False``. The ``says`` text of a verdict names the check, the function or equation and the value
measured, and never an expected value, a tolerance or a threshold; ``details`` follows the same rule, because both are what
an agent may be shown.

Not composed yet (each needs the Engine or a model, and none is copied here):

* the code reading by a second model and the finding loop (model calls);
* the second version of the model's functions (:mod:`core.second_impl`; model calls, and off when the package was not
  written function by function);
* the equation labels and the code-layout stops (``Engine._code_problems_stop``, ``_cannot_repair_stop``);
* the calibration of the reported intervals and standard errors against the true values
  (:func:`core.trial_checks.calibration`): it needs at least ``trial_checks.MIN_TRIALS`` trials per setting and the small
  run makes 3, so it is judged on the full run by ``TrialsRunner``; the ``trial_checks`` verdict says so when the plan
  declares a true value;
* the sizing of the run (``_size_the_run``, ``_pilot_gate``), the hidden check, the gate receipt and the runtime seed check:
  they time or re-run the whole study and read the engine's config.

Differences from the gate pass: the equation tests do not rename a function's parameters to the plan's names first (the
engine does, which rewrites code; a mismatch is a fault here), and the topic's own text is not available to the
typed-constants check (only ``plan.md`` is), so a number traceable only to the topic is flagged.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import code_layout as _layout
from . import equation_tests as _eqt
from . import figure_data_check as _figdata
from . import grid_keys as _grid_keys
from . import in_run as _in_run
from . import mc_error as _mc_error
from . import metric_spec as _metric_spec
from . import optimise as _optimise
from . import optimise_search as _optimise_search
from . import oracle_check as _oracle
from . import outside_run as _outside
from . import run_checks as _run_checks
from . import small_run as _small
from . import split_run as _split_run
from . import static_checks as _static
from . import trial_checks as _trial_checks
from . import trial_runner as _trials
from . import typed_results as _typed
from .execution import scrub_secrets

#: How long one check may take when the caller names no limit (seconds).
DEFAULT_TIMEOUT_S = 120
#: The time the equation tests are given in all (what the engine gives them).
EQUATION_TIMEOUT_S = 300
_LOG = logging.getLogger("frontier_insight.code_checks")


@dataclass
class Verdict:
    """What one check of FI found in the code. ``passed``: ``True`` nothing found, ``False`` a fault, ``None`` it could not
    be judged. ``says``: one plain line (the check, the function or equation, the value measured; no expected value).
    ``details``: the same facts as data, under the same rule."""

    check: str
    passed: bool | None
    says: str
    details: dict[str, Any] = field(default_factory=dict)


def _one(text: Any, limit: int = 300) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def _number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.6g}"


def _files(code_dir: Path) -> dict[str, str]:
    return _grid_keys.code_files(code_dir)


def _equation_sources(code_dir: Path) -> dict[str, str]:
    """The files that can hold a function an equation test imports: the packages and ``simulate.py`` (``experiment.py``
    runs when it is imported, so it is never one). The same set the engine's equation gate reads."""
    out = dict(_layout.package_sources(code_dir))
    simulate = code_dir / _split_run.SIMULATE_NAME
    if simulate.is_file():
        out[_split_run.SIMULATE_NAME] = simulate.read_text(encoding="utf-8")
    return out


# ---- the checks -------------------------------------------------------------------------------------------------------


def _typed_results(code_dir: Path, protocol: dict[str, Any] | None, design: dict[str, Any]) -> Verdict:
    simulate = code_dir / _split_run.SIMULATE_NAME
    if not simulate.is_file():
        return Verdict("typed_results", None, f"there is no {simulate.name} to read")
    found = _typed.typed_results(simulate.read_text(encoding="utf-8"), simulate.name,
                                 exempt=_run_checks.typed_result_exempt(protocol, design))
    if not found:
        return Verdict("typed_results", True, f"{simulate.name} computes the results it returns")
    return Verdict("typed_results", False,
                   "a returned result is a number typed into the code (" + "; ".join(f.says(with_value=False) for f in found[:3]) + ")",
                   {"results": [{"script": f.script, "function": f.function, "key": f.key, "line": f.line} for f in found]})


def _static_verdicts(files: dict[str, str], protocol: dict[str, Any] | None, design: dict[str, Any]) -> list[Verdict]:
    out: list[Verdict] = []
    for name, module, _on_standing in _static.static_checks():
        if not files:
            out.append(Verdict(name, None, "there is no code to read"))
            continue
        try:
            found = module.problems(files, protocol, design)
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            out.append(Verdict(name, None, f"the check could not read the code ({exc!r})"[:300]))
            continue
        if found:
            out.append(Verdict(name, False, _one(module.plain_note(found, protocol)), module.record(found, protocol)))
        else:
            out.append(Verdict(name, True, "nothing found"))
    return out


def _timed_out(small: _small.SmallRun) -> list[str]:
    return [s.key or "the one setting" for s in small.settings if s.timed_out]


async def _small_run_verdicts(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                              env: dict[str, str], timeout_s: int) -> list[Verdict]:
    simulate = quest_root / "code" / _split_run.SIMULATE_NAME
    protocol = protocol if isinstance(protocol, dict) else {}
    entries = _trials.entries(simulate) if simulate.is_file() else set()
    if not entries & {"run_trial", "run_cell"}:
        why = "the simulation does not follow the trial contract (simulate.py defines no run_trial or run_cell)"
        return [Verdict("small_run", None, why), Verdict("trial_checks", None, why)]
    if _optimise.block_of(protocol) is not None:
        why = "the study is a search for the best design, which has no grid to run a small case on"
        return [Verdict("small_run", None, why), Verdict("trial_checks", None, why)]
    grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
    paired = _mc_error.paired(protocol, quest_root)
    inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None
    got = await _small.run(executor, python, quest_root, simulate.relative_to(quest_root).as_posix(), grid,
                           deterministic="run_trial" not in entries, timeout_s=_small.budget(timeout_s), env=dict(env),
                           thresholds=thresholds, paired=paired, in_each_run=inside)
    ran = Verdict("small_run", not got.load_error,
                  (f"the simulation could not be loaded: {_one(got.load_error)}" if got.load_error else got.says()),
                  {"seconds": round(got.seconds, 2), "timed_out": _timed_out(got), "settings": len(got.settings)})
    if got.load_error:
        return [ran, Verdict("trial_checks", None, "the small run gave no trials to count")]
    # The failed trials of each setting, judged by the check the study's own trials go through. The calibration against the
    # true values needs more trials than a small run makes (trial_checks.MIN_TRIALS): it is judged on the full run.
    cells = [{"key": s.key, "planned": s.planned, "failed": s.failed} for s in got.settings]
    reasons = {s.key: Counter(dict(s.reasons)) for s in got.settings}
    later = "calibration against the true values is judged on the full run, not the small run"
    verdict = _trial_checks.judge(cells, protocol, thresholds, reasons, repairs_left=False, no_calibration=later)
    declared = any(spec.get("known_value") for spec in _metric_spec.declared(protocol))
    bad = [f for f in verdict.fails if f.fault]
    if verdict.action == _trial_checks.OK:
        says = got.says() + ("; " + later if declared else "")
        return [ran, Verdict("trial_checks", True, _one(says), {"calibration": later})]
    first = bad[0].says() if bad else verdict.say
    return [ran, Verdict("trial_checks", False, _one(f"{len(bad)} of {len(verdict.fails)} setting(s): more than half of the "
                                                    f"trials failed ({first})"),
                         {"failures": verdict.record().get("failures", []), "calibration": later})]


def _equation_run(quest_root: Path, executor: Any, python: Any, log: Any):
    """The runner :func:`core.equation_tests.run_and_judge` is given: ``python -B <argv>`` from a copy of ``code/`` outside the
    quest folder, in the quest's environment (the engine's ``_run_in_env``, without the engine)."""
    outside = _outside.OutsideExecutor(executor, quest_root, log=log)

    async def run(argv: list[str], *, timeout_s: int = 180, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        # A function rewritten within the same second to the same size would be read from its stale compiled copy.
        for cache in (quest_root / "code").glob("**/__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        env = {k: v for k, v in env.items() if k != "FI_CHECKS_DIR"} if env is not None else None
        try:
            ran = await outside.execute([str(python), "-B", *argv], cwd=quest_root, timeout_s=timeout_s, env=env)
        except Exception as exc:  # noqa: BLE001 -- a check that could not start is a failed check, said as such
            return -1, "", repr(exc)
        return ran.returncode, ran.stdout or "", (ran.stderr or "") + ("\n(timed out)" if ran.timed_out else "")

    return run


async def small_run_refusal(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                            env: dict[str, str], timeout_s: int) -> str:
    """Why FI's small run refuses the code under ``quest_root/code`` (it cannot be loaded, or more than half of the trials of a
    setting fail, e.g. for a returned form the run contract does not read), or ``""`` when it runs or cannot be judged. The
    check the evidence gate's repairs go through before they are kept, for a repair that is run on a copy."""
    for verdict in await _small_run_verdicts(quest_root, protocol, executor=executor, python=python, env=env,
                                             timeout_s=timeout_s):
        if verdict.passed is False:
            # the reasons the trials gave, whole (the line ``says`` is cut short): the model repairing is told what the run
            # contract asked for ("key it ..."), not the start of the message
            reasons = [f"{f.get('key') or 'the one setting'}: {r}" for f in verdict.details.get("failures") or []
                       if isinstance(f, dict) and f.get("fault") for r, _n in (f.get("reasons") or [])[:1]]
            return f"{verdict.says}; the reason: {_one(reasons[0], 700)}" if reasons else verdict.says
    return ""


async def _equation_verdict(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                            env: dict[str, str], log: Any) -> Verdict:
    code_dir = quest_root / "code"
    if not isinstance(protocol, dict) or not (code_dir / _split_run.SIMULATE_NAME).is_file() or not _eqt.eligible(protocol):
        return Verdict("equation_tests", None, "the plan's model has no equation with a worked example to test")
    rows = _eqt.example_rows(protocol)
    sources = _equation_sources(code_dir)
    located = _eqt.locate(protocol, sources, None)
    case_list = _eqt.cases(rows, located)
    untested = _eqt.untested_notes(rows, located)
    if not case_list:
        return Verdict("equation_tests", None, "no equation could be tested: " + (untested[0] if untested else "none has an example"),
                       {"not_tested": untested})
    raw_dir = _split_run.raw_root_of(quest_root) / "equation_tests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_env = {**env, _split_run.RAW_DIR_ENV: _split_run.env_value(raw_dir, quest_root)}
    results, rc, _err = await _eqt.run_and_judge(_equation_run(quest_root, executor, python, log), case_list,
                                                 timeout_s=EQUATION_TIMEOUT_S, env=run_env)
    shape = {str(r.get("id")) for r in results if r.get("status") == _eqt.OUTPUTS}
    failed = [r for r in results if r.get("status") not in (_eqt.OK, _eqt.OUTPUTS)]
    tested = [str(r.get("id")) for r in results if str(r.get("id")) not in shape]
    notes = untested + [_eqt.cannot_judge_note(i) for i in sorted(shape)]
    if not failed:
        return Verdict("equation_tests", True, f"the model's {len(tested)} tested equation(s) agree with the plan's worked examples",
                       {"tested": tested, "not_tested": notes})
    parts = []
    for r in failed:
        head = f"equation {r.get('id')} (`{r.get('function')}`)"
        if r.get("status") == _eqt.MISMATCH:
            got = r.get("got")
            shown = ", ".join(f"{k}={_number(v)}" for k, v in got.items()) if isinstance(got, dict) else (
                _number(got) if got is not None else "no usable number")
            parts.append(f"{head} does not compute what the plan says (it gave {shown})")
        else:
            parts.append(f"{head} could not be tested: {_one(r.get('detail') or r.get('status'), 160)}")
    return Verdict("equation_tests", False, _one("; ".join(parts), 400),
                   {"failed": [{"id": r.get("id"), "function": r.get("function"), "status": r.get("status")} for r in failed],
                    "tested": tested, "not_tested": notes})


async def known_answer_values(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                              env: dict[str, str], timeout_s: int, log: Any = None,
                              notes: list[str] | None = None) -> dict[str, dict[str, Any]] | None:
    """What each known-answer check the engine can judge gives on ``<quest_root>/code`` as it is, run by FI the way the gate
    after the run runs them (a search is called with the conditions held fixed): ``{check: {"passed", "value",
    "undecided"}}``, for a caller that compares two versions of the code (the repair of a reader's finding). ``None`` when
    they cannot be run (no trial contract, no check with numbers to judge by). Nothing in the quest's own records changes."""
    simulate = quest_root / "code" / _split_run.SIMULATE_NAME
    oracles = [o for o in _oracle.declared(protocol) if _oracle.limit_of(o)[1] is not None]
    if not oracles or not simulate.is_file() or not _trials.entries(simulate) & {"run_trial", "run_cell"}:
        return None
    protocol = protocol if isinstance(protocol, dict) else {}
    if _optimise.block_of(protocol) is not None:
        oracles = [{**o, "case": _optimise.complete_case(protocol, o["case"])} if isinstance(o.get("case"), dict) else o
                   for o in oracles]
    run_env = {**env, "PYTHONDONTWRITEBYTECODE": "1"}
    unrun: dict[str, str] = {}
    paired = _mc_error.paired(protocol, quest_root)  # read from the quest's own records
    # Measuring must leave nothing in the quest (raw files, the shared cache, files the code writes): the cases run in a
    # throwaway copy of it, deleted after.
    with _outside.observation_copy(quest_root, notes=notes, log=log) as work:
        raw_dir = _split_run.raw_root_of(work) / "finding_replay"
        raw_dir.mkdir(parents=True, exist_ok=True)
        checks, _problems, _timed_out = await _trials.measure_oracles(
            executor, python, work, simulate.relative_to(quest_root).as_posix(), oracles, timeout_s=timeout_s,
            env={**run_env, "FI_ORACLE": "1", _split_run.RAW_DIR_ENV: _split_run.env_value(raw_dir, work)},
            thresholds=protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None,
            case_env=dict(run_env), in_each_run=_in_run.of_study(protocol), mean_names=_mc_error.mean_names(protocol),
            paired=paired, proportion_names=_mc_error.proportion_names(protocol),
            repeated=_mc_error.repeats_trials(protocol), log=log, unrun=unrun, protocol=protocol)
    judged = _oracle.judged(oracles, {"checks": checks, "engine_measured": True})
    out: dict[str, dict[str, Any]] = {}
    for j in judged:
        name = str(j["name"])
        why = unrun.get(name)
        # A case FI stopped at its time limit was not measured: nothing is learned from it either way (undecided), unlike
        # one that ran and gave nothing or a failing answer. Decided by FI's own stop (``TimeLimit``), never by the text a
        # program printed.
        out[name] = {"passed": j["passed_by_engine"], "value": j["value"],
                     "undecided": bool(j.get("undecided")) or isinstance(why, _trials.TimeLimit),
                     **({"not_run": str(why)[:300]} if why else {})}
    return out


async def _oracle_verdict(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                          env: dict[str, str], log: Any, timeout_s: int) -> Verdict:
    simulate = quest_root / "code" / _split_run.SIMULATE_NAME
    oracles = _oracle.declared(protocol)
    if not oracles or not simulate.is_file():
        return Verdict("known_answers", None, "the plan declares no known-answer check to run" if not oracles
                       else f"there is no {simulate.name} to run")
    incomplete = _oracle.unjudgeable(oracles)
    if incomplete:
        return Verdict("known_answers", None, "a known-answer check of the plan has no numbers FI can judge it by, so none was run",
                       {"unjudgeable": len(incomplete)})
    if not _trials.entries(simulate) & {"run_trial", "run_cell"}:
        return Verdict("known_answers", None, "the simulation does not follow the trial contract, so its cases cannot be run here")
    protocol = protocol if isinstance(protocol, dict) else {}
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
    raw_dir = _split_run.raw_root_of(quest_root) / "oracle_check"
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = {**env, _split_run.SEED_ENV: "0", _split_run.INDEX_ENV: "0"}
    try:
        checks, problems, _timed_out_ = await _trials.measure_oracles(
            executor, python, quest_root, simulate.relative_to(quest_root).as_posix(), oracles, timeout_s=timeout_s,
            env={**base, "FI_ORACLE": "1", _split_run.RAW_DIR_ENV: _split_run.env_value(raw_dir, quest_root)},
            thresholds=thresholds, in_each_run=_in_run.of_study(protocol), mean_names=_mc_error.mean_names(protocol),
            paired=_mc_error.paired(protocol, quest_root), proportion_names=_mc_error.proportion_names(protocol),
            repeated=_mc_error.repeats_trials(protocol), log=log, protocol=protocol)
    except Exception as exc:  # noqa: BLE001 -- a run that cannot start is reported, not raised
        return Verdict("known_answers", None, _one(f"the known-answer checks could not be run ({exc!r})"))
    judged = _oracle.judged(oracles, {"checks": checks, "engine_measured": True})
    # Neither the expected value nor the tolerance leaves this function, and a check that passed gives no value either (it
    # is the expected value to within the tolerance).
    seen = [{"name": j["name"], **({} if j["passed_by_engine"] is True else {"value": j["value"]}),
             "passed": j["passed_by_engine"], **({"undecided": True} if j.get("undecided") else {})} for j in judged]
    wrong = [j for j in seen if j["passed"] is False]
    unmeasured = [j for j in seen if j["passed"] is None]
    if not wrong and not unmeasured:
        return Verdict("known_answers", True, f"{len(seen)} known-answer check(s) measured by running the simulation passed",
                       {"checks": seen})
    parts = [f"check {j['name']!r}: measured {_number(j.get('value'))}, outside what the plan allows" for j in wrong]
    parts += [f"check {j['name']!r}: " + ("the trials could not decide it" if j.get("undecided") else "no number could be measured")
              for j in unmeasured]
    if problems and unmeasured:
        parts.append(_one(problems[0], 200))
    undecided_only = not wrong and all(j.get("undecided") for j in unmeasured)
    return Verdict("known_answers", None if undecided_only else False, _one("; ".join(parts), 400), {"checks": seen})


def _unread_verdict(quest_root: Path, protocol: dict[str, Any] | None, design: dict[str, Any]) -> Verdict:
    simulate = quest_root / "code" / _split_run.SIMULATE_NAME
    grid = protocol.get("grid") if isinstance(protocol, dict) and isinstance(protocol.get("grid"), dict) else {}
    if not grid or not simulate.is_file():
        return Verdict("unread_settings", None, "the plan has no grid, or there is no simulation to read")
    files = {rel: text for rel, text in _files(simulate.parent).items() if rel not in ("run.py", _trials.SUBMIT_NAME)}
    misread, compared, unread, why = _run_checks.unread_findings(files, protocol, design, quest_root / ".fi")
    if misread:
        return Verdict("unread_settings", None, "the code reads a name the plan does not give; the check of the settings (grid_keys) owns that")
    if compared:
        said = "; ".join(f"`{axis}` ({reason})" for axis, reason in compared.items())
        return Verdict("unread_settings", False, _one(f"nothing in the code reads {said}, a setting the study compares"),
                       {"settings": sorted(compared)})
    if unread:
        return Verdict("unread_settings", True,
                       _one("FI's reading finds no use of " + ", ".join(f"`{k}`" for k in unread) + "; every setting is kept and run"),
                       {"unread_not_compared": unread})
    if why:
        return Verdict("unread_settings", None, _one(f"FI could not tell whether the code reads every setting of the grid ({why})"))
    return Verdict("unread_settings", True, "the code reads every setting of the grid")


def _figure_verdict(code_dir: Path) -> Verdict:
    found = _figdata.typed_series_in(code_dir)
    if not found:
        return Verdict("figure_data", True, "no figure is drawn from numbers typed into the code")
    first = [f.says() for series in found.values() for f in series][:2]
    return Verdict("figure_data", False, _one("a figure is drawn from numbers typed into the code (" + "; ".join(first) + ")"),
                   {"scripts": {p.name: [{"line": f.line, "call": f.call, "count": f.count} for f in s] for p, s in found.items()}})


#: What the dry run of the figure bodies prints as its last line, then the outcomes of the bodies as JSON.
_DRY_RUN_MARK = "FI_DRY_RUN: "
#: The most settings of the stand-in table (a larger grid is thinned to its first, middle and last value on each axis).
_DRY_RUN_CELLS = 40
#: Run in the quest's interpreter from a copy of the quest (``sys.argv[1]``: the settings as JSON). It uses only what ``code/``
#: carries (``derive.py``, ``fi_frame.py``, ``fi_table.py``, ``fi_figures.py``), never FI's ``core``.
_DRY_RUN_SCRIPT = r'''
import json, os, shutil, sys, tempfile, warnings
mark = %r
sys.path.insert(0, os.path.join(os.getcwd(), "code"))
try:
    import derive
    import fi_figures
    import fi_table
except Exception as err:
    print(mark + json.dumps({"load": "%%s: %%s" %% (type(err).__name__, err)}))
    sys.exit(0)
ids = list(getattr(derive, "RESULTS", ())) + list(getattr(derive, "AFTERWARDS", ()))
kinds = getattr(derive, "KINDS", {})
compared = getattr(derive, "COMPARED", {})


def metrics(i):
    out = {}
    for r, name in enumerate(ids):
        if kinds.get(name) == "proportion":
            values = [1.0 if (t + i + r) %% 3 else 0.0 for t in range(6)]
        else:
            values = [1.0 + 0.5 * r + 0.1 * i + 0.01 * t for t in range(6)]
        out[name] = {"values": values, "trials": list(range(6))}
    return out


cells = []
for i, setting in enumerate(json.loads(sys.argv[1])):
    cells.append({"cell": setting, "key": fi_table.cell_key(setting), "metrics": metrics(i)})
    for axis, values in compared.items():
        for value in values if isinstance(values, (list, tuple)) else ():
            there = dict(setting, **{axis: value})
            cells.append({"cell": there, "key": fi_table.cell_key(there), "level": {axis: value}, "metrics": metrics(i + 1)})
first = derive.Table(cells, ids)
parts = {}
for name, hold in getattr(derive, "PARTS", {}).items():
    try:
        parts[name] = first.at(**hold)
    except Exception:
        parts[name] = []
table = derive.Table(cells, ids, {}, parts, getattr(derive, "KNOWN", {}))
folder = tempfile.mkdtemp()
done = []
try:
    for fid, spec in getattr(derive, "FIGURES", {}).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = fi_figures.run_figures([dict(spec, id=fid)], derive, table, folder)[0]
        done.append({"id": str(fid), "status": got.get("status"), "why": got.get("why") or ""})
finally:
    shutil.rmtree(folder, ignore_errors=True)
print(mark + json.dumps({"figures": done}))
''' % (_DRY_RUN_MARK,)


def _dry_run_settings(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The settings of the stand-in table: the plan's grid with the values it declares (no value made up), thinned to
    :data:`_DRY_RUN_CELLS` settings when it is larger; one empty setting without a grid."""
    grid = protocol.get("grid") if isinstance(protocol, dict) and isinstance(protocol.get("grid"), dict) else {}
    axes = {str(k): list(v) for k, v in grid.items() if isinstance(v, (list, tuple)) and v}
    while axes and len(list(itertools.islice(itertools.product(*axes.values()), _DRY_RUN_CELLS + 1))) > _DRY_RUN_CELLS:
        widest = max(axes, key=lambda k: len(axes[k]))
        if len(axes[widest]) <= 3:
            break
        axes[widest] = [axes[widest][0], axes[widest][len(axes[widest]) // 2], axes[widest][-1]]
    if not axes:
        return [{}]
    return [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())][:_DRY_RUN_CELLS]


async def _figure_dry_run_verdict(quest_root: Path, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                                  env: dict[str, str], log: Any, timeout_s: int) -> Verdict:
    """Call each ``figure_<id>`` of ``code/derive.py`` on a stand-in table (finite made-up values for the declared results, the
    plan's own settings), before the long run. A body that stops (an id the plan does not declare, a setting that is not
    there) is a warning, never a fault: the stand-in data is not the study's, so this check never stops a run."""
    derive = quest_root / "code" / "derive.py"
    try:
        text = derive.read_text(encoding="utf-8")
    except OSError:
        return Verdict("figure_dry_run", True, "there is no analysis with figures to try")
    if not re.search(r"^def figure_\w+", text, re.MULTILINE):
        return Verdict("figure_dry_run", True, "the analysis draws no figure of its own to try")
    began = time.monotonic()
    run = _equation_run(quest_root, executor, python, log)
    rc, out, err = await run(["-c", _DRY_RUN_SCRIPT, json.dumps(_dry_run_settings(protocol), default=str)],
                             timeout_s=timeout_s, env=env)
    seconds = round(time.monotonic() - began, 1)
    lines = [ln for ln in out.splitlines() if ln.startswith(_DRY_RUN_MARK)]
    try:
        record = json.loads(lines[-1][len(_DRY_RUN_MARK):])
    except (IndexError, ValueError):
        return Verdict("figure_dry_run", None, _one("warning: the figure functions could not be tried on stand-in data ("
                                                     + (err.strip().splitlines()[-1] if err.strip() else f"exit {rc}") + ")"),
                       {"seconds": seconds})
    if record.get("load"):
        return Verdict("figure_dry_run", None, _one(f"warning: the figure functions could not be tried on stand-in data "
                                                     f"(derive.py did not load: {record['load']})"), {"seconds": seconds})
    figures = record.get("figures") or []
    bad = [f for f in figures if f.get("status") not in ("ok", "not_written")]
    if not bad:
        return Verdict("figure_dry_run", True, f"{len(figures)} figure function(s) ran on stand-in data", {"seconds": seconds})
    first = "; ".join(str(f.get("why") or f"figure_{f.get('id')} gave {f.get('status')}") for f in bad[:2])
    return Verdict("figure_dry_run", None, _one(f"warning: {len(bad)} of {len(figures)} figure function(s) stopped on stand-in data: {first}", 400),
                   {"seconds": seconds, "figures": [{"id": f.get("id"), "status": f.get("status")} for f in bad]})


# ---- the entry point --------------------------------------------------------------------------------------------------


async def check_code(quest_root: Path | str, protocol: dict[str, Any] | None, *, executor: Any, python: Any,
                     env: dict[str, str] | None = None, log: Any = None, design: dict[str, Any] | None = None,
                     timeout_s: int | None = None) -> list[Verdict]:
    """Judge ``<quest_root>/code`` with FI's own checks (see the module docstring), in the order of the gate pass. No model is
    asked and nothing in ``code/`` is changed. ``protocol``: the plan's protocol (the frozen one, or the draft). ``executor``
    and ``python``: how the quest's code is run (:mod:`core.execution`). ``env``: the environment of those runs (default: FI's
    own, without secrets). ``design`` and ``timeout_s`` (default :data:`DEFAULT_TIMEOUT_S`; the small run is given
    :func:`core.small_run.budget` of it) are optional. A check that cannot be made is a verdict with ``passed=None``."""
    quest_root = Path(quest_root)
    code_dir = quest_root / "code"
    log = log if log is not None else _LOG
    limit = int(timeout_s or DEFAULT_TIMEOUT_S)
    run_env = dict(env) if env is not None else scrub_secrets()
    try:
        plan_text = (quest_root / "plan.md").read_text(encoding="utf-8")
    except OSError:
        plan_text = ""
    # The text the plan's numbers are read from (typed_constants): the plan as written, after any text the caller gives.
    design = {**(design or {}), "_text": str((design or {}).get("_text") or "") + plan_text}
    files = _files(code_dir)
    out: list[Verdict] = []

    def guarded(name: str, make: Any) -> None:
        try:
            got = make()
        except Exception as exc:  # noqa: BLE001 -- a check that cannot be made flags nothing
            got = Verdict(name, None, _one(f"the check could not be made ({exc!r})"))
        out.extend(got if isinstance(got, list) else [got])

    async def guarded_async(names: list[str], make: Any) -> None:
        try:
            got = await make()
        except Exception as exc:  # noqa: BLE001
            got = [Verdict(n, None, _one(f"the check could not be made ({exc!r})")) for n in names]
        out.extend(got if isinstance(got, list) else [got])

    guarded("typed_results", lambda: _typed_results(code_dir, protocol, design))
    guarded("grid_keys", lambda: _static_verdicts(files, protocol, design))
    await guarded_async(["small_run", "trial_checks"],
                        lambda: _small_run_verdicts(quest_root, protocol, executor=executor, python=python, env=run_env,
                                                    timeout_s=limit))
    await guarded_async(["equation_tests"], lambda: _equation_verdict(quest_root, protocol, executor=executor, python=python,
                                                                       env=run_env, log=log))
    await guarded_async(["known_answers"], lambda: _oracle_verdict(quest_root, protocol, executor=executor, python=python,
                                                                    env=run_env, log=log, timeout_s=limit))
    guarded("unread_settings", lambda: _unread_verdict(quest_root, protocol, design))
    guarded("figure_data", lambda: _figure_verdict(code_dir))
    await guarded_async(["figure_dry_run"], lambda: _figure_dry_run_verdict(quest_root, protocol, executor=executor, python=python,
                                                                             env=run_env, log=log, timeout_s=limit))
    for v in out:
        cost = f" ({v.details['seconds']} s)" if v.check == "figure_dry_run" and "seconds" in v.details else ""
        log.info("[code check] %s: %s - %s%s", v.check, {True: "passed", False: "FAILED", None: "not judged"}[v.passed], v.says, cost)
    return out
