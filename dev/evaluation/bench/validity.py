"""Which planted errors count: the valid-error filter.

A planted change that does not change the answer is an equivalent mutant (about a quarter of model-written mutants
are), and a run that "let it through" let nothing wrong through, so it stays out of the false-pass rate. Before a planted
copy is run, its code is run offline through FI's own trial harness (core/trial_runner.py::run_trials, the same harness a
quest's run uses) and compared with the clean copy's:

* a change to the simulation (``simulate.py`` or its model package): the answer's metric at the answer's setting, from
  FI's own record of those trials. Valid when the planted code's value is outside the answer's tolerance while the clean
  code's is inside it (a clean value already outside means the recording itself is wrong: nothing can be judged).
* a change to the analysis script (``experiment.py``, S1): the analysis is run on the clean run's own trial record, both
  versions. Valid when what it prints (its ``RESULT_JSON``) differs: the answer FI recorded does not move, the result the
  paper is written from does.

The result is written to ``bench/validity.json`` of the planted run; the scorer reads it. A paper or answer plant (R1,
R2) is valid by construction, and L1 when the paper ends up citing the planted paper (score.py).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from core import frozen_protocol as _frozen
from core import trial_runner as _trials
from core.execution import VenvExecutor

from . import answers as _answers
from . import runner as _runner

VALIDITY_FILE = "validity.json"


def _simulation_file(code: Path) -> Path | None:
    for path in [code / "simulate.py", *sorted(code.rglob("*.py"))]:
        if path.is_file() and _trials.entries(path) & {"run_trial", "run_cell"}:
            return path
    return None


def _full_cell(protocol: dict[str, Any], wanted: dict[str, Any]) -> dict[str, Any] | None:
    """The protocol's one setting that is the answer's (the protocol may have more axes than the answer names)."""
    found = [c for c in _trials.cells(dict(protocol.get("grid") or {})) if _answers.cell_matches(c, wanted)]
    return found[0] if len(found) == 1 else None


async def simulation_answer(quest_root: Path, answer: dict[str, Any], *, python: str, timeout_s: int = 600,
                            runs: int | None = None) -> tuple[float | None, str]:
    """The answer's metric when FI's harness runs this quest's simulation on the answer's setting."""
    frozen = _frozen.load(quest_root) or {}
    protocol = frozen.get("protocol") if isinstance(frozen.get("protocol"), dict) else None
    if protocol is None:
        return None, "the quest has no frozen protocol"
    cell = _full_cell(protocol, answer["cell"])
    if cell is None:
        return None, f"the protocol's grid has no single setting matching {answer['cell']}"
    with tempfile.TemporaryDirectory(prefix="fi_bench_") as tmp:
        root = Path(tmp)
        shutil.copytree(Path(quest_root) / "code", root / "code", ignore=shutil.ignore_patterns(".git", "__pycache__"))
        sim = _simulation_file(root / "code")
        if sim is None:
            return None, "no simulation file defines run_trial or run_cell"
        deterministic = "run_trial" not in _trials.entries(sim)
        env = {**os.environ, "FI_REPLICATE_SEED": "0"}
        run = await _trials.run_trials(
            VenvExecutor(), python, root, sim.relative_to(root).as_posix(), {k: [v] for k, v in cell.items()},
            runs_per_setting=int(runs or protocol.get("runs_per_setting") or 1), base_seed=0,
            deterministic=deterministic, timeout_s=timeout_s, env=env,
            thresholds=dict(protocol.get("thresholds") or {}),
        )
        values = [float(r["values"][answer["metric"]]) for c in run.cells for r in c.rows
                  if r.get("status") == "ok" and answer["metric"] in (r.get("values") or {})]
        if not values:
            reason = next((c.load_error or r.get("reason") for c in run.cells for r in c.rows if r.get("status") != "ok"),
                          "") if run.cells else ""
            return None, f"the simulation returned no {answer['metric']!r} ({reason or 'no trial ran'})"
        if answer["statistic"] == "value" and len(values) == 1:
            return values[0], ""
        return sum(values) / len(values), ""


async def analysis_output(quest_root: Path, trials_json: Path, *, python: str, timeout_s: int = 300
                          ) -> tuple[dict[str, Any] | None, str]:
    """What this quest's analysis script prints (its ``RESULT_JSON``) when run on ``trials_json``."""
    with tempfile.TemporaryDirectory(prefix="fi_bench_") as tmp:
        root = Path(tmp)
        shutil.copytree(Path(quest_root) / "code", root / "code", ignore=shutil.ignore_patterns(".git", "__pycache__"))
        script = root / "code" / "experiment.py"
        if not script.is_file():
            return None, "no analysis script (code/experiment.py)"
        result = await VenvExecutor().execute([python, str(script)], cwd=root, timeout_s=timeout_s,
                                              env={**os.environ, _trials.RESULTS_ENV: str(trials_json)})
        found = _trials._last_result_json(result.stdout)
        if found is None:
            return None, f"the analysis printed no RESULT_JSON (exit code {result.returncode}): {result.stderr[-300:]}"
        return found, ""


async def check(clean_run: Path, planted_run: Path, task: dict[str, Any], *, python: str | None = None,
                runs: int | None = None) -> dict[str, Any]:
    """Whether the planted copy's change is a valid error; written to its ``bench/validity.json``."""
    from . import plant as _plant

    python = python or sys.executable
    clean_root, planted_root = _runner.quest_root(clean_run), _runner.quest_root(planted_run)
    _out, bench = _runner.paths(planted_run)
    record = _plant.read_record(bench) or {}
    edits = [e.get("file", "") for e in record.get("edits") or []]
    out: dict[str, Any] = {"error": record.get("error"), "files": edits}
    if not edits or not all(f.startswith("code/") for f in edits):
        out.update(valid=None, why="not a change to the code: judged by its own rule (score.py)")
    elif all(Path(f).name == "experiment.py" for f in edits):
        trials = clean_root / _trials.RAW_DIRNAME / _trials.SUMMARY_NAME
        a, why_a = await analysis_output(clean_root, trials, python=python)
        b, why_b = await analysis_output(planted_root, trials, python=python)
        if a is None or b is None:
            out.update(valid=None, why=why_a or why_b)
        else:
            same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            out.update(valid=not same, why="the analysis prints the same result on FI's trials (an equivalent change)"
                       if same else "the analysis prints a different result on FI's trials")
    else:
        rows = []
        for a in task.get("answers") or []:
            clean, why_c = await simulation_answer(clean_root, a, python=python, runs=runs)
            planted, why_p = await simulation_answer(planted_root, a, python=python, runs=runs)
            rows.append({"metric": a["metric"], "cell": a["cell"], "expected": a["expected"], "clean": clean,
                         "planted": planted, "why": why_c or why_p,
                         "clean_ok": None if clean is None else _answers.within(clean, a),
                         "planted_ok": None if planted is None else _answers.within(planted, a)})
        out["answers"] = rows
        if any(r["clean_ok"] is None or r["planted_ok"] is None for r in rows) or not rows:
            out.update(valid=None, why="; ".join(r["why"] for r in rows if r["why"]) or "no answer to judge")
        elif not all(r["clean_ok"] for r in rows):
            out.update(valid=None, why="the clean code's own answer is outside the tolerance: the recording is wrong")
        else:
            moved = [r for r in rows if not r["planted_ok"]]
            out.update(valid=bool(moved), why=(f"the answer moved beyond its tolerance ({moved[0]['metric']}: "
                                               f"{moved[0]['planted']:.4g}, expected {moved[0]['expected']})")
                       if moved else "the answer stayed inside its tolerance (an equivalent change)")
    bench.mkdir(parents=True, exist_ok=True)
    (bench / VALIDITY_FILE).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out
