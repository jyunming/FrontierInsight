"""Who is laid a failure of the analysis FI writes (core/analysis_frame.py), measured on planted cases: honest studies (a run
that must go on with no repair and no stop), faults planted in one function of the study's derive.py (the repair must go to
that function, never a stop), and faults of FI's own (a plain stop, never a repair). Each case runs FI's real experiment.py on
a record of trials, and then the engine's own route of the outcome (``Engine._fis_analysis_route``) with what the engine reads
of the run. The goal: no false stop, every planted fault caught, every fault of FI's stopped. Made-up topic: a colony's growth
under a rate and a dose. No real model is called."""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from core import analysis_frame as af
from core import numeric_warnings, skeleton, trial_runner
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _cell(rate: float, dose: int, growth: list[float]) -> dict[str, Any]:
    cell = {"rate": rate, "dose": dose}
    return {"cell": cell, "key": trial_runner.cell_key(cell), "planned": len(growth), "ok": len(growth), "failed": 0,
            "metrics": {"growth": {"values": growth, "trials": list(range(len(growth)))}}}


RECORD = {"schema": "fi.trials/v1", "cells": [_cell(0.1, 1, [1.0, 1.5, 2.5]), _cell(0.1, 2, [3.0, 3.5, 4.5]),
                                               _cell(0.2, 1, [1.25, 1.75, 2.25]), _cell(0.2, 2, [5.0, 6.0, 5.5])]}
PEAK = {"id": "peak", "inputs": ["growth"], "role": "result"}
HALF = {"id": "half", "inputs": ["peak"], "role": "result", "formula": "peak / 2"}
SPREAD = {"id": "spread", "inputs": ["growth"], "role": "diagnostic"}
FIGURE = {"id": "fig1", "file": "growth.png", "shows": ["growth"], "x": "rate", "series": "dose"}
GOOD_PEAK = '''
def derive_peak(table):
    return max(table.mean("growth", k) for k in table.keys())
'''
GOOD_FIGURE = '''
def figure_fig1(fig, table):
    for k in table.keys():
        fig.axes[0].plot([1, 2], [table.mean("growth", k)] * 2)
'''


def _design(derived: list[dict[str, Any]], figures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"hypothesis": "growth rises with the rate", "protocol": {
        "form": "contract", "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
        "metrics": [{"id": "growth", "kind": "mean", "role": "result"}], "derived": derived, "figures": figures or []}}


@dataclass
class Case:
    name: str
    kind: str  # "honest" | "body" (a fault planted in one function) | "fi" (a fault of FI's own)
    derived: list[dict[str, Any]]
    bodies: str
    want: str  # "go on" | the function the repair must go to | "stop"
    figures: list[dict[str, Any]] | None = None
    timeout: float | None = None  # the analysis is stopped for its time after this many seconds
    plant: str = ""  # "no_calculator" | "empty_record" | "late_formula": FI's own fault, planted beside the study's code


CASES = [
    # honest: the run goes on, with no repair and no stop
    Case("a clean study", "honest", [PEAK, SPREAD], GOOD_PEAK + '''
def derive_spread(table):
    return max(table.mean("growth", k) for k in table.keys()) - min(table.mean("growth", k) for k in table.keys())
''', "go on"),
    Case("a value said not to exist, with a formula over it", "honest", [PEAK, HALF], '''
def derive_peak(table):
    return table.missing("the growth never peaks inside the range studied")
''', "go on"),
    Case("a formula listed before what it names", "honest", [HALF, PEAK], GOOD_PEAK, "go on"),
    Case("a value missing at some settings, with the reason", "honest", [PEAK], '''
def derive_peak(table):
    return {k: (table.mean("growth", k) if table.setting(k)["dose"] == 2 else table.missing("no peak at the low dose"))
            for k in table.keys()}
''', "go on"),
    Case("a diagnostic that fails", "honest", [PEAK, SPREAD], GOOD_PEAK + '''
def derive_spread(table):
    raise ValueError("not defined for this study")
''', "go on"),
    Case("a figure drawn from the table", "honest", [PEAK], GOOD_PEAK + GOOD_FIGURE, "go on", figures=[FIGURE]),
    Case("a warning that is not about the numbers", "honest", [PEAK], '''
import warnings


def derive_peak(table):
    warnings.warn("this helper is old", DeprecationWarning)
    return max(table.mean("growth", k) for k in table.keys())
''', "go on"),
    # a fault planted in one function: the repair goes to it
    Case("a setting passed where its key goes (FI's table raises)", "body", [PEAK], '''
def derive_peak(table):
    return max(table.mean("growth", table.setting(k)) for k in table.keys())
''', "derive_peak"),
    Case("a function that raises, with a formula over it", "body", [PEAK, HALF], '''
def derive_peak(table):
    return 1 / 0
''', "derive_peak"),
    Case("a function that raises", "body", [PEAK], '''
def derive_peak(table):
    return 1 / 0
''', "derive_peak"),
    Case("a function too slow for the time allowed", "body", [PEAK], '''
import time


def derive_peak(table):
    time.sleep(60)
    return 1.0
''', "derive_peak", timeout=6),
    Case("a function that returns a word", "body", [PEAK], '''
def derive_peak(table):
    return "high"
''', "derive_peak"),
    Case("a figure function that names nothing defined", "body", [PEAK], GOOD_PEAK + '''
def figure_fig1(fig, table):
    fig.axes[0].plot(rates, [1, 2])
''', "figure_fig1", figures=[FIGURE]),
    Case("a numeric warning raised in a function", "body", [PEAK], '''
import math
import warnings


def derive_peak(table):
    warnings.warn("overflow encountered in exp", RuntimeWarning)
    return max(table.mean("growth", k) for k in table.keys())
''', "derive_peak"),
    Case("a function that ends the process", "body", [PEAK], '''
import os


def derive_peak(table):
    os._exit(3)
''', "derive_peak"),
    # a fault of FI's own: a plain stop
    Case("FI's formula calculator is missing", "fi", [PEAK, HALF], GOOD_PEAK, "stop", plant="no_calculator"),
    Case("FI's record of the trials is empty", "fi", [PEAK], GOOD_PEAK, "stop", plant="empty_record"),
    Case("a formula that names nothing declared reaches the run", "fi", [PEAK], GOOD_PEAK, "stop", plant="late_formula"),
]


def _engine(root: Path) -> Engine:
    cfg = Config(topic="a colony's growth", title="colony", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3, ideate_reflect=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True, code_skeleton=True),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=root / "out"))
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    return engine


def _run_case(case: Case, root: Path) -> dict[str, Any]:
    """Run FI's real experiment.py for ``case`` and route its outcome as the engine does: ``{"got": "go on" | <function> |
    "simulation" | "stop", "rc", "why"}``."""
    engine = _engine(root)
    design = _design(case.derived, case.figures)
    quest, code = engine.quest_root, engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(skeleton.frame_source(design, stochastic=True), encoding="utf-8")
    af.ensure_files(code, design)
    bodies = textwrap.dedent(case.bodies)
    if case.plant == "late_formula":  # a derive.py written before the plan was checked for it
        bodies += ('\nDERIVED["odd"] = {"by": "formula", "formula": "growth * nothing_here", "inputs": ["growth"], '
                   '"role": "result"}\n')
    with (code / "derive.py").open("a", encoding="utf-8") as f:
        f.write("\n\n" + bodies)
    if case.plant == "no_calculator":
        shutil.rmtree(code / af.CORE_DIR)
    (quest / "raw").mkdir(parents=True, exist_ok=True)
    (quest / "raw" / "trials.json").write_text("{}" if case.plant == "empty_record" else json.dumps(RECORD), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS")}
    proc = subprocess.Popen([sys.executable, "-B", "experiment.py"], cwd=code, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=case.timeout or 300)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    rc = proc.returncode if not timed_out else (proc.returncode or -9)
    lines = [ln for ln in stdout.splitlines() if ln.startswith("RESULT_JSON: ")]
    result_json = json.loads(lines[-1][len("RESULT_JSON: "):]) if lines and not timed_out else {}
    findings = numeric_warnings.scan(stderr) if rc == 0 else []
    if rc == 0 and not findings:
        return {"got": "go on", "rc": rc, "why": ""}
    exec_result: dict[str, Any] = {
        "returncode": rc, "timed_out": timed_out, "stdout_tail": "\n".join(ln for ln in stdout.splitlines() if ln not in lines),
        "stderr_tail": engine._stderr_for_models(stderr)[-2000:],
        "failed_script": af.failed_script(code / "experiment.py", rc) if rc != 0 else None}
    if rc != 0 and hasattr(af, "running_step"):
        exec_result["analysis_step"] = af.running_step(stderr)
    state = {"design": design, "result_json": result_json}
    kwargs = {"findings": findings, "implausible": []} if "findings" in inspect.signature(engine._fis_analysis_route).parameters else {}
    routed = engine._fis_analysis_route(state, exec_result, rc, [], **kwargs)
    if "stop" in routed:
        return {"got": "stop", "rc": rc, "why": str((routed["stop"].get("exec_result") or {}).get("cannot_repair") or "")}
    return {"got": routed["body"] if routed["body"] is not None else "simulation", "rc": rc, "why": ""}


def bench(root: Path) -> dict[str, Any]:
    """Every case run once: each outcome, and the counts the project goal is stated in."""
    rows = []
    for i, case in enumerate(CASES):
        out = _run_case(case, root / f"c{i}")
        rows.append({"case": case.name, "kind": case.kind, "want": case.want, **out})
    stopped = [r for r in rows if r["got"] == "stop"]
    return {"rows": rows,
            "body_faults": sum(r["kind"] == "body" for r in rows),
            "caught": sum(r["kind"] == "body" and r["got"] == r["want"] for r in rows),
            "false_stops": sum(r["kind"] != "fi" for r in stopped),
            "not_fi_cases": sum(r["kind"] != "fi" for r in rows),
            "fi_faults": sum(r["kind"] == "fi" for r in rows),
            "fi_stopped": sum(r["kind"] == "fi" for r in stopped),
            "honest_left_alone": sum(r["kind"] == "honest" and r["got"] == "go on" for r in rows),
            "honest": sum(r["kind"] == "honest" for r in rows)}


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_each_planted_case_goes_where_it_belongs(case: Case, tmp_path: Path) -> None:
    out = _run_case(case, tmp_path)
    assert out["got"] == case.want, out
    if case.kind == "fi":
        assert "FI's" in out["why"]  # said plainly as FI's, never sent to the model


def test_the_bench_has_no_false_stop(tmp_path: Path) -> None:
    got = bench(tmp_path)
    assert got["honest"] >= 6 and got["body_faults"] >= 6 and got["fi_faults"] >= 2
    assert got["false_stops"] == 0 and got["caught"] == got["body_faults"] and got["fi_stopped"] == got["fi_faults"], got["rows"]
    assert got["honest_left_alone"] == got["honest"]


if __name__ == "__main__":  # python tests/test_analysis_false_stops.py <folder>: the table of every case, for a report
    import tempfile

    where = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    result = bench(where)
    for r in result["rows"]:
        print(f"{r['kind']:6} {r['case'][:58]:58} want={r['want']:12} got={r['got']:12} rc={r['rc']}")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
