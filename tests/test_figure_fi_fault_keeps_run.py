"""A fault of FI's own while drawing one figure never stops the study: every result was produced and only that figure is
missing, so the run exits as it would with no such fault, the figure is recorded and disclosed as not drawn by FI's fault, and
nothing is sent to the model. A formula FI could not compute still stops the run. No real model is called."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core import analysis_frame as frame
from tests.test_analysis_frame import NO_WINDOW, _design, _quest
from tests.test_figure_failure_keeps_results import _DRAWS, _engine, _record

_PLANTED = "FI's own drawing stopped on purpose"


def _run_with_fault(quest: Path) -> dict:
    script = ("import json, os, sys\nsys.path.insert(0, '.')\nimport experiment, fi_figures, fi_table\n"
              "def broken(*a, **k):\n    raise RuntimeError(%r)\n" % _PLANTED +
              "fi_figures._new_figure = broken\n"
              "quest = os.path.dirname(os.path.abspath('.'))\n"
              "out, records, code, doc = experiment.run(fi_table.record_path(quest), quest)\n"
              "print('GOT ' + json.dumps({'code': code, 'out': out, 'faults': doc['fi_faults'], 'figs': doc['figures']}))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS", "FI_FIGURE_FALLBACK")}
    done = subprocess.run([sys.executable, "-c", script], cwd=quest / "code", env=env, capture_output=True, text=True, timeout=300,
                          creationflags=NO_WINDOW)
    return json.loads(next(ln for ln in done.stdout.splitlines() if ln.startswith("GOT "))[4:])


def test_a_figure_fault_of_fi_does_not_stop_the_run(tmp_path: Path) -> None:
    got = _run_with_fault(_quest(tmp_path, _design(), _DRAWS))
    fig = next(o for o in got["figs"] if o["id"] == "fig1")
    assert fig["status"] == "fi_fault" and _PLANTED in fig["said"]  # recorded as not drawn, with FI's reason
    assert any(_PLANTED in f for f in got["faults"])
    assert got["code"] == 0  # the results were produced; the exit code is the quantities' alone


def test_only_figures_failed_ignores_a_figure_fault_of_fi_but_not_another_fault() -> None:
    said = "FI could not make the figure `fig3` (RuntimeError: x)."
    rec = {"schema": frame.DERIVED_SCHEMA, "derived": {}, "fi_faults": [said],
           "figures": [{"id": "fig2", "status": "failed", "file": "a.png"},
                       {"id": "fig3", "status": "fi_fault", "file": "b.png", "said": said}]}
    assert frame.only_figures_failed(rec, 1)
    assert frame.run_produced(1, True, rec, True)
    assert not frame.only_figures_failed({**rec, "fi_faults": [said, "FI's figure step stopped with X"]}, 1)  # another fault stands


def test_a_formula_fault_still_exits_2(tmp_path: Path) -> None:  # the known positive
    from tests.test_analysis_frame import _derived, _run

    late = 'DERIVED["odd"] = {"by": "formula", "formula": "growth * nothing_here", "inputs": ["growth"], "role": "result"}\n'
    code, _printed, record, _err = _run(_quest(tmp_path, _design([_derived("odd", formula="growth * nothing_here")]), late))
    assert code == 2 and record["fi_faults"]


def test_the_figure_not_drawn_by_fis_fault_is_disclosed_and_stops_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path, phased_on=False)
    path = eng.quest_root / frame.DERIVED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": frame.DERIVED_SCHEMA, "derived": {}, "fi_faults": ["x"], "figures": [
        {"id": "fig3", "status": "fi_fault", "file": "b.png", "why": "FI stopped while drawing the figure `fig3`."}]}), encoding="utf-8")
    eng._figure_fallbacks({})
    problems = eng._cannot_repair_problems()
    assert len(problems) == 1 and "fig3" in problems[0]["text"] and "FI's own" in problems[0]["text"]
    assert eng._cannot_repair_stop() is None
