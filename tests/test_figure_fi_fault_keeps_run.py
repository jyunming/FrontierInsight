"""A fault of FI's own while drawing one figure never stops the study: every result was produced and only that figure is
missing, so the run exits as it would with no such fault, the figure is recorded and disclosed as not drawn by FI's fault, and
nothing is sent to the model. A formula FI could not compute still stops the run. No real model is called."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core import analysis_frame as frame
from tests.test_analysis_frame import NO_WINDOW, _design, _quest
from tests.test_figure_failure_keeps_results import _DRAWS, _engine

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

    from tests.test_analysis_frame import _late_in_frame

    quest = _quest(tmp_path, _design([_derived("odd", formula="growth * nothing_here")]))
    _late_in_frame(quest, '"odd": {"by": "formula", "formula": "growth * nothing_here", "inputs": ["growth"], "role": "result"}')
    code, _printed, record, _err = _run(quest)
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


# ---- fix round 1: a crash of the whole figure step is a figure fault too ---------------------------------------------------------
def _run_with_step_crash(quest: Path) -> tuple[int, dict, str]:
    script = ("import sys\nsys.path.insert(0, '.')\nimport fi_figures\n"
              "def broken(*a, **k):\n    raise RuntimeError('the figure step crashed')\n"
              "fi_figures.run_figures = broken\nimport runpy\nrunpy.run_path('experiment.py', run_name='__main__')\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS", "FI_FIGURE_FALLBACK")}
    done = subprocess.run([sys.executable, "-c", script], cwd=quest / "code", env=env, capture_output=True, text=True, timeout=300,
                          creationflags=NO_WINDOW)
    record = json.loads((quest / frame.DERIVED_PATH).read_text(encoding="utf-8"))
    return done.returncode, record, done.stdout


def test_a_crash_of_the_whole_figure_step_does_not_stop_the_run(tmp_path: Path) -> None:
    code, record, out = _run_with_step_crash(_quest(tmp_path, _design(), _DRAWS))
    assert code == 0 and "RESULT_JSON:" in out
    fig = next(o for o in record["figures"] if o["id"] == "fig1")  # each planned figure is recorded as not drawn, FI's fault
    assert fig["status"] == "fi_fault" and "the figure step crashed" in fig["why"]
    assert "FI's own drawing step failed" in fig["said"] and "FI's fault, not the study's" in fig["said"]
    assert frame.run_produced(0, True, record, True)


def test_a_stop_of_fis_own_lines_outside_the_figure_step_still_exits_2(tmp_path: Path) -> None:  # the known positive
    from tests.test_analysis_frame import _run

    quest = _quest(tmp_path, _design(), _DRAWS)
    shutil.rmtree(quest / "code" / frame.CORE_DIR)  # FI's formula calculator is gone: not the figure step
    code, printed, _record, err = _run(quest)
    assert code == 2 and printed is None and "formula calculator" in err and "not the study's" in err


# ---- fix round 1: the same rule where a run's bodies are kept -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_figure_fault_of_fi_does_not_stop_a_run_from_being_kept_but_another_fault_does(tmp_path: Path) -> None:
    from core import kept_bodies as kb
    from tests.test_kept_bodies import DESIGN, _first, _ran

    said = "Figure `over_dose` was not drawn: FI's own drawing step failed (x); this is FI's fault, not the study's."

    async def kept(sub: str, faults: list[str]) -> list[str]:
        engine, _state = await _first(tmp_path / sub)
        _ran(engine, {"derive_peak": "ok", "figure_over_rate": "ok"})
        path = engine.quest_root / frame.DERIVED_PATH
        record = json.loads(path.read_text(encoding="utf-8"))
        record["figures"] = [{"id": "over_dose", "status": "fi_fault", "said": said, "why": "x"}]
        record["fi_faults"] = faults
        path.write_text(json.dumps(record), encoding="utf-8")
        # the run writes its record and then its receipt (core/kept_bodies.py record_run), so the receipt follows the record
        assert kb.write_receipt(engine.quest_root, kb.receipt_inputs(engine.quest_root, None))
        return sorted(kb.snapshot(engine.quest_root, DESIGN))

    assert await kept("figure_only", [said]) == ["derive_peak", "figure_over_rate"]
    assert await kept("other", [said, "FI's analysis stopped inside its own lines"]) == []  # the known positive


# ---- fix round 1: the paper and QUALITY.md say the figure was not drawn, and why ------------------------------------------------
def test_the_paper_and_the_quality_report_say_the_figure_was_not_drawn_by_fis_fault(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    import re

    from core import quality_report
    from tests.test_disclosure import _engine as write_engine

    eng = write_engine(tmp_path)
    record = {"schema": frame.DERIVED_SCHEMA, "derived": {}, "fi_faults": ["x"], "figures": [
        {"id": "fig1", "file": "a.png", "status": "fi_fault", "why": "FI could not make the figure `fig1` (RuntimeError: planted)."}]}
    eng._figure_fallbacks({}, record)  # the existing step after the run
    sentence = ("Figure `fig1` was not drawn: FI's own drawing step failed (FI could not make the figure `fig1` "
                "(RuntimeError: planted)); this is FI's fault, not the study's.")
    seen: list[str] = []

    async def fake_chat(self, messages, **kw):  # a writer that copies into the limitations what FI told it about the figure
        text = " ".join(str(m.get("content") if isinstance(m, dict) else m) for m in messages)
        seen.append(text)
        found = re.search(r"Figure `fig1` was not drawn: FI's own drawing step failed .*?not the study's\.", text)
        limits = found.group(0) if found else "The model is simple."
        return ("# Title\n\n## Methods\nWe simulate a model.\n\n## Results\nThe mean was 0.5.\n\n"
                f"## Limitations\n{limits}\n\n## References\n[1] A.\n")

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    out = asyncio.run(eng._node_write({"topic": "t", "title": "t", "iteration": 0}))
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert sentence in " ".join(seen), "the writer was told, in FI's own sentence"
    assert sentence in paper, "the delivered paper says the figure was not drawn and why"
    quality = quality_report.render(quality_report.build(tmp_path, running=False))
    assert "fig1" in quality and "FI's fault" in quality


# ---- fix round 2 ------------------------------------------------------------------------------------------------------------------
def test_a_default_figure_that_fails_in_fis_own_code_is_disclosed_like_any_other(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from tests.test_disclosure import _engine as write_engine

    eng = write_engine(tmp_path)
    record = {"schema": frame.DERIVED_SCHEMA, "derived": {}, "fi_faults": ["x"], "figures": [
        {"id": "fig1", "file": "a.png", "status": "fi_fault", "fallback": True,
         "why": "FI's drawing of the default figure `fig1` stopped with KeyError: 'k'."}]}
    eng._figure_fallbacks({}, record)
    problems = eng._cannot_repair_problems()
    assert len(problems) == 1 and problems[0]["check"] == "figure_fi_fault"
    assert "Figure `fig1` was not drawn: FI's own drawing step failed" in problems[0]["text"] and "FI's fault" in problems[0]["text"]
    assert eng._cannot_repair_stop() is None  # it stops nothing
    assert "Figure `fig1` was not drawn" in eng._run_checks_note()


def test_faults_of_the_earlier_run_do_not_make_the_run_that_draws_only_the_figures_unproduced(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design(), _DRAWS)
    first = _run_with_fault(quest)  # the first run: a fault of FI's own in fig1, exit 0
    assert first["code"] == 0 and any(_PLANTED in f for f in first["faults"])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS")} | {"FI_FIGURE_FALLBACK": "only"}
    done = subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True, timeout=300,
                          creationflags=NO_WINDOW)
    record = json.loads((quest / frame.DERIVED_PATH).read_text(encoding="utf-8"))
    assert done.returncode == 0 and next(o for o in record["figures"] if o["id"] == "fig1")["status"] == "ok"
    assert frame.stopping_faults(record) == [], record["fi_faults"]  # the earlier run's fault is not this run's
    assert not any(_PLANTED in f for f in record["fi_faults"])


def test_stopping_faults_of_a_figure_are_never_read_from_either_run() -> None:
    said = "Figure `fig1` was not drawn: FI's own drawing step failed (x); this is FI's fault, not the study's."
    figure = {"id": "fig1", "status": "fi_fault", "said": said}
    assert frame.stopping_faults({"figures": [figure], "fi_faults": [said]}) == []
    kept = {"from_earlier_run": True, "figures": [figure], "fi_faults": [said, "an earlier formula fault"],
            "this_run": {"status": "stopped", "why": "the record of the trials is missing"}}
    assert frame.stopping_faults(kept) == ["the record of the trials is missing"]  # only this run's own
    assert frame.stopping_faults({"figures": [], "fi_faults": ["FI's analysis stopped inside its own lines"]}) == [
        "FI's analysis stopped inside its own lines"]  # the known positive


# ---- final review I7: a default figure that failed in FI's own code on the last run ---------------------------------------------
def test_a_default_figure_that_failed_after_the_bodys_last_repair_is_disclosed(tmp_path: Path, monkeypatch) -> None:
    from tests.test_disclosure import _engine as write_engine

    eng = write_engine(tmp_path)
    record = {"schema": frame.DERIVED_SCHEMA, "derived": {}, "fi_faults": [], "figures": [
        {"id": "fig1", "file": "a.png", "status": "failed", "why": "its function raised",
         "fallback_fault": "FI's drawing of the default figure `fig1` stopped with KeyError: 'k'."}]}
    eng._figure_fallbacks({}, record)
    problems = eng._cannot_repair_problems()
    assert len(problems) == 1 and problems[0]["check"] == "figure_fi_fault"
    assert "Figure `fig1` was not drawn: FI's own drawing step failed (FI's drawing of the default figure `fig1` stopped with " \
           "KeyError: 'k')" in problems[0]["text"] and "FI's fault" in problems[0]["text"]
    assert eng._cannot_repair_stop() is None  # it stops nothing
    assert "Figure `fig1` was not drawn" in eng._run_checks_note()
    # the known negative: a body that failed with no fault of the default figure is not FI's fault
    eng._figure_fallbacks({}, {**record, "figures": [{"id": "fig1", "file": "a.png", "status": "failed", "why": "x"}]})
    assert eng._cannot_repair_problems() == []
