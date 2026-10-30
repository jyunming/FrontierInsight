"""A figure of the study's that failed after its repairs belongs to the analysis, not to the run: the run's results were
produced, so the confirm run still confirms them and the improve step is not skipped for "no result". A run that crashed
without results is still no result. No real model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import analysis_frame as frame
from core import phased
from tests.test_confirm_one_shot import _TRIAL, _code, _env
from tests.test_phased import _config


def _record(root: Path, *, figure: str = "failed", result: str = "ok", schema: str = frame.DERIVED_SCHEMA, **more) -> None:
    doc = {"schema": schema, "derived": {"r1": {"role": "result", "by": "derive_r1", "status": result, "why": "x"}},
           "figures": [{"id": "fig2", "status": figure, "file": "fig2.png"}, {"id": "fig1", "status": "ok", "file": "fig1.png"}],
           "fi_faults": [], **more}
    path = root / frame.DERIVED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _engine(tmp_path: Path, *, phased_on: bool):
    from core.engine import Engine

    cfg = _config(tmp_path / "out", phased_on=phased_on)
    return Engine(cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"improve_rounds": 2})}))


def _confirm(engine, exec_result: dict, result_json: dict) -> dict:
    root = engine.quest_root
    _code(root, _TRIAL)
    phased.prepare(root, engine.quest_id)
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    phased.seed_env(root, _env(0, 0))
    assert engine._route_after_evidence_gate(gate) == "confirm"
    phased.seed_env(root, _env(0, 0))
    engine._route_after_evidence_gate({**gate, "result_json": result_json, "exec_result": exec_result})
    return phased.load(root)


def test_a_failed_figure_does_not_wipe_out_the_confirmation_of_the_results(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased_on=True)
    _record(engine.quest_root, figure="failed")
    record = _confirm(engine, {"returncode": 1}, {"a": 2})
    assert phased.status(record) == phased.CONFIRMED, phased.status(record)


def test_a_crash_without_results_still_confirms_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased_on=True)
    _record(engine.quest_root, figure="ok", result="failed")  # a function of the study's for a result failed
    record = _confirm(engine, {"returncode": 1}, {"a": 2})
    assert phased.status(record) != phased.CONFIRMED


def test_a_failed_figure_does_not_skip_the_improve_step_but_a_crash_does(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased_on=False)
    state = {"result_json": {"a": 2}, "exec_result": {"returncode": 1}, "exec_give_up_reason": "the figure could not be drawn"}
    _record(engine.quest_root, figure="failed")
    assert not engine._run_has_no_result(state)
    assert engine._improve_skip(state) != "the run did not produce a result"
    # the same exit code with no record of this run, a record of an earlier run, or no result printed: no result
    (engine.quest_root / frame.DERIVED_PATH).unlink()
    assert engine._run_has_no_result(state)
    _record(engine.quest_root, figure="failed", from_earlier_run=True)
    assert engine._run_has_no_result(state)
    _record(engine.quest_root, figure="failed")
    assert engine._run_has_no_result({**state, "result_json": {}})
    assert engine._run_has_no_result({**state, "exec_result": {"returncode": 2}})


def test_only_figures_failed_reads_fis_record(tmp_path: Path) -> None:
    _record(tmp_path, figure="refused")
    rec = frame.read_record(tmp_path)
    assert frame.only_figures_failed(rec, 1)
    assert not frame.only_figures_failed(rec, 0) and not frame.only_figures_failed(rec, 2)
    assert not frame.only_figures_failed({**rec, "fi_faults": ["FI's own"]}, 1)
    _record(tmp_path, figure="ok")
    assert not frame.only_figures_failed(frame.read_record(tmp_path), 1)
