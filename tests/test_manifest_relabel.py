"""A list under a name of its own that holds another output's values at its setting is that output (r13's probe at
m = 200 printed `x1_power`'s values as `power_values`): counted as it and said in run.log, never "never produced". The
same list under another procedure's key is still numbers that setting never produced."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import run_manifest as rm
from core.engine import Engine
from tests.test_manifest_fabricated_under_warn import _model
from tests.test_manifest_list_attribution import PROCS, PROTO, _fabricated, _found, _honest, _study
from tests.test_run_manifest import ANALYSIS_TRIAL, _cfg, _record


def _probe(vals: dict[str, dict[str, list[float]]], held: dict[str, str]) -> dict[str, Any]:
    """r13's x1 probe at m=200: ``x1_probe/by_procedure/<p>/power_values`` holding the x1_power of procedure ``held[p]``."""
    return {"x1_probe": {"m": 200, "by_procedure": {
        p: {"power_values": list(vals[f"m=200,procedure={held[p]}"]["x1_power"])} for p in PROCS}}}


def test_a_list_holding_another_outputs_values_at_its_setting_is_counted_as_that_output() -> None:
    vals, per_cell = _study()
    rj = {**_honest(vals), **_probe(vals, {p: p for p in PROCS})}
    assert _fabricated(_found(rj, per_cell)) == []
    notes = rm.relabelled(PROTO, rj, per_cell)
    assert notes and all("x1_power" in n for n in notes) and any("x1_probe" in n for n in notes), notes


def test_another_procedures_values_of_that_output_stay_fabricated() -> None:
    vals, per_cell = _study()
    swapped = {"uncorrected": "uncorrected", "bonferroni": "benjamini_hochberg", "benjamini_hochberg": "bonferroni"}
    rj = {**_honest(vals), **_probe(vals, swapped)}
    found = _fabricated(_found(rj, per_cell))
    assert found and any("x1_probe" in f and "bonferroni" in f for f in found), _found(rj, per_cell)


# ---- through the real graph: the run goes on and run.log names it -----------------------------------------------------

ANALYSIS_RENAMED = ANALYSIS_TRIAL.replace(
    "print('RESULT_JSON: ' + json.dumps({'score': 0.987, 'by_cell': out}))",
    "probe = {c['key']: {'final_size_values': c['metrics']['outbreak']['values']} for c in data['cells']}\n"
    "print('RESULT_JSON: ' + json.dumps({'score': 0.987, 'by_cell': out, 'probe': probe}))",
)


@pytest.mark.asyncio
async def test_the_run_goes_on_and_run_log_names_the_output_the_list_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_RENAMED, [ANALYSIS_RENAMED], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "block"}))
    await engine.run()
    assert _record(engine)["status"] == "ok" and calls.count("ExecuteReflect") == 0
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "holds `outbreak`'s values" in log and "probe/R0=0.9/final_size_values" in log
