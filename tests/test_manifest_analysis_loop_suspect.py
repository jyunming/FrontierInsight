"""Through the real graph: an analysis script that loops over only some of an axis's values (``for R0 in (1.5, 3.0)``) is
not stopped before the run (FI runs every setting itself), and under the default ``run_manifest_check: warn`` it is
sent back only when FI's own count shows it reports no trials for a value that loop leaves out. Fake model, real venv."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.engine import Engine
from tests.test_manifest_fabricated_under_warn import _model
from tests.test_run_manifest import ANALYSIS_TRIAL, _cfg, _record

_LOOP = '    out[c["key"]] = {"final_size_values": m["values"], "final_size_count": m["count"]}\n'
# The analysis reports only the settings its literal loop names: R0 = 0.9 is left out of what it reports.
ANALYSIS_LOOP_DROPS = ANALYSIS_TRIAL.replace(
    'for c in data["cells"]:\n    m = c["metrics"]["final_size"]\n' + _LOOP,
    'for R0 in (1.5, 3.0):\n  for c in [c for c in data["cells"] if c["key"] == "R0=%s" % R0]:\n    m = c["metrics"]["final_size"]\n' + _LOOP,
)
# The same loop, as an acceptance check beside a report of every setting.
ANALYSIS_LOOP_CHECKS = ANALYSIS_TRIAL.replace(
    "os.makedirs('figures'",
    "ok = True\nfor R0 in (1.5, 3.0):\n    ok &= any(c['key'] == 'R0=%s' % R0 for c in data['cells'])\nos.makedirs('figures'",
)


def test_the_fixtures_are_what_they_say() -> None:
    compile(ANALYSIS_LOOP_DROPS, "analysis_loop_drops.py", "exec")  # a script that does not even parse proves nothing below
    compile(ANALYSIS_LOOP_CHECKS, "analysis_loop_checks.py", "exec")
    assert ANALYSIS_LOOP_DROPS != ANALYSIS_TRIAL and "for R0 in (1.5, 3.0)" in ANALYSIS_LOOP_DROPS
    assert ANALYSIS_LOOP_CHECKS != ANALYSIS_TRIAL and "for R0 in (1.5, 3.0)" in ANALYSIS_LOOP_CHECKS


@pytest.mark.asyncio
async def test_a_loop_that_leaves_a_setting_out_of_the_report_is_sent_back_under_warn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_LOOP_DROPS, [ANALYSIS_TRIAL], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert calls.count("ExecuteReflect") == 1, "sent back once, as FI counted R0 = 0.9 as not reported"
    assert artifacts.paper_md is not None and _record(engine)["status"] == "ok"


@pytest.mark.asyncio
async def test_a_loop_that_leaves_it_out_and_stays_stops_the_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_LOOP_DROPS, [ANALYSIS_LOOP_DROPS], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert _record(engine)["status"] == "stopped" and artifacts.paper_md is None


@pytest.mark.asyncio
async def test_the_same_loop_as_a_check_beside_a_full_report_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The s5 shape: the quest is not stopped before the run, and FI's count finds nothing left out."""
    calls: list[str] = []
    _model(monkeypatch, ANALYSIS_LOOP_CHECKS, [ANALYSIS_LOOP_CHECKS], calls)
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert calls.count("ExecuteReflect") == 0 and artifacts.paper_md is not None
    assert _record(engine)["status"] == "ok"
