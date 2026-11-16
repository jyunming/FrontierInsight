"""A repair of a reader's finding keeps every result and figure FI's analysis gave before it (core/pipeline_gate.py
``results_held`` / ``results_lost``, used by core/run_checks.py ``_finding_gate``): the analysis is run on the same
trials of FI's small run before and after the repair, and a declared result with no value where it had one, or a figure
no longer drawn, puts the repair back. A change from one value to one per setting that keeps every value stays."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import pipeline_gate as gate

SETTINGS = ["n=10", "n=20", "n=40"]


def _scratch(tmp_path: Path, derived: dict[str, Any], figures: list[dict[str, Any]], *, ok: int = 3) -> Path:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    cells = [{"key": k, "ok": ok, "metrics": {}} for k in SETTINGS] + [{"key": "n=80", "ok": 0, "metrics": {}}]
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "data" / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "results" / "derived.json").write_text(
        json.dumps({"schema": "fi.derived/v1", "derived": derived, "figures": figures}), encoding="utf-8")
    return tmp_path


def _result(held: dict[str, list[str]] | None, drawn: list[str] | None = None, **kw: Any) -> gate.GateResult:
    return gate.GateResult(True, held=held, drawn=drawn, **kw)


def test_what_the_analysis_gave_is_read_from_fis_own_record(tmp_path: Path) -> None:
    root = _scratch(tmp_path, {
        "rate": {"status": "ok", "value": 0.5},
        "spread": {"status": "ok", "by_setting": {"n=10": 1.0, "n=20": None, "n=40": float("nan"), "n=80": 2.0}},
        "cover": {"status": "ok", "by_level": {"input": "kind", "values": {"a": 0.9, "b": None}}},
        "none": {"status": "ok", "value": None},
    }, [{"id": "fig1", "status": "ok"}, {"id": "fig2", "status": "failed"}])
    held, drawn = gate.results_held(root)
    # one value for the study, or one for some compared level, counts at every setting with a completed trial
    assert held == {"rate": SETTINGS, "spread": ["n=10"], "cover": SETTINGS, "none": []}
    assert drawn == ["fig1"]


def test_no_record_means_nothing_to_compare(tmp_path: Path) -> None:
    assert gate.results_held(tmp_path) == (None, None)
    assert gate.results_lost(_result(None), _result({"rate": SETTINGS})) == ""
    assert gate.results_lost(_result({"rate": SETTINGS}), _result(None)) == ""
    assert gate.results_lost(None, _result({"rate": SETTINGS})) == ""


def test_one_value_turned_into_one_per_setting_with_every_value_kept_is_no_loss() -> None:
    before = _result({"rate": SETTINGS}, ["fig1"])
    assert gate.results_lost(before, _result({"rate": SETTINGS}, ["fig1"])) == ""


def test_a_result_with_no_value_where_it_had_one_is_a_loss_named_with_its_setting() -> None:
    why = gate.results_lost(_result({"rate": SETTINGS, "other": ["n=10"]}), _result({"rate": ["n=10"], "other": ["n=10"]}))
    assert "`rate` had a value at 3 setting(s) and has none at 2 of them now (for example n=20)" in why
    assert "`other`" not in why
    assert why.endswith("A repair keeps every result and figure the analysis gave before.")


def test_a_result_gone_from_the_record_is_a_loss() -> None:
    assert "`rate`" in gate.results_lost(_result({"rate": SETTINGS}), _result({}))


def test_a_figure_no_longer_drawn_is_a_loss_with_why_it_stopped() -> None:
    after = gate.GateResult(False, "derive", gate.SMALL_DATA, "code/derive.py, in figure_fig1",
                            "`figure_fig1` stopped with ValueError: `x` is not a value of the plan", "ValueError",
                            body="figure_fig1", held={"rate": SETTINGS}, drawn=["fig2"])
    why = gate.results_lost(_result({"rate": SETTINGS}, ["fig1", "fig2"]), after)
    assert "figure `fig1` was drawn and is not drawn now (`figure_fig1` stopped with ValueError" in why


def test_a_result_that_gains_values_is_no_loss() -> None:
    assert gate.results_lost(_result({"rate": []}, []), _result({"rate": SETTINGS}, ["fig1"])) == ""


def test_what_the_analysis_gave_is_kept_in_the_gate_record_and_read_back() -> None:
    got = _result({"rate": SETTINGS}, ["fig1"])
    rec = got.record()
    back = gate.GateResult(**{k: v for k, v in rec.items() if k in gate.GateResult.__dataclass_fields__})
    assert back.held == {"rate": SETTINGS} and back.drawn == ["fig1"]
    assert "held" not in gate.GateResult(True).record()


# --- the engine: the baseline is the code that is kept ---------------------------------------------------------------


class _Eng:
    """The three methods of core/run_checks.py the guard uses, on a stand-in engine (no quest is built)."""

    from core.run_checks import RunChecksMixin as _M

    _finding_gate = _M._finding_gate
    _finding_analysis_start = _M._finding_analysis_start
    _finding_analysis_commit = _M._finding_analysis_commit

    def __init__(self, results: list[gate.GateResult | None], *, recorded: list[gate.GateResult | None] | None = None,
                 kept: gate.GateResult | None = None) -> None:
        from types import SimpleNamespace

        self.results, self.recorded = results, recorded
        self.config = SimpleNamespace(engine=SimpleNamespace(pipeline_gate=recorded is not None))
        if kept is not None:
            self._gate_kept = ("kept-key", kept)

    async def _analysis_gate(self, state: Any) -> Any:
        return self.recorded.pop(0) if self.recorded else None

    async def _gate_refusal(self, state: Any) -> str:
        return ""

    async def _small_analysis_now(self, state: Any) -> Any:
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_each_repair_is_compared_with_the_code_kept_last() -> None:
    kept, lost, fixed, worse = (_result({"rate": SETTINGS}, ["fig1"]), _result({"rate": ["n=10"]}, ["fig1"]),
                                _result({"rate": SETTINGS, "new": SETTINGS}, ["fig1"]), _result({"rate": SETTINGS}, ["fig1"]))
    eng = _Eng([kept, lost, fixed, worse])
    await eng._finding_analysis_start({})
    assert "`rate`" in await eng._finding_gate({})  # the first try loses values: put back, the baseline stays
    assert await eng._finding_gate({}) == ""  # the second try keeps them and adds one
    eng._finding_analysis_commit()  # ... and is kept: it is the new baseline
    assert "`new`" in await eng._finding_gate({})  # a later repair that drops what the kept one added is a loss


@pytest.mark.asyncio
async def test_no_analysis_on_the_small_run_refuses_nothing() -> None:
    eng = _Eng([None, _result({"rate": []})])
    await eng._finding_analysis_start({})
    assert await eng._finding_gate({}) == ""


@pytest.mark.asyncio
async def test_with_recorded_trials_a_figure_that_stops_only_at_settings_the_small_run_lacks_is_a_loss() -> None:
    small = _result({"rate": SETTINGS}, ["fig1"])  # on the small run's settings the figure still draws
    stopped = gate.GateResult(False, "derive", gate.SMALL_DATA, "code/derive.py, in figure_fig1", "`figure_fig1` stopped",
                              body="figure_fig1", held={"rate": SETTINGS}, drawn=[])
    eng = _Eng([small, small], recorded=[stopped], kept=_result({"rate": SETTINGS}, ["fig1"]))
    await eng._finding_analysis_start({})
    assert "figure `fig1` was drawn and is not drawn now" in await eng._finding_gate({})


@pytest.mark.asyncio
async def test_with_the_switch_off_the_recorded_trials_are_not_looked_at() -> None:
    eng = _Eng([_result({"rate": SETTINGS}), _result({"rate": SETTINGS})], kept=_result({"rate": SETTINGS}, ["fig1"]))
    await eng._finding_analysis_start({})
    assert await eng._finding_gate({}) == ""
