"""A mean over a subset of the trials is checked trial by trial under research (the R2 re-audit, P0-01 and P1-01).

The per-value check compared a conditional mean's values with a pool of every trial's values: 10 successes with y=100
and 90 failures with y=1, reported as `p_count=10` with ten 1s, passed — the count was true and 1 was a real value, just
not a value of the subset. Under research the trials now return the proportion as 1 or 0, and FI takes the subset
from its own record, trial by trial; a record without it is the simulation's to fix. Counts are whole numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core import run_manifest as rm
from core import trial_runner as tr

GRID = {"R0": [1.5, 3.0], "N": [100]}
PROTOCOL = {"grid": GRID, "runs_per_setting": 50, "metrics": [
    {"id": "p", "kind": "proportion", "estimand": "P(major)", "unit": "run"},
    {"id": "m", "kind": "mean", "estimand": "E[y | major]", "unit": "run", "given": "p"},
]}


def _rows(successes: dict[str, int], n: int = 50, *, membership: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Per cell: the first k trials are in the subset (y = 100 + trial), the rest are not (y = 1)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for key, k in successes.items():
        rows = []
        for t in range(n):
            values: dict[str, float] = {"y": float(100 + t) if t < k else 1.0}
            if membership:
                values["p"] = 1.0 if t < k else 0.0
            rows.append({"cell": key, "trial": t, "seed": t, "values": values})
        out[key] = rows
    return out


ROWS = _rows({"R0=1.5,N=100": 10, "R0=3.0,N=100": 40})


def _stratum(values: list[Any], count: Any, total: Any = 50) -> dict[str, Any]:
    return {"p_count": count, "p_total": total, "m_values": values}


def _check(result: Any, rows: dict[str, list[dict[str, Any]]] = ROWS) -> tuple[list[str], list[str]]:
    return tr.given_rows_problems(PROTOCOL, rows, result)


def test_values_of_trials_outside_the_subset_do_not_pass() -> None:
    """The re-audit's counterexample: the count is right and 1 is a real value, but not one of the subset's."""
    forged = {"R0=1.5,N=100": _stratum([1.0] * 10, 10)}
    analysis, simulation = _check(forged)
    assert simulation == [] and any("not the values of the trials whose `p` is 1" in a for a in analysis)


def test_the_subset_s_own_values_pass_in_any_order_and_as_fractions_of_n() -> None:
    honest = {"R0=1.5,N=100": _stratum([float(100 + t) for t in reversed(range(10))], 10),
              "R0=3.0,N=100": _stratum([(100 + t) / 100 for t in range(40)], 40)}
    assert _check(honest) == ([], [])


def test_values_borrowed_from_another_setting_do_not_pass() -> None:
    borrowed = {"R0=1.5,N=100": _stratum([float(100 + t) for t in range(30, 40)], 10)}
    analysis, _ = _check(borrowed)
    assert any("not the values of the trials" in a for a in analysis)


def test_a_count_other_than_the_subset_s_size_is_the_analysis_s_to_fix() -> None:
    top10 = {"R0=3.0,N=100": _stratum([float(100 + t) for t in range(30, 40)], 10)}
    analysis, _ = _check(top10)
    assert any("`p_count` says 10, but FI's trials of those settings returned `p` = 1 for 40" in a for a in analysis)


def test_a_record_without_the_membership_is_the_simulation_s_to_fix() -> None:
    no_p = _rows({"R0=1.5,N=100": 10}, membership=False)
    analysis, simulation = _check({"R0=1.5,N=100": _stratum([1.0] * 10, 10)}, no_p)
    assert analysis == [] and len(simulation) == 1
    assert "run_trial must return `p` as 1" in simulation[0], "a doable repair, not a dead end"
    assert tr.RETURN_MEMBERSHIP.format(given="p") in simulation[0]


def test_counts_must_be_whole_numbers() -> None:
    manifest = {"schema": rm.SCHEMA, "realized_grid": GRID,
                "attempted_per_cell": {"R0=1.5,N=100": 50, "R0=3.0,N=100": 50},
                "successful_per_cell": {"R0=1.5,N=100": 50, "R0=3.0,N=100": 50},
                "failed_trials": [], "thresholds_used": {}}
    for bad in (10.9, float("nan"), True, -1, "10"):
        result = {"R0=1.5,N=100": _stratum([1.0] * 10, bad), "R0=3.0,N=100": _stratum([1.0] * 40, 40)}
        found = rm.problems(PROTOCOL, manifest, result_json=result, trial_mode=True)
        assert any("not a whole number" in f for f in found), (bad, found)
    exact = {"R0=1.5,N=100": _stratum([1.0] * 10, 10.0), "R0=3.0,N=100": _stratum([1.0] * 40, 40)}
    assert not any("whole number" in f for f in rm.problems(PROTOCOL, manifest, result_json=exact, trial_mode=True))


def test_rows_are_read_from_fi_s_record_and_an_edited_row_is_left_out(tmp_path: Path) -> None:
    rows = _rows({"R0=1.5,N=100": 2}, n=3)["R0=1.5,N=100"]
    ledger = []
    for r in rows:
        digest = hashlib.sha256(json.dumps(r["values"], sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
        ledger.append(json.dumps({"event": "trial", "cell": r["cell"], "trial": r["trial"], "status": "ok",
                                  "values_sha256": digest}))
    (tmp_path / tr.RAW_DIRNAME).mkdir()
    (tmp_path / tr.RAW_DIRNAME / tr.LEDGER_NAME).write_text("\n".join(ledger) + "\n", encoding="utf-8")
    record_rows = [{"trial": r["trial"], "seed": r["seed"], "status": "ok", "values": dict(r["values"])} for r in rows]
    record_rows[2]["values"]["y"] = 999.0  # changed after the trials ran
    run = tmp_path / tr.RUN_RECORD
    run.parent.mkdir(parents=True, exist_ok=True)
    run.write_text(json.dumps({"cells": [{"key": "R0=1.5,N=100", "rows": record_rows}]}), encoding="utf-8")
    got = tr.recorded_rows_by_cell(tmp_path)
    assert [r["trial"] for r in got["R0=1.5,N=100"]] == [0, 1]


def test_outside_research_the_engine_does_not_run_the_row_check() -> None:
    import inspect

    from core import engine

    source = inspect.getsource(engine.Engine._run_manifest_problems)
    assert 'if self.config.rigor_profile == "research":' in source and "given_rows_problems" in source
