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
              "R0=3.0,N=100": _stratum([float(100 + t) for t in range(40)], 40)}
    assert _check(honest) == ([], [])
    fractions = {"R0=1.5,N=100": _stratum([(100 + t) / 100 for t in range(10)], 10),
                 "R0=3.0,N=100": _stratum([(100 + t) / 100 for t in range(40)], 40)}
    assert _check(fractions) == ([], [])
    mixed = {"R0=1.5,N=100": _stratum([float(100 + t) for t in range(10)], 10),
             "R0=3.0,N=100": _stratum([(100 + t) / 100 for t in range(40)], 40)}
    assert any("one mean averages one quantity" in a for a in _check(mixed)[0]), "sizes here, fractions there"


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
    assert "run_trial must return, under the proportion's own id `p`, 1" in simulation[0], "a doable repair"
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


def _engine(profile: str, quest_root: Path):
    from types import SimpleNamespace

    from core.engine import Engine

    eng = object.__new__(Engine)
    eng.config = SimpleNamespace(rigor_profile=profile)
    eng.quest_root = quest_root
    return eng


def test_outside_research_the_row_check_does_not_run(tmp_path: Path) -> None:
    forged = {"R0=1.5,N=100": _stratum([1.0] * 10, 10)}
    ledger = [{"cell": "R0=1.5,N=100", "trial": 0, "status": "ok"}]
    assert _engine("default", tmp_path)._given_row_findings(PROTOCOL, ledger, forged) == ([], [])
    analysis, simulation = _engine("research", tmp_path)._given_row_findings(PROTOCOL, ledger, forged)
    assert simulation and "run.json) is missing" in simulation[0], "research: no record while trials ran fails closed"


def test_a_mean_under_a_wrapper_key_is_checked_against_the_whole_run() -> None:
    whole = [float(100 + t) for t in range(10)] + [float(100 + t) for t in range(40)]
    for forged in ({"summary": _stratum([1.0] * 50, 50)},
                   {"results": [_stratum([1.0] * 50, 50)]},
                   {"metrics": {"summary": _stratum([1.0] * 50, 50)}}):
        analysis, _ = _check(forged)
        assert any("not the values of the trials" in a for a in analysis), forged
    assert _check({"summary": _stratum(whole, 50)}) == ([], []), "the honest whole-run mean under a wrapper passes"


def test_a_flag_quantity_never_stands_in_for_the_averaged_one() -> None:
    rows = {k: [{**r, "values": {**r["values"], "ok": 1.0}} for r in v] for k, v in ROWS.items()}
    forged = {"R0=1.5,N=100": _stratum([1.0] * 10, 10)}
    analysis, _ = _check(forged, rows)
    assert any("not the values of the trials" in a for a in analysis)


def test_one_quantity_in_every_stratum() -> None:
    rows = {k: [{**r, "values": {**r["values"], "z": r["values"]["y"] * 3}} for r in v] for k, v in ROWS.items()}
    mixed = {"R0=1.5,N=100": _stratum([float(100 + t) for t in range(10)], 10),
             "R0=3.0,N=100": _stratum([float(100 + t) * 3 for t in range(40)], 40)}
    analysis, _ = _check(mixed, rows)
    assert any("one mean averages one quantity" in a for a in analysis)
    same = {"R0=1.5,N=100": _stratum([float(100 + t) * 3 for t in range(10)], 10),
            "R0=3.0,N=100": _stratum([float(100 + t) * 3 for t in range(40)], 40)}
    assert _check(same, rows) == ([], [])


def test_the_writer_and_the_reader_agree(tmp_path: Path) -> None:
    """_collect writes the ledger, _save_run the record; recorded_rows_by_cell reads them back, hash-checked."""
    folder = tmp_path / ".fi" / "trials"
    folder.mkdir(parents=True)
    plan = tr._plan(tmp_path, "code/simulate.py", {"R0": [1.5]}, runs_per_setting=3, base_seed=7,
                    deterministic=False, folder=folder, out_name="cell{index}.out.jsonl",
                    thresholds={"major": 0.1})
    spec = json.loads((tmp_path / plan[0]["spec"]).read_text(encoding="utf-8"))
    assert spec["thresholds"] == {"major": 0.1}, "the protocol's thresholds reach the harness"
    lines = [json.dumps({"nonce": plan[0]["nonce"], "trial": t["trial"], "seed": t["seed"], "status": "ok",
                         "values": {"p": float(t["trial"] < 2), "y": 10.0 + t["trial"]}}) for t in plan[0]["trials"]]
    (tmp_path / plan[0]["out"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = type("R", (), {"returncode": 0, "timed_out": False, "stderr": ""})()
    run = tr._collect(tmp_path, plan, {0: ok}, run_id="r", thresholds={"major": 0.1}, not_reported="")
    tr._save_run(tmp_path, "k", run)
    rows = tr.recorded_rows_by_cell(tmp_path)
    assert [r["trial"] for r in rows["R0=1.5"]] == [0, 1, 2]
    result = {"R0=1.5": {"p_count": 2, "p_total": 3, "m_values": [10.0, 11.0]}}
    protocol = {**PROTOCOL, "grid": {"R0": [1.5]}}
    assert tr.given_rows_problems(protocol, rows, result) == ([], [])


def test_the_harness_sets_the_thresholds_before_loading_the_simulation() -> None:
    assert 'os.environ["FI_THRESHOLDS"]' in tr.HARNESS_SOURCE
    assert tr.HARNESS_SOURCE.index("FI_THRESHOLDS") < tr.HARNESS_SOURCE.index("exec_module")


def test_a_row_without_a_trial_id_is_never_read_as_trial_0() -> None:
    assert tr._trial_id({"trial": None}) is None and tr._trial_id({}) is None and tr._trial_id({"trial": True}) is None
    assert tr._trial_id({"trial": 0}) == 0
