"""A value a run cannot measure is `null` with its reason under `why_missing`, never a stand-in number: the harness keeps
the reason beside the missing value, the record counts it by reason, and FI's tables say it (core/in_run.py)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from core import in_run, results_by_setting, trial_runner, trial_table


def test_the_reason_is_not_a_result_and_is_kept_only_for_a_missing_value() -> None:
    returned = {"width": None, "depth": 3.0, "why_missing": {"width": "no crossing in the range", "depth": "ignored"}}
    kept, why = in_run.read_returned(returned, [], "run_cell")
    assert why == "" and set(kept) == {"width", "depth"} and math.isnan(kept["width"])
    assert in_run.missing_reasons(returned, kept) == {"width": "no crossing in the range"}


def test_a_run_that_returns_only_reasons_has_no_result() -> None:
    kept, why = in_run.read_returned({"why_missing": {"width": "x"}}, [], "run_cell")
    assert kept is None and "no result" in why


def test_the_reason_counts_by_reason_in_the_record() -> None:
    rows = [{"trial": 0, "status": "ok", "values": {"width": float("nan")}, "why_missing": {"width": "does not print"}},
            {"trial": 1, "status": "ok", "values": {"width": float("nan")}, "why_missing": {"width": "does not print"}},
            {"trial": 2, "status": "ok", "values": {"width": 4.0}}]
    (cell,) = in_run.summary_cells({"gap": 1}, "gap=1", 3, rows, [])
    metric = cell["metrics"]["width"]
    assert metric["non_finite"] == 2 and metric["why_missing"] == {"does not print": 2}
    summary = trial_table.summaries([cell])[0]["metrics"]["width"]
    assert summary["n"] == 1 and summary["missing"] == 2 and summary["why_missing"] == {"does not print": 2}


def test_the_table_says_how_many_runs_have_no_value_and_why(tmp_path: Path) -> None:
    cells = []
    for gap, values, why in ((1.0, [None, None], {"width": "does not print"}), (2.0, [5.0, 6.0], {}),
                             (3.0, [7.0, None], {"width": "no crossing"})):
        rows = [{"trial": t, "status": "ok", "values": {"width": float("nan") if v is None else v},
                 **({"why_missing": why} if v is None else {})} for t, v in enumerate(values)]
        cells += in_run.summary_cells({"gap": gap}, f"gap={gap:g}", 2, rows, [])
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}, allow_nan=True), encoding="utf-8")
    built = results_by_setting.build(trial_table.summaries(trial_table._record(tmp_path)),
                                     {"grid": {"gap": [1.0, 2.0, 3.0]}}, {})
    text = "\n".join(results_by_setting.markdown(t) for t in built.tables)
    assert "no value: does not print" in text
    assert "(1 run with no value: no crossing)" in text


def test_the_harness_keeps_the_reason_of_a_real_run(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text(
        "def run_cell(cell):\n"
        "    if cell['gap'] > 1:\n"
        "        return {'width': None, 'why_missing': {'width': 'does not print'}}\n"
        "    return {'width': 2.0}\n", encoding="utf-8")
    (tmp_path / trial_runner.HARNESS_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / trial_runner.HARNESS_PATH).write_text(trial_runner.HARNESS_SOURCE, encoding="utf-8")
    spec = tmp_path / "spec.json"
    out = tmp_path / "out.jsonl"
    spec.write_text(json.dumps({"module": "code/simulate.py", "entry": "run_cell", "cell": {"gap": 2},
                                "trials": [{"trial": 0, "seed": None}], "nonce": "n1"}), encoding="utf-8")
    import subprocess

    subprocess.run([sys.executable, str(trial_runner.HARNESS_PATH), "spec.json", "out.jsonl"], cwd=tmp_path, check=True,
                   capture_output=True, timeout=120)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert row["status"] == "ok" and row["why_missing"] == {"width": "does not print"}


@pytest.mark.parametrize("prompt", ["agents/implement.md", "agents/implement_body.md", "agents/execute_reflect.md"])
def test_the_code_writer_is_told_one_way_to_say_a_value_is_missing(prompt: str) -> None:
    text = (Path(__file__).parent.parent / prompt).read_text(encoding="utf-8")
    assert "why_missing" in text and '"diverged": true' not in text
