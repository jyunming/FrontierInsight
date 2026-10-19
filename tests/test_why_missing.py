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


def _pooled_copies_quest(tmp_path: Path, rows_of_gap, *, copies: int = 2) -> Path:
    """A quest whose grid has an axis the simulation never reads (`copy`), so its cells pool; every copy gave the same trials."""
    cells = []
    for copy in range(copies):
        for gap, values in rows_of_gap.items():
            rows = [{"trial": t, "status": "ok", "values": {"width": float("nan") if v is None else v},
                     **({"why_missing": {"width": "no crossing"}} if v is None else {})} for t, v in enumerate(values)]
            cells += in_run.summary_cells({"gap": gap, "copy": copy}, f"gap={gap:g},copy={copy}", len(values), rows, [])
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}, allow_nan=True), encoding="utf-8")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(
        "def run_trial(cell, i, seed):\n    return {'width': cell['gap'] * 1.0}\n", encoding="utf-8")
    return tmp_path


PROTOCOL = {"grid": {"gap": [1.0, 2.0], "copy": [0, 1]}}


def test_pooled_copies_of_the_same_trials_count_a_missing_value_once(tmp_path: Path) -> None:
    root = _pooled_copies_quest(tmp_path, {1.0: [5.0, None], 2.0: [6.0, 7.0]})
    built = results_by_setting.from_quest(root, PROTOCOL)[1]
    assert built is not None
    text = "\n".join(results_by_setting.markdown(t) for t in built.tables)
    assert "(1 run with no value: no crossing)" in text and "2 runs with no value" not in text


def test_pooled_copies_with_no_value_at_all_count_it_once(tmp_path: Path) -> None:
    root = _pooled_copies_quest(tmp_path, {1.0: [None, None], 2.0: [6.0, 7.0]})
    built = results_by_setting.from_quest(root, PROTOCOL)[1]
    text = "\n".join(results_by_setting.markdown(t) for t in built.tables)
    assert "no value: no crossing" in text and "4 runs" not in text
    (setting,) = [s for s in built.distinct.settings if s.setting.get("gap") == 1.0]
    assert setting.metrics["width"]["missing"] == 2


# --- the text a table adds for a missing value is read by nothing that takes a number from a table -------------------

PAPER = ("# A title\n\n## Methods\n\nWe ran it.\n\n## Results\n\nThe mean width at the first gap is 5.5.\n\n"
         "{{results_table}}\n\n## Discussion\n\nIt worked.\n\n## Limitations\n\nFew.\n")


def _paper_of(tmp_path: Path, rows_of_gap) -> tuple[Path, str]:
    from core import paper_facts

    root = _pooled_copies_quest(tmp_path, rows_of_gap, copies=1)
    assert results_by_setting.write(root, {"grid": {"gap": [1.0, 2.0]}}) is not None
    return root, paper_facts.render(paper_facts.place_blocks(PAPER, root), root).text


def test_a_table_with_missing_values_is_read_by_the_checks_as_one_without(tmp_path: Path) -> None:
    from core import paper_facts, typed_numbers

    root_a, with_missing = _paper_of(tmp_path / "a", {1.0: [5.0, 6.0, None], 2.0: [6.0, 7.0, 8.0]})
    root_b, without = _paper_of(tmp_path / "b", {1.0: [5.0, 6.0], 2.0: [6.0, 7.0, 8.0]})
    assert "with no value" in with_missing and "with no value" not in without
    # The checks of the writer's own text leave the block out, so the added words are never a typed number.
    assert paper_facts.without_blocks(with_missing) == paper_facts.without_blocks(without)
    assert [n.token for n in typed_numbers.numbers(with_missing)] == [n.token for n in typed_numbers.numbers(without)]
    plan = frozenset()
    assert typed_numbers.report(with_missing, plan) == typed_numbers.report(without, plan)
    # The claim check is given the tables of every setting as evidence text for the model (it parses no cell).
    _rows, built = results_by_setting.from_quest(root_a, {"grid": {"gap": [1.0, 2.0]}})
    note = results_by_setting.writer_note([], built)
    assert "5.5" in note and "(1 run with no value: no crossing)" in note
