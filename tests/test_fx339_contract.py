"""FI computes per-parameter coverage and bias and across-trial statistics from its own record of the trials, and holds the finding
test writer to the names the plan declares. Neutral made-up plan: two estimates with true values, a coverage and a correlation."""

from __future__ import annotations

import json
import math
import string
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import finding_repair, frame_table, skeleton

PROTOCOL = {
    "form": "contract", "grid": {"noise": [0.1, 0.2]}, "in_each_run": {"channel": [0, 1]}, "runs_per_setting": 20,
    "constants": {"truth_a": 2.0, "truth_b": 5.0},
    "metrics": [
        {"id": "est_a", "kind": "mean", "estimand": "the first quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_a", "estimate": "est_a", "standard_error": "se_a", "interval": ["lo_a", "hi_a"], "level": 0.95}},
        {"id": "est_b", "kind": "mean", "estimand": "the second quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_b", "estimate": "est_b", "standard_error": "se_b", "interval": ["lo_b", "hi_b"], "level": 0.95}},
        {"id": "cover", "kind": "proportion", "estimand": "coverage", "unit": "one setting", "role": "physical_bound", "returned": "afterwards"},
        {"id": "corr_ab", "kind": "mean", "estimand": "correlation", "unit": "one setting", "role": "result", "returned": "afterwards"}]}


def _metric(values: list[float], trials: list[int] | None = None) -> dict[str, Any]:
    return {"values": list(values), "trials": list(range(len(values))) if trials is None else list(trials)}


def _cell(noise: float, est_a: list[float], est_b: list[float], width: float = 1.0, trials: list[int] | None = None) -> dict[str, Any]:
    cell = {"noise": noise}
    m = {"est_a": _metric(est_a, trials), "est_b": _metric(est_b, trials),
         "lo_a": _metric([v - width for v in est_a], trials), "hi_a": _metric([v + width for v in est_a], trials),
         "lo_b": _metric([v - width for v in est_b], trials), "hi_b": _metric([v + width for v in est_b], trials),
         "se_a": _metric([0.5] * len(est_a), trials), "se_b": _metric([0.5] * len(est_b), trials)}
    return {"cell": cell, "key": f"noise={noise}", "planned": len(est_a), "ok": len(est_a), "failed": 0, "metrics": m}


def _table(cells: list[dict[str, Any]], declared: tuple[str, ...] = ("est_a", "est_b", "lo_a", "hi_a", "lo_b", "hi_b"), **kw: Any) -> Any:
    ns: dict[str, Any] = {"fi_table": frame_table}
    sys.modules["fi_table"] = frame_table
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    known = {"est_a": {"interval": ["lo_a", "hi_a"], "truth": 2.0}, "est_b": {"interval": ["lo_b", "hi_b"], "truth": 5.0}}
    return ns["Table"](cells, list(declared), known=known, **kw)


# ---- A1: the finding test writer is given the declared names -------------------------------------------------------------------------

def _root(tmp_path: Path) -> Path:
    (tmp_path / "needs").mkdir()
    from core import frozen_protocol

    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": PROTOCOL, "sha256": frozen_protocol.sha256(PROTOCOL)}
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_the_declared_block_names_every_result_and_what_fi_counts(tmp_path: Path) -> None:
    text = finding_repair.declared_block(_root(tmp_path))
    for name in ("est_a", "est_b", "cover", "corr_ab", "channel", "est_a_coverage", "est_b_bias", "se_a", "lo_b", "hi_b"):
        assert name in text, name
    assert finding_repair.declared_block(tmp_path / "nowhere") == "(none stated)"


def test_the_rendered_finding_test_request_holds_the_declared_names(tmp_path: Path) -> None:
    root = _root(tmp_path)
    code = root / "code"
    code.mkdir()
    (code / "sim.py").write_text("def run(x):\n    return x\n", encoding="utf-8")
    template = (Path(__file__).parent.parent / "agents" / "code_finding_test.md").read_text(encoding="utf-8")
    env = SimpleNamespace(root=root, code=code, package="", topic="a made-up topic", equations="[]", scrub=lambda t: t,
                          prompts={"code_finding_test": string.Template(template)})
    issue = {"id": "F-1", "file": "sim.py", "function": "run", "statement": "the coverage of each parameter is wrong"}
    prompt = finding_repair._test_prompt(env, issue, "sim.py", "")
    assert "`est_a`" in prompt and "`corr_ab`" in prompt and "may read only the names the code's trial returns" in prompt
    assert "$declared" not in prompt


# ---- A2: bias and coverage per result, counted by FI ------------------------------------------------------------------------------------

def test_bias_is_the_mean_estimate_minus_the_true_value_and_known_lists_the_results() -> None:
    t = _table([_cell(0.1, [1.5, 2.5, 4.0], [5.0, 5.0, 5.0])])
    assert t.bias("est_a", "noise=0.1") == pytest.approx(8 / 3 - 2.0)
    assert t.bias("est_b", "noise=0.1") == pytest.approx(0.0)
    assert t.known() == ["est_a", "est_b"]
    assert t.coverage("est_a", "noise=0.1") == [1.0, 1.0, 0.0]
    with pytest.raises(type(t).__init__.__globals__["UnknownResult"]):
        t.bias("lo_a", "noise=0.1")


def test_the_printed_result_has_a_bias_beside_each_coverage() -> None:
    from core import analysis_frame

    src = analysis_frame.EXPERIMENT
    assert "_bias" in src and "_coverage" in src


# ---- B: statistics across the trials, paired by trial number ----------------------------------------------------------------------------

def test_paired_values_are_matched_by_trial_number_never_by_position() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    cell = _cell(0.1, a, b)
    del cell["metrics"]["est_b"]["values"][2]  # trial 2 of est_b was dropped: positions no longer line up
    del cell["metrics"]["est_b"]["trials"][2]
    t = _table([cell])
    xs, ys = t.paired("est_a", "est_b", "noise=0.1")
    assert xs == [1.0, 2.0, 4.0, 5.0, 6.0] and ys == [2.0, 4.0, 8.0, 10.0, 12.0]
    assert t.correlation("est_a", "est_b", "noise=0.1") == pytest.approx(1.0), "positional pairing would give another number"


def test_a_nan_trial_drops_that_pair_and_a_record_without_trial_numbers_pairs_nothing() -> None:
    cell = _cell(0.1, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, float("nan"), 4.0])
    assert _table([cell]).paired("est_a", "est_b", "noise=0.1") == ([1.0, 2.0, 4.0], [1.0, 2.0, 4.0])
    bare = _cell(0.1, [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    for name in ("est_a", "est_b"):
        del bare["metrics"][name]["trials"]
    t = _table([bare])
    assert t.paired("est_a", "est_b", "noise=0.1") == ([], []) and t.correlation("est_a", "est_b", "noise=0.1") is None
    twice = _cell(0.1, [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], trials=[0, 1, 1])
    assert _table([twice]).paired("est_a", "est_b", "noise=0.1") == ([], []), "a repeated trial number cannot be paired"


def test_correlation_sd_and_quantile_match_the_plain_formulas() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 10.0]
    b = [2.0, 1.0, 4.0, 3.0, 9.0]
    t = _table([_cell(0.1, a, b)])
    mean_a, mean_b = sum(a) / 5, sum(b) / 5
    expected = (sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
                / math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)))
    assert t.correlation("est_a", "est_b", "noise=0.1") == pytest.approx(expected)
    assert t.sd("est_a", "noise=0.1") == pytest.approx(math.sqrt(sum((x - mean_a) ** 2 for x in a) / 4))
    assert t.quantile("est_a", "noise=0.1", 0.5) == 3.0 and t.quantile("est_a", "noise=0.1", 0.0) == 1.0
    assert t.quantile("est_a", "noise=0.1", 0.25) == 2.0 and t.quantile("est_a", "noise=0.1", 1.0) == 10.0
    assert t.quantile("est_a", "noise=0.1", 0.125) == pytest.approx(1.5)
    assert frame_table.pearson([1.0, 2.0], [1.0, 2.0]) is None, "two pairs say nothing"
    assert frame_table.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None, "no spread"
    assert t.sd("est_a", "noise=0.1") is not None and _table([_cell(0.1, [1.0], [1.0])]).sd("est_a", "noise=0.1") is None


def test_the_reads_are_declared_ids_only_and_the_writer_is_told_about_them() -> None:
    t = _table([_cell(0.1, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])], declared=("est_a",))
    with pytest.raises(type(t).__init__.__globals__["UnknownResult"]):
        t.correlation("est_a", "est_b", "noise=0.1")
    from core import run_checks

    rules = run_checks.analysis_bodies_rules()
    for word in ("table.paired", "table.correlation", "table.sd", "table.quantile", "table.bias"):
        assert word in rules, word
    reads = "\n".join(skeleton.table_reads())
    assert "table.paired(" in reads and "table.correlation(" in reads


# ---- C: an estimate inside its own trial's interval -----------------------------------------------------------------------------------

def _inside_protocol() -> dict[str, Any]:
    return PROTOCOL


def _trial_cells(shift: bool = False, n: int = 40) -> list[dict[str, Any]]:
    cells = []
    for noise in (0.1, 0.2):
        a = [2.0 + 0.1 * ((i * 7) % 11 - 5) for i in range(n)]
        b = [5.0 + 0.6 * ((i * 5) % 9 - 4) for i in range(n)]
        cell = _cell(noise, a, b)
        if shift:  # the estimates of trial t were written under trial t + 1; the intervals stay
            v = cell["metrics"]["est_b"]["values"]
            cell["metrics"]["est_b"]["values"] = v[1:] + v[:1]
        cells.append(cell)
    return cells


def _calibration(cells: list[dict[str, Any]]) -> Any:
    from core import trial_checks

    return trial_checks.calibration(cells, PROTOCOL)


def test_an_estimate_outside_its_own_interval_in_many_trials_is_reported_not_stopped() -> None:
    from core import trial_checks

    cal = _calibration(_trial_cells(shift=False))
    assert not cal.inside, "rows that agree add nothing"
    cal = _calibration(_trial_cells(shift=True))
    inside = cal.inside
    assert [q.label for q in inside] == ["est_b"] and inside[0].below == 2 and inside[0].gross == 0
    assert all(s.value < trial_checks.INSIDE_MIN for s in inside[0].settings)
    assert not inside[0].fault, "a possible misplacement of rows is said, never a repair or a stop"
    assert "inside the interval the same trial reported" in inside[0].says()
    assert not any(q.kind == "inside" for q in cal.quantities), "never among the quantities the stop rules read"
    assert not any(q.label == "est_a" for q in cal.inside)


def test_the_paper_and_the_quality_report_say_the_statistic_is_not_confirmed(tmp_path: Path) -> None:
    from core import paper_facts, quality_report, trial_checks

    verdict = trial_checks.judge(_trial_cells(shift=True), PROTOCOL, None, repairs_left=True)
    rec = verdict.record()
    assert not [q for q in rec["calibration"]["quantities"] if q["kind"] == "inside"]
    inside = next(q for q in rec["calibration"]["inside"] if q["kind"] == "inside")
    assert inside["fault"] is False and inside["below"] == 2
    root = _root(tmp_path)
    trial_checks.write_record(root, rec)
    items: list[tuple[str, str]] = []

    def add(kind: str, name: str, text: str, key: str = "") -> None:
        items.append((name, text))

    paper_facts._trial_check_items(root, add)
    said = [t for _n, t in items if "inside the interval the same trial reported" in t]
    assert said and "may not be right" in said[0], items
    lines = quality_report._trial_checks_summary(quality_report._Quest(root))["lines"]
    assert any("inside the interval the same trial reported" in x and "may not be right" in x for x in lines), lines


# ---- A2 end to end: the printed result carries the coverage and the bias of every result with a true value -------------------------------

def _printed(tmp_path: Path, protocol: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    import os
    import subprocess

    from core import analysis_frame

    quest = tmp_path / "quest"
    (quest / "raw").mkdir(parents=True)
    (quest / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "thresholds": {}, "cells": cells}), encoding="utf-8")
    analysis_frame.ensure_files(quest / "code", {"protocol": protocol})
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS")}
    r = subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True, timeout=300,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON: ")]
    assert last, r.stderr[-1500:]
    return json.loads(last[-1][len("RESULT_JSON: "):])


def test_the_printed_result_gives_one_coverage_and_one_bias_for_each_parameter(tmp_path: Path) -> None:
    out = _printed(tmp_path, PROTOCOL, _trial_cells())
    row = out["noise=0.1"]
    for rid in ("est_a", "est_b"):
        assert rid + "_coverage" in row and rid + "_bias" in row, sorted(row)
    cells = _trial_cells()
    a = cells[0]["metrics"]["est_a"]["values"]
    assert row["est_a_bias"] == pytest.approx(sum(a) / len(a) - 2.0)


def test_a_plan_with_no_true_value_prints_no_coverage_and_no_bias(tmp_path: Path) -> None:
    plain = {**PROTOCOL, "metrics": [{k: v for k, v in m.items() if k != "known_value"} for m in PROTOCOL["metrics"]]}
    out = _printed(tmp_path, plain, _trial_cells())
    assert not [k for row in out.values() if isinstance(row, dict) for k in row if k.endswith(("_coverage", "_bias"))]


def test_the_declared_block_keeps_returned_names_apart_from_what_fi_counts(tmp_path: Path) -> None:
    from core import frozen_protocol

    extra = [
        {"id": "est_c", "kind": "mean", "estimand": "third", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "noise * 2", "estimate": "est_c", "standard_error": "se_c", "interval": ["lo_c", "hi_c"], "level": 0.95}},
        {"id": "est_d", "kind": "mean", "estimand": "fourth", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_a", "estimate": "est_d", "standard_error": "se_d"}}]
    protocol = {**PROTOCOL, "metrics": [*PROTOCOL["metrics"], *extra]}
    (tmp_path / "needs").mkdir()
    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": protocol, "sha256": frozen_protocol.sha256(protocol)}
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")
    lines = finding_repair.declared_block(tmp_path).splitlines()
    returned = next(x for x in lines if x.startswith("Names the code's trial returns"))
    assert "`est_a`" in returned and "`cover`" not in returned and "`corr_ab`" not in returned
    assert not any("channel" in x and x.startswith("Names") for x in lines)
    counted = next(x for x in lines if x.startswith("FI counts these itself"))
    assert "est_a_coverage" in counted and "est_b_bias" in counted
    assert "est_c" not in counted, "a true value that is a formula of a setting is not countable"
    assert "est_d" not in counted, "no interval ends: no coverage"
    assert "does not return them" in counted or "the code does not return them" in counted


def test_what_a_trial_returns_is_read_from_the_record_not_the_plan_flag(tmp_path: Path) -> None:
    """A plan that flags every result 'afterwards' (estimates included) while the trials returned them: the record decides."""
    from core import frozen_protocol

    metrics = [{**m, "returned": "afterwards"} for m in PROTOCOL["metrics"]]
    protocol = {**PROTOCOL, "metrics": metrics}
    (tmp_path / "needs").mkdir()
    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": protocol, "sha256": frozen_protocol.sha256(protocol)}
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")

    def returned_line() -> str:
        return next(x for x in finding_repair.declared_block(tmp_path).splitlines() if x.startswith("Names the code's trial returns"))

    before = returned_line()  # no record yet: an estimate with a true value is still what a trial returns
    assert "`est_a`" in before and "`est_b`" in before and "`cover`" not in before
    (tmp_path / "raw").mkdir()
    cell = _cell(0.1, [2.0, 2.1], [5.0, 4.9])
    cell["metrics"]["cover"] = _metric([1.0, 0.0])
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"schema": 1, "cells": [cell]}), encoding="utf-8")
    after = returned_line()
    assert "`cover`" in after and "`corr_ab`" not in after
    worked = next(x for x in finding_repair.declared_block(tmp_path).splitlines() if x.startswith("Worked out after"))
    assert "`corr_ab`" in worked and "`est_a`" not in worked and "`cover`" not in worked


def test_bias_leaves_out_a_trial_with_no_value_and_says_how_many(tmp_path: Path) -> None:
    cells = _trial_cells()
    cells[0]["metrics"]["est_a"]["values"][3] = float("nan")
    t = _table(cells)
    assert t.bias_dropped("est_a", "noise=0.1") == 1 and t.bias_dropped("est_a", "noise=0.2") == 0
    assert len(t.coverage("est_a", "noise=0.1")) == 40, "coverage counts that trial as a miss"
    out = _printed(tmp_path, PROTOCOL, cells)
    assert out["noise=0.1"]["est_a_bias_dropped"] == 1 and "est_a_bias_dropped" not in out["noise=0.2"]
