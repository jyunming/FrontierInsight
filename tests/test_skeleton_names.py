"""The frame's list of names a trial may return is ONE list: the declared results, the names FI itself reads from a trial (a
known value's estimate, error and interval ends; what a check's ``measure`` uses) and the names FI asks for. A name outside
it is refused at once. Made-up topic: a gauge's offset recovered from noisy readings."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from core import skeleton, small_run
from core.execution import SharedInterpreterExecutor

DESIGN: dict[str, Any] = {"protocol": {
    "grid": {"noise": [0.1, 0.2, 0.3], "readings": [10, 20]}, "runs_per_setting": 4, "constants": {"offset_true": 2.0},
    "metrics": [
        {"id": "est", "kind": "mean", "estimand": "the gauge offset", "unit": "one reading",
         "known_value": {"true_value": "offset_true", "estimate": "est", "standard_error": "est_se",
                         "interval": ["est_lo", "est_hi"], "level": 0.95}},
        {"id": "hit", "kind": "proportion", "estimand": "how often a reading is inside the band", "unit": "one reading"}],
    "oracles": [
        {"name": "gap", "kind": "special_case", "expected": 0, "tolerance": 1e-6, "case": {"noise": 0.1, "readings": 10},
         "measure": "max_gap_vs_reference"},
        {"name": "order", "kind": "invariant", "expected": 0, "tolerance": 1e-9, "case": {"noise": 0.1, "readings": 10},
         "measure": "max(0, est_lo - est, est - est_hi)"},
        {"name": "no_case", "kind": "invariant", "expected": 0, "tolerance": 1e-9}]}}

BODY = '''
def trial(cell, ctx):
    out = TrialResult()
    out["est"] = 2.0 + cell["noise"] * ctx.rng.random()
    out["est_se"] = 0.1
    out["est_lo"] = 1.0
    out["est_hi"] = 3.0
    out["max_gap_vs_reference"] = 0.0
    out["hit_count"] = 1
    out["hit_total"] = 1
    out["why_missing"] = {"est": "none"}
    return out
'''


def _results(text: str) -> tuple[str, ...]:
    return eval(re.search(r"^RESULTS = (.*?)  #", text, re.M).group(1))  # noqa: S307


def test_the_frame_lists_the_names_a_known_value_and_the_checks_read_from_a_trial() -> None:
    protocol = DESIGN["protocol"]
    frame = skeleton.frame_source(DESIGN, stochastic=True, strict=True)
    got = _results(frame)
    for name in ("est_se", "est_lo", "est_hi", "max_gap_vs_reference", "why_missing"):
        assert name in got
    assert got == tuple(skeleton.returnable_names(protocol))
    assert skeleton.required_names(protocol) == ["est_se", "est_lo", "est_hi", "max_gap_vs_reference"], \
        "the estimate is a declared result; a check with no case is answered by oracle(), not by a trial"


def test_these_names_are_diagnostics_never_results_of_the_plan() -> None:
    protocol = DESIGN["protocol"]
    assert skeleton.declared_ids(protocol) == ["est", "hit"]
    assert "est_se" not in skeleton.declared_names(protocol)
    assert skeleton.empty_slots(protocol, ["est_se", "est_lo", "est_hi"]) == ["est", "hit"]
    assert skeleton.unknown_names(protocol, ["est", "est_se"]) == ["est_se"], "kept as a diagnostic, never reported"


def test_the_names_of_a_formula_measure_are_all_listed() -> None:
    design = {"protocol": {**DESIGN["protocol"], "oracles": [
        {"name": "ratio", "kind": "special_case", "expected": 1, "tolerance": 0.1, "case": {"noise": 0.1},
         "measure": "log2(abs(t_coarse - t_ref) / abs(t_fine - t_ref))"}]}}
    assert skeleton.required_names(design["protocol"])[-3:] == ["t_coarse", "t_ref", "t_fine"]


def _quest(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(source, encoding="utf-8")
    (root / "code" / skeleton.RUNTIME_FILE).write_text(skeleton.RUNTIME, encoding="utf-8")
    return root


def _small(root: Path) -> small_run.SmallRun:
    return asyncio.run(small_run.run(SharedInterpreterExecutor(python_version="3.11"), sys.executable, root, "code/simulate.py",
                                     DESIGN["protocol"]["grid"], deterministic=False, timeout_s=120))


def test_a_strict_frame_runs_a_trial_that_returns_those_names(tmp_path: Path) -> None:
    text, why = skeleton.fill(skeleton.frame_source(DESIGN, stochastic=True, strict=True), BODY)
    assert text is not None, why
    got = _small(_quest(tmp_path, text))
    assert not got.crashed, got.says()
    returned = set(got.record()["returned"])
    assert {"est_se", "est_lo", "est_hi", "max_gap_vs_reference"} <= returned
    assert skeleton.empty_slots(DESIGN["protocol"], got.record()["returned"]) == []


def test_a_misspelled_result_and_a_name_nothing_reads_still_fail_the_first_trial_and_name_what_is_allowed(tmp_path: Path) -> None:
    for wrong in ("esst", "extra_curve"):
        body = BODY.replace('out["est"]', f'out["{wrong}"]')
        text, _ = skeleton.fill(skeleton.frame_source(DESIGN, stochastic=True, strict=True), body)
        assert text is not None
        got = _small(_quest(tmp_path / wrong, text))
        assert got.crashed
        assert f"`{wrong}` is not a result the plan declares" in got.directive() and "est_se" in got.directive()


def test_the_trial_note_tells_the_model_what_a_strict_frame_does_with_another_name() -> None:
    frame = skeleton.frame_source(DESIGN, stochastic=True, strict=True)
    assert "fails every trial" in frame
