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
    assert skeleton.required_names(protocol) == ["est_se", "est_lo", "est_hi", "max_gap_vs_reference",
                                                 "max(0, est_lo - est, est - est_hi)"], \
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


# ---- a measure written as words is returnable under exactly the name FI reads ------------------------------------------------

PROSE: dict[str, Any] = {"protocol": {**DESIGN["protocol"], "oracles": [
    {"name": "words", "kind": "special_case", "expected": 0, "tolerance": 1e-6, "case": {"noise": 0.1, "readings": 10},
     "measure": "error rate"},
    {"name": "hyphen", "kind": "special_case", "expected": 0, "tolerance": 1e-6, "case": {"noise": 0.1, "readings": 10},
     "measure": "p-value"},
    {"name": "settings", "kind": "special_case", "expected": 0, "tolerance": 1e-6, "case": {"noise": 0.1, "readings": 10},
     "measure": "noise * offset_true - gap"}]}}

PROSE_BODY = '''
def trial(cell, ctx):
    out = TrialResult()
    out["est"] = 2.0
    out["hit_count"] = 1
    out["hit_total"] = 1
    out["error rate"] = 0.0
    out["p-value"] = 0.5
    out["gap"] = 0.0
    out["noise"] = cell["noise"] * 2  # the check "noise * offset_true - gap" is worked out from what the trial returns
    out["offset_true"] = 2.0
    return out
'''


def test_a_measure_that_is_not_a_formula_is_listed_as_the_one_name_fi_reads() -> None:
    from core import oracle_forms

    assert oracle_forms.read_names("error rate") == ["error rate"]
    assert oracle_forms.read_names(" p-value ")[0] == "p-value"
    assert oracle_forms.read_names("gap") == ["gap"]
    assert oracle_forms.read_names("") == []
    got = skeleton.required_names(PROSE["protocol"])
    assert "error rate" in got and "p-value" in got and "gap" in got
    assert "noise" in got and "offset_true" in got, "a formula is worked out from what the trial returns: its settings too"


def test_a_strict_frame_runs_a_trial_that_returns_a_measure_written_as_words(tmp_path: Path) -> None:
    text, why = skeleton.fill(skeleton.frame_source(PROSE, stochastic=True, strict=True), PROSE_BODY)
    assert text is not None, why
    got = _small(_quest(tmp_path, text))
    assert not got.crashed, got.says()
    assert {"error rate", "p-value", "gap", "noise", "offset_true"} <= set(got.record()["returned"])
    # a setting named in RESULTS changes nothing about how settings are read: the cell still comes from SETTINGS and the grid
    assert skeleton.settings_of(PROSE["protocol"]) == ["noise", "readings"] and got.axes == ["noise", "readings"]
    assert all(s.failed == 0 for s in got.settings)


# ---- a frame written with the short list is made whole ----------------------------------------------------------------------

def _short(design: dict[str, Any], **kw: Any) -> str:
    """The frame as it was written before the names FI reads were added: RESULTS holds the declared results only."""
    frame = skeleton.frame_source(design, stochastic=True, **kw)
    full = tuple(skeleton.returnable_names(design["protocol"]))
    return frame.replace(repr(full), repr(tuple(skeleton.declared_names(design["protocol"]))), 1)


def test_a_frame_with_the_short_list_is_written_again_with_the_full_one() -> None:
    old = _short(DESIGN, strict=True)
    assert "est_se" not in old
    healed = skeleton.heal_results(old, DESIGN, stochastic=True, strict=True)
    assert healed == skeleton.frame_source(DESIGN, stochastic=True, strict=True)
    assert skeleton.heal_results(healed, DESIGN, stochastic=True, strict=True) == healed
    assert skeleton.frame_intact(skeleton.frame_for(DESIGN, stochastic=True, strict=True, existing=old), healed) == ""


def test_a_frame_of_another_plan_or_refusal_is_left_alone() -> None:
    old = _short(DESIGN, strict=True)
    other = {"protocol": {**DESIGN["protocol"], "grid": {"noise": [0.1]}}}
    assert skeleton.heal_results(old, other, stochastic=True, strict=True) == old
    assert skeleton.heal_results(old, DESIGN, stochastic=True, strict=False) == old
    assert skeleton.heal_results("x = 1", DESIGN, stochastic=True, strict=True) == "x = 1"


def test_a_repair_of_a_healed_file_is_compared_with_the_healed_frame(tmp_path: Path) -> None:
    from tests.test_skeleton import _engine_with

    design = {"protocol": {**DESIGN["protocol"], "form": "contract"}}
    old, _ = skeleton.fill(_short(design, strict=True), BODY.replace("est_se", "est_x").replace("est_lo", "est_y"))
    eng = _engine_with(tmp_path, True)
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "simulate.py").write_text(old or "", encoding="utf-8")
    state = {"design": design}
    assert eng._heal_frame(state) is True
    on_disk = (code / "simulate.py").read_text(encoding="utf-8")
    assert "'est_se'" in on_disk and "def trial" in on_disk
    assert eng._heal_frame(state) is False
    new, why = skeleton.fill(skeleton.frame_for(design, stochastic=True, strict=True, existing=on_disk), BODY)
    assert new is not None, why
    assert skeleton.repair_problem(on_disk, new, "simulate.py") == ""
    got = _small(_quest(tmp_path / "run", new))
    assert not got.crashed, got.says()


# ---- the names that are not results stay out of the results table and the writer's quantities --------------------------------

def test_the_diagnostics_of_a_contract_plan_are_not_rows_or_quantities(tmp_path: Path) -> None:
    from core import paper_values, trial_table

    contract_protocol = {**DESIGN["protocol"], "form": "contract"}
    hidden = skeleton.diagnostic_names(contract_protocol)
    assert {"est_se", "est_lo", "est_hi", "max_gap_vs_reference", "why_missing"} <= set(hidden)
    assert skeleton.diagnostic_names(DESIGN["protocol"]) == [], "a plan in the older form lists every name, as before"
    cells = [{"cell": {"noise": n}, "key": f"noise={n}",
              "metrics": {name: {"values": [1.0 + n, 2.0 + n, 3.0 + 2 * n]} for name in ("est", "est_se", "est_lo", "est_hi")}}
             for n in (0.1, 0.2)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(__import__("json").dumps({"cells": cells}), encoding="utf-8")
    shown = {r["metric"] for r in trial_table.rows(tmp_path, hidden)}
    assert shown == {"est"}
    assert {r["metric"] for r in trial_table.rows(tmp_path)} == {"est", "est_se", "est_lo", "est_hi"}
    rec = paper_values.Records(cells=[{**c, "metrics": c["metrics"]} for c in cells], protocol=contract_protocol)
    assert rec.metrics() == ["est"]
    assert paper_values.Records(cells=cells, protocol=DESIGN["protocol"]).metrics() == ["est", "est_se", "est_lo", "est_hi"]


def test_a_known_value_name_equal_to_a_constants_name_is_still_listed() -> None:
    design = {"protocol": {**DESIGN["protocol"], "constants": {"offset_true": 2.0, "est_se": 0.5}}}
    assert "est_se" in skeleton.required_names(design["protocol"])


# ---- the heal waits for the first recorded trial -----------------------------------------------------------------------------

def _healable(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    from tests.test_skeleton import _engine_with

    design = {"protocol": {**DESIGN["protocol"], "form": "contract"}}
    old, _ = skeleton.fill(_short(design, strict=True), BODY.replace("est_se", "est_x").replace("est_lo", "est_y"))
    eng = _engine_with(tmp_path, True)
    path = tmp_path / "code" / "simulate.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text(old or "", encoding="utf-8")
    return eng, {"design": design}, path


def test_the_frame_is_not_healed_once_a_trial_is_recorded(tmp_path: Path) -> None:
    import json

    eng, state, path = _healable(tmp_path)
    ledger = eng._raw_root() / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"event": "trial", "status": "ok"}) + chr(10), encoding="utf-8")
    before = path.read_bytes()
    assert eng._heal_frame(state) is False
    assert path.read_bytes() == before, "the file is not touched after a trial is recorded"
    assert not (tmp_path / "code" / "simulate.py.fi-tmp").exists()
    ledger.unlink()
    assert eng._heal_frame(state) is True and path.read_bytes() != before
    assert not (tmp_path / "code" / "simulate.py.fi-tmp").exists(), "written through a temporary file, then moved"


# ---- a metric the plan declares stays a quantity of the study ---------------------------------------------------------------

def test_a_declared_timing_metric_a_check_uses_is_still_a_quantity(tmp_path: Path) -> None:
    from core import paper_values, trial_table

    protocol = {**DESIGN["protocol"], "form": "contract", "metrics": [
        *DESIGN["protocol"]["metrics"],
        {"id": "t_run", "kind": "mean", "role": "timing", "estimand": "the run time", "unit": "one trial"}],
        "oracles": [{"name": "bound", "kind": "invariant", "expected": 0, "tolerance": 1e-9,
                     "case": {"noise": 0.1, "readings": 10}, "measure": "max(0, viol - t_run)"}]}
    assert "t_run" in skeleton.required_names(protocol) or "t_run" in skeleton.returnable_names(protocol)
    hidden = skeleton.diagnostic_names(protocol)
    assert "viol" in hidden and "t_run" not in hidden
    cells = [{"cell": {"noise": n}, "key": f"noise={n}",
              "metrics": {name: {"values": [1.0 + n, 2.0 + n, 3.0 + 2 * n]} for name in ("est", "viol", "t_run")}}
             for n in (0.1, 0.2)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(__import__("json").dumps({"cells": cells}), encoding="utf-8")
    assert {r["metric"] for r in trial_table.rows(tmp_path, hidden)} == {"est", "t_run"}
    assert paper_values.Records(cells=cells, protocol=protocol).metrics() == ["est", "t_run"]


# ---- a metric the plan says each run returns may be returned, whatever its role ---------------------------------------------

def _with_diagnostics(returned: str | None) -> dict[str, Any]:
    spec: dict[str, Any] = {"id": "drift", "kind": "mean", "role": "diagnostic", "estimand": "the drift of the readings",
                            "unit": "one trial"}
    if returned:
        spec["returned"] = returned
    return {"protocol": {**DESIGN["protocol"], "form": "contract", "metrics": [*DESIGN["protocol"]["metrics"], spec]}}


def test_a_diagnostic_the_plan_says_each_run_returns_is_returnable_but_not_required() -> None:
    protocol = _with_diagnostics("each_run")["protocol"]
    assert skeleton.each_run_names(protocol) == ["drift", "drift_count", "drift_total", "drift_values"]
    got = skeleton.returnable_names(protocol)
    assert all(n in got for n in skeleton.each_run_names(protocol)) and len(got) == len(set(got))
    assert got.index("drift") > got.index("hit_values"), "after the declared results, in the plan's order"
    assert "drift" not in skeleton.declared_ids(protocol) and "drift" not in skeleton.declared_names(protocol)
    assert "drift" not in skeleton.empty_slots(protocol, ["est", "hit"]), "a trial that leaves it out is not missing a result"
    assert "drift" not in skeleton.diagnostic_names(protocol), "a declared metric stays a quantity of the study"
    frame = skeleton.frame_source(_with_diagnostics("each_run"), stochastic=True, strict=True)
    assert "drift" in _results(frame)


def test_a_diagnostic_the_plan_does_not_say_each_run_returns_is_not_returnable() -> None:
    for returned in (None, "afterwards"):
        protocol = _with_diagnostics(returned)["protocol"]
        assert skeleton.each_run_names(protocol) == []
        assert "drift" not in skeleton.returnable_names(protocol), "as before: only a metric each run returns is let through"
    assert skeleton.returnable_names(_with_diagnostics(None)["protocol"]) == skeleton.returnable_names(DESIGN["protocol"])


def test_a_strict_frame_runs_a_trial_that_returns_the_diagnostic_and_one_that_omits_it(tmp_path: Path) -> None:
    design = _with_diagnostics("each_run")
    for name, body in (("with", BODY.replace('    out["why_missing"]', '    out["drift"] = 0.5\n    out["why_missing"]')), ("without", BODY)):
        text, why = skeleton.fill(skeleton.frame_source(design, stochastic=True, strict=True), body)
        assert text is not None, why
        got = _small(_quest(tmp_path / name, text))
        assert not got.crashed, got.says()
        assert ("drift" in got.record()["returned"]) == (name == "with")
        assert skeleton.empty_slots(design["protocol"], got.record()["returned"]) == []


def test_the_analysis_reads_the_diagnostic_from_the_record() -> None:
    """A guard, not a test of the each_run rule: the analysis already listed every declared result, of any role, as readable."""
    from core import analysis_frame

    plan = analysis_frame.plan_of(_with_diagnostics("each_run"))
    assert "drift" in plan["results"], "a body may read it from the Table"


def test_a_frame_written_before_an_each_run_diagnostic_was_added_is_healed_with_its_names() -> None:
    before = {"protocol": {**DESIGN["protocol"], "form": "contract"}}
    after = _with_diagnostics("each_run")
    old = skeleton.frame_source(before, stochastic=True, strict=True)
    assert "drift" not in old
    healed = skeleton.heal_results(old, after, stochastic=True, strict=True)
    assert healed != old and "'drift_values'" in healed
    assert healed == skeleton.frame_source(after, stochastic=True, strict=True)


def test_a_frame_is_not_healed_when_the_plan_gains_a_real_result() -> None:
    before = {"protocol": {**DESIGN["protocol"], "form": "contract"}}
    old = skeleton.frame_source(before, stochastic=True, strict=True)
    metrics = [*DESIGN["protocol"]["metrics"], {"id": "extra", "kind": "mean", "role": "result", "returned": "each_run",
                                                "estimand": "an extra number", "unit": "one trial"}]
    after = {"protocol": {**before["protocol"], "metrics": metrics}}
    assert "extra" in skeleton.declared_ids(after["protocol"])
    assert skeleton.heal_results(old, after, stochastic=True, strict=True) == old
