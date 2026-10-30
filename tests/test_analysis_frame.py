"""The analysis FI writes for a plan in the contract form (core/analysis_frame.py): FI's ``experiment.py`` and the frame of
``derive.py``. A body that returns the wrong thing gets the verdict of the spec (a NaN, a wrong type, an unknown key, a stub, a
body that raises), and one body's failure does not stop the others; a quantity with a formula is worked out by FI's calculator and
equals a hand computation; ``RESULT_JSON`` carries exactly the declared ids on each recorded scratch quest; and
``paper_values.load`` reads the per-setting quantities. Made-up topic (a colony); the recorded scratch quests are used when present."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from core import analysis_frame as af
from core import contract, numeric_warnings, paper_values, trial_runner

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path("C:/fi_runs/c1/skeleton/s4_inputs/scratch")
RECORDED = sorted(p for p in SCRATCH.iterdir() if (p / "raw" / "trials.json").is_file()) if SCRATCH.is_dir() else []
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _cell(rate: float, dose: int, growth: list[float], *, failed: int = 0) -> dict[str, Any]:
    cell = {"rate": rate, "dose": dose}
    return {"cell": cell, "key": trial_runner.cell_key(cell), "planned": len(growth) + failed, "ok": len(growth), "failed": failed,
            "metrics": {"growth": {"values": growth, "trials": list(range(len(growth)))},
                        "alive": {"values": [1.0 if v > 2 else 0.0 for v in growth], "trials": list(range(len(growth)))},
                        "wobble": {"values": [0.5 * v for v in growth], "trials": list(range(len(growth)))}}}


RECORD = {"schema": "fi.trials/v1", "thresholds": {"cutoff": 2.0},
          "cells": [_cell(0.1, 1, [1.0, 1.5, 2.5, 2.0]), _cell(0.1, 2, [3.0, 3.5, 4.5], failed=1),
                    _cell(0.2, 1, [1.25, 1.75, 2.25, 2.75, 3.25]), _cell(0.2, 2, [5.0, 6.0])]}
KEYS = [c["key"] for c in RECORD["cells"]]


def _design(derived: list[dict[str, Any]] | None = None, **protocol: Any) -> dict[str, Any]:
    return {"protocol": {
        "form": "contract", "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "in_each_run": {"mode": ["a", "b"]},
        "constants": {"base": 2.0}, "thresholds": {"cutoff": 2.0}, "shared_draws": True,
        "sub_studies": [{"name": "light", "vary": ["rate"], "hold": {"dose": 1}}],
        "metrics": [{"id": "growth", "kind": "mean", "role": "result", "reference": {"formula": "base*rate", "kind": "value"}},
                    {"id": "alive", "kind": "proportion", "role": "result", "paired": True},
                    {"id": "wobble", "kind": "mean", "role": "diagnostic"}],
        "derived": derived or [], "figures": [{"id": "fig1", "file": "growth.png", "shows": ["growth"], "x": "rate", "series": "dose"}],
        **protocol}}


def _derived(ident: str, role: str = "result", formula: str | None = None, inputs: tuple[str, ...] = ("growth",)) -> dict[str, Any]:
    out = {"id": ident, "inputs": list(inputs), "role": role}
    if formula:
        out["formula"] = formula
    return out


def _quest(tmp_path: Path, design: dict[str, Any], bodies: str = "", record: dict[str, Any] | None = None) -> Path:
    """A quest folder with FI's record, FI's files, and the study's ``bodies`` appended to derive.py (a later definition wins)."""
    quest = tmp_path / "quest"
    (quest / "raw").mkdir(parents=True)
    (quest / "raw" / "trials.json").write_text(json.dumps(record or RECORD), encoding="utf-8")
    af.ensure_files(quest / "code", design)
    if bodies:
        with (quest / "code" / "derive.py").open("a", encoding="utf-8") as f:
            f.write("\n\n" + textwrap.dedent(bodies))
    return quest


def _run(quest: Path) -> tuple[int, dict[str, Any] | None, dict[str, Any], str]:
    """Run FI's experiment.py in a fresh Python with only the code folder on its path (nothing of FI's source is named).
    Returns the exit code, the printed RESULT_JSON, derived.json and stderr."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS")}
    r = subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True, timeout=300,
                       creationflags=NO_WINDOW)
    last = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON: ")]
    printed = json.loads(last[-1][len("RESULT_JSON: "):]) if last else None
    path = quest / af.DERIVED_PATH
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return r.returncode, printed, record, r.stderr


def _ids(result: dict[str, Any]) -> set[str]:
    """The ids a RESULT_JSON carries: under each setting (a key with ``=``) and at the top, without the forms of a result."""
    found: set[str] = set()
    skip = {"n_failed", "why_missing"}
    for k, v in result.items():
        if k in skip:
            continue
        if "=" in k and isinstance(v, dict):
            for name in v:
                if name not in skip:
                    found.add(next((name[: -len(s)] for s in ("_values", "_count", "_total", "_pair_id", "_undefined") if name.endswith(s)), name))
        else:
            found.add(k)
    return found


# ---- the files FI writes ----------------------------------------------------------------------------------------------------
def test_applies_only_for_the_contract_form() -> None:
    assert af.applies(_design()) == ""
    assert "contract" in af.applies({"protocol": {**_design()["protocol"], "form": "older"}})
    assert af.applies({}) != ""


def test_derive_has_a_stub_for_every_body_and_none_for_a_formula() -> None:
    design = _design([_derived("gain", formula="growth - base"), _derived("peak"), _derived("ref_curve", "reference", inputs=()),
                      _derived("twice_base", "reference", formula="2*base", inputs=())])
    text = af.derive_source(design)
    assert sorted(af.stubs(text)) == ["derive_peak", "figure_fig1", "reference_ref_curve"]
    assert "derive_gain" not in text and "reference_twice_base" not in text
    compile(text, "derive.py", "exec")
    assert af.derive_source(design) == text  # the same plan gives the same file
    assert af.derive_intact(text, text) == ""


def test_a_rewritten_derive_keeps_the_bodies_the_study_wrote() -> None:
    design = _design([_derived("peak")])
    text = af.derive_source(design).replace('    raise NotYet("derive_peak is not written yet")', "    return 1.0")
    more = _design([_derived("peak"), _derived("late")])
    again = af.derive_source(more, text)
    assert "return 1.0" in again and "def derive_late(table)" in again
    assert sorted(af.stubs(again)) == ["derive_late", "figure_fig1"]
    assert af.derive_intact(again, again) == ""


def test_what_a_repair_may_and_may_not_change_in_derive() -> None:
    old = af.derive_source(_design([_derived("peak")]))
    body = old.replace('    raise NotYet("derive_peak is not written yet")', "    return max(table.mean('growth', k) for k in table.keys())")
    assert af.analysis_problem(old, body, "derive.py") == ""  # a body, a helper and an import are the study's
    assert af.analysis_problem(old, "import math\n" + body + "\n\ndef helper(x):\n    return x\n", "derive.py") == ""
    assert "frame FI wrote" in af.analysis_problem(old, old.replace("ANALYSIS_VERSION = 1", "ANALYSIS_VERSION = 2"), "derive.py")
    assert "frame FI wrote" in af.analysis_problem(old, old.replace("PAIRED = (", "PAIRED = ('x', "), "derive.py")
    assert "parameters" in af.analysis_problem(old, body.replace("def derive_peak(table):", "def derive_peak(table, extra):"), "derive.py")
    assert "missing" in af.analysis_problem(old, body.replace("def derive_peak(table):", "def derive_other(table):"), "derive.py")
    assert "RESULTS" in af.analysis_problem(old, body + "\nRESULTS = ()\n", "derive.py")
    assert "imported" in af.analysis_problem(old, body + '\nif __name__ == "__main__":\n    print(1)\n', "derive.py")
    assert af.analysis_problem(old, "def (", "derive.py") != ""  # not Python


def test_experiment_is_all_fis() -> None:
    text = af.EXPERIMENT
    assert af.analysis_problem(text, text, "experiment.py") == ""
    assert "FI's own file" in af.analysis_problem(text, text + "\nprint(1)\n", "experiment.py")
    assert af.analysis_problem("print(1)", "print(2)", "experiment.py") == ""  # an analysis the model wrote is not this check's
    assert af.analysis_problem("a", "b", "simulate.py") == ""


# ---- what a body returns ----------------------------------------------------------------------------------------------------
PLANTED = [  # (id, role, body) -> the verdict the spec gives
    ("good_number", "result", "return 2.5"),
    ("good_none_reason", "result", "return table.missing('no trial reached it')"),
    ("good_setting_map", "result", "return {k: table.mean('growth', k) for k in table.keys()}"),
    ("good_level_map_nan", "result", "return {'a': 1.0, 'b': float('nan')}"),
    ("nan_result", "result", "return float('nan')"),
    ("inf_reference", "reference", "return float('inf')"),
    ("nan_bound", "physical_bound", "return float('nan')"),
    ("nan_diagnostic", "diagnostic", "return float('nan')"),
    ("nan_timing", "timing", "return float('nan')"),
    ("nan_in_map", "result", "return {k: float('nan') for k in table.keys()}"),
    ("wrong_type", "result", "return 'high'"),
    ("wrong_bool", "result", "return True"),
    ("wrong_list", "result", "return [1.0, 2.0]"),
    ("none_bare", "result", "return None"),
    ("reason_empty", "result", "return table.missing('')"),
    ("unknown_key", "result", "return {'rate=0.5,dose=9': 1.0}"),
    ("level_left_out", "result", "return {'a': 1.0}"),
    ("stub", "result", "raise NotYet('stub is not written yet')"),
    ("raises", "result", "return table.mean('growth', 'nowhere')"),
    ("reads_unknown_id", "result", "return table.mean('not_declared', table.keys()[0])"),
]


def _planted_design(items: list[tuple[str, str, str]] = PLANTED) -> tuple[dict[str, Any], str]:
    design = _design([_derived(i, role, inputs=()) for i, role, _body in items])
    bodies = "".join((f"def reference_{i}(setting):\n    {body}\n\n\n" if role == "reference"
                      else f"def derive_{i}(table):\n    {body}\n\n\n") for i, role, body in items if body != "stub")
    return design, bodies


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> tuple[int, dict[str, Any] | None, dict[str, Any], str]:
    design, bodies = _planted_design()
    return _run(_quest(tmp_path_factory.mktemp("planted"), design, bodies))


def _status(planted: Any, ident: str) -> dict[str, Any]:
    return planted[2]["derived"][ident]


def test_one_bodys_failure_stops_no_other(planted: Any) -> None:
    code, printed, record, _err = planted
    assert code == 1  # a counting body failed
    assert printed is not None
    got = record["derived"]
    assert {i: got[i]["status"] for i in got} == {
        "good_number": "ok", "good_none_reason": "ok", "good_setting_map": "ok", "good_level_map_nan": "ok",
        "nan_result": "failed", "inf_reference": "failed", "nan_bound": "failed", "nan_diagnostic": "ok", "nan_timing": "ok",
        "nan_in_map": "failed", "wrong_type": "refused", "wrong_bool": "refused", "wrong_list": "refused", "none_bare": "refused",
        "reason_empty": "refused", "unknown_key": "refused", "level_left_out": "refused", "stub": "not_written", "raises": "failed",
        "reads_unknown_id": "failed"}
    assert printed["good_number"] == 2.5  # the good bodies are printed beside the failed ones


def test_the_good_returns_are_kept_as_they_are(planted: Any) -> None:
    code, printed, record, _err = planted
    assert _status(planted, "good_number")["value"] == 2.5
    assert _status(planted, "good_none_reason")["value"] is None
    assert "no trial reached it" in json.dumps(_status(planted, "good_none_reason"))
    by_setting = _status(planted, "good_setting_map")["by_setting"]
    assert set(by_setting) == set(KEYS)
    assert by_setting[KEYS[0]] == pytest.approx(sum([1.0, 1.5, 2.5, 2.0]) / 4)
    level = _status(planted, "good_level_map_nan")["by_level"]
    assert level == {"input": "mode", "values": {"a": 1.0, "b": None}}  # NaN where the result does not exist is the one exception
    assert printed[KEYS[0]]["good_setting_map"] == pytest.approx(by_setting[KEYS[0]])
    assert printed["good_level_map_nan"] == {"a": 1.0, "b": None}


def test_a_nan_follows_the_role_and_is_said_not_printed(planted: Any) -> None:
    printed = planted[1]
    for ident in ("nan_result", "inf_reference", "nan_bound", "nan_in_map"):
        assert _status(planted, ident)["status"] == "failed" and "not a number" in _status(planted, ident)["why"]
        assert printed[ident] is None
    for ident in ("nan_diagnostic", "nan_timing"):
        assert _status(planted, ident)["status"] == "ok" and _status(planted, ident)["value"] is None
        assert "nan" in " ".join(_status(planted, ident)["disclosed"]) and printed[ident] is None
    json.dumps(printed, allow_nan=False)  # raises on a NaN or an infinity


def test_the_refusals_say_what_to_do(planted: Any) -> None:
    why = {i: _status(planted, i)["why"] for i in ("wrong_type", "wrong_bool", "wrong_list", "none_bare", "reason_empty",
                                                   "unknown_key", "level_left_out")}
    assert "must return a number, None with a reason, or a mapping of setting keys; it returned str" in why["wrong_type"]
    assert "returned bool" in why["wrong_bool"] and "returned list" in why["wrong_list"]
    assert "without saying why" in why["none_bare"] and "no reason" in why["reason_empty"]
    assert "`rate=0.5,dose=9`" in why["unknown_key"] and KEYS[0] in why["unknown_key"]  # names the key and the known ones
    assert "leaves out b" in why["level_left_out"]


def test_a_stub_and_a_body_that_raises_are_said_apart(planted: Any) -> None:
    assert _status(planted, "stub")["status"] == "not_written"
    raised = _status(planted, "raises")
    assert raised["status"] == "failed" and "derive.py line" in raised["why"] and not raised.get("fi_fault")
    assert "not_declared" in _status(planted, "reads_unknown_id")["why"] and "growth" in _status(planted, "reads_unknown_id")["why"]
    # the stderr names what failed, not what worked (apart from FI's line naming each step as it begins it)
    said = "\n".join(ln for ln in planted[3].splitlines() if not ln.startswith(af.STEP_MARK))
    assert "good_number" not in said and af.STEP_MARK + "derive_good_number" in planted[3]
    assert "stub is computed by no trial" in planted[3]


def test_only_a_diagnostic_failing_exits_zero(tmp_path: Path) -> None:
    items = [("diag_nan", "diagnostic", "return float('nan')"), ("diag_raises", "diagnostic", "raise ValueError('x')"),
             ("fine", "result", "return 1.0")]
    design, bodies = _planted_design(items)
    code, printed, record, _err = _run(_quest(tmp_path, design, bodies))
    assert code == 0 and printed["fine"] == 1.0
    assert record["derived"]["diag_raises"]["status"] == "failed"


def test_a_stub_alone_is_not_an_error(tmp_path: Path) -> None:
    code, printed, record, err = _run(_quest(tmp_path, _design([_derived("peak")])))
    assert code == 0 and printed["peak"] is None and "not written" in printed["why_missing"]["peak"]
    assert record["derived"]["peak"]["status"] == "not_written"
    assert "peak is computed by no trial and no function of the study yet" in err


# ---- the verdicts of the NaN probe (-172(b)) --------------------------------------------------------------------------------
@pytest.mark.parametrize("role", contract.ROLES)
def test_a_nan_counts_exactly_for_the_roles_numeric_warnings_stops_for(tmp_path: Path, role: str) -> None:
    """The probe's ``declared`` case: a NaN planted in a declared quantity counts for a result, a reference and a physical bound
    and is disclosed for any other role, the same answer as ``numeric_warnings`` gives the printed results."""
    design, bodies = _planted_design([("planted", role, "return float('nan')")])
    code, _printed, record, _err = _run(_quest(tmp_path, design, bodies))
    protocol = {"metrics": [{"id": "planted", "role": role}]}
    stops = bool(numeric_warnings.scan("", {"planted": float("nan")}, results=numeric_warnings.declared_results(protocol)))
    assert (record["derived"]["planted"]["status"] == "failed") == stops == (role in numeric_warnings.COUNTING_ROLES)
    assert code == (1 if stops else 0)


def test_the_probes_headline_and_loose_cases_stop_and_the_diagnostic_one_goes_on(tmp_path: Path) -> None:
    """``headline`` and ``v3_loose`` (a NaN in the declared result itself, at one setting or at every setting) stop; ``diagnostic``
    and ``f33b`` (a quantity that is the measure of a check of a violation kind, a diagnostic by structure) go on."""
    design = _design([_derived("headline", inputs=()), _derived("loose", inputs=()), _derived("diagnostic", "diagnostic", inputs=()),
                      _derived("violation", "diagnostic", inputs=())])
    bodies = ("def derive_headline(table):\n    return {k: (float('nan') if i == 0 else 1.0) for i, k in enumerate(table.keys())}\n\n"
              "def derive_loose(table):\n    return {k: float('nan') for k in table.keys()}\n\n"
              "def derive_diagnostic(table):\n    return {k: float('nan') for k in table.keys()}\n\n"
              "def derive_violation(table):\n    return float('nan')\n")
    _code, _printed, record, _err = _run(_quest(tmp_path, design, bodies))
    verdict = {i: ("stop" if r["status"] == "failed" else "go_on") for i, r in record["derived"].items()}
    assert verdict == {"headline": "stop", "loose": "stop", "diagnostic": "go_on", "violation": "go_on"}
    # the same planted numbers, as the probe judges them
    protocol = {"metrics": [{"id": "headline"}, {"id": "loose"}, {"id": "diagnostic", "role": "diagnostic"},
                            {"id": "violation", "role": "diagnostic"}]}
    results = numeric_warnings.declared_results(protocol)
    assert {i: bool(numeric_warnings.scan("", {i: float("nan")}, results=results)) for i in verdict} == {
        i: v == "stop" for i, v in verdict.items()}


# ---- a quantity with a formula ----------------------------------------------------------------------------------------------
def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_a_formula_equals_a_hand_computation(tmp_path: Path) -> None:
    design = _design([_derived("gain", formula="growth - base"), _derived("twice_base", "reference", formula="2*base", inputs=()),
                      _derived("lift", formula="gain * rate + cutoff", inputs=("gain",)),
                      _derived("wide", formula="min(growth, 3) + ellipk(0.5)")])
    code, printed, record, _err = _run(_quest(tmp_path, design))
    assert code == 0
    got = record["derived"]
    assert got["twice_base"]["value"] == 4.0 and printed["twice_base"] == 4.0  # a constant: one number for the study
    for c in RECORD["cells"]:
        mean = _mean(c["metrics"]["growth"]["values"])
        gain = mean - 2.0
        assert got["gain"]["by_setting"][c["key"]] == pytest.approx(gain, abs=1e-12)
        assert got["lift"]["by_setting"][c["key"]] == pytest.approx(gain * c["cell"]["rate"] + 2.0, abs=1e-12)
        assert printed[c["key"]]["gain"] == pytest.approx(gain, abs=1e-12)
        assert got["wide"]["by_setting"][c["key"]] == pytest.approx(min(mean, 3) + 1.8540746773013719, abs=1e-12)  # K(m=0.5)
    assert got["gain"]["inputs"] == ["growth"]


def test_a_formula_needs_nothing_of_fis_source_and_the_copy_is_cores(tmp_path: Path) -> None:
    files = af.carried_files()
    assert sorted(files) == ["fi_core/__init__.py", "fi_core/oracle_check.py", "fi_core/oracle_forms.py", "fi_figures.py"]
    for name in af.CORE_COPIES:
        assert files[f"fi_core/{name}"] == (ROOT / "core" / name).read_text(encoding="utf-8")
    assert files["fi_figures.py"] == (ROOT / "core" / "analysis_figures.py").read_text(encoding="utf-8")
    quest = _quest(tmp_path, _design([_derived("gain", formula="growth - base")]))
    assert all((quest / "code" / f).is_file() for f in files)
    code, _printed, record, _err = _run(quest)  # a fresh Python that knows no FI folder
    assert code == 0 and record["derived"]["gain"]["status"] == "ok"


def test_a_missing_calculator_is_fis_fault_not_the_models(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design([_derived("gain", formula="growth - base")]))
    shutil.rmtree(quest / "code" / af.CORE_DIR)
    code, printed, _record, err = _run(quest)
    assert code == 2 and printed is None and "formula calculator" in err and "not the study's" in err


def test_a_formula_over_a_name_nobody_gives_is_left_out_of_the_plan_and_fails_in_words_at_run_time(tmp_path: Path) -> None:
    # Found when the plan is read: the quantity is left out, with a gap that says why, before anything runs.
    design = _design([_derived("odd", formula="growth * nothing_here"), _derived("fine", formula="growth * base")])
    kept = contract.from_design(design, fi_analysis=True)
    assert [d.id for d in kept.derived] == ["fine"]
    assert any("`odd` is left out" in g and "nothing_here" in g for g in kept.gaps)
    # Should one reach the run all the same (a derive.py written before this check), it is the plan's fault, said plainly.
    late = 'DERIVED["odd"] = {"by": "formula", "formula": "growth * nothing_here", "inputs": ["growth"], "role": "result"}\n'
    code, _printed, record, err = _run(_quest(tmp_path, design, late))
    odd = record["derived"]["odd"]
    assert odd["status"] == "failed" and "nothing_here" in odd["why"] and odd["fi_fault"]  # a plan fault: FI's, not a body's
    assert code == 2 and "FI's fault" in err and record["fi_faults"]


def test_a_formula_over_the_plans_fixed_values_is_worked_out_and_one_over_what_no_setting_holds_is_left_out(
        tmp_path: Path) -> None:
    # A numerical setting is one value the whole study runs at (a number, or the "search" value of its finer checks): the
    # analysis is given it, as it is given a constant. A value compared inside each run, or a numerical setting with no one
    # value, is no number of a setting: a formula over it is left out when the plan is read, not stopped on after the run.
    design = _design([_derived("per_step", formula="growth / dt"), _derived("per_mesh", formula="growth * mesh"),
                      _derived("by_mode", formula="growth * mode"), _derived("by_tol", formula="growth * tol"),
                      _derived("mid", formula="mean(growth, base)")],
                     numerical_settings={"dt": 0.1, "mesh": {"search": 0.5, "check": [0.25]}, "tol": {"check": [0.001]}})
    kept = contract.from_design(design, fi_analysis=True)
    assert [d.id for d in kept.derived] == ["per_step", "per_mesh", "mid"]
    for ident in ("by_mode", "by_tol"):
        assert any(f"`{ident}` is left out" in g and "which the analysis is not given" in g for g in kept.gaps), kept.gaps
    fixed = af.plan_of(design)["fixed"]
    assert fixed["dt"] == 0.1 and fixed["mesh"] == 0.5 and "tol" not in fixed
    code, _printed, record, _err = _run(_quest(tmp_path, design))
    assert code == 0 and not record["fi_faults"]
    for c in RECORD["cells"]:
        mean = _mean(c["metrics"]["growth"]["values"])
        assert record["derived"]["per_step"]["by_setting"][c["key"]] == pytest.approx(mean / 0.1)
        assert record["derived"]["per_mesh"]["by_setting"][c["key"]] == pytest.approx(mean * 0.5)
        assert record["derived"]["mid"]["by_setting"][c["key"]] == pytest.approx((mean + 2.0) / 2)


def test_a_summary_of_one_number_per_setting_is_left_out_of_the_plan_and_never_worked_out_as_zero(tmp_path: Path) -> None:
    # At a setting the analysis has one number of each result (that setting's mean): max(growth) is growth itself, so
    # max(growth) - min(growth) would be 0 at every setting, with nothing said. It is asked about when the plan is read.
    design = _design([_derived("spread", "diagnostic", formula="max(growth) - min(growth)"),
                      _derived("fine", formula="max(growth, base) - min(growth, base)")])
    kept = contract.from_design(design, fi_analysis=True)
    assert [d.id for d in kept.derived] == ["fine"]
    assert any("`spread` is left out" in g and "`max(growth)` is a maximum over one number per setting, which is that number; "
               "say whether it is over the trials of a setting or over the settings" in g for g in kept.gaps), kept.gaps
    assert "spread" not in {d["id"] for d in af.plan_of(design)["derived"]}
    assert [d.id for d in contract.from_design(design).derived] == ["spread", "fine"]  # without FI's analysis: as before
    # One that reaches the run all the same (a derive.py written before this check) is not worked out, and FI says so.
    late = ('DERIVED["spread"] = {"by": "formula", "formula": "max(growth) - min(growth)", "inputs": ["growth"], '
            '"role": "diagnostic"}\n')
    code, _printed, record, err = _run(_quest(tmp_path, design, late))
    spread = record["derived"]["spread"]
    assert spread["status"] == "failed" and spread["fi_fault"] and "one number per setting" in spread["why"]
    assert spread.get("value") is None and not spread.get("by_setting")
    assert code == 2 and "FI's fault" in err
    assert record["derived"]["fine"]["status"] == "ok"


def test_formulas_that_name_each_other_in_a_circle_are_left_out_of_the_plan() -> None:
    kept = contract.from_design(_design([_derived("a", formula="b + 1"), _derived("b", formula="a * 2"),
                                         _derived("c", formula="growth / 2")]), fi_analysis=True)
    assert [d.id for d in kept.derived] == ["c"]
    assert any("`a`, `b` are left out" in g and "circle" in g for g in kept.gaps)


def test_formulas_are_worked_out_after_what_they_name_whatever_order_the_plan_lists_them_in(tmp_path: Path) -> None:
    design = _design([_derived("half", formula="peak / 2", inputs=("peak",)), _derived("peak")])
    bodies = """
    def derive_peak(table):
        return max(table.mean("growth", k) for k in table.keys())
    """
    code, _printed, record, _err = _run(_quest(tmp_path, design, bodies))
    assert code == 0 and record["derived"]["half"]["status"] == "ok"
    assert record["derived"]["half"]["value"] == pytest.approx(record["derived"]["peak"]["value"] / 2)
    assert list(record["derived"]) == ["half", "peak"]  # the record keeps the plan's order


def test_a_value_a_body_says_does_not_exist_leaves_a_formula_over_it_missing_never_fis_fault(tmp_path: Path) -> None:
    design = _design([_derived("peak"), _derived("half", formula="peak / 2", inputs=("peak",))])
    bodies = """
    def derive_peak(table):
        return table.missing("the growth never peaks inside the range studied")
    """
    code, _printed, record, _err = _run(_quest(tmp_path, design, bodies))
    half = record["derived"]["half"]
    assert code == 0 and half["status"] == "ok" and half["value"] is None and not half.get("fi_fault")
    assert "never peaks" in half["missing"][""] and not record["fi_faults"]


def test_a_formula_over_a_body_that_failed_is_blocked_and_the_repair_is_the_bodys(tmp_path: Path) -> None:
    design = _design([_derived("peak"), _derived("half", formula="peak / 2", inputs=("peak",))])
    bodies = """
    def derive_peak(table):
        return 1 / 0
    """
    code, _printed, record, err = _run(_quest(tmp_path, design, bodies))
    half = record["derived"]["half"]
    assert code == 1 and half["status"] == "blocked" and half["cause"] == "derive_peak" and not half.get("fi_fault")
    got = af.failure(record, err, code, True)
    assert got["kind"] == "body" and got["body"] == "derive_peak" and "ZeroDivisionError" in got["why"]
    assert [i["status"] for i in record["not_computed"] if i["id"] == "half"] == ["blocked"]


def test_what_a_body_raises_inside_fis_table_is_the_bodys_and_its_time_and_warnings_are_kept(tmp_path: Path) -> None:
    design = _design([_derived("peak"), _derived("calm")])
    bodies = """
    import warnings


    def derive_peak(table):
        return max(table.mean("growth", table.setting(k)) for k in table.keys())  # a setting given where its key goes


    def derive_calm(table):
        warnings.warn("overflow encountered in exp", RuntimeWarning)
        return 1.0
    """
    code, _printed, record, err = _run(_quest(tmp_path, design, bodies))
    peak = record["derived"]["peak"]
    assert code == 1 and peak["status"] == "failed" and not peak.get("fi_fault") and not record["fi_faults"]
    assert "derive.py line" in peak["why"] and "over what `derive_peak` gave it" in peak["why"]
    assert af.failure(record, err, code, True)["body"] == "derive_peak"
    calm = record["bodies"]["derive_calm"]
    assert calm["outputs"] == ["calm"] and calm["warnings"] == ["RuntimeWarning: overflow encountered in exp"]
    assert "RuntimeWarning: overflow encountered in exp" in err  # the numeric-warning scan still reads it
    assert record["bodies"]["figure_fig1"]["outputs"] == ["figure:fig1"]


def test_an_analysis_stopped_while_a_body_runs_is_laid_to_that_body(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design([_derived("slow")]), """
    import time


    def derive_slow(table):
        time.sleep(60)
        return 1.0
    """)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS")}
    proc = subprocess.Popen([sys.executable, "experiment.py"], cwd=quest / "code", env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, creationflags=NO_WINDOW)
    try:
        _out, err = proc.communicate(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        _out, err = proc.communicate()
    assert af.running_step(err) == "derive_slow" and af.running_step("", quest) == "derive_slow"
    got = af.failure({}, err, proc.returncode, False, timed_out=True)
    assert got["kind"] == "body" and got["body"] == "derive_slow" and "ran past the time allowed" in got["why"]
    # FI's own step running when it stopped is FI's; a stop before any step (the import of derive.py) is the module's
    assert af.failure({}, af.STEP_MARK + "fi: writing the results\n", 1, False)["kind"] == "fi"
    assert af.failure({}, 'File "x/derive.py", line 3\nNameError: name', 1, False)["kind"] == "module"


def test_fis_own_step_that_prints_the_results_is_said_under_its_own_name(tmp_path: Path) -> None:
    # A stop there (its time ran out) is FI's own step, never the last function of the study's that ran before it.
    quest = _quest(tmp_path, _design([_derived("peak")]), """
    def derive_peak(table):
        return max(table.mean("growth", k) for k in table.keys())
    """)
    code, _printed, _record, err = _run(quest)
    steps = [ln[len(af.STEP_MARK):].strip() for ln in err.splitlines() if ln.startswith(af.STEP_MARK)]
    assert code == 0 and steps[steps.index("derive_peak") + 1] == "fi: printing the results", steps
    got = af.failure({}, af.STEP_MARK + "derive_peak\n" + af.STEP_MARK + "fi: printing the results\n", -9, False, timed_out=True)
    assert got["kind"] == "fi" and "printing the results" in got["why"]


def test_a_file_of_what_was_running_left_by_a_killed_run_is_not_read_as_this_runs(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design([_derived("peak")]), """
    def derive_peak(table):
        return 1.0
    """)
    stale = quest / af.RUNNING_PATH
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({"step": "derive_peak", "run": "a-run-that-was-killed"}), encoding="utf-8")
    with (quest / "code" / "derive.py").open("a", encoding="utf-8") as f:
        f.write("\nthis_name_is_defined_nowhere\n")  # derive.py now stops when it is imported, before any step is said
    code, _printed, record, err = _run(quest)
    assert code == 1 and stale.is_file() and af.running_step(err, quest) == ""
    got = af.failure(record, err, code, False, step=af.running_step(err, quest))
    assert got["kind"] == "module", got
    # With no output at all (lost when the run was stopped) the file is all there is of this run, and it is read.
    assert af.running_step("", quest) == "derive_peak"


def test_a_body_that_raises_keyboardinterrupt_is_that_bodys_own_fault(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design([_derived("peak")]), """
    def derive_peak(table):
        raise KeyboardInterrupt
    """)
    code, _printed, record, err = _run(quest)
    peak = record["derived"]["peak"]
    assert code == 1 and peak["status"] == "failed" and "KeyboardInterrupt" in peak["why"] and not peak.get("fi_fault")
    got = af.failure(record, err, code, True)
    assert got["kind"] == "body" and got["body"] == "derive_peak", got


def test_a_setting_with_no_value_is_missing_with_its_reason(tmp_path: Path) -> None:
    record = copy.deepcopy(RECORD)
    record["cells"][3]["metrics"]["growth"] = {"values": [], "trials": []}
    _code, _printed, got, _err = _run(_quest(tmp_path, _design([_derived("gain", formula="growth - base")]), record=record))
    gain = got["derived"]["gain"]
    assert KEYS[3] not in gain["by_setting"] and "no trial of this setting gave `growth`" in gain["missing"][KEYS[3]]


def test_the_reference_of_a_result_is_fis_formula_and_a_reference_body_is_called_per_setting(tmp_path: Path) -> None:
    design = _design([_derived("ref_curve", "reference", inputs=()), _derived("shown")])
    bodies = """
    def reference_ref_curve(setting):
        return 10 * setting["rate"] + setting["dose"] if setting["dose"] == 1 else None


    def derive_shown(table):
        return {k: table.reference("growth", k) for k in table.keys()}
    """
    _code, _printed, record, _err = _run(_quest(tmp_path, design, bodies))
    ref = record["derived"]["ref_curve"]
    assert ref["by_setting"][KEYS[0]] == pytest.approx(2.0) and ref["by_setting"][KEYS[1]] is None
    assert ref["by"] == "reference_ref_curve"
    assert record["derived"]["shown"]["by_setting"] == {k: pytest.approx(2.0 * c["cell"]["rate"]) for k, c in zip(KEYS, RECORD["cells"])}


def test_a_body_sees_the_sub_studies_and_what_it_served_is_recorded(tmp_path: Path) -> None:
    bodies = """
    def derive_peak(table):
        keys = table.part("light")
        return max(table.mean("growth", k) for k in keys) + sum(1 for _ in table.values("wobble", keys[0])) * 0
    """
    _code, _printed, record, _err = _run(_quest(tmp_path, _design([_derived("peak")]), bodies))
    peak = record["derived"]["peak"]
    assert peak["value"] == pytest.approx(_mean(RECORD["cells"][2]["metrics"]["growth"]["values"]))
    assert peak["inputs"] == ["growth", "wobble"]


# ---- RESULT_JSON ------------------------------------------------------------------------------------------------------------
def test_result_json_has_the_shape_the_analysis_was_always_asked_for(tmp_path: Path) -> None:
    design = _design([_derived("gain", formula="growth - base"), _derived("twice_base", "reference", formula="2*base", inputs=())])
    _code, printed, _record, _err = _run(_quest(tmp_path, design))
    row = printed[KEYS[1]]  # the setting that lost a trial
    assert row["growth_values"] == [3.0, 3.5, 4.5] and row["n_failed"] == 1
    assert row["alive_count"] == 3 and row["alive_total"] == 3 and row["alive_pair_id"] == [0, 1, 2]  # a proportion, a paired one
    assert "growth_count" not in row and "growth_pair_id" not in row and "wobble_pair_id" not in row
    assert _ids(printed) == {"growth", "alive", "wobble", "gain", "twice_base"}
    assert list(printed)[-1] != "RESULT_JSON"


def test_result_json_values_equal_the_record(tmp_path: Path) -> None:
    _code, printed, _record, _err = _run(_quest(tmp_path, _design()))
    for c in RECORD["cells"]:
        assert printed[c["key"]]["growth"] == pytest.approx(_mean(c["metrics"]["growth"]["values"]), abs=1e-12)
        assert printed[c["key"]]["growth_values"] == c["metrics"]["growth"]["values"]


@pytest.mark.skipif(not RECORDED, reason="the recorded scratch quests are not here")
@pytest.mark.parametrize("scratch", RECORDED, ids=lambda p: p.name)
def test_experiment_on_a_recorded_quest_gives_exactly_the_declared_ids(tmp_path: Path, scratch: Path) -> None:
    design = json.loads((scratch / "design.json").read_text(encoding="utf-8"))
    quest = tmp_path / "quest"
    shutil.copytree(scratch / "raw", quest / "raw")  # the scratch folder itself is never written
    af.ensure_files(quest / "code", design)
    code, printed, record, err = _run(quest)
    assert code == 0, err[-800:]
    c = contract.from_design(design)
    assert _ids(printed) == {r.id for r in c.results} | {d.id for d in c.derived}
    stubbed = {d.id for d in c.derived if not (d.formula or "").strip()} | {r.id for r in c.results if r.returned == "afterwards"}
    assert {i for i, r in record["derived"].items() if r["status"] == "not_written"} == stubbed
    assert (quest / af.DERIVED_PATH).is_file() and not (scratch / "data").exists()


# ---- paper_values reads the per-setting quantities ---------------------------------------------------------------------------
def _quest_with_derived(tmp_path: Path, derived: dict[str, Any]) -> Path:
    quest = tmp_path / "q"
    (quest / "raw").mkdir(parents=True)
    (quest / "raw" / "trials.json").write_text(json.dumps(RECORD), encoding="utf-8")
    (quest / af.DERIVED_PATH).parent.mkdir(parents=True)
    (quest / af.DERIVED_PATH).write_text(json.dumps({"schema": af.DERIVED_SCHEMA, "derived": derived}), encoding="utf-8")
    return quest


def test_paper_values_load_reads_a_per_setting_quantity_as_a_metric_of_one_value(tmp_path: Path) -> None:
    derived = {"gain": {"status": "ok", "by_setting": {KEYS[0]: 0.5, KEYS[1]: None, KEYS[2]: 0.25}},
               "growth": {"status": "ok", "by_setting": {KEYS[0]: 99.0}},  # a trial returned it: its own record stays
               "study_wide": {"status": "ok", "value": 4.0},
               "broken": {"status": "failed", "by_setting": {KEYS[0]: 1.0}},
               "nan": {"status": "ok", "by_setting": {KEYS[0]: float("nan")}}}
    rec = paper_values.load(_quest_with_derived(tmp_path, derived))
    assert rec is not None
    got = {c["key"]: c["metrics"]["gain"] for c in rec.carrying("gain")}
    assert got == {KEYS[0]: {"values": [0.5], "trials": [0], "derived": True}, KEYS[2]: {"values": [0.25], "trials": [0], "derived": True}}
    assert "gain" in rec.metrics() and not {"study_wide", "broken", "nan"} & set(rec.metrics())
    assert rec.carrying("growth")[0]["metrics"]["growth"]["values"] == RECORD["cells"][0]["metrics"]["growth"]["values"]
    m = paper_values._interval(rec, rec.carrying("gain")[:1], "gain")
    assert m[0] == 0.5 and m[1] is None and m[2] is None  # one value: no interval


def test_paper_values_load_without_the_file_is_as_before(tmp_path: Path) -> None:
    quest = tmp_path / "q"
    (quest / "raw").mkdir(parents=True)
    (quest / "raw" / "trials.json").write_text(json.dumps(RECORD), encoding="utf-8")
    rec = paper_values.load(quest)
    assert rec is not None and sorted(rec.metrics()) == ["alive", "growth", "wobble"]
    (quest / af.DERIVED_PATH).parent.mkdir(parents=True)
    (quest / af.DERIVED_PATH).write_text("not json", encoding="utf-8")
    assert sorted(paper_values.load(quest).metrics()) == ["alive", "growth", "wobble"]


# ---- FI's own faults and older quests ------------------------------------------------------------------------------------------
def test_a_missing_record_stops_in_plain_words_and_is_not_a_body_fault(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design([_derived("peak")]), "def derive_peak(table):\n    return 1.0\n")
    (quest / "raw" / "trials.json").write_text("not json", encoding="utf-8")
    code, printed, record, err = _run(quest)
    assert code == 2 and printed is None and not record
    assert "FI's record of the trials" in err and "not the study's code" in err and "Traceback" not in err


def test_the_record_the_run_names_is_the_one_read(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design())
    other = copy.deepcopy(RECORD)
    other["cells"] = other["cells"][:1]
    (quest / "raw" / "trials.analysis.json").write_text(json.dumps(other), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT")}
    env.update({"FI_TRIALS": "raw/trials.analysis.json"})
    r = subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True,
                       creationflags=NO_WINDOW)
    named = json.loads([ln for ln in r.stdout.splitlines() if ln.startswith("RESULT_JSON: ")][-1][13:])
    assert [k for k in named if "=" in k] == KEYS[:1]
    _code, plain, _rec, _err = _run(quest)  # without the variable: FI's own raw/trials.json, never the copy beside it
    assert [k for k in plain if "=" in k] == KEYS


def test_a_quest_with_an_older_frame_keeps_its_analysis(tmp_path: Path) -> None:
    older = "# >>> FI frame, version 2: written by FI from the plan; a repair does not change it\n"
    assert "version 2" in af.frame_problem(older)
    assert af.ensure_files(tmp_path / "code", _design(), older) == []
    assert not (tmp_path / "code" / "experiment.py").exists()
    assert af.frame_problem("") == "" and af.ensure_files(tmp_path / "code2", _design(), "") != []


def test_levels_with_an_unknown_axis_is_a_plain_error() -> None:
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    cell = {"rate": 0.1}
    rows = [{"cell": {**cell, "mode": m}, "level": {"mode": m}, "metrics": {"growth": {"values": [1.0, 2.0]}}} for m in ("a", "b")]
    base = {"cell": cell, "key": "rate=0.1", "metrics": {"growth": {"values": [1.0]}}}
    table = ns["Table"]([base, *rows], ["growth"])
    assert table.levels("growth", "mode", "rate=0.1") == {"a": 1.5, "b": 1.5}
    with pytest.raises(ValueError, match="no recorded value"):
        table.levels("growth", "mode", "rate=0.1", nonsense=1)


# ---- the review round -----------------------------------------------------------------------------------------------------------
def _levels_record() -> dict[str, Any]:
    """One setting whose every run compares two things inside it (mode and size): a record for each combination."""
    base = {"rate": 0.1}
    cells = [{"cell": base, "key": "rate=0.1", "planned": 4, "ok": 4, "failed": 0, "metrics": {}}]
    for mode in ("a", "b"):
        for size in (1, 2):
            level = {"mode": mode, "size": size}
            cells.append({"cell": {**base, **level}, "level": level, "key": "", "planned": 4, "ok": 4, "failed": 0,
                          "metrics": {"growth": {"values": [float(size), float(size) + (1 if mode == "b" else 0)]}}})
    return {"schema": "fi.trials/v1", "thresholds": {}, "cells": cells}


def test_two_compared_factors_print_a_value_for_every_combination(tmp_path: Path) -> None:
    design = _design(grid={"rate": [0.1]}, in_each_run={"mode": ["a", "b"], "size": [1, 2]})
    code, printed, record, err = _run(_quest(tmp_path, design, record=_levels_record()))
    assert code == 0, err
    assert printed["rate=0.1"]["growth"] == {"mode=a,size=1": 1.0, "mode=a,size=2": 2.0, "mode=b,size=1": 1.5, "mode=b,size=2": 2.5}
    assert "growth" not in {r["id"] for r in record["not_computed"]}


def test_one_results_shape_never_stops_the_others(tmp_path: Path) -> None:
    rec = _levels_record()
    for c in rec["cells"][1:]:
        c["metrics"]["alive"] = {"values": [1.0, 0.0]}
    rec["cells"][0]["metrics"] = {"wobble": {"values": [1.0, 2.0]}}  # a result with its own values beside the per-level ones
    design = _design(grid={"rate": [0.1]}, in_each_run={"mode": ["a", "b"], "size": [1, 2]})
    code, printed, _record, _err = _run(_quest(tmp_path, design, record=rec))
    assert code == 0 and printed["rate=0.1"]["wobble"] == 1.5 and set(printed["rate=0.1"]["alive"]) == set(printed["rate=0.1"]["growth"])


def test_a_result_the_study_works_out_afterwards_has_a_body_and_a_stub_shows(tmp_path: Path) -> None:
    design = _design()
    design["protocol"]["metrics"].append({"id": "slope", "kind": "mean", "role": "result", "returned": "afterwards"})
    quest = _quest(tmp_path, design)
    assert "derive_slope" in af.stubs((quest / "code" / "derive.py").read_text(encoding="utf-8"))
    code, printed, record, _err = _run(quest)
    assert code == 0 and printed["slope"] is None and record["derived"]["slope"]["status"] == "not_written"
    assert any(i["id"] == "slope" and i["status"] == "not_written" for i in record["not_computed"])
    assert record["complete"] is False
    derive = quest / "code" / "derive.py"
    derive.write_text(derive.read_text(encoding="utf-8") + "\n\ndef derive_slope(table):\n    return 0.5\n", encoding="utf-8")
    _code, printed, record, _err = _run(quest)
    assert printed["slope"] == 0.5 and "slope" not in {i["id"] for i in record["not_computed"]}


def test_derived_json_lists_every_declared_result_that_is_missing(planted: Any, tmp_path: Path) -> None:
    listed = {i["id"]: i["status"] for i in planted[2]["not_computed"]}
    assert listed["stub"] == "not_written" and listed["wrong_type"] == "refused" and listed["raises"] == "failed"
    assert "good_number" not in listed and "nan_diagnostic" not in listed and planted[2]["complete"] is False
    rec = copy.deepcopy(RECORD)
    for c in rec["cells"]:
        c["metrics"].pop("wobble")
    _code, _printed, record, _err = _run(_quest(tmp_path, _design(), record=rec))
    assert any(i["id"] == "wobble" and i["status"] == "no_data" for i in record["not_computed"])


def test_undefined_trials_are_printed_not_dropped(tmp_path: Path) -> None:
    rec = copy.deepcopy(RECORD)
    rec["cells"][0]["metrics"]["growth"]["non_finite"] = 2
    rec["cells"][1]["metrics"]["growth"] = {"values": [], "trials": [], "non_finite": 3}  # no trial of it has a value
    _code, printed, _record, _err = _run(_quest(tmp_path, _design(), record=rec))
    assert printed[KEYS[0]]["growth_undefined"] == 2 and printed[KEYS[1]]["growth_undefined"] == 3
    assert "growth_undefined" not in printed[KEYS[2]]


def test_a_run_that_fails_after_an_accepted_run_keeps_its_figures_and_says_so(tmp_path: Path) -> None:
    bodies = """
    def figure_fig1(fig, table):
        fig.axes[0].plot([1, 2], [1, 2])
    """
    quest = _quest(tmp_path, _design(), bodies)
    code, _printed, first, err = _run(quest)
    assert code == 0, err
    figure = quest / "figures" / "growth.png"
    accepted = figure.read_bytes()
    (quest / "raw" / "trials.json").write_text("not json", encoding="utf-8")
    code, printed, record, _err = _run(quest)
    assert code == 2 and printed is None
    assert figure.read_bytes() == accepted  # the accepted figure is still there
    assert record["from_earlier_run"] is True and record["this_run"]["status"] == "stopped" and "record of the trials" in record["this_run"]["why"]
    assert record["figures_from_earlier_run"] == ["growth.png"] and record["derived"] == first["derived"]
    assert not (quest / af.PREVIOUS_PATH).exists() and not (quest / "figures" / "previous").exists()


def test_a_figure_a_later_run_did_not_draw_is_kept_and_marked(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design(), "def figure_fig1(fig, table):\n    fig.axes[0].plot([1, 2], [1, 2])\n")
    assert _run(quest)[0] == 0
    accepted = (quest / "figures" / "growth.png").read_bytes()
    (quest / "code" / "fi_figures.py").unlink()  # a quest whose analysis has no figure step
    code, _printed, record, _err = _run(quest)
    assert code == 0 and (quest / "figures" / "growth.png").read_bytes() == accepted
    assert record["figures"] == [] and record["figures_from_earlier_run"] == ["growth.png"]
    assert "kept from the earlier run" in " ".join(record["notes"])
    code, _printed, record, _err = _run(quest)  # the file is the earlier run's only until a run draws it
    assert code == 0


def test_a_figure_the_new_run_tried_and_failed_is_never_the_old_one(tmp_path: Path) -> None:
    quest = _quest(tmp_path, _design(), "def figure_fig1(fig, table):\n    fig.axes[0].plot([1, 2], [1, 2])\n")
    assert _run(quest)[0] == 0
    derive = quest / "code" / "derive.py"
    derive.write_text(derive.read_text(encoding="utf-8") + "\n\ndef figure_fig1(fig, table):\n    raise ValueError('x')\n", encoding="utf-8")
    code, _printed, record, _err = _run(quest)
    assert code == 1 and record["figures"][0]["status"] == "failed" and not (quest / "figures" / "growth.png").exists()
    assert "figures_from_earlier_run" not in record


def test_a_result_worked_out_afterwards_can_be_read_from_the_record(tmp_path: Path) -> None:
    design = _design()
    design["protocol"]["metrics"].append({"id": "slope", "kind": "mean", "role": "result", "returned": "afterwards"})
    rec = copy.deepcopy(RECORD)
    for c in rec["cells"]:
        c["metrics"]["slope"] = {"values": [2.0, 4.0], "trials": [0, 1]}
    bodies = "def derive_slope(table):\n    return sum(table.mean('slope', k) for k in table.keys())\n"
    _code, printed, record, _err = _run(_quest(tmp_path, design, bodies, record=rec))
    assert record["derived"]["slope"]["status"] == "ok" and printed["slope"] == 12.0
    assert record["derived"]["slope"]["inputs"] == ["slope"]


def test_undefined_trials_are_counted_from_the_values_when_the_record_does_not_count_them(tmp_path: Path) -> None:
    rec = copy.deepcopy(RECORD)
    rec["cells"][0]["metrics"]["growth"]["values"] = [1.0, None, float("nan"), 2.0]  # no non_finite in the record
    rec["cells"][0]["metrics"]["growth"]["trials"] = [0, 1, 2, 3]
    _code, printed, _record, _err = _run(_quest(tmp_path, _design(), record=rec))
    assert printed[KEYS[0]]["growth_undefined"] == 2 and printed[KEYS[0]]["growth_values"] == [1.0, 2.0]


def test_a_dict_that_leaves_out_settings_says_why(tmp_path: Path) -> None:
    bodies = f"""
    def derive_peak(table):
        return {{{KEYS[0]!r}: 1.0, {KEYS[1]!r}: None}}
    """
    _code, printed, record, _err = _run(_quest(tmp_path, _design([_derived("peak")]), bodies))
    peak = record["derived"]["peak"]
    assert peak["status"] == "ok" and set(peak["missing"]) == set(KEYS[1:])
    assert "no reason" in peak["missing"][KEYS[1]] and "no value for this setting" in peak["missing"][KEYS[2]]
    assert printed[KEYS[2]]["why_missing"]["peak"]


def test_a_body_that_exits_is_refused_and_what_raises_while_a_body_runs_is_the_bodys(tmp_path: Path) -> None:
    items = [("quitter", "result", "import sys\n    sys.exit(0)"), ("fine", "result", "return 1.0")]
    design, bodies = _planted_design(items)
    code, printed, record, _err = _run(_quest(tmp_path, design, bodies))
    assert record["derived"]["quitter"]["status"] == "refused" and "sys.exit" in record["derived"]["quitter"]["why"]
    assert printed["fine"] == 1.0 and code == 1
    # A body that is a function of a library (no line of derive.py at all): what it raises is still the body's, not FI's.
    design = _design([_derived("odd")])
    code, _printed, record, err = _run(_quest(tmp_path / "second", design, "from math import sqrt as derive_odd\n"))
    odd = record["derived"]["odd"]
    assert odd["status"] == "failed" and not odd.get("fi_fault") and "`derive_odd`" in odd["why"] and code == 1
    # FI's own lines failing outside every body (here its formula calculator is gone) is FI's fault: exit code 2.
    quest = _quest(tmp_path / "third", _design([_derived("half", formula="growth / 2")]))
    shutil.rmtree(quest / "code" / af.CORE_DIR)
    code, _printed, record, err = _run(quest)
    assert code == 2 and "FI's" in err and af.failure(record, err, code, False)["kind"] == "fi"


def test_the_figures_run_end_to_end_and_a_failed_body_counts(tmp_path: Path) -> None:
    bodies = """
    def figure_fig1(fig, table):
        for k in table.keys():
            fig.axes[0].plot([1, 2], [table.mean("growth", k)] * 2)
    """
    quest = _quest(tmp_path, _design(), bodies)
    code, _printed, record, err = _run(quest)
    assert code == 0, err
    assert [o["status"] for o in record["figures"]] == ["ok"] and record["figures_said"].startswith("1 of 1 figure(s) drawn")
    assert (quest / "figures" / "growth.png").is_file()
    derive = quest / "code" / "derive.py"
    derive.write_text(derive.read_text(encoding="utf-8") + "\n\ndef figure_fig1(fig, table):\n    import matplotlib.pyplot as plt\n    plt.savefig('x.png')\n", encoding="utf-8")
    code, _printed, record, _err = _run(quest)
    assert code == 1 and record["figures"][0]["status"] == "refused" and not (quest / "figures" / "growth.png").exists()
    code, _printed, record, _err = _run(_quest(tmp_path / "stub", _design()))
    assert code == 0 and record["figures"][0]["status"] == "not_written"


@pytest.mark.skipif(not RECORDED, reason="the recorded scratch quests are not here")
def test_the_figure_step_runs_on_a_recorded_quest(tmp_path: Path) -> None:
    scratch = next(p for p in RECORDED if p.name == "cov1_pendulum_amplitude")
    design = json.loads((scratch / "design.json").read_text(encoding="utf-8"))
    quest = tmp_path / "quest"
    shutil.copytree(scratch / "raw", quest / "raw")
    af.ensure_files(quest / "code", design)
    code, _printed, record, err = _run(quest)
    assert code == 0, err
    assert record["figures"] == [] and record["figures_said"] == ""  # the plan names no figure FI can draw (declared false)
