"""Reference values computed by FI from the frozen plan (core/reference_values.py) and FI's per-setting results tables
(core/results_by_setting.py), replayed on two s1 suite quests (trimmed copies in tests/fixtures/reference_values):

* false_discovery (m x pi1 = 9 settings, Bonferroni vs BH): the references a plan would declare, computed by FI's
  calculator, against the theory values ARIS published for the same grid (aris_fdr_sonnet_r1: outputs/results.json
  and the pi0 x q column of paper/tab_err.tex);
* first_order_kinetics (sigma x n_points x temperature_K = 125 grid cells): the simulation never reads temperature_K, so
  the cells are 25 distinct settings.

No model is called."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

from core import grid_keys, metric_spec, oracle_forms, plan, results_by_setting as rbs, trial_table
from core import reference_values as rv
from tests import reference_values_plans as P

ARIS = P.FIXTURES / "aris_fdr"


def _summaries(name: str) -> list[dict[str, Any]]:
    return P.load(name, "trial_summaries.json")["cells"]


def _build(name: str, protocol: dict[str, Any], judged: Any = "fixture") -> rbs.Built:
    return rbs.build(_summaries(name), protocol, grid_keys.code_files(P.FIXTURES / name / "code"),
                     P.load(name, "judged.json") if judged == "fixture" else judged)


def _table(built: rbs.Built, value: str) -> rbs.Table:
    return next(t for t in built.tables if t.value == value)


def _refs_at(computed: dict[str, Any], metric: str) -> dict[tuple[Any, ...], float | None]:
    return {tuple(r["setting"].values()): r["reference"] for r in computed["by_setting"] if r["metric"] == metric}


# --- the calculator ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("p", [1e-9, 0.000025, 0.025, 0.5, 0.8, 0.975, 0.999975])
def test_ndtri_is_the_inverse_normal(p: float) -> None:
    got = oracle_forms.evaluate(f"ndtri({p})", {}, special=True).value
    assert got == pytest.approx(NormalDist().inv_cdf(p), rel=1e-12)
    assert oracle_forms.evaluate(f"ndtr(ndtri({p}))", {}, special=True).value == pytest.approx(p, rel=1e-9)


def test_erfinv_inverts_erf_and_the_functions_are_only_fis_own() -> None:
    assert oracle_forms.evaluate("erf(erfinv(0.3))", {}, special=True).value == pytest.approx(0.3, rel=1e-12)
    assert oracle_forms.evaluate("ndtri(1.5)", {}, special=True).problem  # outside (0, 1): an error, not a number
    assert oracle_forms.evaluate("ndtri(0.5)", {}).unreadable  # a check's measure may not call it
    for name in ("ndtr", "ndtri", "erfinv"):
        assert name in oracle_forms.EXPECTED_FORMULA_LANGUAGE


# --- the plan side ----------------------------------------------------------------------------------------------------


def test_thresholds_are_names_of_a_reference_but_not_of_a_check() -> None:
    protocol = P.load("kin_s1", "protocol.json")
    assert "Ea_true_J_per_mol" not in oracle_forms.fixed_settings(protocol)  # unchanged: a check's formula never reads it
    assert rv.inputs(protocol)["Ea_true_J_per_mol"] == 50000
    assert rv.value_at({"formula": "Ea_true_J_per_mol"}, {"sigma": 0.01}, rv.inputs(protocol)) == (50000.0, "")


def test_an_old_protocol_without_references_is_still_valid_and_reads_no_reference() -> None:
    protocol = P.load("fdr_s1", "protocol.json")
    fixed, why = plan.normalize_protocol(protocol)
    assert why is None and fixed is not None
    computed = rv.compute(protocol, rv.grid_settings(protocol))
    assert computed["by_setting"] == []
    assert {q["status"] for q in computed["quantities"]} == {"no reference"}
    built = _build("fdr_s1", protocol)
    assert len(built.distinct.settings) == 9
    assert all("no reference value" in t.caption and "reference" not in t.columns for t in built.tables)


def test_the_plans_references_pass_the_plan_check() -> None:
    for protocol in (P.fdr_protocol(), P.kin_protocol()):
        fixed, why = plan.normalize_protocol(protocol)
        assert why is None, why
        assert rv.problems(fixed) == []


@pytest.mark.parametrize("reference, words", [
    ({"formula": "Ea_se_theory"}, "uses Ea_se_theory, which is not a setting of the grid"),
    ({"formula": "0.95", "when": {"temperature": 300}}, "not a setting of `protocol.grid`"),
    ({"formula": "0.95", "when": {"sigma": 0.3}}, "holds for no setting of the grid"),
    ({"formula": "reference_coverage", "kind": "about"}, "`kind` must be"),
    ({"formula": "reference_coverage - 0.1"}, "disagrees with a known-answer check"),  # low_noise_coverage_95: 0.95
])
def test_a_reference_that_cannot_be_used_is_refused_and_only_it_is_dropped(reference: dict[str, Any], words: str) -> None:
    protocol = P.with_references(P.load("kin_s1", "protocol.json"), {"coverage_Ea": reference,
                                                                      "coverage_lnA": {"formula": "reference_coverage"}})
    fixed, why = plan.normalize_protocol(protocol)
    assert fixed is None and words in (why or "") and "`protocol.metrics` entry `coverage_Ea`" in (why or "")
    repaired, notes = plan.repair_protocol(protocol)
    assert repaired is not None
    by_id = {m["id"]: m for m in repaired["metrics"]}
    assert set(by_id) == {m["id"] for m in protocol["metrics"]}  # every metric kept
    assert "reference" not in by_id["coverage_Ea"] and by_id["coverage_lnA"]["reference"]["formula"] == "reference_coverage"
    assert any("reference value of `coverage_Ea` was left out" in n for n in notes)


def test_a_metric_spec_checks_the_shape_of_its_reference() -> None:
    metric = {"id": "x", "estimand": "e", "kind": "mean", "unit": "a run"}
    assert metric_spec.normalize([{**metric, "reference": {"formula": "alpha"}}])[1] is None
    _clean, why = metric_spec.normalize([{**metric, "reference": "alpha"}])
    assert why and "its `reference` must be a mapping" in why


def test_an_altered_formula_disagrees_with_the_check_at_its_case() -> None:
    protocol = P.fdr_protocol()
    for m in protocol["metrics"]:
        if m["id"] == "power_bonferroni":
            m["reference"] = {**m["reference"], "formula": "1 - ndtr(ndtri(1 - alpha) - delta)"}  # alpha/m left out
    problems = rv.problems(protocol)
    assert len(problems) == 1 and "bonferroni_power_exact" in problems[0] and "0.1634" in problems[0]
    computed = rv.compute(protocol, rv.grid_settings(protocol), judged=P.load("fdr_s1", "judged.json"))
    entry = next(q for q in computed["quantities"] if q["metric"] == "power_bonferroni")
    assert entry["status"] == "no reference" and "disagrees" in entry["why"]
    assert not any(r["metric"] == "power_bonferroni" for r in computed["by_setting"])


# --- the FDR replay against ARIS's published references ---------------------------------------------------------------


def _aris() -> dict[tuple[int, float], dict[str, Any]]:
    data = json.loads((ARIS / "theory.json").read_text(encoding="utf-8"))
    return {(c["m"], c["pi1"]): c["theory"] for c in data["cells"]}


@pytest.mark.parametrize("metric, theory", [
    ("fdr_bh", "bh_fdr"), ("any_false_discovery_uncorrected", "unc_fwer"),
    ("any_false_discovery_bonferroni", "bonf_fwer"), ("power_bonferroni", "bonf_power"),
    ("power_uncorrected", "unc_power"),
])
def test_fdr_references_equal_arises_theory_values(metric: str, theory: str) -> None:
    protocol = P.fdr_protocol()
    computed = rv.compute(protocol, rv.grid_settings(protocol), judged=P.load("fdr_s1", "judged.json"))
    ours = _refs_at(computed, metric)
    compared = 0
    for (m, pi1), values in _aris().items():
        theirs = values[theory]
        if theirs is None:  # power with no real effect: ARIS has none, and the plan's `when` gives none
            assert ours[(m, pi1)] is None
            continue
        assert ours[(m, pi1)] == pytest.approx(theirs, rel=1e-9, abs=1e-15), (metric, m, pi1)
        compared += 1
    assert compared >= 6


def test_the_bh_reference_column_matches_arises_printed_table() -> None:
    """tab_err.tex: m & pi1 & uncorrected & Bonferroni & BH & pi0 q & ... (the pi0 q column printed to 3-4 decimals)."""
    protocol = P.fdr_protocol()
    ours = _refs_at(rv.compute(protocol, rv.grid_settings(protocol)), "fdr_bh")
    rows = [line.split("&") for line in (ARIS / "tab_err.tex").read_text(encoding="utf-8").splitlines() if "&" in line]
    assert len(rows) == 9
    for row in rows:
        m, pi1, printed = int(row[0]), float(row[1]), row[5].strip()
        decimals = len(printed.split(".")[1])
        assert f"{ours[(m, pi1)]:.{decimals}f}" == printed, (m, pi1)


def test_bh_power_has_no_reference_the_calculator_cannot_solve_a_fixed_point() -> None:
    built = _build("fdr_s1", P.fdr_protocol())
    table = _table(built, "power_bh")
    assert "reference" not in table.columns and "It has no reference value to compare with." in table.caption and "the plan gives" not in table.caption
    # ARIS's printed limit is not FI's: it is the m -> infinity root of an equation, which FI does not compute
    assert "0.466" in (ARIS / "tab_pow.tex").read_text(encoding="utf-8")


def test_fdr_check_agreement_and_what_each_check_confirms() -> None:
    protocol = P.fdr_protocol()
    computed = rv.compute(protocol, rv.grid_settings(protocol), judged=P.load("fdr_s1", "judged.json"))
    checked = {q["metric"]: q.get("checked", {}) for q in computed["quantities"]}
    assert checked["power_bonferroni"]["state"] == "agrees"  # by its measure: bonferroni_power_exact, passed
    assert checked["power_bonferroni"]["check"] == "bonferroni_power_exact"
    assert checked["any_false_discovery_uncorrected"]["state"] == "agrees"  # through checked_by (another measure name)
    assert checked["any_false_discovery_bh"]["state"] == "agrees"
    assert checked["fdr_bh"]["state"] == "not_confirmed"  # bh_fdr_upper_bound failed: it confirms nothing
    assert checked["any_false_discovery_bonferroni"]["state"] == "unchecked"
    # at plan time (no verdicts yet) the same reference agrees with the expected value of the check
    assert rv.cross_check(protocol, next(m for m in protocol["metrics"] if m["id"] == "power_bonferroni"))["state"] == "agrees"


def test_fdr_table_nine_settings_a_bound_never_agrees() -> None:
    built = _build("fdr_s1", P.fdr_protocol())
    assert len(built.distinct.settings) == 9 and built.distinct.unread == []
    bh = _table(built, "fdr_bh")
    assert len(bh.rows) == 9
    verdicts = {row[-1] for row in bh.rows}
    assert "agrees" not in verdicts and verdicts <= {"within bound", "within bound, within uncertainty", "above bound",
                                                     "above bound, within uncertainty"}
    # below the bound the verdict depends on the interval: at m = 20, pi1 = 0 it reaches past 0.05
    row = next(r for r in bh.rows if r[:2] == ["20", "0.0"])
    assert row[3] == "0.0493 ± 0.0071" and row[-1] == "within bound, within uncertainty"
    row = next(r for r in bh.rows if r[:2] == ["200", "0.2"])
    assert row[-2] == "≤ 0.04" and row[3] == "0.0415 ± 0.0014" and row[-1] == "above bound"
    power = _table(built, "power_bonferroni")
    row = next(r for r in power.rows if r[:2] == ["200", "0.2"])
    assert row[-2] == "0.1634" and row[-1] == "agrees"
    nulls = [r for r in power.rows if r[1] == "0.0"]
    assert nulls and all(r[3] == "no value" for r in nulls)  # power is undefined with no real effect: shown, not hidden


def test_a_bare_number_read_off_the_run_is_dropped_a_nominal_one_is_kept() -> None:
    copied = P.with_references(P.load("fdr_s1", "protocol.json"),
                               {"fdr_uncorrected": {"formula": "0.1775", "when": {"m": 20, "pi1": 0.2}}})
    table = _table(_build("fdr_s1", copied), "fdr_uncorrected")
    assert "reference" not in table.columns and "read off the run" in table.caption
    kin = _table(_build("kin_s1", P.kin_protocol()), "coverage_Ea")
    assert "reference" in kin.columns and all(r[-2] == "0.95" for r in kin.rows)


# --- the KIN replay: 125 grid cells, 25 settings -----------------------------------------------------------------------


def test_kin_temperature_is_never_read_so_125_cells_are_25_settings() -> None:
    protocol = P.load("kin_s1", "protocol.json")
    files = grid_keys.code_files(P.FIXTURES / "kin_s1" / "code")
    assert rbs.unread_axes(protocol, files) == (["temperature_K", "t_max_over_k"], "")
    built = _build("kin_s1", P.kin_protocol())
    assert len(built.distinct.settings) == 25
    assert all(s.metrics["Ea_hat"]["n"] == 2000 and len(s.cells) == 5 for s in built.distinct.settings)
    assert any("found no read of `temperature_K` and `t_max_over_k`" in n for n in built.notes)
    assert any("`Ea_sd`, `corr_Ea_lnA`" in n for n in built.notes)


def test_kin_tables_every_setting_with_its_reference() -> None:
    built = _build("kin_s1", P.kin_protocol())
    assert {t.value for t in built.tables} == {"coverage_Ea", "coverage_lnA", "Ea_hat", "lnA_hat"}
    for t in built.tables:
        assert len(t.rows) == 25 and len({tuple(r[:2]) for r in t.rows}) == 25
    ea = _table(built, "Ea_hat")
    assert {r[-2] for r in ea.rows} == {"50000"} and "noise_free_Ea" in ea.caption
    cov = _table(built, "coverage_Ea")
    assert "low_noise_coverage_95" in cov.caption and {r[-1] for r in cov.rows} <= {"agrees", "differs"}
    pooled = next(s for s in built.distinct.settings if s.setting == {"sigma": 0.02, "n_points": 12})
    alone = [c for c in _summaries("kin_s1") if c["cell"]["sigma"] == 0.02 and c["cell"]["n_points"] == 12]
    assert pooled.metrics["Ea_hat"]["mean"] == pytest.approx(
        sum(c["metrics"]["Ea_hat"]["mean"] for c in alone) / 5, rel=1e-12)


def test_kin_writer_gets_the_settings_tables_instead_of_nothing() -> None:
    built = _build("kin_s1", P.kin_protocol())
    trial_rows = trial_table.rows_of(_summaries("kin_s1"))
    assert len(trial_rows) > trial_table.WRITER_MAX_ROWS and trial_table.write_note(trial_rows) == ""
    note = rbs.writer_note(trial_rows, built)
    assert "25 distinct settings" in note and note.count("| 2000 |") == 100
    slice_note = rbs.writer_note(trial_rows, built, cap=40)
    assert "here is a slice" in slice_note and slice_note.count("| 2000 |") == 4 * 9  # 3 x 3 settings per table
    only_file = rbs.writer_note(trial_rows, built, cap=10)
    assert "even as a slice" in only_file and "| 2000 |" not in only_file
    small = trial_table.rows_of(_summaries("fdr_s1"))
    assert rbs.writer_note(small, _build("fdr_s1", P.fdr_protocol())) == trial_table.write_note(small)


def test_renderings_markdown_latex_csv() -> None:
    built = _build("fdr_s1", P.fdr_protocol())
    table = _table(built, "fdr_bh")
    md = rbs.markdown(table)
    assert md.count("\n|") == 11 and "| 200 | 0.2 | 4000 | 0.0415 ± 0.0014 |" in md
    tex = rbs.latex(table)
    for part in (r"\toprule", r"\midrule", r"\bottomrule", r"\label{tab:fi-results-fdr-bh}", r"$\pm$", r"$\le$ 0.04",
                 r"\texttt{fdr\_bh}"):
        assert part in tex, part
    assert tex.count(r" \\") == 10  # header and nine settings
    rows = built.rows
    assert len([r for r in rows if r["value"] == "fdr_bh"]) == 9
    csv = rbs.csv_text(built)
    assert csv.splitlines()[0].startswith("value,metric,setting,m,pi1,n,mean,sd,ci_lower,ci_upper")
    section = rbs.reference_section(built)
    assert section.startswith("**Reference values.**") and "`power_bh`: no reference" in section
    assert r"\paragraph{Reference values.}" in rbs.reference_section_latex(built)


def test_write_puts_the_files_beside_the_paper(tmp_path: Path) -> None:
    cells = [{"cell": {"m": m, "x": x}, "key": f"m={m},x={x}", "planned": 3, "ok": 3, "failed": 0,
              "metrics": {"hit": {"values": [1.0, 0.0, 1.0] if m == 1 else [0.0, 0.0, 1.0]}}}
             for m in (1, 2) for x in (5, 6)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text("def run_trial(cell, i, seed):\n    return {'hit': float(cell['m'] == 1)}\n",
                                                   encoding="utf-8")
    protocol = {"grid": {"m": [1, 2], "x": [5, 6]}, "thresholds": {"p": 0.5},
                "metrics": [{"id": "hit", "estimand": "P(hit)", "kind": "proportion", "unit": "a run",
                             "reference": {"formula": "p"}}]}
    built = rbs.write(tmp_path, protocol)
    assert built is not None and len(built.distinct.settings) == 2  # x is never read: 4 cells, 2 settings
    for name in ("results_by_setting.md", "results_by_setting.tex", "results_by_setting.csv", "reference_values.json"):
        assert (tmp_path / "paper" / name).is_file(), name
    # the two cells of each setting gave the very same trials (the same seeds): counted once, not twice
    assert {s.metrics["hit"]["n"] for s in built.distinct.settings} == {3}
    assert rbs.write(tmp_path / "nothing", protocol) is None


# --- pooling: when cells are one setting and when they are not -------------------------------------------------------


def _cell(setting: dict[str, Any], values: list[float]) -> dict[str, Any]:
    return {"cell": setting, "key": ",".join(f"{k}={v}" for k, v in setting.items()),
            "metrics": {"y": trial_table.summary(values)}}


def test_cells_whose_means_disagree_are_not_pooled_although_the_axis_is_unread() -> None:
    a = _cell({"s": 1, "t": 1}, [1.0, 1.1, 0.9, 1.0] * 25)
    b = _cell({"s": 1, "t": 2}, [5.0, 5.1, 4.9, 5.0] * 25)
    found = rbs.distinct([a, b], {"grid": {"s": [1], "t": [1, 2]}}, ["t"])
    assert len(found.settings) == 2 and any("were not pooled" in n for n in found.notes)
    c = _cell({"s": 1, "t": 2}, [1.1, 0.9, 1.0, 1.0] * 25)  # the same mean and spread, other trials
    found = rbs.distinct([a, c], {"grid": {"s": [1], "t": [1, 2]}}, ["t"])
    assert len(found.settings) == 1 and found.settings[0].metrics["y"]["n"] == 200


@pytest.mark.parametrize("source, unread_expected", [
    ("def run_trial(cell, i, seed):\n    return {'y': cell['s'] + cell.get('k', 0)}\n", ["t"]),
    ("def run_trial(cell, i, seed):\n    return helper(cell)\n\ndef helper(cell):\n    return {'y': cell['s']}\n", ["t"]),
    ("def run_trial(cell, i, seed):\n    return simulate(**cell)\n", None),
    # FI's one reader (core/unread_settings.py) follows the setting into a function of the code, whatever its
    # parameter is called.
    ("def run_trial(cell, i, seed):\n    return model(cell)\n\ndef model(params):\n    return {'y': params['t']}\n", ["s"]),
    ("def run_trial(cell, i, seed):\n    k = 't'\n    return {'y': cell[k]}\n", None),
    ("def run_trial(cell, i, seed):\n    return {'y': sum(v for v in cell.values())}\n", None),
])
def test_a_use_of_the_setting_fi_cannot_follow_makes_every_axis_read(source: str, unread_expected: Any) -> None:
    protocol = {"grid": {"s": [1, 2], "t": [1, 2]}}
    unread, why = rbs.unread_axes(protocol, {"simulate.py": source})
    if unread_expected is None:  # not followed: every axis counts as read, and the reason is said
        assert unread == [] and why
    else:
        assert unread == unread_expected and not why


def test_an_axis_a_reference_formula_uses_is_never_pooled() -> None:
    protocol = {"grid": {"s": [1, 2], "t": [1, 2]}}
    source = "def run_trial(cell, i, seed):\n    return {'y': cell['s']}\n"
    assert rbs.unread_axes(protocol, {"simulate.py": source}, {"t"}) == ([], "")


def test_values_compared_inside_each_run_are_rows_of_their_own() -> None:
    """A study that compares values inside each run records a number per value under the setting plus that value
    (core/in_run.py). Not exercised by the replays (the FDR quest's per-value records are empty): synthetic."""
    cells = [
        {"cell": {"m": 20}, "key": "m=20", "level": None, "metrics": {}},
        {"cell": {"m": 20, "method": "a"}, "key": "m=20,method=a", "level": {"method": "a"},
         "metrics": {"y": trial_table.summary([0.1, 0.2, 0.3])}},
        {"cell": {"m": 20, "method": "b"}, "key": "m=20,method=b", "level": {"method": "b"},
         "metrics": {"y": trial_table.summary([0.5, 0.6, 0.7])}},
    ]
    built = rbs.build(cells, {"grid": {"m": [20]}}, {})
    table = _table(built, "y")
    assert table.columns[:2] == ["m", "method"] and [r[1] for r in table.rows] == ["a", "b"]


# --- formatting --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mean, lower, upper, text", [
    (0.177483, 0.172008, 0.182958, "0.1775 ± 0.0055"),
    (50005.76, 49900.0, 50111.5, "50010 ± 110"),
    (0.0996, 0.0, 0.1992, "0.10 ± 0.10"),
    (3.0, None, None, "3"),
])
def test_value_plus_minus_half_width(mean: float, lower: Any, upper: Any, text: str) -> None:
    assert rbs.format_pm(mean, lower, upper) == text


@pytest.mark.parametrize("kind, ref, mean, lower, upper, verdict", [
    ("value", 0.95, 0.948, 0.937, 0.958, "agrees"),
    ("value", 0.95, 0.924, 0.912, 0.936, "differs"),
    ("value", 1.0, 1.0, 0.99904, 1.0 - 1e-15, "agrees"),
    ("bound", 0.04, 0.0393, 0.035, 0.044, "within bound, within uncertainty"),
    ("bound", 0.04, 0.0393, 0.035, 0.0399, "within bound"),
    ("bound", 0.04, 0.0415, 0.0401, 0.0429, "above bound"),
    ("bound", 0.04, 0.0413, 0.0376, 0.0450, "above bound, within uncertainty"),
    ("lower_bound", 0.5, 0.45, 0.40, 0.49, "below bound"),
    ("value", None, 0.5, 0.4, 0.6, "no reference"),
])
def test_comparison_words(kind: str, ref: Any, mean: float, lower: float, upper: float, verdict: str) -> None:
    assert rbs.comparison(kind, ref, mean, lower, upper) == verdict


def test_the_trial_table_is_unchanged_by_reading_it_through_summaries(tmp_path: Path) -> None:
    values = {"a": [0.1, 0.4, 0.35, 0.9], "b": [1.0, 0.0, 1.0, 1.0], "i": [0.0, 1.0, 2.0, 3.0]}
    cells = [{"cell": {"m": m}, "key": f"m={m}", "metrics": {k: {"values": [v * m if k == "a" else v for v in vs]}
                                                              for k, vs in values.items()}} for m in (1, 2)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    rows = trial_table.rows(tmp_path)
    assert {r["metric"] for r in rows} == {"a", "b"}
    a = next(r for r in rows if r["metric"] == "a" and r["setting"] == "m=2")
    from core import stats
    ci = stats.confidence_interval([v * 2 for v in values["a"]])
    assert (a["ci_lower"], a["ci_upper"]) == (ci["ci_lower"], ci["ci_upper"]) and a["cell"] == {"m": 2}
    b = next(r for r in rows if r["metric"] == "b" and r["setting"] == "m=1")
    assert (b["ci_lower"], b["ci_upper"]) == stats.wilson_interval(3, 4)
    assert math.isclose(trial_table.summary(values["a"])["mean"], 0.4375)
    assert re.fullmatch(r"[0-9a-f]{16}", trial_table.summary(values["a"])["digest"])


# --- the provenance false alarm (paper/provenance_audit.json of the FDR quest flagged 0.177, 0.198, 0.0415, 0.384) ---


# --- the engine: what the writer gets ---------------------------------------------------------------------------------


def _quest_with_a_dead_axis(root: Path) -> dict[str, Any]:
    """Two settings of `s` times three of `t`, which the code never reads: six grid cells, two distinct settings."""
    cells = []
    for s in (1, 2):
        for t in (1, 2, 3):
            values = [0.2 * s + 0.01 * ((i * 7 + t) % 5) for i in range(40)]
            cells.append({"cell": {"s": s, "t": t}, "key": f"s={s},t={t}", "planned": 40, "ok": 40, "failed": 0,
                          "metrics": {"y": {"values": values}}})
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (root / "code").mkdir(exist_ok=True)
    (root / "code" / "simulate.py").write_text(
        "def run_trial(cell, trial, seed):\n    return {'y': 0.2 * cell['s']}\n", encoding="utf-8")
    return {"grid": {"s": [1, 2], "t": [1, 2, 3]}, "thresholds": {"k": 0.2},
            "metrics": [{"id": "y", "estimand": "E[y]", "kind": "mean", "unit": "a run", "reference": {"formula": "k*s"}}]}


def test_the_writer_gets_every_setting_when_the_trial_table_is_too_long(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from tests.test_disclosure import _engine

    seen: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen.append(messages[-1]["content"] if isinstance(messages, list) else str(messages))
        return "# T\n\n## Results\nThe mean was 0.22.\n\n## Limitations\nNone.\n"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    monkeypatch.setattr(trial_table, "WRITER_MAX_ROWS", 3)  # six trial rows: too many; two settings: given
    protocol = _quest_with_a_dead_axis(tmp_path)
    eng = _engine(tmp_path)
    eng._protocol_block = lambda state: protocol  # type: ignore[assignment]
    asyncio.run(eng._node_write({"topic": "t", "title": "t", "iteration": 0,  # type: ignore[arg-type]
                                 "exec_result": {"returncode": 0}, "result_json": {"x": 1.5}}))
    for name in ("results_table.md", "results_by_setting.md", "results_by_setting.tex", "results_by_setting.csv",
                 "reference_values.json"):
        assert (tmp_path / "paper" / name).is_file(), name
    prompt = next(p for p in seen if "FI's tables of every setting" in p)
    assert "2 distinct settings" in prompt and "found no read of `t`" in prompt and "| 120 |" in prompt
    assert "FI's table of what every trial gave" not in prompt


# --- review of the branch: each probe of C:\fi_compare\reviews\scratch_ref as a test -----------------------------------


# --- the second review of the branch ---


# P0-3: a setting read inside a lambda or a closure is read

@pytest.mark.parametrize("source", [
    "import numpy as np\ndef run_trial(cell, i, seed):\n    noise = lambda: cell['sigma']\n"
    "    return {'y': np.random.default_rng(seed).normal(0, noise(), cell['n']).mean()}\n",
    "def run_trial(cell, i, seed):\n    def model(x):\n        return x * cell['sigma']\n    return {'y': model(cell['n'])}\n",
    "from scipy.integrate import solve_ivp\ndef run_trial(cell, i, seed):\n"
    "    sol = solve_ivp(lambda t, y: -cell['sigma'] * y, (0, cell['n']), [1.0])\n    return {'y': sol.y[0, -1]}\n",
    "def run_trial(cell, i, seed):\n    c = cell\n    def model(x):\n        return x * c['sigma']\n"
    "    return {'y': model(c['n'])}\n",
])
def test_a_setting_read_in_a_lambda_or_closure_is_read(source: str) -> None:
    assert rbs.unread_axes({"grid": {"sigma": [0.01, 0.1], "n": [5, 50]}}, {"simulate.py": source}) == ([], "")


def test_a_setting_put_in_a_global_makes_every_axis_read() -> None:
    """``global CFG; CFG = cell`` hands the setting to every function of the module: FI cannot follow it, so no axis
    counts as unread (and no cells are pooled over ``sigma``, which ``model`` reads through ``CFG``)."""
    source = ("CFG = None\ndef run_trial(cell, i, seed):\n    global CFG\n    CFG = cell\n"
              "    return {'y': model() + cell['n']}\ndef model():\n    return CFG['sigma']\n")
    unread, why = rbs.unread_axes({"grid": {"sigma": [0.01, 0.1], "n": [5, 50]}}, {"simulate.py": source})
    assert unread == [] and "CFG" in why and "line 4" in why


def test_code_fi_cannot_parse_makes_every_axis_read() -> None:
    unread, why = rbs.unread_axes({"grid": {"s": [1, 2], "t": [1, 2]}},
                                  {"simulate.py": "def run_trial(cell, i, seed):\n    return {'y': cell['s']}\n",
                                   "model.py": "def f(:\n"})
    assert unread == [] and "model.py" in why


def test_a_study_file_called_run_py_in_a_package_is_the_studys_code() -> None:
    files = {"run.py": "def run_trial(cell, i, seed):\n    return {'y': cell['t']}\n",  # FI's runner: skipped
             "pkg/run.py": "def run_trial(cell, i, seed):\n    return {'y': cell['s']}\n"}
    assert rbs.unread_axes({"grid": {"s": [1, 2], "t": [1, 2]}}, files) == (["t"], "")


def test_cells_whose_spread_differs_are_not_pooled() -> None:
    """The probe: sigma changes only the spread, the means agree; an unread sigma must not pool the cells."""
    import random

    rng = random.Random(1)
    cells = []
    for sigma in (0.01, 0.1):
        for n in (5, 50):
            vals = [rng.gauss(1.0, sigma) for _ in range(400)]
            cells.append({"cell": {"sigma": sigma, "n": n}, "key": f"sigma={sigma},n={n}",
                          "metrics": {"y": trial_table.summary(vals)}})
    found = rbs.distinct(cells, {"grid": {"sigma": [0.01, 0.1], "n": [5, 50]}}, ["sigma"])
    assert len(found.settings) == 4
    assert {round(s.metrics["y"]["sd"], 2) for s in found.settings} == {0.01, 0.1}
    assert any("were not pooled" in n and "spread" in n for n in found.notes)
    assert not any("pooled and n counts them all" in n for n in found.notes)  # nothing was pooled, so it is not said


def test_cells_whose_failures_differ_are_not_pooled() -> None:
    values = [1.0, 1.1, 0.9, 1.0] * 85
    a = {**_cell({"s": 1, "t": 1}, values), "planned": 400, "ok": 340, "failed": 60}
    b = {**_cell({"s": 1, "t": 2}, values), "planned": 400, "ok": 400, "failed": 0}
    found = rbs.distinct([a, b], {"grid": {"s": [1], "t": [1, 2]}}, ["t"])
    assert len(found.settings) == 2 and any("failed" in n for n in found.notes)
    c = {**_cell({"s": 1, "t": 2}, values[1:] + values[:1]), "planned": 400, "ok": 338, "failed": 62}
    assert len(rbs.distinct([a, c], {"grid": {"s": [1], "t": [1, 2]}}, ["t"]).settings) == 1


# P1-1: a reference value is accounted for only where the paper states it as a reference


# P1-3: no verdict the interval is too wide to give

@pytest.mark.parametrize("kind, ref, mean, lower, upper, n, verdict", [
    ("value", 0.95, 0.5, -4.6, 5.6, 2, "too few trials to judge"),  # the probe: n = 2, values 0.1 and 0.9
    ("bound", 0.05, 0.0, 0.0, 0.5615, 3, "too few trials to judge"),  # the probe: three trials all 0 (Wilson)
    ("value", 0.95, 0.9, 0.55, 1.0, 50, "too few trials to judge"),  # half-width over a quarter of the reference
    ("value", 0.95, 0.5, 0.45, 0.55, 5, "differs"),  # an interval that excludes the reference still says so
    ("bound", 0.04, 0.0393, 0.035, 0.044, 4000, "within bound, within uncertainty"),
    ("bound", 0.04, 0.030, 0.028, 0.032, 4000, "within bound"),
    ("lower_bound", 0.5, 0.55, 0.45, 0.65, 100, "within bound, within uncertainty"),
    ("lower_bound", 0.5, 0.60, 0.55, 0.65, 100, "within bound"),
    ("value", 0.0, 0.01, -0.05, 0.07, 100, "agrees"),  # a reference of 0: only the count of trials can be too small
])
def test_verdicts_need_enough_trials(kind: str, ref: float, mean: float, lower: float, upper: float, n: int,
                                     verdict: str) -> None:
    assert rbs.comparison(kind, ref, mean, lower, upper, n) == verdict


# P2

def test_erfinv_is_odd_and_accurate_near_one() -> None:
    erfinv = oracle_forms.SPECIAL_FUNCTIONS["erfinv"]
    for y in (1e-9, 0.3, 0.9, 1 - 2.0 ** -30, 1 - 2.0 ** -40):
        assert erfinv(-y) == -erfinv(y)
    x = erfinv(1 - 2.0 ** -40)
    assert abs(math.erfc(x) / 2.0 ** -40 - 1) < 1e-9


@pytest.mark.parametrize("formula, thresholds", [
    ("0.17748 + 0*m", {}),  # an expression that evaluates to the measured mean
    ("p_seen", {"p_seen": 0.17748}),  # a threshold holding the number read off the run
])
def test_a_reference_that_evaluates_to_the_measured_value_is_dropped(formula: str, thresholds: dict[str, Any]) -> None:
    protocol = P.load("fdr_s1", "protocol.json")
    protocol = {**protocol, "thresholds": {**protocol["thresholds"], **thresholds}}
    copied = P.with_references(protocol, {"fdr_uncorrected": {"formula": formula, "when": {"m": 20, "pi1": 0.2}}})
    table = _table(_build("fdr_s1", copied), "fdr_uncorrected")
    assert "reference" not in table.columns and "read off the run" in table.caption
