"""FI fills every result number a paper states from its own records (core/paper_values.py)."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from core import paper_values as pv

FIXTURES = Path(__file__).parent / "fixtures" / "paper_values"


def _records(name: str, tmp_path: Path, **kw) -> pv.Records:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / name, tmp_path / "raw" / "trials.json")
    rec = pv.load(tmp_path, **kw)
    assert rec is not None
    return rec


@pytest.fixture()
def s2(tmp_path):
    return _records("s2_kinetics_trials.json", tmp_path)


@pytest.fixture()
def s3(tmp_path):
    return _records("s3_kinetics_trials.json", tmp_path)


def _cells(name: str) -> list[dict]:
    return [c for c in json.loads((FIXTURES / name).read_text())["cells"] if c["metrics"]]


def _within_sd(cells, metric, **where) -> float:
    """Independent numpy recomputation: sqrt(sum((n-1) s^2) / sum(n-1)) over the matching settings."""
    use = [np.asarray(c["metrics"][metric]["values"]) for c in cells
           if all(c["cell"][k] == v for k, v in where.items())]
    num = sum((len(x) - 1) * np.var(x, ddof=1) for x in use)
    return float(math.sqrt(num / sum(len(x) - 1 for x in use)))


# --- A1: the five numbers the old check falsely flagged on s2 --------------------------------------------------


KEYS = [
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=5 | %}}",
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.05, n_points=10 | %}}",
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.005, n_points=40 | %}}",
]
POOLED_KEYS = [  # one SD over settings that differ in n_points or noise_sigma
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}}",
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.05, n_points=all | %}}",
    "{{v:Ea_rel_bias:sd @ noise_sigma=all, n_points=5 | %}}",
    "{{v:Ea_rel_bias:sd @ noise_sigma=all, n_points=10 | %}}",
]


@pytest.mark.parametrize("key", KEYS)
def test_s2_sd_keys_of_one_setting_render_and_match_numpy(s2, key):
    where = {"noise_sigma": float(key.split("noise_sigma=")[1].split(",")[0]),
             "n_points": int(key.split("n_points=")[1].split(" ")[0])}
    got = pv.resolve(key, s2)
    assert got.filled and "(95% chi-square interval " in got.text and got.text.count("%") >= 3
    assert got.numbers[0] == pytest.approx(_within_sd(_cells("s2_kinetics_trials.json"), "Ea_rel_bias", **where))


@pytest.mark.parametrize("key", POOLED_KEYS)
def test_an_sd_over_settings_that_differ_is_not_recorded(s2, key):
    got = pv.resolve(key, s2)
    assert not got.filled and got.hit is not None
    assert "different settings" in got.hit.reason and "sd @ <axis>=each" in got.hit.reason  # tells the writer what to use
    assert not pv.resolve(key.replace(":sd ", ":sd_ci "), s2).filled


def test_s2_ratio_key_of_two_settings_matches_numpy(s2):
    key = "{{v:Ea_rel_bias:sd @ noise_sigma=0.1, n_points=5 | ratio @ noise_sigma=0.005, n_points=5}}"
    got = pv.resolve(key, s2)
    cells = _cells("s2_kinetics_trials.json")
    expected = (_within_sd(cells, "Ea_rel_bias", noise_sigma=0.1, n_points=5)
                / _within_sd(cells, "Ea_rel_bias", noise_sigma=0.005, n_points=5))
    assert got.filled and got.numbers[0] == pytest.approx(expected)
    assert not pv.resolve("{{v:Ea_rel_bias:sd @ noise_sigma=0.1, n_points=all | ratio @ noise_sigma=0.005, n_points=all}}", s2).filled


def test_an_sd_setting_by_setting_is_filled_where_the_pooled_one_is_not(s2):
    cells = _cells("s2_kinetics_trials.json")
    got = pv.resolve("Ea_rel_bias:sd @ noise_sigma=0.01, n_points=each | range", s2)
    each = [_within_sd(cells, "Ea_rel_bias", noise_sigma=0.01, n_points=n) for n in sorted({c["cell"]["n_points"] for c in cells})]
    assert got.filled and got.numbers == [pytest.approx(min(each)), pytest.approx(max(each))]


def test_s2_coverage_wilson_interval_is_two_ends_and_count_is_exact(s2):
    cells = _cells("s2_kinetics_trials.json")
    allv = np.concatenate([c["metrics"]["coverage_Ea"]["values"] for c in cells])
    k, n = int(allv.sum()), len(allv)
    got = pv.resolve("{{v:coverage_Ea:mean_ci @ noise_sigma=all, n_points=all}}", s2)
    assert got.filled and "Wilson interval" in got.text and "±" not in got.text
    assert got.text.startswith(f"{k / n:.4f} (95% Wilson interval ") and got.text.endswith(f", {n} trials)")
    z = 1.959964
    centre = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert f"{centre - half:.4f} to {centre + half:.4f}" in got.text
    assert pv.resolve("coverage_Ea:count @ noise_sigma=all, n_points=all", s2).text == f"{k} of {n}"


def test_a_wilson_interval_never_renders_with_plus_minus(s2):
    # sigma 0.02, n 40: the Wilson interval is not symmetric about the mean
    for form in ("pm", "ci", "mean_ci"):
        got = pv.resolve(f"coverage_Ea:{form} @ noise_sigma=0.02, n_points=40", s2)
        assert got.filled and "±" not in got.text, (form, got.text)
    assert " to " in pv.resolve("coverage_Ea:pm @ noise_sigma=0.02, n_points=40", s2).text


def test_a_t_interval_may_render_with_plus_minus(s2):
    got = pv.resolve("Ea_rel_bias:pm @ noise_sigma=0.02, n_points=40", s2)
    assert got.filled and "±" in got.text


def test_range_over_each_setting(s2):
    got = pv.resolve("{{v:Ea_rel_bias @ noise_sigma=each, n_points=each | range | %}}", s2)
    means = [float(np.mean(c["metrics"]["Ea_rel_bias"]["values"])) * 100 for c in _cells("s2_kinetics_trials.json")]
    assert got.filled and got.text == f"{pv.sig(min(means))}% to {pv.sig(max(means))}%"


def test_marker_round_trip(s2):
    paper = "SD is {{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=5 | %}} here."
    filled, hits = pv.fill(paper, s2)
    assert hits == []
    sd = pv.resolve("Ea_rel_bias:sd @ noise_sigma=0.01, n_points=5 | %", s2).text
    assert "chi-square interval" in sd
    assert filled == ("SD is <!-- fi:v {{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=5 | %}} -->" + sd +
                      "<!-- /fi:v --> here.")
    assert pv.to_placeholders(filled) == paper
    assert pv.without_marks(filled) == f"SD is {sd} here."


# --- s3: the numbers the old heuristic falsely flagged ------------------------------------------------------------


def test_s3_cells_with_metrics_only_are_used(s3):
    # the file has empty cells carrying more axes; they are not counted and not axes
    raw = json.loads((FIXTURES / "s3_kinetics_trials.json").read_text())["cells"]
    assert any(not c["metrics"] and len(c["cell"]) > 2 for c in raw)
    assert set(s3.axes()) == {"noise_sigma", "n_points"}
    assert pv.resolve("Ea_hat:n @ noise_sigma=0.02, n_points=5", s3).text == "400"


@pytest.mark.parametrize("key, text, where", [
    ("{{v:Ea_hat:sd @ noise_sigma=0.02, n_points=5}}", "1.282", (0.02, 5)),
    ("{{v:Ea_hat:sd @ noise_sigma=0.02, n_points=12}}", "0.748", (0.02, 12)),
    ("{{v:Ea_hat:sd @ noise_sigma=0.02, n_points=40}}", "0.425", (0.02, 40)),
    ("{{v:Ea_hat:sd @ noise_sigma=0.1, n_points=5}}", "5.95", (0.1, 5)),
    ("{{v:Ea_hat @ noise_sigma=0.1, n_points=8}}", "49.48", (0.1, 8)),
])
def test_s3_heuristic_false_flags_render(s3, key, text, where):
    got = pv.resolve(key, s3)
    assert got.filled and got.text.split(" (")[0] == text if ":sd" in key else got.text == text
    (cell,) = [c for c in _cells("s3_kinetics_trials.json") if (c["cell"]["noise_sigma"], c["cell"]["n_points"]) == where]
    x = np.asarray(cell["metrics"]["Ea_hat"]["values"])
    expected = float(np.std(x, ddof=1)) if ":sd" in key else float(np.mean(x))
    assert (got.numbers[0] == pytest.approx(expected)) if ":sd" in key else pv.sig(expected) == text


# --- A2: a key FI cannot fill is "not recorded" and a hit ---------------------------------------------------------


@pytest.mark.parametrize("key, reason", [
    ("{{v:Ea_rel_bias:sd @ noise_sigma=0.01}}", "n_points"),                                  # a missing axis
    ("{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all, colour=red}}", "colour"),          # unknown axis
    ("{{v:Ea_rel_bias:sd @ noise_sigma=0.03, n_points=all}}", "0.03"),                        # value not in the record
    ("{{v:Ea_hatt @ noise_sigma=all, n_points=all}}", "Ea_hatt"),                             # unknown metric
    ("{{v:Ea_rel_bias:median @ noise_sigma=all, n_points=all}}", "median"),                   # unknown form
    ("{{v:Ea_rel_bias @ noise_sigma=each, n_points=all}}", "reducer"),                        # each without a reducer
    ("{{v:Ea_rel_bias @ noise_sigma=all, n_points=all | range}}", "each"),                    # reducer without each
    ("{{v:Ea_rel_bias:ref @ noise_sigma=all, n_points=all}}", "plan"),                        # no plan, no reference
    ("{{v:Ea_rel_bias @ noise_sigma=all, n_points=all | frobnicate}}", "frobnicate"),         # unknown part
    ("{{v:Ea_rel_bias}}", "@"),                                                                 # no where
])
def test_bad_keys_are_not_recorded_and_a_hit(s2, key, reason):
    got = pv.resolve(key, s2)
    assert got.text == "not recorded" and got.hit is not None
    assert reason in got.hit.reason


def test_unfilled_key_in_a_paper_is_reported_not_guessed(s2):
    filled, hits = pv.fill("A {{v:nope @ noise_sigma=all, n_points=all}} B", s2)
    assert "<!-- fi:v {{v:nope @ noise_sigma=all, n_points=all}} -->not recorded<!-- /fi:v -->" in filled
    assert [h.key for h in hits] == ["nope @ noise_sigma=all, n_points=all"]
    assert "not recorded" in hits[0].text()


def test_no_trial_record_means_every_key_is_not_recorded():
    got = pv.resolve("{{v:x @ a=all}}", None)
    assert got.text == "not recorded" and "no per-setting" in got.hit.reason


# --- rounding ------------------------------------------------------------------------------------------------------


def test_rounding_rules():
    assert pv.sig(6.05, 4) == "6.050"                       # trailing zeros kept
    assert pv.sig(0.99996, 4) == "1.000"
    assert pv.sig(19.6123, 3) == "19.6"
    assert pv.fmt(pv.Measured(19.6123, "ratio")) == "19.6"
    assert pv.fmt(pv.Measured(5609, "int")) == "5609"
    m = pv.Measured(0.9348, "interval", 0.9283, 0.9408, "95% Wilson interval", 6000)
    assert pv.fmt(m) == "0.9348 (95% Wilson interval 0.9283 to 0.9408, 6000 trials)"
    assert pv.fmt(m, "ci") == "0.9283 to 0.9408"
    assert pv.fmt(m, "pm") == "0.9283 to 0.9408"           # not symmetric by the rule: never ±
    t = pv.Measured(1.003, "interval", 0.991, 1.015, "95% t interval of the mean", 30, symmetric=True)
    assert pv.fmt(t, "pm") == "1.003 ± 0.012"


# --- the other forms, against numpy on the s2 record -------------------------------------------------------------


def test_sd_ci_is_a_chi_square_interval():
    assert pv.chi2_quantile(0.975, 10) == pytest.approx(20.4832, rel=1e-4)
    assert pv.chi2_quantile(0.025, 10) == pytest.approx(3.2470, rel=1e-4)


def test_other_scalar_forms(s2):
    cells = _cells("s2_kinetics_trials.json")
    x = np.concatenate([c["metrics"]["Ea_rel_bias"]["values"] for c in cells if c["cell"]["noise_sigma"] == 0.05])
    assert pv.resolve("Ea_rel_bias:se @ noise_sigma=0.05, n_points=all", s2).numbers[0] == pytest.approx(
        np.std(x, ddof=1) / math.sqrt(len(x)))
    assert pv.resolve("Ea_rel_bias:n @ noise_sigma=0.05, n_points=all", s2).text == str(len(x))
    got = pv.resolve("Ea_rel_bias:sd_ci @ noise_sigma=0.05, n_points=5", s2)
    assert got.filled and "chi-square interval" in got.text and "±" not in got.text
    r = pv.resolve("Ea_rel_bias:corr:coverage_Ea @ noise_sigma=0.05, n_points=all", s2)
    y = np.concatenate([c["metrics"]["coverage_Ea"]["values"] for c in cells if c["cell"]["noise_sigma"] == 0.05])
    assert r.numbers[0] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert "Fisher interval" in pv.resolve("Ea_rel_bias:corr_ci:coverage_Ea @ noise_sigma=0.05, n_points=all", s2).text
    d = pv.resolve("Ea_rel_bias @ noise_sigma=0.1, n_points=all | diff @ noise_sigma=0.005, n_points=all", s2)
    lo = np.mean(np.concatenate([c["metrics"]["Ea_rel_bias"]["values"] for c in cells if c["cell"]["noise_sigma"] == 0.005]))
    hi = np.mean(np.concatenate([c["metrics"]["Ea_rel_bias"]["values"] for c in cells if c["cell"]["noise_sigma"] == 0.1]))
    assert d.numbers[0] == pytest.approx(hi - lo)


def test_argmax_and_max_abs(s2):
    got = pv.resolve("Ea_rel_bias:sd @ noise_sigma=each, n_points=5 | argmax", s2)
    assert got.text == "at noise_sigma = 0.1"
    assert pv.resolve("Ea_rel_bias @ noise_sigma=each, n_points=each | max_abs | %", s2).filled


def test_ref_and_count_where_use_the_plan(s2):
    protocol = {"grid": {"noise_sigma": [0.005, 0.01, 0.02, 0.05, 0.1], "n_points": [5, 10, 20, 40]},
                "thresholds": {"nominal_coverage": 0.95},
                "metrics": [{"id": "coverage_Ea", "reference": {"value": "coverage_Ea", "formula": "nominal_coverage"}}]}
    rec = pv.from_cells(s2.cells, protocol)
    got = pv.resolve("coverage_Ea:ref @ noise_sigma=all, n_points=all", rec)
    assert got.filled and got.text == "0.9500"
    words = pv.resolve("coverage_Ea @ noise_sigma=each, n_points=each | count_where:agrees", rec)
    assert words.filled and words.text in pv._WORDS or words.text.isdigit()
    assert pv.resolve("coverage_Ea:rel_diff @ noise_sigma=all, n_points=all", rec).hit is not None


# --- the sheet ----------------------------------------------------------------------------------------------------


def test_sheet_lists_keys_that_resolve_and_is_deterministic(s2):
    text = pv.sheet(s2)
    assert text == pv.sheet(s2)
    keys = [k for k in __import__("re").findall(r"\{\{v:[^{}]+\}\}", text) if "<" not in k]
    assert len(keys) > 40
    assert all(pv.resolve(k, s2).filled for k in keys)
    assert "{{v:Ea_rel_bias:mean_ci @ noise_sigma=0.01, n_points=5}}" in text
    assert "Name every axis" in text


def test_sheet_cap_keeps_the_grammar_and_the_pooled_lines(s2):
    short = pv.sheet(s2, max_lines=10)
    assert len(short) < len(pv.sheet(s2))
    assert "Name every axis" in short and "noise_sigma=all" in short


def test_a_fallback_too_long_to_read_lists_the_plan_s_quantities_first_and_stays_in_its_limit(s2):
    """A real study recorded 447 quantities over 455,292 records: its fallback list was 44,811 lines (16 MB)."""
    whole = pv.sheet(s2, max_lines=10)
    rec = pv.from_cells(s2.cells, {"metrics": [{"id": "coverage_Ea"}]})
    short = pv.sheet(rec, max_lines=10, max_shown=12)
    lines = [line for line in short.splitlines() if line.startswith("- ")]
    assert len(lines) == 12 and len(set(lines)) == 12 and all(line in whole for line in lines)
    assert short.index("coverage_Ea: lowest and highest mean") < short.index("Ea_rel_bias: lowest and highest mean")  # plan first
    assert "noise_sigma=all" in short and "n_points=all" in short  # each axis's pooled lines, in turn
    assert "Name every axis" in short and "any other key can still be placed" in short
    assert pv.sheet(rec, max_lines=10) == pv.sheet(s2, max_lines=10)  # under the limit: as it was


def test_sheet_without_records_says_so():
    assert "none" in pv.sheet(None)


# --- a value each setting sets (a theory line) has no pooled value -------------------------------------------------
# A paper compared the pooled empirical SD of Ea at n = 40 (1.024) with the mean over noise levels of the theory SD
# (0.8023) and wrote "the empirical sd lay above theory at both". Setting by setting, the SD at n = 40 is 0.914 to
# 0.927 of theory, below it at every noise level. The pooled SD is an RMS over the noise levels and the theory's mean
# is an arithmetic mean of values that differ ten-fold, so the pooled comparison reads "above" by construction.

GRID = "sigma={s}, n_points={n}"


@pytest.fixture()
def s5(tmp_path):
    return _records("s5_kinetics_trials.json", tmp_path)


def test_a_value_each_setting_sets_has_no_pooled_mean(s5):
    for form in ("", ":mean_ci", ":sd", ":se"):
        got = pv.resolve(f"Ea_sd_theory{form} @ sigma=all, n_points=40", s5)
        assert not got.filled, form
        assert "differs between the settings" in got.hit.reason and "ratio" in got.hit.reason


def test_a_ratio_to_the_pooled_theory_is_not_filled_either(s5):
    for key in ("Ea_hat:sd @ sigma=all, n_points=40 | ratio Ea_sd_theory",
                "Ea_hat:sd @ sigma=all, n_points=40 | diff Ea_sd_theory",
                "Ea_hat:sd @ sigma=0.1, n_points=all | ratio Ea_sd_theory",
                "Ea_sd_theory @ sigma=each, n_points=all | max"):
        assert not pv.resolve(key, s5).filled, key


def test_the_theory_at_one_setting_and_the_ratio_over_the_settings_fill(s5):
    one = pv.resolve("Ea_sd_theory @ sigma=0.1, n_points=40", s5)
    assert one.filled and one.text == "2.168"
    rng = pv.resolve("Ea_hat:sd @ sigma=each, n_points=40 | ratio Ea_sd_theory | range", s5)
    assert rng.filled and rng.text == "0.914 to 0.927"  # below theory at every noise level
    five = pv.resolve("Ea_hat:sd @ sigma=each, n_points=5 | ratio Ea_sd_theory | range", s5)
    assert five.filled and five.text == "1.20 to 1.26"  # above it at every noise level
    assert pv.resolve("Ea_hat:sd @ sigma=0.05, n_points=40 | ratio Ea_sd_theory", s5).filled


def test_a_value_that_is_the_same_at_every_setting_still_has_a_pooled_mean(s5):
    assert pv.resolve("Ea_true @ sigma=all, n_points=all", s5).text == "50.00"
    assert pv.resolve("corr_b0_b1_theory @ sigma=all, n_points=40", s5).filled
    # A count of trials is not a mean of references.
    assert pv.resolve("Ea_sd_theory:n @ sigma=all, n_points=40", s5).filled


def test_a_measured_value_with_one_trial_per_setting_is_not_taken_for_a_reference():
    cells = [{"cell": {"h": h}, "key": f"h={h}", "planned": 1, "ok": 1, "failed": 0,
              "metrics": {"err": {"values": [h * 2.0]}}} for h in (0.1, 0.2, 0.4)]
    rec = pv.from_cells(cells)
    got = pv.resolve("err @ h=all", rec)
    assert got.filled and got.text == "0.4667 (mean over 3 settings, one run each; no interval)"


def test_a_measured_value_that_varies_between_trials_is_not_a_reference(s5):
    assert pv.resolve("Ea_hat @ sigma=all, n_points=40", s5).filled
    assert pv.resolve("cover_Ea @ sigma=all, n_points=40", s5).filled


def test_the_sheet_does_not_offer_a_pooled_theory_and_says_how_to_compare(s5):
    text = pv.sheet(s5, max_lines=10_000)
    assert "{{v:Ea_sd_theory:mean_ci @ sigma=all" not in text and "{{v:Ea_sd_theory @ sigma=all" not in text
    assert "{{v:Ea_sd_theory @ sigma=0.1, n_points=40}} -> 2.168" in text  # no interval around a value a setting sets
    assert "Ea_sd_theory" in text.split("Compare with a value each setting sets")[1]
    keys = [k for k in __import__("re").findall(r"\{\{v:[^{}]+\}\}", text) if "<" not in k]
    assert all(pv.resolve(k, s5).filled for k in keys)


def test_a_study_with_no_such_value_gets_no_extra_sheet_line(s2):
    assert "Compare with a value each setting sets" not in pv.sheet(s2)


# --- FI words the comparison with a theory value itself (`| vs`) ----------------------------------------------------
# A writer who typed "above" beside a correct per-setting key flipped it on the s5 record (n = 40 reads below or level with
# theory at every noise level). With `| vs <theory>` the direction word comes from FI's record, setting by setting, decided
# by the measured value's own 95% interval, so the writer has no word to flip.


def test_vs_words_the_direction_per_setting_from_the_interval(s5):
    five = pv.resolve("Ea_hat:sd @ sigma=each, n_points=5 | vs Ea_sd_theory", s5)
    assert five.filled and five.text.startswith("above Ea_sd_theory at all 5 values of sigma")
    assert "(1.20 to 1.26 times it)" in five.text
    forty = pv.resolve("Ea_hat:sd @ sigma=each, n_points=40 | vs Ea_sd_theory", s5)
    assert forty.filled and "above" not in forty.text  # the original wrong word cannot come out
    assert forty.text.startswith("consistent with Ea_sd_theory (inside the measured 95% interval) at all 5 values of sigma")
    assert "(0.914 to 0.927 times it)" in forty.text


def test_vs_counts_a_mixed_outcome(s5):
    got = pv.resolve("Ea_hat:sd @ sigma=each, n_points=each | vs Ea_sd_theory", s5)
    assert got.filled and "of 25 settings" in got.text and "consistent with it at" in got.text


def test_vs_below_when_the_whole_interval_is_under_the_reference(s5):
    got = pv.resolve("Ea_hat @ sigma=each, n_points=40 | vs Ea_sd_theory", s5)  # mean Ea ~50 vs a reference far smaller
    assert got.filled and got.text.startswith("above Ea_sd_theory at all 5")
    cells = [{"cell": {"h": h}, "key": f"h={h}", "planned": 4, "ok": 4, "failed": 0,
              "metrics": {"err": {"values": [1.0, 1.1, 0.9, 1.0]}, "want": {"values": [5.0] * 4}}} for h in (1, 2)]
    low = pv.resolve("err @ h=each | vs want", pv.from_cells(cells))
    assert low.filled and low.text.startswith("below want at all 2 values of h")


def test_vs_refuses_what_it_cannot_judge(s5):
    for key, why in (
        ("Ea_hat:se @ sigma=each, n_points=40 | vs Ea_true", "has none"),
        ("Ea_hat @ sigma=each, n_points=40 | vs Ea_hat", "measured, not set by the setting"),
        ("Ea_hat:sd @ sigma=each, n_points=40 | vs Ea_sd_theory | range", "add no reducer"),
        ("Ea_hat:sd @ sigma=0.1 | vs Ea_sd_theory", "does not name every axis"),
        ("Ea_hat:sd @ sigma=each, n_points=40 | vs @ sigma=0.1, n_points=40", "not with another setting"),
    ):
        got = pv.resolve(key, s5)
        assert not got.filled and why in got.hit.reason, key


def test_vs_names_one_setting_when_no_axis_is_marked_each(s5):
    one = pv.resolve("Ea_hat:sd @ sigma=0.1, n_points=40 | vs Ea_sd_theory", s5)
    assert one.filled, one.hit
    assert one.text.startswith("consistent with Ea_sd_theory (inside the measured 95% interval) at that setting")
    assert " to " not in one.text.split("(")[-1]  # one setting: one ratio, not a range of ratios
    # the same setting taken from the "each" form gives the same verdict
    each = pv.resolve("Ea_hat:sd @ sigma=each, n_points=40 | vs Ea_sd_theory", s5)
    assert "consistent with" in each.text
    cells = [{"cell": {"h": h}, "key": f"h={h}", "planned": 4, "ok": 4, "failed": 0,
              "metrics": {"err": {"values": [1.0, 1.1, 0.9, 1.0]}, "want": {"values": [5.0] * 4}}} for h in (1, 2)]
    low = pv.resolve("err @ h=2 | vs want", pv.from_cells(cells))
    assert low.filled and low.text.startswith("below want at that setting")


def test_vs_takes_a_form_that_carries_its_interval_as_the_statistic_itself(s5):
    for form, plain in (("mean_ci", "mean"), ("sd_ci", "sd")):
        a = pv.resolve(f"Ea_hat:{form} @ sigma=each, n_points=40 | vs Ea_sd_theory", s5)
        b = pv.resolve(f"Ea_hat:{plain} @ sigma=each, n_points=40 | vs Ea_sd_theory", s5)
        assert a.filled and a.text == b.text, (form, a.hit)
    one = pv.resolve("Ea_hat:sd_ci @ sigma=0.1, n_points=40 | vs Ea_sd_theory", s5)
    assert one.filled and "at that setting" in one.text
    assert "has none" in pv.resolve("Ea_hat:se @ sigma=each, n_points=40 | vs Ea_true", s5).hit.reason


def test_vs_with_an_all_axis_and_no_each_is_refused_not_worded_as_one_setting(s5):
    for key in ("Ea_hat:sd @ sigma=all, n_points=40 | vs Ea_sd_theory",
                "Ea_hat:sd_ci @ sigma=all, n_points=40 | vs Ea_sd_theory"):
        got = pv.resolve(key, s5)
        assert not got.filled, key
        assert "pool 5 settings into one interval" in got.hit.reason
        assert "sigma=each" in got.hit.reason and "or give it one value" in got.hit.reason


def test_vs_pooling_unlike_settings_is_refused_where_the_reference_is_the_same_at_each():
    cells = [{"cell": {"h": h}, "key": f"h={h}", "planned": 4, "ok": 4, "failed": 0,
              "metrics": {"err": {"values": [1.0, 1.1, 0.9, 1.0]}, "want": {"values": [5.0] * 4}}} for h in (1, 2)]
    got = pv.resolve("err @ h=all | vs want", pv.from_cells(cells))
    assert not got.filled and "with h=all it would pool 2 settings into one interval" in got.hit.reason
    assert pv.resolve("err @ h=each | vs want", pv.from_cells(cells)).filled


def _two_step_cells(first: list[float], second: list[float]) -> list[dict]:
    return [{"cell": {"h": h}, "key": f"h={h}", "planned": len(v), "ok": len(v), "failed": 0,
             "metrics": {"err": {"values": v}}} for h, v in ((1, first), (2, second))]


def test_a_mean_over_all_of_an_axis_the_result_changes_with_is_refused():
    changing = pv.from_cells(_two_step_cells([1.0, 1.1, 0.9, 1.0], [5.0, 5.1, 4.9, 5.0]))
    got = pv.resolve("err @ h=all", changing)
    assert not got.filled
    assert "err changes with h (from 1.000 to 5.000 over its values)" in got.hit.reason
    assert "describes none of them: give h one value, or mark it =each" in got.hit.reason
    assert pv.resolve("err @ h=2", changing).filled and pv.resolve("err @ h=each | range", changing).filled


def test_a_mean_over_all_of_an_axis_the_result_does_not_change_with_still_fills():
    same = pv.from_cells(_two_step_cells([1.0, 1.1, 0.9, 1.0], [1.05, 0.95, 1.1, 0.9]))
    assert pv.resolve("err @ h=all", same).filled  # the two intervals overlap
    flat = pv.from_cells(_two_step_cells([2.0], [2.0]))
    flat.drew = False
    assert pv.resolve("err @ h=all", flat).filled  # a deterministic value that is the same at each setting


def test_one_run_each_over_an_axis_is_refused_only_when_no_trial_drew_random_numbers():
    rec = pv.from_cells(_two_step_cells([2.0], [3.0]))
    assert rec.drew is None and pv.resolve("err @ h=all", rec).filled  # the record does not say: today's disclosed pool
    rec.drew = True
    assert "mean over 2 settings, one run each" in pv.resolve("err @ h=all", rec).text  # differences may be noise
    rec.drew = False
    got = pv.resolve("err @ h=all", rec)
    assert not got.filled and "err changes with h (from 2.000 to 3.000 over its values)" in got.hit.reason


def test_vs_with_each_and_all_on_another_axis_keeps_its_behaviour(s5):
    got = pv.resolve("Ea_hat:sd @ sigma=each, n_points=all | vs Ea_sd_theory", s5)
    assert "pool" not in (got.hit.reason if got.hit else "")
    assert got.text == "not recorded" or " values of sigma" in got.text


def test_vs_with_a_varied_axis_left_unnamed_is_still_refused(s5):
    got = pv.resolve("Ea_hat:sd @ n_points=40 | vs Ea_sd_theory", s5)
    assert not got.filled and "does not name every axis" in got.hit.reason


def test_vs_with_a_number_typed_as_the_operand_says_what_to_write_instead(s5):
    for number in ("0.04", "1e-3", "50", "-2.5"):
        got = pv.resolve(f"Ea_hat:sd @ sigma=0.1, n_points=40 | vs {number}", s5)
        assert not got.filled
        assert "vs ref" in got.hit.reason and "{{calc:EXPR}}" in got.hit.reason and number in got.hit.reason, number
        assert "no trials of" not in got.hit.reason
    # a quantity's name that merely starts with a digit-free word is still a result id
    assert "measured, not set" in pv.resolve("Ea_hat @ sigma=each, n_points=40 | vs Ea_hat", s5).hit.reason


def test_vs_ref_with_no_plan_reference_is_still_refused_at_one_setting():
    cells = [{"cell": {"h": h}, "key": f"h={h}", "planned": 4, "ok": 4, "failed": 0,
              "metrics": {"err": {"values": [1.0, 1.1, 0.9, 1.0]}}} for h in (1, 2)]
    got = pv.resolve("err @ h=2 | vs ref", pv.from_cells(cells))
    assert not got.filled and "not computable at h = 2" in got.hit.reason and "plan" in got.hit.reason


def test_vs_leaves_the_existing_ratio_and_range_forms_alone(s5):
    assert pv.resolve("Ea_hat:sd @ sigma=each, n_points=40 | ratio Ea_sd_theory | range", s5).text == "0.914 to 0.927"
    assert pv.resolve("Ea_hat:sd @ sigma=each, n_points=5 | ratio Ea_sd_theory | range", s5).text == "1.20 to 1.26"
    assert pv.resolve("Ea_sd_theory @ sigma=0.1, n_points=40", s5).text == "2.168"


def test_the_sheet_and_the_rules_tell_the_writer_to_use_vs(s5):
    assert "| vs " in pv.sheet(s5).split("Compare with a value each setting sets")[1]
    rules = " ".join(pv.WRITER_RULES.split())
    assert "| vs <theory value>" in rules and "Do not type above, below" in rules


# --- a quantity FI left out is not offered or filled; an axis named by a sentence is keyed by FI's short name ----------

_SENTENCE = "uncorrected, Bonferroni and Benjamini-Hochberg applied to the same simulated p-values"


def _tiny_cells(axis_names=("m", "method")) -> list[dict]:
    a, b = axis_names
    cells = []
    for m in (20, 200):
        for method in ("uncorrected", "bh"):
            cells.append({"cell": {a: m, b: method}, "key": f"{m}-{method}",
                          "metrics": {"fdr": {"values": [0.04 + m / 10000, 0.05, 0.06, 0.05]},
                                      "bh_fixed_rejections": {"values": [3.0, 3.0, 3.0, 3.0]}}})
    return cells


def _quest(tmp_path: Path, cells, **records) -> Path:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "needs").mkdir(exist_ok=True)
    for name, body in records.items():
        (tmp_path / "needs" / name).write_text(json.dumps(body), encoding="utf-8")
    return tmp_path


def test_a_quantity_the_results_check_left_out_is_not_offered_or_filled(tmp_path):
    q = _quest(tmp_path, _tiny_cells(), **{"RESULT_CHECK.json": {
        "removed_quantities": ["bh_fixed_rejections"], "note": "it is 3 at every setting"}})
    rec = pv.records_of(q, None)
    assert "bh_fixed_rejections" not in rec.metrics()
    assert "bh_fixed_rejections" not in pv.sheet(rec)
    got = pv.resolve("{{v:bh_fixed_rejections:mean @ m=20, method=all}}", rec)
    assert not got.filled and got.text == pv.NOT_RECORDED
    assert "FI left bh_fixed_rejections out of the results" in got.hit.reason and "3 at every setting" in got.hit.reason
    # positive: another quantity of the same record is still offered and filled
    ok = pv.resolve("{{v:fdr:mean @ m=20, method=all}}", rec)
    assert ok.filled and ok.text != pv.NOT_RECORDED
    assert "{{v:fdr:mean_ci @ m=20, method=all}}" in pv.sheet(rec)


def test_a_quantity_the_typed_number_check_left_out_is_not_filled_and_a_vs_operand_too(tmp_path):
    q = _quest(tmp_path, _tiny_cells(), **{"TYPED_RESULTS_CHECK.json": {
        "removed_quantities": ["bh_fixed_rejections"], "note": "typed into the code"}})
    rec = pv.records_of(q, None)
    assert not pv.resolve("{{v:bh_fixed_rejections @ m=20, method=all}}", rec).filled
    assert not pv.resolve("{{v:fdr @ m=20, method=all | diff bh_fixed_rejections}}", rec).filled
    assert pv.resolve("{{v:fdr @ m=20, method=all}}", rec).filled


def test_without_either_record_nothing_is_left_out(tmp_path):
    rec = pv.records_of(_quest(tmp_path, _tiny_cells()), None)
    assert rec.left_out == {} and "bh_fixed_rejections" in rec.metrics()
    assert pv.resolve("{{v:bh_fixed_rejections @ m=20, method=all}}", rec).filled


def test_one_function_names_what_was_left_out_for_the_analysis_and_the_paper(tmp_path):
    from core import result_checks as rc

    q = _quest(tmp_path, _tiny_cells(), **{"RESULT_CHECK.json": {"removed_quantities": ["a"], "note": "n1"},
                                           "TYPED_RESULTS_CHECK.json": {"removed_quantities": ["b"], "note": "n2"}})
    assert rc.left_out_by_record(q) == {"a": "n1", "b": "n2"}


def test_an_axis_named_by_a_sentence_is_keyed_by_its_short_name(tmp_path):
    cells = _tiny_cells(("m", _SENTENCE))
    q = _quest(tmp_path, cells)
    rec = pv.records_of(q, {"in_each_run": {_SENTENCE: ["uncorrected", "bh"]}})
    assert rec.aliases()[_SENTENCE] == "compared" and rec.aliases()["m"] == "m"
    sheet = pv.sheet(rec)
    assert _SENTENCE not in sheet and "compared=" in sheet
    keys = [k for k in __import__("re").findall(r"\{\{v:[^{}]*\}\}", sheet) if "@" in k and "<" not in k]
    assert keys and all(pv.resolve(k, rec).filled for k in keys)
    assert pv.resolve("{{v:fdr @ m=20, compared=bh}}", rec).filled
    # the short name is told to the writer, not the sentence
    assert "compared" in pv.writer_block(rec).split("The settings it varied:")[1].splitlines()[0]
    # a missing axis is reported by its short name
    assert "compared" in pv.resolve("{{v:fdr @ m=20}}", rec).hit.reason


def test_axes_with_plain_names_are_unchanged_by_the_short_names(tmp_path):
    rec = pv.records_of(_quest(tmp_path, _tiny_cells()), None)
    assert rec.aliases() == {"m": "m", "method": "method"}
    assert pv.resolve("{{v:fdr @ m=20, method=bh}}", rec).filled


def _level_cells(label_for_a: str, with_values: bool) -> list[dict]:
    out = []
    for label, shift in ((label_for_a, 0.0), ("b", 1.0)):
        cell = {"dose": 1, "mode": label}
        vals = [1.0 + shift, 2.0 + shift] if with_values else []
        out.append({"cell": cell, "key": f"dose=1,mode={label}", "level": {"mode": label}, "planned": 2, "ok": len(vals), "failed": 0,
                    "metrics": {"rate": {"values": vals, "trials": list(range(len(vals)))}}})
    return out


def test_a_result_recorded_per_compared_value_is_fillable_when_the_values_name_has_an_equals_sign() -> None:
    protocol = {"metrics": [{"id": "rate", "role": "result", "kind": "mean"}]}
    rec = pv.from_cells(_level_cells("T_0 = 300 (exact)", True), protocol)
    assert "rate" in rec.fillable()
    value = pv.resolve("{{v:rate @ dose=1, mode=T_0 = 300 (exact)}}", rec)
    assert value.filled and value.text.startswith("1.5")


def test_a_result_with_no_value_in_any_trial_is_still_not_fillable() -> None:  # the known positive
    protocol = {"metrics": [{"id": "rate", "role": "result", "kind": "mean"}]}
    rec = pv.from_cells(_level_cells("T_0 = 300 (exact)", False), protocol)
    assert "rate" not in rec.fillable()


# --- a study with one trial per setting is offered no interval key -------------------------------------------------------

def _one_trial_cells() -> list[dict]:
    return [{"cell": {"dose": d, "steps": s}, "key": f"{d}-{s}", "metrics": {"gain": {"values": [d / 10 + s / 1000]}}}
            for d in (1, 2, 3) for s in (16, 64)]


def test_one_trial_per_setting_offers_the_mean_not_an_interval(tmp_path):
    rec = pv.records_of(_quest(tmp_path, _one_trial_cells()), None)
    block = pv.writer_block(rec)
    sheet = pv.sheet(rec)
    assert "{{v:gain @ dose=1, steps=16}}" in sheet and "{{v:gain @ dose=1, steps=all}}" in sheet
    assert ":mean_ci @" not in sheet and ":mean_ci @" not in block.split("The study's quantities")[0]
    assert any(l.startswith("- {{v:gain @ dose=1, steps=all}}") and "mean over 2 settings, one run each; no interval" in l for l in sheet.splitlines())
    assert pv.resolve("{{v:gain @ dose=1, steps=16}}", rec).filled
    assert not pv.resolve("{{v:gain:mean_ci @ dose=1, steps=16}}", rec).filled  # the review's handling is unchanged


def test_a_study_with_repeated_trials_is_offered_the_interval_as_before(tmp_path):  # the known positive
    rec = pv.records_of(_quest(tmp_path, _tiny_cells()), None)
    assert ":mean_ci @ m=20, method=all}}" in pv.sheet(rec)
    assert "{{v:coverage:mean_ci @ noise=0.05, n=20}}" in pv.writer_block(rec)
    assert "one trial in each setting" not in pv.writer_block(rec)


def test_a_pooled_plain_mean_of_one_run_settings_keeps_its_note_in_the_paper(tmp_path):
    rec = pv.records_of(_quest(tmp_path, _one_trial_cells()), None)
    got = pv.resolve("{{v:gain @ dose=1, steps=all}}", rec)
    assert got.filled and got.text == "0.1400 (mean over 2 settings, one run each; no interval)", got.text
    assert pv.resolve("{{v:gain @ dose=1, steps=16}}", rec).text == "0.1160"  # one setting: the value itself
    sheet_line = next(l for l in pv.sheet(rec).splitlines() if l.startswith("- {{v:gain @ dose=1, steps=all}}"))
    assert sheet_line.split(" -> ")[1].startswith(got.text)  # the sheet shows what the key fills


def test_a_mixed_study_does_not_offer_its_single_trial_cells_as_plain_means(tmp_path):
    cells = _one_trial_cells()
    cells[0]["metrics"]["gain"]["values"] = [0.1, 0.12, 0.11]  # one setting repeated, five single-trial
    rec = pv.records_of(_quest(tmp_path, cells), None)
    sheet = pv.sheet(rec)
    assert "{{v:gain:mean_ci @ dose=1, steps=16}}" in sheet  # the repeated setting keeps its interval key
    assert "{{v:gain @ dose=2, steps=16}}" not in sheet and "{{v:gain:mean_ci @ dose=2, steps=16}}" not in sheet
    assert "one trial in each setting" not in pv.writer_block(rec)


# --- a result the plan declares exact is printed as the record holds it (backlog -420) ----------------------------------


def _ex_metric(vals):
    return {"values": list(vals), "trials": list(range(len(vals))), "count": len(vals), "total": float(sum(vals)),
            "non_finite": 0}


def _ex_cells(**series):
    n = max(len(v) for v in series.values())
    return [{"cell": {"size": 100, "method": "direct"}, "key": "size=100,method=direct", "planned": n, "ok": n, "failed": 0,
             "metrics": {k: _ex_metric(v) for k, v in series.items()}}]


def _ex_plan(**exact):
    return {"metrics": [{"id": k, "kind": "mean", "estimand": "x", "unit": "run", **({} if v is None else {"exact": v})}
                        for k, v in exact.items()]}


def _ex_text(cells, plan, name):
    return pv.resolve("{{v:" + name + " @ size=100, method=direct}}", pv.from_cells(cells, plan)).text


# the variation table: (series, what the plan says, expected text); each row is a differently shaped result
@pytest.mark.parametrize("values, said, want", [
    ([52817.0], True, "52817"),                                  # a count, one trial
    ([52817.0] * 3, True, "52817"),                              # a count, three identical trials
    ([0.52817] * 3, True, "0.52817"),                            # a share of counts
    ([0.5281700000001] * 3, True, "0.5281700000001"),            # written as the record holds it, every digit
    ([123456.0], True, "123456"),                                # six digits
    ([0.12345678], False, "0.1235"),                             # a one-trial measurement the plan calls a measurement
    ([0.12345678], None, "0.1235"),                              # no word from the plan, not whole: four figures as before
    ([14281.0], None, "14281"),                                  # no word from the plan, whole: never another whole number
    ([14281.0] * 2, False, "14280"),                         # the plan calls it a measurement: believed
    ([2.16834067696] * 2, True, "2.16834067696"),                # exact as the plan says, whatever the digits look like
    ([2.16834067696] * 2, None, "2.168"),                        # undeclared and not whole: four figures, as before
    ([1.2345678, 1.2445678, 1.2145678], True, "1.231"),         # declared exact but the trials differ: measured, 4 figures
])
def test_exact_is_what_the_plan_says_and_the_record_agrees(values, said, want):
    assert _ex_text(_ex_cells(tally=values), _ex_plan(tally=said), "tally") == want


def test_declared_exact_with_differing_trials_is_logged(caplog):
    with caplog.at_level("WARNING"):
        _ex_text(_ex_cells(tally=[1.0, 2.0, 4.0]), _ex_plan(tally=True), "tally")
    assert "declared exact, but its trials differ" in caplog.text


def test_a_measurement_keeps_its_four_figures_and_interval():
    cells = _ex_cells(tally=[52817.0] * 3, noisy=[1.2345678, 1.2445678, 1.2145678])
    plan = _ex_plan(tally=True, noisy=False)
    rec = pv.from_cells(cells, plan)
    assert pv.resolve("{{v:noisy @ size=100, method=direct}}", rec).text == "1.231"
    noisy = pv.resolve("{{v:noisy:mean_ci @ size=100, method=direct}}", rec)  # the known positive: unchanged
    assert noisy.filled and "1.23" in noisy.text and "1.2312" not in noisy.text


def _ex_two(a, b, plan):
    cells = _ex_cells(tally=[a])
    cells.append({**cells[0], "cell": {"size": 100, "method": "other"}, "key": "size=100,method=other",
                  "metrics": {"tally": _ex_metric([b])}})
    return pv.from_cells(cells, plan)


def test_range_with_a_declared_exact_end_prints_both_ends_in_full():
    got = pv.resolve("{{v:tally @ size=100, method=each | range}}", _ex_two(52817.0, 41.0, _ex_plan(tally=True)))
    assert got.filled and got.text == "41 to 52817"
    got = pv.resolve("{{v:tally @ size=100, method=each | range}}", _ex_two(0.14281, 0.33333, _ex_plan(tally=True)))
    assert got.filled and got.text == "0.14281 to 0.33333"


def test_range_with_no_word_from_the_plan_keeps_a_non_whole_end_at_four_figures():
    got = pv.resolve("{{v:tally @ size=100, method=each | range}}", _ex_two(52817.0, 0.33333, _ex_plan(tally=None)))
    assert got.filled and got.text == "0.3333 to 52817"


def test_metric_spec_and_contract_carry_exact():
    from core import contract, metric_spec

    base = {"id": "tally", "kind": "mean", "estimand": "x", "unit": "run"}
    ok, why = metric_spec.normalize([{**base, "exact": True}])
    assert why is None and ok[0]["exact"] is True
    for bad in ("yes", 1, "true"):
        assert metric_spec.normalize([{**base, "exact": bad}])[0] is None
    assert metric_spec.normalize([base])[1] is None  # left out is allowed


@pytest.mark.parametrize("x, want", [
    (0.123456789012345, "0.123456789012345"),   # past 12 digits: written as the record holds it
    (1e15, "1000000000000000"),
    (123456789012345678.0, "123456789012345680"),  # the float's own integer value, in full
    (0.30000000000000004, "0.30000000000000004"),
    (-0.0, "0"),
    (2.0, "2"),
])
def test_exact_text_is_the_value_in_full(x, want):
    assert pv.exact_text(x) == want


def test_exact_percent_adds_no_float_noise():
    m = pv.Measured(kind="exact", value=0.0712345)
    assert pv.fmt(m, percent=True) == "7.12345%"
