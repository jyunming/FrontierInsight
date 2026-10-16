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


KEYS = {
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}}": ("0.824%", dict(noise_sigma=0.01)),
    "{{v:Ea_rel_bias:sd @ noise_sigma=0.05, n_points=all | %}}": ("4.13%", dict(noise_sigma=0.05)),
    "{{v:Ea_rel_bias:sd @ noise_sigma=all, n_points=5 | %}}": ("6.05%", dict(n_points=5)),
    "{{v:Ea_rel_bias:sd @ noise_sigma=all, n_points=10 | %}}": ("4.28%", dict(n_points=10)),
}


@pytest.mark.parametrize("key", list(KEYS))
def test_s2_sd_keys_render_and_match_numpy(s2, key):
    text, where = KEYS[key]
    got = pv.resolve(key, s2)
    assert got.filled and got.text.startswith(text + " (95% chi-square interval ")
    assert got.numbers[0] == pytest.approx(_within_sd(_cells("s2_kinetics_trials.json"), "Ea_rel_bias", **where))


def test_s2_ratio_key_is_19_6_and_matches_numpy(s2):
    key = "{{v:Ea_rel_bias:sd @ noise_sigma=0.1, n_points=all | ratio @ noise_sigma=0.005, n_points=all}}"
    got = pv.resolve(key, s2)
    cells = _cells("s2_kinetics_trials.json")
    expected = _within_sd(cells, "Ea_rel_bias", noise_sigma=0.1) / _within_sd(cells, "Ea_rel_bias", noise_sigma=0.005)
    assert got.filled and got.text == "19.6"
    assert got.numbers[0] == pytest.approx(expected)


def test_pooled_sd_is_the_within_setting_sd_not_the_concatenated_one(s2):
    cells = _cells("s2_kinetics_trials.json")
    use = [c for c in cells if c["cell"]["n_points"] == 5]
    concatenated = float(np.std(np.concatenate([c["metrics"]["Ea_rel_bias"]["values"] for c in use]), ddof=1))
    within = _within_sd(cells, "Ea_rel_bias", n_points=5)
    assert abs(concatenated - within) > 1e-5  # the two definitions really differ on this record
    got = pv.resolve("Ea_rel_bias:sd @ noise_sigma=all, n_points=5", s2)
    assert got.numbers[0] == pytest.approx(within) and got.numbers[0] != pytest.approx(concatenated, rel=1e-5)


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
    paper = "SD is {{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}} here."
    filled, hits = pv.fill(paper, s2)
    assert hits == []
    sd = "0.824% (95% chi-square interval 0.793% to 0.859%, 1200 trials)"
    assert filled == ("SD is <!-- fi:v {{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}} -->" + sd +
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
    got = pv.resolve("Ea_rel_bias:sd_ci @ noise_sigma=0.05, n_points=all", s2)
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
    got = pv.resolve("Ea_rel_bias:sd @ noise_sigma=each, n_points=all | argmax", s2)
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
    assert got.filled and got.text == "0.4667"


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
        ("Ea_hat:sd @ sigma=0.1, n_points=40 | vs Ea_sd_theory", "mark at least one axis =each"),
        ("Ea_hat:sd @ sigma=each, n_points=40 | vs @ sigma=0.1, n_points=40", "not with another setting"),
    ):
        got = pv.resolve(key, s5)
        assert not got.filled and why in got.hit.reason, key


def test_vs_leaves_the_existing_ratio_and_range_forms_alone(s5):
    assert pv.resolve("Ea_hat:sd @ sigma=each, n_points=40 | ratio Ea_sd_theory | range", s5).text == "0.914 to 0.927"
    assert pv.resolve("Ea_hat:sd @ sigma=each, n_points=5 | ratio Ea_sd_theory | range", s5).text == "1.20 to 1.26"
    assert pv.resolve("Ea_sd_theory @ sigma=0.1, n_points=40", s5).text == "2.168"


def test_the_sheet_and_the_rules_tell_the_writer_to_use_vs(s5):
    assert "| vs " in pv.sheet(s5).split("Compare with a value each setting sets")[1]
    rules = " ".join(pv.WRITER_RULES.split())
    assert "| vs <theory value>" in rules and "Do not type above, below" in rules
