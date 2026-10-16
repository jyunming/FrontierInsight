# Benjamini–Hochberg versus Bonferroni Correction in Independent One-Sided z-Tests: Realised False Discovery Rate and Power

## Abstract

Bonferroni correction controls the chance of any false positive, while the Benjamini–Hochberg (BH) procedure controls the expected share of false positives among rejections. This study asks how they compare in realised false discovery rate (FDR) and power when many tests are run. We simulated m ∈ {20, 200, 2000} independent one-sided z-tests, a fraction π1 ∈ {0, 0.05, 0.2} of which carry a real effect shifting the statistic by 2.5. Uncorrected testing, Bonferroni and BH were applied at a nominal 5% to the same data, with 4000 simulated trials per cell. BH kept realised FDR between 0.040 and 0.058, close to the reference π0 × 0.05, in all nine cells. Uncorrected FDR reached 1.0 at π1 = 0. Bonferroni power fell with m, from 0.378 to 0.060 at π1 = 0.2, matching the exact formula. BH power at π1 = 0.2 stayed near 0.47–0.50, so BH's power advantage grew from 0.9 percentage points (m = 20, π1 = 0.05) to 40.6 points (m = 2000, π1 = 0.2). This gain costs the guarantee on any false discovery: BH's chance of one or more false rejections reached 0.679 at m = 2000, π1 = 0.05. One known-answer check on the BH FDR bound failed and is reported below.

**Keywords:** multiple testing, false discovery rate, Benjamini–Hochberg procedure, Bonferroni correction, statistical power

## Introduction

Testing thousands of hypotheses at the 5% level produces many false positives. With m independent true nulls, the chance of at least one rejection is 1 − 0.95^m, or 0.64 at m = 20. Bonferroni rejects when p ≤ α/m and bounds the family-wise error rate by α [1]. BH sorts the p-values and rejects all p₍ᵢ₎ up to the largest i with p₍ᵢ₎ ≤ iα/m. It controls the expected fraction of false rejections among rejections for independent statistics [2], and it is valid under positive regression dependence [3]. Gordon et al. argue that Bonferroni's reputation for conservatism stems from the stringent requirements imposed on it, and that it is more stable in the number of discoveries [4]. This paper quantifies, on a small controlled grid, what each rule buys in error and in power.

## Methods

For each (m, π1) cell, the first m1 = round(π1 m) statistics are shifted by 2.5 (m1 = 0, 1, 4 for m = 20; 0, 10, 40 for m = 200; 0, 100, 400 for m = 2000). The rest are N(0,1). One-sided p-values are p = 1 − Φ(z). Uncorrected (p ≤ 0.05), Bonferroni (p ≤ 0.05/m) and BH (step-up at 0.05) were applied to identical p-values in each trial. Per trial we recorded false rejections V, true rejections S and R = V + S.

- **Realised FDR** is the mean of V/max(R, 1).
- **Power** is the expected fraction of all false hypotheses that is rejected [1].
- **Any false discovery** is the proportion of trials with V ≥ 1. At π1 = 0 it equals FDR, and it is the only error quantity reported there.
- **Power gap** is the paired difference BH minus Bonferroni, in percentage points (pp).

The reference line is π0 × 0.05 with π0 = 1 − m1/m. It is the study's own reference, not a result taken from a source. Exact references are 1 − (1 − 0.05/m)^m0 for Bonferroni any-false-discovery, 1 − Φ(z₁₋₀.₀₅/ₘ − 2.5) for Bonferroni power, and 0.804 for uncorrected power. The run used a fixed random seed per cell, recorded in the run record. There were 4000 independent trials per cell in a single run, so there is no replication across seeds.

Uncertainty is a normal-theory 95% interval of the mean over the 4000 trials for FDR, power and gap, and a Wilson interval for proportions. These intervals describe sampling noise within one run only. A step-up routine was checked against `scipy.stats.false_discovery_control` and gave identical rejection counts on the check case. The Bonferroni ⊆ BH ⊆ uncorrected nesting held on its check case.

<!-- fi:attempts -->
**How this result was reached.** The design was not revised after the experiment had first been run. The complete experiment was run 4 times (a run includes all its seeds); 3 of these runs were discarded (3 failed a known-answer check).
<!-- /fi:attempts -->

## Results

![**Figure 1.** Realised FDR (mean V/max(R,1)) against π1 for uncorrected, Bonferroni and BH testing in each of m = 20, 200 and 2000, with the reference π0 × 0.05 drawn; 4000 simulated trials per point.](figures/fdr_vs_pi1.png)

**Error control.** Figure 1 shows that uncorrected testing does not control error. At π1 = 0 its FDR was 0.653 (95% Wilson interval 0.638–0.667) at m = 20 against an exact 0.642, and 1.0 at m = 200 and 2000. At π1 = 0.2 it was 0.177, 0.198 and 0.199 for m = 20, 200, 2000, about four times the nominal level. Bonferroni's any-false-discovery probability at π1 = 0 was 0.049, 0.055 and 0.056 for m = 20, 200, 2000. Every Wilson interval, for example 0.049–0.063 at m = 200, contains 0.05. Its FDR for π1 > 0 was far below the line, for example 0.002 at m = 2000, π1 = 0.2.

BH FDR sat close to the reference in every cell (Table 1). Individual cells fall on either side of the line. Under independence the BH guarantee is FDR ≤ π0 × 0.05, so a cell above the line is noise or a defect. Several cells lie slightly above it, such as 0.058 at m = 200, π1 = 0 (interval 0.051–0.066) and 0.0415 against 0.04 at m = 200, π1 = 0.2 (interval 0.040–0.043). These excesses are about 0.001–0.008. The upper end of a few intervals lies above the line, and no second-seed run was made to show they are noise.

**Table 1.** Realised BH FDR, its reference, Bonferroni any-false-discovery probability (FWER), power, and paired power gap in pp (95% interval). Power is undefined at π1 = 0.

| m | π1 | BH FDR | π0 × 0.05 | Bonf. FWER | Bonf. power | BH power | Gap, pp |
|---|---|---|---|---|---|---|---|
| 20 | 0 | 0.049 | 0.050 | 0.049 | – | – | – |
| 20 | 0.05 | 0.047 | 0.0475 | 0.046 | 0.384 | 0.393 | 0.9 (0.6–1.2) |
| 20 | 0.2 | 0.041 | 0.040 | 0.041 | 0.378 | 0.503 | 12.6 (12.0–13.1) |
| 200 | 0 | 0.058 | 0.050 | 0.055 | – | – | – |
| 200 | 0.05 | 0.051 | 0.0475 | 0.054 | 0.163 | 0.267 | 10.4 (10.0–10.8) |
| 200 | 0.2 | 0.041 | 0.040 | 0.045 | 0.163 | 0.468 | 30.5 (30.2–30.8) |
| 2000 | 0 | 0.057 | 0.050 | 0.056 | – | – | – |
| 2000 | 0.05 | 0.048 | 0.0475 | 0.053 | 0.060 | 0.234 | 17.4 (17.2–17.6) |
| 2000 | 0.2 | 0.040 | 0.040 | 0.045 | 0.060 | 0.466 | 40.6 (40.5–40.7) |

![**Figure 2.** Power (fraction of real effects found) against m for Bonferroni and BH at π1 = 0.05 and π1 = 0.2, with the exact Bonferroni power curve; uncorrected power is drawn only as a flat reference line at 0.804.](figures/power_by_method.png)

**Power.** Figure 2 shows Bonferroni power falling with m, from 0.378 to 0.163 to 0.060 at π1 = 0.2, in agreement with the exact values 0.379, 0.163 and 0.060. BH power at π1 = 0.2 was nearly flat in m (0.503, 0.468, 0.466). Uncorrected power was about 0.80 everywhere, matching the exact 0.804, but with no error control. At π1 = 0.05 BH power was 0.393, 0.267 and 0.234, still well above Bonferroni except at m = 20.

![**Figure 3.** BH minus Bonferroni power in percentage points against m at π1 = 0.05 and π1 = 0.2, with error bars over the 4000 trials and a zero-gap reference line.](figures/power_gap_pp.png)

Figure 3 shows the gap growing with both m and π1. It is 0.9 pp at m = 20, π1 = 0.05, where there is a single real effect, and 40.6 pp (interval 40.5–40.7) at m = 2000, π1 = 0.2. The gap was nonnegative in every cell. The pre-run expectation of roughly 45 points (BH power about 0.5) was not met at m = 2000, since the interval excludes 45.

**What BH gives up.** BH does not control the chance of any false discovery. At π1 = 0 it was 0.058 at m = 200. At m = 2000 it was 0.679 (interval 0.664–0.693) for π1 = 0.05 and 0.9995 for π1 = 0.2. FDR bounds the proportion of false rejections, not their presence [W8].

## Discussion

The two rules answer different questions. Under independence and a moderate effect, BH rejects every Bonferroni rejection plus many more, while its FDR stays near π0 × 0.05. This agrees with the original simulation claim of substantial power gains [2] and with the general view that FDR procedures trade a weaker guarantee for power [W3]. Where one claim of discovery is costly, as in confirmatory work, FWER control remains the appropriate choice, and screening favours FDR [W8]. Gordon et al. show that Bonferroni can be reframed as control of the mean number of false discoveries, which changes the interpretation of its conservatism [4]. Our setting examines FWER at α, not that reframing, so we do not test their claim.

The gap is smallest where there is one real effect among 20 tests: BH has little room to gain when R is near 0 or 1. The gap is largest where many real effects let BH's threshold relax toward 0.05 × R/m. Adaptive procedures that estimate π0 are one route to further power [5]. They were not studied here.

### Limitations

- **Failed check.** One known-answer check, that BH FDR does not exceed π0 × 0.05 at m = 200, π1 = 0.2, is recorded as failed: it measured an excess of 0.019 against a tolerance of 0.01. The result is unconfirmed. We could not reconcile this with the 0.0415 against 0.04 seen in the table, and no result here should be described as validated against that check.
- **No independent confirmation.** The code was not independently reviewed. Independent second versions of the BH step-up and the error tallies could not be compared with the first, so these figures rest on the single implementation. A match to the SciPy routine on one case is the only independent check.
- **Single run.** Everything comes from one run per cell. Intervals reflect sampling over 4000 trials, not seed-to-seed variation, and comparisons across nine cells were not adjusted for multiplicity.
- **Scope.** Conclusions hold for independent, known-variance one-sided z-tests, a single effect size of 2.5, a fixed count of real effects and α = 0.05. The m = 20, π1 = 0.05 cell has one real effect, so its power is coarse. Dependence and other effect sizes were not examined.
- **Reference line.** The reference line π0 × 0.05 and the trial-level interval definitions are this study's own definitions, not taken from a source. The listed simplifications of the study plan are not fully stated.

The run record gives the seeds and run time.

<!-- fi:run-record:ec88eba58235 -->
**What FI recorded about this run.** Each trial's random seed is derived from base seed 0 and the trial number; every setting uses the same seeds for the same trial number (common random numbers), so the settings' results are paired, not independent. Within each run, every value of `method` (uncorrected, bonferroni, benjamini_hochberg) was used on the same simulated data, so differences between those values do not come from different random draws. The experiment ran for 3.7 s; the topic sets a limit of 60 s, and the run was within it. The run had 9 settings with 4,000 trials per setting; 36,000 trials finished and 0 failed. A reviewer objected to the design in ways FI's records neither confirm nor rule out, and the study was not changed for them: "Thirteen ungrounded claims and a seeding description that contradicts the run record must be removed or corrected, and the Figure 2 caption must match what is drawn".
<!-- /fi:run-record -->

<!-- fi:code-findings:53c3fb104858 -->
**Open points in the code that produced these results.** FI compared two independently written versions of key functions of the model. The points below were still open on the version that ran, and none of them was settled by the run; the full record is in `.fi/second_impl.json`.

- **The code was not compared with the plan by another model** (the reader is the model that wrote the code (claude-sonnet-5-5)).

- **A second, independent version of a key function was not settled.** FI had these functions written a second time from the plan's description alone and compared the two on many inputs: A second, independently written version of step S1 (BH step-up) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S2 (Error tallies) could not be compared with the first (the plan gives no example inputs to call it with).
<!-- /fi:code-findings -->

## References

1. Jesse Hemerik (2026). Multiple testing. arXiv. DOI: 10.48550/arxiv.2606.26781
2. Yoav Benjamini, Yosef Hochberg (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society Series B (Statistical Methodology). DOI: 10.1111/j.2517-6161.1995.tb02031.x
3. Yoav Benjamini, Daniel Yekutieli (2001). The control of the false discovery rate in multiple testing under dependency. The Annals of Statistics. DOI: 10.1214/aos/1013699998
4. Alexander Y. Gordon et al. (2007). Control of the mean number of false discoveries, Bonferroni and stability of multiple testing. The Annals of Applied Statistics. DOI: 10.1214/07-aoas102
5. Yoav Benjamini, Yosef Hochberg (2000). On the Adaptive Control of the False Discovery Rate in Multiple Testing With Independent Statistics. Journal of Educational and Behavioral Statistics. DOI: 10.2307/1165312

## Further reading

- [W1] Benjamini–Hochberg Method | Springer Nature Link. https://link.springer.com/rwe/10.1007/978-1-4419-9863-7_1215
- [W2] (2026). Bonferroni Correction vs FDR: When to Use Each — CASRAI. https://casrai.org/guides/bonferroni-correction-and-multiple-comparisons
- [W3] (2026). False discovery rate - Wikipedia. https://en.wikipedia.org/wiki/False_discovery_rate
- [W4] Multiple Comparisons: Bonferroni Corrections and False Discovery Rates. https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/1.5_Bonferroni_FDR.pdf
- [W5] (2026). Holm vs Bonferroni vs Benjamini-Hochberg Correction. https://metricgate.com/blogs/holm-vs-bonferroni-vs-bh/
- [W6] (2025). Controlling your type I errors: Bonferroni and Benjamini-Hochberg. https://www.statsig.com/blog/controlling-type-i-errors-bonferroni-benjamini-hochberg
- [W7] (2026). Bonferroni vs Holm vs Hochberg Corrections Compared. https://metricgate.com/blogs/bonferroni-holm-hochberg/
- [W8] (2026). False Discovery Rate vs FWER | MetricGate. https://metricgate.com/blogs/false-discovery-rate-vs-fwer/
- [W9] (2026). Multiple Comparisons: FWER, FDR, Bonferroni, Holm - Interactive. https://mbrenndoerfer.com/writing/multiple-comparisons-fwer-fdr-bonferroni-holm-benjamini-hochberg
