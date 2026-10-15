# False-Discovery Control and Power of Bonferroni and Benjamini–Hochberg Procedures Under Independent Gaussian Tests

## Abstract

Multiple testing creates false positives when many hypotheses are assessed at a nominal 5% level. This simulation compares uncorrected testing, Bonferroni correction, and the Benjamini–Hochberg (BH) procedure for independent one-sided Gaussian z-tests with a common nonnull shift of 2.5. Families of \(m=20\), 200, or 2,000 tests were simulated with nonnull proportions \(\pi_1=0.05\) or 0.20; complete-null families were evaluated separately. BH realized false-discovery rates (FDRs) of 0.0400–0.0487, closely tracking the independence reference \(\pi_0\alpha\) of 0.0400 or 0.0475.   

**Keywords:** multiple hypothesis testing, Benjamini–Hochberg procedure, Bonferroni correction, false discovery rate, statistical power

## Introduction

Testing many hypotheses at an unadjusted 5% threshold makes at least one false positive increasingly likely as the family grows. Bonferroni addresses this by controlling family-wise error, whereas BH controls the expected fraction of false discoveries among rejected hypotheses [1, 2]. Under independent tests, BH controls FDR at \(\pi_0\alpha\), where \(\pi_0\) is the true-null proportion [3]. This distinction can produce a power advantage when genuine signals exist.

This study evaluates that trade-off for independent one-sided z-tests, comparing realized FDR, family-level power, and the complete-null probability of any false discovery.

## Methods

Each simulated family contained \(m\in\{20,200,2000\}\) independent z-statistics. Exactly \(m\pi_1\) statistics had mean 2.5 and the remainder had mean zero, with unit variance; \(\pi_1\in\{0,0.05,0.20\}\). One-sided p-values were evaluated using uncorrected \(\alpha=0.05\), Bonferroni’s \(\alpha/m\), and the BH step-up rule. All three procedures were applied to the same family.

One run of 4,096 families was generated per setting using fixed random seeds. FDR was defined here as the mean family-level false-discovery proportion, and power as the mean proportion of assigned nonnulls rejected. For \(\pi_1=0\), power is undefined; the reported endpoint is instead the chance of any false discovery. The displayed analytic checks were \(\pi_0\alpha\) for BH FDR, closed-form complete-null family-wise error, and single-test power references for uncorrected and Bonferroni rules.

<!-- fi:attempts -->
**How this result was reached.** The design was not revised after the experiment had first been run. The complete experiment was run 9 times (a run includes all its seeds); 8 of these runs were discarded (3 failed or gave no result, 5 did not follow the protocol).
<!-- /fi:attempts -->

## Results

 Its FDR ranged from 0.0400 to 0.0487, versus references of 0.0400 and 0.0475. 

![**Figure 1.** Realized FDR for uncorrected, Bonferroni, and BH testing across family sizes and nonnull proportions, with the BH reference \(\pi_0\alpha\) shown.](figures/fdr_by_m_pi1.png)

BH’s power exceeded Bonferroni’s in all six strata (Table 1; Figure 2). The gain was negligible when only one of 20 tests was nonnull, but grew with larger families and greater signal prevalence.  Uncorrected testing had power near 0.80, but this came with much higher FDR.

| \(m\) | \(\pi_1\) | BH FDR / reference | BH power | Bonferroni power | BH − Bonferroni power |
|---:|---:|---:|---:|---:|---:|



| 200 | 0.20 | 0.0400 / 0.0400 | 0.4710 | 0.1640 | 0.3070 |
| 2000 | 0.05 | 0.0487 / 0.0475 | 0.2360 | 0.0602 | 0.1758 |
| 2000 | 0.20 | 0.0403 / 0.0400 | 0.4657 | 0.0599 | 0.4058 |

![**Figure 2.** Family-level power by procedure, family size, and nonnull proportion.](figures/power_by_m_pi1.png)

   Figure 3 shows agreement with the analytic references and illustrates why uncorrected testing becomes untenable for large complete-null families.

![**Figure 3.** Complete-null probability of any false discovery for the three procedures across family sizes, with analytic references shown.](figures/complete_null_fwer.png)

## Discussion

The results support the central distinction between FDR and family-wise error control. BH’s observed FDR closely followed \(\pi_0\alpha\), consistent with the independent-test result described by Hu, Zhao, and Zhou [3] and with the original BH framework [2]. In exchange for allowing some false discoveries when signals exist, 

The practical difference depends on the testing objective. Bonferroni’s low power at large \(m\) follows from its fixed \(\alpha/m\) threshold; its conservative behavior is widely recognized in large-scale settings [4].  The finding is also consistent with analyses that characterize BH as more powerful than family-wise-error procedures under appropriate assumptions [5].

 High apparent sensitivity therefore cannot be interpreted without its false-discovery burden.

### Limitations

This study concerns independent, correctly calibrated one-sided Gaussian tests with a common shift of 2.5; it does not establish performance under dependence, heterogeneous effects, two-sided tests, or misspecified null distributions. Dependence can alter FDR behavior and requires separate justification or modified procedures [5].

Only one fixed-seed run was available per setting. Numerical uncertainty intervals were not supplied for every reported estimate, so this paper does not report confidence intervals or claim seed-level robustness. The Bernoulli truth-label probe was limited to two \(m=200\) settings; dependence and unpaired-procedure probes were not conducted.

Several estimands—BH reference FDR, FDR and power uncertainty summaries, complete-null family-wise error, \(\pi_0\), and paired power difference—are study-specific definitions rather than source-derived definitions. The study’s simplifications and metric definitions were not fully stated. Independent implementations did not confirm the functions generating truth labels, applying procedures, scoring decisions, or aggregating estimates; therefore, the reported figures and numerical results resting on those functions are not independently confirmed. The final code version also was not independently read after later changes.

<!-- fi:run-record:c0efa448b28a -->
**What FI recorded about this run.** Each trial's random seed is derived from base seed 0 and the trial number; every setting uses the same seeds for the same trial number (common random numbers), so the settings' results are paired, not independent. The experiment ran for 7.4 s; the topic sets a limit of 60 s, and the run was within it. The run had 27 settings with 4,096 trials per setting; 110,592 trials finished and 0 failed. A reviewer objected to the design in ways FI's records neither confirm nor rule out, and the study was not changed for them: "Central conclusions rest on unsupported numerical claims and unverified final-code outputs, while a figure caption misdescribes its figure".
<!-- /fi:run-record -->

<!-- fi:code-findings:c4fddd93a70b -->
**Open points in the code that produced these results.** A second model read the code against the plan. These points were still open on the version that ran. The second model can be wrong, and none of them was settled by the run; the full record is in `.fi/code_review.json`.

- **The final code was not read again.** A second model read the code against the plan, but the code was changed afterwards and the version that produced these results was not read by another model.

- **A second, independent version of a key function was not settled.** FI had these functions written a second time from the plan's description alone and compared the two on many inputs: A second, independently written version of step S1 (Generate truth-labelled z statistics) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S3 (Apply decision procedure) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S4 (Score labelled decisions) could not be compared with the first (the plan gives no example inputs to call it with); A second, independently written version of step S5 (Aggregate uncertainty) could not be compared with the first (the plan gives no example inputs to call it with).
<!-- /fi:code-findings -->

## References

1. Anish Acharya (2014). A Complete Review of Controlling the FDR in a Multiple Comparison Problem Framework -- The Benjamini-Hochberg Algorithm. arXiv (Cornell University). DOI: 10.48550/arxiv.1406.7117
2. Yoav Benjamini, Yosef Hochberg (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society Series B (Statistical Methodology). DOI: 10.1111/j.2517-6161.1995.tb02031.x
3. James X. Hu, Hongyu Zhao, Harrison H. Zhou (2010). False Discovery Rate Control With Groups. Journal of the American Statistical Association. DOI: 10.1198/jasa.2010.tm09329
4. Kyle J. Lesack, Christopher T Naugler (2011). An open-source software program for performing Bonferroni and related corrections for multiple comparisons. Journal of Pathology Informatics. DOI: 10.4103/2153-3539.91130
5. Yoav Benjamini, Daniel Yekutieli (2001). The control of the false discovery rate in multiple testing under dependency. The Annals of Statistics. DOI: 10.1214/aos/1013699998

## Further reading

- [W1] (2021). Multiple Testing, an overview of different decision rules: from Bonferroni to Benjamini-Hochberg. https://soel-micheletti.medium.com/multiple-testing-an-overview-of-different-decision-rules-from-bonferroni-to-benjamini-hochberg-4817904dae29
- [W2] (2025). Controlling your type I errors: Bonferroni and Benjamini-Hochberg. https://www.statsig.com/blog/controlling-type-i-errors-bonferroni-benjamini-hochberg
- [W3] Multiple Testing Corrections. https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/1.5_GenespringMTC.pdf
- [W4] (2026). Bonferroni Correction vs FDR: When to Use Each — CASRAI. https://casrai.org/guides/bonferroni-correction-and-multiple-comparisons
- [W5] Multiple comparisons - Handbook of Biological Statistics. https://www.biostathandbook.com/multiplecomparisons.html
- [W6] (2026). False discovery rate - Wikipedia. https://en.wikipedia.org/wiki/False_discovery_rate
- [W7] (2024). 6.1: Multiple Comparisons - Statistics LibreTexts. https://stats.libretexts.org/Bookshelves/Applied_Statistics/Biological_Statistics_(McDonald)/06:_Multiple_Tests/6.01:_Multiple_Comparisons
