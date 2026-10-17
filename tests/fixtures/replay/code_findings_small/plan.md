# Plan: TOPIC. When many hypotheses are tested at once, how do Bonferroni correction and the Benjamini-Hochberg procedure compare in false discoveries and in 

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This experiment will compare Bonferroni and Benjamini-Hochberg corrections across different hypothesis counts and signal prevalence using Monte Carlo simulations of z-tests. It will quantify the power-vs-error trade-off between FWER and FDR control.

## What the literature says

- **[1]**: Benjamini-Hochberg provides a substantial gain in power compared to FWER control when FDR control is sufficient.
- **[5]**: FWE approach is too stringent for large-scale tests; FDR allows more claims of significant differences given a small fraction of false positives.
- **[10]**: Bonferroni aims to control the FWER, which allows very few occurrences of false positives across the entire family.
- **[32]**: FDR control is equivalent to FWER when all hypotheses are true but allows for more power when some are false.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] A Complete Review of Controlling the FDR in a Multiple Comparison Problem Framework -- The Benjamini-Hochberg Algorithm (DOI 10.48550/arxiv.1406.7117)
- [2] Correcting false discovery rates for their bias toward false positives (DOI 10.1080/03610918.2019.1630432)
- [3] MULTIPLE COMPARISON PROCEDURES FOR HIGH-DIMENSIONAL DATA AND THEIR ROBUSTNESS UNDER NON-NORMALITY (DOI 10.5183/jjscs.1211001_202)
- [4] A Comparative Evaluation of Multiple Hypothesis Testing Adjustment Methods (DOI 10.31222/osf.io/d3svb_v1)
- [5] The False Discovery Rate: A Key Concept in Large-Scale Genetic Studies (DOI 10.1177/107327481001700108)
- [6] Estimation and Use of Correlation in Multiple Hypothesis Testing with High Dimensional Data (DOI 10.1145/3343485.3343489)
- [7] Importance of presenting the variability of the false discovery rate control (DOI 10.1186/s12863-015-0259-z)
- [8] Control of the mean number of false discoveries, Bonferroni and stability of multiple testing (DOI 10.1214/07-aoas102)
- [9] CHOOSING THE LESSER EVIL: TRADE-OFF BETWEEN FALSE DISCOVERY RATE AND NON-DISCOVERY RATE
- [10] MultipleTesting.com: A tool for life science researchers for multiple hypothesis testing correction (DOI 10.1371/journal.pone.0245824)
- [11] False discovery rate controlling procedures for discrete tests (DOI 10.48550/arxiv.1112.4627)
- [12] On the Benjamini–Hochberg method (DOI 10.1214/009053606000000425)
- [13] Multiple testing: correcting for alpha error inflation with false discovery rate (FDR) or family-wise error rate? (DOI 10.1016/j.anbehav.2019.07.001)
- [14] A Novel Weighted Hybrid Method for Multiple Hypothesis Testing of Genomic Data (DOI 10.4038/sljas.v26i2.8201)
- [15] Power and type I error rate of false discovery rate approaches in genome-wide association studies (DOI 10.1186/1471-2156-6-s1-s134)
- [16] Approximate Power and Sample Size Calculations with the Benjamini-Hochberg Method (DOI 10.2202/1557-4679.1018)
- [17] Multiple hypothesis testing in allergy and hypersensitivity diseases investigation: a pedagogical perspective (DOI 10.1016/j.jahd.2024.100014)
- [18] A Bayesian False Discovery Rate for Multiple Testing (DOI 10.1080/02664760600994745)
- [19] Quick calculation for sample size while controlling false discovery rate with application to microarray analysis (DOI 10.1093/bioinformatics/btl664)
- [W1] Multiple Testing, an overview of different decision rules: from Bonferroni to Benjamini-Hochberg
- [20] Effects of dependence in high-dimensional multiple testing problems (DOI 10.1186/1471-2105-9-114)
- [W2] Benjamini–Hochberg Method | Springer Nature Link
- [21] Size, power and false discovery rates (DOI 10.1214/009053606000001460)
- [22] Controlling the number of false discoveries: application to high-dimensional genomic data (DOI 10.1016/s0378-3758(03)00211-8)
- [W3] Multiple hypothesis testing in allergy and hypersensitivity diseases investigation: a pedagogical perspective - ScienceDirect
- [W4] Quantitative Understanding in Biology 1.6 Multiple ...
- [W5] Multiple Testing Corrections
- [24] The positive false discovery rate: a Bayesian interpretation and the q-value (DOI 10.1214/aos/1074290335)
- [W6] False discovery rate - Wikipedia
- [25] On the Adaptive Control of the False Discovery Rate in Multiple Testing With Independent Statistics (DOI 10.3102/10769986025001060)
- [W7] 6.1: Multiple Comparisons - Statistics LibreTexts
- [26] Optimized multi‐stage designs controlling the false discovery or the family‐wise error rate (DOI 10.1002/sim.3300)
- [27] Multiple testing with the structure adaptive Benjamini-Hochberg algorithm (DOI 10.48550/arxiv.1606.07926)
- [28] Empirical null and false discovery rate analysis in neuroimaging (DOI 10.1016/j.neuroimage.2008.04.182)
- [W8] Multiple Comparisons: FWER, FDR, Bonferroni, Holm - Interactive
- [29] The control of the false discovery rate in multiple testing under dependency (DOI 10.1214/aos/1013699998)
- [30] Exact and Approximate Stepdown Methods for Multiple Hypothesis Testing (DOI 10.1198/016214504000000539)
- [31] Augmentation Procedures for Control of the Generalized Family-Wise Error Rate and Tail Probabilities for the Proportion of False Positives (DOI 10.2202/1544-6115.1042)
- [32] Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing (DOI 10.1111/j.2517-6161.1995.tb02031.x)
- [33] Il calcolo delle assicurazioni su gruppi di teste
- [34] IX. On the problem of the most efficient tests of statistical hypotheses (DOI 10.1098/rsta.1933.0009)
- [35] Statistical Inference (DOI 10.1201/9781003456285)
- [36] Testing Statistical Hypotheses (DOI 10.1007/978-1-4757-1923-9)
- [37] A Direct Approach to False Discovery Rates (DOI 10.1111/1467-9868.00346)
- [38] Strong Control, Conservative Point Estimation and Simultaneous Conservative Consistency of False Discovery Rates: A Unified Approach (DOI 10.1111/j.1467-9868.2004.00439.x)
- [39] Resampling-based false discovery rate controlling multiple test procedures for correlated test statistics (DOI 10.1016/s0378-3758(99)00041-5)
- [40] Multiple Hypothesis Testing in Microarray Experiments (DOI 10.1214/ss/1056397487)

## The gap this experiment addresses

While the theoretical behavior is well-documented, the prompt asks for a specific benchmark simulation using m in {20, 200, 2000} and pi1 in {0.0, 0.05, 0.2} to visualize the exact cross-over in power and FDR realization.

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** Simulation of independent one-sided z-tests with known mean shifts and standard normal noise.

**What it assumes:** Test statistics are independent, with standard normal noise; One-sided alternative hypothesis; Exactly mπ1 effects are real in each trial

**Where it holds:** Positive integer m and 0 ≤ π1 ≤ 1 with integer mπ1, under independent standard normal test statistics

**Its equations:**

- **E1** (produces the data): `p = 1 - Phi(z)`. Source: derivation: standard one-sided z-test p-value calculation
- **E2** (analyses the results): `alpha_bonf = alpha / m`. Source: derivation: Bonferroni allocates the family significance level equally across m tests
- **E3** (analyses the results): `pi0 * 0.05`. Source: topic: reference for Benjamini-Hochberg
- **E4** (analyses the results): `the expected share of false positives among the positives found (the false discovery rate)`. Source: topic: definition of false discovery rate

**The checks against known answers, and where each expected value comes from:**

- **bonferroni_fwer_check**, a special or limiting case with a known answer: expects 0.048782; computed as `bonferroni_fwer` from what the simulation returns on `m=100, pi1=0.0, runs_per_setting=10000`; from: derivation: with 100 independent uniform null p-values, P(at least one p ≤ α/100) = 1 - (1 - 0.05/100)**100 => 0.048782
- **power_accuracy_check**, a special or limiting case with a known answer: expects 0.8038; computed as `uncorrected_power` from what the simulation returns on `m=100, pi1=0.2, runs_per_setting=10000`; from: derivation: for a shift of 2.5, power = P(Z > 1.64485 | Z ~ N(2.5, 1)) = 1 - Phi(1.64485 - 2.5) => 0.8038

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **fwer_accuracy**: Does the independent all-null simulation reproduce the exact Bonferroni FWER for m = 100?; how far the check “bonferroni_fwer_check” lands from the value it expects; closest to 0 is best; met within 0.01 of 0; a change smaller than 0.01 counts as no change.
- **power_accuracy**: Does the simulation reproduce the expected uncorrected power for a shift of 2.5?; how far the check “power_accuracy_check” lands from the value it expects; closest to 0 is best; met within 0.01 of 0; a change smaller than 0.01 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/fdr_bonferroni_vs_bh/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 87 more lines of code than two scripts (25 written by the model, 62 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- Realized FDR for BH is significantly lower than Uncorrected but Higher than Bonferroni
- Realized Power for BH is significantly higher than Bonferroni for pi1 > 0
- Bonferroni FWER remains close to 0.05 for pi1 = 0 regardless of m

## Risks

- Low pi1 (0.05) might result in high variance for power estimates if trial count N is too low
- BH FDR is defined as 0 when no discoveries are made, which can bias the mean if discoveries are very rare

## What this quest will not do

- Dependency structures between tests (only independent tests are simulated)
- Alternative corrections like Holm or Storey-BH

## Checks already made

- precision; protocol.runs_per_setting and protocol.precision; The draft specifies 1000 trials and a 0.02 half-width target without showing that 1000 trials can reach it.; A bounded run-level mean can have an approximate worst-case 95% half-width of 0.031 at 1000 trials.; Use 2500 trials per setting, giving an approximate worst-case half-width of 0.0196, and state that this is a planning calculation rather than a guarantee for every observed interval.; high
- estimand; method, protocol.ci_method, and protocol.metrics; The method promises standard errors while the protocol says only 'bootstrap,' and it does not specify how uncertainty in comparisons between methods is calculated.; All three methods use the same simulated trial, so method differences are paired; resampling individual tests would overstate the effective sample size.; Define power as the mean trial-level true-positive fraction, FDR as the mean trial-level false-discovery proportion, and null-case FWER as a trial-level probability; use trial-level bootstrap intervals and paired trial-level bootstrap intervals for differences.; high
- stratum_collapse; figures_planned; The figure names do not require results to remain separate for each m and π1 setting.; Pooling settings would hide changes in power and FDR across the prescribed grid.; Specify that power and FDR figures show each m × π1 setting separately, with intervals and theory references.; medium
- threshold_sensitivity; protocol.acceptance; The criterion 'Realized FDR (BH) <= 0.06' uses an unexplained cutoff and ignores that the theoretical reference is π0 × 0.05, which varies with π1.; Passing that cutoff would not establish agreement with the setting-specific theory.; Remove the pass/fail cutoff and report the BH estimate and interval against π0 × 0.05 for every setting.; medium
- rng_independence; protocol.seed_policy and protocol.metrics; 'One independent stream per setting and run' does not define reproducible stream derivation or state that methods share each trial's data.; Ambiguous seeding can accidentally reuse random draws, while the shared data require paired analysis.; Derive independent setting-and-trial streams from a stated SeedSequence root, apply all methods to the same p-values within each trial, and mark comparisons as paired.; medium
- oracle; protocol.oracles.bonferroni_fwer_check; The stated expected value 0.048889 does not equal the oracle's own formula, 1 − (1 − 0.05/100)^100.; An incorrect reference can make a correct simulation appear wrong.; Set the expected value to 0.048782, specify 10000 null trials in the oracle case, and compare the returned FWER with an absolute tolerance of 0.01.; high
- failed_runs; protocol.failure_policy; 'Excluded and reported' silently changes the estimand if failures depend on simulated outcomes.; The reported means would then describe only successful runs.; Treat any exception or nonfinite result as a failed setting, report its count, and produce no headline estimate for that setting until the cause is fixed and the full predetermined set of trials is rerun.; high
- other; protocol.model.equations.E2.source; The source '[10]' is an unverified citation placeholder.; The requested paper permits citations only to sources actually found and read.; Replace the placeholder with the direct derivation αBonf = α/m.; low
- `protocol.metrics` was left out of the plan because it could not be checked (`protocol.metrics` entry `power`: `given` must name a declared `proportion` metric, not `π1 > 0`); FI asks the plan's model to write it before the experiment runs.
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, protocol, result_assertions, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gemma4:31b).
- The protocol declares no metric spec (`metrics`), so which estimator each headline number gets (independent trials or clusters, paired or independent settings, a proportion or a mean) would be guessed from the names in RESULT_JSON; declare each headline metric while the plan is a draft. The evidence level does not call the statistics adequate while a metric is undeclared.
- The design claims something about how a result changes with a parameter (convergence, scaling, a threshold), but m (3 values), pi1 (3 values) has fewer than five values: a rate, a fit or a critical point cannot be told from so few points. Add values, in particular near where the behaviour changes.
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.
- Equation E1 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E2 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).

- FI asked the plan's model to write `protocol.metrics` again, and it is now in the plan.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gemma4:31b). Its summary: The check can catch a large Bonferroni cutoff error, but its measure does not identify the procedure. Its complete-null case leaves the alternative-effect behavior and the direction of the one-sided test unverified.
  - **bonferroni_fwer_check**: tests the model, would catch a plausible bug, its number is not well defined. The expected value is a well-defined probability, but the measure fwer does not identify which procedure's FWER the simulation must return.; a better check: Select Bonferroni FWER explicitly, then add a mixed-null case whose analytically known power tests the direction of the one-sided p-value..
  - Found (high): bonferroni_fwer_check: The measure does not explicitly select Bonferroni FWER (The simulation compares several procedures, so a generic fwer return value cannot establish which procedure produced the measured probability.). Fix: Use a method-specific returned measure or an explicit method selector.
  - Found (medium): bonferroni_fwer_check: The check can pass if the simulation uses the Benjamini–Hochberg result in place of the Bonferroni result (Under the complete null their FWER probabilities differ by less than the declared absolute tolerance.). Fix: Add method-specific checks in cases where the procedures have clearly separated expected outcomes.
  - Found (high): bonferroni_fwer_check: The sole check cannot detect a reversed one-sided tail or errors in results for real effects (Its complete-null case is symmetric and never exercises a shifted test statistic.). Fix: Add a mixed-null power check against the normal-tail probability implied by E1.
  - Checked, and no problem found: bonferroni_fwer_check: checked that the stated expected formula is the independent-null Bonferroni probability of at least one rejection.; bonferroni_fwer_check: checked that the stated tolerance is reachable with the declared repeat count.; E1: checked the upper-tail p-value against the standard one-sided z-test definition.; E2: checked the Bonferroni per-test cutoff against equal allocation of the family significance level..
  - Equations the checks test: E1, E2; not tested by any: none.
  - The checks' cases were not judged by another model (the reader gave no verdict on every check's case): nobody but the model that wrote them has checked that each case isolates what its measure claims.
- The model's 2 equations were read by gpt-6-sol, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is the standard definition or a correct derivation.

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'bonferroni_fwer_check' changed: measure 'fwer' → 'bonferroni_fwer'; what it checks is worded differently.
- What changed: the check 'power_accuracy_check' was added (expects 0.8038, computed as `uncorrected_power`).

- FI could not compute the expected value of 'power_accuracy_check' itself (its formula cannot be computed: it calls something that is not one of the functions a formula may use); the check is used as the plan wrote it.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: m (20.0, 200.0, 2000.0).
- To vary or compare: pi1 (0.0, 0.05, 0.2).
- To vary or compare: procedure (uncorrected, Bonferroni, Benjamini-Hochberg) - NOT in the plan's settings.
- To report: realised false discovery rate.
- To report: power.
- To report: chance of any false discovery.
- Formula the topic states: `pi0 * 0.05` - NOT in the plan's model.
- Formula the topic states: `the expected share of false positives among the positives found (the false discovery rate)` - NOT in the plan's model.
- To compare: Compare uncorrected, Bonferroni and Benjamini-Hochberg on false discoveries and power across the stated m and pi1 levels.
- Constraint: Use one-sided z-tests in which each real effect shifts the test statistic by 2.5.
- Constraint: Use a nominal 5 percent.
- Constraint: Draw the reference pi0 * 0.05 for Benjamini-Hochberg.
- Constraint: Report uncertainty over repeats.
- Constraint: When pi1 = 0 there are no real effects, so power is undefined; report only the chance of any false discovery.
- Constraint: Provide one Python script using numpy (and scipy where it helps), with matplotlib (Agg backend).
- Constraint: Provide two or three figures that answer the question directly, each with the reference or theory value drawn.
- Constraint: Provide a short paper of at most 4 pages.
- Constraint: Cite only sources you have actually found and read; if a reference cannot be confirmed, say so rather than citing it.
- Constraint: Use CPU only and keep runtime under 60 seconds.
- Constraint: Use fixed random seeds where the run is random, stated in the paper.
- Constraint: Report every number with its uncertainty (for a deterministic result, its numerical error), and say when a result is only a check of the code.

- FI asked the plan's model to add what was missing; it is still missing.

- Still missing (the paper will say it was not studied; the quest stops before anything runs, because the topic's question cannot be answered without it):
  - The topic says to vary or compare procedure (uncorrected, Bonferroni, Benjamini-Hochberg), but the plan's settings do not include it.
  - The topic states the formula `pi0 * 0.05`, but the plan's model does not use it and does not say why it uses another.
  - The topic states the formula `the expected share of false positives among the positives found (the false discovery rate)`, but the plan's model does not use it and does not say why it uses another.

## The design (used as written)

```yaml
hypothesis: Under the specified independent one-sided z-test model, the Benjamini-Hochberg procedure will
  generally have higher power than Bonferroni at α = 0.05 for π1 > 0, while its FDR follows the setting-specific
  reference π0 × α; the size of the power difference will be reported separately for each m and π1.
variables:
  independent:
  - m
  - pi1
  - correction_method
  dependent:
  - power by m and pi1 when pi1 > 0
  - realized_fdr by m and pi1 when pi1 > 0
  - fwer by m when pi1 = 0
  controls:
  - alpha
  - effect_size
method: For each combination of m ∈ {20, 200, 2000} and π1 ∈ {0.0, 0.05, 0.2}, simulate 2500 independent
  trials. In each trial, draw exactly mπ1 statistics from N(2.5, 1) and the remaining statistics from
  N(0, 1), and calculate one-sided p-values with the normal survival function. Apply uncorrected testing
  (p ≤ 0.05), Bonferroni (p ≤ 0.05/m), and Benjamini-Hochberg (the largest k with p(k) ≤ 0.05k/m) to the
  same p-values. When π1 > 0, calculate each trial's power as true positives divided by mπ1 and its false-discovery
  proportion as V/R, defined as 0 when R = 0; report their means over trials. When π1 = 0, report only
  the proportion of trials with V > 0. Report 95% trial-level bootstrap intervals for each estimate and
  paired trial-level bootstrap intervals for method differences. Plot the analytic uncorrected and Bonferroni
  power references, the BH FDR reference π0 × 0.05, and the all-null FWER references.
expected_outcome: Within this independent z-test model, BH power is expected to exceed Bonferroni power
  in settings with real effects, with the size and uncertainty of each difference shown by setting. BH
  FDR is expected to agree with π0 × 0.05 within Monte Carlo uncertainty. In the all-null settings, BH
  FWER is 0.05 in theory, Bonferroni FWER is 1 − (1 − 0.05/m)^m, and uncorrected FWER is 1 − 0.95^m; these
  are theory references and checks of the simulation, not separate empirical discoveries.
figures_planned:
- 'power_comparison.png: for π1 = 0.05 and 0.2, show each method''s power by m with 95% intervals; draw
  the exact uncorrected and Bonferroni power curves as references.'
- 'fdr_comparison.png: for π1 = 0.05 and 0.2, show each method''s mean false-discovery proportion by m
  with 95% intervals; draw a separate π0 × 0.05 reference for each π1.'
- 'fwer_null_case.png: for π1 = 0, show each method''s chance of any false discovery by m with 95% intervals;
  draw the exact all-null FWER references for all three methods.'
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: power
  min: 0
  max: 1
  unit: dimensionless
  reason: Power is a proportion of true effects detected.
- path: realized_fdr
  min: 0
  max: 1
  unit: dimensionless
  reason: FDR is the expectation of the trial-level false-discovery proportion.
- path: fwer
  min: 0
  max: 1
  unit: dimensionless
  reason: FWER is the probability of making at least one type I error.
protocol:
  grid:
    m:
    - 20
    - 200
    - 2000
    pi1:
    - 0.0
    - 0.05
    - 0.2
    correction_method:
    - uncorrected
    - Bonferroni
    - Benjamini-Hochberg
  runs_per_setting: 2500
  thresholds:
    alpha: 0.05
    effect_size: 2.5
  seed_policy: Use numpy SeedSequence(20261008), spawning one independent child stream for each grid setting
    and then one independent child stream for each trial in fixed grid order. Apply all three methods
    to the same p-values within a trial; method comparisons are paired. Use SeedSequence(20261009) for
    the separate oracle case.
  failure_policy: Count and report every exception and nonfinite statistic, p-value, or metric. If any
    occurs, report no headline estimate for that setting; fix the cause and rerun all its predetermined
    trials from the stated streams.
  ci_method: For each setting, resample the 2500 complete trials with replacement 2000 times and use percentile
    95% intervals for the mean trial-level power or false-discovery proportion and for paired method differences.
    For null-case FWER, use a 95% Wilson interval from the count of trials with V > 0. Theory references
    are deterministic formulas, reported with numerical evaluation error rather than stochastic intervals.
  acceptance:
  - For every setting, report the BH estimate and 95% interval beside the setting-specific π0 × 0.05 reference;
    do not use an arbitrary realized-FDR cutoff as a pass/fail conclusion.
  - For the all-null setting, show the observed FWER and 95% interval beside the exact method-specific
    FWER formula.
  precision:
    target_half_width: 0.02
    metric: trial-level power and realized_fdr means
    reason: At 2500 independent trials, the approximate worst-case 95% half-width for a bounded trial-level
      mean is 1.96 × 0.5/√2500 = 0.0196; observed intervals may differ, so their actual widths will be
      reported. The same trial count gives an approximate 0.0085 half-width for an FWER near 0.05.
  oracles:
  - name: bonferroni_fwer_check
    kind: special_case
    check: bonferroni_fwer
    expected: 0.048782
    expected_formula: 1 - (1 - 0.05/m)**m
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: with 100 independent uniform null p-values, P(at least one p ≤ α/100) = 1 - (1 - 0.05/100)**100 => 0.048782'
    case:
      m: 100
      pi1: 0.0
      runs_per_setting: 10000
    measure: bonferroni_fwer
  - name: power_accuracy_check
    kind: special_case
    check: power
    expected: 0.8038
    expected_formula: 0.8038
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: for a shift of 2.5, power = P(Z > 1.64485 | Z ~ N(2.5, 1)) = 1 - Phi(1.64485 - 2.5) => 0.8038'
    case:
      m: 100
      pi1: 0.2
      runs_per_setting: 10000
    measure: uncorrected_power
  criteria:
  - name: fwer_accuracy
    what: Does the independent all-null simulation reproduce the exact Bonferroni FWER for m = 100?
    oracle: bonferroni_fwer_check
    use: error
    direction: target
    target: 0
    tolerance: 0.01
  - name: power_accuracy
    what: Does the simulation reproduce the expected uncorrected power for a shift of 2.5?
    oracle: power_accuracy_check
    use: error
    direction: target
    target: 0
    tolerance: 0.01
  metrics:
  - id: power
    kind: mean
    estimand: mean trial-level power (true positives / mπ1) for π1 > 0
    unit: one trial
  - id: realized_fdr
    kind: mean
    estimand: mean trial-level false-discovery proportion (V/R) for π1 > 0
    unit: one trial
  - id: fwer
    kind: proportion
    estimand: proportion of trials with at least one false discovery when π1 = 0
    unit: one trial
  topic_map:
  - item: m
    by: setting:m
  - item: pi1
    by: setting:pi1
  - item: procedure
    by: setting:correction_method
  - item: realised false discovery rate
    by: metric:realized_fdr
  - item: power
    by: metric:power
  - item: chance of any false discovery
    by: metric:fwer
  - item: pi0 * 0.05
    by: equation:E3
  - item: the expected share of false positives among the positives found (the false discovery rate)
    by: equation:E4
  model:
    summary: Simulation of independent one-sided z-tests with known mean shifts and standard normal noise.
    assumptions:
    - Test statistics are independent, with standard normal noise
    - One-sided alternative hypothesis
    - Exactly mπ1 effects are real in each trial
    holds_for: Positive integer m and 0 ≤ π1 ≤ 1 with integer mπ1, under independent standard normal test
      statistics
    equations:
    - id: E1
      formula: p = 1 - Phi(z)
      role: generates
      source: 'derivation: standard one-sided z-test p-value calculation'
      example:
        inputs:
          z: 1.64485
        expected_formula: '0.05'
    - id: E2
      formula: alpha_bonf = alpha / m
      role: analyses
      source: 'derivation: Bonferroni allocates the family significance level equally across m tests'
      example:
        inputs:
          alpha: 0.05
          m: 100
        expected_formula: '0.0005'
    - id: E3
      formula: pi0 * 0.05
      role: analyses
      source: 'topic: reference for Benjamini-Hochberg'
    - id: E4
      formula: the expected share of false positives among the positives found (the false discovery rate)
      role: analyses
      source: 'topic: definition of false discovery rate'
```
