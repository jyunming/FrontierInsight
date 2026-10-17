# Plan: TOPIC. How accurately can a reaction rate constant and activation energy be recovered from noisy concentration measurements?

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

(not written)

## What the literature says

- (the model named no source that bears on this topic)

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] An exhaustive search approach for chemical kinetics experimental data fitting, rate constants optimization and confidence interval estimation (DOI 10.15388/na.2015.1.10)
- [2] Uncertainty quantification of reaction mechanisms accounting for correlations introduced by rate rules and fitted Arrhenius parameters (DOI 10.1016/j.combustflame.2013.01.008)
- [3] Monte Carlo Simulations for the Analysis of Non-linear Parameter Confidence Intervals in Optimal Experimental Design (DOI 10.3389/fbioe.2019.00122)
- [4] Stochastic Simulation And Parameter Estimation Of First Order Chemical Reactions (DOI 10.5281/zenodo.38668)
- [5] Statistical Prediction of Drug Stability Based on Nonlinear Parameter Estimation (DOI 10.1002/jps.2600730517)
- [W1] Measuring Activation Energy with Arrhenius Equation Adjustments
- [W2] Two Types of Uncertainty in the Values of Activation Energy | Request PDF
- [W3] Uncertainty in the Arrhenius parameters resulting from selecting... | Download Scientific Diagram
- [W4] (PDF) Estimating the parameters of the arrhenius equation
- [W5] A First Course on Kinetics and Reaction Engineering Example 4.6 Problem Purpose
- [W6] Bayesian Methods for the Investigation of Temperature-Dependence in Conductivity
- [W7] Arrhenius Relationship - ReliaWiki
- [W8] Why you might be fitting Arrhenius data badly | scattering and stats
- [W9] Sensitivity and uncertainty analysis of Arrhenius parameters in order to describe the kinetic of solid thermal degradation during fire phenomena - ScienceDirect
- [W10] J Math Chem (2011) 49:1798–1809 DOI 10.1007/s10910-011-9859-7 ORIGINAL PAPER
- [6] An Algorithm for Least-Squares Estimation of Nonlinear Parameters (DOI 10.1137/0111030)
- [7] A method for the solution of certain non-linear problems in least squares (DOI 10.1090/qam/10666)
- [8] Nonlinear Regression (DOI 10.1002/0471725315)
- [9] Nonlinear Regression Analysis and Its Applications (DOI 10.1002/9780470316757)
- [10] Data Reduction and Error Analysis for the Physical Sciences (DOI 10.1063/1.4823194)
- [11] SciPy 1.0: fundamental algorithms for scientific computing in Python (DOI 10.1038/s41592-019-0686-2)

## The gap this experiment addresses

(not written)

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** First-order decay c=c0 exp(-kt) with Arrhenius k(T), additive Gaussian noise, NLS fit of k per temperature, then weighted linear regression of ln k on centred 1/T.

**What it assumes:** c0 known; iid Gaussian noise with constant SD; exact Arrhenius law; T known exactly

**Where it holds:** sigma from 0.005 to 0.1 of c0, 5 to 40 points per temperature, k t_max=3.

**Its equations:**

- **E1** (produces the data): `c(t) = c0 * exp(-k * t)`. Source: derivation: Integrating dc/dt = -k c with c(0)=c0 gives c = c0 exp(-kt).
- **E2** (produces the data): `k = A * exp(-Ea / (R * T))`. Source: derivation: Arrhenius law as given in the topic statement.
- **E3** (analyses the results): `ln k = ln A - Ea / (R * T)`. Source: derivation: Take the natural log of E2; regressing ln k on 1/T has slope -Ea/R and intercept ln A.

**The checks against known answers, and where each expected value comes from:**

- **noise_free_Ea**, a special or limiting case with a known answer: expects 50000.0; computed as `Ea_hat` from what the simulation returns on `sigma=0.0, n_points=12`; from: derivation: with sigma=0, c=exp(-k t) exactly, NLS returns k to solver tolerance, ln k = lnA - Ea/(R T) is exactly linear in 1/T, so the slope is -Ea/R and Ea_hat=50000 J/mol.
- **noise_free_lnA**, a special or limiting case with a known answer: expects 16.0; computed as `lnA_hat` from what the simulation returns on `sigma=0.0, n_points=12`; from: derivation: the intercept at the centred 1/T is lnA-(Ea/R)*xbar, and adding back the slope times xbar returns lnA=16.
- **low_noise_SD_over_theory_SE**, a special or limiting case with a known answer: expects 1.0; computed as `Ea_sd / Ea_se_theory` from what the simulation returns on `sigma=0.005, n_points=40`; from: derivation: for small sigma the NLS estimator is asymptotically normal with the Fisher variance, and the weighted regression is linear in ln k.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **noiseless_Ea_error**: The noise-free pipeline recovers Ea.; how far the check “noise_free_Ea” lands from the value it expects; lower is better; met at 0.05 or below; a change smaller than 0.05 counts as no change.
- **noiseless_lnA_error**: The noise-free pipeline recovers ln A.; how far the check “noise_free_lnA” lands from the value it expects; lower is better; met at 1e-06 or below; a change smaller than 1e-06 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/kinetics_parameter_recovery/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 101 more lines of code than two scripts (30 written by the model, 71 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- (not written)

## Risks

- (not written)

## What this quest will not do

- (not written)

## Checks already made

- failed_runs; protocol.failure_policy; Failures are only 'curve_fit raises or returns non-finite k or variance', but at sigma=0.1, n=5 a fitted k_hat <= 0 is finite and makes ln k undefined (or a huge outlier) in stage two.; Dropping or mishandling those runs silently conditions coverage and bias on 'fit succeeded', which changes the estimand exactly in the high-noise cells where the claim is made.; Treat any temperature with k_hat<=0 or non-finite var as a failed run, count it per cell, report the failure fraction beside coverage, and state that coverage is conditional on success (also report coverage with failures counted as misses as a bound).; high
- oracle; protocol.oracles; Both oracles are noise-free and use unit weights because the NLS variance is zero, so they exercise a different weighting branch from the real pipeline and say nothing about the SE, interval, or coverage code.; A mistake in the weight handling, covariance propagation, or t-quantile code would pass both checks and bias every headline coverage number.; Add a limiting-case oracle at sigma=0.005, n=40 that the observed SD of Ea_hat equals the linearised theoretical SE (ratio 1, tolerance 0.1), and say in the paper that the noise-free oracles are code checks only.; medium
- estimand; protocol.metrics / Ea_mean, Ea_sd; Bias and SD of Ea_hat are summarised only by mean and SD, although ln k from a fit with k_hat near 0 gives a heavy-tailed Ea_hat that the design itself expects.; Mean and SD of a heavy-tailed estimator are dominated by a few runs and their bootstrap intervals can be misleading.; Add median bias and IQR-based spread as secondary summaries alongside mean/SD, and report the number of runs beyond 5 SE.; medium
- precision; protocol.acceptance; Acceptance requires coverage inside the Wilson interval of 0.95 in every cell with sigma<=0.02 and n>=12 (9 cells x 2 parameters) with no allowance for multiplicity.; With ~18 individual 95% checks, about one is expected to fail by chance, so the criterion can reject a correct pipeline.; Require at least 90% of those cell-parameter checks to pass and no pass-rate trend with sigma or n, and report any single failure.; medium
- numerical_convergence; method (curve_fit); The curve_fit start 1/(mean t) is built from the true k via the sampling grid, and no tolerance or convergence check is stated.; A start near the truth hides convergence failures that a real analyst would meet, and the solver tolerance sets the floor for the oracle tolerances.; State the curve_fit ftol/xtol (1e-10), report non-convergence in the failure count, and state that the start is the grid-implied k, which is available to the analyst.; low
- other; method / runtime; 25 cells x 400 runs x 5 curve_fit calls = 50,000 fits plus bootstraps has no timing plan against the 60 s limit.; Overrunning the limit would force an undeclared change of run count.; Time one cell first, choose runs_per_setting from the measured speed (400, minimum 200), use a 1000-resample bootstrap only on the final summaries, and report the runs actually used per cell.; low
- stratum_collapse; dependent variables / figures_planned; 'bias of k_hat at each temperature' is a dependent variable but no figure or table is planned for it.; The analyse step would have nothing to stratify by temperature, and the source of any bias in stage one would be invisible.; Add a per-temperature k_hat bias table written to the results output and reference it in the paper.; low
- `protocol.in_each_run` was left out of the plan because it could not be checked (`protocol.in_each_run.temperature_1_over_T` has one value: a single value is not compared; it is a fixed setting of the run); it is not used
- `criteria` was written beside `protocol` in the draft and FI moved it under `protocol`, where the plan is read
- The methodology audit amended the design: definitions, method, protocol, simplifications, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read again by the model that wrote it: a self-review, not an independent one.
- Equation E2 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E3 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- The check 'low_noise_SD_over_theory_SE' names a derivation for its expected value, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

## The design (used as written)

```yaml
hypothesis: With known A and Ea, the two-stage estimator (per-temperature nonlinear least squares for
  k, then weighted regression of ln k on centred 1/T) recovers Ea and ln A with small bias and 95% interval
  coverage near 0.95 once noise is small and there are enough time points. Coverage falls and bias grows
  at high noise and few points, where the linearised error propagation fails.
variables:
  independent:
  - sigma (additive Gaussian noise SD, in units of c0)
  - n_points (time points per temperature)
  dependent:
  - bias of Ea_hat and lnA_hat (mean and median)
  - SD and IQR-based spread of Ea_hat and lnA_hat across runs
  - ratio of observed SD to linearised theoretical SE
  - coverage of the 95% interval for Ea and ln A (conditional on a successful fit, with failures-as-misses
    bound)
  - correlation of Ea_hat with lnA_hat
  - bias of k_hat at each temperature (tabulated)
  - failure fraction per cell
  controls:
  - Ea=50 kJ/mol
  - ln A=16
  - c0=1 (known)
  - five temperatures equally spaced in 1/T giving exactly a 10-fold k range
  - sampling times linspace(0,3/k,n) at each T
  - R=8.314462618 J/(mol K)
  - weights = 1/var(ln k) from the NLS covariance
  - 1/T centred at its mean
  - fixed seeds
method: 'Truth: ln A=16, Ea=50000 J/mol, R=8.314462618. Take five 1/T values equally spaced from 1/320
  K^-1 to 1/320+ln(10)*R/Ea K^-1, so k spans exactly 10x. At each T, sample t=linspace(0,3/k,n_points)
  and simulate c=exp(-k t)+N(0,sigma^2) (c0=1 known). Fit k with scipy.optimize.curve_fit (start from
  1/(mean t), the grid-implied k available to an analyst; ftol=xtol=1e-10; absolute_sigma=False), giving
  var(ln k)=var(k)/k^2. A run fails if curve_fit raises, does not converge, or returns k_hat<=0 or a non-finite
  k or variance at any temperature; failures are counted per cell. Stage two is weighted least squares
  of ln k on x=1/T-mean(1/T) with weights 1/var(ln k). Then Ea=-R*slope and lnA=intercept-slope*xbar,
  with variances propagated from the 2x2 weighted covariance (including cov(slope,intercept)) and the
  Ea-lnA covariance. Two 95% intervals are computed. The primary one scales the covariance by the residual
  reduced chi-square and uses t quantiles with 3 dof. The secondary one uses the nominal weights and normal
  quantiles. Linearised theory baseline: var(k)=sigma^2/sum(t_i^2 exp(-2 k t_i)), propagated through the
  same weighted regression to theoretical SE(Ea), SE(lnA) and corr. Grid is 5 sigma x 5 n_points, with
  400 runs per cell (reduced toward 200 after timing one cell against the 60 s limit; runs actually used
  are reported). Report mean and median bias, SD, IQR-based spread, RMSE, coverage with a Wilson binomial
  interval, and the Ea-lnA correlation, each with its Monte Carlo uncertainty (bootstrap or standard error),
  plus a per-temperature k_hat bias table. Figures: (1) mean±SD of Ea_hat and lnA_hat vs sigma per n_points,
  with the true value and the theory SE band, plus SD/SE_theory; (2) coverage of Ea and ln A vs sigma
  per n_points with the 0.95 line and Monte Carlo band; (3) scatter of Ea_hat vs lnA_hat at one setting
  with the truth and the theoretical 95% covariance ellipse. Uses scipy (curve_fit, t quantiles) and matplotlib
  Agg. The uncertainty-and-units skill is not used: the error propagation is written out in closed form
  and its checks are the oracles. The noise-free oracles only check the code, not the statistics; the
  SD/SE limiting-case oracle checks the theory against simulation at low noise.'
expected_outcome: At sigma<=0.02 and n>=12 the observed SD should match the linearised SE within a few
  percent and coverage should lie within Monte Carlo error of 0.95. SD should scale roughly linearly with
  sigma and as 1/sqrt(n). At sigma=0.1 with n=5 I expect an SD/SE ratio above 1, a small bias in k_hat
  from the nonlinear fit, a heavy tail in Ea_hat, and coverage below 0.95 for the normal-quantile interval
  (the t-interval is closer). Ea_hat and lnA_hat should be correlated at r close to +1 (>0.99) because
  the intercept is extrapolated from the centred 1/T to 1/T=0.
figures_planned:
- fig1_bias_spread.png
- fig2_coverage.png
- fig3_Ea_lnA_scatter.png
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: coverage_Ea
  min: 0.0
  max: 1.0
  unit: dimensionless
  reason: A proportion of runs lies in [0,1].
- path: corr_Ea_lnA
  min: 0.9
  max: 1.0
  unit: dimensionless
  reason: A correlation is at most 1. With the 1/T centre about 3.3e-3 K^-1 from zero and a 3.8e-4 spread,
    Ea and lnA are almost collinear and positively correlated.
- path: Ea_mean
  min: 35000.0
  max: 65000.0
  unit: J/mol
  reason: Worst-case SE is about 8% of 50 kJ/mol at sigma=0.1, n=5, so the cell mean must stay within
    about 30% of the truth if the pipeline is right.
- path: lnA_mean
  min: 12.0
  max: 20.0
  unit: dimensionless
  reason: lnA error is Ea error times xbar/R, about 0.4 per kJ/mol; a mean outside 12 to 20 would indicate
    a unit or centring error.
- path: Ea_sd
  min: 1.0
  max: 20000.0
  unit: J/mol
  reason: Noise is nonzero in every grid cell, so the SD is strictly positive, and it cannot exceed 40%
    of Ea.
simplifications:
- id: X1
  replaces: fit k with c0 unspecified (c0 may be free)
  because: c0 is known and fixed to 1 so each temperature fit has one parameter
  effect: Uncertainty in k is smaller than with a free c0, and the small-sample nonlinear bias of the
    NLS fit is reduced.
  probe: Optionally rerun one cell with c0 free and compare the SD of Ea_hat.
- id: X2
  replaces: a standard definition of bias, spread and coverage cited from a source
  because: none of the retrieved sources states these definitions word for word
  effect: None on numbers; definitions are the usual ones (bias=mean(hat)-true, spread=SD of hat, coverage=fraction
    of intervals containing truth).
  probe: The definitions are stated explicitly in the paper.
- id: X3
  replaces: linearised (Fisher) error propagation as the sole theory
  because: the theory ignores the stage-two weights being estimated from stage-one data, which is the
    source of any small-sample coverage failure
  effect: The theory SE may understate the observed SD at high noise.
  probe: Compare SD/SE_theory across the grid.
- id: X4
  replaces: the exact temperatures
  because: the topic fixes only a 10-fold range of k, not the temperatures; they are chosen equally spaced
    in 1/T starting at 320 K (about 285 K at the other end)
  effect: A wider or narrower 1/T range would scale the Ea SE inversely with the range.
  probe: State the temperatures and the 1/T range in the paper.
- id: X5
  replaces: all simulated runs entering the statistics
  because: runs with a failed fit or k_hat<=0 at any temperature cannot be log-transformed and are excluded
  effect: Coverage and bias at high noise and few points are conditional on success and may look better
    than the unconditional estimator.
  probe: Report the failure fraction per cell and the coverage bound counting failures as misses.
definitions:
- metric: coverage_Ea
  defined_as: fraction of successful runs whose 95% interval for Ea contains the true Ea
  source: ''
  quote: ''
  differs: X2
- metric: coverage_lnA
  defined_as: fraction of successful runs whose 95% interval for lnA contains the true lnA
  source: ''
  quote: ''
  differs: X2
- metric: Ea_mean
  defined_as: mean of Ea_hat over runs; bias = Ea_mean - Ea_true
  source: ''
  quote: ''
  differs: X2
- metric: lnA_mean
  defined_as: mean of lnA_hat over runs; bias = lnA_mean - lnA_true
  source: ''
  quote: ''
  differs: X2
- metric: Ea_sd
  defined_as: sample standard deviation of Ea_hat over runs (ddof=1)
  source: ''
  quote: ''
  differs: X2
- metric: corr_Ea_lnA
  defined_as: Pearson correlation of Ea_hat and lnA_hat over runs
  source: ''
  quote: ''
  differs: X2
protocol:
  grid:
    sigma:
    - 0.005
    - 0.01
    - 0.02
    - 0.05
    - 0.1
    n_points:
    - 5
    - 8
    - 12
    - 20
    - 40
  runs_per_setting: 400
  min_runs_per_setting: 200
  thresholds:
    confidence_level: 0.95
    reference_coverage: 0.95
    Ea_true_J_per_mol: 50000
    lnA_true: 16
    sampling_t_max_in_units_of_1_over_k: 3
  seed_policy: One independent stream per setting and run, from numpy SeedSequence([20240601, setting_index,
    run_index]); the base seed is stated in the paper. Settings are not paired.
  failure_policy: A trial where curve_fit raises, does not converge, or returns k_hat<=0 or non-finite
    k or variance at any temperature is excluded from the statistics, counted per setting, and listed
    in the output; the failure fraction is reported, coverage is stated as conditional on success, and
    a bound counting failures as misses is also reported.
  ci_method: Wilson binomial interval for coverage; bootstrap (1000 resamples) or analytic standard error
    for bias, SD and correlation (Fisher z).
  acceptance:
  - Coverage of the t interval for Ea and ln A lies within the 95% Wilson interval of 0.95 in at least
    90% of the cell-parameter checks with sigma<=0.02 and n_points>=12, with no trend in pass rate across
    sigma or n_points; any single failure is reported.
  - Observed SD/theoretical SE is within 1±0.1 in those cells.
  - Bias of Ea_hat is below 2 SE_MC (or below 10% of the SD) in those cells.
  - Noise-free oracles recover Ea and lnA to the stated tolerance.
  - Low-noise SD/SE limiting-case oracle is within 0.1 of 1.
  precision:
    target_half_width: 0.03
    metric: coverage_Ea
    reason: Distinguishing coverage 0.95 from the 0.90 typical of the failing regime needs a half-width
      of about 0.03 or less; 400 runs give about 0.021.
  metrics:
  - id: coverage_Ea
    estimand: P(true Ea in 95% interval | sigma, n_points, fit succeeded)
    kind: proportion
    unit: a run
    cluster: null
    paired: false
    family: coverage across the sigma x n_points grid
  - id: coverage_lnA
    estimand: P(true lnA in 95% interval | sigma, n_points, fit succeeded)
    kind: proportion
    unit: a run
    cluster: null
    paired: false
    family: coverage across the sigma x n_points grid
  - id: Ea_mean
    estimand: E[Ea_hat], with bias = Ea_mean - Ea_true (median bias reported as a robust companion)
    kind: mean
    unit: a run
    cluster: null
    paired: false
    family: bias across the grid
  - id: lnA_mean
    estimand: E[lnA_hat], with bias = lnA_mean - lnA_true
    kind: mean
    unit: a run
    cluster: null
    paired: false
    family: bias across the grid
  - id: Ea_sd
    estimand: SD of Ea_hat over runs, reported with its bootstrap uncertainty (IQR-based spread as a robust
      companion)
    kind: mean
    unit: a run
    cluster: null
    paired: false
    family: spread across the grid
  - id: corr_Ea_lnA
    estimand: Pearson correlation of Ea_hat and lnA_hat over runs
    kind: mean
    unit: a run
    cluster: null
    paired: false
    family: correlation across the grid
  oracles:
  - name: noise_free_Ea
    kind: special_case
    check: With sigma=0 the pipeline must recover Ea exactly (unit weights are used because the NLS variance
      is zero).
    expected: 50000.0
    expected_formula: '50000'
    tolerance: 0.05
    tolerance_mode: absolute
    reference: 'derivation: with sigma=0, c=exp(-k t) exactly, NLS returns k to solver tolerance, ln k
      = lnA - Ea/(R T) is exactly linear in 1/T, so the slope is -Ea/R and Ea_hat=50000 J/mol.'
    case:
      sigma: 0.0
      n_points: 12
    measure: Ea_hat
  - name: noise_free_lnA
    kind: special_case
    check: With sigma=0 the pipeline must recover ln A exactly, which checks the centring back-transformation.
    expected: 16.0
    expected_formula: '16'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: 'derivation: the intercept at the centred 1/T is lnA-(Ea/R)*xbar, and adding back the slope
      times xbar returns lnA=16.'
    case:
      sigma: 0.0
      n_points: 12
    measure: lnA_hat
  - name: low_noise_SD_over_theory_SE
    kind: limiting_case
    check: At very low noise and 40 points the linearisation is accurate, so the observed SD of Ea_hat
      over runs equals the closed-form linearised SE (ratio 1); this exercises the real weighted pipeline
      and the theory code.
    expected: 1.0
    expected_formula: '1'
    tolerance: 0.1
    tolerance_mode: absolute
    reference: 'derivation: for small sigma the NLS estimator is asymptotically normal with the Fisher
      variance, and the weighted regression is linear in ln k.'
    case:
      sigma: 0.005
      n_points: 40
    measure: Ea_sd / Ea_se_theory
  criteria:
  - name: noiseless_Ea_error
    what: The noise-free pipeline recovers Ea.
    oracle: noise_free_Ea
    use: error
    direction: lower
    target: 0.05
    tolerance: 0.05
  - name: noiseless_lnA_error
    what: The noise-free pipeline recovers ln A.
    oracle: noise_free_lnA
    use: error
    direction: lower
    target: 1.0e-06
    tolerance: 1.0e-06
  model:
    summary: First-order decay c=c0 exp(-kt) with Arrhenius k(T), additive Gaussian noise, NLS fit of
      k per temperature, then weighted linear regression of ln k on centred 1/T.
    assumptions:
    - c0 known
    - iid Gaussian noise with constant SD
    - exact Arrhenius law
    - T known exactly
    holds_for: sigma from 0.005 to 0.1 of c0, 5 to 40 points per temperature, k t_max=3.
    equations:
    - id: E1
      formula: c(t) = c0 * exp(-k * t)
      role: generates
      source: derivation
      derivation: Integrating dc/dt = -k c with c(0)=c0 gives c = c0 exp(-kt).
      example:
        inputs:
          c0: 2.0
          k: 0.3
          t: 4.0
        expected_formula: 2.0*exp(-0.3*4.0)
    - id: E2
      formula: k = A * exp(-Ea / (R * T))
      role: generates
      source: derivation
      derivation: Arrhenius law as given in the topic statement.
      example:
        inputs:
          A: 10000000.0
          Ea: 50000.0
          R: 8.314462618
          T: 300.0
        expected_formula: 10000000.0*exp(-50000.0/(8.314462618*300.0))
    - id: E3
      formula: ln k = ln A - Ea / (R * T)
      role: analyses
      source: derivation
      derivation: Take the natural log of E2; regressing ln k on 1/T has slope -Ea/R and intercept ln
        A.
      example:
        inputs:
          lnA: 16.0
          Ea: 50000.0
          R: 8.314462618
          T: 300.0
        expected_formula: 16.0-50000.0/(8.314462618*300.0)
    steps:
    - id: S1
      name: NLS fit of k
      does: Fits k to one temperature's noisy c(t) with c0 fixed, returning k_hat and var(k_hat).
      inputs:
        t: s
        c: dimensionless
      output: k_hat (1/s), var_k (1/s^2)
      uses:
      - E1
      source: '[11] scipy.optimize.curve_fit; Levenberg-Marquardt [6][7]'
      relations:
      - noise-free data returns k_hat equal to k to solver tolerance
    - id: S2
      name: Weighted Arrhenius regression
      does: Weighted least squares of ln k_hat on 1/T minus its mean, weights 1/var(ln k), then back-transforms
        to Ea and lnA with covariance.
      inputs:
        k_hat: 1/s
        var_ln_k: dimensionless
        T: K
      output: Ea_hat (J/mol), lnA_hat, 2x2 covariance
      uses:
      - E3
      source: derivation (standard weighted least squares)
      relations:
      - the Ea-lnA covariance is positive because the centre of 1/T is positive
    - id: S3
      name: 95% interval
      does: Forms Ea_hat ± q SE with q the t(3 dof) quantile (SE scaled by residual reduced chi-square)
        or the normal quantile (nominal weights), and checks whether the truth is inside.
      inputs:
        Ea_hat: J/mol
        SE: J/mol
      output: inside (bool)
      uses: []
      source: derivation
```
