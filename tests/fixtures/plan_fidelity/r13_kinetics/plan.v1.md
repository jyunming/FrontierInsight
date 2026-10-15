# Plan: TOPIC. How accurately can a reaction rate constant and activation energy be recovered from noisy concentration measurements?

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This Monte Carlo study measures recovery of k, Ea, and ln A under the required concentration-noise and sampling grids. It compares propagated-variance weighted Arrhenius regression with an unweighted baseline, using bias, spread, interval coverage, and parameter correlation.

## What the literature says

- **[1] On the errors of Arrhenius parameters and estimated rate constant values**: Simulated Arrhenius studies show close parameter correlation and that incorrect weighting can shift fitted parameters and increase variance.
- **[2] Further Consideration of Heteroscedasticity in Fitting Kinetic Models**: Linearisation can induce heteroscedasticity and nonlinear kinetic fitting needs appropriate precision assessment.
- **[8] Improving uncertainty analysis in kinetic evaluations using iteratively reweighted least squares**: Simulated kinetic studies found uncertainty treatment materially changes confidence intervals.
- **[10] Separable nonlinear least squares: the variable projection method and its applications**: Exponential fitting is a nonlinear least-squares problem whose conditioning matters.
- **[19] Nonlinear Regression Analysis and Its Applications**: Nonlinear regression provides the relevant fitting and uncertainty framework.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] On the errors of Arrhenius parameters and estimated rate constant values (DOI 10.1002/kin.550190302)
- [2] Further Consideration of Heteroscedasticity in Fitting Kinetic Models (DOI 10.1080/00401706.1977.10489544)
- [3] PARAMETER ESTIMATION VIA BAYESIAN INVERSION: THEORY, METHODS, AND APPLICATIONS
- [4] KINFIT II: a nonlinear least-squares program for analysis of kinetic binding data. (DOI 10.1016/s0026-895x(25)09169-2)
- [5] Probabilistic inference of reaction rate parameters from summary statistics (DOI 10.1080/13647830.2017.1370557)
- [6] Uncertainty quantification of reaction mechanisms accounting for correlations introduced by rate rules and fitted Arrhenius parameters (DOI 10.1016/j.combustflame.2013.01.008)
- [7] Equivalency of Kinetic Schemes: Causes and an Analysis of Some Model Fitting Algorithms (DOI 10.1002/kin.21092)
- [8] Improving uncertainty analysis in kinetic evaluations using iteratively reweighted least squares (DOI 10.1002/etc.630)
- [9] Sensitivity and uncertainty analysis of Arrhenius parameters in order to describe the kinetic of solid thermal degradation during fire phenomena (DOI 10.1016/j.firesaf.2016.03.007)
- [10] Separable nonlinear least squares: the variable projection method and its applications (DOI 10.1088/0266-5611/19/2/201)
- [11] Optimal designs for the Arrhenius equation (DOI 10.1016/j.chemolab.2004.06.007)
- [12] Statistical Prediction of Drug Stability Based on Nonlinear Parameter Estimation (DOI 10.1002/jps.2600730517)
- [13] Determination of Kinetic Parameter in a Unified Kinetic Model for the Photodegradation of Phenol by Using Nonlinear Regression and the Genetic Algorithm (DOI 10.1515/ijcre-2012-0003)
- [W1] (PDF) Numerical investigation of the uncertainty of Arrhenius parameters
- [14] A global strategy for nonlinear least squares (DOI 10.1016/0025-5564(70)90037-4)
- [15] Bayesian Inference and Optimization for Design of Experiments in Chemical Kinetics (DOI 10.26434/chemrxiv.15000947/v1)
- [W2] Uncertainty of Arrhenius Parameters | Request PDF
- [W3] (PDF) Estimating the parameters of the arrhenius equation
- [W4] Uncertainty of Arrhenius Parameters TIBOR NAGY, TAM ´AS TUR ´ANYI
- [W5] Sensitivity and uncertainty analysis of Arrhenius parameters in order to describe the kinetic of solid thermal degradation during fire phenomena - ScienceDirect
- [16] Nonlinear estimation of Monod growth kinetic parameters from a single substrate depletion curve (DOI 10.1128/aem.45.5.1453-1458.1983)
- [W6] (PDF) Uncertainty of Arrhenius parameters
- [W7] Uncertainty of Arrhenius parameters - Nagy - 2011 - International Journal of Chemical Kinetics - Wiley Online Library
- [W8] (PDF) Determination of the uncertainty domain of the Arrhenius parameters needed for the investigation of combustion kinetic models
- [17] Modeling And Simulation Of Iso-Octane Production Kinetic By Using Nonlinear Least Squares Method. (DOI 10.5281/zenodo.1491079)
- [W9] Determination of the uncertainty domain of the Arrhenius parameters needed for the investigation of combustion kinetic models - ScienceDirect
- [W10] Estimating Arrhenius parameters using temperature programmed molecular dynamics | The Journal of Chemical Physics | AIP Publishing
- [18] Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren (DOI 10.1515/zpch-1889-0416)
- [19] Nonlinear Regression Analysis and Its Applications (DOI 10.1002/9780470316757)
- [20] Data Reduction and Error Analysis for the Physical Sciences (DOI 10.1063/1.4823194)
- [21] Applied Regression Analysis (DOI 10.1198/tech.2005.s303)
- [22] Nonlinear Parameter Estimation (DOI 10.2307/1268363)

## The gap this experiment addresses

The cited work motivates weighting and correlation but does not provide this fixed, reproducible calibration map for additive concentration noise, five temperatures spanning tenfold k, and the specified two-stage estimator.

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** Independent first-order concentration series are generated exactly, perturbed by Gaussian concentration noise, fitted by nonlinear least squares, and transformed to a centred weighted Arrhenius regression.

**What it assumes:** c0 is known exactly.; Temperature is measured without error.; The fitted k remains positive.; Covariance linearisation is adequate for reporting nominal intervals.

**Where it holds:** Positive rates, temperatures, and sampling times; the stated study uses noise_sigma_over_c0 from 0.0025 to 0.04, Ea=50 kJ/mol, lnA=18.42 and k from 0.1 to 1 s^-1.

**Its equations:**

- **E1** (produces the data): `c(t) = c0 * exp(-k * t)`. Source: derivation: For dc/dt=-k c, separate dc/c=-k dt; integrate ln(c/c0)=-kt; exponentiate.
- **E2** (produces the data): `k(T) = A * exp(-Ea / (R * T))`. Source: [1]
- **E3** (analyses the results): `Ea = -R*b; lnA = alpha - b*xbar, where ln(k)=alpha+b*(1/T-xbar)`. Source: derivation: Expand alpha+b(1/T-xbar)=lnA-Ea/(RT); match the coefficient b=-Ea/R and intercept alpha=lnA+b*xbar.

**The checks against known answers, and where each expected value comes from:**

- **noiseless_decay_recovery**, a special or limiting case with a known answer: expects 0; computed as `max_k_relative_error` from what the simulation returns on `noise_sigma_over_c0=0, n_time=9`; from: E1; noiseless observations lie exactly on the fitted first-order model.
- **noiseless_arrhenius_recovery**, a special or limiting case with a known answer: expects 0; computed as `abs(Ea_hat - Ea_true) / Ea_true` from what the simulation returns on `noise_sigma_over_c0=0, n_time=9`; from: E2, E3; noiseless k values lie exactly on the Arrhenius line.
- **first_stage_variance_closed_form**, an independent second implementation: expects 0; computed as `abs(var_k_jacobian - var_k_closed_form) / var_k_closed_form` from what the simulation returns on `noise_sigma_over_c0=0.01, n_time=9`; from: E1; the Gauss-Newton information of a one-parameter exponential is the closed form.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **noiseless_rate_accuracy**: Noiseless nonlinear fitting must recover all five generated rates.; the value the check “noiseless_decay_recovery” measures; lower is better; met where the check itself agrees (at 1e-07 or below: the check accepts within 1e-07); a change smaller than 1e-07 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/kinetics_parameter_recovery/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 101 more lines of code than two scripts (30 written by the model, 71 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- Support the hypothesis if weighted coverage is within 0.08 of 0.95 in most grid cells and closer to 0.95 than unweighted coverage.
- Count against it if weighted estimates have materially larger absolute bias or systematically worse coverage than unweighted estimates.

## Risks

- First-stage covariance estimates can be unstable at the highest noise and fewest time points.
- Nominal Gaussian intervals can miscover because ln k is a nonlinear transformation.
- The synthetic model excludes systematic errors and model misspecification.

## What this quest will not do

- Experimental laboratory data.
- Bayesian posterior inference.
- Generalisation beyond first-order decay and additive Gaussian measurement noise.

## Checks already made

- other; variables.controls / protocol.model.steps S1; The true lnA is never stated, yet S1 constructs temperatures from lnA_true and the lnA error needs a known value.; Without a stated lnA the temperatures, lnA error and lnA coverage cannot be reproduced.; Added lnA_true = ln(1e8) = 18.4207 (A = 1e8 s^-1, matching E2's example) to controls and thresholds; the resulting temperatures are about 290 K to 327 K.; medium
- estimand; protocol.ci_method / method; Interval construction is only 'covariance-based normal 95%', and the unweighted comparator's covariance is unspecified.; With 5 points and 2 parameters (3 dof), z=1.96 intervals are anti-conservative, and unweighted OLS needs residual-variance intervals, so the method comparison could reflect interval construction rather than weighting.; Report two interval variants for both methods: the normal-z interval from the propagated variances, and a t(3) interval from the residual scale. The unweighted comparator uses OLS residual-variance covariance. Both methods are fitted to the same synthetic data in each run.; high
- estimand; protocol.metrics ea_lna_correlation; The correlation has no stated uncertainty.; A headline number without an interval cannot support 'strongly correlated'.; Add a Fisher-z 95% interval from the successful-run count; the paper also reports the theoretical correlation implied by the regression design (from the centred-regression covariance) as the reference line.; medium
- threshold_sensitivity; protocol.acceptance; The 0.08 coverage-error cutoff and the '20 of 25' rule are chosen without justification.; The verdict depends on an arbitrary cut-off.; Evaluate the acceptance rule at 0.05, 0.08 and 0.10 and the count rule at 15, 20 and 23 of 25, and state the verdict only if it holds across the set.; medium
- oracle; protocol.oracles; There is a single oracle and it exercises only the first-stage fit; the Arrhenius stage and the covariance are unchecked.; An error in the E3 transform or in the propagated variance would pass the noiseless test.; Add a second oracle that fits noiseless k values exactly on the Arrhenius line and expects Ea relative error 0. Add a third that compares the Jacobian Var(k) with the closed form sigma^2/sum((t c)^2) and expects 0 relative difference.; medium
- numerical_convergence; protocol.model / protocol.thresholds; There is no statement that the nonlinear fit is converged; only a solver tolerance is listed.; Loose termination would bias k_hat and the covariance.; Check on a 20-run subset that tightening the tolerance 100-fold changes k_hat by less than 1e-6 relative, and require every noiseless fit to satisfy the same bound.; low
- failed_runs; protocol.failure_policy; Failed trials are excluded with no rule for what happens when failures are frequent or depend on the setting.; Excluding failures at high noise and few points conditions the estimand on fit success.; Report the failure fraction per setting and label estimates as conditional on success. If any setting's failure fraction exceeds 1%, flag it and report a bound on the coverage assuming all failures miss.; low
- other; protocol.definitions; The k_hat definition quotes the Arrhenius equation, which does not define an NLS estimate. The metric ids 'Ea_error_J_mol' and 'ea_error_J_mol' also differ, and the definitions of ea_error and lna_error do not match their metric entries.; The definitions must match the source they quote and the metric ids.; Set differs=X1 for k_hat (the NLS estimator is the study's own definition, the quote supports only E2), and use the id ea_error_J_mol consistently.; low
- other; protocol.simplifications; Fixed c0 (not fitted), a well-specified model identical to the generator, and nominal linearised intervals are not listed as simplifications.; These make coverage easier to achieve than in real data and change what the topic names.; Add X3 (c0 known and fixed, single-parameter fit), X4 (the analysis model equals the generating model) and X5 (linearised Jacobian intervals).; medium
- single_point; hypothesis / expected_outcome; Ea, lnA, the k range 0.1 to 1 and the horizon 3/k are fixed at one point, but the hypothesis is stated generally.; The claim cannot extend beyond this parameter point.; Scope the hypothesis and conclusions to Ea = 50 kJ/mol, lnA = 18.42 and the 10-fold k range, with no general language.; low
- stratum_collapse; figures_planned / dependent; Per-temperature k behaviour is aggregated although temperatures are natural strata.; Poor coverage at the fastest or slowest k would be hidden by the mean.; Report k bias and k coverage per temperature, and plot Ea/lnA metrics per noise-by-n_time cell rather than marginal means.; low
- The methodology audit amended the design: expected_outcome, hypothesis, method, protocol, variables (before and after are in needs/DESIGN_CRITIQUE.json). The audit proposed changing `protocol.criteria`, but FI did not apply it: `protocol.criteria` entry 'noiseless_arrhenius_accuracy' names no declared check ('noiseless_arrhenius_recovery'); the protocol's checks are: noiseless_decay_recovery; it was left out of the plan and is not used The audit proposed changing `protocol.thresholds`, but FI did not apply it: the threshold `coverage_error_cutoffs` was left out of `protocol.thresholds` because it is not one number ([0.05, 0.08, 0.1]); it is not used The design was read by a different model (claude-sonnet-5-5), not the one that wrote it (gpt-5.6-terra).
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.

## The design (used as written)

```yaml
hypothesis: At Ea=50 kJ/mol, lnA=18.42 and a 10-fold k range, two-stage weighted, centred Arrhenius regression
  gives lower bias and better-calibrated 95% intervals for Ea and ln A than unweighted regression when
  k estimates have temperature-dependent uncertainty.
variables:
  independent:
  - 'concentration noise standard deviation as a fraction of c0: 0.0025, 0.005, 0.01, 0.02, 0.04'
  - 'number of equally spaced times from 0 to 3/k: 5, 7, 9, 13, 17'
  - 'Arrhenius-regression method: weighted centred versus unweighted centred'
  dependent:
  - run-level Ea error and 95% CI inclusion (z and t(3) intervals)
  - run-level ln A error and 95% CI inclusion (z and t(3) intervals)
  - Monte Carlo spread of Ea and ln A estimates
  - per-temperature fitted k, k bias and k coverage (per temperature)
  - Monte Carlo correlation between Ea and ln A estimates with Fisher-z interval
  controls:
  - c0=1
  - R=8.314462618 J mol^-1 K^-1
  - Ea=50000 J mol^-1
  - A=1e8 s^-1 (lnA=18.4207)
  - five temperatures with k from 0.1 to 1 s^-1, geometrically spaced
  - additive independent Gaussian concentration noise
  - fixed sampling horizon 3/k
  - base seed 20261015
method: 'Use NumPy and SciPy nonlinear least squares to fit c(t)=c0 exp(-kt) independently at five temperatures
  with c0 fixed. Propagate each fitted k covariance by Var(ln k)=Var(k)/k^2, then use centred weighted
  least squares for ln k versus 1/T and an unweighted centred comparator fitted to the same data. Intervals
  for both methods are reported in two forms: normal-z from the propagated or OLS-residual covariance,
  and t(3). Each of 25 noise-by-time settings has 400 independent synthetic five-temperature experiments;
  report Monte Carlo means with t-interval standard errors, SDs with chi-square intervals, Wilson coverage
  intervals, per-temperature k bias and coverage, and the Ea--lnA correlation with a Fisher-z interval.
  Failure counts are reported per setting. A tighter-tolerance check on a 20-run subset confirms solver
  convergence.'
expected_outcome: Weighted fitting should remain approximately unbiased and have coverage nearer 95% than
  the unweighted comparator at this parameter point, while Ea and ln A estimates remain strongly positively
  correlated; this simulation directly tests calibration rather than assuming it.
figures_planned:
- figures/ea_calibration.png
- figures/lna_calibration.png
- figures/parameter_correlation.png
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: ea_coverage_95
  min: 0
  max: 1
  unit: dimensionless
  reason: Coverage is a fraction of Monte Carlo trials.
- path: lna_coverage_95
  min: 0
  max: 1
  unit: dimensionless
  reason: Coverage is a fraction of Monte Carlo trials.
- path: ea_lna_correlation
  min: -1
  max: 1
  unit: dimensionless
  reason: A Pearson correlation is bounded by definition.
- path: k_coverage_95
  min: 0
  max: 1
  unit: dimensionless
  reason: Per-temperature interval coverage is a fraction of trials.
protocol:
  grid:
    noise_sigma_over_c0:
    - 0.0025
    - 0.005
    - 0.01
    - 0.02
    - 0.04
    n_time:
    - 5
    - 7
    - 9
    - 13
    - 17
  runs_per_setting: 400
  min_runs_per_setting: 400
  thresholds:
    ci_level: 0.95
    z_975: 1.959963984540054
    temperature_count: 5
    k_min_s_inv: 0.1
    k_max_s_inv: 1.0
    sampling_horizon_k_times_t: 3
    R_J_mol_K: 8.314462618
    Ea_true_J_mol: 50000
    c0: 1
    solver_relative_tolerance: 1.0e-10
  seed_policy: Every setting and run uses an independent NumPy Generator stream derived from base seed
    20261015, setting indices, and run index; no common random numbers are used across settings or runs.
    Within a run the weighted and unweighted methods are fitted to the same synthetic data, so method
    comparisons are paired.
  failure_policy: A failed or non-finite first-stage or second-stage fit is counted as a failed trial,
    excluded from estimator summaries (which are therefore conditional on success), and listed with setting
    and run index; its frequency is reported per setting, and any setting above 1% is flagged with a worst-case
    coverage bound counting failures as misses.
  ci_method: Each run uses covariance-based 95% intervals in two forms (normal-z and t with 3 dof; the
    unweighted method uses OLS residual-variance covariance); pooled coverage receives a Wilson binomial
    95% interval, Monte Carlo mean bias a t interval, SD uncertainty a chi-square interval, and the Ea--lnA
    correlation a Fisher-z interval. The weighted-minus-unweighted coverage difference is analysed as
    paired.
  acceptance:
  - For each method and interval form, report coverage rather than forcing it to 95%.
  - The weighted hypothesis is supported only if its absolute coverage error is at most the cutoff in
    at least the stated number of the 25 settings, for every cutoff in {0.05, 0.08, 0.10} and count in
    {15, 20, 23}, and its median absolute coverage error is no larger than the unweighted baseline; otherwise
    the verdict is reported as threshold-dependent.
  - All solver failures and excluded trials must be reported.
  precision:
    target_half_width: 0.05
    metric: ea_coverage_95 and lna_coverage_95
    reason: With 400 independent runs the conservative binomial 95% half-width is about 0.049 (about 0.021
      near coverage 0.95), sufficient to distinguish gross interval miscalibration at the 0.08 cutoff
      but not differences under 0.04.
  metrics:
  - id: k_hat_s_inv
    estimand: Mean fitted rate constant at each temperature.
    kind: mean
    unit: one fitted temperature-series rate constant per run
    cluster: run
    paired: true
    family: five temperatures
  - id: ea_error_J_mol
    estimand: Ea_hat minus known Ea.
    kind: mean
    unit: one five-temperature experiment
    cluster: null
    paired: true
    family: Ea method comparison
  - id: lna_error
    estimand: lnA_hat minus known lnA.
    kind: mean
    unit: one five-temperature experiment
    cluster: null
    paired: true
    family: lnA method comparison
  - id: ea_coverage_95
    estimand: Probability that the run-level Ea 95% interval contains true Ea.
    kind: proportion
    unit: one five-temperature experiment
    cluster: null
    paired: true
    family: Ea method comparison
  - id: lna_coverage_95
    estimand: Probability that the run-level lnA 95% interval contains true lnA.
    kind: proportion
    unit: one five-temperature experiment
    cluster: null
    paired: true
    family: lnA method comparison
  - id: k_coverage_95
    estimand: Probability that a run-level k 95% interval contains true k, reported per temperature.
    kind: proportion
    unit: one temperature-series fit
    cluster: run
    paired: true
    family: five temperatures
  - id: ea_lna_correlation
    estimand: Pearson correlation of Ea_hat and lnA_hat across runs.
    kind: mean
    unit: one setting-level Monte Carlo correlation
    cluster: null
    paired: false
    family: weighted method settings
  oracles:
  - name: noiseless_decay_recovery
    kind: special_case
    check: Run one noiseless five-temperature trial and measure the largest relative error of fitted k
      values.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-07
    tolerance_mode: absolute
    reference: E1; noiseless observations lie exactly on the fitted first-order model.
    case:
      noise_sigma_over_c0: 0
      n_time: 9
    measure: max_k_relative_error
  - name: noiseless_arrhenius_recovery
    kind: special_case
    check: Run one noiseless trial and measure the relative error of Ea_hat from the weighted Arrhenius
      fit.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: E2, E3; noiseless k values lie exactly on the Arrhenius line.
    case:
      noise_sigma_over_c0: 0
      n_time: 9
    measure: abs(Ea_hat - Ea_true) / Ea_true
  - name: first_stage_variance_closed_form
    kind: second_implementation
    check: Compare the Jacobian-based Var(k) at the true k on the sampling grid with the closed form sigma^2
      / sum((t*c0*exp(-k t))^2).
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: E1; the Gauss-Newton information of a one-parameter exponential is the closed form.
    case:
      noise_sigma_over_c0: 0.01
      n_time: 9
    measure: abs(var_k_jacobian - var_k_closed_form) / var_k_closed_form
  criteria:
  - name: noiseless_rate_accuracy
    what: Noiseless nonlinear fitting must recover all five generated rates.
    oracle: noiseless_decay_recovery
    use: value
    direction: lower
    target: 0
    tolerance: 1.0e-07
  simplifications:
  - id: X1
    replaces: Literature-sourced formal definitions for bias, spread, and coverage
    because: The supplied sources motivate uncertainty analysis but do not quote operational definitions
      for these Monte Carlo summaries.
    effect: Definitions are standard simulation estimands and do not alter generated data or fits.
    probe: Recompute summaries directly from saved run-level estimates and interval indicators.
  - id: X2
    replaces: A complete experimental uncertainty model
    because: The requested study specifies additive Gaussian concentration noise only.
    effect: Coverage may differ under multiplicative, correlated, or systematic error.
    probe: A later sensitivity run can replace additive noise with multiplicative or correlated noise.
  - id: X3
    replaces: Fitting c0 together with k
    because: The study fixes c0 as known so that each series has one free parameter.
    effect: k intervals are narrower than if c0 were also estimated, so coverage is a best case.
    probe: Refit a subset with c0 free and compare k spread.
  - id: X4
    replaces: A mis-specified analysis model
    because: Data are generated from the same first-order law the fit assumes.
    effect: Results show calibration under a correct model only; model error is not tested.
    probe: State the claim as 'within the first-order Arrhenius model'.
  - id: X5
    replaces: Exact nonlinear confidence intervals
    because: Jacobian linearisation and Var(ln k)=Var(k)/k^2 give nominal intervals.
    effect: Coverage at few points and high noise may deviate from 95% because of the linearisation.
    probe: Compare against empirical SD of k_hat across runs in each cell.
  definitions:
  - metric: k_hat_s_inv
    defined_as: The positive nonlinear-least-squares estimate of the first-order rate constant k in s^-1.
    source: '[1]'
    quote: 'the Arrhenius equation: k = A exp(B/T).'
    differs: X1
  - metric: ea_error_J_mol
    defined_as: Ea_hat minus the known generating Ea, aggregated across independent runs.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: lna_error
    defined_as: lnA_hat minus the known generating lnA, aggregated across independent runs.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: ea_coverage_95
    defined_as: Fraction of successful runs whose nominal 95% Ea interval contains generating Ea.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: lna_coverage_95
    defined_as: Fraction of successful runs whose nominal 95% lnA interval contains generating lnA.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: k_coverage_95
    defined_as: Fraction of successful temperature-series fits whose nominal 95% k interval contains generating
      k.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: ea_lna_correlation
    defined_as: Pearson correlation of paired Ea_hat and lnA_hat values across successful runs in one
      setting.
    source: '[1]'
    quote: The close correlation of the two parameters is shown, which requires the computation of the
      covariance matrix for the representation of uncertainties.
    differs: ''
  model:
    summary: Independent first-order concentration series are generated exactly, perturbed by Gaussian
      concentration noise, fitted by nonlinear least squares, and transformed to a centred weighted Arrhenius
      regression.
    assumptions:
    - c0 is known exactly.
    - Temperature is measured without error.
    - The fitted k remains positive.
    - Covariance linearisation is adequate for reporting nominal intervals.
    holds_for: Positive rates, temperatures, and sampling times; the stated study uses noise_sigma_over_c0
      from 0.0025 to 0.04, Ea=50 kJ/mol, lnA=18.42 and k from 0.1 to 1 s^-1.
    equations:
    - id: E1
      formula: c(t) = c0 * exp(-k * t)
      role: generates
      source: derivation
      derivation: For dc/dt=-k c, separate dc/c=-k dt; integrate ln(c/c0)=-kt; exponentiate.
      example:
        inputs:
          c0: 2
          k: 0.4
          t: 1.5
        expected_formula: c0*exp(-k*t)
    - id: E2
      formula: k(T) = A * exp(-Ea / (R * T))
      role: generates
      source: '[1]'
      example:
        inputs:
          A: 100000000
          Ea: 50000
          R: 8.314462618
          T: 350
        expected_formula: A*exp(-Ea/(R*T))
    - id: E3
      formula: Ea = -R*b; lnA = alpha - b*xbar, where ln(k)=alpha+b*(1/T-xbar)
      role: analyses
      source: derivation
      derivation: Expand alpha+b(1/T-xbar)=lnA-Ea/(RT); match the coefficient b=-Ea/R and intercept alpha=lnA+b*xbar.
      example:
        inputs:
          R: 8.314462618
          b: -6000
          alpha: 2
          xbar: 0.003
        expected_formula:
          Ea: -R*b
          lnA: alpha-b*xbar
    steps:
    - id: S1
      name: temperature construction
      does: Construct five inverse-temperature values so E2 gives geometrically spaced k values from 0.1
        to 1 s^-1.
      inputs:
        Ea_true_J_mol: J mol^-1
        R_J_mol_K: J mol^-1 K^-1
        lnA_true: dimensionless
      output: five temperatures in K
      uses:
      - E2
      source: derivation
      relations:
      - The ratio of the largest to smallest generated rate is exactly 10.
    - id: S2
      name: simulate concentrations
      does: Evaluate E1 at n_time equally spaced times from 0 to 3/k and add independent N(0,(noise_sigma_over_c0*c0)^2)
        noise.
      inputs:
        noise_sigma_over_c0: fraction of c0
        n_time: count
      output: five noisy concentration vectors
      uses:
      - E1
      source: derivation
      relations:
      - Sampling includes t=0 and t=3/k for every temperature.
    - id: S3
      name: first-stage fit
      does: Fit positive k by SciPy nonlinear least squares with c0 fixed, using the Jacobian covariance
        to form Var(ln k).
      inputs:
        concentration: noisy concentration vector
        time_s: sampling times in s
      output: k_hat, Var(k_hat), and Var(ln k_hat)
      uses:
      - E1
      source: '[19]'
      relations:
      - k_hat is strictly positive because the fit is bounded positive.
    - id: S4
      name: Arrhenius fit
      does: Fit centred ln k values by weighted and unweighted linear least squares, then apply E3 and
        covariance propagation.
      inputs:
        k_hat_s_inv: five fitted rates
        var_lnk: five propagated variances
        temperature_K: five temperatures
      output: Ea_hat, lnA_hat, covariance, and nominal intervals
      uses:
      - E3
      source: '[1]'
      relations:
      - Centred predictor values sum to zero.
    - id: S5
      name: Monte Carlo summaries
      does: Aggregate run-level errors, SDs, interval indicators, rate summaries, and Ea--lnA correlations
        by grid cell and method.
      inputs:
        run_results: 400 independent trials per grid cell
      output: bias, spread, coverage, correlation, and their uncertainty
      uses: []
      source: derivation
      relations:
      - Coverage counts lie between zero and the number of successful trials.
```
