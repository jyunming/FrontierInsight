# Plan: TOPIC. How accurately can a reaction rate constant and activation energy be recovered from noisy concentration measurements?

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This study evaluates the reliability of activation energy (Ea) recovery from noisy first-order reaction data by mapping the coverage probability of 95% confidence intervals against noise and sampling density.

## What the literature says

- **[20]**: Discusses the provisos under which parametric standard errors in nonlinear least squares can be trusted.
- **[13]**: Highlights that standard NLS may yield unrealistic confidence intervals if data variances are unequal or improperly handled.
- **[16]**: Notes that linearized methods for biochemical kinetics are often unreliable compared to nonlinear least squares.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] ChemInform Abstract: SENSITIVITY ANALYSIS AND PARAMETER ESTIMATION IN DYNAMIC MODELING OF CHEMICAL KINETICS (DOI 10.1002/chin.198345092)
- [2] Global Dynamic Optimization for Parameter Estimation in Chemical Kinetics (DOI 10.1021/jp0548873)
- [3] Least-squares analysis of data with uncertainty in x and y: A Monte Carlo methods comparison (DOI 10.1016/j.chemolab.2010.07.003)
- [4] Uncertainty analysis of linear least square fitting apply to non-linear model (DOI 10.1088/1742-6596/1144/1/012153)
- [5] EFFICIENT EXPERIMENTAL DESIGNS FOR CHEMICAL KINETICS STUDIES (DOI 10.5935/0100-4042.20140102)
- [6] Toolbox for Parameter Estimation and Simulation in Dynamic Systems with Application to Chemical Kinetics
- [7] Fitting Parameter Uncertainties in Least Squares Fitting
- [8] A numerical compass for experiment design in chemical kinetics and molecular property estimation (DOI 10.1186/s13321-024-00825-0)
- [9] The Statistical Experimental Design for Chemical Reactors Modeling (DOI 10.4236/am.2016.714133)
- [10] Sensitivity analysis and parameter estimation in dynamic modeling of chemical kinetics (DOI 10.1002/kin.550150709)
- [11] Reliability and uncertainty in the estimation of pK a by least squares nonlinear regression analysis of multiwavelength spectrophotometric pH titration data (DOI 10.1007/s00216-006-0993-1)
- [12] The effects of different experimental designs on parameter estimation in the kinetics of a reversible chemical reaction (DOI 10.1016/s0169-7439(00)00084-8)
- [13] Improving uncertainty analysis in kinetic evaluations using iteratively reweighted least squares (DOI 10.1002/etc.630)
- [14] The application of fitting methods to chemical kinetics (DOI 10.1016/0010-4825(75)90023-2)
- [15] KBR (Kinetics in Batch Reactors): a MATLAB-based application with a friendly Graphical User Interface for chemical kinetic model simulation and parameter estimation (DOI 10.1016/j.ece.2018.11.003)
- [16] Spreadsheet Method for Evaluation of Biochemical Reaction Rate Coefficients and Their Uncertainties by Weighted Nonlinear Least-Squares Analysis of the Integrated Monod Equation (DOI 10.1128/aem.64.6.2044-2050.1998)
- [17] D- and T-optimum designs for the kinetics of a reversible chemical reaction (DOI 10.1016/s0169-7439(98)00046-x)
- [18] Catalytic conversion of a biomass-derived oil to fuels and chemicals II: Chemical kinetics, parameter estimation and model predictions (DOI 10.1016/0961-9534(95)00019-4)
- [W1] Parameter Estimation of Complex Chemical Kinetics with ...
- [19] An Algorithm for Least-Squares Estimation of Nonlinear Parameters (DOI 10.1137/0111030)
- [W2] (PDF) KBR (Kinetics in Batch Reactors): a MATLAB-based application with a Friendly Graphical User Interface for chemical kinetic model simulation and parameter estimation
- [W3] Parameter estimation in kinetic models of complex heterogeneous catalytic reactions using Bayesian statistics | Reaction Kinetics, Mechanisms and Catalysis | Springer Nature Link
- [20] Can you trust the parametric standard errors in nonlinear least squares? Yes, with provisos (DOI 10.1016/j.bbagen.2017.12.016)
- [21] Eurokin. Chemical Reaction Kinetics in Practice (DOI 10.1023/a:1011928218694)
- [W4] Kinetic Data Analysis and Parameter Estimation | Springer Nature Link
- [22] Chemical reactor design and operation
- [W5] Predictive Chemical Kinetic Modeling: Where We Succeed, Where We Struggle, and What Comes Next - PMC
- [W6] Parameter Estimation for Kinetic Models of Chemical Reaction Networks from Partial Experimental Data of Species’ Concentrations
- [W7] Parameter estimation in stochastic chemical kinetic models using derivative free optimization and bootstrapping - PMC
- [23] Degrees of freedom for nonlinear least squares estimation (DOI 10.48550/arxiv.1402.2997)
- [W8] Optimization-based framework for modeling and kinetic parameter estimation - ScienceDirect
- [W9] Progress in Accurate Chemical Kinetic Modeling, Simulations, and Parameter Estimation for Heterogeneous Catalysis | ACS Catalysis | ACS Publications
- [24] Numerical methods for optimum experimental design in DAE systems (DOI 10.1016/s0377-0427(00)00300-9)
- [25] Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren (DOI 10.1515/zpch-1889-0416)
- [26] A method for the solution of certain non-linear problems in least squares (DOI 10.1090/qam/10666)
- [27] Numerical Recipes, The Art of Scientific Computing (DOI 10.1119/1.14981)
- [28] Introduction to Linear Regression Analysis. (DOI 10.2307/2290746)
- [29] Chemical Reactor Analysis and Design
- [30] Data Reduction and Error Analysis for the Physical Sciences (DOI 10.1063/1.4823194)
- [31] Introduction to Chemical Engineering Kinetics & Reactor Design
- [32] Optimum Experimental Designs (DOI 10.2307/1270278)

## The gap this experiment addresses

While the bias of NLS is well-studied, the specific 'coverage' (how often the true value resides in the reported CI) for the two-step Arrhenius recovery process as a function of sampling density and noise is not explicitly mapped.

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** First-order reaction kinetics with Arrhenius temperature dependence and known initial concentration, measured in mol/L; k and A are in s^-1.

**What it assumes:** The reaction is strictly first order; The Arrhenius equation holds over the temperature range; The initial concentration and Gaussian noise standard deviation are known; Concentration measurements have independent additive Gaussian errors

**Where it holds:** T from 300 K to 1/(1/300-8.314*ln(10)/50000) K, approximately 338.94 K; the corresponding k range is approximately 19.7 to 197 s^-1. The synthetic model is stipulated over this entire range.

**Its equations:**

- **E1** (produces the data): `c(t) = c0 * exp(-k * t)`. Source: derivation: Integration of dc/dt = -kc
- **E2** (produces the data): `k = A * exp(-Ea / (R * T))`. Source: derivation: Arrhenius law stipulated for this simulation
- **E3** (analyses the results): `ln(k / 1 s^-1) = ln(A / 1 s^-1) - (Ea/R) * (1/T)`. Source: derivation: ln(k / 1 s^-1) = ln(A * exp(-Ea / (R * T)) / 1 s^-1) = ln(A / 1 s^-1) - Ea / (R * T)

**The checks against known answers, and where each expected value comes from:**

- **concentration_closed_form**, a special or limiting case with a known answer: expects 0.36787944117144233; computed as `c_val` from what the simulation returns on `c0=1, k=1, t=1`; from: derivation: c=c0*exp(-k*t) with c0=1, k=1, t=1
- **concentration_nonunit_initial**, a special or limiting case with a known answer: expects 0.7357588823428847; computed as `c_val` from what the simulation returns on `c0=2, k=2, t=0.5`; from: derivation: substitute c0=2, k=2, and t=0.5 into c=c0*exp(-k*t) => c = 2 * exp(-2 * 0.5)
- **arrhenius_closed_form**, a special or limiting case with a known answer: expects 19.7; computed as `k_val` from what the simulation returns on `A=10000000000.0, Ea=50000, R=8.314, T=300`; from: derivation: k=A*exp(-Ea/(R*T)); expected rounded to three significant figures
- **arrhenius_log_relation**, a special or limiting case with a known answer: expects 2.9803286656; computed as `ln_k_val` from what the simulation returns on `A=10000000000.0, Ea=50000, R=8.314, T=300`; from: E3
- **noise_free_ea_recovery**, a special or limiting case with a known answer: expects 50000; computed as `Ea_fit` from what the simulation returns on `c0=1, A=10000000000.0, Ea=50000, R=8.314, T_low=300, T_high_formula=1/(1/300-8.314*ln(10)/50000), num_temperatures=5, num_time_points=10, sampling_horizon_factor=3, noise_ratio=0, Ea_true=50000, use_equal_weights=True`; from: derivation: slope = -Ea/R => Ea = -R * slope

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **physics_check**: Absolute error in concentration for c0=1, k=1, t=1; how far the check “concentration_closed_form” lands from the value it expects; lower is better; met at 1e-06 or below; a change smaller than 1e-07 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/kinetics_parameter_recovery/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 101 more lines of code than two scripts (30 written by the model, 71 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- The hypothesis is supported if the coverage probability drops significantly below 0.95 for noise_ratio > 0.01, despite bias remaining small.

## Risks

- Curve_fit may fail to converge for very high noise levels, potentially biasing the 'coverage' if failures are not handled correctly.

## What this quest will not do

- Exploring non-first-order kinetics
- Using Bayesian MCMC for parameter estimation

## Checks already made

- pseudo_units; method and protocol.model.equations.E3; The design takes ln k and ln A without specifying a reference unit for either dimensional quantity.; A logarithm of a dimensional rate changes numerically when the rate unit changes.; Define k and A in s^-1 and fit ln(k / 1 s^-1), so the reported intercept is ln(A / 1 s^-1).; medium
- stratum_collapse; variables.dependent and figures_planned; The design generates five temperatures but plans only aggregate Ea results, leaving rate-constant recovery at each temperature unreported.; A temperature-specific failure can disappear in an aggregate summary.; Report relative k error and k interval coverage separately at each temperature and sampling setting.; high
- precision; protocol.runs_per_setting and protocol.precision; The target coverage half-width is 0.03, but 1000 trials give a worst-case approximate 95% half-width of 0.031.; The stated run count does not meet its own precision target across possible coverage probabilities.; Use 1200 trials per setting, supporting an approximate worst-case half-width of 0.0283, and benchmark the full grid against the 60-second limit.; medium
- estimand; protocol.ci_method and protocol.metrics; The design gives a per-trial Ea interval but no sampling intervals for reported coverage, bias, or spread; it also omits ln A and k recovery metrics.; A point estimate alone cannot show the precision of a headline result, and the proposed metrics do not answer the full recovery question.; Specify pooled-count Wilson intervals for coverage, trial-level intervals for bias and spread, and corresponding k and ln A metrics.; high
- rng_independence; protocol.seed_policy; The phrase 'derived from a base seed' does not specify a reproducible mapping from settings and trials to streams.; Reusing or changing stream assignments can create hidden pairing or make results irreproducible.; Use SeedSequence with a fixed base seed and the grid indices and trial index as its spawn key.; medium
- numerical_convergence; method and protocol.acceptance; The nonlinear fits have no tolerance-tightening check or stated invariant.; Solver error or an invalid fitted rate could be mistaken for statistical error.; Repeat representative fits with tighter tolerances, check positive finite rates and the noiseless decay invariant, and report the largest numerical change.; medium
- oracle; protocol.oracles.arrhenius_closed_form; The stated Arrhenius expectation, 1.948164 s^-1, is about tenfold below the value implied by its own case.; An incorrect oracle would reject a correct implementation or encourage an incorrect one.; Correct the expected value to 19.7 s^-1 at the stated precision and add a noise-free full-pipeline recovery oracle.; high
- failed_runs; protocol.failure_policy; Failed fits are excluded from coverage, so the headline probability is conditional on successful fitting without saying so.; Failure may become more frequent at high noise and make coverage look better than the procedure users actually obtain.; Count failures as noncoverage for operational coverage, report their rate, and label bias and spread as conditional on valid fits.; high
- other; method and protocol.ci_method; curve_fit covariance and weighted-regression covariance are unspecified, and the default residual scaling can alter the reported intervals.; The simulation supplies a known noise standard deviation, while each five-point fit has few residual degrees of freedom.; Pass the known concentration sigma with absolute_sigma=True and use the unscaled weighted-regression covariance, with delta-method log-rate variances.; high
- other; method and protocol.model.holds_for; The stated A and Ea produce roughly 19.7 s^-1 at 300 K and a rate ratio near 17.5 across 300–350 K, contrary to the claimed tenfold span and 1–11 s^-1 model range.; The stated physical settings do not generate the experiment described.; Set the upper temperature by the exact tenfold-rate equation, approximately 338.94 K, and correct the model range.; high
- other; hypothesis, expected_outcome and protocol.acceptance; The design predicts falling coverage and requires low-noise coverage within 0.05 of 0.95, although these are empirical results rather than guaranteed checks.; A stochastic outcome should not serve as a code-validity condition or a predetermined conclusion.; State coverage deterioration as a hypothesis, compare coverage intervals with the nominal 0.95 line, and reserve acceptance checks for implementation properties.; medium
- a `protocol.metrics` entry was left out of the plan because it could not be checked (`protocol.metrics` entry `ea_spread`: `kind` must be `proportion` or `mean`); the metrics that could be read are used
- a `protocol.metrics` entry was left out of the plan because it could not be checked (`protocol.metrics` entry `k_relative_spread_by_temperature`: `kind` must be `proportion` or `mean`); the metrics that could be read are used
- a `protocol.metrics` entry was left out of the plan because it could not be checked (`protocol.metrics` entry `lnA_spread`: `kind` must be `proportion` or `mean`); the metrics that could be read are used
- a `protocol.metrics` entry was left out of the plan because it could not be checked (`protocol.metrics` entry `ea_lnA_correlation`: `kind` must be `proportion` or `mean`); the metrics that could be read are used
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, protocol, result_assertions, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gemma4:31b).
- The design claims something about how a result changes with a parameter (convergence, scaling, a threshold), but noise_ratio (4 values), num_time_points (3 values) has fewer than five values: a rate, a fit or a critical point cannot be told from so few points. Add values, in particular near where the behaviour changes.
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.
- Equation E2 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E3 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- The check 'noise_free_ea_recovery' takes its expected value from a source this quest did not retrieve (“independent analytic slope of ln(k / 1 s^-1) against 1/T is -Ea/R for exact Arrh”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gemma4:31b). Its summary: The direct formula checks are valid but leave the initial-concentration factor weakly tested. The activation-energy recovery check needs a defined weighting rule before it can test the fit.
  - **concentration_closed_form**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Also check a case with non-unit initial concentration to catch omission of the c0 factor..
  - **arrhenius_closed_form**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures.
  - **noise_free_ea_recovery**: tests the model, would miss a plausible bug, its number is not well defined, its case may not isolate what it measures. The plan gives no rule for fitting when the variances used as weights are zero.; The noise-free case does not produce a defined weighted Ea_fit under the stated fitting method.; a better check: Specify fixed positive equal weights for this noise-free check, or check the log-rate relation directly from returned rates..
  - Found (medium): concentration_closed_form: The case cannot detect omission of the initial-concentration factor (Its initial concentration is unity, so the result is unchanged if that factor is dropped.). Fix: Add the proposed check with non-unit initial concentration.
  - Found (high): noise_free_ea_recovery: The specified weighted fit has undefined weights in this case (Noise-free measurements give zero variance for the fitted log rates, which cannot be inverted to form the stated weights.). Fix: Define positive equal weights for this check and use them in the fit.
  - Checked, and no problem found: concentration_closed_form: traced its case through E1 and verified its expected concentration and tolerance.; arrhenius_closed_form: traced its case through E2 and verified that the rounded expected rate lies within tolerance.; E1: checked the solution against integration of the stated first-order differential equation.; E2: checked the Arrhenius sign, units, and stipulated form.; E3: checked the dimensionless logarithms and algebraic derivation from E2..
  - Equations the checks test: E1, E2; not tested by any: E3.
- The model's 3 equations were read by gpt-6-sol, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is the standard definition or a correct derivation.
  - **E3** is the standard definition or a correct derivation.

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'concentration_nonunit_initial' was added (expects 0.7357588823428847, computed as `c_val`).
- What changed: the check 'arrhenius_log_relation' was added (expects 2.9793437939, computed as `ln_k_val`).
- What changed: the check 'noise_free_ea_recovery' changed: case {'c0': 1, 'A': 10000000000.0, 'Ea': 50000, 'R': 8.314, 'T_low': 300, 'T_high_formula': '1/(1/300-8.314*ln(10)/50000)', 'num_temperatures': 5, 'num_time_points': 10, 'sampling_horizon_factor': 3, 'noise_ratio': 0} → {'c0': 1, 'A': 10000000000.0, 'Ea': 50000, 'R': 8.314, 'T_low': 300, 'T_high_formula': '1/(1/300-8.314*ln(10)/50000)', 'num_temperatures': 5, 'num_time_points': 10, 'sampling_horizon_factor': 3, 'noise_ratio': 0, 'Ea_true': 50000, 'use_equal_weights': True}; where its expected value comes from changed.

- Before stopping or repairing, FI looked at the failing known-answer checks itself (at most one model call per check to work its expected value out again, two more runs of a deterministic check's case at smaller steps, three more seeds of a random check, one retry at twice the time limit): this run spent 1 model call(s) and 3 extra run(s) of the simulation on it.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: temperatures.
- To vary or compare: noise level.
- To vary or compare: number of time points.
- To report: k at each temperature.
- To report: variance of ln k.
- To report: Ea.
- To report: ln A.
- To report: bias.
- To report: spread of the estimates - NOT measured in the plan.
- To report: how often the true value falls inside the reported 95 percent interval.
- Formula the topic states: `c(t) = c0 * exp(-k * t)`.
- Formula the topic states: `k = A * exp(-Ea / (R * T))`.
- To compare: Compare recovery accuracy across noise levels.
- To compare: Compare recovery accuracy across numbers of time points.
- To compare: Compare the estimates with the true values.
- Constraint: Simulate noisy concentration-time data at five temperatures spanning a 10-fold range of k, with known A and Ea.
- Constraint: Use sampling times from 0 to 3/k at each temperature and additive Gaussian noise of a stated size.
- Constraint: Fit k at each temperature by nonlinear least squares.
- Constraint: Fit ln k against 1/T, centre 1/T, and weight each point by the variance of its ln k.
- Constraint: Note that the Ea and ln A estimates are strongly correlated.
- Constraint: Provide one Python script using numpy, with scipy where it helps, and matplotlib with the Agg backend.
- Constraint: Provide two or three figures that answer the question directly, each with the reference or theory value drawn.
- Constraint: Provide a short paper of at most 4 pages.
- Constraint: Cite only sources actually found and read; say when a reference cannot be confirmed rather than citing it.
- Constraint: Use CPU only and keep runtime under 60 seconds.
- Constraint: Use fixed random seeds where the run is random and state them in the paper.
- Constraint: Report every number with its uncertainty, or its numerical error for a deterministic result, and say when a result is only a check of the code.

- FI asked the plan's model to add what was missing; it is still missing.

- Still missing (the paper will say it was not studied; the quest stops before anything runs, because the topic's question cannot be answered without it):
  - The topic asks to report spread of the estimates, but the plan does not measure it.

## The design (used as written)

```yaml
hypothesis: Under the stated first-order, Arrhenius and additive-Gaussian model, nonlinear fitting followed
  by weighted Arrhenius regression recovers k, Ea and ln(A / 1 s^-1) with small bias at low noise. Whether
  nominal 95% interval coverage deteriorates as noise rises or sampling becomes sparse is an empirical
  question; the coverage interval will be compared with 0.95.
variables:
  independent:
  - noise_ratio
  - num_time_points
  dependent:
  - ea_coverage_probability
  - ea_bias
  - ea_spread
  - k_relative_bias_by_temperature
  - k_relative_spread_by_temperature
  - k_coverage_probability_by_temperature
  - lnA_bias
  - lnA_spread
  - lnA_coverage_probability
  - ea_lnA_correlation
  - fit_failure_probability
  - ln_k_variance_by_temperature
  controls:
  - num_temperatures
  - temp_range_k_factor
  - sampling_horizon_factor
method: Use c0=1 mol/L, A=1e10 s^-1, Ea=50000 J/mol and R=8.314 J/(mol K). Choose five temperatures equally
  spaced in 1/T from 300 K to T_high=1/(1/300-R ln(10)/Ea), giving an exact tenfold range of k. At each
  temperature sample num_time_points times uniformly from 0 to 3/k, including both endpoints, and add
  independent N(0,(noise_ratio*c0)^2) concentration errors. For each of the 12 grid settings perform 1200
  independent trials. Fit positive k with c0 fixed by batched one-dimensional nonlinear least squares
  using analytic derivatives and a safeguarded scalar minimization fallback; use known concentration sigma
  and the unscaled inverse least-squares information for sigma_k. A fit is accepted only if its covariance
  is finite, its normalized objective gradient meets the stated tolerance, and its residual sum of squares
  is no greater than that at the known generating k within numerical tolerance. Regress y=ln(k / 1 s^-1)
  on x=1/T minus its inverse-variance-weighted mean, with weights 1/(sigma_k/k)^2; use the unscaled covariance
  (X'WX)^-1 to obtain intervals for Ea=-R*slope and ln(A / 1 s^-1)=centered_intercept-slope*weighted_mean(1/T).
  Retain the joint covariance and report the empirical Ea–ln A correlation. Report pooled operational
  coverage with invalid trials counted as misses, valid-fit bias and standard deviation, and the failure
  rate, for Ea and ln A and for k at each temperature. Refit a prespecified reproducible sample of valid
  trials at solver tolerances 1e-9 and 1e-11 and report the largest parameter change as numerical error;
  check the per-run fit invariants and verify that noiseless concentration is monotone decreasing with
  c(3/k)/c0=exp(-3). Benchmark the complete script on a laptop CPU and retain the design only if it finishes
  within 60 seconds.
expected_outcome: Figures show bias, standard deviation and nominal 95% coverage against noise and sampling
  density, each with its zero-bias or 0.95 reference. The noise-free recovery is a code check. Claims
  about coverage loss or maintained coverage are made only where pooled-count uncertainty supports them,
  and only under the stated simulation model.
figures_planned:
- 'coverage_heatmap.png: Ea and ln A operational 95% coverage across noise ratio and time-point count,
  with 0.95 reference and pooled-count uncertainty annotations.'
- 'accuracy_vs_noise.png: Ea and ln A bias and standard deviation by sampling density, with zero-bias
  references, trial-level uncertainty and Ea–ln A correlation annotations.'
- 'k_by_temperature.png: relative k bias, standard deviation and operational 95% coverage by temperature
  and noise ratio, separated by sampling density, with zero-bias and 0.95 references.'
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: ea_coverage_probability
  min: 0
  max: 1
  unit: dimensionless
  reason: A probability over all attempted trials, including failed fits as misses, must be between 0
    and 1.
protocol:
  runs_per_setting: 1200
  thresholds: {}
  acceptance:
  - Both closed-form oracles and the noise-free full-pipeline oracle meet their stated tolerances.
  - Representative valid fits change Ea by less than 0.01 times its setting-specific Monte Carlo standard
    error when solver tolerance tightens from 1e-9 to 1e-11; report the observed maximum change.
  - Every attempted trial is counted once as valid or failed, and operational coverage lies in [0,1].
  - The complete CPU script finishes within 60 seconds on the target laptop; report measured runtime.
  precision:
    target_half_width: 0.03
    metric: ea_coverage_probability
    reason: At 1200 attempted trials per setting, the approximate worst-case 95% half-width is 0.98/sqrt(1200)=0.0283;
      this resolves coverage differences of roughly 0.10 while keeping 12 settings computationally feasible,
      subject to the runtime benchmark.
  metrics:
  - id: ea_coverage_probability
    estimand: Probability over all attempted trials that the procedure returns a valid Ea interval containing
      true Ea; a failed trial is a miss.
    kind: proportion
    unit: run
    cluster: null
    paired: false
    family: coverage
  - id: ea_bias
    estimand: Mean(Ea_fit - true Ea) conditional on a valid fit.
    kind: mean
    unit: J/mol
    cluster: null
    paired: false
    family: accuracy
  - id: k_relative_bias_by_temperature
    estimand: Mean((k_fit-k_true)/k_true) at each temperature conditional on a valid fit.
    kind: mean
    unit: dimensionless
    cluster: null
    paired: false
    family: accuracy
  - id: k_coverage_probability_by_temperature
    estimand: Probability over all attempted trials that a valid k interval at each temperature contains
      true k; a failed trial is a miss.
    kind: proportion
    unit: run
    cluster: null
    paired: false
    family: coverage
  - id: lnA_bias
    estimand: Mean(fitted ln(A / 1 s^-1) minus true ln(A / 1 s^-1)) conditional on a valid fit.
    kind: mean
    unit: dimensionless
    cluster: null
    paired: false
    family: accuracy
  - id: lnA_coverage_probability
    estimand: Probability over all attempted trials that a valid ln A interval contains the true value;
      a failed trial is a miss.
    kind: proportion
    unit: run
    cluster: null
    paired: false
    family: coverage
  - id: fit_failure_probability
    estimand: Probability over all attempted trials of an invalid first-stage or second-stage fit.
    kind: proportion
    unit: run
    cluster: null
    paired: false
    family: failure
  - id: ea_spread
    estimand: Mean absolute error of Ea fit conditional on a valid fit.
    kind: mean
    unit: J/mol
    cluster: null
    paired: false
    family: accuracy
  - id: k_relative_spread_by_temperature
    estimand: Mean absolute relative error of k fit at each temperature conditional on a valid fit.
    kind: mean
    unit: dimensionless
    cluster: null
    paired: false
    family: accuracy
  - id: lnA_spread
    estimand: Mean absolute error of ln A fit conditional on a valid fit.
    kind: mean
    unit: dimensionless
    cluster: null
    paired: false
    family: accuracy
  - id: ln_k_variance_by_temperature
    estimand: Mean of the variance of ln k used for weighting at each temperature conditional on a valid
      fit.
    kind: mean
    unit: dimensionless
    cluster: null
    paired: false
    family: accuracy
  oracles:
  - name: concentration_closed_form
    kind: special_case
    check: c_val
    expected: 0.36787944117144233
    expected_formula: exp(-1)
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: 'derivation: c=c0*exp(-k*t) with c0=1, k=1, t=1'
    case:
      c0: 1
      k: 1
      t: 1
    measure: c_val
  - name: concentration_nonunit_initial
    kind: special_case
    check: c_val
    expected: 0.7357588823428847
    expected_formula: c0 * exp(-k * t)
    tolerance: 1.0e-06
    tolerance_mode: relative
    reference: 'derivation: substitute c0=2, k=2, and t=0.5 into c=c0*exp(-k*t) => c = 2 * exp(-2 * 0.5)'
    case:
      c0: 2
      k: 2
      t: 0.5
    measure: c_val
  - name: arrhenius_closed_form
    kind: special_case
    check: k_val
    expected: 19.7
    expected_formula: 1e10 * exp(-50000 / (8.314 * 300))
    tolerance: 0.005
    tolerance_mode: relative
    reference: 'derivation: k=A*exp(-Ea/(R*T)); expected rounded to three significant figures'
    case:
      A: 10000000000.0
      Ea: 50000
      R: 8.314
      T: 300
    measure: k_val
  - name: arrhenius_log_relation
    kind: special_case
    check: ln_k_val
    expected: 2.9803286656
    expected_formula: log(A) - (Ea / (R * T))
    tolerance: 1.0e-06
    tolerance_mode: relative
    reference: E3
    case:
      A: 10000000000.0
      Ea: 50000
      R: 8.314
      T: 300
    measure: ln_k_val
  - name: noise_free_ea_recovery
    kind: special_case
    check: Ea_fit
    expected: 50000
    expected_formula: Ea_true
    tolerance: 0.1
    tolerance_mode: absolute
    reference: 'derivation: slope = -Ea/R => Ea = -R * slope'
    case:
      c0: 1
      A: 10000000000.0
      Ea: 50000
      R: 8.314
      T_low: 300
      T_high_formula: 1/(1/300-8.314*ln(10)/50000)
      num_temperatures: 5
      num_time_points: 10
      sampling_horizon_factor: 3
      noise_ratio: 0
      Ea_true: 50000
      use_equal_weights: true
    measure: Ea_fit
  criteria:
  - name: physics_check
    what: Absolute error in concentration for c0=1, k=1, t=1
    oracle: concentration_closed_form
    use: error
    direction: lower
    target: 1.0e-06
    tolerance: 1.0e-07
  topic_map:
  - item: first-order reaction follows c(t) = c0 * exp(-k * t)
    by: equation:E1
  - item: rate constant follows the Arrhenius law k = A * exp(-Ea / (R * T))
    by: equation:E2
  - item: five temperatures
    by: setting:num_temperatures
  - 10-fold range of k: null
    by: control:temp_range_k_factor
  - sampling times from 0 to 3/k: null
    by: control:sampling_horizon_factor
  - additive Gaussian noise: null
    by: setting:noise_ratio
  - Fit k at each temperature by nonlinear least squares: null
    by: method
  - fit ln k against 1/T: null
    by: equation:E3
  - centre 1/T: null
    by: method
  - weight each point by the variance of its ln k: null
    by: metric:ln_k_variance_by_temperature
  - recover Ea and ln A: null
    by: metric:ea_bias
  - Vary the noise level: null
    by: setting:noise_ratio
  - Vary... the number of time points: null
    by: setting:num_time_points
  - report bias: null
    by: metric:ea_bias, metric:k_relative_bias_by_temperature, metric:lnA_bias
  - report... spread: null
    by: metric:ea_spread, metric:k_relative_spread_by_temperature, metric:lnA_spread
  - report... how often the true value falls inside the reported 95 percent interval: null
    by: metric:ea_coverage_probability, metric:k_coverage_probability_by_temperature, metric:lnA_coverage_probability
  - Ea and ln A estimates are strongly correlated: null
    by: metric:ea_lnA_correlation
  grid:
    noise_ratio:
    - 0.001
    - 0.01
    - 0.05
    - 0.1
    num_time_points:
    - 5
    - 10
    - 20
    num_temperatures:
    - 5
  seed_policy: Use base seed 20261008. For zero-based noise index i, time-point index j and trial index
    r, construct numpy.random.SeedSequence(20261008, spawn_key=(0,i,j,r)) and pass it to default_rng;
    draw independent errors for the five temperatures from that generator. Each setting and trial has
    a distinct stream, with no pairing assumed. Use SeedSequence(20261008, spawn_key=(1,i,j,m)) for bootstrap
    metric index m, separate from simulation streams.
  failure_policy: Count every attempted trial. A failed fit, nonfinite covariance, nonpositive k, invalid
    log rate, failed fit invariant, or failed weighted regression is a failure and a miss for operational
    coverage; report failure counts and Wilson intervals by setting. Calculate bias, standard deviation
    and correlation only from valid trials, report their valid-trial counts, and never silently replace
    failed values.
  ci_method: For each valid trial, use Ea_fit +/- 1.96*sigma_Ea and analogous normal-approximation intervals
    for ln A and each k, with sigma from known-noise absolute covariance and unscaled weighted-regression
    covariance. Estimate operational coverage and failure probability from pooled counts over all 1200
    attempts, with 95% Wilson intervals. Estimate valid-fit mean error, mean absolute error and mean ln
    k variance with 95% Student-t intervals over individual valid trials. Estimate valid-fit standard
    deviations and Ea–ln A correlation with 95% percentile bootstrap intervals from 2000 resamples of
    individual valid trials using separate streams. True input parameters have no stochastic uncertainty;
    report solver-tolerance changes as numerical error for noise-free checks.
  model:
    summary: First-order reaction kinetics with Arrhenius temperature dependence and known initial concentration,
      measured in mol/L; k and A are in s^-1.
    assumptions:
    - The reaction is strictly first order
    - The Arrhenius equation holds over the temperature range
    - The initial concentration and Gaussian noise standard deviation are known
    - Concentration measurements have independent additive Gaussian errors
    holds_for: T from 300 K to 1/(1/300-8.314*ln(10)/50000) K, approximately 338.94 K; the corresponding
      k range is approximately 19.7 to 197 s^-1. The synthetic model is stipulated over this entire range.
    equations:
    - id: E1
      formula: c(t) = c0 * exp(-k * t)
      role: generates
      source: derivation
      derivation: Integration of dc/dt = -kc
      example:
        inputs:
          c0: 1.0
          k: 1.0
          t: 1.0
        expected_formula: exp(-1)
    - id: E2
      formula: k = A * exp(-Ea / (R * T))
      role: generates
      source: derivation
      derivation: Arrhenius law stipulated for this simulation
      example:
        inputs:
          A: 10000000000.0
          Ea: 50000
          R: 8.314
          T: 300
        expected_formula: 1e10 * exp(-50000 / (8.314 * 300))
    - id: E3
      formula: ln(k / 1 s^-1) = ln(A / 1 s^-1) - (Ea/R) * (1/T)
      role: analyses
      source: derivation
      derivation: ln(k / 1 s^-1) = ln(A * exp(-Ea / (R * T)) / 1 s^-1) = ln(A / 1 s^-1) - Ea / (R * T)
      example:
        inputs:
          A: 10000000000.0
          Ea: 50000
          R: 8.314
          T: 300
        expected_formula: log(1e10) - (50000 / (8.314 * 300))
```
