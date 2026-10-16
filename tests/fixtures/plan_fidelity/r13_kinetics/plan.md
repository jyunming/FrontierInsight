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

- **noiseless_decay_recovery**, a special or limiting case with a known answer: expects 0; computed as `max_k_relative_error` from what the simulation returns on `noise_sigma_over_c0=0, n_time=9`; from: derivation: each independently constructed series has c_i=c0*exp(-k_true*t_i), so its least-squares residual is 0 at k_hat=k_true => max_i(abs(k_hat_i-k_true_i)/k_true_i)=0.
- **generated_rate_ratio_and_arrhenius**, a special or limiting case with a known answer: expects 10; computed as `max(k_true)/min(k_true)` from what the simulation returns on `noise_sigma_over_c0=0, n_time=9`; from: derivation: S1 spaces k geometrically from 0.1 to 1 s^-1 using E2, so max(k)/min(k) = 1/0.1 = 10.
- **noiseless_arrhenius_recovery**, a special or limiting case with a known answer: expects 0; computed as `abs(Ea_hat - Ea_true) / Ea_true` from what the simulation returns on `noise_sigma_over_c0=0.01, n_time=9`; from: derivation: with y_i=ln(k_true_i)=lnA_true-Ea_true/(R*T_i), the centred weighted line has b=-Ea_true/R for any finite positive weights => abs(Ea_hat-Ea_true)/Ea_true=0.
- **noiseless_lnA_recovery**, a special or limiting case with a known answer: expects 0; computed as `abs(lnA_hat - lnA_true) / lnA_true` from what the simulation returns on `noise_sigma_over_c0=0.01, n_time=9`; from: derivation: noiseless k lie on the line ln(k)=lnA_true-Ea_true/(R*T), so alpha-b*xbar = lnA_true for finite positive weights => abs(lnA_hat-lnA_true)/lnA_true=0.
- **first_stage_variance_closed_form**, an independent second implementation: expects 0; computed as `max_over_temperatures_and_variances(abs(var_reported-var_closed_form)/var_closed_form)` from what the simulation returns on `noise_sigma_over_c0=0.01, n_time=9`; from: second implementation: evaluate Var(k)=sigma^2/sum((t*c0*exp(-k_hat*t))^2) and Var(ln k)=Var(k)/k_hat^2 in a test-only routine sharing no covariance code with the simulation; the one-parameter E1 Jacobian is J=-t*c0*exp(-k_hat*t), so (J'J)^-1*sigma^2 = sigma^2/sum((t*c0*exp(-k_hat*t))^2) => both relative differences are 0.
- **arrhenius_example_rate**, a special or limiting case with a known answer: expects 0; computed as `abs(k_E2_example - 3.448570123) / 3.448570123` from what the simulation returns on `noise_sigma_over_c0=0, n_time=9`; from: derivation: E2 gives k=100000000*exp(-50000/(8.314462618*350))=3.448570123e0 s^-1, so abs(k_E2-3.448570123)/3.448570123=0 at the stated comparison precision.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **noiseless_rate_accuracy**: Noiseless nonlinear fitting must recover all five independently constructed first-order rates when started away from the solution.; the value the check “noiseless_decay_recovery” measures; lower is better; met where the check itself agrees (at 1e-07 or below: the check accepts within 1e-07); a change smaller than 1e-07 counts as no change.

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

- FI asked the plan's model to write `protocol.simplifications` again, and it still has gaps (the experiment goes on; the paper says so).

- FI asked the plan's model to write `protocol.definitions` again, and it is complete now.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (claude-sonnet-5-5), not the one that wrote the plan (gpt-5.6-terra). Its summary: The three checks only confirm that fitter and generator agree and that the Jacobian matches its closed form; they do not test E2 or the S1 temperature construction independently, the lnA intercept, or the interval construction. The noiseless Arrhenius check has zero variances and therefore degenerate weights, which must be resolved before it can run.
  - **noiseless_decay_recovery**: tests the model, would miss a plausible bug, its number is well defined, its case isolates what it measures. The 1e-7 absolute tolerance on a relative error is tight against SciPy least_squares defaults (ftol/xtol/gtol of 1e-8) unless those are tightened or the start is far from the optimum; reachability is not guaranteed.; a better check: Also measure the largest relative error of the generated k against A*exp(-Ea/(R*T)) so S1 is tested, and start the fit away from the true k..
  - **noiseless_arrhenius_recovery**: tests the model, would miss a plausible bug, its number is not well defined, its case may not isolate what it measures. With noise 0 the propagated Var(ln k) is 0, so the weights 1/Var are infinite or NaN; the weighted fit is undefined unless weights are floored or replaced by equal weights, which the plan does not state.; The measured error mixes first-stage fit error with the regression, and the weights are degenerate at zero noise, so it does not isolate the weighted Arrhenius fit.; a better check: Add a noiseless lnA recovery check and use a finite nominal sigma for the weights (or the unweighted fit) in the noiseless case..
  - **first_stage_variance_closed_form**: tests the model, would catch a plausible bug, its number is not well defined, its case isolates what it measures. The case does not say which of the five temperatures is used, nor whether the worst case across them is taken. It evaluates at the true k rather than the fitted k that S3 actually uses, and the sigma used (nominal or residual-estimated) is not specified.; a better check: Take the worst relative violation over all five temperatures and also compare Var(ln k) with closed form / k^2 at the fitted k..
  - Found (high): noiseless_arrhenius_recovery: At zero noise the first-stage variances are zero, so the weights 1/Var(ln k) are infinite or NaN (Weighted least squares with zero variances is undefined, so the check either crashes or silently falls back to something unstated.). Fix: State the weight rule at zero noise (equal weights or a floor) or use a tiny nonzero nominal sigma for the weights.
  - Found (medium): noiseless_arrhenius_recovery: Only Ea is checked, so errors in the lnA intercept (xbar centring in E3) are not detected (Ea depends only on the slope b, not on alpha or xbar.). Fix: Add a noiseless lnA recovery check.
  - Found (medium): noiseless_decay_recovery: The 1e-7 tolerance may be unreachable at SciPy's default tolerances, and the check uses the same E1 for generating and fitting (A consistent mistake in E1 or the time grid cancels, and a convergence tolerance of 1e-8 can leave errors near 1e-7 to 1e-6.). Fix: Tighten ftol/xtol/gtol or relax to about 1e-6, and add an independent check of the generated k against E2.
  - Found (medium): first_stage_variance_closed_form: The case does not specify which temperature or what sigma is used, and the check evaluates at the true k rather than the fitted k (The scalar measure is ambiguous across the five temperatures, and SciPy's pcov is rescaled by the residual variance unless absolute_sigma=True, so what S3 reports may differ from the formula.). Fix: Specify the temperature(s) or take the worst over all five, state that the nominal sigma is used (absolute_sigma=True), and add the Var(ln k)=Var(k)/k^2 comparison.
  - Found (medium): E2: No check tests the E2 step or S1 temperature construction independently of the code that uses it (True k values come from the same function used by the fit comparison, so a wrong R or sign in E2 cancels (the noiseless Arrhenius check reads back Ea).). Fix: Add a check of the generated k ratio, and a hand-computed k at one T from E2's example.
  - Found (medium): S4 / coverage: The plan's approximations are not all listed: intervals use linearised covariance with estimated, hence random, weights and the sigma is taken as known or estimated without stating which; coverage uses nominal normal intervals (not a t-interval at 3 d.o.f. for five points) (With only five Arrhenius points and estimated weights, nominal 95 percent intervals are expected to undercover, and no check tests the interval construction at all.). Fix: List these in the plan's simplifications and add a check of the interval half-width against a hand-computed value for a fixed covariance.
  - Checked, and no problem found: E1: c0*exp(-k t) is the standard first-order integrated rate law and the derivation is correct.; E2: the Arrhenius form k = A exp(-Ea/(R T)) is standard, with Ea in J/mol and R in J/(mol K).; E3: the slope/intercept mapping for the centred regression, b=-Ea/R and lnA=alpha-b*xbar, is algebraically correct.; first_stage_variance_closed_form: sigma^2/sum((t c0 e^{-kt})^2) is the correct Gauss-Newton variance for a one-parameter model, and the tolerance 1e-6 is reachable.; noiseless_decay_recovery: expected 0 and the relative-error form are consistent with the kind (special_case, quantity itself)..
  - Equations the checks test: E1, E3; not tested by any: E2.
- The model's 3 equations were read by claude-sonnet-5-5, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is the standard definition or a correct derivation.
  - **E3** is the standard definition or a correct derivation.

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'noiseless_decay_recovery' changed: what it checks is worded differently; where its expected value comes from changed.
- What changed: the check 'generated_rate_ratio_and_arrhenius' was added (expects 10, computed as `max(k_true)/min(k_true)`).
- What changed: the check 'noiseless_arrhenius_recovery' changed: case {'noise_sigma_over_c0': 0, 'n_time': 9} → {'noise_sigma_over_c0': 0.01, 'n_time': 9}; what it checks is worded differently; where its expected value comes from changed.
- What changed: the check 'noiseless_lnA_recovery' was added (expects 0, computed as `abs(lnA_hat - lnA_true) / lnA_true`).
- What changed: the check 'first_stage_variance_closed_form' changed: measure 'abs(var_k_jacobian - var_k_closed_form) / var_k_closed_form' → 'max_over_temperatures_and_variances(abs(var_reported-var_closed_form)/var_closed_form)'; what it checks is worded differently; where its expected value comes from changed.
- What changed: the check 'arrhenius_example_rate' was added (expects 0, computed as `abs(k_E2_example - 3.448570123) / 3.448570123`).
- Still not in its kind's form (read it before the run): The check 'first_stage_variance_closed_form' computes its number as `max_over_temperatures_and_variances(abs(var_reported-var_closed_form)/var_closed_form)`, which cannot be read (it calls max_over_temperatures_and_variances(), which is not one of abs, acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, hypot, ln, log, log10, log2, max, min, sin, sinh, sqrt, sum, tan, tanh): write `measure` with numbers, the names the simulation function returns, + - * / ** %, parentheses and the functions abs, acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, hypot, ln, log, log10, log2, max, min, sin, sinh, sqrt, sum, tan, tanh (and the constant pi).

- Before stopping or repairing, FI looked at the failing known-answer checks itself (at most one model call per check to work its expected value out again, two more runs of a deterministic check's case at smaller steps, three more seeds of a random check, one retry at twice the time limit): this run spent 1 model call(s) and 3 extra run(s) of the simulation on it.

## What the topic asks for


- The plan did not give all of what it was asked for about its simplifications, definitions and steps (the paper will say so in its limitations):
  - `protocol.simplifications` is incomplete: the topic asks for number of time points, which the plan neither includes nor names in `simplifications`; the topic asks for coverage of the true value by the 95 percent interval, which the plan neither includes nor names in `simplifications`; the topic asks for activation energy Ea, which the plan neither includes nor names in `simplifications`; the topic asks for Arrhenius law for the rate constant, which the plan neither includes nor names in `simplifications`
> Read from the topic by another model than the plan's (claude-sonnet-5-5), then looked for in the plan.

- To vary or compare: temperature.
- To vary or compare: noise level.
- To vary or compare: number of time points.
- To report: bias.
- To report: spread of the estimates.
- To report: coverage of the true value by the 95 percent interval.
- To report: rate constant k at each temperature.
- To report: activation energy Ea.
- To report: ln A.
- To report: correlation between Ea and ln A estimates.
- Formula the topic states: `c(t) = c0 * exp(-k * t)`.
- Formula the topic states: `k = A * exp(-Ea / (R * T))`.
- To compare: How bias, spread and 95 percent interval coverage of the recovered k, Ea and ln A change with the noise level.
- To compare: How bias, spread and 95 percent interval coverage of the recovered k, Ea and ln A change with the number of time points.
- Constraint: Simulate noisy concentration-time data at five temperatures spanning a 10-fold range of k, with known A and Ea.
- Constraint: Sampling times run from 0 to 3/k at each temperature.
- Constraint: Noise is additive Gaussian of a stated size.
- Constraint: Fit k at each temperature by nonlinear least squares.
- Constraint: Fit ln k against 1/T with centred 1/T, weighting each point by the variance of its ln k.
- Constraint: One Python script using numpy (and scipy where it helps), with matplotlib (Agg backend).
- Constraint: Two or three figures that answer the question directly, each with the reference or theory value drawn.
- Constraint: A short paper of at most 4 pages.
- Constraint: Cite only sources actually found and read; say so if a reference cannot be confirmed.
- Constraint: CPU only; under 60 seconds of runtime.
- Constraint: Use fixed random seeds where the run is random, and state them in the paper.
- Constraint: Report every number with its uncertainty (for a deterministic result, its numerical error).
- Constraint: Say when a result is only a check of the code.

- FI asked the plan's model to add what was missing; it is in the plan now.

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
  - per-temperature fitted k, k bias, k spread and k coverage (per temperature)
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
  with c0 fixed and the noise sigma treated as known. Propagate each fitted k covariance by Var(ln k)=Var(k)/k^2,
  then use centred weighted least squares for ln k versus 1/T and an unweighted centred comparator fitted
  to the same data. Intervals for both methods are reported in two forms: normal-z from the propagated
  or OLS-residual covariance, and t(3). Each of 25 noise-by-time settings has 400 independent synthetic
  five-temperature experiments; report Monte Carlo means with t-interval standard errors, SDs with chi-square
  intervals, Wilson coverage intervals, per-temperature k bias, spread and coverage, and the Ea--lnA correlation
  with a Fisher-z interval, drawn against the linearised design-matrix correlation at the true k. Failure
  counts are reported per setting. A convergence check on a 20-run subset tightens the solver tolerance
  100-fold and requires k_hat to change by less than 1e-6 relative in every run.'
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
  - id: k_bias_s_inv
    estimand: Mean fitted rate constant minus generating rate constant at each temperature.
    kind: mean
    unit: one fitted temperature-series rate error per run
    cluster: run
    paired: true
    family: five temperatures
  oracles:
  - name: noiseless_decay_recovery
    kind: special_case
    check: Fit the first-stage fitter, initialized at half the stated rate, to five independently constructed
      noiseless E1 concentration series at the generated five rates and measure the largest relative fitted-rate
      error.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-07
    tolerance_mode: absolute
    reference: 'derivation: each independently constructed series has c_i=c0*exp(-k_true*t_i), so its
      least-squares residual is 0 at k_hat=k_true => max_i(abs(k_hat_i-k_true_i)/k_true_i)=0.'
    case:
      noise_sigma_over_c0: 0
      n_time: 9
    measure: max_k_relative_error
  - name: generated_rate_ratio_and_arrhenius
    kind: special_case
    check: Run the temperature construction and measure the ratio of the largest to the smallest generated
      true k, which must be exactly 10.
    expected: 10
    expected_formula: '10'
    tolerance: 1.0e-09
    tolerance_mode: relative
    reference: 'derivation: S1 spaces k geometrically from 0.1 to 1 s^-1 using E2, so max(k)/min(k) =
      1/0.1 = 10.'
    case:
      noise_sigma_over_c0: 0
      n_time: 9
    measure: max(k_true)/min(k_true)
  - name: noiseless_arrhenius_recovery
    kind: special_case
    check: Fit the weighted Arrhenius stage directly to the five exact generated true k values, using
      finite nominal Var(ln k) values computed with noise_sigma_over_c0=0.01 and absolute_sigma=True,
      and measure the relative error of Ea_hat.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: 'derivation: with y_i=ln(k_true_i)=lnA_true-Ea_true/(R*T_i), the centred weighted line
      has b=-Ea_true/R for any finite positive weights => abs(Ea_hat-Ea_true)/Ea_true=0.'
    case:
      noise_sigma_over_c0: 0.01
      n_time: 9
    measure: abs(Ea_hat - Ea_true) / Ea_true
  - name: noiseless_lnA_recovery
    kind: special_case
    check: Fit the weighted Arrhenius stage directly to the five exact generated true k values, using
      finite nominal Var(ln k) values computed with noise_sigma_over_c0=0.01 and absolute_sigma=True,
      and measure the relative error of lnA_hat; this tests the intercept map and centring in E3.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: 'derivation: noiseless k lie on the line ln(k)=lnA_true-Ea_true/(R*T), so alpha-b*xbar
      = lnA_true for finite positive weights => abs(lnA_hat-lnA_true)/lnA_true=0.'
    case:
      noise_sigma_over_c0: 0.01
      n_time: 9
    measure: abs(lnA_hat - lnA_true) / lnA_true
  - name: first_stage_variance_closed_form
    kind: second_implementation
    check: For all five temperatures, fit independently constructed noiseless concentration series from
      an initial rate of half the true rate, use nominal sigma=0.01*c0 with absolute_sigma=True, and compare
      the fitted-k Jacobian Var(k) and propagated Var(ln k) with separately implemented closed forms;
      measure the largest relative difference over temperature and variance type.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-06
    tolerance_mode: absolute
    reference: 'second implementation: evaluate Var(k)=sigma^2/sum((t*c0*exp(-k_hat*t))^2) and Var(ln
      k)=Var(k)/k_hat^2 in a test-only routine sharing no covariance code with the simulation; the one-parameter
      E1 Jacobian is J=-t*c0*exp(-k_hat*t), so (J''J)^-1*sigma^2 = sigma^2/sum((t*c0*exp(-k_hat*t))^2)
      => both relative differences are 0.'
    case:
      noise_sigma_over_c0: 0.01
      n_time: 9
    measure: max_over_temperatures_and_variances(abs(var_reported-var_closed_form)/var_closed_form)
  - name: arrhenius_example_rate
    kind: special_case
    check: Evaluate the E2 implementation at the stated E2 example inputs and measure its relative difference
      from the independently evaluated numerical reference rate.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-09
    tolerance_mode: absolute
    reference: 'derivation: E2 gives k=100000000*exp(-50000/(8.314462618*350))=3.448570123e0 s^-1, so
      abs(k_E2-3.448570123)/3.448570123=0 at the stated comparison precision.'
    case:
      noise_sigma_over_c0: 0
      n_time: 9
    measure: abs(k_E2_example - 3.448570123) / 3.448570123
  criteria:
  - name: noiseless_rate_accuracy
    what: Noiseless nonlinear fitting must recover all five independently constructed first-order rates
      when started away from the solution.
    oracle: noiseless_decay_recovery
    use: value
    direction: lower
    target: 0
    tolerance: 1.0e-07
  simplifications:
  - id: X1
    replaces: Literature-sourced formal definitions for bias, spread, and coverage.
    because: The supplied sources motivate uncertainty analysis but do not quote operational definitions
      for these Monte Carlo summaries.
    effect: Definitions are standard simulation estimands and do not alter generated data or fits.
    probe: Recompute summaries directly from saved run-level estimates and interval indicators.
  - id: X2
    replaces: A complete experimental uncertainty model.
    because: The requested study specifies additive Gaussian concentration noise only.
    effect: Coverage may differ under multiplicative, correlated, or systematic error.
    probe: A later sensitivity run can replace additive noise with multiplicative or correlated noise.
  - id: X3
    replaces: Fitting c0 together with k.
    because: The study fixes c0 as known so that each series has one free parameter.
    effect: k intervals are narrower than if c0 were also estimated, so coverage is a best case.
    probe: Refit a subset with c0 free and compare k spread.
  - id: X4
    replaces: A mis-specified analysis model.
    because: Data are generated from the same first-order decay law and Arrhenius law used by the analysis.
    effect: Results show calibration under the assumed first-order Arrhenius model only; model error is
      not tested.
    probe: Generate a subset with a perturbed decay or temperature-rate model and assess the resulting
      bias and coverage.
  - id: X5
    replaces: Exact nonlinear confidence intervals.
    because: Jacobian linearisation and Var(ln k)=Var(k)/k^2 give nominal intervals.
    effect: Coverage at few points and high noise may deviate from 95% because of the linearisation.
    probe: Compare reported interval widths with empirical Monte Carlo SDs of the estimates in each cell.
  - id: X6
    replaces: Uncertainty, error, or design variation in temperature.
    because: The five temperatures are constructed deterministically from known Arrhenius parameters to
      span the specified 10-fold k range, and temperatures are treated as measured without error.
    effect: Ea and ln A recovery can be more accurate and better calibrated than with temperature uncertainty
      or a differently spaced temperature design.
    probe: Repeat selected settings with perturbed temperatures or an alternative temperature range and
      compare bias, spread, and coverage.
  - id: X7
    replaces: A continuous or optimised choice of the number and placement of concentration-time observations.
    because: The study considers only five specified counts of equally spaced times from 0 to 3/k.
    effect: Results apply only to these sampling counts and this scaled uniform schedule; different time
      placements may change first-stage k precision.
    probe: Compare selected grid cells against a denser schedule or a schedule concentrated near the most
      informative decay times.
  - id: X8
    replaces: Exact frequentist coverage of an interval procedure with all uncertainty sources accounted
      for.
    because: Coverage is evaluated by finite Monte Carlo counts for nominal covariance-based 95% intervals,
      conditional on successful fits.
    effect: Reported coverage has Monte Carlo uncertainty and may not represent unconditional coverage
      or coverage under a different interval construction.
    probe: Increase the number of simulated runs and include failed fits as misses to assess Monte Carlo
      error and conditionality.
  - id: X9
    replaces: Recovery across a range of activation energies.
    because: Ea is fixed at 50000 J mol^-1 when generating temperatures and data.
    effect: Bias, spread, coverage, and Ea--ln A correlation are local to this activation energy and corresponding
      temperature range.
    probe: Repeat the grid at alternative Ea values while retaining the stated k range and compare the
      calibration summaries.
  - id: X10
    replaces: Testing whether the Arrhenius law itself is valid for the generated reaction rates.
    because: The known rate constants are generated exactly from k=A*exp(-Ea/(R*T)) and then fitted with
      its log-linear form.
    effect: The study measures parameter recovery conditional on a correct Arrhenius relationship and
      cannot detect non-Arrhenius temperature dependence.
    probe: Generate a subset with a non-Arrhenius temperature dependence and quantify resulting Ea and
      ln A bias and coverage.
  definitions:
  - metric: k_hat_s_inv
    defined_as: The positive nonlinear-least-squares estimate of the first-order rate constant k in s^-1.
    source: '[19]'
    quote: ''
    differs: X1
  - metric: ea_error_J_mol
    defined_as: Ea_hat minus the known generating Ea, aggregated across independent successful runs.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: lna_error
    defined_as: lnA_hat minus the known generating lnA, aggregated across independent successful runs.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: ea_coverage_95
    defined_as: Fraction of successful runs whose nominal 95% Ea interval contains the generating Ea.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: lna_coverage_95
    defined_as: Fraction of successful runs whose nominal 95% lnA interval contains the generating lnA.
    source: '[20]'
    quote: ''
    differs: X1
  - metric: k_coverage_95
    defined_as: Fraction of successful temperature-series fits whose nominal 95% k interval contains the
      generating k.
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
  - metric: k_bias_s_inv
    defined_as: Mean fitted rate constant minus the generating rate constant at each temperature, aggregated
      across independent successful runs.
    source: '[20]'
    quote: ''
    differs: X1
  topic_map:
  - item: c(t) = c0 * exp(-k * t)
    by: equation:E1
  - item: k = A * exp(-Ea / (R * T))
    by: equation:E2
  - item: five temperatures spanning a 10-fold range of k
    by: setting:temperature_K
  - item: additive Gaussian noise of a stated size
    by: setting:noise_sigma_over_c0
  - item: number of time points
    by: setting:n_time
  - item: reaction rate constant k bias
    by: metric:k_bias_s_inv
  - item: reaction rate constant k spread
    by: metric:k_spread_s_inv
  - item: reaction rate constant k 95 percent interval inclusion
    by: metric:k_coverage_95
  - item: activation energy Ea bias
    by: metric:ea_error_J_mol
  - item: activation energy Ea spread
    by: metric:ea_spread_J_mol
  - item: activation energy Ea 95 percent interval inclusion
    by: metric:ea_coverage_95
  - item: ln A bias
    by: metric:lna_error
  - item: ln A spread
    by: metric:lna_spread
  - item: ln A 95 percent interval inclusion
    by: metric:lna_coverage_95
  - item: Ea and ln A estimates are strongly correlated
    by: metric:ea_lna_correlation
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
    temperature_K:
    - 290.186778
    - 298.477829
    - 307.256589
    - 316.567394
    - 326.460126
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
