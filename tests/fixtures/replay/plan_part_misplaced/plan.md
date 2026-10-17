# Plan: TOPIC. For ArF immersion lithography at low k1, under which illumination sources do one-dimensional lines and spaces keep a usable depth of focus thro

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This study will quantify the ability of free-form sources to eliminate forbidden pitches at k1=0.28 using a scalar ArF immersion model. We will optimize a pixelated source to maximize common DOF and compare its performance against conventional parametric sources across a wide pitch range.

## What the literature says

- **[1]**: Demonstrates that ACO-based freeform sources can satisfy multiple target features with CD within 10% tolerance.
- **[2]**: Identifies the forbidden pitch dip typically occurring between 1.1 to 1.4 lambda/NA.
- **[6]**: Shows that modified illumination, such as double annular, can reduce the depth of the forbidden pitch dip.
- **[10]**: Notes that attenuated PSMs can still exhibit forbidden pitches depending on the illumination.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] Demonstration of ACO-Based Freeform Source for ArF Laser Immersion Lithography System (DOI 10.1109/access.2017.2694854)
- [2] A Systematic Study of the Forbidden Pitch in the CD Through-Pitch Curve for Beyond 130nm
- [3] Challenges and solutions of few-line patterns with freeform illumination in immersion lithography (DOI 10.1109/cstic.2018.8369217)
- [4] Highly sensitive and fast scanner focus monitoring method using forbidden pitch pattern (DOI 10.1117/1.3658022)
- [5] Application of illumination pupilgram control method with freeform illumination (DOI 10.1117/12.916594)
- [6] Forbidden pitch improvement using modified illumination in lithography (DOI 10.1116/1.3054286)
- [7] Freeform illumination sources: an experimental study of source-mask optimization for 22-nm SRAM cells (DOI 10.1117/12.846918)
- [8] High sensitive and fast scanner focus monitoring method using forbidden pitch pattern (DOI 10.1117/12.879802)
- [9] Imaging application tools for extremely low-k1 ArF immersion lithography (DOI 10.1117/12.2011382)
- [10] Forbidden pitches for 130-nm lithography and below (DOI 10.1117/12.388951)
- [11] Experimental verification of source-mask optimization and freeform illumination for 22-nm node static random access memory cells (DOI 10.1117/1.3541778)
- [12] Analytical equation predicting the forbidden pattern pitch for phase-shifting mask (DOI 10.1117/12.2010770)
- [13] Illumination pupilgram prediction and control method in advanced optical lithography (DOI 10.1117/12.899032)
- [14] Impact of realistic source shape and flexibility on source mask optimization (DOI 10.1117/1.jmm.13.1.011005)
- [15] Improvement of Rayleigh criterion with duty ratio characterization for subwavelength lithography (DOI 10.1116/1.1689304)
- [16] Placement of Sub-Resolution Assist Features Based on a Genetic Algorithm (DOI 10.1109/access.2019.2926102)
- [17] Optimal SRAF placement for process window enhancement in 65-nm/45-nm technology (DOI 10.1117/12.711480)
- [18] MEEF management and the effect of assist feature optical proximity corrections (DOI 10.1117/12.425205)
- [W1] Demonstration of ACO-Based Freeform Source for ArF Laser Immersion Lithography System | IEEE Journals & Magazine | IEEE Xplore
- [19] A systematic study of the dip in the CD through-pitch curve for low-k1 processes (DOI 10.1117/12.655470)
- [W2] High Reliability ArF Light Source for Double Patterning ...
- [W3] ArF Immersion Lithography: Physics, Process Principles, and ...
- [W4] Illumination modes for ArF immersion lithography. | Download Scientific Diagram
- [20] Dipole options for 90-nm lithography technologies and below (DOI 10.1117/12.485432)
- [W5] Imaging application tools for extremely low-k1 ArF immersion lithography - ADS
- [W6] Immersion lithography - Wikipedia
- [W7] Advances in Light Sources: ArF immersion lithography: Understanding light source performance | Laser Focus World
- [21] Pixel-based simultaneous source and mask optimization for resolution enhancement in optical lithography (DOI 10.1364/oe.17.005783)
- [22] "Simulated annealing" algorithm for light source parametric optimization in photolithography (DOI 10.17586/2226-1494-2017-17-2-242-248)
- [23] Modified Rayleigh criterion for 90 nm lithography technologies and below (DOI 10.1016/j.mee.2003.10.003)
- [24] Fundamental Principles of Optical Lithography (DOI 10.1002/9780470723876)
- [25] Introduction to Fourier optics
- [26] Principles of Optics (DOI 10.1017/9781108769914)
- [27] Mask Design for Optical Microlithography—An Inverse Imaging Problem (DOI 10.1109/tip.2006.891332)
- [28] Pixel-based OPC optimization based on conjugate gradients (DOI 10.1364/oe.19.002165)
- [29] Understanding the forbidden pitch phenomenon and assist feature placement (DOI 10.1117/12.473427)
- [30] Binary mask optimization for inverse lithography with partially coherent illumination (DOI 10.1364/josaa.25.002960)
- [31] Binary and phase shifting mask design for optical lithography (DOI 10.1109/66.136275)

## The gap this experiment addresses

While the existence of forbidden pitches is well-known, there is a lack of quantitative comparison between specifically optimized free-form sources and the best possible parametric alternatives for the extremely low k1=0.28 regime across a continuous pitch spectrum.

## What is being optimised

> Shown from the design block below (`study_type` and `protocol.optimisation`) as it was when the plan was written; edit the block, not this section (after an edit, only the block counts). FI runs the search itself, within the budget below, and records every evaluation; the best design is found at the search's own numerical settings, then FI evaluates it and the baseline again at the finer check settings below, nudges it, and compares the starting points. A check that fails is reported as the result and limits the evidence level; it does not stop the quest.

**Kind of study:** find the best design (`study_type: find_best_design`), not a measurement over settings chosen in advance.

- (the plan does not yet say what to make as low or as high as possible, which settings may change and over what range, and the design to beat: FI asks the plan's model to write that before anything runs. To measure over chosen settings instead, say so with the plan box: “measure over the planned settings instead of searching”)

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** Scalar partial-coherence imaging using Abbe and Hopkins formulations, 193.368 nm vacuum wavelength, water index 1.436, NA 1.35, and a thin 6% attenuated phase-shift mask.

**What it assumes:** Thin-mask Kirchhoff approximation with dark-region complex amplitude -sqrt(0.06); Scalar imaging without polarization; at NA 1.35 this can overstate contrast and DOF; Constant-threshold aerial-image resist without blur or process effects

**Where it holds:** Comparisons within this specified scalar model for the stated one-dimensional patterns, physical pitches 80-2000 nm, source support no greater than normalized radius 0.95, and NA 1.35. It is not validated for vector polarization, three-dimensional mask effects, or resist-process prediction.

**Its equations:**

- **E1** (produces the data): `Phi(f, z) = (2*pi*z/lambda) * (n - sqrt(n**2 - (lambda*f)**2))`. Source: derivation: Defocus phase with z and lambda in metres and f in inverse metres; restrict propagating pupil frequencies to the physical NA.
- **E2** (produces the data): `TCC(f1, f2) = integral( S(delta) * P(f1+delta) * conj(P(f2+delta)) d_delta ) / integral( S(delta) d_delta )`. Source: derivation: The Transmission Cross Coefficient is the overlap of the pupil functions shifted by the source points, normalized by the total source power to ensure a clear mask prints with unit intensity. => TCC = (integral S*P1*P2*)/(integral S)
- **E3** (produces the data): `Mutual coherence = 2*J1(u)/u with u = 2*pi*sigma*NA*d/lambda`. Source: topic
- **E4** (produces the data): `d = 0.61*lambda/(sigma*NA)`. Source: topic
- **E5** (produces the data): `sigma = lambda/(2*p*NA)`. Source: topic
- **E6** (produces the data): `Max frequency = NA*(1+sigma_max)/lambda`. Source: topic

**The checks against known answers, and where each expected value comes from:**

- **abbe_hopkins_agreement**, an independent second implementation: expects 0; computed as `max(abs(I_abbe - I_hopkins)) / max(I_abbe)` from what the simulation returns on `pitches=100, sigma=0.7`; from: shares no code with the simulation: it is implemented using a different summation order and separate TCC logic.
- **uniform_clear_mask**, a conserved quantity or other invariant: expects 0; computed as `abs((max(abs(I_abbe - 1))) - 1)` from what the simulation returns on `mask_amplitude=1, sigma=0.7, defocus_z_nm=100`; from: derivation: for a clear mask P=1, TCC(f1,f2) = integral(S(delta) d_delta) / integral(S(delta) d_delta) = 1. Aerial image I = sum_f TCC(f,f) = 1. => max|I-1| = 0
- **focus_symmetry**, a symmetry or scaling law: expects 0; computed as `max(abs(I_z_plus - I_z_minus)) / max(I_z_plus)` from what the simulation returns on `pitches=120, sigma=0.7, defocus_z_nm=[-100, 100]`; from: derivation: for a centrosymmetric source and real mask, the field E(z) is the complex conjugate of E(-z) => I(z) = |E(z)|^2 = |E(-z)|^2 = I(-z) => max|I(z)-I(-z)| = 0
- **uniform_dark_mask**, a special or limiting case with a known answer: expects 0.06; computed as `sum(I_abbe) / image_pixels` from what the simulation returns on `mask_amplitude=-0.2449489743, sigma=0.7, defocus_z_nm=100`; from: derivation: a constant mask amplitude produces a constant field; intensity is its squared magnitude when source power is normalized. => I = |-sqrt(0.06)|^2 = 0.06
- **analytic_defocus_harmonic**, a special or limiting case with a known answer: expects -0.54398; computed as `harmonic_ratio` from what the simulation returns on `pitches=160, sigma=0, duty_cycle=0.5, clear_amplitude=1, dark_amplitude=-0.2449489743`; from: derivation: Fourier-expand the real equal-width mask; the pupil admits the zero and first orders but blocks higher orders, so the first intensity harmonic is proportional to cos(Phi(1/p,z)). This tests the defocus phase magnitude in E1.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **imaging_consistency**: Abbe and Hopkins returned images agree on the oracle case; analytic and symmetry oracles pass; successive numerical refinements bound the error of reported results.; how far the check “abbe_hopkins_agreement” lands from the value it expects; lower is better; met at 0 or below; a change smaller than 0.01 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/duv_lowk1_source_dof_opc_ambit/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 105 more lines of code than two scripts (30 written by the model, 75 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- The minimum common DOF of the optimized free-form source is significantly higher than that of any circular, annular, or dipole source.

## Risks

- Optimization might converge to a local minimum if the starting points are poor.
- Scalar model may hide some contrast losses that a vector model would reveal.

## What this quest will not do

- Resist blur and chemical diffusion effects.
- Lens aberrations (Zernike polynomials).
- Mask 3D effects.

## Checks already made

- circular_evaluation; hypothesis, method, optimisation.numerical_settings; The four pitches used to optimize the free-form source also define its stated minimum DOF, and the draft does not require an independent numerical check.; A source can exploit the selected pitches or coarse sampling while performing worse between them.; Evaluate the claim on a denser through-pitch sweep and recompute every finalist and comparator with finer numerical settings; confine the claim to the stated scalar model.; high
- weak_baseline; method, optimisation.baseline, optimisation.grid; Circular sigma 0.7 is called a standard design without support, and no tuning procedure is given for the parametric competitors.; An untuned comparator can make the free-form gain appear larger.; Identify sigma 0.7 as a predeclared reference and tune each parametric family under the same DOF objective, dose rule, OPC procedure, and evaluation budget.; high
- stratum_collapse; variables.dependent, figures_planned, protocol.metrics; The draft reduces performance to minimum common DOF without explicitly retaining dense, semi-dense, forbidden-pitch, and isolated results.; An aggregate can hide which pitch class gains or loses process window.; Report DOF and OPC measures by pitch and identify the minimum and its pitch for each source.; medium
- precision; protocol.precision, optimisation.evaluation_budget; A 0.005 µm precision target is stated without a grid, focus, source-sampling, or optimizer stopping study that could reach it.; Ten optimizer starts are search attempts, not evidence of 5 nm numerical precision.; Use successive numerical refinements, report the observed change in every headline result, and make the comparison only when its gain exceeds that error.; high
- estimand; protocol.metrics, protocol.ci_method; Deterministic optical calculations are labeled means with bootstrap uncertainty.; Bootstrap intervals over optimizer starts do not measure numerical error or uncertainty in a deterministic DOF.; Define headline metrics as deterministic values, use refinement differences as numerical error, and report optimization-start variability separately.; high
- threshold_sensitivity; protocol.thresholds, method; The 0.1 nm ambit cutoff and process-window cutoffs can determine the conclusions, but no sensitivity check is specified.; A threshold-dependent ranking or distance may change when a reasonable cutoff changes.; Repeat the ambit calculation at 0.05, 0.1, and 0.2 nm and show process-window rankings at adjacent CD and exposure-latitude cutoffs.; medium
- rng_independence; protocol.seed_policy; One independent stream per setting and run does not specify how streams are derived or how optimization starts are separated.; Reused or undocumented streams make stochastic search results difficult to reproduce.; Derive independent streams from a recorded master seed and stable identifiers for family, candidate, and start.; low
- numerical_convergence; optimisation.numerical_settings, protocol.criteria; Only pupil-grid refinement is specified; source sampling, image sampling, focus step, and mask-width perturbation are unchecked.; Any of these discretizations can shift a forbidden-pitch minimum or a 0.1 nm ambit crossing.; Refine each discretization and require nonnegative intensity, normalized source power, and agreement between Abbe and Hopkins at matched quadrature.; high
- oracle; protocol.oracles; The sole oracle uses a script-reported max_relative_diff and cannot independently establish that either image is correct.; Two implementations can agree while sharing a normalization or pupil error.; Compute agreement from returned images and add a uniform-clear-mask analytic case, a focus-symmetry invariant, and a baseline-source check.; high
- failed_runs; protocol.failure_policy; Failed evaluations are excluded, including potentially difficult pitches or candidate sources.; Exclusion can improve the apparent minimum DOF and bias source selection.; Count failures, retry once at finer settings, and assign unresolved candidate or pitch failures zero usable DOF.; high
- optimisation_baseline; optimisation.design_variables, optimisation.baseline; The baseline specifies source_type and sigma but no value for the only design variable, source_pixels.; It is not an executable point in the free-form design space.; Define its 32×32 pixel values by an exact circular-source formula and identify it as a predeclared engineering reference.; high
- optimisation_ranges; optimisation.design_variables, protocol.model.holds_for; Pixel intensities from 0 to 1 have no stated source-support, power, or manufacturability interpretation, and the model is said to hold broadly at low k1.; The optimizer could choose an unphysical source or claim validity beyond the thin-mask scalar approximation.; Constrain pixels to the illumination pupil with unit total power and symmetry, and state the model's quantitative scope and limitations.; high
- other; method, protocol.model.equations; The dose and exposure-latitude convention is ambiguous, the E1 example mixes z units, and E2 cites an unverified reference.; These details prevent reproducible CD and DOF calculations and risk an unsupported citation.; Define a full 5% exposure window, use SI units consistently, and label E2 as a derivation until a source has been read.; medium
- other; method, expected_outcome, result_assertions; The draft predicts a specific forbidden-pitch interval and positive DOF without allowing for source-dependent minima or unprintable cases.; Those assertions can suppress a valid negative result.; Treat forbidden pitches as measured local minima, allow zero DOF, and make the expected outcome conditional.; medium
- the threshold `sensitivity` was left out of `protocol.thresholds` because it is not one number ('Repeat source ranking at CD tolerances 0.08, 0.10, and 0.12 with full exposure latitudes 0.04, 0.05, and 0.06; report ambit distances at CD-change cutoffs 0.05, 0.10, and 0.20 nm. State when rankings change.'); it is not used
- `protocol.metrics` was left out of the plan because it could not be checked (`protocol.metrics` entry `min_common_dof`: `kind` must be `proportion` or `mean`); FI asks the plan's model to write it before the experiment runs.
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, optimisation, protocol, result_assertions, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gemma4:31b).
- The protocol declares no metric spec (`metrics`), so which estimator each headline number gets (independent trials or clusters, paired or independent settings, a proportion or a mean) would be guessed from the names in RESULT_JSON; declare each headline metric while the plan is a draft. The evidence level does not call the statistics adequate while a metric is undeclared.
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.
- Equation E1 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E2 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- The check 'abbe_hopkins_agreement' compares with a second implementation but does not say what code it does not share with the simulation (write, for example, “shares no code with the simulation: it uses scipy's solve_ivp”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'uniform_clear_mask' takes its expected value from a source this quest did not retrieve (“Analytic constant-field solution.”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'focus_symmetry' takes its expected value from a source this quest did not retrieve (“Complex conjugation symmetry of the stated scalar imaging model.”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'baseline_source' takes its expected value from a source this quest did not retrieve (“Exact baseline construction rule in optimisation.baseline.”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

- FI asked the plan's model to write `protocol.metrics` again, and it is now in the plan.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gemma4:31b). Its summary: The implementation comparison and focus symmetry are useful, but the current checks do not establish the defocus phase magnitude. The TCC equation also needs an explicit source normalization convention.
  - **abbe_hopkins_agreement**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Also compare the implementations at nonzero defocus and with another source shape..
  - **uniform_clear_mask**: tests the model, would catch a plausible bug, its number is well defined, its case may not isolate what it measures. A spatially varying image can have unit mean, so the measure does not establish the claimed uniform intensity.; a better check: Measure max(abs(I_abbe - 1)) and expect zero..
  - **focus_symmetry**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Add an analytic defocus check: this symmetry also holds if the defocus coefficient is wrong or defocus is omitted..
  - **baseline_source**: may not test the model, would miss a plausible bug, its number is well defined, its case isolates what it measures. a better check: Keep source construction validation separate from the simulation oracles and use an analytic image check here..
  - Found (medium): uniform_clear_mask: The mean-intensity measure does not check that the image is uniform (Spatial errors with zero mean leave the measured value unchanged.). Fix: Compare the largest pixelwise deviation from unit intensity with zero.
  - Found (medium): baseline_source: The check reads back source construction rather than testing the simulation (An optical implementation can be wrong while this array comparison passes exactly.). Fix: Move it to input validation and replace this oracle with an analytic image result.
  - Found (high): E1: No declared check tests the magnitude of the defocus phase (Focus symmetry survives a wrong coefficient and even an omitted defocus phase.). Fix: Add an analytic nonzero-defocus image check, such as the first-harmonic check proposed here.
  - Found (high): E2: The TCC equation leaves source-power normalization unspecified (Using raw source weights makes the clear-mask intensity depend on total source weight, contrary to the plan's normalized-intensity check.). Fix: Define S as a unit-integral source density or divide the TCC integral by total source power.
  - Checked, and no problem found: abbe_hopkins_agreement: checked the dimensionless discrepancy, expected zero, and its ability to expose a differing TCC implementation.; focus_symmetry: checked conjugation symmetry for a real mask, centrosymmetric source, and opposite defocus.; E1: checked the physical frequency units, medium-index dependence, and the defocus sign convention stated in the topic..
  - Equations the checks test: E2; not tested by any: E1.
- The model's 2 equations were read by gpt-6-sol, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is not the standard definition: TCC(f1, f2) = integral(S(delta) * P(f1+delta) * conj(P(f2+delta)) d_delta) / integral(S(delta) d_delta), unless S is explicitly defined to have unit integral. (The stated normalized-intensity convention requires division by total source power, but the equation does not define S as normalized.).

- FI asked the plan to change a check: The check 'uniform_clear_mask' computes its number as `mean(I_abbe)`, which cannot be read (it calls mean(), which is not one of abs, acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, hypot, log, log10, log2, max, min, sin, sinh, sqrt, sum, tan, tanh): write `measure` with numbers, the names the simulation function returns, + - * / ** %, parentheses and the functions abs, acos, asin, atan, atan2, ceil, cos, cosh, exp, floor, hypot, log, log10, log2, max, min, sin, sinh, sqrt, sum, tan, tanh (and the constant pi).
- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'abbe_hopkins_agreement' changed: where its expected value comes from changed.
- What changed: the check 'uniform_clear_mask' changed: kind 'special_case' → 'invariant'; measure 'mean(I_abbe)' → 'max(abs(I_abbe - 1))'; where its expected value comes from changed.
- What changed: the check 'focus_symmetry' changed: where its expected value comes from changed.
- What changed: the check 'uniform_dark_mask' was added (expects 0.06, computed as `sum(I_abbe) / image_pixels`).
- What changed: the check 'analytic_defocus_harmonic' was added (expects -0.54398, computed as `harmonic_ratio`).
- What changed: the check 'baseline_source' was removed.
- Equation E2 was changed: it was `TCC(f1, f2) = integral( S(delta) * P(f1+delta) * conj(P(f2+delta)) d_delta )`, it is now `TCC(f1, f2) = integral( S(delta) * P(f1+delta) * conj(P(f2+delta)) d_delta ) / integral( S(delta) d_delta )`.
- After the plan was asked once for a worked example, equation E2 is not tested at all: the plan says it has no closed form for its output (The integral depends on source shape; the uniform-mask and independent-implementation oracles test its implementation.) and no limit, symmetry, conservation or second implementation checks it.
- The check 'uniform_clear_mask' is a conserved quantity or other invariant, so its number is the worst violation and it expects 0. It was written expecting 1 for `max(abs(I_abbe - 1))`; FI rewrote it as `abs((max(abs(I_abbe - 1))) - 1)`, expecting 0 within 0.01. Its verdict is the same as before.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: pitches (80 nm, isolated (2000 nm)).
- To vary or compare: illumination sources (conventional, annular, x-dipole, quasar, free-form pixelated) - NOT in the plan's settings.
- To vary or compare: conventional sigma (0.3 to 0.9).
- To vary or compare: inner/outer sigma pairs - NOT in the plan's settings.
- To vary or compare: pole centre (0.5 to 0.95).
- To vary or compare: pole radius (0.1 to 0.2).
- To vary or compare: neighbour distance (80 to 2000).
- To vary or compare: conventional sigma for ambit (0.1 to 0.95).
- To vary or compare: defocus z.
- To report: depth of focus at 5% exposure latitude - NOT measured in the plan.
- To report: NILS at best focus.
- To report: MEEF.
- To report: mask bias OPC needs.
- To report: iso-dense bias before OPC.
- To report: side-lobe printing.
- To report: forbidden pitches.
- To report: common depth of focus.
- To report: CD change against neighbour distance.
- To report: optical ambit.
- To report: oscillation visibility.
- To report: Abbe and Hopkins aerial-image agreement.
- To report: numerical error.
- To report: runtime.
- Formula the topic states: `2*pi/lambda * z * (n - sqrt(n**2 - (lambda*f)**2))`.
- Formula the topic states: `2*J1(u)/u with u = 2*pi*sigma*NA*d/lambda`.
- Formula the topic states: `d = 0.61*lambda/(sigma*NA)`.
- Formula the topic states: `sigma = lambda/(2*p*NA)`.
- Formula the topic states: `NA*(1+sigma_max)/lambda`.
- To compare: Compare usable depth of focus through pitch across illumination sources.
- To compare: Compare NILS, MEEF, iso-dense bias, required mask bias and side-lobe printing through pitch across sources.
- To compare: Compare aerial images computed by Abbe and Hopkins.
- To compare: Compare the optimized free-form source with the best parametric source.
- To compare: Compare the neighbour-induced CD change and optical ambit as illumination goes from coherent to incoherent.
- To compare: Compare the ambit results across conventional sigma values and the best annular, dipole and free-form sources.
- To compare: Compare the neighbour-induced CD oscillation with the mutual coherence function.
- Constraint: Use exposure at wavelength 193.368 nm through water with refractive index 1.436 and a projection lens of NA 1.35.
- Constraint: Use a thin 6% attenuated phase-shift mask whose dark regions transmit amplitude sqrt(0.06) with a phase of pi under the Kirchhoff thin-mask approximation.
- Constraint: Use scalar imaging with no polarization, and state plainly that at NA 1.35 this model overstates contrast and depth of focus.
- Constraint: Use vertical lines and spaces with a target line width of 40 nm.
- Constraint: Use a constant-threshold aerial-image resist model with no resist blur.
- Constraint: Anchor dose so the densest pitch prints at 40 nm at best focus.
- Constraint: Use one-dimensional OPC to set the mask line width at every pitch so it prints at 40 nm at best focus.
- Constraint: Keep mirror symmetry in the free-form source.
- Constraint: Optimize the free-form source for a representative set containing dense, semi-dense, the worst forbidden pitch and isolated pitches.
- Constraint: Use a CD tolerance of +/-10% of 40 nm and 5% exposure latitude for depth of focus.
- Constraint: Use 0.1 nm as the threshold for the reported ambit distance.
- Constraint: Use Python with numpy and scipy and matplotlib with the Agg backend.

- FI asked the plan's model to add what was missing; it is still missing.

- Still missing (the paper will say it was not studied; the quest stops before anything runs, because the topic's question cannot be answered without it):
  - The topic says to vary or compare illumination sources (conventional, annular, x-dipole, quasar, free-form pixelated), but the plan's settings do not include it.
  - The topic says to vary or compare inner/outer sigma pairs, but the plan's settings do not include it.
  - The topic asks to report depth of focus at 5% exposure latitude, but the plan does not measure it.

## The design (used as written)

```yaml
study_type: find_best_design
hypothesis: Within the specified scalar thin-mask ArF immersion model, a mirror-symmetric pixelated source
  can increase the minimum through-pitch usable DOF relative to the best tuned circular, annular, x-dipole,
  or quasar source. The comparison is accepted only after full pitch-sweep and finer-grid checks.
variables:
  independent:
  - illumination_source_shape and tuned source parameters
  - pattern_pitch from 80 to 2000 nm
  - defocus_z in nm
  - exposure dose within the stated latitude
  - isolated-line neighbour distance from 80 to 2000 nm
  dependent:
  - printed CD in nm by pitch and focus
  - usable DOF in nm by pitch and source, including minimum DOF and its pitch
  - best-focus NILS by pitch
  - MEEF by pitch
  - OPC mask bias in nm by pitch
  - pre-OPC iso-dense CD bias in nm
  - side-lobe peak intensity relative to printing threshold and whether it prints
  - neighbour-induced CD change in nm and optical ambit distance in nm
  controls:
  - vacuum wavelength 193.368 nm
  - water index 1.436
  - NA 1.35
  - thin-mask dark-region amplitude -sqrt(0.06)
  - target printed CD 40 nm
  - source power normalized to one
  - constant-threshold aerial-image resist with no blur
  - one dose-calibration and OPC rule applied to every source family
method: Implement scalar Abbe source-point summation and an independently assembled Hopkins TCC in Python.
  Use the specified pupil defocus phase in SI units and normalize source power before imaging. Sweep circular
  sigma 0.3-0.9, feasible annular inner/outer pairs, x-dipole centre 0.5-0.95 with pole radius 0.1-0.2,
  and symmetric quasar parameters; tune each family to maximize the same minimum usable DOF as the free-form
  search. For each source, anchor nominal dose so the 80 nm pitch prints a 40 nm line at best focus, then
  solve a separate one-dimensional mask-width bias at each pitch for 40 nm best-focus CD. Define 5% exposure
  latitude as the full dose interval from 0.975 to 1.025 times nominal dose; usable DOF is the largest
  contiguous focus interval containing best focus for which every dose in that interval prints CD from
  36 to 44 nm and no side lobe prints. If a pitch cannot meet the nominal target or the interval is empty,
  its DOF is zero. Calculate NILS at the target edge, MEEF by centred mask-width perturbations of 1 and
  0.5 nm, pre-OPC iso-dense bias, OPC bias, and maximum side-lobe intensity relative to threshold. Start
  with the listed pitches, insert additional pitches at 10 nm spacing from 80 to 400 nm and around every
  detected local DOF minimum, and refine minima to 5 nm pitch resolution. Report all local forbidden-pitch
  minima rather than assuming their location. Optimize a 32×32 mirror-symmetric, nonnegative, unit-power
  free-form source on representative dense, semi-dense, current worst-dip, and isolated pitches; select
  finalists by full through-pitch minimum DOF, then recompute finalists and all parametric comparators
  with finer pupil, source, image, and focus sampling. For the ambit study, move one neighbouring line
  from 80 to 2000 nm, report signed CD change, the distance after which all farther sampled changes remain
  below each stated cutoff, and visible oscillations only when their amplitude exceeds twice the measured
  numerical CD error. Overlay the circular-source mutual-coherence zeros as a comparison, without treating
  coherence zero as a guaranteed CD zero. Record CPU runtime.
expected_outcome: The study will report whether the free-form source improves the checked minimum through-pitch
  DOF within this scalar model, where each source's forbidden pitches lie, how OPC measures and side lobes
  vary, and how the neighbour effect decays. Scalar imaging at NA 1.35 omits polarization and can overstate
  contrast and DOF, so its absolute process windows are not predictions of a full vector or resist process.
figures_planned:
- 'dof_vs_pitch.png: checked DOF in nm versus pitch for each tuned source, with numerical-error bars and
  forbidden-pitch minima marked'
- 'opc_features.png: pitch-resolved NILS, MEEF, OPC mask bias, pre-OPC iso-dense bias, and side-lobe threshold
  ratio'
- 'source_comparison.png: optimized free-form source beside the best tuned parametric source, with checked
  minimum DOF and its limiting pitch'
- 'optical_ambit.png: signed neighbour-induced CD change versus distance for circular sigma 0.1-0.95 and
  the best annular, dipole, and free-form sources; show circular-source coherence zeros and cutoff sensitivity'
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: min_common_dof
  min: 0
  max: 1
  unit: um
  reason: A zero process window is a valid outcome; values above 1 µm warrant a unit or calculation audit
    rather than silent rejection.
- path: nils_dense
  min: 0
  max: 20
  unit: dimensionless
  reason: This is a broad calculation sanity range, not a criterion for printability.
optimisation:
  objective:
    quantity: min_common_dof
    direction: maximise
    unit: um
    meaning: Minimum usable DOF across representative pitches during search; the claimed improvement uses
      the full checked pitch sweep.
  design_variables:
  - name: source_pixels
    low: 0
    high: 1
    unit: dimensionless relative intensity
    kind: continuous 32x32 mirror-symmetric array supported only within normalized source radius 0.95;
      normalize the array to unit total power
  fixed:
    target_cd: 40
    na: 1.35
    lambda: 193.368
  constraints:
  - quantity: source_intensity_max
    limit: <= 1 before unit-power normalization; zero outside normalized radius 0.95, mirror symmetric,
      and positive total power
    unit: dimensionless relative intensity
  baseline:
    values:
      source_pixels: For each of the 32x32 source-cell centres (sx,sy), value 1 if sx^2+sy^2 <= 0.7^2
        and 0 otherwise; normalize the resulting array to unit total power.
    source: Predeclared conventional circular sigma 0.7 engineering reference from the draft; the comparative
      baseline is separately the best tuned parametric source.
  numerical_settings:
    pupil_grid:
      search: 32
      check:
      - 64
      - 128
      finer: larger; also refine source quadrature 32 to 64 to 128 cells per axis, image spacing 1 to
        0.5 to 0.25 nm, focus step 10 to 5 to 2.5 nm, and pitch spacing near minima 10 to 5 nm; recompute
        each finalist and comparator on the same check settings
  evaluation_budget:
    starts: 10
    per_start: 100
    seconds_per_evaluation: 2
  search_method: Global then local search with 10 independently seeded starts; select finalists only after
    full pitch-sweep evaluation, report the best value by start and total runtime.
  grid:
    sigma:
    - 0.3
    - 0.5
    - 0.7
    - 0.9
  improvement_tolerance:
    value: 0.01
    mode: absolute µm on checked full-sweep minimum DOF, with the observed numerical error also exceeded
protocol:
  grid:
    pitches:
    - 80
    - 100
    - 120
    - 140
    - 160
    - 180
    - 200
    - 300
    - 400
    - 600
    - 1000
    - 2000
    sigma_conventional:
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    - 0.95
    annular_sigmas:
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    dipole_center:
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    - 0.95
    dipole_radius:
    - 0.1
    - 0.15
    - 0.2
    neighbour_distance:
    - 80
    - 100
    - 120
    - 140
    - 160
    - 180
    - 200
    - 300
    - 400
    - 600
    - 1000
    - 2000
    sigma_ambit:
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    - 0.95
    annular_inner:
    - 0.1
    - 0.2
    - 0.3
    - 0.4
    - 0.5
    annular_outer:
    - 0.6
    - 0.7
    - 0.8
    - 0.9
    - 0.95
    quasar_inner:
    - 0.1
    - 0.2
    - 0.3
    quasar_outer:
    - 0.6
    - 0.7
    - 0.8
    defocus_z:
    - -200
    - -100
    - 0
    - 100
    - 200
  thresholds:
    cd_tolerance: 0.1
    el_threshold: 0.05
  seed_policy: Use a recorded master seed and independent NumPy SeedSequence child streams keyed by source
    family, candidate index, and optimizer-start index; deterministic optical reevaluations use no random
    draws.
  failure_policy: Count and report every failed, divergent, or nonfinite evaluation by source and pitch;
    retry once at finer settings, then assign unresolved pitch DOF zero for selection and comparisons.
    Never silently exclude failures.
  ci_method: 'No stochastic confidence interval for deterministic optical metrics: report signed differences
    between successive numerical refinements and the largest relevant absolute difference as numerical
    error. Report optimization-start spread separately without interpreting 10 starts as a sampling interval.'
  acceptance:
  - The free-form source's full-sweep minimum usable DOF exceeds that of the best tuned parametric source
    by at least 10% and 0.01 µm after finer checks, and the gain exceeds the combined reported numerical
    error.
  - Report the conclusion as specific to the stated scalar thin-mask, constant-threshold model; report
    a negative or unresolved result if the criterion fails.
  - Show whether the source ranking holds across the stated process-window threshold sensitivity set.
  precision:
    target_half_width: 0.005
    metric: min_common_dof
    reason: Target numerical error at most 0.005 µm, established by successive grid, source-sampling,
      and focus-step refinements; otherwise report the larger observed error and treat gains within it
      as unresolved.
  oracles:
  - name: abbe_hopkins_agreement
    kind: second_implementation
    check: Compare independently returned aerial-image arrays using the same source quadrature.
    expected: 0
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'shares no code with the simulation: it is implemented using a different summation order
      and separate TCC logic.'
    case:
      pitches: 100
      sigma: 0.7
    measure: max(abs(I_abbe - I_hopkins)) / max(I_abbe)
  - name: uniform_clear_mask
    kind: invariant
    check: A uniform unit-amplitude clear mask has unit normalized intensity for any source and focus.
    expected: 0
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: for a clear mask P=1, TCC(f1,f2) = integral(S(delta) d_delta) / integral(S(delta)
      d_delta) = 1. Aerial image I = sum_f TCC(f,f) = 1. => max|I-1| = 0'
    case:
      mask_amplitude: 1
      sigma: 0.7
      defocus_z_nm: 100
    measure: abs((max(abs(I_abbe - 1))) - 1)
  - name: focus_symmetry
    kind: symmetry
    check: For the real mask and centro-symmetric source, images at equal positive and negative defocus
      have the same intensity.
    expected: 0
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: for a centrosymmetric source and real mask, the field E(z) is the complex
      conjugate of E(-z) => I(z) = |E(z)|^2 = |E(-z)|^2 = I(-z) => max|I(z)-I(-z)| = 0'
    case:
      pitches: 120
      sigma: 0.7
      defocus_z_nm:
      - -100
      - 100
    measure: max(abs(I_z_plus - I_z_minus)) / max(I_z_plus)
  - name: uniform_dark_mask
    kind: special_case
    check: A uniform attenuated phase-shift region has intensity equal to its intensity transmission.
    expected: 0.06
    expected_formula: abs(-sqrt(0.06))**2
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: a constant mask amplitude produces a constant field; intensity is its squared
      magnitude when source power is normalized. => I = |-sqrt(0.06)|^2 = 0.06'
    case:
      mask_amplitude: -0.2449489743
      sigma: 0.7
      defocus_z_nm: 100
    measure: sum(I_abbe) / image_pixels
  - name: analytic_defocus_harmonic
    kind: special_case
    check: For a coherent, equal-width binary grating with only its first diffraction orders admitted
      by the pupil, the signed first intensity harmonic changes with defocus by the cosine of their phase
      difference from the zero order.
    expected: -0.54398
    expected_formula: cos((2*pi*100/193.368)*(1.436-sqrt(1.436**2-(193.368/160)**2)))
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: Fourier-expand the real equal-width mask; the pupil admits the zero and first
      orders but blocks higher orders, so the first intensity harmonic is proportional to cos(Phi(1/p,z)).
      This tests the defocus phase magnitude in E1.'
    case:
      pitches: 160
      sigma: 0
      duty_cycle: 0.5
      clear_amplitude: 1
      dark_amplitude: -0.2449489743
    measure: harmonic_ratio
  criteria:
  - name: imaging_consistency
    what: Abbe and Hopkins returned images agree on the oracle case; analytic and symmetry oracles pass;
      successive numerical refinements bound the error of reported results.
    oracle: abbe_hopkins_agreement
    use: error
    direction: lower
    target: 0
    tolerance: 0.01
  metrics:
  - id: min_common_dof
    kind: mean
    estimand: The minimum usable depth of focus across representative pitches during search.
    unit: um
  - id: dof_by_pitch
    kind: mean
    estimand: The usable depth of focus for a specific pitch.
    unit: nm
  - id: nils_by_pitch
    kind: mean
    estimand: The Normalized Image Log Slope at best focus for a specific pitch.
    unit: dimensionless
  - id: meef_by_pitch
    kind: mean
    estimand: The Mask Error Enhancement Factor for a specific pitch.
    unit: dimensionless
  - id: opc_bias_by_pitch
    kind: mean
    estimand: The mask width bias needed to print target CD at best focus for a specific pitch.
    unit: nm
  - id: iso_dense_bias
    kind: mean
    estimand: The CD difference between isolated and dense features before OPC.
    unit: nm
  - id: side_lobe_intensity
    kind: mean
    estimand: The peak side-lobe intensity relative to the printing threshold.
    unit: ratio
  - id: ambit_cd_change
    kind: mean
    estimand: The CD change of an isolated line induced by a neighbour at a given distance.
    unit: nm
  - id: ambit_distance
    kind: mean
    estimand: The distance beyond which neighbour-induced CD change stays below the cutoff.
    unit: nm
  - id: forbidden_pitch_locations
    kind: mean
    estimand: The pitch value at which the local minimum of the through-pitch DOF curve occurs.
    unit: nm
  - id: oscillation_visibility_distance
    kind: mean
    estimand: The distance beyond which neighbour-induced CD oscillations are no longer visible.
    unit: nm
  - id: abbe_hopkins_agreement_error
    kind: mean
    estimand: The maximum relative difference between the Abbe and Hopkins aerial images.
    unit: dimensionless
  - id: numerical_error
    kind: mean
    estimand: The estimated numerical error for a metric based on successive refinements.
    unit: nm
  - id: runtime
    kind: mean
    estimand: The CPU runtime of the experiment.
    unit: seconds
  topic_map:
  - item: illumination sources
    by: variable:illumination_source_shape
  - item: conventional sigma (0.3 to 0.9)
    by: grid:sigma_conventional
  - item: inner/outer sigma pairs
    by: grid:annular_inner
  - item: pole centre (0.5 to 0.95)
    by: grid:dipole_center
  - item: pole radius (0.1 to 0.2)
    by: grid:dipole_radius
  - item: neighbour distance (80 to 2000)
    by: grid:neighbour_distance
  - item: conventional sigma for ambit (0.1 to 0.95)
    by: grid:sigma_ambit
  - item: defocus z
    by: grid:defocus_z
  - item: depth of focus at 5% exposure latitude
    by: metric:dof_by_pitch
  - item: forbidden pitches
    by: metric:forbidden_pitch_locations
  - item: oscillation visibility
    by: metric:oscillation_visibility_distance
  - item: Abbe and Hopkins aerial-image agreement
    by: metric:abbe_hopkins_agreement_error
  - item: numerical error
    by: metric:numerical_error
  - item: runtime
    by: metric:runtime
  - item: mutual coherence formula
    by: equation:E3
  - item: first zero formula
    by: equation:E4
  - item: symmetric dipole formula
    by: equation:E5
  - item: max frequency formula
    by: equation:E6
  model:
    summary: Scalar partial-coherence imaging using Abbe and Hopkins formulations, 193.368 nm vacuum wavelength,
      water index 1.436, NA 1.35, and a thin 6% attenuated phase-shift mask.
    assumptions:
    - Thin-mask Kirchhoff approximation with dark-region complex amplitude -sqrt(0.06)
    - Scalar imaging without polarization; at NA 1.35 this can overstate contrast and DOF
    - Constant-threshold aerial-image resist without blur or process effects
    holds_for: Comparisons within this specified scalar model for the stated one-dimensional patterns,
      physical pitches 80-2000 nm, source support no greater than normalized radius 0.95, and NA 1.35.
      It is not validated for vector polarization, three-dimensional mask effects, or resist-process prediction.
    equations:
    - id: E1
      formula: Phi(f, z) = (2*pi*z/lambda) * (n - sqrt(n**2 - (lambda*f)**2))
      role: generates
      source: derivation
      derivation: Defocus phase with z and lambda in metres and f in inverse metres; restrict propagating
        pupil frequencies to the physical NA.
      example:
        inputs:
          z: 1.0e-07
          lambda: 1.93368e-07
          n: 1.436
          f: 1000000
        expected_formula: (2*pi*1e-7/1.93368e-7)*(1.436-sqrt(1.436**2-(1.93368e-7*1e6)**2))
    - id: E2
      formula: TCC(f1, f2) = integral( S(delta) * P(f1+delta) * conj(P(f2+delta)) d_delta ) / integral(
        S(delta) d_delta )
      role: generates
      source: derivation
      derivation: The Transmission Cross Coefficient is the overlap of the pupil functions shifted by
        the source points, normalized by the total source power to ensure a clear mask prints with unit
        intensity. => TCC = (integral S*P1*P2*)/(integral S)
      example:
        untestable: The integral depends on source shape; the uniform-mask and independent-implementation
          oracles test its implementation.
    - id: E3
      formula: Mutual coherence = 2*J1(u)/u with u = 2*pi*sigma*NA*d/lambda
      role: generates
      source: topic
    - id: E4
      formula: d = 0.61*lambda/(sigma*NA)
      role: generates
      source: topic
    - id: E5
      formula: sigma = lambda/(2*p*NA)
      role: generates
      source: topic
    - id: E6
      formula: Max frequency = NA*(1+sigma_max)/lambda
      role: generates
      source: topic
```
