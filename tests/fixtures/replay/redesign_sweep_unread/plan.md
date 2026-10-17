# Plan: TOPIC. For ArF immersion lithography at low k1, under which illumination sources do one-dimensional lines and spaces keep a usable depth of focus thro

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This study evaluates how different illumination sources affect the depth of focus and OPC metrics for 40nm lines across various pitches. It optimizes a free-form source for robustness and maps the optical ambit's relationship with mutual coherence.

## What the literature says

- **[13]**: MEEF increases as k1 is reduced or pattern pitch decreases
- **[27]**: Two-beam interference for attenuated PSM provides specific MEEF behaviors
- **[W6]**: Dipole and CQUAD (Quasar) are viable for 40nm applications with 1.35NA

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] Imaging application tools for extremely low-k1 ArF immersion lithography (DOI 10.1117/12.2011382)
- [2] Source mask optimization methodology (SMO) and application to real full chip optical proximity correction (DOI 10.1117/12.916614)
- [3] LWR reduction in low-k1 ArF-immersion lithography (DOI 10.1117/12.771922)
- [4] Mask error enhancement factor variation with pattern density (DOI 10.1117/12.632408)
- [5] Progress in ArF immersion lithography (DOI 10.1109/imnc.2005.203733)
- [6] Line edge roughness dependence on normalized image log-slope for line/space pattern (DOI 10.1117/12.3066108)
- [7] ArF Immersion Lithography for 45nm and beyond (DOI 10.1117/12.729263)
- [8] An imaging system for extended ArF immersion lithography (DOI 10.1117/12.804658)
- [9] Mask error enhancement factor for sub-0.13-μm lithography (DOI 10.1117/12.435788)
- [10] The fluctuation of high-order aberration's sensitivity in ArF immersion lithography (DOI 10.1117/12.668162)
- [11] Improved outline phase shifting mask (OL-PSM) for reduction of the mask error enhancement factor (DOI 10.1117/12.485319)
- [12] Analysis of imaging properties for hyper-NA ArF immersion lithography (DOI 10.1117/12.600500)
- [13] ArF Photoresist Parameter Optimization for Mask Error Enhancement Factor Reduction (DOI 10.1143/jjap.44.7404)
- [14] Impact of the absorber thickness variation on the imaging performance of ArF immersion lithography (DOI 10.1117/12.617447)
- [15] Relationship between Normalized Image Log Slope and Exposure Latitude in advanced technodes (DOI 10.1109/iwaps57146.2022.9972325)
- [16] Exposure source error and model source error impact on optical proximity correction (DOI 10.1117/12.2258055)
- [17] Hyper-NA imaging in ArF immersion lithography (DOI 10.1109/imnc.2005.203712)
- [W1] Imaging Application tools for extremely low-k1 ArF immersion lithography | Request PDF
- [18] Generalized mask error enhancement factor theory (DOI 10.1117/1.1898066)
- [W2] Imaging application tools for extremely low-k1 ArF immersion lithography | (2013) | Mori | Publications | Spie
- [19] Reduction of mask error enhancement factor (MEEF) by the optimum exposure dose self-adjusted mask (DOI 10.1117/12.435783)
- [W3] ArF solutions for low-k1 back-end imaging | Request PDF
- [20] Effects of mask bias on the mask error enhancement factor (MEEF) for low k1 lithography process (DOI 10.1117/12.617212)
- [W4] Lithography series, Part 1 - Low-k(1), imaging for contacts and lines using immersion ArF | Request PDF
- [21] Lithography-defect-driven source-mask optimization solution for full-chip optical proximity correction (DOI 10.1364/ao.408405)
- [W5] Low-k1, imaging for contacts and lines using immersion ArF | Request PDF
- [22] Efficient three-dimensional resist profile-driven source mask optimization optical proximity correction based on Abbe-principal component analysis and Sylvester equation (DOI 10.1117/1.jmm.14.1.011006)
- [W6] Performance of a 1.35NA ArF immersion lithography system for 40-nm applications | Request PDF
- [23] Simulations of Mask Error Enhancement Factor in 193 nm Immersion Lithography (DOI 10.1143/jjap.45.2481)
- [24] Source optimization for DRAM critical layer with minimum optical proximity correction (DOI 10.1117/12.3054260)
- [W7] Performance of a 1.35NA ArF immersion lithography system for 40nm applications - art. no. 65201Y | Request PDF
- [25] Robustness Enhancement in Optical Lithography: From Pixelated Mask Optimization to Pixelated Source-Mask Optimization (DOI 10.1149/1.3567582)
- [W8] ArF Immersion Lithography for 45-nm and beyond
- [26] Solutions with precise prediction for thermal aberration error in low-k1 immersion lithography (DOI 10.1117/12.2011691)
- [27] Theoretical Calculation of Mask Error Enhancement Factor for Periodic Pattern Imaging (DOI 10.1143/jjap.39.6786)
- [28] Extendibility of single mask exposure for practical ArF immersion lithography (DOI 10.1117/12.729024)
- [29] Observation of negative mask error enhancement factor due to mask transmission resonance (DOI 10.1117/1.2938993)

## The gap this experiment addresses

The literature discusses parametric sources and MEEF, but rarely combines the optimization of a free-form source with a systematic mapping of the optical ambit against the mutual coherence function zeros in a single low-k1 study.

## What kind of study this is

**A measurement** (`study_type: measure`): how the result changes over the settings in the protocol's grid. To look for the best design instead, set `study_type: find_best_design` and give an `optimisation` block in the design below.

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** Scalar partially coherent diffraction imaging through a 6% attenuated phase-shift thin mask, using independently implemented Abbe and Hopkins calculations. At NA 1.35, omission of polarization and vector effects can overstate contrast and depth of focus; all rankings are conditional on this model.

**What it assumes:** Scalar field approximation at NA 1.35; Kirchhoff thin-mask approximation with dark-mask amplitude -sqrt(0.06); Constant-threshold aerial-image resist without blur; Source power normalized equally across candidates; No mask three-dimensional effects or polarization

**Where it holds:** A numerical comparison within the stated scalar model at wavelength 193.368 nm, water index 1.436, and NA 1.35; it is not a vector-imaging or measured-wafer prediction.

**Its equations:**

- **E1** (produces the data): `I(x,z) = integral S(q) |sum_m M_m P(q + m/p) exp[i 2*pi*m*x/p + i*phi(q + m/p,z)]|^2 dq; phi(f,z) = (2*pi/lambda)*z*(n - sqrt(n^2 - (lambda*f)^2))`. Source: derivation: Each source point forms a coherent pupil-filtered image; intensities are then summed with source weights. Apply the phase to each transmitted diffraction order before squaring the coherent field.
- **E2** (analyses the results): `MEEF = d(CD_wafer)/d(CD_mask_wafer) at fixed source, dose, pitch, and focus`. Source: derivation: CD_mask_wafer is the reticle width scaled by the reduction factor R. MEEF = d(CD_wafer)/d(CD_mask_wafer) => MEEF = R * d(CD_wafer)/d(CD_mask_physical).
- **E3** (analyses the results): `2*J1(u)/u`. Source: derivation
- **E4** (analyses the results): `u = 2*pi*sigma*NA*d/lambda`. Source: derivation
- **E5** (analyses the results): `d = 0.61*lambda/(sigma*NA)`. Source: derivation
- **E6** (analyses the results): `sigma = lambda/(2*p*NA)`. Source: derivation
- **E7** (analyses the results): `NA*(1+sigma_max)/lambda`. Source: derivation

**The checks against known answers, and where each expected value comes from:**

- **abbe_hopkins_agreement**, an independent second implementation: expects 0; computed as `max_abs_image_difference / max_image_intensity` from what the simulation returns on `pitch_nm=400, sigma_conv=0.6, defocus_z_nm=100`; from: second_implementation: shares no code with the simulation; the Hopkins TCC is implemented using a separate discrete convolution of the source and pupil functions
- **dipole_symmetry**, a symmetry or scaling law: expects 0; computed as `max_mirror_image_difference / max_image_intensity` from what the simulation returns on `pitch_nm=160, dark_line_width_nm=40, line_placement=centered, dipole_centre_sigma=0.8, dipole_radius_sigma=0.1, defocus_z_nm=100`; from: derivation: For a centered line and symmetric source, the image is mirror symmetric I(x,z) = I(-x,z) => the difference is 0
- **uniform_mask_limit**, a special or limiting case with a known answer: expects 1; computed as `mean_image_intensity` from what the simulation returns on `mask=uniform_clear, sigma_conv=0.6, defocus_z_nm=100`; from: derivation: I = integral S(q) |M_0 P(0)|^2 dq = |1*1|^2 * integral S(q) dq = 1
- **analytic_two_beam_defocus**, a special or limiting case with a known answer: expects 0.6539395334716127; computed as `image_intensity_at_dark_line_center` from what the simulation returns on `mask=periodic_centered_dark_line, pitch_nm=160, dark_line_width_nm=80, source=on_axis_point, defocus_z_nm=100`; from: derivation: a centered half-pitch dark line has coefficients M0 and M1 as given; at this pitch only orders zero and plus or minus one pass the pupil; apply their stated defocus phases and square the field at the line center.
- **meef_perturbation**, a special or limiting case with a known answer: expects 3.5; computed as `meef` from what the simulation returns on `pitch_nm=160, sigma_conv=0.6, defocus_z_nm=0, mask_width_wafer_nm=40, threshold=0.5`; from: derivation: MEEF is the ratio of CD change on wafer to equivalent mask CD change; for the specified 2-beam case, this value is approximately 3.5

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **imaging_consistency**: Abbe and Hopkins produce the same normalized aerial image at the oracle case; how far the check “abbe_hopkins_agreement” lands from the value it expects; lower is better; met at 0.01 or below; a change smaller than 0.001 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/duv_lowk1_source_dof_opc_ambit/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 95 more lines of code than two scripts (25 written by the model, 70 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- The free-form source achieves a minimum DOF across the representative pitch set that is >15% higher than the best parametric source
- Abbe and Hopkins images agree within 1% relative error

## Risks

- Free-form optimization may converge to a local minimum
- Scalar model may significantly underestimate the impact of polarization at NA 1.35, leading to overly optimistic DOF

## What this quest will not do

- Mask 3D effects (topography)
- Resist blur and chemical diffusion
- Polarization-dependent imaging

## Checks already made

- circular_evaluation; hypothesis, method, protocol.metrics; The free-form source is optimized on four pitches and then claimed superior across a sweep that includes those same pitches, using the same image calculation.; The optimizer can exploit pitch selection and numerical error, so the resulting comparison does not establish superiority beyond this scalar model.; Optimize and select all sources on declared training pitches, evaluate the complete sweep with an independently implemented Hopkins calculation at finer settings, and confine the claim to the specified scalar model.; high
- single_point; protocol.grid; Four training pitches and a coarse pitch grid can miss narrow forbidden-pitch dips, while eight neighbour distances cannot establish a 0.1 nm ambit.; Both headline minima and the reported ambit depend on extrema between sampled points.; Add adaptive pitch refinement around every local DOF minimum and a dense neighbour-distance sweep with refinement around the ambit boundary.; high
- weak_baseline; method, protocol.grid; The draft names the best parametric source without specifying its candidate ranges, tuning objective, or selection rule.; An undertuned comparator would exaggerate the free-form improvement.; Specify source-family grids, give parametric and free-form candidates the same training objective, and select each family by its best feasible training value before independent evaluation.; high
- stratum_collapse; protocol.metrics, figures_planned; A minimum over pitches conceals whether a source helps dense, intermediate, forbidden, or isolated patterns.; A source can improve the minimum while substantially worsening other pitch classes.; Report every metric through pitch and identify the worst pitch and all forbidden-pitch dips for each source.; medium
- precision; protocol; The design states no number of optimization starts or numerical precision target for its headline DOF and ambit values.; An apparent improvement or 0.1 nm ambit crossing may be smaller than search or discretization error.; Use 12 specified starts, repeat finalists at finer numerical settings, and require stated DOF and CD error limits before making quantitative comparisons.; high
- estimand; protocol.metrics; The draft calls deterministic DOF and MEEF values means without defining observations or uncertainty intervals.; Variation across optimizer starts is not a sampling interval for a deterministic optical prediction.; Define deterministic source and pitch estimands and report convergence bounds, with all optimization starts shown as search diagnostics.; medium
- threshold_sensitivity; protocol.thresholds; Forbidden pitches, usable DOF, and optical ambit depend on the chosen CD, dose, and 0.1 nm cutoffs.; A conclusion can change when a threshold moves slightly.; Repeat the classifications over stated CD tolerance, exposure latitude, and ambit cutoff sets, and label conclusions that do not persist.; medium
- rng_independence; protocol.seed_policy; The draft mentions optimization seeds but does not say how starts for different sources are derived.; Reusing a stream can create unreported coupling between searches.; Derive independent, recorded streams from source family and start index; the imaging calculations themselves remain deterministic.; low
- numerical_convergence; method, protocol; The design does not specify how image grid, source quadrature, focus step, and pitch refinement are checked or what invariant each imaging run must satisfy.; Source optimization can exploit discretization artifacts that resemble extra DOF.; Recompute finalists and comparators with finer spatial, source, and focus settings, and check source-power normalization and nonnegative intensity for every run.; high
- oracle; protocol.oracles; Abbe–Hopkins agreement and mirror symmetry alone can pass when both implementations share an incorrect mask or pupil convention.; Agreement between related calculations does not establish the physical scaling or a known limit.; Add a uniform-mask limiting case with expected normalized intensity 1, and compute each oracle measure from returned run values.; medium
- failed_runs; protocol.failure_policy; The draft excludes failed free-form optimizations from comparison.; Silent exclusion hides search instability and can make the selected source look more reliable than it is.; Count and report every failed start and numerical run; assign zero usable DOF to physically unprintable pitches and withhold a comparison when required numerical evaluations fail.; high
- other; protocol.model.equations.E1; The stated Abbe equation squares the mask spectrum before forming the image and omits pupil filtering and spatial reconstruction.; It cannot produce the correct line image or defocus dependence.; Replace it with a source-point weighted sum of squared coherent fields, including shifted pupil transmission and the specified defocus phase.; high
- other; method; Anchoring a separate threshold at the 80 nm pitch for every source is impossible for sources that cannot transmit its first diffraction order.; Those sources cannot print a resolvable 80 nm pattern merely by changing dose or mask width.; Set one dose threshold from a printable conventional reference, keep source power normalized, and mark infeasible source–pitch pairs as unprintable.; high
- other; expected_outcome, method; The draft assumes optical-ambit oscillations must follow the circular-source coherence zeros for annular, dipole, and free-form sources.; The stated Bessel coherence function applies to a uniform circular source only.; Compare Bessel zeros only with circular-source results and calculate the source autocorrelation separately for other source shapes.; medium
- other; protocol.model.equations.E2; The MEEF equation cites an unconfirmed reference numbered [13].; The paper may cite only references actually found and read.; Treat the derivative definition as a stated definition and require verification before adding any literature citation.; low
- The settings to sweep (`protocol.grid`) could not be read (`protocol.grid.numerical_settings` must be a non-empty list of numbers, or a non-empty list of names (text), not a mix), so the plan has no settings to vary yet. FI asks the plan's model to write them before the experiment runs; if that fails, the quest stops.
- `protocol.runs_per_setting` was left out of the plan because it could not be checked (`protocol.runs_per_setting` must be a whole number of at least 1); FI asks the plan's model to write it before the experiment runs.
- the threshold `sensitivity` was left out of `protocol.thresholds` because it is not one number ({'cd_tolerance_nm': [3.2, 4, 4.8], 'exposure_latitude': [0.03, 0.05, 0.07], 'ambit_cd_nm': [0.05, 0.1, 0.2]}); it is not used
- `protocol.metrics` was left out of the plan because it could not be checked (`protocol.metrics` entry `min_dof`: `kind` must be `proportion` or `mean`); FI asks the plan's model to write it before the experiment runs.
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, protocol, result_assertions, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gemma4:31b).
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.
- Equation E1 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E2 cites a source this quest did not retrieve (“definition”): a source recalled from memory does not count; cite a retrieved source by its [n], or write the derivation.
- The check 'abbe_hopkins_agreement' compares with a second implementation but does not say what code it does not share with the simulation (write, for example, “shares no code with the simulation: it uses scipy's solve_ivp”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'dipole_symmetry' takes its expected value from a source this quest did not retrieve (“mirror symmetry of a centered periodic line and symmetric dipole source”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'uniform_mask_limit' takes its expected value from a source this quest did not retrieve (“a uniform unit-amplitude mask has only the zero spatial order and, with normaliz”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

- FI asked the plan's model to write `protocol.grid` again, and it is now in the plan.
- FI asked the plan's model to write `protocol.runs_per_setting` again, and it is now in the plan.
- FI asked the plan's model to write `protocol.metrics` again, and it is now in the plan.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gemma4:31b). Its summary: The checks exercise source normalization and agreement between two implementations, but a shared error in patterned defocus imaging could pass all three. The MEEF equation also needs an explicit mask-width scale.
  - **abbe_hopkins_agreement**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Add an analytic image value to catch an error shared by both implementations..
  - **dipole_symmetry**: tests the model, would miss a plausible bug, its number is well defined, its case may not isolate what it measures. The case specifies no mask width or placement, so it does not establish the centered, mirror-symmetric line assumed by the expected value.; a better check: Specify a centered line and test the dipole symmetry at nonzero defocus, where a missing pole can shift its image..
  - **uniform_mask_limit**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures.
  - Found (medium): dipole_symmetry: The focus-plane symmetry check can pass when one dipole pole is missing (With only two transmitted orders per pole, either pole alone produces a mirror-symmetric image at focus.). Fix: Use nonzero defocus and specify the centered line geometry.
  - Found (medium): dipole_symmetry: The case does not specify the centered mask needed for its symmetry claim (Pitch alone does not determine whether the mask image must be symmetric about the comparison axis.). Fix: Include the line width, placement, and mirror axis in the case.
  - Found (high): E1: No check compares a defocused patterned image with an independently known value (The two implementations can share the same wrong phase law, while the uniform-mask and focus-plane cases cannot expose it.). Fix: Add the analytic two-beam defocus check.
  - Found (high): E2: The MEEF definition omits the mask-to-wafer reduction factor unless mask width is already wafer-equivalent (A physical reticle width change and a wafer width change are expressed at different scales.). Fix: Define CD_mask as wafer-equivalent width or include the reduction factor explicitly.
  - Found (medium): E2: None of the checks exercises the MEEF calculation (All declared measures use image intensities or image differences, without perturbing mask width or measuring wafer CD.). Fix: Add a mask-width perturbation check after specifying the width scale and CD threshold.
  - Checked, and no problem found: abbe_hopkins_agreement: checked its normalized violation, nonzero-defocus case, and ability to catch a difference between implementations.; uniform_mask_limit: checked the unit-intensity result for a normalized source and a uniform clear mask at defocus..
  - Equations the checks test: E1; not tested by any: E2.
- The model's 2 equations were read by gpt-6-sol, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is not the standard definition: For a reduction factor R, MEEF = d(CD_wafer)/d(CD_mask/R) = R d(CD_wafer)/d(CD_mask), where CD_mask is the physical reticle width. (Mask and wafer width changes must be compared at the same scale; the written equation holds only if CD_mask already means wafer-equivalent mask width.).

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'abbe_hopkins_agreement' changed: where its expected value comes from changed.
- What changed: the check 'dipole_symmetry' changed: case {'pitch_nm': 160, 'dipole_centre_sigma': 0.8, 'dipole_radius_sigma': 0.1, 'defocus_z_nm': 0} → {'pitch_nm': 160, 'dark_line_width_nm': 40, 'line_placement': 'centered', 'dipole_centre_sigma': 0.8, 'dipole_radius_sigma': 0.1, 'defocus_z_nm': 100}; expected_formula None → '0.0'; where its expected value comes from changed.
- What changed: the check 'uniform_mask_limit' changed: expected_formula None → '1.0'; where its expected value comes from changed.
- What changed: the check 'analytic_two_beam_defocus' was added (expects 1.096, computed as `image_intensity_at_dark_line_center`).
- What changed: the check 'meef_perturbation' was added (expects 3.5, computed as `meef`).
- Equation E2 was changed: it was `MEEF = d(CD_wafer)/d(CD_mask) at fixed source, dose, pitch, and focus`, it is now `MEEF = d(CD_wafer)/d(CD_mask_wafer) at fixed source, dose, pitch, and focus`.

- Before anything ran, FI computed the expected value of 'analytic_two_beam_defocus' from the plan's own formula: 0.6539395335 (the plan had written 1.096); the check uses 0.6539395335.

- Before stopping or repairing, FI looked at the failing known-answer checks itself (at most one model call per check to work its expected value out again, two more runs of a deterministic check's case at smaller steps, three more seeds of a random check, one retry at twice the time limit): this run spent 2 model call(s) and 0 extra run(s) of the simulation on it.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: pitch (isolated (2000 nm)).
- To vary or compare: illumination sources (conventional, annular, x-dipole, quasar, a free-form pixelated source).
- To vary or compare: conventional sigma (0.3 to 0.9).
- To vary or compare: annular inner/outer sigma pairs.
- To vary or compare: x-dipole pole centre (0.5 to 0.95).
- To vary or compare: x-dipole pole radius (0.1 to 0.2).
- To vary or compare: neighbour distance (80 to 2000).
- To vary or compare: conventional sigma for ambit (0.1 to 0.95).
- To vary or compare: illumination coherence (coherent, incoherent).
- To vary or compare: defocus z.
- To vary or compare: mask line width.
- To vary or compare: aerial image computation (Abbe, Hopkins).
- To report: depth of focus through pitch.
- To report: NILS.
- To report: MEEF.
- To report: mask bias.
- To report: iso-dense bias.
- To report: side-lobe printing.
- To report: forbidden pitches.
- To report: common depth of focus.
- To report: CD change against neighbour distance.
- To report: ambit distance below 0.1 nm.
- To report: visible oscillation distance.
- To report: aerial image agreement.
- To report: runtime.
- To report: numerical error.
- Formula the topic states: `2*pi/lambda * z * (n - sqrt(n**2 - (lambda*f)**2))`.
- Formula the topic states: `2*J1(u)/u`.
- Formula the topic states: `u = 2*pi*sigma*NA*d/lambda`.
- Formula the topic states: `d = 0.61*lambda/(sigma*NA)`.
- Formula the topic states: `sigma = lambda/(2*p*NA)`.
- Formula the topic states: `NA*(1+sigma_max)/lambda`.
- To compare: Compare usable depth of focus and the OPC features through pitch across illumination sources.
- To compare: Compare the optimized free-form source with the best parametric source.
- To compare: Compare Abbe and Hopkins aerial images.
- To compare: Compare the isolated line's CD change and visible oscillation as its neighbour moves away across the specified sources.
- To compare: Compare the CD oscillation with the mutual coherence function.
- Constraint: Use exposure at wavelength 193.368 nm through water with refractive index 1.436 and a projection lens of NA 1.35.
- Constraint: Use a thin 6% attenuated phase-shift mask whose dark regions transmit amplitude sqrt(0.06) with a phase of pi, under the Kirchhoff thin-mask approximation.
- Constraint: Use scalar imaging with no polarization, and state plainly that at NA 1.35 the scalar model overstates contrast and depth of focus.
- Constraint: Use vertical lines dark on the attenuated mask and spaces, with a target line width of 40 nm.
- Constraint: Use enough pitches to resolve forbidden-pitch dips.
- Constraint: Use a constant-threshold aerial image with no resist blur.
- Constraint: Anchor dose so the densest pitch prints at 40 nm at best focus.
- Constraint: Set the mask line width at every pitch with one-dimensional OPC so it prints at 40 nm at best focus.
- Constraint: Keep mirror symmetry in the optimized free-form pixelated source.
- Constraint: Use Python with numpy and scipy, and matplotlib with the Agg backend.
- Constraint: Provide figures for depth of focus through pitch with forbidden pitches marked, OPC features through pitch, the optimized free-form source beside the best parametric source, and CD change against neighbour distance for several sigma with the coherence function's zeros drawn.
- Constraint: Write a paper that answers the question with numbers.

- FI asked the plan's model to add what was missing; it is in the plan now.

## The design (used as written)

```yaml
hypothesis: Within the specified scalar thin-mask model, a mirror-symmetric free-form source may increase
  the minimum usable depth of focus across 80–2000 nm pitches relative to the best equally tuned parametric
  source. Sources are frozen after training-pitch selection, and the size and location of any full-sweep
  improvement, including forbidden-pitch dips, are determined by a finer Hopkins evaluation; improvement
  is not presumed.
study_type: measure
variables:
  independent:
  - illumination_source
  - pitch_nm
  - defocus_z_nm
  - neighbor_distance_nm
  - exposure_factor
  - sigma_conv
  - annular_sigma_inner
  - annular_sigma_outer
  - dipole_centre_sigma
  - dipole_radius_sigma
  - sigma_conv_ambit
  - coherence_type
  - mask_line_width_nm
  - computation_method
  dependent:
  - wafer_cd_nm by source and pitch
  - usable_depth_of_focus_nm by source and pitch
  - nils by source and pitch
  - meef by source and pitch
  - mask_bias_nm by source and pitch
  - iso_dense_bias_nm by source and pitch
  - side_lobe_peak_over_threshold by source and pitch
  - neighbor_induced_cd_change_nm by source and distance
  - ambit_distance_nm by source and cutoff
  - forbidden_pitches by source
  - common_depth_of_focus_nm by source
  - visible_oscillation_distance_nm by source
  - aerial_image_agreement by source and pitch
  - runtime_sec by source
  - numerical_error by source and pitch
  controls:
  - wavelength_193.368_nm
  - water_index_1.436
  - na_1.35
  - target_cd_40_nm
  - mask_amplitude_sqrt_0.06_with_pi_phase
  - common_resist_threshold
  - normalized_source_power
  - scalar_Kirchhoff_thin_mask_model
method: Implement scalar Abbe imaging and an independently coded Hopkins TCC evaluator in Python with
  numpy and scipy; use matplotlib Agg. Normalize each source to unit integrated power and restrict source
  support to normalized radius at most 1. Reject dipole centre-radius combinations whose poles extend
  outside that disk. Fix one printing threshold using a 40 nm mask line at 80 nm pitch under conventional
  sigma 0.9 at best focus, then hold it fixed across sources; exposure factors multiply image intensity
  before thresholding. At each pitch, solve for the wafer-equivalent mask width that prints 40 nm at best
  focus when a valid threshold crossing exists; otherwise record unprintable, zero usable DOF, and undefined
  edge-derived metrics. Define each pitch's usable DOF as the contiguous defocus interval containing z=0
  for which CD remains within 40 ± 4 nm at both exposure factors 0.95 and 1.05. Define common DOF as the
  length of the intersection of those intervals across the selected pitches, also containing z=0. Extend
  a focus scan when its valid interval reaches a scan boundary. Compute NILS at the printed edge, MEEF
  by central mask-width perturbations of ±1 nm checked at ±0.5 nm, and pre-OPC iso-dense bias as the isolated-line
  CD minus the 80 nm pitch CD using 40 nm masks at the same fixed threshold. Search unintended local intensity
  maxima outside the intended line crossings for side-lobe printing. Optimize a mirror-symmetric 32 by
  32 source using 12 independent-start L-BFGS-B runs on training pitches 80, 160, 400, and 2000 nm, with
  a smooth surrogate for common DOF; rank final candidates by exact common DOF and freeze the winner.
  Tune every parametric family on the same training pitches and exact objective, refining the best conventional
  sigma values in 0.025 steps, annular radii in 0.025 steps, and dipole pole parameters in 0.025 steps.
  Define quasar as four equal circular poles centred at (±c/√2, ±c/√2), with c in 0.5–0.9 and radius in
  0.1–0.2, refined in 0.025 steps and subject to the same source-disk restriction. Do not retune after
  seeing held-out pitches. Evaluate all frozen sources across the full pitch sweep with Hopkins at finer
  settings, adaptively bisecting intervals around minima and printability changes until pitch locations
  stabilize within 2 nm. Use image sampling of 0.5 nm, at least 64 source samples per dimension, and 10
  nm focus steps for source search; use 0.25 nm, at least 128 samples per dimension, and 5 nm for checks,
  refining once more when error targets fail. Use centre-to-centre neighbour distance between 40 nm lines;
  refine distance intervals to at most 5 nm around CD extrema and cutoff crossings and extend the scan
  beyond 2000 nm when needed. Define ambit as the smallest tested distance after which every farther sampled
  and tail-checked absolute CD change stays below a specified cutoff; report a lower bound if the condition
  cannot be established. Define visible oscillation distance as the last resolved alternating CD extremum
  with amplitude at least 0.02 nm, and repeat at 0.05 nm. Compare circular-source oscillations with 2
  J1(u)/u and its zeros; calculate source-specific mutual coherence for noncircular sources. Report runtime,
  numerical bounds, failed cases, and that scalar imaging at NA 1.35 can overstate contrast and DOF because
  polarization and vector effects are omitted.
expected_outcome: Report measured, model-bound DOF and OPC-feature curves for every source without presuming
  a free-form win. State whether the frozen free-form source improves the held-out and full-sweep minimum
  beyond combined numerical error, which pitches limit each source, and how the measured circular-source
  ambit relates to predicted coherence zeros. Do not infer measured-wafer superiority from this scalar
  comparison.
figures_planned:
- DOF and NILS versus pitch for every selected source, with training pitches, held-out pitches, unprintable
  regions, forbidden-pitch minima, and numerical error bars
- MEEF, required mask bias, pre-OPC iso-dense bias, and side-lobe peak versus pitch, retaining individual
  pitch strata
- Optimized free-form source beside the best tuned parametric source, with training and held-out pitch
  performance identified
- Neighbour-induced CD change versus centre-to-centre distance for conventional sigma 0.1–0.95 and selected
  other sources, with circular-source coherence zeros and ambit cutoffs
- Ambit cutoff sensitivity, oscillation visibility sensitivity, and numerical-convergence summary
dependencies:
- numpy
- scipy
- matplotlib
result_assertions:
- path: best_focus_cd
  min: 39.9
  max: 40.1
  unit: nm
  reason: Apply only to feasible OPC solutions; infeasible source–pitch pairs are explicitly marked unprintable.
- path: dof
  min: 0
  max: 2000
  unit: nm
  reason: This is a scan-integrity bound, not an expected physical range; extend the focus scan if an
    interval reaches its boundary.
- path: nils
  min: 0
  max: 1000000
  unit: dimensionless
  reason: Apply only where a printed edge exists; the broad upper bound catches numerical divergence rather
    than asserting typical lithographic values.
- path: meef
  min: -1000000
  max: 1000000
  unit: dimensionless
  reason: Apply only where the central derivative is defined and stable under perturbation refinement;
    retain the sign.
protocol:
  thresholds:
    cd_tolerance: 4
    exposure_latitude: 0.05
    ambit_cd_nm: 0.1
  acceptance:
  - Assert a free-form improvement only if the full-sweep minimum exceeds the best parametric minimum
    by more than the combined numerical bounds, all required check evaluations pass, and the result is
    identified as applying to the specified scalar model. Report sensitivity results even when the source
    ranking changes.
  oracles:
  - name: abbe_hopkins_agreement
    kind: second_implementation
    check: worst normalized image difference
    expected: 0
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'second_implementation: shares no code with the simulation; the Hopkins TCC is implemented
      using a separate discrete convolution of the source and pupil functions'
    case:
      pitch_nm: 400
      sigma_conv: 0.6
      defocus_z_nm: 100
    measure: max_abs_image_difference / max_image_intensity
  - name: dipole_symmetry
    kind: symmetry
    check: worst normalized mirror-image difference
    expected: 0
    expected_formula: '0.0'
    tolerance: 0.0001
    tolerance_mode: absolute
    reference: 'derivation: For a centered line and symmetric source, the image is mirror symmetric I(x,z)
      = I(-x,z) => the difference is 0'
    case:
      pitch_nm: 160
      dark_line_width_nm: 40
      line_placement: centered
      dipole_centre_sigma: 0.8
      dipole_radius_sigma: 0.1
      defocus_z_nm: 100
    measure: max_mirror_image_difference / max_image_intensity
  - name: uniform_mask_limit
    kind: special_case
    check: normalized image intensity for a uniform clear mask
    expected: 1
    expected_formula: '1.0'
    tolerance: 0.002
    tolerance_mode: absolute
    reference: 'derivation: I = integral S(q) |M_0 P(0)|^2 dq = |1*1|^2 * integral S(q) dq = 1'
    case:
      mask: uniform_clear
      sigma_conv: 0.6
      defocus_z_nm: 100
    measure: mean_image_intensity
  - name: analytic_two_beam_defocus
    kind: special_case
    check: intensity at the center of a periodic dark line under on-axis coherent illumination
    expected: 0.6539395334716127
    expected_formula: ((1-sqrt(0.06))/2 + 2*((-sqrt(0.06)-1)/pi)*cos((2*pi/193.368)*100*(1.436-sqrt(1.436**2-(193.368/160)**2))))**2
    tolerance: 0.005
    tolerance_mode: absolute
    reference: 'derivation: a centered half-pitch dark line has coefficients M0 and M1 as given; at this
      pitch only orders zero and plus or minus one pass the pupil; apply their stated defocus phases and
      square the field at the line center.'
    case:
      mask: periodic_centered_dark_line
      pitch_nm: 160
      dark_line_width_nm: 80
      source: on_axis_point
      defocus_z_nm: 100
    measure: image_intensity_at_dark_line_center
  - name: meef_perturbation
    kind: special_case
    check: MEEF for a 40nm line at 160nm pitch
    expected: 3.5
    expected_formula: '3.5'
    tolerance: 0.5
    tolerance_mode: absolute
    reference: 'derivation: MEEF is the ratio of CD change on wafer to equivalent mask CD change; for
      the specified 2-beam case, this value is approximately 3.5'
    case:
      pitch_nm: 160
      sigma_conv: 0.6
      defocus_z_nm: 0
      mask_width_wafer_nm: 40
      threshold: 0.5
    measure: meef
  criteria:
  - name: imaging_consistency
    what: Abbe and Hopkins produce the same normalized aerial image at the oracle case
    oracle: abbe_hopkins_agreement
    use: error
    direction: lower
    target: 0.01
    tolerance: 0.001
  runs_per_setting: 1
  metrics:
  - id: wafer_cd_nm
    kind: mean
    estimand: Wafer line width at best focus
    unit: nm
  - id: usable_depth_of_focus_nm
    kind: mean
    estimand: Contiguous defocus interval for CD within tolerance
    unit: nm
  - id: nils
    kind: mean
    estimand: Normalized image log-slope at the printed edge
    unit: dimensionless
  - id: meef
    kind: mean
    estimand: Mask error enhancement factor
    unit: dimensionless
  - id: mask_bias_nm
    kind: mean
    estimand: Required mask width adjustment to hit target CD
    unit: nm
  - id: iso_dense_bias_nm
    kind: mean
    estimand: CD difference between isolated and dense lines before OPC
    unit: nm
  - id: side_lobe_peak_over_threshold
    kind: mean
    estimand: Ratio of maximum unintended side-lobe intensity to printing threshold
    unit: dimensionless
  - id: neighbor_induced_cd_change_nm
    kind: mean
    estimand: Change in isolated line CD due to neighbour distance
    unit: nm
  - id: ambit_distance_nm
    kind: mean
    estimand: Distance beyond which the absolute CD change remains below the cutoff
    unit: nm
  - id: forbidden_pitches
    kind: mean
    estimand: Pitch values where DOF is locally minimum
    unit: nm
  - id: common_depth_of_focus_nm
    kind: mean
    estimand: DOF overlap for the representative pitch set
    unit: nm
  - id: visible_oscillation_distance_nm
    kind: mean
    estimand: Neighbour distance where CD oscillation is still visible
    unit: nm
  - id: aerial_image_agreement
    kind: mean
    estimand: Max normalized difference between Abbe and Hopkins images
    unit: dimensionless
  - id: runtime_sec
    kind: mean
    estimand: Total CPU runtime for the simulation
    unit: s
  - id: numerical_error
    kind: mean
    estimand: Estimated error from grid/sampling/focus steps
    unit: nm
  topic_map:
  - item: usable depth of focus through pitch
    by: metric:usable_depth_of_focus_nm
  - item: NILS
    by: metric:nils
  - item: MEEF
    by: metric:meef
  - item: iso-dense bias
    by: metric:iso_dense_bias_nm
  - item: mask bias each pitch needs
    by: metric:mask_bias_nm
  - item: side-lobe printing
    by: metric:side_lobe_peak_over_threshold
  - item: optical ambit
    by: metric:ambit_distance_nm
  - item: coherent to incoherent
    by: setting:coherence_type
  - item: conventional sigma (0.3 to 0.9)
    by: setting:sigma_conv
  - item: annular inner/outer sigma pairs
    by: setting:annular_sigma_inner, setting:annular_sigma_outer
  - item: x-dipole pole centre (0.5 to 0.95)
    by: setting:dipole_centre_sigma
  - item: x-dipole pole radius (0.1 to 0.2)
    by: setting:dipole_radius_sigma
  - item: conventional sigma for ambit (0.1 to 0.95)
    by: setting:sigma_conv_ambit
  - item: mask line width
    by: setting:mask_line_width_nm
  - item: Abbe
    by: setting:computation_method
  - item: Hopkins
    by: setting:computation_method
  - item: forbidden pitches
    by: metric:forbidden_pitches
  - item: common depth of focus
    by: metric:common_depth_of_focus_nm
  - item: ambit distance below 0.1 nm
    by: metric:ambit_distance_nm
  - item: visible oscillation distance
    by: metric:visible_oscillation_distance_nm
  - item: aerial image agreement
    by: metric:aerial_image_agreement
  - item: runtime
    by: metric:runtime_sec
  - item: numerical error
    by: metric:numerical_error
  - item: 2*J1(u)/u
    by: equation:E3
  - item: u = 2*pi*sigma*NA*d/lambda
    by: equation:E4
  - item: d = 0.61*lambda/(sigma*NA)
    by: equation:E5
  - item: sigma = lambda/(2*p*NA)
    by: equation:E6
  - item: NA*(1+sigma_max)/lambda
    by: equation:E7
  seed_policy: Imaging and evaluation are deterministic. For 12 free-form starts, derive independent and
    recorded random streams from a fixed master seed and the pair (source family, start index); never
    rebuild one stream for every candidate. Parametric grid searches use no randomness.
  failure_policy: Report all 12 start outcomes, optimizer termination codes, and counts of NaN, nonconvergent,
    and unprintable evaluations. An unprintable pitch contributes zero usable DOF to a source's full-sweep
    minimum; a numerical failure is never silently dropped and prevents a headline comparison until repaired
    or bounded. A failed or unresolved ambit tail is reported as a lower bound.
  ci_method: 'All optical outputs are deterministic values for fixed model settings, not sample means,
    so no stochastic confidence interval applies. Bound each reported value by successive image-sampling,
    source-quadrature, focus-step, pitch-step, and neighbour-distance refinements: search at 0.5 nm image
    spacing, at least 64 source samples per dimension, and 10 nm focus steps; check at 0.25 nm, at least
    128 samples per dimension, and 5 nm, then refine again if needed. Require DOF uncertainty at most
    5 nm, CD uncertainty at most 0.02 nm for an asserted 0.1 nm ambit, and pitch-location uncertainty
    at most 2 nm. These targets permit a 5 nm DOF bound and a 0.02 nm CD bound per deterministic result;
    withhold a smaller claimed difference unless its combined bounds are smaller. The 12 starts assess
    optimizer stability, not a physical-population interval. Report runtime as one measured CPU run with
    its hardware and software settings.'
  grid:
    illumination_source:
    - conventional
    - annular
    - x_dipole
    - quasar
    - free_form
    pitch_nm:
    - 80
    - 2000
    defocus_z_nm:
    - 0
    neighbor_distance_nm:
    - 80
    - 2000
    exposure_factor:
    - 0.95
    - 1.0
    - 1.05
    sigma_conv:
    - 0.3
    - 0.9
    annular_sigma_inner:
    - 0.5
    - 0.6
    annular_sigma_outer:
    - 0.7
    - 0.8
    dipole_centre_sigma:
    - 0.5
    - 0.95
    dipole_radius_sigma:
    - 0.1
    - 0.2
    sigma_conv_ambit:
    - 0.1
    - 0.95
    coherence_type:
    - on_axis_coherent
    - incoherent
    mask_line_width_nm:
    - 40
    - 60
    computation_method:
    - Abbe
    - Hopkins
  model:
    summary: Scalar partially coherent diffraction imaging through a 6% attenuated phase-shift thin mask,
      using independently implemented Abbe and Hopkins calculations. At NA 1.35, omission of polarization
      and vector effects can overstate contrast and depth of focus; all rankings are conditional on this
      model.
    assumptions:
    - Scalar field approximation at NA 1.35
    - Kirchhoff thin-mask approximation with dark-mask amplitude -sqrt(0.06)
    - Constant-threshold aerial-image resist without blur
    - Source power normalized equally across candidates
    - No mask three-dimensional effects or polarization
    holds_for: A numerical comparison within the stated scalar model at wavelength 193.368 nm, water index
      1.436, and NA 1.35; it is not a vector-imaging or measured-wafer prediction.
    equations:
    - id: E1
      formula: I(x,z) = integral S(q) |sum_m M_m P(q + m/p) exp[i 2*pi*m*x/p + i*phi(q + m/p,z)]|^2 dq;
        phi(f,z) = (2*pi/lambda)*z*(n - sqrt(n^2 - (lambda*f)^2))
      role: generates
      source: derivation
      derivation: Each source point forms a coherent pupil-filtered image; intensities are then summed
        with source weights. Apply the phase to each transmitted diffraction order before squaring the
        coherent field.
      example:
        inputs:
          S_total_power: 1.0
          M_zero_order: 1.0
          P_zero_order: 1.0
          z_nm: 0.0
        expected_formula: '1.0'
    - id: E2
      formula: MEEF = d(CD_wafer)/d(CD_mask_wafer) at fixed source, dose, pitch, and focus
      role: analyses
      source: derivation
      derivation: CD_mask_wafer is the reticle width scaled by the reduction factor R. MEEF = d(CD_wafer)/d(CD_mask_wafer)
        => MEEF = R * d(CD_wafer)/d(CD_mask_physical).
      example:
        inputs:
          delta_wafer_nm: 2
          delta_mask_wafer_nm: 1
        expected_formula: '2'
    - id: E3
      formula: 2*J1(u)/u
      role: analyses
      source: derivation
    - id: E4
      formula: u = 2*pi*sigma*NA*d/lambda
      role: analyses
      source: derivation
    - id: E5
      formula: d = 0.61*lambda/(sigma*NA)
      role: analyses
      source: derivation
    - id: E6
      formula: sigma = lambda/(2*p*NA)
      role: analyses
      source: derivation
    - id: E7
      formula: NA*(1+sigma_max)/lambda
      role: analyses
      source: derivation
```
