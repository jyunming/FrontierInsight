# Plan: TOPIC. For ArF immersion lithography at low k1, under which illumination sources do one-dimensional lines and spaces keep a usable depth of focus thro

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

The study will determine whether a symmetry-constrained free-form illumination pupil creates a real common-DOF advantage for low-k1 1D ArF immersion imaging, rather than merely fitting selected pitches. It will compare independently computed Abbe and Hopkins images, optimize sources on a fixed representative set, validate all sources through pitch, and relate isolated-neighbour CD oscillations to partial coherence.

## What the literature says

- **[1] Imaging application tools for extremely low-k1 ArF immersion lithography**: Extremely low-k1 ArF immersion imaging is highly sensitive to illumination pupil shape and motivates freeform pupil adjustment.
- **[2] Challenges and solutions of few-line patterns with freeform illumination in immersion lithography**: Freeform illumination can improve common process windows, but few-line special pitches remain difficult.
- **[4] Freeform source optimization for improving litho-performance of warm spots**: Reference patterns must be retained during source optimization to avoid degrading baseline performance.
- **[9] DOF and Coherence Optimization in Sub-32 nm Contact Lithography**: At low k1, small illumination-sigma changes can strongly alter DOF.
- **[10] Illumination optimization in optical projective lithography**: Pixel-based sources can be optimized under an Abbe partially coherent imaging formulation with source-complexity constraints.
- **[26] MEEF management and the effect of assist feature optical proximity corrections**: Modified illumination preferentially affects semi-dense and dense patterns, while isolated-pattern latitude behaves differently.
- **[35] On the diffraction theory of optical images**: Hopkins theory expresses partially coherent imaging through the effective source, optical transmission, and mask structure.
- **[40] Pixelated source and mask optimization for immersion lithography**: Scalar pixelated optimization is inaccurate for hyper-NA absolute prediction, motivating an explicit scalar-model limitation.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] Imaging application tools for extremely low-k1 ArF immersion lithography (DOI 10.1117/12.2011382)
- [2] Challenges and solutions of few-line patterns with freeform illumination in immersion lithography (DOI 10.1109/cstic.2018.8369217)
- [3] Extension of optical lithography by mask-litho integration with computational lithography (DOI 10.1117/12.866840)
- [4] Freeform source optimization for improving litho-performance of warm spots (DOI 10.1117/12.896978)
- [5] Optical proximity correction: A cross road of data flows (DOI 10.7567/jjap.55.06ga01)
- [6] Extending 1.35 NA immersion lithography down to 1x nm production nodes (DOI 10.1117/12.915807)
- [7] Efficient source mask optimization method for reduction of computational lithography cycles and enhancement of process-window predictability (DOI 10.1117/1.jmm.14.4.043507)
- [8] Hybrid Sub-Resolution Assisting Feature Implementation Methodology by Process Window Aware Optical Proximity Correction Model (DOI 10.1143/jjap.48.036502)
- [9] DOF and Coherence Optimization in Sub-32 nm Contact Lithography (DOI 10.1149/1.3694325)
- [10] Illumination optimization in optical projective lithography (DOI 10.1117/12.2034697)
- [11] Optical Proximity Correction, Methodology and Limitations (DOI 10.1109/cstic52283.2021.9461507)
- [12] Design of freeform illumination sources with arbitrary polarization for immersion lithography (DOI 10.1117/12.2177846)
- [13] Characterization of optical proximity correction features (DOI 10.1117/12.436743)
- [14] Integrated model-based retargeting and optical proximity correction (DOI 10.1117/12.879531)
- [15] SMO for 28-nm logic device and beyond: impact of source and mask complexity on lithography performance (DOI 10.1117/12.846473)
- [16] Lithography target optimization with source-mask optimization (DOI 10.1117/12.918080)
- [17] Mask process correction for optical weak pattern improvement (DOI 10.1117/12.2535770)
- [18] Imaging of 32-nm 1:1 lines and spaces using 193-nm immersion interference lithography with second-generation immersion fluids to achieve a numerical aperture of 1.5 and a k1 of 0.25 (DOI 10.1117/1.2039953)
- [19] Illumination pupilgram prediction and control method in advanced optical lithography (DOI 10.1117/12.899032)
- [20] MOSAIC: Mask optimizing solution with process window aware inverse correction (DOI 10.1109/dac.2014.6881379)
- [W1] Imaging Application tools for extremely low-k1 ArF immersion lithography | Request PDF
- [21] Application of illumination pupilgram control method with freeform illumination (DOI 10.1117/12.916594)
- [22] Restricted Optical Proximity Effect Correction Pattern Generation Based on Optical Proximity Effect Correction-Design for Manufacturability Rule in Model-Based Optical Proximity Effect Correction: Application for Metal Layer in Complementary Metal–Oxide–Semiconductor Logic Process (DOI 10.1143/jjap.46.6140)
- [W2] Imaging application tools for extremely low-k1 ArF immersion lithography | (2013) | Mori | Publications | Spie
- [23] Informational Lithography Approach Based on Source and Mask Optimization (DOI 10.1109/tci.2020.3048271)
- [24] Optical proximity correction considering process latitude (DOI 10.1117/12.354379)
- [W3] ArF solutions for low-k1 back-end imaging | Request PDF
- [25] The Inverse Optimization of an Optical Lithographic Source with a Hybrid Genetic Algorithm (DOI 10.3390/app13095708)
- [26] MEEF management and the effect of assist feature optical proximity corrections (DOI 10.1117/12.425205)
- [W4] Lithography series, Part 1 - Low-k(1), imaging for contacts and lines using immersion ArF | Request PDF
- [W5] Low-k1, imaging for contacts and lines using immersion ArF | Request PDF
- [27] Hybrid approach for the design of mirror array to produce freeform illumination sources in immersion lithography (DOI 10.1016/j.ijleo.2014.06.116)
- [W6] Performance of a 1.35NA ArF immersion lithography system for 40-nm applications | Request PDF
- [28] Inverse Lithography Source Optimization via Particle Swarm Optimization and Genetic Combined Algorithm (DOI 10.1109/jphot.2022.3226266)
- [29] True process variation aware optical proximity correction with variational lithography modeling and model calibration (DOI 10.1117/1.2752814)
- [W7] Performance of a 1.35NA ArF immersion lithography system for 40nm applications - art. no. 65201Y | Request PDF
- [30] Angle monitor of micromirror array for freeform illumination in lithography systems (DOI 10.1063/5.0186334)
- [W8] ArF Immersion Lithography for 45-nm and beyond
- [31] Validating optical proximity correction with models, masks, and wafers (DOI 10.1117/12.746685)
- [32] Fast optical proximity correction method based on nonlinear compressive sensing (DOI 10.1364/oe.26.014479)
- [33] Mask absorber material dependence of 2D OPC in ArF high NA lithography (DOI 10.1117/12.692807)
- [34] Feasibility study of machine learning modeling for full chip optical proximity correction (DOI 10.1117/12.2614060)
- [35] On the diffraction theory of optical images (DOI 10.1098/rspa.1953.0071)
- [36] Principles of Optics: Electromagnetic Theory of Propagation, Interference and Diffraction of Light (DOI 10.1063/1.1325200)
- [37] Fundamentals of Photonics (DOI 10.1002/0471213748)
- [38] Fundamental principles of optical lithography : the science of microfabrication
- [39] Resolution Enhancement Techniques in Optical Lithography (DOI 10.1117/3.401208)
- [40] Pixelated source and mask optimization for immersion lithography (DOI 10.1364/josaa.30.000112)
- [41] Mask Design for Optical Microlithography—An Inverse Imaging Problem (DOI 10.1109/tip.2006.891332)
- [42] Computational Lithography (DOI 10.1002/9780470618943)
- [43] Pixel-based simultaneous source and mask optimization for resolution enhancement in optical lithography (DOI 10.1364/oe.17.005783)

## The gap this experiment addresses

The supplied work supports freeform illumination and warns about low-k1 sensitivity, but does not establish in one controlled 1D ArF-water comparison whether a symmetry-constrained free-form source preserves a common-DOF advantage across dense-to-isolated pitches, how that advantage trades against OPC metrics and side-lobe printing, or how it changes optical ambit relative to the analytic coherence function.

## What is being optimised

> Shown from the design block below (`study_type` and `protocol.optimisation`) as it was when the plan was written; edit the block, not this section (after an edit, only the block counts). FI runs the search itself, within the budget below, and records every evaluation; the best design is found at the search's own numerical settings, then FI evaluates it and the baseline again at the finer check settings below, nudges it, and compares the starting points. A check that fails is reported as the result and limits the evidence level; it does not stop the quest.

**Kind of study:** find the best design (`study_type: find_best_design`), not a measurement over settings chosen in advance.

**Goal:** make common_dof_nm (Worst usable depth of focus across dense, semi-dense, forbidden-pitch, and isolated representative patterns.) as high as possible, in nm.

**What the search may change:**

- source_pixels: an array of shape [8, 8] (64 numbers); each element 0 to 1

**Held fixed:** representative_dense_pitch_nm = 80; representative_semi_dense_pitch_nm = 160; representative_forbidden_pitch_nm = 120; representative_isolated_pitch_nm = 2000; target_cd_nm = 40; wavelength_nm = 193.368; na = 1.35; water_refractive_index = 1.436; attenuated_mask_intensity_transmission = 0.06; attenuated_mask_phase_shift_rad = 3.141592653589793

**Limits every design must meet:** source_pixel_nonnegativity ≥ 0 dimensionless; source_weight_sum ≥ 0.999999999999 dimensionless; source_weight_sum ≤ 1.000000000001 dimensionless; mirror_symmetry_error ≤ 1e-12 dimensionless; pupil_boundary_outer_weight ≤ 0 dimensionless. A second goal is written as one of these limits: there is one goal.

**The design to beat (baseline):** source_pixels = [0, 0, 0, 0, 0, 0, 0, 0, ... (64 values)]. From: Standard mirror-symmetric four-pole x-dipole initialization projected to the 8×8 source grid and normalized to unit sum; its geometry follows the source families considered in [10].

**When a design counts as better:** better than the baseline by more than 1 nm (the plan sets this).

**How the search runs:** global_then_local: a search over the whole range first, then a local search from the best point found.

**Numerical settings (the search's, and the finer ones the best design and the baseline are recomputed at, so an improvement that is only a numerical error does not count):**

- fft_dx_nm: the search uses 1; the check uses 0.5, then 0.25 (the plan's values)
- source_quadrature: the search uses 121; the check uses 289, then 625 (the plan's values)
- focus_step_nm: the search uses 25; the check uses 12.5, then 6.25, then 2 (the plan's values)

**How many evaluations (worked out now, before anything runs):**

- search: 8 starting points × 180 = 1440 evaluations, at most (a limit, not a target)
- check: (3 best designs + the baseline) × 3 finer levels + 2 × 32 nudges around the best design = 76 evaluations at most, at the finer settings (each costs more than a search evaluation); not taken from the search's budget, and with its own time limit (`execution.timeout_s`)
- time: about 8.0 h for the scan and the search (20 s per evaluation, the plan's own estimate; not measured yet)
- the scan and the search stop at the run's time limit (`execution.timeout_s`), whatever is left of the budget; the results say which came first

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** A scalar, one-dimensional, partially coherent Fourier-optical model propagates a thin attenuated-PSM spectrum through a defocused circular pupil and computes aerial intensity by independent Abbe and Hopkins formulations.

**What it assumes:** Kirchhoff thin-mask approximation with dark-mask complex amplitude sqrt(0.06)*exp(i*pi).; Scalar imaging with no polarization despite NA 1.35.; Uniform incoherent mixing among sampled source points.; Aerial-image constant threshold represents printing; no resist blur or development model is included.; The water/resist-side medium has refractive index 1.436 for the supplied defocus phase.

**Where it holds:** Comparative 1D scalar simulations of vertical 40 nm target lines/spaces at wavelength 193.368 nm, NA 1.35, source support sigma<=0.95, pitches 80–2000 nm, and finite sampled focus range; it does not provide vector or wafer-calibrated predictions.

**Its equations:**

- **E1** (produces the data): `t_dark = sqrt(T_dark)*exp(i*phi_dark)`. Source: derivation: The topic fixes dark-region intensity transmission T_dark=0.06 and phase shift phi_dark=pi. Field amplitude is the square root of intensity transmission, and a phase shift multiplies a complex field by exp(i*phase).
- **E2** (produces the data): `pupil_phase_rad = 2*pi/lambda_nm*z_nm*(n_medium-sqrt(n_medium**2-(lambda_nm*f_cpnm)**2))`. Source: derivation: This is the defocus pupil phase supplied in the topic, written with wavelength and spatial frequency in nm and cycles per nm so that lambda_nm*f_cpnm is dimensionless.
- **E3** (analyses the results): `gamma = 2*J1(u)/u, u = 2*pi*sigma*NA*distance_nm/lambda_nm; gamma=1 when u=0`. Source: derivation: The topic supplies the circular-source mutual-coherence function. Its u=0 value follows from the Bessel-series limit J1(u)=u/2+O(u**3).
- **E4** (analyses the results): `coherence_first_zero_nm = 0.61*lambda_nm/(sigma*NA)`. Source: derivation: This is the supplied first-zero approximation for the circular-source coherence function.
- **E5** (analyses the results): `dipole_pole_sigma = lambda_nm/(2*pitch_nm*NA)`. Source: derivation: This is the supplied symmetric two-beam dipole-pole relation for a dense pitch.
- **E6** (analyses the results): `f_max_cpnm = NA*(1+sigma_max)/lambda_nm`. Source: derivation: This is the supplied maximum transmitted spatial-frequency relation: frequencies above it never reach the wafer.
- **E7** (produces the data): `I_Abbe(x) = sum_s w_s*abs(IFFT(P(f+s)*M(f)))**2`. Source: [28]
- **E8** (produces the data): `TCC(f1,f2) = sum_s w_s*P(f1+s)*conj(P(f2+s)); I_Hopkins(x) = sum_f1,sum_f2 TCC(f1,f2)*M(f1)*conj(M(f2))*exp(i*2*pi*(f1-f2)*x)`. Source: [35]
- **E9** (analyses the results): `NILS = abs((1/I_threshold)*(dI/dx)_edge)`. Source: derivation: NILS is defined operationally as the magnitude of normalized aerial-image log slope at the printing threshold; dI/dx is evaluated by centered finite difference at the interpolated threshold edge.
- **E10** (analyses the results): `MEEF = abs((CD_wafer(mask_cd_nm+delta_nm)-CD_wafer(mask_cd_nm-delta_nm))/(2*delta_nm))`. Source: derivation: MEEF is evaluated as the magnitude of the local finite-difference wafer-CD response to mask-CD change, using the fixed ±1 nm perturbation.
- **E11** (analyses the results): `common_dof_nm = min(DOF_dense_nm, DOF_semi_dense_nm, DOF_forbidden_nm, DOF_isolated_nm)`. Source: derivation: The selected source objective is the worst usable DOF among the fixed representative conditions, so maximizing it is a maximin common-window search.

**The checks against known answers, and where each expected value comes from:**

- **coherence_at_zero**, a special or limiting case with a known answer: expects 1.0; computed as `mutual_coherence` from what the simulation returns on `sigma=0.5, distance_nm=0.0`; from: derivation: J1(u)=u/2+O(u**3), so 2*J1(u)/u tends to 1 as u tends to 0.
- **coherence_first_zero**, a special or limiting case with a known answer: expects 0.0; computed as `mutual_coherence` from what the simulation returns on `sigma=0.5, distance_nm=174.76746666666668`; from: derivation: use the supplied first-zero relation d=0.61*lambda/(sigma*NA), substitute it into u=2*pi*sigma*NA*d/lambda, and evaluate 2*J1(u)/u; 0.61 is the standard rounded first-zero coefficient.
- **source_weight_normalization**, a conserved quantity or other invariant: expects 0.0; computed as `abs(source_weight_sum-1)` from what the simulation returns on `source_family=0, conventional_sigma=0.5`; from: derivation: source quadrature weights are explicitly divided by their finite positive sum, therefore their sum is exactly one up to floating-point roundoff.
- **mirror_source_symmetry**, a symmetry or scaling law: expects 0.0; computed as `source_mirror_symmetry_error` from what the simulation returns on `source_pixel_grid=8`; from: derivation: the mirror projection replaces every orbit of reflected pixels by its arithmetic mean, making each reflected pair equal apart from floating-point roundoff.
- **baseline_source_validity**, a conserved quantity or other invariant: expects 0.0; computed as `abs(baseline_weight_sum-1)+baseline_mirror_symmetry_error` from what the simulation returns on `source_family=baseline_pixelated, source_pixel_grid=8`; from: derivation: the baseline design must strictly meet all optimization constraints prior to search.
- **abbe_hopkins_agreement**, an independent second implementation: expects 0.0; computed as `abbe_hopkins_max_abs_error` from what the simulation returns on `pitch_nm=200.0, conventional_sigma=0.3, defocus_nm=0.0, fft_dx_nm=1.0, source_quadrature=121`; from: Abbe implementation: coherent field is formed separately for every sampled source point and intensities are weighted and summed. Hopkins implementation: independently construct TCC(f1,f2)=sum_s w_s P(f1+s)P*(f2+s) and contract it with mask Fourier coefficients; no image-field code is shared.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **Abbe-Hopkins consistency**: The independent partially coherent image formulations must agree on the fixed small validation case.; how far the check “abbe_hopkins_agreement” lands from the value it expects; lower is better; met at 0.002 or below; a change smaller than 0.0005 counts as no change.
- **source normalization**: All normalized source constructions must conserve total source weight.; the value the check “source_weight_normalization” measures; lower is better; met at 1e-12 or below; a change smaller than 5e-13 counts as no change.
- **freeform symmetry**: Projection of a candidate free-form source must enforce the prescribed mirror symmetry.; the value the check “mirror_source_symmetry” measures; lower is better; met at 1e-12 or below; a change smaller than 5e-13 counts as no change.
- **coherence zero**: The implemented circular-source coherence must be close to zero at the supplied first-zero distance.; how far the check “coherence_first_zero” lands from the value it expects; lower is better; met at 0.003 or below; a change smaller than 0.001 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/duv_lowk1_source_dof_opc_ambit/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 125 more lines of code than two scripts (40 written by the model, 85 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- Support: at check resolution, the free-form source exceeds the best feasible parametric source in held-out minimum usable DOF by more than the combined numerical uncertainty and has no held-out pitch with zero usable DOF when the comparator prints.
- Against: the apparent free-form gain vanishes or reverses on held-out pitches or finer numerical checks, or a parametric source matches it within combined numerical uncertainty.
- Support: Abbe and Hopkins normalized aerial images agree within the fixed oracle tolerance on independent small cases.
- Support: isolated-neighbour CD oscillation zero crossings and decay are qualitatively aligned with the analytic mutual-coherence zero locations, while reported discrepancies remain explicit.

## Risks

- Scalar imaging can overstate contrast and DOF at NA 1.35 because vector polarization and mask-3D effects are omitted.
- A source optimized on four representatives can overfit; held-out through-pitch validation is therefore decisive.
- Threshold edge extraction can be unstable near side lobes or low NILS; failed bracketing is retained as an unprintable result.
- Finite source pixels, FFT grid, source quadrature, focus increments, and source-family sampling can shift forbidden-pitch locations.

## What this quest will not do

- Vector electromagnetic mask simulation, polarization, flare, aberrations, scanner pupil realization error, and measured wafer calibration.
- Chemically amplified resist blur, stochastic resist noise, development, etch transfer, collapse, and line-edge roughness.
- Two-dimensional layouts, line ends, SRAFs, and full-chip OPC.
- Claims of manufacturing qualification or quantitative agreement with any specific scanner.

## Checks already made

- optimisation_baseline; protocol.optimisation.baseline and protocol.oracles; The baseline source_pixels matrix contains eight entries of 0.25 summing to 2.0, violating the source_weight_sum constraint of 1.0, and protocol.oracles contains no oracle checking the baseline design.; An unnormalized baseline violates the optimization constraints and invalidates comparative benchmarking, and omitting an oracle leaves the baseline design unverified.; Normalize the baseline source_pixels entries to 0.125 so their sum is exactly 1.0, and add a baseline_source_validity oracle in protocol.oracles that verifies the baseline satisfies unit sum and mirror symmetry.; high
- optimisation_ranges; protocol.optimisation.constraints and protocol.model.holds_for; The 8×8 source_pixels design variable allows unconstrained nonzero weights for corner pixels where sigma exceeds the 0.95 pupil boundary specified in model.holds_for.; The physical illumination pupil cannot transmit light beyond the maximum illuminator radius, and allowing optimizer search outside sigma <= 0.95 exploits unphysical pupil coordinates.; Add an explicit pupil_boundary_outer_weight constraint in protocol.optimisation.constraints requiring all pixels with radial distance sqrt(sigma_x^2 + sigma_y^2) > 0.95 to equal 0.0.; high
- precision; protocol.precision, protocol.thresholds, and protocol.optimisation.numerical_settings; The protocol targets a usable DOF numerical half-width of 1.0 nm, but the focus step in numerical_settings is 25.0 nm for search and 12.5/6.25 nm for check, which cannot resolve a 1.0 nm interval width.; Discrete focus grid sampling cannot achieve an interval half-width tighter than half the grid step, leaving the claimed 1.0 nm precision unsupported by the specified numerical settings.; Add a 2.0 nm check focus step to protocol.thresholds and protocol.optimisation.numerical_settings, and specify bracketed root-solving for the usable defocus boundaries to within 1.0 nm tolerance.; high
- threshold_sensitivity; protocol.thresholds, protocol.acceptance, and protocol.definitions; The classification of forbidden pitches relies on an arbitrary 25% depth drop relative to adjacent maxima without evaluating sensitivity to this cut-off.; A fixed 25% cut-off may arbitrarily include or exclude borderline pitch dips, making the identified forbidden pitches sensitive to the chosen threshold.; Add a forbidden_pitch_depth_fractions array [0.15, 0.20, 0.25, 0.30] to protocol.thresholds and require in protocol.acceptance that reported forbidden pitches persist across this range.; medium
- weak_baseline; method and protocol.optimisation.grid; The design sweeps parametric source families but does not explicitly state the tuning protocol used to select the best comparator in each family.; Without an explicit tuning criterion for the baseline and parametric comparators, the comparative evaluation risks comparing against an ad-hoc or arbitrarily chosen parameter setting.; Specify explicitly in method and protocol.optimisation that the comparator for each parametric family is tuned by selecting the parameter setting that maximizes common DOF on the representative pitches under identical OPC and dose calibration rules.; medium
- other; protocol.metrics and protocol.definitions; The topic requests 'how far its oscillation is still visible' (listed as visible oscillation distance in variables.dependent), but it is omitted from protocol.metrics and lacks an operational entry in protocol.definitions.; Leaving a requested dependent variable out of the metrics and definitions protocol prevents its systematic evaluation and verification during analysis.; Add visible_oscillation_distance_nm to protocol.metrics and add its operational definition to protocol.definitions based on the largest neighbour distance where CD oscillation amplitude exceeds 0.02 nm.; medium
- `protocol.precision` was left out of the plan because it could not be checked (`protocol.precision` must be a mapping with a `target_half_width` between 0 and 1 (for example 0.03)); FI asks the plan's model to write it before the experiment runs.
- `protocol.grid` was left out: a search for the best design has no separate sweep (its coarse scan is `protocol.optimisation.grid`)
- `protocol.optimisation.grid` was left out of the plan because it could not be read (`protocol.optimisation.grid` scans `representative_pitch_nm`, which is neither a design variable nor a condition held fixed: to compare the study across `representative_pitch_nm`, also list it under `protocol.optimisation.fixed` with the value the search holds it at (a setting whose values are names needs no such entry)); it is not used
- The methodology audit amended the design: method, protocol (before and after are in needs/DESIGN_CRITIQUE.json). The audit proposed changing `protocol.criteria`, but FI did not apply it: `protocol.criteria` entry 'baseline source validity' names no declared check ('baseline_source_validity'); the protocol's checks are: coherence_at_zero, coherence_first_zero, source_weight_normalization, abbe_hopkins_agreement, mirror_source_symmetry; it was left out of the plan and is not used The audit proposed changing `protocol.thresholds`, but FI did not apply it: the threshold `forbidden_pitch_depth_fractions` was left out of `protocol.thresholds` because it is not one number ([0.15, 0.2, 0.25, 0.3]); it is not used The design was read by a different model (gemini-3.8-flash-high), not the one that wrote it (gpt-5.6-terra).
- Equation E2 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E4 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E5 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E6 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E9 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E10 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E11 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- The check 'source_weight_normalization' names a derivation for its expected value, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'mirror_source_symmetry' names a derivation for its expected value, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'baseline_source_validity' names a derivation for its expected value, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'abbe_hopkins_agreement' compares with a second implementation but does not say what code it does not share with the simulation (write, for example, “shares no code with the simulation: it uses scipy's solve_ivp”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

- FI asked the plan's model to write `protocol.precision` again, and it is now in the plan.
- FI asked the plan's model to write `protocol.steps` again, and it is complete now.
- FI asked the plan's model to write `protocol.simplifications` again, and it still has gaps (the experiment goes on; the paper says so).

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gemini-3.8-flash-high (the connection did not say which model answered)), not the one that wrote the plan (gpt-5.6-terra). Its summary: The check suite contains critical blind spots: four of the six checks test self-fulfilling arithmetic or static inputs, and the only aerial-image test uses zero defocus, leaving the core defocus model completely unverified. In addition, the model equations for NILS and common DOF deviate from standard lithography definitions by omitting nominal CD normalization and inter-pitch focus shifts.
  - **coherence_at_zero**: may not test the model, would miss a plausible bug, its number is well defined, its case may not isolate what it measures. By evaluating at zero distance, it isolates only the conditional branch in the code and leaves the Bessel coherence physics unexercised.; a better check: Evaluate mutual coherence at a non-zero separation where the first-order Bessel function has a non-trivial value..
  - **coherence_first_zero**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures.
  - **source_weight_normalization**: may not test the model, would miss a plausible bug, its number is well defined, its case may not isolate what it measures. It isolates only the normalization division step while leaving the actual source pupil shape and quadrature weights unverified.; a better check: Test the unnormalized integrated weight against the analytical pupil disk area..
  - **mirror_source_symmetry**: may not test the model, would miss a plausible bug, its number is well defined, its case isolates what it measures. a better check: Verify that an analytical symmetric source retains fourfold quadrant symmetry on the discrete numerical grid..
  - **baseline_source_validity**: may not test the model, would miss a plausible bug, its number is well defined, its case may not isolate what it measures. It combines weight sum deviation and symmetry error into a single sum, conflating two separate constraints..
  - **abbe_hopkins_agreement**: tests the model, would catch a plausible bug, its number is well defined, its case may not isolate what it measures. The case sets defocus to zero and normalizes by peak intensity, completely bypassing the defocus pupil phase and masking absolute scaling errors.; a better check: Evaluate Abbe and Hopkins agreement at non-zero defocus without peak-intensity normalization..
  - Found (high): E9: NILS is defined without the normalizing factor of target linewidth (Normalized Image Log-Slope is dimensionless and defined as CD multiplied by image log-slope; omitting CD yields unnormalized ILS with dimensions of reciprocal length.). Fix: Multiply the image log-slope by the nominal target critical dimension.
  - Found (high): E11: Common DOF is defined as the minimum of individual pitch DOFs rather than the overlapping focus window across pitches (Individual pitches experience different best-focus shifts, so the common usable DOF is the intersection length of their focus windows, which can vanish even if all individual DOFs are large.). Fix: Define common DOF as the length of the intersection of the usable focus intervals across pitches.
  - Found (high): abbe_hopkins_agreement: The simulation check tests Abbe and Hopkins agreement only in-focus with peak-intensity normalization (Omitting defocus completely bypasses the defocus pupil phase in a quest centered on depth of focus, while peak normalization conceals radiometric scale mismatches.). Fix: Test Abbe and Hopkins agreement at a non-zero defocus condition without peak-intensity normalization.
  - Found (medium): source_weight_normalization: The check tests an identity that is satisfied by construction through explicit array division (Because weights are divided by their own sum, their sum is unity by construction regardless of whether the source geometry or quadrature weights are correct.). Fix: Compare the unnormalized integrated source weight against the analytical pupil disk area.
  - Found (medium): coherence_at_zero: The check evaluates mutual coherence solely at zero separation distance where the code returns a hardcoded branch (At vanishing distance, the optical coordinate vanishes identically, so the check tests only the removable-singularity branch and fails to exercise the Bessel coherence formula.). Fix: Evaluate mutual coherence at a non-zero separation where the first-order Bessel function is non-trivial.
  - Found (low): mirror_source_symmetry: The check verifies an orbit-averaging operation that produces a symmetric matrix by mathematical definition (Replacing pixel orbits by their mean guarantees symmetry regardless of whether the physical source distribution is valid.). Fix: Verify that an analytical symmetric illumination source preserves quadrant symmetry when discretized.
  - Found (low): baseline_source_validity: The check tests static input data properties rather than simulation execution, and sums two unrelated error metrics (It checks an initial candidate array rather than optical propagation code, and summing normalization error with symmetry error obscures which constraint failed.). Fix: Separate the normalization and symmetry checks and focus validation on the simulation solver rather than static inputs.
  - Found (medium): model: The plan omits formal definitions and simplifications for multiple key topic metrics including iso-dense bias, mask bias through pitch, side-lobe printing, and optical ambit (The topic explicitly tasks the research quest with analyzing iso-dense bias, mask bias, side-lobe printing, and optical ambit, but the model provides no equations or simplifications for them.). Fix: Add explicit equations and modeling simplifications for iso-dense bias, mask bias per pitch, side-lobe printing thresholding, and optical ambit reach.
  - Found (low): E2: The defocus model assumes a single homogeneous immersion medium without documenting the resist interface simplification (In immersion lithography, water is the immersion fluid while resist has a higher refractive index, creating refraction and thin-film effects that are omitted without being listed under simplifications.). Fix: Explicitly state under model simplifications that propagation inside the resist is approximated as a single homogeneous immersion medium without interface refraction or standing waves.
  - Checked, and no problem found: E1: verified dark-region complex transmission formula against Kirchhoff thin-mask attenuated-PSM standard definition.; E2: verified non-paraxial defocus pupil phase formula against wavevector propagation in an immersion medium.; E3: verified circular source mutual coherence function formula against the van Cittert-Zernike theorem.; E4: verified first mutual coherence zero formula against the first root of the first-order Bessel function.; E5: verified symmetric two-beam illumination condition against pupil diffraction geometry.; E6: verified maximum transmitted spatial frequency formula against the Abbe cutoff limit.; E7: verified discrete Abbe formulation against Fourier optics source-point decomposition.; E8: verified discrete Hopkins transmission cross coefficient definition and bilinear expansion against standard Hopkins theory.; and 2 more.
  - Equations the checks test: E3, E4, E7, E8; not tested by any: E1, E2, E5, E6, E9, E10, E11.
- The model's 11 equations were read by gemini-3.8-flash-high, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is the standard definition or a correct derivation.
  - **E3** is the standard definition or a correct derivation.
  - **E4** is the standard definition or a correct derivation.
  - **E5** is the standard definition or a correct derivation.
  - **E6** is the standard definition or a correct derivation.
  - **E7** is the standard definition or a correct derivation.
  - **E8** is the standard definition or a correct derivation.
  - **E9** is not the standard definition: NILS = target_cd_nm * abs((1/I_threshold)*(dI/dx)_edge) (Standard NILS is normalized by multiplying the log-slope by the nominal target CD, making it dimensionless.).
  - **E10** is the standard definition or a correct derivation.
  - **E11** is not the standard definition: common_dof_nm = max(0.0, min_k(z_max_nm[k]) - max_k(z_min_nm[k])) (Common DOF is the overlapping focus range across pitches at a common threshold, not the minimum of individual pitch DOFs.).

- FI asked the plan, once, to look at what the second reader found.
- The plan could not be rewritten (the revised plan could not be used (the design block is not valid YAML (line 628 of the block): mapping values are not allowed here); plan.md is unchanged); nothing was changed for it.
- After the plan was asked once for a worked example, equation E7 is not tested at all: the plan says it has no closed form for its output (The FFT-based field and finite sampled source sum have no scalar closed-form output for the nontrivial mask used in the experiment.) and no limit, symmetry, conservation or second implementation checks it.
- After the plan was asked once for a worked example, equation E8 is not tested at all: the plan says it has no closed form for its output (The finite TCC contraction has no useful closed form for the nontrivial periodic mask and sampled extended source.) and no limit, symmetry, conservation or second implementation checks it.

## What the topic asks for


- The plan did not give all of what it was asked for about its simplifications, definitions and steps (the paper will say so in its limitations):
  - `protocol.simplifications` is incomplete: the topic asks for pitches, which the plan neither includes nor names in `simplifications`; the topic asks for neighbour moved, which the plan neither includes nor names in `simplifications`; the topic asks for two independent ways, which the plan neither includes nor names in `simplifications`; the topic asks for the mask bias OPC needs, which the plan neither includes nor names in `simplifications`; the topic asks for whether an attenuated-mask side lobe reaches the printing threshold, which the plan neither includes nor names in `simplifications`; the topic asks for forbidden pitches, which the plan neither includes nor names in `simplifications`; the topic asks for the CD change of the isolated line as its neighbour moves away, which the plan neither includes nor names in `simplifications`; the topic asks for the distance beyond which it stays below 0.1 nm, which the plan neither includes nor names in `simplifications`; the topic asks for how far its oscillation is still visible, which the plan neither includes nor names in `simplifications`; the topic asks for partially coherent aerial image, which the plan neither includes nor names in `simplifications`; the topic asks for first zero, which the plan neither includes nor names in `simplifications`
> Read from the topic by another model than the plan's (gemini-3.8-flash-high), then looked for in the plan.

- To vary or compare: Sources (conventional, annular, x-dipole, quasar, free-form pixelated source) - NOT in the plan's settings.
- To vary or compare: pitches (80 to 2000) - NOT in the plan's settings.
- To vary or compare: neighbour moved (80 to 2000) - NOT in the plan's settings.
- To vary or compare: conventional (0.3 to 0.9) - NOT in the plan's settings.
- To vary or compare: annular (inner/outer sigma pairs) - NOT in the plan's settings.
- To vary or compare: pole centre (0.5 to 0.95) - NOT in the plan's settings.
- To vary or compare: pole radius (0.1 to 0.2) - NOT in the plan's settings.
- To vary or compare: sigma (0.1 to 0.95) - NOT in the plan's settings.
- To vary or compare: two independent ways (Abbe, Hopkins) - NOT in the plan's settings.
- To report: depth of focus.
- To report: NILS at best focus.
- To report: MEEF.
- To report: the mask bias OPC needs.
- To report: the iso-dense bias before OPC.
- To report: whether an attenuated-mask side lobe reaches the printing threshold.
- To report: forbidden pitches.
- To report: the CD change of the isolated line as its neighbour moves away.
- To report: the distance beyond which it stays below 0.1 nm.
- To report: how far its oscillation is still visible.
- To report: partially coherent aerial image.
- To report: common depth of focus.
- To report: runtime - NOT measured in the plan.
- To report: numerical error - NOT measured in the plan.
- Formula the topic states: `2*pi/lambda * z * (n - sqrt(n**2 - (lambda*f)**2))`.
- Formula the topic states: `2*J1(u)/u with u = 2*pi*sigma*NA*d/lambda`.
- Formula the topic states: `d = 0.61*lambda/(sigma*NA)`.
- Formula the topic states: `sigma = lambda/(2*p*NA)`.
- Formula the topic states: `NA*(1+sigma_max)/lambda`.
- To compare: Compare the partially coherent aerial images computed independently using the Abbe and Hopkins formulations to verify that they agree.
- To compare: Compare the optimized free-form source next to the best parametric source.
- To compare: Compare the oscillation of the CD change of the isolated line as its neighbour moves away with the mutual coherence function.
- Constraint: Exposure is fixed at wavelength 193.368 nm through water (refractive index 1.436) with a projection lens of NA 1.35.
- Constraint: The mask is held fixed as a thin 6% attenuated phase-shift mask where dark regions transmit amplitude sqrt(0.06) with a phase of pi using the Kirchhoff thin-mask approximation.
- Constraint: Imaging must be scalar with no polarization.
- Constraint: The paper must state plainly that imaging is scalar and what that changes at NA 1.35, specifically that it overstates contrast and depth of focus.
- Constraint: The partially coherent aerial image must be computed two independent ways (Abbe and Hopkins) that must agree.
- Constraint: Patterns are constrained to vertical lines (dark on the attenuated mask) and spaces with a target line width of 40 nm.
- Constraint: Resist imaging is constrained to a constant-threshold aerial image with no resist blur.
- Constraint: Dose must be anchored so the densest pitch prints at 40 nm at best focus.
- Constraint: A one-dimensional OPC must set the mask line width at every pitch so it prints at 40 nm at best focus.
- Constraint: The process window tolerance requires a CD tolerance of +/-10% of 40 nm at 5% exposure latitude.
- Constraint: The free-form pixelated source optimization must keep mirror symmetry.
- Constraint: Computations must be CPU only.
- Constraint: A long run is allowed (source optimization may take hours), and the runtime must be reported.
- Constraint: Every number must be reported with its numerical error (grid, source sampling, focus step) and it must be stated when a result is only a check of the code.
- Constraint: The implementation must use Python with numpy and scipy, and matplotlib with the Agg backend.
- Constraint: The study must produce a paper that answers the question with numbers.
- Constraint: Only sources that have actually been found and read may be cited, and unconfirmed references must be stated as such rather than cited.

- FI asked the plan's model to add what was missing; it is still missing.

- Still missing (the paper will say it was not studied; the quest stops before anything runs, because the topic's question cannot be answered without it):
  - The topic says to vary or compare Sources (conventional, annular, x-dipole, quasar, free-form pixelated source), but the plan's settings do not include it.
  - The topic says to vary or compare pitches (80 to 2000), but the plan's settings do not include it.
  - The topic says to vary or compare neighbour moved (80 to 2000), but the plan's settings do not include it.
  - The topic says to vary or compare conventional (0.3 to 0.9), but the plan's settings do not include it.
  - The topic says to vary or compare annular (inner/outer sigma pairs), but the plan's settings do not include it.
  - The topic says to vary or compare pole centre (0.5 to 0.95), but the plan's settings do not include it.
  - The topic says to vary or compare pole radius (0.1 to 0.2), but the plan's settings do not include it.
  - The topic says to vary or compare sigma (0.1 to 0.95), but the plan's settings do not include it.
  - The topic says to vary or compare two independent ways (Abbe, Hopkins), but the plan's settings do not include it.
  - The topic asks to report runtime, but the plan does not measure it.
  - The topic asks to report numerical error, but the plan does not measure it.

## The design (used as written)

```yaml
study_type: find_best_design
hypothesis: Within the 1D scalar ArF immersion aerial-image model, a mirror-symmetric pixelated free-form
  source optimized for common depth of focus will exceed the best conventional, annular, x-dipole, and
  quasar sources on representative 40 nm line/space pitches, but its benefit will diminish at held-out
  forbidden pitches and will retain source-dependent optical-ambit oscillations.
variables:
  independent:
  - illumination source family and parameters
  - pitch (80–2000 nm plus isolated limit)
  - defocus
  - dose/exposure latitude
  - mask linewidth during one-dimensional OPC
  - neighbour distance (80–2000 nm)
  dependent:
  - usable depth of focus
  - common depth of focus
  - NILS
  - MEEF
  - OPC mask bias
  - pre-OPC iso-dense bias
  - side-lobe threshold crossing
  - forbidden-pitch locations
  - isolated-line CD change
  - ambit distance below 0.1 nm
  - visible oscillation distance
  - Abbe–Hopkins aerial-image discrepancy
  controls:
  - wavelength 193.368 nm
  - water refractive index 1.436
  - projection NA 1.35
  - scalar unpolarized imaging
  - thin 6% attenuated phase-shift mask
  - target wafer CD 40 nm
  - constant-threshold resist model
  - dose anchor at dense pitch best focus
  - CD tolerance ±10% of target
method: Use NumPy FFT scalar imaging with both independent Abbe source-point summation and Hopkins TCC
  evaluation. For every source, calculate 1D aerial images of a thin 6% attenuated PSM, use scipy.optimize.brentq
  for threshold-edge, OPC-width solutions, and bracketed process-window focus boundary determinations,
  scipy.special.j1 for mutual coherence, and scipy.optimize.differential_evolution followed by constrained
  local refinement for the 8×8 mirror-symmetric pixelated source. Each parametric source comparator (conventional,
  annular, x-dipole, quasar) is tuned by sweeping its defined parameter grid (conventional sigma from
  0.3 to 0.9 in steps of 0.05; annular outer sigma 0.5 to 0.95 and ring width 0.1 to 0.3 in steps of 0.05;
  x-dipole pole center 0.5 to 0.95 in steps of 0.05 and pole radius 0.1 to 0.2 in steps of 0.025; quasar
  matching annular radial bounds with 20-degree sector angles) to maximize the identical common-DOF objective
  across the representative pitches under nominal dose calibration and OPC rules. Arbor is used to manage
  objective-driven source-search branches and held-out-pitch validation, preventing representative-set
  overfitting. uncertainty-and-units is used for nm, inverse-nm, focus, source-sampling, grid, and focus-step
  uncertainty propagation. get-available-resources determines a conservative CPU worker count before the
  sweep. Matplotlib Agg plus scientific-visualization produces accessible PNG figures and audits axes,
  uncertainty marks, and export metadata. Free-form optimization maximizes the minimum usable DOF over
  dense, semi-dense, forbidden-pitch, and isolated representatives; all source families are then validated
  through pitch, including held-out pitches. Ambit oscillation visibility is evaluated across the sensitivity
  set [0.01, 0.02, 0.05] nm. Runtime is separately recorded for source sweeps, optimization, and final
  validation. Scalar results are explicitly labelled optimistic at NA 1.35 because polarization, vector
  imaging, resist blur, mask-3D scattering, aberrations, and stochastic resist effects are excluded.
expected_outcome: The free-form source is expected to improve the representative common DOF relative to
  the optimized conventional baseline and likely the best parametric source, while through-pitch validation
  may reveal narrow forbidden-pitch losses; this experiment tests that prediction rather than assuming
  it.
figures_planned:
- figures/dof_through_pitch.png
- figures/opc_metrics_through_pitch.png
- figures/freeform_vs_parametric_source.png
- figures/optical_ambit_and_coherence.png
dependencies:
- numpy
- scipy
- matplotlib
- pint
- uncertainties
result_assertions:
- path: nils
  min: 1.0e-06
  max: 1000000.0
  unit: dimensionless
  reason: NILS is the nonnegative magnitude of the normalized aerial-image log slope at a threshold crossing;
    every retained print has a nonzero edge slope.
- path: meef
  min: 0.0
  max: 1000000.0
  unit: dimensionless
  reason: MEEF is reported as the nonnegative magnitude of the local wafer-CD response to mask-CD perturbation.
- path: dof_nm
  min: 0.0
  max: 100000.0
  unit: nm
  reason: DOF is a nonnegative width of an accepted finite focus interval; the bounded scanned focus domain
    is ±50,000 nm.
- path: cd_nm
  min: 0.0
  max: 10000.0
  unit: nm
  reason: A threshold-defined linewidth is nonnegative and cannot exceed the 10,000 nm simulated spatial
    period/window.
protocol:
  thresholds:
    wavelength_nm: 193.368
    water_refractive_index: 1.436
    na: 1.35
    target_cd_nm: 40.0
    attenuated_mask_intensity_transmission: 0.06
    attenuated_mask_phase_shift_rad: 3.141592653589793
    cd_tolerance_fraction: 0.1
    exposure_latitude_fraction: 0.05
    side_lobe_print_threshold_fraction: 1.0
    ambit_cd_change_nm: 0.1
    mask_bias_perturbation_nm: 1.0
    source_pixel_grid: 8
    source_pixel_sigma_limit: 0.95
    fft_dx_nm_search: 1.0
    fft_dx_nm_check: 0.5
    source_quadrature_search: 121
    source_quadrature_check: 289
    focus_step_nm_search: 25.0
    focus_step_nm_check: 12.5
    isolated_pitch_nm: 2000.0
  acceptance:
  - The Abbe–Hopkins second-implementation oracle must pass before any main sweep.
  - The baseline source validity oracle must pass before starting free-form optimization.
  - A source is eligible for ranking only if its normalized source weights are nonnegative, sum to one
    within 1e-12, obey mirror symmetry within 1e-12, and place zero weight outside sigma 0.95.
  - A usable focus point must print CD within 36–44 nm at all three dose scales 0.95, 1.0, and 1.05; DOF
    is the width of the contiguous accepted focus interval containing best focus.
  - A forbidden pitch is a local through-pitch DOF minimum at least 25% below both available adjacent-pitch
    maxima (and persisting across the depth fraction range 0.15–0.30), after excluding pitches classified
    unprintable at all sources.
  - A free-form advantage is accepted only when it persists at 0.5 nm sampling, 289 source samples, and
    2.0 nm focus step, and exceeds the combined numerical uncertainty.
  metrics:
  - id: dof_nm
    estimand: Width of the contiguous focus interval around nominal focus satisfying the fixed CD and
      5% exposure-latitude criterion for one source and pitch.
    kind: mean
    unit: one deterministic source–pitch optical simulation
    cluster: null
    paired: false
    family: source comparisons through pitch
  - id: common_dof_nm
    estimand: Minimum usable DOF over the four fixed representative pitches for a source.
    kind: mean
    unit: one deterministic source evaluation
    cluster: null
    paired: false
    family: source optimization and validation
  - id: nils
    estimand: Absolute normalized image log slope at the two nominal best-focus threshold edges, averaged
      over edges.
    kind: mean
    unit: one deterministic source–pitch image
    cluster: null
    paired: false
    family: OPC metrics through pitch
  - id: meef
    estimand: Absolute central finite-difference derivative of wafer CD with respect to mask CD using
      ±1 nm mask perturbations.
    kind: mean
    unit: one deterministic source–pitch–mask perturbation triplet
    cluster: null
    paired: false
    family: OPC metrics through pitch
  - id: mask_bias_nm
    estimand: OPC mask linewidth minus 40 nm target linewidth at best focus.
    kind: mean
    unit: one deterministic OPC root solution
    cluster: null
    paired: false
    family: OPC metrics through pitch
  - id: iso_dense_bias_nm
    estimand: Difference between pre-OPC 40 nm-mask printed CD at isolated and densest pitches.
    kind: mean
    unit: one paired deterministic isolated/dense image evaluation
    cluster: null
    paired: true
    family: source iso-dense comparisons
  - id: side_lobe_prints
    estimand: Indicator that an off-target local aerial-image maximum crosses the calibrated printing
      threshold.
    kind: proportion
    unit: one deterministic source–pitch image, reported as 0 or 1
    cluster: null
    paired: false
    family: side-lobe assessments
  - id: cd_change_nm
    estimand: Central isolated-line CD with a neighbour minus its CD at 2000 nm neighbour separation.
    kind: mean
    unit: one deterministic neighbour-distance image
    cluster: null
    paired: true
    family: optical ambit curves
  - id: ambit_distance_nm
    estimand: Smallest tested neighbour distance beyond which absolute CD change remains below 0.1 nm
      for every larger tested distance.
    kind: mean
    unit: one deterministic source ambit curve
    cluster: null
    paired: false
    family: optical ambit source comparisons
  - id: visible_oscillation_distance_nm
    estimand: Largest tested neighbour distance at which the local CD oscillation amplitude exceeds 0.02
      nm.
    kind: mean
    unit: one deterministic source ambit curve
    cluster: null
    paired: false
    family: optical ambit source comparisons
  - id: abbe_hopkins_max_abs_error
    estimand: Maximum absolute difference of independently normalized Abbe and Hopkins aerial intensities
      on the same spatial grid.
    kind: mean
    unit: one deterministic validation image
    cluster: null
    paired: true
    family: implementation validation
  oracles:
  - name: coherence_at_zero
    kind: special_case
    check: Evaluate the analytic mutual-coherence function at zero separation using its removable-limit
      implementation.
    expected: 1.0
    expected_formula: '1'
    tolerance: 1.0e-14
    tolerance_mode: absolute
    reference: 'derivation: J1(u)=u/2+O(u**3), so 2*J1(u)/u tends to 1 as u tends to 0.'
    case:
      sigma: 0.5
      distance_nm: 0.0
    measure: mutual_coherence
  - name: coherence_first_zero
    kind: special_case
    check: Evaluate the stated first mutual-coherence-zero distance for a conventional source and verify
      the coherence is near zero.
    expected: 0.0
    expected_formula: '0'
    tolerance: 0.003
    tolerance_mode: absolute
    reference: 'derivation: use the supplied first-zero relation d=0.61*lambda/(sigma*NA), substitute
      it into u=2*pi*sigma*NA*d/lambda, and evaluate 2*J1(u)/u; 0.61 is the standard rounded first-zero
      coefficient.'
    case:
      sigma: 0.5
      distance_nm: 174.76746666666668
    measure: mutual_coherence
  - name: source_weight_normalization
    kind: invariant
    check: Construct a conventional sigma=0.5 quadrature source and measure absolute source-weight-sum
      error after normalization.
    expected: 0.0
    expected_formula: '0'
    tolerance: 1.0e-12
    tolerance_mode: absolute
    reference: 'derivation: source quadrature weights are explicitly divided by their finite positive
      sum, therefore their sum is exactly one up to floating-point roundoff.'
    case:
      source_family: 0
      conventional_sigma: 0.5
    measure: abs(source_weight_sum-1)
  - name: mirror_source_symmetry
    kind: symmetry
    check: Apply source mirroring to an asymmetric 8×8 candidate and measure the maximum absolute mismatch
      of horizontally and vertically reflected output weights.
    expected: 0.0
    expected_formula: '0'
    tolerance: 1.0e-12
    tolerance_mode: absolute
    reference: 'derivation: the mirror projection replaces every orbit of reflected pixels by its arithmetic
      mean, making each reflected pair equal apart from floating-point roundoff.'
    case:
      source_pixel_grid: 8
    measure: source_mirror_symmetry_error
  - name: baseline_source_validity
    kind: invariant
    check: Evaluate the baseline 8×8 source design and measure weight sum deviation from 1.0 and mirror
      symmetry error.
    expected: 0.0
    expected_formula: '0'
    tolerance: 1.0e-12
    tolerance_mode: absolute
    reference: 'derivation: the baseline design must strictly meet all optimization constraints prior
      to search.'
    case:
      source_family: baseline_pixelated
      source_pixel_grid: 8
    measure: abs(baseline_weight_sum-1)+baseline_mirror_symmetry_error
  - name: abbe_hopkins_agreement
    kind: second_implementation
    check: For a 200 nm pitch, sigma=0.3 conventional source, zero defocus, and 1 nm grid, compare independently
      implemented Abbe source summation against Hopkins TCC contraction after maximum-intensity normalization.
    expected: 0.0
    expected_formula: '0'
    tolerance: 0.002
    tolerance_mode: absolute
    reference: 'Abbe implementation: coherent field is formed separately for every sampled source point
      and intensities are weighted and summed. Hopkins implementation: independently construct TCC(f1,f2)=sum_s
      w_s P(f1+s)P*(f2+s) and contract it with mask Fourier coefficients; no image-field code is shared.'
    case:
      pitch_nm: 200.0
      conventional_sigma: 0.3
      defocus_nm: 0.0
      fft_dx_nm: 1.0
      source_quadrature: 121
    measure: abbe_hopkins_max_abs_error
  simplifications:
  - id: X1
    replaces: Full electromagnetic high-NA immersion imaging with polarization
    because: The topic explicitly fixes scalar imaging with no polarization and requests a CPU-feasible
      experiment.
    effect: Contrast and DOF may be overstated and source ranking could change under vector polarization.
    probe: Repeat selected best and baseline conditions in a future vector model and compare rank order
      and DOF.
  - id: X2
    replaces: A calibrated resist and develop model
    because: The topic specifies constant-threshold aerial imaging with no resist blur.
    effect: Absolute CDs, exposure latitude, MEEF, side-lobe printability, and DOF are optical proxies
      rather than wafer predictions.
    probe: Apply a resist blur/development transfer model to the retained source candidates and measure
      rank stability.
  - id: X3
    replaces: Continuous source distributions and arbitrary free-form pupil resolution
    because: Finite CPU time requires source quadrature and an 8×8 pixelated free-form representation.
    effect: Narrow pupil features and the exact optimum may be missed.
    probe: Re-evaluate the selected free-form source with 289 samples and compare against a 12×12 source-pixel
      refinement if resources permit.
  - id: X4
    replaces: Continuous pitch, neighbour-distance, focus, and dose spaces
    because: The required CPU study must use explicit finite grids, with refinement around observed dips
      and roots.
    effect: Forbidden-pitch and ambit distances have discretization uncertainty.
    probe: Locally halve pitch/neighbour/focus increments around each reported extremum and report the
      change.
  - id: X5
    replaces: All possible quasar implementations
    because: Quasar is operationally represented as four equal annular sectors centered on diagonal azimuths
      with fixed 20-degree half-sector angle and optimized radial bounds from the annular grid.
    effect: A differently parameterized quasar could perform better.
    probe: Sweep sector angle and azimuth offsets after the primary comparison if the fixed quasar is
      competitive.
  - id: X6
    replaces: Exact unbounded isolated feature
    because: The topic specifies pitch and neighbour separation through 2000 nm, so 2000 nm is the fixed
      isolated reference.
    effect: A residual long-range optical interaction below the numerical threshold may remain.
    probe: Double the computational window/reference separation and quantify the central-CD change.
  - id: X7
    replaces: Exact Bessel-function validation under the restricted pre-run expression language
    because: The supplied analytic coherence function contains J1, which is unavailable in the required
      closed-form oracle formula language.
    effect: The first-zero oracle uses the supplied 0.61 approximation and a finite tolerance.
    probe: Compare scipy.special.j1 coherence values with high-precision mpmath values in a separate numerical
      audit.
  - id: X8
    replaces: Full source-mask co-optimization
    because: The chosen direction is symmetry-constrained free-form source optimization; mask correction
      is limited to independently solved 1D linewidth OPC.
    effect: Achievable absolute process window can be lower than an SMO result with SRAFs or freeform
      mask geometry.
    probe: Run a follow-up joint source/mask optimization on the held-out worst pitch and compare gain
      attribution.
  - id: X9
    replaces: Continuous pitch coverage, including exact locations of every forbidden pitch
    because: CPU-only evaluation uses a finite through-pitch grid and locally refined points.
    effect: Reported pitch-dependent DOF, forbidden-pitch locations, and per-pitch OPC metrics can shift
      by the pitch-grid resolution.
    probe: Halve the pitch increment around each reported forbidden-pitch minimum and compare its location
      and all reported per-pitch metrics.
  - id: X10
    replaces: Continuous conventional sigma, annular radial-bound, x-dipole pole-centre and pole-radius,
      and quasar parameter spaces
    because: Conventional sigma, annular, x-dipole, and quasar sources must be represented by finite CPU-feasible
      parameter grids before selecting each family comparator.
    effect: The selected conventional, annular, x-dipole, or quasar optimum may differ from the continuous
      family optimum.
    probe: Insert midpoint parameter values around each selected conventional sigma, annular bounds, and
      x-dipole pole centre and pole radius and compare common DOF.
  - id: X11
    replaces: Two independent full-resolution image calculations for every production source–pitch–focus
      evaluation
    because: Hopkins TCC contraction is computationally expensive, so Abbe produces the sweep data and
      Hopkins is used for the independent agreement checks and retained comparisons.
    effect: A defect that appears only outside the checked conditions could affect unverified sweep images.
    probe: Recompute representative retained source–pitch–focus cases with Hopkins and report their Abbe–Hopkins
      discrepancy.
  - id: X12
    replaces: General OPC correction geometry and process-calibrated mask bias
    because: The topic limits correction to a one-dimensional mask-linewidth root solve in a scalar, constant-threshold
      aerial-image model.
    effect: The mask bias each pitch needs is a 1D optical proxy and can differ from production OPC bias.
    probe: Re-evaluate retained pitches with a calibrated OPC/resist model and compare solved mask biases.
  - id: X13
    replaces: Process-calibrated iso-dense bias before OPC across arbitrary feature contexts
    because: Pre-OPC iso-dense bias is operationally limited to the difference between 40 nm-mask isolated
      and densest periodic features in the stated 1D model.
    effect: The reported iso-dense bias can omit context, resist, and two-dimensional contributions.
    probe: Repeat the pre-OPC comparison for intermediate pitches and a calibrated resist model.
  - id: X14
    replaces: Resist-printed attenuated-mask side-lobe behaviour
    because: Side-lobe printing is classified only by an off-target scalar aerial-image maximum reaching
      the calibrated constant threshold.
    effect: The Boolean side-lobe result can differ after resist blur, development, and stochastic effects.
    probe: Apply a resist transfer model to every threshold-reaching side lobe and compare the print classification.
  - id: X15
    replaces: Continuous movement of an isolated line's neighbour and an asymptotic optical-ambit measurement
    because: The neighbour is moved on a finite 80–2000 nm distance grid, with 2000 nm used as the isolated
      reference and a 0.1 nm CD-change rule.
    effect: The CD-change curve and the distance beyond which it remains below 0.1 nm have neighbour-grid
      and finite-reference uncertainty.
    probe: Halve the neighbour-distance spacing around the reported 0.1 nm crossing and double the isolated
      reference distance.
  - id: X16
    replaces: A continuous or noise-calibrated definition of how far CD oscillation remains visible
    because: Visibility is operationally defined by the finite-grid largest distance with local CD oscillation
      amplitude exceeding 0.02 nm.
    effect: The visible-oscillation distance depends on the numerical CD uncertainty and the selected
      amplitude threshold.
    probe: Repeat the ambit curve at finer grid and source sampling and report the change in the last
      above-0.02 nm distance.
  - id: X17
    replaces: Exact pupil-phase propagation through the physical water-resist interface
    because: The supplied defocus phase is evaluated in one homogeneous resist-side medium of index 1.436,
      without interface refraction, thin-film propagation, or standing waves.
    effect: Defocus-dependent contrast, CD, and DOF can differ from an interface-aware immersion/resist
      calculation.
    probe: Recompute selected source, pitch, and focus conditions with an interface-aware vector or thin-film
      propagation model and compare CD and DOF changes.
  - id: X18
    replaces: Direct mutual-coherence calculation for noncircular, annular, dipole, quasar, and pixelated
      sources
    because: The supplied analytic expression 2J1(u)/u applies only to a circular source, while noncircular
      source coherence is assessed from their simulated CD-change curves rather than an analytic closed
      form.
    effect: Agreement between optical-ambit oscillations and the circular-source coherence function is
      only a conventional-source comparison and cannot be generalized quantitatively to other source shapes.
    probe: Numerically Fourier-transform each retained source distribution and compare its calculated
      mutual coherence with the corresponding ambit curve.
  - id: X19
    replaces: Exact mutual-coherence zero locations for all source shapes
    because: The first-zero relation 0.61*lambda/(sigma*NA) is supplied for a circular source and is used
      only as its rounded conventional-source reference.
    effect: Zero crossings of CD oscillations for annular, dipole, quasar, and free-form sources need
      not coincide with this circular-source prediction.
    probe: Determine zeros from each source's numerical mutual-coherence function and report their displacement
      from CD-oscillation zero crossings.
  - id: X20
    replaces: Unbounded computation with exact runtime and zero numerical error
    because: The study is CPU-only and reports finite-grid, finite-source-sampling, and finite-focus-step
      results with recorded wall-clock runtime.
    effect: The selected optimum, reported process windows, and runtime can change with hardware, implementation,
      and numerical refinement.
    probe: Repeat retained calculations at the stated finer numerical settings, report maximum metric
      changes and elapsed time, and compare them with the primary results.
  definitions:
  - metric: dof_nm
    defined_as: The contiguous focus width centered on nominal focus for which CD remains within ±10%
      of 40 nm at dose scales 0.95, 1.00, and 1.05.
    source: '[26]'
    quote: Illumination and mask enhancements improve aerial image quality and serve to maintain that
      improvement over a suitably larger range of defocus than would otherwise be obtained.
    differs: X2
  - metric: common_dof_nm
    defined_as: The minimum usable DOF among dense, semi-dense, forbidden-pitch, and isolated representative
      conditions.
    source: '[2]'
    quote: It is shown that a proper source with a proper mask optimization strategy can be obtained,
      which gives enough common process window on most of the test pattern.
    differs: X2
  - metric: nils
    defined_as: Magnitude of the aerial-image log slope normalized by threshold intensity at a printed
      edge.
    source: '[14]'
    quote: We utilize the normalized image log slope (NILS) metric, which is available at no extra computational
      cost during OPC.
    differs: X2
  - metric: meef
    defined_as: Absolute local derivative of printed wafer CD with respect to mask CD, evaluated by ±1
      nm central difference.
    source: '[38]'
    quote: 8.7 Mask Error Enhancement Factor (MEEF).
    differs: X2
  - metric: mask_bias_nm
    defined_as: Solved OPC mask linewidth minus the 40 nm target wafer linewidth.
    source: '[11]'
    quote: The flow of the MB-OPC includes the model data collection, model setup and calibration, recipe
      setup, OPC correction and post-OPC verify check.
    differs: X2
  - metric: iso_dense_bias_nm
    defined_as: Printed CD for a 40 nm isolated-reference mask feature minus printed CD for a 40 nm densest-pitch
      mask feature before OPC.
    source: '[26]'
    quote: The more isolated features are unaffected by this approach, so common process latitude is reduced
      since the overlap of the individual windows is driven apart by the difference of sizing energies
      of the features.
    differs: X2
  - metric: side_lobe_prints
    defined_as: A Boolean indicating whether an off-target aerial-image local maximum reaches or exceeds
      the calibrated printing threshold.
    source: '[W3]'
    quote: The experimental layout optimization is discussed, showing how the process is maximized together
      with the dose-margin before any side-lobe printing.
    differs: X2
  - metric: forbidden_pitch
    defined_as: A local through-pitch DOF minimum satisfying the protocol’s depth criterion, verified
      across the 15%–30% threshold range.
    source: '[2]'
    quote: few-line pattern like 3-line, 4-line or 5-line which have some special pitch and are common
      in real chip, are very hard to print with enough process window even with SMO technique.
    differs: X4
  - metric: cd_change_nm
    defined_as: Central-line CD at a stated neighbour separation minus central-line CD using the 2000
      nm neighbour reference.
    source: '[35]'
    quote: The theory is applied to study inter alia the influence of the method of illumination on the
      images of simple periodic structures and of an isolated line.
    differs: X2
  - metric: ambit_distance_nm
    defined_as: The first tested neighbour distance after which every larger tested separation changes
      central CD by less than 0.1 nm.
    source: '[35]'
    quote: The theory is applied to study inter alia the influence of the method of illumination on the
      images of simple periodic structures and of an isolated line.
    differs: X4
  - metric: visible_oscillation_distance_nm
    defined_as: The maximum neighbour separation at which the peak-to-peak CD variation around the isolated
      limit exceeds 0.02 nm.
    source: '[35]'
    quote: The theory is applied to study inter alia the influence of the method of illumination on the
      images of simple periodic structures and of an isolated line.
    differs: X4
  - metric: mutual_coherence
    defined_as: For a circular source, 2J1(u)/u with u=2πσNAd/λ and value one at d=0.
    source: derivation
    quote: ''
    differs: X7
  - metric: abbe_hopkins_max_abs_error
    defined_as: Maximum absolute normalized aerial-image difference between independently evaluated Abbe
      and Hopkins formulations.
    source: '[35]'
    quote: The theory of image formation is formulated in terms of the coherence function in the object
      plane, the diffraction distribution function of the image-forming system and a function describing
      the structure of the object.
    differs: ''
  optimisation:
    objective:
      quantity: common_dof_nm
      direction: maximise
      unit: nm
      meaning: Worst usable depth of focus across dense, semi-dense, forbidden-pitch, and isolated representative
        patterns.
    design_variables:
    - name: source_pixels
      shape:
      - 8
      - 8
      kind: continuous
      low: 0.0
      high: 1.0
    fixed:
      representative_dense_pitch_nm: 80.0
      representative_semi_dense_pitch_nm: 160.0
      representative_forbidden_pitch_nm: 120.0
      representative_isolated_pitch_nm: 2000.0
      target_cd_nm: 40.0
      wavelength_nm: 193.368
      na: 1.35
      water_refractive_index: 1.436
      attenuated_mask_intensity_transmission: 0.06
      attenuated_mask_phase_shift_rad: 3.141592653589793
    constraints:
    - quantity: source_pixel_nonnegativity
      limit: '>= 0'
      unit: dimensionless
    - quantity: source_weight_sum
      limit: '>= 0.999999999999'
      unit: dimensionless
    - quantity: source_weight_sum
      limit: <= 1.000000000001
      unit: dimensionless
    - quantity: mirror_symmetry_error
      limit: <= 1e-12
      unit: dimensionless
    - quantity: pupil_boundary_outer_weight
      limit: <= 0
      unit: dimensionless
    baseline:
      values:
        source_pixels:
        - - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.125
          - 0.125
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.125
          - 0.0
          - 0.0
          - 0.125
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.125
          - 0.0
          - 0.0
          - 0.125
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.125
          - 0.125
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
          - 0.0
      source: Standard mirror-symmetric four-pole x-dipole initialization projected to the 8×8 source
        grid and normalized to unit sum; its geometry follows the source families considered in [10].
    numerical_settings:
      fft_dx_nm:
        search: 1.0
        check:
        - 0.5
        - 0.25
        finer: smaller
      source_quadrature:
        search: 121
        check:
        - 289
        - 625
        finer: larger
      focus_step_nm:
        search: 25.0
        check:
        - 12.5
        - 6.25
        - 2.0
        finer: smaller
    evaluation_budget:
      starts: 8
      per_start: 180
      seconds_per_evaluation: 20
    search_method: global_then_local
    improvement_tolerance:
      value: 1.0
      mode: absolute
  criteria:
  - name: Abbe-Hopkins consistency
    what: The independent partially coherent image formulations must agree on the fixed small validation
      case.
    oracle: abbe_hopkins_agreement
    use: error
    direction: lower
    target: 0.002
    tolerance: 0.0005
  - name: source normalization
    what: All normalized source constructions must conserve total source weight.
    oracle: source_weight_normalization
    use: value
    direction: lower
    target: 1.0e-12
    tolerance: 5.0e-13
  - name: freeform symmetry
    what: Projection of a candidate free-form source must enforce the prescribed mirror symmetry.
    oracle: mirror_source_symmetry
    use: value
    direction: lower
    target: 1.0e-12
    tolerance: 5.0e-13
  - name: coherence zero
    what: The implemented circular-source coherence must be close to zero at the supplied first-zero distance.
    oracle: coherence_first_zero
    use: error
    direction: lower
    target: 0.003
    tolerance: 0.001
  precision:
    target_half_width: 0.5
    metric: usable DOF boundary location
    reason: Bracketed focus-boundary root solving reports each usable-DOF boundary to a numerical half-width
      no greater than 0.5 nm.
  seed_policy: The physical forward model is deterministic. A fixed base seed 20261013 controls only reproducible
    Arbor candidate initialization and scipy differential-evolution initialization; every optimization
    branch receives a stream derived from the base seed and its branch identifier. No stochastic physical
    trials are pooled.
  failure_policy: A failed FFT/image evaluation, nonfinite intensity, missing threshold bracket, or unconverged
    OPC root is recorded with source, pitch, focus, dose, and mask-width identifiers. It is classified
    as unprintable for that condition, never silently excluded, and source-level summaries report the
    count.
  ci_method: Deterministic numerical uncertainty is reported as the maximum absolute change between search
    resolution and each finer check resolution for each reported metric; source-family discretization
    uncertainty is half the adjacent-parameter spacing at the selected optimum. No sampling confidence
    interval is claimed.
  model:
    summary: A scalar, one-dimensional, partially coherent Fourier-optical model propagates a thin attenuated-PSM
      spectrum through a defocused circular pupil and computes aerial intensity by independent Abbe and
      Hopkins formulations.
    assumptions:
    - Kirchhoff thin-mask approximation with dark-mask complex amplitude sqrt(0.06)*exp(i*pi).
    - Scalar imaging with no polarization despite NA 1.35.
    - Uniform incoherent mixing among sampled source points.
    - Aerial-image constant threshold represents printing; no resist blur or development model is included.
    - The water/resist-side medium has refractive index 1.436 for the supplied defocus phase.
    holds_for: Comparative 1D scalar simulations of vertical 40 nm target lines/spaces at wavelength 193.368
      nm, NA 1.35, source support sigma<=0.95, pitches 80–2000 nm, and finite sampled focus range; it
      does not provide vector or wafer-calibrated predictions.
    equations:
    - id: E1
      formula: t_dark = sqrt(T_dark)*exp(i*phi_dark)
      role: generates
      source: derivation
      derivation: The topic fixes dark-region intensity transmission T_dark=0.06 and phase shift phi_dark=pi.
        Field amplitude is the square root of intensity transmission, and a phase shift multiplies a complex
        field by exp(i*phase).
      example:
        inputs:
          T_dark: 0.06
          phi_dark: 3.141592653589793
        expected_formula: sqrt(T_dark)
    - id: E2
      formula: pupil_phase_rad = 2*pi/lambda_nm*z_nm*(n_medium-sqrt(n_medium**2-(lambda_nm*f_cpnm)**2))
      role: generates
      source: derivation
      derivation: This is the defocus pupil phase supplied in the topic, written with wavelength and spatial
        frequency in nm and cycles per nm so that lambda_nm*f_cpnm is dimensionless.
      example:
        inputs:
          lambda_nm: 193.368
          z_nm: 100.0
          n_medium: 1.436
          f_cpnm: 0.001
        expected_formula: 2*pi/lambda_nm*z_nm*(n_medium-sqrt(n_medium**2-(lambda_nm*f_cpnm)**2))
    - id: E3
      formula: gamma = 2*J1(u)/u, u = 2*pi*sigma*NA*distance_nm/lambda_nm; gamma=1 when u=0
      role: analyses
      source: derivation
      derivation: The topic supplies the circular-source mutual-coherence function. Its u=0 value follows
        from the Bessel-series limit J1(u)=u/2+O(u**3).
      example:
        untestable: The planned implementation uses scipy.special.j1, while the pre-run formula language
          does not include J1.
    - id: E4
      formula: coherence_first_zero_nm = 0.61*lambda_nm/(sigma*NA)
      role: analyses
      source: derivation
      derivation: This is the supplied first-zero approximation for the circular-source coherence function.
      example:
        inputs:
          lambda_nm: 193.368
          sigma: 0.5
          NA: 1.35
        expected_formula: 0.61*lambda_nm/(sigma*NA)
    - id: E5
      formula: dipole_pole_sigma = lambda_nm/(2*pitch_nm*NA)
      role: analyses
      source: derivation
      derivation: This is the supplied symmetric two-beam dipole-pole relation for a dense pitch.
      example:
        inputs:
          lambda_nm: 193.368
          pitch_nm: 100.0
          NA: 1.35
        expected_formula: lambda_nm/(2*pitch_nm*NA)
    - id: E6
      formula: f_max_cpnm = NA*(1+sigma_max)/lambda_nm
      role: analyses
      source: derivation
      derivation: 'This is the supplied maximum transmitted spatial-frequency relation: frequencies above
        it never reach the wafer.'
      example:
        inputs:
          NA: 1.35
          sigma_max: 0.95
          lambda_nm: 193.368
        expected_formula: NA*(1+sigma_max)/lambda_nm
    - id: E7
      formula: I_Abbe(x) = sum_s w_s*abs(IFFT(P(f+s)*M(f)))**2
      role: generates
      source: '[28]'
      derivation: ''
      example:
        untestable: The FFT-based field and finite sampled source sum have no scalar closed-form output
          for the nontrivial mask used in the experiment.
      method: Discrete FFT on a periodic 1D mask window with finite source-point quadrature and pupil
        phase from E2.
      tolerance: 0.002
      tolerance_mode: absolute
    - id: E8
      formula: TCC(f1,f2) = sum_s w_s*P(f1+s)*conj(P(f2+s)); I_Hopkins(x) = sum_f1,sum_f2 TCC(f1,f2)*M(f1)*conj(M(f2))*exp(i*2*pi*(f1-f2)*x)
      role: generates
      source: '[35]'
      derivation: ''
      example:
        untestable: The finite TCC contraction has no useful closed form for the nontrivial periodic mask
          and sampled extended source.
      method: Direct discrete TCC construction and Fourier contraction, independently coded from the Abbe
        path.
      tolerance: 0.002
      tolerance_mode: absolute
    - id: E9
      formula: NILS = abs((1/I_threshold)*(dI/dx)_edge)
      role: analyses
      source: derivation
      derivation: NILS is defined operationally as the magnitude of normalized aerial-image log slope
        at the printing threshold; dI/dx is evaluated by centered finite difference at the interpolated
        threshold edge.
      example:
        inputs:
          I_threshold: 0.5
          dI_dx: 0.02
        expected_formula: abs(dI_dx/I_threshold)
    - id: E10
      formula: MEEF = abs((CD_wafer(mask_cd_nm+delta_nm)-CD_wafer(mask_cd_nm-delta_nm))/(2*delta_nm))
      role: analyses
      source: derivation
      derivation: MEEF is evaluated as the magnitude of the local finite-difference wafer-CD response
        to mask-CD change, using the fixed ±1 nm perturbation.
      example:
        inputs:
          cd_plus_nm: 42.0
          cd_minus_nm: 38.0
          delta_nm: 1.0
        expected_formula: abs((cd_plus_nm-cd_minus_nm)/(2*delta_nm))
    - id: E11
      formula: common_dof_nm = min(DOF_dense_nm, DOF_semi_dense_nm, DOF_forbidden_nm, DOF_isolated_nm)
      role: analyses
      source: derivation
      derivation: The selected source objective is the worst usable DOF among the fixed representative
        conditions, so maximizing it is a maximin common-window search.
      example:
        inputs:
          DOF_dense_nm: 40.0
          DOF_semi_dense_nm: 32.0
          DOF_forbidden_nm: 18.0
          DOF_isolated_nm: 25.0
        expected_formula: min(DOF_dense_nm,DOF_semi_dense_nm,DOF_forbidden_nm,DOF_isolated_nm)
    steps:
    - id: S1
      name: Construct periodic mask
      does: Creates a 1D periodic complex thin-PSM line/space mask at the specified pitch and mask linewidth.
      inputs:
        pitch_nm: nm
        mask_cd_nm: nm
      output: complex mask transmission samples
      uses:
      - E1
      source: derivation
      relations:
      - The mask repeats exactly after pitch_nm.
      - Dark-region amplitude magnitude is sqrt(0.06).
    - id: S2
      name: Generate normalized source
      does: Creates conventional, annular, x-dipole, quasar, or projected mirror-symmetric pixelated source
        quadrature points and weights.
      inputs:
        source_family: categorical
        source_parameters: dimensionless
      output: source-point coordinates and dimensionless normalized weights
      uses: []
      source: derivation
      relations:
      - All weights are nonnegative.
      - Weights sum to one.
      - Free-form source has horizontal and vertical mirror symmetry.
    - id: S3
      name: Compute Abbe image
      does: Propagates the mask spectrum through a shifted defocused pupil for each source point and sums
        weighted coherent intensities.
      inputs:
        mask: complex transmission samples
        source: source-point coordinates and dimensionless weights
        defocus_nm: nm
      output: Abbe aerial intensity versus position
      uses:
      - E2
      - E7
      source: '[28]'
    - id: S4
      name: Compute Hopkins image
      does: Builds the source-weighted transmission cross coefficient and contracts it with the same mask
        spectrum.
      inputs:
        mask_spectrum: complex Fourier mask coefficients
        source: source-point coordinates and dimensionless weights
        defocus_nm: nm
      output: Hopkins aerial intensity versus position
      uses:
      - E2
      - E8
      source: '[35]'
    - id: S5
      name: Calibrate dose and extract CD
      does: Sets the global aerial-image threshold/dose so the densest pitch prints 40 nm at nominal focus,
        then finds threshold edges by bracketed interpolation.
      inputs:
        aerial_image: normalized intensity
        target_cd_nm: nm
      output: CD in nm and side-lobe indicator
      uses: []
      source: derivation
      relations:
      - The calibrated dense nominal CD equals 40 nm within one spatial grid interval.
      - A missing central threshold bracket is recorded as unprintable.
    - id: S6
      name: Solve one-dimensional OPC
      does: Uses a bracketing root solve over mask linewidth to produce a 40 nm nominal best-focus wafer
        CD for each source and pitch.
      inputs:
        source: illumination definition
        pitch_nm: nm
        target_cd_nm: nm
      output: OPC mask width and mask bias in nm
      uses: []
      source: derivation
      relations:
      - The accepted root prints 40 nm within one spatial-grid interval.
      - Unbracketed roots are recorded as OPC failure rather than extrapolated.
    - id: S7
      name: Evaluate process-window metrics
      does: Sweeps focus and dose to calculate usable DOF, NILS, MEEF, iso-dense bias, side-lobe printing,
        and forbidden-pitch candidates.
      inputs:
        source: illumination definition
        pitch_nm: nm
        opc_mask_width_nm: nm
      output: through-pitch metric record
      uses:
      - E9
      - E10
      - E11
      source: derivation
    - id: S8
      name: Measure optical ambit
      does: Moves one neighbour of a central isolated 40 nm line through the fixed distance grid and compares
        its CD change to mutual coherence and its first zero.
      inputs:
        source: illumination definition
        neighbour_distance_nm: nm
      output: CD-change and coherence curves plus ambit distances
      uses:
      - E3
      - E4
      source: derivation
    - id: S9
      name: Optimize free-form source
      does: Uses Arbor-managed global candidate branches and local refinement over 8×8 pixel weights,
        projecting each candidate to nonnegative mirror symmetry, zeroing weights outside sigma 0.95,
        and maximizing common DOF subject to source normalization.
      inputs:
        source_pixels: dimensionless
        representative_pitches_nm: nm
      output: optimized source weights and optimization trace
      uses:
      - E11
      source: '[10]'
```
