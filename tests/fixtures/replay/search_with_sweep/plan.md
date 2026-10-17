# Plan: TOPIC. For ArF immersion lithography at low k1, under which illumination sources do one-dimensional lines and spaces keep a usable depth of focus thro

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

The study will determine whether a constrained free-form illumination source creates a larger shared low-k1 process window than the strongest screened conventional, annular, dipole, or quasar source. It will validate Abbe against Hopkins imaging, optimize the free-form source on representative pitches, then report pitch-resolved OPC metrics and optical-ambit behavior with refinement uncertainty.

## What the literature says

- **[3] Gradient-based source mask and polarization optimization with the hybrid Hopkins–Abbe model**: Joint source-related optimization can improve process window, MEEF, and NILS, and Hopkins/Abbe formulations can support efficient optimization.
- **[4] Mutual optimization of resolution enhancement techniques**: Custom illumination, attenuated PSM, and OPC should be understood jointly in frequency space rather than as isolated techniques.
- **[7] Mask 3D effect on 45-nm imaging using attenuated PSM**: Attenuated-PSM dense imaging is sensitive to mask bias, illumination, MEEF, and NILS, while also warning that Kirchhoff mask modeling has limits.
- **[13] ArF Photoresist Parameter Optimization for Mask Error Enhancement Factor Reduction**: MEEF worsens as k1 and pitch decrease and depends on illumination as well as process conditions.
- **[15] Spatial frequency analysis of optical lithography resolution enhancement techniques**: Off-axis illumination and phase-shift methods extend useful spatial-frequency coverage and enable lower-k1 imaging.
- **[20] Electromagnetic diffraction in optical systems, II**: Scalar diffraction is the limiting low-aperture approximation; high-NA fields can require vector treatment.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] Projection optical lithography (DOI 10.1016/s1369-7021(05)00698-x)
- [2] Advanced lithographic patterning technologies: Materials and processes
- [3] Gradient-based source mask and polarization optimization with the hybrid Hopkins–Abbe model (DOI 10.1117/1.jmm.19.3.033201)
- [4] Mutual optimization of resolution enhancement techniques (DOI 10.1117/1.1450596)
- [5] Evaluating the Process Performances of Binary, PSM and OMOG Masks in Advanced Technology Node (DOI 10.1109/iwaps51164.2020.9286801)
- [6] 300 mm Large Area Wire Grid Polarizers with 50 nm Half-Pitch by ArF Immersion Lithography (DOI 10.3390/nano12030481)
- [7] Mask 3D effect on 45-nm imaging using attenuated PSM (DOI 10.1117/12.710549)
- [8] Performance study of chromeless phase lithography mask for the 65nm node and beyond (DOI 10.1117/12.656915)
- [9] Resolution enhancement techniques for optical lithography
- [10] Hp45 lithography in consideration of the mask 3D effect (DOI 10.1117/12.681882)
- [11] Study of the mask materials for PTD process and NTD process in practical ArF immersion lithography (DOI 10.1117/12.2070056)
- [W1] Pulse stretcher for ArF immersion lithography reduces speckle | Laser Focus World
- [12] Mask enhancement factor for 2D local CD error (DOI 10.1117/12.617209)
- [W2] ArF Immersion Lithography: Physics, Process Principles, and ...
- [W3] Advances in Light Sources: ArF immersion lithography: Understanding light source performance | Laser Focus World
- [13] ArF Photoresist Parameter Optimization for Mask Error Enhancement Factor Reduction (DOI 10.1143/jjap.44.7404)
- [14] Relationship between Normalized Image Log Slope and Exposure Latitude in advanced technodes (DOI 10.1109/iwaps57146.2022.9972325)
- [15] Spatial frequency analysis of optical lithography resolution enhancement techniques (DOI 10.1116/1.590669)
- [16] Mask optimization approach for wafer LCDU improvement in ArF lithography (DOI 10.1117/12.2688551)
- [17] Resolution enhancement techniques in optical lithography: It's not just a mask problem (DOI 10.1117/12.438332)
- [W4] Argon fluoride laser - Wikipedia
- [18] Fundamental Principles of Optical Lithography (DOI 10.1002/9780470723876)
- [19] Principles of Optics (DOI 10.1017/9781108769914)
- [20] Electromagnetic diffraction in optical systems, II. Structure of the image field in an aplanatic system (DOI 10.1098/rspa.1959.0200)
- [21] Introduction to Fourier optics
- [22] Resolution Enhancement Techniques in Optical Lithography (DOI 10.1117/3.401208)
- [23] Binary and phase shifting mask design for optical lithography (DOI 10.1109/66.136275)
- [24] Pixel-based OPC optimization based on conjugate gradients (DOI 10.1364/oe.19.002165)
- [25] Robust level-set-based inverse lithography (DOI 10.1364/oe.19.005511)
- [26] Mask Design for Optical Microlithography—An Inverse Imaging Problem (DOI 10.1109/tip.2006.891332)

## The gap this experiment addresses

The retrieved material supports source optimization and low-k1 RET tradeoffs but does not provide a directly comparable scalar ArF-immersion calculation of common DOF, pitch-resolved OPC metrics, and neighbour optical ambit for a free-form source versus a rigorously screened parametric source under the specified 6% attenuated mask.

## What is being optimised

> Shown from the design block below (`study_type` and `protocol.optimisation`) as it was when the plan was written; edit the block, not this section (after an edit, only the block counts). FI runs the search itself, within the budget below, and records every evaluation; the best design is found at the search's own numerical settings, then FI evaluates it and the baseline again at the finer check settings below, nudges it, and compares the starting points. A check that fails is reported as the result and limits the evidence level; it does not stop the quest.

**Kind of study:** find the best design (`study_type: find_best_design`), not a measurement over settings chosen in advance.

**Goal:** make common_dof_nm (The minimum usable DOF over dense, semi-dense, deterministically selected forbidden, and isolated representative pitches.) as high as possible, in nm.

**What the search may change:**

- source_sector_weight_0: 0 to 1 dimensionless
- source_sector_weight_1: 0 to 1 dimensionless
- source_sector_weight_2: 0 to 1 dimensionless
- source_sector_weight_3: 0 to 1 dimensionless
- source_sector_weight_4: 0 to 1 dimensionless
- source_sector_weight_5: 0 to 1 dimensionless
- source_sector_weight_6: 0 to 1 dimensionless
- source_sector_weight_7: 0 to 1 dimensionless

**Held fixed:** source_pixel_grid = 11; mirror_symmetry = True; representative_dense_pitch_nm = 80; representative_semidense_pitch_nm = 120; representative_isolated_pitch_nm = 2000; representative_forbidden_pitch_rule = lowest-DOF pitch in the adaptively refined 80–300 nm screening sweep for the fine-grid selected parametric source; ties select the smaller pitch; freeze this pitch before free-form optimization; source_normalization = eight equal-area, mirror-symmetric sectors partition the sigma<=0.9 source disk; each sector weight is its fraction of total source power, the eight weights sum to 1, and source pixels within a sector receive equal area-weighted intensity

**Limits every design must meet:** source_pixel_weight ≥ 0 dimensionless. A second goal is written as one of these limits: there is one goal.

**The design to beat (baseline):** source_sector_weight_0 = 0.125; source_sector_weight_1 = 0.125; source_sector_weight_2 = 0.125; source_sector_weight_3 = 0.125; source_sector_weight_4 = 0.125; source_sector_weight_5 = 0.125; source_sector_weight_6 = 0.125; source_sector_weight_7 = 0.125. From: Predeclared feasible conventional circular sigma-0.9 illumination, a member of the conventional screening grid, represented by equal power in eight equal-area sectors. The headline comparator is separately selected as the best tuned member of the full parametric screen.

**When a design counts as better:** better than the baseline by more than 10% (the plan sets this).

**How the search runs:** bounded_local: a local search inside the ranges, from several starting points (FI’s choice: the plan names no method).

**Numerical settings (the search's, and the finer ones the best design and the baseline are recomputed at, so an improvement that is only a numerical error does not count):**

- wafer_grid_spacing_nm: the search uses 1; the check uses 0.5, then 0.25 (the plan's values)
- source_samples_per_axis: the search uses 17; the check uses 25, then 33 (the plan's values)
- focus_step_nm: the search uses 5; the check uses 2, then 1 (the plan's values)

**How many evaluations (worked out now, before anything runs):**

- search: 12 starting points × 80 = 960 evaluations, at most (a limit, not a target)
- check: (3 best designs + the baseline) × 2 finer levels + 2 × 8 nudges around the best design = 24 evaluations at most, at the finer settings (each costs more than a search evaluation); not taken from the search's budget, and with its own time limit (`execution.timeout_s`)
- time: about 2.1 h for the scan and the search (8 s per evaluation, the plan's own estimate; not measured yet)
- the scan and the search stop at the run's time limit (`execution.timeout_s`), whatever is left of the budget; the results say which came first

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** A scalar thin-mask Fourier-optics model forms partially coherent one-dimensional aerial images by Abbe source integration and independently by Hopkins TCC summation, followed by threshold-contour process-window extraction.

**What it assumes:** The projection system is shift invariant, scalar, and aberration free apart from specified defocus.; The 6% attenuated dark mask region has amplitude sqrt(0.06) and phase pi.; Source points are mutually incoherent and source weights are nonnegative.; The resist is represented only by a constant aerial-image threshold.

**Where it holds:** The stipulated comparative scalar, one-dimensional thin-mask model over 80–2000 nm pitch, source coordinates no larger than sigma 1, and defocus between -500 and 500 nm. It does not predict polarization, mask-3D, or resist effects at NA 1.35; physical contrast and DOF are likely overestimated.

**Its equations:**

- **E1** (produces the data): `phi_defocus_rad = (2*pi/lambda_nm)*z_nm*(n-sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2))`. Source: derivation: The supplied resist-side pupil phase is the axial wave-vector difference multiplied by propagation distance. 'k_z=(2*pi/lambda_nm)*sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2)' and 'k_0=(2*pi/lambda_nm)*n', so 'phi_defocus_rad=z_nm*(k_0-k_z)=(2*pi/lambda_nm)*z_nm*(n-sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2))'.
- **E2** (produces the data): `gamma = 2*J1(u)/u; u = 2*pi*sigma*NA*separation_nm/lambda_nm; gamma(0)=1`. Source: derivation: Use the supplied mutual-coherence form for a circular source. The u=0 branch is evaluated from the Bessel-series limit 'J1(u)=u/2+O(u^3)', therefore 'gamma(0)=lim_(u->0) 2*J1(u)/u=1'.
- **E3** (produces the data): `I_Abbe(x,z) = sum_s w_s*abs(FFT_inverse[P(f+s)*M(f)*exp(i*phi_defocus_rad(f+s,z))])**2`. Source: derivation: For source point s, a mask frequency f occupies pupil frequency 'f_pupil=f+s'. Therefore 'H_s(f,z)=P(f+s)*exp(i*phi_defocus_rad(f+s,z))'; the coherent field is 'E_s(x,z)=FFT_inverse[H_s(f,z)*M(f)]', and incoherent source integration gives 'I_Abbe(x,z)=sum_s w_s*abs(E_s(x,z))**2'.
- **E4** (produces the data): `I_Hopkins(x,z) = sum_f1 sum_f2 M(f1)*conj(M(f2))*TCC_z(f1,f2)*exp(i*2*pi*(f1-f2)*x)`. Source: derivation: With 'H_s(f,z)=P(f+s)*exp(i*phi_defocus_rad(f+s,z))', expanding the Abbe coherent intensity gives 'TCC_z(f1,f2)=sum_s w_s*H_s(f1,z)*conj(H_s(f2,z))', hence 'I_Hopkins(x,z)=sum_f1 sum_f2 M(f1)*conj(M(f2))*TCC_z(f1,f2)*exp(i*2*pi*(f1-f2)*x)'.
- **E5** (analyses the results): `nils = target_cd_nm*abs((dI_dx)/I_threshold)`. Source: derivation: At threshold, the image log slope is '(1/I_threshold)*dI_dx'; normalizing its magnitude by the target CD gives 'nils=target_cd_nm*abs((dI_dx)/I_threshold)'.
- **E6** (analyses the results): `meef = abs((CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm))`. Source: derivation: The mask-width coordinate and 'mask_cd_step_nm' are wafer-plane equivalent. The centred derivative is '(CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm)', therefore 'meef=abs((CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm))'.

**The checks against known answers, and where each expected value comes from:**

- **uniform_clear_mask_intensity**, a special or limiting case with a known answer: expects 0; computed as `max_abs_intensity_error` from what the simulation returns on `mask_case=uniform_clear, source_case=on_axis, defocus_nm=137, wafer_grid_spacing_nm=1`; from: derivation: for t(x)=1, its only Fourier component is f=0; the unaberrated pupil transmits it with unit amplitude, so I(x)=|1|^2=1 and max_x abs(I(x)-1)=0.
- **defocus_phase_e1**, a special or limiting case with a known answer: expects 0; computed as `phi_defocus_rad` from what the simulation returns on `lambda_nm=193.368, z_nm=100, n=1.436, f_cycles_per_nm=0.002`; from: derivation: phi=(2*pi/193.368)*100*(1.436-sqrt(1.436^2-(193.368*0.002)^2)) by E1.
- **abbe_hopkins_agreement**, an independent second implementation: expects 0; computed as `abbe_hopkins_max_relative_difference` from what the simulation returns on `pitch_nm=200, source_case=single_point_sigma_x_0.5, defocus_nm=100, wafer_grid_spacing_nm=1, source_samples_per_axis=17, relative_difference=max_x(abs(I_Abbe(x)-I_Hopkins(x))/max(max(abs(I_Abbe(x)),abs(I_Hopkins(x))),1e-12))`; from: independent Abbe coherent-image summation versus independently assembled Hopkins TCC summation; the two implementations share only mask Fourier coefficients, source samples, and pupil parameters and share no image-formation or TCC-construction code.
- **mutual_coherence_at_zero_separation**, a special or limiting case with a known answer: expects 1; computed as `mutual_coherence` from what the simulation returns on `separation_nm=0, conventional_sigma=0.7`; from: derivation: J1(u)=u/2+O(u^3), hence lim_{u->0} 2*J1(u)/u=1.
- **mutual_coherence_first_zero**, a special or limiting case with a known answer: expects 0; computed as `mutual_coherence` from what the simulation returns on `separation_nm=124.786, conventional_sigma=0.7`; from: derivation: the first positive zero of J1 is u=3.8317059702075125; substituting u into 2*J1(u)/u gives zero, and the case separation rounds the corresponding distance.
- **feasible_baseline_source_power**, a conserved quantity or other invariant: expects 0; computed as `abs(source_power-1)` from what the simulation returns on `source_case=optimisation_baseline, source_samples_per_axis=17`; from: derivation: source_power=sum_j(area_weight_j*source_intensity_j)=sum_{k=0}^7 0.125=1 after sector normalization, so abs(source_power-1)=0.
- **off_axis_pupil_rejection**, a special or limiting case with a known answer: expects 0; computed as `max_abs_intensity_error` from what the simulation returns on `mask_case=unit_mean_sinusoid_harmonic_0.02_cycles_per_nm, source_case=single_point_sigma_x_0.9, defocus_nm=100, wafer_grid_spacing_nm=1`; from: derivation: P(f+s)=0 for both nonzero harmonics while P(s)=1 for DC, so I(x)=|M(0)|^2=1 and max_x abs(I(x)-1)=0.
- **nils_analytic_ramp**, a special or limiting case with a known answer: expects 0; computed as `nils` from what the simulation returns on `target_cd_nm=40, dI_dx=-0.02, I_threshold=0.8`; from: derivation: nils=40*abs(-0.02/0.8)=1 by E5.
- **meef_centered_difference**, a special or limiting case with a known answer: expects 0; computed as `meef` from what the simulation returns on `CD_plus_nm=41.2, CD_minus_nm=38.8, mask_cd_step_nm=1.5`; from: derivation: meef=abs((41.2-38.8)/(2*1.5))=0.8 by E6.

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **clear_mask_normalization**: The clear-mask special case must preserve unit normalized intensity.; the value the check “uniform_clear_mask_intensity” measures; lower is better; met at 1e-10 or below; a change smaller than 1e-10 counts as no change.
- **abbe_hopkins_consistency**: Independent Abbe and Hopkins image calculations must agree on the resolved, defocused off-axis validation pattern using the specified maximum normalized pointwise difference.; the value the check “abbe_hopkins_agreement” measures; lower is better; met at 0.005 or below; a change smaller than 0.001 counts as no change.
- **coherence_limit**: The circular-source coherence implementation must return its analytic zero-separation limit.; how far the check “mutual_coherence_at_zero_separation” lands from the value it expects; lower is better; met at 1e-12 or below; a change smaller than 1e-12 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/duv_lowk1_source_dof_opc_ambit/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 117 more lines of code than two scripts (40 written by the model, 77 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- Support the hypothesis only if the free-form design improves fine-grid common DOF over the selected parametric winner by more than the combined refinement uncertainty and does not increase representative-set maximum MEEF by more than 10%.
- Count the hypothesis against if the best parametric source matches or exceeds fine-grid free-form common DOF within numerical uncertainty.
- Require the Abbe and Hopkins normalized intensity profiles to agree within 0.5% maximum relative error on the independent validation case before interpreting any source comparison.

## Risks

- Scalar imaging overstates contrast and DOF at NA 1.35 because it omits vector polarization and high-angle field effects.
- The thin Kirchhoff mask ignores the mask-3D effects known to matter for dense attenuated-PSM patterns.
- A finite source-pixel basis can miss a superior free-form source topology.
- Threshold-only resist transfer can make absolute DOF values non-production-predictive even when relative optical trends are useful.

## What this quest will not do

- Vector electromagnetic imaging, polarization optimization, and rigorous mask-3D simulation.
- Resist chemistry, diffusion blur, stochastic photon noise, and wafer experimental validation.
- Two-dimensional layouts, contacts, and manufacturability rules beyond one-dimensional mask-width OPC.

## Checks already made

- optimisation_baseline; protocol.optimisation.baseline; The baseline assigns zero to every source-sector weight, violating the unit-power constraint.; An infeasible design cannot establish a meaningful improvement.; Use equal weights of 0.125 in eight equal-area sectors, representing a conventional sigma-0.9 source, and check that baseline with an oracle.; high
- weak_baseline; method and protocol.optimisation.baseline; The best parametric comparator is named, but the source screen does not specify enough geometry or a consistent fine-grid selection procedure.; Different quasar orientations, clipped dipole poles, or numerical resolutions could change the winner.; Define each source geometry, retain the complete screen, and recompute every candidate close enough to the coarse winner to change its ranking before selecting the comparator.; high
- precision; protocol.precision and protocol.optimisation.numerical_settings; The one-nanometre error target has no rule for what happens when the prescribed refinements fail to reach it.; A fixed refinement schedule does not guarantee the claimed numerical precision.; Refine each numerical axis further until the stated convergence criterion is met, or report the larger observed error and withhold the improvement claim.; high
- estimand; protocol.metrics and protocol.ci_method; Deterministic DOF, MEEF, and NILS values are labeled as means, and refinement differences are presented without distinguishing them from error bounds.; A deterministic model output has numerical sensitivity rather than a sampling confidence interval; successive differences alone are not a guaranteed bound.; Label the metrics deterministic, report separate spatial, source, focus, and finite-difference sensitivity estimates, and make no stochastic interval claim.; medium
- threshold_sensitivity; protocol.thresholds and protocol.acceptance; The conclusions can change at the chosen 10% improvement, 10% MEEF, and 0.1 nm ambit cutoffs.; A result close to one cutoff may not support a stable conclusion.; Report improvement at 0%, 5%, 10%, and 15%, MEEF changes at 0%, 10%, and 20%, and ambit distances at 0.05, 0.1, and 0.2 nm; qualify conclusions that change across these sets.; medium
- numerical_convergence; method and protocol.optimisation.numerical_settings; The design lists finer grids but does not require separate convergence of mask-width derivatives or a per-run physical invariant.; MEEF can be wrong despite converged CD, and an image calculation can fail without an obvious NaN.; Refine the MEEF mask-width step and require nonnegative intensity, unit source power, and the clear-mask intensity invariant for every numerical configuration.; high
- failed_runs; protocol.failure_policy; Failed evaluations are listed, but their counts and effect on the source comparison are not fully specified.; A failed pitch can otherwise disappear from the minimum-DOF calculation.; Count failures by source, pitch, stage, and reason; assign a failed source-pitch evaluation zero usable DOF and exclude any source with a failed required metric from winner selection.; medium
- other; method; The 5% exposure-latitude test, dose anchoring, and threshold calibration are not operationally defined.; Different interpretations produce different DOF and OPC bias.; Use one fixed resist threshold, calibrate each source's nominal dose at the 80 nm pitch after OPC, and require the CD band throughout a contiguous focus interval at both endpoints of a 5% total dose window.; high
- other; protocol.grid.pitch_nm and protocol.optimisation.fixed; The coarse pitch list may miss a narrow forbidden-pitch dip and therefore choose the wrong representative pitch.; The optimization target and reported worst pitch depend on pitch resolution.; Adaptively add pitches around every local DOF minimum until its location and depth stabilize, then freeze the representative forbidden pitch before free-form optimization.; high
- other; protocol.grid.neighbour_distance_nm; A last sampled neighbour distance of 2000 nm cannot establish that the CD change stays below 0.1 nm at all larger distances.; The optical response can oscillate back across the cutoff between or beyond sampled distances.; Refine around crossings and oscillations, verify the criterion over the remaining simulated range, and report a right-censored ambit when it has not stabilized by 2000 nm.; medium
- other; protocol.optimisation.constraints; The hypothesis requires comparable MEEF, but MEEF is absent from the optimization constraints.; Maximizing DOF alone can produce a source with unacceptable mask-error amplification.; Constrain representative-pitch MEEF to at most 10% above the selected parametric comparator and report the full through-pitch MEEF profile.; medium
- `protocol.precision` was left out of the plan because it could not be checked (`protocol.precision` must be a mapping with a `target_half_width` between 0 and 1 (for example 0.03)); FI asks the plan's model to write it before the experiment runs.
- `protocol.metrics` was left out of the plan because it could not be checked (`protocol.metrics` entry `common_dof_nm`: `kind` must be `proportion` or `mean`); FI asks the plan's model to write it before the experiment runs.
- `protocol.grid` was left out: a search for the best design has no separate sweep (its coarse scan is `protocol.optimisation.grid`)
- the limit on normalized_source_power was left out of the optimisation because it could not be read (`protocol.optimisation.constraints` entry 1 (`normalized_source_power`): `limit` must read like "<= 120" or ">= 3" (a bound and a number); '== 1' cannot be read); write it in the plan, or the search will not respect it
- the limit on representative_pitch_meef_ratio_to_selected_parametric was left out of the optimisation because it could not be read (`protocol.optimisation.constraints` entry 2 (`representative_pitch_meef_ratio_to_selected_parametric`): `limit` must read like "<= 120" or ">= 3" (a bound and a number); '<= 1.10 at each of the four representative pitches with finite comparator MEEF' cannot be read); write it in the plan, or the search will not respect it
- `protocol.optimisation.search_method` was left out of the plan because it could not be read (`protocol.optimisation.search_method` must be one of bounded_local, global_then_local, exhaustive, or a library method written as `scipy:<function>` or `optuna:<sampler>`); it is not used
- `protocol.optimisation.grid` was left out of the plan because it could not be read (`protocol.optimisation.grid` scans `source_family_screen`, which is not one of the design variables); it is not used
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, protocol, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gpt-5.6-terra).
- Equation E1 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E3 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E4 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E5 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- Equation E6 is marked as a derivation, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`).
- The check 'abbe_hopkins_agreement' compares with a second implementation but does not say what code it does not share with the simulation (write, for example, “shares no code with the simulation: it uses scipy's solve_ivp”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'feasible_baseline_source_power' takes its expected value from a source this quest did not retrieve (“Eight equal-area sectors each carry power 0.125, so their integrated source powe”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

- FI asked the plan's model to write `protocol.precision` again, and it is now in the plan.
- FI asked the plan's model to write `protocol.metrics` again, and it is now in the plan.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gpt-5.6-terra). Its summary: These checks would catch a DC normalization error and some implementation disagreements, but they do not establish correct line-space imaging through defocus. The model also applies defocus at the wrong frequency for off-axis illumination.
  - **uniform_clear_mask_intensity**: tests the model, would catch a plausible bug, its number is well defined. a better check: Add a defocused, off-axis case with transmitted nonzero mask harmonics to test the pupil phase..
  - **abbe_hopkins_agreement**: tests the model, would miss a plausible bug, its number is not well defined. The denominator and aggregation used for the maximum relative difference are unspecified.; At this pitch and source radius, no nonzero mask harmonic reaches the pupil, so the case produces a uniform image rather than resolved lines and spaces.; a better check: Compare the implementations at a pitch with transmitted nonzero harmonics, including nonzero defocus and an off-axis source..
  - **mutual_coherence_at_zero_separation**: may not test the model, would catch a plausible bug, its number is well defined. a better check: Check mutual coherence at a nonzero separation, such as its first zero..
  - **feasible_baseline_source_power**: may not test the model, would miss a plausible bug, its number is not well defined. It does not say whether source_power is the sum of sector weights or the integral produced by the stated finite source sampling.; a better check: Specify the returned power statistic and separately test an image quantity that depends on source-point positions and weights..
  - Found (high): E3: The defocus phase uses the mask frequency instead of the shifted pupil frequency (Off-axis illumination changes the pupil coordinate at which defocus must be evaluated.). Fix: Evaluate the phase at f+s and use that same transfer function when forming the Hopkins TCC.
  - Found (high): abbe_hopkins_agreement: The chosen case transmits only the mask's DC harmonic (Its source shifts cannot bring the pitch's first harmonic inside the pupil, leaving interference terms untested.). Fix: Use a pitch and source that transmit nonzero harmonics, and repeat the comparison at nonzero defocus.
  - Found (medium): abbe_hopkins_agreement: The relative-difference statistic lacks a defined denominator (Different normalizations can give different pass or fail results for the same images.). Fix: Define the statistic explicitly, including how pixels with small intensity are handled.
  - Found (medium): mutual_coherence_at_zero_separation: The check can pass with an incorrect nonzero-separation coherence function (The analytic zero-separation branch does not use the source radius, numerical aperture, wavelength, or Bessel evaluation.). Fix: Add a nonzero-separation reference value.
  - Found (medium): feasible_baseline_source_power: The check can reduce to reading back the prescribed weight sum (A unit sum does not establish that source positions and weights are used correctly in image formation.). Fix: Define whether source_power is a weight sum or a quadrature integral and add an optical check sensitive to the source.
  - Found (medium): E6: The units of the mask-CD perturbation are unspecified (MEEF requires a wafer-equivalent mask perturbation; a physical reticle perturbation needs the projection reduction factor.). Fix: State the coordinate convention or include the reduction factor in the equation.
  - Found (medium): E5: No declared check tests the NILS computation (An incorrect spatial derivative or CD normalization could pass every oracle.). Fix: Add a small case with an independently calculated threshold slope and NILS.
  - Checked, and no problem found: uniform_clear_mask_intensity: checked the constant field, DC normalization, defocus case, and expected maximum error.; E1: checked the supplied resist-side defocus phase and its units.; E2: checked circular-source coherence normalization and its zero-separation limit.; E4: checked the Hopkins frequency-pair expansion against source-point intensity summation.; E5: checked the target-CD normalization of the image log slope..
  - Equations the checks test: E2, E3, E4; not tested by any: E1, E5, E6.
  - The checks' cases were not judged by another model (the reader gave no verdict on every check's case): nobody but the model that wrote them has checked that each case isolates what its measure claims.
- The model's 6 equations were not read by another model (the reader gave no verdict on 'E6'): nobody but the model that wrote them has checked that each is the standard definition of its quantity.

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'defocus_phase_e1' was added (expects 0, computed as `phi_defocus_rad`).
- What changed: the check 'abbe_hopkins_agreement' changed: case {'pitch_nm': 80, 'conventional_sigma': 0.5, 'defocus_nm': 0, 'wafer_grid_spacing_nm': 1, 'source_samples_per_axis': 17} → {'pitch_nm': 200, 'source_case': 'single_point_sigma_x_0.5', 'defocus_nm': 100, 'wafer_grid_spacing_nm': 1, 'source_samples_per_axis': 17, 'relative_difference': 'max_x(abs(I_Abbe(x)-I_Hopkins(x))/max(max(abs(I_Abbe(x)),abs(I_Hopkins(x))),1e-12))'}; what it checks is worded differently; where its expected value comes from changed.
- What changed: the check 'mutual_coherence_first_zero' was added (expects 0, computed as `mutual_coherence`).
- What changed: the check 'feasible_baseline_source_power' changed: what it checks is worded differently; where its expected value comes from changed.
- What changed: the check 'off_axis_pupil_rejection' was added (expects 0, computed as `max_abs_intensity_error`).
- What changed: the check 'nils_analytic_ramp' was added (expects 0, computed as `nils`).
- What changed: the check 'meef_centered_difference' was added (expects 0, computed as `meef`).
- Equation E3 was changed: it was `I_Abbe(x,z) = sum_s w_s*abs(FFT_inverse[P(f+s)*M(f)*exp(i*phi_defocus_rad(f,z))])**2`, it is now `I_Abbe(x,z) = sum_s w_s*abs(FFT_inverse[P(f+s)*M(f)*exp(i*phi_defocus_rad(f+s,z))])**2`.
- After the plan was asked once for a worked example, equation E2 is not tested at all: the plan says it has no closed form for its output (The required J1 Bessel function is not in the pre-run closed-form expression grammar; the nonzero first-zero case is independently tested by oracle mutual_coherence_first_zero.) and no limit, symmetry, conservation or second implementation checks it.
- After the plan was asked once for a worked example, equation E3 is not tested at all: the plan says it has no closed form for its output (The FFT image depends on a discretized mask spectrum, source quadrature, and spatial grid; its independent Hopkins equivalence on a resolved, nonzero-defocus off-axis case is tested by oracle abbe_hopkins_agreement.) and no limit, symmetry, conservation or second implementation checks it.
- After the plan was asked once for a worked example, equation E4 is not tested at all: the plan says it has no closed form for its output (The finite TCC matrix and Fourier-series sum have no scalar closed form for the specified mask; agreement with independently coded Abbe evaluation is tested by oracle abbe_hopkins_agreement.) and no limit, symmetry, conservation or second implementation checks it.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: illumination sources (conventional, annular, x-dipole, quasar, free-form pixelated source) - NOT in the plan's settings.
- To vary or compare: pitches (isolated) - NOT in the plan's settings.
- To vary or compare: conventional sigma (0.3 to 0.9) - NOT in the plan's settings.
- To vary or compare: annular inner/outer sigma pairs - NOT in the plan's settings.
- To vary or compare: x-dipole pole centre (0.5 to 0.95) - NOT in the plan's settings.
- To vary or compare: x-dipole pole radius (0.1 to 0.2) - NOT in the plan's settings.
- To vary or compare: conventional sources with sigma (0.1 to 0.95) - NOT in the plan's settings.
- To vary or compare: neighbour (80 to 2000) - NOT in the plan's settings.
- To vary or compare: defocus z - NOT in the plan's settings.
- To vary or compare: mask line width - NOT in the plan's settings.
- To report: depth of focus through pitch.
- To report: NILS.
- To report: MEEF.
- To report: mask bias OPC needs.
- To report: iso-dense bias before OPC.
- To report: attenuated-mask side-lobe printing.
- To report: forbidden pitches.
- To report: common depth of focus.
- To report: CD change of the isolated line.
- To report: optical ambit - NOT measured in the plan.
- To report: extent of CD oscillation - NOT measured in the plan.
- To report: Abbe and Hopkins aerial images - NOT measured in the plan.
- To report: numerical error.
- To report: runtime - NOT measured in the plan.
- Formula the topic states: `sqrt(0.06)` - NOT in the plan's model.
- Formula the topic states: `2*pi/lambda * z * (n - sqrt(n**2 - (lambda*f)**2))`.
- Formula the topic states: `2*J1(u)/u with u = 2*pi*sigma*NA*d/lambda`.
- Formula the topic states: `d = 0.61*lambda/(sigma*NA)` - NOT in the plan's model.
- Formula the topic states: `sigma = lambda/(2*p*NA)` - NOT in the plan's model.
- Formula the topic states: `NA*(1+sigma_max)/lambda` - NOT in the plan's model.
- To compare: Compare depth of focus through pitch across illumination sources and identify each source's forbidden pitches.
- To compare: Compare NILS, MEEF, iso-dense bias, required mask bias, and side-lobe printing through pitch as the source changes.
- To compare: Compare the optimized free-form source with the best parametric source.
- To compare: Compare aerial images computed by Abbe and Hopkins.
- To compare: Compare the isolated line's CD change and optical ambit as its neighbour moves away under different illumination.
- To compare: Compare the CD oscillation with the stated mutual coherence function.
- Constraint: Use exposure wavelength 193.368 nm through water of refractive index 1.436 and a projection lens of NA 1.35.
- Constraint: Use a thin 6% attenuated phase-shift mask with dark-region transmission amplitude sqrt(0.06), phase pi, and the Kirchhoff thin-mask approximation.
- Constraint: Use scalar imaging with no polarization, and state plainly that at NA 1.35 it overstates contrast and depth of focus.
- Constraint: Use vertical dark-mask lines and spaces with a target line width of 40 nm.
- Constraint: Use a constant-threshold aerial image with no resist blur.
- Constraint: Anchor dose so the densest pitch prints at 40 nm at best focus.
- Constraint: Set the mask line width by one-dimensional OPC at every pitch so it prints at 40 nm at best focus.
- Constraint: Use 5% exposure latitude and a CD tolerance of +/-10% of 40 nm when reporting depth of focus.
- Constraint: Keep mirror symmetry in the optimized free-form pixelated source.
- Constraint: Use Python with numpy, scipy, and matplotlib with the Agg backend.
- Constraint: Implement both Abbe and Hopkins and make their aerial images agree.
- Constraint: Make figures showing depth of focus through pitch with forbidden pitches marked, OPC features through pitch, the optimized source beside the best parametric source, and CD change against neighbour distance for several sigma with coherence-function zeros drawn.

- FI asked the plan's model to add what was missing; it is still missing.

- Still missing (the paper will say it was not studied; the quest stops before anything runs, because the topic's question cannot be answered without it):
  - The topic says to vary or compare illumination sources (conventional, annular, x-dipole, quasar, free-form pixelated source), but the plan's settings do not include it.
  - The topic says to vary or compare pitches (isolated), but the plan's settings do not include it.
  - The topic says to vary or compare conventional sigma (0.3 to 0.9), but the plan's settings do not include it.
  - The topic says to vary or compare annular inner/outer sigma pairs, but the plan's settings do not include it.
  - The topic says to vary or compare x-dipole pole centre (0.5 to 0.95), but the plan's settings do not include it.
  - The topic says to vary or compare x-dipole pole radius (0.1 to 0.2), but the plan's settings do not include it.
  - The topic says to vary or compare conventional sources with sigma (0.1 to 0.95), but the plan's settings do not include it.
  - The topic says to vary or compare neighbour (80 to 2000), but the plan's settings do not include it.
  - The topic says to vary or compare defocus z, but the plan's settings do not include it.
  - The topic says to vary or compare mask line width, but the plan's settings do not include it.
  - The topic asks to report optical ambit, but the plan does not measure it.
  - The topic asks to report extent of CD oscillation, but the plan does not measure it.
  - The topic asks to report Abbe and Hopkins aerial images, but the plan does not measure it.
  - The topic asks to report runtime, but the plan does not measure it.
  - The topic states the formula `sqrt(0.06)`, but the plan's model does not use it and does not say why it uses another.
  - The topic states the formula `d = 0.61*lambda/(sigma*NA)`, but the plan's model does not use it and does not say why it uses another.
  - The topic states the formula `sigma = lambda/(2*p*NA)`, but the plan's model does not use it and does not say why it uses another.
  - The topic states the formula `NA*(1+sigma_max)/lambda`, but the plan's model does not use it and does not say why it uses another.

## The design (used as written)

```yaml
study_type: find_best_design
hypothesis: Within the stated scalar thin-mask model, a mirror-symmetric pixelated free-form source optimized
  for common process-window depth of focus will exceed the common DOF of the best screened parametric
  illumination on the four predeclared representative pitches while keeping their MEEF within 10% of that
  comparator. Performance over the complete 80–2000 nm pitch sweep is a separately reported check.
variables:
  independent:
  - illumination source shape and source-pixel weights
  - pitch
  - defocus
  - mask line width
  dependent:
  - common depth of focus on the representative pitches and on the full pitch sweep
  - per-pitch depth of focus
  - NILS
  - MEEF
  - OPC mask bias
  - per-pitch iso-dense bias before OPC
  - per-pitch side-lobe printing indicator
  - isolated-line CD change versus neighbour distance
  controls:
  - wavelength 193.368 nm
  - water refractive index 1.436
  - NA 1.35
  - scalar Kirchhoff thin-mask model
  - 6% attenuated phase-shift mask
  - 40 nm target wafer CD
  - constant-threshold aerial-image resist model
  - 5% total exposure latitude
  - plus/minus 10% CD tolerance
method: Implement scalar one-dimensional partially coherent imaging independently by Abbe source-point
  summation and Hopkins TCC summation using NumPy FFTs and SciPy routines. Represent conventional sources
  as uniform circular disks, annular sources as uniform rings, x-dipoles as two circular poles on the
  x source axis, and quasar sources as four equal circular poles at 45, 135, 225, and 315 degrees; intersect
  all sources with the unit source disk, renormalize, and record clipping. Screen conventional sigma 0.3–0.9
  in steps of 0.1; annular outer sigma 0.7, 0.8, 0.9, and 0.95 with inner/outer ratios 0.3, 0.5, and 0.7;
  x-dipole centre sigma 0.5–0.9 in steps of 0.1 plus 0.95, each with pole radius 0.1, 0.15, or 0.2; and
  quasar pole-centre sigma 0.5–0.9 in steps of 0.1 with the same three pole radii. Use identical dose
  calibration and OPC rules for every candidate. Screen pitches every 10 nm from 80 to 300 nm, every 25
  nm from 300 to 600 nm, every 100 nm from 600 to 2000 nm, plus 2000 nm, then adaptively refine every
  local DOF minimum and any change in side-lobe status or MEEF. Name forbidden pitches as refined local
  DOF minima whose prominence exceeds the reported DOF numerical error; report their depth without an
  arbitrary prominence cutoff. Freeze the lowest-DOF pitch of the fine-grid selected parametric source
  before free-form optimization. Select the best parametric source by representative-pitch common DOF
  after fine-grid recomputation of every candidate whose coarse result is within numerical ranking error;
  break remaining ties by larger minimum NILS, then smaller maximum MEEF, then fixed source-grid order.
  Optimize eight nonnegative, mirror-symmetric equal-area sector power fractions on the sigma<=0.9 disk,
  sampled on an 11-by-11 source-pixel representation and resampled at check resolution. Set one resist
  intensity threshold to 0.5 of clear-field intensity. For each source, determine nominal dose so an uncorrected
  40 nm mask line at 80 nm pitch prints 40 nm at best focus; hold that dose and threshold fixed while
  solving one-dimensional OPC mask width at each pitch, including 80 nm. A source for which this calibration
  or any required OPC solution fails is ineligible. Define 5% total exposure latitude as nominal dose
  multiplied by 0.975 to 1.025; usable DOF is the contiguous focus interval containing zero defocus in
  which both endpoint doses and every intermediate dose print 36–44 nm. Report a DOF reaching either focus
  boundary as censored. Compute NILS at the calibrated threshold edge, MEEF by a centred mask-width finite
  difference refined to convergence, and pre-OPC iso-dense bias as uncorrected printed CD at each pitch
  minus uncorrected printed CD at 80 nm. Record every off-target threshold crossing as side-lobe status.
  Recompute the free-form winner and retained parametric candidates at finer spatial, source, and focus
  settings; also double the simulated domain and increase retained Fourier harmonics until CD, DOF, and
  side-lobe status meet the numerical target. Report both representative-pitch common DOF and the independently
  refined minimum over the complete pitch sweep. For the ambit, adaptively sample neighbour separation,
  define visible CD oscillation as a local peak-to-trough amplitude exceeding the larger of 0.02 nm and
  three times its estimated numerical error, and report sensitivity at 0.02, 0.05, and 0.1 nm. Report
  an ambit as right-censored at 2000 nm if stability is unproved. Use SciPy for numerical routines and
  optimization, pint and uncertainties for dimensional and numerical-error audits, and matplotlib with
  the Agg backend for figures. Report runtime and state that scalar imaging at NA 1.35 overstates contrast
  and DOF relative to polarization-aware imaging.
expected_outcome: The optimized free-form source may improve the minimum DOF on the representative dense,
  semi-dense, frozen forbidden, and isolated pitches relative to the best parametric source. Report that
  comparison, its numerical sensitivity, and the separate full-pitch minimum; failure to exceed the comparator
  by the stated margin after fine-grid recomputation rejects the hypothesis within this scalar model.
figures_planned:
- figures/dof_and_meef_freeform_vs_parametric.png — per-pitch DOF and MEEF from 80 to 2000 nm, marked
  forbidden pitches, representative and full-sweep common-DOF values, censoring, and numerical errors
- figures/opc_metrics_through_pitch.png — per-pitch NILS, MEEF, OPC bias, pre-OPC iso-dense bias, and
  side-lobe status for each retained source
- figures/freeform_source_vs_parametric_source.png — normalized source maps, representative-pitch process
  windows, and numerical errors
- figures/optical_ambit_and_mutual_coherence.png — CD change, threshold crossings, coherence zeros, visibility
  cutoffs, and right-censored ambit markers where applicable
dependencies:
- numpy
- scipy
- matplotlib
- pint
- uncertainties
result_assertions:
- path: common_dof_nm
  min: 0
  max: 1000
  unit: nm
  reason: The reported common DOF is by definition clipped to the simulated -500 to +500 nm focus interval;
    a boundary-reaching value is flagged as censored.
- path: printed_cd_nm
  min: 0
  max: 2000
  unit: nm
  reason: A contour-derived one-dimensional printed width cannot be negative or exceed the largest simulated
    pattern period/domain.
- path: aerial_intensity
  min: 0
  max: 1000000
  unit: dimensionless
  reason: Partially coherent aerial intensity is an incoherent weighted sum of squared coherent-field
    magnitudes and is nonnegative.
- path: source_power
  min: 0.999999
  max: 1.000001
  unit: dimensionless
  reason: Every discrete illumination source is normalized to unit integrated source power before image
    formation.
- path: mutual_coherence_abs
  min: 0
  max: 1
  unit: dimensionless
  reason: The magnitude of normalized mutual coherence cannot exceed one.
protocol:
  optimisation:
    objective:
      quantity: common_dof_nm
      direction: maximise
      unit: nm
      meaning: The minimum usable DOF over dense, semi-dense, deterministically selected forbidden, and
        isolated representative pitches.
    design_variables:
    - name: source_sector_weight_0
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_1
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_2
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_3
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_4
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_5
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_6
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    - name: source_sector_weight_7
      low: 0
      high: 1
      unit: dimensionless
      kind: continuous
    fixed:
      source_pixel_grid: 11
      mirror_symmetry: true
      representative_dense_pitch_nm: 80
      representative_semidense_pitch_nm: 120
      representative_isolated_pitch_nm: 2000
      representative_forbidden_pitch_rule: lowest-DOF pitch in the adaptively refined 80–300 nm screening
        sweep for the fine-grid selected parametric source; ties select the smaller pitch; freeze this
        pitch before free-form optimization
      source_normalization: eight equal-area, mirror-symmetric sectors partition the sigma<=0.9 source
        disk; each sector weight is its fraction of total source power, the eight weights sum to 1, and
        source pixels within a sector receive equal area-weighted intensity
    constraints:
    - quantity: source_pixel_weight
      limit: '>= 0'
      unit: dimensionless
    baseline:
      values:
        source_sector_weight_0: 0.125
        source_sector_weight_1: 0.125
        source_sector_weight_2: 0.125
        source_sector_weight_3: 0.125
        source_sector_weight_4: 0.125
        source_sector_weight_5: 0.125
        source_sector_weight_6: 0.125
        source_sector_weight_7: 0.125
      source: Predeclared feasible conventional circular sigma-0.9 illumination, a member of the conventional
        screening grid, represented by equal power in eight equal-area sectors. The headline comparator
        is separately selected as the best tuned member of the full parametric screen.
    numerical_settings:
      wafer_grid_spacing_nm:
        search: 1
        check:
        - 0.5
        - 0.25
        finer: smaller
      source_samples_per_axis:
        search: 17
        check:
        - 25
        - 33
        finer: larger
      focus_step_nm:
        search: 5
        check:
        - 2
        - 1
        finer: smaller
    evaluation_budget:
      starts: 12
      per_start: 80
      seconds_per_evaluation: 8
    improvement_tolerance:
      value: 0.1
      mode: relative
  thresholds:
    wavelength_nm: 193.368
    resist_index: 1.436
    numerical_aperture: 1.35
    attenuated_mask_intensity_transmission: 0.06
    attenuated_mask_phase_shift_rad: 3.141592653589793
    target_cd_nm: 40
    cd_tolerance_fraction: 0.1
    cd_tolerance_nm: 4
    exposure_latitude_fraction: 0.05
    focus_min_nm: -500
    focus_max_nm: 500
    side_lobe_threshold_fraction: 1
    ambit_stable_cd_change_nm: 0.1
    abbe_hopkins_max_relative_difference: 0.005
    meef_acceptance_increase_fraction: 0.1
  acceptance:
  - Abbe and Hopkins must pass the independent second-implementation oracle before the main source screening;
    every run must have nonnegative intensity within floating-point tolerance and unit normalized source
    power.
  - A free-form result supports the hypothesis only when its finest-grid common_dof_nm exceeds the finest-grid
    selected-parametric common_dof_nm by at least 10% and by more than the sum of their reported numerical-error
    estimates.
  - The free-form source must remain normalized, nonnegative, and mirror symmetric at every stored pixel,
    and its MEEF must satisfy the representative-pitch constraint.
  - The source screen must include every listed conventional, annular, x-dipole, and quasar condition,
    with clipped dipole poles recorded, before selecting the parametric comparator.
  - Recompute all parametric candidates within the coarse-grid ranking error of the apparent winner at
    check resolution; refine pitch around local minima until forbidden-pitch location changes by at most
    1 nm and DOF changes by at most 1 nm.
  - Refine spatial spacing, source sampling, and focus step separately; continue beyond the listed checks
    if common DOF changes by more than 1 nm, and withhold a 10% improvement claim if the target remains
    unmet.
  - Recompute MEEF with mask-width perturbations of 2, 1, and 0.5 nm; report the largest change and refine
    further if it exceeds 0.02.
  - Report the comparative conclusion at DOF-improvement cutoffs of 0%, 5%, 10%, and 15%, the MEEF comparison
    at 0%, 10%, and 20%, and the ambit at 0.05, 0.1, and 0.2 nm; qualify any conclusion that depends on
    the cutoff.
  oracles:
  - name: uniform_clear_mask_intensity
    kind: special_case
    check: For a uniform unit-transmission mask and a normalized on-axis source, the normalized aerial
      intensity must be spatially uniform and equal to one.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-10
    tolerance_mode: absolute
    reference: 'derivation: for t(x)=1, its only Fourier component is f=0; the unaberrated pupil transmits
      it with unit amplitude, so I(x)=|1|^2=1 and max_x abs(I(x)-1)=0.'
    case:
      mask_case: uniform_clear
      source_case: on_axis
      defocus_nm: 137
      wafer_grid_spacing_nm: 1
    measure: max_abs_intensity_error
  - name: defocus_phase_e1
    kind: special_case
    check: Evaluate the supplied resist-side defocus phase at nonzero defocus and nonzero pupil frequency.
    expected: 0
    expected_formula: (2*pi/193.368)*100*(1.436-sqrt(1.436^2-(193.368*0.002)^2))
    tolerance: 1.0e-06
    tolerance_mode: relative
    reference: 'derivation: phi=(2*pi/193.368)*100*(1.436-sqrt(1.436^2-(193.368*0.002)^2)) by E1.'
    case:
      lambda_nm: 193.368
      z_nm: 100
      n: 1.436
      f_cycles_per_nm: 0.002
    measure: phi_defocus_rad
  - name: abbe_hopkins_agreement
    kind: second_implementation
    check: Compare independently coded Abbe source-point summation and Hopkins TCC image evaluation for
      a 200 nm periodic attenuated-mask line-space at nonzero defocus under a single off-axis source whose
      transmitted first harmonic reaches the pupil.
    expected: 0
    expected_formula: '0'
    tolerance: 0.005
    tolerance_mode: absolute
    reference: independent Abbe coherent-image summation versus independently assembled Hopkins TCC summation;
      the two implementations share only mask Fourier coefficients, source samples, and pupil parameters
      and share no image-formation or TCC-construction code.
    case:
      pitch_nm: 200
      source_case: single_point_sigma_x_0.5
      defocus_nm: 100
      wafer_grid_spacing_nm: 1
      source_samples_per_axis: 17
      relative_difference: max_x(abs(I_Abbe(x)-I_Hopkins(x))/max(max(abs(I_Abbe(x)),abs(I_Hopkins(x))),1e-12))
    measure: abbe_hopkins_max_relative_difference
  - name: mutual_coherence_at_zero_separation
    kind: special_case
    check: Evaluate the stated circular-source mutual-coherence expression at zero separation using its
      analytic limiting value.
    expected: 1
    expected_formula: '1'
    tolerance: 1.0e-12
    tolerance_mode: absolute
    reference: 'derivation: J1(u)=u/2+O(u^3), hence lim_{u->0} 2*J1(u)/u=1.'
    case:
      separation_nm: 0
      conventional_sigma: 0.7
    measure: mutual_coherence
  - name: mutual_coherence_first_zero
    kind: special_case
    check: Evaluate circular-source mutual coherence at the first zero of J1.
    expected: 0
    expected_formula: '0'
    tolerance: 0.001
    tolerance_mode: absolute
    reference: 'derivation: the first positive zero of J1 is u=3.8317059702075125; substituting u into
      2*J1(u)/u gives zero, and the case separation rounds the corresponding distance.'
    case:
      separation_nm: 124.786
      conventional_sigma: 0.7
    measure: mutual_coherence
  - name: feasible_baseline_source_power
    kind: invariant
    check: Evaluate the predeclared eight-weight conventional sigma-0.9 baseline and verify unit integrated
      source power using the area-weighted discrete source quadrature after clipping and renormalization.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-12
    tolerance_mode: absolute
    reference: 'derivation: source_power=sum_j(area_weight_j*source_intensity_j)=sum_{k=0}^7 0.125=1 after
      sector normalization, so abs(source_power-1)=0.'
    case:
      source_case: optimisation_baseline
      source_samples_per_axis: 17
    measure: abs(source_power-1)
  - name: off_axis_pupil_rejection
    kind: special_case
    check: Verify that a nonzero mask harmonic outside an off-axis shifted pupil does not print while
      the transmitted DC term remains normalized.
    expected: 0
    expected_formula: '0'
    tolerance: 1.0e-10
    tolerance_mode: absolute
    reference: 'derivation: P(f+s)=0 for both nonzero harmonics while P(s)=1 for DC, so I(x)=|M(0)|^2=1
      and max_x abs(I(x)-1)=0.'
    case:
      mask_case: unit_mean_sinusoid_harmonic_0.02_cycles_per_nm
      source_case: single_point_sigma_x_0.9
      defocus_nm: 100
      wafer_grid_spacing_nm: 1
    measure: max_abs_intensity_error
  - name: nils_analytic_ramp
    kind: special_case
    check: Evaluate NILS for a supplied local linear intensity ramp at the threshold.
    expected: 0
    expected_formula: 40*abs(-0.02/0.8)
    tolerance: 1.0e-06
    tolerance_mode: relative
    reference: 'derivation: nils=40*abs(-0.02/0.8)=1 by E5.'
    case:
      target_cd_nm: 40
      dI_dx: -0.02
      I_threshold: 0.8
    measure: nils
  - name: meef_centered_difference
    kind: special_case
    check: Evaluate MEEF from a centred wafer-plane-equivalent mask-CD perturbation.
    expected: 0
    expected_formula: abs((41.2-38.8)/(2*1.5))
    tolerance: 1.0e-06
    tolerance_mode: relative
    reference: 'derivation: meef=abs((41.2-38.8)/(2*1.5))=0.8 by E6.'
    case:
      CD_plus_nm: 41.2
      CD_minus_nm: 38.8
      mask_cd_step_nm: 1.5
    measure: meef
  criteria:
  - name: clear_mask_normalization
    what: The clear-mask special case must preserve unit normalized intensity.
    oracle: uniform_clear_mask_intensity
    use: value
    direction: lower
    target: 1.0e-10
    tolerance: 1.0e-10
  - name: abbe_hopkins_consistency
    what: Independent Abbe and Hopkins image calculations must agree on the resolved, defocused off-axis
      validation pattern using the specified maximum normalized pointwise difference.
    oracle: abbe_hopkins_agreement
    use: value
    direction: lower
    target: 0.005
    tolerance: 0.001
  - name: coherence_limit
    what: The circular-source coherence implementation must return its analytic zero-separation limit.
    oracle: mutual_coherence_at_zero_separation
    use: error
    direction: lower
    target: 1.0e-12
    tolerance: 1.0e-12
  precision:
    target_half_width: 0.03
    metric: relative common-DOF difference between compared illumination sources
    reason: A three-percent target half-width makes numerical sensitivity small relative to the predeclared
      ten-percent common-DOF improvement threshold while retaining the required separate refinement reporting.
  metrics:
  - id: common_dof_nm
    kind: mean
    estimand: Deterministic common usable depth of focus, defined as the minimum usable DOF over the dense,
      semi-dense, frozen forbidden, and isolated representative pitches for one source at one numerical
      configuration.
    unit: nm from one source and numerical-configuration evaluation.
  - id: per_pitch_dof_nm
    kind: mean
    estimand: Deterministic usable depth of focus for one source, pitch, and numerical configuration.
    unit: nm from one source-pitch and numerical-configuration evaluation.
  - id: nils
    kind: mean
    estimand: Deterministic normalized image log slope at the calibrated threshold edge for one source
      and pitch.
    unit: dimensionless value from one source-pitch and numerical-configuration evaluation.
  - id: meef
    kind: mean
    estimand: Deterministic mask error enhancement factor from the converged centred wafer-plane-equivalent
      mask-width perturbation for one source and pitch.
    unit: dimensionless value from one source-pitch and numerical-configuration evaluation.
  - id: opc_mask_bias_nm
    kind: mean
    estimand: Deterministic wafer-plane-equivalent mask-width bias required for one source and pitch to
      print the 40 nm target CD at best focus.
    unit: nm from one source-pitch and numerical-configuration evaluation.
  - id: iso_dense_bias_nm
    kind: mean
    estimand: Deterministic difference between isolated and dense printed CD before OPC at one source
      and numerical configuration.
    unit: nm from one source and numerical-configuration evaluation.
  - id: side_lobe_printing
    kind: proportion
    estimand: Fraction of evaluated source-pitch conditions for which an attenuated-mask off-target side
      lobe reaches the printing threshold.
    unit: one evaluated source-pitch condition.
  - id: isolated_line_cd_change_nm
    kind: mean
    estimand: Deterministic CD change of the isolated line caused by one neighbour at a specified separation,
      source, and numerical configuration.
    unit: nm from one source-neighbour-distance and numerical-configuration evaluation.
  seed_policy: Imaging and all headline metrics are deterministic. Source quadrature, FFT grids, and twelve
    distinct multistart initial vectors follow fixed documented index order; no pseudorandom streams or
    sampling confidence intervals are used.
  failure_policy: Count and report failed evaluations by source, pitch, stage, and reason, including root
    failures, nonfinite values, violated invariants, and optimizer failures. A failed source-pitch evaluation
    has zero usable DOF; a source missing any required OPC, MEEF, or image metric is ineligible for winner
    selection, and no failure is silently omitted.
  ci_method: No stochastic confidence interval applies. For each deterministic headline value report spatial-grid,
    source-quadrature, focus-step, pitch-location, domain-size, Fourier-truncation, and, where applicable,
    mask-width finite-difference sensitivity separately; use the largest successive change on each axis
    as an empirical numerical-error estimate, never as a guaranteed bound. Report the sum of relevant
    estimates for a comparison and qualify any result whose refinement trend is not decreasing. The twelve
    starts assess optimization stability, not uncertainty of the imaging calculation.
  model:
    summary: A scalar thin-mask Fourier-optics model forms partially coherent one-dimensional aerial images
      by Abbe source integration and independently by Hopkins TCC summation, followed by threshold-contour
      process-window extraction.
    assumptions:
    - The projection system is shift invariant, scalar, and aberration free apart from specified defocus.
    - The 6% attenuated dark mask region has amplitude sqrt(0.06) and phase pi.
    - Source points are mutually incoherent and source weights are nonnegative.
    - The resist is represented only by a constant aerial-image threshold.
    holds_for: The stipulated comparative scalar, one-dimensional thin-mask model over 80–2000 nm pitch,
      source coordinates no larger than sigma 1, and defocus between -500 and 500 nm. It does not predict
      polarization, mask-3D, or resist effects at NA 1.35; physical contrast and DOF are likely overestimated.
    equations:
    - id: E1
      formula: phi_defocus_rad = (2*pi/lambda_nm)*z_nm*(n-sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2))
      role: generates
      source: derivation
      derivation: The supplied resist-side pupil phase is the axial wave-vector difference multiplied
        by propagation distance. `k_z=(2*pi/lambda_nm)*sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2)` and
        `k_0=(2*pi/lambda_nm)*n`, so `phi_defocus_rad=z_nm*(k_0-k_z)=(2*pi/lambda_nm)*z_nm*(n-sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2))`.
      example:
        inputs:
          lambda_nm: 193.368
          z_nm: 100
          n: 1.436
          f_cycles_per_nm: 0.002
        expected_formula: (2*pi/lambda_nm)*z_nm*(n-sqrt(n**2-(lambda_nm*f_cycles_per_nm)**2))
    - id: E2
      formula: gamma = 2*J1(u)/u; u = 2*pi*sigma*NA*separation_nm/lambda_nm; gamma(0)=1
      role: generates
      source: derivation
      derivation: Use the supplied mutual-coherence form for a circular source. The u=0 branch is evaluated
        from the Bessel-series limit `J1(u)=u/2+O(u^3)`, therefore `gamma(0)=lim_(u->0) 2*J1(u)/u=1`.
      example:
        untestable: The required J1 Bessel function is not in the pre-run closed-form expression grammar;
          the nonzero first-zero case is independently tested by oracle mutual_coherence_first_zero.
    - id: E3
      formula: I_Abbe(x,z) = sum_s w_s*abs(FFT_inverse[P(f+s)*M(f)*exp(i*phi_defocus_rad(f+s,z))])**2
      role: generates
      source: derivation
      derivation: For source point s, a mask frequency f occupies pupil frequency `f_pupil=f+s`. Therefore
        `H_s(f,z)=P(f+s)*exp(i*phi_defocus_rad(f+s,z))`; the coherent field is `E_s(x,z)=FFT_inverse[H_s(f,z)*M(f)]`,
        and incoherent source integration gives `I_Abbe(x,z)=sum_s w_s*abs(E_s(x,z))**2`.
      example:
        untestable: The FFT image depends on a discretized mask spectrum, source quadrature, and spatial
          grid; its independent Hopkins equivalence on a resolved, nonzero-defocus off-axis case is tested
          by oracle abbe_hopkins_agreement.
    - id: E4
      formula: I_Hopkins(x,z) = sum_f1 sum_f2 M(f1)*conj(M(f2))*TCC_z(f1,f2)*exp(i*2*pi*(f1-f2)*x)
      role: generates
      source: derivation
      derivation: With `H_s(f,z)=P(f+s)*exp(i*phi_defocus_rad(f+s,z))`, expanding the Abbe coherent intensity
        gives `TCC_z(f1,f2)=sum_s w_s*H_s(f1,z)*conj(H_s(f2,z))`, hence `I_Hopkins(x,z)=sum_f1 sum_f2
        M(f1)*conj(M(f2))*TCC_z(f1,f2)*exp(i*2*pi*(f1-f2)*x)`.
      example:
        untestable: The finite TCC matrix and Fourier-series sum have no scalar closed form for the specified
          mask; agreement with independently coded Abbe evaluation is tested by oracle abbe_hopkins_agreement.
    - id: E5
      formula: nils = target_cd_nm*abs((dI_dx)/I_threshold)
      role: analyses
      source: derivation
      derivation: At threshold, the image log slope is `(1/I_threshold)*dI_dx`; normalizing its magnitude
        by the target CD gives `nils=target_cd_nm*abs((dI_dx)/I_threshold)`.
      example:
        inputs:
          target_cd_nm: 40
          dI_dx: -0.02
          I_threshold: 0.8
        expected_formula: target_cd_nm*abs(dI_dx/I_threshold)
    - id: E6
      formula: meef = abs((CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm))
      role: analyses
      source: derivation
      derivation: The mask-width coordinate and `mask_cd_step_nm` are wafer-plane equivalent. The centred
        derivative is `(CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm)`, therefore `meef=abs((CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm))`.
      example:
        inputs:
          CD_plus_nm: 41.2
          CD_minus_nm: 38.8
          mask_cd_step_nm: 1.5
        expected_formula: abs((CD_plus_nm-CD_minus_nm)/(2*mask_cd_step_nm))
```
