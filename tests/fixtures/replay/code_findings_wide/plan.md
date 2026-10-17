# Plan: TOPIC. At high-NA EUV, how do a chemically amplified resist (CAR) and a metal-oxide resist (MOR) compare in stochastic patterning: line-edge and line-

> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,
> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan "..."`). The prose is for
> you and for the paper's methods; the block under *The design (used as written)* is the design, exactly: what it says is
> what runs. Every version is kept in `.fi/plan_versions/`.

## In short

This quest will compare the stochastic performance of CAR and MOR resists using a Monte Carlo approach. It will quantify how absorption and blur influence roughness and defect rates as a function of dose.

## What the literature says

- **[12]**: Stochastic failures (bridges/breaks) are connected to the fundamental randomness of photon absorption in a 'bridge volume'.
- **[7]**: Non-CAR resists avoid acid diffusion and blur, improving resolution and LER.
- **[20]**: Metal-oxide clusters < 1nm are needed to avoid granularity contributions to LCDU.

## The sources this quest found

> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source that is not listed here does not count.

- [1] Study of a positive tone organometal chemically amplified resist for high-NA EUV patterning (DOI 10.1117/12.3051030)
- [2] EUV photolithography: resist progress and challenges (DOI 10.1117/12.2302759)
- [3] Topography effects on high-NA EUV (DOI 10.1117/12.3054358)
- [4] Navigating the research landscape for hyper-NA EUV lithography and future patterning technologies (DOI 10.1117/12.3074981)
- [5] Novel metal containing resists for EUV lithography extendibility (DOI 10.1117/12.2220149)
- [6] Metal Sensitizer in Chemically Amplified EUV Resist: a Study of Sensitivity Enhancement and Dissolution (DOI 10.2494/photopolymer.30.591)
- [7] Photoresist Fabrication for High-NA EUV Lithography: Material and Mechanism (DOI 10.54097/0rggyz09)
- [8] EUV photolithography: resist progress in metal–organic complex photoresists (DOI 10.1117/1.jmm.18.1.011007)
- [9] Molecular design of a positive tone organometal chemically amplified resist for small size contact hall by high-NA EUV lithography (DOI 10.1117/12.3071799)
- [10] Relationship between Stochastic Effect and Line Edge Roughness in Chemically Amplified Resists for Extreme Ultraviolet Lithography Studied by Monte Carlo Simulation (DOI 10.7567/jjap.51.086504)
- [11] Comparative study of line roughness metrics of chemically amplified and inorganic resists for EUV (DOI 10.1117/12.2217766)
- [12] Engineering the Future of EUV Lithography: A Collaborative Journey in Metal-Oxide Resist Innovation (DOI 10.2494/photopolymer.39.67)
- [W1] chemically amplified resist Research Papers - Academia.edu
- [W2] Novel metal-oxide photoresist materials for extreme UV lithography
- [W3] (PDF) Recent Progress in EUV Metal Oxide Photoresists
- [W4] Photoresist: EUV Resist, Chemically Amplified, Metal-Oxide | SemiconductorX
- [W5] (PDF) Integrated Fab Process for Metal Oxide EUV Photoresist
- [13] Analysis of line-and-space resist patterns with sub-20 nm half-pitch fabricated using high-numerical-aperture exposure tool of extreme ultraviolet lithography (DOI 10.7567/jjap.55.096501)
- [W6] Inside EUV Resists
- [14] Exploring the limits of contact hole patterning with high-NA EUV lithography (DOI 10.1117/12.3090509)
- [W7] High Sensitivity Resists for EUV Lithography: A Review of Material Design Strategies and Performance Results - PMC
- [15] Design strategy of extreme ultraviolet resists (DOI 10.35848/1347-4065/ad3a4c)
- [16] Three-dimensional cliff analysis method for defining patterning resolution in EUV lithography (DOI 10.1117/1.jmm.25.2.021203)
- [17] Comparative study of line roughness metrics of chemically amplified and inorganic resists for extreme ultraviolet (DOI 10.1117/1.jmm.15.3.034003)
- [18] High Sensitivity Resists for EUV Lithography: A Review of Material Design Strategies and Performance Results (DOI 10.3390/nano10081593)
- [W8] Metal resist for extreme ultraviolet lithography characterized by scanning transmission electron microscopy - IOPscience
- [19] Chemically amplified resist CDSEM metrology exploration for high NA EUV lithography (DOI 10.1117/1.jmm.21.2.021207)
- [20] Stochastics in extreme ultraviolet lithography: investigating the role of microscopic resist properties for metal-oxide-based resists (DOI 10.1117/1.jmm.17.4.041003)
- [21] X-Ray Interactions: Photoabsorption, Scattering, Transmission, and Reflection at E = 50-30,000 eV, Z = 1-92 (DOI 10.1006/adnd.1993.1013)
- [22] Metal Oxide Nanoparticle Photoresists for EUV Patterning (DOI 10.2494/photopolymer.27.663)
- [23] High-numerical aperture extreme ultraviolet scanner for 8-nm lithography and beyond (DOI 10.1117/1.jmm.16.4.041010)
- [24] Radiation Chemistry in Chemically Amplified Resists (DOI 10.1143/jjap.49.030001)
- [25] Elucidating the patterning mechanism of zirconium-based hybrid photoresists (DOI 10.1117/1.jmm.16.4.041007)

## The gap this experiment addresses

The existing literature discusses CAR and MOR separately or in general terms; there is a lack of a direct, parameter-controlled comparative simulation of LER and $10^{-6}$ defect probabilities for these specific high-NA parameters.

## The model behind the numbers

> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.

**What produces the numbers:** Assumption-driven Monte Carlo of scalar high-NA aerial intensity, Poisson photon arrival and absorption, simplified CAR acid generation and quenching or MOR network change, blur, threshold development, and geometry-based defects; comparative results are valid only within this model.

**What it assumes:** The supplied 0.680 D photon-flux conversion, alpha values, film thicknesses, CAR mean acid yield of 2, CAR diffusion scale near 5 nm, and MOR blur near 2 nm are context inputs rather than newly verified literature values.; The stated dipole and annular pupil shapes, 2 nm CAR electron blur, Poisson acid-yield distribution, 20% quencher fraction, Gaussian kernels, fixed threshold development, and 80 mJ/cm2 line calibration reference are explicit illustrative assumptions selected to make the model reproducible; vary them as specified in the method.; The aerial image supplies only mean photon intensity; photon arrivals and absorption are sampled stochastically.; No parameter in this design is asserted to represent a measured commercial CAR or MOR formulation.; Only absorbed-count statistics and photon bookkeeping have the stated analytical or exact oracles; geometry roughness and rare defects require simulation, independent sampling checks, and grid refinement.

**Where it holds:** The defined 13.5 nm, NA 0.55 scalar thin-mask, 20 to 150 mJ/cm2, 10 nm half-pitch line and 16 nm contact scenarios, subject to the stated parameter sensitivities and numerical convergence checks; it does not establish empirical material rankings.

**Its equations:**

- **E1** (produces the data): `Phi = 0.680 * Dose`. Source: derivation: 0.680 photons/nm^2 per mJ/cm^2 is given in the topic => Phi = 0.680 * Dose
- **E2** (produces the data): `N_abs = Phi * A * (1 - exp(-alpha * t))`. Source: derivation: incident photons * absorption probability = absorbed photons => N_abs = Phi * A * (1 - exp(-alpha * t))
- **E3** (analyses the results): `Relative Noise = 1 / sqrt(N)`. Source: given in the topic
- **E4** (analyses the results): `P(0) = exp(-N)`. Source: given in the topic

**The checks against known answers, and where each expected value comes from:**

- **shot_noise_mean**, a special or limiting case with a known answer: expects 1.0917; computed as `mean_absorbed_photons` from what the simulation returns on `exposure_dose=10, resist_type=CAR, region_area_nm2=1, n_trials=10000, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: N_abs = 10 * 0.680 * 1 * (1 - exp(-0.005 * 35)) ≈ 1.0917
- **shot_noise_variance**, a conserved quantity or other invariant: expects 0; computed as `abs(var_absorbed_photons / mean_absorbed_photons - 1)` from what the simulation returns on `exposure_dose=10, resist_type=CAR, region_area_nm2=1, n_trials=10000, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: For a Poisson process, Var(X) = E[X] => Var/Mean = 1 => abs(1-1) = 0
- **shot_noise_zero**, a special or limiting case with a known answer: expects 0.3356; computed as `fraction_zero_absorption` from what the simulation returns on `exposure_dose=10, resist_type=CAR, region_area_nm2=1, n_trials=10000, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: P(X=0) = exp(-lambda) for Poisson => P(0) = exp(-1.0917) ≈ 0.3356
- **photon_partition**, a conserved quantity or other invariant: expects 0; computed as `max_abs_incident_minus_absorbed_minus_transmitted` from what the simulation returns on `exposure_dose=20, resist_type=MOR, region_area_nm2=1, n_trials=100, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: N_incident = N_absorbed + N_transmitted => N_incident - N_absorbed - N_transmitted = 0
- **absorbed_count_area_scaling**, a special or limiting case with a known answer: expects 4.3668; computed as `mean_absorbed_photons` from what the simulation returns on `exposure_dose=10, resist_type=CAR, region_area_nm2=4, n_trials=10000, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: Convert dose to incident photons per nm2, multiply by region area, then multiply by the CAR absorbed fraction. => 10 * 0.680 * 4 * (1 - exp(-0.005 * 35)) ≈ 4.3668
- **mor_absorbed_fraction**, a special or limiting case with a known answer: expects 0.32968; computed as `mean_absorbed_photons / mean_incident_photons` from what the simulation returns on `exposure_dose=20, resist_type=MOR, region_area_nm2=1, n_trials=10000, uniform_aerial_intensity=1, blur=0, chemistry=off`; from: derivation: Convert the MOR absorption coefficient to inverse nm, multiply by film thickness, and apply the absorbed-fraction equation. => 1 - exp(-0.020 * 20) ≈ 0.32968

## How we will judge whether the code got better

> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the first full run. Shown from `criteria` in the design block below; edit it there.
> When one is not met after a run, FI changes the simulation one small step at a time (at most `engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and none worse by more than that check's own tolerance; the study's own results never choose the version, and every round is in `code/CHANGELOG.md`.

- **shot_noise_accuracy**: The simulated absorbed-photon mean matches the analytical finite-case mean; the other listed oracles must also pass.; how far the check “shot_noise_mean” lands from the value it expects; lower is better; met at 0.04 or below; a change smaller than 0.04 counts as no change.

## How the code will be laid out

The code in `code/` will be a small research tool: the model's equations in the package `code/euv_stochastic_car_vs_mor/` (no scenario values there), `simulate.py` running one setting with it (the function FI calls), `experiment.py` for the analysis, `tests/test_oracles.py` with this plan's checks as unit tests, `METHODS.md` saying which function computes each equation, and `run.py` to run the whole study.

This costs about 4 more files and about 105 more lines of code than two scripts (30 written by the model, 75 by FI); the limit is 400 lines (`execution.code_package_max_extra_lines`). When a reply leaves the package out, the model is asked again: at most 3 more requests to the model in the whole quest (`execution.code_package_max_extra_calls`), after which the quest keeps the code it has and says so. The package's text is also shown to the model each later time the simulation is changed or repaired.

## Success criteria

- MOR shows a lower dose-to-reach-1e-6-defects than CAR.
- MOR shows lower LER at equivalent doses.

## Risks

- Underestimating the effect of the quencher in CAR may overpredict its sensitivity.
- Small critical volume choice may significantly shift the absolute defect probability, though trends should remain.

## What this quest will not do

- Etch bias and pattern collapse effects.
- Mask 3D effects (shadowing).
- Complex aerial image diffraction (using simplified contrast profiles).

## Checks already made

- circular_evaluation; method step 6 and expected_outcome; The defect tail is evaluated with a Poisson CDF for a critical volume selected inside the same simplified model, while the conclusion treats that tail as a resist-level comparison.; The CDF bypasses blur, chemistry, development, and feature connectivity, so it cannot establish the stated bridge, break, missing-hole, or merge probabilities.; Estimate defects from the developed geometry with end-to-end rare-event sampling, validate it against direct Monte Carlo where probabilities overlap, and confine material comparisons to the stated assumed model.; high
- weak_baseline; method step 7; The shot-noise-only reference has no calibration protocol and could be made artificially favorable by retaining a threshold tuned for CAR or MOR.; A mismatched development threshold changes feature size and roughness independently of photon noise.; Use the same aerial images and absorption settings, disable chemical and spatial blur, and independently calibrate its development threshold to the target dimensions on calibration seeds.; medium
- stratum_collapse; variables.dependent and figures_planned; The draft lists defect probabilities without specifying separate results for lines, contacts, and their distinct failure modes.; An aggregate can hide a resist that improves one feature or defect type while worsening another.; Report dose curves and threshold doses separately for bridges, breaks, missing contacts, and merged contacts, with separate line and contact calibration.; high
- precision; protocol.runs_per_setting and protocol.precision; The fixed 1000 runs are not tied to the requested 0.1 nm LER half-width, and raw trials cannot resolve a probability of 1e-6.; At 1000 trials a probability near 0.5 has a 95% half-width near 0.031, while observing zero defects gives little information about a 1e-6 tail.; Use 1000 runs as an initial sample, increase roughness runs from the pilot variance, and require rare-event sampling to reach a stated log-probability interval before claiming a threshold crossing.; high
- estimand; protocol.metrics and protocol.ci_method; Only three metrics have estimands, and the draft gives no interval method for the other defects, dose-to-size, threshold dose, or equal-roughness comparisons.; A point estimate without its sampling unit and interval cannot support the requested numerical comparison.; Define run-level roughness means, pooled defect probabilities, calibration roots, and threshold-dose differences, with intervals from the corresponding independent observations or weighted rare-event samples.; high
- threshold_sensitivity; protocol.thresholds and method; All four defect thresholds equal 1.0 without units, a physical definition, or a sensitivity check.; A chosen cutoff can determine which resist appears to reach the 1e-6 target first.; Use connectivity as the primary defect definition and repeat classification with 0.5, 1, and 2 nm minimum neck or gap widths; report whether each conclusion persists.; high
- rng_independence; protocol.seed_policy; The seed tuple omits pattern, calibration versus evaluation, sampling method, and rare-event replicate.; Reused streams can make calibration and evaluation dependent or silently correlate settings.; Derive streams from a fixed base seed and the full purpose, pattern, resist, dose, run, and replicate tuple, with separate calibration and evaluation seeds.; medium
- numerical_convergence; method and protocol.model; The draft has no spatial grid refinement test or per-run photon conservation check.; Pixel connectivity and a 2 nm blur can change materially with grid spacing, especially in defect tails.; Compare 0.5, 0.25, and selected 0.125 nm grids, refine if the stated tolerances fail, and require incident photons to equal absorbed plus transmitted photons in every run.; high
- oracle; protocol.oracles; The sole oracle checks only the mean count, and its expected value of 10.821 disagrees with its own formula, which gives about 10.917.; It would miss incorrect Poisson variance, zero-event probability, or photon bookkeeping.; Correct the mean and add executable variance-to-mean, zero-event, and exact photon-partition checks with numeric cases and tolerances.; high
- failed_runs; protocol.failure_policy; Trials with unfindable boundaries are excluded.; A missing or merged feature can itself make a boundary unfindable, so exclusion biases both roughness and failure estimates.; Classify topology before roughness extraction, count valid defect outcomes, report failure and undefined-roughness counts, and rerun only numerical or infrastructure failures with a new recorded stream.; high
- other; method, protocol.model, and expected_outcome; The draft assumes Gaussian chemistry and declares MOR superior without specifying the optical source, quencher and development model, parameter basis, or sensitivity to those assumptions.; Absorption, amplification, and blur can trade off, and the simplified parameters alone do not establish a real-material ordering.; Specify the imaging and resist assumptions, vary uncertain parameters, and make the expected result a testable model-dependent comparison rather than a predetermined winner.; high
- The methodology audit amended the design: expected_outcome, figures_planned, hypothesis, method, protocol, result_assertions, variables (before and after are in needs/DESIGN_CRITIQUE.json). The design was read by a different model (gpt-6-sol), not the one that wrote it (gemma4:31b).
- The plan has only one criterion for judging whether the code got better; 2 to 5 independent checks of correctness make a change harder to misjudge.
- Equation E1 cites a source this quest did not retrieve (“CONTEXT”): a source recalled from memory does not count; cite a retrieved source by its [n], or write the derivation.
- Equation E2 cites a source this quest did not retrieve (“CONTEXT”): a source recalled from memory does not count; cite a retrieved source by its [n], or write the derivation.
- The check 'shot_noise_mean' names a derivation for its expected value, but it does not show how the value follows: write at least one equation (with `=`, or a relation such as `≈` or `<`) that shows how the expected value follows (for example `integral of S over the domain = 1, because S is normalised`, or `x-centroid = 0 by the symmetry S(-x) = S(x)`): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'shot_noise_variance' takes its expected value from a source this quest did not retrieve (“Poisson thinning theorem.”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'shot_noise_zero' takes its expected value from a source this quest did not retrieve (“Poisson zero-count probability from the supplied photon-flux and absorption equa”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.
- The check 'photon_partition' takes its expected value from a source this quest did not retrieve (“Exact partition of independently thinned incident photons.”): give its `reference` as `derivation: <how the value follows, with at least one equation using =>` (for example `derivation: integral of S over the domain = 1, because S is normalised`), a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share.

## A second look at the checks against known answers

> What FI did to the checks before anything ran. Each kind of check has one numeric form: the worst violation, expecting 0, for an invariant, a symmetry or a second implementation; the quantity itself for the others.


- The checks were read by a second model (gpt-6-sol (the connection did not say which model answered)), not the one that wrote the plan (gemma4:31b). Its summary: The checks exercise CAR photon statistics and photon bookkeeping, but their unit-area cases cannot reveal the missing area factor in the absorbed-count equation. MOR absorption needs its own numeric check.
  - **shot_noise_mean**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Repeat the check with a larger region to test whether the expected count scales with area..
  - **shot_noise_variance**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures.
  - **shot_noise_zero**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures.
  - **photon_partition**: tests the model, would catch a plausible bug, its number is well defined, its case isolates what it measures. a better check: Add a MOR check of the absorbed fraction against its absorption equation..
  - Found (high): E2: The equation omits region area while naming its result an absorbed photon count (Its right-hand side has units of photons per area, and every declared absorbed-count check uses a unit-area region that hides the omission.). Fix: Include region area in the count equation and check a region larger than unit area.
  - Found (medium): photon_partition: The MOR case checks conservation but does not check the MOR absorption probability (Absorbed and transmitted counts can still sum to incident counts when the MOR absorption coefficient is implemented incorrectly.). Fix: Check the simulated MOR absorbed fraction against the absorption equation.
  - Checked, and no problem found: E1: Checked the dose conversion against photon energy and the area units.; shot_noise_mean: Checked the expected absorbed count, finite-sample tolerance, and uniform unit-area case.; shot_noise_variance: Checked that the measure is a nonnegative violation with the invariant's expected value.; shot_noise_zero: Checked the Poisson zero-count formula and its uniform, chemistry-off case.; photon_partition: Checked that its measure represents the stated conservation identity..
  - Equations the checks test: E1, E2; not tested by any: none.
- The model's 2 equations were read by gpt-6-sol, not the model that wrote them:
  - **E1** is the standard definition or a correct derivation.
  - **E2** is not the standard definition: N_abs = Phi * A * (1 - exp(-alpha * t)), where A is the illuminated region area in nm2. (Phi is photons per nm2, so multiplying by the absorbed fraction alone gives absorbed photon density rather than the absorbed count named by N_abs.).

- FI asked the plan, once, to look at what the second reader found.
- What changed: the check 'shot_noise_mean' changed: where its expected value comes from changed.
- What changed: the check 'shot_noise_variance' changed: where its expected value comes from changed.
- What changed: the check 'shot_noise_zero' changed: expected_formula 'exp(-10 * 0.680 * (1 - exp(-0.005 * 35)))' → 'exp(-10 * 0.680 * 1 * (1 - exp(-0.005 * 35)))'; where its expected value comes from changed.
- What changed: the check 'photon_partition' changed: where its expected value comes from changed.
- What changed: the check 'absorbed_count_area_scaling' was added (expects 4.3668, computed as `mean_absorbed_photons`).
- What changed: the check 'mor_absorbed_fraction' was added (expects 0.32968, computed as `mean_absorbed_photons / mean_incident_photons`).
- Equation E2 was changed: it was `N_abs = Phi * (1 - exp(-alpha * t))`, it is now `N_abs = Phi * A * (1 - exp(-alpha * t))`.

- Before stopping or repairing, FI looked at the failing known-answer checks itself (at most one model call per check to work its expected value out again, two more runs of a deterministic check's case at smaller steps, three more seeds of a random check, one retry at twice the time limit): this run spent 4 model call(s) and 12 extra run(s) of the simulation on it.

## What the topic asks for

> Read from the topic by another model than the plan's (gpt-6-sol), then looked for in the plan.

- To vary or compare: resist (chemically amplified resist (CAR), metal-oxide resist (MOR)).
- To vary or compare: exposure dose (20 to 150).
- To vary or compare: patterns (lines and spaces, contact holes).
- To vary or compare: chemistry (switched off).
- To report: dose-to-size.
- To report: line-edge roughness.
- To report: line-width roughness.
- To report: probability of a bridge per micrometre of line.
- To report: probability of a break per micrometre of line.
- To report: probability that a contact is missing.
- To report: probability that a contact is merged.
- To report: dose where failure rates fall below 1e-6.
- To report: variance of the absorbed-photon count.
- To report: probability of no absorbed events.
- To report: runtime.
- Formula the topic states: `0.680*D photons per nm2`.
- Formula the topic states: `1/sqrt(N)`.
- Formula the topic states: `1 - exp(-alpha*t)`.
- Formula the topic states: `exp(-N)`.
- To compare: Compare CAR and MOR in roughness and stochastic defect probability as exposure dose changes.
- To compare: Compare CAR and MOR at equal dose.
- To compare: Compare CAR and MOR at equal roughness.
- To compare: Compare the reported LER and LWR with the shot-noise-only reference.
- Constraint: Use NA 0.55, a thin binary mask, and scalar imaging.
- Constraint: Use a dipole source for lines and a quasar or annular source for contacts, and state the contact-source choice.
- Constraint: The aerial image only sets the mean exposure; the stochastics come from the resist.
- Constraint: Use lines and spaces at 10 nm half pitch (20 nm pitch), and contact holes of 16 nm at 32 nm pitch.
- Constraint: For CAR, assume alpha = 5 per micrometre, t = 35 nm, about 2 acids per absorbed photon, a base quencher, and acid diffusion of about 5 nm during the post-exposure bake.
- Constraint: For MOR, assume alpha = 20 per micrometre, t = 20 nm, direct change of the metal-oxide network, no chemical amplification, and a short blur of about 2 nm.
- Constraint: State every other assumed parameter with a source or a reason.
- Constraint: Calibrate each resist so lines print at 10 nm and contacts at 16 nm at its dose-to-size.
- Constraint: Report LER and LWR as 3 sigma over a stated line length, and state the high-frequency cut-off.
- Constraint: Estimate failure-rate tails with a stated method and its uncertainty, using extrapolation of a fitted tail or a rare-event method rather than raw counts alone.
- Constraint: Use Python with numpy and scipy, and matplotlib with the Agg backend.
- Constraint: Use Monte Carlo with fixed, stated seeds and enough trials for the reported uncertainty.

- FI asked the plan's model to add what was missing; it is in the plan now.

## The design (used as written)

```yaml
hypothesis: Within the specified assumed high-NA EUV imaging and resist model, MOR's greater absorption
  and shorter blur may reduce some roughness and defect rates relative to CAR, while CAR amplification
  may offset photon scarcity for other endpoints. Test the direction and size of each difference at equal
  dose and equal roughness; do not assume one resist wins every endpoint.
variables:
  independent:
  - exposure_dose in mJ/cm2
  - 'resist_type: CAR or MOR'
  - 'pattern: 10 nm half-pitch lines or 16 nm contacts at 32 nm pitch'
  - assumption-sensitivity setting for absorption, chemical yield, quencher, and blur
  - 'defect classifier minimum neck or gap width: 0, 0.5, 1, or 2 nm'
  - 'chemistry: on or off'
  dependent:
  - dose_to_size for lines and contacts, mJ/cm2
  - mean 3-sigma LER over defined 1 micrometre line segments, nm
  - mean 3-sigma LWR over defined 1 micrometre line segments, nm
  - p_bridge per 1 micrometre line segment
  - p_break per 1 micrometre line segment
  - p_missing per contact
  - p_merged per contact with its nearest neighbors
  - supported dose intervals where each of the four defect probabilities is below 1e-6, mJ/cm2
  - common dose interval where all four defect probabilities are below 1e-6, if one exists
  - CAR-minus-MOR threshold-dose difference for each defect class when a supported first crossing exists,
    mJ/cm2
  - CAR-minus-MOR roughness and defect differences at equal dose and at equal interpolated roughness within
    the observed overlap
  - variance of the absorbed-photon count
  - probability of no absorbed events
  - runtime
  controls:
  - 13.5 nm wavelength and NA 0.55
  - thin binary mask and scalar partially coherent imaging
  - dipole illumination for lines and annular illumination for contacts
  - fixed pattern geometry and simulation window
  - physical spatial scale, boundary handling, and roughness frequency band
  - independent dose-to-size calibration protocol
method: 'Compute normalized scalar aerial images from a thin binary mask at NA 0.55: use a dipole source
  with poles at normalized pupil coordinates ±0.65 and radius 0.10 for lines, and annular illumination
  with inner and outer normalized radii 0.50 and 0.90 for contacts. These source settings are declared
  modeling assumptions because no measured tool pupil is supplied; repeat the comparison under reasonable
  source-radius changes. At each dose, sample incident photons as a spatial Poisson process with mean
  0.680 D per nm2 times the local normalized aerial intensity, and absorb each by independent thinning
  with probability 1-exp(-alpha t). Use CAR alpha=0.005 nm^-1, t=35 nm, and MOR alpha=0.020 nm^-1, t=20
  nm from the supplied context. For CAR, sample acid yield per absorbed event as Poisson with mean 2,
  apply an assumed 2 nm secondary-electron Gaussian blur and 5 nm acid-diffusion Gaussian blur, subtract
  a fixed quencher field equal to 20% of the reference-dose mean generated acid density, then threshold
  the resulting deprotection proxy. For MOR, use a single 2 nm Gaussian energy blur and threshold the
  network-change proxy. Use the same declared positive-tone geometry convention for both illustrative
  materials: proxy values above threshold become developed openings; compare openings and remaining lines
  with their intended masks when classifying defects. The yield distribution, electron blur, quencher
  fraction, Gaussian kernels, development tone, and threshold laws are explicit illustrative assumptions,
  not measured material properties; vary yield by ±25%, quencher fraction from 0 to 40%, and each blur
  by ±25%. Set each resist''s fixed development threshold using independent calibration simulations at
  a declared reference line dose of 80 mJ/cm2; then estimate line dose-to-size from held-out simulations.
  Fix one contact mask bias from optical target geometry before viewing resist outcomes and measure each
  resist''s contact dose-to-size on held-out simulations. Report the calibrated thresholds and common
  contact bias, and treat the chosen reference dose as a model assumption in the sensitivity analysis.
  Use a 0.5 nm primary grid with photon positions sampled in physical coordinates, a guard region of at
  least four blur standard deviations, and no periodic wraparound across unrelated features. Extract line
  edges over independent 1 micrometre segments; remove only the segment mean position or mean width, apply
  the same 0.5 nm spatial-sampling Nyquist cutoff to all resists, and report run-level LER as the average
  of the two edges'' 3 times within-segment standard deviations and LWR as 3 times the within-segment
  width standard deviation. Retain individual edge values. Repeat at 0.25 nm for dose-to-size and proposed
  threshold brackets, and at 0.125 nm wherever the first refinement fails. Define bridges as connectivity
  across adjacent line spaces, breaks as a disconnected intended line, missing contacts as absent developed
  openings, and merges as openings connected to neighbors; primary classification uses topology, with
  0.5, 1, and 2 nm minimum-neck or gap sensitivity. Estimate each probability from final developed geometry.
  Use direct Monte Carlo where practical and end-to-end importance sampling for rare tails: choose the
  photon and acid-yield tilt on separate pilot seeds, retain the exact likelihood ratio for each trajectory,
  and validate the weighted estimates against direct Monte Carlo in their overlap region. Do not infer
  geometry defects from a single critical-volume Poisson CDF. For the shot-noise reference, retain each
  resist''s aerial image and absorption but disable acid chemistry and spatial blur, then calibrate its
  own threshold on separate seeds to the same target size. At several doses and fixed target geometry,
  compare shot-noise-only edge variance with the local Poisson count-and-aerial-gradient prediction; identify
  where that linear approximation fails. Report dose-to-size, all four defect classes separately, equal-dose
  comparisons, and equal-roughness comparisons only within the observed overlap. For each defect class,
  report supported below-1e-6 dose intervals, plus their common intersection; report a first crossing
  only when the data bracket one and the uncertainty supports it. If a tail interval or grid test does
  not support a crossing, report a bound or unresolved crossing rather than extrapolating a precise dose.
  Select example-image seeds before inspecting outcomes. Record CPU runtime.'
expected_outcome: The model may favor MOR for some endpoints because its assumed absorbed fraction is
  about 33% versus about 16% for CAR and its assumed spatial blur is shorter. CAR amplification can alter
  that ordering, so report observed differences with uncertainty, including any reversals across dose,
  pattern, defect type, or sensitivity setting. A common below-1e-6 dose interval may be absent. All comparative
  conclusions apply to this illustrative model and its stated parameter ranges, not to unmeasured commercial
  resists.
figures_planned:
- 'roughness_vs_dose.png: separate line LER and LWR panels for both resists, with calibrated shot-noise-only
  references, 95% intervals, dose-to-size markers, the stated line length and frequency cutoff, and a
  shot-noise edge-variance scaling check'
- 'defects_vs_dose.png: four panels for line bridges, line breaks, missing contacts, and merged contacts,
  each with both resists, 95% intervals, a log probability axis, the 1e-6 line, supported below-threshold
  intervals, and any common dose window'
- 'stochastic_images.png: prespecified-seed physical-scale examples of line and contact outcomes for each
  resist at matched stated doses, with the development threshold and defect classification indicated'
- 'comparison_and_sensitivity.png: equal-dose and within-overlap equal-roughness differences, supported
  threshold-dose differences, and parameter/classifier sensitivity with uncertainty'
dependencies:
- numpy
- scipy
- matplotlib
- scikit-image
- uncertainties
- pint
result_assertions:
- path: LER
  min: 0.0
  max: 10.0
  unit: nm
  reason: This is a broad plausibility flag for the illustrative model, not evidence of physical validity;
    investigate rather than silently discard values outside it.
- path: LWR
  min: 0.0
  max: 20.0
  unit: nm
  reason: This is a broad diagnostic range; no fixed LWR-to-LER ratio is assumed.
- path: p_bridge
  min: 0.0
  max: 1.0
  unit: dimensionless
  reason: A probability per defined 1 micrometre segment must lie in [0, 1].
protocol:
  runs_per_setting: 1000
  thresholds:
    T_bridge: 0.0
    T_break: 0.0
    T_missing: 0.0
    T_merged: 0.0
  acceptance:
  - Report all endpoints with uncertainty even if MOR is worse or differences are inconclusive.
  - Claim a probability below 1e-6 only if its 95% upper bound is below 1e-6 and the rare-event estimator
    passed direct-Monte-Carlo overlap and grid checks.
  - Claim a resist ordering only for endpoints and parameter/classifier settings whose 95% difference
    interval excludes zero; explicitly report reversals and unresolved comparisons.
  - For grid refinement, require dose-to-size changes below 2 mJ/cm2, roughness changes below 0.1 nm,
    and log10 defect-probability changes below 0.2 where estimable; refine further or report numerical
    uncertainty when these fail.
  - Require incident_count = absorbed_count + transmitted_count exactly in every photon-thinning run.
  precision:
    target_half_width: 0.1
    metric: mean 3-sigma LER in nm; additionally target at most 0.3 decades 95% half-width for log10 defect
      probabilities near 1e-6
    reason: 'Start with 1000 independent line realizations per setting: a 0.1 nm 95% half-width is supported
      only when pilot run-level SD is at most 1.61 nm; otherwise increase to ceiling((1.96 times pilot
      SD / 0.1)^2). For a direct probability near 0.5, 1000 trials support about ±0.031 at 95%; rare-event
      sampling continues until the stated 0.3-decade target is met or reports an unresolved bound.'
  metrics:
  - id: LER
    estimand: Mean across independent printable 1 micrometre line realizations of 3 times the within-segment
      edge-position standard deviation, with printable fraction and excluded-from-roughness defect count
      reported
    kind: mean
    unit: nm
    cluster: line_realization
    paired: false
  - id: LWR
    estimand: Mean across independent printable 1 micrometre line realizations of 3 times the within-segment
      line-width standard deviation
    kind: mean
    unit: nm
    cluster: line_realization
    paired: false
  - id: p_bridge
    estimand: Probability that a 1 micrometre intended line-space segment has at least one topological
      bridge
    kind: proportion
    unit: dimensionless per 1 micrometre segment
    cluster: line_realization
    paired: false
  - id: p_break
    estimand: Probability that a 1 micrometre intended line segment has at least one topological break
    kind: proportion
    unit: dimensionless per 1 micrometre segment
    cluster: line_realization
    paired: false
  - id: p_missing
    estimand: Probability that one intended contact opening is absent
    kind: proportion
    unit: dimensionless per contact
    cluster: independent_contact_array_realization
    paired: false
  - id: p_merged
    estimand: Probability that one intended contact opening connects to at least one nearest-neighbor
      opening
    kind: proportion
    unit: dimensionless per contact
    cluster: independent_contact_array_realization
    paired: false
  - id: dose_to_size
    estimand: Interpolated dose at which the ensemble mean printed line width is 10 nm or mean contact
      diameter is 16 nm, separately by resist and pattern
    kind: mean
    unit: mJ/cm2
    cluster: independent_calibration_realization
    paired: false
  - id: dose_at_1e-6
    estimand: First interpolated dose whose defect-probability 95% upper bound is below 1e-6, separately
      by resist and defect class, reported as unresolved if the observed range does not support it
    kind: mean
    unit: mJ/cm2
    cluster: independent_rare_event_root
    paired: false
  - id: absorbed_variance
    estimand: Variance of the absorbed-photon count in a unit region
    kind: mean
    unit: dimensionless
    cluster: independent_photon_realization
    paired: false
  - id: p_zero_absorbed
    estimand: Probability that a unit region expecting N absorbed photons receives none
    kind: proportion
    unit: dimensionless
    cluster: independent_photon_realization
    paired: false
  - id: runtime
    estimand: Wall-clock CPU time for the simulation
    kind: mean
    unit: seconds
    cluster: simulation_run
    paired: false
  oracles:
  - name: shot_noise_mean
    kind: special_case
    check: Mean absorbed count in a uniform 1 nm2 region over 10000 independent trials, with blur and
      chemistry off
    expected: 1.0917
    expected_formula: 10 * 0.680 * 1 * (1 - exp(-0.005 * 35))
    tolerance: 0.04
    tolerance_mode: absolute
    reference: 'derivation: N_abs = 10 * 0.680 * 1 * (1 - exp(-0.005 * 35)) ≈ 1.0917'
    case:
      exposure_dose: 10
      resist_type: CAR
      region_area_nm2: 1
      n_trials: 10000
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: mean_absorbed_photons
  - name: shot_noise_variance
    kind: invariant
    check: Variance-to-mean ratio of absorbed counts over the same 10000 independent uniform-region trials
    expected: 0
    expected_formula: Poisson variance equals Poisson mean.
    tolerance: 0.07
    tolerance_mode: absolute
    reference: 'derivation: For a Poisson process, Var(X) = E[X] => Var/Mean = 1 => abs(1-1) = 0'
    case:
      exposure_dose: 10
      resist_type: CAR
      region_area_nm2: 1
      n_trials: 10000
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: abs(var_absorbed_photons / mean_absorbed_photons - 1)
  - name: shot_noise_zero
    kind: special_case
    check: Fraction of uniform-region trials receiving zero absorbed photons
    expected: 0.3356
    expected_formula: exp(-10 * 0.680 * 1 * (1 - exp(-0.005 * 35)))
    tolerance: 0.02
    tolerance_mode: absolute
    reference: 'derivation: P(X=0) = exp(-lambda) for Poisson => P(0) = exp(-1.0917) ≈ 0.3356'
    case:
      exposure_dose: 10
      resist_type: CAR
      region_area_nm2: 1
      n_trials: 10000
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: fraction_zero_absorption
  - name: photon_partition
    kind: invariant
    check: Worst photon bookkeeping violation across the returned trials
    expected: 0
    expected_formula: Every incident photon is either absorbed or transmitted.
    tolerance: 0
    tolerance_mode: absolute
    reference: 'derivation: N_incident = N_absorbed + N_transmitted => N_incident - N_absorbed - N_transmitted
      = 0'
    case:
      exposure_dose: 20
      resist_type: MOR
      region_area_nm2: 1
      n_trials: 100
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: max_abs_incident_minus_absorbed_minus_transmitted
  - name: absorbed_count_area_scaling
    kind: special_case
    check: Mean absorbed photon count in a uniform 4 nm2 CAR region over independent trials
    expected: 4.3668
    expected_formula: 10 * 0.680 * 4 * (1 - exp(-0.005 * 35))
    tolerance: 0.12
    tolerance_mode: absolute
    reference: 'derivation: Convert dose to incident photons per nm2, multiply by region area, then multiply
      by the CAR absorbed fraction. => 10 * 0.680 * 4 * (1 - exp(-0.005 * 35)) ≈ 4.3668'
    case:
      exposure_dose: 10
      resist_type: CAR
      region_area_nm2: 4
      n_trials: 10000
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: mean_absorbed_photons
  - name: mor_absorbed_fraction
    kind: special_case
    check: Fraction of incident photons absorbed by a uniform MOR film
    expected: 0.32968
    expected_formula: 1 - exp(-0.020 * 20)
    tolerance: 0.01
    tolerance_mode: absolute
    reference: 'derivation: Convert the MOR absorption coefficient to inverse nm, multiply by film thickness,
      and apply the absorbed-fraction equation. => 1 - exp(-0.020 * 20) ≈ 0.32968'
    case:
      exposure_dose: 20
      resist_type: MOR
      region_area_nm2: 1
      n_trials: 10000
      uniform_aerial_intensity: 1
      blur: 0
      chemistry: 'off'
    measure: mean_absorbed_photons / mean_incident_photons
  criteria:
  - name: shot_noise_accuracy
    what: The simulated absorbed-photon mean matches the analytical finite-case mean; the other listed
      oracles must also pass.
    oracle: shot_noise_mean
    use: error
    direction: lower
    target: 0.04
    tolerance: 0.04
  topic_map:
  - item: chemically amplified resist (CAR) and a metal-oxide resist (MOR)
    by: setting:resist_type
  - item: line-edge and line-width roughness
    by:
    - metric:LER
    - metric:LWR
  - item: probability of stochastic defects (bridges and breaks in lines, missing and merged contact holes)
    by:
    - metric:p_bridge
    - metric:p_break
    - metric:p_missing
    - metric:p_merged
  - item: exposure dose
    by: setting:exposure_dose
  - item: lines and spaces, contact holes
    by: setting:pattern
  - item: chemistry (switched off)
    by: setting:chemistry
  - item: variance of the absorbed-photon count
    by: metric:absorbed_variance
  - item: probability that a region expecting N absorbed events receives none
    by: metric:p_zero_absorbed
  - item: runtime
    by: metric:runtime
  - item: relative noise of N expected photons is 1/sqrt(N)
    by: equation:E3
  - item: probability that a region expecting N absorbed events receives none is exp(-N)
    by: equation:E4
  grid:
    exposure_dose:
    - 20
    - 30
    - 40
    - 50
    - 60
    - 70
    - 80
    - 90
    - 100
    - 110
    - 120
    - 135
    - 150
    resist_type:
    - CAR
    - MOR
    pattern:
    - lines
    - contacts
    chemistry:
    - 'on'
    - 'off'
  seed_policy: Use base seed 20261008. Derive independent counter-based streams from (base_seed, purpose,
    pattern, resist_type, dose, sampling_method, replicate, run_index); purpose separates calibration,
    direct evaluation, rare-event pilot, rare-event evaluation, and numerical checks. Settings are unpaired
    unless an explicitly labeled paired sensitivity comparison reuses incident-photon coordinates and
    is analyzed by paired differences.
  failure_policy: 'Classify geometry before extracting boundaries. Count every finite developed pattern
    in the defect denominator, including patterns with no usable edge; mark its roughness undefined and
    report the count and resulting conditional roughness estimand. Never silently drop NaN, divergent,
    or infrastructure failures: record them by setting, repair numerical causes, and rerun with a new
    recorded seed; publish both original and rerun counts. If a numerical failure remains, report bounds
    obtained by counting all unresolved trials first as failures and then as nonfailures.'
  ci_method: Use percentile bootstrap over independent 1 micrometre line realizations for mean run-level
    LER and LWR, and pooled-count Wilson 95% intervals for directly sampled line-defect probabilities.
    For primary contact probabilities, use one prespecified interior contact per independent array and
    pooled-count Wilson intervals; if multiple contacts per array are reported, use an array-level bootstrap.
    For weighted rare-event estimates, use likelihood-weighted pooled estimators with variance from at
    least 100 independent importance-sampling roots, check effective sample size and root-to-root stability,
    and form 95% intervals on log10 probability. Propagate calibration and dose-curve uncertainty by resampling
    their independent runs; use the resulting bootstrap distribution for dose-to-size, equal-roughness
    contrasts, and supported threshold crossings. A deterministic grid result has a refinement error,
    not a Monte Carlo interval.
  model:
    summary: Assumption-driven Monte Carlo of scalar high-NA aerial intensity, Poisson photon arrival
      and absorption, simplified CAR acid generation and quenching or MOR network change, blur, threshold
      development, and geometry-based defects; comparative results are valid only within this model.
    assumptions:
    - The supplied 0.680 D photon-flux conversion, alpha values, film thicknesses, CAR mean acid yield
      of 2, CAR diffusion scale near 5 nm, and MOR blur near 2 nm are context inputs rather than newly
      verified literature values.
    - The stated dipole and annular pupil shapes, 2 nm CAR electron blur, Poisson acid-yield distribution,
      20% quencher fraction, Gaussian kernels, fixed threshold development, and 80 mJ/cm2 line calibration
      reference are explicit illustrative assumptions selected to make the model reproducible; vary them
      as specified in the method.
    - The aerial image supplies only mean photon intensity; photon arrivals and absorption are sampled
      stochastically.
    - No parameter in this design is asserted to represent a measured commercial CAR or MOR formulation.
    - Only absorbed-count statistics and photon bookkeeping have the stated analytical or exact oracles;
      geometry roughness and rare defects require simulation, independent sampling checks, and grid refinement.
    holds_for: The defined 13.5 nm, NA 0.55 scalar thin-mask, 20 to 150 mJ/cm2, 10 nm half-pitch line
      and 16 nm contact scenarios, subject to the stated parameter sensitivities and numerical convergence
      checks; it does not establish empirical material rankings.
    equations:
    - id: E1
      formula: Phi = 0.680 * Dose
      role: generates
      source: 'derivation: 0.680 photons/nm^2 per mJ/cm^2 is given in the topic => Phi = 0.680 * Dose'
      example:
        inputs:
          Dose: 100
        expected_formula: 0.680 * 100
    - id: E2
      formula: N_abs = Phi * A * (1 - exp(-alpha * t))
      role: generates
      source: 'derivation: incident photons * absorption probability = absorbed photons => N_abs = Phi
        * A * (1 - exp(-alpha * t))'
      example:
        inputs:
          Phi: 68.0
          A: 1
          alpha: 0.005
          t: 35
        expected_formula: 68.0 * 1 * (1 - exp(-0.005 * 35))
    - id: E3
      formula: Relative Noise = 1 / sqrt(N)
      role: analyses
      source: given in the topic
    - id: E4
      formula: P(0) = exp(-N)
      role: analyses
      source: given in the topic
```
