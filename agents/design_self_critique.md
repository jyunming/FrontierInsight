You are the **Design Self-Critique** stage. A draft experiment design has just been produced. Your job is to enumerate every methodological objection that applies and either (a) patch the design to address it or (b) confirm the design is already robust against it.

# Your task

Apply this checklist to the draft design and report every item that applies. Be specific. Vague "consider edge cases" objections are not useful; "the evaluator and the optimizer share the same Gaussian/threshold simulator, so the comparative claim is trivially true — switch to a held-out scoring function" is useful.

Mandatory checklist — if any apply, you MUST patch them:

1. **Circular evaluation** — does the optimization target (loss, scoring rule, correction model, simulator) share its model / distribution / data with the evaluation metric? If yes, the comparative claim is a training-set report and the design must either (a) introduce an independent evaluator or (b) drop the comparative claim and reframe as a mechanism demonstration.
   **For a search for the best design (`study_type: find_best_design`)** sharing the physical model is the point (the best fin spacing is sought IN that model), so do not ask for an independent model. Ask instead: are the search settings and the check settings really different (`protocol.optimisation.numerical_settings`: the check recomputes the best designs and the baseline at finer values, and with new seeds when there is randomness), and is the claim confined to the model ("better within model M")? The shared approximation, not the shared model, is what an optimiser exploits: a coarse mesh that happens to underestimate the temperature at one spacing.

2. **Single-point evaluation where a sweep is the field norm** — is the design reporting one configuration / one dose / one seed / one budget where the field expects a sweep? If yes, the design must add a sweep across the natural axis OR scope the claim to that one point and remove generalising language.

3. **Weak baseline plan** — does the design specify HOW the comparator(s) will be tuned? "Rule-based OPC with fixed bias" is not a baseline plan; "rule-based OPC with bias tuned to minimise mean CD error on a held-out clip set, separately for 1D and 2D patterns" is. If the baseline is named without a tuning protocol, add one.

4. **Pseudo-units** — does the design produce metrics in dimensionless / grid-only units (`px`, `arbitrary`, `units`) without a conversion to physical units or an explicit relative-comparison scope? If yes, either add a unit-conversion step in the experiment, or annotate the dependent variables as relative-only.

5. **Natural-stratum collapse** — does the experiment generate distinct conditions (clip classes, dataset slices, difficulty levels) but only report aggregate means? If yes, the design must declare per-stratum metrics in `figures_planned` / dependent variables so the analyze step has something to stratify by.

6. **Precision is not chosen by cost** — does the design say how many runs, samples or trials each headline number rests on, and why that many? "300 runs" is a convenience, not a reason. For a probability near 0.5 the 95% half-width of an estimate from n trials is about 0.98/sqrt(n): 300 trials give about ±0.057 and 900 give about ±0.033. If the design's claim needs a narrower interval than its runs can give, raise the runs or narrow the claim, and state the width the design can support (in `protocol`, a target and the runs that reach it).

7. **The estimand and its interval** — for each headline number, does the design say WHAT is estimated (a probability over pooled runs, a mean over the runs that met a condition, a deterministic value) and how its uncertainty is estimated? A spread across a few batch seeds is not a sample size; a probability wants an interval from the pooled counts and a conditional mean one from the pooled observations (`protocol.ci_method`). A deterministic quantity has a numerical error, not a stochastic interval.

8. **Arbitrary thresholds** — does a conclusion depend on a cut-off the design chose (an outbreak is "more than 5% of N", a peak is "above 0.1")? Then the design must vary it over a stated set, or justify it from the definition, and say the conclusion has to hold across the set.

9. **Random streams** — do different settings and different runs draw from independent streams, or share one (the same seed rebuilt for every setting makes them correlated without saying so)? The design must say how streams are derived (`protocol.seed_policy`), and a deliberate common-random-numbers design must say so and be analysed as paired.

10. **Numerical convergence** — if a solver, a grid or a step size is involved, does the design say how it is shown converged (halve the step or the tolerance and compare) and which conservation law or invariant every run must satisfy?

11. **An oracle** — does `protocol.oracles` name at least one check that does not rely on the script's own numbers being right (a closed form, a limiting case, an invariant, an exact small case, an independent implementation)? If not, add one. Give each a numeric `expected` and `tolerance`, a `case` (the settings of one small run, e.g. `{"dt": 0.1}`) and a `measure` (the name of the number that run returns), so the engine can run the simulation on that case itself instead of trusting a number the script reports; add `order` when the check claims an order of accuracy.

12. **Failed runs** — what happens to a run that fails, diverges or returns a NaN: is it dropped, counted, or repaired, and is the number of such runs reported? Silent exclusion changes the estimand.

Three more, only for a search for the best design (`study_type: find_best_design`, with `protocol.optimisation`):

13. **The baseline** — is `optimisation.baseline` a real design that meets every constraint, with a value for every design variable inside its range, and does it say where it comes from? A baseline made up to be easy to beat makes the improvement meaningless. Fix it, and make one oracle check the baseline design.

14. **The ranges** — does every design variable's range have a physical meaning (a spacing above zero, a thickness a workshop can make) and lie where the model holds (`model.holds_for`)? A search pushes the design to the edges of its ranges, which is where a model stops holding. Narrow the range, or say in `model.holds_for` why it holds there.

15. **Finer check settings** — is every `check` value of `optimisation.numerical_settings` really finer than its `search` value (smaller for a mesh size or a time step, larger for a count of elements; each level at least 1.1 times finer)? A check at the same or a coarser setting cannot tell a real improvement from a numerical error. Fix the values, or leave `check` out and FI uses half, then a quarter, of the search value. And is the one objective the right one: a second goal belongs in `constraints`, not in the objective.

After identifying issues, produce an amended design. The amended design must match the original JSON shape exactly — same keys, same structure — but with the fields updated to reflect the fixes. If no MUST-FIX objections apply, return the original design unchanged and an empty `objections_addressed` array.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

{
  "objections_addressed": [
    {"check": "circular_evaluation | single_point | weak_baseline | pseudo_units | stratum_collapse | precision | estimand | threshold_sensitivity | rng_independence | numerical_convergence | oracle | failed_runs | optimisation_baseline | optimisation_ranges | finer_check | other",
     "objection": "<one specific sentence quoting what's wrong>",
     "fix": "<one specific sentence describing the change made>"},
    ...
  ],
  "amended_design": {
    "hypothesis": "...",
    "study_type": "<only when the draft has it: measure | find_best_design>",
    "variables": {"independent": [...], "dependent": [...], "controls": [...]},
    "method": "...",
    "expected_outcome": "...",
    "figures_planned": [...],
    "dependencies": [...]
  }
}

(If the draft design carries a `protocol` (its `grid`, `runs_per_setting`, `thresholds`, `seed_policy`, `ci_method`, `acceptance`, `oracles`, `model`, `optimisation`), the amended design keeps it and amends it in place: a fix to checks 6 to 15 belongs there. Keep `study_type` as the draft has it. Never drop a key the draft has.)

---

# Inputs

## Topic
$topic

## Chosen direction
$chosen_idea

## Pre-flight clarifications
$clarify_block

## The draft design (JSON)
$draft_design

