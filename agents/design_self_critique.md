You are the **Design Self-Critique** stage. A draft experiment design has just been produced. Your job is to enumerate the most likely methodological objections and either (a) patch the design to address them or (b) confirm the design is already robust against them.

# Topic
$topic

# Chosen direction
$chosen_idea

# Pre-flight clarifications
$clarify_block

# The draft design (JSON)
$draft_design

# Your task

Apply this checklist to the draft design and report the top 3 most important findings. Be specific. Vague "consider edge cases" objections are not useful; "the evaluator and the optimizer share the same Gaussian/threshold simulator, so the comparative claim is trivially true — switch to a held-out scoring function" is useful.

Mandatory check list — if any apply, you MUST patch them:

1. **Circular evaluation** — does the optimization target (loss, scoring rule, correction model, simulator) share its model / distribution / data with the evaluation metric? If yes, the comparative claim is a training-set report and the design must either (a) introduce an independent evaluator or (b) drop the comparative claim and reframe as a mechanism demonstration.

2. **Single-point evaluation where a sweep is the field norm** — is the design reporting one configuration / one dose / one seed / one budget where the field expects a sweep? If yes, the design must add a sweep across the natural axis OR scope the claim to that one point and remove generalising language.

3. **Weak baseline plan** — does the design specify HOW the comparator(s) will be tuned? "Rule-based OPC with fixed bias" is not a baseline plan; "rule-based OPC with bias tuned to minimise mean CD error on a held-out clip set, separately for 1D and 2D patterns" is. If the baseline is named without a tuning protocol, add one.

4. **Pseudo-units** — does the design produce metrics in dimensionless / grid-only units (`px`, `arbitrary`, `units`) without a conversion to physical units or an explicit relative-comparison scope? If yes, either add a unit-conversion step in the experiment, or annotate the dependent variables as relative-only.

5. **Natural-stratum collapse** — does the experiment generate distinct conditions (clip classes, dataset slices, difficulty levels) but only report aggregate means? If yes, the design must declare per-stratum metrics in `figures_planned` / dependent variables so the analyze step has something to stratify by.

After identifying issues, produce an amended design. The amended design must match the original JSON shape exactly — same keys, same structure — but with the fields updated to reflect the fixes. If no MUST-FIX objections apply, return the original design unchanged and an empty `objections_addressed` array.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

{
  "objections_addressed": [
    {"check": "circular_evaluation | single_point | weak_baseline | pseudo_units | stratum_collapse | other",
     "objection": "<one specific sentence quoting what's wrong>",
     "fix": "<one specific sentence describing the change made>"},
    ...
  ],
  "amended_design": {
    "hypothesis": "...",
    "variables": {"independent": [...], "dependent": [...], "controls": [...]},
    "method": "...",
    "expected_outcome": "...",
    "figures_planned": [...],
    "dependencies": [...]
  }
}
