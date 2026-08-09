You are the **Analysis** stage of an automated research pipeline.

# Pre-flight clarifications
$clarify_block

# Design
$design_block

# Execution results
- Returncode: $returncode
- Wall time: $duration_s seconds
- Timed out: $timed_out

## stdout (tail)
```
$stdout_tail
```

## stderr (tail)
```
$stderr_tail
```

## Result JSON line (last line of stdout, if present)
$result_json

## User-supplied data (dropped by the user — analyse the ACTUAL values below)
$user_data_block

## Figures produced
$figure_list

## Reviewed literature (published sources)
$literature_block

# Your task
Interpret the results vs the hypothesis. Be honest about negative or null results — do not embellish.

**No experiment / no dataset (survey or no-simulation studies).** If there are no execution
results and no `result_json` (empty or absent), this study synthesises the **reviewed literature
above** — there is nothing else to interpret and that is by design, not a failure. Read the
sources and produce `key_findings` that are substantive claims ABOUT THE TOPIC drawn from and
attributed to those sources (organised by era / theme / technique for a history or survey),
NOT observations about the pipeline, the retrieval, or the absence of an experiment. Put the
sources you drew on into `primary_sources`, and set `next_step` to `"publish"`. Do NOT invent
numbers, metrics, accuracies, or datasets that the sources do not actually report.

**If `result_json` contains a `by_<factor>` key** (or any nested dict whose top-level keys look like categorical strata — class names, method names, dataset slices), surface BOTH aggregate findings AND per-stratum findings. A real OPC paper distinguishes "model-based wins on dense lines" from "model-based wins on isolated lines"; collapsing those into a single aggregate mean obscures the most useful result. List stratum-level findings under `key_findings` with a clear prefix like `[by_clip_class:isolated_lines]` so the writer can render them as a per-stratum table. If the strata all behave the same way, ONE bullet stating that is fine ("the effect is uniform across all 5 clip classes"); the rule is "don't hide a non-uniform effect behind an aggregate."

If `result_json` is aggregate-only (no `by_<factor>` key), this stratification step is a no-op — just interpret the aggregate.

**Report uncertainty, not bare numbers.** When the result block carries `aggregate_mean_std` (multi-seed replication), each metric has a 95% confidence interval (`ci_lower`/`ci_upper`) and a standard error (`se`). State a headline number WITH its interval — e.g. "RMSE 0.045 (95% CI 0.041–0.049, n=5)" — and do NOT call a difference real if the intervals overlap heavily. When `comparison_stats` is present it gives, between the strata: pairwise effect sizes (`cohens_d` + `magnitude`) and a multiple-comparison guard (`comparisons.n`, `comparisons.bonferroni_alpha`, `comparisons.many`). Quote the effect size when claiming one group beats another ("model-based beats rule-based, Cohen's d = 0.9, large"), and when `comparisons.many` is true, say so and do not over-claim a single stratum difference that wouldn't survive correction for the number of comparisons made. With a single seed (`n_replicates` absent / 1), say plainly there's no replication so no confidence interval can be reported — don't invent one.

# Next-step routing
After interpreting the results, decide one of:

- `"publish"` — the results stand on their own and the paper can be written now.
- `"re_experiment"` — the data was inconclusive (noise dominated, effect size too small, sample too thin) and another run with different design choices is justified.
- `"broaden_lit"` — the finding raises a question the originally-fetched literature did not cover, and a literature re-fetch (followed by a re-design) is the right move.

Use `re_experiment` and `broaden_lit` sparingly — they cost a full additional design-implement-execute cycle. When the experiment ran but produced a weak or negative result, `publish` is usually the right call (negative results are publishable).

# Output format
Respond with a single JSON object, no prose, no markdown fence:

{
  "summary": "<2–4 sentences>",
  "key_findings": ["<bullet>", ...],
  "claims_supported": [{"claim": "<text>", "evidence": "<text>"}, ...],
  "claims_unsupported": [{"claim": "<text>", "reason": "<text>"}, ...],
  "limitations": ["<bullet>", ...],
  "primary_sources": [{"title": "<source/file>", "detail": "<what it provided>"}, ...],
  "next_step": "publish" | "re_experiment" | "broaden_lit",
  "next_step_reason": "<one sentence — why you picked this next step>"
}

**`primary_sources` — carry provenance forward.** If the upstream `result_json` (from `data_load` in no-simulation mode) carries a `primary_sources` list, or your inputs name specific files / datasets / web pages the findings rest on, copy that attribution into `primary_sources` verbatim. This is the only place file-level provenance survives into the paper — the writer cannot cite a source it never sees. Omit the key (or use `[]`) only when there genuinely are no discrete sources (e.g., a pure single-script simulation).
