# 15 — Fault-Injection Red-Team Campaign (PR-5)

**Audit date:** 2026-09-22
**Scope:** For each of the re-audit's four study shapes (SIR / deterministic / paired / clustered), which gate is
supposed to catch which class of bad science, whether a real test exercises that catch today, and what still isn't
caught. Built as pytest, per the user's decision (no live-quest corpus — the existing fixture-based pattern already
covers most of this ground).
**Follows:** the third re-audit ([[project_reaudit_laya_2026_09_22]] in memory; the source document is
`frontierinsight_reaudit_laya_zh-TW_2026-09-22.md`), whose PR order named this PR-5 (its own section 11 calls it
"PR-G": *fresh SIR / deterministic / paired / clustered red-team campaign*).

## Current disposition (2026-09-23)

The 2026-09-23 re-audit (a later, separate document from the one above) found this doc stale in three places. The
analysis below is kept as written — it was accurate against the code at the time — with a marker at each stale
point pointing here, per the re-audit's own instruction to add a disposition note rather than rewrite history.

- **The "names `FI_REPLICATE_SEED` but never uses it" row — fixed by #366.** The runtime decision now also asks
  `replicate_seed_reaches_rng` (an AST check tracing whether the value actually flows into a generator-seeding
  call), so a script that only mentions the variable name is no longer treated as deterministic. The test this row
  cited was renamed from `test_a_script_that_names_but_discards_the_replicate_seed_is_still_called_deterministic` to
  `..._is_no_longer_called_deterministic` (`tests/test_fault_injection.py`) — the old name describes behavior that
  no longer exists. (#366's own PR description calls this fix "P1-5"; this doc's table below labels a *different*
  row, two down, "(P1-5)". Which numbering is canonical is not resolved here — both #365 and #366 are merged and
  described accurately by their own PR text regardless of which label applies to which.)
- **The "(P1-4)" row (design self-critique's report cap) — fixed by #365.** The design-critique prompt now reports
  every applicable finding instead of capping at 5. The `xfail(strict=True)` test this row cited
  (`test_design_self_critique_caps_reported_findings_below_its_own_mandatory_checklist_size`) was renamed to
  `test_design_self_critique_does_not_cap_reported_findings_below_its_own_mandatory_checklist_size` and is now a
  plain regression test, not an xfail.
- **"Not built here" below says P1-4/P1-5 are "the user's own next two PRs"** — stale on both counts; both merged.

Still open, unaffected by the above: the "(P1-5)" row as labeled in this doc's own table (two genuinely-random seeds
producing byte-identical output by coincidence, not caught) and the "(P1-6)" row (a design's wording, not the script
it describes, decides whether `run_manifest` treats the study as stochastic) — neither has a merged fix as of this
note.

## What this found before writing a line

The premise going in was that none of the four shapes had fault-injection coverage yet. That was wrong for three of
the four: PR #356, #359 and #362 each shipped real fault-injection tests as part of fixing the gate they touched.
The actual gap was narrower — one genuinely new source-scan blind spot, and three P1 items whose current state the
2026-09-21/22 audits had recorded from a read of the code that predates the fixes in #356/#359, not the code as it
stands now. Re-verifying every item against the current source (not the audit's prose) was itself part of this PR.

## Coverage table

| Fault class | Shape | Gate | Caught? | Where tested |
|---|---|---|---|---|
| Frozen protocol edited after freeze, hash left stale | any | `frozen_protocol` tamper recovery | **Yes** | `tests/test_frozen_protocol.py` — a full real-graph run: tamper mid-quest → pause → resume without approval pauses again → approve → resume completes on the *restored* content, not the tampered one |
| `run_manifest` swept an axis, or a value, the protocol never listed | SIR (independent) | `run_manifest_check` | **Yes** | `tests/test_run_manifest.py` (27 cases, incl. the audit's own extra-axis counterexample) |
| A script rebuilds its RNG stream for every setting instead of one stream per run | SIR (independent) | `protocol_check.rng_reuse` | **Yes** | `tests/test_precision_rng.py`, real `fr1`/`fr2`/`fr3` scripts in `tests/fixtures/protocol_runs/` |
| A script's grid drifts from the frozen protocol | SIR (independent) | `protocol_check.check` | **Yes** | `tests/test_protocol_check.py::test_the_run_that_changed_its_grid_is_caught_and_the_two_that_did_not_are_not` |
| MetricSpec names an estimator with no `estimand`/`unit` | any | `metric_spec.normalize` | **Yes** | `tests/test_metric_spec.py` (PR #362, 16 cases) |
| Paired design fed mismatched trial counts between two settings in one seed | paired | `metric_spec._contrast` | **Yes** | `tests/test_metric_spec.py::test_a_cluster_design_uses_the_clusters_and_says_what_it_lacks` (the `unequal` case) |
| Cluster design fed no `_clusters` array at all | clustered | `metric_spec._pooled` | **Yes** | same test, the `no_clusters` case |
| Cluster design fed a `_clusters` array that IS present but the wrong length | clustered | `metric_spec._pooled` | **Yes** | `tests/test_fault_injection.py::test_a_cluster_array_of_the_wrong_length_is_refused_not_silently_reshaped` (new — a review of this doc's first draft found the `no_clusters` case above tests a *missing* array, a different code path from a *wrong-length* one, and nothing had exercised the latter) |
| Paired *and* clustered together (unsupported combination) | paired+clustered | `metric_spec._contrast` | **Yes** (refused, not silently guessed) | same test, the `both` case |
| A script *names* `FI_REPLICATE_SEED` (defeats the source scan) but never uses the value; RNG is hardcoded elsewhere | deterministic | the seed-scan heuristic (`_script_reads_replicate_seed`) | ~~No — documented, now proven~~ **Fixed by #366 — see Current disposition above** | `tests/test_fault_injection.py::test_a_script_that_names_but_discards_the_replicate_seed_is_no_longer_called_deterministic` |
| Two seeds happen to produce byte-identical `RESULT_JSON` although the process is genuinely stochastic (a coincidence, not a hardcoded seed) | deterministic | the same heuristic, one level up (P1-5) | **No — open** | not tested; the heuristic compares exactly two data points and calls agreement proof, with no third seed to rule out coincidence |
| A design's wording never says "stochastic" (or any of the ~10 fixed trigger words) although the study genuinely draws random numbers (bootstrap, permutation, a Poisson process) | deterministic | `run_manifest`'s applicability gate (`design_is_stochastic`, P1-6) | **No — open** | `tests/test_fault_injection.py::test_design_is_stochastic_trusts_the_designs_own_wording_not_the_script_it_describes` (new characterization; not an xfail — the fix would need the function to see the script, which it currently cannot) |
| A design self-critique applicable finding goes unpatched because only 5 of 12 mandatory checklist items are ever reported (P1-4) | any | `design_self_critique` | ~~No — open~~ **Fixed by #365 — see Current disposition above** | `tests/test_fault_injection.py::test_design_self_critique_does_not_cap_reported_findings_below_its_own_mandatory_checklist_size` |

## Re-verified against current source (not the audits' prose)

The 2026-09-21/22 audits recorded these from a read of the code at the time; #356 and #359 have since landed. Re-read
against the code on this branch (identical to main at the point this PR started):

- **P1-1 (numeric-scanner fail-open) — already fixed** (#359). A scan that raises now writes an `error` status
  record read by `evidence.py` as a hard gap, not a silent pass. One separate, narrower function
  (`_numeric_oracle_hits`, the paper-vs-results consistency oracle) still has a bare `except Exception: return []`
  with no record — a new finding, not one of the audits' items, logged below and left unfixed here.
- **P1-3 (review panel any-non-empty) — already fixed** (#359). `rigor_profile: research` now checks the panel
  contains `methodologist`, `statistician` and `reproducibility` by name, not merely that it is non-empty.
- **P1-9 (pending amendment consumed before the `before_build` interrupt) — already correct.** `_settle_pending_amendment`
  runs at the top of `_node_design`, before `_maybe_pause_for_user_input(state, "before_build")` later in the same
  call; no path reaches the interrupt with an amendment still unconsumed.
- ~~**P1-4, P1-5, P1-6 — confirmed still open**, per the table above.~~ **Stale for P1-4 — fixed by #365, see
  Current disposition above.** The byte-identical-coincidence row (labeled "(P1-5)" in the table above) and P1-6
  remain open.

## Not built here (converge, don't expand)

- ~~**P1-4 / P1-5 fixes** — the user's own next two PRs in the stated order; this PR proves and records the gaps, it
  does not close them.~~ **Stale — the two PRs that were next (#365, #366) both merged; see Current disposition
  above for exactly what each closed. This does NOT mean the byte-identical-coincidence row labeled "(P1-5)" in the
  table above is fixed — it isn't; #366 closed a different row that its own PR description also called "P1-5" (the
  numbering conflict the Current disposition section flags).**
- **P0-5, a fresh real quest rerun under #356/#359/#362/#363's gates** — still unmeasured. This PR's tests prove the
  gates catch synthetic faults; they say nothing about what a real quest looks like under them today.
- **New finding, not previously listed: `core/engine.py::_numeric_oracle_hits`** has an unguarded
  `except Exception: return []` (distinct from the P1-1 scanner this audit's item names, which was already fixed).
  A crash here silently reads as "no provenance issues found" rather than a gap. Backlog, not fixed in this PR.

## Tests

`tests/test_fault_injection.py` — 4 new (1 characterization proving a documented blind spot is real; 1
`xfail(strict=True)` mutation-tested against both ways the gap could close — the cap raised to cover the checklist,
and the cap phrase removed outright — confirming the suite fails on an XPASS either way; 1 covering the wrong-length
cluster-array path an external review of this PR's own first draft found the table had misattributed; 1
characterization of a trust boundary with no fix shape implied). The rest of the table cites existing tests, run
unchanged.
