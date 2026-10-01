# Wiki Log

Ingest history timeline. Newest entries on top.

## 2026-10-01 — research quests explore first, then confirm once

Updated by hand: [[explore-then-confirm|Explore then confirm]] (on by default under `rigor_profile: research`, an explicit `phased: false` kept and said; older research quests go on as they began; `not_applicable` for a quest with no experiment of its own, rows held back put back; the plan's cost line; the interview default follows what the result is for; the stale "YAML only" and `.fi/phased/original/` statements corrected). Sources: core/config.py, core/phased.py, core/engine.py, core/interview.py, core/plan.py, core/plan_settings.py.

## 2026-09-30 — the model's decisions made visible

Updated by hand: [[model-reasoning-trace|Model reasoning trace]] (Copilot asked for Claude's thinking by default, Codex reasoning items, `--why` shows the reasoning note and `reasons`, stated reasons from every choosing step). Sources: core/thinking_capture.py, core/why.py, core/engine.py (`_audit_stated_reasons`, `_note_served_model`), core/provider.py, vscode-frontier-insight/src/lm-messages.ts.

## 2026-09-30 — index rebuilt; links by file name

`index.md` rebuilt with the repository's own `scripts/wiki_sync_index.py` (English output; `--check` reports a stale
index or a broken link). Title-only links were rewritten as `[[slug|Page Title]]` so they resolve by file name, which
is how the index, Obsidian and link checkers resolve them.

## 2026-09-30 — first compile: changes merged 2026-09-29 to 2026-09-30

Compiled by hand (the `llm-wiki` skill was not available) from the code on `origin/main` at `8338c30`, checked against the source, not the commit messages. `wiki/raw/` was not used; `index.md` was not rebuilt (no `sync_index.py` in this repository).

Pages created (17):
- Correctness: [[how-fi-judges-correctness|How FI judges correctness]], [[oracle-provenance|Oracle provenance]], [[model-behind-the-numbers|The model behind the numbers]], [[scoring-criteria|Scoring criteria]], [[engine-callable-simulations|Engine-callable simulations]], [[skill-self-tests|Skill self-tests]]
- Study design and running: [[study-types|Study types]], [[explore-then-confirm|Explore then confirm]], [[code-project-and-run-data|The code project and run data]]
- Changing a quest: [[refine|Refine]], [[doing-a-step-again|Doing a step again]], [[quest-map|Quest map]], [[renaming-a-quest|Renaming a quest]]
- Pauses and plumbing: [[pauses|Pauses]], [[provider-outages|Waiting out provider outages]], [[literature-screening-record|Literature screening record]], [[model-reasoning-trace|Model reasoning trace]]

Sources: core/oracle_check.py, core/trial_runner.py, core/evidence.py, core/criteria.py, core/optimisation_plan.py, core/phased.py, core/code_project.py, core/plan.py, core/rerun_from.py, core/quest_title.py, core/number_provenance.py, core/data_shape.py, core/thinking_capture.py, core/provider.py, core/skills/, core/engine.py, core/config.py, launch.py, web/, vscode-frontier-insight/src/. Audit: docs/audits/docs-sync-2026-09-30.md.


## 2026-09-30 — the checks' sources: read more shapes, fill once, three ways on

Pages updated: [[oracle-provenance|Oracle provenance]] (a derivation needs an equation with `=`; shapes read at read time; FI fills once under research; the three ways on at the stop; going on as it is, with a name), [[model-behind-the-numbers|The model behind the numbers]] (parts under other names are read; the section is shown again after a rewrite).

Sources: core/oracle_check.py, core/plan.py, core/accepted_checks.py, core/engine.py, agents/plan_revise.md, launch.py, web/server.py, web/static/quest.html, vscode-frontier-insight/src/extension.ts, vscode-frontier-insight/src/skills.ts.

## 2026-09-30 — retracted sources

Pages updated: [[literature-screening-record|Literature screening record]] (a third entry per literature pass: each DOI looked up in Crossref for a retraction; a retracted source is marked `[retracted]`, cannot ground a claim, and is named on the to-do card; no answer is "not checked").

Sources: core/retractions.py, core/engine.py, core/todo.py, agents/claim_check.md, agents/write.md, agents/write_patch.md.
