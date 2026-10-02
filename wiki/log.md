# Wiki Log

Ingest history timeline. Newest entries on top.

## 2026-10-01 — the card when a known-answer check stops the quest

Updated by hand: [[how-fi-judges-correctness|How FI judges correctness]] (one plain name, "known-answer check", on every screen; when the checks stop the quest one card, from one payload in `.fi/pause.json` and `.fi/todo.json`, shows each failing check's expected value and source, the measured value and who measured it, the tolerance and gap, the case, and where the script computes the number, then the likely cause from what FI already has, what FI tried and two or three ways on; the same in the terminal, NEXT_STEP.md, the web page and VS Code; no verdict changes). Sources: core/oracle_card.py, core/todo.py, core/oracle_check.py, core/engine.py, core/trial_runner.py, web/static/quest.html, vscode-frontier-insight/src/stop-card.ts.

## 2026-10-01 — a copied quest judges its own paper

Updated by hand: [[doing-a-step-again|Doing a step again]] (a copied or moved quest rerun from the claims or the review reads the paper in its current folder; found by the self-benchmark's copies, `dev/evaluation/bench/`). Sources: core/rerun_from.py, core/engine.py, core/evidence.py.

## 2026-10-01 — an accept means the person accepts the evidence

Updated by hand: [[pauses|Pauses]] and [[how-fi-judges-correctness|How FI judges correctness]] (the question before an accept is now "Have you reviewed the evidence record, and do you accept these claims and the limits listed?"; only an explicit yes reaches publication_ready; "I did not check" and "partly" with its required note each leave a gap; the receipt binds who, when, the interface, the note, the paper and evidence-record hashes and the limits listed; the old "do the numbers match what you expected" question dropped), [[model-reasoning-trace|Model reasoning trace]] (it is the reasoning summary the provider returned, never the hidden or complete chain of thought, never evidence; `output.save_thinking: false` for sensitive data). Sources: core/acceptance.py, core/evidence.py, core/engine.py, core/audit_log.py, core/vscode_bridge.py, launch.py, web/server.py, web/static/quest.html, vscode-frontier-insight/src/bridge.ts.

## 2026-10-01 — a changed design is disclosed and is a gap

Updated by hand: [[explore-then-confirm|Explore then confirm]] (a design changed after the first run keeps the result below publication-ready unless a confirm run came after the last change, read from `.fi/phased.json`; FI's own methods paragraph on how the result was reached). Sources: core/disclosure.py, core/evidence.py, core/phased.py, core/engine.py.

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

## 2026-09-30 — the best design checked at finer numerical settings

Pages updated: [[study-types|Study types]] (after the search FI checks the best design at finer settings; one verdict, a sentence per check; a failed check limits the evidence level and never stops the quest).

Sources: core/optimum_check.py, core/optimise.py, core/evidence.py, core/optimisation_plan.py.

## 2026-10-01 — a person accepts the result

Pages updated: [[pauses|Pauses]] (before an accept every interface shows what the result does not guarantee and its main gaps and asks one question; the answer is recorded; "no" does not accept), [[how-fi-judges-correctness|How FI judges correctness]] (an accept no person made stays one level below publication ready).

Sources: core/acceptance.py, core/evidence.py, core/engine.py, core/audit_log.py, core/todo.py, core/vscode_bridge.py, launch.py, web/server.py, web/static/quest.html, vscode-frontier-insight/src/bridge.ts.

## 2026-09-30 — CHANGELOG kinds of change and "Did the results change"

Pages updated: [[code-project-and-run-data|The code project and run data]] (each `code/CHANGELOG.md` entry is Added / Changed / Fixed / Tidied, decided by the step that made it; each has a line saying whether the results changed, "not measured yet" until a run of that code finishes with results).

Sources: core/changelog.py, core/code_project.py, core/criteria.py, core/engine.py.

## 2026-10-01 — retrieved text is data

Pages added: [[retrieved-text-is-data|Retrieved text is data]] (every block of retrieved text in a prompt is fenced between two markers a source cannot forge; text addressed to an AI model or hidden from a reader is flagged, recorded in the trace and run.log, and kept).

Sources: core/source_text.py, core/pdf_text.py, core/knowledge.py, core/engine.py, core/summarizer.py, agents/figures_read.md.

## 2026-10-01 — a data quest's confirm run; which rows belong together; isolation

Pages updated: [[explore-then-confirm|Explore then confirm]] (a quest that analyses one table is confirmed on held-back rows by its own data-reading step run once more with the frozen design; the split is decided before the data is first read from the plan's `protocol.split`, the plan.md line *Rows that belong together*, or the table's columns, and a research quest asks when FI cannot tell; held back counts as unseen only in a container or, without one, encrypted with an in-memory key and a clean scan of the quest's code).

Sources: core/phased.py, core/phased_data.py, core/phased_isolation.py, core/engine.py, core/evidence.py, core/disclosure.py.


## 2026-10-01 — who read the checks, and a setting the code never saw

Pages updated: [[oracle-provenance|Oracle provenance]] (under research a second, different model must read the checks, as the record of the calls names the models, and invariant, symmetry and second-implementation checks run again at a setting the code never saw), [[how-fi-judges-correctness|How FI judges correctness]] (both added to what independently validated needs).

Sources: core/oracle_review.py, core/hidden_check.py, core/evidence.py, core/engine.py.

## 2026-10-02 — FI looks at a failing known-answer check itself before repairing

Pages updated: [[how-fi-judges-correctness|How FI judges correctness]] (before the first repair of a failing check: the expected value worked out again by another model blind to the measurement, a smaller step with Richardson extrapolation, three more seeds, a unit factor, an early stop on the same exception, one retry at twice the time; findings under `attempts[].triage` and on the card; disputes kept across a resume while the check is unchanged).

Sources: core/oracle_triage.py, core/engine.py, core/oracle_card.py, core/oracle_forms.py, core/protocol_check.py, core/trial_runner.py, agents/oracle_recompute.md.
