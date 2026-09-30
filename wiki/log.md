# Wiki Log

Ingest history timeline. Newest entries on top.

## 2026-09-30 — first compile: changes merged 2026-09-29 to 2026-09-30

Compiled by hand (the `llm-wiki` skill was not available) from the code on `origin/main` at `8338c30`, checked against the source, not the commit messages. `wiki/raw/` was not used; `index.md` was not rebuilt (no `sync_index.py` in this repository).

Pages created (18):
- Correctness: [[How FI judges correctness]], [[Oracle provenance]], [[The model behind the numbers]], [[Scoring criteria]], [[Engine-callable simulations]], [[Skill self-tests]]
- Study design and running: [[Study types]], [[Explore then confirm]], [[The code project and run data]]
- Changing a quest: [[Refine]], [[Doing a step again]], [[Quest map]], [[Renaming a quest]]
- Pauses and plumbing: [[Pauses]], [[Waiting out provider outages]], [[Literature screening record]], [[Model reasoning trace]]

Sources: core/oracle_check.py, core/trial_runner.py, core/evidence.py, core/criteria.py, core/optimisation_plan.py, core/phased.py, core/code_project.py, core/plan.py, core/rerun_from.py, core/quest_title.py, core/number_provenance.py, core/data_shape.py, core/thinking_capture.py, core/provider.py, core/skills/, core/engine.py, core/config.py, launch.py, web/, vscode-frontier-insight/src/. Audit: docs/audits/docs-sync-2026-09-30.md.

