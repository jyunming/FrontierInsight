# Module registry

What already exists, by area, so a new PR extends something instead of rebuilding it. This is a **module-level**
index (an entry per file, or per tightly-related group of files, never per function) — for the exact function, open
the file and search; for what a feature *does* from a user's side, see `docs/capabilities-reference.md` instead.
Grep this file for a keyword before adding a new module, a new node, or a new top-level concept; update it in the
same PR that adds, splits or renames one.

## The engine and its state

- `core/engine.py` — the async LangGraph DAG (`Engine`), every node, every gate, `QuestState`/`QuestArtifacts`. The
  biggest file in the repo; read `docs/architecture.md` before adding a node here.
- `core/protocol.py` — the typed `ResearchProtocol` a quest's design may declare.
- `core/frozen_protocol.py` — freezing a protocol before the first run, and the amendment approve/apply/decline flow
  (also used, unmodified, for the tamper-recovery restore — see `propose_tamper_recovery`).
- `core/protocol_check.py` — static check: does the experiment's source hold to the frozen protocol?
- `core/run_manifest.py` — runtime check: does what the simulation says it ran (`run_manifest.json`) match the
  protocol's exact Cartesian product? Also the two-script split-analysis lint, and the strata of a `given` mean
  (`NESTING`, `DERIVED`, `_given_findings`, `strata_coverage`, read by `metric_spec.coverage_gaps`).
- `core/oracle_check.py` — an oracle the script must pass before its main run, judged by the engine against the
  protocol's own expected value and tolerance, not by the script's self-report. Also owns what a repair may propose
  about a check (`proposals`, never applied without a person) and the note that carries the engine's verdicts to
  `analyze` (`analysis_note`).
- `core/metric_spec.py` — a metric spec (estimand, estimator, contrasts) per headline number, and the statistics that
  follow from it; `core/stats.py` is the pure-stdlib estimator/interval/test library underneath it.
- `core/evidence.py` — the six-level evidence ladder (`assess`, `summary_line`, `upgrade` for older records); `_trace_completeness_gaps` reads the trace's `quest_finalized` seal (written last by `Engine._seal_trace`, naming `SEALED_FILES`); `SEALED_LEDGERS` and `SEALED_QUERIES` (the signed record of the search queries, `engine._record_query_set` / `_query_set_standing`; a person's own in `inputs/search_queries.txt`) are required in the seal; `read` / `verify_seal` are how every surface reads `needs/EVIDENCE.json` (a record written before its seal says `trace_seal: pending`).
- `core/audit_log.py` — the hash-chained, redacted per-quest trace (`STAGE_PROGRESS`/`_ProgressOnly` for the curated
  console/web view live in `core/engine.py`, next to `_quest_logger`).
- `core/numeric_oracle.py`, `core/stat_claims.py`, `core/number_provenance.py` — the three paper-vs-results audits
  (`needs/{numeric,statistics,provenance}_audit.json`) that `internally_reconciled` requires all three of.
- `core/numeric_warnings.py` — solver/runtime warnings from a run's own stderr that a paper must not be written over.
- `core/plausibility.py` — the design's declared `result_assertions` (legal ranges for a result) checked in code, including caps the script puts on a result (`clamp_constants`; `direct_caps`: a cap written where a value is reported -- proven at its whole path in the printed result, or a warning (`engine._assertion_warnings`); `caps_of_other_values`: caps proven to be another value's, which do not count for a value on its bound).
- `core/goal_coverage.py` — does the experiment use the numbers the topic itself asked for?

## Config, interview, providers

- `core/config.py` — the typed `Config` tree, `rigor_profile: research` (`_RESEARCH_PROFILE`,
  `REQUIRED_REVIEW_ROLES`, `_apply_rigor_profile`), unknown-key auditing.
- `core/attempt_records.py` — what a quest tried and under which conditions: `.fi/attempts.jsonl` (each run's outcome from `OUTCOMES` and `script_hashes`, each failed-check stop, the quest's four fields from `quest_status`; every line with `context_fingerprint` of a `CONTEXT_KINDS` kind, schema `SCHEMA`, a `record_id`) `.fi/branch_ledger.jsonl` (ideas, design revisions, repairs) and `.fi/model_calls.jsonl` (one line per model call attempt: `model_call_row` / `append_model_call`, written by `Engine._recorded_call` (failed attempts from `provider.CALL_ATTEMPTS`) and `provider.append_cost_row` for the generators; `model_call_gaps` compares it with the engine's per-step `model_call_counts` at the seal and the quest's end; after the seal calls go to `model_calls.after_seal.jsonl`); written by `Engine._record` / `_record_stop` / `_attempt_context`, read by nothing that routes yet.
- `core/attempt_memory.py` — shadow recommendations from past failed attempts: `Index` (failed attempts under the output root), `recommend` (BLOCK / VERIFY / INFO / IGNORE for a decision's context), `record` (`.fi/shadow_recommendations.jsonl`, written by `Engine._shadow` at plan / implement / execute / repair), `score` / `report_lines` (`fi tools shadow-report`); per-decision `KEYS`, `failure_signature` (also on every run, stop and crashed-quest record), one locked `index_for` per output root, lineage by `stamp` / `take` / `settle_without_record` (`parent_shadow_ids`; every plan of a quest is kept until it ends); the index is process-shared derived state alongside `ProxySupervisor` (read-only, locked, newest 500 quests, rebuildable); the worker is a daemon thread; `Engine._shadow` / `_shadow_close` (deadline, discard when late, closed before the seal); `engine.attempt_memory: shadow | off`; recorded only, read by nothing that decides.
- `core/plan_settings.py` — the settings a quest was approved with (`.fi/approved_plan.json`), checked at every start
  (`Engine._stop_for_changed_settings`); `interview_update.approve_settings` records a change `--update` approves.
- `core/interview.py` — the single question set shared by the CLI, the web form and VS Code; a question asked only for
  some earlier answer carries `ask_if`, checked by `question_applies` (the web page and VS Code mirror it; today only
  `second_reviewer_model`, for research or a decision); `core/interview_update.py` is mid-quest re-entry.
- `core/provider.py` — every transport (`LLMClient`), `ProxySupervisor`, `missing_api_key`, model pricing; `LAST_CALL` (who answered the current task's last call, and why its answer ended); `ModelAnswerTruncated` / `ModelAnswerFiltered` / `outcome_of` (an answer cut off at its limit or withheld is never returned as whole; `Engine._pause_for_model_output` turns either into a `model_output` pause); `node_output_limit` (`provider.node_max_tokens` per step); `quest_run_log` (a failed call made outside the engine still reaches the quest's run.log); `_http_streams` / `_post_streamed` (Moonshot calls are streamed).
- `core/provider_models_discover.py` — runtime model-list discovery for the provider picker.
- `core/ensemble.py` — the multi-model fan-out-and-merge primitive a node opts into.

## Knowledge / literature

- `core/knowledge.py` — `Knowledge`, the three-layer retrieval (pinned papers → Axon → external router); `add_quest_artifacts` writes accepted or preliminary kinds by `metadata['standing']` (decided in `Engine._write_back_knowledge`; read back by `engine._is_preliminary_memory` / `_preliminary_reminders`); the copy under the other standing (`STANDING_KINDS`) is removed first (`_retire`, `_delete_in_process`), and `retire_stale_standing` / `retire_stale_ref_spines` back `fi tools tidy-knowledge`; a removal that fails stops the write (`last_writeback_problem`, which the engine puts in run.log and `.fi/knowledge_problem.json` for the card); cited papers' entries (`REF_SPINE`) keep every accepted consumer (`_paper_entry`, `_drop_consumer`, under `_paper_entries_lock`).
- `core/axon_http.py`, `core/axon_sidecar.py`, `core/axon_endpoint.py` — talking to the shared Axon service: HTTP
  client (`AxonHTTPBrain`: ingest, delete_documents, list_sources, search_raw), sidecar lifecycle (start/reuse/stale-lock clearing), endpoint discovery.
- `core/passages.py` — relevance-ranked excerpt selection over fetched full text.
- `core/trial_runner.py` — the trial contract: FI runs `run_trial` / `run_cell` of `simulate.py` for every setting,
  one process per setting, writes `raw/ledger.jsonl` and `raw/trials.json` itself (`TrialsRunner` in `_node_execute`,
  `run_oracle` in `_oracle_gate`); `recorded_values_by_cell` / `given_values_not_run` hold each reported value to the trials of its own settings; under research `recorded_rows_by_cell` / `given_rows_problems` (via `Engine._given_row_findings`) check a mean over a subset trial by trial; the harness hands every trial the protocol's thresholds as `FI_THRESHOLDS` (`run_oracle` too; the proportion's 0/1 under its id and the averaged quantity under the mean's own id, `RETURN_MEMBERSHIP`); the older self-looping contract stays in `core/split_run.py` as `self_reported`. On a cluster (`execution.background_jobs`) `prepare_cluster` / `collect_cluster` run the settings as a job array submitted by `code/submit.py` (`TrialsRunner(submit=...)`); `code_changed_while_queued` compares the code at submission with the code at collection.
- `core/profile.py` — the person's author line, asked on the first interview and kept in
  `~/.frontier-insight/profile.json` for the CLI (`launch._run_new`), the web page (`/api/profile`, saved on
  submit) and VS Code (`interview.ts` `loadProfile` / `saveProfile`).
- `core/thinking_capture.py` — the model's own account of its reasoning: a task-local holder (`open_holder` / `note_thinking` / `add_thinking`) that each transport fills (`provider._reasoning_of` and the streamed `reasoning_content` deltas for HTTP, `provider._stream_thinking` for the Claude CLI, `vscode_bridge.LAST_BRIDGE_THINKING` fed by `bridge.ts`'s `lm_done.thinking`); `Engine._recorded_call` opens it and `Engine._save_thinking` writes `.fi/thinking.jsonl` (redacted, one line capped at `THINKING_LINE_CHARS` and the file at `THINKING_FILE_BYTES`, `call_id` joined to `.fi/model_calls.jsonl`, `provider`/`model` of the answering provider, not sealed, `output.save_thinking`); `LLMClient._chat_impl` and each `FallbackLLMClient` rung reset the holder so one attempt's reasoning is never filed under another's; `lm-messages.ts::lmDoneMessage` (both bridges) cuts the reasoning so the `lm_done` line stays under 48 KiB. `_chat_provenance` also carries `call_id` / `requested_model` / `finish_reason` into every `model_claim` and, through `_last_chat`, into an attempt record's `models_used`, whose `call_id` differs per call.
- `core/why.py` — `--why` / web Why? / `@fi /why`: why a quest stopped, why the review asked for a revision, why the evidence is at its level, why a step decided what it did, from the audit trace and records (no model call). `launch._follow_trace` is `--trace --follow`; `provider.set_model_call_archive` / `append_cost_row(messages=, response=)` keep every model call in `.fi/io/` when `output.save_model_calls` is on.
- `core/todo.py` — the to-do card every stop writes (`NEXT_STEP.md` + `.fi/todo.json`): per-kind decision, recommendation and alternatives (`advice`; under a rigor profile, never a setting it refuses: `refused_settings`, and what to do instead by pause kind and freeze: `research_instead`), the other things waiting (`waiting`), printed by `launch.py`, read by the web `/next-step` endpoint and VS Code.
- `core/receipts.py` — the receipt each required check (evidence gate, design audit, claim check) writes under
  `needs/receipts/`; `core/evidence.py` reads them for `publication_ready`; `engine._stop_once_for_check` is the
  research profile's one stop and retry.
- `core/pdf_text.py` — every PDF FI reads: all pages in reading order (PyMuPDF if installed, else pypdfium2), scanned
  pages by OCR (tesseract, else RapidOCR), a size cap that names its page. Fetched scans are OCR'd after the fetch
  (`knowledge._ocr_scanned`); `engine._literature_entry` / `_content_quality` / `_item_content` keep the whole text on
  disk (`data/literature/.full_text/`) and label what it is.
- `core/pdf_figures.py` — captioned figures cut out of a PDF by layout (drawings above the caption; scanned pages from
  `PdfText.ocr_lines`). `knowledge._pdf_figures` caches them, `knowledge._cut_figures` runs after a fetch,
  `engine._save_literature_figures` puts them in `data/literature/figures/`, and
  `engine._read_literature_figures` (in the `pause_after_literature` node; prompts `agents/figures_pick.md` and
  `agents/figures_read.md`, step `figures`) reads the relevant ones into each source's text.
- `core/arxiv_gate.py` — the one shared queue/backoff/cache for every arXiv connection.
- `core/source_failures.py` — per-quest retrieval-failure bookkeeping, surfaced as `[FI] source failures: ...`.
- `core/figure_sources.py` — license-clean web-figure enrichment for the no-simulation path.
- `core/citations.py` — BibTeX / CSL-JSON rendering of the reference list.
- `core/datasets/` — the `DatasetAdapter` base plus shipped adapters (`worldbank.py`, `wikipedia.py`) for the
  no-simulation auto-collect path.

## Execution / sandboxing

- `core/execution.py` — `Executor` protocol, `VenvExecutor` / `SharedInterpreterExecutor` / `DockerExecutor`,
  `make_executor`.
- `core/experiment_deps.py` — what a quest environment is given before a run: requested packages minus the quest's
  own files, the selected skills' `pip_requires`, library skills on `PYTHONPATH`, one-at-a-time install fallback,
  skill names pip cannot install explained as the skill, the repair note for what could not be installed.
- `core/split_run.py` — the two-script contract (`simulate.py` / `experiment.py`), raw-dir naming, replicate seeding.
- `core/job_watch.py` — background jobs (HPC / a cluster) that outlive one `execute` call; `--watch`.
- `core/example_inputs.py` — staging a user's own example files into a quest.
- `core/plot_style.py` — the shared matplotlib house style every generated figure uses.

## Skills

- `core/skills/registry.py` — discovery, self-tests, what may load.
- `core/skills/selection.py` — which skills a quest's catalogue carries (`layers.py` = general vs. domain layer).
- `core/skills/approval.py` — the human sign-off gate (content-hash pinned).
- `core/skills/scan.py` — a static review of a skill's contents for the person about to approve it.
- `core/skills/known_requirements.py` — the pip packages each preset skill needs (`scripts/import_scientist_skills.py`
  reads it) and the versions a skill must not get (`PINS`); the fallback for a skill imported before provenance
  recorded `pip_requires`.
- `core/skills/importer.py`, `core/skills/scaffold.py` — importing a skill written for another agent, or drafting one
  from existing code.
- `core/skills/mounts.py` — making an approved external skill reachable inside the Docker sandbox.
- `core/skills/usage.py`, `core/skills/selftest_cache.py` — usage provenance, and which exact contents last passed
  self-test under which Python.

## Outputs

- `generation/paper.py`, `poster.py`, `slides.py`, `speech.py` — the four output generators.
- `generation/_visual_check.py` — screenshot + measurement (+ optional AI check) of every rendered output.
- `generation/_pdf_measure.py`, `_pdf_engine.py`, `_pandoc.py`, `_html_pdf.py`, `_office_pdf.py` — PDF rendering and
  measurement plumbing shared across generators.
- `generation/_pptx_slides.py`, `_pptx_math.py` — native-PowerPoint deck rendering and equations.
- `generation/_marp.py` — Marp CLI discovery for `slides.html` / `slides.pdf`.
- `generation/_keywords.py`, `_figure_captions.py`, `_tables.py`, `_cjk.py`, `_skip_md.py` — small shared renderers.
- `core/replot_figures.py` — redrawing a line figure as the mean-over-seeds version.
- `core/rerun_from.py` — `--resume <id> --from <step>`: the step names, the checkpoint just before a step (`checkpoint_before`, used by `Engine.run`), the steps a quest reached with what redoing each does (`reached`, `REDOES`, `listing`; `Engine.rerun_steps` for `--from` with no step, the web `rerun-steps` API and `@fi /resume --from`), and moving that step's and later outputs to `.fi/previous/<time>/`.
- `core/paper_patch.py`, `core/paper_trim.py` — a targeted revise (only the flagged passages) and a page-limit trim.

## Command-line tools and the web/VS Code surfaces

- `launch.py:_bootstrap_or_reraise`, `_relaunch` — the self-setup a missing dependency triggers at the top-level
  import (installs into an activated venv/conda env in place, silently relaunches into an already-complete
  `.venv/`, or does the full ask-and-install-and-relaunch dance); see `docs/capabilities-reference.md#getting-started`
  for the exact decision tree. Every `vscode-frontier-insight` spawn of `launch.py` opts out (`FI_SKIP_BOOTSTRAP=1`)
  in favor of its own `frontierInsight.pythonPath` contract.
- `core/critique.py`, `core/digest.py`, `core/portfolio.py`, `core/proposal.py`, `core/analyze_cli.py`,
  `core/summarizer.py`, `core/state_dump.py` — the one-shot CLI tools (`fi tools <name>`, see `launch.py:
  _TOOL_SUBCOMMANDS`); `core/proposal_seed.py` seeds an interview from a saved proposal.
- `web/server.py` — the FastAPI status server (quest list, log stream, evidence, trace, amendment banner, ...).
- `web/interview_routes.py`, `web/skills_routes.py`, `web/tools_routes.py` — the web UI's interview, skills and
  CLI-tools surfaces.
- `web/quest_launcher.py` — the subprocess pool for quests started from the web UI.
- `vscode-frontier-insight/src/extension.ts` — the `@fi` chat participant and every slash command.
- `vscode-frontier-insight/src/bridge.ts`, `persistent-bridge.ts` — the `vscode.lm.*` bridge a spawned `launch.py`
  talks to over a local socket; `lm_done` carries `served_model` (`lm-messages.ts` `servedModel`), which
  `core/vscode_bridge.py` hands to the call's own `LAST_SERVED` and `core/provider.py` records as `LAST_CALL`.
- `vscode-frontier-insight/src/interview-core.ts`, `interview.ts` — the VS Code interview (mirrors `core/interview.py`
  question-for-question; keep both in sync when the question set changes).
- `vscode-frontier-insight/scripts/normalize-vsix.js` — run by `npm run package` after vsce: sorts the zip entries,
  fixes their timestamp and compression so a build is byte-reproducible (same Node major), and writes
  `vscode-frontier-insight.vsix.manifest.json` (sha256 per packed file; committed with the `.vsix` by
  `.github/workflows/vsix-rebuild.yml`).

## Platform / diagnostics

- `core/platform.py` — `--doctor`'s machine-capability detection.
- `core/bridge_path.py`, `vscode-frontier-insight/src/bridge-path.ts` — the canonical persistent-bridge socket path,
  kept identical on both sides.

## Prompts (`agents/*.md`)

One `string.Template` file per node/persona; loaded once at engine init (`core/engine.py`). Grep `agents/` for a
topic before writing a new prompt — most nodes already have one, and a persona variant (`write_persona_*.md`,
`review_persona_*.md`) is usually the right way to add a new voice rather than a new node.
