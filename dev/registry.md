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
  protocol's exact Cartesian product? Also the two-script split-analysis lint.
- `core/oracle_check.py` — an oracle the script must pass before its main run, judged by the engine against the
  protocol's own expected value and tolerance, not by the script's self-report. Also owns what a repair may propose
  about a check (`proposals`, never applied without a person) and the note that carries the engine's verdicts to
  `analyze` (`analysis_note`).
- `core/metric_spec.py` — a metric spec (estimand, estimator, contrasts) per headline number, and the statistics that
  follow from it; `core/stats.py` is the pure-stdlib estimator/interval/test library underneath it.
- `core/evidence.py` — the six-level evidence ladder (`assess`, `summary_line`, `upgrade` for older records).
- `core/audit_log.py` — the hash-chained, redacted per-quest trace (`STAGE_PROGRESS`/`_ProgressOnly` for the curated
  console/web view live in `core/engine.py`, next to `_quest_logger`).
- `core/numeric_oracle.py`, `core/stat_claims.py`, `core/number_provenance.py` — the three paper-vs-results audits
  (`needs/{numeric,statistics,provenance}_audit.json`) that `internally_reconciled` requires all three of.
- `core/numeric_warnings.py` — solver/runtime warnings from a run's own stderr that a paper must not be written over.
- `core/plausibility.py` — the design's declared `result_assertions` (legal ranges for a result) checked in code.
- `core/goal_coverage.py` — does the experiment use the numbers the topic itself asked for?

## Config, interview, providers

- `core/config.py` — the typed `Config` tree, `rigor_profile: research` (`_RESEARCH_PROFILE`,
  `REQUIRED_REVIEW_ROLES`, `_apply_rigor_profile`), unknown-key auditing.
- `core/plan_settings.py` — the settings a quest was approved with (`.fi/approved_plan.json`), checked at every start
  (`Engine._stop_for_changed_settings`); `interview_update.approve_settings` records a change `--update` approves.
- `core/interview.py` — the single question set shared by the CLI, the web form and VS Code; `core/interview_update.py`
  is mid-quest re-entry.
- `core/provider.py` — every transport (`LLMClient`), `ProxySupervisor`, `missing_api_key`, model pricing.
- `core/provider_models_discover.py` — runtime model-list discovery for the provider picker.
- `core/ensemble.py` — the multi-model fan-out-and-merge primitive a node opts into.

## Knowledge / literature

- `core/knowledge.py` — `Knowledge`, the three-layer retrieval (pinned papers → Axon → external router).
- `core/axon_http.py`, `core/axon_sidecar.py`, `core/axon_endpoint.py` — talking to the shared Axon service: HTTP
  client, sidecar lifecycle (start/reuse/stale-lock clearing), endpoint discovery.
- `core/passages.py` — relevance-ranked excerpt selection over fetched full text.
- `core/trial_runner.py` — the trial contract: FI runs `run_trial` / `run_cell` of `simulate.py` for every setting,
  one process per setting, writes `raw/ledger.jsonl` and `raw/trials.json` itself (`TrialsRunner` in `_node_execute`,
  `run_oracle` in `_oracle_gate`); the older self-looping contract stays in `core/split_run.py` as `self_reported`.
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
  talks to over a local socket.
- `vscode-frontier-insight/src/interview-core.ts`, `interview.ts` — the VS Code interview (mirrors `core/interview.py`
  question-for-question; keep both in sync when the question set changes).

## Platform / diagnostics

- `core/platform.py` — `--doctor`'s machine-capability detection.
- `core/bridge_path.py`, `vscode-frontier-insight/src/bridge-path.ts` — the canonical persistent-bridge socket path,
  kept identical on both sides.

## Prompts (`agents/*.md`)

One `string.Template` file per node/persona; loaded once at engine init (`core/engine.py`). Grep `agents/` for a
topic before writing a new prompt — most nodes already have one, and a persona variant (`write_persona_*.md`,
`review_persona_*.md`) is usually the right way to add a new voice rather than a new node.
