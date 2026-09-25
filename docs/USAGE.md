# Using Frontier Insight

How to run quests from VSCode chat or the command line, plus the
YAML config schema.

## In VSCode (recommended)

Open Copilot Chat, type `@fi`. The chat participant exposes these
commands:

| Command | What it does | LLM calls |
|---|---|---|
| `@fi` *(no command)* | Starts the interactive interview — 8 quick questions (topic, title, outputs, paper format, research approach, clarify mode, reviewer panel, knowledge layer), produces a config, runs the quest. Best first-time path. | ~23–28 (one full quest, see below) |
| `@fi /new` | Same as bare `@fi`. | ~23–28 |
| `@fi /start <path-to-yaml>` | Runs a quest from an existing YAML config. | ~23–28 |
| `@fi /fleet <yaml> <yaml> ...` | Runs multiple quests in parallel. Each YAML's `provider.node_models` is honored independently. | ~23–28 × N quests |
| `@fi /resume` | Shows a picker of every quest with a checkpoint; pick one to re-enter from the last completed node. | depends on how many nodes the prior run completed; usually 3–10 to finish from a partial run |
| `@fi /resume <quest_id>` | Resumes that specific quest directly. | same — 3–10 to finish |
| `@fi /plan <quest_id>` | Opens the quest's `plan.md` (what the literature says, the gap, the design) beside the chat to read and edit. | **0** |
| `@fi /plan <quest_id> <what to change>` | Has the model rewrite `plan.md` as you ask; the old version is kept. Then `@fi /resume <quest_id>` runs it. | **1** |
| `@fi /summarize <folder> [kind]` | Walks a folder of mixed content (papers, code, study notes, logs) and writes a structured markdown summary. Optional `kind` ∈ `{auto, literature, code, study, execution, mixed}` — defaults to `auto`. | **1** (single LLM call, content cap'd) |
| `@fi /proposal <topic>` | Pre-quest planning doc. Writes both a markdown proposal and a companion YAML under `outputs/_drafts/`. Use to scope a research question BEFORE committing compute to a full quest. | **1** |
| `@fi /analyze <data-path> <topic>` | No-simulation quest on pre-staged data. Files under `<data-path>` are copied into the new quest's `data/` directory; the engine routes `auto_collect_data → wait_for_data → data_load → analyze → write → review`. Inverse of `/proposal` — when you already have the dataset and just want a paper analyzing it. | **~6** |
| `@fi /digest [days]` | Weekly project-manager digest across your quests: completed, in-progress, themes, ✅/🆕/⚠️/🛑/❓ diff vs prior digest, suggested next quests. Default window: 7 days. | **1** (or 0 if window is empty) |
| `@fi /portfolio` | All-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, prioritized next quests. | **1** (or 0 if no quests on disk) |
| `@fi /critique <quest_id>` | Adversarial second-pass review of a completed quest. For maximum effect, pick a different Copilot model in the picker from the one that wrote the paper. | **1** |
| `@fi /help` | Lists the commands. | **0** |

### Per-quest LLM call breakdown

A single \`/start\` or \`/new\` quest made **23–28 LLM calls** in 17 complete runs of one SIR simulation quest (gemma4 through Ollama; default engine settings — `clarify_mode: off`, a single reviewer, `cross_check_per_finding_k: 3` — plus `knowledge.source_routing: manual`, slides and a poster), counted from each quest's `.fi/cost.jsonl`, which records every model call under its node's name:

| Node | Calls | Notes |
|---|---|---|
| `clarify` | 0–1 | Only when `engine.clarify_mode != off`. |
| `ideate` | 1 | |
| `ideate_reflect` | 0–1 | Optional self-reflection that can swap the chosen idea. Skipped when `ideate_tournament` is on. |
| `ideate_tournament` | 0 or C(N,2) | Off by default. When on with the default 3 ideas, fires 3 parallel pairwise comparisons (~one round-trip wall-clock) and picks the highest-win-count idea. |
| `literature_query` | 1 per literature pass | Turns the topic into keyword search queries. |
| `literature_foundational` | 1 per literature pass | Names the foundational works a keyword search misses. |
| `literature_screen` | 1 per literature pass | Grades every retrieved source for citability. |
| `source_router` | 0, or 1 per literature pass + 1 per cross-check lookup | Only with `knowledge.source_routing: auto` (the default); `manual` makes no routing call. |
| `select_skills` | 0–1 | Picks the skills the quest carries; no call when no skill is a candidate. |
| `plan` | 1 | Writes `plan.md` (the literature read as a reviewer would, the gap, the design). It is the design call of the first pass with a plan directive appended, so `design` makes no call that pass; the counts above were measured before this step existed. |
| `plan_revise` | 0 | One call per `--revise-plan` request (two if the first reply's design block cannot be read). |
| `design` | 1 per later design pass | Runs again when the cross-check or the review sends the quest back; the first pass adopts the design block of `plan.md`. |
| `design_self_critique` | 1 per design pass | Audits the drafted methodology against twelve checks (precision, the estimand and its interval, thresholds, random streams, convergence, an oracle, failed runs among them) and keeps what it found and changed in `needs/DESIGN_CRITIQUE.json`. |
| `implement_outline` | 1 | |
| `implement_oracle` | 0–3 | One repair per attempt (`engine.oracle_repair_attempts`), only when the script does not answer the plan's oracles when run with `FI_ORACLE=1` (`engine.oracle_check`), plus one retry of a repair call that got no answer (a provider timeout); a protocol with no oracle, or one without numbers, first makes `plan_revise` calls (up to the same number, counted apart). None when the oracles pass or the plan has no protocol. |
| `implement_protocol` | 0–3 | One repair per attempt (`engine.protocol_repair_attempts`), only when the written script differs from the protocol in the plan (`engine.protocol_check`), plus one retry of a repair call that got no answer (a provider timeout); none when it agrees or the plan has no protocol. |
| `implement` | 1 per design pass | |
| `execute_reflect` | 0–3 | Only when the experiment fails; capped by `engine.exec_reflect_max_iterations`. Also when the finished run's manifest differs from the frozen protocol (`engine.run_manifest_check`, `engine.run_manifest_repair_attempts`). |
| `analyze` | 1 per design pass | |
| `cross_check` | 0–10 | One per key finding that found related literature, up to 10 findings (skipped when `cross_check_per_finding_k = 0`); 0–8 in the measured runs. |
| `evidence_gate` | 0–1 | Sufficiency check before write (`engine.evidence_gate`, default on); it made no call in 7 of the 17 measured runs. A `broaden` verdict re-enters literature once. |
| `web_plots` | 0–1 | No-simulation mode only — one LLM call to chart the collected data (`engine.web_derived_plots`). |
| `write` | 1–2 | Twice when the review asks for a rewrite (14 of the 17 measured runs). |
| `claim_check` | 1 per write | Grounds each paper claim to evidence before review (`engine.claim_grounding`, default on; no call when off). |
| `review` | 1 per write *or* N+1 | 1 for the single-reviewer flow; with a reviewer panel of N personas → N + 1 moderator per round. |
| `slides`, `poster` | 1 each | Only when those outputs are in `output.kinds`. |
| `human_feedback` | 0 | No LLM call — pauses for the user's accept/reject/refine when the gate is on. |

The measured spread came from the experiment repairs (0–3), the number of findings cross-checked, and whether the review asked for a rewrite. Four runs of the same quest on older engine versions, in which design through review ran twice, logged 27–33 calls, not counting slides and poster.

For dollar-cost estimates against specific providers (Copilot, OpenAI, Anthropic, Gemini, Ollama), see [`PROVIDERS.md#cost-expectations`](PROVIDERS.md#cost-expectations).

All LLM calls route through `vscode.lm.selectChatModels` — whatever
model is selected in your Copilot Chat picker is the model FI uses.
See [`PROVIDERS.md`](PROVIDERS.md).

## From the command line

After `pip install frontier-insight`, the `fi` command is on your PATH.

```bash
# Single quest:
fi --config examples/integrator_bakeoff/config.yaml

# Fleet of quests in parallel:
fi --fleet a.yaml b.yaml c.yaml --max-concurrent 4

# Resume a crashed quest:
fi --config outputs/<quest_id>/config.yaml --resume <quest_id>

# Summarize a folder:
fi --summarize ./papers --summarize-kind literature

# Pre-quest planning doc (writes both <id>-proposal.md + <id>.yaml):
fi --proposal "Compare RK4 vs Verlet on the Kepler problem with eccentric orbits"

# Weekly PM digest across your quests:
fi --digest --days 7

# All-time portfolio synthesis (no time window):
fi --portfolio

# Adversarial second-pass review of a finished quest:
fi --critique 1778452404-euv-mor-photon-shot-noise-ler-e6bfe5 \
   --critique-provider claude_cli

# Permanent paper ingest into Axon (no quest):
fi --ingest paper1.pdf paper2.md

# Local web UI:
fi --serve --output-root ./outputs

# Local web UI routing LLM calls through VSCode Copilot — open a
# VSCode integrated terminal launched by the FI extension (which
# injects FI_VSCODE_BRIDGE_PORT into the env), then:
fi --serve --output-root ./outputs
# …or pass the port directly:
fi --serve --vscode-bridge-port 12345 --output-root ./outputs

# One-time tectonic install for corporate envs:
fi --install-tectonic
```

> **First paper_pdf run takes ~30 s longer** when using tectonic (or a fresh MiKTeX install) because the LaTeX engine downloads required CTAN packages on the first compile. Subsequent runs are instant. Tectonic caches under `%LOCALAPPDATA%\TectonicProject\Tectonic\` on Windows; MiKTeX under its own package cache. No additional intervention needed — FI just waits.

### Run it from your own project folder

FI does not have to be run from its own checkout. Run it from your project folder, and everything relative means that folder: the quest goes to `./outputs/<quest_id>/` (the default `output.output_dir` is `./outputs`), `execution.inputs`, `knowledge.local_papers` and the other paths in your YAML are read relative to it, and nothing is written into FI's folder.

```bash
cd ~/my_project
fi --config quest.yaml                                   # pip install: the `fi` command
python /path/to/FrontierInsight/launch.py --config quest.yaml   # a checkout: the same thing
fi --config quest.yaml --resume <quest_id>               # resume from the same folder (or add --output <dir>)
fi --serve                                               # the web UI watches ./outputs and starts quests from this folder
```

`--resume` looks under the folder's `output.output_dir`; if it does not find the quest it says exactly where it looked. In VSCode, open your project folder and set `frontierInsight.repoPath` to the FrontierInsight folder; quests then run in the project (`frontierInsight.workingDir` overrides that). Skills kept in your project (`.claude/skills`, `.agents/skills`, `./skills`) are found from there too.

### All `fi` flags

| Mode | Args | Notes | LLM calls |
|---|---|---|---|
| `--config <yaml>` | one YAML path | single-quest run | ~23–28 (see chat-command section above for the per-node breakdown) |
| `--fleet <yaml> <yaml> ...` | one or more YAMLs | parallel quests, `--max-concurrent N` controls cap | ~23–28 × N quests |
| `--ingest <file> <file> ...` | one or more PDFs / MDs / TXTs | one-shot Axon ingest, no quest | **0** (embeddings only; no LLM) |
| `--serve` | none | starts the FastAPI status GUI at 127.0.0.1:8765 | **0** (GUI is read-only over existing outputs) |
| `--summarize <folder>` | one folder | folder summarizer, pairs with `--summarize-kind` | **1** |
| `--proposal <topic>` | one topic string | pre-quest planning doc + companion YAML under `outputs/_drafts/` | **1** |
| `--analyze <data-path>` | one directory + `--analyze-topic "<topic>"` | no-simulation quest on pre-staged data (files copied into the new quest's `data/`); routes through `auto_collect_data → wait_for_data → data_load → analyze → write → review` | **~6** |
| `--digest` | none | weekly PM digest, pairs with `--days N` (default 7) | **1** (or 0 if window is empty) |
| `--portfolio` | none | all-time cross-quest synthesis (no time window) | **1** (or 0 if no quests on disk) |
| `--critique <quest_id>` | one quest_id | adversarial second-pass review | **1** |
| `--install-tectonic` | none | downloads tectonic to `tools/` for no-admin LaTeX | **0** (network download only) |

| Flag | Mode | What it does |
|---|---|---|
| `--max-concurrent N` | fleet | cap on parallel quests |
| `--memory-cap-mb N` | fleet | throttle new quest starts when RSS exceeds N MB |
| `--profile` | quest | dump per-quest viztracer trace if viztracer installed |
| `--output <dir>` | quest | override `output.output_dir` in the YAML |
| `--interactive` | quest | with `engine.clarify_mode: interactive`, read clarify answers from stdin |
| `--resume <quest_id>` | quest | re-enter a checkpoint, requires `--config` |
| `--summarize-kind <kind>` | summarize | content-type hint, default `auto` |
| `--summarize-provider <name>` | summarize | LLM provider for the summarize call |
| `--days N` | digest | digest window in days, default 7 |
| `--digest-provider <name>` | digest | LLM provider for the digest |
| `--portfolio-provider <name>` | portfolio | LLM provider for the portfolio synthesis |
| `--critique-provider <name>` | critique | LLM provider; set DIFFERENT from quest's original for max adversarial effect |
| `--proposal-provider <name>` | proposal | LLM provider for the proposal |
| `--axon-config <yaml>` | ingest, summarize, digest, portfolio, critique, proposal | optional AxonConfig path |
| `--output-root <dir>` | serve, summarize, digest, portfolio, critique, proposal | quest output dir / scan root |
| `--host`, `--port` | serve | bind elsewhere than 127.0.0.1:8765 |

## YAML config schema

```yaml
# Required: the research question. Free text, multi-line is fine.
topic: |
  Compare three numerical integrators on a damped harmonic
  oscillator (RK4 vs Velocity-Verlet vs forward Euler). Report
  energy drift over 10⁴ periods.

# Optional: short identifier used in folder names. Defaults to a
# slug derived from the topic.
title: integrator-bakeoff

# Optional: `research` turns on, together, what a study needs before its result can be trusted: the plan is held for you
# to read (pauses.plan: ask), the simulation and its analysis stay in two scripts and a reply without both stops the quest
# (execution.split_analysis: true, split_failure: block), the protocol / oracle / numeric-warning / run-manifest checks
# stop the quest and cannot be turned off, and the cross-check verification and a review panel are on. A config that sets
# one of those to the opposite is refused with the key named. Default: default (nothing changes). The interview sets it
# from "What is the result for?": research (the default answer) or a decision writes research; exploring leaves it at
# default and writes the draft's three cheaper engine settings (ideate_reflect: false, cross_check_per_finding_k: 0,
# enable_analyze_reroute: false), the same on every interface. See docs/rigor.md.
rigor_profile: default

provider:
  name: vscode_extension           # see PROVIDERS.md
  model: gpt-5                     # global default
  base_url: null                   # only for HTTP-direct overrides (OpenAI-compatible proxies, local gateways). Honored by openai/codex/gemini/ollama/vllm transports.
  api_key_env: null                # override the standard env-var name (e.g. CORP_OPENAI_KEY). When null, the provider uses its conventional name (OPENAI_API_KEY, GEMINI_API_KEY, …).
  reasoning_effort: null           # minimal | low | medium | high | xhigh | max. Unset (null) sends nothing, so each provider keeps its own default. Sent as `reasoning_effort` (HTTP), `--effort` (claude_cli, antigravity_cli) or `model_reasoning_effort` (codex_cli); a level a provider cannot take is left out with one warning. See PROVIDERS.md, "Reasoning effort".
  fixed_temperature: null          # HTTP providers only. Some OpenAI-compatible models accept one temperature and answer any other with HTTP 400 (Moonshot's Kimi K2.6 / K3: 0.6 with thinking off, 1 with it on). When set, it is sent on every call in place of the per-node temperatures. Not passed to a fallback provider.
  extra_body: {}                   # HTTP providers only. Fields merged into every request body, e.g. Kimi's {thinking: {type: disabled}}, which turns its reasoning off. Not passed to a fallback provider.
  extra: {}                        # forward-compat transport bag. Currently only ``bridge_port`` is consumed (``vscode_extension`` transport, set automatically by ``launch.py``). Other keys parse fine but no transport reads them today — don't rely on stashing CLI flags or HTTP headers here.
  # Per-node override (optional). Match keys exactly to engine node
  # names. Reviewer-panel personas are routed via
  # `review_panel.<persona>`; the moderator via `review_moderator`.
  node_models:
    clarify:       gpt-4o-mini
    write:         claude-3-5-sonnet
    review:        gpt-5

  # Multi-model ensemble per node. Fans out a node's chat call across
  # `models` in parallel and merges with `merge`. Supported on
  # ideate / analyze / cross_check. Cost: N + 1 calls per ensembled
  # node (N fan-out + 1 moderator), except `merge: vote` which is N
  # (no moderator — pure tally). The interview's `ensemble_profile`
  # slot writes this block for you; edit by hand when you want a
  # custom trio or non-default merger.
  node_ensemble:
    ideate:
      models: [gpt-4o, claude-3-5-sonnet, gemini-2.5-pro]
      merge: tournament          # tournament | synthesize | vote
      moderator: claude-3-5-sonnet
    cross_check:
      models: [gpt-4o, claude-3-5-sonnet, gemini-2.5-pro]
      merge: vote                # pure majority tally — no moderator
    # analyze accepts tournament or vote — never synthesize: the
    # analyze parser expects JSON, but synthesize emits markdown, so
    # the ProviderConfig validator rejects that combination at load.

engine:
  framework: langgraph              # the only value supported today
  max_iterations: 2                 # design-revise loop budget
  review_loop: true                 # enable review-driven revise
  audit_trace: true                 # write .fi/audit.jsonl: what ran, each check, each route, the model's stated reasons; see docs/trace.md
  clarify_mode: auto                # off | auto | interactive
  ideate_reflect: true              # extra self-critique pass (1 LLM call)
  ideate_tournament: false          # pairwise tournament across brainstormed ideas; replaces ideate_reflect; C(N,2) calls in parallel
  exec_reflect_max_iterations: 3    # execute-repair loop bound
  pilot_run: false                  # OPT-IN. Run the experiment SMALL first (FI_PILOT=1, which the implement prompt tells the script to honour), then full scale. execute_reflect already repairs a script that CRASHES; the pilot catches one that runs fine and answers the wrong question — a sweep over the wrong parameter range, a resolution too coarse to show the effect — which otherwise costs the full timeout to discover. The pilot's numbers are DISCARDED: it is a smoke test of the design, not a measurement, and it never fails a quest (a bad pilot warns and the full run proceeds). Off by default: honouring FI_PILOT is a prompt instruction the engine cannot enforce, and a script that ignores it runs full-scale under a fifth of the timeout — timing out and warning on every quest. Enable it once your scripts comply.
  pilot_timeout_frac: 0.2           # Pilot timeout as a fraction of execution.timeout_s, floored at 30s. A pilot that takes as long as the real run buys nothing.
  cross_check_per_finding_k: 3      # per-finding lit-check hits, 0 to disable
  enable_analyze_reroute: true      # analyze can request re_experiment / broaden_lit
  review_panel:                     # empty = single reviewer
    - methodologist
    - statistician
    - devil_advocate
    # available: methodologist, statistician, devil_advocate, reproducibility
  no_simulation: false              # see "Topics that need real data" section below
  survey_mode: false                # literature/history synthesis: NO experiment AND NO dataset (implies no_simulation). See "Survey mode" below
  auto_collect_data: true           # try Axon for evidence before pausing for user data (no_simulation mode)
  auto_collect_top_k: 5             # Axon top_k for auto_collect_data
  dataset_adapters: []              # structured-data + web-fetch adapters. Available: "worldbank", "wikipedia"
  dataset_adapter_top_k: 3          # rows per adapter

execution:
  sandbox: venv                     # venv (default) | docker
  timeout_s: 600
  inputs: []                        # example files/folders for the experiment (any type); copied to inputs/examples/, FI_INPUT_DIR
  background_jobs: false            # the simulation runs as an HPC/cluster job: experiment.py submits it and reports pending; --watch wakes the quest
  split_analysis: auto              # auto (default: on for a stochastic design) | true | false. Keep the simulation (code/simulate.py: run_trial, run by FI, record in raw/) apart from its analysis (code/experiment.py); not with background_jobs
  raw_dir: ""                       # only with split_analysis: where the raw files go (relative to the quest, or absolute; relative with docker); empty = raw/
  shared_interpreter: true          # default: run quest code on the Python that runs FI, no per-quest venv
  python_version: "3.11"            # only when shared_interpreter: false (venv per quest)
  system_site_packages: true        # only when shared_interpreter: false; venv sees FI's packages
  docker_image: python:3.11-slim    # for sandbox=docker

knowledge:
  enabled: true
  # Inline AxonConfig (or pass a path to a YAML). Use Axon's NESTED shape
  # as below, not its flat field names — FI hands this to AxonConfig.load,
  # which does the nesting -> field mapping, env overrides and retired-key
  # filtering itself.
  #
  # CAUTION on `embedding`: changing it on a store that already has vectors
  # makes every search fail with `query dimension mismatch: expected 384,
  # got 768` — embedding dimension is a property of the STORE, not of a run.
  # Omit `embedding` to keep whatever the store was built with; only set it
  # for a fresh store.
  # axon_config applies only with axon_mode: in_process. The default, axon_mode: http, uses the Axon service that is
  # already running (started for you when it is not), which keeps the configuration it was started with.
  axon_config:
    embedding: { provider: ollama, model: nomic-embed-text }
    llm:       { provider: ollama, model: qwen2.5-coder:32b }
  top_k: 8                          # Axon RAG cap — dense hits are precise, 8 strong matches beat 20 medium ones for the writer prompt. The interview's "Axon hits per quest" question.
  external_top_k: 20                # External (arXiv / OpenAlex / Crossref / S2 / ...) cap when Axon misses. Bigger than top_k because web search is coarser; bump to 30 for survey-shaped quests.
  relevance_min_score: 0.20         # Literature relevance FLOOR: drop retrieved docs whose embedding cosine vs the TOPIC is below this, before they reach analyze/write. Runs in the literature node for EVERY quest (unlike relevance_guard, which only runs on the auto_collect path), so survey/simulation quests don't carry off-topic sources (e.g. change-point-math papers for a sculpture-history topic). 0.0 disables. Fail-open when embeddings are unavailable (FI_OFFLINE).
  relevance_min_keep: 3             # Never-starve retention: keep at least this many top-scoring docs even if all fall below the floor (the evidence_gate can then broaden).
  requery_on_low_relevance: true    # When NO doc clears relevance_min_score on its own merits, the query was probably worded badly (a field publishes under different terms than the topic statement uses). Ask the model for an alternative query and search again, instead of handing the writer the relevance_min_keep least-bad hits as if they were evidence. Skipped when embeddings are unavailable — without scores there is no signal the query was bad.
  requery_max: 2                    # Bound on those retries. Each costs one small LLM call plus a retrieval.
  literature_screen: true           # One batched LLM call grades every retrieved source 0-3 ("could the paper cite this?"). Papers need 2, web pages are dropped only at 0; keeps at least relevance_min_keep; fails open. Your own local_papers / inputs/papers are never screened.
  foundational_works: true          # One LLM call names up to 8 foundational works (a method's original paper, a standard textbook); each is looked up by title in OpenAlex, or by author and year when no title matches, and kept only if found, plus the works at least 2 retrieved papers cite. Books count. All go through literature_screen, and the writer is asked to cite the ones that bear on the paper. Up to 18 OpenAlex requests per literature pass.
  write_back_quests: true
  write_back_only_on_accept: true

  external_fallback: [openalex, arxiv, crossref]   # arxiv is searched through OpenAlex's arXiv source (arXiv's own query API is throttled for everyone). Also: semantic_scholar, pubmed, core, openaire, doaj (the last three keyless; good for humanities / social science), google_scholar. Crossref / OpenAlex / OpenAIRE keep papers only, plus books and chapters when the quest has no experiment.
  openalex_api_key: ""              # or env OPENALEX_API_KEY. Without a key OpenAlex allows ~100 searches/day; a quest uses dozens. Env wins over YAML.
  semantic_scholar_api_key: ""      # or env SEMANTIC_SCHOLAR_API_KEY. The keyless pool mostly answers 429.
  source_routing: auto              # auto (LLM picks) | manual
  seed_source_catalog: true

  # Pinned local papers (always first in retrieval):
  local_papers:
    - ~/papers/foundational-paper.pdf
    - ~/papers/local-note.md

  try_fetch_full_text: true         # fetch legal full text: open-access copies, and publisher PDFs your own network can reach. false = abstracts only
  full_text_fetch_timeout_s: 15.0   # per-URL fetch timeout in seconds
  full_text_fetch_total_s: 90.0     # wall-clock cap across all URLs in one query
  full_text_max_kb: 10240           # most text kept of one source, KB (10 MB: every page of a paper, scans read by OCR)
  read_figures: true                # read the values off the papers' figures (saved in data/literature/figures/); model: provider.node_models.figures

output:
  kinds: [paper_md, paper_pdf]
  paper_format: generic             # scientific: generic | neurips | iclr | ieee_access | nature_mi; non-scientific prose: essay | report | policy_brief | whitepaper
  output_dir: ./outputs
  require_pdf: false                # strict mode for paper_pdf — see below
  html_pdf_fallback: true           # when no LaTeX engine: render paper.pdf via pandoc → HTML → headless browser (Edge/Chrome/Chromium). Default on. See below.
  paper_style: latex                # paper.pdf look: latex (Computer Modern article, default) | briefing (FI brand look, HTML-rendered)
  author: ""                        # optional author line on the paper, slides and poster — see below
  affiliation: ""
  contact_email: ""
  url: ""                           # project link; the poster prints it as a QR code
  poster_size: a1_portrait          # a1_portrait (default) | a0_portrait | landscape_48x36

# Reserved free-text steering slot — declared in ``core/config.py``
# but NOT YET wired into any prompt template or ``Engine._chat`` path
# as of today. Parses and round-trips through the schema; setting it
# has no behavioural effect until a future PR threads it into the
# system prompts. Documented here so users see the field exists.
extra_directives: ""
```

### `output.require_pdf` — strict-mode PDF enforcement

By default, if `paper_pdf` is in `output.kinds` but the host lacks
pandoc or a LaTeX engine, the engine emits a WARNING and continues —
the quest runs to completion, writes `paper.md`, and drops a
`paper_pdf_skipped.md` diagnostic file next to the markdown. You
still pay the LLM cost (~15 minutes) but get no PDF.

Set `output.require_pdf: true` to upgrade that warning to a hard
failure in **both** of these moments:

1. **Pre-flight (before any LLM calls)** — the engine checks
   `pandoc` + a LaTeX engine (`pdflatex` on PATH, `tectonic` on PATH,
   or a repo-local `tools/tectonic[.exe]` written by
   `python launch.py --install-tectonic`). If any prerequisite is
   missing, the quest aborts immediately with the install recipe — no
   LLM cost incurred.
2. **Post-LLM compile** — even when prerequisites are present at
   pre-flight, the actual pandoc/LaTeX compile can still fail at the
   end (timeout, nonzero LaTeX exit, output file missing despite
   rc=0). In strict mode these surface as a `RuntimeError` that fails
   the quest, instead of a "completed" quest that silently lacks a
   PDF.

Recommended for unattended / CI runs where a missing PDF means the
output is unusable anyway. Leave at the default `false` for
interactive use where you'd rather have `paper.md` + a diagnostic
than no output at all.

### `output.html_pdf_fallback` — LaTeX-free PDF rendering

A 4th engine tier, **on by default**. When none of `pdflatex` /
`tectonic` / `tools/tectonic` is reachable, FI renders `paper.pdf`
*without* LaTeX: `pandoc` turns `paper.md` into a Computer-Modern-
styled HTML page (the `templates/paper/_html/latexlike.css` theme,
with Latin Modern Roman embedded so it matches the LaTeX `article`
look), then a headless system **browser** (Edge / Chrome / Chromium)
prints it to PDF. Figures and fonts are inlined, so the PDF is
self-contained.

This means a machine with **pandoc + a browser but no TeX
distribution** — common on locked-down corporate laptops where you
can't install MiKTeX — still produces a styled `paper.pdf`. The
pre-flight knows about it too: with `require_pdf: true`, a present
browser satisfies the prerequisite, so the quest isn't aborted just
because no LaTeX engine is installed.

Trade-off: the fallback can't reproduce a venue's two-column LaTeX
class (NeurIPS/IEEE), so it always renders the single-column house
style. Set `html_pdf_fallback: false` to force the strict LaTeX-only
path — then a missing engine skips the PDF (or, with `require_pdf:
true`, aborts) exactly as before.

### Chinese, Japanese and Korean text

pdflatex stops at the first Chinese, Japanese or Korean character,
whether it is in the title, the body or the author line. When
`paper.md` or the author line has such text, FI compiles the paper
with **XeLaTeX** and the `xeCJK` package in an installed CJK font. The
template and its layout stay the same.

- **Font:** the first one installed of Noto Sans CJK / Source Han Sans,
  then the system font for the language. That is Microsoft JhengHei
  (Traditional Chinese), Microsoft YaHei (Simplified), Yu Gothic or
  Meiryo (Japanese), or Malgun Gothic (Korean). FI finds fonts through
  fontconfig (`fc-list`, which ships with MiKTeX and TeX Live) or, on
  Windows, in the fonts folder.
- **XeLaTeX** comes with MiKTeX and TeX Live. tectonic is XeTeX
  underneath and works as is.
- **Linux:** install a CJK font first, for example `sudo apt install
  fonts-noto-cjk`.
- **No XeLaTeX or no CJK font:** the paper goes straight to the HTML
  fallback above, which sets the text in the browser's fonts. With
  `html_pdf_fallback: false` it is skipped with a `cjk_no_xelatex` or
  `cjk_no_font` diagnostic.

### `output.paper_style` — choose the paper.pdf look

`latex` (default) renders `paper.pdf` with the venue LaTeX template —
the classic Computer Modern article. `briefing` instead renders the
Frontier Insight **"Research Briefing"** look: warm off-white paper, a
deep-teal accent, a serif display face, and the brand mark — the same
identity as the slides and poster. It's produced by the same
HTML/Chromium backend as the fallback above (pandoc + a browser, no
LaTeX), so it works on a machine without a TeX distribution, and falls
back to the LaTeX path with a warning when pandoc + a browser aren't
both present. Because it's HTML, it's single-column regardless of
`paper_format`. Pick it in YAML (`output.paper_style: briefing`) or
during the interview (`--new` / `@fi /new` / web — the *Paper style*
question).

### Author line and poster size

`output.author`, `output.affiliation`, `output.contact_email` and
`output.url` put your name on the outputs. The paper prints them under
the title (and uses the author as the PDF's Author field), the slides put
them on the title slide, and the poster puts them in its header, with the
link as a QR code. Every field is optional: with no author set the byline
stays "Frontier Insight", and a field left empty is simply not printed.
The first interview on any of the three interfaces asks for them (press
Enter to skip any); they are kept in `~/.frontier-insight/profile.json`
(`FI_PROFILE_PATH` moves it), which all three read, so later interviews
fill them in without asking and show them on the review screen, where a
change is kept for the next quests too.

These values are written only into the quest's own files and your profile file on this machine (the web page
reads and keeps it only for a page opened on this machine; a request that came through a proxy
saying so is refused, but a plain port forward to this machine looks local, so do not expose
`--serve` that way). They are not
sent to the literature or web search services. If the visual check of the
outputs is on, the page screenshots it sends to your configured LLM
provider show the author line, as they show the rest of the paper.

`output.poster_size` picks the poster sheet: `a1_portrait` (59.4 × 84.1 cm,
two columns, the default), `a0_portrait` (84.1 × 118.9 cm, two columns) or
`landscape_48x36` (48 × 36 in, three columns). It is an advanced
interview question.

Changing any of these on a finished quest takes effect when the output is
rendered again: `python launch.py --config <yaml> --resume <id> --emit poster`
(or `paper_pdf`, `slides`).

### The poster

`poster.pdf` follows published poster guidance rather than squeezing the
paper onto a page.

- **Header:** the paper's main finding is the headline. The paper's title
  goes under it, then the author line. A QR code to `output.url` sits on
  the right when a link is set.
- **Type:**
  - body text is 26 pt on A1 and 36 pt on A0 and 48 × 36 in;
  - headings are about 1.5 times the body, and the headline 80 pt or more;
  - figure captions are numbered, and lines run about 60 characters.

  Type is never shrunk to fit.
- **Content:** the model writes headings, short texts, bullet lists and
  figures with captions to a word budget for the sheet. The generator
  writes all the LaTeX, so a slip in the model's formatting cannot break
  the compile.
- **References:** only the sources the poster cites, at most 8, each as
  author, year, title, venue and DOI. A web page shows its site name,
  never a raw URL. A poster that cites nothing lists five selected sources.
- **Fitting:** FI plans the columns from estimated block heights, compiles
  the poster, and measures the PDF. Each measurement corrects the plan.
  The header and reference band are only known after the first compile, so
  that first plan never cuts content. When a compile measures more room
  than the plan assumed, FI plans again for the measured room. When
  content runs off the sheet or into the reference band, FI cuts in this
  order, stopping as soon as it fits:
  1. Figures narrow, down to 70% width.
  2. The longest list loses its last items.
  3. The longest text loses its last sentences.
  4. Text blocks, and then figures, are dropped from the middle.

  The opening and closing blocks always stay. Short columns are carried
  down with extra space before their headings. FI stops after at most six
  compiles.
- **Report:** `.fi/poster_fit.json` in the quest folder records the sheet,
  the number of compiles, the figure widths, what was cut, the final
  measurements, and any findings still open (for example, columns that end
  a few centimetres apart). `.fi/poster_reply.txt` keeps the model's reply.
- **Chinese, Japanese or Korean** text on the poster compiles with XeLaTeX
  and a CJK font, as for the paper. Without either, the poster is skipped
  with a `cjk_no_xelatex` or `cjk_no_font` diagnostic.

### The visual check

After the outputs render, FI checks each PDF the pass produced: the paper,
the slides and the poster. With [LibreOffice](INSTALL.md#system-tools-optional)
installed, `slides.pptx` is exported to PDF and checked too. LibreOffice shows
the deck's equations as their readable text form, so that is what the check
sees; PowerPoint shows them as native equations.

- **What runs:** the PDF is measured (font sizes, overflow, columns, and a
  paper page left half empty before the paper ends) and screenshotted into
  the report folder. No model is asked by default.
- **Figures on slides:** a figure's tick labels must be at least 8 pt on its
  slide. A slide shows a figure at a fraction of the width it was drawn at, so
  the check takes the tick size the figure was drawn at (from the figure's
  record in `.fi/figure_records/`, the house style's size for an older
  record) times that fraction. A figure that shares its slide with text and
  comes out under 8 pt is a finding the slides are redone for: the model is told to
  give the figure a slide of its own, with no bullets, and to put its
  discussion on the next slide. A figure that already has its slide and is still
  under 8 pt (too many panels for one slide) is reported, not redone. Figures
  without a record (fetched web figures, and every figure under
  `execution.sandbox: docker`) are not measured. The redo is the usual one, at
  most `visual_check_max_redos` times.
- **AI check (`visual_check_ai: true`):** the screenshots, the measurements
  and a fixed checklist also go to your configured provider in one call. The
  checklist asks only what a script cannot see, such as raw LaTeX showing as
  text or a figure that covers a caption. It is off by default because it
  costs about 12,000 tokens a quest.
- **Privacy:** with the AI check on, the screenshots, and so everything
  printed on the pages (including the author line), go to that provider.
- **Grounded findings:** each finding must quote text visible where the
  problem is. A finding that cannot be placed on its page is dropped; the
  report keeps it with the reason.
- **Redo:** when the check finds problems a new version can fix, the slides
  or the poster are generated again with those problems in the prompt. The
  version that checks best is kept. A redo is skipped for poster layout
  findings (empty space, uneven columns), which a new reply would not change.
  The paper is never rewritten: when its last page holds only a line or two,
  it is recompiled once with a text area one line taller. `slides.pptx` is
  checked once, after the slides have settled: it comes from the same
  `slides.md`, so a slides redo already made a new one.
- **Report:** `.fi/visual_check.json` in the quest folder, with the
  screenshots under `.fi/visual_check/<output>/`. The run prints one line
  per output, for example
  `[FI] visual check slides: 0 problem(s) seen on the pages, 1 measured; redone 1 time(s), kept redo 1`,
  or `[FI] visual check slides.pptx: not checked (LibreOffice was not found)`.
  The web quest page shows the same lines.
- **Providers without image input** (see
  [PROVIDERS.md](PROVIDERS.md#which-providers-can-see-images)) check from the
  measurements alone, and the line says so.

```yaml
output:
  visual_check: true          # false turns the check off
  visual_check_ai: false      # true also asks your provider to look at the screenshots
  visual_check_max_redos: 2   # 0 to 2 new versions of the slides or poster
```

### `execution.sandbox: docker` — what it actually does

When you set `sandbox: docker`, FI runs the generated experiment inside a Docker container instead of a fresh Python venv. The defaults:

- **Image**: `python:3.11-slim` (override via `execution.docker_image`).
- **Network**: disabled (`--network none`) — the experiment can't reach the internet, which prevents accidental literature scraping or data exfiltration from generated code.
- **Mount**: the quest output directory is bind-mounted at `/work` inside the container; the experiment's working directory is `/work`. Code reads/writes there.
- **Skills from other agents**: each external skill (found in `~/.claude/skills`, `~/.codex/skills`, ...) that you approved and the quest selected is bind-mounted **read-only** at `/fi-skills/<name>`, and the prompts give that path instead of the host one. Nothing else of yours is mounted; a skill folder that is a symbolic link or leads outside its skills folder is refused and `run.log` says so.
- **Lifetime**: a fresh container per execute step. State doesn't persist between retries — the execute-repair loop sees a clean environment each iteration.

Requires the `docker` Python package (`pip install docker`) and a running Docker daemon. On Windows that means Docker Desktop or WSL2. If you don't have those, leave the default `sandbox: venv` — the per-quest venv is faster anyway.

### `output.paper_format` — which templates ship fully styled

**Scientific (IMRAD):**

| Format | Status |
|---|---|
| `generic` | ✅ Fully styled — IMRAD with default LaTeX article geometry. The default; pick this when you don't have a target venue. |
| `neurips` | ✅ Fully styled — uses the NeurIPS 2024 style sheet. |
| `iclr`, `ieee_access`, `nature_mi` | ⚠️ Minimal stubs — they compile, but the style sheets are placeholders. Treat as starting points; copy the real venue's `.sty` file into `templates/paper/<format>/` to customize. |

**Non-scientific (prose, IMRAD-free):**

| Format | Status |
|---|---|
| `essay` | ✅ Long-form argumentative prose. Wider margins, 1.5× line spacing, serif body. Picks the **essayist** write-persona — opens with thesis, marshals evidence, closes with implications. No IMRAD headings. |
| `report` | ✅ Consulting executive report with cover page + TOC. Sans-serif body. Picks the **senior consulting analyst** persona — exec summary → findings → recommendations. |
| `policy_brief` | ✅ 2-4 page brief for policymakers. Tight margins, header strip, dense layout. Picks the **policy analyst** persona — issue → context → single recommendation. |
| `whitepaper` | ✅ 8-20 page industry analysis. Cover with whitepaper subtitle, modest TOC, sans-serif. Picks the **industry analyst** persona — problem → approach → evidence → conclusions. |

The `clarify` node's `paper_venue` slot accepts both buckets. For
non-simulatable topics (set via the new `simulatability` clarify
slot or legacy `empirical_vs_theoretical: empirical`), the agent
will default to `essay` instead of `generic`. The write-persona
swap is automatic — set `paper_format: policy_brief` in YAML and
the `write` node loads the policy-analyst voice.

### `--fleet` concurrency model

`--fleet` runs each YAML in its own asyncio task. The cap is `--max-concurrent N` (defaults to `min(4, cpu_count)`). When RSS exceeds `--memory-cap-mb`, new quest starts pause until memory drops below the cap. Each quest has its own venv, its own `state.sqlite` checkpoint, and its own provider proxy — failure in one quest doesn't affect the others. Provider proxies are reference-counted across the fleet, so 4 concurrent quests all using `vscode_extension` share one bridge connection rather than spawning four.

## Output artifacts

After each quest, `<output_dir>/<quest_id>/` looks like:

```
paper/paper.md                        ← the IMRAD paper
paper.pdf                             ← if pandoc + a LaTeX engine
slides.md / slides.pptx / slides.pdf  ← if `slides` is in output.kinds
poster.tex / poster.pdf               ← if `poster`
talk.md                               ← if `speech`
figures/*.png                         ← every plot the experiment produced
code/experiment.py                    ← the exact code that ran
config.yaml                           ← copy of the YAML for /resume
.fi/run.log                           ← full run log
.fi/state.sqlite                      ← LangGraph checkpoint
frontier_insight_summary.json         ← machine-readable index
```

## Resuming a crashed quest

```bash
# From the terminal:
fi --config outputs/<quest_id>/config.yaml --resume <quest_id>

# From VSCode chat:
@fi /resume
# pick from the list
```

The engine reads the `state.sqlite` checkpoint, detects what node
last completed, and continues from there. No state is lost — the
`paper.pdf` engine, the YAML's `provider` block, every clarify answer
flows through as if the original run never crashed.

## Topics that need real data (not simulation)

Some research questions can't be answered with a Python script —
*"Compare Belgium and Taiwan culture: collectivism, work-life
balance, public-trust dynamics"* needs real surveys and observations,
not invented numbers.

**How the engine decides to enter no-simulation mode** (in this
precedence — first match wins, decision is logged to `run.log` as
`[clarify] simulatability resolved: ... source=<...>`):

1. `engine.no_simulation: true` in YAML — explicit user override.
   `source=yaml`.
2. The clarify question `simulatability` — the agent asks
   *"can a Python script meaningfully simulate this, or does it
   need real-world data?"* with `default: yes | no | uncertain`
   plus a one-line `reason`. `no` triggers no-simulation
   (`source=clarify_simulatability`); `yes`/`uncertain` keeps the
   simulation path. In `clarify_mode: interactive` you see the
   agent's default + reason and can override; in `clarify_mode:
   auto` the default is accepted but the reason is still logged.
3. Legacy fallback: when the new `simulatability` slot is absent
   (older clarify prompts) the existing `empirical_vs_theoretical:
   empirical` answer still triggers no-simulation
   (`source=clarify_empirical_legacy`).
4. Otherwise: simulate (`source=default`).

### Survey mode (a history / overview — no experiment AND no dataset)

Some topics are neither a simulation nor a data-analysis: *"the
evolution of X"*, *"a history of Y"*, *"an overview of Z"*. These want
a **descriptive synthesis of the published literature** — there is
nothing to measure and no dataset to collect. That is **survey mode**,
a stronger form of no-simulation. Turn it on any of three ways:

* `engine.survey_mode: true` in YAML — explicit override (also forces
  `no_simulation: true`).
* Pick **"Literature synthesis (no experiment)"** as the research
  approach in the interview (CLI / web / VSCode).
* Automatically — the clarify step classifies the topic shape as
  `survey` (triggers: "history of", "evolution of", "overview of", …).

In survey mode the graph skips BOTH the experiment (`implement` /
`execute`) AND the whole data path (`auto_collect_data` →
`wait_for_data` → `data_load` → `web_plots`), routing `design →
web_figures → analyze → write`. `web_figures` still embeds
license-clean illustrative images; `analyze` synthesises the reviewed
sources directly; the paper is a narrative history with a References
section and no fabricated metrics.

For the no-simulation (data-analysis) path — where there IS a dataset
to analyse — the engine runs `clarify → ideate → literature → design →
auto_collect_data → wait_for_data → data_load → analyze → ...`. The
two no-simulation-specific stops:

**`auto_collect_data`** — agent-side data collection. Before
pausing for user input, the engine asks the Knowledge layer (Axon)
for relevant docs using `topic + design.hypothesis` as the query
and writes the top hits into `<quest_root>/data/auto_collected/`
as one Markdown file per doc (with YAML provenance front matter so
the paper can cite back to specific sources). Controlled by:

* `engine.auto_collect_data: true` (default) — try Axon first.
  Set to `false` for "user-only" data flow when you don't trust
  the corpus or want a manual pause every time.
* `engine.auto_collect_top_k: 5` (default) — Axon hits requested.
  5 fits comfortably in a 16k-context data_load prompt; raise
  only on long-context models with topics that genuinely need
  more breadth.

**Dataset adapters**: in addition to the corpus-RAG retrieval,
`auto_collect_data` can invoke structured-data adapters that hit
public APIs and write tabular evidence into
`<quest_root>/data/auto_collected/<adapter>/`. Opt in via:

* `engine.dataset_adapters: [worldbank, wikipedia]` — list of
  registered adapter names. Empty (default) means "Axon only —
  no adapters run". Available adapters: `worldbank`, `wikipedia`.
  Unknown names log a WARNING and are skipped (no hard error on typo).
* `engine.dataset_adapter_top_k: 3` — rows per adapter. Smaller
  default than the Axon knob because each row hits an external API.

The WorldBank adapter heuristically extracts country names from the
query (`"Belgium and Taiwan"` → `[BEL, TWN]`, falling back to
global aggregates `WLD/OED/EUU/HIC/LIC` when no country named),
scores ~1500 indicator names against the query keywords, and writes
the top `top_k` matches as Markdown tables with the last 5 years of
data. Adapter failures (network down, indicator not found, every
write failing) fall through silently to the safety net.

The Wikipedia adapter handles the long-tail "qualitative comparison"
case where neither corpus-RAG nor structured-data fits — e.g.
*"compare the 1968 student protests in Paris and Mexico City"*. It
attempts to compress the query to its top-6 informative keywords
(falls back to the raw trimmed query if every token was filtered
out as a stop-word), calls `api.php?action=opensearch` to get
candidate article titles, then fetches
`api/rest_v1/page/summary/<title>` for each match and writes a
Markdown file per article. The page's `description` and `extract`
land in the document body; YAML front matter carries `source:
wikipedia`, the canonical `title` / `url`, the article's
`wikipedia_type` (e.g. `standard`, `disambiguation`), and
`adapter: wikipedia` for downstream provenance. Articles with
extracts under ~200 characters are dropped as too thin to cite.
Request budget per quest: `1 + dataset_adapter_top_k` HTTP calls.

Auto-collect falls through to the user-data pause (no files
written, `auto_collected_count: 0` in state) in four cases:

* `engine.auto_collect_data: false` — INFO log, **no Axon call**.
* `knowledge.enabled: false` — INFO log, **no Axon call**.
* `Knowledge.asearch` raised — WARNING log; Axon **was called** but
  the exception is caught so the quest survives.
* Axon returned zero hits — INFO log; Axon was called and answered
  legitimately with nothing.

In all four cases `wait_for_data` then makes the final pause-or-
proceed decision based on `data/` contents (manual drops still
count if you pre-staged some).

**`wait_for_data`** — the user-data pause. With files already in
`data/` (either auto-collected or user-supplied), this node
passes through immediately. If the dir is empty (Axon returned
nothing AND no manual drops), the engine exits cleanly (rc=0)
with an instruction file at `outputs/<quest_id>/data/README.md`
telling you what to drop. Then:

```bash
fi --resume <quest_id>
```

The engine picks up at the `data_load` node: walks every file in
`data/` (including `auto_collected/`), classifies them (csv / json
/ pdf / md / xlsx / png), synthesizes a `result_json` via one LLM
call grounded in the designed measurement plan, and then continues
normally through `analyze → cross_check → write → review`. The
paper cites the *specific files dropped or auto-collected* as
primary sources, not invented data.

Permissive about format — drop whatever's natural. The walker
deduplicates and budget-caps the prompt the same way `/summarize`
does (see `core/summarizer.py`). If a file format isn't text-readable
(images, binaries), the engine lists it in the manifest but doesn't
include its contents in the prompt — caption it in an accompanying
`.md` for the model to see.

## Common workflows

### Cheap-then-expensive

Light model for the early nodes; strong model only for write/review:

```yaml
provider:
  name: vscode_extension
  model: gpt-4o-mini
  node_models:
    write:  claude-3-5-sonnet
    review: gpt-5
```

### Let a local model think

Ollama models answer without reasoning unless the request asks for it:

```yaml
provider:
  name: ollama
  model: gemma4:31b-cloud
  reasoning_effort: high          # Ollama takes low | medium | high
```

### Interactive scoping

```yaml
engine:
  clarify_mode: interactive       # pauses for 7 user-answered questions
```

Run with `fi --config my.yaml --interactive` for the terminal pause-and-prompt
flow, or via `@fi /new` for the VSCode-modal flow.

### Fleet of variations

Stamp out N YAMLs each varying one knob (model, panel, depth), then:

```bash
fi --fleet variants/*.yaml --max-concurrent 4
```
