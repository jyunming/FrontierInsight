# Using Frontier Insight

How to run quests from VSCode chat or the command line, plus the
YAML config schema.

## In VSCode (recommended)

Open Copilot Chat, type `@fi`. The chat participant exposes these
commands:

| Command | What it does | LLM calls |
|---|---|---|
| `@fi` *(no command)* | Starts the interactive interview — 7 quick questions, produces a config, runs the quest. Best first-time path. | ~8–18 (one full quest, see below) |
| `@fi /new` | Same as bare `@fi`. | ~8–18 |
| `@fi /start <path-to-yaml>` | Runs a quest from an existing YAML config. | ~8–18 |
| `@fi /fleet <yaml> <yaml> ...` | Runs multiple quests in parallel. Each YAML's `provider.node_models` is honored independently. | ~8–18 × N quests |
| `@fi /resume` | Shows a picker of every quest with a checkpoint; pick one to re-enter from the last completed node. | depends on how many nodes the prior run completed; usually 3–10 to finish from a partial run |
| `@fi /resume <quest_id>` | Resumes that specific quest directly. | same — 3–10 to finish |
| `@fi /summarize <folder> [kind]` | Walks a folder of mixed content (papers, code, study notes, logs) and writes a structured markdown summary. Optional `kind` ∈ `{auto, literature, code, study, execution, mixed}` — defaults to `auto`. | **1** (single LLM call, content cap'd) |
| `@fi /proposal <topic>` | Pre-quest planning doc. Writes both a markdown proposal and a companion YAML under `outputs/_drafts/`. Use to scope a research question BEFORE committing compute to a full quest. | **1** |
| `@fi /digest [days]` | Weekly project-manager digest across your quests: completed, in-progress, themes, ✅/🆕/⚠️/🛑/❓ diff vs prior digest, suggested next quests. Default window: 7 days. | **1** (or 0 if window is empty) |
| `@fi /portfolio` | All-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, prioritized next quests. | **1** (or 0 if no quests on disk) |
| `@fi /critique <quest_id>` | Adversarial second-pass review of a completed quest. For maximum effect, pick a different Copilot model in the picker from the one that wrote the paper. | **1** |
| `@fi /help` | Lists the commands. | **0** |

### Per-quest LLM call breakdown

A single \`/start\` or \`/new\` quest fires roughly **8–18 LLM calls** depending on which engine features are enabled:

| Node | Calls | Notes |
|---|---|---|
| `clarify` | 0–1 | Only when `engine.clarify_mode != off`. |
| `ideate` | 1 | |
| `ideate_reflect` | 0–1 | Optional self-reflection that can swap the chosen idea. |
| `literature` | 1 | One synthesis call (literature *retrieval* uses Axon embeddings, not LLMs). |
| `design` | 1–3 | Hits up to 3× if the cross-check loop sends it back. |
| `implement` | 1–3 | Hits 2-3× if the execute-repair loop fires. |
| `execute_reflect` | 0–3 | Only on a `rc != 0` execution; same retry cap. |
| `analyze` | 1 | |
| `cross_check` | 1–3 | One per finding, up to 3 findings typical. |
| `write` | 1–2 | Possibly twice if `review` returns `revise`. |
| `review` | 1 *or* N+1 | 1 for the single-reviewer flow; with a reviewer panel of N personas → N + 1 moderator. |

Floor (~8): clarify off, every loop hits its happy path, single reviewer.  
Ceiling (~18): full clarify + reflect + 1 design retry + 1 implement retry + 1 execute_reflect retry + 3 cross-check findings + 3-persona reviewer panel + 1 revise loop.

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

# One-time tectonic install for corporate envs:
fi --install-tectonic
```

> **First paper_pdf run takes ~30 s longer** when using tectonic (or a fresh MiKTeX install) because the LaTeX engine downloads required CTAN packages on the first compile. Subsequent runs are instant. Tectonic caches under `%LOCALAPPDATA%\TectonicProject\Tectonic\` on Windows; MiKTeX under its own package cache. No additional intervention needed — FI just waits.

### All `fi` flags

| Mode | Args | Notes | LLM calls |
|---|---|---|---|
| `--config <yaml>` | one YAML path | single-quest run | ~8–18 (see chat-command section above for the per-node breakdown) |
| `--fleet <yaml> <yaml> ...` | one or more YAMLs | parallel quests, `--max-concurrent N` controls cap | ~8–18 × N quests |
| `--ingest <file> <file> ...` | one or more PDFs / MDs / TXTs | one-shot Axon ingest, no quest | **0** (embeddings only; no LLM) |
| `--serve` | none | starts the FastAPI status GUI at 127.0.0.1:8765 | **0** (GUI is read-only over existing outputs) |
| `--summarize <folder>` | one folder | folder summarizer, pairs with `--summarize-kind` | **1** |
| `--proposal <topic>` | one topic string | pre-quest planning doc + companion YAML under `outputs/_drafts/` | **1** |
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

provider:
  name: vscode_extension           # see PROVIDERS.md
  model: gpt-5                     # global default
  # Per-node override (optional). Match keys exactly to engine node
  # names. Reviewer-panel personas are routed via
  # `review_panel.<persona>`; the moderator via `review_moderator`.
  node_models:
    clarify:       gpt-4o-mini
    write:         claude-3-5-sonnet
    review:        gpt-5

engine:
  framework: langgraph              # the only value supported today
  max_iterations: 2                 # design-revise loop budget
  review_loop: true                 # enable review-driven revise
  clarify_mode: auto                # off | auto | interactive
  ideate_reflect: true              # extra self-critique pass
  exec_reflect_max_iterations: 3    # execute-repair loop bound
  cross_check_per_finding_k: 3      # per-finding lit-check hits, 0 to disable
  enable_analyze_reroute: true      # analyze can request re_experiment / broaden_lit
  review_panel:                     # empty = single reviewer
    - methodologist
    - statistician
    - devil_advocate
    # available: methodologist, statistician, devil_advocate, reproducibility
  no_simulation: false              # see "Topics that need real data" section below

execution:
  sandbox: venv                     # venv (default) | docker
  timeout_s: 600
  python_version: "3.11"            # used when creating per-quest venv
  docker_image: python:3.11-slim    # for sandbox=docker

knowledge:
  enabled: true
  # Inline AxonConfig (or pass a path to a YAML):
  axon_config:
    embedding: { provider: ollama, model: nomic-embed-text }
    llm:       { provider: ollama, model: qwen2.5-coder:32b }
  top_k: 5
  write_back_quests: true
  write_back_only_on_accept: true

  external_fallback: [openalex, arxiv, crossref]
  source_routing: auto              # auto (LLM picks) | manual
  seed_source_catalog: true

  # Pinned local papers (always first in retrieval):
  local_papers:
    - ~/papers/foundational-paper.pdf
    - ~/papers/local-note.md

  try_fetch_full_text: false        # opportunistic publisher-PDF fetch (host-network only)
  full_text_fetch_timeout_s: 15.0   # per-URL fetch timeout in seconds
  full_text_fetch_total_s: 90.0     # wall-clock cap across all URLs in one query
  full_text_max_kb: 64              # accepted-PDF size cap, KB; larger PDFs are dropped

output:
  kinds: [paper_md, paper_pdf]
  paper_format: generic             # generic | neurips | iclr | ieee_access | nature_mi
  output_dir: ./outputs
```

### `execution.sandbox: docker` — what it actually does

When you set `sandbox: docker`, FI runs the generated experiment inside a Docker container instead of a fresh Python venv. The defaults:

- **Image**: `python:3.11-slim` (override via `execution.docker_image`).
- **Network**: disabled (`--network none`) — the experiment can't reach the internet, which prevents accidental literature scraping or data exfiltration from generated code.
- **Mount**: the quest output directory is bind-mounted at `/work` inside the container; the experiment's working directory is `/work`. Code reads/writes there.
- **Lifetime**: a fresh container per execute step. State doesn't persist between retries — the execute-repair loop sees a clean environment each iteration.

Requires the `docker` Python package (`pip install docker`) and a running Docker daemon. On Windows that means Docker Desktop or WSL2. If you don't have those, leave the default `sandbox: venv` — the per-quest venv is faster anyway.

### `output.paper_format` — which templates ship fully styled

| Format | Status |
|---|---|
| `generic` | ✅ Fully styled — IMRAD with default LaTeX article geometry. The default; pick this when you don't have a target venue. |
| `neurips` | ✅ Fully styled — uses the NeurIPS 2024 style sheet. |
| `iclr`, `ieee_access`, `nature_mi` | ⚠️ Minimal stubs — they compile, but the style sheets are placeholders. Treat as starting points; copy the real venue's `.sty` file into `templates/paper/<format>/` to customize. |

### `--fleet` concurrency model

`--fleet` runs each YAML in its own asyncio task. The cap is `--max-concurrent N` (defaults to `min(4, cpu_count)`). When RSS exceeds `--memory-cap-mb`, new quest starts pause until memory drops below the cap. Each quest has its own venv, its own `state.sqlite` checkpoint, and its own provider proxy — failure in one quest doesn't affect the others. Provider proxies are reference-counted across the fleet, so 4 concurrent quests all using `vscode_extension` share one bridge connection rather than spawning four.

## Output artifacts

After each quest, `<output_dir>/<quest_id>/` looks like:

```
paper/paper.md                        ← the IMRAD paper
paper/paper.pdf                       ← if pandoc + a LaTeX engine
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
not invented numbers. For these, set `engine.no_simulation: true` in
the YAML (or answer `empirical` to the clarify question
`empirical_vs_theoretical` — same effect; YAML wins when set).

The engine then runs `clarify → ideate → literature → design`, pauses
cleanly (rc=0) with an instruction file at
`outputs/<quest_id>/data/README.md`. You go off and collect data —
survey responses, transcripts, downloaded papers, notes, CSVs,
whatever — and drop the files into that folder. Then:

```bash
fi --resume <quest_id>
```

The engine picks up at the `data_load` node: walks every file in
`data/`, classifies them (csv / json / pdf / md / xlsx / png),
synthesizes a `result_json` via one LLM call grounded in the
designed measurement plan, and then continues normally through
`analyze → cross_check → write → review`. The paper cites the
*specific files you dropped* as primary sources, not invented data.

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
