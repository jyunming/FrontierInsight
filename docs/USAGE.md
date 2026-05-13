# Using Frontier Insight

How to run quests from VSCode chat or the command line, plus the
YAML config schema.

## In VSCode (recommended)

Open Copilot Chat, type `@fi`. The chat participant exposes these
commands:

| Command | What it does |
|---|---|
| `@fi` *(no command)* | Starts the interactive interview — 7 quick questions, produces a config, runs the quest. Best first-time path. |
| `@fi /new` | Same as bare `@fi`. |
| `@fi /start <path-to-yaml>` | Runs a quest from an existing YAML config. |
| `@fi /fleet <yaml> <yaml> ...` | Runs multiple quests in parallel. Each YAML's `provider.node_models` is honored independently. |
| `@fi /resume` | Shows a picker of every quest with a checkpoint; pick one to re-enter from the last completed node. |
| `@fi /resume <quest_id>` | Resumes that specific quest directly. |
| `@fi /summarize <folder> [kind]` | Walks a folder of mixed content (papers, code, study notes, logs) and writes a structured markdown summary. Optional `kind` ∈ `{auto, literature, code, study, execution, mixed}` — defaults to `auto`. |
| `@fi /help` | Lists the commands. |

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

# Permanent paper ingest into Axon (no quest):
fi --ingest paper1.pdf paper2.md

# Local web UI:
fi --serve --output-root ./outputs

# One-time tectonic install for corporate envs:
fi --install-tectonic
```

### All `fi` flags

| Mode | Args | Notes |
|---|---|---|
| `--config <yaml>` | one YAML path | single-quest run |
| `--fleet <yaml> <yaml> ...` | one or more YAMLs | parallel quests, `--max-concurrent N` controls cap |
| `--ingest <file> <file> ...` | one or more PDFs / MDs / TXTs | one-shot Axon ingest, no quest |
| `--serve` | none | starts the FastAPI status GUI at 127.0.0.1:8765 |
| `--summarize <folder>` | one folder | folder summarizer, pairs with `--summarize-kind` |
| `--install-tectonic` | none | downloads tectonic to `tools/` for no-admin LaTeX |

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
| `--axon-config <yaml>` | ingest, summarize | optional AxonConfig path |
| `--output-root <dir>` | serve, summarize | quest output dir for the GUI / summarizer |
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

  try_fetch_full_text: false        # opportunistic publisher-PDF fetch
  full_text_fetch_timeout_s: 15.0

output:
  kinds: [paper_md, paper_pdf]
  paper_format: generic             # generic | neurips | iclr | ieee_access | nature_mi
  output_dir: ./outputs
```

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
