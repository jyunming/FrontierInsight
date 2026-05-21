# Frontier Insight

<picture>
  <source srcset="web/static/favicon.svg" type="image/svg+xml"/>
  <img src="vscode-frontier-insight/images/icon.png" alt="Frontier Insight icon" width="96" align="left" style="margin-right: 16px;"/>
</picture>

**Give it a research topic. Get back a paper, the experiment that produced it, and the figures — all auto-generated, all reproducible.**

Frontier Insight (FI) is an automated research assistant. You write a one-paragraph topic into a YAML file; FI runs an 11-step research loop:

1. surveys ~9 framing slots (`engine.clarify_mode`: `off` (default) / `auto` / `interactive`),
2. brainstorms research directions (or skips ideate→implement→execute entirely when `simulatability == "no"`, routing through data-ingest instead),
3. searches the literature,
4. designs an experiment, then self-critiques the design for circular evaluation / weak baselines,
5. writes the experiment code in two stages — first a scaffold with function signatures and constant sources, then fills in the bodies (avoids ~9 min one-shot stalls on complex topics),
6. runs it in a sandboxed venv (and fixes its own bugs if the script crashes),
7. cross-checks the results against published papers,
8. drafts the paper — IMRAD (NeurIPS / ICLR / IEEE Access / Nature MI / generic) or essay / report / policy_brief / whitepaper,
9. reviews itself (optionally with a panel of reviewer personas; the methodologist persona hard-flags four common-but-fatal patterns),
10. iterates if the review says "revise" — optionally pauses for human accept / reject / refine,
11. saves paper + figures + code + slides + (optional) poster + speech, plus a reproducible `.fi/requirements.lock.txt`.

Everything runs locally on your machine. The only external dependency is an LLM provider (your choice: OpenAI / Anthropic / Gemini API keys, or your GitHub Copilot subscription, or local Ollama).

---

## Why Frontier Insight?

FI is opinionated about the *whole research workflow*, not just the LLM call. If you're already using one of these, here's why FI might still fit:

| If you're already using… | …FI gives you |
|---|---|
| **Vanilla Copilot Chat / Cursor / Cline** | The full research loop end-to-end — ideate → literature → code → execute → analyze → write → review — without you driving each step. Chat tools wait for you; FI doesn't. |
| **A general-purpose AI agent for science** | **Persistent cross-quest memory.** `@fi /digest /portfolio /critique /proposal` accumulate over weeks via the Axon knowledge layer, so FI remembers what you tried last month. Plus a reviewer panel with per-persona model routing (statistician, methodologist, devil's advocate, …) — most agents start fresh each session and use a single reviewer. |
| **Hand-rolled LangGraph + OpenAI scripts** | The *output layer*: venue paper templates (NeurIPS / ICLR / IEEE Access / Nature MI / generic), non-scientific prose templates (essay / report / policy_brief / whitepaper), slides, posters, speech scripts. Plus structured-data adapters (`worldbank`, `wikipedia`) that pull live evidence into `data/auto_collected/` for qualitative/archival topics, no Python experiment required. Plus resumable quests (`--resume` re-enters from the last LangGraph checkpoint), the sanctioned `vscode.lm` Copilot integration (no reverse-engineered proxies), and a corporate-laptop install path (tectonic LaTeX, no admin needed). |

Concretely: **per-node model routing** lets you spend cheap on `clarify` and expensive on `write`. **Adversarial `/critique`** runs a second-pass review with a different model family than the one that wrote the paper. **`--fleet`** runs many quests in parallel with bounded concurrency. **`--install-tectonic`** drops a 70 MB self-bootstrapping LaTeX into `tools/` so corporate locked-down envs can still compile PDFs.

---

## Requirements

- **Python 3.11+** (Windows / macOS / Linux — no WSL needed).
- An LLM provider: a Copilot subscription via VSCode, OpenAI / Anthropic / Gemini API key, or local Ollama. Pick one in the quickstart below.
- *Optional:* `pandoc` + a LaTeX engine for `paper.pdf` output. If you can't install MiKTeX / TeX Live (no admin, corporate laptop), run `python launch.py --install-tectonic` after install — it drops a 70 MB self-contained LaTeX binary into `tools/` and FI picks it up automatically.
- *Optional:* `pip install axon` for the knowledge layer (literature retrieval + cross-quest memory). When present, FI auto-launches the Axon API as a sidecar on `127.0.0.1:8000` so the embedding model + vector indexes stay warm across quests instead of cold-loading per quest (saves ~5-15 s per `/start`). Skip the auto-launch with `--no-axon-sidecar` or `FI_NO_AXON_SIDECAR=1`.
- *Optional, network only:* the WorldBank + Wikipedia dataset adapters (`engine.dataset_adapters: [worldbank, wikipedia]`, opt-in) use stdlib `urllib` against `api.worldbank.org` and `en.wikipedia.org` — no extra `pip install` needed, but the runtime needs outbound HTTPS to those hosts when the adapters are enabled.

---

## What you'll have after one quest

Inside `outputs/<quest_id>/`:

```
paper/paper.md                        ← the finished IMRAD paper
paper/paper.pdf                       ← (if pandoc + LaTeX installed)
figures/*.png                         ← every plot the experiment produced
code/experiment.py                    ← the exact code that ran
slides.md                             ← if output.kinds: [slides]; Marp markdown source
slides.pptx                           ← real PowerPoint (if pandoc installed too)
slides.html / slides.pdf              ← if marp CLI installed
poster.tex, poster.pdf, talk.md       ← optional deliverables (poster.pdf if a LaTeX engine is installed)
data/auto_collected/<rank>_<slug>.md  ← Axon hits (no-simulation mode, written before the user-data pause)
data/auto_collected/<adapter>/<rank>_<slug>.md  ← dataset-adapter hits (worldbank, wikipedia) under per-adapter subdirs
.fi/run.log                           ← engine's structured node-by-node log
.fi/launch.log                        ← subprocess wrapper log (written when launched via --serve / @fi)
.fi/state.sqlite                      ← resumable checkpoint
.fi/requirements.lock.txt             ← pip freeze (only on successful finish; .venv/ is then removed to reclaim disk)
frontier_insight_summary.json        ← machine-readable index
quest_failed.md                       ← only if the quest crashed: failing node, exception, log tail, --resume command
                                        (the web dashboard surfaces this as a red banner on the quest detail page)
<kind>_skipped.md                     ← only if an output kind couldn't be produced: what was requested, why it skipped, how to fix
```

---

## 5-minute quickstart

You'll run a tiny example quest — three numerical integrators on a damped harmonic oscillator — and see a paper land in `outputs/`. Total wall time: ~3 minutes after setup.

### 1. Install

```bash
git clone https://github.com/jyunming/FrontierInsight
cd FrontierInsight
pip install -r requirements.txt
```

(See [Requirements](#requirements) above if you haven't checked Python version yet.)

### 2. Pick an LLM provider (one-time)

Choose **one** of these — whichever you already have access to:

#### Option A — GitHub Copilot via VSCode (recommended if you have Copilot)

This is the safest and cleanest path. You already have the VSCode Copilot Chat extension installed and signed in.

**First-time setup needs Node.js + npm** (~5 min total to build the .vsix). After that, CI auto-rebuilds the .vsix on every push to main when extension sources change — so if you're not modifying the extension yourself, this is a one-time cost.

```bash
# Build the FI VSCode extension into a .vsix (one-time):
cd vscode-frontier-insight
npm install
npm run package    # → vscode-frontier-insight/vscode-frontier-insight.vsix

# Install it. Either:
#   GUI:  Extensions sidebar → ⋯ → "Install from VSIX..." → pick the file
#   CLI:  code --install-extension vscode-frontier-insight/vscode-frontier-insight.vsix
```

In Copilot Chat, just type `@fi` — the extension walks you through the interview (topic, outputs, paper format, research approach, study depth, comparative baseline, success metric, budget, clarify mode, reviewer panel, Axon, multi-model ensemble preset, …) via input modals, generates the config.yaml, and runs the quest. The chat panel streams every step. `@fi /update <quest_id>` opens the same interview pre-filled with the editable subset for a mid-quest tweak. **See [`vscode-frontier-insight/README.md`](https://github.com/jyunming/FrontierInsight/blob/main/vscode-frontier-insight/README.md) for details.**

#### Option B — Headless CLI (no VSCode running)

If you want to run quests overnight or in CI, use one of the chat-style CLI providers. FI shells out to the local binary per call and reuses the CLI's own OAuth:

```bash
# Pick whichever you already have signed in:
claude login                                # → provider.name: claude_cli
codex login                                 # → provider.name: codex_cli
gemini   # one-time interactive sign-in     # → provider.name: gemini_cli
```

Then change `provider.name` in your YAML accordingly.

**Note:** the `copilot_cli` provider is also wired in the codebase but **does not work** as an FI backend — GitHub's standalone Copilot CLI is an *agentic* tool that interprets node prompts as user coding tasks and replies conversationally instead of running stateless LLM inference. FI emits a loud warning at engine init when you select it. For headless Copilot, there isn't currently a clean path; use `vscode_extension` (Option A) when you can, or switch to one of the other CLIs / Option C above.

#### Option C — OpenAI / Gemini API key (direct HTTP)

```bash
export OPENAI_API_KEY=sk-...        # or GEMINI_API_KEY
```

Then change `provider.name` to `openai` or `gemini`.

> **Why `claude_cli` is NOT in this list:** `claude_cli` is the
> Option-B path that shells out to the Claude Code CLI and reuses
> its OAuth — it does NOT read `ANTHROPIC_API_KEY`. For
> Anthropic API-key usage, use Option B (`claude login` + then set
> `provider.name: claude_cli`).

#### What each provider charges you in — read this before picking

FI talks to two billing models. The right pick depends on how
bursty your usage is:

| Provider family | Unit billed | Cost-efficient for |
|---|---|---|
| `vscode_extension` / `copilot_cli` (GitHub Copilot) | **Premium request** — 1 call = 1 unit regardless of token count | Bursty single quests with heavy prompts. A 50-call quest is 50 units whether each call was 200 tokens or 200K. The flat-rate dominates when prompts are large. |
| `openai` / `codex` / `codex_cli` (OpenAI / ChatGPT) | **Per-token** — prompt + completion priced separately | Long-running automations where you can keep prompts skinny. `gpt-4o-mini` is cheap enough that low-value nodes (clarify, cross_check) shouldn't burn budget. |
| `claude_cli` (Claude Code CLI) | **Per-token** | Same as above. Sonnet is the workhorse; reserve Opus for `write` + `review` via `provider.node_models`. |
| `gemini_cli` | **Per-token** | Cheapest cloud option for long-context tasks. |
| `ollama` (local) | **Free** | High-volume nodes where you don't need top-tier model quality, or airgapped runs. |

The web UI's `/quest/<id>` page shows per-quest cost for the
token-priced providers (the engine instruments every LLM call into
`<quest_root>/.fi/cost.jsonl`). Copilot's premium-request unit is
opaque to FI, so cost shows as $0.00 — track those via GitHub's
own usage page.

**Rule of thumb:** if you're running 1-2 quests/day, Copilot's
flat-rate per-call is usually cheapest. If you're running 10+
quests/day, switch to a token-priced provider and lean on
`node_models` to route cheap models at low-value nodes.

#### Option D — Local Ollama (free, no API key)

```bash
ollama serve
ollama pull qwen2.5-coder:32b
```

Then change `provider.name` to `ollama`. Slower but $0.

### 3. Run the example

```bash
python launch.py --config examples/integrator_bakeoff/config.yaml
```

You'll see log lines firing:
```
[FI] start quest_id=...-integrator-bakeoff-xxx
[ideate] topic=Compare three numerical integrators...
[literature] retrieved 5 docs
[design] iteration=0
[implement] generating experiment code
[execute] pip install ['numpy', 'matplotlib']
[execute] rc=0 duration=1.4s figures=2 result_json=True
[analyze] interpreting results
[write] wrote outputs/.../paper/paper.md
[review] verdict=accept score=4
```

When it's done, open `outputs/<quest_id>/paper/paper.md`. That's your paper. The figures it references are in the same folder.

---

## Headless interactive interview — `python launch.py --new`

If you're outside VSCode (CI, a remote shell, or just prefer the terminal), the CLI ships the same 14-question interview the VSCode `@fi /new` flow uses:

```bash
python launch.py --new                  # walks you through, then auto-starts the quest
python launch.py --new --draft-only     # writes the YAML to outputs/_drafts/<id>.yaml and stops
python launch.py --update <quest_id>    # mid-quest re-entry; pre-fills the editable fields
```

CLI-only questions: `provider.name` (openai / codex / claude_cli / codex_cli / copilot_cli / gemini_cli / ollama) and `provider.model` (curated list per provider with "Other (type your own)" escape hatch). VSCode skips both because the active Copilot model is captured automatically.

The interview makes ONE LLM call after the first 5 questions to suggest topic-tuned defaults for `comparative_baseline` / `success_metric` / `budget`; the answers land in `engine.clarify_overrides` so the engine's clarify node short-circuits in auto mode (no wasted call). Pick `--draft-only` when you want to hand-edit the YAML before launching (e.g. tune `engine.cross_check_per_finding_k`, add `knowledge.local_papers`).

---

## Write your own quest

Copy `examples/integrator_bakeoff/config.yaml` to `my_quest.yaml` and edit the `topic:` field. Keep everything else as-is until you want to tune.

Minimum viable config:

```yaml
topic: |
  <Your research question, 1-3 paragraphs.
  Be specific about what's known, what you want to test,
  and what success looks like.>

title: my-quest

provider:
  name: vscode_extension    # or copilot_cli, openai, ollama, claude_cli, ...
  model: gpt-5              # provider-specific

execution:
  sandbox: venv
  timeout_s: 600

knowledge:
  enabled: false            # set true after installing Axon

output:
  kinds: [paper_md]
  output_dir: ./outputs
```

Run it: `python launch.py --config my_quest.yaml`

---

## Common things you might want next

### Resume a crashed quest

If a Copilot HTTP/2 outage or any other transient failure crashes a quest mid-run, FI checkpoints every node to `outputs/<quest_id>/.fi/state.sqlite`. The engine also drops a copy of the source YAML at `outputs/<quest_id>/config.yaml` at startup, so resume is a one-step lookup — no slug match, no picker rummaging.

**From VSCode chat:**
```
@fi /resume
```
Shows a picker of all quests with a checkpoint, most recent first. Picking one reads `<quest_id>/config.yaml` directly and re-enters the LangGraph from the last completed node (legacy quests without that file fall back to a slug match in `outputs/_drafts/`; you'll be prompted only when both fail).

You can also pass the quest_id directly: `@fi /resume 1778650105-mammal-evolution-69ef80`.

**From the terminal:**
```bash
python launch.py --config outputs/<quest_id>/config.yaml \
                 --resume <quest_id>
```

The YAML's `provider` block is honored on resume; everything else (topic, design, literature, analysis) is loaded from the sqlite checkpoint, so you can edit the YAML between runs to change which model handles which node.

### Pull more (or fewer) prior-work citations

Each quest's literature step retrieves `knowledge.top_k` documents from Axon (+ the external router fallback when Axon is empty), and those are what end up in the paper's References. The default is **5**, picked so the prior-work block (5 × ~2000-char excerpt = ~2500 tokens) fits comfortably in the writer prompt alongside the design + analysis blocks. Override per-quest:

```yaml
knowledge:
  enabled: true
  top_k: 12        # pull more — costs ~5K extra tokens per LLM call that uses the block
```

Trade-offs: higher `top_k` gives the writer + cross-check nodes more sources to draw on but inflates every prompt that includes the literature block (`ideate`, `design`, `write`, `cross_check`, `review`), which means more tokens billed AND more chance of the model losing focus across a sprawling context. 5–10 is the sweet spot for a journal-length paper; bump to 15–20 only for a `comprehensive review` study depth.

### Keep internal docs out of an externally-published paper

By default, FI's writer treats every quest as externally-facing (journal submission, open-web release): the References section excludes cross-quest memory artifacts (`kind=fi_critique` / `fi_digest` / `fi_portfolio` / `fi_proposal` / `fi_summary` / `fi_source_catalog`) because an outside reader can't look them up. Real external literature (arxiv / openalex / etc.) and `fi_local_paper` entries you ingested yourself are kept.

If the paper is internal-facing — a project report, onboarding doc, memo for your own team — flip the flag so FI's own prior work can be cited:

```yaml
output:
  paper_format: report
  audience: internal   # default is "external"
```

### Watch progress in a browser instead of the terminal

```bash
python launch.py --serve --output-root ./outputs
```

Then open <http://127.0.0.1:8765/>. You'll see every quest, live log, paper preview, and figure browser. If a quest is paused waiting for clarifying questions, a panel appears for you to answer.

### Let the agent ask you 5 clarifying questions before it runs

Add to your YAML:
```yaml
engine:
  clarify_mode: interactive
```

Run with `--interactive`:
```bash
python launch.py --config my_quest.yaml --interactive
```

The agent prints 5 questions; press Enter to accept each suggested default, or type your own answer.

### Topics that need real data, not a Python script

Some research questions can't honestly be answered by a self-contained Python experiment — cultural comparisons, historical analyses, qualitative cross-case studies, policy reviews. For those, set:

```yaml
engine:
  no_simulation: true
  auto_collect_data: true              # default — try Axon first
  dataset_adapters: [worldbank, wikipedia]  # opt-in external lookups
```

The engine skips `implement → execute` entirely. Instead, `auto_collect_data` queries Axon for relevant docs, runs each enabled adapter (WorldBank pulls country-indicator tables, Wikipedia fetches article summaries), and writes everything into `outputs/<id>/data/auto_collected/`. If any files land, the quest continues through `analyze → cross_check → write → review`. If nothing lands (Axon empty + adapters returned nothing), the engine pauses cleanly with a `data/README.md` telling you what to drop. Drop your own files into `data/` and re-run:

```bash
python launch.py --config <yaml> --resume <quest_id>
```

The `simulatability` slot in the `clarify` agent's questionnaire decides this automatically when `clarify_mode: auto|interactive` — setting `no_simulation: true` in YAML is just the explicit override.

The companion `topic_shape` slot (`experimental` / `review` / `case_study` / `opinion`) classifies the topic's natural shape independently of simulatability. When the shape is non-experimental but the engine still resolves to SIMULATE, run.log carries a WARNING and the design stage shifts to a narrow illustrative experiment with weight on literature synthesis — rather than producing a toy benchmark pretending to answer a broad question. If you'd rather skip the experiment entirely for survey-shaped topics, pin `simulatability: "no"` in `clarify_overrides` (note the **quotes** — PyYAML parses unquoted `no` as boolean `False`; the engine now coerces that to `"no"` but explicit strings are still preferred). The interview pre-fills this for review-shaped topics.

### Make the PDF compile a hard gate

For unattended CI / fleet runs, make a missing pandoc/LaTeX engine fail fast instead of dropping a `paper_pdf_skipped.md` after a full quest:

```yaml
output:
  kinds: [paper_md, paper_pdf]
  require_pdf: true
```

The engine pre-flights `pandoc` + a LaTeX engine (pdflatex / tectonic / repo-local `tools/tectonic[.exe]`) before firing any LLM call. Missing tooling aborts the run with a recipe — saves ~15 min of LLM cost on a quest that was always going to fail at the compile step.

### Run many quests in parallel

```bash
python launch.py --fleet quests/a.yaml quests/b.yaml quests/c.yaml \
                 --max-concurrent 4
```

Each YAML is independent; the runner just bounds how many run at once.

### Use a different model per node

Cheap model for clarify/cross_check, strong model for write/review:

```yaml
provider:
  name: vscode_extension
  model: gpt-5                                 # global default
  node_models:
    clarify:     gpt-4o-mini
    cross_check: gpt-4o-mini
    write:       claude-3-5-sonnet
    review:      gpt-5
    # Per-persona overrides for the reviewer panel:
    review_panel.statistician:   claude-opus-4-7
    review_panel.devil_advocate: gpt-5
    review_panel.methodologist:  gpt-5
    review_moderator:            gpt-4o-mini    # cheap synthesis
```

### Have a panel of reviewers debate the paper

```yaml
engine:
  review_panel:
    - methodologist
    - statistician
    - devil_advocate
```

Each persona reviews independently; a moderator synthesizes. The panel costs ~4× the LLM calls of a single reviewer but catches different failure modes (design flaws, statistical issues, alternative explanations).

### Project-manager commands — when to reach for each

FI ships four commands that live *outside* the per-quest loop. They turn FI from a single-quest tool into a portfolio assistant. One LLM call each (cheap; cap'd content); each ingests its output into Axon so future quests can retrieve it.

| Command | Use it when |
|---|---|
| `@fi /proposal <topic>` | About to run a quest. Want to sanity-check the plan first before committing 8–18 LLM calls of a full run. |
| `@fi /critique <quest_id>` | Quest finished. Want a second opinion that didn't already write the paper — pick a different model family in the picker. |
| `@fi /digest [days]` | End of week. Want a 1-pager "what I shipped + what's stuck", with a structured diff vs last week. |
| `@fi /portfolio` | End of month, or scoping the next push. Want a synthesis of themes + gaps + meta-paper candidates across *every* quest. |

The next four sections walk through each in detail.

### Plan a quest before you commit compute

```
@fi /proposal Compare RK4 vs Verlet on the Kepler problem with eccentric orbits
```

Or:

```bash
python launch.py --proposal "Compare RK4 vs Verlet on the Kepler problem"
```

Inverts the usual flow. Instead of running a full quest and discovering the question was poorly scoped, get a 1-page LLM-written proposal *first* — background, hypothesis, plan, success criteria, risks, recommended next step. The user reviews the plan; if it looks good, runs the auto-generated companion YAML. Writes two files under `outputs/_drafts/`:

- `<id>-proposal.md` — the planning doc (read this, edit if needed).
- `<id>.yaml` — minimal config.yaml with the user's original topic. Run `python launch.py --config outputs/_drafts/<id>.yaml` to start the actual quest.

Ingested into Axon as `kind=fi_proposal` so future quests can retrieve "have I considered this before?"

Every PM-command (`--proposal`, `--critique`, `--digest`, `--portfolio`, `--summarize`, `--analyze`) accepts a `--<tool>-model <id>` companion to `--<tool>-provider` so the picker selection from `/tools/<tool>` actually routes the LLM call. For `vscode_extension` the id is the model id surfaced by the live Copilot catalog (e.g. `gemini-3-flash-preview`); for CLI providers it's the same identifier the CLI's own `--model` flag accepts. Omit the flag to keep the provider default (or — for `vscode_extension` — fall through to whatever the user has selected in their Copilot Chat model picker).

### Get an adversarial second opinion on a completed quest

```
@fi /critique <quest_id>
```

Or:

```bash
python launch.py --critique <quest_id> --critique-provider claude_cli
```

The in-quest review is biased — the same model wrote the paper AND reviewed it. `/critique` runs a fresh adversarial pass that has "never seen this paper before." For maximum effect, pick a different model family in your Copilot Chat picker (or pass `--critique-provider`) from the one that wrote the paper. Writes `outputs/<quest_id>/critique.md` with: a Verdict (accept / revise / reject / inconclusive), Methodology challenges (with quoted objections), Statistical issues (effect sizes, missing baselines, cherry-picked metrics), Reproducibility gaps, Alternative explanations the experiment doesn't rule out, an explicit "What the in-quest review missed" comparison, and Recommended follow-up experiments. Ingested into Axon as `kind=fi_critique`.

### See your whole research portfolio at a glance

```
@fi /portfolio
```

Or from the terminal:

```bash
python launch.py --portfolio
```

Walks every quest under `outputs/` (no time window) and produces `outputs/_portfolio/<YYYY-MM-DD>.md` with:

- **Topic clusters** — 2-6 thematic groupings; for each, the convergent finding (if any) and an open question the cluster surfaces.
- **Near-duplicate detection** — pairs of quests asking substantially the same question with different wording. Recommendation: merge / keep both / retire.
- **Meta-paper candidates** — places where ≥3 quests could be combined into a review/synthesis paper, with an effort estimate.
- **Coverage gaps** — specific research directions the existing themes *should* but *do not* cover, each grounded in a future-work section of a real quest.
- **Suggested next quests** — top 3 most-actionable topic strings ranked, each citing the prior quest it builds on.
- **Portfolio statistics** — total quests, completion split, time span, completion cadence (median days between completions), provider breakdown.

The portfolio also lands in Axon as `kind=fi_portfolio`. Use it monthly or when you're scoping the next research push.

### Get a weekly project-manager digest of your quests

```
@fi /digest          ← rolling 7-day digest
@fi /digest 14       ← last fortnight
@fi /digest 30       ← last month
```

Or from the terminal:

```bash
python launch.py --digest --days 7
```

FI walks `outputs/` for quests touched in the window, classifies each by LangGraph terminal-node state (read from `state.sqlite`), and asks the LLM to produce a markdown report under `outputs/_digests/<YYYY-Www>.md` with:

- **Completed this week** — 1-line synthesis per finished quest, grounded in the abstract.
- **In progress** — quests with checkpoints but no terminal `review`.
- **What changed since last digest** — a *structured* diff (✅ promoted from in-progress to complete, 🆕 newly started, ⚠️ still in progress, 🛑 stalled for 3+ digests, ❓ dropped). The diff is computed in code, not by the LLM, so the model can't hallucinate that you finished something.
- **Themes** — topic clusters spanning 2+ quests this week.
- **Suggested next quests** — concrete topic strings ready to paste into a new YAML, each grounded in a prior quest's future-work section.
- **Velocity** — quest counts and stall flags.

The digest also lands in Axon (kind `fi_digest`) so future quests can retrieve "what we were doing last week."

### Drop a paywalled PDF you downloaded yourself

```yaml
knowledge:
  local_papers:
    - ~/papers/that-paywalled-paper-you-grabbed-on-VPN.pdf
```

The agent will read it and cite it. Requires `pip install pypdf` for PDF support.

---

## Three example quests that already work

| Example | What it does | Wall time |
|---|---|---|
| [`examples/integrator_bakeoff/`](https://github.com/jyunming/FrontierInsight/tree/main/examples/integrator_bakeoff) | The quickstart. RK4 vs Velocity-Verlet vs forward Euler on a damped harmonic oscillator. | **~3 min** |
| [`examples/euv_mor_shot_noise/`](https://github.com/jyunming/FrontierInsight/tree/main/examples/euv_mor_shot_noise) | Theoretical LER floor in metal-oxide EUV resists. Uses the literature router. | ~15 min |
| [`examples/bernstein_vazirani_noise/`](https://github.com/jyunming/FrontierInsight/tree/main/examples/bernstein_vazirani_noise) | Bernstein-Vazirani algorithm under depolarizing noise — pure-numpy state-vector simulator validated against closed-form fidelity. | ~20 min |

---

## Going deeper

Read in this order — each builds on the previous:

1. **Pick your LLM provider strategically**: [`docs/PROVIDERS.md`](https://github.com/jyunming/FrontierInsight/blob/main/docs/PROVIDERS.md) — cost expectations, ToS standing per provider, when to use per-node model routing to spend cheap on `clarify` and expensive on `write`.
2. **Day-to-day reference**: [`docs/USAGE.md`](https://github.com/jyunming/FrontierInsight/blob/main/docs/USAGE.md) — every chat command, every `fi` flag, the YAML config schema, output artifact layout, common workflows.
3. **Full capability reference**: [`docs/capabilities.md`](https://github.com/jyunming/FrontierInsight/blob/main/docs/capabilities.md) — the 11-node DAG, every YAML field with defaults, knowledge layer internals, output kinds.
4. **Architecture & extension points**: [`docs/architecture.md`](https://github.com/jyunming/FrontierInsight/blob/main/docs/architecture.md) — `Config` / `QuestState` / `QuestArtifacts` contracts, the generator protocol, fleet runner internals.
5. **Hit a snag installing?** [`docs/INSTALL.md`](https://github.com/jyunming/FrontierInsight/blob/main/docs/INSTALL.md) — three install paths (standard / no-admin / locked-down) and the troubleshooting section.
6. **For contributors / AI coding assistants**: [`CONTRIBUTING.md`](https://github.com/jyunming/FrontierInsight/blob/main/CONTRIBUTING.md) for PR conventions; [`dev/CLAUDE.md`](https://github.com/jyunming/FrontierInsight/blob/main/dev/CLAUDE.md) for repo-specific guidance that Claude Code agents load automatically.

---

## License & contributing

Apache 2.0 — see [`LICENSE`](https://github.com/jyunming/FrontierInsight/blob/main/LICENSE). Contributions welcome via PR; see [`CONTRIBUTING.md`](https://github.com/jyunming/FrontierInsight/blob/main/CONTRIBUTING.md) if present, otherwise open an issue first.

---

## Sanctioned vs. unsanctioned Copilot paths — honest disclosure

The repo ships **three** Copilot integration points. They are NOT equivalent:

| Provider | ToS standing |
|---|---|
| `vscode_extension` | ✅ Sanctioned — uses VSCode's `vscode.lm.*` Language Model API |
| `copilot_cli` | ⚠️ Agentic CLI — replies conversationally to FI's prompts instead of running stateless inference. Loud warning at engine init. Not usable as an FI backend. |
| `github_copilot_cli`, `github_copilot_vscode` | ⚠️ Third-party reverse-engineered proxy, against Copilot's acceptable-use policy in spirit. The engine prints a warning when you select them. Use only at your own risk. |

For Copilot integration, only `vscode_extension` works today. For headless runs, use `claude_cli` / `codex_cli` / `gemini_cli` or HTTP-direct (`openai` / `gemini` / `ollama` / `vllm`).
