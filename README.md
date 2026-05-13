# Frontier Insight

**Give it a research topic. Get back a paper, the experiment that produced it, and the figures — all auto-generated, all reproducible.**

Frontier Insight (FI) is an automated research assistant. You write a one-paragraph topic into a YAML file; FI runs an 11-step research loop:

1. asks you 5 clarifying questions (optional),
2. brainstorms research directions,
3. searches the literature,
4. designs an experiment,
5. writes the experiment code,
6. runs it in a sandboxed venv (and fixes its own bugs if the script crashes),
7. cross-checks the results against published papers,
8. drafts an IMRAD paper,
9. reviews itself (optionally with a panel of reviewer personas),
10. iterates if the review says "revise",
11. saves the finished paper, figures, code, and a slides deck.

Everything runs locally on your machine. The only external dependency is an LLM provider (your choice: OpenAI / Anthropic / Gemini API keys, or your GitHub Copilot subscription, or local Ollama).

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
poster.tex, talk.md                   ← optional deliverables
.fi/run.log                           ← full run log
.fi/state.sqlite                      ← resumable checkpoint
frontier_insight_summary.json        ← machine-readable index
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

Needs Python 3.11+.

### 2. Pick an LLM provider (one-time)

Choose **one** of these — whichever you already have access to:

#### Option A — GitHub Copilot via VSCode (recommended if you have Copilot)

This is the safest and cleanest path. You already have the VSCode Copilot Chat extension installed and signed in.

```bash
# Build the FI VSCode extension into a .vsix (one-time):
cd vscode-frontier-insight
npm install
npm run package    # → vscode-frontier-insight/vscode-frontier-insight.vsix

# Install it. Either:
#   GUI:  Extensions sidebar → ⋯ → "Install from VSIX..." → pick the file
#   CLI:  code --install-extension vscode-frontier-insight/vscode-frontier-insight.vsix
```

In Copilot Chat, just type `@fi` — the extension walks you through 6 quick questions (topic, outputs, clarify mode, reviewer panel, …) via input modals, generates the config.yaml, and runs the quest. The chat panel streams every step. **See [`vscode-frontier-insight/README.md`](vscode-frontier-insight/README.md) for details.**

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

#### Option C — OpenAI / Anthropic / Gemini API

```bash
export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY
```

Then change `provider.name` to `openai` / `claude_cli` / `gemini`.

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

### Run many quests in parallel

```bash
python launch.py --fleet quests/a.yaml quests/b.yaml quests/c.yaml \
                 --max-concurrent 4
```

Each YAML is independent; the runner just bounds how many run at once.

### Use a different model per node (Phase O)

Cheap model for clarify/cross_check, strong model for write/review:

```yaml
provider:
  name: vscode_extension
  model: gpt-5                  # global default
  node_models:
    clarify: gpt-4o-mini
    cross_check: gpt-4o-mini
    write: claude-3-5-sonnet
    review: gpt-5
```

### Have a panel of reviewers debate the paper (Phase N)

```yaml
engine:
  review_panel:
    - methodologist
    - statistician
    - devil_advocate
```

Each persona reviews independently; a moderator synthesizes. The panel costs ~4× the LLM calls of a single reviewer but catches different failure modes (design flaws, statistical issues, alternative explanations).

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

| Example | What it does |
|---|---|
| [`examples/integrator_bakeoff/`](examples/integrator_bakeoff/config.yaml) | The quickstart. RK4 vs Velocity-Verlet vs forward Euler on a damped harmonic oscillator. ~3 minutes. |
| [`examples/euv_mor_shot_noise/`](examples/euv_mor_shot_noise/config.yaml) | Theoretical LER floor in metal-oxide EUV resists. Uses the literature router. ~15 minutes. |
| [`examples/bernstein_vazirani_noise/`](examples/bernstein_vazirani_noise/config.yaml) | Bernstein-Vazirani algorithm under depolarizing noise — pure-numpy state-vector simulator validated against closed-form fidelity. ~20 minutes. |

---

## Going deeper

- **Full capability inventory** with every YAML field, every provider, every loop: [`docs/capabilities.md`](docs/capabilities.md).
- **Architecture diagram and contracts** (`Config`, `QuestState`, `QuestArtifacts`, `Executor`, generator protocol): [`docs/architecture.md`](docs/architecture.md).
- **Phased development history**: [`docs/plan.md`](docs/plan.md).
- **For Claude Code agents working on this repo**: [`CLAUDE.md`](CLAUDE.md).

---

## License & contributing

Apache 2.0 — see [`LICENSE`](LICENSE). Contributions welcome via PR; see [`CONTRIBUTING.md`](CONTRIBUTING.md) if present, otherwise open an issue first.

---

## Sanctioned vs. unsanctioned Copilot paths — honest disclosure

The repo ships **three** Copilot integration points. They are NOT equivalent:

| Provider | ToS standing |
|---|---|
| `vscode_extension` | ✅ Sanctioned — uses VSCode's `vscode.lm.*` Language Model API |
| `copilot_cli` | ⚠️ Agentic CLI — replies conversationally to FI's prompts instead of running stateless inference. Loud warning at engine init. Not usable as an FI backend. |
| `github_copilot_cli`, `github_copilot_vscode` | ⚠️ Third-party reverse-engineered proxy, against Copilot's acceptable-use policy in spirit. The engine prints a warning when you select them. Use only at your own risk. |

For Copilot integration, only `vscode_extension` works today. For headless runs, use `claude_cli` / `codex_cli` / `gemini_cli` or HTTP-direct (`openai` / `gemini` / `ollama` / `vllm`).
