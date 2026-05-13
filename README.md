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

#### Option B — GitHub Copilot via terminal (headless)

If you want to run quests without VSCode open:

```bash
# One-time: install GitHub's official Copilot CLI
gh extension install github/gh-copilot
gh auth login   # if you haven't already
```

Then change `provider.name` in your YAML to `copilot_cli`.

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

If a Copilot HTTP/2 outage or any other transient failure crashes a quest mid-run, FI checkpoints every node to `outputs/<quest_id>/.fi/state.sqlite`. You can pick up at the failed node without redoing the prior work:

**From VSCode chat:**
```
@fi /resume
```
Shows a picker of all quests with a checkpoint, most recent first. Pick one and it auto-finds the matching draft YAML and re-enters the LangGraph.

You can also pass the quest_id directly: `@fi /resume 1778650105-mammal-evolution-69ef80`.

**From the terminal:**
```bash
python launch.py --config outputs/_drafts/<your-quest>.yaml \
                 --resume 1778650105-mammal-evolution-69ef80
```

The YAML's `provider` block is honored on resume, but the original quest topic / design / literature come from the checkpoint — don't worry if you point at a different YAML, the state is correct as long as the quest_id matches.

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
| `vscode_extension` (Option A above) | ✅ Sanctioned — uses VSCode's `vscode.lm.*` Language Model API |
| `copilot_cli` (Option B above) | ✅ Sanctioned — GitHub's own CLI |
| `github_copilot_cli`, `github_copilot_vscode` | ⚠️ Third-party reverse-engineered proxy, against Copilot's acceptable-use policy in spirit. The engine prints a warning when you select them. Use only at your own risk. |

If you want Copilot integration, use one of the first two.
