# Frontier Insight — VSCode extension

<img src="images/icon.png" alt="Frontier Insight icon" width="96" align="left" style="margin-right: 16px;"/>

Run autonomous research quests inside VSCode using your existing
GitHub Copilot subscription. **No third-party proxy, no scraped tokens
— every LLM call goes through VSCode's sanctioned `vscode.lm` Language
Model API.**

## What this is

The Frontier Insight Python engine drives an 11-node async research
DAG (`clarify → ideate → literature → design → implement → execute →
execute_reflect → analyze → cross_check → write → review`, with three
self-correction loops and an optional reviewer panel). This extension
is the **host** that runs the engine inside VSCode and serves its LLM
calls through your Copilot subscription.

When you type `@fi /start config.yaml` in Copilot Chat, the extension:

1. Binds a free localhost TCP port.
2. Spawns the FI Python engine with `--vscode-bridge-port <N>`.
3. For each LLM call the engine wants to make, fires a real
   `vscode.lm.selectChatModels` + `model.sendRequest` on your behalf.
4. Streams the response back, renders progress in the chat panel.

The engine runs to a finished paper, slide deck, figures, and a
machine-readable summary in `outputs/<quest_id>/`. You see every node
firing live in the chat panel.

## One-time setup

1. **Install GitHub Copilot Chat in VSCode** and sign in.
2. **Clone the FrontierInsight repo** locally and install its Python
   deps: `pip install -r requirements.txt`.
3. **Build the extension and produce a `.vsix`**:

   ```bash
   cd vscode-frontier-insight
   npm install
   npm run package        # produces vscode-frontier-insight.vsix
   ```

   Then install it in VSCode one of two ways:

   - **GUI**: Extensions sidebar → ⋯ menu → "Install from VSIX..." →
     pick `vscode-frontier-insight/vscode-frontier-insight.vsix`. **Then
     `Developer: Reload Window` from the command palette** (the
     extension host doesn't auto-reload on install).
   - **CLI**: `code --install-extension vscode-frontier-insight/vscode-frontier-insight.vsix --force`
     then reload VSCode.

   ⚠️ **If you've installed a previous version of this extension**:
   every time you rebuild the `.vsix`, you must re-run the "Install
   from VSIX..." step AND reload the VSCode window. VSCode caches the
   installed extension; rebuilding the `.vsix` on disk does NOT
   replace what VSCode is running. A symptom of running a stale build:
   the chat panel shows old error text (e.g. *"Check the run log
   under outputs/<quest_id>/.fi/run.log"* — the current build shows
   the actual stderr tail in a fenced block).

   Alternative — develop without packaging: open the
   `vscode-frontier-insight/` folder in a separate VSCode window and
   press **F5**. That launches an "Extension Development Host" window
   with the extension active. Every code edit + F5 picks up the
   latest source — no install/reload cycle needed.
4. **Open the FrontierInsight repo** as your workspace, OR set the
   `frontierInsight.repoPath` setting to the absolute path.

## Settings

| Setting | Default | What it does |
|---|---|---|
| `frontierInsight.pythonPath` | `"python"` | Interpreter for the FI engine. Use a venv path if you don't want FI cluttering your global packages. |
| `frontierInsight.repoPath` | `""` | Absolute path to FrontierInsight repo. Defaults to the open workspace root. |
| `frontierInsight.outputDir` | `"outputs"` | Where finished quests are written (relative to repoPath). |

## Usage

### Easiest — interactive setup (recommended for first-time users)

In the Copilot Chat panel, just type:

```
@fi
```

Or explicitly:

```
@fi /new
```

The extension walks you through 12 quick questions via VSCode-native input modals:

1. **Topic** — what do you want to study? (free text)
2. **Title** — short identifier (auto-suggested from the topic).
3. **Outputs** — paper only / paper + PDF / paper + slides / everything.
4. **Paper format** — generic / NeurIPS / ICLR / IEEE Access / Nature MI (scientific); essay / report / policy brief / whitepaper (prose). Maps to `output.paper_format`.
5. **Research approach** — computational (a Python script can produce the data) vs. observational (real-world data needed). Maps to `engine.no_simulation` — and matches the clarify agent's `simulatability` slot, so picking it here skips the auto-detect path.
6. **Study depth** — brief preprint / journal-length / comprehensive review. Drives paper word count and citation depth. Smart-defaulted off the chosen paper format.
7. **Comparative baseline** — what existing method / dataset to compare against (free text).
8. **Success metric** — what number changing in what direction = headline result (free text).
9. **Time / compute budget** — soft wall-clock cap (free text).
10. **Clarify mode** — just run it / agent self-clarifies / ask me 7 questions.
11. **Reviewer panel** — single reviewer / 3-persona / 4-persona panel.
12. **Knowledge layer** — disabled (default) / Axon (if you have it set up).

The active Copilot model is captured automatically into `provider.model` so the quest stays on a consistent LLM even if you change Copilot model later. Provider / model selection is NOT asked in VSCode — the extension always uses the bridge transport. `@fi /update <quest_id>` re-opens the interview pre-filled with the editable subset for a mid-quest tweak; the same 12-question schema is used for both new-quest setup and mid-quest update.

It writes a config.yaml to `outputs/_drafts/<timestamp>-<title>.yaml`, then immediately starts the quest. You can re-run that same config later with `@fi /start <that-path>`, or edit it and re-run.

Press **Esc** on any modal to cancel.

### Power-user — write your own YAML

```
@fi /start examples/integrator_bakeoff/config.yaml
```

In the chat panel you'll see progress messages:
```
🧪 Starting quest: examples/integrator_bakeoff/config.yaml
  [FI] start quest_id=...-integrator-bakeoff-xxx provider=vscode_extension
  → clarify
  [clarify] mode=off; skipping
  → ideate
  [ideate] topic=Compare numerical integrators...
  → literature
  [literature] retrieved 5 docs
  ...
  → review
  [review] verdict=accept score=4
✅ Quest finished. Paper: outputs/.../paper/paper.md
```

### Fleet (multiple quests in parallel)

```
@fi /fleet quests/a.yaml quests/b.yaml quests/c.yaml
```

All quests share the same VSCode bridge, but each carries its own
`provider.node_models` so they can use different Copilot models from
the same subscription. The chat panel multiplexes — each line is tagged
with the quest's node + iteration so you can follow along.

### Resume a crashed quest

If a quest dies mid-pipeline (Copilot HTTP/2 outage, a kernel panic,
your laptop suspends, etc.), the LangGraph checkpoint at
`outputs/<quest_id>/.fi/state.sqlite` still holds every node that
completed before the crash. Re-enter from the failed node with:

```
@fi /resume
```

With no argument, that shows a QuickPick of every quest under
`outputs/` that has a checkpoint, sorted most-recent first. Pick one
and the extension auto-finds the matching draft YAML by title slug
(falling back to a file picker if no match exists), then spawns FI
with `--resume <quest_id>` so LangGraph continues from the last
completed node instead of redoing `ideate`/`literature`/`design`
from scratch.

You can also pass the quest_id directly:

```
@fi /resume 1778650105-mammal-evolution-69ef80
```

## Topics that need real data (no simulation)

Some research questions can't honestly be answered by a Python
experiment — cultural comparisons, historical analyses, qualitative
cross-case studies, policy reviews. The `/new` interview's
**Research approach** question covers this: pick *"observational"*
and the generated YAML sets `engine.no_simulation: true`. The
engine then skips `implement → execute` entirely and routes through
`auto_collect_data → wait_for_data → data_load → analyze`.

Before pausing, `auto_collect_data` runs:

1. Axon retrieval against `topic + design.hypothesis`. Top hits
   land as Markdown files under
   `outputs/<id>/data/auto_collected/<rank>_<slug>.md` with YAML
   provenance front matter.
2. Each adapter in `engine.dataset_adapters` (e.g. `worldbank`,
   `wikipedia`) runs an external lookup and writes evidence into
   `outputs/<id>/data/auto_collected/<adapter>/`.

If any files land, the chat panel shows the count and the quest
continues — many no-simulation quests run end-to-end without
manual data drops. If Axon was empty AND every adapter returned
nothing, the chat panel shows a *"Quest paused for data — drop
files into `data/`"* line and the engine exits cleanly. Drop your
own files and re-run with `@fi /resume <quest_id>`.

## PDF strict mode

For unattended fleet runs, add to the YAML so a missing pandoc /
LaTeX engine fails fast at pre-flight instead of after a full quest:

```yaml
output:
  kinds: [paper_md, paper_pdf]
  require_pdf: true
```

The engine aborts at startup with a recipe for the missing tool
(`pandoc`, `pdflatex`, or `tectonic`), saving the LLM cost of a
quest that was always going to fail at the compile step. The
default (`require_pdf: false`) keeps the graceful skip — quest
completes, writes `paper.md`, drops a `paper_pdf_skipped.md`
diagnostic next to it. See [`docs/USAGE.md`](../docs/USAGE.md) — the
"strict-mode PDF enforcement" section under the `output.require_pdf`
schema entry.

## Pre-quest proposal

```
@fi /proposal Compare RK4 vs Verlet on the Kepler problem with eccentric orbits
```

Inverts the flow: get a 1-page LLM-written proposal *first* (TL;DR, background, hypothesis, plan, success criteria, risks, scope limits, recommended next step) before committing compute to a full quest. Writes `outputs/_drafts/<id>-proposal.md` (the planning doc) plus `outputs/_drafts/<id>.yaml` (a minimal config you can feed to `/start` once the plan looks right). Ingested as `kind=fi_proposal`.

## Analyze pre-staged data

```
@fi /analyze ./my_data Compare ridership trends across the three regions
```

The inverse of `/proposal`: when you already have the dataset and just want FI to write a paper analyzing it. Pass a directory path followed by a one-sentence analysis topic. The extension copies every file under the directory (recursive, symlinks skipped, common noise like `.DS_Store` / `__pycache__` filtered) into the new quest's `data/` directory, then spawns the engine in no-simulation mode. The graph routes `auto_collect_data` → `wait_for_data` → `data_load` → `analyze → cross_check → write → review` — no `ideate` / `literature` / `design` / `implement` / `execute`, since the user already supplied the data. Cost: ~6 premium requests.

Quote the path if it contains spaces: `@fi /analyze "C:/My Data" Find common failure modes`.

## Adversarial critique

```
@fi /critique <quest_id>
```

The in-quest review is biased — same model wrote the paper and reviewed it. `/critique` runs a fresh adversarial pass with "you have never seen this paper before" framing. For maximum effect, pick a different model family in your Copilot Chat picker from the one that wrote the paper — the produced `critique.md` records both providers so you can see post-hoc which was which. Lands at `outputs/<quest_id>/critique.md` with Verdict / Methodology challenges (quoted objections) / Statistical issues / Reproducibility gaps / Alternative explanations / "What the in-quest review missed" / Recommended follow-up experiments. Ingested as `kind=fi_critique`.

## Cross-quest portfolio synthesis

```
@fi /portfolio
```

Walks every quest under `outputs/` (no time window) and writes `outputs/_portfolio/<YYYY-MM-DD>.md`. Unlike `/digest`, this is the all-time view — run it monthly or when scoping the next research push. The LLM gets the full corpus plus deterministic stats (total / completed / cadence / provider breakdown) and produces topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, and prioritized next-quest suggestions. Ingested as `kind=fi_portfolio`.

## Weekly project-manager digest

```
@fi /digest          ← rolling 7-day digest
@fi /digest 14       ← last fortnight
@fi /digest 30       ← last month
```

The digest walks `outputs/` for quests touched in the window, reads each one's `state.sqlite` to decide whether it completed or is still in progress, and produces a markdown report under `outputs/_digests/<YYYY-Www>.md` with these sections:

- **Completed this week** — 1-line synthesis per finished quest.
- **In progress** — quests with checkpoints but no terminal review.
- **What changed since last digest** — a *structured* diff (✅ promoted, 🆕 new, ⚠️ still-in-progress, 🛑 stalled 3+ digests, ❓ dropped). Computed in code from the prior digest's markdown, not by the LLM — the model can't hallucinate that you finished something.
- **Themes** — topic clusters spanning multiple quests.
- **Suggested next quests** — concrete topic strings grounded in each quest's future-work section.
- **Velocity** — quest counts and stall flags.

Lands in Axon as `kind=fi_digest` so future quests can retrieve "what we were working on last week."

## Per-node model routing

Each quest YAML defines which Copilot model to use per engine node.
For example:

```yaml
provider:
  name: vscode_extension       # set automatically when launched from this extension
  node_models:
    clarify:       gpt-4o-mini       # cheap classification
    ideate:        claude-3-5-sonnet # broad thinker
    cross_check:   gpt-4o-mini
    write:         claude-3-5-sonnet # prose
    review_panel.statistician:   claude-3-5-sonnet
    review_panel.devil_advocate: gpt-4o
    review_moderator:            gpt-4o-mini
```

The extension passes each `model_hint` to `vscode.lm.selectChatModels`.
If the hint doesn't match any model your subscription has access to,
that one call fails with a clear error (`no Copilot model available for
hint '<name>'`) — VSCode handles the gate.

## Cost & rate-limit reality

Every LLM call counts against your **normal Copilot premium-request
budget** — same as if you typed each prompt manually into Copilot Chat.
Approximate per-quest burn:

- Bare quest (clarify=off, single reviewer): ~10 premium requests
- Full panel (3 personas + moderator) + clarify-auto: ~18 premium requests
- Worst case (panel + re_experiment + 2 revise iterations): ~30+
- No-simulation quest (skips `implement → execute → execute_reflect`): ~6 premium requests — the saving comes from cutting the implement/execute self-correction loop entirely.

On Copilot Pro (~300 premium requests/month) you can run ~15–30 quests
a month depending on configuration. On Business / Enterprise the
ceiling is much higher.

## Why this path

The Frontier Insight repo ships **three** Copilot integration paths:

| Provider | ToS standing | When to use |
|---|---|---|
| `vscode_extension` (this) | ✅ Sanctioned — `vscode.lm` API | Interactive use, you have VSCode open |
| `copilot_cli` | ⚠️ Agentic — replies conversationally to FI's prompts instead of running stateless inference; FI emits a loud warning at engine init. Use `claude_cli`, `codex_cli`, `gemini_cli`, or `openai` for headless runs instead. | n/a — broken as a chat backend |
| `github_copilot_cli` / `github_copilot_vscode` | ⚠️ Third-party proxy, against ToS spirit | Don't use these for anything you care about |

If you're inside VSCode anyway, **this is the right path**. The Python
engine doesn't touch Copilot's HTTP API at all — it asks VSCode (the
official client) to make each call for you. User consent is explicit,
calls show up in your Copilot usage dashboard, and there's no
abuse-detection risk from scraped tokens.

## Limitations

- **VSCode must stay open** for the duration of the quest. Closing
  VSCode kills the Python child cleanly via the chat-cancellation
  token, but you lose any uncheckpointed state. The SqliteSaver
  checkpoint lets a killed quest resume with the same `quest_id`.
- **Per-extension rate limits exist** on `vscode.lm`. VSCode docs say
  these "will be expanded as we learn more." Very long quests (e.g.
  full panel + cross_check + many literature hits) can theoretically
  hit them. For fleet runs that need to be robust, use a chat-style
  CLI provider (`claude_cli` / `codex_cli` / `gemini_cli`) or an
  HTTP-direct provider (`openai` / `gemini` with API keys).
  `copilot_cli` is NOT a good fallback — it's agentic and replies
  conversationally to FI's node prompts.
- **Model availability depends on your subscription.** Run
  `vscode.lm.selectChatModels({vendor: 'copilot'})` to see what you
  have access to; copy model names from there into your `node_models`.
- **`engine.dataset_adapters` make direct outbound HTTPS, not via
  `vscode.lm`.** Enabling `worldbank` / `wikipedia` issues stdlib
  `urllib` requests (wrapped in `asyncio.to_thread`) from the
  engine process to `api.worldbank.org` / `en.wikipedia.org` — no
  Copilot routing, no `vscode.lm` involvement. On a network that
  blocks those hosts (corporate proxy, air-gapped VM), leave
  `dataset_adapters` empty and rely on Axon + manual data drops.
