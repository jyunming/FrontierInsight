# Frontier Insight — VSCode extension

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

The extension walks you through 6 quick questions via VSCode-native input modals:

1. **Topic** — what do you want to study? (free text)
2. **Title** — short identifier (auto-suggested from the topic).
3. **Outputs** — paper only / paper + PDF / paper + slides / everything.
4. **Clarify mode** — just run it / agent self-clarifies / ask me 5 questions.
5. **Reviewer panel** — single reviewer / 3-persona / 4-persona panel.
6. **Knowledge layer** — disabled (default) / Axon (if you have it set up).

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
  token, but you lose any uncheckpointed state. (Phase B's
  SqliteSaver lets a killed quest resume with the same `quest_id`.)
- **Per-extension rate limits exist** on `vscode.lm`. VSCode docs say
  these "will be expanded as we learn more." Very long quests (e.g.
  full panel + cross_check + many literature hits) can theoretically
  hit them. Use `copilot_cli` for fleet runs that need to be robust.
- **Model availability depends on your subscription.** Run
  `vscode.lm.selectChatModels({vendor: 'copilot'})` to see what you
  have access to; copy model names from there into your `node_models`.
