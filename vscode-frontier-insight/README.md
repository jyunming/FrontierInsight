# Frontier Insight — VSCode extension

<img src="images/icon.png" alt="Frontier Insight icon" width="96" align="left" style="margin-right: 16px;"/>

Run autonomous research quests inside VSCode using your existing
GitHub Copilot subscription. **No third-party proxy, no scraped tokens
— every LLM call goes through VSCode's sanctioned `vscode.lm` Language
Model API.**

## Start here

Five steps, then everything below is reference you can look up when you need it.

1. Install **GitHub Copilot Chat** in VSCode and sign in.
2. Clone the FrontierInsight repo and install it into the Python the extension will use: `python -m pip install -e .` (that interpreter is the setting `frontierInsight.pythonPath`).
3. Install this extension: build the `.vsix` and use *Install from VSIX...*, then reload the window (details in *One-time setup* below), or press **F5** in this folder to try it without installing.
4. Open your project folder in VSCode. If FrontierInsight is somewhere else, set `frontierInsight.repoPath` to it.
5. In Copilot Chat type **`@fi /new`** and answer the questions. When a quest stops for you (a `paused` message in the chat), read what it asks and continue with **`@fi /resume`**.

The finished paper is in `outputs/<quest id>/paper/paper.md` in your project folder. Words you do not know are in the [glossary](../docs/glossary.md); a first run outside VSCode is in [first-quest.md](../docs/first-quest.md).

## What this is

The Frontier Insight Python engine drives a 21-node async research
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
machine-readable summary in `outputs/<quest_id>/`. A scientific paper opens with
an abstract and 4–6 keywords; a report, brief, essay or whitepaper keeps its
keywords out of sight, and an accepted paper's keywords go into its Axon index
card. FI writes the paper's References itself: the papers the text cites,
numbered in the order it first cites them. The foundational papers and textbooks
the literature search adds (up to eight the model names, looked up by title or, when the
title matches nothing, by author and year, plus the works several retrieved papers cite) are put to the writer with a request to cite the ones that
bear on the paper, and the reviewer sees, as advice, the ones the paper leaves
out. A retrieved work that holds only its title or a short book blurb is marked
so in the writer's prior-work block (`[title only]`, `[short blurb only]`), with
the instruction to cite it only to say the work exists or for what its title
states; a sentence built only on a title-only record, saying more than the title does, is marked unsupported by the
claim check. The References are
also exported as `paper/references.bib` (BibTeX) and `paper/references.csl.json`
(CSL-JSON). The web pages it drew on are listed under Further reading, not
References, and exported as `paper/further_reading.bib` / `.csl.json`. A
`paper/CLAIMS.md` ledger records which of the paper's claims
trace to the experiment, to words quoted from a cited source (the quote must be
in the source, however that source's PDF rendered its mathematics or spaced its
words; a quote spanning a formula the stored text lost is not found), or are
unsupported (unsupported claims force a revise). The
check reads every finding the run recorded, whole, with the results branches
those findings name, so a claim is never called unsupported because the
evidence that proves it was cut short. Each
figure keeps a record of what it draws, and a caption that describes a line the
figure draws flat or not at all, or calls a line flat that it draws varying (found by the line's exact label, and measured on SIR figures only), goes back to the reviewer; the human review in
the chat lists those captions. A figure saved with its legend drawn over its own
lines, or its title over a panel title, is measured on the finished canvas and
sent back to the experiment for one redraw before the paper is written. A figure the design planned and the run drew
that a draft leaves out is put back into the paper as soon as the draft is
written, captioned from that record and placed beside the paragraph that
discusses it and never carried past the reference list to print among the
references, so no review round is spent restoring it; if one still goes
missing by review time the chat lists it among the must-fix items. The deck
ends on one References slide with the sources the paper cites most. With `engine.execute_replicates > 1` the results are reported
with 95% confidence intervals (a probability's kept within 0–1), effect sizes (including between methods nested
inside a parameter sweep), and a multiple-comparison guard instead of bare
numbers. A line figure, error bars included, is drawn as the mean of the seeds, shaded with its 95%
confidence interval, with its legend outside the axes when it has four or more entries or would sit over the lines; bar charts, histograms and scatters show replicate seed 0 only (every run that seed made), and the paper is told so. Every run gets its own seed, spaced far enough apart that no two runs draw the same
ones, so what varies between them is variation the experiment produced; a script that never names its seed is sent back once,
right after it is written, to read it (one extra model call, spent only then); and when the experiment still turns out not to read its seed
at all, the chat says so and the paper reports a single measurement instead of an interval over runs that were identical.
The same call goes out when the script reads the seed and builds a random generator without one anyway -- its runs differ, so nothing looks wrong, but the seed reached none
of the randomness and no run can be reproduced; the chat names the line, and says so in the log if the repair does not land. You see every node firing live in the chat panel.

After the outputs render, a visual check measures and screenshots each PDF;
with `output.visual_check_ai: true` it also asks the chat model to check the pages. The slides and poster are redone, at most twice,
when the check finds problems a new version can fix, and a paper whose last
page holds only a line or two is recompiled one line taller. The slides give each figure a slide of its own, and the check
measures each figure's tick labels on its slide: a deck whose ticks come out under 8 pt is one of the redos. Figures are drawn
with text 1.5 times the old house size, at most three panels to a figure (a prompt whose effect is not measured). With LibreOffice
installed, `slides.pptx` is exported to PDF and checked too; its formulas are native
PowerPoint equations, with a readable text form for other viewers. A VS Code build or chat
model that cannot take images checks from the measurements alone, and the chat
says so.

A topic that states a page limit ("≤ 4 pages", "at most 4 pages", "a 4-page
paper"), or a Page limit set in `@fi /new` under "Edit an advanced field"
(written to `output.page_limit`; leave it blank for no set limit), keeps the
paper to it. The paper
gets a tighter layout (2 cm margins where its template had 1 in, figures at most
a third of the text height, smaller References and Further reading lists), and
the writer gets a word budget. After each draft the review renders it the way
`paper.pdf` will be and counts its pages. A draft over the limit comes back as
the must-flag hit `over_page_limit`, with the page count, the limit and about
how many words to cut, and the paper is written again, shorter. There are at
most two of these rewrites, and they do not count against
`engine.max_iterations`; a draft still over after two is recorded in the review
instead. When the overflow is only FI's own Further reading list, the body is
not rewritten: the list's last entries are dropped, one at a time, until the
PDF fits (no model call). When the body is a few sentences over (about 400 words or fewer), the
model is asked which sentences of the background and discussion to take out instead of writing the
paper again; the engine decides what may go (never the Abstract, Methods or Results, a sentence
that says what the paper does or finds, a number stated nowhere else or a source's only citation),
and a paper it cannot bring within the limit goes to the rewrite as before. Without a limit the paper keeps its usual layout and
length.

## One-time setup

1. **Install GitHub Copilot Chat in VSCode** and sign in.
2. **Clone the FrontierInsight repo** locally and install its Python
   deps: `python -m pip install -e .`.
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
4. **Open the FrontierInsight repo** as your workspace, OR — the usual case — open **your own project
   folder** and set `frontierInsight.repoPath` to the FrontierInsight folder (the one containing
   `launch.py`). Quests then run in your project folder: your YAML, example files and the relative
   `outputs/` all mean that folder, and nothing is written into FI's checkout.

## Settings

- `frontierInsight.approveAs` — name pre-filled in the skill-approval box. A prefill only: the box still has to be typed into and confirmed, because the gate exists so a person decides rather than a setting deciding for them.

| Setting | Default | What it does |
|---|---|---|
| `frontierInsight.pythonPath` | `"python"` | Interpreter for the FI engine, and the one quest code runs on. Install FI's dependencies into this same interpreter (`<that python> -m pip install -e .`); a plain `pip` may belong to a different Python. `run.log` starts with `[env] python=<path>` so you can see which one ran. Use a venv path if you don't want FI cluttering your global packages. The CLI's own self-setup (`python launch.py` creating a `.venv/` on a missing dependency) is off for every command this extension runs, so a missing dependency here is always this setting to fix, not a different environment appearing on its own. |
| `frontierInsight.repoPath` | `""` | Absolute path to the FrontierInsight folder (the one containing `launch.py`). Leave empty when the open workspace IS that folder. |
| `frontierInsight.workingDir` | `""` | The folder quests run in. Defaults to the open workspace folder (FI's own folder when that is the workspace). Relative values are resolved against the workspace. |
| `frontierInsight.outputDir` | `"outputs"` | Where finished quests are written (relative to the working folder, or absolute). |
| `frontierInsight.axonStartupWaitSec` | `600` | How long to keep watching for the Axon sidecar after the editor starts, before saying none was found. Axon loads its embedding model and indexes before it serves, and you may start it well after opening VS Code — so the wait is long and silent. A sidecar that appears at any point during it produces no notification at all. `0` never shows the notice. |
| `frontierInsight.axonUrl` | `""` | Base URL of the Axon sidecar, e.g. `http://127.0.0.1:8420`. Empty means discover it automatically. Set it only when Axon runs somewhere discovery can't see — another machine, a container. Plain HTTP only. It pins the extension to that one instance rather than acting as a first guess, since two Axon instances hold different corpora. |

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

The extension asks a few questions in VSCode-native input modals:

1. **Topic** — what do you want to study? (free text)
2. **What is the result for?** — research (the default) or a decision gets every check a trusted result needs: the plan waits for you, every check can stop the quest, the experiment runs in a clean environment, and four reviewers read the paper. Exploring is a cheaper preliminary draft (no idea self-critique, no per-finding cross-check, no redesign after the analysis). The same answer writes the same settings as the CLI and the web form.
3. **Author line (optional, first time only)** — four boxes for author, affiliation, contact email and a project link. Press Enter on an empty box to skip it. The paper prints them under the title, the slides on the title slide, and the poster in its header, with the link as a QR code. They go only into your own output files and are kept, on this machine only, in `~/.frontier-insight/profile.json` (the same file the CLI and the web form use), so later interviews fill them in without asking.

Then it shows the defaults worked out from your topic, for review: the **paper format** (generic / NeurIPS / ICLR / IEEE Access / Nature MI for scientific work; essay / report / policy brief / whitepaper for prose; `output.paper_format`), the **deliverables** (paper + PDF by default; paper only, paper + slides, or everything), the **study depth** (brief preprint / journal-length / comprehensive review, which sets the word count and citation depth), the title (slugged from the topic), the **research approach** — computational (a Python script can produce the data), observational (real-world data needed), or **literature synthesis / survey** (a history / overview with no experiment and no dataset), mapping to `engine.no_simulation` (+ `engine.survey_mode`) and auto-suggested for "history of X" / "evolution of X" topics — the clarify mode, reviewer panel, knowledge layer, web research, paywalled-paper pause, the mid-quest pause for your own papers / datasets, paper audience, retrieval count and author line. **Edit a default** changes any of them (a changed paper format works the study depth out again; a changed author line is kept for later quests); **Edit an advanced field** sets the **multi-model ensemble** (off by default, or a fan-out profile with its stated cost multiplier, ~1.3× / ~2.0× / ~2.5×, and the models you tick from those Copilot Chat offers; FI picks none, and fewer than two runs single-model), the comparative baseline, success metric, time / compute budget, per-node models, web retrieval count, the paper style (LaTeX article or the Frontier Insight briefing look), the design-revise iteration budget, or the poster size (A1 portrait by default, A0 portrait, or 48 × 36 in landscape).

The active Copilot model is captured automatically into `provider.model` so the quest stays on a consistent LLM even if you change Copilot model later. The model list is not limited to Copilot: it is every chat model VSCode exposes to the extension, so a local Ollama server or a BYOK endpoint registered by another extension appears alongside Copilot's, tagged with its vendor. Provider / model selection is NOT asked in VSCode — the extension always uses the bridge transport. `@fi /update <quest_id>` re-opens the interview pre-filled with the editable subset for a mid-quest tweak; the same question schema is used for both new-quest setup and mid-quest update. It runs in an integrated terminal, which is handed the address of the extension's session-long bridge so the resumed quest's model calls still go through `vscode.lm.*` — keep this VSCode window open while it runs. The quest is both listed and resumed under `frontierInsight.outputDir`, so a custom output directory works rather than reporting the quest missing.

### Other chat commands

- `@fi /generate [<quest_id>] [<format>]` — produce one more output format (PDF / slides / poster / talk script) for an **already-finished** quest **without re-running the research**. Pick the quest + format, or pass them (`@fi /generate <id> paper_pdf`). Opens an integrated terminal running `launch.py --emit`; the quest's existing `paper.md` + figures are reused, so a markdown-only quest can get a PDF later for the cost of one render. The terminal is handed the address of the extension's session-long bridge, so the formats that call the model (slides / poster / talk script) work there too, not just the PDF — keep this VSCode window open while it runs. Mirrors the web quest page's **Outputs** panel.
- `@fi /ingest <paths>` — load PDFs / Markdown / TXT into the Axon corpus as prior work. Opens an integrated terminal.
- `@fi /install-tectonic` — install the tectonic LaTeX binary (~70 MB) into `tools/` so `paper.pdf` works without an admin install of MiKTeX. Opens an integrated terminal.
- `@fi /drafts` — list proposal-draft YAMLs in `outputs/_drafts/` with a one-click `/start` hint for each. Mirrors `python launch.py --list-drafts` and the web `/interview` drafts picker.
- `@fi /axon-status` — check whether the Axon sidecar (`python -m axon.api`) is reachable, and report which endpoint answered. CLI / `--serve` launches auto-start the sidecar so embeddings + indexes stay warm across quests; VSCode users keep their own (the extension probes on activate and offers a one-click "Start in terminal" if it's down — that prompt is non-blocking). Quests started from this chat use that same service for the knowledge base (`knowledge.axon_mode: http`, the default): they switch it to FI's `frontier-insight` project for one operation and back, and if it cannot be reached the quest runs without the knowledge base and says why in its log.
- `@fi /probe [all]` — ask the model selected in the Chat picker whether text FI did not send (a hidden system prompt, tool definitions, a persona) appears to be in its context. `@fi /probe all` does the same for every model VS Code lists, after a confirmation that says how many requests will be sent. Behavioural evidence only; see [Does a model carry a hidden system prompt?](#does-a-model-carry-a-hidden-system-prompt).
- `@fi /trace <quest_id> [--node <name>] [--detail summary|checks|debug]` — the quest's audit trace: what ran, why it went that way, and a check that the record was not edited. Prints exactly what `python launch.py --trace` prints (same file, same descriptions, same hash-chain check) and mirrors the web quest page's **Trace** panel; `--node` shows one step only, `--detail` narrows or widens what is shown (`checks` is the default). See [The trace: what a quest did, and why](../docs/trace.md).

### Skills

A **skill** is what FI has learned about driving one piece of software — when to use it, how to call it, and an executable check that proves it still works. FI ships none; a skill is what it picks up working with you, on this machine. A quest carries only the skills that fit its topic, and sends each where it is used: a writing skill to the writer, the rest to the experiment's design and code (the outline of the code reads a summary of each and only the step that writes the calls gets the full text, and the analysis of an experiment's results reads the titles of the sources rather than their text). A quest does not re-run a self-test that already passed for the same skill content, Python interpreter and installed packages; a change to any of them, or a failure anywhere, runs it again. The record is `~/.frontier-insight/skill_selftest_cache.json`, and deleting it is always safe. Skills another agent installed (`~/.claude/skills`, `~/.codex/skills`, ...) are read where they are once you approve them; with `execution.sandbox: docker`, each one a quest uses is mounted read-only into the container and the prompts name that path.

Two gates stand before any skill reaches a quest, and both are visible here:

- `@fi /skills [--config <quest.yaml>]` — the library, with each entry's status, domain tags and any scan findings. Self-tests are **not** run (each can take up to 120s, which would leave the panel silent); the output says so and names the command that does verify. `--config` also looks in the skill folders that quest's YAML names (`engine.skills_dirs`), as `python launch.py --config quest.yaml --skills` does, so a skill kept only there is listed; `/scan-skill`, `/approve-skill` and `/revoke-skill` take it too (a path with a space goes in quotes), and the "next" commands the panel prints carry it along.
- `@fi /scan-skill <name> [--config <quest.yaml>]` — static review before you approve: instruction-injection phrasing, invisible characters, network access, `eval`, environment reads. Nothing is imported or executed — the files are parsed. It never reports a skill as "safe", only how many findings came out of how many rules.
- `@fi /approve-skill <name> [--config <quest.yaml>]` — the human gate. Shows the review first, then asks who is approving. The name is prefilled from `frontierInsight.approveAs` but never submitted for you: typing it *is* the gate. A high-severity finding takes an extra confirmation. Approval binds to the skill's exact content, so editing it lapses the approval.
- `@fi /approve-all-skills` — approve every skill that passes its gates in one go, after optionally pip-installing what the quarantined ones are missing (a skill whose library is absent fails its self-test and cannot be approved, so the order is install → re-test → approve; if one package cannot be installed on this machine the others still are, and it is named). Still asks who is approving, still refuses a failing self-test, and still holds high-severity findings unless you confirm them.
- `@fi /approve-amendment <quest_id>` — approve the change to a quest's frozen protocol that it stopped to ask about (the protocol is frozen right before the first full run; a redesign that wants another one stops the quest). Shows what changes and why, asks who is approving, and records it; resuming the quest without approving keeps the frozen protocol. If the results had already been seen, the run is archived under `archive/run_<n>/` and the paper says the change was made after them.
- `@fi /revoke-skill <name> [--config <quest.yaml>]` — withdraw approval, returning the skill to proposed.
- `@fi /import-skill [path]` — adapt a skill written for another agent ([Agent Skills](https://agentskills.io/) layout). With no path it opens a picker, then asks for optional domain tags — an untagged skill is general and always offered to a quest, while a tagged one joins only when the topic looks related.
- `@fi /teach-skill <name> <module>` — draft a skill from a library you already have installed, reading its real signatures by introspection rather than generating them.

All of these run the same `launch.py` the CLI does and render the result; none of the gate's logic is reimplemented in the extension, so the three interfaces cannot drift apart.

  You don't have to tell the extension which port Axon is on. It finds the running server through the lock file Axon writes for its store, falling back to Axon's `config.yaml` and then the built-in defaults, and it lists everything it tried when nothing answers. Use `frontierInsight.axonUrl` only for an Axon that discovery can't see, such as one on another machine.

  On activation the extension keeps watching for the sidecar for `frontierInsight.axonStartupWaitSec` (10 minutes by default) before it says anything, because Axon needs time to load its model and open its indexes — and you may start it long after opening the editor. The wait is silent, so starting Axon at any point during it means you never see a notice.

**Air-gapped machines:** the knowledge layer downloads ~184 MB of embedding + reranker models from Hugging Face on first use. To run with no network, on a connected machine run `python launch.py --export-models <dir>`, copy `<dir>` to the offline machine, and set `FI_MODELS_DIR=<dir>` + `FI_OFFLINE=1` (or `knowledge.models_dir` / `knowledge.offline` in YAML). See `docs/INSTALL.md`.

### Does a model carry a hidden system prompt?

`vscode.lm` reports no token usage and no quota, so what a provider wraps around FI's request (a system prompt, tool definitions, a persona) cannot be measured from inside the extension. `@fi /probe` gathers the one kind of evidence the extension can get, which is behavioural. For each model it sends two small requests, each a single User message, as FI's bridge does: a **canary** (`Reply with exactly the single word PONG and nothing else.`, where anything but `PONG` is what wrapped instructions would look like, and also what a chatty default looks like) and a **preamble question** that asks whether anything outside the conversation was in the model's context and to quote its first 300 characters, or reply `NONE`. It also calls `model.countTokens` on the text of each, which counts only what FI sent.

`@fi /probe` probes the model selected in the Chat picker (2 requests). `@fi /probe all` probes every model VS Code lists, third-party providers such as an Ollama server included, one after another, after a confirmation that states how many models and requests (2 per model) it will send, because on a metered plan each request can count against a quota. The result is one table (model, vendor, family, version, max input tokens, canary exact?, preamble said `NONE`?, tokens we sent, milliseconds), then the quoted reply for each model whose preamble answer was not `NONE`, and the exact text of the two requests so a pasted result says what was asked. An error from one model (quota, consent, no capacity) is reported in its row and the run goes on. It needs no folder open and starts no Python.

What it cannot show is printed under the table: a model's description of its own context is not proof (it can deny or invent), no usage figures come back so the real overhead is not measured, the token counts cover only the text FI sent, and a provider's own token accounting, where one exists, is the way to measure overhead.

### Note on Copilot billing units

Copilot bills **per call** — one "premium request" per LLM call regardless of how many tokens the prompt + completion contain. This is meaningfully different from `openai` / `claude_cli` / `gemini_cli`, which bill **per token**. For a 50-call quest with 10K-token prompts, Copilot is 50 units (flat); a token-priced provider could be substantially cheaper or more expensive depending on the prompt size. The web UI's `/quest/<id>` page shows per-quest cost for token-priced providers; Copilot calls show as $0 because the engine can't see the premium-request count from inside FI.

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

### Review the result before it ships

By default the engine pauses after the LLM review so you can accept, reject, or refine the paper before it's final. The chat panel renders the verdict + must-flag hits + suggestions, then surfaces a QuickPick (Accept / Reject / Refine). A number in the paper whose digits are a result's in another order, or whose last digit is one off the result's correct rounding (2.12 for a stored 2.1259), appears on its own line as **advisory** — they are shown for you to judge and never force a rewrite, because a pattern match over prose misreads DOIs and scientific notation often enough that an automatic revise does more harm than a flagged number. One advisory line comes from the results alone rather than from the paper's prose: a reference value the run was supposed to vary — a deterministic limit, a theoretical prediction — that came out as exactly 0 at every setting of the sweep, which is what a root finder returning the trivial root looks like and is how a flat line of zeros ends up in a figure captioned as convergence to that limit. A number that matches *nothing* the run can account for — not a result, not an across-seed aggregate, not a count of what the results hold, not a value the design or the configuration set — is a different finding and appears among the must-flag hits rather than the advisory line, because it sends the paper back to be written again and never sends the experiment back to be re-run. Refine opens a second input box for one line of feedback that the next revise pass honours alongside every previous refinement ask. Another advisory line lists what the experiment does differently from the numbers the topic sets (a set of settings in braces, a count of runs, a number of figures), under "Topic coverage (advisory, not blocking)"; it never sends the experiment back.
The methodologist persona's must-flag rules (circular evaluation, single-point eval, weak baseline without re-run, pseudo-units) are non-bypassable: a flagged paper forces another revise pass even when `engine.review_loop: false` is set. When every hit is about the text (an unsupported claim, a caption that describes what its figure does not show, or a statistic described as something the run never computed), FI rewrites only the paper, with the review in hand, instead of running the experiment again. That last one is its own check, and it is the one that catches a paper whose numbers are all correct and whose sentences are not: a significance threshold printed as a p-value, "Cohen's *d* > 16 for all pairwise comparisons" when the run's own comparisons disagree, a spread across seeds labelled "exact binomial CI", a figure's y-axis limit printed as a bin count, or a probability of the disease dying out that is really the probability of an outbreak. Each finding is recomputed from the run's replicates rather than pattern-matched, so it appears only when it can say exactly which quantity the paper renamed — and because only the words are wrong, it never costs you an experiment re-run. When the review's own reasoning instead names something the run computed — a key of the results, a figure's underlying data, or the experiment file — FI writes and runs the experiment again, once per quest, with that reasoning in front of the model: a value that came out wrong cannot be fixed by rewriting the paper around it. And when every hit of a text-only rewrite names a passage (the unsupported claims the check listed, a caption, the sentence around a number or a statistic), FI does not write the paper again: the writer gets the earlier draft and only those passages and returns edits, FI applies each one only where its text occurs exactly once, and every other word, number, figure and citation of the paper stays as it was. A whole rewrite is still used for a paper over its page limit and when the edits cannot be applied, and `run.log` says which one a round took and why.

To skip the gate entirely, set `pauses.review: off` in the YAML.

Whenever a quest pauses for you — to confirm setup (`pauses.clarify: ask`), to let you download a paywalled paper it found (`pauses.papers`, on by default), to drop in your own papers/data (`pauses.supply`), or to review the result — it writes one `NEXT_STEP.md` and the chat says the quest is waiting for you and shows that card: why it stopped, what to decide, what FI recommends, the alternatives, what to do and the `@fi /resume <id>` command, with everything else waiting listed under it. When a quest finishes, the chat lists anything still worth a look (a check that warned, papers that could not be downloaded).

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

To do a quest again from one of its steps, add `--from` and the step: `code`, `run`, `analysis`, `writing` or `review`. `@fi /resume 1778650105-mammal-evolution-69ef80 --from writing` writes the paper again from the same results. What that step and the later ones made is first moved to `.fi/previous/<time>/`, so you can compare. To change the plan itself, use `@fi /plan <quest_id> <what to change>`.

A quest can also be run in two steps, the literature first and the experiment second: set `pauses.supply: after_literature` in its YAML, or answer *Pause after literature* in the `@fi /new` interview. The quest stops once the literature is saved (`data/literature/`, and `NEXT_STEP.md` says what to do); add papers to `inputs/papers/`, data to `inputs/data/` or your own simulation files to `inputs/examples/`, then `@fi /resume <quest_id>` runs skills, design and the experiment with that literature in hand, without searching again.

Every quest writes `plan.md` (what the literature says, the gap, the design the experiment will run) before the design. Set `pauses.plan: ask`, or answer *Stop to read and edit the plan* in the `@fi /new` interview (advanced), and the quest stops once it is written. The advanced question *Rigor profile* in the same interview (`research` is recommended for a simulation study) turns that on together with the two-script split, the checks that stop the quest instead of only reporting, and a review panel; it is chosen when the quest is created. `@fi /plan <quest_id>` opens the file beside the chat: edit it (the design block at the end is used exactly as written), or ask for a change with `@fi /plan <quest_id> <what to change>` and the plan is rewritten and reopened; repeat, then `@fi /resume <quest_id>` runs it.

When a check fails the quest stops with an *Action needed* message that names what is wrong and what to do; fix it and `@fi /resume <quest_id>`. It stops for:

- a script that sweeps other values than the plan's protocol fixed (after two repairs);
- an oracle the script's measurement does not agree with (the engine judges it, not the script);
- numeric warnings the repairs did not remove (a resume runs the script again if you changed it, and accepts the run if you did not);
- a simulation whose `run_manifest.json` differs from the protocol (fix `code/simulate.py`), or, with `execution.split_failure: block`, a reply without both scripts;
- a redesign that asks to change the frozen protocol: read the request, then `@fi /approve-amendment <quest_id>` (a bare resume keeps the protocol).

A finished quest ends with a plain sentence, `[FI] evidence: <what was shown>`, and what stands between it and the next level; the underlying six levels (`executed` through `publication_ready`) are in [docs/rigor.md](../docs/rigor.md); the record is `needs/EVIDENCE.json`. How the checks work is in [docs/rigor.md](../docs/rigor.md).

**Using a model that is not in the Copilot picker (Moonshot's Kimi, or any OpenAI-compatible endpoint).** Write a YAML whose `provider` names the endpoint and the environment variable that holds the key, and start it with `@fi /start <path>`: a YAML that names its own provider keeps it, so the quest does not go through Copilot. `examples/kimi_moonshot/config.yaml` is a ready one for Kimi K2.6. Put the key in a `.env` file in your workspace folder (the folder VSCode has open, which is where the extension runs FI from; FI reads it at start and the file is git-ignored) or in the environment VSCode was started from, never in the YAML. `provider.fixed_temperature` and `provider.extra_body` are for models that accept one temperature or need a request field (Kimi's `thinking`); the `@fi /new` interview does not ask for them, so write those two by hand.

A quest waiting on a background job (HPC / cluster, `execution.background_jobs`) pauses cleanly and can be woken automatically:

```
@fi /watch
@fi /watch 1778650105-mammal-evolution-69ef80
```

It re-runs the quest's experiment script on a timer, shows every check in the chat, and resumes the quest when the job is done (`launch.py --watch`).

A quest whose design is stochastic (or that sets `execution.split_analysis: true` in its YAML; the default `auto` decides from the design) keeps the simulation (`code/simulate.py`, raw files under `raw/seed<K>/`) apart from its analysis (`code/experiment.py`): the chat's run log says which script ran, an analysis that fails or that the review sends back is rewritten and run again against the raw files already on disk, and the simulation runs again only when its own script changed. `execution.raw_dir` moves the raw files (a big disk, an HPC scratch area).

A resume regenerates only the outputs actually missing on disk — a `paper.pdf` /
slides / poster / talk that already rendered is left untouched — so re-running to
finish an interrupted quest never re-invokes the LLM for work that's already done.

### Long-session reliability

The bridge and engine are built to survive flaky Copilot sessions across a long
`@fi /start` or `--serve` run. A dropped bridge socket **reconnects on the next
call** instead of wedging the session; an HTTP/2 stall surfaces as a retryable
`bridge stalled` error; and the review / evidence gates **fail open** so a
transient model blip can't discard a finished paper. Gate decisions run at
temperature 0, so the same corpus routes the same way every time. If you run
YAML quests on CLI providers, `provider.fallback: [codex_cli, gemini_cli]` adds a
cross-provider failover chain with per-provider circuit breakers.

## Topics that need real data (no simulation)

Some research questions can't honestly be answered by a Python
experiment — cultural comparisons, historical analyses, qualitative
cross-case studies, policy reviews, or factual questions about
companies / markets / current events. The `/new` interview's
**Research approach** question covers this: pick *"observational"*
and the generated YAML sets `engine.no_simulation: true`. The
engine then skips `implement → execute` entirely and routes through
`auto_collect_data → wait_for_data → data_load → web_plots → analyze`.

For a purely descriptive **history / overview** (no experiment *and*
no dataset — e.g. *"the evolution of X"*), pick *"literature synthesis
/ survey"* instead. The YAML sets `engine.survey_mode: true` and the
engine routes `design → web_figures → analyze → write`, skipping the
whole data path — the paper is a narrative synthesis of the cited
sources with license-clean illustrative images and no fabricated
metrics. Survey mode is auto-suggested for "history/evolution/overview
of X" topics and can also be turned on by the clarify step.

The `/new` interview also has a **Web research** question (on by
default): when on, the literature node searches the public web
(Brave / DuckDuckGo) and **downloads every retrieved source to
`outputs/<id>/data/literature/`** — for *every* quest, simulation and
observational alike, not just the no-simulation path. Turn it off to
rely on academic sources only.

Web research does **not** require Axon to be installed, but it does
require the **Knowledge layer** question to be *Enabled*: that setting
is a master switch, and turning it off disables Axon, academic search
(arXiv / OpenAlex / Crossref) **and** web search together — leaving the
quest with no literature at all.

When a source is fetched, FI works to get **real full text**, not a
two-sentence snippet: HTML is cleaned with `trafilatura`; reCAPTCHA
walls and paywall / abstract-only stubs are rejected; and for an
academic source (PMC, a DOI, a preprint, a major publisher) the clean
open-access full text is recovered directly via the PMC BioC API,
Europe PMC, the preprint server, Unpaywall, and (with a key) Semantic
Scholar / CORE. When a PMC article's mathematics would be lost — the
BioC text drops formulae, leaving "either 1 or ." where the paper
states one — FI reads that article from the Europe PMC XML instead, so
the formula reaches the writer and the claim checker as text. A page kept
behind a bot wall is retried in a headless browser when one is installed,
and those renders **take turns** — one browser at a time, because each one
drives its own browser process over a pipe and several at once can break it,
ending the quest outright instead of failing one source. The rest of the
batch still runs in parallel, so only pages that need a browser wait. The
full text is stored uncapped on disk, and each
node's prompt receives the passages most relevant to the question
(`knowledge.literature_excerpt_chars` / `passage_ranking`).

When a relevant paper is genuinely paywalled (SPIE / IEEE / Elsevier …)
and only its abstract is reachable, **Supply paywalled papers** (on by
default; turn it off in the interview for an unattended run) makes the
quest pause and write a ranked
`needs/WANTED_PAPERS.md` (download links + why each matters). After the
run, the chat panel surfaces that list with instructions to drop the
PDFs into `inputs/papers/` and `@fi /resume <quest_id>` — they're then
ingested as real full text. You can also pre-load a folder of papers via
`knowledge.local_papers` (a directory is scanned recursively).

Example files for the experiment itself (a simulation setup, an input deck, a config — any type) go in `execution.inputs`, or into `<quest>/inputs/examples/` while the quest is paused; the design and the code are written from them and the experiment finds the folder in `FI_INPUT_DIR`.

Open-access sources never trigger that pause. An arXiv / PMC / preprint
paper that came back abstract-only means FI's *download* failed — the
host is usually unreachable behind a proxy or firewall — not that the
paper costs money. Stopping to ask you to buy a free paper would be
nonsense, so those are logged as a warning and listed in a separate
"Open access — FI's download failed" section of `WANTED_PAPERS.md` as a
manual fallback (a browser often succeeds where the agent's HTTP client
is blocked). Fixing the network is the real fix.

Before pausing, `auto_collect_data` runs:

1. Knowledge retrieval against `topic + design.hypothesis` — Axon
   **plus a general web search** (Brave when `BRAVE_API_KEY` is set,
   else keyless DuckDuckGo), so non-academic topics get real web
   pages, not irrelevant papers. Top hits land as Markdown files under
   `outputs/<id>/data/auto_collected/<rank>_<slug>.md` with YAML
   provenance front matter (each web hit keeps its source URL). A
   relevance guard drops confident-but-off-topic results.
2. Each adapter in `engine.dataset_adapters` (e.g. `worldbank`,
   `wikipedia`) runs an external lookup and writes evidence into
   `outputs/<id>/data/auto_collected/<adapter>/`.

The `web_plots` step then turns any quantitative data the sources
contain into source-attributed matplotlib figures (drawn in the
Frontier Insight house style), so the paper / poster / slides aren't
text-only — and every web source is cited by its URL across all three.
With `knowledge.fetch_web_figures` (off by default) FI also embeds a
couple of **license-clean illustrative figures** — a real figure from a
cited CC-licensed arXiv paper plus Wikimedia Commons diagrams (CC /
public-domain only), each attributed to its source + license. Set a
free [Brave Search API key](https://brave.com/search/api/) via the
`BRAVE_API_KEY` env var for better relevance (optional; DuckDuckGo is
the keyless default). For the academic side, `OPENALEX_API_KEY` unlocks
OpenAlex's full daily budget — without it a machine gets about 100
searches a day, and arXiv is searched through OpenAlex too — and
`SEMANTIC_SCHOLAR_API_KEY` gets Semantic Scholar off its mostly
rate-limited shared pool. Both are free, and both also work as
`knowledge.openalex_api_key` / `knowledge.semantic_scholar_api_key` in
the quest YAML (an environment variable wins). CORE, OpenAIRE and DOAJ
need no key and are where humanities and social-science topics find
open-access books, theses and journals. Academic search keeps only
citable record types: papers for a quest with an experiment, and papers
plus books and book chapters for a quest without one. Each literature
pass searches three short keyword queries, one per facet of the topic,
worded for its kind (methods and measured quantities, or the names
scholars of the subject write under), and fetches full text once, for
the sources it keeps. Before that, one model call grades every
retrieved source 0–3 on whether the paper could cite it: papers need a
2, web pages are dropped only at 0, and your own papers are never
screened (`knowledge.literature_screen`, on by default).

If any files land, the chat panel shows the count and the quest
continues — many no-simulation quests run end-to-end without
manual data drops. If Axon was empty AND every adapter returned
nothing, the chat panel shows a *"Quest paused for data — drop
files into `data/`"* line and the engine exits cleanly. Drop your
own files and re-run with `@fi /resume <quest_id>`.

## PDF strict mode

For unattended fleet runs, add to the YAML so a missing LaTeX
engine fails fast at pre-flight instead of after a full quest.
(pandoc itself now ships with the install, so in practice the
LaTeX engine is the piece that goes missing):

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

Every output kind follows the same graceful-skip contract:
`slides_skipped.md`, `speech_skipped.md`, `poster_pdf_skipped.md`
each explain what was requested, why it couldn't be produced
(missing CLI, refused LLM response, render-tool error), and how to
fix it — so the user discovers the failure by opening the quest
folder instead of grepping `run.log`. A subsequent successful run
of the same kind removes the stale breadcrumb.

Literature sources get the same treatment. When OpenAlex, Crossref,
arXiv, a publisher page or any other source rate-limits, blocks or
times out during a quest, the chat shows one
`⚠️ source failures: arxiv 3 (http_429=3); …` line at the end of the
run, so a thin bibliography comes with its reason. The detail is in
`.fi/source_failures.json` and in `run.log` (`[source-failure]` lines,
credentials redacted).

arXiv requests from every quest on the machine — including quests
started from this chat — share one queue (one at a time, 3 s apart,
backing off 1 / 2 / 4 minutes on a rate limit) and a 24-hour response
cache, so running several quests at once no longer multiplies arXiv
traffic.

If a quest crashes mid-graph (a `_node_*` raises, or a pre-graph
stage fails), the engine writes `quest_failed.md` to the quest
root with the failing-node name, the exception text, a log tail,
provider context, and a `--resume` command. The chat emits a
single `❌ Quest failed: <reason>` line at the end of the run;
open the quest folder to read `quest_failed.md` for the full
context and the resume hint. This holds even when the failure does
not arrive as an ordinary Python error — some failures reach the
engine from outside Python's usual error hierarchy, and one of
those used to leave an empty quest folder with nothing in it to
read. Stopping the quest yourself writes nothing: cancelling it,
or shutting the machine down, is not a crash to report.

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

Each quest YAML defines which model to use per engine node. In `@fi /new`
you do not have to type it: under **Edit an advanced field → Per-node model
overrides** a menu lists the models this VSCode offers (Copilot's, an Ollama
server's, a bring-your-own-key endpoint's) and nothing is chosen for you, since
`vscode.lm` gives FI no price or tier. It has one entry, **Light nodes → one
cheaper model**, that sets the five nodes measured as safe on a cheaper model
(`cross_check`, `select_skills`, `literature_screen`, `slides`, `poster`), an
entry to set a single node (the untested ones are marked as such), one to clear
every override, and one to type `node:model` pairs directly. On one simulation
topic, three runs each, moving those five nodes to a cheaper model used about
15% fewer tokens and no drop in scores was seen; three runs cannot show a drop
under about ten points, and this was measured with codex models, not through
this extension. For example, the YAML it writes:

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

The extension passes each `model_hint` to `vscode.lm.selectChatModels`,
trying `{id: hint}` first and falling back to `{family: hint}`, Copilot
models first and then any other vendor VSCode lists. The id path honours
exact picks from the menu above and from the dashboard's live model
dropdown (e.g. `gemini-3-flash-preview`, whose family is the coarser
`gemini-3-flash`); the family path keeps legacy fuzzy hints
(`gpt-5`, `claude-opus-4-7`) working. If neither matches, the call uses the
model selected in your Chat picker and the chat prints a warning naming the
hint, so a quest keeps running on a model you chose.

## Reasoning effort

The interview's advanced list (`@fi /new` → review screen → advanced
fields) has a **Reasoning effort** field that writes
`provider.reasoning_effort`: `minimal`, `low`, `medium`, `high`, `xhigh`
or `max`; "Provider default" writes nothing. `vscode.lm` has no
reasoning-effort setting, so on the `vscode_extension` provider the level
is not sent and the quest log says so once — the Copilot model runs at its
own default. The level takes effect when the quest YAML names a provider
that has such a setting (`ollama`, `openai`, `gemini`, `vllm`, `codex_cli`,
`claude_cli` or `antigravity_cli`), for example a YAML you edit and start
with `@fi /start <path>`. `docs/PROVIDERS.md` ("Reasoning effort") lists
which levels each provider takes.

## CLI providers answer, they don't act

When a quest YAML names a signed-in CLI provider (for example one you start
with `@fi /start <path>`), FI's calls to it are answer-only, the same from
VSCode, the command line and the web UI:

- `codex_cli` and `claude_cli` run with web search, shell and code execution,
  MCP servers, skills, plugins, subagents and memory turned off, and save no
  session.
- Every CLI call starts in a new, empty temporary directory that is removed
  when the call ends, so the CLI never sees your workspace folder.
- `codex_cli` does not read `~/.codex/config.toml`: custom model providers,
  profiles and MCP servers defined there are not used. The model comes from
  `provider.model` and the effort from `provider.reasoning_effort`.
- `antigravity_cli`, `copilot_cli` and `gemini_cli` still have their tools on:
  `agy` has no option to turn them off and works in its own fixed workspace,
  `copilot_cli` could not be checked while its quota was used up, and the
  Gemini CLI no longer signs in individual Google accounts.

`vscode_extension` is not a CLI provider and is unaffected. The flags and the
measurements behind them are in `docs/PROVIDERS.md` ("Answer-only CLI calls").

## Cost & rate-limit reality

Every LLM call counts against your **normal Copilot premium-request
budget** — same as if you typed each prompt manually into Copilot Chat.
Measured per-quest burn, counted from the token log (`.fi/cost.jsonl`) of
17 complete runs of one simulation quest (a deterministic vs stochastic SIR
epidemic model) on gemma4 through Ollama, with the default engine settings
(`clarify_mode: off`, a single reviewer, `cross_check_per_finding_k: 3`),
`knowledge.source_routing: manual`, and slides and a poster:

- **23–28 premium requests per quest**: 21–26 for the research and the paper, plus one each for the slides and the poster. The spread comes from the model: 0–3 experiment repair calls, one cross-check call per key finding that found related literature (0–8), and a second write → claim check → review pass when the review asked for a rewrite (14 of the 17 runs).
- Four runs of the same quest on older engine versions, in which design through review ran twice, logged 27–33 requests, not counting slides and poster.
- `knowledge.source_routing: auto` (the default) adds one routing request per literature pass and one per cross-check lookup; set `manual` when requests are metered.
- A reviewer panel of N personas replaces the single review request with N + 1 per review round (the personas plus a moderator).
- No-simulation quests are not in that sample, so their count is not measured here.
- Outputs add one request each for the slide deck, poster and talk script. A slides or poster redo after the visual check adds one more. With `output.visual_check_ai: true`, each checked output (paper, slides, pptx, poster) and each redo's new check add one more as well. The quest's token log (`.fi/cost.jsonl`, charted on the web quest page) counts these calls too.

On Copilot Pro (~300 premium requests/month) that is about 10–13 quests
of this size a month. Those runs used the settings above; a quest answered
*research* or *a decision* (the default) also runs a four-person review panel
and the per-finding cross-check with its verification, so it takes more
requests (not measured here). Answer *exploring* to save requests: it turns
off the idea self-critique, the per-finding cross-check and the redesign
after the analysis. On Business / Enterprise the ceiling is much higher.

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
