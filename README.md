# Frontier Insight

<picture>
  <source srcset="web/static/favicon.svg" type="image/svg+xml"/>
  <img src="vscode-frontier-insight/images/icon.png" alt="Frontier Insight icon" width="96" align="left" style="margin-right: 16px;"/>
</picture>

**One research topic in → a finished paper out** — plus the experiment that produced it, the figures, and the citations. All auto-generated, all reproducible, all on your machine.

<br clear="left"/>

Frontier Insight (FI) is an automated research assistant. You give it a research question; it reads the literature, designs and runs an experiment (or analyses data you supply), writes the paper, checks its own claims against the results and reviews the draft. It also tells you, at the end, **how much of the result was actually checked**, not only that a paper came out.

One engine, **three ways to drive it**: the command line, a local **web UI**, or a **VSCode** chat extension (`@fi`).

---

## Quickstart

You need **Python 3.11 or newer** and a language model FI can talk to (an API key, a signed-in CLI, or VSCode Copilot). Nothing else — FI sets up its own Python environment the first time it runs.

```bash
git clone https://github.com/jyunming/FrontierInsight
cd FrontierInsight
export OPENAI_API_KEY=sk-...       # Windows PowerShell: $env:OPENAI_API_KEY = "sk-..."
python launch.py --config examples/integrator_bakeoff/config.yaml
```

The first run asks once to install its own packages into `.venv/` (usually 30-90s), then continues straight into the quest; every run after that skips straight past it, and a later `git pull` that adds a new dependency is caught and installed the same way, automatically. Prefer to do it yourself first? `python -m venv .venv`, activate it, `pip install -e .` — FI sees that and does nothing further.

That runs the bundled example (three numerical integrators on a damped oscillator). It needs no data. It takes a while (it searches the literature, writes and runs code, then writes and reviews the paper) and prints each stage as it goes. When it ends, open **`outputs/<quest_id>/paper/paper.md`**. (Once set up, `fi` from an activated `.venv` is the same command as `python launch.py`.)

**Not using OpenAI?** Change `provider: name:` at the top of the example's YAML:

| You have… | Set `provider.name` to |
|---|---|
| An API key | `openai` / `gemini` |
| GitHub Copilot, or any model VSCode already has | `vscode_extension` (use the `@fi` chat in VSCode) |
| A signed-in CLI (`claude login`, `codex login`) | `claude_cli` / `codex_cli` |
| Nothing, offline | `ollama` (local, free) |

If a key is missing FI says which variable before it starts anything. If a run stops with `paused`, FI is asking for you: read `outputs/<quest_id>/NEXT_STEP.md`, then `python launch.py --resume <quest_id>`. The whole walk-through, step by step: **[docs/first-quest.md](docs/first-quest.md)**. Setup and cost of each provider: [docs/PROVIDERS.md](docs/PROVIDERS.md).

**Prefer to be asked?** `python launch.py --new` (CLI), the web form, or `@fi /new` (VSCode) asks the topic, what the result is for and (CLI and web) the model, works the rest out from the topic for you to review, and writes the config. **Web UI:** `python launch.py --serve`, then open http://127.0.0.1:8765. **VSCode:** install the `vscode-frontier-insight` extension and type `@fi /help`.

---

## What you get

After one quest, `outputs/<quest_id>/` contains:

```
paper/paper.md            the finished paper (abstract, keywords, sections)
paper.pdf                 typeset PDF (LaTeX, or a LaTeX-free HTML fallback)
paper/references.bib      cited papers as BibTeX + CSL-JSON (Zotero / LaTeX ready)
paper/CLAIMS.md           each claim traced to the experiment, a citation, or marked unsupported
figures/*.png             every plot the experiment produced
code/experiment.py        the exact code that ran (re-runnable)
plan.md                   what the literature says, the gap, and the design that ran
needs/EVIDENCE.json       how much of the result was checked (see below)
slides.* · poster.* · talk.md    optional deck, poster and speech
```

---

## Look at the plan before anything runs

Every quest writes **`plan.md`** before the experiment is designed. Ask FI to stop there so you can read it, edit it, or ask for a change:

```yaml
pauses:
  plan: ask      # or answer "Stop to read and edit the plan" in the interview
```

```bash
python launch.py --config quest.yaml                                   # searches, writes plan.md, stops
python launch.py --config outputs/<id>/config.yaml --resume <id> --revise-plan "use CD error as the metric"
python launch.py --config outputs/<id>/config.yaml --resume <id>       # run the plan as it now reads
```

On the web the quest page has a **Plan** panel; in VSCode use `@fi /plan <quest_id>`. You can also stop after the literature search (`pauses.supply: after_literature`) to read the sources and add your own papers before any compute is spent. To do a later step again (the code, the run, the analysis, the writing or the review), `--resume <id> --from` with no step lists the ones the quest reached, and `--from <step>` redoes from there: [docs/USAGE.md](docs/USAGE.md#doing-a-step-again).

## How much can you trust the result?

Each finished quest ends with an **evidence level**, from `executed` (it ran) up to `publication_ready` (every check passed and the review accepted it), and a plain sentence for what stands in the way of the next level. Nothing is called validated that was not checked, and a check that was turned off shows up as a gap. Set `rigor_profile: research` (the interview recommends it for a simulation study) to make the checks stop the quest instead of only reporting. How the checks work is in **[docs/rigor.md](docs/rigor.md)**; what a quest did, in order, is `python launch.py --trace <quest>` (**[docs/trace.md](docs/trace.md)**).

---

## Highlights

- **The whole loop** — ideate, literature, code, run, analyze, write, review, without you driving each step; it fixes its own code when the experiment fails.
- **Real literature** — academic and open-web sources, with citations exported as BibTeX / CSL-JSON.
- **Checks that are not just another LLM** — a number in the paper is compared with the number the run computed; a claim is traced to the experiment or marked unsupported.
- **A plan you control** — read, edit or re-ask before compute is spent; later changes to the plan need your approval.
- **Fixes what was flagged, not the whole paper** — a review that names passages gets those passages fixed.
- **Three research modes** — run a Python experiment, analyse data you supply, or write a literature survey.
- **Three interfaces, one engine** — CLI, web UI and VSCode drive the same pipeline; every feature works in all three.
- **Runs on locked-down machines** — no-admin LaTeX or a LaTeX-free PDF path; a provider fallback chain for flaky providers.
- **Remembers across quests** — an optional knowledge layer (Axon) keeps what you tried last month.

The long descriptions of each are in [docs/features.md](docs/features.md).

---

## Requirements

- **Python 3.11+**, on Windows, macOS or Linux.
- **One LLM provider** (see above).
- Optional: a LaTeX engine (or any Chromium browser) for typeset PDFs, and `pip install axon` for the knowledge layer. `python launch.py --doctor` shows what this machine has; the details are in [docs/INSTALL.md](docs/INSTALL.md).

---

## Where to go next

- **Your first quest, step by step** → [`docs/first-quest.md`](docs/first-quest.md)
- **The words FI uses** → [`docs/glossary.md`](docs/glossary.md)
- **How FI checks its own experiments** → [`docs/rigor.md`](docs/rigor.md)
- **What a quest did, in order, and why** → [`docs/trace.md`](docs/trace.md)
- **Every setting and flag** → [`docs/USAGE.md`](docs/USAGE.md) · **how-tos** → [`docs/recipes.md`](docs/recipes.md) · **providers and cost** → [`docs/PROVIDERS.md`](docs/PROVIDERS.md)
- **What FI can do, on one page** → [`docs/capabilities.md`](docs/capabilities.md)
- The full map of the docs → [`docs/README.md`](docs/README.md) · Contributing → [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## License

Apache 2.0, see [`LICENSE`](LICENSE). Contributions welcome via PR.

**Copilot, honestly:** only `vscode_extension` is sanctioned (VSCode's official `vscode.lm.*` API). The standalone Copilot CLI is agentic and will not run as an FI backend, and any reverse-engineered Copilot proxy is against the acceptable-use policy in spirit; FI warns when you select those. For headless runs use `claude_cli` / `codex_cli` or an API key.
