# Frontier Insight

<picture>
  <source srcset="web/static/favicon.svg" type="image/svg+xml"/>
  <img src="vscode-frontier-insight/images/icon.png" alt="Frontier Insight icon" width="96" align="left" style="margin-right: 16px;"/>
</picture>

**One research topic in → a finished paper out** — plus the experiment that produced it, the figures, and the citations. All auto-generated, all reproducible, all on your machine.

<br clear="left"/>

Frontier Insight (FI) is an automated research assistant. You give it a research question; it runs a multi-stage pipeline — **clarify → literature → design → experiment _(or gather data)_ → analyze → evidence gate → write → claim-check → review** — and hands back a paper with everything that produced it. The only thing it needs from outside is an LLM provider: an OpenAI / Anthropic / Gemini API key, your **GitHub Copilot** subscription, or local **Ollama**.

Same engine, **three ways to drive it**: the command line, a local **web UI**, or a **VSCode** chat extension (`@fi`).

---

## What you get

After one quest, `outputs/<quest_id>/` contains:

```
paper/paper.md            the finished paper (IMRAD or essay/report/brief/whitepaper)
paper/paper.pdf           typeset PDF — via LaTeX, or a LaTeX-free HTML fallback
paper/references.bib      sources as BibTeX + CSL-JSON (drop into Zotero / a LaTeX flow)
paper/CLAIMS.md           claim-grounding ledger: each claim → experiment / citation / unsupported
figures/*.png             every plot the experiment produced
code/experiment.py        the exact code that ran (re-runnable from .fi/requirements.lock.txt)
slides.* · poster.* · talk.md    optional deck, poster, and speech
data/literature/*         the web + academic sources the quest actually read
```

---

## Quickstart (~5 minutes)

```bash
git clone https://github.com/jyunming/FrontierInsight
cd FrontierInsight
pip install -r requirements.txt

# Configure one LLM provider (see below), then run the example quest:
python launch.py --config examples/integrator_bakeoff/config.yaml
```

That runs a tiny example (three numerical integrators on a damped oscillator, ~3 min) and writes a paper to `outputs/`. Open `outputs/<quest_id>/paper/paper.md` — that's it.

**Pick one LLM provider** — whichever you already have:

| You have… | Use | Set `provider.name` |
|---|---|---|
| GitHub Copilot | the VSCode extension (`@fi`) — sanctioned `vscode.lm` API | `vscode_extension` |
| An API key | OpenAI / Gemini over HTTP | `openai` / `gemini` |
| A signed-in CLI | `claude login` / `codex login` / `gemini` | `claude_cli` / `codex_cli` / `gemini_cli` |
| Nothing / offline | local Ollama (free) | `ollama` |

Full setup, cost trade-offs, and the billing model per provider are in **[docs/recipes.md](docs/recipes.md)** and **[docs/PROVIDERS.md](docs/PROVIDERS.md)**.

**Prefer to be walked through it?** `python launch.py --new` (CLI) or `@fi /new` (VSCode) runs an interview and builds the `config.yaml` for you.

---

## Highlights

- **Whole loop, not just the LLM call** — ideate → literature → code → run → analyze → write → review, end to end, without you driving each step. It even fixes its own code when the experiment crashes.
- **Built-in rigor** — an **evidence gate** weighs the assembled evidence before any paper is written; **claim grounding** traces every substantive claim to the experiment or a cited source; an optional **reviewer panel** (methodologist / statistician / devil's advocate) hard-flags fatal patterns.
- **Two research modes** — run a real Python experiment in a sandboxed venv, *or* a **no-simulation** path that analyses data you supply (or that FI auto-collects from the web + dataset adapters) — for market, policy, or archival topics with no code to run.
- **Knows the literature** — academic **and** open-web research, with real citations exported as BibTeX / CSL-JSON, and figures derived from web-collected data.
- **Three interfaces, one engine** — CLI, web UI, and VSCode chat all drive the same pipeline; every feature works in all three.
- **Runs on locked-down machines** — no-admin LaTeX (`--install-tectonic`), *or* a **LaTeX-free HTML/Chromium PDF fallback** that needs only pandoc + a browser and matches the LaTeX look.
- **Cross-quest memory** — `/digest`, `/portfolio`, `/critique`, `/proposal` accumulate over weeks via the optional Axon knowledge layer, so FI remembers what you tried last month.

---

## Requirements

- **Python 3.11+** — Windows / macOS / Linux, no WSL needed.
- **One LLM provider** — Copilot (VSCode), an OpenAI / Anthropic / Gemini key, a signed-in CLI, or local Ollama.
- *Optional:* `pandoc` for `paper.pdf`. For typeset PDFs add a LaTeX engine (MiKTeX / TeX Live, or the no-admin `--install-tectonic`); **with no LaTeX**, pandoc + any Chromium browser (Edge/Chrome) is enough — FI renders a Computer-Modern-styled PDF that matches the LaTeX look.
- *Optional:* `pip install axon` for the knowledge layer (literature retrieval + cross-quest memory).

---

## Going deeper

- **Recipes & detailed how-to** → [`docs/recipes.md`](docs/recipes.md) — provider setup, the interview, writing your own quest, the human-in-the-loop pauses, and ~30 task recipes.
- **YAML schema & every flag** → [`docs/USAGE.md`](docs/USAGE.md)
- **Full capability reference** (the 20-node DAG, every field) → [`docs/capabilities.md`](docs/capabilities.md)
- **Providers, cost & ToS standing** → [`docs/PROVIDERS.md`](docs/PROVIDERS.md)
- **Architecture & extension points** → [`docs/architecture.md`](docs/architecture.md)
- **Install troubleshooting** (standard / no-admin / locked-down) → [`docs/INSTALL.md`](docs/INSTALL.md)
- **Contributing** → [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Contributions welcome via PR.

**Copilot, honestly:** only `vscode_extension` is sanctioned — it uses VSCode's official `vscode.lm.*` Language Model API. The standalone Copilot CLI is agentic (it won't run as an FI backend), and any reverse-engineered Copilot proxy is against the acceptable-use policy in spirit; FI warns when you select those. For headless runs use `claude_cli` / `codex_cli` / `gemini_cli` or a direct API key. Details in [`docs/recipes.md`](docs/recipes.md).
