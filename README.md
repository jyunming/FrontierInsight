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
- **Checks that aren't just another LLM** — most guardrails in tools like this end in a language model reading text, which cannot catch a number copied wrong. FI adds two that are arithmetic: a **numeric oracle** comparing the paper's figures against what the run actually computed (digit transposition and rounding-unexplainable near-misses become blocking flags), and a **plausibility gate** enforcing the range and unit bounds the design declared — so a unit error or a sign flip goes back for repair instead of into a paper.
- **Keeps what it learns** — an optional **skill** lets a quest call tested simulation code instead of re-deriving the physics every run. FI ships none: skills are what it picks up working with you, on your machine, not a library that arrives with the install. A skill ships its own self-test and its own valid-range assertions, and becomes usable only when that test passes *and* you approve that exact content — editing it lapses the approval, so an unattended run can never adopt capability nobody signed off on. Each quest picks the skills that fit its topic rather than carrying the whole library, so the library can grow without the prompts growing with it; `--why-skills "<topic>"` shows you that choice before you run anything. You teach it one either way round: `--teach-skill --from <module>` drafts a skill from a library you already have installed, and `--import-skill` takes one written for another agent — the envelope is the [Agent Skills](https://agentskills.io/) layout, so those transfer unchanged. Neither shortcut skips the gate: an imported skill still needs a self-test and your signature before any quest can use it. Since a skill's instructions go into a prompt and its self-test gets executed, `--scan-skill` reviews one statically first — injection phrasing, hidden characters, network access, `eval` — without importing or running anything. It shows you the lines; it never decides for you, and it never tells you a skill is safe — though a high-severity finding does stop the approval until you say `--despite-findings`. Because FI is not tied to one field, the catalogue is layered: untagged skills are general and always offered, while ones tagged with a domain join only when the topic looks related. All of it works in all three interfaces — `@fi /skills` in VSCode, a `/skills` page in the web dashboard, and `--skills` on the command line.
- **Three research modes** — run a real Python experiment in a sandboxed venv; a **no-simulation** path that analyses data you supply (or that FI auto-collects from the web + dataset adapters) for market, policy, or archival topics; *or* a **survey** path — a descriptive history / overview synthesised purely from the literature, with no experiment and no dataset, for "history of X" / "evolution of X" topics.
- **Knows the literature** — academic **and** open-web research, with real citations exported as BibTeX / CSL-JSON, and figures derived from web-collected data.
- **Three interfaces, one engine** — CLI, web UI, and VSCode chat all drive the same pipeline; every feature works in all three.
- **Runs on locked-down machines** — no-admin LaTeX (`--install-tectonic`), *or* a **LaTeX-free HTML/Chromium PDF fallback** that needs only pandoc + a browser and matches the LaTeX look.
- **Survives flaky providers** — a single provider outage no longer forfeits a finished quest: an optional **provider fallback chain** with per-provider **circuit breakers**, retry classification that skips doomed 4xx / quota errors, fail-open review gates, VSCode-bridge reconnect, and fleet backpressure (`FI_MAX_CONCURRENT_LLM_CALLS`) keep long runs and fleets moving. Gates run at temperature 0 so routing is reproducible, and a `--resume` completes only the outputs still missing.
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
