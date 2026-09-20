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
paper/paper.md            the finished paper (abstract, keywords and IMRAD, or essay/report/brief/whitepaper)
paper.pdf                 typeset PDF — via LaTeX, or a LaTeX-free HTML fallback
paper/references.bib      cited papers as BibTeX + CSL-JSON (drop into Zotero / a LaTeX flow)
paper/further_reading.bib the web pages the paper drew on, listed apart from its References
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
| Any model your VSCode already has | the same extension — it lists every model VSCode exposes, not only Copilot's (an Ollama server, a BYOK endpoint) | `vscode_extension` |
| An API key | OpenAI / Gemini over HTTP | `openai` / `gemini` |
| A signed-in CLI | `claude login` / `codex login` | `claude_cli` / `codex_cli` |
| Nothing / offline | local Ollama (free) | `ollama` |

Full setup, cost trade-offs, and the billing model per provider are in **[docs/recipes.md](docs/recipes.md)** and **[docs/PROVIDERS.md](docs/PROVIDERS.md)**.

How hard the model reasons is `provider.reasoning_effort` (`minimal` … `max`). Unset, FI sends nothing and each provider keeps its own default — a local Ollama model then does not think at all. How each provider takes the level is in [PROVIDERS.md](docs/PROVIDERS.md#reasoning-effort).

FI's calls to a signed-in CLI are answer-only: `codex_cli` and `claude_cli` run with web search, shell and code execution, MCP servers, skills, plugins and memory turned off, and every CLI call starts in a new, empty temporary directory that is removed when the call ends. `codex_cli` does not read `~/.codex/config.toml`, so custom model providers and profiles defined there are not used — the model comes from `provider.model` and the effort from `provider.reasoning_effort`. `antigravity_cli`, `copilot_cli` and `gemini_cli` still have their tools on. Details in [PROVIDERS.md](docs/PROVIDERS.md#answer-only-cli-calls).

**Prefer to be walked through it?** `python launch.py --new` (CLI) or `@fi /new` (VSCode) runs an interview and builds the `config.yaml` for you. It ends with an optional author line (name, affiliation, email, project link) that the paper, slides and poster print.

---

## Highlights

- **Whole loop, not just the LLM call** — ideate → literature → code → run → analyze → write → review, end to end, without you driving each step. It even fixes its own code when the experiment crashes, and redraws a figure whose legend sits on its own lines.
- **Long simulations** — an experiment that runs as an HPC/cluster job (`execution.background_jobs`) is submitted by an idempotent driver script and the quest pauses cleanly; `--watch <quest>` (or the web button, or `@fi /watch`) re-checks it on a timer and resumes the quest when it is done, logging every check. Start from your own example files (`execution.inputs`), and require skills (`engine.skills_required`), including ones another agent installed (under the Docker sandbox each one the quest uses is mounted read-only into the container).
- **Built-in rigor** — an **evidence gate** weighs the assembled evidence before any paper is written; **claim grounding** traces every substantive claim to the experiment — every finding the run recorded reaches the check whole, along with the results branches those findings name — or to words it quotes from a cited source, and checks the quote is really there — however that source's PDF rendered its mathematics, so a formula quoted correctly is not called unsupported over a subscript (a draft whose citations could not be checked is not accepted); an optional **reviewer panel** (methodologist / statistician / devil's advocate) hard-flags fatal patterns.
- **Checks that aren't just another LLM** — most guardrails in tools like this end in a language model reading text, which cannot catch a number copied wrong. FI adds four that are arithmetic: a **numeric oracle** comparing the paper's figures against what the run actually computed (digit transposition and rounding-unexplainable near-misses are reported for a human to judge, not auto-enforced — a regex over prose will always have an opinion about a DOI, and a wrong rewrite costs more than a flagged number), a **statistics check** asking the question neither of those asks — not "does this number match a result?" but "is it the *quantity* the paper says it is?", which is how a correct number still ships a wrong sentence: an effect-size claim made "for all pairwise comparisons" is re-derived from every comparison the run supports, a p-value that is really the Bonferroni threshold `0.05/n` is named as one, an interval labelled "exact binomial" that is really the spread across the seeds is caught, a figure's y-axis limit printed as a bin count is traced back to the axis it came from, and a probability of the epidemic dying out that is really the probability of an outbreak is reported with its complement — every one of them a recomputation against the run's own replicates, so the check says nothing unless it can name the contradiction, and because the numbers are right and only the words are wrong it sends the paper back to be written again and never re-runs the experiment over a mislabel, a **number-provenance check** asking the question the other two skip — not "does this number match a result?" nor "is it the quantity it claims to be?" but *where did this number come from at all?*, because a paper's wrong figure is often not a near-miss of a right one but a value nothing in the run ever produced: every number the paper prints must trace to something the run can account for — a value the experiment computed, one of its across-seed aggregates (the mean and interval a paper actually reports, which no single seed holds), a count of what the results contain, a number the design or the configuration set, or arithmetic the paper writes out — and one that traces to none of them sends the paper back to be rewritten, with rounding, truncation, percentages and complements all accepted, and with numbers written too imprecisely to carry provenance ("3 seeds", "R0 = 1.5", "95% CI"), years, version numbers, random seeds and citation markers never asked to trace anywhere, so that across the stored benchmark papers it flags only ones whose numbers were independently judged wrong, while a run that recorded no results at all is left alone entirely, and a **plausibility gate** enforcing the range and unit bounds the design declared — so a unit error, a sign flip, a diverging value the script capped to look in range, or one quantity sitting exactly on a bound in several settings (the mark of a trivial answer) goes back for repair instead of into a paper — while a zero that is simply correct does not, because below a threshold where nothing can happen, nothing happening is the result: a sweep reported under two groupings counts each cell once, a zero below the critical point is recognised as the prediction rather than a fault, and zeros confined to one level of one coordinate are read as a regime — and neither does a result that lands exactly on a maximum the design itself worked out for the run, which is the prediction arriving where the design said it would: a number equal to its bound to within a part in a billion is *on* that bound rather than past it, and sitting on a bound counts as evidence of a trivial answer only for a bound something broken could return by accident — nothing, everything, or a constant the script writes down, like the endpoint of a root finder's bracket — unless the quantity never leaves that bound at any setting of the sweep, which is the trivial answer whatever the bound. The oracle carries one more signal that reads no prose at all: a reference value — a deterministic limit, a theoretical prediction — that comes out as exactly 0 at *every* point of the sweep meant to vary it is the signature of a root finder returning the trivial root on its bracket endpoint, which is how a flat line of zeros ends up in a figure captioned as convergence to that limit. Each figure also carries a record of what it actually draws, so a caption describing how a line changes when the figure draws it flat goes back to the reviewer, and a figure the design planned and the run drew that a draft leaves out is put back into the paper as soon as that draft is written — captioned from the same record and placed beside the paragraph that already discusses it, and never carried past the reference list to print among the references — so a finished paper never discusses a figure it does not show, no review round is spent restoring one, and nothing is lost when a draft drops a figure on the last round; the reviewer's own check for a missing figure stays in place behind it. With several seeds, a line figure is drawn again as their mean, shaded with its 95% confidence interval, and the numeric oracle accepts the mean and its interval. Those seeds are FI's to choose — every run gets one, the first included — and they are spaced far enough apart that no two runs can draw the same ones, because a script derives a seed per trial from the base it is handed and bases a few integers apart had three runs of one quest repeating most of each other's trials while the spread across them was printed as a confidence interval. A script that never names its seed is sent back once, right after it is written, to read it (one extra model call, spent only then); and when the experiment still turns out never to read its seed at all, so that every replicate was the same run over again, FI reads that off the script's own source and reports a single measurement rather than an interval over copies.
- **Keeps what it learns** — an optional **skill** lets a quest call tested simulation code instead of re-deriving the physics every run. FI ships none: skills are what it picks up working with you, on your machine, not a library that arrives with the install. A skill ships its own self-test and its own valid-range assertions, and becomes usable only when that test passes *and* you approve that exact content — editing it lapses the approval, so an unattended run can never adopt capability nobody signed off on. A self-test that passed is not re-run by later quests until the skill, the Python interpreter or an installed package changes (the record is `~/.frontier-insight/skill_selftest_cache.json`; deleting it forces every test to run once more), a failing one runs every time, and the tests that do run, run in parallel without stalling the web dashboard. Each quest picks the skills that fit its topic rather than carrying the whole library, and sends each only where it is used — a writing skill to the writer, the rest to the experiment — so the library can grow without the prompts growing with it; `--why-skills "<topic>"` shows you that choice before you run anything. You teach it one either way round: `--teach-skill --from <module>` drafts a skill from a library you already have installed, and `--import-skill` takes one written for another agent — the envelope is the [Agent Skills](https://agentskills.io/) layout, so those transfer unchanged. Neither shortcut skips the gate: an imported skill still needs a self-test and your signature before any quest can use it. Since a skill's instructions go into a prompt and its self-test gets executed, `--scan-skill` reviews one statically first — injection phrasing, hidden characters, network access, `eval` — without importing or running anything. It shows you the lines; it never decides for you, and it never tells you a skill is safe — though a high-severity finding does stop the approval until you say `--despite-findings`. Because FI is not tied to one field, the catalogue is layered: untagged skills are general and always offered, while ones tagged with a domain join only when the topic looks related. All of it works in all three interfaces — `@fi /skills` in VSCode, a `/skills` page in the web dashboard, and `--skills` on the command line. Starting from an empty library? `scripts/import_scientist_skills.py` sources a curated set of 69 scientist-workflow skills, from 11 real upstream repositories, to import — see [docs/recipes.md](docs/recipes.md#bootstrap-a-starter-set-of-scientist-skills), which also covers moving an already-approved library to another machine.
- **Three research modes** — run a real Python experiment on FI's own Python (or in a per-quest venv or Docker if you opt in); a **no-simulation** path that analyses data you supply (or that FI auto-collects from the web + dataset adapters) for market, policy, or archival topics; *or* a **survey** path — a descriptive history / overview synthesised purely from the literature, with no experiment and no dataset, for "history of X" / "evolution of X" topics.
- **Knows the literature** — academic **and** open-web research, with real citations exported as BibTeX / CSL-JSON, and figures derived from web-collected data. Academic search keeps citable records only (papers, plus books and chapters for topics without an experiment) and includes keyless open-access sources for the humanities and social sciences. Each literature pass searches three keyword facets of the topic, adds the foundational papers and textbooks a keyword search misses (up to eight named by the model, then looked up by title, plus the works the retrieved papers cite most), and screens every source for citability; the writer is asked to cite the foundational works that bear on the paper (the original paper for a method it uses, the standard textbook for the field), and the reviewer is shown, as advice, the ones it leaves out; FI writes the source lists itself: the papers the text cites become the numbered References, in the order the text first cites them, and the web pages are listed under Further reading. A page a site keeps behind a bot wall is retried in a headless browser, and those renders take turns — one at a time, because each drives its own browser process over a pipe and several at once can break it, ending the run outright instead of costing one source. The rest of the fetching stays parallel.
- **Keeps to a page limit** — a topic that says "≤ 4 pages" (or a Page limit typed among the advanced answers of the `--new` / `--update` interview, the web form or `@fi /new`, written as `output.page_limit: 4`) gets a tighter paper layout (2 cm margins and smaller reference lists) and a word budget for the writer. Each draft is rendered and its pages counted while the quest runs, and a draft over the limit is written again, shorter, with the page count and about how many words to cut: at most twice, without using the iteration budget. A paper without a limit is laid out and written as before.- **Three interfaces, one engine** — CLI, web UI, and VSCode chat all drive the same pipeline; every feature works in all three.
- **Runs on locked-down machines** — no-admin LaTeX (`--install-tectonic`), *or* a **LaTeX-free HTML/Chromium PDF fallback** that needs only pandoc + a browser and matches the LaTeX look.
- **Survives flaky providers** — a single provider outage no longer forfeits a finished quest: an optional **provider fallback chain** with per-provider **circuit breakers**, retry classification that skips doomed 4xx / quota errors, fail-open review gates, VSCode-bridge reconnect, and fleet backpressure (`FI_MAX_CONCURRENT_LLM_CALLS`) keep long runs and fleets moving. Gates run at temperature 0 so routing is reproducible, and a `--resume` completes only the outputs still missing.
- **Cross-quest memory** — `/digest`, `/portfolio`, `/critique`, `/proposal` accumulate over weeks via the optional Axon knowledge layer, so FI remembers what you tried last month; each accepted paper is indexed with the keywords its writer picked.

---

## Requirements

- **Python 3.11+** — Windows / macOS / Linux, no WSL needed.
- **One LLM provider** — Copilot (VSCode), an OpenAI / Anthropic / Gemini key, a signed-in CLI, or local Ollama.
- **Nothing else for `slides.pptx`** — the deck is rendered in-process (its formulas as native PowerPoint equations), and `pandoc` now installs as a wheel alongside FI, so `paper.pdf` needs no system package either.
- *Optional:* a LaTeX engine (MiKTeX / TeX Live, or the no-admin `--install-tectonic`) for typeset PDFs; **with no LaTeX**, any Chromium browser (Edge/Chrome) is enough — FI renders a Computer-Modern-styled PDF that matches the LaTeX look. `--install-marp` adds `slides.html` / `slides.pdf` without npm. LibreOffice lets the visual check screenshot `slides.pptx` too. Run `--doctor` to see what this machine has. To see what a quest held on a machine where files cannot be copied off it, `--dump-state <quest dir>` prints `.fi/state.sqlite` as text (every state key with a preview, and the path the quest took node by node).
- *Optional:* `pip install axon` for the knowledge layer (literature retrieval + cross-quest memory).

---

## Going deeper

- **Recipes & detailed how-to** → [`docs/recipes.md`](docs/recipes.md) — provider setup, the interview, writing your own quest, the human-in-the-loop pauses, and ~30 task recipes.
- **YAML schema & every flag** → [`docs/USAGE.md`](docs/USAGE.md)
- **Full capability reference** (the 21-node graph, every field) → [`docs/capabilities.md`](docs/capabilities.md)
- **Providers, cost & ToS standing** → [`docs/PROVIDERS.md`](docs/PROVIDERS.md)
- **Architecture & extension points** → [`docs/architecture.md`](docs/architecture.md)
- **Install troubleshooting** (standard / no-admin / locked-down) → [`docs/INSTALL.md`](docs/INSTALL.md)
- **Contributing** → [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Contributions welcome via PR.

**Copilot, honestly:** only `vscode_extension` is sanctioned — it uses VSCode's official `vscode.lm.*` Language Model API. The standalone Copilot CLI is agentic (it won't run as an FI backend), and any reverse-engineered Copilot proxy is against the acceptable-use policy in spirit; FI warns when you select those. For headless runs use `claude_cli` / `codex_cli` or a direct API key. Details in [`docs/recipes.md`](docs/recipes.md).
