# What FI can do

One page, in plain words. Every item links to where it is described in full: the long, exhaustive catalogue is [capabilities-reference.md](capabilities-reference.md) (search it when you need the exact setting), and every setting is in [USAGE.md](USAGE.md). New here? Start with [your first quest](first-quest.md).

## The research loop

FI runs a topic through: ideate, literature search, a written plan, experiment design, code, a run (it repairs its own code when the run fails), analysis, a cross-check against the papers, the paper, and a review that can send it back. The experiment is repeated over several random seeds. What you see while it runs — the CLI console, the web page's **Progress** panel, VS Code's chat — is one plain line per stage, not the internal log behind it (`.fi/run.log` in the quest folder keeps everything). [Details](capabilities-reference.md#engine--execution) · [the graph](architecture.md)

## Checking its own experiments

- The experiment's rules (the *protocol*) are written before the run and **frozen** before the first full run; changing them needs your approval.
- A case with a known answer (an *oracle*) must pass before the main run, judged by the engine, not by the script.
- The simulation writes what it actually ran, and FI compares it with the protocol.
- Suspicious numbers, impossible values and paper numbers that do not match the results are caught by code, not by a model.
- Each headline number names what it estimates, so the matching interval and test are used.
- Each quest ends with an **evidence level** that says what was and was not shown; `rigor_profile: research` makes the checks stop the quest instead of only reporting.

[How it works](rigor.md) · [the words](glossary.md)

## What happened, in order

Every quest keeps an ordered, tamper-evident diary of its steps, checks and routes, including what the model said about why (marked as its own claim). `fi --trace <quest id>`, or the **Trace** panel on the quest page. [trace.md](trace.md)

## You in the loop

FI stops for you at set points: to read and edit the plan, to let you download paywalled papers, to accept or reject the paper, to approve a protocol change or a skill. `NEXT_STEP.md` in the quest folder says what to do; `fi --resume <quest id>` continues. [Details](capabilities-reference.md#engine--execution) · [USAGE.md](USAGE.md)

## Literature and memory

Searches arXiv, OpenAlex, Crossref, Semantic Scholar, PubMed and more, plus the web; drops off-topic results; looks for the canonical works a keyword search misses; fetches legal full text; accepts papers you supply. An optional knowledge layer (Axon) remembers earlier quests and starts by itself. [Details](capabilities-reference.md#knowledge-layer)

## What you get

A paper (Markdown, plus PDF with LaTeX or a fallback), in scientific formats (generic, NeurIPS, ICLR, IEEE Access, Nature MI) or essay, report, policy brief and whitepaper; slides; a poster; a talk script; every figure; BibTeX and CSL-JSON citations; a claims table tying each claim to evidence, a citation, or "unsupported". Rendered PDFs are measured for layout problems (an optional model check looks at them too). [Details](capabilities-reference.md#outputs)

## Three ways to drive it

The command line (`fi`), a local web UI (`fi --serve`), and a VSCode chat extension (`@fi`). All three use the same setup questions (`fi --new`, `/interview`, `@fi /new`), can change a running quest (`--update`), and can resume it. [Details](capabilities-reference.md#interviews--frontends) · [VSCode extension](../vscode-frontier-insight/README.md)

## Models and cost

Bring your own model: an API key, a signed-in CLI, VSCode Copilot, or a local Ollama. FI never chooses the model for you; you can route single stages to other models, fan a stage out over several, and set fallbacks. Cost is tracked per quest. [Providers](PROVIDERS.md) · [Details](capabilities-reference.md#provider-matrix)

## Long and heavy jobs

Experiments can run in the same Python, in a per-quest virtual environment, or in a Docker sandbox with no network. A job longer than one run (a cluster, an HPC queue) is submitted and picked up again with `--watch`. Many quests can run at once (`--fleet`). [Details](capabilities-reference.md#engine--execution)

## Other command-line tools

`fi tools --help` lists them: summarise a folder, write a weekly digest, synthesise across quests, get an adversarial second review of a finished quest, plan a quest without running it (`fi tools proposal`), analyse data you already have (`fi tools analyze`), and load documents into the knowledge layer (`fi tools ingest`). [Details](capabilities-reference.md#cli-tools)

## Example quests

Ready-made configs are in [`examples/`](../examples/README.md); the bundled first run is `examples/integrator_bakeoff/config.yaml`.
