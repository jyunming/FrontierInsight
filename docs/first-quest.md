# Your first quest

A **quest** is one run: you give FI a research question, it gives you back a folder with a paper, the code that produced it and the checks it made. This page takes you from nothing to that folder.

## 1. What you need

- **Python 3.11 or newer** (`python --version`).
- **A language model FI can talk to.** The quickest is an OpenAI API key. Others: Gemini, a signed-in `claude` or `codex` command, GitHub Copilot inside VSCode, or a local Ollama. The table is in the [README](../README.md#quickstart).

## 2. Install

```bash
git clone https://github.com/jyunming/FrontierInsight
cd FrontierInsight
```

That's it — nothing to `pip install` yet. The first time you run FI (step 4), it asks once to set up its own `.venv/` (usually 30-90s) and continues straight into the quest; every run after that skips straight past it. This page writes every command as `python launch.py`, which always works, whether or not you've activated a venv. Prefer to install it yourself first? `python -m venv .venv`, activate it (`.venv\Scripts\activate`, macOS/Linux `source .venv/bin/activate`), `pip install -e .` — that also gives you `fi` as a shorter name for the same command, once that venv is activated. Optional extras (LaTeX for typeset PDFs, the knowledge layer) are in [INSTALL.md](INSTALL.md); you do not need them for this page.

## 3. Tell FI which model to use

Set the key in your terminal (or in a file called `.env` in the folder you run from, one `OPENAI_API_KEY=sk-...` line):

```bash
export OPENAI_API_KEY=sk-...          # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

Using another provider? Open `examples/integrator_bakeoff/config.yaml` and change `provider: name:` (the comment above it lists the choices). If a key is missing, FI tells you which variable before it starts anything.

## 4. Run the bundled example

```bash
python launch.py --config examples/integrator_bakeoff/config.yaml
```

It compares three numerical integrators on a damped oscillator, so it needs no data. It takes a while, and the screen shows one line for each stage as it starts — not the internal detail behind it (that goes only into `.fi/run.log` in the quest folder, for when you want it). Lines beginning with `[<quest id>]` are FI's own; the stages, in order:

| Stage | What it is doing |
|---|---|
| `ideate`, `literature` | picks the angle; searches papers and the web |
| `plan`, `design` | writes `plan.md` (what the papers say, the gap, the experiment it will run) |
| `implement`, `oracle`, `execute` | writes the code, checks it on a case with a known answer, runs it |
| `analyze`, `cross_check` | reads the results; checks them against the literature |
| `write`, `claim_check`, `review` | writes the paper, ties every claim to evidence, reviews it |

Some stages are quiet for a minute or more (searching, running the experiment). Nothing is stuck unless a line says **paused** or **failed**.

## 5. Find your paper

When it ends, the folder `outputs/<quest id>/` holds:

- `paper/paper.md` — the paper (also `paper.pdf` when a PDF engine is available; without LaTeX FI uses a fallback);
- `plan.md` — what the literature said and the experiment that ran;
- `code/` — the exact code that ran; `figures/` — every plot;
- `needs/EVIDENCE.json` — how far the result was checked.

The last line the run prints is a plain sentence saying how far the result was checked and what stands between it and the next level, for example `evidence: The number, statistics and provenance audits found the paper faithful to what the script printed. Next: ...`. **A paper is not a verified result**: the levels are explained in [rigor.md](rigor.md), and every word FI uses is in the [glossary](glossary.md).

Want to see what happened, in order? `python launch.py --trace <quest id>` ([trace.md](trace.md)).

## 6. If it stops

- **`paused`**: FI is asking for you, not failing. Read `outputs/<quest id>/NEXT_STEP.md`: it says what to do (for example, download a few paywalled papers, or accept the paper). Then continue with `python launch.py --resume <quest id>` (it finds that quest's own saved config; no `--config` needed). The bundled example is set not to stop; your own quests stop for paywalled papers and for your review of the result unless you turn that off (`pauses:` in the YAML).
- **`failed`**: `outputs/<quest id>/quest_failed.md` names the step and what went wrong; the full log is `.fi/run.log` in the same folder.
- **A `401`**: the key is wrong for the provider in the YAML.

## 7. Your own question

```bash
python launch.py --new
```

asks a few questions (topic, kind of paper, how deep, which model) and writes the config. The same questions are in the web form (`python launch.py --serve`, then http://127.0.0.1:8765) and in VSCode (`@fi /new`). To read and change the plan before anything runs, answer *Stop to read and edit the plan* (advanced): the quest stops after `plan.md` is written, and `python launch.py --resume <quest id>` runs it.

Next: the [glossary](glossary.md) for the words, [rigor.md](rigor.md) for how FI checks its own experiments, [USAGE.md](USAGE.md) for every setting.
