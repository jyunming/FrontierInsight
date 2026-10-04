# Using Frontier Insight

How to run quests from VSCode chat or the command line, plus the
YAML config schema.

## In VSCode (recommended)

Open Copilot Chat, type `@fi`. The chat participant exposes these
commands:

| Command | What it does | LLM calls |
|---|---|---|
| `@fi` *(no command)* | Starts the interactive interview — 8 quick questions (topic, title, outputs, paper format, research approach, clarify mode, reviewer panel, knowledge layer), produces a config, runs the quest. Best first-time path. | ~23–28 (one full quest, see below) |
| `@fi /new` | Same as bare `@fi`. | ~23–28 |
| `@fi /start <path-to-yaml>` | Runs a quest from an existing YAML config. | ~23–28 |
| `@fi /fleet <yaml> <yaml> ...` | Runs multiple quests in parallel. Each YAML's `provider.node_models` is honored independently. | ~23–28 × N quests |
| `@fi /resume` | Shows a picker of every quest with a checkpoint; pick one to re-enter from the last completed node. | depends on how many nodes the prior run completed; usually 3–10 to finish from a partial run |
| `@fi /resume <quest_id>` | Resumes that specific quest directly. | same — 3–10 to finish |
| `@fi /resume <quest_id> --from <step>` | Does the quest again from `ideas`, `literature`, `plan`, `design`, `skills`, `code`, `run`, `figures`, `analysis`, `crosscheck`, `evidence`, `writing`, `claims` or `review`; what that step and the later ones made is kept in `.fi/previous/<time>/`. With no step, lists the steps that quest reached. | the calls of that step and the ones after it |
| `@fi /plan <quest_id>` | Opens the quest's `plan.md` (what the literature says, the gap, the model behind the numbers, the design) in the editor, next to your other tabs, to read and edit. | **0** |
| `@fi /plan <quest_id> <what to change>` | Has the model rewrite `plan.md` as you ask; the old version is kept. Then `@fi /resume <quest_id>` runs it. | **1** |
| `@fi /summarize <folder> [kind]` | Walks a folder of mixed content (papers, code, study notes, logs) and writes a structured markdown summary. Optional `kind` ∈ `{auto, literature, code, study, execution, mixed}` — defaults to `auto`. | **1** (single LLM call, content cap'd) |
| `@fi /proposal <topic>` | Pre-quest planning doc. Writes both a markdown proposal and a companion YAML under `outputs/_drafts/`. Use to scope a research question BEFORE committing compute to a full quest. | **1** |
| `@fi /analyze <data-path> <topic>` | No-simulation quest on pre-staged data. Files under `<data-path>` are copied into the new quest's `data/` directory; the engine routes `auto_collect_data → wait_for_data → data_load → analyze → write → review`. Inverse of `/proposal` — when you already have the dataset and just want a paper analyzing it. | **~6** |
| `@fi /digest [days]` | Weekly project-manager digest across your quests: completed, in-progress, themes, ✅/🆕/⚠️/🛑/❓ diff vs prior digest, suggested next quests. Default window: 7 days. | **1** (or 0 if window is empty) |
| `@fi /portfolio` | All-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, prioritized next quests. | **1** (or 0 if no quests on disk) |
| `@fi /critique <quest_id>` | Adversarial second-pass review of a completed quest. For maximum effect, pick a different Copilot model in the picker from the one that wrote the paper. | **1** |
| `@fi /rename <quest_id> <new title>` | Changes a finished or paused quest's title (see [Changing a quest's title](#changing-a-quests-title)). | **0** |
| `@fi /help` | Lists the commands. | **0** |

### What runs, in order

In plain words a quest goes through these steps (the quest map in the Web page and in VS Code shows the same steps and lets you run from any of them again):

1. **Ideas** — turns your topic into a research question.
2. **Literature** — finds and reads the papers the question rests on.
3. **Plan** — writes `plan.md`: the gap, the model behind the numbers (what model produces them, its equations and where each comes from, and each check with a known answer (an oracle) with where its expected value comes from), how we will judge whether the code got better (two to five checks of correctness FI computes after every run, never the study's own finding), the method and the numbers to be reported. It also says what kind of study this is: a **measurement** (how a result changes over settings chosen in advance) or a **search for the best design** (the design that makes one result as low or as high as possible, within limits, compared with the design to beat, the baseline). When the topic does not make that clear, the quest asks you one question first (when the setup questions are asked; otherwise the plan decides). A search's plan shows, under *What is being optimised*, the goal, what may change and over what range, the limits, the design to beat and how many evaluations the search and the check at finer settings will take. Such a plan is never run as a plain sweep.
4. **Design** — fixes the experiment: settings, runs per setting, what counts as a pass.
5. **Write the code** — writes the simulation and its analysis as two scripts (one script with `execution.split_analysis: false`), and marks in the simulation where each equation of the plan's model is computed (`# E1`). By default the code is laid out as a small research tool: the model's equations in a package of their own (`code/<package>/`), unit tests from the plan's checks (`code/tests/test_oracles.py`) and `code/METHODS.md` saying which function computes each equation. The plan says what that costs (*How the code will be laid out*); over the limit the quest keeps two scripts and says so.
6. **Run** — first runs the checks with a known answer: FI runs the simulation on each check's small case itself and compares the result with the plan's expected value (a failing check is sent back for repair, and the quest stops before the main run if it still fails; see [rigor.md](rigor.md)); then runs them; a crash goes back to a fix-and-run loop first. When one of the plan's checks of correctness is not met, FI then changes the simulation one small step at a time (at most `engine.improve_rounds`, 3 by default) and keeps a change only when it makes a check better and none worse; a kept version is run once more in full, and that run is the one analysed. Every round is in `code/CHANGELOG.md`. A topic that needs real data collects and loads it here instead. For a search for the best design, FI itself runs the search: it tries one design after another (the baseline, the design to beat, first, then the coarse scan when the plan has one), calls the simulation once per design, stays within the plan's evaluation budget and the run's time limit, and never picks a design that breaks a limit. It writes every evaluation to `raw/optimisation_ledger.jsonl` and the result to `results/best_design.json`: the best design, the baseline scored the same way, how much better it is, the method, and how many evaluations were used. The methods are FI's own (`bounded_local`, `global_then_local`, `exhaustive`); a `scipy:` or `optuna:` method named in the plan is used when the quest's environment already has that package, and otherwise FI says so and uses its own. The best design is found at the search's own numerical settings; then FI checks it: it evaluates the best designs and the baseline (the design to beat) again at the finer settings the plan names (half, then a quarter, of the search's value when it names none), nudges the best design, and compares the designs the separate searches from different starting points ended at. `run.log` gives one plain verdict (for example "the improvement over the baseline holds at finer numerical settings", or "the improvement over the baseline disappears at finer settings") and a line per check; the full record is `needs/OPTIMUM_CHECK.json`. A design counts as better when it beats the baseline by more than the plan's threshold, or, when the plan gives none, by more than the numerical error of the two designs. A check that fails does not stop the quest: it is reported as the result and keeps the evidence level lower (see [rigor.md](rigor.md)). The check's evaluations come on top of the search's budget; `plan.md` shows how many at most.
7. **Analyse and check** — computes the findings and compares them with the literature, then checks whether the evidence is enough to write.
8. **Write the paper** — the paper, then a check that each claim is backed. FI itself (not the model) adds one paragraph to the methods (before the results when the paper has no methods section), *How this result was reached*: how many times the design was revised after the experiment had first been run and why, how many times the whole experiment was run and how many of those runs were discarded; when the design was revised, numbers from before the last revision are called exploratory.
9. **Review** — a reviewer reads it; the quest is done, or goes back.

It can go back at several points, each bounded: a crashed script is fixed and run again; a finding that needs a new experiment returns to the design, and one that needs more reading goes back to the literature first; the evidence check can send it back to the literature once, or to the design once when the run produced no results (with `engine.phased: true`, on by default for research, it also sends an accepted result back to the run once, to confirm the frozen design on held-back data or new seeds); a review that asks for changes returns to writing (the text was the problem), to running (a number was the problem) or to the design. Optional steps are left out above: the setup questions about your topic before the ideas (asked when someone can answer while the quest runs, otherwise answered by the agent; `pauses.clarify: off` skips them), and for a topic with no experiment, data collection instead of running code. When you refine a finished quest, a note about layout only redraws the figures, a missing number extends the existing script, and only a design problem redoes the experiment. The model that writes the paper decides which it is, so read the log line that says which one it picked.

### Per-quest LLM call breakdown

A single \`/start\` or \`/new\` quest made **23–28 LLM calls** in 17 complete runs of one SIR simulation quest (gemma4 through Ollama; the engine settings of those runs — `clarify_mode: off`, a single reviewer, `cross_check_per_finding_k: 3` — plus `knowledge.source_routing: manual`, slides and a poster), counted from each quest's `.fi/cost.jsonl`, which records every model call under its node's name:

| Node | Calls | Notes |
|---|---|---|
| `clarify` | 0–1 | One call unless `pauses.clarify: off`. It asks what you want to see and proposes titles. |
| `ideate` | 1 | |
| `ideate_reflect` | 0–1 | Optional self-reflection that can swap the chosen idea. Skipped when `ideate_tournament` is on. |
| `ideate_tournament` | 0 or C(N,2) | Off by default. When on with the default 3 ideas, fires 3 parallel pairwise comparisons (~one round-trip wall-clock) and picks the highest-win-count idea. |
| `ideate_query` | 1 per quest, when the knowledge layer is on | Turns the topic into a keyword query for the few sources the idea step is grounded in. |
| `literature_query` | 1 per literature pass (a resume that runs the same pass again reuses them) | Turns the topic into keyword search queries. |
| `literature_foundational` | 1 per literature pass | Names the foundational works a keyword search misses. |
| `literature_screen` | 1 per literature pass | Grades every retrieved source for citability. |
| `source_router` | 0, or 1 per literature pass + 1 per cross-check lookup | Only with `knowledge.source_routing: auto` (the default); `manual` makes no routing call. |
| `select_skills` | 0–1 | Picks the skills the quest carries; no call when no skill is a candidate. |
| `plan` | 1 | Writes `plan.md` (the literature read as a reviewer would, the gap, the design). It is the design call of the first pass with a plan directive appended, so `design` makes no call that pass; the counts above were measured before this step existed. |
| `oracle_review` | 0–1 | Once per quest, right after the plan is written, when its protocol has checks against known answers: a second model reads them as a referee would (does each test the model, would a plausible bug make it fail, is its number well defined, is there a better one, which equations no check tests). Its findings go back to the plan in the same one request as the checks not in their kind's form (`plan_revise`). Set its model with `provider.node_models.oracle_review`. Under `rigor_profile: research` the result reaches *independently validated* only when this review gave a usable answer and the record of the calls shows another model gave it than the one that wrote the checks (a connection that does not name the answering model cannot show that). |
| `plan_revise` | 0–4 | One call per `--revise-plan` request (two if the first reply's design block cannot be read). FI itself asks at most one request at the plan step, only when a check against a known answer is not in its kind's numeric form and cannot be put right without the plan, or when the second reading of the checks (`oracle_review`) found something to change, and at most one at the test run of the checks before the study, only when a measured number says the plan and the simulation mean different things by a check (each request can take a second call when its reply cannot be read); these come on top of the oracle requests counted under `implement_oracle`. |
| `design` | 1 per later design pass | Runs again when the cross-check or the review sends the quest back; the first pass adopts the design block of `plan.md`. |
| `design_self_critique` | 1 per design pass | Audits the drafted methodology against twelve checks (precision, the estimand and its interval, thresholds, random streams, convergence, an oracle, failed runs among them) and keeps what it found and changed in `needs/DESIGN_CRITIQUE.json`. |
| `implement_outline` | 1 | |
| `implement_oracle` | 0–3 (+1 per disputed check) | One repair per attempt (`engine.oracle_repair_attempts`), only when a check of the plan's oracles is missing or fails before the main run (`engine.oracle_check`; FI runs the simulation on each oracle's case itself, and a one-script quest answers them when run with `FI_ORACLE=1`), plus one retry of a repair call that got no answer (a provider timeout) and one more for each check a repair says is itself wrong (that answer is set aside and not counted); a protocol with no oracle, or one without numbers, first makes `plan_revise` calls (up to the same number, counted apart). None when the oracles pass or the plan has no protocol. |
| `implement_protocol` | 0–3 | One repair per attempt (`engine.protocol_repair_attempts`), only when the written script differs from the protocol in the plan (`engine.protocol_check`), plus one retry of a repair call that got no answer (a provider timeout); none when it agrees or the plan has no protocol. |
| `implement` | 1 per design pass | |
| `execute_reflect` | 0–3 | Only when the experiment fails; capped by `engine.exec_reflect_max_iterations`. Also when the finished run's manifest differs from the frozen protocol (`engine.run_manifest_check`, `engine.run_manifest_repair_attempts`). |
| `improve` | 0–3 | One per round, only when a check of correctness in the plan (`criteria`) is not met after a run; at most `engine.improve_rounds` (default 3) in the whole quest (a rerun `--from` the run or earlier starts it again). None when every check is met or the plan names none FI can measure. |
| `analyze` | 1 per design pass | |
| `cross_check` | 0–10 | One per key finding that found related literature, up to 10 findings (skipped when `cross_check_per_finding_k = 0`); 0–8 in the measured runs. |
| `evidence_gate` | 0–1 | Sufficiency check before write (`engine.evidence_gate`, default on); it made no call in 7 of the 17 measured runs. A `broaden` verdict re-enters literature once. |
| `web_plots` | 0–1 | No-simulation mode only — one LLM call to chart the collected data (`engine.web_derived_plots`). |
| `write` | 1–2 | Twice when the review asks for a rewrite (14 of the 17 measured runs). |
| `claim_check` | 1 per write | Grounds each paper claim to evidence before review (`engine.claim_grounding`, default on; no call when off). |
| `review` | 1 per write *or* N+1 | 1 for the single-reviewer flow; with a reviewer panel of N personas → N + 1 moderator per round. |
| `slides`, `poster` | 1 each | Only when those outputs are in `output.kinds`. |
| `human_feedback` | 0 | No LLM call — pauses for the user's accept/reject/refine when the gate is on; an accept answers one question first. |
| refine (`--refine`) | 1–3 | A note goes to `write` first. A missing number ("add n=64") extends the existing script and runs it again (one `implement` call, the frozen protocol and figures kept); if the new paper leaves that number out, the writer is asked once more (one more `write` call), and the review pause tells you if it is still missing; a layout note redraws the named figures from the saved data (one `replot_layout` call, a second only when the first script fails or changes no figure; no experiment run; if both fail the review tells you the figures were not changed); only a different study goes back to `design`. |

The measured spread came from the experiment repairs (0–3), the number of findings cross-checked, and whether the review asked for a rewrite. Four runs of the same quest on older engine versions, in which design through review ran twice, logged 27–33 calls, not counting slides and poster.

For dollar-cost estimates against specific providers (Copilot, OpenAI, Anthropic, Gemini, Ollama), see [`PROVIDERS.md#cost-expectations`](PROVIDERS.md#cost-expectations).

All LLM calls route through `vscode.lm.selectChatModels` — whatever
model is selected in your Copilot Chat picker is the model FI uses.
See [`PROVIDERS.md`](PROVIDERS.md).

## From the command line

After installing (see [INSTALL.md](INSTALL.md)), the `fi` command is on your PATH. From a checkout, `python launch.py`
is the same command.

```bash
# The first step: writes fi-demo.yaml here, checks your model at no cost, asks before running it:
fi demo

# What this machine has (LaTeX, a browser, Marp, the model providers), in a few seconds, no network;
# --deep also checks sign-in and the network, and loads the embedding model:
fi --doctor
fi --doctor --deep

# Single quest:
fi --config fi-demo.yaml

# Fleet of quests in parallel:
fi --fleet a.yaml b.yaml c.yaml --max-concurrent 4

# Resume a crashed quest:
fi --config outputs/<quest_id>/config.yaml --resume <quest_id>

# Summarize a folder:
fi --summarize ./papers --summarize-kind literature

# Pre-quest planning doc (writes both <id>-proposal.md + <id>.yaml):
fi --proposal "Compare RK4 vs Verlet on the Kepler problem with eccentric orbits"

# Weekly PM digest across your quests:
fi --digest --days 7

# All-time portfolio synthesis (no time window):
fi --portfolio

# Adversarial second-pass review of a finished quest:
fi --critique 1778452404-euv-mor-photon-shot-noise-ler-e6bfe5 \
   --critique-provider claude_cli

# Permanent paper ingest into Axon (no quest):
fi --ingest paper1.pdf paper2.md

# Local web UI:
fi --serve --output-root ./outputs

# Local web UI routing LLM calls through VSCode Copilot — open a
# VSCode integrated terminal launched by the FI extension (which
# injects FI_VSCODE_BRIDGE_PORT into the env), then:
fi --serve --output-root ./outputs
# …or pass the port directly:
fi --serve --vscode-bridge-port 12345 --output-root ./outputs

# One-time tectonic install for corporate envs:
fi --install-tectonic
```

> **First paper_pdf run takes ~30 s longer** when using tectonic (or a fresh MiKTeX install) because the LaTeX engine downloads required CTAN packages on the first compile. Subsequent runs are instant. Tectonic caches under `%LOCALAPPDATA%\TectonicProject\Tectonic\` on Windows; MiKTeX under its own package cache. No additional intervention needed — FI just waits.

### Run it from your own project folder

FI does not have to be run from its own checkout. Run it from your project folder, and everything relative means that folder: the quest goes to `./outputs/<quest_id>/` (the default `output.output_dir` is `./outputs`), `execution.inputs`, `knowledge.local_papers` and the other paths in your YAML are read relative to it, and nothing is written into FI's folder.

```bash
cd ~/my_project
fi --config quest.yaml                                   # pip install: the `fi` command
python /path/to/FrontierInsight/launch.py --config quest.yaml   # a checkout: the same thing
fi --config quest.yaml --resume <quest_id>               # resume from the same folder (or add --output <dir>)
fi --config quest.yaml --resume <quest_id> --from writing   # write the paper again, keeping the run
fi --serve                                               # the web UI watches ./outputs and starts quests from this folder
```

### Running several studies at once

Give each study its own folder and run it from there: a relative `output.output_dir` (the default `./outputs`) means the folder the command runs in, so each study's results stay in its own folder.

```bash
cd ~/study_a && fi --config a.yaml      # one terminal per study
cd ~/study_b && fi --config b.yaml
fi --fleet a.yaml b.yaml --max-concurrent 2 --memory-cap-mb 4096   # or several YAMLs from one command
```

A fleet writes every quest under the folder it runs from, unless a YAML gives an absolute `output_dir`. In VS Code, open each study folder in its own window (*File → New Window*) and use `@fi /new` or `@fi /start <yaml>` in each: every quest has its own connection to VS Code, and a window's quest lists (`/resume`, `/watch`, `/map`, …) show only its folder's quests (when it has none, the ones FI has run in other folders; see [Finding a quest from any folder](#finding-a-quest-from-any-folder)). (A `fi --serve` you start yourself in a terminal is the exception: it sends its model calls through the window opened first.) With several folders in one window, a command works in the folder holding the YAML it names, else the folder of the file you are editing, else the first folder that is not the FrontierInsight checkout.

The extension finds FrontierInsight without a setting, in this order: the open folder when it is the FrontierInsight folder; the Python in `frontierInsight.pythonPath` when FI is installed in it with `pip install -e`; the folder you picked before; else it asks once, *"Where is FrontierInsight?"*, and remembers the answer in `~/.frontier-insight/fi_location.json` for every window. `frontierInsight.repoPath` still forces one folder when set.

What FI keeps for you across studies (`~/.frontier-insight/`: installed packages, paper downloads, the Axon library's current project) is shared safely: FI takes turns on it. Studies running at the same time share the model account's rate limit (a Copilot quota, a provider's requests per minute), so they may wait or retry; two or three at once is a sensible start.

### Finding a quest from any folder

Several studies, each in its own folder, sometimes at the same time: nothing to set up. Each quest is still written where it always was (`<that folder>/outputs/<quest_id>/`, or your YAML's `output.output_dir`), and FI also remembers where, in one list for this computer: `~/.frontier-insight/quests.json` (`%USERPROFILE%\.frontier-insight\quests.json` on Windows; the `FI_HOME` environment variable moves the folder). A quest is added when it starts, updated when it resumes or is renamed, and several FI runs can write the list at the same time without losing each other's entries.

So a quest can be named from any folder, and by a short id: the six characters after the last dash of its id (`479b06` for `1790003131-energy-drift-479b06`), or any other part the id starts or ends with, as long as it names only one quest. When it matches several, FI lists them, each with its title and folder, and asks for more of the id.

```bash
fi tools quests                       # every quest: short id, where it is (running, waiting for you, finished, ...), title, folder
fi --resume 479b06                    # go on with it from any folder: it runs in its own folder, with its own config.yaml
fi --resume 479b06 --from writing     # the other commands that take a quest id accept it too
fi tools rename 479b06 A Better Title
fi tools quests --prune               # forget the quests whose folder was deleted or moved
```

The quest's own folder is always looked at first, as before. A quest found elsewhere runs as if you had gone there: from the folder it was started in (so the relative paths in its YAML mean what they meant), writing to its own outputs folder, with its own `config.yaml` (a `--config` you give instead is used, FI says so, and it runs from your folder, where that YAML's relative paths point). A quest whose folder was deleted is dropped from the list when it is looked up (when its whole study folder went too, by `--prune`); one on a disk or network share that is only disconnected is left out of lists until it is back, not forgotten, unless you `--prune` meanwhile (on Linux and macOS `--prune` also forgets quests on a disk that is not mounted). A folder you moved is recorded at its new place the next time you resume it from there (`fi --resume <its folder>`). The start line of every run shows the short id (`[FI] start quest_id=... short_id=479b06`). The web dashboard shows each quest's short id and, under **Quests in other folders**, the ones FI has run elsewhere, which open on the same quest page; in VS Code, `@fi /resume <short id>` finds a quest from another folder, and when the open folder has no quests `@fi /resume` and **FI: Quest map** offer the ones FI has run elsewhere.

### Doing a step again

`--resume <quest_id> --from <step>` does a quest again from one of its steps and keeps everything decided before it. What that step and the later ones made is moved to `.fi/previous/<time>/` first, so the old and the new outputs can be compared. `--from` with no step lists the steps that quest reached, which are the only ones it can be done again from, each with one sentence saying what redoing it does; the raw graph names (`ideate`, `cross_check`, `evidence_gate`, `claim_check`, `web_plots`, ...) work as step names too. `figures` exists only for a quest with no simulation. Some parts of the pipeline are not steps: the first question round (`clarify`; that is a new quest), the pauses, the repair of a crashed script (part of `run`) and the human review decision. Beyond this, the web quest page's menu beside **Resume** shows the same list (also while the quest waits for your review decision or setup answers; choosing a step turns the button into **Redo**), and so does `@fi /resume <quest_id> --from` in VSCode.

The web quest page also has a **Quest map** (hidden while the quest is running), and VS Code has the same one: run **FI: Quest map** from the command palette, or `@fi /map <quest_id>` (it opens next to your other tabs, never in a new split, and opening it again for the same quest brings that tab back). Every step of the pipeline is shown in seven big blocks (Understand the question, Read the literature, Plan the work, Build and run the experiment, Or: use real data, Judge the result, Write and review); open a block to see its small steps, each with a plain title and its own name in small type, so you can match it to the log. Each step is marked **finished**, **where it stopped** or **not reached** (a block for the other path, real data or simulation, is dashed). **Where it stopped** is the step the quest will run next: the pause it waits in (the review decision, the papers or plan pause, the setup questions) or the step that failed. A paused quest is therefore never drawn as finished; every step is marked finished only when the quest has nothing left to run and nothing waiting for you (no `.fi/pause.json`, `NEXT_STEP.md` or `quest_failed.md`). Click a step to see what it does, what it reads and writes, what running again from there would keep and what it would redo, the equivalent terminal command (`python launch.py --config <yaml> --resume <id> --from <step>`) and a **Restart from here** button (it asks you to press it a second time to confirm) that does the same as the menu (in VS Code it sends `@fi /resume <id> --from <step>` to chat). Only a step the quest reached can be restarted from. From the design backwards (`ideate`, `literature`, `plan`, `design`) the page says the frozen protocol has to be approved again with your name, shows the command with `--approve-as`, and leaves the button off: those are run from a terminal.

| Step | What doing it again does |
|---|---|
| `skills` | picks the skills again from the quest's current YAML, then writes the code again, runs it and does everything after. The literature, the plan and the frozen protocol are not touched |
| `code` | writes the experiment's code again, then runs it and does everything after |
| `run` | runs the same code again for new results, then analyses, writes and reviews them (with no simulation: collects and loads the data again) |
| `figures` | draws the figures again from the same data (a quest with no simulation only), then analyses, writes and reviews |
| `analysis` | analyses the same results again, then writes and reviews |
| `crosscheck` | checks the analysis against the literature and the design again, then weighs the evidence, writes and reviews |
| `evidence` | weighs how strongly the results are backed again, then writes and reviews |
| `writing` | writes the paper again from the same analysis, then reviews it |
| `claims` | checks every claim in the paper against the results again, then reviews it (the paper is not written again) |
| `review` | reviews the same paper again (and makes the slides and poster from it again) |

**Changing the ideas, the literature, the plan or the design.** `--from ideas`, `literature`, `plan` or `design` go back to before the experiment was designed, so they replace the plan and the frozen protocol; the old ones are kept in `.fi/previous/<time>/`. FI asks for your name first: add `--approve-as <you>` (without it nothing is changed and the message says so). These four are run from a terminal only: the web page and `@fi /resume <quest_id> --from <step>` in VS Code refuse them and show the command to type. The name and the replaced protocol's fingerprint go into the audit trail, and the earlier designs stay in `needs/DESIGN_HISTORY.json`, with the new one added after them as a change made after the protocol was frozen. `--from skills` and every step after the design leave the protocol alone and need no name.

**Changing the skills.** Edit the quest's YAML (`engine.skills_exclude`, `engine.skills`, `engine.skills_required`) and run `fi --config quest.yaml --resume <quest_id> --from skills`. From the command line it uses the YAML you pass, and the copy saved in the quest folder is updated to match it (the earlier copy is kept as `config.yaml.before-resume-<time>`), so a later resume does not go back to the old settings. The web page's **Redo from the skills** uses the copy saved in the quest folder, so edit `engine.skills_exclude` (or the other two keys) in `<quest folder>/config.yaml` first, or the pick will not change. A skill the new pick no longer carries is named in the console with the reason (excluded in the YAML, no longer approved or usable, or not chosen this time), and the code is written without it: its name is not asked of pip and is not offered to the code-writing step. If the plan or design text names a skill that was dropped, FI says so and leaves the plan as it is; change the plan with `--revise-plan` if you want. A skill whose own packages still cannot be installed after a second try is dropped the same way, with a line in `.fi/run.log`. `--from code` keeps the skills the quest already picked.

The steps before the skills are the plan. To change it, use `--revise-plan "<what to change>"` (or the web **Plan** panel, or `@fi /plan <quest_id> <what to change>`; `@fi /resume <quest_id> --revise-plan "<what to change>"` does the same; any other flag after `@fi /resume` is named and nothing runs), then `--resume`. After the rewrite FI says, check by check, whether each check against a known answer now says where its expected value comes from.

**Known-answer checks never stop the quest to ask you.** Whether a check or the simulation is wrong is FI's to work
out, not yours: you are never asked to compare numbers, set a tolerance or fix a script. What FI does instead:

- **A check that does not say where its expected value comes from** (`rigor_profile: research`): FI asks a model once to
  fill it in. When none is found, the quest goes on; the check still runs and is still judged, and it is marked "source not
  confirmed" in the evidence and the paper. One line in the console says which checks.
- **A check FI added or changed after you read the plan** (`pauses.plan: ask`): one line in the console names it.
  FI checks its expected value itself when it runs, and the frozen protocol says you did not approve it.
- **No measure of whether the code got better:** one line says that FI will not try to improve the simulation step by
  step. Nothing to do.
- **A known-answer check that does not pass** before the main run (a case whose right answer is known: a closed form,
  a limiting case, a conserved quantity, a convergence rate): FI first looks at it itself. It works out any arithmetic
  the plan's derivation writes out (a slip there means the check is wrong, whatever a model says), asks another model to
  work the expected value out again without showing it the result, and, when it runs the case itself, runs it again
  at a smaller step or with more seeds. Then:
  - **the check is most likely wrong, and FI has a value worked out independently of the measurement** (the plan's
    own arithmetic, or another model that never saw the result): FI corrects the plan's expected value itself, before
    the protocol is frozen, and measures the checks again. Only the expected value changes, never the tolerance, and
    never to the measured value. The console says it in one line, for example *FI corrected an expected value the plan
    had worked out wrongly: 'exact_90_deg' 1.18034 -> 2.36784, because FI worked out the plan's own derivation: ...*;
    the plan keeps the change as a version by the engine, and the audit trail records it.
  - **the simulation is most likely wrong:** FI repairs the script, up to `engine.oracle_repair_attempts` times (2 by
    default). A check FI found to be wrong is never "fixed" by changing the simulation towards it.
  - **FI cannot tell, or its repairs and corrections run out:** the quest goes on with the check marked
    *unconfirmed*. The check stays failed, the result counts as exploratory (never as checked against a known answer,
    never publication-ready), the evidence record and the paper say so, and the console says why in one line. The
    decision is recorded as FI's, bound to the check and to the version of the code that measured it.

  A recheck by the same model that wrote the plan is never counted as independent evidence; the record says when that
  was the only one. The numbers (expected and measured value, tolerance, case, the line of the script that computes it,
  what FI tried) are in `needs/ORACLE_CHECK.json` and `.fi/run.log`. A choice a person made under an earlier version of
  FI (`--accept-checks <quest_id> --approve-as <your name>`) is still honoured.

**What FI checks itself.** A failing check is often the check's own fault: an expected value worked out wrongly, a tolerance tighter than the method can reach at its step, or one random run judged against a tolerance smaller than its own noise. So before it rewrites the script for a failing check, FI looks for itself. `needs/ORACLE_CHECK.json` and `run.log` record what it found:

- **The arithmetic in the check's derivation is worked out** by FI itself, with the same small calculator the checks' formulas use (numbers, `+ - * / **`, brackets and a short list of named functions; never run as code). When the numbers a step writes out do not give the answer written after them (a real plan wrote `2*pi*sqrt(1/9.81) * (2/pi) * 1.85407 => 1.18034`, which comes to 2.368), the check is wrong whatever a model says: FI does not rewrite the script for it, and corrects the expected value to what the plan's own working gives.

- **The expected value is worked out again** by another model (the one named `oracle_review` in `provider.node_models`; without one, the model that wrote the plan, and the record says so: that recheck is never counted as independent evidence, and neither is one whose own arithmetic does not add up). It sees the check, its case and its reference, never the measured value. If its answer disagrees with the plan and is close to what was measured (within the tolerance, or within 1.5 times where the plan is ten times or more away), the expected value is marked *disputed*: FI does not rewrite the script for it, and, when the answer came from another model, FI corrects the plan's expected value to it. This is one model call per check, and a resume does not ask again.
- **The case is run again at a smaller step** when FI runs the simulation on the check's case itself (the default two-script layout), the check is deterministic, and its case sets a step (`dt`, `h`, `dx`, `n_steps` …). FI runs it at half and at a quarter of the step, which shows the order the scheme really has, and extrapolates to step 0. If the values move towards the expected one at the order the check declares, the card says the tolerance is tighter than the method's own error at this step and gives the numbers, and FI does not rewrite the script for it. Without a declared `order` that is concluded only for a conserved quantity or another rule expecting 0 (at order 1.5 or more); otherwise the card only says what it saw, because RK2 or Euler where RK4 was meant converge too. Give a check with a step its `order` so FI can tell. In the Verlet case, energy expected to stay within 1e-12 drifts by 1.25e-05 = h²/8 at h = 0.01.
- **The case is run with three more seeds** when the check is random (again, when FI runs its case itself). If the runs differ by more than the tolerance, the first run's gap is within three times that spread, and their mean agrees with the expected value, the card says the tolerance is smaller than one run's noise. Four runs cannot tell a bias smaller than about one run's spread from noise. A rule every run must keep exactly (a conserved sum, a worst violation expecting 0) that differs between runs is called the fault instead.
- **A unit or a representation** is named on the card when the measured value would pass after a usual conversion and is at least ten tolerances off without it: ×100 (a percent and a fraction), ×1000, ×10⁶ or ×10⁹ (a unit prefix), ×2π, the reciprocal, or a count in the case such as `n_agents` (a total and a mean). Only the fixed factors go to the plan in the test run before the study.
- **The same error after a repair** stops the repairs early, and the repairs left are kept. **A run that runs out of time** is run once more with twice the time before it counts as a failure.

None of this passes a failing check, and none of it loosens one. The worked-out arithmetic, the recomputed value and the smaller-step and extra-seed runs only decide whether FI rewrites the script, corrects the check's expected value from an independent value, or goes on with the check marked unconfirmed; the unit factor and the other findings are shown, and the script is still repaired for them. What it cost (model calls and extra runs) is added to `plan.md` and to `needs/ORACLE_CHECK.json`. A dispute is kept across a resume as long as the check stays as it was. A changed check (FI's correction included) is checked from scratch; FI corrects a check at most once per quest.

**Refining instead of redoing a step.** When the quest waits for your review, a refine (`--resume <quest_id> --refine "<notes>"`, **Refine** on the web quest page, or Refine in the VS Code review prompt) lets FI choose how much to redo for each point of your notes. The writer answers what the text can answer and marks each other point as one of three kinds: a number or result the study lacks (the existing script is extended as little as possible and run again, with the frozen protocol, the figures and the settings kept; a quest with no simulation collects the missing data instead), a figure to arrange or draw differently (the named figures are redrawn from the numbers the run saved, and the experiment is not run again), or a different study (FI goes back to the design). A number added this way is checked to be in the new paper; if it is left out, the writer is asked once more, and if it is still left out the review says your request was computed but is not in the paper. A redraw that fails twice leaves the figures as they were, and the review says the request was not applied. `run.log` says which kind each refine came to: the paper only, a missing number, a figure redraw or a new experiment (in the decision trace, the `refine_scope` field of the `write` step: `paper`, `data`, `layout` or `experiment`).

**Searching further for a better design.** For a search for the best design, the paper has a section *Best design found* that FI writes from its own records (the baseline against the best design at each numerical setting, the improvement and its numerical error, how the search ran, the check at finer settings, and the limits), and `results/best_design.md` holds the same text. If the best design is not good enough, refine with "push further", a target (`--refine "at most 46 K"`, or `"6 K better than the baseline"`; a number is read in the objective's unit, "9500 mK" is converted to K, a percent such as "at most 5%" means that much better than the best design so far, and a unit that does not fit the objective is refused with a request for the value in the objective's unit; FI restates the target in the objective's unit before searching so a misreading can be caught) or a number of evaluations (`"100 more evaluations"`): FI goes on from the best design so far (the search already run is not run again), stops when a design reaches the target, checks the result at finer settings again and says whether the target was reached. Unless you give a number, it adds one starting point's share of the plan's budget. Such a refine is answered by the search alone, so send any change to the text as a refine of its own. Rerunning from the run (`--from run`) starts the search again at the plan's own budget: the extra rounds a refine added are not kept, so ask for them again. Asking to change what is optimised, a limit or a range is a new study: FI changes nothing and the review pause says so; start a new quest for it.

**Accepting the result.** Before you accept, FI shows what the result does not guarantee and its most important evidence gaps, and asks one question: *Have you reviewed the evidence record, and do you accept these claims and the limits listed?* Answer with the accept: `--resume <quest_id> --accept yes`, `--accept partly "what you do not accept"` or `--accept not-checked`; `--accept` alone asks at a terminal. On the web quest page **Accept** opens the question, and in VS Code a second prompt asks it. Only "yes" lets the result count as publication-ready. "Partly" needs a short note saying what you do not accept; it and "I did not check" both finish the quest and leave a gap in the evidence (the note is shown there) until a new paper is accepted with "yes". "No" does not accept: refine with what is wrong, or read the paper again. The answer is recorded with who gave it (your login name, or `--approve-as <name>`; on the web page the name you type), when, and the paper and evidence it was given for. A result accepted with no person asked (`pauses.auto_accept_on_pass`, or `pauses.review: off`) is marked "not reviewed by a person" and stays one level below `publication_ready`.

### Changing a quest's model

Put the model you want in the quest's `config.yaml` (`provider.model`, in the quest folder) and resume it (`fi --resume <quest_id>` or **Resume** on the web quest page). In VS Code, pick the model in the chat panel and `@fi /resume <quest_id>`: the chat panel's model wins over `config.yaml` for a quest resumed from the chat (with the picker on *Auto*, `config.yaml`'s model is used). That choice is for the run it starts only: the chat shows `[FI] the chat panel's model claude-opus-5 replaces gpt-5.6-luna from config.yaml`, the change goes into the trace and the paper, and `config.yaml` is not changed, so a later resume from a terminal or the web page uses `config.yaml`'s model again. What follows is the `config.yaml` way. The quest does not stop to ask: a different model does not change what the result means or how strictly it is checked, so it needs no approval. The resumed quest says so in one plain line, for example:

```
[FI] model: the model changes from gpt-5.6-luna to claude-opus-5 from here on; steps already done were made by gpt-5.6-luna
```

The same line goes into `.fi/run.log`, the change goes into the quest's decision trace (a `model_changed` event) and into the record of its approved settings, and the paper is told to say in its methods that more than one model produced the study (unless nothing had run on the old model yet: then the line says so and the paper is told nothing). Steps already done are not run again; to redo some with the new model, add `--from <step>` (see *Doing a step again*). Single steps you gave their own model in `provider.node_models` keep it, and the line names the ones still on the old model so you can change them there too. `.fi/cost.jsonl` keeps every call ever made, so the calls made before the change still show the old model; the ones after it show the new one. The same holds for the provider and `provider.node_models`. Two things can still stop the quest: under `rigor_profile: research`, every reviewer ending up on one model, also after the reviewers have already read the paper once (the quest says how to give one reviewer another model), and any setting that changes what the result means or how it is checked (the rigor profile, a check turned down, the reviewers, the multi-model ensemble `provider.node_ensemble`, which decides how many models vote on a check): approve those with `fi --update <quest_id>`, the web quest page's **Update**, or `@fi /update <quest_id>` in VS Code, which runs in the chat, shows what stopped the quest, and asks you to approve the settings in `config.yaml` as they are (or opens the file to change it first).

If you resume with another config (`fi --config edited.yaml --resume <quest_id>`), that config is what runs, and the quest's own `config.yaml` is updated to match it (each earlier one is kept beside it as `config.yaml.before-resume-<time>`), so the next resume keeps the new model.

### Changing a quest's title

When the title the model chose is a poor one, change it once the quest has finished (or while it is paused):

```bash
fi tools rename <quest_id> Energy Drift of Symplectic Integrators at Large Step Sizes
```

This changes the title line of the paper, the `title` in the quest's `config.yaml`, the summary and the saved state a resume reads, and records the change in the quest's trace. Results, data and code are not touched. It is refused while the quest is running, and the title must be one line of at most 200 characters. A PDF, slides, poster or talk script already made still show the old title: the command lists them with the command that makes each again from the same paper, for example `python launch.py --resume <quest_id> --emit paper_pdf` (no re-run of the research). A rerun with `--from <step>` (the writing included) starts from before the rename and may choose another title; rename again afterwards. A title that starts with `-` goes in as `--title="-..."`. Renaming a finished quest does not lower its evidence level: the change is recorded after the trace's seal with the paper's fingerprint before and after. On the web quest page, use **Rename** beside the title (it offers to make the outputs again); in VS Code, `@fi /rename <quest_id> <new title>`.

`--resume` looks under the folder's `output.output_dir` first, then among every quest FI has run on this computer (see [Finding a quest from any folder](#finding-a-quest-from-any-folder)); if it finds none it says where it looked and names close matches. In VSCode, open your project folder; quests run in it (`frontierInsight.workingDir` overrides that), and FrontierInsight is found without a setting (see *Running several studies at once*). Skills kept in your project (`.claude/skills`, `.agents/skills`, `./skills`) are found from there too.

### All `fi` flags

| Mode | Args | Notes | LLM calls |
|---|---|---|---|
| `--config <yaml>` | one YAML path | single-quest run | ~23–28 (see chat-command section above for the per-node breakdown) |
| `--fleet <yaml> <yaml> ...` | one or more YAMLs | parallel quests, `--max-concurrent N` controls cap | ~23–28 × N quests |
| `--ingest <file> <file> ...` | one or more PDFs / MDs / TXTs | one-shot Axon ingest, no quest | **0** (embeddings only; no LLM) |
| `--serve` | none | starts the FastAPI status GUI at 127.0.0.1:8765 | **0** (GUI is read-only over existing outputs) |
| `--summarize <folder>` | one folder | folder summarizer, pairs with `--summarize-kind` | **1** |
| `--proposal <topic>` | one topic string | pre-quest planning doc + companion YAML under `outputs/_drafts/` | **1** |
| `--analyze <data-path>` | one directory + `--analyze-topic "<topic>"` | no-simulation quest on pre-staged data (files copied into the new quest's `data/`); routes through `auto_collect_data → wait_for_data → data_load → analyze → write → review` | **~6** |
| `--digest` | none | weekly PM digest, pairs with `--days N` (default 7) | **1** (or 0 if window is empty) |
| `--portfolio` | none | all-time cross-quest synthesis (no time window) | **1** (or 0 if no quests on disk) |
| `--critique <quest_id>` | one quest_id | adversarial second-pass review | **1** |
| `--rename <quest_id> <new title>` | quest_id, then the title | change a finished or paused quest's title (paper, config, summary, saved state); also `fi tools rename` | 0 |
| `--why <quest_id> [stop\|review\|evidence\|reasons\|<step>]` | quest_id, then what to ask | why the quest stopped, why the review asked for a revision, why the evidence is at its level, why one step decided what it did, or (`reasons`) the reasons the model gave at every step. A step's answer also says whether `.fi/thinking.jsonl` holds the model's reasoning for it and how many characters (the text stays in the file). A reason the model wrote, and a reasoning summary it returned, are its own account of itself, not its hidden reasoning. Web: **Why?** and **Model's reasons** on the quest page; VS Code: `@fi /why`. See [docs/trace.md](trace.md) | 0 |
| `--install-tectonic` | none | downloads tectonic to `tools/` for no-admin LaTeX | **0** (network download only) |

| Flag | Mode | What it does |
|---|---|---|
| `--max-concurrent N` | fleet | cap on parallel quests |
| `--memory-cap-mb N` | fleet | throttle new quest starts when RSS exceeds N MB |
| `--profile` | quest | dump per-quest viztracer trace if viztracer installed |
| `--output <dir>` | quest | override `output.output_dir` in the YAML |
| `--interactive` | quest | talk the topic over first: read the clarify answers (what you want to see, the title, ...) from stdin. Without it a headless run lets the agent answer for itself |
| `--resume <quest_id>` | quest | re-enter a checkpoint. Without `--config` it uses the `config.yaml` saved in the quest's folder; the id may be the short id, and the quest may be one started in another folder (see [Finding a quest from any folder](#finding-a-quest-from-any-folder)) |
| `--quests` | list | every quest FI has run on this computer, from any folder (`fi tools quests`); `--prune` first forgets those whose folder is gone, `--json` prints JSON |
| `--from <step>` | quest | with `--resume` or `--rerun`: do the quest again from `ideas`, `literature`, `plan`, `design`, `skills`, `code`, `run`, `figures`, `analysis`, `crosscheck`, `evidence`, `writing`, `claims` or `review` (see [Doing a step again](#doing-a-step-again)). What that step and the later ones made is moved to `.fi/previous/<time>/` first. `--from` with no step lists the steps the quest reached. The web quest page's menu beside **Resume** and `@fi /resume <quest_id> --from <step>` do the same |
| `--summarize-kind <kind>` | summarize | content-type hint, default `auto` |
| `--summarize-provider <name>` | summarize | LLM provider for the summarize call |
| `--days N` | digest | digest window in days, default 7 |
| `--digest-provider <name>` | digest | LLM provider for the digest |
| `--portfolio-provider <name>` | portfolio | LLM provider for the portfolio synthesis |
| `--critique-provider <name>` | critique | LLM provider; set DIFFERENT from quest's original for max adversarial effect |
| `--proposal-provider <name>` | proposal | LLM provider for the proposal |
| `--axon-config <yaml>` | ingest, summarize, digest, portfolio, critique, proposal | optional AxonConfig path |
| `--output-root <dir>` | serve, summarize, digest, portfolio, critique, proposal | quest output dir / scan root |
| `--host`, `--port` | serve | bind elsewhere than 127.0.0.1:8765 |

## YAML config schema

```yaml
# Required: the research question. Free text, multi-line is fine.
topic: |
  Compare three numerical integrators on a damped harmonic
  oscillator (RK4 vs Velocity-Verlet vs forward Euler). Report
  energy drift over 10⁴ periods.

# Optional: short identifier used in folder names. Defaults to a
# slug derived from the topic.
title: integrator-bakeoff

# Optional: `research` turns on, together, what a study needs before its result can be trusted: the plan is held for you
# to read (pauses.plan: ask), the simulation and its analysis stay in two scripts and a reply without both stops the quest
# (execution.split_analysis: true, split_failure: block), the protocol / oracle / numeric-warning / run-manifest checks
# stop the quest and cannot be turned off, and the cross-check verification and a review panel are on. A config that sets
# one of those to the opposite is refused with the key named. Default: default (nothing changes). The interview sets it
# from "What is the result for?": research (the default answer) or a decision writes research; exploring leaves it at
# default and writes the draft's three cheaper engine settings (ideate_reflect: false, cross_check_per_finding_k: 0,
# enable_analyze_reroute: false), the same on every interface. research also keeps the decision trace on and needs the
# reviewers not all on one model (provider.node_models["review_panel.<role>"]); a panel all on one model stops
# the quest before it runs, saying how to set one. For research or a decision the interview asks for that model
# right away ("A different model for one reviewer", written as review_panel.statistician); its "I only have one model" writes
# engine.one_model_review: true instead, so the quest runs and its result says the review was one model's view and
# is never publication-ready. Under research the plan's checks count as independent evidence only when a second,
# different model read them (provider.node_models.oracle_review) and, for invariants, symmetries and second
# implementations, they also held at a setting FI chose that the code never saw (needs/HIDDEN_CHECK.json).
# See docs/rigor.md.
rigor_profile: default
# What the result is for: explore, research or decision (the interview writes it). Left out, the quest is an
# exploration: it runs, its paper opens with a note that the result is preliminary, and it never reaches
# publication-ready (with rigor_profile: research, left out means research).
result_use: explore

provider:
  name: vscode_extension           # see PROVIDERS.md
  model: gpt-5                     # global default
  base_url: null                   # only for HTTP-direct overrides (OpenAI-compatible proxies, local gateways). Honored by openai/codex/gemini/ollama/vllm transports.
  api_key_env: null                # override the standard env-var name (e.g. CORP_OPENAI_KEY). When null, the provider uses its conventional name (OPENAI_API_KEY, GEMINI_API_KEY, …).
  reasoning_effort: null           # minimal | low | medium | high | xhigh | max. Unset (null) sends nothing, so each provider keeps its own default. Sent as `reasoning_effort` (HTTP), `--effort` (claude_cli, antigravity_cli) or `model_reasoning_effort` (codex_cli); a level a provider cannot take is left out with one warning. See PROVIDERS.md, "Reasoning effort".
  fixed_temperature: null          # HTTP providers only. Some OpenAI-compatible models accept one temperature and answer any other with HTTP 400 (Moonshot's Kimi K2.6 / K3: 0.6 with thinking off, 1 with it on). When set, it is sent on every call in place of the per-node temperatures. Not passed to a fallback provider.
  extra_body: {}                   # HTTP providers only. Fields merged into every request body, e.g. Kimi's {thinking: {type: disabled}}, which turns its reasoning off. Not passed to a fallback provider.
  extra: {}                        # forward-compat transport bag. Currently only ``bridge_port`` is consumed (``vscode_extension`` transport, set automatically by ``launch.py``). Other keys parse fine but no transport reads them today — don't rely on stashing CLI flags or HTTP headers here.
  # Per-node override (optional). Match keys exactly to engine node
  # names. Reviewer-panel personas are routed via
  # `review_panel.<persona>`; the moderator via `review_moderator`.
  node_models:
    clarify:       gpt-4o-mini
    write:         claude-3-5-sonnet
    review:        gpt-5
    oracle_review: claude-3-5-sonnet   # the second reading of the plan's checks; under research a model other than the plan's
  # The most a step's answer may be, in tokens (HTTP providers only; the same step names as node_models). Unset, no
  # limit is sent and the model's own applies. When a step's answer is cut off at its limit, the quest stops and says
  # which step to raise here (or to give another model); a limit set here is first tried once more at twice the size.
  # Step names: plan (the design), implement, write, review, review_panel, claim_check, cross_check, analyze; a name
  # with a dot (write.patch) uses its first part. A model refuses a number above its own maximum: then lower it.
  # Not sent to the CLI providers or the VS Code bridge, which set their own. Ollama can also stop an answer when its
  # context window is full; then Ollama's `num_ctx` needs raising (not checked). Changing node_models (or the model)
  # of a quest that already ran needs no approval: the resumed quest takes it and says so.
  node_max_tokens:
    write:     16000
    implement: 32000

  # Multi-model ensemble per node. Fans out a node's chat call across
  # `models` in parallel and merges with `merge`. Supported on
  # ideate / analyze / cross_check. Cost: N + 1 calls per ensembled
  # node (N fan-out + 1 moderator), except `merge: vote` which is N
  # (no moderator — pure tally). The interview's `ensemble_profile`
  # slot writes this block for you; edit by hand when you want a
  # custom trio or non-default merger.
  node_ensemble:
    ideate:
      models: [gpt-4o, claude-3-5-sonnet, gemini-2.5-pro]
      merge: tournament          # tournament | synthesize | vote
      moderator: claude-3-5-sonnet
    cross_check:
      models: [gpt-4o, claude-3-5-sonnet, gemini-2.5-pro]
      merge: vote                # pure majority tally — no moderator
    # analyze accepts tournament or vote — never synthesize: the
    # analyze parser expects JSON, but synthesize emits markdown, so
    # the ProviderConfig validator rejects that combination at load.

engine:
  framework: langgraph              # the only value supported today
  max_iterations: 2                 # design-revise loop budget
  review_loop: true                 # enable review-driven revise
  audit_trace: true                 # write .fi/audit.jsonl: what ran, each check, each route, the model's stated reasons; see docs/trace.md
  attempt_memory: shadow            # at each decision, record what past failed attempts (of the quests under the same output folder) would recommend, and act on none of it; `off` reads and writes nothing
  phased: false                     # explore, then confirm: the model may try designs and look at results; then the design is frozen and run once more on a part of your data held back before exploration (one CSV/TSV table of at least 40 rows, in inputs/data/ or, for a quest that analyses data, data/; whole subjects/sites or the latest period, decided by the plan's `protocol.split`, plan.md's line "Rows that belong together:", or the table's columns; a research quest asks when it cannot tell), or else on new random seeds; only that confirm run can be publication-ready; costs one more full run (a data quest: one more reading of the held-back rows). Without a container the held-back rows are kept encrypted and the code is checked for paths out of the quest folder. Off here (the default profile); on by default under rigor_profile: research, where `phased: false` keeps it off and plan.md says the result is not confirmed on unseen data. A survey or --analyze says it does not apply. Each version is confirmed once: a retry keeps both verdicts and a failure counts; a version changed after its confirm run (improve, a repair, a re-run the review asked for) is confirmed on its own; every verdict is kept in .fi/confirmations.jsonl. The interview asks it as "Confirm the result on data it never saw" (see docs/rigor.md)
  one_model_review: false           # only one model available: under rigor_profile: research the review panel may run on it (no stop for the reviewers' models); the result is then not publication-ready. Not part of the approved settings: the evidence level already shows it
  # clarify_mode:                  # old name of pauses.clarify (off | auto | interactive = ask). Left out (the default), the quest asks its setup questions when someone can answer while it runs, else answers them itself; see Setup questions below
  ideate_reflect: true              # extra self-critique pass (1 LLM call)
  ideate_tournament: false          # pairwise tournament across brainstormed ideas; replaces ideate_reflect; C(N,2) calls in parallel
  exec_reflect_max_iterations: 3    # execute-repair loop bound
  improve_rounds: 3                 # after a run, when a check of correctness in the plan is not met: how many times (in the whole quest; a rerun --from the run or earlier starts again) FI may change the simulation, one small change at a time, keeping a change only when no check got worse by more than its own tolerance; a kept version runs once more in full. 0 = off. Under rigor_profile: research a change that made a check worse stops the quest for you
  pilot_run: false                  # OPT-IN. Run the experiment SMALL first (FI_PILOT=1, which the implement prompt tells the script to honour), then full scale. execute_reflect already repairs a script that CRASHES; the pilot catches one that runs fine and answers the wrong question — a sweep over the wrong parameter range, a resolution too coarse to show the effect — which otherwise costs the full timeout to discover. The pilot's numbers are DISCARDED: it is a smoke test of the design, not a measurement, and it never fails a quest (a bad pilot warns and the full run proceeds). Off by default: honouring FI_PILOT is a prompt instruction the engine cannot enforce, and a script that ignores it runs full-scale under a fifth of the timeout — timing out and warning on every quest. Enable it once your scripts comply.
  pilot_timeout_frac: 0.2           # Pilot timeout as a fraction of execution.timeout_s, floored at 30s. A pilot that takes as long as the real run buys nothing.
  cross_check_per_finding_k: 3      # per-finding lit-check hits, 0 to disable
  enable_analyze_reroute: true      # analyze can request re_experiment / broaden_lit
  review_panel:                     # empty = single reviewer
    - methodologist
    - statistician
    - devil_advocate
    # available: methodologist, statistician, devil_advocate, reproducibility
  no_simulation: false              # see "Topics that need real data" section below
  survey_mode: false                # literature/history synthesis: NO experiment AND NO dataset (implies no_simulation). See "Survey mode" below
  auto_collect_data: true           # try Axon for evidence before pausing for user data (no_simulation mode)
  auto_collect_top_k: 5             # Axon top_k for auto_collect_data
  dataset_adapters: []              # structured-data + web-fetch adapters. Available: "worldbank", "wikipedia"
  dataset_adapter_top_k: 3          # rows per adapter

execution:
  sandbox: venv                     # venv (default) | docker
  timeout_s: 600
  inputs: []                        # example files/folders for the experiment (any type); copied to inputs/examples/, FI_INPUT_DIR
  background_jobs: false            # the simulation runs as an HPC/cluster job: experiment.py submits it and reports pending; --watch wakes the quest
  split_analysis: auto              # auto (default: two scripts for every quest that runs a simulation, deterministic or random) | true | false. Keep the simulation (code/simulate.py: run_cell for a deterministic study, run_trial for a random one, run by FI, record in raw/) apart from its analysis (code/experiment.py), so FI can run each oracle's case itself; false keeps one script and the result then cannot reach independently_validated; with background_jobs the trials run as a job array (code/submit.py submits FI's tasks)
  code_package: true                # default: with two scripts, the code is a small research tool (the model's equations in code/<package>/, tests/test_oracles.py and METHODS.md written by FI); false keeps two scripts
  code_package_max_extra_lines: 400 # over this many extra lines of code (estimated at plan time) the quest keeps two scripts and says so in plan.md and run.log
  code_package_max_extra_calls: 3   # at most this many extra requests to the model in the whole quest (only when a reply leaves the package out); once spent, the quest keeps the code it has and says so
  raw_dir: ""                       # only with split_analysis: where the raw files go (relative to the quest, or absolute; relative with docker); empty = raw/
  shared_interpreter: true          # default: run quest code on the Python that runs FI, no per-quest venv
  python_version: "3.11"            # only when shared_interpreter: false (venv per quest)
  system_site_packages: true        # only when shared_interpreter: false; venv sees FI's packages
  docker_image: python:3.11-slim    # for sandbox=docker
  docker_memory_gb: 4               # sandbox=docker: memory one experiment may use; over it the run is stopped and run.log says so
  docker_cpus: 2                    # sandbox=docker: CPU cores' worth one experiment may use (0.5 = half a core); over it the run is slowed, not stopped
  docker_max_processes: 1024        # sandbox=docker: processes and threads one experiment may run at once; one more cannot start

knowledge:
  enabled: true
  # Retracted papers: whenever knowledge is on, every source the search finds with a DOI is looked up in Crossref (a paper you pinned or dropped in without one is first matched to its DOI by its exact title, and by first author and year where it states them, else it is "could not be checked"; Crossref holds the Retraction Watch database; arXiv DOIs are not sent, Crossref holds none); there is no switch for it. A retracted one is marked [retracted] for the writer, never counts as support, is left off that pass's list of papers to download, and is named on the to-do card; any sentence that cites one is marked unsupported by the claim check. A lookup with no answer is "could not be checked", never "not retracted", and never stops the quest; offline it costs one request's timeout per literature pass (15 s, up to about 30 s behind a proxy that never answers). One line in run.log; each source's answer in .fi/literature_queries.json.
  # Inline AxonConfig (or pass a path to a YAML). Use Axon's NESTED shape
  # as below, not its flat field names — FI hands this to AxonConfig.load,
  # which does the nesting -> field mapping, env overrides and retired-key
  # filtering itself.
  #
  # CAUTION on `embedding`: changing it on a store that already has vectors
  # makes every search fail with `query dimension mismatch: expected 384,
  # got 768` — embedding dimension is a property of the STORE, not of a run.
  # Omit `embedding` to keep whatever the store was built with; only set it
  # for a fresh store.
  # axon_config applies only with axon_mode: in_process. The default, axon_mode: http, uses the Axon service that is
  # already running (started for you when it is not), which keeps the configuration it was started with.
  axon_config:
    embedding: { provider: ollama, model: nomic-embed-text }
    llm:       { provider: ollama, model: qwen2.5-coder:32b }
  top_k: 8                          # Axon RAG cap — dense hits are precise, 8 strong matches beat 20 medium ones for the writer prompt. The interview's "Axon hits per quest" question.
  external_top_k: 20                # External (arXiv / OpenAlex / Crossref / S2 / ...) cap when Axon misses. Bigger than top_k because web search is coarser; bump to 30 for survey-shaped quests.
  relevance_min_score: 0.20         # Literature relevance FLOOR: drop retrieved docs whose embedding cosine vs the TOPIC is below this, before they reach analyze/write. Runs in the literature node for EVERY quest (unlike relevance_guard, which only runs on the auto_collect path), so survey/simulation quests don't carry off-topic sources (e.g. change-point-math papers for a sculpture-history topic). 0.0 disables. Fail-open when embeddings are unavailable (FI_OFFLINE).
  relevance_min_keep: 3             # Never-starve retention: keep at least this many top-scoring docs even if all fall below the floor (the evidence_gate can then broaden).
  requery_on_low_relevance: true    # When NO doc clears relevance_min_score on its own merits, the query was probably worded badly (a field publishes under different terms than the topic statement uses). Ask the model for an alternative query and search again, instead of handing the writer the relevance_min_keep least-bad hits as if they were evidence. Skipped when embeddings are unavailable — without scores there is no signal the query was bad.
  requery_max: 2                    # Bound on those retries. Each costs one small LLM call plus a retrieval.
                                    # Your own search queries instead of FI's: write them in <quest folder>/inputs/search_queries.txt, one per line (first three used, # lines skipped); remove the file to go back to FI's. Listed in .fi/literature_queries.json.
  literature_screen: true           # One batched LLM call grades every retrieved source 0-3 ("could the paper cite this?"). Papers need 2, web pages are dropped only at 0; keeps at least relevance_min_keep; fails open. Your own local_papers / inputs/papers are never screened. Every source's grade and why it was kept or dropped is listed in .fi/literature_queries.json.
  foundational_works: true          # One LLM call names up to 8 foundational works (a method's original paper, a standard textbook); each is looked up by title in OpenAlex, or by author and year when no title matches, and kept only if found, plus the works at least 2 retrieved papers cite. Books count. All go through literature_screen, and the writer is asked to cite the ones that bear on the paper. Up to 18 OpenAlex requests per literature pass.
  write_back_quests: true
  write_back_only_on_accept: true  # accepted evidence only when a study (research/decision) was accepted AND reached publication_ready; anything else written back is kept as preliminary (a reminder later quests never cite)

  external_fallback: [openalex, arxiv, crossref]   # arxiv is searched through OpenAlex's arXiv source (arXiv's own query API is throttled for everyone). Also: semantic_scholar, pubmed, core, openaire, doaj (the last three keyless; good for humanities / social science), google_scholar. Crossref / OpenAlex / OpenAIRE keep papers only, plus books and chapters when the quest has no experiment.
  openalex_api_key: ""              # or env OPENALEX_API_KEY. Without a key OpenAlex allows ~100 searches/day; a quest uses dozens. Env wins over YAML.
  semantic_scholar_api_key: ""      # or env SEMANTIC_SCHOLAR_API_KEY. The keyless pool mostly answers 429.
  source_routing: auto              # auto (LLM picks) | manual
  seed_source_catalog: true

  # Pinned local papers (always first in retrieval):
  local_papers:
    - ~/papers/foundational-paper.pdf
    - ~/papers/local-note.md

  try_fetch_full_text: true         # download the full text of papers that are free to read (a paywalled paper is never downloaded; you add it yourself). false = abstracts only
  full_text_fetch_timeout_s: 15.0   # per-URL fetch timeout in seconds
  full_text_fetch_total_s: 90.0     # wall-clock cap across all URLs in one query
  full_text_max_kb: 10240           # most text kept of one source, KB (10 MB: every page of a paper, scans read by OCR)
  # (no key) Text in a source that looks like instructions to an AI model, or is hidden from a reader, is flagged in run.log and the trace, never removed.
  read_figures: true                # read the values off the papers' figures (saved in data/literature/figures/); model: provider.node_models.figures

output:
  kinds: [paper_md, paper_pdf]
  paper_format: generic             # scientific: generic | neurips | iclr | ieee_access | nature_mi; non-scientific prose: essay | report | policy_brief | whitepaper
  output_dir: ./outputs
  save_model_calls: false           # keep every prompt and answer, whole, in .fi/io/ (one file per call); see docs/trace.md
  save_thinking: true               # keep the reasoning summary the provider returns, when its connection gives one, in .fi/thinking.jsonl (for reading; it is not the model's hidden or complete chain of thought, and never evidence); false keeps none (set it for a quest that handles sensitive data), and on the VS Code connection stops asking Copilot for the model's reasoning (asked by default: Copilot's undocumented `_enableThinking` option for Claude, whose Opus returns a summary, and `includeEncryptedThinking` for a GPT model's reasoning summary). `--why <quest> <step>` says whether the file holds that step's reasoning and how long it is. It can quote your data and prompts: only credentials and your home folder are removed, so check it before sharing the quest folder
  require_pdf: false                # strict mode for paper_pdf — see below
  html_pdf_fallback: true           # when no LaTeX engine: render paper.pdf via pandoc → HTML → headless browser (Edge/Chrome/Chromium). Default on. See below.
  paper_style: latex                # paper.pdf look: latex (Computer Modern article, default) | briefing (FI brand look, HTML-rendered)
  author: ""                        # optional author line on the paper, slides and poster — see below
  affiliation: ""
  contact_email: ""
  url: ""                           # project link; the poster prints it as a QR code
  poster_size: a1_portrait          # a1_portrait (default) | a0_portrait | landscape_48x36

# Reserved free-text steering slot — declared in ``core/config.py``
# but NOT YET wired into any prompt template or ``Engine._chat`` path
# as of today. Parses and round-trips through the schema; setting it
# has no behavioural effect until a future PR threads it into the
# system prompts. Documented here so users see the field exists.
extra_directives: ""
```

### `output.require_pdf` — strict-mode PDF enforcement

By default, if `paper_pdf` is in `output.kinds` but the host lacks
pandoc or a LaTeX engine, the engine emits a WARNING and continues —
the quest runs to completion, writes `paper.md`, and drops a
`paper_pdf_skipped.md` diagnostic file next to the markdown. You
still pay the LLM cost (~15 minutes) but get no PDF.

Set `output.require_pdf: true` to upgrade that warning to a hard
failure in **both** of these moments:

1. **Pre-flight (before any LLM calls)** — the engine checks
   `pandoc` + a LaTeX engine (`pdflatex` on PATH, `tectonic` on PATH,
   or a repo-local `tools/tectonic[.exe]` written by
   `python launch.py --install-tectonic`). If any prerequisite is
   missing, the quest aborts immediately with the install recipe — no
   LLM cost incurred.
2. **Post-LLM compile** — even when prerequisites are present at
   pre-flight, the actual pandoc/LaTeX compile can still fail at the
   end (timeout, nonzero LaTeX exit, output file missing despite
   rc=0). In strict mode these surface as a `RuntimeError` that fails
   the quest, instead of a "completed" quest that silently lacks a
   PDF.

Recommended for unattended / CI runs where a missing PDF means the
output is unusable anyway. Leave at the default `false` for
interactive use where you'd rather have `paper.md` + a diagnostic
than no output at all.

### `output.html_pdf_fallback` — LaTeX-free PDF rendering

A 4th engine tier, **on by default**. When none of `pdflatex` /
`tectonic` / `tools/tectonic` is reachable, FI renders `paper.pdf`
*without* LaTeX: `pandoc` turns `paper.md` into a Computer-Modern-
styled HTML page (the `templates/paper/_html/latexlike.css` theme,
with Latin Modern Roman embedded so it matches the LaTeX `article`
look), then a headless system **browser** (Edge / Chrome / Chromium)
prints it to PDF. Figures and fonts are inlined, so the PDF is
self-contained.

This means a machine with **pandoc + a browser but no TeX
distribution** — common on locked-down corporate laptops where you
can't install MiKTeX — still produces a styled `paper.pdf`. The
pre-flight knows about it too: with `require_pdf: true`, a present
browser satisfies the prerequisite, so the quest isn't aborted just
because no LaTeX engine is installed.

Trade-off: the fallback can't reproduce a venue's two-column LaTeX
class (NeurIPS/IEEE), so it always renders the single-column house
style. Set `html_pdf_fallback: false` to force the strict LaTeX-only
path — then a missing engine skips the PDF (or, with `require_pdf:
true`, aborts) exactly as before.

### Chinese, Japanese and Korean text

pdflatex stops at the first Chinese, Japanese or Korean character,
whether it is in the title, the body or the author line. When
`paper.md` or the author line has such text, FI compiles the paper
with **XeLaTeX** and the `xeCJK` package in an installed CJK font. The
template and its layout stay the same.

- **Font:** the first one installed of Noto Sans CJK / Source Han Sans,
  then the system font for the language. That is Microsoft JhengHei
  (Traditional Chinese), Microsoft YaHei (Simplified), Yu Gothic or
  Meiryo (Japanese), or Malgun Gothic (Korean). FI finds fonts through
  fontconfig (`fc-list`, which ships with MiKTeX and TeX Live) or, on
  Windows, in the fonts folder.
- **XeLaTeX** comes with MiKTeX and TeX Live. tectonic is XeTeX
  underneath and works as is.
- **Linux:** install a CJK font first, for example `sudo apt install
  fonts-noto-cjk`.
- **No XeLaTeX or no CJK font:** the paper goes straight to the HTML
  fallback above, which sets the text in the browser's fonts. With
  `html_pdf_fallback: false` it is skipped with a `cjk_no_xelatex` or
  `cjk_no_font` diagnostic.

### `output.paper_style` — choose the paper.pdf look

`latex` (default) renders `paper.pdf` with the venue LaTeX template —
the classic Computer Modern article. `briefing` instead renders the
Frontier Insight **"Research Briefing"** look: warm off-white paper, a
deep-teal accent, a serif display face, and the brand mark — the same
identity as the slides and poster. It's produced by the same
HTML/Chromium backend as the fallback above (pandoc + a browser, no
LaTeX), so it works on a machine without a TeX distribution, and falls
back to the LaTeX path with a warning when pandoc + a browser aren't
both present. Because it's HTML, it's single-column regardless of
`paper_format`. Pick it in YAML (`output.paper_style: briefing`) or
during the interview (`--new` / `@fi /new` / web — the *Paper style*
question).

### Author line and poster size

`output.author`, `output.affiliation`, `output.contact_email` and
`output.url` put your name on the outputs. The paper prints them under
the title (and uses the author as the PDF's Author field), the slides put
them on the title slide, and the poster puts them in its header, with the
link as a QR code. Every field is optional: with no author set the byline
stays "Frontier Insight", and a field left empty is simply not printed.
The interview does not ask for them up front: they are one folded line,
*Paper byline (optional)*, on the review screen's "What you get" card, and
the first time a paper is written from a terminal FI asks the name once
(Enter keeps the Frontier Insight byline; with no answer in two minutes the
paper goes on without one and you are asked next time; the web page and
VS Code never wait on it). What you give is kept in
`~/.frontier-insight/profile.json` (`FI_PROFILE_PATH` moves it), which all
three interfaces read, so later interviews fill it in without asking and
show it on the review screen, where a change is kept for the next quests
too.

These values are written only into the quest's own files and your profile file on this machine (the web page
reads and keeps it only for a page opened on this machine; a request that came through a proxy
saying so is refused, but a plain port forward to this machine looks local, so do not expose
`--serve` that way). They are not
sent to the literature or web search services. If the visual check of the
outputs is on, the page screenshots it sends to your configured LLM
provider show the author line, as they show the rest of the paper.

`output.poster_size` picks the poster sheet: `a1_portrait` (59.4 × 84.1 cm,
two columns, the default), `a0_portrait` (84.1 × 118.9 cm, two columns) or
`landscape_48x36` (48 × 36 in, three columns). It is an advanced
interview question.

Changing any of these on a finished quest takes effect when the output is
rendered again: `python launch.py --config <yaml> --resume <id> --emit poster`
(or `paper_pdf`, `slides`).

### The poster

`poster.pdf` follows published poster guidance rather than squeezing the
paper onto a page.

- **Header:** the paper's main finding is the headline. The paper's title
  goes under it, then the author line. A QR code to `output.url` sits on
  the right when a link is set.
- **Type:**
  - body text is 26 pt on A1 and 36 pt on A0 and 48 × 36 in;
  - headings are about 1.5 times the body, and the headline 80 pt or more;
  - figure captions are numbered, and lines run about 60 characters.

  Type is never shrunk to fit.
- **Content:** the model writes headings, short texts, bullet lists and
  figures with captions to a word budget for the sheet. The generator
  writes all the LaTeX, so a slip in the model's formatting cannot break
  the compile.
- **References:** only the sources the poster cites, at most 8, each as
  author, year, title, venue and DOI. A web page shows its site name,
  never a raw URL. A poster that cites nothing lists five selected sources.
- **Fitting:** FI plans the columns from estimated block heights, compiles
  the poster, and measures the PDF. Each measurement corrects the plan.
  The header and reference band are only known after the first compile, so
  that first plan never cuts content. When a compile measures more room
  than the plan assumed, FI plans again for the measured room. When
  content runs off the sheet or into the reference band, FI cuts in this
  order, stopping as soon as it fits:
  1. Figures narrow, down to 70% width.
  2. The longest list loses its last items.
  3. The longest text loses its last sentences.
  4. Text blocks, and then figures, are dropped from the middle.

  The opening and closing blocks always stay. Short columns are carried
  down with extra space before their headings. FI stops after at most six
  compiles.
- **Report:** `.fi/poster_fit.json` in the quest folder records the sheet,
  the number of compiles, the figure widths, what was cut, the final
  measurements, and any findings still open (for example, columns that end
  a few centimetres apart). `.fi/poster_reply.txt` keeps the model's reply.
- **Chinese, Japanese or Korean** text on the poster compiles with XeLaTeX
  and a CJK font, as for the paper. Without either, the poster is skipped
  with a `cjk_no_xelatex` or `cjk_no_font` diagnostic.

### The visual check

After the outputs render, FI checks each PDF the pass produced: the paper,
the slides and the poster. With [LibreOffice](INSTALL.md#system-tools-optional)
installed, `slides.pptx` is exported to PDF and checked too. LibreOffice shows
the deck's equations as their readable text form, so that is what the check
sees; PowerPoint shows them as native equations.

- **What runs:** the PDF is measured (font sizes, overflow, columns, and a
  paper page left half empty before the paper ends) and screenshotted into
  the report folder. No model is asked by default.
- **Figures on slides:** a figure's tick labels must be at least 8 pt on its
  slide. A slide shows a figure at a fraction of the width it was drawn at, so
  the check takes the tick size the figure was drawn at (from the figure's
  record in `.fi/figure_records/`, the house style's size for an older
  record) times that fraction. A figure that shares its slide with text and
  comes out under 8 pt is a finding the slides are redone for: the model is told to
  give the figure a slide of its own, with no bullets, and to put its
  discussion on the next slide. A figure that already has its slide and is still
  under 8 pt (too many panels for one slide) is reported, not redone. Figures
  without a record (fetched web figures, and every figure under
  `execution.sandbox: docker`) are not measured. The redo is the usual one, at
  most `visual_check_max_redos` times.
- **AI check (`visual_check_ai: true`):** the screenshots, the measurements
  and a fixed checklist also go to your configured provider in one call. The
  checklist asks only what a script cannot see, such as raw LaTeX showing as
  text or a figure that covers a caption. It is off by default because it
  costs about 12,000 tokens a quest.
- **Privacy:** with the AI check on, the screenshots, and so everything
  printed on the pages (including the author line), go to that provider.
- **Grounded findings:** each finding must quote text visible where the
  problem is. A finding that cannot be placed on its page is dropped; the
  report keeps it with the reason.
- **Redo:** when the check finds problems a new version can fix, the slides
  or the poster are generated again with those problems in the prompt. The
  version that checks best is kept. A redo is skipped for poster layout
  findings (empty space, uneven columns), which a new reply would not change.
  The paper is never rewritten: when its last page holds only a line or two,
  it is recompiled once with a text area one line taller. `slides.pptx` is
  checked once, after the slides have settled: it comes from the same
  `slides.md`, so a slides redo already made a new one.
- **Report:** `.fi/visual_check.json` in the quest folder, with the
  screenshots under `.fi/visual_check/<output>/`. The run prints one line
  per output, for example
  `[FI] visual check slides: 0 problem(s) seen on the pages, 1 measured; redone 1 time(s), kept redo 1`,
  or `[FI] visual check slides.pptx: not checked (LibreOffice was not found)`.
  The web quest page shows the same lines.
- **Providers without image input** (see
  [PROVIDERS.md](PROVIDERS.md#which-providers-can-see-images)) check from the
  measurements alone, and the line says so.

```yaml
output:
  visual_check: true          # false turns the check off
  visual_check_ai: false      # true also asks your provider to look at the screenshots
  visual_check_max_redos: 2   # 0 to 2 new versions of the slides or poster
```

### `execution.sandbox: docker` — what it actually does

When you set `sandbox: docker`, FI runs the generated experiment inside a Docker container instead of a fresh Python venv. The defaults:

- **Image**: `python:3.11-slim` (override via `execution.docker_image`).
- **Network**: disabled (`--network none`) — the experiment can't reach the internet, which prevents accidental literature scraping or data exfiltration from generated code.
- **Mount**: the quest output directory is bind-mounted at `/work` inside the container; the experiment's working directory is `/work`. Code reads/writes there.
- **Skills from other agents**: each external skill (found in `~/.claude/skills`, `~/.codex/skills`, ...) that you approved and the quest selected is bind-mounted **read-only** at `/fi-skills/<name>`, and the prompts give that path instead of the host one. Nothing else of yours is mounted; a skill folder that is a symbolic link or leads outside its skills folder is refused and `run.log` says so.
- **Lifetime**: a fresh container per execute step. State doesn't persist between retries — the execute-repair loop sees a clean environment each iteration.
- **Limits**: each experiment may use at most 4 GB of memory (`execution.docker_memory_gb`), 2 CPUs (`execution.docker_cpus`) and 1024 processes and threads at once (`execution.docker_max_processes`). An experiment that goes over the memory limit is stopped; one that tries to start more processes or threads than the limit cannot start them, and usually fails. Either way `run.log` and the error the repair step reads say which limit and which setting raises it, for example *the experiment used more than the 4 GB memory limit and was stopped; raise execution.docker_memory_gb (now 4) to give it more*. Over the CPU limit an experiment only runs slower; numerical libraries are told to start that many threads (`OMP_NUM_THREADS` and the like, unless already set to no more than that). A CPU limit above what Docker has is lowered to what it has, and a limit this Docker cannot apply is left out; `run.log` says which.
- **Environment**: the experiment gets FI's environment variables except the host's system and Python paths (`PATH`, `HOME`, `PYTHONHOME`, `LD_LIBRARY_PATH` and the like), which mean nothing in the container or break its Python; a path in the quest folder is given as its `/work/...` path.
- **No extra privileges, not root**: the container has every Linux capability dropped and cannot gain privileges. It runs as an ordinary user: on Linux, the owner of the quest folder (normally you), so what the experiment writes stays yours; under Docker Desktop on Windows and macOS, a fixed non-root user, since Docker Desktop's file sharing writes the files as you anyway. Before the first run FI checks that this user can write the quest folder; when it cannot (Docker Desktop for Linux, which maps you to the container's root), experiments run as the container's root user, and `run.log` says so. Under rootless Docker the container's root user already is you and nothing more, so experiments run as that. A non-root experiment gets `HOME=/tmp`. A quest started before experiments ran as you can hold files or folders root owns; `run.log` names them and the `chown` command that gives them back.

Requires the `docker` Python package (`pip install docker`) and a running Docker daemon. On Windows that means Docker Desktop or WSL2. If you don't have those, leave the default `sandbox: venv` — the per-quest venv is faster anyway.

### `output.paper_format` — which templates ship fully styled

**Scientific (IMRAD):**

| Format | Status |
|---|---|
| `generic` | ✅ Fully styled — IMRAD with default LaTeX article geometry. The default; pick this when you don't have a target venue. |
| `neurips` | ✅ Fully styled — uses the NeurIPS 2024 style sheet. |
| `iclr`, `ieee_access`, `nature_mi` | ⚠️ Minimal stubs — they compile, but the style sheets are placeholders. Treat as starting points; copy the real venue's `.sty` file into `templates/paper/<format>/` to customize. |

**Non-scientific (prose, IMRAD-free):**

| Format | Status |
|---|---|
| `essay` | ✅ Long-form argumentative prose. Wider margins, 1.5× line spacing, serif body. Picks the **essayist** write-persona — opens with thesis, marshals evidence, closes with implications. No IMRAD headings. |
| `report` | ✅ Consulting executive report with cover page + TOC. Sans-serif body. Picks the **senior consulting analyst** persona — exec summary → findings → recommendations. |
| `policy_brief` | ✅ 2-4 page brief for policymakers. Tight margins, header strip, dense layout. Picks the **policy analyst** persona — issue → context → single recommendation. |
| `whitepaper` | ✅ 8-20 page industry analysis. Cover with whitepaper subtitle, modest TOC, sans-serif. Picks the **industry analyst** persona — problem → approach → evidence → conclusions. |

The `clarify` node's `paper_venue` slot accepts both buckets. For
non-simulatable topics (set via the new `simulatability` clarify
slot or legacy `empirical_vs_theoretical: empirical`), the agent
will default to `essay` instead of `generic`. The write-persona
swap is automatic — set `paper_format: policy_brief` in YAML and
the `write` node loads the policy-analyst voice.

### `--fleet` concurrency model

`--fleet` runs each YAML in its own asyncio task. The cap is `--max-concurrent N` (defaults to `min(4, cpu_count)`). When RSS exceeds `--memory-cap-mb`, new quest starts pause until memory drops below the cap. Each quest has its own venv, its own `state.sqlite` checkpoint, and its own provider proxy — failure in one quest doesn't affect the others. Provider proxies are reference-counted across the fleet, so 4 concurrent quests all using `vscode_extension` share one bridge connection rather than spawning four.

## Output artifacts

After each quest, `<output_dir>/<quest_id>/` looks like:

```
paper/paper.md                        ← the IMRAD paper
paper.pdf                             ← if pandoc + a LaTeX engine
slides.md / slides.pptx / slides.pdf  ← if `slides` is in output.kinds
poster.tex / poster.pdf               ← if `poster`
talk.md                               ← if `speech`
figures/*.png                         ← every plot the experiment produced
data/results/                         ← a copy of the data tables an accepted run wrote (and, with two scripts, raw/); a file, or raw/ as a whole, over 200 MB stays where it is
code/                                 ← the code that ran, runnable on its own: experiment.py (+ simulate.py), run.py, requirements.txt, README.md, CHANGELOG.md (each change marked Added / Changed / Fixed / Tidied, with a line saying whether the results changed) (+ its own git history)
run_output/                           ← only if you run `python code/run.py`: its own results, apart from the quest's
config.yaml                           ← copy of the YAML for /resume
.fi/run.log                           ← full run log
.fi/state.sqlite                      ← LangGraph checkpoint
frontier_insight_summary.json         ← machine-readable index
```

## Resuming a crashed quest

```bash
# From the terminal:
fi --config outputs/<quest_id>/config.yaml --resume <quest_id>
# or, from any folder, by its short id (see "Finding a quest from any folder"):
fi --resume 479b06

# From VSCode chat:
@fi /resume
# pick from the list
```

The engine reads the `state.sqlite` checkpoint, detects what node
last completed, and continues from there. No state is lost — the
`paper.pdf` engine, the YAML's `provider` block, every clarify answer
flows through as if the original run never crashed.

## Topics that need real data (not simulation)

Some research questions can't be answered with a Python script —
*"Compare Belgium and Taiwan culture: collectivism, work-life
balance, public-trust dynamics"* needs real surveys and observations,
not invented numbers.

**How the engine decides to enter no-simulation mode** (in this
precedence — first match wins, decision is logged to `run.log` as
`[clarify] simulatability resolved: ... source=<...>`):

1. `engine.no_simulation: true` in YAML — explicit user override.
   `source=yaml`.
2. The clarify question `simulatability` — the agent asks
   *"can a Python script meaningfully simulate this, or does it
   need real-world data?"* with `default: yes | no | uncertain`
   plus a one-line `reason`. `no` triggers no-simulation
   (`source=clarify_simulatability`); `yes`/`uncertain` keeps the
   simulation path. With `pauses.clarify: ask` you see the
   agent's default + reason and can override; with `pauses.clarify:
   auto` the default is accepted but the reason is still logged.
   An answer in your own words counts by its first word: *"yes, but
   more than 100 lines"* is `yes` (*"no idea"* is not a `no`).
3. Legacy fallback: when that answer is missing, blank, or does not
   start with yes / no / uncertain, an `empirical_vs_theoretical:
   empirical` answer still triggers no-simulation
   (`source=clarify_empirical_legacy`).
4. Otherwise `simulatability` in `engine.clarify_overrides` (what the
   setup questions wrote into your YAML), also when clarify is off
   (`source=yaml_clarify_overrides`).
5. Otherwise: simulate (`source=default`).

### Survey mode (a history / overview — no experiment AND no dataset)

Some topics are neither a simulation nor a data-analysis: *"the
evolution of X"*, *"a history of Y"*, *"an overview of Z"*. These want
a **descriptive synthesis of the published literature** — there is
nothing to measure and no dataset to collect. That is **survey mode**,
a stronger form of no-simulation. Turn it on any of three ways:

* `engine.survey_mode: true` in YAML — explicit override (also forces
  `no_simulation: true`).
* Pick **"Literature synthesis (no experiment)"** as the research
  approach in the interview (CLI / web / VSCode).
* Automatically — the clarify step classifies the topic shape as
  `survey` (triggers: "history of", "evolution of", "overview of", …).

In survey mode the graph skips BOTH the experiment (`implement` /
`execute`) AND the whole data path (`auto_collect_data` →
`wait_for_data` → `data_load` → `web_plots`), routing `design →
web_figures → analyze → write`. `web_figures` still embeds
license-clean illustrative images; `analyze` synthesises the reviewed
sources directly; the paper is a narrative history with a References
section and no fabricated metrics.

For the no-simulation (data-analysis) path — where there IS a dataset
to analyse — the engine runs `clarify → ideate → literature → design →
auto_collect_data → wait_for_data → data_load → analyze → ...`. The
two no-simulation-specific stops:

**Confirming a data analysis** (`engine.phased: true`, on by default for research): when you
supply one table (a CSV or TSV of at least 40 rows in `data/`), part of it is held back before
`data_load` first reads it, and when exploration ends the engine reads the held-back rows once more
with the frozen design; only that reading can be publication-ready, and where its numbers differ from
exploration's the paper says so. FI must know which rows belong together (several rows of one
subject, site or device stay on one side): it reads the plan's `protocol.split`, then the line
`Rows that belong together:` in plan.md (a column name, or `independent`), then the table's column
names; a research quest that still cannot tell stops once and asks (answer on that line in plan.md,
then resume). Data that cannot be split (too few rows, only gathered pages) is said in one sentence
in `run.log` and `plan.md`. The held-back rows confirm one version of the design only: a design
changed after that reading is a new version, which cannot be confirmed on rows already read, and
says so. See docs/rigor.md, *Explore, then confirm*.

**`auto_collect_data`** — agent-side data collection. Before
pausing for user input, the engine asks the Knowledge layer (Axon)
for relevant docs using `topic + design.hypothesis` as the query
and writes the top hits into `<quest_root>/data/auto_collected/`
as one Markdown file per doc (with YAML provenance front matter so
the paper can cite back to specific sources). Controlled by:

* `engine.auto_collect_data: true` (default) — try Axon first.
  Set to `false` for "user-only" data flow when you don't trust
  the corpus or want a manual pause every time.
* `engine.auto_collect_top_k: 5` (default) — Axon hits requested.
  5 fits comfortably in a 16k-context data_load prompt; raise
  only on long-context models with topics that genuinely need
  more breadth.

**Dataset adapters**: in addition to the corpus-RAG retrieval,
`auto_collect_data` can invoke structured-data adapters that hit
public APIs and write tabular evidence into
`<quest_root>/data/auto_collected/<adapter>/`. Opt in via:

* `engine.dataset_adapters: [worldbank, wikipedia]` — list of
  registered adapter names. Empty (default) means "Axon only —
  no adapters run". Available adapters: `worldbank`, `wikipedia`.
  Unknown names log a WARNING and are skipped (no hard error on typo).
* `engine.dataset_adapter_top_k: 3` — rows per adapter. Smaller
  default than the Axon knob because each row hits an external API.

The WorldBank adapter heuristically extracts country names from the
query (`"Belgium and Taiwan"` → `[BEL, TWN]`, falling back to
global aggregates `WLD/OED/EUU/HIC/LIC` when no country named),
scores ~1500 indicator names against the query keywords, and writes
the top `top_k` matches as Markdown tables with the last 5 years of
data. Adapter failures (network down, indicator not found, every
write failing) fall through silently to the safety net.

The Wikipedia adapter handles the long-tail "qualitative comparison"
case where neither corpus-RAG nor structured-data fits — e.g.
*"compare the 1968 student protests in Paris and Mexico City"*. It
attempts to compress the query to its top-6 informative keywords
(falls back to the raw trimmed query if every token was filtered
out as a stop-word), calls `api.php?action=opensearch` to get
candidate article titles, then fetches
`api/rest_v1/page/summary/<title>` for each match and writes a
Markdown file per article. The page's `description` and `extract`
land in the document body; YAML front matter carries `source:
wikipedia`, the canonical `title` / `url`, the article's
`wikipedia_type` (e.g. `standard`, `disambiguation`), and
`adapter: wikipedia` for downstream provenance. Articles with
extracts under ~200 characters are dropped as too thin to cite.
Request budget per quest: `1 + dataset_adapter_top_k` HTTP calls.

Auto-collect falls through to the user-data pause (no files
written, `auto_collected_count: 0` in state) in four cases:

* `engine.auto_collect_data: false` — INFO log, **no Axon call**.
* `knowledge.enabled: false` — INFO log, **no Axon call**.
* `Knowledge.asearch` raised — WARNING log; Axon **was called** but
  the exception is caught so the quest survives.
* Axon returned zero hits — INFO log; Axon was called and answered
  legitimately with nothing.

In all four cases `wait_for_data` then makes the final pause-or-
proceed decision based on `data/` contents (manual drops still
count if you pre-staged some).

**`wait_for_data`** — the user-data pause. With files already in
`data/` (either auto-collected or user-supplied), this node
passes through immediately. If the dir is empty (Axon returned
nothing AND no manual drops), the engine exits cleanly (rc=0)
with an instruction file at `outputs/<quest_id>/data/README.md`
telling you what to drop. Then:

```bash
fi --resume <quest_id>
```

The engine picks up at the `data_load` node: walks every file in
`data/` (including `auto_collected/`), classifies them (csv / json
/ pdf / md / xlsx / png), synthesizes a `result_json` via one LLM
call grounded in the designed measurement plan, and then continues
normally through `analyze → cross_check → write → review`. The
paper cites the *specific files dropped or auto-collected* as
primary sources, not invented data.

Permissive about format — drop whatever's natural. The walker
deduplicates and budget-caps the prompt the same way `/summarize`
does (see `core/summarizer.py`). If a file format isn't text-readable
(images, binaries), the engine lists it in the manifest but doesn't
include its contents in the prompt — caption it in an accompanying
`.md` for the model to see.

## Common workflows

### Cheap-then-expensive

Light model for the early nodes; strong model only for write/review:

```yaml
provider:
  name: vscode_extension
  model: gpt-4o-mini
  node_models:
    write:  claude-3-5-sonnet
    review: gpt-5
```

### Let a local model think

Ollama models answer without reasoning unless the request asks for it:

```yaml
provider:
  name: ollama
  model: gemma4:31b-cloud
  reasoning_effort: high          # Ollama takes low | medium | high
```

### Setup questions

Before it starts, a quest asks about ten setup questions (the `clarify` step): what you most want to see at the end,
what the study should be called (three suggested titles; a `title:` in the YAML, which the interview always writes,
skips this one), the baseline to compare against, the success metric, the budget, the outputs, the depth, the venue,
and whether a simulation can answer the topic. Each comes with the agent's own suggested answer; Enter keeps it.

```yaml
pauses:
  clarify: ask        # always stop for the answers; auto: the agent answers itself; off: skip the step
                      # left out (the default): ask when someone can answer while it runs, else answer itself
```

Who can answer, by interface:

| Interface | How the questions reach you | Nobody answers |
|---|---|---|
| Terminal | `fi --config my.yaml --interactive` (also `fi --new --interactive`) prompts on stdin. Without `--interactive` nobody can answer, so the agent answers itself, unless the YAML says `ask`. | with `ask`, the quest stops; the questions are in `<quest>/.fi/clarify_questions.json`; write a JSON object of question key → answer, one per question, to `<quest>/.fi/clarify_answer.json` and `fi --resume <id>`, or resume with `--interactive` to be asked |
| Web (`fi --serve`) | a form on the quest page, marked *Waiting for your answers*, for a quest the page starts or resumes | the defaults after `pauses.timeout_s`; with `ask` the quest stops, and answering on the page resumes it |
| VS Code | one input box per question, for a quest on the VS Code chat model (`@fi /new`, or `@fi /start` / `@fi /resume` of a YAML with `provider.name: vscode_extension`). Esc on any box carries on with the defaults | the defaults after `pauses.timeout_s`; with `ask` the quest stops, and `@fi /resume <id>` asks again |

A YAML that names its own provider (for example Kimi through `@fi /start`) is not on the bridge, so VS Code cannot
show the boxes: it answers itself, or, with `ask`, stops for the answer file above. `pauses.timeout_s` defaults to
1800 s; `0` waits with no limit. A web quest waiting for its answers counts as a running quest against the server's
`--max-concurrent`. A `--fleet` run never asks.

### Fleet of variations

Stamp out N YAMLs each varying one knob (model, panel, depth), then:

```bash
fi --fleet variants/*.yaml --max-concurrent 4
```
