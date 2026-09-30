# Glossary

The words FI uses, in the order you meet them. Each is one plain sentence, with where to read more.

**Quest**: one run. A topic goes in, a folder (`outputs/<quest id>/`) comes out.

**Provider**: which language model answers FI's questions (OpenAI, Gemini, a signed-in CLI, VSCode Copilot, a local model). FI never picks the model for you. [PROVIDERS.md](PROVIDERS.md)

**Stage / node**: one step of a quest (`literature`, `design`, `execute`, `review`, ...). [architecture.md](architecture.md)

**Plan (`plan.md`)**: the document FI writes before it experiments: what the literature says, the gap, the model behind the numbers (for a quest that runs code: what model produces them and its equations, each with its source; how the code will be judged better or worse; and whether it is a measurement or a search for the best design), and the design it will run. You can stop there, edit it, and continue. [USAGE.md](USAGE.md)

**Model behind the numbers**: the section of `plan.md` that says what model produces a simulation's numbers (in a sentence, what it assumes, where it holds) and its core equations (E1, E2...), each with the source it comes from, next to each oracle and where its expected value comes from. It is part of the protocol, so it is frozen with it. [rigor.md](rigor.md)

**Study type**: what kind of study the plan is: a *measurement* (`measure`: how a result changes over settings chosen in advance) or a *search for the best design* (`find_best_design`: the design that makes one result as low or as high as possible within limits, compared with a starting design). This version plans a search but does not run it. [rigor.md](rigor.md)

**Interview**: the questions FI asks before a quest exists (`fi --new`, the web form, `@fi /new`); your answers become the quest's config file. [USAGE.md](USAGE.md)

**Setup questions**: the questions a quest asks you when it starts (what you want to see, the title, the baseline, ...), each with the agent's own suggested answer; the config key is `pauses.clarify`. Left unset, the quest asks when someone can answer (the terminal with `--interactive`, the web quest page, VS Code for a quest on the VS Code chat model) and answers itself when nobody can. [USAGE.md](USAGE.md#setup-questions)

**Pause**: FI stops because it needs you: to answer the setup questions, to download paywalled papers, to read the plan, or to accept, reject or refine the result. It is not a failure: `NEXT_STEP.md` in the quest folder says what to do, and `fi --resume <quest id>` continues (the **Resume** button on the web quest page, `@fi /resume <quest id>` in VS Code). A review answer that decides nothing (a closed prompt, a refine without notes) is never taken as accept: the quest stays paused (an empty refine confirmed with Enter at the terminal or VS Code prompt still means accept, as the prompt says). Some checks also stop the quest the same way, with `NEXT_STEP.md` saying why: a check with a known answer that fails before the main run, a plan that is a search for the best design, a change to the frozen protocol waiting for approval. [recipes.md](recipes.md)

**Refine**: your notes on a finished draft, sent at the review pause; FI answers each point as cheaply as it can (rewrite the text, add a missing number to the existing script, redraw the figures from the saved numbers, or, only for a different study, go back to the design). [USAGE.md](USAGE.md#doing-a-step-again)

**Doing a step again (rerun from a step)**: `fi --resume <quest id> --from <step>` runs a quest again from one of the steps it reached, keeping everything decided before it and moving what that step and the later ones made to `.fi/previous/<time>/`. [USAGE.md](USAGE.md#doing-a-step-again)

**Quest map**: a picture of every step of a quest, each marked finished, where it stopped or not reached, from which you can run the quest again from a step (web quest page, VS Code **FI: Quest map**). [USAGE.md](USAGE.md#doing-a-step-again)

**Protocol**: the experiment's rules, written down before it runs: which settings, how many trials each, how randomness is seeded, how uncertainty is measured, what counts as success. It lives in `plan.md`. [rigor.md](rigor.md)

**Frozen protocol**: the protocol, locked right before the first full run (with explore, then confirm, when exploration ends) and recorded with a hash. What runs is checked against it, not against a later rewrite. [rigor.md](rigor.md)

**Amendment**: a change to the frozen protocol. It always needs a person's explicit approval (`fi --approve-amendment <quest>`); one made after results were seen is recorded as post-hoc. [rigor.md](rigor.md)

**Explore, then confirm**: with `engine.phased: true`, the model first explores (tries designs, looks at results); then the design is frozen and run once more on data held back from exploration, or on new random seeds, and only that confirm run can be publication-ready. [rigor.md](rigor.md)

**Oracle**: a check with a known right answer (a closed form, a limiting case). The plan states the number it must give, how close, and where that number comes from (a derivation written out, a source the quest found, an equation of the model behind the numbers, or a second implementation that shares no code with the simulation); the script only reports what it measured, and FI's engine decides pass or fail. Only a value FI measured by running the simulation itself counts as independent evidence. When a fix says the check itself is wrong, it can only propose new numbers: you decide. [rigor.md](rigor.md)

**Criteria (how we will judge whether the code got better)**: two to five checks of correctness, fixed in the protocol, that FI computes itself after every run (for example how far a check with a known answer lands from its expected value), never the study's own finding; each run adds a row to `.fi/criteria_history.jsonl`. [rigor.md](rigor.md)

**Run manifest**: what the simulation says it actually ran (settings, trials, seeds). FI compares it with the frozen protocol. [rigor.md](rigor.md)

**Metric spec**: for each headline number, what it estimates (a proportion, a mean, a difference), so the matching interval and test are used. [rigor.md](rigor.md)

**Two scripts**: the simulation (`simulate.py`, which only says what one setting, or one trial of it, computes; FI runs it and keeps the record) kept apart from the analysis (`experiment.py`, which reads FI's results), so the analysis can be redone without re-simulating and FI can run each oracle's case itself. Set with `execution.split_analysis`; with the default `auto` every quest that runs a simulation, deterministic or random, gets two scripts (`split_analysis: false` keeps one, and its result cannot reach `independently_validated`). [USAGE.md](USAGE.md)

**Equation label**: a comment such as `# E1` in the simulation, on the code that computes equation E1 of the plan's model, so a wrong number can be traced to the equation or to the code. FI checks that each equation the simulation computes the data with has one. [rigor.md](rigor.md)

**Evidence level**: how far a result was checked, from `executed` (it ran) through `internally_reconciled`, `protocol_runtime_matched`, `independently_validated` and `statistically_adequate` to `publication_ready`. Each level says what it guarantees and what it does not. [rigor.md](rigor.md)

**Rigor profile**: `rigor_profile: research` turns on together what a study needs before its result can be trusted, and makes those checks stop the quest instead of only reporting. The interview recommends it for a simulation study. [rigor.md](rigor.md)

**Trace**: the quest's ordered, tamper-evident diary (`.fi/audit.jsonl`): each step, each check, each route, and what the model said about why. Read it with `fi --trace <quest id>` (`--follow` to watch it live); `fi --why <quest id>` answers why it stopped, why it was sent back, why the evidence is at its level. [trace.md](trace.md)

**Model claim**: the reasons a model states in its answer (why this design, why this verdict), kept in the trace so you can argue with it. It is never a check result. [trace.md](trace.md)

**Model reasoning (`.fi/thinking.jsonl`)**: the reasoning text some connections return beside the answer (a reasoning model over HTTP, Claude's thinking, VS Code's thinking parts), one line per call, kept for you to read. It is not part of the trace, never checked, and can quote your data. [trace.md](trace.md)

**Skill**: what FI has learned about driving one piece of software, with a test that proves it still works. You approve each one. [USAGE.md](USAGE.md)

**Axon**: the knowledge layer FI uses to search literature and to remember earlier quests. It starts by itself; you do not need to run it. [INSTALL.md](INSTALL.md)
