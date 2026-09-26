# Glossary

The words FI uses, in the order you meet them. Each is one plain sentence, with where to read more.

**Quest**: one run. A topic goes in, a folder (`outputs/<quest id>/`) comes out.

**Provider**: which language model answers FI's questions (OpenAI, Gemini, a signed-in CLI, VSCode Copilot, a local model). FI never picks the model for you. [PROVIDERS.md](PROVIDERS.md)

**Stage / node**: one step of a quest (`literature`, `design`, `execute`, `review`, ...). [architecture.md](architecture.md)

**Plan (`plan.md`)**: the document FI writes before it experiments: what the literature says, the gap, and the design it will run. You can stop there, edit it, and continue. [USAGE.md](USAGE.md)

**Pause**: FI stops because it needs you (paywalled papers to download, the plan to read, the result to accept). It is not a failure: `NEXT_STEP.md` in the quest folder says what to do, and `fi --resume <quest id>` continues.

**Protocol**: the experiment's rules, written down before it runs: which settings, how many trials each, how randomness is seeded, how uncertainty is measured, what counts as success. It lives in `plan.md`. [rigor.md](rigor.md)

**Frozen protocol**: the protocol, locked right before the first full run and recorded with a hash. What runs is checked against it, not against a later rewrite. [rigor.md](rigor.md)

**Amendment**: a change to the frozen protocol. It always needs a person's explicit approval (`fi --approve-amendment <quest>`); one made after results were seen is recorded as post-hoc. [rigor.md](rigor.md)

**Oracle**: a check with a known right answer (a closed form, a limiting case). The plan states the number it must give and how close; the script only reports what it measured, and FI's engine decides pass or fail. [rigor.md](rigor.md)

**Run manifest**: what the simulation says it actually ran (settings, trials, seeds). FI compares it with the frozen protocol. [rigor.md](rigor.md)

**Metric spec**: for each headline number, what it estimates (a proportion, a mean, a difference), so the matching interval and test are used. [rigor.md](rigor.md)

**Two scripts**: the simulation (`simulate.py`, which only says what one trial does; FI runs every trial and keeps the record) kept apart from the analysis (`experiment.py`, which reads FI's results), so the analysis can be redone without re-simulating. Set with `execution.split_analysis`. [USAGE.md](USAGE.md)

**Evidence level**: how far a result was checked, from `executed` (it ran) through `internally_reconciled`, `protocol_runtime_matched`, `independently_validated` and `statistically_adequate` to `publication_ready`. Each level says what it guarantees and what it does not. [rigor.md](rigor.md)

**Rigor profile**: `rigor_profile: research` turns on together what a study needs before its result can be trusted, and makes those checks stop the quest instead of only reporting. The setup questions recommend it for a simulation study. [rigor.md](rigor.md)

**Trace**: the quest's ordered, tamper-evident diary (`.fi/audit.jsonl`): each step, each check, each route, and what the model said about why. Read it with `fi --trace <quest id>`. [trace.md](trace.md)

**Model claim**: the model's own account of its reasoning, kept in the trace so you can argue with it. It is never a check result. [trace.md](trace.md)

**Skill**: what FI has learned about driving one piece of software, with a test that proves it still works. You approve each one. [USAGE.md](USAGE.md)

**Axon**: the knowledge layer FI uses to search literature and to remember earlier quests. It starts by itself; you do not need to run it. [INSTALL.md](INSTALL.md)
