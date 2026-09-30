# The trace: what a quest did, and why

Every quest keeps a diary of what it did, in order: `<quest folder>/.fi/audit.jsonl`. It answers three questions after the fact:

- **What ran, and what did each check say?** Every step, every check's verdict (protocol, oracle, run manifest, numeric warnings, evidence, methodology audit), and the files the quest wrote with their sha256: the scripts, the plan, the frozen protocol and the paper, and the records the evidence level is read from (the approved settings, the three checks' receipts, the environment, the run-manifest, oracle and protocol records, FI's trial ledger, summary and run record, the claim ledger), so a change to one after it was written shows in the trace.
- **Why did it go this way?** Each time the quest chose its next step (redo the design, write, stop for you) it writes down the facts it looked at.
- **What did the model say about why?** Every reason a model already writes in its answer at a step that chooses: the idea it picked (and what the pairwise comparison and its self-critique said), the skills it chose or declined, the assumptions and the options it weighed in the design, how each finding sits against the literature, what a repair of the script changed and why (or why it gave up), its reasons for the setup answers it chose, the reviewer's verdict with its one-line reason and objections, a review panel's moderator, the evidence gate's verdict and analyze's next step. Two route-changing answers are also asked for one short reason: the review's verdict (with a review panel, the moderator's reason is what is recorded), and how the writer answers your refine notes (only the text, more data, a new figure layout or a new experiment). These are marked **model claim**: the model's own account, kept so you can argue with it. They are never mixed with a check result and no check reads them.

## Look at it

```bash
python launch.py --trace <quest id or folder>                  # the timeline, and a check that nothing was edited
python launch.py --trace <quest> --trace-node design           # only one step
python launch.py --trace <quest> --trace-detail summary        # just the shape: steps, stops, routes (or: checks, debug)
```

On the web, open a quest and expand **Trace** (same detail and step filters). The list reads the same as the command. In VSCode, `@fi /trace <quest_id> [--node <name>] [--detail summary|checks|debug]` prints the same lines in chat.

## Watch it live

```bash
python launch.py --trace <quest> --follow                      # each new event as it happens, until the quest stops for you, finishes or fails
```

In VSCode, `@fi /follow <quest_id>` shows each step in chat as it happens (`--detail checks` for more); stopping the chat stops following, and the quest goes on. A quest whose process is killed outright (out of memory, the machine off) leaves no record to end on: stop following with Ctrl+C (or by stopping the chat). The web quest page already shows the live log.

## Ask why

```bash
python launch.py --why <quest>                                 # why it stopped, why the review asked for a revision, why the evidence is at its level
python launch.py --why <quest> execute                         # why one step decided what it did
python launch.py --why <quest> reasons                         # the reasons the model gave at every step
```

The answer is put together from what the quest recorded (the stop, the review's route and the facts it read, each check's verdict, the evidence record); no model is asked. The model's own reasons are shown under their own heading, since they are what it said, not something FI checked. For a step, the answer also says whether `.fi/thinking.jsonl` holds the model's reasoning for that run of the step and how long it is (the text stays in the file). **A reason the model wrote, and a reasoning summary it returned, are its own account of itself: not its hidden reasoning, and not something FI checked.** A written reason can be a justification made after the fact. On the web, **Why?** in the Trace section answers for the step chosen there (or overall) and **Model's reasons** lists every step's; in VSCode, `@fi /why <quest_id> [stop|review|evidence|reasons|<step>]`.

## Keep every prompt and answer

The trace records every decision but not the whole text a model was sent. To keep that too, set `output.save_model_calls: true`: each call's whole prompt and answer (the output generators' calls included) goes to `<quest folder>/.fi/io/`, one file per call, with anything that looks like a key redacted and a page image kept as its size. It is off by default, because a quest makes 30–90 calls and the files can reach tens of megabytes.

## What the model said about its own reasoning

When the connection hands it back (a reasoning model's `reasoning_content` through an OpenAI-compatible server such as Kimi or DeepSeek, Claude's thinking through `claude_cli`, the reasoning summaries `codex_cli` prints, the thinking parts of a VS Code chat model), FI keeps it in `<quest folder>/.fi/thinking.jsonl`, one line per model call: the step (`node`), the call (`call_id`, the same id as that call's line in `.fi/model_calls.jsonl`), who answered (`provider`, `model`), the model asked for, whether the call succeeded (`outcome`; a failed call's reasoning is kept too), the text, and a note that it is the model's own account and not evidence. Credentials and your home folder are removed; a line is cut at 64,000 characters (it says how many were left out) and the file stops growing at 32 MB. **It can quote your data and your prompts, since a model often repeats its input while it reasons; only credentials and your home folder are removed, so check it before you share the quest folder.** It is for you to read: it is not part of the trace, is not checked or sealed, and no verdict, route or evidence level ever uses it. When a very long answer leaves the VS Code bridge's message limit with little or no room, the reasoning is cut short or, if nothing fits, left out and that call has no line; when the answer fills the limit, the extension sends a one-line note (how many characters were left out) instead; an answer longer than that sends neither. A connection that returns no reasoning (most do not) leaves no line, and `run.log` says so once per step ("this model/connection returned no reasoning for this step"). It is on by default; set `output.save_thinking: false` to keep none (and, on the VS Code connection, not to ask for it). `--why` (and the web page's **Why?**, VS Code's `@fi /why`) says whether the file holds a step's reasoning and how many characters; the text itself is only in the file. The web quest page's file list and its zip download leave `.fi/` out, so the file is not in them.

**Asking Copilot for Claude's thinking.** On the VS Code connection, a Claude model returns its thinking only when the request asks for it, through Copilot's own model option `_enableThinking`. FI asks for it on every step's call by default. This option is undocumented and internal to Copilot, so a Copilot release may rename it or stop honouring it without notice; with it, Opus returns a **summary** of its thinking, not the full text. A model that refuses the option is asked again once without it, the quest goes on, and `run.log` says so once; it is not asked again for the rest of that chat command's run (for a quest resumed in a terminal, `@fi /update`, for the rest of the VS Code session). Calls FI makes outside a step (the paper, slide and poster generators) do not ask, since nothing keeps their reasoning. Set `output.save_thinking: false` to stop asking. **Codex.** `codex_cli` prints each reasoning summary Codex produced as a `reasoning` item of its `--json` output; FI keeps those. Whether a model returns a summary depends on the model and on Codex's own summary setting; this was read from the Codex binary's list of output types, not seen in a real call. `gemini_cli` and `copilot_cli` hand back no reasoning FI can read.

## How to read a line

| You see | It means |
| --- | --- |
| `design: done in 1.2s, wrote design` | a step finished and which parts of the quest state it changed |
| `check oracle: ok` | a check's verdict; the record with the detail and its hash are in the event |
| `review: next is revise because verdict=revise, ...` | the route the quest took and the facts it read to choose |
| `design: alternative: Wilson pooled -> rejected (assumes independence we can't verify here) [model claim]` | something the model said about its own reasoning, and — when it chose between options — what it decided and why |
| `evidence_gate: verdict: only 2 of 6 sources are on-topic -> broaden [model claim]` | the evidence gate's own stated reason for its verdict |
| `analyze: next_step: the effect is within noise at this sample size -> re_experiment [model claim]` | analyze's stated reason for the next step it picked |
| `select_skills: skill_chosen: the design mixes nm and um -> use units [model claim]` | the model's reason for a choice it made, from the answer it already gave |
| `the model changed from gpt-5.6-luna to claude-opus-5 (the model picked in the VS Code chat panel)` | the quest's model changed; results made before and after it came from different models |
| `waiting for you (plan)` / `stopped for you` | the quest paused for a person; after you resume, the same step starts again from its beginning |

`summary` shows steps, stops and routes; `checks` (the default) adds each check, each file written and the model's reasons; `debug` adds every step's start.

## What it can and cannot tell you

- Each event carries the hash of the one before it. Remove, reorder or edit a line and `--trace` (and the web page) says which event it broke at, and the command exits with 1. This finds accidents and partial edits; someone who can run the code can rewrite the whole file, so it is not a signature.
- The trace is a record, not a control. If it cannot be written the quest goes on and says so once in `run.log`.
- Nothing secret is written: values of environment variables that look like keys, key-shaped strings and your home folder are removed, and long text is cut with a note saying how much.
- It is not the model's private thinking. Only what the model chose to state as reasoning, and what the engine measured, are in it. The reasoning a connection returns is kept apart, in `.fi/thinking.jsonl` (see above).
- Each model claim also carries the `call_id` of its call, the model asked for (`requested_model`) and why the answer ended (`finish_reason`) when the engine recorded one, so a claim joins to its line in `.fi/model_calls.jsonl`. These come from what the engine measured, never from what the model said.
- A model claim also carries who actually said it (`provider`, `model` — the one that answered, even after a fallback took over) and a hash of the exact prompt and reply it came from (`prompt_hash`, `response_hash`), so a claim can be checked against the real call rather than taken on faith. These aren't in the one-line text (they'd repeat on every line and add noise); open the quest page's Trace panel or `.fi/audit.jsonl` directly to see them on an event. A rule-decided verdict (no model was asked) is never recorded as a model claim — only what a model actually said is.
- A finished quest writes one last event, the seal (`quest finished: N events recorded`): how many events came before it, how many could not be written over every run of the quest (the count is kept beside the trace, in `audit.jsonl.lost`, so a pause does not reset it), how many attempt records and model call records could not be written, which steps completed, and the hashes of the paper, the evidence record and every record file (what the quest tried, its choices, every model call it made, and the search queries it used). It is written after everything else the quest keeps, so nothing follows it. A chain that checks out only shows that the lines there were not changed, not that none are missing, so under `rigor_profile: research` a trace whose last event is not the seal, a seal that lost or miscounts events, a seal that leaves a record file out or names it without a hash, a record of model calls with fewer lines than the calls the quest made, or a file it names that changed since, is a gap; it is checked whenever the evidence is shown.
- Turn it off with `engine.audit_trace: false` (not under `rigor_profile: research`, which always keeps it and treats a missing, broken or unsealed trace as a gap).

The engine's own log for a run is still `.fi/run.log`; the trace is the short, ordered, checkable version of it.
