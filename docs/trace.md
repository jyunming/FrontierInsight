# The trace: what a quest did, and why

Every quest keeps a diary of what it did, in order: `<quest folder>/.fi/audit.jsonl`. It answers three questions after the fact:

- **What ran, and what did each check say?** Every step, every check's verdict (protocol, oracle, run manifest, numeric warnings, evidence, methodology audit), and the files the quest wrote with their sha256.
- **Why did it go this way?** Each time the quest chose its next step (redo the design, write, stop for you) it writes down the facts it looked at.
- **What did the model say about why?** The assumptions and the options the model weighed, and what the reviewer objected to. These are marked **model claim**: the model's own account, kept so you can argue with it. They are never mixed with a check result and no check reads them.

## Look at it

```bash
python launch.py --trace <quest id or folder>                  # the timeline, and a check that nothing was edited
python launch.py --trace <quest> --trace-node design           # only one step
python launch.py --trace <quest> --trace-detail summary        # just the shape: steps, stops, routes (or: checks, debug)
```

On the web, open a quest and expand **Trace** (same detail and step filters). The list reads the same as the command. In VSCode, `@fi /trace <quest_id> [--node <name>] [--detail summary|checks|debug]` prints the same lines in chat.

## How to read a line

| You see | It means |
| --- | --- |
| `design: done in 1.2s, wrote design` | a step finished and which parts of the quest state it changed |
| `check oracle: ok` | a check's verdict; the record with the detail and its hash are in the event |
| `review: next is revise because verdict=revise, ...` | the route the quest took and the facts it read to choose |
| `design: alternative: Wilson pooled -> rejected (assumes independence we can't verify here) [model claim]` | something the model said about its own reasoning, and — when it chose between options — what it decided and why |
| `evidence_gate: verdict: only 2 of 6 sources are on-topic -> broaden [model claim]` | the evidence gate's own stated reason for its verdict |
| `analyze: next_step: the effect is within noise at this sample size -> re_experiment [model claim]` | analyze's stated reason for the next step it picked |
| `waiting for you (plan)` / `stopped for you` | the quest paused for a person; after you resume, the same step starts again from its beginning |

`summary` shows steps, stops and routes; `checks` (the default) adds each check, each file written and the model's reasons; `debug` adds every step's start.

## What it can and cannot tell you

- Each event carries the hash of the one before it. Remove, reorder or edit a line and `--trace` (and the web page) says which event it broke at, and the command exits with 1. This finds accidents and partial edits; someone who can run the code can rewrite the whole file, so it is not a signature.
- The trace is a record, not a control. If it cannot be written the quest goes on and says so once in `run.log`.
- Nothing secret is written: values of environment variables that look like keys, key-shaped strings and your home folder are removed, and long text is cut with a note saying how much.
- It is not the model's private thinking. Only what the model chose to state as reasoning, and what the engine measured, are in it.
- A model claim also carries who actually said it (`provider`, `model` — the one that answered, even after a fallback took over) and a hash of the exact prompt and reply it came from (`prompt_hash`, `response_hash`), so a claim can be checked against the real call rather than taken on faith. These aren't in the one-line text (they'd repeat on every line and add noise); open the quest page's Trace panel or `.fi/audit.jsonl` directly to see them on an event. A rule-decided verdict (no model was asked) is never recorded as a model claim — only what a model actually said is.
- Turn it off with `engine.audit_trace: false`.

The engine's own log for a run is still `.fi/run.log`; the trace is the short, ordered, checkable version of it.
