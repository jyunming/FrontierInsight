---
title: Model reasoning trace
sources: [core/thinking_capture.py, core/engine.py, core/provider.py]
updated: 2026-09-30
---
# Model reasoning trace

Some models return their reasoning next to their answer (for example Kimi or DeepSeek over HTTP, Claude's thinking over the CLI, or thinking parts in VS Code). FI keeps it in `.fi/thinking.jsonl`, one line per model call, including failed calls. Each line names the step, the call id (the same as in `.fi/model_calls.jsonl`), the provider, the model that answered and the one requested, the outcome, and the text.

It is the model's own account of its reasoning, **not evidence**: it is not part of the audit trail, not sealed, and no check or decision uses it. Credentials and your home folder are removed, but it can quote your data and prompts. One line is capped at 64 000 characters and the file at 32 MB. A model that returns no reasoning writes no line.

`output.save_thinking: false` turns it off (default on). `--why <quest> <step>` (web **Why?**, `@fi /why`) says whether the file holds that step's reasoning and how many characters; the text stays in the file. `run.log` says once per step when a connection returned none.

On the VS Code connection FI asks Copilot for Claude's thinking by default (Copilot's undocumented internal model option `_enableThinking`; Opus returns a summary, not its full thinking; a model that refuses is asked again without it). For GPT models FI also sends `includeEncryptedThinking: true`, without which Copilot withholds their reasoning summary from other extensions; only the readable summary is kept. Codex's `reasoning` items from `codex exec --json` are kept too. A thinking part whose value is a list of strings is joined.

**Stated reasons.** Separately, every reason a model already writes in its answer (idea choice, skill choice, literature support or conflict, repair summary, setup answers, review verdict and moderator) goes into the audit trace as a `model_claim`, and `--why <quest> reasons` lists them per step. A stated reason and a reasoning summary are both the model's own account, not its hidden reasoning.

Related: [[literature-screening-record|Literature screening record]], [[provider-outages|Waiting out provider outages]].
