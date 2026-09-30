---
title: Model reasoning trace
sources: [core/thinking_capture.py, core/engine.py, core/provider.py]
updated: 2026-09-30
---
# Model reasoning trace

Some models return their reasoning next to their answer (for example Kimi or DeepSeek over HTTP, Claude's thinking over the CLI, or thinking parts in VS Code). FI keeps it in `.fi/thinking.jsonl`, one line per model call, including failed calls. Each line names the step, the call id (the same as in `.fi/model_calls.jsonl`), the provider, the model that answered and the one requested, the outcome, and the text.

It is the model's own account of its reasoning, **not evidence**: it is not part of the audit trail, not sealed, and no check or decision uses it. Credentials and your home folder are removed, but it can quote your data and prompts. One line is capped at 64 000 characters and the file at 32 MB. A model that returns no reasoning writes no line.

`output.save_thinking: false` turns it off (default on). No interface displays it; open the file in the quest folder.

Related: [[literature-screening-record|Literature screening record]], [[provider-outages|Waiting out provider outages]].
