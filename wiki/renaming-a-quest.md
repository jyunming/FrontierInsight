---
title: Renaming a quest
sources: [core/quest_title.py, web/server.py, vscode-frontier-insight/src/trace.ts]
updated: 2026-09-30
---
# Renaming a quest

Change a finished or paused quest's title when the one the model chose is poor. Results, data and code are not touched, and no model is called.

What changes, all together or not at all: the paper's title line, `config.yaml`'s `title`, the summary file, and the saved checkpoint. The change is recorded in the audit trail, so the evidence level (how strongly the result is backed) is not lowered.

Limits: one line, at most 200 characters. It is refused while the quest runs. PDFs, slides, posters and talks already made keep the old title until they are made again. Running a step again ([[doing-a-step-again|Doing a step again]]) may choose another title.

- CLI: `fi tools rename <id> <new title>` (use `--title="-..."` for a title that starts with `-`).
- Web: the Rename button under the quest id. It offers to make the outputs again.
- VS Code: `@fi /rename <id> <new title>`, then `@fi /generate` for the outputs. There is no command-palette entry.
