---
title: Quest map
sources: [core/rerun_from.py, core/engine.py, web/static/quest_map.js, vscode-frontier-insight/src/quest-map.ts]
updated: 2026-09-30
---
# Quest map

A picture of every step of a quest, in seven blocks: understand the question, read the literature, plan the work, build and run the experiment (or use real data), judge the result, write and review. Each step shows a plain title, what it reads and writes, and a status:

- **finished**,
- **where it stopped**: the step a paused quest waits in (the review decision, the plan, papers to download, the setup questions), the step that failed, or where a killed run was,
- **not reached**, or **off** (a path this quest does not take).

A paused quest is not shown as finished. A quest counts as finished only when nothing is left to run and there is no pause, `NEXT_STEP.md` or failure note.

Click a reached step to see what restarting there keeps and redoes, and restart from it ([[Doing a step again]]).

## Interfaces

- Web: a section on the quest page. It is hidden while the quest runs.
- VS Code: `FI: Quest map` or `@fi /map <id>`. It opens in the editor column where scripts and the paper are shown. It does not refresh by itself while a quest runs. Restart sends the `/resume … --from` command to the chat.
- CLI: no drawing. `--from` with no step prints the list; `--from --json` prints the map's data.
