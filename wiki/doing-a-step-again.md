---
title: Doing a step again
sources: [core/rerun_from.py, core/engine.py, web/server.py, vscode-frontier-insight/src]
updated: 2026-09-30
---
# Doing a step again

Run a quest again from one of the steps it already reached, keeping everything decided before that step. What that step and the later ones produced is moved to `.fi/previous/<time>/` first, so old and new can be compared.

Steps, in order: ideas, literature, plan, design, skills, code, run, figures (quests without a simulation), analysis, crosscheck, evidence, writing, claims, review. Only steps the quest actually reached can be chosen.

- **ideas, literature, plan, design** replace the plan and the frozen protocol. That is a change after results were seen, so it needs `--approve-as <you>`, is recorded as a post-hoc amendment, and holds the result below publication-ready.
- **skills** picks the skills again from the current YAML (`engine.skills`, `skills_exclude`, `skills_required`) and leaves plan and protocol alone.
- Later steps just re-run from there.
- A quest folder that was copied or moved is still judged in its own folder: a rerun from the writing, the claims or the review reads the paper in the folder the quest is in now, not the one it was written in, and so does the evidence level (the saved state still names the old path; `core/rerun_from.py::in_quest` takes it to the new one).

For a small change to a finished paper, [[refine|Refine]] is usually cheaper.

## Interfaces

- CLI: `fi --config q.yaml --resume <id> --from <step> [--approve-as <you>]`. `--from` alone lists the steps reached, with what each would redo.
- Web: the menu beside Resume on the quest page (it reads "Redo" at a pause), or click a step on the [[quest-map|Quest map]].
- VS Code: `@fi /resume <id> --from <step>`; `--from` alone lists the steps.

Gap: the web page and VS Code refuse ideas, literature, plan and design, because those need your name as approver; they show the terminal command instead.
