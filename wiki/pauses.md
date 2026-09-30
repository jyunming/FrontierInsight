---
title: Pauses
sources: [core/config.py, core/engine.py, launch.py, web/server.py, web/quest_launcher.py, vscode-frontier-insight/src/bridge.ts]
updated: 2026-09-30
---
# Pauses

A pause is FI stopping because it needs a person. It is not a failure. Every pause writes `NEXT_STEP.md` in the quest folder saying what to do. To continue: `fi --resume <id>` (CLI), the Resume button on the quest page (web), or `@fi /resume <id>` (VS Code). All settings live under `pauses:`.

| Pause | Setting (default) | What it asks |
|---|---|---|
| Setup questions | `pauses.clarify` (unset: ask if someone can answer, else FI answers itself) | the question's scope, before any work |
| Papers | `pauses.papers` (true) | paywalled papers to download |
| Plan | `pauses.plan` (off; `ask` under the research profile) | read or change `plan.md` |
| Review | `pauses.review` (ask) | accept, reject or [[Refine]] the result |

Some checks also stop the quest, such as a failing [[How FI judges correctness|oracle]], a best-design plan ([[Study types]]) or a missing equation label under the research profile.

## Setup questions (clarify)

Before any work, one model call writes the setup questions, each with a suggested answer: what you want to see, the title (three suggestions, or your own), the baseline to compare with, theory or experiment, whether it can be simulated, how success is measured, budget, outputs, depth, venue and the shape of the topic. When the topic leaves it open, it also asks whether this is a measurement or a search for the best design ([[Study types]]).

- `off`: skipped. `auto`: FI uses its own suggestions. `ask`: the quest waits for your answers.
- Unset (what the interview writes): it asks if someone can answer; otherwise FI uses its suggestions.
- A `title:` in the YAML removes the title question. The interview always writes one.

How to answer:

- CLI: `fi --config q.yaml --interactive` asks at the terminal (Enter keeps the suggestion). A paused quest: write your answers to `.fi/clarify_answer.json` (questions are in `.fi/clarify_questions.json`) and resume.
- Web: a quest the web page started or resumed shows a form on the quest page, and its badge reads "Waiting for your answers".
- VS Code: one input box per question, but only for quests on the VS Code chat model. Esc keeps the suggestions (the chat's "quest will fail" message is wrong).

## Review

A decision is accept, reject, or refine *with* notes. Anything else (a closed prompt, an empty answer file, refine with no notes) is not a decision: the quest stops cleanly and asks again on resume. It is never taken as accept.

- CLI: `--resume <id> --accept`, `--reject` or `--refine "notes"`; or `--interactive`.
- Web: the review banner on the quest page.
- VS Code: Accept / Reject / Refine in the chat, for quests on the VS Code chat model. For a quest on another provider, use the CLI or the web page.

`pauses.auto_accept_on_pass: true` accepts a clean result without stopping.

## Plan

- CLI: `--resume <id> --revise-plan "<change>"` changes the plan only; `--resume <id>` runs it.
- Web: the Plan panel. VS Code: `@fi /plan <id> [<change>]`.

Related: [[Quest map]] (a paused quest shows where it stopped), [[The model behind the numbers]], [[Waiting out provider outages]] (a provider outage is waited out, not turned into a pause).
