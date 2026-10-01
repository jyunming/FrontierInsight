---
title: Pauses
sources: [core/config.py, core/engine.py, core/acceptance.py, launch.py, web/server.py, web/quest_launcher.py, vscode-frontier-insight/src/bridge.ts]
updated: 2026-10-01
---
# Pauses

A pause is FI stopping because it needs a person. It is not a failure. Every pause writes `NEXT_STEP.md` in the quest folder saying what to do. To continue: `fi --resume <id>` (CLI), the Resume button on the quest page (web), or `@fi /resume <id>` (VS Code). All settings live under `pauses:`.

| Pause | Setting (default) | What it asks |
|---|---|---|
| Setup questions | `pauses.clarify` (unset: ask if someone can answer, else FI answers itself) | the question's scope, before any work |
| Papers | `pauses.papers` (true) | paywalled papers to download |
| Plan | `pauses.plan` (off; `ask` under the research profile) | read or change `plan.md` |
| Review | `pauses.review` (ask) | accept, reject or [[refine\|Refine]] the result |

Some checks also stop the quest, such as a failing [[how-fi-judges-correctness|oracle]], a best-design plan whose search cannot start ([[study-types|Study types]]) or a missing equation label under the research profile.

## Setup questions (clarify)

Before any work, one model call writes the setup questions, each with a suggested answer: what you want to see, the title (three suggestions, or your own), the baseline to compare with, theory or experiment, whether it can be simulated, how success is measured, budget, outputs, depth, venue and the shape of the topic. When the topic leaves it open, it also asks whether this is a measurement or a search for the best design ([[study-types|Study types]]).

- `off`: skipped. `auto`: FI uses its own suggestions. `ask`: the quest waits for your answers.
- Unset (what the interview writes): it asks if someone can answer; otherwise FI uses its suggestions.
- A `title:` in the YAML removes the title question. The interview always writes one.

How to answer:

- CLI: `fi --config q.yaml --interactive` asks at the terminal (Enter keeps the suggestion). A paused quest: write your answers to `.fi/clarify_answer.json` (questions are in `.fi/clarify_questions.json`) and resume.
- Web: a quest the web page started or resumed shows a form on the quest page, and its badge reads "Waiting for your answers".
- VS Code: one input box per question, but only for quests on the VS Code chat model. Esc keeps the suggestions (the chat's "quest will fail" message is wrong).

## Review

A decision is accept, reject, or refine *with* notes. Anything else (a closed prompt, an empty answer file, refine with no notes) is not a decision: the quest stops cleanly and asks again on resume. It is never taken as accept. The one exception is the prompt itself: at the terminal prompt and the VS Code input box, an empty refine confirmed with Enter means accept, as the prompt says.

- CLI: `--resume <id> --accept yes` (or `partly` / `not-checked`), `--reject` or `--refine "notes"`; or `--interactive`.
- Web: the review banner on the quest page.
- VS Code: Accept / Reject / Refine in the chat, for quests on the VS Code chat model. For a quest on another provider, use the CLI or the web page.

**Before you accept** (`core/acceptance.py`), every interface shows what the result does not guarantee and its most important evidence gaps (two and three at most, none twice), then asks one question: *Do the main numbers match what you expected?* — yes, partly, no, I did not check. The answer is recorded with the accept (`acceptance` in `needs/EVIDENCE.json`, a `result_accepted` event in the trace). "No" does not accept: refine with what is wrong, or look again. An accept without an answer is not a decision.

`pauses.auto_accept_on_pass: true` accepts a clean result without stopping, but no person looked: the result is marked "not reviewed by a person" and stays one level below `publication_ready` ([[how-fi-judges-correctness|How FI judges correctness]]). So does a quest with `pauses.review: off`.

## Plan

- CLI: `--resume <id> --revise-plan "<change>"` changes the plan only; `--resume <id>` runs it.
- Web: the Plan panel. VS Code: `@fi /plan <id> [<change>]` (or `@fi /resume <id> --revise-plan "<change>"`).
- A plan stop because a check does not say where its expected value comes from (research) offers three ways on: let FI fill it in, change it yourself, or go on as it is with your name (`--accept-checks <id> --approve-as <you>`, the web *Go on as it is* button, `@fi /accept-checks <id>`). See [[oracle-provenance|Oracle provenance]].

Related: [[quest-map|Quest map]] (a paused quest shows where it stopped), [[model-behind-the-numbers|The model behind the numbers]], [[provider-outages|Waiting out provider outages]] (a provider outage is waited out, not turned into a pause).
