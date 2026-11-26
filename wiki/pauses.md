---
title: Pauses
sources: [core/config.py, core/engine.py, core/acceptance.py, launch.py, web/server.py, web/quest_launcher.py, vscode-frontier-insight/src/bridge.ts]
updated: 2026-10-01
---
# Pauses

A pause is FI stopping because it needs a person. It is not a failure. Every pause writes `NEXT_STEP.md` in the quest folder saying what to do. To continue: `fi --resume <id>` (CLI), the Resume button on the quest page (web), or `@fi /resume <id>` (VS Code). All settings live under `pauses:`.

| Pause | Setting (default) | What it asks |
|---|---|---|
| Plan | `pauses.plan` (off; `ask` under the research profile) | read or change `plan.md` |
| Review | `pauses.review` (ask) | accept, reject or [[refine\|Refine]] the result |

Some checks also stop the quest, such as a failing [[how-fi-judges-correctness|oracle]], a best-design plan whose search cannot start ([[study-types|Study types]]) or a missing equation label under the research profile.

## No setup questions

A study asks nothing when it starts. The first node (`clarify`, the name is kept for stored studies) makes no model call: it sets the study mode from the commission's `method` and the title from its `title` ([[starting-a-study|Starting a study]]). `pauses.clarify`, `engine.clarify_mode` and `engine.clarify_overrides` still load; the log says each does nothing. A stored study that stopped at the questions runs the first node again on resume and goes on.

## Review

A decision is accept, reject, or refine *with* notes. Anything else (a closed prompt, an empty answer file, refine with no notes) is not a decision: the quest stops cleanly and asks again on resume. It is never taken as accept. The one exception is the prompt itself: at the terminal prompt and the VS Code input box, an empty refine confirmed with Enter means accept, as the prompt says.

- CLI: `--resume <id> --accept yes` (or `--accept partly "what you do not accept"` / `--accept not-checked`; `--approve-as <name>` records that name, else the login name), `--reject` or `--refine "notes"`; or `--interactive`.
- Web: the review banner on the quest page.
- VS Code: Accept / Reject / Refine in the chat, for quests on the VS Code chat model. For a quest on another provider, use the CLI or the web page.

**Before you accept** (`core/acceptance.py`), every interface shows what the result does not guarantee and its most important evidence gaps (two and three at most, none twice), then asks one question: *Have you reviewed the evidence record, and do you accept these claims and the limits listed?* — yes, partly, no, I did not check. Only "yes" lets the result reach `publication_ready`. "I did not check" finishes the quest with the gap "no person reviewed the evidence before accepting"; "partly" needs a short note (what is not accepted), which is kept as a gap until a changed paper is accepted with "yes". The answer is recorded as a receipt (`acceptance` in `needs/EVIDENCE.json`, a `result_accepted` event in the trace): who, when, which interface, the answer and note, the paper and evidence-record sha256 and the limits listed. "No" does not accept: refine with what is wrong, or look again. An accept without an answer, or a "partly" without a note, is not a decision. The old question ("do the main numbers match what you expected?") was dropped rather than kept as a second prompt: it invited confirming an expectation, and the user wanted one question.

`pauses.auto_accept_on_pass: true` accepts a clean result without stopping, but no person looked: the result is marked "not reviewed by a person" and stays one level below `publication_ready` ([[how-fi-judges-correctness|How FI judges correctness]]). So does a quest with `pauses.review: off`.

## Plan

- CLI: `--resume <id> --revise-plan "<change>"` changes the plan only; `--resume <id>` runs it.
- Web: the Plan panel. VS Code: `@fi /plan <id> [<change>]` (or `@fi /resume <id> --revise-plan "<change>"`).
- A plan stop because a check does not say where its expected value comes from (research) offers three ways on: let FI fill it in, change it yourself, or go on as it is with your name (`--accept-checks <id> --approve-as <you>`, the web *Go on as it is* button, `@fi /accept-checks <id>`). See [[oracle-provenance|Oracle provenance]].

Related: [[quest-map|Quest map]] (a paused quest shows where it stopped), [[model-behind-the-numbers|The model behind the numbers]], [[provider-outages|Waiting out provider outages]] (a provider outage is waited out, not turned into a pause).
