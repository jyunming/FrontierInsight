---
title: Picking the study to run
sources: [core/scope_cards.py, agents/scope.md, core/config.py, core/rerun_from.py, core/quality_report.py, core/run_checks.py, launch.py, web/static/quest.html, vscode-frontier-insight/src/bridge.ts]
updated: 2026-10-24
---
# Picking the study to run

A topic can mean more than one study. After the literature is read and before the plan is written, FI offers three or four short study cards, one for each study the topic could mean, and a person picks one. The plan is then written for that study. This is the `scope` step, set by `pauses.scope` ([[pauses|Pauses]]).

## What a card says

Each card has:

- a one-sentence question;
- its kind: *measure and compare*, or *find the best design* ([[study-types|Study types]]). The widest reading of the topic is marked;
- the results it reports and the settings it varies;
- what the sources FI read say about such a study, each line naming its source;
- a rough run time, in words.

Cards that repeat one another (the same kind with the same results) are dropped. A literature line that cites a source FI did not read is removed, and the removal is said in the quality report.

## How to answer

- `pauses.scope: ask` (the default) offers the cards. `off` skips the step: no cards and no record.
- Terminal, with `--interactive`: a number, `e <n>` to edit card n's question, or a sentence of your own.
- Quest page on the web: each card has an edit box, and there is a *Write my own* box.
- VS Code: a quick pick with one item per card, *Edit a card…* and *Write my own…*.
- In advance: write `.fi/scope_answer.json`, for example `{"study": 2}`. It is read before anyone is asked.

With nobody to answer (a `--fleet` or headless run), the cards are still recorded and the plan is written from the topic alone.

## Where the choice goes

- `.fi/scope.json` keeps the cards, the pick (or `nobody`), the removed literature lines and any earlier pick.
- `plan.md` has a section *The study to run*. The plan prompt is told to cover each result and setting of the chosen study, or say why one is left out.
- `QUALITY.md` has a section *The study*.
- The paper's limitations name each result or setting of the chosen study that the plan does not cover. This is a comparison by names, so it is stated as a reading and never stops the quest.

## Running it again

`--resume <id> --from studies --approve-as <you>` offers the studies again, records the new pick, and redoes the plan, design, code and everything after it. The older record is kept in `.fi/previous/`. See [[doing-a-step-again|Doing a step again]].

Related: [[pauses|Pauses]], [[study-types|Study types]], [[literature-screening-record|Literature screening record]].
