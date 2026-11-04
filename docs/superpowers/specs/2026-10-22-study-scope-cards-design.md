# Study scope cards: the person picks the study before FI plans it

Status: design, decided with the maintainer on 2026-10-22 (two rounds of questions, every recommendation accepted).

## Intent

A short topic gives a narrower plan than a full one, and a different one on every run. In a plan-stage test, a
37-word topic was planned as a measurement with 5 results, while the full topic it came from was planned as a search
for the best design with 14 results. A 93-word topic came close to the full plan. FI already brainstorms
directions (`ideate`), but it picks one itself, before it has read any literature, and the person never sees the
choice. `clarify` asks a few questions before the run, and when nobody answers it makes up the answers (once it set a
run-time budget of "a few minutes" for a quest allowed eight hours).

The fix: after the literature, FI writes 3 or 4 short **scope cards**, each a different study the topic could mean.
The person picks one, edits one, or writes their own sentence. The plan is then written for that study. The aim is
the one the maintainer set for short topics: ask less of the person without simplifying the study.

## Goals

- When someone can answer, every quest shows the cards. No judgement of whether a topic is "thin", because that
  judgement would be a model's reading, and a reading does not steer FI's flow.
- The cards come after the literature, so each one can say what the literature usually measures for such a study.
- The chosen card is frozen with the topic. The plan follows it, or says why it does not.
- When nobody can answer, nothing is invented: the cards are still written and recorded, no card is chosen, and the
  plan is written as today.

## Non-goals

- No detection of thin topics.
- No model choosing a card on the person's behalf.
- No new stop. A plan that does not follow the card is disclosed, never stopped (core/stop_power.py: a reading never
  ends a quest).
- No change to `ideate`, `clarify` or the plan pause beyond what is written below.

## Anti-goals (it would be a failure even if it works)

- Cards that are one study written four ways.
- A card that says "the literature measures X" when no source FI read says so.
- A headless run whose QUALITY.md or paper reads as if a person chose the scope.
- A new family of settings, or words a scientist would not use. The person sees "Pick the study to run" and plain
  cards, not "scope" or other words from this codebase.

## Decided

1. **Where.** A new node `scope` between `pause_after_literature` and `select_skills`: it is after the literature,
   and before the skills, the plan and the design. Skills are picked for the chosen study.
   `ideate`'s chosen direction is one of the inputs and is not final.
2. **When.** Every quest with `pauses.scope: ask` (the default) whose run has someone to answer. "Someone to answer"
   is the test `clarify` already uses: `clarify_callback` is set or `.fi/clarify_answer.json` exists (see
   `Engine._clarify_answerable`), extended to the scope answer file below. `pauses.scope: off` skips the node
   entirely: no cards, no record.
3. **The cards.** One model call, routed as node `scope` (so `provider.node_models` can name a model for it), from
   the topic, the clarify answers, ideate's direction and the literature FI read. It returns 3 or 4 cards. Each card
   has:
   - `question`: one sentence;
   - `study_type`, one of the plan's two study kinds, `measure` or `find_best_design`
     (core/optimisation_plan.py). A comparison of fixed alternatives is a `measure` card whose `compared_settings`
     list them; the person sees "measure and compare" or "find the best design";
   - `main_results`: the quantities it would report;
   - `compared_settings`: what it varies;
   - `literature`: a list of `{says, source}` entries, where `source` must be the id of a source in the literature
     record;
   - `run_time`: a rough run-time estimate in words.

   FI then checks the reply against its own records:
   - an entry in `literature` whose source is not one FI read is removed, and the removal is recorded;
   - two cards with the same `study_type` and the same `main_results` set (compared after lower-casing and joining
     spaces, nothing looser) are one card. Whether the cards are really different studies is a reading FI cannot
     make; the prompt asks for it, and the acceptance runs measure it;
   - when fewer than 3 distinct cards are left, the model is asked once more, naming what was the same;
   - one card must be the widest reading of the topic: `find_best_design` when the topic names something to improve,
     otherwise the card with the most main results. FI asks for it in the prompt and records whether the reply has
     it; a reply without it is disclosed, not refused.
4. **The answer.** Through the same channel as clarify's answers:
   - terminal `--interactive`: a numbered list, and the person types a number, `e <n>` to edit a card, or a sentence
     of their own;
   - web: cards to click, each editable;
   - VS Code: a quick pick, with "Edit…" and "Write my own…" entries.

   The answer can also be given in advance as `.fi/scope_answer.json`, for a batch run; the file is looked for when
   the node runs, not only at the start. The web and VS Code paths generalise clarify's waiting/answer files rather
   than adding a second mechanism. A `--fleet` quest has no one to answer and gets item 6. The pause wait time is
   `pauses.timeout_s`, as for the other answer pauses. If nobody answers in time, the run goes on as if nobody could
   answer.
5. **Freezing.** The chosen card (as edited) goes into the quest state (`scope_card`) and into `.fi/scope.json`, with
   `chosen_by: person` and all the cards shown. The topic itself is not changed: `state["topic"]` and
   `config.topic` stay the person's words, so the paper, the claim checks and the oracle prompts are untouched. The
   card is a separate input to the plan and design prompts only. Its `study_type` feeds the study-type channel
   that clarify's study-type question already feeds (`_study_type_asked` / `_study_type_note`); a card a person chose
   wins over a clarify answer the model gave for itself. The plan must cover each `main_results` and
   `compared_settings` entry or say why it leaves one out.
6. **No one to answer.** The node still writes the cards to `.fi/scope.json` with `chosen_by: nobody`. The plan gets
   no card and is written as today. `plan.md` and QUALITY.md list the cards under one plain line: "Nobody picked the
   study to run; FI planned from the topic alone. The studies it could also have been:".
7. **Plan not following the card.** After the plan step, FI compares the plan's metrics and settings with the card's
   lists. Each missing entry without a stated reason is listed in QUALITY.md and, through the existing limitations
   note, in the paper. The comparison is by names (a reading), so it never stops the quest.
8. **Resume and redesign.** The node runs once per quest: when `.fi/scope.json` exists (chosen or not) it is not
   run again, so a resume, a redesign (`cross_check → design`) or a rewrite never asks again, and a headless quest
   resumed with `--interactive` does not ask either. The node is also skipped when the plan step would be skipped
   (`iteration > 0`, `analyze_local_first`, or a `plan.md` a person wrote). To change the study, the person uses the
   existing plan revision (`--revise-plan`); a revision may say a new study, which is recorded in `.fi/scope.json`
   history, and the plan is rewritten the way a revision already rewrites it.
9. **Interfaces.** CLI, web and VS Code all show the cards (three-interface parity). The config key is
   `pauses.scope: ask | off`, with the YAML `off` → False fix that `clarify` and `review` have (new work for this
   key). The node gets a `STAGE_PROGRESS` line, a `provider.node_models` name (`scope`) and a `--from` entry in
   `core/rerun_from.py` (rerunning from it asks again and replaces the plan, so it needs `--approve-as`, as the
   other steps before the design do). It is wired after the supply stop, so a person who supplies papers first
   sees cards built from them. Only the main graph edge carries it; the phased clarify variant needs no change,
   since the node sits after literature.

## Done when (the maintainer's test)

Plan-stage runs, `pauses.plan: ask`, two runs each, on a 37-word and a 93-word version of a full topic:

1. Among the cards there is one whose study type and main results match the full topic's plan.
2. After that card is picked (by the test harness, standing in for the person), the 37-word plan has the full plan's
   study type in both runs.

Until both hold, the package is not done. Also report the number of plan results against the full plan (not a pass
condition). A known positive must still pass: a full topic's plan is unchanged in study type when its widest card is
picked.

## Left to the builder

- The card prompt wording (an `agents/scope.md` `string.Template`).
- The JSON shape inside `.fi/scope.json`, beyond the fields above.
- How the web and VS Code views look, within the existing new-quest or answer-pause screens.
- Test layout. Tests use a fake LLM (`monkeypatch` on `LLMClient.chat`), never a real model.
- The registry and docs entries (`dev/registry.md`, `docs/capabilities-reference.md`, the README's pause list, the
  VS Code README).
