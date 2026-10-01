---
title: Explore then confirm
sources: [core/phased.py, core/engine.py, core/evidence.py, core/config.py, core/interview.py, core/plan.py, core/disclosure.py]
updated: 2026-10-01
---
# Explore then confirm

A study whose design was tuned after looking at results can fool itself. With `engine.phased: true` FI splits the work in two:

1. **Explore.** The model may change the design after seeing results. The protocol is *not* frozen yet.
2. **Confirm.** When the evidence check would accept the result, FI freezes the protocol there and runs the frozen design **once more** on data it held back, or on new random seeds. Then the paper is written.

## When it is on

- **Research quests: on by default.** `rigor_profile: research` fills in `engine.phased: true` where the config is silent (`core/config.py:_RESEARCH_DEFAULTS`). Unlike the rest of the profile, a config may say `phased: false` and keep it: `plan.md` (*How the result will be confirmed*) and `run.log` then say the result is not confirmed on data or seeds the exploration never saw.
- **Default profile: off**, and off changes nothing.
- **Older research quests** that began before this default, whose `config.yaml` does not set it, go on as they began, without the two stages (`phased.kept_off`, applied by `Engine._phased_keep_off_if_began_before` before the approved settings are compared, and by `--update` before it approves them, so the approved record says what runs and nothing stops). "Began before" means: a graph step ran and the hash-chained trace has no `phased_started` event (every start with it on writes one). `run.log` says so at every start; `engine.phased: true` in their `config.yaml`, approved with `--update`, turns it on.

## Held-back data or new seeds

- Held-back data is used only when `inputs/data/` holds exactly one `.csv` or `.tsv` file with at least 40 rows, and at least 10 rows end up held back. About 30 % of rows are held back, chosen by a hash of the quest id and the row's text, so a row keeps its side if rows are added later. The whole files and the held-back part are kept outside the quest folder, in `<output_dir>/_held_back/<quest_id>/`, and the whole files are put back in `inputs/data/` whenever the quest stops.
- Otherwise the confirm run uses fresh seeds that no exploration run used. Turning phased on after the experiment already ran also means fresh seeds.

## A quest with no experiment of its own

A literature survey, a quest that collects and analyses data instead of simulating (no simulation), and `--analyze` have no design to run once more. The status is `not_applicable`: rows held back at the start go back before the data is read (set at the start when the config pins it, or right after the clarify step, off the event loop, when the quest finds it has no simulation), one sentence goes to `run.log` and `plan.md`, and there is no stage gap in the evidence and no stage note in the paper. FI today has no confirm run for a no-simulation data quest: the confirm edge only goes back to `execute`.

## Cost

`plan.md` says what the confirm run costs: one more full run of the frozen design (every setting again, with as many runs per setting as an exploration run), and the checks after it. A record that can no longer be confirmed (compromised) says so instead of promising the run.

## During the confirm run

A request to redesign or read more literature is not followed; a protocol change is an amendment (it needs your approval, see [[how-fi-judges-correctness|How FI judges correctness]]); running on the confirm data twice is recorded as reuse.

## What the result says

`.fi/phased.json` records the status: explore, confirming, confirmed, confirm_failed, confirm_reused, compromised, not_confirmable or not_applicable. Anything but *confirmed* (or *not_applicable*, which adds no gap) keeps the result below publication-ready. The paper carries a note saying which kind of confirmation ran.

With the setting off too, a design revised after the experiment had first been run (a later entry in `needs/DESIGN_HISTORY.json`) keeps the result below publication-ready, unless a confirm run finished after the last revision. That is read from the record (`design_revisions_at_confirm` in `.fi/phased.json`: how many design versions there were when the confirm stage began), not from the setting, so a confirm run followed by another redesign does not count. A paper with at least one run or one revision also carries FI's own paragraph in its methods, *How this result was reached* (`core/disclosure.py`): how often the design was revised and why, how many complete runs were made and how many were discarded. See [[how-fi-judges-correctness|How FI judges correctness]].

## Interfaces

The interview asks it on every interface as an advanced field, *Confirm the result on data it never saw*: on by default for research and a decision, off when exploring (`core/interview.py:smart_default_phased`, mirrored in the web page and the VS Code interview). Off under research writes `phased: false`; an unanswered question writes nothing and the profile decides. It is fixed for the quest and recorded with the approved settings. The result looks the same in all three.

Related: [[study-types|Study types]], [[scoring-criteria|Scoring criteria]].
