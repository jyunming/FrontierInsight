---
title: Explore then confirm
sources: [core/phased.py, core/engine.py, core/evidence.py, core/config.py]
updated: 2026-09-30
---
# Explore then confirm

A study whose design was tuned after looking at results can fool itself. With `engine.phased: true` (off by default) FI splits the work in two:

1. **Explore.** The model may change the design after seeing results. The protocol is *not* frozen yet.
2. **Confirm.** When the evidence check would accept the result, FI freezes the protocol there and runs the frozen design **once more** on data it held back, or on new random seeds. Then the paper is written.

## Held-back data or new seeds

- Held-back data is used only when `inputs/data/` holds exactly one `.csv` or `.tsv` file with at least 40 rows, and at least 10 rows end up held back. About 30 % of rows are held back, chosen by a hash of quest id, file and row, so a row keeps its side if rows are added later. The originals are kept in `.fi/phased/original/` and put back whenever the quest stops.
- Otherwise the confirm run uses fresh seeds that no exploration run used. Turning phased on after the experiment already ran also means fresh seeds.

## During the confirm run

A request to redesign or read more literature is not followed; a protocol change is an amendment (it needs your approval, see [[how-fi-judges-correctness|How FI judges correctness]]); running on the confirm data twice is recorded as reuse.

## What the result says

`.fi/phased.json` records the status: explore, confirming, confirmed, confirm_failed, confirm_reused or compromised. Anything but *confirmed* keeps the result below publication-ready. The paper carries a note saying which kind of confirmation ran.

## Interfaces

YAML only (`engine.phased: true`): there is no CLI flag, interview question, web switch or VS Code switch. The result looks the same in all three.

Related: [[study-types|Study types]], [[scoring-criteria|Scoring criteria]].
