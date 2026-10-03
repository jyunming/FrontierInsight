---
title: Explore then confirm
sources: [core/phased.py, core/phased_data.py, core/phased_isolation.py, core/confirmations.py, core/engine.py, core/evidence.py, core/config.py, core/interview.py, core/plan.py, core/disclosure.py, core/optimise.py, core/todo.py]
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

- Held-back data is used only when the quest has exactly one `.csv` or `.tsv` table you supplied (in `inputs/data/`; for a quest that analyses data, in `data/` only, which is what `data_load` reads) with at least 40 rows, and at least 10 rows end up held back. At the start about 30 % of rows are set aside (by a hash of each row's text) so nothing that runs before the split is decided can read them.
- **Which rows are held back is decided right before the data is first read** (the experiment's first run, or a data quest's first reading: `phased.decide_split`, rules in `core/phased_data.py`). Rows of one subject, site or device must stay on one side, so rows are split one by one only when something says they are independent. In order: the plan's `protocol.split` (`strategy: row | group | time | spatial`, `unit`, `time_column`, `embargo`, `stratify_by`, `min_groups`, `seed`); your answer on plan.md's line `Rows that belong together: <column>` (or `independent`); the table's column names when they leave one reading (one time column such as `date` or `year` → the latest period; one unit column such as `subject_id`, `site`, `patient` whose values repeat → whole units; a column unique on almost every row numbers the rows and is not a unit). Otherwise a research quest stops once and asks (pause `data_split`, answered in plan.md); any other quest holds nothing back and says why. Group and spatial splits hold back about 30 % of the units (at least 10 units, at least 3 on each side); a time split holds back the latest periods (at most half of the rows), never rows at random, with `embargo` periods left out of both parts. Spatial blocks are never guessed from coordinates. The split, its source and a manifest with a zero-overlap check (no unit in both parts) are kept in `.fi/phased.json`; a split with overlap leaves nothing confirmed.
- The whole files and the held-back part are kept outside the quest folder, in `<output_dir>/_held_back/<quest_id>/` (`data/` files under `_held_back/<quest_id>/data/`), and the whole files are put back whenever the quest stops.
- Otherwise the confirm run uses fresh seeds that no exploration run used. Turning phased on after the experiment already ran also means fresh seeds.

## Is held back unseen? (isolation)

Outside the quest folder is not out of reach: a script run in the quest folder can open `../_held_back/...`. `core/phased_isolation.py` records `isolation` when the confirm stage begins: `docker` when every run of the quest's code was in a container that is given the quest folder and nothing that holds the kept files; `encrypted+scanned` when, without a container, the kept files were encrypted (Fernet, the `cryptography` package) with a key made for each run of FI and kept only in its memory (never on disk, in the state or in the quest code's environment), and the quest's code (`code/`, its package and scripts) was read before the confirm run for paths out of the quest folder (an absolute path, `..`, a parent of the working folder, the home folder, the name `_held_back`) and none was found. Otherwise `isolation_unverified`, a `publication_ready` gap in one plain sentence. A new run of FI writes the kept files again with its own key from the whole file in place, checked against the recorded fingerprints; a run that was killed leaves rows it cannot read back, so the next start says so and nothing is confirmed. A background job under encryption holds nothing back (the job outlives the key). A confirm on new seeds hides no data and is not judged here.

## A quest that analyses data

A no-simulation quest that analyses one table you supplied is confirmed on held-back rows too (`data_quest` in the record): its data in `data/` is held back by the same rules before the data-reading step (`data_load`) first reads it, and when exploration ends the protocol is frozen and the engine runs `data_load` once more (the evidence gate's `confirm_data` edge), with the frozen design, on the held-back rows only; the model reads those rows but does not change the design (a design that differs confirms nothing). The confirm result is recorded like a simulation's, and its numbers are set beside exploration's (`phased_data.compare`: a number that changes sign or by more than a quarter); a disagreement is said in run.log, the paper's note, the writer's note (names only) and `needs/EVIDENCE.json` (`phased.confirm_differs`), and is never a gap. When the data cannot be split (fewer than 40 rows, too few units, a time column that cannot be ordered, more than one file, a table in `inputs/data/` instead of `data/`, only material the quest gathered itself) one sentence says why and the quest is `not_applicable`, so a design revised after the analysis stays below publication-ready.

## A quest with no experiment of its own

A literature survey and `--analyze` have no design to run once more. The status is `not_applicable`: rows held back at the start go back before the data is read, one sentence goes to `run.log` and `plan.md`, and there is no stage gap in the evidence and no stage note in the paper.

## Cost

`plan.md` says what the confirm run costs: one more full run of the frozen design (every setting again, with as many runs per setting as an exploration run), and the checks after it. A record that can no longer be confirmed (compromised) says so instead of promising the run.

## During the confirm run

A request to redesign or read more literature is not followed; a protocol change is an amendment (it needs your approval, see [[how-fi-judges-correctness|How FI judges correctness]]); running on the confirm data twice is recorded as reuse. The improve loop does not run (`Engine._improve_skip`), and a search for the best design is not run again: `optimise.run_search(frozen=True)` keeps exploration's best design (matched by `study_key`, the search's key without the seed; an older record by its key for one of exploration's seeds) and only the check at finer settings runs again, never reused (`run_check(fresh=True)`), on the new seeds; with no such record the confirm run gets no result rather than searching on the confirm seeds.

## One version, one confirmation

The stages go one way: exploring → frozen → confirmed. Each frozen version is a **candidate** (`candidate` in `.fi/phased.json`, 1 for the first):

- **Frozen.** `enter_confirm` records `frozen_at_confirm` (`phased.fingerprint`): the hash of `code/` (without the redraw, run and description files FI writes there, `_NOT_THE_STUDY`, dot-folders and `.md` notes, so a redraw is not a new version; `optimise._study_key` reads files the same way), of the frozen protocol and of the environment (recorded, not compared).
- **Verdict, append-only.** `record_confirm` appends the verdict to `.fi/confirmations.jsonl` (`core/confirmations.py`): version, hashes, strategy, seed base, result hash, verdict, time, and `prev_sha256` (the line before). `Engine._phased_seal_confirmations` (from `_phased_log`) writes one `confirmation_recorded` event per line into the hash-chained trace; `confirmations.problems` finds a broken chain or a named line that is gone, and `phased.record_gap` also finds a record that says *confirmed* while a verdict line of the same version does not: both are `publication_ready` gaps (`phased_record_gap`).
- **Same version again** (nothing in code or protocol changed: a retry after a failed confirm, `--from` with the same code): a second line beside the first, status `confirm_reused`; `confirmations.worst` makes a failure count.
- **What is a change.** Code is hashed by `_code_digest` (CRLF = LF; Python by its tokens without comments, stable across Python versions). A change before the confirm run began (`confirm_executions == 0`, and for held-back data before the step was handed the held-back part, `note_confirm_handed`) only ends the freeze: same version, back to `explore`, no verdict, exploration's part of the data put back (`prepare`). Code that changed during the confirm run itself (a repair inside the step) is caught by `record_confirm` comparing `fingerprint` with `frozen_at_confirm`: `not_confirmable`.
- **Changed version** (`phased.new_candidate`, called by `Engine._phased_before_first_run` before each experiment run, and by `data_quest_gate` with the design's hash before a data quest's reading): a confirm run with no verdict yet is recorded `confirm_failed`; the record goes back to `explore` as candidate N+1 (`earlier_candidates`, `earlier_confirm_seed_bases`, so its confirm seeds avoid every earlier confirm's). Held-back rows were read by the first confirm, so a new candidate of a held-back or data quest is `not_confirmable` (`held_back_used`, `HELD_BACK_USED`) and the whole files go back.
- **What counts.** The evidence reads the current candidate only (`phased.version`, `phased.earlier_versions` in `needs/EVIDENCE.json`; the stage gap names the version). The methods paragraph ([[how-fi-judges-correctness|How FI judges correctness]], `disclosure.confirmations`) says how many versions were confirmed and how many confirmations did not hold, inside its closed grammar. A failed verdict for the current version adds the to-do card line (`todo.confirm_item`, `todo.CONFIRM_FAILED`) and the run.log line `phased.ONE_SHOT`.

## What the result says

`.fi/phased.json` records the status of the current version: explore, confirming, confirmed, confirm_failed, confirm_reused, compromised, not_confirmable or not_applicable. Anything but *confirmed* (or *not_applicable*, which adds no gap) keeps the result below publication-ready. The paper carries a note saying which kind of confirmation ran.

A confirmed result with held-back data also needs `isolation` to be `docker` or `encrypted+scanned`, and a split that was decided (a quest that held rows back one by one before the split was decided shows that as a gap).

With the setting off too, a design revised after the experiment had first been run (a later entry in `needs/DESIGN_HISTORY.json`) keeps the result below publication-ready, unless a confirm run finished after the last revision. That is read from the record (`design_revisions_at_confirm` in `.fi/phased.json`: how many design versions there were when the confirm stage began), not from the setting, so a confirm run followed by another redesign does not count. A paper with at least one run or one revision also carries FI's own paragraph in its methods, *How this result was reached* (`core/disclosure.py`): how often the design was revised and why, how many complete runs were made and how many were discarded. See [[how-fi-judges-correctness|How FI judges correctness]].

## Interfaces

The interview asks it on every interface as an advanced field, *Confirm the result on data it never saw*: on by default for research and a decision, off when exploring (`core/interview.py:smart_default_phased`, mirrored in the web page and the VS Code interview). Off under research writes `phased: false`; an unanswered question writes nothing and the profile decides. It is fixed for the quest and recorded with the approved settings. The result looks the same in all three.

Related: [[study-types|Study types]], [[scoring-criteria|Scoring criteria]].
