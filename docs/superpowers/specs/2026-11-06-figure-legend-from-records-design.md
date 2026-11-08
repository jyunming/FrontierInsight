# Every line in a figure has an identity in FI's records; the legend names what tells it apart

Status: direction approved by the maintainer on 2026-11-06; to be measured on replays before building in full.

## Why

A reader judges a figure by its legend first. In the last small study, four of five delivered papers failed the
figure-text check because the legend or caption did not match what the figure draws: six identical "Reference line"
entries, "a curve" / "curve 1" entries, a caption naming three reference lines while four are drawn, and an empty panel
the caption never mentions. Of 137 legend entries in 16 figures, the 88 drawn from FI's result table were all named
correctly; the 49 placeholders (36%) came only from the two calls of the figure frame (`core/frame_figure.py`) through
which a figure body types numbers itself:

- `ref(..., value=<number>)`: a constant line FI cannot name ("Reference line").
- `curve(panel, label, xs, ys)`: x/y computed in the figure body, never in the results, so FI does not know which group
  it belongs to ("a curve"; the body's own label is not trusted).

Two earlier attempts guessed the identity of these lines after drawing (match the numbers to results; trace where the
numbers came from) and were given up: the guesses were coincidental or found nothing, because the numbers never went
through FI's records.

## Goal

Every line, point set, band and reference line a delivered figure draws has an identity in FI's records before it is
drawn, and its legend entry is written by FI from those records.

- The maintainer's rule for the legend: **an entry says what tells that line apart from the others in the same panel**.
  What all lines share goes to the axis label or the caption, not into every entry.
- Two lines that differ in nothing are one line drawn twice: they are merged, not given two invented names.
- The caption's account of what is drawn (how many reference lines, which each is, which panel is empty or left out) is
  written by FI from the figure record, never counted by the model. This extends the 2026-11-01 decision that FI
  composes figure prose.

## Non-goals

- No change to which figures a study has, or to the figure checks' strictness.
- No guessing of a line's identity from its numbers or its code after it is drawn.
- No styling work (layout and overlap were fixed separately).

## Anti-goals (a failure even if it works)

- A legend entry that is wrong. Fewer lines is acceptable; a misnamed line is not.
- A new term or config key a scientist has to learn (AGENTS.md: rigorous inside, simple surface).
- Rules special to one study's quantities; everything must be general.

## The design

1. **Close the two bypasses.**
   - A formula curve (a closed form, an approximation, a theory line) is computed in the analysis step and written to
     the result table with the column(s) that group it (for example a source or method column), on as fine an x grid as
     the figure needs. The figure then draws it like any other table series, and FI names it.
   - A reference line is a plan reference (`ref_id`) or a value read from the results (`at=`). A `value=` constant is
     matched against the plan's references and the results; if it matches exactly one, it takes that identity and name;
     otherwise the figure body is asked to register it in the analysis step.
2. **Legend from what differs.** For each panel, FI takes the grouping columns of the drawn series and reference lines,
   keeps only those whose values differ between entries, and writes each entry from those values (a reference line from
   its plan name and value). Shared parts move to the axis label or caption.
3. **Caption facts from the record.** FI writes the list of drawn reference lines (name and value), and any empty or
   removed panel, into the caption from the figure record.
4. **When it cannot be fixed.** After one repair round, a line that still has no identity is not drawn; the figure is
   kept, and the quality report names the line removed and why. If a panel is left with nothing, the existing
   empty-panel handling applies and the caption says so.

## Measure first (before any merge)

- **Replay, no model calls:** re-render the delivered figure bodies of the fc8, fc10 and fc11 rounds (about 40 figures)
  under the new frame. Measure: placeholders left (target 0); lines removed for want of an identity (the cost); captions
  whose list of drawn lines disagrees with the figure (target 0); and with one planted mislabel per figure, how many are
  caught.
- **Small model A/B (Sonnet, a few figures):** whether the code model, given the repair message, moves a formula curve
  or constant into the analysis step and registers it. Cost stated before running (a few hundred thousand tokens).
- **Then a small study round:** the figure-text check (G3) fails in fewer than the 4 of 5 papers of fc11, with no new
  false stop.

## Settled after the builder check (the author's calls, open to the maintainer)

- **What a line's identity is.** A table series is identified by its result id and the setting level(s) it is drawn at
  (the frame already records `id` and `level`); a reference line by its plan reference id and value; a registered
  formula curve by its registered name and the level(s) it is computed for. "What differs" is computed over these
  fields only. When nothing differs between two entries, they are the same line and are merged.
- **Where a formula curve is registered.** Not as a new result of the frozen plan (the protocol stays frozen): as a
  named curve in the analysis output (`derived.json`), with the levels it is computed for, marked as carrying no
  result. Registering it means the repair rewrites the analysis code as well as the figure body, inside the existing
  figure-repair budget; the analysis is re-run on the recorded trials (no new simulation).
- **What a `value=` constant may match.** Only the plan's references (within the plan's stated precision). A match with a
  result value is not accepted, because a coincidental number would give a wrong name; a line at a result's value uses
  `at=`.
- **What catches a planted mislabel.** A new consistency check between the figure record and the delivered legend and
  caption text. Bar: at least 95% of planted mislabels caught, the same bar as the record-to-claim verifier.
- **What the no-model replay shows.** Placeholders left, caption mismatches, mislabels caught, and how many lines would
  be removed if no repair succeeded (the worst case). Whether repairs succeed is measured only by the model A/B.

## Decided by the maintainer (2026-11-06)

- **A line with no identity after the repair round is always removed**, a formula curve as much as a constant. There is
  no "drawn but unverified" form. `curve()` is therefore not kept for drawing that carries no claim: every drawn line
  goes through the records. The cost (theory overlays lost) is measured and reported, not used to keep such lines.

## Left to the builder

- The exact form of the registration in the analysis step (a new result kind, a column of an existing table, or
  `derive` output) — prefer extending what exists over a new name.
- How the `value=` match tolerates rounding (the plan's stated precision, not a fixed constant).
- Where in the prompts (`agents/*.md`) the figure step is told the new rule.
