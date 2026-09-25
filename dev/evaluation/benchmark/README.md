# Benchmark scores (for debugging)

FI is graded by hand on one fixed topic, a stochastic SIR epidemic study, run again and again on different builds
and different models. The grades live here:

- [rubric.md](rubric.md) — what is graded, and how.
- [scores.md](scores.md) — every graded run: its six dimension scores, the three gates, pages and tokens.

## What these scores are for

They find **FI's faults**. A run that loses points usually does so because a check let something through (a number
the run never computed, a figure its text misdescribes, a citation its source does not support) or stopped something
correct. Each graded run lists those faults, and most fixes to FI's checks began as one.

## What they are not

**They are not a ranking of models.** Read a comparison between two arms with care:

- **One topic.** Every run studies the same SIR question. A model that does well here may not do well elsewhere.
- **Few runs.** Most arms have one to three runs; a difference of a few points is within the spread between two runs
  of the same arm.
- **FI changed between runs.** Each row names the FI build it ran on. Rows on different builds measure FI as much as
  the model.
- **Graded by hand.** The judged items (does the source support the sentence, does the figure match its text) are one
  grader's reading. Where a reasonable grader could decide an item the other way, the run records the alternative
  score.
- **The rubric changed.** Every change is dated in [rubric.md](rubric.md); every run in `scores.md` is re-scored on the
  rubric now in force.

## Adding a run

Grade it by [rubric.md](rubric.md), add a row to [scores.md](scores.md) with the FI build it ran on, and list the FI
faults it showed.
