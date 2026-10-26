# A search whose objective never moves goes back to the plan once

Status: approved by the maintainer in conversation (2026-10-21); this document is the written form.

## Why

A live search for the best design spent its whole budget (6000 evaluations, nearly two hours) on an objective that was
0 for every design it tried. The coarse scan before it had already shown the same thing. The cause was the plan's own
choice of how the design is represented: the region of the design space where the objective is above 0 is narrower
than one element of the representation, and every element that touches it also covers a region that drives the
objective back to 0. Designs finer than the plan's representation reach the objective the topic asks for (measured on
the quest's own simulation: a common working range of 35-45 nm where every element of the plan's representation
gave 0). So the study was possible; the plan made it impossible, and FI never told the plan what it had measured.

FI first learned to stop such a search early and report it (`stopped_because: flat_objective`). That is right for a
search that truly has nothing to find, and wrong for hard research, where the answer exists but the first plan cannot
reach it. The research goal is the hard result, so a flat objective must first go back to the plan.

## Goals

- When FI has measured that the objective does not move across the design space, the plan is told, in FI's own
  measured terms, and gets one chance to change the design so that its objective can be optimised.
- The fact is FI's: counts and values from FI's own record of the evaluations, never a guess from the code or the
  model's prose.
- The check fires as early as FI has the evidence (the coarse scan, or the search's global stage), not after the
  budget is spent.
- When the revised design is still flat, the search runs its whole budget (no early stop) and the paper says plainly
  that the objective did not move in the range tried and that FI already revised the design once.

## Non-goals

- FI does not decide how to change the design. It never names a representation, a resolution or a range: the plan
  does, from the measured fact and the topic.
- No new rule for one kind of objective or one field of research.
- Nothing changes for a search whose objective moves.

## Anti-goals (a failure even if it works)

- A revision that moves the goalposts: dropping what the topic asks for, or redefining the objective so that a
  number above 0 appears without the study answering the topic.
- A revision recorded as if it had been planned before the run. It happened after results were seen, and the record
  and the paper must say so.
- A new config key, CLI flag or term a scientist has to learn.

## Constraints

- The revision counts against the existing `engine.max_iterations` budget (the loops that send a quest back to the
  design already share it). At most one revision for this reason per quest.
- Under explore-then-confirm (`engine.phased`), the confirm stage runs a frozen design once: no revision there (as for
  the analysis's own request to redesign).
- The redesign is a change made after results were seen. It goes through FI's existing route for that
  (`protocol_amendment` with its reason, kept in the record), and the writer is told the first design, what FI
  measured, and what was changed.
- The first design's evaluations are kept and reported; nothing is deleted.

## Decided

1. **When.** As soon as FI holds `NO_EFFECT_DESIGNS` (50) designs sampled across the whole range (the coarse scan's
   settings under the conditions the search holds, and the global stage's designs), which between them moved every
   number of the design, and all gave the starting design's objective and met the limits as it did. This is the
   existing `flat_objective` watch in `core/optimise_search.py`; only what happens next changes.
2. **What is sent.** The number of designs and their common objective value; that the starting design gave it too;
   and which of the results the simulation returns stayed at the starting design's value in every design and which
   moved. When the objective is built from parts (the minimum over several cases, a sum of terms), the plan is asked
   (in the plan prompt) to return each part as a result, so that the list shows which part is stuck.
3. **How often.** Once per quest, within `engine.max_iterations`.
4. **Still flat after the revision.** The search runs its whole budget; the record and the paper say the objective did
   not move in the range tried and that the design was already revised once. The starting design being in the flat
   region is part of the fact sent in 2, not a rule of its own.

5. **Who approves the change to the frozen protocol.** FI approves this one amendment itself (the maintainer's
   decision after the builder check): it is recorded as not pre-specified, made after results were seen, for the
   reason FI measured (the flat objective, with its numbers); the paper says so and the evidence ladder names the gap.
   A person is not waited for, so unattended and fleet runs get the revision too. The revised design is still held to
   the topic like any design drafted at the design step (a revision that drops or eases a case the topic asks for is
   asked again, as today).

## Calls made by the author (not asked of the maintainer)

- The first design's run is archived as today's approved amendments are (`archive_run`): it is not evidence for the
  paper's claims, and the paper's limitations say that the first design's objective did not move, what FI measured
  and what was changed. The two rules do not conflict: archived means not evidence, never hidden.
- A quest whose protocol is not frozen yet (the explore stage of `engine.phased`) is revised the same way without an
  amendment; its `design_history` entry still says the change was made after results were seen.
- FI keeps its own tally of which results the simulation returns stayed at the starting design's value across the
  flat designs (new search state, kept in the search's record and rebuilt from the ledger on a resume), because
  `_watch_effect` stops tracking results once one moves. Without any result beside the objective, the fact is the
  count, the common value and that the starting design gave it too.
- The "revised once" marker is per quest, kept in the quest's state (so it survives `--resume` and the repair loops):
  after it, the flat watch still runs and is recorded, but it no longer stops the search or asks for a revision.

- The threshold stays `NO_EFFECT_DESIGNS`; no new number.
- No new plan field for the objective's parts: the parts are ordinary results the simulation returns, and FI already
  knows which results moved (`_watch_effect` keeps the baseline's values of every result the plan names). Only the
  plan prompt gains one sentence.
- The route is the existing `redesign` edge to `design`; no new node and no new edge label.

## Left to the builder

- Where in the graph the `flat_objective` outcome is turned into the `redesign` route (the execute result, the
  cross-check route or the evidence gate), as long as the rules above hold and the existing bounds count it.
- The exact words of the fact in the design prompt and in the paper's limitations, following the plain-language rule
  of `AGENTS.md`.
- How the second search's record names the first one.

## Evidence before building

A small test with the plan's own model on the real quest's plan and the measured fact (two calls), scored on two
axes: does the revised design make the objective move, and does it still answer the topic. Results are recorded with
the test (<run folder>/ab_redesign, outside the repository). The feature is built only if the
revision is both.

Result (2026-10-21, the plan's model, two calls, the topic as the next run will state it):

- Both revisions made the objective move: each replaced the pass/fail objective by a graded one that still reports
  the strict quantity as the result, and each kept the representation inside the physical range. Measured on the
  quest's own simulation, that range change alone lifts the share of random designs above 0 from 1 in 200 to 9 in 200,
  and the simplest edge designs from 0 to about 33 (the topic's unit).
- One revision kept every case the topic asks for. The other moved its hardest case to an easier one and stopped
  there, while saying it followed the topic: the goalpost-moving the anti-goals name. So the revision must be held to
  the topic as any design drafted at the design step already is (the existing check that a redesign keeps what the
  topic asks for), and the paper must say what was changed.

## How it will be shown fixed

- A replay of the real quest's record: the flat watch fires at the coarse scan or the global stage, and the quest
  routes to `design` once, with the fact in the design prompt.
- A known positive: a search whose objective moves is untouched (the replays of the other real searches keep their
  outcomes).
- A live run of the same topic on the new code.
