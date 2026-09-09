# Skill selection — specification

Decided 2026-09-06 across twelve questions. Supersedes the blanket-injection
behaviour currently in `Engine._skills_block`.

---

## The flaw this fixes

Today every skill listed in `engine.skills` is injected into `design`,
`implement`, `implement_outline` and `implement_body`, unconditionally,
regardless of the topic. Three problems, the first measured:

**Cost scales with the library, and the library is meant to grow.** One skill
renders to ~1,860 characters (~465 tokens) and is sent to four nodes:

| skills | per quest pass |
|---|---|
| 5 | ~9,300 tokens |
| 20 | ~37,200 tokens |
| 50 | ~93,000 tokens |

That is the accumulation mechanism becoming the largest single token cost —
directly against the efficiency goal it was meant to serve.

**The prompt contradicts itself.** It tells the model "a skill used outside its
stated scope is worse than none, read its *when NOT to use* section first" —
then hands it every skill and asks it to show restraint. Putting a document-
conversion skill in front of a lithography design manufactures the misuse the
text warns about.

**The YAML list denies the premise.** Requiring the user to recall which skills
exist and which apply, per quest, defeats "FI remembers what it learned". It
gets worse as the library grows, which is backwards.

---

## Design

### Where it runs

Between **ideate** and **design**.

Selection sees the topic, the clarify answers, and the chosen research
direction. The direction matters: "compare dipole vs quadrupole illumination
for NILS" points at a skill far more precisely than the raw topic does.

Cost of being wrong: ideate has already run, so a bad selection wastes that.
Accepted — the matching quality is worth more.

### Mechanism

One LLM call against a **catalogue**, not the skill bodies. Each candidate
contributes name, kind, and its one-line `description` — roughly 68 characters.
A 50-skill catalogue is ~850 tokens, once, versus ~93,000.

This is why `description` is preserved from imported front matter: it is the
field the source author wrote to make the skill findable.

Semantic matching is the reason for an LLM rather than embeddings: "low-k1
imaging" has to match "aerial image formation", and the failure mode of a
keyword or embedding threshold is a **silent** miss with no signal.

### Candidates

`engine.skills` empty (default) → **every trusted skill is a candidate**. A
non-empty list restricts the candidate set (pin or narrow); it no longer means
"load these".

Only TRUSTED skills are ever candidates. Selection cannot promote anything.

### How many

**No cap.** Ranked by relevance, all the model considers applicable are used.

*Consequence carried deliberately:* cost is unpredictable again for broad
topics. Mitigation is visibility, not a limit — the run logs how many were
selected and the total block size, so an expensive selection is legible rather
than silent.

### Reason travels with the selection

`design` receives, per skill, why it was chosen, and is told it may decline a
skill it judges inapplicable — stating why in `method`.

Selection is one LLM call and will sometimes be wrong; design sees strictly
more context. Making the selection binding would remove the only correction
opportunity downstream.

### Survey / no-simulation mode

Library skills are skipped entirely — no experiment means nothing to compute.

Tool skills are still selected, but for a narrower reason than first written.
"A synthesis might need to read a document" was the wrong justification —
reading documents is the engine's own job, in `data_load`, and that reasoning
only looked sound because the example behind it (pandoc) should never have been
a skill at all. The real case is a literature quest needing to **query a source
FI cannot reach on its own**: a specialist database, an internal API, a
subscription service. That is external software being driven, which is what a
tool skill is.

This is decided in code, not delegated to the model, because "a library skill
is useless when no experiment runs" is a certainty, and routing a certainty
through a probabilistic call only adds failure modes.

### Determinism

Temperature **0 by default**, configurable.

Consistent with FI's other routing gates (`evidence_gate`, `claim_check`), and
required for `--resume` and any before/after comparison to mean anything: the
same topic must select the same skills, or a difference in output cannot be
attributed.

### When the call fails

Retry once. If both attempts fail, proceed with **no skills** — the behaviour
that existed before skills — and record it in the log and the quest output.

A quest must never fail because an additive step did not answer.

### Relevant but not trusted

If a skill matches the topic but is `untested` or `proposed`, the quest runs
without it **and says so explicitly**, in the log and in the quest output, so
the miss is visible and actionable:

```
[skills] selected -> ambit
[skills] note: 'semulator' matches this topic but is proposed
         (selftest passes, awaiting approval)
[skills] approve with: launch.py --approve-skill semulator --approve-as <you>
```

This closes the learning loop. Without it, a skill can sit unapproved forever
while every quest silently does without it.

---

## Provenance write-back

### When

Quest completed **and** review accepted. A paper that passed the numeric
oracle, the plausibility gate and review is the strongest available evidence
that the skill contributed something sound.

### What

Quest id, date, and outcome, appended to `taught_by_projects`.

`provenance.json` is excluded from the content hash (see `base.py`), so
recording use never lapses a skill's approval — which is exactly why the
exclusion exists.

### Concurrency

`--fleet` runs quests in parallel and several may use one skill at once.
Writes take a **file lock**, then read-modify-write, so no record is lost.

This is not hypothetical: FI is concurrency-first and `--fleet` is the primary
production invocation.

---

---

## Ranking, failure, and lifecycle

### History feeds ranking

The catalogue carries each skill's usage record — "used by 6 quests, 5
accepted" — and relevance is broken by it. A skill that has produced accepted
papers outranks an untried one of similar fit.

This is where the compounding actually shows: a skill that works gets reached
for more. The known risk is rich-get-richer — a genuinely better new skill has
to overcome an incumbent's record — which is why history breaks ties rather
than driving the ranking.

### A trusted skill whose self-test fails at use time

Skip that skill, run the quest, and say so. Consistent with every other
failure path here: a skill is additive, never a precondition. The log names
what broke and that re-approval follows a fix.

### Two skills claiming one domain

Send both. Selection does not arbitrate — it sees one line each, while
`design` sees the experiment and both skills' *when NOT to use* sections.
Arbitrating at the point of least information is the wrong place.

Useful side effect: if two skills genuinely cannot be told apart, that shows
up in `design`'s stated reason, which is the signal their descriptions need
sharpening.

### API drift

No new mechanism. The scaffolded self-test asserts its entry points exist with
the expected signatures, so a package upgrade that renames a parameter fails
the self-test and quarantines the skill through the gate that already exists.
Detection is automatic; repair is deliberate.

### Excluding a skill from one quest

`engine.skills_exclude: [name]`, separate from `skills`. Excluding one thing
should not require listing everything else — which matters as the library
grows, and which would also silently exclude anything added later.

Doubles as the A/B handle: run one topic with a skill and once without.

### Retirement

Reported, never automatic. `--skills` flags "never selected" and "quarantined
for N days"; nothing is deleted or disabled on FI's initiative.

Deletion is irreversible, and "never selected" often means the description is
written badly rather than the skill being useless — pruning on that signal
would delete the evidence needed to fix it.

---

## Distillation trigger (path 2, not yet built)

FI proposes turning work into a skill **only when the same approach recurs**
across several accepted quests. Recurrence is itself the evidence that
something is worth keeping; a one-off approach usually suits only that quest.

The threshold is deliberately unset here — too low and the prompt becomes
noise a person learns to dismiss, too high and it never fires. It should be
calibrated against real runs rather than guessed.

---

## Three-interface parity — scope

Web and VSCode need, in priority order:

1. **List and status (read-only)** — what exists, where each is stuck, why.
2. **Approve / revoke** — the gate's actual operating point, today
   terminal-only. Requires deciding how "who approved" is established in a
   browser.
3. **Teach / import** — scaffolding and adaptation.

Explicitly **not** in scope: surfacing the in-quest "you missed a skill"
notice in Web/VSCode. It stays in the log and quest output.

---

## What is being decided in code, not asked

- Selection naming a skill that does not exist or is not trusted → ignored and
  logged. Selection cannot promote.
- Catalogue includes each skill's *when NOT to use* first line as well as its
  description, if cheaply available — precision matters more than 40 characters.
- Selection lives in its own node rather than inside `design`, so a checkpoint
  resume does not re-run it and its cost appears separately in `cost.jsonl`.

## Open

- Web and VSCode surfaces for the "you missed a skill" notice. CLI only for
  now; three-interface parity is still outstanding for the whole skills
  subsystem.
- Whether a `--why-skills "<topic>"` dry run is worth having, to preview
  selection without running a quest.
