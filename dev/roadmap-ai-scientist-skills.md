# Roadmap: FI as a learning AI scientist

Clarified 2026-09-05, from two rounds of decisions. This supersedes the
"trusted kernel" recommendation in `research-accuracy-tools-tokens.md` §Convergence
and §2 (B1–B5). Everything else in that study stands.

---

## What changed, and why it matters

The study proposed a **curated** capability: register a vetted library once, teach
the prompts its API, stop re-deriving physics. That is a real fix for the token
and correctness leaks — but it is static. The registry is only ever as good as
what someone put in it, and nothing about running FI makes it better.

The decision is for an **acquired** capability instead: FI meets unfamiliar
software, works out how to drive it, designs experiments with it, and keeps what
it learned — with skills growing out of the accumulated corpus of projects done
jointly with the user, not from a curated list.

Curation does not compound. Learning does. That single difference is why the
"full pluggable" surface stops being over-engineering: if skills are acquired
rather than hand-registered, discovery has to work without anyone editing FI.

---

## The skills subsystem

### Envelope

One envelope at every maturity level, so nothing has to migrate:

```
skills/<name>/
  SKILL.md        always      — when to use it, workflow, gotchas
  api_surface.md  optional    — signatures fed to design/implement
  skill.py        optional    — executable entry points
  selftest.py     required for promotion (see gate)
  provenance.json always      — where this came from, which quests
```

Maturity ladder: **notes → guided → adapter**. A newly-learned software starts
as `SKILL.md` only; as its stable parts prove out across projects, they harden
into `skill.py`. FI prefers calling code when present and falls back to guided
generation when not.

### Distribution

Entry-point discovery, so a skill can live outside the FI repo and install from
anywhere:

```toml
[project.entry-points."fi.skills"]
ambit = "ambit.fi:Skill"
```

### Storage split (assumption — confirm)

- **Executable code** → filesystem + entry points.
- **Procedural knowledge and cross-project history** → Axon, which is already
  the cross-quest memory layer. This is what makes "based on previous projects
  worked together" mechanically possible: skill distillation mines the Axon
  corpus, not just the single quest that happened to finish.

---

## Acquisition — all four paths

All four are in scope. They are not alternatives; they are how a skill enters
the system at different stages of the relationship with a piece of software.

1. **Learned by exploration.** Point FI at a package or tool. It reads docs,
   introspects the API, runs probe experiments in the sandbox, and writes the
   skill from what it observed. The most autonomous path, and the one that
   carries the "learn to manipulate a new software" claim.
2. **Distilled from successful quests.** After a quest passes its gates, FI
   extracts the reusable procedure from what actually worked and writes it back.
   The platform improves by being used.
3. **Authored by the user.** Highest quality, and the seed for the first skills
   before the loop can bootstrap.
4. **Demonstrated once, generalised.** A working script or session transcript
   becomes a reusable skill. Cheap, and grounded in something known to work.

Paths 2 and 4 are where "a learning process with the user" actually lives —
both take the joint working history as their raw material.

---

## Promotion gate — self-test AND approval

Both are required. A skill is not trusted until it passes its own executable
check *and* a person signs off.

```
FI learns / distills a skill
        ↓
  writes selftest.py with known-good expected values
        ↓
  status: PROPOSED  ──fails selftest──→  QUARANTINED (never loaded)
        ↓ passes
  fi skills review <name>
    [SKILL.md diff] [selftest output] [probe runs] [provenance]
        ↓ approve
  status: TRUSTED — loadable into quest prompts
```

Consequences, stated plainly so they are chosen rather than discovered:

- **A skill that cannot write its own test is never promoted.** Writing the test
  is part of learning the software, not a follow-up chore.
- **The autonomy is in the learning, not the adoption.** FI can explore and
  propose unattended; it cannot grant itself new capability. This deliberately
  mirrors mooting's invariant — the machine deliberates, a person decides.
- **A `--fleet` run cannot use a PROPOSED skill.** It falls back to guided
  generation. Unattended runs never silently adopt unapproved capability.
- The human gate caps how fast the library grows. That is the accepted cost.

### This is one philosophy, not three features

The promotion self-test, the numeric oracle (A2) and the physics-plausibility
gate (A3) are the same idea applied at three points: **a deterministic check
that does not route back through a language model.** The study found that every
existing correctness gate terminates in an LLM reading text. These three are the
correction.

A3 gets better under this model than under the kernel model. The study justified
range and unit assertions by "a kernel knows its own valid domain" — but a
*skill* knows it too, and carries the assertions in the same envelope as its
self-test. One artifact declares what the software does, how to drive it, how to
prove it still works, and what its outputs may legally be.

---

## First targets — both

- **Own software**: ambit as the known-good control (correct answer known, tests
  pass, prior FI quests on the same physics to compare against), then SEMulator
  and gds2sem as genuinely unfamiliar but verifiable.
- **Third-party**: software neither FI nor the user controls — unfamiliar
  vendor conventions, install and licensing, possibly GUI-only, docs of unknown
  quality. This is the real proof of the claim.

Running both matters: the control tells you whether the loop works at all, and
the third-party target tells you whether it works on anything that wasn't
already in the family.

---

## Also in this round (unchanged from the study)

| ID | Work | Note |
|---|---|---|
| A1 | `review_panel: [methodologist, statistician, devil_advocate]` in configs | One line. Activates dormant must-flag enforcement. |
| C1 | Reorder every template: static instructions first, variables last | Mechanical, no behaviour change. |
| A2 | Deterministic numeric oracle, `paper.md` vs `result_json` | Raises `unverified_number` as a **blocking must-flag** → forces a revise pass. |
| A3 | Widen `degenerate_run_guard` into a physics-plausibility gate | Assertions now carried by skills. |
| D2 | `fi review --council` — interactive second opinion, never in `--fleet` | Neutral `cwd`; council recommends, human accepts. |
| D3 | Seeded-defect measurement: single reviewer vs 3-persona panel vs council | Produces the evidence the "better outcomes" claim is waiting on. |

### Explicitly out

- **Kernel track (B1–B5) as a curated registry.** Superseded by skills.
- **HTTP-direct Anthropic provider (C3).** Staying on CLI/VSCode subscriptions:
  no API keys, no per-token billing. C1 only on the provider side.

Consequence to keep in view: with C3 out, token accounting stays a
~4-chars-per-token estimate on your primary transports (study §3, C5). Good for
ranking nodes; not sound for measuring whether C1 helped. Any before/after claim
about token savings needs that caveat attached.

---

## Sequence

1. **A1** — one line, today.
2. **C1** — template reorder. Mechanical, unblocks nothing else but costs nothing.
3. **A2 + A3** — the deterministic oracle pair. Establishes the philosophy the
   skills gate will reuse, on a smaller surface.
4. **Skills envelope + entry-point discovery + promotion gate.** Machinery only,
   no learning yet — seeded by path 3 (you author ambit's skill by hand).
5. **Path 4 then path 2** — demonstrate-once, then distil-from-quests. Both mine
   existing joint work, so they pay off immediately on the projects already done.
6. **Path 1** — learned by exploration, first on SEMulator/gds2sem, then a
   third-party target.
7. **D3** — the measurement experiment, once 1–6 give it a stable baseline.

Steps 1–3 are the study's original quick wins and are independent of the skills
work. Step 4 is the fork where this roadmap departs from the study.

---

## The Axon coupling, after their reply

Axon's team answered the requirements doc
(`docs/architecture/FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md`, branch
`docs/fi-skill-memory-response`). Their findings were verified against the Axon
tree here rather than taken on trust. Three things change.

### FI keeps the word "skill"

The requirements doc offered to rename, since Axon's usage was older and
published. Axon declined the offer: `docs/SKILLS.md` and `docs/MCP_TOOLS.md`
document the same MCP tools, the former claims 30 tools when there are 56 and
soon ~13, and "skill" was never accurate for what those entries are. Axon will
retire the term during its documentation consolidation. **No rename here.**

### The staleness requirement was unfounded

`get_stale_docs` is a read-only report. No retention job exists anywhere in
Axon; deletion is only ever explicit. There is nothing for skill knowledge to be
exempt from, so that requirement is withdrawn.

Worth knowing in the other direction: the report is built from `_source_hashes`,
an in-process dict that resets on server restart, so a corpus ingested last week
never appears in it regardless of age. If FI ever wants genuine staleness
signals over a long-lived corpus, no current mechanism provides them.

### My cross-project premise was wrong about our own layout

The requirements doc argued that "each quest is naturally its own project, so
the query is inherently many-project." **FI does not work that way.**
`core/knowledge.py:189` pins a single project — `frontier-insight`, overridable
via `FI_AXON_PROJECT` — and `:2518` switches to it. Every quest writes into that
one project.

So the expensive Axon-side requirement, many-project retrieval, **is not
currently needed at all**. A single-project query already spans every quest.

What FI actually needs is *per-quest attribution inside one project*. That is
document metadata, not a project boundary — and FI already writes it:
`add_quest_artifacts(quest_id, ...)` lands `fi_*` documents carrying quest
identity, with a `fi_topic_event` keyed by topic slug. The open question is
narrower than the one asked: **does that metadata survive retrieval intact?**

Two consequences:

- The distillation step (path 2) should be designed against **one project with
  rich per-document metadata**, not a fan-out across many. Simpler than planned.
- Axon's same-id merge defect (`vector_store.py:887-890`, which drops colliding
  documents from different sub-stores and tags no origin) only bites if FI later
  moves to project-per-quest. It is a real defect and Axon intends to fix it on
  its own merits, but it is no longer on FI's critical path.

If project-per-quest ever becomes desirable, Axon already supports hierarchy —
`subs/` directories with `parent/child` names (`src/axon/projects.py:213-223`) —
and a parent project already behaves as one retrieval namespace over its
children. That is the "first-class project group" shape, and it exists today.

---

## Open — not blocking, but decide before step 4

- **Does quest metadata survive retrieval?** The one Axon-side question that
  still matters, and the cheapest to answer: ingest a probe document with known
  metadata, retrieve it, inspect what comes back.
- **One project or project-per-quest?** Staying single-project keeps
  distillation simple and avoids the merge defect entirely. Splitting buys
  per-quest isolation and lands on Axon's parent/child machinery. Default to
  staying put until something forces the change.
- **Storage split**: is Axon the right home for procedural knowledge and
  provenance, with only executable code on the filesystem? Still an assumption.
- **Skill scope granularity**: one skill per *software*, or per *capability*
  (a package might expose "aerial image" and "CD metrology" as separate skills
  with separate self-tests)?
- **Three-interface parity**: `fi skills review` is a CLI affordance. Parity
  requires a Web and VSCode surface for approving skills; that is real UI work
  and should be scoped before step 4, not after.
- **What a skill self-test asserts for stochastic or long-running software** —
  tolerance policy, seeded runs, or a cheaper proxy check.
