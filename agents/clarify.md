You are scoping a research quest BEFORE the autonomous loop begins.
Given only the topic below, your job is to produce a short structured
questionnaire whose answers will sharpen every downstream prompt
(`ideate`, `design`, `implement`, `analyze`, `write`).

# Topic
$topic

# Required output (single JSON object, no prose, no fences)

```
{
  "comparative_baseline": {
    "question": "<one sentence asking what existing method / dataset / regime this study should be compared against>",
    "default": "<your best guess answer, derived from the topic alone>"
  },
  "empirical_vs_theoretical": {
    "question": "<one sentence asking how this question gets answered: by writing a Python simulation, by closed-form derivation, by mixing both, OR by the user collecting real-world data (surveys, interviews, field notes, archival sources) that a Python script could not invent>",
    "default": "empirical" or "theoretical" or "mixed"
  },
  // NOTE — this slot is now METHODOLOGY-ONLY. It is no longer the
  // primary routing signal for no-simulation mode: that responsibility
  // moved to `simulatability` below. The engine still falls back on
  // `empirical_vs_theoretical == "empirical"` ONLY when the new
  // `simulatability` slot is missing entirely (legacy back-compat).
  // When BOTH slots are present, `simulatability` wins. So if you
  // pick `empirical` here, you should normally also pick `no` on
  // `simulatability` — otherwise the engine will simulate (because
  // simulatability=yes/uncertain beats empirical_vs_theoretical=empirical),
  // and your two answers will read as contradictory in run.log.
  "simulatability": {
    "question": "Could a self-contained Python script (numpy/scipy/matplotlib, ~100 lines, no internet) produce data that genuinely answers this research question? Or does the question require real-world observation that cannot be simulated without inventing values?",
    "default": "yes" or "no" or "uncertain",
    "reason": "<one-sentence justification for the chosen default. If 'no', name what kind of real-world data would be needed.>"
  },
  // CRITICAL — the engine uses `simulatability` as the ROUTING SIGNAL
  // for the no-simulation flow. This slot is NEW and supersedes the
  // older "empirical_vs_theoretical → empirical" coupling, which
  // conflated two different questions (methodology type AND Python's
  // reach). Decision logic:
  //
  //   simulatability == "no"        → engine routes to wait_for_data
  //                                    (no Python script can answer this;
  //                                     pause for the user / agent to
  //                                     collect real data).
  //   simulatability == "yes"       → engine runs the normal
  //                                    implement → execute pipeline.
  //   simulatability == "uncertain" → engine runs the normal pipeline
  //                                    BUT also includes a "this may
  //                                    have produced invented data"
  //                                    caveat in the review prompt.
  //
  // Pick "no" when the question is about: cultural attitudes, history,
  // social phenomena, biographies, archival document analysis, current
  // events, qualitative comparisons across populations. The user is
  // running an empirical study; a Python script would have to make up
  // numbers.
  //
  // Pick "yes" when the question can be answered with: physics
  // simulation, closed-form math, algorithmic experiments,
  // statistical bootstrapping, synthetic-data experiments with known
  // generators, computational benchmarks. The user is running a
  // computational study; Python can produce real evidence.
  //
  // Pick "uncertain" when the topic could plausibly be either; the
  // engine will run the simulation path but flag the result for
  // extra review scrutiny.
  //
  // The `reason` field is REQUIRED — it gets logged at routing time
  // so the user can see exactly why the engine took the path it did.
  // Actual run.log format:
  //   [clarify] simulatability resolved: NO_SIMULATION
  //   (source=clarify_simulatability, reason='<your one-liner here>')
  // (or SIMULATE / source=yaml / source=clarify_empirical_legacy /
  // source=default depending on the resolved branch — see
  // core/engine.py:_resolve_no_simulation_from_clarify).
  "success_metric": {
    "question": "<one sentence asking what number changing in what direction would count as the headline result>",
    "default": "<your best guess of the metric + direction>"
  },
  "budget": {
    "question": "<one sentence asking the soft cap on wall-clock for the experiment>",
    "default": "<your best guess: e.g. 'a few minutes on a laptop CPU', 'one GPU-hour', 'analytic — no compute'>"
  },
  "output_kinds": {
    "question": "<one sentence asking which deliverables matter (paper, slides, poster, speech)>",
    "default": ["paper_md"]
  },
  "study_depth": {
    "question": "<one sentence asking how mature the resulting paper should be — brief preprint vs journal-length vs comprehensive review>",
    "default": "journal-length"
  },
  "paper_venue": {
    "question": "<one sentence asking which paper template / format style fits — scientific venues (generic, neurips, iclr, ieee_access, nature_mi) or non-scientific prose (essay, report, policy_brief, whitepaper)>",
    "default": "<your best guess; see venue rules below>"
  },
  "topic_shape": {
    "question": "<one sentence asking which intellectual SHAPE this topic has — a specific testable comparison (experimental), a broad synthesis of prior knowledge (review), one system / phenomenon to study in depth (case_study), or a position to argue (opinion)>",
    "default": "experimental" or "review" or "case_study" or "opinion",
    "reason": "<one-sentence justification — quote which words in the topic point to this shape>"
  }
}
```

# Constraints
- ALWAYS supply a non-empty `default` for every field. The default is
  what the agent will use if the user does not override it — so the
  default must already be a workable starting point.
- Questions must be specific to the topic. Avoid generic phrasings like
  "what is your goal?" — the topic should make the question concrete.
- The `output_kinds` default is a list whose entries are drawn from
  `paper_md`, `paper_pdf`, `slides`, `poster`, `speech`.
- The `study_depth` default is one of:
  `brief preprint` (1–2 pages, terse, novel findings only) /
  `journal-length` (4–8 pages, full IMRAD, ~15 citations) /
  `comprehensive review` (10–15 pages, synthesis with extensive prior-work discussion).
  Pick the level that matches the topic's natural scope — narrow benchmarks → brief, established
  research questions → journal-length, broad survey/comparative topics → comprehensive review.
- The `paper_venue` default is one of `{generic, neurips, iclr, ieee_access, nature_mi, essay, report, policy_brief, whitepaper}` — pick by topic + study_depth + empirical_vs_theoretical + simulatability:
  * **Scientific venues** (when `simulatability=yes` AND the topic is computational/experimental):
    - `neurips` / `iclr` — ML benchmarks, learning algorithms, neural-network experiments (empirical + journal-length).
    - `ieee_access` — engineering systems, hardware/software architectures, measurement studies (empirical + journal-length).
    - `nature_mi` — physics / chemistry / materials simulation, scientific-method experiments (empirical + journal-length).
    - `generic` — surveys, comparative reviews, theoretical derivations, brief preprints, scientific topics that don't fit a specific venue.
  * **Non-scientific prose formats** (when `simulatability=no` OR the topic is qualitative/historical/cultural/business):
    - `essay` — long-form argumentative prose. Cultural comparisons, historical analyses, intellectual history, qualitative cross-case studies. Picks when the natural shape is "argue a thesis with evidence", not "report a measurement".
    - `report` — consulting-style executive report with cover + TOC. Business analyses, operational reviews, market assessments. Picks when the audience is decision-makers and the structure is exec-summary → findings → recommendations.
    - `policy_brief` — 2-4 page brief for policymakers. Single decision recommendation backed by issue + context. Picks when the natural shape is "here's the issue, here's what to do, here's why."
    - `whitepaper` — industry/vendor-neutral 8-20 page analysis. Tech trends, architecture comparisons, standards interpretations. Picks when the audience is industry practitioners and the structure is problem → approach → evidence → conclusion.
  * **DEFAULT to `generic` when uncertain** for scientific topics; **DEFAULT to `essay`** for non-simulatable topics that fall in humanities / social-science / archival / current-events.
- The `topic_shape` slot orthogonally classifies the topic's INTELLECTUAL SHAPE — separate from `simulatability` (which asks "can Python answer this?"). Four shapes:
  * `experimental` — the topic names a specific testable comparison ("compare X vs Y on benchmark Z", "does method M reduce error E"). Hypothesis + variables + measurable outcome. Default for most topics that fit `simulatability=yes`.
  * `review` — the topic asks for a synthesis of prior knowledge across a field ("OPC importance in low-k1 era", "history of resolution enhancement", "differences between A and B"). The right output is a literature synthesis, not a toy experiment masquerading as a benchmark. Trigger words: "importance", "history", "evolution", "differences", "review of", "landscape of", "trends in".
  * `case_study` — the topic studies one system / one event / one phenomenon in depth, with no comparator and no sweep ("why did chip X fail at process node Y", "the 2018 PUE incident in datacenter Z"). Trigger words: "the X", "this Y", proper nouns dominating.
  * `opinion` — the topic argues a position with no decisive empirical answer ("should we adopt MLOps", "is the AI safety field underinvested", "what should our team do about Z"). Values + reasoning, not measurements. Trigger words: "should", "is X worth it", "what to do about".
  - When `topic_shape` is `review`, `case_study`, or `opinion` BUT `simulatability` is `yes`, you have a mismatch — the engine WILL run a Python experiment (per simulatability) but the topic doesn't want one. Surface this honestly: keep `topic_shape` accurate; the engine will log the mismatch and the design / write stages will keep the experiment minimal and shift weight to the literature synthesis.
- No prose outside the JSON object. No code fences. No commentary.
