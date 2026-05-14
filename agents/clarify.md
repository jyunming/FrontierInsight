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
  // NOTE on `empirical` as a default: pick "empirical" only when the
  // research question is genuinely answered by REAL-WORLD DATA the
  // user must collect themselves (qualitative comparisons across
  // cultures, history, sociological surveys, archival document
  // analysis, etc.). The engine treats "empirical" as a routing
  // signal — it skips the simulation half of the pipeline, pauses
  // after `design`, and asks the user to drop their collected data
  // into `<quest_root>/data/`. Use "theoretical" for closed-form /
  // analytic derivations and "mixed" for hybrid simulation + closed-
  // form work; both keep the simulation path active.
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
    "question": "<one sentence asking which paper template / venue style this study fits — generic, neurips, iclr, ieee_access, or nature_mi>",
    "default": "<your best guess; see venue rules below>"
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
- The `paper_venue` default is one of `{generic, neurips, iclr, ieee_access, nature_mi}` — pick by topic + study_depth + empirical_vs_theoretical:
  * `neurips` / `iclr` — ML benchmarks, learning algorithms, neural-network experiments (empirical + journal-length).
  * `ieee_access` — engineering systems, hardware/software architectures, measurement studies (empirical + journal-length).
  * `nature_mi` — physics / chemistry / materials simulation, scientific-method experiments (empirical + journal-length).
  * `generic` — surveys, comparative reviews, theoretical derivations, brief preprints, anything that doesn't fit a specific venue. **DEFAULT to `generic` when uncertain.**
- No prose outside the JSON object. No code fences. No commentary.
