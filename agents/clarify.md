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
    "question": "<one sentence asking whether this study runs code and measures something, vs. derives results analytically>",
    "default": "empirical" or "theoretical" or "mixed"
  },
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
- No prose outside the JSON object. No code fences. No commentary.
