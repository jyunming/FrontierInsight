---
title: Literature screening record
sources: [core/engine.py, core/config.py]
updated: 2026-09-30
---
# Literature screening record

For every source a literature search retrieves, FI keeps why it was kept or dropped, in `.fi/literature_queries.json` (next to the queries). Each literature pass adds two entries, and nothing is ever overwritten:

1. **Relevance floor**: each source's similarity score to the question (0–1). Sources below `knowledge.relevance_min_score` (0.20) are dropped, except that the best ones are kept to reach `knowledge.relevance_min_keep` (3). The reason is written out, for example "score 0.14 is below 0.20; kept to reach the minimum of 3 sources".
2. **Screen**: a model grades each source 0–3 (`knowledge.literature_screen`, on by default). Papers need a grade of 2; web pages are dropped only at 0. If grading fails, everything is kept and the record says so. Your own papers are never screened.

The file is covered by the trace's seal, so a verdict edited afterwards shows as a gap. No interface displays it; open the file in the quest folder (the web file list and zip skip `.fi/`).

Related: [[Oracle provenance]] (only sources the quest retrieved can back an expected value).
