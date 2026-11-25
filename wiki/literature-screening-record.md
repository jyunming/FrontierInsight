---
title: Literature screening record
sources: [core/engine.py, core/config.py, core/retractions.py]
updated: 2026-11-25
---
# Literature screening record

For every pass of the literature step FI keeps what it found out about the sources, in `.fi/literature_queries.json`. FI no longer searches by topic, so there is no relevance floor or screen; the one stage left is the retraction lookup, and nothing is ever overwritten:

1. **Retractions**: after de-duplication every DOI is looked up in Crossref (Retraction Watch data included). Each source is `retracted`, `not_retracted`, `not_checked` (no answer, Crossref does not hold the DOI, an arXiv DOI (not sent), or a DOI that cannot be read; never read as "not retracted") or `no_doi`, with the notices Crossref lists. A retracted source is marked `[retracted]` in the prompts, and any sentence of the paper that cites it is marked unsupported by the claim check; `run.log` gets one line.

The file is covered by the trace's seal, so a verdict edited afterwards shows as a gap. No interface displays the file itself; open it in the quest folder (the web file list and zip skip `.fi/`). The to-do card on every interface names any retracted source.

Related: [[oracle-provenance|Oracle provenance]] (only sources the quest retrieved can back an expected value).
