# Review-node rubric

The review node grades each paper on a 1–5 scale and returns a verdict of either `accept` or `revise`. It is intentionally **strict on null/negative results being honestly reported** and lenient on prose polish — downstream formatting will smooth surface issues.

## Score guide

| Score | Meaning |
|---|---|
| 5 | Hypothesis tested cleanly; results are self-consistent; figures are legible; references are concrete; no factual contradictions. |
| 4 | One minor issue (e.g., a figure could be clearer, or one claim is under-supported). Acceptable. |
| 3 | Multiple minor issues, or one moderate issue (e.g., a numerical claim doesn't match the stdout RESULT_JSON). Requires revision. |
| 2 | Significant problem (e.g., conclusions overreach the data, hypothesis isn't actually tested, broken figure). Requires revision. |
| 1 | Paper does not constitute a research output (e.g., no figures, no numerical results, hallucinated experiment). Requires revision. |

## Verdict mapping

- Score 4–5 → `accept`
- Score 1–3 → `revise` (only if `engine.review_loop: true` and `iteration < max_iterations`; otherwise the engine will accept the highest-scored revision so far rather than loop forever)

## Style

The review node is the only node where blunt criticism is the desired behavior. Hallucinated citations, overclaiming, and unstated limitations are revise-triggers. Cosmetic prose issues are not.
