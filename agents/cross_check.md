You are the **Cross-Paper Check** stage. The analysis just produced a specific finding. Your job is to classify the surrounding literature into three buckets — **supporting**, **conflicting**, **neutral** — relative to this finding, so the paper can cite both agreement and disagreement honestly.

# Topic
$topic

# Finding under check
$finding

# Candidate literature retrieved by searching for the finding
$candidate_literature

# Your task

For each numbered candidate above, decide one of:

- **supporting** — the cited paper reports a result consistent with the finding, or makes a claim that the finding would corroborate.
- **conflicting** — the cited paper reports a result that contradicts the finding, or claims something the finding would refute.
- **neutral** — the paper is topically related but does not weigh in either way (e.g., it studies a different regime, uses an incompatible metric, or its abstract is too thin to judge).

Be honest: prefer **neutral** over a forced supporting/conflicting label when the abstract doesn't actually weigh in.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

```
{
  "verdict": "<supporting | conflicting | neutral | mixed>",
  "supporting":  [{"index": <1-based>, "why": "<one-sentence explanation>"}, ...],
  "conflicting": [{"index": <1-based>, "why": "<one-sentence explanation>"}, ...],
  "neutral":     [{"index": <1-based>, "why": "<one-sentence explanation>"}, ...],
  "summary": "<one short paragraph: does the literature broadly agree, disagree, or remain inconclusive?>"
}
```

**`verdict` semantics — pick exactly one based on the balance of the literature:**

- **`supporting`** — the literature, on balance, supports the finding (e.g. supporting citations clearly outweigh conflicting ones, or the only signal at all is corroborating).
- **`conflicting`** — the literature, on balance, contradicts the finding (e.g. conflicting citations clearly outweigh supporting ones).
- **`mixed`** — there is meaningful evidence on BOTH the supporting and conflicting sides, roughly balanced, and a single direction can't be honestly chosen.
- **`neutral`** — the literature is mostly silent / topically tangential; neither supporting nor conflicting evidence is substantive enough to weigh in.

The `verdict` MUST be one of these four exact lowercase strings. It is the single field downstream ensembles tally to pick a majority opinion across multiple models — keep it crisp and consistent with the supporting/conflicting/neutral lists you populated above.
