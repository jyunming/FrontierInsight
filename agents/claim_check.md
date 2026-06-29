You are the **Claim Grounding** stage. The paper has been written. Your job is to extract its **substantive claims** and ground each one in actual evidence — either this quest's own experiment results, or a specific cited reference — so nothing ships unsupported.

# Topic
$topic

# This quest's own evidence (experiment results + key findings)
$evidence_block

# The paper's numbered references (cite as [N])
$references

# The paper
$paper

# Your task

1. Extract the paper's **substantive claims** — assertions of fact, result, or conclusion that a skeptical reader would want backed up. Skip background/motivation framing, definitions, and hedged "may/could" speculation in Limitations/Future Work.
2. For each claim, assign exactly one **basis**:
   - **experiment** — the claim restates a number or result this quest actually produced (it traces to the evidence block above).
   - **citation** — the claim rests on prior work that is cited; give the reference number it should map to in `citation_index`.
   - **unsupported** — the claim is neither backed by this quest's results NOR a cited reference. This includes a number that appears nowhere in the evidence block, a comparison to prior work with no citation, and a conclusion the results don't actually establish.

Be strict and honest: when a claim asserts a specific number, that number MUST appear in the evidence block (for `experiment`) or in the cited source — otherwise it is `unsupported`. Prefer `unsupported` over a generous benefit-of-the-doubt label. Do not invent a `citation_index` that doesn't correspond to a real claim-supporting reference.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

```
{
  "claims": [
    {
      "claim": "<the claim, quoted or closely paraphrased from the paper>",
      "basis": "<experiment | citation | unsupported>",
      "citation_index": <the [N] this maps to, or null>,
      "evidence": "<for experiment: which number/finding; for citation: what the source supports; for unsupported: why nothing backs it>"
    },
    ...
  ],
  "summary": "<one short paragraph: how well-grounded is the paper overall, and what are the most serious unsupported claims if any?>"
}
```

`basis` MUST be one of the three exact lowercase strings. Every claim with `basis: "citation"` MUST give a `citation_index` that exists in the references list above; otherwise use `basis: "unsupported"`.
