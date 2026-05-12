You are the **Ideation Reflection** stage. You just brainstormed a set of research ideas and picked one. Before locking that choice in, critique it honestly.

# Topic
$topic

# Pre-flight clarifications
$clarify_block

# All ideas you brainstormed
$ideas_block

# The idea you initially picked
$chosen_block

# Your task

In one short pass, answer:

1. **Strongest objection** to the chosen idea — the single most likely reason it could fail or produce uninteresting results, given the clarify constraints.
2. **Would you pick differently?** Either confirm the original pick, or swap to a different entry from the brainstormed list above (use its exact `title`).
3. **Refined rationale** — if you're keeping the original pick, restate why in light of the objection.

Do NOT introduce a new idea that wasn't in the brainstormed list. Reflection means picking better from what you already wrote, not generating more.

# Output format

Respond with a single JSON object, no prose, no markdown fence:

```
{
  "strongest_objection": "<one sentence>",
  "swap_to": "<exact title from the brainstormed list, OR empty string to keep the original pick>",
  "refined_rationale": "<one sentence — what is the final reason for the locked-in choice>"
}
```
