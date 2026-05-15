You are judging a pairwise comparison between two research ideas
brainstormed for the same topic. Pick which is more promising as the
basis for a research quest that will run end-to-end (literature →
design → experiment → analysis → paper).

# Topic

$topic

# Clarify answers

$clarify_block

# Idea A

```json
$idea_a
```

# Idea B

```json
$idea_b
```

# Output (single JSON object, no prose outside it)

```
{
  "winner": "A" or "B",
  "reason": "<one short sentence explaining the decisive factor>",
  "margin": "decisive" or "narrow"
}
```

# Judging criteria (weighted, in priority order)

1. **Empirical tractability** — given the user's `budget` and the
   measurement plan implied by the idea, can the experiment actually
   run in the budgeted compute? An idea that's beautiful but
   demands GPU-weeks is dominated by a slightly less novel idea that
   completes in an hour.

2. **Novelty over the chosen comparative baseline** — does the idea
   actually move past the baseline named in `comparative_baseline`?
   Re-deriving an established result with a new framing is worse
   than a small but genuine delta.

3. **Methodological cleanness** — does the idea afford a clean
   independent / dependent variable split? Ideas with too many
   confounders are dominated by tighter framings.

4. **Falsifiability** — picks a hypothesis that COULD be wrong.
   Vague "explore the relationship between X and Y" framings are
   dominated by sharp "X causes Y by mechanism M, measured by Z"
   framings.

# Constraints

- Pick exactly one winner. Ties are not allowed — break with criterion 1.
- `reason` must cite which criterion was decisive.
- `margin: "decisive"` if criterion 1, 2, OR 3 is unambiguous.
  `margin: "narrow"` when the two ideas only differ on falsifiability
  or on a soft criterion. The aggregator uses margin to break
  tournament-level ties; narrow wins still count as one vote.
- No prose outside the JSON object. No code fences. No commentary.
