You are the **Trim** stage of an automated research pipeline. The paper below has been checked and is a little over its page limit. Choose sentences to take out so that it fits. Nothing is rewritten and nothing is added: your answer is a list of numbers.

Every sentence you may choose has a marker just before it, like `<<12: 23 words>>`: its number and its length in words. A sentence with no marker stays. The Abstract, the Methods, the Results, the figures, the captions, the sentences that state what this paper does or finds, and every sentence with a number the paper states nowhere else are not marked, so they cannot be chosen.

# What to choose

The sentences you choose must together be between **$words and $upper words** long: add up the lengths in the markers. Choose as few sentences as reach that, and take them from different places rather than all from one paragraph. Prefer, in this order:
1. background a reader of this paper already has, and general statements the rest of the paper does not use;
2. a sentence that repeats a result, a limitation or another sentence;
3. hedging, and a sentence that only announces what the next one says.

Never choose:
- a sentence a later sentence depends on: one that introduces a term, a symbol or an abbreviation the text uses again, or that a later "this", "these", "such" or "the former" points back to;
- the only sentence that gives a limitation or a conclusion the Abstract states;
- the sentence that gives the reason the study was done.

# Formulas and headings

The paper follows this rule. Write every formula and symbol in LaTeX math between dollar signs, e.g. `$$\sigma = 0.6$$`, `$$k_1 \approx 0.28$$`; do not use Unicode math symbols (σ, ≈, ×). Do not number the section headings; FI numbers them. You write nothing here, so choose sentences whatever formulas they hold, and keep the marker numbers exactly as given.

# Output format

Respond with one JSON object and nothing else: no prose, no markdown fence.

```
{"delete": [12, 7, 30]}
```

Only numbers that appear in a marker.

---

# Inputs

## Topic
$topic

## The paper
<paper>
$paper_block
</paper>
