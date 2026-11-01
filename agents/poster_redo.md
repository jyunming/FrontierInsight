You are fixing parts of a research poster. A check of the rendered poster found the problems listed below. Every other part is final and stays exactly as it is.

$claims_rule
# The poster as it is (read-only)

$poster

# Figures you may use

$figure_list

A sentence about a figure whose caption is in the list above must be copied word for word from the paper's sentence about it; a paraphrase of it is removed.

# Sources you may cite

$source_list

$claims
$feedback

# Your answer

Redo only: $targets. Reply with one JSON object, no prose and no markdown fence, holding only those parts:

{"headline": "<only when the headline is redone>", "blocks": {"<index>": {"type": "heading|text|bullets|figure", ...the same fields that block has now}}}

Use the same block shapes as the poster above, plain text in every string, inline math between dollar signs. Keep what was right and fix the problems named for it.
