You are the **Writing** stage of an automated research pipeline.

# Topic
$topic

# Title (use as the paper title)
$title

# Design
$design_block

# Analysis
$analysis_block

# Prior work (cite at least 3 of these)
$literature_block

# Figures available (reference each by filename)
$figure_list

# Your task
Produce a single Markdown paper in IMRAD format (Introduction, Methods, Results, Discussion). Include all figures using `![caption](figures/<filename>)`. Keep it under ~4 pages worth of prose. End with a `## References` section in numbered-list style citing concrete sources from the prior-work block above (or, if none are usable, plausibly-formatted primary references with DOIs).

# Output format
Respond with the markdown of the paper only — no JSON, no surrounding fence, no preamble. Begin with `# $title`.
