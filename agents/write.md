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

# Pre-flight clarifications (user-supplied or auto-derived)
$clarify_block

# Cross-paper check (literature retrieved per finding, classified as supporting/conflicting/neutral)
$cross_check_block

# Your task

Produce a single Markdown paper in IMRAD format (Introduction, Methods, Results, Discussion). Include all figures using `![caption](figures/<filename>)`. Keep it under ~4 pages of prose.

## Honesty constraints — read this section, do not skip

The experiment may have failed, produced implausible numbers, or
contradicted the hypothesis. The analysis block above will say so
plainly. **When that happens, the paper MUST say so plainly too.** Do
not paper over a broken experiment by:

1. **Reporting numbers that the analysis flagged as implausible** as if they were real findings. The analysis is upstream of the writer for a reason — its caveats are your source of truth.
2. **Inventing additional measurements** that weren't actually computed in the methods. Every number you cite in Results MUST be in the analysis block.
3. **Softening "implausible" / "no signal" / "noise dominated" / "inverted result"** into "interesting" or "surprising". When a result is broken, say it's broken.
4. **Writing a clean conclusion** when the experiment was inconclusive. A null-result paper is fine; a paper claiming results that aren't supported is not.

If the analysis block contains words like *"implausible"*, *"no signal"*, *"inverted"*, *"orders of magnitude larger than expected"*, *"likely a bug"* — those are red flags. The Discussion section must address them directly. Examples of acceptable framing:

- *"The high-NA configuration produced an HV bias of 82 nm, two orders of magnitude larger than literature values (~1 nm). We hypothesize this reflects a bug in the imaging-model implementation rather than a physical phenomenon; the result is not interpreted as evidence for or against the central hypothesis."*
- *"The simulation showed zero bias across all configurations, which is implausible given known mask-3D effects in the EUV regime. Rather than report this as a finding, we treat it as a software-validation failure and outline the next experiment in Future Work."*

## Topic-shape recognition

Some topics are **survey/comparative** ("differences between X and Y", "review of methods for Z", "challenges in deploying W") rather than **experimental**. When the topic is survey-shaped AND the experiment in the methods section was a poor fit (e.g. a narrow numerical simulation pretending to answer a broad comparative question), the right paper is a literature synthesis with comparison tables, not a results-claim-from-numbers.

Recognize this from the topic + analysis. If you're writing about a survey-shaped topic with a thin or broken experiment, structure the paper as:

- **Introduction** — what's the comparative question.
- **Background** (replaces Methods) — what each side of the comparison is.
- **Comparison** (replaces Results) — a table of the axes that differ, citing prior work for each cell, and the experimental finding (if any) for the one cell the experiment actually addressed.
- **Discussion** — what the literature broadly says about the comparison, where consensus exists, where it doesn't.
- **Limitations** — explicitly note that the experimental section addressed one narrow aspect, not the whole comparative question.

End with `## References` in numbered-list style citing concrete sources from the prior-work block above (or, if none are usable, plausibly-formatted primary references with DOIs).

# Output format
Respond with the markdown of the paper only — no JSON, no surrounding fence, no preamble. Begin with `# $title`.
