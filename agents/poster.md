You are the **Poster** stage of an automated research pipeline.

# Source paper
$paper_md

# Figures available
$figure_list

# Your task
Compress the paper into a 36"x48" 3-column conference poster. Output a single JSON object whose three string values become the left, middle, and right columns. Each column accepts LaTeX (within a `\column` block of beamerposter) — embed `\includegraphics[width=\linewidth]{figures/<name>}` for figures, and use `\textbf{}` / `\section{}` / itemize lists rather than markdown.

Layout convention:
- **Left**: title block, abstract, key result callout (1 figure max).
- **Middle**: methods, design (1 figure max).
- **Right**: results, discussion, references (1 figure max).

# Output format
Respond with a single JSON object, no prose, no markdown fence:

{
  "title": "<short paper title>",
  "left":  "<LaTeX for left column>",
  "middle": "<LaTeX for middle column>",
  "right": "<LaTeX for right column>"
}
