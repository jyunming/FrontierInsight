You are the **Slides** stage of an automated research pipeline.

# Source paper (markdown)
$paper_md

# Figures available (reference each by filename)
$figure_list

# Your task
Compress the paper into a Marp slide deck of **8–12 slides**. Use the standard Marp front-matter and `---` separators between slides. Embed each figure exactly once. Keep total spoken duration about 10 minutes.

## Figure sizing — pick the right Marp directive for the figure's shape

Slides are 16:9 (960×540). Default Marp behavior stretches images to fill, which destroys aspect ratio for non-16:9 figures and is the #1 source of "figures too big / wrong proportions" complaints. Use one of these patterns based on what each figure actually looks like:

- **Wide / panoramic figure** (parity plot, time series, comparison chart wider than tall) — dedicate a slide and constrain to slide width:
  ```
  ![w:800](figures/parity_plot.png)
  ```
- **Tall / vertical figure** (feature-importance bar chart, vertical histogram) — constrain by height so it doesn't dominate the slide:
  ```
  ![h:380](figures/feature_importances.png)
  ```
- **Side-by-side with prose** (figure + 1-2 paragraphs on the same slide) — pin the figure to the right 38% and leave room for body text on the left:
  ```
  ![bg right:38%](figures/<name>)
  ```
  **Never use bare `![bg right](...)` without a percentage** — it covers the whole right half regardless of aspect ratio and stretches square figures.
- **Hero / cover figure** (single dominant image, no body text needed):
  ```
  ![bg](figures/<name>)
  ```
  Add a `# Title` heading on the slide so it isn't visually identical to the figure alone.

If a figure's aspect ratio is unknown, default to `![w:720](figures/<name>)` on its own slide — that scales any figure to fit within the slide's content area without distortion.

# Output format
Respond with the Marp markdown only — no JSON, no surrounding fence, no preamble. Begin with the Marp front-matter block (`---\nmarp: true\n...\n---`).
