You are revising the plan of a research quest. The plan is one Markdown file that a person reads, edits and approves before any experiment runs. The person has asked for a change. Make that change, and only that change, and return the whole file.

# Topic

$topic

# The plan as it stands

$plan_md

# What the person asked for

$request

# How to revise

- Return the **entire** file, from the `# Plan:` title to the end, as Markdown. No commentary before or after it, and do not wrap it in a code fence.
- Change what the request touches, and what has to follow from it. If the request moves the hypothesis, the *In short*, *The gap this experiment addresses*, *Success criteria* and the design block must all agree afterwards. If it only adds a risk, add the risk and leave the rest as it is.
- Keep every other sentence exactly as it stands. Do not tidy, re-order or reword what was not asked about.
- Keep the section headings as they are. The last section, *The design (used as written)*, holds a fenced `yaml` block. It is read by a program, so keep it valid YAML with the same keys (`hypothesis`, `variables` with `independent`, `dependent` and `controls`, `method`, `expected_outcome`, `figures_planned`, `dependencies`, `result_assertions`), and edit its values to match the request. Each `result_assertions` entry keeps a `path` and numeric `min` and `max`.
- Do not add sources to *What the literature says* that are not already in the file, and do not invent results. If the request needs evidence the plan does not have, say so in *Risks* instead of writing it as fact.
- If the request cannot be done as asked (it contradicts itself, or it would make the experiment impossible on a CPU within the time limit), make the closest change that can be done and say what you did not do under *Risks*.
