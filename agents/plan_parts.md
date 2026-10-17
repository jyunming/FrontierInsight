You are completing a part of the plan of a research quest. FI keeps the plan file itself and writes its design block; you do not rewrite the plan. You write only the parts that are asked for, as one JSON object.

# Topic

$topic

# The plan as it stands (read it for context; do not copy it back)

$plan_md

# What is asked

$request

# How to answer

- Answer with ONE JSON object and nothing else: no markdown fence, no prose before or after it, no second object, and not the plan.
- The object is shaped like the plan's design block: the parts are keys under `protocol`, written with the exact names the request uses (for example `{"protocol": {"grid": {"x": [1, 2]}}}`). A part the request says belongs at the top of the design block, such as `study_type` or `variables`, is a key of the object itself.
- Write only the asked parts, whole (a list or a mapping is written in full, not as a change to the plan's). Leave out every part that was not asked for: FI keeps the plan's own for them.
- Use only plain JSON: double-quoted keys and strings, numbers as numbers, `true` / `false` / `null`. A backslash in a string (LaTeX) is written doubled.
