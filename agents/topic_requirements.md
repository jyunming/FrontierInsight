# What a Research Topic Asks For

You are reading a research topic that someone wrote for an automated study. You did not write a plan for it and you will not. Your job is only to list, from the topic's text alone, what the topic itself says the study must contain, so that a plan written later can be held against the list.

# The topic

$topic

# What to list

- **factors**: each thing the topic says to vary, sweep, compare, or change between runs. Give its short `name` in the topic's own words, the `levels` the topic states for it (numbers, or names such as the named kinds it lists; an empty list when it states none), the `range` as `[low, high]` when the topic states one (else `[]`), and `stated`: the phrase of the topic that says it, copied exactly.
- **quantities**: each result the topic says to report, measure or compute (`name`, and `stated`: the phrase, copied exactly).
- **formulas**: each formula or definition the topic writes out in full (`name`: what it gives, `text`: the formula exactly as the topic writes it, with the topic's own symbols).
- **comparisons**: each comparison the topic asks for, as one sentence (`what`) and the factors or levels compared (`between`).
- **constraints**: each limit or condition the topic puts on the study (what to hold fixed, what to say plainly, how long it may run), one sentence each.
- **stated_results**: every number or value the topic gives as what the study will find, or as the answer a check should reproduce (an expected value, a known value, a target to be reached). Not a given input, a constant, a definition or a design requirement. Copy each exactly as written. An empty list when the topic gives none.

For every factor, quantity and formula, set `main` to `true` when the question the topic asks cannot be answered without it, and to `false` when it is something the topic also asks for but the question can be answered without.

# What to return

Return ONE JSON object and nothing else:

{"factors": [{"name": "", "levels": [], "range": [], "stated": "", "main": true}],
 "quantities": [{"name": "", "stated": "", "main": true}],
 "formulas": [{"name": "", "text": "", "main": false}],
 "comparisons": [{"what": "", "between": []}],
 "constraints": [""],
 "stated_results": [""]}

# Rules

- List only what the topic says. Do not add a factor, a quantity or a formula that the topic does not name, even when a study of this kind usually has it, and do not correct, simplify or complete a formula.
- Copy `stated` and `text` from the topic; do not paraphrase them.
- A thing the topic fixes at one value is not a factor; it is a constraint if the topic stresses it.
- Do not judge or plan. Do not write any number that is not in the topic.
