# What a Research Topic Asks For

You are reading a research topic that someone wrote for an automated study. You did not write a plan for it and you will not. Your job is only to list, from the topic's text alone, what the topic itself says the study must contain, so that a plan written later can be held against the list.

# The topic

$topic

# What to list

- **factors**: each thing the topic says to vary, sweep, compare, or change between runs. Give its short `name` in the topic's own words, the `levels` the topic states for it (numbers, or names such as the named kinds it lists; an empty list when it states none), the `range` as `[low, high]` when the topic states one (else `[]`), `stated`: the phrase of the topic that says it, copied exactly, and `compared`: `true` when the topic compares results across its values (a result for each value, or how a result changes with it), `false` when the topic only uses its values (all of them together in each run, such as several values fitted together, or a range every run covers) and compares no result across them.
- **searches**: each design the topic asks the study itself to find, choose, tune or optimise, so that the study has to search for it because its values are not given in advance (a shape, a set of weights, gains or dimensions that makes a result as high or as low as possible). Give `what`: the thing whose design is searched for, in the topic's own words; `objective`: the result it is made best for, in the topic's words (empty when the topic names none); `stated`: the phrase of the topic that says it, copied exactly; and `main` as below. Not a search: picking the best among values or methods the topic lists (that is a comparison), estimating or fitting a model's parameters to data, a minimisation that is part of computing one result (a solver, an energy minimum), the value that meets a target, a limit or a tolerance (a calibration to a target, a root, the coarsest grid or smallest step that keeps an error under a limit: these are solved for or found by a sweep, not searched for), or a word such as optimal or best that names a known result or rate. An empty list when the topic asks for none.
- **quantities**: each result the topic says to report, measure or compute (`name`, and `stated`: the phrase, copied exactly).
- **formulas**: each formula or definition the topic writes out in full (`name`: what it gives, `text`: the formula exactly as the topic writes it, with the topic's own symbols).
- **comparisons**: each comparison the topic asks for, as one sentence (`what`) and the factors or levels compared (`between`).
- **constraints**: each limit or condition the topic puts on the study (what to hold fixed, what to say plainly, how long it may run), one sentence each.
- **code_files**: how many files of code the topic asks the study's code to be in, as a whole number (`1` when it asks for one script, one file of code or one module), else `null`. Only a request about the code itself: a file of results, of data or of input, or a script the topic compares with another, is not one. `null` when the topic does not say.
- **stated_results**: every number or value the topic gives as what the study will find, or as the answer a check should reproduce (an expected value, a known value, a target to be reached). Not a given input, a constant, a definition or a design requirement. Copy each exactly as written. An empty list when the topic gives none.

For every factor, search, quantity and formula, set `main` to `true` when the question the topic asks cannot be answered without it, and to `false` when it is something the topic also asks for but the question can be answered without.

# What to return

Return ONE JSON object and nothing else:

{"factors": [{"name": "", "levels": [], "range": [], "stated": "", "compared": true, "main": true}],
 "quantities": [{"name": "", "stated": "", "main": true}],
 "formulas": [{"name": "", "text": "", "main": false}],
 "comparisons": [{"what": "", "between": []}],
 "constraints": [""],
 "searches": [{"what": "", "objective": "", "stated": "", "main": true}],
 "code_files": null,
 "stated_results": [""]}

# Rules

- List only what the topic says. Do not add a factor, a quantity or a formula that the topic does not name, even when a study of this kind usually has it, and do not correct, simplify or complete a formula.
- Copy `stated` and `text` from the topic; do not paraphrase them.
- A thing the topic fixes at one value is not a factor; it is a constraint if the topic stresses it.
- Do not judge or plan. Do not write any number that is not in the topic.
