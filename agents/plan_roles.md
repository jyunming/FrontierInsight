You wrote the plan for a simulation study. Before it runs, answer plain questions about each number the study reports, and about the system it studies. Do not change the plan.

How the study measures (from your plan):
$method

The settings the study changes:
$settings

The numbers the study reports:
$results

For each number, say five things.

1. What kind of number it is. Use one word:
- result: a quantity the study measures and reports.
- reference: a theory or exact value computed beside a measured number, to compare it with.
- timing: a time the study measures as a result.
- physical_bound: a number that by its nature may sit exactly at the end of its range (exactly 0 or exactly 1 at some settings).
- diagnostic: a size of disagreement from a check, or a number kept only to check the code; not reported as a finding.
A number marked in brackets above already has its kind; give it anyway.

2. Which of the settings above it may change with. List the settings you expect could change this number. Use the names exactly as written above. Write [] when it changes with none.

3. Whether each run returns it. Use "each_run" when the simulation returns this number for every run it makes. Use "afterwards" when the study computes it after the runs, from the numbers the runs returned (a width, a threshold, a fitted slope, a count of settings, a total time).

4. Its shape: what one run returns for it. Use one word:
- number: one number (most results; FI itself draws such numbers along the settings the runs differ in).
- curve: a list of numbers along one thing inside a single run.
- map: numbers over two things inside a single run (a field over a plane, a quantity over time and position).
For a curve or a map, give `along`: the one or two things it runs along inside one run, each one of: $along, or another such quantity in a word or two (for example frequency). A setting of the plan is never one of them (FI already records a number for each value of a setting).

5. Its physical bound, only when the physics of the quantity itself fixes one: a probability or a fraction of a whole cannot be above 1 or below 0; a count, an energy or a variance cannot be below 0; an absolute temperature cannot be below 0 K. Give `bound` with `min`, `max` or both, in the unit the study reports the number in, that `unit`, and `why` in one sentence. Write `"bound": null` when physics fixes no bound. A range you only expect the number to fall in is not a physical bound: never give one.

Then describe the system the study simulates, so that FI can draw its set-up: its parts, each with a plain name a reader knows, and what touches or feeds what.
- Each entry has an `id` (a short name in code), a plain `name` with no number in it, and a `kind`: "part" (a body, a region, a component), "boundary" (a wall, an edge or an interface where a condition is set) or "source" (what drives the system: an inflow, a heater, a load, an input signal).
- `carries` lists the plan's names that belong to the entry: a setting, a constant or a number the study reports, each exactly as written here: $names_in_plan. FI writes every value from the plan by these names, so never write a number in a name.
- `links` joins two entries that touch or feed each other, with `what` passes between them in a few words.
Write `"system": null` when the study has no physical set-up to draw (for example a study of a statistical method alone).
$titles$again
Answer with one JSON object and nothing else:
{"results": {"<number>": {"role": "<one word>", "depends_on": ["<setting>"], "returned": "<each_run | afterwards>", "shape": "number", "bound": null}},
 "system": {"parts": [{"id": "<id>", "name": "<plain name>", "kind": "part", "carries": ["<plan name>"]}], "links": [{"from": "<id>", "to": "<id>", "what": "<a few words>"}]}}
For a curve: "shape": "curve", "along": ["<thing>"]. For a bound: "bound": {"min": 0, "max": null, "unit": "<unit>", "why": "<one sentence>"}.
Give every number above, once.
