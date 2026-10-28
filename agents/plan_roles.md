You wrote the plan for a simulation study. Before it runs, answer one plain question about each number the study reports. Do not change the plan.

How the study measures (from your plan):
$method

The settings the study changes:
$settings

The numbers the study reports:
$results

For each number, say two things.

1. What kind of number it is. Use one word:
- result: a quantity the study measures and reports.
- reference: a theory or exact value computed beside a measured number, to compare it with.
- timing: a time the study measures as a result.
- physical_bound: a number that by its nature may sit exactly at the end of its range (exactly 0 or exactly 1 at some settings).
- diagnostic: a size of disagreement from a check, or a number kept only to check the code; not reported as a finding.
A number marked in brackets above already has its kind; give it anyway.

2. Which of the settings above it changes with. List only the settings you expect to change this number. Use the names exactly as written above. Write [] when it changes with none.
$again
Answer with one JSON object and nothing else:
{"results": {"<number>": {"role": "<one word>", "depends_on": ["<setting>"]}}}
Give every number above, once.
