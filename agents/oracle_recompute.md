# Work Out a Known Answer Again

A research quest checks its simulation against a **known answer**: it runs the simulation on one small case and compares the number it computes with the value the plan expects. That check did not pass. Before anyone changes the simulation, the expected value itself is worked out once more, independently, by you. You did not write the plan, and you are not told what the simulation produced: work the value out from the check's own statement, case and reference.

# Topic

$topic

# The check

- Name: `$name`
- Kind: $kind
- What it checks: $check
- The case it runs on: $case
- How its number is computed from what the simulation returns: `$measure`
- Where its expected value comes from (the plan's reference): $reference

# What to do

1. Work out, step by step, the exact value this check's number must have on this case if the simulation is right. Use the reference, the case's parameters and the definition of the number above. If the number is an error of a numerical method at a finite step, it is that error at THIS step (not its limit as the step goes to 0), with its constant worked out. If it is a worst violation of a rule that holds exactly, it is 0 only if the rule holds exactly at this step; if the method keeps the rule only up to its own error, say so and give that error at this step.
2. Do not round early: give the value to as many digits as the working supports.
3. If the value cannot be worked out from what is given, say why and give no number.

Reply with ONE JSON object and nothing else:

```json
{"expected": <the number, or null>, "how": "<the working, in one to four short sentences with the formula and the numbers>"}
```
