# Judge Between Two Workings of a Known Answer

A research quest checks its simulation against a **known answer**: it runs the simulation on one small case and compares the number it computes with the value the plan expects. The plan's working and the code that implements the simulation disagree on that value. The code disputed a step of the plan's working; the plan's model answered and stood by its working. You wrote neither. You are not told what the simulation produced: judge from the two workings alone.

# Topic

$topic

# The check

$check

# The code's objection

- The step it disputes: $step
- The expected value it gives instead, as a formula: `$formula`

# The plan's answer

- Why it stands by its working: $plan_answer
- What it says the code gets wrong: $plan_fault

# What to do

1. Work the value out yourself, step by step, from the definitions and the check's case. Do not side with either because of who wrote it.
2. Decide: `plan` if the plan's working is right, `code` if the objection is right, `undecided` if you cannot tell from what is given.
3. Give `expected_formula`, $language.

Reply with ONE JSON object and nothing else:

```json
{"side": "<plan | code | undecided>", "expected_formula": "<the expected value as a formula>", "why": "<your working, in one to four short sentences with the formula and the numbers>"}
```
