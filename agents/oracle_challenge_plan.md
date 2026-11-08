# Answer an Objection to Your Known Answer

A research quest checks its simulation against a **known answer**: it runs the simulation on one small case and compares the number it computes with the value the plan expects. The plan's working gives one value; the code that implements the simulation disagrees and says a step of the plan's working is wrong. You wrote, or now stand for, the plan's working. You are not told what the simulation produced: answer from the working alone.

# Topic

$topic

# The check

$check

# The objection

- The step it disputes: $step
- The expected value it gives instead, as a formula: `$formula`

# What to do

1. Go through your working step by step, from the definitions, and check the disputed step against the objection. Do not defend the working because it is yours, and do not give in because you are questioned: decide which is right.
2. If your working is wrong, concede and give `expected_formula`, $language.
3. If your working is right, stand by it and say exactly what in the code would produce the other value (the step the code gets wrong).

Reply with ONE JSON object and nothing else:

```json
{"answer": "<concede | stand>", "expected_formula": "<the corrected formula when you concede, else empty>", "code_fault": "<what the code gets wrong when you stand, else empty>", "why": "<the working, in one to four short sentences with the formula and the numbers>"}
```
