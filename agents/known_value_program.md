# Compute a Known Value From Its Source

A study was commissioned with a value already known from one of its sources. Before the study is planned, FI checks that value by computing it itself, and you write the small program that computes it, from the source's own definitions. **You are not given the value**: wherever the text states it, it reads `[value withheld]`. A program that prints a typed-in number instead of computing it counts as a failure.

# The value

- Name: `$name`
- The case it is for: $case
- The sentence of the source that states it: "$passage"
- The source: $title

# The source's text

$excerpt

# What to do

Write ONE short, self-contained Python program that computes this value for this case.

1. At the top, put the case's parameters in one dict, `CASE = {...}`, and read every parameter from it.
2. Compute the value by up to two independent routes:
   - `formula`: evaluate the source's own closed form or construction, as the text writes it (name its equations in a comment);
   - `definition`: compute the defining quantity directly by brute numerical means (integrate, enumerate, sum every term, or simulate the defining equations), never with that closed form.
   Give both when the text gives a closed form; otherwise give `definition` alone.
3. For each route print one line, and nothing after the last one:

```
$line {"route": "formula", "value": <the number>, "error": <one standard error of this number: its sampling or numerical error, not a loose bound>, "how": "<the method, in one sentence>"}
```

   A closed form evaluated exactly has an error of 0. Make each error small: refine the step, the grid or the number of samples until the number stops changing.
4. If the text does not give what is needed to compute the value (its definitions, its equations or its parameters are missing), print only:

```
$line {"cannot": "<what the text does not give>"}
```

Use only the Python standard library, numpy and scipy. Do not open or read any file, run any other program or import anything else. The program must finish within a few minutes.

$again

Reply with the program in one fenced Python block and nothing else.
