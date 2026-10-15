# A Test That Shows One Finding

A research quest has written its code. A second model read it against the plan and reported ONE problem, below. Before anything is changed, FI wants to know whether the problem is real. Your job: write a small test that FAILS on the code as it is now, because of this problem, and that would pass once the problem is fixed. If the code is in fact right, your test will pass and FI will leave the code alone. That is the correct outcome, so do not make the test fail for another reason.

# Topic

$topic

# The plan's equations (as the plan writes them)

```json
$equations
```

# The finding

$finding

# The code the finding is about

$code

# How the test reaches the code

The test runs with the `code/` directory on `sys.path`. Import with: $modules.

# Rules for the test

- A Python file with one or more functions named `test_...` that take no argument. Each shows the problem by an `assert` about the VALUES the code computes, with a message that says what is wrong: the finding's own claim, written as a relation those values must satisfy. Only an `AssertionError` from such an `assert` shows the problem. An exception from the code you call, or from the test itself, shows nothing, and FI reports a crash in the code as a crash, with its file and line: do not catch it, and never turn an exception into `assert False`.
- Small and fast: a few calls with small inputs, a fixed seed, no files read or written, no network, finished in a few seconds.
- Assert a relation that follows from the plan or from the definition of the quantity: an identity, an invariant, a symmetry, a limit, the same quantity computed a second, direct way, what happens to the result when one setting changes. Never copy a number from the plan, never use a value you were not given, and never write a tolerance looser than the quantity needs.
- Test the function or the output the finding names; do not test the whole study.
- Do not change the code, and do not import anything that is not installed.

# What went wrong with the last try

$problem

# What to return

ONE fenced Python block holding the whole test file. Nothing else.
