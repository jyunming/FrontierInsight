# Improve the Simulation Against Its Checks of Correctness

You are improving the simulation of a research study. Before any result was seen, the study's protocol fixed a few
checks of correctness: whether the simulation lands on answers that are known, and whether its trials behave as
independent trials should. FI runs the simulation itself on each check's own setting and reads the number the check
names from what the simulation function returns; nothing the code prints, and no result file it writes, is read. The study's own findings are not shown to you and are not what you are improving: a change is
kept only when it makes a check of correctness better without making another one worse.

## The checks (fixed with the protocol; they cannot be changed here)

$criteria_block

## Where they stand now (the version kept so far, as FI measured it)

$values_block

## Changes already tried in this quest

$history_block

## The simulation's code

The files you may change: $editable. Every other file (the analysis script, FI's own files, the protocol) is out of
reach, and a change to one is refused.

$code_block

## What to do

Find the one thing in the code most likely to make a check that is not met come out wrong (a wrong formula, a wrong
order of steps, a step size or tolerance too coarse for what the check asks, a seed or a random stream reused between
trials), and fix it with ONE small change. $rounds_left

Rules:

- One change: one piece of text that appears exactly once in one of the files above, and the text that replaces it.
  Copy the text to replace exactly, with its indentation.
- Fix the computation. Do not change how the number a check reads is worked out where the simulation returns it (the
  expression beside that key), do not test for the setting a check runs on, and do not write a value a check expects
  into the code: a change that does any of these is refused.
- Do not change what `run_trial` / `run_cell` / `oracle` are called or what they return, nor where the trials' seeds
  come from.
- A change to comments, layout, docstrings or printed text is not a change: it is refused, and so is one that gives
  back a version already tried.

Reply with JSON only, nothing before or after it:

{"file": "simulate.py", "find": "<the exact text to replace>", "replace": "<the text that replaces it>", "why": "<one sentence: what was wrong and which check this should make better>"}
