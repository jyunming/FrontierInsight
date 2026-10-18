# Make the Simulation Faster Without Changing Any Result

You are speeding up the simulation of a research study. FI timed the whole study before running it and it would not
finish within the time allowed. Nothing the study measures may change: FI runs a fixed set of trials of the simulation
itself, before and after your change, with the same random seeds, and a change is kept only when every number the
simulation returns is the same as before (to nine digits) and the time those trials take drops by at least a quarter.

## The topic, as the person wrote it

$topic_block

## The time

$speed_block

## Changes already tried in this quest

$history_block

## The simulation's code

The files you may change: $editable. Every other file is out of reach, and a change to one is refused.

$code_block

## What to do

Find the one thing in the code that costs the most time for nothing: work done again that was done already, a loop in
Python that numpy does for a whole array, a step that goes on after its answer is final, a resolution finer than the
result can show. Remove it with ONE small change that gives the same numbers. $rounds_left

Rules:

- One change: one piece of text that appears exactly once in one of the files above, and the text that replaces it.
  Copy the text to replace exactly, with its indentation.
- The numbers must stay the same, not only close. Do not lower a tolerance, an iteration cap, a number of steps, a
  number of trials or a resolution if that changes a returned number; do not return a stored or guessed value; do not
  test for a setting, a trial number or a seed.
- Do not change what `run_trial` / `run_cell` / `oracle` are called or what they return, nor where the trials' seeds
  come from.
- A change to comments, layout, docstrings or printed text is not a change: it is refused, and so is one that gives
  back a version already tried.

Reply with JSON only, nothing before or after it:

{"file": "simulate.py", "find": "<the exact text to replace>", "replace": "<the text that replaces it>", "why": "<one sentence: what costs the time and why the numbers stay the same>"}
