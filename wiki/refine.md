---
title: Refine
sources: [core/engine.py, core/number_provenance.py, core/data_shape.py, agents/replot_layout.md]
updated: 2026-09-30
---
# Refine

When a quest stops for your review (`pauses.review: ask`, the default) you accept, reject or **refine**. A refine is a note saying what to change. FI answers each point in the cheapest way that can answer it, and never re-runs the whole study for a note the text can answer.

## The four routes

After a refine the writer answers what the text can answer and ends its reply with one line per remaining point. FI removes those lines from the paper and routes on them:

| Your point needs | Route | What happens |
|---|---|---|
| only different text | paper | the paper is rewritten |
| a number the study lacks | missing number | the existing script is extended and run again |
| figures arranged, resized or redrawn | layout redraw | the figures are redrawn from the saved data, without re-running |
| a genuinely different study | experiment | back to the design |

If any point needs a different experiment, that route wins and missing-number points go with it. A refine with no notes is not a decision: `--refine ""` and the web page refuse it, and the quest stays paused (see [[pauses|Pauses]]). Only at the terminal prompt and the VS Code input box does an empty refine confirmed with Enter mean accept, as those prompts say. A refine is always honoured, even past `engine.max_iterations`. The route is recorded in the trace.

## Missing number: extend the script

FI asks the model to change the existing scripts as little as it can to compute what you asked for. The frozen protocol is not rewritten; the difference is recorded as extended at your request, and the protocol checks excuse only that difference. If the model returns no usable script, the old scripts are kept and the paper's limitations say the number could not be added. A quest without a simulation searches its data sources again for exactly that number.

**The number must reach the paper.** After the run, FI checks (without a model call) that each new value appears in the paper at the paper's rounding, or that a new result is named with a number. If not, the writer is asked once more with the computed values. If it is still missing, `run.log` says "Your request … was computed … but is not in the paper", the review shows it, and the paper is never accepted automatically.

## Layout redraw: new figures, same numbers

FI shows the model what each saved data file under `data/results/` and `data/` holds (column names, array shapes, keys; nothing is unpickled), plus the experiment code. The model writes `code/replot_layout.py`. FI runs it for at most min(`execution.timeout_s`, 300) seconds.

- Figures and data are backed up first. Changed numbers are always put back, and a failed try restores everything.
- A failed try, or one that changed no figure, is retried once with the error.
- If both fail, the figures stay as they were, `run.log` says the request was NOT applied, the review shows it first, and the paper says so. The quest is not accepted automatically.

## How to refine

- CLI: `fi --config q.yaml --resume <id> --refine "notes"`.
- Web: the Refine button and text box on the review banner of the quest page.
- VS Code: choose Refine at the review, then type the notes.

Gap: a requested number that was computed but left out of the paper is shown in the terminal, in `NEXT_STEP.md` and in the web page's "Action needed" banner. It is not shown in the web review panel or in the VS Code review chat.

Related: [[doing-a-step-again|Doing a step again]], [[code-project-and-run-data|The code project and run data]], [[pauses|Pauses]].
