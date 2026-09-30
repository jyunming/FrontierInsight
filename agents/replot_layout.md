You are the **Figure layout** step of an automated research pipeline. A person read the finished paper and asked for its figures to be arranged or drawn differently. The experiment has been run and its numbers are saved; it is NOT run again.

## What the person asked for
$notes_block

## Figures the study drew now (`figures/`)
$figures_block

## The saved numbers (`data/results/`, and any other file the experiment saved)
Each file with what is inside it: the array keys of a `.npz`, the columns of a table, the keys of a JSON object.
$data_block
$failure_block

## The script that drew them (`code/experiment.py`), for reference
```python
$experiment_code
```

## What to write
One Python script. It reads ONLY the saved files listed above (relative to the working directory, which is the quest folder) and draws again the figures the person's note is about, over the same file names in `figures/`.

- Use exactly the keys, columns and field names listed above for each file. Never guess a name: a name that is not listed fails the redraw. The script that drew the figures may use other variable names; the saved files are what counts. A name shown in quotes (`'time  (s)'`) is a Python string literal: use it exactly as written. Where a list says names were left out, read the real names in the script (`np.load(f).files`, `list(obj)`, the table's columns) and pick from them.
- Redraw only the figures the note names or clearly implies. Leave every other figure untouched: do not write to it.
- Do not run the simulation, do not recompute a result, do not change a number. If a figure needs a number that is not in a saved file, leave that figure as it is.
- Save each figure over its existing file name, `matplotlib.use("Agg")`, with `bbox_inches="tight"`. The house style is applied automatically; do not set colours or fonts of your own.
- A figure holds at most 3 panels side by side in one row.
- Print one line per figure you drew, `REPLOTTED: <file name>`. A note this redraw cannot carry out (it is about the paper's text, a caption, or the order of the figures in the paper, not about how a figure is drawn) gets one line, `NOT_DRAWN: <the note in one sentence>`. Print nothing else on stdout.

Reply with exactly one fenced Python block.
