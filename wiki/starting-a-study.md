---
title: Starting a study
sources: [core/commission.py, core/config.py, core/demo.py, core/analyze_cli.py, core/plan_settings.py, launch.py, web/server.py, vscode-frontier-insight/src/extension.ts]
updated: 2026-11-25
---
# Starting a study

Every study starts from a **commission**: a short file that says what FI is asked to find out. A config (YAML) names
it with `commission: <file>`, read against the config's own folder; the commission's question becomes the topic, so
the config needs no `topic:`, and one that is not the question word for word is refused.

## What a commission holds

- `question` (one sentence) and `scope` (what the study covers and leaves out).
- `limits`: `hours` and `model_calls`.
- `known_values`: values already known, each with the sentence of its source that states it (`passage`) and the
  source's full text (`source.file`); or `known_values: []` with `no_known_values: true` when none is known.
- `success` and `failure`: each a `condition` and how to `verify` it.
- `doubts`: sentences, `[]` when none.

## The two commands

- `fi new` writes `commission.yaml` (every key, each with a one-line comment) and `study.yaml` (the config that names
  it), never over an existing file.
- `fi check FILE` checks a commission, or the one a config names, by the rules FI runs before a study starts: one
  sentence per problem and exit 2, or exit 0 when it can be used. It makes no model call.

`fi demo` writes a filled-in pair (`fi-demo-commission.yaml`, `fi-demo.yaml`). `--analyze <data> --commission
<file>` analyses a folder of data for the commission's question.

## What is refused

`core/commission.py` `door` runs at every start (single, `--fleet`, `--watch`, the web start route, VS Code through
`launch.py`), before any folder, service or model call. A config that names no commission is refused in one sentence
that names `fi new`. An older quest started without a commission is not resumed, and cannot be given one later (`--resume`, `--rerun`, `--watch`,
`--update`); reading a finished quest (`--trace`, `--why`, `--quality`, the web quest page) still works. The engine
itself does not refuse.

`--update <id>` approves the settings in a quest's `config.yaml` as they are now and resumes it; it asks nothing.

Related: [[pauses|Pauses]], [[how-fi-judges-correctness|How FI judges correctness]].
