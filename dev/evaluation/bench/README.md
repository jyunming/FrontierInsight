# FI's self-benchmark

How often does FI let a wrong result through as publishable, and how often does it hold back a right one? This folder
holds the tools that measure it: tasks whose answer is known, errors planted on purpose, a scorer that reads only what
FI computed and recorded, and a report. It is for FI's developers; a person running a quest never sees it.

The hand-graded scores of the fixed SIR topic are a different thing, in [`../benchmark/`](../benchmark/README.md).

## What is measured

| Number | What it is |
|---|---|
| Wrong results let through | of the runs with a planted error that changes the answer, the share FI would still publish; per error and in total |
| Right results held back | of the clean runs whose answer is right, the share FI would not publish (a provider outage or a spent quota is not counted) |
| Wrong among the published | of the clean runs FI would publish, the share whose answer is wrong |
| Where each error was first caught | per planted error, the first check whose record flagged the run |
| Answer right at each evidence level | per final level, how often the answer was right (counts only under 10 runs) |
| Cost | model calls, tokens and time, from each quest's own records |

Every rate has its 95% Wilson interval. "Would be published" is the evidence level `publication_ready`, or one level
below it when the only gap left is that no person accepted it: benchmark runs accept a passing review automatically.

## The pieces

| File | What it does |
|---|---|
| `../../quest-topics/answers/*.answer.json` | one task each: the quest's YAML, the sentence added to its topic that names the protocol's metric ids and the setting the answer is read at (`ask`), the expected values with their tolerance and source, and which errors it takes |
| `answers.py` | loads and checks the answer files; held-back tasks (below) |
| `reference.py` | the benchmark's own reference computations, where an expected value is not a textbook closed form |
| `catalogue.py` | the 17 planted errors (N1-N5, S1-S4, L1-L4, R1-R3, O1): how each is planted, from which step the copy is run again, which checks should catch it. Planters exist for R1, R2, N3, S1 and L1 |
| `plant.py` | the planters: change a file of a copied quest (R1 the paper, N3 the simulation, S1 the analysis), replace one recorded model answer (R2: the writer's), or add a retracted paper to a fresh run's search results with its DOI (L1) |
| `validity.py` | the valid-error filter: runs the planted code offline through FI's own trial harness on the answer's setting; a change that does not move the answer (or, for the analysis, what it prints) is an equivalent change and is not counted |
| `runner.py` | runs a quest with its model calls served by `core/replay.py`; copies a recorded run; the benchmark's settings |
| `score.py` | one run's outcome and the numbers above |
| `report.py` | `self_benchmark.json` (the one object a web or VS Code page will show) and `self_benchmark.md` |
| `cli.py` | `fi tools bench ...` |
| `../../../core/replay.py` | the replay client and the Crossref replay (shared with the record-and-replay of quests) |

## How a run is served

A recorded call is named by its step and its number within that step in the run (`write#1`), never by a hash of its
prompt: a prompt holds paths, times and retrieved text that differ between two runs.

| Mode | Model calls | Measures | Cost |
|---|---|---|---|
| `record` | all real, all kept | a clean run | one quest |
| `partial` (B) | the calls before the planted one replayed from the source recording, the planted one replaced, every later call real | the checks done by code and the checks a model judges | the steps after the planted one |
| `replay` (A) | every call from an earlier run of the same plant; a call it does not have gets a fixed "cannot fix" answer and a divergence event naming the step | the checks done by code, and regressions in FI's parsing and routing; a judging model's verdict is the recorded one | none |

A call that failed in the recorded run fails again in its replay (it is never answered). Crossref's retraction answers
are recorded with a run that makes them live (`bench/crossref.json`: a recording, a fresh L1 run in `partial`) and
replayed in `replay`; a DOI with no recorded answer is then "not checked", never "not retracted". A copy run again from
a step after the literature makes no lookup of its own.

Three rules keep the numbers honest:

- **After the planted point, everything is real** in mode B: replaying a clean writer after a planted error in the code
  would give a clean paper, and the checks would then catch a mismatch the replay made, not the error.
- **Only valid errors count** (`validity.py`).
- **Every planted run has a control**: the same copy run again from the same step with nothing planted. A flag the
  control has too, for the same reason, is the copy's doing, not a detection (it is listed apart). A planted run is
  not counted when it has no control, when its control would not be published either, when its replay asked for a
  call the recording does not have, or when its planted answer was never asked for; the report says which.
- **L1 counts when it was exercised**: when its control's paper rests on the same kind of source (one never
  retracted, added the same way). Whether FI then used the retracted one is what is measured.

## The benchmark's settings, and why

Every benchmark quest runs with (`runner.BENCH_SETTINGS`):

- `engine.phased: false`. Explore-then-confirm keeps any quest run again from a step after its confirm run as
  "statistically adequate" (`confirm_reused`), so every copy would look caught. The benchmark therefore measures FI
  with it off; an error the confirm run alone would catch is not credited.
- `engine.auto_accept_on_pass: true`, `pauses.review: off`, `pauses.papers: false`. The plan pause is answered as a
  person who approves the plan unchanged.
- `output.save_model_calls: true`, `knowledge.write_back_quests: false`.

The models are not the benchmark's choice: pass them in a YAML with `--settings` (a `provider:` block, and
`provider.node_models` for the reviewer on another model that research needs).

## Planting, step by step

```bash
fi tools bench check                                        # every answer file is usable
fi tools bench record Q4 --out C:/fi_runs/bench/Q4/clean --settings models.yaml        # calls models
fi tools bench plant control --from-run .../Q4/clean --step claims --out .../Q4/control-claims   # for R1
fi tools bench plant control --from-run .../Q4/clean --step run --out .../Q4/control-run         # for N3, S1
fi tools bench plant control --from-run .../Q4/clean --step writing --out .../Q4/control-writing # for R2
fi tools bench run .../Q4/control-claims --mode partial      # each control runs too (calls models)
fi tools bench plant R1 --from-run .../Q4/clean --out .../Q4/R1
fi tools bench plant N3 --from-run .../Q4/clean --out .../Q4/N3 \
    --file code/simulate.py --find "2.0 * beta" --replace "beta"
fi tools bench filter --clean .../Q4/clean --planted .../Q4/N3       # no model
fi tools bench run .../Q4/R1 --mode partial                 # calls models after the planted point
fi tools bench run .../Q4/R1-again --mode replay --recording .../Q4/R1   # no model (plant R1 into R1-again first)
fi tools bench score C:/fi_runs/bench                        # writes dev/evaluation/benchmark/self_benchmark.{md,json}
```

A code plant names its own `--find`/`--replace`: the code is the model's, different in every recording. R1 without
`--find` multiplies the paper's first number with two or more decimals by 1.37. L1 is a fresh run of the task
(`plant L1 --task Q11`, and `--control` for its control with a paper that was never retracted): forking from the
literature step replaces the frozen protocol, which FI records as an amendment made after the results were seen. The
default L1 source is about MMR and autism; for another task (Q4) give a retracted paper on its own topic with
`--paper source.json` (`doi`, `title`, `content`), and its control a never-retracted one, or the run is not exercised.
A copy runs with the settings of the run it was copied from; `--settings` applies only to a run from the start.

Large outputs belong outside the repository (`C:\fi_runs\bench\` on the maintainer's machine).

## Held-back tasks

Four of the twelve tasks are kept out of the development tree so FI is not tuned against them. A held-back task's
answer file lives outside the repository (`FI_BENCH_HELD_OUT`, default `~/.fi/bench_held_out/`), and
`dev/quest-topics/answers/held_out.json` keeps only its SHA-256: an answer file that no longer matches it is refused
(an answer changed after it was committed to cannot be told from one fitted to the results). `fi tools bench hold-out
<answer file>` moves a file out and records its hash; tasks are rotated each quarter by hand. No task is held back yet:
the four are chosen from the seven tasks still to be written.

## What is known not to work yet

- A paper pinned with `knowledge.local_papers` or dropped into `inputs/papers/` is never looked up for a retraction:
  FI keeps no DOI for it. L1 is therefore planted as a search result.
- Replay assumes each step asks for its calls in the same order in two runs. A benchmark run numbers its calls in the
  order they were asked (`calls.jsonl`); a quest recorded outside the benchmark is read from `.fi/io/`, in the order
  the answers came back, which differs for calls one step makes at the same time (the ideas' tournament, an
  ensemble). A change to FI's flow can make an old recording diverge, which the divergence events show.
- The tolerances assume the run counts the `ask` sentences name; a run with far fewer trials can be scored wrong
  although its simulation is right.
- The first recording of the MVP (five tasks, two models) has not been run: phase 0 is the tools, validated on
  fake-model quests (`tests/test_self_benchmark_e2e.py`).
