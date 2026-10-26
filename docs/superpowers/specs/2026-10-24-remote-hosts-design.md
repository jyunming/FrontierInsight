# Running the simulation on other machines — design

Status: approved by the maintainer in three rounds of questions (2026-10-24), after a builder check.

## Intent

FI's host should stay light: it judges, checks, draws the figures, calls the models and writes the paper. Every run of
the person's simulation (FI's small run, the trials of each setting, the search for the best design and its check)
should be able to run on other machines at the same time, so a study is not limited by the host's cores and the host is
not overloaded. The person names the machines; nobody sets up an environment on them.

## Goals

- One YAML line, `execution.hosts: [a, b, c]`, sends the simulation's runs to those machines; without it nothing changes.
- The machines share a drive with the host (the person's assumption). Results come back as the files they already are.
- Python is on each machine; a package it lacks is supplied by FI on the shared drive, not installed on the machine.
- A run's answer is the same whichever machine ran it (same code, same seed), so FI's checks are unchanged.
- The host runs none of the person's simulation while a machine is available.

## Non-goals (this version)

- A batch scheduler (Slurm, LSF). Later, reusing `execution.background_jobs`.
- Windows or macOS remote machines. The host may be Linux or Windows.
- Machines without the shared drive (copying the quest over ssh).
- GPU placement, containers on the remote side.

## Anti-goals

- FI stores a password, a key or a token. It uses the person's own ssh configuration and agent.
- FI changes anything on a remote machine outside the quest's folder on the shared drive.
- A machine's trouble changes a result, hides a setting, or is silent.
- A new noun a scientist would not use in the console or YAML.

## Decided

1. **Remote OS.** Remote machines are Linux. The host is Linux or Windows; a Windows host needs a path map.
2. **Paths.** The shared drive at the same path on every machine needs no setting. Otherwise one map in YAML, e.g.
   `execution.host_paths: {"Z:/team": "/proj/team"}` (the builder may choose the exact key; it must be one line).
3. **Transport.** ssh to the named hosts, with the person's own ssh config (host aliases, keys, agent, jump hosts).
4. **What goes out.** Every run of the person's simulation: the small run, the trial runner's settings, the search's
   evaluations (`core/optimise.py::EvaluationPool`, `evaluate_many`) and its check (`core/optimum_check.py`). FI's own
   work stays on the host: checks, analysis, figures, model calls, the paper, the code checks that run FI's own scripts.
5. **How many per machine.** On first contact FI reads each machine's cores and memory; each machine gets its own
   memory gate (`core/execution.py::SettingsGate`) and runs on its free cores (cores minus its current load), so other
   people's work on it is not taken.
6. **A machine that fails.** Its unfinished settings are run again on another machine (same seed, same answer); the
   machine is not used again in this quest; `run.log` says so in one plain line. With every machine down, the host runs
   them as today.
7. **Configuration.** `execution.hosts`: a list of ssh host names. Absent or empty: today's behaviour.
8. **Missing packages.** FI reads each machine's Python version and platform, downloads on the host the wheels the
   quest's requirements need for that version and platform, and installs them into a folder of the quest on the shared
   drive (e.g. `.fi/hosts/py<version>-<platform>/`), put on the run's path. The remote machine needs no internet. A
   package with no ready wheel for that machine is named in `run.log` and that machine is not used.
9. **The host's own cores.** While any machine is available the host runs none of the simulation.
10. **Other people's load.** Free cores = cores minus the machine's load average, read before work is sent.
11. **Safety.** Model-written code runs on those machines under the person's account, without a container. It is sent
    only when `execution.hosts` is set; the first time in a quest, `run.log` and the console say so once.

12. **How many at once.** With `execution.hosts`, the count is the machines' free slots added up. A
    `settings_at_once` the person set stays a cap on the total.
13. **What counts as a machine failure.** Only the machine's trouble is run again elsewhere: ssh lost, no answer, its
    Python does not start, the shared drive is not seen. The simulation's own failure (an error, its time limit, no
    output) is its answer, handled as today: another machine would fail the same way.
14. **Conflicting settings.** `execution.hosts` with `sandbox: docker` or `background_jobs: true` stops before the run
    with one plain sentence: they cannot be used together yet; pick one.

## Left to the builder (decide by measurement, not judgement)

- **One ssh per run, or one resident worker per machine.** Measure first: the cost of starting one run over ssh
  (connection reuse with ssh ControlMaster included) against FI's evaluation times (about 1 to 100 s in the recorded
  studies), and the delay before a file written on one machine is seen on another (network drives often lag). Pick the
  simpler option that keeps the overhead under 5% of the run time on the recorded studies; report the numbers.
- Where a run's working copy lives: `core/outside_run.py::OutsideCopy` takes a `base`; on remote runs it must be on the
  shared drive (the system temp folder of the host is not visible remotely). The shared cache folder
  (`data/results/cache/`) is reached through a link today; check that a link on the shared drive resolves on Linux.
- The shape of the executor: most likely a remote executor that wraps the per-quest one, used by `OutsideExecutor`,
  with one `SettingsGate` per machine behind a small dispatcher that picks the machine with a free slot.
- The interpreter path on the remote machine (its own `python3`), and how the quest's `.venv` packages are matched
  (same version and platform: the shared `.venv` may be used as is; otherwise item 8).

## Builder's calls (from the builder check; the maintainer may overrule)

- The path map key is `execution.host_paths: {host prefix: remote prefix}`, longest prefix wins.
- A machine's capacity is read over ssh with a small Python probe: cores, memory, one-minute load; free slots =
  floor(cores - load); a machine with none gets no work until the next probe. Thread shares per run follow that
  machine's slots, not the host's `os.cpu_count()`.
- The packages bundle is built from the quest's resolved requirements, per remote Python ABI and platform.
- The shared cache link is checked once on the real mount with a scratch copy; when it does not resolve, each setting
  works out its shared values in its own copy (the existing fallback).

## What the code must change for this (found by the builder check)

- `OutsideCopy` defaults to the host's temp folder (`core/outside_run.py` ~78-85): remote runs must pass a `base` on
  the shared drive.
- The interpreter path: on a Windows host `VenvExecutor.python_path()` is `.venv\Scripts\python.exe`, and
  `OutsideExecutor` does not map it; a remote run uses the machine's own interpreter (item 8), never the host's path.
- `SettingsGate` measures the host's memory and the trial runner builds one gate (`core/trial_runner.py` ~619): one
  gate per machine, measured on that machine.
- `core/optimise.py::_drive` runs FI's own search driver through `OutsideExecutor`: the dispatcher wraps only the
  simulation's evaluation and check runs, never every `OutsideExecutor` use.

## Must hold (tests)

- A study run on two fake "machines" (two local executors behind the dispatcher) gives exactly the records one machine
  gives.
- A machine that dies mid-run: its settings rerun elsewhere, the record is complete and identical, one `run.log` line.
- No machine reachable: the run happens on the host, said once.
- A package missing on a machine: that machine is dropped, named in `run.log`, the study still completes.
- With `execution.hosts` absent, no code path changes (the existing suites pass untouched).
- A live check on two real Linux machines with a shared drive, on one recorded study: same records, the measured
  overhead, the host's load during the run.

## Surface

One YAML key (`execution.hosts`), one optional path map, plain `run.log` lines ("Running the settings on 3 machines
(48 free cores)", "Machine b stopped answering; its 4 settings run again on the others"). Docs: the capabilities
reference (near the background-jobs entry), USAGE, the README's settings list, the VS Code README (the setting is
written in the YAML the extension starts from).
