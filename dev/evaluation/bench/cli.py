"""``fi tools bench``: FI's self-benchmark, for FI's developers.

    fi tools bench check                         check every answer file
    fi tools bench reference Q4                  compute a task's reference values again
    fi tools bench record Q4 --out DIR [--settings models.yaml]
                                                 run a task's clean quest, keeping every model call (calls models)
    fi tools bench plant R1 --from-run DIR --out DIR [--file F --find X --replace Y]
    fi tools bench plant control --from-run DIR --step STEP --out DIR
    fi tools bench plant L1 --task Q11 --out DIR [--control] [--paper source.json]
    fi tools bench filter --clean DIR --planted DIR
    fi tools bench run DIR --mode partial        replay the calls before the planted one, real after (calls models)
    fi tools bench run DIR --mode replay --recording OLD_RUN
                                                 replay every call of an earlier run (no model is called)
    fi tools bench score BENCH_DIR [--out dev/evaluation/benchmark]
    fi tools bench hold-out ANSWER_FILE          hold a task back (its answer leaves the repository)

A run folder holds ``quest/<quest_id>/`` and ``bench/`` (runner.py). See dev/evaluation/bench/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from core.config import Config
from core.replay import CrossrefReplay, Recording

from . import answers as _answers
from . import catalogue
from . import plant as _plant
from . import report as _report
from . import runner as _runner
from . import score as _score
from . import validity as _validity

TASK_FILE = "task.json"
DEFAULT_REPORT_DIR = _answers.REPO / "dev" / "evaluation" / "benchmark"


def _save_task(run_dir: Path, task: dict[str, Any]) -> None:
    _out, bench = _runner.paths(run_dir)
    bench.mkdir(parents=True, exist_ok=True)
    (bench / TASK_FILE).write_text(json.dumps({"task": task["task"]}), encoding="utf-8")


def task_of(run_dir: Path) -> dict[str, Any]:
    _out, bench = _runner.paths(run_dir)
    return _answers.load_task(json.loads((bench / TASK_FILE).read_text(encoding="utf-8"))["task"])


def task_config(task: dict[str, Any], run_dir: Path, *, settings: dict[str, Any] | None = None,
                base: Config | None = None) -> Config:
    """The task's quest config with the benchmark's settings, its ask sentence, and ``settings`` (the models)."""
    cfg = base or Config.from_yaml(_answers.REPO / task["config"])
    return _runner.bench_config(cfg, output_dir=_runner.paths(run_dir)[0], ask=task["ask"], extra=settings)


async def record(task: dict[str, Any], run_dir: Path, *, settings: dict[str, Any] | None = None,
                 base: Config | None = None) -> dict[str, Any]:
    """A clean recording: every call real and kept, Crossref's answers recorded."""
    _save_task(run_dir, task)
    _out, bench = _runner.paths(run_dir)
    config = task_config(task, run_dir, settings=settings, base=base)
    return await _runner.run(config, run_dir, mode="record", crossref=CrossrefReplay(mode="record", out_dir=bench))


def plant(error: str, *, out: Path, from_run: Path | None = None, task: dict[str, Any] | None = None,
          step: str | None = None, file: str | None = None, find: str | None = None, replace: str | None = None,
          control: bool = False, paper: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prepare a planted (or control) run folder; nothing runs yet. Returns its ``plant.json``. ``paper``: L1's source
    (``doi``, ``title``, ``content``; by default the retracted Wakefield 1998, or with ``control`` Madsen 2002)."""
    out = Path(out)
    _o, bench = _runner.paths(out)
    if error.lower() == "control":
        if from_run is None or not step:
            raise ValueError("a control is a copy of a recorded run, run again from the same step as the planted runs "
                             "it is the control of: give --from-run and --step")
        _runner.fork(from_run, out)
        record = {"error": None, "control": True, "how": "edit", "from_step": step, "edits": []}
        task = task or task_of(from_run)
    elif error.upper() == "L1":
        if task is None:
            raise ValueError("L1 runs a task's quest from the start: name the task (--task)")
        record, hits = _plant.plant_l1(paper=paper or (_plant.MADSEN if control else None))
        if control:
            record.update(error=None, control=True)
        record["hits"] = hits
        root = None
    else:
        entry = catalogue.get(error)
        if not entry.planter:
            raise ValueError(f"{entry.id} ({entry.name}) has no planter yet (phase 0 builds R1, R2, N3, S1, L1)")
        if from_run is None:
            raise ValueError(f"{entry.id} is planted in a copy of a recorded run: give --from-run")
        task = task or task_of(from_run)
        root = _runner.fork(from_run, out)
        if entry.id == "R1":
            record = _plant.plant_r1(root, find=find, replace=replace)
        elif entry.id == "R2":
            recording = Recording.load(_runner.paths(from_run)[1] / "calls.jsonl")
            record, answers = _plant.plant_r2(recording)
            record["answers"] = {f"{n}#{i}": text for (n, i), text in answers.items()}
        else:
            if not (file and find is not None and replace is not None):
                raise ValueError(f"{entry.id} changes the code: give --file, --find and --replace")
            record = _plant.plant_code(entry.id, root, file=file, find=find, replace=replace)
    record.update(task=task["task"], source_run=str(Path(from_run).resolve()) if from_run else None)
    _save_task(out, task)
    _plant.write_record(bench, record)
    return record


def _answers_of(record: dict[str, Any]) -> dict[tuple[str, int], str]:
    out = {}
    for key, text in (record.get("answers") or {}).items():
        node, _, idx = key.rpartition("#")
        out[(node, int(idx))] = text
    return out


async def run_planted(run_dir: Path, *, mode: str, recording: Path | None = None,
                      settings: dict[str, Any] | None = None, base: Config | None = None,
                      crossref: CrossrefReplay | None = None) -> dict[str, Any]:
    """Run a folder ``plant`` prepared: ``partial`` (the calls before the planted one replayed from the source
    recording, every later call real) or ``replay`` (every call from ``recording``: an earlier run of the same plant)."""
    run_dir = Path(run_dir)
    _out, bench = _runner.paths(run_dir)
    record = _plant.read_record(bench) or {}
    task = task_of(run_dir)
    plants = _answers_of(record)
    step = record.get("from_step")
    if mode == "replay":
        if recording is None:
            raise ValueError("a full replay needs the run it replays (--recording)")
        rec_dir = Path(recording)
        calls = rec_dir / "bench" / "calls.jsonl" if rec_dir.is_dir() else rec_dir
        rec = Recording.load(calls)
        crossref_file = calls.parent / "crossref.json"
        crossref = crossref or (CrossrefReplay.load(crossref_file, out_dir=bench) if crossref_file.is_file()
                                else CrossrefReplay(mode="replay", out_dir=bench))
    else:
        rec = None
        crossref = crossref or CrossrefReplay(mode="record", out_dir=bench)
        if plants and record.get("source_run"):
            # The calls a rerun from this step makes again, as the source run made them: the ones before the planted one
            # are replayed from here.
            source = Path(record["source_run"])
            started = await _runner.fork_start_time(source, step) if step else None
            if started is None:
                raise ValueError(f"when the {step} step of {source} began cannot be read from its checkpoints, so its "
                                 "calls cannot be lined up with the rerun's")
            rec = _runner.segment_after(source, started)
    if record.get("how") == "input":  # a fresh run of the task's quest (L1)
        config = task_config(task, run_dir, settings=settings, base=base)
        return await _runner.run(config, run_dir, mode="record" if mode == "partial" and not plants else mode,
                                 recording=rec, plants=plants or None, crossref=crossref,
                                 extra_hits=record.get("hits"))
    if settings:
        raise ValueError("a copy runs with the settings of the run it was copied from (its approved plan names them); "
                         "--settings only applies to a run from the start (L1)")
    root = _runner.quest_root(run_dir)
    config = _runner.config_of(run_dir, source=True)
    return await _runner.run(config, run_dir, mode="record" if mode == "partial" and not plants else mode,
                             recording=rec, plants=plants or None, quest_id=root.name, from_step=step,
                             crossref=crossref)


def score_dir(bench_root: Path) -> list[dict[str, Any]]:
    """Score every run folder under ``bench_root`` (one with ``bench/run.json``), each planted run against its control
    (the control run of the same source and step), controls first."""
    runs = []
    for p in sorted(Path(bench_root).rglob("bench/run.json")):
        r = p.parent.parent
        try:
            _runner.quest_root(r)
            runs.append(r)
        except FileNotFoundError as e:
            print(f"[bench] {r} is left out: {e}")
    records = {r: (_plant.read_record(_runner.paths(r)[1]) or {}) for r in runs}

    def key(rec: dict[str, Any]) -> tuple[str, str]:
        source = rec.get("source_run")
        return (str(Path(source).resolve()) if source else str(rec.get("task")), str(rec.get("from_step")))

    controls: dict[tuple[str, str], dict[str, Any]] = {}
    outcomes = []
    for r in [r for r in runs if records[r].get("control")]:
        o = _score.score_run(r, task_of(r))
        if key(records[r]) in controls:
            print(f"[bench] {r.name}: a second control for the same run and step; the first one is used")
            outcomes.append({**o, "role": "control"})
            continue
        controls[key(records[r])] = o
        outcomes.append(o)
    for r in [r for r in runs if not records[r].get("control")]:
        rec = records[r]
        outcomes.append(_score.score_run(r, task_of(r), control=controls.get(key(rec)),
                                         needs_control=bool(rec.get("error"))))
    return outcomes


def _settings(path: str | None) -> dict[str, Any] | None:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fi tools bench", description="FI's self-benchmark (for FI's developers).")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check every answer file")
    r = sub.add_parser("reference", help="compute a task's reference values again")
    r.add_argument("task")
    r = sub.add_parser("record", help="run a task's clean quest, keeping every model call (calls models)")
    r.add_argument("task")
    r.add_argument("--out", required=True)
    r.add_argument("--settings", help="a YAML merged into the task's config: the models (provider:) to use")
    r = sub.add_parser("plant", help="prepare a planted or control run")
    r.add_argument("error", help="R1, R2, N3, S1, L1 or control")
    r.add_argument("--from-run")
    r.add_argument("--task")
    r.add_argument("--out", required=True)
    r.add_argument("--step", help="control: the step its planted runs start from (R1: claims, R2: writing, N3 and S1: run)")
    r.add_argument("--file")
    r.add_argument("--find")
    r.add_argument("--replace")
    r.add_argument("--control", action="store_true", help="L1: a source that was never retracted, as the control")
    r.add_argument("--paper", help="L1: a JSON file with the source to add (doi, title, content), one the topic would use")
    r = sub.add_parser("filter", help="whether a planted change moves the answer (run offline, no model)")
    r.add_argument("--clean", required=True)
    r.add_argument("--planted", required=True)
    r.add_argument("--python", help="the interpreter that runs the code (default: this one)")
    r.add_argument("--runs", type=int, help="trials per setting (default: the protocol's)")
    r = sub.add_parser("run", help="run a planted or control folder")
    r.add_argument("run_dir")
    r.add_argument("--mode", choices=("partial", "replay"), default="partial")
    r.add_argument("--recording", help="replay: the earlier run of the same plant whose calls are replayed")
    r.add_argument("--settings")
    r = sub.add_parser("score", help="score every run under a folder and write the report")
    r.add_argument("bench_root")
    r.add_argument("--out", default=str(DEFAULT_REPORT_DIR))
    r.add_argument("--name", default="self_benchmark")
    r.add_argument("--note", default="")
    r = sub.add_parser("hold-out", help="hold a task back: its answer file leaves the repository")
    r.add_argument("answer_file")
    args = p.parse_args(argv)
    try:
        return _main(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as e:  # AnswerError is a ValueError
        print(f"fi tools bench {args.cmd}: {e}")
        return 2


def _main(args: argparse.Namespace) -> int:
    if args.cmd == "check":
        problems = _answers.check_all()
        for line in problems:
            print(line)
        print(f"{len(_answers.files())} answer file(s); " + ("no problem found" if not problems else
                                                             f"{len(problems)} problem(s)"))
        return 1 if problems else 0
    if args.cmd == "reference":
        from . import reference

        print(json.dumps(reference.compute(args.task), indent=1))
        return 0
    if args.cmd == "record":
        print("[bench] this runs the task's quest with real model calls; every call is kept in <out>/bench/calls.jsonl")
        out = asyncio.run(record(_answers.load_task(args.task), Path(args.out), settings=_settings(args.settings)))
        print(json.dumps(out, indent=1))
        return 0
    if args.cmd == "plant":
        task = _answers.load_task(args.task) if args.task else None
        rec = plant(args.error, out=Path(args.out), from_run=Path(args.from_run) if args.from_run else None,
                    task=task, step=args.step, file=args.file, find=args.find, replace=args.replace,
                    control=args.control,
                    paper=json.loads(Path(args.paper).read_text(encoding="utf-8")) if args.paper else None)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("answers", "hits")}, indent=1))
        return 0
    if args.cmd == "filter":
        out = asyncio.run(_validity.check(Path(args.clean), Path(args.planted), task_of(Path(args.planted)),
                                          python=args.python, runs=args.runs))
        print(json.dumps(out, indent=1, default=str))
        return 0
    if args.cmd == "run":
        if args.mode == "partial":
            print("[bench] partial replay: every call after the planted one goes to the real model")
        out = asyncio.run(run_planted(Path(args.run_dir), mode=args.mode,
                                      recording=Path(args.recording) if args.recording else None,
                                      settings=_settings(args.settings)))
        print(json.dumps(out, indent=1))
        return 0
    if args.cmd == "score":
        outcomes = score_dir(Path(args.bench_root))
        js, md = _report.write(outcomes, Path(args.out), name=args.name, note=args.note)
        print(f"[bench] {len(outcomes)} run(s) scored: {md} and {js}")
        return 0
    if args.cmd == "hold-out":
        dest = _answers.hold_out(Path(args.answer_file))
        print(f"[bench] held back: the answer file is now {dest}; its hash is in "
              f"{_answers.ANSWERS_DIR / _answers.HELD_OUT_INDEX}")
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
