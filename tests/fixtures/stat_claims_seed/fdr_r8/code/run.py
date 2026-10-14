"""Runs this study: the simulation (when there is one), then the analysis. Written by Frontier Insight.

    python run.py

Needs the packages in requirements.txt. Inside a quest folder it works in ../run_output/ (so the quest's own results
stay as FI wrote them); copied elsewhere, in ./output/. Raw results go to raw/ there, and so do the figures and files
the analysis writes.
"""
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUEST = HERE.parent if (HERE.parent / ".fi").is_dir() else None
WORK = (QUEST / "run_output") if QUEST else (HERE / "output")


def cells(grid):
    axes = list(grid.items())
    if not axes:
        return [{}]
    return [dict(zip((a for a, _ in axes), combo)) for combo in itertools.product(*(v for _, v in axes))]


def fmt(value):
    if isinstance(value, list):  # an array setting: no comma, a key is split on them (kept equal to core/trial_runner.py)
        return "[" + ";".join(fmt(x) for x in value) + "]"
    return repr(value) if isinstance(value, float) and value.is_integer() else str(value)


def cell_key(cell):
    return ",".join(f"{axis}={fmt(value)}" for axis, value in cell.items())


def trial_seed(base, key, trial):
    digest = hashlib.sha256(f"{int(base)}|{key}|{int(trial)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_simulation():
    import importlib.util

    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("simulate", HERE / "simulate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["simulate"] = module
    spec.loader.exec_module(module)
    return module


def run_cell_child(study, index, out_path):
    """One setting's trials, in a process of its own (as FI runs them)."""
    module = load_simulation()
    entry = study["entry"]
    fn = getattr(module, entry)
    cell = cells(study.get("grid") or {})[index]
    key = cell_key(cell)
    runs = 1 if entry == "run_cell" else int(study.get("runs_per_setting") or 1)
    rows = []
    for trial in range(runs):
        seed = trial_seed(int(study.get("base_seed") or 0), "" if study.get("paired") else key, trial)
        try:
            value = fn(dict(cell)) if entry == "run_cell" else fn(dict(cell), trial, seed)
            if not isinstance(value, dict) or not value:
                raise TypeError(f"{entry}() must return a dict of numbers")
            bad = [k for k, v in value.items() if not is_number(v)]
            if bad:
                raise TypeError(f"{entry}() returned non-numbers for {bad[:5]}")
            rows.append({"trial": trial, "ok": True, "values": {str(k): float(v) for k, v in value.items()}})
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            traceback.print_exc()
            rows.append({"trial": trial, "ok": False, "reason": f"{type(exc).__name__}: {exc}"[:300]})
    Path(out_path).write_text(json.dumps(rows), encoding="utf-8")


def run_trials(study):
    out = []
    planned = 1 if study["entry"] == "run_cell" else int(study.get("runs_per_setting") or 1)
    for index, cell in enumerate(cells(study.get("grid") or {})):
        key = cell_key(cell)
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "rows.json"
            study_file = Path(tmp) / "study.json"
            study_file.write_text(json.dumps(study), encoding="utf-8")
            subprocess.call([sys.executable, str(HERE / "run.py"), "--cell", str(index), str(result), str(study_file)])
            rows = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else []
        metrics, trials_of = {}, {}
        for row in rows:
            if row.get("ok"):
                for name, value in row["values"].items():
                    metrics.setdefault(name, []).append(value)
                    trials_of.setdefault(name, []).append(row["trial"])
        ok = sum(1 for r in rows if r.get("ok"))
        out.append({
            "cell": cell, "key": key, "planned": planned, "ok": ok, "failed": planned - ok,
            "metrics": {
                name: {
                    "values": values, "trials": trials_of[name], "count": len(values),
                    "total": math.fsum(v for v in values if math.isfinite(v)),
                    "non_finite": sum(1 for v in values if not math.isfinite(v)),
                }
                for name, values in metrics.items()
            },
        })
        print(f"[run] {key or 'the one setting'}: {ok} of {planned} ok", file=sys.stderr)
    return {"schema": "fi.trials/v1", "thresholds": study.get("thresholds") or {}, "cells": out}


def evaluate_child(spec_path, out_path):
    """One design of the search, in a process of its own (as FI runs it)."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    module = load_simulation()
    entry = spec["entry"]
    fn = getattr(module, entry)
    rows = []
    for trial in spec["trials"]:
        try:
            value = fn(dict(spec["cell"])) if entry == "run_cell" else fn(dict(spec["cell"]), trial["trial"], trial["seed"])
            if not isinstance(value, dict) or not value:
                raise TypeError(f"{entry}() must return a dict of numbers")
            bad = [k for k, v in value.items() if not is_number(v)]
            if bad:
                raise TypeError(f"{entry}() returned non-numbers for {bad[:5]}")
            rows.append({"trial": trial["trial"], "ok": True, "values": {str(k): float(v) for k, v in value.items()}})
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            traceback.print_exc()
            rows.append({"trial": trial["trial"], "ok": False, "reason": f"{type(exc).__name__}: {exc}"[:300]})
    Path(out_path).write_text(json.dumps(rows), encoding="utf-8")


def search(study):
    """The search for the best design, as FI ran it (fi_search.py is FI's own search, copied here): the coarse scan
    first when the plan has one, then every design in a process of its own, with the same seed."""
    sys.path.insert(0, str(HERE))
    import fi_search

    block = study["optimisation"]
    entry = study["entry"]
    base = int(study.get("base_seed") or 0)
    runs = 1 if entry == "run_cell" else int(study.get("runs_per_evaluation") or 1)

    def scan(grid):
        summary = run_trials({"entry": entry, "grid": grid, "runs_per_setting": runs, "base_seed": base,
                              "paired": True, "thresholds": study.get("thresholds") or {}})
        (WORK / "raw" / "trials.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.environ["FI_TRIALS"] = "raw/trials.json"
        return fi_search.scan_rows(summary, block)

    def evaluate(cell):
        trials = [{"trial": t, "seed": None if entry == "run_cell" else trial_seed(base, "", t)} for t in range(runs)]
        with tempfile.TemporaryDirectory() as tmp:
            spec, result = Path(tmp) / "spec.json", Path(tmp) / "rows.json"
            spec.write_text(json.dumps({"entry": entry, "cell": cell, "trials": trials}), encoding="utf-8")
            subprocess.call([sys.executable, str(HERE / "run.py"), "--evaluate", str(spec), str(result)])
            rows = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else []
        if len(rows) != len(trials) or not all(r.get("ok") for r in rows):
            raise RuntimeError(next((r.get("reason") for r in rows if r.get("reason")),
                                    "the evaluation's process stopped before it reported"))
        names = set.intersection(*(set(r["values"]) for r in rows))
        return {name: sum(r["values"][name] for r in rows) / len(rows) for name in names}

    # A search FI stopped at its time limit is repeated to the same point, not further.
    outcome = fi_search.run_sync(block, evaluate, seed=base, method=study.get("method"), scan_step=scan,
                                 max_evaluations=study.get("max_evaluations"), rounds=study.get("rounds"))
    record = fi_search.best_design(outcome, block, seed=base, check=study.get("check_settings"))
    (WORK / "raw" / "optimisation_ledger.jsonl").write_bytes(
        ("\n".join(fi_search.ledger_lines(outcome, block)) + "\n").encode("utf-8"))
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    (WORK / "results" / "best_design.json").write_bytes((json.dumps(record, indent=1) + "\n").encode("utf-8"))
    os.environ["FI_OPTIMISATION"] = "raw/optimisation_ledger.jsonl"
    os.environ["FI_BEST_DESIGN"] = "results/best_design.json"
    print(f"[run] {record['says']}", file=sys.stderr)
    if not any(r["status"] == "ok" for r in outcome["rows"]):
        sys.exit("[run] no design produced a result: see the errors above")


def stage_inputs():
    """The quest's own data/ and example inputs, for a script that reads them by their relative path."""
    if QUEST is None:
        return
    data = QUEST / "data"
    if data.is_dir() and not (WORK / "data").exists():
        size = sum(f.stat().st_size for f in data.rglob("*") if f.is_file())
        if size <= 200 * 1024 * 1024:
            shutil.copytree(data, WORK / "data")
        else:
            print("[run] data/ is over 200 MB: not copied; copy what the study reads into run_output/data/",
                  file=sys.stderr)
    examples = QUEST / "inputs" / "examples"
    if examples.is_dir():
        os.environ.setdefault("FI_INPUT_DIR", str(examples))


def main():
    if sys.argv[1:2] == ["--cell"]:
        study_path = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / "study.json"
        study = json.loads(study_path.read_text(encoding="utf-8"))
        os.environ["FI_THRESHOLDS"] = json.dumps(study.get("thresholds") or {})
        run_cell_child(study, int(sys.argv[2]), sys.argv[3])
        return
    if sys.argv[1:2] == ["--evaluate"]:
        study = json.loads((HERE / "study.json").read_text(encoding="utf-8"))
        os.environ["FI_THRESHOLDS"] = json.dumps(study.get("thresholds") or {})
        evaluate_child(sys.argv[2], sys.argv[3])
        return
    (WORK / "raw").mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    os.environ.setdefault("FI_REPLICATE_SEED", "0")
    os.environ.setdefault("FI_REPLICATE_INDEX", "0")
    os.environ["FI_RAW_DIR"] = "raw"
    stage_inputs()
    study_path = HERE / "study.json"
    simulate = HERE / "simulate.py"
    study = json.loads(study_path.read_text(encoding="utf-8")) if study_path.is_file() else {}
    if study.get("simulate") is False:
        pass  # this study's analysis does not use a simulation script
    elif study.get("optimisation") and study.get("entry") and simulate.is_file():
        os.environ["FI_THRESHOLDS"] = json.dumps(study.get("thresholds") or {})
        search(study)
        os.environ.pop("FI_THRESHOLDS")
    elif study.get("entry") and simulate.is_file():
        os.environ["FI_THRESHOLDS"] = json.dumps(study.get("thresholds") or {})
        summary = run_trials(study)
        if summary["cells"] and not any(c["ok"] for c in summary["cells"]):
            sys.exit("[run] no setting produced a result: see the errors above")
        (WORK / "raw" / "trials.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.environ.pop("FI_THRESHOLDS")
        os.environ["FI_TRIALS"] = "raw/trials.json"
    elif simulate.is_file():
        code = subprocess.call([sys.executable, str(simulate)])
        if code:
            sys.exit(code)
    sys.exit(subprocess.call([sys.executable, str(HERE / "experiment.py")]))


if __name__ == "__main__":
    main()
