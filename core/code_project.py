"""The quest's ``code/`` folder as a project that runs on its own.

FI runs the experiment through its own runner (:mod:`core.trial_runner`, :mod:`core.split_run`), but a person who
opens ``code/`` should find something they can run and keep: a ``README.md`` saying in plain words what it computes and
which file is which, a ``requirements.txt``, and ``run.py``, one command that does what FI does (the simulation, then
the analysis). :func:`refresh` writes them after the code is written and after every run or refine that changed it, so
they follow the scripts. A file a person edited is never overwritten: FI records the hash of what it last wrote
(``.fi/code_project.json``) and leaves a file whose bytes differ, or one FI did not write, exactly as it is.

``run.py`` is standard library only. For a simulation that follows the trial contract it repeats what FI's runner does
for replicate 0 (the same cells, the same seeds, one process per setting, the same per-setting summary); the settings
it needs are in ``study.json`` beside it. It works in a folder of its own (``run_output/`` beside ``code/`` inside a
quest, ``output/`` when the folder has been copied elsewhere), so running it never touches the quest's own results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import experiment_deps as _experiment_deps
from .execution import scrub_secrets
from .proc_tree import ProcessTree as _ProcessTree
from . import split_run as _split_run
from . import trial_runner as _trial_runner

RECORD = Path(".fi") / "code_project.json"
INSTALLED = Path(".fi") / "installed_deps.json"
README = "README.md"
REQUIREMENTS = "requirements.txt"
RUN = "run.py"
STUDY = "study.json"
#: The search for the best design, copied beside run.py so it repeats FI's search without FI (core/optimise_search.py).
SEARCH = "fi_search.py"

RUN_SOURCE = r'''"""Runs this study: the simulation (when there is one), then the analysis. Written by Frontier Insight.

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


# @IN_RUN@ (the text of core/in_run.py: what a run returned, read as FI reads it)

MAX_SETTINGS = 500000  # the most settings listed from one grid (core.optimise_search.MAX_LISTED_SETTINGS)


def cells(grid):
    size = math.prod(len(v) for v in grid.values())
    if size > MAX_SETTINGS:
        raise SystemExit("the study asks for {:,} settings; this script does not list or run more than {:,}".format(size, MAX_SETTINGS))
    axes = list(grid.items())
    if not axes:
        return [{}]
    return [dict(zip((a for a, _ in axes), combo)) for combo in itertools.product(*(v for _, v in axes))]


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
    inside = study.get("in_each_run") or {}
    levels = level_cells(inside)
    runs = 1 if entry == "run_cell" else int(study.get("runs_per_setting") or 1)
    rows = []
    for trial in range(runs):
        seed = trial_seed(int(study.get("base_seed") or 0), "" if study.get("paired") else key, trial)
        try:
            given = dict(cell, **inside)  # every value each run compares inside it, as FI gives them
            value = fn(given) if entry == "run_cell" else fn(given, trial, seed)
            kept, why = read_returned(value, levels, entry)
            if kept is None:
                raise TypeError(why)
            rows.append({"trial": trial, "ok": True, "values": kept})
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            traceback.print_exc()
            rows.append({"trial": trial, "ok": False, "reason": f"{type(exc).__name__}: {exc}"[:300]})
    Path(out_path).write_text(json.dumps(rows), encoding="utf-8")


def run_trials(study):
    out = []
    planned = 1 if study["entry"] == "run_cell" else int(study.get("runs_per_setting") or 1)
    levels = level_cells(study.get("in_each_run") or {})
    for index, cell in enumerate(cells(study.get("grid") or {})):
        key = cell_key(cell)
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "rows.json"
            study_file = Path(tmp) / "study.json"
            study_file.write_text(json.dumps(study), encoding="utf-8")
            subprocess.call([sys.executable, str(HERE / "run.py"), "--cell", str(index), str(result), str(study_file)])
            rows = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else []
        out.extend(summary_cells(cell, key, planned, rows, levels))
        ok = sum(1 for r in rows if r.get("ok"))
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
'''
# The text of core/in_run.py goes where the marker is: run.py reads what a run returned as FI does.
RUN_SOURCE = RUN_SOURCE.replace(
    "# @IN_RUN@ (the text of core/in_run.py: what a run returned, read as FI reads it)\n",
    (Path(__file__).with_name("in_run.py")).read_text(encoding="utf-8"), 1)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_record(quest_root: Path) -> dict[str, str]:
    try:
        data = json.loads((quest_root / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _installed(quest_root: Path) -> list[str] | None:
    try:
        data = json.loads((quest_root / INSTALLED).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return [str(d).strip() for d in data if str(d).strip()] if isinstance(data, list) else None


def record_installed(quest_root: Path, deps: list[str]) -> None:
    """Remember the packages FI really installed into the quest's environment (best effort)."""
    try:
        path = Path(quest_root) / INSTALLED
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted({d.strip() for d in deps if d and d.strip()}), indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def requirements_for(code_dir: Path, deps: list[str], *, installed: bool = False) -> list[str]:
    """What the scripts need: ``deps`` (the list FI actually installed, when ``installed``) plus each package the
    scripts import (outside an optional ``try/except ImportError``) that the list lacks. A quest environment can see
    packages it never installed (the ones already on the machine), so the imports are checked either way."""
    deps_mod = _experiment_deps
    lines = [d.strip() for d in deps if d and d.strip()]
    local = deps_mod.local_module_names(code_dir)
    have = {deps_mod.normalize(deps_mod.requirement_name(d) or d) for d in lines}
    # The modules the listed packages already provide (opencv-python-headless is cv2: no second cv2 package).
    provided = {m for d in lines for m in (deps_mod.import_names(d) or [])}
    scripts = deps_mod.code_sources(code_dir)  # every script, sub-folders too, except run.py
    full: set[str] = set()
    for script in scripts:
        full |= deps_mod.imported_modules(script, optional=False) or set()
    # A well-known module under a shared namespace (google.protobuf is protobuf) before the namespace is dropped.
    wanted = [m for m in sorted(deps_mod.IMPORT_TO_PIP) if "." in m and any(n == m or n.startswith(m + ".") for n in full)]
    wanted += sorted(deps_mod.third_party_of(scripts, local, optional=False) - deps_mod.NOT_A_PACKAGE)
    for name in wanted:
        pip = deps_mod.pip_name(name) or name
        if name in provided or deps_mod.normalize(pip) in have or deps_mod.normalize(name) in have:
            continue
        lines.append(pip)
        have.add(deps_mod.normalize(pip))
    return lines


def simulation_left_out(code_dir: Path, split: bool) -> bool:
    """A ``simulate.py`` is there but this quest's two-script split is off: FI does not run it, so ``run.py`` must not."""
    return not split and (code_dir / _split_run.SIMULATE_NAME).is_file()


def study_of(code_dir: Path, protocol: dict[str, Any] | None, *, split: bool = True) -> dict[str, Any] | None:
    """The settings ``run.py`` needs to run FI's trial contract, or None when the simulation is not on that contract
    (or the two-script split is off, so FI itself does not run the simulation)."""
    simulate = code_dir / _split_run.SIMULATE_NAME
    if not split or not simulate.is_file():
        return None
    entries = _trial_runner.entries(simulate)
    entry = "run_trial" if "run_trial" in entries else "run_cell" if "run_cell" in entries else ""
    if not entry:
        return None
    protocol = protocol or {}
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    block = protocol.get("optimisation")
    if isinstance(block, dict):
        # A search for the best design: run.py repeats FI's search (fi_search.py beside it is FI's own search code).
        from . import optimisation_plan as _optimisation_plan

        # The block as FI's search reads it (through the plan's own check), so run.py searches the same way.
        block = _optimisation_plan.normalize(block)[0] or block
        settings = block.get("numerical_settings") if isinstance(block.get("numerical_settings"), dict) else {}
        # The seed FI's search used, and where the time limit stopped it, from FI's own record of it.
        try:
            done = json.loads((code_dir.parent / "results" / "best_design.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            done = {}
        done = done if isinstance(done, dict) else {}
        evaluations = done.get("evaluations") if isinstance(done.get("evaluations"), dict) else {}
        study = {
            "entry": entry,
            "optimisation": block,
            "method": _optimisation_plan.effective_method(block)[0],
            "runs_per_evaluation": int(block.get("runs_per_evaluation") or 1),
            "base_seed": int(done["seed"]) if isinstance(done.get("seed"), int) else 0,
            "thresholds": thresholds,
            "check_settings": {str(n): _optimisation_plan.check_levels(s if isinstance(s, dict) else {"search": s})[0]
                               for n, s in settings.items()},
        }
        if evaluations.get("stopped_because") == "time" and isinstance(evaluations.get("search"), int):
            study["max_evaluations"] = evaluations["search"]
        # The rounds a person asked the search to continue with (core/optimise_refine.py): run.py continues it the same.
        from . import optimise as _optimise

        rounds = _optimise.read_rounds(code_dir.parent, block)
        if rounds:
            study["rounds"] = [{k: r[k] for k in ("added", "target", "search_target") if k in r} for r in rounds]
        return study
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    metrics = protocol.get("metrics") or []
    inside = protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else {}
    return {
        "entry": entry,
        "grid": grid,
        **({"in_each_run": inside} if inside else {}),
        "runs_per_setting": int(protocol.get("runs_per_setting") or 1),
        "base_seed": 0,
        "paired": any(isinstance(m, dict) and m.get("paired") for m in metrics),
        "thresholds": protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {},
    }


def study_differences(code_dir: Path, protocol: dict[str, Any] | None, *, split: bool = True) -> list[str]:
    """One plain sentence per setting of ``code/study.json`` (what ``run.py`` runs) that is not what the protocol says: the
    runs per setting and each setting to vary. The protocol is the source of truth. Nothing when there is no
    ``study.json``, the study is a search for the best design, or the protocol says nothing about the setting (a protocol
    without a run count leaves ``study.json`` its own)."""
    path = Path(code_dir) / STUDY
    wanted = study_of(Path(code_dir), protocol, split=split)
    try:
        have = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(have, dict) or not isinstance(wanted, dict) or "optimisation" in wanted:
        return []
    protocol = protocol if isinstance(protocol, dict) else {}
    found: list[str] = []
    runs = protocol.get("runs_per_setting")
    if isinstance(runs, (int, float)) and not isinstance(runs, bool) and have.get("runs_per_setting") != wanted.get("runs_per_setting"):
        found.append(f"it runs {have.get('runs_per_setting')} per setting, the protocol says {wanted.get('runs_per_setting')}")
    grid = wanted.get("grid") if isinstance(wanted.get("grid"), dict) else {}
    have_grid = have.get("grid") if isinstance(have.get("grid"), dict) else {}
    if not grid:
        return found  # the protocol names no setting to vary: nothing to compare the file's with
    for name, values in grid.items():
        if have_grid.get(name) != values:
            found.append(f"it varies `{name}` over {have_grid.get(name)}, the protocol says {values}")
    for name in have_grid:
        if name not in grid:
            found.append(f"it varies `{name}`, which the protocol does not")
    inside, have_inside = wanted.get("in_each_run") or {}, have.get("in_each_run") or {}
    for name, values in inside.items():
        if have_inside.get(name) != values:
            found.append(f"each run compares `{name}` over {have_inside.get(name)}, the protocol says {values}")
    for name in have_inside:
        if name not in inside:
            found.append(f"each run compares `{name}`, which the protocol does not")
    return found


def _readme(code_dir: Path, *, title: str, question: str, study: dict[str, Any] | None, split: bool = True,
            layout: list[str] | None = None) -> str:
    two = (code_dir / _split_run.SIMULATE_NAME).is_file() and split
    lines = [f"# {title or 'Study code'}", ""]
    if question:
        lines += [question.strip(), ""]
    lines += ["## Run it", "", "```", "pip install -r requirements.txt", "python run.py", "```", ""]
    lines += [
        "`run.py` works in its own folder (`run_output/` next to this one inside a Frontier Insight quest, `output/` "
        "here if you copied this folder elsewhere), so the quest's own results are never touched. "
        "The raw results, the figures and the numbers land there.", "",
    ]
    if two and study and study.get("optimisation"):
        block = study["optimisation"]
        budget = block.get("evaluation_budget") or {}
        scan = " It first runs the coarse scan the plan names, then" if block.get("grid") else " It"
        lines += [
            f"This study searches for the best design.{scan} searches the way Frontier Insight did ({study['method']}, "
            f"at most {budget.get('starts', '?')} starting points x {budget.get('per_start', '?')} evaluations, seed {study.get('base_seed', 0)}), "
            "calling the simulation once per design, each in a process of its own; then it runs the analysis. The "
            "record of every evaluation goes to `raw/optimisation_ledger.jsonl` and the best design to "
            "`results/best_design.json` in the output folder.", "",
        ]
    elif two:
        lines += ["It runs the simulation, then the analysis.", ""]
        if study:
            settings = 1
            for values in study["grid"].values():
                settings *= max(1, len(values))
            runs = 1 if study["entry"] == "run_cell" else study["runs_per_setting"]
            lines += [
                f"It runs {settings} setting(s) x {runs} run(s) each, once (seed 0). "
                "Frontier Insight repeats the study with more seeds, so the numbers in its paper can differ from "
                "a single `run.py`.", "",
            ]
    else:
        lines += ["It runs `experiment.py`.", ""]
    lines += ["## Files", ""]
    if two:
        lines += ["- `simulate.py`: what one run of the study does (the simulation itself).",
                  "- `experiment.py`: reads the simulation's results, computes the numbers and draws the figures."]
        lines += list(layout or [])
    else:
        lines += ["- `experiment.py`: the study: it runs, computes the numbers and draws the figures."]
    if study and study.get("optimisation"):
        lines += ["- `fi_search.py`: the search for the best design (Frontier Insight's own, standard library only; a "
                  "scipy or Optuna method runs only when that package is installed)."]
    lines += ["- `run.py`: the one command that runs it all.",
              "- `requirements.txt`: the packages it needs, with the versions FI used.",
              "- `CHANGELOG.md`: what changed each time the code was changed, why, and whether the results changed."]
    if study or not two and (code_dir / _split_run.SIMULATE_NAME).is_file():
        lines += ["- `study.json`: the settings `run.py` uses (from the plan's frozen protocol)."]
    lines += ["", "## Changes and older versions", "",
              "Every change to this folder is one commit in its own git history (`git log`). To go back to an "
              "earlier version of a file: `git checkout <id> -- <file>`.", "",
              "Frontier Insight keeps `README.md`, `requirements.txt`, `run.py` and `study.json` up to date"
              + (" (and `METHODS.md` and `tests/test_oracles.py`)" if layout else "") + ". "
              "If you edit one of them, it asks before replacing it (an unattended run keeps your version and says "
              "so); to get its version back, delete yours.", ""]
    return "\n".join(lines)


def _json_file(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, str], log: Any = None) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if data:
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        elif path.exists():
            path.unlink()
        return True
    except OSError as exc:
        if log is not None:
            log.warning("[code] could not write %s (%s): your edits to code/ may not be recognised next time", path.name, exc)
        return False


CONFLICTS = Path(".fi") / "code_project_conflicts.json"
ASKED = Path(".fi") / "code_project_asked.json"


def unasked_conflicts(quest_root: Path) -> list[str]:
    """Files a person edited that FI would now change and has not yet asked about."""
    wanted = _json_file(Path(quest_root) / CONFLICTS)
    asked = _json_file(Path(quest_root) / ASKED)
    return sorted(name for name, sha in wanted.items() if asked.get(name) != sha)


def mark_asked(quest_root: Path, names: list[str]) -> None:
    wanted = _json_file(Path(quest_root) / CONFLICTS)
    asked = _json_file(Path(quest_root) / ASKED)
    for name in names:
        if name in wanted:
            asked[name] = wanted[name]
    if not _write_json(Path(quest_root) / ASKED, asked):
        raise OSError("could not record which files were asked about")


def refresh(quest_root: Path, *, deps: list[str] | None = None, protocol: dict[str, Any] | None = None,
            title: str = "", question: str = "", split: bool = True, log: Any = None,
            extra_files: dict[str, str] | None = None, readme_files: list[str] | None = None) -> list[str]:
    """Write or update the project files in ``code/``; return the names written. Best effort: a quest never stops
    for these files. A file a person edited is not replaced; it is listed for :func:`unasked_conflicts`.
    ``extra_files`` (``{path in code/: text}``) and ``readme_files`` (README lines for them) are the research-tool
    layout's own files (``core/code_layout.py``), kept the same way."""
    quest_root = Path(quest_root)
    code_dir = quest_root / "code"
    if not (code_dir / "experiment.py").is_file():
        return []
    study = study_of(code_dir, protocol, split=split)
    installed = _installed(quest_root)
    wanted = {
        README: _readme(code_dir, title=title, question=question, study=study, split=split, layout=readme_files),
        REQUIREMENTS: "\n".join(
            requirements_for(code_dir, installed if installed is not None else list(deps or []),
                             installed=installed is not None)) + "\n",
        RUN: RUN_SOURCE,
    }
    wanted.update(extra_files or {})
    if study:
        wanted[STUDY] = json.dumps(study, indent=2) + "\n"
    elif simulation_left_out(code_dir, split):
        wanted[STUDY] = json.dumps({"simulate": False}, indent=2) + "\n"
    if study and study.get("optimisation"):
        from . import optimise_search as _optimise_search

        wanted[SEARCH] = Path(_optimise_search.__file__).read_text(encoding="utf-8")
    record = _read_record(quest_root)
    written: list[str] = []
    conflicts: dict[str, str] = {}
    for name, text in wanted.items():
        path = code_dir / name
        try:
            if path.is_file():
                current = path.read_bytes().decode("utf-8")
                plain = current.replace("\r\n", "\n")  # a checkout that turned line endings into CRLF is not an edit
                if plain == text:
                    record[name] = _sha(current)
                    continue
                if record.get(name) not in (_sha(current), _sha(plain)):
                    conflicts[name] = _sha(text)  # edited by a person, or not written by FI: left as it is
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            record[name] = _sha(text)
            written.append(name)
        except (OSError, UnicodeDecodeError) as exc:
            if log is not None:
                log.warning("[code] could not write code/%s: %s", name, exc)
    for name in (STUDY, SEARCH):
        stale = code_dir / name
        try:
            if name not in wanted and stale.is_file() and record.get(name) == _sha(stale.read_bytes().decode("utf-8")):
                stale.unlink()  # the simulation left the trial contract (or the search): run.py must not use an old one
                record.pop(name, None)
                written.append(name)
        except (OSError, UnicodeDecodeError):
            pass
    _write_json(quest_root / RECORD, record, log)
    _write_json(quest_root / CONFLICTS, conflicts, log)
    if written and log is not None:
        log.info("[code] code/ is a project you can run on its own: updated %s", ", ".join(written))
    return written


def pin(python: Path | str, deps: list[str]) -> list[str]:
    """``deps`` with each package's exact installed version (``numpy==1.26.4``), read from the environment
    ``python`` belongs to. A requirement that cannot be resolved there (a URL, a name not installed) is kept as given."""
    import subprocess

    names = {}
    for dep in deps:
        name = _experiment_deps.requirement_name(dep) or ""
        if name and ";" not in dep and "@" not in dep:
            names[dep] = name
    if not names:
        return list(deps)
    script = (
        "import json,sys\nfrom importlib import metadata\nout={}\nfor n in json.loads(sys.argv[1]):\n"
        "    try:\n        out[n]=metadata.version(n)\n    except Exception:\n        pass\nprint(json.dumps(out))\n"
    )
    try:
        done = subprocess.run([str(python), "-c", script, json.dumps(sorted(set(names.values())))],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
                              env=scrub_secrets())
        versions = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else {}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return list(deps)
    return [f"{names[d]}=={versions[names[d]]}" if d in names and names[d] in versions else d for d in deps]


def _git(code_dir: Path, *args: str) -> Any:
    import subprocess

    import os

    # No hook and no file-system monitor runs at FI's own commits, and no system or personal git config is read:
    # code/ is written by the experiment's code, which could otherwise plant one. The null device, named absolutely, is
    # never a folder of hooks.
    null = "\\\\.\\nul" if os.name == "nt" else os.devnull
    # And git never looks above code/ for a repository: with code/.git gone or broken, a commit must fail, never land in
    # a repository that happens to hold the quest folder (with whatever that repository has uncommitted).
    env = {**scrub_secrets(), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CEILING_DIRECTORIES": str(Path(code_dir).resolve().parent)}
    return subprocess.run(
        ["git", "-c", "user.name=Frontier Insight", "-c", "user.email=fi@localhost", "-c", "commit.gpgsign=false",
         "-c", "core.autocrlf=false", "-c", f"core.hooksPath={null}", "-c", "core.fsmonitor=false",
         "-c", f"safe.directory={code_dir.as_posix()}", *args],
        cwd=code_dir, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env,
    )


def record_change(quest_root: Path, note: str, *, category: str = "changed", results: str | None = None,
                  undo_pending: str | None = None, log: Any = None) -> bool:
    """One commit in ``code/``'s own git history, and one entry in ``code/CHANGELOG.md``, when the folder changed since
    the last one (your own edits included). Returns whether it made a commit (a change to CHANGELOG.md alone is
    committed with no entry about itself). Best effort; without git it says so.

    ``category`` is the kind of change (:data:`core.changelog.CATEGORIES`), decided by the step that made it; a change
    to ``README.md`` alone is always ``tidied``. ``results`` is the entry's "Did the results change" line when it is
    known now (:mod:`core.changelog`); without it the line says "not measured yet" until :func:`fill_pending`.
    ``undo_pending``: this change puts back a version that ran before, so every earlier entry still waiting for its run
    says it was not measured, and why."""
    import shutil
    import subprocess
    import time

    from . import changelog as _changelog

    code_dir = Path(quest_root) / "code"
    if not code_dir.is_dir():
        return False
    if shutil.which("git") is None:
        if log is not None:
            log.warning("[code] git was not found: code/ has no change history this time")
        return False
    try:
        if not (code_dir / ".git").exists():
            _git(code_dir, "init", "-q")
            if not (code_dir / ".git").is_dir():
                if log is not None:
                    log.warning("[code] git could not start a history in code/: no .git folder was made")
                return False
            for key, value in (("core.autocrlf", "false"), ("user.name", "Frontier Insight"),
                               ("user.email", "fi@localhost"), ("commit.gpgsign", "false")):
                _git(code_dir, "config", "--local", key, value)  # so a person's own commits and checkouts here behave
            exclude = code_dir / ".git" / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
        _git(code_dir, "add", "-A")
        changed = _git(code_dir, "diff", "--cached", "--quiet").returncode
        if changed != 1:
            if changed != 0:
                if log is not None:
                    log.warning("[code] git could not compare code/ with its last version (code %s)", changed)
            elif undo_pending:
                fill_pending(quest_root, _changelog.not_measured(undo_pending), together=False, log=log)
            return False
        files = _git(code_dir, "diff", "--cached", "--name-status").stdout.split("\n")
        changed_files = [ln.split("\t")[-1].strip() for ln in files if ln.strip() and "CHANGELOG.md" not in ln]
        if not changed_files:
            # Only CHANGELOG.md (a person's note in it, or a fill-in that could not be committed): committed as it is,
            # with no entry about itself.
            done = _git(code_dir, "commit", "-q", "-m", "CHANGELOG.md")
            if done.returncode != 0:
                if log is not None:
                    log.warning("[code] could not commit code/CHANGELOG.md: %s", (done.stderr or "").strip()[-200:])
                return False
            if undo_pending:
                fill_pending(quest_root, _changelog.not_measured(undo_pending), together=False, log=log)
            return True  # a new commit (of the same code): whoever asked reads the commit again
        names = ", ".join(ln.split("\t", 1)[-1].strip() for ln in files if ln.strip() and "CHANGELOG.md" not in ln)
        kind = _changelog.category(category, log=log)
        line = results if results is not None else _changelog.PENDING
        if _changelog.documentation_only(changed_files):
            kind, line = "tidied", _changelog.documentation_line(changed_files)
        entry = (f"{_changelog.heading(time.strftime('%Y-%m-%d %H:%M'), kind, note)}\n\n"
                 f"Files: {names or 'none'}\n\n{line}\n\n")
        changelog = code_dir / "CHANGELOG.md"
        before = changelog.read_bytes() if changelog.is_file() else None
        head = before.decode("utf-8") if before is not None else _changelog.HEADER
        if undo_pending:
            head = head.replace(_changelog.PENDING, _changelog.not_measured(undo_pending))
        changelog.write_bytes((head + entry).encode("utf-8"))
        _git(code_dir, "add", "-A")
        done = _git(code_dir, "commit", "-q", "-m", note)
        if done.returncode != 0:
            _git(code_dir, "reset", "-q")
            if before is None:
                changelog.unlink(missing_ok=True)
            else:
                changelog.write_bytes(before)
            if log is not None:
                log.warning("[code] could not record this change in code/'s history: %s", (done.stderr or "").strip()[-200:])
            return False
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        if log is not None:
            log.warning("[code] could not record this change in code/'s history: %s", exc)
        return False


def record_note(quest_root: Path, note: str, *, log: Any = None) -> bool:
    """An entry in ``code/CHANGELOG.md`` (and its commit) when nothing else in the folder changed: something about the
    code a person should read beside its history, such as the study's results changing after a change kept earlier.
    Without a history yet (no git), nothing is written."""
    import shutil
    import subprocess
    import time

    from . import changelog as _changelog

    code_dir = Path(quest_root) / "code"
    if not (code_dir / ".git").exists() or shutil.which("git") is None:
        return False
    changelog = code_dir / "CHANGELOG.md"
    try:
        head_text = changelog.read_bytes().decode("utf-8") if changelog.is_file() else _changelog.HEADER
        changelog.write_bytes((head_text + f"## {time.strftime('%Y-%m-%d %H:%M')} - Note: {note}\n\n").encode("utf-8"))
    except OSError as exc:
        if log is not None:
            log.warning("[code] could not add a note to code/CHANGELOG.md: %s", exc)
        return False
    try:
        _git(code_dir, "add", "-A")
        done = _git(code_dir, "commit", "-q", "-m", note)
    except (OSError, subprocess.SubprocessError) as exc:
        if log is not None:
            log.warning("[code] could not record the note in code/'s history: %s", exc)
        return False
    return done.returncode == 0


def fill_pending(quest_root: Path, line: str, *, together: bool = True, log: Any = None) -> int:
    """Put ``line`` (a "Did the results change" line, :mod:`core.changelog`) in place of every entry of
    ``code/CHANGELOG.md`` still waiting for its run, and commit only that file. With ``together``, an entry measured with
    others by the same run says so. Returns how many entries were filled; best effort, never raises."""
    import shutil
    import subprocess

    from . import changelog as _changelog

    code_dir = Path(quest_root) / "code"
    changelog = code_dir / "CHANGELOG.md"
    if not changelog.is_file() or not (code_dir / ".git").exists() or shutil.which("git") is None:
        return 0
    before: bytes | None = None
    try:
        before = changelog.read_bytes()
        text = before.decode("utf-8")
        count = text.count(_changelog.PENDING)
        if not count:
            return 0
        filled = _changelog.together(line, count - 1) if together else line
        changelog.write_bytes(text.replace(_changelog.PENDING, filled).encode("utf-8"))
        # Only CHANGELOG.md is committed: anything else changed in code/ is left for the next recorded change.
        _git(code_dir, "add", "--", "CHANGELOG.md")
        done = _git(code_dir, "commit", "-q", "-m", "CHANGELOG: whether the results changed, from the run that followed",
                    "--", "CHANGELOG.md")
        if done.returncode != 0:
            changelog.write_bytes(before)
            _git(code_dir, "reset", "-q", "--", "CHANGELOG.md")
            if log is not None:
                log.warning("[code] could not record in code/CHANGELOG.md whether the results changed: %s",
                            (done.stderr or "").strip()[-200:])
            return 0
        return count
    except (OSError, UnicodeDecodeError, subprocess.SubprocessError) as exc:
        # Put the file back as it was committed: an uncommitted CHANGELOG.md would read as a change to the code.
        if before is not None:
            try:
                changelog.write_bytes(before)
            except OSError:
                pass
            try:
                _git(code_dir, "reset", "-q", "--", "CHANGELOG.md")
            except (OSError, subprocess.SubprocessError):
                pass
        if log is not None:
            log.warning("[code] could not record in code/CHANGELOG.md whether the results changed: %s", exc)
        return 0


def head(quest_root: Path) -> tuple[str | None, bool | None]:
    """``(the commit code/ is at, whether code/ has changed since it)`` from code/'s own git history; ``(None, None)``
    when it has none (no git, or nothing recorded yet)."""
    import shutil
    import subprocess

    code_dir = Path(quest_root) / "code"
    if not (code_dir / ".git").exists() or shutil.which("git") is None:
        return None, None
    try:
        rev = _git(code_dir, "rev-parse", "HEAD")
        if rev.returncode != 0:
            return None, None
        status = _git(code_dir, "status", "--porcelain")
        changed = bool(status.stdout.strip()) if status.returncode == 0 else None
        return rev.stdout.strip() or None, changed
    except (OSError, subprocess.SubprocessError):
        return None, None


def _remove_env(env_dir: Path) -> None:
    """Remove the clean environment; on Windows a process still holding a file can leave it half deleted, which is
    said once in the log (the next check removes it again before it builds a new one)."""
    import logging
    import shutil

    shutil.rmtree(env_dir, ignore_errors=True)
    if env_dir.exists():
        logging.getLogger("frontier_insight.code_project").warning(
            "[code] could not remove %s completely (a file is still in use); it is removed again at the next check", env_dir)


def _drain(proc: Any) -> str:
    """What is left on a stopped run's pipes, never waiting for ever: a process that survived the stop and still
    holds a pipe open would otherwise hold FI with it."""
    import subprocess

    try:
        return proc.communicate(timeout=30)[1] or ""
    except subprocess.TimeoutExpired:
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
    except (OSError, ValueError):
        pass
    return ""


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g} min" if seconds >= 60 else f"{seconds:g} s"


def verify(quest_root: Path, *, run_timeout_s: float = 900.0, install_timeout_s: float = 900.0) -> dict[str, Any]:
    """Run ``code/`` the way a person would: a fresh environment, only ``requirements.txt`` installed, then
    ``python run.py``. Never raises; returns ``{"ok": bool, "status": "ran" | "failed" | "not_finished", "reason":
    plain words, "says": ...}`` (also saved to ``needs/CODE_PROJECT_CHECK.json``). ``not_finished`` is a run that
    installed and was still going when ``run_timeout_s`` ran out: FI stopped it (the whole process tree), and that
    is not a failure, so ``ok`` is False but only ``failed`` deserves a warning."""
    import shutil
    import subprocess
    import time

    quest_root = Path(quest_root)
    code_dir = quest_root / "code"
    started = time.monotonic()
    result: dict[str, Any] = {"ok": False, "reason": ""}
    env_dir = quest_root / ".fi" / "project_check"
    step = "creating the clean environment"
    try:
        if not (code_dir / RUN).is_file() or not (code_dir / REQUIREMENTS).is_file():
            result["reason"] = "code/ has no run.py or requirements.txt to try"
        else:
            _remove_env(env_dir)
            subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True, capture_output=True, timeout=300)
            py = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            wanted = [ln for ln in (code_dir / REQUIREMENTS).read_text(encoding="utf-8").split("\n") if ln.strip()]
            step = "installing requirements.txt"
            if wanted:
                # Packages' install hooks and run.py are code FI did not write: they run without FI's API keys.
                done = subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(code_dir / REQUIREMENTS)],
                                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      timeout=install_timeout_s, env=scrub_secrets())
                if done.returncode != 0:
                    result["reason"] = f"installing requirements.txt failed: {(done.stderr or '').strip()[-300:]}"
            if not result["reason"]:
                step = "running run.py"
                # A tree, not just the child: run.py starts a worker process per cell, and a timeout must stop them all.
                with _ProcessTree(
                    [str(py), str(code_dir / RUN)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    encoding="utf-8", errors="replace", cwd=code_dir, env=scrub_secrets(),
                ) as tree:
                    err = ""
                    try:
                        _out, err = tree.proc.communicate(timeout=run_timeout_s)
                    except subprocess.TimeoutExpired:
                        # Still running, unless run.py already exited (non-zero) and a worker it left behind holds the pipes.
                        if tree.proc.poll() is None:
                            result["status"] = "not_finished"  # set first: nothing below may turn it into a failure
                            result["reason"] = (
                                f"it installed and was still running after {_minutes(run_timeout_s)}, so FI stopped "
                                "it there (the full run takes longer; this is not a failure)"
                            )
                        tree.kill()
                        err = _drain(tree.proc)
                    if result.get("status") != "not_finished":
                        if tree.proc.returncode == 0:
                            result["ok"] = True
                        else:
                            result["reason"] = f"run.py stopped with an error: {(err or '').strip()[-300:]}"
    except subprocess.TimeoutExpired:
        result["reason"] = f"{step} took too long and was stopped"
    except (OSError, subprocess.SubprocessError) as exc:
        result["reason"] = f"the check could not start: {exc}"
    finally:
        _remove_env(env_dir)
    result["seconds"] = round(time.monotonic() - started)
    result["status"] = result.get("status") or ("ran" if result["ok"] else "failed")
    result["says"] = (
        "code/ was run again in a clean environment from requirements.txt and it ran."
        if result["ok"] else
        f"code/ was started again in a clean environment from requirements.txt; {result['reason']}. "
        "The results above are not affected."
        if result["status"] == "not_finished" else
        "code/ was run again in a clean environment from requirements.txt and it did NOT run "
        f"({result['reason']}). The results above are not affected; fix requirements.txt or run.py before sharing code/."
    )
    try:
        out = quest_root / "needs" / "CODE_PROJECT_CHECK.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return result
