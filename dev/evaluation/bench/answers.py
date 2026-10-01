"""The benchmark's fixed answers: one ``answer.json`` per task, its loader and its checks.

An answer file (``dev/quest-topics/answers/<topic>.answer.json``)::

    {
      "schema": "fi.bench.answer/v1",
      "task": "Q4",
      "title": "...",
      "config": "dev/quest-topics/sir_final_size.yaml",     # the quest, relative to the repository
      "kind": "simulation" | "literature",
      "ask": "BENCHMARK. ...",            # added to the topic: names the protocol's metric ids and the answer's setting
      "answers": [                         # simulation: what FI's own trial record must hold
        {"metric": "final_fraction", "cell": {"N": 1000, "R0": 2.0}, "statistic": "mean",
         "expected": 0.398, "tolerance": 0.02, "tolerance_kind": "absolute", "source": "..."}
      ],
      "structural": {...},                 # literature: what the paper must and must not cite (see score.py)
      "errors": ["R1", "N3", "L1"],        # the planted errors this task takes (dev/evaluation/bench/catalogue.py)
    }

``ask`` is what makes an answer scorable by a program: the quest's text names the protocol metric ids and the grid
setting the answer is read at, so the scorer finds the number in FI's record without reading the paper.

Held-back tasks. Four tasks are kept out of the development tree so FI is not tuned against their answers: their answer
files live outside the repository (``FI_BENCH_HELD_OUT``, default ``~/.fi/bench_held_out/``), and the repository keeps
only each file's SHA-256 in ``held_out.json``. A held-back answer whose file no longer matches its hash is refused:
an answer changed after it was committed to cannot be told from one fitted to the results. ``fi tools bench hold-out
<file>`` moves a file out and records its hash; rotating a task back in is the reverse, done by hand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import catalogue

REPO = Path(__file__).resolve().parents[3]
ANSWERS_DIR = REPO / "dev" / "quest-topics" / "answers"
HELD_OUT_INDEX = "held_out.json"
HELD_OUT_ENV = "FI_BENCH_HELD_OUT"
SCHEMA = "fi.bench.answer/v1"
KINDS = ("simulation", "literature")
STATISTICS = ("mean", "proportion", "value")
_TASK_RE = re.compile(r"^Q\d{1,2}$")
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class AnswerError(ValueError):
    """An answer file that cannot be used, with every reason."""


def held_out_dir() -> Path:
    return Path(os.environ.get(HELD_OUT_ENV) or Path.home() / ".fi" / "bench_held_out").expanduser()


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def validate(data: Any, *, repo: Path = REPO) -> list[str]:
    """Every reason ``data`` is not a usable answer file; ``[]`` when it is."""
    if not isinstance(data, dict):
        return ["an answer file is a JSON object"]
    problems: list[str] = []
    if data.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    task = str(data.get("task") or "")
    if not _TASK_RE.match(task):
        problems.append("task must be Q1..Q99")
    kind = data.get("kind")
    if kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
    config = str(data.get("config") or "")
    if not config or not (repo / config).is_file():
        problems.append(f"config {config!r} is not a file in the repository")
    ask = str(data.get("ask") or "")
    if not ask.strip():
        problems.append("ask (the sentence added to the topic that names the metric ids and the setting) is empty")
    errors = data.get("errors")
    if not isinstance(errors, list) or not errors:
        problems.append("errors must list the planted errors this task takes")
    else:
        problems += [f"error {e!r} is not in the catalogue" for e in errors if str(e).upper() not in catalogue.CATALOGUE]
    answers = data.get("answers") or []
    if kind == "simulation" and not answers:
        problems.append("a simulation task needs at least one answer")
    for n, a in enumerate(answers if isinstance(answers, list) else [], 1):
        where = f"answers[{n}]"
        if not isinstance(a, dict):
            problems.append(f"{where} is not an object")
            continue
        metric = str(a.get("metric") or "")
        if not _ID_RE.match(metric):
            problems.append(f"{where}: metric must be the protocol's metric id")
        elif not re.search(rf"\b{re.escape(metric)}\b", ask):
            problems.append(f"{where}: the ask sentence does not name the metric {metric!r}")
        cell = a.get("cell")
        if not isinstance(cell, dict):
            problems.append(f"{where}: cell must be an object (axis -> value)")
        else:
            for axis, value in cell.items():
                if not re.search(rf"\b{re.escape(str(axis))}\b", ask) or not re.search(
                        rf"(?<![\w.]){re.escape(_fmt(value))}(?![\w])", ask):
                    problems.append(f"{where}: the ask sentence does not name the setting {axis} = {_fmt(value)}")
        if a.get("statistic") not in STATISTICS:
            problems.append(f"{where}: statistic must be one of {', '.join(STATISTICS)}")
        if not _number(a.get("expected")):
            problems.append(f"{where}: expected must be a number")
        if not _number(a.get("tolerance")) or a.get("tolerance", 0) <= 0:
            problems.append(f"{where}: tolerance must be a positive number")
        if a.get("tolerance_kind", "absolute") not in ("absolute", "relative"):
            problems.append(f"{where}: tolerance_kind must be absolute or relative")
        if not str(a.get("source") or "").strip():
            problems.append(f"{where}: source (where the expected value comes from) is empty")
    if kind == "literature":
        s = data.get("structural")
        if not isinstance(s, dict):
            problems.append("a literature task needs structural")
        else:
            for key in ("must_not_cite", "must_find"):
                rows = s.get(key)
                if not isinstance(rows, list) or not all(isinstance(r, dict) and r.get("doi") for r in rows):
                    problems.append(f"structural.{key} must list objects with a doi")
            share = s.get("min_supported_share")
            if not _number(share) or not 0 <= share <= 1:
                problems.append("structural.min_supported_share must be a number from 0 to 1")
            recall = s.get("min_recall")
            if not _number(recall) or not 0 <= recall <= 1:
                problems.append("structural.min_recall must be a number from 0 to 1")
    return problems


def load(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AnswerError(f"{path}: cannot be read ({e})") from None
    problems = validate(data)
    if problems:
        raise AnswerError(f"{path}: " + "; ".join(problems))
    data["_path"] = str(path)
    return data


def _index(answers_dir: Path) -> dict[str, Any]:
    path = answers_dir / HELD_OUT_INDEX
    if not path.is_file():
        return {"tasks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def files(answers_dir: Path | None = None) -> list[Path]:
    return sorted(Path(answers_dir or ANSWERS_DIR).glob("*.answer.json"))


def load_task(task: str, *, answers_dir: Path | None = None, held_dir: Path | None = None) -> dict[str, Any]:
    """The answer file of task ``task`` (``Q4``): from the repository, or a held-back one from outside it, checked
    against the hash the repository keeps."""
    task = task.upper()
    answers_dir = Path(answers_dir or ANSWERS_DIR)
    for path in files(answers_dir):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("task") == task:
                return load(path)
        except (OSError, ValueError):
            continue
    held = _index(answers_dir).get("tasks", {}).get(task)
    if not held:
        raise AnswerError(f"no answer file for {task}")
    path = (held_dir or held_out_dir()) / str(held["file"])
    if not path.is_file():
        raise AnswerError(f"{task} is held back and its answer file is not at {path} (set {HELD_OUT_ENV})")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != held.get("sha256"):
        raise AnswerError(f"{task}'s held-back answer file changed after its hash was recorded ({path})")
    return load(path)


def hold_out(path: Path, *, answers_dir: Path | None = None, held_dir: Path | None = None) -> Path:
    """Move an answer file out of the repository and record its hash: the task is then held back."""
    answers_dir = Path(answers_dir or ANSWERS_DIR)
    data = load(path)
    dest_dir = held_dir or held_out_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(path).name
    shutil.move(str(path), dest)
    index = _index(answers_dir)
    index.setdefault("tasks", {})[data["task"]] = {"file": dest.name,
                                                   "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (Path(answers_dir) / HELD_OUT_INDEX).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return dest


def check_all(answers_dir: Path | None = None) -> list[str]:
    """Every problem with the answer files in the repository and the held-back index (``fi tools bench check``)."""
    answers_dir = Path(answers_dir or ANSWERS_DIR)
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for path in files(answers_dir):
        try:
            data = load(path)
        except AnswerError as e:
            problems.append(str(e))
            continue
        if data["task"] in seen:
            problems.append(f"{path.name}: task {data['task']} is also answered by {seen[data['task']].name}")
        seen[data["task"]] = path
    for task in _index(answers_dir).get("tasks", {}):
        if task in seen:
            problems.append(f"{task} is held back, but its answer file is in the repository ({seen[task].name})")
    return problems


# --- reading an answer against a run ---------------------------------------------------------------------------------

def same_value(a: Any, b: Any) -> bool:
    if _number(a) and _number(b):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return str(a).strip().lower() == str(b).strip().lower()


def cell_matches(actual: dict[str, Any], wanted: dict[str, Any]) -> bool:
    """Whether a cell of the run is the answer's setting: every axis the answer names has its value (the run may have
    more axes; the scorer then requires exactly one matching cell)."""
    lowered = {str(k).lower(): v for k, v in (actual or {}).items()}
    return all(str(k).lower() in lowered and same_value(lowered[str(k).lower()], v) for k, v in wanted.items())


def within(got: float, answer: dict[str, Any]) -> bool:
    tol = float(answer["tolerance"])
    if answer.get("tolerance_kind", "absolute") == "relative":
        tol *= abs(float(answer["expected"]))
    return abs(float(got) - float(answer["expected"])) <= tol
