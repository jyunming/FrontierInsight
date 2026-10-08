"""Bring runs that already exist into the suite as a round, so the history starts with real numbers.

A seed file (YAML, kept with the suite, not in the repository) lists the rounds::

    rounds:
      v5:
        fi_sha: abc12345                 # optional; "unknown" when not given (the quests do not record it)
        fi_note: where that sha came from
        control: false                   # a control round is shown in the trend but is not a suite run
        topics:
          a: {quest_root: C:/runs/v5_a/out, grade: C:/grades/v5_a.md}   # `grade` optional: a grader's report file
          b: {quest_root: C:/runs/v5_b/out}

Nothing under a ``quest_root`` is changed. Each topic's quest is read as it stands.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import grade as _grade
from . import runner as _runner
from .suite import Suite, SuiteError
from core import quality_report as qr


def _created(quest_roots: list[Path]) -> str:
    """When the round began: the epoch every quest folder name starts with (the earliest), else now."""
    stamps = []
    for root in quest_roots:
        for q in qr.quest_dirs([root]):
            m = re.match(r"(\d{9,11})-", q.name)
            if m:
                stamps.append(int(m.group(1)))
    if stamps:
        return datetime.fromtimestamp(min(stamps), tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    return _runner.now()


def adopt_round(suite: Suite, name: str, spec: dict[str, Any]) -> Path:
    rdir = _runner.round_dir_of(suite, name)
    topics_spec = spec.get("topics")
    if not isinstance(topics_spec, dict) or not topics_spec:
        raise SuiteError(f"round `{name}` in the seed file lists no topics")
    roots = {str(t): Path(str(s["quest_root"])) for t, s in topics_spec.items()}
    for t, root in roots.items():
        if not qr.quest_dirs([root]):
            raise SuiteError(f"round `{name}`, topic `{t}`: no quest found under {root}")
    sha = str(spec.get("fi_sha") or "unknown")
    manifest = {
        "round": name, "group": "adopted", "control": bool(spec.get("control")), "adopted": True, "suite": str(suite.path),
        "fi": None, "fi_git": {"sha": sha, "short": sha[:8], "dirty": None}, "fi_note": spec.get("fi_note"),
        "created_at": _created(list(roots.values())), "max_parallel": 0,
        "topics": {t: {"config": None, "dir": str(root.parent), "quest_root": str(root), "state": "adopted", "pid": None,
                       "started_at": None, "ended_at": None, "exit_code": None} for t, root in roots.items()},
    }
    rdir.mkdir(parents=True, exist_ok=True)
    _runner.save_manifest(rdir, manifest)
    for t, s in topics_spec.items():
        if s.get("grade"):
            _grade.import_grade(rdir, str(t), Path(str(s["grade"])), stop_correct=s.get("stop_correct"),
                                cause_class=s.get("class"), stage=s.get("stage"), note=s.get("class_note"))
        elif s.get("class") or s.get("stop_correct") is not None:
            _write_verdict(rdir, str(t), s)
    return rdir


def _write_verdict(rdir: Path, topic: str, s: dict[str, Any]) -> None:
    import json
    grades = _grade.load_grades(rdir)
    entry = grades.get(topic, {})
    for src, dst in (("stop_correct", "stop_correct"), ("class", "class"), ("stage", "stage"), ("class_note", "class_note")):
        if s.get(src) is not None:
            entry[dst] = s[src]
    grades[topic] = entry
    (rdir / _grade.GRADES).write_text(json.dumps(grades, indent=1) + "\n", encoding="utf-8")


def adopt_file(suite: Suite, seed_path: Path) -> list[Path]:
    try:
        seed = yaml.safe_load(Path(seed_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteError(f"cannot read {seed_path}: {exc}") from exc
    rounds = (seed or {}).get("rounds")
    if not isinstance(rounds, dict) or not rounds:
        raise SuiteError(f"{seed_path} has no `rounds`")
    return [adopt_round(suite, str(n), s) for n, s in rounds.items()]
