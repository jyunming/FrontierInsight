"""A minimal commission for tests that start or resume a study (every study starts from one; core/commission.py)."""
from __future__ import annotations

import json
from pathlib import Path

QUESTION = "How does a made-up quantity change with one setting?"


def write(folder: Path, question: str = QUESTION) -> Path:
    """``folder/commission.yaml``: every key, no known value (made-up tests only)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "commission.yaml"
    path.write_text(
        f"question: {json.dumps(question)}\nscope: one made-up setting\nlimits: {{hours: 1, model_calls: 60}}\n"
        "known_values: []\nno_known_values: true\nsuccess: [{condition: c, verify: v}]\n"
        "failure: [{condition: c, verify: v}]\ndoubts: []\n", encoding="utf-8")
    return path


def line(folder: Path, question: str = QUESTION) -> str:
    """The config line naming a fresh commission in ``folder`` (an absolute path, so any folder may read it)."""
    return f"commission: {json.dumps(write(folder, question).as_posix())}\n"
