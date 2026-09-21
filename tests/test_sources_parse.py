"""Every Python and JSON file in the repository has to parse.

A Bash heredoc has turned ``\\n`` inside a string into a real line break more than once: a page whose script did
not run (``web/static/skills.html``, since fixed) and ``scripts/import_scientist_skills.py``, which could not even
be compiled for days. A file nothing imports, or a script somebody runs by hand, is not exercised by any other
test, so nothing noticed. This looks at every tracked file.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SKIP = {".git", ".venv", "venv", "node_modules", "outputs", ".skill-sources", "__pycache__", "out"}


def _tracked(suffix: str) -> list[Path]:
    if shutil.which("git"):
        done = subprocess.run(
            ["git", "ls-files", f"*{suffix}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        )
        if done.returncode == 0 and done.stdout.strip():
            return [ROOT / line for line in done.stdout.splitlines() if (ROOT / line).is_file()]
    return [
        p for p in ROOT.rglob(f"*{suffix}")
        if not any(part in _SKIP or part.startswith(".pytest_tmp") for part in p.relative_to(ROOT).parts)
    ]


def test_every_python_file_parses() -> None:
    broken = []
    for path in _tracked(".py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            broken.append(f"{path.relative_to(ROOT).as_posix()}, line {e.lineno}: {e.msg}")
    assert not broken, "\n".join(broken)


def test_every_json_file_parses() -> None:
    broken = []
    for path in _tracked(".json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            broken.append(f"{path.relative_to(ROOT).as_posix()}: {e}")
    assert not broken, "\n".join(broken)


def test_the_check_looks_at_the_files_that_were_broken() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in _tracked(".py")}
    assert "scripts/import_scientist_skills.py" in names and "launch.py" in names
    assert any(p.name == "package.json" for p in _tracked(".json"))
