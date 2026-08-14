"""Version parity across the Python package and the VSCode extension (PR-12).

The Python package (pyproject.toml) and the VSCode extension
(vscode-frontier-insight/package.json) ship as one product and must carry the
same version string. Without a guard they drift silently — a bumped Python
release with a stale extension version, or vice versa — and users can't tell
which build they have. This test fails the moment the two disagree, so a
release bump has to touch both.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
PACKAGE_JSON = REPO / "vscode-frontier-insight" / "package.json"


def _pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    # First `version = "X"` under [project] / [tool.poetry].
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    assert m, "no version field found in pyproject.toml"
    return m.group(1)


def _package_json_version() -> str:
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    return data["version"]


def test_python_and_extension_versions_match() -> None:
    py = _pyproject_version()
    ext = _package_json_version()
    assert py == ext, (
        f"Version drift: pyproject.toml is {py!r} but "
        f"vscode-frontier-insight/package.json is {ext!r}. "
        f"Bump both together on a release."
    )
