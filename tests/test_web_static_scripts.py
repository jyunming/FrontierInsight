"""Every inline script in the dashboard's pages must at least parse.

The Skills page shipped, and stayed, with a syntax error in ``approveAll()``: raw newlines
inside string literals, which a Bash heredoc had turned out of ``\\n``. A script that does not
parse does not run at all, so ``load()`` was never called and the page stayed empty. Nothing
noticed, because the tests fetch the HTML and call the API and never execute the JavaScript.

Skips when Node is not installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
_SCRIPT = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S)


def _inline_scripts() -> list[tuple[str, int, str, bool]]:
    out = []
    for page in sorted(STATIC.glob("*.html")):
        for index, m in enumerate(_SCRIPT.finditer(page.read_text(encoding="utf-8"))):
            attrs, body = m.group("attrs"), m.group("body")
            if "src=" in attrs or not body.strip():
                continue
            if "type=" in attrs and "module" not in attrs and "javascript" not in attrs:
                continue  # a JSON island, an importmap...
            out.append((page.name, index, body, "module" in attrs))
    return out


@pytest.mark.parametrize(
    "page,index,body,module",
    [pytest.param(*case, id=f"{case[0]}-{case[1]}") for case in _inline_scripts()],
)
def test_the_inline_script_parses(tmp_path: Path, page: str, index: int, body: str, module: bool) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    source = tmp_path / ("inline.mjs" if module else "inline.js")
    source.write_text(body, encoding="utf-8")
    done = subprocess.run([node, "--check", str(source)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, f"{page}, script {index}: {done.stderr.strip()[:400]}"


def test_there_are_scripts_to_check() -> None:
    names = {page for page, _i, _b, _m in _inline_scripts()}
    assert {"skills.html", "quest.html", "index.html"} <= names
