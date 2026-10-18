"""Everything committed to the repository is public, so it is written in English.

Chinese (or Japanese / Korean) text is allowed in a committed file only where it is data, not prose:
- inside a Python string (a regular expression or a test input for text a user typed in Chinese, a docstring example of
  such input); inside a quoted string in any other code file; inside a backtick code span in prose (Markdown, text);
- in test fixtures (records of real runs), in the historical record ``docs/audits/``, and in a translation ``*.zh-TW.md``
  whose English original sits next to it (the English one is the authority).
A comment or a sentence of prose in Chinese fails, with the file and line named.
"""
from __future__ import annotations

import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile("[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
ALLOWED_DIRS = ("docs/audits/", "tests/fixtures/")
# the data a line talks about: a quoted string or a code span in code; in prose only a code span, since an apostrophe
# in a sentence is not a quote
QUOTED = re.compile(r"`[^`\n]*`|\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*'")
CODE_SPAN = re.compile(r"`[^`\n]*`")
PROSE = (".md", ".txt", ".rst")


def _tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.decode("utf-8").split("\0") if p]


def _text(path: Path) -> str | None:
    """The file's text, or None for a binary file (a NUL byte, or not UTF-8)."""
    raw = path.read_bytes()
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _allowed_file(rel: str, root: Path) -> bool:
    if rel.startswith(ALLOWED_DIRS):
        return True
    if rel.endswith(".zh-TW.md"):
        return (root / (rel[: -len(".zh-TW.md")] + ".md")).exists()
    return False


def _python_hits(text: str) -> list[int]:
    """Lines where CJK text sits outside a string (a comment, or a name)."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return _other_hits(text, prose=False)
    strings = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
    return sorted({tok.start[0] for tok in tokens if tok.type not in strings and CJK.search(tok.string)})


def _other_hits(text: str, prose: bool) -> list[int]:
    """Lines where CJK text sits outside the data on the line (a quoted string or code span; in prose a code span)."""
    data = CODE_SPAN if prose else QUOTED
    return [n for n, line in enumerate(text.splitlines(), start=1) if CJK.search(data.sub("", line))]


def offences(paths: list[str], root: Path = ROOT) -> list[str]:
    found: list[str] = []
    for rel in paths:
        if _allowed_file(rel, root):
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = _text(path)
        if text is None or not CJK.search(text):
            continue
        lines = _python_hits(text) if rel.endswith(".py") else _other_hits(text, prose=rel.endswith(PROSE))
        found += [f"{rel}:{n}" for n in lines]
    return found


def test_committed_text_is_english():
    hits = offences(_tracked())
    assert not hits, ("Chinese prose in committed files (write it in English; Chinese is allowed only as quoted data, "
                      "in tests/fixtures/, docs/audits/ or a *.zh-TW.md translation next to its English original):\n"
                      + "\n".join(hits[:50]))


def test_the_check_catches_prose_and_allows_data(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "a.py").write_text('x = r"最佳"  # 註解\n"""Example input ``遺傳``."""\n',
                                            encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "e.ts").write_text('const a = "中文";\n// 註解\n', encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.md").write_text("A word such as `最佳` is read.\nIt's 最佳 here, isn't it.\n",
                                            encoding="utf-8")
    (tmp_path / "docs" / "c.zh-TW.md").write_text("翻譯\n", encoding="utf-8")
    (tmp_path / "docs" / "d.md").write_text("English\n", encoding="utf-8")
    (tmp_path / "docs" / "d.zh-TW.md").write_text("翻譯\n", encoding="utf-8")
    hits = offences(["core/a.py", "src/e.ts", "docs/b.md", "docs/c.zh-TW.md", "docs/d.zh-TW.md"], root=tmp_path)
    assert "core/a.py:1" in hits and "core/a.py:2" not in hits     # the comment, not the string or the docstring
    assert "src/e.ts:2" in hits and "src/e.ts:1" not in hits       # a comment, not the quoted string
    assert "docs/b.md:2" in hits and "docs/b.md:1" not in hits     # prose: an apostrophe is not a quote
    assert "docs/c.zh-TW.md:1" in hits                             # a translation needs its English original
    assert not any(h.startswith("docs/d.zh-TW.md") for h in hits)
