"""A repair's own file marker: only prose right before an opening fence names the file (core/code_review.py::parse_files)."""
from __future__ import annotations

from core import code_review


# ---- the repair's own file marker: only prose right before an opening fence ----------------------------------------------------
def test_a_file_marker_in_prose_right_before_the_fence_names_the_file() -> None:
    for reply in ("# file: derive.py\n```python\nx = 1\n```\n", "Here it is.\n\n# file: derive.py\n\n```python\nx = 1\n```\n",
                  "```python\n# file: derive.py\nx = 1\n```\n"):
        assert code_review.parse_files(reply, ["derive.py"]) == {"derive.py": "x = 1\n"}
    assert code_review.parse_files("```python\nx = 1\n```\n", ["derive.py"]) == {}


def test_a_marker_like_line_inside_a_fence_is_code_and_never_moved() -> None:
    reply = "```python\n# file: a.py\nx = 1\n# file: b.py\n```\n"
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {"a.py": "x = 1\n# file: b.py\n"}
    reply = '```python\n# file: a.py\ns = """\n# file: b.py\n```x\n"""\n```\n'
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {}  # a fence inside a string ends the block, as before: never rewritten
    reply = '```python\n# file: a.py\ns = """\n# file: b.py\n"""\n```\n'
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {"a.py": 's = """\n# file: b.py\n"""\n'}
