"""The keywords a writer gives a paper, read back out of ``paper.md``.

A scientific paper shows them on a line under its abstract
(``**Keywords:** symplectic integrator, energy drift``). A report, policy
brief, essay or whitepaper keeps them in a comment readers never see
(``<!-- Keywords: ... -->``). Both reach the knowledge layer's index card;
only the visible line is printed.
"""

from __future__ import annotations

import re

# "**Keywords:** a, b", "**Keywords**: a, b", "*Keywords:* a, b" or
# "Keywords: a, b", on a line of its own.
_LINE_RE = re.compile(
    r"^[ \t]*[*_]{0,2}[ \t]*key[ \t]?words[ \t]*[*_]{0,2}[ \t]*:[ \t]*[*_]{0,2}[ \t]*"
    r"(?P<words>[^\s*_].*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_COMMENT_RE = re.compile(
    r"<!--[ \t]*key[ \t]?words[ \t]*:(?P<words>.*?)-->[ \t]*",
    re.IGNORECASE | re.DOTALL,
)
# The paper's front (its title and abstract) ends at the first other section.
_FRONT_END_RE = re.compile(r"^##[ \t]+(?!abstract[ \t]*$)", re.IGNORECASE | re.MULTILINE)


def split_keywords(text: str) -> list[str]:
    """The keywords in ``text``: split on commas and semicolons, with
    emphasis, quotes and a closing full stop taken off and repeats dropped."""
    words: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[;,]", text):
        word = part.strip().rstrip(".").strip().strip("*_`\"'").strip()
        if word and word.lower() not in seen:
            seen.add(word.lower())
            words.append(word)
    return words


def _front_end(markdown: str) -> int:
    m = _FRONT_END_RE.search(markdown)
    return m.start() if m else len(markdown)


def extract_keywords(markdown: str) -> tuple[list[str], str]:
    """The paper's keywords, and ``markdown`` without the line or comment
    that gave them. Only the title and abstract are read, so a line starting
    "Keywords:" further down stays where it is."""
    end = _front_end(markdown)
    front, rest = markdown[:end], markdown[end:]
    for regex in (_LINE_RE, _COMMENT_RE):
        m = regex.search(front)
        if not m:
            continue
        words = split_keywords(m.group("words"))
        if not words:
            continue
        cut = m.end() + (1 if front[m.end():m.end() + 1] == "\n" else 0)
        front = re.sub(r"\n{3,}", "\n\n", front[:m.start()] + front[cut:])
        return words, front + rest
    return [], markdown


def paper_keywords(markdown: str) -> list[str]:
    """The keywords ``markdown`` gives, in either form."""
    return extract_keywords(markdown)[0]


def keywords_block(markdown: str) -> tuple[list[str], str]:
    """For the HTML render: the keywords a visible line gives, with that line
    turned into a ``keywords`` block the themes style. A comment is left for
    the browser to hide and gives no keywords to show."""
    end = _front_end(markdown)
    m = _LINE_RE.search(markdown[:end])
    words = split_keywords(m.group("words")) if m else []
    if not m or not words:
        return [], markdown
    block = "\n\n::: keywords\n**Keywords:** " + ", ".join(words) + "\n:::\n\n"
    front = markdown[:m.start()].rstrip("\n") + block + markdown[m.end():end].lstrip("\n")
    return words, front + markdown[end:]
