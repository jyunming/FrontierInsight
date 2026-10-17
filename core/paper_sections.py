"""The sections a paper must have, read from its structure (headings), never from its wording.

A scientific paper (the persona block of the paper format is empty: ``generic``, ``article``, ``neurips`` ...) opens with
an ``## Abstract`` that has a body. :func:`missing_abstract` says whether a draft lacks it; :func:`with_abstract` puts a
written one in, under the title and above the keywords, the way the writer is told to open the paper.
"""
from __future__ import annotations

import re

_ABSTRACT_RE = re.compile(r"^##[ \t]+Abstract[ \t]*\n(?P<body>.*?)(?=^##[ \t]|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"^#[ \t]+\S.*$", re.MULTILINE)
_KEYWORDS_RE = re.compile(r"^[ \t]*[*_]{0,2}[ \t]*key[ \t]?words\b", re.IGNORECASE)


def missing_abstract(markdown: str) -> bool:
    """True when the paper has no ``## Abstract`` section with a body (keywords alone are not a body)."""
    for m in _ABSTRACT_RE.finditer(markdown):
        body = [ln for ln in m.group("body").splitlines() if ln.strip() and not _KEYWORDS_RE.match(ln)]
        if body:
            return False
    return True


def with_abstract(markdown: str, abstract: str) -> str:
    """``markdown`` with ``## Abstract`` and ``abstract`` right under the title (at the top when there is no title), and
    the keywords line a scientific paper shows under it, turned from the hidden comment form when that is how they were given."""
    from generation._keywords import extract_keywords

    abstract = abstract.strip()
    if not abstract:
        return markdown
    words, rest = extract_keywords(markdown)
    title = _TITLE_RE.search(rest)
    at = title.end() if title else 0
    block = f"\n\n## Abstract\n\n{abstract}\n" + (f"\n**Keywords:** {', '.join(words)}\n" if words else "")
    out = rest[:at] + block + "\n" + rest[at:].lstrip("\n")
    return out if title else out.lstrip("\n")
