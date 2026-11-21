"""The sections a paper must have, read from its structure (headings), never from its wording.

A scientific paper (the persona block of the paper format is empty: ``generic``, ``article``, ``neurips`` ...) opens with
an ``## Abstract`` that has a body. :func:`missing_abstract` says whether a draft lacks it; :func:`with_abstract` puts a
written one in, under the title and above the keywords, the way the writer is told to open the paper.

A section ends at the next heading of its level or above (the title included), and a comment readers never see is not a
body: an ``## Abstract`` heading with nothing under it but comments (a writer's note to itself) is not the paper's
Abstract. :func:`without_bodiless_abstract` takes such a heading out, so the paper keeps the one Abstract that has a body.
"""
from __future__ import annotations

import re

_ABSTRACT_RE = re.compile(r"^##[ \t]+Abstract[ \t]*\n(?P<body>.*?)(?=^##?[ \t]|\Z)", re.MULTILINE | re.DOTALL | re.IGNORECASE)
_TITLE_RE = re.compile(r"^#[ \t]+\S.*$", re.MULTILINE)
_KEYWORDS_RE = re.compile(r"^[ \t]*[*_]{0,2}[ \t]*key[ \t]?words\b", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# A comment that is not a note: the keywords' hidden form, and the markers of FI's own blocks.
_KEPT_COMMENT_RE = re.compile(r"<!--[ \t]*(?:key[ \t]?words[ \t]*:|/?fi:)", re.IGNORECASE)


def _body(section_body: str) -> list[str]:
    """The lines of an Abstract section a reader sees, without the keywords line."""
    return [ln for ln in _COMMENT_RE.sub("", section_body).splitlines() if ln.strip() and not _KEYWORDS_RE.match(ln)]


def missing_abstract(markdown: str) -> bool:
    """True when the paper has no ``## Abstract`` section with a body (keywords alone, or comments, are not a body)."""
    return not any(_body(m.group("body")) for m in _ABSTRACT_RE.finditer(markdown))


def abstract_of(markdown: str) -> str:
    """The body of the paper's ``## Abstract`` section, found by its heading (the pattern :func:`missing_abstract` uses), without
    the keywords line; ``""`` when there is none. The paper's own reviewed text, never a model's summary of the results."""
    for m in _ABSTRACT_RE.finditer(markdown):
        body = _body(m.group("body"))
        if body:
            return "\n".join(ln.rstrip() for ln in body).strip()
    return ""


def without_bodiless_abstract(markdown: str) -> str:
    """``markdown`` without each ``## Abstract`` heading that has no body under it (:func:`missing_abstract`), and without
    the comments under it; a keywords line or comment, or a marker of FI's, stays where it was. A paper with no such
    heading is returned as it is."""
    spans = [m for m in _ABSTRACT_RE.finditer(markdown) if not _body(m.group("body"))]
    if not spans:
        return markdown
    text, at = "", 0
    for m in spans:
        kept = _COMMENT_RE.sub(lambda c: c.group(0) if _KEPT_COMMENT_RE.match(c.group(0)) else "", m.group("body"))
        kept_lines = [ln for ln in kept.splitlines() if ln.strip()]
        text = _join(text + markdown[at:m.start()], "\n".join(kept_lines))
        at = m.end()
    return _join(text, markdown[at:])


def _join(head: str, tail: str) -> str:
    """``head`` and ``tail`` with one blank line between them (none at the top of the paper)."""
    head, tail = head.rstrip("\n"), tail.lstrip("\n")
    if not head or not tail:
        return head + tail if tail else (head + "\n" if head else "")
    return head + "\n\n" + tail


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
