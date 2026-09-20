"""A figure caption as a renderer that numbers figures should print it.

The writer captions a figure ``![**Figure 2.** What it shows](figures/x.png)``
(agents/write.md), since paper.md read as Markdown has no other numbering.
LaTeX and the HTML themes number figures themselves, so paper.pdf printed
"Figure 2: **Figure 2.** What it shows". The writer's number comes off at
render time; paper.md keeps it.
"""

from __future__ import annotations

import re

_DASHES = ".:–—-"
# "**Figure 2.**", "*Fig. 2:*", "__Figure 2__ -": the number inside emphasis.
_EMPHASISED_RE = re.compile(
    rf"^\s*(?P<mark>\*\*|\*|__|_)\s*(?:figure|fig\.?)\s*\d+[a-z]?\s*[{_DASHES}]?\s*"
    rf"(?P=mark)\s*[{_DASHES}]?\s*",
    re.IGNORECASE,
)
# "Figure 2." or "Fig. 2:" without emphasis. The punctuation is required, so
# "Figure 2 shows the drift" keeps its words.
_PLAIN_RE = re.compile(rf"^\s*(?:figure|fig\.?)\s*\d+[a-z]?\s*[{_DASHES}]\s*", re.IGNORECASE)

# An image alone on its line, which pandoc makes a numbered figure when the
# line is its own paragraph. The alt text may hold [3]-style citations.
_FIGURE_LINE_RE = re.compile(
    r"^(?P<lead>\s*!\[)(?P<alt>(?:[^\[\]]|\[[^\[\]]*\])*)(?P<rest>\]\([^)]*\)(?:\{[^}]*\})?\s*)$"
)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def without_number(caption: str) -> str:
    """``caption`` without a leading "Figure N." the writer typed. A caption
    that is nothing but the number is returned as it was."""
    for pattern in (_EMPHASISED_RE, _PLAIN_RE):
        m = pattern.match(caption)
        if m:
            rest = caption[m.end():].strip()
            return rest or caption
    return caption


def numbers_off_figure_captions(markdown: str) -> str:
    """``markdown`` with the writer's "Figure N." taken off each figure line's
    caption. Code blocks are left alone."""
    out = []
    fence = None
    for line in markdown.split("\n"):
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        elif fence is None:
            fig = _FIGURE_LINE_RE.match(line)
            if fig:
                line = fig.group("lead") + without_number(fig.group("alt")) + fig.group("rest")
        out.append(line)
    return "\n".join(out)


def blank_lines_around_figures(markdown: str) -> str:
    """``markdown`` with a blank line before and after each figure line that lacks
    one. Pandoc makes an image a numbered figure, with its caption, only when the
    image is a paragraph of its own: two figure lines one under the other are one
    paragraph of two inline images, which prints both without their captions (a
    stored quest with three in a row lost all three). Code blocks are left alone."""
    lines = markdown.split("\n")
    out: list[str] = []
    fence = None
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        is_figure = fence is None and not m and _FIGURE_LINE_RE.match(line) is not None
        if is_figure and out and out[-1].strip():
            out.append("")
        out.append(line)
        if is_figure and i + 1 < len(lines) and lines[i + 1].strip():
            out.append("")
    return "\n".join(out)
