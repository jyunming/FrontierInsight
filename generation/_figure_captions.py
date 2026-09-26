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


_CAPTION_NUMBER_RE = re.compile(r"^\s*(?:\*\*|\*|__|_)?\s*(?:figure|fig\.?)\s*(\d+)", re.IGNORECASE)
# "Figure 3", "Fig. 3", "Figures 2 and 3", "Figs. 2-4": a reference in the text, with every number it names.
_REFERENCE_RE = re.compile(
    # Not a path or a file name: "figures/figure3.png" is a link, and renumbering it broke the image.
    r"(?<![/\\\w.-])(?P<word>Figures?|Figs?\.?)(?P<gap>\s*)"
    r"(?P<nums>\d+[a-z]?(?:\s*(?:,|and|&|–|—|-|to)\s*\d+[a-z]?)*)(?![\w.]*\.[A-Za-z])",
    re.IGNORECASE,
)


def _figure_order(lines: list[str]) -> dict[int, int] | None:
    """The writer's figure numbers against the place each figure has (1, 2, ...), when every figure line's caption
    names its number and they are not already in order; ``None`` otherwise (nothing to renumber)."""
    numbers: list[int] = []
    fence = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
            continue
        fig = _FIGURE_LINE_RE.match(line) if fence is None else None
        if fig:
            n = _CAPTION_NUMBER_RE.match(fig.group("alt"))
            if not n:
                return None
            numbers.append(int(n.group(1)))
    if len(set(numbers)) != len(numbers) or numbers == list(range(1, len(numbers) + 1)):
        return None
    return {n: i + 1 for i, n in enumerate(numbers)}


def _renumbered(text: str, order: dict[int, int]) -> str:
    def one(m: re.Match[str]) -> str:
        span = re.fullmatch(r"(\d+)(\s*[–—-]\s*|\s+to\s+)(\d+)", m.group("nums"))
        if span and int(span.group(1)) < int(span.group(3)):
            # A range names every figure in it: renumbered, those figures are a range again only when their new
            # numbers are consecutive ("Figs. 2-3" with the two swapped is still "2-3"); otherwise they are listed.
            new = sorted(order.get(k, k) for k in range(int(span.group(1)), int(span.group(3)) + 1))
            if new == list(range(new[0], new[-1] + 1)):
                nums = f"{new[0]}{span.group(2)}{new[-1]}"
            else:
                nums = ", ".join(map(str, new[:-1])) + f" and {new[-1]}"
        else:
            nums = re.sub(r"\d+", lambda d: str(order.get(int(d.group(0)), int(d.group(0)))), m.group("nums"))
        return m.group("word") + m.group("gap") + nums

    return _REFERENCE_RE.sub(one, text)


def numbers_off_figure_captions(markdown: str) -> str:
    """``markdown`` with the writer's "Figure N." taken off each figure line's
    caption. Code blocks are left alone.

    The renderer numbers figures by where they stand. When the writer placed them out of number order (Figure 1, then
    3, then 2), the text's "Figure 3" would sit over the plot printed as Figure 2; so every "Figure N" reference is
    renumbered to the place its figure has. paper.md keeps the writer's numbers."""
    lines = markdown.split("\n")
    order = _figure_order(lines)
    out = []
    fence = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        elif fence is None:
            fig = _FIGURE_LINE_RE.match(line)
            if fig:
                caption = without_number(fig.group("alt"))
                line = fig.group("lead") + (_renumbered(caption, order) if order else caption) + fig.group("rest")
            elif order:
                line = _renumbered(line, order)
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
