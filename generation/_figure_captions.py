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


# A figure image: ``![caption](figures/<file>)`` with an optional {attributes} block. The caption may hold [3]-style citations.
_FIGURE_IMAGE_RE = re.compile(r"!\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(figures/[^)\s]+\)(?:\{[^}]*\})?")
# Text that is only a bold run-in label ("**Spread.**"): a heading for the paragraph, not a sentence.
_RUN_IN_LABEL_RE = re.compile(r"^\*\*[^*\n]{1,60}\*\*$")
# A block that is not plain prose: a list item, table row, quote, heading, HTML or comment, or another image.
_NOT_PROSE_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\||>|#|<|!\[)")


def figures_in_own_paragraph(markdown: str) -> tuple[str, int]:
    """``(markdown, n)``: each figure image that shares its paragraph with text (or with another image) moved into a
    paragraph of its own, right after the paragraph it was in; ``n`` is how many were moved.

    Pandoc captions, numbers and floats an image only when it is alone in its paragraph; elsewhere it is an inline
    image at its natural width, with no caption. After the paragraph, not before it, because the text a figure sits in
    is the text that discusses it, and splitting nothing but the image out keeps every sentence where the writer put it.
    When the rest of the paragraph is only a bold run-in label ("**Spread.**"), the label would be left alone, so it
    goes to the front of the next prose paragraph (the figure stands between), which is the text that label heads.
    Code blocks, list items, table rows and quotes are left as they are, and so are images that are not in
    ``figures/``. Applying it twice changes nothing."""
    lines = markdown.split("\n")
    # Blocks: ("blank", [lines]) or ("text", [lines]); a fenced code block is one block whatever it holds.
    items: list[tuple[str, list[str]]] = []
    cur: list[str] = []
    fence = None
    for line in lines:
        m = _FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
        if fence is None and not m and not line.strip():
            if cur:
                items.append(("text", cur))
                cur = []
            items.append(("blank", [line]))
        else:
            cur.append(line)
    if cur:
        items.append(("text", cur))

    def in_fence(block: list[str]) -> bool:
        return bool(_FENCE_RE.match(block[0]))

    def prose(block: list[str]) -> bool:
        return not in_fence(block) and _NOT_PROSE_RE.match(block[0]) is None

    moved = 0
    pending_label, label_slot = "", 0
    out_items: list[tuple[str, list[str]]] = []
    for kind, block in items:
        if kind == "blank":
            out_items.append((kind, block))
            continue
        if pending_label:
            if prose(block):
                block = [pending_label + " " + block[0].lstrip(), *block[1:]]
            else:
                # No prose after the label: it stays ahead of its figure(s), where they were written.
                out_items[label_slot:label_slot] = [("text", [pending_label]), ("blank", [""])]
            pending_label = ""
        if in_fence(block) or _NOT_PROSE_RE.match(block[0]) and not block[0].lstrip().startswith("!["):
            out_items.append((kind, block))
            continue
        images: list[str] = []
        rest: list[str] = []
        for line in block:
            found = _FIGURE_IMAGE_RE.findall(line)
            if not found:
                rest.append(line)
                continue
            images.extend(found)
            parts = _FIGURE_IMAGE_RE.split(line)
            text = parts[0].rstrip()
            for part in parts[1:]:
                part = part.lstrip()
                text = text + (" " if text and part else "") + part
            if text.strip():
                rest.append(text.strip() if len(parts) > 1 and not parts[0].strip() else text)
        residue = bool("".join(rest).strip())
        if not images or (len(images) == 1 and not residue):
            out_items.append((kind, block))
            continue
        moved += len(images)
        sep = ("blank", [""])
        if residue:
            label = "\n".join(rest).strip()
            if _RUN_IN_LABEL_RE.match(label):
                pending_label, label_slot = label, len(out_items)
            else:
                out_items.append(("text", rest))
                out_items.append(sep)
        for k, image in enumerate(images):
            out_items.append(("text", [image]))
            if k + 1 < len(images) or pending_label:
                out_items.append(sep)
    if pending_label:
        out_items[label_slot:label_slot] = [("text", [pending_label]), ("blank", [""])]
    if not moved:
        return markdown, 0
    return "\n".join(line for _k, block in out_items for line in block), moved
