"""The paper's figures from their paper copies.

Each figure an experiment saves into ``figures/`` also gets a paper copy, ``figures/paper/<stem>.pdf``, drawn at the
width of a one-column page's text with text at the body's size (``core/plot_style.py``). The LaTeX source of
``paper.pdf`` uses that vector copy in place of the PNG, at its own width as a share of the line width, so a point of
text in the figure prints as a point on the page. ``paper.md`` itself, the HTML render (which cannot show a PDF image)
and the slides keep the PNGs. A two-column paper keeps the PNGs too: the paper copy is drawn for one column of text
6.3 in wide, and scaled into a 3.5 in column its text would print at half its size.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.plot_style import PAPER_DIRNAME, PAPER_FULL_WIDTH_IN, PAPER_PANEL_MIN_SHARE, RECORDS_DIRNAME

# The target of a Markdown image (or link) to a figure file in figures/, with an optional {attributes} block after it.
_FIGURE_TARGET = re.compile(r"\]\(figures/([^)\s/]+)\.(png|jpe?g|svg)\)(\{[^}]*\})?", re.IGNORECASE)
_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\]")


def _width_in(pdf: Path) -> float | None:
    """The page width of a one-page PDF matplotlib wrote, in inches; ``None`` when it cannot be read."""
    try:
        match = _MEDIABOX.search(pdf.read_bytes())
    except OSError:
        return None
    if not match:
        return None
    try:
        x0, _y0, x1, _y1 = (float(v) for v in match.groups())
    except ValueError:
        return None
    return (x1 - x0) / 72 if x1 > x0 else None


def recorded_panel_state(root: Path, stem: str) -> tuple[str, float | None]:
    """What the bootstrap recorded of the paper copy of ``figures/<stem>`` (``.fi/figure_records/<stem>.json``):
    ``("ok", share)`` when it measured every axes (``share`` is the smallest panel's share of its drawn size, ``None``
    when the figure has no axes with a size); ``("failed", None)`` when it tried to measure and could not, or only in
    part; ``("old", None)`` when the record says nothing of a measurement (a copy from before it was made)."""
    try:
        record = json.loads((Path(root) / ".fi" / RECORDS_DIRNAME / f"{stem}.json").read_text(encoding="utf-8"))
        paper = record["paper"]
        status = paper.get("panel_measure")
        share = paper.get("panel_share")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "old", None
    if status is None:
        return ("old", None) if share is None else ("ok", float(share) if isinstance(share, (int, float)) else None)
    if status != "ok":
        return "failed", None
    return "ok", float(share) if isinstance(share, (int, float)) and not isinstance(share, bool) else None


def copy_is_unusable(root: Path, stem: str) -> bool:
    """True when the copy is to be left out of the paper: its smallest panel kept under ``PAPER_PANEL_MIN_SHARE``, or
    its panels could not be measured (a copy no one has looked at is not shown). A copy with no record of a
    measurement at all is used, as before."""
    state, share = recorded_panel_state(root, stem)
    return state == "failed" or (state == "ok" and share is not None and share < PAPER_PANEL_MIN_SHARE)


def placed_copies(root: Path) -> list[tuple[int, str, str, float | None]]:
    """The paper copies the paper's source (``paper_pdf_source.md``, which FI wrote) points at, each as
    ``(figure number, stem, state, share)`` (see ``recorded_panel_state``). The number is the figure's place among
    the figure lines, counted by the same walker that numbers the captions. Empty when there is no such source."""
    from generation._figure_captions import figure_lines

    try:
        lines = (Path(root) / "paper_pdf_source.md").read_text(encoding="utf-8").split("\n")
    except OSError:
        return []
    found = []
    for number, (_index, fig) in enumerate(figure_lines(lines), start=1):
        target = re.fullmatch(rf"\]\(figures/{PAPER_DIRNAME}/([^/)\s]+)\.pdf\)(?:\{{[^}}]*\}})?\s*", fig.group("rest"))
        if target:
            state, share = recorded_panel_state(root, target.group(1))
            found.append((number, target.group(1), state, share))
    return found


#: The most of the page's text height a paper copy is set at: a third, so a tall figure leaves room for its caption
#: and the text around it.
PAPER_HEIGHT_SHARE = 33


def prefer_paper_figures(markdown: str, root: Path, *, two_column: bool = False) -> tuple[str, int]:
    """``(markdown, n)``: each image of ``figures/<stem>.png`` (or .jpg, .svg) whose paper copy
    ``figures/paper/<stem>.pdf`` exists under ``root``, beside the figure itself, points at that copy instead, with
    ``{width=NN% height=33%}``: its share of the line width that the copy was drawn at, and at most a third of the
    page's text height (the templates keep a figure's aspect ratio, so a tall figure is scaled down to that height and
    a wide one is set at its width). An attribute block the writer gave is kept as it is. ``n`` is how many were
    swapped. A two-column paper is returned unchanged."""
    if two_column:
        return markdown, 0
    swapped = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal swapped
        stem, suffix, attributes = match.group(1), match.group(2), match.group(3)
        pdf = Path(root) / "figures" / PAPER_DIRNAME / f"{stem}.pdf"
        # Only the copy of a figure that is there: a figure FI took out of figures/ is not shown through a copy left
        # behind.
        if not pdf.is_file() or not (Path(root) / "figures" / f"{stem}.{suffix}").is_file():
            return match.group(0)
        # A copy whose smallest panel the bootstrap recorded under the share a copy keeps is not used: the figure is
        # set from its PNG.
        if copy_is_unusable(root, stem):
            return match.group(0)
        if not attributes:
            width = _width_in(pdf)
            share = min(100, max(10, round(100 * width / PAPER_FULL_WIDTH_IN))) if width else 100
            attributes = f"{{width={share}% height={PAPER_HEIGHT_SHARE}%}}"
        swapped += 1
        return f"](figures/{PAPER_DIRNAME}/{stem}.pdf){attributes}"

    return _FIGURE_TARGET.sub(swap, markdown), swapped


_PLACED_FIGURE = re.compile(r"!\[(?:(?!\]\()[^\n])*\]\((?:\./)?figures/([^)\s/]+)\)")


def figures_in_paper(paper_md: str, files: list[str]) -> list[str]:
    """The files of ``figures/`` that the paper shows, in the order of ``files``.

    FI places the figures in the paper (``core/engine.py:_place_figures``) and leaves out one whose record says it draws
    nothing, but that file stays in ``figures/``. The poster and the slides show the figures the paper shows, not every
    file. A paper that embeds none of ``files`` (an older quest, or one written before FI placed figures) gives no list,
    and then every file stays, as before.
    """
    placed = {m.group(1) for m in _PLACED_FIGURE.finditer(paper_md)}
    kept = [name for name in files if name in placed]
    return kept if kept else list(files)


# -- The paper's own figure list, for the poster and the slides ------------------------------------------------------


def _paper_state(raw_state: dict | None, quest_root: Path) -> dict:
    """The final state of the quest with its root, as ``core.engine`` functions read it."""
    state = dict(raw_state or {})
    state.setdefault("quest_root", quest_root)
    return state


def paper_figure_set(
    paper_md: str, files: list[str], raw_state: dict | None, quest_root: Path,
) -> tuple[list[str], dict[str, int]]:
    """``(files to show, {file: the paper's figure number})``: the figures the poster and the slides take from the paper's
    own list (``core.engine._figure_numbers``, the list FI placed the paper's figures by), not a list of their own.

    A figure the paper left out (its record says it draws nothing) is not in it, even when the paper embeds no figure and
    :func:`figures_in_paper` would give every file back. The numbers returned are those of the figures with a call record
    (FI's record of what they draw): only those get the paper's number and caption on the poster. A quest whose state holds
    no figure record (an older one) gets :func:`figures_in_paper` and no numbers, as before."""
    shown = figures_in_paper(paper_md, files)
    state = _paper_state(raw_state, quest_root)
    if not state.get("figure_records"):
        return shown, {}
    from core import figure_claims as _figure_claims
    from core.engine import _figure_numbers

    numbers = _figure_numbers(state)
    shown = [name for name in shown if name in numbers]
    checked = {
        name: number for name, number in numbers.items()
        if name in shown and _figure_claims.load_calls(quest_root, Path(name).stem)
    }
    return shown, checked


def _fi_sentences(paper_raw: str) -> list[tuple[str, str]]:
    """``(marker key, rendered statement)`` of each figure statement FI composed in the paper (``paper_raw``, with its
    markers): the sentences a slide or a poster may repeat."""
    from core import paper_facts as _facts

    found = []
    for m in _facts._MARK.finditer(paper_raw or ""):
        if _facts._figure_parts(m.group(1)) is not None and m.group(2).strip():
            found.append((m.group(1), m.group(2).strip()))
    return found


def _wrap_verbatim(text: str, paper_raw: str) -> tuple[str, bool]:
    """``text`` with each sentence that is one of FI's figure statements in the paper, word for word (whitespace and
    ``*`` emphasis aside; a closing "(Figure N)" and the full stop may follow), put back in FI's own marker, so the
    pointer rule knows it as FI's. Any other sentence is left as it was. ``(text, whether any was wrapped)``."""
    wrapped = False
    for key, statement in _fi_sentences(paper_raw):
        words = statement.replace("*", " ").rstrip(" .!?").split()
        if not words:
            continue
        star = r"\*{0,2}"
        pattern = re.compile(
            r"(?<![A-Za-z0-9,;])" + star + (star + r"\s+" + star).join(re.escape(w) for w in words) + star
            + r"(?=(?:\s*\((?:Figures?|Figs?\.?)[^()]{0,30}\))?(?:[.!?]+|[ \t]*(?:\n|$)))"
        )
        bare = statement.rstrip(" .!?")
        text, n = pattern.subn(lambda _m: f"<!-- fi:fact-{key} -->{bare}<!-- /fi:fact-{key} -->", text)
        wrapped = wrapped or n > 0
    return text, wrapped


def take_out_unchecked(
    text: str, raw_state: dict | None, quest_root: Path, paper_raw: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """``text`` written by a model for the poster or the slides, without the sentences about a figure FI drew that FI does
    not stand behind (the paper's own rule, ``core.engine._take_out_unpointed``) and without a statement about a figure
    that FI could not fill (``take_out_unfilled(figures_only=True)``): ``(text, taken_out)``. A sentence that is one of
    FI's own figure statements of the paper (``paper_raw``, the paper with its markers) word for word is kept; a
    paraphrase of it goes. Unchanged for a quest none of whose figures has a call record."""
    from core import paper_facts as _facts
    from core.engine import _take_out_unpointed

    if not text:
        return text, []
    state = _paper_state(raw_state, quest_root)
    text, wrapped = _wrap_verbatim(text, paper_raw)
    out, taken = _facts.take_out_unfilled(text, figures_only=True)
    out, pointed = _take_out_unpointed(out, state)
    if wrapped:
        out = _facts.without_marks(out)
    return out, taken + pointed
