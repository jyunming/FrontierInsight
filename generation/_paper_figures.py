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

    For a quest with call records (FI's record of what a figure draws) the list is the figures the paper itself embeds,
    an empty list included: a figure the paper left out is not added back, and there is no "every file" fallback. The
    numbers returned are those of the figures with a call record: only those get the paper's number and caption. A quest
    with no call record (an older one) gets :func:`figures_in_paper` and no numbers, as before."""
    shown = figures_in_paper(paper_md, files)
    state = _paper_state(raw_state, quest_root)
    if not state.get("figure_records"):
        return shown, {}
    from core import figure_claims as _figure_claims
    from core.engine import _figure_numbers

    numbers = _figure_numbers(state)
    with_calls = {name for name in numbers if _figure_claims.load_calls(quest_root, Path(name).stem)}
    if with_calls:
        placed = {m.group(1) for m in _PLACED_FIGURE.finditer(paper_md)}
        shown = [name for name in files if name in placed and name in numbers]
    else:
        shown = [name for name in shown if name in numbers]
    return shown, {name: numbers[name] for name in shown if name in with_calls}


def _fi_sentences(paper_raw: str) -> list[tuple[str, str]]:
    """``(marker key, rendered statement)`` of each figure statement FI composed in the paper (``paper_raw``, with its
    markers): the sentences a slide or a poster may repeat."""
    from core import paper_facts as _facts

    found = []
    for m in _facts._MARK.finditer(paper_raw or ""):
        if _facts._figure_parts(m.group(1)) is not None and m.group(2).strip():
            found.append((m.group(1), m.group(2).strip()))
    return found


def _wrap_verbatim(
    text: str, paper_raw: str, stem_numbers: dict[str, int],
) -> tuple[str, bool, list[dict[str, str]]]:
    """``text`` with each whole sentence that is one of FI's figure statements in the paper, word for word (whitespace and
    ``*`` emphasis aside), put back in FI's own marker, so the pointer rule knows it as FI's. The sentence may close with
    the pointer "(Figure N)" of the figure the statement is about, and no other. A sentence with anything added before or
    after it is not wrapped, and the rule takes it out whole; FI's statement closed by the pointer of another figure is
    taken out here (the rule leaves a pointer to a figure FI did not issue alone). ``(text, any wrapped, taken out)``."""
    from core import paper_facts as _facts

    wrapped = False
    removed: list[dict[str, str]] = []
    for key, statement in _fi_sentences(paper_raw):
        parts = _facts._figure_parts(key)
        number = stem_numbers.get(parts[0]) if parts else None
        words = statement.replace("*", " ").rstrip(" .!?").split()
        if not words or number is None:
            continue
        star = r"\*{0,2}"

        def sentence(pointer: str, words: list[str] = words) -> re.Pattern[str]:
            return re.compile(
                r"(?:^|(?<=[.!?] ))" + star + (star + r"\s+" + star).join(re.escape(w) for w in words) + star
                + pointer + r"(?:(?P<end>[.!?]+)\*{0,2}(?=\s|$)|\*{0,2}(?=[ \t]*$))",
                re.MULTILINE,
            )

        bare = statement.rstrip(" .!?")

        def put(m: re.Match[str], key: str = key, bare: str = bare) -> str:
            return f"<!-- fi:fact-{key} -->{bare}<!-- /fi:fact-{key} -->{m.group('ptr') or ''}{m.group('end') or ''}"

        def drop(m: re.Match[str], stem: str = parts[0]) -> str:
            removed.append({"key": f"figure:{stem}:claim", "sentence": m.group(0).strip(), "how": "sentence removed"})
            return ""

        text, n = sentence(r"(?P<ptr>[ \t]*\((?:Figure|Fig\.?)[ \t]*" + str(number) + r"\))?").subn(put, text)
        wrapped = wrapped or n > 0
        # the same words closed by the pointer of another figure
        text = sentence(r"(?P<ptr>[ \t]*\((?:Figures?|Figs?\.?)[^()]{1,30}\))").sub(drop, text)
    return text, wrapped, removed


# A heading, a list item or a quote: its text is read alone (the pointer rule leaves all three as they are). A table row is
# read cell by cell, and an entirely bold line is a lead sentence, not a heading before one.
_LINE_PREFIX = re.compile(r"^(?:[ \t]*(?:\#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)][ \t]+|(?:>[ \t]*)+))")
_ALL_BOLD = re.compile(r"^(?P<lead>[ \t]*)\*\*(?P<inner>[^*]+?)\*\*(?P<tail>[ \t]*)$")
# A bold sentence with prose after it on the same line; "**Figure 1.**" alone, a bare label, is a heading, not a sentence.
_BOLD_LEAD = re.compile(r"^(?P<lead>\*\*(?P<inner>[^*]+?[.!?])\*\*)(?P<rest>[ \t]+\S.*)$")
_LABEL_ONLY = re.compile(r"^\s*(?:figures?|figs?\.?)\s*\d+\s*[.:]?\s*$", re.IGNORECASE)
_TABLE_ROW = re.compile(r"^[ \t]*\|")
_DELIMITER_CELL = re.compile(r"^\s*:?-{2,}:?\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HOLD = "FIXLINEHOLD{}END"


def _filter_core(text: str, state: dict, paper_raw: str, stem_numbers: dict[str, int]) -> tuple[str, list[dict[str, str]]]:
    from core import paper_facts as _facts
    from core.engine import _take_out_unpointed

    text, wrapped, dropped = _wrap_verbatim(text, paper_raw, stem_numbers)
    out, taken = _facts.take_out_unfilled(text, figures_only=True)
    out, pointed = _take_out_unpointed(out, state)
    if wrapped:
        out = _facts.without_marks(out)
    return out, dropped + taken + pointed


def _filter_content(content: str, state: dict, paper_raw: str, stem_numbers: dict[str, int]):
    """One heading's, list item's, quote's, table cell's or bold line's text, filtered alone: ``(text or "" when nothing is
    left, taken)``."""
    bold = _ALL_BOLD.match(content)
    if bold:
        out, taken = _filter_core(bold.group("inner"), state, paper_raw, stem_numbers)
        if not taken:
            return content, []
        out = out.strip()
        return (bold.group("lead") + "**" + out + "**" + bold.group("tail") if out else ""), taken
    out, taken = _filter_core(content, state, paper_raw, stem_numbers)
    return (out.strip() if taken else content), taken


def _filter_table_row(line: str, state: dict, paper_raw: str, stem_numbers: dict[str, int]):
    """A table row read cell by cell: a cell about a figure that is not FI's own statement is emptied; a row left with no
    text in any cell is dropped (``""``). ``(row, taken)``."""
    from core.engine import _FIGURE_MENTION_RE

    cells = line.split("|")
    taken: list[dict[str, str]] = []
    for k, cell in enumerate(cells):
        if not cell.strip() or _DELIMITER_CELL.match(cell) or not _FIGURE_MENTION_RE.search(cell):
            continue
        content, more = _filter_content(cell.strip(), state, paper_raw, stem_numbers)
        if more:
            taken += more
            cells[k] = " " + content + " " if content else " "
    if taken and not any(c.strip() and not _DELIMITER_CELL.match(c) for c in cells):
        return "", taken
    return ("|".join(cells) if taken else line), taken


def take_out_unchecked(
    text: str, raw_state: dict | None, quest_root: Path, paper_raw: str = "",
) -> tuple[str, list[dict[str, str]]]:
    """``text`` written by a model for the poster or the slides, without the sentences about a figure FI drew that FI does
    not stand behind (the paper's own rule, ``core.engine._take_out_unpointed``) and without a statement about a figure
    that FI could not fill (``take_out_unfilled(figures_only=True)``): ``(text, taken_out)``. A whole sentence that is one
    of FI's own figure statements of the paper (``paper_raw``, the paper with its markers) word for word is kept; any
    other goes. A heading, a list item, a quote, a table cell and an entirely bold line that name a figure are read alone,
    since the rule leaves them as they are in the paper, and so is a bold lead sentence with prose after it on its line (a
    figure label such as "**Figure 1.**" with prose after it is left to the paper's rule, which judges that prose as naming
    the figure and takes the label out with it; a label alone on its line stays); one
    left empty is dropped. Unchanged for a quest none of
    whose figures has a call record."""
    from core.engine import _FIGURE_MENTION_RE, _figure_numbers

    if not text:
        return text, []
    state = _paper_state(raw_state, quest_root)
    stem_numbers = {Path(n).stem: k for n, k in _figure_numbers(state).items()}
    lines = text.split("\n")
    alone: dict[int, tuple[str, str]] = {}
    taken: list[dict[str, str]] = []
    fence = None
    for i, line in enumerate(lines):
        fenced = _FENCE.match(line)
        if fenced:
            fence = None if fence == fenced.group(1) else (fence or fenced.group(1))
            continue
        if fence is not None or not _FIGURE_MENTION_RE.search(line):
            continue
        if _TABLE_ROW.match(line):
            alone[i] = ("table", line)
            continue
        prefix = _LINE_PREFIX.match(line)
        head = prefix.group(0) if prefix else ""
        lead = _BOLD_LEAD.match(line[len(head):])
        if lead and _FIGURE_MENTION_RE.search(lead.group("inner")) and not _LABEL_ONLY.match(lead.group("inner")):
            out, more = _filter_core(lead.group("inner"), state, paper_raw, stem_numbers)
            if more:  # the bold lead is not FI's: it goes (or keeps what is FI's), and its prose is read as usual
                taken += more
                kept = "**" + out.strip() + "** " if out.strip() else ""
                line = lines[i] = head + kept + lead.group("rest").lstrip()
                if not _FIGURE_MENTION_RE.search(line):
                    continue
        if prefix or _ALL_BOLD.match(line):
            alone[i] = ("item", line)
    if alone:
        lines = [_HOLD.format(i) if i in alone else line for i, line in enumerate(lines)]
    out, more = _filter_core("\n".join(lines), state, paper_raw, stem_numbers)
    taken += more
    if alone:
        rebuilt = []
        for line in out.split("\n"):
            held = re.fullmatch(_HOLD.format(r"(\d+)"), line)
            if not held:
                rebuilt.append(line)
                continue
            kind, original = alone[int(held.group(1))]
            if kind == "table":
                row, more = _filter_table_row(original, state, paper_raw, stem_numbers)
                taken += more
                if row or not more:
                    rebuilt.append(row or original)
                continue
            prefix = _LINE_PREFIX.match(original)
            head = prefix.group(0) if prefix else ""
            content, more = _filter_content(original[len(head):], state, paper_raw, stem_numbers)
            taken += more
            if content:
                rebuilt.append(head + content)
        out = "\n".join(rebuilt)
    return out, taken


def log_taken(taken: list[dict[str, str]], quest_root: Path, who: str, module_log) -> None:  # noqa: ANN001
    """One line per removal, in the quest's ``.fi/run.log`` (the log the paper's removals go to) and the module's log,
    not shown on the slide or poster."""
    from core.provider import quest_run_log

    quest_log = quest_run_log(quest_root)
    for t in taken:
        line = '%s: a sentence about a figure FI does not stand behind was left out (%s): "%s"' % (
            who, t.get("key", "figure"), str(t.get("sentence", ""))[:120])
        module_log.warning("%s", line)
        if quest_log is not None:
            quest_log.warning("%s", line)
