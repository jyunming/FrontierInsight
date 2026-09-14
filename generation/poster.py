"""Poster generator.

The model writes the poster's words as JSON blocks: a headline, headings,
short texts, bullet lists, and figures with captions, citing sources as
[n]. This module lays them out on a beamerposter sheet sized by
``output.poster_size`` and compiles it. The engine is pdflatex, or XeLaTeX
when the text holds Chinese, Japanese or Korean; tectonic comes through the
engine discovery shared with the paper.

The layout follows published poster guidance:
- the main finding as the headline, with the paper title under it;
- the author line, and a QR code when a URL is set;
- body text of 24 pt or more, headings about 1.5 times that, numbered
  figure captions;
- columns that end at about the same height;
- a reference band listing only the sources the poster cites (at most 8,
  no raw URLs).

The layout is planned from estimated block heights, compiled, then
measured with ``generation/_pdf_measure.py``. Each measurement corrects
the estimates and the plan is redone:
- whole sections go to columns where they fit, and figure widths
  (70–100%) fill or relieve each column and even the columns out;
- when the content cannot fit, the longest list loses items, then the
  longest text loses its last sentences, then text blocks and finally
  figures from the middle go. The first and last blocks always stay.
- Fonts never shrink.

If no LaTeX engine is reachable, only ``poster.tex`` is produced, with a
``poster_pdf_skipped.md`` diagnostic next to it.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re as _re
import shutil  # noqa: F401 — existing tests monkeypatch ``poster.shutil.which``
import string
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from core.config import Config
from core.engine import (
    QuestArtifacts,
    _latex_esc,
    build_further_reading,
    build_references,
)
from core.provider import (
    LLMClient,
    ProxySupervisor,
    PROXY_PROVIDERS,
    model_for_node,
    resolve_endpoint_async,
)
from generation import _cjk
from generation._figure_captions import without_number
from generation._pdf_engine import find_pdf_engine
from generation._pdf_measure import measure_pdf, poster_report
from generation.paper import _sanitize_unicode_for_latex

_log = logging.getLogger("frontier_insight.poster")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "agents" / "poster.md"
TEMPLATE_PATH = REPO_ROOT / "templates" / "poster" / "poster.tex"
# Frontier Insight brand mark, copied next to poster.tex for the header
# lockup and the reference band.
ICON_PNG = REPO_ROOT / "vscode-frontier-insight" / "images" / "fi_glyph_teal.png"

_PT_PER_CM = 72 / 2.54
_MAX_COMPILES = 6
_MAX_REFERENCES = 8
_SELECTED_SOURCES = 5
_MAX_HEADLINE_WORDS = 20
_FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf")
_DEFAULT_ASPECT = 0.62
# Average advance of a character of English prose in Helvetica, in ems
# (measured: 67 characters in a 25.4 cm column at 24 pt).
_CHAR_EM = 0.46
_MIN_FIGURE_WIDTH = 0.7
_FIGURE_STEP = 0.05
# Columns planned to end within this of each other; the measurement flags
# 5 cm or more.
_BALANCE_CM = 3.0
# Added to the heading a column repeats when it opens inside a section.
_CONTINUED = " (continued)"
# A compile that measures this much more column room than the plan assumed
# is planned again for the measured room.
_MORE_ROOM_CM = 3.0
# Room kept free when planning, for estimates that run a little short.
_PLAN_ROOM = 0.97
# The share of the sheet's height the columns get before a compile has
# measured it (header and reference band take the rest).
_FIRST_GUESS_ROOM = 0.62


@dataclass(frozen=True)
class _Sheet:
    label: str
    width_cm: float
    height_cm: float
    columns: int
    body_pt: float
    head_pt: float
    title_pt: float
    subtitle_pt: float
    author_pt: float
    caption_pt: float
    refs_pt: float
    words: int
    margin_cm: float
    gap_cm: float
    qr_cm: float

    @property
    def column_cm(self) -> float:
        return (self.width_cm - 2 * self.margin_cm - (self.columns - 1) * self.gap_cm) / self.columns


# A0 is A1 scaled by the square root of two in size and type, so it holds
# the same number of words. Columns are about 26 cm on A1 and 37 cm on the
# larger sheets, which sets about 63 characters per line.
_SHEETS: dict[str, _Sheet] = {
    "a1_portrait": _Sheet(
        "A1 portrait (59.4 x 84.1 cm)", 59.4, 84.1, 2,
        26, 40, 80, 36, 28, 20, 16, 380, 2.5, 2.4, 6.5,
    ),
    "a0_portrait": _Sheet(
        "A0 portrait (84.1 x 118.9 cm)", 84.1, 118.9, 2,
        36, 56, 112, 50, 40, 28, 22, 380, 3.5, 3.4, 9.0,
    ),
    "landscape_48x36": _Sheet(
        "48 x 36 inch landscape", 121.92, 91.44, 3,
        36, 56, 112, 50, 40, 28, 22, 500, 3.5, 3.4, 9.0,
    ),
}


# ---------------------------------------------------------------------------
# LaTeX safety

# Emoji / pictographs / dingbats / regional-indicator ranges. pdflatex
# (utf8 inputenc) hard-errors on these ("Unicode character … not set up
# for use with LaTeX"), and an LLM column or a scraped source title can
# easily carry one (a "📊 Global EV Sales Report" citation is what first
# broke poster.pdf). We strip them outright — they carry no meaning the
# poster needs.
_EMOJI_RE = _re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji + symbols + pictographs
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U00002B00-\U00002BFF"   # misc symbols and arrows
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U00002190-\U000021FF"   # arrows
    "]+",
    flags=_re.UNICODE,
)


def _latex_safe_body(body: str) -> str:
    """Make a fully-substituted poster.tex safe for pdflatex: map common
    Unicode typography to LaTeX equivalents (shared with the paper
    generator) and strip emoji/pictographs that have no LaTeX mapping."""
    return _EMOJI_RE.sub("", _sanitize_unicode_for_latex(body))


# Bare LaTeX specials that hard-error pdflatex when an LLM writes them as
# literal prose in a poster column (a stray 'Microlensing & Timing' / '50%').
# Posters are blocks + itemize, never tabular, so a bare ``&`` is always a
# literal ampersand. Already-escaped specials (``\&``) are skipped via the
# negative lookbehind; backslashes and braces are untouched.
_LATEX_TEXT_SPECIAL_RE = _re.compile(r"(?<!\\)([&%#])")


def _escape_latex_text_specials(s: str) -> str:
    """Escape bare ``&``, ``%``, ``#`` in model-written LaTeX so a literal
    ampersand or percent doesn't fatally break the compile."""
    return _LATEX_TEXT_SPECIAL_RE.sub(r"\\\1", s or "")


# A stray beamer ``\column`` inside a column reads the next token as a width
# and stops pdflatex with "Missing number, treated as zero". ``(?![A-Za-z])``
# keeps ``\columnwidth`` / ``\columnsep``; an optional ``{width}`` goes with it.
_BARE_COLUMN_RE = _re.compile(r"\\column(?![A-Za-z])(?:[ \t]*\{[^{}]*\})?")


def _strip_column_commands(s: str) -> str:
    """Remove ``\\column`` commands a model writes into a poster column."""
    return _BARE_COLUMN_RE.sub("", s or "")


# A model that doubles every backslash in its JSON doubles the ``\n`` it meant
# as a line break too, and LaTeX stops on the undefined ``\n`` it receives
# (``\textbf{Rates}\nThe slope``). LaTeX has no ``\n`` command, so a ``\n`` that
# does not begin a real one (``\noindent``, ``\nabla``) is a line break.
_LITERAL_NEWLINE_RE = _re.compile(r"(?<!\\)\\n([A-Za-z]*)")


def _literal_newlines_to_breaks(s: str) -> str:
    """Turn the literal ``\\n`` a model left in a poster column into a newline."""
    def fix(m: "_re.Match[str]") -> str:
        if "n" + m.group(1) in _LATEX_N_COMMANDS:
            return m.group(0)
        return "\n" + m.group(1)
    return _LITERAL_NEWLINE_RE.sub(fix, s or "")


def _clean_model_column(s: str) -> str:
    """The model's column LaTeX with its known JSON and beamer slips undone."""
    return _strip_column_commands(_literal_newlines_to_breaks(s))


_TEXT_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "&": r"\&", "%": r"\%",
    "$": r"\$", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape_text(s: str) -> str:
    """Plain text with every LaTeX special escaped."""
    return "".join(_TEXT_ESCAPES.get(ch, ch) for ch in s or "")


# Inline math the model writes as $...$: no space just inside either dollar,
# so "$5 and $10" stays two amounts of money.
_MATH_RE = _re.compile(r"(?<!\\)\$(?!\s)([^$\n]+?)(?<![\s\\])\$")
_EMPHASIS_RE = _re.compile(r"\*\*(.+?)\*\*|(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_LATEX_EMPHASIS_RE = _re.compile(r"\\(textbf|emph|textit)\{([^{}]*)\}")


def _styled(s: str) -> str:
    out: list[str] = []
    pos = 0
    for m in _EMPHASIS_RE.finditer(s):
        out.append(_escape_text(s[pos:m.start()]))
        if m.group(1) is not None:
            out.append(r"\textbf{" + _escape_text(m.group(1)) + "}")
        else:
            out.append(r"\emph{" + _escape_text(m.group(2)) + "}")
        pos = m.end()
    out.append(_escape_text(s[pos:]))
    return "".join(out)


def _inline_latex(text: str) -> str:
    """The model's plain text as LaTeX: ``$math$`` kept, ``**bold**`` and
    ``*italic*`` set, every other special escaped. A ``\\textbf{}`` the
    model wrote anyway is read as bold rather than printed."""
    text = _LATEX_EMPHASIS_RE.sub(
        lambda m: ("**%s**" if m.group(1) == "textbf" else "*%s*") % m.group(2), text or "",
    )
    out: list[str] = []
    pos = 0
    for m in _MATH_RE.finditer(text):
        math_src = m.group(1)
        if math_src.count("{") != math_src.count("}"):
            continue  # unbalanced braces would stop the compile; print it as text
        out.append(_styled(text[pos:m.start()]))
        out.append("$" + math_src + "$")
        pos = m.end()
    out.append(_styled(text[pos:]))
    return "".join(out)


def _plain(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


# A sentence ends at . ! or ? before a capital, a digit, a citation or math.
_SENTENCE_BREAK_RE = _re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[$(])")
_ABBREVIATION_END_RE = _re.compile(r"(?:\bal|\be\.g|\bi\.e|\bvs|\bFig|\bEq)\.$")


def _sentences(text: str) -> list[str]:
    pieces: list[str] = []
    for piece in _SENTENCE_BREAK_RE.split(text):
        if pieces and _ABBREVIATION_END_RE.search(pieces[-1]):
            pieces[-1] = pieces[-1] + " " + piece
        else:
            pieces.append(piece)
    return pieces


# ---------------------------------------------------------------------------
# Blocks


@dataclass
class _Block:
    kind: str  # heading | text | bullets | figure | raw
    text: str = ""
    items: list[str] = field(default_factory=list)
    file: str = ""
    caption: str = ""

    def texts(self) -> list[str]:
        return [self.text, self.caption, *self.items]


def _blocks_from_reply(parsed: dict, figure_files: set[str]) -> list[_Block]:
    """The blocks of a reply. A reply in the old two-column LaTeX shape
    (``left`` / ``right``) becomes raw blocks, one per column."""
    blocks: list[_Block] = []
    raw_blocks = parsed.get("blocks")
    if isinstance(raw_blocks, list):
        for item in raw_blocks:
            if not isinstance(item, dict):
                continue
            kind = _plain(item.get("type")).lower()
            if kind == "heading" and _plain(item.get("text")):
                blocks.append(_Block("heading", text=_plain(item.get("text"))))
            elif kind in ("text", "paragraph") and _plain(item.get("text")):
                blocks.append(_Block("text", text=_plain(item.get("text"))))
            elif kind in ("bullets", "list"):
                items = [_plain(x) for x in item.get("items") or [] if _plain(x)]
                if items:
                    blocks.append(_Block("bullets", items=items))
            elif kind == "figure":
                name = Path(_plain(item.get("file"))).name
                if name in figure_files:
                    blocks.append(_Block("figure", file=name, caption=_plain(item.get("caption"))))
        return blocks
    for key in ("left", "middle", "right"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            blocks.append(_Block("raw", text=_clean_model_column(value)))
    return blocks


_PAPER_FIGURE_RE = _re.compile(r"!\[(?P<alt>[^\]]*)\]\((?:\./)?figures/(?P<name>[^)\s]+)")


def _paper_captions(paper_md: str) -> dict[str, str]:
    """Each figure's caption as the paper wrote it, without "Figure N."."""
    captions: dict[str, str] = {}
    for m in _PAPER_FIGURE_RE.finditer(paper_md):
        alt = _plain(without_number(m.group("alt")).replace("**", ""))
        captions.setdefault(m.group("name"), alt.strip())
    return captions


def _figure_files(figures_dir: Path | None) -> list[str]:
    if not figures_dir or not figures_dir.is_dir():
        return []
    return sorted(
        p.name for p in figures_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _FIGURE_SUFFIXES
    )


def _with_every_figure(blocks: list[_Block], figures: list[str], captions: dict[str, str]) -> list[_Block]:
    """Add the figures the model left out, before the closing heading, so
    the poster still ends on its conclusion."""
    if any(b.kind == "raw" for b in blocks):
        return blocks
    used = {b.file for b in blocks if b.kind == "figure"}
    missing = [
        _Block("figure", file=name, caption=captions.get(name) or Path(name).stem.replace("_", " "))
        for name in figures if name not in used
    ]
    if not missing:
        return blocks
    headings = [i for i, b in enumerate(blocks) if b.kind == "heading"]
    at = headings[-1] if len(headings) > 1 else len(blocks)
    return blocks[:at] + missing + blocks[at:]


_CITATION_RE = _re.compile(r"\[(\s*W?\d+(?:\s*[,;–-]\s*W?\d+)*\s*)\]")


def _labels_in(group: str) -> list[str]:
    labels: list[str] = []
    for part in _re.split(r"\s*[,;]\s*", group.strip()):
        span = _re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span and 0 <= int(span.group(2)) - int(span.group(1)) < 20:
            labels += [str(n) for n in range(int(span.group(1)), int(span.group(2)) + 1)]
        elif _re.fullmatch(r"W?\d+", part):
            labels.append(part)
    return labels


def _label_order(label: str) -> tuple[bool, int]:
    """Numbered references first, then web pages, each by number."""
    return (label.startswith("W"), int(label.lstrip("W")))


def _cited_labels(blocks: list[_Block], known: set[str]) -> list[str]:
    """Labels of real sources the poster cites: the first eight it cites,
    listed in reference order."""
    order: list[str] = []
    for block in blocks:
        for text in block.texts():
            for m in _CITATION_RE.finditer(text):
                for label in _labels_in(m.group(1)):
                    if label in known and label not in order:
                        order.append(label)
    return sorted(order[:_MAX_REFERENCES], key=_label_order)


def _keep_citations(text: str, kept: set[str]) -> str:
    """Drop citations of sources the band does not list."""
    if not text:
        return text

    def fix(m: "_re.Match[str]") -> str:
        labels = [label for label in _labels_in(m.group(1)) if label in kept]
        return f"[{', '.join(labels)}]" if labels else ""

    out = _CITATION_RE.sub(fix, text)
    out = _re.sub(r"[ \t]{2,}", " ", out)
    return _re.sub(r"\s+([.,;:!?)])", r"\1", out).strip()


def _with_citations(block: _Block, kept: set[str]) -> _Block:
    return replace(
        block,
        text=_keep_citations(block.text, kept),
        caption=_keep_citations(block.caption, kept),
        items=[_keep_citations(item, kept) for item in block.items],
    )


def _band_entry(source: dict, label: str) -> str:
    """A reference as the band prints it: author, year, title, venue and DOI
    for a paper; title and site for a web page. Never a URL."""
    authors = [str(a) for a in source.get("authors") or [] if a]
    who = ""
    if len(authors) > 2:
        who = f"{authors[0]} et al."
    elif authors:
        who = " and ".join(authors)
    year = f"({source['year']})" if source.get("year") else ""
    parts = [" ".join(p for p in (who, year) if p)]
    if source.get("title"):
        parts.append(str(source["title"]).strip().rstrip("."))
    if label.startswith("W"):
        site = source.get("site") or (urlparse(str(source.get("url") or "")).hostname or "")
        parts.append(site.removeprefix("www."))
    else:
        parts.append(str(source.get("venue") or ""))
        if source.get("doi"):
            parts.append(f"doi:{source['doi']}")
        elif source.get("arxiv_id"):
            parts.append(f"arXiv:{source['arxiv_id']}")
    return f"[{label}] " + ". ".join(p for p in parts if p) + "."


def _band_latex(entries: list[str], heading: str, columns: int, icon_ok: bool) -> str:
    mark = (
        r"\hfill\raisebox{-0.25\height}{\includegraphics[height=1.3em]{fi_icon.png}}"
        if icon_ok else ""
    )
    if not entries:
        return r"{\ttfamily\bfseries\color{fiteal}FRONTIER INSIGHT}" + mark + r"\par"
    body = "\n".join(_latex_esc(entry) + r"\par\vspace{0.25em}" for entry in entries)
    # An entry never breaks across the band's columns.
    return (
        r"{\bfseries " + heading + "}" + mark + r"\par\vspace{0.3em}" + "\n"
        + r"\begin{multicols}{" + str(columns) + r"}\raggedcolumns\raggedright\interlinepenalty=10000" + "\n"
        + body + "\n" + r"\end{multicols}"
    )


def _qr_latex(url: str, sheet: _Sheet) -> str:
    if not url:
        return ""
    escaped = "".join("\\" + ch if ch in "#$&^_~%{}\\" else ch for ch in url)
    return (
        r"\hfill\begin{minipage}[t]{" + f"{sheet.qr_cm + 0.5:g}" + r"cm}\raggedleft\vspace{0pt}"
        + r"\qrcode[height=" + f"{sheet.qr_cm:g}" + "cm]{" + escaped + r"}\end{minipage}"
        + r"\hspace*{" + f"{sheet.margin_cm:g}" + "cm}"
    )


# ---------------------------------------------------------------------------
# Layout


def _sections(blocks: list[_Block]) -> list[list[int]]:
    """Each heading with everything up to the next heading."""
    sections: list[list[int]] = []
    for i, block in enumerate(blocks):
        if block.kind == "heading" or not sections:
            sections.append([])
        sections[-1].append(i)
    return sections


def _units(blocks: list[_Block]) -> list[list[int]]:
    """Blocks one by one, except that a heading stays with the next block and
    the text or list after a figure stays with the figure. A column opened on
    the two sentences that discussed the previous column's figure, under no
    heading."""
    units: list[list[int]] = []
    i = 0
    while i < len(blocks):
        unit = [i, i + 1] if blocks[i].kind == "heading" and i + 1 < len(blocks) else [i]
        after = unit[-1] + 1
        if blocks[unit[-1]].kind == "figure" and after < len(blocks) and blocks[after].kind in ("text", "bullets"):
            unit.append(after)
        units.append(unit)
        i = unit[-1] + 1
    return units


def _partition(heights: list[float], columns: int) -> list[tuple[int, int]]:
    """Split ``heights`` into ``columns`` runs in order, keeping the tallest
    run as short as possible. Returns ``(start, end)`` per column. Ties go
    to the later cut, so short content fills the first columns."""
    n = len(heights)
    prefix = [0.0]
    for h in heights:
        prefix.append(prefix[-1] + h)
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(columns + 1)]
    cut = [[0] * (n + 1) for _ in range(columns + 1)]
    best[0][0] = 0.0
    for k in range(1, columns + 1):
        for i in range(n + 1):
            for j in range(i + 1):
                cost = max(best[k - 1][j], prefix[i] - prefix[j])
                if cost <= best[k][i]:
                    best[k][i], cut[k][i] = cost, j
    bounds: list[tuple[int, int]] = []
    i = n
    for k in range(columns, 0, -1):
        j = cut[k][i]
        bounds.append((j, i))
        i = j
    return list(reversed(bounds))


def _aspect(figures_dir: Path | None, name: str) -> float:
    if figures_dir is None:
        return _DEFAULT_ASPECT
    try:
        from PIL import Image

        with Image.open(figures_dir / name) as image:
            width, height = image.size
        return height / width if width else _DEFAULT_ASPECT
    except Exception:
        return _DEFAULT_ASPECT


def _lines(chars: int, width_pt: float, size_pt: float) -> int:
    per_line = max(1, int(width_pt / (_CHAR_EM * size_pt)))
    return max(1, math.ceil(chars / per_line))


class _Layout:
    """Blocks placed in columns: the plan, the estimates it rests on, and
    the corrections each measured compile teaches."""

    def __init__(self, sheet: _Sheet, blocks: list[_Block], aspects: dict[str, float], figure_paths: dict[str, str]):
        self.sheet = sheet
        self.blocks = list(blocks)
        self.factors = [1.0] * len(self.blocks)
        self.widths = [1.0] * len(self.blocks)
        self.aspects = aspects
        self.figure_paths = figure_paths
        self.room = _PLAN_ROOM
        self.trimmed = 0
        self.dropped = 0
        self.groups: list[list[int]] = []
        # The header and band heights are only known once the sheet is
        # compiled, so the first plan never cuts content to fit a guess;
        # a measured spill or measured room decides.
        self.plan(_FIRST_GUESS_ROOM * sheet.height_cm * _PT_PER_CM, cut=False)

    # -- estimates ---------------------------------------------------------

    def _heading_height(self, text: str) -> float:
        s = self.sheet
        return _lines(len(text), s.column_cm * _PT_PER_CM, s.head_pt) * 1.15 * s.head_pt + 1.3 * s.body_pt

    def _raw_height(self, i: int) -> float:
        block, s = self.blocks[i], self.sheet
        width = s.column_cm * _PT_PER_CM
        body, lead = s.body_pt, 1.25 * s.body_pt
        if block.kind == "heading":
            return self._heading_height(block.text)
        if block.kind == "text":
            return _lines(len(block.text), width, body) * lead + 0.7 * body
        if block.kind == "bullets":
            inner = width - 1.3 * body
            return sum(_lines(len(item), inner, body) * lead + 0.3 * body for item in block.items) + 0.7 * body
        if block.kind == "figure":
            text_height = _FIRST_GUESS_ROOM * s.height_cm * _PT_PER_CM
            image = min(self.widths[i] * width * self.aspects.get(block.file, _DEFAULT_ASPECT), 0.4 * text_height)
            caption = _lines(len(block.caption) + 11, width, s.caption_pt) * 1.25 * s.caption_pt
            return image + caption + 1.25 * body
        plain = _re.sub(r"\\[A-Za-z]+|[{}]", "", block.text)
        return (_lines(len(plain), width, body) + block.text.count("\n")) * lead

    def estimate(self, i: int) -> float:
        return self._raw_height(i) * self.factors[i]

    def continued_heading(self, c: int) -> int | None:
        """The heading of the section column ``c`` continues, when the column
        does not open on a heading of its own. The column then repeats it."""
        group = self.groups[c]
        if c == 0 or not group or self.blocks[group[0]].kind == "heading":
            return None
        return next((i for i in range(group[0] - 1, -1, -1) if self.blocks[i].kind == "heading"), None)

    def _continuation_height(self, c: int) -> float:
        heading = self.continued_heading(c)
        return 0.0 if heading is None else self._heading_height(self.blocks[heading].text + _CONTINUED)

    def column_height(self, c: int) -> float:
        return sum(self.estimate(i) for i in self.groups[c]) + self._continuation_height(c)

    def column_heights(self) -> list[float]:
        return [self.column_height(c) for c in range(len(self.groups))]

    def planned_gap(self) -> float:
        """How far apart the planned columns end, added space included."""
        ends = [self.column_height(c) + sum(self.space[i] for i in g) for c, g in enumerate(self.groups)]
        return max(ends) - min(ends) if ends else 0.0

    def rescale(self, measured: list[float]) -> None:
        """Correct each placed block's estimate by how far its column's
        estimate was from the measured column (less the space the plan
        added between its blocks)."""
        for c, height in enumerate(measured[:len(self.groups)]):
            content = height - sum(self.space[i] for i in self.groups[c]) - self._continuation_height(c)
            estimated = sum(self.estimate(i) for i in self.groups[c])
            if self.groups[c] and estimated > 0 and content > 0:
                for i in self.groups[c]:
                    self.factors[i] *= content / estimated

    def signature(self) -> tuple:
        return (
            tuple(tuple(g) for g in self.groups),
            tuple(self.widths),
            tuple(round(s) for s in self.space),
            tuple((b.kind, b.text, tuple(b.items)) for b in self.blocks),
        )

    # -- planning ----------------------------------------------------------

    def _split(self, units: list[list[int]]) -> list[list[int]]:
        heights = [sum(self.estimate(i) for i in unit) for unit in units]
        return [
            [i for unit in units[start:end] for i in unit]
            for start, end in _partition(heights, self.sheet.columns)
        ]

    def plan(self, available: float, *, cut: bool = True) -> None:
        """Place the blocks for columns of ``available`` points.

        Both whole sections and heading-and-block units are tried, with
        figures narrowed only in a column that is over. Whole sections win
        unless splitting them evens the columns out by a clear margin.
        Content is cut only when neither fits and ``cut`` allows it. Short
        columns are then carried down with extra space."""
        self.available = available
        limit = self.room * available
        margin = 2 * _BALANCE_CM * _PT_PER_CM
        while True:
            options = []
            for units in (_sections(self.blocks), _units(self.blocks)):
                self.widths = [1.0] * len(self.blocks)
                self.groups = self._split(units)
                for c in range(len(self.groups)):
                    while self.column_height(c) > limit and self._narrow_figures(c):
                        pass
                heights = self.column_heights() or [0.0]
                options.append((max(heights) <= limit, max(heights) - min(heights), self.groups, self.widths))
            (sections_fit, sections_gap, *sections), (units_fit, units_gap, *units) = options
            if sections_fit and (not units_fit or sections_gap <= units_gap + margin):
                self.groups, self.widths = sections
                break
            if units_fit or not cut or not self._cut():
                self.groups, self.widths = units
                break
        # A first plan is made for a guessed room and never cuts, so it can be
        # over its own limit; the fit loop then plans again for the measured room.
        self.over_limit = max(self.column_heights() or [0.0]) > limit
        self._spread(limit)

    def _narrow_figures(self, c: int) -> bool:
        figures = [
            i for i in self.groups[c]
            if self.blocks[i].kind == "figure" and self.widths[i] - _FIGURE_STEP >= _MIN_FIGURE_WIDTH - 1e-9
        ]
        for i in figures:
            self.widths[i] = round(self.widths[i] - _FIGURE_STEP, 2)
        return bool(figures)

    def _spread(self, limit: float) -> None:
        """Carry each column down toward the band with extra space, first
        before its headings and then between the blocks of a section. The
        space is capped, so a short column keeps its sections together."""
        scale = self.sheet.body_pt / 26
        self.space = [0.0] * len(self.blocks)
        for c, group in enumerate(self.groups):
            slack = limit - self.column_height(c)
            before_headings = [i for i in group[1:] if self.blocks[i].kind == "heading"]
            within = [
                i for prev, i in zip(group, group[1:])
                if self.blocks[i].kind != "heading" and self.blocks[prev].kind != "heading"
            ]
            for gaps, cap_cm in ((before_headings, 4.0), (within, 2.0)):
                if slack <= 0 or not gaps:
                    continue
                each = min(cap_cm * scale * _PT_PER_CM, slack / len(gaps))
                for i in gaps:
                    self.space[i] += each
                slack -= each * len(gaps)

    def _cut(self) -> bool:
        return (
            self._trim_list()
            or self._trim_text()
            or self._drop_middle(("text", "bullets", "raw"))
            or self._drop_middle(("figure",))
        )

    def _trim_list(self) -> bool:
        lists = [i for i, b in enumerate(self.blocks) if b.kind == "bullets" and len(b.items) > 2]
        if not lists:
            return False
        i = max(lists, key=self.estimate)
        self.blocks[i] = replace(self.blocks[i], items=self.blocks[i].items[:-1])
        self.trimmed += 1
        return True

    def _trim_text(self) -> bool:
        texts = [i for i, b in enumerate(self.blocks) if b.kind == "text" and len(_sentences(b.text)) > 1]
        if not texts:
            return False
        i = max(texts, key=self.estimate)
        self.blocks[i] = replace(self.blocks[i], text=" ".join(_sentences(self.blocks[i].text)[:-1]))
        self.trimmed += 1
        return True

    def _drop_middle(self, kinds: tuple[str, ...]) -> bool:
        """Drop the tallest block of ``kinds`` that is neither in the opening
        pair nor the closing pair; a heading left with nothing under it
        goes too."""
        candidates = [i for i in range(2, len(self.blocks) - 2) if self.blocks[i].kind in kinds]
        if not candidates:
            return False
        self._delete(max(candidates, key=self.estimate))
        j = 0
        while j < len(self.blocks):
            following = self.blocks[j + 1].kind if j + 1 < len(self.blocks) else None
            if self.blocks[j].kind == "heading" and following in (None, "heading"):
                self._delete(j)
            else:
                j += 1
        self.dropped += 1
        return True

    def _delete(self, i: int) -> None:
        del self.blocks[i]
        del self.factors[i]
        del self.widths[i]

    # -- LaTeX -------------------------------------------------------------

    def _block_latex(self, i: int, number: int | None) -> str:
        space = r"\vspace{" + f"{self.space[i]:.1f}" + "pt}\n" if self.space[i] >= 1 else ""
        return space + self._block_body(i, number)

    def _block_body(self, i: int, number: int | None) -> str:
        block = self.blocks[i]
        if block.kind == "heading":
            return r"\posterhead{" + _inline_latex(block.text) + "}"
        if block.kind == "text":
            return _inline_latex(block.text) + r"\par\vspace{0.7em}"
        if block.kind == "bullets":
            items = "\n".join(r"\item{} " + _inline_latex(item) for item in block.items)
            return r"\begin{posterlist}" + "\n" + items + "\n" + r"\end{posterlist}\par\vspace{0.7em}"
        if block.kind == "figure":
            path = self.figure_paths.get(block.file, "figures/" + block.file)
            return (
                r"{\centering\includegraphics[width=" + f"{self.widths[i]:g}"
                + r"\linewidth,height=0.4\textheight,keepaspectratio]{" + path + r"}\par}"
                + r"\vspace{0.35em}" + "\n"
                + r"\postercaption{" + str(number) + "}{" + _inline_latex(block.caption) + r"}\vspace{0.9em}"
            )
        return _escape_latex_text_specials(block.text) + r"\par\vspace{0.7em}"

    def columns_latex(self) -> str:
        numbers: dict[int, int] = {}
        for i, block in enumerate(self.blocks):
            if block.kind == "figure":
                numbers[i] = len(numbers) + 1
        columns = []
        for c, group in enumerate(self.groups):
            body = "\n".join(self._block_latex(i, numbers.get(i)) for i in group)
            heading = self.continued_heading(c)
            if heading is not None:
                body = r"\posterhead{" + _inline_latex(self.blocks[heading].text + _CONTINUED) + "}\n" + body
            columns.append(
                r"\begin{column}{" + f"{self.sheet.column_cm:.2f}" + "cm}\n"
                + r"\begin{minipage}[t][\dimexpr\textheight-1cm\relax][t]{\linewidth}" + "\n"
                + r"\posterbody" + "\n" + body + "\n" + r"\vfill" + "\n"
                + r"\end{minipage}" + "\n" + r"\end{column}"
            )
        return r"\begin{columns}[T,totalwidth=\textwidth]" + "\n" + "\n".join(columns) + "\n" + r"\end{columns}"


def _measured_columns(doc, report: dict, sheet: _Sheet) -> tuple[list[float] | None, float]:
    """Each column's measured content height and the room one column has,
    in points. Reference-band text is left out; column text that ran into
    the band or off the sheet counts."""
    metrics = report["metrics"]
    page = doc.pages[0]
    if metrics.get("header_bottom_cm") is None:
        return None, 0.0
    top = page.height - metrics["header_bottom_cm"] * _PT_PER_CM
    band = metrics.get("band_top_cm")
    floor = page.height - band * _PT_PER_CM if band is not None else sheet.margin_cm * _PT_PER_CM
    body_pt = metrics.get("body_pt") or sheet.body_pt
    refs_pt = metrics.get("references_pt") or sheet.refs_pt

    def in_reference_list(line) -> bool:
        return (
            0 <= (line.box[1] + line.box[3]) / 2 <= floor
            and line.size < 0.95 * body_pt and line.size <= 1.1 * refs_pt
        )

    boxes = [line.box for line in page.lines if line.visible and not in_reference_list(line)]
    boxes += [image for image in page.images if image[3] > floor]
    left = sheet.margin_cm * _PT_PER_CM
    pitch = (sheet.column_cm + sheet.gap_cm) * _PT_PER_CM
    bottoms = [top] * sheet.columns
    for box in boxes:
        if (box[1] + box[3]) / 2 >= top:
            continue
        k = min(sheet.columns - 1, max(0, int(((box[0] + box[2]) / 2 - left) / pitch)))
        bottoms[k] = min(bottoms[k], box[1])
    gap = 1.2 * _PT_PER_CM
    return [max(0.0, top - gap - b) for b in bottoms], max(0.0, top - gap - floor)


# ---------------------------------------------------------------------------
# Files


# LaTeX scratch pdflatex/tectonic leave next to poster.pdf — pure compile
# byproducts. We delete them once we have the PDF so the quest dir holds the
# deliverables (poster.pdf + poster.tex), not a pile of poster.aux/.out/…
_POSTER_AUX_EXTS = (
    ".aux", ".out", ".nav", ".snm", ".toc", ".vrb",
    ".fls", ".fdb_latexmk", ".synctex.gz",
)


def _cleanup_poster_artifacts(out_dir: Path, *, keep_log: bool) -> None:
    """Delete the LaTeX scratch files beside poster.pdf, keeping the source
    (poster.tex) and — on a failed compile (``keep_log=True``) — poster.log
    for debugging. Best-effort: a locked file is skipped."""
    exts = list(_POSTER_AUX_EXTS)
    if not keep_log:
        exts.append(".log")
    for ext in exts:
        p = out_dir / f"poster{ext}"
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _copy_poster_icon(out_dir: Path) -> bool:
    """Copy the FI teal glyph into ``out_dir`` as ``fi_icon.png``. Returns
    ``False`` (no brand mark) if the asset is missing or can't be copied —
    the poster still compiles either way."""
    if not ICON_PNG.is_file():
        return False
    try:
        shutil.copyfile(ICON_PNG, out_dir / "fi_icon.png")
        return True
    except OSError:
        return False


def _figure_path(figures_dir: Path | None, name: str, out_dir: Path) -> str:
    if figures_dir is None:
        return "figures/" + name
    target = figures_dir / name
    try:
        return Path(os.path.relpath(target, out_dir)).as_posix()
    except ValueError:  # another drive on Windows
        return target.as_posix()


_CJK_HOW_TO_FIX = (
    "pdflatex cannot set Chinese, Japanese or Korean characters; FI "
    "switches to XeLaTeX with a CJK font when both are installed. XeLaTeX "
    "comes with MiKTeX and TeX Live. Fonts: Windows ships Microsoft "
    "JhengHei, YaHei, Yu Gothic and Malgun Gothic; on Linux install Noto "
    "CJK (`sudo apt install fonts-noto-cjk`)."
)


class PosterGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None = None,
        feedback: str = "",
    ) -> dict[str, Path]:
        """Write, lay out and compile the poster. ``feedback`` is what a visual
        check of the previous version found; it is appended to the prompt."""
        if "poster" not in self.config.output.kinds or art.paper_md is None:
            return {}
        output = self.config.output
        sheet = _SHEETS.get(output.poster_size, _SHEETS["a1_portrait"])

        paper_md = art.paper_md.read_text(encoding="utf-8")
        figures = _figure_files(art.figures_dir)
        captions = _paper_captions(paper_md)
        literature = art.raw_state.get("literature") or []
        sources: dict[str, dict] = {
            str(r["n"]): r for r in build_references(literature, audience=output.audience)
        }
        sources.update({
            str(w["label"]): w for w in build_further_reading(literature, audience=output.audience)
        })

        # safe_substitute: paper_md is model-written prose that may carry
        # stray dollar signs, which substitute() would reject.
        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).safe_substitute(
            paper_md=paper_md[:8000],
            figure_list="\n".join(
                f"- figures/{name}" + (f": {captions[name]}" if captions.get(name) else "")
                for name in figures
            ) or "(none)",
            source_list="\n".join(_band_entry(src, label) for label, src in sources.items()) or "(none)",
            sheet=sheet.label,
            columns=str(sheet.columns),
            word_budget=str(sheet.words),
        )
        if feedback:
            prompt = prompt.rstrip() + "\n\n" + feedback.strip() + "\n"
        own_supervisor = supervisor is None
        sup = supervisor or ProxySupervisor()
        endpoint = await resolve_endpoint_async(self.config.provider, sup)
        client = LLMClient(endpoint)
        try:
            text = await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                model=model_for_node(self.config.provider.node_models, "poster"),
                node="poster",
            )
        finally:
            await client.aclose()
            if self.config.provider.name in PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()
        self._write_reply(out_dir, text)
        parsed = _lenient_json(text) or {}

        # The headline is the finding; the paper's own H1 goes under it. A
        # headline that is missing or runs to a paragraph gives way to the
        # H1, and the model's paraphrased ``title`` only names a paper that
        # has no H1.
        from generation.paper import _FIRST_H1_RE, _FRONTMATTER_RE

        h1 = _FIRST_H1_RE.search(_FRONTMATTER_RE.sub("", paper_md, count=1))
        paper_title = h1.group(1).strip() if h1 else ""
        headline = _plain(parsed.get("headline"))
        if not headline or len(headline.split()) > _MAX_HEADLINE_WORDS:
            headline = paper_title or _plain(parsed.get("title")) or "Untitled"
        subtitle = paper_title if paper_title and paper_title != headline else ""

        blocks = _with_every_figure(_blocks_from_reply(parsed, set(figures)), figures, captions)
        cited = _cited_labels(blocks, set(sources))
        blocks = [_with_citations(b, set(cited)) for b in blocks]
        if cited:
            band_heading, entries = "References", [_band_entry(sources[label], label) for label in cited]
        else:
            picks = list(sources.items())[:_SELECTED_SOURCES]
            band_heading, entries = "Selected sources", [_band_entry(src, label) for label, src in picks]

        author_terms = tuple(t for t in (output.author, output.affiliation, output.contact_email) if t)
        engine = find_pdf_engine()
        content = "\n".join([headline, subtitle, *author_terms, *entries, *(t for b in blocks for t in b.texts())])
        cjk_font: str | None = None
        cjk_problem: str | None = None
        if engine is not None and _cjk.has_cjk(content):
            xelatex = _cjk.find_xelatex(engine)
            cjk_font = _cjk.find_cjk_font(content) if xelatex is not None else None
            if xelatex is None:
                cjk_problem = "cjk_no_xelatex"
            elif cjk_font is None:
                cjk_problem = "cjk_no_font"
            else:
                engine = xelatex

        icon_ok = _copy_poster_icon(out_dir)
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        values = self._template_values(
            sheet, headline=headline, subtitle=subtitle, author_terms=author_terms,
            url=output.url, band=_band_latex(entries, band_heading, sheet.columns, icon_ok),
            icon_ok=icon_ok, cjk_font=cjk_font,
        )
        layout = _Layout(
            sheet, blocks,
            {name: _aspect(art.figures_dir, name) for name in figures},
            {name: _figure_path(art.figures_dir, name, out_dir) for name in figures},
        )
        poster_tex = out_dir / "poster.tex"
        out_pdf = out_dir / "poster.pdf"
        diag_path = out_dir / "poster_pdf_skipped.md"
        result: dict[str, Path] = {"poster_tex": poster_tex}

        def write_tex() -> None:
            body = string.Template(template).safe_substitute(values, columns=layout.columns_latex())
            poster_tex.write_text(_latex_safe_body(body), encoding="utf-8")

        write_tex()
        if engine is None or cjk_problem:
            if engine is None:
                code = "no_latex_engine"
                summary = (
                    "no LaTeX engine found (pdflatex or tectonic); poster.pdf "
                    "skipped. Run `python launch.py --install-tectonic` for a "
                    "no-admin LaTeX install."
                )
                how_to_fix = (
                    "Easiest no-admin path: run `python launch.py "
                    "--install-tectonic` from the repo root. That drops a "
                    "single self-bootstrapping LaTeX binary (~70 MB) into "
                    "`tools/`; FI auto-detects it on the next quest. "
                    "Standard alternative: install MiKTeX (Windows) or "
                    "TeX Live (macOS/Linux) so `pdflatex` lands on PATH."
                )
            else:
                missing = "XeLaTeX" if cjk_problem == "cjk_no_xelatex" else "a font for it"
                code = cjk_problem
                summary = (
                    f"the poster has Chinese, Japanese or Korean text and {missing} "
                    "was not found; poster.pdf skipped."
                )
                how_to_fix = _CJK_HOW_TO_FIX
            _log.warning(summary)
            diag_path.write_text(
                _render_poster_skip_md(code=code, summary=summary, how_to_fix=how_to_fix),
                encoding="utf-8",
            )
            result["poster_pdf_skipped"] = diag_path
            return result

        engine_name = engine[0]
        _log.info("poster.pdf: %s sheet, engine=%s at %s", output.poster_size, engine_name, engine[1])
        compiles = 0
        report: dict | None = None
        while True:
            if compiles:
                write_tex()
            compiles += 1
            failure = self._compile(engine, poster_tex, out_dir, out_pdf)
            if failure is not None:
                code, summary, how_to_fix = failure
                _log.warning(summary)
                diag_path.write_text(
                    _render_poster_skip_md(code=code, summary=summary, how_to_fix=how_to_fix),
                    encoding="utf-8",
                )
                result["poster_pdf_skipped"] = diag_path
                _cleanup_poster_artifacts(out_dir, keep_log=True)
                return result
            doc = measure_pdf(out_pdf)
            if doc is None:
                break  # nothing to measure; keep the first compile
            report = poster_report(doc, columns=sheet.columns, header_terms=author_terms)
            checks = {f["check"] for f in report["findings"]}
            spilled = bool(checks & {"overflow", "band_overlap"})
            # A plan over its own limit fit only because the sheet had more
            # room than guessed, and it may have split a section to get close.
            over = layout.over_limit
            if compiles >= _MAX_COMPILES or not (spilled or over or checks & {"column_balance", "empty_space"}):
                break
            heights, available = _measured_columns(doc, report, sheet)
            if heights is None:
                break
            # The sheet measured more room than the plan assumed: plan again
            # for the real room, so figures take their full width and the
            # columns reach the band.
            more_room = (
                not spilled and "empty_space" in checks
                and available - layout.available > _MORE_ROOM_CM * _PT_PER_CM
            )
            if not (spilled or more_room or over or "column_balance" in checks):
                break
            candidate = copy.deepcopy(layout)
            candidate.rescale(heights)
            candidate.plan(available)
            if spilled and candidate.signature() == layout.signature():
                # The corrected estimates still say it fits; plan tighter.
                candidate.room = max(0.8, candidate.room - 0.05)
                candidate.plan(available)
            if candidate.signature() == layout.signature():
                break
            if not (spilled or more_room or over) and candidate.planned_gap() > max(heights) - min(heights) - 2 * _PT_PER_CM:
                break  # another compile would not even the columns out noticeably
            layout = candidate
            reasons = sorted(checks & {"overflow", "band_overlap", "column_balance", "empty_space"})
            _log.info(
                "poster.pdf: compile %d measured %s; re-planned (figure widths %s, %d cut, %d dropped)",
                compiles, ", ".join(reasons + (["a first plan over its limit"] if over else [])),
                [w for w, b in zip(layout.widths, layout.blocks) if b.kind == "figure"],
                layout.trimmed, layout.dropped,
            )

        result["poster_pdf"] = out_pdf
        self._write_fit_report(out_dir, {
            "sheet": output.poster_size,
            "engine": engine_name,
            "compiles": compiles,
            "figure_widths": [w for w, b in zip(layout.widths, layout.blocks) if b.kind == "figure"],
            "trimmed": layout.trimmed,
            "dropped_blocks": layout.dropped,
            "metrics": report["metrics"] if report else None,
            "findings": report["findings"] if report else None,
        })
        if report and report["findings"]:
            _log.info(
                "poster.pdf: after %d compile(s) the measurements still flag: %s",
                compiles, ", ".join(sorted({f["check"] for f in report["findings"]})),
            )
        # Success — remove any stale skip diagnostic from a prior failed run.
        if diag_path.is_file():
            try:
                diag_path.unlink()
            except OSError:
                pass
        _cleanup_poster_artifacts(out_dir, keep_log=False)
        return result

    @staticmethod
    def _template_values(
        sheet: _Sheet, *, headline: str, subtitle: str, author_terms: tuple[str, ...],
        url: str, band: str, icon_ok: bool, cjk_font: str | None,
    ) -> dict[str, str]:
        brand_pt = 0.8 * sheet.author_pt
        brandline = (
            (
                r"\raisebox{-0.3\height}{\includegraphics[height=" + f"{1.4 * brand_pt / 22:.2f}"
                + r"cm]{fi_icon.png}}\hspace{0.4em}" if icon_ok else ""
            )
            + r"{\ttfamily\bfseries\color{fiteal}\fontsize{" + f"{brand_pt:g}pt" + "}{" + f"{1.2 * brand_pt:g}pt"
            + r"}\selectfont FRONTIER INSIGHT}\par\vspace{0.7cm}"
        )
        subtitle_tex = (
            r"\vspace{0.5cm}{\fontsize{" + f"{sheet.subtitle_pt:g}pt" + "}{" + f"{1.2 * sheet.subtitle_pt:g}pt"
            + r"}\selectfont\color{fimuted}" + _inline_latex(subtitle) + r"\par}"
        ) if subtitle else ""
        author_tex = (
            r"\vspace{0.5cm}{\fontsize{" + f"{sheet.author_pt:g}pt" + "}{" + f"{1.2 * sheet.author_pt:g}pt"
            + r"}\selectfont\color{fiink}" + r"~\textperiodcentered~".join(_escape_text(t) for t in author_terms)
            + r"\par}"
        ) if author_terms else ""
        return {
            "paperwidth": f"{sheet.width_cm:g}",
            "paperheight": f"{sheet.height_cm:g}",
            "margin": f"{sheet.margin_cm:g}",
            "cjksetup": _cjk.xecjk_preamble(cjk_font) if cjk_font else "",
            "bodypt": f"{sheet.body_pt:g}", "bodylead": f"{1.25 * sheet.body_pt:g}",
            "headpt": f"{sheet.head_pt:g}", "headlead": f"{1.15 * sheet.head_pt:g}",
            "titlept": f"{sheet.title_pt:g}", "titlelead": f"{1.08 * sheet.title_pt:g}",
            "captionpt": f"{sheet.caption_pt:g}", "captionlead": f"{1.25 * sheet.caption_pt:g}",
            "refspt": f"{sheet.refs_pt:g}", "refslead": f"{1.2 * sheet.refs_pt:g}",
            "rulewidth": f"{sheet.width_cm - 2 * sheet.margin_cm:g}",
            # The QR code takes its width plus a gap from the header text.
            "headerwidth": f"{sheet.width_cm - 2 * sheet.margin_cm - (sheet.qr_cm + 1.5 if url else 0):g}",
            "qrcode": _qr_latex(url, sheet),
            "brandline": brandline,
            "headline": _inline_latex(headline),
            "subtitle": subtitle_tex,
            "authorline": author_tex,
            "references": band,
        }

    def _compile(
        self, engine: tuple[str, str], poster_tex: Path, out_dir: Path, out_pdf: Path,
    ) -> tuple[str, str, str] | None:
        """One compile. ``None`` on success, else ``(code, summary, how_to_fix)``.
        tectonic accepts the same ``.tex`` and flags as pdflatex."""
        engine_name, engine_path = engine
        try:
            r = subprocess.run(
                [engine_path, "-interaction=nonstopmode", "-halt-on-error", str(poster_tex)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(out_dir), timeout=180,
            )
        except subprocess.TimeoutExpired:
            return (
                f"{engine_name}_timeout",
                f"{engine_name} poster timeout (>180s); skipped",
                f"The {engine_name} compile took longer than 180s. On a fresh "
                f"tectonic install the first run downloads CTAN packages (~30 s); "
                f"retry once the cache is populated. If it consistently times out, "
                f"raise the timeout in `generation/poster.py`.",
            )
        if r.returncode != 0:
            stdout = (r.stdout or "")[-500:]
            _log.warning("%s poster rc=%s stdout=%s stderr=%s", engine_name, r.returncode, stdout, (r.stderr or "")[-200:])
            return (
                f"{engine_name}_rc_{r.returncode}",
                f"{engine_name} poster rc={r.returncode}; poster.pdf skipped",
                f"The LaTeX engine errored on `poster.tex`. The most common cause "
                f"is a missing `beamerposter` or `qrcode` package on a fresh "
                f"MiKTeX/TeX Live install; tectonic auto-fetches them on first "
                f"compile. stdout tail (last 500 chars):\n\n```\n{stdout}\n```",
            )
        if not out_pdf.exists():
            return (
                "output_missing_after_success",
                f"{engine_name} returned rc=0 but `poster.pdf` is not on disk in "
                f"{out_dir}. The subprocess reported success but produced no "
                f"output file.",
                f"Most likely a filesystem-level issue: ``{out_dir}`` may not be "
                f"writable by the FI process, or an antivirus/sync tool deleted "
                f"the file between subprocess exit and our existence check. "
                f"Verify the directory is writable, re-run the quest, and if the "
                f"problem persists try running `{engine_name} poster.tex` "
                f"manually from {out_dir} to isolate whether it's an engine bug "
                f"or an environment one.",
            )
        return None

    @staticmethod
    def _write_fit_report(out_dir: Path, fit: dict) -> None:
        path = out_dir / ".fi" / "poster_fit.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fit, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            _log.info("poster.pdf: could not write %s (%r)", path, exc)

    @staticmethod
    def _write_reply(out_dir: Path, text: str) -> None:
        """Keep the model's reply next to the fit report, so a layout can be
        replayed without another model call."""
        path = out_dir / ".fi" / "poster_reply.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text or "", encoding="utf-8")
        except OSError as exc:
            _log.info("poster.pdf: could not write %s (%r)", path, exc)


def _render_poster_skip_md(*, code: str, summary: str, how_to_fix: str) -> str:
    """Render a ``poster_pdf_skipped.md`` diagnostic. Mirrors the shape
    of ``paper_pdf_skipped.md`` (see ``generation/paper.py``) so a user
    sees the same structure for both skip surfaces — reason code, what
    happened, how to fix it."""
    return (
        f"# poster.pdf was requested but not produced\n\n"
        f"Your `output.kinds` included `poster`, but the poster generator "
        f"couldn't compile a PDF. The `poster.tex` source is still on disk "
        f"next to this file — compile it manually once the prerequisites "
        f"below are in place.\n\n"
        f"## What happened\n\n"
        f"**Reason code:** `{code}`\n\n"
        f"{summary}\n\n"
        f"## How to fix it\n\n"
        f"{how_to_fix}\n\n"
        f"This file is auto-deleted on the next successful poster.pdf compile.\n"
    )


def _double_latex_backslashes(s: str) -> str:
    """Double the lone backslashes that start LaTeX inside a JSON reply.

    A model often leaves LaTeX backslashes single inside its JSON strings.
    ``\\textbf`` then parses as a tab plus "extbf", ``\\frac`` as a form feed,
    and ``\\item`` or ``\\%`` is an invalid escape that fails the whole reply.
    Runs of an even length are already escaped and left alone."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        j = i
        while j < len(s) and s[j] == "\\":
            j += 1
        out.append(s[i:j])
        if (j - i) % 2 == 1 and _latex_not_json_escape(s, j):
            out.append("\\")
        i = j
    return "".join(out)


# LaTeX commands that begin with ``n``: anything else after ``\n`` is a newline.
_LATEX_N_COMMANDS = frozenset({
    "nabla", "natural", "ne", "nearrow", "neg", "neq", "newcommand", "newline",
    "newpage", "nexists", "ngeq", "ni", "nleftarrow", "nleq", "nmid",
    "noindent", "nolinebreak", "nonumber", "normalsize", "not", "notin",
    "nparallel", "nrightarrow", "nsubseteq", "nu", "nwarrow",
})


def _latex_not_json_escape(s: str, k: int) -> bool:
    """Whether the character after a lone backslash starts LaTeX rather than a
    JSON escape: ``\\"`` and ``\\/`` keep their JSON meaning, as do ``\\uXXXX``,
    ``\\n`` before ordinary text and a one-letter ``\\b \\f \\r \\t``."""
    if k >= len(s) or s[k] in "\"/":
        return False
    if not ("a" <= s[k].lower() <= "z"):
        return True
    word = _re.match(r"[A-Za-z]+", s[k:]).group(0)
    if word[0] == "u":
        return _re.match(r"u[0-9a-fA-F]{4}", s[k:]) is None
    if word[0] == "n":
        return word in _LATEX_N_COMMANDS
    if word[0] in "bfrt":
        return len(word) > 1 and word[1].islower()
    return True


def _lenient_json(text: str) -> dict | None:
    s = text.strip()
    # Strip fence if present.
    if s.startswith("```"):
        nl = s.find("\n")
        if nl > 0 and s.endswith("```"):
            s = s[nl + 1 : -3].strip()
    s = _double_latex_backslashes(s)
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None
