"""What a saved figure draws -- its panel layout and its line styles -- stated by FI from its own records.

A writer who describes a figure in words gets the layout wrong ("the upper panel" for two panels side by side, "dotted
baseline curves" in a figure that draws none) because it has not seen the picture. A check that reads the caption's words
afterwards needs a new exception for every new wording. The root fix is the one ``{{seed}}`` and ``{{run_time}}`` use: the
fact comes from FI's records, not from the writer.

The records are the ones the plot-style recorder (``core/plot_style.py::_fi_recording_savefig``) leaves next to each
figure in ``.fi/figure_records/``: ``<stem>.json`` (each panel's place on the figure and its axis labels) and
``<stem>.seed0.json`` (each panel's lines with their line style, marker and label). This module turns them into two plain
sentences per figure, which the paper gets through placeholders (:mod:`core.paper_facts`):

* ``{{figure:<stem>:layout}}`` -- "two panels side by side (left: "A"; right: "B")".
* ``{{figure:<stem>:lines}}`` -- "solid lines and markers are drawn; no dotted or dashed line is drawn".

There is no check of a caption's own words against these records: a live run showed it misreading a negated sentence
("no dotted line is drawn") as naming a dotted line, and a position word can be negated the same way ("not in the upper
panel"). The fix is that the writer need not type these facts at all.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import math_safe_cut as _math_safe

#: A figure of the run in a paper, its caption and any attribute block pandoc reads after it (``{width=...}``). FI places
#: these and writes their captions from its records (``core/engine.py:_place_figures``), so a check of the writer's own
#: words leaves them out. The caption may hold any bracket; it never holds ``](``, which ends it.
RUN_FIGURE_RE = re.compile(r"!\[(?:(?!\]\()[^\n])*\]\((?:\./)?figures/[^)\s]+\)(?:\{[^}\n]*\})?")

#: Where the recorder leaves its files, under the quest's ``.fi`` folder (= ``core.plot_style.RECORDS_DIRNAME``).
RECORDS_DIR = Path(".fi") / "figure_records"
KINDS = ("layout", "lines")

#: matplotlib's line styles as a reader names them.
_STYLE_WORDS = {"-": "solid", "solid": "solid", "--": "dashed", "dashed": "dashed", ":": "dotted", "dotted": "dotted",
                "-.": "dash-dotted", "dashdot": "dash-dotted"}
_NO_LINE = ("none", "", " ")
_NO_MARKER = ("none", "", " ")
#: Two panel boxes (figure coordinates) closer than this are one panel drawn twice (twin axes).
_SAME = 1e-3
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _one(text: Any, limit: int = 50) -> str:
    out = " ".join(str(text or "").split())
    if len(out) <= limit:
        return out
    return _math_safe.head_before_math(out, limit - 3).rstrip() + "..."


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """The part of the shorter of two intervals that the other covers, 0 to 1."""
    shorter = min(a1 - a0, b1 - b0)
    if shorter <= 0:
        return 0.0
    return max(0.0, min(a1, b1) - max(a0, b0)) / shorter


def blank_panels(record: Any, lines_data: Any) -> set[int]:
    """The positions of the panels of ``record`` that hold nothing: no line, no shape, no legend and no text on them in
    the line file (``lines_data``). Such a panel is a place the figure was made with and left unused (a 2 by 2 grid with
    three plots), so it is not counted as a panel the figure shows. Empty when the line file is missing or does not match."""
    shown = record.get("axes") if isinstance(record, dict) else None
    drawn = lines_data.get("axes") if isinstance(lines_data, dict) else None
    if not isinstance(shown, list) or not isinstance(drawn, list) or len(shown) != len(drawn):
        return set()
    return {i for i, (a, b) in enumerate(zip(shown, drawn))
            if isinstance(a, dict) and isinstance(b, dict) and not b.get("lines") and not b.get("draws_other")
            and not b.get("legend") and not b.get("has_text") and not a.get("series")}


def panel_draws_nothing(ax: Any) -> bool:
    """Whether FI's record of one panel says it shows no data: blank (no line, shape, legend or text on it), hidden, a
    colorbar's axes, or every line it was given has no point on it. A panel whose record lists no series (an image, a
    histogram) is not known to be empty, so it is not. The one rule for both the caption's description and its layout
    sentence, so they cannot disagree on which panels exist."""
    if not isinstance(ax, dict):
        return False
    if ax.get("blank") or ax.get("hidden") or ax.get("colorbar"):
        return True
    series = [s for s in ax.get("series") or [] if isinstance(s, dict)]
    return bool(series) and not ax.get("unlabelled_lines") and all(s.get("shows") == "no points" for s in series)


def _panel_words(ax: Any) -> str:
    """How a caption names one panel, from the saved axes: its title, in quotes, when it has one; otherwise what its axes
    say, worded as axis labels (``y axis "..." , x axis "..."``). An axis label is never presented as a title."""
    title = _one(ax.get("title")) if isinstance(ax, dict) else ""
    if title:
        return f'"{title}"'
    parts = [f'{side} axis "{text}"' for side, text in (("y", _one(ax.get("ylabel"))), ("x", _one(ax.get("xlabel")))) if text]
    return ", ".join(parts)


def _boxes(record: Any, skip: set[int] | None = None) -> list[dict[str, Any]]:
    """The distinct panels of a figure record: ``place`` (x, y, width, height in figure coordinates, y up) and a name.
    A twin axis shares its panel's box and is one panel; a small box inside another (an inset) is not a panel."""
    panels: list[dict[str, Any]] = []
    for k, ax in enumerate((record.get("axes") if isinstance(record, dict) else None) or []):
        if skip and k in skip:
            continue
        place = ax.get("place") if isinstance(ax, dict) else None
        if not (isinstance(place, list) and len(place) == 4 and all(_num(v) for v in place)):
            continue
        if any(all(abs(a - b) < _SAME for a, b in zip(place, p["place"])) for p in panels):
            continue
        panels.append({"place": [float(v) for v in place], "name": _panel_words(ax)})
    insets = []
    for p in panels:
        x, y, w, h = p["place"]
        for q in panels:
            if q is p:
                continue
            qx, qy, qw, qh = q["place"]
            if qw * qh > w * h and qx - _SAME <= x and qy - _SAME <= y and x + w <= qx + qw + _SAME \
                    and y + h <= qy + qh + _SAME:
                insets.append(p)
                break
    return [p for p in panels if p not in insets]


def _skipped_panels(record: Any, lines_data: Any) -> set[int]:
    """The positions of the panels the layout does not count: those of :func:`blank_panels` and those
    :func:`panel_draws_nothing` says show no data."""
    axes = record.get("axes") if isinstance(record, dict) else None
    empty = {i for i, ax in enumerate(axes) if panel_draws_nothing(ax)} if isinstance(axes, list) else set()
    return blank_panels(record, lines_data) | empty


def layout(record: Any, lines_data: Any = None) -> dict[str, Any] | None:
    """``{"panels", "arrangement", "rows", "cols", "names"}`` of a figure record, the panels in reading order (top row
    first, left to right); ``arrangement`` is ``single``, ``side_by_side`` (one row), ``stacked`` (one column) or
    ``grid``. ``None`` when the record holds no panel boxes (an old record), so nothing is said."""
    panels = _boxes(record, _skipped_panels(record, lines_data))
    if not panels:
        return None
    for p in panels:
        x, y, w, h = p["place"]
        p["x0"], p["x1"], p["y0"], p["y1"] = x, x + w, y, y + h

    def groups(lo: str, hi: str) -> list[list[dict[str, Any]]]:
        out: list[list[dict[str, Any]]] = []
        for p in sorted(panels, key=lambda p: (p[lo] + p[hi])):
            for g in out:
                if _overlap(p[lo], p[hi], g[0][lo], g[0][hi]) > 0.5:
                    g.append(p)
                    break
            else:
                out.append([p])
        return out

    rows, cols = groups("y0", "y1"), groups("x0", "x1")
    n = len(panels)
    arrangement = ("single" if n == 1 else "side_by_side" if len(rows) == 1 else "stacked" if len(cols) == 1
                   else "grid")
    order = sorted(panels, key=lambda p: (-round((p["y0"] + p["y1"]) / 2 * 20), p["x0"]))
    if arrangement == "grid":
        top_first = sorted(rows, key=lambda g: -sum(p["y0"] + p["y1"] for p in g) / len(g))
        order = [p for g in top_first for p in sorted(g, key=lambda p: p["x0"])]
    elif arrangement == "side_by_side":
        order = sorted(panels, key=lambda p: p["x0"])
    elif arrangement == "stacked":
        order = sorted(panels, key=lambda p: -(p["y0"] + p["y1"]))
    return {"panels": n, "arrangement": arrangement, "rows": len(rows), "cols": len(cols),
            "names": [p["name"] for p in order]}


def layout_sentence(record: Any, lines_data: Any = None) -> str:
    """A plain sentence for the figure's layout, or ``""`` when the record holds none. A blank panel is left out of the
    count and said once ("... (one more place on the figure is left blank)")."""
    got = layout(record, lines_data)
    if got is None:
        return ""
    n, names = got["panels"], got["names"]
    count = _NUMBER_WORDS.get(n, str(n))
    blank = len(blank_panels(record, lines_data))
    note = (f"; {_NUMBER_WORDS.get(blank, str(blank))} more {'place' if blank == 1 else 'places'} on the figure "
            f"{'is' if blank == 1 else 'are'} left blank") if blank else ""
    if got["arrangement"] == "single":
        return "a single panel" + (f" ({names[0]})" if names[0] else "") + note
    positions = {
        "side_by_side": ["left", "centre", "right"] if n == 3 else ["left", "right"] if n == 2 else [],
        "stacked": ["top", "middle", "bottom"] if n == 3 else ["top", "bottom"] if n == 2 else [],
    }.get(got["arrangement"], [])
    if got["arrangement"] == "grid":
        head = f"{count} panels in a grid of {got['rows']} rows by {got['cols']} columns"
        tail = "read left to right, top row first"
    elif got["arrangement"] == "side_by_side":
        head = f"{count} panels side by side in one row"
        tail = "left to right"
    else:
        head = f"{count} panels stacked in one column"
        tail = "top to bottom"
    shown = []
    for i, name in enumerate(names):
        if not name:
            continue
        label = positions[i] if i < len(positions) else f"panel {i + 1}"
        shown.append(f"{label}: {name}")
    sentence = f"{head} ({'; '.join(shown)})" if shown else f"{head} ({tail})"
    return sentence + note


def styles_drawn(lines_data: Any) -> dict[str, int]:
    """How many lines of each style (``solid``, ``dashed``, ``dotted``, ``dash-dotted``) a figure's line file holds, and
    ``markers`` for the lines that draw a marker. A line with no style and no marker draws nothing and is not counted; a
    single-point line draws as a marker."""
    out: dict[str, int] = {}
    for ax in (lines_data.get("axes") if isinstance(lines_data, dict) else None) or []:
        for line in (ax.get("lines") if isinstance(ax, dict) else None) or []:
            if not isinstance(line, dict) or line.get("kind") == "other":
                continue
            style = str(line.get("linestyle") or "").lower()
            marked = str(line.get("marker") or "None").lower() not in _NO_MARKER
            if style not in _NO_LINE and style != "none" and len(line.get("x") or [0, 0]) > 1:
                word = _STYLE_WORDS.get(style)
                if word:
                    out[word] = out.get(word, 0) + 1
            if marked:
                out["markers"] = out.get("markers", 0) + 1
    return out


def lines_sentence(lines_data: Any) -> str:
    """A plain sentence for the line styles a figure draws, naming the dashed and dotted styles it does NOT draw (the
    ones a caption most often invents). ``""`` when the line file holds no line."""
    drawn = styles_drawn(lines_data)
    if not drawn:
        return ""
    parts = [f"{w} lines" for w in ("solid", "dashed", "dotted", "dash-dotted") if w in drawn]
    if "markers" in drawn:
        parts.append("markers")
    absent = [w for w in ("dashed", "dotted") if w not in drawn]
    text = " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts) + " are drawn"
    if absent:
        text += f"; no {' or '.join(absent)} line is drawn"
    return text


#: Series told per figure before "and N more".
_MAX_SERIES = 12


def _bars_phrase(bar: Any, span: Any) -> str:
    """The error bars of one series as the recorder measured them: drawn (with their largest length against the axis span)
    or none visible."""
    n, longest = (bar.get("n"), bar.get("max")) if isinstance(bar, dict) else (None, None)
    if not _num(longest) or not _num(n):
        return ""
    if n == 0:
        return "no error bar is drawn"
    if longest <= 0:
        return "error bars are all zero (none visible)"
    share = f", {longest / span * 100:.2g}% of the axis span" if _num(span) and span > 0 else ""
    return f"error bars are drawn (largest {longest:.3g}{share})"


def series_drawn(record_panel: Any, lines_panel: Any, budget: int = _MAX_SERIES) -> tuple[list[str], int, str]:
    """How each named series of one panel is drawn, from the recorder's own line list (``lines_panel``, a panel of the
    ``<stem>.seed0.json`` file) and the panel's series list (``record_panel``, a panel of ``<stem>.json``): markers only, a
    line, a line with markers, a straight line of two points, a legend entry with no data, and its error bars. The texts
    (at most ``budget``) come with the number left unsaid and a closing line on the panel's error bars. Lines without a legend name are not named here; a horizontal or
    vertical reference line is told with the panel's series."""
    texts: list[str] = []
    ylim = record_panel.get("ylim") if isinstance(record_panel, dict) else None
    span = abs(ylim[1] - ylim[0]) if isinstance(ylim, list) and len(ylim) == 2 and all(map(_num, ylim)) else None
    recorded = bool(isinstance(lines_panel, dict) and lines_panel.get("bars_recorded"))
    seen: set[str] = set()
    for line in (lines_panel.get("lines") if isinstance(lines_panel, dict) else None) or []:
        if not isinstance(line, dict) or line.get("kind") != "data" or not isinstance(line.get("x"), list):
            continue
        label = _one(line.get("label"), 60)
        if not label or label.startswith("_"):
            continue
        n = len(line["x"])
        style = str(line.get("linestyle") or "").lower()
        has_line = style not in _NO_LINE and style != "none" and n > 1
        marked = str(line.get("marker") or "None").lower() not in _NO_MARKER
        word = _STYLE_WORDS.get(style, "")
        if n == 0:
            how = "a legend entry with no data"
        elif has_line and n == 2 and not marked:
            how = f"a straight {word + ' ' if word else ''}line of two points, no markers"
        elif has_line and marked:
            how = f"a {word + ' ' if word else ''}line with markers ({n} points)"
        elif has_line:
            how = f"a {word + ' ' if word else ''}line, no markers ({n} points)"
        elif marked:
            how = f"markers only, no line joins them ({n} point{'s' if n != 1 else ''})"
        else:
            how = "nothing drawn (no line, no marker)"
        if n and recorded and "yerr" in line:
            bars = _bars_phrase(line.get("yerr"), span)
            how += f"; {bars}" if bars else ""
        text = f"{label}: {how}"
        if text not in seen:
            seen.add(text)
            texts.append(text)
    named = {t.split(":", 1)[0] for t in texts}
    for s in (record_panel.get("series") if isinstance(record_panel, dict) else None) or []:
        label = _one(s.get("label"), 60) if isinstance(s, dict) else ""
        if label and not label.startswith("_") and label not in named and s.get("shows") == "no points":
            named.add(label)
            texts.append(f"{label}: a legend entry with no data")
    tail = ""
    if texts:
        if not recorded:
            tail = "error bars: not recorded for this figure"
        else:
            measured = sum(1 for ln in lines_panel.get("lines") or [] if isinstance(ln, dict) and "yerr" in ln)
            sets = lines_panel.get("error_bar_sets")
            if not _num(sets):
                tail = "error bars: not recorded for this figure"
            elif sets > measured:
                tail = ("error bars are drawn (length not recorded)" if not measured
                        else "some more error bars are drawn (length not recorded)")
            elif not measured:
                tail = "no error bars are drawn"
    return texts[:budget], max(0, len(texts) - budget), tail


def attach_series_drawn(record: Any, lines_path: Path) -> None:
    """Add to each panel of ``record`` its ``series_drawn`` list (and ``series_drawn_more``, the number not listed), from the
    line file ``lines_path``, for the note the writer is given about the figure. Nothing is added when the line file is
    missing or does not match the panels, and a figure's budget of series is shared by its panels."""
    full = _read(lines_path)
    panels, drawn = (record.get("axes") if isinstance(record, dict) else None),         (full.get("axes") if isinstance(full, dict) else None)
    if not isinstance(panels, list) or not isinstance(drawn, list) or len(panels) != len(drawn):
        return
    left = _MAX_SERIES
    for panel, got in zip(panels, drawn):
        if not isinstance(panel, dict) or not isinstance(got, dict):
            continue
        texts, more, tail = series_drawn(panel, got, left)
        left -= len(texts)
        if texts:
            panel["series_drawn"] = texts
            if tail:
                panel["series_drawn_tail"] = tail
        if more:
            panel["series_drawn_more"] = more


def _stem(name: str) -> str:
    return Path(str(name)).stem


def records_of(quest_root: Path | str) -> dict[str, tuple[dict[str, Any], Any]]:
    """``{stem: (record, line file)}`` for each figure with a record under ``<quest_root>/.fi/figure_records``; the line
    file is ``None`` when the recorder did not leave one."""
    folder = Path(quest_root) / RECORDS_DIR
    out: dict[str, tuple[dict[str, Any], Any]] = {}
    try:
        files = sorted(folder.glob("*.json"))
    except OSError:
        return out
    for path in files:
        if re.search(r"\.seed\d+$", path.stem):
            continue
        data = _read(path)
        if isinstance(data, dict) and isinstance(data.get("axes"), list):
            out[path.stem] = (data, _read(folder / f"{path.stem}.seed0.json"))
    return out


def facts(quest_root: Path | str) -> dict[str, str]:
    """The figure facts FI holds, by ``"<stem>:<kind>"`` (``kind`` is ``layout`` or ``lines``): the text that goes where
    ``{{figure:<stem>:<kind>}}`` is. A fact the records do not hold is not in it."""
    out: dict[str, str] = {}
    for stem, (record, lines_data) in records_of(quest_root).items():
        for kind, text in (("layout", layout_sentence(record, lines_data)), ("lines", lines_sentence(lines_data))):
            if text:
                out[f"{stem}:{kind}"] = text
    return out
