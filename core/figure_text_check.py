"""Does what a figure's legend and caption say agree with FI's record of what the figure draws?

FI writes a figure's legend (``core/frame_figure.py``, ``finish``) and its caption's account of the lines drawn
(``core/engine.py``, ``_caption_from_calls``) from its record of the drawing calls (``.fi/figure_records/<stem>.calls.json``).
This check reads that record beside the legend kept in it and the caption the delivered paper carries, and lists every
place where they disagree: a legend entry that names no drawn line, a drawn line the legend leaves out, a reference line
the caption leaves out or gives another value, a panel that draws nothing the caption does not say so. A finding is
disclosed in QUALITY.md; it never ends the quest. Stdlib only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

#: A legend text that names nothing: the placeholders older figures carried.
PLACEHOLDER = re.compile(r"^\s*(reference line|a curve|curve \d+|computed x)?\s*$", re.IGNORECASE)
_FIGURE_IMAGE = re.compile(r"!\[(?P<caption>(?:[^\[\]]|\[[^\]]*\])*)\]\((?P<path>[^)\s]+)\)")


def _fmt(value: Any) -> str:
    return f"{float(value):.3g}"


def _panels(calls: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in (calls or {}).get("panels") or [] if isinstance(p, dict)]


def _series_key(s: dict[str, Any]) -> tuple[str, str]:
    level = s.get("level") if isinstance(s.get("level"), dict) else {}
    return str(s.get("id")), str(level.get("text") or "")


def _expected_entries(calls: dict[str, Any], legend: dict[str, Any]) -> int:
    """How many entries the legend has when it names each drawn line once: a series once (two panels drawing the same
    series share one entry when the legend is the figure's), a reference line once."""
    panels = _panels(calls)
    if legend.get("scope") == "panel":
        panels = [p for p in panels if p.get("panel") == legend.get("panel")]
    keys: set[tuple[str, ...]] = set()
    count = 0
    for p in panels:
        entries = [s for s in p.get("series") or [] if isinstance(s, dict) and s.get("kind") != "image"]
        absorbed = {str(m) for s in entries for m in s.get("merged_with") or []}  # merged into another's entry
        named: set[tuple[str, str]] = set()  # a level drawn as a line and as points is one entry
        for s in entries:
            if str(s.get("id")) in absorbed:
                continue
            if isinstance(s.get("level"), dict):
                if _series_key(s) in named:
                    continue
                named.add(_series_key(s))
            keys.add(("series",) + _series_key(s))
            count += 1
        for r in p.get("refs") or []:
            if isinstance(r, dict):
                keys.add(("ref", str(r.get("id"))))
                count += 1
    return len(keys) if legend.get("scope") == "figure" else count


def check_figure_text(calls: dict[str, Any], legend_texts: list[str] | None, caption: str) -> list[dict[str, Any]]:
    """Every disagreement between the record ``calls`` and the legend ``legend_texts`` (``None``: the figure has no legend,
    nothing is checked of it) and the ``caption``: ``[{"kind", "panel", "said"}]``, ``kind`` one of ``legend_extra``,
    ``legend_missing``, ``ref_count``, ``ref_value``, ``empty_panel_unsaid``."""
    out: list[dict[str, Any]] = []
    caption = str(caption or "")
    legend = (calls or {}).get("legend") if isinstance((calls or {}).get("legend"), dict) else None
    if legend_texts is not None and legend is not None:
        texts = [str(t) for t in legend_texts]
        for t in texts:
            if PLACEHOLDER.match(t):
                out.append({"kind": "legend_extra", "panel": legend.get("panel"),
                            "said": f"the legend has an entry that names nothing (\"{t.strip()}\")"})
        if len(set(texts)) != len(texts):
            out.append({"kind": "legend_extra", "panel": legend.get("panel"),
                        "said": "the legend names one thing twice"})
        want = _expected_entries(calls, legend)
        covers = legend.get("covers")  # an entry that stands for lines drawn alike under one name counts for each of them
        if isinstance(covers, list) and len(covers) == len(texts) and all(isinstance(n, int) and n >= 1 for n in covers):
            want -= sum(covers) - len(covers)
        if len(texts) > want:
            out.append({"kind": "legend_extra", "panel": legend.get("panel"),
                        "said": f"the legend has {len(texts)} entries for {want} lines drawn"})
        elif len(texts) < want:
            out.append({"kind": "legend_missing", "panel": legend.get("panel"),
                        "said": f"the legend has {len(texts)} entries for {want} lines drawn"})
    # the caption's list of reference lines: "Marked: A at 0.05; B at 3." (a value's own point is followed by a digit)
    marked = re.split(r"\.(?:\s|$)", caption.rsplit("Marked:", 1)[1], maxsplit=1)[0] if "Marked:" in caption else ""
    refs: dict[str, float] = {}
    for p in _panels(calls):
        for r in p.get("refs") or []:
            if isinstance(r, dict) and isinstance(r.get("value"), (int, float)):
                refs.setdefault(f"{r.get('label')}|{_fmt(r['value'])}", float(r["value"]))
                if not re.search(r"\bat %s(?![\d.]*\d)" % re.escape(_fmt(r["value"])), caption):
                    out.append({"kind": "ref_value", "panel": p.get("panel"),
                                "said": f"the caption does not give the reference line {r.get('label')} at {_fmt(r['value'])}"})
        if not p.get("raw") and not p.get("series") and not p.get("refs") and not p.get("bands"):
            if f"panel {p.get('panel')} draws nothing" not in caption.lower():
                out.append({"kind": "empty_panel_unsaid", "panel": p.get("panel"),
                            "said": f"panel {p.get('panel')} draws nothing and the caption does not say so"})
    listed = len(re.findall(r"\bat -?[\d.]+(?:e[-+]?\d+)?", marked))
    if marked and listed != len(refs):
        out.append({"kind": "ref_count", "panel": None,
                    "said": f"the caption lists {listed} reference line(s); the figure draws {len(refs)}"})
    return out


def paper_captions(paper_md: str) -> dict[str, str]:
    """``{figure file name: its caption}`` of the figures a paper places (``![caption](figures/name.png)``)."""
    return {Path(m.group("path")).name: m.group("caption") for m in _FIGURE_IMAGE.finditer(str(paper_md or ""))}


def quest_findings(root: Path) -> list[dict[str, Any]]:
    """The findings of every figure the delivered paper places that has a record of its drawing calls, each with the
    figure's ``file`` and its ``record`` path."""
    root = Path(root)
    paper = root / "paper" / "paper.md"
    if not paper.exists():
        paper = root / "paper.md"
    try:
        captions = paper_captions(paper.read_text(encoding="utf-8"))
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for name, caption in captions.items():
        rec_path = root / ".fi" / "figure_records" / f"{Path(name).stem}.calls.json"
        try:
            calls = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        legend = calls.get("legend") if isinstance(calls.get("legend"), dict) else None
        texts = list(legend.get("entries") or []) if legend else None
        for f in check_figure_text(calls, texts, caption):
            out.append({**f, "file": name, "record": rec_path.relative_to(root).as_posix()})
    return out
