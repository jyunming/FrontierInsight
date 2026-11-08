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
                keys.add(("ref", str(r.get("id")), str(r.get("label"))))  # a line's id is the panel's own
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


def quest_neutral_declared(root: Path) -> list[dict[str, Any]]:
    """The series of every placed figure that its body declared as a result of the plan but that FI's check found not to draw
    that result (the figure draws it as a computed value): ``[{"file", "number", "label", "record"}]``, one per figure and
    declared result, ``number`` the figure's place among the paper's figures."""
    root = Path(root)
    paper = root / "paper" / "paper.md"
    if not paper.exists():
        paper = root / "paper.md"
    try:
        captions = paper_captions(paper.read_text(encoding="utf-8"))
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for number, name in enumerate(captions, 1):
        rec_path = root / ".fi" / "figure_records" / f"{Path(name).stem}.calls.json"
        try:
            calls = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seen: set[str] = set()
        for p in _panels(calls):
            for s in p.get("series") or []:
                got = s.get("declared_as") if isinstance(s, dict) else None
                if isinstance(got, dict) and got.get("id") and str(got["id"]) not in seen:
                    seen.add(str(got["id"]))
                    out.append({"file": name, "number": number, "label": str(got.get("label") or got["id"]),
                                "checked": "differs" if got.get("checked", "differs") == "differs" else "not compared",
                                "record": rec_path.relative_to(root).as_posix()})
    return out


def scale_notes(root: Path) -> list[dict[str, Any]]:
    """What a reader of each figure the delivered paper places is not told by the figure itself about its axes' scales, once per
    figure: an axis FI drew on a log scale because the plan did not say (the axis label says "log scale", the plan did not ask),
    and a log axis the plan asked for that FI drew linear, with FI's reason. Each item has the figure's ``file``, ``said`` and
    the ``record`` path."""
    root = Path(root)
    paper = root / "paper" / "paper.md"
    if not paper.exists():
        paper = root / "paper.md"
    try:
        captions = paper_captions(paper.read_text(encoding="utf-8"))
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for name in captions:
        rec_path = root / ".fi" / "figure_records" / f"{Path(name).stem}.calls.json"
        try:
            calls = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        said: list[str] = []
        for axis in ("x", "y"):
            for p in _panels(calls):
                info = p.get(axis) if isinstance(p.get(axis), dict) else {}
                if info.get("scale_note"):
                    text = "the %s axis is linear: %s" % (axis, info["scale_note"])
                elif info.get("scale") == "log" and info.get("scale_source") == "FI rule":
                    text = "FI drew the %s axis on a log scale (the plan did not say); its label says so" % axis
                else:
                    continue
                if text not in said:
                    said.append(text)
        if said:
            out.append({"file": name, "said": "; ".join(said), "record": rec_path.relative_to(root).as_posix()})
    return out


# ---- a reference the plan gives for a result the figure draws ---------------------------------------------------------------------

def planned_references(protocol: Any) -> dict[str, dict[str, Any]]:
    """The reference the plan declares for each of its metrics, from the frozen protocol (``metrics[].reference``): ``{result id:
    {"ids", "value", "label", "formula"}}``. ``ids`` are the names a line of it drawn from the plan's records may carry: the result's
    own id, the reference's name (its ``value`` when that is a name, as a plan gives ``rate_theory`` for ``rate``) and, when the
    reference is a level the plan names (a ``formula`` that is one bare identifier, which the frame draws under that name), that
    name. A metric with no reference, or one that names neither a formula nor a value, has none."""
    out: dict[str, dict[str, Any]] = {}
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    for m in metrics if isinstance(metrics, list) else []:
        ref = m.get("reference") if isinstance(m, dict) else None
        if not isinstance(ref, dict) or not m.get("id") or not (ref.get("formula") or ref.get("value") is not None):
            continue
        value = ref.get("value")
        named = value if isinstance(value, str) and value.strip() else None
        formula = str(ref.get("formula") or "")
        level = formula if re.fullmatch(r"[A-Za-z_]\w*", formula) else None
        out[str(m["id"])] = {"ids": [i for i in dict.fromkeys([str(m["id"]), named, level]) if i], "value": value,
                             "label": str(m.get("label") or m["id"]), "formula": formula}
    return out


def _drawn_reference(calls: dict[str, Any], rid: str, ref: dict[str, Any]) -> bool:
    """Whether some panel of the record draws the plan's reference ``ref`` of result ``rid``: a line or curve FI drew from the
    plan's records under one of its ids, a series that carries the reference's own name, or (for a reference given as a
    number) a line at that value."""
    number = ref["value"] if isinstance(ref["value"], (int, float)) and not isinstance(ref["value"], bool) else None
    named = [ref["value"]] if isinstance(ref["value"], str) and ref["value"] != rid else []
    for p in _panels(calls):
        for line in p.get("refs") or []:
            if not isinstance(line, dict):
                continue
            if line.get("source") == "plan" and str(line.get("id")) in ref["ids"]:
                return True
            if number is not None and isinstance(line.get("value"), (int, float)) \
                    and abs(float(line["value"]) - float(number)) <= 1e-9 * max(1.0, abs(float(number))):
                return True
        for s in p.get("series") or []:
            if not isinstance(s, dict):
                continue
            ident = str(s.get("id")).split("@")[0]
            if (s.get("source") == "plan" and ident in ref["ids"]) or ident in named:
                return True
    return False


def missing_references(calls: dict[str, Any], protocol: Any) -> list[dict[str, Any]]:
    """The references the plan declares for a result a figure draws and the figure does not draw: ``[{"result", "label",
    "formula"}]``. A result counts as drawn when a series of a panel FI checked has it as its ``what``; a figure whose every
    panel is a custom drawing says nothing FI can read, so it has none. A fault of the figure (FI's record of what it draws
    against the plan), never a reading of any text."""
    planned = planned_references(protocol)
    panels = _panels(calls)
    if not planned or not panels or all(p.get("raw") for p in panels):
        return []
    out: list[dict[str, Any]] = []
    drawn = {str(s.get("what")) for p in panels if not p.get("raw") for s in p.get("series") or []
             if isinstance(s, dict) and s.get("kind") != "image"}
    for rid, ref in planned.items():
        if rid in drawn and not _drawn_reference(calls, rid, ref):
            out.append({"result": rid, "label": ref["label"], "formula": ref["formula"]})
    return out


def quest_missing_references(root: Path, protocol: Any) -> list[dict[str, Any]]:
    """:func:`missing_references` for every figure that has a call record, once per figure and result: ``[{"file", "number",
    "result", "label", "formula", "record"}]``. With a delivered paper only the figures it places are read, and ``number`` is the
    figure's place in it; before the paper is written every figure with a record is read and ``number`` is ``None``."""
    root = Path(root)
    paper = root / "paper" / "paper.md"
    if not paper.exists():
        paper = root / "paper.md"
    try:
        names = {n: i for i, n in enumerate(paper_captions(paper.read_text(encoding="utf-8")), 1)}
    except OSError:
        folder = root / ".fi" / "figure_records"
        stems = [p.name[:-len(".calls.json")] for p in sorted(folder.glob("*.calls.json"))] if folder.is_dir() else []
        # a record of a figure FI later took out stays in the folder: only a figure whose file is still there is read
        names = {f"{t}.png": None for t in stems if any((root / "figures").glob(f"{t}.*"))}
    out: list[dict[str, Any]] = []
    for name, number in names.items():
        rec_path = root / ".fi" / "figure_records" / f"{Path(name).stem}.calls.json"
        try:
            calls = json.loads(rec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for f in missing_references(calls, protocol):
            out.append({"file": name, "number": number, **f, "record": rec_path.relative_to(root).as_posix()})
    return out
