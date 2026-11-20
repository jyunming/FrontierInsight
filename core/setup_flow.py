"""FI's set-up figure: how the study is set up, drawn by FI from its own records, with no model call.

The paper, the poster and the slides show the study's set-up next to its results: one figure that reads left to right,
from the study's inputs (the settings it varies, with their values or ranges, and the values it holds fixed) through the
steps of its computation to the results it reports. Every word and number on it is taken from FI's records:

- ``needs/FROZEN_PROTOCOL.json`` (the frozen plan): ``grid``, ``sub_studies`` or the design search (``optimisation``),
  ``runs_per_setting``, ``constants``, ``labels``, ``metrics``, ``derived`` and ``model`` (its step cards and equations);
- ``.fi/function_steps.json``: which of the plan's equations and steps passed their own check before the run;
- ``.fi/code_layout.json``: whether the code was written whole instead (then no step has a check of its own to show).

A quantity is named by the plan's own name for a reader (``protocol.labels``, else a metric's ``label``), with its unit;
one the plan gives no such name is never printed by its code name, only counted in its box's "and N more". A value or a
name the plan wrote as text is shown in its own words, an underscore read as a space. An equation is shown only by a name
the plan gives it (ids such as E1 say nothing to a reader), and a step by its card's name. Long lists are cut to their
first entries with "and N more", which the caption FI writes says; no number is typed by a model.

The figure is the paper's Figure 1, so it must exist before FI gives any figure a number: the analysis is the first step
that is shown the numbered figure list (``core.engine._figure_list_for_prompt``) and writes prose that names figures by
number, which the writer then reads. So it is drawn at the start of every analysis (:func:`draw_quietly`, called by
``Engine._node_analyze``), when the frozen plan and the final code's function steps exist, into ``.fi/setup_flow.png``
(each run empties ``figures/``), and the write stage copies it into ``figures/`` (:func:`place_quietly`). FI's record
of what it drew is ``.fi/setup_flow.json``: the boxes and their text, the caption, and the digests of the drawing and of
the frozen plan it was drawn from (or why nothing was drawn). That record, not the file name, is what tells the rest of
FI that this figure is not a result: it is not in ``state["figures"]`` (the run's own figures), the checks that judge
result figures never read it (:func:`without`), and ``core.engine._figure_numbers`` numbers it first (:func:`shown`) in
the analysis, the paper, the poster and the slides alike, with FI's caption. Records too thin to show a flow (no frozen
plan, nothing varied, no result named) give no figure and one line in ``QUALITY.md`` (:func:`quality_line`); a fault
here never stops a study and never raises.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILE = "setup_flow.png"
RECORD = "setup_flow.json"
SCHEMA = 1

#: The width the figure is drawn at: the text width of a one-column paper (``core.plot_style.PAPER_FULL_WIDTH_IN``), so
#: its text prints at the size it was drawn.
WIDTH_IN = 6.3
DPI = 200
BODY_PT = 8.0
TITLE_PT = 8.5
HEADER_PT = 9.5
#: At most this many boxes in a column and lines of entries in a box; what is left is said as "and N more".
MAX_BOXES = 4
MAX_LINES = 5
#: The most lines one entry may wrap to.
MAX_WRAP = 3

_COLOURS = {"inputs": ("#e8f0fb", "#3b5b8c"), "computation": ("#f1f1f1", "#555555"), "results": ("#e9f5ec", "#3c7a4f")}
_HEADERS = {"inputs": "Inputs", "computation": "Computation", "results": "Results reported"}


@dataclass
class Box:
    title: str
    lines: list[str] = field(default_factory=list)
    dashed: bool = False
    source: str = ""


@dataclass
class Flow:
    columns: dict[str, list[Box]]
    shape: str  # "sub_studies" | "grid" | "search"
    dashed: bool = False
    cut: bool = False


# --------------------------------------------------------------------------------------------------- reading the records


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _protocol_file(root: Path) -> Path:
    return root / "needs" / "FROZEN_PROTOCOL.json"


def _record(root: Any) -> dict[str, Any] | None:
    if not root:
        return None
    record = _json(Path(str(root)) / ".fi" / RECORD)
    return record if isinstance(record, dict) else None


def _drawing(root: Any) -> Path:
    """Where FI keeps its drawing until the write stage copies it into ``figures/`` (a run empties ``figures/``)."""
    return Path(str(root)) / ".fi" / FILE


def _ours(root: Any, record: dict[str, Any] | None, name: str) -> bool:
    """Whether ``figures/<name>`` is the set-up figure FI drew, by its record: the record names the file, says it was
    drawn, and the file there has the digest FI recorded (a run's own figure saved under that name is not FI's)."""
    if not record or not record.get("drawn") or record.get("file") != name:
        return False
    digest = str(record.get("png_sha256") or "")
    return bool(digest) and _digest(Path(str(root)) / "figures" / name) == digest


def shown(root: Any) -> str | None:
    """The file name of FI's set-up figure when the study shows it: FI's record says it drew it from the frozen plan as it
    now stands, and its drawing in ``.fi/`` is the one recorded. ``None`` otherwise (no record, nothing drawn, a plan
    changed since). Never raises."""
    try:
        record = _record(root)
        if not record or not record.get("drawn") or not record.get("file"):
            return None
        if _digest(_drawing(root)) != str(record.get("png_sha256") or "-"):
            return None
        if str(record.get("protocol_sha256") or "") != _digest(_protocol_file(Path(str(root)))):
            return None
        return str(record["file"])
    except Exception:  # noqa: BLE001 -- a record that cannot be read gives no figure, never a failure
        return None


def caption(root: Any) -> str:
    """The caption FI wrote for its set-up figure (``""`` when there is none)."""
    record = _record(root) or {}
    return str(record.get("caption") or "") if record.get("drawn") else ""


def without(root: Any, names: list[str]) -> list[str]:
    """``names`` (files of ``figures/``) without FI's set-up figure, for the checks and lists that are about the run's own
    figures. Only the file FI's record says it drew is left out. Never raises."""
    try:
        record = _record(root)
    except Exception:  # noqa: BLE001
        return list(names)
    if not record or not record.get("drawn"):
        return list(names)
    return [n for n in names if not _ours(root, record, Path(str(n)).name)]


def quality_line(root: Any) -> str | None:
    """The one line ``QUALITY.md`` gives when FI drew no set-up figure for the paper, saying why; ``None`` when it drew one
    or never tried (no paper was written)."""
    record = _record(root)
    if not record or record.get("drawn"):
        return None
    return f"No set-up figure in the paper: {record.get('why') or 'FI could not draw it'}."


# --------------------------------------------------------------------------------------------------- what the figure says


def _num(value: Any) -> str:
    """A number of the plan as the figure shows it: as the plan wrote it, a large round integer in powers of ten."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        if abs(value) >= 100000 and value % 10000 == 0:
            mant, exp = f"{value:e}".split("e")
            return f"{mant.rstrip('0').rstrip('.')}e{int(exp)}"
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        if "e" in text:
            mant, exp = text.split("e")
            return f"{mant}e{int(exp)}"
        return text[:-2] if text.endswith(".0") else text
    return " ".join(str(value).split())


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _words(text: Any) -> str:
    """A name or a value the plan wrote as text, as a reader reads it: its own words, an underscore read as a space
    (``stress_free`` is "stress free"); nothing else in it is changed."""
    return " ".join(str(text).replace("_", " ").split())


def _value(value: Any) -> str:
    return _num(value) if _is_num(value) or isinstance(value, bool) else _words(value)


def _values(values: Any) -> str:
    """What a setting takes: its values when there are at most four, else its range and how many."""
    if not isinstance(values, list):
        return _value(values)
    if len(values) <= 4:
        return ", ".join(_value(v) for v in values)
    if all(_is_num(v) for v in values):
        return f"{_num(min(values))} to {_num(max(values))} ({len(values)} values)"
    return ", ".join(_value(v) for v in values[:3]) + f", and {len(values) - 3} more"


def _label(protocol: dict[str, Any], name: str) -> str:
    """The plan's own name of a quantity for a reader (``protocol.labels``, else a metric's ``label``), its unit included;
    ``""`` when the plan gives it none. A quantity with no such name is never printed by its code name: it is counted in
    its box's "and N more"."""
    labels = protocol.get("labels")
    text = labels.get(str(name)) if isinstance(labels, dict) else None
    if not (isinstance(text, str) and text.strip()):
        text = next((m.get("label") for m in protocol.get("metrics") or []
                     if isinstance(m, dict) and str(m.get("id")) == str(name) and isinstance(m.get("label"), str)), None)
    return " ".join(text.split()) if isinstance(text, str) and text.strip() else ""


def _has_unit(label: str) -> bool:
    return "(" in label


def _cut(lines: list[str], what: str, limit: int = MAX_LINES, hidden: int = 0) -> tuple[list[str], bool]:
    """At most ``limit`` lines; what is left, with the ``hidden`` entries the plan gives no reader's name, is counted in a
    last line "and N more <what>" (``what`` is a plural; one is said in the singular)."""
    extra = hidden
    if len(lines) + (1 if extra else 0) > limit:
        keep = limit - 1
        extra += len(lines) - keep
        lines = lines[:keep]
    if not extra:
        return lines, False
    noun = what[:-1] if extra == 1 and what.endswith("s") and not what.endswith("ss") else what
    return lines + [f"and {extra} more {noun}".rstrip() if lines else f"{extra} {noun}".rstrip()], True


def _cap_boxes(boxes: list[Box], what: str, limit: int) -> tuple[list[Box], bool]:
    """At most ``limit`` boxes: the last one kept names the rest."""
    if len(boxes) <= limit:
        return boxes, False
    rest = boxes[limit - 1:]
    names = [b.title for b in rest]
    lines, _ = _cut(names, "", 3)
    lines = [ln.rstrip() for ln in lines]
    return boxes[: limit - 1] + [Box(f"and {len(rest)} more {what}", lines, source="+")], True


def _settings_held(protocol: dict[str, Any]) -> tuple[list[str], int]:
    """The values the study holds fixed that the plan names for a reader, and how many others it holds (counted only)."""
    out: list[str] = []
    hidden = 0
    runs = protocol.get("runs_per_setting")
    if _is_num(runs) and runs:
        out.append(f"Runs per setting: {_num(runs)}")
    held: list[tuple[str, Any]] = []
    grid = protocol.get("grid")
    if isinstance(grid, dict):
        varied = _varied_ids(protocol)
        for key, vals in grid.items():
            if key in varied:
                continue
            held.append((key, vals[0] if isinstance(vals, list) and len(vals) == 1 else
                         vals if not isinstance(vals, list) else None))
    search = protocol.get("optimisation")
    fixed = search.get("fixed") if isinstance(search, dict) else None
    if isinstance(fixed, dict):
        held += list(fixed.items())
    constants = protocol.get("constants")
    if isinstance(constants, dict):
        held += [(k, v) for k, v in constants.items() if _is_num(v)]
    for key, val in held:
        label = _label(protocol, key)
        if label and (_is_num(val) or isinstance(val, str) and len(val) <= 24):
            out.append(f"{label}: {_value(val)}")
        elif val is not None:
            hidden += 1
    return out, hidden


def _varied_ids(protocol: dict[str, Any]) -> set[str]:
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    varied = {str(k) for k, v in grid.items() if isinstance(v, list) and len(v) > 1}
    for sub in protocol.get("sub_studies") or []:
        if isinstance(sub, dict):
            varied |= {str(v) for v in sub.get("vary") or [] if str(v) in grid}
    return varied


def _inputs(protocol: dict[str, Any]) -> tuple[list[Box], str, bool]:
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    search = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
    subs = [s for s in protocol.get("sub_studies") or [] if isinstance(s, dict) and s.get("vary")]
    boxes: list[Box] = []
    cut = False
    shape = ""
    variables = [v for v in (search or {}).get("design_variables") or [] if isinstance(v, dict) and v.get("name")]
    if variables:
        shape = "search"
        lines = []
        for v in variables:
            label = _label(protocol, v["name"])
            if not label:
                continue
            span = f"{_num(v.get('low'))} to {_num(v.get('high'))}" if v.get("low") is not None and v.get("high") is not None \
                else "searched"
            unit = str(v.get("unit") or "").strip()
            if unit and unit.lower() not in ("dimensionless", "-", "none") and not _has_unit(label):
                span += f" {unit}"
            lines.append(f"{label}: {span}")
        lines, cut = _cut(lines, "design variables", hidden=len(variables) - len(lines))
        boxes.append(Box("Design search over", lines, source="optimisation.design_variables"))
    elif subs:
        shape = "sub_studies"
        for i, sub in enumerate(subs):
            keys = [str(x) for x in sub.get("vary") or []]
            lines = [f"{_label(protocol, k)}: {_values(grid[k])}" if k in grid else _label(protocol, k)
                     for k in keys if _label(protocol, k)]
            lines, c = _cut(lines, "settings", hidden=len(keys) - len(lines))
            cut = cut or c
            boxes.append(Box(_words(sub.get("name") or f"sub-study {i + 1}"), lines, source=f"sub_studies[{i}]"))
        boxes, c = _cap_boxes(boxes, "sub-studies", MAX_BOXES - 1)
        cut = cut or c
    else:
        varied = [k for k, v in grid.items() if isinstance(v, list) and len(v) > 1]
        if varied:
            shape = "grid"
            named = [f"{_label(protocol, k)}: {_values(grid[k])}" for k in varied if _label(protocol, k)]
            lines, cut = _cut(named, "settings", hidden=len(varied) - len(named))
            boxes.append(Box("Settings varied", lines, source="grid"))
    if not boxes:
        return [], "", False
    held, hidden = _settings_held(protocol)
    if held or hidden:
        lines, c = _cut(held, "held fixed", hidden=hidden)
        cut = cut or c
        boxes.append(Box("Held fixed", lines, source="runs_per_setting, grid, optimisation.fixed, constants"))
    return boxes, shape, cut


def _statuses(steps_record: Any, layout_record: Any) -> dict[str, str]:
    """``{equation or step id: status}`` from FI's record of the function steps; empty when the code was written whole."""
    if isinstance(layout_record, dict) and layout_record.get("fell_back"):
        return {}
    if not isinstance(steps_record, dict) or steps_record.get("status") == "fell_back":
        return {}
    out: dict[str, str] = {}
    for entry in (steps_record.get("functions") or {}).values():
        if isinstance(entry, dict) and entry.get("equation"):
            out[str(entry["equation"])] = str(entry.get("status") or "")
    return out


def _computation(protocol: dict[str, Any], steps_record: Any, layout_record: Any) -> tuple[list[Box], bool, bool]:
    model = protocol.get("model") if isinstance(protocol.get("model"), dict) else {}
    status = _statuses(steps_record, layout_record)
    steps = [s for s in model.get("steps") or [] if isinstance(s, dict) and str(s.get("name") or "").strip()]
    equations = [e for e in model.get("equations") or [] if isinstance(e, dict) and e.get("id")]
    boxes: list[Box] = []
    dashed = cut = False
    # An equation is shown by the name the plan gives it; the plan's ids (E1, E2, ...) say nothing to a reader, so a plan
    # that records only ids and formulas gives no equations box.
    named = [_words(e.get("name") or e.get("title")) for e in equations
             if str(e.get("name") or e.get("title") or "").strip()]
    room = MAX_BOXES - (1 if named else 0)
    one_box = len(steps) > room  # too many steps for a box each: one box lists them
    rows: list[str] = []
    for s in steps:
        ident = str(s.get("id") or "")
        title = " ".join(str(s.get("name") or "").split())
        if not title:
            continue
        failed = status.get(ident) == "failed"
        if one_box:
            rows.append(title + (" (its check did not pass)" if failed else ""))
            continue
        dashed = dashed or failed
        boxes.append(Box(title, ["its check did not pass"] if failed else [], dashed=failed,
                         source=f"model.steps.{ident}"))
    if one_box:
        rows, cut = _cut(rows, "steps")
        boxes.append(Box("Steps of the method", rows, source="model.steps"))
    if named:
        lines, c = _cut(named, "equations", 4, hidden=len(equations) - len(named))
        cut = cut or c
        boxes.append(Box("Model equations", lines, source="model.equations"))
    return boxes, dashed, cut


def _results(protocol: dict[str, Any]) -> tuple[list[Box], bool]:
    boxes: list[Box] = []
    cut = False
    search = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else None
    objective = (search or {}).get("objective")
    if isinstance(objective, dict) and objective.get("quantity"):
        label = _label(protocol, objective["quantity"])
        unit = str(objective.get("unit") or "").strip()
        if label and unit and not _has_unit(label):
            label += f" ({unit})"
        lines = [f"{str(objective.get('direction') or 'optimise').strip()} {label or 'the objective the plan names'}"]
        constraints = [c for c in (search or {}).get("constraints") or []
                       if isinstance(c, dict) and c.get("quantity") and c.get("limit") is not None]
        shown = [f"with {_label(protocol, c['quantity'])} {c['limit']}" for c in constraints
                 if _label(protocol, c["quantity"])]
        lines, c = _cut(lines + shown, "conditions", hidden=len(constraints) - len(shown))
        cut = cut or c
        boxes.append(Box("Objective", lines, source="optimisation.objective"))
    families: dict[str, list[str]] = {}
    unnamed: dict[str, int] = {}
    for m in protocol.get("metrics") or []:
        if not isinstance(m, dict) or not m.get("id") or str(m.get("role") or "result") != "result":
            continue
        family = _words(m.get("family") or "")
        families.setdefault(family, [])
        label = _label(protocol, m["id"])
        if not label:
            unnamed[family] = unnamed.get(family, 0) + 1
            continue
        unit = str(m.get("physical_unit") or "").strip()
        if unit and not _has_unit(label):
            label += f" ({unit})"
        families[family].append(label)
    for fam, labels in families.items():
        lines, c = _cut(labels, "results", 4, hidden=unnamed.get(fam, 0))
        cut = cut or c
        boxes.append(Box(fam[:1].upper() + fam[1:] if fam else "Results", lines, source=f"metrics:{fam}"))
    derived = [d for d in protocol.get("derived") or [] if isinstance(d, dict) and d.get("id")
               and str(d.get("role") or "result") == "result"]
    if derived and len(boxes) < MAX_BOXES:
        named = [" ".join(str(d["label"]).split()) if isinstance(d.get("label"), str) and d["label"].strip()
                 else _label(protocol, d["id"]) for d in derived]
        lines, c = _cut([n for n in named if n], "results", 4, hidden=sum(1 for n in named if not n))
        cut = cut or c
        boxes.append(Box("Worked out from the runs", lines, source="derived"))
    boxes, c = _cap_boxes(boxes, "result groups", MAX_BOXES)
    return boxes, cut or c


def build(protocol: Any, steps_record: Any = None, layout_record: Any = None) -> tuple[Flow | None, str]:
    """The flow the figure draws, from the records (pure: no file is read or written), or ``(None, why)`` when the records
    are too thin for a flow: no frozen plan, nothing the study varies or searches over, or no result it reports."""
    if not isinstance(protocol, dict) or not protocol:
        return None, "the study has no frozen plan to draw its set-up from"
    inputs, shape, cut_in = _inputs(protocol)
    if not inputs:
        return None, "the plan names no setting the study varies and no design it searches over"
    results, cut_out = _results(protocol)
    if not results:
        return None, "the plan names no result the study reports"
    steps, dashed, cut_steps = _computation(protocol, steps_record, layout_record)
    columns = {"inputs": inputs, **({"computation": steps} if steps else {}), "results": results}
    return Flow(columns, shape, dashed=dashed, cut=cut_in or cut_out or cut_steps), ""


def caption_of(flow: Flow) -> str:
    """FI's caption for the flow, from what the figure shows (no number is in it: the figure itself holds the plan's)."""
    left = {"search": "the design variables it searches over and the values it holds fixed",
            "sub_studies": "its sub-studies with the settings each one varies, and the values it holds fixed",
            "grid": "the settings it varies and the values it holds fixed"}.get(flow.shape, "its inputs")
    parts = [f"How the study is set up, drawn by FI from the study's frozen plan (not a result of the run): {left} (left)"]
    if "computation" in flow.columns:
        equations = any(b.source == "model.equations" for b in flow.columns["computation"])
        parts.append(f"the steps{' and equations' if equations else ''} of its computation (middle)")
    parts.append("and the results it reports (right)")
    text = ", ".join(parts[:-1]) + (", " if len(parts) > 2 else " ") + parts[-1] + "."
    if flow.dashed:
        text += " A dashed box is a step whose own check did not pass before the run."
    if flow.cut:
        text += ' A list that ends with "more" shows its first entries; the plan lists them all.'
    return text


# --------------------------------------------------------------------------------------------------- drawing


#: A word of a label: a ``$...$`` span (spaces inside it included) with the characters joined to it, or any other run
#: of non-space characters.
_TOKEN = re.compile(r"[^\s$]*(?:\$[^$]*\$[^\s$]*)+|\S+")


def _plain(text: str) -> str:
    return text.replace("$", "").replace("\\", "")


class _Painter:
    """Text measured and drawn on one matplotlib ``Figure`` (no pyplot: nothing process-wide is touched)."""

    def __init__(self, height_in: float) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(WIDTH_IN, height_in), dpi=DPI, facecolor="white")
        self.canvas = FigureCanvasAgg(self.fig)
        self.renderer = self.canvas.get_renderer()

    def text(self, x: float, y: float, s: str, *, size: float, bold: bool = False, **kw: Any) -> Any:
        t = self.fig.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal",
                          transform=self.fig.dpi_scale_trans, **kw)
        try:
            t.get_window_extent(self.renderer)
        except Exception:  # noqa: BLE001 -- mathtext the plan's label cannot be drawn in: its words, without the marks
            t.set_text(_plain(s))
            t.set_parse_math(False)
        return t

    def size(self, s: str, *, size: float, bold: bool = False, linespacing: float = 1.25) -> tuple[float, float]:
        t = self.text(0, 0, s, size=size, bold=bold, linespacing=linespacing)
        box = t.get_window_extent(self.renderer)
        t.remove()
        return box.width / DPI, box.height / DPI

    def wrap(self, entry: str, width: float, *, size: float, bold: bool = False) -> list[str]:
        """``entry`` broken into lines at most ``width`` inches wide (a ``$...$`` span is never broken), at most
        :data:`MAX_WRAP` lines, the last cut with "..." when it does not fit."""
        lines: list[str] = []
        current = ""
        for word in _TOKEN.findall(entry):
            trial = f"{current} {word}" if current else word
            if current and self.size(trial, size=size, bold=bold)[0] > width:
                lines.append(current)
                current = word
            else:
                current = trial
            while self.size(current, size=size, bold=bold)[0] > width and len(current) > 4:
                # a single word wider than the box: broken where it fits (a formula in it is then read as its words)
                current = _plain(current) if "$" in current else current
                cut = max(1, int(len(current) * width / max(self.size(current, size=size, bold=bold)[0], 1e-6)) - 1)
                under = current.rfind("_", 0, cut)  # a name in parts is broken after one of its parts
                cut = under + 1 if under > 0 else cut
                head, current = current[:cut], current[cut:]
                lines.append(head)
        if current:
            lines.append(current)
        if len(lines) > MAX_WRAP:
            last = lines[MAX_WRAP - 1]
            while last and self.size(last + " ...", size=size, bold=bold)[0] > width:
                last = last[:-1]
            lines = lines[: MAX_WRAP - 1] + [last.rstrip() + " ..."]
        return lines


_PAD = 0.06
_GAP_Y = 0.09
_GAP_X = 0.40
_MARGIN = 0.05
_HEADER_H = 0.24


def _inside(extent: Any, x: float, y: float, w: float, h: float) -> bool:
    """Whether a text's drawn extent (pixels) lies inside the box at ``(x, y)`` of ``w`` by ``h`` inches."""
    tol = 0.5 / DPI
    return (extent.x0 / DPI >= x - tol and extent.x1 / DPI <= x + w + tol
            and extent.y0 / DPI >= y - tol and extent.y1 / DPI <= y + h + tol)


def render(flow: Flow, png: Path) -> list[dict[str, Any]]:
    """Draw ``flow`` into ``png``; returns each box as drawn (its column, title and lines after wrapping, dashed)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    names = list(flow.columns)
    col_w = (WIDTH_IN - 2 * _MARGIN - _GAP_X * (len(names) - 1)) / len(names)
    inner = col_w - 2 * _PAD
    measure = _Painter(4.0)
    laid: dict[str, list[tuple[Box, list[str], list[str], float, float]]] = {}
    heights: dict[str, float] = {}
    for col in names:
        rows = []
        for box in flow.columns[col]:
            title = measure.wrap(box.title, inner, size=TITLE_PT, bold=True)
            body = [("• " if k == 0 else "   ") + ln for entry in box.lines
                    for k, ln in enumerate(measure.wrap(entry, inner - 0.16, size=BODY_PT))]
            th = measure.size("\n".join(title), size=TITLE_PT, bold=True)[1] if title else 0.0
            bh = measure.size("\n".join(body), size=BODY_PT)[1] if body else 0.0
            h = _PAD * 2 + th + (0.04 + bh if body else 0.0)
            rows.append((box, title, body, th, h))
        laid[col] = rows
        heights[col] = sum(r[4] for r in rows) + _GAP_Y * (len(rows) - 1)
    content = max(heights.values())
    height = _MARGIN * 2 + _HEADER_H + content
    paint = _Painter(height)
    fig = paint.fig
    top = height - _MARGIN - _HEADER_H
    drawn: list[dict[str, Any]] = []
    for i, col in enumerate(names):
        x0 = _MARGIN + i * (col_w + _GAP_X)
        fill, edge = _COLOURS[col]
        paint.text(x0 + col_w / 2, height - _MARGIN - 0.02, _HEADERS[col], size=HEADER_PT, bold=True,
                   ha="center", va="top", color="#222222")
        y = top - (content - heights[col]) / 2
        for box, title, body, th, h in laid[col]:
            fig.add_artist(FancyBboxPatch(
                (x0, y - h), col_w, h, boxstyle="round,pad=0,rounding_size=0.05", transform=fig.dpi_scale_trans,
                facecolor=fill, edgecolor=edge, linewidth=1.3 if box.dashed else 0.9,
                linestyle=(0, (4, 2.5)) if box.dashed else "solid"))
            texts = []
            if title:
                texts.append(paint.text(x0 + _PAD, y - _PAD, "\n".join(title), size=TITLE_PT, bold=True, va="top",
                                        ha="left", color="#111111", linespacing=1.25))
            if body:
                texts.append(paint.text(x0 + _PAD + 0.02, y - _PAD - th - 0.04, "\n".join(body), size=BODY_PT,
                                        va="top", ha="left", color="#222222", linespacing=1.25))
            # FI's own check of the drawing: every text inside its box (the box's padding aside)
            inside = all(_inside(t.get_window_extent(paint.renderer), x0, y - h, col_w, h) for t in texts)
            drawn.append({"column": col, "name": box.title, "entries": list(box.lines), "title": title, "lines": body,
                          "dashed": box.dashed, "from": box.source, "fits": inside})
            y -= h + _GAP_Y
        if i:
            mid = top - content / 2
            fig.add_artist(FancyArrowPatch(
                (x0 - _GAP_X + 0.05, mid), (x0 - 0.05, mid), transform=fig.dpi_scale_trans, arrowstyle="-|>",
                mutation_scale=14, linewidth=1.4, color="#444444"))
    png.parent.mkdir(parents=True, exist_ok=True)
    paint.canvas.print_png(str(png), metadata={"Software": None})
    return drawn


# --------------------------------------------------------------------------------------------------- the write stage's call


def _forget_copy(root: Path, old: dict[str, Any] | None) -> None:
    """Take FI's earlier copy out of ``figures/`` (only the file its record says it drew there)."""
    if old and _ours(root, old, FILE):
        (root / "figures" / FILE).unlink(missing_ok=True)


def draw(root: Path, run_figures: list[str] | None = None) -> dict[str, Any]:
    """Draw FI's set-up figure for the quest at ``root`` into ``.fi/`` and write its record; returns the record. Raises on
    a fault (:func:`draw_quietly` is the analysis stage's call)."""
    root = Path(root)
    record_path = root / ".fi" / RECORD
    record_path.parent.mkdir(parents=True, exist_ok=True)
    old = _record(root)
    png = _drawing(root)
    record: dict[str, Any] = {"schema": SCHEMA, "file": FILE, "drawn": False,
                              "protocol_sha256": _digest(_protocol_file(root))}
    flow, why = None, f"the run drew a figure of its own named {FILE}"
    if FILE not in {Path(str(f)).name for f in run_figures or []}:
        from . import frozen_protocol as _frozen

        flow, why = build(_frozen.protocol_of(root), _json(root / ".fi" / "function_steps.json"),
                          _json(root / ".fi" / "code_layout.json"))
    if flow is None:
        # an earlier drawing of FI's is not shown for a plan that no longer gives one
        _forget_copy(root, old)
        png.unlink(missing_ok=True)
        record["why"] = why
        record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        return record
    boxes = render(flow, png)
    record.update(drawn=True, png_sha256=_digest(png), caption=caption_of(flow), shape=flow.shape, boxes=boxes)
    if old and old.get("png_sha256") != record["png_sha256"]:
        _forget_copy(root, old)  # the write stage copies the new drawing
    record_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def draw_quietly(root: Path, run_figures: list[str] | None, log: Any = None) -> None:
    """:func:`draw` for the analysis stage: a fault is recorded (one ``QUALITY.md`` line) and logged, never raised."""
    old = None
    try:
        old = _record(root)
        record = draw(Path(root), run_figures)
        if log is not None:
            if record.get("drawn"):
                log.info("[analyze] FI drew the study's set-up figure from its records (Figure 1 of the paper)")
            else:
                log.info("[analyze] no set-up figure: %s", record.get("why"))
    except Exception as exc:  # noqa: BLE001 -- the figure is extra: a paper without it is today's paper
        try:
            _forget_copy(Path(root), old)
            _drawing(root).unlink(missing_ok=True)
            path = Path(root) / ".fi" / RECORD
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"schema": SCHEMA, "file": FILE, "drawn": False,
                                        "why": f"FI could not draw it ({type(exc).__name__})"}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        if log is not None:
            log.warning("[analyze] the set-up figure could not be drawn (%r); the study goes on without it", exc)


def place_quietly(root: Path, log: Any = None) -> None:
    """The write stage's call: FI's drawing, when the study shows it (:func:`shown`), copied into ``figures/`` where the
    paper, the poster and the slides embed it. Nothing is drawn here, so the numbers the analysis was given stand. Never
    raises."""
    try:
        root = Path(root)
        name = shown(root)
        if name is None:
            _forget_copy(root, _record(root))
            return
        target = root / "figures" / name
        if _digest(target) != _digest(_drawing(root)):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_drawing(root), target)
    except Exception as exc:  # noqa: BLE001
        if log is not None:
            log.warning("[write] the set-up figure could not be put in figures/ (%r)", exc)


__all__ = ["FILE", "RECORD", "Box", "Flow", "build", "caption", "caption_of", "draw", "draw_quietly", "place_quietly",
           "quality_line", "render", "shown", "without"]
