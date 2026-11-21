"""The drawing calls of the studied system's schematic, and the facts of what they drew.

A model writes a short drawing script that calls only ``sch`` (a :class:`Schematic`): ``canvas``, ``rect``, ``circle``,
``arc``, ``line``, ``polygon``, ``arrow``, ``direction``, ``label`` and ``leader``. Every shape is placed by the id of a part
of the plan's studied system (``protocol.system``) and every word is FI's: ``label(part_id)`` draws FI's text for that part,
``arrow(link_index, label_at=...)`` the words of that link, ``direction(..., label_at=...)`` the words of the system's
direction. No call takes a string to draw, and every artist a call makes is tagged ``gid = "fi:<id>"``.

This file is carried by text into the quest's ``.fi/setup_schematic/`` folder and run there by the quest's own interpreter
(``python fi_schematic_host.py <fill.json> <attempt folder>``), never in FI's process. It therefore uses the standard library
and matplotlib only and imports nothing of FI's. It runs the script in ``body.py`` with ``sch`` and ``math`` and a short list
of builtins, saves ``drawing.png`` (300 dpi) and ``drawing_small.png`` (200 dpi, for reading at its printed size), and writes
``calls.json`` (every call, in order) and ``facts.json``: each text drawn (its tag, its words, its box in pixels, its colour
and size), each shape's tag and box, and which text a line or an outline of another id crosses. FI's own process judges
those facts against the strings it wrote itself (``core/setup_schematic.py``); nothing here decides whether the drawing is
good."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

#: The printed width: one column of a paper.
WIDTH_IN = 3.5
DPI = 300
SMALL_DPI = 200
FONT_PT = 7.0
LINK_PT = 6.0
#: Fill and line colour per kind of entry of the studied system.
KIND_COLOUR = {"part": ("#dbe9f6", "#1f4e79"), "boundary": ("#e6e6e6", "#333333"), "source": ("#fde4cf", "#a0471a")}
ARROW_COLOUR = "#444444"
DIRECTION_COLOUR = "#2f2f8f"
STYLES = ("solid", "light", "hatched", "outline")
HA = ("left", "center", "right")
VA = ("top", "center", "bottom")
#: The calls a drawing script may make, by name.
API = ("canvas", "rect", "circle", "arc", "line", "polygon", "arrow", "direction", "label", "leader")
#: The builtins a drawing script is given (no file, no import, no attribute lookup by name).
SAFE_BUILTINS = ("abs", "min", "max", "round", "range", "len", "enumerate", "zip", "list", "tuple", "float", "int", "sum",
                 "sorted", "reversed", "True", "False", "None", "ValueError", "KeyError")
#: Shrink of a text's box before another id's line counts as crossing it (a line that only grazes its corner does not).
CROSS_SHRINK = (0.9, 0.8)


class Schematic:
    """What the drawing script is given as ``sch``. ``fill``: FI's text per id (``{"parts": {id: {"kind", "text"}},
    "links": {"link<i>": {"from", "to", "text"}}, "direction": {"text", "points"} | None}``)."""

    def __init__(self, fill: dict) -> None:
        self._fill = fill
        self._calls: list[dict] = []
        self._fig = None
        self._ax = None
        self._canvas = None

    # ---- the canvas ----------------------------------------------------------------------------------------------------
    def canvas(self, width, height):
        """The drawing's data units: the figure is always 3.5 in wide and its height follows ``width:height``."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        width, height = float(width), float(height)
        if not (width > 0 and height > 0) or not (0.2 <= height / width <= 3.0):
            raise ValueError("sch.canvas(width, height): both above 0, height between 0.2 and 3 times the width")
        if self._fig is not None:
            raise ValueError("sch.canvas is called once")
        self._fig = Figure(figsize=(WIDTH_IN, WIDTH_IN * height / width), dpi=DPI, facecolor="white")
        self._canvas = FigureCanvasAgg(self._fig)
        self._ax = self._fig.add_axes([0, 0, 1, 1])
        self._ax.set_xlim(0, width)
        self._ax.set_ylim(0, height)
        self._ax.set_aspect("equal")
        self._ax.axis("off")
        self._calls.append({"call": "canvas", "width": width, "height": height})

    def _part(self, pid):
        if self._ax is None:
            raise ValueError("call sch.canvas(width, height) first")
        parts = self._fill.get("parts") or {}
        if pid not in parts:
            raise KeyError(f"unknown part id {pid!r}; the ids are {sorted(parts)}")
        return parts[pid]

    def _style(self, pid, style):
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        face, edge = KIND_COLOUR.get(self._part(pid).get("kind"), KIND_COLOUR["part"])
        kw = {"facecolor": face, "edgecolor": edge, "linewidth": 0.8, "gid": f"fi:{pid}"}
        if style == "light":
            kw["alpha"] = 0.5
        elif style == "hatched":
            kw["hatch"] = "////"
        elif style == "outline":
            kw["facecolor"] = "none"
        return kw

    @staticmethod
    def _xy(point):
        x, y = point
        return float(x), float(y)

    # ---- shapes: no call takes a string to draw -----------------------------------------------------------------------
    def rect(self, part_id, x, y, w, h, style="solid"):
        from matplotlib import patches

        self._ax.add_patch(patches.Rectangle((float(x), float(y)), float(w), float(h), **self._style(part_id, style)))
        self._calls.append({"call": "rect", "id": part_id, "xy": [float(x), float(y), float(w), float(h)], "style": style})

    def circle(self, part_id, cx, cy, r, ring_width=None, style="solid"):
        from matplotlib import patches

        kw = self._style(part_id, style)
        if ring_width:
            self._ax.add_patch(patches.Wedge((float(cx), float(cy)), float(r) + float(ring_width) / 2, 0, 360,
                                             width=float(ring_width), **kw))
        else:
            self._ax.add_patch(patches.Circle((float(cx), float(cy)), float(r), **kw))
        self._calls.append({"call": "circle", "id": part_id, "xy": [float(cx), float(cy), float(r)],
                            "ring_width": float(ring_width) if ring_width else None, "style": style})

    def arc(self, part_id, cx, cy, r, theta1, theta2, thickness=1.5):
        from matplotlib import patches

        _face, edge = KIND_COLOUR.get(self._part(part_id).get("kind"), KIND_COLOUR["part"])
        self._ax.add_patch(patches.Arc((float(cx), float(cy)), 2 * float(r), 2 * float(r), theta1=float(theta1),
                                       theta2=float(theta2), linewidth=float(thickness), color=edge, gid=f"fi:{part_id}"))
        self._calls.append({"call": "arc", "id": part_id, "xy": [float(cx), float(cy), float(r), float(theta1), float(theta2)]})

    def line(self, part_id, points, thickness=1.0, dashed=False):
        _face, edge = KIND_COLOUR.get(self._part(part_id).get("kind"), KIND_COLOUR["part"])
        pts = [self._xy(p) for p in points]
        if len(pts) < 2:
            raise ValueError("sch.line needs at least two points")
        xs, ys = zip(*pts)
        self._ax.plot(xs, ys, color=edge, linewidth=float(thickness), linestyle="--" if dashed else "-",
                      gid=f"fi:{part_id}", solid_capstyle="butt")
        self._calls.append({"call": "line", "id": part_id, "points": [list(p) for p in pts], "dashed": bool(dashed)})

    def polygon(self, part_id, points, style="solid"):
        from matplotlib import patches

        pts = [self._xy(p) for p in points]
        if len(pts) < 3:
            raise ValueError("sch.polygon needs at least three points")
        self._ax.add_patch(patches.Polygon(pts, closed=True, **self._style(part_id, style)))
        self._calls.append({"call": "polygon", "id": part_id, "points": [list(p) for p in pts], "style": style})

    def _arrow(self, gid, start, end, curved, colour, width):
        from matplotlib import patches

        self._ax.add_patch(patches.FancyArrowPatch(self._xy(start), self._xy(end), arrowstyle="-|>", mutation_scale=7,
                                                   linewidth=width, color=colour,
                                                   connectionstyle=f"arc3,rad={float(curved)}", gid=gid))

    def _words(self, gid, text, at, size, colour):
        if not text:
            return
        t = self._ax.text(*self._xy(at), text, fontsize=size, style="italic", ha="center", va="center", color=colour, gid=gid)
        self._mathtext_safe(t)

    def arrow(self, link, start, end, label_at=None, curved=0.0):
        """Link number ``link`` (its place in the plan's list of links, from 0) as an arrow from ``start`` to ``end``; FI
        writes the link's words at ``label_at``."""
        lid = f"link{int(link)}"
        links = self._fill.get("links") or {}
        if self._ax is None:
            raise ValueError("call sch.canvas(width, height) first")
        if lid not in links:
            raise KeyError(f"unknown link {link}; the links are 0 to {len(links) - 1}")
        self._arrow(f"fi:{lid}", start, end, curved, ARROW_COLOUR, 0.8)
        if label_at is not None:
            self._words(f"fi:{lid}", links[lid].get("text") or "", label_at, LINK_PT, ARROW_COLOUR)
        self._calls.append({"call": "arrow", "id": lid, "start": list(self._xy(start)), "end": list(self._xy(end)),
                            "label_at": list(self._xy(label_at)) if label_at is not None else None})

    def direction(self, start, end, label_at=None):
        """The system's one direction (gravity, a flow) as an arrow from ``start`` to ``end``; FI writes its words at
        ``label_at``."""
        d = self._fill.get("direction")
        if self._ax is None:
            raise ValueError("call sch.canvas(width, height) first")
        if not d:
            raise KeyError("the plan states no direction for this system")
        self._arrow("fi:direction", start, end, 0.0, DIRECTION_COLOUR, 1.2)
        if label_at is not None:
            self._words("fi:direction", d.get("text") or "", label_at, FONT_PT, DIRECTION_COLOUR)
        self._calls.append({"call": "direction", "id": "direction", "start": list(self._xy(start)),
                            "end": list(self._xy(end)), "label_at": list(self._xy(label_at)) if label_at is not None else None})

    def label(self, part_id, x, y, ha="center", va="center"):
        """FI's text for part ``part_id`` (its name and the plan's values it carries) at ``(x, y)``."""
        part = self._part(part_id)
        if ha not in HA or va not in VA:
            raise ValueError(f"ha must be one of {HA} and va one of {VA}")
        t = self._ax.text(float(x), float(y), part.get("text") or "", fontsize=FONT_PT, ha=ha, va=va, gid=f"fi:{part_id}",
                          linespacing=1.15, color="#111111",
                          bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        self._mathtext_safe(t)
        self._calls.append({"call": "label", "id": part_id, "xy": [float(x), float(y)], "ha": ha, "va": va})

    def leader(self, part_id, start, end):
        """A thin line from a part's label to the part."""
        self._part(part_id)
        (x0, y0), (x1, y1) = self._xy(start), self._xy(end)
        self._ax.plot([x0, x1], [y0, y1], color="#777777", linewidth=0.5, gid=f"fi:{part_id}")
        self._calls.append({"call": "leader", "id": part_id, "start": [x0, y0], "end": [x1, y1]})

    def _mathtext_safe(self, t):
        """A label whose ``$...$`` matplotlib cannot draw is drawn as typed (its words unchanged, the marks shown)."""
        try:
            t.get_window_extent(self._canvas.get_renderer())
        except Exception:  # noqa: BLE001
            t.set_parse_math(False)


def _safe_builtins() -> dict:
    import builtins

    return {name: getattr(builtins, name) for name in SAFE_BUILTINS if hasattr(builtins, name)}


def facts(sch: Schematic) -> dict:
    """What the drawing shows, as facts for FI to judge: the figure's size in pixels, each visible text (tag, words, box,
    colour, size), each tagged shape's box, and each text a line or an outline of another id crosses."""
    from matplotlib.colors import to_rgba
    from matplotlib.path import Path as MplPath
    from matplotlib.spines import Spine
    from matplotlib.text import Text

    fig = sch._fig
    renderer = sch._canvas.get_renderer()
    sch._canvas.draw()
    out = {"rendered": True, "width_px": fig.bbox.width, "height_px": fig.bbox.height, "width_in": fig.get_figwidth(),
           "dpi": fig.dpi, "texts": [], "shapes": [], "crossed": []}
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    ax = sch._ax
    # the shapes drawn on the drawing (an axis' own tick lines and spines are not; the axes are off)
    shapes = [a for a in (*ax.patches, *ax.lines, *ax.collections, *ax.artists, *fig.patches, *fig.lines, *fig.artists)
              if a.get_visible() and not isinstance(a, Spine)]
    for t in texts:
        bb = t.get_window_extent(renderer)
        out["texts"].append({"gid": t.get_gid() or "", "text": t.get_text(), "box": [bb.x0, bb.y0, bb.x1, bb.y1],
                             "rgba": list(to_rgba(t.get_color())), "alpha": t.get_alpha(), "size_pt": t.get_fontsize()})
    for a in shapes:
        try:
            bb = a.get_window_extent(renderer)
            box = [bb.x0, bb.y0, bb.x1, bb.y1]
        except Exception:  # noqa: BLE001
            box = None
        out["shapes"].append({"gid": a.get_gid() or "", "type": type(a).__name__, "box": box})
    for t in texts:
        tid = t.get_gid() or ""
        bb = t.get_window_extent(renderer).expanded(*CROSS_SHRINK)
        for a in shapes:
            aid = a.get_gid() or ""
            if (aid == tid and tid) or not aid.startswith("fi:"):
                continue  # its own part's shapes and leader; an untagged shape is a finding of its own
            try:
                path = a.get_transform().transform_path(a.get_path())
            except Exception:  # noqa: BLE001
                continue
            if any(MplPath(poly).intersects_bbox(bb, filled=False)
                   for poly in path.to_polygons(closed_only=False) if len(poly) > 1):
                out["crossed"].append([tid, aid])
    return out


def run(fill_path: str, folder: str) -> dict:
    """Run ``<folder>/body.py`` with ``sch``; write the drawing, ``calls.json`` and ``facts.json`` into ``folder``."""
    folder_path = Path(folder)
    fill = json.loads(Path(fill_path).read_text(encoding="utf-8"))
    body = (folder_path / "body.py").read_text(encoding="utf-8")
    sch = Schematic(fill)
    out: dict = {"rendered": False}
    try:
        exec(compile(body, "body.py", "exec"), {"__builtins__": _safe_builtins(), "sch": sch, "math": math,
                                                "__name__": "body"})
        if sch._fig is None:
            raise ValueError("the script never called sch.canvas(width, height)")
        out = facts(sch)
        sch._canvas.print_png(str(folder_path / "drawing.png"), metadata={"Software": None})
        sch._fig.savefig(str(folder_path / "drawing_small.png"), dpi=SMALL_DPI, metadata={"Software": None})
    except Exception as exc:  # noqa: BLE001 -- the script's fault is a fact FI names to the redraw
        out = {"rendered": False, "error": (type(exc).__name__ + ": " + str(exc))[:400],
               "line": _body_line(traceback.extract_tb(exc.__traceback__))}
    (folder_path / "calls.json").write_text(json.dumps(sch._calls, indent=1), encoding="utf-8")
    (folder_path / "facts.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return out


def _body_line(tb) -> int | None:
    lines = [frame.lineno for frame in tb if frame.filename == "body.py"]
    return lines[-1] if lines else None


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
