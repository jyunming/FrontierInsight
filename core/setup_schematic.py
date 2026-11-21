"""The studied system's schematic: a set-up figure a model lays out and FI writes and checks, drawn only when the config
asks for it (``output.system_schematic: true``).

The paper, the poster and the slides may open with a schematic of the system the study simulates: its parts, its
boundaries and sources, what touches or feeds what, and its one direction (gravity, a flow), from the plan's studied
system (``protocol.system``, ``core.contract.System``). A model writes a short drawing script that calls FI's drawing calls
only (``core/setup_schematic_host.py``): it places shapes by the plan's part ids and types no word. Every text on the drawing
is FI's, written here from the plan by id (:func:`fill`): a part's name, the condition of a boundary, and the plan's reader
label and values of each id the part carries; a link's words; the direction's words.

The script runs in the quest's own interpreter (its executor, the way the figure code of a run does), never in FI's
process: FI first holds it to a closed form (:func:`guard`: the drawing calls, arithmetic, loops; no import, no attribute but
the calls, no string but a part id or a style word), then runs FI's host on it, which saves the drawing and the facts of
what it drew. FI judges those facts against the strings it wrote itself (:func:`judge`): every text tagged and equal to FI's
text for its id; every part drawn and labelled exactly once; every link an arrow; the direction pointing the way the plan
says; no label over another, crossed by another part's line or outline, off the drawing, too pale to read or under 6 pt at
3.5 in wide. A drawing that fails is drawn once more with a prompt that names what failed; one that fails again is left out,
and ``QUALITY.md`` says why in one line. One model call then reads the drawing as an image and scores it from 0 to 10; the
score is recorded in ``.fi/setup_schematic.json`` with the checks and does not decide anything yet (no threshold has been
measured). At most :data:`MAX_CALLS` model calls are made for one description of the system: the outcome, a failure
included, is kept by a digest of FI's texts, and an analysis that runs again reuses it.

It is a set-up figure like the flow of ``core/setup_flow.py`` and follows its rules (``setup_flow.figures``): drawn at the
start of the analysis into ``.fi/``, copied into ``figures/`` by the write stage, numbered before the run's figures (this
schematic first, then the flow), kept out of the checks of result figures by FI's record (its digest), given FI's caption.
Nothing here stops a study or raises out of :func:`draw_quietly` and :func:`place_quietly`."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import string
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import setup_flow as _flow
from . import setup_schematic_host as _host

FILE = "setup_schematic.png"
RECORD = "setup_schematic.json"
#: FI's work folder for the drawing (the host, FI's texts, each attempt's script, drawing and facts).
WORK = "setup_schematic"
HOST_FILE = "fi_schematic_host.py"
FILL_FILE = "fill.json"
#: FI's record of the calls of the drawing it kept, beside the call records of the run's figures.
CALLS_RECORD = "setup_schematic.calls.json"
SCHEMA = 1
#: The model calls one description of the system may cost: the drawing, at most one redraw, one reading of the image.
MAX_CALLS = 3
NODE = "setup_schematic"
SCORE_NODE = "setup_schematic.score"
#: How long the host may take on one script (seconds).
RUN_TIMEOUT_S = 180
#: The most lines a part's label has (its name, then the values it carries); the rest is "and N more".
MAX_LINES = 4
#: The most characters of one line of a label before it is cut with "...".
MAX_LINE_CHARS = 60
MIN_FONT_PT = 6.0
#: A text whose colour is lighter than this (relative luminance), or more transparent than :data:`MIN_ALPHA`, is too pale.
PALE_LUMINANCE = 0.75
MIN_ALPHA = 0.5
#: Two label boxes that overlap by more than this many pixels both ways sit on each other.
OVERLAP_PX = 2.0
#: How far (a share of the canvas) a shape may reach past its edge before it runs off the drawing.
EDGE_SHARE = 0.01
VISION_NOTE = ("recorded only: FI does not leave a drawing out by this score yet, because no threshold for it has been "
               "measured")
CAPTION = ("The studied system as the study's plan describes it: its parts, what touches or feeds what, and the values the "
           "plan sets for each (a schematic, not to scale, and not a result of the run). Every word on it is written by FI "
           "from the plan.")


# --------------------------------------------------------------------------------------------------- FI's texts, by id


def _short(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_LINE_CHARS else text[:MAX_LINE_CHARS].rstrip() + " ..."


def _carried(protocol: dict[str, Any], cid: str) -> str | None:
    """The line a part's label gives one id it carries: the plan's reader label for it, with the value or values the plan
    holds for it; ``None`` when the plan gives it no reader label (it is never shown by its code name)."""
    label = _flow._label(protocol, cid)
    if not label:
        return None
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    search = protocol.get("optimisation") if isinstance(protocol.get("optimisation"), dict) else {}
    variables = {str(v.get("name")): v for v in search.get("design_variables") or [] if isinstance(v, dict)}
    fixed = search.get("fixed") if isinstance(search.get("fixed"), dict) else {}
    if cid in grid:
        return f"{label}: {_flow._values(grid[cid])}"
    if cid in variables:
        v = variables[cid]
        if v.get("low") is not None and v.get("high") is not None:
            return f"{label}: {_flow._num(v['low'])} to {_flow._num(v['high'])} (searched)"
        return f"{label} (searched)"
    if cid in fixed:
        return f"{label}: {_flow._value(fixed[cid])}"
    for key in ("constants", "numerical_settings", "thresholds"):
        block = protocol.get(key) if isinstance(protocol.get(key), dict) else {}
        if cid in block:
            value = block[cid]
            if _flow._is_num(value) or isinstance(value, list) or (isinstance(value, str) and len(value) <= 24):
                return f"{label}: {_flow._values(value)}"
            return label
    for key in ("metrics", "derived"):
        if any(isinstance(m, dict) and str(m.get("id")) == cid for m in protocol.get(key) or []):
            return f"{label} (reported)"
    return label


def fill(protocol: Any) -> tuple[dict[str, Any] | None, str]:
    """FI's text for every id of the plan's studied system, or ``(None, why)`` when the plan states none it can use.
    ``{"parts": {id: {"name", "kind", "boundary", "text", "left_out"}}, "links": {"link<i>": {"from", "to", "text"}},
    "direction": {"text", "points"} | None}``. A part's text: its name (with the condition of a boundary), then one line for
    each id it carries that the plan gives a reader label (``left_out`` counts the others), at most :data:`MAX_LINES`."""
    from . import contract as _contract

    if not isinstance(protocol, dict) or not protocol:
        return None, "the study has no frozen plan"
    system = protocol.get("system")
    if not isinstance(system, dict) or not system.get("parts"):
        return None, "the plan does not describe the studied system"
    problems = _contract.check_system(system, protocol)
    if problems:
        return None, f"the plan's description of the studied system cannot be used ({problems[0]})"
    parts: dict[str, Any] = {}
    for part in system["parts"]:
        name = " ".join(str(part["name"]).split())
        condition = str(part.get("boundary") or "")
        if condition and condition not in name.lower():
            name = f"{name} ({condition})"
        lines, left_out = [_short(name)], 0
        for cid in part.get("carries") or []:
            line = _carried(protocol, str(cid))
            if line is None:
                left_out += 1
            else:
                lines.append(_short(line))
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES - 1] + [f"and {len(lines) - MAX_LINES + 1} more"]
        parts[str(part["id"])] = {"name": name, "kind": str(part.get("kind") or "part"), "boundary": condition or None,
                                  "text": "\n".join(lines), "left_out": left_out}
    links = {f"link{i}": {"from": str(link["from"]), "to": str(link["to"]), "text": _short(link.get("what") or "")}
             for i, link in enumerate(system.get("links") or [])}
    direction = system.get("direction")
    way = ({"text": _short(direction["what"]), "points": str(direction["points"])}
           if isinstance(direction, dict) and direction.get("what") else None)
    return {"parts": parts, "links": links, "direction": way}, ""


def expected(f: dict[str, Any]) -> dict[str, str]:
    """``{id: FI's text}`` for every id whose text may be drawn: each part, each link with words, the direction."""
    out = {pid: str(v["text"]) for pid, v in f["parts"].items()}
    out.update({lid: str(v["text"]) for lid, v in f["links"].items() if v.get("text")})
    if f.get("direction"):
        out["direction"] = str(f["direction"]["text"])
    return out


def key_of(f: dict[str, Any], prompts: dict[str, string.Template] | None = None) -> str:
    """The digest an outcome is kept by: FI's texts and the system's ids, with the text of FI's host, of this file (its
    guard and its checks) and of the prompts, so a newer FI judges and draws again rather than keep an older verdict."""
    text = "\n".join([json.dumps(f, sort_keys=True, ensure_ascii=False), _host_source(),
                      Path(__file__).read_text(encoding="utf-8"),
                      *(prompts[k].template for k in sorted(prompts or {}) if k.startswith(NODE))])
    return hashlib.sha256(f"{SCHEMA}\n{text}".encode("utf-8")).hexdigest()


def _host_source() -> str:
    return Path(_host.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------- the prompt


def _shown_chars(line: str) -> int:
    """The characters a reader sees on a line (``$...$`` marks and commands counted as one)."""
    line = re.sub(r"\\[A-Za-z]+", "x", line)
    return len(re.sub(r"[$^_{}]", "", line))


def prompt_fields(f: dict[str, Any]) -> dict[str, str]:
    """What the drawing prompt is filled with: the parts, the links and the direction, with the text FI prints for each."""
    parts = []
    for pid, v in f["parts"].items():
        kind = v["kind"] + (f", condition: {v['boundary']}" if v.get("boundary") else "")
        parts.append(f"- {pid} ({kind}): " + " / ".join(v["text"].split("\n")))
    links = [f"- {lid[4:]}: {v['from']} -> {v['to']}" + (f": {v['text']}" if v["text"] else "")
             for lid, v in f["links"].items()]
    d = f.get("direction")
    direction = (f"The system's direction: \"{d['text']}\", pointing {d['points']}. Draw it with sch.direction from a start "
                 f"to an end that points {d['points']} and place its words with label_at."
                 if d else "The plan states no direction for this system: do not call sch.direction.")
    longest = max([_shown_chars(line) for text in expected(f).values() for line in text.split("\n")] or [10])
    return {"parts": "\n".join(parts), "links": "\n".join(links) or "(none)", "direction": direction,
            "maxchars": str(longest), "maxwidth": f"{longest * 0.055:.1f}"}


def code_of(reply: str) -> str:
    """The script in a reply: the longest fenced block, else the whole reply."""
    blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", reply or "", re.S)
    return max(blocks, key=len).strip() if blocks else (reply or "").strip()


# --------------------------------------------------------------------------------------------------- FI's checks


_ALLOWED_NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass, ast.Call, ast.Name,
                  ast.Constant, ast.Tuple, ast.List, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
                  ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.Subscript, ast.Slice, ast.keyword, ast.Attribute,
                  ast.FunctionDef, ast.Return, ast.arguments, ast.arg, ast.Starred, ast.expr_context, ast.operator,
                  ast.unaryop, ast.boolop, ast.cmpop)
_WORDS = frozenset(_host.STYLES + _host.HA + _host.VA)
_LIST_METHODS = frozenset({"append", "extend"})


def guard(body: str, f: dict[str, Any]) -> list[str]:
    """Why ``body`` is not a drawing script FI runs (empty when it is): Python of the drawing calls, arithmetic and loops
    only. No import; an attribute only of ``sch`` (its calls) or ``math`` (or a list's ``append``/``extend``); no name with
    an underscore in front; no string but a part id of the system or a style word, so no word reaches the drawing but FI's."""
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        return [f"the code is not Python (line {exc.lineno}: {exc.msg})"]
    allowed_strings = set(f["parts"]) | _WORDS
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            out.append(f"line {getattr(node, 'lineno', '?')}: `{type(node).__name__}` is not allowed (only the sch calls, "
                       "arithmetic, loops and conditions)")
        elif isinstance(node, ast.Name) and node.id.startswith("_"):
            out.append(f"line {node.lineno}: the name `{node.id}` is not allowed")
        elif isinstance(node, ast.Attribute):
            base = node.value.id if isinstance(node.value, ast.Name) else None
            if not ((base == "sch" and node.attr in _host.API) or (base == "math" and not node.attr.startswith("_"))
                    or (node.attr in _LIST_METHODS and base not in ("sch", "math"))):
                out.append(f"line {node.lineno}: `.{node.attr}` is not one of the sch calls ({', '.join(_host.API)})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and node.value not in allowed_strings:
            out.append(f"line {node.lineno}: the string {str(node.value)[:40]!r} is not a part id or a style word; FI writes "
                       "every word")
    return list(dict.fromkeys(out))[:8]


def _luminance(rgba: list[float]) -> float:
    r, g, b = (float(x) for x in rgba[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _points_way(start: list[float], end: list[float], points: str) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if points in ("left", "right"):
        return abs(dx) > abs(dy) and (dx > 0) == (points == "right")
    return abs(dy) > abs(dx) and (dy > 0) == (points == "up")


def _call_points(call: dict[str, Any]) -> list[tuple[float, float]]:
    """The extreme points a shape call reaches, in the drawing's units."""
    kind = call.get("call")
    if kind == "rect":
        x, y, w, h = call["xy"]
        return [(x, y), (x + w, y + h)]
    if kind == "circle":
        cx, cy, r = call["xy"]
        r += (call.get("ring_width") or 0) / 2
        return [(cx - r, cy - r), (cx + r, cy + r)]
    if kind == "arc":
        import math

        cx, cy, r, t1, t2 = call["xy"]
        t2 = t2 if t2 >= t1 else t2 + 360
        steps = max(2, int((t2 - t1) / 5) + 1)
        return [(cx + r * math.cos(math.radians(t1 + (t2 - t1) * k / (steps - 1))),
                 cy + r * math.sin(math.radians(t1 + (t2 - t1) * k / (steps - 1)))) for k in range(steps)]
    if kind in ("line", "polygon"):
        return [tuple(p) for p in call.get("points") or []]
    if kind in ("arrow", "direction", "leader"):
        return [tuple(call["start"]), tuple(call["end"])]
    return []


def judge(facts: dict[str, Any], calls: list[dict[str, Any]], f: dict[str, Any]) -> list[str]:
    """FI's checks of a drawing, from the host's facts and calls and FI's own texts (``f``): what is wrong with it, in words
    the redraw is given (empty when it passes)."""
    if not facts.get("rendered"):
        where = f" (line {facts['line']})" if facts.get("line") else ""
        return [f"the code did not run{where}: {facts.get('error') or 'no drawing was saved'}"]
    want = expected(f)
    out: list[str] = []
    seen: dict[str, int] = {}
    boxes: list[tuple[str, list[float]]] = []
    width_in = float(facts.get("width_in") or _host.WIDTH_IN)
    w_px, h_px = float(facts.get("width_px") or 0), float(facts.get("height_px") or 0)
    for t in facts.get("texts") or []:
        gid, text = str(t.get("gid") or ""), str(t.get("text") or "")
        key = gid[3:] if gid.startswith("fi:") else None
        if key is None:
            out.append(f"a text not placed by sch.label, sch.arrow or sch.direction: {text[:40]!r}")
            continue
        if want.get(key) != text:
            out.append(f"the text drawn for `{key}` is not FI's text for it")
            continue
        seen[key] = seen.get(key, 0) + 1
        boxes.append((key, list(t.get("box") or [0, 0, 0, 0])))
        size = float(t.get("size_pt") or 0) * _host.WIDTH_IN / width_in
        if size < MIN_FONT_PT:
            out.append(f"the label of `{key}` is {size:.1f} pt at 3.5 in wide (at least {MIN_FONT_PT:g})")
        alpha = t.get("alpha")
        rgba = list(t.get("rgba") or [0, 0, 0, 1])
        if _luminance(rgba) > PALE_LUMINANCE or float(rgba[3] if len(rgba) > 3 else 1) * float(
                1 if alpha is None else alpha) < MIN_ALPHA:
            out.append(f"the label of `{key}` is drawn in a colour too pale to read")
    for pid in f["parts"]:
        if not seen.get(pid):
            out.append(f"part `{pid}` has no label (sch.label('{pid}', x, y))")
        elif seen[pid] > 1:
            out.append(f"part `{pid}` is labelled {seen[pid]} times; label it once")
    for key, n in seen.items():
        if key not in f["parts"] and n > 1:
            out.append(f"the words of `{key}` are placed {n} times; place them once")
    for s in facts.get("shapes") or []:
        if not str(s.get("gid") or "").startswith("fi:"):
            out.append(f"a shape not drawn by the sch calls ({s.get('type')})")
    drawn = {c.get("id") for c in calls if c.get("call") in ("rect", "circle", "arc", "line", "polygon")}
    for pid in f["parts"]:
        if pid not in drawn:
            out.append(f"part `{pid}` is not drawn (no shape of it)")
    arrows = {c.get("id") for c in calls if c.get("call") == "arrow"}
    for lid, link in f["links"].items():
        if lid not in arrows:
            out.append(f"link {lid[4:]} ({link['from']} -> {link['to']}) is not drawn as an arrow")
    d = f.get("direction")
    ways = [c for c in calls if c.get("call") == "direction"]
    if d:
        if not ways:
            out.append(f"the direction (\"{d['text']}\", pointing {d['points']}) is not drawn (sch.direction)")
        elif not all(_points_way(c["start"], c["end"], d["points"]) for c in ways):
            out.append(f"the direction arrow does not point {d['points']}")
        if ways and not seen.get("direction"):
            out.append("the direction's words are not placed (sch.direction(..., label_at=(x, y)))")
    canvas = next((c for c in calls if c.get("call") == "canvas"), None)
    if canvas:
        cw, ch = float(canvas["width"]), float(canvas["height"])
        tol = EDGE_SHARE * max(cw, ch)
        off = {c.get("id") for c in calls
               if any(x < -tol or y < -tol or x > cw + tol or y > ch + tol for x, y in _call_points(c))}
        out += [f"a shape of `{i}` runs off the drawing" for i in sorted(str(i) for i in off if i)]
    for i, (ki, bi) in enumerate(boxes):
        if bi[0] < -1 or bi[1] < -1 or bi[2] > w_px + 1 or bi[3] > h_px + 1:
            out.append(f"the label of `{ki}` runs off the drawing")
        for kj, bj in boxes[i + 1:]:
            if min(bi[2], bj[2]) - max(bi[0], bj[0]) > OVERLAP_PX and min(bi[3], bj[3]) - max(bi[1], bj[1]) > OVERLAP_PX:
                out.append(f"the labels of `{ki}` and `{kj}` overlap")
    for tid, aid in {(str(a)[3:], str(b)[3:]) for a, b in facts.get("crossed") or []}:
        if tid in want:
            out.append(f"the label of `{tid}` is crossed by a line or an outline of `{aid}`")
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------------------------------- the analysis stage's call

Chat = Callable[[str], Awaitable[str]]
ChatImage = Callable[[str, bytes], Awaitable[str]]
Run = Callable[[list[str]], Awaitable[tuple[int, str, str]]]


def _work(root: Path) -> Path:
    return root / ".fi" / WORK


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _forget(root: Path, old: dict[str, Any] | None, *, all_files: bool = False) -> None:
    """Take FI's earlier schematic out of ``figures/`` and ``.fi/`` (only the files its record says it drew)."""
    _flow._forget_copy(root, old)
    (root / ".fi" / FILE).unlink(missing_ok=True)
    (root / ".fi" / "figure_records" / CALLS_RECORD).unlink(missing_ok=True)
    if all_files:
        (root / ".fi" / RECORD).unlink(missing_ok=True)
        shutil.rmtree(_work(root), ignore_errors=True)


async def _attempt(root: Path, n: int, body: str, f: dict[str, Any], run: Run) -> tuple[list[str], dict[str, Any]]:
    """Hold ``body`` to :func:`guard`, run it in the quest's interpreter and judge what it drew: ``(findings, facts)``."""
    folder = _work(root) / f"attempt{n}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "body.py").write_text(body, encoding="utf-8")
    refused = guard(body, f)
    if refused:
        return refused, {"rendered": False, "refused": refused}
    rel = lambda p: p.relative_to(root).as_posix()  # noqa: E731 -- from the quest folder (a container mounts it)
    rc, _out, err = await run([rel(_work(root) / HOST_FILE), rel(_work(root) / FILL_FILE), rel(folder)])
    facts = _read(folder / "facts.json")
    calls = _read(folder / "calls.json")
    if not isinstance(facts, dict):
        return [f"FI's drawing host did not finish (exit {rc}: {(err or '').strip()[-200:]})"], {"rendered": False}
    return judge(facts, calls if isinstance(calls, list) else [], f), facts


def _score(reply: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except ValueError:
        parsed = {}
    axes = {k: parsed.get(k) for k in ("clarity", "physical_plausibility", "readability")
            if isinstance(parsed.get(k), (int, float)) and not isinstance(parsed.get(k), bool) and 0 <= parsed[k] <= 10}
    if not axes:
        return {"scored": False, "why": "the reply held no score FI could read"}
    problems = [str(p)[:200] for p in parsed.get("problems") or [] if isinstance(p, str)][:4]
    return {"scored": True, **axes, "score": round(sum(axes.values()) / len(axes), 2), "problems": problems}


async def draw(root: Path, *, on: bool, run_figures: list[str] | None, chat: Chat, chat_image: ChatImage, run: Run,
               prompts: dict[str, string.Template], log: Any = None) -> dict[str, Any] | None:
    """Draw the studied system's schematic for the quest at ``root`` into ``.fi/`` and write its record; returns the record
    (``None`` when the config does not ask for it: nothing is drawn, no model is called, and an earlier drawing is taken
    away). Raises on a fault of FI's (:func:`draw_quietly` is the analysis stage's call)."""
    from . import frozen_protocol as _frozen

    root = Path(root)
    old = _flow._record(root, RECORD)
    if not on:
        if old is not None:
            _forget(root, old, all_files=True)
        return None
    protocol_sha = _flow._digest(_flow._protocol_file(root))
    record: dict[str, Any] = {"schema": SCHEMA, "file": FILE, "drawn": False, "protocol_sha256": protocol_sha}
    f, why = fill(_frozen.protocol_of(root))
    if f is not None and FILE in {Path(str(x)).name for x in run_figures or []}:
        f, why = None, f"the run drew a figure of its own named {FILE}"
    if f is None:
        _forget(root, old)
        _write(root / ".fi" / RECORD, {**record, "why": why, "done": True})
        return _flow._record(root, RECORD)
    key = key_of(f, prompts)
    if old and old.get("key") == key and old.get("done"):
        # the same description of the system: its outcome is kept, a failure too, and no model is called again
        kept = {k: v for k, v in old.items() if k not in ("placed",)}
        kept["protocol_sha256"] = protocol_sha
        png = _work(root) / "kept.png"
        if kept.get("drawn"):
            if _flow._digest(png) != kept.get("png_sha256"):
                raise RuntimeError("FI's kept drawing of the studied system is gone")
            if _flow._digest(root / ".fi" / FILE) != kept["png_sha256"]:
                shutil.copyfile(png, root / ".fi" / FILE)
            calls = _work(root) / "kept.calls.json"
            target = root / ".fi" / "figure_records" / CALLS_RECORD
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(calls, target)
        _write(root / ".fi" / RECORD, kept)  # shown from .fi/ until the write stage copies it again, as the flow is
        return kept
    if (old and old.get("key") == key and not old.get("done") and (old.get("checks") or {}).get("passed")
            and _flow._digest(_work(root) / "kept.png") == old.get("png_sha256")):
        # stopped while the drawing that passed FI's checks was read: it is kept, unscored, and no call is made
        kept = {k: v for k, v in old.items() if k not in ("placed",)}
        kept.update(protocol_sha256=protocol_sha, drawn=True, done=True)
        kept["vision"] = {**(kept.get("vision") or {}), "scored": False,
                          "why": "the study stopped while the drawing was read; its call is counted"}
        shutil.copyfile(_work(root) / "kept.png", root / ".fi" / FILE)
        target = root / ".fi" / "figure_records" / CALLS_RECORD
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_work(root) / "kept.calls.json", target)
        _write(root / ".fi" / RECORD, kept)
        return kept
    # An unfinished outcome for the same system (a study stopped mid-way and resumed) goes on with the calls it has left:
    # every call is counted on disk before it is made, so a call in flight when the study stopped is counted too.
    spent = int(old.get("calls_made") or 0) if old and old.get("key") == key and not old.get("done") else 0
    earlier = [m for m in (old or {}).get("model_calls") or []] if spent else []
    _forget(root, old, all_files=True)
    work = _work(root)
    work.mkdir(parents=True, exist_ok=True)
    (work / HOST_FILE).write_text(_host_source(), encoding="utf-8")
    _write(work / FILL_FILE, f)
    record.update(key=key, done=False, calls_made=spent, model_calls=earlier, attempts=[], left_out_values=sum(
        v["left_out"] for v in f["parts"].values()))
    prompt = prompts["setup_schematic"].substitute(**prompt_fields(f))
    _write(root / ".fi" / RECORD, record)  # the key first: a fault after a call never lets an analysis call again
    findings: list[str] = ["no call was left for this system"]
    passed: Path | None = None
    body = ""
    for n in (1, 2):
        if record["calls_made"] >= MAX_CALLS - 1:
            break  # the last call is kept for reading the drawing
        if n == 1:
            ask = prompt
        else:
            ask = prompts["setup_schematic_redraw"].substitute(
                prompt=prompt, code=body, findings="\n".join(f"- {x}" for x in findings[:8]))
        record["calls_made"] += 1
        _write(root / ".fi" / RECORD, record)  # counted before it is made
        try:
            reply = await chat(ask)
        except Exception as exc:  # noqa: BLE001 -- a failed call is one of the calls; the study goes on without it
            record["model_calls"].append({"node": NODE, "ok": False, "error": type(exc).__name__})
            findings = [f"the model call failed ({type(exc).__name__})"]
            break
        record["model_calls"].append({"node": NODE, "ok": True, "chars": len(reply or "")})
        _write(root / ".fi" / RECORD, record)
        body = code_of(reply)
        findings, facts = await _attempt(root, n, body, f, run)
        record["attempts"].append({"attempt": n, "passed": not findings, "findings": findings[:20],
                                   "texts": len(facts.get("texts") or [])})
        if not findings:
            passed = work / f"attempt{n}"
            break
    if passed is None:
        record["why"] = (f"FI's check of the drawing failed {len(record['attempts'])} time(s) ({findings[0]})" if findings
                         and record["attempts"] else f"it could not be drawn ({findings[0] if findings else 'no reply'})")
        record["done"] = True
        _write(root / ".fi" / RECORD, record)
        if log is not None:
            log.info("[analyze] no schematic of the studied system: %s", record["why"])
        return record
    shutil.copyfile(passed / "drawing.png", work / "kept.png")
    calls = _read(passed / "calls.json") or []
    _write(work / "kept.calls.json", {"schema": SCHEMA, "figure": FILE, "kind": "setup_schematic", "calls": calls})
    shutil.copyfile(work / "kept.png", root / ".fi" / FILE)
    target = root / ".fi" / "figure_records" / CALLS_RECORD
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "kept.calls.json", target)
    record.update(drawn=True, png_sha256=_flow._digest(work / "kept.png"), caption=CAPTION,
                  checks={"passed": True, "attempt": len(record["attempts"]), "findings": []})
    record["vision"] = {"gating": False, "note": VISION_NOTE}
    record["calls_made"] += 1
    _write(root / ".fi" / RECORD, {**record, "drawn": False})  # counted before it is made; shown only once scored
    try:
        reply = await chat_image(prompts["setup_schematic_score"].substitute(**_score_fields(f)),
                                 (passed / "drawing_small.png").read_bytes())
        record["vision"].update(_score(reply))
        record["model_calls"].append({"node": SCORE_NODE, "ok": True, "chars": len(reply or "")})
    except Exception as exc:  # noqa: BLE001 -- the score is recorded, never needed
        record["vision"].update(scored=False, why=f"the model could not read the drawing ({type(exc).__name__}: "
                                                  f"{str(exc)[:160]})")
        record["model_calls"].append({"node": SCORE_NODE, "ok": False, "error": type(exc).__name__})
    record["done"] = True
    _write(root / ".fi" / RECORD, record)
    if log is not None:
        log.info("[analyze] a model laid out the schematic of the studied system and FI's checks passed (Figure 1 of "
                 "the paper); its score when read as an image: %s", record["vision"].get("score", "not scored"))
    return record


def _score_fields(f: dict[str, Any]) -> dict[str, str]:
    names = {pid: v["name"] for pid, v in f["parts"].items()}
    parts = "; ".join(f"{v['name']} ({v['kind']})" for v in f["parts"].values())
    links = "; ".join(f"{names[v['from']]} -> {names[v['to']]}" + (f" ({v['text']})" if v["text"] else "")
                      for v in f["links"].values()) or "(none)"
    d = f.get("direction")
    return {"parts": parts, "links": links, "direction": f"{d['text']}, pointing {d['points']}" if d else "(none)"}


async def draw_quietly(root: Path, *, on: bool, run_figures: list[str] | None, chat: Chat, chat_image: ChatImage,
                       run: Run, prompts: dict[str, string.Template], log: Any = None) -> None:
    """:func:`draw` for the analysis stage: a fault is recorded (one ``QUALITY.md`` line) and logged, never raised."""
    old = None
    try:
        old = _flow._record(root, RECORD)
        await draw(Path(root), on=on, run_figures=run_figures, chat=chat, chat_image=chat_image, run=run,
                   prompts=prompts, log=log)
    except Exception as exc:  # noqa: BLE001 -- the schematic is extra: a paper without it is the paper without it
        try:
            now = _flow._record(root, RECORD) or old  # what this draw wrote before the fault (its key, its calls)
            _forget(Path(root), now)
            kept = {k: now[k] for k in ("key", "calls_made", "model_calls", "attempts") if now and k in now}
            _write(Path(root) / ".fi" / RECORD, {"schema": SCHEMA, "file": FILE, "drawn": False, "done": True, **kept,
                                                 "why": f"FI could not draw it ({type(exc).__name__})"})
        except Exception:  # noqa: BLE001
            pass
        if log is not None:
            log.warning("[analyze] the schematic of the studied system could not be drawn (%r); the study goes on "
                        "without it", exc)


# --------------------------------------------------------------------------------------------------- what the rest of FI reads


def shown(root: Any) -> str | None:
    """The file name of the schematic when the study shows it (``setup_flow.shown`` with this record), else ``None``."""
    return _flow.shown(root, RECORD)


def quality_line(root: Any) -> str | None:
    """The one line ``QUALITY.md`` gives when the config asked for the schematic and the paper has none, saying why;
    ``None`` when it is in the paper or was not asked for."""
    record = _flow._record(root, RECORD)
    if not record:
        return None
    if record.get("drawn"):
        if record.get("placed") is False:
            return f"No schematic of the studied system in the paper: {record.get('why') or 'FI could not put it in figures/'}."
        return None
    return f"No schematic of the studied system in the paper: {record.get('why') or 'FI could not draw it'}."


def place_quietly(root: Path, log: Any = None) -> None:
    """The write stage's call: the schematic, when the study shows it, copied into ``figures/``. Never raises."""
    _flow.place_quietly(root, log, RECORD)


__all__ = ["CALLS_RECORD", "FILE", "MAX_CALLS", "NODE", "RECORD", "SCORE_NODE", "code_of", "draw", "draw_quietly",
           "expected", "fill", "guard", "judge", "key_of", "place_quietly", "prompt_fields", "quality_line", "shown"]
