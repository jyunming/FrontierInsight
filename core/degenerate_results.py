"""Results and figures that show nothing are broken, not findings.

After the run, before the paper, from FI's own records (the per-cell record and the figure records the run's plotting
recorder wrote; never from the script's claims):

* a figure panel that draws no data (no line, no bar, no image), and a plotted series whose points are all at one x
  although it has several (a trend drawn against a column that never varied);
* a reported quantity that is exactly the same at every setting while the study varies its settings and other reported
  quantities do vary (a quantity the simulation does not compute from its inputs);
* a reported quantity stuck at its own largest or smallest value for most settings (a clamp, not a measurement).

Only what the analysis reports is judged (a quantity nobody reports is a diagnostic), a setting echoed back is not a
result, and a value of exactly 0 or 1 is never read as stuck at a bound (an indicator or a probability is allowed to be
there). A quantity the plan expects to stay put (what an invariant, symmetry or second-implementation check measures, a declared
bound) is exempt, and one whose line the code marks ``# constant by design: <reason>`` is accepted and noted in the
limitations, never dropped (the mark must sit on the assignment of a named constant the returned value uses, with a reason
that names a source, and the different-model code review confirms or rejects it; unreviewed, it is noted as the writer's
own declaration). The all-identical case is :mod:`core.flat_output`'s; this is the quantity that is flat while others are not.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import grid_keys as _grid_keys
from . import reported_settings as _rs

#: A series is read only when it has at least this many points (a two-point line is a reference line).
MIN_POINTS = 3
#: A quantity is read only when the study has this many settings.
MIN_CELLS = 3
#: A bound must be reached by at least this share of the settings, and at least this many of them.
SATURATED_SHARE = 0.5
SATURATED_MIN_CELLS = 6
SATURATED_MIN_DISTINCT = 3


@dataclass
class FigureProblem:
    figure: str
    panel: str
    kind: str  # "empty" | "one_x"
    detail: str = ""

    def says(self, supplied: list[str] | None = None) -> str:
        where = f"`{self.figure}`" + (f", panel \"{self.panel}\"" if self.panel else "")
        if self.kind == "empty":
            return f"{where} draws no data"
        return f"{where} plots {self.detail}, so it shows no trend"


@dataclass
class FigureGap:
    """A series of a drawn figure that has no point at some settings the figure otherwise shows (or has no value there)."""

    figure: str
    panel: str
    series: str
    absent: list[float]  # the x values with no point
    why: str  # "no value" (None / NaN at that x) or "not plotted" (the panel's other series have a point there)

    def says(self) -> str:
        where = f"`{self.figure}`" + (f", panel \"{self.panel}\"" if self.panel else "")
        at = ", ".join(f"{x:g}" for x in self.absent[:8]) + (" ..." if len(self.absent) > 8 else "")
        what = f"the series \"{self.series}\"" if self.series and not self.series.startswith("_") else "a series"
        if self.why == "no value":
            return f"{where}: {what} has no value at x = {at}, so the figure shows no point there"
        return f"{where}: {what} has no point at x = {at}, where the panel's other series do"


@dataclass
class QuantityProblem:
    name: str
    kind: str  # "constant" | "saturated"
    value: float
    cells: int
    at: int

    def says(self, supplied: list[str] | None = None) -> str:
        if self.kind == "constant":
            return (f"`{self.name}` is {self.value:g} at every one of the {self.cells} settings although other results "
                    "change with them")
        return (f"`{self.name}` is {self.value:g} at {self.at} of the {self.cells} settings, the largest or smallest value it "
                "takes anywhere")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


_DASHED = ("--", ":", "-.", "dashed", "dotted", "dashdot")


def _is_reference(line: Any) -> bool:
    """A recorded line that is a reference drawn on purpose: ``axhline`` / ``axvline``, or a dashed line at one value
    with no markers. A constant drawn with markers is a measured series (each marker is a setting), not a reference."""
    if not isinstance(line, dict):
        return False
    if line.get("kind") in ("hline", "vline"):
        return True
    y = line.get("y")
    return (line.get("kind") == "data" and line.get("linestyle") in _DASHED and isinstance(y, list) and len(y) > 1
            and str(line.get("marker") or "None") in ("None", "none", "", " ")
            and len(_rs._distinct([float(v) for v in y])) == 1)


def _has_data(panel: Any) -> bool:
    if not isinstance(panel, dict):
        return False
    lines = panel.get("lines") if isinstance(panel.get("lines"), list) else []
    return bool(panel.get("draws_other")) or any(not _is_reference(line) for line in lines)


def _shares_place_with_data(panels: list[Any], i: int) -> bool:
    """Another panel in the same place holds data (a twin axis): the same cell of a grid, or the same box on the figure
    for axes made without a grid."""
    me = panels[i] if isinstance(panels[i], dict) else {}
    keys = [k for k in ("grid", "place") if me.get(k) is not None]
    return bool(keys) and any(j != i and isinstance(p, dict) and any(p.get(k) == me[k] for k in keys) and _has_data(p)
                              for j, p in enumerate(panels))


#: A panel's series are compared point for point only when the longer of the two has at most this many points (a grid of
#: settings, not a curve drawn on a dense axis).
MAX_GRID_POINTS = 12


def _numbers(values: Any) -> list[float]:
    out: list[float] = []
    for v in values if isinstance(values, list) else []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f not in out:
            out.append(f)
    return out


def figure_gaps(records_dir: Path, figures: list[str]) -> list[FigureGap]:
    """Where a series of a figure has no point at a setting: from the figure's own record (``<stem>.seed0.json``), two ways.

    * the line holds a missing value (None / NaN) at an x: matplotlib draws nothing there (``missing_x``, counted by the plot
      recorder from the plotted data);
    * the series' x values are a strict part of another series' x values in the same panel, both few enough to be a grid of
      settings (at most :data:`MAX_GRID_POINTS`): the panel's other series have a point where this one has none.

    Reference lines are never compared. Nothing here changes the figure or the results; it says what the figure does not show."""
    out: list[FigureGap] = []
    for name in sorted(set(figures)):
        panels = _read(records_dir / f"{Path(name).stem}.seed0.json").get("axes")
        for panel in panels if isinstance(panels, list) else []:
            if not isinstance(panel, dict):
                continue
            lines = [ln for ln in (panel.get("lines") if isinstance(panel.get("lines"), list) else [])
                     if isinstance(ln, dict) and ln.get("kind") == "data" and not _is_reference(ln)]
            title = str(panel.get("title") or "")
            # Compared at 9 significant digits: the same setting written as 0.1 and as np.arange's 0.1000000000000001.
            xs = [{float(f"{v:.9g}") for v in _numbers(ln.get("x"))} for ln in lines]
            for k, ln in enumerate(lines):
                label = str(ln.get("label") or "")
                gone = _numbers(ln.get("missing_x"))
                if gone:
                    out.append(FigureGap(Path(name).name, title, label, gone, "no value"))
                    continue
                if not xs[k] or label.startswith("_"):
                    continue
                lacking: set[float] = set()
                for j, other in enumerate(lines):
                    if j != k and not str(other.get("label") or "").startswith("_") and xs[k] < xs[j] and len(xs[j]) <= MAX_GRID_POINTS:
                        lacking |= xs[j] - xs[k]
                if lacking:
                    out.append(FigureGap(Path(name).name, title, label, sorted(lacking), "not plotted"))
    return out


def gap_note(gaps: list[FigureGap]) -> str:
    """For the writer: where a figure has no point, and what that means for the sentences about it."""
    shown = "; ".join(g.says() for g in gaps[:6])
    return (f"Where the figures show no point: {shown}. Do not say a figure shows, confirms or reaches a value at those "
            "settings, and do not write \"as shown in Figure N\" for them: where the results hold the value, give it in the text or a "
            "table; where they do not, say it was not computed; and say in the caption which settings the figure has no "
            "point for.")


def figure_problems(records_dir: Path, figures: list[str]) -> list[FigureProblem]:
    """The empty panels and the single-x series of the figures in ``figures`` (file names), from their records."""
    out: list[FigureProblem] = []
    for name in sorted(set(figures)):
        stem = Path(name).stem
        main, lines = _read(records_dir / f"{stem}.json"), _read(records_dir / f"{stem}.seed0.json")
        panels = lines.get("axes") if isinstance(lines.get("axes"), list) else []
        shown = main.get("axes") if isinstance(main.get("axes"), list) else []
        if not panels and not shown:
            continue
        for i, panel in enumerate(panels or [{} for _ in shown]):
            labelled = shown[i] if i < len(shown) and isinstance(shown[i], dict) else {}
            title = str(panel.get("title") or labelled.get("title") or "")
            named = bool(title or panel.get("xlabel") or panel.get("ylabel") or labelled.get("xlabel"))
            drawn = panel.get("lines") if isinstance(panel.get("lines"), list) else []
            # A panel that only holds a legend (shared by the other panels) is not empty.
            if (not drawn and not labelled.get("series") and not panel.get("draws_other") and not panel.get("legend")
                    and named):
                out.append(FigureProblem(Path(name).name, title, "empty"))
                continue
            # A panel whose only lines are reference lines (a threshold, a known value) holds no result: the axes are
            # there and the data are not. Read from the lines themselves, since an unlabelled data line is not among the
            # labelled series; and not when another panel sits in the same place (a twin axis carries the data).
            if (drawn and all(_is_reference(line) for line in drawn) and not panel.get("draws_other") and named
                    and not _shares_place_with_data(panels, i)):
                out.append(FigureProblem(Path(name).name, title, "empty"))
                continue
            for line in drawn:
                x, y = line.get("x"), line.get("y")
                if line.get("kind") != "data" or not isinstance(x, list) or len(x) < MIN_POINTS or not isinstance(y, list):
                    continue
                if len(_rs._distinct([float(v) for v in x])) == 1 and len(_rs._distinct([float(v) for v in y])) > 1:
                    label = str(line.get("label") or "")
                    out.append(FigureProblem(Path(name).name, title, "one_x",
                                             f"{len(x)} points all at x = {float(x[0]):g}"
                                             + (f" ({label})" if label and not label.startswith("_") else "")))
    return out


def _means(cells: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Each returned quantity's mean at every setting, only for those every setting returned."""
    names = set.intersection(*[set(c["metrics"]) for c in cells]) if cells else set()
    return {n: [sum(c["metrics"][n]) / len(c["metrics"][n]) for c in cells] for n in sorted(names)}


def keys_in(value: Any) -> set[str]:
    """Every dictionary key, at any depth, of what the analysis reported."""
    out: set[str] = set()
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(str(k))
            out |= keys_in(v)
    elif isinstance(value, list):
        for v in value[:300]:
            out |= keys_in(v)
    return out


def quantity_problems(cells: list[dict[str, Any]], reported: set[str], exempt: set[str] | frozenset[str] = frozenset()
                      ) -> list[QuantityProblem]:
    """The constant and the stuck quantities among those the analysis reported (``reported``: the keys it printed or wrote).
    A number per run is judged over the settings and a number per value over the settings and values, each over the records
    that hold it (:func:`core.in_run.groups`)."""
    from . import in_run as _in_run

    return [q for group in _in_run.groups(cells) for q in _quantity_problems_in(group, reported, exempt)]


def _quantity_problems_in(cells: list[dict[str, Any]], reported: set[str], exempt: set[str] | frozenset[str]
                          ) -> list[QuantityProblem]:
    varying = _rs._varying(cells)
    seen: dict[str, set[str]] = {}
    for c in cells:   # the values compared inside each run vary a record too, numbers or not (a procedure)
        for k, v in (c.get("level") or {}).items():
            seen.setdefault(str(k), set()).add(str(v))
    for k, vs in seen.items():
        if len(vs) > 1:
            varying.setdefault(k, sorted(vs))  # type: ignore[arg-type]
    if len(cells) < MIN_CELLS or not varying:
        return []
    means = {n: v for n, v in _means(cells).items() if all(math.isfinite(x) for x in v)}
    axes = list(varying)

    def is_reported(name: str) -> bool:
        return any(name == k or _rs.names_setting(name, k) or _rs.names_setting(k, name) for k in reported)

    kept = {n: v for n, v in means.items()
            if n not in exempt and is_reported(n) and not any(_rs.names_setting(n, a) for a in axes)}
    moving = [n for n, v in kept.items() if len(_rs._distinct(v)) > 1]
    out: list[QuantityProblem] = []
    for name, values in kept.items():
        distinct = _rs._distinct(values)
        if len(distinct) == 1:
            if moving:
                out.append(QuantityProblem(name, "constant", distinct[0], len(values), len(values)))
            continue
        lo, hi = min(values), max(values)
        for bound in (hi, lo):
            at = sum(1 for v in values if _rs.same(v, bound))
            if (len(values) >= SATURATED_MIN_CELLS and len(distinct) >= SATURATED_MIN_DISTINCT
                    and at / len(values) >= SATURATED_SHARE and not (_rs.same(bound, 0.0) or _rs.same(bound, 1.0))):
                out.append(QuantityProblem(name, "saturated", bound, len(values), at))
                break
    return out


MARK = re.compile(r"#\s*constant by design\s*:?\s*(.*)$", re.IGNORECASE)
#: The kinds of check whose measured quantity is expected to be the same everywhere (a worst violation of an invariant).
_EXPECTED_KINDS = ("invariant", "symmetry", "second_implementation")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def expected_constants(protocol: dict[str, Any] | None) -> set[str]:
    """Names the plan itself expects to stay put: what a check of an invariant, a symmetry or a second implementation measures,
    and the keys of the plan's declared bounds or limits."""
    from . import oracle_check as _oracle

    out: set[str] = set()
    for o in _oracle.declared(protocol):
        if _oracle.kind_of(o) in _EXPECTED_KINDS:
            out |= set(_IDENT.findall(str(o.get("measure") or "")))
    for key in ("bounds", "limits"):
        block = (protocol or {}).get(key) if isinstance(protocol, dict) else None
        if isinstance(block, dict):
            out |= {str(k) for k in block}
    return out


#: A reason must say where the constant comes from: at least this many words.
MIN_REASON_WORDS = 3
_ENTRIES = ("run_cell", "run_trial")


@dataclass
class Mark:
    """A ``# constant by design: <reason>`` on the assignment of a named constant."""

    script: str
    line: int
    name: str
    reason: str
    scope: str  # "" for a module-level constant, else the function it is local to

    def key(self) -> str:
        """What names this mark in the review's checklist and verdicts: where it is, not only what it is called."""
        return f"{self.script}:{self.line}:{self.scope or '-'}:{self.name}"

    def valid(self) -> bool:
        return len(self.reason.split()) >= MIN_REASON_WORDS

    def asks(self) -> str:
        """The requirement the second model confirms or rejects (no number in it)."""
        return (f"the constant `{self.name}` ({self.script} line {self.line}{', in ' + self.scope if self.scope else ''}) is declared constant by design (\"{self.reason}\"): "
                "it is a genuine fixed constant of the study that the reason names a source for, and a returned quantity may "
                "rest on it; it is not a stand-in for a result the code should compute")


def design_marks(files: dict[str, str]) -> list[Mark]:
    """Every ``# constant by design: <reason>`` that sits on the line of an assignment to one name (a module-level or a local
    constant) and has a reason of at least :data:`MIN_REASON_WORDS` words. A mark anywhere else (a ``return`` line, a
    comment of its own) is no mark."""
    import ast

    out: list[Mark] = []
    for script, text in files.items():
        lines = text.splitlines()
        marked = {i + 1: m.group(1).strip() for i, line in enumerate(lines) if (m := MARK.search(line))}
        if not marked:
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue

        def visit(node: ast.AST, scope: str) -> None:
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
                target = None
                if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    target = child.targets[0].id
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name) and child.value is not None:
                    target = child.target.id
                if target:
                    for ln in range(child.lineno, (getattr(child, "end_lineno", child.lineno) or child.lineno) + 1):
                        if ln in marked:
                            mark = Mark(script, child.lineno, target, marked[ln], scope)
                            if mark.valid():
                                out.append(mark)
                            break
                visit(child, inner)

        visit(tree, "")
    return out


def _uses(files: dict[str, str], mark: Mark, key: str) -> bool:
    """Whether the dictionary ``run_cell`` / ``run_trial`` return holds ``key`` computed from the marked constant: through
    assignments and through the functions that use it, read from each function's own statements. Anything not read in full is
    not a use."""
    import ast

    functions: dict[str, tuple[ast.AST, str]] = {}
    defined: dict[str, set[str]] = {}
    for script, text in files.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return False
        defined[script] = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, (node, script))

    def refers(fn: ast.AST, script: str) -> bool:
        """Whether the function's own statements mean this mark's constant by its name."""
        if mark.scope:
            return script == mark.script and fn.name == mark.scope and any(  # type: ignore[attr-defined]
                isinstance(n, ast.Name) and n.id == mark.name for n in _grid_keys.own_scope_nodes(fn))
        if script == mark.script:
            return any(isinstance(n, ast.Name) and n.id == mark.name for n in _grid_keys.own_scope_nodes(fn))
        # another file means it only by importing it (not by defining a constant of the same name itself)
        return mark.name not in defined.get(script, set()) and any(
            (isinstance(n, ast.Name) and n.id == mark.name) or (isinstance(n, ast.Attribute) and n.attr == mark.name)
            for n in _grid_keys.own_scope_nodes(fn))

    def names(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def calls(nodes: list[ast.AST]) -> set[str]:
        return {(n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else "")
                for n in nodes if isinstance(n, ast.Call)} - {""}

    affected = {n for n, (fn, script) in functions.items() if refers(fn, script)}
    grew = True
    while grew:
        grew = False
        for n, (fn, _script) in functions.items():
            if n not in affected and calls(_grid_keys.own_scope_nodes(fn)) & affected:
                affected.add(n)
                grew = True
    for entry in _ENTRIES:
        if entry not in functions:
            continue
        fn, script = functions[entry]
        own: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            own.append(node)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.extend(ast.iter_child_nodes(node))
        tainted = {mark.name} if refers(fn, script) else set()
        for _ in range(8):
            before = len(tainted)
            for node in own:
                if isinstance(node, ast.Assign) and (names(node.value) & tainted or calls(list(ast.walk(node.value))) & affected):
                    tainted |= {t.id for tt in node.targets for t in ast.walk(tt) if isinstance(t, ast.Name)}
            if len(tainted) == before:
                break
        returns = [n for n in own if isinstance(n, ast.Return)]
        if not returns or not all(isinstance(r.value, ast.Dict) for r in returns):
            continue
        for r in returns:
            for k, v in zip(r.value.keys, r.value.values):
                if isinstance(k, ast.Constant) and k.value == key and (
                        names(v) & tainted or calls(list(ast.walk(v))) & affected):
                    return True
    return False


CONFIRMED, REJECTED, UNCONFIRMED = "confirmed", "rejected", "unconfirmed"


def confirmation(quest_root: Path, code_dir: Path, mark: Mark) -> str:
    """What the different-model code review said of this mark: ``confirmed``, ``rejected``, or ``unconfirmed`` when no other
    model read this version of the code with the mark in it (the review was not run, or the reader is the writer)."""
    from . import code_review as _review

    record = _review.load(quest_root)
    if record.get("read_version") != _review.version(code_dir):
        return UNCONFIRMED
    for v in record.get("verdicts") or []:
        if isinstance(v, dict) and v.get("kind") == _review.MARK_KIND and v.get("name") == mark.key():
            return CONFIRMED if v.get("status") == _review.IMPLEMENTED else REJECTED
    return UNCONFIRMED


def accepted_marks(files: dict[str, str], names: list[str], quest_root: Path, code_dir: Path) -> dict[str, tuple[Mark, str]]:
    """``{quantity: (its mark, confirmed | unconfirmed)}`` for each of ``names`` whose value is computed from a valid mark that
    the other model did not reject. A rejected mark counts as none."""
    marks = design_marks(files)
    out: dict[str, tuple[Mark, str]] = {}
    for name in names:
        for mark in marks:
            if _uses(files, mark, name):
                state = confirmation(quest_root, code_dir, mark)
                if state != REJECTED:
                    out[name] = (mark, state)
                    break
    return out


def directive(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    """What to tell the repair: plain, naming each figure and quantity. The instruction is the last thing in it."""
    lines = [f"- {p.says()}" for p in (*quantities[:6], *figures[:6])]
    text = "FI looked at what the run produced, and some of it shows nothing:\n" + "\n".join(lines)
    if quantities:
        text += ("\n\nFor each quantity named above: if that is not what the model should give, compute it from the inputs "
                 "(no fixed number, no clamp); if it is expected, keep it and mark the line with "
                 "`# constant by design: <reason>`.")
    if figures:
        text += ("\n\nDraw each panel from the saved results (every panel needs data, each point at the setting it was run at); "
                 "a panel with nothing to show is removed, not left empty.")
    return text + " Change nothing else."


def plain_note(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    return "the run's results show nothing in places (" + "; ".join(p.says() for p in (*quantities[:3], *figures[:3])) + ")"


def writer_note(figures: list[FigureProblem], quantities: list[QuantityProblem]) -> str:
    gone = []
    if quantities:
        gone.append("the quantities " + ", ".join(f"`{q.name}`" for q in quantities[:6]))
    if figures:
        gone.append("the figures " + ", ".join(sorted({f"`{p.figure}`" for p in figures})))
    return (f"{plain_note(figures, quantities)}. {' and '.join(gone).capitalize()} {'are' if len(gone) > 1 or len(quantities) > 1 else 'is'} "
            "left out of the results. Do not report, describe or quote anything from "
            f"{'them' if len(gone) > 1 or len(quantities) + len({p.figure for p in figures}) > 1 else 'it'}: say plainly, in the "
            "limitations, that it was left out because it showed nothing.")
