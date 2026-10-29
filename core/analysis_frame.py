"""The analysis FI writes: ``code/derive.py`` (FI's frame and the study's bodies) and ``code/experiment.py`` (all FI's).

For a plan in the contract form (``protocol.form: contract``) the simulation returns the results of each trial and FI keeps the
record of every trial (``raw/trials.json``). What is left for an analysis is small: read that record, work out the quantities the plan
declares beyond it, and print them. FI does the first and the last itself, and does the middle from the plan where the plan
says how:

* a quantity with a ``formula`` is worked out by FI's formula calculator (:func:`core.oracle_forms.evaluate`), at each setting or
  once for the study; no code is written for it;
* a quantity with no formula is a function of the study's own in ``derive.py``: ``derive_<id>(table)``, or for a reference
  ``reference_<id>(setting)``, and ``figure_<id>(fig, table)`` for a figure. The model writes only these bodies; FI wrote the
  rest of the file (:func:`derive_source`) and ``experiment.py`` (:data:`EXPERIMENT`), which calls the bodies, checks what each
  returned (:func:`check_return`, the text of which is in ``experiment.py``), keeps the numbers in ``data/results/derived.json``
  and prints ``RESULT_JSON`` itself, in the shape the analysis script has always been asked for, so what reads it keeps working.

What a body may return is checked at once, one body at a time: one body's failure does not stop the others.

======================================  ====================================================================================
a body returns                          FI's verdict
======================================  ====================================================================================
a finite number                         kept
``None`` or ``table.missing(why)``      kept as "does not exist here"; ``None`` alone, with no reason, is refused
a mapping                               its keys must be the study's settings, or every value one input compared inside each run
                                        has; an unknown key is refused by name (this is where a setting named wrongly is stopped)
anything else, a bool                   refused, said in words
NaN or an infinity                      a result, reference or physical bound: failed; any other role: kept, said (disclosed);
                                        a value compared inside each run may be NaN where the result does not exist
an exception                            failed, with the line of ``derive.py`` it came from
the stub FI wrote (``NotYet``)          not written: said, never an error
======================================  ====================================================================================

The exit code is 1 when a body of a counting role (:data:`core.numeric_warnings.COUNTING_ROLES`) failed or was refused, 0
otherwise. A failure in FI's own lines is FI's, never the model's: the run says so and stops.

``experiment.py`` runs in the quest's own Python, where FI's ``core`` package is not importable; it reads FI's record through the
carried ``fi_table.py`` and the runtime ``fi_frame.py`` (:mod:`core.skeleton`). The formula calculator is FI's own text of
``core/oracle_forms.py`` and ``core/oracle_check.py``, carried unchanged into ``code/fi_core/`` (:func:`carried_files`; the tests
hold the copies equal to core's), so no folder of FI's has to be reachable (a cluster host, ``run.py``). The figures are drawn by
``code/fi_figures.py``, the text of ``core/analysis_figures.py``. A formula FI cannot compute, a missing record and a fault of FI's
own figures are FI's, not the study's: the run says so, exits 2 and nothing is sent to the model. Nothing here knows a field of
science."""

from __future__ import annotations

import ast
import itertools
import pprint
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import analysis_figures as _figures
from . import contract as _contract
from . import function_steps as _steps
from . import reference_values as _reference_values
from . import skeleton as _skeleton

VERSION = 1
DERIVE_FILE = "derive.py"
EXPERIMENT_FILE = "experiment.py"
#: Where FI keeps what it worked out (under the quest folder).
DERIVED_PATH = Path("data") / "results" / "derived.json"
PREVIOUS_PATH = Path("data") / "results" / "derived.previous.json"  # the earlier run's, while a run is under way
DERIVED_SCHEMA = "fi.derived/v1"
#: The module the figure step lives in (the text of ``core/analysis_figures.py``); ``experiment.py`` calls its ``run_figures`` (and
#: ``search_figures`` for a search) and takes ``said`` / ``exit_code`` / ``fi_faults`` from it.
FIGURES_MODULE = "fi_figures"
#: The folder of FI's own formula calculator, copied unchanged from ``core/`` beside ``experiment.py``.
CORE_DIR = "fi_core"
CORE_COPIES = ("oracle_forms.py", "oracle_check.py")

BEGIN = f"# >>> FI analysis frame, version {VERSION}: written by FI from the plan; a repair does not change it"
END = "# <<< FI analysis frame"
_BEGIN = re.compile(r"# >>> FI analysis frame, version (\d+): written by FI from the plan; a repair does not change it")
#: The names the frame binds once; a second binding outside the marked part changes what ``experiment.py`` reads.
FRAME_NAMES = frozenset({"ANALYSIS_VERSION", "RESULTS", "ROLES", "KINDS", "AFTERWARDS", "PAIRED", "DERIVED", "REFERENCES", "HELD", "FIXED",
                         "COMPARED", "PARTS", "FIGURES", "SEARCH", "Table", "Figure", "NotYet", "Missing", "UnknownResult", "fi_frame"})
#: A body's parameters, by the kind of body.
SIGNATURES = {"derive": ["table"], "reference": ["setting"], "figure": ["fig", "table"]}
_BODY_PREFIXES = tuple(f"{k}_" for k in SIGNATURES)


# ---- what the plan declares -------------------------------------------------------------------------------------------------
def applies(design: Any) -> str:
    """``""`` when FI writes the analysis for this plan, else why not: only a plan in the contract form has its results and its
    derived quantities declared apart (every other plan keeps the analysis the model writes)."""
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return "the plan has no protocol"
    if _contract.from_design(design).form != "contract":
        return "the plan is not in the contract form"
    if not _contract.from_design(design).results:
        return "the plan declares no result"
    return ""


def plan_of(design: dict[str, Any]) -> dict[str, Any]:
    """What the analysis needs from the plan, as plain values (they are written into the frame): the ids of the results (every
    role: a body may read a diagnostic) with their role and kind, the quantities derived from them and how each is computed, the
    references that carry a formula, the figures, what each run compares inside it, the sub-studies, and the numbers a formula may use."""
    protocol = design["protocol"]
    c = _contract.from_design(design)
    roles = {r.id: r.role for r in c.results}
    derived = []
    for d in c.derived:
        formula = (d.formula or "").strip()
        by = "formula" if formula else (f"reference_{d.id}" if d.role == "reference" else f"derive_{d.id}")
        derived.append({"id": d.id, "by": by, "formula": formula or None, "inputs": list(d.inputs), "role": d.role, "kind": d.kind,
                        "label": d.label})
    for r in c.results:
        if r.returned == "afterwards" and r.id not in {d["id"] for d in derived}:
            # a result the study works out from the trials (a fitted slope): no trial returns it, so it is a function of the study's
            derived.append({"id": r.id, "by": f"derive_{r.id}", "formula": None, "inputs": [], "role": r.role, "kind": r.kind,
                            "label": r.label})
    afterwards = {d["id"] for d in derived} & {r.id for r in c.results}
    references: dict[str, Any] = {}
    for r in c.results:
        ref = r.reference
        if isinstance(ref, dict) and _reference_values.shape_problem(ref) is None:
            references[r.id] = _reference_values.normalized(ref)
    return {
        "results": [r.id for r in c.results if r.id not in afterwards],
        "afterwards": sorted(afterwards),
        "roles": {**roles, **{d["id"]: d["role"] for d in derived}},
        "kinds": {**{r.id: r.kind for r in c.results}, **{d["id"]: d["kind"] for d in derived}},
        "paired": [r.id for r in c.results if r.paired and c.shared_draws],
        "derived": derived,
        "references": {k: {"formula": v["formula"], "when": v.get("when")} for k, v in references.items()},
        "held": _reference_values.held_values(protocol),
        "fixed": _reference_values.inputs(protocol),
        "compared": {str(k): list(v) for k, v in c.settings.in_each_run.items()},
        "parts": {str(s.get("name")): dict(s.get("hold") or {}) for s in c.settings.sub_studies if s.get("name")},
        "search": c.search is not None,
        "figures": {f.id: {"file": f.file, "shows": list(f.shows), "x": f.x, "series": f.series, "panels": f.panels}
                    for f in c.figures if f.declared},
    }


def body_names(plan: dict[str, Any]) -> list[str]:
    """The functions ``derive.py`` holds for this plan, in the order they are written: one per derived quantity with no formula,
    then one per declared figure."""
    return [d["by"] for d in plan["derived"] if d["by"] != "formula"] + [f"figure_{i}" for i in plan["figures"]]


# ---- derive.py -------------------------------------------------------------------------------------------------------------
def _literal(value: Any) -> str:
    return pprint.pformat(value, width=118, sort_dicts=False)


def _frame_text(plan: dict[str, Any]) -> str:
    body = (
        f"{BEGIN}\n"
        f"ANALYSIS_VERSION = {VERSION}  # the first statement: the study's imports go below the frame, never inside it\n"
        "import os as _os  # noqa: E402\nimport sys as _sys  # noqa: E402\n\n"
        "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        "from fi_frame import Figure, Missing, NotYet, Table, UnknownResult  # noqa: E402,F401\n\n"
        f"RESULTS = {_literal(tuple(plan['results']))}  # every result the plan declares: what a body reads from FI's record, by id\n"
        f"ROLES = {_literal(plan['roles'])}\n"
        f"KINDS = {_literal(plan['kinds'])}\n"
        f"AFTERWARDS = {_literal(tuple(plan['afterwards']))}  # results the study works out from the trials: readable from the record\n"
        f"PAIRED = {_literal(tuple(plan['paired']))}\n"
        f"DERIVED = {_literal({d['id']: {k: d[k] for k in ('by', 'formula', 'inputs', 'role')} for d in plan['derived']})}\n"
        f"REFERENCES = {_literal(plan['references'])}  # results whose reference FI works out from a formula\n"
        f"HELD = {_literal(plan['held'])}\n"
        f"FIXED = {_literal(plan['fixed'])}  # the numbers a formula may name besides a setting's own values\n"
        f"COMPARED = {_literal(plan['compared'])}  # what each run compares inside it\n"
        f"PARTS = {_literal(plan['parts'])}  # the sub-studies and the settings each holds\n"
        f"FIGURES = {_literal(plan['figures'])}\n"
        f"SEARCH = {bool(plan['search'])!r}  # a search for the best design: FI draws its own figures from its record\n"
        f"{END}\n")
    return body


def _stub(plan: dict[str, Any], name: str) -> str:
    kind, ident = name.split("_", 1)
    if kind == "figure":
        f = plan["figures"][ident]
        shows = ", ".join(f["shows"]) or "what the plan names"
        return (f"def {name}(fig, table):\n"
                f'    """Draw on `fig.axes` ({f["panels"]} panel(s)); FI creates, saves and closes the figure. It shows {shows}."""\n'
                f'    raise NotYet("{name} is not written yet")\n')
    d = next(d for d in plan["derived"] if d["id"] == ident)
    what = d["label"] or ident
    if kind == "reference":
        return (f"def {name}(setting):\n"
                f'    """The reference for {what} at one setting (`{{axis: value}}`): a number, or `None` where there is none."""\n'
                f'    raise NotYet("{name} is not written yet")\n')
    inputs = ", ".join(f"`{i}`" for i in d["inputs"]) or "the results the plan declares"
    return (f"def {name}(table):\n"
            f'    """{what}: one number for the study, or `{{setting key: number or None}}`; read {inputs} from `table`, and\n'
            f'    return `table.missing("why")` where it does not exist."""\n'
            f'    raise NotYet("{name} is not written yet")\n')


def derive_source(design: dict[str, Any], existing: str = "") -> str:
    """The text of ``derive.py``: FI's frame, then one stub per body the plan needs. With ``existing`` (the file as it is), the
    frame is written again from the plan and every body the study already wrote is kept, so a changed plan costs the study's
    code only the bodies it no longer asks for or has not got."""
    plan = plan_of(design)
    frame = _frame_text(plan)
    wanted = body_names(plan)
    old = (existing or "").lstrip("﻿")
    spans = _frame_span(old)
    if spans is None:
        head = ('"""The study\'s own part of the analysis. FI wrote the frame between the markers; the study writes the functions\n'
                'below it (`derive_<id>`, `reference_<id>`, `figure_<id>`). FI\'s `experiment.py` calls them."""\n\n')
        return head + frame + "\n\n" + "\n\n".join(_stub(plan, n) for n in wanted)
    rest = old[spans[1]:]
    have = _body_defs(rest)
    stubs = [_stub(plan, n) for n in wanted if n not in have]
    return old[:spans[0]] + frame + rest.rstrip("\n") + ("\n\n\n" + "\n\n".join(stubs) if stubs else "\n")


def _frame_span(source: str) -> tuple[int, int] | None:
    """``(start of the frame, end of its closing line)`` or ``None``."""
    m = _BEGIN.search(source)
    if m is None:
        return None
    j = source.find(END, m.start())
    if j < 0:
        return None
    k = source.find("\n", j)
    return m.start(), (k + 1 if k >= 0 else len(source))


def _body_defs(source: str) -> dict[str, ast.FunctionDef]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith(_BODY_PREFIXES)}


def stubs(source: str) -> list[str]:
    """The bodies of ``derive.py`` that are still the stub FI wrote (the study has not written them)."""
    return [n for n, node in _body_defs(source).items() if _steps.is_stub(source, node.name)]


def _pinned(tree: ast.Module) -> Counter:
    """What binds or changes the frame's names outside the marked part, read as syntax: ``{what: how many}``."""
    out: Counter = Counter()
    for n in _skeleton._module_level(tree.body):
        names: list[str] = []
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [x for t in n.targets for x in _skeleton._targets(t)]
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            names = _skeleton._targets(n.target)
        elif isinstance(n, ast.Delete):
            names = [x for t in n.targets for x in _skeleton._targets(t)]
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if (a.asname or a.name.split(".")[0]) in FRAME_NAMES:
                    out[f"a binding of `{a.asname or a.name}`"] += 1
        for name in names:
            if name in FRAME_NAMES:
                out[f"a binding of `{name}`"] += 1
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)) and _skeleton._base(n) in FRAME_NAMES:
            out[f"a change to `{_skeleton._base(n)}.{n.attr}`"] += 1
    return out


def _guard(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == "__name__" for x in ast.walk(n.test))
               for n in tree.body)


def derive_intact(old: str, new: str) -> str:
    """``""`` when ``new`` is ``old`` with only the study's own parts changed: the marked frame equal (read as syntax, so spacing
    does not count), every body ``old`` has still there with the same parameters, nothing outside the frame that binds a name of
    the frame, and no block that runs when the file is imported (``if __name__ == ...``). Else a plain sentence saying what
    changed."""
    old, new = (old or "").lstrip("﻿"), (new or "").lstrip("﻿")
    before = _frame_span(old)
    if before is None:
        return ""
    after = _frame_span(new)
    if after is None:
        return "the frame FI wrote was changed: the marked part of derive.py is missing"
    try:
        for x, y in itertools.zip_longest(ast.parse(old[before[0]:before[1]]).body, ast.parse(new[after[0]:after[1]]).body):
            if x is None or y is None or ast.dump(x) != ast.dump(y):
                return (f"the frame FI wrote was changed: in the marked part of derive.py, `{_skeleton._short(x, y)}` "
                        f"is now `{_skeleton._short(y, x)}`")
        was, now = ast.parse(old), ast.parse(new)
    except SyntaxError as e:
        return f"derive.py is not valid Python ({e.msg}, line {e.lineno})"
    mine, theirs = _body_defs(old), _body_defs(new)
    for name, node in mine.items():
        got = theirs.get(name)
        if got is None:
            return f"the function `{name}` is missing"
        if _skeleton._param_shape(node.args) != _skeleton._param_shape(got.args):
            return f"the parameters of `{name}` were changed; they must be ({', '.join(SIGNATURES[name.split('_', 1)[0]])})"
    if _pinned(was) != _pinned(now):
        more, less = sorted((_pinned(now) - _pinned(was)).keys()), sorted((_pinned(was) - _pinned(now)).keys())
        return ("the frame FI wrote was changed: derive.py adds " + ", ".join(more) if more
                else "the frame FI wrote was changed: derive.py removes " + ", ".join(less)) + " outside the marked part"
    if _guard(now) and not _guard(was):
        return "derive.py must run nothing when it is imported: remove the `if __name__ == ...` block (FI's experiment.py runs the analysis)"
    return ""


def analysis_problem(old: str, new: str, name: str) -> str:
    """Why ``new`` cannot replace the file ``name`` whose text is ``old``, as far as the analysis FI wrote is concerned, else
    ``""``. ``experiment.py`` with FI's head: any change is refused (it is all FI's); ``derive.py`` with FI's frame:
    :func:`derive_intact`. Any other file, and these two without FI's text, is not this check's (so nothing changes for a quest
    whose analysis the model writes)."""
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    old, new = (old or "").lstrip("﻿"), (new or "").lstrip("﻿")
    if name == EXPERIMENT_FILE:
        if not old.startswith(EXPERIMENT_HEAD):
            return ""
        return "" if new == old else "experiment.py is FI's own file and a repair does not edit it"
    if name == DERIVE_FILE:
        return derive_intact(old, new)
    return ""


def frame_problem(simulate_source: str) -> str:
    """``""`` when the analysis FI writes can sit beside this ``simulate.py``, else why not. A quest whose frame an older FI wrote
    (version 2) has a runtime without ``Table`` and no ``fi_table.py``, and its frame is not rewritten under a run: it keeps the
    analysis the model writes, and :func:`ensure_files` writes nothing for it."""
    framed = _skeleton.frame_version(simulate_source or "")
    if framed is not None and framed != _skeleton.VERSION:
        return (f"this quest's simulation frame is version {framed}, which has no table of the trials; "
                "its analysis stays the one the model writes")
    return ""


def carried_files() -> dict[str, str]:
    """The files FI copies into the code folder besides ``derive.py`` and ``experiment.py``, by their path there: the formula
    calculator (``core/oracle_forms.py`` and ``core/oracle_check.py`` as they are, in a package of their own: the standard library
    is all they need) and the figure step."""
    here = Path(__file__).parent
    out = {f"{CORE_DIR}/__init__.py": '"""FI\'s formula calculator: core/oracle_forms.py and core/oracle_check.py, copied as they are."""\n'}
    for name in CORE_COPIES:
        out[f"{CORE_DIR}/{name}"] = (here / name).read_text(encoding="utf-8")
    out[f"{FIGURES_MODULE}.py"] = Path(_figures.__file__).read_text(encoding="utf-8")
    return out


def ensure_files(code_dir: Path | str, design: dict[str, Any], simulate_source: str = "") -> list[str]:
    """Write ``derive.py`` and ``experiment.py`` into ``code_dir`` (with the runtime and the table they import, as
    :func:`core.skeleton.ensure_runtime` and ``ensure_table`` write them), keeping the bodies a ``derive.py`` already has.
    ``experiment.py`` is FI's whole and is written when it differs. Returns the names written. The caller decides whether to
    write at all (a resume reads these files and never writes them again)."""
    if frame_problem(simulate_source):
        return []
    folder = Path(code_dir)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    if _skeleton.ensure_runtime(folder, simulate_source):
        written.append(_skeleton.RUNTIME_FILE)
    if _skeleton.ensure_table(folder, simulate_source):
        written.append(_skeleton.TABLE_FILE)
    try:
        have = (folder / DERIVE_FILE).read_text(encoding="utf-8")
    except OSError:
        have = ""
    text = derive_source(design, have)
    for name, content in (*carried_files().items(), (DERIVE_FILE, text), (EXPERIMENT_FILE, EXPERIMENT)):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        try:
            if (folder / name).read_text(encoding="utf-8") == content:
                continue
        except OSError:
            pass
        (folder / name).write_text(content, encoding="utf-8")
        written.append(name)
    return written


# ---- experiment.py ---------------------------------------------------------------------------------------------------------
#: ``experiment.py`` is all FI's; this is the whole of it. It uses the standard library, the runtime (``fi_frame``) and the table
#: (``fi_table``) beside it. ``check_return`` and ``run`` can be called after importing it: it runs nothing on import.
EXPERIMENT = '''"""The analysis of this study. Written by FI and never edited: it reads FI's record of the trials, works out what the plan
declares (from the plan's formulas, or from the study's own functions in derive.py), keeps the numbers in
data/results/derived.json and prints RESULT_JSON last."""
# FI analysis, version @VERSION@
import json
import math
import numbers
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import derive  # noqa: E402
import fi_table  # noqa: E402
from fi_frame import Missing, NotYet, Table, UnknownResult  # noqa: E402

COUNTING = ("result", "reference", "physical_bound")


class FIFault(Exception):
    """Something wrong on FI's side (never the study's code): said plainly, never sent to the model as a fault of a body."""


RECORD = os.path.join("data", "results", "derived.json")
PREVIOUS = os.path.join("data", "results", "derived.previous.json")
_CALC = []
_ASIDE = {"files": []}


def _number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _finite(v):
    return _number(v) and math.isfinite(float(v))


def _calculator():
    """FI's formula calculator: FI's own text of core/oracle_forms.py, copied beside this file as fi_core/."""
    if not _CALC:
        try:
            from fi_core import oracle_forms
        except ImportError as e:
            raise FIFault("FI's formula calculator (fi_core/) is not beside experiment.py (%s)" % e)
        _CALC.append(oracle_forms)
    return _CALC[0]


def _why(ev):
    """Why FI's calculator gave no number, in words."""
    if ev.unreadable:
        return "its formula cannot be read (%s)" % ev.unreadable
    if ev.missing:
        return "it uses %s, which is not given here" % ", ".join("`%s`" % n for n in ev.missing)
    return ev.problem or "it gives no number"


def _forms_of(value):
    return {str(value), fi_table._fmt(value)}


def _where(e):
    """A plain sentence for an exception: its text and the line of derive.py it came from."""
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if os.path.basename(f.filename) == "derive.py"]
    text = "%s: %s" % (type(e).__name__, e)
    # FI's own lines raise on purpose when a body misuses the table (an id or a setting that is not there): that is the body's.
    # Anything else raised inside them, or an exception with no line of derive.py at all, is FI's.
    inner = os.path.basename(tb[-1].filename) if tb else ""
    mine = not frames or (inner in ("fi_frame.py", "fi_table.py") and not isinstance(e, (UnknownResult, KeyError, ValueError)))
    if mine:
        return "%s (in FI's own lines: %s line %d)" % (text, inner, tb[-1].lineno if tb else 0), True
    return "%s (derive.py line %d, in %s)" % (text, frames[-1].lineno, frames[-1].name), False


def _scalar(by, role, v, level=False):
    """One value of a body's return: ``("ok", number or None, why it is none, note)`` or ``("failed" | "refused", None, why, "")``."""
    if isinstance(v, Missing):
        if not v.why.strip():
            return "refused", None, "`%s` returned table.missing() with no reason; say why it does not exist" % by, ""
        return "ok", None, v.why, ""
    if v is None:
        return "ok", None, "", ""
    if not _number(v):
        return "refused", None, ("`%s` must return a number, None with a reason, or a mapping of setting keys; it returned %s"
                                  % (by, type(v).__name__)), ""
    v = float(v)
    if math.isfinite(v):
        return "ok", v, "", ""
    if level:
        return "ok", None, "it is %s where the result does not exist" % v, ""
    if role in COUNTING:
        return "failed", None, "`%s` returned %s, which is not a number" % (by, v), ""
    return "ok", None, "", "`%s` is %s (a %s, said and not reported)" % (by, v, role)


def check_return(by, role, value, keys, compared):
    """What FI makes of what the body ``by`` (of a quantity with this ``role``) returned. ``keys``: the study's settings.
    ``compared``: ``{input compared inside each run: its values}``. Returns a dict with ``status`` (ok | failed | refused),
    ``why``, and the value(s) kept: ``value``, ``by_setting`` or ``by_level`` (``{"input": name, "values": {value: number}}``),
    ``missing`` ``{where: why}`` and ``disclosed``."""
    if isinstance(value, Missing) and not value.why.strip():
        return {"status": "refused", "why": "`%s` returned table.missing() with no reason; say why it does not exist" % by}
    if value is None:
        return {"status": "refused", "why": ("`%s` returned None without saying why it does not exist; "
                                              "return table.missing(\\"the reason\\")" % by)}
    if not isinstance(value, dict):
        status, number, why, note = _scalar(by, role, value)
        out = {"status": status, "why": why if status != "ok" else ""}
        if status == "ok":
            out["value"] = number
            if number is None and why:
                out["missing"] = {"": why}
            if note:
                out["disclosed"] = [note]
        return out
    if not value:
        return {"status": "refused", "why": "`%s` returned an empty mapping; it must give a value for each setting" % by}
    known = set(keys)
    unknown = [k for k in value if str(k) not in known]
    factor = None
    if unknown:
        for name, values in compared.items():
            forms = set()
            for v in values:
                forms |= _forms_of(v)
            if all(str(k) in forms for k in value):
                factor = name
                break
        if factor is None:
            shown = ", ".join(sorted(known)[:6]) + (", ..." if len(known) > 6 else "")
            return {"status": "refused", "why": ("`%s` returned a value for `%s`, which is not a setting of the study; the settings are: %s"
                                                  % (by, unknown[0], shown or "(none)"))}
        left = [v for v in compared[factor] if not any(str(k) in _forms_of(v) for k in value)]
        if left:
            return {"status": "refused", "why": ("`%s` leaves out %s of `%s`; a value for every compared value is needed "
                                                  "(None, or NaN, where the result does not exist)"
                                                  % (by, ", ".join(str(v) for v in left), factor))}
    kept, missing, disclosed = {}, {}, []
    for k, v in value.items():
        status, number, why, note = _scalar(by, role, v, level=factor is not None)
        if status != "ok":
            return {"status": status, "why": "%s (under `%s`)" % (why, k)}
        kept[str(k)] = number
        if number is None:
            missing[str(k)] = why or "`%s` gave None here and no reason" % by
        if note:
            disclosed.append(note + " under `%s`" % k)
    out = {"status": "ok", "why": "", "missing": missing}
    if disclosed:
        out["disclosed"] = disclosed
    if factor is None:
        for k in keys:
            if k not in kept:
                missing[k] = "`%s` gave no value for this setting" % by
        out["by_setting"] = kept
    else:
        canon = {}
        for v in compared[factor]:
            for f in _forms_of(v):
                canon[f] = str(v)
        out["by_level"] = {"input": factor, "values": {canon[str(k)]: x for k, x in kept.items()}}
    return out


def _call(by, fn, *args):
    """``(value, None)`` or ``(None, (status, why, FI's own fault))`` for one call of a body."""
    try:
        return fn(*args), None
    except NotYet as e:
        return None, ("not_written", str(e) or "%s is not written yet" % by, False)
    except SystemExit:
        return None, ("refused", "`%s` ended the run itself (sys.exit); a function of the study returns its value" % by, False)
    except Exception as e:  # noqa: BLE001 -- one body's failure must not stop the others
        why, mine = _where(e)
        return None, ("failed", why, mine)


def _thresholds(path):
    if os.path.isdir(path):
        path = os.path.join(path, "raw", "trials.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    got = data.get("thresholds") if isinstance(data, dict) else None
    return {str(k): float(v) for k, v in (got or {}).items() if _finite(v)}


def _setting_values(table, key, base):
    values = dict(base)
    for axis, v in table.setting(key).items():
        if _number(v):
            values[axis] = float(v)
    return values


def _applies(when, table, key):
    cell = table.setting(key)
    if not when:
        return True

    def same(a, b):
        if _number(a) and _number(b):
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        return str(a).strip().lower() == str(b).strip().lower()

    def has(axis, v):
        return same(cell[axis], v) if axis in cell else axis in derive.HELD and same(derive.HELD[axis], v)

    return any(all(has(a, v) for a, v in f.items()) for f in when)


def _references(table, base, notes):
    """FI's reference of each result whose plan gives a formula: ``{id: {setting key: number}}``. A setting where the formula gives
    no number has no reference, and ``notes`` says so."""
    out = {}
    for rid, ref in derive.REFERENCES.items():
        forms = _calculator()
        got, dropped = {}, {}
        for key in table.keys():
            if not _applies(ref.get("when"), table, key):
                continue
            ev = forms.evaluate(ref["formula"], _setting_values(table, key, base), special=True)
            if ev.value is not None:
                got[key] = ev.value
            else:
                dropped[key] = _why(ev)
        if dropped:
            notes.append("the reference of `%s` has no value at %d setting(s) (%s: %s), so none is offered there"
                         % (rid, len(dropped), next(iter(dropped)), next(iter(dropped.values()))))
        out[rid] = got
    return out


def _formula(item, table, base, done):
    """A quantity with a formula, worked out by FI's calculator: one number, or one per setting when it names a setting's value or
    an input of the plan that differs between settings."""
    forms = _calculator()
    names = forms.names(item["formula"], {**forms.FUNCTIONS, **forms.SPECIAL_FUNCTIONS})
    axes = {a for k in table.keys() for a in table.setting(k)}
    per_setting = any(n in axes or n in derive.RESULTS or n in done for n in names)
    if not per_setting:
        ev = forms.evaluate(item["formula"], base, special=True)
        if ev.value is None:
            return {"status": "failed", "fi_fault": True, "why": "the plan's formula for `%s` cannot be computed: %s" % (item["id"], _why(ev))}
        return {"status": "ok", "why": "", "value": ev.value, "inputs": []}
    kept, missing, used = {}, {}, set()
    for key in table.keys():
        values = _setting_values(table, key, base)
        why = ""
        for n in names:
            if n in values:
                continue
            if n in derive.RESULTS:
                mean = table.mean(n, key)
                used.add(n)
                if mean is None:
                    why = "no trial of this setting gave `%s`" % n
                    break
                values[n] = mean
            elif n in done:
                got = done[n] if done[n].get("status") == "ok" else {}
                v = got.get("value") if "value" in got else (got.get("by_setting") or {}).get(key)
                used.add(n)
                if v is None:
                    why = "`%s` has no value here" % n
                    break
                values[n] = v
        if not why:
            ev = forms.evaluate(item["formula"], values, special=True)
            if ev.value is not None:
                kept[key] = ev.value
                continue
            why = _why(ev)
        missing[key] = why
    if not kept:
        return {"status": "failed", "fi_fault": True, "inputs": sorted(used),
                "why": "the plan's formula for `%s` gives no number at any setting (%s)" % (
                    item["id"], next(iter(missing.values()), "the study has no settings"))}
    return {"status": "ok", "why": "", "by_setting": kept, "missing": missing, "inputs": sorted(used)}


def _reference_body(by, role, fn, table):
    """A ``reference_<id>`` body: called once for each setting; the numbers are checked one at a time."""
    kept, missing, disclosed = {}, {}, []
    for key in table.keys():
        value, bad = _call(by, fn, table.setting(key))
        if bad is not None:
            return {"status": bad[0], "why": bad[1], "fi_fault": bad[2]}
        status, number, why, note = _scalar(by, role, value)
        if status != "ok":
            return {"status": status, "why": "%s (at %s)" % (why, key)}
        kept[key] = number
        if number is None and why:
            missing[key] = why
        if note:
            disclosed.append(note + " at %s" % key)
    out = {"status": "ok", "why": "", "by_setting": kept, "missing": missing}
    if disclosed:
        out["disclosed"] = disclosed
    return out


def _derived(table, base, compared):
    records = {}
    for ident, item in derive.DERIVED.items():
        by, role = item["by"], item["role"]
        before = list(table.served)
        if by == "formula":
            rec = _formula({**item, "id": ident}, table, base, records)
        elif getattr(derive, by, None) is None:
            rec = {"status": "not_written", "why": "%s is not in derive.py" % by}
        elif by.startswith("reference_"):
            rec = _reference_body(by, role, getattr(derive, by), table)
        else:
            value, bad = _call(by, getattr(derive, by), table)
            if bad is not None:
                rec = {"status": bad[0], "why": bad[1], "fi_fault": bad[2]}
            else:
                rec = check_return(by, role, value, table.keys(), compared)
        rec.update({"role": role, "by": by})
        rec.setdefault("inputs", [i for i in table.served if i not in before])
        records[ident] = rec
    return records


def _clean(o):
    """``o`` as JSON allows: a number that is not finite is null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, numbers.Real) and not isinstance(o, bool):
        return o if isinstance(o, int) else (float(o) if math.isfinite(float(o)) else None)
    return o


def result_json(table, records, compared):
    """RESULT_JSON in the shape the analysis has always been asked for: one mapping per setting, keyed like ``a=1,b=2``, with each
    result under its own name beside ``<name>_values`` (and ``_count`` / ``_total`` for a proportion, ``_pair_id`` for a paired
    result); the quantities derived for the study stand at the top level."""
    out, why = {}, {}
    seen = set()
    for key in table.keys():
        row = {}
        for rid in derive.RESULTS:
            try:
                vals = table.values(rid, key)
                if vals:
                    seen.add(rid)
                    row[rid] = table.mean(rid, key)
                    row[rid + "_values"] = vals
                    if derive.KINDS.get(rid) == "proportion":
                        row[rid + "_count"] = int(round(sum(vals)))
                        row[rid + "_total"] = len(vals)
                    if rid in derive.PAIRED and table.trials(rid, key):
                        row[rid + "_pair_id"] = table.trials(rid, key)
                else:
                    got = table.level_means(rid, key)
                    if got:
                        seen.add(rid)
                        row[rid] = got
                undefined = table.undefined(rid, key)
                if undefined:
                    seen.add(rid)
                    row[rid + "_undefined"] = undefined
            except Exception as e:  # noqa: BLE001 -- one result's shape never stops the others
                why[rid] = "FI could not print it at %s: %s: %s" % (key, type(e).__name__, e)
        failed = table.failed(key)
        if failed:
            row["n_failed"] = failed
        if row:
            out[key] = row
    for ident, rec in records.items():
        if rec.get("status") != "ok":
            out[ident] = None
            why[ident] = rec.get("why") or rec.get("status")
            continue
        if "value" in rec:
            out[ident] = rec["value"]
            if rec["value"] is None and rec.get("missing"):
                why[ident] = next(iter(rec["missing"].values()))
        elif "by_setting" in rec:
            for key, v in rec["by_setting"].items():
                if v is not None:
                    out.setdefault(key, {})[ident] = v
            for key, reason in (rec.get("missing") or {}).items():
                if reason and (rec["by_setting"].get(key) is None):
                    out.setdefault(key, {}).setdefault("why_missing", {})[ident] = reason
        elif "by_level" in rec:
            out[ident] = rec["by_level"]["values"]
    for rid in derive.RESULTS:
        if rid not in seen:
            out[rid] = None
            why.setdefault(rid, "no trial gave a value for it")
    if why:
        out["why_missing"] = why
    return _clean(out)


def _set_aside(quest_root):
    """Move what the last run left (its derived.json and the figures it drew) out of the places this run writes to, so nothing of it
    can be read as this run's. :func:`_put_back` returns what this run did not replace, marked as the earlier run's."""
    path = os.path.join(quest_root, RECORD)
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return
    figures = os.path.join(quest_root, "figures")
    for outcome in (old.get("figures") if isinstance(old, dict) else None) or []:
        name = os.path.basename(str(outcome.get("file") or "")) if isinstance(outcome, dict) else ""
        if name and os.path.exists(os.path.join(figures, name)):
            os.makedirs(os.path.join(figures, "previous"), exist_ok=True)
            os.replace(os.path.join(figures, name), os.path.join(figures, "previous", name))
            _ASIDE["files"].append(name)
    os.replace(path, os.path.join(quest_root, PREVIOUS))


def _put_back(quest_root, document, why=""):
    """Finish with the earlier run's files. A figure this run drew (or tried to draw) replaces the old one; one it did not touch
    comes back and is listed as the earlier run's. Without a ``document`` (this run stopped before it wrote one) the earlier
    derived.json comes back too, saying that it is the earlier run's and why this run stopped."""
    figures = os.path.join(quest_root, "figures")
    tried = {os.path.basename(str(o.get("file"))) for o in (document or {}).get("figures") or [] if isinstance(o, dict) and o.get("file")}
    kept = []
    for name in _ASIDE["files"]:
        src = os.path.join(figures, "previous", name)
        if not os.path.exists(src):
            continue
        if name in tried:
            os.remove(src)
        else:
            os.replace(src, os.path.join(figures, name))
            kept.append(name)
    try:
        os.rmdir(os.path.join(figures, "previous"))
    except OSError:
        pass
    _ASIDE["files"] = []
    prev = os.path.join(quest_root, PREVIOUS)
    if document is not None:
        if kept:
            document["figures_from_earlier_run"] = kept
        return kept
    try:
        with open(prev, encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return kept
    old["from_earlier_run"] = True
    old["this_run"] = {"status": "stopped", "why": why}
    old["figures_from_earlier_run"] = kept
    with open(os.path.join(quest_root, RECORD), "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1)
    os.remove(prev)
    return kept


def _figures(table, quest_root):
    """``(outcomes, said, exit code of the bodies, FI's own faults)`` of the figure step."""
    if not os.path.exists(os.path.join(HERE, "fi_figures.py")):
        return [], "", 0, []
    try:
        import fi_figures
    except ImportError as e:
        raise FIFault("FI's figure step (fi_figures.py) cannot be loaded (%s)" % e)
    out_dir = os.path.join(quest_root, "figures")
    os.makedirs(out_dir, exist_ok=True)
    specs = [dict(v, id=i) for i, v in derive.FIGURES.items()]
    try:
        outcomes = fi_figures.run_figures(specs, derive, table, out_dir)
        if derive.SEARCH:
            outcomes += fi_figures.search_figures(quest_root, out_dir)
    except Exception as e:  # noqa: BLE001 -- the figure step is FI's own code
        return [], "", 0, ["FI's figure step stopped with %s: %s" % (type(e).__name__, e)]
    return outcomes, fi_figures.said(outcomes), fi_figures.exit_code(outcomes), fi_figures.fi_faults(outcomes)


def run(record_path, quest_root):
    """The whole analysis: ``(RESULT_JSON, records, exit code, derived.json as written)``. Writes data/results/derived.json. The
    exit code is 2 for a fault of FI's own (nothing to send the model), else 1 when a body of a counting role, or a figure body,
    failed or was refused, else 0 (a body not written yet is listed in ``not_computed`` and does not stop the run)."""
    _set_aside(quest_root)
    cells = fi_table.load(record_path)
    if not cells:
        raise FIFault("FI's record of the trials (%s) is missing or empty, so there is nothing to analyse" % record_path)
    notes = []
    base = {**derive.FIXED, **_thresholds(record_path)}
    readable = list(derive.RESULTS) + list(derive.AFTERWARDS)
    first = Table(cells, readable)
    table = Table(cells, readable, _references(first, base, notes),
                  {name: first.at(**hold) for name, hold in derive.PARTS.items()})
    compared = {k: list(v) for k, v in derive.COMPARED.items()}
    records = _derived(table, base, compared)
    out = result_json(table, records, compared)
    outcomes, said, figure_code, figure_faults = _figures(table, quest_root)
    not_computed = [{"id": i, "status": r["status"], "why": r.get("why") or ""} for i, r in records.items() if r["status"] != "ok"]
    not_computed += [{"id": rid, "status": "no_data", "why": "no trial gave a value for it"}
                     for rid in derive.RESULTS if rid in out and out[rid] is None]
    faults = [r["why"] for r in records.values() if r.get("fi_fault")] + list(figure_faults)
    kept_figures = _put_back(quest_root, {"figures": outcomes})
    document = {"schema": "fi.derived/v1", "derived": records, "not_computed": not_computed, "complete": not not_computed,
                "figures": outcomes, "figures_said": said, "fi_faults": faults, "notes": notes}
    if kept_figures:
        document["figures_from_earlier_run"] = kept_figures
        document["notes"] = notes + ["%s: kept from the earlier run, this run did not draw %s" % (
            ", ".join(kept_figures), "it" if len(kept_figures) == 1 else "them")]
    path = os.path.join(quest_root, RECORD)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(document), f, indent=1)
    try:
        os.remove(os.path.join(quest_root, PREVIOUS))
    except OSError:
        pass
    bad = [i for i, r in records.items() if r["status"] in ("failed", "refused") and r["role"] in COUNTING and not r.get("fi_fault")]
    return out, records, (2 if faults else 1 if bad or figure_code else 0), document


def main():
    quest = os.path.dirname(HERE)
    record = fi_table.record_path(quest)
    out, records, code, document = run(record, quest)
    for ident, r in records.items():
        if r["status"] in ("failed", "refused"):
            sys.stderr.write("%s was not worked out%s: %s\\n" % (ident, " (FI's fault, not the study's)" if r.get("fi_fault") else "", r.get("why")))
        elif r["status"] == "not_written":
            sys.stderr.write("%s is computed by no trial and no function of the study yet\\n" % ident)
        for note in r.get("disclosed") or []:
            sys.stderr.write("%s: %s\\n" % (ident, note))
    for item in document["not_computed"]:
        if item["status"] == "no_data":
            sys.stderr.write("%s: %s\\n" % (item["id"], item["why"]))
    for text in [document["figures_said"]] + document["notes"] + document["fi_faults"]:
        if text:
            sys.stderr.write(text + "\\n")
    sys.stdout.write("RESULT_JSON: " + json.dumps(out) + "\\n")
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except FIFault as e:
        _put_back(os.path.dirname(HERE), None, str(e))
        sys.stderr.write("FI's analysis cannot start (this is FI's, not the study's code): %s\\n" % e)
        sys.exit(2)
    except BaseException as e:  # noqa: BLE001
        _put_back(os.path.dirname(HERE), None, "%s: %s" % (type(e).__name__, e))
        sys.stderr.write("FI's analysis stopped inside its own lines, not in the study's code (this is FI's fault): %s: %s\\n"
                         % (type(e).__name__, e))
        traceback.print_exc()
        sys.exit(2)
'''.replace("@VERSION@", str(VERSION))
#: The first lines of ``experiment.py``: a file that starts with them is FI's.
EXPERIMENT_HEAD = EXPERIMENT.split("\n", 1)[0]
