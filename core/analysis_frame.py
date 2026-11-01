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
from . import metric_spec as _metric_spec
from . import reference_values as _reference_values
from . import skeleton as _skeleton

VERSION = 1
DERIVE_FILE = "derive.py"
EXPERIMENT_FILE = "experiment.py"
#: Where FI keeps what it worked out (under the quest folder).
DERIVED_PATH = Path("data") / "results" / "derived.json"
PREVIOUS_PATH = Path("data") / "results" / "derived.previous.json"  # the earlier run's, while a run is under way
DERIVED_SCHEMA = "fi.derived/v1"
#: The roles whose failed function makes experiment.py exit 1 (the same tuple as ``COUNTING`` in the text of experiment.py).
_COUNTING_ROLES = ("result", "reference", "physical_bound")
#: The module the figure step lives in (the text of ``core/analysis_figures.py``); ``experiment.py`` calls its ``run_figures`` (and
#: ``search_figures`` for a search) and takes ``said`` / ``exit_code`` / ``fi_faults`` from it.
FIGURES_MODULE = "fi_figures"
#: The folder of FI's own formula calculator, copied unchanged from ``core/`` beside ``experiment.py``: FI's by its name, never a
#: package of the study's (core/code_layout.py ``FI_CORE_FOLDER``).
CORE_DIR = "fi_core"
CORE_COPIES = ("oracle_forms.py", "oracle_check.py")

BEGIN = f"# >>> FI analysis frame, version {VERSION}: written by FI from the plan; a repair does not change it"
END = "# <<< FI analysis frame"
_BEGIN = re.compile(r"# >>> FI analysis frame, version (\d+): written by FI from the plan; a repair does not change it")
#: The names the frame binds once; a second binding outside the marked part changes what ``experiment.py`` reads.
FRAME_NAMES = frozenset({"ANALYSIS_VERSION", "RESULTS", "ROLES", "KINDS", "AFTERWARDS", "PAIRED", "DERIVED", "REFERENCES", "HELD", "FIXED",
                         "COMPARED", "PARTS", "KNOWN", "FIGURES", "SEARCH", "Table", "Figure", "NotYet", "Missing", "UnknownResult", "fi_frame"})
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
    c = _contract.from_design(design, fi_analysis=True)
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
        "fixed": _reference_values.analysis_inputs(protocol),
        "compared": {str(k): list(v) for k, v in c.settings.in_each_run.items()},
        "parts": {str(s.get("name")): dict(s.get("hold") or {}) for s in c.settings.sub_studies if s.get("name")},
        "known": _known_values(protocol, c),
        "search": c.search is not None,
        "figures": {f.id: {"file": f.file, "shows": list(f.shows), "x": f.x, "series": f.series, "panels": f.panels,
                           **_result_labels(c, f.shows), **_result_info(c), **_panel_words(c, f.panels, (f.x, f.series))}
                    for f in c.figures if f.declared},
    }


def _known_values(protocol: dict[str, Any], c: Any) -> dict[str, dict[str, Any]]:
    """The results whose true value and interval the plan declares (``known_value``), for ``Table.coverage``: the two ends of the
    interval, and the true value as a number or as the setting's own axis. Nothing is guessed: a true value that is neither
    is left out, and so is the result."""
    fixed = _metric_spec.fixed_numbers(protocol)
    axes = {str(k) for part in ("grid", "in_each_run") if isinstance(protocol.get(part), dict) for k in protocol[part]}
    out = {}
    for r in c.results:
        kv = r.known_value if isinstance(r.known_value, dict) else {}
        ends, truth = kv.get("interval"), kv.get("true_value")
        number = _metric_spec.fixed_value(truth, fixed)
        if isinstance(ends, (list, tuple)) and len(ends) == 2 and (number is not None or truth in axes):
            out[r.id] = {"interval": [str(e) for e in ends], **({"truth": number} if number is not None else {"axis": str(truth)})}
    return out


def _result_labels(c: Any, shows: Any) -> dict[str, Any]:
    """``{"result_labels": {id: label}}`` for the declared results a figure shows that have a label in the plan (what the caption
    FI writes calls them); nothing when none has one."""
    names = {r.id: r.label for r in list(c.results) + list(c.derived) if getattr(r, "label", "")}
    got = {str(i): str(names[i]) for i in shows if i in names}
    return {"result_labels": got} if got else {}


def _result_info(c: Any) -> dict[str, Any]:
    """``{"result_info": {id: {"label", "unit", "description"}}}`` for every result and derived quantity of the plan that carries
    a label, a unit or an estimand: what FI writes an axis label from (``Figure.finish``) for any id a panel draws."""
    got = {}
    for r in list(c.results) + list(c.derived):
        info = {k: str(v) for k, v in (("label", getattr(r, "label", "")), ("unit", getattr(r, "unit", "")),
                                       ("description", getattr(r, "estimand", ""))) if v}
        if info:
            got[str(r.id)] = info
    return {"result_info": got} if got else {}


def _panel_words(c: Any, panels: int, axes: tuple[Any, ...] = ()) -> dict[str, Any]:
    """For a figure of several panels: the settings a panel may be declared for with their values, and the plan's label for each
    (``fig.panel``, ``core/analysis_figures.py``). A one-panel figure gets only the plan's labels of its ``axes`` (the setting along
    x and the one that makes its lines), which FI's default figure names its axes by; nothing when the plan labels neither."""
    if panels < 2:
        got = {str(a): c.labels[str(a)] for a in axes if a and c.labels.get(str(a))}
        return {"labels": got} if got else {}
    settings = {str(k): list(v) for k, v in {**c.settings.compared, **c.settings.in_each_run}.items()}
    named = {**settings, **{str(a): [] for a in axes if a}}
    return {"settings": settings, "labels": {k: c.labels[k] for k in named if c.labels.get(k)}}


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
        f"KNOWN = {_literal(plan['known'])}  # results with a true value and an interval: `table.coverage` counts how often it holds the truth\n"
        f"FIGURES = {_literal(plan['figures'])}\n"
        f"SEARCH = {bool(plan['search'])!r}  # a search for the best design: FI draws its own figures from its record\n"
        f"{END}\n")
    return body


def _stub(plan: dict[str, Any], name: str) -> str:
    kind, ident = name.split("_", 1)
    if kind == "figure":
        f = plan["figures"][ident]
        shows = ", ".join(f["shows"]) or "what the plan names"
        panel_note = ("\n    Panels that show different settings: `fig.panel(0, setting=value)` on each (FI writes the title)."
                      if f["panels"] > 1 else "")
        return (f"def {name}(fig, table):\n"
                f'    """Draw on `fig.axes` ({f["panels"]} panel(s)); FI creates, saves and closes the figure. It shows {shows}.{panel_note}"""\n'
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
    if _skeleton.ensure_figure_calls(folder, simulate_source):
        written.append(_skeleton.FIGURE_CALLS_FILE)
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


# ---- what the engine does with the analysis FI wrote -------------------------------------------------------------------------
#: The exit code of FI's experiment.py for a fault of FI's own (a formula FI cannot compute, a record that is missing or empty, a
#: fault of FI's own figures): nothing is sent to the model.
FI_FAULT_EXIT = 2
#: The files FI carries beside experiment.py and derive.py (:func:`carried_files`), never the study's: a repair or an edit does
#: not change them, and they are not part of the simulation.
_CARRIED_TOP = (f"{FIGURES_MODULE}.py",)


def is_fis(text: str) -> bool:
    """Whether ``text`` (an experiment.py) is the analysis FI writes."""
    return (text or "").lstrip("﻿").startswith(EXPERIMENT_HEAD)


def is_fis_file(path: Path | str) -> bool:
    try:
        return is_fis(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return False


def study_code(code_dir: Path | str) -> str | None:
    """The code the model wrote for the analysis, when FI writes the rest: ``derive.py`` without FI's frame. ``None`` when the
    analysis is not FI's (``experiment.py`` is then the model's own). FI's files (``experiment.py``, ``fi_table.py``,
    ``fi_figures.py``, the frame) are never the study's, so a check that reads the code to learn what the study did (a cap on a
    value, say) reads this and not them."""
    folder = Path(code_dir)
    if not is_fis_file(folder / EXPERIMENT_FILE):
        return None
    try:
        source = (folder / DERIVE_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    span = _frame_span(source)
    return source if span is None else source[:span[0]] + source[span[1]:]


def carried_file(code_dir: Path | str, rel: str) -> bool:
    """Whether ``rel`` (a path in the code folder ``code_dir``) is a file FI carries there for its analysis and holds FI's text:
    the figure step and the formula calculator. A module of the study's own with such a name is the study's."""
    rel = str(rel).replace("\\", "/")
    if rel not in _CARRIED_TOP and not rel.startswith(f"{CORE_DIR}/"):
        return False
    want = carried_files().get(rel)
    if want is None:
        return False
    try:
        return (Path(code_dir) / rel).read_text(encoding="utf-8").lstrip("﻿") == want
    except (OSError, UnicodeDecodeError):
        return False


def analysis_file(code_dir: Path | str, rel: str) -> bool:
    """Whether ``rel`` is a file of the analysis FI writes (``experiment.py`` and ``derive.py`` with FI's text, or a file FI
    carries beside them): never a file the simulation depends on."""
    rel = str(rel).replace("\\", "/")
    folder = Path(code_dir)
    if rel == EXPERIMENT_FILE:
        return is_fis_file(folder / rel)
    if rel == DERIVE_FILE:
        try:
            return _BEGIN.search((folder / rel).read_text(encoding="utf-8")) is not None
        except (OSError, UnicodeDecodeError):
            return False
    return carried_file(folder, rel)


def compose(design: dict[str, Any], reply: str, existing: str = "", keep: Any = ()) -> tuple[str, str]:
    """``(derive.py, note)``: FI's frame of ``derive.py`` for this plan with the study's functions from ``reply`` (the reply's
    ``# file: derive.py`` block) put into it, each through :func:`core.function_steps.replace_function_ex` (the path that puts
    the model's ``trial`` into the frame of simulate.py), with the helpers and imports they use. ``existing``: the file the quest
    has now; the bodies it holds are kept unless the reply gives them again. A reply that cannot be used leaves FI's file with the
    stubs (``note`` says why), so the run shows the functions as not written and the repair asks for them. ``keep``: the
    functions FI keeps as they ran (core/kept_bodies.py); a reply that gives one of them again does not replace it."""
    frame = derive_source(design, existing)
    wanted = body_names(plan_of(design))
    if not (reply or "").strip():
        return frame, ("the reply held no functions of derive.py" if [n for n in wanted if n not in keep] else "")
    try:
        tree = ast.parse(reply)
    except SyntaxError as e:
        return frame, f"the derive.py block is not valid Python ({e.msg}, line {e.lineno}); FI's stubs stay"
    given = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted and n.name not in keep]
    unknown = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith(_BODY_PREFIXES)
               and n.name not in wanted]
    text, notes = frame, []
    for name in given:
        made, why, _left = _steps.replace_function_ex(text, name, reply, signature=_skeleton.body_signature(name),
                                                      equation=name, wide=True)
        if made is None:
            notes.append(f"`{name}`: {why}")
            continue
        text = made
    problem = derive_intact(frame, text)
    if problem:
        return frame, problem + "; FI's stubs stay"
    if unknown:
        notes.append("the plan asks for no " + ", ".join(f"`{n}`" for n in unknown) + "; kept as a helper only")
    missing = [n for n in wanted if n not in given and n in stubs(text)]
    if missing:
        notes.append("the reply did not give " + ", ".join(f"`{n}`" for n in missing))
    return text, "; ".join(notes)


def failed_script(analysis: Path | str, returncode: int) -> str:
    """The file a failed run of the analysis ``analysis`` (experiment.py) sends to be repaired: ``derive.py`` when the analysis
    is FI's and it did not stop for a fault of FI's own (exit code 2: then ``experiment.py``, which the repair step reports and
    never sends to the model); the analysis itself otherwise."""
    path = Path(analysis)
    if is_fis_file(path) and returncode != FI_FAULT_EXIT:
        return DERIVE_FILE
    return path.name


def read_record(quest_root: Path | str) -> dict[str, Any]:
    """FI's record of the last run of its analysis (``data/results/derived.json``), or ``{}``."""
    import json

    try:
        data = json.loads((Path(quest_root) / DERIVED_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


_DERIVE_FRAME = re.compile(r'File "[^"]*derive\.py", line (\d+)(?:, in (\S+))?')
#: How FI's experiment.py says on stderr what it does now (a function of derive.py by its name, or FI's own step ``fi: ...``).
STEP_MARK = "[FI analysis] now: "
_STEP = re.compile(r"^\[FI analysis\] now: (.+?)\s*$", re.M)
#: The line a run of FI's analysis starts with, before derive.py is imported: the run's id, which the file of what was running
#: (``RUNNING_PATH``) carries too.
RUN_MARK = "[FI analysis] run: "
_RUN = re.compile(r"^\[FI analysis\] run: (\S+)", re.M)
_MARK = re.compile(r"^\[FI analysis\] (?:now|run): ")
#: The file in which FI's experiment.py keeps what it does now, while it runs.
RUNNING_PATH = "data/results/derived.running.json"


def running_step(stderr: str, quest_root: Path | str | None = None) -> str:
    """What FI's analysis was doing when it last said so: the name of a function of derive.py, or FI's own step (``fi: ...``);
    ``""`` when it said nothing (it stopped before its first step, while derive.py was imported). Read from its stderr, else
    from the file it keeps while it runs (``quest_root``), when that file is this run's: a run killed before it could remove
    the file leaves it, and a later run that stops while derive.py is imported names its own id on stderr, not that one."""
    found = _STEP.findall(stderr or "")
    if found:
        return found[-1].strip()
    if quest_root is not None:
        import json

        try:
            got = json.loads((Path(quest_root) / RUNNING_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if not isinstance(got, dict):
            return ""
        runs = _RUN.findall(stderr or "")
        if runs and str(got.get("run") or "") != runs[-1]:
            return ""  # left by an earlier run
        return str(got.get("step") or "").strip()
    return ""


def _is_body(name: Any) -> bool:
    return str(name or "").startswith(_BODY_PREFIXES)


def made_by(record: dict[str, Any], *, quantity: str = "", figure: str = "") -> str:
    """The function of derive.py that FI's record of the run (``record``) names as having made the derived quantity ``quantity``
    or the figure ``figure`` (its id or its file name). A quantity worked out by a formula of the plan is laid to the one
    function of the study's it is worked out from, when there is exactly one. ``""`` when the record names none (FI's own
    figure, a formula over the trials alone, or no record)."""
    record = record if isinstance(record, dict) else {}
    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    if quantity:
        seen: set[str] = set()
        bodies: list[str] = []

        def walk(ident: str) -> None:
            if ident in seen or not isinstance(derived.get(ident), dict):
                return
            seen.add(ident)
            rec = derived[ident]
            if _is_body(rec.get("by")):
                if rec["by"] not in bodies:
                    bodies.append(str(rec["by"]))
                return
            for n in rec.get("inputs") or []:
                walk(str(n))

        walk(str(quantity))
        return bodies[0] if len(bodies) == 1 else ""
    if figure:
        name = Path(str(figure).replace("\\", "/")).name
        stem = Path(name).stem
        made = record.get("bodies") if isinstance(record.get("bodies"), dict) else None
        for o in record.get("figures") or []:
            if isinstance(o, dict) and (str(o.get("id")) in (str(figure), stem) or Path(str(o.get("file") or "")).name == name):
                body = f"figure_{o.get('id')}"
                if made is None:  # a record of an older FI: a figure it tried to draw from a function of the study's
                    return body if o.get("status") in ("ok", "failed", "refused") else ""
                return body if body in made else ""
    return ""


def failure(record: dict[str, Any], stderr: str, returncode: int, printed: bool, *, step: str = "",
            timed_out: bool = False) -> dict[str, Any]:
    """What a failed run of FI's experiment.py was, from FI's own record of it (``record``: derived.json), what it was doing
    when it stopped (``step``, else read from ``stderr``: :func:`running_step`) and how it ended (``printed``: it printed its
    ``RESULT_JSON`` line, so ``record`` is this run's; ``timed_out``: FI stopped it for its time):
    ``{"kind": "body" | "module" | "fi", "body": the function of derive.py, "why": a plain sentence, "formula": bool}``.

    Who caused it is decided by what was running, never by the file a traceback's last line is in: whatever stopped the
    analysis while a function of the study's ran (FI's table refusing what it was given, a timeout, a crash) is that function's.

    * ``fi``: FI's own fault (exit code 2, or a stop in one of FI's own steps, outside every function of the study's): never
      sent to the model; ``formula`` says it is a formula of the plan FI could not compute.
    * ``module``: a statement of derive.py outside any function stopped its import (a syntax error, a name defined nowhere).
    * ``body``: the function that was running when the analysis stopped, or the one the record names as failed or refused (a
      quantity of a counting role first, also one worked out from the function's quantity, else a figure)."""
    record = record if isinstance(record, dict) else {}
    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    last = next((ln.strip() for ln in reversed((stderr or "").splitlines()) if ln.strip() and not _MARK.match(ln.strip())), "")
    if returncode == FI_FAULT_EXIT and not timed_out:
        fresh = printed and not record.get("from_earlier_run")
        formula = [r for r in derived.values() if isinstance(r, dict) and r.get("fi_fault") and r.get("by") == "formula"] if fresh else []
        said = [str(r.get("why") or "") for r in formula] or ([str(x) for x in record.get("fi_faults") or []] if fresh else [])
        if record.get("from_earlier_run"):
            said = [str((record.get("this_run") or {}).get("why") or "")]
        said = [s for s in said if s]
        return {"kind": "fi", "body": "", "formula": bool(formula), "why": "; ".join(said) or last or "FI's analysis stopped"}
    step = step or running_step(stderr)
    frames = _DERIVE_FRAME.findall(stderr or "")
    if not printed:
        if _is_body(step):
            how = "it ran past the time allowed" if timed_out else f"the analysis ended with exit code {returncode}"
            return {"kind": "body", "body": step, "formula": False,
                    "why": f"`{step}` was running when the analysis stopped ({how})" + (f": {last[:300]}" if last and not timed_out else "")}
        if frames and not step:  # the import of derive.py stopped: a statement outside the functions, or a function's own syntax
            return {"kind": "module", "body": "", "formula": False,
                    "why": f"derive.py stops when it is imported (line {frames[-1][0]}): {last[:300]}"}
        where = f" ({step[3:].strip()})" if step.startswith("fi:") else ""
        if timed_out:
            return {"kind": "fi", "body": "", "formula": False,
                    "why": f"FI's analysis ran past the time allowed in one of its own steps{where}, not in a function of the study's"}
        return {"kind": "fi", "body": "", "formula": False,
                "why": (f"FI's analysis stopped in one of its own steps{where}, not in a function of the study's" if step else
                        "FI's analysis stopped before it began, in its own lines, not in a function of the study's")
                       + (f" ({last[:300]})" if last else "")}
    bad = [(i, r) for i, r in derived.items() if isinstance(r, dict) and r.get("status") in ("failed", "refused")
           and not r.get("fi_fault") and _is_body(r.get("by"))]
    # A quantity worked out from one a function of the study's did not give: the repair is that function's.
    said_by = {str(r.get("by")): r for r in derived.values() if isinstance(r, dict) and _is_body(r.get("by"))}
    for i, r in derived.items():
        cause = said_by.get(str(r.get("cause") or "")) if isinstance(r, dict) and r.get("status") == "blocked" else None
        if cause is not None and cause.get("status") in ("failed", "refused"):
            bad.append((i, {"by": cause["by"], "role": r.get("role"), "why": f"{r.get('why')}: {cause.get('why') or 'failed'}"}))
    bad += [(str(o.get("id")), {"by": f"figure_{o.get('id')}", "why": o.get("why"), "role": "figure"})
            for o in record.get("figures") or [] if isinstance(o, dict) and o.get("status") in ("failed", "refused")]
    if not bad:
        return {"kind": "fi", "body": "", "formula": False,
                "why": "FI's analysis ended with a failure its record does not name a function of the study's for"
                       + (f" ({last[:300]})" if last else "")}
    counting = [b for b in bad if b[1].get("role") in ("result", "reference", "physical_bound")]
    ident, rec = (counting or bad)[0]
    return {"kind": "body", "body": str(rec["by"]), "formula": False, "why": f"`{ident}`: {rec.get('why') or 'failed'}"}


def only_figures_failed(record: dict[str, Any], returncode: int) -> bool:
    """Whether a run of FI's experiment.py that exited 1 produced all its results and only a figure of the study's failed or was
    refused. FI's record of the run (derived.json) says it: ``run()`` exits 1 for a quantity of a counting role it could not work out
    or for a figure, and 0 for neither, so a record of this run with no such quantity and a failed figure is the second. A figure is
    the analysis's, not the run's: the repair of it goes on (``failure`` names its body), but what the run produced stands."""
    if returncode != 1 or not isinstance(record, dict) or record.get("schema") != DERIVED_SCHEMA or record.get("from_earlier_run"):
        return False
    if record.get("fi_faults"):
        return False
    derived = record.get("derived") if isinstance(record.get("derived"), dict) else {}
    status_of = {r.get("by"): r.get("status") for r in derived.values() if isinstance(r, dict)}
    for r in derived.values():
        if not isinstance(r, dict) or r.get("role") not in _COUNTING_ROLES or r.get("fi_fault"):
            continue
        if r.get("status") in ("failed", "refused") or (r.get("status") == "blocked" and status_of.get(r.get("cause")) in ("failed", "refused")):
            return False
    return any(isinstance(o, dict) and o.get("status") in ("failed", "refused") for o in record.get("figures") or [])


def run_produced(returncode: int, printed_result: bool, record: dict[str, Any], analysis_on: bool) -> bool:
    """Whether a run of the study produced its results: it exited 0, or it exited 1 only because a figure of the study's failed
    (FI wrote the analysis, the run printed its RESULT line, and its record says only figures failed). The one rule behind the
    run's ``results_produced`` and the clean rerun of ``code/``, so the two cannot disagree about the same run."""
    if returncode == 0:
        return True
    return bool(returncode == 1 and analysis_on and printed_result and only_figures_failed(record, returncode))


def study_values(record: dict[str, Any]) -> dict[str, float]:
    """The study-wide numbers FI's record of the run holds (a quantity worked out once for the study, by its formula or its
    function): ``{id: value}``, finite values only."""
    import math

    derived = (record.get("derived") if isinstance(record, dict) and record.get("schema") == DERIVED_SCHEMA
               and isinstance(record.get("derived"), dict) else {})
    out: dict[str, float] = {}
    for ident, rec in derived.items():
        v = rec.get("value") if isinstance(rec, dict) and rec.get("status") == "ok" else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[str(ident)] = float(v)
    return out


def not_computed(record: dict[str, Any]) -> list[dict[str, str]]:
    """The declared quantities the last run did not work out, from its record: ``[{"id", "status", "why"}]`` (a function not
    written yet, one that failed or was refused, a result no trial gave)."""
    items = record.get("not_computed") if isinstance(record, dict) and record.get("schema") == DERIVED_SCHEMA else None
    return [{"id": str(i.get("id")), "status": str(i.get("status") or ""), "why": str(i.get("why") or "")}
            for i in items or [] if isinstance(i, dict) and i.get("id")]


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
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# This run's id, said before derive.py is imported and kept in RUNNING beside each step: a file of what was running that a
# killed run left is then never read as this run's.
RUN = "%d-%d" % (os.getpid(), time.time_ns())
if __name__ == "__main__":
    sys.stderr.write("[FI analysis] run: %s\\n" % RUN)
    sys.stderr.flush()

import derive  # noqa: E402
import fi_table  # noqa: E402
from fi_frame import Missing, NotYet, Table, UnknownResult  # noqa: E402

COUNTING = ("result", "reference", "physical_bound")


class FIFault(Exception):
    """Something wrong on FI's side (never the study's code): said plainly, never sent to the model as a fault of a body."""


RECORD = os.path.join("data", "results", "derived.json")
PREVIOUS = os.path.join("data", "results", "derived.previous.json")
RUNNING = os.path.join("data", "results", "derived.running.json")  # what was running: a function of the study's, or FI's own step
_CALC = []
_ASIDE = {"files": []}
_STATE = {"quest": None, "step": None}
_BODIES = {}  # each function of the study's that ran: its time and the warnings it raised


def _progress(step):
    """Say what the analysis does now (a function of the study's by its name, or FI's own step as ``fi: ...``), on stderr and in
    RUNNING, so a run that is stopped for its time, or killed, is laid to what was running."""
    if _STATE["step"] == step:
        return
    _STATE["step"] = step
    sys.stderr.write("[FI analysis] now: %s\\n" % step)
    sys.stderr.flush()
    if _STATE["quest"]:
        try:
            path = os.path.join(_STATE["quest"], RUNNING)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"step": step, "run": RUN}, f)
        except OSError:
            pass


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


def _where(e, by):
    """A plain sentence for what the function ``by`` raised while it ran: its text and the line of derive.py it came from. Whatever
    is raised while a function of the study's runs is its own, also when FI's table or runtime raised it over what the function
    gave them (they are called with the function's own arguments): FI's own faults are found outside the calls of the functions."""
    tb = traceback.extract_tb(e.__traceback__)
    frames = [f for f in tb if os.path.basename(f.filename) == "derive.py"]
    text = "%s: %s" % (type(e).__name__, e)
    at = "derive.py line %d, in %s" % (frames[-1].lineno, frames[-1].name) if frames else "in `%s`" % by
    inner = os.path.basename(tb[-1].filename) if tb else ""
    if frames and inner and inner != "derive.py":
        at += "; raised in %s line %d over what `%s` gave it" % (inner, tb[-1].lineno, by)
    return "%s (%s)" % (text, at)


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
    """``(value, None)`` or ``(None, (status, why))`` for one call of a body. What the body raises while it runs is its own; its time
    and the warnings it raised are kept for it (``_BODIES``), and each warning is written to stderr once, as Python writes it."""
    _progress(by)
    seen = _BODIES.setdefault(by, {"seconds": 0.0, "warnings": []})
    value, bad = None, None
    began = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = fn(*args)
        except NotYet as e:
            bad = ("not_written", str(e) or "%s is not written yet" % by)
        except SystemExit:
            bad = ("refused", "`%s` ended the run itself (sys.exit); a function of the study returns its value" % by)
        except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001 -- one body's failure must not stop the others
            bad = ("failed", _where(e, by))
    seen["seconds"] += time.monotonic() - began
    for w in caught:
        said = "%s: %s" % (w.category.__name__, w.message)
        if said not in seen["warnings"] and len(seen["warnings"]) < 20:
            seen["warnings"].append(said)
            sys.stderr.write(warnings.formatwarning(w.message, w.category, w.filename, w.lineno))
    return value, bad


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


def _plan_fault(item, why, used=()):
    """A formula of the plan FI cannot compute as written (it does not read, or it names what the plan does not declare): the plan's
    fault, which FI says and never sends to the model."""
    return {"status": "failed", "fi_fault": True, "plan": True, "inputs": sorted(used),
            "why": "the plan's formula for `%s` cannot be computed: %s" % (item["id"], why)}


_HOW = {"not_written": "is not written yet", "refused": "was refused", "failed": "failed", "blocked": "was not worked out"}


def _formula(item, table, base, done):
    """A quantity with a formula, worked out by FI's calculator: one number, or one per setting when it names a setting's value, a
    result, or a derived quantity that has one per setting. A quantity it names that was not worked out (its function failed, or is
    not written yet) leaves this one ``blocked`` with that function as its ``cause``; one that does not exist somewhere (a
    reason given) leaves this one missing there, with the reason."""
    forms = _calculator()
    unreadable = forms.evaluate(item["formula"], {}, special=forms.ANALYSIS_SPECIAL).unreadable  # read with FI's functions
    if unreadable:
        return _plan_fault(item, "its formula cannot be read (%s)" % unreadable)
    one = forms.one_number_summary(item["formula"])  # max(growth) of one mean per setting: that mean, never a spread
    if one:
        return _plan_fault(item, one)
    names = forms.names(item["formula"], {**forms.FUNCTIONS, **forms.ANALYSIS_SPECIAL})
    axes = {a for k in table.keys() for a in table.setting(k)}
    results = set(derive.RESULTS) | set(derive.AFTERWARDS)
    unknown = [n for n in names if n not in axes and n not in results and n not in derive.DERIVED and n not in base]
    if unknown:
        return _plan_fault(item, "it uses %s, which the plan does not declare" % ", ".join("`%s`" % n for n in unknown))
    for n in names:
        dep = done.get(n)
        if dep is not None and dep.get("status") != "ok":
            cause = dep.get("cause") or (dep.get("by") if dep.get("by") != "formula" else "")
            return {"status": "blocked", "inputs": [n], "after": n, "cause": cause,
                    "why": "it is worked out from `%s`, which %s" % (n, _HOW.get(dep.get("status"), "was not worked out"))}
    per_setting = any(n in axes or n in results or (n in done and "value" not in done[n]) for n in names)
    if not per_setting:
        values = dict(base)
        for n in names:
            if n in done:
                if done[n].get("value") is None:
                    why = next(iter((done[n].get("missing") or {}).values()), "") or "it has no value"
                    return {"status": "ok", "why": "", "value": None, "missing": {"": "`%s` does not exist: %s" % (n, why)},
                            "inputs": [n]}
                values[n] = done[n]["value"]
        ev = forms.evaluate(item["formula"], values, special=forms.ANALYSIS_SPECIAL)
        out = {"status": "ok", "why": "", "value": ev.value, "inputs": [n for n in names if n in done]}
        if ev.value is None:
            out["missing"] = {"": _why(ev)}
        return out
    kept, missing, used = {}, {}, set()
    for key in table.keys():
        values = _setting_values(table, key, base)
        why = ""
        for n in names:
            if n in values:
                continue
            if n in results:
                if table.per_level(n, key):
                    why = "`%s` is recorded for each value compared inside the runs, not once for this setting" % n
                    break
                mean = table.mean(n, key)
                used.add(n)
                if mean is None:
                    why = "no trial of this setting gave `%s`" % n
                    break
                values[n] = mean
            elif n in done:
                got = done[n]
                v = got.get("value") if "value" in got else (got.get("by_setting") or {}).get(key)
                used.add(n)
                if v is None:
                    said = (got.get("missing") or {}).get(key) or (got.get("missing") or {}).get("")
                    why = "`%s` has no value here%s" % (n, " (%s)" % said if said else "")
                    break
                values[n] = v
        if not why:
            ev = forms.evaluate(item["formula"], values, special=forms.ANALYSIS_SPECIAL)
            if ev.value is not None:
                kept[key] = ev.value
                continue
            why = _why(ev)
        missing[key] = why
    return {"status": "ok", "why": "", "by_setting": kept, "missing": missing, "inputs": sorted(used)}


def _order(items):
    """``(order, circle)``: the plan's derived quantities in an order they can be worked out in, whatever order the plan lists them
    in: every function of the study's first (they read only the table), then each formula after the quantities it names. Formulas
    that name each other round in a circle cannot be ordered and are returned apart."""
    order = [i for i, item in items.items() if item["by"] != "formula"]
    formulas = [i for i, item in items.items() if item["by"] == "formula"]
    if not formulas:
        return order, []
    forms = _calculator()
    waiting = {i: {n for n in forms.names(items[i]["formula"], {**forms.FUNCTIONS, **forms.SPECIAL_FUNCTIONS}) if n in items}
               for i in formulas}
    while waiting:
        ready = [i for i in formulas if i in waiting and waiting[i] <= set(order)]
        if not ready:
            break
        for i in ready:
            order.append(i)
            del waiting[i]
    return order, [i for i in formulas if i in waiting]


def _reference_body(by, role, fn, table):
    """A ``reference_<id>`` body: called once for each setting; the numbers are checked one at a time."""
    kept, missing, disclosed = {}, {}, []
    for key in table.keys():
        value, bad = _call(by, fn, table.setting(key))
        if bad is not None:
            return {"status": bad[0], "why": bad[1]}
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
    order, circle = _order(derive.DERIVED)
    for ident in circle:
        item = derive.DERIVED[ident]
        records[ident] = {**_plan_fault({**item, "id": ident}, "the formulas of %s name each other in a circle, so none of them "
                                        "can be worked out first" % ", ".join("`%s`" % i for i in circle)),
                          "role": item["role"], "by": "formula"}
    for ident in order:
        item = derive.DERIVED[ident]
        by, role = item["by"], item["role"]
        before = list(table.served)
        if by == "formula":
            _progress("fi: the plan's formula for %s" % ident)
            rec = _formula({**item, "id": ident}, table, base, records)
        elif getattr(derive, by, None) is None:
            rec = {"status": "not_written", "why": "%s is not in derive.py" % by}
        elif by.startswith("reference_"):
            rec = _reference_body(by, role, getattr(derive, by), table)
        else:
            value, bad = _call(by, getattr(derive, by), table)
            if bad is not None:
                rec = {"status": bad[0], "why": bad[1]}
            else:
                rec = check_return(by, role, value, table.keys(), compared)
        rec.update({"role": role, "by": by})
        rec.setdefault("inputs", [i for i in table.served if i not in before])
        records[ident] = rec
    return {i: records[i] for i in derive.DERIVED}  # in the plan's order


def _bodies(records, outcomes):
    """FI's record of each function of the study's: what came of it, its seconds, the warnings it raised and what it gives (the
    derived quantities or the figure), so a check that flags a number or a figure finds the function that made it."""
    out = {}
    for ident, rec in records.items():
        by = rec.get("by")
        if by and by != "formula":
            row = out.setdefault(by, {"status": rec.get("status"), "outputs": []})
            row["outputs"].append(ident)
    for o in outcomes:
        if isinstance(o, dict) and o.get("id") in derive.FIGURES and str(o.get("status")) != "skipped":
            out.setdefault("figure_%s" % o["id"], {"status": o.get("status"), "outputs": ["figure:%s" % o["id"]]})
    for by, row in out.items():
        seen = _BODIES.get(by) or {}
        row["seconds"] = round(float(seen.get("seconds") or 0.0), 3)
        if seen.get("warnings"):
            row["warnings"] = list(seen["warnings"])
    return out


def _clean(o):
    """``o`` as JSON allows: a number that is not finite is null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, numbers.Real) and not isinstance(o, bool):
        return o if isinstance(o, int) else (float(o) if math.isfinite(float(o)) else None)
    return o


def _coverage_of(table, rid, key, row, out_levels):
    """FI's own count of how often the interval the trials reported holds the true value the plan gives, printed as
    ``<id>_coverage`` beside the result (for each compared value under its own record key). It is FI's record, so it stands
    whether or not a function of the study's reads it, and whatever a figure of the study's does."""
    if table.per_level(rid, key):
        for _text, c in table._level_cells(rid, key):
            hits = table.coverage(rid, key, **c["level"])
            if hits:
                out_levels.setdefault(table._key_of(c), {}).setdefault(rid + "_coverage", sum(hits) / len(hits))
        return
    hits = table.coverage(rid, key)
    if hits:
        # a result of the study's own under this name keeps its value: FI's count is added beside it, never over it
        row.setdefault(rid + "_coverage", sum(hits) / len(hits))


def result_json(table, records, compared):
    """RESULT_JSON in the shape the analysis has always been asked for: one mapping per setting, keyed like ``a=1,b=2``, with each
    result under its own name beside ``<name>_values`` (and ``_count`` / ``_total`` for a proportion, ``_pair_id`` for a paired
    result); the quantities derived for the study stand at the top level."""
    out, why = {}, {}
    out_levels = {}  # the rows of the values compared inside each run, keyed as FI's record keys them
    seen = set()
    for key in table.keys():
        row = {}
        for rid in derive.RESULTS:
            try:
                vals = [] if table.per_level(rid, key) else table.values(rid, key)
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
                        for rkey, text, vals in table.level_rows(rid, key):  # the trials behind each compared value, under its record key
                            lrow = out_levels.setdefault(rkey, {})
                            lrow[rid] = got.get(text)
                            lrow[rid + "_values"] = vals
                undefined = 0 if table.per_level(rid, key) else table.undefined(rid, key)
                if undefined:
                    seen.add(rid)
                    row[rid + "_undefined"] = undefined
            except Exception as e:  # noqa: BLE001 -- one result's shape never stops the others
                why[rid] = "FI could not print it at %s: %s: %s" % (key, type(e).__name__, e)
        for rid in getattr(derive, "KNOWN", {}):
            try:
                _coverage_of(table, rid, key, row, out_levels)
            except Exception as e:  # noqa: BLE001 -- one coverage never stops the others
                why[rid + "_coverage"] = "FI could not count it at %s: %s: %s" % (key, type(e).__name__, e)
        failed = table.failed(key)
        if failed:
            row["n_failed"] = failed
        if row:
            out[key] = row
    for rkey, lrow in out_levels.items():
        out.setdefault(rkey, {}).update(lrow)
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
    """Finish with the earlier run's files (and with RUNNING, which says what this run did while it ran). A figure this run drew (or tried to draw) replaces the old one; one it did not touch
    comes back and is listed as the earlier run's. Without a ``document`` (this run stopped before it wrote one) the earlier
    derived.json comes back too, saying that it is the earlier run's and why this run stopped."""
    if document is None:
        try:
            os.remove(os.path.join(quest_root, RUNNING))
        except OSError:
            pass
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


def _kept_figure(spec_id, quest_root):
    """For the run that draws only what the earlier run failed to draw (``FI_FIGURE_FALLBACK=only``): the outcome of a figure the
    earlier run drew, with its file put back where it was, else None."""
    try:
        with open(os.path.join(quest_root, PREVIOUS), encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        return None
    for o in (old.get("figures") if isinstance(old, dict) else None) or []:
        if isinstance(o, dict) and o.get("id") == spec_id and o.get("status") == "ok":
            name = os.path.basename(str(o.get("file") or ""))
            src = os.path.join(quest_root, "figures", "previous", name)
            if name in _ASIDE["files"] and os.path.exists(src):
                os.replace(src, os.path.join(quest_root, "figures", name))
                _ASIDE["files"].remove(name)
                return o
    return None


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
        outcomes = []
        only = os.environ.get("FI_FIGURE_FALLBACK") == "only"
        for spec in specs:
            by = "figure_%s" % spec["id"]
            kept = _kept_figure(spec["id"], quest_root) if only else None
            if kept is not None:
                outcomes.append(kept)
                continue
            _progress(by)
            seen = _BODIES.setdefault(by, {"seconds": 0.0, "warnings": []})
            began = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                outcomes += fi_figures.run_figures([spec], derive, table, out_dir)
                if os.environ.get("FI_FIGURE_FALLBACK") in ("1", "only") and outcomes[-1].get("status") in ("failed", "refused", "not_written"):
                    # No repair is left for the body: the plan's own figure is drawn from the table in its place
                    plain = fi_figures.default_figure(spec, table, out_dir)
                    if plain.get("status") in ("ok", "dropped"):
                        outcomes[-1] = plain
                    else:
                        sys.stderr.write("%s\\n" % plain.get("said"))
            seen["seconds"] += time.monotonic() - began
            for w in caught:
                said = "%s: %s" % (w.category.__name__, w.message)
                if said not in seen["warnings"] and len(seen["warnings"]) < 20:
                    seen["warnings"].append(said)
                    sys.stderr.write(warnings.formatwarning(w.message, w.category, w.filename, w.lineno))
        if derive.SEARCH:
            _progress("fi: FI's own figures of the search")
            outcomes += fi_figures.search_figures(quest_root, out_dir)
    except Exception as e:  # noqa: BLE001 -- the figure step is FI's own code
        return [], "", 0, ["FI's figure step stopped with %s: %s" % (type(e).__name__, e)]
    return outcomes, fi_figures.said(outcomes), fi_figures.exit_code(outcomes), fi_figures.fi_faults(outcomes)


def _earlier_record(quest_root):
    """The earlier run's derived.json (set aside at the start of this run), for the run that only draws the figures that failed."""
    try:
        with open(os.path.join(quest_root, PREVIOUS), encoding="utf-8") as f:
            old = json.load(f)
    except (OSError, ValueError):
        old = None
    if not isinstance(old, dict) or old.get("schema") != "fi.derived/v1" or not isinstance(old.get("derived"), dict):
        raise FIFault("there is no record of the earlier run to draw the figures beside")
    return old


def run(record_path, quest_root):
    """The whole analysis: ``(RESULT_JSON, records, exit code, derived.json as written)``. Writes data/results/derived.json. The
    exit code is 2 for a fault of FI's own (nothing to send the model), else 1 when a body of a counting role, or a figure body,
    failed or was refused, else 0 (a body not written yet is listed in ``not_computed`` and does not stop the run)."""
    _STATE["quest"] = quest_root
    try:
        os.remove(os.path.join(quest_root, RUNNING))
    except OSError:
        pass
    _progress("fi: reading FI's record of the trials")
    _set_aside(quest_root)
    cells = fi_table.load(record_path)
    if not cells:
        raise FIFault("FI's record of the trials (%s) is missing or empty, so there is nothing to analyse" % record_path)
    notes = []
    base = {**derive.FIXED, **_thresholds(record_path)}
    readable = list(derive.RESULTS) + list(derive.AFTERWARDS)
    # ``FI_FIGURE_FALLBACK=only``: the figures that failed are drawn beside the earlier run's results, which stay as they were. No
    # function of the study's is run again (it may draw random numbers), so no derived value, reference or result can change.
    only = os.environ.get("FI_FIGURE_FALLBACK") == "only"
    earlier = _earlier_record(quest_root) if only else None
    first = Table(cells, readable)
    table = Table(cells, readable, {} if only else _references(first, base, notes),
                  {name: first.at(**hold) for name, hold in derive.PARTS.items()}, getattr(derive, "KNOWN", {}))
    compared = {k: list(v) for k, v in derive.COMPARED.items()}
    records = earlier["derived"] if only else _derived(table, base, compared)
    _progress("fi: printing the results")
    out = {} if only else result_json(table, records, compared)
    outcomes, said, figure_code, figure_faults = _figures(table, quest_root)
    not_computed = [{"id": i, "status": r["status"], "why": r.get("why") or ""} for i, r in records.items() if r["status"] != "ok"]
    not_computed += [{"id": rid, "status": "no_data", "why": "no trial gave a value for it"}
                     for rid in derive.RESULTS if rid in out and out[rid] is None]
    faults = [r["why"] for r in records.values() if r.get("fi_fault")] + list(figure_faults)
    _progress("fi: writing the results")
    kept_figures = _put_back(quest_root, {"figures": outcomes})
    document = {"schema": "fi.derived/v1", "derived": records, "not_computed": not_computed, "complete": not not_computed,
                "figures": outcomes, "figures_said": said, "fi_faults": faults, "notes": notes,
                "bodies": _bodies(records, outcomes)}
    if only:
        document = dict(earlier, figures=outcomes, figures_said=said, fi_faults=list(earlier.get("fi_faults") or []) + list(figure_faults))
        document.pop("figures_from_earlier_run", None)
        notes = list(earlier.get("notes") or [])
    if kept_figures:
        document["figures_from_earlier_run"] = kept_figures
        document["notes"] = notes + ["%s: kept from the earlier run, this run did not draw %s" % (
            ", ".join(kept_figures), "it" if len(kept_figures) == 1 else "them")]
    path = os.path.join(quest_root, RECORD)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(document), f, indent=1)
    for name in (PREVIOUS, RUNNING):
        try:
            os.remove(os.path.join(quest_root, name))
        except OSError:
            pass
    # A function of the study's that failed or was refused, for a quantity of a counting role or for one such a quantity is worked
    # out from (the quantity is then ``blocked`` by it): its repair.
    status_of = {r.get("by"): r["status"] for r in records.values()}
    bad = [i for i, r in records.items() if r["role"] in COUNTING and not r.get("fi_fault") and (
        r["status"] in ("failed", "refused") or (r["status"] == "blocked" and status_of.get(r.get("cause")) in ("failed", "refused")))]
    return out, records, (2 if faults else 1 if bad or figure_code else 0), document


def main():
    quest = os.path.dirname(HERE)
    record = fi_table.record_path(quest)
    out, records, code, document = run(record, quest)
    for ident, r in records.items():
        if r["status"] in ("failed", "refused", "blocked"):
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
    if os.environ.get("FI_FIGURE_FALLBACK") != "only":
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
