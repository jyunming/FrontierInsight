"""The frame of ``simulate.py``, written by FI from the study contract; the model writes only ``trial(cell, ctx)``.

Today the model writes the whole simulation program and FI then reads its text to guess whether the program did what the plan
said. With the frame, FI writes what the plan already decides: the entry points the trial harness calls (``run_trial`` for a
study with randomness, ``run_cell`` for one without), the seed (the model gets ``ctx.rng``), the thresholds, and the list of
results the plan declares. The model fills one named function, ``trial``, through the same path that fills the functions of
its package one at a time (:func:`core.function_steps.replace_function`). A result is checked against the declared list when
the trial assigns it (``fi_frame.TrialResult``), at run time, so a result under another name is seen at the first trial and an
empty slot in FI's small run, not after the study.

What the frame is NOT: it does not shorten the model's science or check that it is right (about 15 percent of the lines of
recorded simulation and analysis scripts are frame), and the analysis script is the model's still.

Older plans (``form: older``) get the same frame with a soft result check: a name outside the plan's list, or a declared result
no trial set, is recorded and said, never a failed trial. The stdlib-only runtime text (:data:`RUNTIME`) is written to
``code/fi_frame.py`` beside ``simulate.py``. Nothing here knows a field of science."""

from __future__ import annotations

import ast
import itertools
from collections import Counter
from typing import Any

from . import function_steps as _steps
from . import metric_spec as _metric_spec

#: Written into the frame; a quest keeps the frame it was written with (a resume reads the files, never writes them again).
VERSION = 1
RUNTIME_FILE = "fi_frame.py"
TOP_BEGIN = f"# >>> FI frame, version {VERSION}: written by FI from the plan; a repair does not change it"
TOP_END = "# <<< FI frame"
BOTTOM_BEGIN = f"# >>> FI frame, version {VERSION}: the functions FI's trial harness calls"
BOTTOM_END = "# <<< FI frame"
SLOT = "trial"
#: Ends the one line :func:`refuse` puts into a file whose frame was changed; a repair result that still holds it is refused.
RAISE_MARK = "# FI: the frame was changed"
#: The first words of :data:`RUNTIME`; a file called fi_frame.py is FI's only when it starts so.
RUNTIME_HEAD = '"""FI\'s frame of simulate.py'
SLOT_SIGNATURE = "def trial(cell, ctx)"

#: The runtime the frame imports. Standard library only (the study's own interpreter may have nothing else).
RUNTIME = '''"""FI's frame of simulate.py: the trial's context and its result record. Written by FI, never edited."""

import json
import os
import random


class Ctx:
    """What a trial is given besides its settings. ``rng``: a random generator seeded by FI for this trial (numpy's when it is
    installed, else ``random.Random``); ``seed``: the same seed as a plain integer for a library that wants one; ``trial``:
    the trial's number; ``kind``: ``"scan"``, ``"search"`` or ``"check"`` (what FI is asking for); ``thresholds``: the plan's
    frozen thresholds."""

    def __init__(self, seed=None, trial=0):
        self.seed = seed
        self.trial = trial
        self.kind = os.environ.get("FI_CELL_KIND") or "check"
        try:
            self.thresholds = dict(json.loads(os.environ.get("FI_THRESHOLDS") or "{}"))
        except (ValueError, TypeError):
            self.thresholds = {}
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            try:
                import numpy

                self._rng = numpy.random.default_rng(self.seed)
            except ImportError:
                self._rng = random.Random(self.seed)
        return self._rng


class UnknownResult(KeyError):
    """A result assigned under a name the plan does not declare."""


class TrialResult(dict):
    """What one trial returns: a ``dict`` filled by assignment (``out["name"] = value``). A name the plan declares is a
    result; any other is kept as a diagnostic (FI lists such names from what the trials returned; ``strict`` plans refuse it
    at once, naming the declared ones).
    ``declare`` is called by the frame, once, before any trial."""

    declared = frozenset()
    strict = False

    @classmethod
    def declare(cls, names, strict=False):
        cls.declared = frozenset(names)
        cls.strict = bool(strict)

    def _check(self, name):
        if name in self.declared:
            return
        if self.strict:
            raise UnknownResult("`%s` is not a result the plan declares; the declared results are: %s"
                                % (name, ", ".join(sorted(self.declared)) or "(none)"))

    def __setitem__(self, name, value):
        self._check(name)
        dict.__setitem__(self, name, value)

    def update(self, *args, **kwargs):
        for name, value in dict(*args, **kwargs).items():
            self[name] = value

    @classmethod
    def finish(cls, value):
        """The plain ``dict`` the harness reads. A model that returned a plain ``dict`` is checked the same way."""
        if isinstance(value, dict):
            for name in list(value):
                if name not in cls.declared:
                    cls._check(cls(), name)
            return dict(value)
        return value
'''

# What a made-up trial body looks like, shown to the model with the frame.
BODY_EXAMPLE = '''def trial(cell, ctx):
    """One trial at the setting ``cell``. Use ``ctx.rng`` for every random draw and ``ctx.thresholds`` for the plan's
    thresholds; return what the plan declares, by name."""
    out = TrialResult()
    out["name_of_a_declared_result"] = 0.0
    return out
'''


def declared_names(protocol: dict[str, Any] | None) -> list[str]:
    """The names a trial may return: every id the plan declares as a counting result (result, reference or physical bound)
    and the forms it may be returned under (``id``, ``id_count``, ``id_total``, ``id_values``), in the plan's order."""
    from . import contract as _contract

    roles = _contract.raw_roles(protocol)
    ids = [i for i, role in roles.items() if role in ("result", "reference", "physical_bound")]
    out: list[str] = []
    for ident in ids:
        for name in _metric_spec.result_forms(ident):
            if name not in out:
                out.append(name)
    return out


def declared_ids(protocol: dict[str, Any] | None) -> list[str]:
    """The ids (without the forms) of the results a trial is expected to set."""
    from . import contract as _contract

    return [i for i, role in _contract.raw_roles(protocol).items() if role in ("result", "reference", "physical_bound")]


def applies(design: Any) -> str:
    """``""`` when the frame can be written for this plan, else why not (said in run.log; the model then writes the whole
    script as before). The first slices cover a plan that declares at least one result and has none of: a search for the
    best design, a known-answer check that names no case (the model must write ``oracle()``
    for it), or a cluster job (a separate submit script)."""
    from . import oracle_check as _oracle

    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return "the plan has no protocol"
    protocol = design["protocol"]
    if not declared_ids(protocol):
        return "the plan declares no result"
    if protocol.get("optimisation") or design.get("optimisation"):
        return "the plan searches for a best design"
    for o in protocol.get("oracles") or []:
        if isinstance(o, dict) and _oracle.case_of(o) is None:
            return f"the check `{o.get('name')}` names no case"
    return ""


def frame_source(design: dict[str, Any], *, stochastic: bool, strict: bool = False) -> str:
    """The text of ``simulate.py`` with the frame and an empty ``trial``. ``stochastic``: ``run_trial`` (one call per seed), else
    ``run_cell``. ``strict``: a name outside the plan's list is refused (a plan in the contract form), else recorded."""
    protocol = design["protocol"]
    names = declared_names(protocol)
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    settings = [str(k) for k in grid]
    entry = (
        "def run_trial(cell, trial_id, seed):\n"
        "    return TrialResult.finish(trial(cell, Ctx(seed=seed, trial=trial_id)))\n"
        if stochastic else
        "def run_cell(cell):\n"
        "    return TrialResult.finish(trial(cell, Ctx(seed=None, trial=0)))\n")
    return (
        '"""The simulation. FI wrote the parts between the markers (the entry point, the list of results the plan declares);\n'
        'the study\'s own code is the function `trial` and any helper below it."""\n\n'
        f"{TOP_BEGIN}\n"
        f"FRAME_VERSION = {VERSION}  # the first statement: the model's imports go before the frame, never inside it\n"
        "import os as _os  # noqa: E402\nimport sys as _sys  # noqa: E402\n\n"
        "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        "from fi_frame import Ctx, TrialResult  # noqa: E402\n\n"
        f"SETTINGS = {tuple(settings)!r}  # the settings the plan varies\n"
        f"RESULTS = {tuple(names)!r}  # the names a trial returns\n"
        f"TrialResult.declare(RESULTS, strict={bool(strict)!r})\n"
        f"{TOP_END}\n\n\n"
        f"{SLOT_SIGNATURE}:\n"
        '    """One trial at the setting `cell`: draw with `ctx.rng`, read the thresholds from `ctx.thresholds`, return the\n'
        '    results by name (`out = TrialResult(); out["name"] = value; return out`)."""\n'
        f'    raise NotImplementedError("{_steps._STUB_MARK}")\n\n\n'
        f"{BOTTOM_BEGIN}\n{entry}{BOTTOM_END}\n")


# ---- the model's reply ------------------------------------------------------------------------------------------------

def fill(frame: str, reply: str) -> tuple[str | None, str]:
    """``(simulate.py, "")`` with the model's ``trial`` (its helpers and imports too) put into ``frame``, or ``(None, why)``.
    The model's function must take exactly ``(cell, ctx)``. Goes through :func:`core.function_steps.replace_function_ex`, the
    path that fills the functions of the model's package. The result is checked with :func:`frame_intact`."""
    text, why, _left = _fill(frame, reply)
    return text, why


def _fill(frame: str, reply: str) -> tuple[str | None, str, list[str]]:
    """:func:`fill` and, as the third value, what of the reply could not be carried into the file (plain phrases)."""
    text, why, left = _steps.replace_function_ex(frame, SLOT, reply, signature=SLOT_SIGNATURE, equation="trial", wide=True)
    if text is None:
        return None, why, []
    problem = frame_intact(frame, text)
    return (None, problem, []) if problem else (text, "", left)


def _regions(source: str) -> list[str] | None:
    out = []
    for begin in (TOP_BEGIN, BOTTOM_BEGIN):
        i = source.find(begin)
        if i < 0:
            return None
        j = source.find(TOP_END, i)
        if j < 0:
            return None
        out.append(source[i:j])
    return out


#: Names the frame binds once. A second binding outside the marked parts (a reply that defines its own ``run_trial``, or
#: ``TrialResult``) or a change to their attributes (``TrialResult.strict = False``) changes what the harness runs.
_PROTECTED = frozenset({"run_trial", "run_cell", "Ctx", "TrialResult", "UnknownResult", "fi_frame"})


def _targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in _targets(e)]
    if isinstance(node, ast.Starred):
        return _targets(node.value)
    return []


def _module_level(body: list[ast.stmt]) -> list[ast.stmt]:
    """The statements that run when the module is loaded, through its ``if`` / ``try`` / ``with`` / loops (not into a
    function or a class)."""
    out: list[ast.stmt] = []
    for n in body:
        out.append(n)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(n, field, None)
            if isinstance(inner, list):
                out.extend(_module_level(inner))
        for h in getattr(n, "handlers", []) or []:
            out.extend(_module_level(h.body))
    return out


def _base(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _pinned(tree: ast.Module) -> Counter:
    """What binds or changes the frame's own names, read as syntax: ``{what: how many}``. The frame's text has each once."""
    out: Counter = Counter()
    for n in _module_level(tree.body):
        names: list[str] = []
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, ast.Assign):
            names = [x for t in n.targets for x in _targets(t)]
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            names = _targets(n.target)
        elif isinstance(n, ast.Delete):
            names = [x for t in n.targets for x in _targets(t)]
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            if (isinstance(n, ast.ImportFrom) and n.module == "fi_frame") or any(a.name == "fi_frame" for a in n.names):
                out["an import of fi_frame"] += 1
        for name in names:
            if name in _PROTECTED:
                out[f"a binding of `{name}`"] += 1
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del)) and _base(n) in _PROTECTED:
            out[f"a change to `{_base(n)}.{n.attr}`"] += 1
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == "declare" and _base(f) in _PROTECTED:
                out["a call of `TrialResult.declare`"] += 1
            elif isinstance(f, ast.Name) and f.id in ("setattr", "delattr") and n.args and _base(n.args[0]) in _PROTECTED:
                out[f"a change to `{_base(n.args[0])}`"] += 1
    return out


def _short(node: ast.AST | None, other: ast.AST | None = None) -> str:
    """The first line of ``node``; when ``other`` starts with the same line (a function whose body changed), the first line
    where they differ."""
    if node is None:
        return "nothing"
    lines = ast.unparse(node).splitlines()
    text = lines[0]
    if other is not None:
        theirs = ast.unparse(other).splitlines()
        if theirs and theirs[0] == text:
            text = next((a.strip() for a, b in itertools.zip_longest(lines, theirs, fillvalue="") if a != b and a.strip()),
                        text)
    return text if len(text) <= 70 else text[:67] + "..."


def _param_shape(args: ast.arguments) -> tuple:
    """The names (and how many have defaults) of a parameter list; an annotation is not a different parameter."""
    return ([a.arg for a in args.posonlyargs], [a.arg for a in args.args], args.vararg.arg if args.vararg else None,
            [a.arg for a in args.kwonlyargs], args.kwarg.arg if args.kwarg else None, len(args.defaults),
            len(args.kw_defaults))


def frame_intact(old: str, new: str) -> str:
    """``""`` when ``new`` has exactly the frame ``old`` has (both regions, compared as syntax, so a change of spacing does not
    count), the slot ``trial`` with the same parameters, and nothing outside the regions that binds or changes the frame's
    own names (a ``run_trial`` or ``TrialResult`` of the reply's own, ``TrialResult.strict = False``); else a plain sentence
    saying what changed. All of it is read as syntax."""
    old, new = old.lstrip("\ufeff"), new.lstrip("\ufeff")
    before, after = _regions(old), _regions(new)
    if before is None:
        return ""
    if after is None:
        return "the frame FI wrote was changed: the marked parts of simulate.py are missing"
    try:
        for label, a, b in zip(("first", "second"), before, after):
            for x, y in itertools.zip_longest(ast.parse(a).body, ast.parse(b).body):
                if x is None or y is None or ast.dump(x) != ast.dump(y):
                    return (f"the frame FI wrote was changed: in the {label} marked part of simulate.py, `{_short(x, y)}` "
                            f"is now `{_short(y, x)}`")
    except SyntaxError as e:
        return f"the frame is not valid Python ({e.msg}, line {e.lineno})"
    slot_old, slot_new = _steps._find(old, SLOT), _steps._find(new, SLOT)
    if slot_new is None:
        return f"the function `{SLOT}` is missing"
    if slot_old is not None and _param_shape(slot_old.args) != _param_shape(slot_new.args):
        return f"the parameters of `{SLOT}` were changed; they must be (cell, ctx)"
    try:
        was, now = _pinned(ast.parse(old)), _pinned(ast.parse(new))
    except SyntaxError as e:
        return f"the file is not valid Python ({e.msg}, line {e.lineno})"
    if was != now:
        more, less = sorted((now - was).keys()), sorted((was - now).keys())
        return ("the frame FI wrote was changed: the reply adds " + ", ".join(more) if more
                else "the frame FI wrote was changed: the reply removes " + ", ".join(less)) + " outside the marked parts of simulate.py"
    return ""


# ---- a repair of a framed simulate.py ---------------------------------------------------------------------------------

#: Put first in every request to change the code of a study whose ``simulate.py`` has FI's frame.
REPAIR_NOTE = (
    "FIRST, what you may change: FI wrote the frame of simulate.py, the parts between its `# >>> FI frame` and "
    "`# <<< FI frame` lines (the `fi_frame` import, SETTINGS, RESULTS and the function `run_trial` or `run_cell`), and the file "
    "fi_frame.py. Only the function `trial(cell, ctx)` and the helper functions and imports it uses may change. Give each "
    "function you change whole, with its own parameters (in `functions` when the reply has that key); do not give the whole "
    "file, and do not seed, redefine `run_trial`/`run_cell`/`TrialResult` or touch fi_frame.py. A change to the frame is "
    "refused before anything runs and the repair counts as not made. This replaces any instruction below to return the "
    "whole of simulate.py.\n\n")


def repair_note(source: str) -> str:
    """:data:`REPAIR_NOTE` when ``source`` (the text of simulate.py) has FI's frame, else ``""`` (a study without a frame is
    asked exactly as before)."""
    return REPAIR_NOTE if has_frame(source or "") else ""


def repair_problem(old: str, new: str, name: str) -> str:
    """Why ``new`` cannot replace the file ``name`` whose text is ``old``, as far as FI's frame is concerned, else ``""``.
    ``simulate.py`` with a frame: :func:`frame_intact`. ``fi_frame.py``: FI's runtime, which a repair never changes. Any
    other file, and a ``simulate.py`` without a frame, is not this check's (so nothing changes when the frame is off)."""
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    old, new = (old or "").lstrip("\ufeff"), (new or "").lstrip("\ufeff")
    if name == RUNTIME_FILE:  # only FI's own runtime; a module of the quest's that happens to share the name is not
        return "" if new == old or not old.startswith(RUNTIME_HEAD) else (
            "the frame FI wrote was changed: fi_frame.py is FI's own file and a repair does not edit it")
    if name != "simulate.py" or not has_frame(old):
        return ""
    if RAISE_MARK in new:
        return ("the frame FI wrote was changed: the line that stops the run because of that is still in simulate.py "
                "(remove it; the frame itself is FI's)")
    return frame_intact(old, new)


def without_raise(text: str) -> str:
    """``text`` without the line :func:`refuse` put in: it is FI's own line, so a repair of the file never has to remove it and
    never keeps it."""
    if not text or RAISE_MARK not in text:
        return text
    return "".join(ln for ln in text.splitlines(keepends=True) if RAISE_MARK not in ln)


def heal(frame: str, text: str) -> str:
    """The canonical frame with what :func:`heal_ex` could keep of ``text``."""
    return heal_ex(frame, text)[0]


def heal_ex(frame: str, text: str) -> tuple[str, str]:
    """``(the canonical frame with the function trial, the helpers and the imports of text put into it, "")``, or ``(the bare
    frame, why they could not be)``. The damaged frame is never kept: a repair is compared with the frame FI wrote, whatever
    the model's reply did to it."""
    kept = text.lstrip("\ufeff")
    for begin in (TOP_BEGIN, BOTTOM_BEGIN):
        i = kept.find(begin)
        j = kept.find(TOP_END, i) if i >= 0 else -1
        if i >= 0 and j >= 0:
            k = kept.find("\n", j)
            kept = kept[:i] + kept[k + 1:] if k >= 0 else kept[:i]
    kept = without_raise(kept)
    made, why, _left = _fill(frame, kept)
    return (made, "") if made is not None else (frame, why)


def refuse(text: str, problem: str) -> str:
    """``text`` with a first statement that stops the load with ``problem`` (after the docstring and ``__future__`` imports),
    so a file whose frame was changed is never run as it stands: the small run says so, in the words of ``problem``, and the
    simulation goes back to be repaired."""
    line = f"raise RuntimeError({problem!r})  {RAISE_MARK}\n"
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return line + text
    after = 0
    for n in tree.body:
        doc = isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str)
        if doc or (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
            after = n.end_lineno or after
        else:
            break
    lines = text.splitlines(keepends=True)
    return "".join(lines[:after]) + line + "".join(lines[after:])


def has_frame(source: str) -> bool:
    return _regions(source) is not None


def compose(frame: str, simulate_reply: str) -> tuple[str, str]:
    """What becomes ``simulate.py`` when the model was asked for ``trial`` only: ``(source, note)``. A reply that already holds
    the whole frame is accepted when :func:`frame_intact` says so; a reply that is a whole simulation of its own
    (``run_trial`` / ``run_cell`` and no ``trial``) is kept as it is, with a note: the plan then runs as before, without the frame.
    A reply that holds a ``trial`` that does not fit is returned unchanged with the reason (the checks of the code, and the
    harness, say what is wrong with it)."""
    if has_frame(simulate_reply):
        problem = frame_intact(frame, simulate_reply)
        return (without_raise(simulate_reply) if not problem else simulate_reply), problem
    try:
        tree = ast.parse(simulate_reply)
    except SyntaxError:
        return simulate_reply, "the reply is not valid Python"
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if SLOT not in names:
        if names & {"run_trial", "run_cell"}:
            return simulate_reply, "the reply is a whole simulation without the function `trial`: kept as written, without the frame"
        return simulate_reply, "the reply holds no function `trial`"
    text, why, left = _fill(frame, simulate_reply)
    if text is None:
        # The file keeps FI's frame (a later repair is checked against it); the run then stops with the reason.
        return refuse(heal(frame, simulate_reply), why), why
    return text, (("left out of the model's reply, which the file cannot carry: " + "; ".join(left)) if left else "")


# ---- what the small run shows ----------------------------------------------------------------------------------------

def empty_slots(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The declared results no trial of the run returned, under any of their forms (``returned``: the names the entry gave)."""
    got = {str(n) for n in returned or []}
    return [i for i in declared_ids(protocol) if not any(f in got for f in _metric_spec.result_forms(i))]


def unknown_names(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The names the entry returned that the plan does not declare under any form (diagnostics: kept, never reported)."""
    known = set(declared_names(protocol))
    return sorted({str(n) for n in returned or []} - known)
