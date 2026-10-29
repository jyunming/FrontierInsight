"""The frame of ``simulate.py``, written by FI from the study contract; the model writes only ``trial(cell, ctx)``.

Today the model writes the whole simulation program and FI then reads its text to guess whether the program did what the plan
said. With the frame, FI writes what the plan already decides: the entry points the trial harness calls (``run_trial`` for a
study with randomness, ``run_cell`` for one without), the seed (the model gets ``ctx.rng``), the thresholds, and the list of
results the plan declares. The model fills one named function, ``trial``, through the same path that fills the functions of
its package one at a time (:func:`core.function_steps.replace_function`). A result is checked against the declared list when
the trial assigns it (``fi_frame.TrialResult``), at run time, so a result under another name is seen at the first trial and an
empty slot in FI's small run, not after the study.

The frame fits every form of study: a sweep, sub-studies (a cell holds every setting, the ones a part does not vary at the value
it holds), values compared inside each run (``cell[factor]`` is the list of levels; ``out.each``), a search for the best design
(``ctx.kind``; the objective and the constrained quantities are the results), checks against a known answer (``oracle()`` beside
``trial`` for a check with no case), a trial answered for all its seeds at once (``trial_batch``), a cluster job and a standalone
``run.py``. A quest keeps the frame it was written with (:func:`frame_for`); a changed plan writes it again.

What the frame is NOT: it does not shorten the model's science or check that it is right (about 15 percent of the lines of
recorded simulation and analysis scripts are frame), and the analysis script is the model's still.

Older plans (``form: older``) get the same frame with a soft result check: a name outside the plan's list, or a declared result
no trial set, is recorded and said, never a failed trial. The stdlib-only runtime text (:data:`RUNTIME`) is written to
``code/fi_frame.py`` beside ``simulate.py``. Nothing here knows a field of science."""

from __future__ import annotations

import ast
import importlib
import itertools
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import function_steps as _steps
from . import metric_spec as _metric_spec

#: Written into the frame; a quest keeps the frame it was written with (a resume reads the files, never writes them again).
VERSION = 2
RUNTIME_FILE = "fi_frame.py"
TOP_BEGIN = f"# >>> FI frame, version {VERSION}: written by FI from the plan; a repair does not change it"
TOP_END = "# <<< FI frame"
BOTTOM_BEGIN = f"# >>> FI frame, version {VERSION}: the functions FI's trial harness calls"
BOTTOM_END = "# <<< FI frame"
#: The same markers whatever version wrote them: a quest resumed with a newer FI still has the frame it was written with.
_BEGINS = (re.compile(r"# >>> FI frame, version (\d+): written by FI from the plan; a repair does not change it"),
           re.compile(r"# >>> FI frame, version (\d+): the functions FI's trial harness calls"))
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
import sys

FRAME_RUNTIME_VERSION = @VERSION@


def _generator(seed):
    try:
        import numpy

        return numpy.random.default_rng(seed)
    except ImportError:
        return random.Random(seed)


class Ctx:
    """What a trial is given besides its settings. ``rng``: a random generator seeded by FI for this trial (numpy's when it is
    installed, else ``random.Random``); ``seed``: the same seed as a plain integer for a library that wants one; ``trial``:
    the trial's number; ``kind``: ``"scan"``, ``"search"`` or ``"check"`` (what FI is asking for); ``thresholds``: the plan's
    frozen thresholds. A batch (``trial_batch``) is given ``seeds`` and ``trials``, one of each for every trial of the
    setting, in the order its results are returned in; ``rngs()`` is one generator for each of those seeds and ``results()``
    one empty result record for each."""

    def __init__(self, seed=None, trial=0, seeds=None, trials=None):
        self.seed = seed
        self.trial = trial
        self.seeds = list(seeds) if seeds is not None else [seed]
        self.trials = list(trials) if trials is not None else [trial]
        self.kind = os.environ.get("FI_CELL_KIND") or "check"
        try:
            self.thresholds = dict(json.loads(os.environ.get("FI_THRESHOLDS") or "{}"))
        except (ValueError, TypeError):
            self.thresholds = {}
        self._rng = None

    @property
    def rng(self):
        if self._rng is None:
            self._rng = _generator(self.seed)
        return self._rng

    def rngs(self):
        """One random generator for each seed of ``seeds``, in that order."""
        return [_generator(s) for s in self.seeds]

    def results(self):
        """One empty result record for each trial of ``trials``, in that order: a batch fills them and returns the list."""
        out = []
        for trial in self.trials:
            record = TrialResult()
            record._trial = trial
            out.append(record)
        return out


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

    def each(self, name, values):
        """A result the trial gives for every value of what each run compares inside it: a mapping ``{level: number}`` or,
        with one factor compared, a list of numbers in the order of the levels (``cell[factor]``). FI reads it by the same
        rule whichever way it is written."""
        if isinstance(values, dict):
            self[name] = dict(values)
        else:
            self[name] = list(values.tolist() if hasattr(values, "tolist") else values)

    @classmethod
    def finish(cls, value):
        """The plain ``dict`` the harness reads. A model that returned a plain ``dict`` is checked the same way."""
        if isinstance(value, dict):
            for name in list(value):
                if name not in cls.declared:
                    cls._check(cls(), name)
            return dict(value)
        return value


class BatchError(Exception):
    """A batch function that did not give one result for each seed it was given, in order."""


_batches = {}


def _cell_text(cell):
    return json.dumps([[str(k), repr(v)] for k, v in sorted(dict.items(cell), key=lambda kv: str(kv[0]))])


def _harness_trials():
    """The ``[[trial, seed]]`` of the setting the harness (``.fi/trial_harness.py``) is running: read from its ``main()``, where
    its spec is. The harness has no line for this (its text, and every key made from it, is the same with or without the
    frame), so the frame looks at the spec it was given."""
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "main":
            spec = frame.f_locals.get("spec")
            trials = spec.get("trials") if isinstance(spec, dict) else None
            if isinstance(trials, list) and "module" in spec and "entry" in spec:
                return [[t["trial"], t.get("seed")] for t in trials if isinstance(t, dict) and "trial" in t]
        frame = frame.f_back
    return None


def _trials_of(ctx):
    """``[(trial, seed)]`` of the setting this trial belongs to (the harness's spec, or a ``run.py`` that says them in
    FI_TRIAL_SEEDS), else this trial."""
    mine = (ctx.trial, ctx.seed)
    try:
        given = _harness_trials()
        if given is None:
            given = json.loads(os.environ.get("FI_TRIAL_SEEDS") or "[]")
        rows = [(int(t), s) for t, s in given]
    except (ValueError, TypeError, KeyError):
        rows = []
    return rows if mine in rows else [mine]


def _one_per_seed(label, out, ctx):
    if not isinstance(out, (list, tuple)) or len(out) != len(ctx.seeds):
        raise BatchError("`%s` must return a list with one result for each of the %d seed(s) in ctx.seeds, in that order; it "
                         "returned %s" % (label, len(ctx.seeds), ("%d result(s)" % len(out)) if isinstance(out, (list, tuple))
                                                          else type(out).__name__))
    for i, (record, trial) in enumerate(zip(out, ctx.trials)):
        if not isinstance(record, dict):
            raise BatchError("result %d of `%s` is a %s, not a result record" % (i, label, type(record).__name__))
        belongs = getattr(record, "_trial", None)
        if belongs is None and len(out) > 1:
            raise BatchError("result %d of `%s` is not one of ctx.results(): with more than one seed, FI can tell that the "
                             "results are in the order of ctx.seeds only from the records ctx.results() gives; fill those "
                             "and return them" % (i, label))
        if belongs is not None and belongs != trial:
            raise BatchError("result %d of `%s` is the record of trial %s but the seed at that place is trial %s's: keep the "
                             "results in the order of ctx.seeds" % (i, label, belongs, trial))
    return list(out)


def batch_of_trials(fn, cell, ctx):
    """What the entry point gives for one trial when the study writes ``trial_batch(cell, ctx)``: the batch runs once for the
    setting, with every seed of it, and each trial is given its own result."""
    plan = _trials_of(ctx)
    key = (_cell_text(cell), tuple(plan))
    if key not in _batches:
        _batches.clear()  # a process is one setting's
        batch_ctx = Ctx(seed=plan[0][1], trial=plan[0][0], seeds=[s for _t, s in plan], trials=[t for t, _s in plan])
        try:
            _batches[key] = (True, _one_per_seed("trial_batch", fn(cell, batch_ctx), batch_ctx))
        except Exception as e:  # noqa: BLE001 -- the same failure is every trial's of the setting
            _batches[key] = (False, e)
    ok, got = _batches[key]
    if not ok:
        raise got
    return got[plan.index((ctx.trial, ctx.seed))]


def batch_of_cells(fn, cell, ctx):
    """What the entry point gives for one trial when the study writes ``cells_batch(cells, ctx)``: FI runs one setting in a
    process of its own, so the batch it is given is that setting alone."""
    out = fn([cell], ctx)
    if not isinstance(out, (list, tuple)) or len(out) != 1:
        raise BatchError("`cells_batch` must return a list with one result for each cell it is given (1); it returned %s"
                         % (("%d result(s)" % len(out)) if isinstance(out, (list, tuple)) else type(out).__name__))
    return out[0]
'''.replace("@VERSION@", str(VERSION))

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

    ids = declared_ids(protocol)
    out: list[str] = []
    for ident in ids:
        for name in _metric_spec.result_forms(ident):
            if name not in out:
                out.append(name)
    return out


def _search_quantities(protocol: dict[str, Any] | None) -> list[str]:
    """The names a search for the best design reads from each evaluation: the objective's quantity and the quantity of every
    constraint (a search plan does not list them among its ``metrics``)."""
    block = protocol.get("optimisation") if isinstance(protocol, dict) else None
    if not isinstance(block, dict):
        return []
    found = [(block.get("objective") or {}).get("quantity") if isinstance(block.get("objective"), dict) else None]
    constraints = block.get("constraints")
    for c in [constraints] if isinstance(constraints, dict) else constraints if isinstance(constraints, list) else []:
        found.append(c.get("quantity") if isinstance(c, dict) else None)
    return [q.strip() for q in found if isinstance(q, str) and q.strip()]


def declared_ids(protocol: dict[str, Any] | None) -> list[str]:
    """The ids (without the forms) of the results a trial is expected to set: the plan's results, and, for a search for the
    best design, the objective and each constrained quantity."""
    from . import contract as _contract

    ids = [i for i, role in _contract.raw_roles(protocol).items() if role in ("result", "reference", "physical_bound")]
    return ids + [q for q in dict.fromkeys(_search_quantities(protocol)) if q not in ids]


def settings_of(protocol: dict[str, Any] | None) -> list[str]:
    """The names a cell holds: the grid's settings and what each run compares inside it (``in_each_run``: ``cell[name]`` is the
    list of its levels); for a search, the conditions held fixed, the numerical settings, the design variables and what each
    evaluation compares inside it. A sub-study's cell holds every setting too (the ones it does not vary at the value it
    holds), so the union is every cell's list."""
    protocol = protocol if isinstance(protocol, dict) else {}
    names = [str(k) for k in protocol["grid"]] if isinstance(protocol.get("grid"), dict) else []
    block = protocol.get("optimisation")
    if isinstance(block, dict):
        from . import optimise_search as _search

        try:
            names += [*_search.cell_of(block, {}), *_search.check_settings(block)]
        except (KeyError, TypeError, ValueError):
            pass  # a block that cannot be read is the plan check's to name
        names += [str(v["name"]) for v in block.get("design_variables") or [] if isinstance(v, dict) and v.get("name")]
    elif isinstance(protocol.get("in_each_run"), dict):
        names += [str(k) for k in protocol["in_each_run"]]
    return list(dict.fromkeys(names))


def applies(design: Any) -> str:
    """``""`` when the frame can be written for this plan, else why not (said in run.log; the model then writes the whole
    script as before). Every form of study has one: a sweep over settings, sub-studies, values compared inside each run, a
    search for the best design (its objective and constrained quantities are the results), checks against a known answer
    (one that names no case is answered by an ``oracle()`` the model writes beside ``trial``) and a cluster job. What the
    frame needs is a result to declare."""
    if not isinstance(design, dict) or not isinstance(design.get("protocol"), dict):
        return "the plan has no protocol"
    if not declared_ids(design["protocol"]):
        return "the plan declares no result"
    return ""


def case_less_checks(protocol: dict[str, Any] | None) -> list[str]:
    """The names of the plan's checks that name no case FI can run: the model answers them with ``oracle()`` in simulate.py."""
    from . import oracle_check as _oracle

    return [str(o.get("name")) for o in (protocol.get("oracles") if isinstance(protocol, dict) else None) or []
            if isinstance(o, dict) and _oracle.case_of(o) is None]


def frame_source(design: dict[str, Any], *, stochastic: bool, strict: bool = False) -> str:
    """The text of ``simulate.py`` with the frame and an empty ``trial``. ``stochastic``: ``run_trial`` (one call per seed), else
    ``run_cell``. ``strict``: a name outside the plan's list is refused (a plan in the contract form), else recorded."""
    protocol = design["protocol"]
    names = declared_names(protocol)
    settings = settings_of(protocol)
    entry = (
        "def run_trial(cell, trial_id, seed):\n"
        "    return TrialResult.finish(trial(cell, Ctx(seed=seed, trial=trial_id)))\n"
        if stochastic else
        "def run_cell(cell):\n"
        "    return TrialResult.finish(trial(cell, Ctx(seed=None, trial=0)))\n")
    top = (
        f"{TOP_BEGIN}\n"
        f"FRAME_VERSION = {VERSION}  # the first statement: the model's imports go before the frame, never inside it\n"
        "import os as _os  # noqa: E402\nimport sys as _sys  # noqa: E402\n\n"
        "_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))\n"
        "from fi_frame import Ctx, TrialResult  # noqa: E402\n\n"
        f"SETTINGS = {tuple(settings)!r}  # the settings a cell holds\n"
        f"RESULTS = {tuple(names)!r}  # the names a trial returns\n"
        f"TrialResult.declare(RESULTS, strict={bool(strict)!r})\n"
        f"{TOP_END}\n")
    return _assemble(top, f"{BOTTOM_BEGIN}\n{entry}{BOTTOM_END}\n")


def _assemble(top: str, bottom: str) -> str:
    """The frame's text from its two marked parts (each whole, with its closing marker line)."""
    return (
        '"""The simulation. FI wrote the parts between the markers (the entry point, the list of results the plan declares);\n'
        'the study\'s own code is the function `trial` and any helper below it."""\n\n'
        f"{top}\n\n"
        f"{SLOT_SIGNATURE}:\n"
        '    """One trial at the setting `cell`: draw with `ctx.rng`, read the thresholds from `ctx.thresholds`, return the\n'
        '    results by name (`out = TrialResult(); out["name"] = value; return out`)."""\n'
        f'    raise NotImplementedError("{_steps._STUB_MARK}")\n\n\n'
        f"{bottom}")


# ---- the model's reply ------------------------------------------------------------------------------------------------

def fill(frame: str, reply: str) -> tuple[str | None, str]:
    """``(simulate.py, "")`` with the model's ``trial`` (its helpers and imports too) put into ``frame``, or ``(None, why)``.
    The model's function must take exactly ``(cell, ctx)``. Goes through :func:`core.function_steps.replace_function_ex`, the
    path that fills the functions of the model's package. The result is checked with :func:`frame_intact`."""
    text, why, _left = _fill(frame, reply)
    return text, why


#: The batch forms a study may give instead of ``trial``: ``(function, the runtime function that calls it, its parameters)``.
BATCH_FORMS = (("trial_batch", "batch_of_trials", ["cell", "ctx"]), ("cells_batch", "batch_of_cells", ["cells", "ctx"]))


def batch_form(reply: str) -> str:
    """The name of the batch function a reply gives (``trial_batch`` or ``cells_batch``) when it gives no ``trial``, else ``""``."""
    try:
        names = {n.name for n in ast.parse(reply).body if isinstance(n, ast.FunctionDef)}
    except SyntaxError:
        return ""
    return "" if SLOT in names else next((f for f, _h, _p in BATCH_FORMS if f in names), "")


def gives_batch(path: Any) -> bool:
    """Whether the simulation in the file ``path`` answers all the trials of a setting in one call (``trial_batch``)."""
    try:
        return any(isinstance(n, ast.FunctionDef) and n.name == "trial_batch"
                   for n in ast.parse(Path(path).read_text(encoding="utf-8")).body)
    except (OSError, SyntaxError, ValueError):
        return False


def _adapted(reply: str) -> tuple[str, str]:
    """``(reply, "")`` with a ``trial`` written by FI that calls the batch function the reply gives (``trial_batch(cell, ctx)``
    answering all the seeds of a setting at once, ``cells_batch(cells, ctx)``), or ``(reply, why)`` when its parameters are not
    the ones FI calls it with. A reply with ``trial`` is left as it is."""
    form = batch_form(reply)
    if not form:
        return reply, ""
    helper, want = next((h, p) for f, h, p in BATCH_FORMS if f == form)
    node = next(n for n in ast.parse(reply).body if isinstance(n, ast.FunctionDef) and n.name == form)
    if _steps._param_names(node) != want:
        return reply, (f"`{form}` must take exactly ({', '.join(want)}), as FI calls it; the reply has "
                       f"({', '.join(_steps._param_names(node))})")
    return (reply.rstrip("\n") + "\n\n\ndef trial(cell, ctx):\n"
            f"    # FI: this study gives `{form}`; FI calls it for each trial\n"
            f"    from fi_frame import {helper}\n\n"
            f"    return {helper}({form}, cell, ctx)\n"), ""


def _fill(frame: str, reply: str) -> tuple[str | None, str, list[str]]:
    """:func:`fill` and, as the third value, what of the reply could not be carried into the file (plain phrases)."""
    reply, why = _adapted(reply)
    if why:
        return None, why, []
    text, why, left = _steps.replace_function_ex(frame, SLOT, reply, signature=SLOT_SIGNATURE, equation="trial", wide=True)
    if text is None:
        return None, why, []
    problem = frame_intact(frame, text)
    return (None, problem, []) if problem else (text, "", left)


def _spans(source: str) -> list[tuple[int, int, int]] | None:
    """``[(start, end of the marked text, end of its closing line)]`` of the frame's two parts, whatever version wrote them."""
    out = []
    for begin in _BEGINS:
        m = begin.search(source)
        if m is None:
            return None
        j = source.find(TOP_END, m.start())
        if j < 0:
            return None
        k = source.find("\n", j)
        out.append((m.start(), j, k + 1 if k >= 0 else len(source)))
    return out


def _regions(source: str) -> list[str] | None:
    spans = _spans(source)
    if spans is None:
        return None
    out = [source[i:j] for i, j, _k in spans]
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
    "refused before anything runs and the repair counts as not made.\n\n")


#: The first sentences of agents/execute_reflect.md and what they say when FI wrote the frame (the frame owns the entry, the
#: seeds, the pilot and what the script prints).
REFLECT_INTRO = ("Produce a corrected script that runs to completion on the available CPU venv and emits `RESULT_JSON: {...}` as "
                 "the final line of stdout. The fix must address the actual bug shown in the traceback.")
REFLECT_INTRO_FRAME = ("Produce a corrected `trial` (and the helper functions it uses) that runs to completion on the available "
                       "CPU venv and returns the plan's results through FI's frame. The fix must address the actual bug shown in "
                       "the traceback.")


def framed_prompt(source: str, prompt: str) -> str:
    """The repair request ``prompt`` (made from agents/execute_reflect.md) for a simulate.py with text ``source``: the repair
    note first and the intro that speaks of the frame's own business replaced, when ``source`` has FI's frame; ``prompt``
    unchanged otherwise."""
    if not has_frame(source or ""):
        return prompt
    return repair_note(source) + prompt.replace(REFLECT_INTRO, REFLECT_INTRO_FRAME)


REPAIR_NOTE_WHOLE = (
    "FIRST, what you may change: FI wrote the frame of simulate.py, the parts between its `# >>> FI frame` and "
    "`# <<< FI frame` lines (the `fi_frame` import, SETTINGS, RESULTS and the function `run_trial` or `run_cell`), and the file "
    "fi_frame.py. Only the function `trial(cell, ctx)` and the helper functions and imports it uses may change. Give the whole "
    "file with the frame's lines exactly as they are: do not seed, redefine `run_trial`/`run_cell`/`TrialResult` or touch "
    "fi_frame.py. A change to the frame is refused before anything runs and the repair counts as not made.\n\n")


def repair_note(source: str, whole: bool = False) -> str:
    """:data:`REPAIR_NOTE` when ``source`` (the text of simulate.py) has FI's frame, else ``""`` (a study without a frame is
    asked exactly as before). ``whole``: the reply is the whole file (:data:`REPAIR_NOTE_WHOLE`), not functions."""
    return (REPAIR_NOTE_WHOLE if whole else REPAIR_NOTE) if has_frame(source or "") else ""


def repair_problem(old: str, new: str, name: str) -> str:
    """Why ``new`` cannot replace the file ``name`` whose text is ``old``, as far as FI's frame is concerned, else ``""``.
    ``simulate.py`` with a frame: :func:`frame_intact`. ``fi_frame.py``: FI's runtime, which a repair never changes.
    ``experiment.py`` and ``derive.py`` when FI wrote them (the analysis, core/analysis_frame.py): :func:`analysis_problem`.
    Any other file, and a ``simulate.py`` without a frame, is not this check's (so nothing changes when the frame is off)."""
    name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    old, new = (old or "").lstrip("\ufeff"), (new or "").lstrip("\ufeff")
    if name in ANALYSIS_FILES:
        return _analysis_problem(old, new, name)
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
    for _ in _BEGINS:
        spans = _spans(kept)
        if spans is None:
            break
        i, _j, k = spans[0]
        kept = kept[:i] + kept[k:]
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


def level_rule(protocol: Any, source: str) -> str:
    """FI's one rule for a result compared inside each run, for a repair of a simulate.py that has FI's frame (``""``
    otherwise, and when the plan compares nothing inside each run). It is the one exception to "never return NaN in place of a
    failed trial": the request that writes the code (`core.run_checks`) and every repair say it in these words."""
    from . import in_run as _in_run

    if not has_frame(source or ""):
        return ""
    inside = _in_run.of_study(protocol) or {}
    if not inside:
        return ""
    names = ", ".join(f"`{k}`" for k in inside)
    return (f"The one exception: for a result the plan compares inside each run (over the values of {names}), a value where "
            "the result does not exist is NaN (or None) under that value's key, and EVERY compared value is present as a key "
            "(`out.each(\"name\", {value: number})`, keyed " + _in_run.key_forms(_in_run.level_cells(inside)) + "); a value left "
            "out makes the trial fail.")


# The repairs of a simulate.py that has FI's frame: the function the model gives is `trial` and its helpers, whole; the
# frame is FI's. Each replaces the text of the same repair for a simulate.py the model wrote whole.
REFLECT_SIMULATE = (
    "SPLIT EXPERIMENT: this script is simulate.py, the simulation half of a two-script experiment. FI wrote its frame and calls "
    "the function `trial(cell, ctx)` once per trial, keeping the record itself; the trials failed, or what FI ran does not "
    "match the frozen protocol (the reason is below). Repair `trial` and the helper functions it uses, each function you "
    "change given whole. `trial` returns the plan's results (`out = TrialResult()`, `out[\"name\"] = value`) and raises on a "
    "failed trial, with no loop over settings, no files and no RESULT_JSON. Keep every comment that marks where an equation "
    "of the plan's model is computed (`# E1`).\n")

REFLECT_SEARCH = (
    "THIS STUDY IS A SEARCH FOR THE BEST DESIGN: FI calls `trial(cell, ctx)` once for each design it tries; `cell` holds the "
    "design variables, the conditions the plan holds fixed and the search's numerical settings, by name. `trial` returns the "
    "objective and every constrained quantity under the plan's own names, and never searches or loops over designs itself.\n")

ORACLE_NOTE = (
    "\n\nTHE TRIAL CONTRACT: an oracle that names a `case` and a `measure` is NOT measured by anything you write for the "
    "check: FI calls `trial` itself on that case and computes the `measure` (one returned name, or a formula of them) from "
    "what it returns, so `trial` must return every name the `measure` uses, computed by the real simulation. `trial` is ONE "
    "trial: never average many trials inside it. Only an oracle without a `case` is measured by `def oracle() -> dict` in "
    "simulate.py, which computes each such check with the same simulation code and returns them as a dict keyed by the "
    "check's name. Keep every comment that marks where an equation of "
    "the plan's model is computed (`# E1`). When you fix the simulation, give each function you change whole (`trial` and "
    "its helpers; the frame is FI's); when the only thing wrong is a check itself, return its `oracle_change` and leave "
    "`code` and `functions` empty, as said above.\n")

RERUN_NOTE = (
    "\nThe experiment is two scripts. In the `# file: simulate.py` block give `trial` and its helpers back exactly as they are "
    "unless the problem the review found is IN the simulation: the trials FI already ran are then used as they are, and only "
    "experiment.py runs again. Change `trial` only when the simulation itself has to change.\n")

EXTEND_REPLY = ("Return what the reply format above asks for: in the `# file: simulate.py` block only `trial` and the helpers "
                "it uses (FI keeps the frame), and experiment.py whole.")


# ---- a repair of the analysis FI wrote (derive.py: the model's bodies in FI's frame; experiment.py is all FI's) -------------

#: The files of the analysis, when FI writes it (core/analysis_frame.py): a repair of either goes through that module's check.
ANALYSIS_FILES = ("derive.py", "experiment.py")
#: The first line of the part of derive.py that is FI's.
ANALYSIS_MARK = "# >>> FI analysis frame"
#: The parameters of each kind of body (the same as core/analysis_frame.py: SIGNATURES).
BODY_PARAMS = {"derive": "table", "reference": "setting", "figure": "fig, table"}


def _analysis_problem(old: str, new: str, name: str) -> str:
    try:
        _analysis = importlib.import_module(f"{__package__}.analysis_frame")
    except ModuleNotFoundError as exc:  # a checkout without the analysis frame has nothing to check
        if exc.name != f"{__package__}.analysis_frame":
            raise  # an import that fails inside the module is a fault to see, not a reason to check nothing
        return ""
    return _analysis.analysis_problem(old, new, name)


def body_signature(body: str) -> str:
    """``def derive_x(table)`` for the body ``derive_x`` (``reference_x(setting)``, ``figure_x(fig, table)``), else ``""``."""
    kind = str(body).split("_", 1)[0]
    return f"def {body}({BODY_PARAMS[kind]})" if kind in BODY_PARAMS and "_" in str(body) else ""


def has_analysis_frame(source: str) -> bool:
    """Whether ``source`` (the text of derive.py) has FI's frame."""
    return ANALYSIS_MARK in (source or "")


DERIVE_REPAIR_NOTE = (
    "FIRST, what you may change: FI wrote experiment.py, which runs the analysis, and the frame of derive.py (the lines "
    "between its `# >>> FI analysis frame` and `# <<< FI analysis frame` lines). Only the functions named `derive_...`, "
    "`reference_...` and `figure_...` of derive.py, the helper functions and the imports they use may change. Give each "
    "function you change whole, with its own parameters (in `functions`); do not give the whole file, and do not give "
    "experiment.py or touch fi_frame.py. A change to FI's lines is refused before anything runs and the repair counts as "
    "not made.\n\n")

REFLECT_INTRO_DERIVE = ("Produce a corrected function of derive.py (and the helper functions it uses) that runs to completion on "
                        "the available CPU venv and returns what the plan declares through FI's analysis. The fix must address "
                        "the actual bug shown in the traceback.")

#: The one rule for a function that gives a number for each value the plan compares inside each run.
DERIVE_LEVEL_RULE = (
    "When the plan compares values inside each run, a function that gives one number for each compared value returns a mapping "
    "with EVERY compared value as a key, written the way `table.levels` shows it; where the result does not exist for a value "
    "its number is `None` (a left-out key is refused).")

_BODY_RETURNS = {
    "derive": ("It returns one number for the study, a mapping of setting keys to numbers, or `table.missing(\"why\")` where "
               "the number does not exist (never a stand-in such as 0, -1 or a bound, and never a bare `None`). It reads the "
               "results only through `table`."),
    "reference": ("It takes one setting (`{axis: value}`), is called once for each setting, and returns the reference value "
                  "there, or `None` where there is none."),
    "figure": ("It draws on `fig.axes` from `table` and returns nothing: FI makes the figure, saves it and closes it, so it never "
               "calls `savefig`, `show`, `close`, `figure` or `subplots`. It draws only values read from `table`."),
}


def reflect_derive(body: str = "", compared: bool = False) -> str:
    """The text of a repair when the function ``body`` of derive.py failed (the analysis is FI's: it calls the bodies, keeps
    their record and prints the results itself). It replaces the text of the repair of an experiment.py the model wrote.
    ``compared``: the plan compares values inside each run (:data:`DERIVE_LEVEL_RULE` is then added)."""
    sig = body_signature(body)
    kind = str(body).split("_", 1)[0]
    which = (f"The function `{body}` failed, or FI refused what it returned (the reason is below). Repair `{body}` (and the "
             f"helpers it uses) and give it whole with exactly its parameters, `{sig}`. " if sig else
             "A function of derive.py failed, or FI refused what it returned (the reason is below). Repair that function (and "
             "the helpers it uses), each given whole with its own parameters. ")
    return ("THE ANALYSIS IS FI'S: FI wrote experiment.py, which reads its own record of the trials, calls the functions of "
            "derive.py, keeps what they return and prints the results. " + which
            + _BODY_RETURNS.get(kind, "") + (" " + DERIVE_LEVEL_RULE if compared and kind == "derive" else "")
            + " Leave every other function and every line of FI's as they are, and give no experiment.py.\n")


#: The sentences of agents/execute_reflect.md that are about a script that prints its results, and what they say for a repair
#: of a function of derive.py (where FI prints the results and a value that does not exist is `table.missing`).
_REFLECT_SWAPS = (
    ("produce a patched script that fixes it.", "produce a patched function of derive.py that fixes it."),
    (REFLECT_INTRO, REFLECT_INTRO_DERIVE),
    ("If the method genuinely diverges or a value is undefined, that is a finding: emit `null` for that value and say why under "
     "`\"why_missing\"` (`\"why_missing\": {\"<name>\": \"<a few words>\"}`); never a stand-in number.",
     "If the method genuinely diverges or a value is undefined, that is a finding: return `table.missing(\"<a few words why>\")` "
     "for it; never a stand-in number."),
    ("REQUIRED (or `code`) unless", "REQUIRED unless"),
    ("<leave empty. Only if the fix changes a module-level statement (an import, a constant) that no function can carry, give the "
     "whole corrected script here, with every function it has now>", "<leave empty: give the functions only>"),
    ("- RESULT_JSON parsed: $result_json_present\n", ""),
    ("- figures produced: $figures_count\n", ""),
)


def derive_prompt(source: str, prompt: str) -> str:
    """The repair request ``prompt`` (made from agents/execute_reflect.md) for a function of a derive.py with text ``source``:
    the note on what may change first, then the text with every sentence about a whole script that prints its results (the
    intro, the rule for a value that does not exist, the whole-script ``code`` key, the lines about ``RESULT_JSON``) replaced
    by what is true for a function of derive.py; ``prompt`` unchanged when ``source`` has no FI frame."""
    if not has_analysis_frame(source):
        return prompt
    for old, new in _REFLECT_SWAPS:
        prompt = prompt.replace(old, new)
    return DERIVE_REPAIR_NOTE + prompt


RERUN_NOTE_ANALYSIS = (
    "\nThe experiment is a simulation and an analysis. In the `# file: simulate.py` block give `trial` and its helpers back "
    "exactly as they are unless the problem the review found is IN the simulation: the trials FI already ran are then used as "
    "they are, and only the analysis runs again. FI writes experiment.py; if the problem is in a number or a figure the "
    "analysis gives, give the function of derive.py that works it out, whole, in the `# file: derive.py` block.\n")

EXTEND_REPLY_ANALYSIS = ("Return what the reply format above asks for: in the `# file: simulate.py` block only `trial` and the "
                         "helpers it uses (FI keeps the frame), and in the `# file: derive.py` block only the functions that "
                         "work out the new numbers or figures (FI writes experiment.py).")


def compose(frame: str, simulate_reply: str, previous: str = "") -> tuple[str, str]:
    """What becomes ``simulate.py`` when the model was asked for ``trial`` only: ``(source, note)``. A reply that already holds
    the whole frame is accepted when :func:`frame_intact` says so; a reply that is a whole simulation of its own
    (``run_trial`` / ``run_cell`` and no ``trial``) is kept as it is, with a note: the plan then runs as before, without the frame.
    A reply that holds a ``trial`` that does not fit is returned unchanged with the reason (the checks of the code, and the
    harness, say what is wrong with it). ``previous``: the ``simulate.py`` the quest has now; a reply that keeps exactly the
    frame in it while the plan asks for another (a result was added, a setting renamed) is put into the frame for the new
    plan: a change of the plan writes the frame again, and a change the model made to it is still refused."""
    if has_frame(simulate_reply):
        problem = frame_intact(frame, simulate_reply)
        if problem and has_frame(previous) and RAISE_MARK not in previous and not frame_intact(previous, simulate_reply):
            made = heal(frame, simulate_reply)
            if has_frame(made) and not frame_intact(frame, made):
                return made, "the plan changed, so FI wrote the frame again; the model's `trial` and its helpers are kept in it"
        return (without_raise(simulate_reply) if not problem else simulate_reply), problem
    try:
        tree = ast.parse(simulate_reply)
    except SyntaxError:
        return simulate_reply, "the reply is not valid Python"
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if SLOT not in names and not batch_form(simulate_reply):
        if names & {"run_trial", "run_cell"}:
            return simulate_reply, "the reply is a whole simulation without the function `trial`: kept as written, without the frame"
        return simulate_reply, "the reply holds no function `trial`"
    text, why, left = _fill(frame, simulate_reply)
    if text is None:
        # The file keeps FI's frame (a later repair is checked against it); the run then stops with the reason.
        return refuse(heal(frame, simulate_reply), why), why
    return text, (("left out of the model's reply, which the file cannot carry: " + "; ".join(left)) if left else "")


# ---- a frame is written once ---------------------------------------------------------------------------------------------

def frame_version(source: str) -> int | None:
    """The version of FI's frame that wrote the frame of ``source`` (a simulate.py), or ``None`` when it has none."""
    m = _BEGINS[0].search(source or "")
    return int(m.group(1)) if m else None


def runtime_version(text: str) -> int:
    """The version a ``fi_frame.py`` says it is (the first runtime had no line and is version 1)."""
    m = re.search(r"^FRAME_RUNTIME_VERSION = (\d+)", text or "", re.MULTILINE)
    return int(m.group(1)) if m else 1


def _plan_of(source: str) -> tuple | None:
    """What the plan decided in a frame: its settings, its results, whether a name outside them is refused, and which entry
    point it has. The rest of the frame is FI's, and another version of FI may word it otherwise."""
    regions = _regions(source)
    if regions is None:
        return None
    found: dict[str, Any] = {}
    try:
        for node in ast.parse(regions[0]).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("SETTINGS", "RESULTS"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", "") == "declare":
                found["strict"] = next((ast.literal_eval(k.value) for k in node.value.keywords if k.arg == "strict"), False)
        entries = tuple(sorted(n.name for n in ast.parse(regions[1]).body if isinstance(n, ast.FunctionDef)))
    except (SyntaxError, ValueError):
        return None
    return found.get("SETTINGS"), found.get("RESULTS"), found.get("strict"), entries


def frame_for(design: dict[str, Any], *, stochastic: bool, strict: bool = False, existing: str = "") -> str:
    """The frame the model's ``trial`` is put into. A quest keeps the frame it was written with: when ``existing`` (the quest's
    simulate.py) has a frame that another version of FI wrote for the same plan, that frame is returned, so a resume or an
    update of FI never changes it. When the plan changed (a result was added, a setting renamed), or there is no frame yet,
    the frame is written new."""
    fresh = frame_source(design, stochastic=stochastic, strict=strict)
    old = frame_version(existing)
    if old is None or old == VERSION:
        return fresh
    kept = _plan_of(existing)
    if kept is None or kept != _plan_of(fresh):
        return fresh
    spans = _spans(existing)
    assert spans is not None
    return _assemble(existing[spans[0][0]:spans[0][2]], existing[spans[1][0]:spans[1][2]])


def ensure_runtime(code_dir: Path | str, simulate_source: str) -> bool:
    """Write ``fi_frame.py`` beside simulate.py when the frame needs it: it is missing, it is not FI's runtime, or it is an older
    runtime and simulate.py has the frame of this version. A quest whose frame was written by an older version keeps the runtime
    it was written with. Returns whether the file was written."""
    path = Path(code_dir) / RUNTIME_FILE
    try:
        have = path.read_text(encoding="utf-8")
    except OSError:
        have = None
    if have is not None:
        if have.lstrip("\ufeff") == RUNTIME:
            return False
        ours = have.lstrip("\ufeff").startswith(RUNTIME_HEAD)
        framed = frame_version(simulate_source)
        if ours and runtime_version(have) != VERSION and framed is not None and framed != VERSION:
            return False  # the runtime of an older frame stays with it
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RUNTIME, encoding="utf-8")
    return True


# ---- what the small run shows ----------------------------------------------------------------------------------------

def empty_slots(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The declared results no trial of the run returned, under any of their forms (``returned``: the names the entry gave)."""
    got = {str(n) for n in returned or []}
    return [i for i in declared_ids(protocol) if not any(f in got for f in _metric_spec.result_forms(i))]


def unknown_names(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The names the entry returned that the plan does not declare under any form (diagnostics: kept, never reported)."""
    known = set(declared_names(protocol))
    return sorted({str(n) for n in returned or []} - known)
