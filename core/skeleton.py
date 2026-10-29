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
    The model's function must take exactly ``(cell, ctx)``. Goes through :func:`core.function_steps.replace_function`, the
    path that fills the functions of the model's package. The result is checked with :func:`frame_intact`."""
    text, why = _steps.replace_function(frame, SLOT, reply, signature=SLOT_SIGNATURE, equation="trial")
    if text is None:
        return None, why
    problem = frame_intact(frame, text)
    return (None, problem) if problem else (text, "")


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


def frame_intact(old: str, new: str) -> str:
    """``""`` when ``new`` has exactly the frame ``old`` has (both regions, compared as syntax, so a change of spacing does not
    count) and the slot ``trial`` with the same parameters; else a plain sentence naming what changed."""
    before, after = _regions(old), _regions(new)
    if before is None:
        return ""
    if after is None:
        return "the frame FI wrote (the marked parts of simulate.py) is missing"
    try:
        for a, b in zip(before, after):
            if ast.dump(ast.parse(a)) != ast.dump(ast.parse(b)):
                return "the frame FI wrote (the marked parts of simulate.py) was changed"
    except SyntaxError as e:
        return f"the frame is not valid Python ({e.msg}, line {e.lineno})"
    slot_old, slot_new = _steps._find(old, SLOT), _steps._find(new, SLOT)
    if slot_new is None:
        return f"the function `{SLOT}` is missing"
    if slot_old is not None and ast.dump(slot_old.args) != ast.dump(slot_new.args):
        return f"the parameters of `{SLOT}` were changed; they must be (cell, ctx)"
    return ""


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
        return simulate_reply, problem
    try:
        tree = ast.parse(simulate_reply)
    except SyntaxError:
        return simulate_reply, "the reply is not valid Python"
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if SLOT not in names:
        if names & {"run_trial", "run_cell"}:
            return simulate_reply, "the reply is a whole simulation without the function `trial`: kept as written, without the frame"
        return simulate_reply, "the reply holds no function `trial`"
    text, why = fill(frame, simulate_reply)
    return (text, "") if text is not None else (simulate_reply, why)


# ---- what the small run shows ----------------------------------------------------------------------------------------

def empty_slots(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The declared results no trial of the run returned, under any of their forms (``returned``: the names the entry gave)."""
    got = {str(n) for n in returned or []}
    return [i for i in declared_ids(protocol) if not any(f in got for f in _metric_spec.result_forms(i))]


def unknown_names(protocol: dict[str, Any] | None, returned: Any) -> list[str]:
    """The names the entry returned that the plan does not declare under any form (diagnostics: kept, never reported)."""
    known = set(declared_names(protocol))
    return sorted({str(n) for n in returned or []} - known)
