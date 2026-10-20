"""What FI measured about how the run went, kept in one record: ``.fi/run_facts.json``.

Two facts the paper states about the run were wrong when they were read from the wrong place:

* **How long the experiment ran.** ``state["exec_result"]["duration_s"]`` is the time of the LAST execution, and the last
  execution is often only the analysis run again on the trials FI already stored (2.3 s for an experiment that took 223 s).
  The time that counts is the wall time of the run that produced the stored trials. :func:`record_trials` keeps it
  where the trials are run (``core/trial_runner.py``), and an analysis run again on stored trials does not touch it.
  The sum of the per-trial times in the ledger is kept beside it as a cross-check; it is smaller than the wall time (every
  setting runs in its own process) and is not what the paper is checked against. In a quest run in two stages
  (``core/phased.py``) the record says which stage the run belongs to, and keeps the other stage's time beside it, so the
  paper can say which run its time is: the stored trials, whose results are reported, are the last run's.
* **Which seed the code used.** FI gives each trial a seed derived from the study's base seed, but a simulation that never
  passes the ``seed`` its ``run_trial`` is given to a random generator (it builds its generators from a number typed into
  the code, it only copies the seed into its result, it overwrites it, or it draws from a generator made without a seed or
  when the module is loaded) is not run with that seed. :func:`seed_verdict` reads the code (never runs it) and says
  ``reaches`` / ``does_not_reach`` / ``unknown``, with the seeds written into the code that drive the trials (not every
  seed in the code: a self-check's own seed is not the run's). It reuses FI's existing seed checks: a generator built
  without a seed (``engine.unseeded_rng_calls``, minus the ``rng or default_rng()`` fallback of
  ``code_layout.fallback_rng_lines``) and one made when a module is loaded (``code_layout.module_level_randomness``), each
  only where the trial's code can reach it. The paper's statement of the seed is held to it (``core/stat_claims.py``), and
  the paper's ``{{seed}}`` is filled from it (``core/paper_facts.py``).

Nothing here asks a model.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PATH = Path(".fi") / "run_facts.json"

#: Calls that build or seed a random generator whatever they are called on; their first argument is a seed.
_SEEDERS = frozenset({"default_rng", "RandomState", "manual_seed", "manual_seed_all", "PRNGKey", "SeedSequence",
                      "PCG64", "PCG64DXSM", "Philox", "MT19937", "SFC64", "set_seed", "seed_everything"})
#: Names that seed a generator only when called on a random module or generator (``np.random.seed``, ``random.seed``,
#: ``random.Random``, ``jax.random.split``), not on anything else (``str.split``, ``re.split``).
_RANDOM_ONLY_SEEDERS = frozenset({"seed", "Random", "Generator", "split", "fold_in", "key"})
_RANDOM_RECEIVER = re.compile(r"(?:^|\.)(?:random|jrandom|jr|rng|torch|cuda|tf)$", re.I)


def _receiver(call: ast.Call) -> str:
    """The dotted name ``call`` is made on (``np.random`` for ``np.random.seed(...)``), ``""`` for a plain name."""
    f = call.func
    if not isinstance(f, ast.Attribute):
        return ""
    try:
        return ast.unparse(f.value)
    except Exception:  # noqa: BLE001
        return "?"


def _is_seeder(call: ast.Call) -> bool:
    """``call`` builds or seeds a random generator: a constructor of one, or a seed call on a random module."""
    name = _call_name(call)
    if name in _SEEDERS:
        return True
    if name in _RANDOM_ONLY_SEEDERS:
        receiver = _receiver(call)
        return bool(_RANDOM_RECEIVER.search(receiver)) if receiver else name in ("seed", "Random")
    return False


def _without_zeroed(expr: ast.AST) -> ast.AST:
    """``expr`` with every part multiplied by a literal 0 left out: ``seed * 0 + 5`` carries no seed."""

    class _Zero(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            self.generic_visit(node)
            if isinstance(node.op, ast.Mult) and any(isinstance(s, ast.Constant) and s.value == 0
                                                     for s in (node.left, node.right)):
                return ast.Constant(value=0)
            return node

    import copy

    return _Zero().visit(copy.deepcopy(expr))
_SKIP_PARTS = {"__pycache__", "site-packages", "tests", "test"}
#: A name that holds a seed (``seed``, ``BASE_SEED``, ``random_state``), not one that merely contains the word.
_SEED_NAME = re.compile(r"^(?:(?:base|random|rng|global|master|root)_)?seed$|^random_state$", re.I)
#: The parameter of ``run_trial`` that is the seed, by name; else the third positional one.
_SEED_PARAMS = ("seed", "rng_seed", "random_seed", "trial_seed")
#: Calls that only read or reshape a value: a seed given to them does not reach a generator, and does not leave FI unsure.
_NEUTRAL = frozenset({
    "int", "float", "str", "abs", "hash", "len", "print", "repr", "format", "dict", "list", "tuple", "set", "frozenset",
    "max", "min", "sum", "round", "isinstance", "bool", "range", "sorted", "enumerate", "zip", "divmod", "pow", "bin", "hex",
    "dumps", "info", "debug", "warning", "error", "log", "write", "append", "update", "get", "setdefault", "items", "join",
    "uint32", "uint64", "int64", "int32", "array", "asarray", "from_bytes", "to_bytes", "encode", "hexdigest", "sha256",
    "md5", "blake2b", "mod", "floor", "ceil", "exception", "isfinite", "isnan",
    # string methods (``str(cell).split(str(seed))`` reshapes text, it seeds nothing)
    "split", "rsplit", "strip", "lstrip", "rstrip", "replace", "lower", "upper", "startswith", "endswith", "find",
    "index", "count", "zfill", "partition"})

REACHES, DOES_NOT_REACH, UNKNOWN, NO_ENTRY = "reaches", "does_not_reach", "unknown", "no_entry"
#: FI's seed seeds a generator, and a number typed into the code seeds one the trial also draws from.
PARTLY = "partly"


def read(quest_root: Path | str) -> dict[str, Any]:
    """The record, or ``{}`` when there is none or it cannot be read."""
    try:
        data = json.loads((Path(quest_root) / PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(quest_root: Path | str, **changes: Any) -> None:
    path = Path(quest_root) / PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**read(quest_root), **changes}, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError:
        pass  # a record FI cannot keep leaves the paper without the fact, never stops the quest


def record_trials(quest_root: Path | str, *, key: str, wall_s: float | None, trials: int, cells: int,
                  trial_sum_s: float | None, stage: str | None = None) -> None:
    """Keep the wall time of a run of the trials (``wall_s``: seconds from before the first setting to after the last;
    ``None`` when it is not known, as for a cluster job), how many trials and settings it ran, the sum of the per-trial
    times in the ledger, and the stage of a quest run in two stages (``explore`` / ``confirm``; ``None`` otherwise).
    ``key`` is the run's own key (``trial_runner._run_key``). The run of each stage is kept under ``by_stage`` too, so the
    exploring run's time is not lost when the confirming run replaces it as the run whose trials are stored."""
    run = {"key": key, "wall_s": None if wall_s is None else round(float(wall_s), 2), "trials": int(trials),
           "cells": int(cells), "trial_sum_s": None if trial_sum_s is None else round(float(trial_sum_s), 2)}
    if stage:
        run["stage"] = str(stage)
    changes: dict[str, Any] = {"trials_run": run}
    if stage:
        by_stage = read(quest_root).get("by_stage")
        changes["by_stage"] = {**(by_stage if isinstance(by_stage, dict) else {}), str(stage): run}
    _write(quest_root, **changes)


def run_seconds(quest_root: Path | str, fallback: Any = None) -> float | None:
    """The measured wall time of the run that produced the stored trials, in seconds. With no record of one (a study
    that does not run FI's trials: one script, run once) ``fallback``, the time of the last execution, if it is a number."""
    run = read(quest_root).get("trials_run")
    wall = run.get("wall_s") if isinstance(run, dict) else None
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and 0 <= wall < float("inf"):
        return float(wall)
    if isinstance(run, dict):
        return None  # FI ran trials and does not know how long they took (a cluster job): the last execution is not it
    if isinstance(fallback, (int, float)) and not isinstance(fallback, bool) and 0 <= fallback < float("inf"):
        return float(fallback)
    return None


def run_stage(quest_root: Path | str) -> tuple[str | None, dict[str, Any]]:
    """``(the stage of the run whose trials are stored, the other stages' runs)``: ``(None, {})`` outside a quest run in
    two stages."""
    data = read(quest_root)
    run = data.get("trials_run") if isinstance(data.get("trials_run"), dict) else {}
    stage = run.get("stage") if isinstance(run.get("stage"), str) else None
    by_stage = data.get("by_stage") if isinstance(data.get("by_stage"), dict) else {}
    return stage, {k: v for k, v in by_stage.items() if k != stage and isinstance(v, dict)}


# ---- the seed ------------------------------------------------------------------------------------------------------


def _code_files(code_dir: Path) -> list[Path]:
    if not code_dir.is_dir():
        return []
    out = []
    for path in sorted(code_dir.rglob("*.py")):
        parts = path.relative_to(code_dir).parts
        # code/skills/: FI's copy of the selected skills' files (Engine._skill_files_to_code), not the study's code.
        if any(p.startswith(".") or p in _SKIP_PARTS or p.startswith("test_") for p in parts) or parts[:1] == ("skills",):
            continue
        out.append(path)
    return out


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _int_of(node: ast.AST) -> int | None:
    """``node`` as an int when it is one written as a number (``7``, ``-7``, ``seed * 0 + 5`` once the part multiplied by 0
    is left out), else ``None``."""
    value = _fold(node)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _fold(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _fold(node.operand)
        return None if inner is None else -inner
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult) and any(isinstance(s, ast.Constant) and s.value == 0
                                                 for s in (node.left, node.right)):
            return 0
        left, right = _fold(node.left), _fold(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def _literal_seeds(tree: ast.AST, rel: str) -> list[tuple[str, int]]:
    """``("code/x.py:107", 20261015)`` for each seed written as a number in ``tree``: the first argument of a call that
    seeds a generator (or the first item of a list given to ``SeedSequence``), a third argument of a call to
    ``run_trial`` and a name ``..._SEED = N``."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        where = f"{rel}:{line}"
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if _is_seeder(node) and node.args:
                first = node.args[0]
                if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                    first = first.elts[0]
                if (n := _int_of(first)) is not None:
                    found.append((where, n))
            elif name == "run_trial" and len(node.args) >= 3 and (n := _int_of(node.args[2])) is not None:
                found.append((where, n))
            for kw in node.keywords:
                if kw.arg and _SEED_NAME.match(kw.arg) and (n := _int_of(kw.value)) is not None:
                    found.append((where, n))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _SEED_NAME.match(target.id) and (n := _int_of(node.value)) is not None:
                    found.append((where, n))
    return found


def _seeder_literals(fn: ast.AST, rel: str) -> list[tuple[str, int]]:
    """The seeds written as numbers in ``fn``'s own body (not in a function it calls): what seeds the generators it makes."""
    found: list[tuple[str, int]] = []
    for node in _own_nodes(fn):
        if isinstance(node, ast.Call):
            if _is_seeder(node) and node.args:
                first = node.args[0]
                if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                    first = first.elts[0]
                if (n := _int_of(first)) is not None:
                    found.append((f"{rel}:{node.lineno}", n))
            for kw in node.keywords:
                if kw.arg and _SEED_NAME.match(kw.arg) and (n := _int_of(kw.value)) is not None:
                    found.append((f"{rel}:{node.lineno}", n))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _SEED_NAME.match(target.id) and (n := _int_of(node.value)) is not None:
                    found.append((f"{rel}:{node.lineno}", n))
    return found


def _own_nodes(fn: ast.AST) -> list[ast.AST]:
    """The nodes of ``fn``'s body, not those of a function or class defined inside it (a lambda's are kept)."""
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        out.append(node)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(node))
    return out


def _reads(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names and isinstance(n.ctx, ast.Load) for n in ast.walk(node))


def _pos(node: ast.AST, end: bool = False) -> tuple[int, int]:
    if end:
        return (getattr(node, "end_lineno", None) or node.lineno, getattr(node, "end_col_offset", None) or 0)
    return (node.lineno, node.col_offset)


@dataclass
class _Module:
    rel: str           # code/x/y.py
    name: str          # x.y
    source: str
    tree: ast.Module
    defs: dict[str, ast.AST] = field(default_factory=dict)       # top-level functions and classes
    bound: dict[str, ast.AST] = field(default_factory=dict)      # top-level name = expression
    imported: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (module, attribute or "")


@dataclass
class _Flow:
    """What following a seed through one function found."""

    reaches: bool = False        # the seed is given to a random generator
    unsure: list[str] = field(default_factory=list)  # where it went that FI cannot follow
    used: bool = False           # the seed is read at all
    overwritten: bool = False    # the seed's name was given another value before every use
    chain: list[tuple[str, ast.AST]] = field(default_factory=list)  # (module, function) the seed was passed into
    seeded: list[str] = field(default_factory=list)  # the seeding calls the seed reached (``np.random.seed``, ...)


class _Code:
    """The files of ``code/`` read once (never run)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.code_dir = root / "code"
        self.modules: dict[str, _Module] = {}
        for path in _code_files(self.code_dir):
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
            except (SyntaxError, ValueError, OSError):
                continue
            parts = list(path.relative_to(self.code_dir).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if not parts:
                continue
            mod = _Module(path.relative_to(root).as_posix(), ".".join(parts), source, tree)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    mod.defs[node.name] = node
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    mod.bound[node.targets[0].id] = node.value
                elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
                    for a in node.names:
                        mod.imported[a.asname or a.name] = (node.module, a.name)
                elif isinstance(node, ast.ImportFrom) and node.level > 0:
                    base = ".".join(parts[:-node.level]) if len(parts) > node.level else ""
                    full = ".".join(p for p in (base, node.module or "") if p)
                    for a in node.names:
                        mod.imported[a.asname or a.name] = (full, a.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        mod.imported[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], "")
            self.modules[mod.name] = mod

    def module(self, name: str) -> _Module | None:
        """The module of ``code/`` called ``name`` (``pkg.model``, or ``model`` for a file of the package run from code/)."""
        if name in self.modules:
            return self.modules[name]
        tail = [m for key, m in self.modules.items() if key.endswith("." + name)]
        return tail[0] if len(tail) == 1 else None

    def resolve(self, mod: _Module, expr: ast.AST, depth: int = 0) -> tuple[str, Any]:
        """What ``expr`` in ``mod`` names: ``("def", (module, node))`` for a function or class of ``code/``,
        ``("module", module)``, ``("outside", dotted)`` for a name from outside ``code/``, ``("?", "")`` otherwise."""
        if depth > 8:
            return "?", ""
        if isinstance(expr, ast.Name):
            name = expr.id
            if name in mod.defs:
                return "def", (mod, mod.defs[name])
            if name in mod.bound:
                return self.resolve(mod, mod.bound[name], depth + 1)
            if name in mod.imported:
                target, attr = mod.imported[name]
                other = self.module(target)
                if attr:
                    sub = self.module(f"{target}.{attr}")
                    if sub is not None:
                        return "module", sub
                    if other is not None:
                        return self.resolve(other, ast.Name(id=attr, ctx=ast.Load()), depth + 1)
                    return "outside", f"{target}.{attr}"
                return ("module", other) if other is not None else ("outside", target)
            return "?", name
        if isinstance(expr, ast.Attribute):
            kind, value = self.resolve(mod, expr.value, depth + 1)
            if kind == "module":
                return self.resolve(value, ast.Name(id=expr.attr, ctx=ast.Load()), depth + 1)
            if kind == "outside":
                return "outside", f"{value}.{expr.attr}"
            return "?", expr.attr
        return "?", ""


def _params(fn: ast.AST) -> tuple[list[str], list[str]]:
    a = fn.args  # type: ignore[attr-defined]
    return [x.arg for x in [*a.posonlyargs, *a.args]], [x.arg for x in a.kwonlyargs]


def _seed_param(fn: ast.AST) -> str:
    positional, keyword = _params(fn)
    for name in _SEED_PARAMS:
        if name in positional or name in keyword:
            return name
    return positional[2] if len(positional) >= 3 else ""


def _argument_for(fn: ast.AST, call: ast.Call, tainted: set[str], *, skip_self: bool = False) -> list[str]:
    """The parameters of ``fn`` that ``call`` gives a tainted value to (positional, by keyword, or in ``**{...}``)."""
    positional, keyword = _params(fn)
    if skip_self and positional:
        positional = positional[1:]
    out: list[str] = []
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Starred):
            if _reads(arg, tainted):
                out.append("*")
            continue
        if i < len(positional) and _reads(arg, tainted):
            out.append(positional[i])
    for kw in call.keywords:
        if kw.arg is None:
            if isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and isinstance(k.value, str) and _reads(v, tainted):
                        out.append(k.value)
            elif _reads(kw.value, tainted):
                out.append("*")
        elif _reads(kw.value, tainted) and (kw.arg in positional or kw.arg in keyword):
            out.append(kw.arg)
        elif _reads(kw.value, tainted):
            out.append("*")
    return out


class _Follower:
    """Follows the seed FI gives ``run_trial`` through the code, statement by statement in the order they are written
    (branches and loops are read as if every line ran once: a reading, not an execution)."""

    def __init__(self, code: _Code) -> None:
        self.code = code
        self.seen: dict[tuple[str, int, str], _Flow] = {}

    def follow(self, mod: _Module, fn: ast.AST, param: str, depth: int = 0) -> _Flow:
        key = (mod.name, getattr(fn, "lineno", 0), param)
        if key in self.seen:
            return self.seen[key]
        flow = _Flow(chain=[(mod.name, fn)])
        self.seen[key] = flow
        if depth > 6 or not param or param == "*":
            flow.unsure.append(f"{mod.rel}:{getattr(fn, 'lineno', 0)}")
            return flow
        tainted = {param}
        events: list[tuple[tuple[int, int], int, ast.AST]] = []
        for node in _own_nodes(fn):
            if isinstance(node, ast.Call):
                events.append((_pos(node), 0, node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                events.append((_pos(node, end=True), 1, node))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                events.append((_pos(node), 1, node))
            elif isinstance(node, ast.Delete):
                events.append((_pos(node), 1, node))
        events.sort(key=lambda e: (e[0], e[1]))
        ever_read = any(isinstance(n, ast.Name) and n.id == param and isinstance(n.ctx, ast.Load) for n in _own_nodes(fn))
        flow.used = ever_read
        for _p, kind, node in events:
            if not tainted:
                break
            if kind == 0:
                assert isinstance(node, ast.Call)
                if any(_reads(a, tainted) for a in [*node.args, *(k.value for k in node.keywords)]):
                    self._call(mod, node, tainted, flow, depth)
            else:
                self._assign(node, tainted, flow)
        if ever_read and not flow.reaches and not flow.unsure and param not in tainted:
            # The seed's name was given another value (or deleted) before anything that could use it read it.
            flow.overwritten = True
        return flow

    def _assign(self, node: ast.AST, tainted: set[str], flow: _Flow) -> None:
        if isinstance(node, ast.Delete):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    tainted.discard(t.id)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            if _reads(node.iter, tainted):
                tainted.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return
        value = getattr(node, "value", None)
        if isinstance(node, ast.AugAssign):
            return  # x += seed keeps x tainted if it was; a seed added to a tainted name is still the seed
        targets: list[ast.AST] = (node.targets if isinstance(node, ast.Assign)
                                  else [node.target])  # type: ignore[attr-defined]
        carries = value is not None and _reads(value, tainted)
        for target in targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name):
                    (tainted.add if carries else tainted.discard)(n.id)
                elif isinstance(n, ast.Attribute) and carries and isinstance(target, ast.Attribute):
                    # Kept on an object (``self.seed = seed``): FI does not follow attributes.
                    flow.unsure.append(f"{ast.unparse(target)} (line {node.lineno})")

    def _call(self, mod: _Module, call: ast.Call, tainted: set[str], flow: _Flow, depth: int) -> None:
        name = _call_name(call)
        if _is_seeder(call):
            # The seed reaches the generator only through what is not multiplied away (``seed * 0 + 5`` seeds it with 5).
            if any(_reads(_without_zeroed(a), tainted) for a in [*call.args, *(k.value for k in call.keywords)]):
                flow.reaches = True
                flow.seeded.append(_receiver(call) + "." + name if _receiver(call) else name)
            return
        if name == "partial" and call.args:
            kind, target = self.code.resolve(mod, call.args[0])
            if kind == "def":
                rest = ast.Call(func=call.args[0], args=call.args[1:], keywords=call.keywords)
                self._into(target, rest, tainted, flow, depth)
                return
        kind, target = self.code.resolve(mod, call.func)
        if kind == "def":
            self._into(target, call, tainted, flow, depth)
            return
        if name in _NEUTRAL or (kind == "?" and name in ("", "format")):
            return
        flow.unsure.append(f"{ast.unparse(call.func)} (line {call.lineno})")

    def _into(self, target: tuple[_Module, ast.AST], call: ast.Call, tainted: set[str], flow: _Flow, depth: int) -> None:
        tmod, tnode = target
        if isinstance(tnode, ast.ClassDef):
            init = next((n for n in tnode.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == "__init__"), None)
            if init is None:
                flow.unsure.append(f"{tmod.rel}:{tnode.lineno}")
                return
            params = _argument_for(init, call, tainted, skip_self=True)
            fn: ast.AST = init
        else:
            params = _argument_for(tnode, call, tainted)
            fn = tnode
        for p in params or ["*"]:
            inner = self.follow(tmod, fn, p, depth + 1)
            flow.reaches = flow.reaches or inner.reaches
            flow.unsure += inner.unsure
            flow.seeded += [s for s in inner.seeded if s not in flow.seeded]
            # The seed was handed on: the function it went to is where the trial's generators are made, whether it uses
            # the seed or not (a wrapper's run_trial that hands it to the model's, which ignores it).
            flow.chain += [c for c in inner.chain if c not in flow.chain]


def _reachable(code: _Code, mod: _Module, fn: ast.AST) -> list[tuple[_Module, ast.AST]]:
    """Every function and class of ``code/`` that ``fn`` calls or names, directly or through others (read, not run)."""
    out: list[tuple[_Module, ast.AST]] = []
    seen: set[tuple[str, int]] = set()
    todo = [(mod, fn)]
    while todo:
        m, f = todo.pop()
        key = (m.name, getattr(f, "lineno", 0))
        if key in seen:
            continue
        seen.add(key)
        out.append((m, f))
        for node in _own_nodes(f) if not isinstance(f, ast.ClassDef) else list(ast.walk(f)):
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(getattr(node, "ctx", None), ast.Load):
                kind, target = code.resolve(m, node)
                if kind == "def":
                    todo.append(target)
    return out


def _entry(code: _Code) -> tuple[_Module, ast.AST] | str:
    """``run_trial`` as FI calls it (``code/simulate.py``'s, followed through an import or a name bound to it), or why
    there is none to read: ``"no_entry"`` (no simulate.py, or no run_trial in it), ``"outside"`` (imported from outside
    ``code/``)."""
    simulate = code.modules.get("simulate")
    if simulate is None:
        return "no_entry"
    if "run_trial" not in simulate.defs and "run_trial" not in simulate.bound and "run_trial" not in simulate.imported:
        return "no_entry"
    kind, target = code.resolve(simulate, ast.Name(id="run_trial", ctx=ast.Load()))
    if kind == "def" and isinstance(target[1], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return target
    if isinstance(simulate.bound.get("run_trial"), ast.Lambda):
        return simulate, simulate.bound["run_trial"]
    return "outside"


def _unseeded_where(code: _Code, reach: list[tuple[_Module, ast.AST]]) -> list[str]:
    """``code/x.py:52 (np.random.default_rng())`` for each generator the trial's code builds without a seed, outside the
    ``rng or default_rng()`` fallback (FI's existing checks, ``engine.unseeded_rng_calls`` and
    ``code_layout.fallback_rng_lines``)."""
    from .code_layout import fallback_rng_lines
    from .engine import unseeded_rng_calls

    out: list[str] = []
    by_module: dict[str, list[ast.AST]] = {}
    mods: dict[str, _Module] = {}
    for m, f in reach:
        by_module.setdefault(m.name, []).append(f)
        mods[m.name] = m
    for name, fns in by_module.items():
        m = mods[name]
        fallback = fallback_rng_lines(m.source)
        spans = [range(f.lineno, (getattr(f, "end_lineno", None) or f.lineno) + 1) for f in fns]
        for line, expr in unseeded_rng_calls(m.source):
            if line not in fallback and any(line in s for s in spans):
                out.append(f"{m.rel}:{line} ({expr})")
    return out


#: Calls on a random module that seed it, build a generator, or read its state: not draws.
_NOT_DRAWS = frozenset({"seed", "default_rng", "RandomState", "SeedSequence", "Generator", "PCG64", "PCG64DXSM", "Philox",
                        "MT19937", "SFC64", "BitGenerator", "get_state", "set_state", "getstate", "setstate", "Random",
                        "SystemRandom", "get_bit_generator", "set_bit_generator"})


def _module_of(mod: _Module, expr: ast.AST) -> str:
    """The module outside ``code/`` that ``expr`` names (``numpy.random`` for ``np.random`` after ``import numpy as np``),
    ``""`` otherwise."""
    if isinstance(expr, ast.Name):
        target = mod.imported.get(expr.id)
        if not target:
            return ""
        return f"{target[0]}.{target[1]}" if target[1] else target[0]
    if isinstance(expr, ast.Attribute):
        head = _module_of(mod, expr.value)
        return f"{head}.{expr.attr}" if head else ""
    return ""


def _global_draws(reach: list[tuple[_Module, ast.AST]], seeded: list[str]) -> list[str]:
    """``code/x.py:7 (np.random.normal())`` for each draw the trial's code makes from numpy's global generator or from
    stdlib ``random`` that FI's seed does not seed (it reaches no ``np.random.seed`` / ``random.seed``), and each scipy
    ``.rvs(...)`` given no ``random_state``: random numbers drawn without the seed FI gives the trial."""
    numpy_seeded = any(re.search(r"(?:^|\.)(?:np|numpy)\.random\.seed$", s) for s in seeded)
    stdlib_seeded = any(s == "random.seed" for s in seeded)
    out: list[str] = []
    for m, f in reach:
        nodes = list(ast.walk(f)) if isinstance(f, ast.ClassDef) else _own_nodes(f)
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "rvs" and not any(k.arg == "random_state" for k in node.keywords):
                out.append(f"{m.rel}:{node.lineno} ({ast.unparse(node.func)}(...) with no random_state)")
                continue
            if name in _NOT_DRAWS:
                continue
            func = node.func
            where = (_module_of(m, func.value) if isinstance(func, ast.Attribute)
                     else ".".join(p for p in _imported_from(m, func)) if isinstance(func, ast.Name) else "")
            if (where == "numpy.random" and not numpy_seeded) or (where == "random" and not stdlib_seeded):
                out.append(f"{m.rel}:{node.lineno} ({ast.unparse(func)}())")
    return list(dict.fromkeys(out))


def _imported_from(mod: _Module, name: ast.Name) -> tuple[str, ...]:
    """The module a plain name was imported from (``("numpy.random",)`` after ``from numpy.random import normal``)."""
    target = mod.imported.get(name.id)
    return (target[0],) if target and target[1] else ()


def _module_generators(code: _Code, reach: list[tuple[_Module, ast.AST]]) -> list[tuple[str, str, int | None]]:
    """``(code/x.py:2, the name, its seed or None)`` for each generator made when a module is loaded
    (``code_layout.module_level_randomness``) that the trial's code reads by name."""
    from .code_layout import module_level_randomness

    out: list[tuple[str, str, int | None]] = []
    read_names: dict[str, set[str]] = {}
    for m, f in reach:
        names = read_names.setdefault(m.name, set())
        nodes = list(ast.walk(f)) if isinstance(f, ast.ClassDef) else _own_nodes(f)
        names.update(n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    for name, used in read_names.items():
        m = code.modules[name]
        lines = set(module_level_randomness(m.source))
        if not lines:
            continue
        for stmt in m.tree.body:
            if not isinstance(stmt, ast.Assign) or stmt.lineno not in lines:
                continue
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id in used:
                    seed = None
                    if isinstance(stmt.value, ast.Call) and stmt.value.args:
                        first = stmt.value.args[0]
                        if isinstance(first, (ast.List, ast.Tuple)) and first.elts:
                            first = first.elts[0]
                        seed = _int_of(first)
                    out.append((f"{m.rel}:{stmt.lineno}", t.id, seed))
    return out


def seed_verdict(quest_root: Path | str) -> dict[str, Any]:
    """What the code says about the seed, read and not run::

        {"verdict": "reaches" | "does_not_reach" | "unknown" | "no_entry",
         "fi_seed_reaches": True | False | None,     # the verdict as a yes / no / FI cannot tell
         "why": str,                                 # in plain words, when it does not reach or FI cannot tell
         "entry": "code/simulate.py:96",             # where run_trial is
         "driving_seeds": [{"where", "value"}],      # numbers written in the code that seed the trials' generators
         "unseeded": ["code/x.py:52 (...)"],         # generators the trial's code builds without a seed
         "code_seeds": [{"where", "value"}]}         # every seed written as a number anywhere in code/

    ``reaches``: the seed ``run_trial`` is given flows into a call that seeds a generator (directly or through functions of
    ``code/`` it is passed to), and the trial's code draws from no generator made without a seed or when a module is loaded.
    ``does_not_reach``: it is never read, only copied (into the result, a log), overwritten before use, or the trial's code
    also draws from a generator FI's seed does not reach. ``unknown``: it is handed to something FI cannot follow (a
    library, an object's attribute, ``**kwargs``), or ``run_trial`` comes from outside ``code/``. ``no_entry``: there is
    no ``run_trial`` (a study that is not run trial by trial)."""
    root = Path(quest_root)
    code = _Code(root)
    literals = [{"where": w, "value": v} for m in code.modules.values() for w, v in _literal_seeds(m.tree, m.rel)]
    out: dict[str, Any] = {"verdict": NO_ENTRY, "fi_seed_reaches": None, "why": "", "entry": "", "driving_seeds": [],
                           "unseeded": [], "code_seeds": literals}
    entry = _entry(code)
    if entry == "no_entry":
        return out
    if entry == "outside":
        out.update(verdict=UNKNOWN, why="run_trial comes from outside code/, so FI cannot read what it does with its seed")
        return out
    mod, fn = entry  # type: ignore[misc]
    out["entry"] = f"{mod.rel}:{getattr(fn, 'lineno', 0)}"
    param = _seed_param(fn)
    follower = _Follower(code)
    flow = follower.follow(mod, fn, param) if param else _Flow(unsure=["run_trial has no seed parameter FI can find"])
    reach = _reachable(code, mod, fn)
    unseeded = _unseeded_where(code, reach)
    module_rngs = _module_generators(code, reach)
    global_draws = _global_draws(reach, flow.seeded)
    out["unseeded"] = unseeded + global_draws
    driving: list[tuple[str, int]] = []
    for cmod, cfn in flow.chain:
        driving += _seeder_literals(cfn, code.modules[cmod].rel)
    driving += [(where, seed) for where, _name, seed in module_rngs if seed is not None]
    out["driving_seeds"] = [{"where": w, "value": v} for w, v in dict.fromkeys(driving)]
    why: list[str] = []
    handed = [f"`{n.name}` ({code.modules[m].rel}:{n.lineno})" for m, n in flow.chain[1:]
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if param and not flow.used:
        why.append(f"run_trial takes `{param}` but never uses it")
    elif param and flow.overwritten:
        why.append(f"run_trial gives `{param}` another value before anything uses it")
    elif param and not flow.reaches and not flow.unsure and handed:
        why.append(f"run_trial passes `{param}` on to {', '.join(handed[:3])}, and from there it never reaches a random "
                   "generator")
    elif param and not flow.reaches and not flow.unsure:
        why.append(f"run_trial reads `{param}` but never gives it to a random generator (it only copies it, for example "
                   "into its result)")
    if module_rngs:
        why.append("the trial's code draws from " + ", ".join(f"`{name}` ({where})" for where, name, _s in module_rngs[:3])
                   + ", a generator made when the module is loaded")
    if unseeded:
        why.append("the trial's code builds a random generator without a seed (" + "; ".join(unseeded[:3]) + ")")
    if global_draws:
        why.append("the trial's code draws random numbers that the seed FI gives it does not seed ("
                   + "; ".join(global_draws[:3]) + ")")
    fixed = [f"{v}, at {w}" for w, v in dict.fromkeys(driving)]
    if why:
        out.update(verdict=DOES_NOT_REACH, fi_seed_reaches=False, why="; ".join(why))
    elif flow.reaches and fixed:
        # FI's seed seeds a generator, and a number typed into the code seeds another on the trial's own path (or the
        # same one on another branch): the trials are not driven by FI's seed alone, so FI's base seed is not "the" seed.
        out.update(verdict=PARTLY, fi_seed_reaches=False,
                   why="FI's seed seeds a generator, but the trial's code also seeds one with a number written in the "
                       f"code ({'; '.join(fixed[:3])})")
    elif flow.reaches:
        # What else the seed is handed to does not matter once it seeds a generator: the other generators the trial
        # could draw from are the two checks above.
        out.update(verdict=REACHES, fi_seed_reaches=True)
    elif flow.unsure or not param:
        where = "; ".join(dict.fromkeys(flow.unsure)) if flow.unsure else "run_trial, which has no seed parameter FI can find"
        out.update(verdict=UNKNOWN, why=f"FI cannot follow the seed past {where}")
    else:
        out.update(verdict=UNKNOWN, why="FI could not tell from the code what run_trial does with its seed")
    return out


def record_seed(quest_root: Path | str) -> dict[str, Any]:
    """Read the code (:func:`seed_verdict`) and keep the verdict in the record; returns it."""
    verdict = seed_verdict(quest_root)
    _write(quest_root, seed=verdict)
    return verdict
