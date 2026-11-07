"""Which functions of the quest's code the run reaches, read from the code alone (no model, nothing run).

A second model reads the code against the plan and flags problems by file and function. A real paper published, as a
"plotting defect" of its Figure 3, a finding about ``simulate.py``'s ``main()``, which the run never called: the figure was
drawn by ``experiment.py``. A finding about code the run does not reach cannot change a reported number, so it is kept in
FI's record (``.fi/code_review.json``, QUALITY.md) and not put in the paper.

What runs: every script FI runs as a program (its top-level code and its ``if __name__ == "__main__":`` block), the
functions FI's trial harness calls (``run_trial``, ``run_cell``, ``oracle`` in either script: see
:mod:`core.trial_runner`), and every module's top-level code (it runs on import). From there a function is reached when
reached code names it: a call, or the name passed on as a value (a callback, a table of functions). A bare name is the
file's own function or the one it imports; ``module.f`` is that module's ``f``; ``obj.f`` on any other object may be any
function or method called ``f``. Every doubt counts as reached: a file that does not parse, a name computed at run time
(``getattr`` with a computed name, ``globals()`` handed on or indexed by a computed name, ``eval``, ``importlib``; ``globals().get("R")`` names one variable and is read as such) or a function FI cannot find makes every
function reached, so this can only ever spare a finding about code that is plainly never run.

Two computed ``getattr`` calls are read more closely than that. ``getattr(m, name)`` on an imported module ``m`` of the
quest's code may call any function of ``m``, so it reaches all of them (and nothing else). In a file FI wrote itself
(``experiment.py``, ``fi_table.py``, ``fi_figures.py``, ``fi_frame.py``, ``fi_figure_calls.py``, ``fi_core/``: see
:func:`_fis_files`) a computed ``getattr`` on any other object reaches only the functions of FI's own files, since FI's text
is known; so does one on ``table`` or ``fig``, the two objects FI hands every function of ``derive.py`` (:data:`FI_OBJECT_PARAMS`),
in a file the study's model wrote; on any other object there it still means FI cannot tell.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

#: The simulation's functions FI's trial harness calls by name (core/trial_runner.py).
HARNESS_ENTRIES = ("run_trial", "run_cell", "oracle")
#: Calls that reach code by a name computed at run time.
_DYNAMIC_CALLS = {"globals", "locals", "vars", "eval", "exec", "__import__", "import_module"}


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _scope_read(node: ast.AST) -> tuple[ast.Call, str] | None:
    """``(the scope call, the name)`` when ``node`` reads one variable of a scope by a fixed text: ``globals().get("R", 1.0)``,
    ``globals()["R"]``, ``vars()["R"]`` or ``locals().get("R")``. That names a variable, as plain code would; only a name
    computed at run time (or the scope handed on as a whole) is a name FI cannot read."""
    scope: ast.AST | None = None
    key: ast.AST | None = None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
        scope, key = node.func.value, node.args[0]
    elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        scope, key = node.value, node.slice
    if (isinstance(scope, ast.Call) and isinstance(scope.func, ast.Name) and scope.func.id in ("globals", "locals", "vars")
            and not scope.args and not scope.keywords and isinstance(key, ast.Constant) and isinstance(key.value, str)):
        return scope, key.value
    return None


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    sides = [node.test.left, *node.test.comparators]
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


def _definition_time(stmts: list[ast.stmt], *, main: bool) -> list[ast.AST]:
    """What runs when a module is imported (or run, with ``main``): its statements, a function's decorators and defaults
    but not its body, a class's bases, decorators and body."""
    out: list[ast.AST] = []
    for s in stmts:
        if _is_main_guard(s) and not main:
            out.append(s.test)  # type: ignore[attr-defined]
            out.extend(_definition_time(s.orelse, main=main))  # type: ignore[attr-defined]
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.extend(s.decorator_list)
            out.extend(d for d in [*s.args.defaults, *s.args.kw_defaults] if d is not None)
        elif isinstance(s, ast.ClassDef):
            out.extend([*s.decorator_list, *s.bases, *(k.value for k in s.keywords)])
            out.extend(_definition_time(s.body, main=True))
        else:
            out.append(s)
    return out


#: The parameters FI gives every function of ``derive.py`` it calls (``analysis_frame.SIGNATURES``): its own table of the trials and
#: its own figure. A computed ``getattr`` on one of them looks a name up on FI's object, whose text is FI's.
FI_OBJECT_PARAMS = frozenset({"table", "fig"})


def _fis_object(m: "_Module", node: ast.AST | None) -> bool:
    """Whether ``node`` is a name the module takes as a parameter that FI gives it (:data:`FI_OBJECT_PARAMS`)."""
    return isinstance(node, ast.Name) and node.id in m.fi_params


class _Module:
    def __init__(self, rel: str, tree: ast.Module) -> None:
        self.rel = rel
        self.tree = tree
        self.fi_params = {a.arg for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                          for a in [*n.args.args, *n.args.kwonlyargs] if a.arg in FI_OBJECT_PARAMS}
        self.functions: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions.setdefault(node.name, []).append(node)
        self.imports: dict[str, tuple[str, str | None]] = {}  # local name -> (module, name in it or None for the module)
        self.star: list[str] = []
        package = rel[:-3].replace("/", ".").split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    up = package[:len(package) - (node.level - 1)] if node.level - 1 <= len(package) else []
                    base = ".".join([*up, *([base] if base else [])])
                for a in node.names:
                    if a.name == "*":
                        self.star.append(base)
                    else:
                        self.imports[a.asname or a.name] = (base, a.name)


#: Files FI writes beside the study's code whose text starts with a fixed line: a file of that name is FI's only when it does.
def _fis_heads() -> dict[str, str]:
    from . import skeleton as _skeleton

    return {_skeleton.RUNTIME_FILE: _skeleton.RUNTIME_HEAD,
            _skeleton.TABLE_FILE: _skeleton.TABLE_SOURCE[:60],
            _skeleton.FIGURE_CALLS_FILE: _skeleton.FRAME_FIGURE_SOURCE[:60]}


def _fis_files(code_dir: Path, rels: Any) -> set[str]:
    """The files among ``rels`` (paths relative to ``code_dir``) that hold FI's own text: the analysis FI writes
    (``experiment.py``), its figure step and formula calculator (``fi_figures.py``, ``fi_core/``), the frame's runtime, the
    reader of the record and the figure calls. A module of the study's with one of those names is the study's."""
    from . import analysis_frame as _analysis

    heads = _fis_heads()
    out: set[str] = set()
    for rel in rels:
        try:
            if rel == _analysis.EXPERIMENT_FILE:
                ok = _analysis.is_fis_file(code_dir / rel)
            elif rel in heads:
                ok = (code_dir / rel).read_text(encoding="utf-8").lstrip(chr(0xFEFF)).startswith(heads[rel])
            else:
                ok = _analysis.carried_file(code_dir, rel)
        except (OSError, UnicodeDecodeError, ValueError):
            ok = False
        if ok:
            out.add(rel)
    return out


def _python_files(code_dir: Path) -> list[Path]:
    # code/skills/ is FI's copy of the selected skills' files (Engine._skill_files_to_code), not the study's code.
    return sorted(p for p in code_dir.rglob("*.py")
                  if not any(part in ("tests", "__pycache__") or part.startswith(".") for part in p.relative_to(code_dir).parts[:-1])
                  and p.relative_to(code_dir).parts[0] != "skills" and not p.name.startswith("test_"))


def reached(code_dir: Path, *, scripts: list[str], harness: str | None = None,
            start: list[tuple[str, str]] | None = None) -> set[tuple[str, str]] | None:
    """The ``(file, function)`` pairs the run reaches (``file`` relative to ``code_dir``, with ``/``), from the programs
    ``scripts`` FI runs and, when FI's trial harness ran the simulation, the harness's entries in the file ``harness``.
    ``start``: also reach these ``(file, function)`` entries (one result's function, to ask what only that function
    reaches: give no ``scripts`` and no ``harness``; every module's top-level code and special methods still count).
    ``None`` when FI cannot tell (see the module docstring)."""
    code_dir = Path(code_dir)
    modules: dict[str, _Module] = {}
    for path in _python_files(code_dir):
        rel = path.relative_to(code_dir).as_posix()
        try:
            modules[rel] = _Module(rel, ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
            return None
    by_name: dict[str, _Module] = {}
    for rel, m in modules.items():
        dotted = rel[:-3].replace("/", ".")
        for name in {dotted, dotted.removesuffix(".__init__"), dotted.split(".")[-1]}:
            by_name.setdefault(name, m)

    fis = _fis_files(code_dir, modules)
    done: set[tuple[str, str]] = set()
    work: list[tuple[_Module, list[ast.AST]]] = []

    def reach(m: _Module | None, name: str, hops: int = 0) -> bool:
        """Mark what ``name`` in module ``m`` is: its own function, or what it imports under that name (a re-export in a
        package's ``__init__.py`` is followed to the module that defines it). ``False`` when ``m`` holds no such name."""
        if m is None or hops > 20:
            return False
        if name in m.functions:
            if (m.rel, name) not in done:
                done.add((m.rel, name))
                work.append((m, m.functions[name]))
            return True
        if name in m.imports:
            module, attr = m.imports[name]
            if attr is None or f"{module}.{attr}" in by_name:
                return True  # a module, not a function
            return reach(by_name.get(module), attr, hops + 1)
        return any(reach(by_name.get(module), name, hops + 1) for module in m.star)

    def everywhere(name: str) -> None:
        for m in modules.values():
            reach(m, name)

    def all_of(m: _Module, seen: set[str] | None = None) -> None:
        """Every function ``m`` holds, the names it imports (a package's re-exports) and, through ``from x import *``, what
        the modules it star-imports hold."""
        seen = set() if seen is None else seen
        if m.rel in seen:
            return
        seen.add(m.rel)
        for name in [*m.functions, *m.imports]:
            reach(m, name)
        for module in m.star:
            if by_name.get(module) is not None:
                all_of(by_name[module], seen)

    def imported_module(m: _Module, node: ast.AST) -> _Module | None:
        """The module of the quest's code that ``node`` names in ``m`` (``import derive``), else ``None``."""
        if not (isinstance(node, ast.Name) and node.id in m.imports and node.id not in m.functions):
            return None
        module, attr = m.imports[node.id]
        return by_name.get(module if attr is None else f"{module}.{attr}" if module else attr)

    for rel, m in modules.items():
        work.append((m, _definition_time(m.tree.body, main=rel in scripts)))
        for name in list(m.functions):
            if _is_dunder(name):
                reach(m, name)  # a special method runs without its name being written (Sim(3), len(s), s + s, with R():)
    # FI's harness may call these in either script, whether or not a ledger shows it did: they always count as run.
    for script in {*scripts, *([harness] if harness else [])}:
        if script in modules:
            for entry in HARNESS_ENTRIES:
                reach(modules[script], entry)  # defined there, or imported into it

    for rel, name in start or []:
        if rel in modules:
            reach(modules[rel], name)

    while work:
        m, nodes = work.pop()
        named: set[int] = set()  # the ``globals()`` calls read by a fixed name (see :func:`_scope_read`)
        for top in nodes:
            for node in ast.walk(top):
                key = _scope_read(node)
                if key is not None:
                    named.add(id(key[0]))
                    reach(m, key[1])  # the name read may be a function that is then called
                    continue
                if isinstance(node, ast.Call) and id(node) in named:
                    continue
                if isinstance(node, ast.Call):
                    f = node.func
                    called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                    if called in _DYNAMIC_CALLS:
                        return None
                    if called == "getattr":
                        name = node.args[1] if len(node.args) > 1 else None
                        if isinstance(name, ast.Constant) and isinstance(name.value, str):
                            everywhere(name.value)
                            continue
                        target = imported_module(m, node.args[0]) if node.args else None
                        if target is not None:
                            all_of(target)  # a name computed on a module: any function of that module
                        elif m.rel in fis or _fis_object(m, node.args[0] if node.args else None):
                            for rel in fis:  # FI's own text: only FI's own files hold what it can name
                                all_of(modules[rel])
                        else:
                            return None
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    reach(m, node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    owner = node.value
                    if isinstance(owner, ast.Name) and owner.id in m.imports and owner.id not in m.functions:
                        module, name = m.imports[owner.id]
                        target = by_name.get(module if name is None else f"{module}.{name}" if module else name)
                        if target is not None and reach(target, node.attr):
                            continue
                    # Any other object, or a module that does not hold the name itself: any function of that name.
                    everywhere(node.attr)
    return done


def run_entries(quest_root: Path) -> tuple[list[str], str | None]:
    """The programs FI runs and the file whose functions its trial harness calls (``None`` when the trials were not run
    by the harness: no ledger), from the quest's files. ``experiment.py`` always counts as run (the analysis, or the whole
    study); ``simulate.py`` counts as a program unless the harness ran it."""
    code = Path(quest_root) / "code"
    harness = None
    if (Path(quest_root) / "raw" / "ledger.jsonl").is_file():
        harness = "simulate.py" if (code / "simulate.py").is_file() else "experiment.py"
    scripts = ["experiment.py", *(["simulate.py"] if harness != "simulate.py" else [])]
    return scripts, harness


def never_run_in(quest_root: Path) -> "Callable[[Any, Any], bool]":
    """A test of ``(file, function)`` (relative to the quest's ``code/``): ``True`` only when that function is plainly never
    run (:func:`never_run`). The code is read once, on the first question."""
    code = Path(quest_root) / "code"
    scripts, harness = run_entries(quest_root)
    cache: list[set[tuple[str, str]] | None] = []

    def test(file: Any, function: Any) -> bool:
        file = str(file or "").replace("\\", "/").strip()
        judged = _judged(code / file, str(function or "").strip()) if file else None
        if judged is None:
            return False
        if not cache:
            cache.append(reached(code, scripts=scripts, harness=harness))
        return cache[0] is not None and (file, judged) not in cache[0]

    return test


def _defines(path: Path, function: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function for n in ast.walk(tree))


def _judged(path: Path, function: str) -> str | None:
    """The function of ``path`` whose reach decides the question about ``function``, or ``None`` when the file does not define it.
    A finding may name a nested function (``outer.inner``) or a method (``Class.method``): a function is run when the function it
    sits in is run, so the outermost function the file defines is the one judged (a method has no function around it: its own
    name is judged)."""
    if not function:
        return None
    parts = [p for p in function.split(".") if p]
    if any(_is_dunder(p) for p in parts):
        return None  # a special method is run by the language, not by a name: FI cannot tell
    if _defines(path, function):
        return function
    if len(parts) < 2:
        return None
    for part in parts:
        if _defines(path, part):
            return part
    return None


def never_run(code_dir: Path, file: str, function: str, *, scripts: list[str], harness: str | None = None) -> bool:
    """Whether the function ``function`` of ``file`` (relative to ``code_dir``) is plainly never run: FI can read all the
    code, finds the function in that file, and nothing the run reaches names it. ``False`` whenever FI cannot tell. A nested
    ``outer.inner`` or ``Class.method`` is judged by its outermost function (:func:`_judged`)."""
    file = str(file or "").replace("\\", "/").strip()
    judged = _judged(Path(code_dir) / file, str(function or "").strip()) if file else None
    if judged is None:
        return False
    done = reached(code_dir, scripts=scripts, harness=harness)
    return done is not None and (file, judged) not in done


def _names_in(node: ast.AST | None) -> set[str]:
    """Every name an expression uses: a variable, a function (called or only mentioned) and an attribute."""
    out: set[str] = set()
    for n in ast.walk(node) if node is not None else ():
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def _bases(target: ast.AST) -> set[str]:
    """The variables an assignment target changes: ``a``, ``a, b``, ``a[i]``, ``a.b`` (the variable it changes is ``a``)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {b for e in target.elts for b in _bases(e)}
    if isinstance(target, ast.Starred):
        return _bases(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return _bases(target.value)
    return set()


UNUSED_KEY = "\0unused"  # in result_key_names: the names no returned key depends on (a call there may still change a result)


def _has_exit(node: ast.AST) -> bool:
    """Whether a statement can leave the function or the loop early (return, raise, break, continue)."""
    return any(isinstance(n, (ast.Return, ast.Raise, ast.Break, ast.Continue)) for n in ast.walk(node))


def _tests_in(node: ast.AST) -> set[str]:
    """The names every condition (if, while, for's iterable, assert, with, try handlers) inside a statement names."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            out |= _names_in(n.test)
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out |= _names_in(n.iter)
    return out


def result_key_names(path: Path, function: str) -> dict[str, set[str]] | None:
    """For a function that builds its returned results key by key, the names (variables, functions, attributes) each key's
    value depends on, read from the code alone. ``{key: names}`` plus ``UNUSED_KEY``: the names no key depends on.

    Sound narrowing: the answer is given only when the function is understood completely, else ``None`` (the caller then ties a
    helper to every result of the trial). That needs: exactly one variable holds the results, made once from ``{}``, ``dict()``,
    ``TrialResult()`` or a dict literal with fixed keys; every use of it is ``h["k"] = expr`` (fixed key), ``h.setdefault("k",
    expr)`` or a final ``return h`` (or an earlier ``return h`` / ``raise`` / ``break`` / ``continue`` / ``assert``, whose
    conditions are then counted as a dependency of everything after it); it is never read, aliased, handed to a call, named
    in a nested function, lambda or class; no ``global`` / ``nonlocal``, ``del``, ``**``, ``eval``-like call. A variable handed
    to a call, or a method called on it, is taken to depend on everything the call names.

    This is a dependency of names, not a run: a value changed through a module-level variable another function sets is not
    followed."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None
    fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function), None)
    if fn is None:
        return None
    nodes = list(ast.walk(fn))
    imported = {(a.asname or a.name).split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    if any(isinstance(n, (ast.Global, ast.Nonlocal, ast.Delete)) for n in nodes):
        return None
    if any(isinstance(n, ast.keyword) and n.arg is None for n in nodes):
        return None  # ``f(**x)`` hands over results FI cannot name
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _DYNAMIC_CALLS for n in nodes):
        return None
    # the one variable the results are written into
    holder_inits = [n for n in nodes if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and (isinstance(n.value, ast.Dict)
                         or (isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                             and n.value.func.id in ("dict", "TrialResult") and not n.value.args and not n.value.keywords))]
    holders = {n.targets[0].id for n in holder_inits}
    returned_dict = [n for n in nodes if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    if len(holders) > 1 or (holders and returned_dict):
        return None
    holder = next(iter(holders), "")
    if holder and sum(1 for n in nodes if isinstance(n, ast.Name) and n.id == holder and isinstance(n.ctx, ast.Store)) != 1:
        return None  # made twice, or taken from elsewhere
    if not holder and not returned_dict:
        return None
    deps: dict[str, set[str]] = {}
    keys: dict[str, set[str]] = {}
    ok = [True]
    loose: set[str] = set()  # names of calls made as statements: what they change is not tied to any key

    def const_key(node: ast.AST | None) -> str | None:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def add(var_names: set[str], names: set[str]) -> None:
        for v in var_names:
            deps.setdefault(v, set()).update(names)

    def uses_holder(node: ast.AST | None) -> bool:
        return bool(holder) and node is not None and any(isinstance(n, ast.Name) and n.id == holder for n in ast.walk(node))

    def dict_literal(d: ast.Dict, guard: set[str]) -> None:
        for k, val in zip(d.keys, d.values):
            key = const_key(k)
            if key is None or uses_holder(val):
                ok[0] = False
            else:
                keys.setdefault(key, set()).update(_names_in(val) | guard)
                effects(val, guard)

    def effects(node: ast.AST | None, guard: set[str]) -> None:
        """A call may change what it is handed or what it is called on: those variables then depend on everything it names."""
        for c in ast.walk(node) if node is not None else ():
            if isinstance(c, ast.Call):
                names = _names_in(c) | guard
                touched = {n.id for a in [*c.args, *(k.value for k in c.keywords)] for n in ast.walk(a) if isinstance(n, ast.Name)}
                if isinstance(c.func, ast.Attribute):
                    touched |= {n.id for n in ast.walk(c.func.value) if isinstance(n, ast.Name)}
                add(touched - imported, names)  # a module is not a value a call changes (its state is not followed)
            elif isinstance(c, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ok[0] = ok[0] and not uses_holder(c)

    def visit(stmts: list[ast.stmt], guard: set[str], top: bool) -> set[str]:
        """Walk a block; returns the names of the conditions an early exit inside it puts on what follows."""
        extra: set[str] = set()
        for idx, s in enumerate(stmts):
            g = guard | extra
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if uses_holder(s):
                    ok[0] = False
                add({s.name}, _names_in(s))
                continue
            if isinstance(s, ast.Return):
                v = s.value
                if isinstance(v, ast.Dict):
                    dict_literal(v, g)
                elif not (v is None or isinstance(v, ast.Constant) or (isinstance(v, ast.Name) and v.id == holder and holder)):
                    ok[0] = False
                continue
            if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                value = getattr(s, "value", None)
                if any(s is n for n in holder_inits):
                    if isinstance(value, ast.Dict):
                        dict_literal(value, g)
                    continue
                if uses_holder(value):
                    ok[0] = False  # the results are read or aliased
                names = _names_in(value) | g
                if isinstance(s, ast.AugAssign):
                    names |= _names_in(s.target)
                for t in targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and holder and t.value.id == holder:
                        key = const_key(t.slice)
                        if key is None or isinstance(s, ast.AugAssign):
                            ok[0] = False
                        else:
                            keys.setdefault(key, set()).update(names)
                        continue
                    if uses_holder(t):
                        ok[0] = False
                    add(_bases(t), names | _names_in(t))
                effects(value, g)
            elif isinstance(s, (ast.If, ast.While)):
                inner = g | _names_in(s.test) | (_tests_in(s) if isinstance(s, ast.While) and _has_exit(s) else set())
                effects(s.test, g)
                a = visit(s.body, inner, False)
                b = visit(s.orelse, inner, False)
                extra |= a | b
            elif isinstance(s, (ast.For, ast.AsyncFor)):
                inner = g | _names_in(s.iter) | (_tests_in(s) if _has_exit(s) else set())
                if uses_holder(s.iter) or uses_holder(s.target):
                    ok[0] = False
                add(_bases(s.target), inner)
                effects(s.iter, g)
                extra |= visit(s.body, inner, False) | visit(s.orelse, inner, False)
            elif isinstance(s, (ast.With, ast.AsyncWith)):
                for item in s.items:
                    if uses_holder(item.context_expr) or uses_holder(item.optional_vars):
                        ok[0] = False
                    if item.optional_vars is not None:
                        add(_bases(item.optional_vars), _names_in(item.context_expr) | g)
                    effects(item.context_expr, g)
                extra |= visit(s.body, g, False)
            elif isinstance(s, ast.Try):
                for blk in (s.body, *(h.body for h in s.handlers), s.orelse, s.finalbody):
                    extra |= visit(blk, g, False)
                if s.handlers:
                    extra |= _names_in(s)  # a failing call inside may skip what follows
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and holder and isinstance(s.value.func, ast.Attribute) \
                    and isinstance(s.value.func.value, ast.Name) and s.value.func.value.id == holder:
                call = s.value
                key = const_key(call.args[0]) if call.args else None
                if call.func.attr != "setdefault" or len(call.args) != 2 or key is None or uses_holder(call.args[1]):
                    ok[0] = False
                else:
                    keys.setdefault(key, set()).update(_names_in(call.args[1]) | g)
                    effects(call.args[1], g)
            elif isinstance(s, ast.Assert):
                if uses_holder(s):
                    ok[0] = False
                extra |= _names_in(s.test)
                effects(s.test, g)
            elif isinstance(s, (ast.Expr, ast.Pass, ast.Break, ast.Continue, ast.Raise)):
                if uses_holder(s):
                    ok[0] = False
                v = getattr(s, "value", None) or getattr(s, "exc", None)
                effects(v, g)
                loose.update(_names_in(v))
                if isinstance(s, (ast.Raise, ast.Break, ast.Continue)):
                    extra |= g
            else:
                ok[0] = False  # a statement this reading does not know (match, import inside, ...)
            if _has_exit(s) and not isinstance(s, ast.Return):
                extra |= _tests_in(s)  # code after a condition that may leave early depends on that condition
        return extra

    visit(fn.body, set(), True)
    if not ok[0] or not keys:
        return None
    if holder:
        # every mention of the holder must have been one of the allowed uses: count them
        allowed = 1 + sum(1 for n in nodes if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == holder
                          and isinstance(n.ctx, ast.Store)) \
            + sum(1 for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "setdefault"
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == holder) \
            + sum(1 for n in nodes if isinstance(n, ast.Return) and isinstance(n.value, ast.Name) and n.value.id == holder)
        if sum(1 for n in nodes if isinstance(n, ast.Name) and n.id == holder) != allowed:
            return None
    out: dict[str, set[str]] = {}
    covered: set[str] = set()
    for key, names in keys.items():
        seen, todo = set(), list(names)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            todo.extend(deps.get(n, ()))
        out[key] = seen
        covered |= seen
    out[UNUSED_KEY] = {n for n in _names_in(fn) if n not in covered} | loose
    return out
