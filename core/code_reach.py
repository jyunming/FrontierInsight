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
(``getattr`` with a computed name, ``globals()``, ``eval``, ``importlib``) or a function FI cannot find makes every
function reached, so this can only ever spare a finding about code that is plainly never run.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

#: The simulation's functions FI's trial harness calls by name (core/trial_runner.py).
HARNESS_ENTRIES = ("run_trial", "run_cell", "oracle")
#: Calls that reach code by a name computed at run time.
_DYNAMIC_CALLS = {"globals", "locals", "vars", "eval", "exec", "__import__", "import_module"}


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


class _Module:
    def __init__(self, rel: str, tree: ast.Module) -> None:
        self.rel = rel
        self.tree = tree
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


def _python_files(code_dir: Path) -> list[Path]:
    return sorted(p for p in code_dir.rglob("*.py")
                  if not any(part in ("tests", "__pycache__") or part.startswith(".") for part in p.relative_to(code_dir).parts[:-1])
                  and not p.name.startswith("test_"))


def reached(code_dir: Path, *, scripts: list[str], harness: str | None = None) -> set[tuple[str, str]] | None:
    """The ``(file, function)`` pairs the run reaches (``file`` relative to ``code_dir``, with ``/``), from the programs
    ``scripts`` FI runs and, when FI's trial harness ran the simulation, the harness's entries in the file ``harness``.
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

    done: set[tuple[str, str]] = set()
    work: list[tuple[_Module, list[ast.AST]]] = []

    def reach(m: _Module | None, name: str) -> None:
        if m is None or name not in m.functions or (m.rel, name) in done:
            return
        done.add((m.rel, name))
        work.append((m, m.functions[name]))

    def everywhere(name: str) -> None:
        for m in modules.values():
            reach(m, name)

    for rel, m in modules.items():
        work.append((m, _definition_time(m.tree.body, main=rel in scripts)))
    # FI's harness may call these in either script, whether or not a ledger shows it did: they always count as run.
    for script in {*scripts, *([harness] if harness else [])}:
        if script in modules:
            for entry in HARNESS_ENTRIES:
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry for n in modules[script].tree.body):
                    reach(modules[script], entry)

    while work:
        m, nodes = work.pop()
        for top in nodes:
            for node in ast.walk(top):
                if isinstance(node, ast.Call):
                    f = node.func
                    called = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                    if called in _DYNAMIC_CALLS:
                        return None
                    if called == "getattr":
                        name = node.args[1] if len(node.args) > 1 else None
                        if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                            return None
                        everywhere(name.value)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in m.functions:
                        reach(m, node.id)
                    elif node.id in m.imports:
                        module, name = m.imports[node.id]
                        if name is not None:
                            reach(by_name.get(module), name)
                    else:
                        for module in m.star:
                            reach(by_name.get(module), node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    owner = node.value
                    if isinstance(owner, ast.Name) and owner.id in m.imports and owner.id not in m.functions:
                        module, name = m.imports[owner.id]
                        target = by_name.get(module if name is None else f"{module}.{name}" if module else name)
                        if target is not None:
                            reach(target, node.attr)
                            continue
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
        function = str(function or "").strip()
        if not file or not function or not _defines(code / file, function):
            return False
        if not cache:
            cache.append(reached(code, scripts=scripts, harness=harness))
        return cache[0] is not None and (file, function) not in cache[0]

    return test


def _defines(path: Path, function: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function for n in ast.walk(tree))


def never_run(code_dir: Path, file: str, function: str, *, scripts: list[str], harness: str | None = None) -> bool:
    """Whether the function ``function`` of ``file`` (relative to ``code_dir``) is plainly never run: FI can read all the
    code, finds the function in that file, and nothing the run reaches names it. ``False`` whenever FI cannot tell."""
    file = str(file or "").replace("\\", "/").strip()
    function = str(function or "").strip()
    if not file or not function or not _defines(Path(code_dir) / file, function):
        return False
    done = reached(code_dir, scripts=scripts, harness=harness)
    return done is not None and (file, function) not in done
